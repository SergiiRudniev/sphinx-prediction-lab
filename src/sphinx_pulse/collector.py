from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import signal
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets
import yaml

from sphinx_pulse import SCHEMA_VERSION
from sphinx_pulse.storage import HourlyZstdStore, atomic_json


@dataclass(frozen=True)
class PulseConfig:
    gamma_url: str
    websocket_url: str
    output_dir: Path
    discovery_interval_seconds: int
    catalog_snapshot_seconds: int
    status_interval_seconds: int
    heartbeat_seconds: int
    subscription_batch_size: int
    queue_size: int
    writer_batch_size: int
    writer_flush_seconds: float
    max_part_bytes: int
    min_free_disk_gb: float
    max_markets: int

    @classmethod
    def load(cls, path: Path) -> PulseConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pulse config must be a mapping")
        websocket = payload["websocket"]
        storage = payload["storage"]
        selection = payload["selection"]
        return cls(
            gamma_url=str(payload["gamma_url"]).rstrip("/"),
            websocket_url=str(payload["websocket_url"]),
            output_dir=Path(payload["output_dir"]),
            discovery_interval_seconds=int(payload["discovery_interval_seconds"]),
            catalog_snapshot_seconds=int(payload["catalog_snapshot_seconds"]),
            status_interval_seconds=int(payload["status_interval_seconds"]),
            heartbeat_seconds=int(websocket["heartbeat_seconds"]),
            subscription_batch_size=int(websocket["subscription_batch_size"]),
            queue_size=int(websocket["queue_size"]),
            writer_batch_size=int(websocket["writer_batch_size"]),
            writer_flush_seconds=float(websocket["writer_flush_seconds"]),
            max_part_bytes=int(storage["max_part_bytes"]),
            min_free_disk_gb=float(storage["min_free_disk_gb"]),
            max_markets=int(selection["max_markets"]),
        )


def _chunks(values: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def select_markets(
    events: list[dict[str, Any]],
    max_markets: int,
) -> list[dict[str, Any]]:
    candidates: dict[str, tuple[tuple[float, float, float], dict[str, Any]]] = {}
    for event in events:
        markets = event.get("markets", [])
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("closed") or not market.get("active", True):
                continue
            if not market.get("enableOrderBook") or not market.get("acceptingOrders"):
                continue
            values = market.get("clobTokenIds", [])
            if isinstance(values, str):
                with contextlib.suppress(json.JSONDecodeError):
                    values = json.loads(values)
            if not isinstance(values, list):
                continue
            token_ids = [str(value) for value in values if str(value)]
            if not token_ids:
                continue
            identity = str(market.get("conditionId") or market.get("id") or token_ids[0])
            score = (
                _number(market.get("volume24hr")),
                _number(market.get("liquidityNum") or market.get("liquidity")),
                _number(market.get("volumeNum") or market.get("volume")),
            )
            record = {
                "event": {
                    "id": event.get("id"),
                    "slug": event.get("slug"),
                    "title": event.get("title"),
                    "tags": event.get("tags", []),
                },
                "market": market,
                "token_ids": token_ids,
            }
            candidates[identity] = (score, record)
    ranked = sorted(candidates.values(), key=lambda value: value[0], reverse=True)
    return [record for _, record in ranked[: max(1, int(max_markets))]]


def extract_token_ids(events: list[dict[str, Any]], max_markets: int | None = None) -> set[str]:
    selection = select_markets(events, max_markets or 2**31)
    return {token for record in selection for token in record["token_ids"]}


class PulseCollector:
    def __init__(self, config: PulseConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.raw_dir = self.output_dir / "raw"
        self.status_path = self.output_dir / "status" / "collector.json"
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(config.queue_size)
        self.stop = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.subscribed: set[str] = set()
        self.last_catalog_snapshot = 0.0
        self.stats: dict[str, Any] = {
            "connected": False,
            "messages": 0,
            "records": 0,
            "bytes_written": 0,
            "reconnects": 0,
            "last_message_ms": None,
            "last_error": None,
        }
        self.store = HourlyZstdStore(
            self.raw_dir,
            max_part_bytes=config.max_part_bytes,
        )
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0))

    async def fetch_catalog(self) -> list[dict[str, Any]]:
        page_size = 500
        cursor: str | None = None
        seen_cursors: set[str] = set()
        events: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {
                "closed": "false",
                "limit": page_size,
            }
            if cursor is not None:
                params["after_cursor"] = cursor
            response = await self.client.get(
                f"{self.config.gamma_url}/events/keyset",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Gamma keyset response is not an object")
            page = payload.get("events")
            if not isinstance(page, list):
                raise RuntimeError("Gamma keyset events field is not a list")
            valid = [event for event in page if isinstance(event, dict)]
            events.extend(valid)
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                return events
            cursor = str(next_cursor)
            if cursor in seen_cursors:
                raise RuntimeError("Gamma keyset cursor repeated")
            seen_cursors.add(cursor)

    async def enqueue_catalog(
        self,
        markets: list[dict[str, Any]],
        observed_at_ms: int,
    ) -> None:
        for market in markets:
            await self.queue.put(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "market_catalog",
                    "source": "polymarket_gamma_events",
                    "received_at_ms": observed_at_ms,
                    "payload": market,
                }
            )

    async def _send_subscription(self, websocket: Any, operation: str, tokens: set[str]) -> None:
        for batch in _chunks(sorted(tokens), self.config.subscription_batch_size):
            payload: dict[str, Any] = {
                "assets_ids": batch,
                "custom_feature_enabled": True,
            }
            if operation == "initial":
                payload["type"] = "market"
            else:
                payload["operation"] = operation
            async with self.send_lock:
                await websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _heartbeat(self, websocket: Any) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(self.config.heartbeat_seconds)
            async with self.send_lock:
                await websocket.send("PING")

    async def _refresh_subscriptions(self, websocket: Any) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(self.config.discovery_interval_seconds)
            try:
                events = await self.fetch_catalog()
                markets = select_markets(events, self.config.max_markets)
                tokens = {token for market in markets for token in market["token_ids"]}
                additions = tokens - self.subscribed
                removals = self.subscribed - tokens
                if additions:
                    await self._send_subscription(websocket, "subscribe", additions)
                if removals:
                    await self._send_subscription(websocket, "unsubscribe", removals)
                self.subscribed = tokens
                self.stats["selected_markets"] = len(markets)
                now = time.monotonic()
                if now - self.last_catalog_snapshot >= self.config.catalog_snapshot_seconds:
                    await self.enqueue_catalog(markets, int(time.time() * 1000))
                    self.last_catalog_snapshot = now
            except Exception as exc:
                self.stats["last_error"] = f"catalog refresh: {exc!r}"

    async def _consume(self, websocket: Any) -> None:
        async for message in websocket:
            if message == "PONG":
                continue
            received_at_ms = int(time.time() * 1000)
            payload = json.loads(message)
            items = payload if isinstance(payload, list) else [payload]
            self.stats["messages"] += 1
            self.stats["last_message_ms"] = received_at_ms
            for index, item in enumerate(items):
                await self.queue.put(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "market_event",
                        "source": "polymarket_market_websocket",
                        "received_at_ms": received_at_ms,
                        "message_item_index": index,
                        "payload": item,
                    }
                )

    async def _connection_once(self) -> None:
        events = await self.fetch_catalog()
        markets = select_markets(events, self.config.max_markets)
        tokens = {token for market in markets for token in market["token_ids"]}
        if not tokens:
            raise RuntimeError("No active CLOB token IDs discovered")
        await self.enqueue_catalog(markets, int(time.time() * 1000))
        self.last_catalog_snapshot = time.monotonic()
        first, *remaining = list(_chunks(sorted(tokens), self.config.subscription_batch_size))
        async with websockets.connect(
            self.config.websocket_url,
            ping_interval=None,
            open_timeout=30,
            close_timeout=10,
            max_size=None,
            max_queue=2048,
        ) as websocket:
            await self._send_subscription(websocket, "initial", set(first))
            for batch in remaining:
                await self._send_subscription(websocket, "subscribe", set(batch))
            self.subscribed = tokens
            self.stats["selected_markets"] = len(markets)
            self.stats["connected"] = True
            self.stats["last_error"] = None
            heartbeat = asyncio.create_task(self._heartbeat(websocket))
            refresh = asyncio.create_task(self._refresh_subscriptions(websocket))
            try:
                await self._consume(websocket)
            finally:
                heartbeat.cancel()
                refresh.cancel()
                await asyncio.gather(heartbeat, refresh, return_exceptions=True)
                self.stats["connected"] = False
                self.subscribed.clear()

    async def _connection_loop(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            try:
                await self._connection_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["connected"] = False
                self.stats["last_error"] = repr(exc)
                self.stats["reconnects"] += 1
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                backoff = min(60.0, backoff * 2.0)

    def _check_disk(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.output_dir).free
        minimum = int(self.config.min_free_disk_gb * 1024**3)
        if free < minimum:
            raise RuntimeError("Pulse disk reserve reached")

    async def _writer(self) -> None:
        while not self.stop.is_set() or not self.queue.empty():
            records: list[dict[str, Any]] = []
            try:
                records.append(
                    await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.config.writer_flush_seconds,
                    )
                )
            except TimeoutError:
                continue
            deadline = time.monotonic() + self.config.writer_flush_seconds
            while len(records) < self.config.writer_batch_size and time.monotonic() < deadline:
                try:
                    records.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._check_disk()
            count, written = await asyncio.to_thread(self.store.append, records)
            self.stats["records"] += count
            self.stats["bytes_written"] += written

    def _status(self, state: str) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.output_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "heartbeat_ms": int(time.time() * 1000),
            "status": state,
            "queue_depth": self.queue.qsize(),
            "subscribed_assets": len(self.subscribed),
            "disk_free_bytes": usage.free,
            **self.stats,
        }

    async def _status_loop(self) -> None:
        while not self.stop.is_set():
            atomic_json(self.status_path, self._status("collecting"))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self.stop.wait(),
                    timeout=self.config.status_interval_seconds,
                )

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = asyncio.create_task(self._writer(), name="pulse-writer")
        status = asyncio.create_task(self._status_loop(), name="pulse-status")
        connection = asyncio.create_task(self._connection_loop(), name="pulse-websocket")
        stop_wait = asyncio.create_task(self.stop.wait(), name="pulse-stop")
        try:
            done, _ = await asyncio.wait(
                {writer, status, connection, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_wait not in done:
                failed = next(task for task in done if task is not stop_wait)
                exception = failed.exception()
                if exception is not None:
                    raise exception
                raise RuntimeError(f"Pulse task stopped unexpectedly: {failed.get_name()}")
        finally:
            self.stop.set()
            connection.cancel()
            status.cancel()
            stop_wait.cancel()
            await asyncio.gather(connection, status, stop_wait, return_exceptions=True)
            await asyncio.gather(writer, return_exceptions=True)
            await self.client.aclose()
            atomic_json(self.status_path, self._status("stopped"))


async def _main(config_path: Path) -> None:
    collector = PulseCollector(PulseConfig.load(config_path))
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(event, collector.stop.set)
    await collector.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_main(args.config))


if __name__ == "__main__":
    main()
