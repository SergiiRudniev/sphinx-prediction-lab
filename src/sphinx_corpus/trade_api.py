from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from sphinx_corpus import SCHEMA_VERSION
from sphinx_corpus.config import CorpusConfig, parse_utc, utc_text
from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    iter_jsonl_zst,
    load_json,
    now_utc,
    read_json_zst,
    write_json_zst,
    write_jsonl_zst,
)

_CONDITION_ID = re.compile(r"^0x[a-fA-F0-9]{64}$")


class RequestBudgetReached(RuntimeError):
    pass


def _optional_time(value: Any) -> datetime | None:
    if not value:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return parse_utc(str(value))
    return None


def _number(value: Any) -> Decimal:
    with contextlib.suppress(InvalidOperation, TypeError, ValueError):
        return Decimal(str(value or 0))
    return Decimal(0)


def _trade_key(trade: dict[str, Any]) -> str:
    fields = {
        "asset": trade.get("asset"),
        "conditionId": trade.get("conditionId"),
        "outcome": trade.get("outcome"),
        "price": trade.get("price"),
        "proxyWallet": trade.get("proxyWallet"),
        "side": trade.get("side"),
        "size": trade.get("size"),
        "timestamp": trade.get("timestamp"),
        "transactionHash": trade.get("transactionHash"),
    }
    return json.dumps(fields, separators=(",", ":"), sort_keys=True)


class TradeAPIBackfill:
    def __init__(self, config: CorpusConfig, *, max_requests: int | None = None) -> None:
        self.config = config
        source = config.payload["sources"]["ledger"]["primary"]
        self.base_url = str(source["base_url"]).rstrip("/")
        self.endpoint = str(source["endpoint"])
        self.page_size = int(source["page_size"])
        self.maximum_offset = int(source["maximum_offset"])
        self.markets_per_request = int(source["markets_per_request"])
        self.taker_only = bool(source["taker_only"])
        self.minimum_window_seconds = int(source["minimum_window_seconds"])
        self.selection = source["selection"]
        self.max_requests = max_requests
        self.requests = 0
        self.client = httpx.Client(
            timeout=httpx.Timeout(90.0, connect=15.0),
            headers={"User-Agent": "sphinx-prediction-lab/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> TradeAPIBackfill:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _fetch(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise RequestBudgetReached
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = self.client.get(f"{self.base_url}{self.endpoint}", params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"Data API HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or not all(
                    isinstance(item, dict) for item in payload
                ):
                    raise TypeError("Data API trades response must be a list of objects")
                self.requests += 1
                return payload
            except (httpx.HTTPError, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(16.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Data API request failed: {last_error}")

    def _markets(self, explicit: set[str] | None, max_markets: int | None) -> list[dict[str, Any]]:
        atlas_path = self.config.data_dir / "normalized" / "atlas" / "markets.jsonl.zst"
        if not atlas_path.exists():
            raise RuntimeError("Sphinx Atlas markets must be collected before Sphinx Ledger")
        minimum_volume = _number(self.selection["minimum_market_volume_usd"])
        markets: list[dict[str, Any]] = []
        for row in iter_jsonl_zst(atlas_path):
            condition_id = str(row.get("condition_id") or "")
            if not _CONDITION_ID.fullmatch(condition_id):
                continue
            if explicit is not None and condition_id.lower() not in explicit:
                continue
            payload = row.get("source_payload")
            source_payload = payload if isinstance(payload, dict) else {}
            volume = _number(source_payload.get("volumeNum") or source_payload.get("volume"))
            if volume < minimum_volume:
                continue
            markets.append({**row, "_volume": volume})
        markets.sort(
            key=lambda row: (_number(row["_volume"]), str(row["condition_id"])),
            reverse=True,
        )
        configured_max = self.selection.get("maximum_markets")
        cap = max_markets if max_markets is not None else configured_max
        if cap is not None:
            markets = markets[: int(cap)]
        if explicit is not None:
            found = {str(row["condition_id"]).lower() for row in markets}
            missing = explicit - found
            if missing:
                values = ", ".join(sorted(missing))
                raise ValueError(f"Requested markets are absent from Atlas: {values}")
        return markets

    def collect(
        self,
        *,
        market_ids: set[str] | None = None,
        max_markets: int | None = None,
    ) -> dict[str, Any]:
        explicit = {value.lower() for value in market_ids} if market_ids else None
        markets = self._markets(explicit, max_markets)
        completed = 0
        skipped = 0
        rows = 0
        gaps = 0
        budget_reached = False
        groups = [
            markets[offset : offset + self.markets_per_request]
            for offset in range(0, len(markets), self.markets_per_request)
        ]
        for group in groups:
            condition_ids = tuple(str(row["condition_id"]).lower() for row in group)
            scope_id = self._scope_id(condition_ids)
            market_receipt_path = (
                self.config.data_dir / "receipts" / "ledger-api" / f"{scope_id}.json"
            )
            existing = load_json(market_receipt_path)
            if existing.get("complete") is True:
                skipped += 1
                rows += int(existing.get("rows", 0))
                gaps += int(existing.get("gaps", 0))
                continue
            bounds = [value for row in group if (value := self._market_bounds(row))]
            if not bounds:
                atomic_json(
                    market_receipt_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scope_id": scope_id,
                        "condition_ids": condition_ids,
                        "complete": True,
                        "rows": 0,
                        "gaps": 0,
                        "reason": "empty_market_window",
                        "updated_at": now_utc(),
                    },
                )
                completed += 1
                continue
            start = min(value[0] for value in bounds)
            end = max(value[1] for value in bounds)
            try:
                result = self._window(scope_id, condition_ids, start, end)
            except RequestBudgetReached:
                budget_reached = True
                break
            market_receipt = {
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_id,
                "condition_ids": condition_ids,
                "window_start": start,
                "window_end_inclusive": end,
                "complete": result["complete"],
                "rows": result["rows"],
                "gaps": result["gaps"],
                "updated_at": now_utc(),
            }
            atomic_json(market_receipt_path, market_receipt)
            completed += int(bool(result["complete"]))
            rows += int(result["rows"])
            gaps += int(result["gaps"])
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_utc(),
            "markets_selected": len(markets),
            "groups_selected": len(groups),
            "groups_completed_this_run": completed,
            "groups_skipped": skipped,
            "rows": rows,
            "gaps": gaps,
            "requests_this_run": self.requests,
            "request_budget_reached": budget_reached,
            "complete": not budget_reached and completed + skipped == len(groups),
        }
        atomic_json(self.config.data_dir / "receipts" / "ledger-api.json", receipt)
        return receipt

    @staticmethod
    def _scope_id(condition_ids: tuple[str, ...]) -> str:
        if len(condition_ids) == 1:
            return condition_ids[0]
        digest = hashlib.sha256(",".join(condition_ids).encode()).hexdigest()[:24]
        return f"group-{digest}"

    def _market_bounds(self, market: dict[str, Any]) -> tuple[int, int] | None:
        start_time = max(
            self.config.window.start,
            _optional_time(market.get("created_at"))
            or _optional_time(market.get("start_at"))
            or self.config.window.start,
        )
        end_time = min(
            self.config.window.end,
            _optional_time(market.get("closed_at"))
            or _optional_time(market.get("end_at"))
            or self.config.window.end,
        )
        start = int(start_time.timestamp())
        end = int(end_time.timestamp()) - 1
        return (start, end) if start <= end else None

    def _paths(self, scope_id: str, start: int, end: int, offset: int) -> tuple[Path, Path]:
        directory = f"scope={scope_id}"
        stem = f"start={start}-end={end}-offset={offset:05d}"
        raw = self.config.data_dir / "raw" / "ledger-api" / directory / f"{stem}.json.zst"
        receipt = (
            self.config.data_dir / "receipts" / "ledger-api-windows" / directory / f"{stem}.json"
        )
        return raw, receipt

    def _page(
        self,
        scope_id: str,
        condition_ids: tuple[str, ...],
        start: int,
        end: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        raw_path, page_receipt_path = self._paths(scope_id, start, end, offset)
        if raw_path.exists():
            wrapper = read_json_zst(raw_path)
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("response"), list):
                raise TypeError(f"Invalid Ledger API raw page: {raw_path}")
            return [item for item in wrapper["response"] if isinstance(item, dict)]
        check_disk_reserve(
            self.config.data_dir,
            float(self.config.payload["storage"]["minimum_free_gib"]),
        )
        params = {
            "market": ",".join(condition_ids),
            "limit": self.page_size,
            "offset": offset,
            "takerOnly": str(self.taker_only).lower(),
            "start": start,
            "end": end,
        }
        observed_at = now_utc()
        payload = self._fetch(params)
        write_json_zst(
            raw_path,
            {
                "schema_version": SCHEMA_VERSION,
                "source": "polymarket_data_api_trades",
                "observed_at": observed_at,
                "request": params,
                "response": payload,
            },
        )
        atomic_json(
            page_receipt_path,
            {
                "schema_version": SCHEMA_VERSION,
                "observed_at": observed_at,
                "rows": len(payload),
            },
        )
        return payload

    def _window_receipt_path(self, scope_id: str, start: int, end: int) -> Path:
        return (
            self.config.data_dir
            / "receipts"
            / "ledger-api-windows"
            / f"scope={scope_id}"
            / f"start={start}-end={end}.json"
        )

    def _normalized_path(self, scope_id: str, start: int, end: int) -> Path:
        return (
            self.config.data_dir
            / "normalized"
            / "ledger-api"
            / f"scope={scope_id}"
            / f"start={start}-end={end}.jsonl.zst"
        )

    def _window(
        self,
        scope_id: str,
        condition_ids: tuple[str, ...],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        receipt_path = self._window_receipt_path(scope_id, start, end)
        receipt = load_json(receipt_path)
        status = receipt.get("status")
        if status == "leaf_complete" and self._normalized_path(scope_id, start, end).exists():
            return receipt
        if status in {"split", "split_complete"}:
            split = int(receipt["split"])
            left = self._window(scope_id, condition_ids, start, split)
            right = self._window(scope_id, condition_ids, split + 1, end)
            combined = {
                **receipt,
                "status": "split_complete",
                "complete": bool(left["complete"] and right["complete"]),
                "rows": int(left["rows"]) + int(right["rows"]),
                "gaps": int(left["gaps"]) + int(right["gaps"]),
                "updated_at": now_utc(),
            }
            atomic_json(receipt_path, combined)
            if combined["complete"]:
                for offset in (0, self.maximum_offset):
                    raw_path, _ = self._paths(scope_id, start, end, offset)
                    with contextlib.suppress(FileNotFoundError):
                        raw_path.unlink()
            return combined

        first = self._page(scope_id, condition_ids, start, end, 0)
        pages = [first]
        saturated = False
        if len(first) == self.page_size:
            tail = self._page(
                scope_id,
                condition_ids,
                start,
                end,
                self.maximum_offset,
            )
            saturated = len(tail) == self.page_size
            if not saturated:
                for offset in range(self.page_size, self.maximum_offset + 1, self.page_size):
                    page = tail if offset == self.maximum_offset else self._page(
                        scope_id,
                        condition_ids,
                        start,
                        end,
                        offset,
                    )
                    pages.append(page)
                    if len(page) < self.page_size:
                        break
        if saturated:
            if end - start <= self.minimum_window_seconds:
                gap = {
                    "schema_version": SCHEMA_VERSION,
                    "scope_id": scope_id,
                    "condition_ids": condition_ids,
                    "start": start,
                    "end": end,
                    "status": "unresolved_saturation",
                    "complete": False,
                    "rows": 0,
                    "gaps": 1,
                    "updated_at": now_utc(),
                }
                atomic_json(receipt_path, gap)
                return gap
            split = start + (end - start) // 2
            split_receipt = {
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_id,
                "condition_ids": condition_ids,
                "start": start,
                "end": end,
                "status": "split",
                "split": split,
                "complete": False,
                "rows": 0,
                "gaps": 0,
                "updated_at": now_utc(),
            }
            atomic_json(receipt_path, split_receipt)
            return self._window(scope_id, condition_ids, start, end)

        trades = [trade for page in pages for trade in page]
        normalized = self._normalize(condition_ids, start, end, trades)
        normalized_path = self._normalized_path(scope_id, start, end)
        write_jsonl_zst(normalized_path, normalized)
        complete = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": scope_id,
            "condition_ids": condition_ids,
            "start": start,
            "end": end,
            "status": "leaf_complete",
            "complete": True,
            "rows": len(normalized),
            "gaps": 0,
            "updated_at": now_utc(),
        }
        atomic_json(receipt_path, complete)
        return complete

    def _normalize(
        self,
        condition_ids: tuple[str, ...],
        start: int,
        end: int,
        trades: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected_ids = set(condition_ids)
        selected = [
            trade
            for trade in trades
            if start <= int(trade.get("timestamp", -1)) <= end
            and str(trade.get("conditionId", "")).lower() in selected_ids
        ]
        selected.sort(
            key=lambda trade: (
                int(trade.get("timestamp", 0)),
                str(trade.get("transactionHash", "")),
                str(trade.get("proxyWallet", "")),
                _trade_key(trade),
            )
        )
        occurrences: Counter[str] = Counter()
        output: list[dict[str, Any]] = []
        for trade in selected:
            key = _trade_key(trade)
            ordinal = occurrences[key]
            occurrences[key] += 1
            digest = hashlib.sha256(f"{key}#{ordinal}".encode()).hexdigest()
            timestamp = int(trade["timestamp"])
            size = _number(trade.get("size"))
            price = _number(trade.get("price"))
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "public_trade",
                    "source": "polymarket_data_api_trades",
                    "trade_id": digest,
                    "condition_id": str(trade.get("conditionId") or "").lower(),
                    "token_id": str(trade.get("asset") or ""),
                    "wallet": str(trade.get("proxyWallet") or "").lower(),
                    "side": str(trade.get("side") or ""),
                    "size": format(size, "f"),
                    "price": format(price, "f"),
                    "notional_usd": format(size * price, "f"),
                    "timestamp": utc_text(datetime.fromtimestamp(timestamp, tz=UTC)),
                    "timestamp_unix": timestamp,
                    "transaction_hash": str(trade.get("transactionHash") or "").lower(),
                    "outcome": trade.get("outcome"),
                    "outcome_index": trade.get("outcomeIndex"),
                }
            )
        return output
