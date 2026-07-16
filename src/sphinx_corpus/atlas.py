from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime
from typing import Any

import httpx

from sphinx_corpus import SCHEMA_VERSION
from sphinx_corpus.config import CorpusConfig, parse_utc, utc_text
from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    load_json,
    now_utc,
    read_json_zst,
    write_json_zst,
    write_jsonl_zst,
)


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
    return []


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return parse_utc(str(value))
    return None


def market_intersects(market: dict[str, Any], config: CorpusConfig) -> bool:
    created = _timestamp(market.get("createdAt")) or _timestamp(market.get("startDate"))
    ended = (
        _timestamp(market.get("closedTime"))
        or _timestamp(market.get("umaEndDate"))
        or _timestamp(market.get("endDate"))
    )
    if created is not None and created >= config.window.end:
        return False
    return ended is None or ended >= config.window.start


class AtlasBackfill:
    def __init__(self, config: CorpusConfig) -> None:
        self.config = config
        source = config.payload["sources"]["atlas"]
        self.base_url = str(source["base_url"]).rstrip("/")
        self.endpoint = str(source["endpoint"])
        self.page_size = int(source["page_size"])
        self.root = config.data_dir
        self.client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": "sphinx-prediction-lab/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> AtlasBackfill:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.client.get(f"{self.base_url}{self.endpoint}", params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"Gamma HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("Gamma keyset response must be an object")
                if not isinstance(payload.get("markets"), list):
                    raise TypeError("Gamma keyset response has no markets list")
                return payload
            except (httpx.HTTPError, ValueError, RuntimeError, TypeError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Gamma request failed: {last_error}")

    def collect(self, *, max_pages: int | None = None) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for closed in self.config.payload["sources"]["atlas"]["closed_partitions"]:
            summaries[str(closed).lower()] = self._collect_partition(
                bool(closed), max_pages=max_pages
            )
        normalized = self.normalize()
        return {"partitions": summaries, "normalized": normalized}

    def _collect_partition(self, closed: bool, *, max_pages: int | None) -> dict[str, Any]:
        name = "true-windowed" if closed else "false"
        state_path = self.root / "state" / f"atlas-closed-{name}.json"
        state = load_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "closed": closed,
                "next_cursor": None,
                "next_page": 0,
                "rows": 0,
                "complete": False,
            },
        )
        if bool(state.get("complete")):
            return state
        pages_this_run = 0
        seen_cursors: set[str] = set()
        while max_pages is None or pages_this_run < max_pages:
            check_disk_reserve(
                self.root,
                float(self.config.payload["storage"]["minimum_free_gib"]),
            )
            cursor = state.get("next_cursor")
            params: dict[str, Any] = {
                "closed": str(closed).lower(),
                "limit": self.page_size,
            }
            if closed:
                params["end_date_min"] = utc_text(self.config.window.start)
            if cursor:
                params["after_cursor"] = str(cursor)
            observed_at = now_utc()
            payload = self._fetch(params)
            markets = payload["markets"]
            page = int(state["next_page"])
            page_path = (
                self.root
                / "raw"
                / "atlas"
                / f"closed={name}"
                / f"page-{page:06d}.json.zst"
            )
            write_json_zst(
                page_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "source": "polymarket_gamma_markets_keyset",
                    "observed_at": observed_at,
                    "request": params,
                    "response": payload,
                },
            )
            next_cursor = payload.get("next_cursor")
            complete = not next_cursor or not markets
            if next_cursor:
                text = str(next_cursor)
                if text in seen_cursors or text == str(cursor):
                    raise RuntimeError("Gamma keyset cursor repeated")
                seen_cursors.add(text)
            state = {
                **state,
                "next_cursor": None if complete else str(next_cursor),
                "next_page": page + 1,
                "rows": int(state.get("rows", 0)) + len(markets),
                "complete": complete,
                "updated_at": now_utc(),
            }
            atomic_json(state_path, state)
            pages_this_run += 1
            if complete:
                break
        return state

    def normalize(self) -> dict[str, int]:
        page_paths = list(
            (self.root / "raw" / "atlas" / "closed=false").glob("page-*.json.zst")
        ) + list(
            (self.root / "raw" / "atlas" / "closed=true-windowed").glob(
                "page-*.json.zst"
            )
        )
        page_paths.sort()

        def source_markets() -> Any:
            seen: set[str] = set()
            for page_path in page_paths:
                page = read_json_zst(page_path)
                if not isinstance(page, dict) or not isinstance(page.get("response"), dict):
                    raise TypeError(f"Invalid Atlas raw page: {page_path}")
                observed_at = str(page["observed_at"])
                markets = page["response"].get("markets", [])
                if not isinstance(markets, list):
                    raise TypeError(f"Invalid markets list: {page_path}")
                for market in markets:
                    if not isinstance(market, dict) or not market_intersects(
                        market, self.config
                    ):
                        continue
                    identity = str(market.get("conditionId") or market.get("id") or "")
                    if not identity or identity in seen:
                        continue
                    seen.add(identity)
                    yield identity, market, observed_at

        def market_rows() -> Any:
            for identity, market, observed_at in source_markets():
                events = [
                    item
                    for item in _list_value(market.get("events"))
                    if isinstance(item, dict)
                ]
                event_ids = [
                    str(event.get("id") or event.get("slug"))
                    for event in events
                    if event.get("id") or event.get("slug")
                ]
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "market",
                    "market_id": str(market.get("id") or identity),
                    "condition_id": market.get("conditionId"),
                    "question_id": market.get("questionID") or market.get("questionId"),
                    "event_ids": event_ids,
                    "created_at": market.get("createdAt"),
                    "start_at": market.get("startDate"),
                    "end_at": market.get("endDate"),
                    "closed_at": market.get("closedTime"),
                    "active": market.get("active"),
                    "closed": market.get("closed"),
                    "accepting_orders": market.get("acceptingOrders"),
                    "neg_risk": market.get("negRisk"),
                    "resolution_status": market.get("umaResolutionStatus"),
                    "observed_at": observed_at,
                    "source_payload": market,
                }

        def event_rows() -> Any:
            seen: set[str] = set()
            for _, market, observed_at in source_markets():
                for event in _list_value(market.get("events")):
                    if not isinstance(event, dict):
                        continue
                    event_id = str(event.get("id") or event.get("slug") or "")
                    if not event_id or event_id in seen:
                        continue
                    seen.add(event_id)
                    yield {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "event",
                        "event_id": event_id,
                        "observed_at": observed_at,
                        "source_payload": event,
                    }

        def token_rows() -> Any:
            for identity, market, observed_at in source_markets():
                market_id = str(market.get("id") or identity)
                tokens = _list_value(market.get("clobTokenIds"))
                outcomes = _list_value(market.get("outcomes"))
                prices = _list_value(market.get("outcomePrices"))
                for index, token in enumerate(tokens):
                    token_id = str(token)
                    if not token_id:
                        continue
                    yield {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "token",
                        "token_id": token_id,
                        "market_id": market_id,
                        "condition_id": market.get("conditionId"),
                        "outcome_index": index,
                        "outcome": outcomes[index] if index < len(outcomes) else None,
                        "observed_price": prices[index] if index < len(prices) else None,
                        "market_created_at": market.get("createdAt"),
                        "market_start_at": market.get("startDate"),
                        "market_end_at": market.get("endDate"),
                        "market_closed_at": market.get("closedTime"),
                        "observed_at": observed_at,
                    }

        normalized = self.root / "normalized" / "atlas"
        market_count, _ = write_jsonl_zst(normalized / "markets.jsonl.zst", market_rows())
        event_count, _ = write_jsonl_zst(normalized / "events.jsonl.zst", event_rows())
        token_count, _ = write_jsonl_zst(normalized / "tokens.jsonl.zst", token_rows())
        summary = {
            "markets": market_count,
            "events": event_count,
            "tokens": token_count,
        }
        atomic_json(
            self.root / "receipts" / "atlas.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": now_utc(),
                "window_start": utc_text(self.config.window.start),
                "window_end_exclusive": utc_text(self.config.window.end),
                **summary,
            },
        )
        return summary
