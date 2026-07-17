from __future__ import annotations

import contextlib
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from sphinx_corpus import SCHEMA_VERSION
from sphinx_corpus.config import CorpusConfig, parse_utc, utc_text
from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    iter_jsonl_zst,
    now_utc,
    read_json_zst,
    write_json_zst,
    write_jsonl_zst,
)


def _optional_time(value: Any) -> datetime | None:
    if not value:
        return None
    with contextlib.suppress(TypeError, ValueError):
        return parse_utc(str(value))
    return None


class DepthBackfill:
    def __init__(self, config: CorpusConfig) -> None:
        self.config = config
        source = config.payload["sources"]["depth"]
        self.base_url = str(source["base_url"]).rstrip("/")
        self.endpoint = str(source["endpoint"])
        self.fidelity = int(source["fidelity_minutes"])
        self.window_days = int(source["window_days"])
        self.client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": "sphinx-prediction-lab/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> DepthBackfill:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.client.get(f"{self.base_url}{self.endpoint}", params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"CLOB HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
                    raise TypeError("CLOB price history response must contain a history list")
                return payload
            except (httpx.HTTPError, ValueError, RuntimeError, TypeError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"CLOB price history request failed: {last_error}")

    def collect(
        self,
        *,
        max_tokens: int | None = None,
        max_windows: int | None = None,
    ) -> dict[str, Any]:
        token_path = self.config.data_dir / "normalized" / "atlas" / "tokens.jsonl.zst"
        if not token_path.exists():
            raise RuntimeError("Sphinx Atlas tokens must be collected before Sphinx Depth")
        tokens = list(iter_jsonl_zst(token_path))
        tokens.sort(key=lambda row: str(row["token_id"]))
        if max_tokens is not None:
            tokens = tokens[:max_tokens]

        completed = 0
        skipped = 0
        rows = 0
        for token in tokens:
            token_id = str(token["token_id"])
            start = max(
                self.config.window.start,
                _optional_time(token.get("market_created_at"))
                or _optional_time(token.get("market_start_at"))
                or self.config.window.start,
            )
            market_end = (
                _optional_time(token.get("market_closed_at"))
                or _optional_time(token.get("market_end_at"))
                or self.config.window.end
            )
            end = min(self.config.window.end, market_end)
            if start >= end:
                continue
            cursor = start
            while cursor < end:
                if max_windows is not None and completed >= max_windows:
                    return self._receipt(completed, skipped, rows, partial=True)
                window_end = min(end, cursor + timedelta(days=self.window_days))
                start_ts = int(cursor.timestamp())
                end_ts = int(window_end.timestamp())
                partition = f"start={start_ts}-end={end_ts}"
                directory = f"token={token_id}"
                raw_path = (
                    self.config.data_dir / "raw" / "depth" / directory / f"{partition}.json.zst"
                )
                normalized_path = (
                    self.config.data_dir
                    / "normalized"
                    / "depth"
                    / directory
                    / f"{partition}.jsonl.zst"
                )
                if raw_path.exists() and normalized_path.exists():
                    skipped += 1
                    cursor = window_end
                    continue
                check_disk_reserve(
                    self.config.data_dir,
                    float(self.config.payload["storage"]["minimum_free_gib"]),
                )
                params = {
                    "market": token_id,
                    "startTs": start_ts,
                    "endTs": max(start_ts, end_ts - 1),
                    "fidelity": self.fidelity,
                }
                observed_at = now_utc()
                if raw_path.exists():
                    wrapper = read_json_zst(raw_path)
                    if not isinstance(wrapper, dict) or not isinstance(
                        wrapper.get("response"), dict
                    ):
                        raise TypeError(f"Invalid raw Depth window: {raw_path}")
                    payload = wrapper["response"]
                    observed_at = str(wrapper["observed_at"])
                else:
                    payload = self._fetch(params)
                    write_json_zst(
                        raw_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "source": "polymarket_clob_price_history",
                            "observed_at": observed_at,
                            "request": params,
                            "response": payload,
                        },
                    )
                normalized: list[dict[str, Any]] = []
                for point in payload["history"]:
                    if not isinstance(point, dict) or point.get("t") is None:
                        continue
                    timestamp = int(point["t"])
                    if not start_ts <= timestamp < end_ts:
                        continue
                    normalized.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "historical_price",
                            "source": "polymarket_clob_price_history",
                            "token_id": token_id,
                            "timestamp": utc_text(
                                datetime.fromtimestamp(
                                    timestamp,
                                    tz=self.config.window.start.tzinfo,
                                )
                            ),
                            "timestamp_unix": timestamp,
                            "price": str(point.get("p")),
                            "fidelity_minutes": self.fidelity,
                            "observed_at": observed_at,
                        }
                    )
                normalized.sort(key=lambda row: int(row["timestamp_unix"]))
                write_jsonl_zst(normalized_path, normalized)
                completed += 1
                rows += len(normalized)
                cursor = window_end
        return self._receipt(completed, skipped, rows, partial=False)

    def _receipt(self, completed: int, skipped: int, rows: int, *, partial: bool) -> dict[str, Any]:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_utc(),
            "fidelity_minutes": self.fidelity,
            "windows_completed_this_run": completed,
            "windows_skipped": skipped,
            "rows_written_this_run": rows,
            "partial": partial,
        }
        atomic_json(self.config.data_dir / "receipts" / "depth.json", receipt)
        return receipt
