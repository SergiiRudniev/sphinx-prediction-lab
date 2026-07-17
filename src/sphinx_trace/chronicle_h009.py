"""Causal full-universe primitives for the H009 Sphinx Chronicle build."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import zstandard


def parse_json_string_list(value: object) -> tuple[str, ...]:
    """Parse a Gamma JSON-encoded string array without accepting scalar values."""

    if isinstance(value, list):
        parsed: object = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ()
    else:
        return ()
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return ()
    return tuple(parsed)


def parse_optional_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class MarketSeed:
    condition_id: str
    market_id: str
    event_ids: tuple[str, ...]
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]
    question: str
    description: str
    created_at: str | None
    start_at: str | None
    end_at: str | None
    closed_at: str | None
    observed_at: str | None
    resolution_status: str | None
    neg_risk: bool
    source_condition_available: bool

    @property
    def structurally_binary(self) -> bool:
        return len(self.outcomes) == 2 and len(self.token_ids) == 2


def market_seed_from_atlas(
    row: Mapping[str, Any],
    *,
    allow_missing_condition: bool = False,
) -> MarketSeed | None:
    """Read structural metadata while deliberately ignoring terminal prices."""

    source_condition_id = str(row.get("condition_id") or "").lower()
    market_id = str(row.get("market_id") or "")
    payload_value = row.get("source_payload")
    if not market_id or not isinstance(payload_value, Mapping):
        return None
    if not source_condition_id and not allow_missing_condition:
        return None
    condition_id = source_condition_id or (
        "unavailable:" + hashlib.sha256(f"market:{market_id}".encode()).hexdigest()
    )
    payload: Mapping[str, Any] = payload_value
    event_value = row.get("event_ids")
    event_ids = (
        tuple(str(value) for value in event_value if str(value))
        if isinstance(event_value, list)
        else ()
    )
    return MarketSeed(
        condition_id=condition_id,
        market_id=market_id,
        event_ids=event_ids,
        outcomes=parse_json_string_list(payload.get("outcomes")),
        token_ids=parse_json_string_list(payload.get("clobTokenIds")),
        question=str(payload.get("question") or ""),
        description=str(payload.get("description") or ""),
        created_at=_optional_text(row.get("created_at")),
        start_at=_optional_text(row.get("start_at")),
        end_at=_optional_text(row.get("end_at")),
        closed_at=_optional_text(row.get("closed_at")),
        observed_at=_optional_text(row.get("observed_at")),
        resolution_status=_optional_text(row.get("resolution_status")),
        neg_risk=bool(row.get("neg_risk")),
        source_condition_available=bool(source_condition_id),
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def terminal_payout_from_atlas(
    row: Mapping[str, Any],
    *,
    split_id: str | None,
    label_splits: frozenset[str],
) -> tuple[float, ...] | None:
    """Open a terminal payout only after proving that its split permits labels.

    The early split check is intentional: a test payload may use a guard object
    that raises whenever ``outcomePrices`` is touched.
    """

    if split_id not in label_splits:
        return None
    if row.get("resolution_status") != "resolved":
        return None
    payload_value = row.get("source_payload")
    if not isinstance(payload_value, Mapping):
        return None
    prices_raw = payload_value.get("outcomePrices")
    values = parse_json_string_list(prices_raw)
    if len(values) != 2:
        return None
    try:
        prices = tuple(float(value) for value in values)
    except ValueError:
        return None
    if any(value < 0.0 or value > 1.0 for value in prices):
        return None
    if not (
        (prices[0] >= 0.999 and prices[1] <= 0.001) or (prices[1] >= 0.999 and prices[0] <= 0.001)
    ):
        return None
    return prices


class UnionFind:
    """Small deterministic union-find for Atlas event connectivity."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self._parent:
            self._parent[value] = value
            self._rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self._rank[left_root]
        right_rank = self._rank[right_root]
        if left_rank < right_rank or (left_rank == right_rank and left_root > right_root):
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        self._parent[right_root] = left_root
        if left_rank == right_rank:
            self._rank[left_root] += 1

    def add_group(self, values: tuple[str, ...]) -> None:
        if not values:
            return
        first = values[0]
        self.add(first)
        for value in values[1:]:
            self.union(first, value)

    def component_ids(self) -> dict[str, str]:
        members: dict[str, list[str]] = {}
        for value in self._parent:
            members.setdefault(self.find(value), []).append(value)
        output: dict[str, str] = {}
        for values in members.values():
            digest = hashlib.sha256(
                ("event-component-v1\n" + "\n".join(sorted(values))).encode()
            ).hexdigest()
            for value in values:
                output[value] = digest
        return output


def component_id_for_market(
    condition_id: str,
    event_ids: tuple[str, ...],
    event_component_ids: Mapping[str, str],
) -> str:
    if event_ids:
        component = event_component_ids.get(event_ids[0])
        if component is None:
            raise KeyError(event_ids[0])
        if any(event_component_ids.get(event_id) != component for event_id in event_ids):
            raise ValueError("Market event IDs were not joined into one component")
        return component
    return hashlib.sha256(f"orphan-market-v1\n{condition_id}".encode()).hexdigest()


def snapshot_reasons(
    *,
    event_trade_count: int,
    trade_timestamp_unix: int,
    last_snapshot_timestamp_unix: int | None,
    early_maximum: int = 1024,
    stride: int = 128,
    heartbeat_seconds: int = 21_600,
) -> tuple[str, ...]:
    if event_trade_count <= 0:
        raise ValueError("event_trade_count must be positive")
    if stride <= 0 or heartbeat_seconds <= 0:
        raise ValueError("stride and heartbeat must be positive")
    reasons: list[str] = []
    if event_trade_count == 1:
        reasons.append("first")
    elif event_trade_count <= early_maximum and event_trade_count & (event_trade_count - 1) == 0:
        reasons.append("early_power_of_two")
    elif event_trade_count > early_maximum and (event_trade_count - early_maximum) % stride == 0:
        reasons.append("trade_stride")
    if (
        last_snapshot_timestamp_unix is not None
        and trade_timestamp_unix - last_snapshot_timestamp_unix >= heartbeat_seconds
    ):
        reasons.append("heartbeat")
    return tuple(reasons)


def decision_id(component_id: str, timestamp_unix: int, trade_count: int) -> str:
    return hashlib.sha256(
        f"chronicle-decision-v1|{component_id}|{timestamp_unix}|{trade_count}".encode()
    ).hexdigest()


def raw_jsonl_zst_lines(path: Path) -> io.BufferedReader:
    """Return a buffered raw-line reader; the caller must close it."""

    source: BinaryIO = path.open("rb")
    reader = zstandard.ZstdDecompressor().stream_reader(source, closefd=True)
    return io.BufferedReader(reader, buffer_size=1024 * 1024)


def extract_json_string(line: bytes, field: bytes) -> str:
    marker = b'"' + field + b'":"'
    start = line.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = line.find(b'"', start)
    if end < 0:
        return ""
    return line[start:end].decode("ascii")


def extract_json_int(line: bytes, field: bytes) -> int:
    marker = b'"' + field + b'":'
    start = line.find(marker)
    if start < 0:
        raise ValueError(f"Missing integer field {field.decode()}")
    start += len(marker)
    end = start
    while end < len(line) and 48 <= line[end] <= 57:
        end += 1
    if end == start:
        raise ValueError(f"Invalid integer field {field.decode()}")
    return int(line[start:end])


def trade_sort_key(line: bytes) -> tuple[int, str]:
    timestamp = extract_json_int(line, b"timestamp_unix")
    trade_id = extract_json_string(line, b"trade_id")
    if not trade_id:
        raise ValueError("Trade row has no trade_id")
    return timestamp, trade_id
