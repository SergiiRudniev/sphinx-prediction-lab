"""Closed-test catalog selection for H010 development replay."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sphinx_trace.replay_h010 import BinaryMarketContract

DEVELOPMENT_SPLITS = frozenset({"validation", "calibration"})


@dataclass(frozen=True, slots=True)
class MarketResolution:
    condition_id: str
    timestamp_unix: int
    payouts: tuple[Decimal, Decimal]


@dataclass(frozen=True, slots=True)
class CatalogSelection:
    contracts: dict[str, BinaryMarketContract]
    resolutions: tuple[MarketResolution, ...]
    split_counts: dict[str, int]


def _string_pair(value: object, field: str) -> tuple[str, str]:
    payload: object = json.loads(str(value))
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not all(isinstance(item, str) and item for item in payload)
    ):
        raise ValueError(f"Catalog {field} is not a non-empty string pair")
    return str(payload[0]), str(payload[1])


def _payout_pair(value: object) -> tuple[Decimal, Decimal]:
    payload: object = json.loads(str(value))
    if payload not in ([1.0, 0.0], [0.0, 1.0]):
        raise ValueError("Catalog terminal label is not a replayable binary payout")
    assert isinstance(payload, list)
    return Decimal(str(payload[0])), Decimal(str(payload[1]))


def _timestamp(value: object) -> int:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Catalog resolution timestamp has no timezone")
    return int(parsed.timestamp())


def load_development_catalog(
    catalog_path: Path,
    condition_ids: set[str],
    *,
    allowed_splits: frozenset[str] = DEVELOPMENT_SPLITS,
) -> CatalogSelection:
    """Load only requested development contracts without exposing test labels."""

    if not condition_ids:
        return CatalogSelection({}, (), {})
    if not allowed_splits or not allowed_splits <= DEVELOPMENT_SPLITS:
        raise ValueError("H010 catalog selection may use validation/calibration only")
    normalized = {condition.lower() for condition in condition_ids}
    connection = sqlite3.connect(f"file:{catalog_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if metadata.get("test_terminal_fields_accessed") != "false":
        connection.close()
        raise RuntimeError("H010 catalog metadata does not prove closed test")
    test_labels = int(
        connection.execute(
            "SELECT COUNT(*) FROM markets WHERE split_id='test' AND terminal_label IS NOT NULL"
        ).fetchone()[0]
    )
    if test_labels:
        connection.close()
        raise RuntimeError("H010 catalog contains opened test labels")
    connection.execute("CREATE TEMP TABLE selected(condition_id TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO selected(condition_id) VALUES (?)",
        ((condition,) for condition in sorted(normalized)),
    )
    rows = connection.execute(
        """
        SELECT m.condition_id, m.component_id, m.outcomes, m.token_ids,
               m.closed_at, m.split_id, m.terminal_label, m.replayable
        FROM markets AS m
        INNER JOIN selected AS s ON s.condition_id = m.condition_id
        ORDER BY m.condition_id
        """
    )
    contracts: dict[str, BinaryMarketContract] = {}
    resolutions: list[MarketResolution] = []
    split_counts: dict[str, int] = {}
    for row in rows:
        condition_id = str(row["condition_id"])
        split = str(row["split_id"])
        if split not in allowed_splits:
            connection.close()
            raise RuntimeError(f"H010 requested condition is outside selected split: {split}")
        if not bool(row["replayable"]) or row["closed_at"] is None or row["terminal_label"] is None:
            connection.close()
            raise RuntimeError(f"H010 requested market is not replayable: {condition_id}")
        contract = BinaryMarketContract(
            condition_id=condition_id,
            component_id=str(row["component_id"]),
            outcomes=_string_pair(row["outcomes"], "outcomes"),
            token_ids=_string_pair(row["token_ids"], "token_ids"),
        )
        contracts[condition_id] = contract
        resolutions.append(
            MarketResolution(
                condition_id,
                _timestamp(row["closed_at"]),
                _payout_pair(row["terminal_label"]),
            )
        )
        split_counts[split] = split_counts.get(split, 0) + 1
    connection.close()
    missing = normalized - contracts.keys()
    if missing:
        raise RuntimeError(f"H010 catalog is missing {len(missing)} requested markets")
    resolutions.sort(key=lambda row: (row.timestamp_unix, row.condition_id))
    return CatalogSelection(contracts, tuple(resolutions), split_counts)
