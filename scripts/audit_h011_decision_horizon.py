"""Audit that every labeled H011 decision precedes its market close."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file

SPLIT_NAMES = {1: "train", 2: "validation", 3: "calibration", 4: "test"}


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _timestamp(value: object) -> int:
    if value is None or not str(value):
        return 0
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Catalog close timestamp has no timezone")
    return int(parsed.timestamp())


def market_horizons(pack_dir: Path, catalog_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Join sorted H011 market IDs to H009 close times without a giant string map."""

    index_path = pack_dir / "index" / "markets.jsonl.zst"
    connection = sqlite3.connect(f"file:{catalog_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    catalog = iter(
        connection.execute("SELECT condition_id, closed_at FROM markets ORDER BY condition_id")
    )
    try:
        catalog_row = next(catalog)
    except StopIteration as error:
        connection.close()
        raise RuntimeError("H009 catalog has no markets") from error
    end_values: list[int] = []
    close_values: list[int] = []
    previous_condition = ""
    for expected_market_id, index_row in enumerate(iter_jsonl_zst(index_path)):
        market_id = int(index_row["market_id"])
        condition_id = str(index_row["condition_id"])
        if market_id != expected_market_id or condition_id <= previous_condition:
            connection.close()
            raise RuntimeError("H011 market index order changed")
        previous_condition = condition_id
        while str(catalog_row["condition_id"]) < condition_id:
            try:
                catalog_row = next(catalog)
            except StopIteration as error:
                connection.close()
                raise RuntimeError(f"H009 catalog is missing {condition_id}") from error
        if str(catalog_row["condition_id"]) != condition_id:
            connection.close()
            raise RuntimeError(f"H009 catalog is missing {condition_id}")
        end_values.append(int(index_row["end_at_unix"]))
        close_values.append(_timestamp(catalog_row["closed_at"]))
    connection.close()
    return np.asarray(end_values, dtype=np.int64), np.asarray(close_values, dtype=np.int64)


def audit(pack_dir: Path, catalog_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    manifest = _load_object(pack_dir / "manifest.json")
    if manifest.get("valid") is not True or manifest.get("test_labels_opened") is not False:
        raise RuntimeError("H011 horizon audit requires a valid closed-test pack")
    end_times, close_times = market_horizons(pack_dir, catalog_path)
    counts: Counter[str] = Counter()
    after_close_markets: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES.values()}
    maximum_after_close: dict[str, int] = {name: 0 for name in SPLIT_NAMES.values()}
    for shard in sorted((pack_dir / "shards").glob("date=*")):
        if not shard.is_dir():
            continue
        timestamps = np.load(shard / "timestamps.npy", mmap_mode="r")
        market_ids = np.load(shard / "market_ids.npy", mmap_mode="r")
        split_codes = np.load(shard / "split_codes.npy", mmap_mode="r")
        label_mask = np.load(shard / "label_mask.npy", mmap_mode="r")
        if not (len(timestamps) == len(market_ids) == len(split_codes) == len(label_mask)):
            raise RuntimeError(f"H011 shard arrays do not align: {shard.name}")
        if len(market_ids) and int(market_ids.max()) >= len(close_times):
            raise RuntimeError(f"H011 shard has an unknown market ID: {shard.name}")
        for code, split in SPLIT_NAMES.items():
            selected = (split_codes == code) & (label_mask == 1)
            if not selected.any():
                continue
            selected_market_ids = np.asarray(market_ids[selected], dtype=np.int64)
            selected_timestamps = np.asarray(timestamps[selected], dtype=np.int64)
            selected_ends = end_times[selected_market_ids]
            selected_closes = close_times[selected_market_ids]
            after_end = (selected_ends > 0) & (selected_timestamps >= selected_ends)
            after_close = (selected_closes > 0) & (selected_timestamps >= selected_closes)
            counts[f"{split}:labeled_decisions"] += len(selected_timestamps)
            counts[f"{split}:unknown_end"] += int(np.count_nonzero(selected_ends == 0))
            counts[f"{split}:unknown_close"] += int(np.count_nonzero(selected_closes == 0))
            counts[f"{split}:at_or_after_end"] += int(np.count_nonzero(after_end))
            counts[f"{split}:at_or_after_close"] += int(np.count_nonzero(after_close))
            if after_close.any():
                after_close_markets[split].update(
                    int(value) for value in np.unique(selected_market_ids[after_close])
                )
                maximum_after_close[split] = max(
                    maximum_after_close[split],
                    int(np.max(selected_timestamps[after_close] - selected_closes[after_close])),
                )
    splits = {
        split: {
            "labeled_decisions": counts[f"{split}:labeled_decisions"],
            "unknown_end": counts[f"{split}:unknown_end"],
            "unknown_close": counts[f"{split}:unknown_close"],
            "at_or_after_end": counts[f"{split}:at_or_after_end"],
            "at_or_after_close": counts[f"{split}:at_or_after_close"],
            "markets_at_or_after_close": len(after_close_markets[split]),
            "maximum_seconds_after_close": maximum_after_close[split],
        }
        for split in SPLIT_NAMES.values()
    }
    development_post_close = sum(
        int(splits[split]["at_or_after_close"]) for split in ("train", "validation", "calibration")
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_decision_horizon_audit",
        "generated_at": now_utc(),
        "valid": development_post_close == 0,
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "market_index_sha256": sha256_file(pack_dir / "index" / "markets.jsonl.zst"),
        "catalog_bytes": catalog_path.stat().st_size,
        "markets": len(close_times),
        "splits": splits,
        "development_labeled_decisions_at_or_after_close": development_post_close,
        "test_labels_opened": False,
        "evidence_boundary": (
            "This audit qualifies decision timing only; it does not establish model lift or profit."
        ),
    }
    if output_path is not None:
        atomic_json(output_path, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--catalog", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    result = audit(
        args.pack_dir.resolve(),
        args.catalog.resolve(),
        args.output.resolve() if args.output else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
