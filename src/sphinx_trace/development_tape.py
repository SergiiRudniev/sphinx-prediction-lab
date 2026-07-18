"""Build a compact, closed-test H010 tape from the qualified H011 corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import (
    atomic_json,
    iter_jsonl_zst,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_trace.h010_catalog import CatalogSelection, load_development_catalog

SPLIT_CODES = {2: "validation", 3: "calibration"}


@dataclass(frozen=True, slots=True)
class DevelopmentWindow:
    split: str
    first_decision_unix: int


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def discover_development_windows(
    pack_dir: Path,
) -> tuple[dict[str, DevelopmentWindow], dict[str, int]]:
    """Discover every strictly qualified validation/calibration decision market."""

    manifest = _load_object(pack_dir / "manifest.json")
    if manifest.get("valid") is not True or manifest.get("test_labels_opened") is not False:
        raise RuntimeError("H010 tape requires a valid closed-test H011 pack")
    windows: dict[str, DevelopmentWindow] = {}
    decision_counts = {split: 0 for split in SPLIT_CODES.values()}
    shards = sorted(path for path in (pack_dir / "shards").glob("date=*") if path.is_dir())
    if not shards:
        raise RuntimeError("H010 tape found no H011 feature shards")
    for shard in shards:
        split_codes = np.load(shard / "split_codes.npy", mmap_mode="r")
        label_mask = np.load(shard / "label_mask.npy", mmap_mode="r")
        timestamps = np.load(shard / "timestamps.npy", mmap_mode="r")
        if not len(split_codes) == len(label_mask) == len(timestamps):
            raise RuntimeError(f"H011 qualification arrays do not align in {shard.name}")
        example_count = 0
        for row, example in enumerate(iter_jsonl_zst(shard / "examples.jsonl.zst")):
            example_count += 1
            split = SPLIT_CODES.get(int(split_codes[row]))
            if split is None or not bool(label_mask[row]):
                continue
            timestamp = int(timestamps[row])
            if timestamp != int(example["decision_time_unix"]):
                raise RuntimeError(f"H011 decision time changed at {shard.name}:{row}")
            condition_id = str(example["condition_id"]).lower()
            current = windows.get(condition_id)
            if current is not None and current.split != split:
                raise RuntimeError(f"H010 condition crosses development splits: {condition_id}")
            if current is None or timestamp < current.first_decision_unix:
                windows[condition_id] = DevelopmentWindow(split, timestamp)
            decision_counts[split] += 1
        if example_count != len(split_codes):
            raise RuntimeError(f"H011 examples do not align in {shard.name}")
    if not windows:
        raise RuntimeError("H010 tape found no qualified development conditions")
    return windows, decision_counts


def _source_digest(pack_dir: Path, chronicle_dir: Path) -> str:
    digest = hashlib.sha256()
    for label, path in (
        ("pack_manifest", pack_dir / "manifest.json"),
        ("chronicle_receipt", chronicle_dir / "receipt.json"),
        ("stream_manifest", chronicle_dir / "stream-manifest.json"),
        ("validation_receipt", chronicle_dir / "validation-receipt.json"),
    ):
        digest.update(f"{label}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _conditions_digest(
    windows: dict[str, DevelopmentWindow],
    catalog: CatalogSelection,
) -> str:
    resolutions = {row.condition_id: row for row in catalog.resolutions}
    digest = hashlib.sha256()
    for condition_id in sorted(windows):
        window = windows[condition_id]
        contract = catalog.contracts[condition_id]
        resolution = resolutions[condition_id]
        digest.update(
            (
                f"{condition_id}:{window.split}:{window.first_decision_unix}:"
                f"{resolution.timestamp_unix}:{','.join(contract.token_ids)}:"
                f"{','.join(str(value) for value in resolution.payouts)}\n"
            ).encode()
        )
    return digest.hexdigest()


def _condition_records(
    windows: dict[str, DevelopmentWindow],
    catalog: CatalogSelection,
) -> list[dict[str, Any]]:
    resolutions = {row.condition_id: row for row in catalog.resolutions}
    records: list[dict[str, Any]] = []
    for condition_id in sorted(windows):
        window = windows[condition_id]
        contract = catalog.contracts[condition_id]
        resolution = resolutions[condition_id]
        if window.first_decision_unix >= resolution.timestamp_unix:
            raise RuntimeError(f"H010 qualified decision is not before close: {condition_id}")
        records.append(
            {
                "schema_version": "1.0.0",
                "record_type": "h010_development_condition",
                "condition_id": condition_id,
                "component_id": contract.component_id,
                "split": window.split,
                "first_decision_unix": window.first_decision_unix,
                "resolution_unix": resolution.timestamp_unix,
                "outcomes": list(contract.outcomes),
                "token_ids": list(contract.token_ids),
                "payouts": [str(value) for value in resolution.payouts],
            }
        )
    return records


def build_development_tape(
    pack_dir: Path,
    chronicle_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Filter the immutable H009 stream to all causal development replay windows."""

    validation = _load_object(chronicle_dir / "validation-receipt.json")
    stream_manifest = _load_object(chronicle_dir / "stream-manifest.json")
    if (
        validation.get("valid") is not True
        or validation.get("deep_hashes") is not True
        or validation.get("hash_checks", {}).get("stream") is not True
        or int(validation.get("catalog", {}).get("test_terminal_labels", -1)) != 0
        or stream_manifest.get("globally_ordered") is not True
    ):
        raise RuntimeError("H010 tape requires a deep-hash-valid, closed-test H009 stream")
    windows, decision_counts = discover_development_windows(pack_dir)
    catalog = load_development_catalog(
        chronicle_dir / "catalog.sqlite",
        set(windows),
    )
    source_digest = _source_digest(pack_dir, chronicle_dir)
    conditions_digest = _conditions_digest(windows, catalog)
    condition_records = _condition_records(windows, catalog)
    conditions_path = output_dir / "conditions.jsonl.zst"
    if conditions_path.exists():
        conditions_receipt = _load_object(output_dir / "conditions.json")
        if sha256_file(conditions_path) != conditions_receipt.get("sha256"):
            raise RuntimeError("H010 development conditions artifact changed")
        if (
            conditions_receipt.get("conditions_digest") != conditions_digest
            or conditions_receipt.get("source_digest") != source_digest
        ):
            raise RuntimeError("H010 development condition contract changed")
    else:
        rows, size = write_jsonl_zst(conditions_path, condition_records)
        atomic_json(
            output_dir / "conditions.json",
            {
                "schema_version": "1.0.0",
                "record_type": "h010_development_conditions_receipt",
                "generated_at": now_utc(),
                "rows": rows,
                "bytes": size,
                "sha256": sha256_file(conditions_path),
                "conditions_digest": conditions_digest,
                "source_digest": source_digest,
                "split_counts": catalog.split_counts,
                "test_labels_opened": False,
            },
        )

    resolutions = {row.condition_id: row.timestamp_unix for row in catalog.resolutions}
    source_shards = stream_manifest.get("shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise RuntimeError("H010 tape found no source stream shards")
    total_source_rows = 0
    total_rows = 0
    total_bytes = 0
    shard_digest = hashlib.sha256()
    for ordinal, source_value in enumerate(source_shards):
        if not isinstance(source_value, dict):
            raise TypeError("H009 stream shard manifest row must be an object")
        source = dict(source_value)
        date = str(source["date"])
        source_path = chronicle_dir / str(source["path"])
        output_path = output_dir / "stream" / f"date={date}.jsonl.zst"
        receipt_path = output_dir / "receipts" / f"date={date}.json"
        contract = {
            "date": date,
            "ordinal": ordinal,
            "source_sha256": str(source["sha256"]),
            "source_rows": int(source["rows"]),
            "source_digest": source_digest,
            "conditions_digest": conditions_digest,
        }
        if output_path.exists() or receipt_path.exists():
            if not output_path.exists() or not receipt_path.exists():
                raise RuntimeError(f"Incomplete H010 development tape shard exists at {date}")
            receipt = _load_object(receipt_path)
            if any(receipt.get(key) != value for key, value in contract.items()):
                raise RuntimeError(f"H010 development tape contract changed at {date}")
            if receipt.get("sha256") != sha256_file(output_path):
                raise RuntimeError(f"H010 development tape shard changed at {date}")
        else:
            scanned = 0
            retained = 0

            def selected_rows(
                source_path: Path = source_path,
                date: str = date,
            ) -> Iterator[dict[str, Any]]:
                nonlocal scanned, retained
                previous_key: tuple[int, str] | None = None
                for payload in iter_jsonl_zst(source_path):
                    scanned += 1
                    timestamp = int(payload["timestamp_unix"])
                    trade_id = str(payload["trade_id"])
                    key = (timestamp, trade_id)
                    if previous_key is not None and key < previous_key:
                        raise RuntimeError(f"H009 stream order regressed at {date}")
                    previous_key = key
                    condition_id = str(payload["condition_id"]).lower()
                    window = windows.get(condition_id)
                    if window is None or not (
                        window.first_decision_unix <= timestamp <= resolutions[condition_id]
                    ):
                        continue
                    retained += 1
                    yield {**payload, "development_split": window.split}

            rows, size = write_jsonl_zst(output_path, selected_rows())
            if scanned != int(source["rows"]) or rows != retained:
                raise RuntimeError(f"H010 development tape counts changed at {date}")
            receipt = {
                "schema_version": "1.0.0",
                "record_type": "h010_development_tape_shard_receipt",
                "generated_at": now_utc(),
                **contract,
                "rows": rows,
                "bytes": size,
                "sha256": sha256_file(output_path),
            }
            atomic_json(receipt_path, receipt)
        total_source_rows += int(receipt["source_rows"])
        total_rows += int(receipt["rows"])
        total_bytes += int(receipt["bytes"])
        shard_digest.update(f"{date}:{receipt['sha256']}\n".encode())

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h010_development_tape_manifest",
        "generated_at": now_utc(),
        "valid": total_source_rows == int(stream_manifest["rows"]),
        "source_digest": source_digest,
        "conditions_digest": conditions_digest,
        "conditions": len(windows),
        "condition_split_counts": catalog.split_counts,
        "decision_split_counts": decision_counts,
        "source_rows_scanned": total_source_rows,
        "retained_rows": total_rows,
        "bytes": total_bytes,
        "shards": len(source_shards),
        "shard_digest": shard_digest.hexdigest(),
        "test_labels_opened": False,
        "test_rows_consumed": 0,
        "evidence_boundary": (
            "The tape is a causal development trade-liquidity proxy. It is not historical "
            "orderbook depth, executable-profit, untouched-test or paper-forward evidence."
        ),
    }
    if manifest["valid"] is not True:
        raise RuntimeError("H010 development tape did not scan the full H009 stream")
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
