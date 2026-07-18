"""Build a storage-efficient H011 view with strictly pre-close labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_h011_decision_horizon import market_horizons
from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_qualification_v1.json"
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "audit_h011_decision_horizon.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Qualified pack target changed: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _atomic_numpy(path: Path, values: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def qualify(
    config_path: Path,
    source_pack: Path,
    catalog_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    config_hash = sha256_file(config_path)
    implementation_hash = _implementation_digest()
    source_manifest_path = source_pack / "manifest.json"
    source_manifest = _load_object(source_manifest_path)
    if (
        source_manifest.get("valid") is not True
        or source_manifest.get("test_labels_opened") is not False
    ):
        raise RuntimeError("H011 qualification requires a valid closed-test source pack")
    output_dir.mkdir(parents=True, exist_ok=True)
    _end_times, close_times = market_horizons(source_pack, catalog_path)
    totals: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()
    receipt_hashes: list[str] = []
    for source_shard in sorted((source_pack / "shards").glob("date=*")):
        if not source_shard.is_dir():
            continue
        date = source_shard.name.removeprefix("date=")
        source_receipt_path = source_pack / "receipts" / f"date={date}.json"
        source_receipt_hash = sha256_file(source_receipt_path)
        output_receipt_path = output_dir / "receipts" / f"date={date}.json"
        output_shard = output_dir / "shards" / source_shard.name
        if output_receipt_path.exists():
            cached = _load_object(output_receipt_path)
            if (
                cached.get("config_sha256") != config_hash
                or cached.get("implementation_sha256") != implementation_hash
                or cached.get("source_receipt_sha256") != source_receipt_hash
            ):
                raise RuntimeError(f"Qualified H011 receipt changed at {date}")
            mask_path = output_shard / "label_mask.npy"
            if cached.get("label_mask_sha256") != sha256_file(mask_path):
                raise RuntimeError(f"Qualified H011 label mask changed at {date}")
            totals.update({key: int(value) for key, value in cached["counts"].items()})
            receipt_hashes.append(f"{date}:{sha256_file(output_receipt_path)}")
            continue
        timestamps = np.load(source_shard / "timestamps.npy", mmap_mode="r")
        market_ids = np.load(source_shard / "market_ids.npy", mmap_mode="r")
        split_codes = np.load(source_shard / "split_codes.npy", mmap_mode="r")
        source_mask = np.load(source_shard / "label_mask.npy", mmap_mode="r")
        selected_market_ids = np.asarray(market_ids, dtype=np.int64)
        closes = close_times[selected_market_ids]
        development = np.isin(split_codes, [1, 2, 3])
        eligible = (closes > 0) & (np.asarray(timestamps) < closes)
        qualified_mask = np.asarray(
            (source_mask == 1) & (~development | eligible),
            dtype=np.uint8,
        )
        if int(qualified_mask[split_codes == 4].sum()) != 0:
            raise RuntimeError("Qualified H011 view exposed a test label")
        counts = {
            "decision_rows": len(qualified_mask),
            "source_labeled_rows": int(np.asarray(source_mask).sum()),
            "qualified_labeled_rows": int(qualified_mask.sum()),
            "masked_at_or_after_close": int(
                np.count_nonzero((source_mask == 1) & development & (closes > 0) & ~eligible)
            ),
            "masked_missing_close": int(
                np.count_nonzero((source_mask == 1) & development & (closes == 0))
            ),
            "test_label_rows": int(qualified_mask[split_codes == 4].sum()),
        }
        for source_file in source_shard.iterdir():
            if source_file.name == "label_mask.npy" or not source_file.is_file():
                continue
            link_counts[_link_or_copy(source_file, output_shard / source_file.name)] += 1
        mask_path = output_shard / "label_mask.npy"
        _atomic_numpy(mask_path, qualified_mask)
        receipt = {
            "schema_version": "1.0.0",
            "record_type": "h011_qualified_feature_day_receipt",
            "generated_at": now_utc(),
            "date": date,
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "source_receipt_sha256": source_receipt_hash,
            "label_mask_sha256": sha256_file(mask_path),
            "label_mask_bytes": mask_path.stat().st_size,
            "counts": counts,
        }
        atomic_json(output_receipt_path, receipt)
        totals.update(counts)
        receipt_hashes.append(f"{date}:{sha256_file(output_receipt_path)}")
    for name in ("normalization.npz", "normalization.json"):
        link_counts[_link_or_copy(source_pack / name, output_dir / name)] += 1
    for source_file in (source_pack / "index").iterdir():
        if source_file.is_file():
            link_counts[_link_or_copy(source_file, output_dir / "index" / source_file.name)] += 1
    valid = (
        totals["decision_rows"] == int(source_manifest["decision_rows"])
        and totals["masked_at_or_after_close"]
        == int(config["triggering_audit"]["development_labeled_decisions_at_or_after_close"])
        and totals["masked_missing_close"] == 0
        and totals["test_label_rows"] == 0
    )
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_qualified_feature_pack_manifest",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "valid": valid,
        "full_run": bool(source_manifest["full_run"]),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_pack": str(source_pack),
        "days": int(source_manifest["days"]),
        "stream_rows": int(source_manifest["stream_rows"]),
        "decision_rows": int(source_manifest["decision_rows"]),
        "markets": int(source_manifest["markets"]),
        "components": int(source_manifest["components"]),
        "wallets": int(source_manifest["wallets"]),
        "feature_width": int(source_manifest["feature_width"]),
        "feature_names": source_manifest["feature_names"],
        "feature_names_sha256": source_manifest["feature_names_sha256"],
        "normalization": source_manifest["normalization"],
        "qualification": dict(totals),
        "daily_receipts_sha256": hashlib.sha256(
            ("\n".join(receipt_hashes) + "\n").encode()
        ).hexdigest(),
        "storage_methods": dict(link_counts),
        "test_label_rows": 0,
        "test_labels_opened": False,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if not valid:
        raise RuntimeError("Qualified H011 feature view failed acceptance")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--source-pack", type=Path, required=True)
    value.add_argument("--catalog", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = qualify(
        args.config.resolve(),
        args.source_pack.resolve(),
        args.catalog.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "days": result["days"],
                "decision_rows": result["decision_rows"],
                "qualification": result["qualification"],
                "storage_methods": result["storage_methods"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
