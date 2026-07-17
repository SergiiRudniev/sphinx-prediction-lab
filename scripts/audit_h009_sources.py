"""Audit H009 Atlas/Ledger structure without opening terminal payout fields."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.chronicle_h009 import UnionFind, market_seed_from_atlas
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def audit_sources(config_path: Path, data_dir: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    atlas_config = config["sources"]["atlas"]
    ledger_config = config["sources"]["ledger"]
    markets_path = data_dir / str(atlas_config["markets"]["path"])
    counts: Counter[str] = Counter()
    union = UnionFind()
    resolution_statuses: Counter[str] = Counter()
    event_id_multiplicity: Counter[str] = Counter()

    for row in iter_jsonl_zst(markets_path):
        counts["market_rows"] += 1
        seed = market_seed_from_atlas(row)
        if seed is None:
            counts["invalid_structural_rows"] += 1
            continue
        counts["structural_rows"] += 1
        counts["binary_market_rows"] += int(seed.structurally_binary)
        counts["non_binary_market_rows"] += int(not seed.structurally_binary)
        counts["neg_risk_market_rows"] += int(seed.neg_risk)
        counts["closed_timestamp_rows"] += int(seed.closed_at is not None)
        counts["orphan_market_rows"] += int(not seed.event_ids)
        event_id_multiplicity[str(len(seed.event_ids))] += 1
        resolution_statuses[str(seed.resolution_status or "null")] += 1
        union.add_group(seed.event_ids)

    event_components = union.component_ids()
    scope_receipt_root = data_dir / "receipts" / str(ledger_config["namespace"])
    scope_counts: Counter[str] = Counter()
    for receipt_path in sorted(scope_receipt_root.glob("*.json")):
        scope_receipt = _load_object(receipt_path)
        scope_counts["scope_receipts"] += 1
        scope_counts["rows"] += int(scope_receipt.get("rows") or 0)
        scope_counts["condition_ids"] += len(scope_receipt.get("condition_ids") or [])
        scope_counts["complete"] += int(scope_receipt.get("complete") is True)
        scope_counts["gaps"] += int(scope_receipt.get("gaps") or 0)

    audit_path = data_dir / str(ledger_config["audit_receipt"]["path"])
    profile_path = data_dir / str(ledger_config["profile_receipt"]["path"])
    qualification_path = data_dir / str(ledger_config["qualification_receipt"]["path"])
    ledger_audit = _load_object(audit_path)
    source_hashes: dict[str, str] = {}
    for source_id in ("markets", "events", "tokens"):
        source = atlas_config[source_id]
        source_path = data_dir / str(source["path"])
        digest = sha256_file(source_path)
        if digest != source["sha256"]:
            raise RuntimeError(f"Atlas {source_id} hash changed")
        source_hashes[f"atlas_{source_id}"] = digest
    for source_id, source_path, expected in (
        ("ledger_audit", audit_path, ledger_config["audit_receipt"]["sha256"]),
        ("ledger_profile", profile_path, ledger_config["profile_receipt"]["sha256"]),
        (
            "ledger_qualification",
            qualification_path,
            ledger_config["qualification_receipt"]["sha256"],
        ),
    ):
        digest = sha256_file(source_path)
        if digest != expected:
            raise RuntimeError(f"{source_id} hash changed")
        source_hashes[source_id] = digest

    expected_market_rows = int(atlas_config["markets"]["rows"])
    expected_ledger_rows = int(ledger_config["rows"])
    valid = (
        counts["market_rows"] == expected_market_rows
        and scope_counts["scope_receipts"] == int(ledger_config["scope_groups"])
        and scope_counts["rows"] == expected_ledger_rows
        and scope_counts["complete"] == scope_counts["scope_receipts"]
        and scope_counts["gaps"] == 0
        and ledger_audit.get("valid") is True
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_source_audit",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_id": str(config["id"]),
        "config_sha256": sha256_file(config_path),
        "valid": valid,
        "terminal_fields_accessed": False,
        "atlas": {
            "counts": dict(counts),
            "resolution_statuses": dict(resolution_statuses),
            "event_id_multiplicity": dict(event_id_multiplicity),
            "connected_event_components": len(set(event_components.values())),
            "semantic_snapshot_point_in_time": False,
        },
        "ledger": dict(scope_counts),
        "source_hashes": source_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_path, receipt)
    if not valid:
        raise RuntimeError("H009 source audit did not match the registered contract")
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output or data_dir / "receipts" / "sphinx-chronicle-h009-source-audit.json"
    receipt = audit_sources(args.config.resolve(), data_dir, output.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
