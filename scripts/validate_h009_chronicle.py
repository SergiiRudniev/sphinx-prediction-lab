"""Validate H009 causal, completeness and artifact-binding invariants."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _hash_matches(root: Path, item: dict[str, Any]) -> bool:
    path = Path(str(item["path"]))
    resolved = path if path.is_absolute() else root / path
    return resolved.exists() and sha256_file(resolved) == item["sha256"]


def validate(
    config_path: Path,
    output_dir: Path,
    *,
    require_full: bool,
    require_polygon: bool,
    deep_hashes: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    config_hash = sha256_file(config_path)
    catalog_receipt = _load_object(output_dir / "receipts" / "catalog.json")
    episode_receipt = _load_object(output_dir / "receipts" / "episodes.json")
    stream = _load_object(output_dir / "stream-manifest.json")
    decisions = _load_object(output_dir / "decision-manifest.json")
    polygon_path = output_dir / "polygon-manifest.json"
    polygon = _load_object(polygon_path) if polygon_path.exists() else None
    violations: list[str] = []
    for name, payload in (
        ("catalog", catalog_receipt),
        ("episodes", episode_receipt),
        ("stream", stream),
        ("decisions", decisions),
    ):
        if payload.get("config_sha256") != config_hash:
            violations.append(f"{name}_config_hash")
    if catalog_receipt.get("test_terminal_fields_accessed") is not False:
        violations.append("test_terminal_fields_accessed")
    if catalog_receipt.get("semantic_snapshot_point_in_time") is not False:
        violations.append("atlas_snapshot_causality")

    connection = sqlite3.connect(
        f"file:{(output_dir / 'catalog.sqlite').as_posix()}?mode=ro",
        uri=True,
    )
    catalog_checks = {
        "test_terminal_labels": int(
            connection.execute(
                "SELECT COUNT(*) FROM markets WHERE split_id='test' AND terminal_label IS NOT NULL"
            ).fetchone()[0]
        ),
        "noncausal_semantics_enabled": int(
            connection.execute(
                "SELECT COUNT(*) FROM markets WHERE semantic_feature_available != 0"
            ).fetchone()[0]
        ),
        "nonreplayable_terminal_labels": int(
            connection.execute(
                "SELECT COUNT(*) FROM markets WHERE replayable=0 AND terminal_label IS NOT NULL"
            ).fetchone()[0]
        ),
        "market_count": int(connection.execute("SELECT COUNT(*) FROM markets").fetchone()[0]),
        "component_count": int(connection.execute("SELECT COUNT(*) FROM components").fetchone()[0]),
        "multi_market_components": int(
            connection.execute("SELECT COUNT(*) FROM components WHERE market_count > 1").fetchone()[
                0
            ]
        ),
        "neg_risk_components": int(
            connection.execute(
                "SELECT COUNT(*) FROM components WHERE neg_risk_market_count > 0"
            ).fetchone()[0]
        ),
    }
    connection.close()
    if any(
        catalog_checks[key] != 0
        for key in (
            "test_terminal_labels",
            "noncausal_semantics_enabled",
            "nonreplayable_terminal_labels",
        )
    ):
        violations.append("catalog_causal_or_label_gate")
    if catalog_checks["market_count"] != int(config["sources"]["atlas"]["markets"]["rows"]):
        violations.append("atlas_market_count")
    if catalog_checks["multi_market_components"] == 0:
        violations.append("multi_market_coverage")
    if catalog_checks["neg_risk_components"] == 0:
        violations.append("neg_risk_coverage")
    if int(episode_receipt["rows"]) != catalog_checks["component_count"]:
        violations.append("episode_component_count")

    stream_rows = int(stream["rows"])
    decision_stream_rows = int(decisions["stream_rows"])
    if stream.get("globally_ordered") is not True:
        violations.append("stream_global_order")
    if stream_rows != decision_stream_rows:
        violations.append("decision_stream_coverage")
    if stream.get("participants", {}).get("hard_wallet_count_cap", "missing") is not None:
        violations.append("wallet_cap")
    previous_last: tuple[int, str] | None = None
    for shard in stream["shards"]:
        first_raw = shard.get("first_key")
        last_raw = shard.get("last_key")
        first = None if first_raw is None else (int(first_raw[0]), str(first_raw[1]))
        last = None if last_raw is None else (int(last_raw[0]), str(last_raw[1]))
        if previous_last is not None and first is not None and first < previous_last:
            violations.append("stream_shard_boundary_order")
            break
        if last is not None:
            previous_last = last
    full_scope = bool(stream.get("full_scope_set"))
    if require_full and not full_scope:
        violations.append("full_ledger_scope")
    if full_scope and stream_rows != int(config["acceptance"]["ledger_rows_preserved_exactly"]):
        violations.append("registered_ledger_row_count")
    polygon_complete = bool(polygon and polygon.get("complete") is True)
    if require_polygon and not polygon_complete:
        violations.append("polygon_graph_complete")
    if bool(decisions.get("polygon_availability")) != polygon_complete:
        violations.append("decision_polygon_availability")

    hash_checks: dict[str, bool] = {}
    if deep_hashes:
        hash_checks["catalog"] = (
            sha256_file(output_dir / "catalog.sqlite") == catalog_receipt["sha256"]
        )
        hash_checks["episodes"] = (
            sha256_file(output_dir / "episodes.jsonl.zst") == episode_receipt["sha256"]
        )
        hash_checks["participants"] = _hash_matches(output_dir, stream["participants"])
        hash_checks["stream"] = all(_hash_matches(output_dir, shard) for shard in stream["shards"])
        hash_checks["decisions"] = all(
            _hash_matches(output_dir, shard) for shard in decisions["shards"]
        )
        if polygon_complete and polygon is not None:
            hash_checks["polygon"] = all(
                _hash_matches(output_dir, shard) for shard in polygon["consolidation"]["shards"]
            )
        if not all(hash_checks.values()):
            violations.append("artifact_hash")

    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_validation_receipt",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "valid": not violations,
        "require_full": require_full,
        "require_polygon": require_polygon,
        "deep_hashes": deep_hashes,
        "violations": violations,
        "catalog": catalog_checks,
        "stream_rows": stream_rows,
        "decision_rows": int(decisions["rows"]),
        "participant_rows": int(stream["participants"]["rows"]),
        "polygon_complete": polygon_complete,
        "hash_checks": hash_checks,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "validation-receipt.json", receipt)
    if violations:
        raise RuntimeError("H009 validation failed: " + ", ".join(violations))
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--require-full", action="store_true")
    value.add_argument("--require-polygon", action="store_true")
    value.add_argument("--deep-hashes", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    result = validate(
        args.config.resolve(),
        args.output_dir.resolve(),
        require_full=args.require_full,
        require_polygon=args.require_polygon,
        deep_hashes=args.deep_hashes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
