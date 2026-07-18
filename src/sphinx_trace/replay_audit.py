"""Atomic, source-bound audit shards for H010/H012 development replay."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, now_utc, sha256_file, write_jsonl_zst

AUDIT_RECORD_TYPES = frozenset(
    {
        "h010_decision_audit",
        "h010_order_audit",
        "h010_fill_audit",
        "h010_resolution_audit",
    }
)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def decision_audit_record(
    *,
    decision_id: str,
    timestamp_unix: int,
    condition_id: str,
    component_id: str,
    evidence_trade_id: str,
    feature_date: str,
    feature_row: int,
    input_sha256: str,
    action: str,
    probability_outcome0: float,
    size_fraction: float,
    physical_action_mask: Sequence[bool],
    portfolio_features: Sequence[float],
    prediction_memory_features: Sequence[float],
    previous_action_id: int,
    action_logits: Sequence[float],
    outcome_labels: Sequence[str],
) -> dict[str, Any]:
    """Create one compact record that can recover the exact model input by reference."""

    _require_sha256(input_sha256, "input_sha256")
    if timestamp_unix < 0 or feature_row < 0 or previous_action_id < 0:
        raise ValueError("Decision audit ordinals must be non-negative")
    if not 0.0 <= probability_outcome0 <= 1.0 or not 0.0 <= size_fraction <= 1.0:
        raise ValueError("Decision audit probability and size must be bounded")
    if len(portfolio_features) != 9 or len(prediction_memory_features) != 7:
        raise ValueError("Decision audit state widths do not match H012")
    if len(physical_action_mask) != len(action_logits):
        raise ValueError("Decision audit action mask and logits must align")
    if len(outcome_labels) != 2:
        raise ValueError("Decision audit requires two catalog outcome labels")
    return {
        "schema_version": "1.0.0",
        "record_type": "h010_decision_audit",
        "decision_id": decision_id,
        "timestamp_unix": timestamp_unix,
        "condition_id": condition_id,
        "component_id": component_id,
        "evidence_trade_id": evidence_trade_id,
        "feature_ref": {"date": feature_date, "row": feature_row},
        "input_sha256": input_sha256,
        "action": action,
        "probability_outcome0": probability_outcome0,
        "size_fraction": size_fraction,
        "physical_action_mask": list(physical_action_mask),
        "portfolio_features": list(portfolio_features),
        "prediction_memory_features": list(prediction_memory_features),
        "previous_action_id": previous_action_id,
        "action_logits": list(action_logits),
        "outcome_labels": list(outcome_labels),
    }


def write_audit_shard(
    output_dir: Path,
    date: str,
    records: Iterable[dict[str, Any]],
    *,
    source_sha256: str,
    policy_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    """Write an immutable daily audit shard and its validation receipt."""

    for field, value in (
        ("source_sha256", source_sha256),
        ("policy_sha256", policy_sha256),
        ("implementation_sha256", implementation_sha256),
    ):
        _require_sha256(value, field)
    shard_path = output_dir / "shards" / f"date={date}.jsonl.zst"
    receipt_path = output_dir / "receipts" / f"date={date}.json"
    expected_contract = {
        "date": date,
        "source_sha256": source_sha256,
        "policy_sha256": policy_sha256,
        "implementation_sha256": implementation_sha256,
    }
    if receipt_path.exists() or shard_path.exists():
        if not receipt_path.exists() or not shard_path.exists():
            raise RuntimeError(f"Incomplete audit shard exists for {date}")
        cached_receipt = _load_object(receipt_path)
        if any(cached_receipt.get(key) != value for key, value in expected_contract.items()):
            raise RuntimeError(f"Audit shard contract changed for {date}")
        if cached_receipt.get("sha256") != sha256_file(shard_path):
            raise RuntimeError(f"Audit shard digest changed for {date}")
        return cached_receipt

    record_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    minimum_timestamp: int | None = None
    maximum_timestamp: int | None = None
    decision_ids: set[str] = set()

    def validated() -> Iterator[dict[str, Any]]:
        nonlocal minimum_timestamp, maximum_timestamp
        previous_timestamp = -1
        for record in records:
            record_type = str(record.get("record_type") or "")
            if record_type not in AUDIT_RECORD_TYPES:
                raise ValueError(f"Unsupported replay audit record: {record_type}")
            timestamp = int(record["timestamp_unix"])
            if timestamp < previous_timestamp:
                raise ValueError("Replay audit time regressed within a shard")
            previous_timestamp = timestamp
            minimum_timestamp = (
                timestamp if minimum_timestamp is None else min(minimum_timestamp, timestamp)
            )
            maximum_timestamp = (
                timestamp if maximum_timestamp is None else max(maximum_timestamp, timestamp)
            )
            if record_type == "h010_decision_audit":
                decision_id = str(record["decision_id"])
                if decision_id in decision_ids:
                    raise ValueError(f"Decision audit was emitted twice: {decision_id}")
                decision_ids.add(decision_id)
                action_counts[str(record["action"])] += 1
            record_counts[record_type] += 1
            yield record

    rows, size = write_jsonl_zst(shard_path, validated())
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h010_replay_audit_shard_receipt",
        "generated_at": now_utc(),
        **expected_contract,
        "path": shard_path.relative_to(output_dir).as_posix(),
        "rows": rows,
        "bytes": size,
        "sha256": sha256_file(shard_path),
        "minimum_timestamp_unix": minimum_timestamp,
        "maximum_timestamp_unix": maximum_timestamp,
        "record_counts": dict(sorted(record_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build_audit_manifest(
    output_dir: Path,
    *,
    source_sha256: str,
    policy_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    """Verify every receipt and bind their ordered digests into one manifest."""

    receipts: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for receipt_path in sorted((output_dir / "receipts").glob("date=*.json")):
        receipt = _load_object(receipt_path)
        date = str(receipt["date"])
        shard_path = output_dir / str(receipt["path"])
        if any(
            receipt.get(key) != value
            for key, value in {
                "source_sha256": source_sha256,
                "policy_sha256": policy_sha256,
                "implementation_sha256": implementation_sha256,
            }.items()
        ):
            raise RuntimeError(f"Audit manifest contract changed at {date}")
        if not shard_path.exists() or receipt.get("sha256") != sha256_file(shard_path):
            raise RuntimeError(f"Audit manifest shard changed at {date}")
        digest.update(f"{date}:{receipt['sha256']}\n".encode())
        receipts.append(receipt)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h010_replay_audit_manifest",
        "generated_at": now_utc(),
        "valid": True,
        "source_sha256": source_sha256,
        "policy_sha256": policy_sha256,
        "implementation_sha256": implementation_sha256,
        "days": len(receipts),
        "rows": sum(int(receipt["rows"]) for receipt in receipts),
        "shard_digest": digest.hexdigest(),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
