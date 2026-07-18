from __future__ import annotations

from pathlib import Path

import pytest

from sphinx_trace.replay_audit import (
    build_audit_manifest,
    decision_audit_record,
    write_audit_shard,
)


def _record(decision_id: str, timestamp: int) -> dict[str, object]:
    return decision_audit_record(
        decision_id=decision_id,
        timestamp_unix=timestamp,
        condition_id="condition",
        component_id="component",
        evidence_trade_id=f"trade-{decision_id}",
        feature_date="2026-03-01",
        feature_row=timestamp,
        input_sha256="ab" * 32,
        action="SKIP",
        probability_outcome0=0.5,
        size_fraction=0.0,
        physical_action_mask=[True] * 7,
        portfolio_features=[0.0] * 9,
        prediction_memory_features=[0.0] * 7,
        previous_action_id=2,
        action_logits=[0.0] * 7,
        outcome_labels=["Yes", "No"],
    )


def test_atomic_audit_shards_are_cached_and_manifest_bound(tmp_path: Path) -> None:
    contract = {
        "source_sha256": "cd" * 32,
        "policy_sha256": "ef" * 32,
        "implementation_sha256": "12" * 32,
    }
    receipt = write_audit_shard(
        tmp_path,
        "2026-03-01",
        [_record("a", 1), _record("b", 2)],
        **contract,
    )
    assert receipt["rows"] == 2
    assert receipt["action_counts"] == {"SKIP": 2}
    cached = write_audit_shard(tmp_path, "2026-03-01", [], **contract)
    assert cached["sha256"] == receipt["sha256"]
    manifest = build_audit_manifest(tmp_path, **contract)
    assert manifest["valid"] is True
    assert manifest["days"] == 1
    assert manifest["rows"] == 2


def test_audit_rejects_time_regression_and_contract_change(tmp_path: Path) -> None:
    contract = {
        "source_sha256": "cd" * 32,
        "policy_sha256": "ef" * 32,
        "implementation_sha256": "12" * 32,
    }
    with pytest.raises(ValueError, match="time regressed"):
        write_audit_shard(
            tmp_path,
            "2026-03-01",
            [_record("a", 2), _record("b", 1)],
            **contract,
        )
    write_audit_shard(tmp_path, "2026-03-01", [_record("a", 1)], **contract)
    with pytest.raises(RuntimeError, match="contract changed"):
        write_audit_shard(
            tmp_path,
            "2026-03-01",
            [],
            source_sha256="34" * 32,
            policy_sha256=contract["policy_sha256"],
            implementation_sha256=contract["implementation_sha256"],
        )
