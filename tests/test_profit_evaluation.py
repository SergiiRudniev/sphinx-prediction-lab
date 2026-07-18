from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scripts.evaluate_h010_profit import _verify_replay_binding

from sphinx_corpus.io import atomic_json, sha256_file, write_jsonl_zst
from sphinx_trace.profit_evaluation import (
    independent_component_bootstrap,
    moving_block_bootstrap_mean,
    promotion_gates,
)


def test_replay_binding_detects_changed_audit_shard(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    receipts = tmp_path / "receipts"
    shards.mkdir()
    receipts.mkdir()
    shard_path = shards / "date=2026-01-01.jsonl.zst"
    write_jsonl_zst(shard_path, [{"record_type": "test"}])
    contract = {
        "source_sha256": "1" * 64,
        "policy_sha256": "2" * 64,
        "implementation_sha256": "3" * 64,
    }
    receipt = {
        "date": "2026-01-01",
        "path": "shards/date=2026-01-01.jsonl.zst",
        "rows": 1,
        "sha256": sha256_file(shard_path),
        **contract,
    }
    receipt_path = receipts / "date=2026-01-01.json"
    atomic_json(receipt_path, receipt)
    shard_digest = hashlib.sha256(f"2026-01-01:{receipt['sha256']}\n".encode()).hexdigest()
    audit = {"valid": True, "days": 1, "rows": 1, "shard_digest": shard_digest, **contract}
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, audit)
    replay = {"audit_manifest_sha256": sha256_file(manifest_path), **contract}

    _verify_replay_binding(tmp_path, replay, audit)
    write_jsonl_zst(shard_path, [{"record_type": "changed"}])
    with pytest.raises(RuntimeError, match="audit shard changed"):
        _verify_replay_binding(tmp_path, replay, audit)


def test_positive_profit_bootstraps_pass_registered_gates() -> None:
    weekly = moving_block_bootstrap_mean(
        np.linspace(10.0, 30.0, 20), replicates=500, block_length=4, seed=1
    )
    components = independent_component_bootstrap(np.linspace(1.0, 3.0, 200), replicates=500, seed=2)
    gates = promotion_gates(
        weekly,
        components,
        minimum_calls=100,
        minimum_components=100,
        calls=200,
    )

    assert weekly["lower"] > 0
    assert components["lower_mean_profit_usd"] > 0
    assert gates["all_pass"] is True


def test_loss_or_insufficient_breadth_fails_profit_gates() -> None:
    weekly = moving_block_bootstrap_mean(
        np.array([-2.0, 1.0, -3.0, 1.0]), replicates=500, block_length=2, seed=3
    )
    components = independent_component_bootstrap(np.array([-1.0, 0.5]), replicates=500, seed=4)
    gates = promotion_gates(
        weekly,
        components,
        minimum_calls=100,
        minimum_components=100,
        calls=2,
    )

    assert gates["all_pass"] is False
    assert gates["minimum_calls"] is False
    assert gates["minimum_components"] is False
