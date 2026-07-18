from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sphinx_corpus.io import sha256_file, write_jsonl_zst
from sphinx_trace.policy_decisions import (
    PolicyFeatureStore,
    load_policy_decisions,
    policy_input_digest,
)


def _pack(root: Path) -> Path:
    pack = root / "pack"
    shard = pack / "shards" / "date=2026-01-01"
    receipts = pack / "receipts"
    shard.mkdir(parents=True)
    receipts.mkdir()
    features = np.vstack((np.arange(128), np.arange(128) + 1)).astype(np.float32)
    np.save(shard / "features.npy", features)
    np.save(shard / "baselines.npy", np.array([0.65, 0.5], dtype=np.float32))
    np.save(shard / "split_codes.npy", np.array([2, 4], dtype=np.uint8))
    np.save(shard / "label_mask.npy", np.array([1, 0], dtype=np.uint8))
    np.save(shard / "timestamps.npy", np.array([10, 20], dtype=np.int64))
    np.save(shard / "market_ids.npy", np.array([3, 4], dtype=np.int32))
    np.save(shard / "component_ids.npy", np.array([5, 6], dtype=np.int32))
    write_jsonl_zst(
        shard / "examples.jsonl.zst",
        (
            {
                "decision_id": "decision",
                "evidence_trade_id": "trade",
                "decision_time_unix": 10,
                "condition_id": "CONDITION",
                "component_id": "component",
                "market_state_id": 3,
            },
            {
                "decision_id": "test",
                "evidence_trade_id": "test-trade",
                "decision_time_unix": 20,
                "condition_id": "test-condition",
                "component_id": "test-component",
                "market_state_id": 4,
                "component_state_id": 6,
            },
        ),
    )
    normalization = pack / "normalization.npz"
    np.savez(normalization, median=np.zeros(128), scale=np.ones(128))
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "valid": True,
                "test_labels_opened": False,
                "normalization": {"sha256": sha256_file(normalization)},
            }
        ),
        encoding="utf-8",
    )
    (receipts / "date=2026-01-01.json").write_text("{}", encoding="utf-8")
    return pack


def test_policy_decisions_keep_test_closed_and_bind_full_state(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    decisions, shards = load_policy_decisions(pack)
    ref = decisions["trade"][0]
    store = PolicyFeatureStore(pack, shards, feature_clip=12.0, cache_shards=1)

    loaded = store.load(ref)
    digest = policy_input_digest(
        loaded.feature_sha256,
        loaded.market_probability_outcome0,
        (1.0,) * 9,
        (0.0,) * 7,
        2,
        (True, True, True, False, True, False, False),
    )

    assert set(decisions) == {"trade"}
    assert ref.condition_id == "condition"
    assert ref.component_state_id == 5
    assert loaded.normalized[-1] == 12.0
    assert loaded.market_probability_outcome0 == pytest.approx(0.65)
    assert len(loaded.feature_sha256) == 64
    assert len(digest) == 64
    changed = policy_input_digest(
        loaded.feature_sha256,
        loaded.market_probability_outcome0,
        (0.9,) + (1.0,) * 8,
        (0.0,) * 7,
        2,
        (True, True, True, False, True, False, False),
    )
    assert changed != digest
