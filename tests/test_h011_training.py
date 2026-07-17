from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.train_h011_outcome import (
    PackShard,
    _balanced_weights,
    _batch,
    _loss,
    _metrics,
    _row_indices,
)

from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]


def _shard(root: Path) -> PackShard:
    path = root / "shards" / "date=2026-01-01"
    path.mkdir(parents=True)
    features = np.zeros((4, 128), dtype=np.float32)
    features[:, 46] = np.asarray([0.1, 0.9, 0.5, 0.5])
    np.save(path / "features.npy", features)
    np.save(path / "labels.npy", np.asarray([1, 0, 1, 0], dtype=np.float32))
    np.save(path / "label_mask.npy", np.asarray([1, 1, 1, 0], dtype=np.uint8))
    np.save(path / "split_codes.npy", np.asarray([1, 1, 2, 4], dtype=np.uint8))
    np.save(path / "baselines.npy", np.asarray([0.6, 0.4, 0.7, 0.5], dtype=np.float32))
    np.save(path / "component_ids.npy", np.asarray([0, 1, 0, 1], dtype=np.int32))
    return PackShard("2026-01-01", path)


def test_training_selection_never_consumes_unlabeled_test_rows(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    train = _row_indices(shard, 1, seed=17, epoch=0, shuffle=False)
    test = _row_indices(shard, 4, seed=17, epoch=0, shuffle=False)
    assert train.tolist() == [0, 1]
    assert test.tolist() == []


def test_balance_and_batch_use_component_and_raw_lifecycle(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_train_v1.json")
    output = tmp_path / "output"
    balance = _balanced_weights(tmp_path, [shard], output, config, "source")
    features, labels, baselines, weights = _batch(
        shard,
        np.asarray([0, 1], dtype=np.int64),
        np.zeros(128, dtype=np.float32),
        np.ones(128, dtype=np.float32),
        12.0,
        balance,
        10,
        (0.25, 4.0),
    )
    assert features.shape == (2, 128)
    assert labels.tolist() == [1.0, 0.0]
    assert baselines.tolist() == pytest.approx([0.6, 0.4])
    assert np.isfinite(weights).all()


def test_outcome_loss_and_metrics_are_finite() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_train_v1.json")
    output = {
        "terminal_outcome_logit": torch.tensor([1.0, -1.0]),
        "expected_net_edge": torch.tensor([0.2, -0.2]),
        "uncertainty_log_scale": torch.zeros(2),
    }
    loss, parts = _loss(
        output,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.6, 0.4]),
        torch.ones(2),
        config,
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in parts.values())
    metrics = _metrics(np.asarray([0.8, 0.2]), np.asarray([1.0, 0.0]))
    assert metrics["log_loss"] == pytest.approx(-np.log(0.8))
    assert metrics["brier"] == pytest.approx(0.04)
