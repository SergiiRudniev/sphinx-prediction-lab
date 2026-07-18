from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.train_h014_replay_state_policy import (
    StateShard,
    _batch,
    _module_digest,
)


def _shard(tmp_path: Path) -> StateShard:
    state = tmp_path / "state"
    encoding = tmp_path / "encoding"
    pack = tmp_path / "pack"
    state.mkdir()
    encoding.mkdir()
    pack.mkdir()
    np.save(state / "row_indices.npy", np.asarray([3, 7], dtype=np.int64))
    np.save(state / "encoding_offsets.npy", np.asarray([1, 0], dtype=np.int64))
    np.save(state / "component_ids.npy", np.asarray([11, 12], dtype=np.int64))
    np.save(state / "portfolio_features.npy", np.zeros((2, 9), dtype=np.float32))
    np.save(state / "prediction_memory_features.npy", np.zeros((2, 7), dtype=np.float32))
    np.save(state / "previous_action_ids.npy", np.asarray([2, 1], dtype=np.int64))
    np.save(state / "physical_action_masks.npy", np.ones((2, 7), dtype=np.uint8))
    np.save(encoding / "row_indices.npy", np.asarray([7, 3], dtype=np.int64))
    np.save(encoding / "market_latents.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    np.save(encoding / "terminal_logits.npy", np.asarray([0.2, 0.4], dtype=np.float32))
    np.save(
        encoding / "uncertainty_log_scales.npy", np.asarray([-0.2, -0.4], dtype=np.float32)
    )
    labels = np.zeros(8, dtype=np.float32)
    labels[[3, 7]] = [1.0, 0.0]
    baselines = np.zeros(8, dtype=np.float32)
    baselines[[3, 7]] = [0.6, 0.2]
    np.save(pack / "labels.npy", labels)
    np.save(pack / "baselines.npy", baselines)
    return StateShard("2026-01-01", state, encoding, pack, 2)


def test_h014_batch_joins_state_encoding_and_terminal_target_by_source_row(
    tmp_path: Path,
) -> None:
    shard = _shard(tmp_path)

    batch = _batch(shard, np.asarray([0, 1], dtype=np.int64))

    assert batch.market_latents.tolist() == [
        [4.0, 5.0, 6.0, 7.0],
        [0.0, 1.0, 2.0, 3.0],
    ]
    assert batch.labels.tolist() == [1.0, 0.0]
    assert batch.baselines.tolist() == pytest.approx([0.6, 0.2])
    assert batch.component_ids.tolist() == [11, 12]


def test_h014_batch_rejects_encoding_row_drift(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    np.save(shard.encoding / "row_indices.npy", np.asarray([7, 4], dtype=np.int64))

    with pytest.raises(RuntimeError, match="no longer align"):
        _batch(shard, np.asarray([0], dtype=np.int64))


def test_module_digest_changes_with_a_parameter() -> None:
    module = torch.nn.Linear(2, 2)
    before = _module_digest(module)
    with torch.no_grad():
        module.weight[0, 0] += 1.0

    assert _module_digest(module) != before
