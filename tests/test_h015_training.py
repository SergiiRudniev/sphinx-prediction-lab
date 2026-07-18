from pathlib import Path

import numpy as np
import pytest
from scripts.train_h015_portfolio_advantage import (
    StateShard,
    _batch,
    _equal_market_weights,
    _sample_weights,
)


def _shard(tmp_path: Path, behavior_code: int = 0) -> StateShard:
    state = tmp_path / f"state-{behavior_code}"
    encoding = tmp_path / "encoding"
    pack = tmp_path / "pack"
    state.mkdir()
    encoding.mkdir(exist_ok=True)
    pack.mkdir(exist_ok=True)
    np.save(state / "row_indices.npy", np.asarray([3, 7], dtype=np.int64))
    np.save(state / "encoding_offsets.npy", np.asarray([1, 0], dtype=np.int64))
    np.save(state / "market_ids.npy", np.asarray([11, 12], dtype=np.int64))
    np.save(
        state / "behavior_policy_codes.npy",
        np.full(2, behavior_code, dtype=np.uint8),
    )
    np.save(state / "partition_codes.npy", np.zeros(2, dtype=np.uint8))
    np.save(state / "portfolio_features.npy", np.zeros((2, 9), dtype=np.float32))
    np.save(
        state / "prediction_memory_features.npy", np.zeros((2, 7), dtype=np.float32)
    )
    np.save(state / "previous_action_ids.npy", np.asarray([2, 1], dtype=np.int64))
    np.save(state / "physical_action_masks.npy", np.ones((2, 7), dtype=np.uint8))
    np.save(state / "behavior_action_ids.npy", np.asarray([2, 1], dtype=np.int64))
    np.save(
        state / "realized_action_values.npy", np.asarray([0.0, 0.03], dtype=np.float32)
    )
    np.save(
        state / "execution_fractions.npy", np.asarray([0.0, 0.5], dtype=np.float32)
    )
    np.save(encoding / "row_indices.npy", np.asarray([7, 3], dtype=np.int64))
    np.save(
        encoding / "market_latents.npy", np.arange(8, dtype=np.float32).reshape(2, 4)
    )
    np.save(encoding / "terminal_logits.npy", np.asarray([0.2, 0.4], dtype=np.float32))
    np.save(
        encoding / "uncertainty_log_scales.npy",
        np.asarray([-0.2, -0.4], dtype=np.float32),
    )
    labels = np.zeros(8, dtype=np.float32)
    labels[[3, 7]] = [1.0, 0.0]
    baselines = np.zeros(8, dtype=np.float32)
    baselines[[3, 7]] = [0.6, 0.2]
    np.save(pack / "labels.npy", labels)
    np.save(pack / "baselines.npy", baselines)
    return StateShard(
        f"behavior-{behavior_code}",
        behavior_code,
        "2026-01-01",
        state,
        encoding,
        pack,
        2,
    )


def test_h015_batch_joins_state_encoding_terminal_and_logged_targets(
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
    assert batch.market_ids.tolist() == [11, 12]
    assert batch.behavior_action_ids.tolist() == [2, 1]
    assert batch.realized_action_values.tolist() == pytest.approx([0.0, 0.03])
    assert batch.execution_fractions.tolist() == pytest.approx([0.0, 0.5])


def test_h015_equal_market_weights_give_every_market_same_total_per_behavior(
    tmp_path: Path,
) -> None:
    shards = [_shard(tmp_path, 0), _shard(tmp_path, 1)]
    for shard in shards:
        np.save(shard.state / "market_ids.npy", np.asarray([1, 1], dtype=np.int64))
    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra = _shard(extra_root, 0)
    np.save(extra.state / "market_ids.npy", np.asarray([2, 2], dtype=np.int64))
    extra_one_root = tmp_path / "extra-one"
    extra_one_root.mkdir()
    extra_one = _shard(extra_one_root, 1)
    np.save(extra_one.state / "market_ids.npy", np.asarray([2, 2], dtype=np.int64))
    shards.extend([extra, extra_one])
    for behavior_code in (0, 1):
        selection_root = tmp_path / f"selection-{behavior_code}"
        selection_root.mkdir()
        selection = _shard(selection_root, behavior_code)
        np.save(selection.state / "market_ids.npy", np.asarray([3, 3], dtype=np.int64))
        np.save(selection.state / "partition_codes.npy", np.ones(2, dtype=np.uint8))
        shards.append(selection)

    weights, receipt = _equal_market_weights(shards, 2)
    batch = _batch(shards[0], np.asarray([0, 1], dtype=np.int64))

    assert receipt["fit_markets_by_behavior"] == [2, 2]
    assert receipt["fit_rows_by_behavior"] == [4, 4]
    assert receipt["selection_markets_by_behavior"] == [1, 1]
    assert receipt["selection_rows_by_behavior"] == [2, 2]
    assert weights[0, 0, 1] * 2 == pytest.approx(weights[0, 0, 2] * 2)
    assert weights[0, 1, 1] * 2 == pytest.approx(weights[0, 1, 2] * 2)
    assert _sample_weights(batch, weights, 0).tolist() == pytest.approx(
        [weights[0, 0, 1], weights[0, 0, 1]]
    )


def test_h015_batch_rejects_encoding_row_drift(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    np.save(shard.encoding / "row_indices.npy", np.asarray([7, 4], dtype=np.int64))

    with pytest.raises(RuntimeError, match="no longer align"):
        _batch(shard, np.asarray([0], dtype=np.int64))


def test_h017_batch_loads_protocol_targets_when_present(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    np.save(
        shard.state / "winning_payout_multipliers.npy",
        np.asarray([[1.5, 2.0], [4.0, 1.2]], dtype=np.float32),
    )
    np.save(
        shard.state / "reference_action_values.npy",
        np.asarray([[0.02, -0.05, 0.0], [-0.05, 0.01, 0.0]], dtype=np.float32),
    )

    batch = _batch(shard, np.asarray([1, 0], dtype=np.int64))

    assert batch.winning_payout_multipliers is not None
    assert batch.reference_action_values is not None
    assert np.allclose(
        batch.winning_payout_multipliers,
        np.asarray([[4.0, 1.2], [1.5, 2.0]], dtype=np.float32),
    )
    assert batch.reference_action_values[:, 2].tolist() == [0.0, 0.0]
