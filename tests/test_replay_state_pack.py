from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from sphinx_trace.replay_state_pack import (
    array_metadata,
    atomic_numpy,
    extract_replay_state_arrays,
    validate_state_shard,
)


def _decision(row: int, timestamp: int) -> dict[str, Any]:
    return {
        "record_type": "h010_decision_audit",
        "feature_ref": {"date": "2026-01-01", "row": row},
        "timestamp_unix": timestamp,
        "portfolio_features": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "prediction_memory_features": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "previous_action_id": 2,
        "physical_action_mask": [True, True, True, False, False, False, False],
        "action": "CALL_OUTCOME_1",
    }


def test_extracts_exact_states_without_teacher_action() -> None:
    records: list[dict[str, Any]] = [
        {"record_type": "h010_order_audit"},
        _decision(7, 101),
        _decision(3, 100),
    ]
    state = extract_replay_state_arrays(
        records,
        date="2026-01-01",
        expected_row_indices=np.asarray([3, 7], dtype=np.int64),
        expected_timestamps=np.asarray([100, 101], dtype=np.int64),
    )

    assert state.row_indices.tolist() == [3, 7]
    assert state.portfolio_features.shape == (2, 9)
    assert state.prediction_memory_features.shape == (2, 7)
    assert state.previous_action_ids.tolist() == [2, 2]
    assert state.physical_action_masks.shape == (2, 7)
    assert not hasattr(state, "action")


def test_rejects_missing_or_duplicate_decisions() -> None:
    with pytest.raises(RuntimeError, match="do not cover"):
        extract_replay_state_arrays(
            [_decision(3, 100)],
            date="2026-01-01",
            expected_row_indices=np.asarray([3, 7], dtype=np.int64),
            expected_timestamps=np.asarray([100, 101], dtype=np.int64),
        )
    with pytest.raises(RuntimeError, match="repeats"):
        extract_replay_state_arrays(
            [_decision(3, 100), _decision(3, 100)],
            date="2026-01-01",
            expected_row_indices=np.asarray([3], dtype=np.int64),
            expected_timestamps=np.asarray([100], dtype=np.int64),
        )


def test_validates_source_bound_state_arrays(tmp_path: Path) -> None:
    arrays: dict[str, NDArray[Any]] = {
        "row_indices.npy": np.asarray([3], dtype=np.int64),
        "encoding_offsets.npy": np.asarray([0], dtype=np.int64),
        "component_ids.npy": np.asarray([4], dtype=np.int64),
        "timestamps.npy": np.asarray([100], dtype=np.int64),
        "partition_codes.npy": np.asarray([0], dtype=np.uint8),
        "portfolio_features.npy": np.zeros((1, 9), dtype=np.float32),
        "prediction_memory_features.npy": np.zeros((1, 7), dtype=np.float32),
        "previous_action_ids.npy": np.asarray([2], dtype=np.int64),
        "physical_action_masks.npy": np.ones((1, 7), dtype=np.uint8),
    }
    for name, values in arrays.items():
        atomic_numpy(tmp_path / name, values)
    files = {name: array_metadata(tmp_path / name, tmp_path) for name in arrays}

    validate_state_shard(tmp_path, files, expected_rows=1)

    (tmp_path / "row_indices.npy").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size changed"):
        validate_state_shard(tmp_path, files, expected_rows=1)
