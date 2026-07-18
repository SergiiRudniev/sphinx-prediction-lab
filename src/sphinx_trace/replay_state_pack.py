"""Source-bound causal policy states extracted from an exact H010 replay."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import sha256_file
from sphinx_trace.model_h012 import (
    H012_ACTION_COUNT,
    H012_MEMORY_NUMERIC_WIDTH,
    H012_PORTFOLIO_WIDTH,
)

STATE_ARRAY_NAMES = (
    "row_indices.npy",
    "encoding_offsets.npy",
    "component_ids.npy",
    "timestamps.npy",
    "partition_codes.npy",
    "portfolio_features.npy",
    "prediction_memory_features.npy",
    "previous_action_ids.npy",
    "physical_action_masks.npy",
)


@dataclass(frozen=True, slots=True)
class ReplayStateArrays:
    row_indices: NDArray[np.int64]
    portfolio_features: NDArray[np.float32]
    prediction_memory_features: NDArray[np.float32]
    previous_action_ids: NDArray[np.int64]
    physical_action_masks: NDArray[np.uint8]


def extract_replay_state_arrays(
    records: Iterable[dict[str, Any]],
    *,
    date: str,
    expected_row_indices: NDArray[np.int64],
    expected_timestamps: NDArray[np.int64],
) -> ReplayStateArrays:
    """Extract each expected decision exactly once without learning its teacher action."""

    if expected_row_indices.ndim != 1 or expected_timestamps.shape != expected_row_indices.shape:
        raise ValueError("H014 expected replay rows and timestamps must be aligned vectors")
    if len(np.unique(expected_row_indices)) != len(expected_row_indices):
        raise ValueError("H014 expected replay rows must be unique")
    offsets = {int(row): offset for offset, row in enumerate(expected_row_indices.tolist())}
    rows = len(expected_row_indices)
    portfolio = np.empty((rows, H012_PORTFOLIO_WIDTH), dtype=np.float32)
    memory = np.empty((rows, H012_MEMORY_NUMERIC_WIDTH), dtype=np.float32)
    previous = np.empty(rows, dtype=np.int64)
    physical = np.empty((rows, H012_ACTION_COUNT), dtype=np.uint8)
    seen = np.zeros(rows, dtype=np.uint8)
    decisions = 0
    for record in records:
        if record.get("record_type") != "h010_decision_audit":
            continue
        decisions += 1
        feature_ref = record.get("feature_ref")
        if not isinstance(feature_ref, dict) or feature_ref.get("date") != date:
            raise RuntimeError("H014 replay decision has an invalid feature reference")
        row = int(feature_ref.get("row", -1))
        offset = offsets.get(row)
        if offset is None:
            raise RuntimeError(
                f"H014 replay decision is outside the expected source rows: {date}:{row}"
            )
        if seen[offset]:
            raise RuntimeError(f"H014 replay decision repeats: {date}:{row}")
        timestamp = int(record.get("timestamp_unix", -1))
        if timestamp != int(expected_timestamps[offset]):
            raise RuntimeError(f"H014 replay decision timestamp changed: {date}:{row}")
        portfolio_row = np.asarray(record.get("portfolio_features"), dtype=np.float32)
        memory_row = np.asarray(record.get("prediction_memory_features"), dtype=np.float32)
        physical_row = np.asarray(record.get("physical_action_mask"), dtype=np.uint8)
        previous_action = int(record.get("previous_action_id", -1))
        if portfolio_row.shape != (H012_PORTFOLIO_WIDTH,):
            raise RuntimeError(f"H014 portfolio state width changed: {date}:{row}")
        if memory_row.shape != (H012_MEMORY_NUMERIC_WIDTH,):
            raise RuntimeError(f"H014 prediction memory width changed: {date}:{row}")
        if physical_row.shape != (H012_ACTION_COUNT,) or not bool(physical_row.any()):
            raise RuntimeError(f"H014 physical action mask changed: {date}:{row}")
        if previous_action < 0 or previous_action >= H012_ACTION_COUNT:
            raise RuntimeError(f"H014 previous action is invalid: {date}:{row}")
        if not bool(np.isfinite(portfolio_row).all()) or not bool(np.isfinite(memory_row).all()):
            raise RuntimeError(f"H014 replay state contains non-finite values: {date}:{row}")
        portfolio[offset] = portfolio_row
        memory[offset] = memory_row
        previous[offset] = previous_action
        physical[offset] = physical_row
        seen[offset] = 1
    if decisions != rows or not bool(seen.all()):
        missing = expected_row_indices[seen == 0]
        example = int(missing[0]) if len(missing) else None
        raise RuntimeError(
            f"H014 replay decisions do not cover {date}: expected={rows}, "
            f"observed={decisions}, first_missing={example}"
        )
    return ReplayStateArrays(
        row_indices=np.asarray(expected_row_indices, dtype=np.int64),
        portfolio_features=portfolio,
        prediction_memory_features=memory,
        previous_action_ids=previous,
        physical_action_masks=physical,
    )


def atomic_numpy(path: Path, values: NDArray[Any]) -> None:
    """Durably replace one NumPy array and tolerate transient Windows reader locks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.025 * (2**attempt)))
    raise PermissionError(f"Could not atomically replace {path}: {last_error}")


def array_metadata(path: Path, root: Path) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def validate_state_shard(
    shard_dir: Path,
    files: dict[str, Any],
    *,
    expected_rows: int,
) -> None:
    """Verify receipt-bound state arrays before a resumable build reuses them."""

    for name in STATE_ARRAY_NAMES:
        metadata = files.get(name)
        path = shard_dir / name
        if not isinstance(metadata, dict) or not path.is_file():
            raise RuntimeError(f"H014 state shard is incomplete: {path}")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"H014 state shard size changed: {path}")
        if metadata.get("sha256") != sha256_file(path):
            raise RuntimeError(f"H014 state shard hash changed: {path}")
    rows = np.load(shard_dir / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    encodings = np.load(
        shard_dir / "encoding_offsets.npy", mmap_mode="r", allow_pickle=False
    )
    components = np.load(shard_dir / "component_ids.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(shard_dir / "timestamps.npy", mmap_mode="r", allow_pickle=False)
    partitions = np.load(
        shard_dir / "partition_codes.npy", mmap_mode="r", allow_pickle=False
    )
    portfolio = np.load(
        shard_dir / "portfolio_features.npy", mmap_mode="r", allow_pickle=False
    )
    memory = np.load(
        shard_dir / "prediction_memory_features.npy", mmap_mode="r", allow_pickle=False
    )
    previous = np.load(
        shard_dir / "previous_action_ids.npy", mmap_mode="r", allow_pickle=False
    )
    physical = np.load(
        shard_dir / "physical_action_masks.npy", mmap_mode="r", allow_pickle=False
    )
    if not (
        rows.shape
        == encodings.shape
        == components.shape
        == timestamps.shape
        == partitions.shape
        == previous.shape
        == (expected_rows,)
        and portfolio.shape == (expected_rows, H012_PORTFOLIO_WIDTH)
        and memory.shape == (expected_rows, H012_MEMORY_NUMERIC_WIDTH)
        and physical.shape == (expected_rows, H012_ACTION_COUNT)
    ):
        raise RuntimeError(f"H014 state shard arrays do not align: {shard_dir}")
    if rows.dtype != np.int64 or encodings.dtype != np.int64:
        raise RuntimeError(f"H014 state row index dtypes changed: {shard_dir}")
    if partitions.dtype != np.uint8 or not bool(np.isin(partitions, (0, 1)).all()):
        raise RuntimeError(f"H014 state partition codes changed: {shard_dir}")
