"""Bind H011 development predictions back to exact causal feature rows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import iter_jsonl_zst, sha256_file

DEVELOPMENT_SPLITS = frozenset({"validation", "calibration"})


@dataclass(frozen=True, slots=True)
class BoundPrediction:
    split: str
    feature_date: str
    feature_row: int
    decision_id: str
    evidence_trade_id: str
    timestamp_unix: int
    condition_id: str
    component_id: str
    market_state_id: int
    component_state_id: int
    probability_outcome0: float
    label_outcome0: float
    market_probability_outcome0: float
    feature_input_sha256: str


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def feature_digest(
    features: np.ndarray[Any, np.dtype[np.float32]],
    *,
    normalization_sha256: str,
    date: str,
    row: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"normalization:{normalization_sha256}\ndate:{date}\nrow:{row}\n".encode())
    digest.update(np.ascontiguousarray(features, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _pack_source_digest(pack_dir: Path, shards: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(pack_dir / "manifest.json").encode())
    digest.update(sha256_file(pack_dir / "normalization.npz").encode())
    for shard in shards:
        date = shard.name.removeprefix("date=")
        receipt_path = pack_dir / "receipts" / f"date={date}.json"
        digest.update(f"{date}:{sha256_file(receipt_path)}\n".encode())
    return digest.hexdigest()


def bind_development_predictions(
    pack_dir: Path,
    model_dir: Path,
    split: str,
) -> list[BoundPrediction]:
    """Return development predictions with exact H011 provenance; never open test."""

    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("Only validation and calibration predictions may be bound")
    pack_manifest = _load_object(pack_dir / "manifest.json")
    result = _load_object(model_dir / "result.json")
    if pack_manifest.get("valid") is not True or result.get("valid") is not True:
        raise RuntimeError("Prediction binding requires valid pack and model receipts")
    if (
        pack_manifest.get("test_labels_opened") is not False
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("Prediction binding encountered opened test evidence")
    normalization_sha256 = sha256_file(pack_dir / "normalization.npz")
    if pack_manifest.get("normalization", {}).get("sha256") != normalization_sha256:
        raise RuntimeError("H011 normalization digest changed")
    shards = sorted(path for path in (pack_dir / "shards").glob("date=*") if path.is_dir())
    if not shards:
        raise RuntimeError("Prediction binding found no H011 shards")
    if result.get("source_digest") != _pack_source_digest(pack_dir, shards):
        raise RuntimeError("H011 model result belongs to another feature pack")
    predictions_path = model_dir / "predictions.npz"
    if result.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError("H011 model prediction digest changed")

    prefix = f"{split}_"
    required = (
        "logits",
        "labels",
        "baselines",
        "shard_indices",
        "row_indices",
        "timestamps",
        "market_ids",
        "component_ids",
    )
    with np.load(predictions_path, allow_pickle=False) as archive:
        if any(key.startswith("test_") for key in archive.files):
            raise RuntimeError("Prediction artifact contains forbidden test arrays")
        arrays = {name: np.asarray(archive[prefix + name]) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise RuntimeError("H011 prediction metadata arrays have different lengths")

    grouped: dict[int, list[tuple[int, int]]] = {}
    for output_index, (shard_index, row_index) in enumerate(
        zip(arrays["shard_indices"], arrays["row_indices"], strict=True)
    ):
        shard_value = int(shard_index)
        row_value = int(row_index)
        if not 0 <= shard_value < len(shards) or row_value < 0:
            raise RuntimeError("H011 prediction has an invalid feature reference")
        grouped.setdefault(shard_value, []).append((row_value, output_index))

    bound: list[BoundPrediction] = []
    for shard_index, references in sorted(grouped.items()):
        shard = shards[shard_index]
        date = shard.name.removeprefix("date=")
        references.sort()
        if len({row for row, _ in references}) != len(references):
            raise RuntimeError(f"H011 prediction feature row repeated in {date}")
        examples: dict[int, dict[str, Any]] = {}
        wanted = {row for row, _ in references}
        for row, example in enumerate(iter_jsonl_zst(shard / "examples.jsonl.zst")):
            if row in wanted:
                examples[row] = example
            if len(examples) == len(wanted):
                break
        if len(examples) != len(wanted):
            raise RuntimeError(f"H011 prediction feature row is missing in {date}")
        features = np.load(shard / "features.npy", mmap_mode="r")
        timestamps = np.load(shard / "timestamps.npy", mmap_mode="r")
        market_ids = np.load(shard / "market_ids.npy", mmap_mode="r")
        component_ids = np.load(shard / "component_ids.npy", mmap_mode="r")
        for row, output_index in references:
            example = examples[row]
            timestamp = int(arrays["timestamps"][output_index])
            market_id = int(arrays["market_ids"][output_index])
            component_id = int(arrays["component_ids"][output_index])
            if (
                timestamp != int(timestamps[row])
                or timestamp != int(example["decision_time_unix"])
                or market_id != int(market_ids[row])
                or market_id != int(example["market_state_id"])
                or component_id != int(component_ids[row])
            ):
                raise RuntimeError(f"H011 prediction metadata changed at {date}:{row}")
            logit = float(arrays["logits"][output_index])
            probability = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, logit))))
            bound.append(
                BoundPrediction(
                    split=split,
                    feature_date=date,
                    feature_row=row,
                    decision_id=str(example["decision_id"]),
                    evidence_trade_id=str(example["evidence_trade_id"]),
                    timestamp_unix=timestamp,
                    condition_id=str(example["condition_id"]),
                    component_id=str(example["component_id"]),
                    market_state_id=market_id,
                    component_state_id=component_id,
                    probability_outcome0=probability,
                    label_outcome0=float(arrays["labels"][output_index]),
                    market_probability_outcome0=float(arrays["baselines"][output_index]),
                    feature_input_sha256=feature_digest(
                        np.asarray(features[row], dtype=np.float32),
                        normalization_sha256=normalization_sha256,
                        date=date,
                        row=row,
                    ),
                )
            )
    bound.sort(key=lambda row: (row.timestamp_unix, row.decision_id))
    return bound
