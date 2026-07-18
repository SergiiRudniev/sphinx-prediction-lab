"""Index exact H011 decision rows for sequential H012 policy inference."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import iter_jsonl_zst, sha256_file
from sphinx_trace.h011_predictions import feature_digest

SPLIT_CODES = {"validation": 2, "calibration": 3}


@dataclass(frozen=True, slots=True)
class PolicyDecisionRef:
    split: str
    shard_index: int
    feature_date: str
    feature_row: int
    decision_id: str
    evidence_trade_id: str
    timestamp_unix: int
    condition_id: str
    component_id: str
    market_state_id: int
    component_state_id: int


@dataclass(frozen=True, slots=True)
class LoadedPolicyFeature:
    normalized: NDArray[np.float32]
    market_probability_outcome0: float
    feature_sha256: str


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_policy_decisions(
    pack_dir: Path,
    splits: tuple[str, ...] = ("validation", "calibration"),
) -> tuple[dict[str, tuple[PolicyDecisionRef, ...]], tuple[Path, ...]]:
    """Bind every qualified development decision to its evidence trade."""

    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLIT_CODES for split in splits)
    ):
        raise ValueError("H012 decision index supports unique development splits only")
    manifest = _load_object(pack_dir / "manifest.json")
    if manifest.get("valid") is not True or manifest.get("test_labels_opened") is not False:
        raise RuntimeError("H012 decision index requires a valid closed-test pack")
    shards = tuple(path for path in sorted((pack_dir / "shards").glob("date=*")) if path.is_dir())
    if not shards:
        raise RuntimeError("H012 decision index found no feature shards")
    code_to_split = {SPLIT_CODES[split]: split for split in splits}
    by_evidence: dict[str, list[PolicyDecisionRef]] = {}
    decision_ids: set[str] = set()
    for shard_index, shard in enumerate(shards):
        date = shard.name.removeprefix("date=")
        split_codes = np.load(shard / "split_codes.npy", mmap_mode="r")
        label_mask = np.load(shard / "label_mask.npy", mmap_mode="r")
        timestamps = np.load(shard / "timestamps.npy", mmap_mode="r")
        market_ids = np.load(shard / "market_ids.npy", mmap_mode="r")
        component_ids = np.load(shard / "component_ids.npy", mmap_mode="r")
        if not (
            len(split_codes)
            == len(label_mask)
            == len(timestamps)
            == len(market_ids)
            == len(component_ids)
        ):
            raise RuntimeError(f"H012 decision arrays do not align at {date}")
        example_count = 0
        for row, example in enumerate(iter_jsonl_zst(shard / "examples.jsonl.zst")):
            example_count += 1
            split = code_to_split.get(int(split_codes[row]))
            if split is None or not bool(label_mask[row]):
                continue
            timestamp = int(timestamps[row])
            market_state_id = int(market_ids[row])
            component_state_id = int(component_ids[row])
            if (
                timestamp != int(example["decision_time_unix"])
                or market_state_id != int(example["market_state_id"])
                or (
                    "component_state_id" in example
                    and component_state_id != int(example["component_state_id"])
                )
            ):
                raise RuntimeError(f"H012 decision metadata changed at {date}:{row}")
            decision_id = str(example["decision_id"])
            if decision_id in decision_ids:
                raise RuntimeError(f"H012 decision ID repeats: {decision_id}")
            decision_ids.add(decision_id)
            ref = PolicyDecisionRef(
                split=split,
                shard_index=shard_index,
                feature_date=date,
                feature_row=row,
                decision_id=decision_id,
                evidence_trade_id=str(example["evidence_trade_id"]),
                timestamp_unix=timestamp,
                condition_id=str(example["condition_id"]).lower(),
                component_id=str(example["component_id"]),
                market_state_id=market_state_id,
                component_state_id=component_state_id,
            )
            by_evidence.setdefault(ref.evidence_trade_id, []).append(ref)
        if example_count != len(split_codes):
            raise RuntimeError(f"H012 examples do not align at {date}")
    frozen: dict[str, tuple[PolicyDecisionRef, ...]] = {}
    for evidence, refs in by_evidence.items():
        refs.sort(key=lambda row: (row.timestamp_unix, row.decision_id))
        frozen[evidence] = tuple(refs)
    return frozen, shards


class PolicyFeatureStore:
    """Bounded LRU access to normalized feature rows and their source digests."""

    def __init__(
        self,
        pack_dir: Path,
        shards: tuple[Path, ...],
        *,
        feature_clip: float,
        cache_shards: int = 8,
    ) -> None:
        if feature_clip <= 0.0 or cache_shards <= 0:
            raise ValueError("H012 feature-store settings must be positive")
        self.pack_dir = pack_dir
        self.shards = shards
        self.feature_clip = float(feature_clip)
        self.cache_shards = int(cache_shards)
        normalization_path = pack_dir / "normalization.npz"
        manifest = _load_object(pack_dir / "manifest.json")
        normalization_sha256 = sha256_file(normalization_path)
        if manifest.get("normalization", {}).get("sha256") != normalization_sha256:
            raise RuntimeError("H012 feature-store normalization changed")
        with np.load(normalization_path, allow_pickle=False) as archive:
            self.median = np.asarray(archive["median"], dtype=np.float32)
            self.scale = np.asarray(archive["scale"], dtype=np.float32)
        self.normalization_sha256 = normalization_sha256
        self._cache: OrderedDict[int, NDArray[np.float32]] = OrderedDict()
        self._baseline_cache: OrderedDict[int, NDArray[np.float32]] = OrderedDict()

    def _features(self, shard_index: int) -> NDArray[np.float32]:
        cached = self._cache.pop(shard_index, None)
        if cached is None:
            if not 0 <= shard_index < len(self.shards):
                raise IndexError("H012 feature reference has an invalid shard")
            cached = np.load(self.shards[shard_index] / "features.npy", mmap_mode="r")
        self._cache[shard_index] = cached
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return cached

    def _baselines(self, shard_index: int) -> NDArray[np.float32]:
        cached = self._baseline_cache.pop(shard_index, None)
        if cached is None:
            if not 0 <= shard_index < len(self.shards):
                raise IndexError("H012 baseline reference has an invalid shard")
            cached = np.load(self.shards[shard_index] / "baselines.npy", mmap_mode="r")
        self._baseline_cache[shard_index] = cached
        while len(self._baseline_cache) > self.cache_shards:
            self._baseline_cache.popitem(last=False)
        return cached

    def load(self, ref: PolicyDecisionRef) -> LoadedPolicyFeature:
        raw = self._features(ref.shard_index)
        baselines = self._baselines(ref.shard_index)
        if not 0 <= ref.feature_row < len(raw) == len(baselines):
            raise IndexError("H012 feature reference has an invalid row")
        source = np.asarray(raw[ref.feature_row], dtype=np.float32)
        digest = feature_digest(
            source,
            normalization_sha256=self.normalization_sha256,
            date=ref.feature_date,
            row=ref.feature_row,
        )
        normalized = (source - self.median) / self.scale
        np.clip(normalized, -self.feature_clip, self.feature_clip, out=normalized)
        baseline = float(baselines[ref.feature_row])
        if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise RuntimeError("H012 market baseline is invalid")
        return LoadedPolicyFeature(normalized, baseline, digest)


def policy_input_digest(
    feature_sha256: str,
    market_probability_outcome0: float,
    portfolio_features: tuple[float, ...],
    prediction_memory_features: tuple[float, ...],
    previous_action_id: int,
    physical_action_mask: tuple[bool, ...],
) -> str:
    """Bind a policy action to feature, portfolio, memory and physical state."""

    if len(feature_sha256) != 64:
        raise ValueError("H012 feature input digest must be SHA-256")
    payload = {
        "feature_sha256": feature_sha256,
        "market_probability_outcome0": market_probability_outcome0,
        "portfolio_features": portfolio_features,
        "prediction_memory_features": prediction_memory_features,
        "previous_action_id": previous_action_id,
        "physical_action_mask": physical_action_mask,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
