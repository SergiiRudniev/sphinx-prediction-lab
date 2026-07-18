"""Source-bound cached H012 market encodings for exact sequential replay."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import sha256_file
from sphinx_trace.policy_decisions import PolicyDecisionRef


@dataclass(frozen=True, slots=True)
class LoadedPolicyEncoding:
    market_latent: NDArray[np.float32]
    terminal_outcome_logit: float
    uncertainty_log_scale: float


@dataclass(frozen=True, slots=True)
class _EncodingShard:
    shard_index: int
    date: str
    rows: int
    files: dict[str, dict[str, Any]]


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


class PolicyEncodingStore:
    """Verify and lazily load immutable market encodings for policy decisions."""

    FILES = (
        "row_indices.npy",
        "market_latents.npy",
        "terminal_logits.npy",
        "uncertainty_log_scales.npy",
    )

    def __init__(
        self,
        cache_dir: Path,
        pack_dir: Path,
        policy_dir: Path,
        shards: tuple[Path, ...],
        *,
        cache_shards: int = 8,
    ) -> None:
        if cache_shards <= 0:
            raise ValueError("H012 encoding cache_shards must be positive")
        self.cache_dir = cache_dir
        self.pack_dir = pack_dir
        self.policy_dir = policy_dir
        self.cache_shards = int(cache_shards)
        self.manifest_path = cache_dir / "manifest.json"
        manifest = _load_object(self.manifest_path)
        pack_manifest_sha256 = sha256_file(pack_dir / "manifest.json")
        policy_result_sha256 = sha256_file(policy_dir / "result.json")
        if (
            manifest.get("record_type") != "h012_policy_encoding_cache_manifest"
            or manifest.get("valid") is not True
            or manifest.get("test_labels_opened") is not False
            or int(manifest.get("test_rows_consumed", -1)) != 0
            or manifest.get("pack_manifest_sha256") != pack_manifest_sha256
            or manifest.get("policy_result_sha256") != policy_result_sha256
        ):
            raise RuntimeError("H012 policy encoding cache contract changed")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.width = int(manifest.get("latent_width", 0))
        if self.width <= 0 or manifest.get("latent_dtype") != "float32":
            raise RuntimeError("H012 policy encoding cache representation is invalid")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or len(raw_shards) != len(shards):
            raise RuntimeError("H012 policy encoding cache shard coverage changed")
        bound: list[_EncodingShard] = []
        receipt_digest = hashlib.sha256()
        total_rows = 0
        for shard_index, (source_shard, raw) in enumerate(zip(shards, raw_shards, strict=True)):
            if not isinstance(raw, dict):
                raise RuntimeError("H012 policy encoding shard metadata is invalid")
            date = source_shard.name.removeprefix("date=")
            if int(raw.get("shard_index", -1)) != shard_index or raw.get("date") != date:
                raise RuntimeError("H012 policy encoding cache no longer aligns with feature pack")
            receipt_relative = str(raw.get("receipt_path") or "")
            receipt_path = cache_dir / receipt_relative
            receipt_sha256 = str(raw.get("receipt_sha256") or "")
            if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha256:
                raise RuntimeError(f"H012 policy encoding receipt changed at {date}")
            receipt = _load_object(receipt_path)
            source_receipt = pack_dir / "receipts" / f"date={date}.json"
            rows = int(raw.get("rows", -1))
            if (
                receipt.get("record_type") != "h012_policy_encoding_day_receipt"
                or receipt.get("date") != date
                or int(receipt.get("shard_index", -1)) != shard_index
                or int(receipt.get("rows", -1)) != rows
                or receipt.get("pack_manifest_sha256") != pack_manifest_sha256
                or receipt.get("policy_result_sha256") != policy_result_sha256
                or receipt.get("source_receipt_sha256") != sha256_file(source_receipt)
                or receipt.get("files") != raw.get("files")
            ):
                raise RuntimeError(f"H012 policy encoding receipt contract changed at {date}")
            files = raw.get("files")
            if not isinstance(files, dict) or set(files) != set(self.FILES):
                raise RuntimeError(f"H012 policy encoding files are incomplete at {date}")
            bound.append(_EncodingShard(shard_index, date, rows, files))
            receipt_digest.update(f"{date}:{receipt_sha256}\n".encode())
            total_rows += rows
        if (
            total_rows != int(manifest.get("rows", -1))
            or receipt_digest.hexdigest() != manifest.get("daily_receipts_sha256")
        ):
            raise RuntimeError("H012 policy encoding cache totals changed")
        self.shards = tuple(bound)
        self._cache: OrderedDict[
            int,
            tuple[
                NDArray[np.int64],
                NDArray[np.float32],
                NDArray[np.float32],
                NDArray[np.float32],
            ],
        ] = OrderedDict()

    def _load_shard(
        self, shard_index: int
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        if not 0 <= shard_index < len(self.shards):
            raise IndexError("H012 policy encoding reference has an invalid shard")
        shard = self.shards[shard_index]
        arrays: list[NDArray[Any]] = []
        for name in self.FILES:
            metadata = shard.files[name]
            path = self.cache_dir / str(metadata["path"])
            if (
                not path.is_file()
                or int(metadata.get("bytes", -1)) != path.stat().st_size
                or metadata.get("sha256") != sha256_file(path)
            ):
                raise RuntimeError(f"H012 policy encoding file changed: {path}")
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        row_indices, latents, terminal, uncertainty = arrays
        if (
            row_indices.dtype != np.int64
            or row_indices.shape != (shard.rows,)
            or latents.dtype != np.float32
            or latents.shape != (shard.rows, self.width)
            or terminal.dtype != np.float32
            or terminal.shape != (shard.rows,)
            or uncertainty.dtype != np.float32
            or uncertainty.shape != (shard.rows,)
            or (shard.rows > 1 and not bool(np.all(row_indices[1:] > row_indices[:-1])))
        ):
            raise RuntimeError(f"H012 policy encoding arrays are invalid at {shard.date}")
        typed = (row_indices, latents, terminal, uncertainty)
        self._cache[shard_index] = typed
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return typed

    def load(self, ref: PolicyDecisionRef) -> LoadedPolicyEncoding:
        if not 0 <= ref.shard_index < len(self.shards):
            raise IndexError("H012 policy encoding reference has an invalid shard")
        shard = self.shards[ref.shard_index]
        if ref.feature_date != shard.date:
            raise RuntimeError("H012 policy encoding date no longer matches the decision")
        rows, latents, terminal, uncertainty = self._load_shard(ref.shard_index)
        position = int(np.searchsorted(rows, ref.feature_row))
        if position >= len(rows) or int(rows[position]) != ref.feature_row:
            raise KeyError(
                f"H012 policy encoding is missing decision row {ref.feature_date}:{ref.feature_row}"
            )
        latent = np.array(latents[position], dtype=np.float32, copy=True)
        terminal_value = float(terminal[position])
        uncertainty_value = float(uncertainty[position])
        if (
            not bool(np.isfinite(latent).all())
            or not np.isfinite(terminal_value)
            or not np.isfinite(uncertainty_value)
        ):
            raise RuntimeError("H012 policy encoding contains non-finite values")
        return LoadedPolicyEncoding(latent, terminal_value, uncertainty_value)
