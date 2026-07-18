from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, sha256_file
from sphinx_trace.policy_decisions import PolicyDecisionRef
from sphinx_trace.policy_encodings import PolicyEncodingStore


def _metadata(path: Path, root: Path, array: NDArray[Any]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _cache(root: Path) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    pack = root / "pack"
    source_shard = pack / "shards" / "date=2026-01-01"
    source_shard.mkdir(parents=True)
    (pack / "receipts").mkdir()
    atomic_json(
        pack / "manifest.json",
        {"valid": True, "test_labels_opened": False, "test_label_rows": 0},
    )
    source_receipt = pack / "receipts" / "date=2026-01-01.json"
    atomic_json(source_receipt, {"date": "2026-01-01", "rows": 3})
    policy = root / "policy"
    policy.mkdir()
    atomic_json(
        policy / "result.json",
        {"valid": True, "test_labels_opened": False, "test_rows_consumed": 0},
    )
    cache = root / "cache"
    shard = cache / "shards" / "date=2026-01-01"
    receipt_root = cache / "receipts"
    shard.mkdir(parents=True)
    receipt_root.mkdir()
    arrays: dict[str, NDArray[Any]] = {
        "row_indices.npy": np.array([2, 5], dtype=np.int64),
        "market_latents.npy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "terminal_logits.npy": np.array([0.2, 1.5], dtype=np.float32),
        "uncertainty_log_scales.npy": np.array([-0.1, -0.2], dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(shard / name, array, allow_pickle=False)
    files = {name: _metadata(shard / name, cache, array) for name, array in arrays.items()}
    pack_sha256 = sha256_file(pack / "manifest.json")
    policy_sha256 = sha256_file(policy / "result.json")
    receipt = {
        "record_type": "h012_policy_encoding_day_receipt",
        "date": "2026-01-01",
        "shard_index": 0,
        "rows": 2,
        "pack_manifest_sha256": pack_sha256,
        "policy_result_sha256": policy_sha256,
        "source_receipt_sha256": sha256_file(source_receipt),
        "files": files,
    }
    receipt_path = receipt_root / "date=2026-01-01.json"
    atomic_json(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)
    digest = hashlib.sha256(f"2026-01-01:{receipt_sha256}\n".encode()).hexdigest()
    atomic_json(
        cache / "manifest.json",
        {
            "record_type": "h012_policy_encoding_cache_manifest",
            "valid": True,
            "test_labels_opened": False,
            "test_rows_consumed": 0,
            "pack_manifest_sha256": pack_sha256,
            "policy_result_sha256": policy_sha256,
            "latent_width": 2,
            "latent_dtype": "float32",
            "rows": 2,
            "daily_receipts_sha256": digest,
            "shards": [
                {
                    "shard_index": 0,
                    "date": "2026-01-01",
                    "rows": 2,
                    "receipt_path": "receipts/date=2026-01-01.json",
                    "receipt_sha256": receipt_sha256,
                    "files": files,
                }
            ],
        },
    )
    return cache, pack, policy, (source_shard,)


def _ref(row: int) -> PolicyDecisionRef:
    return PolicyDecisionRef(
        "validation", 0, "2026-01-01", row, f"decision-{row}", "trade", 10,
        "condition", "component", 1, 2,
    )


def test_policy_encoding_store_binds_and_finds_exact_source_row(tmp_path: Path) -> None:
    cache, pack, policy, shards = _cache(tmp_path)
    store = PolicyEncodingStore(cache, pack, policy, shards, cache_shards=1)

    loaded = store.load(_ref(5))

    assert loaded.market_latent.tolist() == [3.0, 4.0]
    assert loaded.terminal_outcome_logit == pytest.approx(1.5)
    assert loaded.uncertainty_log_scale == pytest.approx(-0.2)
    with pytest.raises(KeyError, match="missing decision row"):
        store.load(_ref(3))


def test_policy_encoding_store_rejects_policy_or_array_tampering(tmp_path: Path) -> None:
    cache, pack, policy, shards = _cache(tmp_path)
    atomic_json(policy / "result.json", {"changed": True})
    with pytest.raises(RuntimeError, match="contract changed"):
        PolicyEncodingStore(cache, pack, policy, shards)

    cache, pack, policy, shards = _cache(tmp_path / "second")
    with (cache / "shards" / "date=2026-01-01" / "terminal_logits.npy").open("ab") as handle:
        handle.write(b"tamper")
    store = PolicyEncodingStore(cache, pack, policy, shards)
    with pytest.raises(RuntimeError, match="file changed"):
        store.load(_ref(2))


def test_policy_encoding_store_allows_a_frozen_state_only_descendant(tmp_path: Path) -> None:
    cache, pack, policy, shards = _cache(tmp_path)
    source_policy_sha256 = sha256_file(policy / "result.json")
    atomic_json(
        policy / "result.json",
        {
            "valid": True,
            "test_labels_opened": False,
            "test_rows_consumed": 0,
            "market_encoding_policy_result_sha256": source_policy_sha256,
            "market_backbone_frozen": True,
        },
    )

    store = PolicyEncodingStore(cache, pack, policy, shards)

    assert store.load(_ref(2)).terminal_outcome_logit == pytest.approx(0.2)

    result = {
        "valid": True,
        "test_labels_opened": False,
        "test_rows_consumed": 0,
        "market_encoding_policy_result_sha256": source_policy_sha256,
        "market_backbone_frozen": False,
    }
    atomic_json(policy / "result.json", result)
    with pytest.raises(RuntimeError, match="contract changed"):
        PolicyEncodingStore(cache, pack, policy, shards)
