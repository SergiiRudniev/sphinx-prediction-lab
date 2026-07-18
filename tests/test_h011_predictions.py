from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sphinx_corpus.io import sha256_file, write_jsonl_zst
from sphinx_trace.h011_predictions import bind_development_predictions


def _fixture(root: Path) -> tuple[Path, Path]:
    pack = root / "pack"
    model = root / "model"
    shard = pack / "shards" / "date=2026-03-01"
    shard.mkdir(parents=True)
    model.mkdir()
    np.save(shard / "features.npy", np.zeros((1, 128), dtype=np.float32))
    np.save(shard / "timestamps.npy", np.array([10], dtype=np.int64))
    np.save(shard / "market_ids.npy", np.array([3], dtype=np.int32))
    np.save(shard / "component_ids.npy", np.array([4], dtype=np.int32))
    write_jsonl_zst(
        shard / "examples.jsonl.zst",
        [
            {
                "decision_id": "decision",
                "evidence_trade_id": "trade",
                "decision_time_unix": 10,
                "condition_id": "condition",
                "component_id": "component",
                "market_state_id": 3,
            }
        ],
    )
    with (pack / "normalization.npz").open("wb") as handle:
        np.savez(handle, median=np.zeros(128), scale=np.ones(128))
    manifest_path = pack / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "valid": True,
                "test_labels_opened": False,
                "normalization": {"sha256": sha256_file(pack / "normalization.npz")},
            }
        ),
        encoding="utf-8",
    )
    receipts = pack / "receipts"
    receipts.mkdir()
    (receipts / "date=2026-03-01.json").write_text("{}", encoding="utf-8")
    source_digest = hashlib.sha256()
    source_digest.update(sha256_file(manifest_path).encode())
    source_digest.update(sha256_file(pack / "normalization.npz").encode())
    source_digest.update(f"2026-03-01:{sha256_file(receipts / 'date=2026-03-01.json')}\n".encode())
    predictions_path = model / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        validation_logits=np.array([0.0], dtype=np.float32),
        validation_labels=np.array([1.0], dtype=np.float32),
        validation_baselines=np.array([0.4], dtype=np.float32),
        validation_shard_indices=np.array([0], dtype=np.uint16),
        validation_row_indices=np.array([0], dtype=np.int32),
        validation_timestamps=np.array([10], dtype=np.int64),
        validation_market_ids=np.array([3], dtype=np.int32),
        validation_component_ids=np.array([4], dtype=np.int32),
    )
    (model / "result.json").write_text(
        json.dumps(
            {
                "valid": True,
                "test_labels_opened": False,
                "test_rows_consumed": 0,
                "source_digest": source_digest.hexdigest(),
                "predictions_sha256": sha256_file(predictions_path),
            }
        ),
        encoding="utf-8",
    )
    return pack, model


def test_development_prediction_binding_preserves_exact_provenance(tmp_path: Path) -> None:
    pack, model = _fixture(tmp_path)
    first = bind_development_predictions(pack, model, "validation")
    second = bind_development_predictions(pack, model, "validation")
    assert len(first) == 1
    assert first[0].probability_outcome0 == pytest.approx(0.5)
    assert first[0].feature_input_sha256 == second[0].feature_input_sha256
    assert first[0].decision_id == "decision"
    assert first[0].feature_row == 0


def test_prediction_binding_rejects_test_and_changed_metadata(tmp_path: Path) -> None:
    pack, model = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Only validation and calibration"):
        bind_development_predictions(pack, model, "test")
    np.save(model / "changed.npy", np.array([1]))
    with np.load(model / "predictions.npz") as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    np.savez_compressed(
        model / "predictions.npz",
        validation_logits=values["validation_logits"],
        validation_labels=values["validation_labels"],
        validation_baselines=values["validation_baselines"],
        validation_shard_indices=values["validation_shard_indices"],
        validation_row_indices=values["validation_row_indices"],
        validation_timestamps=np.array([11], dtype=np.int64),
        validation_market_ids=values["validation_market_ids"],
        validation_component_ids=values["validation_component_ids"],
    )
    result = json.loads((model / "result.json").read_text(encoding="utf-8"))
    result["predictions_sha256"] = sha256_file(model / "predictions.npz")
    (model / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata changed"):
        bind_development_predictions(pack, model, "validation")
