from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.build_h011_feature_pack import (
    RecurrentState,
    StaticIndex,
    _actor_schedule,
    _process_day,
)

from sphinx_corpus.io import sha256_file, write_jsonl_zst
from sphinx_trace.h011_kernel import python_h011_kernel


def _empty_resolution() -> dict[str, np.ndarray]:
    return {
        "timestamps": np.asarray([], dtype=np.int64),
        "wallet_ids": np.asarray([], dtype=np.int32),
        "edges": np.asarray([], dtype=np.float32),
        "pnls": np.asarray([], dtype=np.float32),
        "wins": np.asarray([], dtype=np.int8),
    }


def _one_resolution() -> dict[str, np.ndarray]:
    return {
        "timestamps": np.asarray([150], dtype=np.int64),
        "wallet_ids": np.asarray([1], dtype=np.int32),
        "edges": np.asarray([0.75], dtype=np.float32),
        "pnls": np.asarray([4.0], dtype=np.float32),
        "wins": np.asarray([1], dtype=np.int8),
    }


def _index() -> StaticIndex:
    return StaticIndex(
        wallet_to_id={"0xa": 0, "0xb": 1},
        wallet_count=2,
        market_to_id={"0xmarket": 0},
        market_conditions=["0xmarket"],
        market_components=np.asarray([0], dtype=np.int32),
        market_created=np.asarray([50], dtype=np.int64),
        market_end=np.asarray([300], dtype=np.int64),
        market_split=np.asarray([1], dtype=np.uint8),
        market_label=np.asarray([1.0], dtype=np.float32),
        component_ids=["component"],
        component_market_count=np.asarray([1], dtype=np.int32),
        component_neg_risk_count=np.asarray([0], dtype=np.int32),
        component_unclosed_count=np.asarray([0], dtype=np.int32),
        participant_source_sha256="participant",
        market_index_sha256="market",
    )


def _trade(
    trade_id: str,
    wallet: str,
    timestamp: int,
    outcome: int,
    price: str,
) -> dict[str, object]:
    return {
        "record_type": "public_trade",
        "trade_id": trade_id,
        "condition_id": "0xmarket",
        "wallet": wallet,
        "timestamp_unix": timestamp,
        "price": price,
        "size": "10",
        "notional_usd": "5",
        "outcome_index": outcome,
        "side": "BUY" if outcome == 0 else "SELL",
    }


def test_actor_schedule_requires_complete_hash_bound_tasks(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "actor.jsonl.zst"
    write_jsonl_zst(task, [{"wallet": "0xa"}])
    manifest = {
        "complete": True,
        "expected_task_count": 1,
        "task_count": 1,
        "end_exclusive": "2026-01-06T00:00:00Z",
        "tasks": [
            {
                "task_id": "task",
                "path": "tasks/actor.jsonl.zst",
                "sha256": sha256_file(task),
                "window_end_exclusive": "2025-08-01T00:00:00Z",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    schedule = _actor_schedule(tmp_path)

    assert schedule.complete is True
    assert schedule.enabled_on("2025-08-01") is True
    assert schedule.enabled_on("2026-01-06") is False
    task.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="task changed"):
        _actor_schedule(tmp_path)


def test_day_builder_aligns_exact_cursor_and_emits_post_trade_state(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl.zst"
    decisions = tmp_path / "decisions.jsonl.zst"
    write_jsonl_zst(
        stream,
        [
            _trade("trade-1", "0xa", 100, 0, "0.7"),
            _trade("trade-2", "0xb", 200, 1, "0.2"),
        ],
    )
    write_jsonl_zst(
        decisions,
        [
            {
                "decision_id": "decision",
                "component_id": "component",
                "decision_time_unix": 200,
                "feature_max_event_time_unix": 200,
                "evidence_trade_id": "trade-2",
                "stream_row": 1,
            }
        ],
    )
    index = _index()
    state = RecurrentState.empty(index)
    receipt = _process_day(
        date="1970-01-01",
        stream_path=stream,
        expected_stream_rows=2,
        decision_path=decisions,
        index=index,
        state=state,
        kernel=python_h011_kernel(),
        chunk_rows=1,
        actor_enabled=False,
        resolution=_one_resolution(),
        output_dir=tmp_path / "output",
        emit=True,
    )
    assert receipt["rows"] == 1
    assert receipt["test_label_rows"] == 0
    assert receipt["resolution_events_applied_in_stream"] == 1
    root = tmp_path / "output" / "shards" / "date=1970-01-01"
    features = np.load(root / "features.npy")
    assert features.shape == (1, 128)
    assert features[0, 8] == pytest.approx(np.log1p(2))
    assert features[0, 11] == pytest.approx(0.8)
    assert features[0, 85] == pytest.approx(np.log1p(1))
    assert features[0, 86] == pytest.approx(0.75)
    assert np.load(root / "labels.npy").tolist() == [1.0]
    assert np.load(root / "label_mask.npy").tolist() == [1]
    assert state.universe_core[0] == 2


def test_day_builder_rejects_decision_trade_mismatch(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl.zst"
    decisions = tmp_path / "decisions.jsonl.zst"
    write_jsonl_zst(stream, [_trade("actual", "0xa", 100, 0, "0.7")])
    write_jsonl_zst(
        decisions,
        [
            {
                "decision_id": "decision",
                "component_id": "component",
                "decision_time_unix": 100,
                "feature_max_event_time_unix": 100,
                "evidence_trade_id": "different",
                "stream_row": 0,
            }
        ],
    )
    with pytest.raises(RuntimeError, match="evidence trade ID mismatch"):
        _process_day(
            date="1970-01-01",
            stream_path=stream,
            expected_stream_rows=1,
            decision_path=decisions,
            index=_index(),
            state=RecurrentState.empty(_index()),
            kernel=python_h011_kernel(),
            chunk_rows=10,
            actor_enabled=False,
            resolution=_empty_resolution(),
            output_dir=tmp_path / "output",
            emit=True,
        )


def test_day_builder_keeps_and_counts_out_of_range_source_price(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl.zst"
    decisions = tmp_path / "decisions.jsonl.zst"
    write_jsonl_zst(stream, [_trade("anomaly", "0xa", 100, 0, "1.1140588235")])
    write_jsonl_zst(
        decisions,
        [
            {
                "decision_id": "decision",
                "component_id": "component",
                "decision_time_unix": 100,
                "feature_max_event_time_unix": 100,
                "evidence_trade_id": "anomaly",
                "stream_row": 0,
            }
        ],
    )
    receipt = _process_day(
        date="1970-01-01",
        stream_path=stream,
        expected_stream_rows=1,
        decision_path=decisions,
        index=_index(),
        state=RecurrentState.empty(_index()),
        kernel=python_h011_kernel(),
        chunk_rows=10,
        actor_enabled=False,
        resolution=_empty_resolution(),
        output_dir=tmp_path / "output",
        emit=True,
    )
    features = np.load(tmp_path / "output" / "shards" / "date=1970-01-01" / "features.npy")
    assert receipt["source_price_anomaly_rows"] == 1
    assert receipt["stream_rows"] == 1
    assert features[0, 11] == 1.0
    assert features[0, 12] == 1.0
