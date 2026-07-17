from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scripts.run_s0_h007_ablation import _bootstrap_delta
from scripts.train_s0_trial_t0 import PackedSplit

from sphinx_corpus.io import write_jsonl_zst
from sphinx_trace.config import load_json
from sphinx_trace.features import WalletEvent, build_feature_sequence, wallet_event
from sphinx_trace.model import SphinxTraceS0, parameter_count

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_train.json"
CONDITION_ID = "0x" + ("ab" * 32)


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index in range(128):
        wallet = "0x" + (("11" if index % 2 else "22") * 20)
        output.append(
            {
                "condition_id": CONDITION_ID,
                "trade_id": f"{index:064x}",
                "transaction_hash": "0x" + f"{index:064x}",
                "wallet": wallet,
                "side": "BUY" if index % 3 else "SELL",
                "outcome_index": index % 2,
                "price": str(0.25 + index / 1000),
                "size": str(index + 1),
                "notional_usd": str((index + 1) * (2 if index % 2 else 1)),
                "timestamp_unix": 1_700_000_000 + index,
            }
        )
    return output


def target() -> dict[str, Any]:
    return {
        "decision_time_unix": 1_700_000_128,
        "resolved_yes": 1,
        "targets": {
            "yes_markout_5m": 0.1,
            "yes_markout_1h": None,
            "yes_markout_1d": -0.2,
            "directional_markout_5m": 0.1,
            "directional_markout_1h": None,
            "directional_markout_1d": -0.2,
            "net_edge_proxy": 0.095,
        },
    }


def histories(source: list[dict[str, Any]]) -> dict[str, list[WalletEvent]]:
    output: dict[str, list[WalletEvent]] = {}
    for row in source:
        event = wallet_event(row)
        assert event is not None
        output.setdefault(str(row["wallet"]), []).append(event)
    return output


def test_s0_parameter_count_stays_in_registered_range() -> None:
    payload = config()
    model = SphinxTraceS0(
        payload,
        sequence_length=int(payload["features"]["sequence_length"]),
        feature_width=int(payload["features"]["feature_width"]),
    )

    count = parameter_count(model)

    assert count == 50_213_128
    assert payload["model"]["parameter_minimum"] <= count
    assert count <= payload["model"]["parameter_maximum"]


def test_feature_pack_is_wallet_identifier_invariant() -> None:
    original = rows()
    renamed = [dict(row) for row in original]
    mapping = {
        "0x" + ("11" * 20): "0x" + ("aa" * 20),
        "0x" + ("22" * 20): "0x" + ("bb" * 20),
    }
    for row in renamed:
        row["wallet"] = mapping[str(row["wallet"])]

    packed_original = build_feature_sequence(
        original,
        histories(original),
        target(),
        config(),
    )
    packed_renamed = build_feature_sequence(
        renamed,
        histories(renamed),
        target(),
        config(),
    )

    assert packed_original is not None
    assert packed_renamed is not None
    for left, right in zip(packed_original, packed_renamed, strict=True):
        np.testing.assert_array_equal(left, right)


def test_future_wallet_events_do_not_change_features() -> None:
    source = rows()
    past_histories = histories(source)
    future_histories = histories(source)
    wallet = str(source[0]["wallet"])
    future_histories[wallet].append(
        WalletEvent(
            timestamp_unix=int(target()["decision_time_unix"]) + 1,
            price=0.99,
            size=1_000_000.0,
            notional=1_000_000.0,
            side=1,
            outcome_index=0,
            market_key=123,
        )
    )

    packed_past = build_feature_sequence(source, past_histories, target(), config())
    packed_future = build_feature_sequence(source, future_histories, target(), config())

    assert packed_past is not None
    assert packed_future is not None
    np.testing.assert_array_equal(packed_past[0], packed_future[0])


def test_missing_targets_are_masked() -> None:
    source = rows()
    packed = build_feature_sequence(source, histories(source), target(), config())

    assert packed is not None
    _, _, target_values, target_mask = packed
    np.testing.assert_array_equal(target_mask, [1, 1, 0, 1, 1, 0, 1, 1])
    assert target_values[2] == 0.0
    assert target_values[5] == 0.0


def test_prior_event_wallet_control_is_causal_and_cross_event(tmp_path: Path) -> None:
    features = np.zeros((3, 224, 16), dtype=np.float16)
    for index in range(3):
        features[index, 128:160] = index + 1
    np.save(tmp_path / "features.npy", features, allow_pickle=False)
    np.save(tmp_path / "token_types.npy", np.zeros((3, 224), dtype=np.uint8))
    np.save(tmp_path / "targets.npy", np.zeros((3, 8), dtype=np.float32))
    np.save(tmp_path / "target_mask.npy", np.ones((3, 8), dtype=np.uint8))
    np.save(tmp_path / "baselines.npy", np.zeros((3, 8), dtype=np.float32))
    write_jsonl_zst(
        tmp_path / "examples.jsonl.zst",
        [
            {"example_id": "a-20", "event_id": "a", "decision_time_unix": 20},
            {"example_id": "b-10", "event_id": "b", "decision_time_unix": 10},
            {"example_id": "a-30", "event_id": "a", "decision_time_unix": 30},
        ],
    )
    dataset = PackedSplit(
        tmp_path,
        wallet_mode="prior_event_control",
        wallet_start=128,
        wallet_tokens=32,
    )

    assert dataset.donor_indices == [1, -1, 1]
    assert dataset.control_audit["donor_time_violations"] == 0
    assert dataset.control_audit["same_event_donors"] == 0
    controlled_features = dataset[2][0].numpy()
    np.testing.assert_array_equal(controlled_features[128:160], features[1, 128:160])


def test_event_bootstrap_detects_uniform_candidate_improvement() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    candidate = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    reference = np.asarray([0.4, 0.4, 0.6, 0.6], dtype=np.float64)

    result = _bootstrap_delta(
        candidate,
        reference,
        labels,
        ["event-a", "event-a", "event-b", "event-b"],
        samples=500,
        seed=71,
    )

    assert result["event_groups"] == 2
    assert result["candidate_minus_reference_log_loss"] < 0.0
    assert result["candidate_better_95pct"] is True
