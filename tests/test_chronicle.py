from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from sphinx_trace.chronicle import (
    MarketResolution,
    SplitPlan,
    build_condition_targets,
    market_resolution_from_atlas,
    target_row_is_causal,
    target_row_matches_contract,
    yes_direction,
    yes_equivalent_price,
)
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0.json"


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def trade(index: int, timestamp: int, *, price: float = 0.4) -> dict[str, Any]:
    return {
        "condition_id": "0x" + ("ab" * 32),
        "trade_id": f"{index:064x}",
        "transaction_hash": "0x" + f"{index:064x}",
        "wallet": "0x" + ("12" * 20),
        "side": "BUY",
        "outcome_index": 0,
        "price": str(price),
        "timestamp_unix": timestamp,
    }


def test_split_plan_enforces_registered_embargoes() -> None:
    plan = SplitPlan.from_config(config())

    assert plan.locate(datetime(2026, 2, 15, tzinfo=UTC)).id == "train"  # type: ignore[union-attr]
    assert plan.locate(datetime(2026, 2, 18, tzinfo=UTC)) is None
    assert plan.locate(datetime(2026, 2, 23, tzinfo=UTC)).id == "validation"  # type: ignore[union-attr]

    broken = deepcopy(config())
    broken["split"]["segments"][1]["start"] = "2026-02-22T00:00:00Z"
    with pytest.raises(ValueError, match="exact 7-day embargo"):
        SplitPlan.from_config(broken)


def test_atlas_resolution_requires_ordered_terminal_yes_no() -> None:
    payload = config()
    row: dict[str, Any] = {
        "condition_id": "0x" + ("ab" * 32),
        "event_ids": ["event-1"],
        "resolution_status": "resolved",
        "closed_at": "2026-01-01T00:00:00Z",
        "observed_at": "2026-07-16T00:00:00Z",
        "source_payload": {
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
        },
    }

    market = market_resolution_from_atlas(row, payload["eligibility"])
    assert market is not None
    assert market.resolved_yes == 1

    row["source_payload"]["outcomes"] = '["No", "Yes"]'
    assert market_resolution_from_atlas(row, payload["eligibility"]) is None


def test_yes_price_and_direction_normalize_both_outcomes() -> None:
    assert yes_equivalent_price({"price": "0.4", "outcome_index": 0}) == pytest.approx(0.4)
    assert yes_equivalent_price({"price": "0.4", "outcome_index": 1}) == pytest.approx(0.6)
    assert yes_direction({"side": "BUY", "outcome_index": 0}) == 1
    assert yes_direction({"side": "SELL", "outcome_index": 0}) == -1
    assert yes_direction({"side": "BUY", "outcome_index": 1}) == -1
    assert yes_direction({"side": "SELL", "outcome_index": 1}) == 1


def test_target_builder_preserves_causal_and_split_boundaries() -> None:
    payload = config()
    plan = SplitPlan.from_config(payload)
    window = plan.by_id("train")
    base = datetime(2026, 1, 1, tzinfo=UTC)
    base_unix = int(base.timestamp())
    rows = [trade(index, base_unix + index) for index in range(128)]
    decision_unix = base_unix + 128
    rows.extend(
        [
            trade(128, decision_unix + 300, price=0.5),
            trade(129, decision_unix + 3600, price=0.6),
            trade(130, decision_unix + 86400, price=0.7),
        ]
    )
    market = MarketResolution(
        condition_id="0x" + ("ab" * 32),
        event_id="event-1",
        resolved_at=base + timedelta(days=3),
        resolved_yes=1,
        atlas_observed_at="2026-07-16T00:00:00Z",
    )

    examples = build_condition_targets(rows, market, window, payload)

    assert len(examples) == 1
    example = examples[0]
    assert example["feature_max_event_time_unix"] < example["decision_time_unix"]
    assert example["targets"]["yes_markout_5m"] == pytest.approx(0.1)
    assert example["targets"]["yes_markout_1h"] == pytest.approx(0.2)
    assert example["targets"]["yes_markout_1d"] == pytest.approx(0.3)
    assert example["targets"]["resolved_directional_edge"] == pytest.approx(0.6)
    assert example["targets"]["net_edge_proxy"] == pytest.approx(0.595)
    assert target_row_is_causal(example, plan) is True
    assert target_row_matches_contract(example, payload, plan) is True

    invalid = deepcopy(example)
    invalid["feature_max_event_time_unix"] = invalid["decision_time_unix"]
    assert target_row_is_causal(invalid, plan) is False

    early_target = deepcopy(example)
    early_target["targets"]["markout_observed_at_5m"] = early_target["decision_time"]
    assert target_row_matches_contract(early_target, payload, plan) is False
