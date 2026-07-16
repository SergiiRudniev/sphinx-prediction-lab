from datetime import UTC, datetime, timedelta

import pytest

from sphinx_trace.contracts import MarketState, ModelSignal, PositionState, TradeAction
from sphinx_trace.policy import PolicyLimits, decide

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def market() -> MarketState:
    return MarketState(
        condition_id="0xcondition",
        observed_at=NOW,
        yes_bid=0.50,
        yes_ask=0.52,
        no_bid=0.47,
        no_ask=0.49,
        available_depth_usd=1_000.0,
        seconds_to_resolution=86_400,
    )


def signal(**overrides: float) -> ModelSignal:
    values = {
        "fair_yes_probability": 0.62,
        "informed_flow_yes": 0.78,
        "expected_yes_edge": 0.07,
        "expected_no_edge": -0.10,
        "downside_edge_q10": 0.01,
        "confidence": 0.80,
    }
    values.update(overrides)
    return ModelSignal(generated_at=NOW, **values)


def test_flat_policy_opens_yes_only_when_edge_passes() -> None:
    decision = decide(signal(), market(), PositionState(), PolicyLimits())
    assert decision.action is TradeAction.BUY_YES
    assert decision.limit_price == pytest.approx(0.52)


def test_open_position_closes_when_edge_disappears() -> None:
    position = PositionState(outcome="YES", shares=100, average_price=0.40)
    decision = decide(
        signal(expected_yes_edge=-0.02),
        market(),
        position,
        PolicyLimits(),
    )
    assert decision.action is TradeAction.CLOSE
    assert decision.limit_price == pytest.approx(0.50)


def test_stale_signal_is_never_executed() -> None:
    stale = ModelSignal(
        generated_at=NOW - timedelta(seconds=60),
        fair_yes_probability=0.70,
        informed_flow_yes=0.90,
        expected_yes_edge=0.15,
        expected_no_edge=-0.20,
        downside_edge_q10=0.05,
        confidence=0.95,
    )
    decision = decide(stale, market(), PositionState(), PolicyLimits())
    assert decision.action is TradeAction.SKIP
    assert decision.reason_code == "STALE_SIGNAL"


def test_market_prices_are_bounded() -> None:
    with pytest.raises(ValueError, match="yes_ask"):
        MarketState(
            condition_id="bad",
            observed_at=NOW,
            yes_bid=0.5,
            yes_ask=1.2,
            no_bid=0.4,
            no_ask=0.5,
            available_depth_usd=100,
            seconds_to_resolution=1,
        )
