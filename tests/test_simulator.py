from __future__ import annotations

from decimal import Decimal

import pytest

from sphinx_trace.simulator import (
    LiquidityEvent,
    OrderSide,
    OrderStatus,
    ReplaySimulator,
    SimulationRules,
)


def _event(
    liquidity_id: str,
    timestamp: int,
    *,
    price: str = "0.50",
    shares: str = "8",
) -> LiquidityEvent:
    return LiquidityEvent(
        liquidity_id=liquidity_id,
        timestamp_unix=timestamp,
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        price=Decimal(price),
        shares=Decimal(shares),
    )


def test_partial_fill_latency_self_fill_guard_resolution_and_resume() -> None:
    rules = SimulationRules(
        initial_cash_usd=Decimal("100"),
        latency_seconds=2,
        maximum_fill_wait_seconds=10,
        available_share_fraction=Decimal("0.5"),
        duplicate_liquidity_haircut=Decimal("1"),
        adverse_price_ticks=1,
        tick_size=Decimal("0.01"),
        fee_bps=Decimal("100"),
    )
    simulator = ReplaySimulator(rules)
    simulator.record_prediction(
        decision_id="decision",
        timestamp_unix=100,
        action="CALL_YES",
        probability="0.8",
        size_fraction="0.1",
        input_sha256="ab" * 32,
    )
    order = simulator.place_order(
        decision_id="decision",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=100,
        requested_shares="10",
        limit_price="0.60",
        evidence_sha256="ab" * 32,
        evidence_liquidity_id="evidence",
    )

    assert simulator.process_liquidity(_event("before-latency", 101)) == []
    assert simulator.process_liquidity(_event("evidence", 102)) == []
    first = simulator.process_liquidity(_event("fill-1", 103))
    assert first[0].shares == Decimal("4.0")
    assert first[0].price == Decimal("0.51")
    assert order.status == OrderStatus.PARTIAL

    restored = ReplaySimulator.from_snapshot(simulator.snapshot())
    assert restored.checkpoint_sha256() == simulator.checkpoint_sha256()
    second = restored.process_liquidity(_event("fill-2", 104, shares="20"))
    assert second[0].shares == Decimal("6.0")
    assert restored.orders[order.order_id].status == OrderStatus.FILLED
    assert restored.positions["yes-token"].shares == Decimal("10.0")
    assert restored.cash_usd == Decimal("94.8490")

    pnl = restored.resolve(
        condition_id="condition",
        timestamp_unix=200,
        token_payouts={"yes-token": 1},
    )
    assert pnl == Decimal("4.8490")
    assert restored.cash_usd == Decimal("104.8490")
    assert restored.metrics()["net_profit_usd"] == pytest.approx(4.849)


def test_cash_share_constraints_sell_and_duplicate_liquidity_guard() -> None:
    rules = SimulationRules(
        initial_cash_usd=Decimal("100"),
        available_share_fraction=Decimal("1"),
        duplicate_liquidity_haircut=Decimal("1"),
        fee_bps=Decimal("100"),
    )
    simulator = ReplaySimulator(rules)
    rejected_buy = simulator.place_order(
        decision_id="too-large",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=1,
        requested_shares="1000",
        limit_price="0.9",
    )
    assert rejected_buy.status == OrderStatus.REJECTED
    assert rejected_buy.reject_reason == "INSUFFICIENT_CASH"

    buy = simulator.place_order(
        decision_id="buy",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=1,
        requested_shares="10",
        limit_price="0.5",
    )
    simulator.process_liquidity(_event("buy-fill", 3, price="0.40", shares="100"))
    assert buy.status == OrderStatus.FILLED

    sell = simulator.place_order(
        decision_id="sell",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.SELL,
        submitted_at_unix=4,
        requested_shares="5",
        limit_price="0.3",
    )
    rejected_sell = simulator.place_order(
        decision_id="sell-too-large",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.SELL,
        submitted_at_unix=4,
        requested_shares="6",
        limit_price="0.3",
    )
    assert rejected_sell.status == OrderStatus.REJECTED
    simulator.process_liquidity(_event("sell-fill", 6, price="0.50", shares="100"))
    assert sell.status == OrderStatus.FILLED
    assert simulator.positions["yes-token"].shares == Decimal("5")
    assert simulator.realized_pnl_usd == Decimal("0.3550")

    with pytest.raises(ValueError, match="replayed twice"):
        simulator.process_liquidity(_event("sell-fill", 6, price="0.50", shares="100"))
    with pytest.raises(ValueError, match="regressed"):
        simulator.cancel_order(buy.order_id, 5)


def test_unfilled_order_expires_without_invented_liquidity() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            latency_seconds=2,
            maximum_fill_wait_seconds=3,
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
        )
    )
    order = simulator.place_order(
        decision_id="decision",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=10,
        requested_shares="1",
        limit_price="0.4",
    )
    assert simulator.process_liquidity(_event("too-expensive", 12, price="0.9")) == []
    simulator.process_liquidity(_event("too-late", 16, price="0.1"))
    assert order.status == OrderStatus.EXPIRED
    assert simulator.fills == []
    assert simulator.cash_usd == Decimal("10000")
