from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest

from sphinx_trace.polymarket_fees import (
    FeeAsset,
    FeeFormula,
    FeeProtocol,
    FeeRounding,
    FeeScheduleBook,
    FeeScheduleEvidence,
)
from sphinx_trace.simulator import (
    LiquidityEvent,
    OrderSide,
    OrderStatus,
    ReplaySimulator,
    SimulatedOrder,
    SimulationRules,
)


def _fee_schedule(
    liquidity_id: str,
    timestamp: int,
    *,
    protocol: FeeProtocol,
) -> FeeScheduleEvidence:
    is_v1 = protocol == FeeProtocol.CLOB_V1
    return FeeScheduleEvidence(
        schedule_id=(f"{timestamp:064x}"),
        liquidity_id=liquidity_id,
        transaction_hash="0x" + f"{timestamp:064x}",
        condition_id="condition",
        timestamp_unix=timestamp,
        protocol=protocol,
        formula=(FeeFormula.V1_MIN_PRICE_CURVE if is_v1 else FeeFormula.POLYMARKET_CURVE),
        rate=Decimal("0.02" if is_v1 else "0.07"),
        exponent=0 if is_v1 else 1,
        taker_only=True,
        collateral_rounding_decimals=6 if is_v1 else 5,
        outcome_rounding_decimals=6,
        rounding=FeeRounding.DOWN if is_v1 else FeeRounding.HALF_UP,
        source="test",
    )


class _IterationOrderedSet(set[str]):
    def __init__(self, values: list[str]) -> None:
        super().__init__(values)
        self._values = tuple(values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)


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
    assert restored.total_cost_basis_usd() == Decimal("5.1510")
    assert restored.marked_exposure_usd() == Decimal("5.00")
    assert restored.positions_for_condition("condition") == (
        restored.positions["yes-token"],
    )

    pnl = restored.resolve(
        condition_id="condition",
        timestamp_unix=200,
        token_payouts={"yes-token": 1},
    )
    assert pnl == Decimal("4.8490")
    assert restored.cash_usd == Decimal("104.8490")
    assert restored.pop_condition_realized_pnl("condition") == Decimal("4.8490")
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
    assert simulator.total_cost_basis_usd() == Decimal("2.0705")
    assert simulator.marked_exposure_usd() == Decimal("2.50")

    with pytest.raises(ValueError, match="replayed twice"):
        simulator.process_liquidity(_event("sell-fill", 6, price="0.50", shares="100"))
    with pytest.raises(ValueError, match="regressed"):
        simulator.cancel_order(buy.order_id, 5)


def test_affordable_fill_rebases_only_decimal_fee_dust() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            initial_cash_usd=Decimal("1"),
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
            adverse_price_ticks=0,
            fee_bps=Decimal("100"),
        )
    )
    order = simulator.place_order(
        decision_id="cash-boundary",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=1,
        requested_shares="2",
        limit_price="0.016",
    )
    simulator.cash_usd = Decimal("0.02")

    fills = simulator.process_liquidity(
        _event("cash-boundary-fill", 3, price="0.016", shares="10")
    )

    assert len(fills) == 1
    assert order.status == OrderStatus.PARTIAL
    assert fills[0].notional_usd + fills[0].fee_usd == Decimal("0.02")
    assert simulator.cash_usd == Decimal("0")


def test_decimal_reservations_ignore_set_iteration_order() -> None:
    rules = SimulationRules(initial_cash_usd=Decimal("2e28"), fee_bps=Decimal("0"))
    simulator = ReplaySimulator(rules)
    reservations = {
        "a": Decimal("1e28"),
        "b": Decimal("6"),
        "c": Decimal("6"),
    }
    for order_id, shares in reservations.items():
        simulator.orders[order_id] = SimulatedOrder(
            order_id=order_id,
            decision_id=f"decision-{order_id}",
            component_id="component",
            condition_id="condition",
            token_id="token",
            outcome="Yes",
            side=OrderSide.BUY,
            submitted_at_unix=0,
            eligible_at_unix=0,
            expires_at_unix=60,
            requested_shares=shares,
            limit_price=Decimal("1"),
        )

    simulator._open_order_ids = _IterationOrderedSet(["a", "b", "c"])
    forward = simulator.available_cash_usd()
    simulator._open_order_ids = _IterationOrderedSet(["b", "c", "a"])
    reordered = simulator.available_cash_usd()

    assert forward == reordered


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


def test_streaming_retention_stays_bounded_on_irrelevant_tape() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            retain_processed_liquidity_ids=False,
            retain_prediction_records=False,
        )
    )
    simulator.record_prediction(
        decision_id="skip",
        timestamp_unix=1,
        action="SKIP",
        probability="0.5",
        size_fraction="0",
        input_sha256="ab" * 32,
    )
    for timestamp in range(2, 102):
        simulator.process_liquidity(_event(f"event-{timestamp}", timestamp))

    assert simulator.predictions == []
    assert simulator.prediction_count == 1
    assert simulator.processed_liquidity_ids == set()
    assert simulator.processed_liquidity_count == 100
    assert simulator.last_liquidity_id == "event-101"
    assert simulator.last_marks == {}
    assert simulator.equity_curve == [(0, Decimal("10000"))]
    assert simulator.metrics()["liquidity_events"] == 100

    restored = ReplaySimulator.from_snapshot(simulator.snapshot())
    assert restored.checkpoint_sha256() == simulator.checkpoint_sha256()
    assert restored.prediction_count == 1
    assert restored.processed_liquidity_count == 100


def test_compacted_audit_history_preserves_metrics_and_checkpoint() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            initial_cash_usd=Decimal("100"),
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
            fee_bps=Decimal("0"),
        )
    )
    order = simulator.place_order(
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
    simulator.process_liquidity(_event("fill", 3, price="0.4", shares="100"))
    simulator.resolve(condition_id="condition", timestamp_unix=10, token_payouts={"yes-token": 1})
    before = simulator.metrics()

    compacted = simulator.compact_history()
    restored = ReplaySimulator.from_snapshot(simulator.snapshot())

    assert order.status == OrderStatus.FILLED
    assert compacted == {"orders": 1, "fills": 1, "closed_pnls": 1}
    assert simulator.orders == {}
    assert simulator.fills == []
    assert simulator.closed_pnls == []
    assert simulator.metrics() == before
    assert restored.metrics() == before
    assert restored.checkpoint_sha256() == simulator.checkpoint_sha256()


def test_compaction_discards_stale_expiry_entries() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            initial_cash_usd=Decimal("100"),
            maximum_fill_wait_seconds=100,
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
            adverse_price_ticks=0,
            fee_bps=Decimal("0"),
        )
    )
    order = simulator.place_order(
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
    simulator.process_liquidity(_event("fill", 3, price="0.4", shares="100"))

    simulator.compact_history()
    simulator.process_liquidity(_event("after-expiry", 200))

    assert order.status == OrderStatus.FILLED
    assert simulator.orders == {}
    assert simulator.current_time_unix == 200


@pytest.mark.parametrize(
    ("protocol", "expected_cash", "expected_position", "expected_fee"),
    [
        (FeeProtocol.CLOB_V1, "50.0", "98.000000", "1.0000000"),
        (FeeProtocol.CLOB_V2, "48.25000", "100", "1.75000"),
    ],
)
def test_protocol_fee_buy_changes_the_correct_portfolio_asset(
    protocol: FeeProtocol,
    expected_cash: str,
    expected_position: str,
    expected_fee: str,
) -> None:
    book = FeeScheduleBook(
        [
            _fee_schedule("evidence", 1, protocol=protocol),
            _fee_schedule("fill", 3, protocol=protocol),
        ],
        manifest_sha256="f" * 64,
    )
    simulator = ReplaySimulator(
        SimulationRules(
            initial_cash_usd=Decimal("100"),
            latency_seconds=0,
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
            adverse_price_ticks=0,
            fee_bps=Decimal("0"),
        ),
        fee_schedule_book=book,
    )
    order = simulator.place_order(
        decision_id="buy",
        component_id="component",
        condition_id="condition",
        token_id="yes-token",
        outcome="YES",
        side=OrderSide.BUY,
        submitted_at_unix=1,
        requested_shares="100",
        limit_price="0.5",
        evidence_liquidity_id="evidence",
    )
    fills = simulator.process_liquidity(_event("fill", 3, price="0.5", shares="100"))

    assert order.status == OrderStatus.FILLED
    assert simulator.cash_usd == Decimal(expected_cash)
    assert simulator.positions["yes-token"].shares == Decimal(expected_position)
    assert simulator.total_fees_usd == Decimal(expected_fee)
    assert fills[0].fee_asset == (
        FeeAsset.OUTCOME if protocol == FeeProtocol.CLOB_V1 else FeeAsset.COLLATERAL
    )
    assert fills[0].position_shares == Decimal(expected_position)

    restored = ReplaySimulator.from_snapshot(
        simulator.snapshot(),
        fee_schedule_book=book,
    )
    assert restored.checkpoint_sha256() == simulator.checkpoint_sha256()
    with pytest.raises(RuntimeError, match="fee schedule binding"):
        ReplaySimulator.from_snapshot(simulator.snapshot())


def test_protocol_fee_schedule_fails_closed_during_order_reservation() -> None:
    book = FeeScheduleBook(
        [_fee_schedule("qualified", 1, protocol=FeeProtocol.CLOB_V2)],
        manifest_sha256="e" * 64,
    )
    simulator = ReplaySimulator(
        SimulationRules(fee_bps=Decimal("0")),
        fee_schedule_book=book,
    )

    with pytest.raises(KeyError, match="unqualified"):
        simulator.place_order(
            decision_id="buy",
            component_id="component",
            condition_id="condition",
            token_id="yes-token",
            outcome="YES",
            side=OrderSide.BUY,
            submitted_at_unix=1,
            requested_shares="1",
            limit_price="0.5",
            evidence_liquidity_id="missing",
        )
