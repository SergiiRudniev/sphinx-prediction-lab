from __future__ import annotations

from decimal import Decimal

import pytest

from sphinx_trace.replay_h010 import (
    BinaryMarketContract,
    H010ReplayAdapter,
    PolicyCall,
    ReplayCursor,
    SelectiveAction,
)
from sphinx_trace.simulator import ReplaySimulator, SimulationRules


def _contract() -> BinaryMarketContract:
    return BinaryMarketContract(
        condition_id="condition",
        component_id="component",
        outcomes=("Yes", "No"),
        token_ids=("yes-token", "no-token"),
    )


def _trade(trade_id: str, timestamp: int, price: str = "0.40") -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "condition_id": "condition",
        "token_id": "yes-token",
        "timestamp_unix": timestamp,
        "price": price,
        "size": "100",
        "side": "SELL",
    }


def _call(
    decision_id: str,
    evidence: str,
    timestamp: int,
    action: SelectiveAction,
    size: str,
) -> PolicyCall:
    return PolicyCall(
        decision_id=decision_id,
        timestamp_unix=timestamp,
        condition_id="condition",
        component_id="component",
        evidence_trade_id=evidence,
        action=action,
        probability_outcome0=Decimal("0.8"),
        size_fraction=Decimal(size),
        input_sha256="ab" * 32,
    )


def test_replay_applies_call_after_evidence_and_tracks_portfolio_memory() -> None:
    simulator = ReplaySimulator(
        SimulationRules(
            initial_cash_usd=Decimal("100"),
            latency_seconds=2,
            maximum_fill_wait_seconds=10,
            available_share_fraction=Decimal("1"),
            duplicate_liquidity_haircut=Decimal("1"),
            fee_bps=Decimal("0"),
            retain_processed_liquidity_ids=False,
            retain_prediction_records=False,
        )
    )
    adapter = H010ReplayAdapter(
        simulator, {"condition": _contract()}, source_sha256="cd" * 32
    )
    orders = adapter.process_trade(
        _trade("evidence", 100),
        (_call("call", "evidence", 100, SelectiveAction.CALL_OUTCOME_0, "0.10"),),
        shard_ordinal=0,
        row_ordinal=0,
    )
    assert len(orders) == 1
    assert simulator.fills == []
    adapter.process_trade(_trade("before-latency", 101), shard_ordinal=0, row_ordinal=1)
    adapter.process_trade(_trade("fill", 102), shard_ordinal=0, row_ordinal=2)
    assert simulator.positions["yes-token"].cost_basis_usd == pytest.approx(10)
    assert adapter.physical_action_mask("condition") == (
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    action_id, memory = adapter.prediction_memory_features("condition", 110)
    assert action_id == 0
    assert memory[0] == pytest.approx(0.8)
    assert memory[3] == 1.0
    assert len(adapter.portfolio_features()) == 9

    restored = H010ReplayAdapter.from_snapshot(
        adapter.snapshot(),
        {"condition": _contract()},
        source_sha256="cd" * 32,
    )
    assert restored.checkpoint_sha256() == adapter.checkpoint_sha256()
    adapter = restored

    close_orders = adapter.process_trade(
        _trade("close-evidence", 103, "0.50"),
        (_call("close", "close-evidence", 103, SelectiveAction.CLOSE, "1"),),
        shard_ordinal=0,
        row_ordinal=3,
    )
    assert len(close_orders) == 1
    adapter.process_trade(
        _trade("close-fill", 105, "0.50"), shard_ordinal=0, row_ordinal=4
    )
    assert adapter.simulator.positions == {}
    assert adapter.simulator.metrics()["net_profit_usd"] == pytest.approx(1.9512195122)


def test_replay_cursor_and_evidence_contract_reject_reuse() -> None:
    cursor = ReplayCursor("ef" * 32).advance(0, 0)
    with pytest.raises(ValueError, match="did not advance"):
        cursor.advance(0, 0)
    adapter = H010ReplayAdapter(
        ReplaySimulator(SimulationRules()),
        {"condition": _contract()},
        source_sha256="ef" * 32,
    )
    with pytest.raises(ValueError, match="evidence trade"):
        adapter.process_trade(
            _trade("actual", 100),
            (_call("call", "different", 100, SelectiveAction.SKIP, "0"),),
            shard_ordinal=0,
            row_ordinal=0,
        )


def test_observe_then_infer_boundary_exposes_post_trade_state() -> None:
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(
                retain_processed_liquidity_ids=False,
                retain_prediction_records=False,
            )
        ),
        {"condition": _contract()},
        source_sha256="ab" * 32,
    )

    prices = adapter.observe_trade(
        _trade("evidence", 100, "0.65"),
        shard_ordinal=0,
        row_ordinal=0,
    )
    portfolio = adapter.portfolio_features()
    memory = adapter.prediction_memory_features("condition", 100)
    orders = adapter.apply_call(
        _call("call", "evidence", 100, SelectiveAction.CALL_OUTCOME_0, "0.1"),
        prices,
    )

    assert prices == {"yes-token": Decimal("0.65"), "no-token": Decimal("0.35")}
    assert portfolio[0:2] == (1.0, 1.0)
    assert memory[1][0] == pytest.approx(0.5)
    assert len(orders) == 1


def test_h021_candidate_execution_context_uses_executable_limits() -> None:
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(
                adverse_price_ticks=1,
                tick_size=Decimal("0.01"),
                fee_bps=Decimal("0"),
            )
        ),
        {"condition": _contract()},
        source_sha256="ab" * 32,
    )

    context = adapter.candidate_execution_context(
        "condition",
        100,
        "evidence",
        {"yes-token": Decimal("0.99"), "no-token": Decimal("0.01")},
    )

    assert context == (
        Decimal("1.00"),
        Decimal("0.02"),
        Decimal("1"),
        Decimal("5E+1"),
    )
