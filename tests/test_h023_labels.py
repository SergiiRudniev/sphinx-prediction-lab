from __future__ import annotations

from decimal import Decimal

from sphinx_trace.h023_labels import realized_decision_labels


def _decision(decision_id: str, action_id: int = 0) -> dict[str, object]:
    return {
        "record_type": "h010_decision_audit",
        "decision_id": decision_id,
        "condition_id": "condition",
        "action": f"CALL_OUTCOME_{action_id}",
        "outcome_labels": ["Yes", "No"],
        "h022_mode": "shadow",
        "h022": {"candidate_action_id": action_id},
    }


def _order(
    decision_id: str,
    order_id: str,
    *,
    outcome: str = "Yes",
    side: str = "BUY",
    shares: str = "10",
    price: str = "0.8",
) -> dict[str, object]:
    return {
        "record_type": "h010_order_audit",
        "decision_id": decision_id,
        "order_id": order_id,
        "condition_id": "condition",
        "outcome": outcome,
        "side": side,
        "requested_shares": shares,
        "limit_price": price,
    }


def _fill(
    order_id: str,
    *,
    side: str = "BUY",
    shares: str = "10",
    position_shares: str = "9.9",
    notional: str = "7.5",
    collateral_fee: str = "0",
    outcome_fee_shares: str = "0.1",
    fee_value: str = "0.075",
) -> dict[str, object]:
    return {
        "record_type": "h010_fill_audit",
        "order_id": order_id,
        "side": side,
        "shares": shares,
        "position_shares": position_shares,
        "notional_usd": notional,
        "collateral_fee_usd": collateral_fee,
        "outcome_fee_shares": outcome_fee_shares,
        "fee_usd": fee_value,
    }


def _resolution() -> dict[str, object]:
    return {
        "record_type": "h010_resolution_audit",
        "condition_id": "condition",
        "payouts": ["1", "0"],
    }


def test_h023_attributes_buy_fill_with_outcome_token_fee() -> None:
    labels, counts = realized_decision_labels(
        [
            _decision("buy"),
            _order("buy", "order"),
            _fill("order"),
            _resolution(),
        ]
    )

    label = labels["buy"]
    assert label.terminal_payout_usd == Decimal("9.9")
    assert label.realized_pnl_usd == Decimal("2.4")
    assert label.fill_fraction == Decimal("1")
    assert label.outcome_token_fee_shares == Decimal("0.1")
    assert counts["candidate_decisions"] == 1


def test_h023_attributes_sell_against_terminal_hold_value() -> None:
    labels, _ = realized_decision_labels(
        [
            _decision("sell"),
            _order("sell", "order", side="SELL"),
            _fill(
                "order",
                side="SELL",
                position_shares="10",
                notional="7.5",
                collateral_fee="0.1",
                outcome_fee_shares="0",
                fee_value="0.1",
            ),
            _resolution(),
        ]
    )

    label = labels["sell"]
    assert label.terminal_payout_usd == Decimal("-10")
    assert label.realized_pnl_usd == Decimal("-2.6")
    assert label.actual_filled_total_cost_usd == Decimal("7.5")


def test_h023_preserves_unfilled_call_as_zero_contribution() -> None:
    labels, _ = realized_decision_labels(
        [_decision("unfilled", 1), _order("unfilled", "order", outcome="No")]
    )

    label = labels["unfilled"]
    assert label.fill_fraction == Decimal("0")
    assert label.realized_pnl_usd == Decimal("0")
    assert label.realized_return_on_filled_cost == Decimal("0")


def test_h023_exact_veto_is_an_audited_zero_contribution() -> None:
    decision = _decision("veto")
    decision["action"] = "SKIP"
    decision["h023"] = {"keep_base_call": False}

    labels, _ = realized_decision_labels(
        [decision], require_action_matches_candidate=False
    )

    assert labels["veto"].realized_pnl_usd == Decimal("0")
