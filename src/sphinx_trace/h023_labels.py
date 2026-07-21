"""Decision-level realized economic labels for H023 shadow replays."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

ZERO = Decimal("0")


def _decimal(value: object) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError("H023 audit contains a non-finite decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class H023RealizedLabel:
    """Additive terminal contribution of one H021 CALL decision."""

    decision_id: str
    condition_id: str
    requested_total_cost_usd: Decimal
    actual_filled_total_cost_usd: Decimal
    requested_shares: Decimal
    actual_gross_filled_shares: Decimal
    actual_position_shares: Decimal
    fill_fraction: Decimal
    outcome_token_fee_shares: Decimal
    collateral_fee_usd: Decimal
    fee_value_usd: Decimal
    terminal_payout_usd: Decimal
    realized_pnl_usd: Decimal
    realized_return_on_requested_cost: Decimal
    realized_return_on_filled_cost: Decimal
    order_count: int
    fill_count: int

    def payload(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }


@dataclass(slots=True)
class _DecisionAccumulator:
    condition_id: str
    requested_total_cost_usd: Decimal = ZERO
    actual_filled_total_cost_usd: Decimal = ZERO
    requested_shares: Decimal = ZERO
    actual_gross_filled_shares: Decimal = ZERO
    actual_position_shares: Decimal = ZERO
    outcome_token_fee_shares: Decimal = ZERO
    collateral_fee_usd: Decimal = ZERO
    fee_value_usd: Decimal = ZERO
    terminal_payout_usd: Decimal = ZERO
    cash_flow_usd: Decimal = ZERO
    order_count: int = 0
    fill_count: int = 0
    payout_units: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _OrderRef:
    decision_id: str
    condition_id: str
    outcome: str
    side: str


def realized_decision_labels(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, H023RealizedLabel], dict[str, int]]:
    """Aggregate strict replay audits into fill- and fee-realized labels.

    BUY contribution is terminal payout minus the actual collateral spend. SELL
    contribution is actual proceeds minus the terminal value of shares sold. The
    two definitions are additive and express the marginal economic effect of the
    order versus doing nothing and holding the pre-decision portfolio unchanged.
    """

    decisions: dict[str, _DecisionAccumulator] = {}
    orders: dict[str, _OrderRef] = {}
    condition_outcomes: dict[str, tuple[str, ...]] = {}
    resolutions: dict[str, tuple[Decimal, ...]] = {}
    counts = {
        "audit_rows": 0,
        "candidate_decisions": 0,
        "orders": 0,
        "fills": 0,
        "resolutions": 0,
    }
    for row in rows:
        counts["audit_rows"] += 1
        record_type = str(row.get("record_type"))
        if record_type == "h010_decision_audit":
            h022 = row.get("h022")
            if h022 is None:
                continue
            if row.get("h022_mode") != "shadow" or not isinstance(h022, dict):
                raise RuntimeError("H023 requires an H022 shadow replay")
            action = str(row.get("action"))
            action_id = int(h022.get("candidate_action_id", -1))
            expected = f"CALL_OUTCOME_{action_id}"
            if action_id not in (0, 1) or action != expected:
                raise RuntimeError("H023 shadow decision changed the H021 CALL")
            decision_id = str(row["decision_id"])
            condition_id = str(row["condition_id"]).lower()
            outcomes = tuple(str(value) for value in row["outcome_labels"])
            if len(outcomes) != 2:
                raise RuntimeError("H023 requires binary outcome labels")
            previous_outcomes = condition_outcomes.get(condition_id)
            if previous_outcomes is not None and previous_outcomes != outcomes:
                raise RuntimeError("H023 condition outcome labels changed")
            condition_outcomes[condition_id] = outcomes
            if decision_id in decisions:
                raise RuntimeError(f"H023 candidate decision repeats: {decision_id}")
            decisions[decision_id] = _DecisionAccumulator(condition_id)
            counts["candidate_decisions"] += 1
        elif record_type == "h010_order_audit":
            decision_id = str(row["decision_id"])
            if decision_id not in decisions:
                continue
            order_id = str(row["order_id"])
            if order_id in orders:
                raise RuntimeError(f"H023 order repeats: {order_id}")
            condition_id = str(row["condition_id"]).lower()
            accumulator = decisions[decision_id]
            if condition_id != accumulator.condition_id:
                raise RuntimeError("H023 order condition changed")
            side = str(row["side"])
            if side not in {"BUY", "SELL"}:
                raise RuntimeError("H023 order side changed")
            requested_shares = _decimal(row["requested_shares"])
            limit_price = _decimal(row["limit_price"])
            if requested_shares <= ZERO or not ZERO <= limit_price <= Decimal("1"):
                raise RuntimeError("H023 order economics are invalid")
            accumulator.requested_shares += requested_shares
            accumulator.requested_total_cost_usd += requested_shares * limit_price
            accumulator.order_count += 1
            orders[order_id] = _OrderRef(
                decision_id,
                condition_id,
                str(row["outcome"]),
                side,
            )
            counts["orders"] += 1
        elif record_type == "h010_fill_audit":
            order = orders.get(str(row["order_id"]))
            if order is None:
                continue
            side = str(row["side"])
            if side != order.side:
                raise RuntimeError("H023 fill side changed")
            shares = _decimal(row["shares"])
            position_shares = _decimal(row["position_shares"])
            notional = _decimal(row["notional_usd"])
            collateral_fee = _decimal(row["collateral_fee_usd"])
            fee_value = _decimal(row["fee_usd"])
            outcome_fee_shares = _decimal(row["outcome_fee_shares"])
            if min(shares, position_shares, notional, collateral_fee, fee_value) < ZERO:
                raise RuntimeError("H023 fill economics are negative")
            accumulator = decisions[order.decision_id]
            accumulator.actual_gross_filled_shares += shares
            accumulator.actual_position_shares += position_shares
            accumulator.outcome_token_fee_shares += outcome_fee_shares
            accumulator.collateral_fee_usd += collateral_fee
            accumulator.fee_value_usd += fee_value
            accumulator.fill_count += 1
            if side == "BUY":
                total_cost = notional + collateral_fee
                accumulator.actual_filled_total_cost_usd += total_cost
                accumulator.cash_flow_usd -= total_cost
            else:
                # The absolute notional is the capital affected by a SELL. The
                # signed cash flow below carries its realized economic direction.
                accumulator.actual_filled_total_cost_usd += notional
                accumulator.cash_flow_usd += notional - collateral_fee
            orders[str(row["order_id"])] = _OrderRef(
                order.decision_id,
                order.condition_id,
                order.outcome,
                order.side,
            )
            # Store signed payout units on a synthetic outcome key. Resolution is
            # deliberately applied after the full stream has been read.
            signed_units = position_shares if side == "BUY" else -shares
            accumulator.payout_units[order.outcome] = (
                accumulator.payout_units.get(order.outcome, ZERO) + signed_units
            )
            counts["fills"] += 1
        elif record_type == "h010_resolution_audit":
            condition_id = str(row["condition_id"]).lower()
            payouts = tuple(_decimal(value) for value in row["payouts"])
            previous = resolutions.get(condition_id)
            if previous is not None and previous != payouts:
                raise RuntimeError("H023 condition resolution repeats differently")
            resolutions[condition_id] = payouts
            counts["resolutions"] += int(previous is None)

    for accumulator in decisions.values():
        if accumulator.payout_units:
            payout_values = resolutions.get(accumulator.condition_id)
            resolved_outcomes = condition_outcomes.get(accumulator.condition_id)
            if payout_values is None or resolved_outcomes is None:
                raise RuntimeError(
                    f"H023 filled condition has no resolution: {accumulator.condition_id}"
                )
            if len(payout_values) != len(resolved_outcomes):
                raise RuntimeError("H023 resolution width changed")
            resolution = dict(zip(resolved_outcomes, payout_values, strict=True))
            for outcome, units in accumulator.payout_units.items():
                if outcome not in resolution:
                    raise RuntimeError(
                        f"H023 resolution has no payout for outcome: {outcome}"
                    )
                accumulator.terminal_payout_usd += units * resolution[outcome]

    labels: dict[str, H023RealizedLabel] = {}
    for decision_id, accumulator in decisions.items():
        requested = accumulator.requested_total_cost_usd
        filled = accumulator.actual_filled_total_cost_usd
        fill_fraction = (
            min(Decimal("1"), accumulator.actual_gross_filled_shares / accumulator.requested_shares)
            if accumulator.requested_shares > ZERO
            else ZERO
        )
        realized = accumulator.cash_flow_usd + accumulator.terminal_payout_usd
        labels[decision_id] = H023RealizedLabel(
            decision_id=decision_id,
            condition_id=accumulator.condition_id,
            requested_total_cost_usd=requested,
            actual_filled_total_cost_usd=filled,
            requested_shares=accumulator.requested_shares,
            actual_gross_filled_shares=accumulator.actual_gross_filled_shares,
            actual_position_shares=accumulator.actual_position_shares,
            fill_fraction=fill_fraction,
            outcome_token_fee_shares=accumulator.outcome_token_fee_shares,
            collateral_fee_usd=accumulator.collateral_fee_usd,
            fee_value_usd=accumulator.fee_value_usd,
            terminal_payout_usd=accumulator.terminal_payout_usd,
            realized_pnl_usd=realized,
            realized_return_on_requested_cost=(realized / requested if requested > ZERO else ZERO),
            realized_return_on_filled_cost=(realized / filled if filled > ZERO else ZERO),
            order_count=accumulator.order_count,
            fill_count=accumulator.fill_count,
        )
    return labels, counts
