from __future__ import annotations

import math
from decimal import Decimal

import pytest

from sphinx_trace.polymarket_fees import (
    FeeFormula,
    FeeProtocol,
    FeeRounding,
    FeeScheduleEvidence,
)
from sphinx_trace.protocol_tail_pack import (
    calendar_week_id,
    protocol_action_targets,
    winning_payout_per_total_cost,
)


def _schedule(*, protocol: FeeProtocol, formula: FeeFormula, rate: str) -> FeeScheduleEvidence:
    return FeeScheduleEvidence(
        schedule_id="s" * 64,
        liquidity_id=None,
        transaction_hash=None,
        condition_id="condition",
        timestamp_unix=0,
        protocol=protocol,
        formula=formula,
        rate=Decimal(rate),
        exponent=1,
        taker_only=True,
        collateral_rounding_decimals=5,
        outcome_rounding_decimals=6,
        rounding=FeeRounding.HALF_UP,
        source="test",
        effective_from_unix=0,
        effective_to_unix=2_000_000_000,
    )


def test_v1_winning_payout_deducts_outcome_shares_not_collateral() -> None:
    schedule = _schedule(
        protocol=FeeProtocol.CLOB_V1,
        formula=FeeFormula.POLYMARKET_CURVE,
        rate="0.072",
    )

    multiplier = winning_payout_per_total_cost(
        schedule, entry_price="0.4", total_cost_usd="500"
    )

    gross_shares = Decimal("1250")
    fee_shares = Decimal("54")
    assert multiplier == (gross_shares - fee_shares) / Decimal("500")


def test_v1_payout_quote_rebases_terminal_decimal_division_dust() -> None:
    schedule = _schedule(
        protocol=FeeProtocol.CLOB_V1,
        formula=FeeFormula.POLYMARKET_CURVE,
        rate="0.072",
    )

    multiplier = winning_payout_per_total_cost(
        schedule, entry_price="0.59", total_cost_usd="500"
    )

    assert multiplier > Decimal(1)


def test_v2_winning_payout_includes_collateral_fee_in_total_cost() -> None:
    schedule = _schedule(
        protocol=FeeProtocol.CLOB_V2,
        formula=FeeFormula.POLYMARKET_CURVE,
        rate="0.07",
    )

    multiplier = winning_payout_per_total_cost(
        schedule, entry_price="0.5", total_cost_usd="500"
    )

    assert float(multiplier) < 2.0
    assert float(multiplier) == pytest.approx(1.93236, rel=1e-4)


def test_protocol_action_targets_preserve_skip_and_terminal_loss() -> None:
    schedule = FeeScheduleEvidence.zero(
        liquidity_id=None,
        transaction_hash=None,
        condition_id="condition",
        timestamp_unix=0,
        protocol=FeeProtocol.CLOB_V1,
        source="test",
    )
    targets = protocol_action_targets(
        schedule,
        market_probability_outcome0=0.6,
        label_outcome0=1.0,
        reference_size=0.05,
        reference_equity_usd=10_000.0,
        adverse_price_ticks=1,
        tick_size=0.01,
        minimum_entry_price=0.01,
    )

    assert targets.entry_prices == pytest.approx((0.61, 0.41))
    assert targets.reference_action_values[0] > 0.0
    assert targets.reference_action_values[1] == pytest.approx(math.log(0.95))
    assert targets.reference_action_values[2] == 0.0


def test_calendar_week_id_uses_utc_monday_boundary() -> None:
    assert calendar_week_id(345_600) == 345_600
    assert calendar_week_id(345_600 + 604_799) == 345_600
    assert calendar_week_id(345_600 + 604_800) == 950_400
