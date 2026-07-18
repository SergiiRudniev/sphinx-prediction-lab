from __future__ import annotations

from decimal import Decimal

from sphinx_trace.fee_schedule_h016 import (
    FeeSourceCandidate,
    OnchainFeeEvidence,
    infer_fee_schedule,
    official_zero_schedule,
)
from sphinx_trace.polymarket_fees import FeeAsset, FeeFormula, FeeProtocol


def _candidate() -> FeeSourceCandidate:
    return FeeSourceCandidate(
        condition_id="0x" + "a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        timestamp_unix=100,
        token_ids=("yes", "no"),
        effective_from_unix=90,
        effective_to_unix=110,
    )


def test_infer_historical_v1_crypto_curve_from_refund_adjusted_fee() -> None:
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash="0x" + "b" * 64,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("1590.06"),
        price=Decimal("0.59"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("23.260830"),
        constituent_maker_fills=7,
    )

    schedule = infer_fee_schedule(_candidate(), evidence)

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.25")
    assert schedule.exponent == 2
    assert schedule.source_fee_amount == Decimal("23.260830")
    assert schedule.effective_from == 90
    assert schedule.effective_to == 110


def test_infer_post_category_v1_usd_curve_converted_to_buy_shares() -> None:
    candidate = FeeSourceCandidate(
        condition_id="0x" + "a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        timestamp_unix=1_775_029_985,
        token_ids=("yes", "no"),
        effective_from_unix=1_775_029_900,
        effective_to_unix=1_775_030_100,
    )
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash=candidate.transaction_hash,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("200"),
        price=Decimal("0.04"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("13.824"),
        constituent_maker_fills=3,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.formula == FeeFormula.POLYMARKET_CURVE
    assert schedule.rate == Decimal("0.072")
    assert schedule.exponent == 1
    assert schedule.outcome_rounding_decimals == 5


def test_post_category_v1_curve_uses_five_decimal_operator_settlement() -> None:
    candidate = FeeSourceCandidate(
        condition_id="0x" + "a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        timestamp_unix=1_775_289_041,
        token_ids=("yes", "no"),
        effective_from_unix=1_775_289_000,
        effective_to_unix=1_775_289_100,
    )
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash=candidate.transaction_hash,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("25.38"),
        price=Decimal("0.99"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("0.01827"),
        constituent_maker_fills=1,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.rate == Decimal("0.072")
    assert schedule.source_fee_amount == Decimal("0.01827")


def test_v1_receipt_selects_output_asset_curve_without_date_imputation() -> None:
    candidate = FeeSourceCandidate(
        condition_id="0x" + "a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        timestamp_unix=1_774_893_593,
        token_ids=("yes", "no"),
        effective_from_unix=1_774_893_500,
        effective_to_unix=1_774_893_700,
    )
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash=candidate.transaction_hash,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("959.26"),
        price=Decimal("0.01"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("0.68376"),
        constituent_maker_fills=2,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.072")


def test_zero_receipt_and_official_pre_fee_schedule_remain_distinct_sources() -> None:
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash="0x" + "b" * 64,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="SELL",
        gross_shares=Decimal("100"),
        price=Decimal("0.4"),
        fee_asset=FeeAsset.COLLATERAL,
        fee_amount=Decimal("0"),
        constituent_maker_fills=1,
    )

    receipt_zero = infer_fee_schedule(_candidate(), evidence)
    chronology_zero = official_zero_schedule(
        condition_id=_candidate().condition_id,
        effective_from_unix=10,
        effective_to_unix=20,
        protocol=FeeProtocol.CLOB_V1,
    )

    assert receipt_zero.formula == chronology_zero.formula == FeeFormula.ZERO
    assert receipt_zero.transaction_hash is not None
    assert chronology_zero.transaction_hash is None
    assert receipt_zero.source != chronology_zero.source
