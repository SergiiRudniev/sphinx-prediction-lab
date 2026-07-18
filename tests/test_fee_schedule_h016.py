from __future__ import annotations

from decimal import Decimal

from sphinx_trace.fee_schedule_h016 import (
    FeeSourceCandidate,
    OnchainFeeEvidence,
    infer_fee_schedule,
    infer_fee_schedule_consensus,
    infer_fee_schedule_from_market_info,
    official_zero_schedule,
)
from sphinx_trace.polymarket_fees import FeeAsset, FeeFormula, FeeProtocol


def _candidate(*, timestamp_unix: int = 100) -> FeeSourceCandidate:
    return FeeSourceCandidate(
        condition_id="0x" + "a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        timestamp_unix=timestamp_unix,
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
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule(_candidate(timestamp_unix=1_774_893_593), evidence)

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.25")
    assert schedule.exponent == 2
    assert schedule.source_fee_amount == Decimal("23.260830")
    assert schedule.effective_from == 90
    assert schedule.effective_to == 110
    assert schedule.outcome_rounding_decimals == 5


def test_infer_pre_category_v1_four_decimal_operator_fee() -> None:
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash="0x" + "b" * 64,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("50"),
        price=Decimal("0.97"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("0.0105"),
        constituent_maker_fills=1,
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule(
        _candidate(timestamp_unix=1_772_390_119),
        evidence,
    )

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.25")
    assert schedule.exponent == 2
    assert schedule.outcome_rounding_decimals == 4


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
        fee_refund_observed=True,
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
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.rate == Decimal("0.072")
    assert schedule.source_fee_amount == Decimal("0.01827")


def test_post_cutover_precision_resolves_small_fill_tariff_alias() -> None:
    candidate = _candidate(timestamp_unix=1_775_335_215)
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash=candidate.transaction_hash,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("50"),
        price=Decimal("0.999"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("0.0015"),
        constituent_maker_fills=1,
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.formula == FeeFormula.POLYMARKET_CURVE
    assert schedule.rate == Decimal("0.03")
    assert schedule.outcome_rounding_decimals == 5


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
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule(candidate, evidence)

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.072")


def test_market_info_resolves_historical_unregistered_rate_and_formula() -> None:
    candidate = _candidate(timestamp_unix=1_772_390_119)
    evidence = OnchainFeeEvidence(
        protocol=FeeProtocol.CLOB_V1,
        transaction_hash=candidate.transaction_hash,
        block_number=1,
        log_index=2,
        order_hash="0x" + "c" * 64,
        side="BUY",
        gross_shares=Decimal("100"),
        price=Decimal("0.2"),
        fee_asset=FeeAsset.OUTCOME,
        fee_amount=Decimal("0.28"),
        constituent_maker_fills=1,
        fee_refund_observed=True,
    )

    schedule = infer_fee_schedule_from_market_info(
        candidate,
        evidence,
        {
            "c": candidate.condition_id,
            "t": [{"t": "yes"}, {"t": "no"}],
            "fd": {"r": 0.0175, "e": 1, "to": True},
        },
    )

    assert schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
    assert schedule.rate == Decimal("0.0175")
    assert schedule.exponent == 1
    assert schedule.source.startswith("clob_market_info")


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


def test_independent_receipts_resolve_rounding_ambiguity_by_consensus() -> None:
    condition_id = "0x" + "a" * 64
    samples = []
    for index, (shares, price, fee) in enumerate(
        (
            ("5", "0.98", "0.003"),
            ("50", "0.999", "0.0015"),
            ("5", "0.99", "0.0015"),
            ("5", "0.999", "0.00015"),
            ("5", "0.999", "0.00015"),
        )
    ):
        transaction_hash = "0x" + f"{index + 1:064x}"
        candidate = FeeSourceCandidate(
            condition_id=condition_id,
            liquidity_id=f"liquidity-{index}",
            transaction_hash=transaction_hash,
            timestamp_unix=1_775_296_800 + index,
            token_ids=("yes", "no"),
            effective_from_unix=1_775_296_700,
            effective_to_unix=1_775_296_900,
        )
        evidence = OnchainFeeEvidence(
            protocol=FeeProtocol.CLOB_V1,
            transaction_hash=transaction_hash,
            block_number=index + 1,
            log_index=2,
            order_hash="0x" + f"{index + 10:064x}",
            side="BUY",
            gross_shares=Decimal(shares),
            price=Decimal(price),
            fee_asset=FeeAsset.OUTCOME,
            fee_amount=Decimal(fee),
            constituent_maker_fills=1,
            fee_refund_observed=True,
        )
        samples.append((candidate, evidence))

    schedule = infer_fee_schedule_consensus(
        samples,
        {
            "c": condition_id,
            "t": [{"t": "yes"}, {"t": "no"}],
            "fd": {"r": 0.05, "e": 1, "to": True},
        },
    )

    assert schedule.formula == FeeFormula.POLYMARKET_CURVE
    assert schedule.rate == Decimal("0.03")
    assert schedule.exponent == 1
    assert schedule.outcome_rounding_decimals == 5
    assert "receipt_consensus" in schedule.source
