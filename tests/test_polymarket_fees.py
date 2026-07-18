from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from sphinx_corpus.io import sha256_file, write_jsonl_zst
from sphinx_trace.polymarket_fees import (
    FeeAsset,
    FeeFormula,
    FeeProtocol,
    FeeRounding,
    FeeScheduleBook,
    FeeScheduleEvidence,
    LiquidityRole,
    apply_polymarket_fee,
    fee_schedule_payload,
)


def _schedule(
    *,
    protocol: FeeProtocol = FeeProtocol.CLOB_V2,
    formula: FeeFormula = FeeFormula.POLYMARKET_CURVE,
    rate: str = "0.07",
    exponent: int = 1,
    rounding: FeeRounding = FeeRounding.HALF_UP,
) -> FeeScheduleEvidence:
    return FeeScheduleEvidence(
        schedule_id="a" * 64,
        liquidity_id="liquidity",
        transaction_hash="0x" + "b" * 64,
        condition_id="0x" + "c" * 64,
        timestamp_unix=100,
        protocol=protocol,
        formula=formula,
        rate=Decimal(rate),
        exponent=exponent,
        taker_only=True,
        collateral_rounding_decimals=5 if protocol == FeeProtocol.CLOB_V2 else 6,
        outcome_rounding_decimals=6,
        rounding=rounding,
        source="test",
    )


def test_v2_current_fee_vectors_and_maker_zero() -> None:
    crypto = _schedule()
    at_mid = apply_polymarket_fee(
        crypto,
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.5",
    )
    low = apply_polymarket_fee(
        crypto,
        side="SELL",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.3",
    )
    high = apply_polymarket_fee(
        crypto,
        side="SELL",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.7",
    )
    maker = apply_polymarket_fee(
        crypto,
        side="BUY",
        liquidity_role=LiquidityRole.MAKER,
        gross_shares="100",
        price="0.5",
    )

    assert at_mid.fee_asset == FeeAsset.COLLATERAL
    assert at_mid.collateral_fee_usd == Decimal("1.75000")
    assert low.collateral_fee_usd == high.collateral_fee_usd == Decimal("1.47000")
    assert maker.is_zero


@pytest.mark.parametrize(
    ("rate", "expected"),
    [("0.04", "1.00000"), ("0.05", "1.25000")],
)
def test_v2_category_midpoint_vectors(rate: str, expected: str) -> None:
    fee = apply_polymarket_fee(
        _schedule(rate=rate),
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.5",
    )
    assert fee.collateral_fee_usd == Decimal(expected)


def test_v2_rounds_to_five_decimals_and_subminimum_to_zero() -> None:
    below = apply_polymarket_fee(
        _schedule(rate="0.000016"),
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="1",
        price="0.5",
    )
    above = apply_polymarket_fee(
        _schedule(rate="0.000024"),
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="1",
        price="0.5",
    )
    assert below.fee_amount == Decimal("0.00000")
    assert above.fee_amount == Decimal("0.00001")


def test_v1_buy_deducts_outcome_and_sell_deducts_collateral() -> None:
    legacy = _schedule(
        protocol=FeeProtocol.CLOB_V1,
        formula=FeeFormula.V1_MIN_PRICE_CURVE,
        rate="0.02",
        exponent=0,
        rounding=FeeRounding.DOWN,
    )
    buy = apply_polymarket_fee(
        legacy,
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.5",
    )
    sell = apply_polymarket_fee(
        legacy,
        side="SELL",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares="100",
        price="0.5",
    )

    assert buy.fee_asset == FeeAsset.OUTCOME
    assert buy.outcome_fee_shares == Decimal("2.000000")
    assert buy.collateral_fee_usd == Decimal("0")
    assert buy.fee_value_usd == Decimal("1.0000000")
    assert sell.fee_asset == FeeAsset.COLLATERAL
    assert sell.collateral_fee_usd == Decimal("1.000000")


@pytest.mark.parametrize(
    ("shares", "price", "expected_fee_shares"),
    [
        ("150", "0.50", "2.343750"),
        # The aggregate active-taker receipt settles 0.000002 lower after its
        # constituent maker-fill rounding; one simulated public fill uses this value.
        ("1590.06", "0.59", "23.260832"),
    ],
)
def test_v1_operator_curve_reconciles_onchain_fee_refunds(
    shares: str,
    price: str,
    expected_fee_shares: str,
) -> None:
    historical_crypto = _schedule(
        protocol=FeeProtocol.CLOB_V1,
        formula=FeeFormula.V1_OUTPUT_ASSET_CURVE,
        rate="0.25",
        exponent=2,
        rounding=FeeRounding.DOWN,
    )

    fee = apply_polymarket_fee(
        historical_crypto,
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares=shares,
        price=price,
    )

    assert fee.fee_asset == FeeAsset.OUTCOME
    assert fee.outcome_fee_shares == Decimal(expected_fee_shares)
    assert fee.fee_value_usd == Decimal(expected_fee_shares) * Decimal(price)


def test_rate_stress_and_fail_closed_book() -> None:
    schedule = _schedule()
    book = FeeScheduleBook([schedule], manifest_sha256="d" * 64)
    stressed = book.quote(
        "liquidity",
        condition_id="0x" + "c" * 64,
        timestamp_unix=100,
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares=Decimal("100"),
        price=Decimal("0.5"),
        rate_multiplier=Decimal("2"),
    )
    assert stressed.collateral_fee_usd == Decimal("3.50000")
    with pytest.raises(KeyError, match="unqualified"):
        book.schedule_for("unknown")


def test_condition_time_interval_qualifies_unseen_liquidity() -> None:
    schedule = replace(
        _schedule(),
        liquidity_id="source-liquidity",
        effective_from_unix=90,
        effective_to_unix=110,
    )
    book = FeeScheduleBook([schedule], manifest_sha256="e" * 64)

    resolved = book.schedule_for(
        "unseen-fill",
        condition_id="0x" + "c" * 64,
        timestamp_unix=105,
        transaction_hash="0x" + "f" * 64,
    )

    assert resolved.schedule_id == schedule.schedule_id
    with pytest.raises(KeyError, match="unqualified"):
        book.schedule_for(
            "outside",
            condition_id="0x" + "c" * 64,
            timestamp_unix=110,
        )


def test_artifact_requires_hash_bound_receipt_proof_coverage(tmp_path: Path) -> None:
    schedule = _schedule()
    data_path = tmp_path / "schedules.jsonl.zst"
    proof_path = tmp_path / "source-receipts.jsonl.zst"
    write_jsonl_zst(data_path, [fee_schedule_payload(schedule)])
    write_jsonl_zst(
        proof_path,
        [
            {
                "schema_version": "1.0.0",
                "record_type": "h016_fee_receipt_proof",
                "schedule_id": schedule.schedule_id,
                "transaction_hash": schedule.transaction_hash,
                "status": "0x1",
            }
        ],
    )
    manifest = {
        "record_type": "h016_fee_schedule_manifest",
        "valid": True,
        "data_path": data_path.name,
        "data_sha256": sha256_file(data_path),
        "rows": 1,
        "receipt_proof_path": proof_path.name,
        "receipt_proof_sha256": sha256_file(proof_path),
        "receipt_proof_rows": 1,
        "test_labels_opened": False,
        "test_rows_consumed": 0,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    book = FeeScheduleBook.from_artifact(tmp_path)
    assert book.schedule_for("liquidity").schedule_id == schedule.schedule_id

    write_jsonl_zst(proof_path, [])
    manifest["receipt_proof_sha256"] = sha256_file(proof_path)
    manifest["receipt_proof_rows"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="proof coverage"):
        FeeScheduleBook.from_artifact(tmp_path)
