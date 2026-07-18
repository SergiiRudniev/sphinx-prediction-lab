"""Receipt-derived historical fee evidence for H016 Polymarket replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sphinx_corpus.config import ExchangeContract
from sphinx_corpus.ledger import decode_order_fill, event_topic
from sphinx_trace.polymarket_fees import (
    FeeAsset,
    FeeFormula,
    FeeProtocol,
    FeeRounding,
    FeeScheduleEvidence,
    LiquidityRole,
    apply_polymarket_fee,
)

ZERO = Decimal("0")
TOKEN_SCALE = Decimal("1000000")
FEE_REFUNDED_TOPIC = event_topic(
    "FeeRefunded(bytes32,address,uint256,uint256,uint256)"
)
KNOWN_CURVE_SCHEDULES = (
    (Decimal("0.25"), 2),
    (Decimal("0.072"), 1),
    (Decimal("0.07"), 1),
    (Decimal("0.05"), 1),
    (Decimal("0.04"), 1),
    (Decimal("0.03125"), 1),
    (Decimal("0.03"), 1),
)


@dataclass(frozen=True, slots=True)
class FeeSourceCandidate:
    condition_id: str
    liquidity_id: str
    transaction_hash: str
    timestamp_unix: int
    token_ids: tuple[str, str]
    effective_from_unix: int
    effective_to_unix: int


@dataclass(frozen=True, slots=True)
class OnchainFeeEvidence:
    protocol: FeeProtocol
    transaction_hash: str
    block_number: int
    log_index: int
    order_hash: str
    side: str
    gross_shares: Decimal
    price: Decimal
    fee_asset: FeeAsset
    fee_amount: Decimal
    constituent_maker_fills: int


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _refunds(receipt: dict[str, Any]) -> dict[str, int]:
    refunds: dict[str, int] = {}
    for log in receipt.get("logs", []):
        if not isinstance(log, dict):
            continue
        topics = log.get("topics")
        if (
            not isinstance(topics, list)
            or len(topics) != 4
            or str(topics[0]).lower() != FEE_REFUNDED_TOPIC
        ):
            continue
        refunds[str(topics[1]).lower()] = int(str(topics[3]), 16)
    return refunds


def decode_onchain_fee_evidence(
    receipt: dict[str, Any],
    candidate: FeeSourceCandidate,
    contracts: tuple[ExchangeContract, ...],
    *,
    chain_id: int,
) -> OnchainFeeEvidence:
    """Find the active taker fill and its net V1 refund-adjusted fee."""

    if str(receipt.get("status")) != "0x1":
        raise RuntimeError("Fee source transaction did not succeed")
    if str(receipt.get("transactionHash", "")).lower() != candidate.transaction_hash:
        raise RuntimeError("Fee source receipt transaction changed")
    by_address = {contract.address.lower(): contract for contract in contracts}
    refunds = _refunds(receipt)
    decoded: list[tuple[dict[str, Any], ExchangeContract]] = []
    for log in receipt.get("logs", []):
        if not isinstance(log, dict):
            continue
        contract = by_address.get(str(log.get("address", "")).lower())
        topics = log.get("topics")
        if (
            contract is None
            or not isinstance(topics, list)
            or not topics
            or str(topics[0]).lower() != event_topic(contract.event_signature)
        ):
            continue
        decoded.append(
            (
                decode_order_fill(
                    log,
                    contract,
                    chain_id=chain_id,
                    block_timestamp=candidate.timestamp_unix,
                ),
                contract,
            )
        )
    token_ids = set(candidate.token_ids)
    active = [
        (row, contract)
        for row, contract in decoded
        if row["taker"] == contract.address.lower() and row["token_id"] in token_ids
    ]
    if len(active) != 1:
        raise RuntimeError(
            f"Fee source has {len(active)} active fills for condition {candidate.condition_id}"
        )
    row, contract = active[0]
    price = Decimal(str(row["price"]))
    if not ZERO < price < Decimal("1"):
        raise RuntimeError("Fee source active fill price is outside (0, 1)")
    gross_shares = Decimal(str(row["token_amount_raw"])) / TOKEN_SCALE
    gross_fee_raw = int(str(row["fee_raw"]))
    actual_fee_raw = refunds.get(str(row["order_hash"]), gross_fee_raw)
    protocol = FeeProtocol(str(row["protocol"]))
    fee_asset = (
        FeeAsset.OUTCOME
        if protocol == FeeProtocol.CLOB_V1 and row["side"] == "BUY"
        else FeeAsset.COLLATERAL
    )
    constituent_fills = sum(
        row_value["taker"] == contract.address.lower()
        or row_value["taker"] != row_value["exchange_address"]
        for row_value, contract_value in decoded
        if contract_value.address.lower() == contract.address.lower()
        and row_value["token_id"] in token_ids
    )
    return OnchainFeeEvidence(
        protocol=protocol,
        transaction_hash=candidate.transaction_hash,
        block_number=int(str(receipt["blockNumber"]), 16),
        log_index=int(row["log_index"]),
        order_hash=str(row["order_hash"]),
        side=str(row["side"]),
        gross_shares=gross_shares,
        price=price,
        fee_asset=fee_asset,
        fee_amount=Decimal(actual_fee_raw) / TOKEN_SCALE,
        constituent_maker_fills=max(1, constituent_fills - 1),
    )


def _candidate_schedule(
    candidate: FeeSourceCandidate,
    evidence: OnchainFeeEvidence,
    *,
    rate: Decimal,
    exponent: int,
    formula: FeeFormula,
) -> FeeScheduleEvidence:
    v1_usd_curve = (
        evidence.protocol == FeeProtocol.CLOB_V1
        and formula == FeeFormula.POLYMARKET_CURVE
    )
    identity = {
        "condition_id": candidate.condition_id,
        "transaction_hash": candidate.transaction_hash,
        "effective_from_unix": candidate.effective_from_unix,
        "effective_to_unix": candidate.effective_to_unix,
        "protocol": evidence.protocol.value,
        "formula": formula.value,
        "rate": str(rate),
        "exponent": exponent,
    }
    return FeeScheduleEvidence(
        schedule_id=_stable_hash(identity),
        liquidity_id=candidate.liquidity_id,
        transaction_hash=candidate.transaction_hash,
        condition_id=candidate.condition_id,
        timestamp_unix=candidate.timestamp_unix,
        protocol=evidence.protocol,
        formula=formula,
        rate=rate,
        exponent=exponent,
        taker_only=True,
        collateral_rounding_decimals=(
            5 if evidence.protocol == FeeProtocol.CLOB_V2 or v1_usd_curve else 6
        ),
        outcome_rounding_decimals=5 if v1_usd_curve else 6,
        rounding=(
            FeeRounding.HALF_UP
            if evidence.protocol == FeeProtocol.CLOB_V2
            else FeeRounding.DOWN
        ),
        source="polygon_transaction_receipt_active_taker_fee",
        effective_from_unix=candidate.effective_from_unix,
        effective_to_unix=candidate.effective_to_unix,
        source_order_hash=evidence.order_hash,
        source_block_number=evidence.block_number,
        source_log_index=evidence.log_index,
        source_price=evidence.price,
        source_gross_shares=evidence.gross_shares,
        source_fee_asset=evidence.fee_asset,
        source_fee_amount=evidence.fee_amount,
    )


def infer_fee_schedule(
    candidate: FeeSourceCandidate,
    evidence: OnchainFeeEvidence,
) -> FeeScheduleEvidence:
    """Infer only a known official curve whose rounded amount matches the receipt."""

    if evidence.fee_amount == ZERO:
        identity = {
            "condition_id": candidate.condition_id,
            "effective_from_unix": candidate.effective_from_unix,
            "effective_to_unix": candidate.effective_to_unix,
            "protocol": evidence.protocol.value,
            "formula": FeeFormula.ZERO.value,
            "source_transaction": candidate.transaction_hash,
        }
        return FeeScheduleEvidence(
            schedule_id=_stable_hash(identity),
            liquidity_id=candidate.liquidity_id,
            transaction_hash=candidate.transaction_hash,
            condition_id=candidate.condition_id,
            timestamp_unix=candidate.timestamp_unix,
            protocol=evidence.protocol,
            formula=FeeFormula.ZERO,
            rate=ZERO,
            exponent=0,
            taker_only=True,
            collateral_rounding_decimals=(
                5 if evidence.protocol == FeeProtocol.CLOB_V2 else 6
            ),
            outcome_rounding_decimals=6,
            rounding=(
                FeeRounding.HALF_UP
                if evidence.protocol == FeeProtocol.CLOB_V2
                else FeeRounding.DOWN
            ),
            source="polygon_transaction_receipt_zero_active_taker_fee",
            effective_from_unix=candidate.effective_from_unix,
            effective_to_unix=candidate.effective_to_unix,
            source_order_hash=evidence.order_hash,
            source_block_number=evidence.block_number,
            source_log_index=evidence.log_index,
            source_price=evidence.price,
            source_gross_shares=evidence.gross_shares,
            source_fee_asset=FeeAsset.NONE,
            source_fee_amount=ZERO,
        )

    formulas = (
        (
            FeeFormula.V1_OUTPUT_ASSET_CURVE,
            FeeFormula.POLYMARKET_CURVE,
        )
        if evidence.protocol == FeeProtocol.CLOB_V1
        else (FeeFormula.POLYMARKET_CURVE,)
    )
    matches: list[tuple[Decimal, FeeScheduleEvidence]] = []
    for formula in formulas:
        five_decimal = (
            evidence.protocol == FeeProtocol.CLOB_V2
            or formula == FeeFormula.POLYMARKET_CURVE
        )
        quantum = Decimal("0.00001" if five_decimal else "0.000001")
        tolerance = quantum * (evidence.constituent_maker_fills + 1)
        for rate, exponent in KNOWN_CURVE_SCHEDULES:
            schedule = _candidate_schedule(
                candidate,
                evidence,
                rate=rate,
                exponent=exponent,
                formula=formula,
            )
            quoted = apply_polymarket_fee(
                schedule,
                side=evidence.side,
                liquidity_role=LiquidityRole.TAKER,
                gross_shares=evidence.gross_shares,
                price=evidence.price,
            )
            difference = abs(quoted.fee_amount - evidence.fee_amount)
            if difference <= tolerance:
                matches.append((difference, schedule))
    if not matches:
        raise RuntimeError(
            "On-chain fee does not match a registered official schedule: "
            f"{candidate.transaction_hash} amount={evidence.fee_amount}"
        )
    matches.sort(key=lambda row: (row[0], row[1].rate, row[1].exponent))
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise RuntimeError(
            f"On-chain fee schedule is ambiguous: {candidate.transaction_hash}"
        )
    return matches[0][1]


def official_zero_schedule(
    *,
    condition_id: str,
    effective_from_unix: int,
    effective_to_unix: int,
    protocol: FeeProtocol,
) -> FeeScheduleEvidence:
    identity = {
        "condition_id": condition_id,
        "effective_from_unix": effective_from_unix,
        "effective_to_unix": effective_to_unix,
        "protocol": protocol.value,
        "formula": FeeFormula.ZERO.value,
        "source": "official_pre_fee_chronology",
    }
    return FeeScheduleEvidence(
        schedule_id=_stable_hash(identity),
        liquidity_id=None,
        transaction_hash=None,
        condition_id=condition_id,
        timestamp_unix=effective_from_unix,
        protocol=protocol,
        formula=FeeFormula.ZERO,
        rate=ZERO,
        exponent=0,
        taker_only=True,
        collateral_rounding_decimals=6,
        outcome_rounding_decimals=6,
        rounding=FeeRounding.DOWN,
        source="official_changelog_pre_2026_01_05_no_platform_taker_fees",
        effective_from_unix=effective_from_unix,
        effective_to_unix=effective_to_unix,
    )
