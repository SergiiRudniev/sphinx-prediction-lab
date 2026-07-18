"""Receipt-derived historical fee evidence for H016 Polymarket replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
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
V1_FIVE_DECIMAL_OPERATOR_FEE_UNIX = 1_774_828_800
V1_USD_CURVE_SOURCE_OF_TRUTH_UNIX = 1_774_915_200
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
    (Decimal("0.0175"), 1),
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
    fee_refund_observed: bool = False


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
    fee_refund_observed = str(row["order_hash"]) in refunds
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
        fee_refund_observed=fee_refund_observed,
    )


def _candidate_schedule(
    candidate: FeeSourceCandidate,
    evidence: OnchainFeeEvidence,
    *,
    rate: Decimal,
    exponent: int,
    formula: FeeFormula,
    operator_decimals: int | None = None,
) -> FeeScheduleEvidence:
    v1_operator_decimals = operator_decimals or (
        4
        if candidate.timestamp_unix < V1_FIVE_DECIMAL_OPERATOR_FEE_UNIX
        or exponent == 2
        else 5
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
        "operator_decimals": v1_operator_decimals,
    }
    return FeeScheduleEvidence(
        schedule_id=_stable_hash(identity),
        liquidity_id=None,
        transaction_hash=candidate.transaction_hash,
        condition_id=candidate.condition_id,
        timestamp_unix=candidate.timestamp_unix,
        protocol=evidence.protocol,
        formula=formula,
        rate=rate,
        exponent=exponent,
        taker_only=True,
        collateral_rounding_decimals=(
            5
            if evidence.protocol == FeeProtocol.CLOB_V2
            else v1_operator_decimals
        ),
        outcome_rounding_decimals=(
            6
            if evidence.protocol == FeeProtocol.CLOB_V2
            else v1_operator_decimals
        ),
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

    if (
        evidence.protocol == FeeProtocol.CLOB_V1
        and evidence.fee_amount > ZERO
        and not evidence.fee_refund_observed
    ):
        raise RuntimeError("V1 active fee is exchange-capped without FeeRefunded evidence")

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
            liquidity_id=None,
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

    matches = _matching_fee_schedules(
        candidate,
        evidence,
        rate_exponents=KNOWN_CURVE_SCHEDULES,
    )
    if not matches:
        raise RuntimeError(
            "On-chain fee does not match a registered official schedule: "
            f"{candidate.transaction_hash} amount={evidence.fee_amount}"
        )
    return _select_fee_schedule(matches, candidate)


def _matching_fee_schedules(
    candidate: FeeSourceCandidate,
    evidence: OnchainFeeEvidence,
    *,
    rate_exponents: tuple[tuple[Decimal, int], ...],
) -> list[tuple[Decimal, FeeScheduleEvidence]]:
    if (
        evidence.protocol == FeeProtocol.CLOB_V1
        and evidence.fee_amount > ZERO
        and not evidence.fee_refund_observed
    ):
        return []
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
        for rate, exponent in rate_exponents:
            precisions: tuple[int | None, ...] = (
                (None,) if evidence.protocol == FeeProtocol.CLOB_V2 else (4, 5)
            )
            for operator_decimals in precisions:
                schedule = _candidate_schedule(
                    candidate,
                    evidence,
                    rate=rate,
                    exponent=exponent,
                    formula=formula,
                    operator_decimals=operator_decimals,
                )
                quoted = apply_polymarket_fee(
                    schedule,
                    side=evidence.side,
                    liquidity_role=LiquidityRole.TAKER,
                    gross_shares=evidence.gross_shares,
                    price=evidence.price,
                )
                difference = abs(quoted.fee_amount - evidence.fee_amount)
                precision = (
                    schedule.outcome_rounding_decimals
                    if evidence.protocol == FeeProtocol.CLOB_V1
                    and evidence.side == "BUY"
                    else schedule.collateral_rounding_decimals
                )
                tolerance = Decimal(1).scaleb(-precision) * (
                    evidence.constituent_maker_fills + 1
                )
                if difference <= tolerance:
                    matches.append((difference, schedule))
    return matches


def infer_fee_schedule_consensus(
    samples: list[tuple[FeeSourceCandidate, OnchainFeeEvidence]],
    market_info: dict[str, Any],
) -> FeeScheduleEvidence:
    """Resolve rounded V1 fills only when independent receipts share one tariff."""

    transaction_hashes = {candidate.transaction_hash for candidate, _ in samples}
    if len(transaction_hashes) < 2:
        raise RuntimeError("Fee consensus requires at least two independent receipts")
    first_candidate, first_evidence = samples[0]
    if str(market_info.get("c", "")).lower() != first_candidate.condition_id:
        raise RuntimeError("CLOB market info condition changed")
    tokens = market_info.get("t")
    if not isinstance(tokens, list) or {
        str(row.get("t")) for row in tokens if isinstance(row, dict)
    } != set(first_candidate.token_ids):
        raise RuntimeError("CLOB market info tokens changed")
    details = market_info.get("fd")
    if not isinstance(details, dict) or details.get("to") is not True:
        raise RuntimeError("CLOB market info has no taker-only fee details")
    exponent = int(details.get("e", -1))
    market_rate = Decimal(str(details.get("r")))
    if exponent < 0 or market_rate < ZERO:
        raise RuntimeError("CLOB market info fee details are invalid")
    rate_exponents = tuple(
        sorted(
            {
                *(row for row in KNOWN_CURVE_SCHEDULES if row[1] == exponent),
                (market_rate, exponent),
            }
        )
    )

    semantic_intersection: set[tuple[FeeProtocol, FeeFormula, Decimal, int]] | None = None
    sample_matches: list[list[tuple[Decimal, FeeScheduleEvidence]]] = []
    for candidate, evidence in samples:
        if (
            candidate.condition_id != first_candidate.condition_id
            or candidate.token_ids != first_candidate.token_ids
            or evidence.protocol != first_evidence.protocol
            or evidence.fee_amount <= ZERO
            or not evidence.fee_refund_observed
        ):
            raise RuntimeError("Fee consensus samples are not comparable refund evidence")
        matches = _matching_fee_schedules(
            candidate,
            evidence,
            rate_exponents=rate_exponents,
        )
        if not matches:
            raise RuntimeError(
                f"Fee consensus receipt has no registered match: {candidate.transaction_hash}"
            )
        minimum_difference = min(row[0] for row in matches)
        best_matches = [row for row in matches if row[0] == minimum_difference]
        preferred_formula = _preferred_formula(candidate, exponent)
        formula_matches = [
            row for row in best_matches if row[1].formula == preferred_formula
        ]
        if formula_matches:
            best_matches = formula_matches
        preferred_decimals = (
            4
            if candidate.timestamp_unix < V1_FIVE_DECIMAL_OPERATOR_FEE_UNIX
            or exponent == 2
            else 5
        )
        precision_matches = [
            row
            for row in best_matches
            if row[1].collateral_rounding_decimals == preferred_decimals
        ]
        if precision_matches:
            best_matches = precision_matches
        semantics = {
            (row.protocol, row.formula, row.rate, row.exponent)
            for _, row in best_matches
        }
        semantic_intersection = (
            semantics
            if semantic_intersection is None
            else semantic_intersection & semantics
        )
        sample_matches.append(best_matches)
    if not semantic_intersection:
        raise RuntimeError(
            "Independent receipts have no shared registered fee schedule"
        )
    preferred_formula = _preferred_formula(first_candidate, exponent)
    chronology_consistent = {
        semantic
        for semantic in semantic_intersection
        if semantic[1] == preferred_formula
    }
    if chronology_consistent:
        semantic_intersection = chronology_consistent
    semantic_scores = {
        semantic: sum(
            min(
                difference
                for difference, schedule in matches
                if (
                    schedule.protocol,
                    schedule.formula,
                    schedule.rate,
                    schedule.exponent,
                )
                == semantic
            )
            for matches in sample_matches
        )
        for semantic in semantic_intersection
    }
    minimum_score = min(semantic_scores.values())
    best_semantics = [
        semantic
        for semantic, score in semantic_scores.items()
        if score == minimum_score
    ]
    if len(best_semantics) != 1:
        raise RuntimeError(
            "Independent receipt consensus is not unique: "
            f"{len(best_semantics)} minimum-residual schedules"
        )
    protocol, formula, rate, matched_exponent = best_semantics[0]
    candidates = [
        (difference, schedule)
        for matches in sample_matches
        for difference, schedule in matches
        if (
            schedule.protocol,
            schedule.formula,
            schedule.rate,
            schedule.exponent,
        )
        == (protocol, formula, rate, matched_exponent)
    ]
    preferred_decimals = (
        4
        if first_candidate.timestamp_unix < V1_FIVE_DECIMAL_OPERATOR_FEE_UNIX
        or matched_exponent == 2
        else 5
    )
    candidates.sort(
        key=lambda row: (
            row[0],
            row[1].collateral_rounding_decimals != preferred_decimals,
            row[1].transaction_hash or "",
        )
    )
    return replace(
        candidates[0][1],
        source=(
            "clob_market_info_market_wide_trade_receipt_consensus_"
            "active_taker_fee"
        ),
    )


def _preferred_formula(candidate: FeeSourceCandidate, exponent: int) -> FeeFormula:
    if exponent == 2 or candidate.timestamp_unix < V1_USD_CURVE_SOURCE_OF_TRUTH_UNIX:
        return FeeFormula.V1_OUTPUT_ASSET_CURVE
    return FeeFormula.POLYMARKET_CURVE


def _select_fee_schedule(
    matches: list[tuple[Decimal, FeeScheduleEvidence]],
    candidate: FeeSourceCandidate,
) -> FeeScheduleEvidence:
    matches.sort(key=lambda row: (row[0], row[1].rate, row[1].exponent))
    best = [row for row in matches if row[0] == matches[0][0]]
    if len(best) == 1:
        return best[0][1]
    preferred = _preferred_formula(candidate, best[0][1].exponent)
    preferred_rows = [row for row in best if row[1].formula == preferred]
    if preferred_rows:
        best = preferred_rows
    preferred_decimals = (
        4
        if candidate.timestamp_unix < V1_FIVE_DECIMAL_OPERATOR_FEE_UNIX
        or best[0][1].exponent == 2
        else 5
    )
    precision_rows = [
        row
        for row in best
        if row[1].collateral_rounding_decimals == preferred_decimals
    ]
    if precision_rows:
        best = precision_rows
    if len(best) == 1:
        return best[0][1]
    rate_exponents = {(row[1].rate, row[1].exponent) for row in best}
    if len(rate_exponents) == 1:
        return best[0][1]
    raise RuntimeError(
        f"On-chain fee schedule is ambiguous: {candidate.transaction_hash}"
    )


def infer_fee_schedule_from_market_info(
    candidate: FeeSourceCandidate,
    evidence: OnchainFeeEvidence,
    market_info: dict[str, Any],
) -> FeeScheduleEvidence:
    """Resolve a receipt-rounded ambiguity with the official per-market fee details."""

    if (
        evidence.protocol == FeeProtocol.CLOB_V1
        and evidence.fee_amount > ZERO
        and not evidence.fee_refund_observed
    ):
        raise RuntimeError("V1 active fee is exchange-capped without FeeRefunded evidence")

    if str(market_info.get("c", "")).lower() != candidate.condition_id:
        raise RuntimeError("CLOB market info condition changed")
    tokens = market_info.get("t")
    if not isinstance(tokens, list) or {
        str(row.get("t")) for row in tokens if isinstance(row, dict)
    } != set(candidate.token_ids):
        raise RuntimeError("CLOB market info tokens changed")
    details = market_info.get("fd")
    if not isinstance(details, dict) or details.get("to") is not True:
        raise RuntimeError("CLOB market info has no taker-only fee details")
    market_rate = Decimal(str(details.get("r")))
    exponent = int(details.get("e", -1))
    if market_rate < ZERO or exponent < 0:
        raise RuntimeError("CLOB market info fee details are invalid")
    rates = {
        rate
        for rate, registered_exponent in KNOWN_CURVE_SCHEDULES
        if registered_exponent == exponent
    }
    rates.add(market_rate)
    matches = _matching_fee_schedules(
        candidate,
        evidence,
        rate_exponents=tuple((rate, exponent) for rate in sorted(rates)),
    )
    if not matches:
        raise RuntimeError(
            "CLOB market fee details do not reconcile to the on-chain receipt: "
            f"{candidate.transaction_hash}"
        )
    return replace(
        _select_fee_schedule(matches, candidate),
        source="clob_market_info_receipt_reconciled_active_taker_fee",
    )


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
