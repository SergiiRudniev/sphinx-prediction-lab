"""Protocol-aware Polymarket platform-fee arithmetic and evidence binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

from sphinx_corpus.io import iter_jsonl_zst, sha256_file

ZERO = Decimal("0")
ONE = Decimal("1")


def decimal(value: Decimal | float | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class FeeProtocol(StrEnum):
    CLOB_V1 = "clob-v1"
    CLOB_V2 = "clob-v2"


class FeeFormula(StrEnum):
    ZERO = "zero"
    POLYMARKET_CURVE = "polymarket_curve"
    V1_OUTPUT_ASSET_CURVE = "v1_output_asset_curve"
    V1_MIN_PRICE_CURVE = "v1_min_price_curve"


class FeeAsset(StrEnum):
    NONE = "NONE"
    COLLATERAL = "COLLATERAL"
    OUTCOME = "OUTCOME"


class LiquidityRole(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class FeeRounding(StrEnum):
    HALF_UP = "HALF_UP"
    DOWN = "DOWN"


@dataclass(frozen=True, slots=True)
class FeeScheduleEvidence:
    """One contemporaneous fee schedule bound to one causal tape event."""

    schedule_id: str
    liquidity_id: str | None
    transaction_hash: str | None
    condition_id: str
    timestamp_unix: int
    protocol: FeeProtocol
    formula: FeeFormula
    rate: Decimal
    exponent: int
    taker_only: bool
    collateral_rounding_decimals: int
    outcome_rounding_decimals: int
    rounding: FeeRounding
    source: str
    effective_from_unix: int | None = None
    effective_to_unix: int | None = None
    source_order_hash: str | None = None
    source_block_number: int | None = None
    source_log_index: int | None = None
    source_price: Decimal | None = None
    source_gross_shares: Decimal | None = None
    source_fee_asset: FeeAsset = FeeAsset.NONE
    source_fee_amount: Decimal = ZERO

    def __post_init__(self) -> None:
        if not self.schedule_id or not self.condition_id:
            raise ValueError("Fee schedule and condition identifiers are required")
        if self.transaction_hash is not None and (
            not self.transaction_hash.startswith("0x") or len(self.transaction_hash) != 66
        ):
            raise ValueError("Fee schedule transaction hash is invalid")
        if self.timestamp_unix < 0:
            raise ValueError("Fee schedule timestamp cannot be negative")
        if self.rate < ZERO or self.exponent < 0:
            raise ValueError("Fee schedule rate and exponent cannot be negative")
        if self.formula == FeeFormula.ZERO and self.rate != ZERO:
            raise ValueError("Zero fee formula must have a zero rate")
        if not 0 <= self.collateral_rounding_decimals <= 18:
            raise ValueError("Collateral fee precision is invalid")
        if not 0 <= self.outcome_rounding_decimals <= 18:
            raise ValueError("Outcome fee precision is invalid")
        if self.source_price is not None and not ZERO < self.source_price < ONE:
            raise ValueError("Fee evidence source price must be inside (0, 1)")
        if self.source_gross_shares is not None and self.source_gross_shares <= ZERO:
            raise ValueError("Fee evidence source shares must be positive")
        if self.source_fee_amount < ZERO:
            raise ValueError("Fee evidence amount cannot be negative")
        if self.effective_from >= self.effective_to:
            raise ValueError("Fee schedule effective interval must be non-empty")

    @property
    def effective_from(self) -> int:
        return self.timestamp_unix if self.effective_from_unix is None else self.effective_from_unix

    @property
    def effective_to(self) -> int:
        return self.timestamp_unix + 1 if self.effective_to_unix is None else self.effective_to_unix

    def covers(self, timestamp_unix: int) -> bool:
        return self.effective_from <= timestamp_unix < self.effective_to

    @classmethod
    def zero(
        cls,
        *,
        liquidity_id: str | None,
        transaction_hash: str | None,
        condition_id: str,
        timestamp_unix: int,
        protocol: FeeProtocol,
        source: str,
    ) -> FeeScheduleEvidence:
        payload = {
            "liquidity_id": liquidity_id,
            "transaction_hash": (None if transaction_hash is None else transaction_hash.lower()),
            "condition_id": condition_id.lower(),
            "timestamp_unix": timestamp_unix,
            "protocol": protocol.value,
            "formula": FeeFormula.ZERO.value,
            "rate": "0",
        }
        schedule_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            schedule_id=schedule_id,
            liquidity_id=liquidity_id,
            transaction_hash=(None if transaction_hash is None else transaction_hash.lower()),
            condition_id=condition_id.lower(),
            timestamp_unix=timestamp_unix,
            protocol=protocol,
            formula=FeeFormula.ZERO,
            rate=ZERO,
            exponent=0,
            taker_only=True,
            collateral_rounding_decimals=5 if protocol == FeeProtocol.CLOB_V2 else 6,
            outcome_rounding_decimals=6,
            rounding=FeeRounding.HALF_UP if protocol == FeeProtocol.CLOB_V2 else FeeRounding.DOWN,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class AppliedFee:
    schedule_id: str
    protocol: FeeProtocol
    liquidity_role: LiquidityRole
    fee_asset: FeeAsset
    fee_amount: Decimal
    fee_value_usd: Decimal
    collateral_fee_usd: Decimal
    outcome_fee_shares: Decimal

    @property
    def is_zero(self) -> bool:
        return self.fee_amount == ZERO


def _quantum(decimals: int) -> Decimal:
    return ONE.scaleb(-decimals)


def _round(value: Decimal, decimals: int, mode: FeeRounding) -> Decimal:
    rounding = ROUND_HALF_UP if mode == FeeRounding.HALF_UP else ROUND_DOWN
    return value.quantize(_quantum(decimals), rounding=rounding)


def _unrounded_fee_value_usd(
    schedule: FeeScheduleEvidence,
    gross_shares: Decimal,
    price: Decimal,
    rate_multiplier: Decimal,
) -> Decimal:
    rate = schedule.rate * rate_multiplier
    if schedule.formula == FeeFormula.ZERO or rate == ZERO:
        return ZERO
    if schedule.formula in {
        FeeFormula.POLYMARKET_CURVE,
        FeeFormula.V1_OUTPUT_ASSET_CURVE,
    }:
        return gross_shares * rate * (price * (ONE - price)) ** schedule.exponent
    if schedule.formula == FeeFormula.V1_MIN_PRICE_CURVE:
        return gross_shares * rate * min(price, ONE - price)
    raise ValueError(f"Unsupported Polymarket fee formula: {schedule.formula}")


def apply_polymarket_fee(
    schedule: FeeScheduleEvidence,
    *,
    side: str,
    liquidity_role: LiquidityRole,
    gross_shares: Decimal | float | int | str,
    price: Decimal | float | int | str,
    rate_multiplier: Decimal | float | int | str = ONE,
) -> AppliedFee:
    """Apply exact fee-asset semantics to one simulated execution."""

    side_value = str(side).upper()
    if side_value not in {"BUY", "SELL"}:
        raise ValueError("Fee quote side must be BUY or SELL")
    shares = decimal(gross_shares)
    price_value = decimal(price)
    multiplier = decimal(rate_multiplier)
    if shares <= ZERO or not ZERO <= price_value <= ONE:
        raise ValueError("Fee quote requires positive shares and a price between zero and one")
    if multiplier < ZERO:
        raise ValueError("Fee rate multiplier cannot be negative")
    if liquidity_role == LiquidityRole.MAKER and schedule.taker_only:
        return AppliedFee(
            schedule.schedule_id,
            schedule.protocol,
            liquidity_role,
            FeeAsset.NONE,
            ZERO,
            ZERO,
            ZERO,
            ZERO,
        )

    raw_value = _unrounded_fee_value_usd(schedule, shares, price_value, multiplier)
    if raw_value == ZERO:
        return AppliedFee(
            schedule.schedule_id,
            schedule.protocol,
            liquidity_role,
            FeeAsset.NONE,
            ZERO,
            ZERO,
            ZERO,
            ZERO,
        )

    if schedule.protocol == FeeProtocol.CLOB_V1 and side_value == "BUY":
        raw_outcome_fee = (
            raw_value
            if schedule.formula == FeeFormula.V1_OUTPUT_ASSET_CURVE
            else raw_value / price_value
        )
        fee_shares = _round(
            raw_outcome_fee,
            schedule.outcome_rounding_decimals,
            schedule.rounding,
        )
        fee_shares = min(shares, fee_shares)
        fee_value = fee_shares * price_value
        return AppliedFee(
            schedule.schedule_id,
            schedule.protocol,
            liquidity_role,
            FeeAsset.OUTCOME,
            fee_shares,
            fee_value,
            ZERO,
            fee_shares,
        )

    collateral_fee = _round(
        raw_value,
        schedule.collateral_rounding_decimals,
        schedule.rounding,
    )
    return AppliedFee(
        schedule.schedule_id,
        schedule.protocol,
        liquidity_role,
        FeeAsset.COLLATERAL if collateral_fee else FeeAsset.NONE,
        collateral_fee,
        collateral_fee,
        collateral_fee,
        ZERO,
    )


class UnqualifiedFeeScheduleError(KeyError):
    """Structured fail-closed signal for extending a historical fee corpus."""

    def __init__(
        self,
        *,
        liquidity_id: str,
        condition_id: str | None,
        timestamp_unix: int | None,
    ) -> None:
        self.liquidity_id = liquidity_id
        self.condition_id = condition_id
        self.timestamp_unix = timestamp_unix
        super().__init__(
            "Polymarket fee schedule is unqualified: "
            f"{liquidity_id}@{condition_id}:{timestamp_unix}"
        )


class FeeScheduleBook:
    """Immutable fail-closed lookup from exact events or condition-time intervals."""

    def __init__(
        self,
        schedules: Iterable[FeeScheduleEvidence],
        *,
        manifest_sha256: str,
    ) -> None:
        if len(manifest_sha256) != 64:
            raise ValueError("Fee schedule manifest SHA-256 is invalid")
        bound: dict[str, FeeScheduleEvidence] = {}
        by_condition: dict[str, list[FeeScheduleEvidence]] = {}
        schedule_ids: set[str] = set()
        for schedule in schedules:
            if schedule.schedule_id in schedule_ids:
                raise ValueError(f"Fee schedule ID repeats: {schedule.schedule_id}")
            schedule_ids.add(schedule.schedule_id)
            if schedule.liquidity_id is not None:
                if schedule.liquidity_id in bound:
                    raise ValueError(f"Fee schedule liquidity ID repeats: {schedule.liquidity_id}")
                bound[schedule.liquidity_id] = schedule
            by_condition.setdefault(schedule.condition_id, []).append(schedule)
        if not schedule_ids:
            raise ValueError("Fee schedule book cannot be empty")
        for condition_id, rows in by_condition.items():
            rows.sort(key=lambda row: (row.effective_from, row.effective_to, row.schedule_id))
            for previous, current in pairwise(rows):
                if previous.effective_to > current.effective_from:
                    raise ValueError(
                        f"Fee schedule intervals overlap for condition: {condition_id}"
                    )
        self._schedules = bound
        self._schedules_by_condition = {
            condition_id: tuple(rows) for condition_id, rows in by_condition.items()
        }
        self._schedule_count = len(schedule_ids)
        self.manifest_sha256 = manifest_sha256

    def __len__(self) -> int:
        return self._schedule_count

    def schedule_for(
        self,
        liquidity_id: str,
        *,
        condition_id: str | None = None,
        timestamp_unix: int | None = None,
        transaction_hash: str | None = None,
    ) -> FeeScheduleEvidence:
        schedule = self._schedules.get(liquidity_id)
        exact = schedule is not None
        if schedule is None and condition_id is not None and timestamp_unix is not None:
            schedule = next(
                (
                    candidate
                    for candidate in self._schedules_by_condition.get(condition_id.lower(), ())
                    if candidate.effective_from_unix is not None
                    and candidate.effective_to_unix is not None
                    and candidate.covers(timestamp_unix)
                ),
                None,
            )
        if schedule is None:
            raise UnqualifiedFeeScheduleError(
                liquidity_id=liquidity_id,
                condition_id=condition_id,
                timestamp_unix=timestamp_unix,
            )
        if condition_id is not None and schedule.condition_id != condition_id.lower():
            raise RuntimeError("Fee schedule condition binding changed")
        if timestamp_unix is not None and not schedule.covers(timestamp_unix):
            raise RuntimeError("Fee schedule timestamp binding changed")
        if (
            exact
            and transaction_hash is not None
            and schedule.transaction_hash is not None
            and schedule.transaction_hash != transaction_hash.lower()
        ):
            raise RuntimeError("Fee schedule transaction binding changed")
        return schedule

    def quote(
        self,
        liquidity_id: str,
        *,
        condition_id: str,
        timestamp_unix: int,
        transaction_hash: str | None = None,
        side: str,
        liquidity_role: LiquidityRole,
        gross_shares: Decimal,
        price: Decimal,
        rate_multiplier: Decimal = ONE,
    ) -> AppliedFee:
        schedule = self.schedule_for(
            liquidity_id,
            condition_id=condition_id,
            timestamp_unix=timestamp_unix,
            transaction_hash=transaction_hash,
        )
        return apply_polymarket_fee(
            schedule,
            side=side,
            liquidity_role=liquidity_role,
            gross_shares=gross_shares,
            price=price,
            rate_multiplier=rate_multiplier,
        )

    @classmethod
    def from_artifact(cls, directory: Path) -> FeeScheduleBook:
        manifest_path = directory / "manifest.json"
        payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Fee schedule manifest must be an object")
        data_path = directory / str(payload.get("data_path") or "schedules.jsonl.zst")
        if (
            payload.get("record_type") != "h016_fee_schedule_manifest"
            or payload.get("valid") is not True
            or payload.get("test_labels_opened") is not False
            or int(payload.get("test_rows_consumed", -1)) != 0
            or payload.get("data_sha256") != sha256_file(data_path)
        ):
            raise RuntimeError("H016 fee schedule artifact contract changed")
        schedules = [_schedule_from_payload(row) for row in iter_jsonl_zst(data_path)]
        if len(schedules) != int(payload.get("rows", -1)):
            raise RuntimeError("H016 fee schedule row count changed")
        proof_path = directory / str(payload.get("receipt_proof_path") or "")
        if not proof_path.is_file() or payload.get("receipt_proof_sha256") != sha256_file(
            proof_path
        ):
            raise RuntimeError("H016 fee receipt proof contract changed")
        receipt_schedule_ids = {
            schedule.schedule_id
            for schedule in schedules
            if schedule.source != "official_changelog_pre_2026_01_05_no_platform_taker_fees"
        }
        proof_schedule_ids: set[str] = set()
        proof_identities: set[tuple[str, str]] = set()
        proof_counts: dict[str, int] = {}
        proof_rows = 0
        for proof in iter_jsonl_zst(proof_path):
            proof_rows += 1
            schedule_id = str(proof.get("schedule_id") or "")
            transaction_hash = str(proof.get("transaction_hash") or "").lower()
            identity = (schedule_id, transaction_hash)
            if (
                proof.get("record_type") != "h016_fee_receipt_proof"
                or proof.get("status") != "0x1"
                or len(transaction_hash) != 66
                or not transaction_hash.startswith("0x")
                or identity in proof_identities
            ):
                raise RuntimeError("H016 fee receipt proof row changed")
            proof_identities.add(identity)
            proof_schedule_ids.add(schedule_id)
            proof_counts[schedule_id] = proof_counts.get(schedule_id, 0) + 1
        if (
            proof_rows != int(payload.get("receipt_proof_rows", -1))
            or proof_schedule_ids != receipt_schedule_ids
        ):
            raise RuntimeError("H016 fee receipt proof coverage changed")
        if any(
            proof_counts.get(schedule.schedule_id, 0) < 2
            for schedule in schedules
            if "receipt_consensus" in schedule.source
        ):
            raise RuntimeError("H016 fee receipt consensus proof coverage changed")
        market_info_path = directory / str(payload.get("market_info_path") or "")
        if not market_info_path.is_file() or payload.get("market_info_sha256") != sha256_file(
            market_info_path
        ):
            raise RuntimeError("H016 fee market-info proof contract changed")
        expected_market_info_conditions = {
            schedule.condition_id for schedule in schedules if "clob_market_info" in schedule.source
        }
        market_info_conditions: set[str] = set()
        market_info_rows = 0
        for proof in iter_jsonl_zst(market_info_path):
            market_info_rows += 1
            condition_id = str(proof.get("condition_id") or "").lower()
            market_payload = proof.get("payload")
            market_payload_sha256 = (
                hashlib.sha256(
                    json.dumps(
                        market_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if isinstance(market_payload, dict)
                else ""
            )
            market_condition_id = (
                str(market_payload.get("c", "")).lower() if isinstance(market_payload, dict) else ""
            )
            if (
                proof.get("record_type") != "h016_fee_market_info_proof"
                or condition_id in market_info_conditions
                or proof.get("payload_sha256") != market_payload_sha256
                or market_condition_id != condition_id
            ):
                raise RuntimeError("H016 fee market-info proof row changed")
            market_info_conditions.add(condition_id)
        if (
            market_info_rows != int(payload.get("market_info_rows", -1))
            or market_info_conditions != expected_market_info_conditions
        ):
            raise RuntimeError("H016 fee market-info proof coverage changed")
        market_trade_path = directory / str(payload.get("market_trade_path") or "")
        if not market_trade_path.is_file() or payload.get("market_trade_sha256") != sha256_file(
            market_trade_path
        ):
            raise RuntimeError("H016 fee market-trade proof contract changed")
        expected_market_trade_schedules = {
            schedule.schedule_id for schedule in schedules if "market_wide_trade" in schedule.source
        }
        market_trade_schedules: set[str] = set()
        market_trade_rows = 0
        for proof in iter_jsonl_zst(market_trade_path):
            market_trade_rows += 1
            schedule_id = str(proof.get("schedule_id") or "")
            trade_row = proof.get("row")
            row_sha256 = (
                hashlib.sha256(
                    json.dumps(
                        trade_row,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if isinstance(trade_row, dict)
                else ""
            )
            if (
                proof.get("record_type") != "h016_fee_market_trade_proof"
                or schedule_id in market_trade_schedules
                or proof.get("row_sha256") != row_sha256
            ):
                raise RuntimeError("H016 fee market-trade proof row changed")
            market_trade_schedules.add(schedule_id)
        if (
            market_trade_rows != int(payload.get("market_trade_rows", -1))
            or market_trade_schedules != expected_market_trade_schedules
        ):
            raise RuntimeError("H016 fee market-trade proof coverage changed")
        return cls(schedules, manifest_sha256=sha256_file(manifest_path))


def _schedule_from_payload(payload: dict[str, Any]) -> FeeScheduleEvidence:
    row = dict(payload)
    row.pop("schema_version", None)
    row.pop("record_type", None)
    row["protocol"] = FeeProtocol(row["protocol"])
    row["formula"] = FeeFormula(row["formula"])
    row["rounding"] = FeeRounding(row["rounding"])
    row["source_fee_asset"] = FeeAsset(row.get("source_fee_asset", FeeAsset.NONE))
    for key in ("rate", "source_price", "source_gross_shares", "source_fee_amount"):
        if row.get(key) is not None:
            row[key] = decimal(row[key])
    return FeeScheduleEvidence(**row)


def fee_schedule_payload(schedule: FeeScheduleEvidence) -> dict[str, Any]:
    row = asdict(schedule)
    for key, value in tuple(row.items()):
        if isinstance(value, Decimal):
            row[key] = str(value)
        elif isinstance(value, StrEnum):
            row[key] = value.value
    return {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_schedule",
        **row,
    }
