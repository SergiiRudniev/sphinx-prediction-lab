"""Protocol-exact counterfactual targets for the H017 tail-utility policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import sha256_file
from sphinx_trace.on_policy_pack import H015_ARRAY_NAMES
from sphinx_trace.polymarket_fees import (
    FeeProtocol,
    FeeScheduleEvidence,
    LiquidityRole,
    apply_polymarket_fee,
    decimal,
)

H017_ARRAY_NAMES = (
    *H015_ARRAY_NAMES,
    "fee_schedule_ids.npy",
    "entry_prices.npy",
    "winning_payout_multipliers.npy",
    "reference_action_values.npy",
    "week_ids.npy",
)

ZERO = Decimal(0)
ONE = Decimal(1)
MONDAY_EPOCH_UNIX = 4 * 86_400
WEEK_SECONDS = 7 * 86_400


@dataclass(frozen=True, slots=True)
class ProtocolActionTargets:
    schedule_id: str
    entry_prices: tuple[float, float]
    winning_payout_multipliers: tuple[float, float]
    reference_action_values: tuple[float, float, float]


def calendar_week_id(timestamp_unix: int) -> int:
    """Return the UTC Monday boundary used by exact replay weekly accounting."""

    if timestamp_unix < 0:
        raise ValueError("H017 timestamp cannot be negative")
    return (
        (timestamp_unix - MONDAY_EPOCH_UNIX) // WEEK_SECONDS * WEEK_SECONDS
        + MONDAY_EPOCH_UNIX
    )


def winning_payout_per_total_cost(
    schedule: FeeScheduleEvidence,
    *,
    entry_price: Decimal | float | str,
    total_cost_usd: Decimal | float | str,
    rate_multiplier: Decimal | float | str = ONE,
) -> Decimal:
    """Quote terminal winning shares received per dollar of total BUY cost."""

    price = decimal(entry_price)
    total_cost = decimal(total_cost_usd)
    multiplier = decimal(rate_multiplier)
    if not ZERO < price <= ONE or total_cost <= ZERO or multiplier < ZERO:
        raise ValueError("H017 payout quote inputs are invalid")

    if schedule.protocol == FeeProtocol.CLOB_V1:
        gross_shares = total_cost / price
    else:
        low = ZERO
        high = total_cost / price
        for _ in range(96):
            candidate = (low + high) / 2
            fee = apply_polymarket_fee(
                schedule,
                side="BUY",
                liquidity_role=LiquidityRole.TAKER,
                gross_shares=candidate,
                price=price,
                rate_multiplier=multiplier,
            )
            candidate_cost = candidate * price + fee.collateral_fee_usd
            if candidate_cost <= total_cost:
                low = candidate
            else:
                high = candidate
        gross_shares = low

    fee = apply_polymarket_fee(
        schedule,
        side="BUY",
        liquidity_role=LiquidityRole.TAKER,
        gross_shares=gross_shares,
        price=price,
        rate_multiplier=multiplier,
    )
    if schedule.protocol == FeeProtocol.CLOB_V1:
        if fee.collateral_fee_usd != ZERO:
            raise RuntimeError("H017 V1 BUY unexpectedly charges collateral")
        actual_cost = total_cost
    else:
        actual_cost = gross_shares * price + fee.collateral_fee_usd
    if actual_cost <= ZERO or actual_cost > total_cost:
        raise RuntimeError("H017 fee quote exceeds registered total cost")
    position_shares = gross_shares - fee.outcome_fee_shares
    if position_shares < ZERO:
        raise RuntimeError("H017 fee quote produces negative winning shares")
    return position_shares / actual_cost


def protocol_action_targets(
    schedule: FeeScheduleEvidence,
    *,
    market_probability_outcome0: float,
    label_outcome0: float,
    reference_size: float,
    reference_equity_usd: float,
    adverse_price_ticks: int,
    tick_size: float,
    minimum_entry_price: float,
    rate_multiplier: float = 1.0,
) -> ProtocolActionTargets:
    """Build exact CALL-0/CALL-1/SKIP utilities for one resolved decision."""

    if not 0.0 <= market_probability_outcome0 <= 1.0:
        raise ValueError("H017 market probability must be between zero and one")
    if label_outcome0 not in {0.0, 1.0}:
        raise ValueError("H017 terminal label must be binary")
    if not 0.0 < reference_size < 1.0 or reference_equity_usd <= 0.0:
        raise ValueError("H017 utility reference is invalid")
    if adverse_price_ticks < 0 or tick_size < 0.0:
        raise ValueError("H017 adverse execution settings are invalid")
    if not 0.0 < minimum_entry_price <= 1.0:
        raise ValueError("H017 minimum entry price is invalid")

    adverse = adverse_price_ticks * tick_size
    prices = (
        min(1.0, max(minimum_entry_price, market_probability_outcome0 + adverse)),
        min(
            1.0,
            max(minimum_entry_price, 1.0 - market_probability_outcome0 + adverse),
        ),
    )
    total_cost = Decimal(str(reference_size * reference_equity_usd))
    payout_multipliers = (
        float(
            winning_payout_per_total_cost(
                schedule,
                entry_price=prices[0],
                total_cost_usd=total_cost,
                rate_multiplier=rate_multiplier,
            )
        ),
        float(
            winning_payout_per_total_cost(
                schedule,
                entry_price=prices[1],
                total_cost_usd=total_cost,
                rate_multiplier=rate_multiplier,
            )
        ),
    )
    winning = (label_outcome0, 1.0 - label_outcome0)
    utilities = tuple(
        math.log(
            max(
                1e-8,
                1.0 - reference_size
                + reference_size * outcome * payout_multiplier,
            )
        )
        for outcome, payout_multiplier in zip(
            winning, payout_multipliers, strict=True
        )
    )
    return ProtocolActionTargets(
        schedule_id=schedule.schedule_id,
        entry_prices=prices,
        winning_payout_multipliers=payout_multipliers,
        reference_action_values=(utilities[0], utilities[1], 0.0),
    )


def validate_protocol_tail_shard(
    shard_dir: Path,
    files: dict[str, Any],
    *,
    expected_rows: int,
) -> None:
    """Validate H017 protocol arrays and their immutable receipt metadata."""

    for name in H017_ARRAY_NAMES:
        metadata = files.get(name)
        path = shard_dir / name
        if not isinstance(metadata, dict) or not path.is_file():
            raise RuntimeError(f"H017 shard is incomplete: {path}")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"H017 shard size changed: {path}")
        if metadata.get("sha256") != sha256_file(path):
            raise RuntimeError(f"H017 shard hash changed: {path}")

    schedule_ids = np.load(
        shard_dir / "fee_schedule_ids.npy", mmap_mode="r", allow_pickle=False
    )
    prices = np.load(shard_dir / "entry_prices.npy", mmap_mode="r", allow_pickle=False)
    payouts = np.load(
        shard_dir / "winning_payout_multipliers.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    values = np.load(
        shard_dir / "reference_action_values.npy", mmap_mode="r", allow_pickle=False
    )
    weeks = np.load(shard_dir / "week_ids.npy", mmap_mode="r", allow_pickle=False)
    if (
        schedule_ids.shape != (expected_rows,)
        or schedule_ids.dtype.kind != "S"
        or prices.shape != (expected_rows, 2)
        or payouts.shape != (expected_rows, 2)
        or values.shape != (expected_rows, 3)
        or weeks.shape != (expected_rows,)
    ):
        raise RuntimeError(f"H017 protocol arrays do not align: {shard_dir}")
    if expected_rows and bool((schedule_ids == b"").any()):
        raise RuntimeError(f"H017 fee schedule binding is empty: {shard_dir}")
    if bool(((prices <= 0.0) | (prices > 1.0)).any()):
        raise RuntimeError(f"H017 entry prices are invalid: {shard_dir}")
    if bool((payouts <= 0.0).any()) or not bool(np.isfinite(payouts).all()):
        raise RuntimeError(f"H017 payout multipliers are invalid: {shard_dir}")
    if not bool(np.isfinite(values).all()) or not bool((values[:, 2] == 0.0).all()):
        raise RuntimeError(f"H017 action values are invalid: {shard_dir}")
    if bool((weeks < 0).any()):
        raise RuntimeError(f"H017 week IDs are invalid: {shard_dir}")
