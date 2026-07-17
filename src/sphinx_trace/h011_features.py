"""Feature contract and low-level causal helpers for the H011 campaign.

The packer represents every observed participant through recurrent state.  Raw
wallet identities are used only as deterministic state keys and are never
emitted as learnable features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

H011_FEATURE_NAMES = (
    # Global clock (8)
    "time.global_trade_count_log1p",
    "time.global_notional_log1p",
    "time.elapsed_days_log1p",
    "time.day_sin",
    "time.day_cos",
    "time.week_sin",
    "time.week_cos",
    "time.global_gap_log1p",
    # Evidence market (40)
    "market.trade_count_log1p",
    "market.notional_log1p",
    "market.size_log1p",
    "market.last_outcome0_probability",
    "market.evidence_raw_price",
    "market.evidence_outcome_index",
    "market.evidence_side_buy",
    "market.evidence_notional_log1p",
    "market.evidence_size_log1p",
    "market.signed_flow_share",
    "market.buy_sell_imbalance",
    "market.outcome0_outcome1_imbalance",
    "market.mean_outcome0_probability",
    "market.std_outcome0_probability",
    "market.min_outcome0_probability",
    "market.max_outcome0_probability",
    "market.gap_log1p",
    "market.age_log1p",
    "market.unique_wallets_log1p",
    "market.repeat_intensity_log1p",
    "market.trade_hhi_proxy",
    "market.large_trade_ratio",
    "market.probability_ema_5m",
    "market.probability_ema_1h",
    "market.probability_ema_6h",
    "market.probability_ema_1d",
    "market.probability_ema_7d",
    "market.flow_ema_5m",
    "market.flow_ema_1h",
    "market.flow_ema_6h",
    "market.flow_ema_1d",
    "market.flow_ema_7d",
    "market.return_from_5m",
    "market.return_from_1h",
    "market.return_from_1d",
    "market.return_from_7d",
    "market.notional_per_trade_log1p",
    "market.notional_velocity_log1p",
    "market.lifecycle_elapsed_fraction",
    "market.lifecycle_remaining_fraction",
    # Linked event component (24)
    "component.trade_count_log1p",
    "component.notional_log1p",
    "component.signed_flow_share",
    "component.buy_sell_imbalance",
    "component.outcome0_outcome1_imbalance",
    "component.mean_outcome0_probability",
    "component.std_outcome0_probability",
    "component.last_outcome0_probability",
    "component.gap_log1p",
    "component.age_log1p",
    "component.unique_markets_log1p",
    "component.repeat_market_intensity_log1p",
    "component.evidence_market_trade_share",
    "component.evidence_market_notional_share",
    "component.market_count_log1p",
    "component.neg_risk_fraction",
    "component.unclosed_fraction",
    "component.probability_ema_1h",
    "component.probability_ema_1d",
    "component.probability_ema_7d",
    "component.flow_ema_1h",
    "component.flow_ema_1d",
    "component.flow_ema_7d",
    "component.notional_velocity_log1p",
    # Evidence wallet (28)
    "wallet.trade_count_log1p",
    "wallet.notional_log1p",
    "wallet.size_log1p",
    "wallet.buy_sell_imbalance",
    "wallet.outcome0_outcome1_imbalance",
    "wallet.signed_flow_share",
    "wallet.mean_price",
    "wallet.std_price",
    "wallet.recency_log1p",
    "wallet.mean_interarrival_log1p",
    "wallet.market_switch_ratio",
    "wallet.same_market_as_previous",
    "wallet.evidence_notional_share",
    "wallet.resolved_market_count_log1p",
    "wallet.resolved_directional_edge_mean",
    "wallet.resolved_pnl_proxy_mean",
    "wallet.resolved_win_rate",
    "wallet.actor_context_available",
    "wallet.actor_maker_fills_log1p",
    "wallet.actor_taker_fills_log1p",
    "wallet.actor_maker_ratio",
    "wallet.actor_buy_sell_imbalance",
    "wallet.actor_notional_log1p",
    "wallet.actor_counterparties_log1p",
    "wallet.actor_assets_log1p",
    "wallet.actor_mean_price",
    "wallet.actor_price_std",
    "wallet.novelty",
    # All-wallet market aggregate (16)
    "wallet_set.activity_mean",
    "wallet_set.activity_std",
    "wallet_set.notional_mean",
    "wallet_set.notional_std",
    "wallet_set.resolved_edge_mean",
    "wallet_set.resolved_edge_std",
    "wallet_set.resolved_pnl_mean",
    "wallet_set.resolved_pnl_std",
    "wallet_set.actor_maker_ratio_mean",
    "wallet_set.actor_available_fraction",
    "wallet_set.resolved_history_fraction",
    "wallet_set.aggregate_weight_log1p",
    "wallet_set.effective_weight_count_log1p",
    "wallet_set.recency_mean",
    "wallet_set.novelty_mean",
    "wallet_set.evidence_wallet_influence",
    # Whole observable universe (12)
    "universe.trade_count_log1p",
    "universe.notional_log1p",
    "universe.mean_outcome0_probability",
    "universe.std_outcome0_probability",
    "universe.signed_flow_share",
    "universe.buy_sell_imbalance",
    "universe.active_markets_log1p",
    "universe.active_components_log1p",
    "universe.component_trade_share",
    "universe.component_notional_share",
    "universe.notional_velocity_log1p",
    "universe.flow_ema_1d",
)

H011_FEATURE_WIDTH = 128
H011_HLL_PRECISION = 6
H011_HLL_REGISTERS = 1 << H011_HLL_PRECISION

if len(H011_FEATURE_NAMES) != H011_FEATURE_WIDTH:  # pragma: no cover - import invariant
    raise RuntimeError("The registered H011 feature contract must contain exactly 128 features")


@dataclass(frozen=True, slots=True)
class ParsedTrade:
    trade_id: str
    condition_id: str
    wallet: str
    timestamp_unix: int
    raw_price: float
    outcome0_probability: float
    size: float
    notional: float
    outcome_index: int
    side_buy: int
    direction_outcome0: int


def parse_trade_payload(payload: dict[str, Any]) -> ParsedTrade:
    """Validate a normalized Ledger row and orient it to catalog outcome zero."""

    trade_id = str(payload.get("trade_id") or "")
    condition_id = str(payload.get("condition_id") or "").lower()
    wallet = str(payload.get("wallet") or "").lower()
    side = str(payload.get("side") or "").upper()
    try:
        timestamp_unix = int(payload["timestamp_unix"])
        raw_price = float(payload["price"])
        size = float(payload["size"])
        notional = float(payload["notional_usd"])
        outcome_index = int(payload["outcome_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Ledger trade has invalid numeric fields") from error
    if not trade_id or not condition_id or not wallet:
        raise ValueError("Ledger trade has an empty identity field")
    if side not in {"BUY", "SELL"} or outcome_index not in {0, 1}:
        raise ValueError("Ledger trade has an unsupported side or outcome index")
    if not 0.0 < raw_price < 1.0 or size <= 0.0 or notional <= 0.0:
        raise ValueError("Ledger trade has values outside the registered domain")
    side_buy = int(side == "BUY")
    direction = 1 if side_buy == int(outcome_index == 0) else -1
    probability = raw_price if outcome_index == 0 else 1.0 - raw_price
    return ParsedTrade(
        trade_id=trade_id,
        condition_id=condition_id,
        wallet=wallet,
        timestamp_unix=timestamp_unix,
        raw_price=raw_price,
        outcome0_probability=probability,
        size=size,
        notional=notional,
        outcome_index=outcome_index,
        side_buy=side_buy,
        direction_outcome0=direction,
    )


def splitmix64(value: int) -> int:
    """Stable 64-bit mixer used by the streaming sketches."""

    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def hll_add(registers: NDArray[np.uint8], value: int) -> None:
    """Add one uncapped identity to a compact HyperLogLog sketch."""

    if registers.shape != (H011_HLL_REGISTERS,):
        raise ValueError("Unexpected HLL register shape")
    hashed = splitmix64(value)
    index = hashed & (H011_HLL_REGISTERS - 1)
    remainder = hashed >> H011_HLL_PRECISION
    remaining_bits = 64 - H011_HLL_PRECISION
    rank = remaining_bits + 1 if remainder == 0 else remaining_bits - remainder.bit_length() + 1
    registers[index] = max(int(registers[index]), rank)


def hll_estimate(registers: NDArray[np.uint8]) -> float:
    """Estimate cardinality, including the small-range linear-counting correction."""

    if registers.shape != (H011_HLL_REGISTERS,):
        raise ValueError("Unexpected HLL register shape")
    count = float(H011_HLL_REGISTERS)
    alpha = 0.709
    inverse_sum = float(np.exp2(-registers.astype(np.float64)).sum())
    estimate = alpha * count * count / max(inverse_sum, np.finfo(np.float64).tiny)
    zeros = int(np.count_nonzero(registers == 0))
    if zeros and estimate <= 2.5 * count:
        estimate = count * math.log(count / zeros)
    return max(estimate, 0.0)


def lifecycle_fractions(timestamp: int, created_at: int, end_at: int) -> tuple[float, float]:
    """Return bounded elapsed and remaining fractions without using future outcomes."""

    if created_at <= 0 or end_at <= created_at:
        return 0.0, 0.0
    duration = float(end_at - created_at)
    elapsed = min(max((timestamp - created_at) / duration, 0.0), 1.0)
    remaining = min(max((end_at - timestamp) / duration, 0.0), 1.0)
    return elapsed, remaining


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def weighted_std(sum_value: float, sum_square: float, weight: float) -> float:
    if weight <= 0.0:
        return 0.0
    mean = sum_value / weight
    return math.sqrt(max(sum_square / weight - mean * mean, 0.0))
