"""Numba-compatible recurrent feature kernel for H011.

The hot loop has no wallet, market, or event truncation.  Every valid Ledger
row updates the persistent arrays; features are materialized only at registered
H009 decision cursors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

numba_module: Any
try:
    import numba as numba_module
except ImportError:  # pragma: no cover - exercised in minimal installations
    numba_module = None

HALF_LIVES = (300.0, 3_600.0, 21_600.0, 86_400.0, 604_800.0)
COMPONENT_HALF_LIVES = (3_600.0, 86_400.0, 604_800.0)
HLL_PRECISION = 6
HLL_REGISTERS = 1 << HLL_PRECISION

Function = TypeVar("Function", bound=Callable[..., Any])


def _jitable(function: Function) -> Function:
    if numba_module is None:
        return function
    decorated: Any = numba_module.extending.register_jitable(function)
    return cast(Function, decorated)


@_jitable
def _mix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


@_jitable
def _hll_add(matrix: NDArray[np.uint8], row: int, value: int) -> None:
    hashed = _mix64(value)
    index = hashed & (HLL_REGISTERS - 1)
    remainder = hashed >> HLL_PRECISION
    remaining = 64 - HLL_PRECISION
    if remainder == 0:
        rank = remaining + 1
    else:
        bits = 0
        cursor = remainder
        while cursor:
            bits += 1
            cursor >>= 1
        rank = remaining - bits + 1
    if rank > matrix[row, index]:
        matrix[row, index] = rank


@_jitable
def _hll_estimate(matrix: NDArray[np.uint8], row: int) -> float:
    inverse_sum = 0.0
    zeros = 0
    for index in range(HLL_REGISTERS):
        register = int(matrix[row, index])
        inverse_sum += 2.0 ** (-register)
        if register == 0:
            zeros += 1
    count = float(HLL_REGISTERS)
    estimate = 0.709 * count * count / max(inverse_sum, 1e-300)
    if zeros > 0 and estimate <= 2.5 * count:
        estimate = count * math.log(count / zeros)
    return max(estimate, 0.0)


@_jitable
def _ema(old: float, value: float, delta_seconds: float, half_life: float) -> float:
    if delta_seconds < 0.0:
        delta_seconds = 0.0
    decay = math.exp(-math.log(2.0) * delta_seconds / half_life)
    return decay * old + (1.0 - decay) * value


@_jitable
def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


@_jitable
def _std(sum_value: float, sum_square: float, weight: float) -> float:
    if weight <= 0.0:
        return 0.0
    mean = sum_value / weight
    return math.sqrt(max(sum_square / weight - mean * mean, 0.0))


@_jitable
def _bounded_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def _update_h011_chunk(
    timestamps: NDArray[np.int64],
    wallet_ids: NDArray[np.int32],
    market_ids: NDArray[np.int32],
    outcome0_probabilities: NDArray[np.float32],
    raw_prices: NDArray[np.float32],
    sizes: NDArray[np.float32],
    notionals: NDArray[np.float32],
    outcome_indices: NDArray[np.int8],
    side_buys: NDArray[np.int8],
    directions: NDArray[np.int8],
    decision_slots: NDArray[np.int32],
    market_components: NDArray[np.int32],
    market_created: NDArray[np.int64],
    market_end: NDArray[np.int64],
    component_market_count: NDArray[np.int32],
    component_neg_risk_count: NDArray[np.int32],
    component_unclosed_count: NDArray[np.int32],
    actor_channel_enabled: int,
    resolution_timestamps: NDArray[np.int64],
    resolution_wallet_ids: NDArray[np.int32],
    resolution_edges: NDArray[np.float32],
    resolution_pnls: NDArray[np.float32],
    resolution_wins: NDArray[np.int8],
    resolution_pointer: int,
    wallet_core: NDArray[np.float64],
    actor_core: NDArray[np.float64],
    market_core: NDArray[np.float64],
    market_probability_ema: NDArray[np.float64],
    market_flow_ema: NDArray[np.float64],
    market_wallet_hll: NDArray[np.uint8],
    market_wallet_aggregate: NDArray[np.float64],
    component_core: NDArray[np.float64],
    component_probability_ema: NDArray[np.float64],
    component_flow_ema: NDArray[np.float64],
    component_market_hll: NDArray[np.uint8],
    universe_core: NDArray[np.float64],
    universe_probability_ema: NDArray[np.float64],
    universe_flow_ema: NDArray[np.float64],
    universe_market_hll: NDArray[np.uint8],
    universe_component_hll: NDArray[np.uint8],
    output_features: NDArray[np.float32],
) -> int:
    for row in range(timestamps.shape[0]):
        timestamp = int(timestamps[row])
        while (
            resolution_pointer < resolution_timestamps.shape[0]
            and int(resolution_timestamps[resolution_pointer]) < timestamp
        ):
            resolved_wallet = int(resolution_wallet_ids[resolution_pointer])
            wallet_core[resolved_wallet, 14] += 1.0
            wallet_core[resolved_wallet, 15] += float(resolution_edges[resolution_pointer])
            wallet_core[resolved_wallet, 16] += float(resolution_pnls[resolution_pointer])
            wallet_core[resolved_wallet, 17] += int(resolution_wins[resolution_pointer])
            resolution_pointer += 1
        wallet_id = int(wallet_ids[row])
        market_id = int(market_ids[row])
        component_id = int(market_components[market_id])
        probability = float(outcome0_probabilities[row])
        raw_price = float(raw_prices[row])
        size = float(sizes[row])
        notional = float(notionals[row])
        outcome_index = int(outcome_indices[row])
        side_buy = int(side_buys[row])
        direction = int(directions[row])
        signed_notional = direction * notional

        previous_global_timestamp = int(universe_core[7])
        previous_market_timestamp = int(market_core[market_id, 12])
        previous_component_timestamp = int(component_core[component_id, 9])
        previous_wallet_timestamp = int(wallet_core[wallet_id, 10])
        previous_wallet_market = int(wallet_core[wallet_id, 13])
        wallet_gap = (
            max(timestamp - previous_wallet_timestamp, 0) if previous_wallet_timestamp else 0
        )
        same_wallet_market = int(
            previous_wallet_market == market_id and wallet_core[wallet_id, 0] > 0
        )

        # Wallet state is post-evidence, while gap/same-market retain the pre-evidence context.
        wallet_core[wallet_id, 0] += 1.0
        wallet_core[wallet_id, 1] += notional
        wallet_core[wallet_id, 2] += size
        wallet_core[wallet_id, 3 + (1 - side_buy)] += 1.0
        wallet_core[wallet_id, 5 + outcome_index] += 1.0
        wallet_core[wallet_id, 7] += signed_notional
        wallet_core[wallet_id, 8] += probability * notional
        wallet_core[wallet_id, 9] += probability * probability * notional
        if previous_wallet_timestamp:
            wallet_core[wallet_id, 11] += wallet_gap
        if previous_wallet_market >= 0 and previous_wallet_market != market_id:
            wallet_core[wallet_id, 12] += 1.0
        wallet_core[wallet_id, 10] = timestamp
        wallet_core[wallet_id, 13] = market_id

        wallet_count = wallet_core[wallet_id, 0]
        wallet_notional = wallet_core[wallet_id, 1]
        wallet_mean_price = _ratio(wallet_core[wallet_id, 8], wallet_notional)
        wallet_price_std = _std(
            wallet_core[wallet_id, 8], wallet_core[wallet_id, 9], wallet_notional
        )
        resolution_count = wallet_core[wallet_id, 14]
        resolution_edge = _ratio(wallet_core[wallet_id, 15], resolution_count)
        resolution_pnl = _ratio(wallet_core[wallet_id, 16], resolution_count)
        actor_fills = actor_core[wallet_id, 0] + actor_core[wallet_id, 1]
        actor_available = int(actor_channel_enabled == 1 and actor_fills > 0.0)
        actor_maker_ratio = (
            _ratio(actor_core[wallet_id, 0], actor_fills) if actor_available else 0.0
        )
        wallet_recency_feature = math.log1p(wallet_gap) / 20.0 if previous_wallet_timestamp else 1.0
        wallet_novelty = 1.0 / math.sqrt(wallet_count)
        wallet_embedding = (
            math.log1p(wallet_count) / 20.0,
            math.log1p(wallet_notional) / 24.0,
            resolution_edge,
            math.tanh(resolution_pnl / 100.0),
            actor_maker_ratio,
            min(wallet_recency_feature, 1.0),
            wallet_novelty,
        )

        # Market state and all-wallet recurrent DeepSet moments.
        first_market_trade = market_core[market_id, 0] == 0.0
        market_core[market_id, 0] += 1.0
        market_core[market_id, 1] += notional
        market_core[market_id, 2] += size
        market_core[market_id, 3] += signed_notional
        market_core[market_id, 4 + (1 - side_buy)] += notional
        market_core[market_id, 6 + outcome_index] += notional
        market_core[market_id, 8] += probability * notional
        market_core[market_id, 9] += probability * probability * notional
        market_core[market_id, 10] = probability
        market_core[market_id, 11] = raw_price
        market_core[market_id, 12] = timestamp
        market_core[market_id, 13] += notional * notional
        if first_market_trade:
            market_core[market_id, 14] = timestamp
            market_core[market_id, 15] = probability
            market_core[market_id, 16] = probability
            for horizon in range(5):
                market_probability_ema[market_id, horizon] = probability
                market_flow_ema[market_id, horizon] = direction
        else:
            market_core[market_id, 15] = min(market_core[market_id, 15], probability)
            market_core[market_id, 16] = max(market_core[market_id, 16], probability)
            market_delta = max(timestamp - previous_market_timestamp, 0)
            for horizon in range(5):
                market_probability_ema[market_id, horizon] = _ema(
                    market_probability_ema[market_id, horizon],
                    probability,
                    market_delta,
                    HALF_LIVES[horizon],
                )
                market_flow_ema[market_id, horizon] = _ema(
                    market_flow_ema[market_id, horizon],
                    direction,
                    market_delta,
                    HALF_LIVES[horizon],
                )
        _hll_add(market_wallet_hll, market_id, wallet_id)
        aggregate = market_wallet_aggregate[market_id]
        aggregate[0] += notional
        aggregate[1] += notional * notional
        for index in range(7):
            value = wallet_embedding[index]
            aggregate[2 + index * 2] += value * notional
            aggregate[3 + index * 2] += value * value * notional
        aggregate[16] += actor_available * notional
        aggregate[17] += int(resolution_count > 0.0) * notional

        # Linked event/component state.
        first_component_trade = component_core[component_id, 0] == 0.0
        component_core[component_id, 0] += 1.0
        component_core[component_id, 1] += notional
        component_core[component_id, 2 + (1 - side_buy)] += notional
        component_core[component_id, 4 + outcome_index] += notional
        component_core[component_id, 6] += probability * notional
        component_core[component_id, 7] += probability * probability * notional
        component_core[component_id, 8] = probability
        component_core[component_id, 9] = timestamp
        component_core[component_id, 11] += signed_notional
        component_core[component_id, 12] += size
        if first_component_trade:
            component_core[component_id, 10] = timestamp
            for horizon in range(3):
                component_probability_ema[component_id, horizon] = probability
                component_flow_ema[component_id, horizon] = direction
        else:
            component_delta = max(timestamp - previous_component_timestamp, 0)
            for horizon in range(3):
                component_probability_ema[component_id, horizon] = _ema(
                    component_probability_ema[component_id, horizon],
                    probability,
                    component_delta,
                    COMPONENT_HALF_LIVES[horizon],
                )
                component_flow_ema[component_id, horizon] = _ema(
                    component_flow_ema[component_id, horizon],
                    direction,
                    component_delta,
                    COMPONENT_HALF_LIVES[horizon],
                )
        _hll_add(component_market_hll, component_id, market_id)

        # Universe state.
        first_universe_trade = universe_core[0] == 0.0
        universe_core[0] += 1.0
        universe_core[1] += notional
        universe_core[2 + (1 - side_buy)] += notional
        universe_core[4] += probability * notional
        universe_core[5] += probability * probability * notional
        universe_core[6] += signed_notional
        universe_core[7] = timestamp
        universe_core[9] += size
        universe_core[10 + outcome_index] += notional
        if first_universe_trade:
            universe_core[8] = timestamp
            for horizon in range(5):
                universe_probability_ema[horizon] = probability
                universe_flow_ema[horizon] = direction
        else:
            universe_delta = max(timestamp - previous_global_timestamp, 0)
            for horizon in range(5):
                universe_probability_ema[horizon] = _ema(
                    universe_probability_ema[horizon],
                    probability,
                    universe_delta,
                    HALF_LIVES[horizon],
                )
                universe_flow_ema[horizon] = _ema(
                    universe_flow_ema[horizon], direction, universe_delta, HALF_LIVES[horizon]
                )
        _hll_add(universe_market_hll, 0, market_id)
        _hll_add(universe_component_hll, 0, component_id)

        slot = int(decision_slots[row])
        if slot < 0:
            continue
        output = output_features[slot]
        market_count = market_core[market_id, 0]
        market_notional = market_core[market_id, 1]
        component_count = component_core[component_id, 0]
        component_notional = component_core[component_id, 1]
        global_count = universe_core[0]
        global_notional = universe_core[1]
        global_age = max(timestamp - int(universe_core[8]), 0)
        global_gap = (
            max(timestamp - previous_global_timestamp, 0) if previous_global_timestamp else 0
        )
        phase_day = (timestamp % 86_400) / 86_400.0
        phase_week = (timestamp % 604_800) / 604_800.0
        output[0] = math.log1p(global_count)
        output[1] = math.log1p(global_notional)
        output[2] = math.log1p(global_age / 86_400.0)
        output[3] = math.sin(2.0 * math.pi * phase_day)
        output[4] = math.cos(2.0 * math.pi * phase_day)
        output[5] = math.sin(2.0 * math.pi * phase_week)
        output[6] = math.cos(2.0 * math.pi * phase_week)
        output[7] = math.log1p(global_gap)

        market_mean = _ratio(market_core[market_id, 8], market_notional)
        market_std = _std(market_core[market_id, 8], market_core[market_id, 9], market_notional)
        market_wallets = _hll_estimate(market_wallet_hll, market_id)
        market_age = max(timestamp - int(market_core[market_id, 14]), 0)
        market_gap = (
            max(timestamp - previous_market_timestamp, 0) if previous_market_timestamp else 0
        )
        output[8] = math.log1p(market_count)
        output[9] = math.log1p(market_notional)
        output[10] = math.log1p(market_core[market_id, 2])
        output[11] = probability
        output[12] = raw_price
        output[13] = outcome_index
        output[14] = side_buy
        output[15] = math.log1p(notional)
        output[16] = math.log1p(size)
        output[17] = _ratio(market_core[market_id, 3], market_notional)
        output[18] = _ratio(market_core[market_id, 4] - market_core[market_id, 5], market_notional)
        output[19] = _ratio(market_core[market_id, 6] - market_core[market_id, 7], market_notional)
        output[20] = market_mean
        output[21] = market_std
        output[22] = market_core[market_id, 15]
        output[23] = market_core[market_id, 16]
        output[24] = math.log1p(market_gap)
        output[25] = math.log1p(market_age)
        output[26] = math.log1p(market_wallets)
        output[27] = math.log1p(_ratio(market_count, max(market_wallets, 1.0)))
        output[28] = _ratio(market_core[market_id, 13], market_notional * market_notional)
        output[29] = _ratio(notional, market_notional)
        for horizon in range(5):
            output[30 + horizon] = market_probability_ema[market_id, horizon]
            output[35 + horizon] = market_flow_ema[market_id, horizon]
        output[40] = probability - market_probability_ema[market_id, 0]
        output[41] = probability - market_probability_ema[market_id, 1]
        output[42] = probability - market_probability_ema[market_id, 3]
        output[43] = probability - market_probability_ema[market_id, 4]
        output[44] = math.log1p(_ratio(market_notional, market_count))
        output[45] = math.log1p(_ratio(market_notional * 86_400.0, max(market_age, 1)))
        created = int(market_created[market_id])
        end = int(market_end[market_id])
        if created > 0 and end > created:
            output[46] = _bounded_fraction(timestamp - created, end - created)
            output[47] = _bounded_fraction(end - timestamp, end - created)

        component_mean = _ratio(component_core[component_id, 6], component_notional)
        component_std = _std(
            component_core[component_id, 6],
            component_core[component_id, 7],
            component_notional,
        )
        component_markets = _hll_estimate(component_market_hll, component_id)
        component_age = max(timestamp - int(component_core[component_id, 10]), 0)
        component_gap = (
            max(timestamp - previous_component_timestamp, 0) if previous_component_timestamp else 0
        )
        static_market_count = max(int(component_market_count[component_id]), 1)
        output[48] = math.log1p(component_count)
        output[49] = math.log1p(component_notional)
        output[50] = _ratio(component_core[component_id, 11], component_notional)
        output[51] = _ratio(
            component_core[component_id, 2] - component_core[component_id, 3],
            component_notional,
        )
        output[52] = _ratio(
            component_core[component_id, 4] - component_core[component_id, 5],
            component_notional,
        )
        output[53] = component_mean
        output[54] = component_std
        output[55] = probability
        output[56] = math.log1p(component_gap)
        output[57] = math.log1p(component_age)
        output[58] = math.log1p(component_markets)
        output[59] = math.log1p(_ratio(component_count, max(component_markets, 1.0)))
        output[60] = _ratio(market_count, component_count)
        output[61] = _ratio(market_notional, component_notional)
        output[62] = math.log1p(static_market_count)
        output[63] = component_neg_risk_count[component_id] / static_market_count
        output[64] = component_unclosed_count[component_id] / static_market_count
        for horizon in range(3):
            output[65 + horizon] = component_probability_ema[component_id, horizon]
            output[68 + horizon] = component_flow_ema[component_id, horizon]
        output[71] = math.log1p(_ratio(component_notional * 86_400.0, max(component_age, 1)))

        wallet_buy_sell = wallet_core[wallet_id, 3] + wallet_core[wallet_id, 4]
        wallet_outcome = wallet_core[wallet_id, 5] + wallet_core[wallet_id, 6]
        output[72] = math.log1p(wallet_count)
        output[73] = math.log1p(wallet_notional)
        output[74] = math.log1p(wallet_core[wallet_id, 2])
        output[75] = _ratio(wallet_core[wallet_id, 3] - wallet_core[wallet_id, 4], wallet_buy_sell)
        output[76] = _ratio(wallet_core[wallet_id, 5] - wallet_core[wallet_id, 6], wallet_outcome)
        output[77] = _ratio(wallet_core[wallet_id, 7], wallet_notional)
        output[78] = wallet_mean_price
        output[79] = wallet_price_std
        output[80] = min(wallet_recency_feature, 1.0)
        output[81] = math.log1p(_ratio(wallet_core[wallet_id, 11], max(wallet_count - 1.0, 1.0)))
        output[82] = _ratio(wallet_core[wallet_id, 12], max(wallet_count - 1.0, 1.0))
        output[83] = same_wallet_market
        output[84] = _ratio(notional, wallet_notional)
        output[85] = math.log1p(resolution_count)
        output[86] = resolution_edge
        output[87] = resolution_pnl
        output[88] = _ratio(wallet_core[wallet_id, 17], resolution_count)
        output[89] = actor_available
        if actor_available:
            actor_buy_sell = actor_core[wallet_id, 2] + actor_core[wallet_id, 3]
            actor_notional = actor_core[wallet_id, 4] + actor_core[wallet_id, 5]
            actor_price_weight = actor_core[wallet_id, 11]
            actor_mean_price = _ratio(actor_core[wallet_id, 9], actor_price_weight)
            output[90] = math.log1p(actor_core[wallet_id, 0])
            output[91] = math.log1p(actor_core[wallet_id, 1])
            output[92] = actor_maker_ratio
            output[93] = _ratio(actor_core[wallet_id, 2] - actor_core[wallet_id, 3], actor_buy_sell)
            output[94] = math.log1p(actor_notional)
            output[95] = math.log1p(actor_core[wallet_id, 7])
            output[96] = math.log1p(actor_core[wallet_id, 8])
            output[97] = actor_mean_price
            output[98] = _std(
                actor_core[wallet_id, 9], actor_core[wallet_id, 10], actor_price_weight
            )
        output[99] = wallet_novelty

        aggregate_weight = aggregate[0]
        for index in range(4):
            output[100 + index * 2] = _ratio(aggregate[2 + index * 2], aggregate_weight)
            output[101 + index * 2] = _std(
                aggregate[2 + index * 2],
                aggregate[3 + index * 2],
                aggregate_weight,
            )
        output[108] = _ratio(aggregate[10], aggregate_weight)
        output[109] = _ratio(aggregate[16], aggregate_weight)
        output[110] = _ratio(aggregate[17], aggregate_weight)
        output[111] = math.log1p(aggregate_weight)
        output[112] = math.log1p(_ratio(aggregate_weight * aggregate_weight, aggregate[1]))
        output[113] = _ratio(aggregate[12], aggregate_weight)
        output[114] = _ratio(aggregate[14], aggregate_weight)
        output[115] = _ratio(notional, aggregate_weight)

        global_mean = _ratio(universe_core[4], global_notional)
        global_std = _std(universe_core[4], universe_core[5], global_notional)
        output[116] = math.log1p(global_count)
        output[117] = math.log1p(global_notional)
        output[118] = global_mean
        output[119] = global_std
        output[120] = _ratio(universe_core[6], global_notional)
        output[121] = _ratio(universe_core[2] - universe_core[3], global_notional)
        output[122] = math.log1p(_hll_estimate(universe_market_hll, 0))
        output[123] = math.log1p(_hll_estimate(universe_component_hll, 0))
        output[124] = _ratio(component_count, global_count)
        output[125] = _ratio(component_notional, global_notional)
        output[126] = math.log1p(_ratio(global_notional * 86_400.0, max(global_age, 1)))
        output[127] = universe_flow_ema[3]
    return resolution_pointer


Kernel = Callable[..., int]


def compile_h011_kernel() -> Kernel:
    """Compile the hot loop lazily so lightweight package imports stay cheap."""

    if numba_module is None:
        raise RuntimeError("H011 feature packing requires the research numba dependency")
    compiled: Any = numba_module.njit(cache=True, nogil=True)(_update_h011_chunk)
    return cast(Kernel, compiled)


def python_h011_kernel() -> Kernel:
    """Expose the reference implementation for deterministic unit tests."""

    return _update_h011_chunk
