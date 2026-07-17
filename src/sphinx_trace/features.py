"""Causal feature packing for the Sphinx Trace S0 learning preflight."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_trace.chronicle import yes_direction, yes_equivalent_price


@dataclass(frozen=True, slots=True)
class WalletEvent:
    timestamp_unix: int
    price: float
    size: float
    notional: float
    side: int
    outcome_index: int
    market_key: int


def wallet_event(row: dict[str, Any]) -> WalletEvent | None:
    try:
        timestamp = int(row["timestamp_unix"])
        price = float(row["price"])
        size = float(row["size"])
        notional = float(row["notional_usd"])
        outcome_index = int(row["outcome_index"])
    except (KeyError, TypeError, ValueError):
        return None
    side_text = str(row.get("side") or "").upper()
    if side_text not in {"BUY", "SELL"} or outcome_index not in {0, 1}:
        return None
    condition_id = str(row.get("condition_id") or "")
    try:
        market_key = int(condition_id[-16:], 16)
    except ValueError:
        return None
    return WalletEvent(
        timestamp_unix=timestamp,
        price=price,
        size=size,
        notional=notional,
        side=1 if side_text == "BUY" else -1,
        outcome_index=outcome_index,
        market_key=market_key,
    )


def _number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _target_arrays(
    target: dict[str, Any], output_order: list[str]
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    values = np.zeros(len(output_order), dtype=np.float32)
    mask = np.zeros(len(output_order), dtype=np.uint8)
    nested = target["targets"]
    for index, name in enumerate(output_order):
        value = target["resolved_yes"] if name == "resolved_yes" else nested.get(name)
        if value is None:
            continue
        values[index] = float(value)
        mask[index] = 1
    return values, mask


def build_feature_sequence(
    market_rows: list[dict[str, Any]],
    wallet_histories: dict[str, list[WalletEvent]],
    target: dict[str, Any],
    config: dict[str, Any],
) -> (
    tuple[
        NDArray[np.float16],
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.uint8],
    ]
    | None
):
    feature_config = config["features"]
    trade_tokens = int(feature_config["market_trade_tokens"])
    wallet_tokens = int(feature_config["wallet_tokens"])
    context_tokens = int(feature_config["context_tokens"])
    feature_width = int(feature_config["feature_width"])
    wallet_limit = int(feature_config["wallet_history_events"])
    sequence_length = trade_tokens + wallet_tokens + context_tokens
    decision_unix = int(target["decision_time_unix"])
    causal_rows = sorted(
        (row for row in market_rows if int(row.get("timestamp_unix") or 0) < decision_unix),
        key=lambda row: (
            int(row.get("timestamp_unix") or 0),
            str(row.get("transaction_hash") or ""),
            str(row.get("trade_id") or ""),
        ),
    )
    if len(causal_rows) < trade_tokens:
        return None
    rows = causal_rows[-trade_tokens:]
    timestamps = np.asarray([int(row["timestamp_unix"]) for row in rows], dtype=np.int64)
    prices = np.asarray([yes_equivalent_price(row) or 0.0 for row in rows])
    sizes = np.asarray([_number(row, "size") for row in rows])
    notionals = np.asarray([_number(row, "notional_usd") for row in rows])
    directions = np.asarray([yes_direction(row) or 0 for row in rows])
    outcomes = np.asarray([int(row.get("outcome_index") or 0) for row in rows])
    wallets = [str(row.get("wallet") or "").lower() for row in rows]
    features = np.zeros((sequence_length, feature_width), dtype=np.float16)
    token_types = np.zeros(sequence_length, dtype=np.uint8)

    deltas = np.diff(timestamps, prepend=timestamps[0])
    day_phase = np.mod(timestamps, 86400) / 86400.0
    prior_counts: Counter[str] = Counter()
    prior_notionals: defaultdict[str, float] = defaultdict(float)
    total_notional = max(float(notionals.sum()), 1.0)
    maximum_notional = max(float(notionals.max()), 1.0)
    market = features[:trade_tokens]
    for index, wallet in enumerate(wallets):
        market[index, 0] = prices[index]
        market[index, 1] = np.clip(np.log1p(sizes[index]) / 12.0, 0.0, 1.0)
        market[index, 2] = np.clip(np.log1p(notionals[index]) / 12.0, 0.0, 1.0)
        market[index, 3] = directions[index]
        market[index, 4] = np.clip(np.log1p(max(int(deltas[index]), 0)) / 12.0, 0.0, 1.0)
        market[index, 5] = outcomes[index]
        market[index, 6] = math.sin(2.0 * math.pi * day_phase[index])
        market[index, 7] = math.cos(2.0 * math.pi * day_phase[index])
        market[index, 8] = min(prior_counts[wallet] / trade_tokens, 1.0)
        market[index, 9] = min(prior_notionals[wallet] / total_notional, 1.0)
        market[index, 10] = index / max(trade_tokens - 1, 1)
        market[index, 11] = notionals[: index + 1].sum() / total_notional
        market[index, 12] = np.clip(directions[: index + 1].sum() / trade_tokens, -1, 1)
        market[index, 13] = np.clip(prices[index] - prices[: index + 1].mean(), -1, 1)
        market[index, 14] = notionals[index] / maximum_notional
        market[index, 15] = 1.0
        prior_counts[wallet] += 1
        prior_notionals[wallet] += float(notionals[index])

    wallet_start = trade_tokens
    token_types[wallet_start : wallet_start + wallet_tokens] = 1
    market_wallet_notional: defaultdict[str, float] = defaultdict(float)
    for wallet, notional in zip(wallets, notionals, strict=True):
        market_wallet_notional[wallet] += float(notional)
    ranked_wallets = sorted(
        market_wallet_notional,
        key=lambda wallet: (-market_wallet_notional[wallet], wallet),
    )[:wallet_tokens]
    for slot, wallet in enumerate(ranked_wallets):
        history = wallet_histories.get(wallet, [])
        history_timestamps = [event.timestamp_unix for event in history]
        cutoff = bisect_left(history_timestamps, decision_unix)
        past = history[max(0, cutoff - wallet_limit) : cutoff]
        if not past:
            continue
        output = features[wallet_start + slot]
        history_notional = sum(event.notional for event in past)
        history_sizes = [event.size for event in past]
        history_prices = [event.price for event in past]
        history_sides = [event.side for event in past]
        output[0] = min(len(past) / wallet_limit, 1.0)
        output[1] = np.clip(np.log1p(history_notional) / 16.0, 0.0, 1.0)
        output[2] = float(np.mean(history_prices))
        output[3] = float(np.mean(history_sides))
        output[4] = min(len({event.market_key for event in past}) / 32.0, 1.0)
        output[5] = np.clip(
            np.log1p(max(past[-1].timestamp_unix - past[0].timestamp_unix, 0)) / 18.0,
            0.0,
            1.0,
        )
        output[6] = np.clip(
            np.log1p(max(decision_unix - past[-1].timestamp_unix, 0)) / 18.0,
            0.0,
            1.0,
        )
        output[7] = np.clip(np.log1p(float(np.mean(history_sizes))) / 12.0, 0.0, 1.0)
        output[8] = min(max(event.notional for event in past) / maximum_notional, 1.0)
        output[9] = min(float(np.std(history_prices)), 1.0)
        output[10] = sum(side > 0 for side in history_sides) / len(past)
        output[11] = float(np.mean([event.outcome_index for event in past]))
        output[12] = min(market_wallet_notional[wallet] / total_notional, 1.0)
        output[13] = min(wallets.count(wallet) / trade_tokens, 1.0)
        output[14] = min(history_notional / max(total_notional, 1.0), 1.0)
        output[15] = 1.0

    context_start = trade_tokens + wallet_tokens
    token_types[context_start:] = 2
    block_size = max(1, trade_tokens // context_tokens)
    for slot in range(context_tokens):
        start = slot * block_size
        end = min(start + block_size, trade_tokens)
        if start >= end:
            break
        output = features[context_start + slot]
        block_prices = prices[start:end]
        block_notionals = notionals[start:end]
        block_directions = directions[start:end]
        output[0] = float(np.mean(block_prices))
        output[1] = float(np.std(block_prices))
        output[2] = np.clip(np.log1p(block_notionals.sum()) / 14.0, 0.0, 1.0)
        output[3] = float(np.mean(block_directions))
        output[4] = min(len(set(wallets[start:end])) / block_size, 1.0)
        output[5] = start / trade_tokens
        output[6] = end / trade_tokens
        output[7] = np.clip(
            np.log1p(max(decision_unix - int(timestamps[end - 1]), 0)) / 18.0,
            0.0,
            1.0,
        )
        output[8] = float(np.mean(block_directions > 0))
        output[9] = block_notionals.sum() / total_notional
        output[10] = float(np.min(block_prices))
        output[11] = float(np.max(block_prices))
        output[12] = float(np.mean(outcomes[start:end]))
        output[13] = np.clip(
            float(block_prices[-1] - block_prices[0]),
            -1.0,
            1.0,
        )
        output[14] = min(float(block_notionals.max()) / maximum_notional, 1.0)
        output[15] = 1.0

    output_order = [str(value) for value in config["targets"]["output_order"]]
    target_values, target_mask = _target_arrays(target, output_order)
    return features, token_types, target_values, target_mask
