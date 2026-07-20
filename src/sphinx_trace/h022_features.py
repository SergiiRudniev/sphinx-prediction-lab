"""Feature and split contract for the H022 conditional net-edge ensemble."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sphinx_trace.h011_features import H011_FEATURE_NAMES, H011_FEATURE_WIDTH

H022_WALLET_START = 72
H022_WALLET_STOP = 116
H022_ENGINEERED_FEATURE_NAMES = (
    "outcome.terminal_logit",
    "outcome.uncertainty_log_scale",
    *(f"portfolio.{index}" for index in range(9)),
    *(f"prediction_memory.{index}" for index in range(7)),
    "base.logit_call0",
    "base.logit_call1",
    "base.logit_skip",
    "base.protocol_value_call0",
    "base.protocol_value_call1",
    "base.protocol_value_skip",
    "execution.entry_price_call0",
    "execution.entry_price_call1",
    "execution.payout_call0",
    "execution.payout_call1",
    "execution.market_probability_call0",
    "execution.market_probability_call1",
    "candidate.side_call0",
    "candidate.side_call1",
    "candidate.entry_price",
    "candidate.opposite_entry_price",
    "candidate.payout",
    "candidate.break_even_probability",
    "candidate.market_probability",
    "candidate.terminal_probability",
    "candidate.market_edge",
    "candidate.terminal_edge",
    "candidate.price_logit",
    "candidate.upside_per_cost",
)
H022_TREE_FEATURE_NAMES = H011_FEATURE_NAMES + H022_ENGINEERED_FEATURE_NAMES
H022_TREE_FEATURE_WIDTH = len(H022_TREE_FEATURE_NAMES)
H022_WALLET_TREE_INDICES = np.arange(
    H022_WALLET_START, H022_WALLET_STOP, dtype=np.int64
)


def component_folds(
    component_ids: NDArray[np.int64], folds: int, seed: int
) -> NDArray[np.uint8]:
    """Assign all rows from one component to one deterministic OOF fold."""

    if component_ids.ndim != 1 or folds < 2 or folds > 255:
        raise ValueError("H022 fold inputs are invalid")
    values = component_ids.astype(np.uint64, copy=True)
    values += np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    return np.asarray(values % np.uint64(folds), dtype=np.uint8)


def assemble_tree_features(
    market_features: NDArray[np.float32],
    terminal_logits: NDArray[np.float32],
    uncertainty_log_scales: NDArray[np.float32],
    portfolio_features: NDArray[np.float32],
    prediction_memory_features: NDArray[np.float32],
    base_action_logits: NDArray[np.float32],
    protocol_action_values: NDArray[np.float32],
    execution_context: NDArray[np.float32],
    candidate_action_ids: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Build the causal price/event/activity/wallet control matrix."""

    rows = len(terminal_logits)
    expected = (
        market_features.shape == (rows, H011_FEATURE_WIDTH)
        and uncertainty_log_scales.shape == (rows,)
        and portfolio_features.shape == (rows, 9)
        and prediction_memory_features.shape == (rows, 7)
        and base_action_logits.shape == (rows, 3)
        and protocol_action_values.shape == (rows, 3)
        and execution_context.shape == (rows, 6)
        and candidate_action_ids.shape == (rows,)
    )
    if not expected or bool(((candidate_action_ids < 0) | (candidate_action_ids > 1)).any()):
        raise ValueError("H022 candidate feature arrays are not aligned")
    side = candidate_action_ids[:, None]
    opposite = 1 - side
    prices = execution_context[:, :2]
    payouts = execution_context[:, 2:4]
    market_probabilities = execution_context[:, 4:6]
    candidate_price = np.take_along_axis(prices, side, axis=1)
    opposite_price = np.take_along_axis(prices, opposite, axis=1)
    candidate_payout = np.take_along_axis(payouts, side, axis=1)
    break_even = np.minimum(1.0, 1.0 / np.maximum(candidate_payout, 1e-8))
    market_probability = np.take_along_axis(market_probabilities, side, axis=1)
    terminal_q0 = 1.0 / (1.0 + np.exp(-np.clip(terminal_logits, -30.0, 30.0)))
    terminal_probabilities = np.stack((terminal_q0, 1.0 - terminal_q0), axis=1)
    terminal_probability = np.take_along_axis(terminal_probabilities, side, axis=1)
    clipped_price = np.clip(candidate_price, 1e-6, 1.0 - 1e-6)
    side_one_hot = np.eye(2, dtype=np.float32)[candidate_action_ids]
    engineered = np.concatenate(
        (
            terminal_logits[:, None],
            uncertainty_log_scales[:, None],
            portfolio_features,
            prediction_memory_features,
            base_action_logits,
            protocol_action_values,
            execution_context,
            side_one_hot,
            candidate_price,
            opposite_price,
            candidate_payout,
            break_even,
            market_probability,
            terminal_probability,
            market_probability - break_even,
            terminal_probability - break_even,
            np.log(clipped_price / (1.0 - clipped_price)),
            candidate_payout - 1.0,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    output = np.concatenate((market_features, engineered), axis=1).astype(
        np.float32, copy=False
    )
    if output.shape != (rows, H022_TREE_FEATURE_WIDTH) or not bool(
        np.isfinite(output).all()
    ):
        raise RuntimeError("H022 assembled feature matrix is invalid")
    return output


def wallet_ablation(
    features: NDArray[np.float32], mode: str, *, seed: int
) -> NDArray[np.float32]:
    """Return a copy with wallet channels zeroed or jointly row-shuffled."""

    if features.ndim != 2 or features.shape[1] != H022_TREE_FEATURE_WIDTH:
        raise ValueError("H022 wallet ablation received an invalid matrix")
    output = np.array(features, copy=True)
    if mode == "none":
        return output
    if mode == "zero":
        output[:, H022_WALLET_TREE_INDICES] = 0.0
        return output
    if mode == "shuffle":
        permutation = np.random.default_rng(seed).permutation(len(output))
        output[:, H022_WALLET_TREE_INDICES] = output[
            permutation[:, None], H022_WALLET_TREE_INDICES[None, :]
        ]
        return output
    raise ValueError(f"Unknown H022 wallet ablation: {mode}")


def selected_reference_utility(
    reference_action_values: NDArray[np.float32],
    candidate_action_ids: NDArray[np.int64],
) -> NDArray[np.float32]:
    if reference_action_values.shape != (len(candidate_action_ids), 3):
        raise ValueError("H022 reference values do not align with candidate actions")
    return np.asarray(
        np.take_along_axis(
            reference_action_values, candidate_action_ids[:, None], axis=1
        )[:, 0],
        dtype=np.float32,
    )
