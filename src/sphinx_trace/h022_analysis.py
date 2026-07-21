"""Pure audit-analysis helpers for H022 exact replay diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

PRICE_BINS = (
    ("below_0_50", 0.0, 0.50),
    ("0_50_to_0_70", 0.50, 0.70),
    ("0_70_to_0_80", 0.70, 0.80),
    ("0_80_to_0_90", 0.80, 0.90),
    ("0_90_to_0_95", 0.90, 0.95),
    ("0_95_to_0_98", 0.95, 0.98),
    ("at_least_0_98", 0.98, float("inf")),
)

DIAGNOSTIC_GATES = (
    "all_h021_candidates",
    "h022_mean_positive",
    "h022_median_nonnegative",
    "h022_optimistic_quantile_nonnegative",
    "h022_mean_or_median_nonnegative",
    "h022_mean_or_optimistic_quantile_nonnegative",
)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    ensemble: float
    q50: float
    q90: float


def price_bin(price: float) -> str:
    """Return the registered left-closed price bucket."""

    if not math.isfinite(price) or price < 0.0:
        raise ValueError("Price must be finite and non-negative")
    for name, lower, upper in PRICE_BINS:
        if lower <= price < upper:
            return name
    raise RuntimeError("Price bin coverage changed")


def reference_log_utility(
    size_fraction: float,
    winning_payout_per_cost: float,
    terminal_payout_rate: float,
) -> float:
    """Evaluate the frozen H022 reference position at terminal payout."""

    if (
        not 0.0 <= size_fraction <= 1.0
        or winning_payout_per_cost < 0.0
        or not 0.0 <= terminal_payout_rate <= 1.0
    ):
        raise ValueError("Reference utility inputs are outside their contract")
    wealth = (
        1.0
        - size_fraction
        + size_fraction * winning_payout_per_cost * terminal_payout_rate
    )
    return math.log(max(wealth, 1e-8))


def diagnostic_gate_keep(gate_id: str, score: CandidateScore) -> bool:
    """Apply one label-free model-score diagnostic gate."""

    if gate_id == "all_h021_candidates":
        return True
    if gate_id == "h022_mean_positive":
        return score.ensemble > 0.0
    if gate_id == "h022_median_nonnegative":
        return score.q50 >= 0.0
    if gate_id == "h022_optimistic_quantile_nonnegative":
        return score.q90 >= 0.0
    if gate_id == "h022_mean_or_median_nonnegative":
        return score.ensemble > 0.0 or score.q50 >= 0.0
    if gate_id == "h022_mean_or_optimistic_quantile_nonnegative":
        return score.ensemble > 0.0 or score.q90 >= 0.0
    raise ValueError(f"Unknown H022 diagnostic gate: {gate_id}")
