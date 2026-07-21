from __future__ import annotations

import pytest

from sphinx_trace.h022_analysis import (
    CandidateScore,
    diagnostic_gate_keep,
    price_bin,
    reference_log_utility,
)


@pytest.mark.parametrize(
    ("price", "expected"),
    ((0.49, "below_0_50"), (0.50, "0_50_to_0_70"), (0.98, "at_least_0_98")),
)
def test_price_bin_is_left_closed(price: float, expected: str) -> None:
    assert price_bin(price) == expected


def test_reference_log_utility_rewards_cheap_winner_and_penalizes_loser() -> None:
    assert reference_log_utility(0.1, 2.0, 1.0) > 0.0
    assert reference_log_utility(0.1, 2.0, 0.0) < 0.0


def test_optimistic_veto_is_less_aggressive_than_mean_gate() -> None:
    score = CandidateScore(ensemble=-0.01, q50=-0.005, q90=0.02)
    assert diagnostic_gate_keep("h022_mean_positive", score) is False
    assert (
        diagnostic_gate_keep("h022_mean_or_optimistic_quantile_nonnegative", score)
        is True
    )
