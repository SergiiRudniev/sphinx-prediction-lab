from __future__ import annotations

import numpy as np
import pytest
from scripts.calibrate_h011_outcome import component_bootstrap, fit_platt


def _config() -> dict[str, float | int]:
    return {
        "initial_slope": 1.0,
        "initial_intercept": 0.0,
        "minimum_slope": 0.01,
        "maximum_slope": 100.0,
        "ridge": 1e-6,
        "maximum_iterations": 100,
        "tolerance": 1e-9,
        "line_search_steps": 24,
    }


def test_platt_fit_recovers_held_out_probability_scale() -> None:
    rng = np.random.default_rng(17)
    logits = rng.normal(size=100_000)
    expected = 1.0 / (1.0 + np.exp(-(0.55 * logits - 0.25)))
    labels = rng.binomial(1, expected).astype(np.float64)
    slope, intercept, history = fit_platt(logits, labels, _config())
    assert slope == pytest.approx(0.55, abs=0.04)
    assert intercept == pytest.approx(-0.25, abs=0.04)
    assert history[-1] <= history[0]


def test_component_bootstrap_is_deterministic_and_component_weighted() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    model = np.array([0.8, 0.8, 0.2, 0.2])
    baseline = np.full(4, 0.5)
    components = np.array([1, 1, 2, 3], dtype=np.int64)
    first = component_bootstrap(
        model,
        baseline,
        labels,
        components,
        replicates=500,
        seed=17,
        confidence=0.95,
    )
    second = component_bootstrap(
        model,
        baseline,
        labels,
        components,
        replicates=500,
        seed=17,
        confidence=0.95,
    )
    assert first == second
    assert first["components"] == 3
    assert float(first["upper"]) < 0.0
