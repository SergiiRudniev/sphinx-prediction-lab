"""Uncertainty estimates for H010 weekly and independent-component profit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def moving_block_bootstrap_mean(
    values: NDArray[np.float64],
    *,
    replicates: int,
    block_length: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a serial weekly mean with circular contiguous blocks."""

    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or replicates <= 0
        or block_length <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("H010 moving-block bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(len(values) / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    sampled = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, len(values), size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % len(values)).reshape(-1)
        sampled[replicate] = values[indices[: len(values)]].mean()
    tail = (1.0 - confidence) / 2.0
    return {
        "observations": len(values),
        "replicates": replicates,
        "block_length": block_length,
        "mean": float(values.mean()),
        "lower": float(np.quantile(sampled, tail)),
        "upper": float(np.quantile(sampled, 1.0 - tail)),
    }


def independent_component_bootstrap(
    component_profit: NDArray[np.float64],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap equal-weight called-component realized profit."""

    if (
        component_profit.ndim != 1
        or not len(component_profit)
        or not np.isfinite(component_profit).all()
        or replicates <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("H010 component bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    batch = max(1, min(128, 2_000_000 // len(component_profit)))
    for offset in range(0, replicates, batch):
        count = min(batch, replicates - offset)
        indices = rng.integers(0, len(component_profit), size=(count, len(component_profit)))
        sampled[offset : offset + count] = component_profit[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "components": len(component_profit),
        "replicates": replicates,
        "mean_profit_usd": float(component_profit.mean()),
        "total_profit_usd": float(component_profit.sum()),
        "positive_fraction": float(np.mean(component_profit > 0.0)),
        "lower_mean_profit_usd": float(np.quantile(sampled, tail)),
        "upper_mean_profit_usd": float(np.quantile(sampled, 1.0 - tail)),
    }


def promotion_gates(
    weekly: dict[str, Any],
    components: dict[str, Any],
    *,
    minimum_calls: int,
    minimum_components: int,
    calls: int,
) -> dict[str, bool]:
    gates = {
        "weekly_lower_positive": float(weekly["lower"]) > 0.0,
        "component_lower_positive": float(components["lower_mean_profit_usd"]) > 0.0,
        "minimum_calls": calls >= minimum_calls,
        "minimum_components": int(components["components"]) >= minimum_components,
    }
    return {**gates, "all_pass": all(gates.values())}
