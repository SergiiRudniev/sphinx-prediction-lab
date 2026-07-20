"""Losses, stacking and diagnostics for H022 conditional net edge."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


def h022_neural_loss(
    output: dict[str, Tensor],
    target_utility: Tensor,
    target_outcome0: Tensor,
    target_fill_fraction: Tensor,
    fill_target_mask: Tensor,
    sample_weights: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Fit calibrated probability, return distribution, fill and keep regret."""

    rows = len(target_utility)
    if any(
        value.shape != (rows,)
        for value in (
            target_outcome0,
            target_fill_fraction,
            fill_target_mask,
            sample_weights,
        )
    ):
        raise ValueError("H022 neural targets are not aligned")
    weights = sample_weights.float()
    if bool((weights <= 0.0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("H022 sample weights must be finite and positive")
    weight_sum = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor, row_weights: Tensor = weights) -> Tensor:
        return (values * row_weights).sum() / row_weights.sum().clamp_min(1e-8)

    outcome_rows = F.binary_cross_entropy_with_logits(
        output["calibrated_outcome_logit"].float(),
        target_outcome0.float(),
        reduction="none",
    )
    mean_rows = F.smooth_l1_loss(
        output["net_return_mean"].float(),
        target_utility.float(),
        reduction="none",
        beta=0.005,
    )
    quantiles = output["net_return_quantiles"].float()
    errors = target_utility[:, None].float() - quantiles
    quantile_levels = torch.tensor(
        [0.1, 0.5, 0.9], dtype=errors.dtype, device=errors.device
    )
    quantile_rows = torch.maximum(
        quantile_levels * errors, (quantile_levels - 1.0) * errors
    ).mean(dim=1)
    fill_rows = F.binary_cross_entropy_with_logits(
        output["fill_logit"].float(),
        target_fill_fraction.float().clamp(0.0, 1.0),
        reduction="none",
    )
    fill_weights = weights * fill_target_mask.float()
    fill_loss = (
        weighted_mean(fill_rows, fill_weights)
        if bool((fill_weights > 0.0).any())
        else torch.zeros((), dtype=weights.dtype, device=weights.device)
    )
    decision_temperature = float(config.get("decision_temperature", 0.005))
    if decision_temperature <= 0.0:
        raise ValueError("H022 decision temperature must be positive")
    keep_probability = torch.sigmoid(
        output["net_return_mean"].float() / decision_temperature
    )
    loss_weights = config["loss_weights"]
    decision_regret = (
        keep_probability
        * torch.relu(-target_utility.float())
        * float(loss_weights["wrong_keep_utility"])
        + (1.0 - keep_probability)
        * torch.relu(target_utility.float())
        * float(loss_weights["profitable_skip_utility"])
    )
    outcome_loss = weighted_mean(outcome_rows)
    mean_loss = weighted_mean(mean_rows)
    quantile_loss = weighted_mean(quantile_rows)
    regret_loss = weighted_mean(decision_regret)
    loss = (
        float(loss_weights["outcome_bce"]) * outcome_loss
        + float(loss_weights["net_return_mean"]) * mean_loss
        + float(loss_weights["net_return_quantiles"]) * quantile_loss
        + float(loss_weights["fill_bce"]) * fill_loss
        + regret_loss
    )
    hard_keep = output["net_return_mean"] > 0.0
    chosen = torch.where(hard_keep, target_utility, torch.zeros_like(target_utility))
    return loss, {
        "loss": loss.detach(),
        "outcome_bce": outcome_loss.detach(),
        "net_return_mean_loss": mean_loss.detach(),
        "net_return_quantile_loss": quantile_loss.detach(),
        "fill_bce": fill_loss.detach(),
        "decision_regret": regret_loss.detach(),
        "weighted_chosen_utility": weighted_mean(chosen).detach(),
        "calls": hard_keep.sum().detach(),
        "weight_sum": weight_sum.detach(),
    }


def fit_weighted_ridge(
    predictors: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    ridge: float,
) -> dict[str, Any]:
    """Fit a reproducible standardized ridge stacker with an unpenalized intercept."""

    rows, width = predictors.shape
    if (
        target.shape != (rows,)
        or weights.shape != (rows,)
        or ridge < 0.0
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("H022 ridge inputs are invalid")
    total = float(weights.sum())
    mean = (predictors * weights[:, None]).sum(axis=0) / total
    variance = (
        np.square(predictors - mean) * weights[:, None]
    ).sum(axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (predictors - mean) / scale
    design = np.concatenate(
        (np.ones((rows, 1), dtype=np.float64), standardized), axis=1
    )
    weighted_design = design * weights[:, None]
    gram = design.T @ weighted_design
    penalty = np.eye(width + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty, weighted_design.T @ target
    )
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "ridge": ridge,
    }


def predict_weighted_ridge(
    predictors: NDArray[np.float64], receipt: dict[str, Any]
) -> NDArray[np.float64]:
    mean = np.asarray(receipt["feature_mean"], dtype=np.float64)
    scale = np.asarray(receipt["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(receipt["coefficients"], dtype=np.float64)
    if predictors.ndim != 2 or predictors.shape[1] != len(mean):
        raise ValueError("H022 stacker predictor width changed")
    return np.asarray(
        float(receipt["intercept"])
        + ((predictors - mean) / scale) @ coefficients,
        dtype=np.float64,
    )


def economic_policy_metrics(
    target_utility: NDArray[np.float64],
    predicted_utility: NDArray[np.float64],
    weights: NDArray[np.float64],
    component_ids: NDArray[np.int64],
    week_ids: NDArray[np.int64],
    behavior_codes: NDArray[np.uint8],
    entry_prices: NDArray[np.float64],
    *,
    source_rows: int,
) -> dict[str, Any]:
    """Measure a natural zero-utility KEEP/SKIP decision without a price cutoff."""

    rows = len(target_utility)
    if any(
        len(values) != rows
        for values in (
            predicted_utility,
            weights,
            component_ids,
            week_ids,
            behavior_codes,
            entry_prices,
        )
    ) or source_rows <= 0:
        raise ValueError("H022 metric arrays are not aligned")
    keep = predicted_utility > 0.0
    profitable = target_utility > 0.0
    chosen = target_utility * keep
    weighted_chosen = chosen * weights

    def rate(mask: NDArray[np.bool_], denominator: NDArray[np.bool_]) -> float:
        count = int(denominator.sum())
        return float(mask.sum() / count) if count else 0.0

    group_keys = np.stack((behavior_codes.astype(np.int64), component_ids), axis=1)
    _, component_inverse = np.unique(group_keys, axis=0, return_inverse=True)
    component_utility = np.bincount(component_inverse, weights=weighted_chosen)
    week_keys = np.stack((behavior_codes.astype(np.int64), week_ids), axis=1)
    _, week_inverse = np.unique(week_keys, axis=0, return_inverse=True)
    week_utility = np.bincount(week_inverse, weights=weighted_chosen)
    tail_components = max(1, math.ceil(0.1 * len(component_utility)))
    tail_weeks = max(1, math.ceil(0.1 * len(week_utility)))
    component_tail = np.partition(component_utility, tail_components - 1)[
        :tail_components
    ]
    week_tail = np.partition(week_utility, tail_weeks - 1)[:tail_weeks]
    high = entry_prices >= 0.8
    return {
        "calls": int(keep.sum()),
        "call_rate_over_source_rows": float(keep.sum() / source_rows),
        "candidate_keep_rate": float(keep.mean()),
        "equal_market_mean_protocol_exact_chosen_utility": float(
            weighted_chosen.sum() / source_rows
        ),
        "weighted_candidate_mean_chosen_utility": float(
            weighted_chosen.sum() / weights.sum()
        ),
        "profitable_candidate_retention_rate": rate(keep & profitable, profitable),
        "harmful_candidate_veto_rate": rate(~keep & ~profitable, ~profitable),
        "kept_call_precision": rate(keep & profitable, keep),
        "kept_independent_components": int(np.unique(component_ids[keep]).size),
        "candidate_component_groups": len(component_utility),
        "candidate_component_lower_tail_utility": float(component_tail.mean()),
        "candidate_week_groups": len(week_utility),
        "candidate_week_lower_tail_utility": float(week_tail.mean()),
        "predicted_utility_mean": float(predicted_utility.mean()),
        "predicted_utility_mae": float(
            np.average(np.abs(predicted_utility - target_utility), weights=weights)
        ),
        "high_price_candidates": int(high.sum()),
        "high_price_calls_kept": int((high & keep).sum()),
        "high_price_weighted_chosen_utility": float(
            (weighted_chosen * high).sum()
        ),
        "below_0_80_weighted_chosen_utility": float(
            (weighted_chosen * ~high).sum()
        ),
        "mean_kept_entry_price": float(entry_prices[keep].mean())
        if bool(keep.any())
        else 0.0,
    }


def calibration_metrics(
    probability0: NDArray[np.float64],
    target_outcome0: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> dict[str, float]:
    probability = np.clip(probability0, 1e-8, 1.0 - 1e-8)
    target = target_outcome0
    log_loss = -np.average(
        target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability),
        weights=weights,
    )
    brier = np.average(np.square(probability - target), weights=weights)
    return {"log_loss": float(log_loss), "brier": float(brier)}
