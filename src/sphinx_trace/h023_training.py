"""Losses and economic diagnostics for H023 fill-realized gating."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


def h023_neural_loss(
    output: dict[str, Tensor],
    target_contribution: Tensor,
    target_conditional_return: Tensor,
    target_fill_fraction: Tensor,
    requested_cost_fraction: Tensor,
    sample_weights: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize actual contribution, conditional return, fill and KEEP regret."""

    rows = len(target_contribution)
    if any(
        value.shape != (rows,)
        for value in (
            target_conditional_return,
            target_fill_fraction,
            requested_cost_fraction,
            sample_weights,
        )
    ):
        raise ValueError("H023 neural targets are not aligned")
    weights = sample_weights.float()
    if bool((weights <= 0.0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("H023 sample weights must be finite and positive")

    def weighted_mean(values: Tensor, row_weights: Tensor = weights) -> Tensor:
        return (values * row_weights).sum() / row_weights.sum().clamp_min(1e-8)

    contribution = target_contribution.float()
    conditional_return = target_conditional_return.float()
    fill_fraction = target_fill_fraction.float().clamp(0.0, 1.0)
    filled_weights = weights * (fill_fraction > 0.0).float()
    loss_weights = config["loss_weights"]
    contribution_rows = F.smooth_l1_loss(
        output["realized_net_contribution_mean"].float(),
        contribution,
        reduction="none",
        beta=float(config["contribution_huber_beta"]),
    )
    return_rows = F.smooth_l1_loss(
        output["conditional_realized_return_mean"].float(),
        conditional_return,
        reduction="none",
        beta=float(config["return_huber_beta"]),
    )
    quantiles = output["conditional_realized_return_quantiles"].float()
    errors = conditional_return[:, None] - quantiles
    levels = torch.tensor([0.1, 0.5, 0.9], dtype=errors.dtype, device=errors.device)
    quantile_rows = torch.maximum(levels * errors, (levels - 1.0) * errors).mean(
        dim=1
    )
    fill_rows = F.binary_cross_entropy_with_logits(
        output["fill_logit"].float(), fill_fraction, reduction="none"
    )
    positive_rows = F.binary_cross_entropy_with_logits(
        output["positive_contribution_logit"].float(),
        (contribution > 0.0).float(),
        reduction="none",
    )
    keep_probability = torch.sigmoid(output["keep_base_call_logit"].float())
    regret_rows = (
        keep_probability
        * torch.relu(-contribution)
        * float(loss_weights["wrong_keep_pnl"])
        + (1.0 - keep_probability)
        * torch.relu(contribution)
        * float(loss_weights["profitable_skip_pnl"])
    )
    nonzero_scale = contribution.abs().mean().clamp_min(1e-8)
    keep_weights = weights * (0.1 + contribution.abs() / nonzero_scale)
    keep_rows = F.binary_cross_entropy_with_logits(
        output["keep_base_call_logit"].float(),
        (contribution > 0.0).float(),
        reduction="none",
    )
    implied_contribution = (
        output["fill_probability"].float()
        * requested_cost_fraction.float().clamp_min(0.0)
        * output["conditional_realized_return_mean"].float()
    )
    consistency_rows = F.smooth_l1_loss(
        implied_contribution,
        contribution,
        reduction="none",
        beta=float(config["contribution_huber_beta"]),
    )
    zero = torch.zeros((), dtype=weights.dtype, device=weights.device)
    return_loss = (
        weighted_mean(return_rows, filled_weights)
        if bool((filled_weights > 0.0).any())
        else zero
    )
    quantile_loss = (
        weighted_mean(quantile_rows, filled_weights)
        if bool((filled_weights > 0.0).any())
        else zero
    )
    positive_loss = (
        weighted_mean(positive_rows, filled_weights)
        if bool((filled_weights > 0.0).any())
        else zero
    )
    contribution_loss = weighted_mean(contribution_rows)
    fill_loss = weighted_mean(fill_rows)
    regret_loss = weighted_mean(regret_rows)
    keep_loss = weighted_mean(keep_rows, keep_weights)
    consistency_loss = weighted_mean(consistency_rows)
    loss = (
        float(loss_weights["realized_contribution_mean"]) * contribution_loss
        + float(loss_weights["conditional_return_mean"]) * return_loss
        + float(loss_weights["conditional_return_quantiles"]) * quantile_loss
        + float(loss_weights["fill_bce"]) * fill_loss
        + float(loss_weights["positive_contribution_bce"]) * positive_loss
        + float(loss_weights["keep_bce"]) * keep_loss
        + float(loss_weights["contribution_consistency"]) * consistency_loss
        + regret_loss
    )
    hard_keep = output["keep_base_call_logit"] > 0.0
    chosen = torch.where(hard_keep, contribution, torch.zeros_like(contribution))
    return loss, {
        "loss": loss.detach(),
        "realized_contribution_mean_loss": contribution_loss.detach(),
        "conditional_return_mean_loss": return_loss.detach(),
        "conditional_return_quantile_loss": quantile_loss.detach(),
        "fill_bce": fill_loss.detach(),
        "positive_contribution_bce": positive_loss.detach(),
        "keep_bce": keep_loss.detach(),
        "decision_regret": regret_loss.detach(),
        "contribution_consistency": consistency_loss.detach(),
        "weighted_chosen_contribution": weighted_mean(chosen).detach(),
        "calls": hard_keep.sum().detach(),
        "weight_sum": weights.sum().detach(),
    }


def realized_policy_metrics(
    target_pnl_usd: NDArray[np.float64],
    predicted_score: NDArray[np.float64],
    component_ids: NDArray[np.int64],
    week_ids: NDArray[np.int64],
    entry_prices: NDArray[np.float64],
    fill_fractions: NDArray[np.float64],
) -> dict[str, Any]:
    """Measure decision-level economic retention without a price threshold."""

    rows = len(target_pnl_usd)
    if any(
        len(values) != rows
        for values in (
            predicted_score,
            component_ids,
            week_ids,
            entry_prices,
            fill_fractions,
        )
    ):
        raise ValueError("H023 metric arrays are not aligned")
    keep = predicted_score > 0.0
    profitable = target_pnl_usd > 0.0
    harmful = target_pnl_usd < 0.0
    chosen = target_pnl_usd * keep
    high = entry_prices >= 0.8

    def rate(mask: NDArray[np.bool_], denominator: NDArray[np.bool_]) -> float:
        count = int(denominator.sum())
        return float(mask.sum() / count) if count else 0.0

    unique_components, component_inverse = np.unique(
        component_ids, return_inverse=True
    )
    component_profit = np.bincount(component_inverse, weights=chosen)
    unique_weeks, week_inverse = np.unique(week_ids, return_inverse=True)
    week_profit = np.bincount(week_inverse, weights=chosen)
    return {
        "candidate_rows": rows,
        "calls": int(keep.sum()),
        "candidate_keep_rate": float(keep.mean()) if rows else 0.0,
        "realized_net_profit_usd": float(chosen.sum()),
        "baseline_keep_all_net_profit_usd": float(target_pnl_usd.sum()),
        "incremental_net_profit_vs_keep_all_usd": float(
            chosen.sum() - target_pnl_usd.sum()
        ),
        "kept_call_precision": rate(keep & profitable, keep),
        "profitable_candidate_retention_rate": rate(keep & profitable, profitable),
        "harmful_candidate_veto_rate": rate(~keep & harmful, harmful),
        "kept_independent_components": int(np.unique(component_ids[keep]).size),
        "candidate_components": len(unique_components),
        "positive_component_fraction": float((component_profit > 0.0).mean()),
        "candidate_weeks": len(unique_weeks),
        "positive_week_fraction": float((week_profit > 0.0).mean()),
        "filled_candidates": int((fill_fractions > 0.0).sum()),
        "high_price_candidates": int(high.sum()),
        "high_price_calls_kept": int((high & keep).sum()),
        "high_price_realized_net_profit_usd": float(chosen[high].sum()),
        "below_0_80_realized_net_profit_usd": float(chosen[~high].sum()),
        "mean_kept_entry_price": float(entry_prices[keep].mean())
        if bool(keep.any())
        else 0.0,
    }
