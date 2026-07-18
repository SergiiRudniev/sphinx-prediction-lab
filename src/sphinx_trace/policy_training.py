"""Causal utility warm-start primitives for the H012 selective policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ComponentTimePartition:
    fit_components: NDArray[np.int64]
    selection_components: NDArray[np.int64]
    cutoff_unix: int


def component_time_partition(
    component_ids: NDArray[np.int64],
    timestamps: NDArray[np.int64],
    fit_fraction: float,
) -> ComponentTimePartition:
    """Assign whole components to chronological fit/selection blocks."""

    if component_ids.shape != timestamps.shape or component_ids.ndim != 1:
        raise ValueError("H012 component partition inputs must be aligned vectors")
    if not len(component_ids) or not 0.0 < fit_fraction < 1.0:
        raise ValueError("H012 component partition settings are invalid")
    unique, inverse = np.unique(component_ids, return_inverse=True)
    anchors = np.full(len(unique), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(anchors, inverse, timestamps)
    order = np.lexsort((unique, anchors))
    fit_count = min(len(unique) - 1, max(1, int(len(unique) * fit_fraction)))
    fit = np.sort(unique[order[:fit_count]])
    selection = np.sort(unique[order[fit_count:]])
    return ComponentTimePartition(
        fit_components=fit,
        selection_components=selection,
        cutoff_unix=int(anchors[order[fit_count - 1]]),
    )


def selective_log_utility_loss(
    output: dict[str, Tensor],
    labels_outcome0: Tensor,
    market_probability_outcome0: Tensor,
    config: dict[str, Any],
    *,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize learned CALL-0/CALL-1/SKIP and sizing by realized log utility."""

    logits = output["action_logits"][:, :3]
    if logits.shape != (len(labels_outcome0), 3):
        raise ValueError("H012 warm-start requires three initial action logits")
    if labels_outcome0.shape != market_probability_outcome0.shape:
        raise ValueError("H012 utility labels and market probabilities must align")
    loss_mode = str(config.get("loss_mode", "expected_utility"))
    temperature = float(config.get("action_value_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("H012 action-value temperature must be positive")
    policy = torch.softmax(logits.float() / temperature, dim=-1)
    alpha = output["position_size_beta_alpha"].float()
    beta = output["position_size_beta_beta"].float()
    size = alpha / (alpha + beta).clamp_min(1e-8)
    execution = config["execution_proxy"]
    tick = float(execution["adverse_price_ticks"]) * float(execution["tick_size"])
    fee_multiplier = 1.0 + float(execution["fee_bps"]) / 10_000.0
    minimum_price = float(execution["minimum_entry_price"])
    price0 = (market_probability_outcome0.float() + tick).clamp(minimum_price, 1.0)
    price1 = (1.0 - market_probability_outcome0.float() + tick).clamp(minimum_price, 1.0)
    label0 = labels_outcome0.float()
    wealth0 = 1.0 - size + size * label0 / (price0 * fee_multiplier)
    wealth1 = 1.0 - size + size * (1.0 - label0) / (price1 * fee_multiplier)
    log_utility = torch.stack(
        (
            torch.log(wealth0.clamp_min(1e-8)),
            torch.log(wealth1.clamp_min(1e-8)),
            torch.zeros_like(wealth0),
        ),
        dim=-1,
    )
    reference_size = float(config.get("action_value_reference_size", 0.05))
    if not 0.0 < reference_size < 1.0:
        raise ValueError("H012 action-value reference size must be between zero and one")
    reference = torch.full_like(size, reference_size)
    reference_wealth0 = 1.0 - reference + reference * label0 / (price0 * fee_multiplier)
    reference_wealth1 = 1.0 - reference + reference * (1.0 - label0) / (price1 * fee_multiplier)
    reference_log_utility = torch.stack(
        (
            torch.log(reference_wealth0.clamp_min(1e-8)),
            torch.log(reference_wealth1.clamp_min(1e-8)),
            torch.zeros_like(reference_wealth0),
        ),
        dim=-1,
    )
    expected_utility = (policy * log_utility).sum(dim=-1)
    entropy = -(policy * torch.log(policy.clamp_min(1e-8))).sum(dim=-1)
    weights = (
        torch.ones_like(expected_utility) if sample_weights is None else sample_weights.float()
    )
    if weights.shape != expected_utility.shape or bool((weights <= 0).any()):
        raise ValueError("H012 utility sample weights must be aligned and positive")
    weight_sum = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor) -> Tensor:
        return (values * weights).sum() / weight_sum

    value_target = expected_utility.detach()
    value_loss = weighted_mean((output["state_value"].float() - value_target).square())
    outcome_rows = F.binary_cross_entropy_with_logits(
        output["terminal_outcome_logit"].float(),
        label0,
        reduction="none",
    )
    outcome_loss = weighted_mean(outcome_rows)
    action_value_rows = F.smooth_l1_loss(
        logits.float(),
        reference_log_utility.detach(),
        reduction="none",
    ).mean(dim=-1)
    action_value_loss = weighted_mean(action_value_rows)
    if loss_mode == "expected_utility":
        action_value_weight = 0.0
        policy_utility_weight = 1.0
    elif loss_mode == "counterfactual_action_value":
        action_value_weight = float(config["action_value_weight"])
        policy_utility_weight = float(config["policy_utility_weight"])
        if action_value_weight <= 0.0 or policy_utility_weight < 0.0:
            raise ValueError("H012 counterfactual loss weights are invalid")
    else:
        raise ValueError(f"Unknown H012 utility loss mode: {loss_mode}")
    loss = (
        action_value_weight * action_value_loss
        - policy_utility_weight * weighted_mean(expected_utility)
        - float(config["entropy_weight"]) * weighted_mean(entropy)
        + float(config["value_weight"]) * value_loss
        + float(config["outcome_auxiliary_weight"]) * outcome_loss
    )
    chosen = policy.argmax(dim=-1)
    chosen_utility = log_utility.gather(1, chosen[:, None]).squeeze(1)
    calls = chosen != 2
    correct = ((chosen == 0) & (label0 == 1)) | ((chosen == 1) & (label0 == 0))
    return loss, {
        "expected_log_utility": weighted_mean(expected_utility).detach(),
        "chosen_log_utility": chosen_utility.mean().detach(),
        "chosen_log_utility_sum": chosen_utility.sum().detach(),
        "weighted_chosen_log_utility": weighted_mean(chosen_utility).detach(),
        "weighted_chosen_log_utility_sum": (chosen_utility * weights).sum().detach(),
        "sample_weight_sum": weight_sum.detach(),
        "entropy": weighted_mean(entropy).detach(),
        "value_loss": value_loss.detach(),
        "outcome_loss": outcome_loss.detach(),
        "action_value_loss": action_value_loss.detach(),
        "mean_call_probability": policy[:, :2].sum(dim=-1).mean().detach(),
        "positive_call_value_fraction": (logits[:, :2].max(dim=-1).values > 0.0)
        .float()
        .mean()
        .detach(),
        "mean_selected_action_value": logits.gather(1, chosen[:, None]).mean().detach(),
        "mean_size": size.mean().detach(),
        "call_rate": calls.float().mean().detach(),
        "call_count": calls.sum().detach(),
        "correct_call_count": correct[calls].sum().detach(),
        "rows": torch.tensor(len(logits), device=logits.device),
        "call_precision": (
            correct[calls].float().mean().detach()
            if bool(calls.any())
            else torch.zeros((), device=logits.device)
        ),
    }


def logged_execution_action_value_loss(
    output: dict[str, Tensor],
    behavior_action_ids: Tensor,
    realized_action_values: Tensor,
    execution_fractions: Tensor,
    *,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Regress only the logged action to its fill- and resolution-aware value."""

    logits = output["action_logits"][:, :3].float()
    rows = len(logits)
    if logits.shape != (rows, 3):
        raise ValueError("H015 logged execution loss requires three initial actions")
    if (
        behavior_action_ids.shape != (rows,)
        or realized_action_values.shape != (rows,)
        or execution_fractions.shape != (rows,)
    ):
        raise ValueError("H015 logged execution targets must align with policy rows")
    actions = behavior_action_ids.long()
    targets = realized_action_values.float()
    fractions = execution_fractions.float()
    if bool(((actions < 0) | (actions >= 3)).any()):
        raise ValueError("H015 logged behavior action is outside CALL-0/CALL-1/SKIP")
    if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(fractions).all()):
        raise ValueError("H015 logged execution targets must be finite")
    if bool(((fractions < 0.0) | (fractions > 1.0)).any()):
        raise ValueError("H015 execution fractions must be between zero and one")
    weights = torch.ones_like(targets) if sample_weights is None else sample_weights.float()
    if weights.shape != targets.shape or bool((weights <= 0.0).any()):
        raise ValueError("H015 logged execution sample weights must be aligned and positive")
    predictions = logits.gather(1, actions[:, None]).squeeze(1)
    loss_rows = F.smooth_l1_loss(predictions, targets.detach(), reduction="none")
    weight_sum = weights.sum().clamp_min(1e-8)
    loss = (loss_rows * weights).sum() / weight_sum
    absolute_error = (predictions - targets).abs()
    filled = fractions > 0.0
    calls = actions != 2
    return loss, {
        "logged_execution_action_value_loss": loss.detach(),
        "logged_execution_absolute_error": (
            (absolute_error * weights).sum() / weight_sum
        ).detach(),
        "logged_execution_prediction_mean": (
            (predictions * weights).sum() / weight_sum
        ).detach(),
        "logged_execution_target_mean": ((targets * weights).sum() / weight_sum).detach(),
        "logged_call_count": calls.sum().detach(),
        "logged_filled_count": filled.sum().detach(),
        "logged_filled_target_mean": (
            targets[filled].mean().detach()
            if bool(filled.any())
            else torch.zeros((), device=logits.device)
        ),
        "sample_weight_sum": weight_sum.detach(),
    }
