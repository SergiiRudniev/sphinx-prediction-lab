"""Conservative protocol-exact residual objective for Sphinx Trace H018."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _lower_tail(values: Tensor, quantile: float, weights: Tensor | None = None) -> Tensor:
    if not 0.0 < quantile <= 1.0 or values.ndim != 1 or not len(values):
        raise ValueError("H018 lower-tail inputs are invalid")
    count = max(1, math.ceil(len(values) * quantile))
    indices = torch.topk(values.detach(), count, largest=False, sorted=False).indices
    selected = values[indices]
    if weights is None:
        return selected.mean()
    selected_weights = weights[indices]
    return (selected * selected_weights).sum() / selected_weights.sum().clamp_min(1e-8)


def _component_utility_means(
    values: Tensor,
    component_ids: Tensor,
    weights: Tensor,
) -> Tensor:
    _, inverse = torch.unique(component_ids.long(), sorted=False, return_inverse=True)
    groups = int(inverse.max().item()) + 1
    weighted_sum = torch.zeros(groups, device=values.device, dtype=torch.float32)
    weight_sum = torch.zeros_like(weighted_sum)
    weighted_sum.scatter_add_(0, inverse, values * weights)
    weight_sum.scatter_add_(0, inverse, weights)
    return weighted_sum / weight_sum.clamp_min(1e-8)


def conservative_protocol_residual_loss(
    output: dict[str, Tensor],
    labels_outcome0: Tensor,
    winning_payout_multipliers: Tensor,
    reference_action_values: Tensor,
    behavior_action_ids: Tensor,
    realized_action_values: Tensor,
    component_ids: Tensor,
    config: dict[str, Any],
    *,
    sample_weights: Tensor | None = None,
    physical_action_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize exact utility while retaining the frozen H014 policy prior."""

    required = {
        "action_logits",
        "base_action_logits",
        "action_residual_logits",
        "protocol_action_values",
        "position_size_beta_alpha",
        "position_size_beta_beta",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"H018 model output is missing: {sorted(missing)}")
    logits = output["action_logits"][:, :3].float()
    base_logits = output["base_action_logits"][:, :3].float().detach()
    residual_logits = output["action_residual_logits"][:, :3].float()
    protocol_values = output["protocol_action_values"].float()
    rows = len(logits)
    if (
        logits.shape != (rows, 3)
        or base_logits.shape != (rows, 3)
        or residual_logits.shape != (rows, 3)
        or protocol_values.shape != (rows, 3)
        or labels_outcome0.shape != (rows,)
        or winning_payout_multipliers.shape != (rows, 2)
        or reference_action_values.shape != (rows, 3)
        or behavior_action_ids.shape != (rows,)
        or realized_action_values.shape != (rows,)
        or component_ids.shape != (rows,)
    ):
        raise ValueError("H018 residual training tensors do not align")
    if not bool(torch.isfinite(protocol_values).all()) or not bool(
        torch.isfinite(reference_action_values).all()
    ):
        raise ValueError("H018 protocol values must be finite")
    if not bool(torch.isfinite(winning_payout_multipliers).all()) or bool(
        (winning_payout_multipliers <= 0.0).any()
    ):
        raise ValueError("H018 payout multipliers must be finite and positive")

    inferred_mask = logits > torch.finfo(torch.float32).min / 2.0
    available = (
        inferred_mask
        if physical_action_mask is None
        else physical_action_mask[:, :3].bool()
    )
    if available.shape != logits.shape or not bool(available.any(dim=1).all()):
        raise ValueError("H018 physical action mask must cover every row")
    actions = behavior_action_ids.long()
    if bool(((actions < 0) | (actions >= 3)).any()) or not bool(
        available.gather(1, actions[:, None]).all()
    ):
        raise ValueError("H018 logged action is invalid")
    weights = (
        torch.ones(rows, device=logits.device, dtype=torch.float32)
        if sample_weights is None
        else sample_weights.float()
    )
    if weights.shape != (rows,) or bool((weights <= 0.0).any()):
        raise ValueError("H018 sample weights must be aligned and positive")
    weight_sum = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor) -> Tensor:
        return (values * weights).sum() / weight_sum

    temperature = float(config["policy_temperature"])
    if temperature <= 0.0:
        raise ValueError("H018 policy temperature must be positive")
    policy = torch.softmax(logits / temperature, dim=-1)
    base_policy = torch.softmax(base_logits / temperature, dim=-1)

    alpha = output["position_size_beta_alpha"].float()
    beta = output["position_size_beta_beta"].float()
    size = alpha / (alpha + beta).clamp_min(1e-8)
    label0 = labels_outcome0.float()
    wealth0 = 1.0 - size + size * label0 * winning_payout_multipliers[:, 0].float()
    wealth1 = (
        1.0
        - size
        + size * (1.0 - label0) * winning_payout_multipliers[:, 1].float()
    )
    exact_utilities = torch.stack(
        (
            torch.log(wealth0.clamp_min(1e-8)),
            torch.log(wealth1.clamp_min(1e-8)),
            torch.zeros_like(wealth0),
        ),
        dim=-1,
    )
    expected_utility = (policy * exact_utilities).sum(dim=-1)
    mean_utility = weighted_mean(expected_utility)
    row_tail = _lower_tail(
        expected_utility,
        float(config["row_lower_tail_quantile"]),
        weights,
    )
    component_means = _component_utility_means(
        expected_utility, component_ids, weights
    )
    component_tail = _lower_tail(
        component_means, float(config["component_lower_tail_quantile"])
    )

    safe_protocol_values = torch.where(
        available, protocol_values, reference_action_values.float()
    )
    value_elements = F.smooth_l1_loss(
        safe_protocol_values,
        reference_action_values.float().detach(),
        reduction="none",
    )
    value_rows = (
        (value_elements * available.float()).sum(dim=-1)
        / available.sum(dim=-1).clamp_min(1)
    )
    protocol_value_loss = weighted_mean(value_rows)
    logged_predictions = protocol_values.gather(1, actions[:, None]).squeeze(1)
    logged_rows = F.smooth_l1_loss(
        logged_predictions,
        realized_action_values.float().detach(),
        reduction="none",
    )
    logged_value_loss = weighted_mean(logged_rows)

    policy_log = torch.log(policy.clamp_min(1e-8))
    base_log = torch.log(base_policy.clamp_min(1e-8))
    policy_kl = weighted_mean((policy * (policy_log - base_log)).sum(dim=-1))
    residual_rows = (
        (residual_logits.square() * available.float()).sum(dim=-1)
        / available.sum(dim=-1).clamp_min(1)
    )
    residual_l2 = weighted_mean(residual_rows)

    utility_scale = float(config["protocol_utility_scale"])
    loss = (
        -utility_scale
        * (
            float(config["mean_protocol_utility_weight"]) * mean_utility
            + float(config["row_lower_tail_weight"]) * row_tail
            + float(config["component_lower_tail_weight"]) * component_tail
        )
        + float(config["protocol_action_value_weight"]) * protocol_value_loss
        + float(config["logged_execution_value_weight"]) * logged_value_loss
        + float(config["H014_policy_KL_weight"]) * policy_kl
        + float(config["residual_logit_L2_weight"]) * residual_l2
    )

    chosen = logits.argmax(dim=-1)
    chosen_utility = exact_utilities.gather(1, chosen[:, None]).squeeze(1)
    calls = chosen != 2
    correct = ((chosen == 0) & (label0 == 1.0)) | (
        (chosen == 1) & (label0 == 0.0)
    )
    return loss, {
        "protocol_exact_expected_utility": mean_utility.detach(),
        "protocol_exact_chosen_utility": weighted_mean(chosen_utility).detach(),
        "protocol_exact_row_tail_utility": row_tail.detach(),
        "protocol_exact_component_tail_utility": component_tail.detach(),
        "protocol_action_value_loss": protocol_value_loss.detach(),
        "logged_execution_value_loss": logged_value_loss.detach(),
        "H014_policy_KL": policy_kl.detach(),
        "residual_logit_L2": residual_l2.detach(),
        "mean_size": weighted_mean(size).detach(),
        "call_rate": calls.float().mean().detach(),
        "call_count": calls.sum().detach(),
        "correct_call_count": correct[calls].sum().detach(),
        "call_precision": (
            correct[calls].float().mean().detach()
            if bool(calls.any())
            else torch.zeros((), device=logits.device)
        ),
        "sample_weight_sum": weight_sum.detach(),
        "component_count": torch.tensor(
            len(component_means), device=logits.device, dtype=torch.int64
        ),
        "rows": torch.tensor(rows, device=logits.device, dtype=torch.int64),
    }
