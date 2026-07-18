"""Training objective for the H017 protocol-exact tail-utility policy."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from sphinx_trace.policy_training import logged_execution_action_value_loss


def protocol_tail_utility_loss(
    output: dict[str, Tensor],
    labels_outcome0: Tensor,
    winning_payout_multipliers: Tensor,
    reference_action_values: Tensor,
    behavior_action_ids: Tensor,
    realized_action_values: Tensor,
    execution_fractions: Tensor,
    config: dict[str, Any],
    *,
    sample_weights: Tensor | None = None,
    physical_action_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize exact terminal utility and the learned lower tail jointly."""

    logits = output["action_logits"][:, :3].float()
    rows = len(logits)
    if logits.shape != (rows, 3) or labels_outcome0.shape != (rows,):
        raise ValueError("H017 logits and labels must align")
    if winning_payout_multipliers.shape != (rows, 2):
        raise ValueError("H017 payout multipliers must have shape [rows, 2]")
    if reference_action_values.shape != (rows, 3):
        raise ValueError("H017 reference action values must have shape [rows, 3]")
    if not bool(torch.isfinite(winning_payout_multipliers).all()) or bool(
        (winning_payout_multipliers <= 0.0).any()
    ):
        raise ValueError("H017 payout multipliers must be finite and positive")
    if not bool(torch.isfinite(reference_action_values).all()):
        raise ValueError("H017 reference action values must be finite")
    inferred_mask = logits > torch.finfo(torch.float32).min / 2.0
    available = (
        inferred_mask
        if physical_action_mask is None
        else physical_action_mask[:, :3].bool()
    )
    if available.shape != logits.shape or not bool(available.any(dim=1).all()):
        raise ValueError("H017 physical action mask must cover every row")

    weights = (
        torch.ones(rows, device=logits.device, dtype=torch.float32)
        if sample_weights is None
        else sample_weights.float()
    )
    if weights.shape != (rows,) or bool((weights <= 0.0).any()):
        raise ValueError("H017 sample weights must be aligned and positive")
    weight_sum = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor, selected_weights: Tensor = weights) -> Tensor:
        return (values * selected_weights).sum() / selected_weights.sum().clamp_min(1e-8)

    temperature = float(config["action_value_temperature"])
    if temperature <= 0.0:
        raise ValueError("H017 action-value temperature must be positive")
    policy = torch.softmax(logits / temperature, dim=-1)
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
    entropy = -(policy * torch.log(policy.clamp_min(1e-8))).sum(dim=-1)

    safe_logits = torch.where(available, logits, reference_action_values.float())
    action_value_elements = F.smooth_l1_loss(
        safe_logits,
        reference_action_values.float().detach(),
        reduction="none",
    )
    action_value_rows = (
        (action_value_elements * available.float()).sum(dim=-1)
        / available.sum(dim=-1).clamp_min(1)
    )
    action_value_loss = weighted_mean(action_value_rows)
    logged_loss, logged_metrics = logged_execution_action_value_loss(
        output,
        behavior_action_ids,
        realized_action_values,
        execution_fractions,
        sample_weights=weights,
        physical_action_mask=physical_action_mask,
    )

    tail_quantile = float(config["lower_tail_quantile"])
    if not 0.0 < tail_quantile <= 1.0:
        raise ValueError("H017 lower-tail quantile must be inside (0, 1]")
    tail_count = max(1, math.ceil(rows * tail_quantile))
    tail_indices = torch.topk(
        expected_utility.detach(), tail_count, largest=False, sorted=False
    ).indices
    tail_utility = weighted_mean(
        expected_utility[tail_indices], weights[tail_indices]
    )

    value_target = expected_utility.detach()
    value_loss = weighted_mean(
        (output["state_value"].float() - value_target).square()
    )
    outcome_rows = F.binary_cross_entropy_with_logits(
        output["terminal_outcome_logit"].float(), label0, reduction="none"
    )
    outcome_loss = weighted_mean(outcome_rows)
    loss = (
        float(config["counterfactual_action_value_weight"]) * action_value_loss
        + float(config["logged_execution_action_value_weight"]) * logged_loss
        - float(config["policy_exact_utility_weight"])
        * weighted_mean(expected_utility)
        - float(config["lower_tail_utility_weight"]) * tail_utility
        - float(config["entropy_weight"]) * weighted_mean(entropy)
        + float(config["value_weight"]) * value_loss
        + float(config["outcome_auxiliary_weight"]) * outcome_loss
    )
    chosen = policy.argmax(dim=-1)
    chosen_utility = exact_utilities.gather(1, chosen[:, None]).squeeze(1)
    calls = chosen != 2
    correct = ((chosen == 0) & (label0 == 1.0)) | (
        (chosen == 1) & (label0 == 0.0)
    )
    metrics = {
        "protocol_exact_expected_utility": weighted_mean(expected_utility).detach(),
        "protocol_exact_chosen_utility": weighted_mean(chosen_utility).detach(),
        "protocol_exact_tail_utility": tail_utility.detach(),
        "protocol_action_value_loss": action_value_loss.detach(),
        "value_loss": value_loss.detach(),
        "outcome_loss": outcome_loss.detach(),
        "entropy": weighted_mean(entropy).detach(),
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
        "rows": torch.tensor(rows, device=logits.device),
        **logged_metrics,
    }
    return loss, metrics
