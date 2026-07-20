"""Economic KEEP/SKIP objective for the price-aware Sphinx Trace H021 gate."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def economic_call_veto_loss(
    output: dict[str, Tensor],
    labels_outcome0: Tensor,
    winning_payout_multipliers: Tensor,
    reference_action_values: Tensor,
    behavior_action_ids: Tensor,
    realized_action_values: Tensor,
    config: dict[str, Any],
    *,
    sample_weights: Tensor | None = None,
    physical_action_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train a strict gate from realized net utility rather than raw correctness."""

    required = {
        "action_logits",
        "base_action_logits",
        "strict_veto_logit",
        "strict_gate_logits",
        "protocol_action_values",
        "position_size_beta_alpha",
        "position_size_beta_beta",
        "calibrated_outcome_logit",
        "outcome_calibration_delta",
        "no_upside_veto",
        "execution_entry_prices",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"H021 model output is missing: {sorted(missing)}")
    logits = output["action_logits"][:, :3].float()
    base_logits = output["base_action_logits"][:, :3].float().detach()
    gate_logits = output["strict_gate_logits"].float()
    veto_logit = output["strict_veto_logit"].float()
    protocol_values = output["protocol_action_values"].float()
    calibrated_logit = output["calibrated_outcome_logit"].float()
    calibration_delta = output["outcome_calibration_delta"].float()
    no_upside = output["no_upside_veto"].bool()
    entry_prices = output["execution_entry_prices"].float()
    rows = len(logits)
    aligned = (
        logits.shape == (rows, 3)
        and base_logits.shape == (rows, 3)
        and gate_logits.shape == (rows, 2)
        and veto_logit.shape == (rows,)
        and protocol_values.shape == (rows, 3)
        and calibrated_logit.shape == (rows,)
        and calibration_delta.shape == (rows,)
        and no_upside.shape == (rows,)
        and entry_prices.shape == (rows, 2)
        and labels_outcome0.shape == (rows,)
        and winning_payout_multipliers.shape == (rows, 2)
        and reference_action_values.shape == (rows, 3)
        and behavior_action_ids.shape == (rows,)
        and realized_action_values.shape == (rows,)
    )
    if not aligned:
        raise ValueError("H021 economic-veto tensors do not align")
    if not bool(torch.isfinite(protocol_values).all()) or not bool(
        torch.isfinite(reference_action_values).all()
    ):
        raise ValueError("H021 protocol values must be finite")
    if not bool(torch.isfinite(winning_payout_multipliers).all()) or bool(
        (winning_payout_multipliers <= 0.0).any()
    ):
        raise ValueError("H021 payout multipliers must be finite and positive")
    available = (
        logits > torch.finfo(torch.float32).min / 2.0
        if physical_action_mask is None
        else physical_action_mask[:, :3].bool()
    )
    if available.shape != logits.shape or not bool(available.any(dim=1).all()):
        raise ValueError("H021 physical action mask must cover every row")
    actions = behavior_action_ids.long()
    if bool(((actions < 0) | (actions >= 3)).any()):
        raise ValueError("H021 logged action is invalid")
    weights = (
        torch.ones(rows, device=logits.device, dtype=torch.float32)
        if sample_weights is None
        else sample_weights.float()
    )
    if weights.shape != (rows,) or bool((weights <= 0.0).any()):
        raise ValueError("H021 sample weights must be aligned and positive")
    weight_sum = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor) -> Tensor:
        return (values * weights).sum() / weight_sum

    alpha = output["position_size_beta_alpha"].float()
    beta = output["position_size_beta_beta"].float()
    size = alpha / (alpha + beta).clamp_min(1e-8)
    label0 = labels_outcome0.float()
    wealth0 = 1.0 - size + size * label0 * winning_payout_multipliers[:, 0].float()
    wealth1 = (
        1.0 - size + size * (1.0 - label0) * winning_payout_multipliers[:, 1].float()
    )
    exact_utilities = torch.stack(
        (
            torch.log(wealth0.clamp_min(1e-8)),
            torch.log(wealth1.clamp_min(1e-8)),
            torch.zeros_like(wealth0),
        ),
        dim=1,
    )
    base_action = base_logits.argmax(dim=-1)
    base_calls = base_action != 2
    base_utility = exact_utilities.gather(1, base_action[:, None]).squeeze(1)
    profitable_base_calls = base_calls & (base_utility > 0.0) & ~no_upside
    harmful_base_calls = base_calls & ~profitable_base_calls
    gate_targets = harmful_base_calls.long()
    if bool(base_calls.any()):
        called_utility = base_utility[base_calls]
        economic = called_utility.abs()
        false_call_multiplier = float(config["false_call_utility_multiplier"])
        if false_call_multiplier < 1.0:
            raise ValueError("H021 false-call multiplier must be at least one")
        economic = torch.where(
            called_utility <= 0.0,
            economic * false_call_multiplier,
            economic,
        )
        curriculum = config["price_curriculum"]
        if bool(curriculum["enabled"]):
            pivot = float(curriculum["pivot_price"])
            softness = float(curriculum["softness"])
            if not 0.0 < pivot < 1.0 or softness <= 0.0:
                raise ValueError("H021 price curriculum shape is invalid")
            selected_prices = entry_prices.gather(
                1, base_action.clamp(max=1)[:, None]
            ).squeeze(1)[base_calls]
            cheapness = torch.sigmoid((pivot - selected_prices) / softness)
            win_bonus = 1.0 + float(curriculum["cheap_win_bonus"]) * cheapness
            loss_penalty = 1.0 + float(curriculum["expensive_loss_penalty"]) * (
                1.0 - cheapness
            )
            economic = economic * torch.where(
                called_utility > 0.0, win_bonus, loss_penalty
            )
        power = float(config["economic_utility_weight_power"])
        if power <= 0.0:
            raise ValueError("H021 economic utility weight power must be positive")
        economic = economic.clamp_min(1e-12).pow(power)
        minimum = float(config["minimum_economic_weight"])
        if minimum < 0.0:
            raise ValueError("H021 minimum economic weight cannot be negative")
        economic = (economic / economic.mean().clamp_min(1e-8)).clamp_min(minimum)
        called_weights = weights[base_calls] * economic
        temperature = float(config["policy_temperature"])
        if temperature <= 0.0:
            raise ValueError("H021 policy temperature must be positive")
        gate_rows = F.cross_entropy(
            gate_logits[base_calls] / temperature,
            gate_targets[base_calls],
            reduction="none",
        )
        veto_action_loss = (
            gate_rows * called_weights
        ).sum() / called_weights.sum().clamp_min(1e-8)
    else:
        veto_action_loss = veto_logit.sum() * 0.0

    calibration_rows = F.binary_cross_entropy_with_logits(
        calibrated_logit, label0, reduction="none"
    )
    outcome_calibration_loss = weighted_mean(calibration_rows)
    calibrated_probability = torch.sigmoid(calibrated_logit)
    outcome_brier_loss = weighted_mean((calibrated_probability - label0).square())
    calibration_l2 = weighted_mean(calibration_delta.square())
    safe_protocol_values = torch.where(
        available, protocol_values, reference_action_values.float()
    )
    value_elements = F.smooth_l1_loss(
        safe_protocol_values,
        reference_action_values.float().detach(),
        reduction="none",
    )
    value_rows = (value_elements * available.float()).sum(dim=1) / available.sum(
        dim=1
    ).clamp_min(1)
    protocol_value_loss = weighted_mean(value_rows)
    logged_predictions = protocol_values.gather(1, actions[:, None]).squeeze(1)
    logged_rows = F.smooth_l1_loss(
        logged_predictions,
        realized_action_values.float().detach(),
        reduction="none",
    )
    logged_value_loss = weighted_mean(logged_rows)
    gate_l2 = weighted_mean(veto_logit.square())
    loss = (
        float(config["veto_action_weight"]) * veto_action_loss
        + float(config["outcome_calibration_weight"]) * outcome_calibration_loss
        + float(config["outcome_brier_weight"]) * outcome_brier_loss
        + float(config["outcome_calibration_L2_weight"]) * calibration_l2
        + float(config["protocol_action_value_weight"]) * protocol_value_loss
        + float(config["logged_execution_value_weight"]) * logged_value_loss
        + float(config["gate_logit_L2_weight"]) * gate_l2
    )

    temperature = float(config["policy_temperature"])
    policy = torch.softmax(logits / temperature, dim=-1)
    chosen = logits.argmax(dim=-1)
    chosen_utility = exact_utilities.gather(1, chosen[:, None]).squeeze(1)
    expected_utility = (policy * exact_utilities).sum(dim=1)
    calls = chosen != 2
    correct = ((chosen == 0) & (label0 == 1.0)) | ((chosen == 1) & (label0 == 0.0))
    return loss, {
        "veto_action_loss": veto_action_loss.detach(),
        "outcome_calibration_loss": outcome_calibration_loss.detach(),
        "outcome_brier_loss": outcome_brier_loss.detach(),
        "outcome_calibration_L2": calibration_l2.detach(),
        "protocol_exact_expected_utility": weighted_mean(expected_utility).detach(),
        "protocol_exact_chosen_utility": weighted_mean(chosen_utility).detach(),
        "protocol_action_value_loss": protocol_value_loss.detach(),
        "logged_execution_value_loss": logged_value_loss.detach(),
        "residual_logit_L2": gate_l2.detach(),
        "base_call_count": base_calls.sum().detach(),
        "profitable_base_call_count": profitable_base_calls.sum().detach(),
        "harmful_base_call_count": harmful_base_calls.sum().detach(),
        "no_upside_base_call_count": no_upside.sum().detach(),
        "retained_profitable_base_call_count": (
            profitable_base_calls & (chosen == base_action)
        )
        .sum()
        .detach(),
        "vetoed_harmful_base_call_count": (harmful_base_calls & (chosen == 2))
        .sum()
        .detach(),
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
        "rows": torch.tensor(rows, device=logits.device, dtype=torch.int64),
    }
