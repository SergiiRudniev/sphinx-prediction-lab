from __future__ import annotations

from typing import Any

import pytest
import torch

from sphinx_trace.economic_veto_training import economic_call_veto_loss


def _config(*, curriculum: bool) -> dict[str, Any]:
    return {
        "policy_temperature": 0.05,
        "veto_action_weight": 1.0,
        "false_call_utility_multiplier": 1.5,
        "economic_utility_weight_power": 1.0,
        "minimum_economic_weight": 0.001,
        "outcome_calibration_weight": 0.2,
        "outcome_brier_weight": 0.2,
        "outcome_calibration_L2_weight": 0.01,
        "protocol_action_value_weight": 0.25,
        "logged_execution_value_weight": 0.1,
        "gate_logit_L2_weight": 0.01,
        "price_curriculum": {
            "enabled": curriculum,
            "pivot_price": 0.8,
            "softness": 0.04,
            "cheap_win_bonus": 0.5,
            "expensive_loss_penalty": 1.0,
        },
    }


def _output(
    entry_prices: torch.Tensor, no_upside: torch.Tensor | None = None
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = len(entry_prices)
    veto = torch.zeros(rows, requires_grad=True)
    zero = torch.zeros_like(veto)
    minimum = torch.full_like(veto, torch.finfo(torch.float32).min)
    calibration = torch.zeros(rows, requires_grad=True)
    protocol_values = torch.zeros((rows, 3), requires_grad=True)
    return (
        {
            "base_action_logits": torch.tensor([[1.0, 0.0, 0.0]]).repeat(rows, 1),
            "strict_veto_logit": veto,
            "strict_gate_logits": torch.stack((zero, veto), dim=1),
            "action_logits": torch.stack((zero, minimum, veto), dim=1),
            "protocol_action_values": protocol_values,
            "position_size_beta_alpha": torch.full((rows,), 2.0),
            "position_size_beta_beta": torch.full((rows,), 8.0),
            "calibrated_outcome_logit": calibration,
            "outcome_calibration_delta": calibration,
            "no_upside_veto": (
                torch.zeros(rows, dtype=torch.bool) if no_upside is None else no_upside
            ),
            "execution_entry_prices": entry_prices,
        },
        veto,
    )


def _loss_and_veto_gradient(*, curriculum: bool) -> tuple[torch.Tensor, torch.Tensor]:
    prices = torch.tensor([[0.40, 0.60], [0.95, 0.05], [0.40, 0.60], [0.95, 0.05]])
    output, veto = _output(prices)
    loss, _ = economic_call_veto_loss(
        output,
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.full((4, 2), 2.0),
        torch.zeros((4, 3)),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4),
        _config(curriculum=curriculum),
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert veto.grad is not None
    return loss.detach(), veto.grad.detach()


def test_h021_price_curriculum_reweights_cheap_wins_and_expensive_losses() -> None:
    _, baseline = _loss_and_veto_gradient(curriculum=False)
    loss, gradient = _loss_and_veto_gradient(curriculum=True)
    assert torch.isfinite(loss)
    assert abs(gradient[0]) > abs(gradient[1])
    assert abs(gradient[3]) > abs(gradient[2])
    assert gradient[0] > 0.0
    assert gradient[3] < 0.0
    assert not torch.allclose(gradient, baseline)


def test_h021_no_upside_call_is_a_harmful_skip_target() -> None:
    output, veto = _output(torch.tensor([[1.0, 0.5]]), no_upside=torch.tensor([True]))
    loss, metrics = economic_call_veto_loss(
        output,
        torch.ones(1),
        torch.tensor([[1.0, 2.0]]),
        torch.zeros((1, 3)),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1),
        _config(curriculum=True),
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(loss)
    assert metrics["harmful_base_call_count"] == pytest.approx(1)
    assert metrics["no_upside_base_call_count"] == pytest.approx(1)
    assert veto.grad is not None and veto.grad.item() < 0.0
