from __future__ import annotations

import pytest
import torch

from sphinx_trace.protocol_veto_training import learned_call_loss_veto_loss


def _config() -> dict[str, float]:
    return {
        "policy_temperature": 0.05,
        "veto_action_weight": 1.0,
        "economic_utility_weight_power": 0.5,
        "minimum_economic_weight": 0.1,
        "protocol_action_value_weight": 0.25,
        "logged_execution_value_weight": 0.1,
        "H014_policy_KL_weight": 0.1,
        "residual_logit_L2_weight": 0.01,
    }


def _output() -> dict[str, torch.Tensor]:
    base = torch.tensor(
        [[0.03, -0.02, 0.0], [-0.03, 0.02, 0.0], [-0.02, -0.01, 0.01], [0.02, -0.03, 0.0]]
    )
    residual = torch.zeros_like(base, requires_grad=True)
    values = torch.zeros_like(base, requires_grad=True)
    return {
        "base_action_logits": base,
        "action_residual_logits": residual,
        "action_logits": base + residual,
        "protocol_action_values": values,
        "position_size_beta_alpha": torch.full((4,), 2.0),
        "position_size_beta_beta": torch.full((4,), 98.0),
    }


def test_h019_targets_keep_correct_calls_and_veto_wrong_calls() -> None:
    output = _output()
    loss, metrics = learned_call_loss_veto_loss(
        output,
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.full((4, 2), 2.0),
        torch.tensor(
            [[0.02, -0.05, 0.0], [0.02, -0.05, 0.0], [-0.05, 0.02, 0.0], [-0.05, 0.02, 0.0]]
        ),
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0.01, -0.01, 0.0, -0.01]),
        _config(),
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert metrics["base_call_count"] == 3
    assert metrics["correct_base_call_count"] == 1
    assert metrics["wrong_base_call_count"] == 2
    assert output["action_residual_logits"].grad is not None
    assert output["protocol_action_values"].grad is not None
    assert bool(torch.isfinite(output["action_residual_logits"].grad).all())
    assert bool(torch.isfinite(output["protocol_action_values"].grad).all())


def test_h019_requires_separate_protocol_value_head() -> None:
    output = _output()
    del output["protocol_action_values"]
    with pytest.raises(ValueError, match="missing"):
        learned_call_loss_veto_loss(
            output,
            torch.ones(4),
            torch.ones(4, 2),
            torch.zeros(4, 3),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4),
            _config(),
        )


def test_h019_no_base_call_batch_has_finite_zero_veto_loss() -> None:
    output = _output()
    output["base_action_logits"] = torch.tensor(
        [[-0.02, -0.03, 0.01]] * 4
    )
    output["action_logits"] = (
        output["base_action_logits"] + output["action_residual_logits"]
    )
    loss, metrics = learned_call_loss_veto_loss(
        output,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.full((4, 2), 2.0),
        torch.zeros(4, 3),
        torch.full((4,), 2, dtype=torch.long),
        torch.zeros(4),
        _config(),
    )
    assert torch.isfinite(loss)
    assert metrics["veto_action_loss"] == pytest.approx(0.0)
