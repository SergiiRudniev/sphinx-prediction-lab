from __future__ import annotations

import pytest
import torch

from sphinx_trace.protocol_residual_training import (
    conservative_protocol_residual_loss,
)


def _config() -> dict[str, float]:
    return {
        "policy_temperature": 0.02,
        "protocol_utility_scale": 1000.0,
        "mean_protocol_utility_weight": 1.0,
        "row_lower_tail_quantile": 0.5,
        "row_lower_tail_weight": 0.15,
        "component_lower_tail_quantile": 0.5,
        "component_lower_tail_weight": 0.35,
        "protocol_action_value_weight": 0.5,
        "logged_execution_value_weight": 0.25,
        "H014_policy_KL_weight": 0.02,
        "residual_logit_L2_weight": 0.01,
    }


def _output() -> dict[str, torch.Tensor]:
    base = torch.tensor(
        [[0.02, -0.03, 0.0], [-0.02, 0.03, 0.0], [0.01, -0.01, 0.0], [0.0, 0.0, 0.01]]
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


def test_h018_loss_updates_residual_and_separate_value_head() -> None:
    output = _output()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    payouts = torch.tensor([[2.0, 2.0], [2.5, 1.5], [1.5, 3.0], [4.0, 1.3]])
    references = torch.tensor(
        [[0.04, -0.05, 0.0], [-0.05, 0.02, 0.0], [0.01, -0.05, 0.0], [-0.05, 0.005, 0.0]]
    )
    loss, metrics = conservative_protocol_residual_loss(
        output,
        labels,
        payouts,
        references,
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([0.02, 0.01, 0.0, -0.01]),
        torch.tensor([11, 11, 12, 13]),
        _config(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert output["action_residual_logits"].grad is not None
    assert output["protocol_action_values"].grad is not None
    assert bool(torch.isfinite(output["action_residual_logits"].grad).all())
    assert bool(torch.isfinite(output["protocol_action_values"].grad).all())
    assert metrics["component_count"] == 3
    assert metrics["H014_policy_KL"] == pytest.approx(0.0, abs=1e-8)


def test_h018_loss_rejects_missing_separate_value_head() -> None:
    output = _output()
    del output["protocol_action_values"]
    with pytest.raises(ValueError, match="missing"):
        conservative_protocol_residual_loss(
            output,
            torch.ones(4),
            torch.ones(4, 2),
            torch.zeros(4, 3),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4),
            torch.arange(4),
            _config(),
        )
