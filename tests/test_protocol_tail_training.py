from __future__ import annotations

import torch

from sphinx_trace.protocol_tail_training import protocol_tail_utility_loss


def _output() -> dict[str, torch.Tensor]:
    return {
        "action_logits": torch.tensor(
            [[0.03, -0.02, 0.0], [-0.01, 0.04, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "position_size_beta_alpha": torch.tensor(
            [1.2, 1.2], dtype=torch.float32, requires_grad=True
        ),
        "position_size_beta_beta": torch.tensor(
            [20.0, 20.0], dtype=torch.float32, requires_grad=True
        ),
        "state_value": torch.zeros(2, dtype=torch.float32, requires_grad=True),
        "terminal_outcome_logit": torch.zeros(
            2, dtype=torch.float32, requires_grad=True
        ),
    }


def _config() -> dict[str, float]:
    return {
        "action_value_temperature": 0.05,
        "counterfactual_action_value_weight": 4.0,
        "logged_execution_action_value_weight": 2.0,
        "policy_exact_utility_weight": 0.2,
        "lower_tail_quantile": 0.5,
        "lower_tail_utility_weight": 0.25,
        "entropy_weight": 0.0001,
        "value_weight": 0.1,
        "outcome_auxiliary_weight": 0.0,
    }


def test_protocol_tail_loss_is_finite_and_backpropagates() -> None:
    output = _output()
    loss, metrics = protocol_tail_utility_loss(
        output,
        torch.tensor([1.0, 0.0]),
        torch.tensor([[1.8, 2.2], [1.6, 2.4]]),
        torch.tensor([[0.04, -0.05, 0.0], [-0.05, 0.06, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0.03, 0.04]),
        torch.tensor([1.0, 0.8]),
        _config(),
        physical_action_mask=torch.ones((2, 7), dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    assert metrics["protocol_exact_tail_utility"] <= metrics[
        "protocol_exact_expected_utility"
    ]
    loss.backward()  # type: ignore[no-untyped-call]
    assert output["action_logits"].grad is not None
    assert bool(torch.isfinite(output["action_logits"].grad).all())


def test_protocol_tail_loss_respects_unavailable_logged_actions() -> None:
    output = _output()
    mask = torch.ones((2, 7), dtype=torch.bool)
    mask[0, 0] = False

    try:
        protocol_tail_utility_loss(
            output,
            torch.tensor([1.0, 0.0]),
            torch.tensor([[1.8, 2.2], [1.6, 2.4]]),
            torch.zeros((2, 3)),
            torch.tensor([0, 1]),
            torch.zeros(2),
            torch.ones(2),
            _config(),
            physical_action_mask=mask,
        )
    except ValueError as error:
        assert "not physically available" in str(error)
    else:
        raise AssertionError("unavailable logged action was accepted")
