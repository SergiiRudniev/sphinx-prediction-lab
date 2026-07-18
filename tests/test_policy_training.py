from __future__ import annotations

import numpy as np
import pytest
import torch

from sphinx_trace.policy_training import component_time_partition, selective_log_utility_loss


def _output(logits: torch.Tensor, size_alpha: float = 2.0, size_beta: float = 2.0):
    rows = len(logits)
    return {
        "action_logits": logits,
        "position_size_beta_alpha": torch.full((rows,), size_alpha),
        "position_size_beta_beta": torch.full((rows,), size_beta),
        "state_value": torch.zeros(rows),
        "terminal_outcome_logit": torch.zeros(rows),
    }


def _config() -> dict[str, object]:
    return {
        "entropy_weight": 0.0,
        "value_weight": 0.0,
        "outcome_auxiliary_weight": 0.0,
        "execution_proxy": {
            "adverse_price_ticks": 0,
            "tick_size": 0.01,
            "fee_bps": 0,
            "minimum_entry_price": 0.01,
        },
    }


def _action_value_config() -> dict[str, object]:
    return {
        **_config(),
        "loss_mode": "counterfactual_action_value",
        "action_value_reference_size": 0.05,
        "action_value_temperature": 0.05,
        "action_value_weight": 4.0,
        "policy_utility_weight": 0.1,
    }


def test_component_partition_never_splits_a_component() -> None:
    components = np.array([1, 1, 2, 3, 3, 4], dtype=np.int64)
    timestamps = np.array([1, 4, 2, 3, 8, 9], dtype=np.int64)

    partition = component_time_partition(components, timestamps, 0.5)

    assert set(partition.fit_components).isdisjoint(partition.selection_components)
    assert set(partition.fit_components) | set(partition.selection_components) == {1, 2, 3, 4}
    assert partition.cutoff_unix == 4


def test_selective_utility_rewards_correct_call_and_learned_skip() -> None:
    labels = torch.tensor([1.0, 0.0])
    markets = torch.tensor([0.5, 0.5])
    correct = _output(torch.tensor([[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]]))
    wrong = _output(torch.tensor([[-20.0, 20.0, -20.0], [20.0, -20.0, -20.0]]))
    skip = _output(torch.tensor([[-20.0, -20.0, 20.0], [-20.0, -20.0, 20.0]]))

    correct_loss, correct_metrics = selective_log_utility_loss(correct, labels, markets, _config())
    wrong_loss, _ = selective_log_utility_loss(wrong, labels, markets, _config())
    skip_loss, skip_metrics = selective_log_utility_loss(skip, labels, markets, _config())

    assert correct_loss < skip_loss < wrong_loss
    assert correct_metrics["call_precision"] == pytest.approx(1.0)
    assert skip_metrics["call_rate"] == pytest.approx(0.0)


def test_selective_utility_backpropagates_into_actions_and_size() -> None:
    logits = torch.zeros((2, 7), requires_grad=True)
    alpha = torch.full((2,), 2.0, requires_grad=True)
    output = _output(logits)
    output["position_size_beta_alpha"] = alpha
    loss, _ = selective_log_utility_loss(
        output,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.4, 0.6]),
        _config(),
    )

    loss.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()


def test_counterfactual_action_value_regression_teaches_both_sides_without_frequency_target() -> (
    None
):
    logits = torch.zeros((2, 7), requires_grad=True)
    output = _output(logits, size_alpha=1.1, size_beta=20.0)
    loss, metrics = selective_log_utility_loss(
        output,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.4, 0.6]),
        _action_value_config(),
    )

    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 0] > 0.0
    assert logits.grad[1, 1] < 0.0
    assert metrics["action_value_loss"] > 0.0
    assert metrics["mean_call_probability"] == pytest.approx(2.0 / 3.0)
