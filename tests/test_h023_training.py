from __future__ import annotations

import numpy as np
import torch
from scripts.train_h023_fill_realized_veto import (
    _ablation_features,
    _predictor_matrix,
    _sample_weights,
)

from sphinx_trace.h023_training import h023_neural_loss, realized_policy_metrics


def _config() -> dict[str, object]:
    return {
        "contribution_huber_beta": 0.001,
        "return_huber_beta": 0.05,
        "loss_weights": {
            "realized_contribution_mean": 2.0,
            "conditional_return_mean": 1.0,
            "conditional_return_quantiles": 1.0,
            "fill_bce": 0.25,
            "positive_contribution_bce": 0.5,
            "keep_bce": 0.5,
            "wrong_keep_pnl": 3.0,
            "profitable_skip_pnl": 1.5,
            "contribution_consistency": 0.25,
        },
    }


def test_h023_loss_is_finite_and_trains_all_realized_heads() -> None:
    keep_logits = torch.tensor([-1.0, 1.0, -0.5], requires_grad=True)
    contribution_mean = torch.zeros(3, requires_grad=True)
    return_mean = torch.zeros(3, requires_grad=True)
    return_quantiles = torch.zeros(3, 3, requires_grad=True)
    fill_logits = torch.zeros(3, requires_grad=True)
    positive_logits = torch.zeros(3, requires_grad=True)
    output = {
        "realized_net_contribution_mean": contribution_mean,
        "conditional_realized_return_mean": return_mean,
        "conditional_realized_return_quantiles": return_quantiles,
        "fill_logit": fill_logits,
        "fill_probability": torch.sigmoid(fill_logits),
        "positive_contribution_logit": positive_logits,
        "keep_base_call_logit": keep_logits,
    }
    loss, metrics = h023_neural_loss(
        output,
        torch.tensor([0.01, -0.02, 0.0]),
        torch.tensor([0.2, -0.4, 0.0]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.tensor([0.05, 0.05, 0.05]),
        torch.ones(3),
        _config(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["decision_regret"] > 0.0
    assert keep_logits.grad is not None
    assert contribution_mean.grad is not None
    assert return_mean.grad is not None
    assert fill_logits.grad is not None
    assert positive_logits.grad is not None


def test_h023_metrics_measure_dollar_pnl_not_outcome_accuracy() -> None:
    metrics = realized_policy_metrics(
        np.asarray([10.0, -8.0, 5.0], dtype=np.float64),
        np.asarray([0.1, -0.1, -0.1], dtype=np.float64),
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([10, 10, 11], dtype=np.int64),
        np.asarray([0.4, 0.95, 0.7], dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )

    assert metrics["realized_net_profit_usd"] == 10.0
    assert metrics["incremental_net_profit_vs_keep_all_usd"] == 3.0
    assert metrics["harmful_candidate_veto_rate"] == 1.0
    assert metrics["high_price_calls_kept"] == 0


def test_h023_training_helpers_preserve_model_driven_price_decision() -> None:
    data = {
        "requested_total_cost_usd": np.asarray([1.0, 100.0, 25.0]),
        "week_ids": np.asarray([1, 1, 2]),
        "target_realized_pnl_usd": np.asarray([2.0, -3.0, 1.0]),
    }
    config = {
        "weighting": {
            "maximum_cost_weight": 10.0,
            "negative_week_multiplier": 1.5,
        }
    }
    weights = _sample_weights(data, config)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] > weights[0]

    neural = {
        "contribution": np.zeros(3, dtype=np.float32),
        "return_mean": np.zeros(3, dtype=np.float32),
        "return_quantiles": np.zeros((3, 3), dtype=np.float32),
        "fill": np.zeros(3, dtype=np.float32),
        "positive": np.zeros(3, dtype=np.float32),
        "keep_logit": np.zeros(3, dtype=np.float32),
    }
    auxiliary = np.zeros((3, 11), dtype=np.float32)
    assert _predictor_matrix(neural, np.zeros(3), auxiliary).shape == (3, 12)

    features = np.ones((3, 170), dtype=np.float32)
    price_zero = _ablation_features(features, "price_execution_zero", seed=17)
    assert np.all(price_zero[:, :128] == 1.0)
    assert np.all(price_zero[:, 128:] == 0.0)
