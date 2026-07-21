from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH
from sphinx_trace.h022_runtime import H022DecisionDebug
from sphinx_trace.h023_runtime import (
    H023_PREDICTOR_NAMES,
    H023DecisionDebug,
    H023EnsembleRuntime,
    h023_debug_payload,
)
from sphinx_trace.model_h023 import SphinxTraceS0H023NeuralMember


class Tree:
    def predict(
        self, values: np.ndarray[Any, Any], *, pred_contrib: bool = False
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if pred_contrib:
            output = np.zeros(
                (len(values), H022_TREE_FEATURE_WIDTH + 1), dtype=np.float64
            )
            output[:, 130] = 0.01
            return output
        return np.zeros(len(values), dtype=np.float64)


def _h022() -> H022DecisionDebug:
    return H022DecisionDebug(
        keep_base_call=False,
        candidate_action_id=0,
        neural_mean_net_return=-0.001,
        neural_return_quantiles=(-0.01, -0.001, 0.01),
        neural_fill_probability=0.5,
        neural_calibrated_probability0=0.7,
        neural_calibrated_candidate_edge=0.05,
        tree_net_return=-0.002,
        ensemble_net_return=-0.0015,
        stacker_intercept=0.0,
        stacker_contributions=(0.0,) * 7,
        neural_group_attention=(1.0 / 7.0,) * 7,
        tree_group_contributions=(0.0,) * 7,
        tree_price_context_contribution=0.0,
        tree_wallet_contribution=0.0,
        tree_event_contribution=0.0,
    )


def _runtime(intercept: float) -> H023EnsembleRuntime:
    neural = SphinxTraceS0H023NeuralMember(
        {
            "hidden_width": 32,
            "layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
            "head_hidden_width": 16,
        }
    )
    statistics = {
        "latent_mean": np.zeros(512, dtype=np.float32),
        "latent_scale": np.ones(512, dtype=np.float32),
        "tree_mean": np.zeros(H022_TREE_FEATURE_WIDTH, dtype=np.float32),
        "tree_scale": np.ones(H022_TREE_FEATURE_WIDTH, dtype=np.float32),
        "aux_mean": np.zeros(11, dtype=np.float32),
        "aux_scale": np.ones(11, dtype=np.float32),
    }
    stacker = {
        "predictor_names": list(H023_PREDICTOR_NAMES),
        "feature_mean": [0.0] * len(H023_PREDICTOR_NAMES),
        "feature_scale": [1.0] * len(H023_PREDICTOR_NAMES),
        "coefficients": [0.0] * len(H023_PREDICTOR_NAMES),
        "intercept": intercept,
    }
    return H023EnsembleRuntime(
        neural,
        Tree(),
        statistics,
        stacker,
        torch.device("cpu"),
        policy_sha256="ab" * 32,
    )


def _score(runtime: H023EnsembleRuntime) -> H023DecisionDebug:
    return runtime.score(
        np.zeros(512, dtype=np.float32),
        np.zeros(128, dtype=np.float32),
        1.0,
        -0.5,
        (1.0,) * 9,
        (0.0,) * 7,
        (2.0, 0.0, -1.0),
        (0.01, -0.01, 0.0),
        (0.61, 0.41, 1.6, 2.4, 0.6, 0.4),
        0,
        0.05,
        _h022(),
    )


def test_h023_runtime_emits_full_model_driven_debug() -> None:
    decision = _score(_runtime(0.01))
    payload = h023_debug_payload(decision)

    assert decision.keep_base_call is True
    assert decision.ensemble_realized_contribution == 0.01
    assert len(decision.top_tree_features) == 16
    assert decision.top_tree_features[0].contribution == 0.01
    assert len(payload["attribution"]["group_ids"]) == 7
    assert len(payload["attribution"]["top_tree_features"]) == 16
    assert payload["gate_reason"] == "positive_expected_realized_net_contribution"


def test_h023_runtime_uses_learned_zero_utility_indifference() -> None:
    decision = _score(_runtime(-0.01))

    assert decision.keep_base_call is False
    assert decision.gate_reason == "nonpositive_expected_realized_net_contribution"
