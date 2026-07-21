from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH
from sphinx_trace.h022_runtime import H022EnsembleRuntime, h022_debug_payload
from sphinx_trace.model_h022 import SphinxTraceS0H022NeuralMember


class Tree:
    def predict(
        self, values: np.ndarray[Any, Any], *, pred_contrib: bool = False
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if pred_contrib:
            return np.zeros((len(values), H022_TREE_FEATURE_WIDTH + 1), dtype=np.float64)
        return np.zeros(len(values), dtype=np.float64)


def _runtime(intercept: float) -> H022EnsembleRuntime:
    neural = SphinxTraceS0H022NeuralMember(
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
    }
    stacker = {
        "feature_mean": [0.0] * 7,
        "feature_scale": [1.0] * 7,
        "coefficients": [0.0] * 7,
        "intercept": intercept,
    }
    return H022EnsembleRuntime(
        neural,
        Tree(),
        statistics,
        stacker,
        torch.device("cpu"),
        policy_sha256="ab" * 32,
    )


def test_h022_runtime_emits_finite_explainable_decision() -> None:
    decision = _runtime(0.01).score(
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
    )

    assert decision.keep_base_call is True
    assert decision.ensemble_net_return == 0.01
    assert len(decision.neural_group_attention) == 7
    assert len(decision.tree_group_contributions) == 7
    assert np.isfinite(decision.neural_return_quantiles).all()
    payload = h022_debug_payload(decision)
    assert payload["keep_base_call"] is True
    assert len(payload["attribution"]["group_ids"]) == 7


def test_h022_runtime_compares_learned_score_to_skip_utility() -> None:
    decision = _runtime(-0.01).score(
        np.zeros(512, dtype=np.float32),
        np.zeros(128, dtype=np.float32),
        0.0,
        0.0,
        (0.0,) * 9,
        (0.0,) * 7,
        (1.0, 0.0, -1.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 1.9, 1.9, 0.5, 0.5),
        0,
    )
    assert decision.keep_base_call is False
