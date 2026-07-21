from __future__ import annotations

import torch

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH
from sphinx_trace.model_h022 import SphinxTraceS0H022NeuralMember
from sphinx_trace.model_h023 import (
    H023_AUX_FEATURE_WIDTH,
    H023_GROUP_IDS,
    SphinxTraceS0H023NeuralMember,
    load_h022_initialization,
)


def _config() -> dict[str, float | int]:
    return {
        "hidden_width": 32,
        "layers": 1,
        "attention_heads": 4,
        "attention_ffn_width": 64,
        "dropout": 0.0,
        "head_hidden_width": 16,
        "quantile_delta_scale": 0.01,
    }


def test_h023_loads_h022_and_emits_realized_gate_outputs() -> None:
    h022 = SphinxTraceS0H022NeuralMember(_config())
    model = SphinxTraceS0H023NeuralMember(_config())
    missing = load_h022_initialization(model, h022.state_dict())

    assert len(missing) == 27
    output = model(
        torch.zeros(3, 512),
        torch.zeros(3, H022_TREE_FEATURE_WIDTH),
        torch.zeros(3),
        torch.tensor([0, 1, 0]),
        torch.full((3,), 0.5),
        torch.zeros(3, H023_AUX_FEATURE_WIDTH),
        return_debug=True,
    )
    assert output["realized_net_contribution_mean"].shape == (3,)
    assert output["conditional_realized_return_quantiles"].shape == (3, 3)
    assert output["fill_probability"].shape == (3,)
    assert output["probability_realized_contribution_positive"].shape == (3,)
    assert output["keep_base_call_logit"].shape == (3,)
    assert output["debug_group_attention"].shape == (3, len(H023_GROUP_IDS))
    assert torch.isfinite(output["conditional_realized_return_quantiles"]).all()
