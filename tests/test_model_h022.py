from __future__ import annotations

import torch

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH
from sphinx_trace.model_h022 import SphinxTraceS0H022NeuralMember


def test_h022_neural_member_outputs_ordered_distribution_and_debug() -> None:
    model = SphinxTraceS0H022NeuralMember(
        {
            "hidden_width": 32,
            "layers": 2,
            "attention_heads": 4,
            "dropout": 0.0,
            "head_hidden_width": 16,
            "quantile_delta_scale": 0.01,
        }
    ).eval()
    rows = 4
    output = model(
        torch.randn(rows, 512),
        torch.randn(rows, H022_TREE_FEATURE_WIDTH),
        torch.randn(rows),
        torch.tensor([0, 1, 0, 1]),
        torch.full((rows,), 0.7),
        return_debug=True,
    )
    quantiles = output["net_return_quantiles"]
    assert quantiles.shape == (rows, 3)
    assert bool((quantiles[:, 0] <= quantiles[:, 1]).all())
    assert bool((quantiles[:, 1] <= quantiles[:, 2]).all())
    assert output["debug_group_attention"].shape == (rows, 7)
    assert bool(torch.isfinite(output["net_return_mean"]).all())
