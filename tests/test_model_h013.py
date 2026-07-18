from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from sphinx_trace.config import load_json
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h013 import (
    SphinxTraceS0H013,
    h013_variant_feature_mask,
    h013_variant_group_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def _model() -> SphinxTraceS0H013:
    config = deepcopy(load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"))
    config["architecture"]["candidates"].append(
        {"id": "test", "width": 64, "heads": 4, "layers": 2, "ffn_width": 128}
    )
    return SphinxTraceS0H013(SphinxTraceS0H011(config, candidate_id="test")).eval()


def test_h013_initial_output_is_exactly_the_market_anchor() -> None:
    model = _model()
    market = torch.tensor([0.2, 0.8])
    with torch.inference_mode():
        output = model(torch.zeros((2, 128)), market, return_debug=True)
    assert torch.sigmoid(output["terminal_outcome_logit"]) == pytest.approx(market)
    assert output["terminal_outcome_residual_logit"] == pytest.approx(torch.zeros(2))
    assert output["debug_attention"].shape == (2, 2, 14, 14)


def test_h013_residual_and_variant_masks_are_learnable_and_matched() -> None:
    model = _model()
    residual_output = model.backbone.outcome.layers[-1]
    assert isinstance(residual_output, nn.Linear)
    with torch.no_grad():
        residual_output.bias.fill_(1.0)
    with torch.inference_mode():
        output = model(torch.zeros((1, 128)), torch.tensor([0.5]))
    assert output["terminal_outcome_logit"].item() == pytest.approx(1.0)
    assert h013_variant_feature_mask("h013_market_residual")[:48].all()
    assert h013_variant_group_mask("h013_market_residual").tolist() == [1, 1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="Unknown H013"):
        h013_variant_feature_mask("missing")
