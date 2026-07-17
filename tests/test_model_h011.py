from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h011 import (
    SphinxTraceS0H011,
    h011_variant_feature_mask,
    h011_variant_group_mask,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("candidate", "target"),
    [("50m", 50_000_000), ("100m", 100_000_000), ("150m", 150_000_000)],
)
def test_h011_candidates_match_registered_capacity(candidate: str, target: int) -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    model = SphinxTraceS0H011(config, candidate_id=candidate)
    assert parameter_count(model) == pytest.approx(target, rel=0.15)


def test_h011_forward_exposes_heads_and_debug_attention() -> None:
    config = load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    model = SphinxTraceS0H011(config, candidate_id="50m").eval()
    features = torch.zeros((2, 128))
    with torch.inference_mode():
        output = model(features, return_debug=True)
    assert output["terminal_outcome_logit"].shape == (2,)
    assert output["position_size_beta_alpha"].min() > 1.0
    assert output["debug_attention"].shape == (2, 14, 14, 14)


def test_h011_ablation_masks_information_without_dropping_rows() -> None:
    market_only = h011_variant_feature_mask("h011_market_only")
    wallet_flow = h011_variant_feature_mask("h011_uncapped_wallet_flow")
    causal = h011_variant_feature_mask("h011_causal_wallet_performance")
    assert market_only.shape == wallet_flow.shape == causal.shape == (128,)
    assert market_only[:48].all()
    assert not market_only[48:].any()
    assert wallet_flow[72:85].all()
    assert not wallet_flow[85:99].any()
    assert causal[85:89].all()
    assert h011_variant_group_mask("h011_market_only").tolist() == [1, 1, 0, 0, 0, 0]
    with pytest.raises(RuntimeError, match="Polygon"):
        h011_variant_feature_mask("h011_temporal_graph")
