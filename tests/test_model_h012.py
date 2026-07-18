from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sphinx_trace.config import load_json
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import H012_ACTION_COUNT, SphinxTraceS0H012
from sphinx_trace.model_h013 import SphinxTraceS0H013

ROOT = Path(__file__).resolve().parents[1]


def _models() -> SphinxTraceS0H012:
    model_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    )
    candidate = {
        "id": "test",
        "width": 64,
        "heads": 4,
        "layers": 2,
        "ffn_width": 128,
    }
    model_config["architecture"]["candidates"].append(candidate)
    policy_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json")
    )
    backbone = SphinxTraceS0H011(model_config, candidate_id="test")
    return SphinxTraceS0H012(backbone, policy_config).eval()


def test_h012_fuses_market_portfolio_and_prediction_memory() -> None:
    model = _models()
    physical = torch.ones((2, H012_ACTION_COUNT), dtype=torch.bool)
    physical[0, 0] = False
    with torch.inference_mode():
        output = model(
            torch.zeros((2, 128)),
            torch.zeros((2, 9)),
            torch.zeros((2, 7)),
            torch.tensor([2, 4]),
            physical_action_mask=physical,
            return_debug=True,
        )
    assert output["action_logits"].shape == (2, H012_ACTION_COUNT)
    assert output["action_logits"][0, 0] == torch.finfo(torch.float32).min
    assert output["position_size_beta_alpha"].min() > 1.0
    assert output["position_size_beta_beta"].min() > 1.0
    assert output["debug_portfolio_token"].shape == (2, 64)
    assert output["debug_prediction_memory_token"].shape == (2, 64)
    assert output["debug_policy_attention"].shape == (2, 4, 7, 7)


def test_h012_action_value_head_starts_at_safe_skip_anchor() -> None:
    model = _models()

    assert torch.count_nonzero(model.action.weight) == 0
    assert model.action.bias.detach().tolist() == pytest.approx(
        [-0.0001, -0.0001, 0.0, -1.0, -1.0, -1.0, -1.0]
    )


def test_h012_cached_market_encoding_matches_full_forward() -> None:
    model = _models()
    market = torch.randn((3, 128))
    portfolio = torch.randn((3, 9))
    memory = torch.randn((3, 7))
    previous = torch.tensor([0, 2, 6])
    physical = torch.ones((3, H012_ACTION_COUNT), dtype=torch.bool)
    with torch.inference_mode():
        backbone = model.outcome_backbone(market, return_latent=True)
        full = model(
            market,
            portfolio,
            memory,
            previous,
            physical_action_mask=physical,
        )
        cached = model.forward_from_market_encoding(
            backbone["debug_latent_state"],
            backbone["terminal_outcome_logit"],
            backbone["uncertainty_log_scale"],
            portfolio,
            memory,
            previous,
            physical_action_mask=physical,
        )

    assert set(cached) == set(full)
    for key in full:
        torch.testing.assert_close(cached[key], full[key], rtol=0.0, atol=0.0)


def test_h012_rejects_nonphysical_or_unknown_action_state() -> None:
    model = _models()
    with pytest.raises(ValueError, match="permit at least one"):
        model(
            torch.zeros((1, 128)),
            torch.zeros((1, 9)),
            torch.zeros((1, 7)),
            torch.tensor([2]),
            physical_action_mask=torch.zeros((1, H012_ACTION_COUNT), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="unknown action"):
        model(
            torch.zeros((1, 128)),
            torch.zeros((1, 9)),
            torch.zeros((1, 7)),
            torch.tensor([H012_ACTION_COUNT]),
        )


def test_h012_preserves_market_anchor_for_residual_backbone() -> None:
    direct = _models()
    assert isinstance(direct.outcome_backbone, SphinxTraceS0H011)
    residual = SphinxTraceS0H013(direct.outcome_backbone)
    model = SphinxTraceS0H012(
        residual,
        deepcopy(
            load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json")
        ),
    ).eval()
    market_probability = torch.tensor([0.73])
    with torch.inference_mode():
        output = model(
            torch.zeros((1, 128)),
            torch.zeros((1, 9)),
            torch.zeros((1, 7)),
            torch.tensor([2]),
            market_probability=market_probability,
        )
    assert torch.sigmoid(output["terminal_outcome_logit"]) == pytest.approx(market_probability)
    with pytest.raises(ValueError, match="market_probability"):
        model(
            torch.zeros((1, 128)),
            torch.zeros((1, 9)),
            torch.zeros((1, 7)),
            torch.tensor([2]),
        )
