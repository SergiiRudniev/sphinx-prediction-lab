from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sphinx_trace.config import load_json
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import H012_ACTION_COUNT
from sphinx_trace.model_h021 import SphinxTraceS0H021

ROOT = Path(__file__).resolve().parents[1]


def _model() -> SphinxTraceS0H021:
    model_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    )
    candidate = {
        "id": "test-h021",
        "width": 64,
        "heads": 4,
        "layers": 2,
        "ffn_width": 128,
    }
    model_config["architecture"]["candidates"].append(candidate)
    policy_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h021_policy_v1.json")
    )
    policy_config["architecture"]["outcome_calibration_head"]["hidden_width"] = 16
    policy_config["architecture"]["strict_veto_head"]["hidden_width"] = 16
    policy_config["architecture"]["protocol_action_value_head"]["hidden_width"] = 16
    backbone = SphinxTraceS0H011(model_config, candidate_id="test-h021")
    return SphinxTraceS0H021(backbone, policy_config).eval()


def _inputs(model: SphinxTraceS0H021, rows: int = 2) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((rows, model.width)),
        torch.zeros(rows),
        torch.zeros(rows),
        torch.zeros((rows, 9)),
        torch.zeros((rows, 7)),
        torch.full((rows,), 2, dtype=torch.long),
    )


def test_h021_zero_gate_keeps_base_call_but_vetoes_no_upside() -> None:
    model = _model()
    with torch.no_grad():
        model.action.bias[:3] = torch.tensor([1.0, 0.0, 0.0])
    execution = torch.tensor(
        [
            [1.0, 0.4, 1.0, 2.4, 0.99, 0.01],
            [0.6, 0.4, 1.6, 2.4, 0.6, 0.4],
        ]
    )
    with torch.inference_mode():
        output = model.forward_from_market_encoding(
            *_inputs(model),
            execution_context=execution,
            physical_action_mask=torch.ones((2, H012_ACTION_COUNT), dtype=torch.bool),
            return_debug=True,
        )
    assert output["base_action_id"].tolist() == [0, 0]
    assert output["action_logits"][:, :3].argmax(dim=-1).tolist() == [2, 0]
    assert output["no_upside_veto"].tolist() == [True, False]
    assert torch.count_nonzero(output["outcome_calibration_delta"]) == 0


def test_h021_cannot_create_or_flip_a_base_action() -> None:
    model = _model()
    execution = torch.tensor([[0.4, 0.6, 2.4, 1.6, 0.4, 0.6]])
    with torch.no_grad():
        model.action.bias[:3] = torch.tensor([0.0, 0.0, 1.0])
        model.strict_veto.output.bias.fill_(-100.0)
    with torch.inference_mode():
        skipped = model.forward_from_market_encoding(
            *_inputs(model, 1), execution_context=execution
        )
    assert skipped["base_action_id"].item() == 2
    assert skipped["action_logits"][:, :3].argmax(dim=-1).item() == 2

    with torch.no_grad():
        model.action.bias[:3] = torch.tensor([0.0, 1.0, 0.0])
    with torch.inference_mode():
        called = model.forward_from_market_encoding(
            *_inputs(model, 1), execution_context=execution
        )
    assert called["base_action_id"].item() == 1
    assert called["action_logits"][:, :3].argmax(dim=-1).item() in {1, 2}


def test_h021_rejects_missing_execution_context_fields() -> None:
    model = _model()
    with pytest.raises(ValueError, match=r"shape \[batch, 6\]"):
        model.forward_from_market_encoding(
            *_inputs(model, 1), execution_context=torch.ones((1, 5))
        )
