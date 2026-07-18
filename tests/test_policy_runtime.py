from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch

from sphinx_trace.config import load_json
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.policy_decisions import LoadedPolicyFeature, PolicyDecisionRef
from sphinx_trace.policy_encodings import LoadedPolicyEncoding
from sphinx_trace.policy_runtime import H012PolicyRuntime
from sphinx_trace.replay_h010 import BinaryMarketContract, H010ReplayAdapter, SelectiveAction
from sphinx_trace.simulator import ReplaySimulator, SimulationRules

ROOT = Path(__file__).resolve().parents[1]


class FeatureStore:
    def load(self, _ref: PolicyDecisionRef) -> LoadedPolicyFeature:
        return LoadedPolicyFeature(np.zeros(128, dtype=np.float32), 0.6, "ab" * 32)


class EncodingStore:
    def load(self, _ref: PolicyDecisionRef) -> LoadedPolicyEncoding:
        return LoadedPolicyEncoding(np.zeros(64, dtype=np.float32), 2.0, -0.5)


def _model() -> SphinxTraceS0H012:
    model_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    )
    model_config["architecture"]["candidates"].append(
        {"id": "runtime", "width": 64, "heads": 4, "layers": 1, "ffn_width": 128}
    )
    policy_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json")
    )
    model = SphinxTraceS0H012(
        SphinxTraceS0H011(model_config, candidate_id="runtime"), policy_config
    )
    with torch.no_grad():
        model.action.weight.zero_()
        model.action.bias.copy_(torch.tensor([10.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]))
    return model


def test_runtime_uses_post_trade_portfolio_memory_and_physical_mask() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(SimulationRules(initial_cash_usd=Decimal("100"))),
        {"condition": contract},
        source_sha256="cd" * 32,
    )
    ref = PolicyDecisionRef(
        "validation",
        0,
        "2026-01-01",
        0,
        "decision",
        "trade",
        100,
        "condition",
        "component",
        1,
        2,
    )
    runtime = H012PolicyRuntime(
        _model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
    )

    inferred = runtime.infer(ref, adapter)

    assert inferred.call.action == SelectiveAction.CALL_OUTCOME_0
    assert inferred.portfolio_features[:2] == (1.0, 1.0)
    assert inferred.prediction_memory_features[0] == 0.5
    assert inferred.physical_action_mask == (
        True,
        True,
        True,
        False,
        True,
        False,
        False,
    )
    assert 0.0 <= inferred.call.size_fraction <= 1.0
    assert len(inferred.call.input_sha256) == 64


def test_runtime_uses_cached_market_encoding_without_changing_state_inputs() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(SimulationRules(initial_cash_usd=Decimal("100"))),
        {"condition": contract},
        source_sha256="cd" * 32,
    )
    ref = PolicyDecisionRef(
        "validation", 0, "2026-01-01", 0, "decision", "trade", 100,
        "condition", "component", 1, 2,
    )
    runtime = H012PolicyRuntime(
        _model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
    )

    inferred = runtime.infer(ref, adapter)

    assert inferred.call.action == SelectiveAction.CALL_OUTCOME_0
    assert float(inferred.call.probability_outcome0) == pytest.approx(
        float(torch.sigmoid(torch.tensor(2.0)))
    )
    assert inferred.portfolio_features[:2] == (1.0, 1.0)
