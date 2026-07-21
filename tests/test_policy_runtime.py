from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch

from sphinx_trace.config import load_json
from sphinx_trace.h022_runtime import H022DecisionDebug
from sphinx_trace.h023_runtime import H023DecisionDebug
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.model_h021 import SphinxTraceS0H021
from sphinx_trace.policy_decisions import LoadedPolicyFeature, PolicyDecisionRef
from sphinx_trace.policy_encodings import LoadedPolicyEncoding
from sphinx_trace.policy_runtime import H012PolicyRuntime
from sphinx_trace.replay_h010 import (
    BinaryMarketContract,
    H010ReplayAdapter,
    SelectiveAction,
)
from sphinx_trace.simulator import ReplaySimulator, SimulationRules

ROOT = Path(__file__).resolve().parents[1]


class FeatureStore:
    def load(self, _ref: PolicyDecisionRef) -> LoadedPolicyFeature:
        return LoadedPolicyFeature(np.zeros(128, dtype=np.float32), 0.6, "ab" * 32)


class EncodingStore:
    def load(self, _ref: PolicyDecisionRef) -> LoadedPolicyEncoding:
        return LoadedPolicyEncoding(np.zeros(64, dtype=np.float32), 2.0, -0.5)


class H022SkipRuntime:
    policy_sha256 = "ef" * 32

    def score(self, *_args: object) -> H022DecisionDebug:
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


class H023SkipRuntime:
    policy_sha256 = "12" * 32

    def score(self, *_args: object) -> H023DecisionDebug:
        return H023DecisionDebug(
            keep_base_call=False,
            gate_reason="nonpositive_expected_realized_net_contribution",
            candidate_action_id=0,
            entry_price=0.61,
            break_even_probability=0.61,
            neural_realized_contribution=-0.001,
            neural_conditional_return_mean=-0.01,
            neural_conditional_return_quantiles=(-0.1, -0.01, 0.05),
            neural_fill_probability=0.5,
            neural_positive_probability=0.25,
            neural_keep_logit=-1.0,
            tree_realized_contribution=-0.002,
            ensemble_realized_contribution=-0.0015,
            stacker_intercept=0.0,
            stacker_contributions=(0.0,) * 12,
            neural_group_attention=(1.0 / 7.0,) * 7,
            tree_group_contributions=(0.0,) * 7,
            top_tree_features=(),
            h022_ensemble_net_return=-0.0015,
        )


def _model() -> SphinxTraceS0H012:
    model_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json")
    )
    model_config["architecture"]["candidates"].append(
        {"id": "runtime", "width": 64, "heads": 4, "layers": 1, "ffn_width": 128}
    )
    policy_config = deepcopy(
        load_json(
            ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
        )
    )
    model = SphinxTraceS0H012(
        SphinxTraceS0H011(model_config, candidate_id="runtime"), policy_config
    )
    with torch.no_grad():
        model.action.weight.zero_()
        model.action.bias.copy_(torch.tensor([10.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]))
    return model


def _h021_model() -> SphinxTraceS0H021:
    base = _model()
    policy_config = deepcopy(
        load_json(ROOT / "configs" / "trace" / "sphinx_trace_s0_h021_policy_v1.json")
    )
    policy_config["architecture"]["outcome_calibration_head"]["hidden_width"] = 16
    policy_config["architecture"]["strict_veto_head"]["hidden_width"] = 16
    policy_config["architecture"]["protocol_action_value_head"]["hidden_width"] = 16
    model = SphinxTraceS0H021(base.outcome_backbone, policy_config)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and not unexpected
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
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
    )

    inferred = runtime.infer(ref, adapter)

    assert inferred.call.action == SelectiveAction.CALL_OUTCOME_0
    assert float(inferred.call.probability_outcome0) == pytest.approx(
        float(torch.sigmoid(torch.tensor(2.0)))
    )
    assert inferred.portfolio_features[:2] == (1.0, 1.0)


def test_h021_runtime_binds_execution_context_and_vetoes_no_upside() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(initial_cash_usd=Decimal("100"), fee_bps=Decimal("0"))
        ),
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
        _h021_model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
    )

    inferred = runtime.infer(
        ref,
        adapter,
        {"yes-token": Decimal("0.99"), "no-token": Decimal("0.01")},
    )

    assert inferred.call.action == SelectiveAction.SKIP
    assert inferred.execution_context == pytest.approx((1.0, 0.02, 1.0, 50.0, 0.6, 0.4))
    assert inferred.base_action_logits is not None
    assert max(range(3), key=inferred.base_action_logits.__getitem__) == 0
    assert inferred.no_upside_veto is True
    assert inferred.calibrated_outcome_probabilities is not None
    assert sum(inferred.calibrated_outcome_probabilities) == pytest.approx(1.0)

    cached_runtime = H012PolicyRuntime(
        _h021_model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
    )
    cached = cached_runtime.infer(
        ref,
        adapter,
        {"yes-token": Decimal("0.99"), "no-token": Decimal("0.01")},
    )
    assert cached.call.action == SelectiveAction.SKIP
    assert cached.no_upside_veto is True
    assert cached.execution_context == inferred.execution_context


def test_h022_runtime_can_only_veto_an_h021_call() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(initial_cash_usd=Decimal("100"), fee_bps=Decimal("0"))
        ),
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
        _h021_model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
        h022_runtime=H022SkipRuntime(),  # type: ignore[arg-type]
    )

    inferred = runtime.infer(
        ref,
        adapter,
        {"yes-token": Decimal("0.60"), "no-token": Decimal("0.40")},
    )

    assert inferred.call.action == SelectiveAction.SKIP
    assert inferred.h022_debug is not None
    assert inferred.h022_debug.candidate_action_id == 0
    assert inferred.protocol_action_values is not None
    assert float(inferred.call.probability_outcome0) == pytest.approx(0.7)
    assert all(np.isfinite(value) for value in inferred.action_logits)
    assert inferred.h022_shadow is False


def test_h022_shadow_scores_without_mutating_h021_call() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(initial_cash_usd=Decimal("100"), fee_bps=Decimal("0"))
        ),
        {"condition": contract},
        source_sha256="cd" * 32,
    )
    ref = PolicyDecisionRef(
        "calibration",
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
        _h021_model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
        h022_runtime=H022SkipRuntime(),  # type: ignore[arg-type]
        h022_shadow=True,
    )

    inferred = runtime.infer(
        ref,
        adapter,
        {"yes-token": Decimal("0.60"), "no-token": Decimal("0.40")},
    )

    assert inferred.call.action == SelectiveAction.CALL_OUTCOME_0
    assert inferred.h022_debug is not None
    assert inferred.h022_debug.keep_base_call is False
    assert inferred.h022_shadow is True
    assert float(inferred.call.probability_outcome0) != pytest.approx(0.7)


def test_h023_veto_replaces_h021_call_after_h022_shadow_scoring() -> None:
    contract = BinaryMarketContract(
        "condition", "component", ("Yes", "No"), ("yes-token", "no-token")
    )
    adapter = H010ReplayAdapter(
        ReplaySimulator(
            SimulationRules(initial_cash_usd=Decimal("100"), fee_bps=Decimal("0"))
        ),
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
        _h021_model(),
        FeatureStore(),  # type: ignore[arg-type]
        torch.ones(128),
        torch.ones(6),
        torch.device("cpu"),
        encoding_store=EncodingStore(),  # type: ignore[arg-type]
        h022_runtime=H022SkipRuntime(),  # type: ignore[arg-type]
        h022_shadow=True,
        h023_runtime=H023SkipRuntime(),  # type: ignore[arg-type]
    )

    inferred = runtime.infer(
        ref,
        adapter,
        {"yes-token": Decimal("0.60"), "no-token": Decimal("0.40")},
    )

    assert inferred.call.action == SelectiveAction.SKIP
    assert inferred.h022_debug is not None
    assert inferred.h022_debug.keep_base_call is False
    assert inferred.h023_debug is not None
    assert inferred.h023_debug.keep_base_call is False
    assert inferred.h023_debug.gate_reason.startswith("nonpositive")
    assert float(inferred.call.probability_outcome0) != pytest.approx(0.7)
    assert all(np.isfinite(value) for value in inferred.action_logits)
