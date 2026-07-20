"""Sequential learned H012 inference against the exact H010 simulator state."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from sphinx_trace.model_h012 import H012_ACTIONS, SphinxTraceS0H012
from sphinx_trace.model_h021 import H021_EXECUTION_CONTEXT_WIDTH, SphinxTraceS0H021
from sphinx_trace.policy_decisions import (
    LoadedPolicyFeature,
    PolicyDecisionRef,
    PolicyFeatureStore,
    policy_input_digest,
)
from sphinx_trace.policy_encodings import PolicyEncodingStore
from sphinx_trace.replay_h010 import (
    H010ReplayAdapter,
    PolicyCall,
    SelectiveAction,
)


@dataclass(frozen=True, slots=True)
class PolicyInference:
    call: PolicyCall
    feature_date: str
    feature_row: int
    feature_sha256: str
    portfolio_features: tuple[float, ...]
    prediction_memory_features: tuple[float, ...]
    previous_action_id: int
    physical_action_mask: tuple[bool, ...]
    action_logits: tuple[float, ...]
    size_alpha: float
    size_beta: float
    execution_context: tuple[float, ...] | None
    base_action_logits: tuple[float, ...] | None
    strict_veto_logit: float | None
    calibrated_outcome_probabilities: tuple[float, float] | None
    execution_break_even_probabilities: tuple[float, float] | None
    no_upside_veto: bool | None


class _EncodedPolicyCore(nn.Module):
    """Compile the small stateful policy graph without the cached outcome backbone."""

    def __init__(self, model: SphinxTraceS0H012) -> None:
        super().__init__()
        self.model = model
        self.market_width = model.width
        self.portfolio_start = self.market_width
        self.memory_start = self.portfolio_start + 9
        self.previous_action_index = self.memory_start + 7
        self.physical_mask_start = self.previous_action_index + 1
        self.physical_mask_stop = self.physical_mask_start + len(H012_ACTIONS)
        self.h021 = isinstance(model, SphinxTraceS0H021)
        self.terminal_logit_index = self.physical_mask_stop
        self.uncertainty_index = self.terminal_logit_index + 1
        self.execution_context_start = self.uncertainty_index + 1
        self.input_width = (
            self.execution_context_start + H021_EXECUTION_CONTEXT_WIDTH
            if self.h021
            else self.physical_mask_stop
        )

    def forward(self, packed_inputs: Tensor) -> Tensor:
        market_latent = packed_inputs[:, : self.market_width]
        portfolio_features = packed_inputs[:, self.portfolio_start : self.memory_start]
        prediction_memory_features = packed_inputs[
            :, self.memory_start : self.previous_action_index
        ]
        previous_action_ids = packed_inputs[:, self.previous_action_index].long()
        physical_action_mask = packed_inputs[
            :, self.physical_mask_start : self.physical_mask_stop
        ].bool()
        if self.h021:
            h021 = cast(SphinxTraceS0H021, self.model)
            output = h021._forward_from_market_encoding_unchecked(
                market_latent,
                packed_inputs[:, self.terminal_logit_index],
                packed_inputs[:, self.uncertainty_index],
                portfolio_features,
                prediction_memory_features,
                previous_action_ids,
                execution_context=packed_inputs[:, self.execution_context_start :],
                physical_action_mask=physical_action_mask,
            )
        else:
            unused_outcome = packed_inputs[:, 0] * 0.0
            output = self.model._forward_from_market_encoding_unchecked(
                market_latent,
                unused_outcome,
                unused_outcome,
                portfolio_features,
                prediction_memory_features,
                previous_action_ids,
                physical_action_mask=physical_action_mask,
            )
        core = (
            output["action_logits"],
            output["position_size_beta_alpha"].unsqueeze(1),
            output["position_size_beta_beta"].unsqueeze(1),
        )
        if not self.h021:
            return torch.cat(core, dim=1)
        return torch.cat(
            (
                *core,
                output["base_action_logits"],
                output["strict_veto_logit"].unsqueeze(1),
                output["calibrated_outcome_probabilities"],
                output["execution_break_even_probabilities"],
                output["no_upside_veto"].unsqueeze(1).float(),
            ),
            dim=1,
        )


class H012PolicyRuntime:
    """Infer one action after its exact evidence trade has entered H010 state."""

    def __init__(
        self,
        model: SphinxTraceS0H012,
        feature_store: PolicyFeatureStore,
        feature_mask: Tensor,
        group_mask: Tensor,
        device: torch.device,
        encoding_store: PolicyEncodingStore | None = None,
    ) -> None:
        if feature_mask.shape != (128,) or group_mask.shape != (6,):
            raise ValueError("H012 runtime feature/group masks have invalid shapes")
        self.model = model.to(device).eval()
        self.feature_store = feature_store
        self.feature_mask = feature_mask.to(device)
        self.group_mask = group_mask.to(device)
        self.device = device
        self.encoding_store = encoding_store
        self.h021 = isinstance(self.model, SphinxTraceS0H021)
        encoded_policy: nn.Module | None = None
        self.encoded_input_width = 0
        if encoding_store is not None:
            core = _EncodedPolicyCore(self.model).to(device).eval()
            self.encoded_input_width = core.input_width
            encoded_policy = core
            if device.type == "cuda":
                encoded_policy = cast(
                    nn.Module,
                    torch.compile(
                        encoded_policy,
                        mode="reduce-overhead",
                        fullgraph=True,
                        dynamic=False,
                    ),
                )
        self.encoded_policy = encoded_policy

    @torch.inference_mode()
    def infer(
        self,
        ref: PolicyDecisionRef,
        adapter: H010ReplayAdapter,
        reference_prices: dict[str, Decimal] | None = None,
    ) -> PolicyInference:
        if ref.condition_id not in adapter.contracts:
            raise RuntimeError("H012 decision has no H010 contract")
        loaded: LoadedPolicyFeature = self.feature_store.load(ref)
        portfolio = adapter.portfolio_features()
        previous_action_id, memory = adapter.prediction_memory_features(
            ref.condition_id, ref.timestamp_unix
        )
        physical = adapter.physical_action_mask(ref.condition_id)
        execution_context: tuple[float, ...] | None = None
        if self.h021:
            if reference_prices is None:
                raise RuntimeError("H021 inference requires causal reference prices")
            quoted = adapter.candidate_execution_context(
                ref.condition_id,
                ref.timestamp_unix,
                ref.evidence_trade_id,
                reference_prices,
            )
            execution_context = (
                *(float(value) for value in quoted),
                loaded.market_probability_outcome0,
                1.0 - loaded.market_probability_outcome0,
            )
        base_logits: tuple[float, ...] | None = None
        veto_logit: float | None = None
        calibrated_probabilities: tuple[float, float] | None = None
        break_even_probabilities: tuple[float, float] | None = None
        no_upside_veto: bool | None = None
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            if self.encoding_store is None:
                portfolio_tensor = torch.tensor(
                    [portfolio], dtype=torch.float32, device=self.device
                )
                memory_tensor = torch.tensor(
                    [memory], dtype=torch.float32, device=self.device
                )
                previous_tensor = torch.tensor(
                    [previous_action_id], dtype=torch.long, device=self.device
                )
                physical_tensor = torch.tensor(
                    [physical], dtype=torch.bool, device=self.device
                )
                features = (
                    torch.from_numpy(np.asarray(loaded.normalized, dtype=np.float32))
                    .to(self.device)
                    .unsqueeze(0)
                    * self.feature_mask
                )
                market_tensor = torch.tensor(
                    [loaded.market_probability_outcome0],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.h021:
                    if execution_context is None:
                        raise RuntimeError("H021 execution context disappeared")
                    output = self.model(
                        features,
                        portfolio_tensor,
                        memory_tensor,
                        previous_tensor,
                        execution_context=torch.tensor(
                            [execution_context],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        market_probability=market_tensor,
                        market_group_mask=self.group_mask.unsqueeze(0),
                        physical_action_mask=physical_tensor,
                    )
                else:
                    output = self.model(
                        features,
                        portfolio_tensor,
                        memory_tensor,
                        previous_tensor,
                        market_probability=market_tensor,
                        market_group_mask=self.group_mask.unsqueeze(0),
                        physical_action_mask=physical_tensor,
                    )
                logits_tensor = output["action_logits"][0].float()
                logits = tuple(float(value) for value in logits_tensor.cpu().tolist())
                alpha = float(output["position_size_beta_alpha"][0].float())
                beta = float(output["position_size_beta_beta"][0].float())
                if self.h021:
                    base_logits = tuple(
                        float(value)
                        for value in output["base_action_logits"][0]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    veto_logit = float(output["strict_veto_logit"][0].float())
                    calibrated_values = tuple(
                        float(value)
                        for value in output["calibrated_outcome_probabilities"][0]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    break_even_values = tuple(
                        float(value)
                        for value in output["execution_break_even_probabilities"][0]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    if len(calibrated_values) != 2 or len(break_even_values) != 2:
                        raise RuntimeError("H021 debug probability widths changed")
                    calibrated_probabilities = (
                        calibrated_values[0],
                        calibrated_values[1],
                    )
                    break_even_probabilities = (
                        break_even_values[0],
                        break_even_values[1],
                    )
                    no_upside_veto = bool(output["no_upside_veto"][0])
                    probability = calibrated_probabilities[0]
                else:
                    probability = float(
                        torch.sigmoid(output["terminal_outcome_logit"][0].float())
                    )
            else:
                if self.encoded_policy is None:
                    raise RuntimeError(
                        "H012 cached encoding has no compiled policy core"
                    )
                encoded = self.encoding_store.load(ref)
                packed = np.empty(self.encoded_input_width, dtype=np.float32)
                market_stop = len(encoded.market_latent)
                portfolio_stop = market_stop + len(portfolio)
                memory_stop = portfolio_stop + len(memory)
                physical_start = memory_stop + 1
                packed[:market_stop] = encoded.market_latent
                packed[market_stop:portfolio_stop] = portfolio
                packed[portfolio_stop:memory_stop] = memory
                packed[memory_stop] = previous_action_id
                physical_stop = physical_start + len(physical)
                packed[physical_start:physical_stop] = physical
                if self.h021:
                    if execution_context is None:
                        raise RuntimeError("H021 execution context disappeared")
                    packed[physical_stop] = encoded.terminal_outcome_logit
                    packed[physical_stop + 1] = encoded.uncertainty_log_scale
                    packed[physical_stop + 2 :] = execution_context
                packed_tensor = torch.from_numpy(packed).to(self.device).unsqueeze(0)
                encoded_output = self.encoded_policy(packed_tensor)
                values = tuple(
                    float(value) for value in encoded_output[0].float().cpu().tolist()
                )
                logits = values[: len(H012_ACTIONS)]
                alpha = values[len(H012_ACTIONS)]
                beta = values[len(H012_ACTIONS) + 1]
                if self.h021:
                    debug_start = len(H012_ACTIONS) + 2
                    base_logits = values[debug_start : debug_start + 3]
                    veto_logit = values[debug_start + 3]
                    calibrated_probabilities = (
                        values[debug_start + 4],
                        values[debug_start + 5],
                    )
                    break_even_probabilities = (
                        values[debug_start + 6],
                        values[debug_start + 7],
                    )
                    no_upside_veto = bool(values[debug_start + 8])
                    probability = calibrated_probabilities[0]
                else:
                    terminal = encoded.terminal_outcome_logit
                    if terminal >= 0.0:
                        probability = 1.0 / (1.0 + math.exp(-terminal))
                    else:
                        exponential = math.exp(terminal)
                        probability = exponential / (1.0 + exponential)
        action_id = max(range(len(logits)), key=logits.__getitem__)
        size = alpha / max(alpha + beta, 1e-8)
        input_sha256 = policy_input_digest(
            loaded.feature_sha256,
            loaded.market_probability_outcome0,
            portfolio,
            memory,
            previous_action_id,
            physical,
            execution_context,
        )
        call = PolicyCall(
            decision_id=ref.decision_id,
            timestamp_unix=ref.timestamp_unix,
            condition_id=ref.condition_id,
            component_id=ref.component_id,
            evidence_trade_id=ref.evidence_trade_id,
            action=SelectiveAction(H012_ACTIONS[action_id]),
            probability_outcome0=Decimal(str(probability)),
            size_fraction=Decimal(str(size)),
            input_sha256=input_sha256,
        )
        return PolicyInference(
            call=call,
            feature_date=ref.feature_date,
            feature_row=ref.feature_row,
            feature_sha256=loaded.feature_sha256,
            portfolio_features=portfolio,
            prediction_memory_features=memory,
            previous_action_id=previous_action_id,
            physical_action_mask=physical,
            action_logits=logits,
            size_alpha=alpha,
            size_beta=beta,
            execution_context=execution_context,
            base_action_logits=base_logits,
            strict_veto_logit=veto_logit,
            calibrated_outcome_probabilities=calibrated_probabilities,
            execution_break_even_probabilities=break_even_probabilities,
            no_upside_veto=no_upside_veto,
        )
