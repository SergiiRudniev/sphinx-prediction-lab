"""Sequential learned H012 inference against the exact H010 simulator state."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from sphinx_trace.model_h012 import H012_ACTIONS, SphinxTraceS0H012
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


class _EncodedPolicyCore(nn.Module):
    """Compile the small stateful policy graph without the cached outcome backbone."""

    def __init__(self, model: SphinxTraceS0H012) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        market_latent: Tensor,
        terminal_outcome_logit: Tensor,
        uncertainty_log_scale: Tensor,
        portfolio_features: Tensor,
        prediction_memory_features: Tensor,
        previous_action_ids: Tensor,
        physical_action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.model._forward_from_market_encoding_unchecked(
            market_latent,
            terminal_outcome_logit,
            uncertainty_log_scale,
            portfolio_features,
            prediction_memory_features,
            previous_action_ids,
            physical_action_mask=physical_action_mask,
        )
        return (
            output["action_logits"],
            output["position_size_beta_alpha"],
            output["position_size_beta_beta"],
            output["state_value"],
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
        encoded_policy: nn.Module | None = None
        if encoding_store is not None:
            encoded_policy = _EncodedPolicyCore(self.model).to(device).eval()
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
    ) -> PolicyInference:
        if ref.condition_id not in adapter.contracts:
            raise RuntimeError("H012 decision has no H010 contract")
        loaded: LoadedPolicyFeature = self.feature_store.load(ref)
        portfolio = adapter.portfolio_features()
        previous_action_id, memory = adapter.prediction_memory_features(
            ref.condition_id, ref.timestamp_unix
        )
        physical = adapter.physical_action_mask(ref.condition_id)
        portfolio_tensor = torch.tensor([portfolio], dtype=torch.float32, device=self.device)
        memory_tensor = torch.tensor([memory], dtype=torch.float32, device=self.device)
        previous_tensor = torch.tensor([previous_action_id], dtype=torch.long, device=self.device)
        physical_tensor = torch.tensor([physical], dtype=torch.bool, device=self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            if self.encoding_store is None:
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
                output = self.model(
                    features,
                    portfolio_tensor,
                    memory_tensor,
                    previous_tensor,
                    market_probability=market_tensor,
                    market_group_mask=self.group_mask.unsqueeze(0),
                    physical_action_mask=physical_tensor,
                )
            else:
                if self.encoded_policy is None:
                    raise RuntimeError("H012 cached encoding has no compiled policy core")
                encoded = self.encoding_store.load(ref)
                market_latent = (
                    torch.from_numpy(encoded.market_latent).to(self.device).unsqueeze(0)
                )
                terminal_logit = torch.tensor(
                    [encoded.terminal_outcome_logit],
                    dtype=torch.float32,
                    device=self.device,
                )
                uncertainty = torch.tensor(
                    [encoded.uncertainty_log_scale],
                    dtype=torch.float32,
                    device=self.device,
                )
                action_logits, size_alpha, size_beta, state_value = self.encoded_policy(
                    market_latent,
                    terminal_logit,
                    uncertainty,
                    portfolio_tensor,
                    memory_tensor,
                    previous_tensor,
                    physical_tensor,
                )
                output = {
                    "action_logits": action_logits,
                    "position_size_beta_alpha": size_alpha,
                    "position_size_beta_beta": size_beta,
                    "state_value": state_value,
                    "terminal_outcome_logit": terminal_logit,
                    "outcome_uncertainty_log_scale": uncertainty,
                }
        logits = output["action_logits"][0].float()
        action_id = int(logits.argmax())
        alpha = float(output["position_size_beta_alpha"][0].float())
        beta = float(output["position_size_beta_beta"][0].float())
        size = alpha / max(alpha + beta, 1e-8)
        probability = float(torch.sigmoid(output["terminal_outcome_logit"][0].float()))
        input_sha256 = policy_input_digest(
            loaded.feature_sha256,
            loaded.market_probability_outcome0,
            portfolio,
            memory,
            previous_action_id,
            physical,
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
            action_logits=tuple(float(value) for value in logits.cpu().tolist()),
            size_alpha=alpha,
            size_beta=beta,
        )
