"""Portfolio-aware selective policy for Sphinx Trace S0 H012."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sphinx_trace.model import RMSNorm
from sphinx_trace.model_h011 import InspectableBlock, ScalarHead, SphinxTraceS0H011
from sphinx_trace.model_h013 import SphinxTraceS0H013

H012_ACTIONS = (
    "CALL_OUTCOME_0",
    "CALL_OUTCOME_1",
    "SKIP",
    "UPDATE",
    "HOLD",
    "REDUCE",
    "CLOSE",
)
H012_ACTION_COUNT = len(H012_ACTIONS)
H012_PORTFOLIO_WIDTH = 9
H012_MEMORY_NUMERIC_WIDTH = 7


class StateTokenEncoder(nn.Module):
    """Project a compact, fully causal state vector into one policy token."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
            RMSNorm(output_width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


class PolicyVectorHead(nn.Module):
    """Optional nonlinear head over the causal policy state."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        *,
        layers: int,
        dropout: float,
        zero_initialize_output: bool,
    ) -> None:
        super().__init__()
        if layers < 1 or hidden_width < 1 or output_width < 1:
            raise ValueError("Policy vector head dimensions must be positive")
        modules: list[nn.Module] = [RMSNorm(input_width)]
        width = input_width
        for _ in range(layers):
            modules.extend(
                (
                    nn.Linear(width, hidden_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            width = hidden_width
        self.features = nn.Sequential(*modules)
        self.output = nn.Linear(width, output_width)
        if zero_initialize_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.output(self.features(inputs)))


class SphinxTraceS0H012(nn.Module):
    """Fuse H011 market evidence with portfolio and prediction-memory state."""

    def __init__(
        self,
        outcome_backbone: SphinxTraceS0H011 | SphinxTraceS0H013,
        policy_config: dict[str, Any],
    ) -> None:
        super().__init__()
        architecture = policy_config["architecture"]
        self.outcome_backbone = outcome_backbone
        raw_backbone = (
            outcome_backbone.backbone
            if isinstance(outcome_backbone, SphinxTraceS0H013)
            else outcome_backbone
        )
        self.width = raw_backbone.width
        self.fusion_latents = int(architecture["fusion_latents"])
        portfolio_width = int(architecture["portfolio_encoder_width"])
        memory_width = int(architecture["prediction_memory_encoder_width"])
        if not raw_backbone.blocks:
            raise ValueError("H012 requires a non-empty H011 outcome backbone")
        backbone_block = cast(InspectableBlock, raw_backbone.blocks[0])
        heads = backbone_block.heads
        if self.width % heads:
            raise ValueError("H012 fusion width must be divisible by attention heads")

        self.portfolio_encoder = StateTokenEncoder(
            H012_PORTFOLIO_WIDTH,
            portfolio_width,
            self.width,
        )
        self.memory_encoder = StateTokenEncoder(
            H012_MEMORY_NUMERIC_WIDTH,
            memory_width,
            self.width,
        )
        self.previous_action = nn.Embedding(H012_ACTION_COUNT, self.width)
        self.memory_norm = RMSNorm(self.width)
        self.policy_latents = nn.Parameter(torch.zeros(1, self.fusion_latents, self.width))
        self.token_type = nn.Parameter(torch.zeros(1, self.fusion_latents + 3, self.width))
        self.blocks = nn.ModuleList(
            [
                InspectableBlock(
                    self.width,
                    heads,
                    backbone_block.gate_up.out_features // 2,
                    backbone_block.dropout,
                )
                for _ in range(int(architecture["fusion_layers"]))
            ]
        )
        self.final_norm = RMSNorm(self.width)
        self.action = nn.Linear(self.width, H012_ACTION_COUNT)
        self.size_alpha = ScalarHead(self.width)
        self.size_beta = ScalarHead(self.width)
        self.value = ScalarHead(self.width)
        residual = architecture.get("residual_policy_head")
        if isinstance(residual, dict) and bool(residual.get("enabled")):
            self.action_residual: PolicyVectorHead | None = PolicyVectorHead(
                self.width,
                int(residual["hidden_width"]),
                H012_ACTION_COUNT,
                layers=int(residual["layers"]),
                dropout=float(residual["dropout"]),
                zero_initialize_output=bool(residual["zero_initialize_action_output"]),
            )
        else:
            self.action_residual = None
        protocol_value = architecture.get("protocol_action_value_head")
        if isinstance(protocol_value, dict) and bool(protocol_value.get("enabled")):
            self.protocol_action_value: PolicyVectorHead | None = PolicyVectorHead(
                self.width,
                int(protocol_value["hidden_width"]),
                3,
                layers=int(protocol_value["layers"]),
                dropout=float(protocol_value["dropout"]),
                zero_initialize_output=True,
            )
        else:
            self.protocol_action_value = None
        nn.init.normal_(self.policy_latents, std=0.02)
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.zeros_(self.action.weight)
        with torch.no_grad():
            self.action.bias.copy_(
                torch.tensor(
                    [-1e-4, -1e-4, 0.0, -1.0, -1.0, -1.0, -1.0],
                    dtype=self.action.bias.dtype,
                )
            )

    def forward(
        self,
        market_features: Tensor,
        portfolio_features: Tensor,
        prediction_memory_features: Tensor,
        previous_action_ids: Tensor,
        *,
        market_probability: Tensor | None = None,
        market_group_mask: Tensor | None = None,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        batch = market_features.shape[0]
        if market_features.shape != (batch, 128):
            raise ValueError("H012 market_features must have shape [batch, 128]")

        if isinstance(self.outcome_backbone, SphinxTraceS0H013):
            if market_probability is None:
                raise ValueError("H012 residual backbone requires market_probability")
            backbone = self.outcome_backbone(
                market_features,
                market_probability,
                group_mask=market_group_mask,
                return_debug=return_debug,
                return_latent=True,
            )
        else:
            backbone = self.outcome_backbone(
                market_features,
                group_mask=market_group_mask,
                return_debug=return_debug,
                return_latent=True,
            )
        output = self.forward_from_market_encoding(
            backbone["debug_latent_state"],
            backbone["terminal_outcome_logit"],
            backbone["uncertainty_log_scale"],
            portfolio_features,
            prediction_memory_features,
            previous_action_ids,
            physical_action_mask=physical_action_mask,
            return_debug=return_debug,
        )
        if return_debug:
            output["debug_market_attention"] = backbone["debug_attention"]
            output["debug_market_group_tokens"] = backbone["debug_group_tokens"]
        return output

    def forward_from_market_encoding(
        self,
        market_latent: Tensor,
        terminal_outcome_logit: Tensor,
        uncertainty_log_scale: Tensor,
        portfolio_features: Tensor,
        prediction_memory_features: Tensor,
        previous_action_ids: Tensor,
        *,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        """Run only the causal state-dependent policy over a bound market encoding."""

        if market_latent.ndim != 2:
            raise ValueError("H012 market_latent must have shape [batch, width]")
        batch = market_latent.shape[0]
        if market_latent.shape != (batch, self.width):
            raise ValueError("H012 market_latent must have shape [batch, width]")
        if terminal_outcome_logit.shape != (batch,):
            raise ValueError("H012 terminal_outcome_logit must have shape [batch]")
        if uncertainty_log_scale.shape != (batch,):
            raise ValueError("H012 uncertainty_log_scale must have shape [batch]")
        if portfolio_features.shape != (batch, H012_PORTFOLIO_WIDTH):
            raise ValueError("H012 portfolio_features must have shape [batch, 9]")
        if prediction_memory_features.shape != (batch, H012_MEMORY_NUMERIC_WIDTH):
            raise ValueError("H012 prediction_memory_features must have shape [batch, 7]")
        if previous_action_ids.shape != (batch,):
            raise ValueError("H012 previous_action_ids must have shape [batch]")
        if previous_action_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("H012 previous_action_ids must be an integer tensor")
        if bool(((previous_action_ids < 0) | (previous_action_ids >= H012_ACTION_COUNT)).any()):
            raise ValueError("H012 previous_action_ids contains an unknown action")
        if physical_action_mask is not None:
            if physical_action_mask.shape != (batch, H012_ACTION_COUNT):
                raise ValueError("H012 physical_action_mask must have shape [batch, actions]")
            if bool((~physical_action_mask.bool().any(dim=1)).any()):
                raise ValueError("H012 physical_action_mask must permit at least one action")
        return self._forward_from_market_encoding_unchecked(
            market_latent,
            terminal_outcome_logit,
            uncertainty_log_scale,
            portfolio_features,
            prediction_memory_features,
            previous_action_ids,
            physical_action_mask=physical_action_mask,
            return_debug=return_debug,
        )

    def _forward_from_market_encoding_unchecked(
        self,
        market_latent: Tensor,
        terminal_outcome_logit: Tensor,
        uncertainty_log_scale: Tensor,
        portfolio_features: Tensor,
        prediction_memory_features: Tensor,
        previous_action_ids: Tensor,
        *,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        """Tensor-only policy core for source-validated sequential replay."""

        batch = market_latent.shape[0]
        portfolio_token = self.portfolio_encoder(portfolio_features)
        memory_token = self.memory_norm(
            self.memory_encoder(prediction_memory_features)
            + self.previous_action(previous_action_ids.long())
        )
        latents = self.policy_latents.expand(batch, -1, -1)
        hidden = torch.cat(
            (
                latents,
                market_latent.unsqueeze(1),
                portfolio_token.unsqueeze(1),
                memory_token.unsqueeze(1),
            ),
            dim=1,
        )
        hidden = hidden + self.token_type
        attentions: list[Tensor] = []
        for block in self.blocks:
            hidden, attention = block(hidden, return_attention=return_debug)
            if attention is not None:
                attentions.append(attention)
        policy_state = self.final_norm(hidden[:, : self.fusion_latents]).mean(dim=1)
        base_action_logits = self.action(policy_state)
        action_logits = base_action_logits
        residual_logits: Tensor | None = None
        if self.action_residual is not None:
            residual_logits = self.action_residual(policy_state)
            action_logits = action_logits + residual_logits
        if physical_action_mask is not None:
            action_mask = physical_action_mask.bool()
            action_logits = action_logits.masked_fill(
                ~action_mask,
                torch.finfo(action_logits.dtype).min,
            )
            base_action_logits = base_action_logits.masked_fill(
                ~action_mask,
                torch.finfo(base_action_logits.dtype).min,
            )

        output = {
            "action_logits": action_logits,
            "position_size_beta_alpha": F.softplus(self.size_alpha(policy_state)) + 1.0,
            "position_size_beta_beta": F.softplus(self.size_beta(policy_state)) + 1.0,
            "state_value": self.value(policy_state),
            "terminal_outcome_logit": terminal_outcome_logit,
            "outcome_uncertainty_log_scale": uncertainty_log_scale,
        }
        if residual_logits is not None:
            output["base_action_logits"] = base_action_logits
            output["action_residual_logits"] = residual_logits
        if self.protocol_action_value is not None:
            output["protocol_action_values"] = self.protocol_action_value(policy_state)
        if return_debug:
            output["debug_policy_state"] = policy_state
            output["debug_portfolio_token"] = portfolio_token
            output["debug_prediction_memory_token"] = memory_token
            output["debug_policy_attention"] = torch.stack(attentions, dim=1)
        return output
