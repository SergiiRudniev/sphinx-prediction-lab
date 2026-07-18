"""Inspectable group-latent architecture for Sphinx Trace S0 H011."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sphinx_trace.model import RMSNorm


class InspectableBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Model width must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.attention_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_output = nn.Linear(width, width)
        self.ffn_norm = RMSNorm(width)
        self.gate_up = nn.Linear(width, ffn_width * 2)
        self.ffn_output = nn.Linear(ffn_width, width)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, width = inputs.shape
        normalized = self.attention_norm(inputs)
        qkv = self.qkv(normalized).view(
            batch,
            length,
            3,
            self.heads,
            self.head_width,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention: Tensor | None = None
        if return_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
            weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            attended = torch.matmul(weights, value)
            attention = weights.mean(dim=1)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
            )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        inputs = inputs + self.attention_output(attended)
        gate, up = self.gate_up(self.ffn_norm(inputs)).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return cast(Tensor, inputs + self.ffn_output(hidden)), attention


class ScalarHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs).squeeze(-1))


class SphinxTraceS0H011(nn.Module):
    """Attend over causal recurrent groups, never over a truncated wallet list."""

    def __init__(self, config: dict[str, Any], *, candidate_id: str = "50m") -> None:
        super().__init__()
        architecture = config["architecture"]
        candidates = {str(candidate["id"]): candidate for candidate in architecture["candidates"]}
        if candidate_id not in candidates:
            raise ValueError(f"Unknown H011 model candidate: {candidate_id}")
        candidate = candidates[candidate_id]
        self.candidate_id = candidate_id
        self.width = int(candidate["width"])
        self.latent_count = int(architecture["latent_tokens"])
        self.group_specs = tuple(
            (str(group["id"]), int(group["start"]), int(group["stop"]))
            for group in config["input"]["groups"]
        )
        self.group_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(stop - start, self.width),
                    nn.GELU(),
                    nn.Linear(self.width, self.width),
                    RMSNorm(self.width),
                )
                for name, start, stop in self.group_specs
            }
        )
        self.group_type = nn.Parameter(torch.zeros(1, len(self.group_specs), self.width))
        self.latents = nn.Parameter(torch.zeros(1, self.latent_count, self.width))
        self.blocks = nn.ModuleList(
            [
                InspectableBlock(
                    self.width,
                    int(candidate["heads"]),
                    int(candidate["ffn_width"]),
                    float(architecture["dropout"]),
                )
                for _ in range(int(candidate["layers"]))
            ]
        )
        self.final_norm = RMSNorm(self.width)
        self.outcome = ScalarHead(self.width)
        self.uncertainty = ScalarHead(self.width)
        self.sufficiency = ScalarHead(self.width)
        self.edge = ScalarHead(self.width)
        self.size_alpha = ScalarHead(self.width)
        self.size_beta = ScalarHead(self.width)
        nn.init.normal_(self.group_type, std=0.02)
        nn.init.normal_(self.latents, std=0.02)

    def forward(
        self,
        features: Tensor,
        *,
        group_mask: Tensor | None = None,
        return_debug: bool = False,
        return_latent: bool = False,
    ) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != 128:
            raise ValueError("H011 features must have shape [batch, 128]")
        groups = torch.stack(
            [
                self.group_encoders[name](features[:, start:stop])
                for name, start, stop in self.group_specs
            ],
            dim=1,
        )
        groups = groups + self.group_type
        if group_mask is not None:
            if group_mask.shape != groups.shape[:2]:
                raise ValueError("H011 group mask must have shape [batch, groups]")
            groups = groups * group_mask.unsqueeze(-1).to(groups.dtype)
        latents = self.latents.expand(features.shape[0], -1, -1)
        hidden = torch.cat((latents, groups), dim=1)
        attentions: list[Tensor] = []
        for block in self.blocks:
            hidden, attention = block(hidden, return_attention=return_debug)
            if attention is not None:
                attentions.append(attention)
        latent_state = self.final_norm(hidden[:, : self.latent_count]).mean(dim=1)
        output = {
            "terminal_outcome_logit": self.outcome(latent_state),
            "uncertainty_log_scale": self.uncertainty(latent_state),
            "call_sufficiency_logit": self.sufficiency(latent_state),
            "expected_net_edge": self.edge(latent_state),
            "position_size_beta_alpha": F.softplus(self.size_alpha(latent_state)) + 1.0,
            "position_size_beta_beta": F.softplus(self.size_beta(latent_state)) + 1.0,
        }
        if return_debug or return_latent:
            output["debug_latent_state"] = latent_state
        if return_debug:
            output["debug_group_tokens"] = groups
            output["debug_attention"] = torch.stack(attentions, dim=1)
        return output


def h011_variant_feature_mask(variant_id: str, *, device: torch.device | None = None) -> Tensor:
    """Return a feature-level ablation mask without dropping source observations."""

    mask = torch.ones(128, dtype=torch.float32, device=device)
    if variant_id == "h011_market_only":
        mask[48:] = 0.0
    elif variant_id == "h011_uncapped_wallet_flow":
        mask[85:99] = 0.0
        mask[104:111] = 0.0
    elif variant_id == "h011_causal_wallet_performance":
        mask[89:99] = 0.0
        mask[108:110] = 0.0
    elif variant_id == "h011_actor_context":
        pass
    elif variant_id == "h011_temporal_graph":
        raise RuntimeError("Temporal graph variant requires the complete Polygon extension")
    else:
        raise ValueError(f"Unknown H011 variant: {variant_id}")
    return mask


def h011_variant_group_mask(variant_id: str, *, device: torch.device | None = None) -> Tensor:
    if variant_id == "h011_temporal_graph":
        raise RuntimeError("Temporal graph variant requires the complete Polygon extension")
    if variant_id not in {
        "h011_market_only",
        "h011_uncapped_wallet_flow",
        "h011_causal_wallet_performance",
        "h011_actor_context",
    }:
        raise ValueError(f"Unknown H011 variant: {variant_id}")
    mask = torch.ones(6, dtype=torch.float32, device=device)
    if variant_id == "h011_market_only":
        mask[2:] = 0.0
    return mask
