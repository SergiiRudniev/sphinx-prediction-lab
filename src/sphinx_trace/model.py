"""Sphinx Trace S0 preflight backbone."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        mean_square = inputs.float().pow(2).mean(-1, keepdim=True)
        normalized = inputs * torch.rsqrt(mean_square + self.epsilon)
        return normalized.to(inputs.dtype) * self.weight


class S0Block(nn.Module):
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

    def forward(self, inputs: Tensor) -> Tensor:
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
        return cast(Tensor, inputs + self.ffn_output(hidden))


class SphinxTraceS0(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        sequence_length: int,
        feature_width: int,
    ) -> None:
        super().__init__()
        model = config["model"]
        width = int(model["width"])
        self.input_projection = nn.Sequential(
            nn.Linear(feature_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        self.token_type = nn.Embedding(3, width)
        self.blocks = nn.ModuleList(
            [
                S0Block(
                    width,
                    int(model["heads"]),
                    int(model["ffn_width"]),
                    float(model["dropout"]),
                )
                for _ in range(int(model["layers"]))
            ]
        )
        self.final_norm = RMSNorm(width)
        self.output = nn.Linear(width, int(model["output_width"]))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, features: Tensor, token_types: Tensor) -> Tensor:
        hidden = self.input_projection(features)
        hidden = hidden + self.position + self.token_type(token_types)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.final_norm(hidden).mean(dim=1)
        return cast(Tensor, self.output(pooled))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
