"""Cross-group neural member for H022 conditional net-edge estimation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH

H022_MARKET_LATENT_WIDTH = 512
H022_GROUP_IDS = (
    "market_latent",
    "clock",
    "market",
    "event_component",
    "wallet_flow",
    "universe",
    "price_portfolio_execution",
)
H022_TREE_GROUP_RANGES = (
    (0, 0),
    (0, 8),
    (8, 48),
    (48, 72),
    (72, 116),
    (116, 128),
    (128, H022_TREE_FEATURE_WIDTH),
)


class H022AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.attention_norm(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        values = values + attended
        values = values + self.ffn(self.ffn_norm(values))
        return values, weights


def _encoder(input_width: int, width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, width),
        nn.GELU(),
        nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        nn.GELU(),
        nn.LayerNorm(width),
    )


def _head(width: int, hidden_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, hidden_width),
        nn.GELU(),
        nn.Linear(hidden_width, output_width),
    )


class SphinxTraceS0H022NeuralMember(nn.Module):
    """Estimate calibrated outcome, net-return distribution and fill chance."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["hidden_width"])
        layers = int(config["layers"])
        heads = int(config.get("attention_heads", 8))
        ffn_width = int(config.get("attention_ffn_width", width * 3))
        dropout = float(config["dropout"])
        if width <= 0 or layers <= 0 or width % heads:
            raise ValueError("H022 neural architecture is invalid")
        group_widths = (
            H022_MARKET_LATENT_WIDTH,
            *(stop - start for start, stop in H022_TREE_GROUP_RANGES[1:]),
        )
        self.group_encoders = nn.ModuleList(
            _encoder(input_width, width, dropout) for input_width in group_widths
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.group_embeddings = nn.Parameter(
            torch.empty(1, len(H022_GROUP_IDS) + 1, width)
        )
        nn.init.trunc_normal_(self.group_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            H022AttentionBlock(width, heads, ffn_width, dropout)
            for _ in range(layers)
        )
        head_width = int(config.get("head_hidden_width", width))
        self.outcome_calibration = _head(width, head_width, 1)
        self.net_return_mean = _head(width, head_width, 1)
        self.net_return_quantiles = _head(width, head_width, 3)
        self.fill_probability = _head(width, head_width, 1)
        self.quantile_delta_scale = float(config.get("quantile_delta_scale", 0.01))
        if self.quantile_delta_scale <= 0.0:
            raise ValueError("H022 quantile delta scale must be positive")

    def forward(
        self,
        market_latents: Tensor,
        tree_features: Tensor,
        terminal_outcome_logits: Tensor,
        candidate_action_ids: Tensor,
        break_even_probabilities: Tensor,
        *,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        rows = len(tree_features)
        if (
            market_latents.shape != (rows, H022_MARKET_LATENT_WIDTH)
            or tree_features.shape != (rows, H022_TREE_FEATURE_WIDTH)
            or terminal_outcome_logits.shape != (rows,)
            or candidate_action_ids.shape != (rows,)
            or break_even_probabilities.shape != (rows,)
        ):
            raise ValueError("H022 neural inputs are not aligned")
        if bool(((candidate_action_ids < 0) | (candidate_action_ids > 1)).any()):
            raise ValueError("H022 candidate side must be CALL-0 or CALL-1")
        group_values = [market_latents]
        group_values.extend(
            tree_features[:, start:stop]
            for start, stop in H022_TREE_GROUP_RANGES[1:]
        )
        tokens = torch.stack(
            [
                encoder(values)
                for encoder, values in zip(
                    self.group_encoders, group_values, strict=True
                )
            ],
            dim=1,
        )
        cls = self.cls_token.expand(rows, -1, -1)
        state = torch.cat((cls, tokens), dim=1) + self.group_embeddings
        attention: Tensor | None = None
        for block in self.blocks:
            state, attention = block(state)
        representation = state[:, 0]
        calibration_delta = self.outcome_calibration(representation).squeeze(1)
        calibrated_outcome_logit = terminal_outcome_logits + calibration_delta
        probability0 = torch.sigmoid(calibrated_outcome_logit.float()).to(
            tree_features.dtype
        )
        selected_probability = torch.where(
            candidate_action_ids == 0, probability0, 1.0 - probability0
        )
        selected_edge = selected_probability - break_even_probabilities
        mean = self.net_return_mean(representation).squeeze(1)
        quantile_raw = self.net_return_quantiles(representation)
        median = quantile_raw[:, 1]
        lower = median - torch.nn.functional.softplus(
            quantile_raw[:, 0]
        ) * self.quantile_delta_scale
        upper = median + torch.nn.functional.softplus(
            quantile_raw[:, 2]
        ) * self.quantile_delta_scale
        quantiles = torch.stack((lower, median, upper), dim=1)
        fill_logit = self.fill_probability(representation).squeeze(1)
        output = {
            "calibration_delta": calibration_delta,
            "calibrated_outcome_logit": calibrated_outcome_logit,
            "calibrated_outcome_probability0": probability0,
            "calibrated_candidate_probability": selected_probability,
            "calibrated_candidate_edge": selected_edge,
            "net_return_mean": mean,
            "net_return_quantiles": quantiles,
            "fill_logit": fill_logit,
            "fill_probability": torch.sigmoid(fill_logit.float()).to(
                tree_features.dtype
            ),
        }
        if return_debug:
            if attention is None:
                raise RuntimeError("H022 attention stack returned no debug weights")
            output["debug_group_attention"] = attention[:, :, 0, 1:].mean(dim=1)
            output["debug_group_tokens"] = state[:, 1:]
        return output
