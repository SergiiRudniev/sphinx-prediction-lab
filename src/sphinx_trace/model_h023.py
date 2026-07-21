"""H022-initialized neural member for H023 fill-realized gating."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sphinx_trace.h022_features import H022_TREE_FEATURE_WIDTH
from sphinx_trace.model_h022 import (
    H022_GROUP_IDS,
    H022_MARKET_LATENT_WIDTH,
    H022_TREE_GROUP_RANGES,
    SphinxTraceS0H022NeuralMember,
)

H023_AUX_FEATURE_WIDTH = 11


def _head(width: int, hidden_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, hidden_width),
        nn.GELU(),
        nn.Linear(hidden_width, output_width),
    )


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


class SphinxTraceS0H023NeuralMember(SphinxTraceS0H022NeuralMember):
    """Predict fill, conditional return, realized contribution and KEEP utility."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        width = int(config["hidden_width"])
        head_width = int(config.get("head_hidden_width", width))
        self.aux_encoder = _encoder(H023_AUX_FEATURE_WIDTH, width, float(config["dropout"]))
        self.aux_gate = nn.Parameter(torch.zeros(width))
        self.realized_net_contribution = _head(width, head_width, 1)
        self.positive_contribution = _head(width, head_width, 1)
        self.keep_utility = _head(width, head_width, 1)

    def forward(
        self,
        market_latents: Tensor,
        tree_features: Tensor,
        terminal_outcome_logits: Tensor,
        candidate_action_ids: Tensor,
        break_even_probabilities: Tensor,
        h023_aux_features: Tensor | None = None,
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
            or h023_aux_features is None
            or h023_aux_features.shape != (rows, H023_AUX_FEATURE_WIDTH)
        ):
            raise ValueError("H023 neural inputs are not aligned")
        if bool(((candidate_action_ids < 0) | (candidate_action_ids > 1)).any()):
            raise ValueError("H023 candidate side must be CALL-0 or CALL-1")
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
        representation = state[:, 0] + self.aux_gate * self.aux_encoder(
            h023_aux_features
        )
        calibration_delta = self.outcome_calibration(representation).squeeze(1)
        calibrated_outcome_logit = terminal_outcome_logits + calibration_delta
        probability0 = torch.sigmoid(calibrated_outcome_logit.float()).to(
            tree_features.dtype
        )
        selected_probability = torch.where(
            candidate_action_ids == 0, probability0, 1.0 - probability0
        )
        selected_edge = selected_probability - break_even_probabilities
        conditional_return_mean = self.net_return_mean(representation).squeeze(1)
        quantile_raw = self.net_return_quantiles(representation)
        median = quantile_raw[:, 1]
        lower = median - torch.nn.functional.softplus(
            quantile_raw[:, 0]
        ) * self.quantile_delta_scale
        upper = median + torch.nn.functional.softplus(
            quantile_raw[:, 2]
        ) * self.quantile_delta_scale
        conditional_return_quantiles = torch.stack((lower, median, upper), dim=1)
        fill_logit = self.fill_probability(representation).squeeze(1)
        positive_logit = self.positive_contribution(representation).squeeze(1)
        keep_logit = self.keep_utility(representation).squeeze(1)
        output = {
            "calibration_delta": calibration_delta,
            "calibrated_outcome_logit": calibrated_outcome_logit,
            "calibrated_outcome_probability0": probability0,
            "calibrated_candidate_probability": selected_probability,
            "calibrated_candidate_edge": selected_edge,
            "conditional_realized_return_mean": conditional_return_mean,
            "conditional_realized_return_quantiles": conditional_return_quantiles,
            "fill_logit": fill_logit,
            "fill_probability": torch.sigmoid(fill_logit.float()).to(
                tree_features.dtype
            ),
            "realized_net_contribution_mean": self.realized_net_contribution(
                representation
            ).squeeze(1),
            "positive_contribution_logit": positive_logit,
            "probability_realized_contribution_positive": torch.sigmoid(
                positive_logit.float()
            ).to(tree_features.dtype),
            "keep_base_call_logit": keep_logit,
            "keep_base_call_probability": torch.sigmoid(keep_logit.float()).to(
                tree_features.dtype
            ),
        }
        if return_debug:
            if attention is None:
                raise RuntimeError("H023 attention stack returned no debug weights")
            output["debug_group_attention"] = attention[:, :, 0, 1:].mean(dim=1)
            output["debug_group_tokens"] = state[:, 1:]
        return output


def load_h022_initialization(
    model: SphinxTraceS0H023NeuralMember,
    h022_state: dict[str, Tensor],
) -> tuple[str, ...]:
    """Load the frozen receipt-selected H022 representation and legacy heads."""

    missing, unexpected = model.load_state_dict(h022_state, strict=False)
    expected_prefixes = (
        "aux_encoder.",
        "aux_gate",
        "realized_net_contribution.",
        "positive_contribution.",
        "keep_utility.",
    )
    if (
        unexpected
        or len(missing) != 27
        or any(not name.startswith(expected_prefixes) for name in missing)
    ):
        raise RuntimeError(
            "H023 initialization no longer matches H022: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(sorted(missing))


H023_GROUP_IDS = H022_GROUP_IDS
