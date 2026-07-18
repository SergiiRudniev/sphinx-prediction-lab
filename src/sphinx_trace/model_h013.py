"""Market-anchored residual outcome architecture for Sphinx Trace H013."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from sphinx_trace.model_h011 import (
    SphinxTraceS0H011,
    h011_variant_feature_mask,
    h011_variant_group_mask,
)

H013_VARIANT_TO_H011 = {
    "h013_market_residual": "h011_market_only",
    "h013_uncapped_wallet_flow_residual": "h011_uncapped_wallet_flow",
    "h013_causal_wallet_performance_residual": "h011_causal_wallet_performance",
    "h013_actor_context_residual": "h011_actor_context",
}


class SphinxTraceS0H013(nn.Module):
    """Learn an additive correction while step zero equals the market exactly."""

    def __init__(
        self,
        backbone: SphinxTraceS0H011,
        *,
        minimum_probability: float = 1e-6,
        maximum_probability: float = 1.0 - 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_probability < maximum_probability < 1.0:
            raise ValueError("H013 anchor probability bounds are invalid")
        self.backbone = backbone
        self.minimum_probability = minimum_probability
        self.maximum_probability = maximum_probability
        residual_output = cast(nn.Linear, backbone.outcome.layers[-1])
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

    def forward(
        self,
        features: Tensor,
        market_probability: Tensor,
        *,
        group_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if market_probability.shape != (features.shape[0],):
            raise ValueError("H013 market_probability must have shape [batch]")
        if not bool(torch.isfinite(market_probability).all()):
            raise ValueError("H013 market_probability must be finite")
        anchor = market_probability.clamp(
            self.minimum_probability,
            self.maximum_probability,
        )
        anchor_logit = torch.logit(anchor)
        output = cast(
            dict[str, Tensor],
            self.backbone(
                features,
                group_mask=group_mask,
                return_debug=return_debug,
            ),
        )
        residual = output["terminal_outcome_logit"]
        output["market_anchor_logit"] = anchor_logit
        output["terminal_outcome_residual_logit"] = residual
        output["terminal_outcome_logit"] = anchor_logit + residual
        return output


def h013_variant_feature_mask(
    variant_id: str,
    *,
    device: torch.device | None = None,
) -> Tensor:
    try:
        h011_variant = H013_VARIANT_TO_H011[variant_id]
    except KeyError as error:
        raise ValueError(f"Unknown H013 variant: {variant_id}") from error
    return h011_variant_feature_mask(h011_variant, device=device)


def h013_variant_group_mask(
    variant_id: str,
    *,
    device: torch.device | None = None,
) -> Tensor:
    try:
        h011_variant = H013_VARIANT_TO_H011[variant_id]
    except KeyError as error:
        raise ValueError(f"Unknown H013 variant: {variant_id}") from error
    return h011_variant_group_mask(h011_variant, device=device)
