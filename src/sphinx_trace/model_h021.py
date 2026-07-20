"""Price-aware strict KEEP/SKIP veto over the frozen H014 policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h012 import (
    H012_ACTION_COUNT,
    PolicyVectorHead,
    SphinxTraceS0H012,
)
from sphinx_trace.model_h013 import SphinxTraceS0H013

H021_EXECUTION_CONTEXT_WIDTH = 6
H021_EXECUTION_CONTEXT_FIELDS = (
    "entry_price_outcome0",
    "entry_price_outcome1",
    "winning_payout_per_total_cost_outcome0",
    "winning_payout_per_total_cost_outcome1",
    "market_probability_outcome0",
    "market_probability_outcome1",
)


class SphinxTraceS0H021(SphinxTraceS0H012):
    """Let H014 propose a call and let a learned economic gate only keep or skip it."""

    def __init__(
        self,
        outcome_backbone: SphinxTraceS0H011 | SphinxTraceS0H013,
        policy_config: dict[str, Any],
    ) -> None:
        super().__init__(outcome_backbone, policy_config)
        architecture = policy_config["architecture"]
        gate = architecture["strict_veto_head"]
        calibration = architecture["outcome_calibration_head"]
        if self.action_residual is not None or self.protocol_action_value is None:
            raise ValueError(
                "H021 requires no generic residual and one protocol-value head"
            )

        calibration_input_width = self.width + 2 + H021_EXECUTION_CONTEXT_WIDTH
        self.outcome_calibration = PolicyVectorHead(
            calibration_input_width,
            int(calibration["hidden_width"]),
            1,
            layers=int(calibration["layers"]),
            dropout=float(calibration["dropout"]),
            zero_initialize_output=bool(calibration["zero_initialize_residual"]),
        )
        gate_input_width = self.width + 3 + 1 + 9 + 7 + 3 + 6 + 2 + 2 + 1
        self.strict_veto = PolicyVectorHead(
            gate_input_width,
            int(gate["hidden_width"]),
            1,
            layers=int(gate["layers"]),
            dropout=float(gate["dropout"]),
            zero_initialize_output=bool(gate["zero_initialize_veto_delta"]),
        )
        self.no_upside_epsilon = float(architecture.get("no_upside_epsilon", 1e-8))
        if self.no_upside_epsilon < 0.0:
            raise ValueError("H021 no-upside epsilon cannot be negative")

    def forward(
        self,
        market_features: Tensor,
        portfolio_features: Tensor,
        prediction_memory_features: Tensor,
        previous_action_ids: Tensor,
        *,
        execution_context: Tensor | None = None,
        market_probability: Tensor | None = None,
        market_group_mask: Tensor | None = None,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if execution_context is None:
            raise ValueError("H021 requires explicit execution context")
        if isinstance(self.outcome_backbone, SphinxTraceS0H013):
            if market_probability is None:
                raise ValueError("H021 residual backbone requires market_probability")
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
            execution_context=execution_context,
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
        execution_context: Tensor | None = None,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if execution_context is None:
            raise ValueError("H021 requires explicit execution context")
        batch = market_latent.shape[0]
        if execution_context.shape != (batch, H021_EXECUTION_CONTEXT_WIDTH):
            raise ValueError("H021 execution_context must have shape [batch, 6]")
        if not bool(torch.isfinite(execution_context).all()):
            raise ValueError("H021 execution context must be finite")
        prices = execution_context[:, :2]
        payouts = execution_context[:, 2:4]
        probabilities = execution_context[:, 4:6]
        if bool(((prices <= 0.0) | (prices > 1.0)).any()):
            raise ValueError("H021 execution prices must be in (0, 1]")
        if bool((payouts <= 0.0).any()):
            raise ValueError("H021 winning payout multipliers must be positive")
        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
            raise ValueError("H021 market probabilities must be in [0, 1]")
        return self._forward_from_market_encoding_unchecked(
            market_latent,
            terminal_outcome_logit,
            uncertainty_log_scale,
            portfolio_features,
            prediction_memory_features,
            previous_action_ids,
            execution_context=execution_context,
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
        execution_context: Tensor | None = None,
        physical_action_mask: Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, Tensor]:
        if execution_context is None:
            raise ValueError("H021 requires explicit execution context")
        base = super()._forward_from_market_encoding_unchecked(
            market_latent,
            terminal_outcome_logit,
            uncertainty_log_scale,
            portfolio_features,
            prediction_memory_features,
            previous_action_ids,
            physical_action_mask=None,
            return_debug=True,
        )
        raw_base_logits = base["action_logits"]
        batch = len(raw_base_logits)
        physical = (
            torch.ones(
                (batch, H012_ACTION_COUNT),
                dtype=torch.bool,
                device=raw_base_logits.device,
            )
            if physical_action_mask is None
            else physical_action_mask.bool()
        )
        if physical.shape != raw_base_logits.shape or not bool(
            physical[:, :3].any(dim=1).all()
        ):
            raise ValueError("H021 physical action mask must permit a selective action")

        minimum = torch.finfo(raw_base_logits.dtype).min
        selective_base_logits = raw_base_logits[:, :3].masked_fill(
            ~physical[:, :3], minimum
        )
        base_action = selective_base_logits.argmax(dim=-1)
        top_two = selective_base_logits.float().topk(2, dim=-1).values
        margin = (top_two[:, 0] - top_two[:, 1]).to(raw_base_logits.dtype)
        policy_state = base["debug_policy_state"]
        calibration_input = torch.cat(
            (
                policy_state,
                terminal_outcome_logit.unsqueeze(1),
                uncertainty_log_scale.unsqueeze(1),
                execution_context,
            ),
            dim=1,
        )
        calibration_delta = self.outcome_calibration(calibration_input).squeeze(1)
        calibrated_logit = terminal_outcome_logit + calibration_delta
        calibrated_q0 = torch.sigmoid(calibrated_logit.float()).to(
            raw_base_logits.dtype
        )
        calibrated_probabilities = torch.stack(
            (calibrated_q0, 1.0 - calibrated_q0), dim=1
        )
        payouts = execution_context[:, 2:4]
        break_even = (
            payouts.float().reciprocal().clamp(max=1.0).to(raw_base_logits.dtype)
        )
        calibrated_edges = calibrated_probabilities - break_even
        uncertainty = (
            uncertainty_log_scale.float()
            .clamp(-20.0, 20.0)
            .exp()
            .to(raw_base_logits.dtype)
        )
        protocol_values = base["protocol_action_values"]
        gate_input = torch.cat(
            (
                policy_state,
                raw_base_logits[:, :3],
                margin.unsqueeze(1),
                portfolio_features,
                prediction_memory_features,
                protocol_values,
                execution_context,
                calibrated_probabilities,
                calibrated_edges,
                uncertainty.unsqueeze(1),
            ),
            dim=1,
        )
        veto_delta = self.strict_veto(gate_input).squeeze(1)
        base_calls = base_action != 2
        selected_side = base_action.clamp(max=1)
        selected_payout = payouts.gather(1, selected_side[:, None]).squeeze(1)
        no_upside = base_calls & (selected_payout <= 1.0 + self.no_upside_epsilon)
        zero = torch.zeros_like(veto_delta)
        keep_logit = torch.where(no_upside, torch.full_like(zero, minimum), zero)
        skip_logit = torch.where(no_upside, zero, veto_delta)
        strict_selective_logits = torch.stack(
            (
                torch.where(
                    base_action == 0, keep_logit, torch.full_like(zero, minimum)
                ),
                torch.where(
                    base_action == 1, keep_logit, torch.full_like(zero, minimum)
                ),
                torch.where(base_calls, skip_logit, zero),
            ),
            dim=1,
        )
        inactive_logits = torch.full(
            (batch, H012_ACTION_COUNT - 3),
            minimum,
            dtype=raw_base_logits.dtype,
            device=raw_base_logits.device,
        )
        output = dict(base)
        output["action_logits"] = torch.cat(
            (strict_selective_logits, inactive_logits), dim=1
        )
        output["base_action_logits"] = selective_base_logits
        output["strict_veto_logit"] = veto_delta
        output["strict_gate_logits"] = torch.stack((keep_logit, skip_logit), dim=1)
        output["base_action_id"] = base_action
        output["no_upside_veto"] = no_upside
        output["calibrated_outcome_logit"] = calibrated_logit
        output["calibrated_outcome_probabilities"] = calibrated_probabilities
        output["execution_break_even_probabilities"] = break_even
        output["calibrated_execution_edges"] = calibrated_edges
        output["outcome_calibration_delta"] = calibration_delta
        output["execution_entry_prices"] = execution_context[:, :2]
        output["winning_payout_multipliers"] = payouts
        if not return_debug:
            for key in tuple(output):
                if key.startswith("debug_"):
                    del output[key]
        return output
