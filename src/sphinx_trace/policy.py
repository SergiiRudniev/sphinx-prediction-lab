"""Conservative deterministic boundary around model recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from sphinx_trace.contracts import (
    MarketState,
    ModelSignal,
    PolicyDecision,
    PositionState,
    TradeAction,
)


@dataclass(frozen=True, slots=True)
class PolicyLimits:
    """Frozen limits selected without test or live labels."""

    minimum_edge: float = 0.04
    minimum_confidence: float = 0.70
    minimum_depth_usd: float = 250.0
    maximum_signal_age_seconds: int = 15
    maximum_position_fraction: float = 0.01
    exit_edge: float = 0.01


def decide(
    signal: ModelSignal,
    market: MarketState,
    position: PositionState,
    limits: PolicyLimits,
) -> PolicyDecision:
    """Convert calibrated outputs into a bounded, auditable recommendation."""

    signal_age = (market.observed_at - signal.generated_at).total_seconds()
    if signal_age < 0 or signal_age > limits.maximum_signal_age_seconds:
        return PolicyDecision(TradeAction.SKIP, 0.0, None, "STALE_SIGNAL")
    if market.available_depth_usd < limits.minimum_depth_usd:
        return PolicyDecision(TradeAction.SKIP, 0.0, None, "LOW_DEPTH")
    if signal.confidence < limits.minimum_confidence:
        return PolicyDecision(TradeAction.SKIP, 0.0, None, "LOW_CONFIDENCE")

    if not position.is_flat:
        active_edge = (
            signal.expected_yes_edge if position.outcome == "YES" else signal.expected_no_edge
        )
        if active_edge < limits.exit_edge:
            executable_bid = market.yes_bid if position.outcome == "YES" else market.no_bid
            return PolicyDecision(TradeAction.CLOSE, 1.0, executable_bid, "EDGE_GONE")
        return PolicyDecision(TradeAction.HOLD, 0.0, None, "EDGE_REMAINS")

    size = limits.maximum_position_fraction
    if signal.expected_yes_edge >= limits.minimum_edge:
        return PolicyDecision(TradeAction.BUY_YES, size, market.yes_ask, "YES_EDGE")
    if signal.expected_no_edge >= limits.minimum_edge:
        return PolicyDecision(TradeAction.BUY_NO, size, market.no_ask, "NO_EDGE")
    return PolicyDecision(TradeAction.SKIP, 0.0, None, "NO_NET_EDGE")
