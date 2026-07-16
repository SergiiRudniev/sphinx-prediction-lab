"""Stable contracts between Sphinx Trace inference and deterministic execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class TradeAction(StrEnum):
    """Actions emitted by the position policy."""

    SKIP = "SKIP"
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    HOLD_TO_RESOLUTION = "HOLD_TO_RESOLUTION"


@dataclass(frozen=True, slots=True)
class MarketState:
    """Executable market state available at one causal decision timestamp."""

    condition_id: str
    observed_at: datetime
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    available_depth_usd: float
    seconds_to_resolution: int

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        for name in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.yes_bid > self.yes_ask or self.no_bid > self.no_ask:
            raise ValueError("bid cannot exceed ask")
        if self.available_depth_usd < 0:
            raise ValueError("available_depth_usd cannot be negative")
        if self.seconds_to_resolution < 0:
            raise ValueError("seconds_to_resolution cannot be negative")


@dataclass(frozen=True, slots=True)
class PositionState:
    """Current position known to the deterministic manager."""

    outcome: str | None = None
    shares: float = 0.0
    average_price: float = 0.0
    realized_pnl_usd: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.shares == 0.0


@dataclass(frozen=True, slots=True)
class ModelSignal:
    """Calibrated outputs from Sphinx Trace S0."""

    generated_at: datetime
    fair_yes_probability: float
    informed_flow_yes: float
    expected_yes_edge: float
    expected_no_edge: float
    downside_edge_q10: float
    confidence: float

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        for name in ("fair_yes_probability", "informed_flow_yes", "confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def neutral(cls) -> ModelSignal:
        return cls(
            generated_at=datetime.now(tz=UTC),
            fair_yes_probability=0.5,
            informed_flow_yes=0.5,
            expected_yes_edge=0.0,
            expected_no_edge=0.0,
            downside_edge_q10=0.0,
            confidence=0.0,
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Auditable decision returned by the policy boundary."""

    action: TradeAction
    size_fraction: float
    limit_price: float | None
    reason_code: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.size_fraction <= 1.0:
            raise ValueError("size_fraction must be between 0 and 1")
        if self.limit_price is not None and not 0.0 <= self.limit_price <= 1.0:
            raise ValueError("limit_price must be between 0 and 1")
