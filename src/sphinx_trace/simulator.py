"""Causal, stateful execution mechanics for Sphinx Trace research replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

ZERO = Decimal("0")
ONE = Decimal("1")


def decimal(value: Decimal | float | int | str) -> Decimal:
    """Convert external numbers without importing their binary float error."""

    return value if isinstance(value, Decimal) else Decimal(str(value))


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


OPEN_ORDER_STATUSES = frozenset({OrderStatus.PENDING, OrderStatus.PARTIAL})


@dataclass(frozen=True, slots=True)
class SimulationRules:
    initial_cash_usd: Decimal = Decimal("10000")
    latency_seconds: int = 2
    maximum_fill_wait_seconds: int = 60
    available_share_fraction: Decimal = Decimal("0.05")
    duplicate_liquidity_haircut: Decimal = Decimal("0.5")
    adverse_price_ticks: int = 1
    tick_size: Decimal = Decimal("0.01")
    fee_bps: Decimal = Decimal("100")
    opposing_side_required: bool = False

    def __post_init__(self) -> None:
        if self.initial_cash_usd <= ZERO:
            raise ValueError("initial_cash_usd must be positive")
        if self.latency_seconds < 0 or self.maximum_fill_wait_seconds < 0:
            raise ValueError("latency and fill wait must be non-negative")
        for name in ("available_share_fraction", "duplicate_liquidity_haircut"):
            value = getattr(self, name)
            if not ZERO <= value <= ONE:
                raise ValueError(f"{name} must be between zero and one")
        if self.adverse_price_ticks < 0 or self.tick_size <= ZERO:
            raise ValueError("adverse ticks must be non-negative and tick size positive")
        if self.fee_bps < ZERO:
            raise ValueError("fee_bps cannot be negative")

    @property
    def fee_rate(self) -> Decimal:
        return self.fee_bps / Decimal("10000")


@dataclass(frozen=True, slots=True)
class LiquidityEvent:
    liquidity_id: str
    timestamp_unix: int
    condition_id: str
    token_id: str
    outcome: str
    price: Decimal
    shares: Decimal
    observed_side: OrderSide | None = None

    def __post_init__(self) -> None:
        if not self.liquidity_id or not self.condition_id or not self.token_id:
            raise ValueError("liquidity, condition and token identifiers are required")
        if self.timestamp_unix < 0:
            raise ValueError("timestamp_unix cannot be negative")
        if not ZERO <= self.price <= ONE:
            raise ValueError("price must be between zero and one")
        if self.shares <= ZERO:
            raise ValueError("shares must be positive")


@dataclass(slots=True)
class SimulatedOrder:
    order_id: str
    decision_id: str
    component_id: str
    condition_id: str
    token_id: str
    outcome: str
    side: OrderSide
    submitted_at_unix: int
    eligible_at_unix: int
    expires_at_unix: int
    requested_shares: Decimal
    limit_price: Decimal
    status: OrderStatus = OrderStatus.PENDING
    filled_shares: Decimal = ZERO
    evidence_sha256: str | None = None
    evidence_liquidity_id: str | None = None
    reject_reason: str | None = None

    @property
    def remaining_shares(self) -> Decimal:
        return max(ZERO, self.requested_shares - self.filled_shares)


@dataclass(slots=True)
class SimulatedPosition:
    condition_id: str
    token_id: str
    outcome: str
    shares: Decimal = ZERO
    cost_basis_usd: Decimal = ZERO

    @property
    def average_price(self) -> Decimal:
        return self.cost_basis_usd / self.shares if self.shares else ZERO


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    fill_id: str
    order_id: str
    liquidity_id: str
    timestamp_unix: int
    side: OrderSide
    shares: Decimal
    price: Decimal
    notional_usd: Decimal
    fee_usd: Decimal


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    decision_id: str
    timestamp_unix: int
    action: str
    probability: Decimal
    size_fraction: Decimal
    input_sha256: str


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _decimal_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Decimal) else value for key, value in payload.items()
    }


class ReplaySimulator:
    """Replay calls against causal tape liquidity and terminal resolutions."""

    def __init__(self, rules: SimulationRules) -> None:
        self.rules = rules
        self.cash_usd = rules.initial_cash_usd
        self.current_time_unix = 0
        self.orders: dict[str, SimulatedOrder] = {}
        self.positions: dict[str, SimulatedPosition] = {}
        self.fills: list[SimulatedFill] = []
        self.predictions: list[PredictionRecord] = []
        self.processed_liquidity_ids: set[str] = set()
        self.last_marks: dict[str, Decimal] = {}
        self.equity_curve: list[tuple[int, Decimal]] = [(0, rules.initial_cash_usd)]
        self.realized_pnl_usd = ZERO
        self.total_fees_usd = ZERO
        self.closed_pnls: list[Decimal] = []

    def _advance(self, timestamp_unix: int) -> None:
        if timestamp_unix < self.current_time_unix:
            raise ValueError("Simulator input time regressed")
        self.current_time_unix = timestamp_unix
        for order in self.orders.values():
            if order.status in OPEN_ORDER_STATUSES and order.expires_at_unix < timestamp_unix:
                order.status = OrderStatus.EXPIRED

    def _reserved_cash(self) -> Decimal:
        return sum(
            (
                order.remaining_shares * order.limit_price * (ONE + self.rules.fee_rate)
                for order in self.orders.values()
                if order.status in OPEN_ORDER_STATUSES and order.side == OrderSide.BUY
            ),
            ZERO,
        )

    def _reserved_shares(self, token_id: str) -> Decimal:
        return sum(
            (
                order.remaining_shares
                for order in self.orders.values()
                if order.status in OPEN_ORDER_STATUSES
                and order.side == OrderSide.SELL
                and order.token_id == token_id
            ),
            ZERO,
        )

    def record_prediction(
        self,
        *,
        decision_id: str,
        timestamp_unix: int,
        action: str,
        probability: Decimal | float | str,
        size_fraction: Decimal | float | str,
        input_sha256: str,
    ) -> PredictionRecord:
        self._advance(timestamp_unix)
        if any(record.decision_id == decision_id for record in self.predictions):
            raise ValueError(f"Prediction already recorded: {decision_id}")
        probability_value = decimal(probability)
        size_value = decimal(size_fraction)
        if not ZERO <= probability_value <= ONE or not ZERO <= size_value <= ONE:
            raise ValueError("probability and size_fraction must be between zero and one")
        if len(input_sha256) != 64:
            raise ValueError("input_sha256 must be a SHA-256 hex digest")
        record = PredictionRecord(
            decision_id=decision_id,
            timestamp_unix=timestamp_unix,
            action=action,
            probability=probability_value,
            size_fraction=size_value,
            input_sha256=input_sha256,
        )
        self.predictions.append(record)
        return record

    def place_order(
        self,
        *,
        decision_id: str,
        component_id: str,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: OrderSide,
        submitted_at_unix: int,
        requested_shares: Decimal | float | str,
        limit_price: Decimal | float | str,
        evidence_sha256: str | None = None,
        evidence_liquidity_id: str | None = None,
    ) -> SimulatedOrder:
        self._advance(submitted_at_unix)
        shares = decimal(requested_shares)
        price = decimal(limit_price)
        if shares <= ZERO:
            raise ValueError("requested_shares must be positive")
        if not ZERO <= price <= ONE:
            raise ValueError("limit_price must be between zero and one")
        order_id = _stable_hash(
            {
                "decision_id": decision_id,
                "token_id": token_id,
                "side": side.value,
                "submitted_at_unix": submitted_at_unix,
            }
        )
        if order_id in self.orders:
            raise ValueError(f"Order already exists: {order_id}")
        order = SimulatedOrder(
            order_id=order_id,
            decision_id=decision_id,
            component_id=component_id,
            condition_id=condition_id,
            token_id=token_id,
            outcome=outcome,
            side=side,
            submitted_at_unix=submitted_at_unix,
            eligible_at_unix=submitted_at_unix + self.rules.latency_seconds,
            expires_at_unix=(
                submitted_at_unix
                + self.rules.latency_seconds
                + self.rules.maximum_fill_wait_seconds
            ),
            requested_shares=shares,
            limit_price=price,
            evidence_sha256=evidence_sha256,
            evidence_liquidity_id=evidence_liquidity_id,
        )
        if side == OrderSide.BUY:
            maximum_cost = shares * price * (ONE + self.rules.fee_rate)
            if maximum_cost > self.cash_usd - self._reserved_cash():
                order.status = OrderStatus.REJECTED
                order.reject_reason = "INSUFFICIENT_CASH"
        else:
            position = self.positions.get(token_id)
            available = (position.shares if position is not None else ZERO) - self._reserved_shares(
                token_id
            )
            if shares > available:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "INSUFFICIENT_SHARES"
        self.orders[order_id] = order
        return order

    def cancel_order(self, order_id: str, timestamp_unix: int) -> SimulatedOrder:
        self._advance(timestamp_unix)
        order = self.orders[order_id]
        if order.status in OPEN_ORDER_STATUSES:
            order.status = OrderStatus.CANCELLED
        return order

    def _side_is_eligible(self, order: SimulatedOrder, event: LiquidityEvent) -> bool:
        if not self.rules.opposing_side_required or event.observed_side is None:
            return True
        return event.observed_side != order.side

    def _execution_price(self, side: OrderSide, tape_price: Decimal) -> Decimal:
        adverse = self.rules.tick_size * self.rules.adverse_price_ticks
        if side == OrderSide.BUY:
            return min(ONE, tape_price + adverse)
        return max(ZERO, tape_price - adverse)

    def process_liquidity(self, event: LiquidityEvent) -> list[SimulatedFill]:
        self._advance(event.timestamp_unix)
        if event.liquidity_id in self.processed_liquidity_ids:
            raise ValueError(f"Liquidity event was replayed twice: {event.liquidity_id}")
        self.processed_liquidity_ids.add(event.liquidity_id)
        self.last_marks[event.token_id] = event.price
        available = (
            event.shares
            * self.rules.available_share_fraction
            * self.rules.duplicate_liquidity_haircut
        )
        emitted: list[SimulatedFill] = []
        candidates = sorted(
            (
                order
                for order in self.orders.values()
                if order.status in OPEN_ORDER_STATUSES
                and order.token_id == event.token_id
                and order.eligible_at_unix <= event.timestamp_unix <= order.expires_at_unix
                and order.evidence_liquidity_id != event.liquidity_id
                and self._side_is_eligible(order, event)
            ),
            key=lambda order: (order.submitted_at_unix, order.order_id),
        )
        for order in candidates:
            if available <= ZERO:
                break
            price = self._execution_price(order.side, event.price)
            if order.side == OrderSide.BUY and price > order.limit_price:
                continue
            if order.side == OrderSide.SELL and price < order.limit_price:
                continue
            shares = min(order.remaining_shares, available)
            if order.side == OrderSide.BUY:
                unit_cost = price * (ONE + self.rules.fee_rate)
                shares = min(shares, self.cash_usd / unit_cost if unit_cost else ZERO)
            else:
                position = self.positions.get(order.token_id)
                shares = min(shares, position.shares if position is not None else ZERO)
            if shares <= ZERO:
                continue
            notional = shares * price
            fee = notional * self.rules.fee_rate
            fill_id = _stable_hash(
                {
                    "order_id": order.order_id,
                    "liquidity_id": event.liquidity_id,
                    "fill_ordinal": len(self.fills),
                    "shares": str(shares),
                    "price": str(price),
                }
            )
            fill = SimulatedFill(
                fill_id=fill_id,
                order_id=order.order_id,
                liquidity_id=event.liquidity_id,
                timestamp_unix=event.timestamp_unix,
                side=order.side,
                shares=shares,
                price=price,
                notional_usd=notional,
                fee_usd=fee,
            )
            self._apply_fill(order, fill)
            emitted.append(fill)
            available -= shares
        self._record_equity(event.timestamp_unix)
        return emitted

    def _apply_fill(self, order: SimulatedOrder, fill: SimulatedFill) -> None:
        if fill.side == OrderSide.BUY:
            total_cost = fill.notional_usd + fill.fee_usd
            if total_cost > self.cash_usd:
                raise RuntimeError("Fill would spend unavailable cash")
            self.cash_usd -= total_cost
            buy_position = self.positions.setdefault(
                order.token_id,
                SimulatedPosition(order.condition_id, order.token_id, order.outcome),
            )
            buy_position.shares += fill.shares
            buy_position.cost_basis_usd += total_cost
        else:
            sell_position = self.positions.get(order.token_id)
            if sell_position is None or fill.shares > sell_position.shares:
                raise RuntimeError("Fill would sell unavailable shares")
            allocated_cost = sell_position.cost_basis_usd * fill.shares / sell_position.shares
            proceeds = fill.notional_usd - fill.fee_usd
            pnl = proceeds - allocated_cost
            sell_position.shares -= fill.shares
            sell_position.cost_basis_usd -= allocated_cost
            self.cash_usd += proceeds
            self.realized_pnl_usd += pnl
            self.closed_pnls.append(pnl)
            if sell_position.shares == ZERO:
                del self.positions[order.token_id]
        self.total_fees_usd += fill.fee_usd
        self.fills.append(fill)
        order.filled_shares += fill.shares
        order.status = (
            OrderStatus.FILLED
            if order.filled_shares == order.requested_shares
            else OrderStatus.PARTIAL
        )

    def resolve(
        self,
        *,
        condition_id: str,
        timestamp_unix: int,
        token_payouts: dict[str, Decimal | float | int | str],
    ) -> Decimal:
        self._advance(timestamp_unix)
        for order in self.orders.values():
            if order.condition_id == condition_id and order.status in OPEN_ORDER_STATUSES:
                order.status = OrderStatus.CANCELLED
        resolution_pnl = ZERO
        for token_id, position in list(self.positions.items()):
            if position.condition_id != condition_id:
                continue
            payout_rate = decimal(token_payouts.get(token_id, ZERO))
            if not ZERO <= payout_rate <= ONE:
                raise ValueError("Terminal payout must be between zero and one")
            payout = position.shares * payout_rate
            pnl = payout - position.cost_basis_usd
            self.cash_usd += payout
            self.realized_pnl_usd += pnl
            self.closed_pnls.append(pnl)
            resolution_pnl += pnl
            del self.positions[token_id]
        self._record_equity(timestamp_unix)
        return resolution_pnl

    def _equity(self) -> Decimal:
        marked_positions = sum(
            (
                position.shares * self.last_marks.get(position.token_id, position.average_price)
                for position in self.positions.values()
            ),
            ZERO,
        )
        return self.cash_usd + marked_positions

    def _record_equity(self, timestamp_unix: int) -> None:
        equity = self._equity()
        if self.equity_curve and self.equity_curve[-1][0] == timestamp_unix:
            self.equity_curve[-1] = (timestamp_unix, equity)
        else:
            self.equity_curve.append((timestamp_unix, equity))

    def metrics(self) -> dict[str, Any]:
        equity = self._equity()
        peak = self.equity_curve[0][1]
        maximum_drawdown = ZERO
        for _, value in self.equity_curve:
            peak = max(peak, value)
            if peak:
                maximum_drawdown = max(maximum_drawdown, (peak - value) / peak)
        gains = sum((value for value in self.closed_pnls if value > ZERO), ZERO)
        losses = -sum((value for value in self.closed_pnls if value < ZERO), ZERO)
        status_counts = {
            status.value: sum(order.status == status for order in self.orders.values())
            for status in OrderStatus
        }
        return {
            "initial_cash_usd": float(self.rules.initial_cash_usd),
            "cash_usd": float(self.cash_usd),
            "equity_usd": float(equity),
            "net_profit_usd": float(equity - self.rules.initial_cash_usd),
            "return_on_initial_cash": float(
                (equity - self.rules.initial_cash_usd) / self.rules.initial_cash_usd
            ),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "total_fees_usd": float(self.total_fees_usd),
            "profit_factor": float(gains / losses) if losses else None,
            "maximum_drawdown": float(maximum_drawdown),
            "orders": len(self.orders),
            "fills": len(self.fills),
            "predictions": len(self.predictions),
            "open_positions": len(self.positions),
            "order_status_counts": status_counts,
        }

    def snapshot(self) -> dict[str, Any]:
        rules = _decimal_dict(asdict(self.rules))
        orders = [_decimal_dict(asdict(order)) for order in self.orders.values()]
        positions = [_decimal_dict(asdict(position)) for position in self.positions.values()]
        fills = [_decimal_dict(asdict(fill)) for fill in self.fills]
        predictions = [_decimal_dict(asdict(record)) for record in self.predictions]
        return {
            "schema_version": "1.0.0",
            "record_type": "simulator_checkpoint",
            "rules": rules,
            "current_time_unix": self.current_time_unix,
            "cash_usd": str(self.cash_usd),
            "orders": orders,
            "positions": positions,
            "fills": fills,
            "predictions": predictions,
            "processed_liquidity_ids": sorted(self.processed_liquidity_ids),
            "last_marks": {key: str(value) for key, value in sorted(self.last_marks.items())},
            "equity_curve": [[timestamp, str(value)] for timestamp, value in self.equity_curve],
            "realized_pnl_usd": str(self.realized_pnl_usd),
            "total_fees_usd": str(self.total_fees_usd),
            "closed_pnls": [str(value) for value in self.closed_pnls],
        }

    def checkpoint_sha256(self) -> str:
        return _stable_hash(self.snapshot())

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> ReplaySimulator:
        rules_value = dict(payload["rules"])
        for key in (
            "initial_cash_usd",
            "available_share_fraction",
            "duplicate_liquidity_haircut",
            "tick_size",
            "fee_bps",
        ):
            rules_value[key] = decimal(rules_value[key])
        simulator = cls(SimulationRules(**rules_value))
        simulator.current_time_unix = int(payload["current_time_unix"])
        simulator.cash_usd = decimal(payload["cash_usd"])
        simulator.orders = {}
        for row_value in payload["orders"]:
            row = dict(row_value)
            row["side"] = OrderSide(row["side"])
            row["status"] = OrderStatus(row["status"])
            row["requested_shares"] = decimal(row["requested_shares"])
            row["limit_price"] = decimal(row["limit_price"])
            row["filled_shares"] = decimal(row["filled_shares"])
            order = SimulatedOrder(**row)
            simulator.orders[order.order_id] = order
        simulator.positions = {}
        for row_value in payload["positions"]:
            row = dict(row_value)
            row["shares"] = decimal(row["shares"])
            row["cost_basis_usd"] = decimal(row["cost_basis_usd"])
            position = SimulatedPosition(**row)
            simulator.positions[position.token_id] = position
        simulator.fills = []
        for row_value in payload["fills"]:
            row = dict(row_value)
            row["side"] = OrderSide(row["side"])
            for key in ("shares", "price", "notional_usd", "fee_usd"):
                row[key] = decimal(row[key])
            simulator.fills.append(SimulatedFill(**row))
        simulator.predictions = []
        for row_value in payload["predictions"]:
            row = dict(row_value)
            row["probability"] = decimal(row["probability"])
            row["size_fraction"] = decimal(row["size_fraction"])
            simulator.predictions.append(PredictionRecord(**row))
        simulator.processed_liquidity_ids = set(payload["processed_liquidity_ids"])
        simulator.last_marks = {key: decimal(value) for key, value in payload["last_marks"].items()}
        simulator.equity_curve = [
            (int(timestamp), decimal(value)) for timestamp, value in payload["equity_curve"]
        ]
        simulator.realized_pnl_usd = decimal(payload["realized_pnl_usd"])
        simulator.total_fees_usd = decimal(payload["total_fees_usd"])
        simulator.closed_pnls = [decimal(value) for value in payload["closed_pnls"]]
        return simulator
