"""Causal H009-to-H010 replay adapter for selective Polymarket policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sphinx_trace.model_h012 import H012_ACTIONS
from sphinx_trace.polymarket_fees import FeeScheduleBook
from sphinx_trace.protocol_tail_pack import winning_payout_per_total_cost
from sphinx_trace.simulator import (
    ONE,
    ZERO,
    LiquidityEvent,
    OrderSide,
    ReplaySimulator,
    SimulatedOrder,
    SimulatedPosition,
    decimal,
)


class SelectiveAction(StrEnum):
    CALL_OUTCOME_0 = "CALL_OUTCOME_0"
    CALL_OUTCOME_1 = "CALL_OUTCOME_1"
    SKIP = "SKIP"
    UPDATE = "UPDATE"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"


if (
    tuple(action.value for action in SelectiveAction) != H012_ACTIONS
):  # pragma: no cover
    raise RuntimeError("H010 replay actions must match the H012 model contract")


@dataclass(frozen=True, slots=True)
class BinaryMarketContract:
    condition_id: str
    component_id: str
    outcomes: tuple[str, str]
    token_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if not self.condition_id or not self.component_id:
            raise ValueError("Market contract identities are required")
        if len(set(self.token_ids)) != 2 or not all(self.token_ids):
            raise ValueError("Binary market contract requires two distinct tokens")

    def outcome_index(self, token_id: str) -> int:
        try:
            return self.token_ids.index(token_id)
        except ValueError as error:
            raise ValueError(
                f"Token is not part of market {self.condition_id}"
            ) from error


@dataclass(frozen=True, slots=True)
class PolicyCall:
    decision_id: str
    timestamp_unix: int
    condition_id: str
    component_id: str
    evidence_trade_id: str
    action: SelectiveAction
    probability_outcome0: Decimal
    size_fraction: Decimal
    input_sha256: str

    def __post_init__(self) -> None:
        if not self.decision_id or not self.evidence_trade_id:
            raise ValueError("Policy call identities are required")
        if self.timestamp_unix < 0:
            raise ValueError("Policy call timestamp cannot be negative")
        if not ZERO <= self.probability_outcome0 <= ONE:
            raise ValueError("Policy probability must be between zero and one")
        if not ZERO <= self.size_fraction <= ONE:
            raise ValueError("Policy size must be between zero and one")
        if len(self.input_sha256) != 64:
            raise ValueError("Policy input must have a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ReplayCursor:
    source_sha256: str
    shard_ordinal: int = -1
    row_ordinal: int = -1

    def advance(self, shard_ordinal: int, row_ordinal: int) -> ReplayCursor:
        if shard_ordinal < 0 or row_ordinal < 0:
            raise ValueError("Replay cursor ordinals must be non-negative")
        if (shard_ordinal, row_ordinal) <= (self.shard_ordinal, self.row_ordinal):
            raise ValueError("Replay cursor did not advance")
        return ReplayCursor(self.source_sha256, shard_ordinal, row_ordinal)


@dataclass(frozen=True, slots=True)
class PredictionMemory:
    action: SelectiveAction
    probability_outcome0: Decimal
    size_fraction: Decimal
    timestamp_unix: int
    token_id: str | None


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class H010ReplayAdapter:
    """Apply model calls only after their exact evidence trade has been observed."""

    def __init__(
        self,
        simulator: ReplaySimulator,
        contracts: dict[str, BinaryMarketContract],
        *,
        source_sha256: str,
    ) -> None:
        if len(source_sha256) != 64:
            raise ValueError("Replay source requires a SHA-256 digest")
        self.simulator = simulator
        self.contracts = contracts
        self.cursor = ReplayCursor(source_sha256)
        self.prediction_memory: dict[str, PredictionMemory] = {}
        self.resolved_conditions: set[str] = set()
        self.source_price_anomalies = 0

    def process_trade(
        self,
        payload: dict[str, Any],
        calls: tuple[PolicyCall, ...] = (),
        *,
        shard_ordinal: int,
        row_ordinal: int,
    ) -> list[SimulatedOrder]:
        """Consume liquidity first, then calls whose features include that trade."""

        reference_prices = self.observe_trade(
            payload,
            shard_ordinal=shard_ordinal,
            row_ordinal=row_ordinal,
        )
        trade_id = str(payload["trade_id"])
        condition_id = str(payload["condition_id"]).lower()
        timestamp = int(payload["timestamp_unix"])
        orders: list[SimulatedOrder] = []
        for call in calls:
            if call.evidence_trade_id != trade_id:
                raise ValueError(
                    "Policy call does not match the current evidence trade"
                )
            if call.timestamp_unix != timestamp or call.condition_id != condition_id:
                raise ValueError(
                    "Policy call time or market does not match its evidence"
                )
            orders.extend(self.apply_call(call, reference_prices))
        return orders

    def observe_trade(
        self,
        payload: dict[str, Any],
        *,
        shard_ordinal: int,
        row_ordinal: int,
    ) -> dict[str, Decimal]:
        """Advance public liquidity and return causal binary reference prices."""

        self.cursor = self.cursor.advance(shard_ordinal, row_ordinal)
        trade_id = str(payload["trade_id"])
        condition_id = str(payload["condition_id"]).lower()
        token_id = str(payload["token_id"])
        timestamp = int(payload["timestamp_unix"])
        raw_price = decimal(payload["price"])
        if not raw_price.is_finite():
            raise ValueError("Replay trade price must be finite")
        if not ZERO <= raw_price <= ONE:
            self.source_price_anomalies += 1
        price = min(ONE, max(ZERO, raw_price))
        shares = decimal(payload["size"])
        side_value = str(payload.get("side") or "").upper()
        observed_side = OrderSide(side_value) if side_value in {"BUY", "SELL"} else None
        contract = self.contracts.get(condition_id)
        if contract is None:
            raise ValueError(f"Replay trade has no market contract: {condition_id}")
        outcome_index = contract.outcome_index(token_id)
        self.simulator.process_liquidity(
            LiquidityEvent(
                liquidity_id=trade_id,
                timestamp_unix=timestamp,
                condition_id=condition_id,
                token_id=token_id,
                outcome=contract.outcomes[outcome_index],
                price=price,
                shares=shares,
                observed_side=observed_side,
                transaction_hash=(
                    None
                    if payload.get("transaction_hash") is None
                    else str(payload["transaction_hash"]).lower()
                ),
            )
        )
        reference_prices = {
            token_id: price,
            contract.token_ids[1 - outcome_index]: ONE - price,
        }
        return reference_prices

    def apply_call(
        self,
        call: PolicyCall,
        reference_prices: dict[str, Decimal],
    ) -> list[SimulatedOrder]:
        contract = self.contracts[call.condition_id]
        if contract.component_id != call.component_id:
            raise ValueError("Policy call component does not match the market contract")
        if call.condition_id in self.resolved_conditions:
            raise ValueError("Policy call targeted an already resolved market")
        self.simulator.record_prediction(
            decision_id=call.decision_id,
            timestamp_unix=call.timestamp_unix,
            action=call.action.value,
            probability=call.probability_outcome0,
            size_fraction=call.size_fraction,
            input_sha256=call.input_sha256,
        )
        previous = self.prediction_memory.get(call.condition_id)
        token_id = self._selected_token(call.action, contract, previous)
        self.prediction_memory[call.condition_id] = PredictionMemory(
            call.action,
            call.probability_outcome0,
            call.size_fraction,
            call.timestamp_unix,
            token_id,
        )
        if call.action in {SelectiveAction.SKIP, SelectiveAction.HOLD}:
            return []
        if call.action in {
            SelectiveAction.CALL_OUTCOME_0,
            SelectiveAction.CALL_OUTCOME_1,
        }:
            assert token_id is not None
            self._cancel_pending(call.condition_id, call.timestamp_unix)
            order = self._buy_to_fraction(
                call, contract, token_id, reference_prices[token_id]
            )
            return [] if order is None else [order]
        if call.action == SelectiveAction.UPDATE:
            self._cancel_pending(call.condition_id, call.timestamp_unix)
            if token_id is None:
                return []
            return self._update_position(
                call, contract, token_id, reference_prices[token_id]
            )
        if call.action in {SelectiveAction.REDUCE, SelectiveAction.CLOSE}:
            return self._sell_positions(call, contract, reference_prices)
        raise RuntimeError(f"Unsupported H012 action: {call.action}")

    def candidate_execution_context(
        self,
        condition_id: str,
        timestamp_unix: int,
        evidence_trade_id: str,
        reference_prices: dict[str, Decimal],
        *,
        reference_total_cost_usd: Decimal = Decimal("500"),
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Quote both causal BUY limits and winning payouts used by H021."""

        contract = self.contracts[condition_id]
        if reference_total_cost_usd <= ZERO:
            raise ValueError("H021 payout reference cost must be positive")
        adverse = (
            self.simulator.rules.tick_size * self.simulator.rules.adverse_price_ticks
        )
        prices = tuple(
            min(ONE, reference_prices[token_id] + adverse)
            for token_id in contract.token_ids
        )
        if any(price <= ZERO for price in prices):
            raise ValueError("H021 candidate entry prices must be positive")
        schedule_book = self.simulator.fee_schedule_book
        if schedule_book is None:
            payouts = tuple(
                ONE / (price * (ONE + self.simulator.rules.fee_rate))
                for price in prices
            )
        else:
            schedule = schedule_book.schedule_for(
                evidence_trade_id,
                condition_id=condition_id,
                timestamp_unix=timestamp_unix,
            )
            payouts = tuple(
                winning_payout_per_total_cost(
                    schedule,
                    entry_price=price,
                    total_cost_usd=reference_total_cost_usd,
                    rate_multiplier=self.simulator.rules.fee_rate_multiplier,
                )
                for price in prices
            )
        return prices[0], prices[1], payouts[0], payouts[1]

    def _selected_token(
        self,
        action: SelectiveAction,
        contract: BinaryMarketContract,
        previous: PredictionMemory | None,
    ) -> str | None:
        if action == SelectiveAction.CALL_OUTCOME_0:
            return contract.token_ids[0]
        if action == SelectiveAction.CALL_OUTCOME_1:
            return contract.token_ids[1]
        if previous is not None and previous.token_id in contract.token_ids:
            return previous.token_id
        positions = self._condition_positions(contract.condition_id)
        return (
            max(positions, key=lambda row: row.cost_basis_usd).token_id
            if positions
            else None
        )

    def _buy_to_fraction(
        self,
        call: PolicyCall,
        contract: BinaryMarketContract,
        token_id: str,
        reference_price: Decimal,
    ) -> SimulatedOrder | None:
        adverse = (
            self.simulator.rules.tick_size * self.simulator.rules.adverse_price_ticks
        )
        limit_price = min(ONE, reference_price + adverse)
        if limit_price <= ZERO:
            return None
        target_cost = self.simulator.equity_usd() * call.size_fraction
        current = self.simulator.positions.get(token_id)
        current_cost = current.cost_basis_usd if current is not None else ZERO
        additional_cost = min(
            max(ZERO, target_cost - current_cost),
            self.simulator.available_cash_usd(),
        )
        requested_shares = self.simulator.buy_shares_for_total_cost(
            total_cost_usd=additional_cost,
            price=limit_price,
            evidence_liquidity_id=call.evidence_trade_id,
            condition_id=call.condition_id,
            timestamp_unix=call.timestamp_unix,
        )
        if requested_shares <= ZERO:
            return None
        index = contract.outcome_index(token_id)
        return self.simulator.place_order(
            decision_id=call.decision_id,
            component_id=call.component_id,
            condition_id=call.condition_id,
            token_id=token_id,
            outcome=contract.outcomes[index],
            side=OrderSide.BUY,
            submitted_at_unix=call.timestamp_unix,
            requested_shares=requested_shares,
            limit_price=limit_price,
            evidence_sha256=call.input_sha256,
            evidence_liquidity_id=call.evidence_trade_id,
        )

    def _update_position(
        self,
        call: PolicyCall,
        contract: BinaryMarketContract,
        token_id: str,
        reference_price: Decimal,
    ) -> list[SimulatedOrder]:
        position = self.simulator.positions.get(token_id)
        target_cost = self.simulator.equity_usd() * call.size_fraction
        if position is None or position.cost_basis_usd < target_cost:
            order = self._buy_to_fraction(call, contract, token_id, reference_price)
            return [] if order is None else [order]
        if position.cost_basis_usd == ZERO:
            return []
        fraction = min(ONE, max(ZERO, ONE - target_cost / position.cost_basis_usd))
        order = self._sell_one(call, contract, position, fraction, reference_price)
        return [] if order is None else [order]

    def _sell_positions(
        self,
        call: PolicyCall,
        contract: BinaryMarketContract,
        reference_prices: dict[str, Decimal],
    ) -> list[SimulatedOrder]:
        fraction = ONE if call.action == SelectiveAction.CLOSE else call.size_fraction
        orders: list[SimulatedOrder] = []
        self._cancel_pending(call.condition_id, call.timestamp_unix)
        for position in self._condition_positions(call.condition_id):
            reference = reference_prices.get(
                position.token_id,
                self.simulator.last_marks.get(
                    position.token_id, position.average_price
                ),
            )
            order = self._sell_one(call, contract, position, fraction, reference)
            if order is not None:
                orders.append(order)
        return orders

    def _sell_one(
        self,
        call: PolicyCall,
        contract: BinaryMarketContract,
        position: SimulatedPosition,
        fraction: Decimal,
        reference_price: Decimal,
    ) -> SimulatedOrder | None:
        shares = position.shares * min(ONE, max(ZERO, fraction))
        if shares <= ZERO:
            return None
        adverse = (
            self.simulator.rules.tick_size * self.simulator.rules.adverse_price_ticks
        )
        limit_price = max(ZERO, reference_price - adverse)
        return self.simulator.place_order(
            decision_id=call.decision_id,
            component_id=call.component_id,
            condition_id=call.condition_id,
            token_id=position.token_id,
            outcome=contract.outcomes[contract.outcome_index(position.token_id)],
            side=OrderSide.SELL,
            submitted_at_unix=call.timestamp_unix,
            requested_shares=shares,
            limit_price=limit_price,
            evidence_sha256=call.input_sha256,
            evidence_liquidity_id=call.evidence_trade_id,
        )

    def _cancel_pending(self, condition_id: str, timestamp_unix: int) -> None:
        for order in self.simulator.open_orders_for_condition(condition_id):
            self.simulator.cancel_order(order.order_id, timestamp_unix)

    def _condition_positions(self, condition_id: str) -> list[SimulatedPosition]:
        return list(self.simulator.positions_for_condition(condition_id))

    def physical_action_mask(self, condition_id: str) -> tuple[bool, ...]:
        if condition_id in self.resolved_conditions:
            return (False, False, True, False, True, False, False)
        positions = self._condition_positions(condition_id)
        has_position = bool(positions)
        has_memory = condition_id in self.prediction_memory
        can_buy = self.simulator.available_cash_usd() > ZERO
        return (
            can_buy,
            can_buy,
            True,
            has_memory or has_position,
            True,
            has_position,
            has_position,
        )

    def portfolio_features(self) -> tuple[float, ...]:
        initial = self.simulator.rules.initial_cash_usd
        equity = self.simulator.equity_usd()
        exposure = self.simulator.marked_exposure_usd()
        cost_basis = self.simulator.total_cost_basis_usd()
        peak = self.simulator.peak_equity_usd()
        drawdown = (peak - equity) / peak if peak > ZERO else ZERO
        pending = self.simulator.pending_order_count()
        return (
            float(self.simulator.cash_usd / initial),
            float(equity / initial),
            float(cost_basis / initial),
            float(exposure / initial),
            float((exposure - cost_basis) / initial),
            float(self.simulator.realized_pnl_usd / initial),
            float(drawdown),
            math.log1p(len(self.simulator.positions)),
            math.log1p(pending),
        )

    def prediction_memory_features(
        self,
        condition_id: str,
        timestamp_unix: int,
    ) -> tuple[int, tuple[float, ...]]:
        memory = self.prediction_memory.get(condition_id)
        if memory is None:
            return H012_ACTIONS.index(SelectiveAction.SKIP.value), (
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        position = (
            self.simulator.positions.get(memory.token_id)
            if memory.token_id is not None
            else None
        )
        equity = self.simulator.equity_usd()
        mark = (
            self.simulator.last_marks.get(position.token_id, position.average_price)
            if position is not None
            else ZERO
        )
        fraction = (
            position.cost_basis_usd / equity
            if position is not None and equity > ZERO
            else ZERO
        )
        elapsed = max(0, timestamp_unix - memory.timestamp_unix)
        action_id = H012_ACTIONS.index(memory.action.value)
        return action_id, (
            float(memory.probability_outcome0),
            float(memory.size_fraction),
            math.log1p(elapsed),
            float(position is not None),
            float(fraction),
            float(position.average_price if position is not None else ZERO),
            float(mark),
        )

    def resolve(
        self,
        condition_id: str,
        timestamp_unix: int,
        payouts: tuple[Decimal, Decimal],
    ) -> Decimal:
        contract = self.contracts[condition_id]
        pnl = self.simulator.resolve(
            condition_id=condition_id,
            timestamp_unix=timestamp_unix,
            token_payouts=dict(zip(contract.token_ids, payouts, strict=True)),
        )
        self.prediction_memory.pop(condition_id, None)
        self.resolved_conditions.add(condition_id)
        return pnl

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "record_type": "h010_replay_adapter_checkpoint",
            "cursor": asdict(self.cursor),
            "simulator": self.simulator.snapshot(),
            "prediction_memory": {
                key: {
                    **asdict(value),
                    "action": value.action.value,
                    "probability_outcome0": str(value.probability_outcome0),
                    "size_fraction": str(value.size_fraction),
                }
                for key, value in sorted(self.prediction_memory.items())
            },
            "resolved_conditions": sorted(self.resolved_conditions),
            "source_price_anomalies": self.source_price_anomalies,
        }

    def checkpoint_sha256(self) -> str:
        return _stable_hash(self.snapshot())

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        contracts: dict[str, BinaryMarketContract],
        *,
        source_sha256: str,
        fee_schedule_book: FeeScheduleBook | None = None,
    ) -> H010ReplayAdapter:
        cursor_value = dict(payload["cursor"])
        if str(cursor_value["source_sha256"]) != source_sha256:
            raise ValueError("Replay checkpoint belongs to another source")
        adapter = cls(
            ReplaySimulator.from_snapshot(
                dict(payload["simulator"]),
                fee_schedule_book=fee_schedule_book,
            ),
            contracts,
            source_sha256=source_sha256,
        )
        adapter.cursor = ReplayCursor(
            source_sha256,
            int(cursor_value["shard_ordinal"]),
            int(cursor_value["row_ordinal"]),
        )
        memory_value = payload.get("prediction_memory", {})
        if not isinstance(memory_value, dict):
            raise TypeError("Replay prediction memory must be an object")
        for condition_id, row_value in memory_value.items():
            if not isinstance(row_value, dict):
                raise TypeError("Replay prediction-memory row must be an object")
            adapter.prediction_memory[str(condition_id)] = PredictionMemory(
                action=SelectiveAction(str(row_value["action"])),
                probability_outcome0=decimal(row_value["probability_outcome0"]),
                size_fraction=decimal(row_value["size_fraction"]),
                timestamp_unix=int(row_value["timestamp_unix"]),
                token_id=(
                    None
                    if row_value.get("token_id") is None
                    else str(row_value["token_id"])
                ),
            )
        adapter.resolved_conditions = {
            str(value) for value in payload.get("resolved_conditions", [])
        }
        adapter.source_price_anomalies = int(payload.get("source_price_anomalies", 0))
        return adapter
