"""Execution-conditioned targets for aggregated exact replay policy states."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import sha256_file
from sphinx_trace.model_h012 import H012_ACTION_COUNT, H012_ACTIONS
from sphinx_trace.replay_state_pack import STATE_ARRAY_NAMES

H015_ARRAY_NAMES = (
    *STATE_ARRAY_NAMES,
    "market_ids.npy",
    "behavior_policy_codes.npy",
    "behavior_action_ids.npy",
    "realized_action_values.npy",
    "realized_conditional_log_utilities.npy",
    "realized_pnl_usd.npy",
    "executed_cost_usd.npy",
    "execution_fractions.npy",
)

_ACTION_IDS = {name: index for index, name in enumerate(H012_ACTIONS[:3])}
_ZERO = Decimal(0)
_ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class LoggedExecutionTarget:
    action_id: int
    realized_action_value: float
    conditional_log_utility: float
    realized_pnl_usd: float
    executed_cost_usd: float
    execution_fraction: float


@dataclass(frozen=True, slots=True)
class LoggedExecutionIndex:
    targets: dict[tuple[str, int], LoggedExecutionTarget]
    action_counts: dict[str, int]
    orders: int
    fills: int
    filled_decisions: int
    requested_shares: Decimal
    filled_shares: Decimal
    executed_cost_usd: Decimal
    realized_pnl_usd: Decimal


@dataclass(frozen=True, slots=True)
class _Decision:
    feature_date: str
    feature_row: int
    action: str
    condition_id: str


@dataclass(frozen=True, slots=True)
class _Order:
    decision_id: str
    condition_id: str
    token_id: str
    requested_shares: Decimal


@dataclass(slots=True)
class _Execution:
    requested_shares: Decimal = _ZERO
    filled_shares: Decimal = _ZERO
    cost_usd: Decimal = _ZERO
    pnl_usd: Decimal = _ZERO
    fills: int = 0


def build_payout_map(
    contracts: Mapping[str, Any],
    resolutions: Iterable[Any],
) -> dict[str, dict[str, Decimal]]:
    """Bind terminal payouts to the token identifiers used by replay fills."""

    output: dict[str, dict[str, Decimal]] = {}
    for resolution in resolutions:
        condition_id = str(resolution.condition_id).lower()
        contract = contracts.get(condition_id)
        if contract is None:
            raise RuntimeError(f"H015 resolution has no binary contract: {condition_id}")
        tokens = tuple(str(value) for value in contract.token_ids)
        payouts = tuple(Decimal(str(value)) for value in resolution.payouts)
        if len(tokens) != 2 or len(payouts) != 2:
            raise RuntimeError(f"H015 condition is not binary: {condition_id}")
        output[condition_id] = dict(zip(tokens, payouts, strict=True))
    if set(output) != {str(value).lower() for value in contracts}:
        raise RuntimeError("H015 payout map does not cover every replay contract")
    return output


def build_logged_execution_index(
    records: Iterable[dict[str, Any]],
    payouts_by_condition: Mapping[str, Mapping[str, Decimal]],
    *,
    reference_size: float,
) -> LoggedExecutionIndex:
    """Attribute each exact BUY fill and terminal payout to its source decision."""

    if not 0.0 < reference_size < 1.0:
        raise ValueError("H015 reference size must be between zero and one")
    reference = Decimal(str(reference_size))
    decisions: dict[str, _Decision] = {}
    feature_keys: set[tuple[str, int]] = set()
    orders: dict[str, _Order] = {}
    executions: dict[str, _Execution] = defaultdict(_Execution)
    action_counts: Counter[str] = Counter()
    fill_count = 0
    observed_resolutions: set[str] = set()
    for record in records:
        record_type = str(record.get("record_type") or "")
        if record_type == "h010_decision_audit":
            decision_id = str(record.get("decision_id") or "")
            action = str(record.get("action") or "")
            feature_ref = record.get("feature_ref")
            if not decision_id or decision_id in decisions:
                raise RuntimeError(f"H015 replay decision repeats: {decision_id}")
            if action not in _ACTION_IDS:
                raise RuntimeError(f"H015 cannot train a non-initial action: {action}")
            if not isinstance(feature_ref, dict):
                raise RuntimeError("H015 replay decision has no feature reference")
            feature_date = str(feature_ref.get("date") or "")
            feature_row = int(feature_ref.get("row", -1))
            feature_key = (feature_date, feature_row)
            if not feature_date or feature_row < 0 or feature_key in feature_keys:
                raise RuntimeError(f"H015 replay feature reference repeats: {feature_key}")
            condition_id = str(record.get("condition_id") or "").lower()
            if condition_id not in payouts_by_condition:
                raise RuntimeError(f"H015 decision condition is outside the tape: {condition_id}")
            decisions[decision_id] = _Decision(
                feature_date, feature_row, action, condition_id
            )
            feature_keys.add(feature_key)
            action_counts[action] += 1
        elif record_type == "h010_order_audit":
            order_id = str(record.get("order_id") or "")
            decision_id = str(record.get("decision_id") or "")
            condition_id = str(record.get("condition_id") or "").lower()
            token_id = str(record.get("token_id") or "")
            side = str(record.get("side") or "")
            requested = Decimal(str(record.get("requested_shares")))
            if not order_id or order_id in orders:
                raise RuntimeError(f"H015 replay order repeats: {order_id}")
            if decision_id not in decisions:
                raise RuntimeError(f"H015 order precedes or loses its decision: {order_id}")
            if side != "BUY":
                raise RuntimeError(f"H015 v1 cannot attribute a {side} order: {order_id}")
            if requested <= _ZERO:
                raise RuntimeError(f"H015 order has invalid requested shares: {order_id}")
            decision = decisions[decision_id]
            if decision.action == "SKIP" or decision.condition_id != condition_id:
                raise RuntimeError(f"H015 order contradicts its source decision: {order_id}")
            payouts = payouts_by_condition.get(condition_id)
            if payouts is None or token_id not in payouts:
                raise RuntimeError(f"H015 order token has no terminal payout: {order_id}")
            orders[order_id] = _Order(decision_id, condition_id, token_id, requested)
            executions[decision_id].requested_shares += requested
        elif record_type == "h010_fill_audit":
            order_id = str(record.get("order_id") or "")
            order = orders.get(order_id)
            if order is None:
                raise RuntimeError(f"H015 fill has no preceding order: {order_id}")
            if str(record.get("side") or "") != "BUY":
                raise RuntimeError(f"H015 v1 cannot attribute a SELL fill: {order_id}")
            shares = Decimal(str(record.get("shares")))
            notional = Decimal(str(record.get("notional_usd")))
            fee = Decimal(str(record.get("fee_usd")))
            position_shares = Decimal(str(record.get("position_shares", shares)))
            collateral_fee = Decimal(
                str(record.get("collateral_fee_usd", fee))
            )
            if (
                shares <= _ZERO
                or notional < _ZERO
                or fee < _ZERO
                or not _ZERO <= position_shares <= shares
                or collateral_fee < _ZERO
            ):
                raise RuntimeError(f"H015 fill contains invalid economics: {order_id}")
            payout = payouts_by_condition[order.condition_id][order.token_id]
            execution = executions[order.decision_id]
            execution.filled_shares += shares
            execution.cost_usd += notional + collateral_fee
            execution.pnl_usd += position_shares * payout - notional - collateral_fee
            execution.fills += 1
            fill_count += 1
        elif record_type == "h010_resolution_audit":
            condition_id = str(record.get("condition_id") or "").lower()
            expected = payouts_by_condition.get(condition_id)
            observed = tuple(Decimal(str(value)) for value in record.get("payouts", ()))
            if expected is None or observed != tuple(expected.values()):
                raise RuntimeError(f"H015 audit payout changed: {condition_id}")
            observed_resolutions.add(condition_id)

    if not decisions:
        raise RuntimeError("H015 replay contains no decisions")
    unresolved = {decision.condition_id for decision in decisions.values()} - observed_resolutions
    if unresolved:
        raise RuntimeError(
            f"H015 decisions are missing terminal audit payouts: {min(unresolved)}"
        )
    targets: dict[tuple[str, int], LoggedExecutionTarget] = {}
    total_requested = _ZERO
    total_filled = _ZERO
    total_cost = _ZERO
    total_pnl = _ZERO
    filled_decisions = 0
    for decision_id, decision in decisions.items():
        execution = executions.get(decision_id, _Execution())
        if execution.filled_shares > execution.requested_shares:
            raise RuntimeError(f"H015 filled more shares than requested: {decision_id}")
        if decision.action == "SKIP" and (
            execution.requested_shares != _ZERO or execution.filled_shares != _ZERO
        ):
            raise RuntimeError(f"H015 SKIP generated execution: {decision_id}")
        fraction = (
            min(_ONE, execution.filled_shares / execution.requested_shares)
            if execution.requested_shares > _ZERO
            else _ZERO
        )
        conditional = 0.0
        action_value = 0.0
        if execution.cost_usd > _ZERO:
            wealth = _ONE + reference * execution.pnl_usd / execution.cost_usd
            conditional = math.log(max(float(wealth), 1e-8))
            action_value = float(fraction) * conditional
            filled_decisions += 1
        elif execution.pnl_usd != _ZERO or execution.filled_shares != _ZERO:
            raise RuntimeError(f"H015 nonzero execution has no cost: {decision_id}")
        target = LoggedExecutionTarget(
            action_id=_ACTION_IDS[decision.action],
            realized_action_value=action_value,
            conditional_log_utility=conditional,
            realized_pnl_usd=float(execution.pnl_usd),
            executed_cost_usd=float(execution.cost_usd),
            execution_fraction=float(fraction),
        )
        targets[(decision.feature_date, decision.feature_row)] = target
        total_requested += execution.requested_shares
        total_filled += execution.filled_shares
        total_cost += execution.cost_usd
        total_pnl += execution.pnl_usd
    if len(targets) != len(decisions):
        raise RuntimeError("H015 execution targets do not cover all decisions")
    return LoggedExecutionIndex(
        targets=targets,
        action_counts=dict(sorted(action_counts.items())),
        orders=len(orders),
        fills=fill_count,
        filled_decisions=filled_decisions,
        requested_shares=total_requested,
        filled_shares=total_filled,
        executed_cost_usd=total_cost,
        realized_pnl_usd=total_pnl,
    )


def aligned_execution_arrays(
    index: LoggedExecutionIndex,
    *,
    date: str,
    expected_row_indices: NDArray[np.int64],
) -> dict[str, NDArray[Any]]:
    """Materialize immutable logged targets in source-row order."""

    rows = len(expected_row_indices)
    actions = np.empty(rows, dtype=np.int64)
    values = np.empty(rows, dtype=np.float32)
    conditional = np.empty(rows, dtype=np.float32)
    pnl = np.empty(rows, dtype=np.float32)
    cost = np.empty(rows, dtype=np.float32)
    fraction = np.empty(rows, dtype=np.float32)
    for offset, row in enumerate(expected_row_indices.tolist()):
        target = index.targets.get((date, int(row)))
        if target is None:
            raise RuntimeError(f"H015 execution target is missing: {date}:{row}")
        actions[offset] = target.action_id
        values[offset] = target.realized_action_value
        conditional[offset] = target.conditional_log_utility
        pnl[offset] = target.realized_pnl_usd
        cost[offset] = target.executed_cost_usd
        fraction[offset] = target.execution_fraction
    return {
        "behavior_action_ids.npy": actions,
        "realized_action_values.npy": values,
        "realized_conditional_log_utilities.npy": conditional,
        "realized_pnl_usd.npy": pnl,
        "executed_cost_usd.npy": cost,
        "execution_fractions.npy": fraction,
    }


def validate_on_policy_shard(
    shard_dir: Path,
    files: dict[str, Any],
    *,
    expected_rows: int,
    expected_behavior_code: int,
) -> None:
    """Verify every receipt-bound H015 array before resumable reuse or training."""

    for name in H015_ARRAY_NAMES:
        metadata = files.get(name)
        path = shard_dir / name
        if not isinstance(metadata, dict) or not path.is_file():
            raise RuntimeError(f"H015 state shard is incomplete: {path}")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"H015 state shard size changed: {path}")
        if metadata.get("sha256") != sha256_file(path):
            raise RuntimeError(f"H015 state shard hash changed: {path}")
    expected_vector = (expected_rows,)
    vector_names = (
        "row_indices.npy",
        "encoding_offsets.npy",
        "component_ids.npy",
        "market_ids.npy",
        "timestamps.npy",
        "partition_codes.npy",
        "previous_action_ids.npy",
        "behavior_policy_codes.npy",
        "behavior_action_ids.npy",
        "realized_action_values.npy",
        "realized_conditional_log_utilities.npy",
        "realized_pnl_usd.npy",
        "executed_cost_usd.npy",
        "execution_fractions.npy",
    )
    values = {
        name: np.load(shard_dir / name, mmap_mode="r", allow_pickle=False)
        for name in vector_names
    }
    if any(value.shape != expected_vector for value in values.values()):
        raise RuntimeError(f"H015 state shard vectors do not align: {shard_dir}")
    portfolio = np.load(
        shard_dir / "portfolio_features.npy", mmap_mode="r", allow_pickle=False
    )
    memory = np.load(
        shard_dir / "prediction_memory_features.npy", mmap_mode="r", allow_pickle=False
    )
    physical = np.load(
        shard_dir / "physical_action_masks.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        portfolio.shape != (expected_rows, 9)
        or memory.shape != (expected_rows, 7)
        or physical.shape != (expected_rows, H012_ACTION_COUNT)
    ):
        raise RuntimeError(f"H015 state tensors do not align: {shard_dir}")
    if not bool(np.isin(values["partition_codes.npy"], (0, 1)).all()):
        raise RuntimeError(f"H015 partition codes changed: {shard_dir}")
    if not bool(np.isin(values["behavior_action_ids.npy"], (0, 1, 2)).all()):
        raise RuntimeError(f"H015 behavior actions changed: {shard_dir}")
    if not bool((values["behavior_policy_codes.npy"] == expected_behavior_code).all()):
        raise RuntimeError(f"H015 behavior policy code changed: {shard_dir}")
    fractions = values["execution_fractions.npy"]
    if bool(((fractions < 0.0) | (fractions > 1.0)).any()):
        raise RuntimeError(f"H015 execution fractions are invalid: {shard_dir}")
    for name in (
        "realized_action_values.npy",
        "realized_conditional_log_utilities.npy",
        "realized_pnl_usd.npy",
        "executed_cost_usd.npy",
    ):
        if not bool(np.isfinite(values[name]).all()):
            raise RuntimeError(f"H015 target contains non-finite values: {shard_dir}")
