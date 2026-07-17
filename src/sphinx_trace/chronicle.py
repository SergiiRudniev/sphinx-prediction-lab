"""Causal target and chronological split primitives for Sphinx Chronicle."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any


def parse_utc(value: str) -> datetime:
    normalized = value.strip().replace(" ", "T")
    if normalized.endswith("+00"):
        normalized += ":00"
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return parsed.astimezone(UTC)


def format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SplitWindow:
    id: str
    start: datetime
    end: datetime

    def contains(self, value: datetime) -> bool:
        return self.start <= value < self.end


@dataclass(frozen=True)
class SplitPlan:
    windows: tuple[SplitWindow, ...]
    embargo: timedelta

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> SplitPlan:
        split = payload["split"]
        windows = tuple(
            SplitWindow(
                id=str(item["id"]),
                start=parse_utc(str(item["start"])),
                end=parse_utc(str(item["end_exclusive"])),
            )
            for item in split["segments"]
        )
        if not windows:
            raise ValueError("At least one split segment is required")
        if any(window.start >= window.end for window in windows):
            raise ValueError("Every split segment must have positive duration")
        if len({window.id for window in windows}) != len(windows):
            raise ValueError("Split segment IDs must be unique")
        embargo = timedelta(days=int(split["embargo_days"]))
        for left, right in pairwise(windows):
            if left.end + embargo != right.start:
                raise ValueError(
                    f"Expected an exact {embargo.days}-day embargo between {left.id} and {right.id}"
                )
        return cls(windows=windows, embargo=embargo)

    def locate(self, value: datetime) -> SplitWindow | None:
        return next((window for window in self.windows if window.contains(value)), None)

    def by_id(self, split_id: str) -> SplitWindow:
        for window in self.windows:
            if window.id == split_id:
                return window
        raise KeyError(split_id)


@dataclass(frozen=True)
class MarketResolution:
    condition_id: str
    event_id: str
    resolved_at: datetime
    resolved_yes: int
    atlas_observed_at: str


@dataclass(frozen=True)
class Horizon:
    id: str
    seconds: int
    maximum_lag_seconds: int


def horizons_from_config(payload: dict[str, Any]) -> tuple[Horizon, ...]:
    horizons = tuple(
        Horizon(
            id=str(item["id"]),
            seconds=int(item["seconds"]),
            maximum_lag_seconds=int(item["maximum_lag_seconds"]),
        )
        for item in payload["targets"]["horizons"]
    )
    if not horizons or any(
        horizon.seconds <= 0 or horizon.maximum_lag_seconds < 0 for horizon in horizons
    ):
        raise ValueError("Target horizons and lags must be non-negative")
    if len({horizon.id for horizon in horizons}) != len(horizons):
        raise ValueError("Target horizon IDs must be unique")
    return horizons


def market_resolution_from_atlas(
    row: dict[str, Any],
    eligibility: dict[str, Any],
) -> MarketResolution | None:
    if row.get("resolution_status") != eligibility["resolution_status"]:
        return None
    event_ids = row.get("event_ids")
    if not isinstance(event_ids, list) or not event_ids:
        return None
    if bool(eligibility["require_single_event_id"]) and len(event_ids) != 1:
        return None
    payload = row.get("source_payload")
    if not isinstance(payload, dict):
        return None
    try:
        outcomes = json.loads(str(payload.get("outcomes") or "[]"))
        prices = [float(value) for value in json.loads(str(payload.get("outcomePrices") or "[]"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if outcomes != eligibility["ordered_outcomes"] or len(prices) != 2:
        return None
    minimum = float(eligibility["terminal_probability_minimum"])
    maximum = float(eligibility["terminal_probability_maximum"])
    if prices[0] >= minimum and prices[1] <= maximum:
        resolved_yes = 1
    elif prices[1] >= minimum and prices[0] <= maximum:
        resolved_yes = 0
    else:
        return None
    condition_id = str(row.get("condition_id") or "").lower()
    closed_at = str(row.get("closed_at") or "")
    observed_at = str(row.get("observed_at") or "")
    if not condition_id or not closed_at or not observed_at:
        return None
    try:
        resolved_at = parse_utc(closed_at)
        parse_utc(observed_at)
    except ValueError:
        return None
    return MarketResolution(
        condition_id=condition_id,
        event_id=str(event_ids[0]),
        resolved_at=resolved_at,
        resolved_yes=resolved_yes,
        atlas_observed_at=observed_at,
    )


def yes_equivalent_price(row: dict[str, Any]) -> float | None:
    try:
        price = float(row["price"])
        outcome_index = int(row["outcome_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 < price < 1.0 or outcome_index not in {0, 1}:
        return None
    return price if outcome_index == 0 else 1.0 - price


def yes_direction(row: dict[str, Any]) -> int | None:
    try:
        outcome_index = int(row["outcome_index"])
    except (KeyError, TypeError, ValueError):
        return None
    side = str(row.get("side") or "").upper()
    if outcome_index not in {0, 1} or side not in {"BUY", "SELL"}:
        return None
    token_direction = 1 if outcome_index == 0 else -1
    side_direction = 1 if side == "BUY" else -1
    return token_direction * side_direction


def _future_yes_price(
    rows: list[dict[str, Any]],
    timestamps: list[int],
    *,
    after_index: int,
    target_unix: int,
    maximum_lag_seconds: int,
    upper_bound_unix: int,
) -> tuple[float, int] | None:
    index = max(after_index + 1, bisect_left(timestamps, target_unix))
    maximum_timestamp = min(target_unix + maximum_lag_seconds, upper_bound_unix - 1)
    while index < len(rows) and timestamps[index] <= maximum_timestamp:
        price = yes_equivalent_price(rows[index])
        if price is not None:
            return price, timestamps[index]
        index += 1
    return None


def _example_id(condition_id: str, decision_unix: int, wallet: str, trade_id: str) -> str:
    payload = f"{condition_id}|{decision_unix}|{wallet}|{trade_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_condition_targets(
    rows: list[dict[str, Any]],
    market: MarketResolution,
    window: SplitWindow,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    decision = payload["decision"]
    minimum_history = int(decision["minimum_history_trades"])
    stride = int(decision["stride_trades"])
    delay = int(decision["decision_delay_seconds"])
    horizons = horizons_from_config(payload)
    fixed_cost = float(payload["targets"]["net_edge_proxy"]["fixed_round_trip_cost_bps"]) / 10_000.0
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("timestamp_unix") or 0),
            str(row.get("transaction_hash") or ""),
            str(row.get("trade_id") or ""),
        ),
    )
    timestamps = [int(row.get("timestamp_unix") or 0) for row in ordered]
    output: list[dict[str, Any]] = []
    if len(ordered) < minimum_history:
        return output
    resolution_unix = int(market.resolved_at.timestamp())
    upper_bound_unix = min(int(window.end.timestamp()), resolution_unix)
    for anchor_index in range(minimum_history - 1, len(ordered), stride):
        anchor = ordered[anchor_index]
        anchor_unix = timestamps[anchor_index]
        decision_unix = anchor_unix + delay
        decision_time = datetime.fromtimestamp(decision_unix, tz=UTC)
        if not window.contains(decision_time):
            continue
        if decision_unix >= resolution_unix or not window.contains(market.resolved_at):
            continue
        current_yes = yes_equivalent_price(anchor)
        direction = yes_direction(anchor)
        wallet = str(anchor.get("wallet") or "").lower()
        trade_id = str(anchor.get("trade_id") or "").lower()
        if current_yes is None or direction is None or not wallet or not trade_id:
            continue

        target_values: dict[str, float | str | None] = {}
        for horizon in horizons:
            future = _future_yes_price(
                ordered,
                timestamps,
                after_index=anchor_index,
                target_unix=decision_unix + horizon.seconds,
                maximum_lag_seconds=horizon.maximum_lag_seconds,
                upper_bound_unix=upper_bound_unix,
            )
            markout = None if future is None else future[0] - current_yes
            target_values[f"yes_markout_{horizon.id}"] = markout
            target_values[f"directional_markout_{horizon.id}"] = (
                None if markout is None else direction * markout
            )
            target_values[f"markout_observed_at_{horizon.id}"] = (
                None if future is None else format_utc(datetime.fromtimestamp(future[1], tz=UTC))
            )

        resolved_edge = direction * (market.resolved_yes - current_yes)
        target_values["resolved_directional_edge"] = resolved_edge
        target_values["net_edge_proxy"] = resolved_edge - fixed_cost
        output.append(
            {
                "schema_version": "1.0.0",
                "record_type": "chronicle_target",
                "example_id": _example_id(
                    market.condition_id,
                    decision_unix,
                    wallet,
                    trade_id,
                ),
                "condition_id": market.condition_id,
                "event_id": market.event_id,
                "split": window.id,
                "decision_time": format_utc(decision_time),
                "decision_time_unix": decision_unix,
                "feature_max_event_time_unix": anchor_unix,
                "history_trade_count": minimum_history,
                "anchor_trade_id": trade_id,
                "anchor_wallet": wallet,
                "anchor_side": str(anchor.get("side") or "").upper(),
                "anchor_outcome_index": int(anchor["outcome_index"]),
                "yes_price": current_yes,
                "yes_direction": direction,
                "resolved_at": format_utc(market.resolved_at),
                "resolved_yes": market.resolved_yes,
                "targets": target_values,
                "source": {
                    "ledger_namespace": str(payload["corpus"]["ledger_namespace"]),
                    "atlas_observed_at": market.atlas_observed_at,
                },
            }
        )
    return output


def target_row_is_causal(row: dict[str, Any], plan: SplitPlan) -> bool:
    decision_unix = int(row["decision_time_unix"])
    if int(row["feature_max_event_time_unix"]) >= decision_unix:
        return False
    split_id = str(row["split"])
    window = plan.by_id(split_id)
    if not window.contains(datetime.fromtimestamp(decision_unix, tz=UTC)):
        return False
    if not window.contains(parse_utc(str(row["resolved_at"]))):
        return False
    targets = row.get("targets")
    if not isinstance(targets, dict):
        return False
    for key, value in targets.items():
        if key.startswith("markout_observed_at_") and value is not None:
            observed_at = parse_utc(str(value))
            if not window.contains(observed_at) or observed_at <= datetime.fromtimestamp(
                decision_unix, tz=UTC
            ):
                return False
    return True


def target_row_matches_contract(
    row: dict[str, Any],
    payload: dict[str, Any],
    plan: SplitPlan,
) -> bool:
    if not target_row_is_causal(row, plan):
        return False
    targets = row["targets"]
    if not isinstance(targets, dict):
        return False
    decision_unix = int(row["decision_time_unix"])
    direction = int(row["yes_direction"])
    for horizon in horizons_from_config(payload):
        markout = targets.get(f"yes_markout_{horizon.id}")
        directional = targets.get(f"directional_markout_{horizon.id}")
        observed = targets.get(f"markout_observed_at_{horizon.id}")
        if observed is None:
            if markout is not None or directional is not None:
                return False
            continue
        if not isinstance(markout, (int, float)) or not isinstance(directional, (int, float)):
            return False
        observed_unix = int(parse_utc(str(observed)).timestamp())
        lag = observed_unix - decision_unix
        if not horizon.seconds <= lag <= horizon.seconds + horizon.maximum_lag_seconds:
            return False
        if not math.isclose(float(directional), direction * float(markout), abs_tol=1e-12):
            return False

    current_yes = float(row["yes_price"])
    resolved_yes = int(row["resolved_yes"])
    resolved_edge = direction * (resolved_yes - current_yes)
    if not math.isclose(
        float(targets["resolved_directional_edge"]),
        resolved_edge,
        abs_tol=1e-12,
    ):
        return False
    fixed_cost = float(payload["targets"]["net_edge_proxy"]["fixed_round_trip_cost_bps"]) / 10_000.0
    return math.isclose(
        float(targets["net_edge_proxy"]),
        resolved_edge - fixed_cost,
        abs_tol=1e-12,
    )
