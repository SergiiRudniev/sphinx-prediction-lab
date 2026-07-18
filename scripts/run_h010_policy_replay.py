"""Run one closed-test H012 policy through the exact stateful H010 tape."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.development_tape import load_tape_conditions
from sphinx_trace.policy_checkpoint import load_policy_checkpoint
from sphinx_trace.policy_decisions import PolicyFeatureStore, load_policy_decisions
from sphinx_trace.policy_encodings import PolicyEncodingStore
from sphinx_trace.policy_runtime import H012PolicyRuntime, PolicyInference
from sphinx_trace.replay_audit import (
    build_audit_manifest,
    decision_audit_record,
    write_audit_shard,
)
from sphinx_trace.replay_h010 import H010ReplayAdapter, SelectiveAction
from sphinx_trace.simulator import ReplaySimulator, SimulationRules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATOR_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_simulator_h010_v1.json"
DEFAULT_POLICY_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "simulator.py",
    ROOT / "src" / "sphinx_trace" / "replay_h010.py",
    ROOT / "src" / "sphinx_trace" / "replay_audit.py",
    ROOT / "src" / "sphinx_trace" / "policy_decisions.py",
    ROOT / "src" / "sphinx_trace" / "policy_runtime.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "policy_encodings.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rules(config: dict[str, Any], cost_multiplier: float) -> SimulationRules:
    if cost_multiplier <= 0.0:
        raise ValueError("H010 cost multiplier must be positive")
    execution = config["execution"]
    proxy = execution["historical_trade_tape_proxy"]
    return SimulationRules(
        initial_cash_usd=Decimal(str(config["portfolio"]["initial_cash_usd"])),
        latency_seconds=int(execution["latency_seconds"]),
        maximum_fill_wait_seconds=int(execution["maximum_fill_wait_seconds"]),
        available_share_fraction=Decimal(str(proxy["available_share_fraction"])),
        duplicate_liquidity_haircut=Decimal(str(proxy["duplicate_liquidity_haircut"])),
        adverse_price_ticks=max(
            0, math.ceil(float(proxy["adverse_price_ticks"]) * cost_multiplier)
        ),
        tick_size=Decimal(str(proxy["tick_size"])),
        fee_bps=Decimal(str(float(proxy["fee_bps"]) * cost_multiplier)),
        opposing_side_required=bool(proxy["opposing_side_required"]),
        retain_processed_liquidity_ids=False,
        retain_prediction_records=False,
    )


def _fill_record(fill: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "h010_fill_audit",
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "liquidity_id": fill.liquidity_id,
        "timestamp_unix": fill.timestamp_unix,
        "side": fill.side.value,
        "shares": str(fill.shares),
        "price": str(fill.price),
        "notional_usd": str(fill.notional_usd),
        "fee_usd": str(fill.fee_usd),
    }


def _order_record(order: Any) -> dict[str, Any]:
    value = asdict(order)
    for key in ("requested_shares", "limit_price", "filled_shares"):
        value[key] = str(value[key])
    value["side"] = order.side.value
    value["status"] = order.status.value
    return {
        "schema_version": "1.0.0",
        "record_type": "h010_order_audit",
        "timestamp_unix": order.submitted_at_unix,
        **value,
    }


def _resolution_record(
    condition_id: str,
    timestamp_unix: int,
    payouts: tuple[Any, Any],
    terminal_pnl: Any,
    total_condition_pnl: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "h010_resolution_audit",
        "condition_id": condition_id,
        "timestamp_unix": timestamp_unix,
        "payouts": [str(value) for value in payouts],
        "terminal_realized_pnl_usd": str(terminal_pnl),
        "total_condition_realized_pnl_usd": str(total_condition_pnl),
    }


def _decision_record(
    inference: PolicyInference,
    outcomes: tuple[str, str],
) -> dict[str, Any]:
    call = inference.call
    return decision_audit_record(
        decision_id=call.decision_id,
        timestamp_unix=call.timestamp_unix,
        condition_id=call.condition_id,
        component_id=call.component_id,
        evidence_trade_id=call.evidence_trade_id,
        feature_date=inference.feature_date,
        feature_row=inference.feature_row,
        input_sha256=call.input_sha256,
        action=call.action.value,
        probability_outcome0=float(call.probability_outcome0),
        size_fraction=float(call.size_fraction),
        physical_action_mask=inference.physical_action_mask,
        portfolio_features=inference.portfolio_features,
        prediction_memory_features=inference.prediction_memory_features,
        previous_action_id=inference.previous_action_id,
        action_logits=inference.action_logits,
        outcome_labels=outcomes,
    )


def _week_metrics(
    daily_equity: list[tuple[str, float]], initial_equity: float
) -> list[dict[str, Any]]:
    weekly: dict[str, list[tuple[str, float]]] = {}
    for date, equity in daily_equity:
        parsed = datetime.fromisoformat(date).date()
        monday = parsed - timedelta(days=parsed.weekday())
        weekly.setdefault(monday.isoformat(), []).append((date, equity))
    output: list[dict[str, Any]] = []
    previous = initial_equity
    for week, rows in sorted(weekly.items()):
        final = rows[-1][1]
        output.append(
            {
                "week": week,
                "start_equity_usd": previous,
                "end_equity_usd": final,
                "net_profit_usd": final - previous,
                "return": (final - previous) / previous if previous else 0.0,
            }
        )
        previous = final
    return output


def replay(
    simulator_config_path: Path,
    policy_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    tape_dir: Path,
    pack_dir: Path,
    outcome_dir: Path,
    policy_dir: Path,
    output_dir: Path,
    *,
    encoding_cache_dir: Path | None = None,
    split: str,
    cost_multiplier: float,
) -> dict[str, Any]:
    if split not in {"validation", "calibration"}:
        raise ValueError("H010 replay supports validation or calibration only")
    simulator_config = load_json(simulator_config_path)
    policy_config = load_json(policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    tape_manifest_path = tape_dir / "manifest.json"
    tape_manifest = _load_object(tape_manifest_path)
    if (
        tape_manifest.get("valid") is not True
        or tape_manifest.get("test_labels_opened") is not False
        or int(tape_manifest.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H010 replay requires a valid closed-test development tape")
    catalog = load_tape_conditions(tape_dir, split)
    decisions, shards = load_policy_decisions(pack_dir, (split,))
    if not decisions:
        raise RuntimeError("H010 replay found no policy decisions")
    decision_conditions = {ref.condition_id for refs in decisions.values() for ref in refs}
    if not decision_conditions <= catalog.contracts.keys():
        raise RuntimeError("H010 policy decisions escape the replay catalog")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_policy_checkpoint(
        policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        policy_config,
        device,
    )
    feature_store = PolicyFeatureStore(
        pack_dir,
        shards,
        feature_clip=12.0,
        cache_shards=8,
    )
    encoding_store = (
        PolicyEncodingStore(
            encoding_cache_dir,
            pack_dir,
            policy_dir,
            shards,
            cache_shards=8,
        )
        if encoding_cache_dir is not None
        else None
    )
    runtime = H012PolicyRuntime(
        checkpoint.model,
        feature_store,
        checkpoint.feature_mask,
        checkpoint.group_mask,
        device,
        encoding_store=encoding_store,
    )
    source_sha256 = sha256_file(tape_manifest_path)
    policy_sha256 = sha256_file(policy_dir / "result.json")
    encoding_manifest_sha256 = (
        encoding_store.manifest_sha256 if encoding_store is not None else None
    )
    implementation_sha256 = _implementation_digest()
    contract_payload = (
        f"simulator:{sha256_file(simulator_config_path)}\n"
        f"policy_config:{sha256_file(policy_config_path)}\n"
        f"source:{source_sha256}\npolicy:{policy_sha256}\n"
        f"encoding_cache:{encoding_manifest_sha256 or 'none'}\n"
        f"implementation:{implementation_sha256}\nsplit:{split}\n"
        f"cost_multiplier:{cost_multiplier}\n"
    )
    contract_sha256 = hashlib.sha256(contract_payload.encode()).hexdigest()
    checkpoint_path = output_dir / "checkpoint.pt"
    start_ordinal = 0
    resolution_position = 0
    daily_equity: list[tuple[str, float]] = []
    action_counts: Counter[str] = Counter()
    called_conditions: set[str] = set()
    pending_call_sides: dict[str, list[int]] = {}
    correct_calls = 0
    resolved_calls = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H010 replay checkpoint belongs to another contract")
        adapter = H010ReplayAdapter.from_snapshot(
            dict(saved["adapter"]), catalog.contracts, source_sha256=source_sha256
        )
        start_ordinal = int(saved["next_shard_ordinal"])
        resolution_position = int(saved["resolution_position"])
        daily_equity = [(str(row[0]), float(row[1])) for row in saved["daily_equity"]]
        action_counts.update(dict(saved["action_counts"]))
        called_conditions = set(saved["called_conditions"])
        pending_call_sides = {
            str(key): [int(value) for value in values]
            for key, values in saved["pending_call_sides"].items()
        }
        correct_calls = int(saved["correct_calls"])
        resolved_calls = int(saved["resolved_calls"])
    else:
        adapter = H010ReplayAdapter(
            ReplaySimulator(_rules(simulator_config, cost_multiplier)),
            catalog.contracts,
            source_sha256=source_sha256,
        )
    resolutions = catalog.resolutions

    def resolve_until(timestamp: int, inclusive: bool, records: list[dict[str, Any]]) -> None:
        nonlocal resolution_position, correct_calls, resolved_calls
        while resolution_position < len(resolutions):
            row = resolutions[resolution_position]
            if row.timestamp_unix > timestamp or (
                row.timestamp_unix == timestamp and not inclusive
            ):
                break
            pnl = adapter.resolve(row.condition_id, row.timestamp_unix, row.payouts)
            total_condition_pnl = adapter.simulator.pop_condition_realized_pnl(row.condition_id)
            records.append(
                _resolution_record(
                    row.condition_id,
                    row.timestamp_unix,
                    row.payouts,
                    pnl,
                    total_condition_pnl,
                )
            )
            winning_side = 0 if row.payouts[0] > row.payouts[1] else 1
            sides = pending_call_sides.pop(row.condition_id, [])
            correct_calls += sum(side == winning_side for side in sides)
            resolved_calls += len(sides)
            resolution_position += 1

    stream_paths = sorted((tape_dir / "stream").glob("date=*.jsonl.zst"))
    if len(stream_paths) != int(tape_manifest["shards"]):
        raise RuntimeError("H010 replay tape shard count changed")
    if start_ordinal:
        processed_dates = {
            path.name.removeprefix("date=").removesuffix(".jsonl.zst")
            for path in stream_paths[:start_ordinal]
        }
        decisions = {
            evidence: tuple(ref for ref in refs if ref.feature_date not in processed_dates)
            for evidence, refs in decisions.items()
            if any(ref.feature_date not in processed_dates for ref in refs)
        }
    for shard_ordinal, stream_path in enumerate(stream_paths):
        if shard_ordinal < start_ordinal:
            continue
        date = stream_path.name.removeprefix("date=").removesuffix(".jsonl.zst")
        records: list[dict[str, Any]] = []
        for row_ordinal, payload in enumerate(iter_jsonl_zst(stream_path)):
            if str(payload["development_split"]) != split:
                continue
            timestamp = int(payload["timestamp_unix"])
            resolve_until(timestamp, False, records)
            fills_before = len(adapter.simulator.fills)
            reference_prices = adapter.observe_trade(
                payload,
                shard_ordinal=shard_ordinal,
                row_ordinal=row_ordinal,
            )
            records.extend(_fill_record(fill) for fill in adapter.simulator.fills[fills_before:])
            refs = decisions.pop(str(payload["trade_id"]), ())
            for ref in refs:
                if (
                    ref.timestamp_unix != timestamp
                    or ref.condition_id != str(payload["condition_id"]).lower()
                ):
                    raise RuntimeError("H010 evidence trade no longer matches H012 decision")
                inference = runtime.infer(ref, adapter)
                contract = catalog.contracts[ref.condition_id]
                records.append(_decision_record(inference, contract.outcomes))
                orders = adapter.apply_call(inference.call, reference_prices)
                records.extend(_order_record(order) for order in orders)
                action = inference.call.action
                action_counts[action.value] += 1
                if action in {
                    SelectiveAction.CALL_OUTCOME_0,
                    SelectiveAction.CALL_OUTCOME_1,
                }:
                    side = 0 if action == SelectiveAction.CALL_OUTCOME_0 else 1
                    called_conditions.add(ref.condition_id)
                    pending_call_sides.setdefault(ref.condition_id, []).append(side)
        day = datetime.fromisoformat(date).replace(tzinfo=UTC)
        day_end = int((day + timedelta(days=1)).timestamp()) - 1
        resolve_until(day_end, True, records)
        write_audit_shard(
            output_dir,
            date,
            records,
            source_sha256=source_sha256,
            policy_sha256=policy_sha256,
            implementation_sha256=implementation_sha256,
        )
        daily_equity.append((date, float(adapter.simulator.equity_usd())))
        adapter.simulator.compact_history()
        _atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": "1.0.0",
                "record_type": "h010_policy_replay_checkpoint",
                "contract_sha256": contract_sha256,
                "next_shard_ordinal": shard_ordinal + 1,
                "resolution_position": resolution_position,
                "adapter": adapter.snapshot(),
                "daily_equity": daily_equity,
                "action_counts": dict(action_counts),
                "called_conditions": sorted(called_conditions),
                "pending_call_sides": pending_call_sides,
                "correct_calls": correct_calls,
                "resolved_calls": resolved_calls,
            },
        )
        if (output_dir / "PAUSE").exists():
            return {
                "status": "paused",
                "next_shard_ordinal": shard_ordinal + 1,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
    final_records: list[dict[str, Any]] = []
    resolve_until(2**63 - 1, True, final_records)
    if final_records:
        final_date = (
            datetime.fromtimestamp(final_records[-1]["timestamp_unix"], tz=UTC).date().isoformat()
        )
        write_audit_shard(
            output_dir,
            f"{final_date}-terminal",
            final_records,
            source_sha256=source_sha256,
            policy_sha256=policy_sha256,
            implementation_sha256=implementation_sha256,
        )
        final_equity = float(adapter.simulator.equity_usd())
        if daily_equity and daily_equity[-1][0] == final_date:
            daily_equity[-1] = (final_date, final_equity)
        else:
            daily_equity.append((final_date, final_equity))
        adapter.simulator.compact_history()
    if resolution_position != len(resolutions) or pending_call_sides:
        raise RuntimeError("H010 replay did not settle every development market call")
    if decisions:
        missing = sum(len(refs) for refs in decisions.values())
        raise RuntimeError(f"H010 replay missed {missing} qualified policy decisions")
    audit_manifest = build_audit_manifest(
        output_dir,
        source_sha256=source_sha256,
        policy_sha256=policy_sha256,
        implementation_sha256=implementation_sha256,
    )
    metrics = adapter.simulator.metrics()
    _atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "1.0.0",
            "record_type": "h010_policy_replay_checkpoint",
            "contract_sha256": contract_sha256,
            "next_shard_ordinal": len(stream_paths),
            "resolution_position": resolution_position,
            "adapter": adapter.snapshot(),
            "daily_equity": daily_equity,
            "action_counts": dict(action_counts),
            "called_conditions": sorted(called_conditions),
            "pending_call_sides": pending_call_sides,
            "correct_calls": correct_calls,
            "resolved_calls": resolved_calls,
        },
    )
    weeks = _week_metrics(daily_equity, float(adapter.simulator.rules.initial_cash_usd))
    weekly_profits = [float(row["net_profit_usd"]) for row in weeks]
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h010_policy_replay_result",
        "completed_at": now_utc(),
        "valid": True,
        "split": split,
        "cost_multiplier": cost_multiplier,
        "contract_sha256": contract_sha256,
        "source_sha256": source_sha256,
        "policy_sha256": policy_sha256,
        "encoding_manifest_sha256": encoding_manifest_sha256,
        "implementation_sha256": implementation_sha256,
        "metrics": metrics,
        "actions": dict(sorted(action_counts.items())),
        "called_conditions": len(called_conditions),
        "resolved_calls": resolved_calls,
        "correct_calls": correct_calls,
        "call_precision": correct_calls / resolved_calls if resolved_calls else 0.0,
        "weeks": weeks,
        "weekly_profit": {
            "weeks": len(weekly_profits),
            "mean_usd": sum(weekly_profits) / len(weekly_profits) if weekly_profits else 0.0,
            "positive_fraction": (
                sum(value > 0.0 for value in weekly_profits) / len(weekly_profits)
                if weekly_profits
                else 0.0
            ),
            "minimum_usd": min(weekly_profits) if weekly_profits else 0.0,
            "maximum_usd": max(weekly_profits) if weekly_profits else 0.0,
        },
        "audit_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "audit_rows": audit_manifest["rows"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Development trade-tape liquidity proxy only; not historical orderbook "
            "executability, untouched-test or paper-forward profit evidence."
        ),
    }
    atomic_json(output_dir / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--simulator-config", type=Path, default=DEFAULT_SIMULATOR_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--tape-dir", type=Path, required=True)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--policy-dir", type=Path, required=True)
    value.add_argument("--encoding-cache-dir", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--split", choices=("validation", "calibration"), required=True)
    value.add_argument("--cost-multiplier", type=float, default=1.0)
    return value


def main() -> None:
    args = parser().parse_args()
    result = replay(
        args.simulator_config.resolve(),
        args.policy_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.tape_dir.resolve(),
        args.pack_dir.resolve(),
        args.outcome_dir.resolve(),
        args.policy_dir.resolve(),
        args.output_dir.resolve(),
        encoding_cache_dir=(
            args.encoding_cache_dir.resolve() if args.encoding_cache_dir is not None else None
        ),
        split=args.split,
        cost_multiplier=args.cost_multiplier,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
