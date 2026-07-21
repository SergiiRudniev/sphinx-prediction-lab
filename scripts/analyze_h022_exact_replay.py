"""Explain one exact H022 replay without treating diagnostics as counterfactual PnL."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.h022_analysis import (
    DIAGNOSTIC_GATES,
    PRICE_BINS,
    CandidateScore,
    diagnostic_gate_keep,
    price_bin,
    reference_log_utility,
)
from sphinx_trace.model_h022 import H022_GROUP_IDS

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h022_analysis.py",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    decision_id: str
    condition_id: str
    component_id: str
    side: int
    price: float
    payout_per_cost: float
    size_fraction: float
    keep: bool
    score: CandidateScore
    neural_attention: tuple[float, ...]
    tree_groups: tuple[float, ...]
    tree_price: float
    tree_wallet: float
    tree_event: float


@dataclass(frozen=True, slots=True)
class Fill:
    decision_id: str
    condition_id: str
    side: int
    price: float
    notional: Decimal
    position_shares: Decimal
    collateral_fee: Decimal


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.relative_to(ROOT).as_posix()}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_replay(replay_dir: Path) -> dict[str, Any]:
    result_path = replay_dir / "result.json"
    result = _load_object(result_path)
    manifest_path = replay_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    if (
        result.get("valid") is not True
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or result.get("h022_policy_sha256") is None
        or result.get("audit_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("H022 analysis requires a valid receipt-bound replay")
    digest = hashlib.sha256()
    rows = 0
    receipts = sorted((replay_dir / "receipts").glob("date=*.json"))
    for receipt_path in receipts:
        receipt = _load_object(receipt_path)
        shard_path = replay_dir / str(receipt["path"])
        if (
            receipt.get("sha256") != sha256_file(shard_path)
            or any(
                receipt.get(field) != result.get(field)
                for field in ("source_sha256", "policy_sha256", "implementation_sha256")
            )
        ):
            raise RuntimeError(f"H022 audit receipt changed: {receipt_path.name}")
        digest.update(f"{receipt['date']}:{receipt['sha256']}\n".encode())
        rows += int(receipt["rows"])
    if (
        len(receipts) != int(manifest.get("days", -1))
        or rows != int(manifest.get("rows", -1))
        or digest.hexdigest() != manifest.get("shard_digest")
    ):
        raise RuntimeError("H022 audit manifest changed")
    return result


def _mean(values: list[float]) -> float:
    return float(np.mean(values, dtype=np.float64)) if values else 0.0


def _candidate_summary(rows: list[tuple[Candidate, float]], gate_id: str) -> dict[str, Any]:
    kept = [row for row in rows if diagnostic_gate_keep(gate_id, row[0].score)]
    utilities = [utility for _, utility in kept]
    high = [(candidate, utility) for candidate, utility in kept if candidate.price >= 0.8]
    return {
        "calls": len(kept),
        "independent_components": len({candidate.component_id for candidate, _ in kept}),
        "mean_reference_log_utility": _mean(utilities),
        "sum_reference_log_utility": float(np.sum(utilities, dtype=np.float64)),
        "positive_reference_utility_fraction": _mean(
            [float(utility > 0.0) for utility in utilities]
        ),
        "price_at_least_0_80_calls": len(high),
        "price_at_least_0_80_mean_reference_log_utility": _mean(
            [utility for _, utility in high]
        ),
        "non_stateful_diagnostic_only": gate_id != "h022_mean_positive",
    }


def _candidate_price_bins(rows: list[tuple[Candidate, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, _, _ in PRICE_BINS:
        bucket = [(row, utility) for row, utility in rows if price_bin(row.price) == name]
        kept = [(row, utility) for row, utility in bucket if row.keep]
        vetoed = [(row, utility) for row, utility in bucket if not row.keep]
        output[name] = {
            "evaluated": len(bucket),
            "kept": len(kept),
            "vetoed": len(vetoed),
            "keep_fraction": len(kept) / len(bucket) if bucket else 0.0,
            "kept_mean_reference_log_utility": _mean(
                [utility for _, utility in kept]
            ),
            "vetoed_mean_reference_log_utility": _mean(
                [utility for _, utility in vetoed]
            ),
            "kept_positive_reference_utility_fraction": _mean(
                [float(utility > 0.0) for _, utility in kept]
            ),
            "vetoed_positive_reference_utility_fraction": _mean(
                [float(utility > 0.0) for _, utility in vetoed]
            ),
        }
    return output


def _group_attribution(rows: list[tuple[Candidate, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "mean_neural_attention": [0.0] * len(H022_GROUP_IDS),
            "mean_absolute_tree_group_contribution": [0.0] * len(H022_GROUP_IDS),
        }
    attention = np.asarray(
        [candidate.neural_attention for candidate, _ in rows], dtype=np.float64
    )
    tree_groups = np.asarray(
        [candidate.tree_groups for candidate, _ in rows], dtype=np.float64
    )
    return {
        "mean_neural_attention": attention.mean(axis=0).tolist(),
        "mean_absolute_tree_group_contribution": np.abs(tree_groups)
        .mean(axis=0)
        .tolist(),
    }


def analyze(replay_dir: Path, baseline_replay_dir: Path, output_path: Path) -> dict[str, Any]:
    result = _validate_replay(replay_dir)
    baseline = _load_object(baseline_replay_dir / "result.json")
    if (
        baseline.get("valid") is not True
        or baseline.get("source_sha256") != result.get("source_sha256")
        or baseline.get("cost_multiplier") != result.get("cost_multiplier")
    ):
        raise RuntimeError("H022 analysis baseline is not comparable")

    candidates: dict[str, Candidate] = {}
    orders: dict[str, tuple[str, str, int]] = {}
    fills: list[Fill] = []
    resolutions: dict[str, tuple[float, float]] = {}
    for shard_path in sorted((replay_dir / "shards").glob("date=*.jsonl.zst")):
        for row in iter_jsonl_zst(shard_path):
            record_type = str(row["record_type"])
            if record_type == "h010_decision_audit" and isinstance(row.get("h022"), dict):
                h021 = row["h021"]
                h022 = row["h022"]
                side = int(h022["candidate_action_id"])
                execution = tuple(float(value) for value in h021["execution_context"])
                quantiles = tuple(float(value) for value in h022["net_return_quantiles"])
                attribution = h022["attribution"]
                candidate = Candidate(
                    decision_id=str(row["decision_id"]),
                    condition_id=str(row["condition_id"]),
                    component_id=str(row["component_id"]),
                    side=side,
                    price=execution[side],
                    payout_per_cost=execution[2 + side],
                    size_fraction=float(row["size_fraction"]),
                    keep=bool(h022["keep_base_call"]),
                    score=CandidateScore(
                        ensemble=float(h022["member_scores"]["ensemble_net_return"]),
                        q50=quantiles[1],
                        q90=quantiles[2],
                    ),
                    neural_attention=tuple(
                        float(value) for value in attribution["neural_group_attention"]
                    ),
                    tree_groups=tuple(
                        float(value) for value in attribution["tree_group_contributions"]
                    ),
                    tree_price=float(attribution["tree_price_context"]),
                    tree_wallet=float(attribution["tree_wallet"]),
                    tree_event=float(attribution["tree_event"]),
                )
                if candidate.keep != str(row["action"]).startswith("CALL_"):
                    raise RuntimeError("H022 keep flag and exact action disagree")
                candidates[candidate.decision_id] = candidate
            elif record_type == "h010_order_audit":
                decision_id = str(row["decision_id"])
                order_candidate = candidates.get(decision_id)
                if order_candidate is not None:
                    orders[str(row["order_id"])] = (
                        decision_id,
                        str(row["condition_id"]),
                        order_candidate.side,
                    )
            elif record_type == "h010_fill_audit":
                order = orders.get(str(row["order_id"]))
                if order is not None:
                    fills.append(
                        Fill(
                            decision_id=order[0],
                            condition_id=order[1],
                            side=order[2],
                            price=float(row["price"]),
                            notional=Decimal(str(row["notional_usd"])),
                            position_shares=Decimal(str(row["position_shares"])),
                            collateral_fee=Decimal(str(row["collateral_fee_usd"])),
                        )
                    )
            elif record_type == "h010_resolution_audit":
                payouts = tuple(float(value) for value in row["payouts"])
                if len(payouts) == 2:
                    resolutions[str(row["condition_id"])] = (payouts[0], payouts[1])

    if not candidates or any(row.condition_id not in resolutions for row in candidates.values()):
        raise RuntimeError("H022 candidates are not fully resolved")
    evaluated = [
        (
            candidate,
            reference_log_utility(
                candidate.size_fraction,
                candidate.payout_per_cost,
                resolutions[candidate.condition_id][candidate.side],
            ),
        )
        for candidate in candidates.values()
    ]

    fill_bins: dict[str, dict[str, Any]] = {
        name: {
            "fills": 0,
            "decision_ids": set(),
            "notional_usd": Decimal(0),
            "realized_pnl_usd": Decimal(0),
        }
        for name, _, _ in PRICE_BINS
    }
    for fill in fills:
        payout = Decimal(str(resolutions[fill.condition_id][fill.side]))
        pnl = (
            fill.position_shares * payout - fill.notional - fill.collateral_fee
        )
        bucket = fill_bins[price_bin(fill.price)]
        bucket["fills"] += 1
        bucket["decision_ids"].add(fill.decision_id)
        bucket["notional_usd"] += fill.notional + fill.collateral_fee
        bucket["realized_pnl_usd"] += pnl
    serialized_bins = {
        name: {
            "fills": int(values["fills"]),
            "decisions": len(values["decision_ids"]),
            "notional_usd": float(values["notional_usd"]),
            "realized_pnl_usd": float(values["realized_pnl_usd"]),
        }
        for name, values in fill_bins.items()
    }

    kept_evaluated = [(row, utility) for row, utility in evaluated if row.keep]
    vetoed_evaluated = [(row, utility) for row, utility in evaluated if not row.keep]
    below_080_pnl = sum(
        values["realized_pnl_usd"]
        for name, values in serialized_bins.items()
        if name in {"below_0_50", "0_50_to_0_70", "0_70_to_0_80"}
    )
    at_least_080_pnl = sum(
        values["realized_pnl_usd"]
        for name, values in serialized_bins.items()
        if name not in {"below_0_50", "0_50_to_0_70", "0_70_to_0_80"}
    )
    exact_profit = float(result["metrics"]["net_profit_usd"])
    baseline_profit = float(baseline["metrics"]["net_profit_usd"])
    output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h022_exact_replay_analysis",
        "generated_at": now_utc(),
        "valid": True,
        "implementation_sha256": _implementation_digest(),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "baseline_result_sha256": sha256_file(baseline_replay_dir / "result.json"),
        "cost_multiplier": result["cost_multiplier"],
        "exact_stateful": {
            "net_profit_usd": exact_profit,
            "baseline_net_profit_usd": baseline_profit,
            "profit_delta_usd": exact_profit - baseline_profit,
            "calls": int(result["resolved_calls"]),
            "called_conditions": int(result["called_conditions"]),
            "call_precision": float(result["call_precision"]),
            "maximum_drawdown": float(result["metrics"]["maximum_drawdown"]),
            "fills": int(result["metrics"]["fills"]),
            "fees_usd": float(result["metrics"]["total_fees_usd"]),
        },
        "candidate_gate": {
            "evaluated_h021_calls_in_h022_state": len(evaluated),
            "kept": sum(candidate.keep for candidate, _ in evaluated),
            "vetoed": sum(not candidate.keep for candidate, _ in evaluated),
            "diagnostic_gates": {
                gate_id: _candidate_summary(evaluated, gate_id)
                for gate_id in DIAGNOSTIC_GATES
            },
            "price_bins": _candidate_price_bins(evaluated),
            "warning": (
                "Alternative gates reuse realized labels in the H022 state trajectory. "
                "They are diagnostic only and require fit-only preregistration plus a "
                "new exact stateful replay."
            ),
        },
        "exact_fill_price_attribution": {
            "bins": serialized_bins,
            "below_0_80_realized_pnl_usd": below_080_pnl,
            "at_least_0_80_realized_pnl_usd": at_least_080_pnl,
        },
        "attribution": {
            "group_ids": list(H022_GROUP_IDS),
            "all_candidates": _group_attribution(evaluated),
            "kept_candidates": _group_attribution(kept_evaluated),
            "vetoed_candidates": _group_attribution(vetoed_evaluated),
            "mean_signed_tree_price_context": _mean(
                [candidate.tree_price for candidate, _ in evaluated]
            ),
            "mean_signed_tree_wallet": _mean(
                [candidate.tree_wallet for candidate, _ in evaluated]
            ),
            "mean_signed_tree_event": _mean(
                [candidate.tree_event for candidate, _ in evaluated]
            ),
        },
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "promotion_allowed": False,
        "evidence_boundary": (
            "Exact development trade-tape replay with receipt-qualified fees. Fill "
            "attribution is exact for this replay; alternative score gates are not "
            "stateful counterfactual profit evidence."
        ),
    }
    atomic_json(output_path, output)
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--replay-dir", type=Path, required=True)
    value.add_argument("--baseline-replay-dir", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    replay_dir = args.replay_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else replay_dir / "h022-analysis.json"
    )
    result = analyze(replay_dir, args.baseline_replay_dir.resolve(), output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
