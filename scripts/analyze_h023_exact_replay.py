"""Summarize exact H023 PnL, price buckets, bootstraps and debug attributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.h022_analysis import PRICE_BINS, price_bin
from sphinx_trace.h023_labels import realized_decision_labels
from sphinx_trace.model_h023 import H023_GROUP_IDS

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h023_labels.py",
    ROOT / "src" / "sphinx_trace" / "h022_analysis.py",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.relative_to(ROOT).as_posix()}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _validate_replay(replay_dir: Path) -> dict[str, Any]:
    result = _load_object(replay_dir / "result.json")
    manifest_path = replay_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    if (
        result.get("valid") is not True
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or result.get("h022_mode") != "shadow"
        or result.get("h023_mode") != "enforced"
        or result.get("h023_policy_sha256") is None
        or result.get("platform_fee_model") != "receipt_qualified_historical"
        or result.get("audit_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("H023 analysis requires a valid exact receipt-bound replay")
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
            raise RuntimeError(f"H023 audit receipt changed: {receipt_path.name}")
        digest.update(f"{receipt['date']}:{receipt['sha256']}\n".encode())
        rows += int(receipt["rows"])
    if (
        len(receipts) != int(manifest.get("days", -1))
        or rows != int(manifest.get("rows", -1))
        or digest.hexdigest() != manifest.get("shard_digest")
    ):
        raise RuntimeError("H023 audit manifest changed")
    return result


def _bootstrap(
    values: np.ndarray[Any, np.dtype[np.float64]], *, seed: int, draws: int = 10_000
) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values):
        raise ValueError("H023 bootstrap requires a non-empty vector")
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    batch = 256
    for offset in range(0, draws, batch):
        width = min(batch, draws - offset)
        sampled = generator.integers(0, len(values), size=(width, len(values)))
        means[offset : offset + width] = values[sampled].mean(axis=1)
    return {
        "groups": len(values),
        "draws": draws,
        "mean_usd": float(values.mean()),
        "lower_95pct_mean_usd": float(np.quantile(means, 0.025)),
        "upper_95pct_mean_usd": float(np.quantile(means, 0.975)),
        "positive_mean_probability": float((means > 0.0).mean()),
    }


def analyze(replay_dir: Path, baseline_dir: Path, output_path: Path) -> dict[str, Any]:
    result = _validate_replay(replay_dir)
    baseline = _load_object(baseline_dir / "result.json")
    if (
        baseline.get("valid") is not True
        or baseline.get("source_sha256") != result.get("source_sha256")
        or baseline.get("cost_multiplier") != result.get("cost_multiplier")
        or baseline.get("fee_schedule_manifest_sha256")
        != result.get("fee_schedule_manifest_sha256")
    ):
        raise RuntimeError("H023 analysis baseline is not comparable")
    decision_rows: dict[str, dict[str, Any]] = {}
    condition_components: dict[str, str] = {}
    resolution_component_pnl: dict[str, float] = defaultdict(float)
    top_feature_counts: Counter[str] = Counter()
    top_feature_signed: dict[str, list[float]] = defaultdict(list)
    attention_rows: list[list[float]] = []
    tree_group_rows: list[list[float]] = []

    def audit_rows() -> Any:
        for shard in sorted((replay_dir / "shards").glob("date=*.jsonl.zst")):
            for row in iter_jsonl_zst(shard):
                record_type = str(row.get("record_type"))
                if record_type == "h010_decision_audit":
                    condition_id = str(row["condition_id"]).lower()
                    condition_components.setdefault(condition_id, str(row["component_id"]))
                    if isinstance(row.get("h023"), dict):
                        decision_id = str(row["decision_id"])
                        decision_rows[decision_id] = row
                        attribution = row["h023"]["attribution"]
                        attention_rows.append(
                            [float(value) for value in attribution["neural_group_attention"]]
                        )
                        tree_group_rows.append(
                            [
                                float(value)
                                for value in attribution["tree_group_contributions"]
                            ]
                        )
                        for feature in attribution["top_tree_features"]:
                            name = str(feature["feature"])
                            top_feature_counts[name] += 1
                            top_feature_signed[name].append(float(feature["contribution"]))
                elif record_type == "h010_resolution_audit":
                    condition_id = str(row["condition_id"]).lower()
                    component_id = condition_components.get(condition_id)
                    if component_id is not None:
                        resolution_component_pnl[component_id] += float(
                            row["total_condition_realized_pnl_usd"]
                        )
                yield row

    labels, audit_counts = realized_decision_labels(
        audit_rows(), require_action_matches_candidate=False
    )
    if set(labels) != set(decision_rows):
        raise RuntimeError("H023 analysis labels do not cover every gate decision")
    attributed_pnl = float(
        np.sum(
            [float(value.realized_pnl_usd) for value in labels.values()],
            dtype=np.float64,
        )
    )
    exact_pnl = float(result["metrics"]["net_profit_usd"])
    if not np.isclose(attributed_pnl, exact_pnl, rtol=0.0, atol=1e-8):
        raise RuntimeError("H023 exact decision attribution does not reproduce PnL")
    buckets: dict[str, dict[str, Any]] = {}
    for name, _, _ in PRICE_BINS:
        selected: list[tuple[dict[str, Any], float]] = []
        for decision_id, row in decision_rows.items():
            h023 = row["h023"]
            if price_bin(float(h023["entry_price"])) == name:
                selected.append((row, float(labels[decision_id].realized_pnl_usd)))
        kept = [value for value in selected if value[0]["h023"]["keep_base_call"]]
        vetoed = [value for value in selected if not value[0]["h023"]["keep_base_call"]]
        buckets[name] = {
            "evaluated": len(selected),
            "kept": len(kept),
            "vetoed": len(vetoed),
            "keep_fraction": len(kept) / len(selected) if selected else 0.0,
            "filled_kept": sum(labels[str(row["decision_id"])].fill_count > 0 for row, _ in kept),
            "exact_realized_pnl_usd": float(
                np.sum([pnl for _, pnl in kept], dtype=np.float64)
            ),
            "mean_ensemble_score": float(
                np.mean(
                    [
                        float(row["h023"]["member_scores"]["ensemble_realized_contribution"])
                        for row, _ in selected
                    ],
                    dtype=np.float64,
                )
            )
            if selected
            else 0.0,
        }
    weeks = np.asarray(
        [float(row["net_profit_usd"]) for row in result["weeks"]], dtype=np.float64
    )
    components = np.asarray(list(resolution_component_pnl.values()), dtype=np.float64)
    attention = np.asarray(attention_rows, dtype=np.float64)
    tree_groups = np.asarray(tree_group_rows, dtype=np.float64)
    top_features = [
        {
            "feature": name,
            "top16_count": count,
            "top16_fraction": count / len(decision_rows),
            "mean_signed_contribution": float(
                np.mean(top_feature_signed[name], dtype=np.float64)
            ),
            "mean_absolute_contribution": float(
                np.mean(np.abs(top_feature_signed[name]), dtype=np.float64)
            ),
        }
        for name, count in top_feature_counts.most_common(32)
    ]
    analysis = {
        "record_type": "h023_exact_replay_analysis",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H023",
        "valid": True,
        "generated_at": now_utc(),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "replay_manifest_sha256": sha256_file(replay_dir / "manifest.json"),
        "baseline_result_sha256": sha256_file(baseline_dir / "result.json"),
        "implementation_sha256": _implementation_digest(),
        "cost_multiplier": float(result["cost_multiplier"]),
        "net_profit_usd": exact_pnl,
        "baseline_H021_net_profit_usd": float(baseline["metrics"]["net_profit_usd"]),
        "net_profit_delta_vs_H021_usd": exact_pnl
        - float(baseline["metrics"]["net_profit_usd"]),
        "return_on_initial_cash": float(result["metrics"]["return_on_initial_cash"]),
        "maximum_drawdown": float(result["metrics"]["maximum_drawdown"]),
        "profit_factor": float(result["metrics"]["profit_factor"]),
        "calls": sum(
            str(row["action"]).startswith("CALL_") for row in decision_rows.values()
        ),
        "vetoes": sum(
            not bool(row["h023"]["keep_base_call"])
            for row in decision_rows.values()
        ),
        "candidate_decisions": len(decision_rows),
        "called_conditions": int(result["called_conditions"]),
        "attributed_realized_pnl_usd": attributed_pnl,
        "audit_counts": audit_counts,
        "price_bins": buckets,
        "bootstrap": {
            "weekly": _bootstrap(weeks, seed=2301),
            "component": _bootstrap(components, seed=2302),
        },
        "attribution": {
            "group_ids": list(H023_GROUP_IDS),
            "mean_neural_attention": attention.mean(axis=0).tolist(),
            "mean_absolute_tree_group_contribution": np.abs(tree_groups)
            .mean(axis=0)
            .tolist(),
            "top_tree_features": top_features,
        },
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Exact stateful development validation replay with receipt-qualified fees; "
            "not historical orderbook, untouched-test or paper-forward evidence."
        ),
    }
    atomic_json(output_path, analysis)
    return analysis


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--replay-dir", type=Path, required=True)
    value.add_argument("--baseline-replay-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = analyze(
        args.replay_dir.resolve(),
        args.baseline_replay_dir.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
