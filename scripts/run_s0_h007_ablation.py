from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h007_ablation.json"
BASE_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_train.json"
TRAIN_SCRIPT = ROOT / "scripts" / "train_s0_trial_t0.py"


def _probability(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def _row_log_loss(
    probability: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> NDArray[np.float64]:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))


def _bootstrap_delta(
    candidate_probability: NDArray[np.float64],
    reference_probability: NDArray[np.float64],
    labels: NDArray[np.float64],
    event_ids: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    row_delta = _row_log_loss(candidate_probability, labels) - _row_log_loss(
        reference_probability,
        labels,
    )
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, event_id in enumerate(event_ids):
        groups[event_id].append(index)
    group_indices = list(groups.values())
    group_sums = np.asarray(
        [float(row_delta[indices].sum()) for indices in group_indices],
        dtype=np.float64,
    )
    group_counts = np.asarray([len(indices) for indices in group_indices], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(group_indices),
        size=(samples, len(group_indices)),
    )
    sampled = group_sums[draws].sum(axis=1) / group_counts[draws].sum(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "candidate_minus_reference_log_loss": float(row_delta.mean()),
        "confidence_interval_95": [float(lower), float(upper)],
        "event_groups": len(group_indices),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "candidate_better_point_estimate": bool(row_delta.mean() < 0.0),
        "candidate_better_95pct": bool(upper < 0.0),
    }


def _hold_to_resolution(
    probability: NDArray[np.float64],
    labels: NDArray[np.float64],
    market_probability: NDArray[np.float64],
    fractions: list[float],
) -> dict[str, Any]:
    score = np.abs(probability - market_probability)
    choose_yes = probability > market_probability
    share_pnl = np.where(
        choose_yes,
        labels - market_probability,
        market_probability - labels,
    )
    contract_cost = np.where(choose_yes, market_probability, 1.0 - market_probability)
    order = np.argsort(-score, kind="stable")
    output: dict[str, Any] = {}
    for fraction in fractions:
        count = max(1, math.ceil(len(order) * fraction))
        selected = order[:count]
        total_cost = float(contract_cost[selected].sum())
        total_pnl = float(share_pnl[selected].sum())
        output[f"top_{fraction:g}"] = {
            "rows": count,
            "mean_probability_gap": float(score[selected].mean()),
            "yes_side_fraction": float(choose_yes[selected].mean()),
            "positive_share_pnl_fraction": float((share_pnl[selected] > 0.0).mean()),
            "mean_share_pnl": float(share_pnl[selected].mean()),
            "one_share_total_cost": total_cost,
            "one_share_total_pnl": total_pnl,
            "one_share_roi_on_cost": total_pnl / max(total_cost, 1e-12),
        }
    return {
        "basis": "validation_uncalibrated_probability_gap",
        "position_rule": "hold_to_resolution",
        "executable_evidence": False,
        "fractions": output,
    }


def _load_predictions(path: Path) -> dict[str, NDArray[np.float64]]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.asarray(arrays[name], dtype=np.float64) for name in arrays.files}


def _run_variant(
    variant: dict[str, Any],
    research_config_path: Path,
    pack_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    variant_id = str(variant["id"])
    variant_dir = output_root / variant_id
    result_path = variant_dir / "result.json"
    research_hash = sha256_file(research_config_path)
    if result_path.exists():
        existing = load_json(result_path)
        predictions_path = variant_dir / "predictions.npz"
        if (
            existing.get("valid") is True
            and existing.get("research_config_sha256") == research_hash
            and existing.get("variant", {}).get("id") == variant_id
            and predictions_path.exists()
            and existing.get("predictions", {}).get("sha256") == sha256_file(predictions_path)
        ):
            return existing
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(BASE_CONFIG),
        "--pack-dir",
        str(pack_dir),
        "--output-dir",
        str(variant_dir),
        "--wallet-mode",
        str(variant["wallet_mode"]),
        "--outputs",
        *[str(value) for value in variant["outputs"]],
        "--research-config",
        str(research_config_path),
        "--variant-id",
        variant_id,
        "--quiet-result",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return load_json(result_path)


def run_ablation(
    config_path: Path,
    pack_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    if sha256_file(BASE_CONFIG) != config["base_training_contract"]["sha256"]:
        raise RuntimeError("Frozen base training contract hash changed")
    metadata_path = pack_dir / "metadata.json"
    if sha256_file(metadata_path) != config["feature_pack"]["metadata_sha256"]:
        raise RuntimeError("Frozen feature pack hash changed")
    metadata = load_json(metadata_path)
    if int(metadata["test_rows_consumed"]) != 0:
        raise RuntimeError("H007 must not consume test rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "result.json"
    if aggregate_path.exists():
        existing_aggregate = load_json(aggregate_path)
        variant_receipts_match = all(
            (output_dir / str(variant["id"]) / "result.json").exists()
            and existing_aggregate.get("variants", {})
            .get(str(variant["id"]), {})
            .get("result_sha256")
            == sha256_file(output_dir / str(variant["id"]) / "result.json")
            for variant in config["variants"]
        )
        if (
            existing_aggregate.get("valid") is True
            and existing_aggregate.get("config_sha256") == sha256_file(config_path)
            and existing_aggregate.get("feature_pack_sha256") == sha256_file(metadata_path)
            and variant_receipts_match
        ):
            return existing_aggregate
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    for raw_variant in config["variants"]:
        variant = dict(raw_variant)
        variant_id = str(variant["id"])
        result = _run_variant(variant, config_path, pack_dir, output_dir)
        results[variant_id] = result
        predictions[variant_id] = _load_predictions(
            output_dir / variant_id / str(result["predictions"]["path"])
        )

    reference_id = "resolution_causal_wallet_history"
    reference = predictions[reference_id]
    labels = reference["validation_labels"]
    market_probability = reference["validation_market_probability"]
    for variant_id, arrays in predictions.items():
        if not np.array_equal(arrays["validation_labels"], labels):
            raise RuntimeError(f"Validation label order differs for {variant_id}")
        if not np.array_equal(
            arrays["validation_market_probability"],
            market_probability,
        ):
            raise RuntimeError(f"Validation market order differs for {variant_id}")

    example_rows = list(iter_jsonl_zst(pack_dir / "validation" / "examples.jsonl.zst"))
    event_ids = [str(row["event_id"]) for row in example_rows]
    if len(event_ids) != len(labels):
        raise RuntimeError("Validation prediction and example counts differ")
    comparison_config = config["comparison"]
    bootstrap_samples = int(comparison_config["bootstrap_samples"])
    bootstrap_seed = int(comparison_config["bootstrap_seed"])
    reference_probability = _probability(reference["validation_logits"])
    comparisons: dict[str, dict[str, Any]] = {}
    for comparison_index, other_id in enumerate(
        ("resolution_no_wallet_history", "resolution_prior_event_wallet_control")
    ):
        comparisons[f"{reference_id}_vs_{other_id}"] = _bootstrap_delta(
            reference_probability,
            _probability(predictions[other_id]["validation_logits"]),
            labels,
            event_ids,
            samples=bootstrap_samples,
            seed=bootstrap_seed + comparison_index,
        )

    fractions = [float(value) for value in comparison_config["top_score_fractions"]]
    diagnostics = {
        variant_id: _hold_to_resolution(
            _probability(arrays["validation_logits"]),
            labels,
            market_probability,
            fractions,
        )
        for variant_id, arrays in predictions.items()
    }
    parameter_counts = [int(result["model"]["parameters"]) for result in results.values()]
    control_time_violations = sum(
        int(audit["donor_time_violations"])
        for result in results.values()
        for audit in result["control_audit"].values()
    )
    same_event_donors = sum(
        int(audit["same_event_donors"])
        for result in results.values()
        for audit in result["control_audit"].values()
    )
    gates = {
        "all_variant_runs_valid": all(result["valid"] is True for result in results.values()),
        "parameter_count_difference_within_limit": (
            max(parameter_counts) - min(parameter_counts)
            <= int(config["controlled_constants"]["parameter_count_maximum_difference"])
        ),
        "control_donor_time_violations_zero": control_time_violations == 0,
        "control_same_event_donors_zero": same_event_donors == 0,
        "test_rows_consumed_zero": all(
            int(result["test_rows_consumed"]) == 0 for result in results.values()
        ),
        "prediction_order_identical": True,
        "predictions_hashed": all(
            result["predictions"]["sha256"]
            == sha256_file(output_dir / variant_id / "predictions.npz")
            for variant_id, result in results.items()
        ),
    }
    wallet_signal_supported = all(
        comparison["candidate_better_95pct"] is True for comparison in comparisons.values()
    )
    variant_summary = {
        variant_id: {
            "result_sha256": sha256_file(output_dir / variant_id / "result.json"),
            "checkpoint_sha256": result["model"]["checkpoint_sha256"],
            "predictions_sha256": result["predictions"]["sha256"],
            "parameters": result["model"]["parameters"],
            "best_epoch": result["training"]["best_epoch"],
            "elapsed_seconds": result["training"]["elapsed_seconds"],
            "validation": result["metrics"]["validation"]["resolved_yes"],
            "calibration": result["metrics"]["calibration"]["resolved_yes"],
            "temperature": result["training"]["temperature"],
            "control_audit": result["control_audit"],
        }
        for variant_id, result in results.items()
    }
    aggregate: dict[str, Any] = {
        "schema_version": "1.0.0",
        "completed_at": now_utc(),
        "valid": all(gates.values()),
        "research_id": str(config["research_id"]),
        "config_id": str(config["id"]),
        "config_sha256": sha256_file(config_path),
        "feature_pack_sha256": sha256_file(metadata_path),
        "test_labels_opened": False,
        "test_rows_consumed": 0,
        "primary_metric": str(comparison_config["primary_metric"]),
        "variants": variant_summary,
        "comparisons": comparisons,
        "wallet_signal_supported": wallet_signal_supported,
        "hold_to_resolution_diagnostics": diagnostics,
        "gates": gates,
        "decision": "diagnostic_only_no_promotion",
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(aggregate_path, aggregate)
    if not aggregate["valid"]:
        raise RuntimeError(f"H007 gates failed; see {output_dir / 'result.json'}")
    return aggregate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--pack-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = run_ablation(
        args.config.resolve(),
        args.pack_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
