"""Build the exact fill-realized H021 CALL candidate pack for H023."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.h022_features import assemble_tree_features, component_folds
from sphinx_trace.h023_labels import H023RealizedLabel, realized_decision_labels
from sphinx_trace.policy_decisions import (
    PolicyDecisionRef,
    PolicyFeatureStore,
    load_policy_decisions,
)
from sphinx_trace.policy_encodings import PolicyEncodingStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h023_experiment_v1.json"
DEFAULT_REGISTRATION = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h023_fill_realized_veto_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h023_labels.py",
    ROOT / "src" / "sphinx_trace" / "h022_features.py",
    ROOT / "src" / "sphinx_trace" / "policy_decisions.py",
    ROOT / "src" / "sphinx_trace" / "policy_encodings.py",
)
H022_MEMBER_FEATURE_NAMES = (
    "h022.neural_mean_net_return",
    "h022.neural_q10",
    "h022.neural_q50",
    "h022.neural_q90",
    "h022.neural_fill_probability",
    "h022.neural_calibrated_probability0",
    "h022.neural_calibrated_candidate_edge",
    "h022.tree_net_return",
    "h022.ensemble_net_return",
    "h022.keep_base_call",
    "h021.position_size_fraction",
)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _atomic_npy(path: Path, values: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_entry(path: Path, values: NDArray[Any], root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _load_replay(replay_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    result_path = replay_dir / "result.json"
    manifest_path = replay_dir / "manifest.json"
    result = load_json(result_path)
    manifest = load_json(manifest_path)
    dependency = config["dependencies"]["calibration_shadow_replay"]
    if (
        result.get("record_type") != "h010_policy_replay_result"
        or result.get("valid") is not True
        or result.get("split") != "calibration"
        or result.get("h022_mode") != "shadow"
        or result.get("platform_fee_model") != "receipt_qualified_historical"
        or float(result.get("cost_multiplier", -1.0)) != 1.0
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or sha256_file(result_path) != dependency["result_sha256"]
        or sha256_file(manifest_path) != dependency["manifest_sha256"]
        or result.get("audit_manifest_sha256") != dependency["manifest_sha256"]
        or manifest.get("valid") is not True
    ):
        raise RuntimeError("H023 requires the registered exact calibration shadow replay")
    return result


def _audit_rows(
    replay_dir: Path,
    decision_records: dict[str, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for shard in sorted((replay_dir / "shards").glob("date=*.jsonl.zst")):
        for row in iter_jsonl_zst(shard):
            if row.get("record_type") == "h010_decision_audit" and row.get("h022"):
                decision_id = str(row["decision_id"])
                if decision_id in decision_records:
                    raise RuntimeError(f"H023 decision audit repeats: {decision_id}")
                decision_records[decision_id] = row
            yield row


def _decision_refs(pack_dir: Path) -> tuple[dict[str, PolicyDecisionRef], tuple[Path, ...]]:
    by_evidence, shards = load_policy_decisions(pack_dir, splits=("calibration",))
    refs: dict[str, PolicyDecisionRef] = {}
    for candidates in by_evidence.values():
        for ref in candidates:
            if ref.decision_id in refs:
                raise RuntimeError(f"H023 policy decision repeats: {ref.decision_id}")
            refs[ref.decision_id] = ref
    return refs, shards


def _h022_features(row: dict[str, Any]) -> NDArray[np.float32]:
    debug = row["h022"]
    member = debug["member_scores"]
    quantiles = debug["net_return_quantiles"]
    values = np.asarray(
        (
            member["neural_mean_net_return"],
            quantiles[0],
            quantiles[1],
            quantiles[2],
            debug["fill_probability"],
            debug["calibrated_probability0"],
            debug["calibrated_candidate_edge"],
            member["tree_net_return"],
            member["ensemble_net_return"],
            float(bool(debug["keep_base_call"])),
            row["size_fraction"],
        ),
        dtype=np.float32,
    )
    if values.shape != (len(H022_MEMBER_FEATURE_NAMES),) or not bool(
        np.isfinite(values).all()
    ):
        raise RuntimeError("H023 H022 member feature contract changed")
    return values


def _summary(
    labels: list[H023RealizedLabel],
    entry_prices: NDArray[np.float32],
    component_ids: NDArray[np.int64],
) -> dict[str, Any]:
    pnl = np.asarray([float(value.realized_pnl_usd) for value in labels])
    filled = np.asarray([value.fill_count > 0 for value in labels])
    high = entry_prices >= 0.8
    return {
        "rows": len(labels),
        "components": int(np.unique(component_ids).size),
        "filled_decisions": int(filled.sum()),
        "unfilled_decisions": int((~filled).sum()),
        "profitable_decisions": int((pnl > 0.0).sum()),
        "harmful_decisions": int((pnl < 0.0).sum()),
        "zero_contribution_decisions": int((pnl == 0.0).sum()),
        "realized_pnl_usd": float(pnl.sum()),
        "entry_price_at_least_0_80_rows": int(high.sum()),
        "entry_price_at_least_0_80_realized_pnl_usd": float(pnl[high].sum()),
        "entry_price_below_0_80_realized_pnl_usd": float(pnl[~high].sum()),
    }


def build(
    config_path: Path,
    registration_path: Path,
    replay_dir: Path,
    pack_dir: Path,
    encoding_dir: Path,
    policy_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    dependency = config.get("dependencies", {})
    if (
        config.get("research_id") != "SPH-T-H023"
        or sha256_file(registration_path) != dependency.get("registration_sha256")
        or sha256_file(pack_dir / "manifest.json")
        != dependency.get("feature_pack_manifest_sha256")
        or sha256_file(encoding_dir / "manifest.json")
        != dependency.get("encoding_manifest_sha256")
        or sha256_file(policy_dir / "result.json")
        != dependency.get("H021_policy_result_sha256")
    ):
        raise RuntimeError("H023 realized-pack dependency receipt changed")
    existing = output_dir / "manifest.json"
    if existing.is_file():
        payload = load_json(existing)
        if payload.get("valid") is True:
            return payload
        raise RuntimeError("H023 output directory contains an invalid manifest")
    replay_result = _load_replay(replay_dir, config)
    decision_records: dict[str, dict[str, Any]] = {}
    labels_by_id, audit_counts = realized_decision_labels(
        _audit_rows(replay_dir, decision_records)
    )
    if set(labels_by_id) != set(decision_records):
        raise RuntimeError("H023 realized labels do not cover every shadow candidate")
    refs, shards = _decision_refs(pack_dir)
    missing_refs = set(labels_by_id) - refs.keys()
    if missing_refs:
        raise RuntimeError(f"H023 feature pack misses {len(missing_refs)} candidates")
    feature_store = PolicyFeatureStore(
        pack_dir,
        shards,
        feature_clip=float(config["candidate_pack"]["feature_clip_after_normalization"]),
    )
    encoding_store = PolicyEncodingStore(
        encoding_dir,
        pack_dir,
        policy_dir,
        shards,
    )
    ordered = sorted(
        labels_by_id,
        key=lambda decision_id: (
            refs[decision_id].timestamp_unix,
            decision_id,
        ),
    )
    labels = [labels_by_id[decision_id] for decision_id in ordered]
    rows = len(labels)
    tree_features = np.empty((rows, 170), dtype=np.float32)
    market_latents = np.empty((rows, encoding_store.width), dtype=np.float16)
    h022_features = np.empty((rows, len(H022_MEMBER_FEATURE_NAMES)), dtype=np.float32)
    candidate_action_ids = np.empty(rows, dtype=np.uint8)
    component_ids = np.empty(rows, dtype=np.int64)
    market_ids = np.empty(rows, dtype=np.int64)
    week_ids = np.empty(rows, dtype=np.int64)
    timestamps = np.empty(rows, dtype=np.int64)
    feature_rows = np.empty(rows, dtype=np.int64)
    entry_prices = np.empty(rows, dtype=np.float32)
    target_pnl_usd = np.empty(rows, dtype=np.float64)
    target_contribution = np.empty(rows, dtype=np.float32)
    target_return_requested = np.empty(rows, dtype=np.float32)
    target_return_filled = np.empty(rows, dtype=np.float32)
    target_fill_fraction = np.empty(rows, dtype=np.float32)
    requested_cost = np.empty(rows, dtype=np.float64)
    filled_cost = np.empty(rows, dtype=np.float64)
    terminal_payout = np.empty(rows, dtype=np.float64)
    collateral_fees = np.empty(rows, dtype=np.float64)
    outcome_fee_shares = np.empty(rows, dtype=np.float64)
    initial_cash = float(replay_result["metrics"]["initial_cash_usd"])
    if initial_cash <= 0.0:
        raise RuntimeError("H023 replay initial cash is invalid")
    for index, decision_id in enumerate(ordered):
        ref = refs[decision_id]
        row = decision_records[decision_id]
        label = labels_by_id[decision_id]
        loaded = feature_store.load(ref)
        encoded = encoding_store.load(ref)
        h021 = row["h021"]
        action_id = int(row["h022"]["candidate_action_id"])
        assembled = assemble_tree_features(
            np.asarray(loaded.normalized, dtype=np.float32)[None, :],
            np.asarray([encoded.terminal_outcome_logit], dtype=np.float32),
            np.asarray([encoded.uncertainty_log_scale], dtype=np.float32),
            np.asarray([row["portfolio_features"]], dtype=np.float32),
            np.asarray([row["prediction_memory_features"]], dtype=np.float32),
            np.asarray([h021["base_action_logits"]], dtype=np.float32),
            np.asarray([h021["protocol_action_values"]], dtype=np.float32),
            np.asarray([h021["execution_context"]], dtype=np.float32),
            np.asarray([action_id], dtype=np.int64),
        )
        if row["feature_ref"] != {"date": ref.feature_date, "row": ref.feature_row}:
            raise RuntimeError("H023 audit feature reference changed")
        tree_features[index] = assembled[0]
        market_latents[index] = encoded.market_latent.astype(np.float16)
        h022_features[index] = _h022_features(row)
        candidate_action_ids[index] = action_id
        component_ids[index] = ref.component_state_id
        market_ids[index] = ref.market_state_id
        week_ids[index] = ref.timestamp_unix // (7 * 24 * 60 * 60)
        timestamps[index] = ref.timestamp_unix
        feature_rows[index] = ref.feature_row
        entry_prices[index] = float(h021["execution_context"][action_id])
        target_pnl_usd[index] = float(label.realized_pnl_usd)
        target_contribution[index] = float(label.realized_pnl_usd) / initial_cash
        target_return_requested[index] = float(label.realized_return_on_requested_cost)
        target_return_filled[index] = float(label.realized_return_on_filled_cost)
        target_fill_fraction[index] = float(label.fill_fraction)
        requested_cost[index] = float(label.requested_total_cost_usd)
        filled_cost[index] = float(label.actual_filled_total_cost_usd)
        terminal_payout[index] = float(label.terminal_payout_usd)
        collateral_fees[index] = float(label.collateral_fee_usd)
        outcome_fee_shares[index] = float(label.outcome_token_fee_shares)
        if index and index % 10_000 == 0:
            atomic_json(
                output_dir / "progress.json",
                {
                    "record_type": "h023_realized_candidate_pack_progress",
                    "rows_complete": index,
                    "rows_total": rows,
                },
            )
    replay_pnl = float(replay_result["metrics"]["net_profit_usd"])
    attributed_pnl = float(target_pnl_usd.sum())
    if not np.isclose(attributed_pnl, replay_pnl, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            "H023 decision attribution does not reproduce replay PnL: "
            f"{attributed_pnl} != {replay_pnl}"
        )
    folds = int(config["candidate_pack"]["component_partition_folds"])
    selection_fold = int(config["candidate_pack"]["selection_fold"])
    fold_codes = component_folds(
        component_ids,
        folds,
        int(config["candidate_pack"]["component_partition_seed"]),
    )
    selection_mask = fold_codes == selection_fold
    if not bool(selection_mask.any()) or bool(selection_mask.all()):
        raise RuntimeError("H023 component partition is empty")
    arrays: dict[str, NDArray[Any]] = {
        "tree_features": tree_features,
        "market_latents": market_latents,
        "h022_member_features": h022_features,
        "candidate_action_ids": candidate_action_ids,
        "component_ids": component_ids,
        "market_ids": market_ids,
        "week_ids": week_ids,
        "timestamps": timestamps,
        "feature_rows": feature_rows,
        "entry_prices": entry_prices,
        "target_realized_pnl_usd": target_pnl_usd,
        "target_realized_net_contribution": target_contribution,
        "target_return_on_requested_cost": target_return_requested,
        "target_return_on_filled_cost": target_return_filled,
        "target_fill_fraction": target_fill_fraction,
        "requested_total_cost_usd": requested_cost,
        "actual_filled_total_cost_usd": filled_cost,
        "terminal_payout_usd": terminal_payout,
        "collateral_fee_usd": collateral_fees,
        "outcome_token_fee_shares": outcome_fee_shares,
        "decision_ids": np.asarray(ordered, dtype="S64"),
    }
    files: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for name, mask in (("fit", ~selection_mask), ("selection", selection_mask)):
        partition_labels = [label for label, keep in zip(labels, mask, strict=True) if keep]
        partition_arrays = {key: value[mask] for key, value in arrays.items()}
        for key, values in partition_arrays.items():
            path = output_dir / name / f"{key}.npy"
            _atomic_npy(path, values)
            files[f"{name}/{key}.npy"] = _array_entry(path, values, output_dir)
        summaries[name] = _summary(
            partition_labels,
            partition_arrays["entry_prices"],
            partition_arrays["component_ids"],
        )
    manifest = {
        "record_type": "h023_fill_realized_candidate_pack_manifest",
        "schema_version": "1.0.0",
        "dataset_id": "sphinx-h023-fill-realized-candidate-pack-v1",
        "research_id": "SPH-T-H023",
        "valid": True,
        "generated_at": now_utc(),
        "config_sha256": sha256_file(config_path),
        "registration_sha256": sha256_file(registration_path),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "replay_manifest_sha256": sha256_file(replay_dir / "manifest.json"),
        "fee_schedule_manifest_sha256": replay_result["fee_schedule_manifest_sha256"],
        "feature_pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "encoding_manifest_sha256": sha256_file(encoding_dir / "manifest.json"),
        "policy_result_sha256": sha256_file(policy_dir / "result.json"),
        "implementation_sha256": _implementation_digest(),
        "audit_counts": audit_counts,
        "partitions": summaries,
        "total": _summary(labels, entry_prices, component_ids),
        "attributed_replay_pnl_usd": attributed_pnl,
        "expected_replay_pnl_usd": replay_pnl,
        "h022_member_feature_names": list(H022_MEMBER_FEATURE_NAMES),
        "files": files,
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": int(replay_result["metrics"]["predictions"]),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Exact development calibration shadow replay with receipt-qualified fees; "
            "validation replay, untouched test and paper-forward evidence remain closed."
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    (output_dir / "progress.json").unlink(missing_ok=True)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    value.add_argument("--replay-dir", type=Path, required=True)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--encoding-dir", type=Path, required=True)
    value.add_argument("--policy-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = build(
        args.config.resolve(),
        args.registration.resolve(),
        args.replay_dir.resolve(),
        args.pack_dir.resolve(),
        args.encoding_dir.resolve(),
        args.policy_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
