"""Build the receipt-bound frozen-H021 candidate pack for H022."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.h022_features import (
    H022_TREE_FEATURE_NAMES,
    assemble_tree_features,
    selected_reference_utility,
)
from sphinx_trace.policy_checkpoint import load_policy_checkpoint

try:
    from scripts.train_h015_portfolio_advantage import (
        _batch,
        _equal_market_weights,
        _indices,
        _sample_weights,
    )
    from scripts.train_h018_conservative_residual_policy import (
        _execution_context,
        _forward_adapter,
        _state_shards,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from train_h015_portfolio_advantage import (  # type: ignore[import-not-found,no-redef]
        _batch,
        _equal_market_weights,
        _indices,
        _sample_weights,
    )
    from train_h018_conservative_residual_policy import (  # type: ignore[import-not-found,no-redef]
        _execution_context,
        _forward_adapter,
        _state_shards,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h022_experiment_v1.json"
)
DEFAULT_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h021_policy_v1.json"
)
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h022_features.py",
    ROOT / "src" / "sphinx_trace" / "model_h021.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "scripts" / "train_h015_portfolio_advantage.py",
    ROOT / "scripts" / "train_h018_conservative_residual_policy.py",
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


def _normalized_market_features(
    pack_shard: Path,
    row_indices: NDArray[np.int64],
    median: NDArray[np.float32],
    scale: NDArray[np.float32],
    clip: float,
) -> NDArray[np.float32]:
    raw = np.load(pack_shard / "features.npy", mmap_mode="r", allow_pickle=False)
    values = np.asarray(raw[row_indices], dtype=np.float32)
    values = (values - median) / scale
    np.clip(values, -clip, clip, out=values)
    if values.shape != (len(row_indices), 128) or not bool(np.isfinite(values).all()):
        raise RuntimeError("H022 normalized causal features are invalid")
    return values


def build(
    config_path: Path,
    policy_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    state_dir: Path,
    encoding_dir: Path,
    pack_dir: Path,
    initial_policy_dir: Path,
    outcome_dir: Path,
    output_dir: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    if config.get("research_id") != "SPH-T-H022" or batch_size <= 0:
        raise RuntimeError("H022 candidate-pack contract is invalid")
    dependency = config["dependencies"]
    if (
        sha256_file(state_dir / "manifest.json")
        != dependency["protocol_tail_pack"]["manifest_sha256"]
        or sha256_file(pack_dir / "manifest.json")
        != dependency["feature_pack_manifest_sha256"]
        or sha256_file(encoding_dir / "manifest.json")
        != dependency["encoding_manifest_sha256"]
        or sha256_file(initial_policy_dir / "result.json")
        != dependency["initial_policy"]["result_sha256"]
        or sha256_file(initial_policy_dir / "best-policy.pt")
        != dependency["initial_policy"]["best_model_sha256"]
        or sha256_file(outcome_dir / "result.json")
        != dependency["outcome_model"]["result_sha256"]
    ):
        raise RuntimeError("H022 candidate-pack source receipt changed")

    existing = output_dir / "manifest.json"
    if existing.is_file():
        payload: object = json.loads(existing.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("valid") is True:
            return payload
        raise RuntimeError("H022 output directory contains an invalid manifest")

    policy_config = load_json(policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_policy_checkpoint(
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        policy_config,
        device,
    )
    model = loaded.model.eval()
    shards, source_manifest = _state_shards(
        state_dir, encoding_dir, pack_dir, initial_policy_dir, config
    )
    market_weights, weighting_receipt = _equal_market_weights(shards, 2)
    with np.load(pack_dir / "normalization.npz", allow_pickle=False) as archive:
        median = np.asarray(archive["median"], dtype=np.float32)
        scale = np.asarray(archive["scale"], dtype=np.float32)
    clip = float(config["candidate_pack"]["feature_clip_after_normalization"])

    partition_names = {0: "fit", 1: "selection"}
    arrays: dict[int, dict[str, list[NDArray[Any]]]] = {
        code: {
            "tree_features": [],
            "market_latents": [],
            "target_net_log_utility": [],
            "target_reference_net_log_utility": [],
            "position_size_fractions": [],
            "target_outcome0": [],
            "target_fill_fraction": [],
            "fill_target_mask": [],
            "sample_weights": [],
            "candidate_action_ids": [],
            "component_ids": [],
            "market_ids": [],
            "week_ids": [],
            "behavior_policy_codes": [],
            "entry_prices": [],
            "timestamps": [],
        }
        for code in partition_names
    }
    observed_rows = {0: 0, 1: 0}
    candidate_rows = {0: 0, 1: 0}
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.inference_mode():
        for shard_index, shard in enumerate(shards):
            state_rows = np.load(
                shard.state / "row_indices.npy", mmap_mode="r", allow_pickle=False
            )
            timestamps = np.load(
                shard.state / "timestamps.npy", mmap_mode="r", allow_pickle=False
            )
            for partition_code in partition_names:
                indices = _indices(
                    shard, partition_code, seed=0, epoch=0, shuffle=False
                )
                observed_rows[partition_code] += len(indices)
                for offset in range(0, len(indices), batch_size):
                    selected = indices[offset : offset + batch_size]
                    batch = _batch(shard, selected)
                    with autocast:
                        output, _, _ = _forward_adapter(model, batch, device)
                    action_ids = (
                        output["action_logits"][:, :3]
                        .float()
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    keep = action_ids != 2
                    if not bool(keep.any()):
                        continue
                    kept = np.flatnonzero(keep).astype(np.int64)
                    candidate = action_ids[kept]
                    if (
                        batch.reference_action_values is None
                        or batch.week_ids is None
                        or batch.winning_payout_multipliers is None
                        or batch.entry_prices is None
                    ):
                        raise RuntimeError("H022 requires protocol-exact H017 targets")
                    execution = _execution_context(batch, device).cpu().numpy()
                    source_row_indices = np.asarray(
                        state_rows[selected], dtype=np.int64
                    )
                    market_features = _normalized_market_features(
                        shard.pack, source_row_indices[kept], median, scale, clip
                    )
                    base_logits = (
                        output["base_action_logits"][kept].float().cpu().numpy()
                    )
                    protocol_values = (
                        output["protocol_action_values"][kept].float().cpu().numpy()
                    )
                    tree_features = assemble_tree_features(
                        market_features,
                        batch.terminal_logits[kept],
                        batch.uncertainty_log_scales[kept],
                        batch.portfolio_features[kept],
                        batch.prediction_memory_features[kept],
                        base_logits.astype(np.float32),
                        protocol_values.astype(np.float32),
                        execution[kept].astype(np.float32),
                        candidate,
                    )
                    target_utility = selected_reference_utility(
                        batch.reference_action_values[kept], candidate
                    )
                    alpha = (
                        output["position_size_beta_alpha"][kept]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    beta = (
                        output["position_size_beta_beta"][kept]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    size = alpha / np.maximum(alpha + beta, 1e-8)
                    candidate_payout = np.take_along_axis(
                        batch.winning_payout_multipliers[kept],
                        candidate[:, None],
                        axis=1,
                    )[:, 0]
                    candidate_wins = (
                        (candidate == 0) & (batch.labels[kept] == 1.0)
                    ) | ((candidate == 1) & (batch.labels[kept] == 0.0))
                    actual_utility = np.log(
                        np.maximum(
                            1.0
                            - size
                            + size * candidate_wins.astype(np.float32) * candidate_payout,
                            1e-8,
                        )
                    ).astype(np.float32)
                    matching_logged_action = (
                        batch.behavior_action_ids[kept] == candidate
                    )
                    selected_prices = np.take_along_axis(
                        batch.entry_prices[kept], candidate[:, None], axis=1
                    )[:, 0]
                    weights = _sample_weights(
                        batch, market_weights, partition_code
                    )[kept]
                    target = arrays[partition_code]
                    target["tree_features"].append(tree_features)
                    target["market_latents"].append(
                        batch.market_latents[kept].astype(np.float16)
                    )
                    target["target_net_log_utility"].append(actual_utility)
                    target["target_reference_net_log_utility"].append(target_utility)
                    target["position_size_fractions"].append(size.astype(np.float32))
                    target["target_outcome0"].append(batch.labels[kept])
                    target["target_fill_fraction"].append(
                        batch.execution_fractions[kept]
                    )
                    target["fill_target_mask"].append(
                        matching_logged_action.astype(np.uint8)
                    )
                    target["sample_weights"].append(weights)
                    target["candidate_action_ids"].append(candidate.astype(np.uint8))
                    target["component_ids"].append(batch.component_ids[kept])
                    target["market_ids"].append(batch.market_ids[kept])
                    target["week_ids"].append(batch.week_ids[kept])
                    target["behavior_policy_codes"].append(
                        batch.behavior_policy_codes[kept]
                    )
                    target["entry_prices"].append(selected_prices.astype(np.float32))
                    target["timestamps"].append(
                        np.asarray(timestamps[selected[kept]], dtype=np.int64)
                    )
                    candidate_rows[partition_code] += len(kept)
            if shard_index % 50 == 0:
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "record_type": "h022_candidate_pack_progress",
                        "shards_complete": shard_index + 1,
                        "shards_total": len(shards),
                        "fit_candidates": candidate_rows[0],
                        "selection_candidates": candidate_rows[1],
                    },
                )

    files: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for partition_code, partition_name in partition_names.items():
        partition_dir = output_dir / partition_name
        written: dict[str, NDArray[Any]] = {}
        for name, parts in arrays[partition_code].items():
            if not parts:
                raise RuntimeError(f"H022 {partition_name} candidate array is empty: {name}")
            values = np.concatenate(parts, axis=0)
            if len(values) != candidate_rows[partition_code]:
                raise RuntimeError(f"H022 {partition_name} candidate arrays misalign")
            path = partition_dir / f"{name}.npy"
            _atomic_npy(path, values)
            written[name] = values
            files[f"{partition_name}/{name}.npy"] = _array_entry(
                path, values, output_dir
            )
        utility = written["target_net_log_utility"].astype(np.float64)
        reference_utility = written[
            "target_reference_net_log_utility"
        ].astype(np.float64)
        prices = written["entry_prices"].astype(np.float64)
        partition_source_rows = observed_rows[partition_code]
        reproduced = float(
            (utility * written["sample_weights"].astype(np.float64)).sum()
            / partition_source_rows
        )
        source_result = loaded.result
        expected = float(
            (
                source_result["fit"]
                if partition_name == "fit"
                else source_result["initial_selection"]
            )["equal_market_mean_protocol_exact_chosen_utility"]
        )
        if not np.isclose(reproduced, expected, rtol=1e-4, atol=1e-10):
            raise RuntimeError(
                f"H022 {partition_name} actual-size utility does not reproduce H021: "
                f"{reproduced} != {expected}"
            )
        summaries[partition_name] = {
            "source_rows": partition_source_rows,
            "candidate_rows": candidate_rows[partition_code],
            "components": int(np.unique(written["component_ids"]).size),
            "markets": int(np.unique(written["market_ids"]).size),
            "logged_fill_targets": int(written["fill_target_mask"].sum()),
            "profitable_candidates": int((utility > 0.0).sum()),
            "harmful_candidates": int((utility <= 0.0).sum()),
            "mean_actual_size_net_log_utility": float(utility.mean()),
            "mean_fixed_0_05_reference_net_log_utility": float(
                reference_utility.mean()
            ),
            "mean_position_size_fraction": float(
                written["position_size_fractions"].mean()
            ),
            "reproduced_H021_equal_market_chosen_utility": reproduced,
            "expected_H021_equal_market_chosen_utility": expected,
            "mean_entry_price": float(prices.mean()),
            "price_at_least_0_80_rows": int((prices >= 0.8).sum()),
            "price_at_least_0_80_mean_utility": float(
                utility[prices >= 0.8].mean() if bool((prices >= 0.8).any()) else 0.0
            ),
        }

    manifest = {
        "record_type": "h022_frozen_candidate_pack_manifest",
        "schema_version": "1.0.0",
        "dataset_id": "sphinx-h022-frozen-h021-candidate-pack-v2",
        "research_id": "SPH-T-H022",
        "generated_at": now_utc(),
        "valid": True,
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": _implementation_digest(),
        "source_manifest_sha256": sha256_file(state_dir / "manifest.json"),
        "feature_pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "encoding_manifest_sha256": sha256_file(encoding_dir / "manifest.json"),
        "initial_policy_result_sha256": sha256_file(
            initial_policy_dir / "result.json"
        ),
        "initial_policy_best_model_sha256": sha256_file(
            initial_policy_dir / "best-policy.pt"
        ),
        "source_partition_sha256": source_manifest["partition_sha256"],
        "tree_feature_names": list(H022_TREE_FEATURE_NAMES),
        "tree_feature_width": len(H022_TREE_FEATURE_NAMES),
        "market_latent_width": 512,
        "partitions": summaries,
        "weighting": weighting_receipt,
        "files": files,
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Frozen H021 candidate states from H017 fit/selection only; this pack is "
            "not replay or profit evidence."
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    (output_dir / "progress.json").unlink(missing_ok=True)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--state-dir", type=Path, required=True)
    value.add_argument("--encoding-dir", type=Path, required=True)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--initial-policy-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--batch-size", type=int, default=4096)
    return value


def main() -> None:
    args = parser().parse_args()
    result = build(
        args.config.resolve(),
        args.policy_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.state_dir.resolve(),
        args.encoding_dir.resolve(),
        args.pack_dir.resolve(),
        args.initial_policy_dir.resolve(),
        args.outcome_dir.resolve(),
        args.output_dir.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(result["partitions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
