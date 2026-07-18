"""Warm-start the H012 learned selective policy on causal development blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h011 import (
    SphinxTraceS0H011,
)
from sphinx_trace.model_h012 import H012_ACTION_COUNT, SphinxTraceS0H012
from sphinx_trace.model_h013 import SphinxTraceS0H013
from sphinx_trace.policy_checkpoint import load_outcome_checkpoint
from sphinx_trace.policy_training import (
    ComponentTimePartition,
    component_time_partition,
    selective_log_utility_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_train_v1.json"
DEFAULT_POLICY_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "policy_training.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


@dataclass(frozen=True, slots=True)
class PackShard:
    date: str
    root: Path


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pack_shards(pack_dir: Path) -> list[PackShard]:
    shards = [
        PackShard(path.name.removeprefix("date="), path)
        for path in sorted((pack_dir / "shards").glob("date=*"))
        if path.is_dir()
    ]
    if not shards:
        raise RuntimeError("H012 training found no feature shards")
    return shards


def _source_digest(pack_dir: Path, shards: list[PackShard]) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(pack_dir / "manifest.json").encode())
    digest.update(sha256_file(pack_dir / "normalization.npz").encode())
    for shard in shards:
        receipt = pack_dir / "receipts" / f"date={shard.date}.json"
        digest.update(f"{shard.date}:{sha256_file(receipt)}\n".encode())
    return digest.hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _partition(
    shards: list[PackShard],
    fit_fraction: float,
) -> ComponentTimePartition:
    components: list[NDArray[np.int64]] = []
    timestamps: list[NDArray[np.int64]] = []
    for shard in shards:
        split = np.load(shard.root / "split_codes.npy", mmap_mode="r")
        mask = np.load(shard.root / "label_mask.npy", mmap_mode="r")
        selected = np.flatnonzero((split == 2) & (mask == 1))
        if not len(selected):
            continue
        component = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        timestamp = np.load(shard.root / "timestamps.npy", mmap_mode="r")
        components.append(np.asarray(component[selected], dtype=np.int64))
        timestamps.append(np.asarray(timestamp[selected], dtype=np.int64))
    if not components:
        raise RuntimeError("H012 training found no qualified validation rows")
    return component_time_partition(
        np.concatenate(components),
        np.concatenate(timestamps),
        fit_fraction,
    )


def _partition_digest(partition: ComponentTimePartition) -> str:
    digest = hashlib.sha256()
    digest.update(partition.fit_components.tobytes())
    digest.update(partition.selection_components.tobytes())
    digest.update(str(partition.cutoff_unix).encode())
    return digest.hexdigest()


def _indices(
    shard: PackShard,
    split_code: int,
    allowed_components: NDArray[np.int64] | None,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> NDArray[np.int64]:
    split = np.load(shard.root / "split_codes.npy", mmap_mode="r")
    mask = np.load(shard.root / "label_mask.npy", mmap_mode="r")
    selected = (split == split_code) & (mask == 1)
    if allowed_components is not None:
        components = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        selected &= np.isin(components, allowed_components, assume_unique=False)
    indices = np.flatnonzero(selected).astype(np.int64)
    if shuffle and len(indices):
        date_seed = int(hashlib.sha256(shard.date.encode()).hexdigest()[:16], 16)
        np.random.default_rng(seed ^ epoch ^ date_seed).shuffle(indices)
    return indices


def _component_weights(
    shards: list[PackShard],
    components: NDArray[np.int64],
) -> NDArray[np.float32]:
    maximum = max(int(components.max()), 0)
    counts = np.zeros(maximum + 1, dtype=np.int64)
    for shard in shards:
        indices = _indices(shard, 2, components, seed=0, epoch=0, shuffle=False)
        if not len(indices):
            continue
        values = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        counts += np.bincount(np.asarray(values[indices], dtype=np.int64), minlength=len(counts))
    observed = counts > 0
    if not bool(observed.any()):
        raise RuntimeError("H012 component balancing found no fit rows")
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[observed] = 1.0 / np.sqrt(counts[observed])
    normalization = np.average(weights[observed], weights=counts[observed])
    weights[observed] /= max(float(normalization), np.finfo(np.float64).tiny)
    return weights.astype(np.float32)


def _load_outcome(
    outcome_dir: Path,
    model_config: dict[str, Any],
    residual_config: dict[str, Any],
    device: torch.device,
) -> tuple[SphinxTraceS0H011 | SphinxTraceS0H013, Tensor, Tensor, dict[str, Any]]:
    checkpoint = load_outcome_checkpoint(
        outcome_dir,
        model_config,
        residual_config,
        device,
    )
    return checkpoint.model, checkpoint.feature_mask, checkpoint.group_mask, checkpoint.result


def _states(rows: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    portfolio = torch.zeros((rows, 9), dtype=torch.float32, device=device)
    portfolio[:, 0:2] = 1.0
    memory = torch.zeros((rows, 7), dtype=torch.float32, device=device)
    memory[:, 0] = 0.5
    previous = torch.full((rows,), 2, dtype=torch.long, device=device)
    physical = torch.zeros((rows, H012_ACTION_COUNT), dtype=torch.bool, device=device)
    physical[:, :3] = True
    return portfolio, memory, previous, physical


def _batch(
    shard: PackShard,
    indices: NDArray[np.int64],
    median: NDArray[np.float32],
    scale: NDArray[np.float32],
    feature_clip: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    raw = np.load(shard.root / "features.npy", mmap_mode="r")
    labels = np.load(shard.root / "labels.npy", mmap_mode="r")
    baselines = np.load(shard.root / "baselines.npy", mmap_mode="r")
    components = np.load(shard.root / "component_ids.npy", mmap_mode="r")
    features = (np.asarray(raw[indices], dtype=np.float32) - median) / scale
    np.clip(features, -feature_clip, feature_clip, out=features)
    return (
        features,
        np.asarray(labels[indices], dtype=np.float32),
        np.asarray(baselines[indices], dtype=np.float32),
        np.asarray(components[indices], dtype=np.int64),
    )


def _forward(
    model: SphinxTraceS0H012,
    features: NDArray[np.float32],
    baselines: NDArray[np.float32],
    feature_mask: Tensor,
    group_mask: Tensor,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    feature_tensor = torch.from_numpy(features).to(device, non_blocking=True) * feature_mask
    baseline_tensor = torch.from_numpy(baselines).to(device, non_blocking=True)
    portfolio, memory, previous, physical = _states(len(features), device)
    output = model(
        feature_tensor,
        portfolio,
        memory,
        previous,
        market_probability=baseline_tensor,
        market_group_mask=group_mask.expand(len(features), -1),
        physical_action_mask=physical,
    )
    return output, feature_tensor, baseline_tensor


@torch.inference_mode()
def _evaluate(
    model: SphinxTraceS0H012,
    shards: list[PackShard],
    split_code: int,
    allowed_components: NDArray[np.int64] | None,
    median: NDArray[np.float32],
    scale: NDArray[np.float32],
    feature_clip: float,
    feature_mask: Tensor,
    group_mask: Tensor,
    utility_config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    totals = {
        "rows": 0.0,
        "loss": 0.0,
        "chosen_log_utility": 0.0,
        "expected_log_utility": 0.0,
        "mean_size": 0.0,
        "calls": 0.0,
        "correct_calls": 0.0,
        "action_value_loss": 0.0,
        "mean_call_probability": 0.0,
        "positive_call_value_fraction": 0.0,
        "mean_selected_action_value": 0.0,
    }
    for shard in shards:
        indices = _indices(
            shard,
            split_code,
            allowed_components,
            seed=0,
            epoch=0,
            shuffle=False,
        )
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            features, labels, baselines, _ = _batch(shard, selected, median, scale, feature_clip)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, _, baseline_tensor = _forward(
                    model, features, baselines, feature_mask, group_mask, device
                )
                loss, metrics = selective_log_utility_loss(
                    output,
                    torch.from_numpy(labels).to(device, non_blocking=True),
                    baseline_tensor,
                    utility_config,
                )
            rows = len(selected)
            totals["rows"] += rows
            totals["loss"] += float(loss) * rows
            totals["chosen_log_utility"] += float(metrics["chosen_log_utility_sum"])
            totals["expected_log_utility"] += float(metrics["expected_log_utility"]) * rows
            totals["mean_size"] += float(metrics["mean_size"]) * rows
            totals["calls"] += float(metrics["call_count"])
            totals["correct_calls"] += float(metrics["correct_call_count"])
            totals["action_value_loss"] += float(metrics["action_value_loss"]) * rows
            totals["mean_call_probability"] += float(metrics["mean_call_probability"]) * rows
            totals["positive_call_value_fraction"] += (
                float(metrics["positive_call_value_fraction"]) * rows
            )
            totals["mean_selected_action_value"] += (
                float(metrics["mean_selected_action_value"]) * rows
            )
    rows = int(totals["rows"])
    if not rows:
        raise RuntimeError(f"H012 evaluation split {split_code} has no selected rows")
    calls = int(totals["calls"])
    return {
        "rows": rows,
        "loss": totals["loss"] / rows,
        "chosen_log_utility": totals["chosen_log_utility"] / rows,
        "expected_log_utility": totals["expected_log_utility"] / rows,
        "mean_size": totals["mean_size"] / rows,
        "calls": calls,
        "call_rate": calls / rows,
        "correct_calls": int(totals["correct_calls"]),
        "call_precision": totals["correct_calls"] / calls if calls else 0.0,
        "skips": rows - calls,
        "action_value_loss": totals["action_value_loss"] / rows,
        "mean_call_probability": totals["mean_call_probability"] / rows,
        "positive_call_value_fraction": totals["positive_call_value_fraction"] / rows,
        "mean_selected_action_value": totals["mean_selected_action_value"] / rows,
    }


def train(
    config_path: Path,
    policy_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    pack_dir: Path,
    outcome_dir: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    policy_config = load_json(policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    utility_config = dict(config["utility_warm_start"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("H012 full policy training requires CUDA")
    device = torch.device("cuda")
    shards = _pack_shards(pack_dir)
    source_digest = _source_digest(pack_dir, shards)
    partition = _partition(shards, float(config["selection"]["fit_component_time_fraction"]))
    partition_sha256 = _partition_digest(partition)
    weights = _component_weights(shards, partition.fit_components)
    implementation_sha256 = _implementation_digest()
    outcome_sha256 = sha256_file(outcome_dir / "result.json")
    contract_payload = (
        f"config:{sha256_file(config_path)}\npolicy:{sha256_file(policy_config_path)}\n"
        f"model:{sha256_file(model_config_path)}\nresidual:{sha256_file(residual_config_path)}\n"
        f"outcome:{outcome_sha256}\nsource:{source_digest}\npartition:{partition_sha256}\n"
        f"implementation:{implementation_sha256}\nseed:{seed}\n"
    )
    contract_sha256 = hashlib.sha256(contract_payload.encode()).hexdigest()
    outcome, feature_mask, group_mask, outcome_result = _load_outcome(
        outcome_dir, model_config, residual_config, device
    )
    model = SphinxTraceS0H012(outcome, policy_config).to(device)
    outcome_ids = {id(parameter) for parameter in model.outcome_backbone.parameters()}
    outcome_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in outcome_ids
    ]
    policy_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in outcome_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": outcome_parameters,
                "lr": float(utility_config["outcome_backbone_learning_rate"]),
            },
            {
                "params": policy_parameters,
                "lr": float(utility_config["policy_learning_rate"]),
            },
        ],
        weight_decay=float(utility_config["weight_decay"]),
    )
    epochs = int(utility_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=1e-6,
    )
    normalization = np.load(pack_dir / "normalization.npz")
    median = np.asarray(normalization["median"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    feature_clip = float(utility_config["feature_clip_after_normalization"])
    batch_size = int(utility_config["batch_size"])
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best-policy.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_selection = -math.inf
    best_epoch = -1
    stale_epochs = 0
    initial_selection: dict[str, float | int] | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H012 checkpoint belongs to another training contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"])
        best_selection = float(checkpoint["best_selection"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        initial_selection = dict(checkpoint["initial_selection"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])

    if initial_selection is None:
        initial_selection = _evaluate(
            model,
            shards,
            2,
            partition.selection_components,
            median,
            scale,
            feature_clip,
            feature_mask,
            group_mask,
            utility_config,
            batch_size * 4,
            device,
        )
    for epoch in range(start_epoch, epochs):
        model.train()
        order = list(range(len(shards)))
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
        loss_sum = 0.0
        rows_seen = 0
        for shard_index in order:
            shard = shards[shard_index]
            indices = _indices(
                shard,
                2,
                partition.fit_components,
                seed=seed,
                epoch=epoch,
                shuffle=True,
            )
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset : offset + batch_size]
                features, labels, baselines, components = _batch(
                    shard, selected, median, scale, feature_clip
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, _, baseline_tensor = _forward(
                        model, features, baselines, feature_mask, group_mask, device
                    )
                    loss, _ = selective_log_utility_loss(
                        output,
                        torch.from_numpy(labels).to(device, non_blocking=True),
                        baseline_tensor,
                        utility_config,
                        sample_weights=torch.from_numpy(weights[components]).to(
                            device, non_blocking=True
                        ),
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(utility_config["gradient_clip_norm"])
                )
                optimizer.step()
                loss_sum += float(loss.detach()) * len(selected)
                rows_seen += len(selected)
        scheduler.step()
        selection = _evaluate(
            model,
            shards,
            2,
            partition.selection_components,
            median,
            scale,
            feature_clip,
            feature_mask,
            group_mask,
            utility_config,
            batch_size * 4,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "fit_loss": loss_sum / max(rows_seen, 1),
                "fit_rows": rows_seen,
                "selection": selection,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        score = float(selection["chosen_log_utility"])
        if score > best_selection:
            best_selection = score
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "selection": selection,
                    "contract_sha256": contract_sha256,
                    "source_digest": source_digest,
                    "outcome_result_sha256": outcome_sha256,
                },
            )
        else:
            stale_epochs += 1
        _atomic_torch_save(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "epoch": epoch + 1,
                "best_selection": best_selection,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "initial_selection": initial_selection,
                "contract_sha256": contract_sha256,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
        )
        if (output_dir / "PAUSE").exists():
            return {
                "status": "paused",
                "epoch": epoch + 1,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        if stale_epochs >= int(utility_config["early_stopping_patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("contract_sha256") != contract_sha256:
        raise RuntimeError("H012 best policy belongs to another training contract")
    model.load_state_dict(best["model"])
    fit = _evaluate(
        model,
        shards,
        2,
        partition.fit_components,
        median,
        scale,
        feature_clip,
        feature_mask,
        group_mask,
        utility_config,
        batch_size * 4,
        device,
    )
    selection = _evaluate(
        model,
        shards,
        2,
        partition.selection_components,
        median,
        scale,
        feature_clip,
        feature_mask,
        group_mask,
        utility_config,
        batch_size * 4,
        device,
    )
    calibration = _evaluate(
        model,
        shards,
        3,
        None,
        median,
        scale,
        feature_clip,
        feature_mask,
        group_mask,
        utility_config,
        batch_size * 4,
        device,
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h012_utility_warm_start_result",
        "completed_at": now_utc(),
        "research_id": str(config["research_id"]),
        "valid": math.isfinite(best_selection),
        "config_sha256": sha256_file(config_path),
        "policy_config_sha256": sha256_file(policy_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_sha256,
        "contract_sha256": contract_sha256,
        "source_digest": source_digest,
        "outcome_result_sha256": outcome_sha256,
        "outcome_candidate_id": outcome_result["candidate_id"],
        "outcome_variant_id": outcome_result["variant_id"],
        "parameters": parameter_count(model),
        "partition_sha256": partition_sha256,
        "fit_components": len(partition.fit_components),
        "selection_components": len(partition.selection_components),
        "cutoff_unix": partition.cutoff_unix,
        "initial_selection": initial_selection,
        "best_epoch": best_epoch,
        "best_selection_chosen_log_utility": best_selection,
        "fit": fit,
        "selection": selection,
        "calibration": calibration,
        "history": history,
        "best_model_sha256": sha256_file(best_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument(
        "--residual-config",
        type=Path,
        default=ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json",
    )
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.policy_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.pack_dir.resolve(),
        args.outcome_dir.resolve(),
        args.output_dir.resolve(),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "complete"),
                "best_epoch": result.get("best_epoch"),
                "selection": result.get("selection"),
                "calibration": result.get("calibration"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
