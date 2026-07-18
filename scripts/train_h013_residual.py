"""Train resumable market-anchored H013 residual outcome models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scripts.train_h011_outcome import (
    EvaluationResult,
    PackShard,
    _atomic_torch_save,
    _balanced_weights,
    _batch,
    _load_object,
    _metrics,
    _pack_shards,
    _row_indices,
    _shard_order,
    _source_digest,
)
from scripts.train_h011_outcome import (
    _loss as _h011_loss,
)
from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h011 import SphinxTraceS0H011
from sphinx_trace.model_h013 import (
    SphinxTraceS0H013,
    h013_variant_feature_mask,
    h013_variant_group_mask,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_train_v1.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "train_h011_outcome.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _contract_digest(
    training_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    implementation_digest: str,
) -> str:
    payload = (
        f"training_config:{sha256_file(training_config_path)}\n"
        f"model_config:{sha256_file(model_config_path)}\n"
        f"residual_config:{sha256_file(residual_config_path)}\n"
        f"implementation:{implementation_digest}\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def residual_loss(
    output: dict[str, Tensor],
    targets: Tensor,
    baselines: Tensor,
    weights: Tensor,
    training_config: dict[str, Any],
    residual_config: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    base_loss, metrics = _h011_loss(output, targets, baselines, weights, training_config)
    residual = output["terminal_outcome_residual_logit"]
    residual_l2 = (residual.square() * weights).sum() / weights.sum().clamp_min(1e-8)
    coefficient = float(residual_config["training"]["residual_l2_weight"])
    total = base_loss + coefficient * residual_l2
    metrics["residual_l2"] = float(residual_l2.detach())
    return total, metrics


@torch.inference_mode()
def evaluate_residual(
    model: SphinxTraceS0H013,
    shards: list[PackShard],
    split_code: int,
    median: NDArray[np.float32],
    scale: NDArray[np.float32],
    feature_clip: float,
    feature_mask: Tensor,
    group_mask: Tensor,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    logits_output: list[NDArray[np.float32]] = []
    labels_output: list[NDArray[np.float32]] = []
    baselines_output: list[NDArray[np.float32]] = []
    shard_indices_output: list[NDArray[np.uint16]] = []
    row_indices_output: list[NDArray[np.int32]] = []
    timestamps_output: list[NDArray[np.int64]] = []
    market_ids_output: list[NDArray[np.int32]] = []
    component_ids_output: list[NDArray[np.int32]] = []
    for shard_index, shard in enumerate(shards):
        indices = _row_indices(shard, split_code, seed=0, epoch=0, shuffle=False)
        raw = np.load(shard.root / "features.npy", mmap_mode="r")
        labels = np.load(shard.root / "labels.npy", mmap_mode="r")
        baselines = np.load(shard.root / "baselines.npy", mmap_mode="r")
        timestamps = np.load(shard.root / "timestamps.npy", mmap_mode="r")
        market_ids = np.load(shard.root / "market_ids.npy", mmap_mode="r")
        component_ids = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            features = (np.asarray(raw[selected], dtype=np.float32) - median) / scale
            np.clip(features, -feature_clip, feature_clip, out=features)
            baseline = np.asarray(baselines[selected], dtype=np.float32)
            tensor = torch.from_numpy(features).to(device, non_blocking=True) * feature_mask
            baseline_tensor = torch.from_numpy(baseline).to(device, non_blocking=True)
            groups = group_mask.expand(len(tensor), -1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(tensor, baseline_tensor, group_mask=groups)
            logits_output.append(output["terminal_outcome_logit"].float().cpu().numpy())
            labels_output.append(np.asarray(labels[selected], dtype=np.float32))
            baselines_output.append(baseline)
            shard_indices_output.append(np.full(len(selected), shard_index, dtype=np.uint16))
            row_indices_output.append(np.asarray(selected, dtype=np.int32))
            timestamps_output.append(np.asarray(timestamps[selected], dtype=np.int64))
            market_ids_output.append(np.asarray(market_ids[selected], dtype=np.int32))
            component_ids_output.append(np.asarray(component_ids[selected], dtype=np.int32))
    if not logits_output:
        raise RuntimeError(f"H013 evaluation split {split_code} has no labeled rows")
    logits = np.concatenate(logits_output)
    labels_array = np.concatenate(labels_output)
    baselines_array = np.concatenate(baselines_output)
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    metrics: dict[str, Any] = _metrics(probabilities, labels_array.astype(np.float64))
    baseline_metrics = _metrics(baselines_array.astype(np.float64), labels_array.astype(np.float64))
    metrics["market_baseline_log_loss"] = baseline_metrics["log_loss"]
    metrics["market_baseline_brier"] = baseline_metrics["brier"]
    metrics["log_loss_delta_vs_market"] = metrics["log_loss"] - baseline_metrics["log_loss"]
    metrics["brier_delta_vs_market"] = metrics["brier"] - baseline_metrics["brier"]
    metrics["rows"] = len(labels_array)
    return EvaluationResult(
        metrics=metrics,
        logits=logits,
        labels=labels_array,
        baselines=baselines_array,
        shard_indices=np.concatenate(shard_indices_output),
        row_indices=np.concatenate(row_indices_output),
        timestamps=np.concatenate(timestamps_output),
        market_ids=np.concatenate(market_ids_output),
        component_ids=np.concatenate(component_ids_output),
    )


def train(
    training_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    pack_dir: Path,
    output_dir: Path,
    *,
    candidate_id: str,
    variant_id: str,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("H013 full training requires CUDA")
    training_config = load_json(training_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    if variant_id not in residual_config["training"]["variants"]:
        raise ValueError(f"H013 variant is not registered: {variant_id}")
    implementation_digest = _implementation_digest()
    contract_digest = _contract_digest(
        training_config_path,
        model_config_path,
        residual_config_path,
        implementation_digest,
    )
    pack_manifest = _load_object(pack_dir / "manifest.json")
    if (
        pack_manifest.get("valid") is not True
        or pack_manifest.get("test_labels_opened") is not False
    ):
        raise RuntimeError("H013 training requires a valid closed-test feature pack")
    shards = _pack_shards(pack_dir)
    source_digest = _source_digest(pack_dir, shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    balance = _balanced_weights(
        pack_dir,
        shards,
        output_dir,
        training_config,
        source_digest,
    )
    with np.load(pack_dir / "normalization.npz", allow_pickle=False) as normalization:
        median = np.asarray(normalization["median"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    anchor_config = residual_config["architecture"]
    model = SphinxTraceS0H013(
        SphinxTraceS0H011(model_config, candidate_id=candidate_id),
        minimum_probability=float(anchor_config["minimum_anchor_probability"]),
        maximum_probability=float(anchor_config["maximum_anchor_probability"]),
    ).to(device)
    training = training_config["training"]
    data = training_config["data"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    feature_mask = h013_variant_feature_mask(variant_id, device=device)
    group_mask = h013_variant_group_mask(variant_id, device=device)
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best-model.pt"
    cursor = {"epoch": 0, "shard_position": 0, "batch_position": 0}
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = -1
    stale_epochs = 0
    initial_validation: dict[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["source_digest"] != source_digest:
            raise RuntimeError("H013 checkpoint belongs to another feature pack")
        if checkpoint.get("training_contract_sha256") != contract_digest:
            raise RuntimeError("H013 checkpoint belongs to another training contract")
        if checkpoint["candidate_id"] != candidate_id or checkpoint["variant_id"] != variant_id:
            raise RuntimeError("H013 checkpoint belongs to another model variant")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        cursor = dict(checkpoint["cursor"])
        history = list(checkpoint["history"])
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        initial_validation = dict(checkpoint["initial_validation"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])

    def save_checkpoint(next_cursor: dict[str, int]) -> None:
        _atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": "1.0.0",
                "source_digest": source_digest,
                "training_contract_sha256": contract_digest,
                "candidate_id": candidate_id,
                "variant_id": variant_id,
                "seed": seed,
                "cursor": next_cursor,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "initial_validation": initial_validation,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
        )
        atomic_json(
            output_dir / "checkpoint.json",
            {
                "updated_at": now_utc(),
                "cursor": next_cursor,
                "best_validation_log_loss": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "sha256": sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
        )

    batch_size = int(training["batch_size"])
    feature_clip = float(data["feature_clip_after_normalization"])
    lifecycle_buckets = int(data["lifecycle_buckets"])
    clip_values = tuple(float(value) for value in data["loss_weight_clip"])
    if len(clip_values) != 2:
        raise RuntimeError("H013 loss weight clip must contain two values")
    weight_clip = (clip_values[0], clip_values[1])
    if initial_validation is None:
        evaluation = evaluate_residual(
            model,
            shards,
            int(data["validation_split_code"]),
            median,
            scale,
            feature_clip,
            feature_mask,
            group_mask,
            int(training["evaluation_batch_size"]),
            device,
        )
        initial_validation = evaluation.metrics
        best_validation = float(initial_validation["log_loss"])
        _atomic_torch_save(
            best_path,
            {
                "model": model.state_dict(),
                "epoch": -1,
                "validation": initial_validation,
                "source_digest": source_digest,
                "training_contract_sha256": contract_digest,
                "candidate_id": candidate_id,
                "variant_id": variant_id,
            },
        )
        save_checkpoint(cursor)

    last_checkpoint = time.monotonic()
    for epoch in range(int(cursor["epoch"]), int(training["epochs"])):
        model.train()
        order = _shard_order(shards, seed, epoch, True)
        epoch_loss = 0.0
        batches_seen = 0
        start_shard = int(cursor["shard_position"]) if epoch == int(cursor["epoch"]) else 0
        for shard_position in range(start_shard, len(order)):
            shard = shards[order[shard_position]]
            indices = _row_indices(
                shard,
                int(data["train_split_code"]),
                seed=seed,
                epoch=epoch,
                shuffle=True,
            )
            start_batch = (
                int(cursor["batch_position"])
                if epoch == int(cursor["epoch"]) and shard_position == start_shard
                else 0
            )
            for batch_position, offset in enumerate(range(0, len(indices), batch_size)):
                if batch_position < start_batch:
                    continue
                selected = indices[offset : offset + batch_size]
                features, targets, baselines, weights = _batch(
                    shard,
                    selected,
                    median,
                    scale,
                    feature_clip,
                    balance,
                    lifecycle_buckets,
                    weight_clip,
                )
                feature_tensor = torch.from_numpy(features).to(device, non_blocking=True)
                feature_tensor = feature_tensor * feature_mask
                target_tensor = torch.from_numpy(targets).to(device, non_blocking=True)
                baseline_tensor = torch.from_numpy(baselines).to(device, non_blocking=True)
                weight_tensor = torch.from_numpy(weights).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        feature_tensor,
                        baseline_tensor,
                        group_mask=group_mask.expand(len(feature_tensor), -1),
                    )
                    loss, _ = residual_loss(
                        output,
                        target_tensor,
                        baseline_tensor,
                        weight_tensor,
                        training_config,
                        residual_config,
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches_seen += 1
                next_cursor = {
                    "epoch": epoch,
                    "shard_position": shard_position,
                    "batch_position": batch_position + 1,
                }
                if batch_position + 1 >= math.ceil(len(indices) / batch_size):
                    next_cursor = {
                        "epoch": epoch,
                        "shard_position": shard_position + 1,
                        "batch_position": 0,
                    }
                if (
                    time.monotonic() - last_checkpoint
                    >= int(training["checkpoint_maximum_interval_seconds"])
                    or (output_dir / "PAUSE").exists()
                ):
                    save_checkpoint(next_cursor)
                    last_checkpoint = time.monotonic()
                if (output_dir / "PAUSE").exists():
                    return {
                        "status": "paused",
                        "cursor": next_cursor,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                    }
        scheduler.step()
        validation_evaluation = evaluate_residual(
            model,
            shards,
            int(data["validation_split_code"]),
            median,
            scale,
            feature_clip,
            feature_mask,
            group_mask,
            int(training["evaluation_batch_size"]),
            device,
        )
        validation = validation_evaluation.metrics
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batches_seen, 1),
                "validation": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if float(validation["log_loss"]) < best_validation:
            best_validation = float(validation["log_loss"])
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "source_digest": source_digest,
                    "training_contract_sha256": contract_digest,
                    "candidate_id": candidate_id,
                    "variant_id": variant_id,
                },
            )
        else:
            stale_epochs += 1
        cursor = {"epoch": epoch + 1, "shard_position": 0, "batch_position": 0}
        save_checkpoint(cursor)
        if stale_epochs >= int(training["early_stopping_patience"]):
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    if (
        best.get("source_digest") != source_digest
        or best.get("training_contract_sha256") != contract_digest
        or best.get("candidate_id") != candidate_id
        or best.get("variant_id") != variant_id
    ):
        raise RuntimeError("H013 best model belongs to another training contract")
    model.load_state_dict(best["model"])
    validation_evaluation = evaluate_residual(
        model,
        shards,
        int(data["validation_split_code"]),
        median,
        scale,
        feature_clip,
        feature_mask,
        group_mask,
        int(training["evaluation_batch_size"]),
        device,
    )
    calibration_evaluation = evaluate_residual(
        model,
        shards,
        int(data["calibration_split_code"]),
        median,
        scale,
        feature_clip,
        feature_mask,
        group_mask,
        int(training["evaluation_batch_size"]),
        device,
    )
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        validation_logits=validation_evaluation.logits,
        validation_labels=validation_evaluation.labels,
        validation_baselines=validation_evaluation.baselines,
        validation_shard_indices=validation_evaluation.shard_indices,
        validation_row_indices=validation_evaluation.row_indices,
        validation_timestamps=validation_evaluation.timestamps,
        validation_market_ids=validation_evaluation.market_ids,
        validation_component_ids=validation_evaluation.component_ids,
        calibration_logits=calibration_evaluation.logits,
        calibration_labels=calibration_evaluation.labels,
        calibration_baselines=calibration_evaluation.baselines,
        calibration_shard_indices=calibration_evaluation.shard_indices,
        calibration_row_indices=calibration_evaluation.row_indices,
        calibration_timestamps=calibration_evaluation.timestamps,
        calibration_market_ids=calibration_evaluation.market_ids,
        calibration_component_ids=calibration_evaluation.component_ids,
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": "h013_market_residual_training_result",
        "completed_at": now_utc(),
        "research_id": str(residual_config["research_id"]),
        "training_config_sha256": sha256_file(training_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_digest,
        "training_contract_sha256": contract_digest,
        "source_digest": source_digest,
        "valid": math.isfinite(best_validation),
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "seed": seed,
        "parameters": parameter_count(model),
        "initial_validation": initial_validation,
        "best_epoch": best_epoch,
        "best_validation_log_loss": best_validation,
        "validation": validation_evaluation.metrics,
        "calibration": calibration_evaluation.metrics,
        "history": history,
        "balance_weights_sha256": balance.sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_model_sha256": sha256_file(best_path),
        "predictions_sha256": sha256_file(predictions_path),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(residual_config["evidence_boundary"]),
    }
    atomic_json(output_dir / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--candidate", default="50m")
    value.add_argument("--variant", default="h013_market_residual")
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.training_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.pack_dir.resolve(),
        args.output_dir.resolve(),
        candidate_id=args.candidate,
        variant_id=args.variant,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "completed"),
                "candidate_id": result.get("candidate_id"),
                "variant_id": result.get("variant_id"),
                "parameters": result.get("parameters"),
                "best_epoch": result.get("best_epoch"),
                "best_validation_log_loss": result.get("best_validation_log_loss"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
