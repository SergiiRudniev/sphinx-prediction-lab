"""Train resumable H011 terminal-outcome models on the full causal feature pack."""

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
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h011 import (
    SphinxTraceS0H011,
    h011_variant_feature_mask,
    h011_variant_group_mask,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_train_v1.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


@dataclass(frozen=True, slots=True)
class PackShard:
    date: str
    root: Path


@dataclass(frozen=True, slots=True)
class BalanceWeights:
    component: NDArray[np.float32]
    lifecycle: NDArray[np.float32]
    sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    logits: NDArray[np.float32]
    labels: NDArray[np.float32]
    baselines: NDArray[np.float32]
    shard_indices: NDArray[np.uint16]
    row_indices: NDArray[np.int32]
    timestamps: NDArray[np.int64]
    market_ids: NDArray[np.int32]
    component_ids: NDArray[np.int32]


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


def _training_contract_digest(
    config_path: Path,
    model_config_path: Path,
    implementation_digest: str,
) -> str:
    payload = (
        f"training_config:{sha256_file(config_path)}\n"
        f"model_config:{sha256_file(model_config_path)}\n"
        f"implementation:{implementation_digest}\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _pack_shards(pack_dir: Path) -> list[PackShard]:
    shards = [
        PackShard(path.name.removeprefix("date="), path)
        for path in sorted((pack_dir / "shards").glob("date=*"))
        if path.is_dir()
    ]
    if not shards:
        raise RuntimeError("H011 training found no feature shards")
    return shards


def _source_digest(pack_dir: Path, shards: list[PackShard]) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(pack_dir / "manifest.json").encode())
    digest.update(sha256_file(pack_dir / "normalization.npz").encode())
    for shard in shards:
        receipt_path = pack_dir / "receipts" / f"date={shard.date}.json"
        digest.update(f"{shard.date}:{sha256_file(receipt_path)}\n".encode())
    return digest.hexdigest()


def _balanced_weights(
    pack_dir: Path,
    shards: list[PackShard],
    output_dir: Path,
    config: dict[str, Any],
    source_digest: str,
) -> BalanceWeights:
    receipt_path = output_dir / "balance-weights.json"
    array_path = output_dir / "balance-weights.npz"
    if receipt_path.exists() and array_path.exists():
        receipt = _load_object(receipt_path)
        if receipt.get("source_digest") != source_digest:
            raise RuntimeError("H011 balance weights belong to another feature pack")
        with np.load(array_path, allow_pickle=False) as archive:
            return BalanceWeights(
                component=np.asarray(archive["component"], dtype=np.float32),
                lifecycle=np.asarray(archive["lifecycle"], dtype=np.float32),
                sha256=sha256_file(array_path),
            )
    train_code = int(config["data"]["train_split_code"])
    buckets = int(config["data"]["lifecycle_buckets"])
    maximum_component = 0
    for shard in shards:
        components = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        if len(components):
            maximum_component = max(maximum_component, int(components.max()))
    component_counts = np.zeros(maximum_component + 1, dtype=np.int64)
    lifecycle_counts = np.zeros(buckets, dtype=np.int64)
    train_rows = 0
    for shard in shards:
        split = np.load(shard.root / "split_codes.npy", mmap_mode="r")
        label_mask = np.load(shard.root / "label_mask.npy", mmap_mode="r")
        selected = np.flatnonzero((split == train_code) & (label_mask == 1))
        if not len(selected):
            continue
        components = np.load(shard.root / "component_ids.npy", mmap_mode="r")
        features = np.load(shard.root / "features.npy", mmap_mode="r")
        component_counts += np.bincount(
            np.asarray(components[selected], dtype=np.int64),
            minlength=len(component_counts),
        )
        lifecycle = np.floor(np.asarray(features[selected, 46]) * buckets).astype(np.int64)
        np.clip(lifecycle, 0, buckets - 1, out=lifecycle)
        lifecycle_counts += np.bincount(lifecycle, minlength=buckets)
        train_rows += len(selected)
    if train_rows == 0:
        raise RuntimeError("H011 training has no labeled train rows")

    def inverse_sqrt(counts: NDArray[np.int64]) -> NDArray[np.float32]:
        weights = np.zeros(len(counts), dtype=np.float64)
        observed = counts > 0
        weights[observed] = 1.0 / np.sqrt(counts[observed])
        normalization = np.average(weights[observed], weights=counts[observed])
        weights[observed] /= max(float(normalization), np.finfo(np.float64).tiny)
        return weights.astype(np.float32)

    component_weights = inverse_sqrt(component_counts)
    lifecycle_weights = inverse_sqrt(lifecycle_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(array_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, component=component_weights, lifecycle=lifecycle_weights)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    receipt = {
        "schema_version": "1.0.0",
        "record_type": "h011_balance_weights",
        "generated_at": now_utc(),
        "source_digest": source_digest,
        "train_rows": train_rows,
        "components_observed": int(np.count_nonzero(component_counts)),
        "lifecycle_buckets": buckets,
        "sha256": sha256_file(array_path),
    }
    atomic_json(receipt_path, receipt)
    return BalanceWeights(component_weights, lifecycle_weights, str(receipt["sha256"]))


def _shard_order(shards: list[PackShard], seed: int, epoch: int, shuffle: bool) -> list[int]:
    order = list(range(len(shards)))
    if shuffle:
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def _row_indices(
    shard: PackShard,
    split_code: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> NDArray[np.int64]:
    split = np.load(shard.root / "split_codes.npy", mmap_mode="r")
    label_mask = np.load(shard.root / "label_mask.npy", mmap_mode="r")
    indices = np.flatnonzero((split == split_code) & (label_mask == 1)).astype(np.int64)
    if shuffle and len(indices):
        digest = int(hashlib.sha256(shard.date.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed ^ epoch ^ digest)
        rng.shuffle(indices)
    return indices


def _batch(
    shard: PackShard,
    indices: NDArray[np.int64],
    median: NDArray[np.float32],
    scale: NDArray[np.float32],
    feature_clip: float,
    balance: BalanceWeights,
    lifecycle_buckets: int,
    weight_clip: tuple[float, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    raw = np.load(shard.root / "features.npy", mmap_mode="r")
    labels = np.load(shard.root / "labels.npy", mmap_mode="r")
    baselines = np.load(shard.root / "baselines.npy", mmap_mode="r")
    components = np.load(shard.root / "component_ids.npy", mmap_mode="r")
    raw_features = np.asarray(raw[indices], dtype=np.float32)
    lifecycle = np.floor(raw_features[:, 46] * lifecycle_buckets).astype(np.int64)
    features = (raw_features - median) / scale
    np.clip(features, -feature_clip, feature_clip, out=features)
    targets = np.asarray(labels[indices], dtype=np.float32)
    baseline = np.asarray(baselines[indices], dtype=np.float32)
    component_ids = np.asarray(components[indices], dtype=np.int64)
    np.clip(lifecycle, 0, lifecycle_buckets - 1, out=lifecycle)
    weights = balance.component[component_ids] * balance.lifecycle[lifecycle]
    np.clip(weights, weight_clip[0], weight_clip[1], out=weights)
    return features, targets, baseline, weights.astype(np.float32)


def _loss(
    output: dict[str, Tensor],
    targets: Tensor,
    baselines: Tensor,
    weights: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    loss_config = config["training"]["loss"]
    logits = output["terminal_outcome_logit"]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    brier = (probabilities - targets).square()
    edge_target = targets - baselines
    edge = F.smooth_l1_loss(output["expected_net_edge"], edge_target, reduction="none")
    scale = F.softplus(output["uncertainty_log_scale"]).clamp(0.05, 10.0)
    heteroscedastic = F.binary_cross_entropy_with_logits(logits / scale, targets, reduction="none")
    total = (
        float(loss_config["binary_cross_entropy"]) * bce
        + float(loss_config["brier"]) * brier
        + float(loss_config["expected_edge_smooth_l1"]) * edge
        + float(loss_config["heteroscedastic_binary_cross_entropy"]) * heteroscedastic
        + float(loss_config["uncertainty_scale_penalty"]) * scale
    )
    weighted = (total * weights).sum() / weights.sum().clamp_min(1e-8)
    return weighted, {
        "bce": float(bce.mean().detach()),
        "brier": float(brier.mean().detach()),
        "edge": float(edge.mean().detach()),
        "uncertainty_scale": float(scale.mean().detach()),
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _metrics(probabilities: NDArray[np.float64], labels: NDArray[np.float64]) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    log_loss = float(-(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)).mean())
    brier = float(np.mean((probabilities - labels) ** 2))
    accuracy = float(np.mean((probabilities >= 0.5) == labels))
    ece = 0.0
    for lower in np.linspace(0.0, 0.95, 20):
        selected = (probabilities >= lower) & (probabilities < lower + 0.05)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return {
        "log_loss": log_loss,
        "brier": brier,
        "accuracy": accuracy,
        "expected_calibration_error": ece,
    }


@torch.inference_mode()
def _evaluate(
    model: SphinxTraceS0H011,
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
            tensor = torch.from_numpy(features).to(device, non_blocking=True) * feature_mask
            groups = group_mask.expand(len(tensor), -1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(tensor, group_mask=groups)
            logits_output.append(output["terminal_outcome_logit"].float().cpu().numpy())
            labels_output.append(np.asarray(labels[selected], dtype=np.float32))
            baselines_output.append(np.asarray(baselines[selected], dtype=np.float32))
            shard_indices_output.append(np.full(len(selected), shard_index, dtype=np.uint16))
            row_indices_output.append(np.asarray(selected, dtype=np.int32))
            timestamps_output.append(np.asarray(timestamps[selected], dtype=np.int64))
            market_ids_output.append(np.asarray(market_ids[selected], dtype=np.int32))
            component_ids_output.append(np.asarray(component_ids[selected], dtype=np.int32))
    if not logits_output:
        raise RuntimeError(f"H011 evaluation split {split_code} has no labeled rows")
    logits = np.concatenate(logits_output)
    labels = np.concatenate(labels_output)
    baselines = np.concatenate(baselines_output)
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    result: dict[str, Any] = _metrics(probabilities, labels.astype(np.float64))
    baseline_metrics = _metrics(baselines.astype(np.float64), labels.astype(np.float64))
    result["market_baseline_log_loss"] = baseline_metrics["log_loss"]
    result["market_baseline_brier"] = baseline_metrics["brier"]
    result["log_loss_delta_vs_market"] = result["log_loss"] - baseline_metrics["log_loss"]
    result["brier_delta_vs_market"] = result["brier"] - baseline_metrics["brier"]
    result["rows"] = len(labels)
    return EvaluationResult(
        metrics=result,
        logits=logits,
        labels=labels,
        baselines=baselines,
        shard_indices=np.concatenate(shard_indices_output),
        row_indices=np.concatenate(row_indices_output),
        timestamps=np.concatenate(timestamps_output),
        market_ids=np.concatenate(market_ids_output),
        component_ids=np.concatenate(component_ids_output),
    )


def train(
    config_path: Path,
    model_config_path: Path,
    pack_dir: Path,
    output_dir: Path,
    *,
    candidate_id: str,
    variant_id: str,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("H011 full training requires CUDA")
    config = load_json(config_path)
    model_config = load_json(model_config_path)
    implementation_digest = _implementation_digest()
    contract_digest = _training_contract_digest(
        config_path,
        model_config_path,
        implementation_digest,
    )
    pack_manifest = _load_object(pack_dir / "manifest.json")
    if (
        pack_manifest.get("valid") is not True
        or pack_manifest.get("test_labels_opened") is not False
    ):
        raise RuntimeError("H011 training requires a valid closed-test feature pack")
    shards = _pack_shards(pack_dir)
    source_digest = _source_digest(pack_dir, shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    balance = _balanced_weights(pack_dir, shards, output_dir, config, source_digest)
    with np.load(pack_dir / "normalization.npz", allow_pickle=False) as normalization:
        median = np.asarray(normalization["median"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SphinxTraceS0H011(model_config, candidate_id=candidate_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"]["minimum_learning_rate"]),
    )
    feature_mask = h011_variant_feature_mask(variant_id, device=device)
    group_mask = h011_variant_group_mask(variant_id, device=device)
    checkpoint_path = output_dir / "checkpoint.pt"
    cursor = {"epoch": 0, "shard_position": 0, "batch_position": 0}
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = -1
    stale_epochs = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["source_digest"] != source_digest:
            raise RuntimeError("H011 checkpoint belongs to another feature pack")
        if checkpoint.get("training_contract_sha256") != contract_digest:
            raise RuntimeError("H011 checkpoint belongs to another training contract")
        if checkpoint["candidate_id"] != candidate_id or checkpoint["variant_id"] != variant_id:
            raise RuntimeError("H011 checkpoint belongs to another model variant")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        cursor = dict(checkpoint["cursor"])
        history = list(checkpoint["history"])
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
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
                "sha256": sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
        )

    training = config["training"]
    data = config["data"]
    batch_size = int(training["batch_size"])
    feature_clip = float(data["feature_clip_after_normalization"])
    lifecycle_buckets = int(data["lifecycle_buckets"])
    clip_values = tuple(float(value) for value in data["loss_weight_clip"])
    if len(clip_values) != 2:
        raise RuntimeError("H011 loss weight clip must contain two values")
    weight_clip = (clip_values[0], clip_values[1])
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
                        group_mask=group_mask.expand(len(feature_tensor), -1),
                    )
                    loss, _ = _loss(output, target_tensor, baseline_tensor, weight_tensor, config)
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
        validation_evaluation = _evaluate(
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
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches_seen, 1),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        if float(validation["log_loss"]) < best_validation:
            best_validation = float(validation["log_loss"])
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                output_dir / "best-model.pt",
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
    best_path = output_dir / "best-model.pt"
    best = torch.load(best_path, map_location=device, weights_only=False)
    if (
        best.get("source_digest") != source_digest
        or best.get("training_contract_sha256") != contract_digest
        or best.get("candidate_id") != candidate_id
        or best.get("variant_id") != variant_id
    ):
        raise RuntimeError("H011 best model belongs to another training contract")
    model.load_state_dict(best["model"])
    validation_evaluation = _evaluate(
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
    calibration_evaluation = _evaluate(
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
        "record_type": "h011_outcome_training_result",
        "completed_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": sha256_file(config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "implementation_sha256": implementation_digest,
        "training_contract_sha256": contract_digest,
        "source_digest": source_digest,
        "valid": math.isfinite(best_validation),
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "seed": seed,
        "parameters": parameter_count(model),
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
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--candidate", default="50m")
    value.add_argument("--variant", default="h011_market_only")
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.model_config.resolve(),
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
                "best_validation_log_loss": result.get("best_validation_log_loss"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
