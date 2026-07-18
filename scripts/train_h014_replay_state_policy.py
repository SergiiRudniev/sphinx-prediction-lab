"""Fine-tune H012 policy fusion on exact causal replay states for H014."""

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
from torch import Tensor, nn

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.policy_checkpoint import load_policy_checkpoint
from sphinx_trace.policy_training import selective_log_utility_loss
from sphinx_trace.replay_state_pack import validate_state_shard

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h014_replay_state_distillation_v1.json"
)
DEFAULT_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
)
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "replay_state_pack.py",
    ROOT / "src" / "sphinx_trace" / "policy_training.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


@dataclass(frozen=True, slots=True)
class StateShard:
    date: str
    state: Path
    encoding: Path
    pack: Path
    rows: int


@dataclass(frozen=True, slots=True)
class StateBatch:
    market_latents: NDArray[np.float32]
    terminal_logits: NDArray[np.float32]
    uncertainty_log_scales: NDArray[np.float32]
    portfolio_features: NDArray[np.float32]
    prediction_memory_features: NDArray[np.float32]
    previous_action_ids: NDArray[np.int64]
    physical_action_masks: NDArray[np.uint8]
    labels: NDArray[np.float32]
    baselines: NDArray[np.float32]
    component_ids: NDArray[np.int64]


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


def _module_digest(module: nn.Module) -> str:
    """Hash named tensor bytes independently of torch serialization metadata."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.025 * (2**attempt)))
    raise PermissionError(f"Could not atomically replace {path}: {last_error}")


def _state_shards(
    state_dir: Path,
    encoding_dir: Path,
    pack_dir: Path,
    initial_policy_dir: Path,
    config: dict[str, Any],
    config_sha256: str,
) -> tuple[list[StateShard], dict[str, Any]]:
    manifest_path = state_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    initial_policy_sha256 = sha256_file(initial_policy_dir / "result.json")
    if (
        manifest.get("record_type") != "h014_replay_state_pack_manifest"
        or manifest.get("valid") is not True
        or manifest.get("test_labels_opened") is not False
        or int(manifest.get("test_rows_consumed", -1)) != 0
        or int(manifest.get("calibration_rows_consumed", -1)) != 0
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("pack_manifest_sha256") != sha256_file(pack_dir / "manifest.json")
        or manifest.get("encoding_manifest_sha256")
        != sha256_file(encoding_dir / "manifest.json")
        or config["dependencies"]["initial_policy"]["result_sha256"]
        != initial_policy_sha256
    ):
        raise RuntimeError("H014 replay-state training source contract changed")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise RuntimeError("H014 replay-state manifest has no shards")
    shards: list[StateShard] = []
    total_rows = 0
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise RuntimeError("H014 replay-state shard metadata is invalid")
        date = str(raw.get("date") or "")
        rows = int(raw.get("rows", -1))
        state = state_dir / "shards" / f"date={date}"
        encoding = encoding_dir / "shards" / f"date={date}"
        pack = pack_dir / "shards" / f"date={date}"
        receipt_path = state_dir / str(raw.get("receipt_path") or "")
        if (
            not state.is_dir()
            or not encoding.is_dir()
            or not pack.is_dir()
            or not receipt_path.is_file()
            or raw.get("receipt_sha256") != sha256_file(receipt_path)
        ):
            raise RuntimeError(f"H014 replay-state shard binding changed: {date}")
        receipt = _load_object(receipt_path)
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise RuntimeError(f"H014 replay-state receipt is invalid: {date}")
        validate_state_shard(state, files, expected_rows=rows)
        shards.append(StateShard(date, state, encoding, pack, rows))
        total_rows += rows
    if total_rows != int(manifest.get("rows", -1)) or total_rows != int(
        config["corpus"]["rows"]
    ):
        raise RuntimeError("H014 replay-state total rows changed")
    return shards, manifest


def _indices(
    shard: StateShard,
    partition_code: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> NDArray[np.int64]:
    partitions = np.load(
        shard.state / "partition_codes.npy", mmap_mode="r", allow_pickle=False
    )
    indices = np.flatnonzero(partitions == partition_code).astype(np.int64)
    if shuffle and len(indices):
        date_seed = int(hashlib.sha256(shard.date.encode()).hexdigest()[:16], 16)
        np.random.default_rng(seed ^ epoch ^ date_seed).shuffle(indices)
    return indices


def _batch(shard: StateShard, indices: NDArray[np.int64]) -> StateBatch:
    rows = np.load(shard.state / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    encoding_offsets = np.load(
        shard.state / "encoding_offsets.npy", mmap_mode="r", allow_pickle=False
    )
    components = np.load(
        shard.state / "component_ids.npy", mmap_mode="r", allow_pickle=False
    )
    portfolio = np.load(
        shard.state / "portfolio_features.npy", mmap_mode="r", allow_pickle=False
    )
    memory = np.load(
        shard.state / "prediction_memory_features.npy", mmap_mode="r", allow_pickle=False
    )
    previous = np.load(
        shard.state / "previous_action_ids.npy", mmap_mode="r", allow_pickle=False
    )
    physical = np.load(
        shard.state / "physical_action_masks.npy", mmap_mode="r", allow_pickle=False
    )
    source_rows = np.load(
        shard.encoding / "row_indices.npy", mmap_mode="r", allow_pickle=False
    )
    latents = np.load(
        shard.encoding / "market_latents.npy", mmap_mode="r", allow_pickle=False
    )
    terminal = np.load(
        shard.encoding / "terminal_logits.npy", mmap_mode="r", allow_pickle=False
    )
    uncertainty = np.load(
        shard.encoding / "uncertainty_log_scales.npy", mmap_mode="r", allow_pickle=False
    )
    labels = np.load(shard.pack / "labels.npy", mmap_mode="r", allow_pickle=False)
    baselines = np.load(shard.pack / "baselines.npy", mmap_mode="r", allow_pickle=False)
    selected_rows = np.asarray(rows[indices], dtype=np.int64)
    selected_offsets = np.asarray(encoding_offsets[indices], dtype=np.int64)
    if bool((selected_offsets < 0).any()) or bool((selected_offsets >= len(source_rows)).any()):
        raise RuntimeError(f"H014 encoding offsets are invalid: {shard.date}")
    if not np.array_equal(source_rows[selected_offsets], selected_rows):
        raise RuntimeError(f"H014 state and market encodings no longer align: {shard.date}")
    output = StateBatch(
        market_latents=np.asarray(latents[selected_offsets], dtype=np.float32),
        terminal_logits=np.asarray(terminal[selected_offsets], dtype=np.float32),
        uncertainty_log_scales=np.asarray(uncertainty[selected_offsets], dtype=np.float32),
        portfolio_features=np.asarray(portfolio[indices], dtype=np.float32),
        prediction_memory_features=np.asarray(memory[indices], dtype=np.float32),
        previous_action_ids=np.asarray(previous[indices], dtype=np.int64),
        physical_action_masks=np.asarray(physical[indices], dtype=np.uint8),
        labels=np.asarray(labels[selected_rows], dtype=np.float32),
        baselines=np.asarray(baselines[selected_rows], dtype=np.float32),
        component_ids=np.asarray(components[indices], dtype=np.int64),
    )
    if (
        output.market_latents.ndim != 2
        or output.market_latents.shape[0] != len(indices)
        or not bool(np.isfinite(output.market_latents).all())
        or not bool(np.isfinite(output.terminal_logits).all())
        or not bool(np.isfinite(output.uncertainty_log_scales).all())
        or not bool(np.isin(output.labels, (0.0, 1.0)).all())
        or not bool(np.isfinite(output.baselines).all())
    ):
        raise RuntimeError(f"H014 training batch is invalid: {shard.date}")
    return output


def _component_weights(shards: list[StateShard]) -> NDArray[np.float32]:
    maximum = 0
    for shard in shards:
        if not shard.rows:
            continue
        components = np.load(
            shard.state / "component_ids.npy", mmap_mode="r", allow_pickle=False
        )
        maximum = max(maximum, int(components.max(initial=0)))
    counts = np.zeros(maximum + 1, dtype=np.int64)
    for shard in shards:
        indices = _indices(shard, 0, seed=0, epoch=0, shuffle=False)
        if not len(indices):
            continue
        components = np.load(
            shard.state / "component_ids.npy", mmap_mode="r", allow_pickle=False
        )
        counts += np.bincount(
            np.asarray(components[indices], dtype=np.int64), minlength=len(counts)
        )
    observed = counts > 0
    if not bool(observed.any()):
        raise RuntimeError("H014 component balancing found no fit rows")
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[observed] = 1.0 / np.sqrt(counts[observed])
    normalization = np.average(weights[observed], weights=counts[observed])
    weights[observed] /= max(float(normalization), np.finfo(np.float64).tiny)
    return weights.astype(np.float32)


def _forward(
    model: SphinxTraceS0H012,
    batch: StateBatch,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    labels = torch.from_numpy(batch.labels).to(device, non_blocking=True)
    baselines = torch.from_numpy(batch.baselines).to(device, non_blocking=True)
    output = model.forward_from_market_encoding(
        torch.from_numpy(batch.market_latents).to(device, non_blocking=True),
        torch.from_numpy(batch.terminal_logits).to(device, non_blocking=True),
        torch.from_numpy(batch.uncertainty_log_scales).to(device, non_blocking=True),
        torch.from_numpy(batch.portfolio_features).to(device, non_blocking=True),
        torch.from_numpy(batch.prediction_memory_features).to(device, non_blocking=True),
        torch.from_numpy(batch.previous_action_ids).to(device, non_blocking=True),
        physical_action_mask=torch.from_numpy(batch.physical_action_masks).to(
            device, non_blocking=True
        ),
    )
    return output, labels, baselines


@torch.inference_mode()
def _evaluate(
    model: SphinxTraceS0H012,
    shards: list[StateShard],
    partition_code: int,
    utility_config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int | dict[str, int]]:
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
    }
    action_counts = np.zeros(3, dtype=np.int64)
    for shard in shards:
        indices = _indices(shard, partition_code, seed=0, epoch=0, shuffle=False)
        for offset in range(0, len(indices), batch_size):
            batch = _batch(shard, indices[offset : offset + batch_size])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, labels, baselines = _forward(model, batch, device)
                loss, metrics = selective_log_utility_loss(
                    output, labels, baselines, utility_config
                )
            rows = len(batch.labels)
            chosen = output["action_logits"][:, :3].argmax(dim=-1)
            action_counts += np.bincount(
                chosen.cpu().numpy().astype(np.int64), minlength=3
            )
            totals["rows"] += rows
            totals["loss"] += float(loss) * rows
            totals["chosen_log_utility"] += float(metrics["chosen_log_utility_sum"])
            totals["expected_log_utility"] += float(metrics["expected_log_utility"]) * rows
            totals["mean_size"] += float(metrics["mean_size"]) * rows
            totals["calls"] += float(metrics["call_count"])
            totals["correct_calls"] += float(metrics["correct_call_count"])
            totals["action_value_loss"] += float(metrics["action_value_loss"]) * rows
            totals["mean_call_probability"] += (
                float(metrics["mean_call_probability"]) * rows
            )
    rows = int(totals["rows"])
    if not rows:
        raise RuntimeError(f"H014 evaluation partition {partition_code} has no rows")
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
        "actions": {
            "CALL_OUTCOME_0": int(action_counts[0]),
            "CALL_OUTCOME_1": int(action_counts[1]),
            "SKIP": int(action_counts[2]),
        },
    }


def _selection_score(metrics: dict[str, float | int | dict[str, int]]) -> float:
    """Select only by learned economic utility, never by a CALL-frequency target."""

    value = metrics["chosen_log_utility"]
    if isinstance(value, dict):
        raise TypeError("H014 selection utility must be numeric")
    return float(value)


def train(
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
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    policy_config = load_json(policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    utility_config = dict(config["training"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("H014 full replay-state training requires CUDA")
    device = torch.device("cuda")
    shards, state_manifest = _state_shards(
        state_dir,
        encoding_dir,
        pack_dir,
        initial_policy_dir,
        config,
        sha256_file(config_path),
    )
    weights = _component_weights(shards)
    implementation_sha256 = _implementation_digest()
    source_hashes = {
        "state_manifest_sha256": sha256_file(state_dir / "manifest.json"),
        "encoding_manifest_sha256": sha256_file(encoding_dir / "manifest.json"),
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "initial_policy_result_sha256": sha256_file(initial_policy_dir / "result.json"),
        "initial_policy_best_model_sha256": sha256_file(
            initial_policy_dir / "best-policy.pt"
        ),
        "outcome_result_sha256": sha256_file(outcome_dir / "result.json"),
    }
    contract_lines = {
        "config_sha256": sha256_file(config_path),
        "policy_config_sha256": sha256_file(policy_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_sha256,
        "seed": str(seed),
        **source_hashes,
    }
    contract_payload = "".join(
        f"{key}:{value}\n" for key, value in sorted(contract_lines.items())
    )
    contract_sha256 = hashlib.sha256(contract_payload.encode()).hexdigest()
    loaded = load_policy_checkpoint(
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        policy_config,
        device,
    )
    model = loaded.model
    model.outcome_backbone.requires_grad_(False)
    market_backbone_sha256 = _module_digest(model.outcome_backbone)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(utility_config["policy_learning_rate"]),
        weight_decay=float(utility_config["weight_decay"]),
    )
    epochs = int(utility_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    batch_size = int(utility_config["batch_size"])
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best-policy.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    stale_epochs = 0
    initial_selection: dict[str, float | int | dict[str, int]] | None = None
    best_selection = -math.inf
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H014 checkpoint belongs to another training contract")
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
            model, shards, 1, utility_config, batch_size * 4, device
        )
        best_selection = _selection_score(initial_selection)
        _atomic_torch_save(
            best_path,
            {
                "model": model.state_dict(),
                "epoch": -1,
                "selection": initial_selection,
                "contract_sha256": contract_sha256,
                **source_hashes,
            },
        )
    for epoch in range(start_epoch, epochs):
        model.train()
        model.outcome_backbone.eval()
        order = list(range(len(shards)))
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
        loss_sum = 0.0
        rows_seen = 0
        for shard_index in order:
            shard = shards[shard_index]
            indices = _indices(
                shard, 0, seed=seed, epoch=epoch, shuffle=True
            )
            for offset in range(0, len(indices), batch_size):
                batch = _batch(shard, indices[offset : offset + batch_size])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, labels, baselines = _forward(model, batch, device)
                    loss, _ = selective_log_utility_loss(
                        output,
                        labels,
                        baselines,
                        utility_config,
                        sample_weights=torch.from_numpy(weights[batch.component_ids]).to(
                            device, non_blocking=True
                        ),
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(utility_config["gradient_clip_norm"])
                )
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch.labels)
                rows_seen += len(batch.labels)
        scheduler.step()
        selection = _evaluate(model, shards, 1, utility_config, batch_size * 4, device)
        score = _selection_score(selection)
        history.append(
            {
                "epoch": epoch,
                "fit_loss": loss_sum / max(rows_seen, 1),
                "fit_rows": rows_seen,
                "selection": selection,
                "selection_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
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
                    **source_hashes,
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
        atomic_json(
            output_dir / "progress.json",
            {
                "record_type": "h014_replay_state_training_progress",
                "contract_sha256": contract_sha256,
                "epoch": epoch + 1,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_selection": best_selection,
                "selection": selection,
                "updated_at": now_utc(),
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
        raise RuntimeError("H014 best policy belongs to another training contract")
    model.load_state_dict(best["model"])
    if _module_digest(model.outcome_backbone) != market_backbone_sha256:
        raise RuntimeError("H014 frozen market backbone changed during state training")
    fit = _evaluate(model, shards, 0, utility_config, batch_size * 4, device)
    selection = _evaluate(model, shards, 1, utility_config, batch_size * 4, device)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h014_replay_state_policy_result",
        "research_id": "SPH-T-H014",
        "completed_at": now_utc(),
        "valid": math.isfinite(best_selection),
        "config_sha256": sha256_file(config_path),
        "policy_config_sha256": sha256_file(policy_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_sha256,
        "contract_sha256": contract_sha256,
        **source_hashes,
        "market_encoding_policy_result_sha256": source_hashes[
            "initial_policy_result_sha256"
        ],
        "market_backbone_frozen": True,
        "market_backbone_sha256": market_backbone_sha256,
        "parameters": parameter_count(model),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "partition_sha256": state_manifest["partition_sha256"],
        "fit_components": state_manifest["fit_components"],
        "selection_components": state_manifest["selection_components"],
        "initial_selection": initial_selection,
        "best_epoch": best_epoch,
        "best_selection_chosen_log_utility": best_selection,
        "fit": fit,
        "selection": selection,
        "history": history,
        "best_model_sha256": sha256_file(best_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "exact_replay_required": True,
        "promotion_allowed": False,
        "evidence_boundary": config["evidence_boundary"],
    }
    atomic_json(output_dir / "result.json", result)
    return result


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
    value.add_argument("--seed", type=int, default=17)
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
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
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "complete"),
                "best_epoch": result.get("best_epoch"),
                "selection": result.get("selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
