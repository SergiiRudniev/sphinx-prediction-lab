from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import SphinxTraceS0, parameter_count

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_train.json"


class PackedSplit(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        root: Path,
        *,
        wallet_mode: str,
        wallet_start: int,
        wallet_tokens: int,
    ) -> None:
        if wallet_mode not in {"original", "zero", "prior_event_control"}:
            raise ValueError(f"Unknown wallet mode: {wallet_mode}")
        self.root = root
        shape = np.load(root / "features.npy", mmap_mode="r").shape
        self.length = int(shape[0])
        self.wallet_mode = wallet_mode
        self.wallet_start = wallet_start
        self.wallet_stop = wallet_start + wallet_tokens
        self.donor_indices = [-1] * self.length
        self.control_audit: dict[str, Any] = {
            "mode": wallet_mode,
            "rows": self.length,
            "donor_rows": 0,
            "zero_rows": self.length if wallet_mode == "zero" else 0,
            "donor_time_violations": 0,
            "same_event_donors": 0,
        }
        if wallet_mode == "prior_event_control":
            self.donor_indices, self.control_audit = self._build_control_donors()
        self.features: NDArray[np.float16] | None = None
        self.token_types: NDArray[np.uint8] | None = None
        self.targets: NDArray[np.float32] | None = None
        self.target_mask: NDArray[np.uint8] | None = None
        self.baselines: NDArray[np.float32] | None = None

    def _build_control_donors(self) -> tuple[list[int], dict[str, Any]]:
        examples = list(iter_jsonl_zst(self.root / "examples.jsonl.zst"))
        if len(examples) != self.length:
            raise RuntimeError("Feature and example row counts differ")
        ordered = sorted(
            range(self.length),
            key=lambda index: (
                int(examples[index]["decision_time_unix"]),
                str(examples[index]["event_id"]),
                str(examples[index]["example_id"]),
            ),
        )
        donors = [-1] * self.length
        donor_time_violations = 0
        same_event_donors = 0
        maximum_lag = 0
        for position, recipient in enumerate(ordered):
            recipient_event = str(examples[recipient]["event_id"])
            recipient_time = int(examples[recipient]["decision_time_unix"])
            for candidate in reversed(ordered[:position]):
                candidate_event = str(examples[candidate]["event_id"])
                if candidate_event == recipient_event:
                    continue
                candidate_time = int(examples[candidate]["decision_time_unix"])
                donors[recipient] = candidate
                donor_time_violations += int(candidate_time > recipient_time)
                same_event_donors += int(candidate_event == recipient_event)
                maximum_lag = max(maximum_lag, recipient_time - candidate_time)
                break
        donor_rows = sum(index >= 0 for index in donors)
        return donors, {
            "mode": self.wallet_mode,
            "rows": self.length,
            "donor_rows": donor_rows,
            "zero_rows": self.length - donor_rows,
            "donor_time_violations": donor_time_violations,
            "same_event_donors": same_event_donors,
            "maximum_donor_lag_seconds": maximum_lag,
        }

    def __len__(self) -> int:
        return self.length

    def _open(self) -> None:
        if self.features is not None:
            return
        self.features = np.load(self.root / "features.npy", mmap_mode="r")
        self.token_types = np.load(self.root / "token_types.npy", mmap_mode="r")
        self.targets = np.load(self.root / "targets.npy", mmap_mode="r")
        self.target_mask = np.load(self.root / "target_mask.npy", mmap_mode="r")
        self.baselines = np.load(self.root / "baselines.npy", mmap_mode="r")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._open()
        assert self.features is not None
        assert self.token_types is not None
        assert self.targets is not None
        assert self.target_mask is not None
        assert self.baselines is not None
        features = np.array(self.features[index], copy=True)
        if self.wallet_mode == "zero":
            features[self.wallet_start : self.wallet_stop] = 0.0
        elif self.wallet_mode == "prior_event_control":
            donor = self.donor_indices[index]
            if donor < 0:
                features[self.wallet_start : self.wallet_stop] = 0.0
            else:
                features[self.wallet_start : self.wallet_stop] = self.features[
                    donor,
                    self.wallet_start : self.wallet_stop,
                ]
        return (
            torch.from_numpy(features),
            torch.from_numpy(np.array(self.token_types[index], dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(self.targets[index], copy=True)),
            torch.from_numpy(np.array(self.target_mask[index], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.baselines[index], copy=True)),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for name in ("features", "token_types", "targets", "target_mask", "baselines"):
            state[name] = None
        return state


def _loader(
    dataset: PackedSplit,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _loss(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    names: list[str],
    weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    pieces: dict[str, Tensor] = {}
    resolution = F.binary_cross_entropy_with_logits(predictions[:, 0], targets[:, 0])
    pieces[names[0]] = resolution
    total = resolution * weights[names[0]]
    for index, name in enumerate(names[1:], start=1):
        available = mask[:, index] > 0
        if torch.any(available):
            value = F.smooth_l1_loss(
                predictions[available, index],
                targets[available, index],
            )
        else:
            value = predictions[:, index].sum() * 0.0
        pieces[name] = value
        total = total + value * weights[name]
    return total, pieces


def _binary_metrics(
    probability: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> dict[str, float]:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return {
        "brier": float(np.mean((clipped - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
    }


def _metrics(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    mask: NDArray[np.float64],
    baselines: NDArray[np.float64],
    names: list[str],
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    probability = 1.0 / (1.0 + np.exp(-predictions[:, 0] / temperature))
    resolution = _binary_metrics(probability, targets[:, 0])
    baseline_resolution = _binary_metrics(baselines[:, 0], targets[:, 0])
    continuous: dict[str, Any] = {}
    for index, name in enumerate(names[1:], start=1):
        available = mask[:, index] > 0
        if not np.any(available):
            continuous[name] = {"rows": 0, "model_mae": None, "zero_baseline_mae": None}
            continue
        continuous[name] = {
            "rows": int(available.sum()),
            "model_mae": float(
                np.mean(np.abs(predictions[available, index] - targets[available, index]))
            ),
            "zero_baseline_mae": float(np.mean(np.abs(targets[available, index]))),
        }
    return {
        "rows": len(targets),
        "temperature": temperature,
        "resolved_yes": {
            "model": resolution,
            "market_baseline": baseline_resolution,
            "brier_delta_vs_market": resolution["brier"] - baseline_resolution["brier"],
            "log_loss_delta_vs_market": (resolution["log_loss"] - baseline_resolution["log_loss"]),
        },
        "continuous": continuous,
    }


def _collect(
    model: torch.nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
    names: list[str],
    weights: dict[str, float],
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    model.eval()
    losses: list[float] = []
    predictions_out: list[NDArray[np.float64]] = []
    targets_out: list[NDArray[np.float64]] = []
    masks_out: list[NDArray[np.float64]] = []
    baselines_out: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for features, token_types, targets, mask, baselines in loader:
            features = features.to(device, dtype=torch.bfloat16, non_blocking=True)
            token_types = token_types.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            mask_device = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(features, token_types)
                loss, _ = _loss(predictions.float(), targets_device, mask_device, names, weights)
            losses.append(float(loss.cpu()))
            predictions_out.append(predictions.float().cpu().numpy().astype(np.float64))
            targets_out.append(targets.numpy().astype(np.float64))
            masks_out.append(mask.numpy().astype(np.float64))
            baselines_out.append(baselines.numpy().astype(np.float64))
    return (
        statistics.fmean(losses),
        np.concatenate(predictions_out),
        np.concatenate(targets_out),
        np.concatenate(masks_out),
        np.concatenate(baselines_out),
    )


def _fit_temperature(logits: NDArray[np.float64], labels: NDArray[np.float64]) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(0.05), math.log(10.0)).exp()
        loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, labels_tensor)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return float(log_temperature.detach().clamp(math.log(0.05), math.log(10.0)).exp())


def train(
    config_path: Path,
    pack_dir: Path,
    output_dir: Path,
    *,
    compile_mode: str | None,
    wallet_mode: str = "original",
    output_names: list[str] | None = None,
    research_config_path: Path | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Trial T0 training")
    config = load_json(config_path)
    metadata = json.loads((pack_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("valid") is not True:
        raise RuntimeError("A valid feature-pack receipt is required")
    if int(metadata.get("test_rows_consumed", -1)) != 0:
        raise RuntimeError("Trial T0 must not consume test rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    registered_names = [str(value) for value in config["targets"]["output_order"]]
    names = registered_names if output_names is None else output_names
    if not names or names[0] != "resolved_yes":
        raise ValueError("The first output must be resolved_yes")
    if any(name not in registered_names for name in names):
        raise ValueError("Output override is outside the registered target contract")
    research_config = load_json(research_config_path) if research_config_path is not None else None
    research_id = str(
        research_config["research_id"] if research_config is not None else config["research_id"]
    )
    experiment_id = str(research_config["id"] if research_config is not None else config["id"])
    runtime_config = deepcopy(config)
    runtime_config["model"]["output_width"] = len(names)
    weights = {name: float(training["loss_weights"][name]) for name in names}
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    batch_size = int(training["batch_size"])
    registered_workers = int(training["loader_workers"])
    workers = 0 if platform.system() == "Windows" else registered_workers
    worker_adjustment = (
        "Single-process loading on Windows avoids spawned torch DLL failures"
        if workers != registered_workers
        else None
    )
    wallet_start = int(config["features"]["market_trade_tokens"])
    wallet_tokens = int(config["features"]["wallet_tokens"])
    train_dataset = PackedSplit(
        pack_dir / "train",
        wallet_mode=wallet_mode,
        wallet_start=wallet_start,
        wallet_tokens=wallet_tokens,
    )
    validation_dataset = PackedSplit(
        pack_dir / "validation",
        wallet_mode=wallet_mode,
        wallet_start=wallet_start,
        wallet_tokens=wallet_tokens,
    )
    calibration_dataset = PackedSplit(
        pack_dir / "calibration",
        wallet_mode=wallet_mode,
        wallet_start=wallet_start,
        wallet_tokens=wallet_tokens,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    calibration_loader = _loader(
        calibration_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    raw_model = SphinxTraceS0(
        runtime_config,
        sequence_length=int(config["features"]["sequence_length"]),
        feature_width=int(config["features"]["feature_width"]),
    ).to(device)
    parameters = parameter_count(raw_model)
    if (
        not int(config["model"]["parameter_minimum"])
        <= parameters
        <= int(config["model"]["parameter_maximum"])
    ):
        raise RuntimeError(f"Parameter count outside registered range: {parameters}")

    registered_compile = str(training["compile_mode"])
    requested_compile = registered_compile if compile_mode is None else compile_mode
    effective_compile = requested_compile
    compile_adjustment: str | None = None
    if platform.system() == "Windows" and requested_compile == "max-autotune":
        effective_compile = "max-autotune-no-cudagraphs"
        compile_adjustment = "CUDA Graphs disabled on Windows; max autotuning retained"
    model: torch.nn.Module = raw_model
    compile_status = "disabled"
    compile_error: str | None = None
    if effective_compile not in {"", "none", "disabled"}:
        try:
            model = cast(
                torch.nn.Module,
                torch.compile(raw_model, mode=effective_compile, dynamic=False),
            )
            compile_status = "requested"
        except Exception as exc:  # pragma: no cover - platform dependent
            compile_status = "fallback_eager"
            compile_error = f"{type(exc).__name__}: {exc}"

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["maximum_epochs"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        for features, token_types, targets, mask, _ in train_loader:
            features = features.to(device, dtype=torch.bfloat16, non_blocking=True)
            token_types = token_types.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = model(features, token_types)
                    loss, _ = _loss(predictions.float(), targets, mask, names, weights)
                loss.backward()  # type: ignore[no-untyped-call]
            except Exception as exc:
                if compile_status != "requested":
                    raise
                compile_status = "fallback_eager"
                compile_error = f"{type(exc).__name__}: {exc}"
                model = raw_model
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = model(features, token_types)
                    loss, _ = _loss(predictions.float(), targets, mask, names, weights)
                loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            if compile_status == "requested":
                compile_status = "active"
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss")
            epoch_losses.append(value)
        validation_loss, _, _, _, _ = _collect(
            model,
            validation_loader,
            device,
            names,
            weights,
        )
        train_loss = statistics.fmean(epoch_losses)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": "1.0.0",
                "status": "training",
                "research_id": research_id,
                "variant_id": variant_id,
                "test_rows_consumed": 0,
                "parameters": parameters,
                "history": history,
            },
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint was selected")
    raw_model.load_state_dict(best_state)
    model = raw_model
    checkpoint_path = output_dir / "best_model_state.pt"
    torch.save(best_state, checkpoint_path)

    (
        validation_loss,
        validation_predictions,
        validation_targets,
        validation_mask,
        validation_baselines,
    ) = _collect(
        model,
        validation_loader,
        device,
        names,
        weights,
    )
    (
        calibration_loss,
        calibration_predictions,
        calibration_targets,
        calibration_mask,
        calibration_baselines,
    ) = _collect(
        model,
        calibration_loader,
        device,
        names,
        weights,
    )
    temperature = _fit_temperature(
        calibration_predictions[:, 0],
        calibration_targets[:, 0],
    )
    validation_metrics = _metrics(
        validation_predictions,
        validation_targets,
        validation_mask,
        validation_baselines,
        names,
    )
    calibration_metrics_uncalibrated = _metrics(
        calibration_predictions,
        calibration_targets,
        calibration_mask,
        calibration_baselines,
        names,
    )
    calibration_metrics = _metrics(
        calibration_predictions,
        calibration_targets,
        calibration_mask,
        calibration_baselines,
        names,
        temperature=temperature,
    )
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        validation_logits=validation_predictions[:, 0],
        validation_labels=validation_targets[:, 0],
        validation_market_probability=validation_baselines[:, 0],
        calibration_logits=calibration_predictions[:, 0],
        calibration_labels=calibration_targets[:, 0],
        calibration_market_probability=calibration_baselines[:, 0],
    )
    finite_losses = all(
        math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["validation_loss"]))
        for row in history
    )
    train_loss_declined = float(history[-1]["train_loss"]) < float(history[0]["train_loss"])
    gates = {
        "causal_feature_violations_zero": (
            int(metadata["features"]["feature_time_violations"]) == 0
        ),
        "event_overlap_zero": int(metadata["features"]["event_overlap_count"]) == 0,
        "test_rows_consumed_zero": int(metadata["test_rows_consumed"]) == 0,
        "parameter_count_in_range": (
            int(config["model"]["parameter_minimum"])
            <= parameters
            <= int(config["model"]["parameter_maximum"])
        ),
        "finite_losses": finite_losses,
        "training_loss_declined": train_loss_declined,
        "baseline_metrics_recorded": True,
        "control_donor_time_violations_zero": all(
            int(dataset.control_audit["donor_time_violations"]) == 0
            for dataset in (train_dataset, validation_dataset, calibration_dataset)
        ),
        "control_same_event_donors_zero": all(
            int(dataset.control_audit["same_event_donors"]) == 0
            for dataset in (train_dataset, validation_dataset, calibration_dataset)
        ),
    }
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "completed_at": now_utc(),
        "valid": all(gates.values()),
        "research_id": research_id,
        "config_id": experiment_id,
        "config_sha256": sha256_file(config_path),
        "research_config_sha256": (
            sha256_file(research_config_path) if research_config_path is not None else None
        ),
        "variant": {
            "id": variant_id,
            "wallet_mode": wallet_mode,
            "outputs": names,
        },
        "feature_pack_sha256": sha256_file(pack_dir / "metadata.json"),
        "test_labels_opened": False,
        "test_rows_consumed": 0,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "model": {
            **runtime_config["model"],
            "parameters": parameters,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_validation_loss": best_validation,
            "final_validation_loss": validation_loss,
            "calibration_loss": calibration_loss,
            "compile_registered": registered_compile,
            "compile_requested": requested_compile,
            "compile_effective": effective_compile,
            "compile_adjustment": compile_adjustment,
            "compile_status": compile_status,
            "compile_error": compile_error,
            "loader_workers_registered": registered_workers,
            "loader_workers_effective": workers,
            "loader_worker_adjustment": worker_adjustment,
            "temperature": temperature,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "metrics": {
            "validation": validation_metrics,
            "calibration_uncalibrated": calibration_metrics_uncalibrated,
            "calibration": calibration_metrics,
        },
        "control_audit": {
            "train": train_dataset.control_audit,
            "validation": validation_dataset.control_audit,
            "calibration": calibration_dataset.control_audit,
        },
        "predictions": {
            "path": predictions_path.name,
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
            "validation_rows": len(validation_predictions),
            "calibration_rows": len(calibration_predictions),
        },
        "gates": gates,
        "decision": "diagnostic_only_no_promotion",
        "evidence_boundary": str(
            research_config["evidence_boundary"]
            if research_config is not None
            else config["evidence_boundary"]
        ),
    }
    atomic_json(output_dir / "result.json", result)
    atomic_json(
        output_dir / "calibration.json",
        {
            "schema_version": "1.0.0",
            "method": str(training["calibration_method"]),
            "temperature": temperature,
            "checkpoint_sha256": result["model"]["checkpoint_sha256"],
        },
    )
    if not result["valid"]:
        raise RuntimeError(f"Trial T0 gates failed; see {output_dir / 'result.json'}")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--pack-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--compile-mode")
    root.add_argument(
        "--wallet-mode",
        choices=("original", "zero", "prior_event_control"),
        default="original",
    )
    root.add_argument("--outputs", nargs="+")
    root.add_argument("--research-config", type=Path)
    root.add_argument("--variant-id")
    root.add_argument("--quiet-result", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.pack_dir.resolve(),
        args.output_dir.resolve(),
        compile_mode=args.compile_mode,
        wallet_mode=args.wallet_mode,
        output_names=args.outputs,
        research_config_path=(
            args.research_config.resolve() if args.research_config is not None else None
        ),
        variant_id=args.variant_id,
    )
    if args.quiet_result:
        print(
            json.dumps(
                {
                    "valid": result["valid"],
                    "research_id": result["research_id"],
                    "variant": result["variant"],
                    "parameters": result["model"]["parameters"],
                    "best_epoch": result["training"]["best_epoch"],
                    "validation": result["metrics"]["validation"]["resolved_yes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
