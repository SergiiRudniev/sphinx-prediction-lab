"""Train the cross-fitted H022 neural/tree conditional net-edge ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.h022_features import (
    H022_TREE_FEATURE_NAMES,
    H022_TREE_FEATURE_WIDTH,
    component_folds,
    wallet_ablation,
)
from sphinx_trace.h022_training import (
    calibration_metrics,
    economic_policy_metrics,
    fit_weighted_ridge,
    h022_neural_loss,
    predict_weighted_ridge,
)
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h022 import SphinxTraceS0H022NeuralMember

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h022_experiment_v1.json"
)
DEFAULT_REGISTRATION = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h022_conditional_net_edge_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h022_features.py",
    ROOT / "src" / "sphinx_trace" / "h022_training.py",
    ROOT / "src" / "sphinx_trace" / "model_h022.py",
)
PREDICTOR_NAMES = (
    "neural_mean_net_return",
    "neural_q10",
    "neural_q50",
    "neural_q90",
    "neural_fill_probability",
    "neural_calibrated_edge",
    "tree_net_return",
)
TERMINAL_LOGIT_INDEX = H022_TREE_FEATURE_NAMES.index("outcome.terminal_logit")
BREAK_EVEN_INDEX = H022_TREE_FEATURE_NAMES.index(
    "candidate.break_even_probability"
)


class PauseRequested(RuntimeError):
    pass


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _indices_digest(indices: NDArray[np.int64]) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def _load_partition(pack_dir: Path, partition: str) -> dict[str, NDArray[Any]]:
    names = (
        "tree_features",
        "market_latents",
        "target_net_log_utility",
        "target_reference_net_log_utility",
        "position_size_fractions",
        "target_outcome0",
        "target_fill_fraction",
        "fill_target_mask",
        "sample_weights",
        "candidate_action_ids",
        "component_ids",
        "market_ids",
        "week_ids",
        "behavior_policy_codes",
        "entry_prices",
        "timestamps",
    )
    values = {
        name: np.load(
            pack_dir / partition / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in names
    }
    rows = len(values["target_net_log_utility"])
    if any(len(value) != rows for value in values.values()):
        raise RuntimeError(f"H022 {partition} arrays no longer align")
    if values["tree_features"].shape != (rows, H022_TREE_FEATURE_WIDTH):
        raise RuntimeError(f"H022 {partition} feature width changed")
    return values


def _statistics(
    values: NDArray[Any], indices: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    selected = np.asarray(values[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    return mean, scale


def _neural_batch(
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    device: torch.device,
    statistics: dict[str, NDArray[np.float32]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    latent = torch.from_numpy(
        np.asarray(data["market_latents"][indices], dtype=np.float32)
    ).to(device, non_blocking=True)
    tree = torch.from_numpy(
        np.asarray(data["tree_features"][indices], dtype=np.float32)
    ).to(device, non_blocking=True)
    latent = (
        latent
        - torch.from_numpy(statistics["latent_mean"]).to(device).unsqueeze(0)
    ) / torch.from_numpy(statistics["latent_scale"]).to(device).unsqueeze(0)
    tree = (
        tree - torch.from_numpy(statistics["tree_mean"]).to(device).unsqueeze(0)
    ) / torch.from_numpy(statistics["tree_scale"]).to(device).unsqueeze(0)
    raw_tree = np.asarray(data["tree_features"][indices], dtype=np.float32)
    terminal = torch.from_numpy(raw_tree[:, TERMINAL_LOGIT_INDEX]).to(
        device, non_blocking=True
    )
    action = torch.from_numpy(
        np.asarray(data["candidate_action_ids"][indices], dtype=np.int64)
    ).to(device, non_blocking=True)
    break_even = torch.from_numpy(raw_tree[:, BREAK_EVEN_INDEX]).to(
        device, non_blocking=True
    )
    return latent, tree, terminal, action, break_even


def _target_batch(
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        torch.from_numpy(np.asarray(data[name][indices], dtype=np.float32)).to(
            device, non_blocking=True
        )
        for name in (
            "target_net_log_utility",
            "target_outcome0",
            "target_fill_fraction",
            "fill_target_mask",
            "sample_weights",
        )
    )  # type: ignore[return-value]


@torch.inference_mode()
def _predict_neural(
    model: SphinxTraceS0H022NeuralMember,
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    device: torch.device,
    statistics: dict[str, NDArray[np.float32]],
    batch_size: int,
    *,
    return_attention: bool,
) -> dict[str, NDArray[np.float32]]:
    model.eval()
    parts: dict[str, list[NDArray[np.float32]]] = {
        "mean": [],
        "quantiles": [],
        "fill": [],
        "probability0": [],
        "edge": [],
        "attention": [],
    }
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset : offset + batch_size]
        inputs = _neural_batch(data, selected, device, statistics)
        output = model(*inputs, return_debug=return_attention)
        parts["mean"].append(output["net_return_mean"].float().cpu().numpy())
        parts["quantiles"].append(
            output["net_return_quantiles"].float().cpu().numpy()
        )
        parts["fill"].append(output["fill_probability"].float().cpu().numpy())
        parts["probability0"].append(
            output["calibrated_outcome_probability0"].float().cpu().numpy()
        )
        parts["edge"].append(
            output["calibrated_candidate_edge"].float().cpu().numpy()
        )
        if return_attention:
            parts["attention"].append(
                output["debug_group_attention"].float().cpu().numpy()
            )
    result = {
        name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for name, values in parts.items()
        if values
    }
    if any(len(value) != len(indices) for value in result.values()):
        raise RuntimeError("H022 neural prediction rows changed")
    return result


def _fit_neural(
    data: dict[str, NDArray[Any]],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64] | None,
    statistics_indices: NDArray[np.int64],
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    pause_file: Path,
    *,
    fixed_epochs: int | None = None,
) -> tuple[SphinxTraceS0H022NeuralMember, dict[str, NDArray[np.float32]], dict[str, Any]]:
    neural_config = config["neural_member"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    statistics = {
        "latent_mean": _statistics(
            data["market_latents"], statistics_indices
        )[0],
        "latent_scale": _statistics(
            data["market_latents"], statistics_indices
        )[1],
        "tree_mean": _statistics(data["tree_features"], statistics_indices)[0],
        "tree_scale": _statistics(data["tree_features"], statistics_indices)[1],
    }
    model = SphinxTraceS0H022NeuralMember(neural_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(neural_config["learning_rate"]),
        weight_decay=float(neural_config["weight_decay"]),
    )
    epochs = int(fixed_epochs or neural_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(neural_config["minimum_learning_rate"]),
    )
    contract_payload = {
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "implementation_sha256": _implementation_digest(),
        "seed": seed,
        "train_indices_sha256": _indices_digest(train_indices),
        "validation_indices_sha256": (
            None
            if validation_indices is None
            else _indices_digest(validation_indices)
        ),
        "statistics_indices_sha256": _indices_digest(statistics_indices),
        "fixed_epochs": fixed_epochs,
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_epoch = 0
    best_epoch = -1
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"H022 neural checkpoint contract changed: {output_dir}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        best_state = checkpoint.get("best_state")
        stale = int(checkpoint["stale"])
        history = list(checkpoint["history"])
    batch_size = int(neural_config["batch_size"])
    patience = int(neural_config["early_stopping_patience"])
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        shuffled = np.array(train_indices, copy=True)
        np.random.default_rng(seed ^ epoch).shuffle(shuffled)
        total_loss = 0.0
        total_weight = 0.0
        for offset in range(0, len(shuffled), batch_size):
            selected = shuffled[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            inputs = _neural_batch(data, selected, device, statistics)
            targets = _target_batch(data, selected, device)
            with autocast:
                output = model(*inputs)
                loss, metrics = h022_neural_loss(
                    output, *targets, neural_config
                )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(neural_config["gradient_clip_norm"])
            )
            optimizer.step()
            weight = float(metrics["weight_sum"])
            total_loss += float(loss.detach()) * weight
            total_weight += weight
        scheduler.step()
        if validation_indices is None:
            score = -total_loss / max(total_weight, 1e-8)
            validation_receipt: dict[str, Any] = {}
        else:
            prediction = _predict_neural(
                model,
                data,
                validation_indices,
                device,
                statistics,
                batch_size,
                return_attention=False,
            )
            target = np.asarray(
                data["target_net_log_utility"][validation_indices],
                dtype=np.float64,
            )
            weights = np.asarray(
                data["sample_weights"][validation_indices], dtype=np.float64
            )
            keep = prediction["mean"] > 0.0
            score = float(np.sum(target * weights * keep) / np.sum(weights))
            validation_receipt = {
                "weighted_chosen_utility": score,
                "calls": int(keep.sum()),
                "rows": len(validation_indices),
            }
        improved = score > best_score + 1e-12
        if improved or best_state is None:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "fit_loss": total_loss / max(total_weight, 1e-8),
                "learning_rate": scheduler.get_last_lr()[0],
                "validation": validation_receipt,
                "best_epoch": best_epoch,
            }
        )
        _atomic_torch_save(
            checkpoint_path,
            {
                "contract_sha256": contract_sha256,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_state": best_state,
                "stale": stale,
                "history": history,
            },
        )
        if pause_file.is_file():
            raise PauseRequested(f"H022 paused after neural epoch {epoch}")
        if validation_indices is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("H022 neural training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    artifact_path = output_dir / "best-neural.pt"
    _atomic_torch_save(
        artifact_path,
        {
            "record_type": "h022_neural_member",
            "contract_sha256": contract_sha256,
            "model": best_state,
            "statistics": statistics,
            "config": neural_config,
            "best_epoch": best_epoch,
        },
    )
    return model, statistics, {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "artifact_sha256": sha256_file(artifact_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameters": parameter_count(model),
        "contract_sha256": contract_sha256,
    }


def _tree_parameters(config: dict[str, Any], seed: int) -> dict[str, Any]:
    source = config["tree_member"]
    names = (
        "objective",
        "metric",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "device_type",
    )
    return {
        **{name: source[name] for name in names},
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": int(source["num_threads"]),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_tree_with_validation(
    features: NDArray[np.float32],
    target: NDArray[np.float32],
    weights: NDArray[np.float32],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64],
    config: dict[str, Any],
    seed: int,
) -> tuple[lgb.Booster, int]:
    train = lgb.Dataset(
        np.asarray(features[train_indices], dtype=np.float32),
        label=np.asarray(target[train_indices], dtype=np.float32),
        weight=np.asarray(weights[train_indices], dtype=np.float32),
        feature_name=list(H022_TREE_FEATURE_NAMES),
        free_raw_data=True,
    )
    validation = lgb.Dataset(
        np.asarray(features[validation_indices], dtype=np.float32),
        label=np.asarray(target[validation_indices], dtype=np.float32),
        weight=np.asarray(weights[validation_indices], dtype=np.float32),
        feature_name=list(H022_TREE_FEATURE_NAMES),
        reference=train,
        free_raw_data=True,
    )
    booster = lgb.train(
        _tree_parameters(config, seed),
        train,
        num_boost_round=int(config["tree_member"]["maximum_rounds"]),
        valid_sets=[validation],
        valid_names=["inner_validation"],
        callbacks=[
            lgb.early_stopping(
                int(config["tree_member"]["early_stopping_rounds"]),
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(booster.best_iteration or booster.current_iteration())
    return booster, best_iteration


def _fit_tree_fixed(
    features: NDArray[np.float32],
    target: NDArray[np.float32],
    weights: NDArray[np.float32],
    indices: NDArray[np.int64],
    rounds: int,
    config: dict[str, Any],
    seed: int,
) -> lgb.Booster:
    dataset = lgb.Dataset(
        np.asarray(features[indices], dtype=np.float32),
        label=np.asarray(target[indices], dtype=np.float32),
        weight=np.asarray(weights[indices], dtype=np.float32),
        feature_name=list(H022_TREE_FEATURE_NAMES),
        free_raw_data=True,
    )
    return lgb.train(
        _tree_parameters(config, seed),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _predictor_matrix(
    neural: dict[str, NDArray[np.float32]], tree: NDArray[np.float64]
) -> NDArray[np.float64]:
    quantiles = neural["quantiles"].astype(np.float64)
    return np.column_stack(
        (
            neural["mean"],
            quantiles[:, 0],
            quantiles[:, 1],
            quantiles[:, 2],
            neural["fill"],
            neural["edge"],
            tree,
        )
    ).astype(np.float64, copy=False)


def _metrics(
    data: dict[str, NDArray[Any]],
    predicted: NDArray[np.float64],
    source_rows: int,
) -> dict[str, Any]:
    return economic_policy_metrics(
        np.asarray(data["target_net_log_utility"], dtype=np.float64),
        predicted,
        np.asarray(data["sample_weights"], dtype=np.float64),
        np.asarray(data["component_ids"], dtype=np.int64),
        np.asarray(data["week_ids"], dtype=np.int64),
        np.asarray(data["behavior_policy_codes"], dtype=np.uint8),
        np.asarray(data["entry_prices"], dtype=np.float64),
        source_rows=source_rows,
    )


def _train_seed(
    config: dict[str, Any],
    candidate_manifest: dict[str, Any],
    fit: dict[str, NDArray[Any]],
    selection: dict[str, NDArray[Any]],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_dir = output_dir / f"seed={seed}"
    result_path = seed_dir / "result.json"
    if result_path.is_file():
        payload: object = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("valid") is True:
            return payload
    seed_dir.mkdir(parents=True, exist_ok=True)
    pause_file = output_dir / str(config["resume"]["graceful_pause_file"])
    folds = int(config["cross_fitting"]["folds"])
    component_ids = np.asarray(fit["component_ids"], dtype=np.int64)
    fold_codes = component_folds(component_ids, folds, seed)
    all_fit = np.arange(len(component_ids), dtype=np.int64)
    target = np.asarray(fit["target_net_log_utility"], dtype=np.float32)
    weights = np.asarray(fit["sample_weights"], dtype=np.float32)
    fit_features = np.asarray(fit["tree_features"], dtype=np.float32)
    selection_features = np.asarray(selection["tree_features"], dtype=np.float32)
    oof_neural: dict[str, NDArray[np.float32]] = {
        "mean": np.full(len(all_fit), np.nan, dtype=np.float32),
        "quantiles": np.full((len(all_fit), 3), np.nan, dtype=np.float32),
        "fill": np.full(len(all_fit), np.nan, dtype=np.float32),
        "probability0": np.full(len(all_fit), np.nan, dtype=np.float32),
        "edge": np.full(len(all_fit), np.nan, dtype=np.float32),
    }
    oof_tree = np.full(len(all_fit), np.nan, dtype=np.float64)
    fold_receipts: list[dict[str, Any]] = []
    neural_epochs: list[int] = []
    tree_rounds: list[int] = []
    for fold in range(folds):
        holdout = np.flatnonzero(fold_codes == fold).astype(np.int64)
        pool = np.flatnonzero(fold_codes != fold).astype(np.int64)
        inner_codes = component_folds(component_ids[pool], 10, seed ^ (fold + 101))
        inner_validation = pool[inner_codes == 0]
        train_indices = pool[inner_codes != 0]
        if not len(holdout) or not len(inner_validation) or not len(train_indices):
            raise RuntimeError(f"H022 seed {seed} fold {fold} is empty")
        model, statistics, neural_receipt = _fit_neural(
            fit,
            train_indices,
            inner_validation,
            pool,
            config,
            seed ^ (fold + 1),
            seed_dir / "cross-fit" / f"fold={fold}" / "neural",
            pause_file,
        )
        prediction = _predict_neural(
            model,
            fit,
            holdout,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            statistics,
            int(config["neural_member"]["batch_size"]),
            return_attention=False,
        )
        for name in oof_neural:
            oof_neural[name][holdout] = prediction[name]
        tree, best_rounds = _fit_tree_with_validation(
            fit_features,
            target,
            weights,
            train_indices,
            inner_validation,
            config,
            seed ^ (fold + 1),
        )
        tree_path = seed_dir / "cross-fit" / f"fold={fold}" / "tree.txt"
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree.save_model(tree_path, num_iteration=best_rounds)
        oof_tree[holdout] = tree.predict(
            fit_features[holdout], num_iteration=best_rounds
        )
        neural_epochs.append(int(neural_receipt["best_epoch"]) + 1)
        tree_rounds.append(best_rounds)
        fold_receipts.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "inner_validation_rows": len(inner_validation),
                "holdout_rows": len(holdout),
                "train_components": int(np.unique(component_ids[train_indices]).size),
                "holdout_components": int(np.unique(component_ids[holdout]).size),
                "neural": neural_receipt,
                "tree_best_iteration": best_rounds,
                "tree_sha256": sha256_file(tree_path),
            }
        )
        atomic_json(
            seed_dir / "progress.json",
            {
                "record_type": "h022_seed_progress",
                "seed": seed,
                "cross_fit_folds_complete": fold + 1,
                "cross_fit_folds_total": folds,
            },
        )
        if pause_file.is_file():
            raise PauseRequested(f"H022 paused after seed {seed} fold {fold}")
    if any(bool(np.isnan(value).any()) for value in oof_neural.values()) or bool(
        np.isnan(oof_tree).any()
    ):
        raise RuntimeError(f"H022 seed {seed} OOF predictions are incomplete")
    oof_predictors = _predictor_matrix(oof_neural, oof_tree)
    decision_weights = weights.astype(np.float64) * np.where(target < 0.0, 2.0, 1.0)
    stacker = fit_weighted_ridge(
        oof_predictors,
        target.astype(np.float64),
        decision_weights,
        float(config["stacking"]["ridge"]),
    )
    stacker["predictor_names"] = list(PREDICTOR_NAMES)
    oof_ensemble = predict_weighted_ridge(oof_predictors, stacker)
    final_epochs = max(1, round(float(np.median(neural_epochs))))
    final_rounds = max(1, round(float(np.median(tree_rounds))))
    final_model, final_statistics, final_neural_receipt = _fit_neural(
        fit,
        all_fit,
        None,
        all_fit,
        config,
        seed ^ 0x5A17,
        seed_dir / "final" / "neural",
        pause_file,
        fixed_epochs=final_epochs,
    )
    final_tree = _fit_tree_fixed(
        fit_features,
        target,
        weights,
        all_fit,
        final_rounds,
        config,
        seed ^ 0x7AEE,
    )
    final_tree_path = seed_dir / "final" / "tree.txt"
    final_tree_path.parent.mkdir(parents=True, exist_ok=True)
    final_tree.save_model(final_tree_path, num_iteration=final_rounds)
    selection_indices = np.arange(
        len(selection["target_net_log_utility"]), dtype=np.int64
    )
    selection_neural = _predict_neural(
        final_model,
        selection,
        selection_indices,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        final_statistics,
        int(config["neural_member"]["batch_size"]),
        return_attention=True,
    )
    selection_tree = np.asarray(
        final_tree.predict(selection_features, num_iteration=final_rounds),
        dtype=np.float64,
    )
    selection_predictors = _predictor_matrix(selection_neural, selection_tree)
    selection_ensemble = predict_weighted_ridge(selection_predictors, stacker)
    source_fit_rows = int(candidate_manifest["partitions"]["fit"]["source_rows"])
    source_selection_rows = int(
        candidate_manifest["partitions"]["selection"]["source_rows"]
    )
    baseline_fit = np.full(len(all_fit), np.finfo(np.float64).eps)
    baseline_selection = np.full(len(selection_indices), np.finfo(np.float64).eps)
    metrics = {
        "oof": {
            "H021_keep_all": _metrics(fit, baseline_fit, source_fit_rows),
            "neural": _metrics(
                fit, oof_neural["mean"].astype(np.float64), source_fit_rows
            ),
            "tree": _metrics(fit, oof_tree, source_fit_rows),
            "ensemble": _metrics(fit, oof_ensemble, source_fit_rows),
            "calibration": calibration_metrics(
                oof_neural["probability0"].astype(np.float64),
                np.asarray(fit["target_outcome0"], dtype=np.float64),
                weights.astype(np.float64),
            ),
        },
        "selection": {
            "H021_keep_all": _metrics(
                selection, baseline_selection, source_selection_rows
            ),
            "neural": _metrics(
                selection,
                selection_neural["mean"].astype(np.float64),
                source_selection_rows,
            ),
            "tree": _metrics(selection, selection_tree, source_selection_rows),
            "ensemble": _metrics(
                selection, selection_ensemble, source_selection_rows
            ),
            "calibration": calibration_metrics(
                selection_neural["probability0"].astype(np.float64),
                np.asarray(selection["target_outcome0"], dtype=np.float64),
                np.asarray(selection["sample_weights"], dtype=np.float64),
            ),
        },
    }
    ablations: dict[str, Any] = {}
    for mode in ("zero", "shuffle"):
        ablated_fit = wallet_ablation(fit_features, mode, seed=seed ^ 0xA811)
        ablated_selection = wallet_ablation(
            selection_features, mode, seed=seed ^ 0xA812
        )
        ablated_tree = _fit_tree_fixed(
            ablated_fit,
            target,
            weights,
            all_fit,
            final_rounds,
            config,
            seed ^ (0xB000 + len(mode)),
        )
        ablated_tree_prediction = np.asarray(
            ablated_tree.predict(
                ablated_selection, num_iteration=final_rounds
            ),
            dtype=np.float64,
        )
        ablated_data = dict(selection)
        ablated_data["tree_features"] = ablated_selection
        ablated_neural = _predict_neural(
            final_model,
            ablated_data,
            selection_indices,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            final_statistics,
            int(config["neural_member"]["batch_size"]),
            return_attention=False,
        )
        ablated_ensemble = predict_weighted_ridge(
            _predictor_matrix(ablated_neural, ablated_tree_prediction), stacker
        )
        ablated_tree_path = seed_dir / "ablations" / f"wallet-{mode}-tree.txt"
        ablated_tree_path.parent.mkdir(parents=True, exist_ok=True)
        ablated_tree.save_model(ablated_tree_path, num_iteration=final_rounds)
        ablations[f"wallet_{mode}"] = {
            "tree": _metrics(
                selection, ablated_tree_prediction, source_selection_rows
            ),
            "ensemble": _metrics(
                selection, ablated_ensemble, source_selection_rows
            ),
            "tree_sha256": sha256_file(ablated_tree_path),
        }
    importance_gain = final_tree.feature_importance(
        importance_type="gain", iteration=final_rounds
    )
    importance_split = final_tree.feature_importance(
        importance_type="split", iteration=final_rounds
    )
    importance: list[dict[str, Any]] = sorted(
        (
            {
                "feature": name,
                "gain": float(gain),
                "split": int(split),
            }
            for name, gain, split in zip(
                H022_TREE_FEATURE_NAMES,
                importance_gain,
                importance_split,
                strict=True,
            )
        ),
        key=lambda value: float(value["gain"]),
        reverse=True,
    )
    stacker_path = seed_dir / "final" / "stacker.json"
    atomic_json(stacker_path, stacker)
    prediction_path = seed_dir / "selection-debug.npz"
    np.savez_compressed(
        prediction_path,
        target_net_log_utility=np.asarray(
            selection["target_net_log_utility"], dtype=np.float32
        ),
        entry_prices=np.asarray(selection["entry_prices"], dtype=np.float32),
        component_ids=np.asarray(selection["component_ids"], dtype=np.int64),
        neural_mean=selection_neural["mean"],
        neural_quantiles=selection_neural["quantiles"],
        neural_fill_probability=selection_neural["fill"],
        neural_probability0=selection_neural["probability0"],
        neural_calibrated_edge=selection_neural["edge"],
        neural_group_attention=selection_neural["attention"],
        tree_net_return=selection_tree,
        ensemble_net_return=selection_ensemble,
        ensemble_keep=(selection_ensemble > 0.0).astype(np.uint8),
    )
    result = {
        "record_type": "h022_conditional_net_edge_seed_result",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H022",
        "seed": seed,
        "valid": True,
        "completed_at": now_utc(),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "implementation_sha256": _implementation_digest(),
        "folds": fold_receipts,
        "final_neural_epochs": final_epochs,
        "final_tree_rounds": final_rounds,
        "final_neural": final_neural_receipt,
        "final_tree_sha256": sha256_file(final_tree_path),
        "stacker_sha256": sha256_file(stacker_path),
        "selection_debug_sha256": sha256_file(prediction_path),
        "stacker": stacker,
        "metrics": metrics,
        "ablations": ablations,
        "tree_feature_importance": importance,
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Cross-fitted H017 development selection only; exact stateful replay, "
            "untouched test and paper-forward evidence remain required."
        ),
    }
    atomic_json(result_path, result)
    (seed_dir / "progress.json").unlink(missing_ok=True)
    return result


def train(
    config_path: Path,
    candidate_pack_dir: Path,
    output_dir: Path,
    seeds: list[int] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    candidate_manifest_path = candidate_pack_dir / "manifest.json"
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    candidate_manifest["manifest_sha256"] = sha256_file(candidate_manifest_path)
    if (
        config.get("research_id") != "SPH-T-H022"
        or candidate_manifest.get("valid") is not True
        or candidate_manifest.get("config_sha256") != sha256_file(config_path)
        or candidate_manifest.get("test_labels_opened") is not False
        or int(candidate_manifest.get("test_rows_consumed", -1)) != 0
        or int(candidate_manifest.get("calibration_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H022 training source contract changed")
    registered = [int(value) for value in config["cross_fitting"]["seeds"]]
    selected_seeds = registered if seeds is None else seeds
    if not selected_seeds or any(seed not in registered for seed in selected_seeds):
        raise RuntimeError("H022 requested seed was not registered")
    fit = _load_partition(candidate_pack_dir, "fit")
    selection = _load_partition(candidate_pack_dir, "selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        for seed in selected_seeds:
            results.append(
                _train_seed(
                    config,
                    candidate_manifest,
                    fit,
                    selection,
                    output_dir,
                    seed,
                )
            )
    except PauseRequested as exc:
        progress = {
            "record_type": "h022_training_progress",
            "status": "paused",
            "message": str(exc),
            "completed_seeds": [int(value["seed"]) for value in results],
            "requested_seeds": selected_seeds,
        }
        atomic_json(output_dir / "progress.json", progress)
        return progress
    registration = load_json(DEFAULT_REGISTRATION)
    minimum_calls = int(registration["acceptance"]["minimum_calls"])
    minimum_components = int(
        registration["acceptance"]["minimum_independent_components"]
    )
    for result in results:
        debug_path = (
            output_dir / f"seed={result['seed']}" / "selection-debug.npz"
        )
        with np.load(debug_path, allow_pickle=False) as debug:
            keep = np.asarray(debug["ensemble_keep"], dtype=np.uint8).astype(bool)
        kept_components = int(
            np.unique(
                np.asarray(selection["component_ids"], dtype=np.int64)[keep]
            ).size
        )
        selection_metrics = result["metrics"]["selection"]["ensemble"]
        selection_metrics["kept_independent_components"] = kept_components
        result["selection_breadth_eligible"] = bool(
            int(selection_metrics["calls"]) >= minimum_calls
            and kept_components >= minimum_components
        )
        atomic_json(
            output_dir / f"seed={result['seed']}" / "result.json", result
        )
    eligible = [value for value in results if value["selection_breadth_eligible"]]
    best_pool = eligible or results
    best = max(
        best_pool,
        key=lambda value: value["metrics"]["selection"]["ensemble"][
            "equal_market_mean_protocol_exact_chosen_utility"
        ],
    )
    complete = sorted(selected_seeds) == sorted(registered)
    result = {
        "record_type": "h022_conditional_net_edge_training_result",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H022",
        "status": "complete" if complete else "partial",
        "valid": True,
        "completed_at": now_utc(),
        "config_sha256": sha256_file(config_path),
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "implementation_sha256": _implementation_digest(),
        "trained_seeds": selected_seeds,
        "registered_seeds": registered,
        "best_seed": int(best["seed"]),
        "best_seed_breadth_eligible": bool(best["selection_breadth_eligible"]),
        "minimum_calls": minimum_calls,
        "minimum_independent_components": minimum_components,
        "best_selection": best["metrics"]["selection"]["ensemble"],
        "seed_results": [
            {
                "seed": int(value["seed"]),
                "result_sha256": sha256_file(
                    output_dir / f"seed={value['seed']}" / "result.json"
                ),
                "selection": value["metrics"]["selection"]["ensemble"],
                "selection_breadth_eligible": bool(
                    value["selection_breadth_eligible"]
                ),
            }
            for value in results
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Development-only member selection; exact stateful replay is required "
            "before any promotion claim."
        ),
    }
    atomic_json(output_dir / "result.json", result)
    (output_dir / "progress.json").unlink(missing_ok=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--candidate-pack-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, action="append")
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.candidate_pack_dir.resolve(),
        args.output_dir.resolve(),
        args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "best_seed": result.get("best_seed"),
                "best_selection": result.get("best_selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
