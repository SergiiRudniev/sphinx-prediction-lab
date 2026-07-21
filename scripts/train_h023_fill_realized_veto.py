"""Train the H022-initialized H023 fill-realized neural/tree veto ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

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
from sphinx_trace.h022_training import fit_weighted_ridge, predict_weighted_ridge
from sphinx_trace.h023_training import h023_neural_loss, realized_policy_metrics
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h023 import (
    H023_AUX_FEATURE_WIDTH,
    SphinxTraceS0H023NeuralMember,
    load_h022_initialization,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h023_experiment_v1.json"
DEFAULT_REGISTRATION = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h023_fill_realized_veto_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h023_training.py",
    ROOT / "src" / "sphinx_trace" / "model_h023.py",
    ROOT / "src" / "sphinx_trace" / "h022_features.py",
    ROOT / "src" / "sphinx_trace" / "h022_training.py",
)
TERMINAL_LOGIT_INDEX = H022_TREE_FEATURE_NAMES.index("outcome.terminal_logit")
BREAK_EVEN_INDEX = H022_TREE_FEATURE_NAMES.index(
    "candidate.break_even_probability"
)
PREDICTOR_NAMES = (
    "neural_realized_contribution",
    "neural_conditional_return_mean",
    "neural_conditional_return_q10",
    "neural_conditional_return_q50",
    "neural_conditional_return_q90",
    "neural_fill_probability",
    "neural_positive_probability",
    "neural_keep_logit",
    "h022_neural_mean",
    "h022_tree",
    "h022_ensemble",
    "h023_tree_realized_contribution",
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


PACK_ARRAY_NAMES = (
    "tree_features",
    "market_latents",
    "h022_member_features",
    "candidate_action_ids",
    "component_ids",
    "market_ids",
    "week_ids",
    "timestamps",
    "feature_rows",
    "entry_prices",
    "target_realized_pnl_usd",
    "target_realized_net_contribution",
    "target_return_on_requested_cost",
    "target_return_on_filled_cost",
    "target_fill_fraction",
    "requested_total_cost_usd",
    "actual_filled_total_cost_usd",
    "terminal_payout_usd",
    "collateral_fee_usd",
    "outcome_token_fee_shares",
    "decision_ids",
)


def _load_partition(pack_dir: Path, partition: str) -> dict[str, NDArray[Any]]:
    values = {
        name: np.load(
            pack_dir / partition / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in PACK_ARRAY_NAMES
    }
    rows = len(values["target_realized_net_contribution"])
    if any(len(value) != rows for value in values.values()):
        raise RuntimeError(f"H023 {partition} arrays no longer align")
    if (
        values["tree_features"].shape != (rows, H022_TREE_FEATURE_WIDTH)
        or values["market_latents"].shape[0] != rows
        or values["h022_member_features"].shape
        != (rows, H023_AUX_FEATURE_WIDTH)
    ):
        raise RuntimeError(f"H023 {partition} feature width changed")
    return values


def _statistics(
    values: NDArray[Any], indices: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    selected = np.asarray(values[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, np.float32(1e-5))


def _sample_weights(
    data: dict[str, NDArray[Any]], config: dict[str, Any]
) -> NDArray[np.float32]:
    requested = np.asarray(data["requested_total_cost_usd"], dtype=np.float64)
    positive = requested[requested > 0.0]
    median = float(np.median(positive)) if len(positive) else 1.0
    weights = 1.0 + np.sqrt(np.maximum(requested, 0.0) / max(median, 1e-8))
    weights = np.minimum(weights, float(config["weighting"]["maximum_cost_weight"]))
    weeks = np.asarray(data["week_ids"], dtype=np.int64)
    pnl = np.asarray(data["target_realized_pnl_usd"], dtype=np.float64)
    _, inverse = np.unique(weeks, return_inverse=True)
    week_pnl = np.bincount(inverse, weights=pnl)
    downside = week_pnl[inverse] < 0.0
    weights *= np.where(
        downside,
        float(config["weighting"]["negative_week_multiplier"]),
        1.0,
    )
    weights /= max(float(weights.mean()), 1e-8)
    output = weights.astype(np.float32)
    if bool((output <= 0.0).any()) or not bool(np.isfinite(output).all()):
        raise RuntimeError("H023 sample weights are invalid")
    return output


def _neural_batch(
    data: dict[str, NDArray[Any]],
    indices: NDArray[np.int64],
    device: torch.device,
    statistics: dict[str, NDArray[np.float32]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    raw_latent = np.asarray(data["market_latents"][indices], dtype=np.float32)
    raw_tree = np.asarray(data["tree_features"][indices], dtype=np.float32)
    raw_aux = np.asarray(data["h022_member_features"][indices], dtype=np.float32)
    latent = torch.from_numpy(
        (raw_latent - statistics["latent_mean"]) / statistics["latent_scale"]
    ).to(device, non_blocking=True)
    tree = torch.from_numpy(
        (raw_tree - statistics["tree_mean"]) / statistics["tree_scale"]
    ).to(device, non_blocking=True)
    auxiliary = torch.from_numpy(
        (raw_aux - statistics["aux_mean"]) / statistics["aux_scale"]
    ).to(device, non_blocking=True)
    terminal = torch.from_numpy(raw_tree[:, TERMINAL_LOGIT_INDEX]).to(
        device, non_blocking=True
    )
    action = torch.from_numpy(
        np.asarray(data["candidate_action_ids"][indices], dtype=np.int64)
    ).to(device, non_blocking=True)
    break_even = torch.from_numpy(raw_tree[:, BREAK_EVEN_INDEX]).to(
        device, non_blocking=True
    )
    return latent, tree, terminal, action, break_even, auxiliary


def _target_batch(
    data: dict[str, NDArray[Any]],
    weights: NDArray[np.float32],
    indices: NDArray[np.int64],
    device: torch.device,
    config: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    contribution = np.asarray(
        data["target_realized_net_contribution"][indices], dtype=np.float32
    )
    conditional_return = np.asarray(
        data["target_return_on_filled_cost"][indices], dtype=np.float32
    )
    maximum_return = float(config["neural_member"]["maximum_absolute_return_target"])
    np.clip(conditional_return, -maximum_return, maximum_return, out=conditional_return)
    fill_fraction = np.asarray(data["target_fill_fraction"][indices], dtype=np.float32)
    requested_fraction = (
        np.asarray(data["requested_total_cost_usd"][indices], dtype=np.float32)
        / float(config["training"]["initial_cash_usd"])
    )
    return tuple(
        torch.from_numpy(value).to(device, non_blocking=True)
        for value in (
            contribution,
            conditional_return,
            fill_fraction,
            requested_fraction,
            np.asarray(weights[indices], dtype=np.float32),
        )
    )  # type: ignore[return-value]


def _h022_checkpoint(
    artifact_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    path = artifact_dir / "final" / "neural" / "best-neural.pt"
    dependency = config["dependencies"]["H022_initialization"]
    if sha256_file(path) != dependency["neural_sha256"]:
        raise RuntimeError("H023 H022 initialization receipt changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("record_type") != "h022_neural_member":
        raise RuntimeError("H023 H022 initialization type changed")
    return cast(dict[str, Any], checkpoint)


def _new_model(
    config: dict[str, Any], h022_checkpoint: dict[str, Any], device: torch.device
) -> tuple[SphinxTraceS0H023NeuralMember, tuple[str, ...]]:
    model = SphinxTraceS0H023NeuralMember(config["neural_member"])
    missing = load_h022_initialization(model, h022_checkpoint["model"])
    return model.to(device), missing


@torch.inference_mode()
def _predict_neural(
    model: SphinxTraceS0H023NeuralMember,
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
        "contribution": [],
        "return_mean": [],
        "return_quantiles": [],
        "fill": [],
        "positive": [],
        "keep_logit": [],
        "attention": [],
    }
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset : offset + batch_size]
        output = model(
            *_neural_batch(data, selected, device, statistics),
            return_debug=return_attention,
        )
        parts["contribution"].append(
            output["realized_net_contribution_mean"].float().cpu().numpy()
        )
        parts["return_mean"].append(
            output["conditional_realized_return_mean"].float().cpu().numpy()
        )
        parts["return_quantiles"].append(
            output["conditional_realized_return_quantiles"].float().cpu().numpy()
        )
        parts["fill"].append(output["fill_probability"].float().cpu().numpy())
        parts["positive"].append(
            output["probability_realized_contribution_positive"]
            .float()
            .cpu()
            .numpy()
        )
        parts["keep_logit"].append(
            output["keep_base_call_logit"].float().cpu().numpy()
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
        raise RuntimeError("H023 neural prediction rows changed")
    return result


def _fit_neural(
    data: dict[str, NDArray[Any]],
    weights: NDArray[np.float32],
    train_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64] | None,
    statistics_indices: NDArray[np.int64],
    config: dict[str, Any],
    h022_checkpoint: dict[str, Any],
    seed: int,
    output_dir: Path,
    pause_file: Path,
    *,
    fixed_epochs: int | None = None,
) -> tuple[
    SphinxTraceS0H023NeuralMember,
    dict[str, NDArray[np.float32]],
    dict[str, Any],
]:
    neural_config = config["neural_member"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    statistics = {
        "latent_mean": _statistics(data["market_latents"], statistics_indices)[0],
        "latent_scale": _statistics(data["market_latents"], statistics_indices)[1],
        "tree_mean": _statistics(data["tree_features"], statistics_indices)[0],
        "tree_scale": _statistics(data["tree_features"], statistics_indices)[1],
        "aux_mean": _statistics(data["h022_member_features"], statistics_indices)[0],
        "aux_scale": _statistics(data["h022_member_features"], statistics_indices)[1],
    }
    model, initialized_parameters = _new_model(config, h022_checkpoint, device)
    new_prefixes = (
        "aux_encoder.",
        "aux_gate",
        "realized_net_contribution.",
        "positive_contribution.",
        "keep_utility.",
    )
    pretrained_parameters: list[Tensor] = []
    new_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        target = (
            new_parameters
            if name.startswith(new_prefixes)
            else pretrained_parameters
        )
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": pretrained_parameters,
                "lr": float(neural_config["pretrained_learning_rate"]),
            },
            {
                "params": new_parameters,
                "lr": float(neural_config["learning_rate"]),
            },
        ],
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
        "h022_neural_sha256": config["dependencies"]["H022_initialization"][
            "neural_sha256"
        ],
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"H023 neural checkpoint contract changed: {output_dir}")
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
            targets = _target_batch(data, weights, selected, device, config)
            with autocast:
                output = model(*inputs)
                loss, metrics = h023_neural_loss(
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
            target_pnl = np.asarray(
                data["target_realized_pnl_usd"][validation_indices],
                dtype=np.float64,
            )
            keep = prediction["contribution"] > 0.0
            score = float(target_pnl[keep].sum())
            validation_receipt = {
                "realized_net_profit_usd": score,
                "baseline_keep_all_net_profit_usd": float(target_pnl.sum()),
                "calls": int(keep.sum()),
                "rows": len(validation_indices),
            }
        improved = score > best_score + 1e-9
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
                "pretrained_learning_rate": scheduler.get_last_lr()[0],
                "new_head_learning_rate": scheduler.get_last_lr()[1],
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
            raise PauseRequested(f"H023 paused after neural epoch {epoch}")
        if validation_indices is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("H023 neural training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    artifact_path = output_dir / "best-neural.pt"
    _atomic_torch_save(
        artifact_path,
        {
            "record_type": "h023_neural_member",
            "contract_sha256": contract_sha256,
            "model": best_state,
            "statistics": statistics,
            "config": neural_config,
            "best_epoch": best_epoch,
            "h022_initialized_parameters": list(initialized_parameters),
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
        "h022_initialized_parameter_tensors": len(initialized_parameters),
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
    *,
    feature_names: tuple[str, ...] = H022_TREE_FEATURE_NAMES,
) -> tuple[lgb.Booster, int]:
    train = lgb.Dataset(
        np.asarray(features[train_indices], dtype=np.float32),
        label=np.asarray(target[train_indices], dtype=np.float32),
        weight=np.asarray(weights[train_indices], dtype=np.float32),
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    validation = lgb.Dataset(
        np.asarray(features[validation_indices], dtype=np.float32),
        label=np.asarray(target[validation_indices], dtype=np.float32),
        weight=np.asarray(weights[validation_indices], dtype=np.float32),
        feature_name=list(feature_names),
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
    return booster, int(booster.best_iteration or booster.current_iteration())


def _fit_tree_fixed(
    features: NDArray[np.float32],
    target: NDArray[np.float32],
    weights: NDArray[np.float32],
    indices: NDArray[np.int64],
    rounds: int,
    config: dict[str, Any],
    seed: int,
    *,
    feature_names: tuple[str, ...] = H022_TREE_FEATURE_NAMES,
) -> lgb.Booster:
    dataset = lgb.Dataset(
        np.asarray(features[indices], dtype=np.float32),
        label=np.asarray(target[indices], dtype=np.float32),
        weight=np.asarray(weights[indices], dtype=np.float32),
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    return lgb.train(
        _tree_parameters(config, seed),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _predictor_matrix(
    neural: dict[str, NDArray[np.float32]],
    h023_tree: NDArray[np.float64],
    h022_aux: NDArray[Any],
) -> NDArray[np.float64]:
    quantiles = neural["return_quantiles"].astype(np.float64)
    auxiliary = np.asarray(h022_aux, dtype=np.float64)
    return np.column_stack(
        (
            neural["contribution"],
            neural["return_mean"],
            quantiles[:, 0],
            quantiles[:, 1],
            quantiles[:, 2],
            neural["fill"],
            neural["positive"],
            neural["keep_logit"],
            auxiliary[:, 0],
            auxiliary[:, 7],
            auxiliary[:, 8],
            h023_tree,
        )
    ).astype(np.float64, copy=False)


def _metrics(
    data: dict[str, NDArray[Any]], predicted: NDArray[np.float64]
) -> dict[str, Any]:
    return realized_policy_metrics(
        np.asarray(data["target_realized_pnl_usd"], dtype=np.float64),
        predicted,
        np.asarray(data["component_ids"], dtype=np.int64),
        np.asarray(data["week_ids"], dtype=np.int64),
        np.asarray(data["entry_prices"], dtype=np.float64),
        np.asarray(data["target_fill_fraction"], dtype=np.float64),
    )


def _ablation_features(
    features: NDArray[np.float32], mode: str, seed: int
) -> NDArray[np.float32]:
    if mode == "wallet_zero":
        return wallet_ablation(features, "zero", seed=seed)
    if mode == "wallet_shuffle":
        return wallet_ablation(features, "shuffle", seed=seed)
    output = np.array(features, copy=True)
    if mode == "price_execution_zero":
        output[:, 128:H022_TREE_FEATURE_WIDTH] = 0.0
        return output
    if mode == "event_component_zero":
        output[:, 48:72] = 0.0
        return output
    raise ValueError(f"Unknown H023 ablation: {mode}")


def _train_seed(
    config: dict[str, Any],
    candidate_manifest: dict[str, Any],
    fit: dict[str, NDArray[Any]],
    selection: dict[str, NDArray[Any]],
    h022_checkpoint: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_dir = output_dir / f"seed={seed}"
    result_path = seed_dir / "result.json"
    if result_path.is_file():
        payload = load_json(result_path)
        if payload.get("valid") is True:
            return payload
    seed_dir.mkdir(parents=True, exist_ok=True)
    pause_file = output_dir / str(config["resume"]["graceful_pause_file"])
    folds = int(config["cross_fitting"]["folds"])
    component_ids = np.asarray(fit["component_ids"], dtype=np.int64)
    fold_codes = component_folds(component_ids, folds, seed)
    all_fit = np.arange(len(component_ids), dtype=np.int64)
    target = np.asarray(fit["target_realized_net_contribution"], dtype=np.float32)
    weights = _sample_weights(fit, config)
    fit_features = np.asarray(fit["tree_features"], dtype=np.float32)
    selection_features = np.asarray(selection["tree_features"], dtype=np.float32)
    oof_neural: dict[str, NDArray[np.float32]] = {
        "contribution": np.full(len(all_fit), np.nan, dtype=np.float32),
        "return_mean": np.full(len(all_fit), np.nan, dtype=np.float32),
        "return_quantiles": np.full((len(all_fit), 3), np.nan, dtype=np.float32),
        "fill": np.full(len(all_fit), np.nan, dtype=np.float32),
        "positive": np.full(len(all_fit), np.nan, dtype=np.float32),
        "keep_logit": np.full(len(all_fit), np.nan, dtype=np.float32),
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
            raise RuntimeError(f"H023 seed {seed} fold {fold} is empty")
        model, statistics, neural_receipt = _fit_neural(
            fit,
            weights,
            train_indices,
            inner_validation,
            pool,
            config,
            h022_checkpoint,
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
                "record_type": "h023_seed_progress",
                "seed": seed,
                "cross_fit_folds_complete": fold + 1,
                "cross_fit_folds_total": folds,
            },
        )
        if pause_file.is_file():
            raise PauseRequested(f"H023 paused after seed {seed} fold {fold}")
    if any(bool(np.isnan(value).any()) for value in oof_neural.values()) or bool(
        np.isnan(oof_tree).any()
    ):
        raise RuntimeError(f"H023 seed {seed} OOF predictions are incomplete")
    oof_predictors = _predictor_matrix(
        oof_neural, oof_tree, fit["h022_member_features"]
    )
    pnl = np.asarray(fit["target_realized_pnl_usd"], dtype=np.float64)
    decision_weights = weights.astype(np.float64) * np.where(
        pnl < 0.0,
        float(config["stacking"]["harmful_keep_weight"]),
        float(config["stacking"]["profitable_veto_weight"]),
    )
    stacker = fit_weighted_ridge(
        oof_predictors,
        target.astype(np.float64),
        decision_weights,
        float(config["stacking"]["ridge"]),
    )
    stacker["predictor_names"] = list(PREDICTOR_NAMES)
    stacker["decision"] = "KEEP_when_expected_realized_net_contribution_gt_zero"
    oof_ensemble = predict_weighted_ridge(oof_predictors, stacker)
    final_epochs = max(1, round(float(np.median(neural_epochs))))
    final_rounds = max(1, round(float(np.median(tree_rounds))))
    final_model, final_statistics, final_neural_receipt = _fit_neural(
        fit,
        weights,
        all_fit,
        None,
        all_fit,
        config,
        h022_checkpoint,
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
        len(selection["target_realized_net_contribution"]), dtype=np.int64
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
    selection_ensemble = predict_weighted_ridge(
        _predictor_matrix(
            selection_neural,
            selection_tree,
            selection["h022_member_features"],
        ),
        stacker,
    )
    baseline_fit = np.full(len(all_fit), np.finfo(np.float64).eps)
    baseline_selection = np.full(len(selection_indices), np.finfo(np.float64).eps)
    metrics = {
        "oof": {
            "H021_keep_all": _metrics(fit, baseline_fit),
            "neural": _metrics(
                fit, oof_neural["contribution"].astype(np.float64)
            ),
            "tree": _metrics(fit, oof_tree),
            "ensemble": _metrics(fit, oof_ensemble),
        },
        "selection": {
            "H021_keep_all": _metrics(selection, baseline_selection),
            "neural": _metrics(
                selection, selection_neural["contribution"].astype(np.float64)
            ),
            "tree": _metrics(selection, selection_tree),
            "ensemble": _metrics(selection, selection_ensemble),
        },
    }
    ablations: dict[str, Any] = {}
    for mode in (
        "wallet_zero",
        "wallet_shuffle",
        "price_execution_zero",
        "event_component_zero",
    ):
        ablated_fit = _ablation_features(fit_features, mode, seed ^ 0xA811)
        ablated_selection = _ablation_features(
            selection_features, mode, seed ^ 0xA812
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
            ablated_tree.predict(ablated_selection, num_iteration=final_rounds),
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
            _predictor_matrix(
                ablated_neural,
                ablated_tree_prediction,
                selection["h022_member_features"],
            ),
            stacker,
        )
        ablated_tree_path = seed_dir / "ablations" / f"{mode}-tree.txt"
        ablated_tree_path.parent.mkdir(parents=True, exist_ok=True)
        ablated_tree.save_model(ablated_tree_path, num_iteration=final_rounds)
        ablations[mode] = {
            "ensemble": _metrics(selection, ablated_ensemble),
            "tree_sha256": sha256_file(ablated_tree_path),
        }
    aux_names = tuple(candidate_manifest["h022_member_feature_names"])
    score_tree, score_rounds = _fit_tree_with_validation(
        np.asarray(fit["h022_member_features"], dtype=np.float32),
        target,
        weights,
        all_fit[fold_codes != 0],
        all_fit[fold_codes == 0],
        config,
        seed ^ 0xC022,
        feature_names=aux_names,
    )
    score_prediction = np.asarray(
        score_tree.predict(
            np.asarray(selection["h022_member_features"], dtype=np.float32),
            num_iteration=score_rounds,
        ),
        dtype=np.float64,
    )
    score_path = seed_dir / "ablations" / "h022-score-only-tree.txt"
    score_tree.save_model(score_path, num_iteration=score_rounds)
    ablations["H022_score_only"] = {
        "metrics": _metrics(selection, score_prediction),
        "tree_sha256": sha256_file(score_path),
        "rounds": score_rounds,
    }
    importance_gain = final_tree.feature_importance(
        importance_type="gain", iteration=final_rounds
    )
    importance_split = final_tree.feature_importance(
        importance_type="split", iteration=final_rounds
    )
    importance: list[dict[str, Any]] = sorted(
        (
            {"feature": name, "gain": float(gain), "split": int(split)}
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
    debug_path = seed_dir / "selection-debug.npz"
    np.savez_compressed(
        debug_path,
        target_realized_pnl_usd=np.asarray(
            selection["target_realized_pnl_usd"], dtype=np.float64
        ),
        entry_prices=np.asarray(selection["entry_prices"], dtype=np.float32),
        component_ids=np.asarray(selection["component_ids"], dtype=np.int64),
        decision_ids=np.asarray(selection["decision_ids"]),
        neural_contribution=selection_neural["contribution"],
        neural_return_mean=selection_neural["return_mean"],
        neural_return_quantiles=selection_neural["return_quantiles"],
        neural_fill_probability=selection_neural["fill"],
        neural_positive_probability=selection_neural["positive"],
        neural_keep_logit=selection_neural["keep_logit"],
        neural_group_attention=selection_neural["attention"],
        tree_contribution=selection_tree,
        ensemble_contribution=selection_ensemble,
        ensemble_keep=(selection_ensemble > 0.0).astype(np.uint8),
    )
    result = {
        "record_type": "h023_fill_realized_veto_seed_result",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H023",
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
        "selection_debug_sha256": sha256_file(debug_path),
        "stacker": stacker,
        "metrics": metrics,
        "ablations": ablations,
        "tree_feature_importance": importance,
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": int(
            candidate_manifest["calibration_rows_consumed"]
        ),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Component-cross-fitted calibration evidence only; exact validation "
            "replay, untouched test and paper-forward evidence remain closed."
        ),
    }
    atomic_json(result_path, result)
    (seed_dir / "progress.json").unlink(missing_ok=True)
    return result


def _merge_partitions(
    fit: dict[str, NDArray[Any]], selection: dict[str, NDArray[Any]]
) -> dict[str, NDArray[Any]]:
    if set(fit) != set(selection):
        raise RuntimeError("H023 calibration partitions changed")
    return {
        name: np.concatenate(
            (np.asarray(fit[name]), np.asarray(selection[name])), axis=0
        )
        for name in fit
    }


def _train_selected_final(
    config: dict[str, Any],
    candidate_manifest: dict[str, Any],
    all_calibration: dict[str, NDArray[Any]],
    h022_checkpoint: dict[str, Any],
    output_dir: Path,
    selected: dict[str, Any],
) -> dict[str, Any]:
    final_dir = output_dir / "selected-final"
    result_path = final_dir / "result.json"
    if result_path.is_file():
        payload = load_json(result_path)
        if payload.get("valid") is True:
            return payload
    seed = int(selected["seed"])
    indices = np.arange(
        len(all_calibration["target_realized_net_contribution"]), dtype=np.int64
    )
    weights = _sample_weights(all_calibration, config)
    pause_file = output_dir / str(config["resume"]["graceful_pause_file"])
    epochs = int(selected["final_neural_epochs"])
    rounds = int(selected["final_tree_rounds"])
    model, statistics, neural_receipt = _fit_neural(
        all_calibration,
        weights,
        indices,
        None,
        indices,
        config,
        h022_checkpoint,
        seed ^ 0x23F1,
        final_dir / "neural",
        pause_file,
        fixed_epochs=epochs,
    )
    del model, statistics
    tree = _fit_tree_fixed(
        np.asarray(all_calibration["tree_features"], dtype=np.float32),
        np.asarray(
            all_calibration["target_realized_net_contribution"], dtype=np.float32
        ),
        weights,
        indices,
        rounds,
        config,
        seed ^ 0x23F2,
    )
    tree_path = final_dir / "tree.txt"
    tree.save_model(tree_path, num_iteration=rounds)
    source_stacker = output_dir / f"seed={seed}" / "final" / "stacker.json"
    stacker = load_json(source_stacker)
    stacker_path = final_dir / "stacker.json"
    atomic_json(stacker_path, stacker)
    analog_path = final_dir / "realized-training-analogs.npz"
    np.savez_compressed(
        analog_path,
        tree_features=np.asarray(
            all_calibration["tree_features"], dtype=np.float16
        ),
        h022_member_features=np.asarray(
            all_calibration["h022_member_features"], dtype=np.float16
        ),
        target_realized_pnl_usd=np.asarray(
            all_calibration["target_realized_pnl_usd"], dtype=np.float64
        ),
        target_fill_fraction=np.asarray(
            all_calibration["target_fill_fraction"], dtype=np.float32
        ),
        entry_prices=np.asarray(all_calibration["entry_prices"], dtype=np.float32),
        component_ids=np.asarray(
            all_calibration["component_ids"], dtype=np.int64
        ),
        decision_ids=np.asarray(all_calibration["decision_ids"]),
    )
    result = {
        "record_type": "h023_selected_full_calibration_runtime",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H023",
        "valid": True,
        "completed_at": now_utc(),
        "selected_seed": seed,
        "rows": len(indices),
        "components": int(
            np.unique(
                np.asarray(all_calibration["component_ids"], dtype=np.int64)
            ).size
        ),
        "neural_epochs": epochs,
        "tree_rounds": rounds,
        "neural": neural_receipt,
        "neural_sha256": sha256_file(final_dir / "neural" / "best-neural.pt"),
        "tree_sha256": sha256_file(tree_path),
        "stacker_sha256": sha256_file(stacker_path),
        "realized_training_analogs_sha256": sha256_file(analog_path),
        "source_seed_result_sha256": sha256_file(
            output_dir / f"seed={seed}" / "result.json"
        ),
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "implementation_sha256": _implementation_digest(),
        "calibration_rows_consumed": int(
            candidate_manifest["calibration_rows_consumed"]
        ),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "validation_labels_opened_for_H023": False,
        "promotion_allowed": False,
        "evidence_boundary": (
            "Selected on grouped calibration development evidence and refit on all "
            "calibration candidates; exact validation replay remains required."
        ),
    }
    atomic_json(result_path, result)
    return result


def train(
    config_path: Path,
    registration_path: Path,
    candidate_pack_dir: Path,
    h022_artifact_dir: Path,
    output_dir: Path,
    seeds: list[int] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    registration = load_json(registration_path)
    candidate_manifest_path = candidate_pack_dir / "manifest.json"
    candidate_manifest = load_json(candidate_manifest_path)
    candidate_manifest["manifest_sha256"] = sha256_file(candidate_manifest_path)
    dependency = config.get("dependencies", {})
    if (
        config.get("research_id") != "SPH-T-H023"
        or registration.get("research_id") != "SPH-T-H023"
        or sha256_file(registration_path) != dependency.get("registration_sha256")
        or candidate_manifest.get("valid") is not True
        or candidate_manifest.get("config_sha256") != sha256_file(config_path)
        or candidate_manifest.get("test_labels_opened") is not False
        or int(candidate_manifest.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H023 training source contract changed")
    registered = [int(value) for value in config["cross_fitting"]["seeds"]]
    selected_seeds = registered if seeds is None else seeds
    if not selected_seeds or any(seed not in registered for seed in selected_seeds):
        raise RuntimeError("H023 requested seed was not registered")
    fit = _load_partition(candidate_pack_dir, "fit")
    selection = _load_partition(candidate_pack_dir, "selection")
    h022_checkpoint = _h022_checkpoint(h022_artifact_dir, config)
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
                    h022_checkpoint,
                    output_dir,
                    seed,
                )
            )
        best = max(
            results,
            key=lambda value: value["metrics"]["selection"]["ensemble"][
                "realized_net_profit_usd"
            ],
        )
        selected_final = _train_selected_final(
            config,
            candidate_manifest,
            _merge_partitions(fit, selection),
            h022_checkpoint,
            output_dir,
            best,
        )
    except PauseRequested as exc:
        progress = {
            "record_type": "h023_training_progress",
            "status": "paused",
            "message": str(exc),
            "completed_seeds": [int(value["seed"]) for value in results],
            "requested_seeds": selected_seeds,
        }
        atomic_json(output_dir / "progress.json", progress)
        return progress
    complete = sorted(selected_seeds) == sorted(registered)
    result = {
        "record_type": "h023_fill_realized_veto_training_result",
        "schema_version": "1.0.0",
        "research_id": "SPH-T-H023",
        "status": "complete" if complete else "partial",
        "valid": True,
        "completed_at": now_utc(),
        "config_sha256": sha256_file(config_path),
        "registration_sha256": sha256_file(registration_path),
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "implementation_sha256": _implementation_digest(),
        "trained_seeds": selected_seeds,
        "registered_seeds": registered,
        "selected_seed": int(best["seed"]),
        "selected_seed_selection": best["metrics"]["selection"]["ensemble"],
        "selected_final_result_sha256": sha256_file(
            output_dir / "selected-final" / "result.json"
        ),
        "selected_final_neural_sha256": selected_final["neural_sha256"],
        "selected_final_tree_sha256": selected_final["tree_sha256"],
        "selected_final_stacker_sha256": selected_final["stacker_sha256"],
        "selected_final_realized_training_analogs_sha256": selected_final[
            "realized_training_analogs_sha256"
        ],
        "seed_results": [
            {
                "seed": int(value["seed"]),
                "result_sha256": sha256_file(
                    output_dir / f"seed={value['seed']}" / "result.json"
                ),
                "selection": value["metrics"]["selection"]["ensemble"],
            }
            for value in results
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows_consumed": int(
            candidate_manifest["calibration_rows_consumed"]
        ),
        "validation_rows_consumed_for_H023": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "promotion_allowed": False,
        "evidence_boundary": (
            "Calibration-only training and selection; exact registered validation "
            "replay is required before any H023 promotion claim."
        ),
    }
    atomic_json(output_dir / "result.json", result)
    (output_dir / "progress.json").unlink(missing_ok=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    value.add_argument("--candidate-pack-dir", type=Path, required=True)
    value.add_argument("--h022-artifact-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, action="append")
    return value


def main() -> None:
    args = parser().parse_args()
    result = train(
        args.config.resolve(),
        args.registration.resolve(),
        args.candidate_pack_dir.resolve(),
        args.h022_artifact_dir.resolve(),
        args.output_dir.resolve(),
        args.seed,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "selected_seed": result.get("selected_seed"),
                "selected_seed_selection": result.get("selected_seed_selection"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
