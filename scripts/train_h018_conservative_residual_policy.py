"""Train H018 as a conservative protocol-exact residual over frozen H014."""

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

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model import parameter_count
from sphinx_trace.model_h012 import SphinxTraceS0H012
from sphinx_trace.on_policy_pack import validate_on_policy_shard
from sphinx_trace.policy_checkpoint import load_policy_checkpoint
from sphinx_trace.protocol_residual_training import (
    conservative_protocol_residual_loss,
)
from sphinx_trace.protocol_tail_pack import validate_protocol_tail_shard
from sphinx_trace.protocol_veto_training import learned_call_loss_veto_loss

try:
    from scripts.train_h015_portfolio_advantage import (
        StateBatch,
        StateShard,
        _atomic_torch_save,
        _batch,
        _equal_market_weights,
        _forward,
        _indices,
        _load_object,
        _module_digest,
        _sample_weights,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from train_h015_portfolio_advantage import (  # type: ignore[import-not-found,no-redef]
        StateBatch,
        StateShard,
        _atomic_torch_save,
        _batch,
        _equal_market_weights,
        _forward,
        _indices,
        _load_object,
        _module_digest,
        _sample_weights,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h018_conservative_residual_policy_v1.json"
)
DEFAULT_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h018_policy_v1.json"
)
DEFAULT_INITIAL_POLICY_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h012_selective_policy_v1.json"
)
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_model_v1.json"
DEFAULT_RESIDUAL_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h013_market_residual_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "train_h015_portfolio_advantage.py",
    ROOT / "src" / "sphinx_trace" / "protocol_residual_training.py",
    ROOT / "src" / "sphinx_trace" / "protocol_veto_training.py",
    ROOT / "scripts" / "train_h019_loss_veto_policy.py",
    ROOT / "src" / "sphinx_trace" / "protocol_tail_pack.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _state_shards(
    state_dir: Path,
    encoding_dir: Path,
    pack_dir: Path,
    initial_policy_dir: Path,
    config: dict[str, Any],
) -> tuple[list[StateShard], dict[str, Any]]:
    manifest_path = state_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    initial_result_path = initial_policy_dir / "result.json"
    initial_result = _load_object(initial_result_path)
    dependency = config["dependencies"]["protocol_tail_pack"]
    if (
        manifest.get("record_type") != "h017_protocol_tail_pack_manifest"
        or manifest.get("valid") is not True
        or manifest.get("test_labels_opened") is not False
        or int(manifest.get("test_rows_consumed", -1)) != 0
        or int(manifest.get("calibration_rows_consumed", -1)) != 0
        or sha256_file(manifest_path) != dependency["manifest_sha256"]
        or manifest.get("pack_manifest_sha256")
        != sha256_file(pack_dir / "manifest.json")
        or manifest.get("encoding_manifest_sha256")
        != sha256_file(encoding_dir / "manifest.json")
        or config["dependencies"]["initial_policy"]["result_sha256"]
        != sha256_file(initial_result_path)
        or config["dependencies"]["initial_policy"]["best_model_sha256"]
        != sha256_file(initial_policy_dir / "best-policy.pt")
        or initial_result.get("test_labels_opened") is not False
    ):
        raise RuntimeError("H018 source contract changed")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise RuntimeError("H018 state manifest has no shards")
    shards: list[StateShard] = []
    behavior_rows: dict[int, int] = {}
    total_rows = 0
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise RuntimeError("H018 state shard metadata is invalid")
        behavior_id = str(raw.get("behavior_policy_id") or "")
        behavior_code = int(raw.get("behavior_policy_code", -1))
        date = str(raw.get("date") or "")
        rows = int(raw.get("rows", -1))
        state = state_dir / "shards" / f"behavior={behavior_id}" / f"date={date}"
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
            raise RuntimeError(f"H018 state binding changed: {behavior_id}:{date}")
        receipt = _load_object(receipt_path)
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise RuntimeError(f"H018 state receipt is invalid: {behavior_id}:{date}")
        validate_on_policy_shard(
            state,
            files,
            expected_rows=rows,
            expected_behavior_code=behavior_code,
        )
        validate_protocol_tail_shard(state, files, expected_rows=rows)
        shards.append(
            StateShard(behavior_id, behavior_code, date, state, encoding, pack, rows)
        )
        total_rows += rows
        behavior_rows[behavior_code] = behavior_rows.get(behavior_code, 0) + rows
    expected_rows = int(dependency["rows"])
    expected_per_behavior = expected_rows // 2
    if (
        total_rows != expected_rows
        or sorted(behavior_rows) != [0, 1]
        or any(rows != expected_per_behavior for rows in behavior_rows.values())
        or str(manifest.get("partition_sha256")) != dependency["partition_sha256"]
    ):
        raise RuntimeError("H018 state coverage changed")
    return shards, manifest


def _initialize_model(
    initial_policy_dir: Path,
    outcome_dir: Path,
    model_config: dict[str, Any],
    residual_config: dict[str, Any],
    initial_policy_config: dict[str, Any],
    target_policy_config: dict[str, Any],
    device: torch.device,
) -> tuple[SphinxTraceS0H012, dict[str, Any], dict[str, Any]]:
    loaded = load_policy_checkpoint(
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        initial_policy_config,
        device,
    )
    source_digest = _module_digest(loaded.model)
    model = SphinxTraceS0H012(loaded.model.outcome_backbone, target_policy_config).to(device)
    missing, unexpected = model.load_state_dict(loaded.model.state_dict(), strict=False)
    allowed_prefixes = ("action_residual.", "protocol_action_value.")
    if unexpected or not missing or any(
        not name.startswith(allowed_prefixes) for name in missing
    ):
        raise RuntimeError(
            f"H018 initialization mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.requires_grad_(False)
    if model.action_residual is None or model.protocol_action_value is None:
        raise RuntimeError("H018 policy configuration did not construct both adapter heads")
    model.action_residual.requires_grad_(True)
    model.protocol_action_value.requires_grad_(True)
    receipt = {
        "source_model_tensor_sha256": source_digest,
        "missing_target_parameters": sorted(missing),
        "unexpected_source_parameters": sorted(unexpected),
        "trainable_modules": ["action_residual", "protocol_action_value"],
    }
    return model, loaded.result, receipt


def _verify_initial_equivalence(
    model: SphinxTraceS0H012,
    initial_policy_dir: Path,
    outcome_dir: Path,
    model_config: dict[str, Any],
    residual_config: dict[str, Any],
    initial_policy_config: dict[str, Any],
    shard: StateShard,
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_policy_checkpoint(
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        initial_policy_config,
        device,
    )
    indices = _indices(shard, 1, seed=0, epoch=0, shuffle=False)
    if not len(indices):
        indices = _indices(shard, 0, seed=0, epoch=0, shuffle=False)
    batch = _batch(shard, indices[: min(512, len(indices))])
    model.eval()
    loaded.model.eval()
    with torch.inference_mode():
        target, _, _ = _forward(model, batch, device)
        source, _, _ = _forward(loaded.model, batch, device)
    keys = (
        "action_logits",
        "position_size_beta_alpha",
        "position_size_beta_beta",
        "state_value",
    )
    equal = {key: bool(torch.equal(target[key], source[key])) for key in keys}
    if not all(equal.values()):
        raise RuntimeError(f"H018 zero residual does not reproduce H014: {equal}")
    return {"rows": len(batch.labels), "tensor_equal": equal, "passed": True}


def _fit_week_downside_weights(
    shards: list[StateShard],
    market_weights: NDArray[np.float32],
    maximum_multiplier: float,
) -> tuple[dict[tuple[int, int], float], NDArray[np.float64], dict[str, Any]]:
    if maximum_multiplier < 1.0:
        raise ValueError("H018 maximum downside multiplier must be at least one")
    weekly_pnl: dict[tuple[int, int], float] = {}
    for shard in shards:
        indices = _indices(shard, 0, seed=0, epoch=0, shuffle=False)
        if not len(indices):
            continue
        weeks = np.load(shard.state / "week_ids.npy", mmap_mode="r", allow_pickle=False)
        pnl = np.load(
            shard.state / "realized_pnl_usd.npy", mmap_mode="r", allow_pickle=False
        )
        for week in np.unique(weeks[indices]):
            selected = indices[weeks[indices] == week]
            key = (shard.behavior_code, int(week))
            weekly_pnl[key] = weekly_pnl.get(key, 0.0) + float(pnl[selected].sum())
    factors: dict[tuple[int, int], float] = {}
    receipt_by_behavior: dict[str, Any] = {}
    for behavior in (0, 1):
        rows = {week: pnl for (code, week), pnl in weekly_pnl.items() if code == behavior}
        maximum_downside = max((-value for value in rows.values() if value < 0.0), default=0.0)
        for week, pnl in rows.items():
            fraction = (-pnl / maximum_downside) if pnl < 0.0 and maximum_downside else 0.0
            factors[(behavior, week)] = 1.0 + (maximum_multiplier - 1.0) * fraction
        receipt_by_behavior[str(behavior)] = {
            "weeks": len(rows),
            "losing_weeks": sum(value < 0.0 for value in rows.values()),
            "minimum_week_pnl_usd": min(rows.values(), default=0.0),
            "maximum_raw_factor": max(
                (factors[(behavior, week)] for week in rows), default=1.0
            ),
        }
    base_totals = np.zeros(2, dtype=np.float64)
    adjusted_totals = np.zeros(2, dtype=np.float64)
    for shard in shards:
        indices = _indices(shard, 0, seed=0, epoch=0, shuffle=False)
        if not len(indices):
            continue
        markets = np.load(
            shard.state / "market_ids.npy", mmap_mode="r", allow_pickle=False
        )
        weeks = np.load(shard.state / "week_ids.npy", mmap_mode="r", allow_pickle=False)
        base = market_weights[0, shard.behavior_code, markets[indices]].astype(np.float64)
        risk = np.asarray(
            [factors[(shard.behavior_code, int(week))] for week in weeks[indices]],
            dtype=np.float64,
        )
        base_totals[shard.behavior_code] += float(base.sum())
        adjusted_totals[shard.behavior_code] += float((base * risk).sum())
    normalization = base_totals / np.maximum(adjusted_totals, 1e-12)
    for behavior in (0, 1):
        receipt_by_behavior[str(behavior)]["normalization"] = float(
            normalization[behavior]
        )
    return factors, normalization, {
        "source_partition": "fit_only",
        "maximum_multiplier": maximum_multiplier,
        "behaviors": receipt_by_behavior,
    }


def _training_weights(
    batch: StateBatch,
    market_weights: NDArray[np.float32],
    downside_factors: dict[tuple[int, int], float],
    downside_normalization: NDArray[np.float64],
) -> NDArray[np.float32]:
    if batch.week_ids is None:
        raise RuntimeError("H018 batch has no week IDs")
    base = _sample_weights(batch, market_weights, 0).astype(np.float64)
    factors = np.asarray(
        [
            downside_factors[(int(behavior), int(week))]
            for behavior, week in zip(
                batch.behavior_policy_codes, batch.week_ids, strict=True
            )
        ],
        dtype=np.float64,
    )
    normalizers = downside_normalization[batch.behavior_policy_codes]
    values = base * factors * normalizers
    if not bool(np.isfinite(values).all()) or bool((values <= 0.0).any()):
        raise RuntimeError("H018 downside weights are invalid")
    return values.astype(np.float32)


def _loss(
    output: dict[str, Tensor],
    labels: Tensor,
    batch: StateBatch,
    weights: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    if batch.winning_payout_multipliers is None or batch.reference_action_values is None:
        raise RuntimeError("H018 batch has no protocol-exact targets")
    payout = torch.from_numpy(batch.winning_payout_multipliers).to(
        labels.device, non_blocking=True
    )
    reference = torch.from_numpy(batch.reference_action_values).to(
        labels.device, non_blocking=True
    )
    behavior_actions = torch.from_numpy(batch.behavior_action_ids).to(
        labels.device, non_blocking=True
    )
    realized = torch.from_numpy(batch.realized_action_values).to(
        labels.device, non_blocking=True
    )
    physical = torch.from_numpy(batch.physical_action_masks).to(
        labels.device, non_blocking=True
    )
    if config.get("loss_mode") == "learned_H014_call_loss_veto":
        return learned_call_loss_veto_loss(
            output,
            labels,
            payout,
            reference,
            behavior_actions,
            realized,
            config,
            sample_weights=weights,
            physical_action_mask=physical,
        )
    return conservative_protocol_residual_loss(
        output,
        labels,
        payout,
        reference,
        behavior_actions,
        realized,
        torch.from_numpy(batch.component_ids).to(labels.device, non_blocking=True),
        config,
        sample_weights=weights,
        physical_action_mask=physical,
    )


def _group_tail(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    behavior_codes: NDArray[np.int64],
    group_ids: NDArray[np.int64],
    quantile: float,
) -> tuple[float, int]:
    keys = np.stack((behavior_codes, group_ids), axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    weighted = np.bincount(inverse, weights=values * weights)
    totals = np.bincount(inverse, weights=weights)
    means = weighted / np.maximum(totals, 1e-12)
    count = max(1, math.ceil(len(means) * quantile))
    tail = np.partition(means, count - 1)[:count]
    return float(tail.mean()), len(means)


@torch.inference_mode()
def _evaluate(
    model: SphinxTraceS0H012,
    shards: list[StateShard],
    partition_code: int,
    training_config: dict[str, Any],
    market_weights: NDArray[np.float32],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    chosen_parts: list[NDArray[np.float64]] = []
    expected_parts: list[NDArray[np.float64]] = []
    weight_parts: list[NDArray[np.float64]] = []
    component_parts: list[NDArray[np.int64]] = []
    week_parts: list[NDArray[np.int64]] = []
    behavior_parts: list[NDArray[np.int64]] = []
    total_loss = 0.0
    total_weight = 0.0
    total_rows = 0
    total_size = 0.0
    total_calls = 0
    total_correct = 0
    weighted_calls = 0.0
    weighted_correct = 0.0
    base_calls_total = 0
    correct_base_calls_total = 0
    wrong_base_calls_total = 0
    retained_correct_base_calls_total = 0
    vetoed_wrong_base_calls_total = 0
    action_counts = np.zeros(3, dtype=np.int64)
    value_error = 0.0
    residual_l2 = 0.0
    policy_kl = 0.0
    for shard in shards:
        indices = _indices(shard, partition_code, seed=0, epoch=0, shuffle=False)
        for offset in range(0, len(indices), batch_size):
            batch = _batch(shard, indices[offset : offset + batch_size])
            if batch.winning_payout_multipliers is None or batch.week_ids is None:
                raise RuntimeError("H018 evaluation batch is incomplete")
            weights_numpy = _sample_weights(batch, market_weights, partition_code)
            weights = torch.from_numpy(weights_numpy).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, labels, _ = _forward(model, batch, device)
                loss, metrics = _loss(
                    output, labels, batch, weights, training_config
                )
            logits = output["action_logits"][:, :3].float()
            temperature = float(training_config["policy_temperature"])
            policy = torch.softmax(logits / temperature, dim=-1)
            alpha = output["position_size_beta_alpha"].float()
            beta = output["position_size_beta_beta"].float()
            size = alpha / (alpha + beta).clamp_min(1e-8)
            payouts = torch.from_numpy(batch.winning_payout_multipliers).to(
                device, non_blocking=True
            )
            label0 = labels.float()
            utility0 = torch.log(
                (1.0 - size + size * label0 * payouts[:, 0]).clamp_min(1e-8)
            )
            utility1 = torch.log(
                (
                    1.0
                    - size
                    + size * (1.0 - label0) * payouts[:, 1]
                ).clamp_min(1e-8)
            )
            utilities = torch.stack((utility0, utility1, torch.zeros_like(utility0)), dim=1)
            chosen = logits.argmax(dim=-1)
            base_chosen = output["base_action_logits"][:, :3].argmax(dim=-1)
            chosen_utility = utilities.gather(1, chosen[:, None]).squeeze(1)
            expected_utility = (policy * utilities).sum(dim=-1)
            calls = chosen != 2
            correct = ((chosen == 0) & (label0 == 1.0)) | (
                (chosen == 1) & (label0 == 0.0)
            )
            base_calls = base_chosen != 2
            base_correct = ((base_chosen == 0) & (label0 == 1.0)) | (
                (base_chosen == 1) & (label0 == 0.0)
            )
            correct_base_calls = base_calls & base_correct
            wrong_base_calls = base_calls & ~base_correct
            weight_sum = float(weights.sum())
            chosen_parts.append(chosen_utility.cpu().numpy().astype(np.float64))
            expected_parts.append(expected_utility.cpu().numpy().astype(np.float64))
            weight_parts.append(weights_numpy.astype(np.float64))
            component_parts.append(batch.component_ids.astype(np.int64))
            week_parts.append(batch.week_ids.astype(np.int64))
            behavior_parts.append(batch.behavior_policy_codes.astype(np.int64))
            total_loss += float(loss) * weight_sum
            total_weight += weight_sum
            total_rows += len(batch.labels)
            total_size += float((size * weights).sum())
            total_calls += int(calls.sum())
            total_correct += int(correct[calls].sum())
            weighted_calls += float(weights[calls].sum())
            weighted_correct += float(weights[calls & correct].sum())
            base_calls_total += int(base_calls.sum())
            correct_base_calls_total += int(correct_base_calls.sum())
            wrong_base_calls_total += int(wrong_base_calls.sum())
            retained_correct_base_calls_total += int(
                (correct_base_calls & (chosen == base_chosen)).sum()
            )
            vetoed_wrong_base_calls_total += int(
                (wrong_base_calls & (chosen == 2)).sum()
            )
            action_counts += np.bincount(chosen.cpu().numpy(), minlength=3)
            value_error += float(metrics["protocol_action_value_loss"]) * weight_sum
            residual_l2 += float(metrics["residual_logit_L2"]) * weight_sum
            policy_kl += float(metrics["H014_policy_KL"]) * weight_sum
    if not total_rows or total_weight <= 0.0:
        raise RuntimeError(f"H018 evaluation partition {partition_code} is empty")
    chosen_values = np.concatenate(chosen_parts)
    expected_values = np.concatenate(expected_parts)
    weights_all = np.concatenate(weight_parts)
    components = np.concatenate(component_parts)
    weeks = np.concatenate(week_parts)
    behaviors = np.concatenate(behavior_parts)
    row_count = max(
        1,
        math.ceil(
            len(chosen_values)
            * float(training_config["row_lower_tail_quantile"])
        ),
    )
    row_tail = float(np.partition(chosen_values, row_count - 1)[:row_count].mean())
    component_tail, component_count = _group_tail(
        chosen_values,
        weights_all,
        behaviors,
        components,
        float(training_config["component_lower_tail_quantile"]),
    )
    week_tail, week_count = _group_tail(
        chosen_values,
        weights_all,
        behaviors,
        weeks,
        float(training_config["component_lower_tail_quantile"]),
    )
    return {
        "rows": total_rows,
        "sample_weight_sum": total_weight,
        "loss": total_loss / total_weight,
        "equal_market_mean_protocol_exact_chosen_utility": float(
            np.sum(chosen_values * weights_all) / np.sum(weights_all)
        ),
        "equal_market_mean_protocol_exact_expected_utility": float(
            np.sum(expected_values * weights_all) / np.sum(weights_all)
        ),
        "row_lower_tail_protocol_exact_chosen_utility": row_tail,
        "component_lower_tail_protocol_exact_chosen_utility": component_tail,
        "week_lower_tail_protocol_exact_chosen_utility": week_tail,
        "component_count": component_count,
        "week_behavior_groups": week_count,
        "protocol_action_value_loss": value_error / total_weight,
        "residual_logit_L2": residual_l2 / total_weight,
        "H014_policy_KL": policy_kl / total_weight,
        "mean_size": total_size / total_weight,
        "calls": total_calls,
        "call_rate": total_calls / total_rows,
        "call_precision": total_correct / total_calls if total_calls else 0.0,
        "equal_market_call_rate": weighted_calls / total_weight,
        "equal_market_call_precision": (
            weighted_correct / weighted_calls if weighted_calls else 0.0
        ),
        "base_calls": base_calls_total,
        "correct_base_calls": correct_base_calls_total,
        "wrong_base_calls": wrong_base_calls_total,
        "correct_base_call_retention_rate": (
            retained_correct_base_calls_total / correct_base_calls_total
            if correct_base_calls_total
            else 0.0
        ),
        "wrong_base_call_veto_rate": (
            vetoed_wrong_base_calls_total / wrong_base_calls_total
            if wrong_base_calls_total
            else 0.0
        ),
        "actions": {
            "CALL_OUTCOME_0": int(action_counts[0]),
            "CALL_OUTCOME_1": int(action_counts[1]),
            "SKIP": int(action_counts[2]),
        },
    }


def _adapter_state(model: SphinxTraceS0H012) -> dict[str, Any]:
    if model.action_residual is None or model.protocol_action_value is None:
        raise RuntimeError("H018 adapter heads are missing")
    return {
        "action_residual": model.action_residual.state_dict(),
        "protocol_action_value": model.protocol_action_value.state_dict(),
    }


def train(
    config_path: Path,
    policy_config_path: Path,
    initial_policy_config_path: Path,
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
    research_id = str(config.get("research_id") or "")
    if research_id not in {"SPH-T-H018", "SPH-T-H019"}:
        raise RuntimeError("Protocol adapter trainer requires H018 or H019")
    training_config = dict(config["training"])
    registered_seeds = [int(value) for value in training_config["seeds"]]
    if seed not in registered_seeds:
        raise RuntimeError(f"H018 seed {seed} was not pre-registered")
    policy_config = load_json(policy_config_path)
    initial_policy_config = load_json(initial_policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("H018 full policy training requires CUDA")
    device = torch.device("cuda")
    shards, state_manifest = _state_shards(
        state_dir, encoding_dir, pack_dir, initial_policy_dir, config
    )
    market_weights, weighting_receipt = _equal_market_weights(shards, 2)
    downside_factors, downside_normalization, downside_receipt = (
        _fit_week_downside_weights(
            shards,
            market_weights,
            float(training_config["maximum_fit_downside_sample_multiplier"]),
        )
    )
    model, initial_result, initialization = _initialize_model(
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        initial_policy_config,
        policy_config,
        device,
    )
    expected_digest = config["dependencies"]["initial_policy"]["model_tensor_sha256"]
    if initialization["source_model_tensor_sha256"] != expected_digest:
        raise RuntimeError("H018 initial H014 tensor digest changed")
    equivalence = _verify_initial_equivalence(
        model,
        initial_policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        initial_policy_config,
        shards[0],
        device,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(
        not name.startswith(("action_residual.", "protocol_action_value."))
        for name in trainable_names
    ):
        raise RuntimeError("H018 trainable parameter boundary is invalid")

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
        "initial_policy_config_sha256": sha256_file(initial_policy_config_path),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best-policy.pt"
    epochs_dir = output_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training_config["policy_learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    batch_size = int(training_config["batch_size"])
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_selection = -math.inf
    stale_epochs = 0
    initial_selection: dict[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H018 checkpoint belongs to another training contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_selection = float(checkpoint["best_selection"])
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
            1,
            training_config,
            market_weights,
            batch_size * 2,
            device,
        )
        best_selection = float(
            initial_selection["equal_market_mean_protocol_exact_chosen_utility"]
        )
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
        model.eval()
        if model.action_residual is None or model.protocol_action_value is None:
            raise RuntimeError("H018 adapter heads disappeared before training")
        model.action_residual.train()
        model.protocol_action_value.train()
        order = list(range(len(shards)))
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
        loss_sum = 0.0
        weight_seen = 0.0
        rows_seen = 0
        for shard_index in order:
            shard = shards[shard_index]
            indices = _indices(shard, 0, seed=seed, epoch=epoch, shuffle=True)
            for offset in range(0, len(indices), batch_size):
                batch = _batch(shard, indices[offset : offset + batch_size])
                weights_numpy = _training_weights(
                    batch,
                    market_weights,
                    downside_factors,
                    downside_normalization,
                )
                weights = torch.from_numpy(weights_numpy).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, labels, _ = _forward(model, batch, device)
                    loss, _ = _loss(
                        output, labels, batch, weights, training_config
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(training_config["gradient_clip_norm"])
                )
                optimizer.step()
                batch_weight = float(weights.sum())
                loss_sum += float(loss.detach()) * batch_weight
                weight_seen += batch_weight
                rows_seen += len(batch.labels)
        scheduler.step()
        selection = _evaluate(
            model,
            shards,
            1,
            training_config,
            market_weights,
            batch_size * 2,
            device,
        )
        score = float(selection["equal_market_mean_protocol_exact_chosen_utility"])
        epoch_path = epochs_dir / f"epoch-{epoch:03d}.pt"
        _atomic_torch_save(
            epoch_path,
            {
                "adapter": _adapter_state(model),
                "epoch": epoch,
                "selection": selection,
                "contract_sha256": contract_sha256,
                "source_model_tensor_sha256": expected_digest,
            },
        )
        history.append(
            {
                "epoch": epoch,
                "fit_loss": loss_sum / max(weight_seen, 1e-8),
                "fit_rows": rows_seen,
                "fit_sample_weight": weight_seen,
                "selection": selection,
                "selection_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "adapter_sha256": sha256_file(epoch_path),
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
                "record_type": (
                    "h019_learned_loss_veto_training_progress"
                    if research_id == "SPH-T-H019"
                    else "h018_conservative_residual_training_progress"
                ),
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
        if stale_epochs >= int(training_config["early_stopping_patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("contract_sha256") != contract_sha256:
        raise RuntimeError("H018 best policy belongs to another training contract")
    model.load_state_dict(best["model"])
    fit = _evaluate(
        model,
        shards,
        0,
        training_config,
        market_weights,
        batch_size * 2,
        device,
    )
    selection = _evaluate(
        model,
        shards,
        1,
        training_config,
        market_weights,
        batch_size * 2,
        device,
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "h019_learned_loss_veto_policy_result"
            if research_id == "SPH-T-H019"
            else "h018_conservative_residual_policy_result"
        ),
        "research_id": research_id,
        "completed_at": now_utc(),
        "valid": math.isfinite(best_selection),
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "policy_config_sha256": sha256_file(policy_config_path),
        "initial_policy_config_sha256": sha256_file(initial_policy_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_sha256,
        "contract_sha256": contract_sha256,
        **source_hashes,
        "market_encoding_policy_result_sha256": initial_result.get(
            "market_encoding_policy_result_sha256"
        ),
        "parameters": parameter_count(model),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_names": trainable_names,
        "partition_sha256": state_manifest["partition_sha256"],
        "fit_components": state_manifest["fit_components"],
        "selection_components": state_manifest["selection_components"],
        "initialization": initialization,
        "initial_equivalence": equivalence,
        "weighting": weighting_receipt,
        "fit_week_downside_weighting": downside_receipt,
        "initial_selection": initial_selection,
        "best_epoch": best_epoch,
        "best_selection_equal_market_chosen_utility": best_selection,
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
    value.add_argument(
        "--initial-policy-config", type=Path, default=DEFAULT_INITIAL_POLICY_CONFIG
    )
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
        args.initial_policy_config.resolve(),
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
