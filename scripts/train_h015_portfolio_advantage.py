"""Train H015 on equal-market counterfactual and logged portfolio advantage."""

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
from sphinx_trace.on_policy_pack import validate_on_policy_shard
from sphinx_trace.policy_checkpoint import load_policy_checkpoint
from sphinx_trace.policy_training import (
    logged_execution_action_value_loss,
    selective_log_utility_loss,
)
from sphinx_trace.protocol_tail_pack import validate_protocol_tail_shard
from sphinx_trace.protocol_tail_training import protocol_tail_utility_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h015_on_policy_portfolio_advantage_v1.json"
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
    ROOT / "scripts" / "train_h017_protocol_tail_policy.py",
    ROOT / "src" / "sphinx_trace" / "on_policy_pack.py",
    ROOT / "src" / "sphinx_trace" / "policy_training.py",
    ROOT / "src" / "sphinx_trace" / "protocol_tail_pack.py",
    ROOT / "src" / "sphinx_trace" / "protocol_tail_training.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model.py",
)


@dataclass(frozen=True, slots=True)
class StateShard:
    behavior_id: str
    behavior_code: int
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
    market_ids: NDArray[np.int64]
    behavior_policy_codes: NDArray[np.uint8]
    behavior_action_ids: NDArray[np.int64]
    realized_action_values: NDArray[np.float32]
    execution_fractions: NDArray[np.float32]
    winning_payout_multipliers: NDArray[np.float32] | None
    reference_action_values: NDArray[np.float32] | None


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
    manifest = _load_object(state_dir / "manifest.json")
    research_id = str(config.get("research_id") or "")
    protocol_tail = research_id == "SPH-T-H017"
    expected_manifest_type = (
        "h017_protocol_tail_pack_manifest"
        if protocol_tail
        else "h015_on_policy_portfolio_advantage_pack_manifest"
    )
    initial_result_path = initial_policy_dir / "result.json"
    initial_result_sha256 = sha256_file(initial_result_path)
    initial_result = _load_object(initial_result_path)
    if (
        manifest.get("record_type") != expected_manifest_type
        or manifest.get("valid") is not True
        or manifest.get("test_labels_opened") is not False
        or int(manifest.get("test_rows_consumed", -1)) != 0
        or int(manifest.get("calibration_rows_consumed", -1)) != 0
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("pack_manifest_sha256")
        != sha256_file(pack_dir / "manifest.json")
        or manifest.get("encoding_manifest_sha256")
        != sha256_file(encoding_dir / "manifest.json")
        or config["dependencies"]["initial_policy"]["result_sha256"]
        != initial_result_sha256
        or config["dependencies"]["initial_policy"]["best_model_sha256"]
        != sha256_file(initial_policy_dir / "best-policy.pt")
        or initial_result.get("test_labels_opened") is not False
    ):
        raise RuntimeError(f"{research_id} training source contract changed")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise RuntimeError("H015 state manifest has no shards")
    shards: list[StateShard] = []
    total_rows = 0
    behavior_rows: dict[int, int] = {}
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise RuntimeError("H015 state shard metadata is invalid")
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
            raise RuntimeError(f"H015 state shard binding changed: {behavior_id}:{date}")
        receipt = _load_object(receipt_path)
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise RuntimeError(f"H015 state receipt is invalid: {behavior_id}:{date}")
        validate_on_policy_shard(
            state,
            files,
            expected_rows=rows,
            expected_behavior_code=behavior_code,
        )
        if protocol_tail:
            validate_protocol_tail_shard(state, files, expected_rows=rows)
        shards.append(
            StateShard(behavior_id, behavior_code, date, state, encoding, pack, rows)
        )
        total_rows += rows
        behavior_rows[behavior_code] = behavior_rows.get(behavior_code, 0) + rows
    expected_rows = int(config["corpus"]["rows_expected"])
    expected_per_behavior = expected_rows // int(config["corpus"]["behavior_policies"])
    if (
        total_rows != int(manifest.get("rows", -1))
        or total_rows != expected_rows
        or sorted(behavior_rows) != list(range(int(config["corpus"]["behavior_policies"])))
        or any(rows != expected_per_behavior for rows in behavior_rows.values())
    ):
        raise RuntimeError(f"{research_id} state row or behavior coverage changed")
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
        key = f"{shard.behavior_code}:{shard.date}"
        shard_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        np.random.default_rng(seed ^ epoch ^ shard_seed).shuffle(indices)
    return indices


def _batch(shard: StateShard, indices: NDArray[np.int64]) -> StateBatch:
    rows = np.load(shard.state / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    encoding_offsets = np.load(
        shard.state / "encoding_offsets.npy", mmap_mode="r", allow_pickle=False
    )
    markets = np.load(shard.state / "market_ids.npy", mmap_mode="r", allow_pickle=False)
    behaviors = np.load(
        shard.state / "behavior_policy_codes.npy", mmap_mode="r", allow_pickle=False
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
    actions = np.load(
        shard.state / "behavior_action_ids.npy", mmap_mode="r", allow_pickle=False
    )
    values = np.load(
        shard.state / "realized_action_values.npy", mmap_mode="r", allow_pickle=False
    )
    fractions = np.load(
        shard.state / "execution_fractions.npy", mmap_mode="r", allow_pickle=False
    )
    payout_path = shard.state / "winning_payout_multipliers.npy"
    reference_path = shard.state / "reference_action_values.npy"
    if payout_path.is_file() != reference_path.is_file():
        raise RuntimeError(f"Protocol target arrays are incomplete: {shard.date}")
    payout_multipliers = (
        np.load(payout_path, mmap_mode="r", allow_pickle=False)
        if payout_path.is_file()
        else None
    )
    reference_values = (
        np.load(reference_path, mmap_mode="r", allow_pickle=False)
        if reference_path.is_file()
        else None
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
    baselines = np.load(
        shard.pack / "baselines.npy", mmap_mode="r", allow_pickle=False
    )
    selected_rows = np.asarray(rows[indices], dtype=np.int64)
    selected_offsets = np.asarray(encoding_offsets[indices], dtype=np.int64)
    if bool((selected_offsets < 0).any()) or bool(
        (selected_offsets >= len(source_rows)).any()
    ):
        raise RuntimeError(f"H015 encoding offsets are invalid: {shard.date}")
    if not np.array_equal(source_rows[selected_offsets], selected_rows):
        raise RuntimeError(f"H015 state and market encodings no longer align: {shard.date}")
    output = StateBatch(
        market_latents=np.asarray(latents[selected_offsets], dtype=np.float32),
        terminal_logits=np.asarray(terminal[selected_offsets], dtype=np.float32),
        uncertainty_log_scales=np.asarray(
            uncertainty[selected_offsets], dtype=np.float32
        ),
        portfolio_features=np.asarray(portfolio[indices], dtype=np.float32),
        prediction_memory_features=np.asarray(memory[indices], dtype=np.float32),
        previous_action_ids=np.asarray(previous[indices], dtype=np.int64),
        physical_action_masks=np.asarray(physical[indices], dtype=np.uint8),
        labels=np.asarray(labels[selected_rows], dtype=np.float32),
        baselines=np.asarray(baselines[selected_rows], dtype=np.float32),
        market_ids=np.asarray(markets[indices], dtype=np.int64),
        behavior_policy_codes=np.asarray(behaviors[indices], dtype=np.uint8),
        behavior_action_ids=np.asarray(actions[indices], dtype=np.int64),
        realized_action_values=np.asarray(values[indices], dtype=np.float32),
        execution_fractions=np.asarray(fractions[indices], dtype=np.float32),
        winning_payout_multipliers=(
            None
            if payout_multipliers is None
            else np.asarray(payout_multipliers[indices], dtype=np.float32)
        ),
        reference_action_values=(
            None
            if reference_values is None
            else np.asarray(reference_values[indices], dtype=np.float32)
        ),
    )
    if (
        output.market_latents.ndim != 2
        or output.market_latents.shape[0] != len(indices)
        or not bool(np.isfinite(output.market_latents).all())
        or not bool(np.isfinite(output.terminal_logits).all())
        or not bool(np.isfinite(output.uncertainty_log_scales).all())
        or not bool(np.isin(output.labels, (0.0, 1.0)).all())
        or not bool(np.isfinite(output.baselines).all())
        or not bool(np.isfinite(output.realized_action_values).all())
        or not bool(np.isin(output.behavior_action_ids, (0, 1, 2)).all())
        or (
            output.winning_payout_multipliers is not None
            and (
                output.winning_payout_multipliers.shape != (len(indices), 2)
                or not bool(np.isfinite(output.winning_payout_multipliers).all())
                or bool((output.winning_payout_multipliers <= 0.0).any())
            )
        )
        or (
            output.reference_action_values is not None
            and (
                output.reference_action_values.shape != (len(indices), 3)
                or not bool(np.isfinite(output.reference_action_values).all())
            )
        )
    ):
        raise RuntimeError(f"H015 training batch is invalid: {shard.date}")
    return output


def _equal_market_weights(
    shards: list[StateShard], behavior_policies: int
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    maximum_market = 0
    for shard in shards:
        if not shard.rows:
            continue
        markets = np.load(
            shard.state / "market_ids.npy", mmap_mode="r", allow_pickle=False
        )
        maximum_market = max(maximum_market, int(markets.max(initial=0)))
    counts = np.zeros((2, behavior_policies, maximum_market + 1), dtype=np.int64)
    behavior_rows = np.zeros((2, behavior_policies), dtype=np.int64)
    for shard in shards:
        markets = np.load(
            shard.state / "market_ids.npy", mmap_mode="r", allow_pickle=False
        )
        for partition_code in (0, 1):
            indices = _indices(
                shard, partition_code, seed=0, epoch=0, shuffle=False
            )
            if not len(indices):
                continue
            counts[partition_code, shard.behavior_code] += np.bincount(
                np.asarray(markets[indices], dtype=np.int64),
                minlength=counts.shape[2],
            )
            behavior_rows[partition_code, shard.behavior_code] += len(indices)
    weights = np.zeros_like(counts, dtype=np.float64)
    market_counts: list[list[int]] = [[], []]
    total_weights: list[list[float]] = [[], []]
    for partition_code in (0, 1):
        for behavior_code in range(behavior_policies):
            observed = counts[partition_code, behavior_code] > 0
            markets = int(observed.sum())
            rows = int(behavior_rows[partition_code, behavior_code])
            if not markets or not rows:
                raise RuntimeError(
                    f"H015 behavior {behavior_code} has no partition {partition_code} markets"
                )
            weights[partition_code, behavior_code, observed] = (
                float(rows)
                / float(markets)
                / counts[partition_code, behavior_code, observed]
            )
            total = float(
                (
                    weights[partition_code, behavior_code]
                    * counts[partition_code, behavior_code]
                ).sum()
            )
            if not math.isclose(total, float(rows), rel_tol=1e-9):
                raise RuntimeError(
                    "H015 equal-market weights do not normalize to partition rows"
                )
            market_counts[partition_code].append(markets)
            total_weights[partition_code].append(total)
    return weights.astype(np.float32), {
        "fit_markets_by_behavior": market_counts[0],
        "selection_markets_by_behavior": market_counts[1],
        "fit_rows_by_behavior": behavior_rows[0].tolist(),
        "selection_rows_by_behavior": behavior_rows[1].tolist(),
        "fit_total_weight_by_behavior": total_weights[0],
        "selection_total_weight_by_behavior": total_weights[1],
        "minimum_positive_weight": float(weights[weights > 0].min()),
        "maximum_weight": float(weights.max()),
    }


def _sample_weights(
    batch: StateBatch,
    market_weights: NDArray[np.float32],
    partition_code: int,
) -> NDArray[np.float32]:
    values = market_weights[
        partition_code, batch.behavior_policy_codes, batch.market_ids
    ]
    if values.shape != batch.market_ids.shape or bool((values <= 0.0).any()):
        raise RuntimeError("H015 batch contains an unweighted fit market")
    return np.asarray(values, dtype=np.float32)


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


def _combined_loss(
    output: dict[str, Tensor],
    labels: Tensor,
    baselines: Tensor,
    batch: StateBatch,
    sample_weights: Tensor,
    utility_config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    physical_action_mask = torch.from_numpy(batch.physical_action_masks).to(
        labels.device, non_blocking=True
    )
    if str(utility_config.get("loss_mode")) == (
        "protocol_exact_counterfactual_plus_logged_execution_plus_lower_tail"
    ):
        if (
            batch.winning_payout_multipliers is None
            or batch.reference_action_values is None
        ):
            raise RuntimeError("H017 batch has no protocol-exact targets")
        return protocol_tail_utility_loss(
            output,
            labels,
            torch.from_numpy(batch.winning_payout_multipliers).to(
                labels.device, non_blocking=True
            ),
            torch.from_numpy(batch.reference_action_values).to(
                labels.device, non_blocking=True
            ),
            torch.from_numpy(batch.behavior_action_ids).to(
                labels.device, non_blocking=True
            ),
            torch.from_numpy(batch.realized_action_values).to(
                labels.device, non_blocking=True
            ),
            torch.from_numpy(batch.execution_fractions).to(
                labels.device, non_blocking=True
            ),
            utility_config,
            sample_weights=sample_weights,
            physical_action_mask=physical_action_mask,
        )
    base_config = {
        **utility_config,
        "loss_mode": "counterfactual_action_value",
        "action_value_weight": float(
            utility_config["counterfactual_action_value_weight"]
        ),
    }
    base_loss, base_metrics = selective_log_utility_loss(
        output,
        labels,
        baselines,
        base_config,
        sample_weights=sample_weights,
        physical_action_mask=physical_action_mask,
    )
    logged_loss, logged_metrics = logged_execution_action_value_loss(
        output,
        torch.from_numpy(batch.behavior_action_ids).to(labels.device, non_blocking=True),
        torch.from_numpy(batch.realized_action_values).to(
            labels.device, non_blocking=True
        ),
        torch.from_numpy(batch.execution_fractions).to(
            labels.device, non_blocking=True
        ),
        sample_weights=sample_weights,
        physical_action_mask=physical_action_mask,
    )
    loss = base_loss + float(
        utility_config["logged_execution_action_value_weight"]
    ) * logged_loss
    return loss, {**base_metrics, **logged_metrics, "combined_loss": loss.detach()}


@torch.inference_mode()
def _evaluate(
    model: SphinxTraceS0H012,
    shards: list[StateShard],
    partition_code: int,
    utility_config: dict[str, Any],
    market_weights: NDArray[np.float32],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    protocol_tail = str(utility_config.get("loss_mode")) == (
        "protocol_exact_counterfactual_plus_logged_execution_plus_lower_tail"
    )
    totals = {
        "rows": 0.0,
        "weight": 0.0,
        "loss": 0.0,
        "chosen": 0.0,
        "expected": 0.0,
        "tail": 0.0,
        "logged_loss": 0.0,
        "logged_absolute_error": 0.0,
        "size": 0.0,
        "calls": 0.0,
        "correct": 0.0,
        "weighted_calls": 0.0,
        "weighted_correct": 0.0,
        "filled": 0.0,
    }
    action_counts = np.zeros(3, dtype=np.int64)
    behavior_action_counts: dict[int, NDArray[np.int64]] = {}
    for shard in shards:
        indices = _indices(shard, partition_code, seed=0, epoch=0, shuffle=False)
        for offset in range(0, len(indices), batch_size):
            batch = _batch(shard, indices[offset : offset + batch_size])
            weights_numpy = _sample_weights(batch, market_weights, partition_code)
            weights = torch.from_numpy(weights_numpy).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, labels, baselines = _forward(model, batch, device)
                loss, metrics = _combined_loss(
                    output, labels, baselines, batch, weights, utility_config
                )
            chosen = output["action_logits"][:, :3].argmax(dim=-1)
            chosen_numpy = chosen.cpu().numpy().astype(np.int64)
            action_counts += np.bincount(chosen_numpy, minlength=3)
            behavior_counts = behavior_action_counts.setdefault(
                shard.behavior_code, np.zeros(3, dtype=np.int64)
            )
            behavior_counts += np.bincount(chosen_numpy, minlength=3)
            calls = chosen != 2
            correct = ((chosen == 0) & (labels == 1)) | (
                (chosen == 1) & (labels == 0)
            )
            weight_sum = float(weights.sum())
            size = output["position_size_beta_alpha"].float() / (
                output["position_size_beta_alpha"].float()
                + output["position_size_beta_beta"].float()
            ).clamp_min(1e-8)
            totals["rows"] += len(batch.labels)
            totals["weight"] += weight_sum
            totals["loss"] += float(loss) * weight_sum
            if protocol_tail:
                totals["chosen"] += (
                    float(metrics["protocol_exact_chosen_utility"]) * weight_sum
                )
                totals["expected"] += (
                    float(metrics["protocol_exact_expected_utility"]) * weight_sum
                )
                totals["tail"] += (
                    float(metrics["protocol_exact_tail_utility"]) * weight_sum
                )
            else:
                totals["chosen"] += float(
                    metrics["weighted_chosen_log_utility_sum"]
                )
                totals["expected"] += (
                    float(metrics["expected_log_utility"]) * weight_sum
                )
            totals["logged_loss"] += (
                float(metrics["logged_execution_action_value_loss"]) * weight_sum
            )
            totals["logged_absolute_error"] += (
                float(metrics["logged_execution_absolute_error"]) * weight_sum
            )
            totals["size"] += float((size * weights).sum())
            totals["calls"] += int(calls.sum())
            totals["correct"] += int(correct[calls].sum())
            totals["weighted_calls"] += float(weights[calls].sum())
            totals["weighted_correct"] += float(weights[calls & correct].sum())
            totals["filled"] += int((batch.execution_fractions > 0.0).sum())
    rows = int(totals["rows"])
    weight_sum = totals["weight"]
    if not rows or weight_sum <= 0.0:
        raise RuntimeError(f"H015 evaluation partition {partition_code} has no rows")
    call_count = int(totals["calls"])
    result = {
        "rows": rows,
        "sample_weight_sum": weight_sum,
        "loss": totals["loss"] / weight_sum,
        "equal_market_mean_chosen_counterfactual_utility": totals["chosen"]
        / weight_sum,
        "equal_market_mean_expected_counterfactual_utility": totals["expected"]
        / weight_sum,
        "logged_execution_action_value_loss": totals["logged_loss"] / weight_sum,
        "logged_execution_absolute_error": totals["logged_absolute_error"] / weight_sum,
        "mean_size": totals["size"] / weight_sum,
        "calls": call_count,
        "call_rate": call_count / rows,
        "correct_calls": int(totals["correct"]),
        "call_precision": totals["correct"] / call_count if call_count else 0.0,
        "equal_market_call_rate": totals["weighted_calls"] / weight_sum,
        "equal_market_call_precision": (
            totals["weighted_correct"] / totals["weighted_calls"]
            if totals["weighted_calls"]
            else 0.0
        ),
        "logged_filled_rows": int(totals["filled"]),
        "actions": {
            "CALL_OUTCOME_0": int(action_counts[0]),
            "CALL_OUTCOME_1": int(action_counts[1]),
            "SKIP": int(action_counts[2]),
        },
        "actions_by_behavior": {
            str(code): {
                "CALL_OUTCOME_0": int(counts[0]),
                "CALL_OUTCOME_1": int(counts[1]),
                "SKIP": int(counts[2]),
            }
            for code, counts in sorted(behavior_action_counts.items())
        },
    }
    if protocol_tail:
        result["equal_market_mean_protocol_exact_chosen_utility"] = (
            totals["chosen"] / weight_sum
        )
        result["equal_market_mean_protocol_exact_expected_utility"] = (
            totals["expected"] / weight_sum
        )
        result["equal_market_mean_protocol_exact_tail_utility"] = (
            totals["tail"] / weight_sum
        )
    return result


def _selection_score(metrics: dict[str, Any]) -> float:
    value = metrics.get(
        "equal_market_mean_protocol_exact_chosen_utility",
        metrics["equal_market_mean_chosen_counterfactual_utility"],
    )
    if isinstance(value, dict):
        raise TypeError("H015 equal-market selection utility must be numeric")
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
    research_id = str(config.get("research_id") or "SPH-T-H015")
    policy_config = load_json(policy_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    utility_config = dict(config["training"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{research_id} full policy training requires CUDA")
    device = torch.device("cuda")
    config_sha256 = sha256_file(config_path)
    shards, state_manifest = _state_shards(
        state_dir,
        encoding_dir,
        pack_dir,
        initial_policy_dir,
        config,
        config_sha256,
    )
    market_weights, weighting_receipt = _equal_market_weights(
        shards, int(config["corpus"]["behavior_policies"])
    )
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
        "config_sha256": config_sha256,
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
    expected_tensor_sha256 = config["dependencies"]["initial_policy"].get(
        "model_tensor_sha256"
    )
    if (
        expected_tensor_sha256 is not None
        and _module_digest(model) != expected_tensor_sha256
    ):
        raise RuntimeError(f"{research_id} initial model tensor digest changed")
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
    initial_selection: dict[str, Any] | None = None
    best_selection = -math.inf
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H015 checkpoint belongs to another training contract")
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
            1,
            utility_config,
            market_weights,
            batch_size * 4,
            device,
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
        weight_seen = 0.0
        rows_seen = 0
        for shard_index in order:
            shard = shards[shard_index]
            indices = _indices(shard, 0, seed=seed, epoch=epoch, shuffle=True)
            for offset in range(0, len(indices), batch_size):
                batch = _batch(shard, indices[offset : offset + batch_size])
                weights_numpy = _sample_weights(batch, market_weights, 0)
                weights = torch.from_numpy(weights_numpy).to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, labels, baselines = _forward(model, batch, device)
                    loss, _ = _combined_loss(
                        output, labels, baselines, batch, weights, utility_config
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(utility_config["gradient_clip_norm"])
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
            utility_config,
            market_weights,
            batch_size * 4,
            device,
        )
        score = _selection_score(selection)
        history.append(
            {
                "epoch": epoch,
                "fit_loss": loss_sum / max(weight_seen, 1e-8),
                "fit_rows": rows_seen,
                "fit_sample_weight": weight_seen,
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
                "record_type": (
                    "h017_protocol_tail_training_progress"
                    if research_id == "SPH-T-H017"
                    else "h015_portfolio_advantage_training_progress"
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
        if stale_epochs >= int(utility_config["early_stopping_patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("contract_sha256") != contract_sha256:
        raise RuntimeError("H015 best policy belongs to another training contract")
    model.load_state_dict(best["model"])
    if _module_digest(model.outcome_backbone) != market_backbone_sha256:
        raise RuntimeError("H015 frozen market backbone changed during training")
    fit = _evaluate(
        model,
        shards,
        0,
        utility_config,
        market_weights,
        batch_size * 4,
        device,
    )
    selection = _evaluate(
        model,
        shards,
        1,
        utility_config,
        market_weights,
        batch_size * 4,
        device,
    )
    market_encoding_source = loaded.result.get("market_encoding_policy_result_sha256")
    if not isinstance(market_encoding_source, str) or len(market_encoding_source) != 64:
        raise RuntimeError("H015 initial policy lost its frozen market encoding source")
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "h017_protocol_tail_policy_result"
            if research_id == "SPH-T-H017"
            else "h015_portfolio_advantage_policy_result"
        ),
        "research_id": research_id,
        "completed_at": now_utc(),
        "valid": math.isfinite(best_selection),
        "config_sha256": config_sha256,
        "policy_config_sha256": sha256_file(policy_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "implementation_sha256": implementation_sha256,
        "contract_sha256": contract_sha256,
        **source_hashes,
        "market_encoding_policy_result_sha256": market_encoding_source,
        "market_backbone_frozen": True,
        "market_backbone_sha256": market_backbone_sha256,
        "parameters": parameter_count(model),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "partition_sha256": state_manifest["partition_sha256"],
        "fit_components": state_manifest["fit_components"],
        "selection_components": state_manifest["selection_components"],
        "weighting": weighting_receipt,
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
