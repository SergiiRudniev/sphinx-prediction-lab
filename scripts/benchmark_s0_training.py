from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from sphinx_corpus.io import iter_jsonl_zst

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_throughput_v1.json"


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _hash_unit(value: str) -> float:
    digest = hashlib.blake2b(value.encode(), digest_size=4).digest()
    integer = int.from_bytes(digest, "big")
    return (integer / 0xFFFFFFFF) * 2.0 - 1.0


def _trade_arrays(rows: list[dict[str, Any]]) -> dict[str, NDArray[np.float64]]:
    timestamps = np.asarray([_number(row, "timestamp_unix") for row in rows])
    prices = np.asarray([_number(row, "price") for row in rows])
    sizes = np.asarray([_number(row, "size") for row in rows])
    notionals = np.asarray([_number(row, "notional_usd") for row in rows])
    sides = np.asarray([1.0 if str(row.get("side", "")).upper() == "BUY" else -1.0 for row in rows])
    outcomes = np.asarray([_number(row, "outcome_index") for row in rows])
    return {
        "timestamps": timestamps,
        "prices": prices,
        "sizes": sizes,
        "notionals": notionals,
        "sides": sides,
        "outcomes": outcomes,
    }


def build_sequence(
    rows: list[dict[str, Any]],
    *,
    trade_tokens: int,
    wallet_tokens: int,
    context_tokens: int,
    feature_width: int,
) -> tuple[NDArray[np.float16], NDArray[np.uint8], NDArray[np.float16]]:
    sequence_length = trade_tokens + wallet_tokens + context_tokens
    features = np.zeros((sequence_length, feature_width), dtype=np.float16)
    token_types = np.zeros(sequence_length, dtype=np.uint8)
    arrays = _trade_arrays(rows)
    timestamps = arrays["timestamps"]
    prices = arrays["prices"]
    sizes = arrays["sizes"]
    notionals = arrays["notionals"]
    sides = arrays["sides"]
    outcomes = arrays["outcomes"]
    deltas = np.diff(timestamps, prepend=timestamps[0])
    hours = np.mod(timestamps, 86400.0) / 86400.0
    wallets = [str(row.get("wallet") or "") for row in rows]
    condition_id = str(rows[-1].get("condition_id") or "")
    condition_hash = _hash_unit(condition_id)

    trade = features[:trade_tokens]
    trade[:, 0] = prices
    trade[:, 1] = np.clip(np.log1p(sizes) / 12.0, 0.0, 1.0)
    trade[:, 2] = np.clip(np.log1p(notionals) / 12.0, 0.0, 1.0)
    trade[:, 3] = sides
    trade[:, 4] = np.clip(np.log1p(np.maximum(deltas, 0.0)) / 12.0, 0.0, 1.0)
    trade[:, 5] = np.clip(outcomes / 16.0, 0.0, 1.0)
    trade[:, 6] = np.sin(2.0 * math.pi * hours)
    trade[:, 7] = np.cos(2.0 * math.pi * hours)
    trade[:, 8] = np.asarray([_hash_unit(wallet) for wallet in wallets])
    trade[:, 9] = condition_hash
    trade[:, 10] = np.linspace(-1.0, 1.0, trade_tokens)
    trade[:, 11] = np.clip(np.cumsum(notionals) / max(float(notionals.sum()), 1.0), 0, 1)
    trade[:, 12] = np.clip(np.cumsum(sides) / trade_tokens, -1, 1)
    trade[:, 13] = np.clip(prices - np.mean(prices), -1, 1)
    trade[:, 14] = np.clip(notionals / max(float(np.max(notionals)), 1.0), 0, 1)
    trade[:, 15] = 1.0

    wallet_start = trade_tokens
    token_types[wallet_start : wallet_start + wallet_tokens] = 1
    by_wallet: dict[str, list[int]] = defaultdict(list)
    for index, wallet in enumerate(wallets):
        by_wallet[wallet].append(index)
    ranked_wallets = sorted(
        by_wallet.items(),
        key=lambda item: (float(notionals[item[1]].sum()), item[0]),
        reverse=True,
    )[:wallet_tokens]
    for slot, (wallet, indices) in enumerate(ranked_wallets):
        output = features[wallet_start + slot]
        index_array = np.asarray(indices, dtype=np.int64)
        wallet_notional = notionals[index_array]
        output[0] = len(indices) / trade_tokens
        output[1] = np.clip(np.log1p(wallet_notional.sum()) / 14.0, 0, 1)
        output[2] = prices[index_array].mean()
        output[3] = sides[index_array].mean()
        output[4] = np.clip(index_array[0] / trade_tokens, 0, 1)
        output[5] = np.clip(index_array[-1] / trade_tokens, 0, 1)
        output[6] = _hash_unit(wallet)
        output[7] = condition_hash
        output[8] = np.clip(wallet_notional.max() / max(float(notionals.max()), 1.0), 0, 1)
        output[9] = np.clip(np.std(prices[index_array]), 0, 1)
        output[15] = 1.0

    context_start = trade_tokens + wallet_tokens
    token_types[context_start:] = 2
    block_size = max(1, trade_tokens // context_tokens)
    for slot in range(context_tokens):
        left = min(slot * block_size, trade_tokens - 1)
        right = min(left + block_size, trade_tokens)
        output = features[context_start + slot]
        output[0] = prices[left:right].mean()
        output[1] = np.clip(np.log1p(notionals[left:right].sum()) / 14.0, 0, 1)
        output[2] = sides[left:right].mean()
        output[3] = np.clip((timestamps[right - 1] - timestamps[left]) / 86400.0, 0, 1)
        output[4] = slot / max(context_tokens - 1, 1)
        output[5] = condition_hash
        output[6] = np.sin(2.0 * math.pi * slot / context_tokens)
        output[7] = np.cos(2.0 * math.pi * slot / context_tokens)
        output[8] = len(ranked_wallets) / max(wallet_tokens, 1)
        output[9] = np.clip(np.std(prices), 0, 1)
        output[10] = np.clip(np.mean(notionals), 0, 1)
        output[15] = 1.0

    target = np.asarray(
        [
            prices[-1],
            np.clip(prices[-1] - prices[-2], -1, 1),
            np.clip(prices[-1] - prices[trade_tokens // 2], -1, 1),
            np.clip(prices[-1] - prices[0], -1, 1),
            np.clip(sides[-16:].mean(), -1, 1),
            np.clip(np.log1p(notionals[-16:].sum()) / 14.0, 0, 1),
            np.clip(np.std(prices), 0, 1),
            1.0 if prices[-1] >= 0.5 else 0.0,
        ],
        dtype=np.float16,
    )
    return features, token_types, target


def spread_paths(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    return [
        paths[min(int(index * len(paths) / maximum), len(paths) - 1)] for index in range(maximum)
    ]


def pack_ledger(config: dict[str, Any], data_dir: Path, output_dir: Path) -> dict[str, Any]:
    source = config["source"]
    sequence_count = int(source["sequence_count"])
    trade_tokens = int(source["trade_tokens"])
    wallet_tokens = int(source["wallet_tokens"])
    context_tokens = int(source["context_tokens"])
    feature_width = int(source["feature_width"])
    stride = int(source["stride_trades"])
    namespace = str(source["namespace"])
    source_root = data_dir / namespace
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    source_listing_at = utc_now()
    paths = spread_paths(
        sorted(source_root.rglob("*.jsonl.zst")),
        int(source["source_file_limit"]),
    )
    if not paths:
        raise RuntimeError(f"No normalized Ledger files found under {source_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_length = trade_tokens + wallet_tokens + context_tokens
    feature_path = output_dir / "features.npy"
    type_path = output_dir / "token_types.npy"
    target_path = output_dir / "targets.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(sequence_count, sequence_length, feature_width),
    )
    token_types = np.lib.format.open_memmap(
        type_path,
        mode="w+",
        dtype=np.uint8,
        shape=(sequence_count, sequence_length),
    )
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float16,
        shape=(sequence_count, int(config["model"]["output_width"])),
    )

    written = 0
    rows_read = 0
    files_read = 0
    used_paths: list[str] = []
    started = time.perf_counter()
    for path in paths:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in iter_jsonl_zst(path):
            grouped[str(row.get("condition_id") or "")].append(row)
            rows_read += 1
        files_read += 1
        produced_from_file = False
        for rows in grouped.values():
            rows.sort(
                key=lambda row: (
                    int(row.get("timestamp_unix") or 0),
                    str(row.get("transaction_hash") or ""),
                    str(row.get("trade_id") or ""),
                )
            )
            if len(rows) < trade_tokens:
                continue
            for offset in range(0, len(rows) - trade_tokens + 1, stride):
                batch = rows[offset : offset + trade_tokens]
                sequence, types, target = build_sequence(
                    batch,
                    trade_tokens=trade_tokens,
                    wallet_tokens=wallet_tokens,
                    context_tokens=context_tokens,
                    feature_width=feature_width,
                )
                features[written] = sequence
                token_types[written] = types
                targets[written] = target
                written += 1
                produced_from_file = True
                if written >= sequence_count:
                    break
            if written >= sequence_count:
                break
        if produced_from_file:
            used_paths.append(path.relative_to(data_dir).as_posix())
        if written >= sequence_count:
            break

    features.flush()
    token_types.flush()
    targets.flush()
    if written < sequence_count:
        raise RuntimeError(
            f"Packed only {written} of {sequence_count} registered sequences "
            f"from {files_read} files"
        )
    source_digest = hashlib.sha256("\n".join(used_paths).encode()).hexdigest()
    metadata = {
        "schema_version": "1.0.0",
        "research_id": config["research_id"],
        "config_id": config["id"],
        "created_at": utc_now(),
        "data_dir": str(data_dir.resolve()),
        "source_namespace": namespace,
        "source_listing_at": source_listing_at,
        "source_max_mtime_ns": max(path.stat().st_mtime_ns for path in paths),
        "source_paths_sha256": source_digest,
        "source_files_considered": len(paths),
        "source_files_read": files_read,
        "source_files_used": len(used_paths),
        "source_rows_read": rows_read,
        "sequence_count": written,
        "sequence_length": sequence_length,
        "feature_width": feature_width,
        "pack_seconds": time.perf_counter() - started,
        "features_bytes": feature_path.stat().st_size,
        "token_types_bytes": type_path.stat().st_size,
        "targets_bytes": target_path.stat().st_size,
        "evidence_boundary": config["evidence_boundary"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


class PackedChronicle(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, root: Path) -> None:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.root = root
        self.length = int(metadata["sequence_count"])
        self.features: NDArray[np.float16] | None = None
        self.token_types: NDArray[np.uint8] | None = None
        self.targets: NDArray[np.float16] | None = None

    def __len__(self) -> int:
        return self.length

    def _open(self) -> None:
        if self.features is None:
            self.features = np.load(self.root / "features.npy", mmap_mode="c")
            self.token_types = np.load(self.root / "token_types.npy", mmap_mode="c")
            self.targets = np.load(self.root / "targets.npy", mmap_mode="c")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        self._open()
        assert self.features is not None
        assert self.token_types is not None
        assert self.targets is not None
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.token_types[index].astype(np.int64)),
            torch.from_numpy(self.targets[index]),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["features"] = None
        state["token_types"] = None
        state["targets"] = None
        return state


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        mean_square = inputs.float().pow(2).mean(-1, keepdim=True)
        normalized = inputs * torch.rsqrt(mean_square + self.epsilon)
        return normalized.to(inputs.dtype) * self.weight


class S0Block(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        ffn_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Model width must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.attention_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_output = nn.Linear(width, width)
        self.ffn_norm = RMSNorm(width)
        self.gate_up = nn.Linear(width, ffn_width * 2)
        self.ffn_output = nn.Linear(ffn_width, width)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, width = inputs.shape
        normalized = self.attention_norm(inputs)
        qkv = self.qkv(normalized).view(
            batch,
            length,
            3,
            self.heads,
            self.head_width,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        inputs = inputs + self.attention_output(attended)
        gate, up = self.gate_up(self.ffn_norm(inputs)).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return cast(Tensor, inputs + self.ffn_output(hidden))


class SphinxTraceS0Benchmark(nn.Module):
    def __init__(self, config: dict[str, Any], sequence_length: int, feature_width: int) -> None:
        super().__init__()
        model = config["model"]
        width = int(model["width"])
        self.input_projection = nn.Sequential(
            nn.Linear(feature_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        self.token_type = nn.Embedding(3, width)
        self.blocks = nn.ModuleList(
            [
                S0Block(
                    width,
                    int(model["heads"]),
                    int(model["ffn_width"]),
                    float(model["dropout"]),
                )
                for _ in range(int(model["layers"]))
            ]
        )
        self.final_norm = RMSNorm(width)
        self.output = nn.Linear(width, int(model["output_width"]))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, features: Tensor, token_types: Tensor) -> Tensor:
        hidden = self.input_projection(features)
        hidden = hidden + self.position + self.token_type(token_types)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.final_norm(hidden).mean(dim=1)
        return cast(Tensor, self.output(pooled))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def run_benchmark(
    config: dict[str, Any],
    pack_dir: Path,
    output_path: Path,
    *,
    batch_size: int | None,
    measured_steps: int | None,
    warmup_steps: int | None,
    compile_mode: str | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SPH-T-H003")
    training = config["training"]
    metadata = json.loads((pack_dir / "metadata.json").read_text(encoding="utf-8"))
    selected_batch = batch_size or int(training["batch_size"])
    selected_steps = measured_steps or int(training["measured_steps"])
    selected_warmup = warmup_steps if warmup_steps is not None else int(training["warmup_steps"])
    requested_compile = compile_mode if compile_mode is not None else str(training["compile_mode"])
    device = torch.device("cuda")
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = PackedChronicle(pack_dir)
    workers = int(training["loader_workers"])
    loader = DataLoader(
        dataset,
        batch_size=selected_batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=bool(training["persistent_workers"]) and workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        drop_last=True,
    )
    model = SphinxTraceS0Benchmark(
        config,
        int(metadata["sequence_length"]),
        int(metadata["feature_width"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    minimum = int(config["model"]["parameter_minimum"])
    maximum = int(config["model"]["parameter_maximum"])
    if not minimum <= parameter_count <= maximum:
        raise RuntimeError(f"Parameter count {parameter_count} is outside {minimum}..{maximum}")

    compile_status = "disabled"
    compile_error: str | None = None
    compile_seconds = 0.0
    if requested_compile not in {"", "none", "disabled"}:
        compile_started = time.perf_counter()
        try:
            model = cast(
                SphinxTraceS0Benchmark,
                torch.compile(model, mode=requested_compile, dynamic=False),
            )
            compile_status = "requested"
        except Exception as exc:  # pragma: no cover - platform dependent
            compile_error = f"{type(exc).__name__}: {exc}"
            compile_status = "fallback_eager"
        compile_seconds = time.perf_counter() - compile_started

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            fused=True,
        )
        optimizer_mode = "fused_adamw"
    except RuntimeError as exc:  # pragma: no cover - platform dependent
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        optimizer_mode = f"foreach_adamw_fallback: {exc}"

    data_iterator = iter(loader)

    def next_batch() -> tuple[Tensor, Tensor, Tensor]:
        nonlocal data_iterator
        try:
            return cast(tuple[Tensor, Tensor, Tensor], next(data_iterator))
        except StopIteration:
            data_iterator = iter(loader)
            return cast(tuple[Tensor, Tensor, Tensor], next(data_iterator))

    def train_batch(batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        features, token_types, targets = batch
        features = features.to(device, dtype=torch.bfloat16, non_blocking=True)
        token_types = token_types.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(features, token_types)
            loss = F.mse_loss(predictions.float(), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return loss.detach()

    first_batch = next_batch()
    compile_run_started = time.perf_counter()
    try:
        train_batch(first_batch)
        torch.cuda.synchronize()
        if compile_status == "requested":
            compile_status = "active"
    except Exception as exc:
        if compile_status != "requested":
            raise
        compile_error = f"{type(exc).__name__}: {exc}"
        compile_status = "fallback_eager"
        del optimizer
        del model
        torch.cuda.empty_cache()
        model = SphinxTraceS0Benchmark(
            config,
            int(metadata["sequence_length"]),
            int(metadata["feature_width"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            fused=True,
        )
        optimizer_mode = "fused_adamw"
        train_batch(first_batch)
        torch.cuda.synchronize()
    compile_seconds += time.perf_counter() - compile_run_started

    for _ in range(max(0, selected_warmup - 1)):
        train_batch(next_batch())
    torch.cuda.synchronize()
    free_after_warmup, device_memory_total = torch.cuda.mem_get_info()
    allocated_after_warmup = torch.cuda.memory_allocated()
    reserved_after_warmup = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    step_seconds: list[float] = []
    data_seconds: list[float] = []
    losses: list[float] = []
    measured_started = time.perf_counter()
    for _ in range(selected_steps):
        data_started = time.perf_counter()
        batch = next_batch()
        data_seconds.append(time.perf_counter() - data_started)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        loss = train_batch(batch)
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.cpu()))
    measured_seconds_total = time.perf_counter() - measured_started
    free_after_measurement, _ = torch.cuda.mem_get_info()

    sequence_length = int(metadata["sequence_length"])
    tokens = selected_steps * selected_batch * sequence_length
    tokens_per_second = tokens / measured_seconds_total
    extrapolated = {
        str(int(token_count)): token_count / tokens_per_second / 3600.0
        for token_count in config["extrapolation_tokens"]
    }
    result = {
        "schema_version": "1.0.0",
        "research_id": config["research_id"],
        "config_id": config["id"],
        "completed_at": utc_now(),
        "evidence_boundary": config["evidence_boundary"],
        "source_pack": metadata,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "triton_windows": package_version("triton-windows"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]
        },
        "model": {
            "parameters": parameter_count,
            **config["model"],
        },
        "training": {
            "batch_size": selected_batch,
            "sequence_length": sequence_length,
            "warmup_steps": selected_warmup,
            "measured_steps": selected_steps,
            "measured_tokens": tokens,
            "precision": "bfloat16",
            "optimizer": optimizer_mode,
            "compile_requested": requested_compile,
            "compile_status": compile_status,
            "compile_error": compile_error,
            "compile_and_first_step_seconds": compile_seconds,
        },
        "measurement": {
            "wall_seconds": measured_seconds_total,
            "tokens_per_second": tokens_per_second,
            "sequences_per_second": selected_steps * selected_batch / measured_seconds_total,
            "step_seconds_mean": statistics.fmean(step_seconds),
            "step_seconds_p50": statistics.median(step_seconds),
            "step_seconds_p95": _percentile(step_seconds, 0.95),
            "data_wait_seconds_mean": statistics.fmean(data_seconds),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "device_memory_total_bytes": device_memory_total,
            "device_memory_used_after_warmup_bytes": (device_memory_total - free_after_warmup),
            "device_memory_used_after_measurement_bytes": (
                device_memory_total - free_after_measurement
            ),
            "allocated_after_warmup_bytes": allocated_after_warmup,
            "reserved_after_warmup_bytes": reserved_after_warmup,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "estimated_training_hours": extrapolated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = root.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--data-dir", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    benchmark = subparsers.add_parser("run")
    benchmark.add_argument("--pack-dir", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--steps", type=int)
    benchmark.add_argument("--warmup-steps", type=int)
    benchmark.add_argument("--compile-mode")
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "pack":
        result = pack_ledger(config, args.data_dir, args.output_dir)
    elif args.command == "run":
        result = run_benchmark(
            config,
            args.pack_dir,
            args.output,
            batch_size=args.batch_size,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_mode=args.compile_mode,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
