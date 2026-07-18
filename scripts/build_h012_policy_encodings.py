"""Precompute immutable H012 market encodings in large GPU batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.model_h013 import SphinxTraceS0H013
from sphinx_trace.policy_checkpoint import load_policy_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_h012_policy_encoding_cache_v1.json"
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
    ROOT / "src" / "sphinx_trace" / "model_h011.py",
    ROOT / "src" / "sphinx_trace" / "model_h012.py",
    ROOT / "src" / "sphinx_trace" / "model_h013.py",
    ROOT / "src" / "sphinx_trace" / "policy_checkpoint.py",
    ROOT / "src" / "sphinx_trace" / "policy_encodings.py",
)
DEVELOPMENT_SPLITS = {2: "validation", 3: "calibration"}


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


def _atomic_npy(path: Path, array: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _metadata(path: Path, root: Path, array: NDArray[Any]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _cached_receipt(
    receipt_path: Path,
    output_dir: Path,
    *,
    date: str,
    shard_index: int,
    rows: int,
    contract_sha256: str,
    source_receipt_sha256: str,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = _load_object(receipt_path)
    if (
        receipt.get("record_type") != "h012_policy_encoding_day_receipt"
        or receipt.get("date") != date
        or int(receipt.get("shard_index", -1)) != shard_index
        or int(receipt.get("rows", -1)) != rows
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("source_receipt_sha256") != source_receipt_sha256
    ):
        raise RuntimeError(f"H012 cached policy encoding receipt changed at {date}")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"H012 cached policy encoding files are invalid at {date}")
    for metadata in files.values():
        if not isinstance(metadata, dict):
            raise RuntimeError(f"H012 cached policy encoding file metadata changed at {date}")
        path = output_dir / str(metadata.get("path") or "")
        if (
            not path.is_file()
            or int(metadata.get("bytes", -1)) != path.stat().st_size
            or metadata.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"H012 cached policy encoding file changed at {date}")
    return receipt


def build_policy_encodings(
    cache_config_path: Path,
    model_config_path: Path,
    residual_config_path: Path,
    policy_config_path: Path,
    pack_dir: Path,
    outcome_dir: Path,
    policy_dir: Path,
    output_dir: Path,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if batch_size is not None and batch_size <= 0:
        raise ValueError("H012 policy encoding batch size must be positive")
    cache_config = load_json(cache_config_path)
    model_config = load_json(model_config_path)
    residual_config = load_json(residual_config_path)
    policy_config = load_json(policy_config_path)
    batch = batch_size or int(cache_config["build"]["batch_size"])
    feature_clip = float(cache_config["build"]["feature_clip_after_normalization"])
    pack_manifest_path = pack_dir / "manifest.json"
    policy_result_path = policy_dir / "result.json"
    pack_manifest = _load_object(pack_manifest_path)
    policy_result = _load_object(policy_result_path)
    if (
        pack_manifest.get("valid") is not True
        or pack_manifest.get("test_labels_opened") is not False
        or int(pack_manifest.get("test_label_rows", -1)) != 0
        or policy_result.get("valid") is not True
        or policy_result.get("test_labels_opened") is not False
        or int(policy_result.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H012 policy encoding cache requires closed development sources")
    expected_configs = {
        "model_config_sha256": sha256_file(model_config_path),
        "residual_config_sha256": sha256_file(residual_config_path),
        "policy_config_sha256": sha256_file(policy_config_path),
    }
    if any(policy_result.get(key) != value for key, value in expected_configs.items()):
        raise RuntimeError("H012 policy encoding configs no longer match the selected policy")
    pack_sha256 = sha256_file(pack_manifest_path)
    policy_sha256 = sha256_file(policy_result_path)
    implementation_sha256 = _implementation_digest()
    contract_payload = (
        f"cache_config:{sha256_file(cache_config_path)}\n"
        f"pack:{pack_sha256}\npolicy:{policy_sha256}\n"
        f"implementation:{implementation_sha256}\n"
    )
    contract_sha256 = hashlib.sha256(contract_payload.encode()).hexdigest()
    existing_manifest_path = output_dir / "manifest.json"
    if existing_manifest_path.exists():
        existing = _load_object(existing_manifest_path)
        if existing.get("contract_sha256") != contract_sha256:
            raise RuntimeError("H012 policy encoding output belongs to another contract")
        return existing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_policy_checkpoint(
        policy_dir,
        outcome_dir,
        model_config,
        residual_config,
        policy_config,
        device,
    )
    model = checkpoint.model.eval()
    backbone = model.outcome_backbone
    width = model.width
    with np.load(pack_dir / "normalization.npz", allow_pickle=False) as archive:
        median = np.asarray(archive["median"], dtype=np.float32)
        scale = np.asarray(archive["scale"], dtype=np.float32)
    if median.shape != (128,) or scale.shape != (128,) or bool((scale <= 0.0).any()):
        raise RuntimeError("H012 policy encoding normalization is invalid")
    feature_mask = checkpoint.feature_mask.detach().cpu().numpy().astype(np.float32)
    group_mask = checkpoint.group_mask.to(device)
    source_shards = tuple(
        path for path in sorted((pack_dir / "shards").glob("date=*")) if path.is_dir()
    )
    if len(source_shards) != int(pack_manifest.get("days", -1)):
        raise RuntimeError("H012 policy encoding feature shard coverage changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir = output_dir / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    split_totals: Counter[str] = Counter()
    receipt_rows: list[dict[str, Any]] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for shard_index, source_shard in enumerate(source_shards):
        date = source_shard.name.removeprefix("date=")
        split_codes = np.load(source_shard / "split_codes.npy", mmap_mode="r")
        label_mask = np.load(source_shard / "label_mask.npy", mmap_mode="r")
        if split_codes.shape != label_mask.shape:
            raise RuntimeError(f"H012 policy encoding source arrays misalign at {date}")
        selected_mask = np.isin(split_codes, tuple(DEVELOPMENT_SPLITS)) & label_mask.astype(bool)
        row_indices = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
        rows = len(row_indices)
        selected_codes = np.asarray(split_codes[row_indices], dtype=np.uint8)
        day_counts = Counter(
            {
                name: count
                for code, name in DEVELOPMENT_SPLITS.items()
                if (count := int(np.count_nonzero(selected_codes == code))) > 0
            }
        )
        split_totals.update(day_counts)
        source_receipt_sha256 = sha256_file(pack_dir / "receipts" / f"date={date}.json")
        receipt_path = receipts_dir / f"date={date}.json"
        cached = _cached_receipt(
            receipt_path,
            output_dir,
            date=date,
            shard_index=shard_index,
            rows=rows,
            contract_sha256=contract_sha256,
            source_receipt_sha256=source_receipt_sha256,
        )
        if cached is not None:
            receipt_rows.append(cached)
            print(json.dumps({"date": date, "rows": rows, "status": "cached"}), flush=True)
            continue
        features = np.load(source_shard / "features.npy", mmap_mode="r")
        baselines = np.load(source_shard / "baselines.npy", mmap_mode="r")
        if features.shape != (len(split_codes), 128) or baselines.shape != split_codes.shape:
            raise RuntimeError(f"H012 policy encoding source features misalign at {date}")
        market_latents = np.empty((rows, width), dtype=np.float32)
        terminal_logits = np.empty(rows, dtype=np.float32)
        uncertainty_scales = np.empty(rows, dtype=np.float32)
        for offset in range(0, rows, batch):
            stop = min(rows, offset + batch)
            source_rows = row_indices[offset:stop]
            normalized = np.asarray(features[source_rows], dtype=np.float32)
            normalized = (normalized - median) / scale
            np.clip(normalized, -feature_clip, feature_clip, out=normalized)
            normalized *= feature_mask
            feature_tensor = torch.from_numpy(normalized).to(device)
            group_tensor = group_mask.unsqueeze(0).expand(stop - offset, -1)
            with torch.inference_mode(), autocast:
                if isinstance(backbone, SphinxTraceS0H013):
                    market_tensor = torch.from_numpy(
                        np.asarray(baselines[source_rows], dtype=np.float32)
                    ).to(device)
                    encoded = backbone(
                        feature_tensor,
                        market_tensor,
                        group_mask=group_tensor,
                        return_latent=True,
                    )
                else:
                    encoded = backbone(
                        feature_tensor,
                        group_mask=group_tensor,
                        return_latent=True,
                    )
            market_latents[offset:stop] = (
                encoded["debug_latent_state"].float().cpu().numpy()
            )
            terminal_logits[offset:stop] = (
                encoded["terminal_outcome_logit"].float().cpu().numpy()
            )
            uncertainty_scales[offset:stop] = (
                encoded["uncertainty_log_scale"].float().cpu().numpy()
            )
        if (
            not bool(np.isfinite(market_latents).all())
            or not bool(np.isfinite(terminal_logits).all())
            or not bool(np.isfinite(uncertainty_scales).all())
        ):
            raise RuntimeError(f"H012 policy encoding produced non-finite values at {date}")
        shard_dir = output_dir / "shards" / f"date={date}"
        arrays: dict[str, NDArray[Any]] = {
            "row_indices.npy": row_indices,
            "market_latents.npy": market_latents,
            "terminal_logits.npy": terminal_logits,
            "uncertainty_log_scales.npy": uncertainty_scales,
        }
        for name, array in arrays.items():
            _atomic_npy(shard_dir / name, array)
        files = {
            name: _metadata(shard_dir / name, output_dir, array)
            for name, array in arrays.items()
        }
        receipt: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_type": "h012_policy_encoding_day_receipt",
            "generated_at": now_utc(),
            "date": date,
            "shard_index": shard_index,
            "rows": rows,
            "split_rows": dict(sorted(day_counts.items())),
            "contract_sha256": contract_sha256,
            "pack_manifest_sha256": pack_sha256,
            "policy_result_sha256": policy_sha256,
            "source_receipt_sha256": source_receipt_sha256,
            "implementation_sha256": implementation_sha256,
            "files": files,
            "test_rows_consumed": 0,
            "test_labels_opened": False,
        }
        atomic_json(receipt_path, receipt)
        receipt_rows.append(receipt)
        atomic_json(
            output_dir / "progress.json",
            {
                "contract_sha256": contract_sha256,
                "completed_shards": len(receipt_rows),
                "shards": len(source_shards),
                "rows": sum(int(row["rows"]) for row in receipt_rows),
                "updated_at": now_utc(),
            },
        )
        print(json.dumps({"date": date, "rows": rows, "status": "built"}), flush=True)
        if (output_dir / "PAUSE").exists():
            return {
                "status": "paused",
                "completed_shards": len(receipt_rows),
                "shards": len(source_shards),
            }
    receipt_digest = hashlib.sha256()
    manifest_shards: list[dict[str, Any]] = []
    total_rows = 0
    for shard_index, receipt in enumerate(receipt_rows):
        date = str(receipt["date"])
        receipt_path = receipts_dir / f"date={date}.json"
        receipt_sha256 = sha256_file(receipt_path)
        receipt_digest.update(f"{date}:{receipt_sha256}\n".encode())
        rows = int(receipt["rows"])
        total_rows += rows
        manifest_shards.append(
            {
                "shard_index": shard_index,
                "date": date,
                "rows": rows,
                "receipt_path": receipt_path.relative_to(output_dir).as_posix(),
                "receipt_sha256": receipt_sha256,
                "files": receipt["files"],
            }
        )
    expected_rows = (
        int(policy_result["fit"]["rows"])
        + int(policy_result["selection"]["rows"])
        + int(policy_result["calibration"]["rows"])
    )
    valid = (
        len(manifest_shards) == len(source_shards)
        and total_rows == expected_rows
        and split_totals["validation"] == int(policy_result["selection"]["rows"]) + int(
            policy_result["fit"]["rows"]
        )
        and split_totals["calibration"] == int(policy_result["calibration"]["rows"])
    )
    # The validation cache covers both the early fit block and late policy-selection block.
    expected_total = split_totals["validation"] + split_totals["calibration"]
    valid = valid and total_rows == expected_total
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h012_policy_encoding_cache_manifest",
        "generated_at": now_utc(),
        "research_id": str(cache_config["research_id"]),
        "valid": valid,
        "contract_sha256": contract_sha256,
        "cache_config_sha256": sha256_file(cache_config_path),
        "implementation_sha256": implementation_sha256,
        "pack_manifest_sha256": pack_sha256,
        "policy_result_sha256": policy_sha256,
        "policy_checkpoint_sha256": str(policy_result["best_model_sha256"]),
        "latent_width": width,
        "latent_dtype": "float32",
        "head_dtype": "float32",
        "rows": total_rows,
        "split_rows": dict(sorted(split_totals.items())),
        "days": len(manifest_shards),
        "daily_receipts_sha256": receipt_digest.hexdigest(),
        "shards": manifest_shards,
        "device": str(device),
        "batch_size": batch,
        "elapsed_seconds": time.perf_counter() - started,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": str(cache_config["evidence_boundary"]),
    }
    atomic_json(existing_manifest_path, manifest)
    if not valid:
        raise RuntimeError("H012 policy encoding cache failed acceptance")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--cache-config", type=Path, default=DEFAULT_CACHE_CONFIG)
    value.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    value.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    value.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    value.add_argument("--pack-dir", type=Path, required=True)
    value.add_argument("--outcome-dir", type=Path, required=True)
    value.add_argument("--policy-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--batch-size", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    result = build_policy_encodings(
        args.cache_config.resolve(),
        args.model_config.resolve(),
        args.residual_config.resolve(),
        args.policy_config.resolve(),
        args.pack_dir.resolve(),
        args.outcome_dir.resolve(),
        args.policy_dir.resolve(),
        args.output_dir.resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
