"""Build the source-bound H014 policy-state pack from one exact H010 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.policy_training import ComponentTimePartition, component_time_partition
from sphinx_trace.replay_state_pack import (
    STATE_ARRAY_NAMES,
    array_metadata,
    atomic_numpy,
    extract_replay_state_arrays,
    validate_state_shard,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "trace" / "sphinx_trace_s0_h014_replay_state_distillation_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "replay_state_pack.py",
    ROOT / "src" / "sphinx_trace" / "policy_training.py",
    ROOT / "src" / "sphinx_corpus" / "io.py",
)


@dataclass(frozen=True, slots=True)
class SourceShard:
    date: str
    pack: Path
    encoding: Path
    audit: Path


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


def _partition_digest(partition: ComponentTimePartition) -> str:
    digest = hashlib.sha256()
    digest.update(partition.fit_components.tobytes())
    digest.update(partition.selection_components.tobytes())
    digest.update(str(partition.cutoff_unix).encode())
    return digest.hexdigest()


def _source_shards(pack_dir: Path, encoding_dir: Path, replay_dir: Path) -> list[SourceShard]:
    output: list[SourceShard] = []
    for pack in sorted((pack_dir / "shards").glob("date=*")):
        if not pack.is_dir():
            continue
        date = pack.name.removeprefix("date=")
        encoding = encoding_dir / "shards" / pack.name
        audit = replay_dir / "shards" / f"date={date}.jsonl.zst"
        if not encoding.is_dir() or not audit.is_file():
            raise RuntimeError(f"H014 source shard is incomplete: {date}")
        output.append(SourceShard(date, pack, encoding, audit))
    if not output:
        raise RuntimeError("H014 found no source shards")
    return output


def _validation_rows(shard: SourceShard) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    encoding_rows = np.load(
        shard.encoding / "row_indices.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(shard.pack / "split_codes.npy", mmap_mode="r", allow_pickle=False)
    label_mask = np.load(shard.pack / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    if encoding_rows.ndim != 1 or bool((encoding_rows < 0).any()) or bool(
        (encoding_rows >= len(split)).any()
    ):
        raise RuntimeError(f"H014 encoding row indices are invalid: {shard.date}")
    selected = (split[encoding_rows] == 2) & (label_mask[encoding_rows] == 1)
    return (
        np.asarray(encoding_rows[selected], dtype=np.int64),
        np.flatnonzero(selected).astype(np.int64),
    )


def _partition(shards: list[SourceShard], fit_fraction: float) -> ComponentTimePartition:
    components: list[NDArray[np.int64]] = []
    timestamps: list[NDArray[np.int64]] = []
    for shard in shards:
        rows, _ = _validation_rows(shard)
        if not len(rows):
            continue
        component = np.load(
            shard.pack / "component_ids.npy", mmap_mode="r", allow_pickle=False
        )
        timestamp = np.load(shard.pack / "timestamps.npy", mmap_mode="r", allow_pickle=False)
        components.append(np.asarray(component[rows], dtype=np.int64))
        timestamps.append(np.asarray(timestamp[rows], dtype=np.int64))
    if not components:
        raise RuntimeError("H014 found no qualified validation rows")
    return component_time_partition(
        np.concatenate(components),
        np.concatenate(timestamps),
        fit_fraction,
    )


def _contract(
    config_path: Path,
    pack_dir: Path,
    encoding_dir: Path,
    replay_dir: Path,
    implementation_sha256: str,
    partition_sha256: str,
) -> tuple[str, dict[str, str]]:
    sources = {
        "config_sha256": sha256_file(config_path),
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "encoding_manifest_sha256": sha256_file(encoding_dir / "manifest.json"),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "replay_profit_evaluation_sha256": sha256_file(
            replay_dir / "profit-evaluation.json"
        ),
        "replay_audit_manifest_sha256": sha256_file(replay_dir / "manifest.json"),
        "implementation_sha256": implementation_sha256,
        "partition_sha256": partition_sha256,
    }
    payload = "".join(f"{key}:{value}\n" for key, value in sorted(sources.items()))
    return hashlib.sha256(payload.encode()).hexdigest(), sources


def _verify_sources(config: dict[str, Any], sources: dict[str, str], replay_dir: Path) -> None:
    dependencies = config["dependencies"]
    if dependencies["feature_pack"]["manifest_sha256"] != sources["pack_manifest_sha256"]:
        raise RuntimeError("H014 feature-pack manifest changed after registration")
    if (
        dependencies["market_encoding_cache"]["manifest_sha256"]
        != sources["encoding_manifest_sha256"]
    ):
        raise RuntimeError("H014 encoding manifest changed after registration")
    teacher = dependencies["teacher_replay"]
    if teacher["audit_manifest_sha256"] != sources["replay_audit_manifest_sha256"]:
        raise RuntimeError("H014 teacher audit manifest changed after registration")
    trigger = config["trigger"]
    if trigger["result_sha256"] != sources["replay_result_sha256"]:
        raise RuntimeError("H014 teacher replay result changed after registration")
    if trigger["profit_evaluation_sha256"] != sources["replay_profit_evaluation_sha256"]:
        raise RuntimeError("H014 teacher profit evaluation changed after registration")
    replay = _load_object(replay_dir / "result.json")
    if (
        replay.get("valid") is not True
        or replay.get("split") != "validation"
        or replay.get("test_labels_opened") is not False
        or int(replay.get("test_rows_consumed", -1)) != 0
    ):
        raise RuntimeError("H014 requires a valid closed-test validation replay")


def _receipt_valid(
    receipt_path: Path,
    shard_dir: Path,
    *,
    contract_sha256: str,
    expected_rows: int,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = _load_object(receipt_path)
    if (
        receipt.get("contract_sha256") != contract_sha256
        or int(receipt.get("rows", -1)) != expected_rows
    ):
        raise RuntimeError(f"H014 existing receipt belongs to another contract: {receipt_path}")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"H014 existing receipt has no file manifest: {receipt_path}")
    validate_state_shard(shard_dir, files, expected_rows=expected_rows)
    return receipt


def _build_day(
    shard: SourceShard,
    output_dir: Path,
    partition: ComponentTimePartition,
    contract_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    rows, encoding_offsets = _validation_rows(shard)
    output_shard = output_dir / "shards" / f"date={shard.date}"
    receipt_path = output_dir / "receipts" / f"date={shard.date}.json"
    reused = _receipt_valid(
        receipt_path,
        output_shard,
        contract_sha256=contract_sha256,
        expected_rows=len(rows),
    )
    if reused is not None:
        return reused
    components_source = np.load(
        shard.pack / "component_ids.npy", mmap_mode="r", allow_pickle=False
    )
    timestamps_source = np.load(
        shard.pack / "timestamps.npy", mmap_mode="r", allow_pickle=False
    )
    labels_source = np.load(shard.pack / "labels.npy", mmap_mode="r", allow_pickle=False)
    baselines_source = np.load(
        shard.pack / "baselines.npy", mmap_mode="r", allow_pickle=False
    )
    components = np.asarray(components_source[rows], dtype=np.int64)
    timestamps = np.asarray(timestamps_source[rows], dtype=np.int64)
    labels = np.asarray(labels_source[rows], dtype=np.float32)
    baselines = np.asarray(baselines_source[rows], dtype=np.float32)
    if not bool(np.isin(labels, (0.0, 1.0)).all()):
        raise RuntimeError(f"H014 source labels are invalid: {shard.date}")
    if not bool(np.isfinite(baselines).all()) or bool(
        ((baselines < 0.0) | (baselines > 1.0)).any()
    ):
        raise RuntimeError(f"H014 source market probabilities are invalid: {shard.date}")
    state = extract_replay_state_arrays(
        iter_jsonl_zst(shard.audit),
        date=shard.date,
        expected_row_indices=rows,
        expected_timestamps=timestamps,
    )
    in_fit = np.isin(components, partition.fit_components, assume_unique=False)
    in_selection = np.isin(components, partition.selection_components, assume_unique=False)
    if not bool((in_fit ^ in_selection).all()):
        raise RuntimeError(f"H014 component partition does not cover {shard.date}")
    partition_codes = np.where(in_fit, 0, 1).astype(np.uint8)
    arrays: dict[str, NDArray[Any]] = {
        "row_indices.npy": state.row_indices,
        "encoding_offsets.npy": encoding_offsets,
        "component_ids.npy": components,
        "timestamps.npy": timestamps,
        "partition_codes.npy": partition_codes,
        "portfolio_features.npy": state.portfolio_features,
        "prediction_memory_features.npy": state.prediction_memory_features,
        "previous_action_ids.npy": state.previous_action_ids,
        "physical_action_masks.npy": state.physical_action_masks,
    }
    for name, values in arrays.items():
        atomic_numpy(output_shard / name, values)
    files = {name: array_metadata(output_shard / name, output_dir) for name in STATE_ARRAY_NAMES}
    receipt = {
        "schema_version": "1.0.0",
        "record_type": "h014_replay_state_day_receipt",
        "research_id": "SPH-T-H014",
        "generated_at": now_utc(),
        "date": shard.date,
        "rows": len(rows),
        "fit_rows": int(in_fit.sum()),
        "selection_rows": int(in_selection.sum()),
        "contract_sha256": contract_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_pack_receipt_sha256": sha256_file(
            shard.pack.parents[1] / "receipts" / f"date={shard.date}.json"
        ),
        "encoding_receipt_sha256": sha256_file(
            shard.encoding.parents[1] / "receipts" / f"date={shard.date}.json"
        ),
        "audit_shard_sha256": sha256_file(shard.audit),
        "files": files,
        "teacher_action_stored": False,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build(
    config_path: Path,
    pack_dir: Path,
    encoding_dir: Path,
    replay_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    shards = _source_shards(pack_dir, encoding_dir, replay_dir)
    partition = _partition(shards, float(config["partition"]["fit_component_fraction"]))
    partition_sha256 = _partition_digest(partition)
    implementation_sha256 = _implementation_digest()
    contract_sha256, sources = _contract(
        config_path,
        pack_dir,
        encoding_dir,
        replay_dir,
        implementation_sha256,
        partition_sha256,
    )
    _verify_sources(config, sources, replay_dir)
    receipts: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        receipt = _build_day(
            shard,
            output_dir,
            partition,
            contract_sha256,
            implementation_sha256,
        )
        receipts.append(receipt)
        atomic_json(
            output_dir / "progress.json",
            {
                "record_type": "h014_replay_state_build_progress",
                "contract_sha256": contract_sha256,
                "days_complete": index + 1,
                "days_total": len(shards),
                "rows_complete": sum(int(item["rows"]) for item in receipts),
                "updated_at": now_utc(),
            },
        )
        if (output_dir / "PAUSE").exists():
            return {
                "status": "paused",
                "days_complete": index + 1,
                "rows_complete": sum(int(item["rows"]) for item in receipts),
                "contract_sha256": contract_sha256,
            }
    rows = sum(int(receipt["rows"]) for receipt in receipts)
    fit_rows = sum(int(receipt["fit_rows"]) for receipt in receipts)
    selection_rows = sum(int(receipt["selection_rows"]) for receipt in receipts)
    expected = config["partition"]
    if (
        rows != int(config["corpus"]["rows"])
        or fit_rows != int(expected["fit_rows_expected"])
        or selection_rows != int(expected["selection_rows_expected"])
    ):
        raise RuntimeError(
            "H014 state-pack row counts changed: "
            f"rows={rows}, fit={fit_rows}, selection={selection_rows}"
        )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "h014_replay_state_pack_manifest",
        "research_id": "SPH-T-H014",
        "dataset_id": config["corpus"]["id"],
        "generated_at": now_utc(),
        "valid": True,
        "days": len(shards),
        "rows": rows,
        "fit_rows": fit_rows,
        "selection_rows": selection_rows,
        "fit_components": len(partition.fit_components),
        "selection_components": len(partition.selection_components),
        "cutoff_unix": partition.cutoff_unix,
        "partition_sha256": partition_sha256,
        "contract_sha256": contract_sha256,
        **sources,
        "shards": [
            {
                "date": receipt["date"],
                "rows": receipt["rows"],
                "fit_rows": receipt["fit_rows"],
                "selection_rows": receipt["selection_rows"],
                "receipt_path": f"receipts/date={receipt['date']}.json",
                "receipt_sha256": sha256_file(
                    output_dir / "receipts" / f"date={receipt['date']}.json"
                ),
            }
            for receipt in receipts
        ],
        "teacher_action_stored": False,
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": config["evidence_boundary"],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--encoding-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.config.resolve(),
        args.pack_dir.resolve(),
        args.encoding_dir.resolve(),
        args.replay_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
