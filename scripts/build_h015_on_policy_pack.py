"""Build the H015 aggregated exact-replay portfolio-advantage corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.development_tape import load_tape_conditions
from sphinx_trace.on_policy_pack import (
    H015_ARRAY_NAMES,
    LoggedExecutionIndex,
    aligned_execution_arrays,
    build_logged_execution_index,
    build_payout_map,
    validate_on_policy_shard,
)
from sphinx_trace.policy_training import ComponentTimePartition, component_time_partition
from sphinx_trace.replay_state_pack import (
    array_metadata,
    atomic_numpy,
    extract_replay_state_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "trace"
    / "sphinx_trace_s0_h015_on_policy_portfolio_advantage_v1.json"
)
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "on_policy_pack.py",
    ROOT / "src" / "sphinx_trace" / "replay_state_pack.py",
    ROOT / "src" / "sphinx_trace" / "policy_training.py",
    ROOT / "src" / "sphinx_trace" / "development_tape.py",
    ROOT / "src" / "sphinx_corpus" / "io.py",
)


@dataclass(frozen=True, slots=True)
class SourceShard:
    date: str
    pack: Path
    encoding: Path


@dataclass(frozen=True, slots=True)
class BehaviorReplay:
    behavior_id: str
    code: int
    replay_dir: Path
    registered_result_sha256: str
    registered_audit_manifest_sha256: str


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


def _source_shards(pack_dir: Path, encoding_dir: Path) -> list[SourceShard]:
    output: list[SourceShard] = []
    for pack in sorted((pack_dir / "shards").glob("date=*")):
        if not pack.is_dir():
            continue
        date = pack.name.removeprefix("date=")
        encoding = encoding_dir / "shards" / pack.name
        if not encoding.is_dir():
            raise RuntimeError(f"H015 encoding shard is missing: {date}")
        output.append(SourceShard(date, pack, encoding))
    if not output:
        raise RuntimeError("H015 found no source feature shards")
    return output


def _validation_rows(shard: SourceShard) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    encoding_rows = np.load(
        shard.encoding / "row_indices.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(shard.pack / "split_codes.npy", mmap_mode="r", allow_pickle=False)
    label_mask = np.load(
        shard.pack / "label_mask.npy", mmap_mode="r", allow_pickle=False
    )
    if encoding_rows.ndim != 1 or bool((encoding_rows < 0).any()) or bool(
        (encoding_rows >= len(split)).any()
    ):
        raise RuntimeError(f"H015 encoding row indices are invalid: {shard.date}")
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
        timestamp = np.load(
            shard.pack / "timestamps.npy", mmap_mode="r", allow_pickle=False
        )
        components.append(np.asarray(component[rows], dtype=np.int64))
        timestamps.append(np.asarray(timestamp[rows], dtype=np.int64))
    if not components:
        raise RuntimeError("H015 found no qualified validation rows")
    return component_time_partition(
        np.concatenate(components), np.concatenate(timestamps), fit_fraction
    )


def _behavior_replays(
    config: dict[str, Any], h012_replay_dir: Path, h014_replay_dir: Path
) -> list[BehaviorReplay]:
    registered = config["dependencies"]["behavior_replays"]
    if not isinstance(registered, list) or len(registered) != 2:
        raise RuntimeError("H015 requires exactly two pre-registered behavior replays")
    directories = [h012_replay_dir, h014_replay_dir]
    output: list[BehaviorReplay] = []
    for code, (raw, directory) in enumerate(zip(registered, directories, strict=True)):
        if not isinstance(raw, dict):
            raise RuntimeError("H015 behavior replay registration is invalid")
        output.append(
            BehaviorReplay(
                behavior_id=str(raw["id"]),
                code=code,
                replay_dir=directory,
                registered_result_sha256=str(raw["result_sha256"]),
                registered_audit_manifest_sha256=str(raw["audit_manifest_sha256"]),
            )
        )
    return output


def _verify_behavior(behavior: BehaviorReplay, expected_rows: int) -> dict[str, Any]:
    result_path = behavior.replay_dir / "result.json"
    manifest_path = behavior.replay_dir / "manifest.json"
    result = _load_object(result_path)
    result_sha256 = sha256_file(result_path)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        result_sha256 != behavior.registered_result_sha256
        or manifest_sha256 != behavior.registered_audit_manifest_sha256
        or result.get("valid") is not True
        or result.get("split") != "validation"
        or result.get("test_labels_opened") is not False
        or int(result.get("test_rows_consumed", -1)) != 0
        or int(result.get("metrics", {}).get("predictions", -1)) != expected_rows
    ):
        raise RuntimeError(f"H015 behavior replay contract changed: {behavior.behavior_id}")
    manifest = _load_object(manifest_path)
    if manifest.get("valid") is not True or int(manifest.get("days", -1)) <= 0:
        raise RuntimeError(f"H015 behavior audit is invalid: {behavior.behavior_id}")
    return result


def _replay_records(replay_dir: Path) -> Iterator[dict[str, Any]]:
    shards = sorted((replay_dir / "shards").glob("date=*.jsonl.zst"))
    if not shards:
        raise RuntimeError(f"H015 replay contains no audit shards: {replay_dir}")
    for shard in shards:
        yield from iter_jsonl_zst(shard)


def _contract(
    config_path: Path,
    pack_dir: Path,
    encoding_dir: Path,
    tape_dir: Path,
    behaviors: list[BehaviorReplay],
    implementation_sha256: str,
    partition_sha256: str,
) -> tuple[str, dict[str, str]]:
    sources = {
        "config_sha256": sha256_file(config_path),
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "encoding_manifest_sha256": sha256_file(encoding_dir / "manifest.json"),
        "tape_manifest_sha256": sha256_file(tape_dir / "manifest.json"),
        "tape_conditions_receipt_sha256": sha256_file(tape_dir / "conditions.json"),
        "implementation_sha256": implementation_sha256,
        "partition_sha256": partition_sha256,
    }
    for behavior in behaviors:
        prefix = f"behavior_{behavior.code}"
        sources[f"{prefix}_result_sha256"] = sha256_file(
            behavior.replay_dir / "result.json"
        )
        sources[f"{prefix}_audit_manifest_sha256"] = sha256_file(
            behavior.replay_dir / "manifest.json"
        )
    payload = "".join(f"{key}:{value}\n" for key, value in sorted(sources.items()))
    return hashlib.sha256(payload.encode()).hexdigest(), sources


def _verify_registered_sources(config: dict[str, Any], sources: dict[str, str]) -> None:
    dependencies = config["dependencies"]
    expected = {
        "feature_pack_manifest_sha256": sources["pack_manifest_sha256"],
        "encoding_manifest_sha256": sources["encoding_manifest_sha256"],
        "development_tape_manifest_sha256": sources["tape_manifest_sha256"],
        "development_conditions_receipt_sha256": sources[
            "tape_conditions_receipt_sha256"
        ],
    }
    for key, actual in expected.items():
        if dependencies.get(key) != actual:
            raise RuntimeError(f"H015 registered dependency changed: {key}")


def _receipt_valid(
    receipt_path: Path,
    shard_dir: Path,
    *,
    contract_sha256: str,
    expected_rows: int,
    expected_behavior_code: int,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = _load_object(receipt_path)
    if (
        receipt.get("contract_sha256") != contract_sha256
        or int(receipt.get("rows", -1)) != expected_rows
        or int(receipt.get("behavior_policy_code", -1)) != expected_behavior_code
    ):
        raise RuntimeError(f"H015 existing receipt belongs to another contract: {receipt_path}")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"H015 existing receipt has no file manifest: {receipt_path}")
    validate_on_policy_shard(
        shard_dir,
        files,
        expected_rows=expected_rows,
        expected_behavior_code=expected_behavior_code,
    )
    return receipt


def _build_day(
    shard: SourceShard,
    behavior: BehaviorReplay,
    index: LoggedExecutionIndex,
    output_dir: Path,
    partition: ComponentTimePartition,
    contract_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    rows, encoding_offsets = _validation_rows(shard)
    behavior_name = f"behavior={behavior.behavior_id}"
    output_shard = output_dir / "shards" / behavior_name / f"date={shard.date}"
    receipt_path = output_dir / "receipts" / behavior_name / f"date={shard.date}.json"
    reused = _receipt_valid(
        receipt_path,
        output_shard,
        contract_sha256=contract_sha256,
        expected_rows=len(rows),
        expected_behavior_code=behavior.code,
    )
    if reused is not None:
        return reused
    components_source = np.load(
        shard.pack / "component_ids.npy", mmap_mode="r", allow_pickle=False
    )
    markets_source = np.load(
        shard.pack / "market_ids.npy", mmap_mode="r", allow_pickle=False
    )
    timestamps_source = np.load(
        shard.pack / "timestamps.npy", mmap_mode="r", allow_pickle=False
    )
    components = np.asarray(components_source[rows], dtype=np.int64)
    markets = np.asarray(markets_source[rows], dtype=np.int64)
    timestamps = np.asarray(timestamps_source[rows], dtype=np.int64)
    audit = behavior.replay_dir / "shards" / f"date={shard.date}.jsonl.zst"
    if not audit.is_file():
        raise RuntimeError(f"H015 behavior shard is missing: {behavior.behavior_id}:{shard.date}")
    state = extract_replay_state_arrays(
        iter_jsonl_zst(audit),
        date=shard.date,
        expected_row_indices=rows,
        expected_timestamps=timestamps,
    )
    in_fit = np.isin(components, partition.fit_components, assume_unique=False)
    in_selection = np.isin(
        components, partition.selection_components, assume_unique=False
    )
    if not bool((in_fit ^ in_selection).all()):
        raise RuntimeError(f"H015 component partition does not cover {shard.date}")
    arrays: dict[str, NDArray[Any]] = {
        "row_indices.npy": state.row_indices,
        "encoding_offsets.npy": encoding_offsets,
        "component_ids.npy": components,
        "market_ids.npy": markets,
        "timestamps.npy": timestamps,
        "partition_codes.npy": np.where(in_fit, 0, 1).astype(np.uint8),
        "portfolio_features.npy": state.portfolio_features,
        "prediction_memory_features.npy": state.prediction_memory_features,
        "previous_action_ids.npy": state.previous_action_ids,
        "physical_action_masks.npy": state.physical_action_masks,
        "behavior_policy_codes.npy": np.full(len(rows), behavior.code, dtype=np.uint8),
        **aligned_execution_arrays(index, date=shard.date, expected_row_indices=rows),
    }
    for name, values in arrays.items():
        atomic_numpy(output_shard / name, values)
    files = {
        name: array_metadata(output_shard / name, output_dir)
        for name in H015_ARRAY_NAMES
    }
    receipt = {
        "schema_version": "1.0.0",
        "record_type": "h015_on_policy_portfolio_advantage_day_receipt",
        "research_id": "SPH-T-H015",
        "generated_at": now_utc(),
        "behavior_policy_id": behavior.behavior_id,
        "behavior_policy_code": behavior.code,
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
        "audit_shard_sha256": sha256_file(audit),
        "files": files,
        "teacher_action_used_as_imitation_label": False,
        "calibration_rows_consumed": 0,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build(
    config_path: Path,
    pack_dir: Path,
    encoding_dir: Path,
    tape_dir: Path,
    h012_replay_dir: Path,
    h014_replay_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    shards = _source_shards(pack_dir, encoding_dir)
    expected_rows_per_behavior = int(config["corpus"]["rows_expected"]) // 2
    partition = _partition(shards, float(config["partition"]["fit_component_fraction"]))
    partition_sha256 = _partition_digest(partition)
    if partition_sha256 != config["partition"]["partition_sha256"]:
        raise RuntimeError("H015 whole-component partition changed after registration")
    behaviors = _behavior_replays(config, h012_replay_dir, h014_replay_dir)
    behavior_results = {
        behavior.behavior_id: _verify_behavior(behavior, expected_rows_per_behavior)
        for behavior in behaviors
    }
    implementation_sha256 = _implementation_digest()
    contract_sha256, sources = _contract(
        config_path,
        pack_dir,
        encoding_dir,
        tape_dir,
        behaviors,
        implementation_sha256,
        partition_sha256,
    )
    _verify_registered_sources(config, sources)
    catalog = load_tape_conditions(tape_dir, "validation")
    payout_map = build_payout_map(catalog.contracts, catalog.resolutions)
    receipts: list[dict[str, Any]] = []
    behavior_summaries: list[dict[str, Any]] = []
    total_steps = len(behaviors) * len(shards)
    completed_steps = 0
    for behavior in behaviors:
        index = build_logged_execution_index(
            _replay_records(behavior.replay_dir),
            payout_map,
            reference_size=float(config["training"]["action_value_reference_size"]),
        )
        replay_result = behavior_results[behavior.behavior_id]
        metrics = replay_result["metrics"]
        if (
            len(index.targets) != expected_rows_per_behavior
            or index.action_counts != replay_result["actions"]
            or index.orders != int(metrics["orders"])
            or index.fills != int(metrics["fills"])
            or abs(float(index.realized_pnl_usd) - float(metrics["net_profit_usd"])) > 1e-8
        ):
            raise RuntimeError(
                f"H015 logged execution attribution changed: {behavior.behavior_id}"
            )
        behavior_receipts: list[dict[str, Any]] = []
        for shard in shards:
            receipt = _build_day(
                shard,
                behavior,
                index,
                output_dir,
                partition,
                contract_sha256,
                implementation_sha256,
            )
            receipts.append(receipt)
            behavior_receipts.append(receipt)
            completed_steps += 1
            atomic_json(
                output_dir / "progress.json",
                {
                    "record_type": "h015_on_policy_pack_build_progress",
                    "contract_sha256": contract_sha256,
                    "behavior_policy_id": behavior.behavior_id,
                    "steps_complete": completed_steps,
                    "steps_total": total_steps,
                    "rows_complete": sum(int(item["rows"]) for item in receipts),
                    "updated_at": now_utc(),
                },
            )
            if (output_dir / "PAUSE").exists():
                return {
                    "status": "paused",
                    "steps_complete": completed_steps,
                    "rows_complete": sum(int(item["rows"]) for item in receipts),
                    "contract_sha256": contract_sha256,
                }
        behavior_summaries.append(
            {
                "id": behavior.behavior_id,
                "code": behavior.code,
                "rows": len(index.targets),
                "fit_rows": sum(int(item["fit_rows"]) for item in behavior_receipts),
                "selection_rows": sum(
                    int(item["selection_rows"]) for item in behavior_receipts
                ),
                "action_counts": index.action_counts,
                "orders": index.orders,
                "fills": index.fills,
                "filled_decisions": index.filled_decisions,
                "requested_shares": str(index.requested_shares),
                "filled_shares": str(index.filled_shares),
                "executed_cost_usd": str(index.executed_cost_usd),
                "realized_pnl_usd": str(index.realized_pnl_usd),
                "result_sha256": sha256_file(behavior.replay_dir / "result.json"),
                "audit_manifest_sha256": sha256_file(
                    behavior.replay_dir / "manifest.json"
                ),
            }
        )
    rows = sum(int(receipt["rows"]) for receipt in receipts)
    fit_rows = sum(int(receipt["fit_rows"]) for receipt in receipts)
    selection_rows = sum(int(receipt["selection_rows"]) for receipt in receipts)
    if (
        rows != int(config["corpus"]["rows_expected"])
        or fit_rows != int(config["partition"]["fit_rows_expected"])
        or selection_rows != int(config["partition"]["selection_rows_expected"])
    ):
        raise RuntimeError(
            f"H015 row counts changed: rows={rows}, fit={fit_rows}, selection={selection_rows}"
        )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "h015_on_policy_portfolio_advantage_pack_manifest",
        "research_id": "SPH-T-H015",
        "dataset_id": config["corpus"]["id"],
        "generated_at": now_utc(),
        "valid": True,
        "behavior_policies": behavior_summaries,
        "days_per_behavior": len(shards),
        "shards": [
            {
                "behavior_policy_id": receipt["behavior_policy_id"],
                "behavior_policy_code": receipt["behavior_policy_code"],
                "date": receipt["date"],
                "rows": receipt["rows"],
                "fit_rows": receipt["fit_rows"],
                "selection_rows": receipt["selection_rows"],
                "receipt_path": (
                    f"receipts/behavior={receipt['behavior_policy_id']}/"
                    f"date={receipt['date']}.json"
                ),
                "receipt_sha256": sha256_file(
                    output_dir
                    / "receipts"
                    / f"behavior={receipt['behavior_policy_id']}"
                    / f"date={receipt['date']}.json"
                ),
            }
            for receipt in receipts
        ],
        "rows": rows,
        "fit_rows": fit_rows,
        "selection_rows": selection_rows,
        "fit_components": len(partition.fit_components),
        "selection_components": len(partition.selection_components),
        "cutoff_unix": partition.cutoff_unix,
        "partition_sha256": partition_sha256,
        "contract_sha256": contract_sha256,
        **sources,
        "teacher_action_used_as_imitation_label": False,
        "equal_market_training_weights_required": True,
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
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--h012-replay-dir", type=Path, required=True)
    parser.add_argument("--h014-replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.config.resolve(),
        args.pack_dir.resolve(),
        args.encoding_dir.resolve(),
        args.tape_dir.resolve(),
        args.h012_replay_dir.resolve(),
        args.h014_replay_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
