from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sphinx_corpus.io import (
    atomic_json,
    iter_jsonl_zst,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_trace.config import load_json
from sphinx_trace.features import WalletEvent, build_feature_sequence, wallet_event

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0_train.json"


def spread_paths(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0:
        raise ValueError("source file limit must be positive")
    if len(paths) <= maximum:
        return paths
    return [
        paths[min(int(index * len(paths) / maximum), len(paths) - 1)] for index in range(maximum)
    ]


def _load_targets(
    target_dir: Path,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    expected = config["dataset"]["target_sha256"]
    for split_id in ("train", "validation", "calibration"):
        path = target_dir / f"split={split_id}.jsonl.zst"
        digest = sha256_file(path)
        if digest != expected[split_id]:
            raise RuntimeError(
                f"Target hash mismatch for {split_id}: {digest} != {expected[split_id]}"
            )
        rows = list(iter_jsonl_zst(path))
        registered_rows = int(config["dataset"]["target_rows"][split_id])
        if len(rows) != registered_rows:
            raise RuntimeError(
                f"Target row mismatch for {split_id}: {len(rows)} != {registered_rows}"
            )
        output[split_id] = rows
    if int(config["dataset"]["target_rows"]["test"]) != 0:
        raise RuntimeError("The development pack must register zero test rows")
    return output


def _path_digest(paths: list[Path], data_dir: Path) -> str:
    return hashlib.sha256(
        "\n".join(path.relative_to(data_dir).as_posix() for path in paths).encode()
    ).hexdigest()


def pack_features(
    config_path: Path,
    data_dir: Path,
    target_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    targets_by_split = _load_targets(target_dir, config)
    targets_by_condition: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in targets_by_split.values():
        for row in rows:
            targets_by_condition[str(row["condition_id"])].append(row)

    target_receipt = json.loads((target_dir / "receipt.json").read_text(encoding="utf-8"))
    if (
        not isinstance(target_receipt, dict)
        or target_receipt.get("test_labels_opened") is not False
    ):
        raise RuntimeError("A valid label-withheld target receipt is required")
    namespace = str(target_receipt["source"]["ledger_namespace"])
    ledger_root = data_dir / "normalized" / namespace
    all_paths = sorted(ledger_root.rglob("*.jsonl.zst"))
    selected_count = int(target_receipt["source"]["ledger_files_sampled"])
    paths = spread_paths(all_paths, selected_count)
    path_digest = _path_digest(paths, data_dir)
    if path_digest != target_receipt["source"]["ledger_paths_sha256"]:
        raise RuntimeError("Ledger source path selection no longer matches the target pilot")

    market_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    wallet_histories: defaultdict[str, list[WalletEvent]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for path in paths:
        counters["files_read"] += 1
        for row in iter_jsonl_zst(path):
            counters["rows_read"] += 1
            wallet = str(row.get("wallet") or "").lower()
            event = wallet_event(row)
            if wallet and event is not None:
                wallet_histories[wallet].append(event)
                counters["wallet_events"] += 1
            condition_id = str(row.get("condition_id") or "").lower()
            if condition_id in targets_by_condition:
                market_rows[condition_id].append(row)
                counters["target_market_rows"] += 1

    for history in wallet_histories.values():
        history.sort(key=lambda event: event.timestamp_unix)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_outputs: dict[str, dict[str, Any]] = {}
    event_ids_by_split: dict[str, set[str]] = defaultdict(set)
    feature_time_violations = 0
    missing_histories = 0
    for split_id, targets in targets_by_split.items():
        feature_rows: list[NDArray[np.float16]] = []
        token_type_rows: list[NDArray[np.uint8]] = []
        target_rows: list[NDArray[np.float32]] = []
        mask_rows: list[NDArray[np.uint8]] = []
        baseline_rows: list[NDArray[np.float32]] = []
        index_rows: list[dict[str, Any]] = []
        for target in targets:
            condition_id = str(target["condition_id"])
            packed = build_feature_sequence(
                market_rows[condition_id],
                wallet_histories,
                target,
                config,
            )
            if packed is None:
                missing_histories += 1
                continue
            features, token_types, target_values, target_mask = packed
            if int(target["feature_max_event_time_unix"]) >= int(target["decision_time_unix"]):
                feature_time_violations += 1
            baseline = np.zeros_like(target_values)
            baseline[0] = float(target["yes_price"])
            feature_rows.append(features)
            token_type_rows.append(token_types)
            target_rows.append(target_values)
            mask_rows.append(target_mask)
            baseline_rows.append(baseline)
            event_ids_by_split[split_id].add(str(target["event_id"]))
            index_rows.append(
                {
                    "example_id": target["example_id"],
                    "condition_id": condition_id,
                    "event_id": target["event_id"],
                    "decision_time_unix": target["decision_time_unix"],
                    "yes_price": target["yes_price"],
                    "split": split_id,
                }
            )
        if not feature_rows:
            raise RuntimeError(f"No features packed for {split_id}")
        split_dir = output_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, NDArray[Any]] = {
            "features": np.stack(feature_rows),
            "token_types": np.stack(token_type_rows),
            "targets": np.stack(target_rows),
            "target_mask": np.stack(mask_rows),
            "baselines": np.stack(baseline_rows),
        }
        files: dict[str, dict[str, Any]] = {}
        for name, array in arrays.items():
            path = split_dir / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            files[name] = {
                "path": path.relative_to(output_dir).as_posix(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        index_path = split_dir / "examples.jsonl.zst"
        index_count, index_bytes = write_jsonl_zst(index_path, index_rows)
        files["examples"] = {
            "path": index_path.relative_to(output_dir).as_posix(),
            "rows": index_count,
            "bytes": index_bytes,
            "sha256": sha256_file(index_path),
        }
        split_outputs[split_id] = {"rows": len(feature_rows), "files": files}

    overlap: set[str] = set()
    split_ids = tuple(event_ids_by_split)
    for index, left in enumerate(split_ids):
        for right in split_ids[index + 1 :]:
            overlap.update(event_ids_by_split[left] & event_ids_by_split[right])
    if feature_time_violations or overlap or missing_histories:
        raise RuntimeError(
            f"Feature pack invalid: time={feature_time_violations}, "
            f"event_overlap={len(overlap)}, missing_histories={missing_histories}"
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": now_utc(),
        "valid": True,
        "research_id": str(config["research_id"]),
        "config_id": str(config["id"]),
        "config_sha256": sha256_file(config_path),
        "test_rows_consumed": 0,
        "raw_wallet_id_feature": False,
        "source": {
            "target_dir": str(target_dir),
            "target_receipt_sha256": sha256_file(target_dir / "receipt.json"),
            "ledger_namespace": namespace,
            "ledger_files_total": len(all_paths),
            "ledger_files_read": counters["files_read"],
            "ledger_paths_sha256": path_digest,
            "ledger_rows_read": counters["rows_read"],
            "wallet_events_indexed": counters["wallet_events"],
            "unique_wallets_indexed": len(wallet_histories),
            "target_market_rows": counters["target_market_rows"],
        },
        "features": {
            **config["features"],
            "feature_time_violations": feature_time_violations,
            "event_overlap_count": len(overlap),
            "missing_histories": missing_histories,
        },
        "splits": split_outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "metadata.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--data-dir", type=Path, required=True)
    root.add_argument("--target-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = pack_features(
        args.config.resolve(),
        args.data_dir.resolve(),
        args.target_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
