"""Bootstrap weekly and called-component H010 development profit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file
from sphinx_trace.config import load_json
from sphinx_trace.profit_evaluation import (
    independent_component_bootstrap,
    moving_block_bootstrap_mean,
    promotion_gates,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_h010_profit_evaluation_v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "src" / "sphinx_trace" / "profit_evaluation.py"):
        digest.update(f"{path.relative_to(ROOT).as_posix()}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _verify_replay_binding(replay_dir: Path, replay: dict[str, Any], audit: dict[str, Any]) -> None:
    manifest_path = replay_dir / "manifest.json"
    if replay.get("audit_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("H010 replay result is not bound to its audit manifest")
    for field in ("source_sha256", "policy_sha256", "implementation_sha256"):
        if audit.get(field) != replay.get(field):
            raise RuntimeError(f"H010 audit and replay disagree on {field}")
    digest = hashlib.sha256()
    rows = 0
    receipts = sorted((replay_dir / "receipts").glob("date=*.json"))
    for receipt_path in receipts:
        receipt = _load_object(receipt_path)
        date = str(receipt["date"])
        shard_path = replay_dir / str(receipt["path"])
        if any(
            receipt.get(field) != replay.get(field)
            for field in (
                "source_sha256",
                "policy_sha256",
                "implementation_sha256",
            )
        ):
            raise RuntimeError(f"H010 audit receipt contract changed at {date}")
        if not shard_path.exists() or receipt.get("sha256") != sha256_file(shard_path):
            raise RuntimeError(f"H010 audit shard changed at {date}")
        digest.update(f"{date}:{receipt['sha256']}\n".encode())
        rows += int(receipt["rows"])
    if (
        int(audit.get("days", -1)) != len(receipts)
        or int(audit.get("rows", -1)) != rows
        or audit.get("shard_digest") != digest.hexdigest()
    ):
        raise RuntimeError("H010 audit manifest no longer matches its receipts")


def evaluate(
    config_path: Path,
    replay_dir: Path,
    tape_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    result_path = replay_dir / "result.json"
    replay = _load_object(result_path)
    audit = _load_object(replay_dir / "manifest.json")
    if (
        replay.get("valid") is not True
        or replay.get("test_labels_opened") is not False
        or int(replay.get("test_rows_consumed", -1)) != 0
        or audit.get("valid") is not True
    ):
        raise RuntimeError("H010 profit evaluation requires a valid closed-test replay")
    _verify_replay_binding(replay_dir, replay, audit)
    split = str(replay["split"])
    condition_components: dict[str, str] = {}
    for row in iter_jsonl_zst(tape_dir / "conditions.jsonl.zst"):
        if str(row["split"]) == split:
            condition_components[str(row["condition_id"])] = str(row["component_id"])
    called: set[str] = set()
    condition_profit: dict[str, Decimal] = {}
    for shard_path in sorted((replay_dir / "shards").glob("date=*.jsonl.zst")):
        for row in iter_jsonl_zst(shard_path):
            record_type = str(row["record_type"])
            if record_type == "h010_decision_audit" and str(row["action"]) in {
                "CALL_OUTCOME_0",
                "CALL_OUTCOME_1",
            }:
                called.add(str(row["condition_id"]))
            elif record_type == "h010_resolution_audit":
                condition_profit[str(row["condition_id"])] = Decimal(
                    str(row["total_condition_realized_pnl_usd"])
                )
    if (
        not called
        or not called <= condition_components.keys()
        or not called <= condition_profit.keys()
    ):
        raise RuntimeError("H010 profit audit cannot settle every called condition")
    component_profit: defaultdict[str, Decimal] = defaultdict(Decimal)
    for condition_id in called:
        component_profit[condition_components[condition_id]] += condition_profit[condition_id]
    settings = config["bootstrap"]
    weekly_values = np.asarray(
        [float(row["net_profit_usd"]) for row in replay["weeks"]], dtype=np.float64
    )
    component_values = np.asarray(list(component_profit.values()), dtype=np.float64)
    weekly = moving_block_bootstrap_mean(
        weekly_values,
        replicates=int(settings["replicates"]),
        block_length=int(settings["weekly_block_length"]),
        seed=int(settings["seed"]),
        confidence=float(settings["confidence"]),
    )
    components = independent_component_bootstrap(
        component_values,
        replicates=int(settings["replicates"]),
        seed=int(settings["seed"]) + 1,
        confidence=float(settings["confidence"]),
    )
    gates = promotion_gates(
        weekly,
        components,
        minimum_calls=int(config["promotion"]["minimum_calls"]),
        minimum_components=int(config["promotion"]["minimum_independent_components"]),
        calls=int(replay["resolved_calls"]),
    )
    output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h010_profit_evaluation",
        "generated_at": now_utc(),
        "valid": True,
        "config_sha256": sha256_file(config_path),
        "replay_result_sha256": sha256_file(result_path),
        "audit_manifest_sha256": sha256_file(replay_dir / "manifest.json"),
        "implementation_sha256": _implementation_digest(),
        "split": split,
        "cost_multiplier": replay["cost_multiplier"],
        "weekly": weekly,
        "components": components,
        "gates": gates,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_path, output)
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--replay-dir", type=Path, required=True)
    value.add_argument("--tape-dir", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    replay_dir = args.replay_dir.resolve()
    output = args.output.resolve() if args.output else replay_dir / "profit-evaluation.json"
    result = evaluate(
        args.config.resolve(),
        replay_dir,
        args.tape_dir.resolve(),
        output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
