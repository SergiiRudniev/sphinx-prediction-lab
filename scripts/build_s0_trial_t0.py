from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sphinx_corpus.io import (
    atomic_json,
    iter_jsonl_zst,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_trace.chronicle import (
    MarketResolution,
    SplitPlan,
    build_condition_targets,
    market_resolution_from_atlas,
    target_row_is_causal,
    target_row_matches_contract,
)
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_trial_t0.json"


def spread_paths(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0:
        raise ValueError("source file limit must be positive")
    if len(paths) <= maximum:
        return paths
    return [
        paths[min(int(index * len(paths) / maximum), len(paths) - 1)] for index in range(maximum)
    ]


def _load_market_index(
    atlas_path: Path,
    config: dict[str, Any],
    plan: SplitPlan,
) -> tuple[dict[str, MarketResolution], dict[str, str], dict[str, Any]]:
    markets: dict[str, MarketResolution] = {}
    event_latest_resolution: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    for row in iter_jsonl_zst(atlas_path):
        counts["atlas_markets"] += 1
        market = market_resolution_from_atlas(row, config["eligibility"])
        if market is None:
            continue
        counts["eligible_binary_markets"] += 1
        markets[market.condition_id] = market
        current = event_latest_resolution.get(market.event_id)
        if current is None or market.resolved_at > current:
            event_latest_resolution[market.event_id] = market.resolved_at

    event_splits: dict[str, str] = {}
    events_by_split: Counter[str] = Counter()
    for event_id, resolved_at in event_latest_resolution.items():
        window = plan.locate(resolved_at)
        if window is None:
            events_by_split["purged_or_outside"] += 1
            continue
        event_splits[event_id] = window.id
        events_by_split[window.id] += 1
    counts["eligible_event_groups"] = len(event_latest_resolution)
    return (
        markets,
        event_splits,
        {
            "counts": dict(counts),
            "event_groups_by_split": dict(events_by_split),
        },
    )


def build_pilot(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    source_file_limit: int | None,
    maximum_examples: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    plan = SplitPlan.from_config(config)
    previous_receipt_path = output_dir / "receipt.json"
    previous_receipt: dict[str, Any] | None = None
    if previous_receipt_path.exists():
        previous_value = json.loads(previous_receipt_path.read_text(encoding="utf-8"))
        if isinstance(previous_value, dict):
            previous_receipt = previous_value
    atlas_path = data_dir / "normalized" / "atlas" / "markets.jsonl.zst"
    namespace = str(config["corpus"]["ledger_namespace"])
    ledger_root = data_dir / "normalized" / namespace
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)
    if not ledger_root.exists():
        raise FileNotFoundError(ledger_root)

    markets, event_splits, atlas_receipt = _load_market_index(atlas_path, config, plan)
    registered_limit = int(config["pilot"]["source_file_limit"])
    selected_limit = source_file_limit or registered_limit
    registered_examples = int(config["pilot"]["maximum_examples"])
    selected_examples = maximum_examples or registered_examples
    all_paths = sorted(ledger_root.rglob("*.jsonl.zst"))
    paths = spread_paths(all_paths, selected_limit)
    path_digest = hashlib.sha256(
        "\n".join(path.relative_to(data_dir).as_posix() for path in paths).encode()
    ).hexdigest()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    rows_retained = 0
    test_rows_withheld = 0
    for path in paths:
        for row in iter_jsonl_zst(path):
            rows_read += 1
            condition_id = str(row.get("condition_id") or "").lower()
            market = markets.get(condition_id)
            if market is None:
                continue
            split_id = event_splits.get(market.event_id)
            if split_id is None:
                continue
            if split_id == "test":
                test_rows_withheld += 1
                continue
            grouped[condition_id].append(row)
            rows_retained += 1

    examples_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "calibration": [],
    }
    event_ids_by_split: dict[str, set[str]] = defaultdict(set)
    duplicate_trade_ids = 0
    for condition_id in sorted(grouped):
        if sum(len(values) for values in examples_by_split.values()) >= selected_examples:
            break
        market = markets[condition_id]
        split_id = event_splits[market.event_id]
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in grouped[condition_id]:
            trade_id = str(row.get("trade_id") or "")
            if trade_id in rows_by_id:
                duplicate_trade_ids += 1
            rows_by_id[trade_id] = row
        condition_examples = build_condition_targets(
            list(rows_by_id.values()),
            market,
            plan.by_id(split_id),
            config,
        )
        remaining = selected_examples - sum(len(values) for values in examples_by_split.values())
        selected = condition_examples[:remaining]
        examples_by_split[split_id].extend(selected)
        if selected:
            event_ids_by_split[split_id].add(market.event_id)

    event_overlap = set()
    split_ids = tuple(examples_by_split)
    for index, left in enumerate(split_ids):
        for right in split_ids[index + 1 :]:
            event_overlap.update(event_ids_by_split[left] & event_ids_by_split[right])
    causal_violations = sum(
        int(not target_row_is_causal(row, plan))
        for rows in examples_by_split.values()
        for row in rows
    )
    contract_violations = sum(
        int(not target_row_matches_contract(row, config, plan))
        for rows in examples_by_split.values()
        for row in rows
    )
    if event_overlap or causal_violations or contract_violations:
        raise RuntimeError(
            f"Trial T0 invariants failed: event_overlap={len(event_overlap)}, "
            f"causal_violations={causal_violations}, "
            f"contract_violations={contract_violations}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, dict[str, Any]] = {}
    availability: Counter[str] = Counter()
    total_examples = 0
    for split_id, rows in examples_by_split.items():
        path = output_dir / f"split={split_id}.jsonl.zst"
        count, size = write_jsonl_zst(path, rows)
        total_examples += count
        output_files[split_id] = {
            "path": path.name,
            "rows": count,
            "bytes": size,
            "sha256": sha256_file(path),
        }
        for row in rows:
            targets = row["targets"]
            for horizon in config["targets"]["horizons"]:
                horizon_id = str(horizon["id"])
                availability[f"{split_id}:{horizon_id}:available"] += int(
                    targets[f"yes_markout_{horizon_id}"] is not None
                )

    if total_examples == 0:
        raise RuntimeError("The bounded Trial T0 pilot produced no target rows")
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_id": str(config["id"]),
        "config_sha256": sha256_file(config_path),
        "valid": True,
        "test_labels_opened": False,
        "test_rows_emitted": 0,
        "source": {
            "atlas_path": atlas_path.relative_to(data_dir).as_posix(),
            "ledger_namespace": namespace,
            "ledger_files_total": len(all_paths),
            "ledger_files_sampled": len(paths),
            "ledger_paths_sha256": path_digest,
            "ledger_rows_read": rows_read,
            "ledger_rows_retained_for_development": rows_retained,
            "test_source_rows_withheld": test_rows_withheld,
        },
        "atlas": atlas_receipt,
        "examples": {
            "total": total_examples,
            "by_split": {split_id: len(rows) for split_id, rows in examples_by_split.items()},
            "events_by_split": {
                split_id: len(event_ids) for split_id, event_ids in event_ids_by_split.items()
            },
            "markout_availability": dict(availability),
            "duplicate_source_trade_ids_dropped": duplicate_trade_ids,
            "causal_violations": causal_violations,
            "contract_violations": contract_violations,
            "event_overlap_count": len(event_overlap),
        },
        "outputs": output_files,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    previous_outputs = previous_receipt.get("outputs") if previous_receipt is not None else None
    previous_outputs_dict: dict[str, Any] = (
        previous_outputs if isinstance(previous_outputs, dict) else {}
    )
    comparable_previous = (
        previous_receipt is not None
        and previous_receipt.get("config_sha256") == result["config_sha256"]
        and bool(previous_outputs_dict)
        and previous_receipt.get("source", {}).get("ledger_paths_sha256") == path_digest
    )
    result["reproducibility"] = {
        "previous_comparable_receipt_found": comparable_previous,
        "output_hashes_match_previous": (
            None
            if not comparable_previous
            else all(
                isinstance(previous_outputs_dict.get(split_id), dict)
                and previous_outputs_dict[split_id].get("sha256") == output["sha256"]
                for split_id, output in output_files.items()
            )
        ),
    }
    atomic_json(output_dir / "receipt.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--data-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--source-file-limit", type=int)
    root.add_argument("--maximum-examples", type=int)
    return root


def main() -> None:
    args = parser().parse_args()
    result = build_pilot(
        args.config.resolve(),
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        source_file_limit=args.source_file_limit,
        maximum_examples=args.maximum_examples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
