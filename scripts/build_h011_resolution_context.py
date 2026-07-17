"""Build causal wallet-performance updates at public market resolution times."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    iter_jsonl_zst,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_trace.chronicle_h009 import parse_optional_utc, raw_jsonl_zst_lines
from sphinx_trace.config import load_json
from sphinx_trace.h011_sources import load_ledger_scope_index

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_h011_resolution_context_v1.json"
DEFAULT_H009_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"
DEVELOPMENT_SPLITS = frozenset({"train", "validation", "calibration"})


@dataclass(frozen=True, slots=True)
class ResolutionMarket:
    condition_id: str
    split: str
    resolution_unix: int
    payout0: int


@dataclass(frozen=True, slots=True)
class ResolutionScope:
    path: Path
    scope_id: str
    markets: tuple[ResolutionMarket, ...]
    expected_rows: int


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _digest(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _resolution_scopes(
    data_dir: Path,
    chronicle_dir: Path,
    namespace: str,
    scope_limit: int | None,
    h009_config: dict[str, Any],
) -> tuple[list[ResolutionScope], dict[str, int]]:
    ledger_root = data_dir / "normalized" / namespace
    source = h009_config["sources"]["ledger"]
    ledger_scopes, _ = load_ledger_scope_index(
        data_dir,
        chronicle_dir,
        namespace=namespace,
        expected_scopes=int(source["scope_groups"]),
        expected_markets=int(source["markets"]),
        expected_rows=int(source["rows"]),
        source_manifest_sha256=str(source["manifest_sha256"]),
    )
    connection = sqlite3.connect(
        f"file:{(chronicle_dir / 'catalog.sqlite').as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    test_labels = int(
        connection.execute(
            "SELECT COUNT(*) FROM markets WHERE split_id='test' AND terminal_label IS NOT NULL"
        ).fetchone()[0]
    )
    if test_labels:
        raise RuntimeError("H011 resolution builder encountered opened test labels")
    resolution_markets: dict[str, ResolutionMarket] = {}
    counts: Counter[str] = Counter()
    for row in connection.execute(
        """
        SELECT condition_id, split_id, closed_at, terminal_label
        FROM markets
        WHERE split_id IN ('train', 'validation', 'calibration')
          AND terminal_label IS NOT NULL
        ORDER BY condition_id
        """
    ):
        condition_id = str(row["condition_id"])
        split = str(row["split_id"])
        terminal = row["terminal_label"]
        parsed = parse_optional_utc(row["closed_at"])
        payout: object = json.loads(str(terminal))
        if (
            parsed is None
            or not isinstance(payout, list)
            or len(payout) != 2
            or payout not in ([1.0, 0.0], [0.0, 1.0])
        ):
            counts["invalid_resolution_scopes"] += 1
            continue
        resolution_markets[condition_id] = ResolutionMarket(
            condition_id=condition_id,
            split=split,
            resolution_unix=int(parsed.timestamp()),
            payout0=int(float(payout[0]) > 0.5),
        )
        counts[f"selected_{split}_markets"] += 1
    connection.close()
    selected = [
        ResolutionScope(
            path=ledger_root / f"scope={scope.scope_id}",
            scope_id=scope.scope_id,
            markets=tuple(
                resolution_markets[condition]
                for condition in scope.condition_ids
                if condition in resolution_markets
            ),
            expected_rows=scope.rows,
        )
        for scope in ledger_scopes
        if any(condition in resolution_markets for condition in scope.condition_ids)
    ]
    counts["selected_markets"] = sum(len(scope.markets) for scope in selected)
    counts["selected_scopes"] = len(selected)
    if scope_limit is not None:
        selected = selected[:scope_limit]
    counts["selected_run_scopes"] = len(selected)
    counts["selected_expected_rows"] = sum(scope.expected_rows for scope in selected)
    return selected, dict(counts)


def _scope_updates(scope: ResolutionScope) -> tuple[list[dict[str, Any]], dict[str, int]]:
    wallets: dict[tuple[str, str], list[float]] = {}
    counts: Counter[str] = Counter()
    markets = {market.condition_id: market for market in scope.markets}
    for path in sorted(scope.path.glob("*.jsonl.zst")):
        reader = raw_jsonl_zst_lines(path)
        try:
            for line in reader:
                payload: object = orjson.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("Resolution Ledger row is not an object")
                counts["source_rows"] += 1
                condition_id = str(payload.get("condition_id") or "").lower()
                market = markets.get(condition_id)
                if market is None:
                    continue
                timestamp = int(payload["timestamp_unix"])
                if timestamp > market.resolution_unix:
                    counts["post_resolution_rows_excluded"] += 1
                    continue
                wallet = str(payload.get("wallet") or "").lower()
                side = str(payload.get("side") or "").upper()
                outcome = int(payload["outcome_index"])
                size = float(payload["size"])
                notional = float(payload["notional_usd"])
                if not wallet or side not in {"BUY", "SELL"} or outcome not in {0, 1}:
                    raise RuntimeError(f"Invalid normalized resolution trade in {condition_id}")
                values = wallets.setdefault((condition_id, wallet), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                winning_outcome = 0 if market.payout0 else 1
                side_sign = 1.0 if side == "BUY" else -1.0
                outcome0_sign = 1.0 if outcome == 0 else -1.0
                winner_sign = 1.0 if outcome == winning_outcome else -1.0
                values[0] += abs(size)
                values[1] += side_sign * outcome0_sign * size
                values[2] += -notional if side == "BUY" else notional
                if outcome == winning_outcome:
                    values[3] += side_sign * size
                values[4] += winner_sign * side_sign * size
                values[5] += 1.0
        finally:
            reader.close()
    if counts["source_rows"] != scope.expected_rows:
        raise RuntimeError(
            f"Resolution source rows changed for {scope.scope_id}: "
            f"{counts['source_rows']} != {scope.expected_rows}"
        )
    rows = [
        {
            "schema_version": "1.0.0",
            "record_type": "h011_wallet_resolution_update",
            "resolution_time_unix": markets[condition_id].resolution_unix,
            "condition_id": condition_id,
            "split": markets[condition_id].split,
            "wallet": wallet,
            "observed_trade_count": int(values[5]),
            "gross_observed_shares": values[0],
            "outcome0_net_shares": values[1],
            "directional_edge": values[4] / values[0] if values[0] > 0.0 else 0.0,
            "pnl_proxy_usd": values[2] + values[3],
            "profitable_proxy": values[2] + values[3] > 0.0,
            "payout0": markets[condition_id].payout0,
        }
        for (condition_id, wallet), values in sorted(wallets.items())
        if values[0] > 0.0
    ]
    counts["wallet_updates"] = len(rows)
    return rows, dict(counts)


def _build_run(
    output_dir: Path,
    config_hash: str,
    run_id: str,
    scopes: list[ResolutionScope],
) -> dict[str, Any]:
    root = output_dir / "runs" / f"run={run_id}"
    receipt_path = root / "receipt.json"
    scope_digest = _digest([scope.scope_id for scope in scopes])
    if receipt_path.exists():
        receipt = _load_object(receipt_path)
        if (
            receipt.get("config_sha256") != config_hash
            or receipt.get("scope_digest") != scope_digest
        ):
            raise RuntimeError(f"Resolution run {run_id} belongs to another contract")
        return receipt
    by_date: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for scope in scopes:
        updates, scope_counts = _scope_updates(scope)
        counts.update(scope_counts)
        for update in updates:
            date = (
                datetime.fromtimestamp(int(update["resolution_time_unix"]), tz=UTC)
                .date()
                .isoformat()
            )
            by_date[date].append(update)
    shards: list[dict[str, Any]] = []
    for date, rows in sorted(by_date.items()):
        rows.sort(
            key=lambda row: (
                int(row["resolution_time_unix"]),
                str(row["condition_id"]),
                str(row["wallet"]),
            )
        )
        path = root / f"date={date}.jsonl.zst"
        row_count, size = write_jsonl_zst(path, rows)
        shards.append(
            {
                "date": date,
                "path": path.relative_to(output_dir).as_posix(),
                "rows": row_count,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema_version": "1.0.0",
        "record_type": "h011_resolution_run_receipt",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "run_id": run_id,
        "scope_digest": scope_digest,
        "scope_count": len(scopes),
        "source_rows": counts["source_rows"],
        "expected_source_rows": sum(scope.expected_rows for scope in scopes),
        "post_resolution_rows_excluded": counts["post_resolution_rows_excluded"],
        "rows": counts["wallet_updates"],
        "shards": shards,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _merge_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["resolution_time_unix"]),
        str(row["condition_id"]),
        str(row["wallet"]),
    )


def _merge_iterators(iterators: list[Iterator[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    heap: list[tuple[tuple[int, str, str], int, dict[str, Any]]] = []
    for index, iterator in enumerate(iterators):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (_merge_key(row), index, row))
    while heap:
        _, index, row = heapq.heappop(heap)
        yield row
        try:
            next_row = next(iterators[index])
        except StopIteration:
            continue
        heapq.heappush(heap, (_merge_key(next_row), index, next_row))


def build_resolution_context(
    config_path: Path,
    h009_config_path: Path,
    data_dir: Path,
    chronicle_dir: Path,
    output_dir: Path,
    *,
    workers: int,
    scope_limit: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    h009_config = load_json(h009_config_path)
    config_hash = sha256_file(config_path)
    if workers <= 0 or (scope_limit is not None and scope_limit <= 0):
        raise ValueError("Resolution workers and scope limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = (
        _load_object(output_dir / "manifest.json")
        if (output_dir / "manifest.json").exists()
        else None
    )
    check_disk_reserve(output_dir, float(config["storage"]["minimum_free_gib"]))
    namespace = str(h009_config["sources"]["ledger"]["namespace"])
    scopes, catalog_counts = _resolution_scopes(
        data_dir, chronicle_dir, namespace, scope_limit, h009_config
    )
    full_run = scope_limit is None
    group_size = int(config["storage"]["run_scope_group_size"])
    tasks: list[tuple[Path, str, str, list[ResolutionScope]]] = []
    for offset in range(0, len(scopes), group_size):
        group = scopes[offset : offset + group_size]
        digest = _digest([scope.scope_id for scope in group])
        run_id = f"{offset // group_size:05d}-{digest[:12]}"
        tasks.append((output_dir, config_hash, run_id, group))
    receipts: list[dict[str, Any]] = []
    for offset in range(0, len(tasks), workers):
        if (output_dir / "PAUSE").exists():
            raise InterruptedError("Resolution context paused; remove PAUSE to resume")
        batch = tasks[offset : offset + workers]
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = [executor.submit(_build_run, *task) for task in batch]
            receipts.extend(future.result() for future in futures)
    receipts.sort(key=lambda row: str(row["run_id"]))
    runs_manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_resolution_runs_manifest",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "full_run": full_run,
        "catalog_counts": catalog_counts,
        "scope_count": len(scopes),
        "source_rows": sum(int(row["source_rows"]) for row in receipts),
        "expected_source_rows": sum(int(row["expected_source_rows"]) for row in receipts),
        "rows": sum(int(row["rows"]) for row in receipts),
        "post_resolution_rows_excluded": sum(
            int(row["post_resolution_rows_excluded"]) for row in receipts
        ),
        "runs": [str(row["run_id"]) for row in receipts],
    }
    atomic_json(output_dir / "runs-manifest.json", runs_manifest)

    dates = sorted({str(shard["date"]) for receipt in receipts for shard in receipt["shards"]})
    daily: list[dict[str, Any]] = []
    for date in dates:
        if (output_dir / "PAUSE").exists():
            raise InterruptedError("Resolution merge paused; remove PAUSE to resume")
        output_path = output_dir / "events" / f"date={date}.jsonl.zst"
        receipt_path = output_dir / "receipts" / f"date={date}.json"
        if receipt_path.exists() and output_path.exists():
            receipt = _load_object(receipt_path)
            if receipt.get("config_sha256") != config_hash:
                raise RuntimeError(f"Resolution day {date} belongs to another contract")
            daily.append(receipt)
            continue
        paths = [
            output_dir / str(shard["path"])
            for receipt in receipts
            for shard in receipt["shards"]
            if str(shard["date"]) == date
        ]
        iterators = [iter_jsonl_zst(path) for path in paths]
        rows, size = write_jsonl_zst(output_path, _merge_iterators(iterators))
        receipt = {
            "schema_version": "1.0.0",
            "record_type": "h011_resolution_day_receipt",
            "generated_at": now_utc(),
            "config_sha256": config_hash,
            "date": date,
            "path": output_path.relative_to(output_dir).as_posix(),
            "rows": rows,
            "bytes": size,
            "sha256": sha256_file(output_path),
        }
        atomic_json(receipt_path, receipt)
        daily.append(receipt)
    rows = sum(int(receipt["rows"]) for receipt in daily)
    current_shard_digest = _digest(
        [
            f"{receipt['path']}:{receipt['rows']}:{receipt['bytes']}:{receipt['sha256']}"
            for receipt in daily
        ]
    )
    previous_comparable = bool(
        previous_manifest is not None
        and previous_manifest.get("config_sha256") == config_hash
        and previous_manifest.get("full_run") == full_run
        and int(previous_manifest.get("scope_count", -1)) == len(scopes)
    )
    previous_shards = [] if previous_manifest is None else previous_manifest.get("shards", [])
    previous_shard_digest = (
        _digest(
            [
                f"{receipt['path']}:{receipt['rows']}:{receipt['bytes']}:{receipt['sha256']}"
                for receipt in previous_shards
                if isinstance(receipt, dict)
            ]
        )
        if previous_comparable and isinstance(previous_shards, list)
        else None
    )
    output_hashes_match_previous = bool(
        previous_comparable and previous_shard_digest == current_shard_digest
    )
    valid = (
        rows == int(runs_manifest["rows"])
        and int(runs_manifest["source_rows"]) == int(runs_manifest["expected_source_rows"])
        and (not full_run or len(scopes) == int(catalog_counts["selected_scopes"]))
        and (not previous_comparable or output_hashes_match_previous)
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "h011_resolution_context_manifest",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "valid": valid,
        "complete": valid and full_run,
        "full_run": full_run,
        "test_labels_opened": False,
        "test_terminal_fields_accessed": False,
        "hard_wallet_cap": None,
        "hard_market_cap": None,
        "scope_count": len(scopes),
        "source_rows": runs_manifest["source_rows"],
        "post_resolution_rows_excluded": runs_manifest["post_resolution_rows_excluded"],
        "rows": rows,
        "days": len(daily),
        "shards": daily,
        "shard_receipts_sha256": current_shard_digest,
        "reproducibility": {
            "previous_comparable_manifest_found": previous_comparable,
            "output_hashes_match_previous": output_hashes_match_previous,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if not valid:
        raise RuntimeError("H011 resolution context failed acceptance")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--h009-config", type=Path, default=DEFAULT_H009_CONFIG)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--chronicle-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--workers", type=int)
    value.add_argument("--scope-limit", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    config = load_json(args.config.resolve())
    data_dir = args.data_dir.resolve()
    chronicle_dir = (
        args.chronicle_dir or data_dir / "derived" / "sphinx-chronicle-h009-v1"
    ).resolve()
    output_dir = (
        args.output_dir or data_dir / str(config["storage"]["default_output_dir"])
    ).resolve()
    result = build_resolution_context(
        args.config.resolve(),
        args.h009_config.resolve(),
        data_dir,
        chronicle_dir,
        output_dir,
        workers=args.workers or int(config["storage"]["workers"]),
        scope_limit=args.scope_limit,
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "complete": result["complete"],
                "scope_count": result["scope_count"],
                "source_rows": result["source_rows"],
                "rows": result["rows"],
                "days": result["days"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
