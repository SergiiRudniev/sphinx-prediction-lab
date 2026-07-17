"""Build the resumable H009 full-universe Sphinx Chronicle."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import zstandard

from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    iter_jsonl_zst,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_trace.chronicle import SplitPlan
from sphinx_trace.chronicle_h009 import (
    UnionFind,
    component_id_for_market,
    decision_id,
    extract_json_int,
    extract_json_string,
    market_seed_from_atlas,
    parse_optional_utc,
    raw_jsonl_zst_lines,
    snapshot_reasons,
    terminal_payout_from_atlas,
    trade_sort_key,
)
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"


def _json_text(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _digest_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _check_pause(output_dir: Path) -> None:
    if (output_dir / "PAUSE").exists():
        raise InterruptedError(
            f"Graceful pause requested by {output_dir / 'PAUSE'}; remove it to resume"
        )


def _spread_paths(paths: Sequence[Path], maximum: int | None) -> list[Path]:
    if maximum is None or maximum >= len(paths):
        return list(paths)
    if maximum <= 0:
        raise ValueError("scope_limit must be positive")
    return [paths[min(index * len(paths) // maximum, len(paths) - 1)] for index in range(maximum)]


def _open_catalog(path: Path, *, writable: bool = False) -> sqlite3.Connection:
    if writable:
        connection = sqlite3.connect(path)
    else:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _catalog_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        CREATE TABLE markets (
          condition_id TEXT PRIMARY KEY,
          atlas_row INTEGER NOT NULL UNIQUE,
          market_id TEXT NOT NULL,
          component_id TEXT NOT NULL,
          event_ids TEXT NOT NULL,
          outcomes TEXT NOT NULL,
          token_ids TEXT NOT NULL,
          question TEXT NOT NULL,
          description TEXT NOT NULL,
          created_at TEXT,
          start_at TEXT,
          end_at TEXT,
          closed_at TEXT,
          observed_at TEXT,
          resolution_status TEXT,
          neg_risk INTEGER NOT NULL,
          structurally_binary INTEGER NOT NULL,
          replayable INTEGER NOT NULL,
          split_id TEXT,
          terminal_label TEXT,
          semantic_feature_available INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX markets_component_idx ON markets(component_id, market_id);
        CREATE TABLE components (
          component_id TEXT PRIMARY KEY,
          event_ids TEXT NOT NULL,
          latest_closed_unix INTEGER,
          latest_closed_at TEXT,
          split_id TEXT,
          market_count INTEGER NOT NULL,
          neg_risk_market_count INTEGER NOT NULL,
          structurally_binary_market_count INTEGER NOT NULL,
          unclosed_market_count INTEGER NOT NULL,
          terminal_label_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )


def _component_members(event_component_ids: Mapping[str, str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for event_id, component_id in event_component_ids.items():
        output.setdefault(component_id, []).append(event_id)
    for members in output.values():
        members.sort()
    return output


def build_catalog(
    config: dict[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipts" / "catalog.json"
    catalog_path = output_dir / "catalog.sqlite"
    config_hash = sha256_file(config_path)
    if receipt_path.exists() and catalog_path.exists():
        cached_receipt = _load_object(receipt_path)
        if cached_receipt.get("config_sha256") == config_hash:
            return cached_receipt
        raise RuntimeError("Existing catalog belongs to a different H009 contract")

    started = time.perf_counter()
    atlas_path = data_dir / str(config["sources"]["atlas"]["markets"]["path"])
    union = UnionFind()
    first_pass_counts: Counter[str] = Counter()
    for row in iter_jsonl_zst(atlas_path):
        first_pass_counts["rows"] += 1
        seed = market_seed_from_atlas(row, allow_missing_condition=True)
        if seed is None:
            first_pass_counts["invalid"] += 1
            continue
        union.add_group(seed.event_ids)
        first_pass_counts["markets_with_multiple_event_ids"] += int(len(seed.event_ids) > 1)
    event_component_ids = union.component_ids()
    component_members = _component_members(event_component_ids)

    temporary = catalog_path.with_suffix(".sqlite.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = _open_catalog(temporary, writable=True)
    _catalog_schema(connection)
    market_sql = """
      INSERT INTO markets (
        condition_id, atlas_row, market_id, component_id, event_ids, outcomes,
        token_ids, question, description, created_at, start_at, end_at, closed_at,
        observed_at, resolution_status, neg_risk, structurally_binary, replayable
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    component_sql = """
      INSERT INTO components (
        component_id, event_ids, latest_closed_unix, latest_closed_at, split_id,
        market_count, neg_risk_market_count, structurally_binary_market_count,
        unclosed_market_count
      ) VALUES (?, ?, ?, ?, NULL, 1, ?, ?, ?)
      ON CONFLICT(component_id) DO UPDATE SET
        latest_closed_unix = CASE
          WHEN excluded.latest_closed_unix IS NULL THEN components.latest_closed_unix
          WHEN components.latest_closed_unix IS NULL THEN excluded.latest_closed_unix
          WHEN excluded.latest_closed_unix > components.latest_closed_unix
            THEN excluded.latest_closed_unix
          ELSE components.latest_closed_unix END,
        latest_closed_at = CASE
          WHEN excluded.latest_closed_unix IS NULL THEN components.latest_closed_at
          WHEN components.latest_closed_unix IS NULL THEN excluded.latest_closed_at
          WHEN excluded.latest_closed_unix > components.latest_closed_unix
            THEN excluded.latest_closed_at
          ELSE components.latest_closed_at END,
        market_count = components.market_count + 1,
        neg_risk_market_count = components.neg_risk_market_count + excluded.neg_risk_market_count,
        structurally_binary_market_count = components.structurally_binary_market_count
          + excluded.structurally_binary_market_count,
        unclosed_market_count = components.unclosed_market_count
          + excluded.unclosed_market_count
    """
    market_batch: list[tuple[object, ...]] = []
    component_batch: list[tuple[object, ...]] = []
    second_pass_counts: Counter[str] = Counter()
    for atlas_row, row in enumerate(iter_jsonl_zst(atlas_path)):
        seed = market_seed_from_atlas(row, allow_missing_condition=True)
        if seed is None:
            continue
        component_id = component_id_for_market(
            seed.condition_id,
            seed.event_ids,
            event_component_ids,
        )
        closed = parse_optional_utc(seed.closed_at)
        closed_unix = int(closed.timestamp()) if closed is not None else None
        market_batch.append(
            (
                seed.condition_id,
                atlas_row,
                seed.market_id,
                component_id,
                _json_text(seed.event_ids),
                _json_text(seed.outcomes),
                _json_text(seed.token_ids),
                seed.question,
                seed.description,
                seed.created_at,
                seed.start_at,
                seed.end_at,
                seed.closed_at,
                seed.observed_at,
                seed.resolution_status,
                int(seed.neg_risk),
                int(seed.structurally_binary),
                int(seed.source_condition_available and seed.structurally_binary),
            )
        )
        event_ids = component_members.get(component_id, [])
        component_batch.append(
            (
                component_id,
                _json_text(event_ids),
                closed_unix,
                seed.closed_at,
                int(seed.neg_risk),
                int(seed.structurally_binary),
                int(closed is None),
            )
        )
        second_pass_counts["markets"] += 1
        second_pass_counts["binary_markets"] += int(seed.structurally_binary)
        second_pass_counts["neg_risk_markets"] += int(seed.neg_risk)
        if len(market_batch) >= 4096:
            connection.executemany(market_sql, market_batch)
            connection.executemany(component_sql, component_batch)
            market_batch.clear()
            component_batch.clear()
    if market_batch:
        connection.executemany(market_sql, market_batch)
        connection.executemany(component_sql, component_batch)
    connection.commit()

    plan = SplitPlan.from_config(config)
    component_splits: list[tuple[str | None, str]] = []
    split_counts: Counter[str] = Counter()
    for component in connection.execute(
        """
        SELECT component_id, latest_closed_unix, unclosed_market_count
        FROM components ORDER BY component_id
        """
    ):
        closed_unix = component["latest_closed_unix"]
        window = (
            None
            if closed_unix is None or int(component["unclosed_market_count"]) > 0
            else plan.locate(datetime.fromtimestamp(int(closed_unix), tz=UTC))
        )
        split_id = None if window is None else window.id
        component_splits.append((split_id, str(component["component_id"])))
        split_counts[str(split_id or "purged_or_unresolved")] += 1
    connection.executemany(
        "UPDATE components SET split_id = ? WHERE component_id = ?",
        component_splits,
    )
    connection.execute(
        """
        UPDATE markets
        SET split_id = (
          SELECT components.split_id FROM components
          WHERE components.component_id = markets.component_id
        )
        """
    )
    connection.commit()

    label_splits = frozenset(str(value) for value in config["episode"]["development_labels"])
    ordered_splits = iter(
        connection.execute(
            "SELECT condition_id, split_id, replayable FROM markets ORDER BY atlas_row"
        )
    )
    label_batch: list[tuple[str, str]] = []
    label_counts: Counter[str] = Counter()
    split_rows_seen = 0
    for row in iter_jsonl_zst(atlas_path):
        if market_seed_from_atlas(row, allow_missing_condition=True) is None:
            continue
        split_row = next(ordered_splits)
        split_rows_seen += 1
        split_id_value = split_row["split_id"]
        split_id = None if split_id_value is None else str(split_id_value)
        payout = (
            terminal_payout_from_atlas(
                row,
                split_id=split_id,
                label_splits=label_splits,
            )
            if bool(split_row["replayable"])
            else None
        )
        label_counts[f"{split_id or 'none'}:markets"] += 1
        if payout is not None:
            label_batch.append((_json_text(payout), str(split_row["condition_id"])))
            label_counts[f"{split_id}:terminal_labels"] += 1
        if len(label_batch) >= 4096:
            connection.executemany(
                "UPDATE markets SET terminal_label = ? WHERE condition_id = ?",
                label_batch,
            )
            label_batch.clear()
    if label_batch:
        connection.executemany(
            "UPDATE markets SET terminal_label = ? WHERE condition_id = ?",
            label_batch,
        )
    try:
        next(ordered_splits)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Catalog label pass did not consume every structural Atlas market")
    if split_rows_seen != second_pass_counts["markets"]:
        raise RuntimeError("Catalog label pass row count changed")
    connection.execute(
        """
        UPDATE components
        SET terminal_label_count = (
          SELECT COUNT(*) FROM markets
          WHERE markets.component_id = components.component_id
            AND markets.terminal_label IS NOT NULL
        )
        """
    )
    metadata = {
        "schema_version": "1.0.0",
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "atlas_markets_sha256": str(config["sources"]["atlas"]["markets"]["sha256"]),
        "test_terminal_fields_accessed": "false",
        "semantic_feature_available": "false",
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        list(metadata.items()),
    )
    connection.commit()
    catalog_counts = {
        "markets": int(connection.execute("SELECT COUNT(*) FROM markets").fetchone()[0]),
        "components": int(connection.execute("SELECT COUNT(*) FROM components").fetchone()[0]),
        "multi_market_components": int(
            connection.execute("SELECT COUNT(*) FROM components WHERE market_count > 1").fetchone()[
                0
            ]
        ),
        "neg_risk_components": int(
            connection.execute(
                "SELECT COUNT(*) FROM components WHERE neg_risk_market_count > 0"
            ).fetchone()[0]
        ),
        "incomplete_lifecycle_components": int(
            connection.execute(
                "SELECT COUNT(*) FROM components WHERE unclosed_market_count > 0"
            ).fetchone()[0]
        ),
    }
    connection.close()
    _replace(temporary, catalog_path)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_catalog_receipt",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "valid": True,
        "test_labels_opened": False,
        "test_terminal_fields_accessed": False,
        "semantic_snapshot_point_in_time": False,
        "first_pass": dict(first_pass_counts),
        "counts": catalog_counts,
        "split_components": dict(split_counts),
        "labels": dict(label_counts),
        "path": catalog_path.name,
        "bytes": catalog_path.stat().st_size,
        "sha256": sha256_file(catalog_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _episode_rows(catalog_path: Path) -> Iterator[dict[str, Any]]:
    connection = _open_catalog(catalog_path)
    rows = connection.execute(
        """
        SELECT
          c.component_id AS c_component_id,
          c.event_ids AS c_event_ids,
          c.split_id AS c_split_id,
          c.latest_closed_at AS c_latest_closed_at,
          c.neg_risk_market_count AS c_neg_risk_market_count,
          c.unclosed_market_count AS c_unclosed_market_count,
          c.market_count AS c_market_count,
          m.condition_id,
          m.market_id,
          m.outcomes,
          m.token_ids,
          m.created_at,
          m.start_at,
          m.end_at,
          m.closed_at,
          m.resolution_status,
          m.neg_risk,
          m.terminal_label,
          m.semantic_feature_available,
          m.replayable
        FROM components AS c
        JOIN markets AS m ON m.component_id = c.component_id
        ORDER BY c.component_id, m.market_id
        """
    )
    component_row: sqlite3.Row | None = None
    markets: list[dict[str, Any]] = []

    def emit(component: sqlite3.Row, values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "record_type": "chronicle_episode",
            "component_id": component["c_component_id"],
            "event_ids": json.loads(component["c_event_ids"]),
            "split": component["c_split_id"],
            "anchor_closed_at": component["c_latest_closed_at"],
            "neg_risk": int(component["c_neg_risk_market_count"]) > 0,
            "complete_lifecycle": int(component["c_unclosed_market_count"]) == 0,
            "market_count": component["c_market_count"],
            "markets": values,
            "test_labels_opened": False,
        }

    try:
        for row in rows:
            if (
                component_row is not None
                and row["c_component_id"] != component_row["c_component_id"]
            ):
                yield emit(component_row, markets)
                markets = []
            component_row = row
            markets.append(
                {
                    "condition_id": row["condition_id"],
                    "market_id": row["market_id"],
                    "outcomes": json.loads(row["outcomes"]),
                    "token_ids": json.loads(row["token_ids"]),
                    "created_at": row["created_at"],
                    "start_at": row["start_at"],
                    "end_at": row["end_at"],
                    "closed_at": row["closed_at"],
                    "resolution_status": row["resolution_status"],
                    "neg_risk": bool(row["neg_risk"]),
                    "replayable": bool(row["replayable"]),
                    "terminal_label": (
                        None if row["terminal_label"] is None else json.loads(row["terminal_label"])
                    ),
                    "semantic_feature_available": bool(row["semantic_feature_available"]),
                }
            )
        if component_row is not None:
            yield emit(component_row, markets)
    finally:
        connection.close()


def build_episodes(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    receipt_path = output_dir / "receipts" / "episodes.json"
    output_path = output_dir / "episodes.jsonl.zst"
    config_hash = sha256_file(config_path)
    if receipt_path.exists() and output_path.exists():
        cached_receipt = _load_object(receipt_path)
        if cached_receipt.get("config_sha256") == config_hash:
            return cached_receipt
        raise RuntimeError("Existing episodes belong to a different H009 contract")
    started = time.perf_counter()
    count, size = write_jsonl_zst(output_path, _episode_rows(output_dir / "catalog.sqlite"))
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_episode_receipt",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "rows": count,
        "bytes": size,
        "sha256": sha256_file(output_path),
        "test_labels_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(receipt_path, receipt)
    return receipt


class ScopeLineIterator:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = iter(paths)
        self.reader: BinaryIO | None = None
        self.last_key: tuple[int, str] | None = None

    def __iter__(self) -> ScopeLineIterator:
        return self

    def __next__(self) -> bytes:
        while True:
            if self.reader is None:
                path = next(self.paths)
                self.reader = raw_jsonl_zst_lines(path)
            line = self.reader.readline()
            if not line:
                self.reader.close()
                self.reader = None
                continue
            key = trade_sort_key(line)
            if self.last_key is not None and key[0] < self.last_key[0]:
                raise RuntimeError("Ledger scope is not monotonically event-time ordered")
            self.last_key = key
            return line

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None


@dataclass
class RawShardWriter:
    path: Path
    temporary: Path
    source: BinaryIO
    writer: BinaryIO
    rows: int = 0
    first_key: tuple[int, str] | None = None
    last_key: tuple[int, str] | None = None

    @classmethod
    def open(cls, path: Path) -> RawShardWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        source = temporary.open("wb")
        writer = zstandard.ZstdCompressor(level=6, write_checksum=True).stream_writer(
            source,
            closefd=False,
        )
        return cls(path=path, temporary=temporary, source=source, writer=writer)

    def write(self, line: bytes, key: tuple[int, str]) -> None:
        if self.last_key is not None and key[0] < self.last_key[0]:
            raise RuntimeError("Output shard order regressed")
        self.writer.write(line if line.endswith(b"\n") else line + b"\n")
        if self.first_key is None:
            self.first_key = key
        self.last_key = key
        self.rows += 1

    def close(self) -> dict[str, Any]:
        self.writer.close()
        self.source.flush()
        os.fsync(self.source.fileno())
        self.source.close()
        _replace(self.temporary, self.path)
        return {
            "path": self.path.as_posix(),
            "rows": self.rows,
            "bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
            "first_key": self.first_key,
            "last_key": self.last_key,
        }


def _merge_raw_iterators(
    iterators: Sequence[Iterator[bytes]],
) -> Iterator[tuple[tuple[int, str], bytes]]:
    heap: list[tuple[tuple[int, str], int, bytes]] = []
    for index, iterator in enumerate(iterators):
        try:
            line = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (trade_sort_key(line), index, line))
    while heap:
        key, index, line = heapq.heappop(heap)
        yield key, line
        try:
            next_line = next(iterators[index])
        except StopIteration:
            continue
        heapq.heappush(heap, (trade_sort_key(next_line), index, next_line))


def _scope_id(path: Path) -> str:
    if not path.name.startswith("scope="):
        raise ValueError(f"Unexpected scope directory: {path}")
    return path.name.removeprefix("scope=")


def _scope_expected_rows(data_dir: Path, namespace: str, scope: Path) -> int:
    receipt = _load_object(data_dir / "receipts" / namespace / f"{_scope_id(scope)}.json")
    if receipt.get("complete") is not True or int(receipt.get("gaps") or 0) != 0:
        raise RuntimeError(f"Ledger scope is incomplete: {scope}")
    return int(receipt["rows"])


def _build_run_group(
    data_dir: Path,
    namespace: str,
    output_dir: Path,
    config_hash: str,
    group: list[Path],
    run_id: str,
) -> dict[str, Any]:
    run_root = output_dir / "runs" / f"run={run_id}"
    receipt_path = run_root / "receipt.json"
    if receipt_path.exists():
        cached_receipt = _load_object(receipt_path)
        if cached_receipt.get("config_sha256") != config_hash:
            raise RuntimeError(f"Run {run_id} belongs to another contract")
        return cached_receipt
    group_digest = _digest_text(path.name for path in group)
    expected_rows = sum(_scope_expected_rows(data_dir, namespace, path) for path in group)
    iterators = [ScopeLineIterator(sorted(scope.glob("*.jsonl.zst"))) for scope in group]
    shards: list[dict[str, Any]] = []
    wallets: set[str] = set()
    writer: RawShardWriter | None = None
    current_date: str | None = None
    rows = 0
    try:
        for key, line in _merge_raw_iterators(iterators):
            date = datetime.fromtimestamp(key[0], tz=UTC).date().isoformat()
            if date != current_date:
                if writer is not None:
                    shards.append(writer.close())
                current_date = date
                writer = RawShardWriter.open(run_root / f"date={date}.jsonl.zst")
            if writer is None:
                raise AssertionError("run writer was not initialized")
            writer.write(line, key)
            wallet = extract_json_string(line, b"wallet")
            if wallet:
                wallets.add(wallet)
            rows += 1
        if writer is not None:
            shards.append(writer.close())
            writer = None
    finally:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.writer.close()
                writer.source.close()
        for iterator in iterators:
            iterator.close()
    if rows != expected_rows:
        raise RuntimeError(f"Run {run_id} lost rows: {rows} != {expected_rows}")
    wallet_path = run_root / "wallets.jsonl.zst"
    wallet_rows, wallet_bytes = write_jsonl_zst(
        wallet_path,
        ({"wallet": wallet} for wallet in sorted(wallets)),
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_scope_run_receipt",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "run_id": run_id,
        "scope_count": len(group),
        "scope_digest": group_digest,
        "rows": rows,
        "wallet_index": {
            "path": wallet_path.as_posix(),
            "rows": wallet_rows,
            "bytes": wallet_bytes,
            "sha256": sha256_file(wallet_path),
        },
        "shards": shards,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build_runs(
    config: dict[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    scope_limit: int | None,
    workers: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_hash = sha256_file(config_path)
    namespace = str(config["sources"]["ledger"]["namespace"])
    ledger_root = data_dir / "normalized" / namespace
    scopes = _spread_paths(
        sorted(path for path in ledger_root.iterdir() if path.is_dir()),
        scope_limit,
    )
    scope_digest = _digest_text(path.name for path in scopes)
    group_size = int(config["storage"]["run_scope_group_size"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    run_tasks: list[tuple[Path, str, Path, str, list[Path], str]] = []
    for offset in range(0, len(scopes), group_size):
        group = scopes[offset : offset + group_size]
        group_digest = _digest_text(path.name for path in group)
        run_id = f"{offset // group_size:05d}-{group_digest[:12]}"
        run_tasks.append((data_dir, namespace, output_dir, config_hash, group, run_id))
    run_receipts: list[dict[str, Any]] = []
    if workers == 1:
        for task in run_tasks:
            _check_pause(output_dir)
            run_receipts.append(_build_run_group(*task))
    else:
        for offset in range(0, len(run_tasks), workers):
            _check_pause(output_dir)
            task_batch = run_tasks[offset : offset + workers]
            with ProcessPoolExecutor(max_workers=len(task_batch)) as executor:
                futures = [executor.submit(_build_run_group, *task) for task in task_batch]
                run_receipts.extend(future.result() for future in futures)
    run_receipts.sort(key=lambda receipt: str(receipt["run_id"]))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_runs_manifest",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "scope_digest": scope_digest,
        "scope_count": len(scopes),
        "full_scope_set": scope_limit is None,
        "rows": sum(int(receipt["rows"]) for receipt in run_receipts),
        "runs": [str(receipt["run_id"]) for receipt in run_receipts],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if scope_limit is None and result["rows"] != int(config["sources"]["ledger"]["rows"]):
        raise RuntimeError("Full run manifest does not preserve the registered Ledger row count")
    atomic_json(output_dir / "receipts" / "runs-manifest.json", result)
    return result


def _run_date_paths(output_dir: Path, run_ids: Sequence[str], date: str) -> list[Path]:
    paths = [output_dir / "runs" / f"run={run_id}" / f"date={date}.jsonl.zst" for run_id in run_ids]
    return [path for path in paths if path.exists()]


def build_stream(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_hash = sha256_file(config_path)
    runs_manifest = _load_object(output_dir / "receipts" / "runs-manifest.json")
    if runs_manifest.get("config_sha256") != config_hash:
        raise RuntimeError("Run manifest belongs to another H009 contract")
    run_ids = [str(value) for value in runs_manifest["runs"]]
    scope_digest = str(runs_manifest["scope_digest"])
    dates: set[str] = set()
    for run_id in run_ids:
        receipt = _load_object(output_dir / "runs" / f"run={run_id}" / "receipt.json")
        for shard in receipt["shards"]:
            name = Path(str(shard["path"])).name
            dates.add(name.removeprefix("date=").removesuffix(".jsonl.zst"))
    stream_receipt_root = output_dir / "receipts" / "stream"
    shards: list[dict[str, Any]] = []
    previous_last: tuple[int, str] | None = None
    for date in sorted(dates):
        _check_pause(output_dir)
        output_path = output_dir / "stream" / f"date={date}.jsonl.zst"
        receipt_path = stream_receipt_root / f"date={date}.json"
        if receipt_path.exists() and output_path.exists():
            cached_shard = _load_object(receipt_path)
            shard = cached_shard if cached_shard.get("scope_digest") == scope_digest else {}
        else:
            shard = {}
        if not shard:
            paths = _run_date_paths(output_dir, run_ids, date)
            readers = [raw_jsonl_zst_lines(path) for path in paths]
            writer = RawShardWriter.open(output_path)
            try:
                for key, line in _merge_raw_iterators(readers):
                    writer.write(line, key)
                shard = writer.close()
            finally:
                for reader in readers:
                    reader.close()
            shard["date"] = date
            shard["path"] = output_path.relative_to(output_dir).as_posix()
            shard["config_sha256"] = config_hash
            shard["scope_digest"] = scope_digest
            atomic_json(receipt_path, shard)
        first_raw = shard.get("first_key")
        last_raw = shard.get("last_key")
        first = None if first_raw is None else (int(first_raw[0]), str(first_raw[1]))
        last = None if last_raw is None else (int(last_raw[0]), str(last_raw[1]))
        if previous_last is not None and first is not None and first < previous_last:
            raise RuntimeError("Chronicle daily stream order regressed")
        if last is not None:
            previous_last = last
        shards.append(shard)
    rows = sum(int(shard["rows"]) for shard in shards)
    if rows != int(runs_manifest["rows"]):
        raise RuntimeError(f"Chronicle stream lost rows: {rows} != {runs_manifest['rows']}")
    participant_receipt_path = output_dir / "receipts" / "participants.json"
    participant_path = output_dir / "participants.jsonl.zst"
    if participant_receipt_path.exists() and participant_path.exists():
        cached_participants = _load_object(participant_receipt_path)
        participant_receipt = (
            cached_participants if cached_participants.get("scope_digest") == scope_digest else {}
        )
    else:
        participant_receipt = {}
    if not participant_receipt:
        participants: set[str] = set()
        for run_id in run_ids:
            run_receipt = _load_object(output_dir / "runs" / f"run={run_id}" / "receipt.json")
            wallet_index = run_receipt["wallet_index"]
            for row in iter_jsonl_zst(Path(str(wallet_index["path"]))):
                wallet = str(row.get("wallet") or "")
                if wallet:
                    participants.add(wallet)
        participant_rows, participant_bytes = write_jsonl_zst(
            participant_path,
            ({"wallet": wallet} for wallet in sorted(participants)),
        )
        participant_receipt = {
            "path": participant_path.relative_to(output_dir).as_posix(),
            "rows": participant_rows,
            "bytes": participant_bytes,
            "sha256": sha256_file(participant_path),
            "hard_wallet_count_cap": None,
            "scope_digest": scope_digest,
        }
        atomic_json(participant_receipt_path, participant_receipt)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_stream_manifest",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "scope_digest": scope_digest,
        "full_scope_set": runs_manifest["full_scope_set"],
        "rows": rows,
        "globally_ordered": True,
        "participants": participant_receipt,
        "shards": shards,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "stream-manifest.json", manifest)
    return manifest


def _decision_state(path: Path, config_hash: str, scope_digest: str) -> sqlite3.Connection:
    connection = _open_catalog(path, writable=True)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS event_state (
          component_id TEXT PRIMARY KEY,
          trade_count INTEGER NOT NULL,
          last_snapshot_unix INTEGER
        );
        CREATE TABLE IF NOT EXISTS processed_days (
          date TEXT PRIMARY KEY,
          output_path TEXT NOT NULL,
          stream_rows INTEGER NOT NULL,
          decision_rows INTEGER NOT NULL,
          output_bytes INTEGER NOT NULL,
          output_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    expected = {"config_sha256": config_hash, "scope_digest": scope_digest}
    if existing and existing != expected:
        raise RuntimeError("Decision state belongs to another H009 source set")
    if not existing:
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            list(expected.items()),
        )
        connection.commit()
    return connection


def build_decisions(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_hash = sha256_file(config_path)
    stream_manifest = _load_object(output_dir / "stream-manifest.json")
    if stream_manifest.get("config_sha256") != config_hash:
        raise RuntimeError("Stream manifest belongs to another H009 contract")
    catalog = _open_catalog(output_dir / "catalog.sqlite")
    condition_components = {
        str(row["condition_id"]): str(row["component_id"])
        for row in catalog.execute("SELECT condition_id, component_id FROM markets")
    }
    component_splits = {
        str(row["component_id"]): (None if row["split_id"] is None else str(row["split_id"]))
        for row in catalog.execute("SELECT component_id, split_id FROM components")
    }
    catalog.close()
    polygon_manifest_path = output_dir / "polygon-manifest.json"
    polygon_available = (
        polygon_manifest_path.exists()
        and _load_object(polygon_manifest_path).get("complete") is True
    )
    state_id = hashlib.sha256(
        f"{stream_manifest['scope_digest']}|polygon={str(polygon_available).lower()}".encode()
    ).hexdigest()[:16]
    state_connection = _decision_state(
        output_dir / f"decision-state-{state_id}.sqlite",
        config_hash,
        f"{stream_manifest['scope_digest']}|polygon={str(polygon_available).lower()}",
    )
    states: dict[str, tuple[int, int | None]] = {
        str(row["component_id"]): (
            int(row["trade_count"]),
            None if row["last_snapshot_unix"] is None else int(row["last_snapshot_unix"]),
        )
        for row in state_connection.execute("SELECT * FROM event_state")
    }
    processed = {
        str(row["date"]): dict(row)
        for row in state_connection.execute("SELECT * FROM processed_days")
    }
    day_receipts: list[dict[str, Any]] = []
    total_unmapped = 0
    for stream_shard in stream_manifest["shards"]:
        _check_pause(output_dir)
        date = str(stream_shard["date"])
        output_path = output_dir / "decisions" / f"date={date}.jsonl.zst"
        if date in processed:
            stored = processed[date]
            if not output_path.exists() or sha256_file(output_path) != stored["output_sha256"]:
                raise RuntimeError(f"Committed decision shard is missing or changed: {date}")
            receipt = {
                "date": date,
                "path": output_path.relative_to(output_dir).as_posix(),
                "stream_rows": stored["stream_rows"],
                "rows": stored["decision_rows"],
                "bytes": stored["output_bytes"],
                "sha256": stored["output_sha256"],
            }
            day_receipts.append(receipt)
            continue
        stream_path = output_dir / str(stream_shard["path"])
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = temporary.open("wb")
        writer: BinaryIO = zstandard.ZstdCompressor(
            level=6,
            write_checksum=True,
        ).stream_writer(
            source,
            closefd=False,
        )
        decision_rows = 0
        stream_rows = 0
        dirty: set[str] = set()
        unmapped = 0
        try:
            reader = raw_jsonl_zst_lines(stream_path)
            try:
                for stream_row, line in enumerate(reader):
                    stream_rows += 1
                    condition_id = extract_json_string(line, b"condition_id")
                    component_id = condition_components.get(condition_id)
                    if component_id is None:
                        unmapped += 1
                        continue
                    timestamp = extract_json_int(line, b"timestamp_unix")
                    count, last_snapshot = states.get(component_id, (0, None))
                    count += 1
                    reasons = snapshot_reasons(
                        event_trade_count=count,
                        trade_timestamp_unix=timestamp,
                        last_snapshot_timestamp_unix=last_snapshot,
                    )
                    if reasons:
                        last_snapshot = timestamp
                        record = {
                            "schema_version": "1.0.0",
                            "record_type": "chronicle_decision",
                            "decision_id": decision_id(component_id, timestamp, count),
                            "component_id": component_id,
                            "split": component_splits[component_id],
                            "decision_time_unix": timestamp,
                            "feature_max_event_time_unix": timestamp,
                            "event_trade_count": count,
                            "evidence_trade_id": extract_json_string(line, b"trade_id"),
                            "stream_shard": str(stream_shard["path"]),
                            "stream_shard_sha256": str(stream_shard["sha256"]),
                            "stream_row": stream_row,
                            "reason": list(reasons),
                            "availability": {
                                "ledger": True,
                                "atlas_point_in_time_semantics": False,
                                "polygon_graph": polygon_available,
                                "historical_depth": False,
                            },
                        }
                        writer.write(_json_text(record).encode() + b"\n")
                        decision_rows += 1
                    states[component_id] = (count, last_snapshot)
                    dirty.add(component_id)
            finally:
                reader.close()
            writer.close()
            source.flush()
            os.fsync(source.fileno())
            source.close()
            _replace(temporary, output_path)
        except BaseException:
            with contextlib.suppress(Exception):
                writer.close()
                source.close()
            raise
        output_hash = sha256_file(output_path)
        output_bytes = output_path.stat().st_size
        state_connection.execute("BEGIN IMMEDIATE")
        state_connection.executemany(
            """
            INSERT INTO event_state(component_id, trade_count, last_snapshot_unix)
            VALUES (?, ?, ?)
            ON CONFLICT(component_id) DO UPDATE SET
              trade_count=excluded.trade_count,
              last_snapshot_unix=excluded.last_snapshot_unix
            """,
            [
                (component_id, states[component_id][0], states[component_id][1])
                for component_id in dirty
            ],
        )
        state_connection.execute(
            "INSERT INTO processed_days VALUES (?, ?, ?, ?, ?, ?)",
            (
                date,
                output_path.relative_to(output_dir).as_posix(),
                stream_rows,
                decision_rows,
                output_bytes,
                output_hash,
            ),
        )
        state_connection.commit()
        total_unmapped += unmapped
        receipt = {
            "date": date,
            "path": output_path.relative_to(output_dir).as_posix(),
            "stream_rows": stream_rows,
            "rows": decision_rows,
            "bytes": output_bytes,
            "sha256": output_hash,
            "unmapped_rows": unmapped,
        }
        atomic_json(output_dir / "receipts" / "decisions" / f"date={date}.json", receipt)
        day_receipts.append(receipt)
    state_connection.close()
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_decision_manifest",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "scope_digest": stream_manifest["scope_digest"],
        "stream_rows": sum(int(value["stream_rows"]) for value in day_receipts),
        "rows": sum(int(value["rows"]) for value in day_receipts),
        "unmapped_stream_rows_in_this_invocation": total_unmapped,
        "hard_wallet_count_cap": None,
        "test_labels_opened": False,
        "polygon_availability": polygon_available,
        "shards": day_receipts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "decision-manifest.json", manifest)
    return manifest


def build_receipt(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    catalog = _load_object(output_dir / "receipts" / "catalog.json")
    episodes = _load_object(output_dir / "receipts" / "episodes.json")
    stream = _load_object(output_dir / "stream-manifest.json")
    decisions = _load_object(output_dir / "decision-manifest.json")
    polygon_manifest_path = output_dir / "polygon-manifest.json"
    polygon = _load_object(polygon_manifest_path) if polygon_manifest_path.exists() else None
    full_scope_set = bool(stream["full_scope_set"])
    stream_rows_valid = (
        int(stream["rows"]) == int(config["acceptance"]["ledger_rows_preserved_exactly"])
        if full_scope_set
        else int(stream["rows"]) > 0
    )
    polygon_complete = bool(polygon and polygon.get("complete") is True)
    structural_valid = (
        catalog.get("test_terminal_fields_accessed") is False
        and int(catalog["counts"]["multi_market_components"]) > 0
        and int(catalog["counts"]["neg_risk_components"]) > 0
        and stream_rows_valid
        and int(decisions["stream_rows"]) == int(stream["rows"])
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_build_receipt",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": sha256_file(config_path),
        "structural_valid": structural_valid,
        "fully_qualified": structural_valid and full_scope_set and polygon_complete,
        "full_scope_set": full_scope_set,
        "test_labels_opened": False,
        "test_terminal_fields_accessed": False,
        "hard_wallet_count_cap": None,
        "catalog": catalog,
        "episodes": episodes,
        "stream": {
            "rows": stream["rows"],
            "shards": len(stream["shards"]),
            "globally_ordered": stream["globally_ordered"],
        },
        "decisions": {
            "rows": decisions["rows"],
            "stream_rows": decisions["stream_rows"],
        },
        "polygon": polygon or {"complete": False, "availability": False},
        "qualification_blockers": (
            []
            if structural_valid and full_scope_set and polygon_complete
            else [
                reason
                for reason, blocked in (
                    ("structural_invariants", not structural_valid),
                    ("full_ledger_scope", not full_scope_set),
                    ("polygon_graph_backfill", not polygon_complete),
                )
                if blocked
            ]
        ),
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--scope-limit", type=int)
    value.add_argument("--workers", type=int)
    value.add_argument(
        "--phase",
        choices=("all", "catalog", "episodes", "runs", "stream", "decisions", "receipt"),
        default="all",
    )
    return value


def _cli_summary(results: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase, payload in results.items():
        summary = {
            key: payload[key]
            for key in (
                "record_type",
                "valid",
                "structural_valid",
                "fully_qualified",
                "full_scope_set",
                "rows",
                "stream_rows",
                "scope_count",
                "elapsed_seconds",
                "qualification_blockers",
            )
            if key in payload
        }
        for key in ("shards", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
        if phase == "catalog":
            summary["counts"] = payload.get("counts")
        output[phase] = summary
    return output


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    data_dir = args.data_dir.resolve()
    configured_output = data_dir / str(config["storage"]["default_output_dir"])
    output_dir = (args.output_dir or configured_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_disk_reserve(output_dir, float(config["storage"]["minimum_free_gib"]))
    phases = (
        ("catalog", "episodes", "runs", "stream", "decisions", "receipt")
        if args.phase == "all"
        else (args.phase,)
    )
    results: dict[str, Any] = {}
    for phase in phases:
        if phase == "catalog":
            results[phase] = build_catalog(config, config_path, data_dir, output_dir)
        elif phase == "episodes":
            results[phase] = build_episodes(config, config_path, output_dir)
        elif phase == "runs":
            results[phase] = build_runs(
                config,
                config_path,
                data_dir,
                output_dir,
                scope_limit=args.scope_limit,
                workers=args.workers or int(config["storage"]["run_workers"]),
            )
        elif phase == "stream":
            results[phase] = build_stream(config, config_path, output_dir)
        elif phase == "decisions":
            results[phase] = build_decisions(config, config_path, output_dir)
        elif phase == "receipt":
            results[phase] = build_receipt(config, config_path, output_dir)
    print(json.dumps(_cli_summary(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
