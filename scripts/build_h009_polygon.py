"""Backfill the required H009 Polygon participant transfer graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file, write_jsonl_zst
from sphinx_corpus.polygon_graph import (
    ERC20_TRANSFER_TOPIC,
    ERC1155_TRANSFER_BATCH_TOPIC,
    ERC1155_TRANSFER_SINGLE_TOPIC,
    TransferQuery,
    normalize_transfer_log,
    transfer_log_endpoints,
    transfer_queries,
)
from sphinx_corpus.rpc import PolygonRPC, RPCError
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"


@dataclass(frozen=True)
class ContractBoundary:
    id: str
    kind: str
    address: str
    start_block: int
    end_block: int


@dataclass(frozen=True)
class GraphTask:
    boundary: ContractBoundary
    batch_index: int
    wallets: list[str]
    wallet_digest: str
    query_index: int
    query: TransferQuery

    @property
    def id(self) -> str:
        return (
            f"{self.boundary.id}-{self.batch_index:08d}-{self.wallet_digest[:12]}-"
            f"{self.query_index:02d}"
        )


@dataclass(frozen=True)
class ContractScanTask:
    boundary: ContractBoundary
    edge_type: str
    topic: str
    start_block: int
    end_block: int
    participant_digest: str

    @property
    def id(self) -> str:
        return (
            f"contract-scan-{self.participant_digest[:12]}-{self.boundary.id}-"
            f"{self.edge_type}-{self.start_block:09d}-"
            f"{self.end_block:09d}"
        )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return parsed.astimezone(UTC)


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _can_split_rpc_error(error: RPCError) -> bool:
    text = str(error).lower()
    markers = (
        "block range",
        "too many",
        "response size",
        "query returned",
        "limit exceeded",
        "more than",
        "timeout",
        "timed out",
        "400 bad request",
        "413 payload too large",
        "413 request entity too large",
    )
    return any(marker in text for marker in markers)


def adaptive_logs(
    rpc: PolygonRPC,
    *,
    address: str,
    topics: list[Any],
    start_block: int,
    end_block: int,
    target_logs: int,
) -> list[dict[str, Any]]:
    pending = [(start_block, end_block)]
    output: list[dict[str, Any]] = []
    while pending:
        start, end = pending.pop()
        try:
            logs = rpc.logs_filter(
                address=address,
                topics=topics,
                start_block=start,
                end_block=end,
            )
        except RPCError as error:
            if start == end or not _can_split_rpc_error(error):
                raise
            middle = (start + end) // 2
            pending.extend(((middle + 1, end), (start, middle)))
            continue
        if len(logs) > target_logs and start < end:
            middle = (start + end) // 2
            pending.extend(((middle + 1, end), (start, middle)))
        else:
            output.extend(logs)
    return output


def _boundaries(
    rpc: PolygonRPC,
    contracts: list[dict[str, Any]],
) -> list[ContractBoundary]:
    latest = rpc.latest_block_number()
    boundaries: list[ContractBoundary] = []
    for contract in contracts:
        start = rpc.block_at_or_after(_parse_utc(str(contract["active_from"])), latest=latest)
        end_exclusive = rpc.block_at_or_after(
            _parse_utc(str(contract["active_until_exclusive"])),
            latest=latest,
        )
        boundaries.append(
            ContractBoundary(
                id=str(contract["id"]),
                kind=str(contract["kind"]),
                address=str(contract["address"]).lower(),
                start_block=start,
                end_block=end_exclusive - 1,
            )
        )
    return boundaries


def _participant_batches(path: Path, size: int, limit: int | None) -> list[list[str]]:
    if size <= 0:
        raise ValueError("wallet_batch_size must be positive")
    wallets: list[str] = []
    for row in iter_jsonl_zst(path):
        wallet = str(row.get("wallet") or "").lower()
        if wallet:
            wallets.append(wallet)
        if limit is not None and len(wallets) >= limit:
            break
    return [wallets[offset : offset + size] for offset in range(0, len(wallets), size)]


def _participant_set(path: Path, limit: int | None) -> set[str]:
    wallets: set[str] = set()
    for row in iter_jsonl_zst(path):
        wallet = str(row.get("wallet") or "").lower()
        if wallet:
            wallets.add(wallet)
        if limit is not None and len(wallets) >= limit:
            break
    return wallets


def _participant_digest(wallets: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(wallets)).encode()).hexdigest()


def _scan_topics(kind: str) -> tuple[tuple[str, str], ...]:
    if kind == "erc20":
        return (("erc20_transfer", ERC20_TRANSFER_TOPIC),)
    if kind == "erc1155":
        return (
            ("erc1155_transfer_single", ERC1155_TRANSFER_SINGLE_TOPIC),
            ("erc1155_transfer_batch", ERC1155_TRANSFER_BATCH_TOPIC),
        )
    raise ValueError(f"Unsupported graph contract kind: {kind}")


def _graph_index(path: Path, config_hash: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        PRAGMA cache_size=-262144;
        CREATE TABLE IF NOT EXISTS edges (
          edge_id TEXT PRIMARY KEY,
          timestamp_unix INTEGER NOT NULL,
          block_number INTEGER NOT NULL,
          transaction_hash TEXT NOT NULL,
          log_index INTEGER NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS edges_time_idx
          ON edges(timestamp_unix, block_number, transaction_hash, log_index);
        CREATE TABLE IF NOT EXISTS processed_tasks (
          task_id TEXT PRIMARY KEY,
          task_sha256 TEXT NOT NULL,
          source_rows INTEGER NOT NULL,
          inserted_rows INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if metadata and metadata != {"config_sha256": config_hash}:
        raise RuntimeError("Polygon graph index belongs to another H009 contract")
    if not metadata:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('config_sha256', ?)",
            (config_hash,),
        )
        connection.commit()
    return connection


def _edge_payloads(path: Path) -> list[tuple[str, int, int, str, int, str]]:
    output: list[tuple[str, int, int, str, int, str]] = []
    for row in iter_jsonl_zst(path):
        output.append(
            (
                str(row["edge_id"]),
                int(row["timestamp_unix"]),
                int(row["block_number"]),
                str(row["transaction_hash"]),
                int(row["log_index"]),
                json.dumps(row, separators=(",", ":"), ensure_ascii=False),
            )
        )
    return output


def consolidate_polygon_tasks(
    output_dir: Path,
    config_hash: str,
    task_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    connection = _graph_index(output_dir / "polygon-index.sqlite", config_hash)
    processed = {
        str(row["task_id"]): str(row["task_sha256"])
        for row in connection.execute("SELECT task_id, task_sha256 FROM processed_tasks")
    }
    new_tasks = 0
    for receipt in task_receipts:
        task_id = str(receipt["task_id"])
        task_hash = str(receipt["sha256"])
        if task_id in processed:
            if processed[task_id] != task_hash:
                raise RuntimeError(f"Polygon task changed after indexing: {task_id}")
            continue
        task_path = output_dir / str(receipt["path"])
        edge_rows = _edge_payloads(task_path)
        before = connection.total_changes
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
        inserted = connection.total_changes - before
        connection.execute(
            "INSERT INTO processed_tasks VALUES (?, ?, ?, ?)",
            (task_id, task_hash, len(edge_rows), inserted),
        )
        connection.commit()
        new_tasks += 1
    dates = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT date(timestamp_unix, 'unixepoch')
            FROM edges ORDER BY timestamp_unix
            """
        )
    ]
    shards: list[dict[str, Any]] = []
    for date in dates:
        output_path = output_dir / "polygon" / "stream" / f"date={date}.jsonl.zst"
        receipt_path = output_dir / "receipts" / "polygon" / f"date={date}.json"
        if new_tasks == 0 and receipt_path.exists() and output_path.exists():
            shard = _load_object(receipt_path)
        else:
            start_unix = int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp())
            end_unix = start_unix + 86_400
            cursor = connection.execute(
                """
                SELECT payload FROM edges
                WHERE timestamp_unix >= ? AND timestamp_unix < ?
                ORDER BY timestamp_unix, block_number, transaction_hash, log_index
                """,
                (start_unix, end_unix),
            )
            output_rows, size = write_jsonl_zst(
                output_path,
                (json.loads(str(row["payload"])) for row in cursor),
            )
            shard = {
                "date": date,
                "path": output_path.relative_to(output_dir).as_posix(),
                "rows": output_rows,
                "bytes": size,
                "sha256": sha256_file(output_path),
            }
            atomic_json(receipt_path, shard)
        shards.append(shard)
    unique_rows = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
    indexed_tasks_total = int(
        connection.execute("SELECT COUNT(*) FROM processed_tasks").fetchone()[0]
    )
    current_task_ids = {str(receipt["task_id"]) for receipt in task_receipts}
    indexed_current_tasks = sum(
        1
        for row in connection.execute("SELECT task_id FROM processed_tasks")
        if str(row["task_id"]) in current_task_ids
    )
    connection.close()
    if sum(int(shard["rows"]) for shard in shards) != unique_rows:
        raise RuntimeError("Polygon daily shards do not preserve every unique graph edge")
    return {
        "newly_indexed_tasks": new_tasks,
        "indexed_current_tasks": indexed_current_tasks,
        "indexed_tasks_total": indexed_tasks_total,
        "unique_rows": unique_rows,
        "duplicate_task_rows_removed": sum(int(receipt["rows"]) for receipt in task_receipts)
        - unique_rows,
        "shards": shards,
    }


def _execute_graph_task(
    task: GraphTask,
    *,
    config_hash: str,
    output_dir: Path,
    rpc_urls: list[str],
    endpoint_offset: int,
    target_logs: int,
) -> dict[str, Any]:
    task_root = output_dir / "polygon" / "tasks" / task.boundary.id
    output_path = task_root / f"task={task.id}.jsonl.zst"
    receipt_path = task_root / f"task={task.id}.json"
    if receipt_path.exists() and output_path.exists():
        cached_receipt = _load_object(receipt_path)
        if (
            cached_receipt.get("config_sha256") != config_hash
            or cached_receipt.get("wallet_batch_sha256") != task.wallet_digest
        ):
            raise RuntimeError(f"Polygon task {task.id} belongs to another source set")
        return cached_receipt
    errors: list[str] = []
    logs: list[dict[str, Any]] | None = None
    timestamps: dict[int, int] = {}
    endpoint_fingerprint = ""
    for attempt in range(len(rpc_urls)):
        rpc_url = rpc_urls[(endpoint_offset + attempt) % len(rpc_urls)]
        try:
            with PolygonRPC(rpc_url) as rpc:
                logs = adaptive_logs(
                    rpc,
                    address=task.boundary.address,
                    topics=task.query.topics,
                    start_block=task.boundary.start_block,
                    end_block=task.boundary.end_block,
                    target_logs=target_logs,
                )
                blocks = {int(str(log["blockNumber"]), 16) for log in logs}
                timestamps = rpc.timestamps(blocks, batch_size=100)
            endpoint_fingerprint = hashlib.sha256(rpc_url.encode()).hexdigest()
            break
        except RPCError as error:
            errors.append(type(error).__name__ + ":" + str(error)[:200])
    if logs is None:
        raise RPCError(
            f"Every registered RPC failed for graph task {task.id}: {' | '.join(errors)}"
        )
    by_edge_id: dict[str, dict[str, Any]] = {}
    for log in logs:
        block = int(str(log["blockNumber"]), 16)
        edge = normalize_transfer_log(
            log,
            edge_type=task.query.edge_type,
            timestamp_unix=timestamps[block],
        )
        by_edge_id[str(edge["edge_id"])] = edge
    ordered = sorted(
        by_edge_id.values(),
        key=lambda edge: (
            int(edge["block_number"]),
            str(edge["transaction_hash"]),
            int(edge["log_index"]),
        ),
    )
    rows, size = write_jsonl_zst(output_path, ordered)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_polygon_task_receipt",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "endpoint_fingerprint": endpoint_fingerprint,
        "task_id": task.id,
        "contract_id": task.boundary.id,
        "edge_type": task.query.edge_type,
        "wallet_batch_index": task.batch_index,
        "wallet_batch_sha256": task.wallet_digest,
        "wallet_count": len(task.wallets),
        "start_block": task.boundary.start_block,
        "end_block": task.boundary.end_block,
        "rows": rows,
        "bytes": size,
        "sha256": sha256_file(output_path),
        "path": output_path.relative_to(output_dir).as_posix(),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _execute_contract_scan_task(
    task: ContractScanTask,
    *,
    config_hash: str,
    output_dir: Path,
    participants: set[str],
    rpc_urls: list[str],
    endpoint_offset: int,
    target_logs: int,
) -> dict[str, Any]:
    task_root = output_dir / "polygon" / "tasks" / "contract-scan" / task.boundary.id
    output_path = task_root / f"task={task.id}.jsonl.zst"
    receipt_path = task_root / f"task={task.id}.json"
    if receipt_path.exists() and output_path.exists():
        cached_receipt = _load_object(receipt_path)
        if (
            cached_receipt.get("config_sha256") != config_hash
            or cached_receipt.get("participant_set_sha256") != task.participant_digest
        ):
            raise RuntimeError(f"Polygon task {task.id} belongs to another source set")
        return cached_receipt

    errors: list[str] = []
    logs: list[dict[str, Any]] | None = None
    selected: list[dict[str, Any]] = []
    timestamps: dict[int, int] = {}
    endpoint_fingerprint = ""
    for attempt in range(len(rpc_urls)):
        rpc_url = rpc_urls[(endpoint_offset + attempt) % len(rpc_urls)]
        try:
            with PolygonRPC(rpc_url) as rpc:
                logs = adaptive_logs(
                    rpc,
                    address=task.boundary.address,
                    topics=[task.topic],
                    start_block=task.start_block,
                    end_block=task.end_block,
                    target_logs=target_logs,
                )
                selected = [
                    log
                    for log in logs
                    if any(
                        endpoint in participants
                        for endpoint in transfer_log_endpoints(log, task.edge_type)
                    )
                ]
                blocks = {int(str(log["blockNumber"]), 16) for log in selected}
                timestamps = rpc.timestamps(blocks, batch_size=100)
            endpoint_fingerprint = hashlib.sha256(rpc_url.encode()).hexdigest()
            break
        except (RPCError, ValueError) as error:
            errors.append(type(error).__name__ + ":" + str(error)[:200])
            logs = None
    if logs is None:
        raise RPCError(
            f"Every registered RPC failed for contract scan {task.id}: " + " | ".join(errors)
        )

    by_edge_id: dict[str, dict[str, Any]] = {}
    for log in selected:
        block = int(str(log["blockNumber"]), 16)
        edge = normalize_transfer_log(
            log,
            edge_type=task.edge_type,
            timestamp_unix=timestamps[block],
        )
        by_edge_id[str(edge["edge_id"])] = edge
    ordered = sorted(
        by_edge_id.values(),
        key=lambda edge: (
            int(edge["block_number"]),
            str(edge["transaction_hash"]),
            int(edge["log_index"]),
        ),
    )
    rows, size = write_jsonl_zst(output_path, ordered)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_polygon_task_receipt",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "strategy": "contract-scan",
        "endpoint_fingerprint": endpoint_fingerprint,
        "task_id": task.id,
        "contract_id": task.boundary.id,
        "edge_type": task.edge_type,
        "participant_set_sha256": task.participant_digest,
        "participant_count": len(participants),
        "start_block": task.start_block,
        "end_block": task.end_block,
        "source_logs": len(logs),
        "rows": rows,
        "bytes": size,
        "sha256": sha256_file(output_path),
        "path": output_path.relative_to(output_dir).as_posix(),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build_polygon_graph(
    config_path: Path,
    output_dir: Path,
    rpc_urls: list[str],
    *,
    strategy: str,
    wallet_batch_size: int,
    wallet_limit: int | None,
    block_span: int,
    block_limit: int | None,
    target_logs: int,
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    config_hash = sha256_file(config_path)
    stream_manifest = _load_object(output_dir / "stream-manifest.json")
    participant_info = stream_manifest["participants"]
    participant_path = output_dir / str(participant_info["path"])
    if not rpc_urls:
        raise ValueError("At least one Polygon RPC URL is required")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if block_span <= 0:
        raise ValueError("block_span must be positive")
    if block_limit is not None and block_limit <= 0:
        raise ValueError("block_limit must be positive")
    if strategy not in {"contract-scan", "participant-topics"}:
        raise ValueError(f"Unsupported Polygon collection strategy: {strategy}")
    endpoint_fingerprint = hashlib.sha256("\n".join(sorted(rpc_urls)).encode()).hexdigest()
    boundaries: list[ContractBoundary] | None = None
    boundary_errors: list[str] = []
    for rpc_url in rpc_urls:
        try:
            with PolygonRPC(rpc_url) as rpc:
                boundaries = _boundaries(
                    rpc,
                    list(config["sources"]["polygon"]["contracts"]),
                )
            break
        except RPCError as error:
            boundary_errors.append(type(error).__name__ + ":" + str(error)[:200])
    if boundaries is None:
        raise RPCError(
            "Every registered RPC failed while resolving contract boundaries: "
            + " | ".join(boundary_errors)
        )

    participants = _participant_set(participant_path, wallet_limit)
    participant_digest = _participant_digest(participants)
    if not participants:
        raise RuntimeError("Participant index is empty")
    tasks: list[GraphTask | ContractScanTask] = []
    if strategy == "contract-scan":
        for boundary in boundaries:
            effective_end = boundary.end_block
            if block_limit is not None:
                effective_end = min(effective_end, boundary.start_block + block_limit - 1)
            for edge_type, topic in _scan_topics(boundary.kind):
                for start_block in range(boundary.start_block, effective_end + 1, block_span):
                    tasks.append(
                        ContractScanTask(
                            boundary=boundary,
                            edge_type=edge_type,
                            topic=topic,
                            start_block=start_block,
                            end_block=min(effective_end, start_block + block_span - 1),
                            participant_digest=participant_digest,
                        )
                    )
    else:
        batches = _participant_batches(participant_path, wallet_batch_size, wallet_limit)
        for boundary in boundaries:
            for batch_index, wallets in enumerate(batches):
                wallet_digest = hashlib.sha256("\n".join(wallets).encode()).hexdigest()
                tasks.extend(
                    GraphTask(
                        boundary=boundary,
                        batch_index=batch_index,
                        wallets=wallets,
                        wallet_digest=wallet_digest,
                        query_index=query_index,
                        query=query,
                    )
                    for query_index, query in enumerate(transfer_queries(boundary.kind, wallets))
                )
    task_receipts: list[dict[str, Any]] = []
    for offset in range(0, len(tasks), workers):
        if (output_dir / "PAUSE").exists():
            raise InterruptedError("Polygon backfill paused; remove PAUSE to resume")
        task_batch = tasks[offset : offset + workers]
        with ThreadPoolExecutor(max_workers=len(task_batch)) as executor:
            futures = {}
            for index, task in enumerate(task_batch):
                if isinstance(task, ContractScanTask):
                    future = executor.submit(
                        _execute_contract_scan_task,
                        task,
                        config_hash=config_hash,
                        output_dir=output_dir,
                        participants=participants,
                        rpc_urls=rpc_urls,
                        endpoint_offset=offset + index,
                        target_logs=target_logs,
                    )
                else:
                    future = executor.submit(
                        _execute_graph_task,
                        task,
                        config_hash=config_hash,
                        output_dir=output_dir,
                        rpc_urls=rpc_urls,
                        endpoint_offset=offset + index,
                        target_logs=target_logs,
                    )
                futures[future] = task.id
            for future in as_completed(futures):
                task_receipts.append(future.result())
    task_receipts.sort(key=lambda receipt: str(receipt["task_id"]))
    participants_selected = len(participants)
    full_participant_set = wallet_limit is None and participants_selected == int(
        participant_info["rows"]
    )
    consolidation = consolidate_polygon_tasks(output_dir, config_hash, task_receipts)
    expected_tasks = len(tasks)
    tasks_complete = len(task_receipts) == expected_tasks
    full_time_window = block_limit is None
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_polygon_manifest",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "endpoint_fingerprint": endpoint_fingerprint,
        "rpc_endpoint_count": len(rpc_urls),
        "strategy": strategy,
        "availability": True,
        "complete": (
            full_participant_set
            and full_time_window
            and tasks_complete
            and int(consolidation["indexed_current_tasks"]) == expected_tasks
        ),
        "full_participant_set": full_participant_set,
        "full_time_window": full_time_window,
        "participants": participants_selected,
        "participant_set_sha256": participant_digest,
        "wallet_batch_size": wallet_batch_size,
        "block_span": block_span,
        "block_limit": block_limit,
        "contracts": [boundary.__dict__ for boundary in boundaries],
        "task_count": len(task_receipts),
        "expected_task_count": expected_tasks,
        "rows_with_cross_task_duplicates_possible": sum(
            int(receipt["rows"]) for receipt in task_receipts
        ),
        "contract_scan_source_logs": sum(
            int(receipt.get("source_logs", 0)) for receipt in task_receipts
        ),
        "required_edge_types": config["sources"]["polygon"]["edge_types"],
        "task_receipts_sha256": hashlib.sha256(
            "\n".join(
                f"{receipt['task_id']}:{receipt['sha256']}" for receipt in task_receipts
            ).encode()
        ).hexdigest(),
        "consolidation": consolidation,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": "Transfer graph provenance only; no insider label or attribution.",
    }
    atomic_json(output_dir / "polygon-manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--rpc-url", action="append")
    value.add_argument(
        "--strategy",
        choices=("contract-scan", "participant-topics"),
        default="contract-scan",
    )
    value.add_argument("--wallet-batch-size", type=int, default=500)
    value.add_argument("--wallet-limit", type=int)
    value.add_argument("--block-span", type=int, default=20_000)
    value.add_argument("--block-limit", type=int)
    value.add_argument("--target-logs", type=int, default=5000)
    value.add_argument("--workers", type=int, default=8)
    return value


def main() -> None:
    args = parser().parse_args()
    config = load_json(args.config)
    env_name = str(config["sources"]["polygon"]["rpc_env"])
    env_value = os.environ.get(env_name)
    rpc_urls = args.rpc_url or ([value for value in (env_value or "").split(";") if value])
    if not rpc_urls:
        raise RuntimeError(f"Set {env_name} or pass --rpc-url for the Polygon graph backfill")
    result = build_polygon_graph(
        args.config.resolve(),
        args.output_dir.resolve(),
        rpc_urls,
        strategy=args.strategy,
        wallet_batch_size=args.wallet_batch_size,
        wallet_limit=args.wallet_limit,
        block_span=args.block_span,
        block_limit=args.block_limit,
        target_logs=args.target_logs,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
