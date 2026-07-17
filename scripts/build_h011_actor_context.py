"""Build resumable point-in-time maker/taker actor aggregates from CryptoHouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sphinx_corpus.cryptohouse import (
    CryptoHouseClient,
    CryptoHouseQuotaError,
    CryptoHouseResultLimitError,
    single_array,
)
from sphinx_corpus.io import atomic_json, now_utc, sha256_file, write_jsonl_zst


@dataclass(frozen=True, slots=True)
class ActorTask:
    start: datetime
    end_exclusive: datetime
    partition: int
    partitions: int

    @property
    def id(self) -> str:
        return (
            f"{self.start:%Y%m%d}-{self.end_exclusive:%Y%m%d}-"
            f"{self.partition:04d}-of-{self.partitions:04d}"
        )

    def split(self) -> tuple[ActorTask, ActorTask]:
        child_partitions = self.partitions * 2
        return (
            ActorTask(self.start, self.end_exclusive, self.partition, child_partitions),
            ActorTask(
                self.start,
                self.end_exclusive,
                self.partition + self.partitions,
                child_partitions,
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat().replace("+00:00", "Z"),
            "end_exclusive": self.end_exclusive.isoformat().replace("+00:00", "Z"),
            "partition": self.partition,
            "partitions": self.partitions,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ActorTask:
        return cls(
            _parse_utc(str(payload["start"])),
            _parse_utc(str(payload["end_exclusive"])),
            int(payload["partition"]),
            int(payload["partitions"]),
        )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _month_windows(start: datetime, end_exclusive: datetime) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end_exclusive:
        if cursor.month == 12:
            boundary = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            boundary = cursor.replace(month=cursor.month + 1, day=1)
        window_end = min(boundary, end_exclusive)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


def actor_query(task: ActorTask) -> str:
    start = task.start.strftime("%Y-%m-%d %H:%M:%S")
    end = task.end_exclusive.strftime("%Y-%m-%d %H:%M:%S")
    common = (
        f"timestamp >= toDateTime('{start}') AND timestamp < toDateTime('{end}') AND is_deleted = 0"
    )
    return f"""
SELECT groupArray((
    wallet, maker_fills, taker_fills, buy_fills, sell_fills,
    buy_notional_usd, sell_notional_usd, shares,
    counterparties, assets, average_price, price_std
)) AS actors
FROM
(
    SELECT
        wallet,
        sum(maker_fill) AS maker_fills,
        sum(taker_fill) AS taker_fills,
        sum(is_buy) AS buy_fills,
        sum(1 - is_buy) AS sell_fills,
        toFloat64(sumIf(notional_usd, is_buy = 1)) AS buy_notional_usd,
        toFloat64(sumIf(notional_usd, is_buy = 0)) AS sell_notional_usd,
        toFloat64(sum(shares_value)) AS shares,
        uniqCombined64(counterparty) AS counterparties,
        uniqCombined64(asset) AS assets,
        avg(price_value) AS average_price,
        stddevPop(price_value) AS price_std
    FROM
    (
        SELECT
            lower(maker) AS wallet,
            lower(taker) AS counterparty,
            if(maker_asset_id = '0', taker_asset_id, maker_asset_id) AS asset,
            1 AS maker_fill,
            0 AS taker_fill,
            maker_asset_id = '0' AS is_buy,
            if(maker_asset_id = '0', maker_amount_filled, taker_amount_filled)
                / 1000000 AS notional_usd,
            if(maker_asset_id = '0', taker_amount_filled, maker_amount_filled)
                / 1000000 AS shares_value,
            if(shares_value = 0, 0, notional_usd / shares_value) AS price_value
        FROM polymarket.orders_filled
        WHERE {common}
          AND cityHash64(lower(maker)) % {task.partitions} = {task.partition}
          AND length(maker) = 42

        UNION ALL

        SELECT
            lower(taker) AS wallet,
            lower(maker) AS counterparty,
            if(taker_asset_id = '0', maker_asset_id, taker_asset_id) AS asset,
            0 AS maker_fill,
            1 AS taker_fill,
            taker_asset_id = '0' AS is_buy,
            if(taker_asset_id = '0', taker_amount_filled, maker_amount_filled)
                / 1000000 AS notional_usd,
            if(taker_asset_id = '0', maker_amount_filled, taker_amount_filled)
                / 1000000 AS shares_value,
            if(shares_value = 0, 0, notional_usd / shares_value) AS price_value
        FROM polymarket.orders_filled
        WHERE {common}
          AND cityHash64(lower(taker)) % {task.partitions} = {task.partition}
          AND length(taker) = 42
    )
    GROUP BY wallet
    ORDER BY wallet
)
FORMAT JSON
""".strip()


def _actor_rows(task: ActorTask, values: list[Any]) -> list[dict[str, Any]]:
    keys = (
        "wallet",
        "maker_fills",
        "taker_fills",
        "buy_fills",
        "sell_fills",
        "buy_notional_usd",
        "sell_notional_usd",
        "shares",
        "counterparties",
        "assets",
        "average_price",
        "price_std",
    )
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, list) or len(value) != len(keys):
            raise RuntimeError("CryptoHouse actor tuple has an unexpected shape")
        actor = dict(zip(keys, value, strict=True))
        rows.append(
            {
                "schema_version": "1.0.0",
                "record_type": "chronicle_actor_context_delta",
                "available_at": task.end_exclusive.isoformat().replace("+00:00", "Z"),
                "window_start": task.start.isoformat().replace("+00:00", "Z"),
                "window_end_exclusive": task.end_exclusive.isoformat().replace("+00:00", "Z"),
                **actor,
                "source": "cryptohouse_polymarket_orders_filled",
            }
        )
    return rows


def _task_paths(task: ActorTask, output_dir: Path) -> tuple[Path, Path]:
    task_dir = output_dir / "tasks" / f"window={task.start:%Y-%m-%d}"
    return (
        task_dir / f"partition={task.partition:04d}.jsonl.zst",
        task_dir / f"partition={task.partition:04d}.json",
    )


def _execute_task(task: ActorTask, output_dir: Path, url: str) -> dict[str, Any]:
    output_path, receipt_path = _task_paths(task, output_dir)
    query = actor_query(task)
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    if receipt_path.exists() and output_path.exists():
        payload: object = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("query_sha256") != query_hash:
            raise RuntimeError(f"Actor task receipt changed: {task.id}")
        return payload
    with CryptoHouseClient(url) as client:
        result = client.query_json(query, query_id=f"sphinx-h011-{task.id}")
        actors = _actor_rows(task, single_array(result, "actors"))
        endpoint_fingerprint = client.endpoint_fingerprint
    rows, size = write_jsonl_zst(output_path, actors)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_actor_context_task_receipt",
        "generated_at": now_utc(),
        "task_id": task.id,
        "query_sha256": query_hash,
        "endpoint_fingerprint": endpoint_fingerprint,
        "window_start": task.start.isoformat().replace("+00:00", "Z"),
        "window_end_exclusive": task.end_exclusive.isoformat().replace("+00:00", "Z"),
        "partition": task.partition,
        "partitions": task.partitions,
        "rows": rows,
        "bytes": size,
        "sha256": sha256_file(output_path),
        "path": output_path.relative_to(output_dir).as_posix(),
        "source_rows_read": int(result.get("statistics", {}).get("rows_read", 0)),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _write_task_plan(
    path: Path,
    *,
    start: datetime,
    end_exclusive: datetime,
    initial_partitions: int,
    partition_limit: int | None,
    tasks: dict[str, ActorTask],
) -> None:
    atomic_json(
        path,
        {
            "schema_version": "1.0.0",
            "record_type": "chronicle_actor_context_task_plan",
            "updated_at": now_utc(),
            "start": start.isoformat().replace("+00:00", "Z"),
            "end_exclusive": end_exclusive.isoformat().replace("+00:00", "Z"),
            "initial_partitions": initial_partitions,
            "partition_limit": partition_limit,
            "adaptive_result_splitting": True,
            "tasks": [tasks[task_id].payload() for task_id in sorted(tasks)],
        },
    )


def _load_task_plan(
    path: Path,
    *,
    start: datetime,
    end_exclusive: datetime,
    initial_partitions: int,
    partition_limit: int | None,
) -> dict[str, ActorTask] | None:
    if not path.exists():
        return None
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Actor task plan is not an object")
    expected = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end_exclusive": end_exclusive.isoformat().replace("+00:00", "Z"),
        "initial_partitions": initial_partitions,
        "partition_limit": partition_limit,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Actor task plan changed at {key}")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise RuntimeError("Actor task plan has no task list")
    tasks = {
        task.id: task
        for raw in raw_tasks
        if isinstance(raw, dict)
        for task in [ActorTask.from_payload(raw)]
    }
    if len(tasks) != len(raw_tasks):
        raise RuntimeError("Actor task plan contains duplicate or invalid tasks")
    return tasks


def build_actor_context(
    output_dir: Path,
    *,
    url: str,
    start: datetime,
    end_exclusive: datetime,
    partitions: int,
    partition_limit: int | None,
    workers: int,
    retry_rounds: int = 6,
    presplit_after: datetime | None = None,
    presplit_partitions: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if partitions <= 0 or workers <= 0 or retry_rounds <= 0:
        raise ValueError("partitions, workers and retry rounds must be positive")
    if (presplit_after is None) != (presplit_partitions is None):
        raise ValueError("presplit_after and presplit_partitions must be supplied together")
    if presplit_partitions is not None and presplit_partitions < partitions:
        raise ValueError("presplit_partitions cannot be smaller than initial partitions")
    selected_partitions = range(
        min(partitions, partition_limit) if partition_limit is not None else partitions
    )
    windows = _month_windows(start, end_exclusive)
    initial_tasks = [
        ActorTask(window_start, window_end, partition, partitions)
        for window_start, window_end in windows
        for partition in selected_partitions
    ]
    plan_path = output_dir / "task-plan.json"
    coverage_tasks = _load_task_plan(
        plan_path,
        start=start,
        end_exclusive=end_exclusive,
        initial_partitions=partitions,
        partition_limit=partition_limit,
    )
    if coverage_tasks is None:
        coverage_tasks = {task.id: task for task in initial_tasks}
        _write_task_plan(
            plan_path,
            start=start,
            end_exclusive=end_exclusive,
            initial_partitions=partitions,
            partition_limit=partition_limit,
            tasks=coverage_tasks,
        )
    if presplit_after is not None and presplit_partitions is not None:
        changed = True
        while changed:
            changed = False
            for task in list(coverage_tasks.values()):
                output_path, receipt_path = _task_paths(task, output_dir)
                if (
                    task.start < presplit_after
                    or task.partitions >= presplit_partitions
                    or (output_path.exists() and receipt_path.exists())
                ):
                    continue
                if presplit_partitions % task.partitions:
                    raise ValueError(
                        "presplit_partitions must be a power-of-two multiple of leaf partitions"
                    )
                coverage_tasks.pop(task.id)
                children = task.split()
                coverage_tasks.update((child.id, child) for child in children)
                changed = True
        _write_task_plan(
            plan_path,
            start=start,
            end_exclusive=end_exclusive,
            initial_partitions=partitions,
            partition_limit=partition_limit,
            tasks=coverage_tasks,
        )
    receipts_by_id: dict[str, dict[str, Any]] = {}
    pending = list(coverage_tasks.values())
    failures: dict[str, str] = {}
    for retry_round in range(retry_rounds):
        next_pending: list[ActorTask] = []
        quota_wait_seconds = 0.0
        for offset in range(0, len(pending), workers):
            if (output_dir / "PAUSE").exists():
                raise InterruptedError("Actor-context build paused; remove PAUSE to resume")
            batch = pending[offset : offset + workers]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(_execute_task, task, output_dir, url): task for task in batch
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        receipts_by_id[task.id] = future.result()
                        failures.pop(task.id, None)
                    except CryptoHouseQuotaError as error:
                        failures[task.id] = f"{type(error).__name__}: {error}"
                        next_pending.append(task)
                        quota_wait_seconds = max(quota_wait_seconds, error.wait_seconds)
                    except CryptoHouseResultLimitError as error:
                        failures.pop(task.id, None)
                        coverage_tasks.pop(task.id)
                        children = task.split()
                        coverage_tasks.update((child.id, child) for child in children)
                        next_pending.extend(children)
                        failures[task.id] = (
                            f"{type(error).__name__}: split {task.partitions} into "
                            f"{children[0].partitions} partitions"
                        )
                    except Exception as error:
                        failures[task.id] = f"{type(error).__name__}: {error}"
                        next_pending.append(task)
            atomic_json(
                output_dir / "progress.json",
                {
                    "schema_version": "1.0.0",
                    "record_type": "chronicle_actor_context_progress",
                    "updated_at": now_utc(),
                    "completed_tasks": len(receipts_by_id),
                    "expected_tasks": len(coverage_tasks),
                    "pending_tasks": len(next_pending) + max(len(pending) - offset - workers, 0),
                    "retry_round": retry_round + 1,
                    "recent_failures": dict(sorted(failures.items())[-16:]),
                    "quota_wait_seconds": quota_wait_seconds,
                },
            )
            _write_task_plan(
                plan_path,
                start=start,
                end_exclusive=end_exclusive,
                initial_partitions=partitions,
                partition_limit=partition_limit,
                tasks=coverage_tasks,
            )
            if quota_wait_seconds > 0.0:
                next_pending.extend(pending[offset + workers :])
                break
        pending = sorted(next_pending, key=lambda task: task.id)
        if not pending:
            break
        if retry_round + 1 < retry_rounds:
            remaining_wait = max(quota_wait_seconds, min(16.0, float(2**retry_round)))
            while remaining_wait > 0.0:
                wait = min(30.0, remaining_wait)
                time.sleep(wait)
                remaining_wait -= wait
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "schema_version": "1.0.0",
                        "record_type": "chronicle_actor_context_progress",
                        "updated_at": now_utc(),
                        "completed_tasks": len(receipts_by_id),
                        "expected_tasks": len(coverage_tasks),
                        "pending_tasks": len(pending),
                        "retry_round": retry_round + 1,
                        "quota_wait_seconds": max(remaining_wait, 0.0),
                        "recent_failures": dict(sorted(failures.items())[-16:]),
                    },
                )
    if pending:
        sample = dict(sorted(failures.items())[:16])
        raise RuntimeError(
            f"Actor context still has {len(pending)} failed tasks after {retry_rounds} "
            f"rounds: {sample}"
        )
    missing_receipts = set(coverage_tasks) - set(receipts_by_id)
    if missing_receipts:
        raise RuntimeError(f"Actor context is missing {len(missing_receipts)} leaf receipts")
    receipts = [receipts_by_id[task_id] for task_id in coverage_tasks]
    receipts.sort(key=lambda row: str(row["task_id"]))
    expected_tasks = len(coverage_tasks)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_actor_context_manifest",
        "generated_at": now_utc(),
        "research_id": "SPH-T-H011",
        "endpoint_fingerprint": hashlib.sha256(url.rstrip("/").encode()).hexdigest(),
        "start": start.isoformat().replace("+00:00", "Z"),
        "end_exclusive": end_exclusive.isoformat().replace("+00:00", "Z"),
        "temporal_granularity": "calendar_month_delta_available_after_window_end",
        "partitions": partitions,
        "adaptive_result_splitting": True,
        "leaf_partition_counts": sorted({task.partitions for task in coverage_tasks.values()}),
        "partition_limit": partition_limit,
        "task_count": len(receipts),
        "expected_task_count": expected_tasks,
        "complete": partition_limit is None and len(receipts) == expected_tasks,
        "hard_actor_cap": None,
        "rows": sum(int(receipt["rows"]) for receipt in receipts),
        "bytes": sum(int(receipt["bytes"]) for receipt in receipts),
        "task_receipts_sha256": hashlib.sha256(
            "\n".join(f"{receipt['task_id']}:{receipt['sha256']}" for receipt in receipts).encode()
        ).hexdigest(),
        "tasks": receipts,
        "elapsed_seconds": time.perf_counter() - started,
        "availability_boundary": (
            "The channel is masked after end_exclusive and before each monthly "
            "available_at timestamp."
        ),
        "evidence_boundary": (
            "Public on-chain maker/taker actor aggregates, not funding-transfer "
            "edges, terminal labels, insider attribution or profit evidence."
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--url", default="https://crypto-clickhouse.clickhouse.com")
    value.add_argument("--start", default="2025-07-16T00:00:00Z")
    value.add_argument("--end-exclusive", default="2026-01-06T00:00:00Z")
    value.add_argument("--partitions", type=int, default=512)
    value.add_argument("--partition-limit", type=int)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--retry-rounds", type=int, default=6)
    value.add_argument("--presplit-after")
    value.add_argument("--presplit-partitions", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    result = build_actor_context(
        args.output_dir.resolve(),
        url=args.url,
        start=_parse_utc(args.start),
        end_exclusive=_parse_utc(args.end_exclusive),
        partitions=args.partitions,
        partition_limit=args.partition_limit,
        workers=args.workers,
        retry_rounds=args.retry_rounds,
        presplit_after=None if args.presplit_after is None else _parse_utc(args.presplit_after),
        presplit_partitions=args.presplit_partitions,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "task_count": result["task_count"],
                "expected_task_count": result["expected_task_count"],
                "rows": result["rows"],
                "bytes": result["bytes"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
