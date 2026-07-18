"""Build the closed-test H016 historical Polymarket fee-schedule artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx

from sphinx_corpus.config import CorpusConfig
from sphinx_corpus.io import (
    atomic_json,
    iter_jsonl_zst,
    load_json,
    now_utc,
    sha256_file,
    write_jsonl_zst,
)
from sphinx_corpus.ledger import event_topic
from sphinx_corpus.rpc import PolygonRPC, RPCError
from sphinx_trace.fee_schedule_h016 import (
    FEE_REFUNDED_TOPIC,
    FeeSourceCandidate,
    OnchainFeeEvidence,
    decode_onchain_fee_evidence,
    infer_fee_schedule,
    infer_fee_schedule_consensus,
    infer_fee_schedule_from_market_info,
    official_zero_schedule,
)
from sphinx_trace.polymarket_fees import (
    FeeProtocol,
    FeeScheduleEvidence,
    fee_schedule_payload,
)

ROOT = Path(__file__).resolve().parents[1]
ZERO = Decimal("0")
DEFAULT_CORPUS_CONFIG = ROOT / "configs" / "corpus" / "sphinx_corpus_v1.json"
FIRST_PLATFORM_FEE_UNIX = int(
    datetime(2026, 1, 5, tzinfo=UTC).timestamp()
)
CLOB_V2_CUTOVER_UNIX = int(
    datetime(2026, 4, 28, 11, tzinfo=UTC).timestamp()
)
REPLAY_FILL_HORIZON_SECONDS = 61
SCHEDULE_BOUNDARIES = tuple(
    int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    for value in (
        "2026-01-05T00:00:00Z",
        "2026-02-12T00:00:00Z",
        "2026-02-18T00:00:00Z",
        "2026-03-06T00:00:00Z",
        "2026-03-30T00:00:00Z",
        "2026-03-31T00:00:00Z",
        "2026-04-28T11:00:00Z",
    )
)


@dataclass(frozen=True, slots=True)
class TargetSegment:
    segment_id: str
    condition_id: str
    effective_from_unix: int
    effective_to_unix: int
    protocol: FeeProtocol
    token_ids: tuple[str, str]
    candidates: tuple[dict[str, Any], ...]


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _segment_id(condition_id: str, start: int, end: int) -> str:
    return _stable_hash(
        {"condition_id": condition_id, "effective_from_unix": start, "effective_to_unix": end}
    )


def _source_contract(
    tape_dir: Path,
    replay_dirs: tuple[Path, ...],
    corpus_config_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_schedule_build_contract",
        "tape_manifest_sha256": sha256_file(tape_dir / "manifest.json"),
        "replay_manifests": {
            directory.name: sha256_file(directory / "manifest.json")
            for directory in sorted(replay_dirs)
        },
        "corpus_config_sha256": sha256_file(corpus_config_path),
        "schedule_boundaries": list(SCHEDULE_BOUNDARIES),
        "interval_policy": "condition_by_registered_fee_epoch_active_order_hull",
        "candidate_policy": "prefer_single_public_fill_transaction_then_signal",
        "coverage_policy": "all_replay_decisions_plus_observed_orders_and_fills",
        "decision_fill_horizon_seconds": REPLAY_FILL_HORIZON_SECONDS,
        "first_platform_fee_unix": FIRST_PLATFORM_FEE_UNIX,
        "clob_v2_cutover_unix": CLOB_V2_CUTOVER_UNIX,
        "test_rows_consumed": 0,
        "test_labels_opened": False,
    }


def _extract_replay_targets(
    replay_dirs: tuple[Path, ...],
) -> tuple[
    dict[str, list[tuple[int, int]]],
    dict[str, set[tuple[str, int]]],
    dict[str, int],
]:
    condition_windows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    source_refs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    counts = {"orders": 0, "fills": 0}
    seen_decisions: set[str] = set()
    for replay_dir in replay_dirs:
        order_conditions: dict[str, str] = {}
        for shard in sorted((replay_dir / "shards").glob("*.jsonl.zst")):
            for row in iter_jsonl_zst(shard):
                record_type = str(row.get("record_type"))
                if record_type == "h010_order_audit":
                    condition_id = str(row["condition_id"]).lower()
                    submitted = int(row["submitted_at_unix"])
                    end = int(row["expires_at_unix"]) + 1
                    condition_windows[condition_id].append((submitted, end))
                    liquidity_id = str(row["evidence_liquidity_id"])
                    source_refs[liquidity_id].add((condition_id, submitted))
                    order_conditions[str(row["order_id"])] = condition_id
                    counts["orders"] += 1
                elif record_type == "h010_fill_audit":
                    fill_condition_id = order_conditions.get(str(row["order_id"]))
                    if fill_condition_id is None:
                        raise RuntimeError("H016 fill has no preceding replay order")
                    source_refs[str(row["liquidity_id"])].add(
                        (fill_condition_id, int(row["timestamp_unix"]))
                    )
                    counts["fills"] += 1
                elif record_type == "h010_decision_audit":
                    decision_id = str(row["decision_id"])
                    if decision_id in seen_decisions:
                        continue
                    seen_decisions.add(decision_id)
                    condition_id = str(row["condition_id"]).lower()
                    timestamp = int(row["timestamp_unix"])
                    evidence_trade_id = str(row["evidence_trade_id"])
                    condition_windows[condition_id].append(
                        (timestamp, timestamp + REPLAY_FILL_HORIZON_SECONDS)
                    )
                    source_refs[evidence_trade_id].add((condition_id, timestamp))
    counts["decisions"] = len(seen_decisions)
    if not condition_windows or not source_refs:
        raise RuntimeError("H016 found no replay fee targets")
    return dict(condition_windows), source_refs, counts


def _split_ranges(
    condition_windows: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, int, str]]]:
    output: dict[str, list[tuple[int, int, str]]] = {}
    epoch_bounds = (0, *SCHEDULE_BOUNDARIES, 2**63 - 1)
    for condition_id, windows in condition_windows.items():
        by_epoch: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for start, end in sorted(set(windows)):
            for epoch, (epoch_start, epoch_end) in enumerate(pairwise(epoch_bounds)):
                left = max(start, epoch_start)
                right = min(end, epoch_end)
                if left < right:
                    by_epoch[epoch].append((left, right))
        rows: list[tuple[int, int, str]] = []
        for pieces in by_epoch.values():
            start = min(row[0] for row in pieces)
            end = max(row[1] for row in pieces)
            rows.append(
                (start, end, _segment_id(condition_id, start, end))
            )
        output[condition_id] = sorted(rows)
    return output


def _optional_segment_for(
    ranges: dict[str, list[tuple[int, int, str]]],
    condition_id: str,
    timestamp_unix: int,
) -> str | None:
    for start, end, segment_id in ranges.get(condition_id, ()):
        if start <= timestamp_unix < end:
            return segment_id
    return None


def _load_token_ids(tape_dir: Path, conditions: set[str]) -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    for row in iter_jsonl_zst(tape_dir / "conditions.jsonl.zst"):
        condition_id = str(row["condition_id"]).lower()
        if condition_id not in conditions:
            continue
        values = tuple(str(value) for value in row["token_ids"])
        if len(values) != 2:
            raise RuntimeError("H016 requires binary condition token pairs")
        output[condition_id] = (values[0], values[1])
    missing = conditions - output.keys()
    if missing:
        raise RuntimeError(f"H016 tape catalog misses {len(missing)} target conditions")
    return output


def _candidate_score(row: dict[str, Any]) -> float:
    price = float(row["price"])
    curve = max(0.0, price * (1.0 - price))
    return float(row["size"]) * curve * curve


def _scan_candidates(
    tape_dir: Path,
    source_refs: dict[str, set[tuple[str, int]]],
    ranges: dict[str, list[tuple[int, int, str]]],
    *,
    candidates_per_segment: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    by_transaction: dict[
        str, dict[str, tuple[int, float, str, dict[str, Any]]]
    ] = defaultdict(dict)
    remaining = set(source_refs)
    for shard in sorted((tape_dir / "stream").glob("date=*.jsonl.zst")):
        for row in iter_jsonl_zst(shard):
            liquidity_id = str(row["trade_id"])
            refs = source_refs.get(liquidity_id)
            row_condition = str(row["condition_id"]).lower()
            timestamp = int(row["timestamp_unix"])
            if refs is not None:
                remaining.discard(liquidity_id)
                for condition_id, expected_timestamp in refs:
                    if row_condition != condition_id or timestamp != expected_timestamp:
                        raise RuntimeError(
                            "H016 replay liquidity binding changed in the tape"
                        )
            segment_id = _optional_segment_for(ranges, row_condition, timestamp)
            if segment_id is None:
                continue
            candidate = {
                "liquidity_id": liquidity_id,
                "transaction_hash": str(row["transaction_hash"]).lower(),
                "timestamp_unix": timestamp,
                "price": str(row["price"]),
                "size": str(row["size"]),
            }
            score = _candidate_score(row)
            key = f"{candidate['transaction_hash']}:{liquidity_id}"
            transaction_hash = str(candidate["transaction_hash"])
            bucket = by_transaction[segment_id]
            previous = bucket.get(transaction_hash)
            if previous is None:
                bucket[transaction_hash] = (1, score, key, candidate)
            else:
                best_key = previous[2]
                best_candidate = previous[3]
                if (score, key) > (previous[1], previous[2]):
                    best_key = key
                    best_candidate = candidate
                bucket[transaction_hash] = (
                    previous[0] + 1,
                    max(previous[1], score),
                    best_key,
                    best_candidate,
                )
            if len(bucket) > 100:
                retained = sorted(
                    bucket.items(),
                    key=lambda value: (
                        value[1][0] == 1,
                        -value[1][0],
                        value[1][1],
                        value[1][2],
                    ),
                    reverse=True,
                )[:20]
                by_transaction[segment_id] = dict(retained)
    if remaining:
        raise RuntimeError(f"H016 tape misses {len(remaining)} replay liquidity events")
    return {
        segment_id: tuple(
            {**row[3], "source_public_rows": row[0]}
            for row in sorted(
                values.values(),
                key=lambda value: (
                    value[0] == 1,
                    -value[0],
                    value[1],
                    value[2],
                ),
                reverse=True,
            )[:candidates_per_segment]
        )
        for segment_id, values in by_transaction.items()
    }


def build_plan(
    tape_dir: Path,
    replay_dirs: tuple[Path, ...],
    corpus_config_path: Path,
    *,
    candidates_per_segment: int,
) -> dict[str, Any]:
    contract = _source_contract(tape_dir, replay_dirs, corpus_config_path)
    ranges, source_refs, replay_counts = _extract_replay_targets(replay_dirs)
    split_ranges = _split_ranges(ranges)
    token_ids = _load_token_ids(tape_dir, set(ranges))
    candidates = _scan_candidates(
        tape_dir,
        source_refs,
        split_ranges,
        candidates_per_segment=candidates_per_segment,
    )
    segments: list[dict[str, Any]] = []
    for condition_id, rows in sorted(split_ranges.items()):
        for start, end, segment_id in rows:
            protocol = (
                FeeProtocol.CLOB_V1
                if end <= CLOB_V2_CUTOVER_UNIX
                else FeeProtocol.CLOB_V2
            )
            source_candidates = candidates.get(segment_id, ())
            if not source_candidates:
                # An order window can cross a registered tariff boundary without
                # any subsequent tape liquidity. Such a segment cannot fill and
                # needs no fee lookup; retaining it would invent a schedule source.
                continue
            segments.append(
                {
                    "segment_id": segment_id,
                    "condition_id": condition_id,
                    "effective_from_unix": start,
                    "effective_to_unix": end,
                    "protocol": protocol.value,
                    "token_ids": list(token_ids[condition_id]),
                    "candidates": list(source_candidates),
                }
            )
    return {
        **contract,
        "contract_sha256": _stable_hash(contract),
        "generated_at": now_utc(),
        "replay_rows": replay_counts,
        "target_conditions": len(ranges),
        "target_liquidity_ids": len(source_refs),
        "segments": segments,
        "segment_count": len(segments),
    }


class ReceiptCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS receipts "
            "(transaction_hash TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, transaction_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload FROM receipts WHERE transaction_hash = ?",
            (transaction_hash,),
        ).fetchone()
        if row is None:
            return None
        value: object = json.loads(str(row[0]))
        if not isinstance(value, dict):
            raise TypeError("H016 cached receipt must be an object")
        return value

    def put_many(self, rows: list[tuple[str, dict[str, Any]]]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO receipts(transaction_hash, payload) VALUES (?, ?)",
            [
                (transaction_hash, json.dumps(payload, separators=(",", ":")))
                for transaction_hash, payload in rows
            ],
        )
        self.connection.commit()


class MarketInfoCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS market_info "
            "(condition_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, condition_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload FROM market_info WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        value: object = json.loads(str(row[0]))
        if not isinstance(value, dict):
            raise TypeError("H016 cached CLOB market info must be an object")
        return value

    def put_many(self, rows: list[tuple[str, dict[str, Any]]]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO market_info (condition_id, payload) VALUES (?, ?)",
            [
                (condition_id, json.dumps(payload, sort_keys=True, separators=(",", ":")))
                for condition_id, payload in rows
            ],
        )
        self.connection.commit()


class MarketTradeCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS market_trades "
            "(condition_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, condition_id: str) -> list[dict[str, Any]] | None:
        row = self.connection.execute(
            "SELECT payload FROM market_trades WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        value: object = json.loads(str(row[0]))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise TypeError("H016 cached Data API market trades must be objects")
        return value

    def put_many(self, rows: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO market_trades (condition_id, payload) VALUES (?, ?)",
            [
                (condition_id, json.dumps(payload, sort_keys=True, separators=(",", ":")))
                for condition_id, payload in rows
            ],
        )
        self.connection.commit()


class RequestRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("H016 RPC request rate must be positive")
        self.interval_seconds = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            allowed = max(now, self.next_allowed)
            self.next_allowed = allowed + self.interval_seconds
        delay = allowed - now
        if delay > 0:
            time.sleep(delay)


def _fetch_missing_market_info(
    client: httpx.Client,
    cache: MarketInfoCache,
    condition_ids: list[str],
    *,
    workers: int,
    requests_per_second: float,
    output_dir: Path,
) -> bool:
    missing = [value for value in sorted(set(condition_ids)) if cache.get(value) is None]
    limiter = RequestRateLimiter(requests_per_second)

    def fetch(condition_id: str) -> tuple[str, dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(8):
            limiter.wait()
            try:
                response = client.get(f"/clob-markets/{condition_id}")
                response.raise_for_status()
                payload: object = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("CLOB market info response must be an object")
                return condition_id, payload
            except (httpx.HTTPError, TypeError, ValueError) as error:
                last_error = error
                time.sleep(min(30.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"H016 CLOB market info fetch failed: {last_error}")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h016-clob") as executor:
        for wave_start in range(0, len(missing), workers * 4):
            if (output_dir / "PAUSE").exists():
                return False
            wave = missing[wave_start : wave_start + workers * 4]
            rows = list(executor.map(fetch, wave))
            cache.put_many(rows)
            previous_completed = completed
            completed += len(rows)
            if completed // 100 > previous_completed // 100 or completed == len(missing):
                print(f"market info {completed}/{len(missing)}", flush=True)
    return True


def _fetch_missing_market_trades(
    client: httpx.Client,
    cache: MarketTradeCache,
    condition_ids: list[str],
    *,
    workers: int,
    requests_per_second: float,
    output_dir: Path,
) -> bool:
    missing = [value for value in sorted(set(condition_ids)) if cache.get(value) is None]
    limiter = RequestRateLimiter(requests_per_second)

    def fetch(condition_id: str) -> tuple[str, list[dict[str, Any]]]:
        last_error: Exception | None = None
        for attempt in range(8):
            limiter.wait()
            try:
                response = client.get(
                    "/trades",
                    params={"market": condition_id, "limit": 1000, "offset": 0},
                )
                response.raise_for_status()
                payload: object = response.json()
                if not isinstance(payload, list) or not all(
                    isinstance(item, dict) for item in payload
                ):
                    raise TypeError("Data API market trades response must be objects")
                return condition_id, payload
            except (httpx.HTTPError, TypeError, ValueError) as error:
                last_error = error
                time.sleep(min(30.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"H016 Data API market trades fetch failed: {last_error}")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h016-data") as executor:
        for wave_start in range(0, len(missing), workers * 4):
            if (output_dir / "PAUSE").exists():
                return False
            wave = missing[wave_start : wave_start + workers * 4]
            rows = list(executor.map(fetch, wave))
            cache.put_many(rows)
            completed += len(rows)
            print(f"market trades {completed}/{len(missing)}", flush=True)
    return True


def _fetch_missing(
    rpc: PolygonRPC,
    cache: ReceiptCache,
    transaction_hashes: list[str],
    *,
    batch_size: int,
    workers: int,
    requests_per_second: float,
    output_dir: Path,
) -> bool:
    missing = [value for value in sorted(set(transaction_hashes)) if cache.get(value) is None]
    batches = [
        missing[offset : offset + batch_size]
        for offset in range(0, len(missing), batch_size)
    ]
    completed = 0
    limiter = RequestRateLimiter(requests_per_second)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h016-rpc") as executor:
        for wave_start in range(0, len(batches), workers * 4):
            wave = batches[wave_start : wave_start + workers * 4]
            if (output_dir / "PAUSE").exists():
                return False
            wave_results = list(
                executor.map(
                    lambda batch: _fetch_receipt_batch(rpc, batch, limiter),
                    wave,
                )
            )
            rows: list[tuple[str, dict[str, Any]]] = []
            for batch, results in zip(wave, wave_results, strict=True):
                for transaction_hash, result in zip(batch, results, strict=True):
                    if not isinstance(result, dict):
                        raise RuntimeError(f"H016 receipt not found: {transaction_hash}")
                    rows.append((transaction_hash, result))
            cache.put_many(rows)
            previous_completed = completed
            completed += len(rows)
            if (
                completed // 250 > previous_completed // 250
                or completed == len(missing)
            ):
                print(f"receipts {completed}/{len(missing)}", flush=True)
    return True


def _fetch_receipt_batch(
    rpc: PolygonRPC,
    transaction_hashes: list[str],
    limiter: RequestRateLimiter,
) -> list[Any]:
    last_error: RPCError | None = None
    for cooldown_attempt in range(6):
        limiter.wait()
        try:
            return rpc.batch(
                ("eth_getTransactionReceipt", [value])
                for value in transaction_hashes
            )
        except RPCError as error:
            last_error = error
            if "429" not in str(error):
                break
            time.sleep(min(30.0, 2.0 * (2**cooldown_attempt)))
    if len(transaction_hashes) == 1:
        raise RPCError(f"H016 receipt fetch failed after cooldown: {last_error}")
    middle = len(transaction_hashes) // 2
    return [
        *_fetch_receipt_batch(rpc, transaction_hashes[:middle], limiter),
        *_fetch_receipt_batch(rpc, transaction_hashes[middle:], limiter),
    ]


def _source_candidate(segment: TargetSegment, row: dict[str, Any]) -> FeeSourceCandidate:
    return FeeSourceCandidate(
        condition_id=segment.condition_id,
        liquidity_id=str(row["liquidity_id"]),
        transaction_hash=str(row["transaction_hash"]).lower(),
        timestamp_unix=int(row["timestamp_unix"]),
        token_ids=segment.token_ids,
        effective_from_unix=segment.effective_from_unix,
        effective_to_unix=segment.effective_to_unix,
    )


def _market_wide_candidates(
    segment: TargetSegment,
    rows: list[dict[str, Any]],
    *,
    limit: int = 64,
) -> list[tuple[FeeSourceCandidate, dict[str, Any]]]:
    ranked: dict[str, tuple[Decimal, dict[str, Any]]] = {}
    for row in rows:
        if (
            str(row.get("conditionId", "")).lower() != segment.condition_id
            or str(row.get("asset", "")) not in segment.token_ids
        ):
            continue
        transaction_hash = str(row.get("transactionHash", "")).lower()
        try:
            price = Decimal(str(row["price"]))
            size = Decimal(str(row["size"]))
            timestamp_unix = int(row["timestamp"])
        except (KeyError, ValueError):
            continue
        if (
            not transaction_hash.startswith("0x")
            or len(transaction_hash) != 66
            or not ZERO < price < Decimal("1")
            or size <= ZERO
            or timestamp_unix >= CLOB_V2_CUTOVER_UNIX
        ):
            continue
        score = size * price * (Decimal("1") - price)
        previous = ranked.get(transaction_hash)
        if previous is None or score > previous[0]:
            ranked[transaction_hash] = (score, row)
    output: list[tuple[FeeSourceCandidate, dict[str, Any]]] = []
    for _, row in sorted(ranked.values(), key=lambda value: value[0], reverse=True)[:limit]:
        transaction_hash = str(row["transactionHash"]).lower()
        candidate = FeeSourceCandidate(
            condition_id=segment.condition_id,
            liquidity_id=_stable_hash(
                {
                    "source": "data_api_market_wide_fee_proof",
                    "condition_id": segment.condition_id,
                    "transaction_hash": transaction_hash,
                }
            ),
            transaction_hash=transaction_hash,
            timestamp_unix=int(row["timestamp"]),
            token_ids=segment.token_ids,
            effective_from_unix=segment.effective_from_unix,
            effective_to_unix=segment.effective_to_unix,
        )
        output.append((candidate, row))
    return output


def _segments(plan: dict[str, Any]) -> tuple[TargetSegment, ...]:
    output: list[TargetSegment] = []
    for row_value in plan["segments"]:
        row = dict(row_value)
        tokens = tuple(str(value) for value in row["token_ids"])
        if len(tokens) != 2:
            raise RuntimeError("H016 plan token pair changed")
        output.append(
            TargetSegment(
                segment_id=str(row["segment_id"]),
                condition_id=str(row["condition_id"]),
                effective_from_unix=int(row["effective_from_unix"]),
                effective_to_unix=int(row["effective_to_unix"]),
                protocol=FeeProtocol(str(row["protocol"])),
                token_ids=(tokens[0], tokens[1]),
                candidates=tuple(dict(value) for value in row["candidates"]),
            )
        )
    return tuple(output)


def _receipt_proof(
    receipt: dict[str, Any],
    schedule: FeeScheduleEvidence,
    config: CorpusConfig,
) -> dict[str, Any]:
    contract_topics = {
        (contract.address.lower(), event_topic(contract.event_signature))
        for contract in config.contracts
    }
    proof_logs = []
    for log in receipt.get("logs", []):
        if not isinstance(log, dict):
            continue
        topics = log.get("topics")
        if not isinstance(topics, list) or not topics:
            continue
        identity = (str(log.get("address", "")).lower(), str(topics[0]).lower())
        is_exchange = identity in contract_topics
        is_refund = (
            str(topics[0]).lower() == FEE_REFUNDED_TOPIC
            and len(topics) > 1
            and str(topics[1]).lower() == schedule.source_order_hash
        )
        if is_exchange or is_refund:
            proof_logs.append(log)
    return {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_receipt_proof",
        "schedule_id": schedule.schedule_id,
        "transaction_hash": schedule.transaction_hash,
        "status": receipt.get("status"),
        "block_number": receipt.get("blockNumber"),
        "block_hash": receipt.get("blockHash"),
        "transaction_index": receipt.get("transactionIndex"),
        "logs": proof_logs,
    }


def _market_info_proof(condition_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_market_info_proof",
        "condition_id": condition_id,
        "source_url": f"https://clob.polymarket.com/clob-markets/{condition_id}",
        "payload_sha256": _stable_hash(payload),
        "payload": payload,
    }


def _market_trade_proof(
    schedule: FeeScheduleEvidence,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_market_trade_proof",
        "schedule_id": schedule.schedule_id,
        "condition_id": schedule.condition_id,
        "transaction_hash": schedule.transaction_hash,
        "source_url": (
            "https://data-api.polymarket.com/trades?market="
            f"{schedule.condition_id}&limit=1000&offset=0"
        ),
        "row_sha256": _stable_hash(row),
        "row": row,
    }


def qualify(
    plan: dict[str, Any],
    config: CorpusConfig,
    output_dir: Path,
    *,
    rpc_url: str,
    batch_size: int,
    workers: int,
    requests_per_second: float,
    clob_url: str,
    market_info_requests_per_second: float,
    data_api_url: str,
    market_trade_requests_per_second: float,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    segments = _segments(plan)
    schedules: dict[str, FeeScheduleEvidence] = {}
    proofs: dict[str, dict[str, Any]] = {}
    supplemental_receipt_proofs: list[dict[str, Any]] = []
    market_info_proofs: dict[str, dict[str, Any]] = {}
    market_trade_proofs: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, TargetSegment] = {}
    for segment in segments:
        if segment.effective_to_unix <= FIRST_PLATFORM_FEE_UNIX:
            schedules[segment.segment_id] = official_zero_schedule(
                condition_id=segment.condition_id,
                effective_from_unix=segment.effective_from_unix,
                effective_to_unix=segment.effective_to_unix,
                protocol=segment.protocol,
            )
        else:
            unresolved[segment.segment_id] = segment
    resolved_cache_dir = output_dir if cache_dir is None else cache_dir
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ReceiptCache(resolved_cache_dir / "receipt-cache.sqlite3")
    market_info_cache = MarketInfoCache(resolved_cache_dir / "market-info-cache.sqlite3")
    market_trade_cache = MarketTradeCache(resolved_cache_dir / "market-trade-cache.sqlite3")
    failures: dict[str, list[str]] = defaultdict(list)
    try:
        with PolygonRPC(rpc_url, retries=2) as rpc:
            maximum_rank = max((len(row.candidates) for row in unresolved.values()), default=0)
            for rank in range(maximum_rank):
                current = [
                    (segment_id, segment, segment.candidates[rank])
                    for segment_id, segment in unresolved.items()
                    if rank < len(segment.candidates)
                ]
                if not _fetch_missing(
                    rpc,
                    cache,
                    [str(row[2]["transaction_hash"]) for row in current],
                    batch_size=batch_size,
                    workers=workers,
                    requests_per_second=requests_per_second,
                    output_dir=output_dir,
                ):
                    return {
                        "status": "paused",
                        "qualified_segments": len(schedules),
                        "unresolved_segments": len(unresolved),
                    }
                resolved_now: list[str] = []
                for segment_id, segment, row in current:
                    candidate = _source_candidate(segment, row)
                    receipt = cache.get(candidate.transaction_hash)
                    if receipt is None:
                        raise RuntimeError("H016 receipt disappeared from cache")
                    try:
                        evidence = decode_onchain_fee_evidence(
                            receipt,
                            candidate,
                            config.contracts,
                            chain_id=config.chain_id,
                        )
                        if evidence.protocol != segment.protocol:
                            raise RuntimeError("H016 planned protocol does not match receipt")
                        schedule = infer_fee_schedule(candidate, evidence)
                    except (RuntimeError, ValueError) as error:
                        failures[segment_id].append(
                            f"{candidate.transaction_hash}: {type(error).__name__}: {error}"
                        )
                        continue
                    schedules[segment_id] = schedule
                    proofs[schedule.schedule_id] = _receipt_proof(receipt, schedule, config)
                    resolved_now.append(segment_id)
                for segment_id in resolved_now:
                    del unresolved[segment_id]
                print(
                    f"rank {rank}: qualified={len(schedules)}/{len(segments)} "
                    f"unresolved={len(unresolved)}",
                    flush=True,
                )
                if not unresolved:
                    break
        if unresolved:
            with httpx.Client(
                base_url=clob_url.rstrip("/"),
                timeout=30.0,
                headers={"User-Agent": "sphinx-prediction-lab-h016/1.0"},
            ) as client:
                if not _fetch_missing_market_info(
                    client,
                    market_info_cache,
                    [segment.condition_id for segment in unresolved.values()],
                    workers=workers,
                    requests_per_second=market_info_requests_per_second,
                    output_dir=output_dir,
                ):
                    return {
                        "status": "paused",
                        "qualified_segments": len(schedules),
                        "unresolved_segments": len(unresolved),
                    }
            resolved_now = []
            for segment_id, segment in unresolved.items():
                market_info = market_info_cache.get(segment.condition_id)
                if market_info is None:
                    raise RuntimeError("H016 CLOB market info disappeared from cache")
                for row in segment.candidates:
                    candidate = _source_candidate(segment, row)
                    receipt = cache.get(candidate.transaction_hash)
                    if receipt is None:
                        continue
                    try:
                        evidence = decode_onchain_fee_evidence(
                            receipt,
                            candidate,
                            config.contracts,
                            chain_id=config.chain_id,
                        )
                        schedule = infer_fee_schedule_from_market_info(
                            candidate,
                            evidence,
                            market_info,
                        )
                    except (RuntimeError, ValueError, TypeError) as error:
                        failures[segment_id].append(
                            "market-info+"
                            f"{candidate.transaction_hash}: {type(error).__name__}: {error}"
                        )
                        continue
                    schedules[segment_id] = schedule
                    proofs[schedule.schedule_id] = _receipt_proof(receipt, schedule, config)
                    market_info_proofs[segment.condition_id] = _market_info_proof(
                        segment.condition_id,
                        market_info,
                    )
                    resolved_now.append(segment_id)
                    break
            for segment_id in resolved_now:
                del unresolved[segment_id]
            print(
                f"market info: qualified={len(schedules)}/{len(segments)} "
                f"unresolved={len(unresolved)}",
                flush=True,
            )
        if unresolved:
            with httpx.Client(
                base_url=data_api_url.rstrip("/"),
                timeout=30.0,
                headers={"User-Agent": "sphinx-prediction-lab-h016/1.0"},
            ) as client:
                if not _fetch_missing_market_trades(
                    client,
                    market_trade_cache,
                    [segment.condition_id for segment in unresolved.values()],
                    workers=workers,
                    requests_per_second=market_trade_requests_per_second,
                    output_dir=output_dir,
                ):
                    return {
                        "status": "paused",
                        "qualified_segments": len(schedules),
                        "unresolved_segments": len(unresolved),
                    }
            candidates_by_segment: dict[
                str,
                list[tuple[FeeSourceCandidate, dict[str, Any]]],
            ] = {}
            for segment_id, segment in unresolved.items():
                market_rows = market_trade_cache.get(segment.condition_id)
                if market_rows is None:
                    raise RuntimeError("H016 Data API market trades disappeared from cache")
                candidates_by_segment[segment_id] = _market_wide_candidates(
                    segment,
                    market_rows,
                )
            with PolygonRPC(rpc_url, retries=2) as rpc:
                if not _fetch_missing(
                    rpc,
                    cache,
                    [
                        candidate.transaction_hash
                        for rows in candidates_by_segment.values()
                        for candidate, _ in rows
                    ],
                    batch_size=batch_size,
                    workers=workers,
                    requests_per_second=requests_per_second,
                    output_dir=output_dir,
                ):
                    return {
                        "status": "paused",
                        "qualified_segments": len(schedules),
                        "unresolved_segments": len(unresolved),
                    }
            resolved_now = []
            for segment_id, segment in unresolved.items():
                market_info = market_info_cache.get(segment.condition_id)
                if market_info is None:
                    raise RuntimeError("H016 CLOB market info disappeared from cache")
                consensus_samples: list[
                    tuple[
                        FeeSourceCandidate,
                        OnchainFeeEvidence,
                        dict[str, Any],
                        dict[str, Any],
                    ]
                ] = []
                for candidate, trade_row in candidates_by_segment[segment_id]:
                    receipt = cache.get(candidate.transaction_hash)
                    if receipt is None:
                        continue
                    try:
                        evidence = decode_onchain_fee_evidence(
                            receipt,
                            candidate,
                            config.contracts,
                            chain_id=config.chain_id,
                        )
                        if evidence.fee_amount > ZERO and evidence.fee_refund_observed:
                            consensus_samples.append(
                                (candidate, evidence, receipt, trade_row)
                            )
                        try:
                            schedule = infer_fee_schedule(candidate, evidence)
                        except RuntimeError:
                            schedule = infer_fee_schedule_from_market_info(
                                candidate,
                                evidence,
                                market_info,
                            )
                    except (RuntimeError, ValueError, TypeError) as error:
                        failures[segment_id].append(
                            "market-wide+"
                            f"{candidate.transaction_hash}: {type(error).__name__}: {error}"
                        )
                        continue
                    schedule = replace(
                        schedule,
                        source=(
                            "clob_market_info_market_wide_trade_receipt_fee"
                            if schedule.source.startswith("clob_market_info")
                            else "market_wide_trade_receipt_active_taker_fee"
                        ),
                    )
                    schedules[segment_id] = schedule
                    proofs[schedule.schedule_id] = _receipt_proof(receipt, schedule, config)
                    market_trade_proofs[schedule.schedule_id] = _market_trade_proof(
                        schedule,
                        trade_row,
                    )
                    if schedule.source.startswith("clob_market_info"):
                        market_info_proofs[segment.condition_id] = _market_info_proof(
                            segment.condition_id,
                            market_info,
                        )
                    resolved_now.append(segment_id)
                    break
                if segment_id in schedules or len(consensus_samples) < 2:
                    continue
                try:
                    schedule = infer_fee_schedule_consensus(
                        [
                            (candidate, evidence)
                            for candidate, evidence, _, _ in consensus_samples
                        ],
                        market_info,
                    )
                except (RuntimeError, ValueError, TypeError) as error:
                    failures[segment_id].append(
                        f"receipt-consensus: {type(error).__name__}: {error}"
                    )
                    continue
                representative = next(
                    row
                    for row in consensus_samples
                    if row[0].transaction_hash == schedule.transaction_hash
                )
                representative_candidate, _, _, representative_trade = representative
                schedules[segment_id] = schedule
                market_info_proofs[segment.condition_id] = _market_info_proof(
                    segment.condition_id,
                    market_info,
                )
                market_trade_proofs[schedule.schedule_id] = _market_trade_proof(
                    schedule,
                    representative_trade,
                )
                for candidate, evidence, receipt, _ in consensus_samples:
                    proof_schedule = replace(
                        schedule,
                        transaction_hash=candidate.transaction_hash,
                        source_order_hash=evidence.order_hash,
                        source_block_number=evidence.block_number,
                        source_log_index=evidence.log_index,
                        source_price=evidence.price,
                        source_gross_shares=evidence.gross_shares,
                        source_fee_asset=evidence.fee_asset,
                        source_fee_amount=evidence.fee_amount,
                    )
                    proof = _receipt_proof(receipt, proof_schedule, config)
                    if candidate == representative_candidate:
                        proofs[schedule.schedule_id] = proof
                    else:
                        supplemental_receipt_proofs.append(proof)
                resolved_now.append(segment_id)
            for segment_id in resolved_now:
                del unresolved[segment_id]
            print(
                f"market-wide receipts: qualified={len(schedules)}/{len(segments)} "
                f"unresolved={len(unresolved)}",
                flush=True,
            )
    finally:
        cache.close()
        market_info_cache.close()
        market_trade_cache.close()
    if unresolved:
        failure_payload = {
            "schema_version": "1.0.0",
            "record_type": "h016_fee_schedule_failures",
            "generated_at": now_utc(),
            "unresolved_segments": len(unresolved),
            "unresolved_segment_ids": sorted(unresolved),
            "unresolved_failures": {
                key: failures.get(key, []) for key in sorted(unresolved)
            },
            "failures": {key: value for key, value in sorted(failures.items())},
        }
        atomic_json(output_dir / "failures.json", failure_payload)
        raise RuntimeError(f"H016 could not qualify {len(unresolved)} fee segments")

    ordered = [schedules[segment.segment_id] for segment in segments]
    data_path = output_dir / "schedules.jsonl.zst"
    proof_path = output_dir / "source-receipts.jsonl.zst"
    market_info_path = output_dir / "source-market-info.jsonl.zst"
    market_trade_path = output_dir / "source-market-trades.jsonl.zst"
    rows, _ = write_jsonl_zst(data_path, (fee_schedule_payload(row) for row in ordered))
    receipt_proof_payloads = [*proofs.values(), *supplemental_receipt_proofs]
    receipt_proof_payloads.sort(
        key=lambda row: (str(row["schedule_id"]), str(row["transaction_hash"]))
    )
    proof_rows, _ = write_jsonl_zst(proof_path, receipt_proof_payloads)
    market_info_rows, _ = write_jsonl_zst(
        market_info_path,
        (market_info_proofs[key] for key in sorted(market_info_proofs)),
    )
    market_trade_rows, _ = write_jsonl_zst(
        market_trade_path,
        (market_trade_proofs[key] for key in sorted(market_trade_proofs)),
    )
    source_counts: dict[str, int] = defaultdict(int)
    for schedule in ordered:
        source_counts[schedule.source] += 1
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "h016_fee_schedule_manifest",
        "generated_at": now_utc(),
        "valid": True,
        "data_path": data_path.name,
        "data_sha256": sha256_file(data_path),
        "rows": rows,
        "receipt_proof_path": proof_path.name,
        "receipt_proof_sha256": sha256_file(proof_path),
        "receipt_proof_rows": proof_rows,
        "market_info_path": market_info_path.name,
        "market_info_sha256": sha256_file(market_info_path),
        "market_info_rows": market_info_rows,
        "market_trade_path": market_trade_path.name,
        "market_trade_sha256": sha256_file(market_trade_path),
        "market_trade_rows": market_trade_rows,
        "plan_sha256": sha256_file(output_dir / "plan.json"),
        "contract_sha256": plan["contract_sha256"],
        "target_conditions": plan["target_conditions"],
        "target_liquidity_ids": plan["target_liquidity_ids"],
        "source_counts": dict(sorted(source_counts.items())),
        "test_rows_consumed": 0,
        "test_labels_opened": False,
        "evidence_boundary": (
            "Receipt-qualified historical platform-fee schedules for development replay only; "
            "not historical orderbook or forward-profit evidence."
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--tape-dir", type=Path, required=True)
    value.add_argument("--replay-dir", type=Path, action="append", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--corpus-config", type=Path, default=DEFAULT_CORPUS_CONFIG)
    value.add_argument("--rpc-url", default=os.environ.get("POLYGON_RPC_URL", "https://polygon.drpc.org"))
    value.add_argument("--batch-size", type=int, default=20)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--requests-per-second", type=float, default=12.0)
    value.add_argument("--clob-url", default="https://clob.polymarket.com")
    value.add_argument("--market-info-requests-per-second", type=float, default=5.0)
    value.add_argument("--data-api-url", default="https://data-api.polymarket.com")
    value.add_argument("--market-trade-requests-per-second", type=float, default=5.0)
    value.add_argument("--cache-dir", type=Path)
    value.add_argument("--candidates-per-segment", type=int, default=5)
    value.add_argument("--prepare-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        args.batch_size < 1
        or args.workers < 1
        or args.requests_per_second <= 0
        or args.market_info_requests_per_second <= 0
        or args.market_trade_requests_per_second <= 0
        or args.candidates_per_segment < 1
    ):
        raise ValueError("H016 batch and candidate counts must be positive")
    tape_dir = args.tape_dir.resolve()
    replay_dirs = tuple(path.resolve() for path in args.replay_dir)
    output_dir = args.output_dir.resolve()
    corpus_config_path = args.corpus_config.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _source_contract(tape_dir, replay_dirs, corpus_config_path)
    plan_path = output_dir / "plan.json"
    plan = load_json(plan_path)
    if plan:
        if plan.get("contract_sha256") != _stable_hash(contract):
            raise RuntimeError("H016 plan belongs to another source contract")
    else:
        plan = build_plan(
            tape_dir,
            replay_dirs,
            corpus_config_path,
            candidates_per_segment=args.candidates_per_segment,
        )
        atomic_json(plan_path, plan)
    if args.prepare_only:
        summary_keys = (
            "segment_count",
            "target_conditions",
            "target_liquidity_ids",
        )
        print(json.dumps({key: plan[key] for key in summary_keys}, indent=2))
        return
    config = CorpusConfig.load(corpus_config_path, tape_dir.parents[1])
    result = qualify(
        plan,
        config,
        output_dir,
        rpc_url=str(args.rpc_url),
        batch_size=args.batch_size,
        workers=args.workers,
        requests_per_second=args.requests_per_second,
        clob_url=str(args.clob_url),
        market_info_requests_per_second=args.market_info_requests_per_second,
        data_api_url=str(args.data_api_url),
        market_trade_requests_per_second=args.market_trade_requests_per_second,
        cache_dir=None if args.cache_dir is None else args.cache_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
