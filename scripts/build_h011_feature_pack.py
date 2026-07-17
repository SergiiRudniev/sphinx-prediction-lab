"""Build the uncapped causal H011 feature pack from the H009 Chronicle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import orjson
from numpy.typing import NDArray

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
from sphinx_trace.h011_features import H011_FEATURE_NAMES, H011_FEATURE_WIDTH, H011_HLL_REGISTERS
from sphinx_trace.h011_kernel import Kernel, compile_h011_kernel
from sphinx_trace.h011_sources import load_ledger_scope_index

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "trace" / "sphinx_trace_s0_h011_pack_v1.json"
DEFAULT_H009_CONFIG = ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"
SPLIT_CODES = {None: 0, "train": 1, "validation": 2, "calibration": 3, "test": 4}
DEVELOPMENT_SPLITS = frozenset({"train", "validation", "calibration"})
IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src" / "sphinx_trace" / "h011_features.py",
    ROOT / "src" / "sphinx_trace" / "h011_kernel.py",
    ROOT / "src" / "sphinx_trace" / "h011_sources.py",
)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


@dataclass(slots=True)
class StaticIndex:
    wallet_to_id: dict[str, int]
    wallet_count: int
    market_to_id: dict[str, int]
    market_conditions: list[str]
    market_components: NDArray[np.int32]
    market_created: NDArray[np.int64]
    market_end: NDArray[np.int64]
    market_split: NDArray[np.uint8]
    market_label: NDArray[np.float32]
    component_ids: list[str]
    component_market_count: NDArray[np.int32]
    component_neg_risk_count: NDArray[np.int32]
    component_unclosed_count: NDArray[np.int32]
    participant_source_sha256: str
    market_index_sha256: str


@dataclass(slots=True)
class RecurrentState:
    wallet_core: NDArray[np.float64]
    actor_core: NDArray[np.float64]
    market_core: NDArray[np.float64]
    market_probability_ema: NDArray[np.float64]
    market_flow_ema: NDArray[np.float64]
    market_wallet_hll: NDArray[np.uint8]
    market_wallet_aggregate: NDArray[np.float64]
    component_core: NDArray[np.float64]
    component_probability_ema: NDArray[np.float64]
    component_flow_ema: NDArray[np.float64]
    component_market_hll: NDArray[np.uint8]
    universe_core: NDArray[np.float64]
    universe_probability_ema: NDArray[np.float64]
    universe_flow_ema: NDArray[np.float64]
    universe_market_hll: NDArray[np.uint8]
    universe_component_hll: NDArray[np.uint8]

    @classmethod
    def empty(cls, index: StaticIndex) -> RecurrentState:
        wallets = index.wallet_count
        markets = len(index.market_conditions)
        components = len(index.component_ids)
        wallet_core = np.zeros((wallets, 18), dtype=np.float64)
        wallet_core[:, 13] = -1.0
        return cls(
            wallet_core=wallet_core,
            actor_core=np.zeros((wallets, 12), dtype=np.float64),
            market_core=np.zeros((markets, 17), dtype=np.float64),
            market_probability_ema=np.zeros((markets, 5), dtype=np.float64),
            market_flow_ema=np.zeros((markets, 5), dtype=np.float64),
            market_wallet_hll=np.zeros((markets, H011_HLL_REGISTERS), dtype=np.uint8),
            market_wallet_aggregate=np.zeros((markets, 18), dtype=np.float64),
            component_core=np.zeros((components, 13), dtype=np.float64),
            component_probability_ema=np.zeros((components, 3), dtype=np.float64),
            component_flow_ema=np.zeros((components, 3), dtype=np.float64),
            component_market_hll=np.zeros((components, H011_HLL_REGISTERS), dtype=np.uint8),
            universe_core=np.zeros(12, dtype=np.float64),
            universe_probability_ema=np.zeros(5, dtype=np.float64),
            universe_flow_ema=np.zeros(5, dtype=np.float64),
            universe_market_hll=np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
            universe_component_hll=np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
        )

    def arrays(self) -> dict[str, NDArray[Any]]:
        return {
            "wallet_core": self.wallet_core,
            "actor_core": self.actor_core,
            "market_core": self.market_core,
            "market_probability_ema": self.market_probability_ema,
            "market_flow_ema": self.market_flow_ema,
            "market_wallet_hll": self.market_wallet_hll,
            "market_wallet_aggregate": self.market_wallet_aggregate,
            "component_core": self.component_core,
            "component_probability_ema": self.component_probability_ema,
            "component_flow_ema": self.component_flow_ema,
            "component_market_hll": self.component_market_hll,
            "universe_core": self.universe_core,
            "universe_probability_ema": self.universe_probability_ema,
            "universe_flow_ema": self.universe_flow_ema,
            "universe_market_hll": self.universe_market_hll,
            "universe_component_hll": self.universe_component_hll,
        }


@dataclass(frozen=True, slots=True)
class ActorSchedule:
    by_available_date: dict[str, tuple[Path, ...]]
    first_available_date: str | None
    coverage_end_date: str | None
    complete: bool
    manifest_sha256: str | None

    def enabled_on(self, date: str) -> bool:
        return bool(
            self.complete
            and self.first_available_date is not None
            and self.coverage_end_date is not None
            and self.first_available_date <= date < self.coverage_end_date
        )


@dataclass(frozen=True, slots=True)
class ResolutionSchedule:
    by_date: dict[str, tuple[Path, ...]]
    complete: bool
    manifest_sha256: str | None


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _timestamp(value: object) -> int:
    parsed = parse_optional_utc(value)
    return 0 if parsed is None else int(parsed.timestamp())


def _feature_names_hash() -> str:
    return hashlib.sha256(("\n".join(H011_FEATURE_NAMES) + "\n").encode()).hexdigest()


def _save_npy(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _active_conditions(
    data_dir: Path,
    chronicle_dir: Path,
    h009_config: dict[str, Any],
) -> list[str]:
    source = h009_config["sources"]["ledger"]
    scopes, _ = load_ledger_scope_index(
        data_dir,
        chronicle_dir,
        namespace=str(source["namespace"]),
        expected_scopes=int(source["scope_groups"]),
        expected_markets=int(source["markets"]),
        expected_rows=int(source["rows"]),
        source_manifest_sha256=str(source["manifest_sha256"]),
    )
    conditions = sorted({condition for scope in scopes for condition in scope.condition_ids})
    if not conditions:
        raise RuntimeError("H011 found no H009 Ledger scopes")
    return conditions


def _load_wallet_index(chronicle_dir: Path) -> tuple[dict[str, int], str]:
    manifest = _load_object(chronicle_dir / "stream-manifest.json")
    participant = manifest["participants"]
    path = chronicle_dir / str(participant["path"])
    expected_hash = str(participant["sha256"])
    if sha256_file(path) != expected_hash:
        raise RuntimeError("H009 participant index hash changed")
    wallet_to_id: dict[str, int] = {}
    previous = ""
    for row in iter_jsonl_zst(path):
        wallet = str(row.get("wallet") or "").lower()
        if not wallet or wallet <= previous:
            raise RuntimeError("H009 participant index is not strictly sorted")
        wallet_to_id[wallet] = len(wallet_to_id)
        previous = wallet
    if len(wallet_to_id) != int(participant["rows"]):
        raise RuntimeError("H009 participant index row count changed")
    return wallet_to_id, expected_hash


def _build_static_index(
    data_dir: Path,
    chronicle_dir: Path,
    output_dir: Path,
    h009_config: dict[str, Any],
) -> StaticIndex:
    wallet_to_id, participant_hash = _load_wallet_index(chronicle_dir)
    active_conditions = _active_conditions(data_dir, chronicle_dir, h009_config)
    connection = sqlite3.connect(
        f"file:{(chronicle_dir / 'catalog.sqlite').as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TEMP TABLE active_conditions(condition_id TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO active_conditions(condition_id) VALUES (?)",
        ((condition,) for condition in active_conditions),
    )
    rows = list(
        connection.execute(
            """
        SELECT m.condition_id, m.component_id, m.created_at, m.end_at, m.split_id,
               m.terminal_label, m.outcomes, m.market_id
        FROM markets AS m
        INNER JOIN active_conditions AS a ON a.condition_id = m.condition_id
        ORDER BY m.condition_id
        """
        )
    )
    found = {str(row["condition_id"]) for row in rows}
    missing = set(active_conditions) - found
    if missing:
        raise RuntimeError(f"H011 catalog is missing {len(missing)} active conditions")
    component_ids = sorted({str(row["component_id"]) for row in rows})
    component_to_id = {value: index for index, value in enumerate(component_ids)}
    component_static: dict[str, tuple[int, int, int]] = {}
    for row in connection.execute(
        """
        SELECT component_id, market_count, neg_risk_market_count, unclosed_market_count
        FROM components ORDER BY component_id
        """
    ):
        component = str(row["component_id"])
        if component in component_to_id:
            component_static[component] = (
                int(row["market_count"]),
                int(row["neg_risk_market_count"]),
                int(row["unclosed_market_count"]),
            )
    connection.close()
    if len(component_static) != len(component_ids):
        raise RuntimeError("H011 could not load every active component")

    market_conditions = [str(row["condition_id"]) for row in rows]
    market_to_id = {value: index for index, value in enumerate(market_conditions)}
    market_components = np.empty(len(rows), dtype=np.int32)
    market_created = np.zeros(len(rows), dtype=np.int64)
    market_end = np.zeros(len(rows), dtype=np.int64)
    market_split = np.zeros(len(rows), dtype=np.uint8)
    market_label = np.full(len(rows), np.nan, dtype=np.float32)
    market_index_rows: list[dict[str, Any]] = []
    for market_id, row in enumerate(rows):
        condition = str(row["condition_id"])
        component = str(row["component_id"])
        split_value = row["split_id"]
        split = None if split_value is None else str(split_value)
        if split not in SPLIT_CODES:
            raise RuntimeError(f"Unexpected H009 split: {split}")
        market_components[market_id] = component_to_id[component]
        market_created[market_id] = _timestamp(row["created_at"])
        market_end[market_id] = _timestamp(row["end_at"])
        market_split[market_id] = SPLIT_CODES[split]
        terminal_raw = row["terminal_label"]
        if split == "test" and terminal_raw is not None:
            raise RuntimeError("H009 test terminal label was unexpectedly materialized")
        if split in DEVELOPMENT_SPLITS and terminal_raw is not None:
            payout: object = json.loads(str(terminal_raw))
            if not isinstance(payout, list) or len(payout) != 2:
                raise RuntimeError("H009 terminal label has an invalid shape")
            market_label[market_id] = float(payout[0])
        market_index_rows.append(
            {
                "market_id": market_id,
                "condition_id": condition,
                "component_id": component,
                "component_index": int(market_components[market_id]),
                "split": split,
                "outcomes": json.loads(str(row["outcomes"])),
                "gamma_market_id": str(row["market_id"]),
                "created_at_unix": int(market_created[market_id]),
                "end_at_unix": int(market_end[market_id]),
            }
        )
    index_path = output_dir / "index" / "markets.jsonl.zst"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_rows, _ = write_jsonl_zst(index_path, market_index_rows)
    if index_rows != len(rows):
        raise RuntimeError("H011 market index lost rows")
    index_hash = sha256_file(index_path)
    component_values = [component_static[value] for value in component_ids]
    return StaticIndex(
        wallet_to_id=wallet_to_id,
        wallet_count=len(wallet_to_id),
        market_to_id=market_to_id,
        market_conditions=market_conditions,
        market_components=market_components,
        market_created=market_created,
        market_end=market_end,
        market_split=market_split,
        market_label=market_label,
        component_ids=component_ids,
        component_market_count=np.asarray([value[0] for value in component_values], dtype=np.int32),
        component_neg_risk_count=np.asarray(
            [value[1] for value in component_values], dtype=np.int32
        ),
        component_unclosed_count=np.asarray(
            [value[2] for value in component_values], dtype=np.int32
        ),
        participant_source_sha256=participant_hash,
        market_index_sha256=index_hash,
    )


def _actor_schedule(actor_dir: Path | None) -> ActorSchedule:
    if actor_dir is None or not (actor_dir / "manifest.json").exists():
        return ActorSchedule({}, None, None, False, None)
    manifest_path = actor_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    schedule: defaultdict[str, list[Path]] = defaultdict(list)
    for task in manifest.get("tasks", []):
        if not isinstance(task, dict):
            raise RuntimeError("Actor manifest task is not an object")
        schedule[str(task["window_end_exclusive"])[:10]].append(actor_dir / str(task["path"]))
    first = min(schedule) if schedule else None
    return ActorSchedule(
        by_available_date={key: tuple(sorted(value)) for key, value in schedule.items()},
        first_available_date=first,
        coverage_end_date=str(manifest["end_exclusive"])[:10],
        complete=bool(manifest.get("complete")),
        manifest_sha256=sha256_file(manifest_path),
    )


def _resolution_schedule(resolution_dir: Path | None) -> ResolutionSchedule:
    if resolution_dir is None or not (resolution_dir / "manifest.json").exists():
        return ResolutionSchedule({}, False, None)
    manifest_path = resolution_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    schedule: defaultdict[str, list[Path]] = defaultdict(list)
    for shard in manifest.get("shards", []):
        if not isinstance(shard, dict):
            raise RuntimeError("Resolution manifest shard is not an object")
        schedule[str(shard["date"])].append(resolution_dir / str(shard["path"]))
    return ResolutionSchedule(
        by_date={key: tuple(sorted(value)) for key, value in schedule.items()},
        complete=bool(manifest.get("complete")),
        manifest_sha256=sha256_file(manifest_path),
    )


def _apply_actor_delta(
    paths: Iterable[Path],
    index: StaticIndex,
    actor_core: NDArray[np.float64],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        for row in iter_jsonl_zst(path):
            counts["source_rows"] += 1
            wallet_id = index.wallet_to_id.get(str(row.get("wallet") or "").lower())
            if wallet_id is None:
                counts["non_ledger_actor_rows"] += 1
                continue
            maker = float(row["maker_fills"])
            taker = float(row["taker_fills"])
            fills = maker + taker
            mean_price = float(row["average_price"])
            price_std = float(row["price_std"])
            actor_core[wallet_id, 0] += maker
            actor_core[wallet_id, 1] += taker
            actor_core[wallet_id, 2] += float(row["buy_fills"])
            actor_core[wallet_id, 3] += float(row["sell_fills"])
            actor_core[wallet_id, 4] += float(row["buy_notional_usd"])
            actor_core[wallet_id, 5] += float(row["sell_notional_usd"])
            actor_core[wallet_id, 6] += float(row["shares"])
            actor_core[wallet_id, 7] += float(row["counterparties"])
            actor_core[wallet_id, 8] += float(row["assets"])
            actor_core[wallet_id, 9] += mean_price * fills
            actor_core[wallet_id, 10] += (price_std * price_std + mean_price * mean_price) * fills
            actor_core[wallet_id, 11] += fills
            counts["ledger_actor_rows"] += 1
    return dict(counts)


def _resolution_arrays(
    paths: Iterable[Path],
    index: StaticIndex,
) -> dict[str, NDArray[Any]]:
    timestamps: list[int] = []
    wallet_ids: list[int] = []
    edges: list[float] = []
    pnls: list[float] = []
    wins: list[int] = []
    previous_key: tuple[int, str, str] | None = None
    for path in paths:
        for row in iter_jsonl_zst(path):
            timestamp = int(row["resolution_time_unix"])
            condition = str(row["condition_id"])
            wallet = str(row["wallet"]).lower()
            key = (timestamp, condition, wallet)
            if previous_key is not None and key < previous_key:
                raise RuntimeError("H011 resolution events are not globally ordered")
            previous_key = key
            wallet_id = index.wallet_to_id.get(wallet)
            if wallet_id is None:
                raise RuntimeError("H011 resolution event has an unmapped Ledger wallet")
            timestamps.append(timestamp)
            wallet_ids.append(wallet_id)
            edges.append(float(row["directional_edge"]))
            pnls.append(float(row["pnl_proxy_usd"]))
            wins.append(int(bool(row["profitable_proxy"])))
    return {
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "wallet_ids": np.asarray(wallet_ids, dtype=np.int32),
        "edges": np.asarray(edges, dtype=np.float32),
        "pnls": np.asarray(pnls, dtype=np.float32),
        "wins": np.asarray(wins, dtype=np.int8),
    }


def _apply_remaining_resolutions(
    state: RecurrentState,
    resolution: dict[str, NDArray[Any]],
    start: int,
) -> int:
    wallet_ids = resolution["wallet_ids"][start:]
    if not len(wallet_ids):
        return 0
    np.add.at(state.wallet_core[:, 14], wallet_ids, 1.0)
    np.add.at(state.wallet_core[:, 15], wallet_ids, resolution["edges"][start:])
    np.add.at(state.wallet_core[:, 16], wallet_ids, resolution["pnls"][start:])
    np.add.at(state.wallet_core[:, 17], wallet_ids, resolution["wins"][start:])
    return len(wallet_ids)


def _checkpoint_arrays(path: Path, state: RecurrentState) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez(handle, **state.arrays())  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_checkpoint(
    output_dir: Path,
    config_hash: str,
    implementation_hash: str,
    stream_scope_digest: str,
    stream_manifest_sha256: str,
    decision_manifest_sha256: str,
    actor_manifest_sha256: str | None,
    resolution_manifest_sha256: str | None,
    index: StaticIndex,
    state: RecurrentState,
    last_date: str,
) -> dict[str, Any]:
    state_path = output_dir / "state" / "latest.npz"
    _checkpoint_arrays(state_path, state)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_feature_state_checkpoint",
        "generated_at": now_utc(),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "stream_scope_digest": stream_scope_digest,
        "stream_manifest_sha256": stream_manifest_sha256,
        "decision_manifest_sha256": decision_manifest_sha256,
        "actor_manifest_sha256": actor_manifest_sha256,
        "resolution_manifest_sha256": resolution_manifest_sha256,
        "participant_source_sha256": index.participant_source_sha256,
        "market_index_sha256": index.market_index_sha256,
        "wallets": index.wallet_count,
        "markets": len(index.market_conditions),
        "components": len(index.component_ids),
        "last_date": last_date,
        "path": state_path.relative_to(output_dir).as_posix(),
        "bytes": state_path.stat().st_size,
        "sha256": sha256_file(state_path),
    }
    atomic_json(output_dir / "state" / "latest.json", receipt)
    return receipt


def _restore_checkpoint(
    output_dir: Path,
    config_hash: str,
    implementation_hash: str,
    stream_scope_digest: str,
    stream_manifest_sha256: str,
    decision_manifest_sha256: str,
    actor_manifest_sha256: str | None,
    resolution_manifest_sha256: str | None,
    index: StaticIndex,
) -> tuple[RecurrentState, str | None]:
    receipt_path = output_dir / "state" / "latest.json"
    if not receipt_path.exists():
        return RecurrentState.empty(index), None
    receipt = _load_object(receipt_path)
    expected = {
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "stream_scope_digest": stream_scope_digest,
        "stream_manifest_sha256": stream_manifest_sha256,
        "decision_manifest_sha256": decision_manifest_sha256,
        "actor_manifest_sha256": actor_manifest_sha256,
        "resolution_manifest_sha256": resolution_manifest_sha256,
        "participant_source_sha256": index.participant_source_sha256,
        "market_index_sha256": index.market_index_sha256,
        "wallets": index.wallet_count,
        "markets": len(index.market_conditions),
        "components": len(index.component_ids),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("H011 feature checkpoint belongs to another source contract")
    path = output_dir / str(receipt["path"])
    if sha256_file(path) != receipt["sha256"]:
        raise RuntimeError("H011 feature checkpoint hash changed")
    empty = RecurrentState.empty(index)
    with np.load(path, allow_pickle=False) as archive:
        for name, target in empty.arrays().items():
            source = archive[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise RuntimeError(f"H011 state array changed shape or dtype: {name}")
            target[...] = source
    return empty, str(receipt["last_date"])


def _decisions(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl_zst(path))
    previous = -1
    for row in rows:
        stream_row = int(row["stream_row"])
        if stream_row <= previous:
            raise RuntimeError("H009 decisions are not strictly stream-row ordered")
        previous = stream_row
    return rows


def _kernel_call(
    kernel: Kernel,
    count: int,
    inputs: dict[str, NDArray[Any]],
    actor_enabled: bool,
    resolution: dict[str, NDArray[Any]],
    resolution_pointer: int,
    index: StaticIndex,
    state: RecurrentState,
    output_features: NDArray[np.float32],
) -> int:
    return kernel(
        inputs["timestamps"][:count],
        inputs["wallet_ids"][:count],
        inputs["market_ids"][:count],
        inputs["probabilities"][:count],
        inputs["raw_prices"][:count],
        inputs["sizes"][:count],
        inputs["notionals"][:count],
        inputs["outcomes"][:count],
        inputs["side_buys"][:count],
        inputs["directions"][:count],
        inputs["decision_slots"][:count],
        index.market_components,
        index.market_created,
        index.market_end,
        index.component_market_count,
        index.component_neg_risk_count,
        index.component_unclosed_count,
        int(actor_enabled),
        resolution["timestamps"],
        resolution["wallet_ids"],
        resolution["edges"],
        resolution["pnls"],
        resolution["wins"],
        resolution_pointer,
        state.wallet_core,
        state.actor_core,
        state.market_core,
        state.market_probability_ema,
        state.market_flow_ema,
        state.market_wallet_hll,
        state.market_wallet_aggregate,
        state.component_core,
        state.component_probability_ema,
        state.component_flow_ema,
        state.component_market_hll,
        state.universe_core,
        state.universe_probability_ema,
        state.universe_flow_ema,
        state.universe_market_hll,
        state.universe_component_hll,
        output_features,
    )


def _input_arrays(chunk_rows: int) -> dict[str, NDArray[Any]]:
    return {
        "timestamps": np.empty(chunk_rows, dtype=np.int64),
        "wallet_ids": np.empty(chunk_rows, dtype=np.int32),
        "market_ids": np.empty(chunk_rows, dtype=np.int32),
        "probabilities": np.empty(chunk_rows, dtype=np.float32),
        "raw_prices": np.empty(chunk_rows, dtype=np.float32),
        "sizes": np.empty(chunk_rows, dtype=np.float32),
        "notionals": np.empty(chunk_rows, dtype=np.float32),
        "outcomes": np.empty(chunk_rows, dtype=np.int8),
        "side_buys": np.empty(chunk_rows, dtype=np.int8),
        "directions": np.empty(chunk_rows, dtype=np.int8),
        "decision_slots": np.full(chunk_rows, -1, dtype=np.int32),
    }


def _parse_trade(
    line: bytes,
    index: StaticIndex,
) -> tuple[str, str, int, int, int, float, float, float, int, int, int, float, int]:
    payload: object = orjson.loads(line)
    if not isinstance(payload, dict):
        raise TypeError("H009 Ledger row is not an object")
    trade_id = str(payload.get("trade_id") or "")
    condition = str(payload.get("condition_id") or "").lower()
    wallet = str(payload.get("wallet") or "").lower()
    wallet_id = index.wallet_to_id.get(wallet)
    market_id = index.market_to_id.get(condition)
    if wallet_id is None or market_id is None:
        missing = "wallet" if wallet_id is None else "market"
        raise RuntimeError(f"H011 encountered an unmapped {missing}")
    timestamp = int(payload["timestamp_unix"])
    raw_price = float(payload["price"])
    size = float(payload["size"])
    notional = float(payload["notional_usd"])
    outcome = int(payload["outcome_index"])
    side_text = str(payload.get("side") or "").upper()
    invalid: list[str] = []
    if not trade_id:
        invalid.append("trade_id")
    if outcome not in {0, 1}:
        invalid.append("outcome_index")
    if side_text not in {"BUY", "SELL"}:
        invalid.append("side")
    if not math.isfinite(raw_price):
        invalid.append("price_non_finite")
    if not math.isfinite(size) or size <= 0.0:
        invalid.append("size")
    if not math.isfinite(notional) or notional <= 0.0:
        invalid.append("notional_usd")
    if invalid:
        raise RuntimeError(
            "H011 encountered an invalid normalized Ledger row: "
            f"fields={','.join(invalid)} trade_id={trade_id} condition_id={condition} "
            f"price={raw_price!r} size={size!r} notional_usd={notional!r}"
        )
    source_price_anomaly = int(not 0.0 <= raw_price <= 1.0)
    model_price = min(1.0, max(0.0, raw_price))
    side_buy = int(side_text == "BUY")
    direction = 1 if side_buy == int(outcome == 0) else -1
    probability = model_price if outcome == 0 else 1.0 - model_price
    return (
        trade_id,
        wallet,
        wallet_id,
        market_id,
        timestamp,
        probability,
        model_price,
        size,
        outcome,
        side_buy,
        direction,
        notional,
        source_price_anomaly,
    )


def _write_day(
    output_dir: Path,
    date: str,
    features: NDArray[np.float32],
    labels: NDArray[np.float32],
    label_mask: NDArray[np.uint8],
    baselines: NDArray[np.float32],
    split_codes: NDArray[np.uint8],
    market_ids: NDArray[np.int32],
    component_ids: NDArray[np.int32],
    wallet_ids: NDArray[np.int32],
    timestamps: NDArray[np.int64],
    debug_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = output_dir / "shards"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"date={date}.tmp-{uuid.uuid4().hex}"
    final = root / f"date={date}"
    if final.exists():
        raise RuntimeError(f"H011 day already exists without a receipt: {date}")
    temporary.mkdir(parents=True)
    arrays: dict[str, NDArray[Any]] = {
        "features.npy": features,
        "labels.npy": labels,
        "label_mask.npy": label_mask,
        "baselines.npy": baselines,
        "split_codes.npy": split_codes,
        "market_ids.npy": market_ids,
        "component_ids.npy": component_ids,
        "wallet_ids.npy": wallet_ids,
        "timestamps.npy": timestamps,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        path = temporary / name
        _save_npy(path, array)
        files[name] = {
            "path": f"shards/date={date}/{name}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    debug_path = temporary / "examples.jsonl.zst"
    debug_count, _ = write_jsonl_zst(debug_path, debug_rows)
    if debug_count != len(features):
        raise RuntimeError("H011 debug rows do not align with feature rows")
    files[debug_path.name] = {
        "path": f"shards/date={date}/{debug_path.name}",
        "bytes": debug_path.stat().st_size,
        "sha256": sha256_file(debug_path),
        "rows": debug_count,
    }
    os.replace(temporary, final)
    return final, files


def _process_day(
    *,
    date: str,
    stream_path: Path,
    expected_stream_rows: int,
    decision_path: Path | None,
    index: StaticIndex,
    state: RecurrentState,
    kernel: Kernel,
    chunk_rows: int,
    actor_enabled: bool,
    resolution: dict[str, NDArray[Any]],
    output_dir: Path,
    emit: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    decisions = _decisions(decision_path) if emit and decision_path is not None else []
    features = np.zeros((len(decisions), H011_FEATURE_WIDTH), dtype=np.float32)
    evidence_market_ids = np.full(len(decisions), -1, dtype=np.int32)
    evidence_wallet_ids = np.full(len(decisions), -1, dtype=np.int32)
    evidence_timestamps = np.zeros(len(decisions), dtype=np.int64)
    evidence_wallets = [""] * len(decisions)
    inputs = _input_arrays(chunk_rows)
    decision_pointer = 0
    resolution_pointer = 0
    stream_row = 0
    chunk_count = 0
    source_price_anomaly_rows = 0
    reader: BinaryIO = raw_jsonl_zst_lines(stream_path)
    try:
        for line in reader:
            (
                trade_id,
                wallet,
                wallet_id,
                market_id,
                timestamp_value,
                probability,
                raw_price,
                size,
                outcome,
                side_buy,
                direction,
                notional,
                source_price_anomaly,
            ) = _parse_trade(line, index)
            source_price_anomaly_rows += source_price_anomaly
            timestamp = int(timestamp_value)
            slot = -1
            if decision_pointer < len(decisions):
                decision = decisions[decision_pointer]
                target_row = int(decision["stream_row"])
                if stream_row > target_row:
                    raise RuntimeError("H011 passed an unmatched decision cursor")
                if stream_row == target_row:
                    if str(decision["evidence_trade_id"]) != trade_id:
                        raise RuntimeError("H011 decision evidence trade ID mismatch")
                    slot = decision_pointer
                    evidence_market_ids[slot] = market_id
                    evidence_wallet_ids[slot] = wallet_id
                    evidence_timestamps[slot] = timestamp
                    evidence_wallets[slot] = wallet
                    decision_pointer += 1
            offset = chunk_count
            inputs["timestamps"][offset] = timestamp
            inputs["wallet_ids"][offset] = wallet_id
            inputs["market_ids"][offset] = market_id
            inputs["probabilities"][offset] = probability
            inputs["raw_prices"][offset] = raw_price
            inputs["sizes"][offset] = size
            inputs["notionals"][offset] = notional
            inputs["outcomes"][offset] = outcome
            inputs["side_buys"][offset] = side_buy
            inputs["directions"][offset] = direction
            inputs["decision_slots"][offset] = slot
            chunk_count += 1
            stream_row += 1
            if chunk_count == chunk_rows:
                resolution_pointer = _kernel_call(
                    kernel,
                    chunk_count,
                    inputs,
                    actor_enabled,
                    resolution,
                    resolution_pointer,
                    index,
                    state,
                    features,
                )
                inputs["decision_slots"][:chunk_count] = -1
                chunk_count = 0
        if chunk_count:
            resolution_pointer = _kernel_call(
                kernel,
                chunk_count,
                inputs,
                actor_enabled,
                resolution,
                resolution_pointer,
                index,
                state,
                features,
            )
    finally:
        reader.close()
    if stream_row != expected_stream_rows:
        raise RuntimeError(f"H011 stream row count changed for {date}")
    if decision_pointer != len(decisions):
        raise RuntimeError(f"H011 did not match every decision for {date}")
    deferred_resolution_events = _apply_remaining_resolutions(
        state,
        resolution,
        resolution_pointer,
    )
    resolution_events = len(resolution["timestamps"])
    if not emit:
        return {
            "date": date,
            "stream_rows": stream_row,
            "rows": 0,
            "resolution_events": resolution_events,
            "resolution_events_applied_in_stream": resolution_pointer,
            "resolution_events_deferred_until_day_end": deferred_resolution_events,
            "source_price_anomaly_rows": source_price_anomaly_rows,
            "replay_only": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    if np.any(evidence_market_ids < 0) or np.any(evidence_wallet_ids < 0):
        raise RuntimeError("H011 emitted an unmapped decision identity")
    non_finite = int(np.count_nonzero(~np.isfinite(features)))
    if non_finite:
        raise RuntimeError(f"H011 generated {non_finite} non-finite features")
    split_codes = index.market_split[evidence_market_ids]
    labels = index.market_label[evidence_market_ids]
    label_mask = np.isfinite(labels).astype(np.uint8)
    labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)
    if int(label_mask[split_codes == SPLIT_CODES["test"]].sum()) != 0:
        raise RuntimeError("H011 emitted test labels")
    baselines = features[:, 11].copy()
    evidence_component_ids = index.market_components[evidence_market_ids]
    debug_rows = [
        {
            "schema_version": "1.0.0",
            "record_type": "h011_feature_example",
            "decision_id": str(decision["decision_id"]),
            "decision_time_unix": int(decision["decision_time_unix"]),
            "evidence_trade_id": str(decision["evidence_trade_id"]),
            "condition_id": index.market_conditions[int(evidence_market_ids[slot])],
            "component_id": str(decision["component_id"]),
            "wallet": evidence_wallets[slot],
            "wallet_state_id": int(evidence_wallet_ids[slot]),
            "market_state_id": int(evidence_market_ids[slot]),
            "split_code": int(split_codes[slot]),
            "label_available": bool(label_mask[slot]),
            "feature_max_event_time_unix": int(decision["feature_max_event_time_unix"]),
            "same_second_ordering": "deterministic_without_causal_precedence_claim",
        }
        for slot, decision in enumerate(decisions)
    ]
    _, files = _write_day(
        output_dir,
        date,
        features,
        labels,
        label_mask,
        baselines,
        split_codes,
        evidence_market_ids,
        evidence_component_ids,
        evidence_wallet_ids,
        evidence_timestamps,
        debug_rows,
    )
    counts = Counter(int(value) for value in split_codes)
    return {
        "schema_version": "1.0.0",
        "record_type": "h011_feature_day_receipt",
        "generated_at": now_utc(),
        "date": date,
        "stream_rows": stream_row,
        "rows": len(decisions),
        "rows_by_split_code": {str(key): value for key, value in sorted(counts.items())},
        "labeled_rows": int(label_mask.sum()),
        "test_label_rows": int(label_mask[split_codes == SPLIT_CODES["test"]].sum()),
        "non_finite_features": non_finite,
        "actor_channel_enabled": actor_enabled,
        "resolution_events": resolution_events,
        "resolution_events_applied_in_stream": resolution_pointer,
        "resolution_events_deferred_until_day_end": deferred_resolution_events,
        "source_price_anomaly_rows": source_price_anomaly_rows,
        "files": files,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _normalization(
    output_dir: Path,
    receipts: list[dict[str, Any]],
    *,
    bins: int = 4096,
) -> dict[str, Any]:
    minimum = np.full(H011_FEATURE_WIDTH, np.inf, dtype=np.float64)
    maximum = np.full(H011_FEATURE_WIDTH, -np.inf, dtype=np.float64)
    train_rows = 0
    for receipt in receipts:
        date = str(receipt["date"])
        root = output_dir / "shards" / f"date={date}"
        split = np.load(root / "split_codes.npy", mmap_mode="r")
        selected = np.flatnonzero(split == SPLIT_CODES["train"])
        if not len(selected):
            continue
        features = np.load(root / "features.npy", mmap_mode="r")
        values = np.asarray(features[selected], dtype=np.float32)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        train_rows += len(selected)
    if train_rows == 0:
        raise RuntimeError("H011 cannot fit normalization without train rows")
    spans = maximum - minimum
    spans[spans <= 0.0] = 1.0
    histograms = np.zeros((H011_FEATURE_WIDTH, bins), dtype=np.int64)
    for receipt in receipts:
        date = str(receipt["date"])
        root = output_dir / "shards" / f"date={date}"
        split = np.load(root / "split_codes.npy", mmap_mode="r")
        selected = np.flatnonzero(split == SPLIT_CODES["train"])
        if not len(selected):
            continue
        features = np.load(root / "features.npy", mmap_mode="r")
        values = np.asarray(features[selected], dtype=np.float64)
        scaled = np.floor((values - minimum) / spans * (bins - 1)).astype(np.int64)
        np.clip(scaled, 0, bins - 1, out=scaled)
        for feature in range(H011_FEATURE_WIDTH):
            histograms[feature] += np.bincount(scaled[:, feature], minlength=bins)

    def quantile(level: float) -> NDArray[np.float64]:
        output = np.empty(H011_FEATURE_WIDTH, dtype=np.float64)
        target = math.ceil(level * train_rows)
        for feature in range(H011_FEATURE_WIDTH):
            cumulative = np.cumsum(histograms[feature])
            bucket = int(np.searchsorted(cumulative, target, side="left"))
            output[feature] = minimum[feature] + (bucket + 0.5) / bins * spans[feature]
        return output

    q25 = quantile(0.25)
    median = quantile(0.5)
    q75 = quantile(0.75)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    path = output_dir / "normalization.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            minimum=minimum,
            maximum=maximum,
            q25=q25,
            median=median,
            q75=q75,
            scale=scale,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    normalization_receipt: dict[str, Any] = {
        "method": "train_only_deterministic_histogram_median_iqr",
        "bins": bins,
        "train_rows": train_rows,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    atomic_json(output_dir / "normalization.json", normalization_receipt)
    return normalization_receipt


def build_feature_pack(
    config_path: Path,
    h009_config_path: Path,
    data_dir: Path,
    chronicle_dir: Path,
    actor_dir: Path | None,
    resolution_dir: Path | None,
    output_dir: Path,
    *,
    chunk_rows: int,
    day_limit: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    h009_config = load_json(h009_config_path)
    config_hash = sha256_file(config_path)
    implementation_hash = _implementation_hash()
    if _feature_names_hash() != config["features"]["names_sha256"]:
        raise RuntimeError("H011 feature names changed after registration")
    if chunk_rows <= 0 or (day_limit is not None and day_limit <= 0):
        raise ValueError("H011 chunk rows and day limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = (
        _load_object(output_dir / "manifest.json")
        if (output_dir / "manifest.json").exists()
        else None
    )
    check_disk_reserve(output_dir, float(config["storage"]["minimum_free_gib"]))
    stream_manifest = _load_object(chronicle_dir / "stream-manifest.json")
    decision_manifest = _load_object(chronicle_dir / "decision-manifest.json")
    stream_manifest_hash = sha256_file(chronicle_dir / "stream-manifest.json")
    decision_manifest_hash = sha256_file(chronicle_dir / "decision-manifest.json")
    if stream_manifest["scope_digest"] != decision_manifest["scope_digest"]:
        raise RuntimeError("H009 stream and decision manifests use different scopes")
    if int(stream_manifest["rows"]) != int(decision_manifest["stream_rows"]):
        raise RuntimeError("H009 decisions do not cover the full stream")
    index = _build_static_index(data_dir, chronicle_dir, output_dir, h009_config)
    schedule = _actor_schedule(actor_dir)
    resolution_schedule = _resolution_schedule(resolution_dir)
    state, checkpoint_date = _restore_checkpoint(
        output_dir,
        config_hash,
        implementation_hash,
        str(stream_manifest["scope_digest"]),
        stream_manifest_hash,
        decision_manifest_hash,
        schedule.manifest_sha256,
        resolution_schedule.manifest_sha256,
        index,
    )
    kernel = compile_h011_kernel()
    decision_by_date = {str(row["date"]): row for row in decision_manifest["shards"]}
    selected_shards = list(stream_manifest["shards"])
    if day_limit is not None:
        selected_shards = selected_shards[:day_limit]
    full_run = day_limit is None and bool(stream_manifest["full_scope_set"])
    receipts: list[dict[str, Any]] = []
    actor_counts: Counter[str] = Counter()
    resolution_dates = sorted(resolution_schedule.by_date)
    resolution_cursor = 0
    while (
        checkpoint_date is not None
        and resolution_cursor < len(resolution_dates)
        and resolution_dates[resolution_cursor] <= checkpoint_date
    ):
        resolution_cursor += 1
    last_checkpoint = time.monotonic()
    last_date = checkpoint_date
    for stream_shard in selected_shards:
        date = str(stream_shard["date"])
        if (output_dir / "PAUSE").exists():
            raise InterruptedError("H011 feature pack paused; remove PAUSE to resume")
        receipt_path = output_dir / "receipts" / f"date={date}.json"
        cached = _load_object(receipt_path) if receipt_path.exists() else None
        if cached is not None and (
            cached.get("config_sha256") != config_hash
            or cached.get("implementation_sha256") != implementation_hash
        ):
            raise RuntimeError("H011 daily receipt belongs to another implementation contract")
        if checkpoint_date is not None and date <= checkpoint_date:
            if cached is None:
                raise RuntimeError("H011 checkpoint is ahead of its daily receipts")
            receipts.append(cached)
            last_date = date
            continue
        actor_update = _apply_actor_delta(
            schedule.by_available_date.get(date, ()), index, state.actor_core
        )
        actor_counts.update(actor_update)
        resolution_paths: list[Path] = []
        while (
            resolution_cursor < len(resolution_dates)
            and resolution_dates[resolution_cursor] <= date
        ):
            resolution_paths.extend(
                resolution_schedule.by_date[resolution_dates[resolution_cursor]]
            )
            resolution_cursor += 1
        resolution = _resolution_arrays(resolution_paths, index)
        stream_path = chronicle_dir / str(stream_shard["path"])
        if cached is not None:
            replay = _process_day(
                date=date,
                stream_path=stream_path,
                expected_stream_rows=int(stream_shard["rows"]),
                decision_path=None,
                index=index,
                state=state,
                kernel=kernel,
                chunk_rows=chunk_rows,
                actor_enabled=schedule.enabled_on(date),
                resolution=resolution,
                output_dir=output_dir,
                emit=False,
            )
            if int(replay["stream_rows"]) != int(cached["stream_rows"]):
                raise RuntimeError("H011 deterministic replay changed a day row count")
            if int(replay["source_price_anomaly_rows"]) != int(cached["source_price_anomaly_rows"]):
                raise RuntimeError("H011 deterministic replay changed source-price anomalies")
            receipts.append(cached)
        else:
            decision_shard = decision_by_date.get(date)
            if decision_shard is None:
                raise RuntimeError(f"H009 decision manifest has no day {date}")
            receipt = _process_day(
                date=date,
                stream_path=stream_path,
                expected_stream_rows=int(stream_shard["rows"]),
                decision_path=chronicle_dir / str(decision_shard["path"]),
                index=index,
                state=state,
                kernel=kernel,
                chunk_rows=chunk_rows,
                actor_enabled=schedule.enabled_on(date),
                resolution=resolution,
                output_dir=output_dir,
                emit=True,
            )
            receipt["config_sha256"] = config_hash
            receipt["implementation_sha256"] = implementation_hash
            atomic_json(receipt_path, receipt)
            receipts.append(receipt)
        last_date = date
        if time.monotonic() - last_checkpoint >= int(
            config["storage"]["checkpoint_maximum_interval_seconds"]
        ):
            _save_checkpoint(
                output_dir,
                config_hash,
                implementation_hash,
                str(stream_manifest["scope_digest"]),
                stream_manifest_hash,
                decision_manifest_hash,
                schedule.manifest_sha256,
                resolution_schedule.manifest_sha256,
                index,
                state,
                date,
            )
            checkpoint_date = date
            last_checkpoint = time.monotonic()
    if last_date is None:
        raise RuntimeError("H011 selected no stream days")
    checkpoint = _save_checkpoint(
        output_dir,
        config_hash,
        implementation_hash,
        str(stream_manifest["scope_digest"]),
        stream_manifest_hash,
        decision_manifest_hash,
        schedule.manifest_sha256,
        resolution_schedule.manifest_sha256,
        index,
        state,
        last_date,
    )
    normalization = _normalization(output_dir, receipts)
    stream_rows = sum(int(row["stream_rows"]) for row in receipts)
    decision_rows = sum(int(row["rows"]) for row in receipts)
    test_label_rows = sum(int(row["test_label_rows"]) for row in receipts)
    non_finite = sum(int(row["non_finite_features"]) for row in receipts)
    source_price_anomaly_rows = sum(int(row["source_price_anomaly_rows"]) for row in receipts)
    resolution_totals: Counter[str] = Counter()
    for receipt in receipts:
        resolution_totals["events"] += int(receipt.get("resolution_events", 0))
        resolution_totals["applied_in_stream"] += int(
            receipt.get("resolution_events_applied_in_stream", 0)
        )
        resolution_totals["deferred_until_day_end"] += int(
            receipt.get("resolution_events_deferred_until_day_end", 0)
        )
    valid = (
        decision_rows > 0
        and test_label_rows == 0
        and non_finite == 0
        and (
            not full_run
            or (
                stream_rows == int(config["acceptance"]["full_stream_rows"])
                and decision_rows == int(decision_manifest["rows"])
            )
        )
    )
    previous_comparable = bool(
        previous_manifest
        and previous_manifest.get("config_sha256") == config_hash
        and previous_manifest.get("implementation_sha256") == implementation_hash
        and previous_manifest.get("stream_scope_digest") == stream_manifest["scope_digest"]
        and previous_manifest.get("full_run") == full_run
        and int(previous_manifest.get("days") or -1) == len(receipts)
    )
    hashes_match_previous = (
        None
        if not previous_comparable or previous_manifest is None
        else (
            previous_manifest.get("daily_receipts_sha256")
            == hashlib.sha256(
                "\n".join(
                    f"{row['date']}:{row['files']['features.npy']['sha256']}" for row in receipts
                ).encode()
            ).hexdigest()
            and previous_manifest.get("checkpoint", {}).get("sha256") == checkpoint["sha256"]
            and previous_manifest.get("normalization", {}).get("sha256") == normalization["sha256"]
        )
    )
    valid = valid and hashes_match_previous is not False
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_feature_pack_manifest",
        "generated_at": now_utc(),
        "research_id": str(config["research_id"]),
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "valid": valid,
        "full_run": full_run,
        "stream_scope_digest": stream_manifest["scope_digest"],
        "stream_rows": stream_rows,
        "decision_rows": decision_rows,
        "days": len(receipts),
        "wallets": index.wallet_count,
        "markets": len(index.market_conditions),
        "components": len(index.component_ids),
        "hard_wallet_cap": None,
        "hard_market_cap": None,
        "hard_trade_cap": None,
        "feature_width": H011_FEATURE_WIDTH,
        "feature_names": list(H011_FEATURE_NAMES),
        "feature_names_sha256": _feature_names_hash(),
        "test_labels_opened": False,
        "test_label_rows": test_label_rows,
        "non_finite_features": non_finite,
        "source_price_domain": {
            "policy": str(config["input"]["source_price_domain_policy"]),
            "anomaly_rows": source_price_anomaly_rows,
            "trades_dropped": 0,
        },
        "actor_context": {
            "complete": schedule.complete,
            "manifest_sha256": schedule.manifest_sha256,
            "coverage_end_date": schedule.coverage_end_date,
            "applied_counts": dict(actor_counts),
        },
        "resolved_wallet_performance": {
            "complete": resolution_schedule.complete,
            "manifest_sha256": resolution_schedule.manifest_sha256,
            "availability_masked": not resolution_schedule.complete,
            "counts": dict(resolution_totals),
            "qualification_blocker": (
                None
                if resolution_schedule.complete
                else "causal_resolution_update_ledger_not_attached"
            ),
        },
        "normalization": normalization,
        "checkpoint": checkpoint,
        "reproducibility": {
            "previous_comparable_manifest_found": previous_comparable,
            "output_hashes_match_previous": hashes_match_previous,
        },
        "daily_receipts_sha256": hashlib.sha256(
            "\n".join(
                f"{row['date']}:{row['files']['features.npy']['sha256']}" for row in receipts
            ).encode()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": str(config["evidence_boundary"]),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if not valid:
        raise RuntimeError(f"H011 feature pack failed acceptance: {output_dir / 'manifest.json'}")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--h009-config", type=Path, default=DEFAULT_H009_CONFIG)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--chronicle-dir", type=Path)
    value.add_argument("--actor-dir", type=Path)
    value.add_argument("--resolution-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--chunk-rows", type=int)
    value.add_argument("--day-limit", type=int)
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
    default_resolution_dir = data_dir / "derived" / "sphinx-h011-resolution-context-v1"
    resolution_dir = args.resolution_dir or (
        default_resolution_dir if default_resolution_dir.exists() else None
    )
    result = build_feature_pack(
        args.config.resolve(),
        args.h009_config.resolve(),
        data_dir,
        chronicle_dir,
        args.actor_dir.resolve() if args.actor_dir is not None else None,
        resolution_dir.resolve() if resolution_dir is not None else None,
        output_dir,
        chunk_rows=args.chunk_rows or int(config["storage"]["chunk_rows"]),
        day_limit=args.day_limit,
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "full_run": result["full_run"],
                "stream_rows": result["stream_rows"],
                "decision_rows": result["decision_rows"],
                "wallets": result["wallets"],
                "markets": result["markets"],
                "components": result["components"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
