from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sphinx_corpus.config import CorpusConfig
from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc

_ADDRESS = re.compile(r"^0x[a-f0-9]{40}$")
_HASH = re.compile(r"^0x[a-f0-9]{64}$")
_TRADE_ID = re.compile(r"^[a-f0-9]{64}$")
_REQUIRED = (
    "schema_version",
    "record_type",
    "source",
    "trade_id",
    "condition_id",
    "token_id",
    "wallet",
    "side",
    "size",
    "price",
    "notional_usd",
    "timestamp",
    "timestamp_unix",
    "transaction_hash",
)


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def spread_paths(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("sample file count must be positive")
    if len(paths) <= count:
        return paths
    return [paths[min(int(index * len(paths) / count), len(paths) - 1)] for index in range(count)]


def profile(config_path: Path, data_dir: Path, sample_files: int) -> dict[str, Any]:
    config = CorpusConfig.load(config_path, data_dir)
    namespace = str(config.payload["sources"]["ledger"]["primary"]["storage_namespace"])
    source_root = data_dir / "normalized" / namespace
    all_paths = sorted(source_root.rglob("*.jsonl.zst"))
    paths = spread_paths(all_paths, sample_files)
    path_digest = hashlib.sha256(
        "\n".join(path.relative_to(data_dir).as_posix() for path in paths).encode()
    ).hexdigest()

    counts: Counter[str] = Counter()
    sides: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    trade_ids: set[str] = set()
    wallets: set[str] = set()
    condition_ids: set[str] = set()
    minimum_timestamp: int | None = None
    maximum_timestamp: int | None = None
    minimum_price: Decimal | None = None
    maximum_price: Decimal | None = None
    minimum_notional: Decimal | None = None
    below_filter_minimum_notional: Decimal | None = None
    below_filter_by_side: Counter[str] = Counter()
    below_filter_buckets: Counter[str] = Counter()

    for path in paths:
        previous_timestamp: int | None = None
        for row in iter_jsonl_zst(path):
            counts["rows"] += 1
            for key in _REQUIRED:
                value = row.get(key)
                if value is None or value == "":
                    missing[key] += 1

            trade_id = str(row.get("trade_id") or "")
            if not _TRADE_ID.fullmatch(trade_id):
                counts["invalid_trade_id"] += 1
            if trade_id in trade_ids:
                counts["duplicate_trade_id"] += 1
            trade_ids.add(trade_id)

            condition_id = str(row.get("condition_id") or "")
            if not _HASH.fullmatch(condition_id):
                counts["invalid_condition_id"] += 1
            condition_ids.add(condition_id)

            wallet = str(row.get("wallet") or "")
            if not _ADDRESS.fullmatch(wallet):
                counts["invalid_wallet"] += 1
            wallets.add(wallet)

            transaction_hash = str(row.get("transaction_hash") or "")
            if not _HASH.fullmatch(transaction_hash):
                counts["invalid_transaction_hash"] += 1

            token_id = str(row.get("token_id") or "")
            if not token_id.isdigit():
                counts["invalid_token_id"] += 1

            side = str(row.get("side") or "")
            sides[side] += 1
            if side not in {"BUY", "SELL"}:
                counts["invalid_side"] += 1

            if row.get("schema_version") != "1.0.0":
                counts["invalid_schema_version"] += 1
            if row.get("record_type") != "public_trade":
                counts["invalid_record_type"] += 1
            if row.get("source") != "polymarket_data_api_trades":
                counts["invalid_source"] += 1

            price = _decimal(row.get("price"))
            size = _decimal(row.get("size"))
            notional = _decimal(row.get("notional_usd"))
            if price is None or not Decimal(0) < price < Decimal(1):
                counts["invalid_price"] += 1
            elif minimum_price is None or price < minimum_price:
                minimum_price = price
            if price is not None and (maximum_price is None or price > maximum_price):
                maximum_price = price
            if size is None or size <= 0:
                counts["invalid_size"] += 1
            if notional is None or notional <= Decimal(0):
                counts["invalid_notional"] += 1
            elif notional < Decimal(25):
                counts["notional_below_filter"] += 1
                below_filter_by_side[side] += 1
                below_filter_minimum_notional = (
                    notional
                    if below_filter_minimum_notional is None
                    else min(below_filter_minimum_notional, notional)
                )
                if notional < Decimal(1):
                    below_filter_buckets["lt_1"] += 1
                elif notional < Decimal(5):
                    below_filter_buckets["1_to_lt_5"] += 1
                elif notional < Decimal(10):
                    below_filter_buckets["5_to_lt_10"] += 1
                elif notional < Decimal(20):
                    below_filter_buckets["10_to_lt_20"] += 1
                else:
                    below_filter_buckets["20_to_lt_25"] += 1
                if size is not None and size >= Decimal(25):
                    counts["below_filter_with_size_gte_25"] += 1
                if (
                    price is not None
                    and size is not None
                    and size * (Decimal(1) - price) >= Decimal(25)
                ):
                    counts["below_filter_with_complement_notional_gte_25"] += 1
                if (
                    price is not None
                    and size is not None
                    and max(size * price, size * (Decimal(1) - price)) >= Decimal(25)
                ):
                    counts["below_filter_with_either_notional_gte_25"] += 1
            if notional is not None and (minimum_notional is None or notional < minimum_notional):
                minimum_notional = notional
            if (
                price is not None
                and size is not None
                and notional is not None
                and size * price != notional
            ):
                counts["inconsistent_notional"] += 1

            try:
                timestamp = int(str(row.get("timestamp_unix")))
            except (TypeError, ValueError):
                counts["invalid_timestamp"] += 1
                continue
            if (
                not int(config.window.start.timestamp())
                <= timestamp
                < int(config.window.end.timestamp())
            ):
                counts["timestamp_outside_window"] += 1
            expected_timestamp = (
                datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
            )
            if row.get("timestamp") != expected_timestamp:
                counts["timestamp_text_mismatch"] += 1
            if previous_timestamp is not None and timestamp < previous_timestamp:
                counts["file_timestamp_order_violation"] += 1
            previous_timestamp = timestamp
            minimum_timestamp = (
                timestamp if minimum_timestamp is None else min(minimum_timestamp, timestamp)
            )
            maximum_timestamp = (
                timestamp if maximum_timestamp is None else max(maximum_timestamp, timestamp)
            )
            counts["outcome_null"] += int(row.get("outcome") is None)
            counts["outcome_index_null"] += int(row.get("outcome_index") is None)

    critical_keys = (
        "invalid_trade_id",
        "duplicate_trade_id",
        "invalid_condition_id",
        "invalid_wallet",
        "invalid_transaction_hash",
        "invalid_token_id",
        "invalid_side",
        "invalid_schema_version",
        "invalid_record_type",
        "invalid_source",
        "invalid_price",
        "invalid_size",
        "invalid_notional",
        "inconsistent_notional",
        "invalid_timestamp",
        "timestamp_outside_window",
        "timestamp_text_mismatch",
        "file_timestamp_order_violation",
    )
    rows = counts["rows"]
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": now_utc(),
        "namespace": namespace,
        "valid": rows > 0 and not missing and all(counts[key] == 0 for key in critical_keys),
        "sample": {
            "files_total": len(all_paths),
            "files_sampled": len(paths),
            "file_sample_rate": len(paths) / max(len(all_paths), 1),
            "paths_sha256": path_digest,
            "rows": rows,
            "condition_ids": len(condition_ids),
            "wallets": len(wallets),
            "trade_ids": len(trade_ids),
        },
        "missing_required": dict(missing),
        "violations": {key: counts[key] for key in critical_keys},
        "filter_diagnostics": {
            "notional_below_25": counts["notional_below_filter"],
            "notional_below_25_rate": counts["notional_below_filter"] / max(rows, 1),
            "minimum_notional_below_25": (
                str(below_filter_minimum_notional)
                if below_filter_minimum_notional is not None
                else None
            ),
            "by_side": dict(below_filter_by_side),
            "buckets": dict(below_filter_buckets),
            "with_size_gte_25": counts["below_filter_with_size_gte_25"],
            "with_complement_notional_gte_25": counts[
                "below_filter_with_complement_notional_gte_25"
            ],
            "with_either_notional_gte_25": counts["below_filter_with_either_notional_gte_25"],
        },
        "optional_nulls": {
            "outcome": counts["outcome_null"],
            "outcome_index": counts["outcome_index_null"],
        },
        "sides": dict(sides),
        "minimum_price": str(minimum_price) if minimum_price is not None else None,
        "maximum_price": str(maximum_price) if maximum_price is not None else None,
        "minimum_notional_usd": (str(minimum_notional) if minimum_notional is not None else None),
        "minimum_timestamp_unix": minimum_timestamp,
        "maximum_timestamp_unix": maximum_timestamp,
    }
    output_path = data_dir / "receipts" / f"{namespace}-profile.json"
    atomic_json(output_path, result)
    if not result["valid"]:
        raise RuntimeError(f"Fast Ledger profile failed; see {output_path}")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument(
        "--config",
        type=Path,
        default=Path("configs/corpus/sphinx_corpus_s0_fast_v1.json"),
    )
    root.add_argument("--data-dir", type=Path, required=True)
    root.add_argument("--sample-files", type=int, default=2048)
    return root


def main() -> None:
    args = parser().parse_args()
    result = profile(args.config, args.data_dir.resolve(), args.sample_files)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
