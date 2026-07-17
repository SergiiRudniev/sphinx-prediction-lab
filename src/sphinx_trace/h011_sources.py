"""Exact H009 Ledger scope-to-condition index used by H011 builders."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, now_utc, sha256_file, write_jsonl_zst


@dataclass(frozen=True, slots=True)
class LedgerScope:
    scope_id: str
    condition_ids: tuple[str, ...]
    rows: int


def _load_object(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_scope_receipt(path: Path) -> LedgerScope:
    receipt = _load_object(path)
    if receipt.get("complete") is not True or int(receipt.get("gaps") or 0) != 0:
        raise RuntimeError(f"Incomplete Ledger scope receipt: {path}")
    scope_id = str(receipt.get("scope_id") or path.stem)
    raw_conditions = receipt.get("condition_ids")
    if isinstance(raw_conditions, list):
        condition_ids = tuple(str(value).lower() for value in raw_conditions)
    elif scope_id.startswith("0x"):
        condition_ids = (scope_id.lower(),)
    else:
        raise RuntimeError(f"Ledger scope receipt has no condition index: {path}")
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise RuntimeError(f"Ledger scope receipt has empty or duplicate conditions: {path}")
    return LedgerScope(scope_id=scope_id, condition_ids=condition_ids, rows=int(receipt["rows"]))


def load_ledger_scope_index(
    data_dir: Path,
    chronicle_dir: Path,
    *,
    namespace: str,
    expected_scopes: int,
    expected_markets: int,
    expected_rows: int,
    source_manifest_sha256: str,
    workers: int = 16,
) -> tuple[list[LedgerScope], dict[str, Any]]:
    """Build once from authoritative receipts and subsequently verify the cached index."""

    if workers <= 0:
        raise ValueError("Ledger scope-index workers must be positive")
    path = chronicle_dir / "index" / "ledger-scopes.jsonl.zst"
    receipt_path = chronicle_dir / "receipts" / "ledger-scopes.json"
    expected = {
        "namespace": namespace,
        "source_manifest_sha256": source_manifest_sha256,
        "scope_count": expected_scopes,
        "market_count": expected_markets,
        "source_rows": expected_rows,
    }
    if path.exists() and receipt_path.exists():
        cached_receipt = _load_object(receipt_path)
        if any(cached_receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Cached Ledger scope index belongs to another source contract")
        if sha256_file(path) != cached_receipt.get("sha256"):
            raise RuntimeError("Cached Ledger scope index hash changed")
        scopes = [
            LedgerScope(
                scope_id=str(row["scope_id"]),
                condition_ids=tuple(str(value) for value in row["condition_ids"]),
                rows=int(row["rows"]),
            )
            for row in iter_jsonl_zst(path)
        ]
        return scopes, cached_receipt

    receipt_root = data_dir / "receipts" / namespace
    receipt_paths = sorted(receipt_root.glob("*.json"))
    if len(receipt_paths) != expected_scopes:
        raise RuntimeError(
            f"Ledger scope receipt count changed: {len(receipt_paths)} != {expected_scopes}"
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scopes = list(executor.map(_read_scope_receipt, receipt_paths))
    scopes.sort(key=lambda scope: scope.scope_id)
    conditions = {condition for scope in scopes for condition in scope.condition_ids}
    source_rows = sum(scope.rows for scope in scopes)
    if len(conditions) != expected_markets or source_rows != expected_rows:
        raise RuntimeError(
            "Ledger scope index changed market or row coverage: "
            f"markets={len(conditions)} rows={source_rows}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, size = write_jsonl_zst(
        path,
        (
            {
                "scope_id": scope.scope_id,
                "condition_ids": list(scope.condition_ids),
                "rows": scope.rows,
            }
            for scope in scopes
        ),
    )
    if rows != expected_scopes:
        raise RuntimeError("Ledger scope index lost scope rows")
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "h011_ledger_scope_index",
        "generated_at": now_utc(),
        **expected,
        "path": path.relative_to(chronicle_dir).as_posix(),
        "bytes": size,
        "sha256": sha256_file(path),
        "hard_market_cap": None,
    }
    atomic_json(receipt_path, receipt)
    return scopes, receipt
