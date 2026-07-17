from __future__ import annotations

import json
from pathlib import Path

import pytest

from sphinx_trace.h011_sources import load_ledger_scope_index


def _receipt(path: Path, scope_id: str, conditions: list[str], rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "gaps": 0,
                "scope_id": scope_id,
                "condition_ids": conditions,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_scope_index_preserves_group_to_market_mapping_and_reloads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    chronicle = tmp_path / "chronicle"
    root = data / "receipts" / "ledger"
    _receipt(root / "group-a.json", "group-a", ["0xa", "0xb"], 3)
    _receipt(root / "0xc.json", "0xc", ["0xc"], 4)
    scopes, receipt = load_ledger_scope_index(
        data,
        chronicle,
        namespace="ledger",
        expected_scopes=2,
        expected_markets=3,
        expected_rows=7,
        source_manifest_sha256="source",
        workers=2,
    )
    assert [scope.scope_id for scope in scopes] == ["0xc", "group-a"]
    assert {value for scope in scopes for value in scope.condition_ids} == {"0xa", "0xb", "0xc"}
    assert receipt["market_count"] == 3
    reloaded, second = load_ledger_scope_index(
        data,
        chronicle,
        namespace="ledger",
        expected_scopes=2,
        expected_markets=3,
        expected_rows=7,
        source_manifest_sha256="source",
        workers=2,
    )
    assert reloaded == scopes
    assert second["sha256"] == receipt["sha256"]


def test_scope_index_rejects_market_count_drift(tmp_path: Path) -> None:
    root = tmp_path / "data" / "receipts" / "ledger"
    _receipt(root / "group.json", "group", ["0xa"], 1)
    with pytest.raises(RuntimeError, match="market or row coverage"):
        load_ledger_scope_index(
            tmp_path / "data",
            tmp_path / "chronicle",
            namespace="ledger",
            expected_scopes=1,
            expected_markets=2,
            expected_rows=1,
            source_manifest_sha256="source",
        )
