from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sphinx_trace.h010_catalog import load_development_catalog


def _catalog(path: Path, *, split: str = "validation", terminal: str | None = "[1.0,0.0]") -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata VALUES ('test_terminal_fields_accessed', 'false');
        CREATE TABLE markets(
          condition_id TEXT, component_id TEXT, outcomes TEXT, token_ids TEXT,
          closed_at TEXT, split_id TEXT, terminal_label TEXT, replayable INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO markets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "condition",
            "component",
            '["Yes","No"]',
            '["yes-token","no-token"]',
            "2026-03-01T00:00:00Z",
            split,
            terminal,
            1,
        ),
    )
    connection.commit()
    connection.close()


def test_catalog_selection_binds_contract_and_resolution(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    _catalog(path)
    selection = load_development_catalog(path, {"condition"})
    assert selection.contracts["condition"].token_ids == ("yes-token", "no-token")
    assert selection.resolutions[0].payouts[0] == 1
    assert selection.split_counts == {"validation": 1}


def test_catalog_selection_rejects_test_or_missing_market(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    _catalog(path, split="test", terminal=None)
    with pytest.raises(RuntimeError, match="outside selected split"):
        load_development_catalog(path, {"condition"})
    with pytest.raises(RuntimeError, match="missing"):
        load_development_catalog(path, {"unknown"})
