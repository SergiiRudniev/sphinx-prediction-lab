from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from sphinx_corpus.io import iter_jsonl_zst, sha256_file, write_jsonl_zst
from sphinx_trace.development_tape import build_development_tape, load_tape_conditions


def _market(
    connection: sqlite3.Connection,
    condition_id: str,
    split: str,
    closed_at: str,
    terminal: str | None,
) -> None:
    connection.execute(
        "INSERT INTO markets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            condition_id,
            f"component-{condition_id}",
            '["Yes","No"]',
            f'["yes-{condition_id}","no-{condition_id}"]',
            closed_at,
            split,
            terminal,
            1,
        ),
    )


def _fixture(root: Path) -> tuple[Path, Path]:
    pack = root / "pack"
    shard = pack / "shards" / "date=2026-01-01"
    receipts = pack / "receipts"
    shard.mkdir(parents=True)
    receipts.mkdir()
    (pack / "manifest.json").write_text(
        json.dumps({"valid": True, "test_labels_opened": False}), encoding="utf-8"
    )
    (receipts / "date=2026-01-01.json").write_text("{}", encoding="utf-8")
    np.save(shard / "split_codes.npy", np.array([2, 3, 4], dtype=np.uint8))
    np.save(shard / "label_mask.npy", np.array([True, True, False], dtype=np.bool_))
    np.save(shard / "timestamps.npy", np.array([1001, 1002, 1003], dtype=np.int64))
    write_jsonl_zst(
        shard / "examples.jsonl.zst",
        (
            {
                "condition_id": "condition-a",
                "decision_time_unix": 1001,
            },
            {
                "condition_id": "condition-b",
                "decision_time_unix": 1002,
            },
            {
                "condition_id": "test-condition",
                "decision_time_unix": 1003,
            },
        ),
    )

    chronicle = root / "chronicle"
    stream = chronicle / "stream"
    stream.mkdir(parents=True)
    catalog = sqlite3.connect(chronicle / "catalog.sqlite")
    catalog.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata VALUES ('test_terminal_fields_accessed', 'false');
        CREATE TABLE markets(
          condition_id TEXT, component_id TEXT, outcomes TEXT, token_ids TEXT,
          closed_at TEXT, split_id TEXT, terminal_label TEXT, replayable INTEGER
        );
        """
    )
    _market(catalog, "condition-a", "validation", "1970-01-01T00:16:45Z", "[1.0,0.0]")
    _market(catalog, "condition-b", "calibration", "1970-01-01T00:16:46Z", "[0.0,1.0]")
    _market(catalog, "test-condition", "test", "1970-01-01T00:16:47Z", None)
    catalog.commit()
    catalog.close()
    source_path = stream / "date=1970-01-01.jsonl.zst"
    source_rows = (
        {
            "timestamp_unix": 1000,
            "trade_id": "a0",
            "condition_id": "condition-a",
        },
        {
            "timestamp_unix": 1001,
            "trade_id": "a1",
            "condition_id": "condition-a",
        },
        {
            "timestamp_unix": 1002,
            "trade_id": "b1",
            "condition_id": "condition-b",
        },
        {
            "timestamp_unix": 1004,
            "trade_id": "unknown",
            "condition_id": "unknown",
        },
        {
            "timestamp_unix": 1005,
            "trade_id": "a2",
            "condition_id": "condition-a",
        },
        {
            "timestamp_unix": 1006,
            "trade_id": "a3",
            "condition_id": "condition-a",
        },
    )
    write_jsonl_zst(source_path, source_rows)
    stream_manifest = {
        "globally_ordered": True,
        "rows": 6,
        "shards": [
            {
                "date": "1970-01-01",
                "path": "stream/date=1970-01-01.jsonl.zst",
                "rows": 6,
                "sha256": sha256_file(source_path),
            }
        ],
    }
    (chronicle / "stream-manifest.json").write_text(json.dumps(stream_manifest), encoding="utf-8")
    (chronicle / "receipt.json").write_text("{}", encoding="utf-8")
    (chronicle / "validation-receipt.json").write_text(
        json.dumps(
            {
                "valid": True,
                "deep_hashes": True,
                "hash_checks": {"stream": True},
                "catalog": {"test_terminal_labels": 0},
            }
        ),
        encoding="utf-8",
    )
    return pack, chronicle


def test_build_development_tape_filters_exact_causal_windows_and_resumes(
    tmp_path: Path,
) -> None:
    pack, chronicle = _fixture(tmp_path)
    output = tmp_path / "output"

    first = build_development_tape(pack, chronicle, output)
    second = build_development_tape(pack, chronicle, output)

    assert first["valid"] is True
    assert first["conditions"] == 2
    assert first["decision_split_counts"] == {"validation": 1, "calibration": 1}
    assert first["retained_rows"] == 3
    assert second["shard_digest"] == first["shard_digest"]
    rows = list(iter_jsonl_zst(output / "stream" / "date=1970-01-01.jsonl.zst"))
    assert [row["trade_id"] for row in rows] == ["a1", "b1", "a2"]
    assert {row["development_split"] for row in rows} == {"validation", "calibration"}
    conditions = list(iter_jsonl_zst(output / "conditions.jsonl.zst"))
    assert {row["condition_id"] for row in conditions} == {"condition-a", "condition-b"}
    validation = load_tape_conditions(output, "validation")
    assert set(validation.contracts) == {"condition-a"}
    assert validation.resolutions[0].payouts == (1, 0)
