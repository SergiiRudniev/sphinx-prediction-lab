from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from scripts.audit_h011_decision_horizon import audit

from sphinx_corpus.io import write_jsonl_zst


def test_horizon_audit_detects_post_close_decisions(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    shard = pack / "shards" / "date=2026-03-01"
    index = pack / "index"
    shard.mkdir(parents=True)
    index.mkdir()
    (pack / "manifest.json").write_text(
        json.dumps({"valid": True, "test_labels_opened": False}),
        encoding="utf-8",
    )
    write_jsonl_zst(
        index / "markets.jsonl.zst",
        [
            {
                "market_id": 0,
                "condition_id": "condition",
                "end_at_unix": 90,
            }
        ],
    )
    np.save(shard / "timestamps.npy", np.array([99, 100], dtype=np.int64))
    np.save(shard / "market_ids.npy", np.array([0, 0], dtype=np.int32))
    np.save(shard / "split_codes.npy", np.array([2, 2], dtype=np.uint8))
    np.save(shard / "label_mask.npy", np.array([1, 1], dtype=np.uint8))
    catalog = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(catalog)
    connection.execute("CREATE TABLE markets(condition_id TEXT, closed_at TEXT)")
    connection.execute(
        "INSERT INTO markets VALUES (?, ?)",
        ("condition", "1970-01-01T00:01:40Z"),
    )
    connection.commit()
    connection.close()

    result = audit(pack, catalog)
    assert result["valid"] is False
    validation = result["splits"]["validation"]
    assert validation["at_or_after_end"] == 2
    assert validation["at_or_after_close"] == 1
    assert validation["markets_at_or_after_close"] == 1
