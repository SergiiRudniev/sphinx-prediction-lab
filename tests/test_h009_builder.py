from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.build_h009_chronicle import (
    build_catalog,
    build_decisions,
    build_episodes,
    build_receipt,
    build_runs,
    build_stream,
)
from scripts.validate_h009_chronicle import validate

from sphinx_corpus.io import atomic_json, iter_jsonl_zst, write_jsonl_zst
from sphinx_trace.config import load_json

ROOT = Path(__file__).resolve().parents[1]


def market(
    condition: str,
    market_id: str,
    event_ids: list[str],
    closed_at: str,
    *,
    neg_risk: bool = False,
    payout: tuple[str, str] = ("1", "0"),
) -> dict[str, Any]:
    return {
        "condition_id": condition,
        "market_id": market_id,
        "event_ids": event_ids,
        "closed_at": closed_at,
        "resolution_status": "resolved",
        "neg_risk": neg_risk,
        "observed_at": "2026-07-16T00:00:00Z",
        "source_payload": {
            "question": f"Market {market_id}?",
            "description": "Rules",
            "outcomes": '["Yes","No"]',
            "outcomePrices": json.dumps(payout),
            "clobTokenIds": json.dumps([f"{market_id}0", f"{market_id}1"]),
        },
    }


def trade(index: int, condition: str, timestamp: int, wallet: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "public_trade",
        "trade_id": f"{index:064x}",
        "condition_id": condition,
        "token_id": "1",
        "wallet": wallet,
        "side": "BUY",
        "size": "1",
        "price": "0.5",
        "notional_usd": "0.5",
        "timestamp": "2026-01-01T00:00:00Z",
        "timestamp_unix": timestamp,
        "transaction_hash": f"0x{index:064x}",
        "outcome": "Yes",
        "outcome_index": 0,
    }


def write_scope(
    data_dir: Path,
    namespace: str,
    scope_id: str,
    rows: list[dict[str, Any]],
) -> None:
    scope = data_dir / "normalized" / namespace / f"scope={scope_id}"
    write_jsonl_zst(scope / "start=1-end=9.jsonl.zst", rows)
    atomic_json(
        data_dir / "receipts" / namespace / f"{scope_id}.json",
        {
            "complete": True,
            "gaps": 0,
            "condition_ids": sorted({str(row["condition_id"]) for row in rows}),
            "rows": len(rows),
        },
    )


def fixture_config(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "chronicle"
    atlas_rows = [
        market("0x01", "1", ["event-a", "event-b"], "2026-01-03T00:00:00Z"),
        market(
            "0x02",
            "2",
            ["event-b"],
            "2026-01-04T00:00:00Z",
            neg_risk=True,
            payout=("0", "1"),
        ),
        market("", "invalid", ["event-a"], "2026-01-05T00:00:00Z"),
        market("0x03", "3", ["event-test"], "2026-06-03T00:00:00Z"),
    ]
    atlas_path = data_dir / "normalized" / "atlas" / "markets.jsonl.zst"
    write_jsonl_zst(atlas_path, atlas_rows)
    namespace = "ledger-test"
    write_scope(
        data_dir,
        namespace,
        "scope-a",
        [
            trade(1, "0x01", 100, "0xwallet-a"),
            trade(3, "0x02", 300, "0xwallet-b"),
        ],
    )
    write_scope(
        data_dir,
        namespace,
        "scope-b",
        [
            trade(2, "0x01", 200, "0xwallet-c"),
            trade(4, "0x03", 400, "0xwallet-d"),
        ],
    )
    config = deepcopy(load_json(ROOT / "configs" / "corpus" / "sphinx_chronicle_h009_v1.json"))
    config["sources"]["atlas"]["markets"]["path"] = "normalized/atlas/markets.jsonl.zst"
    config["sources"]["atlas"]["markets"]["rows"] = 4
    config["sources"]["ledger"]["namespace"] = namespace
    config["sources"]["ledger"]["rows"] = 4
    config["sources"]["ledger"]["scope_groups"] = 2
    config["sources"]["ledger"]["files"] = 2
    config["acceptance"]["ledger_rows_preserved_exactly"] = 4
    config["storage"]["run_scope_group_size"] = 1
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    return config, config_path, data_dir, output_dir


def test_h009_builder_preserves_all_rows_test_labels_and_resume(tmp_path: Path) -> None:
    config, config_path, data_dir, output_dir = fixture_config(tmp_path)
    catalog = build_catalog(config, config_path, data_dir, output_dir)
    episodes = build_episodes(config, config_path, output_dir)
    runs = build_runs(
        config,
        config_path,
        data_dir,
        output_dir,
        scope_limit=None,
        workers=2,
    )
    stream = build_stream(config, config_path, output_dir)
    decisions = build_decisions(config, config_path, output_dir)
    receipt = build_receipt(config, config_path, output_dir)
    validation = validate(
        config_path,
        output_dir,
        require_full=True,
        require_polygon=False,
        deep_hashes=True,
    )

    assert catalog["test_terminal_fields_accessed"] is False
    assert catalog["counts"]["multi_market_components"] == 1
    assert catalog["counts"]["neg_risk_components"] == 1
    assert episodes["rows"] == 2
    assert runs["rows"] == 4
    assert stream["rows"] == 4
    assert decisions["stream_rows"] == 4
    assert receipt["structural_valid"] is True
    assert receipt["fully_qualified"] is False
    assert receipt["qualification_blockers"] == ["polygon_graph_backfill"]
    assert validation["valid"] is True

    episode_rows = list(iter_jsonl_zst(output_dir / "episodes.jsonl.zst"))
    test_episode = next(row for row in episode_rows if row["split"] == "test")
    assert all(market_row["terminal_label"] is None for market_row in test_episode["markets"])
    invalid_market = next(
        market_row
        for row in episode_rows
        for market_row in row["markets"]
        if market_row["market_id"] == "invalid"
    )
    assert invalid_market["replayable"] is False
    assert invalid_market["terminal_label"] is None

    first_decision_hash = decisions["shards"][0]["sha256"]
    resumed = build_decisions(config, config_path, output_dir)
    assert resumed["stream_rows"] == 4
    assert resumed["shards"][0]["sha256"] == first_decision_hash
