from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from sphinx_corpus.atlas import AtlasBackfill, market_intersects
from sphinx_corpus.config import CorpusConfig, ExchangeContract, Window
from sphinx_corpus.io import (
    build_manifest,
    count_jsonl_zst,
    iter_jsonl_zst,
    write_json_zst,
    write_jsonl_zst,
)
from sphinx_corpus.ledger import decode_order_fill, event_topic
from sphinx_corpus.trade_api import RequestWindowTimeout, TradeAPIBackfill

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig.load(
        ROOT / "configs" / "corpus" / "sphinx_corpus_v1.json",
        tmp_path,
    )


def _fast_config(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig.load(
        ROOT / "configs" / "corpus" / "sphinx_corpus_s0_fast_v1.json",
        tmp_path,
    )


def _topic_address(address: str) -> str:
    return "0x" + ("0" * 24) + address.removeprefix("0x").lower()


def _data(*values: int | str) -> str:
    words = []
    for value in values:
        if isinstance(value, int):
            words.append(f"{value:064x}")
        else:
            words.append(value.removeprefix("0x").rjust(64, "0"))
    return "0x" + "".join(words)


def _log(data: str) -> dict[str, Any]:
    return {
        "address": "0xexchange",
        "blockNumber": "0x64",
        "blockHash": "0xblock",
        "transactionHash": "0xtx",
        "transactionIndex": "0x2",
        "logIndex": "0x3",
        "removed": False,
        "topics": [
            "0xtopic",
            "0x" + ("ab" * 32),
            _topic_address("0x1111111111111111111111111111111111111111"),
            _topic_address("0x2222222222222222222222222222222222222222"),
        ],
        "data": data,
    }


def _contract(protocol: str) -> ExchangeContract:
    return ExchangeContract(
        id=f"{protocol}-standard",
        protocol=protocol,
        market_type="standard",
        address="0xExchange",
        active=Window(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2027, 1, 1, tzinfo=UTC),
        ),
        event_signature="unused",
    )


def test_decodes_v1_buy_order_fill() -> None:
    record = decode_order_fill(
        _log(_data(0, 99, 40, 100, 1)),
        _contract("clob-v1"),
        chain_id=137,
        block_timestamp=1_700_000_000,
    )

    assert record["side"] == "BUY"
    assert record["token_id"] == "99"
    assert record["price"] == "0.4"
    assert record["collateral_amount_raw"] == "40"
    assert record["maker"] == "0x1111111111111111111111111111111111111111"


def test_decodes_v2_sell_order_fill() -> None:
    record = decode_order_fill(
        _log(_data(1, 77, 100, 60, 2, "0x1234", "0x5678")),
        _contract("clob-v2"),
        chain_id=137,
        block_timestamp=1_700_000_000,
    )

    assert record["side"] == "SELL"
    assert record["token_id"] == "77"
    assert record["price"] == "0.6"
    assert record["builder"].endswith("1234")
    assert record["metadata"].endswith("5678")


def test_event_topics_differ_across_protocol_versions() -> None:
    v1 = event_topic("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
    v2 = event_topic(
        "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)"
    )
    assert len(v1) == 66
    assert len(v2) == 66
    assert v1 != v2


def test_atlas_normalization_deduplicates_and_filters_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    in_window = {
        "id": "market-1",
        "conditionId": "condition-1",
        "createdAt": "2025-08-01T00:00:00Z",
        "endDate": "2025-09-01T00:00:00Z",
        "events": [{"id": "event-1", "title": "Event"}],
        "clobTokenIds": json.dumps(["yes", "no"]),
        "outcomes": json.dumps(["Yes", "No"]),
    }
    outside = {
        "id": "market-old",
        "conditionId": "condition-old",
        "createdAt": "2024-01-01T00:00:00Z",
        "closedTime": "2024-02-01T00:00:00Z",
    }
    page = {
        "schema_version": "1.0.0",
        "observed_at": "2026-07-16T00:00:00Z",
        "response": {"markets": [in_window, outside]},
    }
    write_json_zst(
        tmp_path / "raw" / "atlas" / "closed=true-windowed" / "page-000000.json.zst",
        page,
    )

    with AtlasBackfill(config) as collector:
        summary = collector.normalize()

    assert summary == {"markets": 1, "events": 1, "tokens": 2}
    tokens = list(iter_jsonl_zst(tmp_path / "normalized" / "atlas" / "tokens.jsonl.zst"))
    assert {row["token_id"] for row in tokens} == {"yes", "no"}
    assert market_intersects(in_window, config) is True
    assert market_intersects(outside, config) is False


class FakeTradeBackfill(TradeAPIBackfill):
    def __init__(self, config: CorpusConfig) -> None:
        super().__init__(config)
        self.page_size = 2
        self.maximum_offset = 2

    def _page(
        self,
        scope_id: str,
        condition_ids: tuple[str, ...],
        start: int,
        end: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        condition_id = condition_ids[0]
        if end - start > 2:
            return [
                {
                    "conditionId": condition_id,
                    "timestamp": start,
                    "proxyWallet": "0xwallet",
                    "asset": "token",
                    "side": "BUY",
                    "size": 1,
                    "price": 0.5,
                    "transactionHash": f"0x{offset}-{index}",
                }
                for index in range(self.page_size)
            ]
        return [
            {
                "conditionId": condition_id,
                "timestamp": start,
                "proxyWallet": "0xwallet",
                "asset": "token",
                "side": "BUY",
                "size": 1,
                "price": 0.5,
                "transactionHash": f"0x{start}",
            }
        ]


class ParallelTradeBackfill(TradeAPIBackfill):
    def __init__(self, config: CorpusConfig) -> None:
        super().__init__(config, workers=4, requests_per_second=1000)
        self.target_group_volume = Decimal(1)
        self.barrier = threading.Barrier(4)

    def _markets(
        self,
        explicit: set[str] | None,
        max_markets: int | None,
    ) -> list[dict[str, Any]]:
        del explicit, max_markets
        return [
            {
                "condition_id": "0x" + (f"{index:02x}" * 32),
                "_volume": Decimal(2),
            }
            for index in range(1, 5)
        ]

    def _collect_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        self.barrier.wait(timeout=5)
        return {"status": "completed", "rows": len(group), "gaps": 0}


class TimeoutTradeBackfill(FakeTradeBackfill):
    def _page(
        self,
        scope_id: str,
        condition_ids: tuple[str, ...],
        start: int,
        end: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        if end - start > 2:
            raise RequestWindowTimeout("simulated timeout")
        condition_id = condition_ids[0]
        return [
            {
                "conditionId": condition_id,
                "timestamp": start,
                "proxyWallet": "0xwallet",
                "asset": "token",
                "side": "BUY",
                "size": 1,
                "price": 0.5,
                "transactionHash": f"0x{start}",
            }
        ]


def test_trade_api_splits_saturated_time_windows(tmp_path: Path) -> None:
    condition_id = "0x" + ("ab" * 32)
    with FakeTradeBackfill(_config(tmp_path)) as collector:
        result = collector._window(condition_id, (condition_id,), 100, 107)

    assert result["complete"] is True
    assert result["gaps"] == 0
    assert result["rows"] == 4
    files = list((tmp_path / "normalized" / "ledger-api").rglob("*.jsonl.zst"))
    assert len(files) == 4


def test_trade_api_splits_timed_out_windows(tmp_path: Path) -> None:
    condition_id = "0x" + ("bc" * 32)
    with TimeoutTradeBackfill(_config(tmp_path)) as collector:
        result = collector._window(condition_id, (condition_id,), 100, 107)

    assert result["complete"] is True
    assert result["gaps"] == 0
    assert result["rows"] == 4
    receipt_path = (
        tmp_path
        / "receipts"
        / "ledger-api-windows"
        / f"scope={condition_id}"
        / "start=100-end=107.json"
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["reason"] == "request_timeout"


def test_trade_api_maps_repeated_http_408_to_window_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sphinx_corpus.trade_api.time.sleep", lambda _: None)
    transport = httpx.MockTransport(lambda _: httpx.Response(408))
    with TradeAPIBackfill(_config(tmp_path)) as collector:
        client = httpx.Client(transport=transport)
        collector._thread_clients.client = client
        collector._clients.append(client)

        with pytest.raises(RequestWindowTimeout):
            collector._fetch({"market": "condition"})


def test_trade_api_preserves_indistinguishable_repeated_rows(tmp_path: Path) -> None:
    condition_id = "0x" + ("cd" * 32)
    trade = {
        "conditionId": condition_id,
        "timestamp": 101,
        "proxyWallet": "0xWallet",
        "asset": "token",
        "side": "SELL",
        "size": 2,
        "price": 0.25,
        "transactionHash": "0xtx",
    }
    with TradeAPIBackfill(_config(tmp_path)) as collector:
        rows = collector._normalize((condition_id,), 100, 102, [trade, trade.copy()])

    assert len(rows) == 2
    assert rows[0]["trade_id"] != rows[1]["trade_id"]
    assert rows[0]["notional_usd"] == "0.50"


def test_fast_trade_api_profile_freezes_filter_and_rate_limit(tmp_path: Path) -> None:
    condition_id = "0x" + ("ef" * 32)
    with TradeAPIBackfill(_fast_config(tmp_path)) as collector:
        params = collector._request_params((condition_id,), 100, 200, 0)

        assert collector.storage_namespace == "ledger-api-cash-min-25"
        assert collector.workers == 24
        assert collector.request_gate.rate == 18
        assert params["filterType"] == "CASH"
        assert params["filterAmount"] == 25


def test_trade_api_keeps_only_collection_fields_in_memory(tmp_path: Path) -> None:
    condition_id = "0x" + ("12" * 32)
    config = _fast_config(tmp_path)
    write_jsonl_zst(
        tmp_path / "normalized" / "atlas" / "markets.jsonl.zst",
        [
            {
                "condition_id": condition_id,
                "created_at": "2025-08-01T00:00:00Z",
                "start_at": "2025-08-02T00:00:00Z",
                "end_at": "2025-08-03T00:00:00Z",
                "closed_at": "2025-08-04T00:00:00Z",
                "source_payload": {
                    "volumeNum": 100,
                    "large_unused_field": "unused" * 1000,
                },
            }
        ],
    )

    with TradeAPIBackfill(config) as collector:
        markets = collector._markets(None, None)

    assert markets == [
        {
            "condition_id": condition_id,
            "created_at": "2025-08-01T00:00:00Z",
            "start_at": "2025-08-02T00:00:00Z",
            "end_at": "2025-08-03T00:00:00Z",
            "closed_at": "2025-08-04T00:00:00Z",
            "_volume": Decimal(100),
        }
    ]


def test_trade_api_groups_large_markets_separately(tmp_path: Path) -> None:
    markets = [
        {"condition_id": "large", "_volume": "1500000"},
        {"condition_id": "medium-a", "_volume": "600000"},
        {"condition_id": "medium-b", "_volume": "300000"},
        {"condition_id": "medium-c", "_volume": "200000"},
    ]
    with TradeAPIBackfill(_fast_config(tmp_path)) as collector:
        groups = collector._groups(markets)

    assert [[row["condition_id"] for row in group] for group in groups] == [
        ["large"],
        ["medium-a", "medium-b"],
        ["medium-c"],
    ]
    assert collector._initial_windows(groups[0], 100, 199) == [
        (100, 149),
        (150, 199),
    ]
    assert collector._initial_windows(groups[1], 100, 199) == [(100, 199)]


def test_trade_api_collects_independent_groups_concurrently(tmp_path: Path) -> None:
    with ParallelTradeBackfill(_fast_config(tmp_path)) as collector:
        receipt = collector.collect()

    assert receipt["complete"] is True
    assert receipt["groups_completed_this_run"] == 4
    assert receipt["rows"] == 4


def test_manifest_excludes_registered_ignored_prefixes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_json_zst(tmp_path / "raw" / "atlas" / "closed=true" / "old.json.zst", {})
    write_json_zst(
        tmp_path / "raw" / "atlas" / "closed=true-windowed" / "kept.json.zst",
        {},
    )

    manifest = build_manifest(
        tmp_path,
        corpus_id=config.id,
        version=config.version,
        research_id=config.research_id,
        source_config=config.payload,
    )

    paths = {str(row["path"]) for row in manifest["files"]}
    assert "raw/atlas/closed=true/old.json.zst" not in paths
    assert "raw/atlas/closed=true-windowed/kept.json.zst" in paths


def test_manifest_parallel_workers_preserve_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_jsonl_zst(
        tmp_path / "normalized" / "one.jsonl.zst",
        [{"row": 1}, {"row": 2}],
    )

    manifest = build_manifest(
        tmp_path,
        corpus_id=config.id,
        version=config.version,
        research_id=config.research_id,
        source_config=config.payload,
        workers=2,
    )

    assert manifest["row_count"] == 2
    assert manifest["file_count"] == 1
    assert count_jsonl_zst(tmp_path / "normalized" / "one.jsonl.zst") == 2
