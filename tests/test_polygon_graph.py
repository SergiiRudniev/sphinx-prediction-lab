from __future__ import annotations

from pathlib import Path

from scripts.build_h009_polygon import consolidate_polygon_tasks

from sphinx_corpus.io import sha256_file, write_jsonl_zst
from sphinx_corpus.polygon_graph import (
    ERC20_TRANSFER_TOPIC,
    address_topic,
    normalize_transfer_log,
    topic_address,
    transfer_log_endpoints,
    transfer_queries,
)


def test_wallet_topic_round_trip_and_query_directions() -> None:
    wallet = "0x" + ("12" * 20)
    assert topic_address(address_topic(wallet)) == wallet
    queries = transfer_queries("erc20", [wallet])
    assert len(queries) == 2
    assert queries[0].topics == [ERC20_TRANSFER_TOPIC, [address_topic(wallet)], None]
    assert queries[1].topics == [ERC20_TRANSFER_TOPIC, None, [address_topic(wallet)]]


def test_normalize_erc20_transfer_log() -> None:
    source = "0x" + ("12" * 20)
    target = "0x" + ("34" * 20)
    log = {
        "address": "0x" + ("56" * 20),
        "topics": [ERC20_TRANSFER_TOPIC, address_topic(source), address_topic(target)],
        "data": "0x" + f"{123:064x}",
        "blockNumber": "0x10",
        "transactionHash": "0x" + ("ab" * 32),
        "logIndex": "0x2",
    }
    edge = normalize_transfer_log(log, edge_type="erc20_transfer", timestamp_unix=42)
    assert edge["source_address"] == source
    assert edge["target_address"] == target
    assert edge["amount"] == "123"
    assert edge["block_number"] == 16
    assert edge["timestamp_unix"] == 42
    assert transfer_log_endpoints(log, "erc20_transfer") == (source, target)


def test_polygon_consolidation_removes_cross_task_duplicates(tmp_path: Path) -> None:
    edge = {
        "schema_version": "1.0.0",
        "record_type": "chronicle_polygon_edge",
        "edge_id": "ab" * 32,
        "chain_id": 137,
        "edge_type": "erc20_transfer",
        "block_number": 16,
        "timestamp_unix": 1_700_000_000,
        "transaction_hash": "0x" + ("cd" * 32),
        "log_index": 2,
        "source_address": "0x" + ("12" * 20),
        "target_address": "0x" + ("34" * 20),
        "contract_address": "0x" + ("56" * 20),
        "token_id": None,
        "amount": "123",
        "observed_via": "polygon_json_rpc_eth_getLogs",
    }
    receipts = []
    for index in range(2):
        path = tmp_path / "polygon" / "tasks" / f"task={index}.jsonl.zst"
        _, size = write_jsonl_zst(path, [edge])
        receipts.append(
            {
                "task_id": f"task-{index}",
                "sha256": sha256_file(path),
                "path": path.relative_to(tmp_path).as_posix(),
                "rows": 1,
                "bytes": size,
            }
        )
    result = consolidate_polygon_tasks(tmp_path, "config-hash", receipts)
    assert result["indexed_current_tasks"] == 2
    assert result["indexed_tasks_total"] == 2
    assert result["unique_rows"] == 1
    assert result["duplicate_task_rows_removed"] == 1
    assert len(result["shards"]) == 1
