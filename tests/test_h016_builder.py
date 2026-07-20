from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.build_h016_fee_schedule import (
    MarketInfoCache,
    MarketTradeCache,
    ReceiptCache,
    _merge_receipt_payload,
    _source_contract,
    seed_qualified_caches,
)

from sphinx_corpus.io import atomic_json, sha256_file, write_jsonl_zst


def test_partial_replay_contract_binds_checkpoint_and_every_shard(tmp_path: Path) -> None:
    tape_dir = tmp_path / "tape"
    replay_dir = tmp_path / "partial-replay"
    tape_dir.mkdir()
    (replay_dir / "shards").mkdir(parents=True)
    corpus_config = tmp_path / "corpus.json"
    atomic_json(tape_dir / "manifest.json", {"tape": 1})
    corpus_config.write_text("{}", encoding="utf-8")
    (replay_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    shard = replay_dir / "shards" / "day-0001.jsonl.zst"
    shard.write_bytes(b"shard-one")

    first = _source_contract(tape_dir, (replay_dir,), corpus_config)
    assert first["replay_manifests"][replay_dir.name].startswith("partial:")

    shard.write_bytes(b"shard-two")
    second = _source_contract(tape_dir, (replay_dir,), corpus_config)
    assert first["replay_manifests"] != second["replay_manifests"]


def test_partial_replay_contract_rejects_uncheckpointed_input(tmp_path: Path) -> None:
    tape_dir = tmp_path / "tape"
    replay_dir = tmp_path / "broken-replay"
    tape_dir.mkdir()
    replay_dir.mkdir()
    corpus_config = tmp_path / "corpus.json"
    atomic_json(tape_dir / "manifest.json", {"tape": 1})
    corpus_config.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial checkpoint"):
        _source_contract(tape_dir, (replay_dir,), corpus_config)


def test_receipt_proof_merge_preserves_distinct_refund_logs() -> None:
    transaction_hash = "0x" + "1" * 64
    first = {
        "transaction_hash": transaction_hash,
        "status": "0x1",
        "block_number": "0x10",
        "block_hash": "0x" + "2" * 64,
        "transaction_index": "0x1",
        "logs": [{"logIndex": "0x1", "data": "0x01"}],
    }
    second = {
        **first,
        "logs": [{"logIndex": "0x2", "data": "0x02"}],
    }

    merged = _merge_receipt_payload(None, first)
    merged = _merge_receipt_payload(merged, second)

    assert merged["transactionHash"] == transaction_hash
    assert {row["logIndex"] for row in merged["logs"]} == {"0x1", "0x2"}


def test_seed_qualified_caches_verifies_and_rehydrates_proofs(
    tmp_path: Path,
) -> None:
    schedule_dir = tmp_path / "schedule"
    cache_dir = tmp_path / "cache"
    schedule_dir.mkdir()
    transaction_hash = "0x" + "3" * 64
    condition_id = "0x" + "4" * 64
    receipt_path = schedule_dir / "source-receipts.jsonl.zst"
    market_info_path = schedule_dir / "source-market-info.jsonl.zst"
    market_trade_path = schedule_dir / "source-market-trades.jsonl.zst"
    receipt_rows = [
        {
            "record_type": "h016_fee_receipt_proof",
            "transaction_hash": transaction_hash,
            "status": "0x1",
            "block_number": "0x10",
            "block_hash": "0x" + "5" * 64,
            "transaction_index": "0x1",
            "logs": [{"logIndex": "0x1", "data": "0x01"}],
        }
    ]
    market_info = {"c": condition_id, "fd": {"r": 0.25}}
    trade = {"conditionId": condition_id, "transactionHash": transaction_hash}
    write_jsonl_zst(receipt_path, receipt_rows)
    write_jsonl_zst(
        market_info_path,
        [
            {
                "record_type": "h016_fee_market_info_proof",
                "condition_id": condition_id,
                "payload_sha256": _stable_json_hash(market_info),
                "payload": market_info,
            }
        ],
    )
    write_jsonl_zst(
        market_trade_path,
        [
            {
                "record_type": "h016_fee_market_trade_proof",
                "condition_id": condition_id,
                "row_sha256": _stable_json_hash(trade),
                "row": trade,
            }
        ],
    )
    atomic_json(
        schedule_dir / "manifest.json",
        {
            "record_type": "h016_fee_schedule_manifest",
            "valid": True,
            "receipt_proof_path": receipt_path.name,
            "receipt_proof_sha256": sha256_file(receipt_path),
            "receipt_proof_rows": 1,
            "market_info_path": market_info_path.name,
            "market_info_sha256": sha256_file(market_info_path),
            "market_info_rows": 1,
            "market_trade_path": market_trade_path.name,
            "market_trade_sha256": sha256_file(market_trade_path),
            "market_trade_rows": 1,
        },
    )

    summary = seed_qualified_caches((schedule_dir,), cache_dir)
    assert summary["receipt_proofs"] == 1
    receipt_cache = ReceiptCache(cache_dir / "receipt-cache.sqlite3")
    market_info_cache = MarketInfoCache(cache_dir / "market-info-cache.sqlite3")
    market_trade_cache = MarketTradeCache(cache_dir / "market-trade-cache.sqlite3")
    try:
        assert receipt_cache.get(transaction_hash)["blockNumber"] == "0x10"
        assert market_info_cache.get(condition_id) == market_info
        assert market_trade_cache.get(condition_id) == [trade]
    finally:
        receipt_cache.close()
        market_info_cache.close()
        market_trade_cache.close()


def _stable_json_hash(payload: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
