"""Point-in-time Polygon transfer graph primitives for Sphinx Chronicle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ERC1155_TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
ERC1155_TRANSFER_BATCH_TOPIC = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"


@dataclass(frozen=True)
class TransferQuery:
    edge_type: str
    topics: list[Any]


def address_topic(address: str) -> str:
    normalized = address.lower().removeprefix("0x")
    if len(normalized) != 40 or any(value not in "0123456789abcdef" for value in normalized):
        raise ValueError(f"Invalid EVM address: {address}")
    return "0x" + ("0" * 24) + normalized


def topic_address(topic: object) -> str:
    value = str(topic).lower().removeprefix("0x")
    if len(value) != 64:
        raise ValueError(f"Invalid address topic: {topic}")
    return "0x" + value[-40:]


def transfer_log_endpoints(log: dict[str, Any], edge_type: str) -> tuple[str, str]:
    topics_value = log.get("topics")
    if not isinstance(topics_value, list):
        raise ValueError("Transfer log has no topics")
    if edge_type == "erc20_transfer":
        if len(topics_value) < 3:
            raise ValueError("ERC-20 Transfer log has fewer than three topics")
        return topic_address(topics_value[1]), topic_address(topics_value[2])
    if len(topics_value) < 4:
        raise ValueError("ERC-1155 Transfer log has fewer than four topics")
    return topic_address(topics_value[2]), topic_address(topics_value[3])


def transfer_queries(kind: str, wallets: list[str]) -> tuple[TransferQuery, ...]:
    wallet_topics = [address_topic(wallet) for wallet in wallets]
    if not wallet_topics:
        return ()
    if kind == "erc20":
        return (
            TransferQuery("erc20_transfer", [ERC20_TRANSFER_TOPIC, wallet_topics, None]),
            TransferQuery("erc20_transfer", [ERC20_TRANSFER_TOPIC, None, wallet_topics]),
        )
    if kind == "erc1155":
        return (
            TransferQuery(
                "erc1155_transfer_single",
                [ERC1155_TRANSFER_SINGLE_TOPIC, None, wallet_topics, None],
            ),
            TransferQuery(
                "erc1155_transfer_single",
                [ERC1155_TRANSFER_SINGLE_TOPIC, None, None, wallet_topics],
            ),
            TransferQuery(
                "erc1155_transfer_batch",
                [ERC1155_TRANSFER_BATCH_TOPIC, None, wallet_topics, None],
            ),
            TransferQuery(
                "erc1155_transfer_batch",
                [ERC1155_TRANSFER_BATCH_TOPIC, None, None, wallet_topics],
            ),
        )
    raise ValueError(f"Unsupported graph contract kind: {kind}")


def _uint256(data: str, word: int) -> int:
    raw = data.removeprefix("0x")
    start = word * 64
    end = start + 64
    if len(raw) < end:
        raise ValueError("EVM data is shorter than the requested word")
    return int(raw[start:end], 16)


def normalize_transfer_log(
    log: dict[str, Any],
    *,
    edge_type: str,
    timestamp_unix: int,
) -> dict[str, Any]:
    topics_value = log.get("topics")
    if not isinstance(topics_value, list):
        raise ValueError("Transfer log has no topics")
    topics = [str(value) for value in topics_value]
    if edge_type == "erc20_transfer":
        if len(topics) < 3:
            raise ValueError("ERC-20 Transfer log has fewer than three topics")
        source, target = transfer_log_endpoints(log, edge_type)
        token_id: str | None = None
        amount: str | None = str(_uint256(str(log.get("data") or "0x"), 0))
    else:
        if len(topics) < 4:
            raise ValueError("ERC-1155 Transfer log has fewer than four topics")
        source, target = transfer_log_endpoints(log, edge_type)
        if edge_type == "erc1155_transfer_single":
            token_id = str(_uint256(str(log.get("data") or "0x"), 0))
            amount = str(_uint256(str(log.get("data") or "0x"), 1))
        else:
            token_id = None
            amount = None
    transaction_hash = str(log["transactionHash"]).lower()
    log_index = int(str(log["logIndex"]), 16)
    contract_address = str(log["address"]).lower()
    edge_id = hashlib.sha256(
        f"137|{transaction_hash}|{log_index}|{contract_address}".encode()
    ).hexdigest()
    return {
        "schema_version": "1.0.0",
        "record_type": "chronicle_polygon_edge",
        "edge_id": edge_id,
        "chain_id": 137,
        "edge_type": edge_type,
        "block_number": int(str(log["blockNumber"]), 16),
        "timestamp_unix": timestamp_unix,
        "transaction_hash": transaction_hash,
        "log_index": log_index,
        "source_address": source,
        "target_address": target,
        "contract_address": contract_address,
        "token_id": token_id,
        "amount": amount,
        "observed_via": "polygon_json_rpc_eth_getLogs",
    }
