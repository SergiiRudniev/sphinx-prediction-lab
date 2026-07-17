from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, localcontext
from typing import Any

from Crypto.Hash import keccak

from sphinx_corpus import SCHEMA_VERSION
from sphinx_corpus.config import CorpusConfig, ExchangeContract, Window, utc_text
from sphinx_corpus.io import (
    atomic_json,
    check_disk_reserve,
    load_json,
    now_utc,
    write_jsonl_zst,
)
from sphinx_corpus.rpc import PolygonRPC, RPCError


def event_topic(signature: str) -> str:
    digest = keccak.new(digest_bits=256)
    digest.update(signature.encode())
    return "0x" + digest.hexdigest()


def _hex_int(value: Any) -> int:
    return int(str(value), 16)


def _address(topic: str) -> str:
    value = topic.removeprefix("0x")
    if len(value) != 64:
        raise ValueError("Indexed address topic must be 32 bytes")
    return "0x" + value[-40:].lower()


def _words(data: str) -> list[str]:
    value = data.removeprefix("0x")
    if len(value) % 64:
        raise ValueError("Event data length is not a multiple of 32 bytes")
    return [value[offset : offset + 64] for offset in range(0, len(value), 64)]


def _price(collateral_amount: int, token_amount: int) -> str | None:
    if token_amount == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return format(Decimal(collateral_amount) / Decimal(token_amount), "f")


def decode_order_fill(
    log: dict[str, Any],
    contract: ExchangeContract,
    *,
    chain_id: int,
    block_timestamp: int,
) -> dict[str, Any]:
    topics = log.get("topics")
    if not isinstance(topics, list) or len(topics) != 4:
        raise ValueError("OrderFilled log must have four topics")
    words = _words(str(log.get("data", "")))
    order_hash = str(topics[1]).lower()
    maker = _address(str(topics[2]))
    taker = _address(str(topics[3]))
    builder: str | None = None
    metadata: str | None = None
    if contract.protocol == "clob-v1":
        if len(words) != 5:
            raise ValueError("CLOB v1 OrderFilled must contain five data words")
        maker_asset_id, taker_asset_id, maker_amount, taker_amount, fee = (
            int(word, 16) for word in words
        )
        if maker_asset_id == 0 and taker_asset_id != 0:
            side = "BUY"
            token_id = taker_asset_id
            collateral_amount = maker_amount
            token_amount = taker_amount
        elif taker_asset_id == 0 and maker_asset_id != 0:
            side = "SELL"
            token_id = maker_asset_id
            collateral_amount = taker_amount
            token_amount = maker_amount
        else:
            raise ValueError("CLOB v1 fill does not contain exactly one collateral asset id")
    elif contract.protocol == "clob-v2":
        if len(words) != 7:
            raise ValueError("CLOB v2 OrderFilled must contain seven data words")
        side_value, token_id, maker_amount, taker_amount, fee = (
            int(word, 16) for word in words[:5]
        )
        if side_value not in (0, 1):
            raise ValueError(f"Unknown CLOB v2 side: {side_value}")
        side = "BUY" if side_value == 0 else "SELL"
        maker_asset_id = 0 if side == "BUY" else token_id
        taker_asset_id = token_id if side == "BUY" else 0
        collateral_amount = maker_amount if side == "BUY" else taker_amount
        token_amount = taker_amount if side == "BUY" else maker_amount
        builder = "0x" + words[5].lower()
        metadata = "0x" + words[6].lower()
    else:
        raise ValueError(f"Unsupported protocol: {contract.protocol}")

    timestamp = datetime.fromtimestamp(block_timestamp, tz=UTC)
    transaction_index = log.get("transactionIndex")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "order_fill",
        "source": "polygon_eth_getLogs",
        "chain_id": chain_id,
        "protocol": contract.protocol,
        "market_type": contract.market_type,
        "exchange_id": contract.id,
        "exchange_address": contract.address.lower(),
        "block_number": _hex_int(log["blockNumber"]),
        "block_hash": str(log.get("blockHash", "")).lower(),
        "block_time": utc_text(timestamp),
        "transaction_hash": str(log["transactionHash"]).lower(),
        "transaction_index": _hex_int(transaction_index) if transaction_index else None,
        "log_index": _hex_int(log["logIndex"]),
        "removed": bool(log.get("removed", False)),
        "order_hash": order_hash,
        "maker": maker,
        "taker": taker,
        "side": side,
        "token_id": str(token_id),
        "maker_asset_id": str(maker_asset_id),
        "taker_asset_id": str(taker_asset_id),
        "maker_amount_raw": str(maker_amount),
        "taker_amount_raw": str(taker_amount),
        "collateral_amount_raw": str(collateral_amount),
        "token_amount_raw": str(token_amount),
        "price": _price(collateral_amount, token_amount),
        "fee_raw": str(fee),
        "builder": builder,
        "metadata": metadata,
    }


class LedgerBackfill:
    def __init__(self, config: CorpusConfig, rpc: PolygonRPC) -> None:
        self.config = config
        self.rpc = rpc
        scan = config.payload["sources"]["ledger"]["scan"]
        self.initial_span = int(scan["initial_block_span"])
        self.minimum_span = int(scan["minimum_block_span"])
        self.maximum_span = int(scan["maximum_block_span"])
        self.target_logs = int(scan["target_logs_per_chunk"])
        self.header_batch_size = int(scan["block_header_batch_size"])

    def collect(
        self,
        *,
        exchange_ids: set[str] | None = None,
        max_chunks: int | None = None,
    ) -> dict[str, Any]:
        selected = [
            contract
            for contract in self.config.contracts
            if exchange_ids is None or contract.id in exchange_ids
        ]
        if exchange_ids is not None:
            missing = exchange_ids - {contract.id for contract in selected}
            if missing:
                raise ValueError(f"Unknown exchange ids: {', '.join(sorted(missing))}")
        return {
            contract.id: self._collect_contract(contract, max_chunks=max_chunks)
            for contract in selected
        }

    def probe(
        self,
        *,
        exchange_ids: set[str] | None = None,
        samples: int = 8,
        sample_blocks: int = 200,
    ) -> dict[str, Any]:
        if samples < 1 or sample_blocks < 1:
            raise ValueError("Probe samples and block span must be positive")
        selected = [
            contract
            for contract in self.config.contracts
            if exchange_ids is None or contract.id in exchange_ids
        ]
        if exchange_ids is not None:
            missing = exchange_ids - {contract.id for contract in selected}
            if missing:
                raise ValueError(f"Unknown exchange ids: {', '.join(sorted(missing))}")
        results: dict[str, Any] = {}
        for contract in selected:
            start_block, end_block = self._bounds(contract.active)
            total_blocks = end_block - start_block + 1
            available = max(0, total_blocks - sample_blocks)
            starts = [
                start_block + round(available * index / max(1, samples - 1))
                for index in range(samples)
            ]
            observations: list[dict[str, int]] = []
            topic = event_topic(contract.event_signature)
            for sample_start in sorted(set(starts)):
                sample_end = min(end_block, sample_start + sample_blocks - 1)
                count = len(self.rpc.logs(contract.address, topic, sample_start, sample_end))
                observations.append(
                    {
                        "start_block": sample_start,
                        "end_block": sample_end,
                        "blocks": sample_end - sample_start + 1,
                        "rows": count,
                    }
                )
            sampled_blocks = sum(item["blocks"] for item in observations)
            sampled_rows = sum(item["rows"] for item in observations)
            rows_per_block = sampled_rows / sampled_blocks
            results[contract.id] = {
                "start_block": start_block,
                "end_block": end_block,
                "total_blocks": total_blocks,
                "sampled_blocks": sampled_blocks,
                "sampled_rows": sampled_rows,
                "rows_per_block": rows_per_block,
                "estimated_rows": round(rows_per_block * total_blocks),
                "observations": observations,
            }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_utc(),
            "method": "evenly_spaced_block_samples",
            "samples_requested": samples,
            "sample_blocks": sample_blocks,
            "exchanges": results,
        }
        atomic_json(self.config.data_dir / "receipts" / "ledger-probe.json", receipt)
        return receipt

    def _bounds(self, active: Window) -> tuple[int, int]:
        effective = self.config.window.intersect(active)
        if effective is None:
            raise ValueError("Exchange active window does not intersect corpus window")
        latest = self.rpc.latest_block_number()
        confirmations = int(self.config.payload["network"]["confirmations"])
        confirmed_latest = latest - confirmations
        start_block = self.rpc.block_at_or_after(effective.start, latest=confirmed_latest)
        end_exclusive = self.rpc.block_at_or_after(effective.end, latest=confirmed_latest)
        return start_block, end_exclusive - 1

    def _initial_state(self, contract: ExchangeContract) -> dict[str, Any]:
        start_block, end_block = self._bounds(contract.active)
        return {
            "schema_version": SCHEMA_VERSION,
            "exchange_id": contract.id,
            "address": contract.address.lower(),
            "topic": event_topic(contract.event_signature),
            "start_block": start_block,
            "end_block": end_block,
            "next_block": start_block,
            "next_span": self.initial_span,
            "chunks": 0,
            "rows": 0,
            "complete": False,
            "updated_at": now_utc(),
        }

    def _collect_contract(
        self,
        contract: ExchangeContract,
        *,
        max_chunks: int | None,
    ) -> dict[str, Any]:
        state_path = self.config.data_dir / "state" / f"ledger-{contract.id}.json"
        state = load_json(state_path)
        if not state:
            state = self._initial_state(contract)
            atomic_json(state_path, state)
        if str(state.get("address")) != contract.address.lower():
            raise RuntimeError(f"Ledger state address mismatch for {contract.id}")
        if str(state.get("topic")) != event_topic(contract.event_signature):
            raise RuntimeError(f"Ledger state event topic mismatch for {contract.id}")
        if bool(state.get("complete")):
            return state

        chunks_this_run = 0
        while int(state["next_block"]) <= int(state["end_block"]):
            if max_chunks is not None and chunks_this_run >= max_chunks:
                break
            check_disk_reserve(
                self.config.data_dir,
                float(self.config.payload["storage"]["minimum_free_gib"]),
            )
            start_block = int(state["next_block"])
            span = max(self.minimum_span, min(self.maximum_span, int(state["next_span"])))
            logs: list[dict[str, Any]]
            while True:
                end_block = min(int(state["end_block"]), start_block + span - 1)
                try:
                    logs = self.rpc.logs(
                        contract.address,
                        str(state["topic"]),
                        start_block,
                        end_block,
                    )
                except RPCError:
                    if span <= self.minimum_span:
                        raise
                    span = max(self.minimum_span, span // 2)
                    continue
                if len(logs) > self.target_logs and span > self.minimum_span:
                    span = max(self.minimum_span, span // 2)
                    continue
                break

            logs.sort(key=lambda item: (_hex_int(item["blockNumber"]), _hex_int(item["logIndex"])))
            timestamps = self.rpc.timestamps(
                (_hex_int(item["blockNumber"]) for item in logs),
                self.header_batch_size,
            )
            observed_at = now_utc()
            raw_records: list[dict[str, Any]] = []
            normalized: list[dict[str, Any]] = []
            for item in logs:
                block_number = _hex_int(item["blockNumber"])
                raw_records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source": "polygon_eth_getLogs",
                        "observed_at": observed_at,
                        "exchange_id": contract.id,
                        "block_timestamp": timestamps[block_number],
                        "payload": item,
                    }
                )
                normalized.append(
                    decode_order_fill(
                        item,
                        contract,
                        chain_id=self.config.chain_id,
                        block_timestamp=timestamps[block_number],
                    )
                )

            partition = f"blocks={start_block:09d}-{end_block:09d}"
            raw_path = (
                self.config.data_dir
                / "raw"
                / "ledger"
                / f"exchange={contract.id}"
                / f"{partition}.jsonl.zst"
            )
            normalized_path = (
                self.config.data_dir
                / "normalized"
                / "ledger"
                / f"exchange={contract.id}"
                / f"{partition}.jsonl.zst"
            )
            write_jsonl_zst(raw_path, raw_records)
            write_jsonl_zst(normalized_path, normalized)
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "exchange_id": contract.id,
                "start_block": start_block,
                "end_block": end_block,
                "rows": len(normalized),
                "observed_at": observed_at,
            }
            atomic_json(
                self.config.data_dir
                / "receipts"
                / "ledger"
                / f"{contract.id}-{start_block:09d}-{end_block:09d}.json",
                receipt,
            )
            next_span = span
            if len(logs) < max(1, self.target_logs // 4):
                next_span = min(self.maximum_span, span * 2)
            elif len(logs) > int(self.target_logs * 0.8):
                next_span = max(self.minimum_span, span // 2)
            state = {
                **state,
                "next_block": end_block + 1,
                "next_span": next_span,
                "chunks": int(state.get("chunks", 0)) + 1,
                "rows": int(state.get("rows", 0)) + len(normalized),
                "complete": end_block >= int(state["end_block"]),
                "updated_at": now_utc(),
            }
            atomic_json(state_path, state)
            chunks_this_run += 1
        return state
