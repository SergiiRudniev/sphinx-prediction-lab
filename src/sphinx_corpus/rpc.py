from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import httpx


class RPCError(RuntimeError):
    pass


class PolygonRPC:
    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 60.0,
        retries: int = 5,
    ) -> None:
        self.url = url
        self.retries = retries
        self._ids = itertools.count(1)
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            headers={"User-Agent": "sphinx-prediction-lab/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> PolygonRPC:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _post(self, payload: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.post(self.url, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise RPCError(f"RPC HTTP {response.status_code}")
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError, RPCError) as exc:
                last_error = exc
                if attempt + 1 == self.retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RPCError(f"RPC request failed after {self.retries} attempts: {last_error}")

    def call(self, method: str, params: list[Any]) -> Any:
        request_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        result = self._post(payload)
        if not isinstance(result, dict):
            raise RPCError("RPC response is not an object")
        if result.get("error") is not None:
            raise RPCError(f"RPC {method} error: {result['error']}")
        if "result" not in result:
            raise RPCError(f"RPC {method} response has no result")
        return result["result"]

    def batch(self, calls: Iterable[tuple[str, list[Any]]]) -> list[Any]:
        requests: list[dict[str, Any]] = []
        ordered_ids: list[int] = []
        for method, params in calls:
            request_id = next(self._ids)
            ordered_ids.append(request_id)
            requests.append(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
        if not requests:
            return []
        payload = self._post(requests)
        if not isinstance(payload, list):
            raise RPCError("RPC batch response is not a list")
        by_id = {int(item["id"]): item for item in payload if isinstance(item, dict)}
        results: list[Any] = []
        for request_id in ordered_ids:
            item = by_id.get(request_id)
            if item is None:
                raise RPCError(f"RPC batch response missing id {request_id}")
            if item.get("error") is not None:
                raise RPCError(f"RPC batch error: {item['error']}")
            results.append(item.get("result"))
        return results

    def latest_block_number(self) -> int:
        return int(str(self.call("eth_blockNumber", [])), 16)

    def block(self, number: int) -> dict[str, Any]:
        result = self.call("eth_getBlockByNumber", [hex(number), False])
        if not isinstance(result, dict):
            raise RPCError(f"Block {number} not found")
        return result

    def block_timestamp(self, number: int) -> int:
        return int(str(self.block(number)["timestamp"]), 16)

    def block_at_or_after(self, timestamp: datetime, *, latest: int | None = None) -> int:
        target = int(timestamp.timestamp())
        high = latest if latest is not None else self.latest_block_number()
        if self.block_timestamp(high) < target:
            raise RPCError("Requested timestamp is newer than the latest RPC block")
        low = 0
        while low < high:
            middle = (low + high) // 2
            if self.block_timestamp(middle) < target:
                low = middle + 1
            else:
                high = middle
        return low

    def logs(
        self,
        address: str,
        topic: str,
        start_block: int,
        end_block: int,
    ) -> list[dict[str, Any]]:
        result = self.call(
            "eth_getLogs",
            [
                {
                    "address": address,
                    "fromBlock": hex(start_block),
                    "toBlock": hex(end_block),
                    "topics": [topic],
                }
            ],
        )
        if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
            raise RPCError("eth_getLogs result is not a list of objects")
        return result

    def timestamps(self, block_numbers: Iterable[int], batch_size: int) -> dict[int, int]:
        unique = sorted(set(block_numbers))
        output: dict[int, int] = {}
        for offset in range(0, len(unique), batch_size):
            batch = unique[offset : offset + batch_size]
            try:
                payloads = self.batch(
                    ("eth_getBlockByNumber", [hex(number), False]) for number in batch
                )
            except RPCError:
                payloads = [self.block(number) for number in batch]
            for number, payload in zip(batch, payloads, strict=True):
                if not isinstance(payload, dict):
                    raise RPCError(f"Block header missing for {number}")
                output[number] = int(str(payload["timestamp"]), 16)
        return output
