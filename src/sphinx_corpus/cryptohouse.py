"""Read-only access to the public CryptoHouse ClickHouse endpoint."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

CRYPTOHOUSE_URL = "https://crypto-clickhouse.clickhouse.com"


class CryptoHouseError(RuntimeError):
    pass


class CryptoHouseQuotaError(CryptoHouseError):
    def __init__(self, message: str, reset_at: datetime | None) -> None:
        super().__init__(message)
        self.reset_at = reset_at

    @property
    def wait_seconds(self) -> float:
        if self.reset_at is None:
            return 60.0
        return max((self.reset_at - datetime.now(tz=UTC)).total_seconds() + 2.0, 2.0)


def _quota_reset(detail: str) -> datetime | None:
    match = re.search(r"Interval will end at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", detail)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


class CryptoHouseClient:
    def __init__(
        self,
        url: str = CRYPTOHOUSE_URL,
        *,
        timeout_seconds: float = 75.0,
        retries: int = 5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.retries = retries
        self.client = httpx.Client(
            base_url=self.url,
            auth=("crypto", ""),
            timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "text/plain; charset=utf-8",
                "User-Agent": "sphinx-prediction-lab/0.1",
            },
            transport=transport,
        )

    @property
    def endpoint_fingerprint(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> CryptoHouseClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def query_json(self, query: str, *, query_id: str | None = None) -> dict[str, Any]:
        normalized = query.lstrip().upper()
        if not normalized.startswith(("SELECT", "WITH", "SHOW", "DESCRIBE")):
            raise ValueError("CryptoHouse client accepts read-only SQL only")
        last_error: Exception | None = None
        params = {"query_id": query_id} if query_id else None
        for attempt in range(self.retries):
            try:
                response = self.client.post("/", params=params, content=query.encode())
                if response.status_code == 429 or response.status_code >= 500:
                    detail = " ".join(response.text[:500].split())
                    if "QUOTA_EXCEEDED" in detail:
                        raise CryptoHouseQuotaError(
                            f"CryptoHouse quota exceeded: {detail}",
                            _quota_reset(detail),
                        )
                    raise CryptoHouseError(
                        f"CryptoHouse HTTP {response.status_code}: {detail or 'empty response'}"
                    )
                response.raise_for_status()
                payload: object = response.json()
                if not isinstance(payload, dict):
                    raise CryptoHouseError("CryptoHouse response is not an object")
                if payload.get("exception"):
                    raise CryptoHouseError(str(payload["exception"]))
                data = payload.get("data")
                if not isinstance(data, list):
                    raise CryptoHouseError("CryptoHouse response has no data array")
                return payload
            except (httpx.HTTPError, ValueError, CryptoHouseError) as error:
                last_error = error
                if isinstance(error, CryptoHouseQuotaError):
                    raise
                if attempt + 1 == self.retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise CryptoHouseError(
            f"CryptoHouse query failed after {self.retries} attempts: {last_error}"
        )


def single_array(payload: dict[str, Any], column: str) -> list[Any]:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise CryptoHouseError("Expected exactly one CryptoHouse result row")
    value = data[0].get(column)
    if not isinstance(value, list):
        raise CryptoHouseError(f"CryptoHouse result column {column!r} is not an array")
    return value
