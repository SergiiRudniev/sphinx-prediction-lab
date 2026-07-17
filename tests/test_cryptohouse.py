from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from scripts.build_h011_actor_context import ActorTask, _actor_rows, actor_query

from sphinx_corpus.cryptohouse import (
    CryptoHouseClient,
    CryptoHouseError,
    CryptoHouseQuotaError,
    single_array,
)


def test_client_accepts_read_only_query_and_parses_one_array() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        return httpx.Response(
            200,
            json={"data": [{"actors": [["0xabc", 1]]}], "rows": 1},
        )

    client = CryptoHouseClient(transport=httpx.MockTransport(handler))
    try:
        payload = client.query_json("SELECT 1 FORMAT JSON")
    finally:
        client.close()
    assert single_array(payload, "actors") == [["0xabc", 1]]


def test_client_rejects_write_sql_and_server_exception() -> None:
    client = CryptoHouseClient(
        retries=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [], "exception": "quota"})
        ),
    )
    try:
        with pytest.raises(ValueError, match="read-only"):
            client.query_json("DROP TABLE anything")
        with pytest.raises(CryptoHouseError, match="quota"):
            client.query_json("SELECT 1")
    finally:
        client.close()


def test_actor_query_is_partitioned_and_rows_are_available_after_window() -> None:
    task = ActorTask(
        datetime(2025, 7, 16, tzinfo=UTC),
        datetime(2025, 8, 1, tzinfo=UTC),
        7,
        512,
    )
    query = actor_query(task)
    assert "cityHash64(lower(maker)) % 512 = 7" in query
    assert "cityHash64(lower(taker)) % 512 = 7" in query
    assert "FORMAT JSON" in query
    values = [
        [
            "0x" + ("12" * 20),
            2,
            3,
            4,
            1,
            10.0,
            2.0,
            20.0,
            5,
            4,
            0.5,
            0.1,
        ]
    ]
    rows = _actor_rows(task, json.loads(json.dumps(values)))
    assert rows[0]["available_at"] == "2025-08-01T00:00:00Z"
    assert rows[0]["maker_fills"] == 2


def test_quota_response_is_not_retried_immediately() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            500,
            text=(
                "QUOTA_EXCEEDED Interval will end at 2099-01-01 00:00:00. Name of quota template"
            ),
        )

    client = CryptoHouseClient(retries=5, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(CryptoHouseQuotaError) as raised:
            client.query_json("SELECT 1")
    finally:
        client.close()
    assert calls == 1
    assert raised.value.reset_at == datetime(2099, 1, 1, tzinfo=UTC)
