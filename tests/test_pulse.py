from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import zstandard

from sphinx_pulse.collector import extract_token_ids, select_markets
from sphinx_pulse.publisher import AssetSpec, DayPublisher
from sphinx_pulse.storage import HourlyZstdStore, count_zstd_jsonl


class FakeReleaseClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.assets: dict[str, AssetSpec] = {}

    def ensure_release(self, day: str) -> dict[str, Any]:
        return {"id": 42, "html_url": f"https://example.test/pulse-{day}"}

    def upload_verified(self, release: dict[str, Any], asset: AssetSpec) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("upload failed")
        self.assets[asset.name] = asset
        return {
            "name": asset.name,
            "size": asset.size,
            "digest": f"sha256:{asset.digest}",
        }

    def verify_assets(self, release_id: int, assets: Iterable[AssetSpec]) -> None:
        values = list(assets)
        assert release_id == 42
        assert {asset.name for asset in values} == set(self.assets)


def test_extracts_only_tradable_clob_tokens() -> None:
    events = [
        {
            "markets": [
                {
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "acceptingOrders": True,
                    "clobTokenIds": '["yes", "no"]',
                },
                {
                    "active": True,
                    "closed": True,
                    "enableOrderBook": True,
                    "acceptingOrders": True,
                    "clobTokenIds": '["closed"]',
                },
            ]
        }
    ]

    assert extract_token_ids(events) == {"yes", "no"}


def test_market_selection_honors_volume_rank_and_cap() -> None:
    events = [
        {
            "id": "event",
            "markets": [
                {
                    "id": value,
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "acceptingOrders": True,
                    "volume24hr": volume,
                    "clobTokenIds": json.dumps([f"{value}-yes", f"{value}-no"]),
                }
                for value, volume in [("low", 1), ("high", 100), ("mid", 50)]
            ],
        }
    ]

    selected = select_markets(events, max_markets=2)

    assert [item["market"]["id"] for item in selected] == ["high", "mid"]


def test_hourly_store_writes_valid_concatenated_zstd(tmp_path: Path) -> None:
    store = HourlyZstdStore(tmp_path, max_part_bytes=1024 * 1024)
    timestamp = int(datetime(2026, 7, 16, 14, tzinfo=UTC).timestamp() * 1000)
    records = [{"received_at_ms": timestamp, "payload": {"sequence": index}} for index in range(3)]

    store.append(records[:2])
    store.append(records[2:])

    path = next(tmp_path.rglob("*.jsonl.zst"))
    assert count_zstd_jsonl(path) == 3
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source) as reader,
    ):
        decoded = [json.loads(line) for line in reader.read().splitlines()]
    assert [row["payload"]["sequence"] for row in decoded] == [0, 1, 2]


def _completed_day(data_dir: Path) -> Path:
    day_dir = data_dir / "raw" / "date=2026-07-15" / "hour=23"
    store = HourlyZstdStore(data_dir / "raw")
    timestamp = int(datetime(2026, 7, 15, 23, tzinfo=UTC).timestamp() * 1000)
    store.append([{"received_at_ms": timestamp, "payload": {"event_type": "book"}}])
    return day_dir.parent


def test_publisher_deletes_day_only_after_remote_verification(tmp_path: Path) -> None:
    day_dir = _completed_day(tmp_path)
    client = FakeReleaseClient()
    publisher = DayPublisher(tmp_path, client, stable_age_seconds=0)

    receipt = publisher.publish_day(day_dir)

    assert receipt["release_id"] == 42
    assert not day_dir.exists()
    assert (tmp_path / "receipts" / "2026-07-15.json").is_file()
    assert any(name.endswith("manifest.json") for name in client.assets)


def test_publisher_keeps_day_when_upload_fails(tmp_path: Path) -> None:
    day_dir = _completed_day(tmp_path)
    publisher = DayPublisher(tmp_path, FakeReleaseClient(fail=True), stable_age_seconds=0)

    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish_day(day_dir)

    assert day_dir.is_dir()
    assert not (tmp_path / "receipts" / "2026-07-15.json").exists()


def test_only_completed_utc_days_are_eligible(tmp_path: Path) -> None:
    _completed_day(tmp_path)
    publisher = DayPublisher(tmp_path, FakeReleaseClient(), stable_age_seconds=0)

    eligible = publisher.eligible_days(today=date(2026, 7, 16))

    assert [path.name for path in eligible] == ["date=2026-07-15"]
