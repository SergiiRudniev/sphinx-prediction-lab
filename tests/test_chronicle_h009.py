from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from sphinx_trace.chronicle_h009 import (
    UnionFind,
    component_id_for_market,
    extract_json_int,
    extract_json_string,
    market_seed_from_atlas,
    snapshot_reasons,
    terminal_payout_from_atlas,
    trade_sort_key,
)


class GuardedPayload(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key == "outcomePrices":
            raise AssertionError("test terminal field was accessed")
        return None

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def atlas_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": "0xabc",
        "market_id": "42",
        "event_ids": ["event-a", "event-b"],
        "neg_risk": True,
        "resolution_status": "resolved",
        "source_payload": payload,
    }


def test_market_seed_excludes_terminal_price_access() -> None:
    seed = market_seed_from_atlas(atlas_row(GuardedPayload()))
    assert seed is not None
    assert seed.event_ids == ("event-a", "event-b")
    assert seed.neg_risk is True


def test_test_terminal_field_is_not_accessed() -> None:
    assert (
        terminal_payout_from_atlas(
            atlas_row(GuardedPayload()),
            split_id="test",
            label_splits=frozenset({"train", "validation", "calibration"}),
        )
        is None
    )


def test_development_terminal_payout_requires_one_hot_binary_resolution() -> None:
    row = atlas_row({"outcomePrices": '["1", "0"]'})
    assert terminal_payout_from_atlas(
        row,
        split_id="train",
        label_splits=frozenset({"train"}),
    ) == (1.0, 0.0)
    row["source_payload"] = {"outcomePrices": '["0.7", "0.3"]'}
    assert (
        terminal_payout_from_atlas(
            row,
            split_id="train",
            label_splits=frozenset({"train"}),
        )
        is None
    )


def test_connected_event_component_is_stable_and_includes_neg_risk_links() -> None:
    union = UnionFind()
    union.add_group(("event-a", "event-b"))
    union.add_group(("event-b", "event-c"))
    ids = union.component_ids()
    assert len({ids[event] for event in ("event-a", "event-b", "event-c")}) == 1
    assert component_id_for_market("0x1", ("event-a",), ids) == component_id_for_market(
        "0x2", ("event-c",), ids
    )
    assert component_id_for_market("0x-orphan", (), ids) != ids["event-a"]


@pytest.mark.parametrize(
    ("count", "timestamp", "last", "expected"),
    [
        (1, 100, None, ("first",)),
        (2, 101, 100, ("early_power_of_two",)),
        (3, 102, 101, ()),
        (1152, 103, 102, ("trade_stride",)),
        (3, 21_701, 100, ("heartbeat",)),
    ],
)
def test_adaptive_snapshot_schedule(
    count: int,
    timestamp: int,
    last: int | None,
    expected: tuple[str, ...],
) -> None:
    assert (
        snapshot_reasons(
            event_trade_count=count,
            trade_timestamp_unix=timestamp,
            last_snapshot_timestamp_unix=last,
        )
        == expected
    )


def test_raw_trade_extractors_preserve_global_sort_key() -> None:
    line = b'{"trade_id":"abcd","condition_id":"0x123","timestamp_unix":42,"wallet":"0xbeef"}\n'
    assert extract_json_string(line, b"condition_id") == "0x123"
    assert extract_json_int(line, b"timestamp_unix") == 42
    assert trade_sort_key(line) == (42, "abcd")
