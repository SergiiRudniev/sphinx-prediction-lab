from __future__ import annotations

from pathlib import Path

import pytest
from scripts.build_h011_resolution_context import (
    ResolutionMarket,
    ResolutionScope,
    _scope_updates,
)

from sphinx_corpus.io import write_jsonl_zst


def _trade(
    *,
    wallet: str,
    timestamp: int,
    side: str,
    outcome: int,
    size: float,
    notional: float,
) -> dict[str, object]:
    return {
        "condition_id": "market",
        "wallet": wallet,
        "timestamp_unix": timestamp,
        "side": side,
        "outcome_index": outcome,
        "size": str(size),
        "notional_usd": str(notional),
    }


def test_resolution_proxy_is_winner_oriented_and_excludes_future_rows(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope=market"
    write_jsonl_zst(
        scope_path / "part.jsonl.zst",
        [
            _trade(
                wallet="0xA",
                timestamp=100,
                side="BUY",
                outcome=0,
                size=10,
                notional=6,
            ),
            _trade(
                wallet="0xA",
                timestamp=150,
                side="SELL",
                outcome=1,
                size=10,
                notional=4,
            ),
            _trade(
                wallet="0xA",
                timestamp=201,
                side="BUY",
                outcome=1,
                size=100,
                notional=50,
            ),
        ],
    )
    rows, counts = _scope_updates(
        ResolutionScope(
            path=scope_path,
            scope_id="group",
            markets=(
                ResolutionMarket(
                    condition_id="market",
                    split="train",
                    resolution_unix=200,
                    payout0=1,
                ),
            ),
            expected_rows=3,
        )
    )
    assert counts["post_resolution_rows_excluded"] == 1
    assert len(rows) == 1
    assert rows[0]["wallet"] == "0xa"
    assert rows[0]["observed_trade_count"] == 2
    assert rows[0]["directional_edge"] == pytest.approx(1.0)
    assert rows[0]["pnl_proxy_usd"] == pytest.approx(8.0)


def test_resolution_proxy_validates_exact_source_count(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope=market"
    write_jsonl_zst(
        scope_path / "part.jsonl.zst",
        [
            _trade(
                wallet="0xa",
                timestamp=100,
                side="BUY",
                outcome=0,
                size=1,
                notional=0.5,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="source rows changed"):
        _scope_updates(
            ResolutionScope(
                path=scope_path,
                scope_id="group",
                markets=(
                    ResolutionMarket(
                        condition_id="market",
                        split="train",
                        resolution_unix=200,
                        payout0=1,
                    ),
                ),
                expected_rows=2,
            )
        )
