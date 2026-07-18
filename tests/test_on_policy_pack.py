from __future__ import annotations

import math
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from sphinx_trace.on_policy_pack import (
    aligned_execution_arrays,
    build_logged_execution_index,
    build_payout_map,
)


def _decision(decision_id: str, row: int, action: str) -> dict[str, object]:
    return {
        "record_type": "h010_decision_audit",
        "decision_id": decision_id,
        "action": action,
        "condition_id": "condition",
        "feature_ref": {"date": "2026-01-01", "row": row},
    }


def test_logged_execution_value_attributes_partial_fill_and_terminal_payout() -> None:
    payouts = {"condition": {"token0": Decimal(1), "token1": Decimal(0)}}
    records = [
        _decision("call", 4, "CALL_OUTCOME_0"),
        {
            "record_type": "h010_order_audit",
            "order_id": "order",
            "decision_id": "call",
            "condition_id": "condition",
            "token_id": "token0",
            "side": "BUY",
            "requested_shares": "10",
        },
        {
            "record_type": "h010_fill_audit",
            "order_id": "order",
            "side": "BUY",
            "shares": "5",
            "notional_usd": "2",
            "fee_usd": "0.02",
        },
        _decision("skip", 9, "SKIP"),
        {
            "record_type": "h010_resolution_audit",
            "condition_id": "condition",
            "payouts": ["1", "0"],
        },
    ]

    index = build_logged_execution_index(records, payouts, reference_size=0.05)
    arrays = aligned_execution_arrays(
        index,
        date="2026-01-01",
        expected_row_indices=np.asarray([9, 4], dtype=np.int64),
    )

    conditional = math.log(1.0 + 0.05 * (2.98 / 2.02))
    assert index.action_counts == {"CALL_OUTCOME_0": 1, "SKIP": 1}
    assert index.orders == 1
    assert index.fills == 1
    assert index.filled_decisions == 1
    assert arrays["behavior_action_ids.npy"].tolist() == [2, 0]
    assert arrays["realized_pnl_usd.npy"].tolist() == pytest.approx([0.0, 2.98])
    assert arrays["executed_cost_usd.npy"].tolist() == pytest.approx([0.0, 2.02])
    assert arrays["execution_fractions.npy"].tolist() == pytest.approx([0.0, 0.5])
    assert arrays["realized_conditional_log_utilities.npy"].tolist() == pytest.approx(
        [0.0, conditional]
    )
    assert arrays["realized_action_values.npy"].tolist() == pytest.approx(
        [0.0, 0.5 * conditional]
    )


def test_unfilled_call_receives_zero_logged_value_without_becoming_skip_label() -> None:
    payouts = {"condition": {"token0": Decimal(1), "token1": Decimal(0)}}
    records = [
        _decision("call", 1, "CALL_OUTCOME_0"),
        {
            "record_type": "h010_order_audit",
            "order_id": "order",
            "decision_id": "call",
            "condition_id": "condition",
            "token_id": "token0",
            "side": "BUY",
            "requested_shares": "10",
        },
        {
            "record_type": "h010_resolution_audit",
            "condition_id": "condition",
            "payouts": ["1", "0"],
        },
    ]

    index = build_logged_execution_index(records, payouts, reference_size=0.05)
    target = index.targets[("2026-01-01", 1)]

    assert target.action_id == 0
    assert target.realized_action_value == 0.0
    assert target.execution_fraction == 0.0


def test_logged_execution_index_rejects_sell_attribution() -> None:
    payouts = {"condition": {"token0": Decimal(1), "token1": Decimal(0)}}
    records = [
        _decision("call", 1, "CALL_OUTCOME_0"),
        {
            "record_type": "h010_order_audit",
            "order_id": "order",
            "decision_id": "call",
            "condition_id": "condition",
            "token_id": "token0",
            "side": "SELL",
            "requested_shares": "10",
        },
    ]

    with pytest.raises(RuntimeError, match="cannot attribute a SELL"):
        build_logged_execution_index(records, payouts, reference_size=0.05)


def test_payout_map_preserves_contract_token_order() -> None:
    contracts = {
        "condition": SimpleNamespace(token_ids=("token0", "token1")),
    }
    resolutions = [
        SimpleNamespace(condition_id="condition", payouts=(Decimal(0), Decimal(1)))
    ]

    result = build_payout_map(contracts, resolutions)

    assert result == {"condition": {"token0": Decimal(0), "token1": Decimal(1)}}
