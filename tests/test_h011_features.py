from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sphinx_trace.h011_features import (
    H011_FEATURE_NAMES,
    H011_FEATURE_WIDTH,
    H011_HLL_REGISTERS,
    hll_add,
    hll_estimate,
    lifecycle_fractions,
    parse_trade_payload,
    splitmix64,
    weighted_std,
)
from sphinx_trace.h011_kernel import compile_h011_kernel, python_h011_kernel


def test_h011_feature_contract_is_named_and_fixed_width() -> None:
    assert len(H011_FEATURE_NAMES) == H011_FEATURE_WIDTH == 128
    assert len(set(H011_FEATURE_NAMES)) == H011_FEATURE_WIDTH


@pytest.mark.parametrize(
    ("side", "outcome_index", "direction", "probability"),
    [
        ("BUY", 0, 1, 0.8),
        ("SELL", 0, -1, 0.8),
        ("BUY", 1, -1, 0.2),
        ("SELL", 1, 1, 0.2),
    ],
)
def test_trade_orientation_is_relative_to_catalog_outcome_zero(
    side: str,
    outcome_index: int,
    direction: int,
    probability: float,
) -> None:
    row = {
        "trade_id": "trade",
        "condition_id": "0xabc",
        "wallet": "0xDEF",
        "timestamp_unix": 100,
        "price": "0.8",
        "size": "10",
        "notional_usd": "8",
        "outcome_index": outcome_index,
        "side": side,
    }
    trade = parse_trade_payload(row)
    assert trade.direction_outcome0 == direction
    assert trade.outcome0_probability == pytest.approx(probability)
    assert trade.wallet == "0xdef"


def test_source_price_anomaly_is_retained_and_clamped_for_model_state() -> None:
    trade = parse_trade_payload(
        {
            "trade_id": "anomaly",
            "condition_id": "0xabc",
            "wallet": "0xdef",
            "timestamp_unix": 100,
            "price": "1.1140588235",
            "size": "68",
            "notional_usd": "75.755999998",
            "outcome_index": 1,
            "side": "SELL",
        }
    )
    assert trade.source_price_anomaly is True
    assert trade.raw_price == 1.0
    assert trade.outcome0_probability == 0.0


def test_hll_processes_every_identity_without_retaining_identity_tokens() -> None:
    registers = np.zeros(H011_HLL_REGISTERS, dtype=np.uint8)
    for value in range(10_000):
        hll_add(registers, value)
    estimate = hll_estimate(registers)
    assert estimate == pytest.approx(10_000, rel=0.3)
    assert registers.nbytes == H011_HLL_REGISTERS
    assert splitmix64(7) == splitmix64(7)


def test_lifecycle_and_weighted_moments_are_bounded() -> None:
    assert lifecycle_fractions(150, 100, 200) == (0.5, 0.5)
    assert lifecycle_fractions(50, 100, 200) == (0.0, 1.0)
    assert lifecycle_fractions(250, 100, 200) == (1.0, 0.0)
    assert weighted_std(6.0, 14.0, 3.0) == pytest.approx(math.sqrt(2 / 3))


def _kernel_arguments() -> tuple[Any, ...]:
    wallet_core = np.zeros((1, 18), dtype=np.float64)
    wallet_core[:, 13] = -1
    return (
        np.asarray([100, 200], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([0.7, 0.8], dtype=np.float32),
        np.asarray([0.7, 0.2], dtype=np.float32),
        np.asarray([10.0, 20.0], dtype=np.float32),
        np.asarray([7.0, 4.0], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int8),
        np.asarray([1, 0], dtype=np.int8),
        np.asarray([1, 1], dtype=np.int8),
        np.asarray([-1, 0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([50], dtype=np.int64),
        np.asarray([300], dtype=np.int64),
        np.asarray([1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        1,
        np.asarray([150], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        np.asarray([1], dtype=np.int8),
        0,
        wallet_core,
        np.zeros((1, 12), dtype=np.float64),
        np.zeros((1, 17), dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
        np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
        np.zeros((1, 18), dtype=np.float64),
        np.zeros((1, 13), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
        np.zeros(12, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
        np.zeros((1, H011_HLL_REGISTERS), dtype=np.uint8),
        np.zeros((1, H011_FEATURE_WIDTH), dtype=np.float32),
    )


def test_numba_kernel_matches_reference_and_uses_every_trade() -> None:
    reference = _kernel_arguments()
    compiled = _kernel_arguments()
    assert python_h011_kernel()(*reference) == 1
    assert compile_h011_kernel()(*compiled) == 1
    np.testing.assert_allclose(compiled[-1], reference[-1], rtol=1e-6, atol=1e-6)
    assert np.isfinite(compiled[-1]).all()
    wallet_core = compiled[24]
    assert isinstance(wallet_core, np.ndarray)
    assert wallet_core[0, 0] == 2
    assert compiled[-1][0, 11] == pytest.approx(0.8)


def test_resolution_update_is_not_visible_in_the_same_second() -> None:
    arguments = list(_kernel_arguments())
    arguments[18] = np.asarray([200], dtype=np.int64)
    arguments[19] = np.asarray([1], dtype=np.int32)
    pointer = python_h011_kernel()(*arguments)
    assert pointer == 0
    assert arguments[-1][0, 85] == 0.0
