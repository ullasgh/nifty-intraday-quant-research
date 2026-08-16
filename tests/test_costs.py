"""Tests for transaction cost models."""

import time

import numpy as np
import pytest

from nifty_quant.execution.costs import (
    Charges,
    FillBatch,
    FixedBpsCost,
    NSEIntradayEquityCosts,
    ZeroCost,
    breakeven_cost_bps,
)


def test_intraday_round_trip_golden_contract():
    model = NSEIntradayEquityCosts()
    fills = FillBatch(
        notional=np.array([100_000.0, 100_000.0]),
        is_buy=np.array([True, False]),
    )
    charges = model.charges(fills)
    totals = charges.sum()

    print("NSEIntradayEquityCosts Rs 1,00,000 per leg round trip:")
    for name in (
        "brokerage",
        "stt",
        "exchange_txn",
        "sebi",
        "ipft",
        "stamp_duty",
        "gst",
        "total",
    ):
        print(f"{name:>12}: {getattr(totals, name):.4f}")

    assert np.isclose(totals.total, 82.65, atol=0.05, rtol=0.0)
    assert np.isclose(totals.brokerage, 40.00, atol=0.01, rtol=0.0)
    assert np.isclose(totals.stt, 25.00, atol=0.01, rtol=0.0)
    assert np.isclose(totals.exchange_txn, 5.94, atol=0.01, rtol=0.0)
    assert np.isclose(totals.sebi, 0.20, atol=0.01, rtol=0.0)
    assert np.isclose(totals.ipft, 0.20, atol=0.01, rtol=0.0)
    assert np.isclose(totals.stamp_duty, 3.00, atol=0.01, rtol=0.0)
    assert np.isclose(totals.gst, 8.3052, atol=0.01, rtol=0.0)


def test_intraday_round_trip_bps_contract():
    model = NSEIntradayEquityCosts()
    assert np.isclose(model.round_trip_bps(100_000), 8.3, atol=0.1, rtol=0.0)
    assert np.isclose(model.round_trip_bps(1_000_000), 4.0, atol=0.15, rtol=0.0)


def test_brokerage_kink_continuity():
    model = NSEIntradayEquityCosts()

    below_fills = FillBatch(notional=np.array([66_000.0]), is_buy=np.array([True]))
    below = model.charges(below_fills).brokerage[0]
    assert np.isclose(below, 0.0003 * 66_000, atol=1e-12, rtol=0.0)

    above_fills = FillBatch(notional=np.array([67_000.0]), is_buy=np.array([True]))
    above = model.charges(above_fills).brokerage[0]
    assert np.isclose(above, 20.0, atol=1e-12, rtol=0.0)

    kink_low = model.charges(
        FillBatch(notional=np.array([66666.666666]), is_buy=np.array([True]))
    )
    kink_high = model.charges(
        FillBatch(notional=np.array([66666.666668]), is_buy=np.array([True]))
    )
    assert np.isclose(kink_low.brokerage[0], 20.0, atol=0.01, rtol=0.0)
    assert np.isclose(kink_high.brokerage[0], 20.0, atol=0.01, rtol=0.0)


def test_intraday_charges_properties():
    model = NSEIntradayEquityCosts()
    rng = np.random.default_rng(0)
    notional = rng.uniform(1.0, 1_000_000.0, size=20)
    is_buy = rng.random(20) > 0.5
    charges = model.charges(FillBatch(notional=notional, is_buy=is_buy))

    components = (
        charges.brokerage,
        charges.stt,
        charges.exchange_txn,
        charges.sebi,
        charges.ipft,
        charges.stamp_duty,
        charges.gst,
    )
    for comp in components:
        assert np.all(comp >= 0)

    ascending = np.geomspace(1.0, 1_000_000.0, 20)
    mono_charges = model.charges(
        FillBatch(notional=ascending, is_buy=np.full(ascending.shape, True))
    )
    assert np.all(np.diff(mono_charges.total) >= -1e-9)

    zero_charges = model.charges(
        FillBatch(notional=np.array([0.0]), is_buy=np.array([True]))
    )
    assert zero_charges.total[0] == 0.0

    buy_only = model.charges(
        FillBatch(notional=ascending, is_buy=np.full(ascending.shape, True))
    )
    assert np.all(buy_only.stt == 0.0)

    sell_only = model.charges(
        FillBatch(notional=ascending, is_buy=np.full(ascending.shape, False))
    )
    assert np.all(sell_only.stamp_duty == 0.0)


def test_gst_base_excludes_ipft_and_stamp_duty():
    model = NSEIntradayEquityCosts()
    rng = np.random.default_rng(1)
    notional = rng.uniform(1.0, 1_000_000.0, size=50)
    is_buy = rng.random(50) > 0.5
    charges = model.charges(FillBatch(notional=notional, is_buy=is_buy))

    expected_base = charges.brokerage + charges.exchange_txn + charges.sebi
    assert np.allclose(charges.gst, 0.18 * expected_base, atol=1e-9, rtol=0.0)

    hybrid_base = expected_base + charges.ipft + charges.stamp_duty
    assert np.any(charges.ipft + charges.stamp_duty > 0)
    assert not np.allclose(charges.gst, 0.18 * hybrid_base, atol=1e-9, rtol=0.0)


def test_zero_cost_returns_zero_float64_arrays():
    rng = np.random.default_rng(2)
    notional = rng.uniform(1.0, 1_000_000.0, size=10)
    is_buy = rng.random(10) > 0.5
    fills = FillBatch(notional=notional, is_buy=is_buy)
    charges = ZeroCost().charges(fills)

    for comp in (
        charges.brokerage,
        charges.stt,
        charges.exchange_txn,
        charges.sebi,
        charges.ipft,
        charges.stamp_duty,
        charges.gst,
    ):
        assert comp.dtype == np.float64
        assert comp.shape == fills.notional.shape
        assert np.all(comp == 0.0)


def test_vectorized_performance_and_manual_loop_match():
    rng = np.random.default_rng(123)
    notional = rng.uniform(0.0, 1_000_000.0, size=100_000)
    is_buy = rng.random(100_000) > 0.5
    fills = FillBatch(notional=notional, is_buy=is_buy)

    model = NSEIntradayEquityCosts()
    start = time.perf_counter()
    model.charges(fills)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

    n_500 = notional[:500]
    buy_500 = is_buy[:500]
    vec = model.charges(FillBatch(notional=n_500, is_buy=buy_500))

    brokerage = []
    stt = []
    exchange_txn = []
    sebi = []
    ipft = []
    stamp_duty = []
    gst = []
    for i in range(500):
        notional_i = n_500[i]
        brokerage_i = min(20.0, 0.0003 * notional_i)
        exchange_i = 0.0000297 * notional_i
        sebi_i = 0.000001 * notional_i
        stt_i = 0.0 if buy_500[i] else 0.00025 * notional_i
        stamp_i = 0.00003 * notional_i if buy_500[i] else 0.0
        ipft_i = 0.000001 * notional_i
        gst_i = 0.18 * (brokerage_i + exchange_i + sebi_i)

        brokerage.append(brokerage_i)
        stt.append(stt_i)
        exchange_txn.append(exchange_i)
        sebi.append(sebi_i)
        ipft.append(ipft_i)
        stamp_duty.append(stamp_i)
        gst.append(gst_i)

    ref = Charges(
        brokerage=np.asarray(brokerage, dtype=np.float64),
        stt=np.asarray(stt, dtype=np.float64),
        exchange_txn=np.asarray(exchange_txn, dtype=np.float64),
        sebi=np.asarray(sebi, dtype=np.float64),
        ipft=np.asarray(ipft, dtype=np.float64),
        stamp_duty=np.asarray(stamp_duty, dtype=np.float64),
        gst=np.asarray(gst, dtype=np.float64),
    )
    for vec_arr, ref_arr in (
        (vec.brokerage, ref.brokerage),
        (vec.stt, ref.stt),
        (vec.exchange_txn, ref.exchange_txn),
        (vec.sebi, ref.sebi),
        (vec.ipft, ref.ipft),
        (vec.stamp_duty, ref.stamp_duty),
        (vec.gst, ref.gst),
    ):
        assert np.allclose(vec_arr, ref_arr, atol=1e-12, rtol=0.0)


def test_breakeven_cost_bps_returns_zero_sharpe():
    rng = np.random.default_rng(42)
    gross_returns = rng.normal(loc=0.0005, scale=0.01, size=5000)
    turnover = np.full(5000, 1.0)

    cost_bps = breakeven_cost_bps(gross_returns, turnover)
    net_returns = gross_returns - (cost_bps / 10_000.0) * turnover
    sharpe = float(np.mean(net_returns) / np.std(net_returns, ddof=1))
    assert abs(sharpe) < 1e-6
    assert np.isfinite(cost_bps)


def test_breakeven_cost_bps_negative_edge_returns_zero_no_raise():
    rng = np.random.default_rng(7)
    gross_returns = rng.normal(loc=-0.0005, scale=0.01, size=1000)
    turnover = np.full(1000, 1.0)

    cost_bps = breakeven_cost_bps(gross_returns, turnover)
    assert cost_bps == 0.0
    assert np.isfinite(cost_bps)


def test_breakeven_cost_bps_zero_turnover_returns_nan_no_raise():
    rng = np.random.default_rng(8)
    gross_returns = rng.normal(loc=0.0005, scale=0.01, size=1000)
    turnover = np.zeros(1000)

    cost_bps = breakeven_cost_bps(gross_returns, turnover)
    assert np.isnan(cost_bps)


def test_breakeven_cost_bps_never_returns_inf():
    rng = np.random.default_rng(9)

    negative_edge = breakeven_cost_bps(
        rng.normal(loc=-0.001, scale=0.01, size=500), np.full(500, 1.0)
    )
    zero_turnover = breakeven_cost_bps(
        rng.normal(loc=0.001, scale=0.01, size=500), np.zeros(500)
    )
    positive_edge = breakeven_cost_bps(
        rng.normal(loc=0.0005, scale=0.01, size=500), np.full(500, 1.0)
    )

    for cost_bps in (negative_edge, zero_turnover, positive_edge):
        assert cost_bps != float("inf")
        assert cost_bps != float("-inf")


def test_fixed_bps_zero_matches_zero_cost_total():
    rng = np.random.default_rng(3)
    notional = rng.uniform(1.0, 1_000_000.0, size=20)
    is_buy = rng.random(20) > 0.5
    fills = FillBatch(notional=notional, is_buy=is_buy)

    fixed_total = FixedBpsCost(bps=0.0).charges(fills).total
    zero_total = ZeroCost().charges(fills).total
    assert np.allclose(fixed_total, zero_total, atol=0, rtol=0.0)


def test_fillbatch_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        FillBatch(notional=np.array([100.0, 200.0]), is_buy=np.array([True]))


def test_fillbatch_rejects_negative_notional():
    with pytest.raises(ValueError, match="non-negative"):
        FillBatch(notional=np.array([100.0, -1.0]), is_buy=np.array([True, False]))
