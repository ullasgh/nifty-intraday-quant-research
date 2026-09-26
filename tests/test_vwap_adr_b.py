"""Independent test suite (author B) for specs/vwap_adr_fade.md.

Written from the spec ALONE, before any implementation exists. Does not read, import or
reference the sibling suite (test_vwap_adr_a.py) or the implementation.

Strategy of this suite: slow, obviously-correct reference implementations written as
plain Python loops straight from the spec text (``_ref_*``), compared against the module
on many small randomized panels (NaN holes, random tradable masks, zero volumes, sessions
of arbitrary length incl. 60 and 105, sessions straddling 15:20, evening sessions entirely
after 15:20, float32 inputs), plus hand-built edge cases whose expected trades are written
out explicitly. The hand-built cases are also run through the reference so that a defect
in the reference itself shows up as a failure rather than silently weakening the
randomized comparisons.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from nifty_quant.backtest.metrics import sharpe_ratio
from nifty_quant.execution.costs import FillBatch, NSEIntradayEquityCosts
from nifty_quant.execution.fills import SqrtImpactSlippage
from nifty_quant.research.vwap_adr.features import (
    SESSION_START_MINUTE,
    SQUARE_OFF_MINUTE,
    adr_pct,
    max_excursion,
    session_range_pct,
    session_vwap,
)
from nifty_quant.research.vwap_adr.report import daily_pnl, summarize
from nifty_quant.research.vwap_adr.simulate import FadeParams, simulate_fade

COLS = [
    "symbol",
    "day_idx",
    "side",
    "signal_row",
    "entry_row",
    "entry_price",
    "exit_row",
    "exit_price",
    "exit_reason",
    "qty",
    "gross_pnl",
    "costs",
    "net_pnl",
    "gross_bps",
    "net_bps",
    "holding_bars",
]
INT_COLS = ["day_idx", "side", "signal_row", "entry_row", "exit_row", "holding_bars"]
FLOAT_COLS = [
    "entry_price",
    "exit_price",
    "qty",
    "gross_pnl",
    "costs",
    "net_pnl",
    "gross_bps",
    "net_bps",
]
STR_COLS = ["symbol", "exit_reason"]

SUMMARY_KEYS = {
    "trades",
    "trades_per_day",
    "win_rate",
    "pct_exit_vwap",
    "pct_exit_square_off",
    "pct_exit_no_bar",
    "mean_gross_bps",
    "mean_net_bps",
    "median_holding_bars",
    "total_gross_pnl",
    "total_net_pnl",
    "total_costs",
    "sharpe_net",
    "sharpe_gross",
    "max_concurrent",
    "worst_trade_net_bps",
    "worst_day_net_pnl",
    "long_trades",
    "short_trades",
}


# ---------------------------------------------------------------------------------------
# Reference implementations: plain loops, transcribed from the spec text.
# ---------------------------------------------------------------------------------------


def _fin(x: float) -> bool:
    return bool(np.isfinite(x))


def _ref_session_vwap(high, low, close, volume, mod, offs, start=556):
    high, low, close, volume = (np.asarray(a, dtype=np.float64) for a in (high, low, close, volume))
    n, m = close.shape
    out = np.full((n, m), np.nan)
    for s in range(m):
        for d in range(len(offs) - 1):
            pv = 0.0
            vv = 0.0
            for r in range(int(offs[d]), int(offs[d + 1])):
                if mod[r] < start:
                    continue  # NaN, and does not contribute
                h, lo, c, v = high[r, s], low[r, s], close[r, s], volume[r, s]
                if _fin(h) and _fin(lo) and _fin(c) and _fin(v) and v >= 0:
                    pv += (h + lo + c) / 3.0 * v
                    vv += v
                if vv > 0 and _fin(c):
                    out[r, s] = pv / vv
    return out


def _ref_session_range_pct(high, low, close, mod, offs, start=556):
    high, low, close = (np.asarray(a, dtype=np.float64) for a in (high, low, close))
    n_days = len(offs) - 1
    m = close.shape[1]
    out = np.full((n_days, m), np.nan)
    for s in range(m):
        for d in range(n_days):
            hi = -math.inf
            lo = math.inf
            last_c = None
            for r in range(int(offs[d]), int(offs[d + 1])):
                if mod[r] < start:
                    continue
                h, lw, c = high[r, s], low[r, s], close[r, s]
                if _fin(h) and _fin(lw) and _fin(c):
                    hi = max(hi, h)
                    lo = min(lo, lw)
                    last_c = c
            if last_c is None or last_c <= 0:
                continue
            out[d, s] = (hi - lo) / last_c
    return out


def _ref_adr_pct(rp, n):
    rp = np.asarray(rp, dtype=np.float64)
    nd, m = rp.shape
    out = np.full((nd, m), np.nan)
    for s in range(m):
        for d in range(nd):
            if d < n:
                continue
            window = [rp[j, s] for j in range(d - n, d)]
            if any(not _fin(x) for x in window):
                continue
            out[d, s] = sum(window) / n
    return out


def _ref_max_excursion(close, vwap, adr_day, mod, offs, start=556, end=920):
    close, vwap, adr_day = (np.asarray(a, dtype=np.float64) for a in (close, vwap, adr_day))
    n_days = len(offs) - 1
    m = close.shape[1]
    out = np.full((n_days, m), np.nan)
    for s in range(m):
        for d in range(n_days):
            a = adr_day[d, s]
            if not _fin(a) or a <= 0:
                continue
            best = None
            for r in range(int(offs[d]), int(offs[d + 1])):
                if not (start <= mod[r] < end):
                    continue
                c, w = close[r, s], vwap[r, s]
                if _fin(c) and _fin(w) and w > 0:
                    val = abs(c - w) / (w * a)
                    best = val if best is None else max(best, val)
            if best is not None:
                out[d, s] = best
    return out


def _ref_simulate(
    open_,
    close,
    volume,
    vwap,
    adr_day,
    tradable,
    mod,
    offs,
    symbols,
    *,
    k,
    notional,
    start=556,
    square_off=920,
    cost_model=None,
    slippage=None,
):
    open_, close, volume, vwap, adr_day = (
        np.asarray(a, dtype=np.float64) for a in (open_, close, volume, vwap, adr_day)
    )
    tradable = np.asarray(tradable, dtype=bool)
    n, m = close.shape
    rows = []

    def present(r, s):
        return _fin(close[r, s])

    def fillable(r, s):
        o, v = open_[r, s], volume[r, s]
        return _fin(o) and o > 0 and _fin(v) and v > 0

    for s in range(m):
        for d in range(len(offs) - 1):
            a, b = int(offs[d]), int(offs[d + 1])
            ad = adr_day[d, s]
            r = a
            while r < b:
                side = 0
                w = vwap[r, s]
                if (
                    present(r, s)
                    and _fin(w)
                    and w > 0
                    and _fin(ad)
                    and ad > 0
                    and start <= mod[r] < square_off
                ):
                    upper = w * (1 + k * ad)
                    lower = w * (1 - k * ad)
                    if close[r, s] > upper:
                        side = -1
                    elif close[r, s] < lower:
                        side = +1
                if side == 0:
                    r += 1
                    continue
                e = r + 1
                if not (e < b and mod[e] < square_off and fillable(e, s) and tradable[e, s]):
                    r += 1  # dropped, not deferred
                    continue
                entry_price = open_[e, s]
                qty = notional / entry_price
                x = None
                pending = False
                for q in range(e, b):
                    if pending:
                        if fillable(q, s):
                            x, exit_price, reason = q, open_[q, s], "vwap"
                            break
                        continue
                    if mod[q] >= square_off and fillable(q, s):
                        x, exit_price, reason = q, open_[q, s], "square_off"
                        break
                    if present(q, s) and _fin(vwap[q, s]):
                        if (side == 1 and close[q, s] >= vwap[q, s]) or (
                            side == -1 and close[q, s] <= vwap[q, s]
                        ):
                            pending = True
                if x is None:
                    for q in range(b - 1, e - 1, -1):
                        c, v = close[q, s], volume[q, s]
                        if _fin(c) and c > 0 and _fin(v) and v > 0:
                            x, exit_price, reason = q, c, "no_bar"
                            break
                assert x is not None, "reference: row e must qualify for no_bar"
                gross = side * qty * (exit_price - entry_price)
                en = qty * entry_price
                xn = qty * exit_price
                charges = 0.0
                if cost_model is not None:
                    fb = FillBatch(
                        notional=np.array([en, xn], dtype=np.float64),
                        is_buy=np.array([side == 1, side == -1], dtype=bool),
                    )
                    charges = float(cost_model.charges(fb).total.sum())
                slip = 0.0
                if slippage is not None:
                    for fn, fr in ((en, e), (xn, x)):
                        btv = close[fr, s] * volume[fr, s]
                        bps = float(np.asarray(slippage.bps(np.array([fn]), np.array([btv])))[0])
                        slip += fn * bps / 1e4
                costs = charges + slip
                net = gross - costs
                rows.append(
                    {
                        "symbol": str(symbols[s]),
                        "day_idx": d,
                        "side": side,
                        "signal_row": r,
                        "entry_row": e,
                        "entry_price": entry_price,
                        "exit_row": x,
                        "exit_price": exit_price,
                        "exit_reason": reason,
                        "qty": qty,
                        "gross_pnl": gross,
                        "costs": costs,
                        "net_pnl": net,
                        "gross_bps": gross / notional * 1e4,
                        "net_bps": net / notional * 1e4,
                        "holding_bars": x - e,
                    }
                )
                r = x  # re-evaluate entry starting at the exit row itself
    df = pd.DataFrame(rows, columns=COLS)
    if len(df):
        df = df.sort_values(["entry_row", "symbol"], kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def _grid(sessions):
    """sessions: list of (start_minute, n_bars). Contiguous 1-minute bars per session."""
    minutes = np.concatenate([np.arange(st, st + n) for st, n in sessions]).astype(np.int64)
    offs = np.array([0, *np.cumsum([n for _, n in sessions])], dtype=np.int64)
    return minutes, offs


def _hand(sessions, n_sym=1):
    """Flat panel: price 100 everywhere, vwap 100, adr 1%, all tradable, volume 1000.

    With k=1 the bands are exactly 99 / 101.
    """
    minutes, offs = _grid(sessions)
    n = len(minutes)
    return {
        "open_": np.full((n, n_sym), 100.0),
        "close": np.full((n, n_sym), 100.0),
        "volume": np.full((n, n_sym), 1000.0),
        "vwap": np.full((n, n_sym), 100.0),
        "adr_day": np.full((len(sessions), n_sym), 0.01),
        "tradable": np.ones((n, n_sym), dtype=bool),
        "minute_of_day": minutes,
        "day_offsets": offs,
        "symbols": [f"S{i}" for i in range(n_sym)],
    }


_ORDER = [
    "open_",
    "close",
    "volume",
    "vwap",
    "adr_day",
    "tradable",
    "minute_of_day",
    "day_offsets",
    "symbols",
]


def _sim(p, *, k=1.0, notional=100_000.0, start=556, square_off=920, **kw):
    params = FadeParams(k=k, notional=notional, start_minute=start, square_off_minute=square_off)
    return simulate_fade(*[p[x] for x in _ORDER], params, **kw)


def _ref(p, *, k=1.0, notional=100_000.0, start=556, square_off=920, **kw):
    return _ref_simulate(
        *[p[x] for x in _ORDER],
        k=k,
        notional=notional,
        start=start,
        square_off=square_off,
        **kw,
    )


def _assert_trades_equal(got: pd.DataFrame, exp: pd.DataFrame):
    assert list(got.columns) == COLS
    assert len(got) == len(exp), f"got {len(got)} trades, expected {len(exp)}\n{got}\n{exp}"
    got = got.reset_index(drop=True)
    exp = exp.reset_index(drop=True)
    for c in STR_COLS:
        assert [str(x) for x in got[c]] == [str(x) for x in exp[c]], c
    for c in INT_COLS:
        np.testing.assert_array_equal(
            got[c].to_numpy().astype(np.int64), exp[c].to_numpy().astype(np.int64), err_msg=c
        )
    for c in FLOAT_COLS:
        np.testing.assert_allclose(
            got[c].to_numpy(dtype=np.float64),
            exp[c].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-9,
            err_msg=c,
        )


def _check(p, expected_rows, **kw):
    """Assert both the module and the reference produce exactly ``expected_rows``.

    expected_rows: list of dicts with keys side, signal_row, entry_row, exit_row,
    exit_reason, and optionally entry_price / exit_price / symbol / day_idx.
    """
    got = _sim(p, **kw)
    ref = _ref(p, **kw)
    for name, df in (("module", got), ("reference", ref)):
        assert len(df) == len(expected_rows), f"{name}: {len(df)} trades\n{df}"
        df = df.reset_index(drop=True)
        for i, row in enumerate(expected_rows):
            for key, val in row.items():
                actual = df.loc[i, key]
                if isinstance(val, float):
                    assert actual == pytest.approx(val, rel=1e-12), (name, i, key)
                else:
                    assert actual == val, (name, i, key, actual, val)
    _assert_trades_equal(got, ref)
    return got


def _rand_feature_panel(seed, float32=False):
    rng = np.random.default_rng(seed)
    n_sym = int(rng.integers(1, 5))
    n_days = int(rng.integers(1, 6))
    sessions = []
    for _ in range(n_days):
        n_bars = int(rng.choice([1, 2, 5, 17, 60, 105]))
        st = int(rng.choice([553, 555, 556, 700, 900]))
        sessions.append((st, n_bars))
    if seed % 5 == 0:
        sessions.append((555, 375))
    minutes, offs = _grid(sessions)
    n = len(minutes)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (n, n_sym)), axis=0)
    high = close + np.abs(rng.normal(0, 0.3, (n, n_sym)))
    low = close - np.abs(rng.normal(0, 0.3, (n, n_sym)))
    volume = rng.integers(0, 5000, (n, n_sym)).astype(np.float64)
    u = rng.random((n, n_sym))
    volume[u < 0.08] = 0.0
    volume[(u >= 0.08) & (u < 0.11)] = -5.0  # negative volume: never contributes
    volume[(u >= 0.11) & (u < 0.15)] = np.nan
    close[rng.random((n, n_sym)) < 0.12] = np.nan
    high[rng.random((n, n_sym)) < 0.05] = np.nan
    low[rng.random((n, n_sym)) < 0.05] = np.nan
    bad = rng.random((n, n_sym)) < 0.02
    close[bad] = rng.choice([0.0, -1.0])
    if float32:
        high, low, close, volume = (a.astype(np.float32) for a in (high, low, close, volume))
    return high, low, close, volume, minutes, offs


def _rand_sim_panel(seed, float32=False):
    rng = np.random.default_rng(10_000 + seed)
    n_sym = int(rng.integers(1, 4))
    n_days = int(rng.integers(1, 5))
    sessions = []
    for _ in range(n_days):
        n_bars = int(rng.choice([6, 12, 25, 40, 60, 105]))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            st = 553  # includes pre-09:16 rows
        elif kind == 1:
            st = 920 - int(rng.integers(1, n_bars + 1))  # straddles (or just precedes) 15:20
        elif kind == 2:
            st = 1095  # evening (Muhurat-like) session: everything after 15:20
        else:
            st = int(rng.integers(556, 900))
        sessions.append((st, n_bars))
    minutes, offs = _grid(sessions)
    n = len(minutes)
    close = np.empty((n, n_sym))
    open_ = np.empty((n, n_sym))
    for d in range(n_days):
        a, b = offs[d], offs[d + 1]
        for s in range(n_sym):
            c = 100.0 + np.cumsum(rng.normal(0, 0.4, b - a))
            o = np.r_[100.0 + rng.normal(0, 0.2), c[:-1] + rng.normal(0, 0.2, b - a - 1)]
            close[a:b, s] = c
            open_[a:b, s] = o
    volume = rng.integers(1, 2000, (n, n_sym)).astype(np.float64)
    volume[rng.random((n, n_sym)) < 0.1] = 0.0
    hole = rng.random((n, n_sym)) < 0.12
    close[hole] = np.nan
    open_[hole] = np.nan
    volume[hole] = np.nan
    open_[rng.random((n, n_sym)) < 0.04] = np.nan  # present but not fillable
    tradable = rng.random((n, n_sym)) < 0.85
    vwap = np.full((n, n_sym), np.nan)
    for d in range(n_days):
        a, b = offs[d], offs[d + 1]
        for s in range(n_sym):
            tot = 0.0
            cnt = 0
            for r in range(a, b):
                if np.isfinite(close[r, s]):
                    tot += close[r, s]
                    cnt += 1
                    vwap[r, s] = tot / cnt
    u = rng.random((n, n_sym))
    vwap[u < 0.05] = np.nan
    vwap[(u >= 0.05) & (u < 0.07)] = 0.0
    vwap[(u >= 0.07) & (u < 0.08)] = -3.0
    adr = rng.uniform(0.001, 0.012, (n_days, n_sym))
    u = rng.random((n_days, n_sym))
    adr[u < 0.08] = np.nan
    adr[(u >= 0.08) & (u < 0.12)] = 0.0
    adr[(u >= 0.12) & (u < 0.15)] = -0.005
    if float32:
        open_, close, volume, vwap, adr = (
            x.astype(np.float32) for x in (open_, close, volume, vwap, adr)
        )
    p = {
        "open_": open_,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        "adr_day": adr,
        "tradable": tradable,
        "minute_of_day": minutes,
        "day_offsets": offs,
        "symbols": ["AAA", "BBB", "CCC"][:n_sym],
    }
    k = float(rng.choice([0.3, 0.8, 1.5]))
    start, square_off = (556, 920) if rng.random() < 0.7 else (560, 915)
    return p, {"k": k, "start": start, "square_off": square_off}


# ---------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------


def test_constants():
    assert SESSION_START_MINUTE == 556
    assert SQUARE_OFF_MINUTE == 920


# ---------------------------------------------------------------------------------------
# 1.1 session_vwap
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_session_vwap_matches_reference_randomized(seed):
    high, low, close, volume, mod, offs = _rand_feature_panel(seed, float32=seed % 3 == 0)
    snapshot = [a.copy() for a in (high, low, close, volume)]
    got = session_vwap(high, low, close, volume, mod, offs)
    exp = _ref_session_vwap(high, low, close, volume, mod, offs)
    assert got.dtype == np.float64
    assert got.shape == close.shape
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-9, equal_nan=True)
    for before, after in zip(snapshot, (high, low, close, volume)):
        np.testing.assert_array_equal(before, after)  # inputs untouched (no fill in place)


@pytest.mark.parametrize("seed", range(6))
def test_session_vwap_custom_start_minute(seed):
    high, low, close, volume, mod, offs = _rand_feature_panel(100 + seed)
    got = session_vwap(high, low, close, volume, mod, offs, start_minute=700)
    exp = _ref_session_vwap(high, low, close, volume, mod, offs, start=700)
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-9, equal_nan=True)


def test_session_vwap_resets_across_60_105_375_sessions():
    """Invariant 1: VWAP resets at every boundary; mixed session lengths in one panel."""
    rng = np.random.default_rng(7)
    minutes, offs = _grid([(555, 375), (555, 60), (555, 105), (555, 375), (555, 60)])
    n = len(minutes)
    close = 100 + np.cumsum(rng.normal(0, 0.5, (n, 2)), axis=0)
    high = close + 0.2
    low = close - 0.3
    volume = rng.integers(1, 1000, (n, 2)).astype(np.float64)
    got = session_vwap(high, low, close, volume, minutes, offs)
    np.testing.assert_allclose(
        got, _ref_session_vwap(high, low, close, volume, minutes, offs), rtol=1e-10, atol=1e-9,
        equal_nan=True,
    )
    tp = (high + low + close) / 3
    for d in range(len(offs) - 1):
        a = offs[d]
        assert np.all(np.isnan(got[a]))  # 09:15 row
        # First contributing row of every session is its own typical price: the sum reset.
        np.testing.assert_allclose(got[a + 1], tp[a + 1], rtol=1e-12)


def test_session_vwap_no_fixed_stride_assumption():
    """Same data split into different session lengths gives different, reference-exact VWAP."""
    rng = np.random.default_rng(3)
    n = 165
    close = 100 + np.cumsum(rng.normal(0, 0.5, (n, 1)), axis=0)
    high, low = close + 0.1, close - 0.1
    volume = rng.integers(1, 100, (n, 1)).astype(np.float64)
    minutes = np.r_[np.arange(555, 555 + 60), np.arange(555, 555 + 105)].astype(np.int64)
    offs = np.array([0, 60, 165])
    got = session_vwap(high, low, close, volume, minutes, offs)
    np.testing.assert_allclose(got, _ref_session_vwap(high, low, close, volume, minutes, offs),
                               rtol=1e-10, equal_nan=True)
    # Row 61 (09:16 of the 105-bar session) is its own typical price.
    assert got[61, 0] == pytest.approx((high[61, 0] + low[61, 0] + close[61, 0]) / 3, rel=1e-12)


def test_session_vwap_0915_row_excluded():
    """Invariant 2: 09:15 row is NaN and never contributes."""
    minutes, offs = _grid([(555, 10)])
    close = np.linspace(100, 101, 10).reshape(-1, 1)
    high, low = close + 0.5, close - 0.5
    volume = np.full((10, 1), 100.0)
    base = session_vwap(high, low, close, volume, minutes, offs)
    h2, l2, c2, v2 = high.copy(), low.copy(), close.copy(), volume.copy()
    h2[0], l2[0], c2[0], v2[0] = 1e6, 1.0, 5e5, 1e9
    moved = session_vwap(h2, l2, c2, v2, minutes, offs)
    assert np.isnan(base[0, 0]) and np.isnan(moved[0, 0])
    np.testing.assert_array_equal(base, moved)


def test_session_vwap_nan_row_equals_deleted_row():
    """Invariant 3: a NaN bar gets NaN VWAP; later rows equal the VWAP with that row deleted."""
    rng = np.random.default_rng(11)
    minutes, offs = _grid([(555, 30), (555, 20)])
    n = len(minutes)
    close = 100 + np.cumsum(rng.normal(0, 0.5, (n, 1)), axis=0)
    high, low = close + 0.3, close - 0.2
    volume = rng.integers(1, 500, (n, 1)).astype(np.float64)
    j = 12
    for a in (close, high, low, volume):
        a[j] = np.nan
    full = session_vwap(high, low, close, volume, minutes, offs)
    assert np.isnan(full[j, 0])
    keep = np.arange(n) != j
    offs_del = np.array([0, 29, 49])
    deleted = session_vwap(
        high[keep], low[keep], close[keep], volume[keep], minutes[keep], offs_del
    )
    np.testing.assert_allclose(full[keep], deleted, rtol=1e-12, equal_nan=True)


def test_session_vwap_present_but_noncontributing_row():
    """A present bar with NaN volume does not contribute but still gets the running VWAP;
    an absent bar gets NaN even though the running sum is defined."""
    minutes, offs = _grid([(556, 4)])
    close = np.array([[100.0], [102.0], [np.nan], [104.0]])
    high = close.copy()
    low = close.copy()
    volume = np.array([[10.0], [np.nan], [5.0], [30.0]])
    high[2] = 999.0  # irrelevant: close NaN means no bar at row 2
    got = session_vwap(high, low, close, volume, minutes, offs)
    assert got[0, 0] == pytest.approx(100.0)
    assert got[1, 0] == pytest.approx(100.0)  # present, contributes nothing
    assert np.isnan(got[2, 0])
    assert got[3, 0] == pytest.approx((100 * 10 + 104 * 30) / 40)


def test_session_vwap_zero_cum_volume_is_nan():
    minutes, offs = _grid([(556, 3)])
    close = np.array([[100.0], [101.0], [102.0]])
    volume = np.array([[0.0], [-3.0], [4.0]])
    got = session_vwap(close, close, close, volume, minutes, offs)
    assert np.isnan(got[0, 0]) and np.isnan(got[1, 0])
    assert got[2, 0] == pytest.approx(102.0)


def test_session_vwap_float32_upcast():
    high, low, close, volume, mod, offs = _rand_feature_panel(4, float32=True)
    got = session_vwap(high, low, close, volume, mod, offs)
    assert got.dtype == np.float64
    exp = _ref_session_vwap(*(a.astype(np.float64) for a in (high, low, close, volume)), mod, offs)
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize(
    "bad",
    ["volume_shape", "low_shape", "mod_len", "offs_start", "offs_end", "offs_flat", "offs_dec"],
)
def test_session_vwap_raises(bad):
    minutes, offs = _grid([(555, 5), (555, 5)])
    x = np.full((10, 2), 100.0)
    args = {"high": x, "low": x, "close": x, "volume": x.copy(), "mod": minutes, "offs": offs}
    if bad == "volume_shape":
        args["volume"] = np.full((10, 3), 1.0)
    elif bad == "low_shape":
        args["low"] = np.full((9, 2), 1.0)
    elif bad == "mod_len":
        args["mod"] = minutes[:-1]
    elif bad == "offs_start":
        args["offs"] = np.array([1, 5, 10])
    elif bad == "offs_end":
        args["offs"] = np.array([0, 5, 9])
    elif bad == "offs_flat":
        args["offs"] = np.array([0, 5, 5, 10])
    elif bad == "offs_dec":
        args["offs"] = np.array([0, 6, 4, 10])
    with pytest.raises(ValueError):
        session_vwap(
            args["high"], args["low"], args["close"], args["volume"], args["mod"], args["offs"]
        )


# ---------------------------------------------------------------------------------------
# 1.2 session_range_pct
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_session_range_pct_matches_reference_randomized(seed):
    high, low, close, _, mod, offs = _rand_feature_panel(seed, float32=seed % 4 == 1)
    got = session_range_pct(high, low, close, mod, offs)
    exp = _ref_session_range_pct(high, low, close, mod, offs)
    assert got.dtype == np.float64
    assert got.shape == (len(offs) - 1, close.shape[1])
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_session_range_pct_hand_values_and_0915_excluded():
    minutes, offs = _grid([(555, 4), (555, 3)])
    high = np.array([[500.0], [101.0], [103.0], [102.0], [10.0], [np.nan], [np.nan]])
    low = np.array([[1.0], [99.0], [100.0], [98.0], [9.0], [np.nan], [np.nan]])
    close = np.array([[200.0], [100.0], [101.0], [100.0], [9.5], [np.nan], [np.nan]])
    got = session_range_pct(high, low, close, minutes, offs)
    assert got[0, 0] == pytest.approx((103.0 - 98.0) / 100.0, rel=1e-15)
    assert np.isnan(got[1, 0])  # only the 09:15 row has data -> no rows


def test_session_range_pct_last_row_close_nonpositive_is_nan():
    minutes, offs = _grid([(556, 3)])
    high = np.array([[101.0], [102.0], [1.0]])
    low = np.array([[99.0], [98.0], [-1.0]])
    close = np.array([[100.0], [101.0], [0.0]])
    assert np.isnan(session_range_pct(high, low, close, minutes, offs)[0, 0])
    # the last *qualifying* row is used: a trailing row with NaN high is skipped
    high[2] = np.nan
    got = session_range_pct(high, low, close, minutes, offs)
    assert got[0, 0] == pytest.approx((102.0 - 98.0) / 101.0)


# ---------------------------------------------------------------------------------------
# 1.3 adr_pct
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_adr_pct_matches_reference_randomized(seed):
    rng = np.random.default_rng(500 + seed)
    rp = rng.uniform(0.005, 0.03, (int(rng.integers(1, 30)), int(rng.integers(1, 4))))
    rp[rng.random(rp.shape) < 0.1] = np.nan
    if seed % 2:
        rp = rp.astype(np.float32)
    for n in (1, 2, 5, 40):
        got = adr_pct(rp, n)
        assert got.dtype == np.float64 and got.shape == rp.shape
        np.testing.assert_allclose(
            got, _ref_adr_pct(rp, n), rtol=1e-10, atol=1e-14, equal_nan=True
        )


@pytest.mark.parametrize("n", [1, 3, 7])
def test_adr_pct_causal_perturbation(n):
    """Invariant 4: changing range_pct[j] can change only out[j+1 : j+n+1]."""
    rng = np.random.default_rng(n)
    nd = 25
    rp = rng.uniform(0.01, 0.02, (nd, 2))
    base = adr_pct(rp, n)
    for j in range(nd):
        rp2 = rp.copy()
        rp2[j, 0] *= 3.0
        out = adr_pct(rp2, n)
        changed = np.flatnonzero(~np.isclose(out[:, 0], base[:, 0], equal_nan=True, rtol=0,
                                             atol=0))
        allowed = set(range(j + 1, min(j + n + 1, nd)))
        assert set(changed.tolist()) <= allowed
        # and every allowed index with a full window actually does change (not stale)
        assert set(changed.tolist()) == {i for i in allowed if i >= n}
        np.testing.assert_array_equal(out[:, 1], base[:, 1])
        # NaN in session j poisons exactly the same rows
        rp3 = rp.copy()
        rp3[j, 0] = np.nan
        out3 = adr_pct(rp3, n)
        nan_rows = set(np.flatnonzero(np.isnan(out3[:, 0])).tolist())
        assert nan_rows == set(range(min(n, nd))) | allowed


def test_adr_pct_hand_values():
    rp = np.array([[0.01], [0.03], [0.02], [0.04]])
    got = adr_pct(rp, 2)
    assert np.isnan(got[0, 0]) and np.isnan(got[1, 0])
    assert got[2, 0] == pytest.approx(0.02)
    assert got[3, 0] == pytest.approx(0.025)


@pytest.mark.parametrize("n", [0, -1])
def test_adr_pct_raises_for_n_below_1(n):
    with pytest.raises(ValueError):
        adr_pct(np.ones((5, 2)), n)


# ---------------------------------------------------------------------------------------
# 1.4 max_excursion
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_max_excursion_matches_reference_randomized(seed):
    high, low, close, volume, mod, offs = _rand_feature_panel(seed, float32=seed % 3 == 2)
    rng = np.random.default_rng(900 + seed)
    vwap = _ref_session_vwap(high, low, close, volume, mod, offs)
    u = rng.random(vwap.shape)
    vwap[u < 0.05] = np.nan
    vwap[(u >= 0.05) & (u < 0.08)] = 0.0
    vwap[(u >= 0.08) & (u < 0.1)] = -2.0
    nd, m = len(offs) - 1, close.shape[1]
    adr = rng.uniform(0.002, 0.02, (nd, m))
    u = rng.random((nd, m))
    adr[u < 0.15] = np.nan
    adr[(u >= 0.15) & (u < 0.25)] = 0.0
    adr[(u >= 0.25) & (u < 0.3)] = -0.01
    got = max_excursion(close, vwap, adr, mod, offs)
    assert got.dtype == np.float64 and got.shape == (nd, m)
    np.testing.assert_allclose(
        got, _ref_max_excursion(close, vwap, adr, mod, offs), rtol=1e-12, atol=1e-12,
        equal_nan=True,
    )
    got2 = max_excursion(close, vwap, adr, mod, offs, start_minute=560, end_minute=720)
    np.testing.assert_allclose(
        got2,
        _ref_max_excursion(close, vwap, adr, mod, offs, start=560, end=720),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_max_excursion_window_edges():
    """09:15 and >= 15:20 rows excluded; 09:16 and 15:19 included."""
    minutes = np.array([555, 556, 700, 919, 920, 921], dtype=np.int64)
    offs = np.array([0, 6])
    vwap = np.full((6, 1), 100.0)
    adr = np.array([[0.01]])
    close = np.full((6, 1), 100.0)
    close[0] = 150.0
    close[4] = 160.0
    close[5] = 40.0
    close[1] = 101.0  # 1.0
    close[3] = 97.0  # 3.0
    got = max_excursion(close, vwap, adr, minutes, offs)
    assert got[0, 0] == pytest.approx(3.0, rel=1e-12)
    close[3] = 100.0
    assert max_excursion(close, vwap, adr, minutes, offs)[0, 0] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------------------
# 2.1 FadeParams
# ---------------------------------------------------------------------------------------


def test_fade_params_defaults_and_frozen():
    p = FadeParams(k=1.0, notional=1e5)
    assert p.start_minute == 556 and p.square_off_minute == 920
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.k = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"k": 0.0, "notional": 1e5},
        {"k": -1.0, "notional": 1e5},
        {"k": 1.0, "notional": 0.0},
        {"k": 1.0, "notional": -5.0},
        {"k": 1.0, "notional": 1e5, "start_minute": 920, "square_off_minute": 920},
        {"k": 1.0, "notional": 1e5, "start_minute": 921, "square_off_minute": 920},
    ],
)
def test_fade_params_invalid(kw):
    with pytest.raises(ValueError):
        FadeParams(**kw)


# ---------------------------------------------------------------------------------------
# 2.2 simulate_fade: hand-built edge cases
#   _hand(): price 100, vwap 100, adr 1%, k=1 -> bands 99 / 101.
# ---------------------------------------------------------------------------------------

S = [(900, 26)]  # rows 0..25 <-> minutes 900..925; row 20 is 15:20


def test_basic_short_fill_at_next_open_and_vwap_exit():
    """Invariant 5: fill at open[r+1], not close[r] nor the band."""
    p = _hand(S)
    c, o = p["close"], p["open_"]
    c[2] = 102.0  # > 101 -> short signal
    o[3] = 101.5
    c[3] = c[4] = 101.2  # stays above vwap
    c[5] = 99.9  # <= vwap -> pending
    o[6] = 100.3
    df = _check(
        p,
        [
            dict(side=-1, signal_row=2, entry_row=3, exit_row=6, exit_reason="vwap",
                 entry_price=101.5, exit_price=100.3, symbol="S0", day_idx=0, holding_bars=3),
        ],
    )
    qty = 100_000.0 / 101.5
    assert df.loc[0, "qty"] == pytest.approx(qty, rel=1e-15)
    assert df.loc[0, "gross_pnl"] == pytest.approx(-qty * (100.3 - 101.5), rel=1e-12)
    assert df.loc[0, "entry_price"] not in (102.0, 101.0)


def test_basic_long():
    p = _hand(S)
    c, o = p["close"], p["open_"]
    c[4] = 98.0
    o[5] = 98.4
    c[5] = 98.9
    c[6] = 100.0  # >= vwap -> pending (equality counts)
    o[7] = 99.7
    _check(p, [dict(side=1, signal_row=4, entry_row=5, exit_row=7, exit_reason="vwap",
                    entry_price=98.4, exit_price=99.7)])


def test_close_on_band_is_not_a_signal():
    p = _hand(S)
    p["close"][3] = 100.0 * (1 + 1.0 * 0.01)  # exactly on the upper band: strict '>'
    assert len(_sim(p)) == 0


def test_signal_on_last_pre_square_off_bar_dropped():
    p = _hand(S)
    p["close"][19] = 102.0  # minute 919; r+1 is 15:20 -> no entry
    assert len(_sim(p)) == 0
    assert len(_ref(p)) == 0


def test_no_signal_at_or_after_square_off():
    p = _hand(S)
    p["close"][20] = 102.0
    p["close"][21] = 97.0
    assert len(_sim(p)) == 0


def test_signal_on_0915_row_excluded():
    p = _hand([(555, 10)])
    p["close"][0] = 102.0  # 09:15 -- excluded even though vwap is supplied finite
    assert len(_sim(p)) == 0
    p["close"][1] = 102.0  # 09:16 -- first eligible row
    _check(p, [dict(side=-1, signal_row=1, entry_row=2, exit_row=3, exit_reason="vwap")])


def test_signal_whose_next_row_is_next_session_dropped():
    p = _hand([(900, 11), (900, 26)])
    p["close"][10] = 102.0  # last row of session 0
    assert len(_sim(p)) == 0
    assert len(_ref(p)) == 0


def test_next_bar_present_but_not_tradable_drops_signal_not_deferred():
    """Invariant 6 (entry side)."""
    p = _hand(S)
    p["close"][2] = 102.0
    p["tradable"][3] = False  # present, fillable, but not tradable
    assert len(_sim(p)) == 0
    assert len(_ref(p)) == 0


def test_next_bar_absent_drops_signal():
    p = _hand(S)
    p["close"][2] = 102.0
    for a in ("open_", "close", "volume"):
        p[a][3] = np.nan
    assert len(_sim(p)) == 0


def test_zero_volume_next_bar_drops_signal():
    p = _hand(S)
    p["close"][2] = 102.0
    p["volume"][3] = 0.0
    assert len(_sim(p)) == 0


def test_nonpositive_or_nan_open_next_bar_drops_signal():
    for val in (0.0, np.nan, -1.0):
        p = _hand(S)
        p["close"][2] = 102.0
        p["open_"][3] = val
        assert len(_sim(p)) == 0, val


def test_signal_row_itself_need_not_be_tradable():
    """The spec's entry conditions do not include tradable[r]; only tradable[r+1]."""
    p = _hand(S)
    p["close"][2] = 102.0
    p["tradable"][2] = False
    _check(p, [dict(side=-1, signal_row=2, entry_row=3, exit_row=4, exit_reason="vwap")])


@pytest.mark.parametrize("bad", ["adr_nan", "adr_zero", "vwap_nan", "vwap_zero"])
def test_no_signal_without_valid_vwap_or_adr(bad):
    p = _hand(S)
    p["close"][2] = 50.0 if bad == "vwap_zero" else 102.0
    if bad == "adr_nan":
        p["adr_day"][0] = np.nan
    elif bad == "adr_zero":
        p["adr_day"][0] = 0.0
    elif bad == "vwap_nan":
        p["vwap"][2] = np.nan
    else:
        p["vwap"][2] = 0.0
    assert len(_sim(p)) == 0


def test_entry_bar_close_can_trigger_exit():
    p = _hand(S)
    p["close"][2] = 102.0
    p["close"][3] = 99.5  # entry bar's own close <= vwap
    p["open_"][4] = 99.8
    _check(p, [dict(entry_row=3, exit_row=4, exit_reason="vwap", exit_price=99.8,
                    holding_bars=1)])


def test_exit_and_immediate_reentry_on_same_row():
    p = _hand(S)
    c, o = p["close"], p["open_"]
    c[2] = 102.0
    o[3] = 101.4
    c[3] = 99.5  # pending exit
    o[4] = 100.2  # exit fills here
    c[4] = 98.5  # and row 4's close signals a long
    o[5] = 98.7
    c[5] = 100.0  # long exit pending (>=)
    o[6] = 99.9
    _check(
        p,
        [
            dict(side=-1, signal_row=2, entry_row=3, exit_row=4, entry_price=101.4,
                 exit_price=100.2, exit_reason="vwap"),
            dict(side=1, signal_row=4, entry_row=5, exit_row=6, entry_price=98.7,
                 exit_price=99.9, exit_reason="vwap"),
        ],
    )


def test_no_signal_while_in_position():
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:9] = 103.0  # would be fresh short signals if flat
    c[9] = 100.0
    _check(p, [dict(signal_row=2, entry_row=3, exit_row=10, exit_reason="vwap")])


def test_pending_skips_unfillable_rows_and_needs_no_tradable():
    """Invariant 6 (exit side): exits fill even when not tradable."""
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3] = 101.5
    c[4] = 99.0  # pending
    p["volume"][5] = 0.0  # not fillable
    p["open_"][6] = np.nan  # not fillable
    p["tradable"][7] = False  # fillable, not tradable -> exit here
    p["open_"][7] = 100.7
    _check(p, [dict(entry_row=3, exit_row=7, exit_reason="vwap", exit_price=100.7)])


def test_pending_vwap_exit_landing_on_square_off_bar_keeps_vwap_reason():
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:19] = 101.5
    c[19] = 99.9  # minute 919 -> pending
    p["open_"][20] = 100.1
    _check(p, [dict(entry_row=3, exit_row=20, exit_reason="vwap", exit_price=100.1)])


def test_square_off_at_1520_open():
    """Invariant 7."""
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:] = 101.5  # never reverts
    p["open_"][20] = 101.8
    p["tradable"][20] = False  # exits do not need tradable
    _check(p, [dict(entry_row=3, exit_row=20, exit_reason="square_off", exit_price=101.8,
                    holding_bars=17)])


def test_square_off_checked_before_vwap_rule():
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:20] = 101.5
    c[20] = 99.0  # would be a vwap trigger, but rule 1 fires first on this row
    p["open_"][20] = 101.6
    _check(p, [dict(exit_row=20, exit_reason="square_off", exit_price=101.6)])


def test_square_off_bar_missing_exits_at_next_fillable():
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:] = 101.5
    for a in ("open_", "close", "volume"):
        p[a][20] = np.nan
    p["open_"][21] = 101.9
    _check(p, [dict(exit_row=21, exit_reason="square_off", exit_price=101.9)])


def test_unfillable_square_off_bar_with_reverting_close_becomes_vwap_exit():
    """Row 20: rule 1 fails (zero volume), so rule 2 is evaluated and sets a pending exit;
    the pending exit takes precedence on row 21 -> reason 'vwap'."""
    p = _hand(S)
    c = p["close"]
    c[2] = 102.0
    c[3:20] = 101.5
    c[20] = 99.5
    p["volume"][20] = 0.0
    c[21:] = 101.5
    p["open_"][21] = 100.4
    _check(p, [dict(exit_row=21, exit_reason="vwap", exit_price=100.4)])


def test_no_bar_exit_short_session_without_square_off():
    """60-bar session never reaches 15:20: exit at last qualifying close, reason no_bar."""
    p = _hand([(555, 60)])
    c = p["close"]
    c[5] = 102.0
    c[6:] = 101.5
    c[57] = 101.7
    for a in ("open_", "close", "volume"):
        p[a][59] = np.nan
    p["volume"][58] = 0.0
    _check(p, [dict(entry_row=6, exit_row=57, exit_reason="no_bar", exit_price=101.7,
                    holding_bars=51)])


def test_no_bar_when_pending_has_no_later_fillable_row():
    p = _hand([(555, 60)])
    c = p["close"]
    c[5] = 102.0
    c[6:] = 101.5
    c[58] = 99.6  # pending at the second-to-last row
    for a in ("open_", "close", "volume"):
        p[a][59] = np.nan
    _check(p, [dict(entry_row=6, exit_row=58, exit_reason="no_bar", exit_price=99.6)])


def test_no_bar_exit_at_entry_row_when_nothing_after_qualifies():
    p = _hand([(555, 12)])
    c = p["close"]
    c[9] = 102.0
    c[10] = 101.3
    p["volume"][11] = 0.0
    c[11] = 101.4
    _check(p, [dict(entry_row=10, exit_row=10, exit_reason="no_bar", exit_price=101.3,
                    holding_bars=0)])


def test_evening_session_all_after_square_off_has_no_trades():
    p = _hand([(1095, 60)])
    p["close"][5] = 102.0
    assert len(_sim(p)) == 0


def test_position_never_spans_sessions():
    p = _hand([(555, 60), (900, 26)])
    c = p["close"]
    c[5] = 102.0
    c[6:60] = 101.5  # never reverts in session 0
    c[60:] = 99.0  # session 1 opens below vwap -> would be exits for a carried short
    df = _sim(p)
    assert (df["exit_row"] < 60).sum() >= 1
    first = df.iloc[0]
    assert first["exit_row"] <= 59 and first["exit_reason"] == "no_bar"
    _assert_trades_equal(df, _ref(p))


def test_symbols_independent_and_sorted_by_entry_row_then_symbol():
    p = _hand(S, n_sym=3)
    p["symbols"] = ["ZED", "ALPHA", "MID"]
    c = p["close"]
    c[2, 0] = 102.0  # ZED entry row 3
    c[2, 1] = 97.0  # ALPHA entry row 3
    c[1, 2] = 102.0  # MID entry row 2
    df = _sim(p)
    assert list(df["symbol"]) == ["MID", "ALPHA", "ZED"]
    assert list(df["entry_row"]) == [2, 3, 3]
    _assert_trades_equal(df, _ref(p))


def test_empty_result_has_exact_columns():
    p = _hand(S)
    df = _sim(p)
    assert len(df) == 0
    assert list(df.columns) == COLS


def test_output_dtypes():
    p = _hand(S)
    p["close"][2] = 102.0
    df = _sim(p)
    assert len(df) == 1
    for col in INT_COLS:
        assert pd.api.types.is_integer_dtype(df[col]), col
    for col in FLOAT_COLS:
        assert df[col].dtype == np.float64, col
    assert isinstance(df.loc[0, "symbol"], str)
    assert isinstance(df.loc[0, "exit_reason"], str)


def test_inputs_not_mutated():
    p, cfg = _rand_sim_panel(3)
    snap = {key: np.array(v, copy=True) for key, v in p.items() if key != "symbols"}
    _sim(p, **cfg)
    for key, v in snap.items():
        np.testing.assert_array_equal(np.asarray(p[key]), v, err_msg=key)


@pytest.mark.parametrize(
    "bad", ["close_shape", "tradable_shape", "adr_shape", "symbols_len", "mod_len"]
)
def test_simulate_raises_on_shape_mismatch(bad):
    p = _hand(S, n_sym=2)
    if bad == "close_shape":
        p["close"] = p["close"][:-1]
    elif bad == "tradable_shape":
        p["tradable"] = p["tradable"][:, :1]
    elif bad == "adr_shape":
        p["adr_day"] = np.full((2, 2), 0.01)
    elif bad == "symbols_len":
        p["symbols"] = ["S0"]
    elif bad == "mod_len":
        p["minute_of_day"] = p["minute_of_day"][:-1]
    with pytest.raises(ValueError):
        _sim(p)


# ---------------------------------------------------------------------------------------
# 2.2 simulate_fade: randomized comparison with the loop reference
# ---------------------------------------------------------------------------------------


_N_RAND = 60


@pytest.mark.parametrize("seed", range(_N_RAND))
def test_simulate_matches_reference_randomized(seed):
    p, cfg = _rand_sim_panel(seed, float32=seed % 4 == 3)
    got = _sim(p, **cfg)
    exp = _ref(p, **cfg)
    _assert_trades_equal(got, exp)
    if len(got):
        for col in FLOAT_COLS:
            assert got[col].dtype == np.float64, col


def test_randomized_panels_exercise_every_branch():
    """Guard against the randomized comparison being vacuous."""
    reasons = []
    reentries = 0
    for seed in range(_N_RAND):
        p, cfg = _rand_sim_panel(seed)
        df = _ref(p, **cfg)
        reasons += list(df["exit_reason"])
        for _, g in df.groupby("symbol"):
            sig, ext = g["signal_row"].to_numpy(), g["exit_row"].to_numpy()
            reentries += int(np.sum(sig[1:] == ext[:-1]))
    reasons = pd.Series(reasons)
    assert len(reasons) >= 200
    for r in ("vwap", "square_off", "no_bar"):
        assert (reasons == r).sum() >= 5, r
    assert reentries >= 5


@pytest.mark.parametrize("seed", range(0, _N_RAND, 3))
def test_simulate_structural_invariants(seed):
    """Invariants 3, 5, 7, 10 checked directly on module output."""
    p, cfg = _rand_sim_panel(seed)
    df = _sim(p, **cfg)
    mod = p["minute_of_day"]
    offs = p["day_offsets"]
    sym_idx = {s: i for i, s in enumerate(p["symbols"])}
    o = p["open_"].astype(np.float64)
    c = p["close"].astype(np.float64)
    v = p["volume"].astype(np.float64)
    order = np.lexsort((df["symbol"].to_numpy(), df["entry_row"].to_numpy()))
    np.testing.assert_array_equal(order, np.arange(len(df)))
    for _, t in df.iterrows():
        s = sym_idx[t["symbol"]]
        d = int(t["day_idx"])
        a, b = offs[d], offs[d + 1]
        assert a <= t["signal_row"] < t["entry_row"] <= t["exit_row"] < b  # same session
        assert t["entry_row"] == t["signal_row"] + 1
        assert cfg["start"] <= mod[t["signal_row"]] < cfg["square_off"]
        assert mod[t["entry_row"]] < cfg["square_off"]
        assert p["tradable"][t["entry_row"], s]
        assert t["entry_price"] == o[t["entry_row"], s]  # open of r+1, exactly
        assert v[t["entry_row"], s] > 0
        if t["exit_reason"] == "no_bar":
            assert t["exit_price"] == c[t["exit_row"], s]
        else:
            assert t["exit_price"] == o[t["exit_row"], s]
            assert v[t["exit_row"], s] > 0
        if t["exit_reason"] == "square_off":
            assert mod[t["exit_row"]] >= cfg["square_off"]
        assert np.isfinite(t["entry_price"]) and np.isfinite(t["exit_price"])
        assert t["holding_bars"] == t["exit_row"] - t["entry_row"]
    for _, g in df.groupby("symbol"):
        g = g.sort_values("entry_row")
        # at most one open position: next entry strictly after previous exit row
        assert np.all(g["entry_row"].to_numpy()[1:] > g["exit_row"].to_numpy()[:-1])
        assert np.all(g["signal_row"].to_numpy()[1:] >= g["exit_row"].to_numpy()[:-1])


# ---------------------------------------------------------------------------------------
# Costs and slippage (invariants 8, 9)
# ---------------------------------------------------------------------------------------


def _manual_nse_charges(notional, is_buy):
    """NSE intraday charges written out by hand (independent of the cost model code)."""
    brokerage = min(20.0, 0.0003 * notional)
    exch = 0.0000297 * notional
    sebi = 0.000001 * notional
    stt = 0.0 if is_buy else 0.00025 * notional
    stamp = 0.00003 * notional if is_buy else 0.0
    ipft = 0.000001 * notional
    gst = 0.18 * (brokerage + exch + sebi)
    return brokerage + exch + sebi + stt + stamp + ipft + gst


@pytest.mark.parametrize("seed", range(0, 40, 2))
def test_zero_costs_net_equals_gross_exactly(seed):
    p, cfg = _rand_sim_panel(seed)
    df = _sim(p, **cfg)
    assert np.all(df["costs"].to_numpy() == 0.0)
    np.testing.assert_array_equal(df["net_pnl"].to_numpy(), df["gross_pnl"].to_numpy())
    np.testing.assert_array_equal(df["net_bps"].to_numpy(), df["gross_bps"].to_numpy())


@pytest.mark.parametrize("seed", range(1, 40, 2))
def test_nse_costs_reconcile_per_trade(seed):
    p, cfg = _rand_sim_panel(seed)
    base = _sim(p, **cfg)
    df = _sim(p, cost_model=NSEIntradayEquityCosts(), **cfg)
    # costs never change which trades happen or their gross P&L
    for col in ["entry_row", "exit_row", "side", "signal_row"]:
        np.testing.assert_array_equal(df[col].to_numpy(), base[col].to_numpy())
    np.testing.assert_array_equal(df["gross_pnl"].to_numpy(), base["gross_pnl"].to_numpy())
    for _, t in df.iterrows():
        side = int(t["side"])
        en = t["qty"] * t["entry_price"]
        xn = t["qty"] * t["exit_price"]
        exp = _manual_nse_charges(en, side == 1) + _manual_nse_charges(xn, side == -1)
        assert t["costs"] == pytest.approx(exp, rel=1e-12)
        assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["costs"], rel=1e-12, abs=1e-9)
        assert t["net_bps"] == pytest.approx(t["net_pnl"] / 100_000.0 * 1e4, rel=1e-12,
                                             abs=1e-12)
        assert en == pytest.approx(100_000.0, rel=1e-12)


def test_nse_costs_side_asymmetry_hand_case():
    """Short: entry is a SELL (STT), exit is a BUY (stamp). Swapped flags would differ."""
    p = _hand(S)
    p["close"][2] = 102.0
    p["open_"][3] = 101.0
    p["open_"][4] = 90.0  # exit notional very different from entry notional
    df = _sim(p, cost_model=NSEIntradayEquityCosts())
    t = df.iloc[0]
    qty = 100_000.0 / 101.0
    exp = _manual_nse_charges(qty * 101.0, False) + _manual_nse_charges(qty * 90.0, True)
    wrong = _manual_nse_charges(qty * 101.0, True) + _manual_nse_charges(qty * 90.0, False)
    assert abs(exp - wrong) > 1.0
    assert t["costs"] == pytest.approx(exp, rel=1e-12)


@pytest.mark.parametrize("seed", range(0, 30, 3))
def test_slippage_and_costs_reconcile_per_trade(seed):
    p, cfg = _rand_sim_panel(seed)
    df = _sim(p, cost_model=NSEIntradayEquityCosts(), slippage=SqrtImpactSlippage(), **cfg)
    sym_idx = {s: i for i, s in enumerate(p["symbols"])}
    c = p["close"].astype(np.float64)
    v = p["volume"].astype(np.float64)
    for _, t in df.iterrows():
        s = sym_idx[t["symbol"]]
        side = int(t["side"])
        en = t["qty"] * t["entry_price"]
        xn = t["qty"] * t["exit_price"]
        charges = _manual_nse_charges(en, side == 1) + _manual_nse_charges(xn, side == -1)
        slip = 0.0
        for fn, fr in ((en, int(t["entry_row"])), (xn, int(t["exit_row"]))):
            btv = c[fr, s] * v[fr, s]  # close * volume of the fill row
            slip += fn * (1.5 + 10.0 * math.sqrt(fn / btv)) / 1e4
        assert t["costs"] == pytest.approx(charges + slip, rel=1e-12)
        assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["costs"], rel=1e-12, abs=1e-9)
    _assert_trades_equal(
        df,
        _ref(p, cost_model=NSEIntradayEquityCosts(), slippage=SqrtImpactSlippage(), **cfg),
    )


def test_slippage_uses_close_times_volume_not_open():
    p = _hand(S)
    p["close"][2] = 102.0
    p["open_"][3] = 100.5
    p["close"][3] = 99.0  # exit pending; entry-row btv = 99 * 1000
    p["volume"][3] = 1000.0
    p["open_"][4] = 100.0
    p["close"][4] = 50.0  # exit-row btv = 50 * 2000
    p["volume"][4] = 2000.0
    df = _sim(p, slippage=SqrtImpactSlippage())
    t = df.iloc[0]
    qty = 100_000.0 / 100.5
    en, xn = qty * 100.5, qty * 100.0
    exp = en * (1.5 + 10 * math.sqrt(en / (99.0 * 1000))) / 1e4 + xn * (
        1.5 + 10 * math.sqrt(xn / (50.0 * 2000))
    ) / 1e4
    assert t["costs"] == pytest.approx(exp, rel=1e-12)


def test_no_bar_slippage_uses_exit_row_close():
    p = _hand([(555, 12)])
    c = p["close"]
    c[5] = 102.0
    c[6:] = 101.5
    c[11] = 101.9
    p["volume"][11] = 300.0
    df = _sim(p, slippage=SqrtImpactSlippage())
    t = df.iloc[0]
    assert t["exit_reason"] == "no_bar" and t["exit_price"] == 101.9
    en = t["qty"] * 100.0
    xn = t["qty"] * 101.9
    exp = en * (1.5 + 10 * math.sqrt(en / (101.5 * 1000))) / 1e4 + xn * (
        1.5 + 10 * math.sqrt(xn / (101.9 * 300))
    ) / 1e4
    assert t["costs"] == pytest.approx(exp, rel=1e-12)


# ---------------------------------------------------------------------------------------
# 3. report
# ---------------------------------------------------------------------------------------


def _hand_trades():
    rows = [
        # sym, day, side, entry, exit, gross, costs, net, gbps, nbps, hold, reason
        ("X", 0, 1, 2, 5, 100.0, 10.0, 90.0, 10.0, 9.0, 3, "vwap"),
        ("W", 0, -1, 4, 6, 30.0, 5.0, 25.0, 3.0, 2.5, 2, "no_bar"),
        ("Y", 0, -1, 4, 8, -50.0, 10.0, -60.0, -5.0, -6.0, 4, "square_off"),
        ("Z", 0, 1, 5, 6, 10.0, 10.0, 0.0, 1.0, 0.0, 1, "vwap"),
        ("X", 2, 1, 20, 20, -200.0, 10.0, -210.0, -20.0, -21.0, 0, "no_bar"),
    ]
    recs = []
    for sym, day, side, e, x, g, cst, net, gb, nb, hold, reason in rows:
        recs.append(
            {
                "symbol": sym,
                "day_idx": day,
                "side": side,
                "signal_row": e - 1,
                "entry_row": e,
                "entry_price": 100.0,
                "exit_row": x,
                "exit_price": 100.0,
                "exit_reason": reason,
                "qty": 1000.0,
                "gross_pnl": g,
                "costs": cst,
                "net_pnl": net,
                "gross_bps": gb,
                "net_bps": nb,
                "holding_bars": hold,
            }
        )
    return pd.DataFrame(recs, columns=COLS)


def test_daily_pnl_hand():
    tr = _hand_trades()
    net = daily_pnl(tr, 4)
    assert net.dtype == np.float64 and net.shape == (4,)
    np.testing.assert_allclose(net, [55.0, 0.0, -210.0, 0.0])
    gross = daily_pnl(tr, 4, column="gross_pnl")
    np.testing.assert_allclose(gross, [90.0, 0.0, -200.0, 0.0])
    np.testing.assert_array_equal(daily_pnl(tr.iloc[:0], 3), np.zeros(3))


def test_summarize_hand():
    tr = _hand_trades()
    out = summarize(tr, 4)
    assert set(out) == SUMMARY_KEYS
    assert out["trades"] == 5
    assert out["trades_per_day"] == pytest.approx(5 / 4)
    assert out["win_rate"] == pytest.approx(2 / 5)  # net == 0 is not a win
    total = out["pct_exit_vwap"] + out["pct_exit_square_off"] + out["pct_exit_no_bar"]
    # 'pct' may be a fraction or a percentage; the spec does not pin the scale.
    assert total == pytest.approx(1.0) or total == pytest.approx(100.0)
    assert out["pct_exit_vwap"] / total == pytest.approx(2 / 5)
    assert out["pct_exit_square_off"] / total == pytest.approx(1 / 5)
    assert out["pct_exit_no_bar"] / total == pytest.approx(2 / 5)
    assert out["mean_gross_bps"] == pytest.approx(np.mean([10, 3, -5, 1, -20]))
    assert out["mean_net_bps"] == pytest.approx(np.mean([9, 2.5, -6, 0, -21]))
    assert out["median_holding_bars"] == pytest.approx(2.0)
    assert out["total_gross_pnl"] == pytest.approx(-110.0)
    assert out["total_net_pnl"] == pytest.approx(-155.0)
    assert out["total_costs"] == pytest.approx(45.0)
    assert out["sharpe_net"] == pytest.approx(
        sharpe_ratio(np.array([55.0, 0.0, -210.0, 0.0]), periods_per_year=252), rel=1e-12
    )
    assert out["sharpe_gross"] == pytest.approx(
        sharpe_ratio(np.array([90.0, 0.0, -200.0, 0.0]), periods_per_year=252), rel=1e-12
    )
    # Occupancy [entry, exit): row 4 -> X,W,Y ; row 5 -> W,Y,Z (X left at 5). Zero-length
    # trade occupies nothing. A closed-interval implementation would report 4.
    assert out["max_concurrent"] == 3
    assert out["worst_trade_net_bps"] == pytest.approx(-21.0)
    assert out["worst_day_net_pnl"] == pytest.approx(-210.0)
    assert out["long_trades"] == 3
    assert out["short_trades"] == 2


def test_summarize_empty():
    p = _hand(S)
    empty = _sim(p)
    for tr in (empty, _hand_trades().iloc[:0]):
        out = summarize(tr, 5)
        assert set(out) == SUMMARY_KEYS
        assert out["trades"] == 0
        assert out["trades_per_day"] == 0
        for key in ("win_rate", "pct_exit_vwap", "pct_exit_square_off", "pct_exit_no_bar",
                    "mean_gross_bps", "mean_net_bps"):
            assert np.isnan(out[key]), key
        assert out["long_trades"] == 0 and out["short_trades"] == 0


def test_summarize_consistent_with_simulation_output():
    p, cfg = _rand_sim_panel(5)
    df = _sim(p, cost_model=NSEIntradayEquityCosts(), **cfg)
    n_days = len(p["day_offsets"]) - 1
    out = summarize(df, n_days)
    assert out["trades"] == len(df)
    assert out["total_costs"] == pytest.approx(df["costs"].sum(), rel=1e-12)
    assert out["total_net_pnl"] == pytest.approx(daily_pnl(df, n_days).sum(), rel=1e-9,
                                                 abs=1e-9)
    assert out["long_trades"] + out["short_trades"] == len(df)


# ---------------------------------------------------------------------------------------
# Invariant 11: null behaviour on driftless random walks (rate over >= 30 seeds)
# ---------------------------------------------------------------------------------------


def _random_walk_panel(seed, n_sym=6, n_days=14):
    """Driftless additive Gaussian random walk; open = previous close + independent noise,
    so every fill price is a martingale step away from the decision close."""
    rng = np.random.default_rng(seed)
    minutes, offs = _grid([(555, 375)] * n_days)
    n = len(minutes)
    close = np.empty((n, n_sym))
    open_ = np.empty((n, n_sym))
    for d in range(n_days):
        a, b = offs[d], offs[d + 1]
        level = 100.0 + rng.normal(0, 2.0, n_sym)
        steps = rng.normal(0, 0.05, (b - a, n_sym))
        c = level + np.cumsum(steps, axis=0)
        o = np.vstack([level + rng.normal(0, 0.01, n_sym), c[:-1]])
        o = o + rng.normal(0, 0.01, o.shape)
        close[a:b] = c
        open_[a:b] = o
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.02, close.shape))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.02, close.shape))
    volume = rng.integers(100, 5000, close.shape).astype(np.float64)
    return open_, high, low, close, volume, minutes, offs


def test_null_random_walk_rate_of_significant_mean_gross_bps():
    n_seeds = 30
    t_crit = 2.58  # from the spec (invariant 11)
    max_rate = 0.2  # from the spec (invariant 11)
    exceed = 0
    for seed in range(n_seeds):
        open_, high, low, close, volume, mod, offs = _random_walk_panel(seed)
        vwap = session_vwap(high, low, close, volume, mod, offs)
        adr = adr_pct(session_range_pct(high, low, close, mod, offs), 5)
        # k is not hand-chosen (CLAUDE.md rule 8): it is the median of the measured
        # max-excursion distribution on this very panel, so about half of the eligible
        # symbol-days reach the band.
        k = float(np.nanmedian(max_excursion(close, vwap, adr, mod, offs)))
        assert np.isfinite(k) and k > 0
        tradable = np.ones(close.shape, dtype=bool)
        df = simulate_fade(
            open_, close, volume, vwap, adr, tradable, mod, offs,
            [f"R{i}" for i in range(close.shape[1])], FadeParams(k=k, notional=100_000.0),
        )
        assert np.all(df["costs"].to_numpy() == 0.0)
        g = df["gross_bps"].to_numpy(dtype=np.float64)
        assert len(g) >= 20, f"seed {seed}: only {len(g)} trades; null test lacks power"
        t = g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))
        if abs(t) > t_crit:
            exceed += 1
    assert exceed / n_seeds <= max_rate, f"{exceed}/{n_seeds} seeds with |t| > {t_crit}"
