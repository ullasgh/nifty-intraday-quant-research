"""Independent test suite "A" for ``nifty_quant.research.vwap_adr`` (spec: specs/vwap_adr_fade.md).

Written from the spec alone, before any implementation exists (CLAUDE.md rule 1).

SPEC AMBIGUITIES AND HOW THIS SUITE READS THEM:
* ``pct_exit_*`` in ``summarize``: read as FRACTIONS in [0, 1] (consistent with
  ``win_rate`` being a "fraction"), not percentages in [0, 100].
* ``summarize`` on zero trades: only the quantities the spec names as NaN (win_rate,
  means, percentages) are pinned; totals / worst-* / median are left unpinned.
* ``worst_day_net_pnl``: the hand example is chosen so that it does not matter whether
  zero-trade days are included.
* ``max_concurrent``: counted across ALL symbols, over half-open ``[entry_row, exit_row)``.
* Rule 3 "row e always qualifies": a fillable row guarantees finite open/volume but not a
  finite close. The suite never builds a fillable row with a NaN close, so it does not
  pin that corner.
* A pending VWAP exit that never finds a later fillable row ends as ``"no_bar"`` (rule 3
  "no fill found by 1 or 2").
* After a ``no_bar`` exit the session is over; no re-entry is expected.
* Rule 2 (pending exit) needs only a present bar and a finite vwap; ``vwap > 0`` is not
  required for exits (never exercised with vwap <= 0 after entry).
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
from nifty_quant.research.vwap_adr import features, report, simulate

COLUMNS = [
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

NOTIONAL = 100_000.0


# --------------------------------------------------------------------------------------
# independent reference helpers (written from the spec; NOT the code under test)
# --------------------------------------------------------------------------------------


def _ref_vwap(high, low, close, volume, minute, offs, start=556):
    high, low, close, volume = (np.asarray(a, dtype=np.float64) for a in (high, low, close, volume))
    n_rows, n_sym = close.shape
    out = np.full((n_rows, n_sym), np.nan)
    for s in range(n_sym):
        for d in range(len(offs) - 1):
            pv = 0.0
            v = 0.0
            for r in range(offs[d], offs[d + 1]):
                if minute[r] < start:
                    continue
                h, lo, c, vol = high[r, s], low[r, s], close[r, s], volume[r, s]
                if all(math.isfinite(x) for x in (h, lo, c, vol)) and vol >= 0:
                    pv += (h + lo + c) / 3.0 * vol
                    v += vol
                if math.isfinite(c) and v > 0:
                    out[r, s] = pv / v
    return out


def _fillable(open_, volume, x, s):
    o, v = open_[x, s], volume[x, s]
    return bool(np.isfinite(o) and o > 0 and np.isfinite(v) and v > 0)


def _ref_trades(open_, close, volume, vwap, adr, tradable, minute, offs, k, start=556, sq=920):
    """Direct transcription of spec 2.2 entry/exit rules; returns list of dicts."""
    out = []
    n_sym = close.shape[1]
    for s in range(n_sym):
        for d in range(len(offs) - 1):
            lo_r, hi_r = int(offs[d]), int(offs[d + 1])
            a = adr[d, s]
            r = lo_r
            while r < hi_r:
                side = 0
                c, v = close[r, s], vwap[r, s]
                if (
                    np.isfinite(c)
                    and np.isfinite(v)
                    and v > 0
                    and np.isfinite(a)
                    and a > 0
                    and start <= minute[r] < sq
                ):
                    if c > v * (1 + k * a):
                        side = -1
                    elif c < v * (1 - k * a):
                        side = 1
                if side == 0:
                    r += 1
                    continue
                e = r + 1
                if not (
                    e < hi_r
                    and minute[e] < sq
                    and _fillable(open_, volume, e, s)
                    and tradable[e, s]
                ):
                    r += 1
                    continue
                entry = open_[e, s]
                pending = False
                x = None
                reason = None
                for j in range(e, hi_r):
                    if pending:
                        if _fillable(open_, volume, j, s):
                            x, reason = j, "vwap"
                            break
                        continue
                    if minute[j] >= sq and _fillable(open_, volume, j, s):
                        x, reason = j, "square_off"
                        break
                    cj, vj = close[j, s], vwap[j, s]
                    if np.isfinite(cj) and np.isfinite(vj):
                        if (side == 1 and cj >= vj) or (side == -1 and cj <= vj):
                            pending = True
                if x is None:
                    for j in range(hi_r - 1, e - 1, -1):
                        cj, vol = close[j, s], volume[j, s]
                        if np.isfinite(cj) and cj > 0 and np.isfinite(vol) and vol > 0:
                            x = j
                            break
                    assert x is not None
                    reason = "no_bar"
                    exit_price = close[x, s]
                else:
                    exit_price = open_[x, s]
                out.append(
                    dict(
                        sym=s,
                        day_idx=d,
                        side=side,
                        signal_row=r,
                        entry_row=e,
                        entry_price=float(entry),
                        exit_row=x,
                        exit_price=float(exit_price),
                        exit_reason=reason,
                    )
                )
                if reason == "no_bar":
                    break
                r = x
    return out


def _base(n_rows, n_sym=1, minutes=None, offs=None):
    minute = np.arange(556, 556 + n_rows) if minutes is None else np.asarray(minutes)
    day_offsets = np.array([0, n_rows]) if offs is None else np.asarray(offs)
    n_days = len(day_offsets) - 1
    return dict(
        open_=np.full((n_rows, n_sym), 100.0),
        close=np.full((n_rows, n_sym), 100.0),
        volume=np.full((n_rows, n_sym), 1000.0),
        vwap=np.full((n_rows, n_sym), 100.0),
        adr_day=np.full((n_days, n_sym), 0.01),
        tradable=np.ones((n_rows, n_sym), dtype=bool),
        minute_of_day=minute.astype(np.int64),
        day_offsets=day_offsets.astype(np.int64),
        symbols=[f"S{i}" for i in range(n_sym)],
    )


def _run(inp, k=1.0, notional=NOTIONAL, cost_model=None, slippage=None, **pkw):
    params = simulate.FadeParams(k=k, notional=notional, **pkw)
    return simulate.simulate_fade(
        inp["open_"],
        inp["close"],
        inp["volume"],
        inp["vwap"],
        inp["adr_day"],
        inp["tradable"],
        inp["minute_of_day"],
        inp["day_offsets"],
        inp["symbols"],
        params,
        cost_model=cost_model,
        slippage=slippage,
    )


def _one(df):
    assert len(df) == 1, df
    return df.iloc[0]


# ======================================================================================
# 1.1 session_vwap
# ======================================================================================


def test_constants():
    assert features.SESSION_START_MINUTE == 556
    assert features.SQUARE_OFF_MINUTE == 920


def test_session_vwap_hand_values_and_0915_excluded():
    minute = np.array([555, 556, 557, 558])
    high = np.array([[1000.0], [11.0], [13.0], [8.0]])
    low = np.array([[1000.0], [9.0], [11.0], [8.0]])
    close = np.array([[1000.0], [10.0], [12.0], [8.0]])
    vol = np.array([[1e6], [100.0], [300.0], [100.0]])
    out = features.session_vwap(high, low, close, vol, minute, np.array([0, 4]))
    assert out.shape == (4, 1)
    assert out.dtype == np.float64
    assert np.isnan(out[0, 0])
    assert out[1, 0] == pytest.approx(10.0, rel=1e-14)
    assert out[2, 0] == pytest.approx(4600.0 / 400.0, rel=1e-14)
    assert out[3, 0] == pytest.approx(5400.0 / 500.0, rel=1e-14)


def test_session_vwap_resets_across_mixed_session_lengths():
    """Invariant 1: sessions of 60, 105 and 375 bars in one panel."""
    rng = np.random.default_rng(11)
    lengths = [60, 375, 105, 375, 60]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = np.concatenate([[0], np.cumsum(lengths)])
    n = offs[-1]
    close = 100 + rng.standard_normal((n, 2)).cumsum(axis=0) * 0.1
    high = close + rng.uniform(0, 0.5, (n, 2))
    low = close - rng.uniform(0, 0.5, (n, 2))
    vol = rng.integers(1, 1000, (n, 2)).astype(float)
    out = features.session_vwap(high, low, close, vol, minute, offs)
    exp = _ref_vwap(high, low, close, vol, minute, offs)
    np.testing.assert_allclose(out, exp, rtol=1e-12, equal_nan=True)
    # Reset: first contributing row of each session equals its own typical price.
    for d in range(len(lengths)):
        r = offs[d] + 1  # the 09:16 row
        tp = (high[r] + low[r] + close[r]) / 3.0
        np.testing.assert_allclose(out[r], tp, rtol=1e-12)
        assert np.all(np.isnan(out[offs[d]]))  # the 09:15 row


def test_session_vwap_0915_never_contributes_even_with_huge_values():
    """Invariant 2."""
    minute = np.array([555, 556, 557])
    base = np.array([[50.0], [10.0], [20.0]])
    vol = np.array([[1e9], [1.0], [1.0]])
    a = features.session_vwap(base, base, base, vol, minute, np.array([0, 3]))
    base2 = base.copy()
    base2[0] = 1e6
    b = features.session_vwap(base2, base2, base2, vol * 7, minute, np.array([0, 3]))
    np.testing.assert_array_equal(a[1:], b[1:])
    assert a[2, 0] == pytest.approx(15.0)


def test_session_vwap_nan_bar_equals_row_deleted():
    """Invariant 3: NaN bar -> NaN VWAP there; later rows equal VWAP with the row deleted."""
    rng = np.random.default_rng(3)
    lengths = [60, 105]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = np.array([0, 60, 165])
    n = 165
    close = 100 + rng.standard_normal((n, 1)).cumsum(axis=0) * 0.1
    high = close + 0.3
    low = close - 0.3
    vol = rng.integers(1, 1000, (n, 1)).astype(float)
    drop = 20
    for a in (high, low, close, vol):
        a[drop] = np.nan
    snapshot = [a.copy() for a in (high, low, close, vol)]
    out = features.session_vwap(high, low, close, vol, minute, offs)
    # inputs untouched (no fill of any kind)
    for a, b in zip((high, low, close, vol), snapshot, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.isnan(out[drop, 0])
    keep = np.arange(n) != drop
    out_del = features.session_vwap(
        high[keep], low[keep], close[keep], vol[keep], minute[keep], np.array([0, 59, 164])
    )
    np.testing.assert_allclose(out[keep], out_del, rtol=1e-13, equal_nan=True)


def test_session_vwap_present_but_noncontributing_rows():
    minute = np.array([556, 557, 558, 559, 560])
    close = np.array([[10.0], [20.0], [30.0], [40.0], [np.nan]])
    high = close.copy()
    low = close.copy()
    vol = np.array([[0.0], [1.0], [np.nan], [-5.0], [1.0]])
    high[4] = 99.0
    low[4] = 99.0
    out = features.session_vwap(high, low, close, vol, minute, np.array([0, 5]))
    assert np.isnan(out[0, 0])  # cum_v == 0 -> NaN
    assert out[1, 0] == pytest.approx(20.0)
    assert out[2, 0] == pytest.approx(20.0)  # present, NaN volume: does not contribute
    assert out[3, 0] == pytest.approx(20.0)  # negative volume: does not contribute
    assert np.isnan(out[4, 0])  # not present -> NaN even though sums are defined


def test_session_vwap_float32_upcast():
    rng = np.random.default_rng(5)
    minute = 555 + np.arange(30)
    close = (100 + rng.standard_normal((30, 2))).astype(np.float32)
    high = close + np.float32(0.25)
    low = close - np.float32(0.25)
    vol = rng.integers(1, 100, (30, 2)).astype(np.float32)
    out = features.session_vwap(high, low, close, vol, minute, np.array([0, 30]))
    assert out.dtype == np.float64
    exp = _ref_vwap(high, low, close, vol, minute, [0, 30])
    np.testing.assert_allclose(out, exp, rtol=1e-12, equal_nan=True)


def test_session_vwap_custom_start_minute():
    minute = np.array([556, 557, 558])
    x = np.array([[10.0], [20.0], [30.0]])
    v = np.ones((3, 1))
    out = features.session_vwap(x, x, x, v, minute, np.array([0, 3]), start_minute=557)
    assert np.isnan(out[0, 0])
    assert out[1, 0] == pytest.approx(20.0)
    assert out[2, 0] == pytest.approx(25.0)


@pytest.mark.parametrize(
    "case",
    ["close_shape", "volume_nsym", "minute_len", "offs_start", "offs_end", "offs_nonincr"],
)
def test_session_vwap_value_errors(case):
    n = 6
    x = np.full((n, 2), 10.0)
    h, lo, c, v = x.copy(), x.copy(), x.copy(), x.copy()
    minute = 556 + np.arange(n)
    offs = np.array([0, 3, 6])
    if case == "close_shape":
        c = np.full((n - 1, 2), 10.0)
    elif case == "volume_nsym":
        v = np.full((n, 3), 10.0)
    elif case == "minute_len":
        minute = 556 + np.arange(n + 1)
    elif case == "offs_start":
        offs = np.array([1, 3, 6])
    elif case == "offs_end":
        offs = np.array([0, 3, 5])
    elif case == "offs_nonincr":
        offs = np.array([0, 3, 3, 6])
    with pytest.raises(ValueError):
        features.session_vwap(h, lo, c, v, minute, offs)


# ======================================================================================
# 1.2 session_range_pct
# ======================================================================================


def test_session_range_pct_hand_values():
    # session 0: rows 0..3 (09:15 row has an absurd high that must be ignored)
    # session 1: rows 4..5, last row is a NaN bar -> last close is row 4's
    # session 2: only a 09:15 row -> NaN
    # session 3: last close <= 0 -> NaN
    minute = np.array([555, 556, 557, 558, 556, 557, 555, 556, 557])
    offs = np.array([0, 4, 6, 7, 9])
    high = np.array([500.0, 102.0, 104.0, 101.0, 55.0, np.nan, 10.0, 5.0, 6.0])[:, None]
    low = np.array([1.0, 99.0, 98.0, 100.0, 45.0, np.nan, 9.0, 4.0, 3.0])[:, None]
    close = np.array([400.0, 100.0, 103.0, 100.0, 50.0, np.nan, 9.5, 4.5, 0.0])[:, None]
    out = features.session_range_pct(high, low, close, minute, offs)
    assert out.shape == (4, 1)
    assert out.dtype == np.float64
    assert out[0, 0] == pytest.approx((104.0 - 98.0) / 100.0, rel=1e-14)
    assert out[1, 0] == pytest.approx((55.0 - 45.0) / 50.0, rel=1e-14)
    assert np.isnan(out[2, 0])
    assert np.isnan(out[3, 0])


def test_session_range_pct_row_needs_all_three_finite():
    minute = np.array([556, 557, 558])
    high = np.array([[110.0], [200.0], [105.0]])
    low = np.array([[95.0], [1.0], [99.0]])
    close = np.array([[100.0], [np.nan], [102.0]])  # row 1 excluded entirely
    out = features.session_range_pct(high, low, close, minute, np.array([0, 3]))
    assert out[0, 0] == pytest.approx((110.0 - 95.0) / 102.0, rel=1e-14)


def test_session_range_pct_mixed_lengths_and_0915():
    """Invariants 1/2 for range: per-session, 09:15 excluded."""
    rng = np.random.default_rng(8)
    lengths = [375, 60, 105]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = np.concatenate([[0], np.cumsum(lengths)])
    n = offs[-1]
    close = 100 + rng.standard_normal((n, 2)).cumsum(axis=0) * 0.1
    high = close + rng.uniform(0, 0.4, (n, 2))
    low = close - rng.uniform(0, 0.4, (n, 2))
    for d in range(3):
        high[offs[d]] = 1e5
        low[offs[d]] = 1e-3
    out = features.session_range_pct(high, low, close, minute, offs)
    for d in range(3):
        sl = slice(offs[d] + 1, offs[d + 1])
        exp = (high[sl].max(axis=0) - low[sl].min(axis=0)) / close[offs[d + 1] - 1]
        np.testing.assert_allclose(out[d], exp, rtol=1e-13)


# ======================================================================================
# 1.3 adr_pct
# ======================================================================================


def test_adr_pct_hand_values():
    rp = np.array([0.01, 0.02, 0.03, 0.04, np.nan, 0.05, 0.06, 0.07])[:, None]
    out = features.adr_pct(rp, 2)
    assert out.shape == (8, 1)
    assert out.dtype == np.float64
    exp = [np.nan, np.nan, 0.015, 0.025, 0.035, np.nan, np.nan, 0.055]
    np.testing.assert_allclose(out[:, 0], exp, rtol=1e-13, equal_nan=True)


def test_adr_pct_n1_is_previous_day():
    rp = np.array([[0.01, 0.1], [0.02, 0.2], [0.03, 0.3]])
    out = features.adr_pct(rp, 1)
    assert np.all(np.isnan(out[0]))
    np.testing.assert_allclose(out[1:], rp[:-1], rtol=1e-15)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_adr_pct_causal(n):
    """Invariant 4: perturbing range_pct[j] changes only out[j+1 : j+n+1]."""
    rng = np.random.default_rng(n)
    rp = rng.uniform(0.005, 0.03, (15, 2))
    base = features.adr_pct(rp, n)
    for j in range(15):
        rp2 = rp.copy()
        rp2[j, 0] += 0.5
        out = features.adr_pct(rp2, n)
        changed = ~np.isclose(out[:, 0], base[:, 0], rtol=0, atol=0, equal_nan=True)
        expected = np.zeros(15, dtype=bool)
        expected[j + 1 : j + n + 1] = True
        expected[: n] = False  # NaN rows stay NaN
        np.testing.assert_array_equal(changed, expected)
        np.testing.assert_array_equal(out[:, 1], base[:, 1])


def test_adr_pct_rejects_n_below_one():
    with pytest.raises(ValueError):
        features.adr_pct(np.ones((3, 1)), 0)


# ======================================================================================
# 1.4 max_excursion
# ======================================================================================


def test_max_excursion_hand_values():
    minute = np.array([555, 556, 557, 919, 920, 556, 557])
    offs = np.array([0, 5, 7])
    close = np.array(
        [
            [150.0, 150.0, 150.0, 150.0],
            [101.0, 101.0, 101.0, 101.0],
            [97.0, 97.0, 97.0, 97.0],
            [102.0, 102.0, 102.0, 102.0],
            [200.0, 200.0, 200.0, 200.0],
            [104.0, 104.0, 104.0, 104.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    vwap = np.full_like(close, 100.0)
    vwap[2, 1] = np.nan  # sym1: row 2 excluded
    vwap[3, 1] = -1.0  # sym1: row 3 excluded (vwap <= 0)
    adr = np.array([[0.02, 0.02, 0.0, np.nan], [0.02, 0.02, 0.02, 0.02]])
    out = features.max_excursion(close, vwap, adr, minute, offs)
    assert out.shape == (2, 4)
    assert out.dtype == np.float64
    assert out[0, 0] == pytest.approx(3.0 / 2.0, rel=1e-13)
    assert out[0, 1] == pytest.approx(1.0 / 2.0, rel=1e-13)
    assert np.isnan(out[0, 2])  # adr == 0
    assert np.isnan(out[0, 3])  # adr NaN
    np.testing.assert_allclose(out[1], 4.0 / 2.0, rtol=1e-13)


def test_max_excursion_no_rows_is_nan():
    minute = np.array([555, 920, 921])
    close = np.full((3, 1), 150.0)
    vwap = np.full((3, 1), 100.0)
    out = features.max_excursion(close, vwap, np.array([[0.01]]), minute, np.array([0, 3]))
    assert np.isnan(out[0, 0])


# ======================================================================================
# 2.1 FadeParams
# ======================================================================================


def test_fade_params_defaults_and_frozen():
    p = simulate.FadeParams(k=1.5, notional=1e5)
    assert p.start_minute == 556
    assert p.square_off_minute == 920
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.k = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        dict(k=0.0, notional=1e5),
        dict(k=-1.0, notional=1e5),
        dict(k=1.0, notional=0.0),
        dict(k=1.0, notional=-1.0),
        dict(k=1.0, notional=1e5, start_minute=600, square_off_minute=600),
        dict(k=1.0, notional=1e5, start_minute=600, square_off_minute=599),
    ],
)
def test_fade_params_invalid(kw):
    with pytest.raises(ValueError):
        simulate.FadeParams(**kw)


# ======================================================================================
# 2.2 simulate_fade -- hand-built scenarios
# ======================================================================================


def test_short_entry_and_vwap_exit_all_columns():
    inp = _base(10)
    c, o = inp["close"], inp["open_"]
    c[2, 0] = 102.0  # > upper (101) -> short signal at row 2
    o[3, 0] = 101.5  # fill price
    c[3, 0] = 101.2  # short not yet back to vwap
    c[4, 0] = 99.5  # <= vwap -> pending
    o[5, 0] = 99.8  # exit fill
    df = _run(inp)
    assert list(df.columns) == COLUMNS
    t = _one(df)
    qty = NOTIONAL / 101.5
    gross = -1 * qty * (99.8 - 101.5)
    assert t["symbol"] == "S0"
    assert t["day_idx"] == 0
    assert t["side"] == -1
    assert t["signal_row"] == 2
    assert t["entry_row"] == 3
    assert t["entry_price"] == 101.5
    assert t["exit_row"] == 5
    assert t["exit_price"] == 99.8
    assert t["exit_reason"] == "vwap"
    assert t["qty"] == pytest.approx(qty, rel=1e-15)
    assert t["gross_pnl"] == pytest.approx(gross, rel=1e-12)
    assert t["gross_bps"] == pytest.approx(gross / NOTIONAL * 1e4, rel=1e-12)
    assert t["holding_bars"] == 2
    # Invariant 5: never at close[r] or at the band.
    assert t["entry_price"] != 102.0
    assert t["entry_price"] != pytest.approx(101.0)
    for col in INT_COLS:
        assert pd.api.types.is_integer_dtype(df[col]), col
    for col in FLOAT_COLS:
        assert df[col].dtype == np.float64, col
    assert isinstance(t["symbol"], str)


def test_long_entry_exit_on_equality():
    inp = _base(10)
    c, o = inp["close"], inp["open_"]
    c[2, 0] = 98.5
    o[3, 0] = 98.7
    c[3, 0] = 99.9
    c[4, 0] = 100.0  # == vwap counts for a long
    o[5, 0] = 100.2
    t = _one(_run(inp))
    qty = NOTIONAL / 98.7
    assert t["side"] == 1
    assert (t["signal_row"], t["entry_row"], t["exit_row"]) == (2, 3, 5)
    assert t["entry_price"] == 98.7
    assert t["exit_price"] == 100.2
    assert t["exit_reason"] == "vwap"
    assert t["gross_pnl"] == pytest.approx(qty * (100.2 - 98.7), rel=1e-12)


def test_band_boundary_is_strict():
    # adr=0.0625, k=0.5, vwap=128: upper = 132, lower = 124 exactly in binary.
    for px, expect in [(132.0, 0), (124.0, 0), (132.5, 1), (123.5, 1)]:
        inp = _base(6)
        inp["vwap"][:] = 128.0
        inp["close"][:] = 128.0
        inp["open_"][:] = 128.0
        inp["adr_day"][:] = 0.0625
        inp["close"][1, 0] = px
        assert len(_run(inp, k=0.5)) == expect, px


def test_entry_bar_close_can_trigger_exit():
    inp = _base(8)
    c, o = inp["close"], inp["open_"]
    c[2, 0] = 102.0
    o[3, 0] = 101.5
    c[3, 0] = 99.5  # entry bar's own close back through vwap
    o[4, 0] = 100.3
    t = _one(_run(inp))
    assert (t["entry_row"], t["exit_row"], t["exit_reason"]) == (3, 4, "vwap")
    assert t["exit_price"] == 100.3
    assert t["holding_bars"] == 1


def test_present_not_tradable_signal_dropped_not_deferred():
    """Invariant 6 (entry half)."""
    inp = _base(10)
    inp["close"][2, 0] = 102.0
    inp["tradable"][3, 0] = False  # row 3 present and fillable but not tradable
    assert np.isfinite(inp["close"][3, 0])
    df = _run(inp)
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


def test_new_signal_after_drop_is_its_own_signal():
    inp = _base(10)
    inp["close"][2, 0] = 102.0
    inp["tradable"][3, 0] = False
    inp["close"][3, 0] = 102.0  # a fresh signal at row 3
    inp["open_"][4, 0] = 101.7
    inp["close"][4, 0] = 99.0
    t = _one(_run(inp))
    assert t["signal_row"] == 3
    assert t["entry_row"] == 4
    assert t["entry_price"] == 101.7


@pytest.mark.parametrize(
    "case", ["open_nan", "open_zero", "open_neg", "vol_zero", "vol_nan", "nan_bar"]
)
def test_unfillable_next_row_drops_signal(case):
    inp = _base(10)
    inp["close"][2, 0] = 102.0
    if case == "open_nan":
        inp["open_"][3, 0] = np.nan
    elif case == "open_zero":
        inp["open_"][3, 0] = 0.0
    elif case == "open_neg":
        inp["open_"][3, 0] = -1.0
    elif case == "vol_zero":
        inp["volume"][3, 0] = 0.0
    elif case == "vol_nan":
        inp["volume"][3, 0] = np.nan
    elif case == "nan_bar":
        for key in ("open_", "close", "volume"):
            inp[key][3, 0] = np.nan
    assert len(_run(inp)) == 0


def test_signal_on_last_row_of_session_dropped():
    inp = _base(10, offs=[0, 5, 10])
    inp["close"][4, 0] = 102.0  # r+1 = 5 belongs to the next session
    assert len(_run(inp)) == 0


@pytest.mark.parametrize(
    "case", ["vwap_nan", "vwap_zero", "adr_nan", "adr_zero", "adr_neg", "close_nan", "before_start"]
)
def test_no_signal_conditions(case):
    minutes = np.arange(555, 565)
    inp = _base(10, minutes=minutes)
    r = 3
    if case == "before_start":
        r = 0  # the 09:15 row
    inp["close"][r, 0] = 102.0
    if case == "vwap_nan":
        inp["vwap"][r, 0] = np.nan
    elif case == "vwap_zero":
        inp["vwap"][r, 0] = 0.0
    elif case == "adr_nan":
        inp["adr_day"][0, 0] = np.nan
    elif case == "adr_zero":
        inp["adr_day"][0, 0] = 0.0
    elif case == "adr_neg":
        inp["adr_day"][0, 0] = -0.01
    elif case == "close_nan":
        inp["close"][r, 0] = np.nan
    assert len(_run(inp)) == 0


def test_custom_start_minute_param():
    inp = _base(10)  # minutes 556..565
    inp["close"][2, 0] = 102.0  # minute 558 < 560 -> no signal
    inp["close"][5, 0] = 102.0  # minute 561 -> signal
    inp["close"][6, 0] = 99.0
    t = _one(_run(inp, start_minute=560))
    assert t["signal_row"] == 5


# ---- square-off (minutes 910..924; row 10 is 15:20) -----------------------------------


def _sq_inputs():
    inp = _base(15, minutes=np.arange(910, 925))
    inp["close"][2, 0] = 102.0  # short signal at 15:12
    inp["open_"][3, 0] = 101.5
    inp["close"][3:, 0] = 100.5  # never back to vwap
    return inp


def test_square_off_at_1520_open():
    """Invariant 7."""
    inp = _sq_inputs()
    inp["open_"][10, 0] = 100.7
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (10, "square_off", 100.7)


def test_square_off_checked_before_vwap_rule_on_same_row():
    inp = _sq_inputs()
    inp["close"][10, 0] = 99.0  # 15:20 row also closes through vwap
    inp["open_"][10, 0] = 100.7
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (10, "square_off", 100.7)


def test_square_off_skips_unfillable_1520_row():
    inp = _sq_inputs()
    for key in ("open_", "close", "volume"):
        inp[key][10, 0] = np.nan  # 15:20 bar missing
    inp["volume"][11, 0] = 0.0  # 15:21 bar zero volume
    inp["open_"][12, 0] = 100.9
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (12, "square_off", 100.9)


def test_pending_vwap_exit_takes_precedence_over_square_off():
    inp = _sq_inputs()
    inp["close"][9, 0] = 99.5  # 15:19 close through vwap -> pending
    inp["open_"][10, 0] = 100.1
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (10, "vwap", 100.1)


def test_pending_vwap_exit_skips_unfillable_rows_past_square_off():
    inp = _sq_inputs()
    inp["close"][9, 0] = 99.5
    inp["open_"][10, 0] = np.nan
    inp["open_"][11, 0] = 100.2
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (11, "vwap", 100.2)


def test_no_signal_at_or_after_square_off_and_no_fill_at_1520():
    inp = _base(15, minutes=np.arange(910, 925))
    inp["close"][9, 0] = 102.0  # 15:19 signal would fill at 15:20 -> dropped
    inp["close"][10, 0] = 102.0  # 15:20: no signal
    inp["close"][11, 0] = 97.0  # 15:21: no signal
    assert len(_run(inp)) == 0


def test_custom_square_off_minute():
    inp = _base(10)  # minutes 556..565
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3:, 0] = 100.5
    inp["open_"][6, 0] = 100.6  # minute 562
    t = _one(_run(inp, square_off_minute=562))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (6, "square_off", 100.6)


# ---- exits do not need tradable -------------------------------------------------------


def test_exit_fills_when_not_tradable():
    """Invariant 6 (exit half)."""
    inp = _base(10)
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3, 0] = 101.2
    inp["close"][4, 0] = 99.5
    inp["open_"][5, 0] = 99.8
    inp["tradable"][4:, 0] = False
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (5, "vwap", 99.8)


def test_pending_exit_skips_unfillable_and_nan_rows():
    inp = _base(12)
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3, 0] = 101.2
    inp["close"][4, 0] = 99.5  # pending
    inp["open_"][5, 0] = np.nan  # not fillable
    for key in ("open_", "close", "volume"):
        inp[key][6, 0] = np.nan  # no bar
    inp["volume"][7, 0] = 0.0  # zero volume
    inp["open_"][8, 0] = 99.9
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (8, "vwap", 99.9)


# ---- no_bar ---------------------------------------------------------------------------


def test_no_bar_exit_at_last_qualifying_close():
    inp = _base(8)  # minutes 556..563, never reaches 15:20
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3:, 0] = 100.5
    inp["close"][5, 0] = 100.8
    inp["volume"][6, 0] = 0.0  # present but zero volume -> not x
    for key in ("open_", "close", "volume"):
        inp[key][7, 0] = np.nan  # no bar
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (5, "no_bar", 100.8)
    assert t["gross_pnl"] == pytest.approx(-(NOTIONAL / 101.5) * (100.8 - 101.5), rel=1e-12)


def test_pending_without_later_fillable_row_becomes_no_bar():
    inp = _base(6)
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3:, 0] = 100.5
    inp["close"][5, 0] = 99.5  # pending on last row
    t = _one(_run(inp))
    assert (t["exit_row"], t["exit_reason"], t["exit_price"]) == (5, "no_bar", 99.5)


def test_entry_on_last_row_exits_no_bar_same_row():
    inp = _base(6)
    inp["close"][4, 0] = 102.0
    inp["open_"][5, 0] = 101.5
    inp["close"][5, 0] = 100.5
    t = _one(_run(inp))
    assert (t["entry_row"], t["exit_row"], t["exit_reason"]) == (5, 5, "no_bar")
    assert t["exit_price"] == 100.5
    assert t["holding_bars"] == 0


def test_position_never_spans_sessions():
    """Invariant 7: 60-bar session ends open -> no_bar in that session; next session flat."""
    inp = _base(120, offs=[0, 60, 120], minutes=np.concatenate([556 + np.arange(60)] * 2))
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3:60, 0] = 100.5
    inp["close"][59, 0] = 100.9
    inp["close"][60, 0] = 99.5  # would be a short exit if the position leaked
    t = _one(_run(inp))
    assert (t["day_idx"], t["exit_row"], t["exit_reason"], t["exit_price"]) == (
        0,
        59,
        "no_bar",
        100.9,
    )


# ---- re-entry, sorting, empty ---------------------------------------------------------


def test_reentry_evaluated_from_exit_row():
    inp = _base(12)
    c, o = inp["close"], inp["open_"]
    c[2, 0] = 102.0
    o[3, 0] = 101.5
    c[3, 0] = 101.2
    c[4, 0] = 99.5
    o[5, 0] = 99.8  # first exit at row 5
    c[5, 0] = 102.5  # row 5's close signals a new short
    o[6, 0] = 102.1
    c[6, 0] = 101.5
    c[7, 0] = 99.0
    o[8, 0] = 99.6
    df = _run(inp)
    assert len(df) == 2
    first, second = df.iloc[0], df.iloc[1]
    assert first["exit_row"] == 5
    assert (second["signal_row"], second["entry_row"], second["exit_row"]) == (5, 6, 8)
    assert second["entry_price"] == 102.1
    assert second["exit_price"] == 99.6


def test_sorted_by_entry_row_then_symbol():
    inp = _base(12, n_sym=3)
    inp["symbols"] = ["ZED", "ABC", "MID"]
    c = inp["close"]
    # ZED enters first (row 2), ABC and MID and ZED(again) later at row 7.
    c[1, 0] = 102.0
    c[2, 0] = 99.0  # ZED exits pending -> fill at row 3
    c[6, :] = 102.0  # all three signal at row 6 -> entry row 7
    c[7, :] = 99.0
    df = _run(inp)
    got = list(zip(df["entry_row"], df["symbol"], strict=True))
    assert got == [(2, "ZED"), (7, "ABC"), (7, "MID"), (7, "ZED")]


def test_empty_result_has_exact_columns():
    inp = _base(20, n_sym=2)
    df = _run(inp)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


@pytest.mark.parametrize(
    "case",
    ["open", "close", "volume", "vwap", "tradable", "adr_rows", "adr_syms", "symbols", "minute"],
)
def test_simulate_value_errors(case):
    inp = _base(10, n_sym=2, offs=[0, 5, 10])
    if case == "open":
        inp["open_"] = np.full((9, 2), 100.0)
    elif case == "close":
        inp["close"] = np.full((10, 3), 100.0)
    elif case == "volume":
        inp["volume"] = np.full((11, 2), 1.0)
    elif case == "vwap":
        inp["vwap"] = np.full((10, 1), 100.0)
    elif case == "tradable":
        inp["tradable"] = np.ones((9, 2), dtype=bool)
    elif case == "adr_rows":
        inp["adr_day"] = np.full((3, 2), 0.01)
    elif case == "adr_syms":
        inp["adr_day"] = np.full((2, 3), 0.01)
    elif case == "symbols":
        inp["symbols"] = ["A"]
    elif case == "minute":
        inp["minute_of_day"] = np.arange(556, 565)
    with pytest.raises(ValueError):
        _run(inp)


def test_inputs_not_mutated_and_float32_accepted():
    inp = _base(10)
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = np.nan
    inp["close"][5, 0] = 102.0
    for key in ("open_", "close", "volume", "vwap", "adr_day"):
        inp[key] = inp[key].astype(np.float32)
    snap = {k: np.array(v, copy=True) for k, v in inp.items() if isinstance(v, np.ndarray)}
    df = _run(inp)
    for k, v in snap.items():
        np.testing.assert_array_equal(inp[k], v)
    t = _one(df)
    assert t["signal_row"] == 5
    assert df["gross_pnl"].dtype == np.float64


# ======================================================================================
# 2.2 costs (invariants 8, 9)
# ======================================================================================


def test_costs_zero_without_models():
    """Invariant 8."""
    inp = _random_panel(1)
    df = _run(inp, k=1.0)
    assert len(df) > 0
    assert (df["costs"] == 0.0).all()
    assert (df["net_pnl"] == df["gross_pnl"]).all()
    assert (df["net_bps"] == df["gross_bps"]).all()


def _expected_costs(df, inp, cost_model, slippage):
    close, volume = inp["close"], inp["volume"]
    sym_idx = {s: i for i, s in enumerate(inp["symbols"])}
    out = []
    for t in df.itertuples(index=False):
        s = sym_idx[t.symbol]
        n_in = t.qty * t.entry_price
        n_out = t.qty * t.exit_price
        charges = 0.0
        if cost_model is not None:
            fb = FillBatch(
                notional=np.array([n_in, n_out]),
                is_buy=np.array([t.side == 1, t.side == -1]),
            )
            charges = float(cost_model.charges(fb).total.sum())
        slip = 0.0
        if slippage is not None:
            for notional, row in ((n_in, t.entry_row), (n_out, t.exit_row)):
                btv = float(close[row, s]) * float(volume[row, s])
                bps = float(slippage.bps(np.array([notional]), np.array([btv]))[0])
                slip += notional * bps / 1e4
        out.append(charges + slip)
    return np.array(out)


@pytest.mark.parametrize("use_cost,use_slip", [(True, False), (False, True), (True, True)])
def test_costs_recomputed_independently(use_cost, use_slip):
    """Invariant 9 (and slippage per spec 2.2)."""
    inp = _random_panel(2)
    cm = NSEIntradayEquityCosts() if use_cost else None
    sl = SqrtImpactSlippage() if use_slip else None
    df = _run(inp, k=1.0, cost_model=cm, slippage=sl)
    assert len(df) > 5
    assert set(df["exit_reason"]) >= {"vwap"}
    exp = _expected_costs(df, inp, cm, sl)
    np.testing.assert_allclose(df["costs"].to_numpy(), exp, rtol=1e-12)
    assert (df["costs"] > 0).all()
    np.testing.assert_allclose(
        df["net_pnl"].to_numpy(), df["gross_pnl"].to_numpy() - exp, rtol=1e-12, atol=1e-9
    )
    np.testing.assert_allclose(
        df["net_bps"].to_numpy(), df["net_pnl"].to_numpy() / NOTIONAL * 1e4, rtol=1e-12, atol=1e-12
    )


def test_slippage_uses_close_times_volume_of_fill_row_including_no_bar():
    inp = _base(8)
    inp["close"][2, 0] = 102.0
    inp["open_"][3, 0] = 101.5
    inp["close"][3:, 0] = 100.5
    inp["close"][3, 0] = 100.2
    inp["volume"][3, 0] = 37.0
    inp["close"][7, 0] = 100.9
    inp["volume"][7, 0] = 11.0
    sl = SqrtImpactSlippage()
    t = _one(_run(inp, slippage=sl))
    assert t["exit_reason"] == "no_bar"
    qty = NOTIONAL / 101.5
    n_in, n_out = qty * 101.5, qty * 100.9
    b_in = 1.5 + 10.0 * math.sqrt(n_in / (100.2 * 37.0))
    b_out = 1.5 + 10.0 * math.sqrt(n_out / (100.9 * 11.0))
    exp = n_in * b_in / 1e4 + n_out * b_out / 1e4
    assert t["costs"] == pytest.approx(exp, rel=1e-12)


# ======================================================================================
# 2.2 full-rule cross-check against an independent transcription (invariants 5, 7, 10)
# ======================================================================================


def _random_panel(seed, n_sym=3):
    rng = np.random.default_rng(seed)
    lengths = [375, 60, 105, 375, 105, 60]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = np.concatenate([[0], np.cumsum(lengths)])
    n = int(offs[-1])
    vwap = 100 + np.cumsum(rng.standard_normal((n, n_sym)) * 0.02, axis=0)
    close = vwap + 1.2 * rng.standard_normal((n, n_sym))
    open_ = vwap + 1.2 * rng.standard_normal((n, n_sym))
    volume = rng.integers(0, 6, (n, n_sym)).astype(float) * 100.0
    tradable = rng.random((n, n_sym)) < 0.85
    nan_bar = rng.random((n, n_sym)) < 0.05
    for a in (open_, close, volume):
        a[nan_bar] = np.nan
    vwap[nan_bar] = np.nan
    adr = rng.uniform(0.005, 0.015, (len(lengths), n_sym))
    adr[1, 0] = np.nan
    adr[2, 1] = 0.0
    return dict(
        open_=open_,
        close=close,
        volume=volume,
        vwap=vwap,
        adr_day=adr,
        tradable=tradable,
        minute_of_day=minute.astype(np.int64),
        day_offsets=offs.astype(np.int64),
        symbols=[f"SYM{i}" for i in range(n_sym)],
    )


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_matches_independent_rule_transcription(seed):
    inp = _random_panel(seed)
    k = 1.0
    df = _run(inp, k=k)
    ref = _ref_trades(
        inp["open_"],
        inp["close"],
        inp["volume"],
        inp["vwap"],
        inp["adr_day"],
        inp["tradable"],
        inp["minute_of_day"],
        inp["day_offsets"],
        k,
    )
    ref.sort(key=lambda t: (t["entry_row"], inp["symbols"][t["sym"]]))
    assert len(ref) > 20
    assert {t["exit_reason"] for t in ref} >= {"vwap", "no_bar"}  # square_off: hand tests
    assert len(df) == len(ref)
    assert list(df["symbol"]) == [inp["symbols"][t["sym"]] for t in ref]
    for col in ("day_idx", "side", "signal_row", "entry_row", "exit_row", "exit_reason"):
        assert list(df[col]) == [t[col] for t in ref], col
    np.testing.assert_array_equal(df["entry_price"].to_numpy(), [t["entry_price"] for t in ref])
    np.testing.assert_array_equal(df["exit_price"].to_numpy(), [t["exit_price"] for t in ref])
    qty = NOTIONAL / np.array([t["entry_price"] for t in ref])
    np.testing.assert_allclose(df["qty"].to_numpy(), qty, rtol=1e-14)
    gross = (
        np.array([t["side"] for t in ref])
        * qty
        * (np.array([t["exit_price"] for t in ref]) - np.array([t["entry_price"] for t in ref]))
    )
    np.testing.assert_allclose(df["gross_pnl"].to_numpy(), gross, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(
        df["holding_bars"].to_numpy(), (df["exit_row"] - df["entry_row"]).to_numpy()
    )


@pytest.mark.parametrize("seed", [20, 21])
def test_structural_invariants_on_random_panel(seed):
    """Invariants 5, 6, 7, 10 checked directly from the output rows."""
    inp = _random_panel(seed)
    df = _run(inp, k=1.0)
    assert len(df) > 0
    offs, minute = inp["day_offsets"], inp["minute_of_day"]
    sidx = {s: i for i, s in enumerate(inp["symbols"])}
    for t in df.itertuples(index=False):
        s = sidx[t.symbol]
        lo, hi = offs[t.day_idx], offs[t.day_idx + 1]
        assert lo <= t.signal_row < t.entry_row <= t.exit_row < hi
        assert t.entry_row == t.signal_row + 1
        assert 556 <= minute[t.signal_row] < 920
        assert minute[t.entry_row] < 920
        assert inp["tradable"][t.entry_row, s]
        assert t.entry_price == inp["open_"][t.entry_row, s]
        if t.exit_reason == "no_bar":
            assert t.exit_price == inp["close"][t.exit_row, s]
        else:
            assert t.exit_price == inp["open_"][t.exit_row, s]
            assert _fillable(inp["open_"], inp["volume"], t.exit_row, s)
        if t.exit_reason == "square_off":
            assert minute[t.exit_row] >= 920
    # Invariant 10: at most one open position per symbol at any row.
    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("entry_row")
        assert (g["signal_row"].to_numpy()[1:] >= g["exit_row"].to_numpy()[:-1]).all()
        assert (g["entry_row"].to_numpy()[1:] > g["exit_row"].to_numpy()[:-1]).all()
    keys = list(zip(df["entry_row"], df["symbol"], strict=True))
    assert keys == sorted(keys)


# ======================================================================================
# Invariant 11 -- null behaviour
# ======================================================================================


def _null_run(seed):
    rng = np.random.default_rng(seed)
    lengths = [375, 105, 375, 60, 375, 105, 375, 375, 60, 375, 105, 375]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = np.concatenate([[0], np.cumsum(lengths)])
    n = int(offs[-1])
    n_sym = 6
    path = 1000.0 + np.cumsum(rng.standard_normal((n + 1, n_sym)) * 0.5, axis=0)
    open_ = path[:-1].copy()
    close = path[1:].copy()
    wick = np.abs(rng.standard_normal((2, n, n_sym))) * 0.2
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = rng.integers(100, 5000, (n, n_sym)).astype(float)
    gap = rng.random((n, n_sym)) < 0.01
    for a in (open_, close, high, low, volume):
        a[gap] = np.nan
    vwap = features.session_vwap(high, low, close, volume, minute, offs)
    adr = features.adr_pct(features.session_range_pct(high, low, close, minute, offs), 3)
    # k derived from the measured excursion distribution on THIS panel (rule 8): its median.
    exc = []
    for d in range(len(lengths)):
        rows = np.arange(offs[d], offs[d + 1])
        rows = rows[(minute[rows] >= 556) & (minute[rows] < 920)]
        for s in range(n_sym):
            a = adr[d, s]
            c, v = close[rows, s], vwap[rows, s]
            ok = np.isfinite(c) & np.isfinite(v) & (v > 0)
            if np.isfinite(a) and a > 0 and ok.any():
                exc.append(np.max(np.abs(c[ok] - v[ok]) / (v[ok] * a)))
    k = float(np.median(exc))
    assert np.isfinite(k) and k > 0
    df = simulate.simulate_fade(
        open_,
        close,
        volume,
        vwap,
        adr,
        np.ones((n, n_sym), dtype=bool),
        minute,
        offs,
        [f"N{i}" for i in range(n_sym)],
        simulate.FadeParams(k=k, notional=NOTIONAL),
    )
    return df


def test_null_random_walk_has_no_systematic_edge():
    """Invariant 11: rate of |t| > 2.58 across >= 30 seeds is <= 0.2. Seeds are fixed."""
    n_seeds = 30
    rejections = 0
    counts = []
    for seed in range(1000, 1000 + n_seeds):
        df = _null_run(seed)
        assert (df["costs"] == 0.0).all()
        x = df["gross_bps"].to_numpy(dtype=np.float64)
        counts.append(len(x))
        if len(x) >= 2 and np.std(x, ddof=1) > 0:
            t = np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x)))
            if abs(t) > 2.58:
                rejections += 1
    assert np.median(counts) >= 10, counts  # the null test must have trades to be meaningful
    assert rejections / n_seeds <= 0.2, rejections


# ======================================================================================
# 3. report
# ======================================================================================


def _trades_df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in INT_COLS:
        df[c] = df[c].astype(np.int64)
    for c in FLOAT_COLS:
        df[c] = df[c].astype(np.float64)
    return df


def _row(sym, day, side, entry, exit_, reason, gross, costs):
    net = gross - costs
    return [
        sym, day, side, max(entry - 1, 0), entry, 100.0, exit_, 100.0, reason,
        1000.0, gross, costs, net, gross, net, exit_ - entry,
    ]  # fmt: skip


def _hand_trades():
    # notional implied = 1e4 so bps == pnl
    return _trades_df(
        [
            _row("A", 0, 1, 0, 5, "vwap", 10.0, 2.0),
            _row("B", 0, -1, 3, 7, "square_off", -4.0, 1.0),
            _row("A", 1, -1, 12, 13, "vwap", 6.0, 3.0),
            _row("C", 3, 1, 30, 36, "no_bar", -1.0, 1.0),
        ]
    )


def test_daily_pnl():
    df = _hand_trades()
    net = report.daily_pnl(df, 5)
    assert net.dtype == np.float64
    np.testing.assert_array_equal(net, [3.0, 3.0, 0.0, -2.0, 0.0])
    gross = report.daily_pnl(df, 5, column="gross_pnl")
    np.testing.assert_array_equal(gross, [6.0, 6.0, 0.0, -1.0, 0.0])


def test_daily_pnl_empty():
    out = report.daily_pnl(_trades_df([]), 3)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.zeros(3))


def test_summarize_hand_values():
    df = _hand_trades()
    s = report.summarize(df, 5)
    required = {
        "trades", "trades_per_day", "win_rate", "pct_exit_vwap", "pct_exit_square_off",
        "pct_exit_no_bar", "mean_gross_bps", "mean_net_bps", "median_holding_bars",
        "total_gross_pnl", "total_net_pnl", "total_costs", "sharpe_net", "sharpe_gross",
        "max_concurrent", "worst_trade_net_bps", "worst_day_net_pnl", "long_trades",
        "short_trades",
    }  # fmt: skip
    assert required <= set(s)
    assert s["trades"] == 4
    assert s["trades_per_day"] == pytest.approx(0.8)
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["pct_exit_vwap"] == pytest.approx(0.5)
    assert s["pct_exit_square_off"] == pytest.approx(0.25)
    assert s["pct_exit_no_bar"] == pytest.approx(0.25)
    assert s["mean_gross_bps"] == pytest.approx(2.75)
    assert s["mean_net_bps"] == pytest.approx(1.0)
    assert s["median_holding_bars"] == pytest.approx(4.5)
    assert s["total_gross_pnl"] == pytest.approx(11.0)
    assert s["total_net_pnl"] == pytest.approx(4.0)
    assert s["total_costs"] == pytest.approx(7.0)
    exp_sn = sharpe_ratio(np.array([3.0, 3.0, 0.0, -2.0, 0.0]), periods_per_year=252)
    exp_sg = sharpe_ratio(np.array([6.0, 6.0, 0.0, -1.0, 0.0]), periods_per_year=252)
    assert s["sharpe_net"] == pytest.approx(exp_sn, rel=1e-12)
    assert s["sharpe_gross"] == pytest.approx(exp_sg, rel=1e-12)
    assert s["max_concurrent"] == 2
    assert s["worst_trade_net_bps"] == pytest.approx(-5.0)
    assert s["worst_day_net_pnl"] == pytest.approx(-2.0)
    assert s["long_trades"] == 2
    assert s["short_trades"] == 2


def test_summarize_max_concurrent_half_open():
    df = _trades_df(
        [
            _row("A", 0, 1, 0, 5, "vwap", 1.0, 0.0),
            _row("B", 0, 1, 5, 8, "vwap", 1.0, 0.0),  # starts where A ends -> not overlapping
        ]
    )
    assert report.summarize(df, 1)["max_concurrent"] == 1
    df3 = _trades_df(
        [
            _row("A", 0, 1, 0, 5, "vwap", 1.0, 0.0),
            _row("B", 0, 1, 3, 7, "vwap", 1.0, 0.0),
            _row("C", 0, -1, 4, 6, "vwap", 1.0, 0.0),
        ]
    )
    assert report.summarize(df3, 1)["max_concurrent"] == 3


def test_summarize_no_trades():
    s = report.summarize(_trades_df([]), 4)
    assert s["trades"] == 0
    assert s["trades_per_day"] == 0.0
    for key in (
        "win_rate", "pct_exit_vwap", "pct_exit_square_off", "pct_exit_no_bar",
        "mean_gross_bps", "mean_net_bps",
    ):  # fmt: skip
        assert math.isnan(s[key]), key
    assert s["long_trades"] == 0
    assert s["short_trades"] == 0
