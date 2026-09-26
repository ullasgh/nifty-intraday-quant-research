"""Independent test suite B for ``nifty_quant.research.liqprov`` (spec ``specs/liqprov_fade.md``).

Written from the spec alone, before and without sight of any implementation or of the other
independent suite. Style: adversarial / property-based. Every function under test has a slow,
deliberately naive plain-Python reference written straight from the spec text below, and the
module is compared to it on many small randomized panels (NaN holes, zero volume, random
tradable / gate / usable masks, 60 / 105 / 375-bar sessions, sessions crossing 15:20, float32
inputs, prices built on a dyadic grid so that trade-through edges such as ``low == L`` and
``low == L - tick`` occur exactly and are not lost to rounding). The randomized comparisons
also assert that every exit reason and both entry modes were actually exercised, so they
cannot pass vacuously.

Interpretations of ambiguous spec text are recorded next to the test that relies on them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from nifty_quant.execution.costs import FillBatch, NSEIntradayEquityCosts
from nifty_quant.execution.fills import SqrtImpactSlippage
from nifty_quant.research.liqprov import evaluate as E
from nifty_quant.research.liqprov import features as F
from nifty_quant.research.liqprov import simulate as S
from nifty_quant.research.vwap_adr import features as VF
from nifty_quant.research.vwap_adr.simulate import FadeParams, simulate_fade

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
START = 556
SQ = 920
NAN = float("nan")


# ----------------------------------------------------------------------------------------
# Calendar helpers
# ----------------------------------------------------------------------------------------


def _calendar(sessions):
    """sessions: list of (first_minute, n_bars) -> (minute_of_day, day_offsets)."""
    mods = [np.arange(m0, m0 + n, dtype=np.int64) for m0, n in sessions]
    offs = np.concatenate([[0], np.cumsum([n for _, n in sessions])]).astype(np.int64)
    return np.concatenate(mods), offs


def _day_of_row(offs):
    return np.repeat(np.arange(len(offs) - 1), np.diff(offs))


def _fin(x):
    return math.isfinite(x)


# ----------------------------------------------------------------------------------------
# Reference implementations of features.py, straight from spec section 1
# ----------------------------------------------------------------------------------------


def ref_index_twap(high, low, close, mod, offs, start_minute=START):
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape[0], NAN)
    for d in range(len(offs) - 1):
        tps = []
        for t in range(offs[d], offs[d + 1]):
            if mod[t] >= start_minute and _fin(h[t]) and _fin(lo[t]) and _fin(c[t]):
                tps.append((h[t] + lo[t] + c[t]) / 3.0)
            if _fin(c[t]) and mod[t] >= start_minute and tps:
                out[t] = sum(tps) / len(tps)
    return out


def ref_session_pairs(stock_close, index_close, mod, offs, start_minute=START):
    """pairs[d][s] = list of (x, y) 1-min log-return pairs of session d, symbol s."""
    sc = np.asarray(stock_close, dtype=np.float64)
    ic = np.asarray(index_close, dtype=np.float64)
    n_sym = sc.shape[1]
    pairs = []
    for d in range(len(offs) - 1):
        per_sym = [[] for _ in range(n_sym)]
        for t in range(offs[d] + 1, offs[d + 1]):
            if mod[t - 1] < start_minute:
                continue
            i0, i1 = ic[t - 1], ic[t]
            if not (_fin(i0) and _fin(i1) and i0 > 0 and i1 > 0):
                continue
            for s in range(n_sym):
                s0, s1 = sc[t - 1, s], sc[t, s]
                if _fin(s0) and _fin(s1) and s0 > 0 and s1 > 0:
                    per_sym[s].append((math.log(i1 / i0), math.log(s1 / s0)))
        pairs.append(per_sym)
    return pairs


def ref_prior_session_beta(stock_close, index_close, mod, offs, usable, lookback=20,
                           start_minute=START):
    pairs = ref_session_pairs(stock_close, index_close, mod, offs, start_minute)
    n_days = len(offs) - 1
    n_sym = np.asarray(stock_close).shape[1]
    out = np.full((n_days, n_sym), NAN)
    for d in range(n_days):
        prior = [j for j in range(d) if usable[j]]
        if len(prior) < lookback:
            continue
        use = prior[-lookback:]
        for s in range(n_sym):
            if any(len(pairs[j][s]) == 0 for j in use):
                continue
            pooled = [p for j in use for p in pairs[j][s]]
            n = len(pooled)
            sx = sum(p[0] for p in pooled)
            sy = sum(p[1] for p in pooled)
            sxy = sum(p[0] * p[1] for p in pooled)
            sxx = sum(p[0] * p[0] for p in pooled)
            den = n * sxx - sx * sx
            if den <= 0:
                continue
            out[d, s] = (n * sxy - sx * sy) / den
    return out


def ref_index_dev(index_close, twap):
    i = np.asarray(index_close, dtype=np.float64)
    w = np.asarray(twap, dtype=np.float64)
    out = np.full(i.shape[0], NAN)
    for t in range(i.shape[0]):
        if _fin(i[t]) and _fin(w[t]) and w[t] > 0:
            out[t] = i[t] / w[t] - 1.0
    return out


def ref_residual_stretch(close, vwap, idx_dev, beta_day, adr_day, offs):
    c = np.asarray(close, dtype=np.float64)
    vw = np.asarray(vwap, dtype=np.float64)
    x = np.asarray(idx_dev, dtype=np.float64)
    b = np.asarray(beta_day, dtype=np.float64)
    a = np.asarray(adr_day, dtype=np.float64)
    out = np.full(c.shape, NAN)
    day = _day_of_row(offs)
    for t in range(c.shape[0]):
        d = day[t]
        for s in range(c.shape[1]):
            vals = (c[t, s], vw[t, s], x[t], b[d, s], a[d, s])
            if not all(_fin(v) for v in vals) or vw[t, s] <= 0 or a[d, s] <= 0:
                continue
            out[t, s] = ((c[t, s] / vw[t, s] - 1.0) - b[d, s] * x[t]) / a[d, s]
    return out


def ref_vix_change(vix_close, mod, offs, start_minute=START):
    v = np.asarray(vix_close, dtype=np.float64)
    out = np.full(v.shape[0], NAN)
    for d in range(1, len(offs) - 1):
        prev = NAN
        for t in range(offs[d - 1], offs[d]):
            if _fin(v[t]):
                prev = v[t]
        if not (_fin(prev) and prev > 0):
            continue
        for t in range(offs[d], offs[d + 1]):
            if _fin(v[t]) and mod[t] >= start_minute:
                out[t] = v[t] / prev - 1.0
    return out


def ref_session_extreme(x, mod, offs, how, start_minute=START, end_minute=SQ):
    arr = np.asarray(x, dtype=np.float64)
    two_d = arr.ndim == 2
    a2 = arr if two_d else arr[:, None]
    n_days = len(offs) - 1
    out = np.full((n_days, a2.shape[1]), NAN)
    for d in range(n_days):
        for s in range(a2.shape[1]):
            vals = [a2[t, s] for t in range(offs[d], offs[d + 1])
                    if start_minute <= mod[t] < end_minute and _fin(a2[t, s])]
            if vals:
                out[d, s] = max(vals) if how == "max" else min(vals)
    return out if two_d else out[:, 0]


def ref_rolling_quantile(per_session, usable, q, window=250, min_sessions=120):
    ps = np.asarray(per_session, dtype=np.float64)
    out = np.full(ps.shape[0], NAN)
    for d in range(ps.shape[0]):
        idx = [j for j in range(d) if usable[j] and _fin(ps[j])]
        vals = ps[idx[-window:]] if idx else np.array([])
        if len(vals) >= min_sessions:
            out[d] = np.quantile(vals, q)
    return out


def ref_stress_gate(vix_chg, idx_dev_adr, v_day, m_day, offs):
    day = _day_of_row(offs)
    n = len(vix_chg)
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        d = day[t]
        vals = (float(vix_chg[t]), float(idx_dev_adr[t]), float(v_day[d]), float(m_day[d]))
        if all(_fin(v) for v in vals):
            out[t] = vals[0] >= vals[2] and vals[1] <= vals[3]
    return out


# ----------------------------------------------------------------------------------------
# Reference implementation of simulate_liqprov, straight from spec section 2
# ----------------------------------------------------------------------------------------


def ref_simulate(inp, k, notional, mode, tick=0.05, start=START, sq=SQ, cost_model=None,
                 slippage=None, stats=None):
    """Row-by-row state machine. ``stats`` (a Counter) records which edges were exercised."""
    if stats is None:
        stats = Counter()
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    o, lo, c, v, vw = (f64(inp[n]) for n in ("open_", "low", "close", "volume", "vwap"))
    idv = f64(inp["idx_dev"])
    beta, adr = f64(inp["beta_day"]), f64(inp["adr_day"])
    gate = np.asarray(inp["gate"], dtype=bool)
    trd = np.asarray(inp["tradable"], dtype=bool)
    mod = np.asarray(inp["minute_of_day"], dtype=np.int64)
    offs = np.asarray(inp["day_offsets"], dtype=np.int64)
    symbols = list(inp["symbols"])
    recs = []

    for s in range(c.shape[1]):
        for d in range(len(offs) - 1):
            a, b = int(offs[d]), int(offs[d + 1])
            B, A = beta[d, s], adr[d, s]

            def band(t):
                V, X = vw[t, s], idv[t]
                if not (_fin(V) and _fin(X) and _fin(B) and _fin(A) and V > 0 and A > 0):
                    return NAN
                return V * (1.0 + B * X - k * A)

            def fillable(x):
                return _fin(o[x, s]) and o[x, s] > 0 and _fin(v[x, s]) and v[x, s] > 0

            def find_exit(e):
                pending = False
                for r in range(e, b):
                    if pending and fillable(r):
                        return r, o[r, s], "vwap"
                    if not pending and mod[r] >= sq and fillable(r):
                        return r, o[r, s], "square_off"
                    if (not pending and _fin(c[r, s]) and _fin(vw[r, s])
                            and c[r, s] >= vw[r, s]):
                        pending = True
                for x in range(b - 1, e - 1, -1):
                    if _fin(c[x, s]) and c[x, s] > 0 and _fin(v[x, s]) and v[x, s] > 0:
                        return x, c[x, s], "no_bar"
                raise AssertionError("reference: no no_bar exit row")  # pragma: no cover

            def record(sig, e, px, market):
                x, xpx, why = find_exit(e)
                recs.append((symbols[s], d, sig, e, px, x, xpx, why, market, s))
                stats["exit_" + why] += 1
                return x

            if mode == "passive":
                t = a + 1
                while t < b:
                    p = t - 1
                    L = band(p)
                    ok = (mod[p] >= start and mod[t] < sq and bool(gate[p]) and _fin(L)
                          and L > 0 and fillable(t) and bool(trd[t, s]) and _fin(lo[t, s]))
                    if ok and lo[t, s] <= L - tick:
                        px = min(L, o[t, s])
                        stats["gap_entry" if o[t, s] < L else "limit_entry"] += 1
                        if lo[t, s] == L - tick:
                            stats["fill_at_exactly_L_minus_tick"] += 1
                        t = record(p, t, px, False) + 1
                        continue
                    if ok and lo[t, s] <= L:
                        stats["no_fill_low_within_tick"] += 1
                        if lo[t, s] == L:
                            stats["no_fill_low_eq_L"] += 1
                    t += 1
            else:
                r = a
                while r < b:
                    bd = band(r)
                    sig = (_fin(c[r, s]) and bool(gate[r]) and start <= mod[r] < sq
                           and _fin(bd) and bd > 0 and c[r, s] <= bd)
                    if sig:
                        e = r + 1
                        if e < b and mod[e] < sq and fillable(e) and bool(trd[e, s]):
                            if c[r, s] == bd:
                                stats["signal_close_eq_band"] += 1
                            r = record(r, e, o[e, s], True)
                            continue
                        stats["dropped_signal"] += 1
                    r += 1

    rows = []
    for sym, d, sig, e, px, x, xpx, why, market, s in recs:
        qty = notional / px
        gross = qty * (xpx - px)
        en, xn = qty * px, qty * xpx
        charges = 0.0
        if cost_model is not None:
            fb = FillBatch(notional=np.array([en, xn]), is_buy=np.array([True, False]))
            charges = float(np.sum(cost_model.charges(fb).total))
        slip = 0.0
        if slippage is not None:
            btv_x = c[x, s] * v[x, s]
            slip += xn * float(slippage.bps(np.array([xn]), np.array([btv_x]))[0]) / 1e4
            if market:
                btv_e = c[e, s] * v[e, s]
                slip += en * float(slippage.bps(np.array([en]), np.array([btv_e]))[0]) / 1e4
        costs = charges + slip
        net = gross - costs
        rows.append((sym, d, 1, sig, e, px, x, xpx, why, qty, gross, costs, net,
                     gross / notional * 1e4, net / notional * 1e4, x - e))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values(["entry_row", "symbol"], kind="mergesort").reset_index(drop=True)


def _run_module(inp, k, notional, mode, tick, cost_model=None, slippage=None):
    params = S.LiqParams(k=k, notional=notional, mode=mode, tick=tick)
    return S.simulate_liqprov(
        inp["open_"], inp["high"], inp["low"], inp["close"], inp["volume"], inp["vwap"],
        inp["idx_dev"], inp["beta_day"], inp["adr_day"], inp["gate"], inp["tradable"],
        inp["minute_of_day"], inp["day_offsets"], inp["symbols"], params,
        cost_model=cost_model, slippage=slippage,
    )


def _assert_frames_match(got, exp, ctx=""):
    assert list(got.columns) == COLUMNS, ctx
    assert len(got) == len(exp), f"{ctx}: {len(got)} trades vs reference {len(exp)}\n" \
        f"got:\n{got[COLUMNS[:9]]}\nexp:\n{exp[COLUMNS[:9]]}"
    if len(exp) == 0:
        return
    assert list(got["symbol"]) == list(exp["symbol"]), ctx
    assert list(got["exit_reason"]) == list(exp["exit_reason"]), ctx
    for col in INT_COLS:
        assert np.array_equal(got[col].to_numpy().astype(np.int64),
                              exp[col].to_numpy().astype(np.int64)), f"{ctx}: {col}"
    for col in ("entry_price", "exit_price", "qty"):
        np.testing.assert_allclose(got[col].to_numpy(np.float64), exp[col].to_numpy(np.float64),
                                   rtol=1e-12, atol=0, err_msg=f"{ctx}: {col}")
    for col in ("gross_pnl", "costs", "net_pnl", "gross_bps", "net_bps"):
        np.testing.assert_allclose(got[col].to_numpy(np.float64), exp[col].to_numpy(np.float64),
                                   rtol=1e-12, atol=1e-8, err_msg=f"{ctx}: {col}")


def _assert_structural_invariants(trades, offs):
    """Invariant 7 (+ output bookkeeping): long only, one session, one position per symbol."""
    if len(trades) == 0:
        return
    assert (trades["side"] == 1).all()
    day = _day_of_row(offs)
    er = trades["entry_row"].to_numpy(np.int64)
    xr = trades["exit_row"].to_numpy(np.int64)
    assert np.array_equal(day[er], day[xr]), "a position spans two sessions"
    assert np.array_equal(trades["day_idx"].to_numpy(np.int64), day[er])
    assert np.array_equal(trades["holding_bars"].to_numpy(np.int64), xr - er)
    assert np.array_equal(trades["signal_row"].to_numpy(np.int64), er - 1)
    assert (xr >= er).all()
    for _, g in trades.groupby("symbol"):
        g = g.sort_values("entry_row")
        e2 = g["entry_row"].to_numpy(np.int64)
        x2 = g["exit_row"].to_numpy(np.int64)
        assert (e2[1:] > x2[:-1]).all(), "overlapping positions for one symbol"
    key = list(zip(er, trades["symbol"]))
    assert key == sorted(key), "not sorted by (entry_row, symbol)"


# ----------------------------------------------------------------------------------------
# Randomized simulate panels on a dyadic price grid (exact in float32 and float64)
# ----------------------------------------------------------------------------------------

_SESSION_TEMPLATES = [
    (555, 375),  # full session, crosses 15:20
    (555, 60),  # short session ending 10:14 (no square-off possible)
    (555, 105),  # shortened session ending before 15:20
    (860, 70),  # crosses 15:20 at row 60
    (835, 105),  # 105-bar session crossing 15:20
    (1095, 60),  # Muhurat-style evening session, every minute >= 920
    (890, 45),
]
_SYMBOLS = ["ZETA", "ALPHA", "MU"]  # deliberately not in sorted order
_TICK = 1.0 / 16.0  # dyadic tick so L - tick is exact under any evaluation order


def _g(x):
    """Round onto the 1/4096 grid (values < 256 are then exact in float32)."""
    return np.round(np.asarray(x, dtype=np.float64) * 4096.0) / 4096.0


def _band_array(vwap, idx_dev, beta_day, adr_day, offs, k):
    day = _day_of_row(offs)
    with np.errstate(invalid="ignore"):
        b = vwap * (1.0 + beta_day[day] * idx_dev[:, None] - k * adr_day[day])
        ok = (np.isfinite(vwap) & np.isfinite(idx_dev)[:, None] & np.isfinite(beta_day[day])
              & np.isfinite(adr_day[day]) & (vwap > 0) & (adr_day[day] > 0))
    return np.where(ok, b, NAN)


def _rand_sim_panel(seed):
    rng = np.random.default_rng(seed)
    n_days = int(rng.integers(2, 5))
    choice = rng.choice(len(_SESSION_TEMPLATES), size=n_days)
    sessions = [_SESSION_TEMPLATES[i] for i in choice]
    mod, offs = _calendar(sessions)
    n, m = len(mod), len(_SYMBOLS)
    k = float(rng.choice([0.5, 1.0, 2.0]))

    steps = rng.choice([-0.25, 0.0, 0.25], size=(n, m))
    vwap = np.clip(np.round(rng.uniform(100, 140, size=m) * 4) / 4 + np.cumsum(steps, 0), 70, 180)
    close = _g(vwap * (1 + rng.normal(-0.008, 0.012, (n, m))))
    open_ = _g(vwap * (1 + rng.normal(-0.008, 0.012, (n, m))))
    low = _g(np.minimum(open_, close) - vwap * np.abs(rng.normal(0, 0.004, (n, m))))
    high = _g(np.maximum(open_, close) + vwap * np.abs(rng.normal(0, 0.004, (n, m))))
    volume = rng.integers(1, 5000, size=(n, m)).astype(np.float64)

    idx_dev = rng.integers(-8, 9, size=n) / 512.0
    idx_dev[rng.random(n) < 0.05] = NAN
    beta_day = rng.choice([-0.5, 0.0, 0.5, 1.0, 1.5, NAN], size=(n_days, m),
                          p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
    adr_day = rng.choice([2 / 256, 3 / 256, 4 / 256, 6 / 256, 0.0, -1 / 256, NAN],
                         size=(n_days, m), p=[0.22, 0.22, 0.22, 0.14, 0.07, 0.06, 0.07])

    # Engineer exact edges against the band (computed before holes are punched).
    band = _band_array(vwap, idx_dev, beta_day, adr_day, offs, k)
    prev_same = np.zeros(n, dtype=bool)
    prev_same[1:] = _day_of_row(offs)[1:] == _day_of_row(offs)[:-1]
    for t in np.flatnonzero(prev_same):
        for s in range(m):
            L = band[t - 1, s]
            u = rng.random()
            if np.isfinite(L) and L > 0:
                if u < 0.07:
                    low[t, s] = L  # must NOT fill
                elif u < 0.14:
                    low[t, s] = L - _TICK  # must fill, at L unless open is lower
                elif u < 0.18:
                    open_[t, s] = L - 2 * _TICK  # gap fill at the open
                    low[t, s] = min(low[t, s], open_[t, s])
            u2 = rng.random()
            if u2 < 0.05 and np.isfinite(band[t, s]) and band[t, s] > 0:
                close[t, s] = band[t, s]  # market signal edge (<= is inclusive)
            elif u2 < 0.10:
                close[t, s] = vwap[t, s]  # vwap exit edge (>= is inclusive)

    # Holes and degenerate bars. A missing bar is missing in every field (no fill anywhere).
    hole = rng.random((n, m)) < 0.07
    for arr in (open_, high, low, close, volume, vwap):
        arr[hole] = NAN
    volume[(rng.random((n, m)) < 0.07) & ~hole] = 0.0  # present, not fillable
    volume[(rng.random((n, m)) < 0.02) & ~hole] = NAN  # present, volume unknown
    low[(rng.random((n, m)) < 0.02) & ~hole] = NAN  # present, low missing
    vwap[rng.random((n, m)) < 0.03] = NAN
    vwap[(rng.random((n, m)) < 0.005) & ~hole] = 0.0
    neg = (rng.random((n, m)) < 0.005) & ~hole
    vwap[neg] = -vwap[neg]

    gate = rng.random(n) < 0.7
    tradable = rng.random((n, m)) < 0.85
    inp = dict(open_=open_, high=high, low=low, close=close, volume=volume, vwap=vwap,
               idx_dev=idx_dev, beta_day=beta_day, adr_day=adr_day, gate=gate,
               tradable=tradable, minute_of_day=mod, day_offsets=offs, symbols=list(_SYMBOLS))
    is_f32 = seed % 3 == 0
    if is_f32:
        for key in ("open_", "high", "low", "close", "volume", "vwap", "idx_dev", "beta_day",
                    "adr_day"):
            f32 = inp[key].astype(np.float32)
            assert np.array_equal(f32.astype(np.float64), inp[key], equal_nan=True)
            inp[key] = f32
    return inp, k, sessions, is_f32


_N_SIM_SEEDS = 60


@pytest.mark.parametrize("mode", ["passive", "market"])
def test_simulate_matches_reference_on_random_panels(mode):
    stats = Counter()
    lengths_seen = set()
    crossing_seen = False
    f32_trades = 0
    costed_trades = 0
    for seed in range(_N_SIM_SEEDS):
        inp, k, sessions, is_f32 = _rand_sim_panel(seed)
        lengths_seen.update(n for _, n in sessions)
        crossing_seen |= any(m0 < SQ <= m0 + n - 1 for m0, n in sessions)
        costs = (NSEIntradayEquityCosts(), SqrtImpactSlippage()) if seed % 2 else (None, None)
        exp = ref_simulate(inp, k, 250_000.0, mode, tick=_TICK, cost_model=costs[0],
                           slippage=costs[1], stats=stats)
        got = _run_module(inp, k, 250_000.0, mode, _TICK, cost_model=costs[0],
                          slippage=costs[1])
        _assert_frames_match(got, exp, ctx=f"mode={mode} seed={seed}")
        _assert_structural_invariants(got, inp["day_offsets"])
        if len(got):
            assert got["gross_bps"].dtype == np.float64
        f32_trades += len(got) if is_f32 else 0
        costed_trades += len(got) if costs[0] is not None else 0

    # Non-vacuity: the random panels must actually exercise what they claim to.
    assert {60, 105, 375} <= lengths_seen
    assert crossing_seen
    assert f32_trades > 0 and costed_trades > 0
    for why in ("vwap", "square_off", "no_bar"):
        assert stats["exit_" + why] > 0, (mode, why, stats)
    if mode == "passive":
        assert stats["limit_entry"] > 0 and stats["gap_entry"] > 0, stats
        assert stats["fill_at_exactly_L_minus_tick"] > 0, stats
        assert stats["no_fill_low_eq_L"] > 0, stats
    else:
        assert stats["dropped_signal"] > 0, stats
        assert stats["signal_close_eq_band"] > 0, stats


# ----------------------------------------------------------------------------------------
# Targeted simulate edge cases on a hand-built one-symbol panel
# ----------------------------------------------------------------------------------------


def _flat_panel(n_bars=12, first_minute=555, sessions=None, vwap_level=128.0):
    """One symbol. vwap=128, beta=0, idx_dev=0, adr=1/64 -> with k=1 band = 126 exactly."""
    sessions = sessions or [(first_minute, n_bars)]
    mod, offs = _calendar(sessions)
    n = len(mod)
    col = lambda val: np.full((n, 1), val, dtype=np.float64)  # noqa: E731
    inp = dict(
        open_=col(127.0), high=col(127.5), low=col(126.5), close=col(127.0),
        volume=col(1000.0), vwap=col(vwap_level), idx_dev=np.zeros(n),
        beta_day=np.zeros((len(sessions), 1)), adr_day=np.full((len(sessions), 1), 1 / 64),
        gate=np.ones(n, dtype=bool), tradable=np.ones((n, 1), dtype=bool),
        minute_of_day=mod, day_offsets=offs, symbols=["AAA"],
    )
    return inp


def _sim(inp, mode, k=1.0, tick=0.05, notional=100_000.0, **kw):
    params = S.LiqParams(k=k, notional=notional, mode=mode, tick=tick)
    return S.simulate_liqprov(
        inp["open_"], inp["high"], inp["low"], inp["close"], inp["volume"], inp["vwap"],
        inp["idx_dev"], inp["beta_day"], inp["adr_day"], inp["gate"], inp["tradable"],
        inp["minute_of_day"], inp["day_offsets"], inp["symbols"], params, **kw,
    )


L_FLAT = 126.0  # band of _flat_panel with k = 1


def test_flat_panel_band_is_what_the_targeted_tests_assume():
    inp = _flat_panel()
    b = _band_array(inp["vwap"], inp["idx_dev"], inp["beta_day"], inp["adr_day"],
                    inp["day_offsets"], 1.0)
    assert np.all(b[:, 0] == L_FLAT)


def test_passive_low_equal_to_limit_does_not_fill():
    inp = _flat_panel()
    inp["low"][5, 0] = L_FLAT
    tr = _sim(inp, "passive")
    assert len(tr) == 0
    assert list(tr.columns) == COLUMNS


def test_passive_low_one_tick_through_fills_at_limit():
    inp = _flat_panel()
    inp["low"][5, 0] = L_FLAT - 0.05
    tr = _sim(inp, "passive")
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["signal_row"] == 4 and t["entry_row"] == 5
    assert t["entry_price"] == L_FLAT
    assert t["side"] == 1


def test_passive_gap_below_limit_fills_at_open():
    inp = _flat_panel()
    inp["open_"][5, 0] = 125.5
    inp["low"][5, 0] = 125.4
    tr = _sim(inp, "passive")
    assert len(tr) == 1 and tr.iloc[0]["entry_price"] == 125.5


def test_passive_open_between_limit_and_trade_through_fills_at_open():
    # open in (L - tick, L): min(L, open) = open.
    inp = _flat_panel()
    inp["open_"][5, 0] = 125.98
    inp["low"][5, 0] = 125.9
    tr = _sim(inp, "passive")
    assert len(tr) == 1 and tr.iloc[0]["entry_price"] == 125.98


def test_passive_uses_gate_of_previous_row_not_current():
    inp = _flat_panel()
    inp["low"][5, 0] = 125.0
    inp["gate"][:] = False
    inp["gate"][5] = True  # gate[t] True, gate[t-1] False -> no fill at t
    assert len(_sim(inp, "passive")) == 0
    inp["gate"][:] = False
    inp["gate"][4] = True  # gate[t-1] True, gate[t] False -> fills
    assert len(_sim(inp, "passive")) == 1


def test_passive_uses_band_of_previous_row():
    inp = _flat_panel()
    inp["low"][5, 0] = 125.9  # through L=126 by more than a tick
    inp["vwap"][4, 0] = 127.0  # band[4] = 127*(63/64) = 125.015625 -> not reached
    assert len(_sim(inp, "passive")) == 0
    inp = _flat_panel()
    inp["low"][5, 0] = 125.9
    inp["vwap"][5, 0] = 127.0  # band at t itself is irrelevant
    tr = _sim(inp, "passive")
    assert len(tr) == 1 and tr.iloc[0]["entry_price"] == L_FLAT


@pytest.mark.parametrize("field,value", [("tradable", False), ("volume", 0.0),
                                         ("volume", NAN), ("open_", NAN), ("open_", 0.0),
                                         ("low", NAN)])
def test_passive_entry_row_must_be_fillable_tradable_with_finite_low(field, value):
    inp = _flat_panel()
    inp["low"][5, 0] = 125.0
    if field != "low":
        inp[field][5, 0] = value
    else:
        inp["low"][5, 0] = value
    tr = _sim(inp, "passive")
    assert not (tr["entry_row"] == 5).any()


def test_passive_present_but_not_tradable_gives_no_fill_and_no_deferral():
    inp = _flat_panel()
    inp["low"][5, 0] = 125.0
    inp["tradable"][5, 0] = False
    inp["low"][6, 0] = 126.2  # not through at row 6
    assert len(_sim(inp, "passive")) == 0


def test_passive_no_fill_when_order_row_is_0915_bar():
    inp = _flat_panel()  # row 0 is 09:15 (555), row 1 is 09:16
    inp["low"][1, 0] = 125.0
    assert len(_sim(inp, "passive")) == 0


def test_passive_no_fill_at_or_after_square_off_minute():
    inp = _flat_panel(n_bars=6, first_minute=917)  # minutes 917..922
    inp["low"][3, 0] = 125.0  # row 3 = 920
    assert len(_sim(inp, "passive")) == 0
    inp = _flat_panel(n_bars=6, first_minute=917)
    inp["low"][2, 0] = 125.0  # row 2 = 919 fills, then square-off at 920's open
    tr = _sim(inp, "passive")
    assert len(tr) == 1
    assert tr.iloc[0]["entry_row"] == 2 and tr.iloc[0]["exit_row"] == 3
    assert tr.iloc[0]["exit_reason"] == "square_off"


def test_passive_order_does_not_cross_session_boundary():
    inp = _flat_panel(sessions=[(600, 5), (600, 5)])
    inp["low"][5, 0] = 125.0  # first row of session 1; t-1 is in session 0
    tr = _sim(inp, "passive")
    assert not (tr["entry_row"] == 5).any()


def test_passive_reentry_only_from_exit_row_plus_one():
    inp = _flat_panel(n_bars=14, first_minute=600)
    inp["low"][3, 0] = 125.0  # entry at 3 (order from row 2)
    inp["close"][3, 0] = 127.0  # < vwap, hold
    inp["close"][5, 0] = 128.0  # close >= vwap -> exit pending -> fills at open[6]
    inp["low"][6, 0] = 125.0  # would fill at row 6 but position is open until 6's open
    inp["low"][7, 0] = 125.0  # order placed at close of 6 -> fills at 7
    inp["close"][7, 0] = 127.0
    tr = _sim(inp, "passive")
    assert list(tr["entry_row"]) == [3, 7]
    assert list(tr["signal_row"]) == [2, 6]
    assert tr.iloc[0]["exit_row"] == 6 and tr.iloc[0]["exit_reason"] == "vwap"


def test_entry_bar_close_can_trigger_vwap_exit():
    inp = _flat_panel(n_bars=10, first_minute=600)
    inp["low"][3, 0] = 125.0
    inp["close"][3, 0] = 128.0  # entry bar's own close reaches vwap
    inp["open_"][4, 0] = 127.25
    tr = _sim(inp, "passive")
    assert tr.iloc[0]["exit_row"] == 4 and tr.iloc[0]["exit_price"] == 127.25
    assert tr.iloc[0]["exit_reason"] == "vwap"


def test_square_off_skips_unfillable_rows_and_vwap_pending_takes_precedence():
    inp = _flat_panel(n_bars=8, first_minute=915)  # 915..922
    inp["low"][2, 0] = 125.0  # entry at 917
    inp["volume"][5, 0] = 0.0  # 920 not fillable
    inp["open_"][6, 0] = 126.75
    tr = _sim(inp, "passive")
    assert tr.iloc[0]["exit_row"] == 6 and tr.iloc[0]["exit_reason"] == "square_off"
    assert tr.iloc[0]["exit_price"] == 126.75
    inp["close"][4, 0] = 128.0  # 919 close >= vwap -> pending -> fills at 920? not fillable
    tr = _sim(inp, "passive")  # -> fills at 921 with reason vwap
    assert tr.iloc[0]["exit_row"] == 6 and tr.iloc[0]["exit_reason"] == "vwap"


def test_no_bar_exit_at_last_close_with_volume():
    inp = _flat_panel(n_bars=10, first_minute=600)  # never reaches 920
    inp["low"][3, 0] = 125.0
    inp["close"][8, 0] = 126.5
    inp["volume"][9, 0] = 0.0  # last row not eligible
    tr = _sim(inp, "passive")
    t = tr.iloc[0]
    assert t["exit_reason"] == "no_bar" and t["exit_row"] == 8 and t["exit_price"] == 126.5


def test_market_close_equal_to_band_signals_and_fills_next_open():
    inp = _flat_panel(n_bars=10, first_minute=600)
    inp["close"][3, 0] = L_FLAT
    inp["open_"][4, 0] = 126.25
    tr = _sim(inp, "market")
    assert len(tr) == 1
    t = tr.iloc[0]
    assert (t["signal_row"], t["entry_row"], t["entry_price"]) == (3, 4, 126.25)
    inp["close"][3, 0] = L_FLAT + 1 / 64
    assert len(_sim(inp, "market")) == 0


def test_market_dropped_signal_is_not_deferred():
    inp = _flat_panel(n_bars=10, first_minute=600)
    inp["close"][3, 0] = 125.0
    inp["tradable"][4, 0] = False  # present, fillable, but not tradable
    tr = _sim(inp, "market")
    assert len(tr) == 0
    inp["tradable"][4, 0] = True
    inp["volume"][4, 0] = 0.0
    assert len(_sim(inp, "market")) == 0


def test_market_requires_gate_on_signal_row_and_present_bar():
    inp = _flat_panel(n_bars=10, first_minute=600)
    inp["close"][3, 0] = 125.0
    inp["gate"][3] = False
    inp["gate"][4] = True
    assert len(_sim(inp, "market")) == 0


def test_market_no_signal_at_or_after_square_off():
    inp = _flat_panel(n_bars=4, first_minute=919)
    inp["close"][1, 0] = 125.0  # 920
    assert len(_sim(inp, "market")) == 0
    inp = _flat_panel(n_bars=4, first_minute=918)
    inp["close"][1, 0] = 125.0  # 919 signals, but 920 fill is >= square-off -> dropped
    assert len(_sim(inp, "market")) == 0


def test_market_reevaluates_from_exit_row():
    inp = _flat_panel(n_bars=12, first_minute=600)
    inp["close"][2, 0] = 125.0  # signal -> entry at 3
    inp["close"][3, 0] = 128.5  # exit pending -> fills at open[4]
    inp["close"][4, 0] = 125.0  # row 4 itself signals again -> entry at 5
    tr = _sim(inp, "market")
    assert list(tr["entry_row"]) == [3, 5]
    assert list(tr["exit_row"])[0] == 4


def test_market_mode_matches_vwap_adr_long_entries():
    """Invariant 6: with beta=0, gate all True and no short triggers, market-mode trades are
    exactly simulate_fade's long trades (entry timing, drops, exits)."""
    rng = np.random.default_rng(12345)
    mod, offs = _calendar([(555, 375), (555, 60), (835, 105), (555, 105)])
    n, m = len(mod), 4
    vwap = 100 + np.cumsum(rng.normal(0, 0.05, (n, m)), 0)
    adr = rng.uniform(0.005, 0.02, (4, m))
    day = _day_of_row(offs)
    upper = vwap * (1 + 1.3 * adr[day])
    close = np.minimum(vwap * (1 + rng.normal(-0.01, 0.01, (n, m))), upper * 0.999)
    open_ = vwap * (1 + rng.normal(-0.005, 0.01, (n, m)))
    vol = rng.integers(0, 50, (n, m)).astype(float)
    hole = rng.random((n, m)) < 0.08
    close[hole] = open_[hole] = vol[hole] = vwap[hole] = NAN
    trd = rng.random((n, m)) < 0.8
    syms = ["D", "B", "C", "A"]
    fade = simulate_fade(open_, close, vol, vwap, adr, trd, mod, offs, syms,
                         FadeParams(k=1.3, notional=1e5))
    assert (fade["side"] == 1).all() and len(fade) > 20
    inp = dict(open_=open_, high=close + 1, low=close - 1, close=close, volume=vol, vwap=vwap,
               idx_dev=np.zeros(n), beta_day=np.zeros((4, m)), adr_day=adr,
               gate=np.ones(n, bool), tradable=trd, minute_of_day=mod, day_offsets=offs,
               symbols=syms)
    ours = _run_module(inp, 1.3, 1e5, "market", 0.05)
    cols = ["symbol", "day_idx", "signal_row", "entry_row", "entry_price", "exit_row",
            "exit_price", "exit_reason", "gross_pnl"]
    pd.testing.assert_frame_equal(ours[cols].reset_index(drop=True),
                                  fade[cols].reset_index(drop=True), check_dtype=False)


def test_empty_result_has_exact_columns():
    inp = _flat_panel()
    inp["gate"][:] = False
    inp["low"][:, 0] = 100.0
    inp["close"][:, 0] = 100.0
    for mode in ("passive", "market"):
        tr = _sim(inp, mode)
        assert len(tr) == 0 and list(tr.columns) == COLUMNS


def test_output_sorted_by_entry_row_then_symbol_name():
    inp = _flat_panel(n_bars=10, first_minute=600)
    for key in ("open_", "high", "low", "close", "volume", "vwap", "tradable"):
        inp[key] = np.repeat(inp[key], 3, axis=1)
    for key in ("beta_day", "adr_day"):
        inp[key] = np.repeat(inp[key], 3, axis=1)
    inp["symbols"] = ["ZZ", "AA", "MM"]
    inp["low"][4, :] = 125.0
    tr = _sim(inp, "passive")
    assert list(tr["symbol"]) == ["AA", "MM", "ZZ"]


# ----------------------------------------------------------------------------------------
# Costs (invariant 8)
# ----------------------------------------------------------------------------------------


def test_zero_cost_models_give_net_equal_gross():
    for seed in range(6):
        inp, k, _, _ = _rand_sim_panel(1000 + seed)
        for mode in ("passive", "market"):
            tr = _run_module(inp, k, 1e6, mode, _TICK)
            assert (tr["costs"] == 0).all()
            assert np.array_equal(tr["net_pnl"].to_numpy(), tr["gross_pnl"].to_numpy())
            assert np.array_equal(tr["net_bps"].to_numpy(), tr["gross_bps"].to_numpy())


@pytest.mark.parametrize("notional", [100_000.0, 1_000_000.0])
def test_cost_and_slippage_reconciliation(notional):
    """Recompute every trade's costs independently: charges on a buy then a sell fill;
    slippage on the exit fill only for passive entries, on both fills for market entries."""
    cm, sl = NSEIntradayEquityCosts(), SqrtImpactSlippage()
    seen = Counter()
    for seed in range(2000, 2012):
        inp, k, _, _ = _rand_sim_panel(seed)
        c = np.asarray(inp["close"], np.float64)
        v = np.asarray(inp["volume"], np.float64)
        for mode in ("passive", "market"):
            tr = _run_module(inp, k, notional, mode, _TICK, cost_model=cm, slippage=sl)
            for t in tr.itertuples():
                s = inp["symbols"].index(t.symbol)
                qty = notional / t.entry_price
                assert t.qty == pytest.approx(qty, rel=1e-12)
                en, xn = qty * t.entry_price, qty * t.exit_price
                ch = cm.charges(FillBatch(notional=np.array([en, xn]),
                                          is_buy=np.array([True, False]))).total.sum()
                slip_x = xn * sl.bps(np.array([xn]),
                                     np.array([c[t.exit_row, s] * v[t.exit_row, s]]))[0] / 1e4
                slip_e = en * sl.bps(np.array([en]),
                                     np.array([c[t.entry_row, s] * v[t.entry_row, s]]))[0] / 1e4
                assert slip_e > 0 and slip_x > 0
                want = ch + slip_x + (slip_e if mode == "market" else 0.0)
                assert t.costs == pytest.approx(want, rel=1e-12), (mode, seed)
                # Discriminating: the wrong slippage treatment is measurably different.
                wrong = ch + slip_x + (0.0 if mode == "market" else slip_e)
                assert not t.costs == pytest.approx(wrong, rel=1e-9)
                assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs, rel=1e-12, abs=1e-9)
                assert t.net_bps == pytest.approx(t.net_pnl / notional * 1e4, rel=1e-12,
                                                  abs=1e-9)
                seen[mode] += 1
    assert seen["passive"] > 5 and seen["market"] > 5


# ----------------------------------------------------------------------------------------
# LiqParams and shape validation
# ----------------------------------------------------------------------------------------


def test_liqparams_defaults_and_frozen():
    p = S.LiqParams(k=1.0, notional=1e5, mode="passive")
    assert p.tick == 0.05 and p.start_minute == 556 and p.square_off_minute == 920
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.k = 2.0  # type: ignore[misc]
    S.LiqParams(k=1.0, notional=1e5, mode="market")


@pytest.mark.parametrize("kw", [
    dict(k=0.0), dict(k=-1.0), dict(k=NAN), dict(k=float("inf")),
    dict(notional=0.0), dict(notional=-5.0), dict(notional=NAN), dict(notional=float("inf")),
    dict(mode="limit"), dict(mode="Passive"), dict(mode=""),
    dict(tick=0.0), dict(tick=-0.05), dict(tick=NAN), dict(tick=float("inf")),
    dict(square_off_minute=556), dict(start_minute=920, square_off_minute=900),
])
def test_liqparams_validation(kw):
    base = dict(k=1.0, notional=1e5, mode="passive")
    base.update(kw)
    with pytest.raises(ValueError):
        S.LiqParams(**base)


@pytest.mark.parametrize("which", ["open_", "high", "low", "close", "volume", "vwap",
                                   "tradable", "idx_dev", "gate", "minute_of_day", "beta_day",
                                   "adr_day", "symbols"])
def test_simulate_shape_mismatch_raises(which):
    inp = _flat_panel(sessions=[(600, 6), (600, 6)])
    bad = dict(inp)
    arr = inp[which]
    if which == "symbols":
        bad[which] = ["AAA", "BBB"]
    elif which in ("idx_dev", "gate", "minute_of_day"):
        bad[which] = arr[:-1]
    elif which in ("beta_day", "adr_day"):
        bad[which] = arr[:1]
    else:
        bad[which] = arr[:-1]
    with pytest.raises(ValueError):
        _sim(bad, "passive")
    if which in ("open_", "close", "vwap"):
        bad[which] = np.repeat(arr, 2, axis=1)  # wrong n_sym
        with pytest.raises(ValueError):
            _sim(bad, "market")


# ----------------------------------------------------------------------------------------
# Features: randomized comparison to the references
# ----------------------------------------------------------------------------------------

_FEAT_TEMPLATES = [(555, 60), (555, 105), (860, 70), (900, 30), (555, 20), (1095, 60)]


def _rand_feature_panel(seed, n_days=None, n_sym=3):
    rng = np.random.default_rng(seed)
    n_days = n_days or int(rng.integers(6, 10))
    sessions = [_FEAT_TEMPLATES[i] for i in rng.choice(len(_FEAT_TEMPLATES), n_days)]
    if seed % 4 == 0:
        sessions[int(rng.integers(n_days))] = (555, 375)
    mod, offs = _calendar(sessions)
    n = len(mod)
    idx = 20000 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    beta_true = rng.uniform(0.3, 1.7, n_sym)
    lr = np.diff(np.log(idx), prepend=np.log(idx[0]))
    stock = 500 * np.exp(np.cumsum(beta_true * lr[:, None] + rng.normal(0, 6e-4, (n, n_sym)), 0))
    ih = idx * (1 + np.abs(rng.normal(0, 2e-4, n)))
    il = idx * (1 - np.abs(rng.normal(0, 2e-4, n)))
    ic = idx.copy()
    ic[rng.random(n) < 0.05] = NAN
    ih[rng.random(n) < 0.03] = NAN  # index bar present (finite close) but high missing
    stock[rng.random((n, n_sym)) < 0.06] = NAN
    stock[rng.random((n, n_sym)) < 0.004] = 0.0  # non-positive close: never in a pair
    vix = 15 * np.exp(np.cumsum(rng.normal(0, 3e-3, n)))
    vix[rng.random(n) < 0.08] = NAN
    if n_days > 3:  # one whole session without VIX
        dd = int(rng.integers(1, n_days - 1))
        vix[offs[dd]:offs[dd + 1]] = NAN
    usable = rng.random(n_days) < 0.8
    return dict(rng=rng, mod=mod, offs=offs, idx_h=ih, idx_l=il, idx_c=ic, stock=stock,
                vix=vix, usable=usable, n_days=n_days)


def _close(a, b, rtol=1e-12):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=rtol,
                               atol=0, equal_nan=True)
    assert np.array_equal(np.isnan(np.asarray(a, np.float64)), np.isnan(np.asarray(b)))


def test_index_twap_random():
    nonnan = 0
    for seed in range(15):
        p = _rand_feature_panel(seed)
        args = [p["idx_h"], p["idx_l"], p["idx_c"]]
        if seed % 2:
            args = [a.astype(np.float32) for a in args]
        got = F.index_twap(*args, p["mod"], p["offs"])
        assert got.dtype == np.float64 and got.shape == p["idx_c"].shape
        exp = ref_index_twap(*args, p["mod"], p["offs"])
        _close(got, exp, rtol=1e-12)
        nonnan += np.isfinite(got).sum()
    assert nonnan > 100


def test_index_twap_resets_excludes_0915_and_never_fills():
    """Invariant 1 on a hand panel of 60 / 105 / 375-bar sessions."""
    mod, offs = _calendar([(555, 60), (555, 105), (555, 375)])
    n = len(mod)
    c = np.arange(n, dtype=np.float64) + 1000.0
    h, lo = c + 3.0, c - 3.0  # tp == c
    c[offs[1]] = 1e9  # 09:15 bar of session 1: must never contribute
    h[offs[1]], lo[offs[1]] = 1e9, 1e9
    c[offs[2] + 10] = NAN  # absent index bar
    got = F.index_twap(h, lo, c, mod, offs)
    for d in range(3):
        a = offs[d]
        assert np.isnan(got[a])  # 09:15 row
        assert got[a + 1] == pytest.approx(c[a + 1], rel=1e-15)  # reset
    assert np.isnan(got[offs[2] + 10])
    r = offs[2] + 11
    vals = [c[t] for t in range(offs[2] + 1, r + 1) if np.isfinite(c[t])]
    assert got[r] == pytest.approx(np.mean(vals), rel=1e-13)
    r = offs[1] + 50
    assert got[r] == pytest.approx(np.mean(c[offs[1] + 1:r + 1]), rel=1e-13)


def test_index_dev_random_and_edges():
    rng = np.random.default_rng(3)
    i = rng.uniform(100, 200, 300)
    w = rng.uniform(100, 200, 300)
    i[rng.random(300) < 0.1] = NAN
    w[rng.random(300) < 0.1] = NAN
    w[rng.random(300) < 0.05] = 0.0
    w[rng.random(300) < 0.05] = -3.0
    w[:3] = [np.inf, 0.0, -1.0]
    got = F.index_dev(i.astype(np.float32), w)
    _close(got, ref_index_dev(i.astype(np.float32), w))
    assert np.isnan(got[:3]).all()


def test_prior_session_beta_random():
    finite = 0
    for seed in range(12):
        p = _rand_feature_panel(seed)
        lookback = 2 + seed % 3
        sc, ic = p["stock"], p["idx_c"]
        if seed % 3 == 0:
            sc, ic = sc.astype(np.float32), ic.astype(np.float32)
        got = F.prior_session_beta(sc, ic, p["mod"], p["offs"], p["usable"], lookback=lookback)
        assert got.shape == (p["n_days"], sc.shape[1]) and got.dtype == np.float64
        exp = ref_prior_session_beta(sc, ic, p["mod"], p["offs"], p["usable"], lookback)
        _close(got, exp, rtol=1e-9)
        finite += np.isfinite(got).sum()
    assert finite > 20


def _pairs_by_hand(sc, ic, mod, offs, sessions, s):
    """Independent construction: within-session diffs of logs, dropping any diff whose
    endpoints are not both present, or whose first row is the 09:15 bar."""
    xs, ys = [], []
    for d in sessions:
        a, b = offs[d], offs[d + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            li = np.log(ic[a:b])
            ls = np.log(sc[a:b, s])
        dx, dy = np.diff(li), np.diff(ls)
        ok = np.isfinite(dx) & np.isfinite(dy) & (mod[a:b - 1] >= START)
        xs.append(dx[ok])
        ys.append(dy[ok])
    return np.concatenate(xs), np.concatenate(ys)


def test_prior_session_beta_equals_polyfit_skips_unusable_and_never_bridges():
    """Invariant 2."""
    rng = np.random.default_rng(77)
    mod, offs = _calendar([(555, 60), (555, 105), (555, 60), (555, 375), (555, 60), (555, 105),
                           (555, 60)])
    n = len(mod)
    ic = 20000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    sc = np.column_stack([
        300 * np.exp(np.cumsum(0.8 * np.diff(np.log(ic), prepend=np.log(ic[0]))
                               + rng.normal(0, 5e-4, n))),
        900 * np.exp(np.cumsum(rng.normal(0, 5e-4, n))),
    ])
    # Poison candidates that a wrong implementation would pick up:
    for d in range(1, 7):
        sc[offs[d], 0] *= 1.3  # 09:15 jump: 09:15->09:16 excluded, overnight excluded
    sc[offs[3] + 40, 0] = NAN  # a hole: rows 39->41 must not be bridged
    sc[offs[3] + 41, 0] = sc[offs[3] + 39, 0] * 1.2
    ic[offs[5] + 7] = NAN
    usable = np.array([True, True, False, True, True, True, True])
    got = F.prior_session_beta(sc, ic, mod, offs, usable, lookback=3)
    # d = 5: usable prior sessions [0, 1, 3, 4] -> last 3 = [1, 3, 4]
    for s in range(2):
        x, y = _pairs_by_hand(sc, ic, mod, offs, [1, 3, 4], s)
        assert got[5, s] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-8)
        x2, y2 = _pairs_by_hand(sc, ic, mod, offs, [2, 3, 4], s)  # counting unusable session
        assert abs(np.polyfit(x2, y2, 1)[0] - got[5, s]) > 1e-6
    # d = 6: prior usable [0,1,3,4,5] -> [3,4,5]; session 5 has an index hole.
    x, y = _pairs_by_hand(sc, ic, mod, offs, [3, 4, 5], 0)
    assert got[6, 0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-8)
    # A bridged pair across the stock hole would be an outlier; show it matters.
    xb = np.append(x, math.log(ic[offs[3] + 41] / ic[offs[3] + 39]))
    yb = np.append(y, math.log(sc[offs[3] + 41, 0] / sc[offs[3] + 39, 0]))
    assert abs(np.polyfit(xb, yb, 1)[0] - got[6, 0]) > 1e-4
    # Fewer than `lookback` usable prior sessions -> NaN; d=3 has only [0, 1].
    assert np.isnan(got[:4]).all()
    assert np.isfinite(got[4:]).all()


def test_prior_session_beta_nan_rules_and_validation():
    mod, offs = _calendar([(555, 20)] * 5)
    n = len(mod)
    rng = np.random.default_rng(5)
    ic = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    sc = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (n, 2)), 0))
    sc[offs[2]:offs[3], 1] = NAN  # symbol 1 has zero pairs in session 2
    usable = np.ones(5, bool)
    got = F.prior_session_beta(sc, ic, mod, offs, usable, lookback=2)
    assert np.isnan(got[:2]).all()
    assert np.isfinite(got[2:, 0]).all()
    assert np.isfinite(got[2, 1]) and np.isnan(got[3, 1]) and np.isnan(got[4, 1])
    # One lone present bar in session 2 still yields zero pairs.
    sc2 = sc.copy()
    sc2[offs[2] + 5, 1] = 100.0
    assert np.isnan(F.prior_session_beta(sc2, ic, mod, offs, usable, lookback=2)[3, 1])
    # Constant index -> zero denominator -> NaN.
    flat = np.full(n, 100.0)
    assert np.isnan(F.prior_session_beta(sc, flat, mod, offs, usable, lookback=2)).all()
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            F.prior_session_beta(sc, ic, mod, offs, usable, lookback=bad)


def test_prior_session_beta_is_causal():
    for seed in range(8):
        p = _rand_feature_panel(100 + seed, n_days=9)
        offs, lb = p["offs"], 2
        base = F.prior_session_beta(p["stock"], p["idx_c"], p["mod"], offs, p["usable"], lb)
        rng = np.random.default_rng(seed)
        changed_later = False
        for d in range(1, 8):
            sc, ic, us = p["stock"].copy(), p["idx_c"].copy(), p["usable"].copy()
            r0 = offs[d]
            sc[r0:] = sc[r0:] * np.exp(rng.normal(0, 0.01, sc[r0:].shape))
            sc[r0:][rng.random(sc[r0:].shape) < 0.2] = NAN
            ic[r0:] = ic[r0:] * np.exp(rng.normal(0, 0.01, ic[r0:].shape))
            us[d:] = ~us[d:]
            pert = F.prior_session_beta(sc, ic, p["mod"], offs, us, lb)
            _close(pert[:d + 1], base[:d + 1], rtol=1e-12)
            changed_later |= not np.allclose(pert[d + 1:], base[d + 1:], equal_nan=True)
        assert changed_later


def test_residual_stretch_random():
    rng = np.random.default_rng(9)
    mod, offs = _calendar([(555, 60), (860, 70), (555, 105)])
    n, m = len(mod), 3
    close = rng.uniform(90, 110, (n, m))
    vwap = rng.uniform(90, 110, (n, m))
    idv = rng.normal(0, 0.01, n)
    beta = rng.uniform(-0.5, 2, (3, m))
    adr = rng.uniform(0.005, 0.03, (3, m))
    close[rng.random((n, m)) < 0.1] = NAN
    vwap[rng.random((n, m)) < 0.05] = NAN
    vwap[rng.random((n, m)) < 0.03] = 0.0
    vwap[rng.random((n, m)) < 0.03] = -10.0
    idv[rng.random(n) < 0.05] = NAN
    beta[0, 1] = NAN
    adr[1, 2] = 0.0
    adr[2, 0] = -0.01
    adr[2, 1] = np.inf
    got = F.residual_stretch(close.astype(np.float32), vwap, idv, beta, adr, offs)
    exp = ref_residual_stretch(close.astype(np.float32), vwap, idv, beta, adr, offs)
    assert got.shape == (n, m) and got.dtype == np.float64
    _close(got, exp, rtol=1e-12)
    assert np.isnan(got[offs[0]:offs[1], 1]).all()
    assert np.isnan(got[offs[1]:offs[2], 2]).all()
    assert np.isfinite(got).sum() > n


def test_band_equivalence_with_residual_stretch():
    """Spec note: close <= band  <=>  residual_stretch <= -k (checked away from the boundary)."""
    rng = np.random.default_rng(10)
    mod, offs = _calendar([(555, 105), (860, 70)])
    n, m = len(mod), 3
    close, vwap = rng.uniform(95, 105, (n, m)), rng.uniform(95, 105, (n, m))
    idv, beta = rng.normal(0, 0.01, n), rng.uniform(0, 2, (2, m))
    adr = rng.uniform(0.005, 0.03, (2, m))
    k = 1.1
    z = F.residual_stretch(close, vwap, idv, beta, adr, offs)
    band = _band_array(vwap, idv, beta, adr, offs, k)
    away = np.abs(z + k) > 1e-9
    assert np.array_equal((close <= band)[away], (z <= -k)[away])
    assert (close <= band).any() and (close > band).any()


def test_vix_change_random_and_previous_session_only():
    for seed in range(12):
        p = _rand_feature_panel(200 + seed)
        v = p["vix"].astype(np.float32) if seed % 2 else p["vix"]
        got = F.vix_change(v, p["mod"], p["offs"])
        assert got.dtype == np.float64
        _close(got, ref_vix_change(v, p["mod"], p["offs"]), rtol=1e-12)
    # Invariant 4, by hand.
    mod, offs = _calendar([(555, 10), (555, 10), (555, 10), (555, 10)])
    v = np.arange(40, dtype=np.float64) + 10.0
    v[offs[1]:offs[2]] = NAN  # session 1 has no VIX at all
    v[offs[2] + 9] = NAN  # session 2's last row missing -> its last finite close is row 8
    got = F.vix_change(v, mod, offs)
    assert np.isnan(got[:offs[1]]).all()  # session 0
    assert np.isnan(got[offs[2]:offs[3]]).all()  # prev session (1) empty; never looks at 0
    assert np.isnan(got[offs[3]])  # 09:15 row
    prev = v[offs[2] + 8]
    np.testing.assert_allclose(got[offs[3] + 1:offs[4]], v[offs[3] + 1:offs[4]] / prev - 1,
                               rtol=1e-15)


def test_vix_change_prev_may_be_any_minute_and_must_be_positive():
    mod, offs = _calendar([(1000, 3), (555, 5), (555, 5)])  # session 0 entirely after 15:20
    v = np.array([10.0, 12.0, 16.0, 17.0, 18.0, 19.0, 20.0, 0.0, 30, 31, 32, 33, 34])
    got = F.vix_change(v, mod, offs)
    np.testing.assert_allclose(got[4:8], v[4:8] / 16.0 - 1)
    assert np.isnan(got[3])
    assert np.isnan(got[8:]).all()  # prev close <= 0


def test_session_extreme_random_1d_2d_and_validation():
    for seed in range(8):
        p = _rand_feature_panel(300 + seed)
        x2 = p["stock"] / 500 - 1
        x1 = p["vix"]
        for how in ("max", "min"):
            g2 = F.session_extreme(x2, p["mod"], p["offs"], how)
            g1 = F.session_extreme(x1.astype(np.float32), p["mod"], p["offs"], how)
            assert g2.shape == (p["n_days"], 3) and g1.shape == (p["n_days"],)
            _close(g2, ref_session_extreme(x2, p["mod"], p["offs"], how))
            _close(g1, ref_session_extreme(x1.astype(np.float32), p["mod"], p["offs"], how))
    mod, offs = _calendar([(555, 3), (918, 4), (1000, 2)])
    x = np.array([100.0, 1.0, 2.0, 5.0, -1.0, -50.0, 50.0, 7.0, 8.0])
    assert np.array_equal(F.session_extreme(x, mod, offs, "max")[:2], [2.0, 5.0])
    assert np.array_equal(F.session_extreme(x, mod, offs, "min")[:2], [1.0, -1.0])
    assert np.isnan(F.session_extreme(x, mod, offs, "max")[2])
    for bad in ("mean", "MAX", ""):
        with pytest.raises(ValueError):
            F.session_extreme(x, mod, offs, bad)


def test_rolling_session_quantile_random_causal_and_validation():
    rng = np.random.default_rng(11)
    for trial in range(10):
        nd = int(rng.integers(5, 60))
        ps = rng.normal(size=nd)
        ps[rng.random(nd) < 0.15] = NAN
        ps[rng.random(nd) < 0.05] = np.inf
        us = rng.random(nd) < 0.8
        q = float(rng.uniform(0.01, 0.99))
        mn = int(rng.integers(1, 6))
        w = mn + int(rng.integers(0, 10))
        got = F.rolling_session_quantile(ps, us, q, window=w, min_sessions=mn)
        assert got.shape == (nd,) and got.dtype == np.float64
        _close(got, ref_rolling_quantile(ps, us, q, w, mn), rtol=1e-13)
        # Invariant 3: out[d] independent of per_session[d:] and usable[d:].
        for d in range(0, nd, 3):
            ps2, us2 = ps.copy(), us.copy()
            ps2[d:] = rng.normal(size=nd - d) * 100
            us2[d:] = ~us2[d:]
            got2 = F.rolling_session_quantile(ps2, us2, q, window=w, min_sessions=mn)
            _close(got2[:d + 1], got[:d + 1], rtol=0)
    # Defaults: window 250, min_sessions 120.
    ps = np.arange(400, dtype=np.float64)
    got = F.rolling_session_quantile(ps, np.ones(400, bool), 0.9)
    assert np.isnan(got[:120]).all() and np.isfinite(got[120:]).all()
    assert got[120] == np.quantile(ps[:120], 0.9)
    assert got[399] == np.quantile(ps[149:399], 0.9)
    for kw in (dict(q=0.0), dict(q=1.0), dict(q=-0.1), dict(q=1.5),
               dict(q=0.5, window=5, min_sessions=6), dict(q=0.5, window=5, min_sessions=0)):
        with pytest.raises(ValueError):
            F.rolling_session_quantile(ps, np.ones(400, bool), **kw)


def test_stress_gate_random():
    rng = np.random.default_rng(12)
    mod, offs = _calendar([(555, 60), (555, 105), (860, 70), (555, 20)])
    n = len(mod)
    for _ in range(5):
        vc = np.round(rng.normal(0, 0.02, n), 2)
        ida = np.round(rng.normal(0, 0.5, n), 1)
        vd = np.round(rng.normal(0.01, 0.01, 4), 2)
        md = np.round(rng.normal(-0.3, 0.2, 4), 1)
        vc[rng.random(n) < 0.1] = NAN
        ida[rng.random(n) < 0.1] = np.inf
        vd[rng.integers(4)] = NAN
        got = F.stress_gate(vc, ida, vd, md, offs)
        exp = ref_stress_gate(vc, ida, vd, md, offs)
        assert got.dtype == bool and got.shape == (n,)
        assert np.array_equal(got, exp)
        assert got.any() and (~got).any()
    # Inclusive comparisons.
    offs1 = np.array([0, 2])
    g = F.stress_gate(np.array([0.05, 0.05]), np.array([-1.0, -0.99]),
                      np.array([0.05]), np.array([-1.0]), offs1)
    assert list(g) == [True, False]


# ----------------------------------------------------------------------------------------
# evaluate.py (invariant 9) -- hand-built trade frames
# ----------------------------------------------------------------------------------------


def _trades(rows):
    """rows: list of (day_idx, net_pnl). Notional 1e6 -> net_bps = net_pnl / 100."""
    out = []
    for i, (d, pnl) in enumerate(rows):
        out.append(dict(symbol=f"S{i % 3}", day_idx=d, side=1, signal_row=10 * i,
                        entry_row=10 * i + 1, entry_price=100.0, exit_row=10 * i + 2,
                        exit_price=100.0, exit_reason="vwap", qty=1e4, gross_pnl=pnl + 50.0,
                        costs=50.0, net_pnl=float(pnl), gross_bps=(pnl + 50.0) / 100,
                        net_bps=pnl / 100, holding_bars=1))
    return pd.DataFrame(out, columns=COLUMNS)


_DATES = [dt.date(2018, 1, 2), dt.date(2019, 1, 2), dt.date(2020, 1, 2), dt.date(2021, 1, 4),
          dt.date(2022, 1, 3), dt.date(2023, 1, 2), dt.date(2023, 6, 1), dt.date(2023, 8, 1)]
_ROWS_PASS = [(0, 300), (0, 100), (1, 500), (2, 200), (2, 400), (3, 700), (4, 100), (4, 200),
              (5, 800), (6, 50), (6, 150), (7, 900), (7, 100)]
# net_bps per trade: 3 1 | 5 | 2 4 | 7 | 1 2 | 8 | .5 1.5 | 9 1 -> sum 45 over 13 trades.
# day means (bps): 2, 5, 3, 7, 1.5, 8, 1, 5. day totals: 400 500 600 700 300 800 200 1000.
# top-5 days by net_pnl: 7, 5, 3, 2, 1 -> remaining days 0, 4, 6 -> bps 3 1 1 2 .5 1.5 = 9/6.
_DAY_MEANS = np.array([2, 5, 3, 7, 1.5, 8, 1, 5], dtype=np.float64)
_T_PASS = _DAY_MEANS.mean() / (_DAY_MEANS.std(ddof=1) / math.sqrt(8))


def test_day_clustered_t_hand_values():
    tr = _trades(_ROWS_PASS)
    assert E.day_clustered_t(tr) == pytest.approx(_T_PASS, rel=1e-12)
    assert _T_PASS == pytest.approx(4.4242, abs=1e-3)
    # gross_bps = net_bps + 0.5 per trade -> day means shift by 0.5.
    g = _DAY_MEANS + 0.5
    assert E.day_clustered_t(tr, column="gross_bps") == pytest.approx(
        g.mean() / (g.std(ddof=1) / math.sqrt(8)), rel=1e-12)
    assert math.isnan(E.day_clustered_t(_trades([(3, 100), (3, 200)])))  # one day
    assert math.isnan(E.day_clustered_t(_trades([(1, 100), (2, 50), (2, 150)])))  # zero std
    assert math.isnan(E.day_clustered_t(_trades([])))


def test_day_clustered_t_weights_days_not_trades():
    # day 0: ten trades of +10 bps; day 1: one of -10; day 2: one of +2.
    tr = _trades([(0, 1000)] * 10 + [(1, -1000), (2, 200)])
    m = np.array([10.0, -10.0, 2.0])
    assert E.day_clustered_t(tr) == pytest.approx(m.mean() / (m.std(ddof=1) / math.sqrt(3)))


def test_yearly_net_hand_values():
    tr = _trades(_ROWS_PASS)
    got = E.yearly_net(tr, _DATES)
    assert got == {2018: 400.0, 2019: 500.0, 2020: 600.0, 2021: 700.0, 2022: 300.0,
                   2023: 2000.0}
    assert all(isinstance(k, (int, np.integer)) for k in got)
    tr2 = _trades([(1, -300), (1, 300), (3, -5)])
    assert E.yearly_net(tr2, _DATES) == {2019: 0.0, 2021: -5.0}


def test_mean_net_bps_without_top_days_hand_values():
    tr = _trades(_ROWS_PASS)
    assert E.mean_net_bps_without_top_days(tr) == pytest.approx(1.5, rel=1e-12)
    # n_top=1 drops day 7 only: remaining bps sum 45 - 10 = 35 over 11 trades.
    assert E.mean_net_bps_without_top_days(tr, n_top=1) == pytest.approx(35 / 11, rel=1e-12)
    # Ranked by total net_pnl, not by mean bps: day 0 has two trades totalling 400 > day 1's 350.
    tr2 = _trades([(0, 200), (0, 200), (1, 350), (2, -100)])
    assert E.mean_net_bps_without_top_days(tr2, n_top=1) == pytest.approx((3.5 - 1) / 2)
    assert math.isnan(E.mean_net_bps_without_top_days(_trades([(0, 1), (1, 2)]), n_top=5))


def test_kill_criteria_all_pass():
    tr = _trades(_ROWS_PASS)
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    kc = E.kill_criteria(tr, _DATES, years)
    assert kc["mean_net_bps"] == pytest.approx(45 / 13, rel=1e-12)
    assert kc["day_t"] == pytest.approx(_T_PASS, rel=1e-12)
    assert kc["positive_years"] == 6 and isinstance(kc["positive_years"], (int, np.integer))
    assert kc["n_years"] == 6
    assert kc["mean_net_bps_ex_top5"] == pytest.approx(1.5, rel=1e-12)
    for c in ("c1_mean_net_positive", "c2_day_t_gt_2", "c3_positive_years_ge_4",
              "c4_ex_top5_positive", "passes"):
        assert kc[c] is True or kc[c] == np.True_, c


def test_kill_criteria_failures_and_years_without_trades():
    tr = _trades(_ROWS_PASS)
    # Years with no trades count as not positive: only 2018 and 2019 have trades here.
    kc = E.kill_criteria(tr, _DATES, [2014, 2015, 2016, 2017, 2018, 2019])
    assert kc["positive_years"] == 2 and kc["n_years"] == 6
    assert not kc["c3_positive_years_ge_4"] and not kc["passes"]
    assert kc["c1_mean_net_positive"] and kc["c2_day_t_gt_2"] and kc["c4_ex_top5_positive"]
    # Ex-top-5 failure: the remaining days lose.
    rows = [(0, -300), (1, 900), (2, 800), (3, 700), (4, 600), (5, 500), (6, -100)]
    kc = E.kill_criteria(_trades(rows), _DATES, [2018, 2019, 2020, 2021, 2022, 2023])
    assert kc["mean_net_bps_ex_top5"] == pytest.approx(-2.0)
    assert kc["mean_net_bps"] == pytest.approx(31 / 7)
    m = np.array([-3, 9, 8, 7, 6, 5, -1], dtype=float)
    assert kc["day_t"] == pytest.approx(m.mean() / (m.std(ddof=1) / math.sqrt(7)))
    assert kc["positive_years"] == 5  # 2018 is -300; 2023 is 500-100
    assert kc["c1_mean_net_positive"] and kc["c3_positive_years_ge_4"]
    assert not kc["c4_ex_top5_positive"] and not kc["passes"]
    assert kc["c2_day_t_gt_2"] == (kc["day_t"] > 2)


def test_kill_criteria_nan_never_passes():
    kc = E.kill_criteria(_trades([(0, 500), (0, 700)]), _DATES, [2018])
    assert math.isnan(kc["day_t"]) and math.isnan(kc["mean_net_bps_ex_top5"])
    assert not kc["c2_day_t_gt_2"] and not kc["c4_ex_top5_positive"] and not kc["passes"]
    kc = E.kill_criteria(_trades([]), _DATES, [2018, 2019])
    assert math.isnan(kc["mean_net_bps"]) and kc["positive_years"] == 0
    assert kc["n_years"] == 2 and not kc["passes"] and not kc["c1_mean_net_positive"]


# ----------------------------------------------------------------------------------------
# Invariant 10: null behaviour (CLAUDE.md rule 9) -- a RATE over >= 30 seeds
# ----------------------------------------------------------------------------------------

# Seeds are fixed and are never changed to make this pass. Measured against this suite's own
# reference simulator (2026-09-26): |t| > 2.58 on 1/40 seeds in each mode; per-seed t has
# mean -0.45 (passive: a limit fill that needs a one-tick trade-through is adversely selected
# by about a tick, a real effect kept small here by a 2000-rupee price) / -0.34 (market) and
# sd ~1.2, so the 0.2 bound (8/40) is far from the expected ~4% rate.
_NULL_SEEDS = range(40)
_NULL_T = 2.58
_NULL_MAX_RATE = 0.2


def _null_panel(seed):
    """Driftless arithmetic random walks (martingale prices, so any stopping-time entry has
    zero expected gross P&L), OHLC from an 8-step intrabar path, random-walk index, random
    gate, every bar tradable."""
    rng = np.random.default_rng(seed)
    sessions = [[(555, 375), (835, 105), (555, 60)][j % 3] for j in range(24)]
    mod, offs = _calendar(sessions)
    n, m, sub = len(mod), 6, 8

    def ohlc(p0, step_sd, beta_steps=None):
        steps = rng.normal(0, step_sd, (n, sub) + p0.shape)
        if beta_steps is not None:
            steps = steps + beta_steps
        path = p0 + np.cumsum(steps.reshape((n * sub,) + p0.shape), 0).reshape(steps.shape)
        return path[:, 0], path.max(1), path.min(1), path[:, -1], steps

    io, ih, il, ic, isteps = ohlc(np.array([20000.0]), 3.0)
    betas = rng.uniform(0.5, 1.5, m)
    so, sh, sl, sc, _ = ohlc(np.full(m, 2000.0), 0.4, isteps * (betas * 2000 / 20000))
    io, ih, il, ic = io[:, 0], ih[:, 0], il[:, 0], ic[:, 0]
    vol = np.exp(rng.normal(8, 1, (n, m)))
    return dict(mod=mod, offs=offs, io=io, ih=ih, il=il, ic=ic, so=so, sh=sh, sl=sl, sc=sc,
                vol=vol, gate=rng.random(n) < 0.5)


@pytest.mark.parametrize("mode", ["passive", "market"])
def test_null_false_positive_rate(mode):
    flagged, finite, n_trades = 0, 0, []
    for seed in _NULL_SEEDS:
        p = _null_panel(seed)
        mod, offs = p["mod"], p["offs"]
        n_days = len(offs) - 1
        vwap = VF.session_vwap(p["sh"], p["sl"], p["sc"], p["vol"], mod, offs)
        adr = VF.adr_pct(VF.session_range_pct(p["sh"], p["sl"], p["sc"], mod, offs), 5)
        twap = F.index_twap(p["ih"], p["il"], p["ic"], mod, offs)
        idev = F.index_dev(p["ic"], twap)
        beta = F.prior_session_beta(p["sc"], p["ic"], mod, offs, np.ones(n_days, bool), 5)
        # k is measured, not chosen: the median per-(symbol, session) minimum residual stretch
        # of this very panel (the same construction the prereg uses for its k grid).
        z = F.residual_stretch(p["sc"], vwap, idev, beta, adr, offs)
        k = -float(np.nanquantile(F.session_extreme(z, mod, offs, "min"), 0.5))
        assert k > 0
        tr = S.simulate_liqprov(p["so"], p["sh"], p["sl"], p["sc"], p["vol"], vwap, idev, beta,
                                adr, p["gate"], np.ones_like(p["sc"], bool), mod, offs,
                                [f"S{i}" for i in range(6)],
                                S.LiqParams(k=k, notional=1e6, mode=mode))
        n_trades.append(len(tr))
        t = E.day_clustered_t(tr, column="gross_bps")
        if math.isfinite(t):
            finite += 1
            flagged += abs(t) > _NULL_T
    assert len(_NULL_SEEDS) >= 30
    assert finite >= 0.9 * len(_NULL_SEEDS), "null panels too thin to be informative"
    assert np.median(n_trades) >= 50, n_trades
    rate = flagged / len(_NULL_SEEDS)
    assert rate <= _NULL_MAX_RATE, f"{mode}: {flagged}/{len(_NULL_SEEDS)} seeds |t| > 2.58"
