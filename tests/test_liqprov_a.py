"""Independent test suite "A" for ``nifty_quant.research.liqprov`` (spec: specs/liqprov_fade.md).

Written from the spec alone, before any implementation exists (CLAUDE.md rule 1). Every
expected value is computed by hand (or by an explicit loop in this file) -- never by calling
another function under test.

SPEC READINGS / AMBIGUITIES (how this suite interprets them):
* Passive trade-through: ``low[t] <= L - tick`` is tested with a dyadic tick (0.25) and a
  dyadic band (70.0) so ``L - tick`` is exact in binary and the edge is unambiguous. With the
  default tick 0.05 the edge would depend on float rounding of ``L - tick`` vs ``L - low``.
* ``gate`` is 1-D ``(n_rows,)`` (market-wide), per the shape clause of 2.2.
* ``signal_row`` is ``t-1`` for passive and ``r`` for market; both give entry_row - signal_row
  == 1.
* Exit rule 2 in passive mode needs only a present bar and finite vwap (vwap_adr reading).
  A pending vwap exit that never finds a later fillable row ends as ``"no_bar"``.
* Passive re-entry "from row x+1": row x (the exit row) can never be a passive entry row;
  row x+1 can (order placed at x's close, using gate[x] and band[x]).
* ``kill_criteria``: ``positive_years`` counts years IN ``years`` whose ``yearly_net`` > 0.
  Hand examples keep every trade year inside ``years``.
* ``day_clustered_t`` / ``mean_net_bps_without_top_days`` with zero trades -> NaN.
* Invariant 10 is asserted per mode (passive and market), each on the same 30 fixed seeds.
  ``k`` is derived from the measured distribution of per-(symbol, session) minimum residual
  stretch on each panel (its 10% quantile, as in spec 5.1), not hand-chosen.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from nifty_quant.execution.costs import FillBatch, NSEIntradayEquityCosts
from nifty_quant.execution.fills import SqrtImpactSlippage
from nifty_quant.research.liqprov import evaluate as le
from nifty_quant.research.liqprov import features as lf
from nifty_quant.research.liqprov import simulate as ls
from nifty_quant.research.vwap_adr import features as vf

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

NAN = np.nan


def _offs(lengths):
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def _eq_nan(a, b, **kw):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True, **kw
    )


# ======================================================================================
# 1.1 index_twap
# ======================================================================================


def test_index_twap_hand_values_reset_and_0915_excluded():
    # session 0: minutes 555..558; session 1: 555..557
    minute = np.array([555, 556, 557, 558, 555, 556, 557])
    offs = _offs([4, 3])
    high = np.array([50.0, 12.0, 15.0, 9.0, 99.0, 30.0, 33.0])
    low = np.array([40.0, 6.0, 9.0, 3.0, 1.0, 24.0, 27.0])
    close = np.array([45.0, 9.0, 12.0, 6.0, 50.0, 27.0, 30.0])
    tp = (high + low + close) / 3.0  # 9, 12, 6 in s0 ; 27, 30 in s1
    out = lf.index_twap(high, low, close, minute, offs)
    expected = [
        NAN,
        tp[1],
        (tp[1] + tp[2]) / 2,
        (tp[1] + tp[2] + tp[3]) / 3,
        NAN,
        tp[5],
        (tp[5] + tp[6]) / 2,
    ]
    _eq_nan(out, expected, rtol=1e-14)
    assert out.shape == (7,)
    assert out.dtype == np.float64


def test_index_twap_absent_and_noncontributing_rows_no_fill():
    minute = np.array([556, 557, 558, 559, 560])
    offs = _offs([5])
    high = np.array([12.0, 14.0, NAN, 20.0, 22.0])
    low = np.array([6.0, 8.0, 10.0, 14.0, 16.0])
    close = np.array([9.0, NAN, 12.0, 17.0, 19.0])
    # row 1: close NaN -> absent -> NaN, does not contribute.
    # row 2: close finite (present) but high NaN -> does not contribute; twap = mean of row 0.
    out = lf.index_twap(high, low, close, minute, offs)
    tp0 = 9.0
    tp3 = 17.0
    tp4 = 19.0
    expected = [tp0, NAN, tp0, (tp0 + tp3) / 2, (tp0 + tp3 + tp4) / 3]
    _eq_nan(out, expected, rtol=1e-14)


def test_index_twap_no_contributing_rows_yet_is_nan_and_start_minute_param():
    minute = np.array([556, 557, 558])
    offs = _offs([3])
    h = np.array([3.0, 6.0, 9.0])
    out = lf.index_twap(h, h, h, minute, offs, start_minute=557)
    _eq_nan(out, [NAN, 6.0, 7.5])
    h2 = np.array([NAN, 6.0, 9.0])  # row 0: close finite (present) but high NaN
    c2 = np.array([3.0, 6.0, 9.0])
    out2 = lf.index_twap(h2, c2, c2, minute, offs)
    # row 0 present (close finite) but not contributing, nothing yet -> NaN
    _eq_nan(out2, [NAN, 6.0, 7.5])


def _ref_twap(h, lo, c, minute, offs, start=556):
    out = np.full(len(c), np.nan)
    for d in range(len(offs) - 1):
        acc = []
        for r in range(offs[d], offs[d + 1]):
            if minute[r] >= start and all(np.isfinite([h[r], lo[r], c[r]])):
                acc.append((h[r] + lo[r] + c[r]) / 3.0)
            if np.isfinite(c[r]) and minute[r] >= start and acc:
                out[r] = sum(acc) / len(acc)
    return out


def test_index_twap_mixed_session_lengths_invariant_1():
    rng = np.random.default_rng(1)
    lengths = [60, 105, 375, 60]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = _offs(lengths)
    n = int(offs[-1])
    c = 20000 + np.cumsum(rng.standard_normal(n))
    h = c + 1.0
    lo = c - 1.0
    c[rng.random(n) < 0.05] = NAN
    h[rng.random(n) < 0.02] = NAN
    out = lf.index_twap(
        h.astype(np.float32), lo.astype(np.float32), c.astype(np.float32), minute, offs
    )
    ref = _ref_twap(
        h.astype(np.float32).astype(float),
        lo.astype(np.float32).astype(float),
        c.astype(np.float32).astype(float),
        minute,
        offs,
    )
    assert out.dtype == np.float64
    _eq_nan(out, ref, rtol=1e-10)
    # 09:15 rows always NaN
    assert np.all(np.isnan(out[offs[:-1]]))


# ======================================================================================
# 1.2 prior_session_beta
# ======================================================================================


def _pairs(S, ix, rows_t):
    """Explicit list of pair rows t (pair uses t-1, t)."""
    xs, ys = [], []
    for t in rows_t:
        xs.append(math.log(ix[t] / ix[t - 1]))
        ys.append(math.log(S[t] / S[t - 1]))
    return xs, ys


def _layout_5x(n_days):
    # every session: minutes 555..559. Valid pair rows within a session (local): 2, 3, 4
    lengths = [5] * n_days
    minute = np.concatenate([555 + np.arange(5) for _ in lengths])
    return minute, _offs(lengths)


def _rand_prices(rng, n, n_sym):
    ix = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    S = 50 * np.exp(np.cumsum(rng.standard_normal((n, n_sym)) * 0.01, axis=0))
    return S, ix


def test_prior_session_beta_matches_polyfit_on_hand_pooled_pairs():
    rng = np.random.default_rng(7)
    n_days = 4
    minute, offs = _layout_5x(n_days)
    S, ix = _rand_prices(rng, 20, 2)
    usable = np.ones(n_days, dtype=bool)
    out = lf.prior_session_beta(S, ix, minute, offs, usable, lookback=2)
    assert out.shape == (n_days, 2)
    assert out.dtype == np.float64
    assert np.all(np.isnan(out[:2]))

    def pair_rows(days):
        return [offs[d] + k for d in days for k in (2, 3, 4)]

    for d, days in ((2, [0, 1]), (3, [1, 2])):
        for s in range(2):
            x, y = _pairs(S[:, s], ix, pair_rows(days))
            assert out[d, s] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)


def test_prior_session_beta_boundary_pair_not_bridged_when_session_starts_at_0916():
    # sessions start at 09:16, so local row 1 IS a valid pair (t-1 minute 556). The first
    # row of session d must not pair with the last row of session d-1.
    rng = np.random.default_rng(8)
    lengths = [4, 4, 4]
    minute = np.concatenate([556 + np.arange(n) for n in lengths])
    offs = _offs(lengths)
    S, ix = _rand_prices(rng, 12, 1)
    out = lf.prior_session_beta(S, ix, minute, offs, np.ones(3, bool), lookback=2)
    rows = [1, 2, 3, 5, 6, 7]
    x, y = _pairs(S[:, 0], ix, rows)
    assert out[2, 0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)


def test_prior_session_beta_unusable_sessions_skipped_not_counted():
    rng = np.random.default_rng(9)
    n_days = 5
    minute, offs = _layout_5x(n_days)
    S, ix = _rand_prices(rng, 25, 1)
    usable = np.array([True, False, True, True, True])
    out = lf.prior_session_beta(S, ix, minute, offs, usable, lookback=2)
    # d=2: only session 0 usable before it -> NaN
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0]) and np.isnan(out[2, 0])
    # d=3: sessions 0 and 2 (1 skipped)
    rows = [offs[d] + k for d in (0, 2) for k in (2, 3, 4)]
    x, y = _pairs(S[:, 0], ix, rows)
    assert out[3, 0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)
    # d=4: sessions 2 and 3
    rows = [offs[d] + k for d in (2, 3) for k in (2, 3, 4)]
    x, y = _pairs(S[:, 0], ix, rows)
    assert out[4, 0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)


def test_prior_session_beta_unusable_session_d_itself_irrelevant():
    rng = np.random.default_rng(10)
    minute, offs = _layout_5x(3)
    S, ix = _rand_prices(rng, 15, 1)
    a = lf.prior_session_beta(S, ix, minute, offs, np.array([True, True, True]), lookback=2)
    b = lf.prior_session_beta(S, ix, minute, offs, np.array([True, True, False]), lookback=2)
    assert a[2, 0] == pytest.approx(b[2, 0], rel=0, abs=0)
    assert np.isfinite(a[2, 0])


def test_prior_session_beta_gap_never_bridged():
    rng = np.random.default_rng(11)
    minute, offs = _layout_5x(3)
    S, ix = _rand_prices(rng, 15, 2)
    # symbol 0: stock bar missing at row 3 (session 0) -> pairs t=3 and t=4 vanish
    S[3, 0] = NAN
    # symbol 1: index missing at row 8 (session 1) -> pairs t=8 and t=9 vanish (for sym 1 AND 0)
    ix[8] = NAN
    # symbol 1 also: non-positive close at row 2 -> pairs t=2 and t=3 vanish
    S[2, 1] = 0.0
    out = lf.prior_session_beta(S, ix, minute, offs, np.ones(3, bool), lookback=2)
    x, y = _pairs(S[:, 0], ix, [2, 7])
    assert out[2, 0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)
    x, y = _pairs(S[:, 1], ix, [4, 7])
    assert out[2, 1] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-9)


def test_prior_session_beta_zero_pair_session_gives_nan_for_that_symbol_only():
    rng = np.random.default_rng(12)
    n_days = 5
    minute, offs = _layout_5x(n_days)
    S, ix = _rand_prices(rng, 25, 2)
    S[offs[1] : offs[2], 1] = NAN  # symbol 1 does not trade in session 1
    out = lf.prior_session_beta(S, ix, minute, offs, np.ones(n_days, bool), lookback=2)
    assert np.isnan(out[2, 1]) and np.isnan(out[3, 1])
    assert np.isfinite(out[4, 1])
    assert np.all(np.isfinite(out[2:, 0]))
    # single-bar session (no pairs) is also a zero-pair session
    S2 = S.copy()
    S2[offs[1] : offs[2], 1] = NAN
    S2[offs[1] + 3, 1] = 50.0  # one bar only -> no pair
    out2 = lf.prior_session_beta(S2, ix, minute, offs, np.ones(n_days, bool), lookback=2)
    assert np.isnan(out2[2, 1]) and np.isnan(out2[3, 1])


def test_prior_session_beta_nonpositive_denominator_nan():
    minute, offs = _layout_5x(3)
    rng = np.random.default_rng(13)
    S, _ = _rand_prices(rng, 15, 1)
    ix = np.full(15, 100.0)  # constant index -> sum x = sum x^2 = 0 -> denominator 0
    out = lf.prior_session_beta(S, ix, minute, offs, np.ones(3, bool), lookback=2)
    assert np.all(np.isnan(out))


def test_prior_session_beta_causal_invariant_2():
    rng = np.random.default_rng(14)
    lengths = [30, 60, 45, 30, 60, 30, 45, 30]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = _offs(lengths)
    n = int(offs[-1])
    S, ix = _rand_prices(rng, n, 3)
    usable = rng.random(len(lengths)) < 0.8
    usable[:4] = True
    base = lf.prior_session_beta(S, ix, minute, offs, usable, lookback=3)
    assert np.isfinite(base[4:]).any()
    for d in (3, 5, 7):
        S2, ix2, u2 = S.copy(), ix.copy(), usable.copy()
        S2[offs[d] :] *= np.exp(rng.standard_normal(S2[offs[d] :].shape))
        ix2[offs[d] :] *= np.exp(rng.standard_normal(ix2[offs[d] :].shape))
        u2[d:] = ~u2[d:]
        other = lf.prior_session_beta(S2, ix2, minute, offs, u2, lookback=3)
        np.testing.assert_array_equal(other[: d + 1], base[: d + 1])


def test_prior_session_beta_lookback_validation():
    minute, offs = _layout_5x(3)
    S = np.ones((15, 1))
    ix = np.ones(15)
    for lb in (1, 0, -3):
        with pytest.raises(ValueError):
            lf.prior_session_beta(S, ix, minute, offs, np.ones(3, bool), lookback=lb)


# ======================================================================================
# 1.3 index_dev / 1.4 residual_stretch
# ======================================================================================


def test_index_dev_values_and_nans():
    ix = np.array([101.0, 99.0, NAN, 100.0, 100.0, 100.0, np.inf])
    tw = np.array([100.0, 100.0, 100.0, NAN, 0.0, -5.0, 100.0])
    out = lf.index_dev(ix, tw)
    _eq_nan(out, [0.01, -0.01, NAN, NAN, NAN, NAN, NAN], rtol=1e-14)
    assert out.dtype == np.float64


def test_residual_stretch_hand_values_and_nans():
    offs = _offs([2, 2])
    close = np.array([[102.0, 50.0], [99.0, 50.0], [104.0, 50.0], [100.0, 50.0]])
    vwap = np.array([[100.0, 50.0], [100.0, 0.0], [100.0, 50.0], [NAN, 50.0]])
    idx = np.array([0.01, -0.02, 0.005, 0.0])
    beta = np.array([[2.0, 1.0], [0.5, np.inf]])
    adr = np.array([[0.02, 0.01], [0.04, 0.01]])
    out = lf.residual_stretch(close, vwap, idx, beta, adr, offs)
    e00 = ((102 / 100 - 1) - 2.0 * 0.01) / 0.02  # 0.0
    e10 = ((99 / 100 - 1) - 2.0 * -0.02) / 0.02  # 1.5
    e01 = (0.0 - 1.0 * 0.01) / 0.01  # -1
    e20 = ((104 / 100 - 1) - 0.5 * 0.005) / 0.04
    expected = [[e00, e01], [e10, NAN], [e20, NAN], [NAN, NAN]]
    _eq_nan(out, expected, rtol=1e-12, atol=1e-12)
    # adr <= 0 -> NaN
    adr2 = adr.copy()
    adr2[0, 0] = 0.0
    adr2[1, 0] = -0.01
    out2 = lf.residual_stretch(close, vwap, idx, beta, adr2, offs)
    assert np.all(np.isnan(out2[:, 0]))
    # idx_dev NaN -> NaN
    idx2 = idx.copy()
    idx2[0] = NAN
    assert np.isnan(lf.residual_stretch(close, vwap, idx2, beta, adr, offs)[0, 0])


# ======================================================================================
# 1.5 vix_change
# ======================================================================================


def test_vix_change_hand_values_invariant_4():
    # s0: 555..557 ; s1: 555..558 ; s2: 556..557 ; s3: 556..557
    minute = np.array([555, 556, 557, 555, 556, 557, 558, 556, 557, 556, 557])
    offs = _offs([3, 4, 2, 2])
    vix = np.array([10.0, 12.0, NAN, 30.0, 16.0, NAN, 18.0, NAN, NAN, 20.0, 22.0])
    out = lf.vix_change(vix, minute, offs)
    # s0 -> NaN everywhere. s1 prev = 12 (last FINITE close of s0; row 2 is NaN).
    # s1 row 3 minute 555 -> NaN; row 5 VIX absent -> NaN.
    # s2 prev = 18. both bars absent -> NaN. s3 prev = last finite of s2 = none -> NaN
    # (session 1's 18 must NOT be used: previous session only).
    expected = [NAN, NAN, NAN, NAN, 16 / 12 - 1, NAN, 18 / 12 - 1, NAN, NAN, NAN, NAN]
    _eq_nan(out, expected, rtol=1e-14)
    assert out.dtype == np.float64


def test_vix_change_prev_from_any_minute_and_nonpositive_prev():
    minute = np.array([555, 556, 557, 556, 557])
    offs = _offs([1, 2, 2])
    vix = np.array([10.0, 11.0, 0.0, 5.0, 6.0])
    out = lf.vix_change(vix, minute, offs)
    # s1 prev = 10 (a 09:15 bar still counts as prev); s2 prev = 0.0 -> NaN
    _eq_nan(out, [NAN, 0.1, -1.0, NAN, NAN], rtol=1e-14)
    # same session's earlier closes never used
    out2 = lf.vix_change(np.array([10.0, 11.0, 12.0]), np.array([556, 556, 557]), _offs([1, 2]))
    _eq_nan(out2, [NAN, 0.1, 0.2], rtol=1e-14)


# ======================================================================================
# 1.6 session_extreme
# ======================================================================================


def test_session_extreme_1d_and_2d():
    minute = np.array([555, 556, 700, 919, 920, 555, 556, 557])
    offs = _offs([5, 3])
    x = np.array([100.0, 3.0, -2.0, 7.0, 999.0, -50.0, NAN, NAN])
    mx = lf.session_extreme(x, minute, offs, "max")
    mn = lf.session_extreme(x, minute, offs, "min")
    _eq_nan(mx, [7.0, NAN])
    _eq_nan(mn, [-2.0, NAN])
    assert mx.shape == (2,)
    x2 = np.column_stack([x, np.arange(8, dtype=float)])
    mx2 = lf.session_extreme(x2, minute, offs, "max")
    mn2 = lf.session_extreme(x2, minute, offs, "min")
    assert mx2.shape == (2, 2)
    _eq_nan(mx2, [[7.0, 3.0], [NAN, 7.0]])
    _eq_nan(mn2, [[-2.0, 1.0], [NAN, 6.0]])
    # custom window
    _eq_nan(
        lf.session_extreme(x, minute, offs, "max", start_minute=555, end_minute=921), [999.0, -50.0]
    )


@pytest.mark.parametrize("how", ["mean", "MAX", ""])
def test_session_extreme_bad_how(how):
    with pytest.raises(ValueError):
        lf.session_extreme(np.ones(3), np.array([556, 557, 558]), _offs([3]), how)


# ======================================================================================
# 1.7 rolling_session_quantile
# ======================================================================================


def test_rolling_session_quantile_hand_slices():
    ps = np.array([1.0, 5.0, NAN, 3.0, 10.0, 2.0, 7.0, 4.0])
    usable = np.array([True, True, True, True, False, True, True, True])
    q = 0.25
    out = lf.rolling_session_quantile(ps, usable, q, window=3, min_sessions=2)
    slices = {
        0: None,
        1: None,  # only [1]
        2: [1, 5],
        3: [1, 5],  # 2 is NaN
        4: [1, 5, 3],
        5: [1, 5, 3],  # session 4 unusable
        6: [5, 3, 2],
        7: [3, 2, 7],
    }
    expected = [
        NAN if v is None else float(np.quantile(np.array(v, float), q)) for v in slices.values()
    ]
    _eq_nan(out, expected, rtol=1e-14)
    assert out.dtype == np.float64


def _ref_rsq(ps, usable, q, window, min_s):
    out = np.full(len(ps), np.nan)
    for d in range(len(ps)):
        vals = [ps[j] for j in range(d) if usable[j] and np.isfinite(ps[j])]
        vals = vals[-window:]
        if len(vals) >= min_s:
            out[d] = np.quantile(np.array(vals), q)
    return out


def test_rolling_session_quantile_matches_reference_and_causal_invariant_3():
    rng = np.random.default_rng(3)
    n = 60
    ps = rng.standard_normal(n)
    ps[rng.random(n) < 0.1] = NAN
    usable = rng.random(n) < 0.85
    out = lf.rolling_session_quantile(ps, usable, 0.9, window=12, min_sessions=5)
    _eq_nan(out, _ref_rsq(ps, usable, 0.9, 12, 5), rtol=1e-12)
    for d in (5, 20, 45):
        ps2, u2 = ps.copy(), usable.copy()
        ps2[d:] = rng.standard_normal(n - d) * 100
        u2[d:] = ~u2[d:]
        other = lf.rolling_session_quantile(ps2, u2, 0.9, window=12, min_sessions=5)
        np.testing.assert_array_equal(other[: d + 1], out[: d + 1])


@pytest.mark.parametrize(
    "q,window,min_s",
    [
        (0.0, 10, 5),
        (1.0, 10, 5),
        (-0.1, 10, 5),
        (1.5, 10, 5),
        (0.5, 4, 5),
        (0.5, 10, 0),
        (0.5, 0, 0),
    ],
)
def test_rolling_session_quantile_validation(q, window, min_s):
    with pytest.raises(ValueError):
        lf.rolling_session_quantile(
            np.ones(20), np.ones(20, bool), q, window=window, min_sessions=min_s
        )


def test_rolling_session_quantile_window_equal_min_ok():
    out = lf.rolling_session_quantile(
        np.arange(5.0), np.ones(5, bool), 0.5, window=2, min_sessions=2
    )
    _eq_nan(out, [NAN, NAN, 0.5, 1.5, 2.5])


# ======================================================================================
# 1.8 stress_gate
# ======================================================================================


def test_stress_gate_hand_values():
    offs = _offs([3, 3])
    vix = np.array([0.10, 0.05, 0.20, 0.30, 0.30, NAN])
    dev = np.array([-0.5, -0.5, -0.4, -1.0, -0.9, -2.0])
    v_day = np.array([0.10, 0.25])
    m_day = np.array([-0.5, -1.0])
    g = lf.stress_gate(vix, dev, v_day, m_day, offs)
    assert g.dtype == np.bool_
    assert g.shape == (6,)
    # row0 edges both inclusive -> True; row1 vix < v -> False; row2 dev > m -> False
    # row3 True (edge m); row4 dev -0.9 > -1.0 -> False; row5 vix NaN -> False
    np.testing.assert_array_equal(g, [True, False, False, True, False, False])


def test_stress_gate_nonfinite_thresholds_false():
    offs = _offs([2, 2])
    vix = np.array([1.0, 1.0, 1.0, 1.0])
    dev = np.array([-9.0, NAN, -9.0, -9.0])
    g = lf.stress_gate(vix, dev, np.array([0.0, NAN]), np.array([0.0, 0.0]), offs)
    np.testing.assert_array_equal(g, [True, False, False, False])
    g2 = lf.stress_gate(vix, dev, np.array([0.0, 0.0]), np.array([np.inf, NAN]), offs)
    np.testing.assert_array_equal(g2, [False, False, False, False])


# ======================================================================================
# 2.1 LiqParams
# ======================================================================================


def test_liqparams_defaults_and_frozen():
    p = ls.LiqParams(k=1.0, notional=1e5, mode="passive")
    assert p.tick == 0.05
    assert p.start_minute == 556
    assert p.square_off_minute == 920
    assert ls.LiqParams(k=1.0, notional=1e5, mode="market").mode == "market"
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.k = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        dict(k=0.0),
        dict(k=-1.0),
        dict(k=NAN),
        dict(k=np.inf),
        dict(notional=0.0),
        dict(notional=-5.0),
        dict(notional=NAN),
        dict(notional=np.inf),
        dict(mode="limit"),
        dict(mode="Passive"),
        dict(tick=0.0),
        dict(tick=-0.05),
        dict(tick=NAN),
        dict(tick=np.inf),
        dict(square_off_minute=556),
        dict(square_off_minute=500),
        dict(start_minute=920),
    ],
)
def test_liqparams_validation(kw):
    base = dict(k=1.0, notional=1e5, mode="passive")
    base.update(kw)
    with pytest.raises(ValueError):
        ls.LiqParams(**base)


# ======================================================================================
# 2.2 simulate_liqprov -- hand scenarios
#   k = 0.5, adr = 0.25, vwap = 80, idx_dev = 0 -> band = 80 * (1 - 0.125) = 70.0 (exact)
#   tick = 0.25 -> L - tick = 69.75 (exact). Default bar price 75: above band, below vwap.
# ======================================================================================

K = 0.5
ADR = 0.25
VW = 80.0
BAND = 70.0
TICK = 0.25
BASE = 75.0
NOTIONAL = 100_000.0


def _sc(minutes, lengths=None, n_sym=1, symbols=None):
    minutes = np.asarray(minutes, dtype=np.int64)
    n = len(minutes)
    offs = _offs([n] if lengths is None else lengths)
    n_days = len(offs) - 1

    def f(v):
        return np.full((n, n_sym), v, dtype=np.float64)

    return dict(
        open_=f(BASE),
        high=f(BASE),
        low=f(BASE),
        close=f(BASE),
        volume=f(1000.0),
        vwap=f(VW),
        idx_dev=np.zeros(n),
        beta_day=np.ones((n_days, n_sym)),
        adr_day=np.full((n_days, n_sym), ADR),
        gate=np.ones(n, dtype=bool),
        tradable=np.ones((n, n_sym), dtype=bool),
        minute_of_day=minutes,
        day_offsets=offs,
        symbols=list(symbols) if symbols is not None else [f"S{i}" for i in range(n_sym)],
    )


def _run(sc, mode="passive", cost_model=None, slippage=None, tick=TICK, **pkw):
    p = ls.LiqParams(k=K, notional=NOTIONAL, mode=mode, tick=tick, **pkw)
    return ls.simulate_liqprov(**sc, params=p, cost_model=cost_model, slippage=slippage)


def _trip(df):
    return [
        (int(r.signal_row), int(r.entry_row), int(r.exit_row), r.exit_reason)
        for r in df.itertuples()
    ]


M10 = list(range(556, 566))  # one session, 10 rows, 09:16..09:25


def _check_trade(row, *, entry, exit_, sig, ent, ex, reason, day=0, sym="S0"):
    qty = NOTIONAL / entry
    gross = qty * (exit_ - entry)
    assert row["symbol"] == sym
    assert int(row["day_idx"]) == day
    assert int(row["side"]) == 1
    assert int(row["signal_row"]) == sig
    assert int(row["entry_row"]) == ent
    assert int(row["exit_row"]) == ex
    assert row["exit_reason"] == reason
    assert int(row["holding_bars"]) == ex - ent
    assert row["entry_price"] == pytest.approx(entry, rel=1e-14)
    assert row["exit_price"] == pytest.approx(exit_, rel=1e-14)
    assert row["qty"] == pytest.approx(qty, rel=1e-14)
    assert row["gross_pnl"] == pytest.approx(gross, rel=1e-12)
    assert row["gross_bps"] == pytest.approx(gross / NOTIONAL * 1e4, rel=1e-12)


# ---- passive: invariant 5 ------------------------------------------------------------


def test_passive_low_equal_L_does_not_fill():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND
    df = _run(sc)
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


def test_passive_low_L_minus_tick_fills_at_L():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    df = _run(sc)
    assert len(df) == 1
    _check_trade(df.iloc[0], entry=BAND, exit_=BASE, sig=2, ent=3, ex=9, reason="no_bar")
    assert list(df.columns) == COLUMNS


def test_passive_gap_below_limit_fills_at_open():
    sc = _sc(M10)
    sc["open_"][3, 0] = 60.0
    sc["low"][3, 0] = 59.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=60.0, exit_=BASE, sig=2, ent=3, ex=9, reason="no_bar")
    # open between L - tick and L also fills at the open (min(L, open))
    sc = _sc(M10)
    sc["open_"][3, 0] = 69.9
    sc["low"][3, 0] = 69.5
    df = _run(sc)
    assert df.iloc[0]["entry_price"] == pytest.approx(69.9, rel=1e-14)


def test_passive_default_tick_and_deep_trade_through():
    sc = _sc(M10)
    sc["low"][3, 0] = 69.0
    df = _run(sc, tick=0.05)
    _check_trade(df.iloc[0], entry=BAND, exit_=BASE, sig=2, ent=3, ex=9, reason="no_bar")


def test_passive_gate_uses_t_minus_1():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["gate"][2] = False
    assert len(_run(sc)) == 0
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["gate"][:] = False
    sc["gate"][2] = True  # gate[t] False, gate[t-1] True -> fills
    df = _run(sc)
    assert _trip(df) == [(2, 3, 9, "no_bar")]


def test_passive_present_but_not_tradable_no_fill():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["tradable"][3, 0] = False
    assert len(_run(sc)) == 0


@pytest.mark.parametrize(
    "field,value",
    [("volume", 0.0), ("volume", NAN), ("open_", NAN), ("open_", 0.0), ("low", NAN)],
)
def test_passive_unfillable_row_no_fill(field, value):
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc[field][3, 0] = value
    assert len(_run(sc)) == 0


def test_passive_t_minus_1_before_start_minute_no_fill():
    sc = _sc(list(range(555, 565)))
    sc["low"][1, 0] = BAND - TICK  # t-1 = row 0 is 09:15
    assert len(_run(sc)) == 0
    sc["low"][2, 0] = BAND - TICK  # t-1 = row 1 (09:16) -> fills
    assert _trip(_run(sc)) == [(1, 2, 9, "no_bar")]


def test_passive_band_is_that_of_t_minus_1():
    sc = _sc(M10)
    sc["vwap"][3, 0] = 40.0  # band[3] would be 35; L must come from row 2 (70)
    sc["low"][3, 0] = BAND - TICK
    df = _run(sc)
    assert df.iloc[0]["entry_price"] == pytest.approx(BAND, rel=1e-14)
    assert int(df.iloc[0]["entry_row"]) == 3


def test_passive_band_with_beta_and_idx_dev():
    # beta 0.5, idx_dev -0.25: band = 80 * (1 - 0.125 - 0.125) = 60
    sc = _sc(M10)
    sc["beta_day"][:] = 0.5
    sc["idx_dev"][2] = -0.25
    sc["low"][3, 0] = BAND - TICK  # 69.75: not below 60 - 0.25
    assert len(_run(sc)) == 0
    sc["low"][3, 0] = 59.75
    df = _run(sc)
    assert df.iloc[0]["entry_price"] == pytest.approx(60.0, rel=1e-14)


@pytest.mark.parametrize(
    "what", ["vwap_nan", "vwap_zero", "idx_nan", "beta_nan", "adr_nan", "adr_zero", "band_neg"]
)
def test_passive_undefined_band_no_fill(what):
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    if what == "vwap_nan":
        sc["vwap"][2, 0] = NAN
    elif what == "vwap_zero":
        sc["vwap"][2, 0] = 0.0
    elif what == "idx_nan":
        sc["idx_dev"][2] = NAN
    elif what == "beta_nan":
        sc["beta_day"][0, 0] = NAN
    elif what == "adr_nan":
        sc["adr_day"][0, 0] = NAN
    elif what == "adr_zero":
        sc["adr_day"][0, 0] = 0.0
    elif what == "band_neg":
        sc["adr_day"][0, 0] = 4.0  # band = 80 * (1 - 2) = -80
        sc["low"][3, 0] = -100.0
    assert len(_run(sc)) == 0


def test_passive_no_entry_at_or_after_square_off():
    minutes = [917, 918, 919, 920, 921, 922]
    sc = _sc(minutes)
    sc["low"][3, 0] = BAND - TICK  # t minute 920 -> no fill
    sc["low"][4, 0] = BAND - TICK
    assert len(_run(sc)) == 0


def test_passive_no_fill_across_session_boundary():
    sc = _sc(M10[:6] + M10[:6], lengths=[6, 6])
    sc["low"][6, 0] = BAND - TICK  # first row of session 1; t-1 is last row of session 0
    assert len(_run(sc)) == 0


def test_passive_reentry_from_x_plus_1_not_x():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["close"][3, 0] = 85.0  # entry row's own close >= vwap -> pending vwap exit at row 4
    sc["open_"][4, 0] = 76.0
    sc["low"][4, 0] = BAND - TICK  # exit row x=4: must NOT be an entry row
    sc["low"][5, 0] = BAND - TICK  # x+1: order placed at row 4's close -> fills
    df = _run(sc)
    assert _trip(df) == [(2, 3, 4, "vwap"), (4, 5, 9, "no_bar")]
    _check_trade(df.iloc[0], entry=BAND, exit_=76.0, sig=2, ent=3, ex=4, reason="vwap")


# ---- market mode: invariant 6 --------------------------------------------------------


def test_market_signal_at_band_fills_next_open():
    sc = _sc(M10)
    sc["close"][2, 0] = BAND  # close <= band (inclusive)
    sc["open_"][3, 0] = 71.0
    df = _run(sc, mode="market")
    assert len(df) == 1
    _check_trade(df.iloc[0], entry=71.0, exit_=BASE, sig=2, ent=3, ex=9, reason="no_bar")


def test_market_close_above_band_no_signal():
    sc = _sc(M10)
    sc["close"][2, 0] = BAND + 0.25
    sc["low"][2, 0] = 10.0  # the low is irrelevant in market mode
    assert len(_run(sc, mode="market")) == 0


def test_market_gate_at_r():
    sc = _sc(M10)
    sc["close"][2, 0] = BAND
    sc["gate"][2] = False
    assert len(_run(sc, mode="market")) == 0
    sc["gate"][:] = False
    sc["gate"][2] = True
    assert _trip(_run(sc, mode="market")) == [(2, 3, 9, "no_bar")]


@pytest.mark.parametrize("block", ["tradable", "volume", "open_nan"])
def test_market_dropped_signal_not_deferred(block):
    sc = _sc(M10)
    sc["close"][2, 0] = BAND
    if block == "tradable":
        sc["tradable"][3, 0] = False
    elif block == "volume":
        sc["volume"][3, 0] = 0.0
    else:
        sc["open_"][3, 0] = NAN
    assert len(_run(sc, mode="market")) == 0


def test_market_signal_window_and_session_edge():
    # r minute < start
    sc = _sc(list(range(555, 565)))
    sc["close"][0, 0] = BAND
    assert len(_run(sc, mode="market")) == 0
    # r+1 at 15:20 -> dropped; r at 15:20 -> no signal
    sc = _sc([917, 918, 919, 920, 921])
    sc["close"][2, 0] = BAND
    sc["close"][3, 0] = BAND
    assert len(_run(sc, mode="market")) == 0
    # r is the last row of a session -> r+1 in next session -> dropped
    sc = _sc(M10[:5] + M10[:5], lengths=[5, 5])
    sc["close"][4, 0] = BAND
    assert len(_run(sc, mode="market")) == 0
    # absent bar at r cannot signal (NaN close) even with gate True
    sc = _sc(M10)
    sc["close"][2, 0] = NAN
    assert len(_run(sc, mode="market")) == 0


def test_market_band_with_beta():
    sc = _sc(M10)
    sc["beta_day"][:] = 0.5
    sc["idx_dev"][2] = -0.25  # band[2] = 60
    sc["close"][2, 0] = 65.0
    assert len(_run(sc, mode="market")) == 0
    sc["close"][2, 0] = 60.0
    assert _trip(_run(sc, mode="market")) == [(2, 3, 9, "no_bar")]


def test_market_reentry_from_x():
    sc = _sc(M10)
    sc["close"][2, 0] = BAND
    sc["close"][3, 0] = 85.0  # entry row close >= vwap -> exit at open[4]
    sc["open_"][4, 0] = 77.0
    sc["close"][4, 0] = BAND  # exit row x=4 signals again -> fills at 5
    sc["open_"][5, 0] = 72.0
    df = _run(sc, mode="market")
    assert _trip(df) == [(2, 3, 4, "vwap"), (4, 5, 9, "no_bar")]
    _check_trade(df.iloc[0], entry=BASE, exit_=77.0, sig=2, ent=3, ex=4, reason="vwap")
    _check_trade(df.iloc[1], entry=72.0, exit_=BASE, sig=4, ent=5, ex=9, reason="no_bar")


def test_market_never_shorts():
    sc = _sc(M10)
    sc["close"][2, 0] = 200.0  # far above any upper band
    sc["close"][5, 0] = 1.0  # far below -> long
    df = _run(sc, mode="market")
    assert (df["side"] == 1).all()
    assert _trip(df) == [(5, 6, 9, "no_bar")]


# ---- exits ---------------------------------------------------------------------------


def test_exit_square_off_at_open_of_first_fillable_row():
    sc = _sc([917, 918, 919, 920, 921, 922])
    sc["low"][1, 0] = BAND - TICK  # entry row 1
    sc["open_"][3, 0] = 77.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=77.0, sig=0, ent=1, ex=3, reason="square_off")
    sc["volume"][3, 0] = 0.0  # 15:20 row not fillable -> next
    sc["open_"][4, 0] = 78.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=78.0, sig=0, ent=1, ex=4, reason="square_off")


def test_exit_pending_vwap_takes_precedence_on_square_off_row():
    sc = _sc([917, 918, 919, 920, 921])
    sc["low"][1, 0] = BAND - TICK
    sc["close"][2, 0] = 81.0
    df = _run(sc)
    assert _trip(df) == [(0, 1, 3, "vwap")]


def test_exit_vwap_ignores_tradable_and_skips_unfillable():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["close"][4, 0] = VW  # close >= vwap (equality)
    sc["tradable"][5:, 0] = False
    sc["volume"][5, 0] = 0.0  # row 5 not fillable -> row 6
    sc["open_"][6, 0] = 79.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=79.0, sig=2, ent=3, ex=6, reason="vwap")


def test_exit_no_bar_last_valid_row():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["close"][8, 0] = 76.0
    sc["volume"][9, 0] = 0.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=76.0, sig=2, ent=3, ex=8, reason="no_bar")
    sc["volume"][9, 0] = 1000.0
    sc["close"][9, 0] = NAN
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=76.0, sig=2, ent=3, ex=8, reason="no_bar")


def test_exit_pending_vwap_never_filled_becomes_no_bar():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["close"][3, 0] = 85.0
    sc["open_"][4:, 0] = NAN  # present (close finite) but not fillable
    sc["close"][9, 0] = 74.0
    df = _run(sc)
    _check_trade(df.iloc[0], entry=BAND, exit_=74.0, sig=2, ent=3, ex=9, reason="no_bar")


def test_position_never_spans_sessions():
    sc = _sc(M10[:6] + M10[:6], lengths=[6, 6])
    sc["low"][3, 0] = BAND - TICK
    df = _run(sc)
    assert _trip(df) == [(2, 3, 5, "no_bar")]
    assert int(df.iloc[0]["day_idx"]) == 0
    sc2 = _sc(M10[:6] + M10[:6], lengths=[6, 6])
    sc2["close"][3, 0] = BAND
    df2 = _run(sc2, mode="market")
    assert _trip(df2) == [(3, 4, 5, "no_bar")]
    # day_idx of a session-1 trade
    sc3 = _sc(M10[:6] + M10[:6], lengths=[6, 6])
    sc3["low"][8, 0] = BAND - TICK
    df3 = _run(sc3)
    assert _trip(df3) == [(7, 8, 11, "no_bar")]
    assert int(df3.iloc[0]["day_idx"]) == 1


def test_one_position_at_a_time_passive():
    sc = _sc(M10)
    sc["low"][3:8, 0] = BAND - TICK  # repeated trade-throughs while long
    assert _trip(_run(sc)) == [(2, 3, 9, "no_bar")]


def test_multi_symbol_sorting_and_independence():
    sc = _sc(M10, n_sym=3, symbols=["ZZZ", "AAA", "MMM"])
    sc["low"][2, 0] = BAND - TICK  # ZZZ entry 2
    sc["low"][3, 1] = BAND - TICK  # AAA entry 3
    sc["low"][3, 2] = BAND - TICK  # MMM entry 3
    sc["tradable"][2, 1] = False  # only affects AAA (which has no trade-through at 2 anyway)
    sc["close"][4, 1] = 90.0  # AAA exits at 5
    df = _run(sc)
    assert list(df["symbol"]) == ["ZZZ", "AAA", "MMM"]
    assert list(df["entry_row"]) == [2, 3, 3]
    assert list(df["exit_row"]) == [9, 5, 9]
    assert list(df.columns) == COLUMNS
    # symbol-specific tradable
    sc["tradable"][3, 2] = False
    df = _run(sc)
    assert list(df["symbol"]) == ["ZZZ", "AAA"]


# ---- costs: invariant 8 --------------------------------------------------------------


def _charges(entry_notional, exit_notional):
    return float(
        NSEIntradayEquityCosts()
        .charges(
            FillBatch(
                notional=np.array([entry_notional, exit_notional]), is_buy=np.array([True, False])
            )
        )
        .total.sum()
    )


def _slip(notional, btv):
    return (
        notional * float(SqrtImpactSlippage().bps(np.array([notional]), np.array([btv]))[0]) / 1e4
    )


def test_zero_cost_models():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    for mode in ("passive", "market"):
        if mode == "market":
            sc["close"][2, 0] = BAND
        df = _run(sc, mode=mode)
        assert len(df) >= 1
        assert (df["costs"] == 0.0).all()
        assert (df["net_pnl"] == df["gross_pnl"]).all()
        assert (df["net_bps"] == df["gross_bps"]).all()


def test_passive_costs_slippage_on_exit_only():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["volume"][3, 0] = 5000.0
    sc["close"][5, 0] = 85.0  # vwap exit at open[6]
    sc["open_"][6, 0] = 78.0
    sc["close"][6, 0] = 79.0
    sc["volume"][6, 0] = 2000.0
    df = _run(sc, cost_model=NSEIntradayEquityCosts(), slippage=SqrtImpactSlippage())
    r = df.iloc[0]
    qty = NOTIONAL / BAND
    en, ex = qty * BAND, qty * 78.0
    expected = _charges(en, ex) + _slip(ex, 79.0 * 2000.0)
    assert r["costs"] == pytest.approx(expected, rel=1e-12)
    wrong = expected + _slip(en, BASE * 5000.0)
    assert abs(r["costs"] - wrong) > 1e-3
    gross = qty * (78.0 - BAND)
    assert r["gross_pnl"] == pytest.approx(gross, rel=1e-12)
    assert r["net_pnl"] == pytest.approx(gross - expected, rel=1e-12)
    assert r["net_bps"] == pytest.approx((gross - expected) / NOTIONAL * 1e4, rel=1e-12)
    # slippage only (no cost model) -> exit slippage alone
    df = _run(sc, slippage=SqrtImpactSlippage())
    assert df.iloc[0]["costs"] == pytest.approx(_slip(ex, 79.0 * 2000.0), rel=1e-12)
    # cost model only -> charges alone
    df = _run(sc, cost_model=NSEIntradayEquityCosts())
    assert df.iloc[0]["costs"] == pytest.approx(_charges(en, ex), rel=1e-12)


def test_passive_no_bar_exit_slippage_uses_exit_row():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    sc["close"][9, 0] = 76.0
    sc["volume"][9, 0] = 300.0
    df = _run(sc, cost_model=NSEIntradayEquityCosts(), slippage=SqrtImpactSlippage())
    qty = NOTIONAL / BAND
    en, ex = qty * BAND, qty * 76.0
    assert df.iloc[0]["costs"] == pytest.approx(
        _charges(en, ex) + _slip(ex, 76.0 * 300.0), rel=1e-12
    )


def test_market_costs_slippage_on_both_fills():
    sc = _sc(M10)
    sc["close"][2, 0] = BAND
    sc["open_"][3, 0] = 71.0
    sc["close"][3, 0] = 72.0
    sc["volume"][3, 0] = 5000.0
    sc["close"][9, 0] = 76.0
    sc["volume"][9, 0] = 700.0
    df = _run(sc, mode="market", cost_model=NSEIntradayEquityCosts(), slippage=SqrtImpactSlippage())
    r = df.iloc[0]
    qty = NOTIONAL / 71.0
    en, ex = qty * 71.0, qty * 76.0
    expected = _charges(en, ex) + _slip(en, 72.0 * 5000.0) + _slip(ex, 76.0 * 700.0)
    assert r["costs"] == pytest.approx(expected, rel=1e-12)
    assert r["net_pnl"] == pytest.approx(qty * (76.0 - 71.0) - expected, rel=1e-12)


# ---- output / shapes -----------------------------------------------------------------


def test_empty_frame_columns_both_modes():
    for mode in ("passive", "market"):
        df = _run(_sc(M10), mode=mode)
        assert len(df) == 0
        assert list(df.columns) == COLUMNS


def test_output_dtypes():
    sc = _sc(M10)
    sc["low"][3, 0] = BAND - TICK
    df = _run(sc)
    for c in ("day_idx", "side", "signal_row", "entry_row", "exit_row", "holding_bars"):
        assert pd.api.types.is_integer_dtype(df[c]), c
    for c in (
        "entry_price",
        "exit_price",
        "qty",
        "gross_pnl",
        "costs",
        "net_pnl",
        "gross_bps",
        "net_bps",
    ):
        assert df[c].dtype == np.float64, c


@pytest.mark.parametrize(
    "key,bad",
    [
        ("open_", lambda n, s, d: np.ones((n, s + 1))),
        ("high", lambda n, s, d: np.ones((n + 1, s))),
        ("low", lambda n, s, d: np.ones((n, s + 1))),
        ("close", lambda n, s, d: np.ones((n - 1, s))),
        ("volume", lambda n, s, d: np.ones((n, s + 1))),
        ("vwap", lambda n, s, d: np.ones((n, s + 1))),
        ("tradable", lambda n, s, d: np.ones((n, s + 1), bool)),
        ("idx_dev", lambda n, s, d: np.zeros(n + 1)),
        ("gate", lambda n, s, d: np.ones(n - 1, bool)),
        ("minute_of_day", lambda n, s, d: np.arange(556, 556 + n + 1)),
        ("beta_day", lambda n, s, d: np.ones((d + 1, s))),
        ("beta_day", lambda n, s, d: np.ones((d, s + 1))),
        ("adr_day", lambda n, s, d: np.ones((d, s + 1))),
        ("adr_day", lambda n, s, d: np.ones((d + 1, s))),
        ("symbols", lambda n, s, d: ["A", "B", "C"]),
    ],
)
def test_shape_mismatch_raises(key, bad):
    sc = _sc(M10[:5] + M10[:5], lengths=[5, 5], n_sym=2)
    sc[key] = bad(10, 2, 2)
    for mode in ("passive", "market"):
        with pytest.raises(ValueError):
            _run(sc, mode=mode)


# ---- structural invariants on random panels (invariant 7) ----------------------------


def _random_sc(seed):
    rng = np.random.default_rng(seed)
    lengths = [60, 105, 30, 60]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = _offs(lengths)
    n, n_sym = int(offs[-1]), 3
    # push the last 5 rows of session 1 to 15:18..15:22 to exercise square-off
    s1_end = offs[2]
    minute[s1_end - 5 : s1_end] = np.arange(918, 923)
    path = 100 * np.exp(np.cumsum(rng.standard_normal((n + 1, n_sym)) * 0.004, axis=0))
    open_, close = path[:-1].copy(), path[1:].copy()
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal((n, n_sym))) * 0.002)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal((n, n_sym))) * 0.002)
    volume = rng.integers(0, 3000, (n, n_sym)).astype(float)
    gap = rng.random((n, n_sym)) < 0.03
    for a in (open_, close, high, low, volume):
        a[gap] = NAN
    vwap = vf.session_vwap(high, low, close, volume, minute, offs)
    return dict(
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        idx_dev=rng.standard_normal(n) * 0.002,
        beta_day=rng.uniform(0.5, 1.5, (len(lengths), n_sym)),
        adr_day=rng.uniform(0.005, 0.02, (len(lengths), n_sym)),
        gate=rng.random(n) < 0.6,
        tradable=rng.random((n, n_sym)) < 0.9,
        minute_of_day=minute,
        day_offsets=offs,
        symbols=["X", "Y", "Z"],
    )


@pytest.mark.parametrize("seed", [31, 32, 33])
@pytest.mark.parametrize("mode", ["passive", "market"])
def test_structural_invariants_random(seed, mode):
    sc = _random_sc(seed)
    p = ls.LiqParams(k=0.3, notional=NOTIONAL, mode=mode)
    df = ls.simulate_liqprov(**sc, params=p)
    assert list(df.columns) == COLUMNS
    assert len(df) > 0
    offs = sc["day_offsets"]
    minute = sc["minute_of_day"]
    assert (df["side"] == 1).all()
    assert set(df["exit_reason"]) <= {"vwap", "square_off", "no_bar"}
    keys = list(zip(df["entry_row"], df["symbol"]))
    assert keys == sorted(keys)
    for sym, g in df.groupby("symbol"):
        s = sc["symbols"].index(sym)
        g = g.sort_values("entry_row")
        er, xr = g["entry_row"].to_numpy(), g["exit_row"].to_numpy()
        assert np.all(er[1:] > xr[:-1])  # one position at a time; re-entry after exit
        for r in g.itertuples():
            d = int(np.searchsorted(offs, r.entry_row, side="right") - 1)
            assert r.day_idx == d
            assert offs[d] <= r.signal_row and r.exit_row < offs[d + 1]
            assert r.entry_row - r.signal_row == 1
            assert r.exit_row >= r.entry_row
            assert r.holding_bars == r.exit_row - r.entry_row
            sig, e = r.signal_row, r.entry_row
            band = sc["vwap"][sig, s] * (
                1 + sc["beta_day"][d, s] * sc["idx_dev"][sig] - 0.3 * sc["adr_day"][d, s]
            )
            assert sc["gate"][sig]
            assert sc["tradable"][e, s]
            assert sc["volume"][e, s] > 0 and sc["open_"][e, s] > 0
            assert minute[sig] >= 556 and minute[e] < 920
            if mode == "passive":
                assert sc["low"][e, s] <= band - 0.05 + 1e-9
                assert r.entry_price == pytest.approx(min(band, sc["open_"][e, s]), rel=1e-12)
            else:
                assert sc["close"][sig, s] <= band * (1 + 1e-12)
                assert r.entry_price == sc["open_"][e, s]
            assert r.qty * r.entry_price == pytest.approx(NOTIONAL, rel=1e-12)
            if r.exit_reason == "square_off":
                assert minute[r.exit_row] >= 920
                assert r.exit_price == sc["open_"][r.exit_row, s]
            if r.exit_reason == "no_bar":
                assert r.exit_price == sc["close"][r.exit_row, s]
        assert (g["costs"] == 0).all()


# ======================================================================================
# 3. evaluate
# ======================================================================================


def _trades(day_idx, net_bps, net_pnl=None, gross_bps=None):
    day_idx = list(day_idx)
    n = len(day_idx)
    net_bps = np.asarray(net_bps, dtype=float)
    net_pnl = net_bps * 10.0 if net_pnl is None else np.asarray(net_pnl, dtype=float)
    gross_bps = net_bps + 1.0 if gross_bps is None else np.asarray(gross_bps, dtype=float)
    return pd.DataFrame(
        {
            "symbol": ["S"] * n,
            "day_idx": np.asarray(day_idx, dtype=np.int64),
            "side": np.ones(n, dtype=np.int64),
            "signal_row": np.arange(n, dtype=np.int64),
            "entry_row": np.arange(n, dtype=np.int64) + 1,
            "entry_price": np.full(n, 100.0),
            "exit_row": np.arange(n, dtype=np.int64) + 2,
            "exit_price": np.full(n, 100.0),
            "exit_reason": ["vwap"] * n,
            "qty": np.full(n, 1000.0),
            "gross_pnl": gross_bps * 10.0,
            "costs": (gross_bps - net_bps) * 10.0,
            "net_pnl": net_pnl,
            "gross_bps": gross_bps,
            "net_bps": net_bps,
            "holding_bars": np.ones(n, dtype=np.int64),
        },
        columns=COLUMNS,
    )


def test_day_clustered_t_hand():
    tr = _trades([0, 0, 3, 7, 7, 7], [10, 20, 5, -1, 3, 7])
    means = [15.0, 5.0, 3.0]
    m = sum(means) / 3
    sd = math.sqrt(sum((v - m) ** 2 for v in means) / 2)
    assert le.day_clustered_t(tr) == pytest.approx(m / (sd / math.sqrt(3)), rel=1e-12)
    gmeans = [16.0, 6.0, 4.0]
    gm = sum(gmeans) / 3
    gsd = math.sqrt(sum((v - gm) ** 2 for v in gmeans) / 2)
    assert le.day_clustered_t(tr, column="gross_bps") == pytest.approx(
        gm / (gsd / math.sqrt(3)), rel=1e-12
    )


def test_day_clustered_t_nan_cases():
    assert math.isnan(le.day_clustered_t(_trades([4, 4, 4], [1, 2, 3])))  # one day
    assert math.isnan(le.day_clustered_t(_trades([1, 2, 2], [5, 4, 6])))  # zero std of day means
    assert math.isnan(le.day_clustered_t(_trades([], [])))


def test_yearly_net_hand():
    dates = [dt.date(2018, 12, 31), dt.date(2019, 1, 1), dt.date(2019, 6, 3), dt.date(2021, 2, 1)]
    tr = _trades([0, 0, 1, 2, 3], [0, 0, 0, 0, 0], net_pnl=[100.0, -30.0, 5.0, 7.5, -2.0])
    out = le.yearly_net(tr, dates)
    assert set(out) == {2018, 2019, 2021}
    assert out[2018] == pytest.approx(70.0)
    assert out[2019] == pytest.approx(12.5)
    assert out[2021] == pytest.approx(-2.0)
    assert le.yearly_net(_trades([], []), dates) == {}


def test_mean_net_bps_without_top_days_hand():
    # day totals of net_pnl: d0 = 50, d1 = -10, d2 = 90, d3 = 20
    tr = _trades(
        [0, 0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0, 5.0], net_pnl=[20.0, 30.0, -10.0, 90.0, 20.0]
    )
    # drop 1 -> day 2 removed: remaining trade bps [1,2,3,5]
    assert le.mean_net_bps_without_top_days(tr, n_top=1) == pytest.approx(11.0 / 4)
    # drop 2 -> days 2 and 0 removed: [3,5]
    assert le.mean_net_bps_without_top_days(tr, n_top=2) == pytest.approx(4.0)
    # drop 3 -> days 2, 0, 3 removed: [3]
    assert le.mean_net_bps_without_top_days(tr, n_top=3) == pytest.approx(3.0)
    assert math.isnan(le.mean_net_bps_without_top_days(tr, n_top=4))
    assert math.isnan(le.mean_net_bps_without_top_days(tr, n_top=10))


def test_mean_net_bps_without_top_days_default_is_5():
    days = list(range(7))
    pnl = [70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    bps = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    tr = _trades(days, bps, net_pnl=pnl)
    assert le.mean_net_bps_without_top_days(tr) == pytest.approx(1.5)


YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def _dates_two_per_year():
    return [dt.date(y, m, 2) for y in YEARS for m in (3, 9)]  # day_idx 0..11


def _expected_kill(tr, dates, years):
    bps = tr["net_bps"].to_numpy()
    mean = float(np.mean(bps)) if len(bps) else NAN
    daymeans = tr.groupby("day_idx")["net_bps"].mean().to_numpy()
    if len(daymeans) >= 2 and np.std(daymeans, ddof=1) > 0:
        t = float(np.mean(daymeans) / (np.std(daymeans, ddof=1) / math.sqrt(len(daymeans))))
    else:
        t = NAN
    ysum = {}
    for d, p in zip(tr["day_idx"], tr["net_pnl"]):
        ysum[dates[d].year] = ysum.get(dates[d].year, 0.0) + p
    pos = sum(1 for y in years if ysum.get(y, 0.0) > 0)
    tot = tr.groupby("day_idx")["net_pnl"].sum().sort_values(ascending=False)
    keep = set(tot.index[5:])
    rest = tr[tr["day_idx"].isin(keep)]["net_bps"]
    ex = float(rest.mean()) if len(rest) else NAN
    return mean, t, pos, ex


def _assert_kill(out, tr, dates, years):
    mean, t, pos, ex = _expected_kill(tr, dates, years)
    for key in (
        "mean_net_bps",
        "day_t",
        "positive_years",
        "n_years",
        "mean_net_bps_ex_top5",
        "c1_mean_net_positive",
        "c2_day_t_gt_2",
        "c3_positive_years_ge_4",
        "c4_ex_top5_positive",
        "passes",
    ):
        assert key in out, key
    _eq_nan(out["mean_net_bps"], mean, rtol=1e-12)
    _eq_nan(out["day_t"], t, rtol=1e-12)
    _eq_nan(out["mean_net_bps_ex_top5"], ex, rtol=1e-12)
    assert out["positive_years"] == pos
    assert isinstance(out["positive_years"], (int, np.integer))
    assert out["n_years"] == len(years)
    c = (mean > 0, t > 2, pos >= 4, ex > 0)
    assert bool(out["c1_mean_net_positive"]) is c[0]
    assert bool(out["c2_day_t_gt_2"]) is c[1]
    assert bool(out["c3_positive_years_ge_4"]) is c[2]
    assert bool(out["c4_ex_top5_positive"]) is c[3]
    for k in (
        "c1_mean_net_positive",
        "c2_day_t_gt_2",
        "c3_positive_years_ge_4",
        "c4_ex_top5_positive",
        "passes",
    ):
        assert isinstance(out[k], (bool, np.bool_)), k
    return c


def test_kill_criteria_passing_hand_invariant_9():
    dates = _dates_two_per_year()
    bps = [10.0, 12.5, 11.0, 9.0, 10.5, 13.0, 8.0, 11.5, 12.0, 9.5, 8.5, 11.25]
    tr = _trades(range(12), bps)
    out = le.kill_criteria(tr, dates, YEARS)
    c = _assert_kill(out, tr, dates, YEARS)
    assert all(c)
    assert bool(out["passes"]) is True
    # explicit hand numbers
    assert out["mean_net_bps"] == pytest.approx(sum(bps) / 12, rel=1e-12)
    assert out["positive_years"] == 6
    ex_bps = sorted(bps)[:7]  # net_pnl = 10 * bps, so top-5 days are the 5 largest bps
    assert out["mean_net_bps_ex_top5"] == pytest.approx(sum(ex_bps) / 7, rel=1e-12)


def test_kill_criteria_year_without_trades_not_positive():
    dates = _dates_two_per_year()
    # trades only in 2018..2020 (day_idx 0..5), plus positive 2021 -> 4 positive years
    bps4 = [10.0, 12.5, 11.0, 9.0, 10.5, 13.0, 8.0, 11.5]
    tr4 = _trades(range(8), bps4)
    out4 = le.kill_criteria(tr4, dates, YEARS)
    _assert_kill(out4, tr4, dates, YEARS)
    assert out4["positive_years"] == 4 and out4["n_years"] == 6
    assert bool(out4["c3_positive_years_ge_4"]) is True
    tr3 = _trades(range(6), bps4[:6])
    out3 = le.kill_criteria(tr3, dates, YEARS)
    _assert_kill(out3, tr3, dates, YEARS)
    assert out3["positive_years"] == 3
    assert bool(out3["c3_positive_years_ge_4"]) is False
    assert bool(out3["passes"]) is False


def test_kill_criteria_failing_mix():
    dates = _dates_two_per_year()
    # 2019 and 2021 negative; mean still positive; heavy top-day concentration
    bps = [5.0, 2.0, -30.0, 1.0, 200.0, 3.0, -4.0, -6.0, 150.0, 1.5, 120.0, 2.5]
    tr = _trades(range(12), bps)
    out = le.kill_criteria(tr, dates, YEARS)
    c = _assert_kill(out, tr, dates, YEARS)
    assert bool(out["passes"]) is all(c)
    assert bool(out["passes"]) is False


def test_kill_criteria_nan_values_do_not_pass():
    dates = _dates_two_per_year()
    out = le.kill_criteria(_trades([], []), dates, YEARS)
    assert math.isnan(out["mean_net_bps"]) and math.isnan(out["day_t"])
    assert out["positive_years"] == 0
    assert bool(out["c1_mean_net_positive"]) is False
    assert bool(out["passes"]) is False
    # one trade day only: day_t NaN, ex-top5 NaN -> passes False even though mean > 0
    tr = _trades([0, 0], [10.0, 20.0])
    out = le.kill_criteria(tr, dates, YEARS)
    _assert_kill(out, tr, dates, YEARS)
    assert math.isnan(out["day_t"])
    assert bool(out["c1_mean_net_positive"]) is True
    assert bool(out["c2_day_t_gt_2"]) is False
    assert bool(out["passes"]) is False


# ======================================================================================
# Invariant 10 -- null (CLAUDE.md rule 9): rate across 30 fixed seeds, never re-seeded
# ======================================================================================


def _null_run(seed, mode):
    rng = np.random.default_rng(seed)
    lengths = [60, 105, 60, 60, 105, 60, 60, 105, 60, 60]
    minute = np.concatenate([555 + np.arange(n) for n in lengths])
    offs = _offs(lengths)
    n, n_sym, n_days = int(offs[-1]), 5, len(lengths)
    path = 100 * np.exp(np.cumsum(rng.standard_normal((n + 1, n_sym)) * 0.001, axis=0))
    open_, close = path[:-1].copy(), path[1:].copy()
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal((n, n_sym))) * 0.0003)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal((n, n_sym))) * 0.0003)
    volume = rng.integers(100, 5000, (n, n_sym)).astype(float)
    vwap = vf.session_vwap(high, low, close, volume, minute, offs)
    adr = vf.adr_pct(vf.session_range_pct(high, low, close, minute, offs), 2)
    idx = 20000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.0008))
    # own index deviation from a session running mean of close (independent of index_twap)
    idx_dev = np.full(n, NAN)
    for d in range(n_days):
        seg = np.arange(offs[d], offs[d + 1])
        seg = seg[minute[seg] >= 556]
        idx_dev[seg] = idx[seg] / (np.cumsum(idx[seg]) / np.arange(1, len(seg) + 1)) - 1
    beta = rng.uniform(0.5, 1.5, (n_days, n_sym))
    gate = rng.random(n) < 0.5
    # k measured on this panel (rule 8): median of per-(sym, session) min residual z
    day_of_row = np.repeat(np.arange(n_days), lengths)
    z = ((close / vwap - 1) - beta[day_of_row] * idx_dev[:, None]) / adr[day_of_row]
    minz = []
    for d in range(n_days):
        rows = np.arange(offs[d], offs[d + 1])
        rows = rows[(minute[rows] >= 556) & (minute[rows] < 920)]
        for s in range(n_sym):
            v = z[rows, s]
            v = v[np.isfinite(v)]
            if len(v):
                minz.append(v.min())
    k = -float(np.quantile(minz, 0.50))
    assert np.isfinite(k) and k > 0
    return ls.simulate_liqprov(
        open_,
        high,
        low,
        close,
        volume,
        vwap,
        idx_dev,
        beta,
        adr,
        gate,
        np.ones((n, n_sym), dtype=bool),
        minute,
        offs,
        [f"N{i}" for i in range(n_sym)],
        ls.LiqParams(k=k, notional=NOTIONAL, mode=mode),
    )


@pytest.mark.parametrize("mode", ["passive", "market"])
def test_null_random_walk_rejection_rate_invariant_10(mode):
    n_seeds = 30
    rejections = 0
    counts = []
    for seed in range(5000, 5000 + n_seeds):
        df = _null_run(seed, mode)
        assert (df["costs"] == 0.0).all()
        assert (df["side"] == 1).all()
        counts.append(len(df))
        t = le.day_clustered_t(df, column="gross_bps")
        if np.isfinite(t) and abs(t) > 2.58:
            rejections += 1
    assert np.median(counts) >= 10, counts
    assert rejections / n_seeds <= 0.2, (rejections, counts)
