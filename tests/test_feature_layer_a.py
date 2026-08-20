"""Independent test suite A for specs/feature_layer.md (Spec D -- the feature layer).

Written from the spec ALONE, before any implementation exists (rule 1: dual independent
suites). RED is the expected state here: an ImportError / AttributeError on a name that does
not exist yet IS the correct signal, not a bug in this suite. Nothing here is weakened,
skipped, or guarded with hasattr/try-except to force a false GREEN.

`stitch_overnight_gaps`, `hurst_on_stitched`, `breakout_strength`, `garman_klass_volatility`,
`rogers_satchell_volatility`, `SIGMA_FLOOR`/`sigma_risk`, `efficiency_ratio`, and the whole
`nifty_quant.research.ic` module do not exist anywhere in the repo yet (verified by grep).
Their names/signatures below are a best-faith reading of specs/feature_layer.md, matching the
existing naming and day_offsets/min_count conventions in features/core.py and
features/persistence.py (see e.g. parkinson_volatility, breakout_up, rolling_hurst). If the
real implementation lands under different names, the resulting AttributeError/ImportError is
still the correct RED signal, not a defect in this suite -- report the mismatch, don't paper
over it.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.stats

from nifty_quant.features import core, persistence

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _irregular_day_offsets() -> np.ndarray:
    """Muhurat-shaped (60), disaster-recovery-shaped (105), regular (375) -- rule 5."""
    return np.array([0, 60, 165, 540], dtype=np.int64)


def _four_session_day_offsets() -> np.ndarray:
    """Four irregular sessions, EVERY ONE strictly shorter than 390 bars -- the precondition
    for obligation 4's arithmetic-impossibility claim (no window=390 lookback fits inside any
    single session here)."""
    return np.array([0, 60, 165, 540, 915], dtype=np.int64)


def _lognormal_close(
    rng: np.random.Generator,
    day_offsets: np.ndarray,
    n_symbols: int = 2,
    sigma: float = 0.004,
    gap_sigma: float = 0.02,
) -> np.ndarray:
    """A strictly-positive continuous cumulative price path with an extra jump injected at
    every session-start row (an ordinary overnight gap) -- this is what a raw multi-session
    `close` panel looks like before stitching. No special session reset."""
    n_rows = int(day_offsets[-1])
    returns = rng.standard_normal((n_rows, n_symbols)) * sigma
    for start in day_offsets[1:-1]:
        returns[start] += rng.standard_normal(n_symbols) * gap_sigma
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return close.astype(np.float64)


def _ohlc_fixture(
    rng: np.random.Generator, n_rows: int, n_symbols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random OHLC-consistent (high >= close >= low > 0, high > low) panel."""
    close = np.abs(np.cumsum(rng.standard_normal((n_rows, n_symbols)) * 0.5, axis=0)) + 50.0
    high = close + rng.uniform(0.1, 1.5, size=(n_rows, n_symbols))
    low = close - rng.uniform(0.1, 1.5, size=(n_rows, n_symbols))
    low = np.clip(low, 0.01, None)
    return close.astype(np.float64), high.astype(np.float64), low.astype(np.float64)


def _orthonormal_pair(n_symbols: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two sample-mean-0, sample-variance-1, sample-orthogonal vectors x, z. For any rho in
    [-1, 1], y = rho*x + sqrt(1-rho**2)*z has SAMPLE Pearson correlation with x EXACTLY rho
    (an exact linear-algebra identity, not an approximation) -- used to build an IC-decay
    fixture with an analytically known target correlation per horizon (obligation 12)."""
    raw_x = np.arange(1, n_symbols + 1, dtype=np.float64)
    x = raw_x - raw_x.mean()
    x = x / np.sqrt(np.mean(x**2))
    rng = np.random.default_rng(seed)
    raw_z = rng.standard_normal(n_symbols)
    z = raw_z - raw_z.mean()
    z = z - (np.dot(z, x) / np.dot(x, x)) * x
    z = z / np.sqrt(np.mean(z**2))
    return x, z


# ---------------------------------------------------------------------------
# Obligations 1-2: stitch_overnight_gaps recurrence
# ---------------------------------------------------------------------------


def test_obligation_1_within_session_log_returns_unchanged():
    """Obligation 1: within-session log-returns of the stitched path are bit-for-bit
    identical (`==`) to the raw close's log-returns."""
    rng = np.random.default_rng(1)
    day_offsets = _irregular_day_offsets()
    close = _lognormal_close(rng, day_offsets, n_symbols=3)

    stitched = persistence.stitch_overnight_gaps(close, day_offsets)

    diff_close = np.diff(np.log(close), axis=0)
    diff_stitched = np.diff(np.log(stitched), axis=0)

    n_rows = close.shape[0]
    boundary_rows = set(int(x) for x in day_offsets[1:-1])
    # diff_close[i] / diff_stitched[i] is the return arriving at row (i + 1).
    within_session_mask = np.array(
        [(t not in boundary_rows) for t in range(1, n_rows)], dtype=bool
    )

    assert np.array_equal(
        diff_stitched[within_session_mask], diff_close[within_session_mask]
    ), "within-session log-returns must be EXACTLY unchanged by stitching (bit for bit)"


def test_obligation_2_cross_session_log_diff_is_exactly_zero():
    """Obligation 2: stitching sets the cross-session log-difference to exactly 0.0 at every
    session boundary."""
    rng = np.random.default_rng(2)
    day_offsets = _irregular_day_offsets()
    close = _lognormal_close(rng, day_offsets, n_symbols=3)
    stitched = persistence.stitch_overnight_gaps(close, day_offsets)

    diff_stitched = np.diff(np.log(stitched), axis=0)
    boundary_rows = day_offsets[1:-1]
    boundary_diff_idx = boundary_rows - 1  # diff_stitched[t-1] is the return arriving at row t

    assert np.all(diff_stitched[boundary_diff_idx, :] == 0.0), (
        "cross-session log-difference of the stitched path must be EXACTLY 0.0 at every "
        "session boundary"
    )


# ---------------------------------------------------------------------------
# Obligation 3: rule-6 guard -- stitching must never manufacture a bar
# ---------------------------------------------------------------------------


def test_obligation_3_interior_nan_never_manufactured_or_filled():
    """Obligation 3: a NaN in `close` produces NaN in `stitched` at the SAME index -- the
    stitch never forward-fills / manufactures a bar (rule 6). Interior NaNs, not edges."""
    rng = np.random.default_rng(3)
    day_offsets = _irregular_day_offsets()
    close = _lognormal_close(rng, day_offsets, n_symbols=2)
    close[10, 0] = np.nan  # interior of the 60-bar (Muhurat-shaped) session
    close[200, 1] = np.nan  # interior of the 105-bar session
    close[400, 0] = np.nan  # interior of the 375-bar session

    stitched = persistence.stitch_overnight_gaps(close, day_offsets)

    nan_positions = np.isnan(close)
    assert nan_positions.any(), "fixture precondition: at least one interior NaN required"
    assert np.all(np.isnan(stitched[nan_positions])), (
        "rule 6: every close-NaN index must be stitched-NaN at the SAME index -- a stitched "
        "value produced where close is NaN would mean the stitch silently became a "
        "forward-fill"
    )


# ---------------------------------------------------------------------------
# Obligation 4: the arithmetic-impossibility regression test
# ---------------------------------------------------------------------------


def test_obligation_4_stitched_hurst_majority_finite_where_day_bounded_is_all_nan():
    """Obligation 4: day-bounded rolling_hurst(window=390) is ALL-NaN when no session reaches
    390 bars (measured arithmetic impossibility); hurst_on_stitched on the SAME data, on the
    stitched path without day bounds, is MAJORITY finite. Irregular sessions (60, 105, 375,
    375), none >= 390 -- rule 5."""
    rng = np.random.default_rng(4)
    day_offsets = _four_session_day_offsets()
    close = _lognormal_close(rng, day_offsets, n_symbols=2)

    lengths = np.diff(day_offsets)
    assert lengths.max() < 390, "fixture precondition: no session may reach 390 bars"

    day_bounded = persistence.rolling_hurst(
        close, window=390, day_offsets=day_offsets, warn_short_window=False
    )
    assert np.all(np.isnan(day_bounded)), (
        "a 390-bar window cannot fit inside any session here (max session length "
        f"{lengths.max()} < 390); day-bounded rolling_hurst must be all-NaN -- this must "
        "fail loudly if day-bounding is ever silently reintroduced into the stitched path"
    )

    stitched_result = persistence.hurst_on_stitched(close, day_offsets, window=390)
    n_finite = int(np.isfinite(stitched_result).sum())
    assert n_finite > stitched_result.size // 2, (
        f"stitched-path Hurst must be MAJORITY finite ({n_finite}/{stitched_result.size} "
        "finite); the day-bounded call on the same data gave 0 finite"
    )


# ---------------------------------------------------------------------------
# Obligations 5-6: breakout_strength
# ---------------------------------------------------------------------------


def test_obligation_5_breakout_strength_sign_matches_breakout_up():
    """Obligation 5: `breakout_strength > 0` is elementwise identical to `breakout_up` on
    random OHLC data, wherever `breakout_strength` is finite."""
    rng = np.random.default_rng(5)
    day_offsets = _irregular_day_offsets()
    close, high, low = _ohlc_fixture(rng, int(day_offsets[-1]), n_symbols=3)
    window = 20

    strength = core.breakout_strength(close, high, low, window, day_offsets=day_offsets)
    up = core.breakout_up(close, high, window, day_offsets=day_offsets)

    finite = np.isfinite(strength)
    assert finite.any(), "fixture must produce at least one comparable position"
    assert np.array_equal(strength[finite] > 0, up[finite])


def test_obligation_6_breakout_strength_signed_and_nonzero_below_level():
    """Obligation 6: `breakout_strength` is signed (negative) and non-zero when price sits
    below the breakout level, where `breakout_up` is False."""
    window = 5
    n_rows = 30
    high = np.full((n_rows, 1), 110.0)
    low = np.full((n_rows, 1), 100.0)
    close = np.full((n_rows, 1), 105.0)
    # After the warm-up window, price drops well below the prior-window high of 110.
    high[window:] = 92.0
    low[window:] = 88.0
    close[window:] = 90.0

    strength = core.breakout_strength(close, high, low, window, day_offsets=None)
    up = core.breakout_up(close, high, window, day_offsets=None)

    probe = slice(window, window + 5)
    assert not up[probe, 0].any()
    assert np.all(strength[probe, 0] < 0)
    assert np.all(strength[probe, 0] != 0.0)


# ---------------------------------------------------------------------------
# Obligation 7: Garman-Klass and Rogers-Satchell
# ---------------------------------------------------------------------------


def test_obligation_7_garman_klass_and_rogers_satchell_match_hand_computation():
    """Obligation 7: GK and RS match hand-computed values, propagate NaN on the
    parkinson_volatility convention, and respect day_offsets."""
    open_ = np.array([[100.0], [102.0], [101.0], [103.0]])
    high = np.array([[103.0], [104.0], [102.5], [105.0]])
    low = np.array([[99.0], [100.5], [99.5], [102.0]])
    close = np.array([[101.0], [101.5], [102.0], [104.0]])
    window = 2

    gk = core.garman_klass_volatility(open_, high, low, close, window)
    rs = core.rogers_satchell_volatility(open_, high, low, close, window)

    hl_term = 0.5 * np.log(high / low) ** 2
    co_term = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    gk_term = hl_term - co_term
    rs_term = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(
        low / open_
    )

    expected_gk = np.full((4, 1), np.nan)
    expected_rs = np.full((4, 1), np.nan)
    for t in range(window - 1, 4):
        expected_gk[t, 0] = np.sqrt(np.mean(gk_term[t - window + 1 : t + 1, 0]))
        expected_rs[t, 0] = np.sqrt(np.mean(rs_term[t - window + 1 : t + 1, 0]))

    finite_gk = np.isfinite(expected_gk[:, 0])
    assert finite_gk.any()
    np.testing.assert_allclose(gk[finite_gk, 0], expected_gk[finite_gk, 0], rtol=1e-10)
    finite_rs = np.isfinite(expected_rs[:, 0])
    assert finite_rs.any()
    np.testing.assert_allclose(rs[finite_rs, 0], expected_rs[finite_rs, 0], rtol=1e-10)

    # NaN propagation: an invalid bar (high < low) becomes NaN and poisons any window
    # containing it, on the parkinson_volatility convention.
    bad_high = high.copy()
    bad_low = low.copy()
    bad_high[1, 0] = 50.0
    bad_low[1, 0] = 60.0  # high < low at row 1 -> invalid bar
    gk_bad = core.garman_klass_volatility(open_, bad_high, bad_low, close, window)
    assert np.isnan(gk_bad[1, 0])
    assert np.isnan(gk_bad[2, 0])  # window [1, 2] contains the poisoned bar

    # day_offsets respected: two 2-bar sessions makes the first row of each session NaN
    # (window=2 cannot look back across the session boundary).
    day_offsets = np.array([0, 2, 4])
    gk_sessioned = core.garman_klass_volatility(
        open_, high, low, close, window, day_offsets=day_offsets
    )
    assert np.isnan(gk_sessioned[0, 0])
    assert np.isnan(gk_sessioned[2, 0])


# ---------------------------------------------------------------------------
# Obligation 8: SIGMA_FLOOR derivation
# ---------------------------------------------------------------------------


def test_obligation_8_sigma_floor_has_recorded_derivation_and_is_not_a_round_number():
    """Obligation 8: SIGMA_FLOOR has a recorded measured derivation beside its value and is
    not a hand-picked round number (rule 8)."""
    floor = core.SIGMA_FLOOR
    assert isinstance(floor, float)
    round_numbers = {0.0, 0.001, 0.01, 0.1, 0.05, 0.005, 0.0001, 1e-5, 1e-6}
    assert floor not in round_numbers, (
        f"SIGMA_FLOOR={floor} looks like a hand-picked round number, not a measured "
        "derivation (rule 8)"
    )

    src = inspect.getsource(core)
    idx = src.index("SIGMA_FLOOR")
    nearby = src[idx : idx + 800].lower()
    derivation_markers = ("percentile", "measured", "derivation", "distribution")
    assert any(marker in nearby for marker in derivation_markers), (
        "SIGMA_FLOOR must have a recorded measured derivation in the source beside its "
        "value (rule 8) -- a bare literal is not enough"
    )

    sigma_ewma = np.array([[1e-8, 0.05], [floor * 2.0, np.nan]])
    risk = core.sigma_risk(sigma_ewma, floor)
    assert risk[0, 0] == floor
    assert risk[0, 1] == pytest.approx(0.05)
    assert risk[1, 0] == pytest.approx(floor * 2.0)
    assert np.isnan(risk[1, 1])


# ---------------------------------------------------------------------------
# Obligation 9: efficiency_ratio
# ---------------------------------------------------------------------------


def test_obligation_9_efficiency_ratio_ramp_one_zigzag_near_zero():
    """Obligation 9: efficiency_ratio is exactly 1.0 on a monotone ramp and near 0 on an
    equal-step zig-zag."""
    window = 10
    n_rows = 40

    ramp = (np.arange(n_rows, dtype=np.float64) + 100.0).reshape(-1, 1)
    er_ramp = core.efficiency_ratio(ramp, window)
    assert np.all(er_ramp[window:, 0] == 1.0)

    zigzag = np.empty((n_rows, 1), dtype=np.float64)
    zigzag[0, 0] = 100.0
    for t in range(1, n_rows):
        zigzag[t, 0] = zigzag[t - 1, 0] + (1.0 if t % 2 == 1 else -1.0)
    er_zigzag = core.efficiency_ratio(zigzag, window)
    assert np.all(er_zigzag[window:, 0] < 0.2)


# ---------------------------------------------------------------------------
# Obligations 10-13: research.ic
# ---------------------------------------------------------------------------


def test_obligation_10_information_coefficient_spearman_perfect_rank_monotone():
    """Obligation 10: information_coefficient(method="spearman") is exactly 1.0 on a
    perfectly rank-monotone fixture, and -1.0 when reversed. >= 5 symbols (min_names trap)."""
    from nifty_quant.research import ic

    n_bars, n_symbols = 4, 6
    base = np.arange(1, n_symbols + 1, dtype=np.float64)
    feature = np.tile(base, (n_bars, 1))
    fwd = feature**3 + np.arange(n_bars, dtype=np.float64).reshape(-1, 1) * 1000.0
    day_offsets = np.array([0, n_bars])

    result = ic.information_coefficient(feature, fwd, day_offsets, method="spearman")
    assert result.mean == pytest.approx(1.0, abs=1e-9)

    fwd_reversed = -feature
    result_rev = ic.information_coefficient(
        feature, fwd_reversed, day_offsets, method="spearman"
    )
    assert result_rev.mean == pytest.approx(-1.0, abs=1e-9)


def test_obligation_11_ic_is_cross_sectional_not_pooled():
    """Obligation 11: IC is computed cross-sectionally per bar, NOT pooled across bars.
    Fixture where pooled and cross-sectional answers DIFFER; assert the cross-sectional one."""
    from nifty_quant.research import ic

    # Each bar is individually perfectly rank-monotone increasing (per-bar spearman = +1),
    # but the LEVELS shift between bars so the pooled (bar-boundary-ignoring) ranking scrambles:
    # large feature values in bar 1 pair with SMALL fwd values from bar 1's own low range.
    feature = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ]
    )
    fwd = np.array(
        [
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    day_offsets = np.array([0, 2])

    pooled_rho, _ = scipy.stats.spearmanr(feature.ravel(), fwd.ravel())
    assert abs(pooled_rho - 1.0) > 1.0, (
        "fixture precondition: pooled spearman must NOT be 1.0, otherwise this test cannot "
        "distinguish pooled from cross-sectional"
    )

    result = ic.information_coefficient(feature, fwd, day_offsets, method="spearman")
    assert result.mean == pytest.approx(1.0, abs=1e-9), (
        f"cross-sectional per-bar IC must be +1.0 (every bar is perfectly rank-monotone); "
        f"got {result.mean}, which looks like the POOLED answer ({pooled_rho}) leaked through"
    )


def test_obligation_12_ic_decay_returns_grid_and_matches_known_half_life():
    """Obligation 12: ic_decay returns the full horizon grid (not a scalar), and its fitted
    half-life matches a synthetic exponentially-decaying IC by construction.

    Construction: only row 0 has cross-sectional feature variation (all later rows are
    constant across symbols -> zero cross-sectional variance -> undefined/NaN correlation,
    excluded from the aggregate), so there is exactly ONE valid origin bar per horizon, and
    its target correlation is set EXACTLY via the exact sample-correlation identity in
    `_orthonormal_pair` -- IC(h) should equal the target rho(h) up to floating-point/estimator
    detail, not sampling noise. The 25% tolerance below is a sanity check on the fit, not a
    claim of bit-exactness (Pearson vs Spearman on the same construction differ slightly).
    """
    from nifty_quant.research import ic

    n_symbols = 20
    true_half_life = 2.0
    rho0 = 0.9
    horizons = [1, 2, 4, 8]
    max_h = max(horizons)
    n_rows = max_h + 1
    day_offsets = np.array([0, n_rows])

    x, z = _orthonormal_pair(n_symbols, seed=12)

    feature = np.zeros((n_rows, n_symbols), dtype=np.float64)
    feature[0, :] = x  # every other row is constant -> zero cross-sectional variance

    close = np.full((n_rows, n_symbols), 100.0, dtype=np.float64)
    for h in range(1, n_rows):
        rho_h = rho0 * (0.5 ** (h / true_half_life))
        y_h = rho_h * x + np.sqrt(max(1.0 - rho_h**2, 0.0)) * z
        close[h, :] = 100.0 * np.exp(y_h * 0.05)

    result = ic.ic_decay(feature, close, day_offsets, horizons=horizons)

    assert list(result.horizons) == list(horizons)
    assert np.ndim(result.ic) >= 1
    assert len(result.ic) == len(horizons)
    assert result.half_life == pytest.approx(true_half_life, rel=0.25)


def test_obligation_13_ic_se_differs_from_naive_iid_formula():
    """Obligation 13: IC aggregate SE comes from the overlap-aware block-bootstrap path --
    assert it differs from the naive iid SE on an overlapping-horizon fixture."""
    from nifty_quant.research import ic

    n_rows = 300
    n_symbols = 6
    horizon = 20  # long relative to n_rows -> strong overlap-induced serial dependence
    day_offsets = np.array([0, n_rows])

    rng = np.random.default_rng(13)
    base_rank = np.arange(1, n_symbols + 1, dtype=np.float64)
    feature = np.tile(base_rank, (n_rows, 1))

    one_bar_signal = 0.001 * (base_rank - base_rank.mean())
    one_bar_ret = (
        np.tile(one_bar_signal, (n_rows, 1))
        + rng.standard_normal((n_rows, n_symbols)) * 0.01
    )

    fwd = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    for t in range(n_rows - horizon):
        fwd[t, :] = one_bar_ret[t + 1 : t + 1 + horizon, :].sum(axis=0)

    result = ic.information_coefficient(feature, fwd, day_offsets, method="spearman")

    finite_ic = result.per_bar[np.isfinite(result.per_bar)]
    assert finite_ic.size >= 30, "fixture must produce enough per-bar IC observations"
    naive_se = finite_ic.std(ddof=1) / np.sqrt(finite_ic.size)

    assert not np.isclose(result.se, naive_se, rtol=1e-6), (
        f"IC aggregate SE ({result.se}) must come from the overlap-aware block-bootstrap "
        f"path, not the naive iid formula ({naive_se}), on data constructed to have strong "
        "overlap-induced dependence"
    )


# ---------------------------------------------------------------------------
# Obligation 14: vwap_reversion uses the shared bars_since_open
# ---------------------------------------------------------------------------


def test_obligation_14_vwap_reversion_uses_shared_bars_since_open():
    """Obligation 14: vwap_reversion uses market.bars_since_open and no longer hand-rolls
    its own."""
    from nifty_quant.strategy.plugins import vwap_reversion

    src = inspect.getsource(vwap_reversion)
    assert "bars_since_open" in src

    # This is the exact hand-rolled line that exists TODAY (vwap_reversion.py:~144-149),
    # before the D6 fix. Its presence means the module has NOT been migrated.
    assert "bars_once = rows - day_start_rows" not in src, (
        "vwap_reversion must no longer hand-roll bars_since_open (spec D6) -- found the "
        "pre-fix hand-rolled computation still present"
    )
    assert (
        "from nifty_quant.features import market" in src
        or "from nifty_quant.features.market import" in src
    ), "vwap_reversion must import the shared market.bars_since_open helper"
