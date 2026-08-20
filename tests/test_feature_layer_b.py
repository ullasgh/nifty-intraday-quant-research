"""Independent test suite B for specs/feature_layer.md, written from the spec alone.

Rule 1: this suite does not see the implementation (none exists yet for the new
items) and does not see the sibling suite. Expected to be RED.

Obligation -> test name map is restated in each test's docstring first line, and
summarised in the final report to the orchestrator.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import stats

from nifty_quant.features import core, market, persistence

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _irregular_day_offsets(session_lengths: list[int]) -> np.ndarray:
    """Boundary-format day_offsets: [0, len0, len0+len1, ...]."""
    return np.concatenate([[0], np.cumsum(session_lengths)]).astype(np.int64)


def _row_day_id(day_offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(day_offsets) - 1), np.diff(day_offsets))


def _geometric_price_path(
    n_rows: int, n_cols: int, *, sigma: float = 0.01, seed: int
) -> np.ndarray:
    """Positive float64 price panel from a small-sigma multiplicative random walk."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=(n_rows, n_cols))
    log_path = np.cumsum(steps, axis=0)
    return (100.0 * np.exp(log_path)).astype(np.float64)


# ===========================================================================
# D1 -- stitch_overnight_gaps (obligations 1, 2, 3)
# ===========================================================================


class TestObligation1WithinSessionLogReturnsUnchanged:
    """Obligation 1: stitch_overnight_gaps -> test_within_session_log_returns_bit_exact."""

    def test_within_session_log_returns_bit_exact(self):
        session_lengths = [60, 105, 375]
        day_offsets = _irregular_day_offsets(session_lengths)
        n_rows = int(day_offsets[-1])
        close = _geometric_price_path(n_rows, 3, seed=1)

        stitched = persistence.stitch_overnight_gaps(close, day_offsets)

        day_id = _row_day_id(day_offsets)
        log_close = np.log(close)
        log_stitched = np.log(stitched)

        diff_close = log_close[1:] - log_close[:-1]
        diff_stitched = log_stitched[1:] - log_stitched[:-1]

        within_session = day_id[1:] == day_id[:-1]

        # AMENDMENT 4: NOT bit-for-bit. Stitching rescales every session after the
        # first, which in log space is a shared additive offset `c`, and in IEEE-754
        # `(a+c)-(b+c) != a-b` for ~62%% of random triples. Exactness would require a
        # power-of-two scale (the only factor that leaves the mantissa untouched), and
        # that cannot achieve LEVEL CONTINUITY -- the whole point of stitching.
        # Worst measured deviation is 4.44e-16, ~1e12x smaller than a genuine
        # mis-stitch (which would show the full ~1e-2 overnight gap), so this
        # tolerance cannot hide a real defect. Obligations 2 and 3 stay EXACT.
        assert np.allclose(
            diff_stitched[within_session],
            diff_close[within_session],
            rtol=0,
            atol=1e-14,
        ), "within-session log-returns must survive stitching to within round-off"


class TestObligation2CrossSessionLogDiffIsZero:
    """Obligation 2: stitch_overnight_gaps -> test_cross_session_log_diff_is_exactly_zero."""

    def test_cross_session_log_diff_is_exactly_zero(self):
        session_lengths = [60, 105, 375]
        day_offsets = _irregular_day_offsets(session_lengths)
        n_rows = int(day_offsets[-1])
        close = _geometric_price_path(n_rows, 3, seed=2)

        stitched = persistence.stitch_overnight_gaps(close, day_offsets)

        day_id = _row_day_id(day_offsets)
        log_stitched = np.log(stitched)
        diff_stitched = log_stitched[1:] - log_stitched[:-1]

        cross_session = day_id[1:] != day_id[:-1]
        assert cross_session.sum() == len(session_lengths) - 1

        assert np.all(diff_stitched[cross_session] == 0.0), (
            "cross-session log-difference must be EXACTLY 0.0 at every boundary"
        )


class TestObligation3NaNNeverManufacturesABar:
    """Obligation 3: stitch_overnight_gaps -> test_interior_nan_stays_nan_rule6."""

    def test_interior_nan_stays_nan_rule6(self):
        session_lengths = [60, 105, 375]
        day_offsets = _irregular_day_offsets(session_lengths)
        n_rows = int(day_offsets[-1])
        close = _geometric_price_path(n_rows, 2, seed=3)

        # Interior NaN in the middle of the second (105-bar) session, column 0.
        nan_row = 60 + 52
        close_with_gap = close.copy()
        close_with_gap[nan_row, 0] = np.nan

        stitched = persistence.stitch_overnight_gaps(close_with_gap, day_offsets)

        assert np.isnan(stitched[nan_row, 0]), (
            "a NaN in close must produce a NaN in stitched at the SAME index "
            "(rule 6: the stitch must never manufacture a bar)"
        )

        # Anti-vacuity guard: an all-NaN-output bug would also satisfy the
        # assertion above. Confirm a distant, unaffected bar in a DIFFERENT
        # session is still finite, so the function isn't just NaN-everywhere.
        far_row = day_offsets[2] + 10  # early in the third (375-bar) session
        assert np.isfinite(stitched[far_row, 0])
        # Column 1 was never touched by the injected NaN; must be fully finite.
        assert np.all(np.isfinite(stitched[:, 1]))


# ===========================================================================
# D1 -- hurst_on_stitched (obligation 4, THE load-bearing one)
# ===========================================================================


class TestObligation4HurstArithmeticImpossibilityRegression:
    """Obligation 4: hurst_on_stitched ->
    test_stitched_hurst_majority_finite_vs_day_bounded_all_nan.

    This is the arithmetic-impossibility regression test. window=390 cannot fit
    inside any of the irregular sessions used here (60, 105, 375 bars), so the
    day-bounded call on the unstitched path MUST be all-NaN. The stitched,
    non-day-bounded call must be majority-finite. Repeats the irregular
    session pattern 3x (1620 total rows) so the finite fraction clears 50%:
    with window=390, at most (1620 - 390 + 1) = 1231 of 1620 rows (76%) can be
    finite -- comfortably a majority, while every individual session (max 375
    bars) is still smaller than the 390-bar window.
    """

    def test_stitched_hurst_majority_finite_vs_day_bounded_all_nan(self):
        session_lengths = [60, 105, 375] * 3
        assert max(session_lengths) < 390, "fixture must keep every session < window"
        day_offsets = _irregular_day_offsets(session_lengths)
        n_rows = int(day_offsets[-1])
        close = _geometric_price_path(n_rows, 1, sigma=0.005, seed=4)

        # Day-bounded, UNSTITCHED: arithmetic impossibility -> all NaN.
        day_bounded = persistence.rolling_hurst(
            close, window=390, day_offsets=day_offsets, warn_short_window=False
        )
        assert np.sum(np.isfinite(day_bounded)) == 0, (
            "390-bar window cannot fit inside any session <= 375 bars; "
            "day-bounded unstitched Hurst must be all-NaN"
        )

        # Stitched, causal call path: majority finite.
        stitched_hurst = persistence.hurst_on_stitched(close, day_offsets, window=390)
        n_finite = int(np.sum(np.isfinite(stitched_hurst)))
        assert n_finite > n_rows // 2, (
            f"expected a MAJORITY-finite result on the stitched path, "
            f"got {n_finite}/{n_rows} finite"
        )


# ===========================================================================
# D2 -- breakout_strength (obligations 5, 6)
# ===========================================================================


class TestObligation5BreakoutStrengthSignMatchesBoolean:
    """Obligation 5: breakout_strength -> test_breakout_strength_positive_matches_breakout_up."""

    def test_breakout_strength_positive_matches_breakout_up(self):
        n_rows, n_cols, window = 200, 4, 20
        day_offsets = np.array([0, n_rows], dtype=np.int64)
        rng = np.random.default_rng(5)

        close = _geometric_price_path(n_rows, n_cols, sigma=0.01, seed=5)
        high = close * (1.0 + np.abs(rng.normal(0.0, 0.003, size=close.shape)))
        low = close * (1.0 - np.abs(rng.normal(0.0, 0.003, size=close.shape)))

        strength = core.breakout_strength(close, high, low, window, day_offsets=day_offsets)
        is_breakout = core.breakout_up(close, high, window, day_offsets=day_offsets)

        finite = np.isfinite(strength)
        assert finite.sum() > 0, "fixture must produce some finite breakout_strength values"

        assert np.array_equal(
            (strength[finite] > 0), np.asarray(is_breakout)[finite]
        ), "breakout_strength > 0 must be identical to breakout_up, elementwise"


class TestObligation6BreakoutStrengthSignedBelowLevel:
    """Obligation 6: breakout_strength -> test_breakout_strength_negative_when_below_level."""

    def test_breakout_strength_negative_when_below_level(self):
        window = 10
        n_rows = 40
        day_offsets = np.array([0, n_rows], dtype=np.int64)

        # Spike up early (sets a high prior-window high), then a hard decline
        # so later closes sit well below the prior-window level.
        close = np.concatenate(
            [
                np.linspace(100.0, 130.0, 15),
                np.linspace(130.0, 90.0, n_rows - 15),
            ]
        ).reshape(-1, 1)
        high = close * 1.001
        low = close * 0.999

        strength = core.breakout_strength(close, high, low, window, day_offsets=day_offsets)
        is_breakout = core.breakout_up(close, high, window, day_offsets=day_offsets)

        tail = strength[-5:, 0]
        assert np.all(np.isfinite(tail))
        assert np.all(tail < 0.0), "price well below the prior level must give strength < 0"
        assert np.all(tail != 0.0)
        assert not np.any(np.asarray(is_breakout)[-5:, 0])


# ===========================================================================
# D3 -- Garman-Klass / Rogers-Satchell / sigma_floor (obligations 7, 8)
# ===========================================================================


class TestObligation7GarmanKlassRogersSatchell:
    """Obligation 7: garman_klass_volatility / rogers_satchell_volatility ->
    test_gk_rs_match_hand_computed_and_respect_nan_and_day_offsets,
    test_window_mean_is_clipped_at_zero_not_individual_bar_terms.

    Formulas pinned in AMENDMENT 1 item 1 (both as variance per bar, averaged
    over the window, then sqrt'd, matching parkinson_volatility's scaling/NaN/
    day_offsets convention):
        GK: sigma_sq = mean_over_window[ 0.5*ln(H/L)**2 - (2*ln(2)-1)*ln(C/O)**2 ]
        RS: sigma_sq = mean_over_window[ ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ]
    with window=1 so the rolling mean reduces to a single-bar value directly
    comparable to the hand formula.
    """

    def test_gk_rs_match_hand_computed_and_respect_nan_and_day_offsets(self):
        # Two sessions of 3 bars each; column 0 only.
        day_offsets = np.array([0, 3, 6], dtype=np.int64)
        open_ = np.array([[100.0], [101.0], [99.0], [100.0], [102.0], [98.0]])
        high = np.array([[102.0], [103.0], [100.0], [101.0], [104.0], [99.5]])
        low = np.array([[99.0], [100.5], [97.5], [98.5], [101.0], [96.0]])
        close = np.array([[101.0], [100.0], [98.5], [100.5], [103.0], [97.0]])

        log2 = np.log(2.0)
        gk_sq = 0.5 * np.log(high / low) ** 2 - (2 * log2 - 1) * np.log(close / open_) ** 2
        expected_gk = np.sqrt(np.maximum(gk_sq, 0.0))

        rs_sq = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(
            low / open_
        )
        expected_rs = np.sqrt(np.maximum(rs_sq, 0.0))

        gk = core.garman_klass_volatility(
            open_, high, low, close, window=1, day_offsets=day_offsets
        )
        rs = core.rogers_satchell_volatility(
            open_, high, low, close, window=1, day_offsets=day_offsets
        )

        np.testing.assert_allclose(gk, expected_gk, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(rs, expected_rs, rtol=1e-10, atol=1e-12)

        # NaN propagation, same convention as parkinson_volatility: a bad
        # bar's window becomes NaN.
        open_nan = open_.copy()
        open_nan[1, 0] = np.nan
        gk_nan = core.garman_klass_volatility(
            open_nan, high, low, close, window=2, day_offsets=day_offsets
        )
        rs_nan = core.rogers_satchell_volatility(
            open_nan, high, low, close, window=2, day_offsets=day_offsets
        )
        # Row 1 itself, and row 2 (whose window=2 lookback includes row 1),
        # must be NaN; row 0 (window not yet full) is also NaN regardless.
        assert np.isnan(gk_nan[1, 0])
        assert np.isnan(gk_nan[2, 0])
        assert np.isnan(rs_nan[1, 0])
        assert np.isnan(rs_nan[2, 0])
        # Second session (rows 3-5) is untouched by the first session's NaN:
        # day_offsets must confine the corruption to session 1.
        assert np.isfinite(gk_nan[5, 0])
        assert np.isfinite(rs_nan[5, 0])

    def test_window_mean_is_clipped_at_zero_not_individual_bar_terms(self):
        """AMENDMENT 1 item 1: clip the WINDOW MEAN at zero, never a per-bar term.

        Row 0 ("stress" OHLC: a narrow high/low range with a far-apart
        open/close) has a strongly NEGATIVE per-bar term for both GK and RS.
        Row 1 (a normal, wide-range bar) has a strongly positive term, large
        enough that the TRUE two-bar mean is still positive. A per-bar-clipped
        implementation would instead floor row 0's term to 0 before
        averaging, giving a measurably different (larger) result -- exactly
        the upward bias the amendment says destroys the estimator's
        efficiency advantage.
        """
        day_offsets = np.array([0, 2], dtype=np.int64)
        open_ = np.array([[95.0], [105.0]])
        high = np.array([[100.5], [115.19]])
        low = np.array([[99.5], [100.0]])
        close = np.array([[105.0], [105.0]])

        log2 = np.log(2.0)
        gk_terms = 0.5 * np.log(high / low) ** 2 - (2 * log2 - 1) * np.log(close / open_) ** 2
        rs_terms = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(
            low / open_
        )
        assert gk_terms[0, 0] < 0.0 and rs_terms[0, 0] < 0.0, "fixture sanity: row 0 is negative"
        assert gk_terms[1, 0] > 0.0 and rs_terms[1, 0] > 0.0, "fixture sanity: row 1 is positive"

        true_mean_gk = float(np.mean(gk_terms))
        true_mean_rs = float(np.mean(rs_terms))
        assert true_mean_gk > 0.0 and true_mean_rs > 0.0, "fixture sanity: true mean is positive"

        wrong_clipped_mean_gk = float(np.mean(np.maximum(gk_terms, 0.0)))
        wrong_clipped_mean_rs = float(np.mean(np.maximum(rs_terms, 0.0)))
        # These must genuinely differ, or the test has no power.
        assert abs(true_mean_gk - wrong_clipped_mean_gk) > 1e-4
        assert abs(true_mean_rs - wrong_clipped_mean_rs) > 1e-4

        gk = core.garman_klass_volatility(
            open_, high, low, close, window=2, day_offsets=day_offsets
        )
        rs = core.rogers_satchell_volatility(
            open_, high, low, close, window=2, day_offsets=day_offsets
        )

        np.testing.assert_allclose(gk[1, 0], np.sqrt(true_mean_gk), rtol=1e-10)
        np.testing.assert_allclose(rs[1, 0], np.sqrt(true_mean_rs), rtol=1e-10)
        assert gk[1, 0] != pytest.approx(np.sqrt(wrong_clipped_mean_gk), rel=1e-6)
        assert rs[1, 0] != pytest.approx(np.sqrt(wrong_clipped_mean_rs), rel=1e-6)

    def test_window_mean_negative_floors_to_zero_not_nan(self):
        """When even the WINDOW mean is negative, the result is 0.0, not NaN."""
        day_offsets = np.array([0, 2], dtype=np.int64)
        # Two "stress" bars, both individually negative-term, whose mean stays negative.
        open_ = np.array([[95.0], [96.0]])
        high = np.array([[100.5], [100.4]])
        low = np.array([[99.5], [99.6]])
        close = np.array([[105.0], [104.0]])

        gk = core.garman_klass_volatility(
            open_, high, low, close, window=2, day_offsets=day_offsets
        )
        rs = core.rogers_satchell_volatility(
            open_, high, low, close, window=2, day_offsets=day_offsets
        )

        assert gk[1, 0] == pytest.approx(0.0, abs=1e-12)
        assert rs[1, 0] == pytest.approx(0.0, abs=1e-12)
        assert not np.isnan(gk[1, 0])
        assert not np.isnan(rs[1, 0])


class TestObligation8SigmaFloorDerivationRecorded:
    """Obligation 8: sigma_floor -> test_sigma_floor_has_recorded_derivation_and_is_not_round.

    Mirrors the HURST_MIN_WINDOW convention in features/persistence.py: a
    module-level constant immediately followed by a docstring recording its
    derivation. Rule 8 requires the derivation to be measured, not chosen.
    Module placement (core.SIGMA_FLOOR, core.sigma_risk) and the
    `sigma_risk(sigma_ewma, *, floor=SIGMA_FLOOR)` keyword-only signature are
    pinned in AMENDMENT 1 items 2-3.
    """

    def test_sigma_floor_has_recorded_derivation_and_is_not_round(self):
        assert hasattr(core, "SIGMA_FLOOR"), "expected a module-level core.SIGMA_FLOOR constant"
        value = core.SIGMA_FLOOR
        assert isinstance(value, float)
        assert value > 0.0

        source = inspect.getsource(core)
        idx = source.index("SIGMA_FLOOR")
        nearby = source[idx : idx + 800].lower()
        assert any(
            keyword in nearby for keyword in ("percentile", "measured", "distribution")
        ), "SIGMA_FLOOR's derivation must be recorded beside its value (rule 8)"

        # Not a hand-picked round number: a measured low percentile of a real
        # volatility distribution should not land on a tidy few-sig-fig value.
        assert round(value, 4) != value, (
            f"SIGMA_FLOOR={value} looks like a round number chosen by hand, "
            "not derived from a measured distribution"
        )

    def test_sigma_risk_is_elementwise_max_of_ewma_and_floor(self):
        sigma_ewma = np.array([0.0001, 0.05, 0.2, np.nan])
        floor = 0.01
        result = core.sigma_risk(sigma_ewma, floor=floor)
        expected = np.array([floor, 0.05, 0.2, np.nan])
        np.testing.assert_allclose(result[:3], expected[:3])
        assert np.isnan(result[3])

    def test_sigma_risk_default_floor_is_sigma_floor(self):
        sigma_ewma = np.array([0.0])
        result = core.sigma_risk(sigma_ewma)
        assert result[0] == pytest.approx(core.SIGMA_FLOOR)


# ===========================================================================
# D4 -- efficiency_ratio (obligation 9)
# ===========================================================================


class TestObligation9EfficiencyRatioRampAndZigzag:
    """Obligation 9: efficiency_ratio -> test_efficiency_ratio_ramp_is_one_zigzag_is_near_zero,
    test_efficiency_ratio_first_window_minus_one_rows_of_each_session_are_nan.

    AMENDMENT 1 item 4 pins `window` as BARS INCLUSIVE (rolling_max/rolling_std
    convention):
        efficiency_ratio[t] = abs(close[t] - close[t-window+1])
                              / sum(abs(diff(close))[t-window+2 : t+1])
    numerator spans `window` bars, denominator sums `window-1` first
    differences; first `window-1` rows of each session are NaN.
    """

    def test_efficiency_ratio_ramp_is_one_zigzag_is_near_zero(self):
        window = 11  # window - 1 = 10 diffs: even, so a perfect zigzag sums to exactly 0
        n_rows = 32
        day_offsets = np.array([0, n_rows], dtype=np.int64)
        valid = np.arange(n_rows) >= window - 1

        ramp = np.arange(1, n_rows + 1, dtype=np.float64).reshape(-1, 1)
        er_ramp = core.efficiency_ratio(ramp, window, day_offsets=day_offsets)
        np.testing.assert_allclose(er_ramp[valid, 0], 1.0, rtol=0, atol=1e-12)

        # Equal-magnitude zigzag, window-1=10 (even) diffs -> numerator
        # exactly 0 regardless of phase.
        steps = np.tile([1.0, -1.0], n_rows // 2)
        zigzag = (100.0 + np.cumsum(steps)).reshape(-1, 1)
        er_zigzag = core.efficiency_ratio(zigzag, window, day_offsets=day_offsets)
        np.testing.assert_allclose(er_zigzag[valid, 0], 0.0, atol=1e-12)

    def test_efficiency_ratio_first_window_minus_one_rows_of_each_session_are_nan(self):
        window = 11
        day_offsets = np.array([0, 16, 32], dtype=np.int64)
        steps = np.tile([1.0, -1.0], 16)
        close = (100.0 + np.cumsum(steps)).reshape(-1, 1)

        er = core.efficiency_ratio(close, window, day_offsets=day_offsets)

        # Exactly window-1=10 leading NaN rows in EACH session -- pinned, not
        # approximate. A rolling window may not reach back across a boundary.
        assert np.all(np.isnan(er[0:10, 0]))
        assert np.all(np.isfinite(er[10:16, 0]))
        assert np.all(np.isnan(er[16:26, 0]))
        assert np.all(np.isfinite(er[26:32, 0]))


# ===========================================================================
# D5 -- research/ic.py (obligations 10, 11, 12, 13)
# ===========================================================================


def _orthonormal_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two zero-mean, unit-population-variance, mutually orthogonal vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, 2))
    centered = raw - raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(centered)
    scaled = q * np.sqrt(n)
    return scaled[:, 0], scaled[:, 1]


class TestObligation10SpearmanICPerfectRankMonotone:
    """Obligation 10: information_coefficient -> test_spearman_ic_is_plus_one_and_minus_one."""

    def test_spearman_ic_is_plus_one_and_minus_one(self):
        from nifty_quant.research import ic

        n_symbols = 6  # >= min_names=5 trap
        n_rows = 4
        day_offsets = np.array([0, n_rows], dtype=np.int64)

        base = np.arange(n_symbols, dtype=np.float64)
        feature = np.tile(base, (n_rows, 1))
        fwd_up = np.tile(base, (n_rows, 1))
        fwd_down = np.tile(base[::-1].copy(), (n_rows, 1))

        result_up = ic.information_coefficient(
            feature, fwd_up, day_offsets, horizon=1, method="spearman"
        )
        result_down = ic.information_coefficient(
            feature, fwd_down, day_offsets, horizon=1, method="spearman"
        )

        assert result_up.mean == pytest.approx(1.0, abs=1e-9)
        assert result_down.mean == pytest.approx(-1.0, abs=1e-9)
        assert result_up.method == "spearman"
        assert result_up.horizon == 1
        assert result_up.n_bars == n_rows


class TestObligation11CrossSectionalNotPooled:
    """Obligation 11: information_coefficient -> test_ic_is_cross_sectional_not_pooled.

    Fixture where the pooled (flatten-everything) spearman answer and the
    correct cross-sectional (per-bar, then averaged) answer DIFFER. Row 0 is
    perfectly rank-positive, row 1 is perfectly rank-negative at a different
    scale -> cross-sectional average = 0.0 exactly, but pooling the raw
    values across both rows (as if it were one big sample) is dominated by
    row 1's larger magnitude and gives ~0.76, not 0.
    """

    def test_ic_is_cross_sectional_not_pooled(self):
        from nifty_quant.research import ic

        day_offsets = np.array([0, 2], dtype=np.int64)  # 5 symbols: the min_names floor

        feature = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]
        )
        fwd = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0]]
        )

        pooled_rho, _ = stats.spearmanr(feature.flatten(), fwd.flatten())
        assert pooled_rho == pytest.approx(0.7576, abs=1e-3), (
            "sanity check on the fixture itself, not on the function under test"
        )

        result = ic.information_coefficient(
            feature, fwd, day_offsets, horizon=1, method="spearman"
        )

        assert result.mean == pytest.approx(0.0, abs=1e-9), (
            f"cross-sectional per-bar mean must be 0.0 (average of +1 and -1); "
            f"got {result.mean}, which looks like the POOLED answer "
            f"({pooled_rho:.4f}) instead"
        )
        assert abs(result.mean - pooled_rho) > 0.1


class TestObligation12ICDecayHorizonGridAndHalfLife:
    """Obligation 12: ic_decay -> test_ic_decay_returns_full_grid_and_matches_known_half_life.

    Single anchor bar, exact orthogonal-noise construction: for horizon h,
    the cross-sectional Pearson correlation of the h-bar-forward return with
    the feature is EXACTLY rho**h (rho=0.5), because signal and noise are
    constructed to be exactly orthonormal across the 6 symbols. Only the
    anchor row carries a non-NaN feature, so it is the only row that can
    possibly be used -- no other (anchor, horizon) pair can contaminate the
    aggregate. rho=0.5 gives a known half-life of exactly 1.0 bar
    (rho**halflife = 0.5 => halflife = ln(0.5)/ln(rho) = 1.0).
    """

    def test_ic_decay_returns_full_grid_and_matches_known_half_life(self):
        from nifty_quant.research import ic

        n_symbols = 6
        rho = 0.5
        k = 1e-4
        horizons = [1, 2, 3]
        n_rows = max(horizons) + 1
        day_offsets = np.array([0, n_rows], dtype=np.int64)

        s, noise = _orthonormal_pair(n_symbols, seed=42)

        feature = np.full((n_rows, n_symbols), np.nan)
        feature[0, :] = s

        close = np.full((n_rows, n_symbols), np.nan)
        close[0, :] = 100.0
        for h in horizons:
            ret_h = k * (rho**h * s + np.sqrt(1.0 - rho ** (2 * h)) * noise)
            close[h, :] = 100.0 * (1.0 + ret_h)

        result = ic.ic_decay(feature, close, day_offsets, horizons=horizons)

        assert len(result.horizons) == len(horizons), (
            "ic_decay must return the full horizon grid, not a scalar"
        )
        assert list(result.horizons) == horizons
        assert len(result.ic) == len(horizons)
        assert len(result.se) == len(horizons)  # ICDecay.se, pinned in AMENDMENT 1 item 5

        expected_ic = [rho**h for h in horizons]
        np.testing.assert_allclose(result.ic, expected_ic, rtol=0.05)

        expected_half_life = 1.0
        assert result.half_life == pytest.approx(expected_half_life, rel=0.1)


class TestObligation13OverlapAwareSEDiffersFromNaive:
    """Obligation 13: information_coefficient SE -> test_overlap_aware_se_differs_from_naive_iid_se.

    Overlapping h-bar-forward returns (h=5) induce positive autocorrelation
    between consecutive bars' cross-sectional IC, purely as a mechanical
    consequence of sharing (h-1)/h of their underlying one-bar returns. The
    naive iid SE (std of per-bar IC / sqrt(n_bars)) ignores that and must
    differ from -- and, per standard overlap theory, understate -- the
    overlap-aware SE.

    AMENDMENT 1 item 5 pins the required `horizon` keyword and confirms
    `ICResult.se` comes from the specs/overlap_se.md machinery (now
    implemented and verified: false-positive rate 34% -> 7%).
    """

    def test_overlap_aware_se_differs_from_naive_iid_se(self):
        from nifty_quant.research import ic

        n_symbols = 6
        n_rows = 60
        horizon = 5
        day_offsets = np.array([0, n_rows], dtype=np.int64)

        close = _geometric_price_path(n_rows, n_symbols, sigma=0.01, seed=13)
        feature = np.tile(np.arange(n_symbols, dtype=np.float64), (n_rows, 1))

        fwd = np.full((n_rows, n_symbols), np.nan)
        fwd[: n_rows - horizon, :] = (
            close[horizon:, :] / close[: n_rows - horizon, :] - 1.0
        )

        result = ic.information_coefficient(
            feature, fwd, day_offsets, method="pearson", horizon=horizon
        )

        per_bar_ic = np.full(n_rows, np.nan)
        for t in range(n_rows):
            if np.all(np.isfinite(fwd[t, :])):
                r, _ = stats.pearsonr(feature[t, :], fwd[t, :])
                per_bar_ic[t] = r
        valid = np.isfinite(per_bar_ic)
        naive_se = np.std(per_bar_ic[valid], ddof=1) / np.sqrt(valid.sum())

        assert result.se != pytest.approx(naive_se, rel=1e-6), (
            "the overlap-aware SE must differ from the naive iid SE on an "
            "overlapping-horizon fixture"
        )
        assert result.se > naive_se, (
            "overlapping windows induce positive autocorrelation; the naive "
            "iid SE should UNDERSTATE the true (overlap-aware) SE"
        )
        assert result.horizon == horizon
        assert result.method == "pearson"


# ===========================================================================
# D6 -- vwap_reversion wiring (obligation 14)
# ===========================================================================


class TestObligation14VwapReversionUsesMarketBarsSinceOpen:
    """Obligation 14: vwap_reversion -> test_vwap_reversion_imports_market_bars_since_open.

    Strong identity check: the name `bars_since_open` bound inside
    strategy.plugins.vwap_reversion must literally BE
    features.market.bars_since_open (a real import/reuse), not a
    same-named local re-implementation.
    """

    def test_vwap_reversion_imports_market_bars_since_open(self):
        from nifty_quant.strategy.plugins import vwap_reversion

        assert hasattr(vwap_reversion, "bars_since_open"), (
            "vwap_reversion module must import bars_since_open from "
            "nifty_quant.features.market"
        )
        assert vwap_reversion.bars_since_open is market.bars_since_open, (
            "vwap_reversion.bars_since_open must be the SAME function object as "
            "market.bars_since_open, not a duplicated local implementation"
        )
