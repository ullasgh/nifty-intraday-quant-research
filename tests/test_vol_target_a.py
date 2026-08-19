"""Independent test suite A for specs/portfolio_vol_target.md (Phase A track A4).

Written from the spec alone, before any `VolTargetSizer` implementation exists.
Author is one of two independent test writers for this spec; per CLAUDE.md rule 1
this file was written without reading the other author's file or any implementation.

Existing contract read to stay compatible (rule: match GrossNotionalSizer.to_shares):
  - src/nifty_quant/backtest/portfolio.py:64-77   SizingResult (shares, unmet_gross_pct)
  - src/nifty_quant/backtest/portfolio.py:79-224  GrossNotionalSizer.to_shares
  - src/nifty_quant/features/market.py:337-        median_pairwise_correlation

=================================== SPEC DEFECTS ===================================

Findings 1-6 below were reported against the ORIGINAL spec body and are now RESOLVED
by "AMENDMENT 2026-08-19" at the bottom of specs/portfolio_vol_target.md, which the
lead accepted in full and which wins on conflict. Kept here (not deleted) as a record
of what was found and how it was fixed, per the amendment's own instruction to keep
the falsifying algebra "as a regression guard against the spec drifting back."

1. [RESOLVED, amendment #3] VolTargetSizer's constructor is now specified exactly:
   `target_vol_ann` (required, no default), `max_weight=0.10`, `gross=1.0` (a CAP,
   not a target), `min_trade_notional=0.0`, `lot_size: int | None = None`. Tests
   below use this signature.

   NEW FINDING: the amended constructor has no `whole_shares` field at all (unlike
   GrossNotionalSizer), and `lot_size`'s semantics are never explained -- does
   `lot_size=None` mean "fractional shares allowed", "round to 1", or something
   else? Tests 4-7 and 10 in this file were written before the amendment and
   instantiate `VolTargetSizer(..., whole_shares=False)`, a keyword that no longer
   exists on the amended constructor -- those calls will raise TypeError once the
   real class lands, not fail cleanly on the assertions they were meant to check.
   This coordinator round's instructions scoped changes to tests 8/9/11 only
   ("and only that file... as follows", followed by exactly three numbered items),
   so tests 4-7/10 are left as-is per that scope, but this is flagged as a required
   follow-up: either `lot_size` needs a documented meaning that subsumes
   `whole_shares=False` (e.g. `lot_size=1` alone implying whole-share rounding,
   with fractional shares needing some other explicit opt-out), or those five
   tests need a one-line kwarg fix once that's decided.

2. [RESOLVED, amendment #2] `to_shares` always returns `SizingResult`, unconditionally.
   Tests below rely on this.

3. [RESOLVED, amendment #2, implicitly] SizingResult gains the four diagnostic
   fields; amendment #2 doesn't restate the shared-type-vs-subclass question but
   "always returns SizingResult" only makes sense if it's the shared type (a
   VolTargetSizer-only result type would just be its own dataclass, not "gains" on
   the existing one) -- treated as settled.

4. [RESOLVED, amendment #3 + #4] `gross` is now explicitly a cap (not a target),
   and `clip_binding` now explicitly includes capacity water-filling ("capacity is
   the most common binder in this repo (73.7%... unfilled)"). Tests 4-7 (unchanged,
   out of this round's scope) still only exercise bar_traded_value=None, so they
   never touch the capacity-inclusion half of this fix -- that remains untested in
   this file and should be picked up in a future round explicitly targeting
   capacity-driven clip_binding.

5. [MOOT] the amended constructor's `corr` semantics are superseded by the
   endpoint-max fallback (amendment #1); see finding 6 below and the new test 8.

6. [WAS THE HEADLINE FINDING, now amendment #1] Confirmed by the coordinator's own
   independent 2,000-draw check (rho=0 under-estimated true variance in 2000/2000
   same-signed cases) and now fixed spec-side: the fallback is
   `var_conservative = max(sum_i a_i^2, (sum_i a_i)^2)` with `a_i = w_i sigma_i`,
   not `rho = 0`. Test 8 below is rewritten to the new contract; the falsifying
   algebra from the old test 8c is kept as test_8c_regression_guard, re-pointed to
   prove the NEW fallback actually is conservative (unlike the old one) so a future
   spec drift back to "just use rho=0" gets caught again.

7. [RESOLVED, amendment #5, module + signature corrected by amendments #2 and #3]
   Annualization is not the sizer's job. It moves to a named helper,
   `annualization_factor(day_offsets, *, sessions_per_year=252)`, living in
   `nifty_quant.features.core` beside the estimator family (amendment #2 item 2),
   with NO `window` parameter (amendment #3): bars-per-session is a calendar
   property, not a price property, so the estimate is the median of
   `np.diff(day_offsets)` over every session in the panel -- no window constant,
   no lookahead. Superseded findings, kept for the record: this file originally
   ASSUMED `nifty_quant.backtest.portfolio.annualization_factor` with a
   `window` argument; both guesses were wrong and are now corrected.

8. [RESOLVED, amendment #6, named and re-homed by amendment #2 item 2] The
   EWMA-halflife gap is accepted as real and has a required test (11): halflife
   is a declared input to the sizer's CALLER (not a VolTargetSizer field --
   confirmed absent from the amended constructor in #1 above), and omitting it
   must raise, not silently default. The function is
   `ewma_volatility_ann(close, day_offsets, *, halflife, sessions_per_year=252)`
   in `nifty_quant.features.core`, beside `parkinson_volatility`/`rolling_std`
   -- NOT `causal_ewma_sigma_ann` in `backtest/portfolio.py`, this file's
   original guess before the function was named. `halflife` stays required with
   no default (amendment #3): it is a genuine research choice Phase E will
   measure, unlike `window` above, which was never a choice at all. test_11
   pins this via `inspect.signature` (asserting `halflife` is present with no
   default) AND a direct `pytest.raises(TypeError)` call omitting it.

   THIRD DEFECT FOUND AND FIXED (amendment #3, off the back of this file):
   amendment #2's fix for finding 8 gave `ewma_volatility_ann` no `window`
   parameter while stating it annualizes "internally via" `annualization_factor`,
   whose `window` argument had no default -- relocating the rule-8 violation
   from `halflife` to `window` rather than removing it. Amendment #3 deletes
   `window` entirely rather than defaulting it, per the reasoning in finding 7
   above.

9. [NOT CHANGED, confirmed correct by the coordinator] `min_names` stays out of
   scope -- it governs correlation measurement in the engine, tested there.

10. [NOT CHANGED, confirmed correct by the coordinator] the no-n×n-matrix
    constraint stays review-enforced, not black-box tested; test 3 already pins
    the arithmetic it's meant to protect.
=====================================================================================
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nifty_quant.backtest.portfolio import GrossNotionalSizer

try:
    from nifty_quant.backtest.portfolio import VolTargetSizer
except ImportError:  # expected during TDD red phase -- VolTargetSizer does not exist yet
    VolTargetSizer = None  # type: ignore[assignment,misc]

try:
    from nifty_quant.features.core import annualization_factor
except ImportError:  # expected during TDD red phase -- amendment #2 places this
    annualization_factor = None  # type: ignore[assignment,misc]

try:
    from nifty_quant.features.core import ewma_volatility_ann
except ImportError:  # expected during TDD red phase -- amendment #2 names this
    ewma_volatility_ann = None  # type: ignore[assignment,misc]


def _require_sut() -> None:
    if VolTargetSizer is None:
        pytest.fail(
            "VolTargetSizer does not exist yet in "
            "src/nifty_quant/backtest/portfolio.py -- expected TDD red."
        )


def _require_annualization_factor() -> None:
    if annualization_factor is None:
        pytest.fail(
            "nifty_quant.features.core.annualization_factor does not exist yet "
            "-- expected TDD red (amendment #5, module corrected by amendment #2 "
            "item 2: this lives with the estimator family in features/core.py, "
            "not backtest/portfolio.py)."
        )


def _require_ewma_sigma_fn() -> None:
    if ewma_volatility_ann is None:
        pytest.fail(
            "nifty_quant.features.core.ewma_volatility_ann does not exist yet "
            "-- expected TDD red (amendment #6, named and placed by amendment #2 "
            "item 2: it lives beside parkinson_volatility/rolling_std in "
            "features/core.py, not backtest/portfolio.py)."
        )


# --------------------------------------------------------------------------------
# Oracles independently written from the spec's own formulas (spec:42-49), used to
# check the SUT (or, for the pure-algebra defect-evidence tests, to check the spec
# itself). These are NOT copies of any implementation -- they exist so this file
# has ground truth to compare a future VolTargetSizer against.
# --------------------------------------------------------------------------------


def _materialized_variance(w: np.ndarray, sigma: np.ndarray, rho: float) -> float:
    """w' Sigma w with Sigma = D C D materialised as an n x n matrix."""
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = w.size
    corr = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    d = np.diag(sigma)
    sigma_matrix = d @ corr @ d
    return float(w @ sigma_matrix @ w)


def _closed_form_variance(w: np.ndarray, sigma: np.ndarray, rho: float) -> float:
    """spec:49 -- (1-rho) * sum_i (w_i sigma_i)^2 + rho * (sum_i w_i sigma_i)^2."""
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ws = w * sigma
    return (1.0 - rho) * float(np.sum(ws**2)) + rho * float(np.sum(ws)) ** 2


def _materialized_sigma_portfolio(w: np.ndarray, sigma: np.ndarray, rho: float) -> float:
    """sqrt(w' Sigma w). Only used with rho in [0, 1] where the variance is
    guaranteed non-negative; see test_3 for the wider-rho variance-only check."""
    return float(np.sqrt(_materialized_variance(w, sigma, rho)))


def _closed_form_sigma_portfolio(w: np.ndarray, sigma: np.ndarray, rho: float) -> float:
    """sqrt of the spec:49 closed form. Only used with rho in [0, 1]; see test_3."""
    return float(np.sqrt(_closed_form_variance(w, sigma, rho)))


def _expected_annualization_factor(
    day_offsets: np.ndarray, sessions_per_year: int = 252
) -> float:
    """Oracle for amendment #3's `annualization_factor`: median session length
    (in bars) over EVERY session present in `day_offsets`, not the mean. No
    trailing window -- amendment #3 establishes bars-per-session is a calendar
    property, not a price property, so the full panel is admissible with no
    lookahead and no window constant is needed."""
    offsets = np.asarray(day_offsets, dtype=np.int64)
    session_lengths = np.diff(offsets)
    bars_per_session = float(np.median(session_lengths))
    return float(np.sqrt(sessions_per_year * bars_per_session))


def _conservative_variance_endpoint_max(w: np.ndarray, sigma: np.ndarray) -> float:
    """Amendment #1's fix: var(rho) is linear in rho, so its max over rho in
    [0, 1] is at an endpoint -- max(sum a_i^2, (sum a_i)^2) with a_i = w_i sigma_i.
    Conservative for ANY sign structure, no case analysis."""
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    a = w * sigma
    s2 = float(np.sum(a**2))  # rho = 0 endpoint
    s1_sq = float(np.sum(a)) ** 2  # rho = 1 endpoint
    return max(s2, s1_sq)


# --------------------------------------------------------------------------------
# Item 1: independent names, equal sigma_i, rho = 0.
# --------------------------------------------------------------------------------


def test_1_independent_names_closed_form_matches_matrix_and_analytic() -> None:
    w = np.array([0.4, -0.25, 0.15, -0.5, 0.1])
    sigma = np.full(5, 0.22)
    rho = 0.0

    analytic = 0.22 * float(np.sqrt(np.sum(w**2)))
    matrix = _materialized_sigma_portfolio(w, sigma, rho)
    closed = _closed_form_sigma_portfolio(w, sigma, rho)

    assert closed == pytest.approx(matrix, rel=1e-10, abs=1e-12)
    assert closed == pytest.approx(analytic, rel=1e-10, abs=1e-12)


# --------------------------------------------------------------------------------
# Item 2: perfect correlation, same-signed book.
# --------------------------------------------------------------------------------


def test_2_perfect_correlation_matches_matrix_and_analytic() -> None:
    w = np.array([0.4, 0.25, 0.15, 0.5, 0.1])  # all same sign
    sigma = np.array([0.18, 0.22, 0.30, 0.15, 0.25])
    rho = 1.0

    analytic = float(np.sum(np.abs(w * sigma)))
    matrix = _materialized_sigma_portfolio(w, sigma, rho)
    closed = _closed_form_sigma_portfolio(w, sigma, rho)

    assert closed == pytest.approx(matrix, rel=1e-10, abs=1e-12)
    assert closed == pytest.approx(analytic, rel=1e-10, abs=1e-12)


# --------------------------------------------------------------------------------
# Item 3: load-bearing algebraic identity, hammered over many random draws.
# --------------------------------------------------------------------------------


def test_3_closed_form_equals_matrix_form_random_draws() -> None:
    """Compared in the pre-sqrt variance domain (spec:49 is the variance formula;
    spec:45 sqrt's it separately) so the identity is checked over the FULL rho
    sweep, including rho outside [0, 1] where C is not a valid (PSD) correlation
    matrix and w' Sigma w can go negative -- sqrt would turn a genuine algebraic
    match into a NaN-vs-NaN comparison failure there, which is a property of sqrt,
    not of the identity being tested.
    """
    rng = np.random.default_rng(20260819)
    n_draws = 3000
    max_abs_diff = 0.0
    for _ in range(n_draws):
        n = int(rng.integers(1, 9))
        w = rng.uniform(-2.0, 2.0, size=n)
        sigma = rng.uniform(1e-4, 1.0, size=n)
        # rho swept well outside [0, 1] too: the identity is a pure algebraic
        # rearrangement of D C D and must hold regardless of whether C is a
        # valid (PSD) correlation matrix for this draw.
        rho = float(rng.uniform(-3.0, 3.0))

        matrix_var = _materialized_variance(w, sigma, rho)
        closed_var = _closed_form_variance(w, sigma, rho)
        max_abs_diff = max(max_abs_diff, abs(matrix_var - closed_var))

        assert closed_var == pytest.approx(matrix_var, rel=1e-8, abs=1e-8), (n, w, sigma, rho)

    # Regression guard on the whole sweep, not just the last draw.
    assert max_abs_diff < 1e-7

    # Also cover the sqrt'd, spec:45 form directly, restricted to rho in [0, 1]
    # where it is always well-defined (no negative-variance edge case).
    for _ in range(500):
        n = int(rng.integers(1, 9))
        w = rng.uniform(-2.0, 2.0, size=n)
        sigma = rng.uniform(1e-4, 1.0, size=n)
        rho = float(rng.uniform(0.0, 1.0))
        assert _closed_form_sigma_portfolio(w, sigma, rho) == pytest.approx(
            _materialized_sigma_portfolio(w, sigma, rho), rel=1e-8, abs=1e-8
        )


# --------------------------------------------------------------------------------
# Items 4/5: the behavioural pair -- target hit when no clip binds, missed and
# flagged when a clip does. Both use bar_traded_value=None to stay off the
# capacity-water-filling ambiguity (defect 4 above).
# --------------------------------------------------------------------------------

_W_RAW = np.array([0.5, -0.3, 0.2, -0.4])
_SIGMA = np.array([0.20, 0.25, 0.18, 0.22])
_RHO = 0.3
_PRICES = np.array([100.0, 200.0, 150.0, 50.0])
_CAPITAL = 10_000_000.0
_TARGET_VOL_ANN = 0.15


def test_4_target_hit_when_no_clip_binds() -> None:
    _require_sut()
    sigma_portfolio_raw = _closed_form_sigma_portfolio(_W_RAW, _SIGMA, _RHO)
    scale = _TARGET_VOL_ANN / sigma_portfolio_raw
    w_scaled = _W_RAW * scale
    assert np.max(np.abs(w_scaled)) < 0.9  # sanity: clip is genuinely loose below

    sizer = VolTargetSizer(target_vol_ann=_TARGET_VOL_ANN, max_weight=0.99, whole_shares=False)
    result = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=_SIGMA, corr=_RHO)

    assert result.clip_binding is False
    assert result.sigma_portfolio_ann == pytest.approx(sigma_portfolio_raw, rel=1e-9)
    assert result.vol_scale_applied == pytest.approx(scale, rel=1e-9)
    assert result.vol_target_achieved == pytest.approx(_TARGET_VOL_ANN, rel=1e-6)

    expected_notional = w_scaled * _CAPITAL
    expected_shares = expected_notional / _PRICES
    assert result.shares == pytest.approx(expected_shares, rel=1e-6)


def test_5_target_missed_when_clip_binds() -> None:
    _require_sut()
    tight_max_weight = 0.05  # well below every unclipped scaled weight (>= 0.22)
    sigma_portfolio_raw = _closed_form_sigma_portfolio(_W_RAW, _SIGMA, _RHO)
    scale = _TARGET_VOL_ANN / sigma_portfolio_raw
    w_scaled = _W_RAW * scale
    assert np.all(np.abs(w_scaled) > tight_max_weight)  # sanity: clip WILL bind

    w_clipped = np.clip(w_scaled, -tight_max_weight, tight_max_weight)
    expected_achieved = _closed_form_sigma_portfolio(w_clipped, _SIGMA, _RHO)
    assert abs(expected_achieved - _TARGET_VOL_ANN) > 0.01  # sanity: materially off target

    sizer = VolTargetSizer(
        target_vol_ann=_TARGET_VOL_ANN, max_weight=tight_max_weight, whole_shares=False
    )
    result = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=_SIGMA, corr=_RHO)

    assert result.clip_binding is True
    # Diagnostic BEFORE clips is unaffected by the clip -- still the raw estimate.
    assert result.sigma_portfolio_ann == pytest.approx(sigma_portfolio_raw, rel=1e-9)
    assert result.vol_target_achieved == pytest.approx(expected_achieved, rel=1e-6)
    assert result.vol_target_achieved != pytest.approx(_TARGET_VOL_ANN, rel=1e-3)


# --------------------------------------------------------------------------------
# Item 6: scale invariance of the target.
# --------------------------------------------------------------------------------


def test_6_doubling_sigma_halves_weights() -> None:
    _require_sut()
    sizer = VolTargetSizer(target_vol_ann=_TARGET_VOL_ANN, max_weight=0.99, whole_shares=False)

    result_orig = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=_SIGMA, corr=_RHO)
    result_doubled = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=2.0 * _SIGMA, corr=_RHO)

    assert result_orig.clip_binding is False
    assert result_doubled.clip_binding is False
    assert result_doubled.shares == pytest.approx(0.5 * result_orig.shares, rel=1e-6)
    assert result_doubled.sigma_portfolio_ann == pytest.approx(
        2.0 * result_orig.sigma_portfolio_ann, rel=1e-9
    )


# --------------------------------------------------------------------------------
# Item 7: non-finite sigma_i zeroes that name without poisoning the estimate.
# --------------------------------------------------------------------------------


def test_7_non_finite_sigma_zeroes_name_without_poisoning_estimate() -> None:
    _require_sut()
    sigma_with_nan = _SIGMA.copy()
    sigma_with_nan[1] = np.nan  # symbol index 1 (BBB-equivalent) has no valid sigma

    sizer = VolTargetSizer(target_vol_ann=_TARGET_VOL_ANN, max_weight=0.99, whole_shares=False)
    result = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=sigma_with_nan, corr=_RHO)

    assert np.isfinite(result.shares).all()
    assert result.shares[1] == 0.0
    assert np.isfinite(result.sigma_portfolio_ann)
    assert np.isfinite(result.vol_scale_applied)
    assert np.isfinite(result.vol_target_achieved)

    # The estimate should match treating the poisoned name as absent (w_i = 0),
    # not a NaN- or zero-imputed sigma feeding the closed form.
    w_masked = _W_RAW.copy()
    w_masked[1] = 0.0
    expected_sigma_portfolio = _closed_form_sigma_portfolio(w_masked, _SIGMA, _RHO)
    assert result.sigma_portfolio_ann == pytest.approx(expected_sigma_portfolio, rel=1e-6)


# --------------------------------------------------------------------------------
# Item 8 (amendment #1): rho fallback is now the endpoint-max, not rho = 0.
# Required test per the amendment: "the fallback never produces a LARGER book than
# the true-rho sizing would, for both a same-signed and a mixed-sign book. Assert
# the direction, not the mechanism." test_8a/test_8b do exactly that, via the SUT.
# test_8c_regression_guard keeps the old falsifying algebra (amendment's own
# instruction) re-pointed at proving the NEW fallback really is conservative.
# --------------------------------------------------------------------------------

_TRUE_RHO_FOR_FALLBACK_TESTS = 0.6  # within the amendment's measured range [0.05, 0.9]


def test_8a_fallback_never_larger_book_same_signed() -> None:
    _require_sut()
    w_same_signed = np.array([0.5, 0.3, 0.2, 0.4])  # all positive, no hedging
    sigma = np.array([0.20, 0.25, 0.18, 0.22])

    sizer = VolTargetSizer(target_vol_ann=_TARGET_VOL_ANN, max_weight=0.99, gross=10.0)
    result_true = sizer.to_shares(
        w_same_signed, _PRICES, _CAPITAL, sigma=sigma, corr=_TRUE_RHO_FOR_FALLBACK_TESTS
    )
    result_fallback = sizer.to_shares(w_same_signed, _PRICES, _CAPITAL, sigma=sigma, corr=None)

    assert result_true.clip_binding is False
    assert result_fallback.clip_binding is False
    # DIRECTION only, per the amendment: fallback book must never be bigger.
    assert result_fallback.vol_scale_applied <= result_true.vol_scale_applied + 1e-12
    assert result_fallback.sigma_portfolio_ann >= result_true.sigma_portfolio_ann - 1e-12


def test_8b_fallback_never_larger_book_mixed_sign() -> None:
    """Uses the spec's own worked example (amendment #1): w=[0.2,0.3,-0.4,0.1],
    sigma=[0.2,0.5,0.3,0.9], where the conservative endpoint is rho=0 (var=0.0466)
    rather than rho=1 (var=0.0256) -- the opposite endpoint from the same-signed
    case in test_8a, which is exactly why the fix needs no sign-structure case
    analysis: max() picks whichever endpoint is larger either way.
    """
    _require_sut()
    w_mixed = np.array([0.2, 0.3, -0.4, 0.1])
    sigma = np.array([0.2, 0.5, 0.3, 0.9])
    prices = np.array([100.0, 200.0, 150.0, 50.0])

    sizer = VolTargetSizer(target_vol_ann=_TARGET_VOL_ANN, max_weight=0.99, gross=10.0)
    result_true = sizer.to_shares(
        w_mixed, prices, _CAPITAL, sigma=sigma, corr=_TRUE_RHO_FOR_FALLBACK_TESTS
    )
    result_fallback = sizer.to_shares(w_mixed, prices, _CAPITAL, sigma=sigma, corr=float("nan"))

    assert result_true.clip_binding is False
    assert result_fallback.clip_binding is False
    assert result_fallback.vol_scale_applied <= result_true.vol_scale_applied + 1e-12
    assert result_fallback.sigma_portfolio_ann >= result_true.sigma_portfolio_ann - 1e-12


def test_8c_regression_guard_endpoint_max_conservative_old_rho0_was_not() -> None:
    """Pure algebra, no SUT -- true today regardless of implementation status.

    Two claims, both from the amendment:
      (a) the OLD rho=0 fallback was NOT conservative for a same-signed book
          (this is the falsification that triggered the amendment -- kept as a
          regression guard so the spec cannot silently drift back to rho=0);
      (b) the NEW endpoint-max fallback IS conservative (>= true variance) for
          ANY rho in [0, 1], swept over many draws, for both a same-signed and a
          mixed-sign book -- this is the actual invariant the fix relies on.
    """
    rng = np.random.default_rng(20260819)

    # (a) regression guard: the old claim was false for a same-signed book.
    w_same_signed = np.array([0.5, 0.3, 0.2, 0.4])
    sigma_fixed = np.array([0.20, 0.25, 0.18, 0.22])
    rho_true = 0.3
    var_rho0 = _closed_form_variance(w_same_signed, sigma_fixed, 0.0)
    var_true = _closed_form_variance(w_same_signed, sigma_fixed, rho_true)
    assert var_rho0 < var_true  # rho=0 under-estimated -- the old bug, still true

    # (b) the new fix: max(S1, S2) dominates var(rho) for every rho in [0, 1],
    # for random same-signed AND mixed-sign books.
    for _ in range(1000):
        n = int(rng.integers(2, 8))
        sigma = rng.uniform(1e-3, 1.0, size=n)
        rho = float(rng.uniform(0.0, 1.0))

        w_same = rng.uniform(0.05, 2.0, size=n)  # all positive: same-signed
        assert _conservative_variance_endpoint_max(w_same, sigma) >= _closed_form_variance(
            w_same, sigma, rho
        ) - 1e-12

        w_mixed = rng.uniform(-2.0, 2.0, size=n)  # free sign: mixed-sign
        assert _conservative_variance_endpoint_max(w_mixed, sigma) >= _closed_form_variance(
            w_mixed, sigma, rho
        ) - 1e-12

    # Cross-check against the amendment's own worked example verbatim.
    w_worked = np.array([0.2, 0.3, -0.4, 0.1])
    sigma_worked = np.array([0.2, 0.5, 0.3, 0.9])
    assert _closed_form_variance(w_worked, sigma_worked, 0.0) == pytest.approx(0.0466, abs=1e-6)
    assert _closed_form_variance(w_worked, sigma_worked, 1.0) == pytest.approx(0.0256, abs=1e-6)
    assert _closed_form_variance(w_worked, sigma_worked, 0.5) == pytest.approx(0.0361, abs=1e-6)
    assert _conservative_variance_endpoint_max(w_worked, sigma_worked) == pytest.approx(
        0.0466, abs=1e-6
    )


# --------------------------------------------------------------------------------
# Item 9 (amendment #5, signature corrected by amendment #3): annualization is
# not the sizer's job -- it moves to a named, separately testable helper,
# `annualization_factor(day_offsets, *, sessions_per_year=252)`, in
# `nifty_quant.features.core`. This test targets that helper DIRECTLY (it is no
# longer a proxy through the sizer, which never sees a Panel or day_offsets).
#
# No `window` parameter (amendment #3): bars-per-session is a CALENDAR property,
# not a price property, so estimating it over the full panel introduces no
# lookahead -- knowing a future session was 60-bar Muhurat tells you nothing
# about any return. The estimator is the median of np.diff(day_offsets) over
# every session present.
# --------------------------------------------------------------------------------


def test_9_annualization_uses_derived_bars_per_session_not_375() -> None:
    _require_annualization_factor()

    # One 60-bar session (Muhurat-like, rule 5) and one 105-bar session
    # (shortened/disaster-recovery-like, rule 5). Median of [60, 105] = 82.5.
    day_offsets = np.array([0, 60, 165], dtype=np.int64)

    derived = annualization_factor(day_offsets, sessions_per_year=252)
    expected = _expected_annualization_factor(day_offsets)

    assert derived == pytest.approx(expected, rel=1e-12)
    assert derived == pytest.approx(float(np.sqrt(252 * 82.5)), rel=1e-9)
    assert derived != pytest.approx(float(np.sqrt(252 * 375)), rel=1e-6)


# --------------------------------------------------------------------------------
# Item 10: GrossNotionalSizer is untouched. Runs today -- no SUT dependency.
# --------------------------------------------------------------------------------


def test_10_gross_notional_sizer_bit_identical_snapshot() -> None:
    weights = np.array([0.5, -1.5, 0.05])  # middle name intentionally over max_weight
    prices = np.array([100.0, 50.0, 20.0])
    capital = 1_000_000.0

    sizer = GrossNotionalSizer(gross=2.0, max_weight=0.10, min_trade_notional=25_000.0)
    shares = sizer.to_shares(weights, prices, capital)

    # Hand-derived golden values, independent of the sizer's own code:
    #   clipped weights: [0.10, -0.10, 0.05]
    #   target notional: clipped * gross(2.0) * capital(1e6) = [200000, -200000, 100000]
    #   shares = notional / price, all whole already, all >= min_trade_notional
    expected = np.array([2000.0, -4000.0, 5000.0])
    assert np.array_equal(shares, expected)


# --------------------------------------------------------------------------------
# Item 11 (amendment #6): EWMA halflife is required and cannot be omitted.
# --------------------------------------------------------------------------------


def test_11_ewma_halflife_required_raises_if_omitted() -> None:
    _require_ewma_sigma_fn()

    signature = inspect.signature(ewma_volatility_ann)
    assert "halflife" in signature.parameters
    assert signature.parameters["halflife"].default is inspect.Parameter.empty

    close = np.array([100.0, 100.5, 99.8, 100.2])
    day_offsets = np.array([0, 4], dtype=np.int64)

    with pytest.raises(TypeError):
        ewma_volatility_ann(close, day_offsets)


# --------------------------------------------------------------------------------
# Item 12 (amendment #4 item 1 + amendment #5): `max_gross` binds via a
# PROPORTIONAL rescale of the whole book, never a per-name clip -- "a per-name
# clip would distort the cross-section; a proportional rescale is the only
# operation that caps exposure without changing what the strategy said about
# relative attractiveness." The discriminating signature of a proportional
# rescale is that the ratio between any two names' final notional equals the
# ratio of their RAW input weights; a per-name clip would not preserve that
# ratio (whichever names individually exceeded some bound would compress
# relative to the others).
# --------------------------------------------------------------------------------


def test_12_max_gross_proportional_rescale_preserves_weight_ratios() -> None:
    _require_sut()
    sigma_portfolio_raw = _closed_form_sigma_portfolio(_W_RAW, _SIGMA, _RHO)
    scale = _TARGET_VOL_ANN / sigma_portfolio_raw
    w_vol = _W_RAW * scale
    assert np.max(np.abs(w_vol)) < 0.99  # sanity: max_weight clip (step 3) does NOT bind
    max_gross = 0.8
    assert np.sum(np.abs(w_vol)) > max_gross  # sanity: max_gross WILL bind (step 5)

    sizer = VolTargetSizer(
        target_vol_ann=_TARGET_VOL_ANN,
        max_weight=0.99,
        gross=1.0,
        max_gross=max_gross,
        whole_shares=False,
    )
    result = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=_SIGMA, corr=_RHO)

    assert result.clip_binding is True
    # The target is missed -- max_gross fought the vol target, same as any other
    # structural binder (amendment 5: clip_binding True iff step 3, 5 or 6 moved
    # the book).
    assert result.vol_target_achieved != pytest.approx(_TARGET_VOL_ANN, rel=1e-3)

    notional = result.shares * _PRICES
    assert np.sum(np.abs(notional)) / _CAPITAL == pytest.approx(max_gross, rel=1e-6)

    # Discriminating check: final notional ratios equal RAW weight ratios. This
    # holds ONLY under a uniform proportional rescale; it fails under a per-name
    # clip.
    for i in range(1, len(_W_RAW)):
        assert notional[i] / notional[0] == pytest.approx(_W_RAW[i] / _W_RAW[0], rel=1e-6)


# --------------------------------------------------------------------------------
# Item 13 (amendment #5): the normative pipeline order. Step 5 (max_gross
# proportional rescale) comes AFTER step 4 (the `gross` multiply), so
# `max_gross` bounds the FINAL exposure sum(|notional|)/capital for ANY value of
# `gross`. An implementation that rescaled to the ceiling BEFORE the `gross`
# multiply would let the final book exceed `max_gross` by a factor of `gross` --
# exactly the defect amendment 5 exists to rule out. With gross=2.0 and
# max_gross=0.5, the wrong (pre-gross-multiply) order produces a realised
# exposure of max_gross * gross == 1.0, not 0.5 -- that is the concrete
# fingerprint of the ordering bug, not a hypothetical.
# --------------------------------------------------------------------------------


def test_13_max_gross_bounds_final_exposure_after_gross_multiply() -> None:
    _require_sut()
    gross = 2.0
    max_gross = 0.5

    sigma_portfolio_raw = _closed_form_sigma_portfolio(_W_RAW, _SIGMA, _RHO)
    scale = _TARGET_VOL_ANN / sigma_portfolio_raw
    w_vol = _W_RAW * scale
    assert np.max(np.abs(w_vol)) < 0.99  # sanity: max_weight clip (step 3) does NOT bind
    # sanity: max_gross WILL bind after the gross multiply (step 4 then step 5)
    assert np.sum(np.abs(w_vol)) * gross > max_gross

    sizer = VolTargetSizer(
        target_vol_ann=_TARGET_VOL_ANN,
        max_weight=0.99,
        gross=gross,
        max_gross=max_gross,
        whole_shares=False,
    )
    result = sizer.to_shares(_W_RAW, _PRICES, _CAPITAL, sigma=_SIGMA, corr=_RHO)

    assert result.clip_binding is True
    notional = result.shares * _PRICES
    realised = np.sum(np.abs(notional)) / _CAPITAL

    assert realised == pytest.approx(max_gross, rel=1e-6)
    # The ordering bug's exact fingerprint: rescaling BEFORE the gross multiply
    # would yield max_gross * gross == 1.0 instead.
    assert realised != pytest.approx(1.0, rel=1e-3)
