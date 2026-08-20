# SPEC DEFECTS / AMBIGUITIES NOTICED:
#
# NOTE (re-point pass after amendments 2, 3, 4): this file was written from the spec body plus
# AMENDMENT 1 only. Amendments 2 and 3 fixed the VolTargetSizer constructor (whole_shares /
# min_trade_notional=25_000.0 replace the lot_size field I had copied from amendment 1's own
# wrong draft) and renamed/relocated/reshaped the vol-path helper (ewma_volatility_ann in
# nifty_quant.features.core, takes `close` prices, no `window` param on either helper).
# Constructor calls and imports below have been updated to match; defects 16-19 (found by me,
# about gaps amendment 1 left open) are resolved by amendment 4 and marked as such rather than
# left to teach a future reader the wrong contract. Defect 20 is new, found during this
# re-point pass. Defects 1-15 that remain genuinely open are otherwise unchanged.
#
# 1. `VolTargetSizer.to_shares` signature says `sigma=None` must raise ValueError, but the
#    dataclass field list does not include `sigma` as a field -- it is only a parameter to
#    `to_shares`. This is fine, but the spec text is ambiguous about whether `sigma` is a
#    constructor arg or a method arg. The API decision block shows it as a method arg, so we
#    follow that.
#
# 2. The spec says `corr` is "a single scalar rho" but also says "or `None`/`nan` if
#    unavailable". It does not specify what to do if `corr` is a finite scalar outside [0,1]
#    (e.g. -0.5 or 1.5). The closed form uses `rho = corr` directly, but the fallback path
#    uses `max(sum a_i^2, (sum a_i)^2)` which is only valid for rho in [0,1]. The spec says
#    "clipped/used as given -- do not second-guess a caller-supplied rho", so we assume the
#    caller is responsible for passing a valid rho in [0,1]. Tests only use rho in [0,1].
#
# 3. The spec says `sigma_portfolio_ann` is computed from "RAW (unscaled, unclipped) input
#    weights" but step 1 says to zero out weights where `sigma` is non-finite. So "raw" means
#    "after sigma-zeroing but before vol scaling and clipping". This is consistent but the
#    wording could be clearer.
#
# 4. The spec says `vol_target_achieved` is recomputed from "FINAL weights
#    (`shares*prices/capital`)". But if `whole_shares` rounding or `min_trade_notional` zeroing
#    changes the shares, the final weights may not exactly equal the clipped/water-filled
#    weights. The spec does not say whether `vol_target_achieved` should use the exact
#    post-rounding weights or the pre-rounding weights. We assume post-rounding (as written),
#    which means `clip_binding` could be True due to rounding even if no clip bound. Tests
#    avoid this by using `whole_shares=False` and `min_trade_notional=0.0` where exactness
#    matters. STILL OPEN after amendments 2-4.
#
# 5. RESOLVED by amendment 4 item 2: `clip_binding` covers structural binders only
#    (max_weight, max_gross, capacity) and explicitly NOT `min_trade_notional`/`whole_shares`
#    quantisation, confirming the assumption this suite already made. Tests still avoid the
#    quantisation paths (`whole_shares=False`, `min_trade_notional=0.0`) where `clip_binding`
#    is asserted, since amendment 4 also says `vol_target_achieved` (not `clip_binding`) is
#    the field that stays sensitive to rounding, and this suite never needed to exercise that.
#
# 6. RESOLVED by amendment 4 item 1: `gross` is NOT a cap at all -- confirmed a bare
#    multiplier, semantics identical to `GrossNotionalSizer` (`target_notional = clipped_weight
#    * self.gross * capital`, no rescale-if-exceeded step). The actual ceiling is the new
#    `max_gross: float | None = None`, which rescales the whole book proportionally so that
#    `sum(|w|) == max_gross` when it binds. This suite's original `gross=100.0`-everywhere
#    workaround is kept for tests that must not have `gross` interact at all (it stays an
#    inert multiplier at that value); the new `max_gross`-binding case is exercised directly
#    below in `test_clip_binding_when_max_gross_binds_and_rescales_proportionally`, using
#    `gross=1.0` (identity) so defect 20 (below) cannot contaminate it.
#
# 7. The spec says capacity water-filling "redistribute shortfall iteratively into names with
#    spare room". It does not specify the exact iteration order or convergence criterion.
#    Since it says "reuse the pattern" from `GrossNotionalSizer`, we assume the implementation
#    will match that exactly. Tests only check that capacity binding sets `clip_binding=True`
#    and `vol_target_achieved` is not close to target, not the exact water-filled weights.
#
# 8. PARTIALLY RESOLVED by amendment 3: `window` is deleted entirely, so the
#    window-out-of-range question this defect raised no longer applies. Still open: the spec
#    does not say what `annualization_factor` should do if `day_offsets` has fewer than 2
#    elements (median of an empty diff array). Tests only use valid inputs.
#
# 9. RESOLVED (renamed) by amendments 2-4: the function is `ewma_volatility_ann` in
#    `nifty_quant.features.core`, takes `close` prices (not returns), and has no `window`.
#    The original ambiguity about whether `halflife=None` raises `TypeError` or `ValueError`
#    is STILL OPEN -- amendment 4 does not say, so the test still accepts either.
#
# 10. The spec says `SizingResult` gains four new fields, but does not specify their order in
#     the dataclass or whether they have defaults. Since `GrossNotionalSizer` still returns a
#     bare ndarray when `bar_traded_value is None`, the new fields must be optional or have
#     defaults for `GrossNotionalSizer` to construct `SizingResult` without them. We assume
#     they have defaults (e.g. `sigma_portfolio_ann: float = 0.0`, etc.) or are only set by
#     `VolTargetSizer`. Tests only access them on `VolTargetSizer` results.
#
# 11. The spec says `corr` fallback uses `var_conservative = max(sum_i a_i**2, (sum_i a_i)**2)`.
#     This is only conservative for rho in [0,1]. For rho < 0, the true variance could be
#     smaller than `sum a_i^2`, so the fallback would be larger (more conservative). For
#     rho > 1, the true variance could be larger than `(sum a_i)^2`, so the fallback would be
#     smaller (less conservative). The spec says "conservative fallback" but only for rho in
#     [0,1]. Tests only use rho in [0,1].
#
# 12. The spec says `sigma=None` raises `ValueError("sigma is required")`. But it does not
#     specify what to do if `sigma` is provided but has wrong length. We assume the
#     implementation will raise a `ValueError` or let numpy broadcasting raise. Tests do not
#     cover this.
#
# 13. The spec says `corr` is "optional and defaults to the fallback path when omitted". But
#     the method signature shows `corr=None` as a keyword-only arg. This is consistent: if
#     `corr` is not passed, it defaults to `None`, which triggers the fallback. Tests pass
#     `corr=None` explicitly to test the fallback.
#
# 14. The spec says `vol_scale_applied` is "computed BEFORE any clip". But it does not specify
#     whether it should be stored even if `sigma_portfolio_ann` is 0 (which would cause
#     division by zero). Tests avoid this by using non-zero sigma and weights.
#
# 15. The spec says `clip_binding` uses `np.isclose(vol_target_achieved, target_vol_ann,
#     rtol=1e-6, atol=1e-9)`. This is very tight. Tests use `whole_shares=False` and
#     `min_trade_notional=0.0` to avoid rounding noise, but floating-point error in the
#     water-filling or matrix multiplication could still cause false positives. We assume the
#     implementation is careful enough to hit this tolerance when nothing binds. STILL OPEN.
#
# 16. RESOLVED by amendment 4 item 4: `annualization_factor` and `ewma_volatility_ann` both
#     live in `nifty_quant.features.core`, not `nifty_quant.backtest.portfolio` as I guessed.
#     My guess was flagged as a guess and was wrong; imports below now match amendment 4.
#
# 17. RESOLVED by amendments 2-4: the function is `ewma_volatility_ann`
#     (`nifty_quant.features.core`), takes `close` prices rather than pre-computed returns,
#     and has no `window` parameter (deleted by amendment 3, not defaulted). My guessed name
#     (`causal_ewma_vol_ann`), module, and signature were all wrong; test 11 is updated below.
#
# 18. RESOLVED by amendment 4 item 3: the four new `SizingResult` fields carry defaults
#     (`sigma_portfolio_ann`, `vol_scale_applied`, `vol_target_achieved` all `float("nan")`,
#     `clip_binding` `False`). NaN means "not produced by a vol-targeted sizer", not zero.
#     `GrossNotionalSizer`'s construction site is confirmed untouched. This suite still never
#     constructs `SizingResult` directly and never reads the new fields off a
#     `GrossNotionalSizer` result, which remains the correct thing to do regardless.
#
# 19. RESOLVED by amendment 4 item 1 -- this was my sharpest finding, confirmed correct against
#     source (`backtest/portfolio.py:148`, `target_notional = clipped_weight * self.gross *
#     capital`, no rescale-if-exceeded step anywhere in `GrossNotionalSizer.to_shares`).
#     Amendment 1's "gross remains a cap" language invented behaviour the sibling class does
#     not have. Fixed by splitting the idea in two: `gross` stays an inert multiplier
#     (identical semantics to `GrossNotionalSizer`), and a NEW `max_gross: float | None = None`
#     is the actual ceiling, enforced by a proportional whole-book rescale (not a per-name
#     clip) when `sum(|w|)` exceeds it after vol scaling. Now tested directly in
#     `test_clip_binding_when_max_gross_binds_and_rescales_proportionally`.
#
# 20. RESOLVED by AMENDMENT 5, which was the correct and larger fix: rather than just answering
#     "before or after `gross`", amendment 5 writes the entire seven-step pipeline in order for
#     the first time across five documents, closing off this whole CLASS of ordering question
#     rather than only this instance of it. The load-bearing fact: `max_gross` bounds
#     `sum(|notional|) / capital` AFTER the `gross` multiply (step 5 after step 4), which is why
#     it bounds what it claims to bound for any `gross` value. `clip_binding` is set by steps 3,
#     5, 6 only -- never 4 (the inert multiply) or 7 (rounding/dust). The numeric case this
#     enables is now tested directly in
#     `test_max_gross_ceiling_applies_after_gross_multiplier_not_before`: with `gross=2.0` and
#     `max_gross=0.5`, an implementation that rescaled to the ceiling BEFORE the `gross` multiply
#     would land the book at `2.0 * 0.5 = 1.0`, silently doubling past the ceiling -- exactly
#     the failure mode this defect predicted. My original `max_gross` test (`gross=1.0`) could
#     not have caught this by construction; it stays as-is per the lead's instruction and this
#     new test is additional, not a replacement.

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.backtest.portfolio import GrossNotionalSizer, VolTargetSizer
from nifty_quant.features.core import annualization_factor, ewma_volatility_ann


def _materialised_sigma_portfolio(
    weights: np.ndarray, sigma: np.ndarray, rho: float
) -> float:
    """Compute sqrt(w' Sigma w) using a materialised covariance matrix."""
    n = len(weights)
    sigma_vec = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    corr_mat = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    Sigma = np.outer(sigma_vec, sigma_vec) * corr_mat
    return float(np.sqrt(w @ Sigma @ w))


def test_sigma_portfolio_matches_materialised_matrix_when_rho_is_zero() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, -0.03, 0.02, 0.04, -0.01])
    sigma = np.full(5, 0.3)
    prices = np.full(5, 100.0)
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.0)
    expected = 0.3 * np.sqrt(np.sum(weights**2))
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected, rtol=1e-12)
    expected_mat = _materialised_sigma_portfolio(weights, sigma, 0.0)
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected_mat, rtol=1e-12)


def test_sigma_portfolio_matches_materialised_matrix_when_rho_is_one() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, 0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.full(4, 100.0)
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=1.0)
    expected = np.sum(np.abs(weights * sigma))
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected, rtol=1e-12)
    expected_mat = _materialised_sigma_portfolio(weights, sigma, 1.0)
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected_mat, rtol=1e-12)


def test_sigma_portfolio_matches_materialised_matrix_on_random_draws() -> None:
    rng = np.random.default_rng(0)
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    for i in range(100):
        n = int(rng.integers(3, 8))
        weights = rng.uniform(-1.0, 1.0, size=n)
        sigma = rng.uniform(0.05, 1.0, size=n)
        rho = float(rng.uniform(0.0, 1.0))
        prices = np.full(n, 100.0)
        capital = 1_000_000.0
        result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=rho)
        expected = _materialised_sigma_portfolio(weights, sigma, rho)
        np.testing.assert_allclose(
            result.sigma_portfolio_ann,
            expected,
            rtol=1e-10,
            err_msg=f"draw {i}",
        )


def test_vol_target_achieved_when_nothing_binds() -> None:
    # AMENDMENT 6 (lead-adjudicated contract correction, recorded here as such):
    # `gross=100.0` is deliberately non-identity in this fixture to PROVE
    # `vol_target_achieved` tracks the REAL, levered book rather than a
    # gross-normalised fiction. `gross` genuinely multiplies notional at step 4,
    # so a book run at gross=100 really does carry ~100x the risk of the same
    # book at gross=1 -- reporting `vol_target_achieved == target_vol_ann` here
    # (the previous version of this assertion) would read "0.15, on target" for a
    # book actually running at 15.0, which is the original target_vol_ann-in-a-
    # meta-dict defect wearing a better name, this time inside the very
    # diagnostic built to catch it. `vol_scale_applied` is the separate,
    # gross-free field that answers "did the steps 1-3 targeting arithmetic do
    # its job"; this field answers "what risk does the book actually carry".
    sizer = VolTargetSizer(
        target_vol_ann=0.15, max_weight=1.0, gross=100.0, min_trade_notional=0.0, whole_shares=False
    )
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)

    np.testing.assert_allclose(result.vol_target_achieved, 0.15 * 100.0, rtol=1e-6)
    # gross never sets clip_binding: it is a declared leverage dial, not a
    # constraint that fought the target (amendment 5), even when it produces a
    # book running at many multiples of the nominal target_vol_ann.
    assert result.clip_binding is False

    # Leverage-visibility check: the field above is only meaningful if the book
    # itself really is running at ~gross x capital here. Nothing binds (max_weight
    # and gross are both loose, no capacity, no rounding), so the realised
    # notional gross is exactly `gross * sum(|w_vol|)` where w_vol is the
    # vol-scaled (pre-clip) weight -- computed independently via the materialised
    # matrix, not the SUT's own formula, so this is a real check on the actual
    # output, not a restatement of the implementation.
    sigma_portfolio_raw = _materialised_sigma_portfolio(weights, sigma, 0.3)
    w_vol = weights * (0.15 / sigma_portfolio_raw)
    expected_book_leverage = 100.0 * float(np.sum(np.abs(w_vol)))
    assert expected_book_leverage > 50.0  # sanity: this fixture really is heavily levered

    final_notional = result.shares * prices
    book_leverage = float(np.sum(np.abs(final_notional))) / capital
    assert book_leverage == pytest.approx(expected_book_leverage, rel=1e-6)


def test_clip_binding_when_max_weight_binds() -> None:
    sizer = VolTargetSizer(
        target_vol_ann=0.15,
        max_weight=0.001,
        gross=100.0,
        min_trade_notional=0.0,
        whole_shares=False,
    )
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)
    assert result.clip_binding is True
    assert not np.isclose(result.vol_target_achieved, 0.15, rtol=1e-6, atol=1e-9)


def test_clip_binding_when_capacity_binds() -> None:
    sizer = VolTargetSizer(
        target_vol_ann=0.15, max_weight=1.0, gross=100.0, min_trade_notional=0.0, whole_shares=False
    )
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    bar_traded_value = np.array([1_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0])
    result = sizer.to_shares(
        weights,
        prices,
        capital,
        sigma=sigma,
        corr=0.3,
        bar_traded_value=bar_traded_value,
        max_participation=0.02,
    )
    assert result.clip_binding is True
    assert not np.isclose(result.vol_target_achieved, 0.15, rtol=1e-6, atol=1e-9)


def test_clip_binding_when_max_gross_binds_and_rescales_proportionally() -> None:
    # AMENDMENT 4 item 1: `max_gross` (NOT `gross`, which is a bare multiplier identical to
    # GrossNotionalSizer's) is the actual exposure ceiling, and binding it must rescale the
    # WHOLE book proportionally rather than clip individual names. `gross=1.0` (identity) is
    # used deliberately: the spec never states whether `max_gross`'s sum(|w|) check happens
    # before or after `gross`'s multiply (see defect 20 below), so this test sidesteps that
    # gap entirely rather than assert into it.
    sizer = VolTargetSizer(
        target_vol_ann=0.15,
        gross=1.0,
        max_weight=1.0,
        min_trade_notional=0.0,
        whole_shares=False,
        max_gross=0.5,
    )
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)

    assert result.clip_binding is True
    assert not np.isclose(result.vol_target_achieved, 0.15, rtol=1e-6, atol=1e-9)

    # A proportional rescale is a single uniform scalar applied to every name, so it must
    # preserve the ratio between any two names' weights all the way back to the RAW input
    # weights (vol-scaling in step 3 is itself a uniform scalar too, and nothing else binds
    # here: max_weight=1.0 is loose and there is no bar_traded_value). A per-name clip would
    # NOT preserve this ratio, so this assertion is the discriminating check between the two
    # mechanisms amendment 4 explicitly rejects the per-name-clip alternative for.
    final_notional = result.shares * prices
    ratio_actual = final_notional[0] / final_notional[2]
    ratio_expected = weights[0] / weights[2]
    np.testing.assert_allclose(ratio_actual, ratio_expected, rtol=1e-6)

    # And the achieved gross must land exactly on the ceiling, not merely below it.
    gross_achieved = float(np.sum(np.abs(final_notional)) / capital)
    np.testing.assert_allclose(gross_achieved, 0.5, rtol=1e-6)


def test_max_gross_ceiling_applies_after_gross_multiplier_not_before() -> None:
    # AMENDMENT 5, the fix for defect 20: the seven-step pipeline puts the max_gross
    # proportional rescale (step 5) AFTER the gross multiplier (step 4), so max_gross bounds
    # sum(|notional|) / capital for ANY gross value. This is the numeric case that would have
    # caught the wrong order: with gross=2.0 and max_gross=0.5, an implementation that rescaled
    # to the ceiling BEFORE multiplying by gross would land the achieved gross at
    # 2.0 * 0.5 == 1.0, silently doubling past the ceiling. Correct order lands at exactly 0.5.
    sizer = VolTargetSizer(
        target_vol_ann=0.15,
        gross=2.0,
        max_weight=1.0,
        min_trade_notional=0.0,
        whole_shares=False,
        max_gross=0.5,
    )
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)

    final_notional = result.shares * prices
    achieved_gross = float(np.sum(np.abs(final_notional)) / capital)
    np.testing.assert_allclose(achieved_gross, 0.5, rtol=1e-6)
    assert not np.isclose(achieved_gross, 1.0, rtol=1e-6)
    assert result.clip_binding is True


def test_doubling_sigma_halves_vol_scale_applied() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result1 = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)
    result2 = sizer.to_shares(weights, prices, capital, sigma=2.0 * sigma, corr=0.3)
    assert result2.vol_scale_applied == pytest.approx(0.5 * result1.vol_scale_applied, rel=1e-9)
    assert result2.sigma_portfolio_ann == pytest.approx(2.0 * result1.sigma_portfolio_ann, rel=1e-9)


def test_nonfinite_sigma_inf_zeroes_weight_and_does_not_poison_estimate() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, np.inf, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)
    assert result.shares[1] == 0.0
    weights_zeroed = weights.copy()
    weights_zeroed[1] = 0.0
    sigma_finite = np.array([0.2, 0.0, 0.1, 0.3])
    expected = _materialised_sigma_portfolio(weights_zeroed, sigma_finite, 0.3)
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected, rtol=1e-10)
    assert np.isfinite(result.sigma_portfolio_ann)


def test_nonfinite_sigma_nan_zeroes_weight_and_does_not_poison_estimate() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, -0.03, 0.02, 0.04])
    sigma = np.array([0.2, np.nan, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    result = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=0.3)
    assert result.shares[1] == 0.0
    weights_zeroed = weights.copy()
    weights_zeroed[1] = 0.0
    sigma_finite = np.array([0.2, 0.0, 0.1, 0.3])
    expected = _materialised_sigma_portfolio(weights_zeroed, sigma_finite, 0.3)
    np.testing.assert_allclose(result.sigma_portfolio_ann, expected, rtol=1e-10)
    assert np.isfinite(result.sigma_portfolio_ann)


def test_fallback_vol_scale_is_never_larger_than_true_rho_same_sign() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, 0.03, 0.02, 0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    rng = np.random.default_rng(42)
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0] + list(rng.uniform(0.0, 1.0, size=20))
    for rho in rhos:
        result_true = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=rho)
        result_fallback = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=None)
        assert result_fallback.vol_scale_applied <= result_true.vol_scale_applied + 1e-9


def test_fallback_vol_scale_is_never_larger_than_true_rho_mixed_sign() -> None:
    sizer = VolTargetSizer(target_vol_ann=0.10, max_weight=1.0, gross=100.0)
    weights = np.array([0.05, -0.03, 0.02, -0.04])
    sigma = np.array([0.2, 0.4, 0.1, 0.3])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    capital = 1_000_000.0
    rng = np.random.default_rng(43)
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0] + list(rng.uniform(0.0, 1.0, size=20))
    for rho in rhos:
        result_true = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=rho)
        result_fallback = sizer.to_shares(weights, prices, capital, sigma=sigma, corr=None)
        assert result_fallback.vol_scale_applied <= result_true.vol_scale_applied + 1e-9


def test_annualization_factor_uses_median_bars_per_session() -> None:
    # AMENDMENT 3 deletes `window`: the factor is derived from the median session length
    # over the WHOLE panel, not a trailing slice. A fixture where most sessions are 375
    # bars (as my original draft used) would make the whole-panel median equal 375 again
    # and silently defeat this assertion -- this fixture is the spec's own worked example
    # (60-bar session, then 105-bar session) precisely to avoid that trap.
    day_offsets = np.array([0, 60, 165])
    result = annualization_factor(day_offsets)
    expected = np.sqrt(252 * 82.5)
    np.testing.assert_allclose(result, expected, rtol=1e-9)
    assert not np.isclose(result, np.sqrt(252 * 375))


def test_gross_notional_sizer_plain_path_is_bit_identical_to_golden_values() -> None:
    sizer = GrossNotionalSizer(
        gross=1.0,
        max_weight=0.10,
        min_trade_notional=25_000.0,
        whole_shares=True,
    )
    weights = np.array([0.05, -0.08, 0.15, np.nan, 0.03])
    prices = np.array([100.0, 250.0, 50.0, 300.0, np.nan])
    capital = 1_000_000.0
    out = sizer.to_shares(weights, prices, capital)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([500.0, -320.0, 2000.0, 0.0, 0.0]))


def test_gross_notional_sizer_capacity_path_is_bit_identical_to_golden_values() -> None:
    sizer = GrossNotionalSizer(
        gross=1.0,
        max_weight=0.10,
        min_trade_notional=0.0,
        whole_shares=True,
    )
    weights = np.array([0.10, -0.10, 0.08, 0.05])
    prices = np.array([100.0, 200.0, 50.0, 400.0])
    btv = np.array([10_000.0, 5_000.0, 1_000_000.0, 1_000_000.0])
    capital = 1_000_000.0
    res = sizer.to_shares(weights, prices, capital, bar_traded_value=btv, max_participation=0.02)
    np.testing.assert_array_equal(res.shares, np.array([2.0, -0.0, 400.0, 50.0]))
    assert res.unmet_gross_pct == pytest.approx(0.8778787878787879, rel=1e-12)


def test_ewma_volatility_ann_halflife_required_keyword_only() -> None:
    # AMENDMENT 4 item 4: renamed from my guessed `causal_ewma_vol_ann`, moved to
    # `nifty_quant.features.core`, and takes `close` PRICES (not pre-computed returns).
    # AMENDMENT 3 also deletes `window` from this function entirely. Trivial placeholder
    # args are fine: this call must raise before doing any real work.
    close = np.full((10, 3), 100.0)
    day_offsets = np.array([0, 10])
    with pytest.raises(TypeError):
        ewma_volatility_ann(close, day_offsets)


def test_ewma_volatility_ann_halflife_none_raises() -> None:
    close = np.full((10, 3), 100.0)
    day_offsets = np.array([0, 10])
    with pytest.raises((TypeError, ValueError)):
        ewma_volatility_ann(close, day_offsets, halflife=None)
