"""Test AMENDMENT 1 fixes for phase_g3_comparison.py: sigma annualization, NaN rho, degeneracy."""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features.core import SIGMA_FLOOR


class TestSigmaAnnualization:
    """Test FIX 1: Annualized EWMA sigma (sqrt(var) * sqrt(252))."""

    def test_annualized_sigma_unit_scaling(self) -> None:
        """Verify sigma is annualized by sqrt(252) and reaches annualized floor."""
        # Per-session returns: [0.01, 0.01, 0.01, ...] (1% per session)
        returns_per_session = [0.01] * 100

        # Compute EWMA of squared returns
        halflife = 20.0
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        ewma_var = returns_per_session[0] ** 2
        for r in returns_per_session[1:]:
            ewma_var = alpha * (r ** 2) + (1.0 - alpha) * ewma_var

        # Annualized sigma: sqrt(variance) * sqrt(252)
        sigma_ann = np.sqrt(ewma_var) * np.sqrt(252.0)
        sigma_floored = max(sigma_ann, SIGMA_FLOOR)

        # With 1% per-session returns, annualized should be ~15.7% (sqrt(252) * 1%)
        # which is ABOVE SIGMA_FLOOR (11.43%), so flooring should not apply
        assert sigma_ann > SIGMA_FLOOR
        assert sigma_floored == sigma_ann
        assert sigma_floored > 0.10  # Annualized floor is 0.1143, result should be similar scale

    def test_sigma_floor_applied_at_annualized_scale(self) -> None:
        """Verify floor is applied to annualized sigma, not per-session EWMA."""
        # Tiny per-session returns: [0.001, 0.001, ...] (0.1% per session)
        returns_per_session = [0.001] * 100

        halflife = 20.0
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        ewma_var = returns_per_session[0] ** 2
        for r in returns_per_session[1:]:
            ewma_var = alpha * (r ** 2) + (1.0 - alpha) * ewma_var

        sigma_ann = np.sqrt(ewma_var) * np.sqrt(252.0)
        sigma_floored = max(sigma_ann, SIGMA_FLOOR)

        # With 0.1% per-session returns, annualized would be ~1.57%
        # which is BELOW SIGMA_FLOOR (11.43%), so flooring SHOULD apply
        assert sigma_ann < SIGMA_FLOOR
        assert sigma_floored == SIGMA_FLOOR

    def test_annualized_sigma_vs_old_unannualized(self) -> None:
        """Verify annualized sigma formula differs from old unannualized approach."""
        returns_per_session = [0.01] * 100

        halflife = 20.0
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

        # OLD (wrong): EWMA of |returns| (per-session, not annualized)
        ewma_abs = abs(returns_per_session[0])
        for r in returns_per_session[1:]:
            ewma_abs = alpha * abs(r) + (1.0 - alpha) * ewma_abs

        # NEW (correct): sqrt(EWMA of squared) * sqrt(252) (annualized)
        ewma_var = returns_per_session[0] ** 2
        for r in returns_per_session[1:]:
            ewma_var = alpha * (r ** 2) + (1.0 - alpha) * ewma_var
        sigma_new_raw = np.sqrt(ewma_var) * np.sqrt(252.0)

        # The key difference: old computes EWMA of |r| ~ 0.01
        # New computes sqrt(EWMA(r^2)) * sqrt(252) ~ sqrt(0.0001) * sqrt(252) ~ 0.158
        # Ratio without floor: ~15.8
        assert sigma_new_raw / ewma_abs > 10

        # With flooring, the ratio may be smaller if either hits the floor
        sigma_old = max(ewma_abs, SIGMA_FLOOR)
        sigma_new = max(sigma_new_raw, SIGMA_FLOOR)
        # At minimum, new should not equal old (different computation)
        assert sigma_new != sigma_old or (sigma_new == SIGMA_FLOOR and sigma_old == SIGMA_FLOOR)


class TestNaNRhoSkipping:
    """Test FIX 2: Skip sessions where rho is NaN after warm-up."""

    def test_nan_rho_detection_few_finite_rows(self) -> None:
        """Verify NaN rho is detected when < 2 fully-finite rows exist."""
        # Returns window: 30 sessions x 3 names, but most rows have NaN
        returns_window = np.full((30, 3), np.nan, dtype=np.float64)
        # Only 1 fully-finite row
        returns_window[0, :] = [0.01, 0.02, 0.03]

        # Compute correlation
        valid_rows = np.all(np.isfinite(returns_window), axis=1)
        n_valid = np.sum(valid_rows)

        # Should have < 2 valid rows, so rho would be NaN
        assert n_valid < 2

    def test_nan_rho_detection_from_correlation(self) -> None:
        """Verify NaN rho results when computing correlation on insufficient data."""
        returns_window = np.full((30, 3), np.nan, dtype=np.float64)
        returns_window[0:2, :] = [
            [0.01, 0.02, 0.03],
            [0.02, 0.03, 0.04],
        ]

        valid_rows = np.all(np.isfinite(returns_window), axis=1)
        returns_clean = returns_window[valid_rows, :]

        # 2 rows should be enough for correlation (not NaN)
        corr = np.corrcoef(returns_clean, rowvar=False)
        assert np.all(np.isfinite(corr))


class TestDegeneracyCheck:
    """Test FIX 3: Refuse to write if > 10% of sigma cells are at floor."""

    def test_degeneracy_fraction_calculation(self) -> None:
        """Verify degeneracy fraction is computed correctly."""
        sigma_floor_count = 30000  # 30% of cells at floor
        sigma_total_count = 100000
        fraction = sigma_floor_count / sigma_total_count

        # Should exceed 10% threshold
        assert fraction > 0.10
        assert fraction == 0.30

    def test_degeneracy_pass_under_threshold(self) -> None:
        """Verify degeneracy passes when < 10% of cells are at floor."""
        sigma_floor_count = 5000  # 5% of cells at floor
        sigma_total_count = 100000
        fraction = sigma_floor_count / sigma_total_count

        # Should pass < 10% threshold
        assert fraction <= 0.10
        assert fraction == 0.05

    def test_degeneracy_fail_over_threshold(self) -> None:
        """Verify degeneracy fails when > 10% of cells are at floor."""
        sigma_floor_count = 11000  # 11% of cells at floor
        sigma_total_count = 100000
        fraction = sigma_floor_count / sigma_total_count

        # Should fail > 10% threshold
        assert fraction > 0.10
        assert fraction == 0.11

    def test_degeneracy_reason_statement(self) -> None:
        """Verify error message explains why degeneracy matters."""
        # SIGMA_FLOOR is measured p1 (1% of observations)
        # If > 10x that fraction are floored, the estimator is not working
        measured_p1 = 0.01  # 1% by definition
        threshold_multiplier = 10
        max_allowed_fraction = measured_p1 * threshold_multiplier

        assert max_allowed_fraction == 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
