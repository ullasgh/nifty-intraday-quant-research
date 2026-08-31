"""Test coverage gaps in weighting.py module.

Covers edge cases and exception paths not exercised by the obligation tests.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nifty_quant.backtest.weighting import (
    WEIGHT_SCHEME_REGISTRY,
    SchemeResult,
    apply_weight_scheme,
)


class TestEdgeCases:
    """Edge cases in individual schemes."""

    def test_equal_weight_all_masked_out(self):
        """Test equal_weight when no names are tradable."""
        signal = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.zeros(3, dtype=bool)  # All masked

        weights = WEIGHT_SCHEME_REGISTRY["equal_weight"](signal, sigma, 0.0, tradable)
        assert np.allclose(weights, 0.0)
        assert np.sum(np.abs(weights)) == pytest.approx(0.0)

    def test_z_weight_constant_signal(self):
        """Test z_weight when signal is constant (std=0)."""
        signal = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
        tradable = np.ones(4, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["z_weight"](signal, sigma, 0.0, tradable)
        # Constant signal -> zero z-scores -> zero weights (or all masked to zero)
        assert np.allclose(weights, 0.0)

    def test_z_weight_all_masked(self):
        """Test z_weight when no names are tradable."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.zeros(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["z_weight"](signal, sigma, 0.0, tradable)
        assert np.allclose(weights, 0.0)

    def test_z_over_vol_constant_signal(self):
        """Test z_over_vol when signal is constant (std=0)."""
        signal = np.array([1.5, 1.5, 1.5], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["z_over_vol"](signal, sigma, 0.0, tradable)
        # Constant signal -> zero z-scores -> zero weights
        assert np.allclose(weights, 0.0)

    def test_z_over_vol_all_nan_sigma(self):
        """Test z_over_vol when all sigmas are NaN."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["z_over_vol"](signal, sigma, 0.0, tradable)
        # All NaN sigmas -> all zero weights
        assert np.allclose(weights, 0.0)

    def test_rank_weight_all_masked(self):
        """Test rank_weight when no names are tradable."""
        signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
        tradable = np.zeros(4, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["rank_weight"](signal, sigma, 0.0, tradable)
        assert np.allclose(weights, 0.0)

    def test_rank_weight_identical_signals(self):
        """Test rank_weight when all signals are identical."""
        signal = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
        tradable = np.ones(4, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["rank_weight"](signal, sigma, 0.0, tradable)
        # Identical signals -> all have same rank -> zero rank-centered weights
        # argsort of constant gives [0, 1, 2, 3] (stable sort)
        # rank-centered: [0-1.5, 1-1.5, 2-1.5, 3-1.5] = [-1.5, -0.5, 0.5, 1.5]
        # sum(abs) = 1.5 + 0.5 + 0.5 + 1.5 = 4.0
        # normalized: [-1.5/4, -0.5/4, 0.5/4, 1.5/4]
        # This should sum to gross
        assert np.sum(np.abs(weights)) == pytest.approx(1.0)

    def test_risk_parity_all_masked(self):
        """Test risk_parity when no names are tradable."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.zeros(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["risk_parity"](signal, sigma, 0.0, tradable)
        assert np.allclose(weights, 0.0)

    def test_risk_parity_with_scalar_equicorrelation_nonzero(self):
        """Test risk_parity with scalar rho != 0 (equicorrelation structure)."""
        signal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        sigma = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float64)  # Equal vol
        tradable = np.ones(4, dtype=bool)
        rho = 0.3  # Non-zero correlation

        weights = WEIGHT_SCHEME_REGISTRY["risk_parity"](signal, sigma, rho, tradable)
        # On equal vol, should still be equal weight
        expected = np.full(4, 0.25, dtype=np.float64)
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_covariance_aware_all_masked(self):
        """Test covariance_aware when no names are tradable."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.zeros(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, 0.0, tradable
        )
        assert np.allclose(weights, 0.0)

    def test_covariance_aware_with_2d_covariance(self):
        """Test covariance_aware with explicit 2-D covariance matrix."""
        signal = np.array([0.5, -0.5, 0.2, -0.3], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
        # Diagonal covariance
        cov = np.diag(sigma ** 2)
        tradable = np.ones(4, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # On diagonal covariance, should match inverse_vol
        inv_vol_weights = WEIGHT_SCHEME_REGISTRY["inverse_vol"](
            signal, sigma, 0.0, tradable
        )
        np.testing.assert_allclose(weights, inv_vol_weights, atol=1e-10)

    def test_covariance_aware_with_off_diagonal_covariance(self):
        """Test covariance_aware with off-diagonal structure."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        # Covariance with off-diagonal terms
        cov = np.array(
            [[0.01, 0.005, 0.003], [0.005, 0.04, 0.01], [0.003, 0.01, 0.0225]],
            dtype=np.float64,
        )
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # Should return valid weights
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))
        assert np.sum(np.abs(weights)) == pytest.approx(1.0)

    def test_covariance_aware_with_scalar_nonzero_rho(self):
        """Test covariance_aware with scalar correlation (rho != 0)."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        rho = 0.4  # Non-zero correlation
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, rho, tradable
        )
        # Should return valid weights
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))
        assert np.sum(np.abs(weights)) == pytest.approx(1.0)

    def test_covariance_aware_ill_conditioned_matrix(self):
        """Test covariance_aware with ill-conditioned covariance."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        # Ill-conditioned covariance (nearly singular)
        cov = np.array(
            [[1.0, 0.9999, 0.9998], [0.9999, 1.0, 0.9999], [0.9998, 0.9999, 1.0]],
            dtype=np.float64,
        )
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # Should return valid weights despite ill-conditioning
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))

    def test_covariance_aware_singular_matrix_regularization(self):
        """Test covariance_aware with singular/near-singular covariance."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        # Singular-like covariance (very high correlation)
        cov = np.array(
            [[0.01, 0.009999, 0.009999], [0.009999, 0.01, 0.009999], [0.009999, 0.009999, 0.01]],
            dtype=np.float64,
        )
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # Should handle the singular/ill-conditioned case gracefully
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))
        assert np.sum(np.abs(weights)) == pytest.approx(1.0, abs=1e-9)

    def test_covariance_aware_very_high_condition_number(self):
        """Test covariance_aware with extremely ill-conditioned matrix."""
        signal = np.array([1.0, 2.0], dtype=np.float64)
        sigma = np.array([1e-8, 1.0], dtype=np.float64)  # Extreme scale difference
        # Build a covariance with extreme condition number
        cov = np.array([[1e-16, 1e-8], [1e-8, 1.0]], dtype=np.float64)
        tradable = np.ones(2, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # Should handle even with high condition number
        assert weights.shape == (2,)
        assert np.all(np.isfinite(weights))

    def test_covariance_aware_truly_singular_matrix(self):
        """Test covariance_aware with a truly singular covariance matrix."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        # A truly singular covariance (rank deficient)
        # Two identical rows/cols
        cov = np.array(
            [[1.0, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
            dtype=np.float64,
        )
        tradable = np.ones(3, dtype=bool)

        weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma, cov, tradable
        )
        # Should handle even with singular matrix
        assert weights.shape == (3,)
        assert np.all(np.isfinite(weights))

    def test_covariance_aware_inversion_exception_fallback(self):
        """Test covariance_aware fallback when matrix inversion fails."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        cov = np.eye(3, dtype=np.float64)  # Simple well-conditioned matrix
        tradable = np.ones(3, dtype=bool)

        # Mock np.linalg.inv in weighting module to throw exception
        import nifty_quant.backtest.weighting as weighting_module

        original_inv = np.linalg.inv
        call_count = [0]

        def mock_inv(a):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call fails
                raise np.linalg.LinAlgError("Mocked inversion failure")
            else:
                # Subsequent calls succeed
                return original_inv(a)

        with patch.object(weighting_module.np.linalg, "inv", side_effect=mock_inv):
            weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
                signal, sigma, cov, tradable
            )
            # Should fall back to diagonal approach
            assert weights.shape == (3,)
            assert np.all(np.isfinite(weights))


class TestApplyWeightSchemeEdgeCases:
    """Edge cases in apply_weight_scheme wrapper."""

    def test_apply_weight_scheme_with_invalid_name(self):
        """Test apply_weight_scheme with unknown scheme name."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        with pytest.raises(ValueError, match="Unknown scheme"):
            apply_weight_scheme(
                "nonexistent_scheme",
                signal=signal,
                sigma=sigma,
                corr=0.0,
                tradable=tradable,
            )

    def test_apply_weight_scheme_result_type(self):
        """Test that apply_weight_scheme returns SchemeResult."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        result = apply_weight_scheme(
            "equal_weight",
            signal=signal,
            sigma=sigma,
            corr=0.0,
            tradable=tradable,
            gross=1.0,
            max_weight=1.0,
        )

        assert isinstance(result, SchemeResult)
        assert hasattr(result, "weights")
        assert hasattr(result, "clip_binding")
        assert hasattr(result, "gross_before_clip")

    def test_apply_weight_scheme_no_clip(self):
        """Test clip_binding=False when max_weight is loose."""
        signal = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        sigma = np.array([0.15, 0.30, 0.20], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        result = apply_weight_scheme(
            "inverse_vol",
            signal=signal,
            sigma=sigma,
            corr=0.0,
            tradable=tradable,
            gross=1.0,
            max_weight=1.0,  # Loose bound
        )

        assert result.clip_binding is False
        assert result.gross_before_clip == pytest.approx(1.0)

    def test_apply_weight_scheme_clip_binding_true(self):
        """Test clip_binding=True when max_weight is tight."""
        signal = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        sigma = np.array([0.02, 0.30, 0.20], dtype=np.float64)  # Very small sigma at [0]
        tradable = np.ones(3, dtype=bool)

        result = apply_weight_scheme(
            "inverse_vol",
            signal=signal,
            sigma=sigma,
            corr=0.0,
            tradable=tradable,
            gross=1.0,
            max_weight=0.3,  # Tight bound
        )

        assert result.clip_binding is True
        # Clipped weights should not exceed max_weight
        assert np.all(np.abs(result.weights) <= 0.3 + 1e-9)

    def test_apply_weight_scheme_clipping_no_renormalization(self):
        """Test that clipping doesn't re-normalize weights."""
        signal = np.array([1.0, -0.8, 0.6], dtype=np.float64)
        sigma = np.array([0.02, 0.30, 0.25], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        result = apply_weight_scheme(
            "inverse_vol",
            signal=signal,
            sigma=sigma,
            corr=0.0,
            tradable=tradable,
            gross=1.0,
            max_weight=0.15,
        )

        # Gross after clipping should be <= gross_before_clip (no re-normalization)
        gross_after = np.sum(np.abs(result.weights))
        assert gross_after <= result.gross_before_clip + 1e-9

    def test_registry_has_all_seven_schemes(self):
        """Test that registry contains exactly the seven required schemes."""
        required_schemes = {
            "equal_weight",
            "inverse_vol",
            "z_weight",
            "z_over_vol",
            "rank_weight",
            "risk_parity",
            "covariance_aware",
        }
        assert set(WEIGHT_SCHEME_REGISTRY.keys()) == required_schemes

    def test_scheme_with_different_gross_values(self):
        """Test schemes with non-unit gross."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        tradable = np.ones(3, dtype=bool)

        for scheme_name in WEIGHT_SCHEME_REGISTRY:
            # Test with gross=2.0
            weights = WEIGHT_SCHEME_REGISTRY[scheme_name](
                signal, sigma, 0.0, tradable, gross=2.0
            )
            assert np.sum(np.abs(weights)) == pytest.approx(2.0, abs=1e-9)

            # Test with gross=0.5
            weights = WEIGHT_SCHEME_REGISTRY[scheme_name](
                signal, sigma, 0.0, tradable, gross=0.5
            )
            assert np.sum(np.abs(weights)) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------------
# Review-derived edits (2026-08-31): pragma audit of covariance_aware. The three
# "unreachable" pragmas were removed -- one guarded an always-true branch, one claimed
# numpy no longer raises on singular matrices (it does; pinv now makes the question
# moot), and one guarded a branch that IS reachable (all-False tradable). These tests
# pin the post-audit behaviour.
# ---------------------------------------------------------------------------------


def test_covariance_aware_all_masked_returns_zero_book_no_error():
    """The sum_abs == 0 branch is REACHABLE: an all-False tradable masks every weight
    to zero and must return the all-zero book, never divide by zero or raise."""
    signal = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    sigma = np.array([0.2, 0.3, 0.25], dtype=np.float64)
    tradable = np.zeros(3, dtype=bool)

    weights = WEIGHT_SCHEME_REGISTRY["covariance_aware"](
        signal, sigma, 0.5, tradable, gross=1.0
    )
    weights = np.asarray(weights, dtype=np.float64)
    assert np.all(weights == 0.0)
    assert np.all(np.isfinite(weights))


def test_covariance_aware_ill_conditioned_uses_exact_pinv_no_shrinkage():
    """An ill-conditioned covariance is solved exactly via pinv -- there is no hidden
    condition-number cutoff that silently switches to a shrunk matrix (rule 8: the
    reviewed-out implementation blended 0.5*cov + 0.5*diag above cond 1e10, changing
    weights by an unstated threshold). The expected value is computed independently
    here from the SAME matrix the scheme receives."""
    n = 3
    # Nearly-collinear two-name block drives the condition number far above 1e10.
    sigma = np.array([0.2, 0.2, 0.3], dtype=np.float64)
    corr = np.array(
        [
            [0.04, 0.04 - 1e-13, 0.0],
            [0.04 - 1e-13, 0.04, 0.0],
            [0.0, 0.0, 0.09],
        ],
        dtype=np.float64,
    )
    assert np.linalg.cond(corr) > 1e10
    signal = np.ones(n, dtype=np.float64)
    tradable = np.ones(n, dtype=bool)

    weights = np.asarray(
        WEIGHT_SCHEME_REGISTRY["covariance_aware"](signal, sigma, corr, tradable, gross=1.0),
        dtype=np.float64,
    )

    expected_raw = np.linalg.pinv(corr) @ np.ones(n)
    expected = expected_raw / np.sum(np.abs(expected_raw))
    np.testing.assert_allclose(weights, expected, atol=1e-12)


def test_covariance_aware_singular_covariance_finite_deterministic():
    """A SINGULAR covariance (two perfectly correlated names) must not raise and must
    give the deterministic minimum-norm pinv solution -- numpy's inv DOES raise
    LinAlgError here, contrary to the reviewed-out comment claiming otherwise."""
    sigma = np.array([0.2, 0.2, 0.3], dtype=np.float64)
    corr = np.array(
        [
            [0.04, 0.04, 0.0],
            [0.04, 0.04, 0.0],
            [0.0, 0.0, 0.09],
        ],
        dtype=np.float64,
    )
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(corr)  # documents WHY pinv, not inv

    signal = np.ones(3, dtype=np.float64)
    tradable = np.ones(3, dtype=bool)
    weights = np.asarray(
        WEIGHT_SCHEME_REGISTRY["covariance_aware"](signal, sigma, corr, tradable, gross=1.0),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(weights))
    assert np.sum(np.abs(weights)) == pytest.approx(1.0)


def test_covariance_aware_nan_sigma_off_tradable_solves_on_submatrix():
    """Defect found by the G3 runner (session 2018-02-28, reproduced minimally): a
    NON-tradable name with NaN sigma poisoned the full-universe covariance and made
    pinv's SVD fail outright. The solve now runs on the tradable-and-finite-sigma
    submatrix; the NaN name carries no information the solve needs (its weight is
    exactly 0.0 by contract). Weights must equal the same call with the name absent,
    scattered back."""
    sigma_with_nan = np.array([0.2, 0.3, np.nan, 0.25, 0.4], dtype=np.float64)
    tradable = np.array([True, True, False, True, True])
    signal = np.array([0.9, 0.1, 0.5, 0.7, 0.3], dtype=np.float64)

    w_full = np.asarray(
        WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal, sigma_with_nan, 0.3, tradable, gross=1.0
        ),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(w_full))
    assert w_full[2] == 0.0

    keep = np.array([0, 1, 3, 4])
    w_sub = np.asarray(
        WEIGHT_SCHEME_REGISTRY["covariance_aware"](
            signal[keep], sigma_with_nan[keep], 0.3, np.ones(4, dtype=bool), gross=1.0
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(w_full[keep], w_sub, atol=1e-12)
