"""Tests for the overflow defect in breakeven_cost_bps.

The defect occurs when root_guess or upper overflows to non-finite values
before bracketing, causing the function to return inf instead of nan.
"""

import math

import numpy as np

from nifty_quant.execution.costs import breakeven_cost_bps


def test_overflow_repro_returns_nan_not_inf() -> None:
    """Primary regression guard: overflow case must return nan, not inf."""
    gross = np.array([1e300, 1e300])
    turn = np.array([1e-300, 1e-300])

    result = breakeven_cost_bps(gross, turn)

    assert np.isnan(result)


def test_overflow_repro_does_not_raise() -> None:
    """Overflow case must not raise an exception."""
    gross = np.array([1e300, 1e300])
    turn = np.array([1e-300, 1e-300])

    # Should not raise
    breakeven_cost_bps(gross, turn)


def test_ordinary_input_a_exact_value() -> None:
    """Ordinary in-range input with seeded RNG must be unaffected."""
    rng = np.random.default_rng(42)
    gross_returns = rng.normal(loc=0.0005, scale=0.01, size=5000)
    turnover = np.full(5000, 1.0)

    result = breakeven_cost_bps(gross_returns, turnover)

    assert math.isclose(result, 3.0122960508238914, rel_tol=1e-12, abs_tol=0.0)


def test_ordinary_input_b_small_array_exact_value() -> None:
    """Small deterministic array must produce exact pinned value."""
    gross2 = np.array([0.001, 0.002, -0.0005, 0.0015, 0.0008])
    turn2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = breakeven_cost_bps(gross2, turn2)

    assert result == 9.599999999965076


def test_near_seed_bracketing_exact_value() -> None:
    """Near-seed bracketing case confirms normal convergence untouched."""
    gross3 = np.full(10, 0.0001)
    gross3[0] = 0.0011
    turn3 = np.full(10, 1.0)

    result = breakeven_cost_bps(gross3, turn3)

    assert result == 2.000000000029104


def test_negative_edge_returns_zero() -> None:
    """Documented edge case: mean_gross <= 0 returns 0.0 exactly."""
    rng = np.random.default_rng(7)
    gross_returns = rng.normal(loc=-0.0005, scale=0.01, size=1000)
    turnover = np.full(1000, 1.0)

    result = breakeven_cost_bps(gross_returns, turnover)

    assert result == 0.0


def test_degenerate_turnover_returns_nan() -> None:
    """Documented edge case: mean_turn <= 0 returns nan."""
    rng = np.random.default_rng(8)
    gross_returns = rng.normal(loc=0.0005, scale=0.01, size=1000)
    turnover = np.zeros(1000)

    result = breakeven_cost_bps(gross_returns, turnover)

    assert np.isnan(result)


def test_boundary_below_overflow_finite() -> None:
    """Boundary probe: exp=293 does not overflow, returns finite exact value."""
    gross = np.array([10.0**293, 10.0**293])
    turn = np.array([1e-10, 1e-10])

    result = breakeven_cost_bps(gross, turn)

    assert np.isfinite(result)
    assert result == 9.999999999999999e306


def test_boundary_at_overflow_returns_nan() -> None:
    """Boundary probe: exp=294 overflows, must return nan."""
    gross = np.array([10.0**294, 10.0**294])
    turn = np.array([1e-10, 1e-10])

    result = breakeven_cost_bps(gross, turn)

    assert np.isnan(result)
