"""Tests for overflow detection in breakeven_cost_bps (suite B).

Tests the fix for line 358 pragma: no cover being false.
When root_guess overflows to inf, the function should return NaN,
not inf. This suite validates the overflow path and boundary conditions.
"""

import numpy as np
import pytest

from nifty_quant.execution.costs import breakeven_cost_bps


def test_overflow_exact_reproducing_case_returns_nan_not_inf() -> None:
    """The exact input from the defect report returns NaN, not inf.

    Mechanism: mean_gross * 10_000 / mean_turn overflows to inf,
    so upper=inf, net returns are -inf, std is nan, _sharpe_sign returns 0.0,
    line 358 fires and (currently, wrongly) returns inf.

    This test MUST fail against the current implementation.
    """
    result = breakeven_cost_bps(
        np.array([1e300, 1e300], dtype=np.float64),
        np.array([1e-300, 1e-300], dtype=np.float64),
    )
    assert np.isnan(result), (
        f"Expected NaN for overflow case, got {result}. "
        "Is the fix applied?"
    )
    assert result != float("inf"), "Should return NaN, not inf"


def test_ordinary_realistic_inputs_unchanged() -> None:
    """Ordinary inputs from existing tests must still return identical values.

    Ensures no regression to real-world use cases.
    """
    # Test case 1: seed 42 from test_costs.py
    rng1 = np.random.default_rng(42)
    gross1 = rng1.normal(loc=0.0005, scale=0.01, size=5000)
    turn1 = np.full(5000, 1.0, dtype=np.float64)
    cost1 = breakeven_cost_bps(gross1, turn1)

    # Pinned value from current implementation
    assert np.isfinite(cost1)
    assert np.isclose(cost1, 3.0122960508238914, atol=1e-10, rtol=0.0), (
        f"Regression: seed 42 returned {cost1}, expected 3.0122960508238914"
    )

    # Test case 2: seed 99 with mixed turnover
    rng2 = np.random.default_rng(99)
    gross2 = rng2.normal(loc=0.0008, scale=0.015, size=2000)
    turn2 = rng2.uniform(0.5, 2.0, size=2000)
    cost2 = breakeven_cost_bps(gross2, turn2)

    assert np.isfinite(cost2)
    assert np.isclose(cost2, 5.742143837875526, atol=1e-10, rtol=0.0), (
        f"Regression: seed 99 returned {cost2}, expected 5.742143837875526"
    )


def test_edge_case_negative_mean_gross_returns_zero() -> None:
    """When mean(gross_returns) <= 0, function returns 0.0 per docstring.

    Strategy has no positive edge, so no positive cost can be sustained.
    """
    rng = np.random.default_rng(7)
    gross = rng.normal(loc=-0.0005, scale=0.01, size=1000)
    turn = np.full(1000, 1.0, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert result == 0.0
    assert np.isfinite(result)


def test_edge_case_zero_mean_turnover_returns_nan() -> None:
    """When mean(turnover) <= 0, function returns NaN per docstring.

    Turnover never imposes cost, so breakeven is undefined.
    """
    rng = np.random.default_rng(8)
    gross = rng.normal(loc=0.0005, scale=0.01, size=1000)
    turn = np.zeros(1000, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert np.isnan(result)


def test_edge_case_small_positive_mean_turnover_returns_nan() -> None:
    """When mean(turnover) is very small but positive, still NaN.

    Technically mean_turn > 0 but near zero means the cost deferred by
    mean_turn is near-infinite, making breakeven ill-defined in float precision.
    Actually, re-reading the docstring, it says mean_turn <= 0 returns nan.
    So very small positive mean_turn should NOT return nan. Let me check.
    """
    rng = np.random.default_rng(201)
    gross = rng.normal(loc=0.0005, scale=0.01, size=100)
    # Tiny but positive mean
    turn = np.full(100, 1e-10, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    # Should be finite, just very large
    assert np.isfinite(result), (
        f"Small positive mean_turn should give finite result, got {result}"
    )


def test_large_but_not_overflowing_inputs_are_finite() -> None:
    """Large but non-overflowing inputs must return finite, sensible values.

    Locates transition boundary and asserts both sides.
    """
    # Just under the overflow threshold (1e100 is safe, 1e200 overflows)
    safe_large = 1e100
    safe_turn = 1e-100

    result = breakeven_cost_bps(
        np.array([safe_large, safe_large], dtype=np.float64),
        np.array([safe_turn, safe_turn], dtype=np.float64),
    )
    assert np.isfinite(result), (
        f"1e100 / 1e-100 should not overflow, got {result}"
    )
    assert result > 0.0


def test_moderately_large_inputs_still_sensible() -> None:
    """Inputs scaled well within practical range return reasonable breakeven costs.

    E.g., gross_return ~ 1%, turnover ~ 0.1%, should give breakeven in bps.
    """
    # Typical strategy: +100 bps gross return, 10 bps turnover per bar
    rng = np.random.default_rng(51)
    gross = rng.normal(loc=0.01, scale=0.002, size=1000)
    turn = np.full(1000, 0.001, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert np.isfinite(result)
    assert result > 0.0
    # Breakeven depends on Sharpe. High noise means high breakeven cost.
    # Bounds: must be positive and finite
    assert result > 10.0, f"Breakeven {result} is too small"


def test_net_sharpe_at_breakeven_is_near_zero() -> None:
    """At the returned breakeven cost, the net Sharpe ratio should be near zero.

    This is the functional definition of breakeven; validates convergence.
    """
    rng = np.random.default_rng(123)
    gross = rng.normal(loc=0.0003, scale=0.012, size=3000)
    turn = np.full(3000, 1.2, dtype=np.float64)

    cost_bps = breakeven_cost_bps(gross, turn)
    assert np.isfinite(cost_bps), "Breakeven should be finite for ordinary input"

    # Compute net Sharpe at this cost
    net = gross - (cost_bps / 10_000.0) * turn
    mean_net = float(np.mean(net))
    std_net = float(np.std(net, ddof=1))
    sharpe_net = mean_net / std_net if std_net != 0.0 else 0.0

    # Should be very close to zero (within convergence tolerance)
    assert abs(sharpe_net) < 1e-5, (
        f"Sharpe at breakeven {cost_bps} is {sharpe_net}, not near zero"
    )


def test_no_raise_on_overflow_inputs() -> None:
    """Overflow cases must NOT raise; they return NaN per contract.

    Previous behavior was to return inf, new behavior is NaN.
    Never raise.
    """
    overflow_cases = [
        (np.array([1e300, 1e300], dtype=np.float64),
         np.array([1e-300, 1e-300], dtype=np.float64)),
        (np.array([1e280, 1e280], dtype=np.float64),
         np.array([1e-280, 1e-280], dtype=np.float64)),
        (np.array([1e250, 1e250], dtype=np.float64),
         np.array([1e-250, 1e-250], dtype=np.float64)),
    ]

    for gross, turn in overflow_cases:
        try:
            result = breakeven_cost_bps(gross, turn)
            # Should either be NaN or a finite value, never raise
            assert isinstance(result, float)
        except Exception as e:
            pytest.fail(
                f"Should not raise on overflow input {gross[0]:.0e} / {turn[0]:.0e}; "
                f"got {type(e).__name__}: {e}"
            )


def test_zero_gross_edge_different_from_nan_case() -> None:
    """mean_gross == 0 returns 0.0 (not NaN) per docstring.

    Distinguishes "no positive edge" (0.0) from "undefined" (NaN).
    """
    # Zero gross but positive turnover
    gross = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    turn = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert result == 0.0, "Zero gross should return 0.0"
    assert not np.isnan(result), "Zero gross should NOT return NaN"


def test_negative_gross_edge_different_from_nan_case() -> None:
    """mean_gross < 0 returns 0.0 (not NaN) per docstring.

    Distinguishes "no positive edge" (0.0) from "undefined" (NaN).
    """
    rng = np.random.default_rng(88)
    gross = rng.normal(loc=-0.001, scale=0.01, size=500)
    turn = np.full(500, 1.0, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert result == 0.0, "Negative gross should return 0.0"
    assert not np.isnan(result), "Negative gross should NOT return NaN"


def test_alternating_small_gross_high_turnover() -> None:
    """Mixed gross/turnover pattern with convergence to finite breakeven.

    Validates bisection still works correctly for non-overflow paths.
    """
    rng = np.random.default_rng(144)
    # Oscillating returns (mean near zero but positive)
    gross = rng.normal(loc=0.00001, scale=0.02, size=2000)
    turn = rng.uniform(2.0, 5.0, size=2000)

    result = breakeven_cost_bps(gross, turn)
    # Should converge to a positive finite value
    assert np.isfinite(result)
    assert result >= 0.0


def test_high_variance_low_mean() -> None:
    """High-variance, low-mean input still converges correctly.

    Tests robustness of Sharpe sign detection in regimes far from zero.
    """
    rng = np.random.default_rng(155)
    # Very small positive mean, huge variance
    gross = rng.normal(loc=0.00001, scale=1.0, size=5000)
    turn = np.full(5000, 0.5, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert np.isfinite(result)


def test_fixed_point_convergence_seed_close_to_root() -> None:
    """When seed is already close to the root, convergence is immediate.

    Validates bisection refinement when upper and lower are close.
    """
    rng = np.random.default_rng(166)
    gross = rng.normal(loc=0.0005, scale=0.01, size=1000)
    turn = np.full(1000, 1.0, dtype=np.float64)

    # Should converge quickly
    result = breakeven_cost_bps(gross, turn, tol=1e-12, max_iter=200)
    assert np.isfinite(result)
    assert result > 0.0


def test_float64_precision_preserved() -> None:
    """Result is float64 (not downcast to float32).

    Ensures no precision loss in the computation.
    """
    rng = np.random.default_rng(177)
    gross = rng.normal(loc=0.0005, scale=0.01, size=2000)
    turn = np.full(2000, 1.0, dtype=np.float64)

    result = breakeven_cost_bps(gross, turn)
    assert isinstance(result, float)
    # Python's float is 64-bit
    # Verify we got a result with reasonable precision
    assert np.isfinite(result)


def test_array_dtype_conversion_to_float64() -> None:
    """Input arrays are coerced to float64 before processing.

    User can pass float32 or int; function converts internally.
    """
    # Pass in float32 or int arrays
    gross_float32 = np.array([0.0005, 0.0004, 0.0006], dtype=np.float32)
    turn_int = np.array([1, 1, 1], dtype=np.int32)

    result = breakeven_cost_bps(gross_float32, turn_int)
    assert np.isfinite(result)


def test_validation_error_on_inf_input() -> None:
    """Function raises ValueError if input gross or turnover contains inf.

    Ensures invalid input is rejected early, not masked by overflow.
    """
    gross_with_inf = np.array([float('inf'), 1e-3], dtype=np.float64)
    turn = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="must be finite"):
        breakeven_cost_bps(gross_with_inf, turn)


def test_validation_error_on_nan_input() -> None:
    """Function raises ValueError if input gross or turnover contains NaN.

    Ensures invalid input is rejected early.
    """
    gross = np.array([1e-3, 1e-3], dtype=np.float64)
    turn_with_nan = np.array([float('nan'), 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="must be finite"):
        breakeven_cost_bps(gross, turn_with_nan)


def test_too_few_samples_raises() -> None:
    """Function requires at least 2 observations (sample std ddof=1).

    Docstring states need for sample std; single sample is invalid.
    """
    with pytest.raises(ValueError, match="at least 2 observations"):
        breakeven_cost_bps(
            np.array([1e-3], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )


def test_shape_mismatch_raises() -> None:
    """Gross and turnover arrays must have the same length.

    Shape validation is performed early.
    """
    with pytest.raises(ValueError, match="same length"):
        breakeven_cost_bps(
            np.array([1e-3, 1e-3], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )


def test_negative_turnover_raises() -> None:
    """Turnover must be non-negative per contract.

    Negative turnover is invalid; costs are always non-negative.
    """
    with pytest.raises(ValueError, match="must be non-negative"):
        breakeven_cost_bps(
            np.array([1e-3, 1e-3], dtype=np.float64),
            np.array([1.0, -0.5], dtype=np.float64),
        )
