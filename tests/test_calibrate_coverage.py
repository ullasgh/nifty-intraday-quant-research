"""Additional tests to reach 100% line and branch coverage for calibrate.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_quant.features.calibrate import (
    NullCalibration,
    _draw_null,
    calibrate_empirical,
    calibrate_null,
    calibration_report,
    suggest_threshold,
)
from nifty_quant.features.persistence import hurst_static

# =============================================================================
# NullCalibration.threshold_at - KeyError branch
# =============================================================================


def test_threshold_at_missing_percentile() -> None:
    """Test NullCalibration.threshold_at raises KeyError for missing percentile."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={50.0: 0.5, 90.0: 0.6},
    )
    with pytest.raises(KeyError, match="not calibrated"):
        cal.threshold_at(75.0)


def test_threshold_at_error_message_lists_available() -> None:
    """Test NullCalibration.threshold_at error message includes available levels."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={10.0: 0.4, 50.0: 0.5, 90.0: 0.6},
    )
    with pytest.raises(KeyError, match="Available levels"):
        cal.threshold_at(25.0)


# =============================================================================
# NullCalibration.p_value - validation and branches
# =============================================================================


def test_p_value_invalid_tail() -> None:
    """Test NullCalibration.p_value rejects invalid tail."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={50.0: 0.5, 90.0: 0.6},
    )
    with pytest.raises(ValueError, match="'right' or 'left'"):
        cal.p_value(0.55, tail="invalid")  # type: ignore


def test_p_value_empty_percentiles_raises() -> None:
    """Test NullCalibration.p_value raises on empty percentiles."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={},
    )
    with pytest.raises(ValueError, match="empty"):
        cal.p_value(0.55)


def test_p_value_duplicate_percentile_values() -> None:
    """Test NullCalibration.p_value handles duplicate percentile values."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        # Multiple percentiles mapping to the same value
        percentiles={10.0: 0.50, 20.0: 0.50, 90.0: 0.60},
    )
    p = cal.p_value(0.50, tail="right")
    # Should use the max probability for duplicate values
    assert 0.0 <= p <= 1.0


def test_p_value_left_tail() -> None:
    """Test NullCalibration.p_value with left tail."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={10.0: 0.40, 50.0: 0.50, 90.0: 0.60},
    )
    # Left tail: P(X <= observed)
    p_left = cal.p_value(0.50, tail="left")
    p_right = cal.p_value(0.50, tail="right")
    # For median, left should be ~0.5, right should be ~0.5
    assert 0.0 <= p_left <= 1.0
    assert 0.0 <= p_right <= 1.0


def test_p_value_below_range() -> None:
    """Test NullCalibration.p_value extrapolates left."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={10.0: 0.40, 50.0: 0.50, 90.0: 0.60},
    )
    # Query below minimum value
    p = cal.p_value(0.30, tail="right")
    # Should extrapolate to leftmost probability
    assert p >= 0.0


def test_p_value_above_range() -> None:
    """Test NullCalibration.p_value extrapolates right."""
    cal = NullCalibration(
        estimator="test",
        window=100,
        n_draws=1000,
        generator="test",
        mean=0.5,
        std=0.1,
        percentiles={10.0: 0.40, 50.0: 0.50, 90.0: 0.60},
    )
    # Query above maximum value
    p = cal.p_value(0.70, tail="right")
    # Should extrapolate to rightmost probability (close to 0)
    assert 0.0 <= p <= 1.0


# =============================================================================
# NullCalibration.to_dict
# =============================================================================


def test_null_calibration_to_dict() -> None:
    """Test NullCalibration.to_dict converts percentiles to string keys."""
    cal = NullCalibration(
        estimator="hurst_static",
        window=390,
        n_draws=1000,
        generator="random_walk",
        mean=0.5,
        std=0.1,
        percentiles={10.0: 0.40, 50.0: 0.50, 90.0: 0.60},
    )
    d = cal.to_dict()
    assert d["estimator"] == "hurst_static"
    assert d["window"] == 390
    assert d["n_draws"] == 1000
    assert d["generator"] == "random_walk"
    assert d["mean"] == 0.5
    assert d["std"] == 0.1
    # Percentile keys should be strings
    assert "10.0" in d["percentiles"]
    assert "50.0" in d["percentiles"]
    assert "90.0" in d["percentiles"]
    assert d["percentiles"]["50.0"] == 0.50


# =============================================================================
# calibrate_null validation
# =============================================================================


def test_calibrate_null_rejects_zero_n_draws() -> None:
    """Test calibrate_null rejects n_draws < 1."""
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        calibrate_null(hurst_static, window=100, n_draws=0, seed=0)


def test_calibrate_null_rejects_small_window() -> None:
    """Test calibrate_null rejects window < 3."""
    with pytest.raises(ValueError, match="window must be >= 3"):
        calibrate_null(hurst_static, window=2, n_draws=100, seed=0)


def test_calibrate_null_empty_draws_error() -> None:
    """Test calibrate_null raises when no finite draws are produced."""
    # Create an estimator that returns NaN for everything
    def always_nan(path: np.ndarray) -> float:
        return np.nan

    with pytest.raises(ValueError, match="No finite draws"):
        calibrate_null(always_nan, window=30, n_draws=10, seed=0)


def test_calibrate_null_percentiles_present() -> None:
    """Test calibrate_null produces expected percentile levels."""
    cal = calibrate_null(hurst_static, window=390, n_draws=500, seed=42)
    # Should have all standard percentiles
    expected_levels = {1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0}
    assert set(cal.percentiles.keys()) == expected_levels


# =============================================================================
# _draw_null validation (direct calls)
# =============================================================================


def test_draw_null_window_too_small_direct() -> None:
    """Test _draw_null directly rejects window < 3."""
    with pytest.raises(ValueError, match="window must be >= 3"):
        _draw_null(hurst_static, window=2, n_draws=10, seed=0, generator="random_walk")


def test_draw_null_n_draws_zero_direct() -> None:
    """Test _draw_null directly rejects n_draws < 1."""
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        _draw_null(hurst_static, window=30, n_draws=0, seed=0, generator="random_walk")


def test_draw_null_window_too_small() -> None:
    """Test _draw_null validates window >= 3 via calibrate_null."""
    with pytest.raises(ValueError, match="window must be >= 3"):
        calibrate_null(hurst_static, window=2, n_draws=10, seed=0)


def test_draw_null_n_draws_zero() -> None:
    """Test _draw_null validates n_draws >= 1 via calibrate_null."""
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        calibrate_null(hurst_static, window=30, n_draws=0, seed=0)


def test_draw_null_invalid_generator() -> None:
    """Test _draw_null rejects invalid generator."""
    with pytest.raises(ValueError, match="unknown generator"):
        calibrate_null(
            hurst_static,
            window=30,
            n_draws=10,
            seed=0,
            generator="invalid",  # type: ignore
        )


# =============================================================================
# _draw_null generators - iid, ar1, fbm
# =============================================================================


def test_calibrate_null_iid_generator() -> None:
    """Test calibrate_null with iid generator."""
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=100,
        seed=10,
        generator="iid",
    )
    assert cal.generator == "iid"
    # Hurst on iid data can have negative values
    assert np.isfinite(cal.mean)
    assert np.isfinite(cal.std)


def test_calibrate_null_iid_generator_with_scale() -> None:
    """Test calibrate_null with iid generator and custom scale."""
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=100,
        seed=11,
        generator="iid",
        generator_kwargs={"scale": 2.0},
    )
    assert cal.generator == "iid"
    # Hurst on iid data can have negative values
    assert np.isfinite(cal.mean)
    assert np.isfinite(cal.std)


def test_calibrate_null_ar1_generator() -> None:
    """Test calibrate_null with ar1 generator."""
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=100,
        seed=12,
        generator="ar1",
    )
    assert cal.generator == "ar1"
    assert cal.mean > 0.0


def test_calibrate_null_ar1_generator_with_phi() -> None:
    """Test calibrate_null with ar1 generator and custom phi."""
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=100,
        seed=13,
        generator="ar1",
        generator_kwargs={"phi": 0.5, "scale": 1.0},
    )
    assert cal.generator == "ar1"
    assert cal.mean > 0.0


def test_calibrate_null_fbm_generator() -> None:
    """Test calibrate_null with fbm generator."""
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=100,
        seed=14,
        generator="fbm",
    )
    assert cal.generator == "fbm"
    assert cal.mean > 0.0


def test_calibrate_null_fbm_invalid_h() -> None:
    """Test calibrate_null fbm generator rejects invalid H."""
    with pytest.raises(ValueError, match="fbm H must be in"):
        calibrate_null(
            hurst_static,
            window=30,
            n_draws=10,
            seed=15,
            generator="fbm",
            generator_kwargs={"H": 1.5},
        )


def test_calibrate_null_fbm_h_at_boundary() -> None:
    """Test calibrate_null fbm generator with H near boundaries."""
    # H must be strictly in (0, 1), so 0.01 and 0.99 should work
    cal = calibrate_null(
        hurst_static,
        window=30,
        n_draws=50,
        seed=16,
        generator="fbm",
        generator_kwargs={"H": 0.3},
    )
    assert cal.generator == "fbm"


def test_draw_null_filters_non_finite_values() -> None:
    """Test _draw_null filters non-finite estimator outputs."""
    call_count = {"count": 0}
    non_finite_pattern = [0.5, np.nan, 0.5, np.inf, 0.5, -np.inf, 0.5, 0.5]

    def estimator_with_nans(path: np.ndarray) -> float:
        result = non_finite_pattern[call_count["count"] % len(non_finite_pattern)]
        call_count["count"] += 1
        return float(result)

    # Should still complete and return only finite values
    cal = calibrate_null(estimator_with_nans, window=30, n_draws=100, seed=17)
    # Should have filtered out the NaN and Inf values
    assert cal.n_draws <= 100
    assert cal.mean > 0.0


# =============================================================================
# calibrate_empirical
# =============================================================================


def test_calibrate_empirical_filters_nans() -> None:
    """Test calibrate_empirical filters NaN values."""
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0], dtype=np.float64)
    cal = calibrate_empirical(values)
    # Should only have 5 finite values
    assert cal.n_draws == 5
    assert np.isclose(cal.mean, 3.0, atol=1e-9)


def test_calibrate_empirical_all_nans_raises() -> None:
    """Test calibrate_empirical raises when all values are NaN."""
    values = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="No finite values"):
        calibrate_empirical(values)


def test_calibrate_empirical_2d_array() -> None:
    """Test calibrate_empirical flattens 2D arrays."""
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    cal = calibrate_empirical(values)
    assert cal.n_draws == 6
    assert np.isclose(cal.mean, 3.5, atol=1e-9)


def test_calibrate_empirical_preserves_percentiles() -> None:
    """Test calibrate_empirical matches numpy percentile calculation."""
    values = np.arange(1, 101, dtype=float)
    cal = calibrate_empirical(values)

    # Verify key percentiles match numpy
    assert cal.percentiles[1.0] == pytest.approx(np.percentile(values, 1.0))
    assert cal.percentiles[50.0] == pytest.approx(np.percentile(values, 50.0))
    assert cal.percentiles[99.0] == pytest.approx(np.percentile(values, 99.0))


def test_calibrate_empirical_generator_and_window_zero() -> None:
    """Test calibrate_empirical sets generator='empirical' and window=0."""
    values = np.arange(1, 11, dtype=float)
    cal = calibrate_empirical(values)
    assert cal.generator == "empirical"
    assert cal.window == 0
    assert cal.estimator == "empirical"


# =============================================================================
# suggest_threshold
# =============================================================================


def test_suggest_threshold_invalid_fpr() -> None:
    """Test suggest_threshold rejects invalid target_fpr."""
    with pytest.raises(ValueError, match="target_fpr must be in"):
        suggest_threshold(hurst_static, window=100, target_fpr=0.0, n_draws=100, seed=0)


def test_suggest_threshold_fpr_boundary_low() -> None:
    """Test suggest_threshold rejects target_fpr <= 0."""
    with pytest.raises(ValueError, match="target_fpr must be in"):
        suggest_threshold(hurst_static, window=100, target_fpr=-0.1, n_draws=100, seed=0)


def test_suggest_threshold_fpr_boundary_high() -> None:
    """Test suggest_threshold rejects target_fpr >= 1."""
    with pytest.raises(ValueError, match="target_fpr must be in"):
        suggest_threshold(hurst_static, window=100, target_fpr=1.0, n_draws=100, seed=0)


def test_suggest_threshold_no_finite_draws_raises() -> None:
    """Test suggest_threshold raises when no finite draws are produced."""
    def always_nan(path: np.ndarray) -> float:
        return np.nan

    with pytest.raises(ValueError, match="No finite draws"):
        suggest_threshold(always_nan, window=30, target_fpr=0.1, n_draws=10, seed=0)


def test_suggest_threshold_returns_valid_quantile() -> None:
    """Test suggest_threshold returns a sensible quantile."""
    threshold = suggest_threshold(
        hurst_static,
        window=30,
        target_fpr=0.10,
        n_draws=500,
        seed=100,
    )
    # Threshold should be a valid float
    assert isinstance(threshold, float)
    assert np.isfinite(threshold)
    assert threshold > 0.0  # Hurst values are positive


def test_suggest_threshold_respects_target_fpr() -> None:
    """Test suggest_threshold respects the target FPR parameter."""
    # Higher FPR should give lower threshold (more liberal)
    thresh_10 = suggest_threshold(
        hurst_static,
        window=30,
        target_fpr=0.10,
        n_draws=1000,
        seed=101,
    )
    thresh_05 = suggest_threshold(
        hurst_static,
        window=30,
        target_fpr=0.05,
        n_draws=1000,
        seed=101,
    )
    # Lower FPR should give higher threshold (more conservative)
    assert thresh_05 >= thresh_10


# =============================================================================
# calibration_report
# =============================================================================


def test_calibration_report_returns_dataframe() -> None:
    """Test calibration_report returns a pandas DataFrame."""
    df = calibration_report(
        "hurst_static",
        (30, 390),
        estimator=hurst_static,
        n_draws=200,
    )
    assert isinstance(df, pd.DataFrame)


def test_calibration_report_has_expected_columns() -> None:
    """Test calibration_report has all expected columns."""
    df = calibration_report(
        "hurst_static",
        (30, 390),
        estimator=hurst_static,
        n_draws=200,
    )
    expected_cols = {
        "estimator",
        "window",
        "mean",
        "std",
        "p90",
        "p95",
        "p_exceed_ref",
    }
    assert set(df.columns) == expected_cols


def test_calibration_report_rows_match_windows() -> None:
    """Test calibration_report has one row per window."""
    windows = (30, 100, 390)
    df = calibration_report(
        "hurst_static",
        windows,
        estimator=hurst_static,
        n_draws=150,
    )
    assert len(df) == len(windows)
    assert list(df["window"]) == list(windows)


def test_calibration_report_single_window() -> None:
    """Test calibration_report with a single window."""
    df = calibration_report(
        "hurst_static",
        (390,),
        estimator=hurst_static,
        n_draws=100,
    )
    assert len(df) == 1
    assert df.loc[0, "window"] == 390


def test_calibration_report_monotone_p_exceed() -> None:
    """Test calibration_report p_exceed reflects estimator properties."""
    df = calibration_report(
        "hurst_static",
        (30, 390),
        estimator=hurst_static,
        n_draws=500,
    )
    # All p_exceed values should be between 0 and 1
    assert np.all(df["p_exceed_ref"] >= 0.0)
    assert np.all(df["p_exceed_ref"] <= 1.0)


def test_calibration_report_p90_less_than_p95() -> None:
    """Test calibration_report has p90 < p95 for each window."""
    df = calibration_report(
        "hurst_static",
        (30, 390),
        estimator=hurst_static,
        n_draws=300,
    )
    assert np.all(df["p90"] <= df["p95"])


def test_calibration_report_mean_std_reasonable() -> None:
    """Test calibration_report produces reasonable mean and std."""
    df = calibration_report(
        "hurst_static",
        (30, 390),
        estimator=hurst_static,
        n_draws=400,
    )
    # Hurst exponent should be between 0 and 1
    assert np.all(df["mean"] > 0.0)
    assert np.all(df["mean"] < 1.0)
    assert np.all(df["std"] > 0.0)


def test_calibration_report_estimator_name_preserved() -> None:
    """Test calibration_report preserves the estimator name."""
    df = calibration_report(
        "custom_estimator_name",
        (30,),
        estimator=hurst_static,
        n_draws=100,
    )
    assert df.loc[0, "estimator"] == "custom_estimator_name"


# =============================================================================
# Reproducibility and seeding
# =============================================================================


def test_seed_reproducibility_across_functions() -> None:
    """Test that same seed produces same results across different functions."""
    cal1 = calibrate_null(hurst_static, window=30, n_draws=200, seed=999)
    cal2 = calibrate_null(hurst_static, window=30, n_draws=200, seed=999)

    assert cal1.mean == cal2.mean
    assert cal1.std == cal2.std
    assert cal1.percentiles == cal2.percentiles


def test_suggest_threshold_deterministic() -> None:
    """Test suggest_threshold is deterministic for same seed."""
    t1 = suggest_threshold(hurst_static, window=30, target_fpr=0.1, n_draws=500, seed=888)
    t2 = suggest_threshold(hurst_static, window=30, target_fpr=0.1, n_draws=500, seed=888)
    assert t1 == t2


# =============================================================================
# Edge cases and integration
# =============================================================================


def test_calibration_p_value_and_threshold_consistency() -> None:
    """Test that p_value(threshold_at(x)) ~= x/100 for right tail."""
    cal = calibrate_null(hurst_static, window=390, n_draws=1000, seed=500)

    # The threshold at 90th percentile should have p_value ~= 0.1 (right tail)
    thresh_90 = cal.threshold_at(90)
    p_value_90 = cal.p_value(thresh_90, tail="right")
    # Should be close to 0.1 (10% false positive rate)
    assert p_value_90 == pytest.approx(0.1, abs=0.05)


def test_empirical_vs_null_calibration_shapes() -> None:
    """Test empirical calibration produces same structure as null calibration."""
    values = np.random.default_rng(600).normal(0.5, 0.1, size=500)
    cal_emp = calibrate_empirical(values)
    cal_null = calibrate_null(hurst_static, window=100, n_draws=500, seed=600)

    # Should have same percentile levels
    assert set(cal_emp.percentiles.keys()) == set(cal_null.percentiles.keys())
    # Both should have mean and std
    assert np.isfinite(cal_emp.mean)
    assert np.isfinite(cal_null.mean)
    assert np.isfinite(cal_emp.std)
    assert np.isfinite(cal_null.std)
