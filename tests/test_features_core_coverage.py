from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features.core import (
    _apply_by_session,
    breakout_down,
    breakout_up,
    cross_sectional_rank,
    cross_sectional_zscore,
    deseasonalize_by_time_of_day,
    log_returns,
    parkinson_volatility,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    time_of_day_mean,
    time_of_day_mean_expanding,
    volume_zscore,
)


def test_rolling_mean_rejects_1d_input() -> None:
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="must be a 2-D array"):
        rolling_mean(x, window=2)


def test_rolling_mean_rejects_zero_window() -> None:
    x = np.zeros((2, 1))
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rolling_mean(x, window=0)


def test_rolling_mean_rejects_negative_min_count() -> None:
    x = np.zeros((2, 1))
    with pytest.raises(ValueError, match="min_count must be non-negative"):
        rolling_mean(x, window=2, min_count=-1)


def test_apply_by_session_requires_array() -> None:
    with pytest.raises(ValueError, match="_apply_by_session requires at least one array"):
        _apply_by_session(lambda a: a)


def test_apply_by_session_rejects_2d_day_offsets() -> None:
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match="day_offsets must be 1-D"):
        _apply_by_session(lambda a: a, x, day_offsets=np.array([[0, 3]]))


def test_apply_by_session_rejects_short_day_offsets() -> None:
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match="day_offsets must have at least 2 entries"):
        _apply_by_session(lambda a: a, x, day_offsets=np.array([0]))


def test_apply_by_session_rejects_offsets_not_start_end() -> None:
    x = np.zeros((5, 1))
    with pytest.raises(ValueError, match="day_offsets must start at 0 and end at n_rows"):
        _apply_by_session(lambda a: a, x, day_offsets=np.array([1, 5]))


def test_apply_by_session_empty_first_array_returns_empty() -> None:
    x = np.empty((0, 2))
    result = _apply_by_session(lambda a: a, x, day_offsets=np.array([0, 0]))
    assert result.shape == (0, 2)
    assert result.dtype == np.float64


def test_rolling_max_empty_rows() -> None:
    x = np.empty((0, 2))
    result = rolling_max(x, window=3)
    assert result.shape == (0, 2)
    assert result.dtype == np.float64


def test_rolling_max_min_window_one_handcomputed() -> None:
    x = np.array([[2.0, 4.0], [1.0, 3.0]])
    # window=1: the rolling max/min of a single bar is that bar itself.
    expected = np.array([[2.0, 4.0], [1.0, 3.0]])
    np.testing.assert_allclose(rolling_max(x, window=1), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rolling_min(x, window=1), expected, rtol=1e-9, atol=1e-9)


def test_log_returns_single_row_returns_all_nan() -> None:
    close = np.array([[10.0, 20.0]])
    result = log_returns(close)
    assert result.shape == close.shape
    assert np.isnan(result).all()


def test_breakout_up_shift_one_bar_one_row_session() -> None:
    close = np.array([[5.0], [2.0], [4.0], [3.0], [1.0]])
    high = np.array([[10.0], [3.0], [3.0], [4.0], [2.0]])
    day_offsets = np.array([0, 1, 5])

    # window=1 rolling max of high is high itself; _shift_one_bar yields NaN
    # for the first row of each session. Later rows compare to the prior row's high.
    expected = np.array([[False], [False], [True], [False], [False]])

    result = breakout_up(close, high, window=1, day_offsets=day_offsets)
    np.testing.assert_array_equal(result, expected)


def test_rolling_std_rejects_negative_ddof() -> None:
    x = np.zeros((2, 1))
    with pytest.raises(ValueError, match="ddof must be non-negative"):
        rolling_std(x, window=2, ddof=-1)


def test_parkinson_volatility_rejects_shape_mismatch() -> None:
    high = np.ones((2, 2))
    low = np.ones((2, 1))
    with pytest.raises(ValueError, match="high and low must have identical shapes"):
        parkinson_volatility(high, low, window=2)


def test_cross_sectional_zscore_robust_handcomputed() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
    # median = 3.0; median absolute deviation = 1.0; scale = 1.4826 * 1.0
    expected = [
        (1.0 - 3.0) / 1.4826,
        (2.0 - 3.0) / 1.4826,
        (3.0 - 3.0) / 1.4826,
        (4.0 - 3.0) / 1.4826,
        (100.0 - 3.0) / 1.4826,
    ]
    result = cross_sectional_zscore(x, robust=True)
    np.testing.assert_allclose(result[0], expected, rtol=1e-9, atol=1e-9)


def test_time_of_day_mean_rejects_wrong_minute_length() -> None:
    x = np.zeros((3, 1))
    minute_of_day = np.array([0, 1])
    with pytest.raises(ValueError, match="minute_of_day must be 1-D with length equal to n_rows"):
        time_of_day_mean(x, minute_of_day)


def test_time_of_day_mean_expanding_rejects_wrong_minute_length() -> None:
    x = np.zeros((3, 1))
    minute_of_day = np.array([0, 1])
    with pytest.raises(ValueError, match="minute_of_day must be 1-D with length equal to n_rows"):
        time_of_day_mean_expanding(x, minute_of_day, day_offsets=np.array([0, 3]))


def test_deseasonalize_by_time_of_day_rejects_invalid_mode() -> None:
    x = np.zeros((3, 1))
    minute_of_day = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="mode must be 'expanding' or 'in_sample'"):
        deseasonalize_by_time_of_day(x, minute_of_day, mode="bogus")


def test_volume_zscore_without_day_offsets_handcomputed() -> None:
    volume = np.array([[1.0], [2.0]])
    minute_of_day = np.array([0, 0])
    # window=2: only the second row has a full window.
    # mean = (1 + 2) / 2 = 1.5; sample std = sqrt(((1-1.5)^2 + (2-1.5)^2) / 1)
    # = sqrt(0.5)
    result = volume_zscore(volume, minute_of_day, window=2, deseasonalize=False)
    expected = np.array([[np.nan], [(2.0 - 1.5) / np.sqrt(0.5)]])
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_breakout_up_rejects_shape_mismatch() -> None:
    close = np.ones((2, 2))
    high = np.ones((2, 1))
    with pytest.raises(ValueError, match="close and high must have identical shapes"):
        breakout_up(close, high, window=2)


def test_breakout_down_rejects_shape_mismatch() -> None:
    close = np.ones((2, 2))
    low = np.ones((2, 1))
    with pytest.raises(ValueError, match="close and low must have identical shapes"):
        breakout_down(close, low, window=2)


def test_time_of_day_mean_expanding_causal_handcomputed() -> None:
    x = np.array([[10.0], [20.0], [30.0], [40.0]])
    minute_of_day = np.array([0, 1, 0, 1])
    day_offsets = np.array([0, 2, 4])

    # Session 0 has no prior sessions -> NaN.
    # Session 1 prior-session means: minute 0 = 10.0, minute 1 = 20.0.
    expected = np.array([[np.nan], [np.nan], [10.0], [20.0]])

    result = time_of_day_mean_expanding(x, minute_of_day, day_offsets)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_deseasonalize_in_sample_emits_lookahead_warning() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    minute_of_day = np.array([0, 0, 0])

    with pytest.warns(UserWarning, match="lookahead"):
        result = deseasonalize_by_time_of_day(x, minute_of_day, mode="in_sample")

    # Whole-array in-sample mean for the single minute bucket is 2.0.
    expected = np.array([[0.5], [1.0], [1.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_parkinson_volatility_high_low_equal_handcomputed() -> None:
    high = np.array([[100.0], [100.0], [100.0]])
    low = np.array([[100.0], [100.0], [100.0]])

    # high/low = 1, log(1) = 0, so rolling mean of squared log ranges is 0.
    # Result = sqrt(0 / (4 * ln(2))) = 0.0.
    result = parkinson_volatility(high, low, window=3, min_count=1)
    expected = np.zeros((3, 1))

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_rolling_max_min_all_nan_window_returns_nan() -> None:
    x = np.full((3, 1), np.nan)

    result_max = rolling_max(x, window=2)
    result_min = rolling_min(x, window=2)

    assert np.isnan(result_max).all()
    assert np.isnan(result_min).all()


def test_cross_sectional_rank_exact_min_names_handcomputed() -> None:
    x = np.array([[10.0, 30.0, 20.0, 50.0, 40.0]])
    # Sorted ranks: [1, 3, 2, 5, 4]; pct = (rank - 1) / (5 - 1)
    expected = np.array([[0.0, 0.5, 0.25, 1.0, 0.75]])

    result = cross_sectional_rank(x)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_cross_sectional_rank_below_min_names_row_all_nan() -> None:
    x = np.array([[10.0, 30.0, 20.0, 50.0, np.nan]])
    result = cross_sectional_rank(x)
    assert np.isnan(result).all()
