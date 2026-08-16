"""Tests for core feature computations."""


import math
import time

import numpy as np
import pytest

from nifty_quant.features import core


def _pure_rolling_mean(x: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Pure Python, per-column rolling mean reference."""
    n_rows, n_cols = x.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for col in range(n_cols):
        for row in range(n_rows):
            start = max(0, row - window + 1)
            total = 0.0
            count = 0
            for i in range(start, row + 1):
                value = float(x[i, col])
                if not math.isnan(value):
                    total += value
                    count += 1
            if count >= min_count and count > 0:
                out[row, col] = total / count

    return out


def _pure_rolling_std(
    x: np.ndarray, window: int, min_count: int, ddof: int = 1
) -> np.ndarray:
    """Pure Python, per-column rolling standard deviation reference."""
    n_rows, n_cols = x.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for col in range(n_cols):
        for row in range(n_rows):
            start = max(0, row - window + 1)
            total = 0.0
            total_sq = 0.0
            count = 0
            for i in range(start, row + 1):
                value = float(x[i, col])
                if not math.isnan(value):
                    total += value
                    total_sq += value * value
                    count += 1
            if count >= min_count and count > ddof:
                variance = max(0.0, total_sq - total * total / count) / (count - ddof)
                out[row, col] = math.sqrt(variance)

    return out


def _pure_rolling_extreme(
    x: np.ndarray, window: int, min_count: int, use_max: bool
) -> np.ndarray:
    """Pure Python, per-column rolling max/min reference."""
    n_rows, n_cols = x.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for col in range(n_cols):
        for row in range(n_rows):
            start = max(0, row - window + 1)
            count = 0
            best = None
            for i in range(start, row + 1):
                value = float(x[i, col])
                if not math.isnan(value):
                    count += 1
                    if best is None:
                        best = value
                    elif use_max:
                        best = max(best, value)
                    else:
                        best = min(best, value)
            if count >= min_count and count > 0:
                out[row, col] = best

    return out


# 1. Reference-implementation equivalence
def test_reference_equivalence_rolling_functions():
    rng = np.random.default_rng(12345)
    base = rng.standard_normal((500, 4)).astype(np.float32)

    x = base.copy()
    mask_x = rng.random((500, 4)) < 0.05
    x[mask_x] = np.nan

    noise_high = np.abs(rng.standard_normal((500, 4)).astype(np.float32))
    noise_low = np.abs(rng.standard_normal((500, 4)).astype(np.float32))
    high = (base + noise_high).astype(np.float32)
    low = (base - noise_low).astype(np.float32)

    mask_high = rng.random((500, 4)) < 0.05
    mask_low = rng.random((500, 4)) < 0.05
    high[mask_high] = np.nan
    low[mask_low] = np.nan

    window = 10
    mean_expected = _pure_rolling_mean(x, window, window)
    std_expected = _pure_rolling_std(x, window, window, ddof=1)
    max_expected = _pure_rolling_extreme(high, window, window, use_max=True)
    min_expected = _pure_rolling_extreme(low, window, window, use_max=False)

    np.testing.assert_allclose(
        core.rolling_mean(x, window), mean_expected, atol=1e-9, rtol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        core.rolling_std(x, window), std_expected, atol=1e-9, rtol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        core.rolling_max(high, window), max_expected, atol=1e-9, rtol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        core.rolling_min(low, window), min_expected, atol=1e-9, rtol=0, equal_nan=True
    )


# 1b. rolling_max/rolling_min must reduce over TIME, not across symbols.
# A multi-column reference where each column has a distinct, easily distinguishable
# trend makes a symbols-axis-vs-time-axis mixup obvious (columns would leak into each
# other and the result would stop matching a strictly per-column computation).
def test_rolling_extreme_reduces_over_time_not_symbols():
    n, k = 60, 5
    # column j is a monotonically increasing ramp offset by j*1000, so columns never
    # overlap in value range and any cross-column mixing is immediately visible.
    col_offsets = np.arange(k, dtype=np.float64) * 1000.0
    ramp = np.arange(n, dtype=np.float64)[:, None]
    x = (ramp + col_offsets[None, :]).astype(np.float32)

    window = 7
    rolled_max = core.rolling_max(x, window)
    rolled_min = core.rolling_min(x, window)

    for col in range(k):
        # Each column is a pure increasing ramp: rolling max at row t (t>=window-1)
        # must equal x[t, col] and rolling min must equal x[t-window+1, col].
        valid_rows = np.arange(window - 1, n)
        np.testing.assert_allclose(
            rolled_max[valid_rows, col], x[valid_rows, col], atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            rolled_min[valid_rows, col],
            x[valid_rows - window + 1, col],
            atol=1e-9,
            rtol=0,
        )
        # No column's rolling stats may ever fall outside that column's own value
        # range (which they would if the window mixed rows from other columns).
        col_lo, col_hi = x[:, col].min(), x[:, col].max()
        finite_max = rolled_max[np.isfinite(rolled_max[:, col]), col]
        finite_min = rolled_min[np.isfinite(rolled_min[:, col]), col]
        assert np.all((finite_max >= col_lo) & (finite_max <= col_hi))
        assert np.all((finite_min >= col_lo) & (finite_min <= col_hi))


# 2. Breakout regression: shifted rolling extremum is required
def test_breakout_up_regression_shift_is_required():
    n, k = 500, 4
    rng = np.random.default_rng(202)

    trend = np.arange(n, dtype=np.float32)[:, None]
    high = np.repeat(trend, k, axis=1)
    high = (high + np.abs(rng.standard_normal((n, k))).astype(np.float32)).astype(np.float32)
    close = (high - rng.uniform(0.01, 0.05, size=(n, k)).astype(np.float32)).astype(
        np.float32
    )
    close = np.minimum(close, high)
    assert np.all(close <= high)

    result = core.breakout_up(close, high, 30)
    assert result.dtype == np.bool_
    assert result.sum() > 0

    naive = close > core.rolling_max(high, 30)
    assert naive.sum() == 0

    expected = np.zeros_like(result, dtype=bool)
    for row in range(30, n):
        for col in range(k):
            prior_max = max(float(value) for value in high[row - 30 : row, col])
            expected[row, col] = float(close[row, col]) > prior_max

    np.testing.assert_array_equal(result, expected)


def test_breakout_down_regression_shift_is_required():
    n, k = 500, 4
    rng = np.random.default_rng(203)

    trend = -np.arange(n, dtype=np.float32)[:, None]
    low = np.repeat(trend, k, axis=1)
    low = (low - np.abs(rng.standard_normal((n, k))).astype(np.float32)).astype(np.float32)
    close = (low + rng.uniform(0.01, 0.05, size=(n, k)).astype(np.float32)).astype(
        np.float32
    )
    close = np.maximum(close, low)
    assert np.all(close >= low)

    result = core.breakout_down(close, low, 30)
    assert result.dtype == np.bool_
    assert result.sum() > 0

    naive = close < core.rolling_min(low, 30)
    assert naive.sum() == 0

    expected = np.zeros_like(result, dtype=bool)
    for row in range(30, n):
        for col in range(k):
            prior_min = min(float(value) for value in low[row - 30 : row, col])
            expected[row, col] = float(close[row, col]) < prior_min

    np.testing.assert_array_equal(result, expected)


# 3. Parkinson volatility
def test_parkinson_volatility_constant_ratio_and_invalid_rows():
    low = np.ones((40, 3), dtype=np.float32) * np.float32(10.0)
    r = 1.2
    high = (low * np.float32(r)).astype(np.float32)

    expected = math.sqrt((math.log(r) ** 2) / (4.0 * math.log(2.0)))
    result = core.parkinson_volatility(high, low, 5)

    assert np.all(np.isnan(result[:4]))
    np.testing.assert_allclose(result[4:], expected, atol=1e-9, rtol=0)

    bad_high = np.full((5, 3), 10.0, dtype=np.float32)
    bad_low = np.full((5, 3), 10.0, dtype=np.float32)
    bad_high[1, :] = 5.0
    bad_high[3, :] = 9.0

    bad_result = core.parkinson_volatility(bad_high, bad_low, 1)

    assert np.all(np.isnan(bad_result[1]))
    assert np.all(np.isnan(bad_result[3]))
    assert np.all(np.isfinite(bad_result[[0, 2, 4]]))


# 4. Cross-sectional z-score
def test_cross_sectional_zscore_statistics_and_invariants():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 8))
    mask = rng.random((50, 8)) < 0.10
    x[mask] = np.nan

    result = core.cross_sectional_zscore(x, min_names=5)
    finite_mask = np.isfinite(x)
    finite_count = finite_mask.sum(axis=1)

    for row in range(x.shape[0]):
        if finite_count[row] < 5:
            assert np.all(np.isnan(result[row]))
            continue

        row_finite = finite_mask[row]
        row_result = result[row, row_finite]

        assert np.all(np.isfinite(row_result))
        assert np.all(np.isnan(result[row, ~row_finite]))
        np.testing.assert_allclose(np.nanmean(row_result), 0.0, atol=1e-8)
        np.testing.assert_allclose(np.nanstd(row_result, ddof=0), 1.0, atol=1e-8)

    special = np.full((2, 8), np.nan)
    special[0, :3] = [1.0, 2.0, 3.0]
    special[1, :] = 5.0
    special_result = core.cross_sectional_zscore(special, min_names=5)
    assert np.all(np.isnan(special_result))

    a, b = 2.5, -3.0
    affine_result = core.cross_sectional_zscore(a * x + b, min_names=5)
    np.testing.assert_allclose(result, affine_result, atol=1e-9, rtol=0, equal_nan=True)


# 4b. cross_sectional_rank: NaNs must be excluded from ranking on a per-row basis,
# not "propagate" to invalidate every finite entry in a row that merely contains
# some NaN elsewhere. A row's finite values must still get real, distinguishable
# ranks even when that same row also has NaN entries.
def test_cross_sectional_rank_excludes_nans_without_propagating():
    x = np.array(
        [
            [30.0, np.nan, 10.0, 20.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    result = core.cross_sectional_rank(x, pct=False, min_names=3)

    # Row 0 has one NaN and five finite values; ranks among the finite values must
    # be assigned normally (1..5 low-to-high), NOT all turned into NaN because one
    # entry in the row happens to be NaN.
    assert np.isnan(result[0, 1])
    expected_row0 = {2: 1.0, 3: 2.0, 0: 3.0, 4: 4.0, 5: 5.0}
    for col, rank in expected_row0.items():
        assert result[0, col] == rank

    # Row 1 has no NaNs at all; ranks must be the plain 1..6 ordering.
    np.testing.assert_array_equal(result[1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # pct=True variant: finite entries must still be spread across [0, 1], not all
    # collapsed to NaN just because the row contains a NaN.
    pct_result = core.cross_sectional_rank(x, pct=True, min_names=3)
    assert np.isnan(pct_result[0, 1])
    finite_pct = pct_result[0, np.isfinite(x[0])]
    assert np.all(np.isfinite(finite_pct))
    np.testing.assert_allclose(finite_pct.min(), 0.0, atol=1e-9)
    np.testing.assert_allclose(finite_pct.max(), 1.0, atol=1e-9)


# 5. NaN discipline sweep
def test_nan_discipline_no_inf_across_module():
    n, k = 30, 4
    rng = np.random.default_rng(99)

    x = rng.standard_normal((n, k)).astype(np.float32)
    mask = rng.random((n, k)) < 0.05
    x[mask] = np.nan
    x[0, 0] = 0.0
    x[1, 1] = -5.0

    close = rng.standard_normal((n, k)).astype(np.float32)
    close[2, 2] = 0.0
    close[3, 3] = -1.0

    noise = np.abs(rng.standard_normal((n, k)).astype(np.float32))
    high = (x + noise + 5.0).astype(np.float32)
    low = (x - noise - 5.0).astype(np.float32)

    day_offsets = np.array([0, 10, 20, 30])
    minute_of_day = np.arange(n) % 5

    float_outputs = [
        core.log_returns(close),
        core.rolling_mean(x, 5),
        core.rolling_std(x, 5),
        core.rolling_max(x, 5),
        core.rolling_min(x, 5),
        core.rolling_zscore(x, 5),
        core.parkinson_volatility(high, low, 5),
        core.cross_sectional_zscore(x, min_names=2),
        core.cross_sectional_rank(x, pct=True, min_names=2),
        core.time_of_day_mean(x, minute_of_day),
        core.deseasonalize_by_time_of_day(x, minute_of_day),
        core.volume_zscore(x, minute_of_day, 5, day_offsets=day_offsets),
    ]

    for result in float_outputs:
        assert np.all(np.isfinite(result) | np.isnan(result))

    assert core.breakout_up(close, high, 5).dtype == np.bool_
    assert core.breakout_down(close, low, 5).dtype == np.bool_


# 6. Session boundaries
def test_session_boundaries_are_tight():
    n = 1125
    rng = np.random.default_rng(7)

    x = np.empty((n, 4), dtype=np.float32)
    x[:375] = 100.0
    x[375:750] = rng.standard_normal((375, 4)).astype(np.float32)
    x[750:] = 50.0

    day_offsets = np.array([0, 375, 750, 1125])
    window = 20

    mean = core.rolling_mean(x, window, day_offsets=day_offsets)
    std = core.rolling_std(x, window, day_offsets=day_offsets)
    mx = core.rolling_max(x, window, day_offsets=day_offsets)
    mn = core.rolling_min(x, window, day_offsets=day_offsets)

    for start in (0, 375, 750):
        warmup = slice(start, start + window - 1)
        assert np.all(np.isnan(mean[warmup]))
        assert np.all(np.isnan(std[warmup]))
        assert np.all(np.isnan(mx[warmup]))
        assert np.all(np.isnan(mn[warmup]))

    assert np.all(np.isnan(std[750:769]))
    np.testing.assert_allclose(std[769:], 0.0, atol=1e-15, rtol=0)

    assert np.all(np.isnan(mean[375]))
    np.testing.assert_allclose(mean[374], 100.0, atol=1e-9, rtol=0)


# 7. Dtype policy
def test_dtype_policy_float32_in_float64_out():
    n, k = 10, 3
    rng = np.random.default_rng(8)

    x = rng.standard_normal((n, k)).astype(np.float32)
    close = (np.abs(x) + np.float32(1.0)).astype(np.float32)
    high = (close + np.abs(rng.standard_normal((n, k)).astype(np.float32))).astype(
        np.float32
    )
    low = (close - np.abs(rng.standard_normal((n, k)).astype(np.float32))).astype(
        np.float32
    )

    minute_of_day = np.arange(n, dtype=int) % 3
    day_offsets = np.array([0, 10])

    float_outputs = [
        core.log_returns(close),
        core.rolling_mean(x, 3),
        core.rolling_std(x, 3),
        core.rolling_max(x, 3),
        core.rolling_min(x, 3),
        core.rolling_zscore(x, 3),
        core.parkinson_volatility(high, low, 3),
        core.cross_sectional_zscore(x, min_names=2),
        core.cross_sectional_rank(x, pct=True, min_names=2),
        core.time_of_day_mean(x, minute_of_day),
        core.deseasonalize_by_time_of_day(x, minute_of_day),
        core.volume_zscore(x, minute_of_day, 3, day_offsets=day_offsets),
    ]

    for result in float_outputs:
        assert result.dtype == np.float64

    assert core.breakout_up(close, high, 3).dtype == np.bool_
    assert core.breakout_down(close, low, 3).dtype == np.bool_


# 8. Performance smoke
def test_performance_smoke_rolling_zscore_large():
    x = np.random.default_rng(0).standard_normal((100_000, 50)).astype(np.float32)

    start = time.perf_counter()
    result = core.rolling_zscore(x, 20)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0
    assert result.shape == x.shape
    assert result.dtype == np.float64


# 9. Causal probe: future rows must not change earlier rows for the time-of-day
# deseasonalizer or for volume_zscore. This is the same manual perturbation that
# guards.causal(Strictness.FULL) performs, reproduced here as a regression check.
def test_causal_probe_time_of_day_and_volume_zscore():
    n_rows, n_cols = 600, 3
    rng = np.random.default_rng(1234)

    volume = rng.uniform(10.0, 1000.0, size=(n_rows, n_cols))
    minute_of_day = np.tile(np.arange(100), 6)
    day_offsets = np.arange(0, n_rows + 1, 100)

    cut = 350
    base = core.deseasonalize_by_time_of_day(volume, minute_of_day, day_offsets)

    perturbed_volume = volume.copy()
    perturbed_volume[cut + 1 :] = rng.uniform(
        10.0, 1000.0, size=(n_rows - cut - 1, n_cols)
    )
    perturbed = core.deseasonalize_by_time_of_day(
        perturbed_volume, minute_of_day, day_offsets
    )

    assert np.array_equal(base[: cut + 1], perturbed[: cut + 1], equal_nan=True)

    window = 10
    base_z = core.volume_zscore(volume, minute_of_day, window, day_offsets=day_offsets)
    perturbed_z = core.volume_zscore(
        perturbed_volume, minute_of_day, window, day_offsets=day_offsets
    )

    assert np.array_equal(base_z[: cut + 1], perturbed_z[: cut + 1], equal_nan=True)


# 10. The first session cannot see a prior session, so the causal expanding
# mean must be all-NaN there under the default min_sessions=1 setting.
def test_time_of_day_mean_expanding_first_session_all_nan():
    n_rows, n_cols = 600, 3
    rng = np.random.default_rng(56)

    volume = rng.uniform(10.0, 1000.0, size=(n_rows, n_cols))
    minute_of_day = np.tile(np.arange(100), 6)
    day_offsets = np.arange(0, n_rows + 1, 100)

    result = core.time_of_day_mean_expanding(
        volume, minute_of_day, day_offsets, min_sessions=1
    )

    first_session_end = day_offsets[1]
    assert np.all(np.isnan(result[:first_session_end]))


# 11. The old in-sample normaliser is still available for research, but it must
# loudly warn because it has full lookahead bias.
def test_deseasonalize_in_sample_mode_warns():
    n_rows, n_cols = 600, 3
    rng = np.random.default_rng(78)

    volume = rng.uniform(10.0, 1000.0, size=(n_rows, n_cols))
    minute_of_day = np.tile(np.arange(100), 6)
    day_offsets = np.arange(0, n_rows + 1, 100)

    with pytest.warns(UserWarning):
        core.deseasonalize_by_time_of_day(
            volume, minute_of_day, day_offsets, mode="in_sample"
        )


# 12. Expanding mean must match a hand-built three-session fixture exactly.
def test_time_of_day_mean_expanding_hand_built_fixture():
    day_offsets = np.array([0, 2, 4, 6])
    minute_of_day = np.array([0, 1, 0, 1, 0, 1])
    x = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])

    result = core.time_of_day_mean_expanding(
        x, minute_of_day, day_offsets, min_sessions=1
    )

    expected = np.array(
        [
            [np.nan],
            [np.nan],
            [10.0],
            [20.0],
            [20.0],
            [30.0],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=0, rtol=0, equal_nan=True)
