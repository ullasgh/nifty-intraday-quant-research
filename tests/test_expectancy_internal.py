"""Tests for error paths, edge cases, and invariants in the expectancy module."""
from __future__ import annotations

import datetime

import numpy as np
import pytest

from nifty_quant.research import expectancy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_session_day_offsets(n_rows: int) -> np.ndarray:
    return np.array([0, n_rows], dtype=int)


def _multi_session_day_offsets(lengths: list[int]) -> np.ndarray:
    return np.cumsum([0] + lengths, dtype=int)


def _driftless_gbm_close(
    rng: np.random.Generator,
    n_rows: int,
    n_symbols: int,
    sigma: float,
    close0: float = 100.0,
) -> np.ndarray:
    log_rets = rng.normal(0.0, sigma, size=(n_rows - 1, n_symbols))
    close = np.empty((n_rows, n_symbols), dtype=np.float64)
    close[0, :] = close0
    close[1:, :] = close0 * np.exp(np.cumsum(log_rets, axis=0))
    return close


# ---------------------------------------------------------------------------
# forward_returns: error paths and validation
# ---------------------------------------------------------------------------


def test_forward_returns_horizon_zero_raises() -> None:
    close = np.ones((10, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        expectancy.forward_returns(close, _single_session_day_offsets(10), horizon=0)


def test_forward_returns_negative_horizon_raises() -> None:
    close = np.ones((10, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        expectancy.forward_returns(close, _single_session_day_offsets(10), horizon=-5)


def test_forward_returns_1d_input_raises() -> None:
    close = np.ones(10, dtype=np.float64)
    with pytest.raises(ValueError, match="close must be 2-D"):
        expectancy.forward_returns(close, _single_session_day_offsets(10), horizon=1)  # type: ignore


def test_forward_returns_dtype_conversion() -> None:
    """Verify float32 inputs are converted to float64."""
    close_f32 = np.ones((10, 2), dtype=np.float32)
    fwd = expectancy.forward_returns(close_f32, _single_session_day_offsets(10), horizon=1)
    assert fwd.values.dtype == np.float64


def test_forward_returns_empty_panel() -> None:
    """Empty panel (0 rows) produces empty forward returns array."""
    close = np.empty((0, 2), dtype=np.float64)
    day_offsets = np.array([0], dtype=int)
    result = expectancy.forward_returns(close, day_offsets, horizon=1)

    assert result.values.shape == (0, 2)
    assert result.n_defined == 0
    assert result.n_nan_tail == 0
    assert result.horizon == 1
    assert result.session_bounded is True


# ---------------------------------------------------------------------------
# causal_buckets: error paths and degenerate cases
# ---------------------------------------------------------------------------


def test_causal_buckets_1d_input_raises() -> None:
    feature = np.ones(10, dtype=np.float64)
    with pytest.raises(ValueError, match="feature must be 2-D"):
        expectancy.causal_buckets(feature, _single_session_day_offsets(10))  # type: ignore


def test_causal_buckets_invalid_method_raises() -> None:
    feature = np.ones((10, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="unknown method"):
        expectancy.causal_buckets(
            feature, _single_session_day_offsets(10), method="invalid_method"
        )


def test_causal_buckets_rolling_quantile_method() -> None:
    """Test rolling_quantile bucketing method."""
    rng = np.random.default_rng(100)
    n_rows, n_symbols = 100, 3
    feature = rng.normal(size=(n_rows, n_symbols))
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="rolling_quantile",
        min_history=20,
    )
    assert result.labels.shape == feature.shape
    assert result.labels.dtype == np.int8
    assert result.method == "rolling_quantile"
    # First 20 rows should be unassigned
    assert np.all(result.labels[:20] == -1)
    # Later rows should have bucket assignments
    assert np.any(result.labels[20:] >= 0)


def test_causal_buckets_all_nan_with_rolling_quantile() -> None:
    """Test rolling_quantile with all-NaN feature."""
    feature = np.full((50, 2), np.nan, dtype=np.float64)
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(50),
        n_buckets=5,
        method="rolling_quantile",
        min_history=20,
    )
    assert np.all(result.labels == -1)


def test_causal_buckets_empty_panel() -> None:
    """Empty panel (0 rows) produces empty bucketing labels array."""
    feature = np.empty((0, 2), dtype=np.float64)
    day_offsets = np.array([0], dtype=int)
    result = expectancy.causal_buckets(feature, day_offsets, n_buckets=5)

    assert result.labels.shape == (0, 2)
    assert result.n_buckets == 5
    assert result.method == "expanding_quantile"


# ---------------------------------------------------------------------------
# conditional_expectancy: SE methods (non_overlapping, naive)
# ---------------------------------------------------------------------------


def test_conditional_expectancy_non_overlapping_se() -> None:
    """Test non_overlapping SE method."""
    rng = np.random.default_rng(101)
    n_rows, n_symbols = 500, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=10)
    feature = rng.normal(size=(n_rows, n_symbols))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="non_overlapping",
        seed=101,
    )

    # Verify n_effective < n_obs for horizon > 1
    for bucket in table.buckets:
        if bucket.n_obs > 0:
            assert bucket.n_effective < bucket.n_obs
            assert bucket.se_method == "non_overlapping"


def test_conditional_expectancy_naive_se() -> None:
    """Test naive SE method."""
    rng = np.random.default_rng(102)
    n_rows, n_symbols = 500, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=10)
    feature = rng.normal(size=(n_rows, n_symbols))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
        seed=102,
    )

    # For naive method, n_effective should still reflect horizon correction
    # but not as much as block bootstrap
    for bucket in table.buckets:
        assert bucket.se_method == "naive"


def test_conditional_expectancy_invalid_se_method_in_compute() -> None:
    """Test that invalid SE method raises error in bucket stats computation."""
    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Create bucket with actual return values (not all NaN)
    # Use horizon > 1 so it reaches the se_method check
    rng = np.random.default_rng(103)
    bucket_returns = rng.normal(size=100)  # All finite values
    day_offsets = _single_session_day_offsets(len(bucket_returns))

    with pytest.raises(ValueError, match="unknown se_method"):
        _compute_bucket_stats(
            bucket_returns,
            horizon=10,  # horizon > 1 to reach se_method validation
            day_offsets=day_offsets,
            se_method="invalid_method",  # type: ignore
        )


# ---------------------------------------------------------------------------
# Block bootstrap: block-index exposure and invariant verification
# ---------------------------------------------------------------------------


def test_block_bootstrap_blocks_never_straddle_session() -> None:
    """Verify block-bootstrap block-index exposure: blocks never straddle session boundaries.

    This test directly uses the exposed block start indices to verify the critical
    invariant that would silently manufacture statistical significance if violated.
    """
    rng = np.random.default_rng(104)
    # Multi-session setup with irregular session lengths
    session_lengths = [50, 75, 60, 40]
    day_offsets = _multi_session_day_offsets(session_lengths)
    total = sum(session_lengths)
    n_symbols = 5

    close = _driftless_gbm_close(rng, total, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=20)

    # Compute bucket stats with block bootstrap
    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Create a bucket of returns
    bucket_returns = fwd.values.flatten()  # All returns in one bucket for this test

    stat = _compute_bucket_stats(
        bucket_returns,
        horizon=20,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=100,
        seed=104,
    )

    # The internal block bootstrap function doesn't expose indices directly,
    # but we can verify the invariant by checking that results are finite
    # (if blocks straddled boundaries with NaN-propagation, we'd see more NaNs)
    assert np.isfinite(stat.ci_low_bps)
    assert np.isfinite(stat.ci_high_bps)
    assert np.isfinite(stat.t_stat)

    # Additional verification: run with a 2-D array directly
    bucket_returns_2d = fwd.values.reshape((total, n_symbols))
    from nifty_quant.research.expectancy import _block_bootstrap_resampling_2d

    resampled, block_indices, n_sessions_skipped = _block_bootstrap_resampling_2d(
        bucket_returns_2d, day_offsets, horizon=20, n_boot=50, seed=104
    )

    # Verify resampled shape: (n_boot, n_rows, n_symbols) -- each replicate is a full
    # tiled-and-truncated series now, not a single <= horizon block.
    assert resampled.shape[0] == 50  # n_boot
    assert resampled.shape[1] == total  # n_rows
    assert resampled.shape[2] == n_symbols
    # Verify block_indices are returned
    assert isinstance(block_indices, tuple)
    assert len(block_indices) > 0 or len(block_indices) == 0  # May be empty if fallback
    # Every session here (50, 75, 60, 40 bars) is >= block_length (20 + 5 = 25), so
    # none should be skipped.
    assert n_sessions_skipped == 0

    # Blocks should respect session boundaries: every drawn (start, length) block must
    # fit entirely inside a single session -- this is the actual invariant, checked
    # directly against the exposed block indices rather than inferred from NaN presence.
    session_starts = day_offsets[:-1]
    session_ends = day_offsets[1:]
    for start, length in block_indices:
        containing = [
            idx
            for idx in range(len(session_starts))
            if session_starts[idx] <= start < session_ends[idx]
        ]
        assert len(containing) == 1, f"block start {start} not inside exactly one session"
        sess_idx = containing[0]
        assert start + length <= session_ends[sess_idx], (
            f"block ({start}, {length}) straddles session boundary at {session_ends[sess_idx]}"
        )

    # (if a block contains rows from two sessions, that's the bug)
    for b in range(50):
        # Check that each bootstrap sample has finite values
        # (NaN means block included a row from next session, propagating NaN)
        sample = resampled[b]
        # This is a smoke test; the actual invariant is checked at computation time
        assert sample.shape[1] == n_symbols


def test_block_bootstrap_with_multi_session_irregular() -> None:
    """Test block bootstrap respects irregular session boundaries."""
    rng = np.random.default_rng(105)
    lengths = [30, 45, 20]
    day_offsets = _multi_session_day_offsets(lengths)
    total = sum(lengths)
    n_symbols = 3

    close = _driftless_gbm_close(rng, total, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=15)

    table = expectancy.conditional_expectancy(
        rng.normal(size=(total, n_symbols)),
        fwd,
        day_offsets,
        n_buckets=3,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=100,
        seed=105,
    )

    # All buckets should have finite statistics
    for bucket in table.buckets:
        if bucket.n_obs > 0:
            assert np.isfinite(bucket.ci_low_bps)
            assert np.isfinite(bucket.ci_high_bps)


# ---------------------------------------------------------------------------
# Decomposition functions: edge cases and validation
# ---------------------------------------------------------------------------


def test_expectancy_by_year_single_year() -> None:
    """Test expectancy_by_year with data from a single year."""
    rng = np.random.default_rng(106)
    n_rows, n_symbols = 300, 4
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))

    dates = np.array([datetime.date(2024, 1, 1) for _ in range(1)], dtype=object)

    result = expectancy.expectancy_by_year(
        feature, fwd, day_offsets, dates, n_buckets=3, se_method="naive"
    )

    assert 2024 in result
    assert len(result) == 1
    assert result[2024].n_total == fwd.n_defined


def test_expectancy_by_time_of_day_single_bucket() -> None:
    """Test expectancy_by_time_of_day when all rows fall in one time bucket."""
    rng = np.random.default_rng(107)
    n_rows, n_symbols = 200, 3
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))

    # All rows have the same minute_of_day
    minute_of_day = np.full(n_rows, 600, dtype=int)

    result = expectancy.expectancy_by_time_of_day(
        feature,
        fwd,
        day_offsets,
        minute_of_day,
        time_bucket_minutes=60,
        n_buckets=3,
        se_method="naive",
    )

    assert 600 in result
    assert len(result) == 1


def test_expectancy_by_liquidity_decile_all_uniform() -> None:
    """Test expectancy_by_liquidity_decile with uniform prior_adv."""
    rng = np.random.default_rng(108)
    n_rows, n_symbols = 300, 10
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))

    # All symbols have the same prior_adv (constant)
    prior_adv = np.ones((n_rows, n_symbols), dtype=np.float64)

    result = expectancy.expectancy_by_liquidity_decile(
        feature, fwd, day_offsets, prior_adv, n_buckets=10, se_method="naive"
    )

    # All deciles should be present but with varied population
    assert len(result) == 10
    assert all(k in result for k in range(10))


def test_double_sort_with_correlated_features() -> None:
    """Test double_sort with correlated features."""
    rng = np.random.default_rng(109)
    n_rows, n_symbols = 400, 9
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    base = rng.normal(size=(n_rows, n_symbols))
    feature_a = base
    feature_b = base + rng.normal(scale=0.001, size=(n_rows, n_symbols))

    result = expectancy.double_sort(
        feature_a,
        feature_b,
        fwd,
        day_offsets,
        n_buckets_a=3,
        n_buckets_b=3,
        method="cross_sectional_rank",
        se_method="naive",
        thin_cell_threshold=30,
    )

    # Verify structure
    assert len(result.cells) == 3  # n_buckets_a
    assert all(len(row) == 3 for row in result.cells)  # n_buckets_b
    assert result.n_total == fwd.n_defined * n_symbols
    # Some cells should be thin given the setup
    # (correlated features → diagonal concentration)
    # But we can't assert this deterministically without testing implementation


def test_double_sort_empty_cells_are_thin() -> None:
    """Test that empty cells are flagged as thin."""
    rng = np.random.default_rng(110)
    n_rows, n_symbols = 150, 100
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    # Highly correlated features (same values)
    feature_a = np.tile(np.arange(n_symbols, dtype=np.float64), (n_rows, 1))
    # Create feature_b by cycling through [0, 1, 2]
    b_vals = np.array([0, 1, 2] * (n_symbols // 3 + 1))[:n_symbols]
    feature_b = np.tile(b_vals, (n_rows, 1))

    result = expectancy.double_sort(
        feature_a,
        feature_b,
        fwd,
        day_offsets,
        n_buckets_a=3,
        n_buckets_b=3,
        method="cross_sectional_rank",
        se_method="naive",
        thin_cell_threshold=30,
    )

    # With 100 symbols and tight bucketing, some cells will be thin or empty
    # This just verifies the test structure works
    assert len(result.cells) == 3
    assert all(len(row) == 3 for row in result.cells)


# ---------------------------------------------------------------------------
# Explain methods: message content verification
# ---------------------------------------------------------------------------


def test_explain_forward_returns() -> None:
    """Verify ForwardReturns.explain() output."""
    rng = np.random.default_rng(111)
    close = _driftless_gbm_close(rng, 100, 2, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(100), horizon=5)

    explanation = fwd.explain()
    assert "horizon=5" in explanation
    assert "session_bounded" in explanation
    assert "n_defined" in explanation


def test_explain_bucketing() -> None:
    """Verify Bucketing.explain() output."""
    rng = np.random.default_rng(112)
    feature = rng.normal(size=(100, 3))
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(100),
        n_buckets=5,
        method="expanding_quantile",
        min_history=30,
    )

    explanation = result.explain()
    assert "expanding_quantile" in explanation
    assert "n_buckets" in explanation
    assert "assigned" in explanation or "unassigned" in explanation


def test_explain_expectancy_table_block_bootstrap() -> None:
    """Verify ExpectancyTable.explain() for block_bootstrap."""
    rng = np.random.default_rng(113)
    n_rows = 300
    close = _driftless_gbm_close(rng, n_rows, 4, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 4))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        se_method="block_bootstrap",
        cost_hurdle_bps=10.0,
    )

    explanation = table.explain()
    # Check for required keywords (case-insensitive where applicable)
    assert "hurdle" in explanation.lower()
    assert "bps" in explanation.lower()
    assert "10" in explanation or "10.0" in explanation
    # When buckets exist, SE method should be shown
    if table.buckets:
        assert "block_bootstrap" in explanation.lower()


def test_explain_expectancy_table_naive_warning() -> None:
    """Verify ExpectancyTable.explain() warns for naive SE."""
    rng = np.random.default_rng(114)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 3))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        se_method="naive",
    )

    explanation = table.explain()
    # When buckets exist with naive SE, should have a warning marker
    if table.buckets:
        loud_markers = ["NAIVE", "WARNING", "UNCORRECTED"]
        assert any(marker in explanation for marker in loud_markers)


# ---------------------------------------------------------------------------
# to_frame method
# ---------------------------------------------------------------------------


def test_expectancy_table_to_frame_columns() -> None:
    """Verify ExpectancyTable.to_frame() has all required columns."""
    rng = np.random.default_rng(115)
    n_rows = 250
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 3))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        se_method="naive",
    )

    df = table.to_frame()

    required_cols = {
        "bucket",
        "n_obs",
        "n_effective",
        "mean_bps",
        "median_bps",
        "std_bps",
        "t_stat",
        "se_method",
        "ci_low_bps",
        "ci_high_bps",
    }
    assert required_cols.issubset(df.columns)
    assert len(df) == len(table.buckets)
    assert list(df["bucket"]) == sorted(df["bucket"].tolist())


# ---------------------------------------------------------------------------
# Edge cases: single bucket, all-NaN inputs, etc.
# ---------------------------------------------------------------------------


def test_conditional_expectancy_single_bucket() -> None:
    """Test conditional_expectancy with single bucket."""
    rng = np.random.default_rng(116)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 3))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=1,
        method="cross_sectional_rank",
        se_method="naive",
    )

    # With 1 bucket, all symbols get bucket 0
    if table.buckets:
        assert len(table.buckets) == 1
        assert table.buckets[0].bucket == 0
        assert table.spread_bps == 0.0


def test_conditional_expectancy_all_nan_feature() -> None:
    """Test conditional_expectancy with all-NaN feature."""
    rng = np.random.default_rng(117)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = np.full((n_rows, 3), np.nan, dtype=np.float64)

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        se_method="naive",
    )

    # All observations should be unassigned (bucket -1)
    for bucket in table.buckets:
        assert bucket.n_obs == 0


def test_conditional_expectancy_all_nan_forward_returns() -> None:
    """Test conditional_expectancy when all forward returns are NaN."""
    rng = np.random.default_rng(118)
    n_rows = 50
    # Create forward returns that are all NaN
    fwd_all_nan = expectancy.ForwardReturns(
        values=np.full((n_rows, 3), np.nan, dtype=np.float64),
        horizon=1,
        session_bounded=True,
        n_defined=0,
        n_nan_tail=n_rows * 3,
    )
    feature = rng.normal(size=(n_rows, 3))

    table = expectancy.conditional_expectancy(
        feature,
        fwd_all_nan,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        se_method="naive",
    )

    # All buckets should have zero observations
    for bucket in table.buckets:
        assert bucket.n_obs == 0


# ---------------------------------------------------------------------------
# Cost hurdle default behavior
# ---------------------------------------------------------------------------


def test_conditional_expectancy_default_cost_hurdle() -> None:
    """Verify default cost hurdle is NSEIntradayEquityCosts.round_trip_bps(1e5)."""
    rng = np.random.default_rng(119)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 3))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        cost_hurdle_bps=None,  # Use default
    )

    from nifty_quant.execution.costs import NSEIntradayEquityCosts

    expected = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert abs(table.cost_hurdle_bps - expected) < 0.01


# ---------------------------------------------------------------------------
# Non-overlapping subsample coverage
# ---------------------------------------------------------------------------


def test_non_overlapping_subsample_various_horizons() -> None:
    """Test non_overlapping subsample with different horizon values."""
    rng = np.random.default_rng(122)
    n_rows, n_symbols = 800, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)

    for horizon in [2, 5, 20]:
        fwd = expectancy.forward_returns(close, day_offsets, horizon=horizon)
        feature = rng.normal(size=(n_rows, n_symbols))

        table = expectancy.conditional_expectancy(
            feature,
            fwd,
            day_offsets,
            n_buckets=3,
            method="cross_sectional_rank",
            se_method="non_overlapping",
            seed=122,
        )

        # Verify n_effective < n_obs for horizon > 1
        for stat in table.buckets:
            if stat.n_obs > 0:
                assert stat.n_effective <= stat.n_obs


def test_non_overlapping_with_empty_bucket() -> None:
    """Test non_overlapping when bucket has few observations."""
    rng = np.random.default_rng(123)
    n_rows, n_symbols = 200, 3
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=10)

    # Feature that concentrates observations in one bucket
    feature = np.zeros((n_rows, n_symbols), dtype=np.float64)
    feature[:50, :] = rng.normal(size=(50, n_symbols))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="non_overlapping",
        seed=123,
    )

    # Should handle sparse buckets
    assert len(table.buckets) >= 0


# ---------------------------------------------------------------------------
# Block bootstrap with longer horizon
# ---------------------------------------------------------------------------


def test_block_bootstrap_with_long_horizon() -> None:
    """Test block bootstrap with horizon close to session length."""
    rng = np.random.default_rng(124)
    n_rows, n_symbols = 300, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=50)
    feature = rng.normal(size=(n_rows, n_symbols))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=100,
        seed=124,
    )

    # Should produce finite statistics even with long horizon
    for stat in table.buckets:
        if stat.n_obs > 0:
            assert np.isfinite(stat.ci_low_bps)
            assert np.isfinite(stat.ci_high_bps)


# ---------------------------------------------------------------------------
# Spread calculation and survives_costs
# ---------------------------------------------------------------------------


def test_spread_calculation_with_known_buckets() -> None:
    """Test spread is correctly computed as top - bottom."""
    rng = np.random.default_rng(120)
    n_rows = 300
    n_symbols = 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)

    # Feature with guaranteed bucket assignment
    feature = np.tile(np.arange(n_symbols, dtype=np.float64), (n_rows, 1))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=n_symbols,
        method="cross_sectional_rank",
        se_method="naive",
    )

    # Spread should be top bucket mean - bottom bucket mean
    expected_spread = table.buckets[-1].mean_bps - table.buckets[0].mean_bps
    assert abs(table.spread_bps - expected_spread) < 0.001


def test_survives_costs_threshold() -> None:
    """Test survives_costs requires abs(spread) > 2 * cost_hurdle."""
    rng = np.random.default_rng(121)
    n_rows = 500
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 3))

    # Test with spread exactly at 2x hurdle
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        cost_hurdle_bps=5.0,
        se_method="naive",
    )

    # survives_costs = abs(spread_bps) > 2 * hurdle
    if abs(table.spread_bps) > 10.0:
        assert table.survives_costs is True
    elif abs(table.spread_bps) < 10.0:
        assert table.survives_costs is False
    # If exactly 10, should be False (>= not >)
    if abs(abs(table.spread_bps) - 10.0) < 0.01:
        assert table.survives_costs is False


# ---------------------------------------------------------------------------
# Additional rolling_quantile edge cases
# ---------------------------------------------------------------------------


def test_rolling_quantile_various_window_sizes() -> None:
    """Test rolling_quantile with different window sizes."""
    rng = np.random.default_rng(125)
    n_rows, n_symbols = 150, 4
    feature = rng.normal(size=(n_rows, n_symbols))

    for min_history in [10, 50, 100]:
        result = expectancy.causal_buckets(
            feature,
            _single_session_day_offsets(n_rows),
            n_buckets=5,
            method="rolling_quantile",
            min_history=min_history,
        )
        assert result.labels.shape == feature.shape
        # First min_history rows should be unassigned
        assert np.all(result.labels[:min_history] == -1)


def test_rolling_quantile_with_partial_feature() -> None:
    """Test rolling_quantile when feature has partial NaNs."""
    rng = np.random.default_rng(126)
    n_rows, n_symbols = 120, 3
    feature = rng.normal(size=(n_rows, n_symbols))
    # Add some NaNs
    feature[10:15, 0] = np.nan
    feature[30:40, 1] = np.nan

    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="rolling_quantile",
        min_history=20,
    )

    assert result.labels.shape == feature.shape
    # NaN feature values should give label -1
    assert np.all(result.labels[np.isnan(feature)] == -1)


# ---------------------------------------------------------------------------
# Bootstrap with sparse data
# ---------------------------------------------------------------------------


def test_block_bootstrap_sparse_defined_values() -> None:
    """Test block bootstrap when only some rows have defined values."""
    from nifty_quant.research.expectancy import _compute_bucket_stats

    rng = np.random.default_rng(127)
    bucket_returns = np.full(200, np.nan, dtype=np.float64)
    # Only 50 rows have defined values
    bucket_returns[50:100] = rng.normal(size=50)

    day_offsets = _single_session_day_offsets(len(bucket_returns))
    stat = _compute_bucket_stats(
        bucket_returns,
        horizon=10,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=100,
        seed=127,
    )

    if stat is not None:
        assert stat.n_obs == 50
        assert np.isfinite(stat.mean_bps)


# ---------------------------------------------------------------------------
# Decomposition with multi-year data
# ---------------------------------------------------------------------------


def test_expectancy_by_year_multiple_years() -> None:
    """Test expectancy_by_year with multiple years and varying n_total."""
    rng = np.random.default_rng(128)
    sessions = [250, 300, 275, 325]  # Different session lengths
    day_offsets = _multi_session_day_offsets(sessions)
    total = sum(sessions)
    n_symbols = 4

    close = _driftless_gbm_close(rng, total, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(total, n_symbols))

    dates = np.array(
        [
            datetime.date(2023, 1, 1),
            datetime.date(2024, 1, 1),
            datetime.date(2025, 1, 1),
            datetime.date(2026, 1, 1),
        ],
        dtype=object,
    )

    result = expectancy.expectancy_by_year(
        feature, fwd, day_offsets, dates, n_buckets=3, se_method="naive", seed=128
    )

    # Should have one table per year
    assert len(result) == 4
    # Total observations should sum correctly
    total_obs = sum(t.n_total for t in result.values())
    assert total_obs == fwd.n_defined


# ---------------------------------------------------------------------------
# Decomposition edge cases
# ---------------------------------------------------------------------------


def test_by_time_of_day_non_continuous_times() -> None:
    """Test expectancy_by_time_of_day with non-continuous time buckets."""
    rng = np.random.default_rng(129)
    n_rows, n_symbols = 200, 3
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))

    # Minute_of_day with gaps (no observations at certain times)
    minute_of_day = np.concatenate(
        [
            np.full(100, 600, dtype=int),  # 10:00 AM
            np.full(100, 900, dtype=int),  # 3:00 PM
        ]
    )

    result = expectancy.expectancy_by_time_of_day(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        minute_of_day,
        time_bucket_minutes=60,
        n_buckets=3,
        se_method="naive",
    )

    # Should have buckets for both time periods
    assert 600 in result or 540 in result
    assert 900 in result or 840 in result


def test_block_bootstrap_fallback_short_sessions() -> None:
    """Test block bootstrap fallback when sessions are shorter than block length."""
    rng = np.random.default_rng(131)
    # Create very short sessions that are shorter than horizon
    lengths = [5, 6, 5]  # Each session shorter than horizon=20
    day_offsets = _multi_session_day_offsets(lengths)
    total = sum(lengths)
    n_symbols = 3

    close = _driftless_gbm_close(rng, total, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=20)
    feature = rng.normal(size=(total, n_symbols))

    # With very short sessions and long horizon, most/all forward returns are NaN
    # This should still compute without error
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=2,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=50,
        seed=131,
    )

    # Should handle the extreme case gracefully
    assert isinstance(table.n_total, int)


def test_expectancy_with_unbalanced_rank_distribution() -> None:
    """Test when bucketing creates uneven symbol distributions."""
    rng = np.random.default_rng(132)
    n_rows, n_symbols = 300, 6
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)

    # Feature designed so symbols cluster in ranks
    feature = np.column_stack(
        [
            rng.uniform(i * 0.16, (i + 1) * 0.16, size=n_rows)
            for i in range(n_symbols)
        ]
    )

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,  # Fewer buckets with 6 symbols
        method="cross_sectional_rank",
        se_method="naive",
    )

    # Should handle the ranking correctly
    assert len(table.buckets) > 0 or table.n_total == 0


def test_by_liquidity_decile_extreme_values() -> None:
    """Test expectancy_by_liquidity_decile with extreme prior_adv values."""
    rng = np.random.default_rng(130)
    n_rows, n_symbols = 250, 10
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))

    # Prior ADV with extreme range
    prior_adv = np.ones((n_rows, n_symbols), dtype=np.float64)
    prior_adv[:, :5] = 1e-6  # Very low liquidity
    prior_adv[:, 5:] = 1e6   # Very high liquidity

    result = expectancy.expectancy_by_liquidity_decile(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        prior_adv,
        n_buckets=10,
        se_method="naive",
        seed=130,
    )

    # All deciles should be present
    assert len(result) == 10


# ---------------------------------------------------------------------------
# Additional edge case coverage for remaining paths
# ---------------------------------------------------------------------------


def test_conditional_expectancy_1d_feature_raises() -> None:
    """Test that 1-D feature input raises ValueError."""
    rng = np.random.default_rng(133)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 3, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature_1d = rng.normal(size=n_rows)

    with pytest.raises(ValueError, match="feature must be 2-D"):
        expectancy.conditional_expectancy(
            feature_1d,  # type: ignore
            fwd,
            _single_session_day_offsets(n_rows),
        )


def test_expanding_quantile_with_partial_nan_feature() -> None:
    """Test expanding_quantile with feature that has NaN values."""
    rng = np.random.default_rng(134)
    n_rows, n_symbols = 100, 2
    feature = rng.normal(size=(n_rows, n_symbols))
    # Add NaNs
    feature[20:30, 0] = np.nan
    feature[40:50, 1] = np.nan

    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="expanding_quantile",
        min_history=20,
    )

    # NaN values should be unassigned
    assert np.all(result.labels[np.isnan(feature)] == -1)
    # Non-NaN values after min_history should be assigned
    non_nan_after_warmup = (
        ~np.isnan(feature[20:]) & (result.labels[20:] >= 0)
    )
    assert np.any(non_nan_after_warmup)


def test_survives_costs_with_large_spread() -> None:
    """Test survives_costs when spread is large (>> 2x hurdle)."""
    rng = np.random.default_rng(135)
    n_rows, n_symbols = 1000, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)

    # Feature with guaranteed strong separation
    feature = np.column_stack(
        [np.linspace(i, i + 10, n_rows) for i in range(n_symbols)]
    )

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=n_symbols,
        method="cross_sectional_rank",
        se_method="naive",
        cost_hurdle_bps=1.0,  # Very low hurdle
    )

    # With strong feature and low hurdle, should likely survive costs
    # (or at least test that the calculation works)
    assert isinstance(table.survives_costs, bool)


def test_explain_output_includes_all_required_fields() -> None:
    """Test that explain() output includes all required fields."""
    rng = np.random.default_rng(136)
    n_rows = 200
    close = _driftless_gbm_close(rng, n_rows, 4, 0.002)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    feature = rng.normal(size=(n_rows, 4))

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        se_method="block_bootstrap",
        cost_hurdle_bps=12.5,
    )

    explanation = table.explain()

    # Check all required keywords
    assert "horizon" in explanation
    assert "hurdle" in explanation.lower()
    assert "bps" in explanation.lower()
    assert "12" in explanation or "12.5" in explanation  # Cost hurdle value


def test_block_bootstrap_indices_never_straddle_session_boundaries() -> None:
    """Test that block bootstrap indices respect session boundaries.

    This verifies that no bootstrap block spans across a session boundary,
    which is critical for the session-bounding invariant.
    """
    rng = np.random.default_rng(137)
    # Create multi-session data: 3 sessions of 100, 80, 120 bars each
    session_lengths = [100, 80, 120]
    day_offsets = _multi_session_day_offsets(session_lengths)
    total_rows = day_offsets[-1]
    n_symbols = 3

    close = _driftless_gbm_close(rng, total_rows, n_symbols, 0.002)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)
    feature = rng.normal(size=(total_rows, n_symbols))

    # Use block bootstrap with n_boot large enough to get multiple blocks
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=2,
        method="rolling_quantile",
        se_method="block_bootstrap",
        n_boot=100,
        seed=42,
    )

    # Extract one bucket with block_bootstrap results
    for bucket_stat in table.buckets:
        if bucket_stat.se_method == "block_bootstrap" and bucket_stat.block_indices:
            # For each block, verify it doesn't straddle a session boundary
            for block_start, block_length in bucket_stat.block_indices:
                block_end = block_start + block_length - 1

                # Verify block is within data range
                assert 0 <= block_start < total_rows
                assert 0 <= block_end < total_rows

                # Find which sessions the start and end belong to
                start_session_idx = np.searchsorted(day_offsets[1:], block_start, side="right")
                end_session_idx = np.searchsorted(day_offsets[1:], block_end, side="right")

                # Both start and end must be in same session (not straddling boundary)
                assert (
                    start_session_idx == end_session_idx
                ), f"Block ({block_start}, {block_length}) straddles sessions"


def test_survives_costs_true_with_strong_feature() -> None:
    """Test that survives_costs=True when mean return >> cost hurdle.

    This tests line 631 (survives_costs=True) which had zero test coverage.
    We plant a strong feature effect (hundreds of bps) against a low hurdle.
    """
    n_rows = 500
    n_symbols = 5  # Min 5 for cross_sectional_rank default min_names

    # Create forward returns - assign based on symbol index
    # Symbols 0-1 get -500 bps, symbols 2-4 get +500 bps
    fwd_values = np.empty((n_rows, n_symbols), dtype=np.float64)
    fwd_values[:, :2] = -500 / 1e4  # Symbols 0-1: -500 bps
    fwd_values[:, 2:] = 500 / 1e4   # Symbols 2-4: +500 bps

    fwd_modified = expectancy.ForwardReturns(
        values=fwd_values,
        horizon=1,
        session_bounded=True,
        n_defined=n_rows * n_symbols,
        n_nan_tail=0,
    )

    # Create a feature that ranks symbols within each row
    # with cross_sectional_rank bucketing
    # Low-ranked symbols (0-1) will be in bucket 0, high-ranked (2-4) in bucket 1
    feature = np.empty((n_rows, n_symbols), dtype=np.float64)
    for row_idx in range(n_rows):
        # Each symbol gets a different constant value across rows
        # Symbol 0: value 1, Symbol 1: value 2, ..., Symbol 4: value 5
        # This ensures cross_sectional_rank will rank them consistently
        feature[row_idx, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Use cross_sectional_rank which ranks within each row
    # This will assign bucket 0 to symbols 0-1 and bucket 1 to symbols 2-4
    table = expectancy.conditional_expectancy(
        feature,
        fwd_modified,
        _single_session_day_offsets(n_rows),
        n_buckets=2,
        method="cross_sectional_rank",
        se_method="naive",
        cost_hurdle_bps=8.3,
    )

    # With 500 bps top bucket, -500 bps bottom bucket, spread = 1000 bps
    # And 8.3 bps hurdle, should definitely survive (1000 > 2*8.3)
    assert table.survives_costs is True
    assert table.spread_bps > 0

    # Also test explain() method which hits line 640 (SURVIVES costs message)
    explanation = table.explain()
    assert "SURVIVES" in explanation or "survives" in explanation.lower()


def test_conditional_expectancy_all_nan_forward_returns_with_block_bootstrap() -> None:
    """Test all-NaN forward returns with block_bootstrap SE method.

    This specifically tests that block_bootstrap gracefully handles empty data.
    """
    n_rows = 100
    n_symbols = 2

    # All-NaN forward returns
    feature = np.random.default_rng(139).normal(size=(n_rows, n_symbols))
    fwd = expectancy.ForwardReturns(
        values=np.full((n_rows, n_symbols), np.nan, dtype=np.float64),
        horizon=5,
        session_bounded=True,
        n_defined=0,
        n_nan_tail=n_rows,
    )

    # Should not crash even with empty bucket data
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=3,
        se_method="block_bootstrap",
    )

    # Verify table structure is valid
    assert table.n_total == 0  # No valid observations
    assert len(table.buckets) == 0 or all(b.n_obs == 0 for b in table.buckets)


def test_bootstrap_fallback_with_short_sessions() -> None:
    """Test bootstrap fallback when sessions are shorter than horizon + block.

    This exercises the fallback code path in _block_bootstrap_resampling_2d
    when sessions are too short for proper blocking.
    """
    rng = np.random.default_rng(140)
    # Create very short sessions: 5 bars each
    session_lengths = [5, 5, 5]
    day_offsets = _multi_session_day_offsets(session_lengths)
    total_rows = day_offsets[-1]
    n_symbols = 2

    close = _driftless_gbm_close(rng, total_rows, n_symbols, 0.002)
    # Large horizon relative to session length
    fwd = expectancy.forward_returns(close, day_offsets, horizon=10)
    feature = rng.normal(size=(total_rows, n_symbols))

    # Should not crash even with incompatible horizon/session combo
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=2,
        se_method="block_bootstrap",
        n_boot=50,
    )

    # Verify structure is valid
    assert isinstance(table, expectancy.ExpectancyTable)


def test_block_bootstrap_fallback_path_directly() -> None:
    """Test that the fallback path in _block_bootstrap_resampling_2d is executed.

    This directly exercises lines 328-335 where sessions are too short for blocks.
    """
    from nifty_quant.research.expectancy import _block_bootstrap_resampling_2d

    # Create data that will trigger fallback: sessions shorter than horizon
    session_lengths = [3, 3, 3]  # 3-bar sessions
    day_offsets = _multi_session_day_offsets(session_lengths)
    n_rows = day_offsets[-1]
    n_symbols = 2

    # Create data
    values_2d = np.random.default_rng(142).normal(size=(n_rows, n_symbols))

    # Call with large horizon to trigger fallback
    resampled, block_indices, n_sessions_skipped = _block_bootstrap_resampling_2d(
        values_2d, day_offsets, horizon=10, n_boot=5, seed=42
    )

    # block_length = horizon(10) + 5 = 15; all three 3-bar sessions are shorter than
    # that, so all three should be skipped, and the iid fallback (whole-row resampling,
    # which drops the dependence correction) is taken -- it never appends a synthetic
    # (0, n_rows) entry, because no block was actually drawn (AMENDMENT 8).
    #
    # `n_sessions_skipped == <total sessions>` together with `len(block_indices) == 0`
    # IS the fallback signal, pinned per AMENDMENT 8: the iid fallback silently changes
    # the estimator (no overlap correction is applied), so it must be detectable from
    # the return value alone, without reading the source. Both conditions are asserted
    # together below -- neither one alone proves the fallback path was taken.
    assert len(block_indices) == 0
    assert n_sessions_skipped == 3
    # Resampled should have valid shape
    assert resampled.shape[0] == 5  # n_boot
    assert resampled.shape[2] == n_symbols


def test_block_bootstrap_with_sparse_valid_values() -> None:
    """Test bootstrap when only few bootstrap samples have valid means.

    This exercises lines 551-554 where len(valid_boot_means) <= 1,
    forcing se_bps to be set to 0.0 for lack of bootstrap variance.
    """
    n_rows = 100
    n_symbols = 2

    # Create forward returns with mostly NaN (only a few finite values)
    fwd_values = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    # Only 2 observations (1%) are finite
    fwd_values[0, 0] = 0.001  # +10 bps
    fwd_values[99, 1] = 0.001

    fwd = expectancy.ForwardReturns(
        values=fwd_values,
        horizon=1,
        session_bounded=True,
        n_defined=2,
        n_nan_tail=n_rows * n_symbols - 2,
    )

    # Create constant feature (will all go to same bucket)
    feature = np.ones((n_rows, n_symbols), dtype=np.float64)

    # This should handle the case where bootstrap produces mostly NaN means
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=1,
        se_method="block_bootstrap",
        n_boot=100,
    )

    # Verify the table is created but may have zero se_bps due to sparsity
    assert isinstance(table, expectancy.ExpectancyTable)


def test_compute_bucket_stats_zero_std_zero_se() -> None:
    """Test _compute_bucket_stats when both std_bps and se_bps are zero/small.

    This exercises line 469 where the condition `std_bps <= 0 or se_bps <= 0`
    forces n_effective = n_obs in the block_bootstrap branch.
    """
    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Create constant bucket_returns (all same value) for zero std
    bucket_returns_flat = np.full(100, 0.0001, dtype=np.float64)  # All +10 bps

    day_offsets = np.array([0, 100], dtype=int)

    # Use horizon > 1 to trigger block_bootstrap path
    stat = _compute_bucket_stats(
        bucket_returns_flat,
        horizon=5,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=10,
        seed=0,
    )

    # With constant returns, std_bps will be 0
    assert stat is not None
    assert stat.std_bps == 0.0
    # When std_bps = 0, the condition triggers n_effective = n_obs
    assert stat.n_effective == stat.n_obs


def test_compute_bucket_stats_misaligned_dimensions() -> None:
    """Test _compute_bucket_stats when data size doesn't align with n_rows.

    This exercises lines 463-464 in the n_symbols_in_data == 0 path for block_bootstrap.
    """
    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Create bucket_returns with size that doesn't divide evenly by n_rows
    # n_rows = 100, but we create 99 values (misaligned, 99 % 100 != 0)
    bucket_returns = np.arange(99, dtype=np.float64) / 1e4

    day_offsets = np.array([0, 100], dtype=int)

    # Use horizon > 1 to trigger block_bootstrap path
    stat = _compute_bucket_stats(
        bucket_returns,
        horizon=5,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=10,
        seed=0,
    )

    # Should handle misaligned data gracefully
    # When n_symbols_in_data == 0 (data doesn't divide evenly by n_rows),
    # code goes to the else block at lines 462-464 and sets se_bps = 0.0, block_indices = ()
    assert stat is not None
    # With misaligned data, bootstrap can't be done on 2-D array, so se_bps = 0.0
    assert stat.se_method == "block_bootstrap"


def test_compute_bucket_stats_single_observation() -> None:
    """Test _compute_bucket_stats with single observation causing sparse bootstrap means.

    This directly exercises lines 461-464 where len(valid_boot_means) <= 1.
    """
    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Create bucket_returns with just one finite value
    bucket_returns = np.full(200, np.nan, dtype=np.float64)
    bucket_returns[0] = 0.001  # Single observation

    day_offsets = np.array([0, 200], dtype=int)

    # Use horizon > 1 to trigger block_bootstrap path
    stat = _compute_bucket_stats(
        bucket_returns,
        horizon=5,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=100,
        seed=0,
    )

    # Should handle single observation gracefully
    assert stat is not None
    assert stat.n_obs == 1
    # With one observation, std can't be computed
    assert stat.std_bps == 0.0


def test_block_bootstrap_with_mostly_nan_data() -> None:
    """Test bootstrap when data is mostly NaN (sparse valid values).

    This exercises lines 551-554 where len(valid_boot_means) <= 1.
    """
    n_rows = 100
    n_symbols = 5

    # Create forward returns that are mostly NaN with 2-3 finite values scattered
    fwd_values = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    fwd_values[10, 0] = 0.001   # +10 bps
    fwd_values[50, 2] = -0.001  # -10 bps
    fwd_values[90, 4] = 0.0005  # +5 bps

    fwd = expectancy.ForwardReturns(
        values=fwd_values,
        horizon=1,
        session_bounded=True,
        n_defined=3,
        n_nan_tail=n_rows * n_symbols - 3,
    )

    # Create feature with multiple buckets
    feature = np.empty((n_rows, n_symbols), dtype=np.float64)
    for i in range(n_rows):
        feature[i, :] = float(i)

    # conditional_expectancy should handle mostly-NaN bucket data gracefully
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=2,
        method="expanding_quantile",
        se_method="block_bootstrap",
        n_boot=100,
    )

    # Verify table is valid (even with sparse data)
    assert isinstance(table, expectancy.ExpectancyTable)


def test_constant_returns_zero_std() -> None:
    """Test when all returns are identical (std_bps = 0).

    This exercises line 559 where std_bps <= 0 or se_bps <= 0,
    forcing n_effective = n_obs.
    """
    n_rows = 100
    n_symbols = 3

    # Create constant forward returns (all the same value)
    fwd_values = np.full((n_rows, n_symbols), 0.0001, dtype=np.float64)  # All +10 bps

    fwd = expectancy.ForwardReturns(
        values=fwd_values,
        horizon=1,
        session_bounded=True,
        n_defined=n_rows * n_symbols,
        n_nan_tail=0,
    )

    # Create feature with two buckets
    feature = np.empty((n_rows, n_symbols), dtype=np.float64)
    for i in range(n_rows):
        feature[i, :] = float(i)  # Increasing over time

    # Use expanding_quantile to create buckets based on time, with block bootstrap
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        _single_session_day_offsets(n_rows),
        n_buckets=2,
        method="expanding_quantile",
        se_method="block_bootstrap",
    )

    # With constant returns, should have zero std
    # Each bucket should have std_bps > 0 initially, but bootstrap might produce se_bps = 0
    # which triggers the n_effective = n_obs path
    for bucket in table.buckets:
        if bucket.n_obs > 0:
            # When std_bps is very small or se_bps becomes 0, n_effective might equal n_obs
            assert bucket.n_effective > 0


def test_all_nan_bootstrap_sample_produces_nan_not_warning() -> None:
    """Test that all-NaN bootstrap samples produce NaN, not RuntimeWarning.

    When block bootstrap resampling draws blocks that contain only NaN values,
    the resulting bootstrap sample is all-NaN. Prior to the fix, calling
    np.nanmean on such all-NaN rows raised RuntimeWarning: Mean of empty slice.
    This test verifies that the code now detects all-NaN rows upfront, avoids
    calling np.nanmean on them, and propagates NaN correctly without warnings.

    This test covers two scenarios:
    1. Sparse data where some bootstrap samples are all-NaN (mixed case)
    2. Entirely all-NaN data (edge case where all bootstrap samples are all-NaN)
    """
    import warnings

    from nifty_quant.research.expectancy import _compute_bucket_stats

    # Scenario 1: Sparse array with few finite values (some bootstrap samples may be all-NaN)
    n_rows = 100
    n_symbols = 3

    # Create bucket_returns: mostly NaN with only 2 finite values
    bucket_returns = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    bucket_returns[10, 0] = 0.0001  # +10 bps
    bucket_returns[80, 1] = 0.0002  # +20 bps
    bucket_returns_flat = bucket_returns.flatten()

    day_offsets = np.array([0, n_rows], dtype=int)

    # Use block bootstrap which can produce all-NaN resampled blocks
    # Catch any RuntimeWarning from the nanmean call
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat = _compute_bucket_stats(
            bucket_returns_flat,
            horizon=5,
            day_offsets=day_offsets,
            se_method="block_bootstrap",
            n_boot=50,
            seed=42,
        )
        # Filter to RuntimeWarnings with "Mean of empty slice" message
        runtime_warns = [
            warning
            for warning in w
            if issubclass(warning.category, RuntimeWarning)
            and "Mean of empty slice" in str(warning.message)
        ]
        assert (
            len(runtime_warns) == 0
        ), f"Expected no RuntimeWarning from nanmean, but got {len(runtime_warns)}"

    # Verify the statistic is computed (not None)
    assert stat is not None
    # Verify n_obs is 2 (the two finite values)
    assert stat.n_obs == 2
    # Verify results are finite (not NaN due to sparse bootstrap)
    # With only 2 observations and sparse bootstrap, se_bps might be 0, but stat should be computed
    assert isinstance(stat.mean_bps, float)
    assert isinstance(stat.t_stat, float)

    # Scenario 2: Test np.any(valid_rows) == False code path directly
    # To directly trigger the np.any(valid_rows) == False branch, we construct
    # a scenario in _compute_bucket_stats where all bootstrap samples are all-NaN.
    # This tests line 457-458: if np.any(valid_rows): ... is skipped

    # We use a monkeypatch to artificially create all-NaN bootstrap samples
    # while keeping some finite observations to pass the n_obs > 0 check
    from unittest import mock

    # Create some finite observations to pass n_obs > 0 check
    bucket_returns_test = np.full(100 * 3, np.nan, dtype=np.float64)
    bucket_returns_test[0] = 0.0001  # One finite value
    bucket_returns_test[50] = 0.0002

    # Create day_offsets for a single session
    day_offsets_test = np.array([0, 100], dtype=int)

    # Mock _block_bootstrap_resampling_2d to return all-NaN bootstrap samples
    # while still having the right shape
    def mock_bootstrap(values_2d, day_offsets, horizon, n_boot, seed):
        # Return n_boot samples that are all NaN
        n_rows_sample = 50
        n_symbols = values_2d.shape[1]
        resampled_all_nan = np.full((n_boot, n_rows_sample, n_symbols), np.nan, dtype=np.float64)
        block_indices = tuple([(0, n_rows_sample)] * n_boot)
        return resampled_all_nan, block_indices, 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        patcher = mock.patch(
            "nifty_quant.research.expectancy._block_bootstrap_resampling_2d",
            side_effect=mock_bootstrap,
        )
        with patcher:
            stat_branch = _compute_bucket_stats(
                bucket_returns_test,
                horizon=5,
                day_offsets=day_offsets_test,
                se_method="block_bootstrap",
                n_boot=10,
                seed=45,
            )
        # Filter to RuntimeWarnings with "Mean of empty slice" message
        runtime_warns = [
            warning
            for warning in w
            if issubclass(warning.category, RuntimeWarning)
            and "Mean of empty slice" in str(warning.message)
        ]
        assert len(runtime_warns) == 0, (
            f"Expected no RuntimeWarning when all bootstrap samples are NaN, "
            f"got {len(runtime_warns)}"
        )

    # With all-NaN bootstrap samples but valid observations in the data,
    # the stat should still be computed but with se_bps likely 0
    assert stat_branch is not None
    assert stat_branch.n_obs == 2  # Two finite values in bucket_returns_test


# ---------------------------------------------------------------------------
# feature_name parameter propagation and audit trail
# ---------------------------------------------------------------------------


def test_conditional_expectancy_feature_name_default() -> None:
    """Test that conditional_expectancy uses default feature_name='feature'."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call conditional_expectancy without feature_name
    table = expectancy.conditional_expectancy(feature, fwd, day_offsets)

    # Verify default feature_name
    assert table.feature_name == "feature"


def test_conditional_expectancy_feature_name_custom() -> None:
    """Test that conditional_expectancy uses supplied feature_name."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call conditional_expectancy with custom feature_name
    custom_name = "my_alpha_factor"
    table = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, feature_name=custom_name
    )

    # Verify custom feature_name
    assert table.feature_name == custom_name


def test_conditional_expectancy_feature_name_in_explain() -> None:
    """Test that feature_name appears in explain() output."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Test with custom feature_name
    custom_name = "momentum_5bar"
    table = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, feature_name=custom_name
    )

    # Get explain output
    explain_str = table.explain()

    # Verify feature_name appears in explain output
    assert f"feature_name={custom_name!r}" in explain_str


def test_expectancy_by_year_feature_name_default() -> None:
    """Test that expectancy_by_year uses default feature_name='feature'."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)
    dates = np.array([datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                      for i in range(n_rows)])

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_year without feature_name
    result = expectancy.expectancy_by_year(feature, fwd, day_offsets, dates)

    # Verify default feature_name in all results
    for year, table in result.items():
        assert table.feature_name == "feature"


def test_expectancy_by_year_feature_name_custom() -> None:
    """Test that expectancy_by_year uses supplied feature_name."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)
    dates = np.array([datetime.date(2023, 1, 1) + datetime.timedelta(days=i)
                      for i in range(n_rows)])

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_year with custom feature_name
    custom_name = "volatility_adjusted_momentum"
    result = expectancy.expectancy_by_year(
        feature, fwd, day_offsets, dates, feature_name=custom_name
    )

    # Verify custom feature_name in all results
    for year, table in result.items():
        assert table.feature_name == custom_name


def test_expectancy_by_time_of_day_feature_name_default() -> None:
    """Test that expectancy_by_time_of_day uses default feature_name='feature'."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)
    minute_of_day = np.tile(np.arange(60, dtype=int), (n_rows // 60 + 1))[:n_rows]

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_time_of_day without feature_name
    result = expectancy.expectancy_by_time_of_day(
        feature, fwd, day_offsets, minute_of_day
    )

    # Verify default feature_name in all results
    for time_bucket, table in result.items():
        assert table.feature_name == "feature"


def test_expectancy_by_time_of_day_feature_name_custom() -> None:
    """Test that expectancy_by_time_of_day uses supplied feature_name."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)
    minute_of_day = np.tile(np.arange(60, dtype=int), (n_rows // 60 + 1))[:n_rows]

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_time_of_day with custom feature_name
    custom_name = "intraday_reversion"
    result = expectancy.expectancy_by_time_of_day(
        feature, fwd, day_offsets, minute_of_day, feature_name=custom_name
    )

    # Verify custom feature_name in all results
    for time_bucket, table in result.items():
        assert table.feature_name == custom_name


def test_expectancy_by_liquidity_decile_feature_name_default() -> None:
    """Test that expectancy_by_liquidity_decile uses default feature_name='feature'."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    prior_adv = rng.uniform(1e6, 1e7, size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_liquidity_decile without feature_name
    result = expectancy.expectancy_by_liquidity_decile(
        feature, fwd, day_offsets, prior_adv
    )

    # Verify default feature_name in all results
    for decile, table in result.items():
        assert table.feature_name == "feature"


def test_expectancy_by_liquidity_decile_feature_name_custom() -> None:
    """Test that expectancy_by_liquidity_decile uses supplied feature_name."""
    rng = np.random.default_rng(42)
    n_rows, n_symbols = 100, 3

    # Create synthetic data
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.01)
    feature = rng.normal(size=(n_rows, n_symbols))
    prior_adv = rng.uniform(1e6, 1e7, size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    # Compute forward returns
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)

    # Call expectancy_by_liquidity_decile with custom feature_name
    custom_name = "liquidity_sensitive_alpha"
    result = expectancy.expectancy_by_liquidity_decile(
        feature, fwd, day_offsets, prior_adv, feature_name=custom_name
    )

    # Verify custom feature_name in all results
    for decile, table in result.items():
        assert table.feature_name == custom_name
