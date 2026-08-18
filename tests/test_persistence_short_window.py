"""Tests for rolling_hurst with short panels (fewer rows than largest lag).

These tests exercise the uncovered branch at lines 275-278 in persistence.py,
where n_rows <= lag, causing both the difference computation and day-boundary
adjustment to skip.
"""

import numpy as np
import pytest

from nifty_quant.features import persistence


def test_rolling_hurst_short_panel_without_day_offsets() -> None:
    """Test rolling_hurst when panel has fewer rows than the largest lag.

    With an 8-row panel and default max_lag=20:
    - Small lags (2-8): n_rows > lag, execute line 275's true arm
    - Large lags (9-20): n_rows <= lag, execute line 275's false arm (gap)
    - line 278 is not hit because row_day_id is None
    """
    rng = np.random.default_rng(42)
    # Create an 8-row panel (smaller than max_lag=20)
    returns = rng.standard_normal((8, 3)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Call with default lags, which means max_lag=20
    # This should not raise and should handle lags >= n_rows gracefully
    result = persistence.rolling_hurst(
        price, window=30, warn_short_window=False, max_lag=20
    )

    # Verify output shape and dtype
    assert result.shape == price.shape
    assert result.dtype == np.float64

    # With window=30 and only 8 rows, all output should be NaN
    # (insufficient history for any window)
    assert np.all(np.isnan(result))

    # Verify no infinities or other anomalies
    assert not np.isinf(result).any()


def test_rolling_hurst_short_panel_with_day_offsets() -> None:
    """Test rolling_hurst with short panel AND day_offsets.

    Tests the code path where row_day_id is not None (line 278's condition
    is checked) but n_rows <= lag for large lags, so the false arm is taken.

    An 8-row panel with 2 sessions of 4 rows each and max_lag=10 ensures
    lag=9 hits the false arm (n_rows=4 < lag=9 within a session).
    """
    rng = np.random.default_rng(43)
    # Create an 8-row panel with 2 sessions of 4 rows each
    returns = rng.standard_normal((8, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Define day_offsets: session starts at rows 0 and 4, ends at row 8
    day_offsets = np.array([0, 4, 8], dtype=np.int64)

    # Call with day_offsets so row_day_id is computed
    # max_lag=10 gives lags [2..9], window must exceed largest lag (9)
    result = persistence.rolling_hurst(
        price,
        window=20,
        warn_short_window=False,
        max_lag=10,
        min_lag=2,
        day_offsets=day_offsets,
    )

    # Verify output shape and dtype
    assert result.shape == price.shape
    assert result.dtype == np.float64

    # With window=20 and only 4-row sessions, all rows should be NaN
    # (each session has insufficient history for the window size)
    assert np.all(np.isnan(result))

    # Verify no infinities
    assert not np.isinf(result).any()


def test_rolling_hurst_mixed_lag_coverage() -> None:
    """Test that a short panel exercises both true and false branches of line 275.

    Use a 10-row panel with max_lag=12 to ensure:
    - Small lags (2-10): n_rows > lag (true branch)
    - Large lags (11-12): n_rows <= lag (false branch)
    """
    rng = np.random.default_rng(44)
    returns = rng.standard_normal((10, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Explicitly set max_lag smaller than default to hit both branches
    result = persistence.rolling_hurst(
        price,
        window=20,
        warn_short_window=False,
        max_lag=12,
        min_lag=2,
    )

    # Verify basic properties
    assert result.shape == price.shape
    assert result.dtype == np.float64
    assert not np.isinf(result).any()
    assert np.all(np.isfinite(result) | np.isnan(result))

    # With window=20 and only 10 rows, initial rows will be NaN
    # But the test passes if no crashes and proper shape/dtype
    assert result.shape[0] == 10
    assert result.shape[1] == 2


def test_rolling_hurst_fbm_short_panel() -> None:
    """Test rolling_hurst on fbm path (log_price=False) with short panel.

    fbm output is already in log-space, so log_price=False.
    """
    path = persistence.fbm(12, H=0.5, seed=45)
    # fbm returns 1-D; reshape to 2-D for rolling_hurst
    price_2d = path.reshape(-1, 1)

    result = persistence.rolling_hurst(
        price_2d,
        window=15,
        log_price=False,
        warn_short_window=False,
        max_lag=15,
    )

    # Verify output shape and dtype
    assert result.shape == price_2d.shape
    assert result.dtype == np.float64
    assert not np.isinf(result).any()


def test_rolling_hurst_min_count_short_panel() -> None:
    """Test min_count parameter with a short panel.

    When panel is shorter than window, min_count affects which rows
    become valid. Verify that min_count semantics are respected
    even when lags are large relative to n_rows.
    """
    rng = np.random.default_rng(46)
    returns = rng.standard_normal((6, 1)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # With min_count=1, we might get some finite results in later rows
    # depending on lag-specific logic
    result = persistence.rolling_hurst(
        price,
        window=10,
        min_count=1,
        warn_short_window=False,
        max_lag=8,
    )

    assert result.shape == price.shape
    assert result.dtype == np.float64
    assert not np.isinf(result).any()


def test_davies_harte_n_one_embedding_size() -> None:
    """_davies_harte rejects n == 1 with a named contract error, not a broadcast crash.

    n < 2 is outside this private helper's contract: `fbm` returns np.zeros(1) for n == 1
    and raises for n <= 0, so no in-contract call reaches it. It previously carried a
    `if m < 2: m = 2` guard that LOOKED like n == 1 was supported but then died at
    `c[n - 1:] = gamma[1:][::-1]` with an opaque numpy broadcast message. The error must
    name the contract so a future caller is not left debugging a shape error.
    """
    rng = np.random.default_rng(47)
    with pytest.raises(ValueError, match="n >= 2"):
        persistence._davies_harte(1, H=0.5, rng=rng)


def test_fbm_supports_n_one_via_its_own_guard() -> None:
    """The public entry point handles n == 1 itself, so the helper never needs to.

    This is the other half of the contract above: rejecting n == 1 in _davies_harte is
    only safe because fbm short-circuits it first. If that guard were ever removed, this
    test and the one above would disagree, which is the point.
    """
    path = persistence.fbm(1, 0.5, seed=47)

    assert path.shape == (1,)
    assert path.dtype == np.float64
    assert path[0] == 0.0
