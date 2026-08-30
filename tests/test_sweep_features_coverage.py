"""Coverage for adapter functions in sweep_features.py.

This suite tests _tradable_overnight_return and _opening_range (lines 293-308),
which were never executed by any test. It calls them directly with small synthetic panels,
verifying shape, dtype, and value correctness.

Both adapters use synthetic data built at float32 (per repo rule 3) in the fixture, and
use day_offsets with unequal session lengths to reflect real data variance (never assume
375 bars per session; see rule 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features import market as _market
from nifty_quant.research.sweep_features import (
    _OPEN_MINUTE_IST,
    _opening_range,
    _tradable_overnight_return,
)

# ---------------------------------------------------------------------------
# Fixture: minimal synthetic panel with unequal sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small hand-built panel: 3 sessions, 2 symbols, float32 at rest per rule 3.

    Sessions: 10 bars, 8 bars, 12 bars (unequal, reflecting real variance).
    Returns: (close, open_, high, low, day_offsets) all float32 except day_offsets.
    """
    # Session 1: 10 rows, 2 symbols
    close_s1 = np.array(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
            [104.0, 204.0],
            [105.0, 205.0],  # bar at 09:20 (5 bars into session)
            [106.0, 206.0],
            [107.0, 207.0],
            [108.0, 208.0],
            [109.0, 209.0],  # session 1 closes at 109.0, 209.0
        ],
        dtype=np.float32,
    )

    # Session 2: 8 rows, 2 symbols (different price level for overnight gap)
    close_s2 = np.array(
        [
            [110.0, 210.0],
            [111.0, 211.0],
            [112.0, 212.0],
            [113.0, 213.0],  # bar at 09:19 (4 bars into session)
            [114.0, 214.0],  # bar at 09:20 (5 bars into session)
            [115.0, 215.0],
            [116.0, 216.0],
            [117.0, 217.0],  # session 2 closes at 117.0, 217.0
        ],
        dtype=np.float32,
    )

    # Session 3: 12 rows, 2 symbols
    close_s3 = np.array(
        [
            [118.0, 218.0],
            [119.0, 219.0],
            [120.0, 220.0],
            [121.0, 221.0],
            [122.0, 222.0],
            [123.0, 223.0],
            [124.0, 224.0],
            [125.0, 225.0],  # 8 bars into session; beyond 15-bar opening_range window
            [126.0, 226.0],
            [127.0, 227.0],
            [128.0, 228.0],
            [129.0, 229.0],  # session 3 closes at 129.0, 229.0
        ],
        dtype=np.float32,
    )

    close = np.vstack([close_s1, close_s2, close_s3]).astype(np.float32)
    day_offsets = np.array([0, 10, 18, 30], dtype=np.int64)

    # Build open_ with a realistic overnight gap
    open_ = close.copy()
    open_[0, :] = [100.5, 200.5]  # slight gap at open
    open_[10, :] = [110.5, 210.5]  # overnight gap between sessions 1 and 2
    open_[18, :] = [118.5, 218.5]  # overnight gap between sessions 2 and 3

    # high/low: high is always >= close, low is always <= close
    # Opening range is computed from first 15 bars, so we need meaningful highs/lows there.
    high = close + 0.5  # each bar's high is 0.5 above close
    low = close - 0.3  # each bar's low is 0.3 below close

    # Session 1 (bars 0-9): craft specific highs/lows for opening range test
    # We want the opening high/low over first 10 bars, but opening_range uses first 15 bars
    # so for session 1 (only 10 bars), rows 0-14 should be NaN, and rows 15+ should have the range
    high[0:10, 0] = [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5]
    low[0:10, 0] = [99.7, 100.7, 101.7, 102.7, 103.7, 104.7, 105.7, 106.7, 107.7, 108.7]

    high[0:10, 1] = [200.5, 201.5, 202.5, 203.5, 204.5, 205.5, 206.5, 207.5, 208.5, 209.5]
    low[0:10, 1] = [199.7, 200.7, 201.7, 202.7, 203.7, 204.7, 205.7, 206.7, 207.7, 208.7]

    # Session 2 (bars 10-17): 8 bars, all should be NaN for opening_range (< 15 bars)
    high[10:18, 0] = [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5]
    low[10:18, 0] = [109.7, 110.7, 111.7, 112.7, 113.7, 114.7, 115.7, 116.7]

    high[10:18, 1] = [210.5, 211.5, 212.5, 213.5, 214.5, 215.5, 216.5, 217.5]
    low[10:18, 1] = [209.7, 210.7, 211.7, 212.7, 213.7, 214.7, 215.7, 216.7]

    # Session 3 (bars 18-29): 12 bars, still < 15, so all NaN for opening_range
    high_s3 = [118.5, 119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5, 126.5, 127.5, 128.5, 129.5]
    low_s3 = [117.7, 118.7, 119.7, 120.7, 121.7, 122.7, 123.7, 124.7, 125.7, 126.7, 127.7, 128.7]
    high[18:30, 0] = high_s3
    low[18:30, 0] = low_s3

    high_s3_1 = [218.5, 219.5, 220.5, 221.5, 222.5, 223.5, 224.5, 225.5, 226.5, 227.5, 228.5, 229.5]
    low_s3_1 = [217.7, 218.7, 219.7, 220.7, 221.7, 222.7, 223.7, 224.7, 225.7, 226.7, 227.7, 228.7]
    high[18:30, 1] = high_s3_1
    low[18:30, 1] = low_s3_1

    # Ensure float32 at rest
    open_ = open_.astype(np.float32)
    high = high.astype(np.float32)
    low = low.astype(np.float32)
    close = close.astype(np.float32)

    return close, open_, high, low, day_offsets


# ---------------------------------------------------------------------------
# Test `_tradable_overnight_return`: adapter calls market.tradable_overnight_return
# ---------------------------------------------------------------------------


def test_tradable_overnight_return_shape_dtype(synthetic_panel: tuple) -> None:
    """_tradable_overnight_return returns shape matching input and dtype float64."""
    close, open_, high, low, day_offsets = synthetic_panel
    n_rows, n_symbols = close.shape

    result = _tradable_overnight_return(close, day_offsets, open_=open_)

    # Output shape must match input
    expected_shape = (n_rows, n_symbols)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Output dtype must be float64 (rule 3: float64 in motion)
    assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"


def test_tradable_overnight_return_first_session_is_nan(synthetic_panel: tuple) -> None:
    """First session has no prior session to reference, so all rows are NaN."""
    close, open_, high, low, day_offsets = synthetic_panel

    result = _tradable_overnight_return(close, day_offsets, open_=open_)

    # Session 1 (rows 0-9) should be all NaN
    assert np.all(np.isnan(result[0:10, :]))


def test_tradable_overnight_return_broadcasts_within_session(synthetic_panel: tuple) -> None:
    """Within a session, all rows must have the same overnight return value.

    Session 2 (rows 10-17) uses:
    - Entry open at 09:16 (16 minutes since 09:00, i.e. minute 555 + 16 = 571)
    - Prior session's close at 09:20 (20 minutes, i.e. minute 555 + 20 = 575)

    In our synthetic data:
    - Session 1 closes at [109.0, 209.0]
    - Session 2: row 10 is at minute 555 (first bar), row 14 is at minute 559
      (from _minute_of_day_proxy: minute = 555 + bars_since_session_start)
    - Row 14 is at minute 559, row 15 would be at minute 560, etc.
    - We need bar at 09:16 (minute 555+16=571) and prior session bar at 09:20 (minute 555+20=575).

    Actually: minute_of_day = 555 + bars_since_open, so:
    - Bar 0 of session -> minute 555 (09:15)
    - Bar 1 of session -> minute 556 (09:16)
    - Bar 4 of session -> minute 559 (09:19)
    - Bar 5 of session -> minute 560 (09:20)

    So in session 2 (rows 10-17):
    - Row 10 is bar 0 -> minute 555
    - Row 14 is bar 4 -> minute 559
    - Row 15 is bar 5 -> minute 560

    In session 1 (rows 0-9):
    - Row 9 is bar 9 -> minute 564

    The function defaults to entry_hhmm="09:16" (minute 556) and exit_hhmm="15:20" (minute 920).

    Session 1 (rows 0-9):
    - Entry bar (minute 556) is row 1
    - Exit bar (minute 920) doesn't exist (session only goes to minute 564)

    So session 2 will look for:
    - Entry bar (minute 556) in rows 10-17: row 11 (bar 1 of session -> 555+1=556)
    - Exit bar (minute 920) in session 1 (rows 0-9): doesn't exist

    So session 2 should also be NaN (missing exit bar in prior session).

    Let's verify the function broadcasts correctly by checking that all rows
    in a session have identical values (whether NaN or a number).
    """
    close, open_, high, low, day_offsets = synthetic_panel

    result = _tradable_overnight_return(close, day_offsets, open_=open_)

    # Session 2 (rows 10-17): all rows must be equal (or all NaN if labels missing)
    session2_vals = result[10:18, 0]  # First symbol, session 2
    # All values in the session must be identical
    if np.isnan(session2_vals[0]):
        msg = "Session 2 rows should all be NaN or all have same value"
        assert np.all(np.isnan(session2_vals)), msg
    else:
        np.testing.assert_allclose(session2_vals, session2_vals[0], rtol=1e-15)

    session2_vals_s1 = result[10:18, 1]  # Second symbol, session 2
    if np.isnan(session2_vals_s1[0]):
        msg = "Session 2 rows should all be NaN or all have same value"
        assert np.all(np.isnan(session2_vals_s1)), msg
    else:
        np.testing.assert_allclose(session2_vals_s1, session2_vals_s1[0], rtol=1e-15)


def test_tradable_overnight_return_uses_minute_of_day_proxy(synthetic_panel: tuple) -> None:
    """Verify the adapter correctly computes and passes minute_of_day to the underlying function.

    The _minute_of_day_proxy should produce: _OPEN_MINUTE_IST (555) + bars_since_open.
    First bar of session -> minute 555 (09:15).
    Second bar -> minute 556 (09:16), etc.
    """
    close, open_, high, low, day_offsets = synthetic_panel

    # Manually build what _minute_of_day_proxy should produce
    n_rows = close.shape[0]
    bars_since_open = _market.bars_since_open(day_offsets, n_rows)
    expected_minute_of_day = (_OPEN_MINUTE_IST + bars_since_open).astype(np.int64)

    # Call the adapter
    result = _tradable_overnight_return(close, day_offsets, open_=open_)

    # The result should match calling market.tradable_overnight_return directly
    # with the correct minute_of_day
    direct_result = _market.tradable_overnight_return(
        open_, close, day_offsets, expected_minute_of_day
    )

    np.testing.assert_array_equal(result, direct_result)


# ---------------------------------------------------------------------------
# Test `_opening_range`: adapter calls market.opening_range and returns range width
# ---------------------------------------------------------------------------


def test_opening_range_shape_dtype(synthetic_panel: tuple) -> None:
    """_opening_range returns shape matching input and dtype float64."""
    close, open_, high, low, day_offsets = synthetic_panel
    n_rows, n_symbols = close.shape

    result = _opening_range(close, day_offsets, high=high, low=low)

    # Output shape must match input
    expected_shape = (n_rows, n_symbols)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Output dtype must be float64
    assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"


def test_opening_range_before_n_bars_is_nan(synthetic_panel: tuple) -> None:
    """Rows before n_bars (default 15) of each session must be NaN (no lookahead).

    Session 1: 10 bars total, all rows should be NaN (< 15).
    Session 2: 8 bars total, all rows should be NaN (< 15).
    Session 3: 12 bars total, all rows should be NaN (< 15).
    """
    close, open_, high, low, day_offsets = synthetic_panel

    result = _opening_range(close, day_offsets, high=high, low=low)

    # All rows are in sessions < 15 bars, so all should be NaN
    assert np.all(np.isnan(result)), "All rows should be NaN when all sessions are < n_bars"


def test_opening_range_computes_high_minus_low_as_float64(synthetic_panel: tuple) -> None:
    """_opening_range should return (high_range - low_range) as float64.

    Construct a panel where we can verify this by hand.
    """
    # Small panel: one 20-bar session, 1 symbol
    # We'll set specific highs/lows in the opening 15 bars
    # Then after bar 15, verify that the output is the correct range width.

    close = np.arange(20.0, dtype=np.float32).reshape(20, 1) + 100.0  # 100, 101, 102, ..., 119
    high = close + 0.5
    low = close - 0.5
    day_offsets = np.array([0, 20], dtype=np.int64)

    # Manually compute the opening range over first 15 bars
    # opening_high = max(high[0:15]) = max(100.5, 101.5, ..., 114.5) = 114.5
    # opening_low = min(low[0:15]) = min(99.5, 100.5, ..., 113.5) = 99.5
    # range = 114.5 - 99.5 = 15.0

    result = _opening_range(close, day_offsets, high=high, low=low)

    # Rows 0-14 should be NaN (no lookahead)
    assert np.all(np.isnan(result[0:15, 0])), "Rows 0-14 should be NaN"

    # Rows 15+ should all equal the range: 114.5 - 99.5 = 15.0
    expected_range = 114.5 - 99.5
    np.testing.assert_allclose(result[15:, 0], expected_range, rtol=1e-14)


def test_opening_range_broadcasts_within_session(synthetic_panel: tuple) -> None:
    """Within a session, after row n_bars, all rows should have the same range value."""
    # Create a larger panel with one 30-bar session, 2 symbols
    close = np.ones((30, 2), dtype=np.float32) * 100.0
    high = close + np.linspace(0.5, 1.5, 30).reshape(-1, 1)  # Varying highs
    low = close - np.linspace(0.3, 0.8, 30).reshape(-1, 1)  # Varying lows
    day_offsets = np.array([0, 30], dtype=np.int64)

    result = _opening_range(close, day_offsets, high=high, low=low)

    # After bar 15 (row 15), all rows in the session should have the same value
    session_range = result[15:30, 0]
    # All values must be identical (the opening range width)
    if not np.all(np.isnan(session_range)):
        expected = session_range[0]
        np.testing.assert_allclose(session_range, expected, rtol=1e-14)

    session_range_s1 = result[15:30, 1]
    if not np.all(np.isnan(session_range_s1)):
        expected = session_range_s1[0]
        np.testing.assert_allclose(session_range_s1, expected, rtol=1e-14)


def test_opening_range_per_symbol_independence(synthetic_panel: tuple) -> None:
    """Each symbol's range should be computed independently.

    Create panel with very different ranges per symbol.
    """
    close = np.ones((20, 2), dtype=np.float32) * 100.0
    day_offsets = np.array([0, 20], dtype=np.int64)

    # Symbol 0: tight range (0.5 - (-0.5) = 1.0)
    high_s0 = np.ones(20, dtype=np.float32) * 100.5
    low_s0 = np.ones(20, dtype=np.float32) * 99.5

    # Symbol 1: wide range (2.0 - (-2.0) = 4.0)
    high_s1 = np.ones(20, dtype=np.float32) * 102.0
    low_s1 = np.ones(20, dtype=np.float32) * 98.0

    high = np.column_stack([high_s0, high_s1])
    low = np.column_stack([low_s0, low_s1])

    result = _opening_range(close, day_offsets, high=high, low=low)

    # After row 15, symbol 0 should have range ~1.0
    np.testing.assert_allclose(result[15, 0], 1.0, rtol=1e-14)

    # After row 15, symbol 1 should have range ~4.0
    np.testing.assert_allclose(result[15, 1], 4.0, rtol=1e-14)


def test_opening_range_close_parameter_unused(synthetic_panel: tuple) -> None:
    """The close parameter is unused in _opening_range (docstring says so).

    Verify that changing close doesn't affect the result.
    """
    close, open_, high, low, day_offsets = synthetic_panel
    close2 = close * 2.0  # Drastically different close

    result1 = _opening_range(close, day_offsets, high=high, low=low)
    result2 = _opening_range(close2, day_offsets, high=high, low=low)

    np.testing.assert_array_equal(result1, result2)
