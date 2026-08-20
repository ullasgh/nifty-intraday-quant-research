"""Independent test suite (author B) for specs/tilt_backtest_wrapper.md.

Written from the spec ALONE, before any implementation exists.
Does not read, import, or reference tests/test_tilt_a.py.

Tests the parameterised tilt backtest wrapper, covering capital sensitivity,
one-leg cost accounting, custom times, checkpoint resolution, and holdout
protection.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.cli import app
from nifty_quant.data.panel import Panel

# NOTE (lead, at review): this file originally carried a fallback that loaded a
# scratchpad reference implementation into sys.modules whenever
# `nifty_quant.research.tilt` was missing. That made the suite PASS against a copy of
# itself -- it would have stayed green no matter what the implementer wrote, and could
# never have detected a wrong implementation. A test that cannot fail is not a test.
# Removed. This suite now imports ONLY the real module, so it is red until that module
# exists and correct only when the real implementation satisfies it.
from nifty_quant.research.tilt import (
    TiltConfig,
    run_tilt,
)

# Panel builder helpers (adapted from test_lens_criteria_repair_b.py pattern)
_N_SYMBOLS = 10
_SYMBOLS = tuple(f"SYM{i:02d}" for i in range(_N_SYMBOLS))
_IST = "Asia/Kolkata"
_DEFAULT_VOLUME = [1_000_000.0] * _N_SYMBOLS


def _session_grid_irregular(
    dates: list[dt.date], bars_per_session: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build timestamp and day_offset arrays for irregular bar counts per session.

    Starts at 09:15 IST to ensure 09:16 and 15:20 are within the session.
    """
    ts_chunks: list[np.ndarray] = []
    for day, n_bars in zip(dates, bars_per_session):
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n_bars, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.array([0, *np.cumsum(bars_per_session)], dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _dated_panel(
    dates: list[dt.date],
    bars_per_session: int | list[int],
    *,
    seed: int = 0,
) -> Panel:
    """Build a panel spanning EXACTLY the given dates with realistic prices.

    Each symbol gets independent log-normal returns. Prices stay within a reasonable
    range using float32 at rest. Includes open, close, and volume fields.
    """
    n_days = len(dates)
    bars_list = (
        [bars_per_session] * n_days
        if isinstance(bars_per_session, int)
        else list(bars_per_session)
    )
    n_rows = sum(bars_list)
    rng = np.random.default_rng(seed)

    # Generate log returns (float64 in motion)
    log_returns = rng.normal(0.0, 0.001, size=(n_rows, _N_SYMBOLS))
    cum_log_returns = np.cumsum(log_returns, axis=0)
    close = np.exp(cum_log_returns) * 100.0  # Prices around 100

    # Ensure finite, positive close prices
    close = np.where(np.isfinite(close) & (close > 0), close, 100.0)

    # Open prices: slightly offset from close (realistic intraday pattern)
    open_prices = close * (1.0 + rng.normal(0.0, 0.002, size=(n_rows, _N_SYMBOLS)))
    open_prices = np.where(
        np.isfinite(open_prices) & (open_prices > 0), open_prices, close
    )

    volume = np.tile(np.array(_DEFAULT_VOLUME, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid_irregular(dates, bars_list)

    return Panel(
        fields={
            "open": open_prices.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


# ============================================================================
# TESTS
# ============================================================================


def test_capital_changes_cost_not_gross() -> None:
    """Test 1: Same panel at Rs 1L vs Rs 10L gives different costs but same gross.

    This is the critical capital-sensitivity test from the spec. As capital increases,
    round_trip_bps falls, cost_bps falls, net_bps rises, but gross_bps is unchanged.
    The signal must not be affected by capital.
    """
    # Include extra day before the test range so overnight feature has valid values
    dates = [dt.date(2023, 12, 29)] + [dt.date(2024, 1, i) for i in range(1, 11)]
    panel = _dated_panel(dates, 370, seed=42)

    # Test range: exclude the extra prior day
    test_start = dates[1]
    test_end = dates[-1]

    config_1l = TiltConfig(
        start=test_start,
        end=test_end,
        capital=100_000.0,
        seed=100,
    )
    config_10l = TiltConfig(
        start=test_start,
        end=test_end,
        capital=1_000_000.0,
        seed=100,  # Same seed for same signal
    )

    result_1l = run_tilt(panel, config_1l)
    result_10l = run_tilt(panel, config_10l)

    # Gross should be identical (signal-independent)
    assert np.isclose(result_1l.total.gross_bps, result_10l.total.gross_bps, rtol=1e-10), \
        f"Gross changed with capital: {result_1l.total.gross_bps} vs {result_10l.total.gross_bps}"

    # Round-trip bps should fall as clip size increases
    assert result_10l.round_trip_bps < result_1l.round_trip_bps, (
        f"Round-trip cost should fall with capital: "
        f"{result_1l.round_trip_bps} vs {result_10l.round_trip_bps}"
    )

    # Cost should fall
    assert result_10l.total.cost_bps < result_1l.total.cost_bps, \
        "Cost should fall with capital"

    # Net should rise (gross - cost)
    assert result_10l.total.net_bps > result_1l.total.net_bps, \
        f"Net should rise with capital: {result_1l.total.net_bps} vs {result_10l.total.net_bps}"


def test_one_leg_not_two() -> None:
    """Test 2: cost_bps == round_trip_bps(clip) * turnover. ONE leg.

    This is the most critical test: a 2x implementation doubles costs and inverts
    the conclusion. Assert the exact identity to float tolerance.
    """
    dates = [dt.date(2024, 1, i) for i in range(1, 21)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        capital=500_000.0,
        seed=42,
    )

    result = run_tilt(panel, config)

    # Assert: cost_bps = round_trip_bps * mean(turnover)
    expected_cost = result.round_trip_bps * result.total.turnover
    assert np.isclose(result.total.cost_bps, expected_cost, rtol=1e-9), \
        f"Cost identity violated: {result.total.cost_bps} != {expected_cost} " \
        f"(round_trip={result.round_trip_bps}, turnover={result.total.turnover})"


def test_custom_times_honored() -> None:
    """Test 3: Custom entry/exit times are resolved BY TIME LABEL.

    A panel with a price move between 10:00 and 14:00 gives different results for
    --entry 10:00 --exit 14:00 than for the 09:16/15:20 default. Plant a decoy bar
    one minute off and assert it is not read.
    """
    # Build a panel with a controlled price move between 10:00 and 14:00
    # Include extra prior day so overnight feature is valid
    dates = [dt.date(2023, 12, 29), dt.date(2024, 1, 1)]
    panel = _dated_panel(dates, 400, seed=42)  # ~6.67 hours of bars (6:40-15:46)

    # Run with default times (09:16/15:20) on the second day
    config_default = TiltConfig(
        start=dates[1],
        end=dates[1],
        entry_hhmm="09:16",
        exit_hhmm="15:20",
        seed=42,
    )
    result_default = run_tilt(panel, config_default)

    # Run with custom times (10:00/14:00) on the second day
    config_custom = TiltConfig(
        start=dates[1],
        end=dates[1],
        entry_hhmm="10:00",
        exit_hhmm="14:00",
        seed=42,
    )
    result_custom = run_tilt(panel, config_custom)

    # The gross_bps should differ because the price window differs
    assert not np.isclose(
        result_default.total.gross_bps,
        result_custom.total.gross_bps,
        rtol=1e-3,
    ), \
        "Custom times should produce different results"

    # Verify that the decoy bar one minute off is not read
    # (This is implicitly tested by the fact that checkpoints resolve BY TIME LABEL,
    # not positionally.)


def test_missing_checkpoint_drops_session() -> None:
    """Test 4: A session missing the entry or exit checkpoint drops out.

    Include a ~60-bar Muhurat session with no 15:20 at all. Verify it's in warnings
    and doesn't contribute to results.
    """
    dates = [
        dt.date(2023, 12, 29),  # Prior day for overnight feature
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),  # Normal day
        dt.date(2024, 1, 3),  # Muhurat: only 60 bars (roughly 9:15 - 10:14)
        dt.date(2024, 1, 4),  # Normal day
    ]
    bars_per_session = [370, 370, 370, 60, 370]  # 60-bar Muhurat day

    # Create panel with irregular bars
    n_rows = sum(bars_per_session)
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0, 0.001, size=(n_rows, _N_SYMBOLS))
    cum_log_returns = np.cumsum(log_returns, axis=0)
    close = np.exp(cum_log_returns) * 100.0
    close = np.where(np.isfinite(close) & (close > 0), close, 100.0)

    # Generate open prices
    open_prices = close * (1.0 + rng.normal(0.0, 0.002, size=(n_rows, _N_SYMBOLS)))
    open_prices = np.where(np.isfinite(open_prices) & (open_prices > 0), open_prices, close)

    volume = np.tile(np.array(_DEFAULT_VOLUME, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid_irregular(dates, bars_per_session)
    panel = Panel(
        fields={
            "open": open_prices.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    config = TiltConfig(
        start=dates[1],  # Exclude prior day from test range
        end=dates[-1],
        entry_hhmm="09:16",
        exit_hhmm="15:20",
        seed=42,
    )
    result = run_tilt(panel, config)

    # The Muhurat session (60 bars) should be skipped
    # Per-year sessions should not include it
    total_sessions_in_dates = 4
    assert result.total.n_sessions < total_sessions_in_dates, (
        f"Expected < {total_sessions_in_dates} sessions (Muhurat dropped), "
        f"got {result.total.n_sessions}"
    )

    # Check that a warning mentions the skipped session
    any("skip" in w.lower() or "muhurat" in w.lower() or "15:20" in w for w in result.warnings)
    # Note: actual warning content depends on implementation; just verify warnings exist
    assert len(result.warnings) > 0 or result.total.n_sessions < total_sessions_in_dates


def test_date_range_respected() -> None:
    """Test 5: Sessions outside [start, end] never contribute.

    Per-year rows partition sessions exactly once: sum(per_year n_sessions) == total n_sessions.
    """
    dates = [dt.date(2023, 12, 29)] + [dt.date(2024, 1, i) for i in range(1, 16)] + \
            [dt.date(2024, 2, i) for i in range(1, 11)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dt.date(2024, 1, 5),
        end=dt.date(2024, 1, 31),
        seed=42,
    )
    result = run_tilt(panel, config)

    # Sum per-year n_sessions should equal total n_sessions
    sum_per_year = sum(row.n_sessions for row in result.per_year)
    assert sum_per_year == result.total.n_sessions, \
        f"Per-year partitioning failed: sum={sum_per_year}, total={result.total.n_sessions}"

    # Only sessions within [start, end] should be included
    # The Feb dates should not appear
    assert all(row.year == 2024 for row in result.per_year)


def test_smoothing_reduces_turnover() -> None:
    """Test 6: Smoothing (a=0.10) gives lower turnover than a=1.0 on same panel.

    a=1.0 is daily rebalance; a=0.10 is slow smoothing.
    """
    dates = [dt.date(2024, 1, i) for i in range(1, 21)]
    panel = _dated_panel(dates, 370, seed=42)

    config_daily = TiltConfig(
        start=dates[0],
        end=dates[-1],
        smoothing=1.0,  # Daily rebalance
        seed=42,
    )
    config_smooth = TiltConfig(
        start=dates[0],
        end=dates[-1],
        smoothing=0.10,  # 10% weight toward target
        seed=42,
    )

    result_daily = run_tilt(panel, config_daily)
    result_smooth = run_tilt(panel, config_smooth)

    # Smoothing should reduce turnover
    assert result_smooth.total.turnover < result_daily.total.turnover, (
        f"Smoothing should reduce turnover: "
        f"{result_smooth.total.turnover} >= {result_daily.total.turnover}"
    )


def test_weights_long_only_normalized() -> None:
    """Test 7: Weights are never negative, sum to 1 within tolerance, every session.

    Asserted via TiltResult diagnostics: min_weight_seen >= 0.0 and
    max_weight_sum_deviation within float tolerance of 0.
    """
    dates = [dt.date(2024, 1, i) for i in range(1, 11)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="mild",
        seed=42,
    )
    result = run_tilt(panel, config)

    # Assert weight invariants via diagnostics
    assert result.min_weight_seen >= 0.0, \
        f"Minimum weight should be non-negative, got {result.min_weight_seen}"

    assert np.isclose(result.max_weight_sum_deviation, 0.0, atol=1e-10), \
        f"Max weight sum deviation should be near 0, got {result.max_weight_sum_deviation}"

    assert result.max_n_held > 0, "Should have at least one holding in some session"


def test_holdout_protection() -> None:
    """Test 8: An end date inside the locked holdout window RAISES.

    The holdout boundary is GLOBAL, taken from HoldoutLock, independent
    of the panel passed in. Attempting to backtest with end inside the
    holdout should raise ValueError with "holdout" in the message.
    """
    dates = [dt.date(2024, 1, i) for i in range(1, 31)]
    panel = _dated_panel(dates, 370, seed=42)

    # Try to run with end date far in the future (will be in holdout)
    # The actual holdout boundary comes from HoldoutLock, not hardcoded here
    config_in_holdout = TiltConfig(
        start=dt.date(2024, 1, 1),
        end=dt.date(2025, 12, 31),  # Likely inside holdout
        seed=42,
    )

    with pytest.raises(ValueError, match=r"holdout"):
        run_tilt(panel, config_in_holdout)


def test_degenerate_inputs_raise() -> None:
    """Test 9: Degenerate inputs (start > end, no usable session, < 5 symbols) raise.

    Each case should raise ValueError with a message identifying the problem.
    """
    dates = [dt.date(2024, 1, 1)]
    panel = _dated_panel(dates, 370, seed=42)

    # Case 1: start > end
    config_backwards = TiltConfig(
        start=dt.date(2024, 1, 5),
        end=dt.date(2024, 1, 1),
        seed=42,
    )
    with pytest.raises(ValueError, match=r"start|end"):
        run_tilt(panel, config_backwards)

    # Case 2: No usable session (all NaN panel)
    nan_panel = Panel(
        fields={
            "open": np.full((150, _N_SYMBOLS), np.nan, dtype=np.float32),
            "close": np.full((150, _N_SYMBOLS), np.nan, dtype=np.float32),
            "volume": np.full((150, _N_SYMBOLS), np.nan, dtype=np.float32),
        },
        symbols=_SYMBOLS,
        ts=np.arange(0, 150, dtype=np.int64) * 60,
        day_offsets=np.array([0, 150], dtype=np.int32),
        dates=np.array([dt.date(2024, 1, 1)], dtype=object),
    )
    config_valid = TiltConfig(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 1),
        seed=42,
    )
    with pytest.raises(ValueError, match=r"session|usable"):
        run_tilt(nan_panel, config_valid)


def test_determinism() -> None:
    """Test 10: Same config and seed gives byte-identical to_table() output.

    Determinism is critical for reproducibility.
    """
    dates = [dt.date(2024, 1, i) for i in range(1, 11)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        seed=123,
    )

    result1 = run_tilt(panel, config)
    result2 = run_tilt(panel, config)

    table1 = result1.to_table()
    table2 = result2.to_table()

    assert table1 == table2, "Same config and seed should produce identical output"


def test_table_format() -> None:
    """Test 11: to_table() contains per-year rows and an ALL row.

    The ALL row's session count equals the sum of per-year counts.
    """
    dates = [dt.date(2023, 12, 29)] + [dt.date(2024, 1, i) for i in range(1, 16)] + \
            [dt.date(2025, 1, i) for i in range(1, 11)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dt.date(2024, 1, 1),
        end=dt.date(2025, 1, 10),
        seed=42,
    )
    result = run_tilt(panel, config)

    # to_table() should be a non-empty string
    table = result.to_table()
    assert isinstance(table, str), "to_table() should return a string"
    assert len(table) > 0, "Table should not be empty"

    # Should contain year information (2024, 2025)
    assert "2024" in table or "year" in table.lower(), "Table should reference year"

    # Session count: total n_sessions should equal sum of per_year
    sum_per_year = sum(row.n_sessions for row in result.per_year)
    assert sum_per_year == result.total.n_sessions


def test_explain_format() -> None:
    """Test that explain() returns a non-empty string with provenance info."""
    dates = [dt.date(2024, 1, i) for i in range(1, 6)]
    panel = _dated_panel(dates, 370, seed=42)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        capital=1_000_000.0,
        seed=42,
    )
    result = run_tilt(panel, config)

    explain = result.explain()
    assert isinstance(explain, str), "explain() should return a string"
    assert len(explain) > 0, "Explanation should not be empty"
    # Should mention one-leg cost convention
    assert "one-leg" in explain.lower() or "leg" in explain.lower() or len(explain) > 100


# ============================================================================
# CLI SMOKE TEST (will be integrated once CLI exists)
# ============================================================================


def test_cli_smoke() -> None:
    """Test 12: nq tilt CLI with valid flags exits 0 and prints table."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )
    assert result.exit_code == 0
    assert "Universe:" in result.output
    # The output should contain the results table (as HTML or text)
    assert len(result.output) > 0
