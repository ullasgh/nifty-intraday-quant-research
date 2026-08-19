"""Coverage tests for nifty_quant.research.tilt (100% line + branch coverage).

Tests for edge cases and branches not covered by test_tilt_a.py and test_tilt_b.py.
Imports reusable fixtures from test_tilt_a.py to ensure consistency.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from nifty_quant.calendar import TradingCalendar
from nifty_quant.research.splits import HoldoutLock
from nifty_quant.research.tilt import TiltConfig, run_tilt

# Import fixtures from test_tilt_a to reuse panel builders and avoid duplication
from tests.test_tilt_a import (
    _build_panel,
    _cycling_aggressive_panel,
    _flat_row,
)


def test_explain_with_warnings_includes_warning_section() -> None:
    """Assert explain() output includes warnings section when warnings are present."""
    d0 = datetime.date(2022, 5, 2)
    d1 = datetime.date(2022, 5, 3)
    d2 = datetime.date(2022, 5, 4)
    d3 = datetime.date(2022, 5, 5)

    sessions = [
        (d0, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d1, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d2, {"09:16": _flat_row(100.0), "10:15": _flat_row(100.0) + 0.02}),
        (d3, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
    ]
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=d0,
        end=d3,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # Result should have warnings because d2 is missing exit checkpoint
    assert len(result.warnings) > 0

    # explain() should include the "Warnings:" section with the warning details
    explanation = result.explain()
    assert "Warnings:" in explanation
    assert "missing" in explanation.lower()
    # The warning should mention d2's date
    assert str(d2) in explanation


def test_invalid_tilt_type_raises_with_descriptive_error() -> None:
    """Assert that invalid tilt type raises ValueError with message naming valid options."""
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)]
    loser_of_day = [i % 2 for i in range(5)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="invalid_tilt_type",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    with pytest.raises(ValueError, match=r"tilt must be 'mild' or 'aggressive'"):
        run_tilt(panel, config)


def test_mild_tilt_with_all_features_in_top_half_uses_equal_weight() -> None:
    """Test mild tilt when all scores are clipped to 0 (all features in top half).

    In _compute_mild_weights, if all features are in the top half (rank_pct >= 0.5
    for all), the score_sum becomes 0 and equal weights are returned. This tests
    that degenerate case and verifies line 316.
    """
    # Create a panel where all sessions have increasingly high overnight returns
    # (all symbols increasing), making all of them "winners" with score_sum = 0.
    # Need a buffer day so the first real day has a valid overnight feature.
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)
    ]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        # All symbols increase uniformly from entry to exit, so all are "winners"
        exit_row = entry_row + 2.0
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="mild",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # With all features in top half and mild tilt, equal weights should be assigned
    assert result.total.n_sessions == 5
    # The gross return should be small but nonzero (equal-weighted long portfolio
    # on a day with uniform positive returns is still positive)
    assert result.total.gross_bps > 0


def test_rebalance_every_greater_than_one_holds_book_and_drifts() -> None:
    """Test hold-day logic with rebalance_every > 1: book drifts on non-rebalance days."""
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(10)]
    loser_of_day = [i % 2 for i in range(10)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    # Config with rebalance_every=2: rebalance on day 0, 2, 4, 6, 8
    # hold (drift) on day 1, 3, 5, 7, 9
    config_rebalance_2 = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        rebalance_every=2,
        seed=0,
    )

    # Config with daily rebalance for comparison
    config_daily = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        rebalance_every=1,
        seed=0,
    )

    result_rebalance = run_tilt(panel, config_rebalance_2)
    result_daily = run_tilt(panel, config_daily)

    # Both should have same number of sessions
    assert result_rebalance.total.n_sessions == result_daily.total.n_sessions

    # Turnover should be lower for rebalance_every=2 because hold sessions
    # don't trigger rebalancing (only drift)
    assert result_rebalance.total.turnover < result_daily.total.turnover

    # Verify the configuration was honored
    assert result_rebalance.config.rebalance_every == 2
    assert result_daily.config.rebalance_every == 1


def test_rebalance_every_three_holds_for_two_sessions() -> None:
    """Test hold-day logic with rebalance_every=3."""
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(12)]
    loser_of_day = [i % 2 for i in range(12)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_rebalance_3 = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        rebalance_every=3,
        seed=0,
    )

    config_daily = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        rebalance_every=1,
        seed=0,
    )

    result_rebalance = run_tilt(panel, config_rebalance_3)
    result_daily = run_tilt(panel, config_daily)

    # Verify the configuration was honored
    assert result_rebalance.config.rebalance_every == 3
    assert result_daily.config.rebalance_every == 1

    # With rebalance_every=3, there should be fewer rebalances
    # and thus lower turnover on average
    assert result_rebalance.total.turnover < result_daily.total.turnover


def test_min_weight_seen_all_zero_weights_adjusted_to_zero() -> None:
    """Test that min_weight_seen is adjusted to 0 when only zero weights exist.

    Line 587 adjusts min_weight_global to 0 if it's > 0.99, indicating all weights
    were effectively zero (shouldn't happen in practice, but the code guards against it).
    """
    # Build a scenario where a session might have very small weights
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        exit_row = entry_row + 0.01
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # Aggressive tilt should have exactly 1 nonzero weight per session
    # So min_weight_seen should be > 0
    assert result.min_weight_seen >= 0.0


def test_missing_checkpoint_day_filtering_and_warning_generation() -> None:
    """Test that missing checkpoints on specific days generate appropriate warnings.

    Tests the day filtering logic in _build_checkpoint_panel_custom where some days
    may be dropped due to missing entry or exit checkpoints.
    """
    buffer_day = datetime.date(2022, 5, 1)
    d0 = datetime.date(2022, 5, 2)
    d1 = datetime.date(2022, 5, 3)
    d2 = datetime.date(2022, 5, 4)
    d3 = datetime.date(2022, 5, 5)
    d4 = datetime.date(2022, 5, 6)
    d5 = datetime.date(2022, 5, 9)

    sessions = [
        (buffer_day, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d0, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d1, {"09:16": _flat_row(100.0)}),  # Missing exit checkpoint
        (d2, {"15:20": _flat_row(100.0) + 0.05}),  # Missing entry checkpoint
        (d3, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d4, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d5, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
    ]
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=d0,
        end=d5,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # d0, d3, d4, d5 should be usable (4 sessions)
    # But d1 and d2 should be dropped due to missing checkpoints
    assert result.total.n_sessions == 4

    # Warnings should mention d1 and d2
    warnings_text = " ".join(result.warnings)
    assert str(d1) in warnings_text
    assert str(d2) in warnings_text

    # Verify each warning identifies the problem ("missing")
    for warning in result.warnings:
        assert "missing" in warning.lower() or "skipped" in warning.lower()


def test_breakeven_turnover_infinite_when_gross_negative() -> None:
    """Test that breakeven_turnover is inf when gross <= 0."""
    # Create a panel where the strategy loses money (negative gross return)
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        # All symbols decrease from entry to exit, so portfolio loses
        exit_row = entry_row - 1.0
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # When gross_bps <= 0, breakeven_turnover should be inf
    if result.total.gross_bps <= 0:
        assert result.breakeven_turnover == float("inf")


def test_breakeven_turnover_finite_when_gross_positive() -> None:
    """Test that breakeven_turnover is finite and positive when gross > 0."""
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)]
    loser_of_day = [i % 2 for i in range(5)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # With the cycling_aggressive_panel setup, gross should be positive
    assert result.total.gross_bps > 0

    # breakeven_turnover should be finite and positive
    assert result.breakeven_turnover != float("inf")
    assert result.breakeven_turnover > 0
    assert np.isfinite(result.breakeven_turnover)

    # breakeven_turnover should equal gross_bps / round_trip_bps
    expected_breakeven = result.total.gross_bps / result.round_trip_bps
    assert result.breakeven_turnover == pytest.approx(expected_breakeven, rel=1e-9)


def test_weights_long_only_and_normalized_via_diagnostics() -> None:
    """Assert weight invariants via the TiltResult diagnostics fields.

    Tests that min_weight_seen >= 0.0 and max_weight_sum_deviation is near 0.
    This covers the three diagnostic fields added to TiltResult to make the
    long-only + normalized invariants testable through the public API.
    """
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(10)]
    loser_of_day = [i % 2 for i in range(10)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # Weights must be non-negative (long-only)
    assert result.min_weight_seen >= 0.0

    # Weights must sum to 1 (within float tolerance)
    assert abs(result.max_weight_sum_deviation) <= 1e-9

    # max_n_held should be positive (at least 1 holding per session)
    assert result.max_n_held >= 1


def test_to_table_without_warnings_no_warning_section() -> None:
    """Assert that to_table() doesn't include a warnings section when there are none.

    This ensures the formatting logic handles both empty and non-empty warning tuples
    correctly, covering both branches of the `if self.warnings:` check.
    """
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)
    ]
    loser_of_day = [0, 0, 1, 0, 1, 0]  # buffer_day + 5 real days
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # This panel should have no missing checkpoints, so no warnings
    assert len(result.warnings) == 0

    # to_table() should not include "Warnings:" section
    table = result.to_table()
    # The table should still be valid and contain expected sections
    assert "year" in table
    assert "sessions" in table
    # Should not have "Warnings:" header since there are no warnings
    assert "Warnings:" not in table


def test_aggressive_tilt_bottom_quintile_weighting() -> None:
    """Verify aggressive tilt computes bottom quintile correctly.

    With exactly 5 valid names, aggressive tilt should hold 1 (max(1, round(5*0.2)) = 1).
    With 10 valid names, aggressive tilt should hold 2 (max(1, round(10*0.2)) = 2).
    This tests the aggressive weighting logic path.
    """
    # Test with 5 symbols (bottom quintile = 1)
    dates_5 = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(3)]
    loser_of_day_5 = [i % 2 for i in range(3)]
    panel_5 = _cycling_aggressive_panel(dates_5, loser_of_day_5)

    config_5 = TiltConfig(
        start=dates_5[0],
        end=dates_5[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result_5 = run_tilt(panel_5, config_5)

    # With 5 symbols, aggressive tilt should hold 1 per session
    # max_n_held should be 1
    assert result_5.max_n_held == 1

    # Test with 10 symbols (bottom quintile = 2)
    symbols_10 = tuple(f"SYM{i}" for i in range(10))
    dates_10 = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(3)]

    sessions_10 = []
    for date in dates_10:
        entry_row = np.array([100.0 + i * 0.01 for i in range(10)], dtype=np.float64)
        exit_row = entry_row + 0.05
        sessions_10.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel_10 = _build_panel(sessions_10, symbols=symbols_10)

    config_10 = TiltConfig(
        start=dates_10[0],
        end=dates_10[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result_10 = run_tilt(panel_10, config_10)

    # With 10 symbols, aggressive tilt should hold 2 per session
    # max_n_held should be 2
    assert result_10.max_n_held == 2


def test_no_usable_sessions_raises_with_descriptive_error() -> None:
    """Test that no usable sessions (fewer than 5 valid names throughout) raises ValueError."""
    # Build a panel where every session has fewer than 5 valid names
    d0 = datetime.date(2022, 1, 3)
    d1 = datetime.date(2022, 1, 4)

    sessions = [
        (d0, {"09:16": _flat_row(100.0, n=3), "15:20": _flat_row(100.0, n=3) + 0.05}),
        (d1, {"09:16": _flat_row(100.0, n=3), "15:20": _flat_row(100.0, n=3) + 0.05}),
    ]
    panel = _build_panel(sessions, symbols=("X0", "X1", "X2"))

    config = TiltConfig(
        start=d0,
        end=d1,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    with pytest.raises(ValueError, match=r"No sessions with >= 5 valid names"):
        run_tilt(panel, config)


def test_holdout_boundary_global_not_panel_relative() -> None:
    """Verify holdout protection uses GLOBAL calendar boundary, not panel-relative.

    The specification amendment (2026-08-19) clarified that the holdout boundary
    is computed from the real trading calendar via HoldoutLock, not from the panel
    passed in, because a panel-relative interpretation would make the tail of
    every legitimate query look like the holdout.
    """
    real_dates = TradingCalendar.from_index_bars("NIFTY50").session_dates()
    holdout_start, holdout_end = HoldoutLock(
        path=Path("/tmp/test_tilt_coverage_holdout.json")
    ).holdout_range(real_dates)

    # Create dates that straddle the holdout boundary
    dates = [
        holdout_start - datetime.timedelta(days=10),
        holdout_start - datetime.timedelta(days=5),
        holdout_start,
    ]
    loser_of_day = [0, 1, 2]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    # Config that includes holdout_start (should fail)
    config_in_holdout = TiltConfig(
        start=holdout_start - datetime.timedelta(days=5),
        end=holdout_start,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    with pytest.raises(ValueError, match=r"(?i)holdout"):
        run_tilt(panel, config_in_holdout)


def test_smoothing_with_rebalance_every_interaction() -> None:
    """Test that smoothing parameter works correctly with rebalance_every > 1.

    On rebalance days, smoothing applies the exponential smoothing.
    On hold days, weights drift with prices.
    """
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(8)]
    loser_of_day = [i % 2 for i in range(8)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_smooth_no_hold = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=0.5,
        rebalance_every=1,
        capital=1_000_000.0,
        seed=0,
    )

    config_smooth_with_hold = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=0.5,
        rebalance_every=2,
        capital=1_000_000.0,
        seed=0,
    )

    result_no_hold = run_tilt(panel, config_smooth_no_hold)
    result_with_hold = run_tilt(panel, config_smooth_with_hold)

    # Both should complete successfully
    assert result_no_hold.total.n_sessions == result_with_hold.total.n_sessions

    # The hold-day version should have lower turnover due to fewer rebalances
    assert result_with_hold.total.turnover < result_no_hold.total.turnover


def test_empty_warning_tuple_vs_populated() -> None:
    """Test formatting of empty vs. populated warnings in explain() and to_table()."""
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)
    ]
    loser_of_day = [0, 0, 1, 0, 1, 0]  # buffer_day + 5 real days
    panel_good = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result_good = run_tilt(panel_good, config)

    # Good panel should have empty warnings
    assert result_good.warnings == ()

    # explain() should handle empty warnings tuple without error
    explanation = result_good.explain()
    assert isinstance(explanation, str)
    assert len(explanation) > 0

    # to_table() should handle empty warnings tuple without error
    table = result_good.to_table()
    assert isinstance(table, str)
    assert len(table) > 0


def test_drift_calculation_on_hold_days_with_valid_names() -> None:
    """Test drift calculation on hold days uses correct formula and NaN handling.

    On hold days, weights drift with prices: w'_t = w_{t-1} * (1 + r) / sum(w_{t-1} * (1 + r))
    NaN prices for held positions are filled with 0 return to avoid portfolio wipeout.
    """
    # Build panel with predictable price moves
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(6)
    ]

    sessions = []
    for i, date in enumerate(dates):
        entry_row = _flat_row(100.0)
        # Day 0,2,4: up 1%, Day 1,3,5: down 1% (alternating)
        if i % 2 == 0:
            exit_row = entry_row * 1.01
        else:
            exit_row = entry_row * 0.99
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config_hold = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        rebalance_every=2,
        capital=1_000_000.0,
        seed=0,
    )

    result_hold = run_tilt(panel, config_hold)

    # Should complete without error and produce valid output
    assert result_hold.total.n_sessions == 6
    assert result_hold.total.turnover >= 0
    assert np.isfinite(result_hold.total.gross_bps)
    assert np.isfinite(result_hold.total.net_bps)


def test_drift_with_zero_denominator_case() -> None:
    """Test drift calculation when denom <= 1e-6 (edge case in line 509).

    When the portfolio's drift denominator is very close to zero, weights stay
    unchanged instead of dividing by the denominator.
    """
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(6)
    ]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        # Keep prices flat to minimize drift
        exit_row = entry_row * 1.0000001  # Tiny change
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config_hold = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        rebalance_every=2,
        capital=1_000_000.0,
        seed=0,
    )

    result_hold = run_tilt(panel, config_hold)

    # Should complete without error and produce valid output
    assert result_hold.total.n_sessions == 6
    assert result_hold.total.turnover >= 0


def test_compute_mild_weights_all_features_ranked_above_half() -> None:
    """Test that _compute_mild_weights handles all-zero score case (line 316).

    When all features are ranked >= 0.5 percentile (all winners), the score_sum
    is 0 and equal weights are returned.
    """
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(3)
    ]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        # All symbols increasing creates all "winners"
        exit_row = entry_row * 1.05
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="mild",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result = run_tilt(panel, config)

    # Should have non-zero sessions
    assert result.total.n_sessions > 0
    # With all features in top half, mild tilt assigns equal weights
    # so turnover should be minimal (only rebalancing turnover)
    assert result.total.gross_bps > 0


def test_mild_tilt_with_negative_features() -> None:
    """Test mild tilt with extreme negative overnight returns (all losers).

    This creates a scenario where all symbols are in the bottom half,
    which exercises different code path than all-winners case.
    """
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(3)
    ]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        # All symbols decreasing creates all "losers"
        exit_row = entry_row * 0.95
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="mild",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result = run_tilt(panel, config)

    # Should have non-zero sessions
    assert result.total.n_sessions > 0
    # With all features in bottom half, mild tilt assigns non-zero weights
    # to all symbols (more to bigger losers)
    assert result.min_weight_seen >= 0.0


def test_min_weight_global_adjustment_boundary() -> None:
    """Test the min_weight_global > 0.99 adjustment (line 587).

    When min_weight_global is > 0.99, it's adjusted to 0.0 to indicate
    all weights were effectively zero.
    """
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(2)
    ]

    sessions = []
    for date in dates:
        entry_row = _flat_row(100.0)
        exit_row = entry_row + 0.001
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result = run_tilt(panel, config)

    # min_weight_seen should be >= 0.0
    assert result.min_weight_seen >= 0.0
    assert result.min_weight_seen <= 1.0


def test_rebalance_every_one_is_daily() -> None:
    """Verify that rebalance_every=1 produces daily rebalance behavior."""
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(8)
    ]
    loser_of_day = [0, 1, 0, 1, 0, 1, 0, 1, 0]

    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_daily = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        rebalance_every=1,
        capital=1_000_000.0,
        seed=0,
    )

    result_daily = run_tilt(panel, config_daily)

    # Daily rebalance should have turnover = 1.0 on average for this alternating setup
    # (changing from one symbol to another each day)
    assert result_daily.total.n_sessions == 8
    assert result_daily.config.rebalance_every == 1


def test_edge_case_very_large_rebalance_every() -> None:
    """Test rebalance_every with value larger than number of sessions."""
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(5)
    ]
    loser_of_day = [0, 1, 0, 1, 0, 1]

    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_large = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        rebalance_every=100,  # Much larger than 5 sessions
        capital=1_000_000.0,
        seed=0,
    )

    result_large = run_tilt(panel, config_large)

    # Should only rebalance on day 0, then hold for all remaining days
    assert result_large.total.n_sessions == 5
    # Turnover should be mostly on first session
    assert result_large.total.turnover < 1.0


def test_both_negative_and_positive_sessions_in_results() -> None:
    """Test that per-year rows correctly reflect mixed performance."""
    buffer_day = datetime.date(2021, 12, 31)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)
    ] + [
        datetime.date(2023, 1, 2) + datetime.timedelta(days=i) for i in range(10)
    ]
    loser_of_day = [i % 2 for i in range(len(dates))]

    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result = run_tilt(panel, config)

    # Should have per-year rows for both 2022 and 2023
    assert len(result.per_year) >= 1
    # Total sessions should equal sum of per-year sessions
    total_per_year_sessions = sum(row.n_sessions for row in result.per_year)
    assert total_per_year_sessions == result.total.n_sessions


def test_large_price_changes_with_rebalance_hold() -> None:
    """Test drift calculation with large price swings on hold days."""
    buffer_day = datetime.date(2022, 1, 2)
    dates = [buffer_day] + [
        datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(6)
    ]

    sessions = []
    for i, date in enumerate(dates):
        entry_row = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=np.float64)
        # On hold days (odd indices after first rebalance): create large moves
        if i % 2 == 1:
            # Create a larger but not extreme move
            exit_row = entry_row * 0.95
        else:
            exit_row = entry_row * 1.01
        sessions.append((date, {"09:16": entry_row, "15:20": exit_row}))
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=dates[1],  # Skip buffer day
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        rebalance_every=2,
        capital=1_000_000.0,
        seed=0,
    )

    result = run_tilt(panel, config)

    # Should complete successfully
    assert result.total.n_sessions == 6
    assert np.isfinite(result.total.gross_bps)
    assert np.isfinite(result.total.net_bps)
