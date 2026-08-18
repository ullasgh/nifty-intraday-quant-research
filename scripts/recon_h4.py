#!/usr/bin/env python3
"""Reproduce H4: volatility compression expanding into continuation.

Signal at 10:00: morning range (09:16-10:00) divided by a strictly PRIOR 20-session
mean range, multiplied by sign of the morning move.

Trade: enter at 10:00 close, exit at 15:20 close, cross-sectionally demeaned.

Three pre-registered thresholds: expand > 1.0, > 1.5, > 2.0.

Window: 2018-01-01 .. 2025-07-31 (last 12 months held out).
Universe: all_equity (145 symbols after survivorship).
"""

from __future__ import annotations

import datetime
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from nifty_quant.data.panel import Panel

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.universe.static import load_universe, survivorship_report
from nifty_quant.execution.costs import NSEIntradayEquityCosts


def _compute_morning_range_mean(
    panel: Panel,
    open_minute: int,
    morning_minute: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the morning range (high - low) for each session, and a 20-session prior mean.

    Returns:
        (morning_ranges, prior_mean_by_session): both shape (n_symbols, n_sessions)
        NaN for sessions that lack either 09:16 or 10:00, or if prior mean cannot be
        computed (first 20 sessions).
    """
    n_symbols = panel.n_symbols()
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1

    panel_open = panel.field("open").astype(np.float64)
    panel_high = panel.field("high").astype(np.float64)
    panel_low = panel.field("low").astype(np.float64)
    minute_of_day = panel.minute_of_day()

    # Allocate output: (n_symbols, n_sessions) for each session's morning range
    morning_ranges = np.full((n_symbols, n_sessions), np.nan, dtype=np.float64)
    prior_mean = np.full((n_symbols, n_sessions), np.nan, dtype=np.float64)

    for i in range(n_sessions):
        curr_start = day_offsets[i]
        curr_end = day_offsets[i + 1]

        session_minutes = minute_of_day[curr_start:curr_end]
        open_rows = np.where(session_minutes == open_minute)[0]
        morning_rows = np.where(session_minutes == morning_minute)[0]

        # Session must have both 09:16 and 10:00
        if len(open_rows) == 0 or len(morning_rows) == 0:
            continue

        open_row = curr_start + open_rows[0]
        morning_row = curr_start + morning_rows[0]

        # Compute morning range: high[09:16:10:00] - low[09:16:10:00]
        # We compute range over the session's full session bars (all bars in [09:16, 10:00])
        session_high = panel_high[curr_start:curr_end, :].astype(np.float64)
        session_low = panel_low[curr_start:curr_end, :].astype(np.float64)

        # Find all rows in [09:16, 10:00] (inclusive)
        range_mask = (session_minutes >= open_minute) & (session_minutes <= morning_minute)
        range_rows = np.where(range_mask)[0]

        if len(range_rows) == 0:
            continue

        # High and low within the range window
        range_high = np.nanmax(session_high[range_rows, :], axis=0)
        range_low = np.nanmin(session_low[range_rows, :], axis=0)
        range_val = range_high - range_low

        morning_ranges[:, i] = range_val

        # Compute prior 20-session mean (sessions i-20:i, exclusive of i)
        if i >= 20:
            prior_ranges = morning_ranges[:, i - 20 : i]
            # For each symbol, compute mean of prior 20 sessions
            prior_mean[:, i] = np.nanmean(prior_ranges, axis=1)

    return morning_ranges, prior_mean


def _compute_morning_move_sign(
    panel: Panel,
    open_minute: int,
    morning_minute: int,
) -> np.ndarray:
    """Compute sign of morning move: sign(log(close[10:00] / open[09:16])).

    Returns:
        Array of shape (n_symbols, n_sessions), dtype float64.
        NaN for sessions lacking either checkpoint.
    """
    n_symbols = panel.n_symbols()
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1

    panel_open = panel.field("open").astype(np.float64)
    panel_close = panel.field("close").astype(np.float64)
    minute_of_day = panel.minute_of_day()

    move_sign = np.full((n_symbols, n_sessions), np.nan, dtype=np.float64)

    for i in range(n_sessions):
        curr_start = day_offsets[i]
        curr_end = day_offsets[i + 1]

        session_minutes = minute_of_day[curr_start:curr_end]
        open_rows = np.where(session_minutes == open_minute)[0]
        morning_rows = np.where(session_minutes == morning_minute)[0]

        if len(open_rows) == 0 or len(morning_rows) == 0:
            continue

        open_row = curr_start + open_rows[0]
        morning_row = curr_start + morning_rows[0]

        entry_open = panel_open[open_row, :].astype(np.float64)
        morning_close = panel_close[morning_row, :].astype(np.float64)

        valid = (
            np.isfinite(entry_open)
            & np.isfinite(morning_close)
            & (entry_open > 0)
            & (morning_close > 0)
        )

        log_ret = np.full(n_symbols, np.nan, dtype=np.float64)
        log_ret[valid] = np.log(morning_close[valid] / entry_open[valid])
        move_sign[:, i] = np.sign(log_ret)

    return move_sign


def _compute_intraday_returns(
    panel: Panel,
    morning_minute: int,
    exit_minute: int,
) -> np.ndarray:
    """Compute log returns from 10:00 close to 15:20 close, one per session.

    Returns:
        Array of shape (n_symbols, n_sessions), dtype float64.
        NaN for sessions lacking either checkpoint.
    """
    n_symbols = panel.n_symbols()
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1

    panel_close = panel.field("close").astype(np.float64)
    minute_of_day = panel.minute_of_day()

    intraday_ret = np.full((n_symbols, n_sessions), np.nan, dtype=np.float64)

    for i in range(n_sessions):
        curr_start = day_offsets[i]
        curr_end = day_offsets[i + 1]

        session_minutes = minute_of_day[curr_start:curr_end]
        morning_rows = np.where(session_minutes == morning_minute)[0]
        exit_rows = np.where(session_minutes == exit_minute)[0]

        if len(morning_rows) == 0 or len(exit_rows) == 0:
            continue

        morning_row = curr_start + morning_rows[0]
        exit_row = curr_start + exit_rows[0]

        morning_close = panel_close[morning_row, :].astype(np.float64)
        exit_close = panel_close[exit_row, :].astype(np.float64)

        valid = (
            np.isfinite(morning_close)
            & np.isfinite(exit_close)
            & (morning_close > 0)
            & (exit_close > 0)
        )

        ret = np.full(n_symbols, np.nan, dtype=np.float64)
        ret[valid] = np.log(exit_close[valid] / morning_close[valid])
        intraday_ret[:, i] = ret

    return intraday_ret


def _cross_sectional_demean(returns: np.ndarray) -> np.ndarray:
    """Cross-sectionally demean returns within each session (column-wise).

    For each column (session), subtract the mean of finite values.
    Returns a copy.
    """
    demeaned = returns.copy()
    for j in range(returns.shape[1]):
        col = returns[:, j]
        finite_mask = np.isfinite(col)
        if np.any(finite_mask):
            col_mean = np.mean(col[finite_mask])
            demeaned[finite_mask, j] = col[finite_mask] - col_mean
    return demeaned


def run_h4_recon(
    start: datetime.date,
    end: datetime.date,
    universe_name: str = "all_equity",
) -> None:
    """Run the H4 reconnaissance.

    Parameters:
    -----------
    start : datetime.date
        Start date (inclusive).
    end : datetime.date
        End date (inclusive).
    universe_name : str
        Universe name (default "all_equity").
    """
    # Load universe
    universe = load_universe(universe_name)
    print(f"Universe: {universe_name}, {len(universe.symbols)} symbols")

    # Load panel
    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=universe.symbols,
        start=start,
        end=end,
    )
    print(f"Loading panel {start} .. {end} ...")
    panel = load_panel(spec)
    print(f"Panel: {panel.n_rows()} rows, {panel.n_symbols()} symbols, {len(panel.dates)} sessions")

    # Time labels (in minutes from midnight)
    open_minute = 9 * 60 + 16  # 09:16
    morning_minute = 10 * 60  # 10:00
    exit_minute = 15 * 60 + 20  # 15:20

    print(f"Computing morning ranges and prior means ...")
    morning_ranges, prior_mean = _compute_morning_range_mean(
        panel, open_minute, morning_minute
    )

    print(f"Computing morning move signs ...")
    move_signs = _compute_morning_move_sign(panel, open_minute, morning_minute)

    print(f"Computing intraday returns ...")
    intraday_rets = _compute_intraday_returns(panel, morning_minute, exit_minute)

    # Compute expansion signal: (morning_range / prior_20_mean) * sign(move)
    expansion_ratio = morning_ranges / prior_mean * move_signs

    # Cross-sectionally demean the intraday returns
    demeaned_rets = _cross_sectional_demean(intraday_rets)

    # Cost hurdle (per-round-trip)
    cost_model = NSEIntradayEquityCosts()
    cost_hurdle_bps = cost_model.round_trip_bps(1e5)
    cost_gate_bps = 2 * cost_hurdle_bps  # doubled for survival criterion

    print(f"Cost hurdle (1x): {cost_hurdle_bps:.5f} bps")
    print(f"Cost gate (2x): {cost_gate_bps:.5f} bps")

    # Run test for each threshold
    thresholds = [1.0, 1.5, 2.0]
    results = []

    for thresh in thresholds:
        # Mask: expansion_ratio > thresh
        mask = expansion_ratio > thresh
        demeaned_rets_masked = demeaned_rets.copy()
        demeaned_rets_masked[~mask] = np.nan

        # Flatten to compute statistics
        rets_flat = demeaned_rets_masked.ravel()
        rets_finite = rets_flat[np.isfinite(rets_flat)]

        # Count observations and sessions
        n_obs = len(rets_finite)
        n_sessions_hit = np.sum(np.nansum(np.isfinite(demeaned_rets_masked), axis=0) > 0)

        if n_obs == 0:
            print(f"Threshold {thresh}: no observations")
            results.append({
                "threshold": thresh,
                "edge_bps": np.nan,
                "t_stat": np.nan,
                "n_sessions": 0,
                "n_obs": 0,
            })
            continue

        # Mean return and t-statistic
        mean_ret = np.mean(rets_finite)
        std_ret = np.std(rets_finite, ddof=1)
        t_stat = mean_ret / (std_ret / np.sqrt(n_obs)) if std_ret > 0 else 0.0

        # Convert to bps
        mean_ret_bps = mean_ret * 10000

        results.append({
            "threshold": thresh,
            "edge_bps": mean_ret_bps,
            "t_stat": t_stat,
            "n_sessions": n_sessions_hit,
            "n_obs": n_obs,
        })

    # Summary statistics
    n_total_obs = np.sum(np.isfinite(demeaned_rets))
    n_total_sessions = np.sum(np.nansum(np.isfinite(demeaned_rets), axis=0) > 0)

    print(f"\n{'='*80}")
    print(f"H4 Reconnaissance Results")
    print(f"{'='*80}")
    print(f"Universe: {universe_name}")
    print(f"Symbols in result: {panel.n_symbols()}")
    print(f"Window: {start} .. {end}")
    print(f"Total observations (all sessions): {n_total_obs}")
    print(f"Total sessions with any valid observation: {n_total_sessions}")
    print()

    # Print table header
    print(f"{'threshold':<20} {'edge bps':>12} {'t':>8} {'n_sessions':>12} {'n_obs':>12}")
    print(f"{'-'*75}")

    for res in results:
        print(
            f"expand > {res['threshold']:<3.1f}      {res['edge_bps']:>12.2f} "
            f"{res['t_stat']:>8.2f} {res['n_sessions']:>12} {res['n_obs']:>12}"
        )

    # Yearly breakdown
    print()
    print("Yearly breakdown:")
    print(f"{'threshold':<20} {'2018':>12} {'2024':>12} {'2025':>12}")
    print(f"{'-'*60}")

    dates = panel.dates
    # Convert dates (numpy array of date objects) to years
    years_array = np.array([d.year for d in dates])

    for thresh in thresholds:
        mask = expansion_ratio > thresh
        demeaned_rets_masked = demeaned_rets.copy()
        demeaned_rets_masked[~mask] = np.nan

        year_results = {}
        for year in [2018, 2024, 2025]:
            year_mask = (years_array == year)
            year_cols = np.where(year_mask)[0]

            rets_year = demeaned_rets_masked[:, year_cols].ravel()
            rets_year_finite = rets_year[np.isfinite(rets_year)]

            if len(rets_year_finite) > 0:
                mean_ret_year = np.mean(rets_year_finite)
                year_results[year] = mean_ret_year * 10000
            else:
                year_results[year] = np.nan

        print(
            f"expand > {thresh:<3.1f}      {year_results.get(2018, np.nan):>12.1f} "
            f"{year_results.get(2024, np.nan):>12.1f} {year_results.get(2025, np.nan):>12.1f}"
        )

    # Survivorship report
    print()
    print("Survivorship report:")
    report = survivorship_report(universe, start, end)
    print(report.warning_line())


if __name__ == "__main__":
    # 2018-01-01 .. 2025-07-31 (last 12 months held out)
    START = datetime.date(2018, 1, 1)
    END = datetime.date(2025, 7, 31)

    try:
        run_h4_recon(START, END)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
