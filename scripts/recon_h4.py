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
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.universe.static import load_universe, survivorship_report


def _compute_morning_range_mean(
    panel: Panel,
    open_minute: int,
    morning_minute: int,
    enforce_min_prior_count: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the morning range (high - low) for each session, and a 20-session prior mean.

    Parameters:
    -----------
    enforce_min_prior_count : bool
        If True, enforce minimum of 20 non-NaN prior sessions per symbol.
        If False, allow any count of prior sessions (for reporting pre-enforcement).

    Returns:
        (morning_ranges, prior_mean_by_session): both shape (n_symbols, n_sessions)
        NaN for sessions that lack either 09:16 or 10:00, or if prior mean cannot be
        computed (first 20 sessions).
    """
    n_symbols = panel.n_symbols()
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1

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
        # Enforce minimum of 20 non-NaN prior sessions per symbol (the literal reading
        # of "strictly prior 20-session mean", not a tuned threshold).
        if i >= 20:
            prior_ranges = morning_ranges[:, i - 20 : i]
            # For each symbol, compute mean only if at least 20 non-NaN values exist
            n_prior = np.sum(np.isfinite(prior_ranges), axis=1)
            mean_prior = np.nanmean(prior_ranges, axis=1)
            if enforce_min_prior_count:
                prior_mean[:, i] = np.where(n_prior >= 20, mean_prior, np.nan)
            else:
                prior_mean[:, i] = mean_prior

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

    print("Computing morning ranges and prior means (with enforcement)...")
    morning_ranges, prior_mean = _compute_morning_range_mean(
        panel, open_minute, morning_minute, enforce_min_prior_count=True
    )

    print("Computing prior means WITHOUT enforcement (for comparison)...")
    _, prior_mean_no_enforce = _compute_morning_range_mean(
        panel, open_minute, morning_minute, enforce_min_prior_count=False
    )

    print("Computing morning move signs ...")
    move_signs = _compute_morning_move_sign(panel, open_minute, morning_minute)

    print("Computing intraday returns ...")
    intraday_rets = _compute_intraday_returns(panel, morning_minute, exit_minute)

    # Compute expansion signal: (morning_range / prior_20_mean) * sign(move)
    # NOTE: This is a LONG-ONLY, UP-MOVES-ONLY test by construction:
    #   expansion_ratio = ratio * sign(move) where sign is -1 / 0 / +1
    #   Mask = expansion_ratio > thresh (positive threshold)
    #   DOWN moves (sign=-1) give negative expansion_ratio, can never exceed positive thresh
    #   FLAT moves (sign=0) give zero expansion_ratio, can never exceed positive thresh
    # The reported edge is therefore the long leg's raw demeaned return, not sign*ret.
    expansion_ratio = morning_ranges / prior_mean * move_signs
    expansion_ratio_no_enforce = morning_ranges / prior_mean_no_enforce * move_signs

    # Cross-sectionally demean the intraday returns
    demeaned_rets = _cross_sectional_demean(intraday_rets)

    # Count symbol-sessions dropped due to one-sided constraint
    flat_move_mask = (move_signs == 0.0)
    down_move_mask = (move_signs < 0.0)
    total_flat_moves = np.sum(flat_move_mask)
    total_down_moves = np.sum(down_move_mask)
    print("Note: test is one-sided (long-only, up-moves only):")
    print(f"  Flat moves (sign=0) dropped: {total_flat_moves} symbol-sessions")
    print(f"  Down moves (sign<0) dropped: {total_down_moves} symbol-sessions")

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

    # Summary statistics: condition on signal availability (expansion_ratio valid AND finite)
    # WITH enforcement
    signal_mask = np.isfinite(expansion_ratio)
    signal_obs = demeaned_rets[signal_mask]
    n_total_obs = np.sum(np.isfinite(signal_obs))
    n_total_sessions = np.sum(np.sum(signal_mask, axis=0) > 0)
    symbols_with_signal = np.sum(np.sum(signal_mask, axis=1) > 0)

    # WITHOUT enforcement (for comparison)
    signal_mask_no_enforce = np.isfinite(expansion_ratio_no_enforce)
    signal_obs_no_enforce = demeaned_rets[signal_mask_no_enforce]
    n_total_obs_no_enforce = np.sum(np.isfinite(signal_obs_no_enforce))
    n_total_sessions_no_enforce = np.sum(np.sum(signal_mask_no_enforce, axis=0) > 0)
    symbols_with_signal_no_enforce = np.sum(np.sum(signal_mask_no_enforce, axis=1) > 0)

    print(f"\n{'='*80}")
    print("H4 Reconnaissance Results")
    print(f"{'='*80}")
    print(f"Universe: {universe_name}")
    print(f"Window: {start} .. {end}")
    print()
    print("BEFORE enforcing minimum 20-session prior count (defect 3 unenforced):")
    print(f"  Symbols with signal: {symbols_with_signal_no_enforce}")
    print(f"  Total observations: {n_total_obs_no_enforce}")
    print(f"  Total sessions: {n_total_sessions_no_enforce}")
    print()
    print("AFTER enforcing minimum 20-session prior count (defect 3 fixed):")
    print(f"  Symbols with signal: {symbols_with_signal}")
    print(f"  Total observations: {n_total_obs}")
    print(f"  Total sessions: {n_total_sessions}")
    print()

    # Print table header
    print(f"{'threshold':<20} {'edge bps':>12} {'t':>8} {'n_sessions':>12} {'n_obs':>12}")
    print(f"{'-'*75}")

    for res in results:
        print(
            f"expand > {res['threshold']:<3.1f}      {res['edge_bps']:>12.2f} "
            f"{res['t_stat']:>8.2f} {res['n_sessions']:>12} {res['n_obs']:>12}"
        )

    # Yearly breakdown (all 8 years)
    all_years = list(range(2018, 2026))
    dates = panel.dates
    # Convert dates (numpy array of date objects) to years
    years_array = np.array([d.year for d in dates])

    # BEFORE enforcement
    print()
    print("Yearly breakdown BEFORE enforcement (edge in bps):")
    header = "threshold".ljust(20)
    for year in all_years:
        header += f"{year:>10}"
    print(header)
    print(f"{'-'*(20 + 10*len(all_years))}")

    for thresh in thresholds:
        mask = expansion_ratio_no_enforce > thresh
        demeaned_rets_masked = demeaned_rets.copy()
        demeaned_rets_masked[~mask] = np.nan

        year_results = {}
        for year in all_years:
            year_mask = (years_array == year)
            year_cols = np.where(year_mask)[0]

            rets_year = demeaned_rets_masked[:, year_cols].ravel()
            rets_year_finite = rets_year[np.isfinite(rets_year)]

            if len(rets_year_finite) > 0:
                mean_ret_year = np.mean(rets_year_finite)
                year_results[year] = mean_ret_year * 10000
            else:
                year_results[year] = np.nan

        row = f"expand > {thresh:<3.1f}    "
        for year in all_years:
            val = year_results.get(year, np.nan)
            if np.isnan(val):
                row += f"{'nan':>10}"
            else:
                row += f"{val:>10.1f}"
        print(row)

    # AFTER enforcement
    print()
    print("Yearly breakdown AFTER enforcement (edge in bps):")
    header = "threshold".ljust(20)
    for year in all_years:
        header += f"{year:>10}"
    print(header)
    print(f"{'-'*(20 + 10*len(all_years))}")

    for thresh in thresholds:
        mask = expansion_ratio > thresh
        demeaned_rets_masked = demeaned_rets.copy()
        demeaned_rets_masked[~mask] = np.nan

        year_results = {}
        for year in all_years:
            year_mask = (years_array == year)
            year_cols = np.where(year_mask)[0]

            rets_year = demeaned_rets_masked[:, year_cols].ravel()
            rets_year_finite = rets_year[np.isfinite(rets_year)]

            if len(rets_year_finite) > 0:
                mean_ret_year = np.mean(rets_year_finite)
                year_results[year] = mean_ret_year * 10000
            else:
                year_results[year] = np.nan

        row = f"expand > {thresh:<3.1f}    "
        for year in all_years:
            val = year_results.get(year, np.nan)
            if np.isnan(val):
                row += f"{'nan':>10}"
            else:
                row += f"{val:>10.1f}"
        print(row)

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
