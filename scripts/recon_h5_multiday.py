"""
Multi-day holding period reconstruction for the H5 (DOI/Open Interest) signal.

Every previously-killed hypothesis in this program (including H5's own one-day
result of -7.27 bps, t = -4.86, n = 1618) was tested as ONE round trip per day,
charged against a hurdle levied PER ROUND TRIP. Holding period was never varied.
This script asks the single largest open question that follows from that: does
the cumulative k-day gross edge grow faster than the one-day figure, so that a
longer hold clears a hurdle paid only once instead of once per day?

Extends the original single-day H5 analysis to holding periods k in
{1, 2, 3, 5, 10} usable sessions. The signal itself (lagged DOI/OI ratio) does
not depend on the holding period; only the forward-return horizon changes.

Key methodology points (all non-negotiable, carried over from the verified
single-day script):
- "T-1" for the signal means the previous USABLE session (not a raw calendar-
  day subtraction) -- the bhavcopy for date d publishes AFTER that session
  closes, so using anything else would leak information.
- "T+k" for the forward return means the k-th next USABLE session, resolved
  purely from the already-time-label-resolved checkpoint arrays -- never a
  positional row offset on the raw panel.
- Returns run from the 09:16 open checkpoint to the 15:20 close checkpoint,
  both resolved by time label via Panel.rows_at_time().
- Cross-sectional demeaning and quintile spread construction (top signal
  quintile mean return minus bottom signal quintile mean return, buckets
  drawn from the SIGNAL, never from the outcome) follow the original verified
  script exactly.
- Sessions whose T+k horizon runs past the end of the usable-session window
  are DROPPED and counted (window_end_dropped), never filled or imputed.
- Two views are reported per k: naive/overlapping (all daily entries -- the
  naive t-stat is inflated by roughly sqrt(k) because consecutive entries
  share k-1 days of holding window and is NOT presented as a clean number)
  and a non-overlapping subsample (entries spaced exactly k usable sessions
  apart, so holding windows never overlap and the t-stat is not inflated by
  serial dependence in the sampling scheme).
"""

import datetime
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.universe.static import load_universe


def load_checkpoints():
    """
    Load universe, panel, and OI data; extract 09:16 open and 15:20 close
    checkpoints for each usable session (a session that has bars at BOTH
    checkpoint times, resolved by time label via Panel.rows_at_time()).

    Returns
    -------
    panel : Panel
        The loaded panel object.
    open_916 : np.ndarray
        Shape (n_usable, n_symbols) float64 array of 09:16 open prices.
    close_1520 : np.ndarray
        Shape (n_usable, n_symbols) float64 array of 15:20 close prices.
    usable_days : list
        Sorted list of panel day indices that have both 09:16 and 15:20 bars.
    oi_lookup : dict
        Dictionary mapping (symbol, date) -> (oi, doi) from the OI parquet.
    """
    # Load universe and panel. Window is 2019-01-01..2025-07-31: the OI file
    # starts in 2019, and the last 12 months of available data stay held out
    # for future validation rather than being spent on this recon.
    universe = load_universe("all_equity")
    spec = PanelSpec(
        freq="1",
        fields=("open", "close"),
        symbols=universe.symbols,
        start=datetime.date(2019, 1, 1),
        end=datetime.date(2025, 7, 31),
    )
    panel = load_panel(spec)

    # Load OI data
    oi_df = pd.read_parquet("data/external/fo_bhavcopy/stock_oi_daily.parquet")
    oi_df = oi_df[oi_df["d"].dt.date <= datetime.date(2025, 7, 31)]

    # Build OI lookup dict
    oi_lookup = {}
    for _, row in oi_df.iterrows():
        key = (row["sym"], row["d"].date())
        oi_lookup[key] = (float(row["oi"]), float(row["doi"]))

    # Get checkpoint rows, resolved BY TIME LABEL (never positionally)
    open_rows = panel.rows_at_time("09:16")
    close_rows = panel.rows_at_time("15:20")
    open_days = np.searchsorted(panel.day_offsets, open_rows, side="right") - 1
    close_days = np.searchsorted(panel.day_offsets, close_rows, side="right") - 1
    open_row_by_day = dict(zip(open_days.tolist(), open_rows.tolist()))
    close_row_by_day = dict(zip(close_days.tolist(), close_rows.tolist()))
    usable_days = sorted(set(open_row_by_day) & set(close_row_by_day))

    n_usable = len(usable_days)
    n_symbols = len(panel.symbols)
    open_916 = np.full((n_usable, n_symbols), np.nan, dtype=np.float64)
    close_1520 = np.full((n_usable, n_symbols), np.nan, dtype=np.float64)

    open_field = panel.field("open").astype(np.float64)
    close_field = panel.field("close").astype(np.float64)

    for idx, day in enumerate(usable_days):
        open_row = open_row_by_day[day]
        close_row = close_row_by_day[day]
        open_916[idx, :] = open_field[open_row, :]
        close_1520[idx, :] = close_field[close_row, :]

    return panel, open_916, close_1520, usable_days, oi_lookup


def build_signal(panel, oi_lookup, usable_days, n_symbols):
    """
    Build the lagged DOI/OI signal array: doi_norm[i, s] = doi[T-1] / oi[T-1].

    The signal at entry session i (usable_days[i]) uses OI data from the
    PREVIOUS USABLE SESSION (usable_days[i-1]), never a raw date subtraction.
    This is identical for all holding periods k -- the signal does not depend
    on how long the position is subsequently held.

    Parameters
    ----------
    panel : Panel
        Panel object with .symbols and .dates attributes.
    oi_lookup : dict
        Dictionary mapping (symbol, date) -> (oi, doi).
    usable_days : list
        Sorted list of panel day indices.
    n_symbols : int
        Number of symbols.

    Returns
    -------
    np.ndarray
        Shape (n_usable, n_symbols) float64 array of doi_norm_lagged.
    """
    n_usable = len(usable_days)
    doi_norm_lagged = np.full((n_usable, n_symbols), np.nan, dtype=np.float64)

    for i in range(1, n_usable):
        t_minus_1_date = panel.dates[usable_days[i - 1]]
        for s_idx, sym in enumerate(panel.symbols):
            if (sym, t_minus_1_date) in oi_lookup:
                oi_l, doi_l = oi_lookup[(sym, t_minus_1_date)]
                if oi_l > 0:
                    doi_norm_lagged[i, s_idx] = doi_l / oi_l

    return doi_norm_lagged


def forward_return(open_916, close_1520, k):
    """
    Compute k-session forward log returns from checkpoint arrays.

    r_fwd_k[i, s] = log(close_1520[i+k, s] / open_916[i, s])
    when i + k < n_usable and both prices are finite and > 0, else NaN.

    "T+k" means the k-th NEXT USABLE SESSION -- this is computed purely from
    the already time-label-resolved open_916/close_1520 checkpoint arrays,
    never via positional row arithmetic on the raw panel.

    Parameters
    ----------
    open_916 : np.ndarray
        Shape (n_usable, n_symbols) float64 array.
    close_1520 : np.ndarray
        Shape (n_usable, n_symbols) float64 array.
    k : int
        Holding period in usable sessions.

    Returns
    -------
    np.ndarray
        Shape (n_usable, n_symbols) float64 array of forward returns.
    """
    n_usable, n_symbols = open_916.shape
    r_fwd_k = np.full((n_usable, n_symbols), np.nan, dtype=np.float64)

    for i in range(n_usable - k):
        entry = open_916[i, :]
        exit_ = close_1520[i + k, :]
        valid = (
            np.isfinite(entry) & np.isfinite(exit_) & (entry > 0) & (exit_ > 0)
        )
        if np.any(valid):
            r_fwd_k[i, valid] = np.log(exit_[valid] / entry[valid])

    return r_fwd_k


def session_spread(signal, r_fwd_k, entry_years, n_usable, k):
    """
    Compute cross-sectional quintile spreads for each valid entry session.

    For each session i with i >= 1 (signal needs T-1) and i + k <= n_usable - 1
    (forward return needs T+k in range):
    - Take symbols where both signal[i,:] and r_fwd_k[i,:] are finite.
    - Require >= 15 such symbols else skip (count as "few_symbols").
    - Cross-sectionally demean r_fwd_k among those symbols.
    - Split into quintiles by the SIGNAL with
      pd.qcut(sig_valid, 5, labels=False, duplicates="drop") -- buckets are
      drawn from the signal, never from the outcome (bucketing on the return
      itself would be circular and would show a spread regardless of whether
      the signal has any predictive power).
    - Require >= 2 resulting buckets else skip (count as "qcut_collapse").
    - Spread = mean(demeaned r in top signal bucket)
             - mean(demeaned r in bottom signal bucket).

    Sessions where i + k > n_usable - 1 are counted as "window_end_dropped"
    (a distinct counter), never filled or imputed.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_usable, n_symbols) float64 array.
    r_fwd_k : np.ndarray
        Shape (n_usable, n_symbols) float64 array.
    entry_years : np.ndarray
        Shape (n_usable,) int64 array, calendar year of each usable session.
    n_usable : int
        Number of usable sessions.
    k : int
        Holding period in usable sessions.

    Returns
    -------
    spreads : np.ndarray
        Array of per-session spreads (one per entry session that produced a result).
    years : np.ndarray
        Aligned array of entry-session years.
    entry_indices : np.ndarray
        Aligned array of entry-session indices i.
    qcut_collapse : int
        Number of sessions skipped due to qcut collapse.
    few_symbols : int
        Number of sessions skipped due to too few symbols.
    window_end_dropped : int
        Number of sessions dropped because i + k > n_usable - 1.
    """
    spreads = []
    years = []
    entry_indices = []
    qcut_collapse = 0
    few_symbols = 0
    window_end_dropped = 0

    for i in range(1, n_usable):
        if i + k > n_usable - 1:
            window_end_dropped += 1
            continue

        # Get finite signal and return values
        valid_mask = np.isfinite(signal[i, :]) & np.isfinite(r_fwd_k[i, :])
        n_valid = np.sum(valid_mask)

        if n_valid < 15:
            few_symbols += 1
            continue

        sig_valid = signal[i, valid_mask]
        ret_valid = r_fwd_k[i, valid_mask]

        # Cross-sectionally demean returns
        ret_demeaned = ret_valid - np.mean(ret_valid)

        # Quintile split by SIGNAL (not by return -- see docstring)
        try:
            quintiles = pd.qcut(sig_valid, 5, labels=False, duplicates="drop")
        except ValueError:
            qcut_collapse += 1
            continue

        n_buckets = len(np.unique(quintiles))
        if n_buckets < 2:
            qcut_collapse += 1
            continue

        # Top and bottom signal-bucket mean returns
        top_mask = quintiles == np.max(quintiles)
        bottom_mask = quintiles == np.min(quintiles)
        spread = np.mean(ret_demeaned[top_mask]) - np.mean(ret_demeaned[bottom_mask])

        spreads.append(spread)
        years.append(entry_years[i])
        entry_indices.append(i)

    return (
        np.array(spreads, dtype=np.float64),
        np.array(years, dtype=np.int64),
        np.array(entry_indices, dtype=np.int64),
        qcut_collapse,
        few_symbols,
        window_end_dropped,
    )


def compute_statistics(spreads, years, entry_indices, k, break_even_bps):
    """
    Compute summary statistics for a set of spreads.

    Parameters
    ----------
    spreads : np.ndarray
        Array of per-session spreads.
    years : np.ndarray
        Aligned array of entry-session years.
    entry_indices : np.ndarray
        Aligned array of entry-session indices.
    k : int
        Holding period in usable sessions.
    break_even_bps : float
        Break-even hurdle in basis points.

    Returns
    -------
    dict
        Dictionary with all computed statistics.
    """
    n = len(spreads)
    if n == 0:
        return {
            "k": k,
            "hold_days": k,
            "cum_spread_bps": np.nan,
            "t_stat": np.nan,
            "n_sessions": 0,
            "per_day_bps": np.nan,
            "year_means": {},
            "clears_hurdle": False,
        }

    cum_spread_bps = np.mean(spreads) * 1e4
    std = np.std(spreads, ddof=1)
    t_stat = np.mean(spreads) / (std / np.sqrt(n)) if std > 0 and n > 1 else np.nan
    # k=0 is the same-session baseline (entry and exit both on session T); it
    # is already a 1-session figure, so there is no separate "per-day" rate.
    per_day_bps = cum_spread_bps if k == 0 else cum_spread_bps / k

    # Per-year means for specified years
    year_list = [2019, 2021, 2023, 2024, 2025]
    year_means = {}
    for yr in year_list:
        mask = years == yr
        if np.sum(mask) > 0:
            year_means[yr] = np.mean(spreads[mask]) * 1e4
        else:
            year_means[yr] = np.nan

    clears_hurdle = abs(cum_spread_bps) > break_even_bps

    return {
        "k": k,
        "hold_days": k,
        "cum_spread_bps": cum_spread_bps,
        "t_stat": t_stat,
        "n_sessions": n,
        "per_day_bps": per_day_bps,
        "year_means": year_means,
        "clears_hurdle": clears_hurdle,
    }


def non_overlap_subsample(spreads, years, entry_indices, k):
    """
    Select a non-overlapping subsample of entries.

    Entries are consecutive usable-session indices i. Pick a fixed residue
    class: let i0 be the smallest valid i for this k; keep only observations
    whose entry index i satisfies (i - i0) % k == 0. This guarantees
    consecutive kept entries are exactly k usable sessions apart, so their
    holding windows never overlap, even if some entries in between were
    skipped for other reasons (few_symbols / qcut_collapse).

    Parameters
    ----------
    spreads : np.ndarray
        Array of per-session spreads.
    years : np.ndarray
        Aligned array of entry-session years.
    entry_indices : np.ndarray
        Aligned array of entry-session indices.
    k : int
        Holding period in usable sessions.

    Returns
    -------
    tuple
        (sub_spreads, sub_years, sub_entry_indices)
    """
    if len(entry_indices) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    i0 = np.min(entry_indices)
    keep_mask = (entry_indices - i0) % k == 0

    return (
        spreads[keep_mask],
        years[keep_mask],
        entry_indices[keep_mask],
    )


def print_table(stats_list, table_title, t_col_name, break_even_bps):
    """
    Print a formatted table of statistics.

    Parameters
    ----------
    stats_list : list
        List of dicts from compute_statistics.
    table_title : str
        Title for the table.
    t_col_name : str
        Name for the t-statistic column.
    break_even_bps : float
        Break-even hurdle in basis points (used only for the column label).
    """
    print(f"\n{table_title}")
    print("=" * 130)

    clears_label = f"clears_{break_even_bps:.2f}bps?"
    header = (
        f"{'k':>3} {'hold_days':>10} {'cum_spread_bps':>15} "
        f"{t_col_name:>37} {'n_sessions':>10} {'per_day_bps':>12} "
        f"{'2019':>8} {'2021':>8} {'2023':>8} {'2024':>8} {'2025':>8} "
        f"{clears_label:>20}"
    )
    print(header)
    print("-" * 130)

    for stats in stats_list:
        year_str = ""
        for yr in [2019, 2021, 2023, 2024, 2025]:
            val = stats["year_means"].get(yr, np.nan)
            if np.isnan(val):
                year_str += f"{'n/a':>8} "
            else:
                year_str += f"{val:>8.2f} "

        clears_str = "YES" if stats["clears_hurdle"] else "NO"
        t_str = "n/a" if np.isnan(stats["t_stat"]) else f"{stats['t_stat']:.3f}"
        cum_str = "n/a" if np.isnan(stats["cum_spread_bps"]) else f"{stats['cum_spread_bps']:.2f}"
        per_day_str = "n/a" if np.isnan(stats["per_day_bps"]) else f"{stats['per_day_bps']:.2f}"
        print(
            f"{stats['k']:>3} {stats['hold_days']:>10} "
            f"{cum_str:>15} "
            f"{t_str:>37} {stats['n_sessions']:>10} "
            f"{per_day_str:>12} {year_str}"
            f"{clears_str:>20}"
        )


def print_full_year_breakdown(spreads, years, k, year_list):
    """
    Print full year-by-year breakdown for a specific k (all 7 years, for
    eyeballing sign stability across the full window rather than just the
    5 sampled years in the main table).

    Parameters
    ----------
    spreads : np.ndarray
        Array of per-session spreads.
    years : np.ndarray
        Aligned array of entry-session years.
    k : int
        Holding period.
    year_list : list
        List of years to report.
    """
    print(f"\nFull year-by-year breakdown for k={k} (naive spreads, bps):")
    print("-" * 60)
    for yr in year_list:
        mask = years == yr
        if np.sum(mask) > 0:
            mean_bps = np.mean(spreads[mask]) * 1e4
            n_obs = np.sum(mask)
            print(f"  {yr}: {mean_bps:>8.2f} bps  (n={n_obs})")
        else:
            print(f"  {yr}: {'n/a':>8}")


def main():
    """
    Main execution: load data, compute spreads for all k, print reports.
    """
    # Load data and build signal
    panel, open_916, close_1520, usable_days, oi_lookup = load_checkpoints()
    n_usable = len(usable_days)
    n_symbols = len(panel.symbols)

    signal = build_signal(panel, oi_lookup, usable_days, n_symbols)
    entry_years = np.array(
        [panel.dates[d].year for d in usable_days], dtype=np.int64
    )

    # Compute break-even hurdle from the cost model, not a hardcoded literal.
    # Two legs (top-minus-bottom spread trade), Rs 1 lakh clip per leg.
    break_even_bps = 2.0 * NSEIntradayEquityCosts().round_trip_bps(1e5)
    print(f"\nBreak-even hurdle: {break_even_bps:.5f} bps")
    print("  (= 2 legs x NSEIntradayEquityCosts().round_trip_bps(1e5))")

    # Baseline validation (k=0): entry AND exit both on session T (same
    # session, no overnight hold at all). This is the ORIGINAL one-day H5
    # result already on record (-7.27 bps, t=-4.86, n=1618) -- distinct from
    # the k=1..10 sweep below, whose "k" counts sessions held BEYOND the
    # entry session (k=1 means exit at T+1 close, i.e. one full overnight
    # hold). Printed here purely to confirm the join/lag logic in this
    # script reproduces the known number before trusting the new k>=1 rows.
    r_fwd_baseline = forward_return(open_916, close_1520, 0)
    (
        spreads_0,
        years_0,
        entry_indices_0,
        qcut_collapse_0,
        few_symbols_0,
        window_end_dropped_0,
    ) = session_spread(signal, r_fwd_baseline, entry_years, n_usable, 0)
    stats_baseline = compute_statistics(spreads_0, years_0, entry_indices_0, 0, break_even_bps)
    print_table(
        [stats_baseline],
        "Baseline validation (k=0, entry+exit both on session T, same-session -- "
        "should reproduce the known -7.27 bps / t=-4.86 / n=1618 one-day result)",
        "t_baseline",
        break_even_bps,
    )

    # Holding periods to test, in usable sessions -- k counts sessions held
    # BEYOND the entry session (exit at close of session T+k).
    k_values = [1, 2, 3, 5, 10]

    all_stats_naive = []
    all_stats_nonoverlap = []
    all_skip_counters = {}
    all_spreads_by_k = {}

    for k in k_values:
        r_fwd_k = forward_return(open_916, close_1520, k)

        (
            spreads,
            years,
            entry_indices,
            qcut_collapse,
            few_symbols,
            window_end_dropped,
        ) = session_spread(signal, r_fwd_k, entry_years, n_usable, k)

        all_skip_counters[k] = {
            "qcut_collapse": qcut_collapse,
            "few_symbols": few_symbols,
            "window_end_dropped": window_end_dropped,
        }
        all_spreads_by_k[k] = (spreads, years)

        # Naive/overlapping statistics (all daily entries)
        stats_naive = compute_statistics(spreads, years, entry_indices, k, break_even_bps)
        all_stats_naive.append(stats_naive)

        # Non-overlapping subsample (entries spaced k usable sessions apart)
        sub_spreads, sub_years, sub_entry_indices = non_overlap_subsample(
            spreads, years, entry_indices, k
        )
        stats_nonoverlap = compute_statistics(
            sub_spreads, sub_years, sub_entry_indices, k, break_even_bps
        )
        all_stats_nonoverlap.append(stats_nonoverlap)

    print_table(
        all_stats_naive,
        "Table 1: Naive/Overlapping (all daily entries) -- t IS INFLATED, see note below",
        "t_naive_UNCORRECTED_overlap_inflated",
        break_even_bps,
    )
    print(
        "\nNOTE: with k > 1, consecutive daily entries share k-1 days of "
        "holding window. t_naive above is UNCORRECTED for this overlap and "
        "is inflated by roughly sqrt(k). Do not treat it as a clean estimate "
        "-- see Table 2 for the non-overlapping, serially-independent version."
    )

    print_table(
        all_stats_nonoverlap,
        "Table 2: Non-overlapping subsample (entries spaced k sessions apart, corrected t)",
        "t_nonoverlap_corrected",
        break_even_bps,
    )

    year_list_full = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for k in [1, 10]:
        spreads_k, years_k = all_spreads_by_k[k]
        print_full_year_breakdown(spreads_k, years_k, k, year_list_full)

    print("\nSkip-reason counters per k:")
    print("-" * 60)
    print(f"{'k':>3} {'qcut_collapse':>15} {'few_symbols':>12} {'window_end_dropped':>20}")
    print("-" * 60)
    for k in k_values:
        counters = all_skip_counters[k]
        print(
            f"{k:>3} {counters['qcut_collapse']:>15} "
            f"{counters['few_symbols']:>12} {counters['window_end_dropped']:>20}"
        )


if __name__ == "__main__":
    main()
