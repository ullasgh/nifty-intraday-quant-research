#!/usr/bin/env python3
"""Recon: liquidity profile of the low-turnover tilt book.

This script measures where the a=0.10 smoothed tilt book actually sits in
liquidity space, using the raw panel's rupee turnover (close * volume) as the
liquidity metric. It answers four questions:

1. What fraction of the book's weight sits in each liquidity decile, and how
   does that compare to the equal-weight benchmark?
2. Where does the book's excess return come from, by liquidity decile?
3. Does the edge survive if we exclude the illiquid tail (bottom 1, 2, 3, or 5
   deciles)?
4. What is the implied capacity of the book under a crude 2%-of-ADV position
   sizing heuristic?

The liquidity metric is causal: it uses a trailing 20-session mean of daily
rupee turnover, lagged one session so that session i's liquidity is computed
only from sessions strictly before i. The book itself is the a=0.10 smoothed
mild-tilt book from ``recon_low_turnover_tilt.py``, reproduced exactly via
``base.simulate_smoothing`` and a byte-for-byte copy of its update rule.

All arithmetic follows the base script's conventions: float64 accumulation,
NaN means "no bar" (never forward-filled), turnover is sum of absolute weight
changes, costs charge one leg on turnover only at the primary clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_low_turnover_tilt as base

from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    _build_checkpoint_panel,
    build_overnight_feature,
)
from nifty_quant.universe.static import load_universe

START = base.START
END = base.END
PRIMARY_CLIP = base.PRIMARY_CLIP
SECONDARY_CLIP = base.SECONDARY_CLIP

TILT_NAMES = base.TILT_NAMES
UNIVERSE_NAMES = base.UNIVERSE_NAMES
SUMMARY_YEARS = base.SUMMARY_YEARS

# Pre-registered liquidity parameters.
ADV_WINDOW = 20
CAPACITY_ADV_FRACTION = 0.02
EXCLUSION_NS = (10, 9, 8, 7, 5)


def simulate_smoothing_with_books(
    sessions: list[base.PrecomputedSession],
    tilt: str,
    a: float,
    cost_bps_primary: float,
) -> tuple[list[base.SessionRecord], np.ndarray, np.ndarray]:
    """Simulate weight smoothing with parameter a, returning books and returns.

    This is a byte-for-byte copy of ``base.simulate_smoothing``'s update rule
    (``book_start = (1-a)*book + a*target``; same turnover/excess/net formulas)
    but additionally returns, per included session, the full-length
    ``book_start`` vector and ``r_full`` vector.

    Returns:
        records: list of SessionRecord, identical to base.simulate_smoothing.
        book_matrix: (n_sessions, n_symbols) float64 array of book_start vectors.
        r_matrix: (n_sessions, n_symbols) float64 array of r_full vectors.
    """
    n_symbols = len(sessions[0].r_full)
    book = np.zeros(n_symbols, dtype=np.float64)
    records: list[base.SessionRecord] = []
    book_matrix = np.zeros((len(sessions), n_symbols), dtype=np.float64)
    r_matrix = np.zeros((len(sessions), n_symbols), dtype=np.float64)

    for i, sess in enumerate(sessions):
        target = (
            sess.mild_target_full if tilt == "mild" else sess.aggressive_target_full
        )
        r_j = sess.r_full

        book_start = (1.0 - a) * book + a * target
        turnover = float(np.sum(np.abs(book_start - book)))
        return_j = float(np.dot(book_start, r_j))
        excess_bps = (return_j - sess.benchmark_return) * 10_000.0
        net_bps = excess_bps - cost_bps_primary * turnover

        records.append(
            base.SessionRecord(
                date=sess.date,
                excess_bps=excess_bps,
                turnover=turnover,
                net_bps_primary=net_bps,
            )
        )

        book_matrix[i, :] = book_start
        r_matrix[i, :] = r_j
        book = book_start.copy()

    return records, book_matrix, r_matrix


def compute_daily_dollar_vol(panel: Panel) -> np.ndarray:
    """Compute daily rupee turnover (close * volume) per raw day and symbol.

    Returns:
        daily_dollar_vol: (n_days, n_symbols) float64 array. NaN where no
        finite close*volume bar exists for that day/symbol.
    """
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    n_days = panel.n_days()
    n_symbols = panel.n_symbols()
    daily_dollar_vol = np.full((n_days, n_symbols), np.nan, dtype=np.float64)

    for d in range(n_days):
        start = int(panel.day_offsets[d])
        end = int(panel.day_offsets[d + 1])
        if start >= end:
            continue
        close_day = close[start:end, :]
        volume_day = volume[start:end, :]
        product = close_day * volume_day
        # A bar contributes NaN if either close or volume is NaN.
        product[~(np.isfinite(close_day) & np.isfinite(volume_day))] = np.nan
        # nansum over all-NaN slice returns 0.0; guard explicitly.
        day_sum = np.nansum(product, axis=0)
        all_nan = np.all(np.isnan(product), axis=0)
        day_sum[all_nan] = np.nan
        daily_dollar_vol[d, :] = day_sum

    return daily_dollar_vol


def compute_lagged_liquidity(
    daily_dollar_vol: np.ndarray,
    panel: Panel,
) -> np.ndarray:
    """Compute trailing 20-session nanmean of daily dollar vol, lagged one day.

    Returns:
        liq_lagged: (n_days, n_symbols) float64 array. Row d holds the mean of
        days [d-20, d-1], never including day d itself. Row 0 is NaN.
    """
    df = pd.DataFrame(
        daily_dollar_vol,
        index=range(panel.n_days()),
        columns=range(panel.n_symbols()),
    )
    rolled = df.rolling(window=ADV_WINDOW, min_periods=1).mean()
    shifted = rolled.shift(1)
    return shifted.to_numpy(dtype=np.float64)


def assign_deciles(
    liq_by_session: np.ndarray,
) -> np.ndarray:
    """Assign liquidity deciles per session.

    For each session, classify every symbol with finite liquidity into decile
    0..9 (0 = least liquid, 9 = most liquid) via stable-argsort rank.
    Symbols with non-finite liquidity get decile -1.

    Returns:
        decile_by_session: (n_sessions, n_symbols) int array.
    """
    n_sessions, n_symbols = liq_by_session.shape
    decile_by_session = np.full((n_sessions, n_symbols), -1, dtype=np.int64)

    for i in range(n_sessions):
        liq_row = liq_by_session[i, :]
        finite_mask = np.isfinite(liq_row)
        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            continue
        if n_finite == 1:
            # Lone finite symbol goes to decile 9.
            decile_by_session[i, finite_mask] = 9
            continue

        finite_values = liq_row[finite_mask]
        order = np.argsort(finite_values, kind="stable")
        ranks = np.empty(n_finite, dtype=np.float64)
        ranks[order] = np.arange(n_finite, dtype=np.float64)
        rank_pct = ranks / (n_finite - 1)
        deciles = np.clip(np.floor(rank_pct * 10.0), 0, 9).astype(np.int64)
        finite_indices = np.where(finite_mask)[0]
        decile_by_session[i, finite_indices] = deciles

    return decile_by_session


def recover_valid_and_n_valid(
    sessions_full: list[base.PrecomputedSession],
    checkpoint_panel: Panel,
    checkpoint_feature,
    n_symbols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover per-session valid mask and n_valid for the full universe.

    This mirrors the "full" branch of base.precompute_sessions exactly. It is
    indexed 1:1 with sessions_full by POSITION (row i of the returned arrays
    corresponds to sessions_full[i]) -- NOT by naively assuming
    entry_row == 2*i on checkpoint_panel, because base.precompute_sessions
    SKIPS any checkpoint session with n_valid < 5 (this includes checkpoint
    session 0 itself: the window's first session has a NaN overnight feature
    for every symbol, per the base script's own docstring). If we assumed a
    constant entry_row = 2*i correspondence, every single session's valid
    mask would be pulled from the WRONG checkpoint row once even one earlier
    session was skipped. Instead we look up each sessions_full[i].date's true
    row in checkpoint_panel.dates before indexing.

    Returns:
        valid_matrix: (n_sessions, n_symbols) bool array.
        n_valid_array: (n_sessions,) int array.
    """
    close_field = checkpoint_panel.field("close")
    date_to_cp_idx = {date: cp_i for cp_i, date in enumerate(checkpoint_panel.dates)}
    n_sessions = len(sessions_full)
    valid_matrix = np.zeros((n_sessions, n_symbols), dtype=bool)
    n_valid_array = np.zeros(n_sessions, dtype=np.int64)

    for i, sess in enumerate(sessions_full):
        cp_idx = date_to_cp_idx[sess.date]
        entry_row = 2 * cp_idx
        exit_row = 2 * cp_idx + 1

        feat = checkpoint_feature.values[entry_row, :].astype(np.float64)
        price_entry = close_field[entry_row, :].astype(np.float64)
        price_exit = close_field[exit_row, :].astype(np.float64)

        valid = (
            np.isfinite(feat)
            & np.isfinite(price_entry)
            & np.isfinite(price_exit)
            & (price_entry > 0)
        )
        n_valid = int(valid.sum())
        # n_valid < 5 cannot happen here: sessions_full already excludes such
        # sessions by construction (base.precompute_sessions's own skip
        # rule), and we only ever look up dates that ARE in sessions_full.
        valid_matrix[i, :] = valid
        n_valid_array[i] = n_valid

    return valid_matrix, n_valid_array


def build_restricted_sessions(
    sessions_full: list[base.PrecomputedSession],
    decile_by_session: np.ndarray,
    n_keep: int,
    checkpoint_panel: Panel,
    checkpoint_feature,
    n_symbols: int,
) -> list[base.PrecomputedSession]:
    """Build a new session list restricted to the top-N liquidity deciles.

    Re-runs the same "full" branch of base.precompute_sessions verbatim, but
    intersects valid with an extra per-session eligibility mask:
    decile >= (10 - n_keep) and decile != -1. Applies the same n_valid < 5
    session-skip rule.

    Args:
        sessions_full: the official full-universe session list.
        decile_by_session: (n_sessions, n_symbols) int array.
        n_keep: number of top liquidity deciles to keep (1..10).
        checkpoint_panel, checkpoint_feature: for recomputing valid.
        n_symbols: number of symbols.

    Returns:
        restricted_sessions: list of PrecomputedSession, aligned by date with
        the subset of sessions_full that survive the restriction.

    IMPORTANT: `decile_by_session` is indexed by POSITION in `sessions_full`
    (built that way by main()), so `decile_by_session[i, :]` correctly lines
    up with `sessions_full[i]`. But `checkpoint_panel`/`checkpoint_feature`
    rows do NOT: base.precompute_sessions skips any checkpoint session with
    n_valid < 5 (including checkpoint session 0 itself, whose overnight
    feature is NaN for every symbol per the base script's own docstring), so
    `entry_row = 2 * i` would silently pull the WRONG checkpoint session's
    prices/feature for every i once even one earlier session was skipped.
    We look up each sess.date's true checkpoint row instead.
    """
    close_field = checkpoint_panel.field("close")
    date_to_cp_idx = {date: cp_i for cp_i, date in enumerate(checkpoint_panel.dates)}
    restricted: list[base.PrecomputedSession] = []

    for i, sess in enumerate(sessions_full):
        cp_idx = date_to_cp_idx[sess.date]
        entry_row = 2 * cp_idx
        exit_row = 2 * cp_idx + 1

        feat = checkpoint_feature.values[entry_row, :].astype(np.float64)
        price_entry = close_field[entry_row, :].astype(np.float64)
        price_exit = close_field[exit_row, :].astype(np.float64)

        base_valid = (
            np.isfinite(feat)
            & np.isfinite(price_entry)
            & np.isfinite(price_exit)
            & (price_entry > 0)
        )

        # Extra liquidity restriction.
        liq_mask = decile_by_session[i, :] >= (10 - n_keep)
        liq_mask &= decile_by_session[i, :] != -1
        valid = base_valid & liq_mask

        n_valid = int(valid.sum())
        if n_valid < 5:
            continue

        r = (price_exit[valid] / price_entry[valid]) - 1.0
        feat_valid = feat[valid]

        benchmark_return = float(np.mean(r))

        # Mild tilt: clipped-rank weights over the bottom half by feature.
        order = np.argsort(feat_valid, kind="stable")
        ranks = np.empty(n_valid, dtype=np.float64)
        ranks[order] = np.arange(n_valid, dtype=np.float64)
        rank_pct = ranks / (n_valid - 1)
        score = np.clip(0.5 - rank_pct, 0.0, None)
        score_sum = float(score.sum())
        if score_sum > 0.0:
            mild_weights_valid = score / score_sum
        else:
            mild_weights_valid = np.full(n_valid, 1.0 / n_valid, dtype=np.float64)

        # Aggressive tilt: bottom 20% by feature.
        n_bottom = max(1, int(round(n_valid * 0.2)))
        bottom_idx = order[:n_bottom]
        aggressive_weights_valid = np.zeros(n_valid, dtype=np.float64)
        aggressive_weights_valid[bottom_idx] = 1.0 / n_bottom

        r_full = np.zeros(n_symbols, dtype=np.float64)
        r_full[valid] = r

        mild_target_full = np.zeros(n_symbols, dtype=np.float64)
        mild_target_full[valid] = mild_weights_valid

        aggressive_target_full = np.zeros(n_symbols, dtype=np.float64)
        aggressive_target_full[valid] = aggressive_weights_valid

        raw_return_all = price_exit / price_entry - 1.0
        raw_return_all[
            ~(np.isfinite(price_entry) & (price_entry > 0) & np.isfinite(price_exit))
        ] = np.nan

        restricted.append(
            base.PrecomputedSession(
                date=sess.date,
                benchmark_return=benchmark_return,
                r_full=r_full,
                mild_target_full=mild_target_full,
                aggressive_target_full=aggressive_target_full,
                raw_return_all=raw_return_all.copy(),
            )
        )

    return restricted


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of values.

    Sorts by values ascending, cumulative weights, returns the value at which
    cumulative weight first reaches >= 50% of total weight.
    """
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    if total_weight <= 0:
        return float("nan")
    target = 0.5 * total_weight
    idx = int(np.searchsorted(cum_weights, target, side="left"))
    if idx >= len(sorted_values):
        idx = len(sorted_values) - 1
    return float(sorted_values[idx])


def print_task1_weight_profile(
    book_matrix_mild: np.ndarray,
    book_matrix_aggressive: np.ndarray,
    decile_by_session: np.ndarray,
    valid_matrix: np.ndarray,
    n_valid_array: np.ndarray,
) -> None:
    """Print weight-by-decile profile for mild and aggressive books."""
    n_sessions, n_symbols = book_matrix_mild.shape
    deciles = list(range(10))

    mild_weight_by_decile = np.zeros((n_sessions, 10), dtype=np.float64)
    aggressive_weight_by_decile = np.zeros((n_sessions, 10), dtype=np.float64)
    benchmark_weight_by_decile = np.zeros((n_sessions, 10), dtype=np.float64)
    mild_unclassified = np.zeros(n_sessions, dtype=np.float64)

    for i in range(n_sessions):
        for d in deciles:
            mask = decile_by_session[i, :] == d
            mild_weight_by_decile[i, d] = np.sum(book_matrix_mild[i, mask])
            aggressive_weight_by_decile[i, d] = np.sum(book_matrix_aggressive[i, mask])
            valid_mask = valid_matrix[i, :] & mask
            if n_valid_array[i] > 0:
                benchmark_weight_by_decile[i, d] = (
                    np.count_nonzero(valid_mask) / n_valid_array[i]
                )
        unclassified_mask = (decile_by_session[i, :] == -1) & (book_matrix_mild[i, :] > 0)
        mild_unclassified[i] = np.sum(book_matrix_mild[i, unclassified_mask])

    print("\n=== Task 1: Weight by liquidity decile (a=0.10) ===", flush=True)
    print(
        "decile  mild_book_%  aggressive_book_%  benchmark_%  book/bench_ratio(mild)",
        flush=True,
    )
    for d in deciles:
        mild_mean = float(np.mean(mild_weight_by_decile[:, d])) * 100.0
        aggressive_mean = float(np.mean(aggressive_weight_by_decile[:, d])) * 100.0
        bench_mean = float(np.mean(benchmark_weight_by_decile[:, d])) * 100.0
        ratio = mild_mean / bench_mean if bench_mean > 1e-12 else float("nan")
        print(
            f"{d:>6}  {mild_mean:11.2f}  {aggressive_mean:17.2f}  {bench_mean:11.2f}  "
            f"{ratio:22.3f}",
            flush=True,
        )
    mild_unclassified_mean = float(np.mean(mild_unclassified)) * 100.0
    print(
        f"unclassified  {mild_unclassified_mean:11.2f}  {'N/A':>17}  {'N/A':>11}  "
        f"{'N/A':>22}",
        flush=True,
    )


def print_task2_attribution(
    book_matrix_mild: np.ndarray,
    r_matrix_mild: np.ndarray,
    decile_by_session: np.ndarray,
    valid_matrix: np.ndarray,
    n_valid_array: np.ndarray,
    records_control: list[base.SessionRecord],
    stats_control: dict[str, float],
) -> None:
    """Print return attribution by liquidity decile for mild tilt."""
    n_sessions, n_symbols = book_matrix_mild.shape
    deciles = list(range(10))

    excess_by_decile = np.zeros((n_sessions, 10), dtype=np.float64)

    for i in range(n_sessions):
        for d in deciles:
            mask = decile_by_session[i, :] == d
            book_return_d = np.sum(book_matrix_mild[i, mask] * r_matrix_mild[i, mask])
            valid_mask = valid_matrix[i, :] & mask
            if n_valid_array[i] > 0:
                bench_return_d = (
                    np.sum(r_matrix_mild[i, valid_mask]) / n_valid_array[i]
                )
            else:
                bench_return_d = 0.0
            excess_by_decile[i, d] = 10000.0 * (book_return_d - bench_return_d)

    # Unclassified residual.
    official_excess = np.array([r.excess_bps for r in records_control], dtype=np.float64)
    sum_decile_excess = np.sum(excess_by_decile, axis=1)
    residual = official_excess - sum_decile_excess

    print("\n=== Task 2: Return attribution by liquidity decile (mild, a=0.10) ===", flush=True)
    print("decile  mean_excess_bps/day  %_of_total_gross", flush=True)
    total_gross = stats_control["mean_gross_bps"]
    for d in deciles:
        mean_excess = float(np.mean(excess_by_decile[:, d]))
        pct = mean_excess / total_gross * 100.0 if abs(total_gross) > 1e-12 else float("nan")
        print(f"{d:>6}  {mean_excess:18.2f}  {pct:16.2f}", flush=True)
    mean_residual = float(np.mean(residual))
    pct_residual = mean_residual / total_gross * 100.0 if abs(total_gross) > 1e-12 else float("nan")
    print(
        f"unclassified  {mean_residual:18.2f}  {pct_residual:16.2f}",
        flush=True,
    )


def print_task3_exclusion_ladder(
    sessions_full: list[base.PrecomputedSession],
    decile_by_session: np.ndarray,
    checkpoint_panel: Panel,
    checkpoint_feature,
    n_symbols: int,
    cost_bps_primary: float,
    records_control: list[base.SessionRecord],
    stats_control: dict[str, float],
) -> None:
    """Print illiquid-tail exclusion ladder for mild and aggressive tilts."""
    print("\n=== Task 3: Illiquid-tail exclusion ladder (a=0.10) ===", flush=True)

    for tilt in ("mild", "aggressive"):
        print(f"\n--- {tilt} tilt ---", flush=True)
        print(
            "N   gross_bps   turnover   NET_bps   ann_net_%   2024_net   2025_net",
            flush=True,
        )
        for n_keep in EXCLUSION_NS:
            if n_keep == 10:
                records = records_control if tilt == "mild" else None
                if tilt == "aggressive":
                    # Need aggressive control for N=10.
                    records_agg, _, _ = simulate_smoothing_with_books(
                        sessions_full, "aggressive", 0.10, cost_bps_primary
                    )
                    records = records_agg
                stats = base.pooled_stats(records)
                by_year = base.aggregate_by_year(records)
                net_2024 = by_year.get(2024, {}).get("net_bps_primary", float("nan"))
                net_2025 = by_year.get(2025, {}).get("net_bps_primary", float("nan"))
            else:
                restricted = build_restricted_sessions(
                    sessions_full,
                    decile_by_session,
                    n_keep,
                    checkpoint_panel,
                    checkpoint_feature,
                    n_symbols,
                )
                if not restricted:
                    print(f"{n_keep:>2}  {'no data':>9}", flush=True)
                    continue
                records = base.simulate_smoothing(
                    restricted, tilt, 0.10, cost_bps_primary
                )
                stats = base.pooled_stats(records)
                by_year = base.aggregate_by_year(records)
                net_2024 = by_year.get(2024, {}).get("net_bps_primary", float("nan"))
                net_2025 = by_year.get(2025, {}).get("net_bps_primary", float("nan"))

            print(
                f"{n_keep:>2}  {stats['mean_gross_bps']:9.2f}  {stats['mean_turnover']:9.4f}  "
                f"{stats['mean_net_bps']:9.2f}  {stats['ann_net_pct']:9.2f}  "
                f"{net_2024:9.2f}  {net_2025:9.2f}",
                flush=True,
            )


def print_task4_capacity(
    book_matrix_mild: np.ndarray,
    liq_by_session: np.ndarray,
) -> None:
    """Print capacity analysis for the a=0.10 mild book."""
    weights_list = []
    adv_list = []
    n_held_total = 0
    n_held_unclassified = 0

    n_sessions, n_symbols = book_matrix_mild.shape
    for i in range(n_sessions):
        for s in range(n_symbols):
            w = book_matrix_mild[i, s]
            if w > 0:
                n_held_total += 1
                adv = liq_by_session[i, s]
                if np.isfinite(adv):
                    weights_list.append(w)
                    adv_list.append(adv)
                else:
                    n_held_unclassified += 1

    weights_array = np.array(weights_list, dtype=np.float64)
    adv_array = np.array(adv_list, dtype=np.float64)

    if len(weights_array) == 0:
        print("\n=== Task 4: Capacity (a=0.10 mild) ===", flush=True)
        print("No finite-liquidity held observations.", flush=True)
        return

    median_adv_weighted = weighted_median(adv_array, weights_array)
    typical_weight = weighted_median(weights_array, weights_array)
    implied_capacity_aum = (
        CAPACITY_ADV_FRACTION * median_adv_weighted / typical_weight
        if typical_weight > 1e-12
        else float("nan")
    )

    print("\n=== Task 4: Capacity (a=0.10 mild) ===", flush=True)
    print(f"Observations (session, symbol) held: {n_held_total}", flush=True)
    print(
        f"Observations with no finite liquidity (excluded): {n_held_unclassified} "
        f"({n_held_unclassified / n_held_total * 100.0:.2f}% of held)",
        flush=True,
    )
    print(
        f"Weighted median ADV (INR): {median_adv_weighted:,.0f}",
        flush=True,
    )
    print(
        f"Typical weight (weight-weighted median): {typical_weight * 100.0:.4f}%",
        flush=True,
    )
    print(
        f"Implied capacity AUM (INR): {implied_capacity_aum:,.0f}",
        flush=True,
    )
    print(
        "Note: This is a crude position-sizing heuristic (position <= 2% of ADV), "
        "NOT a turnover-based capacity estimate. Actual daily-trading capacity "
        "(given ~11% daily book turnover) would be materially smaller.",
        flush=True,
    )


def print_verdict(
    stats_control: dict[str, float],
    stats_n5_mild: dict[str, float],
    stats_n7_mild: dict[str, float],
    stats_n8_mild: dict[str, float],
) -> None:
    """Print a programmatic verdict based on computed numbers."""
    print("\n=== Verdict ===", flush=True)
    control_net = stats_control["mean_net_bps"]
    n5_net = stats_n5_mild["mean_net_bps"]
    n7_net = stats_n7_mild["mean_net_bps"]
    n8_net = stats_n8_mild["mean_net_bps"]

    if n5_net > 0 and n5_net >= 0.5 * control_net:
        print(
            "Edge appears to survive exclusion of the illiquid tail: N=5 mild-tilt "
            f"net is {n5_net:.2f} bps/day vs control {control_net:.2f} bps/day.",
            flush=True,
        )
    elif n7_net < 0 or n7_net < 0.5 * control_net:
        print(
            "Edge concentrates in the illiquid tail and does not survive exclusion "
            f"-- NOT investable at meaningful size. Breakpoint: N=7 net is "
            f"{n7_net:.2f} bps/day vs control {control_net:.2f} bps/day.",
            flush=True,
        )
    elif n8_net < 0 or n8_net < 0.5 * control_net:
        print(
            "Edge concentrates in the illiquid tail and does not survive exclusion "
            f"-- NOT investable at meaningful size. Breakpoint: N=8 net is "
            f"{n8_net:.2f} bps/day vs control {control_net:.2f} bps/day.",
            flush=True,
        )
    else:
        print(
            "Edge partially survives exclusion of the illiquid tail, but the "
            f"breakpoint is between N=7 ({n7_net:.2f} bps/day) and N=5 "
            f"({n5_net:.2f} bps/day) vs control {control_net:.2f} bps/day.",
            flush=True,
        )


def main() -> None:
    """Run the liquidity recon and print the report."""
    print("Loading universe...", flush=True)
    symbols = load_universe("all_equity").symbols
    print(f"Universe has {len(symbols)} symbols.", flush=True)

    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=symbols,
        start=START,
        end=END,
    )

    print("Loading panel (this may take minutes)...", flush=True)
    panel = load_panel(spec, memmap=True)
    print(
        f"Panel loaded: {panel.n_rows()} rows, {panel.n_days()} days, "
        f"{panel.n_symbols()} symbols.",
        flush=True,
    )

    print("Building overnight feature...", flush=True)
    feature = build_overnight_feature(panel)
    print("Reducing to checkpoint panel...", flush=True)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, feature)
    print(
        f"Checkpoint panel: {checkpoint_panel.n_days()} sessions, "
        f"{checkpoint_panel.n_rows()} rows.",
        flush=True,
    )

    print("Computing continuous-coverage mask...", flush=True)
    cont_mask = base.continuous_coverage_mask(panel)
    n_cont = int(cont_mask.sum())
    print(f"Continuous-coverage symbols: {n_cont} / {panel.n_symbols()}.", flush=True)

    print("Precomputing session data...", flush=True)
    sessions_by_universe = base.precompute_sessions(
        checkpoint_panel,
        checkpoint_feature,
        cont_mask,
        panel.n_symbols(),
    )
    sessions_full = sessions_by_universe["full"]
    print(f"Full-universe sessions: {len(sessions_full)}.", flush=True)

    costs = NSEIntradayEquityCosts()
    cost_bps_primary = costs.round_trip_bps(PRIMARY_CLIP)

    # Step 3: Control check.
    print("\n=== CONTROL CHECK ===", flush=True)
    records_control = base.simulate_smoothing(
        sessions_full, "mild", 0.10, cost_bps_primary
    )
    stats_control = base.pooled_stats(records_control)
    print(
        f"Control (a=0.10 mild, full universe): gross={stats_control['mean_gross_bps']:.4f} "
        f"turnover={stats_control['mean_turnover']:.4f} net={stats_control['mean_net_bps']:.4f}",
        flush=True,
    )
    ref_gross, ref_turnover, ref_net = 3.20, 0.1100, 2.29
    tol = 0.02
    if (
        abs(stats_control["mean_gross_bps"] - ref_gross) < tol
        and abs(stats_control["mean_turnover"] - ref_turnover) < tol
        and abs(stats_control["mean_net_bps"] - ref_net) < tol
    ):
        print("CONTROL CHECK: PASS", flush=True)
    else:
        print("CONTROL CHECK: MISMATCH -- STOP AND REPORT", flush=True)

    # Step 4: Book capture.
    print("\n=== BOOK CAPTURE CHECK ===", flush=True)
    records_mild, book_matrix_mild, r_matrix_mild = simulate_smoothing_with_books(
        sessions_full, "mild", 0.10, cost_bps_primary
    )
    stats_mild_capture = base.pooled_stats(records_mild)
    if abs(stats_mild_capture["mean_net_bps"] - stats_control["mean_net_bps"]) < 1e-9:
        print("BOOK CAPTURE CHECK: PASS", flush=True)
    else:
        print("BOOK CAPTURE CHECK: FAIL", flush=True)

    records_aggressive, book_matrix_aggressive, r_matrix_aggressive = (
        simulate_smoothing_with_books(
            sessions_full, "aggressive", 0.10, cost_bps_primary
        )
    )

    # Step 5: Liquidity metric.
    print("\nComputing daily dollar volume...", flush=True)
    daily_dollar_vol = compute_daily_dollar_vol(panel)
    print("Computing lagged 20-session liquidity...", flush=True)
    liq_lagged = compute_lagged_liquidity(daily_dollar_vol, panel)

    # Map checkpoint session dates to raw day indices.
    date_to_raw_day = {date: d for d, date in enumerate(panel.dates)}
    n_checkpoint_sessions = len(sessions_full)
    liq_by_session = np.full(
        (n_checkpoint_sessions, panel.n_symbols()), np.nan, dtype=np.float64
    )
    for i, sess in enumerate(sessions_full):
        raw_day_idx = date_to_raw_day.get(sess.date)
        if raw_day_idx is not None:
            liq_by_session[i, :] = liq_lagged[raw_day_idx, :]
    assert len(sessions_full) == liq_by_session.shape[0]

    # Step 6: Decile assignment.
    print("Assigning liquidity deciles...", flush=True)
    decile_by_session = assign_deciles(liq_by_session)

    # Recover valid masks for benchmark weighting.
    valid_matrix, n_valid_array = recover_valid_and_n_valid(
        sessions_full,
        checkpoint_panel,
        checkpoint_feature,
        panel.n_symbols(),
    )

    # Step 7: Task 1.
    print_task1_weight_profile(
        book_matrix_mild,
        book_matrix_aggressive,
        decile_by_session,
        valid_matrix,
        n_valid_array,
    )

    # Step 8: Task 2.
    print_task2_attribution(
        book_matrix_mild,
        r_matrix_mild,
        decile_by_session,
        valid_matrix,
        n_valid_array,
        records_control,
        stats_control,
    )

    # Step 9: Task 3.
    print_task3_exclusion_ladder(
        sessions_full,
        decile_by_session,
        checkpoint_panel,
        checkpoint_feature,
        panel.n_symbols(),
        cost_bps_primary,
        records_control,
        stats_control,
    )

    # Compute stats for verdict.
    restricted_n5 = build_restricted_sessions(
        sessions_full,
        decile_by_session,
        5,
        checkpoint_panel,
        checkpoint_feature,
        panel.n_symbols(),
    )
    records_n5_mild = base.simulate_smoothing(
        restricted_n5, "mild", 0.10, cost_bps_primary
    )
    stats_n5_mild = base.pooled_stats(records_n5_mild)

    restricted_n7 = build_restricted_sessions(
        sessions_full,
        decile_by_session,
        7,
        checkpoint_panel,
        checkpoint_feature,
        panel.n_symbols(),
    )
    records_n7_mild = base.simulate_smoothing(
        restricted_n7, "mild", 0.10, cost_bps_primary
    )
    stats_n7_mild = base.pooled_stats(records_n7_mild)

    restricted_n8 = build_restricted_sessions(
        sessions_full,
        decile_by_session,
        8,
        checkpoint_panel,
        checkpoint_feature,
        panel.n_symbols(),
    )
    records_n8_mild = base.simulate_smoothing(
        restricted_n8, "mild", 0.10, cost_bps_primary
    )
    stats_n8_mild = base.pooled_stats(records_n8_mild)

    # Step 10: Task 4.
    print_task4_capacity(book_matrix_mild, liq_by_session)

    # Step 11: Verdict.
    print_verdict(stats_control, stats_n5_mild, stats_n7_mild, stats_n8_mild)


if __name__ == "__main__":
    main()
