#!/usr/bin/env python3
"""Recon: long-only, index-relative tilt on H2's overnight-reversal signal.

The five hypotheses tested so far in this program are all zero-net-exposure
long/short spreads: they pay a TWO-LEG cost hurdle (2 * round_trip_bps) and must
generate absolute alpha starting from zero return. A long-only, index-relative
TILT is structurally different: it trades one leg (half the hurdle) and starts
from the index's own return, so it only has to beat the benchmark, not create
return from flat. "Five market-neutral spreads failed" does not by itself say
anything about this axis -- this script measures it directly, using H2's signal
(the largest real effect measured in the program: pooled -24.30 bps, t = -16.66,
8/8 years sign-consistent for the top-minus-bottom cross-sectional spread).

Window
------
START = 2018-01-01, END = 2025-07-31 (inclusive). The last 12 months of the
pre-registered window are held out and never loaded here.

Construction
------------
1. Load one 1-minute panel over [START, END] for the ``all_equity`` universe.
2. Build the H2 overnight-return feature (``r_on = log(open[09:16] /
   close[15:20, d-1])``) via ``h2_overnight_reversal.build_overnight_feature``,
   then reduce to a checkpoint panel with exactly two rows per usable session via
   ``h2_overnight_reversal._build_checkpoint_panel``: row ``2*i`` is the 09:16
   open (stored in the reduced panel's ``"close"`` field) and row ``2*i + 1`` is
   the 15:20 close. The overnight feature is broadcast identically to both rows
   of its session and is known at 09:16, before that session's intraday return
   is realized, so using it to set that session's portfolio weights is causal.
   The panel is already loaded exactly over [START, END], so no further ``.sub``
   restriction is applied; the window's very first session has a NaN overnight
   feature for every symbol (no prior session inside the loaded window) and is
   excluded naturally by the finite-value checks below, not special-cased.
3. Continuous-coverage mask: a symbol is "continuous" iff it has at least one
   non-NaN ``close`` bar on the raw panel's first trading day AND at least one
   non-NaN ``close`` bar on the raw panel's last trading day. This mirrors the
   definition already used by ``scripts/recon_survivorship.py`` Part B, computed
   only from the panel's own bars (never ``data/MANIFEST.json``, which spans
   years outside this study window).
4. For each session and each of two eligibility universes ("full" = all
   symbols, "continuous" = the mask above):
   - ``valid`` = finite overnight feature, finite positive entry price, finite
     exit price, and in-universe.
   - If fewer than 5 valid names, the session is skipped entirely for that
     universe (too few names to form a quintile): it contributes no return and
     no turnover step, and the previous session's book is carried forward
     unchanged for the next included session's turnover computation.
   - Benchmark ("index proxy"): equal-weight over valid names.
   - Mild tilt: rank valid names by feature ascending (rank 0 = most negative
     overnight return = biggest "loser"). ``rank_pct = rank / (n_valid - 1)`` in
     [0, 1]. ``score = clip(0.5 - rank_pct, 0, None)`` -- zero for the top half
     of names by overnight return, positive and linearly increasing toward the
     biggest loser for the bottom half. Weight = ``score / sum(score)``,
     renormalized to sum to 1 over valid names, 0 elsewhere.
   - Aggressive tilt: hold only the bottom 20% (``max(1, round(n_valid *
     0.2))``) of names by feature (biggest losers), equal-weighted.
   - Turnover is tracked separately per (tilt, universe) combination as the sum
     of absolute weight changes from the previous INCLUDED session's book (a
     length-``n_symbols`` vector, starting at all-zero, so the first included
     session's turnover is the full buy-in from flat).
   - Costs charge ONE leg on turnover only:
     ``cost_bps = NSEIntradayEquityCosts().round_trip_bps(clip) * turnover``,
     at a PRIMARY clip of Rs 1,00,000 and a SECONDARY clip of Rs 10,00,000 for a
     footnote. ``round_trip_bps`` is a pure function of ``clip`` and is called
     once outside the per-session loop.
5. Aggregate per calendar year and pooled across the whole window.
   Annualization convention (arithmetic, not compounding):
   ``ann_excess_pct = mean(excess_bps) * 252.0 / 100.0`` -- bps/day to
   fraction/day is /10_000, times 252 trading days/year, times 100 for percent,
   net multiplier 252 / 100 = 2.52.
   Pooled significance: one-sample t-stat on the daily ``excess_bps`` series,
   ``t = mean / (std(ddof=1) / sqrt(n))``. One observation per session, so no
   overlap correction is needed (same reasoning as
   ``h2_overnight_reversal``'s module docstring).

Honesty
-------
This script does not tune tilt strength, clip size, or universe to manufacture
a positive result -- it implements exactly the pre-registered construction
above and reports whatever the numbers are, net of cost, for both universes and
both tilts.

Side effects
------------
None beyond stdout. Never modifies ``src/``, ``tests/``, or ``data/``.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import defaultdict

import numpy as np

from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    _build_checkpoint_panel,
    build_overnight_feature,
)
from nifty_quant.universe.static import load_universe

START = datetime.date(2018, 1, 1)
END = datetime.date(2025, 7, 31)

PRIMARY_CLIP = 100_000.0
SECONDARY_CLIP = 1_000_000.0

TILT_NAMES = ("mild", "aggressive")
UNIVERSE_NAMES = ("full", "continuous")

SUMMARY_YEARS = (2019, 2021, 2023, 2024, 2025)


@dataclasses.dataclass
class SessionRecord:
    """One session's results for one (tilt, universe) combination."""

    date: datetime.date
    excess_bps: float
    turnover: float
    cost_bps_primary: float
    net_bps_primary: float
    cost_bps_secondary: float
    net_bps_secondary: float


def continuous_coverage_mask(panel: Panel) -> np.ndarray:
    """Return a boolean mask over ``panel.symbols``.

    A symbol is True iff it has at least one non-NaN close bar on the panel's
    first trading day AND at least one non-NaN close bar on the panel's last
    trading day. Uses only the raw panel's own close field -- never
    ``data/MANIFEST.json``, which spans years outside this study window.
    """
    close = panel.field("close")
    first_day_rows = slice(int(panel.day_offsets[0]), int(panel.day_offsets[1]))
    last_day_rows = slice(int(panel.day_offsets[-2]), int(panel.day_offsets[-1]))
    first_day_ok = np.any(np.isfinite(close[first_day_rows, :]), axis=0)
    last_day_ok = np.any(np.isfinite(close[last_day_rows, :]), axis=0)
    return first_day_ok & last_day_ok


def compute_session_books(
    checkpoint_panel: Panel,
    checkpoint_feature,
    eligible_mask: np.ndarray,
    n_symbols: int,
) -> dict[tuple[str, str], list[SessionRecord]]:
    """Compute per-session returns, turnover, and costs for all tilt/universe combos.

    Parameters
    ----------
    checkpoint_panel : Panel
        Reduced panel with 2 rows per session (entry at 2*i, exit at 2*i+1).
    checkpoint_feature : Feature
        Overnight-return feature broadcast to both rows of each session.
    eligible_mask : np.ndarray
        Boolean mask over symbols for the "continuous" universe.
    n_symbols : int
        Total number of symbols in the panel.

    Returns
    -------
    dict mapping (tilt_name, universe_name) -> list of SessionRecord.
    """
    costs = NSEIntradayEquityCosts()
    cost_bps_primary = costs.round_trip_bps(PRIMARY_CLIP)
    cost_bps_secondary = costs.round_trip_bps(SECONDARY_CLIP)

    # Turnover bookkeeping: one previous-weight vector per (tilt, universe).
    prev_weights: dict[tuple[str, str], np.ndarray] = {}
    for tilt in TILT_NAMES:
        for universe in UNIVERSE_NAMES:
            prev_weights[(tilt, universe)] = np.zeros(n_symbols, dtype=np.float64)

    records: dict[tuple[str, str], list[SessionRecord]] = defaultdict(list)

    n_sessions = checkpoint_panel.n_days()
    close_field = checkpoint_panel.field("close")

    for i in range(n_sessions):
        date = checkpoint_panel.dates[i]
        entry_row = 2 * i
        exit_row = 2 * i + 1

        feat = checkpoint_feature.values[entry_row, :].astype(np.float64)
        price_entry = close_field[entry_row, :].astype(np.float64)
        price_exit = close_field[exit_row, :].astype(np.float64)

        base_valid = (
            np.isfinite(feat)
            & np.isfinite(price_entry)
            & np.isfinite(price_exit)
            & (price_entry > 0)
        )

        for universe in UNIVERSE_NAMES:
            if universe == "full":
                valid = base_valid
            else:
                valid = base_valid & eligible_mask

            n_valid = int(valid.sum())
            if n_valid < 5:
                # Skip session for this universe; do not update prev_weights.
                continue

            r = (price_exit[valid] / price_entry[valid]) - 1.0
            feat_valid = feat[valid]

            # Benchmark (equal-weight over valid names) -- the index proxy.
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
            mild_return = float(np.dot(mild_weights_valid, r))

            # Aggressive tilt: bottom 20% by feature (smallest feat = biggest loser).
            n_bottom = max(1, int(round(n_valid * 0.2)))
            bottom_idx = order[:n_bottom]
            aggressive_weights_valid = np.zeros(n_valid, dtype=np.float64)
            aggressive_weights_valid[bottom_idx] = 1.0 / n_bottom
            aggressive_return = float(np.dot(aggressive_weights_valid, r))

            # Full-length weight vectors (over all panel symbols) for turnover.
            mild_weights_full = np.zeros(n_symbols, dtype=np.float64)
            mild_weights_full[valid] = mild_weights_valid
            aggressive_weights_full = np.zeros(n_symbols, dtype=np.float64)
            aggressive_weights_full[valid] = aggressive_weights_valid

            for tilt, weights_full, tilt_return in (
                ("mild", mild_weights_full, mild_return),
                ("aggressive", aggressive_weights_full, aggressive_return),
            ):
                key = (tilt, universe)
                excess_bps = (tilt_return - benchmark_return) * 10_000.0
                turnover = float(np.sum(np.abs(weights_full - prev_weights[key])))
                prev_weights[key] = weights_full.copy()

                records[key].append(
                    SessionRecord(
                        date=date,
                        excess_bps=excess_bps,
                        turnover=turnover,
                        cost_bps_primary=cost_bps_primary * turnover,
                        net_bps_primary=excess_bps - cost_bps_primary * turnover,
                        cost_bps_secondary=cost_bps_secondary * turnover,
                        net_bps_secondary=excess_bps - cost_bps_secondary * turnover,
                    )
                )

    return dict(records)


def aggregate_by_year(
    records: list[SessionRecord],
) -> dict[int, dict[str, float]]:
    """Aggregate a list of SessionRecord by calendar year.

    Returns a dict mapping year -> dict with keys:
    n_sessions, excess_bps, turnover, net_bps_primary, net_bps_secondary
    (all except n_sessions are per-day means within that year).
    """
    by_year: dict[int, list[SessionRecord]] = defaultdict(list)
    for rec in records:
        by_year[rec.date.year].append(rec)

    result: dict[int, dict[str, float]] = {}
    for year, recs in sorted(by_year.items()):
        n = len(recs)
        result[year] = {
            "n_sessions": float(n),
            "excess_bps": float(np.mean([r.excess_bps for r in recs])),
            "turnover": float(np.mean([r.turnover for r in recs])),
            "net_bps_primary": float(np.mean([r.net_bps_primary for r in recs])),
            "net_bps_secondary": float(np.mean([r.net_bps_secondary for r in recs])),
        }
    return result


def pooled_stats(records: list[SessionRecord]) -> dict[str, float]:
    """Compute pooled (whole-window) statistics for a list of SessionRecord."""
    n = len(records)
    excess = np.array([r.excess_bps for r in records], dtype=np.float64)
    turnover = np.array([r.turnover for r in records], dtype=np.float64)
    net_primary = np.array([r.net_bps_primary for r in records], dtype=np.float64)

    mean_excess = float(np.mean(excess)) if n > 0 else float("nan")
    mean_turnover = float(np.mean(turnover)) if n > 0 else float("nan")
    mean_net_primary = float(np.mean(net_primary)) if n > 0 else float("nan")
    ann_excess_pct = mean_excess * 252.0 / 100.0 if n > 0 else float("nan")

    if n >= 2:
        t_stat = float(mean_excess / (np.std(excess, ddof=1) / np.sqrt(n)))
    else:
        t_stat = float("nan")

    return {
        "n_sessions": float(n),
        "mean_excess_bps": mean_excess,
        "ann_excess_pct": ann_excess_pct,
        "mean_turnover": mean_turnover,
        "mean_net_bps_primary": mean_net_primary,
        "t_stat": t_stat,
    }


def print_compact_summary(
    records_by_key: dict[tuple[str, str], list[SessionRecord]],
    universe: str,
) -> None:
    """Print the compact summary table for one universe."""
    print(f"\n=== Compact summary: {universe} universe ===", flush=True)
    header = (
        "tilt        excess_bps/day   ann_excess_%   turnover/day   "
        "net_after_cost_bps   2019  2021  2023  2024  2025"
    )
    print(header, flush=True)
    for tilt in TILT_NAMES:
        records = records_by_key.get((tilt, universe), [])
        stats = pooled_stats(records)
        by_year = aggregate_by_year(records)
        year_vals = []
        for year in SUMMARY_YEARS:
            if year in by_year:
                year_vals.append(f"{by_year[year]['excess_bps']:8.2f}")
            else:
                year_vals.append(f"{'NaN':>8}")
        print(
            f"{tilt:<11} {stats['mean_excess_bps']:15.2f} {stats['ann_excess_pct']:13.2f} "
            f"{stats['mean_turnover']:13.4f} {stats['mean_net_bps_primary']:19.2f} "
            f"{'  '.join(year_vals)}",
            flush=True,
        )


def print_full_report(
    records_by_key: dict[tuple[str, str], list[SessionRecord]],
) -> None:
    """Print pooled t-stats and full 2018-2025 per-year tables for all combinations."""
    print("\n=== Pooled t-statistics and session counts ===", flush=True)
    for tilt in TILT_NAMES:
        for universe in UNIVERSE_NAMES:
            stats = pooled_stats(records_by_key.get((tilt, universe), []))
            print(
                f"{tilt:<11} {universe:<11} n={int(stats['n_sessions']):>4}  "
                f"t-stat(excess_bps)={stats['t_stat']:8.3f}  "
                f"mean_net_bps(primary)={stats['mean_net_bps_primary']:8.3f}",
                flush=True,
            )

    print("\n=== Full per-year tables (2018-2025) ===", flush=True)
    for tilt in TILT_NAMES:
        for universe in UNIVERSE_NAMES:
            records = records_by_key.get((tilt, universe), [])
            by_year = aggregate_by_year(records)
            print(f"\n--- {tilt} tilt, {universe} universe ---", flush=True)
            print(
                "year  n_sessions  excess_bps/day  ann_excess_pct  "
                "turnover/day  net_bps(primary)  net_bps(secondary)",
                flush=True,
            )
            for year in sorted(by_year):
                y = by_year[year]
                ann = y["excess_bps"] * 252.0 / 100.0
                print(
                    f"{year}  {int(y['n_sessions']):>10}  {y['excess_bps']:14.2f}  "
                    f"{ann:13.2f}  {y['turnover']:12.4f}  "
                    f"{y['net_bps_primary']:16.2f}  {y['net_bps_secondary']:18.2f}",
                    flush=True,
                )


def print_honest_summary(
    records_by_key: dict[tuple[str, str], list[SessionRecord]],
) -> None:
    """Print a plain-language, un-tuned summary of net results."""
    print("\n=== Honest summary (net of primary-clip cost) ===", flush=True)
    for universe in UNIVERSE_NAMES:
        print(f"\n{universe.capitalize()} universe:", flush=True)
        for tilt in TILT_NAMES:
            records = records_by_key.get((tilt, universe), [])
            stats = pooled_stats(records)
            by_year = aggregate_by_year(records)

            pooled_net = stats["mean_net_bps_primary"]
            pooled_word = "positive" if pooled_net > 0 else "negative"

            recent_years = []
            for year in (2024, 2025):
                if year in by_year:
                    net = by_year[year]["net_bps_primary"]
                    word = "positive" if net > 0 else "negative"
                    recent_years.append(f"{year}: {word} ({net:.2f} bps/day)")
                else:
                    recent_years.append(f"{year}: no data")

            print(
                f"  {tilt:<11} pooled net (primary clip): {pooled_word} "
                f"({pooled_net:.2f} bps/day); recent: {', '.join(recent_years)}",
                flush=True,
            )


def main() -> None:
    """Run the full long-only-tilt reconstruction and print the report."""
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
    cont_mask = continuous_coverage_mask(panel)
    n_cont = int(cont_mask.sum())
    print(f"Continuous-coverage symbols: {n_cont} / {panel.n_symbols()}.", flush=True)

    print("Computing session books...", flush=True)
    records_by_key = compute_session_books(
        checkpoint_panel,
        checkpoint_feature,
        cont_mask,
        panel.n_symbols(),
    )

    print_compact_summary(records_by_key, "full")
    print_compact_summary(records_by_key, "continuous")
    print_full_report(records_by_key)
    print_honest_summary(records_by_key)


if __name__ == "__main__":
    main()
