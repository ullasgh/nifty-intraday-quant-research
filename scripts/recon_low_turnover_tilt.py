#!/usr/bin/env python3
"""Recon: low-turnover harvesting of H2's overnight-reversal tilt.

The baseline script ``scripts/recon_long_only_tilt.py`` measures a long-only,
index-relative tilt on H2's overnight-reversal signal, rebalancing every
session. It reports mild-tilt pooled gross excess of 10.03 bps/day at 1.2573
turnover/day (net -0.36 bps after primary-clip cost) and aggressive-tilt pooled
gross excess of 12.49 bps/day at 1.5026 turnover/day (net +0.07 bps). The
signal is real but the cost hurdle is large: at the primary clip, one-leg
round-trip cost is roughly 8 bps per unit turnover, so daily rebalancing
consumes most of the edge.

This script asks a different question: can the SAME signal, benchmark, target
weights, and one-leg cost accounting be harvested at materially lower turnover
by changing ONLY the rule that turns per-session target weights into a traded
book? Three constructions are tested, each with a single pre-registered
parameter grid:

1. Periodic rebalance (k in {1, 2, 3, 5, 10}): rebalance every k-th included
   session; between rebalances the book drifts organically with price (no
   trades, no turnover, no cost on hold days).
2. Hysteresis / no-trade band (b in {0.0, 0.25, 0.5, 1.0}): rebalance daily
   but only trade names whose target weight differs from the current book by
   more than b times the average target weight. No organic drift is modeled
   for this construction (kept simpler than construction 1 by design).
3. Weight smoothing (a in {1.0, 0.5, 0.25, 0.1}): rebalance daily, moving the
   book a fraction a of the way from its current state toward the target.

The control rows k=1 / b=0.0 / a=1.0 must reproduce the baseline script's
per-session turnover and excess_bps exactly for all four (tilt, universe)
combinations. This is the built-in sanity check.

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
     universe (too few names to form a quintile): it contributes no record and
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

Stated assumption (drift convention)
------------------------------------
Construction 1 (periodic rebalance) drifts the book between rebalances using
the raw per-name return ``price_exit / price_entry - 1``. When a held name has
a NaN raw return on a hold day (halt / missing bar), that name's return is
treated as 0.0 for that day ONLY when computing the drift. This is a
portfolio-accounting convention: it does NOT forward-fill price, it only means
"the position does not compound today because we cannot observe a return for
it." This is stated explicitly, not silently assumed. Drift only accumulates
across actual HOLD sessions; a rebalance session resets the carried-forward
book to exactly the target weight with no in-session drift applied, so that
the k=1 control reproduces the baseline exactly (see the fixed bug note in
``simulate_periodic``'s docstring below).

Honesty
-------
This script does not tune tilt strength, clip size, universe, or the
rebalancing parameters beyond the pre-registered grids above. It reports all
four (tilt, universe) combinations, all three constructions, all parameter
values, and calls out 2024 and 2025 net results explicitly. The breakeven
turnover is computed for the single best construction only, as a descriptive
statistic, not as a tuning target.

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

# Pre-registered parameter grids. The control values k=1 / b=0.0 / a=1.0 must
# reproduce the baseline script's per-session turnover and excess_bps exactly.
PERIODIC_KS = (1, 2, 3, 5, 10)
HYSTERESIS_BS = (0.0, 0.25, 0.5, 1.0)
SMOOTHING_AS = (1.0, 0.5, 0.25, 0.1)


@dataclasses.dataclass
class SessionRecord:
    """One session's results for one (construction, param, tilt, universe)."""

    date: datetime.date
    excess_bps: float
    turnover: float
    net_bps_primary: float


@dataclasses.dataclass
class PrecomputedSession:
    """Precomputed per-session data for one universe's included session."""

    date: datetime.date
    benchmark_return: float
    r_full: np.ndarray  # length n_symbols, zero outside valid
    mild_target_full: np.ndarray  # length n_symbols
    aggressive_target_full: np.ndarray  # length n_symbols
    raw_return_all: np.ndarray  # length n_symbols, NaN where not finite


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


def precompute_sessions(
    checkpoint_panel: Panel,
    checkpoint_feature,
    eligible_mask: np.ndarray,
    n_symbols: int,
) -> dict[str, list[PrecomputedSession]]:
    """Precompute per-session target weights, returns, and raw returns.

    Returns a dict mapping universe name -> list of PrecomputedSession, one per
    included session (sessions with n_valid < 5 are skipped).
    """
    close_field = checkpoint_panel.field("close")
    n_sessions = checkpoint_panel.n_days()

    sessions_by_universe: dict[str, list[PrecomputedSession]] = defaultdict(list)

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

        # Universe-independent raw return vector for drift (construction 1 only).
        raw_return_all = price_exit / price_entry - 1.0
        raw_return_all[
            ~(np.isfinite(price_entry) & (price_entry > 0) & np.isfinite(price_exit))
        ] = np.nan

        for universe in UNIVERSE_NAMES:
            if universe == "full":
                valid = base_valid
            else:
                valid = base_valid & eligible_mask

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

            # Full-length vectors.
            r_full = np.zeros(n_symbols, dtype=np.float64)
            r_full[valid] = r

            mild_target_full = np.zeros(n_symbols, dtype=np.float64)
            mild_target_full[valid] = mild_weights_valid

            aggressive_target_full = np.zeros(n_symbols, dtype=np.float64)
            aggressive_target_full[valid] = aggressive_weights_valid

            sessions_by_universe[universe].append(
                PrecomputedSession(
                    date=date,
                    benchmark_return=benchmark_return,
                    r_full=r_full,
                    mild_target_full=mild_target_full,
                    aggressive_target_full=aggressive_target_full,
                    raw_return_all=raw_return_all.copy(),
                )
            )

    return dict(sessions_by_universe)


def simulate_periodic(
    sessions: list[PrecomputedSession],
    tilt: str,
    k: int,
    cost_bps_primary: float,
) -> list[SessionRecord]:
    """Simulate periodic rebalancing with parameter k.

    Rebalance iff ``j % k == 0`` (j = 0-indexed position in the included-session
    list). On a rebalance session the book is set to that session's target with
    a real turnover charge; on a hold session no trade occurs (turnover = 0,
    cost = 0) and the book drifts organically using that session's own realized
    per-name return.

    Bugfix note: an earlier draft of this function drifted the carried-forward
    book after EVERY session, including rebalance sessions. That contaminated
    the k=1 control with a spurious one-session drift, so it no longer matched
    the baseline's turnover (which never models drift at all -- it simply diffs
    consecutive raw target-weight vectors). Fixed by resetting the carried
    book to the raw ``target`` (no drift) whenever the session just processed
    was itself a rebalance session; drift now only accumulates across genuine
    hold-session runs, compounding day-by-day.
    """
    n_symbols = len(sessions[0].r_full)
    book = np.zeros(n_symbols, dtype=np.float64)
    records: list[SessionRecord] = []

    for j, sess in enumerate(sessions):
        target = (
            sess.mild_target_full if tilt == "mild" else sess.aggressive_target_full
        )
        r_j = sess.r_full

        if j % k == 0:
            book_start = target.copy()
            turnover = float(np.sum(np.abs(book_start - book)))
        else:
            book_start = book.copy()
            turnover = 0.0

        return_j = float(np.dot(book_start, r_j))
        excess_bps = (return_j - sess.benchmark_return) * 10_000.0
        net_bps = excess_bps - cost_bps_primary * turnover

        records.append(
            SessionRecord(
                date=sess.date,
                excess_bps=excess_bps,
                turnover=turnover,
                net_bps_primary=net_bps,
            )
        )

        # Drift for next session. A rebalance session defines the carried-forward
        # book to be exactly `target`, with NO drift applied for that session --
        # this matches the baseline, which never tracks intraday drift at all, and
        # is required for the k=1 control to reproduce the baseline's turnover
        # exactly. Drift only accumulates across actual HOLD sessions, compounding
        # day-by-day using each hold session's own return.
        if j % k == 0:
            book = target.copy()
        else:
            raw = sess.raw_return_all.copy()
            held_mask = book_start != 0.0
            raw_filled = np.where(held_mask & np.isnan(raw), 0.0, raw)
            raw_filled = np.where(np.isnan(raw_filled), 0.0, raw_filled)
            denom = 1.0 + float(np.dot(book_start, raw_filled))
            if denom > 1e-6:
                book = book_start * (1.0 + raw_filled) / denom
            else:
                book = book_start.copy()

    return records


def simulate_hysteresis(
    sessions: list[PrecomputedSession],
    tilt: str,
    b: float,
    cost_bps_primary: float,
) -> list[SessionRecord]:
    """Simulate hysteresis / no-trade band with parameter b.

    Rebalances daily, but a name only trades toward its target if
    ``|target - book| > b * avg_target`` where ``avg_target`` is the mean
    target weight over names with a strictly positive target weight that
    session (symmetric between the mild and aggressive tilts). No organic
    between-session price drift is modeled for this construction (kept
    simpler than construction 1 by design).
    """
    n_symbols = len(sessions[0].r_full)
    book = np.zeros(n_symbols, dtype=np.float64)
    records: list[SessionRecord] = []

    for sess in sessions:
        target = (
            sess.mild_target_full if tilt == "mild" else sess.aggressive_target_full
        )
        r_j = sess.r_full

        n_nonzero = int(np.count_nonzero(target))
        avg_target = 1.0 / n_nonzero if n_nonzero > 0 else 0.0
        threshold = b * avg_target

        trade_mask = np.abs(target - book) > threshold
        candidate = np.where(trade_mask, target, book)
        candidate_sum = float(candidate.sum())
        if candidate_sum > 1e-9:
            book_start = candidate / candidate_sum
        else:
            book_start = target.copy()

        turnover = float(np.sum(np.abs(book_start - book)))
        return_j = float(np.dot(book_start, r_j))
        excess_bps = (return_j - sess.benchmark_return) * 10_000.0
        net_bps = excess_bps - cost_bps_primary * turnover

        records.append(
            SessionRecord(
                date=sess.date,
                excess_bps=excess_bps,
                turnover=turnover,
                net_bps_primary=net_bps,
            )
        )

        book = book_start.copy()

    return records


def simulate_smoothing(
    sessions: list[PrecomputedSession],
    tilt: str,
    a: float,
    cost_bps_primary: float,
) -> list[SessionRecord]:
    """Simulate weight smoothing with parameter a.

    Rebalances daily: ``book_start = (1 - a) * book + a * target``. On the
    first included session ``book`` is all-zero, so ``book_start = a *
    target`` -- a partial buy-in scaled by a, which is intended behaviour, not
    a bug. The convex combination always sums to 1.0 (up to float error), so
    no renormalization is needed.
    """
    n_symbols = len(sessions[0].r_full)
    book = np.zeros(n_symbols, dtype=np.float64)
    records: list[SessionRecord] = []

    for sess in sessions:
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
            SessionRecord(
                date=sess.date,
                excess_bps=excess_bps,
                turnover=turnover,
                net_bps_primary=net_bps,
            )
        )

        book = book_start.copy()

    return records


def aggregate_by_year(
    records: list[SessionRecord],
) -> dict[int, dict[str, float]]:
    """Aggregate a list of SessionRecord by calendar year."""
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
        }
    return result


def pooled_stats(records: list[SessionRecord]) -> dict[str, float]:
    """Compute pooled (whole-window) statistics for a list of SessionRecord."""
    n = len(records)
    if n == 0:
        return {
            "n_sessions": 0.0,
            "mean_gross_bps": float("nan"),
            "mean_turnover": float("nan"),
            "mean_net_bps": float("nan"),
            "ann_net_pct": float("nan"),
            "t_stat": float("nan"),
        }

    excess = np.array([r.excess_bps for r in records], dtype=np.float64)
    turnover = np.array([r.turnover for r in records], dtype=np.float64)
    net = np.array([r.net_bps_primary for r in records], dtype=np.float64)

    mean_excess = float(np.mean(excess))
    mean_turnover = float(np.mean(turnover))
    mean_net = float(np.mean(net))
    ann_net_pct = mean_net * 252.0 / 100.0

    if n >= 2:
        t_stat = float(mean_excess / (np.std(excess, ddof=1) / np.sqrt(n)))
    else:
        t_stat = float("nan")

    return {
        "n_sessions": float(n),
        "mean_gross_bps": mean_excess,
        "mean_turnover": mean_turnover,
        "mean_net_bps": mean_net,
        "ann_net_pct": ann_net_pct,
        "t_stat": t_stat,
    }


def print_compact_table(
    tilt: str,
    universe: str,
    results: dict[tuple[str, float], list[SessionRecord]],
) -> None:
    """Print the compact summary table for one (tilt, universe) combination."""
    print(f"\n=== {tilt} tilt, {universe} universe ===", flush=True)
    header = (
        "construction        param   gross_bps   turnover   NET_bps   ann_net_%   "
        "t_stat   2019    2021    2023    2024    2025"
    )
    print(header, flush=True)

    # Periodic
    for k in PERIODIC_KS:
        records = results.get(("periodic", float(k)), [])
        stats = pooled_stats(records)
        by_year = aggregate_by_year(records)
        year_vals = []
        for year in SUMMARY_YEARS:
            if year in by_year:
                year_vals.append(f"{by_year[year]['net_bps_primary']:8.2f}")
            else:
                year_vals.append(f"{'NaN':>8}")
        print(
            f"periodic            k={k:<5} {stats['mean_gross_bps']:9.2f} "
            f"{stats['mean_turnover']:9.4f} {stats['mean_net_bps']:9.2f} "
            f"{stats['ann_net_pct']:9.2f}  {stats['t_stat']:6.2f}  {'  '.join(year_vals)}",
            flush=True,
        )

    # Hysteresis
    for b in HYSTERESIS_BS:
        records = results.get(("hysteresis", float(b)), [])
        stats = pooled_stats(records)
        by_year = aggregate_by_year(records)
        year_vals = []
        for year in SUMMARY_YEARS:
            if year in by_year:
                year_vals.append(f"{by_year[year]['net_bps_primary']:8.2f}")
            else:
                year_vals.append(f"{'NaN':>8}")
        print(
            f"hysteresis          b={b:<5.2f} {stats['mean_gross_bps']:9.2f} "
            f"{stats['mean_turnover']:9.4f} {stats['mean_net_bps']:9.2f} "
            f"{stats['ann_net_pct']:9.2f}  {stats['t_stat']:6.2f}  {'  '.join(year_vals)}",
            flush=True,
        )

    # Smoothing
    for a in SMOOTHING_AS:
        records = results.get(("smoothing", float(a)), [])
        stats = pooled_stats(records)
        by_year = aggregate_by_year(records)
        year_vals = []
        for year in SUMMARY_YEARS:
            if year in by_year:
                year_vals.append(f"{by_year[year]['net_bps_primary']:8.2f}")
            else:
                year_vals.append(f"{'NaN':>8}")
        print(
            f"smoothing           a={a:<5.2f} {stats['mean_gross_bps']:9.2f} "
            f"{stats['mean_turnover']:9.4f} {stats['mean_net_bps']:9.2f} "
            f"{stats['ann_net_pct']:9.2f}  {stats['t_stat']:6.2f}  {'  '.join(year_vals)}",
            flush=True,
        )


def print_full_year_tables(
    tilt: str,
    universe: str,
    results: dict[tuple[str, float], list[SessionRecord]],
) -> None:
    """Print full 2018-2025 per-year net_bps tables for all constructions."""
    print(f"\n--- Full per-year net_bps: {tilt} tilt, {universe} universe ---", flush=True)
    print(
        "construction        param  2018    2019    2020    2021    2022    2023    2024    2025",
        flush=True,
    )

    all_years = list(range(2018, 2026))

    for construction, params in (
        ("periodic", PERIODIC_KS),
        ("hysteresis", HYSTERESIS_BS),
        ("smoothing", SMOOTHING_AS),
    ):
        for param in params:
            records = results.get((construction, float(param)), [])
            by_year = aggregate_by_year(records)
            year_vals = []
            for year in all_years:
                if year in by_year:
                    year_vals.append(f"{by_year[year]['net_bps_primary']:8.2f}")
                else:
                    year_vals.append(f"{'NaN':>8}")
            if construction == "periodic":
                label = f"periodic            k={param:<5}"
            elif construction == "hysteresis":
                label = f"hysteresis          b={param:<5.2f}"
            else:
                label = f"smoothing           a={param:<5.2f}"
            print(f"{label} {'  '.join(year_vals)}", flush=True)


def print_honest_summary(
    tilt: str,
    universe: str,
    results: dict[tuple[str, float], list[SessionRecord]],
) -> None:
    """Print 2024/2025 net_bps with positive/negative words."""
    print(f"\n--- Honest summary: {tilt} tilt, {universe} universe ---", flush=True)
    for construction, params in (
        ("periodic", PERIODIC_KS),
        ("hysteresis", HYSTERESIS_BS),
        ("smoothing", SMOOTHING_AS),
    ):
        for param in params:
            records = results.get((construction, float(param)), [])
            by_year = aggregate_by_year(records)
            recent = []
            for year in (2024, 2025):
                if year in by_year:
                    net = by_year[year]["net_bps_primary"]
                    word = "positive" if net > 0 else "negative"
                    recent.append(f"{year}: {word} ({net:.2f} bps/day)")
                else:
                    recent.append(f"{year}: no data")
            if construction == "periodic":
                label = f"periodic k={param}"
            elif construction == "hysteresis":
                label = f"hysteresis b={param:.2f}"
            else:
                label = f"smoothing a={param:.2f}"
            print(f"  {label:<25} {', '.join(recent)}", flush=True)


def print_best_construction(
    all_results: dict[tuple[str, str, str, float], list[SessionRecord]],
    cost_bps_primary: float,
) -> None:
    """Find and print the single best construction across all combinations."""
    best_key = None
    best_mean_net = -np.inf
    best_stats = None

    for key, records in all_results.items():
        stats = pooled_stats(records)
        if stats["mean_net_bps"] > best_mean_net:
            best_mean_net = stats["mean_net_bps"]
            best_key = key
            best_stats = stats

    if best_key is None:
        print("\n=== Best construction: no data ===", flush=True)
        return

    tilt, universe, construction, param = best_key
    breakeven_turnover = best_stats["mean_gross_bps"] / cost_bps_primary

    print("\n=== Best construction (highest pooled mean_net_bps) ===", flush=True)
    print(
        f"  {tilt} tilt, {universe} universe, {construction} param={param}",
        flush=True,
    )
    print(
        f"  Pooled net: {best_stats['mean_net_bps']:.2f} bps/day "
        f"({best_stats['ann_net_pct']:.2f}% annualized)",
        flush=True,
    )
    print(
        f"  Breakeven turnover: {breakeven_turnover:.4f} (turnover/day at which "
        f"net would hit exactly zero given its actual gross of "
        f"{best_stats['mean_gross_bps']:.2f} bps/day and primary-clip cost of "
        f"{cost_bps_primary:.2f} bps per unit turnover). This is the turnover "
        f"headroom the construction has before cost erases the edge.",
        flush=True,
    )


def main() -> None:
    """Run the full low-turnover-tilt reconstruction and print the report."""
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

    print("Precomputing session data...", flush=True)
    sessions_by_universe = precompute_sessions(
        checkpoint_panel,
        checkpoint_feature,
        cont_mask,
        panel.n_symbols(),
    )

    costs = NSEIntradayEquityCosts()
    cost_bps_primary = costs.round_trip_bps(PRIMARY_CLIP)

    # all_results maps (tilt, universe, construction, param) -> list[SessionRecord]
    all_results: dict[tuple[str, str, str, float], list[SessionRecord]] = {}

    print("Simulating constructions...", flush=True)
    for tilt in TILT_NAMES:
        for universe in UNIVERSE_NAMES:
            sessions = sessions_by_universe.get(universe, [])
            if not sessions:
                continue

            for k in PERIODIC_KS:
                records = simulate_periodic(sessions, tilt, k, cost_bps_primary)
                all_results[(tilt, universe, "periodic", float(k))] = records

            for b in HYSTERESIS_BS:
                records = simulate_hysteresis(sessions, tilt, b, cost_bps_primary)
                all_results[(tilt, universe, "hysteresis", float(b))] = records

            for a in SMOOTHING_AS:
                records = simulate_smoothing(sessions, tilt, a, cost_bps_primary)
                all_results[(tilt, universe, "smoothing", float(a))] = records

    # Print reports for each (tilt, universe) combination.
    for tilt in TILT_NAMES:
        for universe in UNIVERSE_NAMES:
            results = {
                key[2:]: records
                for key, records in all_results.items()
                if key[0] == tilt and key[1] == universe
            }
            if not results:
                continue
            print_compact_table(tilt, universe, results)
            print_full_year_tables(tilt, universe, results)
            print_honest_summary(tilt, universe, results)

    print_best_construction(all_results, cost_bps_primary)


if __name__ == "__main__":
    main()
