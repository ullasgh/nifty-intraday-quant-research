#!/usr/bin/env python3
"""Recon: run Lens criterion 4 (liquidity / time-of-day concentration) on the TILT's signal.

WHY THIS EXISTS. `results/RESEARCH_REPORT.md` lists criterion 4 of the Lens verdict --
"the edge is NOT concentrated in the bottom liquidity decile or a single time-of-day" --
as NOT RUN for the low-turnover index-relative tilt, the repo's only live candidate. The
machinery has existed and been tested for months (`research/lens.py:537-600` builds
`StabilityReport.by_liquidity_decile`; `lens.py:703-706` consumes it) but had never been
executed on the tilt's own signal. This script closes that gap, and is committed so that
every number published from it is reproducible -- this program has already had to redo
Newey-West figures once because they came from a scratchpad script that was deleted.

WHAT THE TILT'S SIGNAL IS. `research/tilt.py` imports `build_overnight_feature` from
`research/hypotheses/h2_overnight_reversal.py`: the tilt is a long-only, index-relative
weighting of H2's overnight-reversal feature. So the correct criterion-4 path is exactly
H2's path -- the two-rows-per-session CHECKPOINT panel (09:16 entry / 15:20 exit) with
`horizon=1` -- and NOT the raw 1-minute panel. That matters for more than tidiness:
`lens.CONCENTRATION_RATIO_THRESHOLD = 4.8695` was calibrated on the checkpoint-panel path
with the real `compute_prior_adv`, and an earlier calibration run on the raw minute panel
with an expanding-mean liquidity proxy produced 5.5484, which was too permissive. Using
the threshold off its calibrated geometry would repeat that error.

THREE SECTIONS ARE REPORTED, and they answer different questions:

  A. LENS CRITERION 4 (primary; this is the gated one). Per liquidity decile, the
     cross-sectional 5-bucket expectancy SPREAD of the signal (top-quintile minus
     bottom-quintile forward session return). This is what `Lens.verdict()` feeds to
     criterion 4, and the only quantity the measured 4.8695 null applies to.

  B. BOOK ACTIVE-RETURN ATTRIBUTION. The a=0.10 smoothed mild tilt book's realised
     gross excess, split by the SAME prior-ADV deciles. Per session i and decile d,

         contribution_d = 1e4 * sum_{j in d} (w_book[i,j] - w_bench[i,j]) * r[i,j]

     with `w_bench = valid / n_valid` the equal-weight benchmark. By construction the ten
     decile contributions sum EXACTLY to the session's gross excess in bps, so this is an
     identity, not an approximation -- it says where the realised rupees actually came
     from, which the bucket-spread of (A) does not directly measure.

  C. MEASURED NULL FOR (B). Section B originally reported "the bottom two deciles supply
     33.0% against a 20% even-spread expectation" with NO verdict attached, because no
     measured null existed for it and the flat 20% is an intuition, not a measurement.
     Section C supplies the null -- the SAME within-session permutation construction that
     calibrated 4.8695, applied to the decile LABEL so the book's total gross excess and
     every decile's size stay invariant -- and adjudicates the observed share against its
     p95. It also prints a mechanism diagnostic (C2) separating "the book deviates more
     in illiquid names" from "illiquid names' returns disperse more", because the two
     have different remedies.

  A AND C CAN DISAGREE, and here they do. (A) asks whether the SIGNAL's bucket spread is
  concentrated; (C) asks where the realised RUPEES came from. (A) passes at 2.7596 against
  4.8695 while (C) fires by more than five null standard deviations, which says the
  spread-based criterion is the less sensitive instrument for this particular concern --
  not that the concern is absent.

RULE 8. No threshold in this script is hand-chosen. Two cutoffs are applied.
`lens.CONCENTRATION_RATIO_THRESHOLD` (4.8695) governs section A; its derivation is at
`research/lens.py:24-50` (300 within-session permutation replicates, seed 42, on the true
production code path; measured bottom-is-argmax rate under the null 10.7%).
`BOTTOM_K_SHARE_THRESHOLDS` governs section C; its derivation is recorded beside the value
below (10,000 replicates, seed 42, same construction, measured on the arrays section B
actually attributes). main() re-measures the section-C null on every run and refuses to
finish if the recorded values do not reproduce.

HOLDOUT. The window runs 2018-01-01..2025-07-31 (`recon_low_turnover_tilt.START/END`).
The locked holdout [2025-08-14, 2026-08-14] is never loaded. The script prints the
`HoldoutLock` read count at start and at end; they must be equal.

Run:
    NQ_CACHE_ROOT=/some/scratch/dir .venv/bin/python \
        scripts/recon_tilt_liquidity_decomposition.py

(~3 min: full 1-minute panel load, a 10-decile x 5-bucket block bootstrap, then two
10,000-replicate permutation nulls at ~8 ms/replicate.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon_low_turnover_tilt as base  # noqa: E402
import recon_tilt_liquidity as liq  # noqa: E402

from nifty_quant.data.liquidity import compute_prior_adv  # noqa: E402
from nifty_quant.execution.costs import NSEIntradayEquityCosts  # noqa: E402
from nifty_quant.research import expectancy  # noqa: E402
from nifty_quant.research.hypotheses.h2_overnight_reversal import (  # noqa: E402
    _build_checkpoint_panel,
    build_overnight_feature,
)
from nifty_quant.research.lens import CONCENTRATION_RATIO_THRESHOLD, Lens  # noqa: E402
from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path  # noqa: E402
from nifty_quant.universe.static import load_universe  # noqa: E402

# Explicit and pre-registered. SEED feeds the block bootstrap inside
# conditional_expectancy; 0 is Lens's own default, chosen here so the numbers match a
# plain `Lens(panel)` construction rather than a private variant of it.
SEED = 0
HORIZON = 1  # one checkpoint row ahead == entry-to-exit, see _build_checkpoint_panel
TILT = "mild"
SMOOTHING_A = 0.10
UNIVERSE = "all_equity"
N_DECILES = 10

# A decile whose expectancy table is built from fewer than this many defined session-rows
# is reported but marked THIN. This is a REPORTING annotation, not a cutoff that changes
# any verdict, so rule 8 does not require a measured null for it: 30 is the repo's
# existing thin-cell convention (`expectancy.double_sort(thin_cell_threshold=30)`).
THIN_ROWS = 30

# --- Section C: measured null for the bottom-k share of realised gross excess -------
# Pre-registered and explicit. NULL_SEED is 42, matching the seed used for the
# CONCENTRATION_RATIO_THRESHOLD calibration (lens.py:24-50), so the two measured nulls in
# this file are drawn from the same declared seed rather than one chosen per statistic.
NULL_SEED = 42
N_NULL_REPLICATES = 10_000  # >= 300 as required; this null costs ~8 ms/replicate
BOTTOM_K_VALUES = (1, 2, 3)
NULL_PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0)
# The repo's existing convention for turning a measured null into a cutoff: the 95th
# percentile. Confirmed from lens.py:35 -- CONCENTRATION_RATIO_THRESHOLD = 4.8695 is the
# p95 entry of that grid (p50 2.4290 / p75 3.4406 / p90 4.2849 / p95 4.8695 / p99 6.7422).
THRESHOLD_PERCENTILE = 95.0

# Derived from a measured null distribution, per CLAUDE.md rule 8.
#
# 10,000 within-session permutation replicates (seed 42) of the SAME null construction
# already validated for lens.CONCENTRATION_RATIO_THRESHOLD (lens.py:24-50): one column
# permutation per session, applied identically to every row of that session. Here the
# book's attribution has exactly one row per session and it is the LIQUIDITY DECILE LABEL
# that is permuted across symbols, so every symbol keeps its own realised active weight,
# its own return and therefore its own realised contribution -- only its liquidity rank is
# scrambled. That destroys the symbol-liquidity association while preserving EXACTLY (not
# approximately): the book's total gross excess (a bijection on the columns leaves the
# denominator invariant), every decile's size, the per-session cross-sectional
# distribution of contributions, and the residual on unlabelled columns.
#
# Measured on the TRUE PRODUCTION PATH -- the a=0.10 smoothed mild tilt book simulated by
# recon_tilt_liquidity.simulate_smoothing_with_books over 1,856 sessions, with deciles from
# the real compute_prior_adv on the two-rows-per-session checkpoint panel. This is the same
# geometry lesson lens.py records: 5.5484 was wrong because it was calibrated on the full
# 1-minute panel with an expanding-mean liquidity proxy, so this null is deliberately
# measured on the arrays section B actually attributes, not on a convenient stand-in.
#
#     n=10000, share of gross excess supplied by the bottom-k ADV deciles, in %:
#       k=1:  p50 10.3399  p75 11.4874  p90 12.5299  p95 13.1044  p99 14.2547  max 16.6738
#       k=2:  p50 20.1018  p75 21.6246  p90 22.9740  p95 23.8221  p99 25.2949  max 28.2646
#       k=3:  p50 30.2160  p75 31.8639  p90 33.3575  p95 34.2157  p99 35.9981  max 39.2505
#
# The null CENTRES ON THE EVEN-SPREAD VALUE: p50 is 10.34 / 20.10 / 30.22 against 10 / 20 /
# 30. The prior worry that gross-excess contribution might be far from uniform under the
# null even when liquidity is irrelevant is NOT borne out here -- it is uniform in
# expectation, and merely has a measurable spread (sd 2.24pp at k=2) that the flat-20%
# intuition could not have supplied. The threshold is what that intuition was missing.
#
# A tradability-stratified variant (permuting only within {labelled, tradable} and
# {labelled, not tradable}, which additionally holds each decile's tradable-name count
# fixed) gives p95 13.0661 / 23.8054 / 34.1893 -- indistinguishable. The verdict therefore
# does not rest on how the permutation treats non-tradable names.
#
# p95 is the repo's existing convention for turning a measured null into a cutoff
# (lens.py:35: CONCENTRATION_RATIO_THRESHOLD = 4.8695 is the p95 entry of its grid).
#
# OBSERVED for the tilt: 17.7939 / 32.9505 / 43.8640 %. All three sit above p99 and above
# the MAXIMUM of all 10,000 replicates (n_ge = 0, empirical p < 1e-4), at 4.36 / 5.75 /
# 5.43 null standard deviations. main() re-measures this null on every run and asserts the
# values below reproduce, so the constants cannot silently drift from their derivation.
BOTTOM_K_SHARE_THRESHOLDS: dict[int, float] = {
    1: 0.131044,
    2: 0.238221,
    3: 0.342157,
}


def _fmt(value: float | None, width: int = 10, places: int = 4) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:{width}.{places}f}"


def report_lens_criterion_4(stab) -> tuple[str, list[str], dict[int, float]]:
    """Print the Lens by-liquidity-decile and by-time-of-day tables; return the verdict.

    Reimplements `Lens.verdict()`'s criterion-4 comparison line for line (lens.py:703-777)
    so the intermediate ratio is visible instead of collapsed into a PASS/FAIL string. The
    comparison itself is NOT re-derived here -- it and its threshold come from lens.py.
    """
    print("\n" + "=" * 78)
    print("A. LENS CRITERION 4 -- by liquidity decile (0 = LEAST liquid)")
    print("=" * 78)
    print(
        f"{'decile':>6} {'spread_bps':>12} {'spread_t':>10} {'n_rows':>9} "
        f"{'n_obs':>11} {'note':>8}"
    )

    decile_spreads: dict[int, float] = {}
    for decile in range(N_DECILES):
        table = stab.by_liquidity_decile.get(decile)
        if table is None:
            print(f"{decile:>6} {'ABSENT (no rows assigned to this decile)':>55}")
            continue
        n_obs = int(sum(b.n_obs for b in table.buckets))
        note = "THIN" if table.n_total < THIN_ROWS else ""
        decile_spreads[decile] = float(table.spread_bps)
        print(
            f"{decile:>6} {_fmt(table.spread_bps, 12)} {_fmt(table.spread_t, 10)} "
            f"{table.n_total:>9} {n_obs:>11} {note:>8}"
        )

    print("\n" + "-" * 78)
    print("A2. LENS CRITERION 4 -- by time-of-day bucket")
    print("-" * 78)
    print(f"{'bucket':>10} {'spread_bps':>12} {'spread_t':>10} {'n_rows':>9} {'note':>8}")
    tod_spreads: dict[str, float] = {}
    for name, table in stab.by_time_of_day.items():
        tod_spreads[name] = float(table.spread_bps)
        note = "VACUOUS" if table.n_total == 0 else ""
        print(
            f"{name:>10} {_fmt(table.spread_bps, 12)} {_fmt(table.spread_t, 10)} "
            f"{table.n_total:>9} {note:>8}"
        )
    print(
        "  NOTE: on the two-row checkpoint panel the EXIT row's horizon=1 forward return\n"
        "  runs off the session end and is NaN by construction, so only the entry-time\n"
        "  bucket can ever carry a defined edge. The time-of-day leg of criterion 4 is\n"
        "  therefore STRUCTURALLY VACUOUS on this path -- it cannot fail, and its PASS\n"
        "  must not be read as evidence the edge is spread across the session."
    )

    # --- criterion-4 comparison, mirroring lens.py:703-777 -------------------------
    reasons: list[str] = []
    result = "PASS"

    bottom = decile_spreads.get(0)
    print("\n" + "-" * 78)
    print("A3. Concentration comparison (threshold from lens.py, measured null)")
    print("-" * 78)
    print(f"  CONCENTRATION_RATIO_THRESHOLD = {CONCENTRATION_RATIO_THRESHOLD:.4f}")
    print("  (95th percentile of a 300-replicate within-session permutation null,")
    print("   seed 42, measured on this same checkpoint-panel path -- lens.py:26-56)")

    if bottom is None or bottom == 0.0:
        print("  bottom decile absent or exactly zero -> liquidity leg cannot fire")
    else:
        nonzero = [e for e in decile_spreads.values() if e != 0.0]
        if len(nonzero) == 1:
            result = "FAIL"
            reasons.append("concentrated in bottom liquidity decile (only decile with edge)")
            print("  only ONE decile carries any edge, and it is the bottom decile")
        else:
            ordered = sorted(abs(e) for e in nonzero)
            max_edge = ordered[-1]
            median_edge = ordered[len(ordered) // 2]
            ratio = max_edge / median_edge if median_edge > 0 else float("inf")
            bottom_is_argmax = abs(bottom) == max_edge
            print(f"  bottom decile spread      = {bottom:.4f} bps")
            print(f"  max |spread| over deciles = {max_edge:.4f} bps")
            print(f"  median |spread|           = {median_edge:.4f} bps")
            print(f"  observed ratio max/median = {ratio:.4f}")
            print(f"  bottom decile is argmax   = {bottom_is_argmax}")
            if (
                median_edge > 0
                and max_edge > median_edge * CONCENTRATION_RATIO_THRESHOLD
                and bottom_is_argmax
            ):
                result = "FAIL"
                reasons.append("concentrated in bottom liquidity decile")

    if stab.by_time_of_day:
        if len(stab.by_time_of_day) == 1:
            result = "FAIL"
            reasons.append("concentrated in single time-of-day bucket")
            print("\n  only ONE time-of-day bucket exists -> time-of-day leg FAILS")
        else:
            edges = [e for e in tod_spreads.values() if e != 0.0]
            if edges:
                same_sign = all((e > 0) == (edges[0] > 0) for e in edges)
                if not same_sign:
                    result = "FAIL"
                    reasons.append("time-of-day buckets disagree in sign")
                    print("\n  time-of-day buckets have MIXED signs -> FAIL")
                else:
                    ordered = sorted(abs(e) for e in edges)
                    if len(ordered) > 1:
                        max_e = ordered[-1]
                        med_e = ordered[len(ordered) // 2]
                        ratio = max_e / med_e if med_e > 0 else float("inf")
                        print(f"\n  time-of-day ratio max/median = {ratio:.4f}")
                        if med_e > 0 and max_e > med_e * CONCENTRATION_RATIO_THRESHOLD:
                            result = "FAIL"
                            reasons.append("concentrated in single time-of-day bucket")

    return result, reasons, decile_spreads


def report_book_attribution(
    checkpoint_panel,
    checkpoint_feature,
    decile_labels_entry: np.ndarray,
    sessions_full: list,
    n_symbols: int,
    cost_bps_primary: float,
) -> BookAttribution:
    """Print the exact per-decile split of the tilt book's realised gross excess.

    Returns the per-decile mean bps/session of ACTIVE contribution together with the
    (n_sessions, n_symbols) cell-level arrays section C permutes. The per-decile values
    sum to the book's mean gross excess in bps (up to float64 rounding); that identity
    is asserted below rather than assumed.
    """
    records, book_matrix, r_matrix = liq.simulate_smoothing_with_books(
        sessions_full, TILT, SMOOTHING_A, cost_bps_primary
    )
    valid_matrix, n_valid_array = liq.recover_valid_and_n_valid(
        sessions_full, checkpoint_panel, checkpoint_feature, n_symbols
    )

    n_sessions = len(sessions_full)
    bench_w = np.zeros((n_sessions, n_symbols), dtype=np.float64)
    nonzero = n_valid_array > 0
    bench_w[nonzero] = valid_matrix[nonzero] / n_valid_array[nonzero, None]

    active_w = book_matrix - bench_w
    # r_full carries 0.0 for symbols that were not valid that session (base's own
    # convention), so a non-valid symbol contributes exactly nothing either way.
    contrib = active_w * r_matrix * 1e4  # (n_sessions, n_symbols) bps

    # Decile label for each session/symbol, taken at that session's ENTRY row.
    date_to_cp_idx = {d: i for i, d in enumerate(checkpoint_panel.dates)}
    labels = np.full((n_sessions, n_symbols), -1, dtype=np.int64)
    for i, sess in enumerate(sessions_full):
        labels[i, :] = decile_labels_entry[2 * date_to_cp_idx[sess.date], :]

    print("\n" + "=" * 78)
    print("B. BOOK ACTIVE-RETURN ATTRIBUTION by the SAME prior-ADV deciles")
    print("   (mean bps/session; the ten rows sum to the book's gross excess)")
    print("=" * 78)
    print(
        f"{'decile':>6} {'contrib_bps':>13} {'share_%':>9} {'mean_wt':>9} "
        f"{'bench_wt':>9} {'names/sess':>11}"
    )

    per_decile: dict[int, float] = {}
    total = 0.0
    gross_total = float(np.mean([r.excess_bps for r in records]))
    for decile in range(N_DECILES):
        mask = labels == decile
        mean_bps = float(np.sum(np.where(mask, contrib, 0.0), axis=1).mean())
        mean_wt = float(np.sum(np.where(mask, book_matrix, 0.0), axis=1).mean())
        mean_bw = float(np.sum(np.where(mask, bench_w, 0.0), axis=1).mean())
        names = float(np.sum(mask & valid_matrix, axis=1).mean())
        share = 100.0 * mean_bps / gross_total if gross_total != 0.0 else float("nan")
        per_decile[decile] = mean_bps
        total += mean_bps
        print(
            f"{decile:>6} {_fmt(mean_bps, 13)} {share:>9.1f} {_fmt(mean_wt, 9)} "
            f"{_fmt(mean_bw, 9)} {names:>11.1f}"
        )

    unassigned = float(np.sum(np.where(labels < 0, contrib, 0.0), axis=1).mean())
    print(f"{'none':>6} {_fmt(unassigned, 13)} {'':>9}   (symbols with no prior-ADV decile)")
    print("-" * 78)
    print(f"{'sum':>6} {_fmt(total + unassigned, 13)}")
    print(f"{'gross':>6} {_fmt(gross_total, 13)}   (mean excess_bps from the simulation)")
    identity_gap = abs(total + unassigned - gross_total)
    print(f"identity gap = {identity_gap:.3e} bps  (must be ~0; this is an identity)")
    assert identity_gap < 1e-6, f"attribution identity broken: gap={identity_gap}"
    print(f"n_sessions = {n_sessions}")
    return BookAttribution(
        per_decile=per_decile,
        contrib=contrib,
        labels=labels,
        valid=valid_matrix.astype(bool),
        active_w=active_w,
        returns=r_matrix.astype(np.float64),
        gross_total=gross_total,
    )


# ---------------------------------------------------------------------------
# Section C: measured null for the bottom-k share of realised gross excess
# ---------------------------------------------------------------------------


class BookAttribution(NamedTuple):
    """Everything section C needs from section B, so the identity is computed once."""

    per_decile: dict[int, float]
    contrib: np.ndarray  # (n_sessions, n_symbols) float64, bps, sums to gross_total
    labels: np.ndarray  # (n_sessions, n_symbols) int64, decile at that session's entry row
    valid: np.ndarray  # (n_sessions, n_symbols) bool, symbol tradable that session
    active_w: np.ndarray  # (n_sessions, n_symbols) float64, w_book - w_bench
    returns: np.ndarray  # (n_sessions, n_symbols) float64, realised session return
    gross_total: float  # mean bps/session of the book's realised gross excess


def bottom_k_share(contrib: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of TOTAL contribution supplied by deciles 0..k-1 (0 = least liquid).

    The denominator is the total over every cell, including cells with no decile
    label, so this is the same quantity section B prints as ``share_%`` summed over
    the bottom k rows -- not a renormalised within-labelled share.
    """
    selected = (labels >= 0) & (labels < k)
    total = float(np.sum(contrib, dtype=np.float64))
    if total == 0.0:
        return float("nan")
    return float(np.sum(np.where(selected, contrib, 0.0), dtype=np.float64) / total)


def _canonical_layout(
    labels: np.ndarray, strata: np.ndarray, ks: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Per-session masks over PERMUTED COLUMN RANKS for each bottom-k group.

    Columns are sorted per session by ``(stratum, label)``. Under a permutation that
    is uniform within each stratum, rank ``t`` of a stratum block holds a uniformly
    chosen column of that stratum, and the label it receives is the ``t``-th smallest
    label in the block -- so the fixed, precomputed mask below applied to the permuted
    contributions is exactly "sum the contributions that landed in deciles 0..k-1".
    """
    order = np.lexsort((labels, strata), axis=1)
    labels_canon = np.take_along_axis(labels, order, axis=1)
    return {k: ((labels_canon >= 0) & (labels_canon < k)) for k in ks}


def null_bottom_k_distribution(
    contrib: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    ks: tuple[int, ...],
    n_replicates: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Null distribution of ``bottom_k_share`` under within-session label permutation.

    The construction is the one already validated for
    ``lens.CONCENTRATION_RATIO_THRESHOLD`` (lens.py:24-50): ONE column permutation per
    session, applied identically to every row of that session. Here the book's
    attribution has exactly one row per session, and it is the LIQUIDITY LABEL that is
    permuted across symbols -- each symbol keeps its own realised active weight, its
    own return and therefore its own realised contribution, and only its liquidity
    rank is scrambled. That destroys the symbol-liquidity association while preserving,
    exactly and not approximately: the book's total gross excess (the permutation is a
    bijection on the columns, so the denominator is invariant), every decile's size,
    the per-session cross-sectional distribution of contributions, and the residual
    sitting on unlabelled columns.

    ``strata`` restricts the permutation: columns are only ever exchanged with columns
    in the same stratum. Passing a stratum that separates tradable from non-tradable
    columns additionally holds each decile's tradable-name count fixed.
    """
    masks = _canonical_layout(labels, strata, ks)
    total = float(np.sum(contrib, dtype=np.float64))
    keys_base = strata.astype(np.float64) * 2.0
    out = {k: np.empty(n_replicates, dtype=np.float64) for k in ks}
    for rep in range(n_replicates):
        order = np.argsort(keys_base + rng.random(contrib.shape), axis=1)
        permuted = np.take_along_axis(contrib, order, axis=1)
        for k in ks:
            out[k][rep] = float(np.sum(permuted * masks[k], dtype=np.float64) / total)
    return out


def report_bottom_k_null(attribution: BookAttribution) -> dict[str, dict[int, dict]]:
    """Measure the null for the bottom-k share and adjudicate the observed value.

    Two nulls are run. They differ ONLY in what the permutation is allowed to exchange:

      unconditional  -- any labelled column may take any other labelled column's decile.
                        This is the direct analogue of the construction already validated
                        for CONCENTRATION_RATIO_THRESHOLD, with the minimum of extra
                        conditioning, and it is the one the threshold is taken from.
      tradable-strat -- columns are exchanged only within {labelled and tradable} and
                        {labelled and not tradable}. This additionally holds each
                        decile's TRADABLE-NAME COUNT fixed, which matters because a
                        decile with few tradable names can only ever contribute little.
                        Reported as a sensitivity check: if the two disagree, the
                        conclusion is driven by coverage rather than by P&L.
    """
    contrib = attribution.contrib
    labels = attribution.labels
    labelled = labels >= 0

    strata_uncond = np.where(labelled, 0, 1).astype(np.int64)
    strata_strat = np.where(labelled & attribution.valid, 0, np.where(labelled, 1, 2)).astype(
        np.int64
    )

    observed = {k: bottom_k_share(contrib, labels, k) for k in BOTTOM_K_VALUES}
    even_spread = {k: k / N_DECILES for k in BOTTOM_K_VALUES}

    print("\n" + "=" * 78)
    print("C. MEASURED NULL for the bottom-k share of realised gross excess")
    print("=" * 78)
    print(
        f"  construction: one within-session permutation of the LIQUIDITY DECILE LABEL\n"
        f"  across symbols, per replicate -- every symbol keeps its own realised active\n"
        f"  weight, return and contribution; only its liquidity rank is scrambled. The\n"
        f"  book's total gross excess, each decile's size and the unlabelled residual are\n"
        f"  therefore invariant, so the share is well defined on every replicate.\n"
        f"  n_replicates={N_NULL_REPLICATES} seed={NULL_SEED} "
        f"threshold percentile=p{THRESHOLD_PERCENTILE:g}"
    )

    out: dict[str, dict[int, dict]] = {}
    variants = (("unconditional", strata_uncond), ("tradable-strat", strata_strat))
    for name, strata in variants:
        rng = np.random.default_rng(NULL_SEED)
        null = null_bottom_k_distribution(
            contrib, labels, strata, BOTTOM_K_VALUES, N_NULL_REPLICATES, rng
        )
        print("\n" + "-" * 78)
        print(f"  null: {name}")
        print("-" * 78)
        header = "".join(f"{'p' + f'{q:g}':>9}" for q in NULL_PERCENTILES)
        print(
            f"{'k':>2} {'observed':>9} {'even':>6}{header} {'max':>9} {'sd':>7} "
            f"{'z':>7} {'n_ge':>6} {'verdict':>13}"
        )
        out[name] = {}
        for k in BOTTOM_K_VALUES:
            draws = null[k]
            grid = [float(np.percentile(draws, q)) for q in NULL_PERCENTILES]
            cutoff = float(np.percentile(draws, THRESHOLD_PERCENTILE))
            obs = observed[k]
            sd = float(np.std(draws, ddof=1))
            centre = float(np.median(draws))
            n_ge = int(np.sum(draws >= obs))
            z = (obs - centre) / sd if sd > 0 else float("inf")
            verdict = "ABOVE p95" if obs > cutoff else "unremarkable"
            cells = "".join(f"{100.0 * g:>9.4f}" for g in grid)
            print(
                f"{k:>2} {100.0 * obs:>9.4f} {100.0 * even_spread[k]:>6.1f}{cells} "
                f"{100.0 * float(np.max(draws)):>9.4f} {100.0 * sd:>7.4f} {z:>7.2f} "
                f"{n_ge:>6} {verdict:>13}"
            )
            out[name][k] = {
                "observed": obs,
                "grid": grid,
                "cutoff": cutoff,
                "sd": sd,
                "z": z,
                "n_ge": n_ge,
                "fires": bool(obs > cutoff),
            }
        print(
            f"  n_ge = replicates with a bottom-k share >= the observed one, out of "
            f"{N_NULL_REPLICATES}; n_ge=0 means the empirical p-value is < "
            f"{1.0 / N_NULL_REPLICATES:.0e}."
        )

    _report_concentration_mechanism(attribution)
    return out


def _report_concentration_mechanism(attribution: BookAttribution) -> None:
    """Separate the two things that can make a decile's contribution large.

    A decile's contribution is sum(active_weight * return). It can be large because the
    book DEVIATES more there (a deliberate bet on illiquid names -- a capacity problem
    the portfolio construction could fix) or because RETURNS DISPERSE more there (the
    same even-handed active weight simply lands on noisier names -- a property of the
    universe, not of the construction). These have different remedies, so the null's
    verdict is reported alongside the evidence for which one is operating. Nothing here
    is a threshold; it is descriptive, and no PASS/FAIL reads it.
    """
    labels = attribution.labels
    valid = attribution.valid
    print("\n" + "-" * 78)
    print("  C2. mechanism: deviation size vs return dispersion, per decile")
    print("-" * 78)
    print(f"{'decile':>6} {'mean|active_w|':>15} {'sd(return)_bps':>16} {'names/sess':>11}")
    for decile in range(N_DECILES):
        cell = (labels == decile) & valid
        if not np.any(cell):
            continue
        mean_abs_w = float(np.mean(np.abs(attribution.active_w[cell])))
        sd_r = float(np.std(attribution.returns[cell])) * 1e4
        names = float(np.sum(cell, axis=1).mean())
        print(f"{decile:>6} {mean_abs_w:>15.6f} {sd_r:>16.2f} {names:>11.1f}")


def main() -> None:
    lock = HoldoutLock(default_holdout_lock_path())
    reads_before = lock.read_count()
    print(f"HoldoutLock read count BEFORE: {reads_before}", flush=True)
    print(f"Research window: {base.START} .. {base.END} (holdout never loaded)", flush=True)
    print(f"seed={SEED} horizon={HORIZON} tilt={TILT} a={SMOOTHING_A}", flush=True)

    symbols = load_universe(UNIVERSE).symbols
    spec = base.PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=symbols,
        start=base.START,
        end=base.END,
    )
    print(f"loading panel (1-minute, {len(symbols)} symbols)...", flush=True)
    panel = base.load_panel(spec, memmap=True)

    feature = build_overnight_feature(panel)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, feature)
    print(
        f"checkpoint panel: {checkpoint_panel.n_days()} sessions, "
        f"{checkpoint_panel.n_symbols()} symbols, "
        f"{len(checkpoint_panel.dates)} dates "
        f"[{checkpoint_panel.dates[0]} .. {checkpoint_panel.dates[-1]}]",
        flush=True,
    )

    print("running Lens.stability() ...", flush=True)
    lens = Lens(checkpoint_panel, seed=SEED)
    stab = lens.stability(checkpoint_feature, HORIZON)
    print(stab.explain(), flush=True)

    result, reasons, decile_spreads = report_lens_criterion_4(stab)

    # Section B reuses the SAME decile definition Lens used, so the two sections are
    # comparable: causal_buckets on compute_prior_adv over the checkpoint panel.
    prior_adv = compute_prior_adv(checkpoint_panel)
    bucketing = expectancy.causal_buckets(
        prior_adv,
        checkpoint_panel.day_offsets,
        n_buckets=N_DECILES,
        method="cross_sectional_rank",
    )

    cont_mask = base.continuous_coverage_mask(panel)
    sessions_by_universe = base.precompute_sessions(
        checkpoint_panel, checkpoint_feature, cont_mask, panel.n_symbols()
    )
    sessions_full = sessions_by_universe["full"]
    cost_bps_primary = NSEIntradayEquityCosts().round_trip_bps(base.PRIMARY_CLIP)
    attribution = report_book_attribution(
        checkpoint_panel,
        checkpoint_feature,
        bucketing.labels.astype(np.int64),
        sessions_full,
        panel.n_symbols(),
        cost_bps_primary,
    )
    book_decile = attribution.per_decile
    null_result = report_bottom_k_null(attribution)

    print("\n" + "=" * 78)
    print("CRITERION 4 VERDICT (Lens path, section A)")
    print("=" * 78)
    print(f"  result: {result}")
    for reason in reasons:
        print(f"  - {reason}")
    if not reasons:
        print("  - no concentration leg fired")
    thin = [d for d, t in stab.by_liquidity_decile.items() if t.n_total < THIN_ROWS]
    missing = [d for d in range(N_DECILES) if d not in stab.by_liquidity_decile]
    if thin or missing:
        print(f"  CAVEAT: thin deciles {thin}, absent deciles {missing}")
    print(
        f"  bottom-decile share of realised gross excess: "
        f"{book_decile.get(0, float('nan')):.4f} bps/session"
    )

    print("\n" + "=" * 78)
    print("CRITERION 4 VERDICT (book attribution, sections B+C)")
    print("=" * 78)
    for name, per_k in null_result.items():
        for k in BOTTOM_K_VALUES:
            rec = per_k[k]
            print(
                f"  [{name:>14}] bottom-{k}: observed {100.0 * rec['observed']:.2f}% "
                f"vs p{THRESHOLD_PERCENTILE:g} {100.0 * rec['cutoff']:.2f}% "
                f"-> {'FIRES' if rec['fires'] else 'no'} "
                f"({rec['z']:.2f} null sd, n_ge={rec['n_ge']}/{N_NULL_REPLICATES})"
            )
    for k, recorded in BOTTOM_K_SHARE_THRESHOLDS.items():
        measured = null_result["unconditional"][k]["cutoff"]
        if abs(measured - recorded) > 5e-7:
            raise AssertionError(
                f"BOTTOM_K_SHARE_THRESHOLDS[{k}] = {recorded} does not reproduce: this run "
                f"measured p{THRESHOLD_PERCENTILE:g} = {measured:.6f}. The recorded "
                "derivation and the code have diverged; fix one or the other, do not "
                "loosen this check."
            )
    print(
        f"  recorded thresholds reproduce: "
        f"{ {k: round(v, 6) for k, v in BOTTOM_K_SHARE_THRESHOLDS.items()} }"
    )
    headline = null_result["unconditional"][2]
    print(
        f"  HEADLINE (k=2, unconditional null): "
        f"{'CONCENTRATED' if headline['fires'] else 'NOT CONCENTRATED'} in the illiquid tail"
    )
    _ = decile_spreads

    reads_after = lock.read_count()
    print(f"\nHoldoutLock read count AFTER: {reads_after}", flush=True)
    if reads_after != reads_before:
        print("  *** HOLDOUT WAS READ -- STOP AND REPORT ***", flush=True)
    else:
        print("  holdout untouched (count unchanged)", flush=True)


if __name__ == "__main__":
    main()
