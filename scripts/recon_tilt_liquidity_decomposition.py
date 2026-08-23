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

TWO DECOMPOSITIONS ARE REPORTED, and they answer different questions:

  A. LENS CRITERION 4 (primary; this is the gated one). Per liquidity decile, the
     cross-sectional 5-bucket expectancy SPREAD of the signal (top-quintile minus
     bottom-quintile forward session return). This is what `Lens.verdict()` feeds to
     criterion 4, and the only quantity the measured 4.8695 null applies to.

  B. BOOK ACTIVE-RETURN ATTRIBUTION (secondary; no calibrated threshold exists for it,
     so it informs but does not decide). The a=0.10 smoothed mild tilt book's realised
     gross excess, split by the SAME prior-ADV deciles. Per session i and decile d,

         contribution_d = 1e4 * sum_{j in d} (w_book[i,j] - w_bench[i,j]) * r[i,j]

     with `w_bench = valid / n_valid` the equal-weight benchmark. By construction the ten
     decile contributions sum EXACTLY to the session's gross excess in bps, so this is an
     identity, not an approximation -- it says where the realised rupees actually came
     from, which the bucket-spread of (A) does not directly measure.

RULE 8. No threshold in this script is hand-chosen. The only cutoff applied is
`lens.CONCENTRATION_RATIO_THRESHOLD` (4.8695), whose derivation is recorded at
`research/lens.py:26-56` (300 within-session permutation replicates, seed 42, on the true
production code path; measured bottom-is-argmax rate under the null 10.7%). Section B has
no such measured null and therefore carries no PASS/FAIL of its own.

HOLDOUT. The window runs 2018-01-01..2025-07-31 (`recon_low_turnover_tilt.START/END`).
The locked holdout [2025-08-14, 2026-08-14] is never loaded. The script prints the
`HoldoutLock` read count at start and at end; they must be equal.

Run:
    NQ_CACHE_ROOT=/some/scratch/dir .venv/bin/python \
        scripts/recon_tilt_liquidity_decomposition.py

(~10-20 min: full 1-minute panel load, then a 10-decile x 5-bucket block bootstrap.)
"""

from __future__ import annotations

import sys
from pathlib import Path

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
) -> dict[int, float]:
    """Print the exact per-decile split of the tilt book's realised gross excess.

    Returns a mapping decile -> mean bps/session of ACTIVE contribution. The values sum
    to the book's mean gross excess in bps (up to float64 rounding); that identity is
    asserted below rather than assumed.
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
    return per_decile


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
    book_decile = report_book_attribution(
        checkpoint_panel,
        checkpoint_feature,
        bucketing.labels.astype(np.int64),
        sessions_full,
        panel.n_symbols(),
        cost_bps_primary,
    )

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
        "  section B (book attribution) is reported for context only: no measured null "
        "exists for it, so it carries no verdict of its own."
    )
    print(
        f"  bottom-decile share of realised gross excess: "
        f"{book_decile.get(0, float('nan')):.4f} bps/session"
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
