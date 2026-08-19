#!/usr/bin/env python3
"""Significance and plateau-grid analysis for the smoothing construction.

This script answers two questions about the smoothing construction (parameter
``a``) from ``scripts/recon_low_turnover_tilt.py``:

1. Is the recent-window (2024-01-01..2025-07-31) NET performance of the
   smoothing construction statistically distinguishable from zero at the 95%
   confidence level? The finding parameter ``a=0.10`` is compared against the
   daily-rebalance control ``a=1.0`` for all four (tilt, universe) combinations.

2. What does the pooled and recent-year performance look like across a finer
   grid of smoothing parameters ``a`` in ``(0.20, 0.15, 0.10, 0.07, 0.05,
   0.03, 0.02, 0.01)``? This is a descriptive plateau-vs-spike grid; no
   classification is printed by the script itself.

The script reuses ``scripts/recon_low_turnover_tilt.py`` as ``base``: it loads
the same panel, builds the same feature, precomputes the same sessions, and
calls ``base.simulate_smoothing``, ``base.pooled_stats``, and
``base.aggregate_by_year``. A correctness gate verifies that the ``a=1.0``
control reproduces the baseline script's pooled numbers before any results are
printed.

Side effects: none beyond stdout. Never modifies ``src/``, ``tests/``, or
``data/``.
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recon_low_turnover_tilt as base


RECENT_START = datetime.date(2024, 1, 1)
RECENT_END = datetime.date(2025, 7, 31)

SIG_SMOOTHING_AS = (0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01)


def compute_recent_window_stats(
    records: list[base.SessionRecord],
    a: float,
) -> dict[str, float]:
    """Compute recent-window NET statistics for a list of SessionRecord.

    Filters records to [RECENT_START, RECENT_END], extracts the net_bps_primary
    series, and computes mean, SE, t-stat, two-sided p-value, and a 95% CI
    using the t-distribution with n-1 degrees of freedom.
    """
    filtered = [r for r in records if RECENT_START <= r.date <= RECENT_END]
    net = np.array([r.net_bps_primary for r in filtered], dtype=np.float64)

    n = len(net)
    if n == 0:
        return {
            "a": float(a),
            "n": 0.0,
            "mean_net_bps": float("nan"),
            "se": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    mean = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if n >= 2 else float("nan")
    se = sd / np.sqrt(n) if n >= 2 else float("nan")
    t_stat = mean / se if n >= 2 and se > 0 else float("nan")

    if n >= 2 and np.isfinite(t_stat):
        tcrit = float(sps.t.ppf(0.975, df=n - 1))
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
        p_value = float(2 * (1 - sps.t.cdf(abs(t_stat), df=n - 1)))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
        p_value = float("nan")

    return {
        "a": float(a),
        "n": float(n),
        "mean_net_bps": mean,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def print_recent_window_table(
    rows: list[dict[str, float]],
) -> None:
    """Print the recent-window significance table."""
    print("\n" + "=" * 100, flush=True)
    print("TASK 1: RECENT-WINDOW SIGNIFICANCE (2024-01-01..2025-07-31)", flush=True)
    print("=" * 100, flush=True)
    print(
        "This is the decisive test. The smoothing construction with a=0.10 is the "
        "finding; a=1.0 is the daily-rebalance control. All statistics are computed "
        "on the NET series (net_bps_primary), which already subtracts one-leg "
        "turnover cost at the primary clip. A 95% confidence interval that excludes "
        "zero is the threshold for statistical significance.",
        flush=True,
    )
    print(
        f"{'combo':<28} {'a':>5} {'n':>5} {'mean_net':>9} {'se':>8} "
        f"{'t_stat':>7} {'p_value':>9} {'ci_low':>8} {'ci_high':>8}  verdict",
        flush=True,
    )
    print("-" * 100, flush=True)

    for row in rows:
        combo = row["combo"]
        a = row["a"]
        n = int(row["n"])
        mean_net = row["mean_net_bps"]
        se = row["se"]
        t_stat = row["t_stat"]
        p_value = row["p_value"]
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]

        if np.isfinite(ci_low) and np.isfinite(ci_high):
            if ci_low > 0 or ci_high < 0:
                verdict = "SIGNIFICANT (95%)"
            else:
                verdict = "NOT DISTINGUISHABLE FROM ZERO"
        else:
            verdict = "INSUFFICIENT DATA"

        print(
            f"{combo:<28} {a:>5.2f} {n:>5} {mean_net:>9.2f} {se:>8.2f} "
            f"{t_stat:>7.2f} {p_value:>9.4f} {ci_low:>8.2f} {ci_high:>8.2f}  {verdict}",
            flush=True,
        )


def print_plateau_grid(
    tilt: str,
    universe: str,
    sessions: list[base.PrecomputedSession],
    cost_bps_primary: float,
) -> None:
    """Print the plateau-vs-spike grid for one (tilt, universe) combination."""
    print(f"\n=== Plateau grid: {tilt} tilt, {universe} universe ===", flush=True)
    print(
        f"{'a':>5} {'gross_bps':>9} {'turnover':>9} {'pooled_net':>11} "
        f"{'ann_net_pct':>11} {'2024_net':>9} {'2025_net':>9}",
        flush=True,
    )
    print("-" * 70, flush=True)

    for a in SIG_SMOOTHING_AS:
        records = base.simulate_smoothing(sessions, tilt, a, cost_bps_primary)
        stats = base.pooled_stats(records)
        by_year = base.aggregate_by_year(records)

        net_2024 = by_year.get(2024, {}).get("net_bps_primary", float("nan"))
        net_2025 = by_year.get(2025, {}).get("net_bps_primary", float("nan"))

        print(
            f"{a:>5.2f} {stats['mean_gross_bps']:>9.2f} "
            f"{stats['mean_turnover']:>9.4f} {stats['mean_net_bps']:>11.2f} "
            f"{stats['ann_net_pct']:>11.2f} {net_2024:>9.2f} {net_2025:>9.2f}",
            flush=True,
        )


def run_correctness_gate(
    sessions_by_universe: dict[str, list[base.PrecomputedSession]],
    cost_bps_primary: float,
) -> None:
    """Verify that the a=1.0 control reproduces the baseline script's numbers."""
    print("\nRunning correctness gate...", flush=True)

    records = base.simulate_smoothing(
        sessions_by_universe["full"], "mild", 1.0, cost_bps_primary
    )
    stats = base.pooled_stats(records)

    expected_gross = 10.03
    expected_turnover = 1.2573
    expected_net = -0.36

    actual_gross = stats["mean_gross_bps"]
    actual_turnover = stats["mean_turnover"]
    actual_net = stats["mean_net_bps"]

    gross_ok = abs(actual_gross - expected_gross) < 0.05
    turnover_ok = abs(actual_turnover - expected_turnover) < 0.001
    net_ok = abs(actual_net - expected_net) < 0.05

    if not (gross_ok and turnover_ok and net_ok):
        print("CORRECTNESS GATE FAILED", flush=True)
        print(f"  Expected gross_bps:  {expected_gross:.2f}", flush=True)
        print(f"  Actual gross_bps:    {actual_gross:.2f}", flush=True)
        print(f"  Expected turnover:   {expected_turnover:.4f}", flush=True)
        print(f"  Actual turnover:     {actual_turnover:.4f}", flush=True)
        print(f"  Expected net_bps:    {expected_net:.2f}", flush=True)
        print(f"  Actual net_bps:      {actual_net:.2f}", flush=True)
        sys.exit(1)

    print("CORRECTNESS GATE PASSED", flush=True)
    print(
        f"  gross_bps={actual_gross:.2f} (expected {expected_gross:.2f}), "
        f"turnover={actual_turnover:.4f} (expected {expected_turnover:.4f}), "
        f"net_bps={actual_net:.2f} (expected {expected_net:.2f})",
        flush=True,
    )


def main() -> None:
    """Run the significance and plateau-grid analysis."""
    print("Loading universe...", flush=True)
    symbols = base.load_universe("all_equity").symbols
    print(f"Universe has {len(symbols)} symbols.", flush=True)

    spec = base.PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=symbols,
        start=base.START,
        end=base.END,
    )

    print("Loading panel (this may take minutes)...", flush=True)
    panel = base.load_panel(spec, memmap=True)
    print(
        f"Panel loaded: {panel.n_rows()} rows, {panel.n_days()} days, "
        f"{panel.n_symbols()} symbols.",
        flush=True,
    )

    print("Building overnight feature...", flush=True)
    feature = base.build_overnight_feature(panel)
    print("Reducing to checkpoint panel...", flush=True)
    checkpoint_panel, checkpoint_feature = base._build_checkpoint_panel(panel, feature)
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

    costs = base.NSEIntradayEquityCosts()
    cost_bps_primary = costs.round_trip_bps(base.PRIMARY_CLIP)

    # Correctness gate must pass before any results are printed.
    run_correctness_gate(sessions_by_universe, cost_bps_primary)

    # Task 1: recent-window significance table.
    rows: list[dict[str, float]] = []
    for tilt in base.TILT_NAMES:
        for universe in base.UNIVERSE_NAMES:
            sessions = sessions_by_universe.get(universe, [])
            if not sessions:
                continue
            for a in (0.10, 1.0):
                records = base.simulate_smoothing(
                    sessions, tilt, a, cost_bps_primary
                )
                stats = compute_recent_window_stats(records, a)
                stats["combo"] = f"{tilt}/{universe}"
                rows.append(stats)

    print_recent_window_table(rows)

    # Task 2: plateau-vs-spike grid.
    for tilt in base.TILT_NAMES:
        for universe in base.UNIVERSE_NAMES:
            sessions = sessions_by_universe.get(universe, [])
            if not sessions:
                continue
            print_plateau_grid(tilt, universe, sessions, cost_bps_primary)


if __name__ == "__main__":
    main()
