#!/usr/bin/env python3
"""Measurement script for H2 reconstruction.

Purpose:
  Part A re-runs the H2 verdict at three per-leg clip sizes -- Rs 1,00,000,
  Rs 10,00,000 and Rs 25,00,000 -- using the corresponding round-trip cost
  hurdle, and reports criterion 1 / criterion 7 outcomes plus the numbers
  behind them.
  Part B decomposes the liquidity-concentration finding from the Rs 10L
  verdict, and independently reconstructs the liquidity-decile assignment to
  identify symbols that habitually populate liquidity decile 0.  It also
  computes each qualifying symbol's typical prior-session rupee-value ADV.

Universe/window:
  load_universe("all_equity"), 2018-01-01 through 2025-07-31.

Side effects:
  None.  This script only writes to stdout.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    _build_checkpoint_panel,
    build_overnight_feature,
    run_h2,
)
from nifty_quant.universe.static import load_universe

START = date(2018, 1, 1)
END = date(2025, 7, 31)

CLIP_LABELS = {
    100_000: "Rs 1L",
    1_000_000: "Rs 10L",
    2_500_000: "Rs 25L",
}


def _fmt_bps(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:,.4f}"


def _fmt_inr(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"Rs {x:,.0f}"


def _criterion7_token(line: str) -> str:
    m = re.search(
        r"Recent-years cost gate criterion:\s*(PASS|FAIL|NOT_EVALUATED)",
        line,
    )
    return m.group(1) if m else ""


def run_part_a(panel):
    """Run H2 at the three clip sizes and print criterion 1 / criterion 7."""
    print("\nPart A: clip-size sensitivity for H2 cost hurdle", flush=True)
    cost_model = NSEIntradayEquityCosts()
    clips = [100_000, 1_000_000, 2_500_000]
    verdicts = {}
    c1_flags = {}
    c7_flags = {}

    header = (
        f"{'clip':>7} {'roundtrip_bps':>13} {'2x_hurdle_bps':>14} "
        f"{'edge_bps':>12} {'C1':>5} {'recent_years':>12} "
        f"{'mean_recent_bps':>16} {'C7':>5} {'survived':>8}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for clip in clips:
        cost = cost_model.round_trip_bps(clip)
        verdict = run_h2(
            panel,
            start=START,
            end=END,
            cost_hurdle_bps=cost,
        )
        verdicts[clip] = verdict

        # Criterion 1: read back from the returned verdict fields.
        edge_bps = verdict.expectancy.spread_bps
        c1_pass = abs(edge_bps) > 2.0 * verdict.cost_hurdle_bps
        c1_flags[clip] = c1_pass

        # Criterion 7: independently recompute the recent-years edge from
        # verdict.stability.by_year, as Lens does.
        complete_years = sorted(verdict.stability.by_year.keys())
        if len(complete_years) >= 2:
            recent_years = complete_years[-2:]
            edge_a = verdict.stability.by_year[recent_years[0]].spread_bps
            edge_b = verdict.stability.by_year[recent_years[1]].spread_bps
            mean_recent_edge = (edge_a + edge_b) / 2.0
            c7_pass = abs(mean_recent_edge) > 2.0 * verdict.cost_hurdle_bps
        else:
            recent_years = tuple(complete_years)
            mean_recent_edge = float("nan")
            c7_pass = False

        reason7 = verdict.reasons[6] if len(verdict.reasons) > 6 else ""
        token = _criterion7_token(reason7)
        if token in {"PASS", "FAIL"}:
            if (token == "PASS") != c7_pass:
                print(
                    f"!! LOUD WARNING: clip {clip}: recomputed criterion7="
                    f"{'PASS' if c7_pass else 'FAIL'} but verdict.reasons[6] "
                    f"token={token}",
                    flush=True,
                )
        else:
            print(
                f"!! LOUD WARNING: clip {clip}: could not parse a PASS/FAIL "
                f"criterion7 token from reason[6]: {reason7!r}",
                flush=True,
            )

        years_str = ",".join(str(y) for y in recent_years)
        c7_flags[clip] = c7_pass

        print(
            f"{CLIP_LABELS.get(clip, str(clip)):>7} "
            f"{verdict.cost_hurdle_bps:>13.4f} "
            f"{2.0 * verdict.cost_hurdle_bps:>14.4f} "
            f"{edge_bps:>12.4f} "
            f"{'PASS' if c1_pass else 'FAIL':>5} "
            f"{years_str:>12} "
            f"{_fmt_bps(mean_recent_edge):>16} "
            f"{'PASS' if c7_pass else 'FAIL':>5} "
            f"{str(verdict.survived):>8}",
            flush=True,
        )

    return verdicts, c1_flags, c7_flags


def compute_decile0_membership(panel):
    """Reproduce Lens's volume-decile assignment and return decile-0 names.

    The decile is calculated on the same sliced checkpoint panel that run_h2
    feeds to Lens: build_overnight_feature -> _build_checkpoint_panel ->
    .sub(start, end).  For each symbol, the fraction of its finite checkpoint
    volume cells assigned to decile 0 is computed.  Symbols with fraction
    greater than 0.5 are considered decile-0 names.
    """
    print(
        "\nBuilding checkpoint panel for liquidity-decile membership ...",
        flush=True,
    )
    feature = build_overnight_feature(panel)
    checkpoint_panel, _ = _build_checkpoint_panel(panel, feature)
    del feature, _

    cp_sliced = checkpoint_panel.sub(start=START, end=END)
    volume = cp_sliced.field("volume").astype(np.float64)
    finite = np.isfinite(volume)

    deciles = np.full(volume.shape, -1, dtype=np.int8)
    if finite.any():
        volume_quantiles = np.quantile(
            volume[finite],
            np.linspace(0.0, 1.0, 11),
        )
        deciles[finite] = np.searchsorted(
            volume_quantiles[1:-1],
            volume[finite],
            side="right",
        ).astype(np.int8)

    per_finite = np.sum(finite, axis=0)
    per_decile0 = np.sum(deciles == 0, axis=0)

    fraction = np.zeros(len(cp_sliced.symbols), dtype=np.float64)
    np.divide(
        per_decile0,
        per_finite,
        out=fraction,
        where=per_finite > 0,
    )

    names = [
        (sym, float(fraction[i]))
        for i, sym in enumerate(cp_sliced.symbols)
        if fraction[i] > 0.5
    ]
    names.sort(key=lambda pair: pair[1], reverse=True)
    return names


def compute_typical_prior_session_adv(panel):
    """Compute each symbol's typical prior-session rupee-value ADV.

    Definition:
      On the original full 1-minute raw panel restricted to [START, END],
      each session's rupee value for a symbol is the sum over that session's
      bars of close_price * volume.  Any row where either close or volume is
      NaN contributes zero to that session's sum.  Segmentation is via
      panel.day_offsets.  A symbol's "typical prior-session ADV" is the median
      of that symbol's per-session rupee-value series over all sessions,
      excluding sessions with exactly zero rupee value (symbol absent that
      day).
    """
    print(
        "\nComputing per-session rupee-value ADV on the full raw panel ...",
        flush=True,
    )
    adv_panel = panel.sub(start=START, end=END)
    close = adv_panel.field("close")
    volume = adv_panel.field("volume")

    n_days = adv_panel.n_days()
    n_symbols = adv_panel.n_symbols()
    n_rows = adv_panel.n_rows()
    day_offsets = adv_panel.day_offsets

    per_day_total = np.zeros((n_days, n_symbols), dtype=np.float64)

    for day_idx in range(n_days):
        start_row = int(day_offsets[day_idx])
        end_row = (
            int(day_offsets[day_idx + 1])
            if day_idx + 1 < n_days
            else n_rows
        )
        if end_row <= start_row:
            continue

        close_day = close[start_row:end_row]
        volume_day = volume[start_row:end_row]

        valid = np.isfinite(close_day) & np.isfinite(volume_day)
        safe_close = np.where(valid, close_day, 0.0).astype(np.float64)
        safe_volume = np.where(valid, volume_day, 0.0).astype(np.float64)

        per_day_total[day_idx] = np.sum(
            safe_close * safe_volume,
            axis=0,
            dtype=np.float64,
        )

    typical_adv = np.full(n_symbols, np.nan, dtype=np.float64)
    for sym_idx in range(n_symbols):
        symbol_values = per_day_total[:, sym_idx]
        nonzero = symbol_values[symbol_values != 0.0]
        if nonzero.size > 0:
            typical_adv[sym_idx] = float(np.median(nonzero))

    return adv_panel, typical_adv


def run_part_b(panel, verdict_10l):
    """Print liquidity-decile decomposition and identify decile-0 names."""
    print("\nPart B: liquidity concentration decomposition", flush=True)
    stability = verdict_10l.stability

    print("\nLiquidity decile expectancy tables:", flush=True)
    for decile_key in range(10):
        if decile_key in stability.by_liquidity_decile:
            table = stability.by_liquidity_decile[decile_key]
            print(
                f"  decile {decile_key}: "
                f"spread_bps={table.spread_bps:.4f} "
                f"n_total={table.n_total:,}",
                flush=True,
            )
        else:
            print(f"  decile {decile_key}: MISSING", flush=True)

    liquidity_edges = [
        t.spread_bps for t in stability.by_liquidity_decile.values()
    ]
    nonzero_liquidity_edges = [e for e in liquidity_edges if e != 0]

    if nonzero_liquidity_edges:
        sorted_liquidity_edges = sorted(
            abs(e) for e in nonzero_liquidity_edges
        )
        max_liquidity_edge = sorted_liquidity_edges[-1]
        median_liquidity_edge = sorted_liquidity_edges[
            len(sorted_liquidity_edges) // 2
        ]
    else:
        max_liquidity_edge = float("nan")
        median_liquidity_edge = float("nan")

    bottom_decile_edge = None
    if 0 in stability.by_liquidity_decile:
        bottom_decile_edge = stability.by_liquidity_decile[0].spread_bps

    ratio = float("nan")
    if nonzero_liquidity_edges and median_liquidity_edge > 0:
        ratio = max_liquidity_edge / median_liquidity_edge

    concentrated = (
        median_liquidity_edge > 0
        and max_liquidity_edge > median_liquidity_edge * 2
        and bottom_decile_edge is not None
        and abs(bottom_decile_edge) == max_liquidity_edge
    )

    print("\nCriterion-4 reconstruction:", flush=True)
    if bottom_decile_edge is not None:
        print(
            f"  decile 0 spread_bps: {bottom_decile_edge:.4f}",
            flush=True,
        )
    else:
        print("  decile 0 spread_bps: MISSING", flush=True)
    print(
        f"  max_liquidity_edge (abs spread): {_fmt_bps(max_liquidity_edge)}",
        flush=True,
    )
    print(
        f"  median_liquidity_edge: {_fmt_bps(median_liquidity_edge)}",
        flush=True,
    )
    print(f"  ratio (max/median): {_fmt_bps(ratio)}", flush=True)
    print(
        "  Criterion 4 concentration fires: "
        f"{'YES' if concentrated else 'NO'}",
        flush=True,
    )

    decile0_names = compute_decile0_membership(panel)
    print(
        "\nDecile-0 membership methodology: "
        "volume deciles are computed on the sliced checkpoint panel; "
        "fraction = finite checkpoint cells in decile 0 / finite checkpoint "
        "cells.",
        flush=True,
    )
    print(
        f"  Number of symbols with fraction > 0.5: {len(decile0_names)}",
        flush=True,
    )
    for sym, fraction in decile0_names:
        print(
            f"    {sym:<24} decile0_fraction={fraction:>6.3f}",
            flush=True,
        )

    print(
        "\nPrior-session ADV methodology: on the original full 1-minute "
        "panel, each session's rupee value is sum(close * volume) over its "
        "bars, with NaN close/volume treated as zero.  A symbol's typical "
        "ADV is the median of its per-session rupee-value series, excluding "
        "sessions with exactly zero.",
        flush=True,
    )

    adv_panel, typical_adv = compute_typical_prior_session_adv(panel)

    adv_vals = []
    missing_adv_names = []
    for sym, fraction in decile0_names:
        sym_idx = adv_panel.sym_ix.get(sym)
        if sym_idx is None:
            adv_value = None
        else:
            adv_value = typical_adv[sym_idx]

        if adv_value is None or not np.isfinite(adv_value):
            display = "N/A"
            missing_adv_names.append(sym)
        else:
            adv_vals.append(float(adv_value))
            display = _fmt_inr(adv_value)

        print(
            f"    {sym:<24} decile0_fraction={fraction:>6.3f} "
            f"typical_ADV={display}",
            flush=True,
        )

    if missing_adv_names:
        print(
            "  WARNING: some decile-0 names had no finite ADV and were "
            f"excluded from the median: {', '.join(missing_adv_names)}",
            flush=True,
        )

    median_adv = float(np.median(adv_vals)) if adv_vals else float("nan")
    print(
        f"  Median typical ADV over {len(decile0_names)} decile-0 names: "
        f"{_fmt_inr(median_adv)}",
        flush=True,
    )

    return median_adv, len(decile0_names)


def main():
    print(
        "Loading 1-minute panel for all_equity 2018-01-01..2025-07-31 ...",
        flush=True,
    )
    spec = PanelSpec(
        freq="1",
        fields=("open", "close", "volume"),
        symbols=load_universe("all_equity").symbols,
        start=START,
        end=END,
    )
    panel = load_panel(spec)
    print("Panel loaded.", flush=True)

    verdicts, c1_flags, c7_flags = run_part_a(panel)
    median_d0_adv, n_d0_names = run_part_b(
        panel,
        verdicts[1_000_000],
    )

    c1_10l = c1_flags[1_000_000]
    c7_10l = c7_flags[1_000_000]
    passes_at_10l = c1_10l and c7_10l

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print(
        "H2 passes criteria 1 and 7 at Rs 10L: "
        f"{'YES' if passes_at_10l else 'NO'}",
        flush=True,
    )
    print(
        f"Decile 0 median prior-session ADV: {_fmt_inr(median_d0_adv)} "
        f"across {n_d0_names} names",
        flush=True,
    )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
