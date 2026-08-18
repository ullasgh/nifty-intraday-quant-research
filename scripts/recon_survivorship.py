"""Recon survey: does the all_equity universe have survivorship-flavoured coverage,
and does restricting to continuous-coverage names change the H2 overnight-reversal
spread in the direction claimed by prior verdicts?

Every hypothesis verdict in results/hypotheses/ (H2, H4, H5) argues: "the universe is a
fixed CURRENT-DAY list, so EARLY years are survivorship-inflated and RECENT years are the
least-biased data; therefore a decaying edge cannot be rescued by a survivorship argument."
That reasoning has never been measured -- results/hypotheses/H2/verdict.md's warning line
is a template string (see nifty_quant.universe.static.SurvivorshipReport.warning_line) that
fires unconditionally, not a computed effect on the spread itself.

This script does two things:

PART A: Using only the loaded panel as ground truth (never the data/MANIFEST.json cache,
which spans years outside this study window), it computes each symbol's first and last
calendar date with at least one non-NaN `close` bar, and the EXACT set of calendar years in
which it has at least one bar (not inferred from first/last, since a mid-window gap
spanning an entire year would otherwise be missed). It reports how many names cover 2018,
how many cover 2025, how many STOP before the panel's last date (delisted/stopped inside the
window -- if >0 the universe is NOT survivors-only), and how many START after the panel's
first date (later listings).

PART B: It compares per-year H2 spreads (via the production nifty_quant.research.hypotheses
.h2_overnight_reversal.run_h2 -- this script never reimplements H2's feature construction,
checkpoint reduction, or expectancy math) on the full 149-name all_equity universe vs the
subset of symbols with data on both the very first and the very last panel date (the
survivor-only construction). Direction is ambiguous a priori: H2's spread is a
cross-sectionally DEMEANED top-minus-bottom spread, so a common level shift from omitting
distressed names cancels; survivorship only inflates the spread if the omission correlates
with the signal itself. The survivorship story predicts the continuous subset's early years
should look MORE NEGATIVE (larger-magnitude reversal) than the full universe's early years.
This script prints the per-year numbers and the delta; it does not bake a hand-picked
pass/fail threshold into the verdict line -- see CLAUDE.md rule 8 (no un-measured
thresholds) -- the delta is reported plainly for a human to read against the noise floor
visible in the later, non-predicted years.

Reads only: nifty_quant panel/universe/hypothesis APIs. Never touches data/ beyond reading
through those APIs, and never loads past 2025-07-31 (the last 12 months are the holdout
window used by every hypothesis verdict and must not be read here either).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.research.hypotheses.h2_overnight_reversal import run_h2
from nifty_quant.universe.static import load_universe

FIELDS = ("open", "close", "volume")
START_DATE = date(2018, 1, 1)
END_DATE = date(2025, 7, 31)  # Holdout starts after this; never load past this date.


def compute_presence(panel: Panel) -> tuple[np.ndarray, np.ndarray, dict[str, set[int]]]:
    """Per-symbol coverage facts derived only from the panel's own bars.

    Returns (first_day_idx, last_day_idx, years_present):
      - first_day_idx / last_day_idx: index into panel.dates of the first/last trading day
        on which that symbol has >= 1 non-NaN close bar (-1 if never present).
      - years_present: symbol -> exact set of calendar years with >= 1 non-NaN close bar.
        Computed directly per day, not inferred from first/last, so a symbol with real bars
        in 2018 and 2023 but a total gap through 2019-2022 is never miscounted as "present"
        in the gap years.

    NaN means "no bar" throughout; nothing here forward-fills or interpolates.
    """
    close = panel.field("close")
    n_days = panel.n_days()
    n_symbols = panel.n_symbols()

    day_has_bar = np.zeros((n_days, n_symbols), dtype=bool)
    for day_idx in range(n_days):
        start = int(panel.day_offsets[day_idx])
        end = int(panel.day_offsets[day_idx + 1])
        day_has_bar[day_idx] = np.any(~np.isnan(close[start:end, :]), axis=0)

    first_idx = np.full(n_symbols, -1, dtype=np.int64)
    last_idx = np.full(n_symbols, -1, dtype=np.int64)
    for day_idx in range(n_days):
        has_bar = day_has_bar[day_idx]
        first_idx[(first_idx == -1) & has_bar] = day_idx
        last_idx[has_bar] = day_idx

    day_years = np.array([d.year for d in panel.dates], dtype=np.int64)
    years_present: dict[str, set[int]] = {}
    for sym_idx, sym in enumerate(panel.symbols):
        years_present[sym] = set(day_years[day_has_bar[:, sym_idx]].tolist())

    return first_idx, last_idx, years_present


def build_presence(
    symbols: tuple[str, ...],
    first_idx: np.ndarray,
    last_idx: np.ndarray,
    dates: list[date],
) -> dict[str, dict[str, object]]:
    """Convert per-symbol day indices into a small presence record."""
    presence: dict[str, dict[str, object]] = {}
    for sym, fi, li in zip(symbols, first_idx, last_idx):
        first_date = dates[int(fi)] if fi >= 0 else None
        last_date = dates[int(li)] if li >= 0 else None
        presence[sym] = {
            "first_idx": int(fi) if fi >= 0 else None,
            "last_idx": int(li) if li >= 0 else None,
            "first_date": first_date,
            "last_date": last_date,
        }
    return presence


def main() -> None:
    print("=" * 78)
    print("Recon: survivorship coverage and direction test for all_equity / H2")
    print("=" * 78)
    print(f"Panel window: {START_DATE} -> {END_DATE}")
    print(f"Fields: {', '.join(FIELDS)}")
    print()

    universe = load_universe("all_equity")
    full_symbols = tuple(universe.symbols)
    print(f"Loaded all_equity universe: {len(full_symbols)} symbols")

    print("Loading full-universe panel (this may take a while on first cache build)...")
    full_panel = load_panel(
        PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=full_symbols,
            start=START_DATE,
            end=END_DATE,
        ),
        memmap=True,
    )
    print(
        f"Full panel: {full_panel.n_rows()} rows, {full_panel.n_days()} days, "
        f"{full_panel.n_symbols()} symbols"
    )

    # ------------------------------------------------------------------
    # PART A -- is the universe actually survivor-biased?
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("PART A -- is the universe actually survivor-biased?")
    print("=" * 78)

    first_idx, last_idx, years_present = compute_presence(full_panel)
    dates = list(full_panel.dates)
    presence = build_presence(full_panel.symbols, first_idx, last_idx, dates)

    first_panel_date = dates[0]
    last_panel_date = dates[-1]

    names_2018 = sum(1 for sym in full_symbols if 2018 in years_present[sym])
    names_2025 = sum(1 for sym in full_symbols if 2025 in years_present[sym])

    stop_early = [
        (sym, row["last_date"])
        for sym, row in presence.items()
        if row["last_date"] is not None and row["last_date"] < last_panel_date
    ]
    start_late = [
        (sym, row["first_date"])
        for sym, row in presence.items()
        if row["first_date"] is not None and row["first_date"] > first_panel_date
    ]

    print(f"Universe size: {len(full_symbols)}")
    print(f"Symbols with >=1 bar in 2018: {names_2018}")
    print(f"Symbols with >=1 bar in 2025: {names_2025}")
    print(f"Symbols that STOP before panel last date {last_panel_date}: {len(stop_early)}")
    print(f"Symbols that START after panel first date {first_panel_date}: {len(start_late)}")

    print()
    print("Names that stop early (name, last_with_data date), sorted by date ascending:")
    for sym, last_date in sorted(stop_early, key=lambda item: item[1]):
        print(f"  {sym:<14} {last_date}")

    print()
    print("Names that start late (name, first_with_data date), sorted by date descending:")
    for sym, first_date in sorted(start_late, key=lambda item: item[1], reverse=True):
        print(f"  {sym:<14} {first_date}")

    # ------------------------------------------------------------------
    # PART B -- the direction test
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("PART B -- full universe vs continuous-coverage universe, per-year H2 spreads")
    print("=" * 78)

    continuous_symbols = tuple(
        sym
        for sym, row in presence.items()
        if row["first_date"] == first_panel_date and row["last_date"] == last_panel_date
    )
    continuous_set = set(continuous_symbols)
    excluded_symbols = [sym for sym in full_symbols if sym not in continuous_set]

    print(f"continuous_universe size: {len(continuous_symbols)}")
    print(f"Excluded from continuous_universe: {len(excluded_symbols)}")
    if excluded_symbols:
        for sym in excluded_symbols:
            row = presence[sym]
            print(f"  {sym:<14} first={row['first_date']} last={row['last_date']}")

    if not continuous_symbols:
        print("No continuous-coverage symbols found; stopping Part B.")
        return

    print()
    print("Loading continuous-universe panel...")
    continuous_panel = load_panel(
        PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=continuous_symbols,
            start=START_DATE,
            end=END_DATE,
        ),
        memmap=True,
    )
    print(
        f"Continuous panel: {continuous_panel.n_rows()} rows, "
        f"{continuous_panel.n_days()} days, {continuous_panel.n_symbols()} symbols"
    )

    results: dict[int, dict[str, object]] = {}

    print()
    print("Per-year H2 spreads (spread_bps, n_total):")
    header = (
        f"{'year':>4} | {'full_universe':>24} | {'continuous_universe':>28} | "
        f"{'delta (cont-full)':>18}"
    )
    print(header)
    print("-" * len(header))

    def fmt_value(value: object) -> str:
        if value is None:
            return "N/A"
        assert isinstance(value, tuple)
        spread, n = value
        return f"{spread:>10.2f} ({n:>6d})"

    for year in range(START_DATE.year, END_DATE.year + 1):
        start = date(year, 1, 1)
        # For 2025 this is deliberately date(2025, 12, 31): Panel.sub's inclusive date
        # search clips it to whatever trading days are actually present in the panel, and
        # the panel itself was never loaded past END_DATE, so no holdout data can leak in.
        end = date(year, 12, 31)

        row: dict[str, object] = {"full": None, "continuous": None, "delta": None}

        for universe_name, panel in (("full", full_panel), ("continuous", continuous_panel)):
            try:
                verdict = run_h2(panel, start=start, end=end)
                spread_bps = float(verdict.expectancy.spread_bps)
                n_total = int(verdict.expectancy.n_total)
                if n_total == 0:
                    print(f"  NOTE: {year} {universe_name} returned n_total=0; skipping")
                else:
                    row[universe_name] = (spread_bps, n_total)
            except Exception as exc:  # noqa: BLE001 - per-year failure isolation, see docstring
                print(
                    f"  NOTE: {year} {universe_name} failed with "
                    f"{type(exc).__name__}: {exc}; skipping"
                )

        full_value = row["full"]
        continuous_value = row["continuous"]
        if full_value is not None and continuous_value is not None:
            row["delta"] = float(continuous_value[0] - full_value[0])  # type: ignore[index]

        results[year] = row

        delta = row["delta"]
        delta_str = "N/A" if delta is None else f"{float(delta):>10.2f}"
        print(
            f"{year:>4} | {fmt_value(full_value):>24} | {fmt_value(continuous_value):>28} | "
            f"{delta_str:>18}"
        )

    # Restricted table: only years with data in both universes.
    valid_years = {
        year: row
        for year, row in results.items()
        if row["full"] is not None and row["continuous"] is not None
    }

    print()
    print("Restricted table: only years with data in both universes")
    if not valid_years:
        print("  No years had valid data in both universes.")
    else:
        print(header)
        print("-" * len(header))
        for year in sorted(valid_years):
            row = valid_years[year]
            delta = row["delta"]
            delta_str = "N/A" if delta is None else f"{float(delta):>10.2f}"
            print(
                f"{year:>4} | {fmt_value(row['full']):>24} | "
                f"{fmt_value(row['continuous']):>28} | {delta_str:>18}"
            )
        if len(valid_years) != len(results):
            skipped = sorted(set(results) - set(valid_years))
            print(f"  Skipped years (one or both universes failed/empty): {skipped}")

    # No hand-picked pass/fail threshold here (CLAUDE.md rule 8: thresholds must come from a
    # measured null distribution, and none has been measured for "how big a delta is noise"
    # in this setting). Report the early-year vs late-year deltas plainly and let the reader
    # -- or the calling agent's own writeup -- judge magnitude against the visible spread of
    # deltas across all 8 years, which is the only noise-floor evidence this script has.
    early_years = [y for y in (2018, 2019) if results.get(y, {}).get("delta") is not None]
    late_years = [
        y for y in (2023, 2024, 2025) if results.get(y, {}).get("delta") is not None
    ]
    print()
    print("=" * 78)
    print("Direction-test summary (no hardcoded threshold -- read the deltas above)")
    if early_years:
        early_deltas = [float(results[y]["delta"]) for y in early_years]  # type: ignore[arg-type]
        print(
            f"  Early-year deltas (continuous - full), {early_years}: "
            f"{[round(d, 2) for d in early_deltas]} bps"
        )
    else:
        print("  Early-year deltas: N/A (no valid years)")
    if late_years:
        late_deltas = [float(results[y]["delta"]) for y in late_years]  # type: ignore[arg-type]
        print(
            f"  Late-year deltas  (continuous - full), {late_years}: "
            f"{[round(d, 2) for d in late_deltas]} bps"
        )
    else:
        print("  Late-year deltas: N/A (no valid years)")
    print(
        "  Survivorship story predicts: early deltas negative (continuous universe shows a "
        "MORE negative, i.e. more reversal-consistent, spread than the full universe in "
        "2018-2019) and larger in magnitude than the late-year deltas."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
