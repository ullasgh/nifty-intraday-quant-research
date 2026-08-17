# Spec: `nifty_quant.research.audit.adjustment_audit`

**Status:** contract for TDD. Tests are written from this document alone, before any
implementation exists. If the implementation and this document disagree, this document wins
until it is explicitly amended.

## Why this exists

Spot bars in `data/` are corporate-action adjusted; there is **no corporate-actions table**, so
we cannot look up what was adjusted or by how much. We can only infer it from the bars.

The failure mode we are hunting is specific and silent: **price adjusted, volume not**. If a
1:1 bonus halves the historical price series but leaves the historical share count untouched,
then `close x volume` (traded value) takes a step change at the ex-date. Traded value feeds:

- `features.core.volume_zscore` (deseasonalized volume surprise)
- `data.validate.tradable_mask` -> the 20-session trailing ADV liquidity filter
- `execution.fills.SqrtImpactSlippage` -> `bar_traded_value`, hence every fill price
- `backtest.portfolio.GrossNotionalSizer` -> participation caps and water-filling

A step in traded value therefore manufactures a fake volume spike, a fake liquidity regime
change, and a fake capacity change on every split date in the sample — all of which look like
signal. One verified event (RELIANCE 1:1 bonus, ex-2024-10-28) suggests volume *is* adjusted,
but one event is not evidence about 149 symbols over 9 years.

## Detection logic

Because price is adjusted, a split leaves **no price discontinuity**. Detection must therefore
run off volume/traded-value, not price. The three signatures:

| price step | traded-value step | meaning | classification |
|---|---|---|---|
| none | none | both adjusted consistently | `CONSISTENT` |
| none | ~ split factor | price adjusted, volume not | `VOLUME_UNADJUSTED` |
| ~ split factor | ~ split factor | neither adjusted | `PRICE_UNADJUSTED` |
| none | step, not near any plausible factor | genuine activity change, or something else | `AMBIGUOUS` |

"Step" is measured on **trailing vs leading medians**, not adjacent days, because daily volume
is extremely noisy.

## KNOWN GAP — `PRICE_UNADJUSTED` is not actually detectable by this design

Found 2026-08-17 while reviewing the generated tests. Recorded here rather than silently fixed,
because the contract below is already implemented against and the gap does not affect the case
we actually care about. **Do not treat a clean audit as proof that price is adjusted.**

Work the arithmetic through for a 1:1 bonus (share count doubles, price halves):

| series | price step | volume step | traded-value step | detected? |
|---|---|---|---|---|
| both adjusted | none | none | none | n/a — correctly silent |
| price adjusted, volume raw | none | x2 | **x2** | YES -> `VOLUME_UNADJUSTED` |
| neither adjusted | /2 | x2 | **none** (the two cancel) | **NO** |

Traded value is `price x volume`, so in a fully unadjusted series the price fall and the share
count rise cancel exactly and traded value stays flat. Step 3 of the algorithm gates every
candidate on `abs(log(traded_value_ratio)) >= min_traded_value_log_step`, so a fully unadjusted
symbol never becomes a candidate at all and the `PRICE_UNADJUSTED` branch in step 5 is
unreachable from real data.

The generated test `test_injected_reciprocal_step_is_price_unadjusted` therefore injects
volume x4 with price /2 — arithmetically consistent with the classification rule as written,
but not a physically realisable corporate action. It tests the rule, not the world.

This is acceptable for now: we independently verified on RELIANCE (1:1 bonus, ex-2024-10-28)
that price IS adjusted, and the risk this audit exists to catch is the price-adjusted /
volume-raw case, which IS detected. But closing it needs a **second detection channel**: scan
for a price-level discontinuity (`abs(log(price_ratio))` near a plausible factor) with FLAT
traded value, independent of the traded-value gate. Tracked as a follow-up unit.

## Public API

All names below are the public surface. Everything else is private (`_`-prefixed).

```python
# nifty_quant/research/audit/adjustment_audit.py

class AdjustmentClass(StrEnum):
    CONSISTENT = "consistent"
    VOLUME_UNADJUSTED = "volume_unadjusted"
    PRICE_UNADJUSTED = "price_unadjusted"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class SuspectEvent:
    symbol: str
    date: datetime.date
    price_ratio: float          # close[d] / close[d-1], raw adjacent-day
    volume_ratio: float         # median(volume[d : d+w]) / median(volume[d-w : d])
    traded_value_ratio: float   # same window medians, on close*volume
    nearest_factor: float | None    # closest entry in PLAUSIBLE_FACTORS, or None if no match
    factor_error: float             # |log(traded_value_ratio) - log(nearest_factor)|; inf if None
    classification: AdjustmentClass
    n_days_before: int          # actual non-NaN days used in the trailing window
    n_days_after: int

    def explain(self) -> str:
        """Multi-line human-readable provenance: every input, the thresholds applied,
        and why this classification and not the others."""


@dataclass(frozen=True)
class AuditReport:
    events: tuple[SuspectEvent, ...]         # sorted by (symbol, date)
    symbols_scanned: tuple[str, ...]
    years: tuple[int, ...]
    params: "AuditParams"

    def by_class(self, cls: AdjustmentClass) -> tuple[SuspectEvent, ...]: ...
    def exclusion_windows(self) -> dict[str, tuple[tuple[date, date], ...]]:
        """symbol -> disjoint, sorted, merged (start, end) windows to exclude from research.

        Only VOLUME_UNADJUSTED and PRICE_UNADJUSTED events produce windows.
        A window spans [event_date - params.window_days, event_date + params.window_days],
        inclusive on both ends, because the rolling features that consume traded value are
        contaminated for a full window on either side of the step. Overlapping windows for the
        same symbol are merged.
        """
    def is_clean(self) -> bool:
        """True iff there are no VOLUME_UNADJUSTED and no PRICE_UNADJUSTED events."""
    def explain(self) -> str: ...
    def to_frame(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class AuditParams:
    window_days: int = 20
    min_traded_value_log_step: float = 2.3    # ~10x; see "Empirical calibration" below
    min_volume_log_step: float = 2.1          # ~8.2x; independent volume-step corroboration gate
    factor_tolerance: float = 0.04            # max |log(ratio) - log(factor)| to call a match
    max_price_step: float = 0.05              # |log(price_ratio)| below this == "no price step"
    min_days_each_side: int = 10              # else the event is not evaluable
    def __post_init__(self) -> None:
        """Raise ValueError on: window_days < 2, any threshold <= 0,
        min_days_each_side > window_days."""


PLAUSIBLE_FACTORS: Final[tuple[float, ...]]
# Common NSE split/bonus factors and their inverses:
# 2, 3, 4, 5, 10, 20, 10/3  and 1/x for each. Sorted ascending, deduplicated.
# AMENDED 2026-08-17: 3/2, 2/3, 5/2, 2/5 removed. See "Empirical calibration" below --
# these sit too close to ordinary volume/traded-value noise on this dataset to be
# defensible, and were responsible for most of the false-positive-storm histogram mass.


def daily_aggregate(symbol: str, years: Sequence[int]) -> pd.DataFrame:
    """Aggregate 1-minute bars to daily for one symbol.

    Reads data/bars/1/<symbol>/<year>.parquet. Returns a DataFrame indexed by
    `date` (datetime.date, ascending, unique) with float64 columns:
      open   -- first non-NaN open of the session
      high   -- max high
      low    -- min low
      close  -- last non-NaN close of the session
      volume -- sum of volume
      traded_value -- sum of (close * volume) over bars, NOT close_daily * volume_daily
      n_bars -- int64 count of bars present in the session

    Sessions are derived from the actual timestamps (IST calendar date), never from a fixed
    375-bar stride. NaN bars are excluded from the aggregation, never forward-filled; a
    session with zero non-NaN bars is omitted from the result entirely. A missing year file
    contributes no rows and is not an error. If no requested year exists, return an empty
    frame with the correct columns and dtypes.
    """


def scan_symbol(symbol: str, years: Sequence[int], params: AuditParams = ...) -> tuple[SuspectEvent, ...]: ...


def run_audit(
    symbols: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    params: AuditParams = ...,
    *,
    workers: int = 8,
) -> AuditReport:
    """symbols=None -> universe.static.equity_symbols(). years=None -> 2018..2026
    (DEFAULT_RESEARCH_START is 2018-01-01; 2017 is 74/105 of all irregular sessions)."""
```

CLI entry point: `python -m nifty_quant.research.audit.adjustment_audit --years 2018-2026
[--symbols A,B] [--out PATH]`. Prints `AuditReport.explain()`; writes `to_frame()` to
`--out` as parquet if given. Exit code **0** if `is_clean()`, **1** otherwise.

## Algorithm (normative)

For each symbol:
1. `df = daily_aggregate(symbol, years)`. If `len(df) < 2 * params.window_days + 1`, return `()`.
2. For each candidate index `i` in `[window_days, len(df) - window_days)`:
   - trailing = `df.iloc[i - window_days : i]`, leading = `df.iloc[i : i + window_days]`
     — note the candidate day itself belongs to **leading**, since the ex-date is the first
     day on the new basis.
   - `n_days_before/after` = count of finite `traded_value` in each window.
     If either `< params.min_days_each_side`, skip this candidate.
   - `traded_value_ratio = median(leading.traded_value) / median(trailing.traded_value)`,
     `volume_ratio` likewise on `volume`. Medians ignore NaN. If a trailing median is 0 or
     non-finite, skip.
   - `price_ratio = df.close.iloc[i] / df.close.iloc[i - 1]`.
3. Keep candidate `i` only if BOTH:
   - `abs(log(volume_ratio)) >= min_volume_log_step` (volume_ratio finite), AND
   - `abs(log(traded_value_ratio)) >= min_traded_value_log_step`.
   The volume gate (AMENDED 2026-08-17) is checked first and is independent of the
   traded-value gate: it requires `volume` itself, not just `close * volume`, to show a
   genuine step. Without it, a candidate can pass on traded-value alone from a sustained
   price trend with no real volume-adjustment mismatch. See "Empirical calibration" below.
4. **Non-maximum suppression**: group candidates by the GAP to the immediately preceding
   candidate in the (index-sorted) `candidates` list, not by span from the group's first
   candidate (AMENDED 2026-08-17 -- the original rule was wrong, see below). Within each
   group, keep only the one with the largest `abs(log(traded_value_ratio))`. This prevents
   one split emitting many near-identical events.

   **NMS grouping bug (found and fixed 2026-08-17).** The original rule measured every
   candidate's distance from the group's FIRST candidate. On real (noisy) data a single
   underlying step or regime change produces a contiguous run of candidate indices that can
   be much wider than `window_days` -- observed empirically at 56 trading days wide for one
   ZYDUSLIFE 2019 cluster, versus the ~19-39 width a clean synthetic single-day step
   produces. Because the anchor never advanced, one wide contiguous run got sliced into
   multiple fixed-size chunks measured from an arbitrary start point, so two genuinely
   adjacent local maxima (observed: 2019-06-17 and 2019-06-19, two trading days apart) ended
   up in different suppression groups and both survived as separate events -- inside what
   was meant to be a single `window_days = 20` suppression span. The fix: a new group starts
   only when the gap to the PREVIOUS candidate (not the group's first) exceeds `window_days`.
   This correctly merges an arbitrarily wide contiguous run into one group while still
   breaking the group at any genuine gap in the candidate list wider than `window_days`.

   **Tie-break (AMENDED 2026-08-17 — the original rule was wrong).** The first version of this
   spec said "ties break to the earlier date". That is incorrect, and it breaks test 23. Because
   the windows are summarised by a MEDIAN, a step at day `D` yields an *exactly identical* ratio
   for every candidate in `[D - (w/2 - 1), D + (w/2 - 1)]`: the leading median has already
   flipped to the new basis once more than half of its `w` rows lie at or past `D`, and the
   trailing median is still purely the old basis by the same argument. At the default
   `window_days = 20` that is a 19-way exact tie, and "earliest wins" would report `D - 9`.

   The correct tie-break uses the one quantity that is genuinely maximised at the ex-date and
   nowhere else — the **adjacent-day** jump, not the windowed one:

       tie_key(i) = abs(log(traded_value[i] / traded_value[i - 1]))

   At the true ex-date, day `i-1` is the last day on the old basis and day `i` the first on the
   new, so this is maximal there and approximates the split factor directly. Among tied
   candidates pick the largest `tie_key`; if `tie_key` is itself tied or non-finite, THEN fall
   back to the earlier date for determinism.

   Do NOT substitute a "contamination count" or any other heuristic that inspects how many days
   in a window deviate from that window's median. It happens to work on a clean synthetic step
   but it is not defined for real data, where daily volume noise makes almost every day deviate
   from the median.
5. Classify each surviving candidate:
   - `nearest_factor` = the entry of `PLAUSIBLE_FACTORS` minimising
     `abs(log(traded_value_ratio) - log(factor))`; `factor_error` is that minimum.
     If `factor_error > params.factor_tolerance`, `nearest_factor = None`, `factor_error = inf`.
   - `price_step = abs(log(price_ratio))`
   - if `nearest_factor is None` -> `AMBIGUOUS`
   - elif `price_step <= params.max_price_step` -> `VOLUME_UNADJUSTED`
   - elif `abs(log(price_ratio) + log(traded_value_ratio)) <= params.factor_tolerance`
     -> `PRICE_UNADJUSTED`  (price fell by the same factor traded value rose, i.e. neither leg
     was adjusted and the two effects are reciprocal)
   - else -> `AMBIGUOUS`
6. `CONSISTENT` is **never** emitted as an event. It is the absence of an event. It exists in
   the enum only so `AuditReport.by_class(CONSISTENT)` is well-defined and returns `()`.

Determinism: identical inputs must produce byte-identical `to_frame()` output regardless of
`workers`. No RNG anywhere.

## Empirical calibration (2026-08-17)

The 0.35 / 0.10 defaults produced a false-positive storm on real data: 8,217 events across
149 symbols x 2018-2026, `nearest_factor` dominated by 0.667/1.5/2.0/0.5, and the NMS bug
above double-counting single events. Recalibrated from a direct measurement, not a guess:

**Null distribution.** Computed `abs(log(20d-median traded_value ratio))` at every valid
day-index across all 149 symbols x 2018-2026 (294,932 observations; real splits are ~0-3 per
symbol over 9 years and cannot dominate this sample except in its most extreme tail).
Quantiles: p50=0.231, p90=0.624, p99=1.280, p99.9=2.340, p99.99=3.117, max=3.944. The
volume-only distribution is close: p99.9=2.142. Ordinary daily traded-value noise on this
dataset already reaches ~1.87x at just the 90th percentile -- well inside the range of every
"small" split factor (2x, 3/2x, 2/3x) that was previously in `PLAUSIBLE_FACTORS`.

**Thresholds.** `min_traded_value_log_step = 2.3` sits at ~p99.9 of the traded-value null;
`min_volume_log_step = 2.1` sits at ~p99.9 of the volume-only null. Run against the real
panel (with the NMS fix and volume-corroboration gate applied), this produces 25 total
events (4 `VOLUME_UNADJUSTED`, 0 `PRICE_UNADJUSTED`, 21 `AMBIGUOUS`) across 12 of 149
symbols -- order tens, not thousands, and reviewable by hand.

**`factor_tolerance = 0.04`**: the tightest gap between adjacent `log(factor)` values in the
revised `PLAUSIBLE_FACTORS` is 0.1054 (between 3.0 and 10/3, and between 0.3 and 1/3);
tolerance must be strictly below half that (0.0527) to prevent two factors' neighbourhoods
from overlapping. 0.10 (the old value) exceeded that half-gap and let 3.0 and 10/3 collide.

**Known, deliberate limitation.** Because ordinary noise already reaches ~1.87x at p90 and
~3.6x at p99, this detector -- traded-value step plus volume corroboration, on daily bars,
with no company-specific baseline or corporate-actions reference table -- CANNOT reliably
separate genuine small/medium split factors (2x-6x, which covers the majority of real NSE
bonus/split ratios: 1:1, 1:2, 1:4, 1:5) from ordinary noise without either the
false-positive storm or blindness to those factors. The calibrated defaults choose blindness:
only factors with `abs(log(factor)) >= 2.3` (10x, 20x, and their reciprocals 0.1, 0.05) are
reachable at all. This is a real, reported limitation of the method on this dataset, not
something to hide behind a re-tuned number. It remains useful as a coarse screen for large,
rare anomalies (>=10x) with a low false-positive rate, and its `AMBIGUOUS` output at these
large magnitudes correlates strongly with real IPO/listing/demerger liquidity ramps (e.g.
IRCTC, NIACL, MAZDOCK, the Adani-group demerger names), not corporate-action adjustments.

**RELIANCE 1:1 bonus (ex-2024-10-28)**: max observed `abs(log(traded_value_ratio))` near the
ex-date is 0.413 (2024-10-18, ratio 0.66x), below `min_traded_value_log_step` at any
calibration used here, so it is never even a candidate and cannot be flagged
`VOLUME_UNADJUSTED`. Under the OLD 0.35/0.10 defaults it WAS misclassified as
`VOLUME_UNADJUSTED` (nearest_factor 2/3, now removed from `PLAUSIBLE_FACTORS`) -- this was
one instance of the false-positive storm, not a separate bug.

## Required tests (`tests/test_adjustment_audit.py`)

Written from this spec against synthetic data. **No test may read `data/`** except the ones
explicitly marked real-data, which must `pytest.skip` when `data/MANIFEST.json` is absent.

### `AuditParams`
1. `test_params_defaults_match_spec` — defaults calibrated per "Empirical calibration" below
2. `test_params_rejects_window_days_below_two`
3. `test_params_rejects_non_positive_thresholds` — parametrized over each threshold field,
   including the new `min_volume_log_step`
4. `test_params_rejects_min_days_exceeding_window`
5. `test_params_is_frozen`

### `PLAUSIBLE_FACTORS`
6. `test_plausible_factors_sorted_unique_and_positive`
7. `test_plausible_factors_closed_under_reciprocal` — for every f, `1/f` is present within 1e-12

### `daily_aggregate`
8. `test_aggregates_ohlcv_correctly` — synthetic 2-session parquet in `tmp_path`; assert open is
   first, close is last, high is max, low is min, volume is sum, `n_bars` correct.
9. `test_traded_value_is_bar_level_sum_not_daily_product` — construct a session where
   `sum(close*volume) != close_last * sum(volume)` and assert the former.
10. `test_nan_bars_excluded_not_forward_filled` — a session with NaN close in the middle; the
    NaN must not propagate and must not be filled from the prior bar.
11. `test_session_with_all_nan_bars_is_omitted`
12. `test_missing_year_file_is_not_an_error`
13. `test_no_years_present_returns_empty_frame_with_correct_dtypes`
14. `test_irregular_session_handled` — a 60-bar Muhurat-length session aggregates normally and
    is not padded or truncated to 375.
15. `test_index_is_ascending_unique_dates`
16. `test_session_boundary_uses_ist_not_utc` — bars at 09:15 and 15:29 IST on one date must land
    in the same session; a UTC-naive split would put them in different days.

### `scan_symbol` — the core detection logic
17. `test_clean_series_emits_no_events` — geometric random walk, constant-mean volume.
18. `test_injected_volume_only_step_is_detected_as_volume_unadjusted` — build a clean series,
    then multiply volume by a factor that clears both `min_traded_value_log_step` (2.3) and
    `min_volume_log_step` (2.1) from day D onward leaving price untouched (20.0, not 2.0 --
    see "Empirical calibration"). Assert exactly one event, on date D,
    `classification == VOLUME_UNADJUSTED`, `nearest_factor == 20.0`.
19. `test_injected_reciprocal_step_is_price_unadjusted` — divide price and multiply volume by
    magnitudes that both clear the new thresholds and satisfy `volume_ratio = 1 /
    price_ratio**2` (price /20, volume *400, giving `traded_value_ratio = 20.0`). Assert
    `PRICE_UNADJUSTED`.
20. `test_step_not_near_a_plausible_factor_is_ambiguous` — factor 15.0 (clears both
    thresholds; sits > `factor_tolerance` from both 10.0 and 20.0).
21. `test_step_below_min_log_step_emits_nothing` — factor 1.2 with default thresholds.
22. `test_non_maximum_suppression_emits_one_event_per_split` — assert a single injected step
    produces exactly 1 event, not `window_days` of them.
23. `test_event_date_is_the_first_day_on_the_new_basis` — the exact off-by-one that matters.
24. `test_insufficient_history_returns_empty` — series shorter than `2*window_days+1`.
25. `test_candidate_with_too_few_finite_days_is_skipped` — NaN out most of one window.
26. `test_zero_trailing_median_is_skipped_not_divide_by_zero`
27. `test_parametrized_over_all_plausible_factors` — every factor in `PLAUSIBLE_FACTORS` is
    detected when injected, with the correct `nearest_factor`; skips factors whose
    `abs(log(factor)) < min_traded_value_log_step` (now most of the list -- see "Empirical
    calibration" -- this is the designed SPEC-AMBIGUITY skip, not new).
28. `test_detection_is_direction_symmetric` — a 0.05x step is found as readily as 20.0x.

### `AuditReport`
29. `test_by_class_filters_and_consistent_returns_empty`
30. `test_exclusion_windows_span_window_days_each_side_inclusive`
31. `test_exclusion_windows_merge_overlapping_events`
32. `test_exclusion_windows_omit_ambiguous_events`
33. `test_exclusion_windows_are_sorted_and_disjoint`
34. `test_is_clean_true_only_when_no_unadjusted_events` — parametrized over each class
35. `test_to_frame_columns_dtypes_and_row_order`
36. `test_to_frame_on_empty_report_has_correct_schema`

### `explain()` — these are the debuggability contract, not cosmetics
37. `test_suspect_event_explain_names_every_input_and_threshold` — assert the string contains
    the ratios, the nearest factor, the classification, and the threshold that decided it.
38. `test_report_explain_states_counts_per_class_and_params`
39. `test_explain_is_deterministic`

### Determinism / integration
40. `test_run_audit_is_worker_count_invariant` — `workers=1` and `workers=4` give identical
    `to_frame()`.
41. `test_run_audit_default_years_start_at_2018`
42. `test_run_audit_default_symbols_is_equity_universe_excluding_indices` — NIFTY50/NIFTY100/
    NIFTYBANK/INDIAVIX must not be scanned.
43. `test_cli_exit_code_zero_when_clean_one_when_not`
44. `test_reliance_2024_bonus_is_not_flagged_volume_unadjusted` — **real-data test**,
    `@pytest.mark.slow`, skipped when `data/MANIFEST.json` is absent. RELIANCE 1:1 bonus
    ex-2024-10-28 is the one event we have independently verified; if the audit flags it as
    `VOLUME_UNADJUSTED`, either the audit or our reading of the data is wrong. This test is the
    bridge between the synthetic suite and reality.

## Constraints

- Vectorized numpy/pandas. No `iterrows`. A loop over symbols or over years is fine and
  expected; a loop over bars or over days within a symbol is not.
- float32 at rest, float64 in motion (CLAUDE.md rule 3): cast on read, never accumulate in f32.
- Never write to `data/` (CLAUDE.md rule 2). Caches go under `cache/`.
- `ts` is int64 epoch-seconds UTC; convert with
  `pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata")` (CLAUDE.md rule 4).
- 100% line and branch coverage, no un-justified `# pragma: no cover`.
- `ruff check` and `mypy` clean; `disallow_untyped_defs = true` is on.
