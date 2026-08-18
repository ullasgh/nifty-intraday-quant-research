# Spec: `nifty_quant.research.expectancy` — the conditional-expectancy engine

**Status:** contract for TDD. Tests written from this document alone, before implementation.

## Why this exists

This repo can backtest a strategy but it cannot answer the question that should come *first*:

> Does `E[R_{t+h} | feature bucket]` differ from zero at all, by enough to pay costs?

Every strategy here was built by assuming an effect existed and going straight to a backtest.
All of them lost. `volume_breakout` (liquid25, 2024, real costs) is gross Sharpe **-4.058**, net
**-3.521**, breakeven **0.0 bps**, `ruined: True` — it has no gross edge to pay any cost from.
A conditional-expectancy pass would have shown that in minutes instead of via a full engine run.

This module is the gate every hypothesis passes through before a single line of strategy code
is written.

## The calibration rule (learned the hard way, 2026-08-17)

The adjustment audit shipped with a threshold I chose by reasoning rather than measurement
(`min_traded_value_log_step = 0.35`). On real data it produced **8,217 false positives across
all 149 symbols** — because 1.42x is well inside normal daily volume variation. Every synthetic
test passed.

**Therefore: no threshold in this module may be a hand-chosen constant.** Every cutoff must be
derived from a measured null distribution on real data, and the derivation recorded in the
result object. A function that takes a magic number must also take, and report, the null it was
calibrated against. This is a hard requirement, not a style preference.

## Public API

```python
@dataclass(frozen=True)
class ForwardReturns:
    values: np.ndarray          # (n_rows, n_symbols) float64, NaN where undefined
    horizon: int                # bars
    session_bounded: bool       # always True; recorded so a caller cannot forget
    n_defined: int
    n_nan_tail: int             # count NaN'd because the horizon ran off the session end
    def explain(self) -> str: ...


def forward_returns(
    close: np.ndarray, day_offsets: np.ndarray, horizon: int
) -> ForwardReturns:
    """h-bar forward LOG return, SESSION-BOUNDED.

    values[t] = log(close[t + horizon] / close[t]), and is NaN whenever t and t + horizon
    are not in the same session. Rows within `horizon` of a session end are therefore NaN --
    this is not padding, it is the absence of a defined forward return. Never crosses a
    session boundary; never uses `t + horizon` from the next day. Never assumes a fixed
    375-bar stride -- session membership comes from `day_offsets` (CLAUDE.md rule 5).

    NaN in `close` (no bar) propagates to NaN and is NEVER forward-filled (rule 6).
    Computed in float64 (rule 3). Raises ValueError for horizon < 1.
    """


@dataclass(frozen=True)
class Bucketing:
    labels: np.ndarray          # (n_rows, n_symbols) int8; -1 == unassigned
    n_buckets: int
    method: Literal["expanding_quantile", "rolling_quantile", "cross_sectional_rank"]
    warmup_rows: int
    edges_source: str           # human description of where the cut points came from
    def explain(self) -> str: ...


def causal_buckets(
    feature: np.ndarray,
    day_offsets: np.ndarray,
    n_buckets: int = 5,
    method: str = "expanding_quantile",
    min_history: int = 5000,
) -> Bucketing:
    """Assign each observation to a feature quantile bucket using ONLY prior information.

    THE POINT OF THIS FUNCTION: a full-sample quantile is itself a lookahead leak. Bucketing
    by `np.quantile(feature)` over the whole panel uses the future to decide what counted as
    "extreme" today, which inflates every downstream expectancy. This is the single easiest
    way to manufacture a fake edge in this repo, and it looks completely innocent.

    - "expanding_quantile": cut points at row t come from rows [0, t). Rows before
      `min_history` are labelled -1 (unassigned) rather than bucketed on thin data.
    - "rolling_quantile": cut points from a trailing window.
    - "cross_sectional_rank": rank across symbols WITHIN row t only -- no time dimension, so
      it is causal by construction. Reuse `features.core.cross_sectional_rank`.

    Must be decorated `@causal` (see `nifty_quant.guards`) so the lookahead prober exercises it.
    """


@dataclass(frozen=True)
class BucketStat:
    bucket: int
    n_obs: int
    n_effective: float          # after the overlap correction; << n_obs when horizon > 1
    mean_bps: float
    median_bps: float
    std_bps: float
    t_stat: float
    se_method: Literal["block_bootstrap", "non_overlapping", "naive"]
    ci_low_bps: float
    ci_high_bps: float


@dataclass(frozen=True)
class ExpectancyTable:
    buckets: tuple[BucketStat, ...]
    horizon: int
    feature_name: str
    n_total: int
    cost_hurdle_bps: float
    spread_bps: float           # top bucket mean minus bottom bucket mean
    spread_t: float
    survives_costs: bool        # abs(spread_bps) > 2 * cost_hurdle_bps
    def explain(self) -> str:
        """Full provenance: every stat, the SE method, the overlap correction, the cost
        hurdle used and where it came from, and the verdict with its reasoning. The repo's
        recorded failure mode is a plausible float with no context attached."""
    def to_frame(self) -> pd.DataFrame: ...


def conditional_expectancy(
    feature: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    *,
    n_buckets: int = 5,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    cost_hurdle_bps: float | None = None,
) -> ExpectancyTable: ...
```

### Overlap correction — mandatory, and the most likely source of a false positive

Consecutive `h`-bar forward returns share `h - 1` bars. Treating them as independent inflates
`t` by roughly `sqrt(h)`. At `h = 30` on 1-minute bars that is a **5.5x** overstatement — more
than enough to turn noise into a "highly significant" result.

`se_method` must be one of:
- `"block_bootstrap"` (default) — stationary/circular block bootstrap with block length
  `>= horizon`, resampling WITHIN sessions so blocks never straddle a session boundary.
  Deterministic given `seed`.
- `"non_overlapping"` — subsample every `horizon`-th observation. Lower power, zero overlap.
- `"naive"` — no correction. **Permitted only so tests can demonstrate the inflation**; any
  `ExpectancyTable` carrying `se_method="naive"` must say so loudly in `explain()`.

`n_effective` must reflect the correction, and must be strictly less than `n_obs` when
`horizon > 1` under the two real methods.

### Cost hurdle

`cost_hurdle_bps=None` resolves from `execution.costs.NSEIntradayEquityCosts.round_trip_bps`
(measured: **8.26 bps** at Rs 1,00,000, **4.02 bps** at Rs 10,00,000, **3.59 bps** at Rs 1 Cr notional).
`survives_costs` requires the top-minus-bottom spread to exceed **2x** the hurdle. **The 2x is an
accounting identity, not a margin:** a hypothesis is a two-leg spread (long N shares of top quintile,
short N of bottom), so P&L is `N × spread` while cost is `2 × round_trip_bps(N) × N/1e4`, giving
break-even at `spread > 2 × round_trip_bps`. **The gate has zero slippage allowance:** `NSEIntradayEquityCosts`
prices only brokerage + STT + statutory charges; the repo's own fill model adds `half_spread_bps=1.5`
and `impact_coef=10.0` (four fills per round-trip), making the true hurdle several bps higher.
Hypotheses near 2x should be read as failing, not marginal.

**Size-dependence (verified via `NSEIntradayEquityCosts.round_trip_bps()`):**

| notional per leg | round_trip_bps | 2x break-even |
|---|---|---|
| Rs 1L | 8.26452 | 16.52904 |
| Rs 10L | 4.01652 | 8.03304 |
| Rs 1Cr | 3.59172 | 7.18344 |

A verdict quoting a hurdle is meaningless without naming the clip size. Above roughly Rs 10L,
the modelled cost stops being the binding constraint because spread and impact are unpriced.

### Other required functions

### Decomposition and double-sort — SIGNATURES PINNED (amended 2026-08-17)

The first draft of this spec described these in prose only, so the test author had to infer
signatures. The tests are the contract, so the inferred forms are now normative. Match them
exactly:

```python
def double_sort(
    feature_a: np.ndarray, feature_b: np.ndarray, fwd: ForwardReturns,
    day_offsets: np.ndarray, *,
    n_buckets_a: int = 5, n_buckets_b: int = 5,
    method: str = "expanding_quantile", se_method: str = "block_bootstrap",
    seed: int = 0, thin_cell_threshold: int = 30,
) -> SortResult: ...
# SortResult.cells is ROW-MAJOR nested (a-bucket outer, b-bucket inner), each a CellStat with
# `n_obs`; SortResult.n_total; sum of all cell n_obs == n_total. Cells with
# n_obs < thin_cell_threshold must be flagged, not silently averaged.

def expectancy_by_year(
    feature, fwd, day_offsets, dates, *,
    n_buckets=5, method=..., se_method=..., seed=0,
) -> dict[int, ExpectancyTable]:  # keyed by calendar year

def expectancy_by_time_of_day(
    feature, fwd, day_offsets, minute_of_day, *,
    n_buckets=5, time_bucket_minutes=60, method=..., se_method=..., seed=0,
) -> dict[int, ExpectancyTable]:  # keyed by time bucket; uses minute_of_day, NOT row index

def expectancy_by_liquidity_decile(
    feature, fwd, day_offsets, prior_adv, *,
    n_buckets=10, method=..., se_method=..., seed=0,
) -> dict[int, ExpectancyTable]:  # keys exactly range(n_buckets)
```

`prior_adv` is a CALLER contract: it must already be lagged (strictly prior sessions). The
function does not re-derive it, and `@causal` cannot detect a leak in it — so document that
loudly at the call site. `ExpectancyTable` must expose `n_total`.

An effect living in one year, one time-of-day bucket, or the bottom liquidity decile is not an
effect.

### `explain()` vocabulary — PINNED, because the tests assert on it

`ExpectancyTable.explain()` must contain, case-insensitively: the `se_method` name verbatim
(e.g. `block_bootstrap`), the word `overlap` or `correction`, the word `hurdle`, the unit `bps`,
and the cost-hurdle value formatted so `str(value)` appears (e.g. `12.5`). When
`se_method == "naive"` it must additionally contain at least one UPPERCASE marker from
`NAIVE`, `WARNING`, `UNCORRECTED`. Provenance is a tested contract here, not decoration.

### Block bootstrap must expose its blocks

Test 21 (`blocks never straddle a session`) is otherwise only assertable as a smoke test. Make
the invariant observable: the bootstrap helper must return, or record on the result, the
resampled block start indices and length actually used, so a test can verify directly that no
block spans a `day_offsets` boundary. An invariant that cannot be observed cannot be tested,
and this is the one whose violation would silently manufacture significance.

## Required tests (`tests/test_expectancy.py`)

### `forward_returns`
1. `test_forward_return_is_log_ratio_h_bars_ahead` — hand-computed.
2. `test_never_crosses_session_boundary` — last `horizon` rows of each session are NaN.
3. `test_irregular_sessions_handled` — 375-bar day then a 60-bar Muhurat day.
4. `test_nan_close_propagates_and_is_not_filled`
5. `test_horizon_one_equals_next_bar_log_return`
6. `test_horizon_zero_and_negative_raise_value_error`
7. `test_horizon_longer_than_session_yields_all_nan_for_that_session`
8. `test_n_nan_tail_counts_horizon_truncation_correctly`
9. `test_dtype_is_float64`

### `causal_buckets` — the lookahead guarantees
10. `test_expanding_quantile_uses_only_prior_rows` — perturb rows AFTER t, assert label at t
    is unchanged. This is the core anti-leak test.
11. `test_full_sample_quantile_would_differ` — demonstrate the leak this prevents: full-sample
    bucketing assigns a DIFFERENT label than causal bucketing on the same data.
12. `test_rows_before_min_history_are_unassigned_not_bucketed`
13. `test_cross_sectional_rank_is_causal_by_construction`
14. `test_bucket_labels_are_balanced_asymptotically`
15. `test_constant_feature_degenerate_case`
16. `test_all_nan_feature_yields_all_unassigned`
17. `test_causal_decorator_probe_passes_at_full_strictness`

### Overlap correction — the false-positive guard
18. `test_naive_se_is_inflated_relative_to_block_bootstrap` — on synthetic overlapping data,
    assert naive `t` exceeds corrected `t` by roughly `sqrt(horizon)`.
19. `test_n_effective_less_than_n_obs_when_horizon_gt_one`
20. `test_non_overlapping_subsample_has_no_shared_bars`
21. `test_block_bootstrap_blocks_never_straddle_a_session`
22. `test_block_bootstrap_is_deterministic_given_seed`
23. `test_horizon_one_needs_no_correction` — `n_effective == n_obs`.

### The null — this is the acceptance test for the whole module
24. `test_pure_noise_yields_no_significant_bucket` — **AMENDED 2026-08-17; the original
    formulation was a multiple-comparisons error on my part.**

    The first version said "assert NO bucket clears significance". With `n_buckets=5` and 95%
    CIs, each bucket contains zero with probability 0.95 under a CORRECT implementation, so all
    five do with probability 0.95^5 ~= 0.77. The test therefore fails ~23% of the time on
    correct code — and it duly did, at seed 24.

    Measured to settle it: across 40 independent seeds the per-bucket "CI excludes zero" rate
    is **3.0% against a nominal 5%** (slightly conservative), and the all-five-pass rate is 88%.
    The standard errors are sound; there is no leak. The test was wrong, not the code.

    **Do NOT fix this by changing the seed.** Tuning a seed until a null test passes is the same
    failure as re-tuning a strategy default to make a verification test pass, which this repo
    has a standing rule against. Fix the DESIGN:

    - Assert on the **spread** (top bucket minus bottom bucket), which is a single comparison
      and is the quantity a strategy would actually trade: its CI must contain zero.
    - Assert `survives_costs is False`.
    - Assert the **false-positive RATE** over >= 30 independent seeds is consistent with
      nominal — e.g. the per-bucket CI-excludes-zero rate lies in [0, 0.12] for a 5% nominal
      level. This is a strictly stronger null test than any single-seed assertion: a leak
      inflates the rate far above nominal, and no individual seed can hide that. Mark it
      `@pytest.mark.slow`.

    **Use a verified-driftless generator**: the repo's shared fixture once baked ~0.0002/bar
    drift, ~8x dominant over noise, which made trend strategies score spuriously well. Keep the
    in-test driftlessness assertion.
25. `test_planted_effect_is_recovered` — plant a known conditional mean; assert the recovered
    **spread** CI contains the true planted spread. Guards against a module that only ever
    says "no".

    **AMENDED 2026-08-17, same flaw class as test 24.** The original asserted the top AND
    bottom bucket CIs each contain their true value. That is a COVERAGE check: each CI covers
    the true value with ~95% probability under correct code *regardless of effect size*, so two
    comparisons pass jointly only ~90% of the time — a ~10% flake rate on a correct
    implementation. Measured at seed 25: top CI [49.83, 50.93], bottom [-50.26, -49.13],
    half-width ~0.55 bps each. It passes at that seed by luck. Collapse it to ONE comparison on
    the spread against the true planted spread (100 bps for a +50/-50 plant).

    **The general rule this establishes** — apply it to every future statistical test here:
    - A **coverage** check ("does the CI contain the true value") is alpha-fragile by
      construction. Its failure rate is ~alpha per comparison no matter how large the effect or
      the sample. Minimise the number of such comparisons, or assert on a rate across seeds.
    - A **power/detection** check ("is this effect distinguishable from zero") with a large
      effect-to-SE ratio is effectively deterministic and is NOT fragile. Test 26 measures
      z ~= 25, so it will not flake and needs no change.
    Confusing the two is how a correct implementation acquires a flaky test suite, and how a
    genuinely broken one gets excused as "just an unlucky seed".
26. `test_planted_effect_below_cost_hurdle_does_not_survive_costs`

### `ExpectancyTable`
27. `test_spread_is_top_minus_bottom_bucket`
28. `test_survives_costs_requires_two_x_hurdle`
29. `test_cost_hurdle_defaults_to_nse_intraday_round_trip`
30. `test_explain_names_se_method_overlap_correction_and_hurdle`
31. `test_explain_warns_loudly_when_se_method_is_naive`
32. `test_to_frame_schema_and_row_order`

### Stability decomposition
33. `test_expectancy_by_year_partitions_observations_exactly`
34. `test_effect_confined_to_one_year_is_visible_in_by_year`
35. `test_by_time_of_day_uses_minute_of_day_not_row_index`
36. `test_by_liquidity_decile_uses_strictly_prior_adv`

### `double_sort`
37. `test_double_sort_cell_counts_sum_to_total`
38. `test_double_sort_reports_thin_cells`

## Constraints

- Vectorized numpy. No `iterrows`, no per-bar Python loops. Looping over buckets/years is fine.
- float64 throughout (CLAUDE.md rule 3).
- Never assume a 375-bar stride (rule 5); session membership from `day_offsets`.
- NaN means "no bar"; never forward-fill (rule 6).
- Reuse `features/core.py` primitives (`cross_sectional_rank`, `rolling_*`) — do not reimplement.
- `@causal` from `nifty_quant.guards` on every feature-consuming entry point.
- 100% line and branch coverage; no un-justified `# pragma: no cover`.
- `ruff check` and full type annotation (`disallow_untyped_defs = true`).
