# Phase E — full re-timing of `FEATURE_REGISTRY` after the two optimisations

Measured 2026-08-23 by a dedicated measurement agent. **No source file was modified.**
Every number below is a `time.monotonic()` measurement of the production call path
(`sweep_features.FEATURE_REGISTRY[i].fn(close, day_offsets, **required_fields)` and
`expectancy.conditional_expectancy(...)`), not an extrapolation, unless a row says otherwise.

## Run conditions

| | |
|---|---|
| Slice | `2018-01-01 .. 2025-07-31`, `all_equity` = 149 symbols, 1-minute bars |
| Panel | **701,863 rows x 149 symbols**, 1,880 sessions (the exact production window `recon_low_turnover_tilt.START/END`) |
| Seed | `0` (fixed, passed to `conditional_expectancy(seed=0)` and every bootstrap draw) |
| Buckets | `n_buckets=5`, `method="cross_sectional_rank"` (run_sweep's own defaults) |
| Horizons | `(1, 5, 15, 30, 60, "EOD")` -- 6 |
| dtype | float64 in motion (rule 3), panels cast on load exactly as `run_shard` does |
| Cache | `NQ_CACHE_ROOT` pointed at a private scratch dir; repo `cache/` read-only |
| Host | 16 GB, `MEMORY_LIMIT_GB=4.0`, ~5 other agents active (see *Caveats*) |

`MEMORY_LIMIT_GB=4.0` forbids loading all five OHLCV fields of the full window in one
process (5 x 0.837 GB = 4.19 GB). Features were therefore timed in four passes over the
**identical** row/symbol geometry, each loading `close` plus only the fields its own
features declare -- exactly what `run_phase_e_sweep.run_shard` does. Timings are
comparable across passes because the array shapes are identical.

## 1. Seconds per feature -- all 22, full production window, sorted descending

| # | feature | seconds | finite frac | `required_fields` |
|---:|---|---:|---:|---|
| 10 | `sector_relative_return` | 533.224 | 0.9334 | - |
| 6 | `hurst_on_stitched` | 437.631 | 0.8930 | - |
| 13 | `median_pairwise_correlation` | 141.186 | 0.9196 | - |
| 9 | `beta_residual_return` | 13.529 | 0.8470 | - |
| 0 | `volume_zscore` | 11.012 | 0.8518 | volume |
| 1 | `breakout_strength` | 10.955 | 0.8473 | high, low |
| 4 | `rogers_satchell_volatility` | 8.368 | 0.8519 | high, low, open_ |
| 3 | `garman_klass_volatility` | 7.575 | 0.8519 | high, low, open_ |
| 2 | `parkinson_volatility` | 7.570 | 0.8519 | high, low |
| 7 | `variance_ratio` | 6.635 | 0.0000 | - |
| 11 | `breadth` | 4.608 | 0.9973 | - |
| 12 | `cross_sectional_dispersion` | 4.107 | 0.9973 | - |
| 16 | `close_location_value` | 4.077 | 0.9371 | high, low |
| 14 | `vol_ratio` | 3.824 | 0.8477 | - |
| 15 | `rv_to_vix_ratio` | 3.163 | **RAISED** | - |
| 8 | `rolling_beta` | 3.079 | 0.8497 | - |
| 17 | `signed_volume_proxy` | 2.982 | 0.9371 | high, low, volume |
| 20 | `tradable_overnight_return` | 2.796 | 0.9301 | open_ |
| 18 | `amihud_illiquidity` | 2.738 | 0.8497 | - |
| 19 | `overnight_return` | 1.792 | 0.9317 | - |
| 5 | `efficiency_ratio` | 1.298 | 0.8504 | - |
| 21 | `opening_range` | 0.348 | 0.9011 | high, low |
| | **sum** | **1212.5** | | |

### Measurement failures (rule: a raise is a failure, not a null)

- **`rv_to_vix_ratio` RAISES on every slice tested** (8,250 / 22,401 / 91,837 / 701,863 rows):
  `ValueError: realized_vol_ann must be a 1-D array`. The registry wrapper
  `sweep_features._rv_to_vix_ratio` feeds `core.ewma_volatility_ann(close, ...)` -- a 2-D
  `(n_rows, n_symbols)` array -- into `market.rv_to_vix_ratio`, whose contract
  (`features/market.py:586`) is 1-D in both arguments. This is unconditional: the feature
  can never produce a value, so all 6 of its horizons are FAILED trials. It is *not* an
  AMENDMENT-4 missing-field failure -- the field it needs is a VIX series that the registry
  already substitutes with a flat proxy. **1 of 22 registry entries is dead on arrival.**
- **`variance_ratio` returns an ALL-NaN panel on the production window** (`finite_frac = 0.0000`)
  while being 62-66% finite on short slices (0.6577 at 1 month, 0.6242 at 3 months). It is
  computed per symbol on the 1-D log-price series and broadcast; over a full 7.6-year window
  every symbol accumulates at least one gap, and the estimator collapses to NaN for all 149.
  This is the AMENDMENT-4 failure mode the spec was written to stop -- an all-NaN feature that
  the trial matrix reads as a measured null -- surviving in a different place. It does *not*
  raise, so `run_sweep` records it as a clean trial. **This is a silent non-measurement, and
  it only appears at production window length; every short-slice smoke test passes.**
- The other 20 features ran clean, 84.7%-99.7% finite.

## 2. Per-trial `conditional_expectancy` cost (measured separately, as asked)

Timed on the same 701,863 x 149 panel. `run_sweep` calls `_forward_returns_for_horizon` +
`conditional_expectancy` once per (feature, horizon), so the per-trial cost is their sum.

| horizon | `forward_returns` s | CE `efficiency_ratio` (dense) | CE `overnight_return` (dense) | CE `breadth` (broadcast) | CE `variance_ratio` (all-NaN) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.102 | 12.802 | 9.496 | 7.453 | 5.287 |
| 5 | 1.086 | 31.976 | 24.882 | 9.558 | 5.740 |
| 15 | 1.425 | 30.364 | 22.028 | 8.544 | 4.638 |
| 30 | 1.200 | 25.846 | 22.887 | 9.824 | 4.733 |
| 60 | 1.239 | 25.068 | 23.095 | 8.560 | 4.254 |
| EOD | 0.744 | 35.731 | 28.034 | 16.162 | 4.641 |
| **mean** | **1.133** | **26.96** | **21.74** | **10.02** | **4.88** |

Per-trial cost is **not** feature-independent -- it depends on how the feature's values
bucket cross-sectionally:

| class | features | per-trial (CE + fwd) |
|---|---:|---:|
| dense (values vary across symbols within a row) | 17 | **25.48 s** |
| broadcast (one value per row, repeated across symbols: `breadth`, `cross_sectional_dispersion`, `median_pairwise_correlation`) | 3 | **11.15 s** |
| all-NaN (`variance_ratio`) | 1 | **6.01 s** |
| raises before CE (`rv_to_vix_ratio`) | 1 | **~0 s** |

## 3. Projection for the full sweep

Post-optimisation the feature is computed **once per feature**, not once per horizon, so:

```
total = sum(feature_times) + 132 * per_trial_expectancy_cost
      = 1212.5 s        + 2836.1 s   (132 trials, per-class rates above)
      = 4048.6 s  =  67.5 minutes of CPU work
```
Using the flat dense rate for all 132 trials instead (the conservative upper bound):
`1212.5 + 132 x 25.48 = 4576.3 s = 76.3 min`.

**The expectancy step is now 2.3x the entire feature cost.** Feature computation
is 30% of the sweep; `conditional_expectancy` is 70%.

For reference, the same sweep under the OLD 6x-recompute code would be
`6 x 1212.5 + 2836.1 = 168.5 min` of CPU -- the de-duplication saves
2.5x overall, not 6x, because expectancy was never duplicated.

## 4. Wall-clock ETA, sharded as `scripts/run_phase_e_sweep.py` does it

`shard_features_for` partitions `features[i::n_shards]`. With the script's defaults
(`--n-shards 4 --workers 4`), shards are badly imbalanced because `hurst_on_stitched` (index 6)
and `sector_relative_return` (index 10) both land in shard 2:

| shard | features | cost |
|---|---|---:|
| 0 | `volume_zscore`, `rogers_satchell_volatility`, `rolling_beta`, `cross_sectional_dispersion`, `close_location_value`, `tradable_overnight_return` | **14.4 min** |
| 1 | `breakout_strength`, `efficiency_ratio`, `beta_residual_return`, `median_pairwise_correlation`, `signed_volume_proxy`, `opening_range` | **16.7 min** |
| 2 | `parkinson_volatility`, `hurst_on_stitched`, `sector_relative_return`, `vol_ratio`, `amihud_illiquidity` | **29.2 min** |
| 3 | `garman_klass_volatility`, `variance_ratio`, `breadth`, `rv_to_vix_ratio`, `overnight_return` | **7.2 min** |

**Wall-clock ETA: ~29 minutes** (shard 2 is the critical path), plus ~10 s
panel load per shard. Total CPU across shards: 67.5 min.

A longest-processing-time-first partition of the same 4 shards would give:

| shard | cost |
|---|---:|
| 0 | 16.7 min |
| 1 | 17.0 min |
| 2 | 16.6 min |
| 3 | 17.2 min |

i.e. **~17 min wall clock** -- a 1.7x speedup for a partitioning change alone, and within
5% of the perfect-balance floor (67.5 / 4 = 16.9 min). No single feature is the binding
constraint at 4 shards: the heaviest one, `sector_relative_return` + its 6 trials, is 11.4 min.
`i::N` striping is simply the wrong partition when two features (`hurst_on_stitched` and
`sector_relative_return`, 24% of the sweep between them) can collide in one stride class.

**Caveat on the ETA:** these are single-process timings. Four concurrent shards on this
16 GB host will not achieve 4x -- see the `hurst_on_stitched` memory finding in section 6.

## 5. The bootstrap hypothesis: TESTED, and it does NOT explain the gap

The hypothesis under test was: *`hurst_on_stitched` does ~37 s of real work on the full panel
but cost ~36 min per shard; removing the 6x leaves ~10x unexplained; the residual is the
per-trial block bootstrap in `conditional_expectancy`.*

### 5a. What the bootstrap actually costs

Decomposition of one trial on the full panel (feature = `efficiency_ratio`), timing the same
four stages `conditional_expectancy` runs internally:

| horizon | `causal_buckets` | mask + flatten (5 buckets) | mean/median/std | **block bootstrap** | sum | bootstrap share |
|---|---:|---:|---:|---:|---:|---:|
| 1 (fwd.horizon=1) | 11.671 | 2.619 | 2.016 | **0.000** | 16.305 | **0.0%** |
| 5 (fwd.horizon=5) | 8.679 | 4.654 | 2.972 | **17.231** | 33.536 | **51.4%** |
| 60 (fwd.horizon=60) | 9.005 | 4.746 | 1.905 | **13.402** | 29.058 | **46.1%** |
| EOD (fwd.horizon=375) | 8.527 | 4.724 | 2.199 | **22.439** | 37.889 | **59.2%** |

The bootstrap is **exactly zero at horizon 1** -- `_compute_bucket_stats` takes the
`if horizon == 1` analytic-SE branch and never enters `_block_bootstrap_means_chunked`.
At every other horizon it is **46-59% of the trial**, and averaged over all six horizons it is
**~51% of per-trial expectancy cost**. So yes, the block bootstrap is the single largest
component of a trial, and since expectancy is 70% of the whole sweep, the bootstrap is
**~36% of total sweep cost**. That part of the hypothesis is correct and worth acting on.

### 5b. But it is nowhere near a 10x residual -- the 37 s figure was simply wrong

`hurst_on_stitched` costs **437.6 s** on the full panel, not ~37 s.
The extrapolation recorded in `feature_sweep.run_sweep`'s docstring is wrong by **~11.8x**.

Measured scaling in rows (149 symbols throughout, same code path):

| rows | window | `hurst_on_stitched` | `sector_relative_return` | `median_pairwise_correlation` |
|---:|---|---:|---:|---:|
| 8,250 | 1 month | 0.432 s | 5.605 s | 1.541 s |
| 22,401 | 3 months | 1.235 s | 15.469 s | 4.297 s |
| 91,837 | 12 months | 7.111 s | 66.309 s | 17.942 s |
| 701,863 | full window | 437.631 s | 533.224 s | 141.186 s |
| | **91,837 -> 701,863 (7.64x rows)** | **61.5x** | **8.0x** | **7.9x** |
| | implied exponent | **2.02** | 1.03 | 1.02 |

`sector_relative_return` and `median_pairwise_correlation` are linear in rows; extrapolating
them from a slice is safe. **`hurst_on_stitched` is quadratic**, and the exponent gets worse
with size (1.05, then 1.24, then 2.02 across the three steps). Any linear extrapolation from
a short slice understates it by an order of magnitude, which is precisely how ~37 s was
arrived at.

### 5c. The arithmetic that closes the gap

```
old shard cost for hurst_on_stitched, 6x recompute : 6 x 437.6 s = 43.8 min
observed shard wall time                           : ~36 min
hurst's 6 expectancy trials (incl. bootstrap)      : 6 x 25.5 s  = 2.5 min
```
The 6x feature recompute **on its own** accounts for the whole ~36 min (43.8 min single-
process, and the shard ran under less contention than this measurement). Expectancy -- bootstrap
included -- contributes 2.5 min, about 7% of the observed shard time.

**Verdict: the hypothesis does not hold as the explanation of the ~10x residual.** There was no
residual to explain. `sum(feature) + 6 x per-trial` already over-covers the 36 min once
`hurst_on_stitched` is measured rather than extrapolated. The bootstrap is a real and large
cost (~51% of a trial, ~36% of the sweep) but it is a *sweep-wide* cost, not a `hurst`-specific one.

## 6. Why `hurst_on_stitched` is quadratic: it is memory, not arithmetic

`rolling_hurst` is closed-form vectorised over 20 lags, which should be O(n_rows x n_symbols x max_lag)
-- linear. Measured peak RSS on the **12-month** slice (`close` alone = 0.11 GB) with only
`hurst_on_stitched` running: **3.61 GB, or ~33x the close panel.**

At full-window scale `close` is 0.837 GB, so the same working set is **~28 GB on a 16 GB host**.
The 437.6 s is dominated by paging, and that is the entire source of the apparent quadratic
exponent. Two consequences:

1. `hurst_on_stitched`'s cost is **not portable** -- on a host with >32 GB it should return to
   roughly linear (~55-60 s extrapolated), and on this host it will get worse, not better,
   when 4 shards run concurrently. The ~29 min ETA in section 4 is optimistic for that reason.
2. It is the one feature where an implementation change (chunking `rolling_hurst` over row
   blocks) would buy more than every other optimisation in this document combined.

## 7. Caveats and provenance

- **Contention.** ~5 other agents were active on this host throughout. Repeat measurements of
  the same feature varied by up to 6x on the cheapest entries (`variance_ratio` measured 6.635 s
  and 1.114 s in two full-window runs). Treat sub-10-second rows as order-of-magnitude only;
  the three features that set the sweep cost (`sector_relative_return`, `hurst_on_stitched`,
  `median_pairwise_correlation` = 92% of `sum(feature_times)`) are large enough that contention
  noise does not change any conclusion.
- **Holdout untouched.** `results/holdout_lock.json` read `count: 7` before and `count: 7` after.
  Nothing under `data/` was written, moved, or added (rule 2); panels came from the
  existing `cache/` year files. No source file in `src/`, `tests/`, `scripts/` or `specs/` was modified.
- **Cache.** Repo `cache/` was 2,497,580 KB before and 2,810,264 KB after. **The growth is not
  from this task.** `NQ_CACHE_ROOT` was a private scratch dir for every run, that dir's
  `_materialized/` is empty, and all 5 new `_materialized/` entries in the repo cache carry
  `n_rows: 92520` with `n_symbols` in {1, 5, 8, 20, 149} -- no run here used 92,520 rows, and
  every run used 149 symbols. They belong to another concurrent agent.
- **Reproduce:** `scratchpad/retime/time_features.py` (measurement harness, scratch only),
  raw JSON in `scratchpad/retime/full_*.json` and `slice_*.json`.
