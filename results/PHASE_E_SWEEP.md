# Phase E — conditional-analysis sweep

## DEFINITIVE RUN 2026-08-30 (supersedes the 2026-08-21 run below)

Re-run after `specs/phase_e_sweep.md` AMENDMENTS 4+5: real OHLCV loaded per-shard (union of
declared fields, no proxies), `rv_to_vix_ratio` adapter fixed, `variance_ratio` per-segment NaN
handling fixed, all-NaN features fail loudly. Panel 2018-01-01..2025-07-31, all_equity
(149 symbols), 701,863 bars. `holdout_intent="never"`; window ends two weeks before the holdout
boundary and `run_sweep` asserts it. Raw report:
`results/phase_e_sweep_report_2026-08-30.txt` (git-ignored, regenerable from
`results/phase_e_shards/` via `merge_shards`).

### Headline

**No new edge. This time the claim covers 19 of 22 features genuinely measured** (vs 10 last
run); the remaining 3 are structurally untestable by cross-sectional ranking (below), reported
as exclusions rather than fake nulls.

    TRIAL MATRIX (T x n_trials) = 514,070 x 114
      rows 701,863 -> 514,070 after the finite-row intersection (187,793 dropped, 26.76%)
    MEASURED effective_n_trials = 18.7607     (planned = 132)
    PBO (CSCV)                  = 0.0002
    var_trial_sharpes (MEASURED)= 0.0389957   (expected_max_sharpe -> DSR sr0 = 0.369744)

**Every deflated Sharpe is 0.0000.** PBO=0.0002 with all Sharpes ~0 means the sweep reliably
ranks nothing — stable noise, as in the prior run. `var_trial_sharpes = 0.0389957` is the value
criterion 6 should consume (supersedes 0.0310657 from the partially-degenerate run).

### Best raw Sharpe per feature (any horizon, pre-cost), all deflated to 0.0000

    volume_zscore               +0.1080  (h=1, decaying monotonically with horizon)
    efficiency_ratio            +0.0728
    opening_range               +0.0392
    variance_ratio              +0.0384  (now finite everywhere: the C2 fix worked in prod)
    vol_ratio                   +0.0349
    hurst_on_stitched           +0.0118
    rv_to_vix_ratio             +0.0058  (now measurable at all: the C1 fix worked in prod)
    rolling_beta                +0.0055
    parkinson_volatility        +0.0062
    garman_klass_volatility     +0.0040
    rogers_satchell_volatility  +0.0022
    amihud_illiquidity          -0.0010
    tradable_overnight_return   -0.0131
    overnight_return            -0.0170
    breakout_strength           -0.0526  (best is NEGATIVE; worst -0.1817 at h=1)
    signed_volume_proxy         -0.0981
    sector_relative_return      -0.0953
    close_location_value        -0.1668  (worst -1.1196 at h=1)
    beta_residual_return        -0.1066  (worst -0.8367 at h=1)

None approaches the E4 promotion bar. `volume_zscore`'s +0.1080 at h=1 pre-cost echoes the
killed `volume_breakout` finding exactly: a short-horizon volume signal whose economics died at
1 minute of latency and the spread — and its deflated Sharpe here is 0.0000 anyway.

### Excluded (structural, not defects): 3 broadcast features

`breadth`, `cross_sectional_dispersion`, `median_pairwise_correlation` are market-level row
stats broadcast to every symbol — cross-sectionally constant, so rank bucketing is undefined.
All-NaN spread series, excluded from the matrix and reported as exclusions. They could only be
tested as regime conditioners on another signal, a different harness. (`variance_ratio`, in
this category last run, is per-symbol and IS now measured.)

### No trials raised. 132 planned, 132 recorded.

### What this means for Phase F

All four v2 components now have measured verdicts, and all four fail the spec's own bar:

    hurst_on_stitched     +0.0118, no monotone response  -> does not go in (unchanged)
    beta_residual_return  -0.8367 at h=1                 -> does not go in (unchanged)
    breakout_strength     NEGATIVE at every horizon      -> newly measured, dead
    volume_zscore         +0.1080 raw, deflated 0.0000   -> newly measured, does not clear

Per the kill criteria declared in `specs/volume_breakout_v2.md` BEFORE the build, v2 cannot be
assembled. **Phase F is dead on its own pre-declared terms** — pending only the formal
write-up against the spec's kill-criteria list.

### Cost actuals (sequential, --workers 1, nice 15, 16 GB host)

Shards (LPT-balanced): 12 / 13 / 24 / 33 min; merge ~37 min cold (~3 min warm — dominated by
cold shard I/O + `pbo_cscv`'s 12,870 full-matrix passes). The 2026-08-23 cost table under-reads
full-panel cost ~2x uniformly. DO NOT run 4 workers on a 16 GB host: post-AMENDMENT-4 shards
load the OHLCV union (up to ~5x the close-only RSS) and 4-way concurrency swap-thrashed the
machine. Owed: pbo_cscv via per-block sufficient statistics (~100x), runner persists its own
merged report (print-only nearly lost this run's result).

---

# SUPERSEDED — run 2026-08-21 (7 features fed degenerate proxies; kept as history)

Run 2026-08-21. Panel 2018-01-01..2025-07-31, all_equity (149 symbols), 701,863 bars, 1,880
sessions. Contract `holdout_intent="never"`; the window ends 2025-07-31, two weeks before the
holdout boundary at 2025-08-14, and `run_sweep` asserts this.

## Headline

**No new edge found. But the sweep genuinely measured 10 of 22 features, not 22 — read the
exclusions before reading the table.**

    TRIAL MATRIX (T x n_trials) = 521,107 x 60
      rows 701,863 -> 521,107 after the finite-row intersection (180,756 dropped, 25.75%)
    MEASURED effective_n_trials = 19.2969      (planned = 132)
    PBO (CSCV)                  = 0.0267
    var_trial_sharpes (MEASURED)= 0.0310657    (expected_max_sharpe -> DSR sr0 = 0.332222)

**Every deflated Sharpe is 0.0000.** Best raw Sharpe in the sweep is `vol_ratio` at h=1, **0.0349**,
before costs, decaying monotonically with horizon (0.0349 / 0.0270 / 0.0221 / 0.0145 / 0.0100 /
0.0075) -- the signature of noise, not signal.

### Reading the three headline numbers

**`n_eff = 19.3` against 132 planned.** The registry is correlated by construction (three
volatility estimators on the same bars; Hurst and variance-ratio measuring the same persistence),
so 132 trials were ~19 independent looks. Reporting 132 would overstate the search 7x; reporting 1
would understate it as badly.

**PBO = 0.0267 is NOT good news here.** A low probability of backtest overfitting means the ranking
is stable out of sample. With every Sharpe near zero, that says the sweep reliably identifies
*nothing*. Stable noise is still noise.

## Best raw Sharpe per measured feature (any horizon, pre-cost)

    efficiency_ratio            +0.0728
    vol_ratio                   +0.0349
    opening_range               +0.0318
    hurst_on_stitched           +0.0118
    tradable_overnight_return   +0.0065
    rolling_beta                +0.0055
    amihud_illiquidity          -0.0010
    overnight_return            -0.0170
    sector_relative_return      -0.0953
    beta_residual_return        -0.1066

None clears the E4 promotion bar (2x cost hurdle, |spread_t| > 1.96 on the corrected SE, monotone
buckets, deflated Sharpe at the measured n_eff), evaluated on recent years rather than pooled.

## The 12 features that were NOT measured, and why

This is the part that qualifies the headline. **Two distinct causes -- one is a real finding, the
other is a defect in this harness.**

### (a) DEFECT -- 7 features were fed degenerate inputs and never actually tested

`run_sweep`'s pinned signature takes only `close` + `day_offsets` (`specs/phase_e_sweep.md`
AMENDMENT 1). The registry therefore synthesises proxies: `volume = ones_like(close)` and
`high = low = open = close`. Consequences:

    volume_zscore              z-score of a CONSTANT volume -> all-NaN
    signed_volume_proxy        same
    breakout_strength          high == close -> no breakout range -> all-NaN
    parkinson_volatility       log(H/L) == 0 -> zero variance -> all-NaN
    garman_klass_volatility    same
    rogers_satchell_volatility same
    close_location_value       (C-L)/(H-L) -> 0/0 -> all-NaN

**These were not tested. Recording them as "no signal" would be false.** The implementer flagged
the proxy assumption and predicted the degeneration; I accepted it as "a legitimate recorded
result", and that was wrong -- a measurement that cannot happen is not a null result.

**Fix required before any conclusion covers them:** `run_sweep` must accept the OHLCV fields the
registry needs, or the registry must declare its required fields and the runner load them. That is
a signature change with a blast radius (both dual suites assert the current shape), so it is
recorded here rather than patched silently.

### (b) FINDING -- 4 features are structurally unsuited to cross-sectional ranking

    breadth                      market-level scalar, broadcast to every symbol
    cross_sectional_dispersion   same
    median_pairwise_correlation  same
    variance_ratio               time-invariant per-symbol characteristic, broadcast

Ranking a value that is identical across symbols at each bar produces no dispersion, hence no
buckets, hence no spread. This IS a legitimate result: these are market-state or per-symbol-static
quantities, not cross-sectional signals, and they cannot be swept this way. They would need a
different harness -- e.g. as regime CONDITIONERS on another signal, not as the ranked signal itself.

### (c) FAILURE -- 1 feature raised on all 6 horizons

    rv_to_vix_ratio   ValueError: realized_vol_ann must be a 1-D array

A genuine defect in a feature that has unit tests and had never been run on real data. Recorded as
a failed trial per obligation 11, not dropped.

## What this means for Phase F

`volume_breakout` v2 was designed around four components. Two of them now have measured verdicts,
and both are discouraging:

- `hurst_on_stitched` best raw Sharpe **+0.0118** across all horizons -- so `H > 0.55` has no
  monotone conditional response to rest on. Per `specs/volume_breakout_v2.md`, a component with no
  measured response does NOT go into v2.
- `beta_residual_return` **-0.1066**, the worst in the sweep.

The other two -- `breakout_strength` and `volume_zscore` -- fall in category (a) and remain
UNMEASURED. Phase F cannot honestly proceed on them until the OHLCV plumbing is fixed.

## Cost, for whoever re-runs this

Four features dominate: `rolling_beta` ~47 min, `hurst_on_stitched` ~36 min,
`beta_residual_return` ~85 min, `median_pairwise_correlation` ~25 min. The other eighteen took
about ten minutes combined. The merge itself (132-column correlation + CSCV) took ~25 min.
Profiling those four is worth more than it costs -- the block bootstrap went 44.02s -> 3.02s with a
bit-identical RNG stream once someone looked.
