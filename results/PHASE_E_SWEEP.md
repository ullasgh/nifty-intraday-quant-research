# Phase E — conditional-analysis sweep

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
