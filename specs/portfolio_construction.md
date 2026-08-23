# Spec G — portfolio construction

Status: spec, written before implementation. Author: lead. Date 2026-08-23.

## Starting position, from inventory

**`backtest/portfolio.py` contains ZERO weighting schemes.** `GrossNotionalSizer` (:297) and
`VolTargetSizer` (:369) both only CONSUME a weight vector. The single real scheme in the repo is
equal-weight-within-bucket at `plugins/xsec_zscore.py:197-204`.

**`VolTargetSizer` is UNWIRED.** `engine.py:34` imports only `GrossNotionalSizer`; `engine.py:65`
types the config field as `GrossNotionalSizer`; `engine.py:963` passes no `sigma`/`corr`, so
substituting it today raises `ValueError("sigma is required")`. Phase A4's "real volatility
targeting" is a tested library on no production path — the same shape as the 14-of-15 dead
functions in `features/market.py` and `costs.as_bps_of` before Phase C wired it.

`features/market.py:337 median_pairwise_correlation(returns, window, *, day_offsets=None,
min_names=5)` is the natural `corr=` source and nothing connects it.

## G1 — a weighting-scheme registry

    equal_weight        1/n within the selected set
    inverse_vol         w ∝ 1/sigma
    z_weight            w ∝ signal z-score
    z_over_vol          w ∝ z / sigma
    rank_weight         w ∝ cross-sectional rank of the signal
    risk_parity         equal risk contribution, diagonal first
    covariance_aware    uses an estimated covariance, not just the diagonal

Explicit registry, same discipline as `sweep_features.FEATURE_REGISTRY`: a declared list, not a
module glob, because the comparison across schemes is a multiple-testing exercise and the
denominator must be declared.

Every scheme is a pure function of (signal, sigma, corr, mask) -> weights summing to the declared
gross. **The present/tradable distinction is NOT optional** (repo rule 7): a scheme sees only
tradable names, and `present but not tradable` must never leak into a weight.

## G2 — wire `VolTargetSizer` into the engine

`BacktestConfig.sizer` accepts either sizer. `engine.py` supplies `sigma` and `corr`:

- `sigma` from `features/core.py`'s estimator family — `sigma_risk(sigma_ewma, floor=SIGMA_FLOOR)`
  with the derived floor (0.1143218, p1 of 71,580 measured symbol-session cells).
- `corr` from `median_pairwise_correlation`, closing the seam
  `specs/portfolio_vol_target.md:57` already named.

**The clip-breaks-target reporting must survive wiring.** `VolTargetSizer` sets `clip_binding`
(:513) and reports `vol_target_achieved` from final weights (:515) — that is the honest part of it,
and a wiring that drops those fields would turn real vol targeting back into a decorative one, which
is the exact defect Phase A4 was written to fix.

## G3 — the comparison, run as a pre-registered experiment

Compare all seven schemes on the tilt candidate and any Phase E survivors (currently none).

Seven schemes x however many candidates is a multiple-testing exercise BY CONSTRUCTION. So:
- one `ResearchContract` per run with `n_planned_trials` declared up front;
- `effective_n_trials` MEASURED, not assumed — these schemes are heavily correlated (inverse_vol
  and z_over_vol share a denominator; rank_weight and z_weight share an ordering), so the honest
  n_eff will be far below seven. Phase E measured 19.3 against 132 planned; expect a similar ratio;
- deflated Sharpe at that measured n_eff;
- promotion on RECENT years, never pooled. H2 is the standing example: sign-stable 8/8 years and a
  -24.30 bps pooled edge, killed on the recent-years gate at -9.62 bps by 2025.

## What this phase must NOT do

It must not become a search for the scheme with the best backtest. Turnover and capacity change
with the scheme, so a scheme that wins gross can lose net — Phase H's cost ladder is what settles
that, and G's output feeds it rather than concluding without it.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. Each of the seven schemes returns weights summing to the declared gross (within float tolerance).
2. Each respects `max_weight` and reports when the clip binds.
3. No scheme assigns weight to a name that is `present` but not `tradable` (rule 7).
4. `inverse_vol` with a zero/NaN sigma does not produce inf — `sigma_risk`'s floor applies, and a
   test asserts the floor is what prevents it.
5. `risk_parity` on an equal-vol, zero-correlation input degenerates EXACTLY to equal weight.
6. `covariance_aware` with a diagonal covariance degenerates EXACTLY to `inverse_vol`.
7. `VolTargetSizer` is reachable from `BacktestConfig` and `run_backtest` supplies `sigma`+`corr`;
   a test asserts a run completes with it and that `sigma_portfolio_ann` is populated.
8. A clip that breaks the vol target sets `clip_binding` and `vol_target_achieved` reflects the
   FINAL weights — asserted through the engine, not just on the sizer in isolation.
9. `effective_n_trials` across the seven schemes is measured and is LESS than 7 on correlated
   inputs — the anti-overstatement test.
10. Promotion is evaluated on the recent window; a fixture where pooled passes and recent fails
    yields NO promotion.
