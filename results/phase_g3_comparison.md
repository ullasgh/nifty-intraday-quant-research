# Phase G3 Comparison: 7-Scheme Results

## Frozen Parameters

- Window: 2018-01-01 to 2025-07-31
- Holdout boundary: 2025-08-14
- Vol sigma halflife (sessions): 20.0
- Vol corr window (sessions): 30
- Capital: Rs 1,000,000
- Gross weight: 1.0
- Max weight per name: 0.05
- Recent window (for DSR): 252 sessions
- DSR promotion threshold: 0.95

## Calibration Null

Equal-weight scheme gross excess (must be ~0, |mean| < SE):
  Mean: -0.00 bps/day
  SE: 0.00 bps/day
  PASSED: True

## Warm-up and Degeneracy

- Sessions skipped for warmup: 51
- Sessions skipped for NaN rho (AMENDMENT 1): 0
- Sigma cells at floor: 0.07% (191 / 274011)
- Sessions processed: 1817

## Per-Scheme Statistics

| Scheme | Gross (bps/day) | Net (bps/day) | Turnover/day | Cost (bps/day) | Clip % | Recent DSR | Promoted |
|--------|-----------|-----------|-----------|----------|------|---------|---------|
| equal_weight         |      -0.00 |      -0.02 |     0.0018 |       0.02 |    0.0 |    0.0000 |     False |
| inverse_vol          |       0.73 |       0.53 |     0.0183 |       0.19 |    0.0 |    0.9482 |     False |
| z_weight             |      16.61 |       3.43 |     1.2409 |      13.18 |    0.0 |    0.4706 |     False |
| z_over_vol           |      14.98 |       1.73 |     1.2468 |      13.25 |    0.0 |    0.4355 |     False |
| rank_weight          |      16.52 |       3.34 |     1.2400 |      13.17 |    0.0 |    0.4708 |     False |
| risk_parity          |       0.73 |       0.53 |     0.0183 |       0.19 |    0.0 |    0.9482 |     False |
| covariance_aware     |       7.66 |       6.77 |     0.0836 |       0.89 |   39.8 |    0.9485 |     False |

## Effective Trials

- Effective n_trials: 1.6522
- Sessions per scheme: 1817

