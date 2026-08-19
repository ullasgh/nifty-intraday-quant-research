<!-- nifty_quant.research.hypotheses.h2_overnight_reversal.run_h2
     all_equity (149 names), 2018-01-01..2025-07-31; last 12 months held out, NOT read.
     Module verified against 63 tests (22 DeepSeek + 30 Luna written independently
     pre-implementation, + 11 internal). Cross-checked vs an independent reconnaissance:
     -24.58 vs -24.30 bps (~1%). Includes criterion 7 (recent-years cost gate). -->

# H2_overnight_reversal

**Verdict:** KILLED (INCOMPLETE: one or more criteria NOT_EVALUATED)

## Kill Criteria

- 1. Edge criterion: PASS (edge=-24.30 bps, 2x hurdle=16.53 bps)
- 2. Sign stability criterion: PASS (8/8 years)
- 3. Overlap correction criterion: PASS
- 4. Concentration criterion: FAIL (concentrated in bottom liquidity decile)
- 5. Latency profile criterion: NOT_EVALUATED
- 6. Deflated Sharpe criterion: NOT_EVALUATED (strategy_returns not supplied, trials=1)
- 7. Recent-years cost gate criterion: FAIL (years=2023,2024; edges=-19.41,-10.98 bps; mean=-15.20 bps; 2x hurdle=16.53 bps; dominant_sign='-'; excluded partial years: 2025)
- Observed direction: NEGATIVE top-minus-bottom spread (-24.3049 bps) -- consistent with REVERSAL, the hypothesized direction. horizon=1 bar(s) (horizon_mode='session'): one observation per symbol-day, non-overlapping, so no block-bootstrap overlap correction is needed even though Lens defaults to requesting it.
- UNIVERSE h2_panel WITH NO AS-OF DATE; 149 names; 15 of 149 names had no data in 2018. Returns before 2018 are survivorship-inflated.

## Context

- Cost hurdle: 8.26 bps (round-trip)
- SE method: block_bootstrap
- Seed: 0

## Per-year spread (bps)

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| -39.18 | -25.86 | -41.73 | -26.03 | -17.95 | -19.41 | -10.98 | -9.62 |
<!-- HAND-WRITTEN, PRESERVED -->

## Reconnaissance cross-check

The concentration result was recalibrated with a dedicated script
(`scripts/calibrate_concentration_threshold_v2.py`), 150 within-session permutation
replicates, seed 42, real data, corrected liquidity definition. The resulting null
distribution shows the old cutoff of 2.0 was hand-chosen with poor error control: it sits
at only the 27th percentile of pure noise, i.e. it fires on roughly 73% of noise
replicates. H2's own observed ratio of 2.7596 sits at the 62.3rd percentile of that same
null -- unremarkable, not evidence of a meaningful bottom-decile concentration. The
measured p95 threshold is 4.8695, at which the joint false-positive rate is 0.0067 (1
false positive in 150 replicates).

CORRECTION NOTE (2026-08-19): The above ratios (2.7596 observed, 4.8695 p95) differ from
the initially published 2.1189 and 5.5484 because the first calibration ran on the full
1-minute panel with an expanding-mean liquidity proxy, whereas production code reduces to
a two-rows-per-session checkpoint panel before Lens sees anything and uses a trailing-20-
session window in compute_prior_adv. These are different statistics measuring different
geometries. The recalibrated numbers reflect the true production path. The old 5.5484 was
too permissive for checkpoint-panel hypotheses: a future ratio between 4.87 and 5.55
would have wrongly passed. This discrepancy is the trailing-20-vs-expanding sensitivity
check that specs/lens_criterion_4_repair.md required and which was not run before
publication. Despite the changed numbers, criterion 4 still does not fire for H2, and by
a wider margin than the original 36.7th-vs-27th comparison implied.

The input definition was also wrong. The original criterion-4 analysis used raw share
count instead of rupee turnover, and used the full sample window to define liquidity
deciles, which is a lookahead. Under the corrected turnover-based, no-lookahead
definition, the "bottom liquidity decile" the concentration was blamed on actually
contained names like MRF, PAGEIND, BOSCHLTD, and SHREECEM -- high-share-price blue chips
trading roughly Rs 22 crore/day, not illiquid stocks. Criterion 4 was therefore wrong on
both the quantity it measured and the threshold it was judged against.

## Why this was built after the answer was known

This correction exists to audit a published kill decision, not to swap an unfavourable
result for a favourable one. The cost analysis showed criterion 7's outcome depended
entirely on the hardcoded Rs 1L clip-size assumption. The concentration analysis showed
criterion 4 combined a hand-chosen cutoff with a mis-specified liquidity measure and a
lookahead. Those are measurement failures, not economic evidence against overnight
reversal -- correcting them removes the original reasons for rejection, but it does not
remove the capacity constraint, the recent-listing dependence, or the two unevaluated
criteria below.

## Conclusion

H2's measured edge is real: mean spread -24.30 bps, t = -16.66, 8/8 years sign-consistent,
1,868 sessions. That does not make this a complete verdict. Status stays INCOMPLETE
because criteria 5 (latency profile) and 6 (deflated Sharpe) remain NOT_EVALUATED, and H2
must not be described as having survived, passed, or cleared review while that is true.

Both original kill reasons have collapsed under measurement, but they leave real questions
behind rather than disappearing. Criterion 7 was an artifact of the assumed clip size: at
Rs 10L the two-leg cost gate is 8.03 bps and the recent-years mean edge of -15.20 bps
clears it, but at Rs 1L the gate is 16.53 bps and the same edge still fails. Capacity and
clip size are therefore a genuine, unresolved limitation on what H2 is worth in practice,
not a solved problem. Criterion 4 was measuring noise rather than a real concentration
effect, per the calibration above (2.7596 observed vs. a measured p95 of 4.8695, joint
false-positive rate 0.0067) -- it provides no evidence H2 is genuinely concentrated in an
adverse liquidity segment.

The recent-year result carries its own composition limitation: in 2024 the full-universe
edge is -10.98 bps, but only -3.04 bps among continuous-coverage names present across the
full window. Recent-year performance leans on recently-listed names, which limits how much
of the apparent recent edge is attributable to the established universe versus newer,
potentially less mature or less liquid names.

The survivorship finding is already settled and is not a new measurement in this update:
earlier work via `recon_survivorship.py` established that the `all_equity` universe
contains zero delisted names within the panel window, so classic survivorship bias is
unmeasurable from this dataset -- nothing was dropped, so there is nothing to measure. That
earlier work also found new listings strengthen recent-year H2 performance, and that the
strongest early-year results (2018-2020) reflect later-listed names entering the sample
partway through, which is staggered listing exposure, not classic survivorship bias, and
must not be conflated with it.

The honest label: H2's edge is real, capacity-limited, and clip-size dependent, and it was
previously killed by a concentration criterion that was measuring noise rather than a real
effect. The overall verdict remains INCOMPLETE pending criteria 5 and 6.
