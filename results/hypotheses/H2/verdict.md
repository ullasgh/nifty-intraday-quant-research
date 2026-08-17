<!-- nifty_quant.research.hypotheses.h2_overnight_reversal.run_h2
     all_equity (149 names), 2018-01-01..2025-07-31; last 12 months held out, NOT read.
     Module verified against 63 tests (22 DeepSeek + 30 Luna written independently
     pre-implementation, + 11 internal). Cross-checked vs an independent reconnaissance:
     -24.58 vs -24.30 bps (~1%). Includes criterion 7 (recent-years cost gate). -->

# H2_overnight_reversal

**Verdict:** KILLED

## Kill Criteria

- 1. Edge criterion: PASS (edge=-24.30 bps, 2x hurdle=16.53 bps)
- 2. Sign stability criterion: PASS (8/8 years)
- 3. Overlap correction criterion: PASS
- 4. Concentration criterion: FAIL (concentrated in bottom liquidity decile)
- 5. Latency profile criterion: NOT_EVALUATED
- 6. Deflated Sharpe criterion: PASS (trials=1)
- 7. Recent-years cost gate criterion: FAIL (years=2024,2025; edges=-10.98,-9.62 bps; mean=-10.30 bps; 2x hurdle=16.53 bps; dominant_sign='-')
- Observed direction: NEGATIVE top-minus-bottom spread (-24.3049 bps) -- consistent with REVERSAL, the hypothesized direction. horizon=1 bar(s) (horizon_mode='session'): one observation per symbol-day, non-overlapping, so no block-bootstrap overlap correction is needed even though Lens defaults to requesting it.
- UNIVERSE h2_panel WITH NO AS-OF DATE; 149 names; 15 of 149 names had no data in 2018. Returns before 2018 are survivorship-inflated.

## Context

- Cost hurdle: 8.26 bps (round-trip)
- SE method: block_bootstrap
- Seed: 0

