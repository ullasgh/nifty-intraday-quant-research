<!-- RECONNAISSANCE ONLY -- no formal module, by deliberate decision (see "Why no module").
     all_equity (149 names) x data/external/fo_bhavcopy/stock_oi_daily.parquet,
     2019-01-01..2025-07-31. The OI dataset starts 2019-01-01, one year later than the
     equity panel, so H5 gets 7 years and not 8. Last 12 months held out, NOT read.
     Numbers below were re-run by the lead from the agent's script, not transcribed. -->

# H5_oi_conditioned_direction

**Verdict:** KILLED

## The hypothesis

Daily F&O open-interest change as a slow conditioner on next-session intraday direction.
This was ranked last on evidence but FIRST on freshness: it is the only hypothesis in the
program built on a NON-PRICE dataset, and therefore the least likely to be already arbitraged.
H1-H4 were all price-derived and all died.

- **Signal** (per symbol, per session T): `doi_norm = doi[T-1] / oi[T-1]`, normalised because
  raw `doi` scales with contract size and is not comparable across names. Also tested as a
  within-session cross-sectional z-score, and as the classic four-quadrant
  `sign(doi[T-1]) * sign(r_prev)` read.
- **Trade:** enter at the 09:16 open on session T, exit at the 15:20 close on session T,
  cross-sectionally demeaned so the leg is market-neutral. One round trip per name per day.

## The lag discipline, and what it caught

The F&O bhavcopy for date `d` publishes AFTER that session closes. So a trader entering at
09:16 on session T has the `T-1` bhavcopy at the latest. `T-1` is resolved as the previous
USABLE session in the panel's own calendar, never a raw date subtraction.

The test was run BOTH ways on purpose -- correctly lagged, and deliberately same-day -- so the
lookahead would be visible as a number rather than silently avoided:

    variant                        spread_bps        t   n_sessions
    doi_norm (lagged, TRADABLE)         -7.27    -4.86         1618
    doi_norm (same-day, LEAKY)         -48.96   -21.59         1618

**The leak is worth 6.7x.** The same-day version clears the cost gate three times over and
would have looked like the best result in the entire program. It is not alpha; it is a signal
computed from the very session whose return it "predicts". This is the single most valuable
output of the H5 test.

## Per-year spread, correctly lagged (bps)

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| +3.17 | -9.87 | -12.60 | -8.20 | -13.37 | -5.45 | -2.41 |

Sign stability PASSES at 6 of 7 years negative. Statistically this is the second most robust
result in the program after H2 (t = -4.86). It still dies on magnitude.

## Why it dies -- and it is not marginal

Cost hurdle `NSEIntradayEquityCosts().round_trip_bps()`, measured at three sizes:

| notional per name | 1x round trip | 2x gate | edge 7.27 clears? |
|---|---|---|---|
| Rs 1L | 8.265 bps | 16.529 bps | **no -- fails even 1x** |
| Rs 10L | 4.017 bps | 8.033 bps | no |
| Rs 1Cr | 3.592 bps | 7.183 bps | nominally yes, by 0.09 bps |

At retail size the edge does not cover even a SINGLE round trip, so this is not a near-miss.
The Rs 1Cr row is not a rescue either: at Rs 1Cr per name across 149 names that is ~Rs 149Cr
gross, where realised market impact would dwarf a 0.09 bps margin, and the cost model here does
not price impact beyond the participation cap.

**And the decay is decisive at every size.** 2024 = -5.45 and 2025 = -2.41 fail every gate in
the table above. Since the universe is a fixed current-day list, the EARLY years are the
survivorship-inflated ones -- so 2024-2025 are the least-biased data available and they are the
weakest years. This is precisely the signature that killed H2 and H4.

## The four-quadrant variant is inconclusive, not merely small

`quadrant(lagged)` came in at -2.18 bps, t = -0.54 -- but it is also broken by construction:
the signal is ternary {-1, 0, +1}, so quintile bucketing collapses on **1,492 of 1,622
sessions**, leaving n = 126. It fails on methodology as well as magnitude. Per the
no-post-hoc-tuning rule it does NOT get a rescue attempt with different bucketing; a variant
search run after seeing the result is how false positives are manufactured.

Note `zscore(lagged)` is numerically identical to `doi_norm(lagged)`. That is expected, not a
bug: a within-session z-score is a monotonic linear transform, and the bucketing is rank-based.

## Coverage of the join

241,678 symbol-session pairs (1,622 usable T/T-1 session pairs x 149 symbols). **136 of 149
symbols** matched the F&O bhavcopy at least once; 13 never matched (AWL, BAJAJHLDNG, EMAMILTD,
ENRIN, GICRE, GLAXO, HYUNDAI, LTM, NIACL, PGHH, TATACAP, TMCV, TMPV -- mostly non-F&O or
recently listed). 4 session pairs had zero bhavcopy rows for their T-1 date and were DROPPED,
never filled.

## Why no module

H1-H3 each got a formal module because a verdict nobody can reproduce is not a verdict. H5's
reconnaissance is decisive on magnitude by a factor of ~2.3 against the retail gate and fails
outright in the two least-biased years, and the result was independently re-run by the lead
from the script rather than transcribed from the implementing agent's report. Building a module
here would add audit trail, not information. If H5 is ever revisited -- with an impact model
honest enough to price Rs 1Cr clips, which is the only regime where the number is even
arguable -- it should be formalised then.

## Conclusion

Killed on magnitude and on decay. This closes Phase 3 at **five hypotheses tested, five
killed**, and the last of them was the only non-price family in the program.
