<!-- REPRODUCIBLE via `.venv/bin/python scripts/recon_h4.py`. No formal module (see "Why no module").
     all_equity, 147 symbols contributing >=1 signal-conditioned observation, 2018-01-01..2025-07-31.
     Last 12 months held out, NOT read.

     SUPERSEDES the numbers first published here on 2026-08-18. Those came from a throwaway
     script that no longer exists and could NOT be reproduced -- see "Reproduction failure". -->

# H4_volatility_compression_expansion

**Verdict:** KILLED

## The hypothesis

Volatility compression resolving into expansion, with CONTINUATION in the direction of the
resolving move. This is the closest surviving relative of the strategy in the original request
(volume z-score + breakout), reframed as a measured conditional expectancy rather than a
hardcoded breakout rule.

- **Signal at 10:00 on session T:** the morning range (09:16-10:00) divided by a STRICTLY PRIOR
  20-session mean of that same daily morning range, times the SIGN of the morning move
  `log(close[10:00] / open[09:16])`. The prior window excludes session T; an inclusive window is
  a lookahead leak and measurably different (-7.38 vs -5.74 at the >2.0 threshold).
- **Trade:** enter at the 10:00 close, exit at the 15:20 close, cross-sectionally demeaned.
- Three pre-registered expansion thresholds.

## Result

With the 20-session warmup ENFORCED per symbol (the literal reading of "a 20-session mean";
223,976 observations, 1,607 sessions):

    threshold        edge bps
    expand > 1.0        -1.84
    expand > 1.5        -2.89
    expand > 2.0        -6.26

Without warmup enforcement, where a late-listing symbol's "20-session mean" can be a 1-sample
mean (258,023 observations, 1,847 sessions):

    threshold        edge bps        t
    expand > 1.0        -1.37    -1.83
    expand > 1.5        -2.06    -1.39
    expand > 2.0        -5.74    -2.06

## Per-year spread at the >2.0 threshold (bps)

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| -13.7 | -13.1 | -18.3 | +1.7 | +1.3 | +12.8 | -16.6 | -8.0 |

## Why it dies

1. **Magnitude.** The strongest reading is 6.26 bps against a **16.52904 bps** gate -- two times
   `NSEIntradayEquityCosts().round_trip_bps(1e5)`, which is the two-leg break-even ACCOUNTING
   IDENTITY for a long/short spread, not a safety margin. Short by ~2.6x. Because the cost model
   prices no spread and no impact, that gate is a FLOOR on the true hurdle.
2. **Significance.** Maximum |t| across all three thresholds is 2.06.
3. **The sign is WRONG.** Every threshold produces a NEGATIVE edge -- weak REVERSAL after an
   expansion, the direct opposite of the continuation hypothesis. As with H3, a verdict reporting
   only |edge| would be actively misleading here.
4. **The year pattern is a REGIME FLIP, not decay.** Negative 2018-2020, POSITIVE 2021-2023,
   negative again 2024-2025. An effect that changes sign for three consecutive years and back is
   not a tradable edge under any holding rule, and it is not the "concentrated in the earliest
   year" signature this verdict previously claimed.

## Reproduction failure -- the process finding, which outlives the result

The numbers first published here (-0.34 / -1.08 / -5.12 bps; 2018 -18.8, 2024 **+1.3**,
2025 -2.9; 145 symbols, 258,204 obs) came from a throwaway script that was never committed. It is
not recoverable -- absent from git history, stash, and `git fsck --lost-found`.

A verifier reimplemented H4 independently from the spec, matched `scripts/recon_h4.py` to the
last observation, then swept **34 alternative conventions**: one-sided vs two-sided masks, signed
vs raw returns, demeaned vs raw, strictly-prior vs inclusive windows, window in {5,10,20,30,60},
min_count in {1,10,15,20}, three morning-range windows, four sign sources, three entry and four
exit points, plus 1%/5% trimming and session-equal weighting.

**Across all 34, 2024 at the >2.0 threshold lands between -5.3 and -17.4 bps. The published +1.3
is unreachable.** Nor is it an outlier artifact: that bucket is n=1011, mean -16.6, median -19.8,
1%-trimmed -16.9, 5%-trimmed -19.6, session-equal-weighted -14.8.

The pooled headline numbers were within ~1 bps of reproducible values, so the lost script's
overall method was probably close. It is the PER-YEAR decomposition specifically that is
unreachable, which points at a year-attribution defect -- the same bug class already recorded in
this repo, where a 2-D mask flattened and destroyed row/symbol structure. That is a hypothesis,
not a demonstrated cause, and it is recorded as such.

**The lesson is the one this verdict originally violated: a verdict nobody can reproduce is not a
verdict.** H4 and H5 were published recon-only with no committed script. Both now have one.

## Known limitations of the current script, stated rather than hidden

- **The test as implemented is ONE-SIDED (long-only).** `expansion_ratio = ratio * sign` combined
  with `ratio > threshold` means a DOWN morning move yields a negative signed ratio that can never
  exceed a positive threshold, so 136,518 down-move symbol-sessions are dropped, along with 2,812
  flat-move ones. The two-sided reading measures -0.34 / -0.35 / -2.15 -- WEAKER. It is reported
  here rather than substituted, because switching conventions after seeing results is exactly the
  result-shopping this project forbids.
- The warmup-enforced and unenforced results are both printed, so the effect of that choice is
  visible rather than silently baked in.

## Why no module

The reconnaissance fails on magnitude, significance and direction simultaneously, and the sign is
opposite to the hypothesis, so no reading changes the decision. Unlike the original publication,
the script is now committed and the numbers are reproducible on demand.

## Conclusion

Killed on magnitude, on significance, and on direction. The year pattern is a sign-flipping regime
alternation rather than the survivorship-concentration signature previously claimed here -- a
different, and arguably more interesting, negative result.
