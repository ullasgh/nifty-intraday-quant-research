<!-- RECONNAISSANCE ONLY -- no formal module, by deliberate decision (see "Why no module").
     all_equity, 145 symbols matching the feature's warmup requirement, 1,850 sessions,
     258,204 observations, 2018-01-01..2025-07-31. Last 12 months held out, NOT read. -->

# H4_volatility_compression_expansion

**Verdict:** KILLED

## The hypothesis

Volatility compression resolving into expansion, with CONTINUATION in the direction of the
resolving move. This is the closest surviving relative of the strategy in the original request
(volume z-score + breakout), reframed as a measured conditional expectancy rather than a
hardcoded breakout rule.

- **Signal at 10:00:** the morning range (09:16-10:00) divided by a strictly PRIOR 20-day mean
  range, multiplied by the sign of the morning move. The prior-window mean is strictly prior --
  a full-sample or inclusive window would be a lookahead leak.
- **Trade:** enter at the 10:00 close, exit at the 15:20 close, cross-sectionally demeaned so
  the leg is market-neutral.
- Three expansion thresholds tested, pre-registered before the run.

## Result

    threshold        edge bps       t   n_sessions     2018    2024    2025
    expand > 1.0        -0.34   -0.50         1850     -4.7    -0.5    -2.1
    expand > 1.5        -1.08   -0.70         1848     -7.7    +1.0    +0.6
    expand > 2.0        -5.12   -1.74         1756    -18.8    +1.3    -2.9

## Why it dies

1. **Magnitude.** The strongest reading is 5.12 bps against a 16.53 bps gate
   (`NSEIntradayEquityCosts().round_trip_bps(1e5)` = 8.26452, doubled) -- short by ~3.2x.
2. **Significance.** Maximum |t| across all three thresholds is 1.74.
3. **The sign is WRONG.** Every threshold produces a NEGATIVE edge, i.e. weak REVERSAL after an
   expansion -- the direct opposite of the continuation hypothesis. As with H3, a verdict that
   reported only |edge| would be actively misleading here.
4. **Concentration in the most survivorship-inflated year.** The strongest reading
   (expand > 2.0, -18.8 bps in 2018) is driven almost entirely by 2018, with 2024 at +1.3 and
   2025 at -2.9. The universe is a fixed current-day list, so 2018 is the MOST biased year in
   the sample and 2024-2025 the least. An effect that lives only in the earliest year and
   vanishes in the cleanest data is the H2 signature, and it is not tradable today.

Note the monotonic pattern across thresholds: the more extreme the compression filter, the
larger the apparent effect and the smaller the sample. That is the shape of a result being
driven by a shrinking tail, not of an effect strengthening.

## Why no module

The reconnaissance fails on all four axes simultaneously, and the direction is opposite to the
hypothesis, so there is no reading under which a formal module changes the decision. H1-H3 were
formalised because their verdicts were close enough to the gate, or structurally tricky enough
(H2's session-horizon bug), that reproducibility mattered. If a future variant of this family
is proposed, it should be specced and formalised then -- not reconstructed from this note.

## Conclusion

Killed on magnitude, on significance, on direction, and on year-concentration. Together with
H1-H3 and H5 this closes Phase 3 at five tested, five killed.
