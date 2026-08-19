# Nifty Intraday Quantitative Research -- Final Report

## The Question and the Answer

Can we beat the Nifty index on a yearly basis using 1-minute intraday data on Nifty-100 equities? We tested five hypotheses ranked in advance, each a cross-sectional signal to enter at a specific morning time and exit at 15:20 close, measured against the two-leg cost hurdle for a long/short spread. All five were killed. Two follow-on reconstructions tested a long-only index-relative tilt -- a one-leg construction using H2's overnight reversal signal. The tilt has real gross excess (21-31% annualised) but is killed by turnover in the daily-rebalance form. A low-turnover variant (weight smoothing, a=0.10) is net-positive in all eight years pooled and in 2024-2025 individually, but four specific unverified conditions must clear before this is a result: a short-window significance test, a plateau-vs-boundary check, a liquidity decomposition, and read of the locked holdout. This is a candidate, not yet confirmed.

---

## The Five Hypotheses

| Hypothesis | Signal | Measured Edge | t-stat | Verdict | Primary Kill Reason |
|---|---|---|---|---|---|
| **H1** | Index morning -> afternoon momentum | 0.09 bps | 1.95 | **KILLED** | Magnitude 180x below cost gate |
| **H2** | Overnight cross-sectional reversal | -24.30 bps | -16.66 | **KILLED** | Concentrated in illiquid decile; recent-year decay |
| **H3** | Intraday cross-sectional reversal | +1.21 bps | 1.04 | **KILLED** | Sign is momentum, not reversal; magnitude trivial |
| **H4** | Volatility compression -> expansion | -6.26 bps | -2.06 | **KILLED** | Sign reversed (reversal, not continuation); regime alternation |
| **H5** | F&O open-interest conditioner | -7.27 bps | -4.86 | **KILLED** | Edge fails at retail clip size; strong recent-year decay |

### H1 -- Market Intraday Momentum (NIFTY50 level)

Signal: index 09:16 close versus index 10:00 close. Edge of 0.0879 bps against a two-leg hurdle of 16.53 bps -- 188x too small. Sign stable in only 5 of 8 years (2018-2025). The tested direction (morning index momentum continuing to afternoon) does not exist on the Nifty-50 spot price. Verified against 54 tests from two independent authors.

### H2 -- Overnight Cross-Sectional Reversal (all_equity, 149 names)

This is the program's single largest measured effect. Signal: the cross-sectional rank of each name's overnight (close to open) log return; at 10:00 entry the top decile of overnight losers reversed at -24.30 bps net of transaction costs, with t = -16.66 over 1,867 sessions, sign stable 8 of 8 years. The effect is real and large, and would be a survivor except for two kill reasons:

1. **Concentration:** The edge concentrates in the bottom liquidity decile. A 10x5 double sort (10 liquidity deciles, 5 feature quintiles) reports a concentration ratio of 2.1189 (bottom decile spread / median spread across deciles). This sits 6% above a hand-chosen cutoff of 2.0. **This ratio is PENDING a null-distribution calibration**: if the derived p95 from `scripts/calibrate_concentration_threshold_v2.py` (currently running) exceeds 2.1189, criterion 4 should not fire and H2 becomes "real but capacity-limited" rather than killed. Any update to this section will appear here when that measurement completes.

2. **Recent-year decay:** The 2024 edge is -10.98 bps and 2025 is -9.62 bps, both below the 16.53 bps two-leg cost hurdle at Rs 1L. At Rs 10L the hurdle is 8.03 bps, and both years still fail. This edge has been arbitraged or decayed below the point of tradability in current data.

**The survivorship argument, previously cited in four verdicts, has been measured and REFUTED.**
It was asserted, never measured. `scripts/recon_survivorship.py` rebuilt the universe two ways --
all 149 names vs the 129 with continuous 2018-2025 coverage -- and the measured deltas run
BACKWARD to the claim: tiny and wrong-signed in the early years (2018 +1.54, 2019 +0.57 bps) and
LARGEST in the recent ones (2023 +7.52, 2024 +7.94).

Two things matter more than the refutation:

1. **Survivorship is UNMEASURABLE from this dataset.** Zero of the 149 names stop before the
   window end -- the wealth-destroyers (YESBANK, IDEA, ZEEL, PNB, BHEL, SAIL) all collapsed in
   price but stayed LISTED. Survivorship concerns DELISTED names, and those were never
   downloaded. Their absence is not evidence against bias; it is evidence the data cannot speak
   to it. Claim neither direction.
2. **What the deltas actually measure is NEW LISTINGS.** 18 names IPO'd inside the window
   (IRCTC, SBICARD, NYKAA, ETERNAL, JIOFIN, LICI, HYUNDAI...), and they ADD reversal signal in
   recent years: **2024 is -10.98 bps on the full universe but only -3.04 on continuous-coverage
   names.** So H2's recent edge is substantially carried by recently-listed stocks -- the ones
   with the least history and the least borrow availability for a short leg. On a like-for-like
   universe H2's decay is STEEPER (2018 -37.64 -> 2024 -3.04), not shallower.

Note the excluded-names test is not the same experiment as restoring delistings, and this report
does not claim otherwise.

### H3 -- Intraday Cross-Sectional Reversal (all_equity, 149 names)

Signal: cross-sectional rank of the morning (09:16 to 10:00) log return. Measured edge +1.21 bps, t = 1.04, across 1,866 sessions -- 14x below the cost gate. More critically, the observed direction is **momentum, not reversal**: the top quintile of morning winners outperformed on the afternoon's trade, the opposite of the hypothesized reversal. Sign flipped in 2020 and stayed flipped; two opposing regimes cancel in the pooled number, and neither clears the gate alone. The verdict holds across 52 tests verified against two independent authors.

The two-way cross-sectional reversal (overnight and intraday) teaches a liquidity lesson: overnight is where order imbalance accumulates with no continuous trading to absorb it; intraday, continuous liquidity clears imbalance as it arrives, leaving nothing to revert. Further reversal variants on price-derived intraday signals are low-value.

### H4 -- Volatility Compression Expanding into Continuation (all_equity, 147 symbols)

Signal at 10:00: the morning range (09:16-10:00 high-low spread) divided by a strictly-prior 20-session rolling mean range, multiplied by the sign of the morning close-to-open log return. Trade: 10:00 close to 15:20 close, cross-sectionally demeaned. Three thresholds pre-registered; strongest result at expansion > 2.0 was -6.26 bps, t = -2.06, against the 16.53 bps gate -- 2.6x short and 40% below significance.

The sign is **consistently wrong**: all three thresholds produce a negative edge (weak reversal after expansion), opposite to the hypothesized continuation. The per-year pattern was previously reported as "concentrated in 2018" but under scrutiny with 34 alternative bucketing conventions swept, that reading was **unreproducible**. The true pattern is a **regime alternation**: negative 2018-2020, positive 2021-2023, negative again 2024-2025. An effect that changes sign for three consecutive years and back is not tradable under any holding rule. Why no module: the signal is decisive on magnitude, significance, and direction simultaneously, so no methodological refinement changes the kill.

### H5 -- F&O Open-Interest Conditioner (all_equity x OI bhavcopy, 136 names, 7 years)

Signal: daily F&O open-interest change, normalised by open interest from the previous usable session (`doi[T-1] / oi[T-1]`). Trade: 09:16 open to 15:20 close, cross-sectionally demeaned. One round trip per name per session. The key methodological finding is an **explicit lookahead audit**: the signal was tested both correctly lagged (using T-1 OI) and deliberately with same-day lookahead (using T OI).

| Variant | Edge | t-stat | Leak Multiple |
|---|---|---|---|
| Correctly lagged (tradable) | -7.27 bps | -4.86 | -- |
| Same-day lookahead (leaky) | -48.96 bps | -21.59 | **6.7x** |

The correctly lagged edge is the second-most statistically robust result in the program (only H2's -24.30 bps is stronger). Yet it fails on magnitude: the cost hurdle at three sizes is shown below.

| Notional per name | 1x Round Trip | 2x Gate | Edge -7.27 Clears? |
|---|---|---|---|
| Rs 1L | 8.265 bps | 16.529 bps | **No** -- fails even 1x |
| Rs 10L | 4.017 bps | 8.033 bps | **No** |
| Rs 1Cr | 3.592 bps | 7.183 bps | Nominally yes, by 0.09 bps |

At retail size (Rs 1L) the edge does not cover a single round trip. The Rs 1Cr row appears marginal until context: at that size across 149 names (~Rs 149Cr gross), realised market impact would dwarf a 0.09 bps margin, and the cost model prices zero impact. **Decay is decisive at all sizes.** 2024 = -5.45 bps and 2025 = -2.41 bps, both below every gate.
(Do NOT justify weighting recent years by survivorship -- that argument was measured and refuted;
see above. The reason to weight them is simply that they are the most recent evidence of what the
effect does now.)

---

## What Was Found -- and Why It Cannot Be Traded

Effects exist and are measurable, but they live where transaction costs prohibit trading:

1. **H2's overnight reversal is real** (-24.30 bps, t = -16.66, 8 of 8 years sign-consistent, 63 tests from three independent authors). It concentrates entirely in symbols with the lowest rupee-volume liquidity. At Rs 1L per-name clip size it fails both the current-cost gate and the recent-years gate; at Rs 10L it passes the cost gate but recent years (2024 -10.98 bps, 2025 -9.62 bps) still fail. This edge has decayed or been arbitraged below tradability.

2. **H5's open-interest conditioner is real** (-7.27 bps lagged, t = -4.86, 6 of 7 years). It is also weak -- below the cost of a single round trip at retail size, and 2024-2025 are deteriorating. The non-price dataset provides no rescue from the cost problem.

3. **H1, H3, and H4 are not real.** H1 has trivial magnitude. H3's sign flips across regime; it measures intraday momentum, the opposite of the hypothesized reversal. H4's sign is also reversed (reversal after expansion, not continuation) and its year pattern is a regime alternation, not a drift or skill degradation.

**The structural finding:** Every effect with real magnitude is overnight (when order imbalance accumulates) or concentrated in the least-liquid tail (where reversal pressure is strongest but costs are prohibitive). Nothing intraday and liquid clears costs. On Nifty-100 large caps, price-derived signals read as arbitraged out, not undiscovered.

---

## The Cost Arithmetic -- Why It Dominates

Every hypothesis is a cross-sectional long/short spread: long the top signal quintile, short the bottom quintile, equal weight, one round trip per name per day. P&L is `N x spread_bps` (notional per leg times the quintile spread); cost is `2 x round_trip_bps(N)` -- two legs, each hitting both bid and ask. Break-even at `spread > 2 x round_trip_bps`. **The 2x is not a safety margin; it is the two-leg break-even accounting identity.**

The cost model here prices only brokerage, STT, and statutory charges. It does not price bid-ask spread or market impact. The repo's own backtester adds `half_spread_bps=1.5` (a bid-ask assumption) and `impact_coef=10.0` (four fills per round trip), pushing the true hurdle several basis points higher. The 2x gates shown below are therefore a **floor on the true cost.**

### Cost Hurdles by Notional Size

| Size (per-leg) | 1x Round Trip | 2x Break-Even Gate |
|---|---|---|
| Rs 1L | 8.26452 bps | **16.52904 bps** |
| Rs 10L | 4.01652 bps | **8.03304 bps** |
| Rs 1Cr | 3.59172 bps | **7.18344 bps** |

H2 at -24.30 bps passes the Rs 1L gate (16.53 bps) but fails concentration and recent-year decay. H5 at -7.27 bps fails the Rs 1L gate, passes Rs 10L / Rs 1Cr on the pooled statistic but fails in 2024-2025. The cost model choice is not academic: a strategy that "clears" costs under one size assumption may not under another.

---

## What Was Ruled Out and What Remains Open

### Ruled Out (by explicit measurement)

**H5's multi-day holding period (k = 1, 2, 3, 5, 10 usable sessions):** The per-day rate collapses with holding length. The cumulative k-day gross edge does not grow faster than one-day, so amortising the cost over multiple days does not rescue the hurdle. This axis is closed.

**The survivorship argument:** Measured via `scripts/recon_survivorship.py`. The all_equity universe has zero delisted names within the panel window (survivorship is unmeasurable here). New listings actually strengthen recent-year H2 performance. Early-years inflation comes from later-listed names, not survivorship. Strike this argument from all verdicts.

**H4's "concentrated in 2018" claim:** Recon numbers were unreproducible (34 bucketing conventions swept; the published 2024 +1.3 is unreachable, true range -5.3 to -17.4 bps). The real pattern is a regime alternation (negative 2018-2020, positive 2021-2023, negative again 2024-2025), not a skill decay or concentration story.

**Long-only index-relative tilt, daily-rebalance form:** Tested on H2's overnight reversal signal. Construction halves the two-leg hurdle to ~8.26 bps at Rs 1L and starts from the index return rather than zero. Real gross index-relative excess of 21-31% annualised, but entirely consumed by turnover: the cross-sectional overnight-return rank reshuffles the portfolio 125-150% every session. Net is flat to negative in recent years across all combinations. The binding constraint is structural turnover from daily rebalancing, not the cost hurdle. This axis is closed.

### Still Open (Unverified Conditions on the Low-Turnover Candidate)

**The low-turnover variant (weight smoothing, a=0.10)** passes the pooled test and survives 2024-2025 in every combination, but four specific conditions remain unverified:

1. **2024-2025-only significance:** Current t-stats are pooled over the full 2018-2025 window and dominated by early-year gross. No short-window power test has been run on the weak recent years.

2. **Grid boundary:** a=0.10 is the most extreme point on the pre-registered grid. A plateau-vs-spike check is running to rule out over-fitting at the edge of the search space.

3. **Liquidity decomposition:** H2 edge concentrates in the least-liquid decile. The low-turnover book has not been decomposed by liquidity bucket; if it inherits that concentration, it is not investable at the sizes tested.

4. **Holdout (last 12 months):** The out-of-sample verification set remains locked and unread.

---

## The Tilt Findings -- A Candidate That Clears Costs (With Unverified Conditions)

### Long-Only Index-Relative Tilt: Real Gross Excess, Killed by Turnover

A one-leg long-only tilt uses H2's overnight reversal signal, known at 09:16, to overweight losers in an equal-weight Nifty-100 portfolio. Mild tilt = clipped-rank weights (zero on the top half, rising toward the biggest overnight loser). Aggressive = bottom quintile equal-weighted. Entry 09:16 open, exit 15:20 close. Cost charged only on turnover (one leg, not two).

The gross excess is real and large:

| Variant | Gross (bps/day) | Annualised | Turnover/day | NET (bps/day) | t-stat |
|---|---|---|---|---|---|
| Mild / Full | 10.03 | 25.27% | 1.2573 | -0.36 | 11.92 |
| Aggressive / Full | 12.49 | 31.46% | 1.5026 | +0.07 | 10.96 |
| Mild / Continuous | 8.60 | 21.67% | 1.2624 | -1.83 | 10.37 |
| Aggressive / Continuous | 10.66 | 26.86% | 1.5085 | -1.81 | 9.49 |

21-31% annualised index-relative excess at t up to 11.9, across 1,856 sessions. Entirely consumed by turnover. The portfolio rotates 125-150% every day because a cross-sectional overnight-return rank reshuffles the loser bucket almost completely session to session. Pooled net is between -1.83 and +0.07 bps/day (indistinguishable from zero) and clearly negative in recent years across all combinations (2024 range: -6.02 to -11.43 bps/day; 2025 range: -7.14 to -10.25 bps/day).

**Why this kill differs from the other five:** It is not the cost hurdle. Halving the hurdle (from two legs to one) changed nothing, because the binding constraint is not the leg count -- it is the construction itself. A daily-rebalanced cross-sectional rank has structural turnover near a full book rotation.

### Low-Turnover Tilt: Net-Positive in Every Year

The tilt was killed by turnover, not by the cost gate itself. Testing lower-turnover constructions on the same signal: periodic rebalance (k in {1,2,3,5,10}) and weight smoothing (a in {1.0, 0.5, 0.25, 0.1}, where w_t = a*w_t-1 + (1-a)*target_t). Pre-registered grids only; control (k=1, a=1.0) reproduces the daily baseline to two decimals.

The single construction net-positive in both 2024 and 2025, across all four (tilt, universe) combinations, is smoothing a=0.10 -- turnover cut roughly 9-11x:

| Combo | Pooled NET (bps/day) | Annualised | t-stat | 2024 | 2025 |
|---|---|---|---|---|---|
| Mild / Full | 2.29 | 5.77% | 12.60 | +1.92 | +0.84 |
| Mild / Continuous | 1.92 | 4.83% | 11.55 | +1.13 | +0.51 |
| Aggressive / Full | 2.40 | 6.05% | 10.89 | +1.57 | +0.56 |
| Aggressive / Continuous | 1.97 | 4.97% | 9.98 | +0.55 | +0.26 |

Mild/full is net-positive in all eight years: 3.14, 1.65, 3.11, 2.14, 2.34, 2.56, 1.92, 0.84 bps/day. It survives on the continuous-coverage universe, so it is not purely a newly-listed-names artifact. No construction in this program has cleared costs in every measured year before.

### Why This Is Not Yet a Result -- The Four Unverified Conditions

State these prominently because they gate the finding:

1. **No 2024-2025-only significance test has been run.** Every t-stat quoted is pooled over the entire window and dominated by larger early-year gross. The weakest surviving case is +0.26 bps/day (aggressive/continuous 2025). Rule 9 applies directly: a small effect-to-SE ratio in a short sub-window is not a confirmed edge without an explicit power check.

2. **a=0.10 is the most extreme point on the pre-registered grid.** Results that appear only at the boundary of a tested range are exactly where over-fitting hides. A plateau-vs-spike check is running.

3. **No liquidity decomposition has been run on the low-turnover book.** H2's overnight reversal edge concentrates in the least-liquid decile. If this construction inherits that, it is not investable at the sizes tested.

4. **The holdout (last 12 months) remains locked and unread.** No out-of-sample verification on withheld data.

**The degenerate corner, recorded honestly:** Hysteresis b=1.00 on the aggressive tilt collapses turnover to ~0.0005-0.0086/day (a buy-once-never-trade book whose few-name concentration makes the no-trade threshold enormous relative to any weight change). Its positive recent numbers reflect almost no rebalancing over 7.5 years and must not be read as validating the hysteresis mechanism.

---

## Method Notes -- The Findings That Outlived the Results

These are the operational lessons most relevant to future work on this dataset.

### Reconnaissance as an Independent Oracle

Every formal hypothesis result was independently reconstructed via a separate reconnaissance script (a different code path, same window, same panel). **This caught errors the test suites did not.** Example: H2's formal module reported -24.58 bps, the reconnaissance oracle -24.30 bps. Both are close enough that sampling noise explains it. But H2's *horizon measurement* (an accidentally-lagged observation) was off by 25,000x: the module measured one minute where it should have measured 373 minutes. Fifty-two tests from two independent authors all passed against unfixed code because the horizon parameter was silently baked wrong into every test fixture. **A verdict that nobody can reproduce from independent code is not a verdict.** Reconnaissance proved more valuable than additional test coverage.

### Dual Independent Test Suites and What They Caught

Every formal module (H1, H2, H3) was written from a spec by two independent test suites, each written blind (neither author saw the other's tests or the implementation). This caught divergences in spec interpretation: the same spec read two ways revealed ambiguities. Example: the overlap-correction design rule (block-bootstrap resampling must never straddle a session) was clarified by asking "how would I test this?" -- and the answer ("make the blocks observable") forced an implementation choice that had been latent in prose.

### The Lookahead Audit -- Deliberately Leaky vs. Correctly Lagged

H5 was tested both ways explicitly:

| Variant | Edge | Status |
|---|---|---|
| Same-day (leaky) | -48.96 bps | Looks like the program's best result |
| Correctly lagged (tradable) | -7.27 bps | Fails at retail size |

**The leak is worth 6.7x.** If H5 were tested only as same-day, the verdict would have been a survivor, because -48.96 bps clears the cost gate three times over. No agent is going to volunteer a test where their signal loses 87% of its power; this audit had to be explicit in the spec and in the output. The difference is a signal computed from the very session whose return it "predicts" -- a lookahead that is unattainable in practice.

### Judging on Recent Years, Not Pooled Statistics

This was learned the hard way via H2:

| Metric | Pooled | 2024 | 2025 | Status |
|---|---|---|---|---|
| Edge (bps) | -24.30 | -10.98 | -9.62 | Pooled passes gate, recent fails |
| t-stat | -16.66 | | | |

H2's pooled edge clears costs; its recent years do not. A pooled average therefore overstates what
is tradable today, and every verdict reports per-year numbers so this is visible. A hypothesis
whose most recent complete years fail the cost gate is not a survivor regardless of its pooled
t-stat.

**Note the justification carefully.** The reason to weight recent years is that they are the most
recent evidence of what the effect does NOW -- not survivorship. The survivorship version of this
argument appeared in four verdicts, was never measured, and was refuted when it finally was. This
report keeps the conclusion and discards the reasoning that turned out to be wrong.

### Measurement Errors I Corrected (and the Tests Would Have Caught)

**Median convention:** The concentration-ratio computation used `np.median(spreads)` (which averages the middle two values on even-length arrays) instead of production's upper median. On a thin margin (2.1189 vs. 2.0 cutoff), this choice matters.

**Bucket geometry mismatch:** `expectancy_by_liquidity_decile` couples the decile loop and the feature-bucketing to one `n_buckets` parameter, so it cannot express production's 10 liquidity deciles x 5 feature quintiles in a single call. The implementer must either add a `feature_n_buckets` parameter or expose decile assignment directly for use by a caller.

**Min-names floor in cross-sectional rank:** The `cross_sectional_rank` function silently returns NaN when a row has fewer than 5 finite values. Below ~44 symbols (enough to fill 5 per decile at 8+ per bucket), every decile returns 0.0 spread against both old and new code. A 2-3 symbol test fixture is useless for regression testing; any future test asserting on decile spreads must use >= 50 symbols, confirmed explicitly in the test.

**Causality in decay measurement:** The concentration ratio uses `np.quantile` on the full panel (including future data) to define decile boundaries. When re-measured with causal bucketing (cross-sectional rank on strictly-prior rupee-volume ADV), the ratio dropped from 2.4538 (full-sample, share volume) to 1.9959 (causal, share volume) to 2.1189 (causal, rupee volume, production geometry). The causal 1.9959 looked like a rescue until the bucketing geometry correction (10x5, not 10x10). Causality matters, and so does the unit -- share count vs. rupee turnover partition the bottom decile almost entirely differently.

---

## Conclusion

We set out to beat the Nifty-100 index on a yearly basis using 1-minute intraday price and volume data. Five hypotheses, ranked in advance and tested under independent dual-suite verification and reconstruction-based oracle audit, were killed by cost hurdles or magnitude. Two post-hoc reconstructions tested a long-only index-relative tilt using the largest real effect (overnight reversal, -24.30 bps, t = -16.66). The daily-rebalance form is killed by structural turnover, not the cost gate: the cross-sectional rank reshuffles 125-150% daily. A low-turnover variant (weight smoothing, a=0.10) is net-positive in all eight measured years pooled, and specifically in 2024-2025. This is a candidate, not yet confirmed: four unverified conditions (short-window power test, boundary plateau check, liquidity decomposition, holdout verification) gate the finding.

The methodological output remains valuable regardless:

1. **Reconnaissance-first testing.** Independent code paths find errors that test suites miss.
2. **Dual independent specs + blind suites.** Different readings of a spec catch ambiguities.
3. **Explicit lookahead audit.** Deliberate side-by-side comparison of leaky vs. correct timings.
4. **Recent years as ground truth.** Pooled statistics on this dataset overstate tradability.

All code, test suites, and reconstruction scripts are committed to the repository and reproducible on demand. The four pending conditions listed above are the next open questions.
