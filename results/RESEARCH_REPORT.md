# Nifty Intraday Quantitative Research -- Final Report

## The Question and the Answer

Can we beat the Nifty index on a yearly basis using 1-minute intraday data on Nifty-100 equities? We tested five hypotheses ranked in advance, each a cross-sectional signal to enter at a specific morning time and exit at 15:20 close, measured against the two-leg cost hurdle for a long/short spread.

This program found three things:

1. **Four hypotheses are killed** (H1, H3, H4, H5) by cost hurdles, magnitude, or wrong sign.
2. **One hypothesis is real** (H2, overnight cross-sectional reversal): -24.30 bps, t = -16.66, 8 of 8 years sign-stable. Its initial kill has collapsed under measurement. Two original kill reasons failed when tested rigorously: a hardcoded capacity clip was an artifact (clears at higher notional), and a concentration threshold was hand-chosen (sits at 27th percentile of a measured null, does not fire at its designed threshold).
3. **A candidate exists** (long-only index-relative tilt using H2's signal, weight smoothing a=0.10): net-positive in all eight years pooled and in 2024-2025 individually, ~5.8% annualised net excess. Of the four gating conditions, **2 are cleared** (boundary plateau; liquidity decomposition -- the edge is genuinely concentrated in the bottom-2 ADV deciles, 32.95% of gross excess vs a permutation null of 20.1%, z=5.75), **1 is partial** (recent-window significance: Newey-West t=3.20 full-universe, but the continuous-coverage subset sits at p~0.046 and flips on estimator choice), and **1 is NOT RUN** (the out-of-sample holdout -- deliberately last, spending it requires a reviewed code change).

What is NOT established: that the candidate survives at size (its edge lives exactly where capacity binds), or that it confirms on unseen data.

---

## Program status update -- 2026-08-31

Everything below reflects the definitive Phase E re-run and the completion of Phases F-H.

- **Phase E (feature sweep), definitive:** 19 of 22 registry features genuinely measured on real OHLCV (the first run had fed 7 features degenerate proxies). Trial matrix 514,070 x 114; measured effective_n_trials 18.76 of 132 planned; PBO 0.0002; **every deflated Sharpe 0.0000**. Three market-level broadcast features are structurally untestable under cross-sectional ranking and are reported as exclusions, not nulls. `var_trial_sharpes` measured at 0.0389957 and now wired into the lens's criterion 6, replacing the 1.0 placeholder.
- **Phase F (volume_breakout v2): KILLED at its own pre-registered component gate.** All four components have measured verdicts and none passes: hurst (no monotone response), beta-residual (worst in sweep), breakout_strength (negative at every horizon), volume_zscore (+0.108 raw at h=1, deflated 0.0000 -- and the exact short-horizon volume shape whose economics v1 already measured as dead at one minute of latency). Criteria 1-5 never ran; nothing survived to build.
- **Phase G (portfolio construction): built and closed.** Seven-scheme weighting registry, VolTargetSizer wired through the engine with honest clip/vol-target reporting, and the pre-registered G3 comparison run on the tilt signal: **no scheme promotes** at the 0.95 recent-window DSR bar (closest 0.9485); measured effective_n_trials 1.65 of 7 (inverse_vol and risk_parity are structurally identical under diagonal-first ERC). The signal-carrying schemes die on their own turnover costs. The tilt's own low-turnover construction remains the only candidate. Seven trials registered; the registry's config-hash fix verified in production (7 distinct rows, where the old defect would have silently kept 1).
- **Phase H (execution & capacity): built.** The cost/participation ladder, the derived brokerage crossover (Rs 66,666.67 = flat/pct, bps constant below and decaying above), and the corrected two-constraint capacity result: a position limit and a daily-trade limit bound different risks, the trade constraint is the position constraint divided by turnover fraction, so at the tilt's 0.127 daily turnover the trade constraint is 7.9x LOOSER and **the position limit is the operative bound** -- the ~Rs 400 crore ADV-based figure stands, not as an upper bound to revise down, but per-liquidity-decile evaluation of it is still owed where the edge concentrates.
- **Trial accounting to date:** trials.db carries 31 rows across sweep, tilt pre-registration, and the G3 comparison; every multiple-testing statistic in this report is computed against measured effective trial counts, never planned ones.
- **The holdout remains unspent** (read-count 7, unchanged). It is the final gate and requires an explicit, reviewed decision.

---

## The Five Hypotheses

| Hypothesis | Signal | Measured Edge | t-stat | Verdict | Primary Kill Reason |
|---|---|---|---|---|---|
| **H1** | Index morning -> afternoon momentum | 0.09 bps | 1.95 | **KILLED** | Magnitude 180x below cost gate |
| **H2** | Overnight cross-sectional reversal | -24.30 bps | -16.66 | **INCOMPLETE** | Recent-year decay at Rs 1L; criteria 5-6 NOT_EVALUATED |
| **H3** | Intraday cross-sectional reversal | +1.21 bps | 1.04 | **KILLED** | Sign is momentum, not reversal; magnitude trivial |
| **H4** | Volatility compression -> expansion | -6.26 bps | -2.06 | **KILLED** | Sign reversed (reversal, not continuation); regime alternation |
| **H5** | F&O open-interest conditioner | -7.27 bps | -4.86 | **KILLED** | Edge fails at retail clip size; strong recent-year decay |

### H1 -- Market Intraday Momentum (NIFTY50 level)

Signal: index 09:16 close versus index 10:00 close. Edge of 0.0879 bps against a two-leg hurdle of 16.53 bps -- 188x too small. Sign stable in only 5 of 8 years (2018-2025). The tested direction (morning index momentum continuing to afternoon) does not exist on the Nifty-50 spot price. Verified against 54 tests from two independent authors.

### H2 -- Overnight Cross-Sectional Reversal (all_equity, 149 names)

This is the program's single largest measured effect. Signal: the cross-sectional rank of each name's overnight (close to open) log return; at 10:00 entry the top decile of overnight losers reversed at -24.30 bps net of transaction costs, with t = -16.66 over 1,867 sessions, sign stable 8 of 8 years. The effect is real and large. It passes cost and concentration thresholds but is capacity-limited by recent-year decay.

**Concentration criterion does not fire:** The edge concentrates in the bottom liquidity decile. A null-distribution calibration on 150 within-session permutation replicates (seed 42, causal cross-sectional-rank bucketing on strictly-prior rupee turnover, production upper-median) derived a p95 threshold of 4.8695. H2's observed concentration ratio is 2.7596, which sits at the 62.3rd percentile of this null distribution. The joint false-positive rate at p95=4.8695 is 0.0067 (1/150). Criterion 4 does not fire.

**Recent-year decay:** The 2024 edge is -10.98 bps and 2025 is -9.62 bps, both below the 16.53 bps two-leg cost hurdle at Rs 1L. At Rs 10L the hurdle is 8.03 bps, and both years still fail. This edge has been arbitraged or decayed below the point of tradability in current data.

**What the recent performance actually measures:** 18 names IPO'd inside the 2018-2025 window (IRCTC, SBICARD, NYKAA, ETERNAL, JIOFIN, LICI, HYUNDAI...), and they add reversal signal in recent years. 2024 is -10.98 bps on the full universe but only -3.04 on continuous-coverage names. So H2's recent edge is substantially carried by recently-listed stocks -- the ones with the least history and the least borrow availability for a short leg. On a like-for-like universe H2's decay is steeper (2018 -37.64 -> 2024 -3.04), not shallower.

**Verdict:** H2 passes criteria 1-4 (cost, concentration, precision). Criteria 5 and 6 (recent-year power, out-of-sample holdout) remain NOT_EVALUATED. The initial kill rested on criterion 7 at a Rs 1L clip; this clears at Rs 10L but both years still fail at that size too. **H2's verdict is INCOMPLETE, not KILLED, because measured conditions that appeared disqualifying proved to be artifacts of thresholds or clip sizes rather than fundamental failures.**

#### Corrections and Refinements to H2

**Concentration ratio measurement (2026-08-19):** The concentration ratios reported here (2.7596 observed, 4.8695 p95) differ from the initially published 2.1189 and 5.5484 because the first calibration ran on the full 1-minute panel with an expanding-mean liquidity proxy, whereas production code reduces to a two-rows-per-session checkpoint panel before Lens sees anything and uses a trailing-20-session window in compute_prior_adv. These are different statistics measuring different geometries. The recalibrated numbers reflect the true production path. The old 5.5484 was too permissive for checkpoint-panel hypotheses: a future ratio between 4.87 and 5.55 would have wrongly passed. Despite the changed numbers, criterion 4 still does not fire for H2.

**Survivorship argument refutation:** The survivorship argument appeared in four verdicts, was never measured, and was refuted when it finally was. `scripts/recon_survivorship.py` rebuilt the universe two ways -- all 149 names vs the 129 with continuous 2018-2025 coverage -- and the measured deltas run backward to the claim: tiny and wrong-signed in the early years (2018 +1.54, 2019 +0.57 bps) and largest in the recent ones (2023 +7.52, 2024 +7.94). Survivorship is unmeasurable from this dataset: zero of the 149 names stop before the window end (the wealth-destroyers stayed listed), and delisted names were never downloaded. What the deltas actually measure is new listings, not survivor bias.

### H3 -- Intraday Cross-Sectional Reversal (all_equity, 149 names)

Signal: cross-sectional rank of the morning (09:16 to 10:00) log return. Measured edge +1.21 bps, t = 1.04, across 1,866 sessions -- 14x below the cost gate. More critically, the observed direction is **momentum, not reversal**: the top quintile of morning winners outperformed on the afternoon's trade, the opposite of the hypothesized reversal. Sign flipped in 2020 and stayed flipped; two opposing regimes cancel in the pooled number, and neither clears the gate alone. The verdict holds across 52 tests verified against two independent authors.

The two-way cross-sectional reversal (overnight and intraday) teaches a liquidity lesson: overnight is where order imbalance accumulates with no continuous trading to absorb it; intraday, continuous liquidity clears imbalance as it arrives, leaving nothing to revert. Further reversal variants on price-derived intraday signals are low-value.

### H4 -- Volatility Compression Expanding into Continuation (all_equity, 147 symbols)

Signal at 10:00: the morning range (09:16-10:00 high-low spread) divided by a strictly-prior 20-session rolling mean range, multiplied by the sign of the morning close-to-open log return. Trade: 10:00 close to 15:20 close, cross-sectionally demeaned. Three thresholds pre-registered; strongest result at expansion > 2.0 was -6.26 bps, t = -2.06, against the 16.53 bps gate -- 2.6x short and 40% below significance.

The sign is **consistently wrong**: all three thresholds produce a negative edge (weak reversal after expansion), opposite to the hypothesized continuation. The per-year pattern was previously reported as "concentrated in 2018" but under scrutiny with 34 alternative bucketing conventions swept, that reading was unreproducible. The true pattern is a **regime alternation**: negative 2018-2020, positive 2021-2023, negative again 2024-2025. An effect that changes sign for three consecutive years and back is not tradable under any holding rule. Why no module: the signal is decisive on magnitude, significance, and direction simultaneously, so no methodological refinement changes the kill.

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

At retail size (Rs 1L) the edge does not cover a single round trip. The Rs 1Cr row appears marginal until context: at that size across 149 names (~Rs 149Cr gross), realised market impact would dwarf a 0.09 bps margin, and the cost model prices zero impact. Decay is decisive at all sizes: 2024 = -5.45 bps and 2025 = -2.41 bps, both below every gate.

---

## The Candidate: Low-Turnover Index-Relative Tilt

H2 is real but capacity-limited on a daily-rebalance two-leg construction. Two follow-on reconstructions tested a long-only index-relative tilt using H2's overnight reversal signal as a ranking mechanism, side-stepping the short leg to halve the cost gate. The daily-rebalance form is killed by structural turnover, but a low-turnover variant (weight smoothing, a=0.10) is net-positive in all eight years pooled and specifically in 2024-2025. This is the program's only candidate for tradable edge.

**How to reproduce:** `nq tilt` is a parameterised backtest command that reproduces the tilt results. Refer to the repository README and specs for usage.

### Daily-Rebalance Form: Real Gross Excess, Killed by Turnover

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

### Why This Is Not Yet a Result -- Scorecard: 2 Cleared (1 with a vacuous half), 1 Partial, 1 NOT RUN

State these prominently because they gate the finding. Do not read "NOT RUN" as anything softer than it says: that check has not been executed.

1. **Boundary plateau check -- PASSED.** Turnover glides smoothly from ~0.30 to ~0.011 across a in {0.20...0.01} with no discontinuity, and a=0.10 is not an isolated peak. Every a=1.0 control is decisively NEGATIVE in the same window (-6.44 to -10.82 bps/day, all p<0.002), confirming smoothing does something real relative to daily rebalancing.

2. **Recent-window significance test -- PARTIAL.** On 2024-01-01..2025-07-31 alone (n=389 sessions, NET series), only mild/full-universe a=0.10 clears: +1.51 bps/day, se=0.55, t=2.75, p=0.0063. This survives Bonferroni at 4 comparisons (0.0125 threshold) but is 1 of 4 related constructions and depends on recently-listed names: continuous-coverage (like-for-like universe excluding the 18 names which IPO'd inside the window) does NOT clear at p=0.077. The construction is not a confirmed recent-window edge but is not ruled out either; its significance is universe-dependent and recently-listed-names-dependent.

   **Newey-West update (measured 2026-08-21):** the tilt's daily excess returns are NOT autocorrelated (lag-1 -0.010, lag-2 -0.115), so the iid standard errors above were CONSERVATIVE. Newey-West (Bartlett kernel, L=5, n=389) raises full-universe t from 2.75 to 3.20, and continuous-coverage t from 1.77 to 2.0009. **Caveat, stated prominently:** t=2.0009 against a 1.96 threshold is razor-thin (p~0.046). A verdict that flips on the choice of standard-error estimator is FRAGILE, not established, and the 1-of-4-related-constructions multiple-testing concern above is unchanged by this recomputation.

3. **Liquidity decomposition -- MEASURED; liquidity leg PASSES, time-of-day leg NOT EVALUABLE.** Concentration ratio 2.7596 against the measured 4.8695 threshold (~62nd percentile of the permutation null); all ten deciles populated, sign-consistent and individually significant. The book's realised excess is positive in every decile, but the bottom two deciles supply 32.95% of it against a MEASURED p95 of 23.82% (10,000 within-session label permutations, seed 42) -- above every replicate drawn, 5.75 null standard deviations out. The lean toward less-liquid names is real and now adjudicated: the illiquid-tail leg FIRES even though the signal-spread leg passes. The time-of-day half of the criterion is structurally vacuous on the checkpoint geometry and is NOT claimed as cleared. Full tables and method: "Liquidity Decomposition of the Tilt -- Criterion 4, Measured" below; reproduce with `scripts/recon_tilt_liquidity_decomposition.py`.

4. **Out-of-sample holdout -- NOT RUN.** The holdout window (last 12 months) has not been read. Note what "locked" means here: `HoldoutLock.record_read()` (`src/nifty_quant/research/splits.py`) appends a log entry and increments a counter -- it never inspects the count and never raises. The only consumer of `read_count()` in the repository is `cli.py:1432`, which prints it. Nothing in code prevents a second read; what prevents it is the `--allow-holdout` flag plus human discipline, not a technical lock.

**The degenerate corner, recorded honestly:** Hysteresis b=1.00 on the aggressive tilt collapses turnover to ~0.0005-0.0086/day (a buy-once-never-trade book whose few-name concentration makes the no-trade threshold enormous relative to any weight change). Its positive recent numbers reflect almost no rebalancing over 7.5 years and must not be read as validating the hysteresis mechanism.

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

H2 at -24.30 bps passes the Rs 1L gate (16.53 bps) and the concentration threshold but fails recent-year decay at that size. H5 at -7.27 bps fails the Rs 1L gate, passes Rs 10L / Rs 1Cr on the pooled statistic but fails in 2024-2025. The cost model choice is not academic: a strategy that "clears" costs under one size assumption may not under another.

---

## What Was Ruled Out and What Remains Open

### Ruled Out (by explicit measurement)

**H5's multi-day holding period (k = 1, 2, 3, 5, 10 usable sessions):** The per-day rate collapses with holding length. The cumulative k-day gross edge does not grow faster than one-day, so amortising the cost over multiple days does not rescue the hurdle. This axis is closed.

**The survivorship argument:** Measured via `scripts/recon_survivorship.py`. The all_equity universe has zero delisted names within the panel window (survivorship is unmeasurable here). New listings actually strengthen recent-year H2 performance. Early-years inflation comes from later-listed names, not survivorship. Strike this argument from all verdicts.

**H4's "concentrated in 2018" claim:** Recon numbers were unreproducible (34 bucketing conventions swept; the published 2024 +1.3 is unreachable, true range -5.3 to -17.4 bps). The real pattern is a regime alternation (negative 2018-2020, positive 2021-2023, negative again 2024-2025), not a skill decay or concentration story.

**Long-only index-relative tilt, daily-rebalance form:** Tested on H2's overnight reversal signal. Construction halves the two-leg hurdle to ~8.26 bps at Rs 1L and starts from the index return rather than zero. Real gross index-relative excess of 21-31% annualised, but entirely consumed by turnover: the cross-sectional overnight-return rank reshuffles the portfolio 125-150% every session. Net is flat to negative in recent years across all combinations. The binding constraint is structural turnover from daily rebalancing, not the cost hurdle. This axis is closed.

### Still Open (Unverified Conditions on the Low-Turnover Candidate)

The low-turnover variant (weight smoothing, a=0.10) passes the pooled test and survives 2024-2025 in every combination. Of the four gating conditions: boundary plateau is cleared; recent-window significance is partial; the liquidity decomposition is now measured and its signal-spread leg PASSES, but its book-attribution leg FIRES against a measured null (bottom two ADV deciles supply 32.95% of realised excess vs p95 23.82%), and its time-of-day leg is not evaluable; the holdout remains NOT RUN. See the scorecard in the Candidate section above.

---

## Phase E -- Conditional-Analysis Sweep

Run 2026-08-21 (`results/PHASE_E_SWEEP.md`). Panel 2018-01-01..2025-07-31, all_equity (149 symbols), 701,863 bars, 1,880 sessions, `holdout_intent="never"` (window ends two weeks before the holdout boundary).

**No new edge found.** `effective_n_trials` = 19.2969 of 132 planned (the registry is correlated by construction, so 132 trials were ~19 independent looks). PBO (CSCV) = 0.0267. Every deflated Sharpe is 0.0000. Best raw pre-cost Sharpe in the sweep is `vol_ratio` at horizon h=1, +0.0349, decaying monotonically with horizon -- the signature of noise, not signal.

**Only 10 of 22 features were genuinely measured; carry this caveat with any conclusion drawn from the sweep:**

- **7 features were fed degenerate proxies and never actually tested** (`volume_zscore`, `signed_volume_proxy`, `breakout_strength`, `parkinson_volatility`, `garman_klass_volatility`, `rogers_satchell_volatility`, `close_location_value`). `run_sweep`'s pinned signature takes only `close` + `day_offsets`, so the registry synthesised `volume = ones_like(close)` and `high = low = open = close`, degenerating each of these to all-NaN or zero-variance. These are not null results -- they were not tested.
- **4 features are market-level scalars or per-symbol-static values, structurally unsuited to cross-sectional ranking** (`breadth`, `cross_sectional_dispersion`, `median_pairwise_correlation`, `variance_ratio`). A value identical across symbols at each bar produces no dispersion and no spread -- a legitimate finding about the harness, not the signal.
- **1 feature raised on all 6 horizons**: `rv_to_vix_ratio` (`ValueError: realized_vol_ann must be a 1-D array`), recorded as a failed trial.

**Consequence for Phase F:** `specs/volume_breakout_v2.md` designed v2 around 4 components. Two now have measured, discouraging verdicts (`hurst_on_stitched` +0.0118, no monotone response; `beta_residual_return` -0.1066, worst in the sweep). The other two (`breakout_strength`, `volume_zscore`) fall in the degenerate-proxy category above and remain unmeasured; Phase F cannot proceed on them until the OHLCV plumbing is fixed.

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

H2's pooled edge clears costs; its recent years do not. A pooled average therefore overstates what is tradable today, and every verdict reports per-year numbers so this is visible. A hypothesis whose most recent complete years fail the cost gate is not a survivor regardless of its pooled t-stat.

**Note the justification carefully.** The reason to weight recent years is that they are the most recent evidence of what the effect does NOW -- not survivorship. The survivorship version of this argument appeared in four verdicts, was never measured, and was refuted when it finally was. This report keeps the conclusion and discards the reasoning that turned out to be wrong.

### Measurement Errors I Corrected (and the Tests Would Have Caught)

**Median convention:** The concentration-ratio computation used `np.median(spreads)` (which averages the middle two values on even-length arrays) instead of production's upper median. On a thin margin (2.7596 vs. 2.0 cutoff), this choice matters.

**Bucket geometry mismatch:** `expectancy_by_liquidity_decile` couples the decile loop and the feature-bucketing to one `n_buckets` parameter, so it cannot express production's 10 liquidity deciles x 5 feature quintiles in a single call. The implementer must either add a `feature_n_buckets` parameter or expose decile assignment directly for use by a caller.

**Min-names floor in cross-sectional rank:** The `cross_sectional_rank` function silently returns NaN when a row has fewer than 5 finite values. Below ~44 symbols (enough to fill 5 per decile at 8+ per bucket), every decile returns 0.0 spread against both old and new code. A 2-3 symbol test fixture is useless for regression testing; any future test asserting on decile spreads must use >= 50 symbols, confirmed explicitly in the test.

**Causality in decay measurement:** The concentration ratio uses `np.quantile` on the full panel (including future data) to define decile boundaries. When re-measured with causal bucketing (cross-sectional rank on strictly-prior rupee-volume ADV), the ratio dropped from 2.4538 (full-sample, share volume) to 1.9959 (causal, share volume) to 2.7596 (causal, rupee volume, production geometry). The causal 1.9959 looked like a rescue until the bucketing geometry correction (10x5, not 10x10). Causality matters, and so does the unit -- share count vs. rupee turnover partition the bottom decile almost entirely differently.

### Quality Gate: First Green Run in Project History

The project's CI gate (`make gate`) now exits 0 with: OK=48, DEBT=2, REGRESSED=0, UNGATED=0. Test suite: 2574 passed, mypy 172/172, ruff clean. The gate had been red since its creation (mypy ceiling was seeded at 162 when the true count was already 172). This marks the first green run in the project's history.

---

## Conclusion

We set out to beat the Nifty-100 index on a yearly basis using 1-minute intraday price and volume data. Five hypotheses were ranked in advance and tested under independent dual-suite verification and reconnaissance-based oracle audit.

**Four are dead** (H1, H3, H4, H5): killed by cost hurdles, magnitude, or wrong sign.

**One is real but incomplete** (H2, overnight cross-sectional reversal): -24.30 bps, t = -16.66, 8 of 8 years sign-stable, 63+ tests from multiple independent authors. It passes costs and concentration thresholds but is capacity-limited by recent-year decay (2024 -10.98 bps, 2025 -9.62 bps, both below Rs 1L hurdle at 16.53 bps). Its verdict is INCOMPLETE, not KILLED: criteria 5 and 6 (recent-window power, out-of-sample confirmation) remain NOT_EVALUATED. Recent performance is substantially carried by newly-listed names (2024: -10.98 bps full universe vs -3.04 bps continuous-coverage).

**One is a candidate** (long-only index-relative tilt using H2's signal, weight smoothing a=0.10): net-positive in all eight measured years pooled, and specifically in 2024-2025, at ~5.8% annualised net excess (mild/full combination: 2.29 bps/day pooled, +1.92 bps/day in 2024, +0.84 bps/day in 2025). Of the four gating conditions: (1) the boundary plateau check PASSED -- a=0.10 is not an isolated grid-edge artifact; (2) the recent-window significance test is PARTIAL -- full-universe clears (t=2.75, p=0.0063; Newey-West 3.20), continuous-coverage does not at the iid SE (p=0.077) and is razor-thin under Newey-West (t=2.0009, p~0.046); (3) the liquidity decomposition is now MEASURED and SPLITS -- the signal-spread leg PASSES (concentration ratio 2.7596 vs the measured 4.8695 threshold), the time-of-day leg is structurally not evaluable, and the book-attribution leg FIRES: the bottom two liquidity deciles supply 32.95% of the realised gross excess against a measured p95 of 23.82% (10,000 within-session label permutations, seed 42; above all 10,000 replicates, 5.75 null sd); (4) the out-of-sample holdout is NOT RUN -- unread, and not technically prevented from being read (the lock records reads, it does not block them; see the scorecard above). The candidate is driveable via `nq tilt`.

The methodological output remains valuable regardless:

1. **Reconnaissance-first testing.** Independent code paths find errors that test suites miss.
2. **Dual independent specs + blind suites.** Different readings of a spec catch ambiguities.
3. **Explicit lookahead audit.** Deliberate side-by-side comparison of leaky vs. correct timings.
4. **Recent years as ground truth.** Pooled statistics on this dataset overstate tradability.

All code, test suites, and reconstruction scripts are committed to the repository and reproducible on demand. Of the tilt's four gating conditions, two are cleared (one of them with a vacuous time-of-day half), one is partial, and one -- the holdout -- remains NOT RUN. That is the next open question.

---

## Liquidity Decomposition of the Tilt -- Criterion 4, Measured

Scorecard item 3 ("Liquidity decomposition -- NOT RUN") is now run. Reproduce every number
below with:

```
NQ_CACHE_ROOT=<scratch> .venv/bin/python scripts/recon_tilt_liquidity_decomposition.py
```

Window 2018-01-01..2025-07-31 (research period only), `all_equity` 149 symbols, 1868
checkpoint sessions, seed 0, horizon 1, mild tilt, a=0.10. The `HoldoutLock` read count was
7 before the run and 7 after: the holdout was not touched.

**Path note, and it matters.** `research/tilt.py` imports `build_overnight_feature` from
H2 -- the tilt is a long-only index-relative weighting of H2's overnight-reversal signal.
The correct criterion-4 geometry is therefore H2's: the two-rows-per-session checkpoint
panel (09:16 entry / 15:20 exit) with `horizon=1`, not the raw 1-minute panel. This is the
geometry `lens.CONCENTRATION_RATIO_THRESHOLD = 4.8695` was calibrated on; the earlier
5.5484 came from the raw minute panel with an expanding-mean liquidity proxy and was too
permissive.

### A. Lens criterion 4 -- signal spread by liquidity decile (decile 0 = LEAST liquid)

Per decile, the cross-sectional 5-bucket expectancy spread (top quintile minus bottom
quintile) of the next session's return. This is the quantity `Lens.verdict()` gates on.

| decile | spread (bps) | t | rows | observations |
|---|---|---|---|---|
| 0 (least liquid) | -47.6493 | -11.4453 | 1867 | 26,690 |
| 1 | -33.3115 | -7.7567 | 1867 | 25,366 |
| 2 | -25.4835 | -6.0861 | 1867 | 26,088 |
| 3 | -22.8008 | -5.5177 | 1867 | 25,699 |
| 4 | -13.1114 | -3.1782 | 1867 | 25,465 |
| 5 | -10.3371 | -2.4258 | 1867 | 26,279 |
| 6 | -17.2667 | -3.9301 | 1867 | 25,831 |
| 7 | -11.6343 | -2.6036 | 1867 | 25,564 |
| 8 | -16.9234 | -3.7639 | 1867 | 25,636 |
| 9 (most liquid) | -15.9510 | -3.0323 | 1867 | 26,910 |

No decile is thin: all ten carry 1867 defined session-rows and 25k-27k symbol-observations.
Every decile has the same (negative, i.e. reversal) sign and every one is individually
significant at |t| > 2.4.

**Concentration statistic:** max |spread| = 47.6493 (decile 0), median |spread| = 17.2667,
**ratio = 2.7596** against the measured threshold **4.8695** -- the 95th percentile of a
300-replicate within-session permutation null (seed 42, measured on this same checkpoint
path; derivation at `research/lens.py:26-56`, where the bottom-is-argmax rate under the
null is 10.7%). 2.7596 sits at roughly the 62nd percentile of that null.

**Liquidity leg: PASS.** The bottom decile is the strongest single decile, but not by a
margin that a permutation null would find surprising.

*Control, and a caveat about what this does and does not establish.* 2.7596 is exactly the
figure `lens.py` records for H2 on this path -- which is the point: the tilt's signal IS
H2's signal, so this measurement confirms the script runs on the true production code path,
and equally means section A is a statement about the SIGNAL, not about the tilt's long-only
weighting. That is why section B was measured too.

### A2. Time-of-day leg -- NOT EVALUABLE on this geometry

| bucket | spread (bps) | t | rows |
|---|---|---|---|
| open (09:15-10:30) | -24.3049 | -16.6575 | 1868 |
| close (14:00-15:30) | 0.0000 | 0.0000 | 0 |

On the two-row checkpoint panel the EXIT row's `horizon=1` forward return runs off the
session end and is NaN by construction, so only the entry-time bucket can ever carry a
defined edge. The time-of-day half of criterion 4 is therefore **structurally vacuous**
here: it cannot fail, and its PASS must not be read as evidence the edge is spread across
the session. State it as NOT EVALUABLE, not as cleared.

### B. Where the realised rupees actually came from

Section A measures the signal's bucket spread. The tilt is long-only and index-relative, so
its P&L is not that spread. Section B splits the book's realised gross excess by the SAME
prior-ADV deciles, as an exact identity:

`contribution_d = 1e4 * sum_{j in decile d} (w_book - w_benchmark) * r`, with
`w_benchmark = valid / n_valid`. The ten rows sum to the session's gross excess in bps; the
script asserts the identity and measured a gap of exactly 0.0.

| decile | contribution (bps/session) | share of gross | mean book wt | mean bench wt | names/session |
|---|---|---|---|---|---|
| 0 (least liquid) | 0.5691 | 17.8% | 0.1006 | 0.1028 | 14.4 |
| 1 | 0.4848 | 15.2% | 0.0929 | 0.0978 | 13.7 |
| 2 | 0.3491 | 10.9% | 0.0976 | 0.1005 | 14.1 |
| 3 | 0.3431 | 10.7% | 0.0981 | 0.0990 | 13.8 |
| 4 | 0.2262 | 7.1% | 0.0974 | 0.0981 | 13.7 |
| 5 | 0.2450 | 7.7% | 0.1010 | 0.1013 | 14.2 |
| 6 | 0.3195 | 10.0% | 0.1000 | 0.0995 | 13.9 |
| 7 | 0.1630 | 5.1% | 0.1003 | 0.0985 | 13.8 |
| 8 | 0.2975 | 9.3% | 0.0992 | 0.0988 | 13.8 |
| 9 (most liquid) | 0.2011 | 6.3% | 0.1080 | 0.1037 | 14.5 |
| **total** | **3.1984** | 100% | | | |

(1856 sessions; total matches the simulation's mean gross excess of 3.1984 bps exactly.)

**Read this against the candidate, not for it.** Every decile contributes positively -- the
edge is not a single-decile artifact, and the book is not overweight the illiquid tail
(book weight 0.1006 vs benchmark 0.1028 in decile 0; the tilt is if anything slightly
UNDERweight there). But the contribution profile is not flat: the bottom two deciles supply
**32.95%** of the gross excess against a 20% even-spread expectation, the bottom three supply
43.86%, and the top three supply 20.7%. The realised edge does lean toward the less-liquid
half of the universe -- and section C measures a null for that lean rather than leaving the
20% figure to stand as an intuition.

### C. Measured null for the bottom-k share -- the lean is real

The 20% even-spread figure above is an intuition, not a measurement, and gross-excess
contribution is not obliged to be uniform across deciles even when liquidity is irrelevant:
it depends on how many names and how much weight sit in each decile and on return
dispersion, all of which vary systematically with liquidity. So the share was measured
against a null rather than read against a round number.

**Construction.** The same within-session permutation that calibrated 4.8695
(`lens.py:24-50`): one column permutation per session, applied identically to every row of
that session. The book's attribution has exactly one row per session, and it is the
**liquidity decile label** that is permuted across symbols -- every symbol keeps its own
realised active weight, its own return and therefore its own realised contribution, and only
its liquidity rank is scrambled. Because a permutation is a bijection on the columns, the
book's total gross excess, every decile's size and the unlabelled residual are invariant, so
the share is well defined on every replicate. n = 10,000 replicates, seed 42, measured on
exactly the arrays section B attributes -- not on a convenient stand-in, which is the
geometry error that produced 5.5484.

Share of gross excess supplied by the bottom k ADV deciles, in %:

| k | observed | even spread | p50 | p75 | p90 | p95 | p99 | null max | null sd | (obs - p50)/sd | replicates >= observed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17.7939 | 10 | 10.3399 | 11.4874 | 12.5299 | **13.1044** | 14.2547 | 16.6738 | 1.7084 | 4.36 | 0 / 10,000 |
| 2 | 32.9505 | 20 | 20.1018 | 21.6246 | 22.9740 | **23.8221** | 25.2949 | 28.2646 | 2.2354 | 5.75 | 0 / 10,000 |
| 3 | 43.8640 | 30 | 30.2160 | 31.8639 | 33.3575 | **34.2157** | 35.9981 | 39.2505 | 2.5127 | 5.43 | 0 / 10,000 |

p95 is the repo's existing convention for turning a measured null into a cutoff
(`CONCENTRATION_RATIO_THRESHOLD = 4.8695` is the p95 entry of its own grid). The three
cutoffs are recorded as `BOTTOM_K_SHARE_THRESHOLDS` in
`scripts/recon_tilt_liquidity_decomposition.py` with the derivation beside them, and the
script re-measures the null on every run and refuses to finish if they do not reproduce.

**The prior worry did not materialise.** The null centres on the even-spread value --
p50 of 10.34 / 20.10 / 30.22 against 10 / 20 / 30. What the flat-20% intuition was missing
was not its centre but its *width*: a standard deviation of 2.24 percentage points at k=2,
so anything up to ~23.8% would have been unremarkable. 32.95% is not.

**Verdict: the bottom-2 share FIRES.** The observed 32.95% sits above p99, above the
**maximum of all 10,000 replicates** (empirical p < 1e-4), at 5.75 null standard deviations.
k=1 and k=3 fire by the same margin (4.36 and 5.43 sd, both with zero replicates at or above
the observed value). A tradability-stratified variant of the permutation -- exchanging
columns only within {labelled, tradable} and {labelled, not tradable}, which additionally
holds each decile's tradable-name count fixed -- gives p95 of 13.0661 / 23.8054 / 34.1893,
indistinguishable from the unconditional null. The conclusion does not rest on how the
permutation treats non-tradable names.

**Mechanism (descriptive; no threshold reads it).** A decile's contribution can be large
because the book deviates more there or because returns disperse more there, and the two
have different remedies. Neither is operating here:

| decile | mean abs(active weight) | sd(return), bps | names/session |
|---|---|---|---|
| 0 (least liquid) | 0.002352 | 200.96 | 14.4 |
| 1 | 0.002255 | 207.47 | 13.7 |
| 2 | 0.002121 | 201.57 | 14.1 |
| 5 | 0.002194 | 206.08 | 14.2 |
| 9 (most liquid) | 0.002325 | 226.03 | 14.5 |

Active-weight magnitude is essentially flat across deciles (0.00212 to 0.00235, with the
*most* liquid decile nearly as high as the least), and return dispersion is **lowest** in
decile 0 and highest in decile 9 -- the opposite of what a volatility explanation requires.
The concentration is therefore in the signal's hit rate per unit of active weight, not in
bet size and not in noise. The tilt genuinely works better on less-liquid names.

**What this does and does not overturn.** It does not change the section A PASS: the two
statistics measure different things and section A is simply the less sensitive instrument
here. Section A asks whether the signal's bucket spread is concentrated (2.7596 against
4.8695, 62nd percentile -- and note the bottom decile *is* the argmax there, so it points
the same way without clearing its bar); section C asks where the realised rupees came from
and answers with five-plus standard deviations. Read together: the same lean is visible in
both, and only the P&L-side measurement has the power to resolve it. The operational
consequence is a capacity one, and `scripts/recon_tilt_liquidity.py`'s exclusion ladder --
does the edge survive dropping the illiquid tail -- is the next thing to weigh, now with a
measured reason to weigh it.

### Criterion 4 verdict

**PASS on the liquidity leg** (ratio 2.7596 vs measured threshold 4.8695, ~62nd percentile
of the null; all ten deciles populated, sign-consistent and individually significant).
**NOT EVALUABLE on the time-of-day leg** -- the checkpoint geometry admits only one bucket
with a defined forward return, so that half of the criterion is vacuous and is not claimed
as cleared.

**FIRES on the book's realised attribution (section C).** The bottom two ADV deciles supply
32.95% of the realised gross excess against a measured p95 of 23.82% -- above every one of
10,000 permutation replicates, 5.75 null standard deviations out. This is now adjudicated,
not merely reported.

Two caveats stand with the section A PASS: (1) section A measures the tilt's SIGNAL, which
is H2's signal, so it inherits H2's already-published concentration result rather than
testing the tilt construction independently; section B/C is the construction-specific
measurement, and (2) that measurement disagrees with section A in emphasis. The signal-side
spread test passes; the P&L-side share test fails by a wide margin. The disagreement is a
power difference, not a contradiction -- the bottom decile is the argmax in section A too --
and the honest summary is that the tilt's edge is concentrated in less-liquid names to a
degree that no permutation of the liquidity labels can reproduce.
