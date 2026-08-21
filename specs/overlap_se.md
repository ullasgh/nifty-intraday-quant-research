# Spec — overlap-aware standard errors (fixes L7 and L8)

Status: spec, written before implementation. Author: lead. Date 2026-08-20.
Findings this fixes: `specs/research_validity_findings.md` L7, L8.

## Why this is urgent

Phase E fans out a wide feature sweep whose entire output is significance tests. The machinery
those tests run through currently reports t-statistics **1.4x to 2.5x too high**, worsening with
horizon. Running a wide sweep through it would produce winners that are selection artefacts, and
the deflation machinery downstream cannot recover a t-statistic that was wrong on input.

This spec must land before Phase E, not after.

## The two defects, measured

**L7.** `_block_bootstrap_resampling_2d` (`expectancy.py:294-359`, loop at `:339-350`) draws
exactly ONE contiguous block of length `<= horizon` per replicate and appends it. There is no
tiling to reconstruct a series of the original length. `_compute_bucket_stats:456-461` then takes
`nanmean` of that single padded block. Measured: raw bootstrap SE is **34x-48x** ground truth.

**L8.** The raw SE then passes through `:470-471`:

    n_effective = n_obs / (std_bps / se_bps) ** 2

which is not the standard identity. It over-corrects, producing a final SE at **0.40x-0.71x** of
ground truth — i.e. the two defects do not cancel, they compound in direction.

Ground truth in both cases is the empirical SE across 300-400 independent resimulations, 15,000
rows, real `day_offsets`, measured by calling the unmodified repo functions.

## Required behaviour

### 1. The block bootstrap resamples a SERIES, not a block

Each replicate must draw blocks repeatedly and concatenate until the resampled series reaches
`n_rows`, truncating the final block. Blocks are drawn with replacement from all valid start
positions. This is the standard moving-block bootstrap.

**Session boundaries.** A block must not straddle a session boundary, because the dependence being
preserved is intraday and the overnight gap breaks it. Blocks are drawn within a session using
`day_offsets`. Sessions shorter than the block length contribute no start positions and are
skipped — and the count of skipped sessions must be reported, not silently dropped. Per rule 5,
never assume 375 bars: a 60-bar Muhurat session will be skipped entirely at horizons above 60, and
that must be visible.

### 2. Block length is DERIVED, never chosen

Rule 8 applies and this is exactly the place it gets violated by convention. `L = horizon` is the
obvious choice and it is not automatically correct: overlapping h-bar forward returns induce
dependence spanning at least h bars, but the underlying returns carry their own serial correlation
on top of that.

**Requirement:** block length is derived from the measured autocorrelation of the series being
resampled, on real data, via the existing `features/calibrate.py` machinery where it fits. The
derivation and the measured ACF decay must be recorded next to the value, in the same style as the
`CONCENTRATION_RATIO_THRESHOLD = 4.8695` derivation at `lens.py:24-50` — which is the one
rule-8-compliant threshold in the repo and is the model to copy.

A defensible floor is `L >= horizon`; the derivation decides how far above it to go. Recording
"we chose L = horizon because the labels overlap by h" is a reasoned-not-measured threshold and is
precisely what rule 8 was written after.

### 3. `n_effective` stops being a corrector

The bootstrap SE **is** the standard error. Nothing further is applied to it.

`n_effective` becomes a derived REPORTING quantity only:

    n_effective = (std_bps / se_bps) ** 2

and it is never used to modify `se_bps`, `spread_t`, or any downstream statistic. Delete the
`:470-471` transform.

**Required invariant:** `spread_t == spread_bps / se_bps` exactly, with `se_bps` the bootstrap SE.
Any additional factor between them is the L8 defect returning under another name.

### 4. Everything that consumed the old numbers is invalidated

`lens.py` criterion 3 reads `exp_table.spread_t` unmodified. Once this lands, every recorded
verdict that leaned on criterion 3 was computed on a wrong SE. The implementation must not quietly
re-run and overwrite them. Recomputation is a separate, deliberate step, and the old and new
values must be reported side by side so the size of the correction is visible.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

**Rule 9 governs the statistical tests here and it is not optional.** A "does the CI contain the
true value" assertion fails with probability ~alpha per comparison regardless of correctness;
asserting it across 5 buckets flakes ~23% of the time on correct code. Do not write coverage
checks. And no failing statistical test is ever fixed by changing the seed.

1. A replicate's resampled series has length `n_rows`, not `<= horizon`. Assert on the length
   directly — this is the L7 regression test and it needs no statistics at all.
2. No block straddles a session boundary: with `day_offsets` marking sessions, every block's
   start and end fall in the same session. Use IRREGULAR sessions (60, 105, 375 bars) in the
   fixture.
3. Sessions shorter than the block length are skipped AND the skip count is reported.
4. On iid data with no autocorrelation, the bootstrap SE matches the analytic iid SE within
   tolerance. This is the calibration anchor: if it fails, the estimator is wrong independent of
   any overlap question.
5. **False-positive RATE, not coverage.** Over >= 30 independent seeds on a TRUE-NULL series
   (no real spread), the fraction of seeds where `abs(spread_t) > 1.96` must be at or below
   roughly alpha. Assert on the rate across seeds. A single-seed assertion here is a coin flip.
6. Power: on a series with a large known spread, `abs(spread_t)` exceeds 1.96 in effectively
   every seed. Large effect-to-SE ratio makes this deterministic, which is why it belongs here
   and a coverage check does not.
7. `spread_t == spread_bps / se_bps` exactly. This is the L8 regression test.
8. `n_effective == (std_bps / se_bps) ** 2` and changing it does not change `spread_t`.
9. On AR(1) data with rho = 0.6 and h = 5, the SE is within a stated tolerance of the empirical
   SE from independent resimulation. Report the ratio; the pre-fix value was 47.9x raw and 0.71x
   as fed forward, so both failure directions must be excluded.
10. The block length in use is derived, not literal: assert that a recorded derivation exists
    beside the value and that the value is `>= horizon`.

---

# AMENDMENT 1 — 2026-08-20. The invariants must be OBSERVABLE.

A test author reported, correctly, that `BucketStat` exposes only `t_stat`, `mean_bps` and
`std_bps` — **not `se_bps`**. Obligation 7 asks for `spread_t == spread_bps / se_bps` and obligation
8 for `n_effective == (std_bps / se_bps) ** 2`, and neither can be observed through the current API.

Faced with that, the suite softened the assertions to things it COULD observe:

    assert stat.t_stat != 0.0 or stat.mean_bps == 0.0     # obligation 7
    assert block_len >= 1                                  # obligation 10
    assert abs(stat.t_stat) < 100                          # obligation 9

All three are vacuous. The first is trivially true. The second passes on the pre-fix one-block
code it was written to catch. The third passes on an SE that is 47x wrong. They reported GREEN
against code with a measured 34x-48x error, which is exactly what a vacuous test looks like — and
this repo has already shipped two tests that survived every mutant including "return all zeros".

**This is a defect in the spec, not in the suite.** I specified an invariant over a quantity the
API does not expose. An invariant that cannot be observed is not an invariant, and asking for it
without providing the observation forces the author to choose between an impossible test and a
weak one.

## Required API additions

1. **`BucketStat` gains `se_bps: float` and `n_effective: float`.** Both are already computed
   internally and thrown away. Exposing them is what makes obligations 7 and 8 real.

2. **The bootstrap exposes the block length actually used** and **the count of sessions skipped
   for being shorter than that block length.** Obligations 3 and 10 both assert on these, and
   neither is reachable today. Return them alongside the resampled array, or on a small frozen
   result dataclass — implementer's choice, but they must be readable from a test.

## Restated obligations

- **7** — `stat.spread_t == stat.spread_bps / stat.se_bps` to within floating-point equality,
  read directly off the returned object. No proxy, no "consistency check".
- **8** — `stat.n_effective == (stat.std_bps / stat.se_bps) ** 2`, and mutating `n_effective`
  leaves `spread_t` unchanged.
- **9** — assert the RATIO of `stat.se_bps` to the empirical SE from independent resimulation
  lies in a stated band. Name the band. `abs(t_stat) < 100` does not test this.
- **10** — assert `block_length >= horizon` on the value actually used, AND that a recorded
  derivation exists beside it. `block_len >= 1` does not test this.

## Standing rule for both suites

If an obligation cannot be tested because the quantity is not observable, **report it and leave
the test failing** — do not substitute a weaker assertion that passes. A RED test that names a
missing observation is useful. A GREEN test that asserts a tautology is worse than no test, because
it will be counted as coverage of a thing nobody checked.

## AMENDMENT 1, addendum — a known insensitivity in obligation 5

Obligation 5 (false-positive rate on a true-null series) is ONE-SIDED as specified: it asserts the
FP rate is at or below alpha. On a true null, `t = mean / se` with `mean ~ 0`, so an SE that is
47x TOO LARGE drives t toward zero and the FP rate toward zero — and the test passes.

So obligation 5 cannot catch an over-conservative SE. It catches only the understating direction.
That is not a defect in the test; a null test asserting "we do not over-reject" is the correct
assertion. But it must not be read as evidence the SE is right.

The inflating direction is covered by obligation 4 (iid SE matches the analytic SE) and obligation
9 (the SE ratio falls inside a stated two-sided band). Both are RED against current code, so the
defect IS caught — just not by obligation 5. Recorded here so nobody later cites a green
obligation 5 as proof the estimator is calibrated.

Deliberately NOT adding a lower bound on the FP rate: per rule 9, a two-sided rate assertion across
30-50 seeds adds flake probability for coverage that obligations 4 and 9 already provide
deterministically.

---

# AMENDMENT 2 — 2026-08-20. Retracting the addendum: obligation 5 DOES catch L8.

The two independent suites disagreed on obligation 5 — one GREEN, one RED — and chasing it showed
the AMENDMENT 1 addendum above is **wrong**. It is retracted.

## What the addendum claimed, and why it was wrong

It argued obligation 5 is one-sided and cannot catch an over-conservative SE, because on a true
null `mean ~ 0` drives `t` toward zero regardless. That reasoning accepted the GREEN suite's framing
and stopped there.

It is wrong in the direction that matters. L8's net effect is an **understated** SE, which inflates
`t`, which RAISES the false-positive rate. A one-sided "we do not over-reject" assertion is exactly
the right instrument for that. Measured by the RED suite across 30 seeds: **false-positive rate
36.7% against a 20% ceiling.** Obligation 5 is one of the sharpest tests in the set, not a blind one.

## The real cause of the disagreement: the two suites measured different quantities

- The GREEN suite asserted on `BucketStat.t_stat`. That path carries only the L7 raw-SE inflation,
  which makes `t` too CONSERVATIVE — so its FP rate collapses toward zero and a one-sided test
  passes trivially.
- The RED suite asserted on `ExpectancyTable.spread_t`. That is the quantity this spec names
  verbatim, and the only one that passes through the L8 `n_effective` over-correction.

**Pinned:** obligations 5 and 6 assert on `ExpectancyTable.spread_t`, obtained by building a
2-bucket table through `conditional_expectancy` with `method="cross_sectional_rank"`. Not
`BucketStat.t_stat`.

## A trap that must be written down

`cross_sectional_rank` enforces `min_names = 5`. With fewer symbols it silently produces ALL-NaN
bucket labels and `spread_t == 0.0` for every seed — a false negative that looks like a real
measurement. Fixtures for obligations 5 and 6 must use **`n_symbols >= 5`**, matching the existing
repo pattern in `test_spread_calculation_with_known_buckets`.

This is the same species as the vacuous assertions amended out earlier: a result that is produced
without the computation ever happening. Worth more than the test it protects.

## Two corrections to the spec body

1. **Section 2's pointer to `features/calibrate.py` for "the existing ACF machinery" is
   inaccurate.** That module has no autocorrelation function at all — only null-distribution
   percentile calibration. The block-length derivation must ADD an ACF/dependence-length
   measurement; it cannot reuse one that does not exist. (Consistent with what is already recorded
   elsewhere: `features/calibrate.py` is 100% dead outside its own tests.)

2. **Obligation 3's return contract is pinned, not "implementer's choice".** The bootstrap returns
   a 3-tuple `(resampled, block_indices, n_sessions_skipped)`. Leaving it open meant one suite had
   to guess, and a `ValueError` on unpack is indistinguishable from the feature being absent —
   which defeats the purpose of a RED test naming a missing observation.

## Note on process

I got this one wrong by reasoning from a single suite's explanation instead of checking the
measured number. The 36.7% was available and would have settled it immediately. When two suites
disagree, the disagreement is the finding — reconcile it against data, not against whichever
explanation sounds more plausible.

---

# AMENDMENT 3 — 2026-08-20. The null must use a PERSISTENT feature.

The two suites reported obligation-5 false-positive rates of **36.7%** and **0.0%** on what both
believed was the same test. Neither is wrong. They built different nulls, and the difference
turns out to be the most important thing learned about this obligation.

## Measured, by the lead, on the current defective code

n_rows=3000, n_symbols=5, horizon=5, 2 buckets, `cross_sectional_rank`, 20 seeds. The ONLY thing
varied is the serial persistence `rho` of the FEATURE; the price series and everything else are
identical:

    feature rho=0.0      spread_t max=1.124   FP(>1.96) =  0.0%
    feature rho=0.9      spread_t max=1.780   FP(>1.96) =  0.0%
    feature rho=0.99     spread_t max=2.054   FP(>1.96) = 10.0%
    feature rho=0.999    spread_t max=2.473   FP(>1.96) =  5.0%

A white-noise feature (`rho = 0`) — which is what the 0.0% suite used, drawing
`rng.normal(0, 1, (n_rows, n_symbols))` fresh at every bar — produces a false-positive rate of
ZERO and **cannot detect the defect at any sample size**. I confirmed this at n_rows of 300, 3000
and 15000: FP stays 0.0% in all three.

## Why, and why it generalises beyond this test

The L8 under-statement bites through OVERLAP. Overlapping forward labels only induce dependence in
the bucket statistic if the same names stay in the same buckets across adjacent bars. A feature
redrawn independently every bar reshuffles bucket membership completely at each step, so the
overlap in the labels is never compounded by overlap in the groupings, and the naive-ish SE is
close to right.

**Real features are persistent.** Volume z-scores, breakout strength, Hurst, beta-residual return
— every primitive Phase E will sweep is slow-moving relative to a one-minute bar. So a white-noise
feature is not a conservative null, it is an unrepresentative one: it tests the estimator in the
single regime where the defect does not appear.

This is the same class of error as a flat fixture collapsing what a test measures. The null has to
resemble the data-generating process of the thing it is a null FOR.

## Required

**Both suites use a persistent feature for obligations 5 and 6.** `rho` is a THRESHOLD and rule 8
applies: derive it from the MEASURED lag-1 autocorrelation of a real feature on real panel data —
`volume_zscore` and `breakout_strength` are the natural choices — and record the measurement beside
the value. Do not hand-pick `0.99` because it appears in the table above; that table is a
demonstration that persistence matters, not a calibration.

**Use >= 30 seeds.** The 10.0% / 5.0% non-monotonicity at rho 0.99 vs 0.999 above is 20-seed noise
and should not be read as a real effect. Per rule 9 the assertion is on the RATE across seeds, and
the rate needs enough seeds to be stable.

**Report the measured FP rate in the test output either way.** When two suites disagree on a
number, the number settles it — and it can only settle it if both print it.

## Consequence for obligation 4

Obligation 4 (iid data, bootstrap SE vs analytic SE) is unaffected and stays as written: it targets
the L7 raw-SE inflation directly and is RED at ~29x in both suites. Between them, obligation 4
catches the inflating direction and obligation 5 — with a persistent feature — catches the
understating one. Neither alone is sufficient, which is why both exist.

---

# AMENDMENT 4 — 2026-08-20. Obligation 5 is DEMOTED. Rule 9 called this in advance.

## The measurements, all on the same defective code

    suite A, 30 seeds, persistence incidental      FP = 36.7%
    lead,    20 seeds, rho = 0.99                  FP = 10.0%
    lead,    20 seeds, rho = 0.999                 FP =  5.0%
    suite B, 30 seeds, rho = 0.99                  FP =  6.7%
    suite B, 50 seeds, rho = 0.99                  FP =  0.0%

Two of those differ only in seed COUNT — 6.7% at 30 and 0.0% at 50, same rho, same fixture. That is
not a defect in either measurement. It is what rule 9 says happens:

> Coverage checks flake; power checks do not. A "does the CI contain X" assertion fails with
> probability ~alpha per comparison regardless of effect size. A detection check with large
> effect-to-SE ratio is effectively deterministic.

An FP-rate assertion is the flaky family. We have now spent three rounds refining it and it is
still not stable enough to gate on.

## Decision

**Obligation 5 is demoted from a gating assertion to a reported diagnostic.** It stays in the
suite, it prints its measured rate, and it is read as evidence — but the suite's verdict does not
hinge on it.

**The gating detectors are the deterministic ones:**

- **Obligation 4** — bootstrap SE vs analytic iid SE. Measured 29.5x in both suites,
  independently. A 29x discrepancy has no seed sensitivity worth the name.
- **Obligation 6** — power on a large known effect. Measured 23.3% against a >= 90% requirement.
  Large effect-to-SE ratio, so effectively deterministic, exactly as rule 9 describes.
- **Obligation 9** — the SE ratio against an independently resimulated empirical SE, inside a
  stated two-sided band. Deterministic once `se_bps` is exposed.

Between them, obligation 4 catches the inflating direction and obligations 6 and 9 catch the
understating one, both without depending on a rate estimated from 30 draws.

## The prohibition this makes concrete

The 6.7%-at-30 / 0.0%-at-50 pair is precisely the situation where someone reaches for the seed
count that produces the preferred answer. **Do not.** Rule 9's second half is not advisory:

> Never fix such a test by changing the seed -- that is the same failure as re-tuning a strategy
> default to make a null test pass.

Choosing the seed COUNT to get a wanted rate is the same act as choosing the seed. If the rate is
unstable, the correct response is to stop gating on the rate, which is what this amendment does.

## Still outstanding in BOTH suites

`rho = 0.99` is currently HARD-CODED. AMENDMENT 3 required it to be DERIVED from the measured lag-1
autocorrelation of a real feature (`volume_zscore` or `breakout_strength`) on real panel data, with
the measurement recorded beside the value. That is not yet done and remains required — rule 8 does
not exempt a fixture parameter just because it lives in a test. It is now the ONLY hand-chosen
constant left in this spec.

---

# AMENDMENT 5 — 2026-08-20. AMENDMENT 4's demotion is REVERSED. Obligation 5 gates again.

## What changed

AMENDMENT 4 demoted obligation 5 to a diagnostic because its measured FP rate would not stabilise
across three rounds. That judgement was made on an UNDER-POWERED fixture, and it was wrong.

One suite went and did what AMENDMENT 3 actually asked: it DERIVED the persistence from real panel
data instead of hard-coding it, and rebuilt the fixture around it. Independently verified by the
lead, three base seeds of 40 draws each:

    base_seed=   0   FP = 45.0%   (ceiling 15%)
    base_seed=1000   FP = 35.0%   (ceiling 15%)
    base_seed=2000   FP = 45.0%   (ceiling 15%)

That is not a marginal statistic. The effect-to-ceiling ratio is 2.3x-3x and it does not cross the
threshold on any draw. In rule 9's own terms this has become a large effect-to-SE detection check,
which is the family the rule says IS reliable — it just had to be made powerful enough to qualify.

**Obligation 5 is restored as a GATING assertion**, under the pinned configuration below and no
other. AMENDMENT 4's demotion is reversed; its prohibition on shopping for a seed count stands
unchanged and is untouched by this.

## The pinned configuration

    feature        AR(1) per symbol, independent across symbols, rho = 0.9613 (DERIVED, see below)
    n_rows         3000
    n_symbols      5          (>= 5: cross_sectional_rank enforces min_names = 5)
    n_buckets      5
    horizon        20
    n_boot         150
    n_seeds        >= 100
    ceiling        0.15       (3x nominal alpha, stated explicitly as a 3x allowance)

## The derivation, which is the part that matters

`rho = 0.9613` is the measured lag-1 autocorrelation of `breakout_strength`
`= (close - prior_window_high) / sigma` at `window = 30` — the repo's own production default
(`VolumeBreakoutConfig.breakout_window`) — computed on real panel data for ten liquid names
(RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, ITC, LT, AXISBANK, KOTAKBANK) over
2024-01-01..2024-03-31. Decay: 0.9298 (lag 2), 0.8489 (lag 5), 0.7332 (lag 10).

**`volume_zscore` was also measured and REJECTED**: lag-1 ACF only 0.2837, decaying to ~0 by lag
10, and confirmed empirically to give 0.0% FP at n_rows 300 / 3000 / 15000 — genuinely too weak to
exercise L8. This is recorded so that nobody picks it later believing it was simply overlooked. The
choice was made on measured structural persistence, not on which candidate produced a bigger
number.

## The general lesson, worth more than this obligation

"This statistic is unstable" and "this fixture is under-powered" look identical from the outside and
are opposite problems. The first calls for abandoning the check; the second calls for strengthening
it. AMENDMENT 4 reached for the first without ruling out the second.

The distinguishing question is cheap: does the effect move away from the threshold when the fixture
is made stronger? Here, raising buckets 2 -> 5, horizon 5 -> 20 and seeds 30 -> 100 took the rate
from a 0%-37% smear straddling the ceiling to a 35%-45% band that never approaches it. Ask that
question before demoting a test.

## Required of both suites

Both adopt the pinned configuration above verbatim, so their numbers are directly comparable. A
suite still running `rho = 0.99` hard-coded, `n_buckets = 2`, `horizon = 5` is measuring a
different, weaker experiment and its FP number should not be compared with the other's.

---

# AMENDMENT 6 — 2026-08-20. The denominator is pinned, and MY ceiling violated rule 8.

The suites now derive different persistence from nominally the same measurement, and it flips the
verdict:

    suite A:  rho = 0.9613   ->  FP 35-45%  ->  obligation 5 RED
    suite B:  rho = 0.82     ->  FP 11.0%   ->  obligation 5 GREEN

Two problems, one theirs and one mine.

## 1. The feature's denominator was ambiguous. Pinned to Parkinson.

Suite B computed `breakout_strength = (close - prior_30bar_high) / std(close)`. Suite A used
`parkinson_volatility`. `specs/feature_layer.md` D2 defines the sigma as coming from
`parkinson_volatility` on the same window, so suite A matches the spec and suite B measured a
DIFFERENT FEATURE — which is why the ACFs differ (0.82 vs 0.9613) and why one suite detects the
defect and the other does not.

**Pinned:** `sigma = parkinson_volatility(high, low, window=30)`. Both suites reproduce the
derivation with that denominator and record the per-symbol ACF table beside the constant.

This is worth noticing as a pattern: a feature specified by formula but not by its inputs is not
specified. The same gap produced the `n_symbols_used` / `symbol_count` split and the
`ResearchContract` seed placement. An unnamed input is an interface with no owner.

## 2. My `ceiling = 0.15` was a hand-chosen constant. Rule 8 applies to MY numbers too.

AMENDMENT 5 pinned "ceiling 0.15 (3x nominal alpha, stated explicitly as a 3x allowance)". That is
exactly the thing rule 8 forbids — a cutoff reasoned rather than measured — and I wrote it into a
spec whose entire purpose is fixing a statistical defect. It also does not work: it admits an 11%
false-positive rate, which is **2.2x nominal alpha**, as a pass.

**The ceiling is now DERIVED** from the sampling distribution of the rate itself. Under a correct
estimator the count of false positives is `Binomial(n_seeds, alpha)`, so the ceiling is a high
quantile of that, divided by `n_seeds`:

    n_seeds=100  99.9th pct -> 0.1300      n_seeds=100  99.99th pct -> 0.1500
    n_seeds=200  99.9th pct -> 0.1050      n_seeds=200  99.99th pct -> 0.1150
    n_seeds=400  99.9th pct -> 0.0875      n_seeds=400  99.99th pct -> 0.0950

**Pinned: `n_seeds = 400`, `ceiling = 0.0875`** (the 99.9th percentile of `Binomial(400, 0.05)/400`,
computed with `scipy.stats.binom.ppf(0.999, 400, 0.05)/400`). Compute it in the test rather than
hard-coding 0.0875, so the derivation is executable and moves correctly if `n_seeds` changes.

At 400 seeds this ceiling rejects an 11% rate — so obligation 5 goes RED under BOTH suites'
persistence values, and the verdict stops depending on which sigma someone picked. A test whose
outcome hinges on an unpinned implementation detail is not a test.

The 99.9th percentile means a correct estimator flakes here about 1 run in 1000, which is the
flake budget rule 9 asks us to spend deliberately rather than by accident.

## Required of both suites

- `sigma = parkinson_volatility(high, low, window=30)` in the rho derivation.
- `n_seeds = 400`; ceiling computed in-test from `binom.ppf(0.999, n_seeds, 0.05) / n_seeds`.
- Report the measured FP rate AND the per-symbol lag-1 ACF table.
- If your measured rho differs materially from the other suite's, say so rather than proceeding —
  that disagreement has now twice been the most informative output of this obligation.

---

# AMENDMENT 7 — 2026-08-20. Where L8 actually lived; and both suites need reconciling.

## 1. The real L8 location was NOT `:470-471` alone

The implementer found, while fixing it, that `ExpectancyTable.spread_t` was never computed from
`BucketStat.t_stat` / `se_bps` at all. `conditional_expectancy` recomputed a SEPARATE spread SE as
`std_bps / sqrt(n_effective)` — and that is where the over-correction actually leaked into the
table-level statistic the Lens reads.

So the defect had two mouths: the `n_effective` transform, and a table-level SE that bypassed the
bucket SEs entirely. Fixed by combining `bucket_stats[-1].se_bps` and `bucket_stats[0].se_bps` in
quadrature.

This also explains AMENDMENT 2's finding cleanly: `BucketStat.t_stat` and `ExpectancyTable.spread_t`
were not two views of one number, they were two different computations. A suite asserting on the
former could not see the defect in the latter.

## 2. The derived block length

`block_length = horizon + BLOCK_LENGTH_EXTRA_BARS`, `BLOCK_LENGTH_EXTRA_BARS = 5`, DERIVED from
lag-1..30 autocorrelation of same-session 1-minute log returns for 10 liquid names over
2024-01-01..2024-03-31, read directly from the panel. All ten show the classic bid-ask-bounce
NEGATIVE lag-1 ACF (-0.040 to -0.066) decaying below 0.02 within 2-5 bars; p95 decay lag = 5. Table
recorded in source beside the constant, `lens.py`-style. This satisfies rule 8.

## 3. Suite B: SIX tests are unconditional `pytest.fail()` and can never pass

Obligations 7, 8 (x2), 9, 10 (x2) build a result and then call `pytest.fail(...)` unconditionally,
with a comment describing what the assertion should have been.

That was the right shape when the observation did not exist — I asked for exactly that. **It is now
wrong**, because `se_bps`, `n_effective`, `block_length` and the skip count all exist. An
unconditional failure is not a test; it cannot distinguish a correct implementation from a broken
one, which is the same defect as a tautology pointing the other way.

**Required:** convert all six into real assertions against the now-existing API.

## 4. Suite A: three tests still unpack a 2-tuple

Obligations 1, 2 and 10 do `resampled, _ = _block_bootstrap_resampling_2d(...)` while obligation 3
IN THE SAME FILE correctly expects the AMENDMENT-2-pinned 3-tuple. Internally inconsistent, and
can never pass against a spec-compliant implementation.

AMENDMENT 2 anticipated this exact failure mode when it pinned the arity: "a `ValueError` on unpack
is indistinguishable from the feature being absent." Reconcile all call sites to the 3-tuple
`(resampled, block_indices, n_sessions_skipped)`.

## 5. `test_expectancy_internal.py` — three stale tests, LEAD ADJUDICATION

Not one of this spec's dual suites, but three of its tests unpack the 2-tuple or mock a 2-tuple
return:

    test_block_bootstrap_blocks_never_straddle_session
    test_block_bootstrap_fallback_path_directly
    test_all_nan_bootstrap_sample_produces_nan_not_warning   (its mock branch)

They predate the pinned 3-tuple contract and are now stale. **Adjudicated: update them to the
3-tuple.** Their assertions are otherwise correct and must be preserved — in particular the
never-straddle-a-session check, which is exactly what the rewritten bootstrap must keep honouring.

## 6. Both suites were left unreconciled against AMENDMENT 2

Neither suite was fully updated after AMENDMENT 2 published the pinned arity — suite B papered over
it with a both-arities helper AND unconditional failures, suite A applied it to one obligation
only. Worth noting as a process fact: when an amendment changes an interface, EVERY suite written
against that spec needs an explicit reconciliation pass, and "the agent said it applied the
amendment" is not the same as every call site having been updated.

---

# AMENDMENT 8 — 2026-08-20. The iid fallback must be VISIBLE, not silent.

A test author found a real behavioural change and reported it instead of patching the assertion,
which is the correct call and is why this adjudication exists at all.

## The change

`test_expectancy_internal.py::test_block_bootstrap_fallback_path_directly` asserts
`len(block_indices) == 5`. It now gets 0.

- OLD fallback: appended a synthetic `(0, n_rows)` entry to `block_indices` per replicate.
- NEW fallback: does whole-row iid resampling and never appends, so `block_indices` stays empty.

## Adjudication: the NEW behaviour is correct, and the test updates

An entry of `(0, n_rows)` claims a block of length `n_rows` was drawn. No such block was drawn, and
`n_rows != block_length`, so it also violates the invariant that every block has the derived length
— which obligation 10 asserts. A synthetic entry would make `block_indices` lie about what happened.

Empty is honest: no blocks were drawn, because no session was long enough to draw one from.

## But an empty list is not, by itself, an adequate signal

Falling back to iid resampling **changes the estimator**. Every dependence correction this spec
exists to provide is gone on that path, and a caller who does not notice gets a naive SE while
believing they got an overlap-aware one. That is the same species as rule 6's prohibition on silent
gap-filling: *"any gap-repair/fill must be opt-in and visible at the call site, never silent."*

**Required: the fallback must be detectable from the return value alone**, without reading the
source. It already is, and the contract is now pinned as such:

    n_sessions_skipped == <total sessions>  AND  len(block_indices) == 0   <=>  iid fallback was used

Both conditions together, and the docstring must state this explicitly so a caller can test for it.
If a future change makes that conjunction ambiguous, add an explicit flag rather than leaving it
inferable.

**Restated test:** `test_block_bootstrap_fallback_path_directly` asserts
`len(block_indices) == 0` AND `n_sessions_skipped == 3` (all three 3-bar sessions are shorter than
`block_length = 10 + 5 = 15`), and states in a comment that the two together ARE the fallback
signal. That is a stronger assertion than the count it replaces, because it pins the detectability
of a silent estimator change rather than an incidental list length.

## Also noted

The same author replaced that file's old "we can't check this directly" comment with a real
per-block session-boundary check, now that `block_indices` exposes `(start, length)`. That is
worth more than the arity fix it came in with: the test's name promised blocks never straddle a
session, and until now the file could not actually verify it.

---

# AMENDMENT 9 — 2026-08-21. Row-weighted, not cell-weighted. Adjudicated.

The bucket bootstrap now reduces the masked 2-D bucket panel to a length-`n_rows` series of per-row
bucket MEANS before resampling, rather than resampling the 2-D panel. An implementer flagged that
this moves `se_bps` and stopped rather than shipping, per the condition attached to the task. Correct
call, and the change is adopted. Here is why, and what the numbers actually say.

## The measured disagreement

    adversarial fixture (6 symbols, 60% coverage, one all-NaN symbol):  7.3% - 14.7%
    realistic geometry  (149 symbols, 95% coverage):                    0.06% - 0.25%

The gap is **row-weighted vs cell-weighted averaging under heterogeneous per-row coverage**, and it
only becomes large when coverage is patchy and the cross-section is small.

## Why row-weighting is CORRECT, not merely cheaper

This is the deciding argument, and it is about what the statistic is supposed to represent.

Cell-weighting averages over stock-days. A bar with 30 names in the bucket gets ten times the
influence of a bar with 3. That answers "what is the average return of a stock-day in this bucket".

**Row-weighting averages over bars, which is what a strategy actually earns.** You trade each bar
with whatever names are available that bar; the realised return for that bar is the equal-weighted
mean of the bucket's members; the strategy's return is the average across bars. A bar is one
decision and one P&L event regardless of how many names happened to be eligible.

So the two estimators answer different questions and we want the tradable one. That the change is
also ~149x cheaper is a happy accident, not the justification — and it was NOT adopted for that
reason.

Secondary benefit: `research/ic.py` already aggregates per-bar and then across bars. Two code paths
that should agree now do.

## Consequence: one pinned test is stale, not regressed

`tests/test_expectancy_bootstrap_chunking.py::test_se_bps_matches_recorded_pre_fix_baseline` pins
the pre-change `se_bps` to 1e-12. It was written to prove a MEMORY fix changed nothing, which was
true and valuable at the time. This change deliberately alters the estimator's granularity, so that
baseline no longer describes the intended behaviour.

**Adjudicated: re-baseline it to the row-weighted values, and rename it** so it states which
estimator it pins. Its purpose survives — catching an unintended drift in `se_bps` — but it must
pin the estimator we chose, not the one we replaced. Do NOT delete it: an unpinned SE is how L7 and
L8 went unnoticed for the life of the repo.

## What must be re-verified after the change

The false-positive rate must still sit near alpha, and the SE ratio against independently
resimulated empirical SEs must still be near 1.0 (0.977 pre-change). Those two measurements, not
the pinned baseline, are what establish the estimator is right.
