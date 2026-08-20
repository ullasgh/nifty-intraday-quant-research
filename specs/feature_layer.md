# Spec D — the feature layer

Status: spec, written before implementation. Author: lead. Date 2026-08-20.

## Context established by inventory, so the work is smaller than it looks

**14 of the 15 public functions in `features/market.py` are DEAD** — built, spec'd, tested, and
called by nothing. Only `tradable_overnight_return` is wired (`h2_overnight_reversal.py:50,78`,
reached via `research/tilt.py:380`). No plugin imports `features.market` at all.

So most of what the external review asked for already exists and is unwired. The genuinely new work
is small, and the largest single item is not a feature at all — it is the gap-stitched price path,
because without it the Hurst estimate cannot be made causal at the session boundary.

## D1 — the gap-stitched price path. This one is load-bearing.

**The problem, measured.** `rolling_hurst(price, window=390, ..., day_offsets=None)` is called by
`volume_breakout.py:132-146` with `day_offsets=None` **deliberately**: with day bounds and a
390-bar window against ~375-bar sessions, `_segment_bounds` confines every window to one session
and the result was `0/92520 finite on RELIANCE 2024`. Not a bug — an arithmetic impossibility. A
390-bar window cannot fit inside a 375-bar session.

So today the estimator sees overnight jumps as if they were one-minute returns, and the documented
workaround is to accept that. Neither branch is acceptable: day-bounded gives no data, unbounded
gives a contaminated estimate.

**The resolution.** Stitch the path, then estimate on the stitched path without day bounds:

    stitched[0]        = close[0]
    stitched[t]        = stitched[t-1] * (close[t] / close[t-1])       within a session
    stitched[t]        = stitched[t-1]                                  across a session boundary

i.e. the overnight return is set to zero rather than carried, so the series is continuous in level
and the cross-session log-difference is 0 instead of a jump. The window may then legitimately span
sessions, because what it spans is a series with the overnight jumps removed.

**This is NOT forward-filling and must not become it (rule 6).** A stitched value is only ever
produced where a real bar exists. `NaN` in `close` stays `NaN` in `stitched`; the stitch never
manufactures a bar, it only removes a jump between two real ones. The function must be named so
this is obvious (`stitch_overnight_gaps`, not `fill_*`), and it must be an explicit, visible call
at the call site — never applied silently inside another estimator.

Nothing like this exists today: zero `stitch` hits in `src/`. `strategy/base.py:105 _ffill_2d` is a
different thing (opt-in NaN carry) and must not be reused for this.

**Then** `hurst_on_stitched(close, day_offsets, window=390)` is the causal call path, and
`H > 0.55` stops being an assumption: D5 measures whether the relationship is monotonic at all.

## D2 — breakout strength, continuous

`breakout_up`/`breakout_down` (`core.py:660`, `:685`) return booleans. A boolean throws away how far
through the level the price went, which is the part that carries information.

    breakout_strength = (close - prior_window_high) / sigma

with `sigma` from the existing `parkinson_volatility` on the same window, and the prior window high
taken from `rolling_max` shifted one bar (the existing convention — do not change it). Negative
values are meaningful: they say how far BELOW the level price sits.

Keep the booleans for compatibility. They become `breakout_strength > 0`, and a test must assert
that identity so the two cannot drift apart.

## D3 — volatility estimator family

Today exactly three exist: `parkinson_volatility` (`core.py:311`), `ewma_volatility_ann`
(`core.py:377`, **zero callers**), and close-to-close assembled from `log_returns` + `rolling_std`.

Add **Garman-Klass** and **Rogers-Satchell**, both day-offset aware and NaN-propagating on the same
convention as `parkinson_volatility`. Add:

    sigma_risk = max(sigma_ewma, sigma_floor)

available to the sizer. `sigma_floor` is a THRESHOLD and rule 8 applies: derive it from the measured
distribution of realised per-symbol volatility on real data and record the derivation beside it. Do
not pick a round number. The floor exists to stop the inverse-vol sizer from taking an unbounded
position in a temporarily-still name, so the honest derivation is a low percentile of the measured
distribution, and which percentile must itself be justified by what position size it implies.

## D4 — efficiency ratio

    efficiency_ratio = abs(close[t] - close[t-n]) / sum(abs(diff(close))[t-n+1 : t+1])

Session-bounded via `day_offsets`. Complements `amihud_illiquidity` (which exists and is dead).
Ranges [0, 1]; 1 is a straight line, near 0 is pure churn.

## D5 — IC, rank IC, and decay. None of this exists.

Confirmed by grep: zero hits in `src/` for `spearman`, `rank_ic`, `information_coefficient`,
`decay`, or `half_life`. The only near-hits are unrelated (`sharpe_standard_error(adjust_autocorr=)`,
`ewma_volatility_ann(halflife=)`).

New module `research/ic.py`:

- `information_coefficient(feature, fwd, day_offsets, *, method="pearson"|"spearman")` — computed
  CROSS-SECTIONALLY per bar, then aggregated across bars. Not pooled: pooling across time and
  symbols conflates a cross-sectional signal with a time-series one, and this program's live
  candidate is cross-sectional.
- `ic_decay(feature, close, day_offsets, *, horizons)` — IC at each horizon, plus the fitted
  half-life. Returns the horizon grid and the IC at each, never a single summary number.

**The SE on any IC aggregate must come from the corrected overlap-aware machinery in
`specs/overlap_se.md`, not a naive iid formula.** Overlapping forward returns make the naive SE
wrong in exactly the way L7/L8 documents. This spec must not land before that one.

## D6 — wire `features/market.py` into the research harness

Beta-residual return, breadth, dispersion, signed volume, CLV and the VIX ratio become reachable
from the Phase E sweep. This is plumbing, not new computation — but note `vwap_reversion.py:149`
hand-rolls its own `bars_since_open` instead of importing `market.py`'s, and that duplication is
removed as part of the wiring.

Also note `median_pairwise_correlation` (`market.py:337`) is named by
`specs/portfolio_vol_target.md:57` as the rho source while `backtest/portfolio.py` takes rho as a
scalar parameter. That is a live seam between two specs and ONE agent must own both sides of it.

## D7 — rename the volume z-score concept layer

`volume_zscore` keeps its function name. The CONCEPT layer — docs, verdict text, any strategy
parameter naming — becomes `abnormal_volume_activity`.

Reason, and it is not cosmetic: without order-level data a 1-minute volume spike is abnormal
activity, not proven institutional flow. The current framing asserts a mechanism the data cannot
support, and this program has already killed one strategy built on that assertion.

## What must NOT happen

`features/calibrate.py` (`calibrate_null`, `suggest_threshold`) is **100% dead outside its own
tests** — every rule-8 threshold in production was instead derived by standalone
`scripts/calibrate_*.py`. Do not add a ninth threshold derived by yet another one-off script. Either
route new derivations through `features/calibrate.py`, or record explicitly why it does not fit
(its `estimator` is a 1-D price -> scalar callable, which is incompatible with the 2-D rolling
contract `persistence.null_distribution` uses — that mismatch is itself worth fixing).

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. `stitch_overnight_gaps` leaves within-session log-returns EXACTLY unchanged, bit for bit.
2. It sets the cross-session log-difference to exactly 0.0 at every session boundary.
3. `NaN` in `close` produces `NaN` in the stitched output at the same index — the stitch never
   manufactures a bar (rule 6). Assert on a fixture with interior NaNs.
4. `rolling_hurst` on the stitched path with `window=390` returns a MAJORITY-finite result on a
   multi-session fixture, where the same call day-bounded on the unstitched path returns
   all-NaN. Use IRREGULAR sessions (60, 105, 375) — this is the arithmetic-impossibility
   regression test and it must fail loudly if someone reintroduces day-bounding.
5. `breakout_strength > 0` is identical to `breakout_up` on random data, elementwise.
6. `breakout_strength` is signed and non-zero when price is below the level.
7. Garman-Klass and Rogers-Satchell match hand-computed values on a small fixture, propagate NaN
   on the `parkinson_volatility` convention, and respect `day_offsets`.
8. `sigma_floor` has a recorded derivation from measured data beside its value; the value is not a
   round number chosen by hand.
9. `efficiency_ratio` is exactly 1.0 on a monotone ramp and near 0 on a zig-zag with equal steps.
10. `information_coefficient` with `method="spearman"` on a perfectly rank-monotone fixture is
    exactly 1.0, and -1.0 when reversed.
11. IC is computed cross-sectionally per bar and NOT pooled — construct a fixture where the pooled
    and cross-sectional answers DIFFER, and assert the cross-sectional one. A fixture where both
    agree tests nothing.
12. `ic_decay` returns the full horizon grid, not a scalar, and its half-life on a synthetic
    exponentially-decaying IC matches the known value.
13. IC aggregate SEs come from the overlap-aware path — assert they differ from the naive iid SE
    on an overlapping-horizon fixture.
14. `vwap_reversion` uses `market.bars_since_open` and no longer defines its own.

---

# AMENDMENT 1 — 2026-08-20. Five unspecified interfaces, pinned.

A test author reported five places where this spec named a thing without specifying it. Each one
forced a guess, and two independent guesses become two suites that cannot both pass. Pinning all
five, plus module placement, which I omitted entirely.

## 1. Garman-Klass and Rogers-Satchell formulas

Named but never given, unlike every other formula in this spec. Pinned, both as VARIANCE per bar,
averaged over the window and then square-rooted, matching `parkinson_volatility`'s existing
convention for scaling, NaN propagation and `day_offsets` handling:

    GK:  sigma_sq = mean_over_window[ 0.5 * ln(H/L)**2 - (2*ln(2) - 1) * ln(C/O)**2 ]
    RS:  sigma_sq = mean_over_window[ ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ]

Garman-Klass can go NEGATIVE on a single bar (the subtracted term can dominate when open and close
straddle a narrow range). Clip the WINDOW MEAN at zero before the square root, never the individual
bar terms — clipping per bar biases the estimator upward, and this estimator's whole selling point
is efficiency relative to close-to-close.

## 2. Module placement — omitted entirely, my error

    features/persistence.py :  stitch_overnight_gaps, hurst_on_stitched
    features/core.py        :  breakout_strength, garman_klass_volatility,
                               rogers_satchell_volatility, efficiency_ratio,
                               sigma_risk, SIGMA_FLOOR
    research/ic.py          :  information_coefficient, ic_decay   (already named in the spec)

An unnamed module means a RED suite fails on `ImportError` at a guessed path, which is
indistinguishable from a genuine gap — the same failure mode already recorded in
`specs/research_contract.md` AMENDMENT 1 item 2. I made the identical omission again here.

## 3. `SIGMA_FLOOR` and `sigma_risk`

    core.SIGMA_FLOOR   -- module-level constant, DERIVED, with the derivation recorded beside it
                          in the style of lens.py:24-50 (CONCENTRATION_RATIO_THRESHOLD)
    core.sigma_risk(sigma_ewma, *, floor=SIGMA_FLOOR) -> np.ndarray   # elementwise max

Rule 8 governs `SIGMA_FLOOR`: derive it from the measured distribution of realised per-symbol
volatility on real data, and justify the chosen percentile by the position size it implies for the
inverse-vol sizer. A round number is a rule-8 violation regardless of how sensible it looks.

## 4. `efficiency_ratio` window convention

`window` counts BARS INCLUSIVE, matching `rolling_max` / `rolling_std` / `parkinson_volatility`:

    efficiency_ratio[t] = abs(close[t] - close[t - window + 1])
                          / sum(abs(diff(close))[t - window + 2 : t + 1])

so the numerator spans `window` bars and the denominator sums `window - 1` first differences.
Session-bounded via `day_offsets`; the first `window - 1` rows of each session are NaN.

## 5. `information_coefficient` needs `horizon`, and both return types are pinned

The signature in the spec body has no `horizon`, yet obligation 13 requires overlap-aware SEs,
which cannot be computed without knowing the label horizon. That is a genuine contradiction in my
text. Pinned:

    information_coefficient(feature, fwd, day_offsets, *, horizon,
                            method="pearson"|"spearman") -> ICResult
        ICResult:  .mean  .se  .n_bars  .method  .horizon

    ic_decay(feature, close, day_offsets, *, horizons) -> ICDecay
        ICDecay:   .horizons  .ic  .se  .half_life

`ICResult.se` MUST come from the overlap-aware machinery in `specs/overlap_se.md` — that spec is
now implemented and verified (false-positive rate 34% -> 7%), so there is no excuse for a naive iid
SE here. Obligation 13 asserts the two differ on an overlapping-horizon fixture.

## Note on how these were found

Every one of the five was reported rather than guessed silently, with the assumption documented
in the test's own docstring. That is what makes them cheap to fix now instead of expensive to
discover when two suites disagree.

## AMENDMENT 1, addendum — obligation 13's DIRECTION is pinned, not just "differs"

A test author asserted `result.se > naive_se` and flagged it as stronger than the obligation's
"must differ" text, noting they could not verify the direction from the spec. The direction is
sound and is now pinned.

Overlapping forward labels induce POSITIVE dependence between adjacent observations: two labels
sharing `horizon - 1` bars of their return window move together by construction. An iid SE assumes
independence, so it divides by an effective sample size larger than the truth, and therefore
**understates**. An overlap-aware SE must exceed it.

    obligation 13:  result.se > naive_se     (not merely !=)

This is also the direction the implemented `specs/overlap_se.md` work already confirmed from the
other end: the pre-fix estimator's understated SE inflated `spread_t` and drove a 34% false-positive
rate, which fell to 7% once the SE was computed with dependence respected.

An implementation producing a SMALLER SE than naive on an overlapping-horizon fixture is wrong, and
this obligation should say so rather than accept any difference.

---

# AMENDMENT 2 — 2026-08-20. Obligation 13's "reuse the overlap machinery" was under-specified.

A test author fed IC values through `_compute_bucket_stats` and got back essentially the naive SE.
Diagnosis, and it is correct: that machinery is built for **bps-scaled price returns partitioned
into buckets**, not for a **correlation-coefficient series**. The spec said "reuse the corrected
overlap-aware machinery" without saying which part, and the two parts have very different
generality.

## The resolution: reuse the RESAMPLER, not the bucket-stats wrapper

The block bootstrap itself is unit-agnostic — it resamples rows of a 2-D array while respecting
session boundaries, and it does not care whether those rows are basis points or correlations. The
bucket-stats layer above it is what encodes bps semantics and a two-bucket spread.

    REUSE:  the moving-block resampler, its DERIVED block length (horizon + 5, measured), its
            within-session drawing, and its skipped-session accounting
    DO NOT REUSE: _compute_bucket_stats, which is bucket-and-bps specific

`research/ic.py` builds its own aggregator on top of the shared resampler: resample the per-bar
cross-sectional IC series, take the mean per replicate, and take the standard deviation across
replicates. Same dependence handling, correct units.

**Required refactor:** the resampler must be importable independently of the bucket-stats wrapper.
If it is currently only reachable through `_compute_bucket_stats`, extract it — with a single
implementation, not a copy. Two block bootstraps that could drift apart is precisely the
near-duplicate hazard already recorded twice in this program (`filled_frac`, `config_hash`).

**The block length is shared and stays derived.** `horizon + 5`, from the measured ACF of
same-session 1-minute log returns. Do NOT re-derive a separate constant for IC without measuring
it; if the IC series' own dependence structure turns out to differ materially, that is a
measurement to make and record, not a number to pick.

## Why the naive-SE result was the right thing to report

Getting back "essentially the naive SE" is the exact signature of dependence handling silently not
applying — the same class as the `min_names=5` all-NaN trap and the iid fallback. It looked like a
working pipeline and was not. Reporting it instead of accepting the number is what kept obligation
13 meaningful, and the strict `se != naive_se` assertion should stay strict.
