# Spec: repair kill criterion 4 (liquidity concentration) in `Lens`

**Status:** contract for TDD. Two independent test suites written from this document ALONE.
Neither author reads the other's file, and no implementation exists yet.

Same file as `specs/lens_criteria_6_7_repair.md` (`src/nifty_quant/research/lens.py`). A SINGLE
implementer handles both specs in one pass — never two agents at this file concurrently.

**Why this matters more than the other two:** criterion 4 is the SOLE surviving kill reason for
H2, the largest measured effect in the program (pooled -24.30 bps, t = -16.66, 8/8 years).
Once clip size is accounted for, H2 clears criteria 1 and 7 at a Rs 10L clip. So this criterion
alone decides whether H2 is dead.

---

## Defect 1 — it buckets on SHARE COUNT, not rupee turnover

`Lens.stability_report`, around `lens.py:468`:

```python
volume = self.panel.field("volume").astype(np.float64)
```

`volume` is raw share count. A name trading at Rs 1 lakh/share has tiny share volume and large
rupee turnover, so it is classified "least liquid" while being highly liquid. **Measured**: the
share-volume bottom decile is `BAJAJHLDNG, BOSCHLTD, GLAXO, MRF, OFSS, PAGEIND, PGHH, SHREECEM,
SOLARINDS` — high-share-price blue chips with a median rupee ADV of ~Rs 22 crore/day. The
rupee-turnover bottom decile is `BAJAJHLDNG, EMAMILTD, GICRE, GLAXO, NIACL, PGHH, SOLARINDS`.
**Only 4 of 12 names overlap.**

Liquidity must be **rupee turnover = close * volume**, the quantity `tradable_mask` and capacity
sizing already use.

## Defect 2 — full-sample lookahead

```python
# Compute volume deciles (causal, per row)
volume_quantiles = np.quantile(volume[np.isfinite(volume)], np.linspace(0, 1, 11))
```

`np.quantile` over the whole flattened array derives thresholds from the ENTIRE panel including
the future. The comment claims "causal, per row"; it is not. A full-sample quantile is exactly
the lookahead this repo's `causal_buckets` exists to prevent.

The correct machinery already exists and is unused here:
`expectancy.expectancy_by_liquidity_decile(feature, fwd, day_offsets, prior_adv, ...)`, whose
docstring states `prior_adv` "must already be lagged (strictly prior sessions). The function
does not re-derive it, and @causal cannot detect a leak in it — so document that loudly at the
call site."

---

## Required behaviour

1. **Liquidity is rupee turnover**: `close * volume`, float64, NaN-preserving (NaN means "no
   bar" and must never be forward-filled or zero-filled).
2. **`prior_adv` follows the repo's EXISTING liquidity convention.** `data/validate.py:727-738`
   (`tradable_mask`) already defines ADV, and criterion 4 must not invent a second,
   inconsistent definition of "liquidity" for the same repo:

       day_value[s, :] = np.nansum(close[day_slice] * volume[day_slice], axis=0)   # per-session TOTAL
       adv[s, :]       = np.nanmean(day_value[max(0, s - 20) : s, :], axis=0)      # trailing 20, STRICTLY PRIOR

   - Aggregate per session with **`nansum`**, not a per-bar mean. This matters and is
     directionally biased: a name trading in 10 of 375 bars has the same per-BAR mean as one
     trading all 375, but 1/37 the daily value. Those thin names are exactly the bottom decile
     this criterion is about, so a per-bar mean systematically flatters illiquid names and
     corrupts decile-0 membership. Note at the call site that the per-session `nansum` is a
     statistic, not a rule-6 fill.
   - Then a **trailing 20-session, strictly prior** mean; session 0 is NaN.
   - Session boundaries come from `panel.day_offsets` — **never** a fixed 375-bar stride
     (Muhurat ~60 bars, shortened ~105).

   **On rule 8 and the window length:** 20 is NOT a new hand-chosen constant — it is the
   repo's existing liquidity convention, the same one gating `tradable` and driving capacity
   sizing. Rule 8 governs DECISION CUTOFFS measured against a null, not every estimator
   hyperparameter; choosing a demonstrably worse estimator to dodge it optimises the wrong
   constraint. An expanding mean also freezes a name's rank on 2018-2020 data by 2025, so it
   would test concentration in HISTORICALLY illiquid names rather than names illiquid at trade
   time — the wrong quantity for a capacity criterion.

   **Required sensitivity check:** report the criterion-4 outcome under BOTH trailing-20 and
   expanding-mean `prior_adv`. The kill decision must agree under both. If it does not, that
   disagreement is the finding and must be reported, not resolved by picking one.
3. **Delegate the bucketing** to `expectancy.expectancy_by_liquidity_decile` rather than the
   inline `np.quantile` + per-cell loop. Do not reimplement it.
4. **`method="cross_sectional_rank"` is MANDATORY at the call site.** That function DEFAULTS to
   `method="expanding_quantile"`, which needs `min_history` rows per column and can never
   bucket a once-per-session signal — it silently returns tables whose `spread_bps` is `0.0`,
   indistinguishable from "no effect". This is the same trap that produced H2's original
   25,000x error, and it is the DEFAULT. Passing it explicitly is part of the contract.
5. **The concentration threshold stays a named module-level constant** with its derivation
   recorded beside it (see below). It must not be an inline literal.
6. Behaviour when a decile is empty or all-NaN is unchanged: it contributes nothing, never 0.0,
   never a crash.

## The threshold — pending a measured null distribution

Criterion 4 currently fires when the bottom decile is the argmax of `|spread|` AND
`max > 2 * median`. **That `2` is a hand-chosen constant, which CLAUDE.md rule 8 forbids** —
every cutoff must be derived from a measured null distribution on real data, with the derivation
recorded next to the value. Rule 8 exists because a reasoned-not-measured threshold once shipped
at `0.35` and produced 8,217 false positives.

`scripts/calibrate_concentration_threshold.py` is deriving that null now (500 within-session
permutation replicates). **Implement the threshold as a named constant read from one place**, so
substituting the derived value is a one-line change. Tests must NOT hardcode `2.0` as the
expected cutoff — parameterise on the module constant, so the suites survive its replacement.

**THE MEDIAN CONVENTION MATTERS AND IS EASY TO GET WRONG.** Production (`lens.py:665`) uses
`sorted_liquidity_edges[len(sorted_liquidity_edges) // 2]` — the UPPER median for an even
count. `np.median` averages the 5th and 6th values and is therefore SMALLER, inflating the
ratio. Every number below uses the production convention. A first pass at this measurement used
`np.median` and read 2.0104 where production gives 1.9959 — on a margin this thin, that
difference alone flips the criterion. Any test or script touching this must use the production
median.

**Measured on real data (2018-01-01..2025-07-31, all_equity), production median:**

    measurement                                          decile-0   max/median   fires
    share volume, full-sample, 10x5   (current code)      -55.33      2.4538      yes
    rupee turnover, full-sample, 10x5                     -60.20      2.6399      yes
    rupee turnover, strictly-prior causal, 10x10          -51.81      1.9959      no
    rupee turnover, strictly-prior causal, 10x5  <-- THIS -47.75      2.1189      yes

**The last row is the one that decides H2**: 10 liquidity deciles x 5 feature QUINTILES, which
is production's actual geometry. The 10x10 row is included only to show how much the feature
bucket count moves the answer — it is not production and must not be quoted as the result.

Observed **2.1189 against a threshold of 2.0** — a 6% margin against a cutoff nobody derived.
That is why the null distribution matters.

## Bucket geometry — state it explicitly, the delegate cannot express it

Production uses **10 liquidity deciles x 5 feature quintiles**: `lens.py` loops
`for decile in range(10)` while passing `n_buckets=5` to `conditional_expectancy` (`lens.py:511`).

`expectancy_by_liquidity_decile` passes ONE `n_buckets` to BOTH the ADV bucketing
(`expectancy.py:946`) and the feature bucketing (`:975`), so it can only do 10x10 or 5x5.
**It cannot reproduce production's 10x5.** The implementer must therefore either:

- (preferred) add a separate `feature_n_buckets` parameter to `expectancy_by_liquidity_decile`,
  defaulting to `n_buckets` so no existing caller changes behaviour — this puts a small,
  additive change to `expectancy.py` IN SCOPE; or
- keep the decile loop in `lens.py` and call `causal_buckets` + `conditional_expectancy`
  directly, which preserves 10x5 but forgoes the delegation.

Either is acceptable. What is NOT acceptable is silently adopting 10x10 or 5x5, which changes
the published statistic while appearing to be a pure refactor.

## Dependency: the null distribution must match this statistic

`scripts/calibrate_concentration_threshold.py` (v1) ports the OLD bucketing — share volume,
full-sample quantiles. Cross-sectional-rank deciles have equal per-row membership by
construction; pooled full-sample deciles do not. The dispersion of decile spreads, and hence
the null of max/median, is a different object. **v1's p95 may not be used to gate the corrected
criterion** — that would be a rule-8 violation dressed as a rule-8 fix.
`scripts/calibrate_concentration_threshold_v2.py` derives the null under the corrected
bucketing at the same 10x5 geometry and the same production median. Both are reported; the
threshold comes from v2.

## Required tests

Two INDEPENDENT suites: `tests/test_lens_criterion4_repair_a.py`, `..._b.py`.

### BLOCKING FIXTURE CONSTRAINT — read before writing a single test

`cross_sectional_rank` returns **all-NaN for any row with fewer than 5 finite values**
(`features/core.py:378,402-403`), and that gate applies TWICE in the delegated path: once to
`prior_adv` (needs >= 5 names per row) and again to the feature INSIDE each decile (needs >= 5
names per decile per row). Measured through the real function:

    n_symbols:  2, 3, 6, 12, 30, 40  ->  0 of 10 deciles produce a non-zero spread
    n_symbols:  45                   ->  5 of 10
    n_symbols:  50+                  ->  10 of 10

**Below ~44 symbols every decile returns `spread_bps == 0.0`** — precisely the "ten silent
zeroes" failure item 7 exists to catch. A 2- or 3-symbol fixture therefore CANNOT distinguish
the old implementation from the new: nothing is bucketed at all, so share-count and
rupee-turnover bucketing are indistinguishable and the test passes against BOTH. That is the
lazy-but-conforming failure mode, and the first draft of suite B fell into it — its units and
causality guards passed against the unfixed code, which means they tested nothing.

**Any test asserting on decile spreads (items 1, 2, 7, 8) MUST use a fixture with >= 50
symbols.** Items 3, 4 and 6 assert on the `prior_adv` array directly and are fine at 1-2
symbols. Ties are safe: `rankdata(method="average")` gives equal values the same percentile
(`core.py:389`), so two names with identical turnover land in the same decile as item 1
requires.

1. **Units regression guard — the one that would have caught this.** Build a panel with two
   symbols: one HIGH price / LOW share count, one LOW price / HIGH share count, with equal
   rupee turnover. Under the correct implementation they must land in the SAME liquidity
   decile. Under share-count bucketing they land at opposite ends. This test must fail against
   the current implementation.
2. **Causality guard.** A symbol whose turnover is tiny early and huge late must be bucketed by
   its STRICTLY PRIOR turnover, not by its full-sample rank. Construct it so a full-sample
   quantile and a strictly-prior one disagree, and assert the prior one wins. Must also fail
   against the current implementation.
3. Session 0 has no prior turnover -> NaN -> those cells are unbucketed, not silently decile 0.
4. `prior_adv` for session `s` excludes session `s` — assert directly on the computed array for
   a hand-checkable fixture, not only through the verdict.
5. An irregular-session panel (~60-bar Muhurat alongside a full session) buckets correctly;
   nothing assumes a fixed bars-per-session stride.
6. NaN turnover (no bar) propagates and is never forward-filled or treated as zero.
7. `method="cross_sectional_rank"` is actually used: a once-per-session signal must produce
   NON-ZERO decile spreads. A suite that would pass against ten silent `0.0` spreads has not
   tested anything — assert at least one decile spread is non-zero and finite.
8. Criterion 4 fires when the bottom decile is the argmax AND ratio exceeds the module
   constant; does NOT fire when the bottom decile is not the argmax, however large the ratio;
   does NOT fire when the ratio is below the constant. Parameterise on the constant.
9. All SEVEN criteria still reported, in order, after this change.
10. Determinism: same inputs and seed -> identical verdict text.

## Constraints

100% line+branch with pragma exclusion DISABLED. Fully annotated; ruff clean. Never touch
`data/`. float64 in motion. `present` and `tradable` stay DISTINCT. Reuse existing `Lens`
fixtures rather than inventing parallel ones.
