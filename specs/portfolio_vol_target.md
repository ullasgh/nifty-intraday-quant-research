# Spec: real portfolio volatility targeting

Phase A track A4.

## Why this exists

`target_vol_ann: 0.15` appears in configs and reads as a portfolio risk target. It is not one.

- `plugins/volume_breakout.py:360-372`, `vwap_reversion.py:213-230`,
  `eod_overextension.py:300-312`: weights are `sign / sigma`, clipped to `max_weight`, then
  scaled down only if gross exceeds the cap. `target_vol_ann` never enters the arithmetic — it
  is written into the meta dict (`volume_breakout.py:379`) and nothing reads it.
- `carver_trend.py:310-317` DOES scale by `target_vol_ann / vol_ann`, but per name, and then
  `:318-321` clips to `max_weight` and rescales to `gross`, which destroys the target whenever
  either constraint binds.
- `grep` for `sigma_portfolio` / `portfolio_vol` / covariance across `backtest/` and `strategy/`:
  nothing. `GrossNotionalSizer.to_shares` (`backtest/portfolio.py:115-190`) is
  `weight * gross * capital` plus capacity water-filling, with no risk input at all.

So the delivered construction is **inverse-volatility allocation normalised to a gross exposure**.
That is a legitimate scheme; the defect is that it is labelled as something else, and that
portfolio risk is therefore whatever the cross-sectional correlation happens to make it.

A second, separate defect surfaced while reading this code: `carver_trend.py:43-45` annualizes
with `_ANNUALIZATION_FACTOR = sqrt(252 * 375)`. **375 is a hardcoded bars-per-session constant**,
which rule 5 forbids — Muhurat sessions are 60 bars and shortened sessions can be 105. Any
annualization introduced by this spec derives bars-per-session from `panel.day_offsets`.

## Required behaviour

### A. `VolTargetSizer`

A new sizer in `backtest/portfolio.py` implementing the same call contract as
`GrossNotionalSizer.to_shares` so the engine can use either:

    to_shares(weights, prices, capital, *, bar_traded_value=None, max_participation=0.02,
              sigma=None, corr=None) -> np.ndarray | SizingResult

Order of operations, which matters:

1. Start from the strategy's raw weights `w_raw` (already `signal / sigma_i` in the plugins).
2. Estimate portfolio volatility under a **constant-correlation** covariance:

       Sigma = D C D,  D = diag(sigma_i),  C = (1 - rho) I + rho * 1 1'
       sigma_portfolio = sqrt(w' Sigma w)

   Computed in closed form — never materialise an n x n matrix:

       w' Sigma w = (1 - rho) * sum_i (w_i sigma_i)^2  +  rho * (sum_i w_i sigma_i)^2

3. Scale the WHOLE book: `w = w_raw * (sigma_target_ann / sigma_portfolio_ann)`.
4. THEN apply `max_weight` clip and the gross cap, and THEN capacity water-filling as today.
5. Report whether step 4 broke the target.

### B. `rho` is measured, not chosen

Rule 8. `features/market.py:337 median_pairwise_correlation` already exists and is causal-capable
— use it. `rho` is the median pairwise correlation of session-bounded log returns over a
strictly-prior trailing window, recomputed per session, never full-sample. Record the derivation
next to the value. If the measured `rho` is not finite (too few names, first session), the sizer
falls back to `rho = 0` (independence), which is the conservative direction: it under-estimates
portfolio vol and therefore under-levers rather than over-levers. State that reasoning in code.

### C. `sigma_i` is causal and comes from the engine

The engine computes a per-symbol volatility at the decision row and passes it in. Use a causal
EWMA of session-bounded 1-bar log returns. Annualize with `sqrt(sessions_per_year *
bars_per_session)` where `bars_per_session` is derived from `np.diff(panel.day_offsets)`
(use the median over the trailing window, not a constant), and `sessions_per_year` is the
existing 252 convention.

Names with non-finite `sigma_i` get weight zero, exactly as non-finite prices do today.

### D. Diagnostics, not silence

`SizingResult` gains:

    sigma_portfolio_ann: float     # BEFORE the clips
    vol_scale_applied: float       # sigma_target_ann / sigma_portfolio_ann
    vol_target_achieved: float     # realised ex-ante portfolio vol AFTER clips
    clip_binding: bool             # True if max_weight or gross moved the book off target

A vol target that the gross cap silently overrides is the original defect wearing a new name.
`clip_binding` must be surfaced in the backtest summary, not buried.

### E. Clean up the lie

Delete `target_vol_ann` from the params and meta dict of every plugin that does not use it
(`volume_breakout`, `vwap_reversion`). Plugins that opt into `VolTargetSizer` keep it and it is
now load-bearing. `carver_trend`'s per-name scaling is replaced by the portfolio-level one, and
its hardcoded 375 is removed with it.

`GrossNotionalSizer` stays and stays the default. This spec adds an option; it does not silently
change every existing backtest.

## Required tests

1. **Independent names.** With `rho = 0` and equal `sigma_i`, `sigma_portfolio` equals
   `sigma * sqrt(sum w_i^2)`; assert the closed form against an explicit `w' Sigma w` computed
   with a materialised matrix on a small case.
2. **Perfect correlation.** With `rho = 1`, `sigma_portfolio` equals `sum_i |w_i sigma_i|` for a
   same-signed book. Assert against the materialised matrix.
3. **Closed form equals the matrix form** on random `w`, `sigma`, `rho` over many draws — this
   is the load-bearing algebraic claim of the spec.
4. **The target is actually hit** when no clip binds: realised ex-ante vol equals
   `target_vol_ann` to numerical tolerance, and `clip_binding is False`.
5. **The target is NOT hit when a clip binds**, and `clip_binding is True`. A silent miss is the
   bug this spec exists to remove.
6. **Doubling every `sigma_i` halves every weight** (scale invariance of the target).
7. **Non-finite `sigma_i`** zeroes that name and does not poison the portfolio estimate.
8. **`rho` fallback.** Fewer than the minimum names, or a non-finite measured correlation, falls
   back to `rho = 0`; assert the resulting book is smaller (under-levered), never larger.
9. **No 375.** A panel with a 60-bar session and a 105-bar session annualizes using the derived
   bars-per-session; assert the result differs from what a hardcoded 375 would give.
10. **`GrossNotionalSizer` is untouched:** an existing backtest using it is bit-identical.

## Constraints

- Vectorized; no n x n covariance matrix is ever allocated in the production path.
- float64 throughout the risk arithmetic (rule 3).
- No hand-chosen constants (rule 8): `rho` is measured; `target_vol_ann` is a declared user
  input, not a threshold; the EWMA halflife must be justified by a measured criterion or raised
  to the lead as a spec gap rather than picked.

---

# AMENDMENT 2026-08-19 — six defects found by a test author before implementation

A test author writing an independent suite from this spec found six defects, one of them a
mathematical error in the spec's own reasoning. All are accepted. Where this amendment conflicts
with the text above, **the amendment wins.**

## 1. THE `rho` FALLBACK REASONING WAS BACKWARDS — this is the important one

Section B said: fall back to `rho = 0`, which "is the conservative direction: it under-estimates
portfolio vol and therefore under-levers rather than over-levers."

**The second half does not follow from the first, and the conclusion is wrong.** The book is
scaled by `sigma_target / sigma_portfolio`. Under-estimating `sigma_portfolio` makes that ratio
LARGER, which makes the book BIGGER. Under-estimating risk over-levers. It cannot do anything
else.

And `rho = 0` does under-estimate, for exactly the book we care about. With `a_i = w_i sigma_i`:

    var(rho) = (1 - rho) * sum_i a_i^2  +  rho * (sum_i a_i)^2

For a same-signed (long-only) book the cross terms are positive, so `(sum a_i)^2 >= sum a_i^2`
and `var` is increasing in `rho`. Measured over 2,000 random same-signed books with true
`rho` drawn from [0.05, 0.9]: `rho = 0` under-estimated the true variance in **2000 of 2000**
cases. The tilt candidate — the one construction in this repo that beats the index — is
**long-only**. This spec would have over-levered it.

### The fix

`var(rho)` is **linear in `rho`**. Therefore its maximum over `rho` in [0, 1] is at an endpoint:

    var_conservative = max( sum_i a_i^2 ,  (sum_i a_i)^2 )

Use that whenever the measured `rho` is unavailable or non-finite. It is conservative for ANY
sign structure — same-signed books are bounded by the `rho = 1` endpoint, hedged long/short books
by the `rho = 0` endpoint — so it needs no case analysis and cannot be got backwards. When a
finite measured `rho` IS available, use it directly; the fallback is only for the degenerate case.

Worked example, `w = [0.2, 0.3, -0.4, 0.1]`, `sigma = [0.2, 0.5, 0.3, 0.9]`:

    rho=0.00 var=0.046600   rho=0.50 var=0.036100   rho=1.00 var=0.025600
    max(S2, S1^2) = 0.046600   <- the rho=0 endpoint, because this book is mixed-sign

Replace required test 8 with: **the fallback never produces a LARGER book than the true-`rho`
sizing would, for both a same-signed and a mixed-sign book.** Assert the direction, not the
mechanism. The old test 8 asserted the wrong direction and would have locked the defect in.

## 2. `to_shares` ALWAYS returns `SizingResult`

The spec never said whether `VolTargetSizer.to_shares` mirrors `GrossNotionalSizer`'s
`np.ndarray | SizingResult` split. It must not: if diagnostics vanish whenever
`bar_traded_value` is None, then `clip_binding` is invisible on the default path, which
reproduces the exact silent-miss this spec exists to remove. `VolTargetSizer.to_shares` returns
`SizingResult` unconditionally. `GrossNotionalSizer` keeps its existing split unchanged.

## 3. Constructor

    @dataclass(frozen=True)
    class VolTargetSizer:
        target_vol_ann: float          # required, load-bearing, no default
        max_weight: float = 0.10
        gross: float = 1.0             # a CAP, not a target -- see below
        min_trade_notional: float = 0.0
        lot_size: int | None = None

`gross` remains a cap. Under vol targeting it no longer *sets* exposure — the vol target does —
so a binding `gross` means the requested risk was unreachable, and that is precisely what
`clip_binding` reports.

## 4. `clip_binding` includes capacity

`clip_binding` is True when `max_weight`, the `gross` cap, **or capacity water-filling** moved the
book off the vol target. Capacity is the most common binder in this repo (73.7% of desired
notional went unfilled in the reference backtest), so excluding it would make the flag read
"target achieved" in the very situation where it was not.

## 5. Annualization is NOT the sizer's job — required test 9 moves

The author correctly reported that test 9 is unassertable through the API in section A: `sigma`
arrives at `to_shares` already annualized, and the sizer never sees a `Panel` or `day_offsets`.

Annualization moves to a named, separately testable helper, which is where the rule-5 assertion
belongs:

    def annualization_factor(day_offsets: np.ndarray, *, window: int, sessions_per_year: int = 252) -> float

It derives bars-per-session from `np.diff(day_offsets)` over the trailing `window` sessions —
median, not mean, so one 60-bar Muhurat session does not drag the estimate — and returns
`sqrt(sessions_per_year * bars_per_session)`. Required test 9 now targets this helper: a panel
containing a 60-bar and a 105-bar session must produce a factor differing from `sqrt(252 * 375)`.

## 6. The EWMA halflife is an open rule-8 gap, and is now a test item

Section C says "causal EWMA" without specifying a halflife, and the constraints say it "must be
justified by a measured criterion or raised to the lead as a spec gap rather than picked." The
author raised it. **Raised and accepted: it is a gap.**

Resolution: the halflife is a DECLARED INPUT to the sizer's caller, not a constant inside it, and
the run's provenance records it (see `specs/run_provenance.md`). Phase E measures which halflife
best predicts realised forward volatility on this data and that measurement sets the default;
until then there is no default and the caller must pass one. A sizer that silently picks a
halflife is a hand-chosen constant wearing a parameter's clothes.

New required test 11: constructing the vol path without an explicit halflife RAISES. It does not
quietly default.

## Items considered and NOT changed

- "`min_names` is invisible to the sizer's contract" — correct, and intended. `min_names` governs
  the correlation MEASUREMENT, which happens in the engine and is tested there.
- "the no-`n x n`-matrix constraint is white-box and not black-box testable" — correct. It stays a
  constraint on the implementation, enforced at review, not by a test. Required test 3 (closed
  form equals the materialised form) already pins the arithmetic; the constraint only pins how it
  is computed.

---

# AMENDMENT 2 — 2026-08-19, later. Two defects in AMENDMENT 1.

Found by the test author applying amendment 1. Both are mine. Amendment 2 wins over amendment 1
wins over the body.

## 1. The constructor in amendment 1 section 3 was WRONG — it dropped two fields

I listed `VolTargetSizer` with `min_trade_notional: float = 0.0` and no `whole_shares`. The
sizer it must mirror, `GrossNotionalSizer` (`backtest/portfolio.py`), actually declares:

    gross: float = 1.0
    max_weight: float = 0.10
    min_trade_notional: float = 25_000.0      <- NOT 0.0
    whole_shares: bool = True                 <- I omitted this entirely

Consequences of my error, both real:
- Tests 4-7 and 10 already construct `VolTargetSizer(..., whole_shares=False)`, which is the
  correct thing to do — a whole-share floor quantises the weights and would mask the vol-target
  arithmetic those tests exist to check. Against my constructor they would have raised
  `TypeError` at run time and never exercised their assertions. A test that raises before
  reaching its assertion is a test that passes review and checks nothing.
- A `min_trade_notional` default of 0.0 instead of 25,000 would have silently changed dust
  behaviour between the two sizers, so a book switched from one to the other would trade
  differently for a reason nobody declared.

**Corrected constructor:**

    @dataclass(frozen=True)
    class VolTargetSizer:
        target_vol_ann: float                     # required, load-bearing, NO default
        gross: float = 1.0                        # a CAP, not a target
        max_weight: float = 0.10
        min_trade_notional: float = 25_000.0      # matches GrossNotionalSizer exactly
        whole_shares: bool = True                 # matches GrossNotionalSizer exactly

Rule for the implementer: every field `VolTargetSizer` shares with `GrossNotionalSizer` carries
the SAME default. A divergent default is an undeclared behaviour change.

## 2. Amendment 1 item 6 never named the function it required a test for

It said "constructing the vol path without an explicit EWMA halflife must RAISE" without naming
the vol path. The test author had to guess, and guessed `causal_ewma_sigma_ann` in
`nifty_quant.backtest.portfolio`. Reasonable, but the wrong home: `backtest/portfolio.py` sizes
books, it does not estimate volatility. There is currently **no EWMA estimator anywhere in
`src/`** (grep for `ewma`/`halflife` returns nothing), so this is a genuinely new function.

**Named, and placed with the estimator family** in `features/core.py`, beside
`parkinson_volatility` and `rolling_std`:

    def ewma_volatility_ann(
        close: np.ndarray,
        day_offsets: np.ndarray,
        *,
        halflife: float,                 # REQUIRED, no default (rule 8 -- see amendment 1 item 6)
        sessions_per_year: int = 252,
    ) -> np.ndarray

Session-bounded causal EWMA of 1-bar log returns, annualized internally via:

    def annualization_factor(
        day_offsets: np.ndarray, *, window: int, sessions_per_year: int = 252
    ) -> float

also in `features/core.py`. Both are Phase-D deliverables (the vol-estimator family) pulled
forward because Phase A4 needs them; Phase D adds Garman-Klass and the rest beside them rather
than relocating these.

Required test 11 targets `ewma_volatility_ann`: calling it without `halflife` must raise, and the
parameter must carry no default. Required test 9 targets `annualization_factor`, unchanged
otherwise.

## Note on how both of these were found

Amendment 1 was itself the product of a test author falsifying the spec body. Amendment 2 is a
test author falsifying amendment 1. The spec has now been wrong twice and corrected twice, both
times before a line of implementation existed, both times by someone whose only job was to read
it adversarially. That is the process working, not the process failing.

---

# AMENDMENT 3 — 2026-08-19, later still. A defect in AMENDMENT 2.

Found by the same test author. Accepted. Amendment 3 wins over 2 wins over 1 wins over the body.

## The defect

Amendment 2 named:

    ewma_volatility_ann(close, day_offsets, *, halflife, sessions_per_year=252)
    annualization_factor(day_offsets, *, window, sessions_per_year=252)

`ewma_volatility_ann` has no `window`, but it must call `annualization_factor`, whose `window` has
no default. So the implementer's only options were to **hardcode a `window` internally** — a new
hand-chosen constant, which is precisely the rule-8 violation amendment 1 item 6 existed to
remove, merely relocated from `halflife` to `window` — or to derive it from something unstated.

That is the third time this spec has smuggled in a constant, and the second time in a correction.

## The fix: delete `window` entirely

`window` was never a research choice. It exists only to estimate **bars per session**, and bars
per session is a **calendar property, not a price property**. Estimating it over the whole panel
introduces no price lookahead: knowing that some future session was a 60-bar Muhurat session
tells you nothing about any return. The repo's causal machinery exists to stop *prices* leaking
backwards, and applying it reflexively to the trading calendar would be cargo-culting it.

**This reasoning is the justification, and it is recorded here deliberately so the question is
decided rather than merely unasked.** If a future reader disagrees, the thing to challenge is the
claim "bars per session carries no return information", not the absence of a window.

Corrected signature:

    def annualization_factor(day_offsets: np.ndarray, *, sessions_per_year: int = 252) -> float
        """sqrt(sessions_per_year * median_bars_per_session).

        Bars per session is the MEDIAN of np.diff(day_offsets) over every session present.
        Median, not mean, so a 60-bar Muhurat session or a 105-bar shortened session does not
        drag the estimate. No window parameter: see AMENDMENT 3 for why the full panel is
        admissible here and why no constant is needed.
        """

    def ewma_volatility_ann(
        close, day_offsets, *, halflife: float, sessions_per_year: int = 252
    ) -> np.ndarray

`halflife` stays required with no default — it IS a research choice, it governs how price history
is weighted, and Phase E measures which value best predicts realised forward volatility.
`sessions_per_year = 252` stays a default because it is a stated convention already used
throughout the repo, not an estimate.

Required test 9 is unchanged in intent and simplifies: `day_offsets = [0, 60, 165]` gives sessions
of 60 and 105 bars, median 82.5, so the factor is `sqrt(252 * 82.5)` and differs from
`sqrt(252 * 375)`.

## Standing rule for the rest of this program

Every time a required parameter is removed from one function, check whether it reappeared as a
literal inside another. All three defects in this spec have been the same shape: a number that
had to come from somewhere, with the spec silent about where. Rule 8 is not satisfied by moving
the number — only by measuring it or by proving it is not a choice.

---

# AMENDMENT 4 — 2026-08-19. Three gaps found by the SECOND test author.

The two authors found disjoint defect sets, which is the whole argument for writing two suites.
Amendment 4 wins over 3 wins over 2 wins over 1 wins over the body.

## 1. `gross` is a MULTIPLIER, not a cap — amendment 1 invented behaviour

Amendment 1 item 3 said "`gross` remains a cap. Under vol targeting it no longer *sets* exposure".
**That is not what `gross` does in the sibling class.** Verified in source:

    backtest/portfolio.py:148   target_notional = clipped_weight * self.gross * capital

It is a bare multiplier. There is no rescale-if-exceeded step anywhere in
`GrossNotionalSizer.to_shares`. The "scale down only if `abs_sum > gross`" behaviour the spec's
background section describes lives in the PLUGINS, not in the sizer. Amendment 2 established the
rule that every field shared with `GrossNotionalSizer` carries the same default; the same must
hold for its SEMANTICS, and amendment 1 broke that.

**Resolution — split the two ideas apart instead of overloading one name:**

    gross: float = 1.0            # multiplier, IDENTICAL semantics to GrossNotionalSizer
    max_gross: float | None = None  # NEW, optional exposure ceiling. None = no ceiling.

When `max_gross` is set and `sum(|w|)` exceeds it after vol scaling, the whole book is rescaled
**proportionally** (preserving relative weights) so that `sum(|w|) == max_gross`, and that
rescale sets `clip_binding = True`. A per-name clip would distort the cross-section; a
proportional rescale is the only operation that caps exposure without changing what the strategy
said about relative attractiveness.

A vol target with no ceiling can demand arbitrary leverage, so `max_gross` is how a caller bounds
it — explicitly, under its own name, rather than by quietly redefining an existing field.

## 2. `clip_binding` covers STRUCTURAL binders only

Author B asked whether `min_trade_notional` dust filtering and `whole_shares` rounding also set
the flag. They must not.

    clip_binding = True  iff  max_weight, max_gross, or capacity moved the book off target.
    Quantisation -- whole_shares rounding and min_trade_notional dust -- does NOT set it.

Reason: whole-share rounding perturbs the realised vol on essentially every call, so including it
would make the flag permanently True, and **a flag that is always True carries no information.**
The effect stays visible regardless, because `vol_target_achieved` reports the realised ex-ante
portfolio vol unconditionally. `clip_binding` answers "did a constraint fight the target";
`vol_target_achieved` answers "what did we actually get". Both are needed and they are not the
same question.

## 3. `SizingResult`'s new fields carry defaults

Author B correctly noted that `SizingResult` (`portfolio.py:65-77`) is already constructed by
`GrossNotionalSizer` on its capacity path with exactly two fields, so adding four required fields
would force edits there — contradicting section E's promise that `GrossNotionalSizer` stays
untouched.

    sigma_portfolio_ann: float = float("nan")
    vol_scale_applied:   float = float("nan")
    vol_target_achieved: float = float("nan")
    clip_binding:        bool  = False

**NaN means "this result did not come from a vol-targeted sizer"** — it is not a missing value to
be filled in later, and any consumer must treat NaN as "not applicable" rather than coercing it to
zero. Document that at the field. `GrossNotionalSizer`'s construction site is not touched.

## 4. Names and module, restated because author B was dispatched before amendments 2 and 3

Author B guessed `annualization_factor` and `causal_ewma_vol_ann` in
`nifty_quant.backtest.portfolio`, and flagged both as guesses. Both guesses are superseded:

    from nifty_quant.features.core import annualization_factor, ewma_volatility_ann

    annualization_factor(day_offsets, *, sessions_per_year=252) -> float
    ewma_volatility_ann(close, day_offsets, *, halflife, sessions_per_year=252) -> np.ndarray

No `window` parameter on either — see amendment 3 for why it was deleted rather than defaulted.
`ewma_volatility_ann` takes `close` PRICES, not pre-computed returns.

## What the two-author rule bought here

Author A found six defects, then two more in the correction, then one in the correction to the
correction. Author B, working from the same text without seeing A's file, found three that A did
not — including the `gross` semantics error, which A's tests never exercised. **Neither author
alone would have produced a correct spec.** The disjointness is the point, and it is the concrete
argument for `CLAUDE.md` rule 1 over letting one author write both the tests and the contract.

## Addendum to amendment 4 — degenerate panels in `annualization_factor`

Raised by author A while applying amendment 3. `np.diff(day_offsets)` is empty when `day_offsets`
has fewer than two entries, and `np.median` of an empty array is NaN, so the factor would silently
become NaN and poison every downstream volatility.

Note the convention first, because it makes the real case narrower than it looks: for N sessions
`day_offsets` carries N+1 entries (N starts plus the final end), as the engine's
`panel.day_offsets[day_idx + 1] - 1` indexing shows. So a ONE-session panel gives
`diff` of length 1 and is fine. The degenerate case is a ZERO-session panel.

Required: `annualization_factor` RAISES `ValueError` on a panel with no sessions. It does not
return NaN. A NaN annualization factor propagates silently into every sigma, every weight and
every share count, and would surface as an empty book rather than as an error — the single worst
failure shape available. Add it as a required test.

This is the same principle as `ewma_volatility_ann` raising on a missing `halflife`: at the panel
boundary, fail loudly rather than produce a number nobody can interpret.

---

# AMENDMENT 5 — 2026-08-19. The pipeline ORDER was never stated. It is now.

Author B, defect 20. Accepted, and it is the most consequential of the four rounds because it
silently defeats a guarantee rather than merely under-specifying one.

## The defect

Amendment 4 introduced `max_gross` as an exposure ceiling with a proportional rescale, and said it
applies "after vol scaling". That does not say where it sits relative to `max_weight`'s clip or
`gross`'s multiply. If a caller sets both `gross != 1.0` and `max_gross`, an implementation that
rescales to the ceiling BEFORE applying the `gross` multiply produces a final book exceeding
`max_gross` by a factor of `gross` — **the ceiling silently fails to bound the thing it exists to
bound.**

The root cause is not `max_gross`. It is that this spec has described a pipeline in prose across
five documents and never once written the steps in order. Every step-ordering question so far has
been answered by implication.

## The fix: the complete ordered pipeline, normative

`VolTargetSizer.to_shares` executes exactly these steps, in exactly this order:

    1. w_raw            <- caller's weights (already signal_i / sigma_i)
    2. w_vol            <- w_raw * (sigma_target_ann / sigma_portfolio_ann)
                           sigma_portfolio from the closed form; see amendment 1 item 1 for the
                           conservative fallback when rho is unavailable.
    3. w_clip           <- clip(w_vol, -max_weight, +max_weight)          [per-name]
    4. notional         <- w_clip * gross * capital                        [gross is a MULTIPLIER,
                           identical to GrossNotionalSizer:148]
    5. notional         <- proportional rescale IF max_gross is not None
                           and sum(|notional|) > max_gross * capital,
                           so that sum(|notional|) == max_gross * capital  [whole-book]
    6. notional         <- capacity cap + water-filling                    [per-name, existing]
    7. shares           <- whole_shares rounding, then min_trade_notional dust filter

**Step 5 comes AFTER step 4.** `max_gross` is a ceiling on the FINAL gross exposure as a fraction
of capital — `sum(|notional|) / capital` — not on the pre-multiply weight sum. Defined that way it
bounds what it claims to bound for any value of `gross`, and the ordering question cannot recur.

Steps 6 and 7 can only ever REDUCE exposure, so neither can breach the step-5 ceiling. That is why
they sit after it, and it is the reason the order is safe rather than merely conventional.

`clip_binding` is True if step 3, step 5, or step 6 moved the book. Steps 4 and 7 never set it
(step 4 is a declared multiplier, step 7 is quantisation — amendment 4 item 2).

## Required test, added

With `gross = 2.0` and `max_gross = 0.5`, the realised `sum(|notional|) / capital` must equal
**0.5**, not 1.0. This is the exact case author B's suite could not cover — it used `gross = 1.0`
deliberately so its assertions could not be contaminated by whichever order got implemented, which
was the right call for an unspecified contract and is now testable.

## A second good catch from the same pass, recorded because it generalises

While re-pointing, author B found that its own `annualization_factor` fixture — a 5-session panel
verified against a 3-session trailing slice — collapses to a whole-panel median of exactly 375
once `window` is deleted. The test would still have PASSED while asserting nothing: its
"differs from sqrt(252 * 375)" check would have compared 375 against 375 and been vacuously
satisfied only by floating-point luck, or failed for the right reason by accident.

**Generalise it: when a spec change alters what a fixture computes, re-derive the fixture, do not
just re-point the call.** A fixture tuned to the old contract can turn a real assertion into a
vacuous one without any visible failure. Same family as amendment 2's finding that a test raising
`TypeError` before its assertion is a test that checks nothing.

---

# AMENDMENT 6 — 2026-08-19. `vol_target_achieved` was reporting success for a 100x-levered book.

The implementer flagged this as the one judgment call the spec text did not settle, and
implemented what the contract tests demanded. It followed the tests correctly. **The tests were
wrong, and so was the spec for not settling it.** Measured on the delivered implementation:

    gross=1.0    reported vol_target_achieved = 0.1500   book = 1.1x capital    TRUE vol = 0.1500
    gross=100.0  reported vol_target_achieved = 0.1500   book = 105.4x capital  TRUE vol = 15.0000

A field named `vol_target_achieved` reading "0.15, on target" for a book running at **fifteen
times its target risk** is precisely the class of defect this entire spec was written to remove:
**a diagnostic that reports success while the book is somewhere else.** It is the original
`target_vol_ann`-in-a-meta-dict defect wearing a better name.

## Resolution

`vol_target_achieved` reports the **TRUE ex-ante annualised volatility of the ACTUAL book**, in
capital terms:

    w_final = shares * prices / capital          # NOT / (capital * gross)
    vol_target_achieved = sqrt(w_final' Sigma w_final)

No information is lost by this change: `vol_scale_applied` already reports the targeting
arithmetic (`sigma_target / sigma_portfolio`) separately, so a caller who wants "did steps 1-3 do
their job" reads that field. The two fields now answer two different questions, which is the same
principle amendment 4 item 2 applied to `clip_binding` versus `vol_target_achieved`.

At `gross = 1.0` — the default, and every current caller — nothing changes.

`clip_binding` is NOT affected and stays `False` here. `gross` is a declared leverage dial, not a
constraint that fought the target (amendment 5, step 4 never sets the flag). The combination
`vol_target_achieved = 15.0` with `clip_binding = False` is the correct and informative reading:
nothing constrained us, and your gross dial put the book at 100x the requested risk.

## Contract test correction — LEAD-AUTHORED, stated as such

`tests/test_vol_target_b.py::test_vol_target_achieved_when_nothing_binds` uses `gross=100.0` and
asserts `vol_target_achieved == target_vol_ann`. Under this amendment it must assert
`vol_target_achieved == target_vol_ann * gross`, or set `gross=1.0` and add a separate
leverage-visibility test. Per rule 1 the implementer may never edit a test; this is the lead
adjudicating a contract, and it is recorded here as such.

Suite A never exercised `gross != 1.0` on this field, so it did not encode the wrong contract —
another instance of the two suites failing differently rather than identically.

## Why this one is worth dwelling on

Six amendments in, the defect that survived longest was **not** a missing number or an unstated
module. It was a plausible-sounding semantic choice that both the spec author and one test author
got wrong in the same direction, and that only surfaced when a third party ran the code and
printed the actual book size. **Specs and tests can agree with each other and still both be wrong;
only execution settles it.** Keep that in mind for Phase E and F, where the equivalent mistake
produces a number nobody can check by inspection.

## Section E status — 2026-08-19. Two of three done; `carver_trend` DEFERRED to Phase D, deliberately.

DONE: dead `target_vol_ann` deleted from `volume_breakout` and `vwap_reversion` (field, meta-dict
write and shipped YAML), each with a mutation-verified dead-code-removal test.

NOT DONE, and this is a lead decision rather than an omission: `carver_trend.py`'s
`_ANNUALIZATION_FACTOR = sqrt(252 * 375)`. The implementer correctly refused to force it and
escalated. The obstacle is real: `_ANNUALIZATION_FACTOR` is consumed inside `on_decision`, and
`on_decision`'s `MarketView` — the real `ArrayMarketView` and both test stubs — carries no
`day_offsets`. Only `precompute(panel)` has them. So there are two routes, both cross-cutting:

    (a) add `day_offsets` to the MarketView protocol in `strategy/base.py`
        -> ripples into EVERY strategy's view stub across the suite
    (b) annualize upstream in `precompute`, baking it into the "vol" signal
        -> changes that signal's contract from raw-per-bar to already-annualized, breaking
           ~10 assertions that hand-construct raw "vol" and call on_decision directly

**Decision: defer to Phase D, and take route (a) when we get there.**

Reasoning, in order:
1. **Measured impact is small and conservative.** The factor overstates annualised vol by
   sqrt(375/105) = 1.89x on a shortened session and sqrt(375/60) = 2.50x on Muhurat. Overstated
   vol makes `target_vol_ann / vol` scaling UNDER-size, so the error is in the safe direction, and
   it touches ~105 of ~1,880 sessions.
2. **`carver_trend` is not in use.** Raw daily Sharpe +3.678, net -12.638; it is the repo's
   canonical "great t-stat, no tradable edge" example. Rebuilding the suite around a net-negative
   strategy now buys nothing.
3. **Route (a) is the right fix and it is bigger than this bug.** Rule 5 forbids assuming a fixed
   session stride, yet `on_decision` currently CANNOT know where sessions begin or end. That is a
   latent rule-5 hazard in EVERY plugin, not just this one — the hardcoded 375 is the symptom that
   happens to be visible. Fixing it as a one-line swap would leave the class of defect in place.
4. Phase D already touches annualization (`annualization_factor`, the vol-estimator family), so
   the work lands there with its natural neighbours rather than as an isolated ripple now.

Recorded so this is a decision with reasons, not a gap. If Phase D slips, route (a) still owes
the repo a fix.
