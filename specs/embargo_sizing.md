# Spec: embargo sized for the whole dependence horizon

Phase B track B2.

## Why this exists

Two independent mechanisms disagree, and the stricter one is switched off in production.

**`WalkForwardSplitter`** (`research/splits.py:61-154`) defaults to `embargo_days: int = 5` and
validates it only against a feature lookback:

    def split(self, trading_dates, *, max_lookback_days: int = 0):
        if max_lookback_days > 0 and self.embargo_days < max_lookback_days:
            raise EmbargoTooShortError(...)          # splits.py:72-77

`max_lookback_days` **defaults to 0 and is never passed by any caller**. `cli.py:851-852`
constructs the splitter and calls `splitter.split(trading_dates)` with no lookback at all, so
`EmbargoTooShortError` is unreachable from `nq walkforward`. The only callers that pass it are
tests. A guard that cannot fire in production is decoration.

**`PurgedKFold` / `CombinatorialPurgedCV`** (`research/cv.py:37-136`, `:237-340`) does model the
label horizon, but as *purging* — window `[test_start - label_horizon, test_end)` at `:112-119` —
with embargo as a separate one-sided `embargo_frac` (default 0.01 of rows) applied after the test
block at `:121-128`. Holding period is not represented at all. `PurgedKFold` is not imported by
`cli.py`.

Neither mechanism accounts for the holding period. A strategy holding 30 bars, or the tilt's
smoothed book with a ~10-session effective memory at a=0.10, leaves training labels dependent on
test observations well past any feature lookback.

## Required behaviour

### A. One sizing rule, applied by both mechanisms

    required_embargo >= feature_lookback + label_horizon + holding_period + execution_horizon

where:
- `feature_lookback` — the longest window any feature in the run reads (e.g. 390 bars of Hurst,
  20 sessions of ADV).
- `label_horizon` — the forward-return horizon of the label.
- `holding_period` — the expected position lifetime. For a smoothed book with weight parameter
  `a`, this is the **effective memory**, not one session: for `w_t = (1-a) w_{t-1} + a * target_t`
  the memory is `1/a` sessions to a factor. Compute it; do not assume one day. This is the term
  the current code omits entirely and it is the largest one for the tilt candidate.
- `execution_horizon` — the decision-to-fill lag, in sessions (usually 0 for one-bar lag, but it
  is not always).

All four terms are converted to SESSIONS via `panel.day_offsets` — never via a bars-per-day
constant (rule 5).

### B. The guard must be armable and armed

- `split()` takes the four components, not one pre-summed number, so the error message can say
  which term dominated.
- `cli.py` passes them. `EmbargoTooShortError` must be reachable from `nq walkforward` and the
  test suite must demonstrate that it is.
- The components are recorded in the trial provenance (Phase B3) so a published walk-forward
  result carries the embargo that produced it.

### C. Reconcile the two mechanisms

`PurgedKFold`'s purge covers `label_horizon` on the left of the test block; its `embargo_frac`
covers a fraction of rows on the right. Under this spec both sides are expressed in the same
units and derived from the same four components. `embargo_frac` as a fraction of total rows is
replaced by an absolute session count, because a fraction of the sample is not a property of the
strategy and shrinks as the sample grows — precisely backwards.

Where the two mechanisms genuinely differ (K-fold purges on both sides, walk-forward embargoes
only forward), document the difference at each site rather than forcing a false unification.

## Required tests

1. `EmbargoTooShortError` fires from `nq walkforward` when the components exceed `embargo_days`.
   This is the load-bearing test: it must fail against HEAD.
2. The error message names the dominant term.
3. A smoothed book with `a=0.10` produces a holding-period term of order 10 sessions, not 1.
   Assert against the effective-memory formula, not a hardcoded number.
4. Component-to-session conversion uses `day_offsets` — a panel with a 60-bar and a 105-bar
   session converts correctly and differs from a 375-bar assumption.
5. Purged K-fold and walk-forward, given identical components, produce embargo regions that are
   consistent in size (the K-fold's two-sided, the walk-forward's one-sided).
6. `embargo_frac` no longer shrinks the embargo as the sample grows: doubling the sample leaves
   the absolute embargo unchanged.
7. Train and test index sets are disjoint and separated by at least the required embargo in every
   split, over many random configurations.
8. Zero-lookback, zero-horizon, zero-holding-period degenerates to the current behaviour.

## Constraints

- No hand-chosen constants (rule 8). Every term is derived from a declared strategy or feature
  property. If a term cannot be derived for some strategy, `split()` must RAISE rather than
  default it to zero — a silently-zero embargo term is how this defect got here.

---

# AMENDMENT 1 — 2026-08-20. Defects found by a test author before implementation.

## 1. MY STATED MECHANISM FOR `embargo_frac` WAS BACKWARDS

Section C says `embargo_frac` "is not a property of the strategy and **shrinks as the sample
grows** — precisely backwards." The second clause is itself backwards. Measured:

    frac = 0.01,  embargo = ceil(frac * n_rows)
    n_rows     1,000  ->     10 rows
    n_rows    10,000  ->    100 rows
    n_rows 1,000,000  -> 10,000 rows

The ABSOLUTE embargo **GROWS** with the sample. It does not shrink.

**The substantive objection survives intact, and it is the reason the change still stands:** a
fraction of SAMPLE SIZE is not a property of the STRATEGY. The dependence horizon between train
and test is set by the feature lookback, the label horizon, the holding period and the execution
lag — none of which changes when you add three more years of data. Tying the embargo to `n_rows`
leaves it untethered from the thing it exists to bound, **in either direction**: too small on a
short sample, arbitrarily large on a long one. Growing without reason is no better than shrinking
without reason.

Required test 6 is UNCHANGED and was already correct: doubling the sample must leave the absolute
embargo unchanged, because the embargo is a strategy property. Only my justification was wrong.

## 2. Boundary, rounding and scope items — accepted, resolved here

- **`>=` vs `<`.** `required_embargo` is a MINIMUM: the check fails when the configured embargo is
  strictly LESS than the requirement. Equality passes.
- **Rounding.** Components sum as floats and are converted to sessions with `math.ceil` at the
  END, once — never per-component, which would over-count by up to one session per term.
- **`holding_period` for non-EMA strategies.** The `1/a` effective-memory formula applies to
  exponential smoothing only. For a fixed k-session hold it is `k`; for a strategy that cannot
  declare one, `split()` RAISES rather than defaulting to zero. That was already the spec's rule
  and it is restated because the author was right that the formula alone does not cover the space.
- **`embargo_frac`'s fate.** DELETED, not silently ignored. A field that is read by nothing but
  still accepted in a constructor is the same class of lie as the `target_vol_ann` this program
  already removed from two plugins.
- **`PurgedKFold`'s left purge.** It covers `label_horizon` only, NOT the full
  `required_sessions()`. The purge and the embargo bound different dependencies — purging removes
  label overlap on the left, the embargo removes serial dependence on the right. Do not unify them.
- **Provenance recording** is Phase B3's job and is correctly untestable from this spec.

## 3. A FOURTH INSTANCE OF THE DEGENERATE-FIXTURE PATTERN — caught by the author itself

Author B found that its own regression guard used `holding_period=10`, which **coincidentally
equalled the old fraction-based width at `n_rows=1000`** — so the assertion "must differ from the
old behaviour" would have PASSED against unfixed code. It changed the constant to 7 and said so.

That is the fourth time in this program that a fixture value collapsed onto the very thing it was
meant to distinguish: flat prices making returns identically zero; an `annualization_factor` panel
whose median landed on exactly 375; per-row turnover `[0.0, 0.6]` where sum equals compound; and
now this. **Whenever a test asserts "X must differ from Y", compute Y and check the fixture does
not sit on it.** The author catching this in its own draft, unprompted, is the behaviour to
reinforce.
