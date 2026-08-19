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
