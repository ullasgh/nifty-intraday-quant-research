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

---

# AMENDMENT 2 — 2026-08-20. Adjudicating THREE direct contradictions between the two suites.

The implementer found genuine, mutually-exclusive contradictions and correctly refused to resolve
them by editing tests. **This is the dual-suite rule producing exactly what it exists to produce:
disagreements a single author would have resolved silently and wrongly.** Each is decided here.

## 1. `embargo_frac` — DELETED. Amendment 1 stands; suite B is wrong.

Suite A requires `PurgedKFold(embargo_frac=0.01)` to raise `TypeError`. Suite B keeps the kwarg
"for signature compatibility" and expects no error. Both cannot hold for the same call.

**Amendment 1 item 2 already decided this: DELETED, not silently ignored.** A field accepted by a
constructor and read by nothing is the same lie this program removed from `volume_breakout` and
`vwap_reversion`. Suite B's tests 5, 6 and 7 are updated to the deletion contract.

Consequence, foreseeable and accepted: **`tests/test_purged_cv.py` loses 10 tests to
`TypeError: unexpected keyword argument 'embargo_frac'`.** That is a spec-mandated contract change,
not an accident, and those tests get the same lead-authored adjudication treatment as Phase A's.

## 2. Suite A contradicts ITSELF, and the older test loses

`test_item6_documents_current_embargo_frac_scaling` was written to be GREEN at HEAD, documenting
the fraction-based width — and it passes `embargo_frac=`, which
`test_item6_embargo_frac_is_rejected_as_deleted_api` in the SAME FILE requires to raise. One must
fail under any implementation.

**RETIRE the documentation test.** It documented behaviour that this spec deletes; a test whose
purpose is to pin the old contract has no role once the contract changes. The deletion test is the
one that was RED at HEAD and is meant to go green — keep it.

## 3. `bars_to_sessions` walks BACKWARD from the end. Suite A is right.

Suite A's assertions (90 bars -> 1 session, 120 -> 2) are only consistent with walking backward
from the end of `day_offsets`. Suite B's (60 -> 1, 61 -> 2) only with walking forward from the
start. No implementation satisfies both.

**Decided on semantics, not on which suite is louder:** the function converts a LOOKBACK window
into sessions, and a lookback looks BACKWARD from now. Forward-from-start answers a question no
caller asks ("how many sessions do the FIRST n bars occupy"). Suite B's `test_4` is updated.

## 4. The CLI must pass the FOUR COMPONENTS, not a pre-summed number

The implementer passed a pre-summed value through the legacy `max_lookback_days` keyword, because
suite A's spy monkeypatches `split` with exactly that signature and the four-kwarg call would
`TypeError` inside the spy.

That is a reasonable local decision and it is the wrong global one: **pre-summing at the CLI
destroys the dominant-term diagnostic at exactly the boundary a user sees.** The entire reason
section B takes four components rather than one number is so the error can say WHICH term forced
the embargo. A user who hits `EmbargoTooShortError` from `nq walkforward` and is told only a total
learns nothing actionable.

**Suite A's spy is over-constraining and must be updated** to accept the four keyword arguments.
The CLI passes them separately. `max_lookback_days` remains supported for backward compatibility
but is not the path the CLI uses.

## 5. Two genuine test bugs, fixed not worked around

- Suite B `test_7`: `rng.integers(10, min(50, remaining + 1))` can produce `low >= high` when
  `remaining < 9`, which raises regardless of implementation. A real defect in the draft.
- `tests/test_cli_coverage.py::test_walkforward_split_setup_failure` monkeypatches `split` with
  `lambda self, trading_dates: ...` — no `max_lookback_days` at all. Any correct wiring breaks it.
  **This also explains the earlier "order-dependent test" mystery: it is not order-dependent, it is
  SIGNATURE-FRAGILE**, and it appeared to flake because the tree was changing under the
  measurement. One less phantom to chase.

## Amendment 2, clause 6 — BOTH splitters take the four components

Implemented asymmetrically: `WalkForwardSplitter.split()` takes the four components, while
`PurgedKFold.split()` takes only `n_rows`/`day_offsets` with a pre-summed `embargo_sessions` on the
constructor. Suite B's tests 5, 6 and 7 fail on
`TypeError: PurgedKFold.split() got an unexpected keyword argument 'feature_lookback'`.

**Suite B is right and this is an implementation gap, not a test bug.** Required test 5 — present
in BOTH suites — says the two mechanisms, *given identical components*, must produce embargo
regions consistent in size. That sentence is only expressible if both mechanisms ACCEPT the
components. Passing a pre-summed integer to one and four floats to the other makes "identical
components" untestable by construction.

It is also the same defect as amendment 2 item 4, one layer down: **pre-summing anywhere destroys
the dominant-term diagnostic.** A `PurgedKFold` that raises `EmbargoTooShortError` should be able to
say which term forced it, exactly as the walk-forward splitter can.

Required: `PurgedKFold.split()` and `CombinatorialPurgedCV.split()` accept
`feature_lookback`, `label_horizon`, `holding_period`, `execution_horizon` as keyword-only floats,
defaulting to 0.0, and derive the embargo width from them via the same
`required_embargo_sessions(...)` the walk-forward path uses. `embargo_sessions` remains as an
explicit override for a caller that genuinely wants to set the width directly.

Unchanged, and worth restating so it is not "helpfully" unified: the LEFT purge stays
`label_horizon`-only. Purging removes label overlap on the left; the embargo removes serial
dependence on the right. They bound different dependencies and must not be merged.
