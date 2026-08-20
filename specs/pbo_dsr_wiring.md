# Spec: make PBO and the trial count REAL

Phase C3. **This spec exists because a recorded belief in this repo is false.**

## Why this exists

`.claude/progress/` records that `pbo_cscv` "has never worked and cannot be fixed by wiring
existing files". That was true when written and is **now false**. Measured by running it: a real
3-parameter sweep produced **PBO = 0.0326** at the default `n_splits=16` on 22 days of data.
`nq sweep` DOES write `returns.parquet` and DOES record `result_path`.

Three defects stand between that and a number anyone sees.

### D1 — `build_trial_matrix` loads the WRONG artifact on a hash collision

`research/registry.py`, the per-`trial_ids` lookup selects `ORDER BY ts, rowid LIMIT 1` — the
**OLDEST** row for a `config_hash`. A sweep trial whose hash collides with an earlier walk-forward
run therefore loaded a **3-day walk-forward test slice** instead of the 22-day sweep result.

Consequences, measured: the aligned matrix collapsed from `(22, 3)` to `(3, 3)`; `pbo_cscv` raised
`ValueError` (`n_splits=16 > T=3`); and all three `deflated_sharpe` calls returned `nan` because
`T < 4`. **Silently** — `TrialMatrix.explain()` reported "kept 3; dropped 0. No trials were
dropped." It has no way to say "I loaded a stale artifact".

**Fix:** prefer the NEWEST row that actually has an artifact —
`WHERE config_hash = ? AND result_path IS NOT NULL ORDER BY ts DESC, rowid DESC LIMIT 1`.
This one change turns `(3,3)` + `ValueError` into `(22,3)` + `PBO = 0.0326`.

`explain()` must additionally report WHICH artifact path each trial resolved to, and the row's
`ts` and `split_id`. A selection that can pick the wrong row must say which row it picked.

### D2 — nothing ever calls PBO

`pbo_cscv` and `effective_n_trials` have **zero call sites** in `src/` or `scripts/`. `nq
walkforward` hardcodes `pbo = float("nan")` with a comment explaining a single-strategy run cannot
supply a trial matrix — correct for walkforward, and it is why `verdict_line` renders `PBO=nan`.

**Fix:** `nq sweep` assembles a matrix from the trials it just wrote and reports PBO. A sweep is
exactly the multi-trial object PBO was designed for; it is the natural home and it did not exist.
Guard the split count: the default 16 needs `T >= 16`, so either use `min(16, T - (T % 2))` or
REFUSE below 16 sessions with a message saying so. Do not silently produce a PBO from 3 periods.

### D3 — `effective_n_trials` is decorative, and the comment defending it is FALSE

`cli.py` sets `n_eff = float(n_trials)` — a raw COUNT — with a comment claiming the registry
"only stores summary Sharpes, not full per-trial return series, so effective_n_trials cannot be
computed honestly here." **The registry stores return series now.** The comment is stale and the
value is wrong in a direction that matters:

    three near-identical sweep trials:  honest effective count 1.055   reported 3.000
    a nine-trial exploration set:       honest value near 1            reported 9.000

`effective_n_trials` accounts for CORRELATION between trials — nine variants of one idea are not
nine independent bets. Feeding a raw count into `expected_max_sharpe` inflates `sr0`... and
therefore **UNDER-deflates** the Deflated Sharpe. Every DSR this program has printed is wrong in
the permissive direction, which is the dangerous one.

**Fix:** compute `n_eff` from an assembled matrix. Where no matrix can be assembled, report
`n_eff` as NOT AVAILABLE rather than substituting the count — a wrong number that looks right is
worse than a missing one.

## What must NOT change

- `pbo_cscv`, `effective_n_trials` and `deflated_sharpe` are all CORRECT. They returned finite,
  sensible values on real 22-day sweep data. **No metric arithmetic changes in this spec.**
- `deflated_sharpe` silently returns `nan` for `T < 4`. Leave the behaviour; DOCUMENT it at the
  function, because a `nan` DSR currently looks like a computation failure rather than "too few
  periods".
- `nq walkforward` keeps printing that PBO is unavailable for a single-strategy run. That message
  is honest and should not be replaced by a fabricated matrix.

## Required tests

1. **The collision.** Two registry rows share a `config_hash`; the older points at a 3-period
   artifact, the newer at a 22-period one. Assert the matrix has 22 periods, not 3. This is the
   load-bearing test and it must FAIL against current code.
2. `explain()` names the resolved artifact path, `ts` and `split_id` for each trial.
3. A real sweep of >= 2 params produces a FINITE PBO in `[0, 1]`, from the CLI, end to end.
4. A sweep with `T < n_splits` REFUSES with a message naming both numbers, rather than raising a
   bare `ValueError` from inside `pbo_cscv`.
5. `effective_n_trials` on three highly-correlated trials returns a value near 1, NOT 3. Assert it
   is strictly less than the raw count — that gap is the entire point of the statistic.
6. Where no matrix can be assembled, `n_eff` is reported as unavailable and is NOT the raw count.
7. Feeding the honest `n_eff` into `expected_max_sharpe` yields a LOWER `sr0` than the raw count
   does, and therefore a HIGHER DSR. Pin the direction so a regression cannot silently re-inflate.
8. `deflated_sharpe` with `T = 3` returns `nan` and the caller reports "too few periods".

## Constraints

- Rule 8: `n_splits=16` is an inherited default, not a measured threshold. Do not add new
  constants. If the guard needs a minimum `T`, derive it from `n_splits` rather than picking one.
- Rule 9: PBO is a statistic. Any test asserting on its VALUE uses a fixed seed and a
  deterministic fixture, or asserts a range rather than a point.

---

# AMENDMENT 1 — 2026-08-20. My own worked example would CRASH.

A test author found that item 7's illustration is internally inconsistent, and it is the kind of
hole that surfaces as a production traceback rather than a wrong number. Verified:

    expected_max_sharpe(n_trials=1.055) -> ValueError: n_trials must be at least 2
    expected_max_sharpe(n_trials=1.0)   -> ValueError: n_trials must be at least 2
    expected_max_sharpe(n_trials=2.0)   -> 0.1040

The spec quotes "a nine-trial exploration set: honest value near 1" AND instructs the honest
`n_eff` to be fed into `expected_max_sharpe`. **Those two instructions together crash.** And the
case is not hypothetical: nine near-identical trials measure at `effective_n_trials = 1.0002`.

## Resolution — and this is a research decision, not a clamp

`expected_max_sharpe` requires `n_trials >= 2` because the expected maximum of fewer than two
draws is not defined. That is correct mathematics and must not be worked around by clamping.

**When honest `n_eff < 2`, do NOT call `expected_max_sharpe`. Report instead:**

    sr0 = 0.0, with an explicit note that the trials are effectively a SINGLE trial
    (n_eff = <value>), so there is no multiple-testing penalty to deflate against.

That is the honest reading. If nine variants are effectively one bet, the expected maximum Sharpe
across "them" is just that one trial's own expectation — there is no selection effect to correct
for, and `sr0 = 0` says exactly that.

**The warning matters more than the number.** `n_eff` near 1 means the sweep explored ONE idea
nine ways. The DSR will look flattering because nothing was selected over anything, and the
correct response is to note that the sweep had no breadth — not to celebrate an undeflated Sharpe.
Require that message.

Required test 7 is restated: with a moderate-correlation matrix where honest `n_eff` lands
comfortably above 2, the honest value yields a LOWER `sr0` and a HIGHER DSR than the raw count.
Add a required test 9: honest `n_eff < 2` reports `sr0 = 0.0` with the single-trial warning and
does NOT raise.

## Also resolved: item 4 offered two incompatible fixes

The body says "use `min(16, T - (T % 2))` OR refuse below 16 sessions". The author correctly
flagged that as a live contradiction. **REFUSE.** Silently reducing `n_splits` changes the
statistic being computed without telling anyone, which is the same class of failure as every other
silent degradation this program has found. A refusal naming both `T` and `n_splits` is actionable;
a quietly weakened PBO is not.
