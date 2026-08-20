# Spec: the holdout lock must guard a FIXED window, and must refuse

Repairs defects F10 and F11. **This is the machinery protecting the single unbiased read this
entire research program has.** It currently does not work.

## Why this exists

### F10 — the boundary is computed from whatever you hand it

`research/splits.py::HoldoutLock.holdout_range(trading_dates)` computes
`holdout_end = trading_dates[-1]` and `holdout_start = holdout_end - holdout_months`. The boundary
is therefore derived from the CALLER'S window, and `cli.walkforward` passes the current run's dates:

    full calendar (the TRUE holdout)   ->  2025-08-14 .. 2026-08-14
    a Jan-2024-only run                ->  2024-01-01 .. 2024-01-31
    a 2024-only run                    ->  2024-01-01 .. 2024-12-31

A short run **manufactures a fake holdout inside its own window** and records a read against it.
Measured consequence: `results/holdout_lock.json` shows **6 reads**, all
`walkforward split rolling_000`, all from short agent runs, while the true window
2025-08-14..2026-08-14 has **never been read**. The counter says 6; the true count is 0.

**A guard whose threshold is computed from the same input it guards cannot guard anything.** This
is the third instance of that shape in this program (F9's one-predicate-two-callers, and
`embargo_frac` defined as a fraction of the sample it bounds).

### F10b — two callers, two boundaries, two behaviours, two FILES

- `cli.walkforward` derives the boundary from the run window and only **COUNTS**; it never refuses.
- `research/tilt.py` derives it from GLOBAL index bars (`TradingCalendar.from_index_bars`) — the
  TRUE boundary — and does **REFUSE**.
- `cli.walkforward` writes `results/holdout_lock.json`. `research/tilt.py` writes a hardcoded
  `/tmp/nifty_quant_holdout_lock.json`. **Two locks, neither aware of the other.**

### F11 — the negative-cash guard is documented but was never written

`backtest/engine.py` states that FULL strictness "raises on any negative cash via
`check_cash_non_negative`", and `run_backtest` does run under FULL. Grep finds that name **only in
that comment**. The function does not exist. That is why a book ran Rs 1.26 crore into overdraft
for a simulated month without tripping anything. Same family as F6 — a guard that cannot see the
failure it names.

## Required behaviour

### A. One boundary, fixed and recorded

The holdout window is computed ONCE from the FULL trading calendar and **stored in the lock file**:

    {"holdout_start": "2025-08-14", "holdout_end": "2026-08-14", "count": N, "log": [...]}

Every caller reads the stored boundary. `holdout_range()` recomputes it ONLY when the file has
none (first-ever initialisation), and then persists it. A boundary that moves with the caller's
window is not a boundary.

If a stored boundary exists and a recomputation would disagree — because the dataset gained new
sessions — **RAISE**, naming both. A silently shifting holdout is the failure this spec exists to
remove; extending it must be a deliberate, recorded act.

### B. One lock file

`settings.RESULTS_ROOT / "holdout_lock.json"`, shared by every caller. **Delete the `/tmp` path.**
A lock in `/tmp` is per-machine, world-writable, and evaporates on reboot.

### C. Refuse by default, and count only real reads

`cli.walkforward` REFUSES when a split's test window intersects the stored holdout, unless
`--allow-holdout` is passed. Only an allowed read increments the counter. Counting without
refusing is precisely what let six spurious reads accumulate unnoticed.

`nq backtest` and `nq sweep` gain the same check. Today they do not consult the lock at all, so
either can read the holdout freely.

### D. Reconcile the existing six entries — annotate, do not delete

The six recorded reads are false positives against manufactured windows. **Mark them as such in
the log with a reason; do not reset the counter.** The audit trail should record that they
happened and why they did not count. Deleting them would erase evidence of the defect.

### E. F11 — write the guard, or delete the claim

`guards.check_cash_non_negative(cash, *, floor=0.0)`, FULL-strictness only, raising
`ContractViolation` naming the row and the balance.

**And it must not repeat F6.** `abs(x) > tol` is a MAGNITUDE check, not a validity check: include
an explicit `not np.isfinite(cash)` test beside the comparison, or a NaN cash balance will pass
exactly as a NaN discrepancy passed `check_accounting`.

**Wiring is a DECISION, not a default.** Measured: negative cash reaches -Rs 64,43,280 routinely
because `BacktestConfig.compound=False` sizes off day-one capital forever. Turning the guard on
unconditionally would make ordinary runs crash. So: implement the guard, add the field to
`BacktestResult` reporting, and leave it OPT-IN behind a config flag until the `compound` question
is settled. **If it stays unwired, DELETE the comment that claims it fires.** A comment describing
a guard that does not run is worse than silence.

## Required tests

1. The stored boundary is used, not recomputed: a run over a SHORT window sees the SAME holdout
   window as a run over the full calendar. This must fail against current code.
2. A short-window run records ZERO reads, where today it records one.
3. A run genuinely intersecting the holdout REFUSES without `--allow-holdout`, and RECORDS exactly
   one read with it.
4. `nq backtest` and `nq sweep` refuse a holdout-intersecting window.
5. Both `cli` and `research/tilt.py` resolve to the SAME lock path; the `/tmp` path appears
   nowhere.
6. A stored boundary that disagrees with a recomputation RAISES and names both.
7. The six historical entries are annotated, the count is unchanged, and the log is not truncated.
8. `check_cash_non_negative` raises on negative cash AND on NaN cash — the second is the F6 lesson.
9. The counter increments exactly once per allowed read, never per split evaluated.

## Constraints

- Never modify anything under `data/` (rule 2).
- The lock write stays atomic under its advisory file lock; do not weaken that to add fields.
- No new hand-chosen constants (rule 8). `holdout_months = 12` is an existing declared parameter,
  not a threshold to re-derive.

---

# AMENDMENT 1 — 2026-08-20. Eight defects from a test author, and the leak is ONGOING.

## 0. The count is still climbing

It was 6 when the defect was found; it is **7** now, incremented by another agent's
`nq walkforward` run mid-session. **This is not a historical artifact — it leaks continuously.**

The author found why, and it is worse than first recorded: `holdout_months = 12` combined with the
clamp-to-first-available fallback means a run SHORTER than 12 months manufactures a holdout
spanning its **ENTIRE window**. Verified:

    Jan-2024 window : 2024-01-01 .. 2024-01-31   (22 sessions)
    computed holdout: 2024-01-01 .. 2024-01-31   <- the whole thing

So **every split of a sub-12-month run intersects the "holdout" and records a read.** F10 is
maximally severe for exactly the short exploratory runs this program does most.

**INTERIM MEASURE until the fix lands: no agent runs `nq walkforward` without an explicit `--end`
inside the research window.** Recorded here and in the progress file.

## 1. `check_cash_non_negative` cannot name the row — CONTRADICTION, resolved

Section E pins the signature as `check_cash_non_negative(cash, *, floor=0.0)` — no `row`
parameter — while the prose requires the raise to "name the row and the balance". The author was
right that these cannot both hold.

**Resolved: add the row.** `check_cash_non_negative(cash, *, row=None, floor=0.0)`, and the message
names the balance always and the row when supplied. A guard that says only "cash went negative"
in a 121M-row backtest is not actionable, which is the whole reason the prose asked for it. The
caller in `run_backtest` has `t` in hand.

## 2. Items 3, 4, 5 and 6 — the APIs the spec failed to name

The author had to invent all four, and flagged each as an assumption. Pinning them now:

- **Annotating the legacy entries (item 7):**
  `HoldoutLock.annotate_legacy_reads(reason: str) -> int`, returning the number annotated.
- **The shared path (item 5):** `research/splits.py::default_holdout_lock_path() -> Path`, returning
  `settings.RESULTS_ROOT / "holdout_lock.json"`. Every caller uses it. The author's
  naming-agnostic source scan for `/tmp` should STAY as well — it is the more robust check and it
  survives a future rename.
- **The disagreement exception (item 6):** a dedicated `HoldoutBoundaryError(ValueError)`. A bare
  `ValueError` is indistinguishable from a date-parsing failure at the call site.
- **`--allow-holdout` on `backtest`/`sweep` (item 5):** yes, all three commands get the same flag,
  for the same reason `walkforward` needs it — a deliberate, recorded read must remain possible.

## 3. Item 8 — the migration path, which the spec omitted entirely

The real lock file today has only `count` and `log`. It has never had `holdout_start`/`holdout_end`.
**A first-run upgrade path is required and must be tested:** on encountering a file with no stored
boundary, compute it from the FULL calendar, persist it, and leave `count` and `log` untouched.

The author's fixtures all use the post-fix shape, so nothing currently exercises the upgrade. Add
it — a migration that has never been run is a migration that does not work.

## 4. Accepted without change

The author's decision to assert only the BALANCE in the message (not a row) is superseded by item 1
above. Its choice to accept any `Exception` for item 6 is superseded by the named
`HoldoutBoundaryError`. Everything else in its 16 tests stands: 15 correctly RED with each failure
traced to its specific defect rather than a fixture bug, and one hygiene test proving the real lock
file is byte-identical before and after — which is the single most important test in that file.
