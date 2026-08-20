# Spec: adversarial engine and portfolio invariants

Phase A track A5. This spec describes **required behaviour under hostile inputs**. It adds no
new feature. Every test written from it either passes against the current engine — in which case
the invariant is confirmed — or fails, in which case the failure is a FINDING and is reported to
the lead, never "fixed" by weakening the assertion.

## Why this exists

The suite is broad (2,612 tests, 99.8% line coverage) but the coverage is mostly constructive:
it exercises paths that were built, using inputs those paths expect. An audit of the engine and
portfolio test files found no test anywhere for:

- a position crossing through zero from long to short in one order
- short to flat as a named case
- negative cash
- a gap through a stop
- a halt or missing bar arriving after entry

High line coverage with these missing is exactly the shape of a suite that agrees with its own
implementation. See also `.claude/progress/` on the parallel-build integration seam: 242/242 green
while the system was broken at a boundary neither side tested.

## The invariants

These must hold on EVERY row of EVERY backtest, whatever the input:

**I1 — Accounting identity.** `cash + positions_value + costs == initial_capital + pnl`, to
`atol=1e-6`. Already enforced at `guards.py:655-677` but only at bar open on `close[t-1]`
(`engine.py:412-419`). These tests assert it **after fills, after stops, after square-off and
after the terminal liquidation** as well.

**I2 — Session ends flat.** `np.all(portfolio.shares == 0)` at `is_last_row`. Non-negotiable:
this repo backtests intraday strategies and a carried position is a modelling failure, not a
position.

**I3 — No lookahead.** A fill at row `t` uses only data at or before `t`. A strategy that reads
`close[t]` and trades on it cannot make money (`engine.py:250` already covers the basic case;
these tests widen it to stops and square-off).

**I4 — Finite everywhere.** Equity, returns, turnover, weights and share counts are finite on
every row. NaN in an input never becomes NaN in an accounting quantity — it becomes a skipped
name.

**I5 — `present` is not `tradable`.** A bar that exists but fails the tradable filter is never
filled against. (Rule 7.)

**I6 — Costs are non-negative and monotone in notional.** Trading more never costs less.

## Required tests

Grouped by hostility. Each author writes their own file; both cover all groups.

### Position transitions
1. Long -> short in a single order, crossing zero. Assert the realised P&L on the closing leg and
   the new short's cost basis are separated correctly, and I1 holds after the fill.
2. Short -> flat. Assert cash returns and I1.
3. Short -> long crossing zero.
4. Long -> flat -> long again within one session.
5. A target identical to the current holding emits no order and charges no cost.

### Fills
6. Zero fill (bar traded value zero): order rejected, position unchanged, no charge, I1 holds.
7. Repeated partial fills across consecutive rows.
8. A rejected order does not decrement `in_flight` incorrectly, and the same target is not
   double-ordered on the next decision row.
9. Fill at a price of exactly zero, and at a NaN price: both must be rejected, not divided by.

### Market pathologies
10. Halt after entry: a symbol becomes untradable while a position is open. Assert the position
    is not marked with a stale price into P&L, and that square-off/terminal liquidation still
    reaches flat (I2). If it cannot, that is a finding.
11. Missing bar (NaN OHLCV) after entry, same assertions.
12. Gap through a stop: `open[t]` is already beyond the stop price. Per `engine.py:9-11` the fill
    must be at the conservative price — for a long stop, `min(stop_price, open[t])` — never at
    the stop price when the open is worse.
13. A symbol whose entire column is NaN (absent symbol) never enters a cross-sectional
    computation and never receives an order.
14. Extreme price move (10x in one bar) does not produce non-finite equity.

### Capital
15. Negative cash: drive the book so cash goes negative. Assert this is either prevented or
    surfaced explicitly — a silently negative cash balance financing further trades is a finding.
16. Ruin: equity reaches zero. Assert `ruined` is set and `ruin_index` points at the right row.

### Session structure
17. A 60-bar session (Muhurat) and a 105-bar session in the same panel. All invariants hold; no
    code path assumes 375. (Rule 5.)
18. A single-row session.
19. A session where the square-off time does not exist in the panel at all.
20. Two adjacent sessions with a weekend gap: no state leaks across `on_session_start`.

## Constraints

- Build panels synthetically and deterministically. Do not touch `data/` (rule 2).
- Never forward-fill to make a test convenient (rule 6). NaN means no bar occurred, and that is
  frequently the point of the test.
- A statistical assertion, if any, follows rule 9: assert on a single spread or on a
  false-positive rate over >= 30 seeds, never a per-bucket coverage check, and never fix a
  failure by changing the seed.

---

# CONFIRMED FINDINGS — 2026-08-19. Four real engine defects, verified by the lead.

Author B's suite ran against the CURRENT engine: **16 of 20 invariants hold, 4 fail.** I
reproduced all four independently and confirmed each in source rather than trusting the report.
These are now implementation work, not test work.

## F1 — a decision on a session's LAST ROW fills on the NEXT SESSION's first bar. SEVERE.

`engine.py:568-569`:

    fill_row = t + 1 + int(config.decision_latency_bars)
    if fill_row < n_rows:                      # <- PANEL bounds. Never SESSION bounds.

and `engine.py:404-409`, the session reset, clears `active_stops` and `square_off_queued` and
**not** `pending_orders` or `in_flight`.

So an order queued by a Friday decision at the last bar of the session survives the weekend and
fills at Monday's open. Confirmed against the trades ledger. **In a repo whose entire premise is
intraday trading with a forced-flat EOD invariant, this is an overnight position created by
machinery that nobody declared.** It also charges intraday costs for what is economically a
delivery trade.

**Fix belongs in `specs/order_lifecycle.md`** — same code region, same implementer. `fill_row`
must be bounded by the session, not the panel: an order that cannot fill within the session in
which it was decided is DROPPED, and the drop is counted and reported (a silent drop is how this
class of bug hides). Add to the session reset: assert `pending_orders` is empty at
`is_first_row`, so this can never regress silently.

Note this is strictly BROADER than the square-off guard in `order_lifecycle` section D. That guard
stops decisions at or after 15:20; F1 fires for any decision on a session's final row regardless
of square-off configuration.

## F2 — cash goes negative, silently

Measured: a 100%-of-capital entry plus real transaction costs drives cash to **-90.826** while
`result.ruined` stays `False` and equity stays strongly positive. Nothing prevents it and nothing
in `BacktestResult` reports it.

**Decision: SURFACE, do not prevent.** Preventing it would mean the sizer silently under-fills
against a target the strategy asked for, which is a worse lie than the current one. Add
`min_cash_seen: float` and `n_rows_negative_cash: int` to `BacktestResult`, and a FULL-strictness
guard that raises when cash goes below a configured floor (default: raise on any negative). A
backtest financing trades from an overdraft it never declared is not a backtest of a strategy
anyone could run.

## F3 — `ruin_index` is one row late

`engine.py:122`:

    bad[1:] = (~np.isfinite(prev)) | (prev <= 0.0) | (~np.isfinite(equity[1:]))

A row is flagged when its PRIOR equity was <= 0. Note what is absent: `equity[1:] <= 0.0`. So the
row where equity first reaches zero is never itself flagged — only the row after it. Confirmed:
crash at index 1, `ruin_index` reported as 2.

**Fix:** add `(equity[1:] <= 0.0)` to the disjunction so the crash row is flagged where it occurs.

**Qualification from author A, who reached the same area from a different angle and is right:**
`ruin_index` indexes `equity_curve`, which is sampled at DECISION rows plus one final append —
**not at every bar**. So `ruin_index` is a decision-row index, and "the right row" is meaningless
without saying in which space. Fix the off-by-one AND document the index space at the field. Two
independent authors converging on one under-specified field is a strong signal it is genuinely
under-specified.

## F4 — one NaN close on a HELD symbol poisons equity

`portfolio.py:30-35`:

    safe_prices = np.where(self.shares != 0, prices, 0.0)

The comment says this masks zero-share columns "so a NaN price in an absent or flat symbol
contributes exactly zero, never NaN". True — and it is silent about the case it does not handle:
where `shares != 0`, the RAW price is kept, so a NaN close on an OPEN position makes the whole
`np.dot` NaN and writes a literal `nan` into `equity_curve`.

**This is a docstring that describes the protection it has and not the hole beside it**, which is
how it survived review.

**Fix requires a decision, not just code.** A held symbol with no bar has an unknown
mark-to-market. Options: carry the last known price (violates rule 6 — that is forward-filling),
or mark at the last known price EXPLICITLY and count it, or refuse to mark and raise. Recommended:
mark using the last observed close for that symbol, increment a new
`n_stale_marks` counter, and surface it — visible, opt-out-able, never silent. Route to the lead
before implementing; this one changes reported P&L.

## What held

Items 1-9, 12-14, 17-19 all PASS. Notably the accounting identity, session-ends-flat, the
conservative gap-through-stop fill, all-NaN column exclusion, the 10x price move, single-row
sessions, and the 60-bar/105-bar irregular session handling. **The engine is substantially sound;
these four are specific, locatable and fixable.**

One non-fatal note, worth recording: I2 (session ends flat) holds even for a halted symbol, but
only because `_execute_direct_fill` — used by forced square-off and EOD liquidation — bypasses the
`tradable` mask entirely. The invariant holds for a reason nobody chose. In reality you cannot
liquidate a halted stock. Not a defect in the invariant; a limitation of the model, and it should
be stated where `forced_eod_liquidation_days` is reported.

## F5 — forced-EOD liquidation charges the flat DP fee for symbols that were NEVER TRADED

Found independently by the second adversarial author while debugging its own draft, i.e. not by
any test in either committed suite. Verified by the lead:

    NSEDeliveryEquityCosts().charges(FillBatch(
        notional=[1_050_000, 0, 0], is_buy=[False, False, False]))
    -> stt = [1065.93, 15.93, 15.93]

`costs.py:200-213` applies `dp_charge_per_scrip` to EVERY row where `is_buy` is False:

    stt = np.where(fills.is_buy, stt, stt + self.dp_charge_per_scrip)

with no zero-notional exclusion. And the engine always hands forced-EOD liquidation the FULL
`-portfolio.shares.copy()` array (`engine.py:615`), which is mostly zeros in any real universe.

Measured overcharge per forced-EOD event:

    universe  10 symbols,  3 held  ->  Rs 111.51
    universe 149 symbols,  3 held  ->  Rs 2,325.78

This silently inflates `total_costs`, and it inflates it **specifically on the path that F1 and
the square-off latch make more frequent than it should be** — so the three defects compound: the
latch strands positions, they hit forced liquidation, and forced liquidation then overcharges by
a flat fee times the count of symbols that were never involved.

**Fix:** apply `dp_charge_per_scrip` only where `notional != 0`. This is a genuine per-scrip
charge and a scrip with no trade incurs none. Add a regression test with a mostly-flat share
vector — neither committed suite caught this, because both used single-symbol panels for the
EOD-liquidation cases.

Note the direction: this one makes reported costs LOWER once fixed, where F1's session bound and
the `order_lifecycle` cost composition make them HIGHER. They must be landed and measured
together, not one at a time, or each will look like it broke the other's baseline.

---

# F6 — THE ACCOUNTING GUARD HAS A BLIND SPOT FOR EXACTLY THE FAILURE IT EXISTS TO CATCH

Found by the second adversarial author (suite A) while investigating F4. **This is the most
important finding of the batch** and it was invisible to five earlier reviews of this file.

`guards.py:671-672`:

    discrepancy = (cash + positions_value + costs) - (initial_capital + pnl)
    if abs(discrepancy) > atol:
        raise ContractViolation(...)

If any input is NaN, `discrepancy` is NaN, and **`abs(nan) > atol` is `False` in IEEE-754**, so the
guard returns cleanly. Verified directly:

    guards.check_accounting(cash=nan, positions_value=nan, costs=0.0,
                            initial_capital=1e7, pnl=0.0)   under FULL strictness
    -> GUARD PASSED

So the accounting identity — the invariant this repo holds up as its strongest correctness
control, checked on every row of every backtest under FULL — **cannot detect a NaN book.** And F4
produces exactly a NaN book. The safety net has a hole shaped precisely like the failure it was
built for.

**Fix:**

    if not np.isfinite(discrepancy) or abs(discrepancy) > atol:

**Scope checked, and it is NOT systemic — do not over-correct.** The other numeric comparisons in
`guards.py` filter to finite values before comparing (`bounded_output` at `:520`, `validate_weights`
at `:616`), so they are NaN-safe by construction. `check_accounting` at `:672` is the only guard
that computes a scalar and compares it without a finiteness test. One fix, one place.

**The generalisable lesson:** `abs(x) > tol` is not a validity check, it is a magnitude check, and
it silently answers "fine" for the one input that is neither large nor small. Any guard whose
predicate is a bare inequality needs an explicit finiteness test beside it. Worth a sweep of any
NEW guard this program adds — including the `in_flight` guard in `specs/order_lifecycle.md`
amendment 1, which must not repeat this.

# F7 — forced liquidation ignores the `tradable` mask entirely

Both authors reached this; suite A asserts it as an I5 violation, suite B recorded it as a
non-fatal note. Suite A is right to assert it.

`_execute_direct_fill`, used by forced EOD liquidation (`engine.py:612-623`), bypasses the
`tradable` mask. A halted symbol is still filled. So I2 (session ends flat) holds **for a reason
nobody chose**: the model can always liquidate because it never asks whether it may.

This is a MODELLING limitation, not a code bug, and it flatters results — in reality a halted
position is carried, with overnight risk and delivery costs. Rule 7 (`present` and `tradable` are
distinct and must never be conflated) is being honoured everywhere except the one path where the
distinction has the largest consequence.

**Required: surface it, do not silently "fix" it by refusing to liquidate** — refusing would break
I2 and change every result. Add a counter for forced liquidations executed against a non-tradable
bar, and state the limitation where `forced_eod_liquidation_days` is reported. A number that
depends on an impossible fill must be labelled as such.

## Corroboration note

Two authors, independent, never reading each other's files, converged on F1, F2, F3, F4 and F7
from different test constructions. Suite A additionally found F6; suite B additionally found F5.
**Neither suite alone found all seven.** Suite A: 21 tests, 17 pass / 4 fail. Suite B: 20 tests,
16 pass / 4 fail. The overlap is corroboration; the disjoint findings are the argument for the
dual-suite rule.

---

# F8 — DAILY TURNOVER IS MISATTRIBUTED ACROSS SESSION BOUNDARIES

Found by a test author performing a mutation/vacuity audit on the `daily_results` suite, not by
any adversarial test. Verified by the lead in source.

`notional_since_snapshot` accumulates at every fill site (`engine.py:600, 650, 771, 798`) but is
read out and reset **only at decision rows** (`:670, :675`) and at the final row (`:822, :827`).

Lines 771 and 798 are the square-off and forced-EOD fills. Those execute **after the session's
last decision row**. Their notional therefore sits in the accumulator until the NEXT decision row
reads it — which is in the **following session**.

Measured on a two-decision-per-day strategy with the default `square_off_time="15:20"`:

    trades:   +10,000 @ 1,000,000  ENTRY     (10:00)
              +20,000 @ 2,000,000  ENTRY     (10:30)
              -30,000 @ 3,000,000  EOD_EXIT  (15:20, after day 0's last decision snapshot)

    per-row turnover  [0., 0.1, 0.5, 0.1, 0.5]
    daily turnover    [0.1, 1.1]      <- day 0 UNDERSTATED, day 1 OVERSTATED

Day 0's true turnover is 0.6 (0.1 + 0.2 + 0.3). It reports 0.1. The square-off's entire ₹3,000,000
lands on day 1.

**And `specs/daily_results.md` made it worse, not better.** Its section A says the engine should
record "the current `day_idx`" wherever it appends — which is the day of the SNAPSHOT ROW, not the
day the fill executed. Followed literally, that instruction cements the misattribution. My spec
was wrong.

**Fix:** attribute turnover to the session in which the fill EXECUTED. Accumulate per-day (or
stamp the accumulator with the day at fill time and flush it at session end), rather than reading
whatever has piled up at the next snapshot. Session end must flush, so no notional can cross a
boundary.

**Blast radius — narrower than it looks, but check before assuming.** `research/tilt.py` computes
its own turnover and does NOT use the engine, so the tilt candidate's published
turnover/cost numbers are unaffected. What IS affected is every engine-derived turnover figure —
including `nq backtest`'s reported `mean turnover` and any per-day cost attribution built on it.
For a strategy that squares off every session, that is every session in the sample.

Corollary worth stating: **the pooled/total turnover is correct; only its per-day attribution is
wrong.** So aggregate cost estimates survive and per-day ones do not. Do not over-claim the
damage, and do not under-claim it either — anything conditioning on daily turnover (regime splits,
per-year tables, cost-per-day series) has been reading a shifted series.

---

# F9 — THE F3 FIX OVER-CORRECTED AND BROKE RETURNS/EQUITY RECONCILIATION

Found by the lead while writing the contract-update briefs for F3, i.e. only because I stopped to
compute the expected values instead of assuming them. **Caused by my own F3 brief**, which said
"add `equity[1:] <= 0.0` to the disjunction" without noticing the disjunction is used for TWO
different purposes.

`_compute_returns` used ONE mask both to LOCATE ruin and to ZERO returns. Adding `cur <= 0.0`
correctly moved `first_ruin_index` onto the crash row — and simultaneously zeroed the return INTO
the crash, which is well defined and is exactly the -100% that wiped the book out.

Measured, `equity = [1e7, 5e6, 0.0, 5e6]`:

    returns AFTER the F3 fix   [0.0, -0.5,  0.0, 0.0]   -> compounds to 5,000,000
    equity curve                [1e7,  5e6,  0.0, 5e6]   -> says 0
                                                             ^ IRRECONCILABLE

A returns series that cannot reproduce its own equity curve is worse than the off-by-one it
replaced: total return computed from returns would have MISSED A TOTAL WIPEOUT.

**Fix — separate the two predicates, because they answer different questions:**

    detect   = bad_denominator | bad_numerator | (cur <= 0.0)   # WHERE ruin occurs -> ruin_index
    unusable = bad_denominator | bad_numerator                  # where a return CANNOT be computed

`cur <= 0.0` belongs only in `detect`. A finite `cur` over a good positive `prev` is a computable
return, however catastrophic. Rows AFTER ruin need no special case: their own denominator is the
ruined equity, so `prev <= 0.0` already catches them.

Verified after the fix:

    equity  [1e7, 5e6, 0.0, 5e6]  -> returns [0.0, -0.5, -1.0, 0.0]  ruin_index 2
    compounding reproduces the wipeout exactly.
    equity  [1e7, -1e6, 5e6]      -> returns [0.0, -1.1,  0.0]       ruin_index 1
    healthy series unchanged.

Guarded by a new regression test, `test_returns_reconcile_with_equity_through_ruin`.

**The lesson, and it is the sharpest one of the program so far:** F3 was a one-line change to a
boolean expression, described in a spec, implemented correctly as described, and it introduced a
worse defect than it fixed. It was caught only because writing the test-update brief required
computing the expected numbers, and the numbers did not reconcile. **A one-line fix to a predicate
that serves two callers is a two-line fix.** Before changing any mask, enumerate every consumer of
it.

---

# F2 MEASURED ON REAL DATA — the backtest runs on an UNDECLARED OVERDRAFT

F2 added `min_cash_seen` and `n_rows_negative_cash`. Now that they reach `metrics.json`, they can
be read on real data for the first time. Measured by the lead, `volume_breakout`, all_equity
(149 names), January 2024, capital Rs 1,00,00,000:

    min_cash_seen                Rs -64,43,280      peak overdraft = 0.64x capital
    n_rows_negative_cash         568 rows
    peak gross book financed     Rs 1,64,43,280     = 1.64x declared capital

**Every published `volume_breakout` number describes a book running at up to ~1.64x, not 1.0x.**
The config declares `gross: 1.0` and `capital: 1e7`; the engine has been financing the difference
from a cash balance nobody constrained.

## Why this matters more than the Sharpe

`volume_breakout` is dead either way, so this does not resurrect or further kill it. What it
changes is the READING of every cost and capacity conclusion drawn from the engine:

- **Cost-per-rupee ratios are computed against a denominator that is too small.** Turnover as a
  fraction of "capital" overstates churn relative to the capital actually at work.
- **Any capacity statement is wrong by the leverage factor.** Phase H's ladder ("at what capital
  does the edge die") would have inherited this silently.
- **Risk numbers are understated.** A vol target computed on declared capital does not describe a
  book holding 1.64x that notional -- which is exactly the defect `specs/portfolio_vol_target.md`
  amendment 6 removed at the sizer level, reappearing here at the engine level.

## Mechanism NOT yet established -- do not guess it

The most likely candidate is fill sequencing: entries for a rotation filling at `open[t+1]` before
the corresponding exits have settled, so gross exposure transiently exceeds the target. With
`gross = 1.0` and `max_weight = 0.10` over 149 names, steady-state cash should sit near zero, not
64% below it. Costs alone cannot explain a number this size.

**This is stated as a hypothesis and must be measured, not assumed.** Required follow-up:
1. Reconstruct cash row by row from the trade blotter and find the rows where it dives.
2. Determine whether those rows coincide with rotations (simultaneous entry and exit).
3. Decide the intended semantics DELIBERATELY: either the engine models a cash-settled book and
   must reject orders it cannot fund, or it models a margin account and must DECLARE the facility
   and charge for it. Right now it does neither and simply lets the balance go negative.

Until that is settled, treat any engine-derived capacity or cost-per-capital figure as carrying an
unquantified leverage factor.

## Also measured, and quieter than expected

    n_orders_dropped_at_session_end            = 0
    n_forced_liquidations_against_nontradable  = 0
    n_stale_marks                              = 0

So on this configuration F1's session leak, F7's halted-symbol liquidation and F4's stale marks do
NOT fire. They were real defects and the fixes are correct, but their frequency here is zero --
worth knowing before attributing any number change to them. It also means the F1/F5/composed-cost
mechanisms had nothing to act on in the 2024 baseline diff, which is why the cost figures did not
move. That confirms the reading recorded in `results/baselines/PHASE_A_COST_DIFF.md`.
