# Spec: order lifecycle — intent tagging, netted queue, and square-off completion

Covers Phase A tracks A1 and A2. They are ONE spec because both restructure the same region of
`backtest/engine.py` (the pending-order queue, ~lines 236 and 560-625); splitting them would put
two implementers in the same lines.

## Why this exists

Three live defects, all in `src/nifty_quant/backtest/engine.py`.

### 1. Square-off is a LATCH, not a state

`engine.py:591` gates on `if t >= square_off_row_for_day[day_idx] and not square_off_queued:`
and `:610` sets `square_off_queued = True` at the moment the order is QUEUED — before any fill
occurs. The queued order then fills through the partial-fill path (`execution/fills.py:108`,
capped at `max_participation` of bar traded value) and `fills.py:73` states outright that it
partial-fills "without carrying a shortfall". `engine.py:429` decrements `in_flight` by the FULL
order regardless of what actually filled.

Consequence: if the square-off bar is thin, the position is only partly liquidated, the latch
refuses to re-queue for the rest of the session, and the remainder survives to `is_last_row`
where it is force-liquidated (`engine.py:612-621`, `forced_eod_liquidation_days += 1`) at the
final close under a different cost model. This inflates forced liquidation, costs, turnover and
the unfilled-notional metric on exactly the thin-liquidity names where it matters most.

No existing test covers it: every square-off test in the suite runs with default participation
and ample volume.

### 2. The documented square-off contract is not implemented

`engine.py:14` documents step 5 as: "If t is at or past square_off_time (default 15:20): force
targets to zero." The decision block (`engine.py:484-588`) contains **no square-off guard**. A
decision row at or after the square-off row therefore calls the strategy, sizes targets and
queues a fresh ENTRY order — re-opening a book that the latch (defect 1) then refuses to
liquidate a second time.

### 3. `pending_orders` cannot hold two orders for one fill row

`engine.py:236` declares `pending_orders: dict[int, np.ndarray]` — one array per fill row. Both
collision sites (`:570-582` decision side, `:606-608` square-off side) handle a collision by
subtracting the old order from `in_flight` and **overwriting** it. That is cancel-and-replace,
which is a defensible policy but an undeclared one: an EOD liquidation silently discards a
pending entry with no record that it happened, and nothing anywhere in `src/` tags an order with
its purpose. There is no `OrderIntent` enum, and the trade blotter (`engine.py:254-267`) has no
intent column, so a post-hoc reader cannot tell an entry from a stop-out from an EOD hammer.

## Required behaviour

### A. `OrderIntent`

A public enum in `backtest/engine.py` (or a new `backtest/orders.py` if the implementer prefers;
either is acceptable, but it must be importable as `from nifty_quant.backtest.orders import
OrderIntent` OR `from nifty_quant.backtest.engine import OrderIntent` — pick one and document it):

    ENTRY       -- from the decision block, opening or adjusting a target
    REBALANCE   -- reserved; not emitted by the current engine. Declared so the blotter schema
                   is stable when Phase G adds rebalancing. An emitted REBALANCE must behave
                   exactly like ENTRY.
    RISK_EXIT   -- from the intrabar stop clock
    EOD_EXIT    -- from the square-off block
    FORCED_EOD  -- from the terminal `is_last_row` liquidation

`FORCED_EOD` is distinct from `EOD_EXIT` on purpose: the first is a failure to liquidate in
time, the second is normal end-of-day behaviour, and conflating them would hide the very metric
(`forced_eod_liquidation_days`) that defect 1 corrupts.

### B. Netted order queue

`pending_orders` becomes `dict[int, list[PendingOrder]]` where `PendingOrder` is a frozen
dataclass carrying at minimum:

    intent: OrderIntent
    shares: np.ndarray      # float64, length n_symbols, signed
    queued_row: int         # the row t at which this order was queued

Rules:

1. Queueing an order for a fill row that already has entries **appends**. It does not overwrite.
2. At the fill row, orders are combined by **summation in queue order** into one net share
   vector, and that net vector is executed as a single fill batch. Summation is the only netting
   rule; there is no priority or cancellation.
3. `in_flight` remains the sum of all queued (not yet filled) share vectors and must equal
   `sum(o.shares for row in pending_orders.values() for o in row)` at the end of every row. This
   is an invariant the tests must assert directly, not infer.
4. The blotter gains an `intent` column. When a fill row nets orders of different intents, the
   blotter records the intent of the **largest absolute contribution per symbol**; ties resolve
   to the later-queued order. This is a reporting convention, not accounting — the netted
   quantity is what is executed either way. Document the convention where it is implemented.

The `# pragma: no cover` unreachability argument at `engine.py:571-580` is deleted along with
the overwrite branch it protects. Its reasoning holds only for a fixed `decision_latency_bars`
and is now moot: appending is correct for any latency.

### C. Square-off as state

Replace the `square_off_queued` boolean latch with in-flight tracking:

1. The square-off block re-arms whenever no `EOD_EXIT` order for this session is currently
   pending. Concretely: it queues an `EOD_EXIT` for `-portfolio.shares` when
   `t >= square_off_row_for_day[day_idx]` AND `np.any(portfolio.shares != 0)` AND no `EOD_EXIT`
   is presently in `pending_orders`.
2. That means a partial square-off fill causes a fresh `EOD_EXIT` for the REMAINING shares on
   the next row, and again on the row after, until flat or the session ends.
3. State resets at `is_first_row`, as `square_off_queued` does today (`engine.py:408`).
4. The existing special case where the square-off row IS the last row of the session
   (`engine.py:593`, direct fill at `close[t]`) is retained unchanged in shape, but tagged
   `EOD_EXIT`.
5. `forced_eod_liquidation_days` must therefore only increment when retrying genuinely could not
   finish the job — i.e. the book is still non-flat at `is_last_row` after every intervening row
   has attempted liquidation.

### D. No re-opening after square-off

The decision block is gated: when `t >= square_off_row_for_day[day_idx]`, the engine does not
call `strategy.on_decision` and queues no `ENTRY` order. `on_session_end` still fires. This makes
`engine.py:14` true.

### E. Forced-EOD costs are COMPOSED, not substituted

`engine.py:620` passes `NSEDeliveryEquityCosts()` alone to `_execute_direct_fill`. That class's
own docstring (`execution/costs.py:189`) calls it a **"Supplementary delivery cost model"** —
it emits zero brokerage, zero exchange, zero SEBI, zero IPFT, zero stamp and zero GST, because
it was written to be ADDED to the intraday charges, not to replace them. As used today the
forced-EOD leg pays STT and the DP charge and nothing else, so a forced liquidation is
systematically CHEAPER per rupee in six of seven components than an ordinary sale.

Required: the forced-EOD leg is charged `config.cost_model` charges PLUS
`NSEDeliveryEquityCosts()` charges, summed component-wise. Introduce a small composite —
e.g. `CompositeCostModel(models: tuple[CostModel, ...])` in `execution/costs.py` whose
`charges()` sums the components of each member — rather than special-casing the arithmetic
inside the engine. `Charges` already has the right shape for this.

Note for the implementer: this makes forced liquidation more expensive than it is today, so the
`make verify` volume_breakout 2024 reference numbers WILL move. That is a correction, not a
regression. Record the before/after both-ways in the commit message.

## What must NOT change

- The mandatory one-bar execution lag (`engine.py:568`) and its structural guarantee.
- The `in_flight` netting that prevents double-ordering across the latency window
  (`engine.py:567`).
- The accounting invariant at `guards.py:655-677`.
- Ordinary-path results. A backtest with ample liquidity, no stops and no post-square-off
  decision rows must produce **bit-identical** equity, trades and turnover to HEAD, except for
  the new `intent` blotter column and except where defect E's cost composition applies.

## Known contract conflicts — REPORT, DO NOT EDIT

Two existing tests encode the OLD overwrite contract and are expected to fail against this spec:

    tests/test_engine_coverage2.py:462  test_pending_order_fill_row_collision_overwrites
    tests/test_engine_coverage.py:2225  test_pending_orders_collision_same_fill_row

Per `CLAUDE.md` rule 1, an implementer may **never** edit a test to make it pass. Report them to
the lead with the failing assertion. The lead adjudicates and rewrites them as
lead-authored contract updates, stated as such.

## Required tests

Both test authors write these independently, from this spec alone, without seeing an
implementation or each other's file.

1. **Partial square-off retries to flat.** A session where the square-off bar has traded value
   small enough that `max_participation` allows only a fraction of the required liquidation.
   Assert: the book is flat before `is_last_row`, `forced_eod_liquidation_days == 0`, and more
   than one `EOD_EXIT` fill appears in the blotter for that session.
2. **Repeated partial square-off.** Same, with thin volume on several consecutive rows, so at
   least three retries are needed. Assert monotonically decreasing absolute position and a flat
   book by session end.
3. **Square-off that genuinely cannot finish** — thin volume all the way to the last row.
   Assert `forced_eod_liquidation_days == 1` and the terminal liquidation still fires, i.e.
   retrying does not remove the safety net.
4. **No re-opening after square-off.** A strategy that requests a non-zero target on every
   decision row, with a decision row scheduled after `square_off_time`. Assert no `ENTRY` order
   is queued at or after the square-off row and the session ends flat.
5. **`in_flight` invariant.** After every row of a multi-symbol multi-session backtest,
   `in_flight` equals the summed shares of all pending orders. Assert on every row, not just at
   the end.
6. **Append, not overwrite.** Force two orders of different intents onto one fill row (e.g. a
   stop-out and a square-off) and assert the executed quantity is their SUM, and that both
   intents are represented in the blotter for that row per the largest-contribution convention.
7. **Intent tagging.** Every row of the blotter carries a valid `OrderIntent`. A backtest with
   stops enabled produces at least one `RISK_EXIT`; a normal session produces `EOD_EXIT`, not
   `FORCED_EOD`.
8. **Composed forced-EOD costs.** For one forced liquidation, assert the charged total equals
   `NSEIntradayEquityCosts().charges(batch).total + NSEDeliveryEquityCosts().charges(batch).total`
   component-wise, and specifically that brokerage and GST on that leg are non-zero (they are
   zero at HEAD).
9. **Ordinary path unchanged.** A liquid, stop-free, no-late-decision backtest reproduces HEAD's
   equity curve and trade count exactly. Build this as a numeric regression against values the
   test itself computes from a deterministic synthetic panel, not against hardcoded magic
   numbers.
10. **Accounting invariant holds throughout**, including on rows where a square-off retry fills.

## Constraints

- Fully vectorized across symbols; the per-row Python loop over `t` is existing engine structure
  and stays.
- float32 at rest, float64 in motion (rule 3). Share vectors are float64.
- No fixed 375-bar session assumption anywhere (rule 5): index via `panel.day_offsets`.
- `present` and `tradable` stay distinct (rule 7).
- No new hand-chosen constant (rule 8). This spec introduces no thresholds; if the implementer
  believes one is needed, that is a spec defect — report it rather than choosing a number.

---

# AMENDMENT 1 — 2026-08-19. Four defects found by a test author before implementation.

Accepted. The amendment wins where it conflicts with the body.

## 1. Required test 5 demanded an assertion with no seam to make it — FIXED BY A GUARD, NOT A HOOK

Required test 5 says `in_flight` must be asserted "on every row, not just at the end". There is no
way to observe `in_flight` from outside `run_backtest`; it is a local. The author assumed a
`row_hook` keyword parameter and correctly flagged that as a divergence risk against the sibling
suite, which will have assumed something else.

**Do not add a `row_hook`.** A test-only parameter on a production entry point is exactly the kind
of seam that later gets used for real and then constrains the engine.

Instead the invariant becomes a GUARD, enforced inside the engine on every row, in the same style
and at the same site as the existing `guards.check_accounting` call (`engine.py:414-425`):

    guards.check(
        in_flight equals the summed shares of every order still in pending_orders,
        "in_flight desynchronised from the pending order book",
    )

FULL strictness only, like `check_accounting`. This is strictly better than the test the spec asked
for: the invariant is then checked on every row of **every** backtest run under FULL, not only
inside one test. Required test 5 becomes: run a multi-symbol multi-session backtest under FULL
strictness and assert it completes without `ContractViolation`; plus a negative control that
deliberately desynchronises the two and asserts the guard fires.

Both test authors: converge on this. Neither `row_hook` assumption survives.

## 2. Required test 8 asked for a component-wise assertion the blotter cannot support

The trade blotter (`engine.py:254-267`) stores a single pre-summed scalar `charges` per trade.
There is no per-component breakdown anywhere in `BacktestResult`, so "assert the charged total
equals ... component-wise" is unassertable as literally written.

Two ways out. **Phase A takes the cheap one; Phase C takes the real one.**

- **Phase A (now):** the test reconstructs a `FillBatch` from the trade's own qty/price and calls
  the two cost models directly, comparing against the blotter's scalar `charges`. The author
  already did this and it is the right call. The component-wise claim is then verified against the
  models, and the total against the engine.
- **Phase C1 (later):** the TCA record adds `fees_bps` and the component split to the blotter, at
  which point the assertion can be made directly. `costs.as_bps_of()` (`costs.py:116`) already
  exists and is called by nothing.

Required test 8 is restated accordingly. The substantive assertion is unchanged and remains the
point: **brokerage and GST on the forced-EOD leg must be NON-ZERO**, where they are zero at HEAD.

## 3. Required test 8 hardcoded a cost model that section E deliberately made generic

The test formula named `NSEIntradayEquityCosts()` explicitly, while section E's fix composes
`config.cost_model` with the delivery supplement — generic over whatever the caller configured.
As written, neither author's test could cover a non-default `cost_model` without contradicting
the spec's own worked example.

Corrected: the assertion is

    charges(forced EOD leg) == config.cost_model.charges(batch) + NSEDeliveryEquityCosts().charges(batch)

component-wise. With the default config `config.cost_model` IS `NSEIntradayEquityCosts()`, so the
worked example still holds; but a second case with a non-default cost model (e.g. `ZeroCost`) must
also be covered, and under `ZeroCost` the forced-EOD leg must charge EXACTLY the delivery
supplement and nothing else. That second case is what proves the composition is generic rather
than a hardcoded pair.

## 4. The two known contract-conflict tests, restated

`tests/test_engine_coverage2.py:462` and `tests/test_engine_coverage.py:2225` encode the OLD
overwrite semantics and will fail. Both authors correctly left them alone. They are the LEAD's to
rewrite as a stated contract update, per rule 1. Recorded here so the implementer does not
"helpfully" fix them.

## Standing rule carried from the portfolio_vol_target spec

That spec needed five amendments, and every defect was the same shape: a number, or here a seam,
that had to come from somewhere with the spec silent about where. Before implementation begins on
THIS spec, the implementer should check the same thing: **if a required behaviour cannot be
observed through any existing API, the spec is asking for a hook it did not declare.** Say so
rather than inventing one.

---

# AMENDMENT 2 — 2026-08-19. The second author found my worked example is IMPOSSIBLE.

## 1. Required test 6's example cannot occur. Only TWO intents ever reach the queue.

Test 6 said "force two orders of different intents onto one fill row (e.g. a stop-out and a
square-off)". Verified in source: that collision is unreachable. Stops fill **immediately at row
t** via `_execute_direct_fill` (`engine.py:484-486`) and never enter `pending_orders` at all.

So the queue's real membership is:

    ENTER THE QUEUE          ENTRY, EOD_EXIT           (filled at open[t+1] via _execute_model_fill)
    DIRECT FILLS, NEVER      RISK_EXIT, FORCED_EOD     (filled at row t via _execute_direct_fill)
    NOT EMITTED              REBALANCE                 (reserved; see body section A)

This changes the design, not just the test. **`PendingOrder.intent` can only ever be `ENTRY` or
`EOD_EXIT`.** `RISK_EXIT` and `FORCED_EOD` are blotter-only tags — they describe a fill that
already happened, not an order that is waiting. The implementer must not build queue machinery for
intents that cannot be queued.

Corrected test 6: collide an `ENTRY` with an `EOD_EXIT` on one fill row and assert the executed
quantity is their SUM. The second author independently arrived at this and it is right.

## 2. Module and representation, pinned

The body allowed `OrderIntent` in either of two modules, which guaranteed the two authors would
guess differently. They did. Pinned:

    nifty_quant.backtest.orders     OrderIntent, PendingOrder
    nifty_quant.execution.costs     CompositeCostModel

The blotter's `intent` column stores the enum's **`.value` as a `str`** — parquet-safe and stable
across processes, unlike an enum member or an int. `results/trials/<hash>/result.parquet` is
written with `to_parquet` and must keep round-tripping.

## 3. "COUNTED" was not a metric. Now it is.

Amendment 1 required that an order which cannot fill within its own session be "DROPPED, and the
drop counted and reported", without naming where. Add to `BacktestResult`:

    n_orders_dropped_at_session_end: int

Do NOT fold this into `rejected_order_rate`. A rejection means the market refused the order; a
session-end drop means the ENGINE cancelled it because it would otherwise have leaked into the
next session. Conflating them would hide F1's frequency behind an unrelated number — and F1's
frequency is the thing we most need to see after the fix, because it tells us how much of every
historical backtest was affected.

## 4. Fault injection for required test 5's negative control

Amendment 1 asked for a negative control that "deliberately desynchronises `in_flight`". With no
declared seam that means monkeypatching an engine internal, which the author flagged. Acceptable
approach: the test may monkeypatch inside `nifty_quant.backtest.engine`'s module namespace to
corrupt the invariant, then assert `ContractViolation`. It is a test-only intrusion into a
private, and it is justified precisely because the alternative — a production `row_hook` — is
worse. Note it at the test.

## Standing note

Three spec defects across two amendments here; five plus an addendum on `portfolio_vol_target`.
The pattern in this file is different from that one: there, unstated NUMBERS. Here, **unstated
STRUCTURE** — which module, which representation, which intents can actually occur. Both are the
same underlying failure: prose that reads as complete because the author already knows the answer.
