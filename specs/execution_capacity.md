# Spec H — execution realism and capacity

Status: spec, written before implementation. Author: lead. Date 2026-08-23.

## The question this phase answers

**At what capital does the edge disappear?** A strategy that works at Rs 1 lakh and dies at Rs 10
crore is not a strategy, it is a demonstration. This program has one candidate and no idea what it
can carry.

## Starting position, from inventory

`execution/costs.py` has the full NSE stack and it is real (`:170-177`, `:189-209`):
brokerage `min(20, 0.0003*N)`, STT 0.00025 sell-side, exchange 0.0000297, SEBI 1e-6, IPFT 1e-6,
stamp 0.00003 buy-side, GST 18% on (brokerage+exchange+SEBI) only. `round_trip_bps(notional)` at
`:211`. `FixedBpsCost(bps=0.0)` at `:277` is the hook for a cost ladder.

`execution/fills.py`: `FillModel(slippage, max_participation=0.02).fill(...)` caps at
`max_participation * bar_traded_value` (:139), prices slippage on the CAPPED notional (:148), and
**drops unfilled quantity — it is never requeued** (:167-179), only reported as
`unfilled_notional` / `rejected`.

That last detail is the crux of this phase. `volume_breakout` measured **73.7% of desired notional
unfilled**. A backtest that silently drops the other 73.7% is not measuring the strategy.

Capacity work today is partial: `_capacity_water_fill` + `unmet_gross_pct` in-library;
`scripts/recon_tilt_liquidity.py:589` computes `implied_capacity_aum = 0.02 * weighted_median_ADV /
typical_weight`, which the script itself calls crude at `:615` and which is NOT turnover-based.

**No cost x participation ladder exists. The 7x5 grid is greenfield.**

## H1 — the cost ladder

Cost in {0, 2, 5, 8, 10, 15, 20} bps x participation in {0.5, 1, 2, 5, 10}% = 35 cells.

For each cell: net Sharpe, net bps/day, turnover, unfilled fraction. Output a surface, not a
number. `FixedBpsCost` drives the cost axis; `FillModel.max_participation` drives the other.

The 0 bps column is not padding — it separates "the edge is small" from "the costs are large",
which are different diagnoses with different fixes.

## H2 — the Rs 66,667 crossover, named and tested

NSE brokerage is `min(Rs 20, 0.03%)`, so below **Rs 66,666.67** per order the percentage term binds
and round-trip bps is CONSTANT; above it, brokerage is flat and bps DECAYS with size.

Measured `round_trip_bps`: 10.6245 at 50k / 66,666 / 66,667, then 8.2645 at 100k, 3.5917 at 1cr.

This is real behaviour at `costs.py:190-192` but **the crossover is nowhere named in code and no
test asserts it.** It must be, because it inverts the usual intuition: below the crossover, raising
capital per name changes nothing about cost in bps.

**Correction to carry:** `specs/tilt_backtest_wrapper.md:155` quotes 4.55 bps round-trip at
Rs 66,667. The real figure is **10.62** (`research/tilt.py:552` uses plain `NSEIntradayEquityCosts`).
Do not propagate 4.55.

## H3 — capacity from turnover, not just ADV

The existing estimate is a position-size heuristic (position <= 2% of ADV). The tilt turns its book
over ~11% daily, so the binding constraint is the DAILY TRADE, not the held position. Capacity must
be derived from turnover x participation cap, and the two numbers reported side by side — the
existing ~Rs 400 crore ADV-based figure is an upper bound, and saying so is part of the deliverable.

## H4 — unfilled quantity becomes visible

Report unfilled notional as a first-class result of every ladder cell, not a footnote. A cell where
the edge survives costs but 60% of the intended trade never happens has not demonstrated anything.

**Do NOT change the engine to requeue unfilled orders in this phase.** That is an execution-policy
change with its own consequences and belongs to its own spec; here it is measured, not fixed.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. `round_trip_bps` is CONSTANT in bps below Rs 66,666.67 per order and DECAYS above it — assert at
   50k / 66,666 / 66,667 / 100k / 1cr against the measured 10.6245 / 10.6245 / 10.6245 / 8.2645 /
   3.5917. This is the crossover regression test.
2. The crossover constant is DERIVED in code from `brokerage_flat / brokerage_pct`, not written as
   a literal 66667 (rule 8 — and it is exactly derivable, so a literal is inexcusable).
3. The GST base is (brokerage + exchange + SEBI) only — a test asserts STT and stamp are OUTSIDE it.
4. STT is sell-side only; stamp is buy-side only.
5. The ladder produces 35 cells and each carries net Sharpe, turnover AND unfilled fraction — a
   cell missing the unfilled fraction fails.
6. At `max_participation` low enough to starve fills, unfilled fraction rises and net Sharpe is
   reported alongside it, not silently improved by dropping the unfilled part.
7. Capacity derived from turnover is LOWER than the ADV-based figure on the tilt's ~11% daily
   turnover — assert the ordering, since the whole point is that ADV alone overstates.
8. `FixedBpsCost(0.0)` yields gross results identical to `ZeroCost`.

---

## AMENDMENT 1 (2026-08-23) — H3's required ordering is WRONG. Lead's error.

H3 above says the ADV-based figure "is an upper bound", i.e. that turnover-based capacity comes out
LOWER. **That is backwards for this book, and test suite A was correct to refuse to invent a
formula that produced the required ordering.** Both suites' authors flagged the tension rather than
resolving it silently; that is exactly the behaviour rule 1's dual-suite design exists to produce.

### The correction

A position limit and a daily-trade limit are **two different constraints on two different risks**,
not two estimates of one number:

    position (liquidation risk):  AUM <= cap * ADV_i / w_i
    daily trade (impact risk):    AUM <= cap * ADV_i / (w_i * turnover_frac)

The trade constraint is the position constraint divided by `turnover_frac`. So:

    turnover_frac < 1  ->  trade constraint is LOOSER  ->  POSITION limit binds
    turnover_frac > 1  ->  trade constraint is TIGHTER ->  TRADE limit binds

Measured for the tilt, from the frozen pre-registration (`turnover = 0.1272107659289835`):

    ratio = 1 / 0.1272 = 7.861x LOOSER

So the trade constraint does not bind at all, and the ~Rs 400 crore ADV-based figure is the
**operative** number rather than an upper bound to be revised downward. H3's premise — "the tilt
turns its book over ~11% daily, so the binding constraint is the DAILY TRADE" — inverts the
implication: low turnover is precisely what makes the trade constraint NOT bind.

### What obligation 7 must assert instead

REMOVE the required ordering `turnover_capacity < adv_capacity`. Replace with:

1. `turnover_capacity / adv_capacity == 1 / turnover_frac`, to tolerance. This is the real
   relationship and it is directionless — it holds for high- and low-turnover books alike.
2. Reported capacity is `min(position_capacity, trade_capacity)`, with WHICH ONE BOUND recorded in
   the output. A capacity figure that does not say which constraint produced it is not actionable.
3. A regression pinning that at `turnover_frac < 1` the position limit is the binding one, and at
   `turnover_frac > 1` the trade limit is — so the direction cannot silently flip again.

### Consequence for the candidate, and why H3 now interacts with the liquidity finding

Low turnover is a genuine capacity ADVANTAGE, worth ~7.9x on the trade-impact constraint. But it
does NOT rescue the position constraint, and the position constraint is exactly where the
2026-08-23 liquidity result bites: the tilt's edge is concentrated in the bottom two ADV deciles
(32.95% of gross excess vs a null p95 of 23.82%, above the max of 10,000 replicates). Those names
have the SMALLEST ADV, so `cap * ADV_i / w_i` is smallest precisely where the edge lives.

**Therefore capacity must be computed PER LIQUIDITY DECILE, not on a pooled average.** A pooled
figure averages the binding constraint away and will overstate capacity for this strategy
specifically. Add this as obligation 9; it is now the most important number in Phase H.

---

## AMENDMENT 2 (2026-08-31) — interface pinned; the pre-amendment ordering tests replaced

Audited BOTH suites' interface guesses exhaustively before implementation (the G-phase lesson).
Findings and adjudication, binding:

**A2.1 Module.** `src/nifty_quant/execution/capacity.py` (suite B's guess — it orchestrates
`costs.py`/`fills.py` and lives beside them). Suite A's `research.capacity` imports migrate via
alias, call sites untouched.

**A2.2 Ladder.** Canonical name `cost_participation_ladder` (suite B), aliased as `cost_ladder`
in suite A's imports. Signature satisfying both call conventions:

    cost_participation_ladder(gross_returns, turnover, orders=None, prices=None,
        bar_traded_value=None, tradable=None, *, day_offsets=None,
        capital_ref=1_000_000.0,
        cost_bps_grid=(0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0),
        participation_grid=(0.005, 0.01, 0.02, 0.05, 0.10)) -> list[LadderCell]

Cell fields: cost_bps, participation, net_sharpe, net_bps_per_day, turnover,
unfilled_fraction. Grid order cost-major (suite A pins it). Suite A also pins the arithmetic:
net_bps_per_day = mean(gross - cost_bps/1e4 * turnover) * 1e4, cell.turnover = mean(turnover).
Unfilled model: per-row desired order notional = |orders|*prices when orders are given, else
turnover * capital_ref (the tilt's existing Rs 1,000,000 capital convention, not a new
constant); fillable = participation * bar_traded_value; unfilled_fraction = pooled
max(0, desired - fillable) / desired, in [0, 1], never dropped from any cell.

**A2.3 Capacity functions.** In `execution.capacity`:

    capacity_from_adv(median_adv, typical_weight=None, max_participation=0.02)
        -> max_participation * median_adv, divided by typical_weight when given
           (suite A's formula) and per-name when not (suite B's).
    capacity_from_turnover(median_adv, typical_weight=None, max_participation=0.02,
        daily_turnover_fraction=None)
        -> capacity_from_adv(...) / daily_turnover_fraction   (AMENDMENT 1's formula)
    combined_capacity(...) -> NamedTuple(value, binding) with value =
        min(position, trade) and binding in {"position", "trade"} (AMENDMENT 1 point 2).

**A2.4 Crossover.** Both guesses stand, no conflict: property
`NSEIntradayEquityCosts.brokerage_crossover_notional` (suite A) is canonical, derived
`brokerage_flat / brokerage_pct`; module function `costs.brokerage_crossover_notional(model)`
(suite B) is a thin accessor delegating to it.

**A2.5 The ordering tests, replaced per AMENDMENT 1's own mandate.** BOTH suites still carry
the retracted `turnover_capacity < adv_capacity` assertion (suite B predates the amendment;
suite A's author flagged the tension in a comment but asserted the retracted ordering anyway).
AMENDMENT 1 explicitly lists what obligation 7 must assert instead; the lead replaces each
suite's ordering test with those three assertions, in that suite's own style: (1) the ratio
identity turnover_capacity/adv_capacity == 1/daily_turnover_fraction; (2) combined_capacity
reports min() plus WHICH constraint bound; (3) the direction regression — turnover_fraction
below 1 makes POSITION binding, above 1 makes TRADE binding. Suite B's
scales-inversely-with-turnover test is CONSISTENT with the amendment formula and stands
unchanged.
