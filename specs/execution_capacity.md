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
