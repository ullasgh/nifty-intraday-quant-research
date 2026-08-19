# Spec: parameterised tilt backtest wrapper (`nq tilt`)

**Status:** contract for TDD. Two independent test suites written from this document ALONE,
before any implementation exists.

## What the user asked for

> "I should be able to test for the days I want, time I want and capital I want, so that if I
> give input the wrapper should run and give a neat table on how the strategy will work."

Today the long-only tilt — the program's only candidate that clears costs — exists ONLY inside
throwaway reconnaissance scripts (`scripts/recon_low_turnover_tilt.py`,
`recon_tilt_significance.py`, `recon_tilt_liquidity.py`) with hardcoded windows, hardcoded
09:16/15:20 checkpoints, and a hardcoded Rs 1L clip. There is no way to ask "what does this do
over MY dates, at MY times, with MY capital". This spec builds that.

## Two deliverables

### 1. `src/nifty_quant/research/tilt.py` — the library unit

The simulation logic lifted out of the recon scripts into a tested module. The scripts stay as
they are (they are the audit trail for published numbers); this is the reusable path.

```python
@dataclass(frozen=True)
class TiltConfig:
    start: datetime.date
    end: datetime.date
    entry_hhmm: str = "09:16"      # resolved BY TIME LABEL, never positionally
    exit_hhmm: str = "15:20"
    capital: float = 1_000_000.0   # TOTAL book size in rupees
    tilt: str = "mild"             # "mild" | "aggressive"
    smoothing: float = 0.10        # w_t = (1-a)*w_{t-1} + a*target_t; 1.0 = daily rebalance
    rebalance_every: int = 1       # hold the book k sessions between recomputes
    universe: str = "all_equity"
    continuous_only: bool = False  # restrict to names with coverage across the whole window
    seed: int = 0

@dataclass(frozen=True)
class TiltYearRow:
    year: int
    n_sessions: int
    gross_bps: float          # mean daily index-relative excess, before cost
    turnover: float           # mean daily sum of |weight change|
    cost_bps: float           # mean daily cost = round_trip_bps(clip) * turnover
    net_bps: float            # gross_bps - cost_bps
    ann_net_pct: float        # net_bps annualised, using the year's own session count

@dataclass(frozen=True)
class TiltResult:
    config: TiltConfig
    per_year: tuple[TiltYearRow, ...]
    total: TiltYearRow                 # year = -1 sentinel meaning "all"
    clip_per_name: float               # capital / typical n_held
    round_trip_bps: float              # at clip_per_name
    breakeven_turnover: float          # gross_bps / round_trip_bps; inf if gross <= 0
    n_symbols: int
    warnings: tuple[str, ...]
    def to_table(self) -> str: ...     # the "neat table"
    def explain(self) -> str: ...      # provenance: config, universe, holdout statement

def run_tilt(panel: Panel, config: TiltConfig) -> TiltResult: ...
```

### 2. `nq tilt` — the CLI command

Added to the existing Typer app in `src/nifty_quant/cli.py`, matching the conventions of
`nq backtest` (read it first — option naming, `_parse_date`, `_fail`, universe loading).

    nq tilt --start 2024-01-01 --end 2025-07-31 \
            --entry 09:16 --exit 15:20 \
            --capital 1000000 \
            --tilt mild --smoothing 0.10

Every `TiltConfig` field gets a flag. Defaults match the dataclass.

## THE CAPITAL INPUT IS THE POINT — get this right

Capital is not cosmetic. **Cost is size-dependent**, and that dependence is what decides whether
this strategy works:

    NSEIntradayEquityCosts().round_trip_bps(clip):
        Rs 1L   -> 8.26452 bps
        Rs 10L  -> 4.01652 bps
        Rs 1Cr  -> 3.59172 bps

So `capital` -> `clip_per_name = capital / n_held` -> `round_trip_bps(clip_per_name)` ->
`cost_bps = round_trip_bps * turnover`. A user raising capital from Rs 1L to Rs 10L roughly
HALVES their cost per unit turnover, and the tool must show that.

**ONE LEG, not two.** This is a long-only book; it buys and sells one side. Charge
`round_trip_bps(clip)` once per unit of turnover — NOT `2 *`. The 2x hurdle used throughout the
rest of this repo is the TWO-leg break-even identity for a long/short spread and does not apply
here. Getting this wrong doubles the cost and inverts the conclusion.

**Cost is charged on TURNOVER, not on notional.** A book that does not trade pays nothing.

## Required behaviour

1. **Time labels, not positions.** `entry_hhmm` / `exit_hhmm` resolve via `panel.minute_of_day()`.
   A session lacking either checkpoint DROPS OUT and is counted in `warnings` — never filled,
   never approximated by a neighbouring bar. NEVER assume 375 bars/session.
2. **Signal is H2's overnight return**, known at the entry checkpoint, cross-sectionally
   demeaned. Reuse `build_overnight_feature` and `_build_checkpoint_panel` from
   `h2_overnight_reversal.py`; do NOT reimplement.
3. **Weights are long-only**: non-negative, summing to 1, no shorts. `mild` = clipped-rank
   (zero on the top half by overnight return, rising toward the biggest loser); `aggressive` =
   bottom quintile equal-weighted.
4. **Benchmark is the equal-weight universe** on the same sessions. Reported figures are EXCESS
   over that benchmark.
5. **Smoothing** `w_t = (1 - a) * w_{t-1} + a * target_t`, with `a = 1.0` meaning daily
   rebalance. `rebalance_every = k` holds the book k sessions between recomputes; on held
   sessions the book drifts with prices and turnover is only that drift.
6. **A session with fewer than 5 valid names is SKIPPED** and counted — `cross_sectional_rank`
   returns all-NaN below `min_names=5`, so anything less produces silent zeros.
7. **The holdout is protected.** If `end` falls inside the locked holdout window, RAISE with a
   message naming the boundary. This tool must not be the way the holdout gets spent by
   accident.
8. Degenerate windows (no usable session, one symbol, `start > end`) raise `ValueError` naming
   the problem — never return a table of zeros.

## The table

`to_table()` returns something a person can read at a glance:

    Tilt backtest  mild / smoothing 0.10 / capital Rs 10,00,000
    Universe all_equity (149 names)   2024-01-01 .. 2025-07-31   389 sessions
    Clip per name Rs 66,667   round-trip 4.55 bps   breakeven turnover 1.23

    year   sessions   gross_bps   turnover   cost_bps   net_bps   ann_net%
    2024        246        4.12      0.110       0.50      3.62      9.13
    2025        143        3.05      0.108       0.49      2.56      6.45
    ------------------------------------------------------------------
    ALL         389        3.73      0.109       0.50      3.23      8.14

Numbers are illustrative of FORMAT only — the tests must not assert these values.

`explain()` states the universe, window, holdout boundary, the one-leg cost convention, and any
warnings (skipped sessions and why).

## Required tests

Two INDEPENDENT suites: `tests/test_tilt_a.py`, `tests/test_tilt_b.py`.

1. **Capital changes the answer, in the right direction.** Same window and dates at Rs 1L vs
   Rs 10L: `round_trip_bps` FALLS, `cost_bps` falls, `net_bps` RISES. Assert the direction and
   that gross is UNCHANGED — capital must not touch the signal.
2. **One leg, not two.** Assert `cost_bps == round_trip_bps(clip) * turnover` to float
   tolerance. A 2x implementation fails this. This is the single most important test here.
3. **Custom times are honoured.** A panel with a planted price move between 10:00 and 14:00
   gives a different result for `--entry 10:00 --exit 14:00` than for the 09:16/15:20 default,
   and both resolve BY LABEL — plant a decoy bar one minute off and assert it is not read.
4. **A session missing the entry or exit checkpoint drops out**, is counted in `warnings`, and
   is NOT filled. Include a ~60-bar Muhurat session with no 15:20 at all.
5. **Date range is respected**: sessions outside `[start, end]` never contribute; per-year rows
   partition sessions exactly once (sum of `n_sessions` equals the total).
6. **Smoothing reduces turnover**: `a=0.10` gives materially lower turnover than `a=1.0` on the
   same panel; `a=1.0` reproduces the daily-rebalance case.
7. **Weights are long-only and normalised**: never negative, sum to 1 within tolerance, on
   every session.
8. **Holdout protection**: an `end` inside the locked window RAISES, with the boundary in the
   message.
9. **Degenerate inputs raise** with a message identifying the problem (`start > end`, no usable
   session, fewer than 5 valid names throughout).
10. **Determinism**: same config and seed gives byte-identical `to_table()`.
11. **`to_table()` contains the per-year rows and an ALL row**, and the ALL row's session count
    equals the sum of the per-year counts.
12. CLI smoke: `nq tilt` with explicit flags exits 0 and prints a table; a bad `--tilt` value
    exits non-zero naming the valid options.

## Constraints

100% line+branch coverage with pragma exclusion DISABLED. Fully annotated; ruff clean;
mypy adds no new errors. float64 for all P&L and return accumulation; float32 only at rest.
NaN means "no bar" and is never forward-filled. `present` and `tradable` stay DISTINCT.
Never touch `data/`. Reuse `NSEIntradayEquityCosts`, `build_overnight_feature`,
`_build_checkpoint_panel`, and `cross_sectional_rank` rather than reimplementing any of them.
