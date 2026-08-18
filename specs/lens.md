# Spec: `nifty_quant.research.lens` — the research facade

**Status:** contract for TDD. Depends on `research/expectancy.py` (done),
`features/market.py` and `research/cv.py` (in progress). LOWEST priority of the three —
Phase 3 can call those modules directly if this is not ready.

## Why this exists

You asked for heavy wrapping so the codebase is easy to debug. This is where it pays off, and
it is deliberately the ONLY place a wrapper layer is added — the numeric kernels in
`features/core.py` and `features/market.py` stay plain vectorized functions, because a class
layer there would buy nothing and cost speed.

The problem it solves is specific and this repo has been burned by it repeatedly: **a plausible
float with no context attached.** Concrete instances from this session alone — a Sharpe ratio
that was net of slippage but labelled "gross"; a `pbo_cscv` that returned `nan` for two years
because it was silently fed the wrong artifact; an audit that reported 8,217 findings because a
threshold was reasoned rather than measured; a coverage number collected from a `--cov` path
that measured nothing.

Every one of those was a number that looked fine. `Lens` exists so that a research result
carries its own provenance and cannot be read without it.

## Design rules

1. **One entry point.** `Lens` owns the panel, the universe, the calendar and the cost model, so
   a caller cannot get session bounding, causal bucketing, or the cost hurdle wrong
   independently of each other.
2. **No bare floats or tuples cross the boundary.** Every result is a typed object with
   `.explain()`.
3. **`Lens` computes nothing itself.** It delegates to `expectancy`, `market`, `cv` and
   `execution.costs`. It is a composition and provenance layer, not a second implementation —
   duplicating maths here is the failure mode to avoid.

## Public API

```python
class Lens:
    def __init__(self, panel: Panel, *, universe=None, cost_model=None, seed: int = 0) -> None:
        """Holds panel, day_offsets, minute_of_day, symbols, cost model. Derives nothing
        eagerly; features are computed on demand and memoized by (name, params)."""

    # feature access — returns typed handles, never raw arrays
    def feature(self, name: str, **params) -> Feature: ...
    def available_features(self) -> tuple[str, ...]: ...

    # the research question
    def expectancy(self, feature: str | Feature, horizon: int, **kw) -> ExpectancyTable: ...
    def double_sort(self, a, b, horizon: int, **kw) -> SortResult: ...
    def stability(self, feature, horizon: int) -> StabilityReport: ...
        """Runs by_year / by_time_of_day / by_liquidity_decile together and reports whether the
        sign is stable across >= 6 of 8 years — the Phase 3 kill criterion."""

    def verdict(self, hypothesis_id: str, feature, horizon: int, **kw) -> HypothesisVerdict: ...
        """Applies ALL the Phase 3 kill criteria in one call and returns SURVIVED / KILLED with
        the reason. See below."""


@dataclass(frozen=True)
class Feature:
    name: str
    values: np.ndarray          # (n_rows, n_symbols) or (n_rows,) float64
    kind: Literal["return", "level", "ratio", "count", "market"]
    warmup_bars: int
    params: Mapping[str, Any]
    def explain(self) -> str: ...
# `kind` exists so a LEVEL can never be silently bucketed as a RETURN. Lens raises on a
# kind/usage mismatch rather than producing a meaningless number.


@dataclass(frozen=True)
class HypothesisVerdict:
    hypothesis_id: str
    survived: bool
    reasons: tuple[str, ...]      # one line per criterion, PASS or FAIL, always all of them
    expectancy: ExpectancyTable
    stability: StabilityReport
    cost_hurdle_bps: float
    def explain(self) -> str: ...
    def to_markdown(self) -> str:
        """The committed `results/hypotheses/<id>/verdict.md` body."""
```

## Kill criteria (encoded, not remembered)

`verdict()` applies these and reports each explicitly, PASS or FAIL, whether or not an earlier
one already failed — a verdict that short-circuits hides which criteria a near-miss cleared:

1. Mean conditional edge > **2x** the round-trip cost hurdle at that horizon.
2. Sign stable in >= **6 of 8** years (2018-2025).
3. Survives the overlap correction (block-bootstrap t-stat, never naive).
4. Not concentrated in the bottom liquidity decile or a single time-of-day bucket.
5. Latency profile flat across 0/1/2-minute decision lag (a signal that dies at one minute is
   spread capture, not alpha — this repo's established discriminator).
6. Deflated Sharpe accounts for the number of hypotheses tested (`effective_n_trials`).

Criterion 5 needs a backtest, so `verdict()` accepts an optional pre-computed latency profile
and reports criterion 5 as `NOT_EVALUATED` when absent. It must never silently score it as PASS.

### HOW TO BUILD A LATENCY PROFILE — corrected 2026-08-17 after I got this wrong

**Hold the SIGNAL fixed and vary ONLY the execution time.** I originally tested H2's latency by
moving the entry time and recomputing the signal at each new time. That measures how fast a signal
DECAYS, which is a different question from whether it survives EXECUTION DELAY — and it produced a
false kill.

Measured on H2, same data, both ways:

    signal recomputed at each entry (WRONG):   09:16 -24.58   09:20 -15.03   09:30  -9.40
    signal fixed at 09:16, execution delayed:  09:16 -24.58   09:17 -19.30   09:20 -16.26

The wrong method showed a 39% collapse in four minutes and looked exactly like bid-ask bounce. The
correct method shows a 21% drop at a one-minute lag with the edge still clearing the cost hurdle
(-19.30 vs 16.53), i.e. NOT microstructure. I had declared H2 dead on the strength of the wrong
test and had to retract it.

**Procedure:**
1. Compute the signal ONCE, at the decision timestamp.
2. Re-run the P&L with execution at `t+1`, `t+2`, `t+3` bars, signal unchanged.
3. Report all lags. A signal that dies between lag 0 and lag 1 is spread capture; one that decays
   gracefully is not.

Lag 0 is diagnostic only — it is unattainable, since it means transacting at the same print that
generated the signal. **The lag-1 number is the one that matters**, and it is what the backtest
engine's mandatory one-bar execution lag already enforces.

## CRITERION 7 — RECENT-YEARS COST GATE (added 2026-08-17; prose promoted to a check)

**This is a NEW, seventh kill criterion and it must be implemented.** The rule below was written
as advisory prose, and H2 proved prose does not run: its formal verdict reported

    1. Edge criterion:   PASS (-24.30 bps vs 16.53 gate)
    2. Sign stability:   PASS (8/8 years)
    4. Concentration:    FAIL   <- the only failure

while its 2024 (-10.98) and 2025 (-9.62) edges BOTH sit under the 16.53 gate. Criterion 2 counts
the SIGN of each year's edge, not its MAGNITUDE, so a monotonically decaying edge passes it 8/8
right up to the point of being worthless. **The single fact that makes H2 untradable today was
invisible to all six criteria.** Had concentration not independently failed, H2 would have been
recorded as a survivor.

Definition:

    criterion 7 = mean edge over the LAST TWO complete years in the sample
                  must exceed 2 * cost_hurdle_bps, in magnitude, with the dominant sign

- Report the two years used, their individual edges, and their mean, so the number is auditable.
- Fewer than two complete years of data -> `NOT_EVALUATED`, never a silent PASS.
- A year with fewer than ~20 usable sessions is not "complete" — skip it and say so.
- This criterion is deliberately REDUNDANT with criterion 1 on a stable edge, and deliberately
  DIVERGENT on a decaying one. That divergence is the whole point: criterion 1 answers "was there
  an edge", criterion 7 answers "is there one now".

Rationale, and why recent years are the RIGHT data rather than merely the latest: the universe is
a fixed CURRENT-DAY list, so survivorship inflates the EARLY years. The recent years are the
least-biased observations available, and correcting for survivorship would steepen an apparent
decay, never flatten it. A pooled statistic on this dataset systematically overstates what is
tradable today.

### ALSO REQUIRED: judge on RECENT years, not pooled

H2 taught this. Its pooled edge at lag 1 clears the hurdle (-19.30 vs 16.53) while 2024 (-7.8) and
2025 (-6.0) do NOT. The effect was real and tradable and has been arbitraged below costs.

Survivorship makes this worse, not better: the universe is a fixed CURRENT-DAY list, so the EARLY
years are the inflated ones and the recent years are the least-biased data available. A pooled
average on this dataset systematically overstates what is tradable today. Every verdict must
therefore report the per-year series, and a hypothesis whose last two years fail the cost gate is
NOT a survivor regardless of its pooled statistic.

## Required tests (`tests/test_lens.py`)

1-4. Construction: panel/universe/calendar wiring, `available_features` non-empty, memoization
     returns the identical object, distinct params give distinct entries.
5-8. `Feature`: `kind` recorded; `warmup_bars` propagated; **a `level` feature passed where a
     `return` is required raises** (name the exception and message); `explain()` names the
     params.
9-12. Delegation: `Lens.expectancy` returns the same table as calling
     `expectancy.conditional_expectancy` directly with equivalent arguments — this is the test
     that stops `Lens` from re-implementing the maths.
13-16. `stability()`: aggregates all three decompositions; the >= 6-of-8-years rule; a
     one-year-only effect is reported unstable; partial-year data handled.
17-24. `verdict()`: each of the 6 criteria PASSes and FAILs independently (parametrized); ALL
     criteria are reported even after one fails; criterion 5 is `NOT_EVALUATED` without a
     latency profile and **never silently PASS**; `survived` is True only when every evaluated
     criterion passes.
25-28. `explain()` / `to_markdown()`: contain the hypothesis id, every criterion with its
     verdict, the cost hurdle, the SE method, and an explicit SURVIVED/KILLED line.
29-30. Determinism: identical inputs give identical verdicts; the seed is recorded in output.

## Constraints

- Delegate, never duplicate. A test asserts equality with the underlying module's own output.
- No new thresholds (CLAUDE.md rule 8) — the six criteria above are the agreed ones and their
  numbers come from measured values (8.26 bps round trip, 2x break-even accounting identity for
  two-leg spreads, 6-of-8 years).
- 100% line and branch coverage; full annotation; `ruff` clean.
