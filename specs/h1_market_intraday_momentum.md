# Spec: H1 — Market Intraday Momentum

**Status:** contract for TDD. Two independent test suites are written from this document alone,
before any implementation exists.

## The hypothesis

Gao, Han, Li & Zhou (2018), *Market Intraday Momentum*: **the first half-hour return of the
market index predicts the last half-hour return of the same day.** Documented on SPY since 1993
and replicated in several international markets. The proposed mechanism is late-day rebalancing
by traders who infer the day's direction from the open, plus infrequent-rebalancing flows.

Falsifiable claim for NSE: `sign(r_first)` and magnitude carry information about `r_last` on the
NIFTY index, beyond what the intervening day's return explains.

## Why this is H1 (ranked first of five)

**Cost survival, which is the binding constraint in this repo.** It trades ONCE per day: enter at
the start of the last half hour, exit at square-off. One round trip against a measured **8.3 bps**
cost at Rs 1,00,000 notional (`NSEIntradayEquityCosts.round_trip_bps`). Every strategy this repo
has tested died on turnover — `volume_breakout` at 21,708 trades/year on 8 symbols, `carver_trend`
with a ~16 Sharpe-point raw-to-net gap. A one-trade-a-day signal is the only shape with real
headroom.

It is also cheap to test: one observation per session, ~2,000 rows over 2018-2025, versus 121M
bars for a cross-sectional hypothesis.

## Definitions (exact, and they matter)

Sessions and bar labels follow the repo rules. Bars are LEFT-labelled: the bar labelled `T`
covers `[T, T+60s)`.

- **First half hour**: `r_first = log(close[09:45] / open[09:16])`.
  **Start at 09:16, NOT 09:15.** All 61 of 61 observed OHLC violations in 2024 occur at the
  09:15 bar, always `close > high`, because the pre-open call auction leaks into it
  (`nifty-dataset-facts`). 09:15 is unusable.
- **Last half hour**: `r_last = log(close[15:20] / open[14:50])`. 15:20 is the repo's
  `square_off_time`, so this is the last window actually tradable flat-to-flat.
- **Middle**: `r_mid = log(open[14:50] / close[09:45])` — needed for the confound test below.
- A session missing ANY of those four checkpoints is DROPPED, not interpolated. NaN means "no
  bar" and is never forward-filled. Report the drop count.
- **Never assume a 375-bar session.** Resolve checkpoints by time-of-day label via
  `Panel.rows_at_time` / `day_offsets`, never by row arithmetic. Muhurat sessions are ~60 bars
  and will not contain 14:50 at all — they must drop out naturally, and a test must confirm that.

## Public API

```python
# src/nifty_quant/research/hypotheses/h1_market_intraday_momentum.py

@dataclass(frozen=True)
class SessionObservations:
    dates: np.ndarray            # object array of datetime.date, ascending, unique
    r_first: np.ndarray          # float64
    r_mid: np.ndarray            # float64
    r_last: np.ndarray           # float64
    n_sessions_dropped: int
    drop_reasons: Mapping[str, int]   # e.g. {"missing_09:16": 3, "missing_14:50": 60}
    def explain(self) -> str: ...


def extract_session_observations(panel: Panel, symbol: str) -> SessionObservations: ...


@dataclass(frozen=True)
class H1Result:
    symbol: str
    n_sessions: int
    beta: float                  # OLS slope of r_last on r_first
    t_stat: float
    r_squared: float
    mean_edge_bps: float         # mean |r_last| earned by the sign(r_first) rule, in bps
    hit_rate: float              # fraction of sessions where sign(r_last) == sign(r_first)
    cost_hurdle_bps: float
    survives_costs: bool         # mean_edge_bps > 2 * cost_hurdle_bps
    by_year: Mapping[int, float] # per-year mean edge in bps
    years_sign_consistent: int
    beta_controlling_for_mid: float   # slope on r_first with r_mid included
    t_stat_controlling_for_mid: float
    def explain(self) -> str: ...
    def to_markdown(self) -> str: ...


def run_h1(panel: Panel, symbol: str = "NIFTY50", *,
           start: date | None = None, end: date | None = None,
           cost_hurdle_bps: float | None = None) -> H1Result: ...
# `start`/`end` mirror `Panel.sub` and exist so the caller can pass the PRE-HOLDOUT window.
# (AMENDED 2026-08-17: the first draft demanded a date range in prose but omitted it from this
# signature block. A test author flagged the inconsistency rather than guessing silently.)
```

**Cost hurdle — use the resolved value, not a rounded one.** AMENDED 2026-08-17: this spec said
"8.3 bps". The actual value is `NSEIntradayEquityCosts().round_trip_bps(1e5) == 8.26452`, so the
2x gate is **16.52904**, not 16.6. Tests must assert against the value resolved FROM the cost
model, never a literal — otherwise a future fee change silently invalidates them.

## Statistical requirements

- **One observation per session, so the observations do NOT overlap.** A naive OLS t-stat is
  therefore appropriate here — no block bootstrap needed. **State this in the docstring**, because
  everywhere else in this repo overlapping samples require correction, and a reader must be able
  to see why this case is exempt rather than assume it was forgotten.
- `float64` throughout. Never accumulate returns in float32.
- **The confound test is mandatory.** `r_last` could be predicted by the whole day's drift rather
  than specifically the first half hour. Report `beta_controlling_for_mid` from the bivariate
  regression `r_last ~ a + b1*r_first + b2*r_mid`. If `b1` collapses once `r_mid` is included, the
  effect is day-level drift, not intraday momentum, and the hypothesis is falsified in its stated
  form. This is not optional decoration — it is the difference between a finding and an artifact.
- **The trading rule for `mean_edge_bps`** is: at 14:50 go long if `r_first > 0`, short if
  `r_first < 0`, flat if exactly zero; exit at 15:20. `mean_edge_bps = 1e4 * mean(sign(r_first) *
  r_last)`. This is the quantity that must clear costs — NOT `beta`, NOT `r_squared`. A
  statistically significant beta with a sub-cost edge is a kill.
### AMBIGUITIES RESOLVED 2026-08-17 (both test suites disagreed here — these are now binding)

Two independent test authors read the following differently, which means the spec was genuinely
under-specified. Resolved as follows; both suites and the implementation must match.

**`hit_rate` — zero `r_first` sessions are excluded from BOTH numerator and denominator.**
The rule is "long if `r_first > 0`, short if `r_first < 0`, FLAT if exactly zero". A flat day is
not a trade, so it cannot be a hit or a miss. `hit_rate` therefore answers "of the sessions we
actually traded, how often was the direction right":

    traded = r_first != 0
    hit_rate = mean(sign(r_last[traded]) == sign(r_first[traded]))

With 4 sessions of which 1 has `r_first == 0` and 2 of the remaining 3 agree in sign, the answer
is **2/3**, not 2/4. If no session trades, `hit_rate` is `nan`, not 0.0 — zero would falsely read
as "traded and always wrong".

**`years_sign_consistent` — count years matching the MODAL NON-ZERO sign, not the sign of the
pooled mean.** Procedure, exactly:
1. Compute each year's mean edge, and its sign.
2. Discard years whose mean edge is exactly zero (they vote for neither).
3. `dominant_sign` = the sign held by the most remaining years. On an exact tie, `"mixed"`.
4. `years_sign_consistent` = the count of years holding `dominant_sign`.

Counting years, NOT weighting by magnitude, is deliberate: it is what makes the criterion robust
to a single dominant year. H1's own reconnaissance is the illustration — its pooled mean edge is
positive only because 2022 contributed +4.11 bps while five of eight years were negative. A
magnitude-weighted rule would have called that "consistent"; a year-count rule correctly calls it
5-of-8 for the NEGATIVE sign and fails the >= 6 gate.

**`drop_reasons` — a session missing several checkpoints increments ONE key**, the first missing
checkpoint in the order `09:16, 09:45, 14:50, 15:20`. So `sum(drop_reasons.values()) ==
n_sessions_dropped` exactly, which is an assertable invariant.

**DEGENERATE REGRESSIONS RETURN NaN, THEY NEVER RAISE.** (Added 2026-08-17 after a test author
noticed the all-zero-`r_first` case leaves `beta`/`t_stat` undefined.) Any statistic whose
regression is not identified must be `nan`:
- constant (zero-variance) `r_first` -> `beta`, `t_stat`, `r_squared` are all `nan`
- constant `r_mid` -> `beta_controlling_for_mid` and its `t_stat` are `nan` (the bivariate design
  matrix is singular; do NOT let `numpy.linalg.LinAlgError` escape)
- fewer than 3 usable sessions -> every regression statistic is `nan`
- a YEAR with fewer than 3 sessions -> that year's entry is `nan`, and it is DISCARDED from the
  `years_sign_consistent` vote exactly like a zero-edge year

Rationale: this is a research harness that will be pointed at 149 symbols and 8 years of real
data, where some slice will inevitably be degenerate. An exception aborts the entire run and
loses the other 148 symbols; a NaN is visible, propagates honestly, and is caught by the kill
criteria (a NaN edge cannot exceed the cost hurdle, so it fails closed). **Failing closed on
NaN, and never silently substituting 0.0, is the requirement** — a zero edge would read as
"measured and found flat" rather than "not measurable".

- Kill criteria, reported individually and all of them even when one fails:
  1. `mean_edge_bps > 2 * cost_hurdle_bps` (default hurdle 8.3, so > 16.6 bps)
  2. sign of yearly mean edge consistent in >= 6 of 8 years (2018-2025)
  3. `abs(t_stat) > 1.96`
  4. `b1` survives the `r_mid` control (retains >= 50% of its univariate magnitude AND stays
     significant)
- Holdout: the last 12 months stay locked via `research.splits.HoldoutLock`. Do NOT read them.
  `run_h1` must accept a date range and the caller must pass the pre-holdout window.

## Required tests

Two INDEPENDENT suites, written from this document with no implementation in existence:
- `tests/test_h1_deepseek.py`
- `tests/test_h1_luna.py`

Each must cover, at minimum:

**Checkpoint extraction (the highest-risk area):**
1. `r_first` uses the 09:16 OPEN and the 09:45 CLOSE — hand-computed on a synthetic session.
2. 09:15 is never referenced. A session whose 09:15 bar is deliberately corrupted
   (`close > high`, the real observed defect) must produce an unchanged `r_first`.
3. `r_last` uses the 14:50 OPEN and the 15:20 CLOSE.
4. `r_mid` bridges 09:45 close to 14:50 open, so `r_first + r_mid + r_last` equals the
   09:16-open-to-15:20-close return, to float64 tolerance.
5. A session missing 14:50 is dropped and counted, not interpolated.
6. A ~60-bar Muhurat session (no 14:50) drops out naturally and appears in `drop_reasons`.
7. NaN at a required checkpoint drops the session; NO forward-fill.
8. Checkpoints are resolved by time label, not row offset — a session with a LATE start or a
   mid-session gap still resolves correctly.
9. `dates` is ascending and unique; arrays are all the same length.

**Statistics:**
10. `beta` matches a hand-computed OLS slope on a small fixture.
11. `t_stat` sign matches `beta` sign.
12. A planted relationship (`r_last = 0.5 * r_first + noise`) is recovered within tolerance.
13. Pure-noise input yields `abs(t_stat) < 1.96` and `survives_costs is False`. Use a
    VERIFIED-DRIFTLESS generator and assert driftlessness in the test — this repo's shared
    fixture once baked an ~8x-dominant drift.

    **DRIFTLESSNESS IS NOT ENOUGH — the SEED must also be checked.** A test author's first draft
    used a driftless, mean-centred generator (seed 20260817, n=512, scale=0.003) that produces
    **corr = +0.1033, t = +2.345 on PURE NOISE** — a spurious significant result baked in as the
    expected behaviour. Verified directly. It was replaced with seed 4 (t = -0.016). So: after
    choosing a noise fixture, COMPUTE its t-stat and confirm it is insignificant before asserting
    that the implementation should find it insignificant. A driftless generator can still hand you
    an unlucky draw, and a test that encodes one asserts the opposite of what it intends.
14. `mean_edge_bps` equals `1e4 * mean(sign(r_first) * r_last)` exactly.
15. `hit_rate` counts sign agreement; a zero `r_first` is excluded from the rule.
16. `survives_costs` requires strictly more than 2x the hurdle; test just-below and just-above.
17. Default `cost_hurdle_bps` resolves to `NSEIntradayEquityCosts.round_trip_bps` (8.3 at Rs 1e5).

**The confound (mandatory):**
18. When `r_last` is generated purely from `r_mid` and NOT from `r_first`, the univariate `beta`
    may be non-zero but `beta_controlling_for_mid` must collapse toward zero.
19. When `r_last` is generated purely from `r_first`, `beta_controlling_for_mid` must survive.

**Stability and reporting:**
20. `by_year` partitions sessions exactly once each; keys are calendar years.
21. `years_sign_consistent` counts years whose mean edge shares the dominant sign.
22. An effect present in only one year is visible as low `years_sign_consistent`.
23. `explain()` / `to_markdown()` contain the symbol, n_sessions, every kill criterion with its
    verdict, the cost hurdle, and an explicit SURVIVED / KILLED line.
24. Determinism: identical input gives an identical result; the seed is recorded.

## Constraints

- Vectorized numpy/pandas; no per-bar Python loops. A loop over sessions or years is acceptable.
- Never modify anything under `data/` (rule 2).
- 100% line and branch coverage on the new module, measured with pragma exclusion DISABLED.
- Fully annotated (`disallow_untyped_defs = true`); `ruff` clean, line length 100.
- The implementation must NOT read the locked holdout window.
