# Spec: H2 — Cross-Sectional Overnight Reversal

**Status:** contract for TDD. Two independent test suites written from this document alone,
before any implementation exists.

## The hypothesis

Berkman, Koch, Tuttle & Zhang (2012), *Paying Attention*, and the wider overnight/intraday
literature: **overnight returns systematically REVERSE during the following trading day.**
Attention-driven and liquidity-demanding buyers push prices up at the open; the pressure unwinds
intraday. Documented on US equities; the overnight/intraday return decomposition has since become
a standard result.

Falsifiable claim for NSE: stocks with the **largest overnight gains** underperform intraday, and
stocks with the **largest overnight losses** outperform intraday, cross-sectionally across the
Nifty-100 universe.

## Why H2 is second

Same cost profile that made H1 worth testing first — **one round trip per name per day**, entering
at 09:16 and exiting at 15:20, so `NSEIntradayEquityCosts.round_trip_bps(1e5)` = **8.26452 bps**
and the 2x gate is **16.52904 bps**. Nothing in this repo has ever survived turnover; only
once-daily shapes have headroom.

But H2 is a **stronger statistical proposition than H1** for two reasons:
1. **Cross-sectional, so ~149 observations per day instead of 1** — roughly 278,000 observations
   over 2018-2025 versus H1's 1,867. H1 died partly on power (t = 1.95 on 1,867 points).
2. **Market-neutral by construction** (long the overnight losers, short the winners), so it is not
   a bet on Nifty drift. H1's index-level signal could not separate the two.

H2 also has a positive prior in this repo: `eod_overextension`, the only strategy here with an
**overnight** holding period, produced the single positive structural result — a **flat latency
profile** (net Sharpe -0.464 / -0.484 / -0.384 at 0/1/2-minute lag). Flat latency means the signal
is not bid-ask spread capture, which is the discriminator that killed everything else.

## Definitions

Repo rules apply: bars are LEFT-labelled; `ts` is int64 epoch-seconds UTC; sessions are IST dates.

- **Overnight return** for symbol `i` on session `d`:
  `r_on[i,d] = log(open[09:16, d] / close[15:20, d-1])`
  This is the ONE quantity permitted to span a session boundary — that is its purpose. Reuse
  `features.market.overnight_return`, which already exists and is `@causal`-guarded.
- **Intraday return** (the tradable leg): `r_id[i,d] = log(close[15:20, d] / open[09:16, d])`
- **09:16, never 09:15.** All 61 of 61 observed OHLC violations in 2024 sit in the 09:15 bar
  (`close > high`) from pre-open call-auction leakage.
- **15:20** is the repo's `square_off_time`, so open-to-close is genuinely flat-to-flat tradable.
- `d-1` is the **previous USABLE session for that symbol**, not the previous calendar day, and not
  a fixed row offset. Sessions are irregular (Muhurat ~60 bars). A symbol missing either
  checkpoint on `d` or the 15:20 close on `d-1` is DROPPED for that day and counted.

## Method — run it through `Lens`, do not rebuild it

Phase 2 built exactly this machinery. H2 must USE it rather than reimplement:

```python
# src/nifty_quant/research/hypotheses/h2_overnight_reversal.py

def build_overnight_feature(panel: Panel) -> Feature:
    """(n_rows, n_symbols) overnight return, broadcast to every row of its session.
    kind="return". Delegates to features.market.overnight_return."""

def run_h2(panel: Panel, *, start=None, end=None, horizon_mode: str = "session",
           cost_hurdle_bps: float | None = None, seed: int = 0) -> HypothesisVerdict: ...
```

`run_h2` constructs a `Lens`, builds the overnight feature, and calls `Lens.verdict()`. The six
kill criteria, the cost hurdle, the block-bootstrap overlap correction, per-year sign stability and
the concentration checks are ALREADY implemented and tested there — reusing them is the point, and
it also exercises `Lens` end-to-end on real data for the first time.

**Expected sign is NEGATIVE.** Reversal means high overnight return -> low intraday return, so the
top-minus-bottom spread should be negative. Criterion 1 uses `abs(spread)` (a negative
top-minus-bottom spread just means you trade it inverted), so the sign does not affect survival —
but `explain()` must state the observed direction, because a POSITIVE spread would mean overnight
*momentum*, a different hypothesis that happens to clear the same gate. Do not let a sign flip pass
silently as a confirmation of reversal.

## CLARIFICATIONS ADDED 2026-08-17 (found by a test author reading the real `Lens` code)

**1. `HypothesisVerdict.cost_hurdle_bps` holds the RAW 1x round-trip cost (~8.26452), not 2x.**
`Lens.verdict()` does `cost_hurdle_bps = exp_table.cost_hurdle_bps` and applies the multiplier
inline: `if abs(edge_bps) > 2 * cost_hurdle_bps`. So a test asserting the FIELD equals 16.52904
would fail a correct implementation. Assert the field against the 1x value resolved from
`NSEIntradayEquityCosts`, and assert the 2x GATE only via criterion 1's behaviour. My earlier
wording ("the 2x gate is 16.52904") described the gate, not the field — corrected here.

**2. `method="cross_sectional_rank"` is MANDATORY for H2, not a stylistic choice.**
`Lens.verdict`/`Lens.expectancy` default to `expanding_quantile` bucketing, which needs
`min_history` rows of prior data per column before it will assign any bucket. A once-per-session
cross-sectional signal can never accumulate that, so the default silently produces zero usable
buckets and every verdict comes back empty — indistinguishable from "no effect". `run_h2` must
pass `method="cross_sectional_rank"`, which ranks across symbols WITHIN each row and is therefore
causal by construction and available from the first session. This is the same class of silent
degeneracy that made `Lens.stability()`'s liquidity axis return all-NaN.

**3. The EXIT checkpoint must also be resolved by time label, not "last bar of session".**
The spec called this out for the entry side (09:16 vs the corrupt 09:15) but not the exit. An
implementation that takes the session's final bar instead of resolving `15:20` would silently use
a 15:21+ print if one exists. Test it: plant a `15:21` bar with a wildly different close and assert
the feature still uses the 15:20 close.

## Statistical requirements

- **Overlap**: with `horizon_mode="session"` there is one observation per symbol-day and the
  intraday windows do not overlap, so no block-bootstrap correction is needed — but `Lens` defaults
  to `block_bootstrap`, which is harmless and stricter. State which was used in `explain()`.
- **Survivorship must be reported, not buried.** The universe is a fixed CURRENT-DAY list, so
  pre-2021 results are survivorship-inflated. `run_h2` must emit `survivorship_report(...)
  .warning_line()` from `universe.static` into its output, and the per-year table exists partly so
  this bias is visible as a trend rather than hidden in a pooled number.
- **Degenerate slices return NaN and never raise** (same contract as H1): a day with fewer than
  `min_names` usable symbols yields NaN for that day, not 0.0. A NaN edge cannot clear the hurdle,
  so it fails closed.
- Holdout: last 12 months stay locked. `run_h2` accepts `start`/`end`; the caller passes the
  pre-holdout window.

## Required tests

Two INDEPENDENT suites, from this document, no implementation in existence:
- `tests/test_h2_deepseek.py`
- `tests/test_h2_luna.py`

Each must cover:

**Overnight return construction (highest risk):**
1. `r_on` uses previous session's 15:20 CLOSE and today's 09:16 OPEN — hand-computed.
2. 09:15 is never read; corrupting that bar (`close > high`) leaves `r_on` bit-identical.
3. The FIRST session in the panel has no prior close -> NaN, not dropped silently without a count.
4. `d-1` is the previous USABLE session, not the previous calendar day — insert a gap (weekend /
   holiday) and confirm it bridges correctly.
5. A ~60-bar Muhurat session with no 15:20 is handled: the NEXT day's `r_on` must not silently use
   a stale close.
6. Symbol-specific gaps: symbol A trades on day `d-1`, symbol B does not — B's `r_on` on `d` is
   NaN while A's is finite. Per-symbol, not per-panel.
7. NaN propagates and is never forward-filled.
8. The feature is broadcast to every row of its session and has `kind == "return"`.

**Reversal detection:**
9. A PLANTED reversal (`r_id = -k * r_on + noise`) is recovered with a NEGATIVE spread.
10. A planted MOMENTUM relationship yields a POSITIVE spread, and `explain()` reports the
    direction — this is the guard against reporting momentum as confirmed reversal.
11. Pure noise yields no significant spread and `survived is False`. Verified-driftless generator,
    AND confirm the chosen seed's realised t-stat is insignificant before asserting it — a
    driftless generator can still hand you an unlucky draw (a previous fixture produced
    t = +2.345 on pure noise at seed 20260817).
12. A planted reversal BELOW the cost hurdle is KILLED despite being statistically significant.

**Integration with `Lens`:**
13. `run_h2` returns a `HypothesisVerdict` with all six criteria reported, even after one fails.
14. Criterion 5 is `NOT_EVALUATED` without a latency profile and NEVER silently PASS.
15. `run_h2` does not reimplement expectancy maths — assert its table equals a direct
    `Lens.expectancy(...)` call with equivalent arguments.

**Reporting and hygiene:**
16. Output contains the survivorship warning line.
17. Per-year table partitions sessions exactly once; a one-year-only effect shows low
    `years_sign_consistent`.
18. `start`/`end` restrict the window; a holdout-excluding call sees strictly fewer sessions.
19. Determinism: identical inputs, identical verdict; seed recorded in output.
20. Degenerate day (fewer than `min_names` usable symbols) -> NaN, no raise.

## Constraints

- Reuse `features.market.overnight_return`, `research.lens.Lens`, `research.expectancy`. Do NOT
  reimplement any of their maths — a test asserts equality against the underlying call.
- Vectorized numpy; loops over sessions/years/symbols acceptable, over bars not.
- float64 in motion; never accumulate returns in float32.
- Never modify `data/`. 100% line and branch coverage measured with pragma exclusion DISABLED.
- Fully annotated; `ruff` clean, line length 100.
