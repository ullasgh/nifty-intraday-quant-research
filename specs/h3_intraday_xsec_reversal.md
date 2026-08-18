# Spec: H3 — Intraday Cross-Sectional Reversal on Beta-Residual Returns

**Status:** contract for TDD. Two independent test suites from this document alone, before any
implementation exists.

**The research answer is already known** — reconnaissance below is decisive. This module exists
for the AUDIT TRAIL and to close Phase 3's record, not to discover anything. Build it to the same
standard regardless: a verdict nobody can reproduce is not a verdict.

## The hypothesis

Short-horizon cross-sectional reversal (Lehmann 1990; Nagel 2012) as a liquidity-provision
premium: stocks that move most against their peers revert as liquidity is replenished. H2 found
this IS present in the overnight gap. H3 asks whether it is also present in an INTRADAY move.

Falsifiable claim: stocks whose morning residual move is most negative outperform intraday, and
those most positive underperform, over the remainder of the same session.

## Reconnaissance result (already measured, 149 symbols, 1,867 sessions, 260,665 obs)

    execute   spread bps       t     2018    2021    2024    2025
      10:00        +1.06    0.68    -13.1    +6.2    +6.6    -0.0
      10:01        +1.21    0.77    -12.8    +6.2    +6.8    +0.1
      10:05        +1.81    1.16    -11.6    +7.2    +7.0    -0.4

Fails on four counts: magnitude ~16x short of the 16.53 bps gate; t = 0.68; sign flips across
years; and the spread is POSITIVE — weak MOMENTUM, the opposite of the hypothesised reversal.
Latency-insensitive, which merely confirms there is nothing to decay.

**The implementation must reproduce approximately these numbers on the same window.** If it does
not, the implementation is wrong — this is the H2 lesson, where a formal run returned -0.001 bps
against a -24.58 bps reconnaissance because `horizon=1` measured one minute on a 373-bar session.

## Definitions

- **Signal** (per symbol, per session): `r_morning = log(close[10:00] / open[09:16])`,
  then cross-sectionally demeaned within the session -> the beta-residual proxy. Demeaning
  against the equal-weighted universe is the agreed neutralisation; a rolling per-symbol beta is
  NOT required and must not be silently substituted.
- **Forward leg** (the tradable return): `r_rest = log(close[15:20] / close[10:00])`,
  cross-sectionally demeaned -> market-neutral P&L.
- 09:16, not 09:15. 15:20 is `square_off_time`. Resolve ALL checkpoints BY TIME LABEL.
- One round trip per name per day: enter 10:00, exit 15:20. Hurdle
  `NSEIntradayEquityCosts().round_trip_bps(1e5)` = 8.26452, gate = 16.52904.

## Method — reuse H2's proven path

`h2_overnight_reversal.py` already solved every structural problem here. H3 must follow it:

```python
# src/nifty_quant/research/hypotheses/h3_intraday_xsec_reversal.py

def build_morning_residual_feature(panel: Panel) -> Feature: ...
def run_h3(panel, *, start=None, end=None, cost_hurdle_bps=None, seed=0) -> HypothesisVerdict: ...
```

**Mandatory, inherited from H2's failures:**
1. **Reduce the raw panel to a 2-rows-per-session checkpoint panel** (10:00 entry, 15:20 exit),
   resolving both BY TIME LABEL, before handing anything to `Lens`. Otherwise `horizon=1` measures
   one minute on a real 373-bar session. This is the single most important requirement here.
2. **`method="cross_sectional_rank"` is mandatory** — `Lens` defaults to `expanding_quantile`,
   which needs `min_history` rows per column and can never bucket a once-per-session signal,
   silently yielding empty verdicts indistinguishable from "no effect".
3. **`explain()` must STATE the observed direction.** Criterion 1 uses `abs(spread)`, so MOMENTUM
   clears the same gate as REVERSAL. H3's real answer is a positive (momentum-ish) spread, so a
   verdict that fails to name the direction would be actively misleading.
4. Emit `survivorship_report(...).warning_line()`. Degenerate slices return NaN, never raise,
   never 0.0. Accept `start`/`end`; do not read the holdout.
5. `HypothesisVerdict.cost_hurdle_bps` is the RAW 1x value; the 2x is applied inside criterion 1.

## RESOLVED 2026-08-18 — `build_morning_residual_feature` returns DEMEANED values

The spec left open whether the feature returns the raw morning return or the cross-sectionally
demeaned residual. **Both independent test authors read it as DEMEANED**, and both assert each
session row sums to ~0. That agreement settles it: the function performs the demeaning, and
`Feature.kind == "return"`.

Two corollaries both suites depend on:
- **No min-names floor inside the demean step.** A row with a single finite symbol demeans to
  `0.0`, not NaN. That is distinct from — and earlier than — `cross_sectional_rank`'s own
  `min_names=5` gate, which applies later during bucketing inside `Lens`. Conflating the two
  would make the per-symbol-NaN test meaningless.
- Demeaning is done in float64 even though the panel stores float32 at rest, so a row sum is
  ~0 to machine precision rather than to float32 precision.

## Required tests

Two INDEPENDENT suites: `tests/test_h3_deepseek.py`, `tests/test_h3_luna.py`. Neither author reads
the other's file.

**Checkpoint construction:**
1. `r_morning` uses the 09:16 OPEN and the 10:00 CLOSE — hand-computed.
2. `r_rest` uses the 10:00 CLOSE and the 15:20 CLOSE. Note BOTH legs share the 10:00 close, so
   `r_morning + r_rest` telescopes to `log(close[15:20]/open[09:16])` — assert that identity.
3. 09:15 is never read; corrupting it leaves the signal bit-identical.
4. Resolution is BY TIME LABEL: plant a 10:01 bar with a wild close and assert it is ignored.
5. A ~60-bar Muhurat session (no 10:00 or 15:20) drops out with a counted reason.
6. Per-SYMBOL NaN: symbol A finite while B is missing a checkpoint in the same session.
7. NaN propagates; no forward-fill.

**REALISTIC-SESSION FIXTURE — mandatory, this is the H2 regression:**
8. A panel with ~100-375 bars per session must give the SAME answer as its 2-bar checkpoint
   equivalent. Both H2 suites used only 2-bar fixtures and neither caught `horizon=1` measuring
   one minute. Mutation-verify by removing the reduction step and confirming this test fails.

**Cross-sectional behaviour:**
9. Demeaning makes each row's signal sum to ~0.
10. A planted REVERSAL gives a negative spread; a planted MOMENTUM gives a positive spread, and
    `explain()` names the direction in both cases.
11. Pure noise -> no significant spread, `survived is False`. Verified-driftless generator AND
    confirm the chosen seed's realised t-stat is insignificant before asserting it.
12. A planted effect below the cost gate is KILLED despite being statistically significant.

**Lens integration:**
13. All SEVEN criteria reported, even after one fails.
14. Criterion 5 `NOT_EVALUATED` without a latency profile; never a silent PASS.
15. Criterion 7 (recent-years gate) present and correctly evaluated.
16. `run_h3` does not reimplement expectancy maths — assert equality against a direct
    `Lens.expectancy(...)` call with equivalent arguments.

**Reporting:** survivorship line present; per-year table partitions sessions once; `start`/`end`
restrict the window; determinism.

## Constraints

Reuse `Lens`, `expectancy`, `features.core`. 100% line+branch with pragma exclusion DISABLED.
Fully annotated; ruff clean; never touch `data/`; never assume 375 bars/session.
