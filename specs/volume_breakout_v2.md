# Spec F — rebuild volume_breakout and re-measure

Status: spec, written before implementation. Author: lead. Date 2026-08-21.

## The prior, stated up front so this is a fair test and not a rescue mission

`volume_breakout` v1 is DEAD with measured numbers: **gross Sharpe -0.048, net -0.233, 21,708
trades, 73.7% of desired notional unfilled.** It is the repo's reference regression result.

The program's meta-pattern across H1-H5 is that every effect with tradable magnitude in this
dataset is overnight, illiquid, or concentrated in the most survivorship-inflated year. Nothing
intraday and liquid has cleared costs. v2 is therefore **pre-registered with kill criteria declared
BEFORE it is built**, so a negative result is a result rather than a prompt to keep tuning.

## What was actually wrong with v1, and what changes

| v1 | v2 |
|---|---|
| `H > 0.55` binary, 390-bar window crossing the overnight gap with `day_offsets=None` | `hurst_on_stitched` — the gap is removed from the PATH, so the window may legitimately span sessions. Threshold from a measured null (rule 8), or dropped entirely if D5's IC says Hurst carries no monotone signal |
| binary `close > rolling_high_30` | `breakout_strength` in sigma units, continuous |
| `volume_z > 2.0`, hand-chosen | abnormal-volume percentile from its own measured null |
| `target_vol_ann` decorative (meta-dict only) | Phase A4 portfolio volatility targeting, with clip-breaks-target REPORTED not silently voided |
| no relative/idiosyncratic filter | beta-residual return vs NIFTY, so index-wide moves are not read as stock information |
| costs discovered after the fact | capacity and cost ladder run as PART of the test |

## The kill criteria, declared now

v2 is DEAD unless, on the **recent window** (never pooled — H2 is the worked example: sign-stable
8/8 years and a -24.30 bps pooled edge, yet killed on the recent-years gate as it decayed to -9.62):

1. Net Sharpe > 0 after the full NSE fee stack at a realistic participation cap.
2. `abs(spread_bps) > 2 * cost_hurdle_bps`.
3. `abs(spread_t) > 1.96` on the CORRECTED overlap-aware SE (the pre-fix SE gave a 34%
   false-positive rate; that machinery is now measured at 7%).
4. Deflated Sharpe clears at the MEASURED `effective_n_trials` — which must include Phase E's
   trials, because v2's components were selected using them. Pretending v2 is trial number one
   after a 132-trial sweep would be the exact multiple-testing dishonesty this program exists to
   avoid.
5. Fill realism: unfilled notional materially below v1's 73.7%. If v2 cannot be filled it is not a
   strategy, whatever its paper edge.

**Failing ANY of these kills it.** No re-tuning a threshold and re-running: that is how the
`adjustment_audit` shipped a reasoned-not-measured 0.35 cutoff and produced 8,217 false positives.

## What Phase E must supply first

v2 must not be assembled from primitives that Phase E measured as dead. Before implementation,
read the sweep's verdict for `breakout_strength`, `volume_zscore`, `hurst_on_stitched` and
`beta_residual_return`:

- If a component shows no monotone conditional response and no IC at any horizon, it does NOT go
  into v2. Including it anyway would be building on a measured null.
- `H > 0.55` in particular stops being an assumption: D5 measures `E[R | H-quantile]`, and Hurst
  is kept ONLY if that relationship is monotone.

This ordering is the point of running E before F.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. The strategy declares a `ResearchContract` and refuses to run without one.
2. Every threshold in the config has a recorded derivation from a measured null — a test asserts
   no bare numeric literal survives as a decision threshold.
3. `hurst_on_stitched` is used, NOT `rolling_hurst(day_offsets=None)`; a test asserts the
   gap-crossing call path is gone.
4. `breakout_strength` is used as a continuous input; `breakout_strength > 0` still matches the
   legacy boolean elementwise.
5. Volatility targeting is real: `sigma_portfolio` is estimated and the book is scaled by
   `sigma_target / sigma_portfolio`; a test asserts a clip that breaks the target is REPORTED.
6. `stop_loss_pct` / `target_pct` are gone — v1 declared them, shipped them in YAML, and ignored
   them. A test asserts they are absent from the params model.
7. The five kill criteria are evaluated on the recent window and each independently blocks
   promotion.
8. `make verify`'s volume_breakout 2024 regression result is unchanged — v2 is a NEW strategy, it
   does not overwrite v1's recorded numbers.


# VERDICT — KILLED 2026-08-31, at the component gate. Criteria 1-5 never ran.

Author: lead. This is the formal kill write-up this spec demanded of itself ("failing ANY of
these kills it"), applied one gate earlier than the five numbered criteria: the section "What
Phase E must supply first" forbids assembling v2 from components Phase E measured as dead, and
the definitive sweep (2026-08-30, 19 of 22 features genuinely measured on real OHLCV —
`results/PHASE_E_SWEEP.md`) returned a verdict on all four:

    component              best raw Sharpe (any horizon)   verdict against this spec's gate
    hurst_on_stitched      +0.0118                          no monotone conditional response ->
                                                            H-threshold has nothing to rest on
    beta_residual_return   -0.1066 (EOD); -0.8367 at h=1    measured NEGATIVE, worst in sweep
    breakout_strength      -0.0526 (best); -0.1817 at h=1   NEGATIVE at every horizon
    volume_zscore          +0.1080 at h=1, decaying          deflated Sharpe 0.0000 at the
                           monotonically with horizon        measured n_eff=18.76 (sr0=0.3697)

Every deflated Sharpe in the 114-trial matrix is 0.0000 (PBO 0.0002 — the sweep stably ranks
nothing). `volume_zscore` deserves the explicit note: its raw +0.1080 at h=1 is the largest
raw Sharpe in the sweep and is exactly the v1 shape — a short-horizon volume signal — whose
economics v1 already measured as dead at one minute of latency (gross -0.048, net -0.233,
73.7% unfilled). A component that (a) does not clear the multiple-testing bar and (b) repeats
a measured execution failure does not go in.

With zero of four components passing the gate, there is nothing to assemble; evaluating
criteria 1-5 would require building a strategy from measured nulls, which the gate exists to
prevent. Per the spec's own terms: **no re-tuning, no re-runs, no component swaps** — a new
volume hypothesis would be a NEW spec with its trial count charged on top of Phase E's 132
(n_eff 18.76), not a v2 revision.

v1's reference regression (obligation 8) is untouched and still guards `make verify`.
