# Research-validity findings (Lens and expectancy layer)

Audit date 2026-08-20. These are CONFIRMED findings against the current tree, each with a
file:line. They are recorded here rather than fixed inline because several of them change what
published verdicts mean, and that adjudication is a lead decision, not an implementer's.

Numbering continues the engine series (F1-F13 in `engine_adversarial_invariants.md`) under an
`L` prefix for the research layer.

---

## L1 — `survived` silently EXCLUDES unevaluated criteria from the conjunction

`lens.py:901-909`:

    evaluated_results = [t for t in reason_tokens if t != "NOT_EVALUATED"]
    survived = all(r == "PASS" for r in evaluated_results)

A criterion whose input was never supplied is neither a pass nor a fail — it is dropped. So
`survived=True` is reachable with two of seven kill criteria never run, and
`tests/test_lens.py:819-836` asserts that as intended behaviour.

This is the single most consequential finding in the layer. A "kill criterion" that vanishes
when unsupplied is not a kill criterion; it is a comment. The `any_not_evaluated` flag and the
"(INCOMPLETE)" suffix are the only surviving signal, and neither is a gate.

**Adjudication (lead):** an unevaluated criterion must make the verdict INCONCLUSIVE, not
survived. A verdict object should not be able to say `survived=True` while admitting it did not
run part of the test. Deferred to the Phase E spec, because changing it re-opens every recorded
verdict and that has to be done deliberately, all at once.

## L2 — criteria 5 and 6 are unevaluated for EVERY production caller

Only two callers exist in `src/`: `h2_overnight_reversal.py:292` and
`h3_intraday_xsec_reversal.py:307`. Both pass only `method` and `cost_hurdle_bps`. No caller
passes `latency_profile` or `strategy_returns`, and `effective_n_trials` is left at 1.

Confirmed in committed output, not merely inferred: `results/hypotheses/H2/verdict.md:17-18` and
`H3/verdict.md:18-19` both record 5 and 6 as NOT_EVALUATED.

`effective_n_trials` IS computed on the walkforward path (`cli.py:1203`, `:1635`) and simply never
routed into the Lens. The multiple-testing deflation the program believes it applies has never
been applied to a hypothesis verdict.

Note the interaction with L1: because unevaluated criteria drop out, H2 and H3 both read as
survivors of a seven-criterion screen they passed five of.

## L3 — criterion 5 would be WRONG for the repo's only live signal family

`lens.py:785-786` computes the latency ratio such that a negative-signed edge hard-FAILs. H2's
measured edge is **-24.30 bps** — negative by construction, because it is a *reversal* signal that
is traded short-side.

So criterion 5 is not merely unevaluated; wiring it up naively would reject the overnight-reversal
family for having the sign it is supposed to have. The ratio has to be taken on the magnitude of
the edge relative to its own lag-0 value, preserving sign as a separate check.

This matters more than the other findings because latency is where this program's last strategy
died: `volume_breakout`'s edge does not survive one minute of latency. Criterion 5 is the criterion
that would have caught it, and it is both off and mis-specified.

## L4 — `Lens.universe` is stored and never read

Assigned at `lens.py:239`; grep finds reads nowhere (only 217/221/227/239, all assignment/plumbing).
Passing a restricted universe to a Lens silently has no effect, so any verdict that believed it was
universe-restricted was not.

## L5 — rule 8: exactly one of seven thresholds is compliant

| Criterion | Threshold | Derivation |
|---|---|---|
| 1 | `> 2 * cost_hurdle_bps` | argued identity, not measured |
| 2 | `n_years_sign_consistent >= 6` | asserted (`specs/lens.md:194-197`) |
| 3 | `abs(spread_t) > 1.96` | convention, no null measured |
| 4 | ratio > `4.8695` | **COMPLIANT** — 300 permutations, seed 42, derivation recorded at `lens.py:24-50` |
| 5 | ratios `>= 0.5` | undocumented |
| 6 | `dsr > DSR_SIGNIFICANCE = 0.95` | self-declared `# TODO(rule-8): unmeasured placeholder` |
| 7 | `> 2x` hurdle | same as 1 |

Also `var_trial_sharpes=1.0` at `lens.py:815` is a placeholder feeding the deflated Sharpe.

Criterion 4 is the standing example of what rule 8 asks for — and it is the one that was corrected
twice (5.5484 -> 4.8695) precisely because it was measured. The other six have never been tested
against a null and so have never had the chance to be found wrong.

## L6 — the published report mislabels which criteria are missing

`results/RESEARCH_REPORT.md:41` describes criteria 5 and 6 as "recent-year power, out-of-sample
holdout". They are actually latency profile and deflated Sharpe; the recent-year gate is criterion
7 and it IS evaluated. A reader of the report cannot tell which checks were skipped.

---

## Consequence for the tilt candidate

The tilt path does not call `Lens.verdict()` at all — its four conditions are a separate checklist
(`RESEARCH_REPORT.md:11`, `:230`), so the "2 of 4 cleared" claim is not directly overstated by L1.

But the tilt trades **H2's signal**, and H2's Lens verdict is INCOMPLETE in exactly the two places
that matter most for a candidate about to be taken to a holdout: it has never had a latency profile
measured, and it has never been deflated for the number of trials this program has actually run.

Both must close before the holdout is spent. That is a Phase I precondition, and it is now a
stronger precondition than the plan recorded.

---

## L7 — the block bootstrap draws ONE block per replicate

`expectancy.py:294-359`, replicate loop at `:339-350`. Each replicate draws a single contiguous
block of length `<= horizon` and appends it. There is no tiling loop reconstructing a series of
anything like the original length, so a "bootstrap replicate" is one short block rather than a
resampled series. `_compute_bucket_stats:456-461` then takes `nanmean` of that single padded block.

Measured against ground truth (empirical SE across 300-400 independent resimulations, 15,000 rows,
real `day_offsets`, calling the unmodified repo functions), the raw bootstrap SE is **34x to 48x**
the true SE.

## L8 — and the `n_effective` transform over-corrects, understating the final SE

This is the part I had wrong when I commissioned the audit. I assumed the bootstrap SE never
reached criterion 3 and that a naive iid SE was used instead. That is **refuted**: the bootstrap
`se_bps` (`:463`) does feed forward, through a second transform at `:470-471`:

    n_effective = n_obs / (std_bps / se_bps) ** 2

which is not the standard `n_eff = (std/se)**2` identity. The two defects compound in direction
even though they partially cancel in magnitude: L7 inflates the raw SE ~35-48x, and this transform
then over-corrects it into a SE that **understates** the truth, at 0.40x-0.71x of ground truth.

| construction | raw bootstrap SE / truth | SE as fed to `spread_t` | reported `spread_t` | true `spread_t` | inflation |
|---|---|---|---|---|---|
| AR(1) rho=0.6, h=5 | 47.9x | 0.71x | 146.6 | 105.3 | 1.39x |
| MA overlap, h=5 | 47.5x | 0.60x | 87.1 | 52.3 | 1.66x |
| MA overlap, h=10 | 34.1x | 0.40x | 59.0 | 24.1 | 2.45x |

**Consequence.** Criterion 3 gates at `abs(spread_t) > 1.96`. Dividing that threshold back through
the measured 1.4x-2.5x inflation puts the honest t-statistic of anything that *barely* cleared the
gate at roughly **0.8 to 1.4** — below conventional significance in every case tested. The
inflation worsens with horizon, so the longer-horizon results are the least trustworthy.

**Both must be fixed together.** Fixing only the one-block bootstrap would leave the bad
`n_effective` transform applied to a now-correct SE; fixing only the transform leaves it correcting
a 40x-inflated input. Re-derive `n_effective` after L7 is fixed, then re-evaluate every result that
passed criterion 3 marginally.

Note for the record: this finding came from an audit brief whose stated premise was wrong. The
agent checked the premise instead of accepting it, refuted it, and found the deeper defect. That is
the behaviour the dual-suite rule exists to produce, and it is worth more than agreement.

---

## L7/L8 — FIXED and VERIFIED, 2026-08-20

Measured on the SAME fixture, before and after the fix:

| quantity | pre-fix | post-fix | nominal / truth |
|---|---|---|---|
| false-positive rate (persistent feature, rho=0.9613, horizon=20) | **34%** | **7%** | alpha = 5% |
| SE ratio vs independently resimulated empirical SE, AR(1) rho=0.6 | 47.9x raw / 0.71x as-fed | **0.977** | 1.0 |
| bootstrap SE vs analytic iid SE on iid data | 29.5x | within tolerance | 1.0 |

A 34% false-positive rate means roughly one in three null results would have been reported as
significant at `t > 1.96`. Post-fix it is 7% against a nominal 5% — close enough that the residual
is consistent with finite-sample noise rather than a systematic error.

**What the fix comprised**, and note the second item was NOT where this spec originally said the
defect lived:

1. The moving-block bootstrap now tiles blocks to the full series length instead of returning one
   block per replicate, drawing only within sessions and reporting the count of sessions skipped
   for being shorter than the block.
2. `ExpectancyTable.spread_t` was never computed from `BucketStat.se_bps` at all —
   `conditional_expectancy` recomputed a separate spread SE as `std_bps / sqrt(n_effective)`, and
   that is where the over-correction actually reached the statistic `lens.py` criterion 3 reads.
   It now combines the two extreme buckets' `se_bps` in quadrature.
3. `n_effective` is reporting-only: `(std_bps / se_bps) ** 2`, never feeding back.
4. Block length DERIVED, not defaulted to `horizon`: `horizon + 5`, where 5 comes from measured
   lag-1..30 ACF of same-session 1-minute log returns across 10 liquid names over 2024Q1 (all
   showing the bid-ask-bounce negative lag-1 ACF, -0.040 to -0.066, decaying below 0.02 within
   2-5 bars; p95 decay lag = 5).

**Consequence that must not be forgotten:** every result previously gated on criterion 3
(`abs(spread_t) > 1.96`) was computed on the broken SE. Those verdicts are not automatically wrong,
but they are unverified. Recomputing them is a separate deliberate step and has NOT been done — the
implementation was explicitly instructed not to silently re-run and overwrite recorded numbers.

---

## CORRECTION 2026-08-21 — the L7/L8 recomputation scope was overstated

I wrote that "every result previously gated on criterion 3 was computed on the broken SE." That is
too broad, and the code says so plainly (`expectancy.py:496`):

    if horizon == 1:
        # For horizon=1, no overlap correction is needed
        se_bps = std_bps / np.sqrt(n_obs)

A `horizon == 1` verdict never reaches the block bootstrap at all. Its SE is `std/sqrt(n)`, and for
genuinely NON-OVERLAPPING observations that is the correct estimator, not a defective one.

**H2 runs at `horizon=1`** (`horizon_mode='session'`, one observation per symbol-day, explicitly
non-overlapping — stated in its own verdict). So H2's criterion 3 is unaffected by L7/L8 and needs
no recomputation. The same applies to any other `horizon=1` verdict.

**Narrowed scope:** recomputation is required for verdicts with `horizon > 1` only. All five
recorded hypotheses are KILLED regardless, and H3's criterion 3 already reads FAIL, so the
retrospective reach of this fix is small.

**Where it DOES matter is forward-looking, and that is the whole point:** Phase E sweeps horizons of
5, 15, 30, 60 bars and EOD. Every one of those lands on the overlapping path, which is exactly the
regime where the pre-fix false-positive rate was 34%.

## CORRECTION 2026-08-21 — H2 is KILLED, and I described it loosely

Finding L3 and its commit message called H2 "the only live signal family in this program". H2's
verdict is **KILLED** — it fails criterion 7 (recent-years cost gate: 2023-24 mean -15.20 bps
against a 2x hurdle of 16.53 bps), and its per-year spread decays from -39.18 bps in 2018 to
-9.62 bps in 2025.

The L3 fix is still correct and still necessary: criterion 5 branched on `if lag_0_edge > 0` and
would hard-FAIL any negative-signed edge, and H2's edge is negative BY CONSTRUCTION because it is a
reversal traded short-side. The reasoning holds. The characterisation of H2 as "live" does not.

**Accurate statement:** what is live is the low-turnover index-relative TILT built on H2's signal,
which survives by paying a one-leg hurdle (8.26 bps) and starting from the index return rather than
from zero. The underlying effect is real and decaying; the tilt is the construction that can still
use it.
