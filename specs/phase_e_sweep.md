# Spec E — the conditional-analysis sweep

Status: spec, written before implementation. Author: lead. Date 2026-08-21.

## What this phase is, and the trap it must not fall into

Measure `Feature_t -> ForwardReturn_{t+h}` for every feature primitive in the repo, at
h in {1, 5, 15, 30, 60 bars, EOD}. That is roughly **25 features x 6 horizons = ~150 trials**.

A 150-trial sweep is a machine for manufacturing winners. At a nominal alpha of 0.05, pure noise
yields ~7.5 "significant" results, and the best of 150 noise draws has an expected max Sharpe well
above zero. **This phase's output is worthless unless the multiple-testing accounting is honest**,
which is why it runs behind three things that had to land first:

- L7/L8 fixed: the overlapping-horizon SE was understating, giving a **34% false-positive rate**,
  now 7%. Five of the six horizons here are overlapping, so this was disqualifying.
- L1 fixed: `Lens.verdict()` can no longer report `survived=True` with kill criteria unrun.
- `ResearchContract`: every run is pre-registered and lands a `TrialRecord`.

## E1 — the feature registry

A single declared list, `research/sweep_features.py`, naming every primitive to sweep with its
call signature and required inputs. Not a glob over module contents — an explicit list, because
`n_planned_trials` is declared from it and a registry that silently grows changes the denominator.

Include at minimum: `volume_zscore`, `breakout_strength`, `parkinson_volatility`,
`garman_klass_volatility`, `rogers_satchell_volatility`, `efficiency_ratio`, `hurst_on_stitched`,
`variance_ratio`, `rolling_beta`, `beta_residual_return`, `sector_relative_return`, `breadth`,
`cross_sectional_dispersion`, `median_pairwise_correlation`, `vol_ratio`, `rv_to_vix_ratio`,
`close_location_value`, `signed_volume_proxy`, `amihud_illiquidity`, `overnight_return`,
`tradable_overnight_return`, `opening_range`.

**14 of the 15 functions in `features/market.py` have no production call sites.** This sweep is the
first thing to exercise them, so expect some to fail on real data in ways their unit tests did not
provoke. A feature that raises is a RESULT (report it), not a reason to drop it silently.

## E2 — what is measured per (feature, horizon)

    ic              Pearson IC, cross-sectional per bar          research/ic.py
    rank_ic         Spearman IC, cross-sectional per bar         research/ic.py
    ic_se           overlap-aware SE                             research/ic.py
    bucket_means    conditional mean return by CAUSAL quantile   expectancy.conditional_expectancy
    spread_bps      top-minus-bottom bucket spread
    spread_t        via the CORRECTED overlap-aware SE
    monotonic       is E[R | bucket] monotone across buckets
    hit_rate        fraction of bars with correctly-signed spread
    decay           IC across the full horizon grid + half-life  research/ic.ic_decay
    turnover        implied daily turnover of a book on this feature
    cost_hurdle     round-trip cost at the relevant participation

**Bucketing must be `cross_sectional_rank`, not the default `expanding_quantile`.** The default
needs `min_history` prior rows per column and silently yields zero usable buckets for a
once-per-session signal — indistinguishable from "no effect". This is the exact trap H2 documents.

**Every fixture and every run needs >= 5 symbols**: `cross_sectional_rank` enforces `min_names = 5`
and below it returns all-NaN, which downstream reads as `spread_t == 0.0` — a real-looking number
produced without the computation happening.

## E3 — multiple-testing accounting. The part that decides whether any of this counts.

1. **`n_planned_trials` is declared in the contract BEFORE the sweep runs**, computed from the
   registry as `len(features) * len(horizons)`. Running trial `k+1` raises. A sweep that quietly
   extends its own denominator has invalidated itself.
2. **`effective_n_trials` is MEASURED, not assumed.** `metrics.effective_n_trials(trial_returns)`
   takes the (T, n_trials) matrix and accounts for correlation between trials. The features here
   are heavily correlated by construction — three volatility estimators on the same bars, Hurst and
   variance-ratio measuring the same persistence — so the honest `n_eff` will be far below 150.
   Reporting 150 would be as wrong as reporting 1.
   *Precedent: the program has recorded `effective_n_trials` reporting 3.000 where the honest value
   was 1.055. Both directions of that error have been seen.*
3. **Deflated Sharpe uses that measured `n_eff`**, via `expected_max_sharpe(n_eff, var_trial_sharpes)`.
   `var_trial_sharpes = 1.0` is a KNOWN unmeasured placeholder (`lens.py:815`); it must be measured
   from the trial Sharpes this sweep produces, and the derivation recorded. Rule 8.
4. **PBO via `pbo_cscv`** on the trial matrix. The repo's recorded belief that PBO "can never work"
   here is WRONG — real PBO = 0.0326 after an SQL fix. Report it.

## E4 — promotion criteria

Nothing is promoted to a strategy on pooled statistics. **Recent years only** — the program's
standing rule, because survivorship inflates the early years on this dataset, and because H2 is the
worked example: sign-stable 8/8 years and a -24.30 bps pooled edge, yet KILLED on the recent-years
cost gate as it decayed to -9.62 bps by 2025.

A feature is a CANDIDATE only if, on the recent window:
- `abs(spread_bps) > 2 * cost_hurdle_bps`, and
- `abs(spread_t) > 1.96` on the corrected SE, and
- `E[R | bucket]` is monotone across buckets, and
- the deflated Sharpe clears its threshold at the MEASURED `n_eff`.

Monotonicity is not decoration: a spread with a non-monotone interior is usually a tail artefact,
and it is the cheapest available check that a result is not one.

## E5 — outputs

Per (feature, horizon): a `TrialRecord` with `contract_hash`, non-null `seed` and `git_sha`, and a
`returns.parquet` so `pbo_cscv` has a real matrix. Plus one summary table, sorted by deflated
Sharpe, with the measured `n_eff` and PBO printed at the top so no reader sees a ranked list
without its denominator.

**A negative result is a result.** If nothing clears, that is the finding, and it must be published
as plainly as a positive one would be — see `killed-hypotheses.md`, which exists so dead ideas stay
dead.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. The feature registry is an explicit list; adding an entry changes `n_planned_trials`.
2. A sweep declaring `n_planned_trials=k` raises on trial `k+1`.
3. Every trial writes a `TrialRecord` with non-null `contract_hash`, `seed`, `git_sha`.
4. Bucketing uses `cross_sectional_rank`; a run with < 5 symbols RAISES rather than silently
   returning all-NaN. **This is the anti-silent-failure test and it must not be softened.**
5. `effective_n_trials` on a matrix of near-identical trials returns ~1, not the column count.
6. `effective_n_trials` on a matrix of independent trials returns ~the column count.
7. `var_trial_sharpes` is measured from the trial Sharpes, with a recorded derivation — not 1.0.
8. Deflated Sharpe uses the measured `n_eff`, and a larger `n_eff` lowers the DSR for fixed returns.
9. `pbo_cscv` returns a finite number on a live trial matrix, not NaN.
10. IC SEs come from the overlap-aware path — assert they differ from a naive iid SE at h > 1,
    and note h = 1 legitimately uses the naive SE (`expectancy.py:496`) because non-overlapping.
11. A feature that RAISES on real data is recorded as a failed trial with its exception, not
    silently dropped from the summary.
12. Promotion requires ALL FOUR E4 conditions; a candidate failing any one is not promoted, and a
    test asserts each condition independently blocks promotion.
13. Promotion is evaluated on the recent window, not pooled — construct a fixture where pooled
    passes and recent fails, and assert NO promotion.

---

# AMENDMENT 1 — 2026-08-21. Obligation 10 conflated two different SEs. Both now pinned.

A test author found that obligation 10's sentence groups the `expectancy.py:496` citation under
"**IC** SEs", but that line is in `_compute_bucket_stats` — the BUCKET-SPREAD SE path, not the IC
path. E2 itself lists `ic_se` and `spread_t` as separate rows, so the spec contradicted its own
table. Two distinct computations:

    spread_t / bucket SE   expectancy._compute_bucket_stats   DOES special-case horizon == 1
    ic_se                  ic.information_coefficient          does NOT special-case it
                                                               (its docstring says so verbatim)

The author wrote tests for BOTH readings rather than picking one, which is how the contradiction
surfaced instead of being silently resolved.

## Is the `horizon == 1` shortcut actually safe? Measured, because I asserted it was.

I claimed non-overlapping labels make the naive SE correct. That is not automatic: **non-overlapping
LABELS do not imply independent OBSERVATIONS** if bucket membership persists — a persistent feature
keeps the same names in the same buckets on consecutive bars, which is a dependence channel that has
nothing to do with label overlap.

Measured on a TRUE NULL at horizon=1, varying only feature persistence, 40 seeds, 5 buckets,
6 symbols, 3000 rows:

    rho = 0.0      FP rate 7.5%
    rho = 0.9      FP rate 2.5%
    rho = 0.9613   FP rate 0.0%      (the derived real-feature persistence)

**The shortcut is CONSERVATIVE, not anti-conservative.** Feature persistence at h=1 pushes the
false-positive rate DOWN, not up, so the naive SE does not manufacture significance there. It is
safe to keep, and the h=0.0 reading of 7.5% is within binomial noise of alpha at 40 seeds.

Note this is the opposite direction from the overlapping case, where the pre-fix SE understated and
drove a 34% false-positive rate. Overlap inflates; feature persistence at h=1 deflates. They are
different mechanisms and must not be reasoned about interchangeably.

## Restated obligation 10, as two obligations

- **10a** — the bucket-spread SE at `horizon == 1` equals the naive `std/sqrt(n)` exactly
  (`expectancy.py:496`), and this is documented as conservative-for-persistent-features, not as an
  assumption of independence.
- **10b** — `ic_se` does NOT special-case `horizon == 1`; it always goes through the overlap-aware
  resampler. That is strictly more conservative and is correct: IC dependence can arise from feature
  persistence alone, and `ic.py` declines to assume otherwise.
- At `h > 1`, both paths must differ from a naive iid SE.

## Interface names, previously unpinned

The author had to guess the sweep module's API, which is my omission again — the third time in this
program. Pinned:

    nifty_quant/research/sweep_features.py
        FEATURE_REGISTRY            explicit list, not a module glob
        HORIZONS                    the horizon grid
        n_planned_trials()          len(FEATURE_REGISTRY) * len(HORIZONS)
        run_sweep(*, contract, close, day_offsets, horizons, feature_registry=None)
        measure_var_trial_sharpes(trial_sharpes)
        evaluate_promotion(...)     returns a decision plus the per-condition results

---

# AMENDMENT 2 — 2026-08-21. Module name collision, and the trap demonstrated live.

## 1. `research/sweep.py` is TAKEN. The Phase E runner goes in `research/feature_sweep.py`.

`src/nifty_quant/research/sweep.py` already exists — an unrelated grid-parameter / constraint-parsing
module (`_evaluate_constraint`, YAML grids). One suite guessed `research.sweep.run_sweep` for the
Phase E runner, which would have collided with it.

Pinned, and this supersedes the partial pin in AMENDMENT 1:

    research/sweep_features.py     FEATURE_REGISTRY, HORIZONS, n_planned_trials()
    research/feature_sweep.py      run_sweep(...), measure_var_trial_sharpes(...),
                                   evaluate_promotion(...)

Do NOT extend `research/sweep.py`. Two unrelated notions of "sweep" in one module is how the three
`config_hash` readings happened.

## 2. Both suites independently found the obligation-10 conflation

Suite A and suite B, written without sight of each other, each reported that obligation 10 conflates
`ic.information_coefficient` (never special-cases h=1) with `expectancy._compute_bucket_stats`
(does). Two independent readings converging on the same defect is the strongest confirmation this
process produces, and it is why the dual-suite rule earns its cost. Resolved in AMENDMENT 1.

## 3. The `min_names = 5` trap, demonstrated on the real function

Suite B's obligation-4 test does not describe the trap — it runs it. Confirmed independently by the
lead, `conditional_expectancy` with **3 symbols**, `method="cross_sectional_rank"`, horizon 5:

    spread_bps = 0.0     spread_t = 0.0     no exception raised

Two exact zeros that look like a measured "no effect", produced without the computation ever
happening. This is why obligation 4 requires the SWEEP to raise: the underlying primitive will not,
and a 150-trial sweep that silently scores a mis-configured feature as 0.0 has no way to tell that
apart from a real null.

**Obligation 4 must not be softened**, and the guard belongs in the sweep runner, not in
`cross_sectional_rank` — changing the primitive's return contract would ripple through every
existing caller and is out of scope here. Record it as a known sharp edge of the primitive.

---

# AMENDMENT 3 — 2026-08-21. The sweep must be structurally unable to reach the holdout.

Verified current state:

    holdout window starts   2025-08-14   (read count 7, all prior false positives)
    program START..END      2018-01-01 .. 2025-07-31
    END < holdout_start     True          -- a 14-day buffer

That buffer is a CONVENTION today, held only by two module-level constants in a recon script. A
~150-trial sweep is the single largest opportunity this program has to spend the holdout by
accident, and spending it is irreversible — there is no second first look at out-of-sample data.

**Required, both of them:**

1. The sweep's `ResearchContract` declares `holdout_intent = "never"`. Per
   `specs/research_contract.md`, only `reading_now` may reach the holdout, and a mismatch between
   declared intent and the `--allow-holdout` flag is refused in BOTH directions.

2. `run_sweep` ASSERTS that its data window ends strictly before the stored `holdout_start`, read
   from the live lock via `HoldoutLock`, and RAISES otherwise. Not a comment, not a default — an
   assertion that fires. The check costs microseconds and guards the one asset in this repo that
   cannot be recovered once spent.

A test must drive `run_sweep` with a window that overlaps the holdout and assert it raises. Note
that `tests/conftest.py`'s autouse fixture seeds a far-future (2099) boundary, so such a test needs
the `@pytest.mark.holdout_aware` marker to receive the REAL boundary — or should seed its own
lock in `tmp_path` with a boundary the fixture window deliberately crosses.

This is added as obligation 14.

---

# AMENDMENT 4 — 2026-08-23. `run_sweep` must receive OHLCV. Seven features were never measured.

## The defect

`run_sweep`'s signature takes only `close` + `day_offsets` (pinned in AMENDMENT 1 — my omission).
`research/sweep_features.py` therefore synthesises proxies at 18 call sites:

    volume = np.ones_like(close)          # constant
    high = low = open_ = close            # zero range

Consequence, measured on the full-panel run (`results/PHASE_E_SWEEP.md`): a z-score of constant
volume is NaN; `log(H/L)` with `H == L` is exactly zero. Seven features returned ALL-NaN and were
excluded from the trial matrix:

    volume_zscore, signed_volume_proxy, breakout_strength,
    parkinson_volatility, garman_klass_volatility, rogers_satchell_volatility,
    close_location_value

**These are not null results. They are measurements that did not happen.** The implementer flagged
the proxy assumption and predicted the degeneration; I accepted it as "a legitimate recorded
result", which was wrong. A feature fed constant inputs has not been tested, and reporting it
alongside genuinely-measured features implies it has.

This matters beyond tidiness: `breakout_strength` and `volume_zscore` are two of the four
components `specs/volume_breakout_v2.md` designs v2 around. Phase F cannot proceed until they have
real verdicts.

## Required change

`run_sweep` accepts the full OHLCV set. `close` remains required; the rest are optional and
default to `None`:

    run_sweep(*, contract, close, day_offsets, horizons,
              open_=None, high=None, low=None, volume=None,
              feature_registry=None, feature_registry_override=None,
              n_buckets=..., seed=0) -> list[TrialRecord]

**A feature whose required field is absent must RAISE, not silently substitute a proxy.** Each
`FeatureSpec` declares the fields it needs; `run_sweep` checks them before running the trial and
records a FAILED trial (obligation 11) naming the missing field. Silently substituting `close` for
`high` is precisely what produced seven fake nulls, and the fix is worthless if the substitution
merely moves.

The runner (`scripts/run_phase_e_sweep.py`) loads the fields the registry declares, rather than
hardcoding `fields=("close",)`.

**Memory note:** the runner currently loads close-only deliberately — loading all five fields was
5x the memory and contributed to a cache-contention failure. Load only the union of fields the
registry actually declares, and keep `memmap=False`.

## Test obligations (extending the existing suites)

15. `run_sweep` accepts `open_`/`high`/`low`/`volume` and passes them to features that declare them.
16. A feature declaring a field that is NOT supplied RAISES and is recorded as a failed trial naming
    the missing field — it must NOT fall back to a proxy.
17. With real (non-degenerate) OHLCV, each of the seven previously-excluded features produces a
    spread series with **> 0 finite observations**. This is the regression test for the defect: it
    would have failed on the committed run.
18. `FeatureSpec` exposes its required fields, and the runner's loaded field set is exactly their
    union — not a hardcoded list, so adding a feature that needs volume cannot silently regress.

## Ownership

This changes a signature both `tests/test_phase_e_sweep_a.py` and `_b.py` assert on, plus the
registry and the runner. **ONE agent owns all of it.** This repo has already produced a broken build
from two agents each correctly implementing opposite sides of one interface.


# AMENDMENT 5 — 2026-08-30. Two features record non-measurements as results. The sweep must
# refuse to book an all-NaN feature as a clean trial.

## The defects, both verified by reading the code (not conjecture)

**5a. `rv_to_vix_ratio` raises unconditionally at EVERY window length.** The adapter
`sweep_features._rv_to_vix_ratio` feeds the 2-D `(n_rows, n_symbols)` output of
`core.ewma_volatility_ann` straight into `market.rv_to_vix_ratio`, whose contract is 1-D
(`_as_float64_1d` at `market.py:586` raises on 2-D input). Every trial for this feature records
as FAILED — which obligation 11 treats as a result, but an unconditional shape mismatch is a
HARNESS defect, not a feature verdict. 1 of 22 registry entries has never been measurable.

**5b. `variance_ratio` books a fake null at production length.** `_variance_ratio_1d_daily`
(`persistence.py:575-576`) returns NaN if ANY value in the whole series is non-finite — even
though the very next lines process days as independent segments. At the production window
(701,863 rows) every symbol has at least one missing bar, so all 149 symbols return NaN, the
feature is all-NaN, and the sweep records a clean trial that reads as "no edge". On 1- and
3-month slices 62-66% of symbols are complete, so every short-slice smoke test passes. This is
the same shape as the seven AMENDMENT-4 fake nulls: a measurement that did not happen,
indistinguishable in the output from a genuine negative.

## Required changes

**C1 (`sweep_features.py`).** `_rv_to_vix_ratio` applies the 1-D feature per symbol — loop over
columns exactly as `_overnight_return` already does. Do NOT change `market.rv_to_vix_ratio`'s
1-D contract; it is decorated, documented and tested as 1-D.

**C2 (`persistence.py`).** In `_variance_ratio_1d_daily`, remove the whole-series finiteness
gate. Instead, skip any day segment containing a non-finite value (alongside the existing
too-short-segment skip). Rationale: rule 6 — NaN means "no bar occurred"; a missing bar
invalidates its own day's segment, never the whole year. If every segment is skipped the result
stays NaN (unchanged). On fully-finite input the result must be BIT-IDENTICAL to today's — this
change may only ADD segments' worth of data relative to the current all-or-nothing behaviour on
NaN-bearing input, and must change nothing on clean input. No forward-filling, no interpolation.

**C3 (`feature_sweep.py`).** Per-feature finite-fraction check in `run_sweep`: after
`feature_values` is computed, if the fraction of finite entries is EXACTLY 0.0, record every
horizon's trial as FAILED with an error naming the condition (e.g.
`"AllNaNFeature: feature 'variance_ratio' produced 0 finite values over <n> cells"`), never a
clean trial. Rule-8 note: 0.0 is not a tuned threshold — it is exact degeneracy, "no measurement
occurred". Any cutoff ABOVE 0.0 would be a hand-chosen constant and is deliberately NOT
introduced; a partially-finite feature records normally and its finite fraction is already
visible via `n_defined`.

## Test obligations (extending the existing dual suites; numbering continues AMENDMENT 4)

19. `_rv_to_vix_ratio` on a small multi-symbol panel returns a `(n_rows, n_symbols)` float64
    array with > 0 finite entries — it must not raise. (Regression for 5a: it raises today.)
20. `variance_ratio` (1-D + day_offsets path): (i) fully-finite input → bit-identical to the
    pre-change value; (ii) input with NaN confined to one day → equals the value computed on
    the same input with that day's segment absent, and is FINITE (regression for 5b: NaN
    today); (iii) all days NaN-bearing → NaN.
21. `run_sweep` with a feature that returns all-NaN records ALL its horizons as FAILED trials
    whose error names the all-NaN condition; the trial count is unchanged (obligation 2/11:
    still one register_trial per horizon).
22. A feature with a small positive finite fraction still records clean (non-FAILED) trials —
    the check must not fire above exact zero.

## Ownership

C1, C2, C3 are three files with NO shared interface — one agent per file is safe here. The two
test-suite extensions are written INDEPENDENTLY from this amendment alone (one author per suite,
neither sees the other's tests, before any implementation exists), per rule 1.
