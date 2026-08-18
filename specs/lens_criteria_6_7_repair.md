# Spec: repair kill criteria 6 and 7 in `Lens.verdict()`

**Status:** contract for TDD. Two independent test suites written from this document ALONE,
before any implementation exists. Neither author reads the other's file.

Both defects were found by an adversarial critic reviewing published verdicts. Both cause a
verdict to REPORT something it did not measure. Neither changes a kill outcome by itself — but
`results/hypotheses/H2/verdict.md` and `H3/verdict.md` are already published carrying these
lines, so a reader is currently being told multiple-testing was accounted for when it was not,
and that a 7-month year is complete when it is not.

Single unit, single file: `src/nifty_quant/research/lens.py`. ONE implementer only — criteria 6
and 7 sit in the same function and must not be split across concurrent agents.

---

## Defect 1 — criterion 6 deflates the wrong quantity

Current code (around `lens.py:742-748`):

```python
fwd_flat = fwd.values.flatten()
valid_mask = np.isfinite(fwd_flat)
valid_returns = fwd_flat[valid_mask]
if len(valid_returns) > 0:
    dsr = deflated_sharpe(valid_returns, sr0=0.0)
```

`fwd.values` is the UNCONDITIONAL forward return of every symbol at every row. It is not the
strategy's return series. The hypothesis's actual P&L is the top-minus-bottom spread series,
which criterion 6 never sees.

With the default `effective_n_trials=1` (neither H2 nor H3 passes anything else), the branch
reduces to `dsr > 0.0` — i.e. **"did the universe drift up over the sample"**. It then prints
`6. Deflated Sharpe criterion: PASS (trials=1)`, which reads as "multiple testing was accounted
for". It was not: the program ran at least five hypotheses plus variants (three H4 thresholds,
three H5 signal variants x two lag modes, H3 recon at two execution times) and no deflation was
ever applied.

### Required behaviour

`verdict()` gains a new keyword-only parameter:

```python
def verdict(self, ..., strategy_returns: np.ndarray | None = None, ...) -> HypothesisVerdict:
```

- `strategy_returns` is a 1-D float64 array of PER-PERIOD STRATEGY returns (the realised
  spread series, one value per rebalance period). It is NOT a 2-D panel.
- **If `strategy_returns is None`, criterion 6 MUST report `NOT_EVALUATED`** with a reason
  naming what is missing. It must NEVER report PASS or FAIL from `fwd.values`.
- If `strategy_returns` is supplied AND `effective_n_trials >= 2`, deflate CORRECTLY — see the
  units defect below. Do NOT preserve today's comparison.

### A SECOND defect in criterion 6: it compares a probability to a Sharpe ratio

`deflated_sharpe` returns a **probability** — `stats.norm.cdf(z)` clipped to [0, 1]
(`metrics.py:492-493`). `expected_max_sharpe(n, var_trial_sharpes=1.0)` returns a **Sharpe
LEVEL**:

    n:      2       3       4       5       8       10      20
    value:  0.5198  0.8528  1.0521  1.1926  1.4590  1.5746  1.9007

So `dsr > exp_max_sharpe` is **impossible for n >= 4**, and meaningless for n = 2 or 3. Since
this program ran at least five hypotheses plus variants, a realistic `effective_n_trials` is
~10 — meaning that the moment anyone actually passes `strategy_returns`, criterion 6 becomes an
**automatic FAIL for every hypothesis**, killing everything including H2. That is a worse
misreport than the one being fixed, so it must be repaired in the same pass.

**Correct usage** (Bailey / Lopez de Prado): pass the expected-max Sharpe as the benchmark,
then compare the resulting PROBABILITY against a significance level:

    dsr = deflated_sharpe(strategy_returns, sr0=expected_max_sharpe(n_trials, var_trial_sharpes))
    c6  = "PASS" if dsr > DSR_SIGNIFICANCE else "FAIL"

- `DSR_SIGNIFICANCE` is a rule-8 decision cutoff. Define it as a NAMED module constant with its
  derivation recorded beside it. Until derived, the implementer may set it to 0.95 with an
  explicit `TODO(rule-8)` comment naming what must be measured — an undocumented inline literal
  is not acceptable.
- `var_trial_sharpes=1.0` is itself an unmeasured constant. `metrics.effective_n_trials(...)`
  exists to measure trial structure from data; note in the code that fixing the variance at 1.0
  is a placeholder, not a measurement.
- **Any NaN `dsr` -> `NOT_EVALUATED`, never FAIL.** `deflated_sharpe` returns NaN for `t < 4`
  (it needs skew and kurtosis, `metrics.py:476-487`) and for negligible std (`:471`). A
  3-element or constant `strategy_returns` array is none of the listed degenerate cases yet
  still yields NaN, and today's code maps NaN -> FAIL.
- Partially-NaN `strategy_returns`: drop the non-finite entries and evaluate on the remainder
  if >= 4 finite values survive; otherwise `NOT_EVALUATED`. State this explicitly so the two
  suites cannot diverge.
- If `strategy_returns` is supplied but `effective_n_trials < 2`, report `NOT_EVALUATED` with a
  reason stating that a single declared trial cannot be deflated. A one-trial "deflated" Sharpe
  is just a Sharpe; calling it deflated is the misreport being fixed.
- A supplied `strategy_returns` that is empty, all-NaN, or shorter than 2 finite values also
  yields `NOT_EVALUATED`, never a crash and never 0.0.

**How `NOT_EVALUATED` affects the overall verdict — literal, because my first draft of this
line contradicted itself.** Criterion 5's actual precedent (`lens.py:824-828`) EXCLUDES
`NOT_EVALUATED` from the conjunction, so `survived` can be `True` with a criterion unevaluated.
Criterion 6 must follow that same precedent: **`NOT_EVALUATED` does not block `survived`**, and
a test asserting on this must expect `survived is True` when every evaluated criterion passes.

That is deliberate but uncomfortable, and the discomfort must be visible rather than papered
over: with nothing in the repo currently passing `strategy_returns`, criterion 6 will report
`NOT_EVALUATED` for every hypothesis, so a verdict can read "survived" with **zero
multiple-testing correction applied**. Therefore:

- `HypothesisVerdict` must expose a boolean (e.g. `any_not_evaluated`) that is True when any
  criterion is `NOT_EVALUATED`, and `explain()` must state prominently that the verdict is
  INCOMPLETE — not merely list `NOT_EVALUATED` among seven reason lines where a reader skims
  past it.
- The published verdict header must not read a bare "SURVIVED" while a criterion is
  unevaluated.

This is the honest middle path between deleting criterion 6 (losing a real check) and leaving
it permanently decorative (implying seven criteria were applied when six were).

---

## Defect 2 — criterion 7 calls a 7-month year "complete"

Current code (around `lens.py:775-778`):

```python
complete_years = sorted(
    year for year in stab_report.by_year if session_counts.get(year, 0) >= 20
)
```

The research window ends 2025-07-31, so 2025 holds ~145 sessions, clears `>= 20`, and is used
as one of the two "recent complete years". Every recent-years number in H2, H3 and H5 therefore
averages a full 2024 against a 7-month 2025 — and the decay narrative leans hard on those small
2025 figures (H3 -1.07, H5 -2.41).

Note also that `>= 20` is itself a hand-chosen constant, which CLAUDE.md rule 8 forbids.

### Required behaviour — threshold-free by construction

A calendar year is **complete** if and only if the panel contains at least one session in
**each of the 12 calendar months** of that year.

```
complete(year)  <==>  {d.month for d in panel.dates if d.year == year} == {1, 2, ..., 12}
```

This introduces NO tunable threshold — 12 is definitional, not tunable — so it satisfies rule 8
by having no cutoff to derive.

**Why not the obvious date-span test.** My first draft used
`panel_first_date <= date(year,1,1) AND panel_last_date >= date(year,12,31)`. It is WRONG,
because `panel.dates` holds only TRADING sessions and `Panel.sub` slices with `dates <= end`:

- A window ending 2023-12-31 has `panel_last_date = 2023-12-29` (the last NSE session), so
  2023 would be marked PARTIAL despite being fully covered — and criterion 7 would silently
  fall back to (2021, 2022). Same for 2022, whose last session is 2022-12-30. That is a
  verdict-moving misclassification of a complete year.
- The mirror risk at the start of a window (Jan 1 being a market holiday) does not bite the
  published window — 2018-01-01 IS a real session — but the rule was fragile in principle.

Verified against the real session calendar: 2025 (Jan-Jul only) -> incomplete; 2022 and 2023
-> complete; window starting 2018-06-01 -> 2018 incomplete; window ending 2024-12-31 -> 2024
complete. Interior years are complete under both rules, so this is a strict improvement.

**Scope of the exclusion:** partial years are excluded from criterion 7 ONLY. Criterion 2's
`n_years_total` (`stability_report`) continues to count every year present, including partial
ones — do not change it in this pass. State this in the code so the two behaviours are not
mistaken for an inconsistency.

- Years failing this test are PARTIAL and must be excluded from `complete_years`.
- If fewer than two complete years remain, criterion 7 reports `NOT_EVALUATED` (existing
  behaviour) — never a silent PASS.
- The reason string must NAME any partial year it excluded, e.g.
  `(excluded partial years: 2025)`. A reader must be able to see that 2025 was dropped and why;
  silently changing which years are used would replace one misreport with another.

### Consequence, which is intended

On the standard 2018-01-01..2025-07-31 window this changes criterion 7 from (2024, 2025) to
(2023, 2024). Published verdicts WILL move. That is correct — they were averaging a partial
year. Every affected verdict must be regenerated afterwards; do not hand-edit the numbers in
the markdown.

---

## Required tests

Two INDEPENDENT suites: `tests/test_lens_criteria_repair_a.py` and
`tests/test_lens_criteria_repair_b.py`.

**Criterion 6:**
1. `strategy_returns=None` -> criterion 6 is `NOT_EVALUATED`, and the reason names the missing
   input. Assert the exact criterion line, not just the overall verdict.
2. `NOT_EVALUATED` does not count as a PASS in the overall verdict — construct a case where
   every other criterion passes and assert the overall result matches criterion 5's precedent.
3. `strategy_returns` supplied with `effective_n_trials=1` -> `NOT_EVALUATED`, reason states a
   single trial cannot be deflated.
4. `strategy_returns` supplied with `effective_n_trials>=2` -> PASS/FAIL as appropriate; assert
   BOTH a passing and a failing case, driven by the returns you construct.
5. Empty / all-NaN / single-finite-value `strategy_returns` -> `NOT_EVALUATED`, no exception.
6. **Regression guard for the actual bug:** a panel whose `fwd.values` drift strongly POSITIVE
   while `strategy_returns` are flat or negative must NOT report criterion 6 as PASS. This is
   the test that would have caught the defect; it must fail against the old implementation.

**Criterion 7:**
7. A window ending mid-year (e.g. 2025-07-31) excludes that year as partial, and the reason
   names it.
8. A window ending exactly 2024-12-31 treats 2024 as complete.
9. A window STARTING mid-year (e.g. 2018-06-01) excludes 2018 as partial — the rule is
   symmetric and both ends must be tested.
10. With fewer than two complete years -> `NOT_EVALUATED`, never PASS.
11. The two years actually used are the two most recent COMPLETE years, asserted explicitly by
    year number, not merely by the resulting figure.

**Both:**
12. Every one of the SEVEN criteria is still reported, in order, even after one fails.
13. Determinism: same inputs and seed -> identical verdict text.

## Constraints

100% line+branch coverage with pragma exclusion DISABLED. Fully annotated; ruff clean.
Never touch `data/`. Never assume 375 bars/session — include an irregular-session fixture.
Bars are LEFT-labelled; NaN means "no bar"; float64 in motion. Reuse existing `Lens` fixtures
and helpers where they exist rather than inventing parallel ones.
