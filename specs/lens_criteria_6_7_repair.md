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
- If `strategy_returns` is supplied AND `effective_n_trials >= 2`, deflate against
  `expected_max_sharpe(effective_n_trials, var_trial_sharpes=1.0)` as today.
- If `strategy_returns` is supplied but `effective_n_trials < 2`, report `NOT_EVALUATED` with a
  reason stating that a single declared trial cannot be deflated. A one-trial "deflated" Sharpe
  is just a Sharpe; calling it deflated is the misreport being fixed.
- A supplied `strategy_returns` that is empty, all-NaN, or shorter than 2 finite values also
  yields `NOT_EVALUATED`, never a crash and never 0.0.

`NOT_EVALUATED` must never be silently counted as a PASS when computing the overall verdict —
mirror exactly how criterion 5 already treats `NOT_EVALUATED` (read it and follow it).

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

A calendar year is **complete** if and only if the panel window fully spans it:

```
complete(year)  <==>  panel_first_date <= date(year, 1, 1)  AND  panel_last_date >= date(year, 12, 31)
```

This introduces NO tunable threshold, which is the point — it satisfies rule 8 by having no
cutoff to derive rather than by deriving one. Use the panel's own first and last dates.

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
