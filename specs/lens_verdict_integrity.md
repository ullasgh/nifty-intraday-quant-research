# Spec — Lens verdict integrity (fixes L1, L3, L4)

Status: spec, written before implementation. Author: lead. Date 2026-08-20.
Findings this fixes: `specs/research_validity_findings.md` L1, L3, L4.

## Why

`Lens.verdict()` encodes seven kill criteria and is the gate every hypothesis passes through. Three
defects mean it can report a survivor it did not fully test.

## L1 — an unevaluated criterion must not vanish

Current (`lens.py:901-909`):

    evaluated_results = [t for t in reason_tokens if t != "NOT_EVALUATED"]
    survived = all(r == "PASS" for r in evaluated_results)

A criterion whose input was not supplied is dropped from the conjunction — neither pass nor fail.
`survived=True` with two of seven criteria never run is reachable, and it is the actual state of
both committed verdicts (`results/hypotheses/H2/verdict.md:17-18`, `H3/verdict.md:18-19`).

A kill criterion that disappears when unsupplied is not a criterion.

**Required:** the verdict becomes tri-state. `HypothesisVerdict` gains an `outcome` field taking
`SURVIVED`, `KILLED`, or `INCONCLUSIVE`:

- `KILLED` — any evaluated criterion FAILed. A failure is decisive regardless of what else ran.
- `SURVIVED` — all seven evaluated AND all PASS.
- `INCONCLUSIVE` — no failure, but at least one criterion NOT_EVALUATED.

`survived` remains as a property for compatibility and MUST be `outcome == SURVIVED`, so it can
never again be true while a criterion is unrun. Any caller reading `survived` therefore tightens
automatically rather than silently keeping the old meaning.

Ordering matters: KILLED beats INCONCLUSIVE. A hypothesis that failed a criterion it did run is
killed, not "inconclusive pending the ones we skipped".

`tests/test_lens.py:819-836` currently asserts the old behaviour as intended. It is a test
defending a defect — the same shape as the D1 tests already retired this phase — and must be
rewritten to the tri-state contract, not deleted.

## L3 — criterion 5 must not reject an edge for having the right sign

`lens.py:767-790`:

    if lag_0_edge > 0:
        lag_1_ratio = lag_1_edge / lag_0_edge
        ...
    else:
        c5_result = "FAIL"

A negative `lag_0_edge` hard-FAILs. H2's measured edge is **-24.30 bps** — negative by
construction, because it is a reversal signal traded short-side. Wiring criterion 5 up as written
would reject the entire overnight-reversal family, which is the only live signal family in this
program, for having the sign it is supposed to have.

**Required:** the retention ratio is computed on MAGNITUDE, with sign agreement checked separately.

    retention_k = abs(lag_k_edge) / abs(lag_0_edge)          for k in (1, 2)
    sign_holds_k = sign(lag_k_edge) == sign(lag_0_edge)

Criterion 5 passes when, for both k: `retention_k >= threshold` AND `sign_holds_k`. An edge that
keeps its magnitude but FLIPS sign at lag 1 is not a surviving edge, so the sign check must stay —
it just must not be smuggled in via a positivity assumption on lag 0.

`abs(lag_0_edge) == 0` (or non-finite) is a genuine FAIL with a distinct reason string: there is no
edge to retain, which is different from an edge that decayed.

The `0.5` threshold is currently undocumented and violates rule 8. It is NOT resolved here.
Criterion 5 must report its threshold as `UNCALIBRATED` alongside its result until a null is
measured, so a reader can see the result rests on an unmeasured cutoff rather than discovering it
by reading the source. Do not invent a derivation to make this look finished.

## L4 — `Lens.universe` is stored and never read

Assigned at `lens.py:239`, read nowhere. A caller passing a restricted universe gets a verdict
computed on the full one, silently.

**Required:** either the universe restricts the panel before any statistic is computed, or the
parameter is removed. **Recommend restricting**, because Phase B established that universe choice
materially changes results — the tilt's recent-window significance clears on the full universe
(t=2.75) and fails on continuous coverage (p=0.077). A silently-ignored universe argument on that
evidence is not a dead parameter, it is a trap.

If restriction is implemented, the restriction happens once at construction and the resulting
symbol count is recorded on the verdict, so a reader can confirm it took effect.

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. All seven criteria PASS and all evaluated -> `outcome == SURVIVED`, `survived is True`.
2. One criterion NOT_EVALUATED, rest PASS -> `outcome == INCONCLUSIVE`, **`survived is False`**.
   This is the L1 regression test and the most important test in the suite.
3. One criterion FAIL plus one NOT_EVALUATED -> `outcome == KILLED`. KILLED beats INCONCLUSIVE.
4. `survived` is exactly `outcome == SURVIVED` for every combination — assert the property, not a
   reimplementation of it.
5. Criterion 5 with a NEGATIVE `lag_0_edge` that retains magnitude and sign at lags 1 and 2
   PASSES. This is the L3 regression test. Use H2's actual sign: `lag_0_edge = -24.30`.
6. Criterion 5 with an edge that retains magnitude but FLIPS sign at lag 1 FAILS.
7. Criterion 5 with `lag_0_edge == 0` FAILS with a reason distinguishable from ordinary decay.
8. Criterion 5's reported result carries an `UNCALIBRATED` marker for its threshold.
9. A `Lens` constructed with a restricted universe computes its statistics on the restricted
   symbol set — assert via a fixture where a symbol excluded by the universe would visibly change
   the answer if included. A test that passes whether or not restriction happened is vacuous here.
10. The verdict records the symbol count actually used.

---

# AMENDMENT 1 — 2026-08-20. Interface pinned; and a vacuity guard.

The two independent suites disagreed on obligation 4 — one RED, one GREEN. Chasing that
disagreement rather than averaging it found a vacuous test and two unpinned names. This is the
dual-suite rule doing exactly what it exists for, and it is recorded as such.

## 1. `hasattr` guards are FORBIDDEN in a tests-first suite

Suite A's obligation 4 wrapped every assertion:

    if hasattr(verdict1, "outcome"):
        assert verdict1.survived == (verdict1.outcome == "SURVIVED")

`outcome` does not exist yet, so the guard is false, the body executes ZERO assertions, and the
test reports GREEN. Suite B, asserting directly, went RED on `AttributeError` — which is the
correct state for a tests-first suite against unimplemented code.

**Rule: a tests-first suite must fail on the missing attribute.** `AttributeError` IS the
signal. Never guard an assertion behind `hasattr`, `getattr(..., default)`, `try/except
AttributeError`, or `pytest.skip` on absence — every one of those converts "not implemented" into
"passing", which is the single most expensive kind of test to have in a suite, because it is
counted as coverage of something nobody checked.

Same shape as the three tautologies amended out of `specs/overlap_se.md`. Two independent
occurrences in one day makes it a pattern, not an accident.

## 2. `outcome` is a plain string, not an enum

Values `"SURVIVED"`, `"KILLED"`, `"INCONCLUSIVE"`, typed as a `Literal`. This matches the existing
convention: every criterion result and `StabilityReport.dominant_sign` is a plain string `Literal`,
never an enum. Suite B's reading was correct.

## 3. The symbol count is `n_symbols_used`

Obligation 10 named no attribute; the suites chose `symbol_count` and `n_symbols_used`
independently. Pinned to **`n_symbols_used`**, matching `n_years_total` /
`n_years_sign_consistent` already on `StabilityReport`.

Recording why this is pinned in the spec rather than left to the implementer: this repo has already
produced a broken build from two agents each correctly implementing opposite sides of one
interface. An unnamed attribute in a spec is an interface with no owner.

## 4. Obligation 6 is GREEN on current code, and that is CORRECT

Both suites report obligation 6 (edge retains magnitude but flips sign at lag 1 -> FAIL) as GREEN.
It passes for the wrong reason: current code FAILs *any* negative `lag_0_edge`, so it happens to
produce the required outcome via an over-broad rule rather than a correct one.

It stays, as a permanent regression test: once L3 is fixed to compute retention on magnitude, a
partial fix that accepts sign flips would newly break it. A test that is green for the wrong reason
today but guards the right thing tomorrow is worth keeping — provided the reason is written down,
which it now is.
