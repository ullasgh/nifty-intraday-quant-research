# Spec: universe-causality and execution-causality guards

Phase B track B4.

## Why this exists

`guards.py` (777 lines) enforces 14 contracts and is one of the strongest parts of the repo. It
has caught a real full-sample volume-deseasonalization leak. But its causality guard is
**temporal only**: `@causal` (`guards.py:267`) perturbs future ROWS of a numeric array and checks
the output does not move. Two other kinds of lookahead are not enforced anywhere.

**Universe causality** — the set of names in the cross-section at `t` must not depend on data
after `t`. Nothing in `guards.py` references membership, eligibility or symbol sets. This is
exactly the leak that a static current-day universe creates, and Phase B1 makes it a live risk in
a new way: an eligibility mask is computed from data, so it can be computed wrongly.

**Execution causality** — a fill at `t` must not read volume, spread or price after `t`. This is
currently a structural property of the engine (`engine.py:13,19,523`) plus the adversarial tier
(`tests/verification/test_causality.py`, `test_leak_canary.py`, `test_seam_registry.py`). A
structural guarantee that is only documented and tested is one refactor away from being neither.

## Required behaviour

### A. `universe_causal`

A decorator/context in the same style as `@causal`, wrapping a function that produces an
eligibility mask or symbol selection:

1. Compute the mask on the full panel.
2. Perturb panel data strictly after session `d` — including making names appear or disappear.
3. Recompute and assert row `d` of the mask is bit-identical.
4. Repeat over several `d`, chosen to include the first and last sessions and at least one
   session adjacent to a symbol's first present bar (the boundary that matters).

Making a name *appear* later must not retroactively make it eligible earlier; making one
*disappear* later must not make it ineligible earlier. Both directions are tested, because a
one-directional check passes on a mask built from `np.any(present, axis=0)`, which is exactly the
bug.

### B. `execution_causal`

Wraps the fill path. Perturb `volume`, `high`, `low` and `close` strictly after row `t` and assert
the fill quantity, fill price and charges at `t` are unchanged. This turns `engine.py`'s
structural claim into an enforced one.

Must cover the participation cap specifically: `fills.py:101` caps at
`max_participation * bar_traded_value`, and `bar_traded_value` is the one input where reading the
wrong row is both easy and invisible in the output.

### C. Strictness

Both new guards are FULL-strictness only, like `@causal` and `deterministic` — they re-run the
wrapped computation, so they cannot be on by default. They must be exercised in
`tests/verification/`, which is the tier that runs under FULL.

### D. Honest failure

`research/expectancy.py:939` documents a place where `@causal` **cannot** detect a leak. Any
equivalent blind spot in the new guards is documented at the site in the same way. A guard that
quietly cannot see a class of leak is worse than no guard, because it licenses trust.

## Required tests

1. A deliberately leaky eligibility function — one using `np.any(present, axis=0)` over the whole
   panel — is CAUGHT by `universe_causal`. This is the canary; if it passes the guard, the guard
   is broken.
2. A correct causal eligibility function passes.
3. Both perturbation directions (name appears later / disappears later) are exercised, and a
   guard that only checks one direction is shown to miss the canary.
4. A deliberately leaky fill model — one reading `bar_traded_value[t+1]` — is CAUGHT by
   `execution_causal`.
5. The real `FillModel` passes.
6. Both guards are no-ops at OFF and CHEAP strictness and active at FULL.
7. Guard failures raise `ContractViolation` with a message naming the row/session and the
   quantity that moved.
8. The guards do not fire on legitimate non-determinism that is not lookahead — e.g. a function
   whose output genuinely depends on a seed — or if they cannot distinguish the two, that
   limitation is documented per section D.

## Constraints

- Rule 9 applies to any statistical assertion: assert on a single spread or a false-positive rate
  over >= 30 seeds; never a per-bucket coverage check; never fix a failure by changing the seed.
- Guards must not be so slow that FULL strictness becomes unusable — perturb a sample of rows,
  not every row, and say so.
