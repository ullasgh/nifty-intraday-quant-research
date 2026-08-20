# Spec C4 — the ResearchContract

Status: spec, written before implementation. Author: lead. Date 2026-08-20.

## Why

The program's stated verification standard is that *every published number is reproducible from a
registry row*. It is not currently met, and the gap is widest exactly where it matters most.

A contract is a declared-before-run schema covering data, features, label, execution, portfolio and
validation, which the research entry points refuse to run without. Its purpose is not paperwork: it
is the mechanism that makes multiple-testing accounting honest. A sweep that can silently
re-specify itself between runs can manufacture a winner, and `effective_n_trials` cannot see it
happen. Phase E fans out one agent per feature across many horizons; without pre-registration that
phase is a p-hacking engine with good intentions.

## Verified current state

### P1 — `tilt` bypasses the entire research spine. This is the headline.

`nq tilt` (`cli.py:1661-1740`) calls `run_tilt()` (`research/tilt.py:332`), which never calls
`run_backtest()` and never writes to `TrialRegistry`. Confirmed by direct grep: `research/tilt.py`
contains **zero** occurrences of `TrialRegistry`, `TrialRecord`, or `run_backtest`.

The tilt is **the only construction in this repo that has beaten the index net of costs** — net
positive in all eight measured years, ~5.8% annualised net excess. It is the candidate headed for
the holdout. And it has no trial record, no provenance, no config hash, and no registry row.

So the one result the program intends to act on is the one result its reproducibility machinery has
never seen. Any enforcement that gates only `run_backtest()` leaves `tilt` as a live escape hatch
and would produce a contract that looks complete while exempting the thing it most needs to bind.

### P2 — three `config_hash` implementations, with different semantics

| location | hashes over | used by |
|---|---|---|
| `config.py:72-74` | strategy, params, universe, start, end | `backtest` |
| `strategy/registry.py:62-67` | strategy, params ONLY | `walkforward`, `sweep` |
| `research/sweep.py:124-127` | — | **dead code**, never imported |

The one `walkforward` and `sweep` actually use omits universe, dates, costs and seed. Two runs over
different date ranges, different universes, or different cost models therefore collide on the same
`config_hash`. That is the same class of defect as D1 (the registry-collision bug this phase
already reversed), one level up: D1 was about which colliding row wins, and this is about runs
colliding that were never the same run.

### P3 — `seed` is stubbed `None` at every write site

`cli.py` ~659, ~1543, and by omission in `walkforward`. `RunConfig.seed` exists and is simply not
threaded through. A trial whose seed is unrecorded is not reproducible, whatever else the row says.

### P4 — the pooled/confirmation record has the weakest provenance

`walkforward`'s per-split records (`cli.py:1060-1082`) and its pooled record (`cli.py:1268-1289`)
populate **none** of the 12 provenance fields added by `TrialRecord` Amendment 1. The pooled record
is the one representing the corrected, validated verdict — the row a reader would trust most — and
it carries the least evidence.

### P5 — nothing resembling pre-registration exists

No hits for `preregister`/`pre_regist` anywhere. `data/MANIFEST.json` (`data/manifest.py:22-62`) is
a read-only descriptor of the raw corpus, not a pre-registration. The `contract` hits in
`guards.py` are unrelated runtime ndarray shape/dtype checks. Only strategy `params` are
schema-validated, via a per-plugin pydantic model (`strategy/base.py:192-203`); the full run scope
has no unified schema, and `walkforward`/`sweep` do not use the single `RunConfig` model that does
exist.

## The contract

`ResearchContract` is a frozen, hashable declaration with six required sections. Every field must
be either supplied or explicitly declared absent — there is no default that silently fills in a
research choice, because a default IS a research choice.

| section | contents |
|---|---|
| `data` | panel id, `panel_hash`, `start`, `end`, bar interval, `universe_name`, `universe_hash` |
| `features` | ordered feature ids with their parameters, and `feature_version` |
| `label` | forward-return horizon in bars, label construction, whether overlapping |
| `execution` | `cost_model_id`, `slippage_model_id`, `decision_latency_bars`, participation cap |
| `portfolio` | sizing scheme, gross/max-weight clips, target volatility if any |
| `validation` | split scheme, purge width, embargo width, `n_planned_trials`, holdout intent |

`contract_hash` is the hash of all six sections. It REPLACES the three `config_hash` functions:
`config.py` and `strategy/registry.py` become thin wrappers that delegate, and `sweep.py`'s dead
copy is deleted rather than left as a fourth reading of the same idea.

**`n_planned_trials` is declared before the sweep runs**, and the realised trial count is checked
against it. A sweep that runs more trials than it declared has changed its own multiple-testing
denominator mid-flight, and that must be an error, not a footnote.

**`holdout intent`** is an explicit tri-state: `never`, `after_conditions_close`, or
`reading_now`. Only `reading_now` may reach the holdout, and it must match the `--allow-holdout`
flag; a disagreement between declared intent and passed flag is refused.

## Enforcement

Two gates, not one:

1. `run_backtest()` (`engine.py:229`) — reached synchronously by `backtest`, `walkforward`, `sweep`.
2. `run_tilt()` (`research/tilt.py:332`) — reached by `tilt`, and by nothing else.

Gating only (1) is the failure mode this spec exists to prevent (see P1). Both entry points refuse
to run without a contract, and both write a `TrialRecord` carrying `contract_hash`.

Implementation note: adding a required parameter to `run_backtest` touches every call site
including tests. **One agent owns both sides of that interface** — this repo has already produced a
broken build from two agents each correctly implementing opposite sides of one seam. Do not split
the signature change from the call-site updates.

## What this does not do

It does not make a result correct, and it must not be described as if it did. A pre-registered bad
hypothesis is still a bad hypothesis; the contract only guarantees that what was measured is what
was declared, and that the trial count is honest.

## Test obligations

Dual independent suites per rule 1, written from this spec alone.

1. Constructing a contract with any of the six sections missing raises; the message names the
   missing section.
2. `contract_hash` is stable across process restarts and insensitive to key ordering within a
   section.
3. `contract_hash` CHANGES when any one of: universe, start, end, cost model id, seed, or embargo
   width changes. One test per field — this is the P2 defect and it needs one assertion each,
   not a single combined case.
4. Two contracts differing only in a field the old `strategy/registry.py:62` hash ignored
   (universe, dates, costs, seed) produce DIFFERENT hashes. This is the regression test for P2.
5. `run_backtest()` without a contract raises.
6. **`run_tilt()` without a contract raises.** This is the regression test for P1 and is the most
   important test in the suite.
7. A completed `run_tilt()` writes a `TrialRecord` whose `contract_hash` matches the declared
   contract, with a non-null `seed` and a non-null `git_sha`.
8. A sweep declaring `n_planned_trials=k` that attempts trial `k+1` raises.
9. `holdout intent = never` combined with `--allow-holdout` is refused, and so is
   `reading_now` without the flag.
10. Every `TrialRecord` written by `walkforward`, including the POOLED record, has all 12
    Amendment-1 provenance fields populated. This is the regression test for P4.
11. `research/sweep.py:config_hash` no longer exists.

---

# AMENDMENT 1 — 2026-08-20. Three ambiguities, reported by a test author, resolved.

An independent test author writing from this spec alone found three places where the spec did not
determine the tests. All three are resolved here rather than left to be guessed differently by two
suites — an ambiguity resolved two ways is a disagreement the dual-suite rule cannot adjudicate,
because both readings would be faithful.

## 1. `seed` had no home. It becomes a TOP-LEVEL field, not a section member.

Obligation 3 requires `contract_hash` to change when `seed` changes, but the six-section table
never lists `seed`. The reporting author placed it in `execution` as a documented assumption; that
is a reasonable reading and it is not the one adopted.

**Resolution: `seed` is a required top-level field of `ResearchContract`, sibling to the six
sections, and is included in `contract_hash`.**

The reason is not tidiness. A seed in this repo cuts across sections: it drives bootstrap
resampling and permutation nulls (validation), CSCV split assignment (validation), AND any
stochastic tie-break inside a strategy (portfolio/execution). Filing it under one section asserts
it belongs to that stage, which would be false, and the first person to add a seeded step in a
different stage would either mis-file it or silently add a second seed. A cross-cutting field gets
a cross-cutting home.

Obligation 3 is unchanged in substance: one test per field, `seed` included.

## 2. The module is `nifty_quant/research/contract.py`.

Unspecified before; matches its siblings `research/tilt.py`, `research/registry.py`,
`research/splits.py`. `ResearchContract` and `contract_hash` are exported from there.

Worth stating why this mattered: with the module unnamed, a RED suite fails on `ImportError` at a
guessed path, and a later real implementation at a different path leaves it RED for a reason that
looks identical to a genuine gap. The suite would stay red and mean nothing.

## 3. Obligation 10 is tested through the CLI, not through an invented function.

The author had to invent `run_walkforward_to_registry(registry=...)` because `walkforward` writes
its records inline in `cli.py` (`:1060-1082`, `:1268-1289`) and exposes no callable seam.

**Do NOT add a function purely to be testable here.** Obligation 10 is tested by invoking
`nq walkforward` through `CliRunner` with `RESULTS_ROOT` redirected, then opening the resulting
registry and asserting on the POOLED row's 12 provenance fields. That tests the path production
actually takes; a bespoke seam would test a path nothing uses.

Restated obligation 10: *invoking `nq walkforward` end-to-end produces a pooled `TrialRecord`
whose 12 Amendment-1 provenance fields are all populated, with non-null `seed` and `git_sha`.*

## 4. Consequential note on the GREEN test

The author included one deliberately GREEN test asserting the existing `TrialRecord` has exactly
12 Amendment-1 fields, to distinguish a future field-list drift from a real P4 regression. That is
good practice and it stays. It is the only test in the suite permitted to be green before
implementation, and the reason is recorded here so a later reader does not "fix" it.
