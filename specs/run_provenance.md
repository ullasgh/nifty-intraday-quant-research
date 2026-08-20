# Spec: run provenance and experiment lineage

Phase B track B3.

## Why this exists

`TrialRegistry` (`research/registry.py:96-165`) is real: SQLite, append-only, unique on
`(config_hash, split_id)`. The structure is fine. What it records is not enough to answer "why
did trial X win", and one field actively lies.

`TrialRecord` (`registry.py:16-34`) holds `config_hash`, `ts`, `strategy`, `params_json`,
`split_id`, `purpose`, `sharpe_gross`, `sharpe_net`, `n_trades`, `turnover`, `breakeven_bps`,
`git_sha`, `data_fingerprint`, `code_version`, `wall_s`, `result_path`, `error`, `ruined`,
`ruin_index`.

Missing entirely: **seed**, **universe hash**, **cost-model identity**, **slippage-model
identity**, **date range**, **panel hash**, **parent trial**, **feature version**, **embargo
components**.

Worse: `git_sha` is a column that is **hard-coded `None` at all seven write sites**
(`cli.py:732`, `:766`, `:977`, `:1144`, `:1332`, `:1358`). Live evidence: `results/trials.db`
has 13 rows and the sampled rows carry `git_sha=None` and `data_fingerprint=None`. A provenance
field that is always null is worse than an absent one, because a reader assumes it means
"unknown commit" rather than "never recorded".

And registry writes are swallowed: `cli.py:775-776` is `except Exception: pass`. A research run
whose lineage silently failed to record looks identical to one that recorded fine.

`data_fingerprint` (from `Manifest.fingerprint`, `data/manifest.py:29,60`) is a whole-dataset
fingerprint. It does not identify the PANEL actually loaded — symbols, date range, fields,
adjustments — so two runs on very different slices of the same dataset are indistinguishable.
`Manifest.cache_key()` (`manifest.py:80-83`) hashes `fingerprint|adjustments|resolution|
PANEL_VERSION` for cache invalidation, which is close but is not the loaded slice either.

## Required behaviour

### A. Fields added to `TrialRecord`

    seed: int | None
    universe_name: str
    universe_hash: str            # from specs/pit_universe.md section D
    panel_hash: str               # see B below
    start: str                    # ISO date of the data window actually used
    end: str
    cost_model_id: str            # class name + non-default field values, canonicalised
    slippage_model_id: str
    fill_model_id: str
    embargo_components: str       # JSON: the four terms from specs/embargo_sizing.md
    parent_trial_id: str | None   # the trial this one was derived from, for lineage chains
    feature_version: str

### B. `panel_hash`

A hash of the LOADED SLICE, not the dataset: sorted symbol list, first and last `ts`, row count,
field names, adjustment settings, and `PANEL_VERSION`. Cheap to compute (no bar data is hashed),
and it distinguishes runs that `data_fingerprint` cannot. Both are recorded; they answer
different questions.

### C. `git_sha` is populated or the run is marked dirty

Read the actual HEAD sha. If the working tree is dirty, record `"<sha>-dirty"` — do not record a
clean sha for a modified tree, and do not record `None`. If git is unavailable, record
`"no-git"`. Every one of the seven write sites.

### D. Registry write failures are visible

`except Exception: pass` (`cli.py:775-776`) becomes: log the exception, set a non-zero
`registry_write_failed` marker on the run output, and continue. Research must not be blocked by a
registry problem, but it must not silently lose lineage either.

### E. Provenance block in `metrics.json`

`results/trials/<config_hash>/metrics.json` currently holds 39 performance keys and no
provenance. Add a `provenance` object carrying the fields above, so an artifact on disk is
self-describing without a database lookup.

## Required tests

1. Every field round-trips through the registry and back.
2. `git_sha` is never `None` after a write; a dirty tree yields a `-dirty` suffix.
3. `panel_hash` differs for two runs on different date ranges of the same dataset, where
   `data_fingerprint` is identical. This is the specific failure `panel_hash` exists to fix.
4. `panel_hash` is stable across runs on the same slice.
5. `cost_model_id` distinguishes `NSEIntradayEquityCosts()` from one with a non-default field,
   and is identical for two default instances.
6. A registry write failure is surfaced, not swallowed, and does not abort the backtest.
7. `metrics.json` contains a complete provenance block.
8. `parent_trial_id` chains: a derived trial points at its parent and the chain is walkable.
9. A trial written by `nq sweep` carries the same provenance completeness as one from `nq
   backtest` — the sweep path is where lineage matters most and is currently the thinnest.

## Constraints

- Hashing must not read bar DATA (too slow at 121M rows); metadata only.
- No behaviour change to the numbers a run produces. This spec adds records, not arithmetic.

---

# AMENDMENT 1 — 2026-08-20. Nine ambiguities from a test author, all resolved here.

The author was right that items 6 and 7 could not be tested as written. It responded by writing
EMPTY PLACEHOLDER TESTS THAT PASS, which is the wrong answer — an empty passing test reports
coverage that does not exist, and this program has been bitten by that five times. **The correct
response to an untestable spec item is a test that FAILS loudly with the reason, or no test plus a
reported gap. Never a green no-op.** The spec is fixed below so both items become testable.

## 1. `cost_model_id` / `slippage_model_id` / `fill_model_id` — canonical form

    "<ClassName>(<k=v for each field differing from the dataclass default, sorted by k>)"

Examples: `NSEIntradayEquityCosts()` for a default instance;
`NSEIntradayEquityCosts(brokerage_flat=10.0)` for one non-default field. Deterministic, stable
across processes, and human-readable in a `metrics.json` a person is reading six months later.
Floats format with `repr()` so round-tripping cannot lose precision.

## 2. `embargo_components` — the four keys, inlined

    {"feature_lookback": <float sessions>, "label_horizon": <float sessions>,
     "holding_period": <float sessions>, "execution_horizon": <float sessions>}

Units are SESSIONS, floats, pre-`ceil` (the ceiling is applied once at the sum — see
`specs/embargo_sizing.md` amendment 1). Serialised with sorted keys so the JSON string is stable.

## 3. `universe_hash` — inlined rather than cross-referenced

blake2s over the canonical JSON of: universe name, the per-session sorted eligible symbol sets,
and the eligibility parameters. Cross-referencing `specs/pit_universe.md` was not enough; a spec
that cannot be implemented without opening another document is under-specified.

## 4. "Same slice" for `panel_hash` stability

Two runs are the same slice iff ALL of: sorted symbol list, first and last `ts`, row count, field
names, adjustment settings, `PANEL_VERSION`. **Nothing else** — not wall-clock time, not the
output directory, not the trial id. If any of those six differ the hash MUST differ; if none do it
MUST match.

## 5. Item 6 — the registry-failure marker, made concrete

On a registry write failure the run must:
1. log the exception at WARNING with the trial id (never swallow it — `cli.py`'s bare
   `except Exception: pass` is the defect),
2. set `registry_write_failed: true` in `metrics.json`,
3. still write the trial artifact directory and still return a successful backtest.

So the test is: monkeypatch `TrialRegistry.record` to raise, run a backtest, assert the command
exits 0, the artifact exists, `metrics.json["registry_write_failed"] is True`, and a warning was
emitted. **`registry_write_failed` is `false` on every normal run**, so it is a field with two
observable states rather than a flag that only ever appears on failure.

## 6. Item 7 — "complete provenance block", enumerated

`metrics.json["provenance"]` is an object containing EXACTLY these keys, every one present, with
`null` permitted only for `seed` and `parent_trial_id`:

    config_hash, git_sha, code_version, data_fingerprint, panel_hash, universe_name,
    universe_hash, seed, start, end, cost_model_id, slippage_model_id, fill_model_id,
    embargo_components, parent_trial_id, feature_version

A test asserting the exact key SET (not merely a subset) is required, so a later field addition
cannot silently drop one.

## 7. Item 8 — chain depth

A three-generation chain (grandparent -> parent -> child) must be walkable. Two levels is not
enough to catch an implementation that stores the parent but cannot follow more than one hop.

## 8. Item 9 — what makes a trial "sweep-derived"

Its `parent_trial_id` is non-null and points at the sweep's base trial. The requirement is that
`nq sweep` populates the SAME provenance fields as `nq backtest` — no field may be left null by
the sweep path that the backtest path fills.

## 9. Item 1 — round-trip semantics

BYTE-FOR-BYTE for stored strings; exact equality for ints; `repr()`-stable for floats. Not
"re-computed equivalence" — the registry is a record of what happened, so a value written and read
back must be identical, never merely equivalent.

---

# AMENDMENT 2 — 2026-08-20. `parent_trial_id` semantics, pinned.

The implementer found a real hole and reported it instead of papering over it: `nq sweep` sets
`parent_trial_id` to the base-params config hash, but **no row for those base params is ever
written**, so the chain is structurally unwalkable while every field is literally correct.
Amendment 1 item 7 demanded a "walkable" three-generation chain without saying what walkable means
when a link has never been run.

**Resolution — define the semantics rather than add a field or churn either suite.**

`parent_trial_id` references a **CONFIG HASH**, not a guaranteed row. That is coherent because
trials are keyed by `config_hash` in the first place: naming a parent config is meaningful whether
or not that config was itself executed.

Walkability is therefore defined as: **a chain of trials that were ALL RUN is fully walkable, and a
walk terminates cleanly at the first ancestor that was never executed.** A sweep's base params are
a configuration the sweep varies FROM, not a run that happened, so a walk terminating there is the
honest answer — not a defect to engineer around.

Required consequences:
- The registry exposes a walk that returns the ancestors it can resolve and reports where it
  stopped and why. It must NOT raise on a missing ancestor, and must NOT silently return a short
  chain as if it were complete — the caller has to be able to tell "root reached" from
  "ancestor never run".
- Required test 8's three-generation chain uses trials that WERE all recorded, so it is fully
  walkable and unaffected.
- Do NOT fabricate a base-params row to make the sweep chain walk. A trial row is a record that
  something RAN; inventing one for a configuration that never executed would put a lie in the
  audit trail to satisfy a pointer.

**The general principle, and it is the same one this program has now applied several times:** a
reference that cannot be resolved is worse than an absent one *only when it pretends otherwise*.
Making the semantics explicit — config reference, walk-until-unrun — removes the pretence without
removing the information.

## Suite B's CLI fixture cannot reach the code it tests

`tests/test_run_provenance_b.py::_make_panel()` builds `Panel(fields={}, symbols=(), ...)` — an
genuinely empty panel — and `_backtest_args()` never passes `--no-tradable-filter`. The CLI's
`tradable_mask` therefore calls `panel.field("close")` and dies with
`Field 'close' not found. Available fields: ` before any provenance code runs. Five of its tests
fail this way against ANY correct implementation.

Suite A's equivalent fixture populates real OHLCV and always passes `--no-tradable-filter`, which
is why its 17 tests reach the code under test. Suite B's fixture is corrected to match. **This is
the seventh degenerate-fixture instance in this program** — and the first where the fixture was not
merely degenerate but structurally unable to execute the path under test at all.
