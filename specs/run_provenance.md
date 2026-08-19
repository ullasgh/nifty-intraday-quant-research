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
