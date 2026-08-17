# Spec: per-trial return-series persistence (`returns.parquet`)

**Status:** contract for TDD. Tests written from this document alone, before implementation.

## Why this exists

`backtest/metrics.py` implements the full overfitting-control suite — `deflated_sharpe`,
`expected_max_sharpe`, `effective_n_trials`, `pbo_cscv` (Combinatorially Symmetric CV). All of
it is unreachable in practice, because **the per-trial return series is never persisted**.

Verified 2026-08-17: `results/trials/<hash>/result.parquet` is a **trade-fill log**
(`ts, symbol, qty, price, notional, is_buy, charges, decision_price, fill_price, shortfall_bps,
participation, filled_frac`; 28,656 rows in a real trial), written by
`result.trades.to_parquet()` at `cli.py:506`. No return-series artifact exists anywhere.
Both the README and `.claude/progress/verification-and-strategies.md:38-39` assert otherwise;
they are wrong.

Second defect: `sweep` and `walkforward` — the commands that generate the MANY trials PBO needs
— never set `result_path` at all. Only single `backtest` runs do.

Net effect: the Probability of Backtest Overfitting has been decorative for the life of this
repo. Phase 3 tests five hypothesis families, so multiple-testing control is the difference
between a finding and a false positive.

`research/registry.py` already implements the consumer side (`TrialMatrix`,
`TrialRegistry.build_trial_matrix`) against the contract below, and currently drops all 10 real
trials with `"missing returns.parquet (incomplete trial artifact)"`. This spec is the producer.

## The artifact

Path: `<result_path>/returns.parquet`, alongside the existing `result.parquet`.

Schema — exactly two columns, in this order:

| column | dtype | meaning |
|---|---|---|
| `ts` | int64 | epoch SECONDS UTC, one row per TRADING DAY, strictly increasing, unique |
| `return` | float64 | that day's simple return, compounded from decision-row returns |

**The daily basis is mandatory and is the whole point.** `BacktestResult.returns` is one entry
per DECISION ROW plus one unconditional final row — NOT per day. Annualizing those at 252 is
silently wrong. `cli._daily_returns` (`cli.py:68-137`) already performs the correct
reconstruction and compounding; **reuse it, do not reimplement it**. The stored series must be
exactly what `cli` itself feeds to `compute_metrics`.

`ts` is the session date at 00:00:00 UTC — a day label, not a bar timestamp. Derive it from the
session dates the engine already carries; never from a fixed 375-bar stride (CLAUDE.md rule 5).

float64 at rest here, deliberately overriding the float32-at-rest convention: this is a
compounded P&L series, and CLAUDE.md rule 3 forbids accumulating returns in float32.

## Required changes

1. **`backtest` command** — after writing `result.parquet`, write `returns.parquet` from the
   same `_daily_returns` output already computed for the metrics block. No recomputation.
2. **`walkforward` command** — persist a per-split artifact directory and set `result_path` on
   every `TrialRecord`. Each split writes its own `returns.parquet` covering that split's TEST
   window only. Splits must not overlap in `ts`.
3. **`sweep` command** — same: every trial in the grid gets an artifact dir, a `result_path`,
   and a `returns.parquet`.
4. A `ruined` run (`BacktestResult.ruined is True`) must STILL write its returns, and the fact
   must be recorded — the ruin guard zeroes returns after `ruin_index`, so a consumer that
   cannot tell a ruined series from a flat one would compute a meaningless Sharpe. Record
   `ruined` and `ruin_index` in the `TrialRecord` (or artifact metadata) so
   `build_trial_matrix` can drop or flag it. Do NOT silently skip writing.

## Required tests (`tests/test_returns_persistence.py`)

Synthetic panels and `tmp_path` only. Never write into the real `results/`.

### Artifact shape
1. `test_backtest_writes_returns_parquet_next_to_result_parquet`
2. `test_returns_parquet_has_exactly_ts_and_return_columns_in_order`
3. `test_ts_is_int64_epoch_seconds_utc_at_midnight`
4. `test_return_is_float64`
5. `test_ts_is_strictly_increasing_and_unique`
6. `test_one_row_per_trading_day_not_per_decision_row` — the core distinction; build a run with
   many decision rows per day and assert `len(returns) == n_trading_days`.
7. `test_stored_series_equals_cli_daily_returns_output` — byte-for-byte against
   `cli._daily_returns` on the same result. This is what stops a reimplementation drifting.
8. `test_irregular_session_produces_one_row` — a 60-bar Muhurat session yields exactly one row.

### Round trip with the consumer
9. `test_build_trial_matrix_accepts_written_artifact` — write via the producer, read via
   `TrialRegistry.build_trial_matrix`, assert `n_dropped == 0`.
10. `test_pbo_cscv_returns_finite_value_on_written_trials` — >= 2 trials, assert the result is
    finite and in [0, 1]. This is the end-to-end proof the machinery now works.

### walkforward / sweep
11. `test_walkforward_sets_result_path_on_every_trial_record`
12. `test_walkforward_writes_one_returns_parquet_per_split`
13. `test_walkforward_split_ts_ranges_do_not_overlap`
14. `test_sweep_sets_result_path_on_every_trial_record`
15. `test_sweep_writes_returns_parquet_per_trial`

### Ruin
16. `test_ruined_run_still_writes_returns`
17. `test_ruined_flag_and_index_are_recorded`
18. `test_consumer_can_distinguish_ruined_from_flat_series`

### Determinism
19. `test_two_identical_runs_write_identical_bytes`

## Constraints

- Reuse `cli._daily_returns`; do not reimplement daily compounding.
- Never modify `data/` (CLAUDE.md rule 2). Tests write only under `tmp_path`.
- float64 for returns throughout (rule 3).
- Never assume a 375-bar session stride (rule 5).
- `ruff check` clean; every function annotated (`disallow_untyped_defs = true`).
- Do not lower coverage on `cli.py` (floor 60) or `research/registry.py` (floor 97).
