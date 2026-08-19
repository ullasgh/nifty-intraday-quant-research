# Spec: daily results as a first-class backtest output

Phase A track A3.

## Why this exists

`BacktestResult.equity_curve` / `.returns` / `.turnover` (`backtest/engine.py:56-73`) carry one
entry per DECISION row plus one unconditional final row (`engine.py:484-486`, `:627`). They are
not daily, and annualizing them at 252 is wrong.

The repo already knows this. `cli.py:100-266` carries three private helpers — `_daily_returns`,
`_daily_turnover`, `_daily_return_ts` — that **reconstruct** the engine's row selection from
outside the engine, by re-deriving decision rows from `Strategy.data_request().decision_times`
and `Panel.rows_at_time`, then re-mapping rows to days with `np.searchsorted`. They raise
`ValueError` on any length mismatch, which is the honest failure of a fragile design: the engine
knows exactly which rows it appended and the CLI has to guess.

Two consequences:
1. Anything calling `run_backtest` directly — every test, every script, every future research
   module — gets decision-row returns and must re-derive the daily series itself or silently
   annualize by the wrong factor.
2. The reconstruction is a second implementation of a selection rule that already exists inside
   the engine, so the two can drift.

There is no daily EQUITY object anywhere; `cli.py:267 _write_returns_parquet` writes a daily
return vector and nothing else.

## Required behaviour

### A. The engine records the day index it is already standing on

The engine loop already computes `day_idx = int(day_index[t])` on every row (`engine.py:403`).
Wherever it appends to `equity_vals` / `gross_vals` / `turnover`, it also appends the current
`day_idx` and the row index `t`. No reconstruction, no searchsorted, no re-derivation of decision
times.

### B. `DailyResult`

A frozen dataclass in a new module `src/nifty_quant/backtest/daily.py`:

    @dataclass(frozen=True)
    class DailyResult:
        dates: np.ndarray        # int64 epoch-seconds, one per trading day, ascending
        equity: np.ndarray       # float64, end-of-day equity
        returns: np.ndarray      # float64, compounded within the day
        gross_returns: np.ndarray
        turnover: np.ndarray     # float64, SUMMED within the day (additive, never compounded)
        n_days: int

Aggregation rules, which are not symmetric and must not be unified:
- **Returns compound** within a session. Reuse `metrics.aggregate_returns_by_group`.
- **Turnover sums** within a session. It is notional churned; compounding it is meaningless.
  The existing rationale at `cli.py:170-175` is correct and must survive the move.
- **Equity** takes the LAST value within the session.
- **Dates** are the session dates from `panel.dates`, not synthesised UTC midnights derived from
  a row timestamp. `_daily_return_ts`'s "best-effort fallback ... repeats day zero as needed"
  (`cli.py:200-211`) is a workaround for the reconstruction problem and is deleted, not ported.

### C. Wiring

- `BacktestResult` gains `daily: DailyResult`.
- `cli.py` stops reconstructing. `_daily_returns`, `_daily_turnover`, `_daily_return_ts` and
  `_decision_and_final_rows` are deleted; their five call sites (`cli.py:576-577`, `:934-936`,
  `:1053`, `:1290`, and `_write_returns_parquet`) read `result.daily` instead.
- Every `compute_metrics` call keeps receiving daily returns, so **published metrics must not
  move**. This is a refactor with a correctness dividend, not a change of numbers.

### D. Sessions with no decision row

A session in the panel on which the engine appended nothing (possible for a checkpoint strategy
whose decision time does not exist in a shortened session — Muhurat is 60 bars, a
disaster-recovery session can be 105) must appear in `DailyResult` with **zero return, zero
turnover and carried-forward equity**, not be omitted. Omitting it silently shortens the series
and changes annualization. Rule 5 forbids assuming a session length; this is the same failure in
a different disguise.

## Required tests

1. **Daily returns match the old reconstruction.** On a synthetic multi-session panel, the new
   `result.daily.returns` equals what compounding the decision-row returns by day produces.
2. **Turnover sums, returns compound.** A day with two decision rows of turnover 0.1 and 0.2
   yields daily turnover 0.3; two returns of +1% yield 1.01*1.01-1, not 0.02.
3. **Equity is end-of-day**, matching the last decision-row equity of that session.
4. **Irregular sessions.** A panel containing a 60-bar and a 105-bar session alongside full ones
   produces one daily row per session, in date order, with no length assumption anywhere.
5. **Session with no decision row** yields a zero-return, zero-turnover, carried-equity row.
6. **`n_days` equals `len(panel.dates)`** for a backtest spanning the whole panel.
7. **Metrics unchanged.** `compute_metrics(result.daily.returns)` equals
   `compute_metrics(old_reconstruction(result.returns))` on the same backtest, to float equality.
8. **Empty panel / empty returns** produce an empty `DailyResult` without raising.
9. **Dates are the panel's session dates**, not derived from row timestamps.

## Constraints

- Rule 4: `ts` is int64 epoch-seconds UTC with no timezone attached.
- Rule 5: no fixed 375-bar stride; index via `panel.day_offsets` and `panel.dates`.
- Rule 3: float64 for anything accumulating P&L or returns.

---

# AMENDMENT 1 — 2026-08-19. Five gaps found by a test author before implementation.

## 1. Section C and section D CONTRADICT each other. Section D wins.

Section C promises "published metrics must not move ... a refactor with a correctness dividend,
not a change of numbers." Section D requires that a session with no decision row appears with a
zero return rather than being omitted. **Both cannot hold.** Adding a period changes the period
count, which changes annualization, which moves every annualized metric.

Resolution, and the promise is narrowed rather than the behaviour:

- For a backtest in which EVERY session has at least one decision row — the normal case, and the
  case every published number in this repo comes from — metrics must be **bit-identical**. That is
  the regression guard and it stays.
- For a backtest containing decision-less sessions, metrics **are expected to move**, and the
  movement is a **correction**: omitting a flat session understates the denominator and therefore
  overstates annualized return and Sharpe. Section C's blanket promise was wrong.

Required test 7 is split accordingly: identical metrics on the all-sessions-have-decisions panel,
and a demonstrated, explained move on a panel with a decision-less session.

## 2. `dates` conversion convention was never stated

`DailyResult.dates` is typed int64 epoch-seconds; `Panel.dates` holds `datetime.date`. The
conversion was only implied. **Normative: UTC midnight of the session date**, i.e.
`datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()`, as int64.

This is consistent with rule 4 (`ts` is int64 epoch-seconds UTC with no timezone attached) and
with what `_daily_return_ts` did before deletion. It is a DATE key, not a bar timestamp; nothing
should ever convert it to Asia/Kolkata and read a trading time off it.

## 3. `gross_returns` had no aggregation rule

Listed as a field, with no rule. **It compounds within the session, exactly like `returns`** — it
is the same quantity measured before costs, so aggregating it differently would make
`returns` and `gross_returns` incomparable, and their difference is how cost drag is read.

## 4. A decision-less FIRST session has nothing to carry forward

Section D says equity is "carried forward". If the first session of the panel has no decision row
there is no prior equity. **Normative: it carries `config.capital`** — the book is flat and
untraded, so its equity is the initial capital by definition. Add a required test.

## 5. No named construction entry point

The spec described `DailyResult` as a dataclass and its population as happening inside the engine,
leaving no way to build one from raw per-row state in a test. Add:

    def build_daily(
        row_day_index: np.ndarray,   # the day_idx the engine recorded per appended row
        equity: np.ndarray,
        returns: np.ndarray,
        gross_returns: np.ndarray,
        turnover: np.ndarray,
        dates: np.ndarray,           # int64 epoch-seconds, one per SESSION in the panel
        *,
        initial_capital: float,
    ) -> DailyResult

in `backtest/daily.py`. The engine calls it; tests call it directly. This is the seam the
`order_lifecycle` amendment warned about in general terms: **a required behaviour that cannot be
observed through any existing API means the spec is asking for a seam it did not declare.**
Declaring it beats letting two authors each invent a different one.

Note that `dates` carries one entry per session in the PANEL, not per session that produced rows.
That is what makes defect 4 and section D expressible at all.
