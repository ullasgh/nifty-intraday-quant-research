# nifty_quant

`nifty_quant` is a research toolkit for NSE India intraday (1-minute) equity data. It includes a data pipeline, feature library, strategy framework, a vectorized backtest engine with realistic cost and execution modelling, and a walk-forward research harness with overfitting controls (deflated Sharpe, PBO via CSCV, holdout lock). The dataset covers 153 symbols (149 tradable equities plus 4 index/vol series: NIFTY50, NIFTY100, NIFTYBANK, INDIAVIX), contains 121.3M 1-minute bars, and spans 2017-07 through 2026-08. Spot bars are corporate-action adjusted; NSE futures bars are unadjusted (see Limitations).

## Quickstart

```bash
uv sync                                   # install deps (typer, numpy, pandas, ...)
uv run nq info                            # resolved paths + existence
uv run nq symbols                         # symbol counts + manifest coverage
uv run nq build-panel --freq 1 --years 2024   # build the memmapped panel cache
uv run nq validate --year 2024 --symbols RELIANCE   # data-quality checks
uv run nq strategies                      # list registered strategies + params
uv run nq backtest --strategy volume_breakout \
    --config configs/strategies/volume_breakout.yaml \
    --start 2024-01-01 --end 2024-12-31   # gross vs net, costs, breakeven, latency
```

Run `uv run nq --help` for the full command list (`symbols`, `build-panel`, `validate`, `strategies`, `backtest`, `walkforward`, `sweep`, `cache info` / `cache gc`).

## Architecture

### nifty_quant/data/

Manifest-driven ingestion (`manifest.py`), a memmapped per-year `.npy` panel cache built from raw parquet bars (`panel_builder.py`), a dense aligned multi-symbol `Panel`/`PanelSpec` abstraction with a memory guard (`panel.py`), fixed-time-of-day checkpoint extraction (`checkpoints.py`), and data-quality checks that never repair or forward-fill (`validate.py`).

### nifty_quant/calendar.py

Session calendar derived from actual NIFTY50 bar timestamps, not an assumed fixed stride. Classifies every session (regular, Muhurat, special session, halted, corrupt, minor).

### nifty_quant/features/

Vectorized, causal-by-construction feature functions (`core.py`: volume z-score, breakout, Parkinson volatility, deseasonalization) and persistence/Hurst estimators (`persistence.py`).

### nifty_quant/strategy/

`Strategy` ABC plus a pydantic `Params` schema (`base.py`), a name-keyed registry (`registry.py`), and pluggable strategy implementations under `strategy/plugins/` (`volume_breakout`, `xsec_zscore`), auto-registered on import.

### nifty_quant/execution/

Realistic NSE cost models (`costs.py`: `NSEIntradayEquityCosts`, `NSEDeliveryEquityCosts`, `breakeven_cost_bps`) and fill/slippage models (`fills.py`).

### nifty_quant/backtest/

The event-driven-semantics-but-vectorized engine (`engine.py`: mandatory one-bar execution lag, decision-latency sweep, forced EOD liquidation) and performance/overfitting statistics (`metrics.py`: Sharpe, Sortino, deflated Sharpe, PBO via CSCV, `verdict_line`).

### nifty_quant/research/

Walk-forward splitting with embargo (`splits.py`: `WalkForwardSplitter`, `HoldoutLock`), a SQLite trial registry (`registry.py`), and parameter-grid sweep expansion (`sweep.py`).

### nifty_quant/universe/

Static/current equity universe, config-file-driven universes, and survivorship-gap diagnostics (`static.py`).

### nifty_quant/guards.py

Runtime contract decorators (see the Guard layer section below).

### nifty_quant/cli.py

The `nq` command-line entry point tying all of the above together.

## Measured findings

- 105 of 2,250 sessions are irregular (Muhurat sessions ~60 bars, NSE DR Saturday special sessions 105 bars, COVID-era halts, and two corrupt 2017 days); 74 of those 105 fall in 2017 alone, which is why `DEFAULT_RESEARCH_START = 2018-01-01` (`nifty_quant.calendar.DEFAULT_RESEARCH_START`).
- 100% of OHLC violations found (61 of 61, across 60 symbols, in 2024) occur at the 09:15 bar, always as `close > high` — this is pre-open call-auction price leakage into the first minute bar, not a data-quality defect elsewhere. The earliest honest entry checkpoint is therefore **09:16**, not 09:15 (see `xsec_zscore`'s `entry_time` default).
- Never assume a fixed 375-bar session stride; index sessions via `day_offsets` (`SessionGrid`/`TradingCalendar`), because irregular sessions break that assumption silently.
- Spot bars are corporate-action adjusted; NSE futures bars are **not** adjusted, and this repo has **no corporate-actions table**. Any feature that compares spot to futures levels is therefore blocked until that table exists — do not build one assuming the two series are on the same basis.
- A naive rolling Hurst estimator (lag-variance, per-window `np.polyfit`) is statistically broken at short windows: on a true-H=0.5 random walk, a 30-bar window returns H = 0.090 (mean), so a `H > 0.55` breakout filter never fires at that window length. At window 390 (one NSE session) the estimator recovers: median H ~= 0.467, p90 ~= 0.556 — so `hurst_threshold = 0.55` at `hurst_window = 390` is a sensible ~90th-percentile cut, not an arbitrary one.
- Concretely: `volume_breakout` on RELIANCE, 2024, produces **0 signals** as originally specified (window too short for the Hurst filter to ever fire), **219 signals** once corrected to use the 390-bar Hurst window, and **1,844 signals** with the Hurst filter removed entirely. This is the single clearest illustration in the repo of why the window-length bug mattered.
- NSE intraday (MIS) round-trip cost is **8.3 bps** at ₹1,00,000 notional and **4.0 bps** at ₹10,00,000 notional (`NSEIntradayEquityCosts.round_trip_bps`); market impact dominates above that size, which `SqrtImpactSlippage` / `breakeven_cost_bps` account for separately from these flat/percentage charges.
- A real end-to-end `nq backtest` run (`volume_breakout`, 8 symbols, full year 2024, real NSE costs) confirms the pipeline works but does **not** show a working strategy: gross Sharpe **-0.048**, net Sharpe **-0.233**, ~₹541k in total costs, 21,708 trades, and **73.7%** of desired notional unfilled. Treat this as proof the plumbing (data, costs, fills, reporting) is verified, not as evidence `volume_breakout` is profitable — the honest next step is fixing its sizing/churn (21.7k trades on 8 symbols in a year is very high turnover for the signal counts above), not trusting the current numbers.

## The guard layer

`nifty_quant/guards.py` provides runtime contract decorators. The `NQ_STRICT` environment variable controls how much checking is enabled: `0` (OFF), `1` (CHEAP, the default), `2` (FULL). The `causal()` decorator's lookahead probing only runs at FULL strictness, because it perturbs future rows and re-calls the function — not cheap.

`@causal` wraps a feature function. It probes the function by perturbing rows after the current one and re-calling it, then raises `ContractViolation` if the output for row `t` changes. This enforces that row `t` of the output can only depend on rows `<= t`. This is not a hypothetical safeguard: it caught a real full-sample lookahead bug in the volume z-score feature — a full-sample-mean deseasonalization leak that a naive per-call check would have missed. That is why the decorator exists on every feature function in `features/core.py` and `features/persistence.py`.

`float32` at rest / `float64` in motion: the panel cache stores OHLCV as `float32` (`panel_builder.py`, `FIELDS`) to keep the ~121M-row dataset memory-manageable, but every numerical computation (features, backtest engine) upcasts to `float64` before doing arithmetic, to avoid float32 precision loss compounding across a session or a multi-year backtest.

NaN means "no bar" and is **never** forward-filled anywhere in `data/validate.py` or the panel-building path — a missing bar stays missing. Strategies and features must handle NaN explicitly (e.g. via `ffill=True` on a specific `MarketView` call), never rely on an implicit fill.

## Limitations

- The tradable universe (`equity_symbols()` / `configs/universe/all_equity.yaml`) is a **fixed, current-day universe**, not a point-in-time one. Backtests before roughly 2021 are survivorship-inflated because delisted/renamed names are not in it; `Universe.as_of()` and `survivorship_report()` are data-availability proxies, not real index membership history, and every `nq backtest` / `nq walkforward` run prints a `survivorship_report(...).warning_line()` for exactly this reason — read it, don't just skip past it.
- **No point-in-time index membership** table exists for NIFTY50/NIFTY100/NIFTYBANK; `INDEX_SYMBOLS` in `universe/static.py` are excluded from `equity_symbols()` precisely because they are not tradable constituents, but there is no way to reconstruct "who was actually in the index on date X."
- **No corporate-actions table** — splits/bonuses/dividends are baked into the spot adjustment already applied upstream, but there is no explicit table of what action happened when, which blocks any spot-vs-futures feature (see Measured findings above) and any analysis that needs to know the adjustment factor itself rather than just its already-applied effect.
- `.git` is about 1.5 GB because the parquet bar files are committed directly to the repository without Git LFS. Cloning is slow; do not add more raw data files to git history without addressing this first.
