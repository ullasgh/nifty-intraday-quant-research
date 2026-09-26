# Spec: VWAP ± k·ADR fade (`nifty_quant.research.vwap_adr`)

Base intraday mean-reversion study. When a stock's 1-minute close stretches far from its
session VWAP -- "far" measured in units of its own Average Daily Range (ADR) -- fade the
move; exit when price comes back to VWAP, else square off at 15:20 IST. No stop loss.

This is a **research module**, separate from the `Strategy`/engine plugin framework. It
does not modify, import from, or depend on `strategy/plugins/vwap_reversion.py`.

All functions take plain numpy arrays (not a `Panel`) so they are testable on tiny
synthetic inputs. Shapes: `(n_rows, n_sym)` for per-bar arrays, `(n_days, n_sym)` for
per-session arrays. `day_offsets` is int, length `n_days + 1`, `day_offsets[0] == 0`,
strictly increasing, `day_offsets[-1] == n_rows`; session `d` covers rows
`day_offsets[d] : day_offsets[d+1]`. Sessions have **arbitrary** lengths (60, 105, 375 ...)
-- never assume a fixed stride. `minute_of_day` is int, length `n_rows`, IST minutes
since midnight of each row's bar label (09:15 -> 555, 09:16 -> 556, 15:20 -> 920).
Bars are left-labelled: the bar labelled T covers [T, T+60s).

A bar is **present** for (row, sym) iff `close[row, sym]` is finite. NaN means "no bar".
Nothing in this module forward-fills, back-fills, or interpolates any input.

All arithmetic is float64; float32 inputs are upcast before any computation. Every
function returns float64 arrays (or a DataFrame with float64 numeric columns).

Constants (module level, `features.py`):
- `SESSION_START_MINUTE = 556` (09:16). The 09:15 bar is excluded everywhere: it is measured
  to carry pre-open call-auction leakage (close > high in 61/61 observed OHLC violations).
- `SQUARE_OFF_MINUTE = 920` (15:20). NSE MIS square-off.

---

## 1. `features.py`

### 1.1 `session_vwap(high, low, close, volume, minute_of_day, day_offsets, start_minute=SESSION_START_MINUTE) -> np.ndarray`

Session-anchored cumulative VWAP, shape `(n_rows, n_sym)`.

- Typical price `tp = (high + low + close) / 3`.
- Row `r` of symbol `s` **contributes** iff `minute_of_day[r] >= start_minute` AND
  `high, low, close, volume` are all finite at `(r, s)` AND `volume[r, s] >= 0`.
- Within each session, `cum_pv[r] = Σ tp·volume` and `cum_v[r] = Σ volume` over
  contributing rows from the session's first row up to and including `r`. Sums reset at
  every session boundary (`day_offsets`).
- `vwap[r, s] = cum_pv / cum_v` when `cum_v > 0` **and the bar at (r, s) is present**;
  otherwise NaN. (A row with no bar gets NaN even though the running sum is defined.)
- Rows with `minute_of_day < start_minute` are NaN.
- Raises `ValueError` if array shapes disagree or `day_offsets` is invalid.

### 1.2 `session_range_pct(high, low, close, minute_of_day, day_offsets, start_minute=SESSION_START_MINUTE) -> np.ndarray`

Shape `(n_days, n_sym)`. For session `d`, symbol `s`, over rows in the session with
`minute_of_day >= start_minute` where `high, low, close` are all finite:
`(max(high) - min(low)) / close_of_the_last_such_row`. NaN if there are no such rows or the
last close is <= 0.

### 1.3 `adr_pct(range_pct, n) -> np.ndarray`

Shape `(n_days, n_sym)`. **Causal**: `out[d, s] = mean(range_pct[d-n : d, s])` -- the `n`
sessions strictly before `d`; session `d` itself is never used. `out[d, s]` is NaN if
`d < n` or if any of those `n` values is NaN. Raises `ValueError` if `n < 1`.

### 1.4 `max_excursion(close, vwap, adr_day, minute_of_day, day_offsets, start_minute=SESSION_START_MINUTE, end_minute=SQUARE_OFF_MINUTE) -> np.ndarray`

Shape `(n_days, n_sym)`. For session `d`, symbol `s`: the max over rows with
`start_minute <= minute_of_day < end_minute` where close and vwap are finite and vwap > 0 of
`|close - vwap| / (vwap * adr_day[d, s])`. NaN if no such rows, or `adr_day[d, s]` is not
finite or <= 0. This is the measured distribution the k grid is drawn from (CLAUDE.md
rule 8); it is computed before any P&L.

---

## 2. `simulate.py`

### 2.1 `FadeParams` (frozen dataclass)

| field | type | default | constraint |
|---|---|---|---|
| `k` | float | required | > 0 |
| `notional` | float | required | > 0 (rupees per trade) |
| `start_minute` | int | 556 | |
| `square_off_minute` | int | 920 | > start_minute |

Invalid values raise `ValueError` in `__post_init__`.

### 2.2 `simulate_fade(open_, close, volume, vwap, adr_day, tradable, minute_of_day, day_offsets, symbols, params, *, cost_model=None, slippage=None) -> pd.DataFrame`

`tradable` is a bool `(n_rows, n_sym)` array, **distinct** from presence: a bar can be
present but not tradable. `adr_day` is `(n_days, n_sym)` (output of `adr_pct`).
`symbols` is a sequence of `n_sym` names. `cost_model` is anything with
`.charges(FillBatch) -> Charges` (e.g. `NSEIntradayEquityCosts`); `slippage` is anything
with `.bps(notional, bar_traded_value) -> ndarray` (e.g. `SqrtImpactSlippage`). `None`
means zero for each.

Each symbol is simulated independently, session by session. Positions never cross a
session boundary. At most one open position per symbol at a time.

**Bands** at row `r` of session `d`:
`upper[r] = vwap[r] * (1 + k * adr_day[d])`, `lower[r] = vwap[r] * (1 - k * adr_day[d])`.

**Entry.** While flat, row `r` produces a signal iff all hold:
- bar present at `r`, `vwap[r]` finite and > 0, `adr_day[d]` finite and > 0;
- `start_minute <= minute_of_day[r] < square_off_minute`;
- `close[r] > upper[r]` -> **short** (side = -1), or `close[r] < lower[r]` -> **long** (side = +1).

A row `x` is **fillable** iff `open_[x]` is finite and > 0 AND `volume[x]` is finite and
> 0 (a zero-volume bar is not a fill opportunity; this also guarantees a finite, positive
`bar_traded_value` at every fill).

The signal fills at `open_[r+1]` iff `r+1` is in the same session, `minute_of_day[r+1] <
square_off_minute`, row `r+1` is fillable, and `tradable[r+1]` is True.
Otherwise the signal is **dropped** (not deferred to a later bar). No fill ever happens at
the band price or at `close[r]`.

`qty = notional / entry_price` (fractional shares; float64).

**Exit.** Once filled at row `e`, rows `r >= e` of the same session are checked in order
(the entry bar's own close can trigger the exit):
1. If `minute_of_day[r] >= square_off_minute` and row `r` is fillable: exit at
   `open_[r]`, reason `"square_off"`. (Checked before rule 2 on every row.)
2. Else if bar present at `r`, `vwap[r]` finite, and (long: `close[r] >= vwap[r]`;
   short: `close[r] <= vwap[r]`): an exit is **pending**; it fills at the `open_` of the
   first later fillable row in the same session, reason `"vwap"` -- even if that row is
   at/after square-off (a pending exit takes precedence over rule 1 on that row: reason
   stays `"vwap"`). Exits do **not** require `tradable`.
3. If the session ends with the position still open (no fill found by 1 or 2): exit at
   `close[x]` where `x` is the last row of the session, `x >= e`, with `close[x]` finite
   and > 0 and `volume[x]` finite and > 0 (row `e` always qualifies), reason `"no_bar"`.

After an exit fills at row `x`, the symbol is flat and entry signals are evaluated again
starting at row `x` (row `x`'s close may signal a new entry filling at `x+1`).

**P&L per trade** (float64):
- `gross_pnl = side * qty * (exit_price - entry_price)`
- Fills: entry fill is a buy iff side = +1; exit fill is a buy iff side = -1. Fill notional
  = `qty * price` at that fill.
- `charges = cost_model.charges(FillBatch(notional=[entry_notional, exit_notional],
  is_buy=[...])).total.sum()` (`total` is a property; 0 if `cost_model is None`).
- `slippage_cost = Σ_fills notional_fill * slippage.bps(notional_fill,
  bar_traded_value_fill) / 1e4`, `bar_traded_value = close * volume` of the fill row (0 if
  `slippage is None`). For a `"no_bar"` exit the fill row is `x` and the price is
  `close[x]`.
- `costs = charges + slippage_cost`; `net_pnl = gross_pnl - costs`.
- `gross_bps = gross_pnl / notional * 1e4`; `net_bps = net_pnl / notional * 1e4`.

**Output** DataFrame, one row per completed trade, sorted by `(entry_row, symbol)`, columns
in this order:

`symbol` (str), `day_idx` (int), `side` (int, +1/-1), `signal_row` (int), `entry_row`
(int), `entry_price`, `exit_row` (int), `exit_price`, `exit_reason` (str: `vwap` |
`square_off` | `no_bar`), `qty`, `gross_pnl`, `costs`, `net_pnl`, `gross_bps`, `net_bps`,
`holding_bars` (int, `exit_row - entry_row`).

Zero trades -> an empty DataFrame with exactly these columns.

Raises `ValueError` on shape mismatches (per-bar arrays vs `(n_rows, n_sym)`, `adr_day` vs
`(n_days, n_sym)`, `len(symbols) != n_sym`, `len(minute_of_day) != n_rows`).

---

## 3. `report.py`

### 3.1 `daily_pnl(trades, n_days, column="net_pnl") -> np.ndarray`
Length `n_days` float64; element `d` = sum of `column` over trades with `day_idx == d`;
0.0 for days with no trades.

### 3.2 `summarize(trades, n_days) -> dict[str, float]`
Keys: `trades`, `trades_per_day` (trades / n_days), `win_rate` (fraction with
`net_pnl > 0`; NaN if no trades), `pct_exit_vwap`, `pct_exit_square_off`, `pct_exit_no_bar`,
`mean_gross_bps`, `mean_net_bps`, `median_holding_bars`, `total_gross_pnl`,
`total_net_pnl`, `total_costs`, `sharpe_net` (annualised Sharpe of `daily_pnl(...,
"net_pnl")` using `nifty_quant.backtest.metrics.sharpe_ratio`, 252 periods),
`sharpe_gross`, `max_concurrent` (max number of trades simultaneously open, where a trade
occupies rows `[entry_row, exit_row)`), `worst_trade_net_bps`, `worst_day_net_pnl`,
`long_trades`, `short_trades`. Means/percentages are NaN when there are no trades.

The markdown report writer and the grid/runner live in `scripts/run_vwap_adr_backtest.py`
and are verified end-to-end, not by these unit specs.

---

## 4. Invariants the tests must pin

1. VWAP resets at every `day_offsets` boundary for sessions of length 60, 105 and 375 in
   the same panel.
2. The 09:15 row never contributes to VWAP or session range.
3. NaN bars: VWAP NaN at that row; later rows' VWAP equals the VWAP computed with that row
   deleted. No fill of any kind.
4. `adr_pct` is causal: changing `range_pct[j]` can change only `out[j+1 : j+n+1]`;
   in particular `out[d]` never depends on session `d` or later.
5. Entry fills at `open_[r+1]`, never at `close[r]` or the band.
6. `present` != `tradable`: a signal whose `r+1` bar is present but not tradable is dropped;
   an exit fills even when not tradable.
7. Square-off at the 15:20 bar's open; no entry signal at rows >= 15:20; no position ever
   spans two sessions.
8. With `cost_model=None, slippage=None`: `costs == 0` and `net == gross` exactly.
9. With `NSEIntradayEquityCosts()`: per-trade `costs` equals the charges total recomputed
   independently for that trade's two fills (rel tol 1e-12).
10. At most one open position per symbol at any row.
11. Null behaviour (CLAUDE.md rule 9): on driftless Gaussian random-walk panels with a
    VWAP from `session_vwap` on that panel, ADR from `adr_pct(session_range_pct(...))`,
    all bars tradable, and zero costs, `mean_gross_bps` across >= 30 seeds is not
    systematically non-zero -- assert on the *rate*: the fraction of seeds whose trade-level
    t-stat of `gross_bps` exceeds 2.58 in absolute value is <= 0.2. Never change a seed to
    make this pass.

---

## 5. Runner: `scripts/run_vwap_adr_backtest.py`

End-to-end, verified by running it on real data (not unit-tested here). CLI via `argparse`:
`--universe nifty50`, `--years 2` (window length), `--adr-n 5 10 20`,
`--notional 100000 1000000`, `--warmup-days 60` (calendar days of extra history loaded
only so ADR exists on day one; no trade is taken before the window start), `--out
results/vwap_adr`.

Steps:
1. **Universe.** `load_universe(name)`; abort if `source` is the silent fallback
   (`all_data_symbols_fallback`) or a symbol has no `data/bars/1/<sym>` directory.
2. **Window from the holdout, never hand-typed.** `TradingCalendar.from_index_bars("NIFTY50")`
   -> `HoldoutLock(default_holdout_lock_path()).holdout_range(dates)`. Window end = last
   session date strictly before `holdout_start`; window start = first session date >=
   end minus `years` calendar years (+1 day). Abort if end >= holdout_start. Never calls
   `record_read`; print the lock's read count before and after and abort if it changed.
3. **Panel.** `load_panel(PanelSpec(freq="1", fields=OHLCV, symbols, start - warmup, end))`.
   Tradable = `nifty_quant.data.validate.tradable_mask(panel)` (its default ADV floor is a
   pre-existing repo constant, flagged in the report). Sessions not usable per
   `TradingCalendar.is_usable` are excluded: no trades on them (set tradable False for
   their rows), but they still count in `n_days` as zero-P&L days only if inside the window.
4. **Features.** `session_vwap`, `session_range_pct`, `adr_pct(n)` per N.
5. **k grid (rule 8)**, computed before any P&L: for N = the middle value of `--adr-n`
   (10 by default), `max_excursion` over in-window usable sessions; grid = quantiles
   {0.50, 0.75, 0.90, 0.95, 0.99} of its finite values, rounded to 3 dp, plus literal 3.0,
   deduplicated and sorted. The same k grid is used for every N. Record the quantiles, the
   sample size, and the fraction of (symbol, session) pairs with excursion >= 3.0.
6. **Grid run.** For each (k, N, notional): `simulate_fade` with `NSEIntradayEquityCosts()`
   and `SqrtImpactSlippage()`; drop trades whose `day_idx` is before the window start
   (there are none by construction, assert it); `summarize`.
7. **Overfitting controls.** Build the trial matrix of daily net P&L / notional (rows =
   in-window days, cols = (k, N) cells at the Rs 1L notional). Report
   `effective_n_trials`, `deflated_sharpe` of the best cell by Sharpe using the effective
   trial count, and `pbo_cscv(matrix, n_splits=16)`.
8. **Outputs** under `--out`: `REPORT.md` (caveats first: survivorship, holdout untouched
   with the exact window, no stop loss, next-bar-open fills, slippage model assumed not
   calibrated, ADV floor constant; then the k-grid derivation; then one table per notional
   with every summarize key; then per-half-year and long/short breakdown for each cell;
   per-symbol net P&L for the best and the k=3.0 cells; overfitting stats; worst trades),
   `grid.csv` (one row per cell), `trades_<k>_<N>_<notional>.parquet` for every cell,
   `k_grid.json`. No styling in the markdown.
