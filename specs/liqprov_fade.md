# Spec: stress-gated residual liquidity-provision fade (`nifty_quant.research.liqprov`)

## 0. Why this exists

The base VWAP ± k·ADR fade (`specs/vwap_adr_fade.md`, `results/vwap_adr/`) is net-negative
on 2023-08 → 2025-08. Its only profitable pockets were market-wide liquidation days
(election result, Union Budgets). Exploratory diagnostics on those trades suggested that:
- shorts lose;
- fading a market-aligned stretch loses;
- a stock-specific stretch is gross-positive;
- VIX-spike days are gross-positive.

The hypothesis under test: **a long-only fade of a stock's beta-adjusted (residual) stretch
below VWAP, taken only while the market is liquidating, earns a liquidity-provision
premium.**

That hypothesis was generated on 2023-08 → 2025-08, so that window is NOT evidence for it.
The test window is **2018-01-01 → 2023-08-13**, which was never used for this family. The
holdout (≥ 2025-08-14) is not touched.

Conventions are identical to `specs/vwap_adr_fade.md`:
- `day_offsets`, `minute_of_day` (IST minutes; 09:16 → 556, 15:20 → 920);
- present = finite `close`; NaN = no bar; never filled;
- float64 in motion; arbitrary session lengths;
- the 09:15 bar is excluded everywhere (`start_minute = 556`).

This module reuses `nifty_quant.research.vwap_adr.features`: `session_vwap`,
`session_range_pct`, `adr_pct`, `SESSION_START_MINUTE`, `SQUARE_OFF_MINUTE`. It modifies
nothing outside its own package.

Index arrays (NIFTY50, INDIAVIX) are 1-D, length `n_rows`, aligned to the same rows as the
stock panel. Per-session arrays are `(n_days,)` or `(n_days, n_sym)`.

---

## 1. `features.py`

### 1.1 `index_twap(high, low, close, minute_of_day, day_offsets, start_minute=556) -> (n_rows,)`
This is the session-anchored time-weighted average of the index, used because index bars
have no volume.
- A row **contributes** iff `minute_of_day >= start_minute` and `high, low, close` are all
  finite.
- `twap[t]` = the mean of `(h+l+c)/3` over contributing rows of `t`'s session up to and
  including `t`.
- `twap[t]` is NaN if the index bar at `t` is absent (non-finite close), if
  `minute_of_day[t] < start_minute`, or if there are no contributing rows yet.
- The mean resets at every session boundary.

### 1.2 `prior_session_beta(stock_close, index_close, minute_of_day, day_offsets, usable, lookback=20, start_minute=556) -> (n_days, n_sym)`
- **1-min log-return pairs.** For row `t` with `t−1` in the same session and
  `minute_of_day[t−1] >= start_minute`, the pair is `x_t = ln(I_t/I_{t−1})`,
  `y_t = ln(S_t/S_{t−1})`.
  - It exists only if all four closes are finite and > 0.
  - Row `t−1` must be the immediately preceding row: a gap is never bridged.
- **Per-session sums.** For each (session, symbol): `n, Σx, Σy, Σxy, Σx²` over existing
  pairs.
- **Pooled beta for session `d`.** Use the `lookback` most recent sessions `j < d` with
  `usable[j]` True. Session `d` itself is never used.
  - `beta = (nΣxy − ΣxΣy) / (nΣx² − (Σx)²)`, over the pooled sums.
  - NaN if fewer than `lookback` usable prior sessions exist.
  - NaN if the symbol has zero pairs in any of those sessions (it must have traded in each
    one).
  - NaN if the denominator is ≤ 0.
- `usable` is a bool `(n_days,)`. Raise `ValueError` if `lookback < 2`.

### 1.3 `index_dev(index_close, twap) -> (n_rows,)`
`I/TWAP − 1`. NaN where either input is non-finite or `twap <= 0`.

### 1.4 `residual_stretch(close, vwap, idx_dev, beta_day, adr_day, day_offsets) -> (n_rows, n_sym)`
- `z[t, s] = ((close/vwap − 1) − beta_day[d, s] · idx_dev[t]) / adr_day[d, s]`, where `d` is
  `t`'s session.
- NaN if any input is non-finite, `vwap <= 0`, or `adr_day <= 0`.

### 1.5 `vix_change(vix_close, minute_of_day, day_offsets, start_minute=556) -> (n_rows,)`
- `vix_close[t] / prev − 1`, where `prev` is the last finite VIX close (any minute) of
  session `d−1`.
- NaN for session 0, if `prev` is missing or ≤ 0, if the VIX bar at `t` is absent, or if
  `minute_of_day[t] < start_minute`.

### 1.6 `session_extreme(x, minute_of_day, day_offsets, how, start_minute=556, end_minute=920) -> (n_days,) or (n_days, n_sym)`
- `how` is `"max"` or `"min"`, taken over rows with `start_minute <= minute < end_minute`
  and finite `x`.
- NaN if there are no such rows.
- Accepts 1-D `(n_rows,)` or 2-D `(n_rows, n_sym)` input and returns the matching rank.
- Raise `ValueError` for any other `how`.

### 1.7 `rolling_session_quantile(per_session, usable, q, window=250, min_sessions=120) -> (n_days,)`
- **Causal.** `out[d]` = `np.quantile(vals, q)` (numpy default "linear" method).
- `vals` is the most recent `window` values `per_session[j]` with `j < d`, `usable[j]` True
  and `per_session[j]` finite.
- NaN if fewer than `min_sessions` such values exist.
- Raise `ValueError` unless `0 < q < 1`, `window >= min_sessions >= 1`.

### 1.8 `stress_gate(vix_chg, idx_dev_adr, v_day, m_day, day_offsets) -> bool (n_rows,)`
- True at row `t` of session `d` iff `vix_chg[t] >= v_day[d]` and `idx_dev_adr[t] <= m_day[d]`.
- All four values must be finite; otherwise False.
- `idx_dev_adr` = `idx_dev / index ADR` (the runner computes it; here it is just an input).
- Thresholds, set by the runner (rule 8: measured, causal):
  - `v_day = rolling_session_quantile(session_extreme(vix_chg, "max"), usable, q)`;
  - `m_day = rolling_session_quantile(session_extreme(idx_dev_adr, "min"), usable, 1 − q)`.

---

## 2. `simulate.py`

### 2.1 `LiqParams` (frozen dataclass, validated in `__post_init__`, `ValueError`)

| field | type | default | constraint |
|---|---|---|---|
| `k` | float | required | finite, > 0 |
| `notional` | float | required | finite, > 0 |
| `mode` | str | required | `"passive"` or `"market"` |
| `tick` | float | 0.05 | finite, > 0 (NSE equity tick for 2018-2023) |
| `start_minute` | int | 556 | |
| `square_off_minute` | int | 920 | > start_minute |

### 2.2 `simulate_liqprov(open_, high, low, close, volume, vwap, idx_dev, beta_day, adr_day, gate, tradable, minute_of_day, day_offsets, symbols, params, *, cost_model=None, slippage=None) -> pd.DataFrame`

**Long only.** Every trade has `side = +1`. Each symbol is independent, positions never
cross sessions, and there is at most one open position per symbol.

**Band** at row `t` of session `d`:
`band[t] = vwap[t] · (1 + beta_day[d] · idx_dev[t] − k · adr_day[d])`.
- It is defined only if `vwap[t] > 0`, `adr_day[d] > 0`, and all of `vwap[t]`,
  `idx_dev[t]`, `beta_day[d]`, `adr_day[d]` are finite; otherwise NaN.
- Note: `close[t] <= band[t]` ⇔ `residual_stretch[t] <= −k`.

**Fillable** row `x` (as in vwap_adr): `open_[x]` finite and > 0, and `volume[x]` finite
and > 0.

**Passive entry (`mode="passive"`).** A buy limit order is placed at the close of row `t−1`
at price `L = band[t−1]`. While flat, row `t` fills it iff all hold:
- `t−1` and `t` are in the same session;
- `minute_of_day[t−1] >= start_minute` and `minute_of_day[t] < square_off_minute`;
- `gate[t−1]` is True, and `band[t−1]` is finite and > 0;
- row `t` is fillable, `tradable[t]` is True, and `low[t]` is finite;
- `low[t] <= L − tick` (strict trade-through by one tick; `low[t] == L` does NOT fill).

The fill price is `min(L, open_[t])`: a gap below the limit fills at the open. It then
holds that `signal_row = t−1` and `entry_row = t`. After an exit filled at row `x`, passive
entries are evaluated again from row `x + 1` (the order is placed at `x`'s close).
Passive entry fills pay **no slippage**.

**Market entry (`mode="market"`).** While flat, row `r` signals iff all hold:
- the bar at `r` is present and `gate[r]` is True;
- `start_minute <= minute_of_day[r] < square_off_minute`;
- `band[r]` is finite and > 0, and `close[r] <= band[r]`.

It fills at `open_[r+1]` iff `r+1` is in the same session, `minute_of_day[r+1] <
square_off_minute`, row `r+1` is fillable, and `tradable[r+1]` is True. Otherwise the
signal is dropped. After an exit filled at row `x`, signals are evaluated again from row
`x`. Market entry fills pay slippage.

**Exit.** Identical to `vwap_adr` spec §2.2 rules 1-3 for a long position:
1. Square-off at the open of a fillable row with `minute >= square_off_minute`.
2. Otherwise a pending exit once `close[r] >= vwap[r]` (the bar is present and vwap
   finite). It fills at the next fillable row's open with reason `"vwap"`, taking
   precedence over square-off on that row.
3. Otherwise `"no_bar"` at `close[x]` of the last row `x >= e` of the session with finite
   close > 0 and volume > 0.

The entry row's own close can trigger rule 2. Exit fills pay slippage.

**P&L and costs.** Same formulas as `vwap_adr` §2.2:
- `qty = notional / entry_price`;
- charges via `cost_model.charges(FillBatch(...)).total` (the entry fill is a buy, the exit
  a sell);
- slippage = `notional_fill · slippage.bps(notional_fill, close·volume at the fill row) /
  1e4`, applied only to fills that pay slippage (see above);
- `None` models mean zero.

**Output.** Columns in this order, sorted by `(entry_row, symbol)`, with an empty frame
having the same columns:

`symbol, day_idx, side, signal_row, entry_row, entry_price, exit_row, exit_price,
exit_reason, qty, gross_pnl, costs, net_pnl, gross_bps, net_bps, holding_bars`

Raise `ValueError` on any shape mismatch: per-bar `(n_rows, n_sym)`; `idx_dev`, `gate`,
`minute_of_day` of length `n_rows`; `beta_day` and `adr_day` of shape `(n_days, n_sym)`;
`len(symbols) == n_sym`.

---

## 3. `evaluate.py` (kill criteria; all computed on `net_bps` unless stated)

- `day_clustered_t(trades, column="net_bps") -> float`: collapse trades to one value per
  `day_idx` (the mean over that day's trades), then `mean / (std(ddof=1) / sqrt(n_days))`.
  NaN if fewer than 2 trade-days or zero std.
- `yearly_net(trades, session_dates) -> dict[int, float]`: the sum of `net_pnl` by calendar
  year of `session_dates[day_idx]`. `session_dates` is a sequence of `datetime.date`
  indexed by `day_idx`. It has keys only for years that have trades.
- `mean_net_bps_without_top_days(trades, n_top=5) -> float`: rank days by total
  `net_pnl`, drop the `n_top` highest, and return the mean `net_bps` of the remaining
  trades. NaN if none remain.
- `kill_criteria(trades, session_dates, years) -> dict`:
  - keys `mean_net_bps`, `day_t`, `positive_years` (int), `n_years` (= `len(years)`),
    `mean_net_bps_ex_top5`;
  - booleans `c1_mean_net_positive` (> 0), `c2_day_t_gt_2` (> 2),
    `c3_positive_years_ge_4` (a year with no trades counts as not positive),
    `c4_ex_top5_positive` (> 0);
  - `passes` = all four booleans True (False if any value is NaN).

---

## 4. Invariants the tests must pin

1. `index_twap` resets per session (60/105/375-bar sessions); 09:15 excluded; no filling.
2. `prior_session_beta` equals `np.polyfit(x, y, 1)[0]` on the pooled pairs of exactly the
   prior `lookback` usable sessions. It is causal: changing any data in session `d` or
   later never changes `beta[:d+1]`. Unusable sessions are skipped, not counted. No pair
   bridges a missing row or a session boundary.
3. `rolling_session_quantile` is causal (`out[d]` independent of `per_session[d:]`) and
   equals `np.quantile` on the stated slice.
4. `vix_change` uses the previous session's last finite close only.
5. Passive fill: `low == L` gives no fill; `low == L − tick` fills; `open < L − tick` fills
   at `open`; `gate[t]` True with `gate[t−1]` False gives no fill at `t`; present but not
   tradable gives no fill.
6. Market mode matches `vwap_adr` entry timing (next-row open; dropped signals are not
   deferred).
7. No trade with `side != 1` ever. No position spans sessions. At most one open position
   per symbol.
8. Zero cost models: `costs == 0`, `net == gross`. With `NSEIntradayEquityCosts` and
   `SqrtImpactSlippage`: passive-entry trades are charged slippage on the exit fill only;
   market-entry trades on both fills (rel tol 1e-12).
9. `kill_criteria` on hand-built trade frames matches hand-computed values for every key.
10. Null (CLAUDE.md rule 9): on ≥ 30 seeds of driftless random-walk stock panels, a
    random-walk index, a random `gate`, and zero costs, the fraction of seeds with
    `|day_clustered_t(gross_bps)| > 2.58` is ≤ 0.2. Seeds are never changed to pass.

---

## 5. Scripts (verified end-to-end, not unit-tested)

### 5.1 `scripts/prereg_liqprov.py` (no P&L computed)
1. Universe `nifty50` (abort on fallback).
2. Test window = [2018-01-01, 2023-08-13]. Abort if its end is ≥ the holdout start. Never
   call `record_read`, and verify the read count is unchanged.
3. Load the panel from 2017-07-17 → 2023-08-13 with the 50 stocks plus NIFTY50 and
   INDIAVIX. `usable[d]` = `TradingCalendar.is_usable`.
4. Compute features: stock and index ADR (N=10), beta (lookback 20), z, vix_change,
   idx_dev_adr, and `v_day`/`m_day` for each q ∈ {0.80, 0.90, 0.95}.
5. **k grid:** take the per-(symbol, session) `session_extreme(z, "min")` over usable
   in-window sessions. Then `k90 = −quantile(minz, 0.10)` and `k95 = −quantile(minz,
   0.05)`, rounded to 3 dp.
6. Record for each q: the fraction of in-window usable sessions where the gate fires at
   least once, and the count of such sessions.
7. Write `results/liqprov/prereg.json` with: window, holdout, universe, the full 12-trial
   grid (q × k × mode), notional 1,000,000 (and 100,000 as a reported sensitivity), the
   measured quantiles and sample sizes, gate frequencies, and the kill criteria text plus
   the DSR bar (0.95). Add a `config_hash` = blake2s(8 bytes) over the canonical JSON of
   everything except timestamps and git sha.

### 5.2 `scripts/run_liqprov_backtest.py`
1. Refuse to run unless `prereg.json` exists and its recomputed `config_hash` matches.
   Recompute the k grid and gate thresholds and abort if they differ from the prereg.
2. Same panel and features. Tradable = `tradable_mask(panel)` on stock columns only.
   Unusable and out-of-window sessions are untradable.
3. For each of the 12 trials, at both notionals: `simulate_liqprov` with
   `NSEIntradayEquityCosts()` + `SqrtImpactSlippage()`, then `vwap_adr.report.summarize`
   and `evaluate.kill_criteria` (years 2018-2023).
4. At ₹10L: a trial matrix of daily net P&L/notional; `effective_n_trials`; the
   `deflated_sharpe` of the best cell (per-period, as in the vwap_adr runner);
   `pbo_cscv(n_splits=16)`.
5. Decomposition per trade: `beta_d · (I_open[exit_row] / I_open[entry_row] − 1)` (market
   part, in bps) vs `gross_bps − market part` (residual). Report the means per trial.
6. Always report: March-2020 P&L per trial, per-year net, worst day, worst trade,
   max_concurrent, and the 10 worst trades with IST times.
7. Write `results/liqprov/REPORT.md` (caveats first, plain markdown, no styling),
   `grid.csv`, and the trade parquets. Holdout read count unchanged.
