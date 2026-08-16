"""Bars are left-labelled: the bar labelled T covers [T, T+60s). Decision time IS fill
time: for decision_time="11:00" the cursor sits on the bar labelled 10:59 (which closed
at 11:00:00) and the order fills at the open of the bar labelled 11:00.

Per-row order of operations, for each row t in the session:
  1. Mark open positions to market at close[t-1] (already known).
  2. Execute orders emitted at the previous decision: fill at open[t] plus slippage.
     Reject if the symbol is not tradable at t; count rejections.
  3. Risk clock (only if DataRequest.needs_intrabar_risk): evaluate stops against
     low[t]/high[t]. Fill at the conservative price -- for a long stop,
     min(stop_price, open[t]) -- because the intrabar path is unknown.
  4. If t is a decision row: build the view with cursor t-1, call strategy.on_decision,
     diff targets against current holdings -> orders queued for open[t+1].
  5. If t is at or past square_off_time (default "15:20"): force targets to zero.
  6. At session end assert positions == 0. If not, force-liquidate at the final close,
     set forced_eod_liquidation, and charge delivery costs (STT 0.1% both sides).

Mandatory one-bar execution lag: a weight emitted for bar t is never filled before
t+1. This is a structural guarantee -- even a strategy that peeks at bar t's close
cannot convert it into P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import nifty_quant.guards as guards
from nifty_quant.backtest.portfolio import GrossNotionalSizer, Portfolio
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import (
    CostModel,
    FillBatch,
    NSEDeliveryEquityCosts,
    NSEIntradayEquityCosts,
)
from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
from nifty_quant.strategy.base import ArrayMarketView, PortfolioState, Strategy


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 1e7
    square_off_time: str = "15:20"
    decision_latency_bars: int = 0
    compound: bool = False
    cost_model: CostModel = field(default_factory=NSEIntradayEquityCosts)
    fill_model: FillModel = field(
        default_factory=lambda: FillModel(slippage=SqrtImpactSlippage())
    )
    sizer: GrossNotionalSizer = field(default_factory=GrossNotionalSizer)


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: pd.DataFrame
    gross_returns: np.ndarray
    total_costs: float
    n_trades: int
    turnover: np.ndarray
    rejected_order_rate: float
    unfilled_notional_pct: float
    forced_eod_liquidation_days: int
    initial_capital: float = 1e7
    n_symbols_absent: int = 0
    absent_symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, float | int]:
        if self.equity_curve.size and self.initial_capital != 0.0:
            final_equity = float(self.equity_curve[-1])
            total_return = float(final_equity / self.initial_capital - 1.0)
        else:
            final_equity = float(self.initial_capital)
            total_return = 0.0
        return {
            "total_costs": self.total_costs,
            "n_trades": self.n_trades,
            "rejected_order_rate": self.rejected_order_rate,
            "unfilled_notional_pct": self.unfilled_notional_pct,
            "forced_eod_liquidation_days": self.forced_eod_liquidation_days,
            "final_equity": final_equity,
            "total_return": total_return,
            "mean_turnover": float(np.mean(self.turnover)) if self.turnover.size else 0.0,
        }


def _parse_hhmm(value: str) -> int:
    hour_str, minute_str = value.split(":")
    return int(hour_str) * 60 + int(minute_str)


def _compute_returns(equity: np.ndarray, initial_capital: float) -> np.ndarray:
    if equity.size == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(equity.size, dtype=np.float64)
    out[0] = equity[0] / initial_capital - 1.0 if initial_capital != 0.0 else 0.0
    for idx in range(1, equity.size):
        denom = equity[idx - 1]
        out[idx] = equity[idx] / denom - 1.0 if denom != 0.0 else 0.0
    return out


def run_backtest(
    strategy: Strategy,
    panel: Panel,
    config: BacktestConfig,
    *,
    tradable: np.ndarray | None = None,
) -> BacktestResult:
    with guards.strictness(guards.Strictness.FULL):
        req = strategy.data_request()

        open_ = np.asarray(panel.field("open"), dtype=np.float64)
        high = np.asarray(panel.field("high"), dtype=np.float64)
        low = np.asarray(panel.field("low"), dtype=np.float64)
        close = np.asarray(panel.field("close"), dtype=np.float64)
        volume = np.asarray(panel.field("volume"), dtype=np.float64)

        present_open = np.isfinite(open_) & (open_ > 0.0)
        present_close = np.isfinite(close) & (close > 0.0)

        n_rows = panel.n_rows()
        n_sym = panel.n_symbols()

        absent_mask = ~(np.any(present_open, axis=0) | np.any(present_close, axis=0))
        n_symbols_absent = int(np.sum(absent_mask))
        absent_symbols = tuple(
            str(panel.symbols[i]) for i in np.flatnonzero(absent_mask)
        )

        field_names = tuple(dict.fromkeys((*req.fields, *req.extra_series)))
        panel_arrays = {
            name: np.asarray(panel.field(name), dtype=np.float64) for name in field_names
        }

        signals = strategy.precompute(panel)

        guards.check_day_offsets(panel.day_offsets, n_rows)

        if tradable is None:
            tradable_full = np.ones((n_rows, n_sym), dtype=bool)
        else:
            if tradable.shape != (n_rows, n_sym):
                raise ValueError(
                    f"tradable shape {tradable.shape} != {(n_rows, n_sym)}"
                )
            tradable_full = tradable.astype(bool, copy=False)

        open_exec = np.where(np.isfinite(open_) & (open_ > 0.0), open_, 0.0)
        tradable_exec = tradable_full & np.isfinite(open_) & (open_ > 0.0)
        volume_safe = np.where(np.isfinite(volume) & (volume > 0.0), volume, 0.0)

        day_index = (
            np.searchsorted(panel.day_offsets, np.arange(n_rows, dtype=np.int64), side="right")
            - 1
        )
        minute_of_day = panel.minute_of_day().astype(np.int64)

        if req.decision_times is None:
            decision_rows = np.arange(1, n_rows, dtype=np.int64)
        else:
            parts = []
            for decision_time in req.decision_times:
                _parse_hhmm(decision_time)
                parts.append(panel.rows_at_time(decision_time))
            if parts:
                decision_rows = np.unique(np.concatenate(parts)).astype(np.int64, copy=False)
            else:
                decision_rows = np.empty(0, dtype=np.int64)
            decision_rows = decision_rows[decision_rows != 0]
        is_decision_row = np.zeros(n_rows, dtype=bool)
        is_decision_row[decision_rows] = True

        square_off_minute = _parse_hhmm(config.square_off_time)

        day_offsets = panel.day_offsets
        square_off_row_for_day = np.empty(len(day_offsets) - 1, dtype=np.int64)
        for d in range(len(day_offsets) - 1):
            start = int(day_offsets[d])
            end = int(day_offsets[d + 1])
            session_minutes = minute_of_day[start:end]
            pos = int(np.searchsorted(session_minutes, square_off_minute, side="left"))
            if pos < session_minutes.shape[0]:
                square_off_row_for_day[d] = start + pos
            else:
                square_off_row_for_day[d] = end - 1

        portfolio = Portfolio(
            shares=np.zeros(n_sym, dtype=np.float64),
            cash=float(config.capital),
            initial_capital=float(config.capital),
        )

        pending_orders: dict[int, np.ndarray] = {}
        active_stops: dict[int, float] = {}
        square_off_queued = False

        records: list[dict[str, object]] = []
        n_submitted = 0
        n_rejected = 0
        sum_unfilled = 0.0
        sum_desired = 0.0
        forced_eod_liquidation_days = 0

        equity_vals: list[float] = []
        gross_vals: list[float] = []
        positions_list: list[np.ndarray] = []
        turnover_list: list[float] = []
        notional_since_snapshot = 0.0

        trade_columns = (
            "ts",
            "symbol",
            "qty",
            "price",
            "notional",
            "is_buy",
            "charges",
        )

        def _record_trade(
            t: int, sym_idx: int, qty: float, price: float, charges: float
        ) -> None:
            records.append(
                {
                    "ts": int(panel.ts[t]),
                    "symbol": panel.symbols[sym_idx],
                    "qty": float(qty),
                    "price": float(price),
                    "notional": abs(float(qty) * float(price)),
                    "is_buy": bool(qty > 0),
                    "charges": float(charges),
                }
            )

        def _remove_zero_stops() -> None:
            for sym_idx in np.flatnonzero(portfolio.shares == 0):
                active_stops.pop(int(sym_idx), None)

        def _execute_model_fill(order: np.ndarray, t: int) -> float:
            nonlocal n_submitted, n_rejected, sum_unfilled, sum_desired

            order = np.asarray(order, dtype=np.float64)
            prices = open_exec[t]
            bar_traded_value = volume_safe[t] * prices
            exec_trad = tradable_exec[t]

            result = config.fill_model.fill(order, prices, bar_traded_value, exec_trad)
            filled = result.filled_qty
            fill_price_arr = result.fill_price

            batch = FillBatch(notional=np.abs(filled * fill_price_arr), is_buy=filled > 0)
            charges_arr = config.cost_model.charges(batch)
            total_charges = float(np.sum(charges_arr.total))
            portfolio.apply_fills(filled, fill_price_arr, total_charges)

            for sym_idx in np.flatnonzero(filled != 0):
                _record_trade(
                    t,
                    int(sym_idx),
                    float(filled[sym_idx]),
                    float(fill_price_arr[sym_idx]),
                    float(charges_arr.total[sym_idx]),
                )

            fill_notional = float(np.sum(np.abs(filled) * fill_price_arr))
            has_order = order != 0
            n_submitted += int(np.sum(has_order))
            n_rejected += int(np.sum(result.rejected))
            sum_unfilled += float(np.sum(result.unfilled_notional))
            sum_desired += float(
                np.sum(np.where(has_order, np.abs(order) * prices, 0.0))
            )

            _remove_zero_stops()
            return fill_notional

        def _execute_direct_fill(
            t: int, order: np.ndarray, price_arr: np.ndarray, cost_model: CostModel
        ) -> float:
            nonlocal n_submitted, sum_desired

            order = np.asarray(order, dtype=np.float64)
            price_arr = np.asarray(price_arr, dtype=np.float64)

            # Mask zero-order columns to 0.0 so NaN prices in absent symbols cannot
            # produce NaN notional in FillBatch or fills, even with 0 * NaN.
            price_arr = np.where(order != 0, price_arr, 0.0)

            batch = FillBatch(notional=np.abs(order * price_arr), is_buy=order > 0)
            charges_arr = cost_model.charges(batch)
            total_charges = float(np.sum(charges_arr.total))
            portfolio.apply_fills(order, price_arr, total_charges)

            for sym_idx in np.flatnonzero(order != 0):
                _record_trade(
                    t,
                    int(sym_idx),
                    float(order[sym_idx]),
                    float(price_arr[sym_idx]),
                    float(charges_arr.total[sym_idx]),
                )

            fill_notional = float(np.sum(np.abs(order) * price_arr))
            has_order = order != 0
            n_submitted += int(np.sum(has_order))
            sum_desired += float(
                np.sum(np.where(has_order, np.abs(order) * price_arr, 0.0))
            )

            _remove_zero_stops()
            return fill_notional

        for t in range(n_rows):
            day_idx = int(day_index[t])
            is_first_row = t == panel.day_offsets[day_idx]

            if is_first_row:
                active_stops.clear()
                square_off_queued = False
                strategy.on_session_start(panel.dates[day_idx])

            if t > 0:
                prev_close = close[t - 1]
                portfolio.mark(prev_close)
                guards.check_accounting(
                    cash=portfolio.cash,
                    positions_value=float(
                        np.dot(
                            portfolio.shares,
                            np.where(portfolio.shares != 0, prev_close, 0.0),
                        )
                    ),
                    costs=portfolio.cum_costs,
                    initial_capital=config.capital,
                    pnl=portfolio.cum_pnl,
                )

            pending = pending_orders.pop(t, None)
            if pending is not None:
                notional_since_snapshot += _execute_model_fill(pending, t)

            if req.needs_intrabar_risk:
                for sym_idx in list(active_stops):
                    if portfolio.shares[sym_idx] == 0:
                        continue

                    stop_price = active_stops[sym_idx]
                    if not np.isfinite(stop_price):
                        continue

                    position = portfolio.shares[sym_idx]
                    triggered = False
                    trigger_price = 0.0

                    if position > 0:
                        low_val = low[t, sym_idx]
                        if np.isfinite(low_val) and low_val <= stop_price:
                            triggered = True
                            open_val = open_[t, sym_idx]
                            if np.isfinite(open_val) and open_val > 0:
                                trigger_price = min(stop_price, open_val)
                            else:
                                trigger_price = stop_price
                    elif position < 0:
                        high_val = high[t, sym_idx]
                        if np.isfinite(high_val) and high_val >= stop_price:
                            triggered = True
                            open_val = open_[t, sym_idx]
                            if np.isfinite(open_val) and open_val > 0:
                                trigger_price = max(stop_price, open_val)
                            else:
                                trigger_price = stop_price

                    if not triggered:
                        continue

                    stop_order = np.zeros(n_sym, dtype=np.float64)
                    stop_order[sym_idx] = -position
                    stop_price_arr = np.zeros(n_sym, dtype=np.float64)
                    stop_price_arr[sym_idx] = trigger_price
                    notional_since_snapshot += _execute_direct_fill(
                        t, stop_order, stop_price_arr, config.cost_model
                    )

            if is_decision_row[t]:
                portfolio.mark(close[t])
                equity_vals.append(portfolio.equity(close[t]))
                gross_vals.append(config.capital + portfolio.cum_pnl)
                positions_list.append(portfolio.shares.copy())

                denom_equity = (
                    equity_vals[-2] if len(equity_vals) > 1 else config.capital
                )
                if not np.isfinite(denom_equity) or denom_equity <= 0.0:
                    denom_equity = config.capital
                turnover_val = (
                    notional_since_snapshot / denom_equity
                    if denom_equity > 0.0
                    else 0.0
                )
                turnover_list.append(turnover_val)
                notional_since_snapshot = 0.0

                cursor = t - 1
                view = ArrayMarketView(
                    panel_arrays=panel_arrays,
                    cursor=cursor,
                    symbols=panel.symbols,
                    ts_array=panel.ts,
                    tradable=tradable_exec[cursor],
                    session_date=panel.dates[day_idx],
                    minute_of_day=minute_of_day,
                    day_offsets=panel.day_offsets,
                )
                state = PortfolioState(
                    shares=portfolio.shares.copy(),
                    cash=portfolio.cash,
                    equity=portfolio.equity(close[cursor]),
                    ts=int(panel.ts[cursor]),
                )

                # Slice at cursor (t - 1, the last fully closed bar the view is built on),
                # not t: row t is the bar the strategy is not allowed to see yet because
                # the fill happens at open[t+1]. Slicing at t would reintroduce lookahead.
                signals_row = {
                    key: arr[cursor] for key, arr in signals.items()
                }
                for key, arr in signals_row.items():
                    guards.check(
                        arr.shape == (n_sym,),
                        (
                            f"Signal {key!r} after slicing at cursor {cursor} has "
                            f"shape {arr.shape}; expected 1-D shape ({n_sym},) "
                            "aligned with view.symbols."
                        ),
                    )

                target = strategy.on_decision(view, signals_row, state)
                if target is not None:
                    target.validate(n_sym)
                    mark_prices = close[cursor]
                    capital_now = (
                        portfolio.equity(mark_prices)
                        if config.compound
                        else config.capital
                    )
                    target_weights = np.asarray(
                        target.weights, dtype=np.float64
                    )
                    orig_gross = float(np.sum(np.abs(target_weights)))
                    effective_mask = present_close[cursor]
                    masked_weights = np.where(effective_mask, target_weights, 0.0)
                    masked_gross = float(np.sum(np.abs(masked_weights)))
                    if orig_gross > 0.0 and 0.0 < masked_gross < orig_gross:
                        target_weights = masked_weights * (orig_gross / masked_gross)
                    else:
                        target_weights = masked_weights
                    target_shares = config.sizer.to_shares(
                        target_weights, mark_prices, capital_now
                    )
                    order = target_shares - portfolio.shares
                    fill_row = t + 1 + int(config.decision_latency_bars)
                    if fill_row < n_rows:
                        pending_orders[fill_row] = order

                    for symbol, sym_idx in panel.sym_ix.items():
                        stop_key = f"stop:{symbol}"
                        if stop_key in target.meta:
                            active_stops[sym_idx] = float(target.meta[stop_key])

            is_last_row = t == panel.day_offsets[day_idx + 1] - 1
            if t >= square_off_row_for_day[day_idx] and not square_off_queued:
                if np.any(portfolio.shares != 0):
                    if square_off_row_for_day[day_idx] == panel.day_offsets[day_idx + 1] - 1:
                        eod_order = -portfolio.shares.copy()
                        notional_since_snapshot += _execute_direct_fill(
                            t,
                            eod_order,
                            close[t],
                            config.cost_model,
                        )
                        square_off_queued = True
                    else:
                        fill_row = t + 1
                        if fill_row < n_rows and not is_last_row:
                            pending_orders[fill_row] = -portfolio.shares.copy()
                            square_off_queued = True

            if is_last_row:
                if not np.all(portfolio.shares == 0):
                    forced_eod_liquidation_days += 1
                    eod_order = -portfolio.shares.copy()
                    notional_since_snapshot += _execute_direct_fill(
                        t,
                        eod_order,
                        close[t],
                        NSEDeliveryEquityCosts(),
                    )
                assert np.all(portfolio.shares == 0)
                strategy.on_session_end(panel.dates[day_idx])

        if n_rows > 0:
            portfolio.mark(close[n_rows - 1])
            equity_vals.append(portfolio.equity(close[n_rows - 1]))
            gross_vals.append(config.capital + portfolio.cum_pnl)
            positions_list.append(portfolio.shares.copy())

            denom_equity = (
                equity_vals[-2] if len(equity_vals) > 1 else config.capital
            )
            if not np.isfinite(denom_equity) or denom_equity <= 0.0:
                denom_equity = config.capital
            turnover_val = (
                notional_since_snapshot / denom_equity
                if denom_equity > 0.0
                else 0.0
            )
            turnover_list.append(turnover_val)
            notional_since_snapshot = 0.0

        equity_curve_arr = np.asarray(equity_vals, dtype=np.float64)
        gross_curve_arr = np.asarray(gross_vals, dtype=np.float64)
        turnover_arr = np.asarray(turnover_list, dtype=np.float64)
        positions_arr = (
            np.asarray(positions_list, dtype=np.float64)
            if positions_list
            else np.empty((0, n_sym), dtype=np.float64)
        )

        returns_arr = _compute_returns(equity_curve_arr, float(config.capital))
        gross_returns_arr = _compute_returns(gross_curve_arr, float(config.capital))

        trades = pd.DataFrame(records, columns=trade_columns)
        trades = trades.astype(
            {
                "ts": np.int64,
                "symbol": object,
                "qty": np.float64,
                "price": np.float64,
                "notional": np.float64,
                "is_buy": bool,
                "charges": np.float64,
            }
        )

        return BacktestResult(
            equity_curve=equity_curve_arr,
            returns=returns_arr,
            positions=positions_arr,
            trades=trades,
            gross_returns=gross_returns_arr,
            total_costs=float(portfolio.cum_costs),
            n_trades=len(trades),
            turnover=turnover_arr,
            rejected_order_rate=(n_rejected / n_submitted if n_submitted else 0.0),
            unfilled_notional_pct=(sum_unfilled / sum_desired if sum_desired > 0 else 0.0),
            forced_eod_liquidation_days=forced_eod_liquidation_days,
            initial_capital=float(config.capital),
            n_symbols_absent=n_symbols_absent,
            absent_symbols=absent_symbols,
        )
