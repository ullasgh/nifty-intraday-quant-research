"""Daily P&L aggregation and summary statistics for VWAP +/- k*ADR fade trades.

Operates on the trade DataFrame produced by ``simulate.simulate_fade`` (one row per
completed trade). All sums are float64.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_quant.backtest.metrics import sharpe_ratio


def daily_pnl(trades: pd.DataFrame, n_days: int, column: str = "net_pnl") -> np.ndarray:
    """Per-session sum of ``column`` over trades, length ``n_days`` float64.

    Element ``d`` is the sum over trades with ``day_idx == d``; 0.0 for days with no
    trades. Raises ``ValueError`` if ``n_days < 0`` or any ``day_idx`` is outside
    ``[0, n_days)``.
    """
    if n_days < 0:
        raise ValueError(f"n_days must be >= 0, got {n_days}")
    n_days = int(n_days)
    if len(trades) == 0:
        return np.zeros(n_days, dtype=np.float64)
    day_idx = np.asarray(trades["day_idx"], dtype=np.int64)
    if np.any(day_idx < 0) or np.any(day_idx >= n_days):
        raise ValueError(f"trades day_idx must lie in [0, {n_days})")
    values = np.asarray(trades[column], dtype=np.float64)
    out = np.zeros(n_days, dtype=np.float64)
    np.add.at(out, day_idx, values)
    return out


def _max_concurrent(entry_row: np.ndarray, exit_row: np.ndarray) -> int:
    """Max number of trades simultaneously open; a trade occupies ``[entry, exit)``."""
    keep = exit_row > entry_row  # zero-length trades occupy no row
    entry_row, exit_row = entry_row[keep], exit_row[keep]
    if entry_row.size == 0:
        return 0
    rows = np.concatenate([exit_row, entry_row])
    deltas = np.concatenate(
        [-np.ones(exit_row.size, dtype=np.int64), np.ones(entry_row.size, dtype=np.int64)]
    )
    # Sort by row, closing (-1) before opening (+1) at the same row: half-open intervals.
    order = np.lexsort((deltas, rows))
    return int(np.cumsum(deltas[order]).max())


def summarize(trades: pd.DataFrame, n_days: int) -> dict[str, float]:
    """Summary statistics for a trade list over ``n_days`` sessions.

    Means/percentages (and ``worst_trade_net_bps``, ``median_holding_bars``) are NaN when
    there are no trades. ``sharpe_net`` / ``sharpe_gross`` are annualised (252 periods)
    Sharpe ratios of the per-session P&L series from :func:`daily_pnl`, including zero
    days. ``trades_per_day`` and ``worst_day_net_pnl`` are NaN when ``n_days == 0``.
    """
    n = len(trades)
    daily_net = daily_pnl(trades, n_days, "net_pnl")
    daily_gross = daily_pnl(trades, n_days, "gross_pnl")
    nan = float("nan")

    if n > 0:
        net_pnl = np.asarray(trades["net_pnl"], dtype=np.float64)
        gross_pnl = np.asarray(trades["gross_pnl"], dtype=np.float64)
        costs = np.asarray(trades["costs"], dtype=np.float64)
        net_bps = np.asarray(trades["net_bps"], dtype=np.float64)
        gross_bps = np.asarray(trades["gross_bps"], dtype=np.float64)
        reason = np.asarray(trades["exit_reason"]).astype(str)
        side = np.asarray(trades["side"], dtype=np.int64)
        entry_row = np.asarray(trades["entry_row"], dtype=np.int64)
        exit_row = np.asarray(trades["exit_row"], dtype=np.int64)
        holding = np.asarray(trades["holding_bars"], dtype=np.float64)

        win_rate = float(np.mean(net_pnl > 0))
        pct_vwap = float(np.mean(reason == "vwap"))
        pct_square_off = float(np.mean(reason == "square_off"))
        pct_no_bar = float(np.mean(reason == "no_bar"))
        mean_gross_bps = float(np.mean(gross_bps))
        mean_net_bps = float(np.mean(net_bps))
        median_holding = float(np.median(holding))
        total_gross = float(np.sum(gross_pnl))
        total_net = float(np.sum(net_pnl))
        total_costs = float(np.sum(costs))
        max_conc = float(_max_concurrent(entry_row, exit_row))
        worst_trade = float(np.min(net_bps))
        long_trades = float(np.sum(side == 1))
        short_trades = float(np.sum(side == -1))
    else:
        win_rate = pct_vwap = pct_square_off = pct_no_bar = nan
        mean_gross_bps = mean_net_bps = median_holding = worst_trade = nan
        total_gross = total_net = total_costs = 0.0
        max_conc = long_trades = short_trades = 0.0

    return {
        "trades": float(n),
        "trades_per_day": float(n) / n_days if n_days > 0 else nan,
        "win_rate": win_rate,
        "pct_exit_vwap": pct_vwap,
        "pct_exit_square_off": pct_square_off,
        "pct_exit_no_bar": pct_no_bar,
        "mean_gross_bps": mean_gross_bps,
        "mean_net_bps": mean_net_bps,
        "median_holding_bars": median_holding,
        "total_gross_pnl": total_gross,
        "total_net_pnl": total_net,
        "total_costs": total_costs,
        "sharpe_net": float(sharpe_ratio(daily_net, periods_per_year=252)),
        "sharpe_gross": float(sharpe_ratio(daily_gross, periods_per_year=252)),
        "max_concurrent": max_conc,
        "worst_trade_net_bps": worst_trade,
        "worst_day_net_pnl": float(np.min(daily_net)) if n_days > 0 else nan,
        "long_trades": long_trades,
        "short_trades": short_trades,
    }
