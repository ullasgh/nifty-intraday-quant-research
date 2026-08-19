"""Daily results as a first-class backtest output.

`BacktestResult.equity_curve` / `.returns` / `.turnover` carry one entry per DECISION
row plus one unconditional final row -- they are not daily, and annualizing them at 252
is wrong. `DailyResult` aggregates those per-row series into one entry per trading
SESSION, using the day index the engine already computes for every appended row
(`engine.py`'s `day_idx = int(day_index[t])`), never by re-deriving decision rows from
outside the engine.

Aggregation rules, which are not symmetric and must not be unified:
  - `returns` and `gross_returns` COMPOUND within a session (`aggregate_returns_by_group`).
  - `turnover` SUMS within a session -- it is notional churned, and compounding it is
    meaningless.
  - `equity` takes the LAST value within the session.
  - A session with no engine row (e.g. a checkpoint strategy whose decision time does
    not exist in a shortened session) still gets a row: zero return, zero turnover, and
    carried-forward equity -- `config.capital` if it is the first session in the panel,
    since there is nothing else to carry forward.

See `specs/daily_results.md`, including AMENDMENT 1, for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nifty_quant.backtest.metrics import aggregate_returns_by_group


@dataclass(frozen=True)
class DailyResult:
    """One row per trading session in the panel.

    `dates` is int64 epoch-seconds, UTC midnight of the session date (AMENDMENT 1
    item 2) -- a DATE KEY, not a bar timestamp; never convert it to Asia/Kolkata and
    read a trading time off it.
    """

    dates: np.ndarray  # int64 epoch-seconds, one per trading day, ascending
    equity: np.ndarray  # float64, end-of-day equity
    returns: np.ndarray  # float64, compounded within the day
    gross_returns: np.ndarray  # float64, compounded within the day
    turnover: np.ndarray  # float64, SUMMED within the day (additive, never compounded)
    n_days: int


def build_daily(
    row_day_index: np.ndarray,  # the day_idx the engine recorded per appended row
    equity: np.ndarray,
    returns: np.ndarray,
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    dates: np.ndarray,  # int64 epoch-seconds, one per SESSION in the panel
    *,
    initial_capital: float,
) -> DailyResult:
    """Build a `DailyResult` from raw per-row engine state.

    `dates` carries one entry per session in the PANEL, not per session that produced
    rows -- that is what makes a decision-less session expressible at all (spec section
    D / AMENDMENT 1 item 4). `row_day_index` must be non-decreasing (rows are appended
    by the engine in row order, and `day_idx` is non-decreasing in row order); entries
    sharing the same day must be contiguous.
    """
    dates_arr = np.asarray(dates, dtype=np.int64).reshape(-1)
    n_days = dates_arr.shape[0]

    daily_equity = np.empty(n_days, dtype=np.float64)
    daily_returns = np.zeros(n_days, dtype=np.float64)
    daily_gross_returns = np.zeros(n_days, dtype=np.float64)
    daily_turnover = np.zeros(n_days, dtype=np.float64)

    row_day_index = np.asarray(row_day_index, dtype=np.int64).reshape(-1)

    present = np.zeros(n_days, dtype=bool)

    if row_day_index.size and n_days:
        equity = np.asarray(equity, dtype=np.float64).reshape(-1)
        returns = np.asarray(returns, dtype=np.float64).reshape(-1)
        gross_returns = np.asarray(gross_returns, dtype=np.float64).reshape(-1)
        turnover = np.asarray(turnover, dtype=np.float64).reshape(-1)

        # `aggregate_returns_by_group` (called below on `returns`/`gross_returns`)
        # already raises a clear ValueError on a length mismatch against
        # `row_day_index`. `turnover` and `equity` have no such guard -- their
        # consumers (`np.add.reduceat`, fancy indexing) raise incidental,
        # uninformative exceptions (e.g. `IndexError`) instead. Found by a
        # migration agent while porting `daily_results`: validate all four
        # row-level arrays up front so every mismatched-length caller error
        # raises the same, name-and-length-bearing `ValueError`.
        for name, arr in (
            ("equity", equity),
            ("returns", returns),
            ("gross_returns", gross_returns),
            ("turnover", turnover),
        ):
            if arr.shape[0] != row_day_index.shape[0]:
                raise ValueError(
                    f"build_daily: {name} has length {arr.shape[0]}, "
                    f"row_day_index has length {row_day_index.shape[0]}"
                )

        # Contiguous per-day groups: row_day_index is non-decreasing, so a group
        # boundary is any position where the day index changes from the previous row.
        group_starts = np.r_[0, np.flatnonzero(np.diff(row_day_index) != 0) + 1]
        group_ends = np.r_[group_starts[1:], row_day_index.size]
        group_days = row_day_index[group_starts]

        compounded_returns = aggregate_returns_by_group(returns, row_day_index)
        compounded_gross = aggregate_returns_by_group(gross_returns, row_day_index)
        summed_turnover = np.add.reduceat(turnover, group_starts)
        end_of_group_equity = equity[group_ends - 1]

        daily_returns[group_days] = compounded_returns
        daily_gross_returns[group_days] = compounded_gross
        daily_turnover[group_days] = summed_turnover
        daily_equity[group_days] = end_of_group_equity
        present[group_days] = True

    prev_equity = float(initial_capital)
    for day in range(n_days):
        if present[day]:
            prev_equity = float(daily_equity[day])
        else:
            daily_equity[day] = prev_equity

    return DailyResult(
        dates=dates_arr,
        equity=daily_equity,
        returns=daily_returns,
        gross_returns=daily_gross_returns,
        turnover=daily_turnover,
        n_days=n_days,
    )
