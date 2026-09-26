"""Pre-registered kill criteria for the liquidity-provision fade (spec section 3).

Operates on the trade DataFrame produced by ``simulate.simulate_liqprov`` (one row per
completed trade, with at least ``day_idx``, ``net_pnl`` and ``net_bps``). All arithmetic is
float64.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "day_clustered_t",
    "yearly_net",
    "mean_net_bps_without_top_days",
    "kill_criteria",
]


def _per_day(trades: pd.DataFrame, column: str, how: str) -> pd.Series:
    values = pd.Series(
        np.asarray(trades[column], dtype=np.float64),
        index=np.asarray(trades["day_idx"], dtype=np.int64),
    )
    grouped = values.groupby(level=0, sort=True)
    return grouped.mean() if how == "mean" else grouped.sum()


def day_clustered_t(trades: pd.DataFrame, column: str = "net_bps") -> float:
    """t-statistic of per-day mean ``column``: ``mean / (std(ddof=1) / sqrt(n_days))``.

    Trades are first collapsed to one value per ``day_idx`` (the mean of that day's
    trades). NaN if there are fewer than 2 trade-days or the per-day std is zero.
    """
    if len(trades) == 0:
        return float("nan")
    per_day = _per_day(trades, column, "mean").to_numpy(dtype=np.float64)
    n = per_day.shape[0]
    if n < 2:
        return float("nan")
    sd = float(np.std(per_day, ddof=1))
    if not sd > 0:
        return float("nan")
    return float(np.mean(per_day) / (sd / math.sqrt(n)))


def yearly_net(trades: pd.DataFrame, session_dates: Sequence[date]) -> dict[int, float]:
    """Sum of ``net_pnl`` by calendar year of ``session_dates[day_idx]``.

    Only years that have at least one trade appear as keys.
    """
    if len(trades) == 0:
        return {}
    day_idx = np.asarray(trades["day_idx"], dtype=np.int64)
    n_dates = len(session_dates)
    if np.any(day_idx < 0) or np.any(day_idx >= n_dates):
        raise ValueError(f"trades day_idx must lie in [0, {n_dates})")
    years = np.array([session_dates[int(i)].year for i in day_idx], dtype=np.int64)
    net = np.asarray(trades["net_pnl"], dtype=np.float64)
    out: dict[int, float] = {}
    for y in np.unique(years):
        out[int(y)] = float(np.sum(net[years == y]))
    return out


def mean_net_bps_without_top_days(trades: pd.DataFrame, n_top: int = 5) -> float:
    """Mean ``net_bps`` of trades left after dropping the ``n_top`` best days.

    Days are ranked by total ``net_pnl`` (descending); ties are broken by the lower
    ``day_idx`` ranking higher, so the choice is deterministic. NaN if no trades remain.
    """
    if isinstance(n_top, bool) or int(n_top) != n_top or n_top < 0:
        raise ValueError(f"n_top must be an integer >= 0, got {n_top!r}")
    n_top = int(n_top)
    if len(trades) == 0:
        return float("nan")
    totals = _per_day(trades, "net_pnl", "sum")  # index sorted by day_idx ascending
    day_ids = totals.index.to_numpy(dtype=np.int64)
    order = np.lexsort((day_ids, -totals.to_numpy(dtype=np.float64)))
    dropped = day_ids[order[:n_top]]
    keep = ~np.isin(np.asarray(trades["day_idx"], dtype=np.int64), dropped)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.asarray(trades["net_bps"], dtype=np.float64)[keep]))


def kill_criteria(
    trades: pd.DataFrame, session_dates: Sequence[date], years: Sequence[int]
) -> dict[str, Any]:
    """Evaluate the four pre-registered kill criteria on ``net_bps`` / ``net_pnl``.

    Returns ``mean_net_bps``, ``day_t``, ``positive_years`` (years in ``years`` whose
    summed ``net_pnl`` is > 0; a year with no trades is not positive), ``n_years``
    (``len(years)``), ``mean_net_bps_ex_top5``, the booleans ``c1_mean_net_positive``
    (> 0), ``c2_day_t_gt_2`` (> 2), ``c3_positive_years_ge_4`` (>= 4),
    ``c4_ex_top5_positive`` (> 0), and ``passes`` (all four True; any NaN fails).
    """
    if len(trades) > 0:
        mean_net = float(np.mean(np.asarray(trades["net_bps"], dtype=np.float64)))
    else:
        mean_net = float("nan")
    day_t = day_clustered_t(trades, "net_bps")
    by_year = yearly_net(trades, session_dates)
    positive_years = int(sum(1 for y in years if by_year.get(int(y), 0.0) > 0))
    ex_top5 = mean_net_bps_without_top_days(trades, 5)

    c1 = bool(mean_net > 0)
    c2 = bool(day_t > 2)
    c3 = bool(positive_years >= 4)
    c4 = bool(ex_top5 > 0)
    any_nan = any(math.isnan(v) for v in (mean_net, day_t, ex_top5))
    return {
        "mean_net_bps": mean_net,
        "day_t": day_t,
        "positive_years": positive_years,
        "n_years": len(years),
        "mean_net_bps_ex_top5": ex_top5,
        "c1_mean_net_positive": c1,
        "c2_day_t_gt_2": c2,
        "c3_positive_years_ge_4": c3,
        "c4_ex_top5_positive": c4,
        "passes": bool(c1 and c2 and c3 and c4 and not any_nan),
    }
