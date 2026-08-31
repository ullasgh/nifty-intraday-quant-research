"""Phase H: execution realism and capacity modeling.

Spec: specs/execution_capacity.md with AMENDMENT 1 and AMENDMENT 2 (A2.2/A2.3).

This module computes a cost-participation ladder for a strategy and derives
capacity constraints from turnover and ADV. It is the first to directly expose
unfilled notional as a first-class metric and to derive the binding constraint
between two different risks:

    position (liquidation risk):  AUM <= cap * ADV_i / w_i
    daily trade (impact risk):    AUM <= cap * ADV_i / (w_i * turnover_frac)

The trade constraint is the position constraint divided by turnover_frac, so:
    - turnover_frac < 1  ->  trade constraint is LOOSER  ->  POSITION limit binds
    - turnover_frac > 1  ->  trade constraint is TIGHTER ->  TRADE limit binds

(AMENDMENT 1 corrected the lead's initial backwards reasoning.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class LadderCell:
    """One cell of the cost-participation ladder.

    Fields
    ------
    cost_bps : float
        Round-trip cost assumption in basis points.
    participation : float
        Max participation cap as a fraction (e.g. 0.01 for 1%).
    net_sharpe : float
        Mean / std (ddof=1) of the per-row net return series, finite.
    net_bps_per_day : float
        Mean of net per-row returns in basis points (mean(net) * 1e4).
    turnover : float
        Mean turnover across all rows (invariant per cost_bps and participation).
    unfilled_fraction : float
        Pooled unfilled notional / desired notional, in [0, 1].
        A cell missing this value is an implementation defect (spec H4).
    """

    cost_bps: float
    participation: float
    net_sharpe: float
    net_bps_per_day: float
    turnover: float
    unfilled_fraction: float


class CapacityResult(NamedTuple):
    """Result of combined_capacity: the binding constraint and its value.

    Fields
    ------
    value : float
        Capacity in rupees: min(position_capacity, trade_capacity).
    binding : str
        Which constraint bound: "position" or "trade".
    """

    value: float
    binding: str


def cost_participation_ladder(
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    orders: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    bar_traded_value: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
    *,
    day_offsets: np.ndarray | None = None,
    capital_ref: float = 1_000_000.0,
    cost_bps_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0),
    participation_grid: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10),
) -> list[LadderCell]:
    """Build a cost x participation ladder for a single-symbol strategy.

    Computes net Sharpe, net bps/day, turnover, and unfilled fraction for each
    cell in the 7x5 cost-participation grid. Grid order is cost-major: the output
    list iterates through each cost level, then through each participation level
    within that cost.

    Arithmetic (per A2.2 and test suite A)
    -----------------------------------
    - net_bps_per_day = mean(gross - cost_bps/1e4 * turnover) * 1e4
    - cell.turnover = mean(turnover)
    - net_sharpe = mean(net_series) / std(net_series, ddof=1), finite
    - unfilled_fraction = pooled sum(max(0, desired - fillable)) / sum(desired), in [0, 1]
      - desired = |orders| * prices when orders given; else turnover * capital_ref
      - fillable = participation * bar_traded_value

    Parameters
    ----------
    gross_returns : np.ndarray
        1-D float64 array of gross returns (fractions, not basis points).
    turnover : np.ndarray
        1-D float64 array of turnover fractions (same length as gross_returns).
    orders : np.ndarray, optional
        1-D float64 array of order quantities. If given with prices and
        bar_traded_value, used to compute desired notional. Otherwise, desired
        is turnover * capital_ref.
    prices : np.ndarray, optional
        1-D float64 array of prices (required if orders is given).
    bar_traded_value : np.ndarray, optional
        1-D float64 array of bar traded values (required if orders is given).
    tradable : np.ndarray, optional
        1-D bool array indicating which bars are eligible for trading.
        If None, all bars are considered tradable. Not used by the ladder
        arithmetic itself; passed through per API contract.
    day_offsets : np.ndarray, optional
        Session day boundaries (rule 5 compliance). Not used in this
        implementation; retained for API compatibility.
    capital_ref : float, default 1_000_000.0
        Reference capital for the tilt's unfilled-notional model when orders
        are not provided. The existing convention (tilt's Rs 1,000,000).
    cost_bps_grid : tuple of float, default (0.0, 2.0, ..., 20.0)
        Round-trip cost levels to sweep.
    participation_grid : tuple of float, default (0.005, 0.01, ..., 0.10)
        Max participation caps (as fractions) to sweep.

    Returns
    -------
    list[LadderCell]
        35 cells (7 costs x 5 participations) in cost-major order.
    """
    gross_returns = np.asarray(gross_returns, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)

    if gross_returns.ndim != 1 or turnover.ndim != 1:
        raise ValueError("gross_returns and turnover must be 1-D arrays")
    if gross_returns.shape != turnover.shape:
        raise ValueError("gross_returns and turnover must have the same length")
    if not np.all(np.isfinite(gross_returns)):
        raise ValueError("gross_returns must be finite")
    if not np.all(np.isfinite(turnover)):
        raise ValueError("turnover must be finite")

    # Compute unfilled fraction once (invariant across costs and participations).
    bar_traded_value_arr = None
    if orders is not None:
        orders = np.asarray(orders, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        bar_traded_value_arr = np.asarray(bar_traded_value, dtype=np.float64)

        if orders.ndim != 1 or prices.ndim != 1 or bar_traded_value_arr.ndim < 1:
            raise ValueError("orders, prices, bar_traded_value must be at least 1-D")
        if not (
            orders.shape[0]
            == prices.shape[0]
            == bar_traded_value_arr.shape[0]
            == gross_returns.shape[0]
        ):
            raise ValueError(
                "orders, prices, bar_traded_value must have the same length as gross_returns"
            )

        desired_notional = np.abs(orders) * prices
    else:
        desired_notional = turnover * capital_ref
        bar_traded_value_arr = (
            np.asarray(bar_traded_value, dtype=np.float64)
            if bar_traded_value is not None
            else None
        )

    # Compute mean turnover once (invariant per the spec).
    mean_turnover = float(np.mean(turnover))

    # Compute unfilled fractions for each participation level.
    unfilled_fractions = {}
    for part in participation_grid:
        if bar_traded_value_arr is not None:
            # Handle both 1-D and multi-D bar_traded_value: sum across symbols if 2D.
            if bar_traded_value_arr.ndim == 1:
                total_fillable_per_bar = part * bar_traded_value_arr
            else:
                # 2-D or higher: sum across the symbol dimension (axis 1+)
                total_fillable_per_bar = part * np.sum(
                    bar_traded_value_arr, axis=tuple(range(1, bar_traded_value_arr.ndim))
                )
        else:
            # No bar_traded_value: create a scalar ndarray for type consistency
            total_fillable_per_bar = np.asarray(part * 1.0, dtype=np.float64)

        unfilled = np.maximum(desired_notional - total_fillable_per_bar, 0.0)
        unfilled_frac = float(np.sum(unfilled) / np.sum(desired_notional))
        unfilled_fractions[part] = unfilled_frac

    # Build the ladder in cost-major order.
    cells = []
    for cost_bps in cost_bps_grid:
        for part in participation_grid:
            # Net return per row: gross - cost as a fraction of turnover.
            cost_frac = cost_bps / 10_000.0
            net_returns = gross_returns - cost_frac * turnover

            # Net Sharpe: mean / std with Bessel correction (ddof=1).
            mean_net = float(np.mean(net_returns))
            std_net = float(np.std(net_returns, ddof=1))
            if std_net == 0.0:
                # Avoid division by zero: if all net returns are identical,
                # Sharpe is +/- inf or 0 depending on the sign of the mean.
                if mean_net > 0.0:
                    net_sharpe = float("inf")
                elif mean_net < 0.0:
                    net_sharpe = float("-inf")
                else:
                    net_sharpe = 0.0
            else:
                net_sharpe = mean_net / std_net

            # Net bps per day: mean of net returns * 1e4.
            net_bps_per_day = mean_net * 10_000.0

            # Unfilled fraction (invariant to cost).
            unfilled_frac = unfilled_fractions[part]

            cell = LadderCell(
                cost_bps=cost_bps,
                participation=part,
                net_sharpe=net_sharpe,
                net_bps_per_day=net_bps_per_day,
                turnover=mean_turnover,
                unfilled_fraction=unfilled_frac,
            )
            cells.append(cell)

    return cells


def capacity_from_adv(
    median_adv: float,
    typical_weight: float | None = None,
    max_participation: float = 0.02,
) -> float:
    """Capacity from ADV-based position limit.

    Per A2.3: capacity = max_participation * median_adv, divided by typical_weight
    when given (suite A's formula) and per-name when not (suite B's).

    This is the "position" constraint: AUM <= cap * ADV_i / w_i.

    Parameters
    ----------
    median_adv : float
        Weighted median ADV of the portfolio.
    typical_weight : float, optional
        Typical portfolio weight of a single name. If None, returns just
        max_participation * median_adv (per-name capacity).
    max_participation : float, default 0.02
        Maximum participation cap as a fraction.

    Returns
    -------
    float
        Capacity in rupees.
    """
    if median_adv < 0.0:
        raise ValueError("median_adv must be non-negative")
    if max_participation < 0.0 or max_participation > 1.0:
        raise ValueError("max_participation must be in [0, 1]")
    if typical_weight is not None:
        if typical_weight <= 0.0:
            raise ValueError("typical_weight must be positive")
        return max_participation * median_adv / typical_weight

    return max_participation * median_adv


def capacity_from_turnover(
    median_adv: float,
    typical_weight: float | None = None,
    max_participation: float = 0.02,
    daily_turnover_fraction: float | None = None,
) -> float:
    """Capacity from turnover-based daily trade limit.

    Per A2.3 (AMENDMENT 1's formula):
        capacity_from_turnover = capacity_from_adv(...) / daily_turnover_fraction

    This is the "trade" constraint: AUM <= cap * ADV_i / (w_i * turnover_frac).

    When turnover_frac < 1, this constraint is LOOSER (higher capacity) than
    the position constraint. When turnover_frac > 1, this constraint is TIGHTER
    (lower capacity). Which constraint binds determines the actual capacity.

    Parameters
    ----------
    median_adv : float
        Weighted median ADV.
    typical_weight : float, optional
        Typical portfolio weight. Passed to capacity_from_adv.
    max_participation : float, default 0.02
        Maximum participation cap.
    daily_turnover_fraction : float, optional
        Daily turnover as a fraction (e.g., 0.11 for 11% daily turnover).
        If None, returns capacity_from_adv (the numerator is undefined).

    Returns
    -------
    float
        Capacity in rupees.
    """
    adv_capacity = capacity_from_adv(median_adv, typical_weight, max_participation)
    if daily_turnover_fraction is None or daily_turnover_fraction <= 0.0:
        raise ValueError("daily_turnover_fraction must be positive")

    return adv_capacity / daily_turnover_fraction


def combined_capacity(
    median_adv: float,
    typical_weight: float | None = None,
    max_participation: float = 0.02,
    daily_turnover_fraction: float | None = None,
) -> CapacityResult:
    """Determine the binding constraint between position and trade limits.

    Per AMENDMENT 1 point 2: reports min(position, trade) and says WHICH
    constraint produced it. A capacity figure that does not state which
    constraint bound is not actionable.

    The direction regression (per A2.5):
    - turnover_frac < 1  ->  position limit binds  (trade is looser)
    - turnover_frac > 1  ->  trade limit binds     (position is looser)

    Parameters
    ----------
    median_adv : float
        Weighted median ADV.
    typical_weight : float, optional
        Typical portfolio weight.
    max_participation : float, default 0.02
        Maximum participation cap.
    daily_turnover_fraction : float, optional
        Daily turnover fraction. If None, only the position constraint is
        computed and reported.

    Returns
    -------
    CapacityResult
        NamedTuple with fields:
        - value: the binding capacity in rupees
        - binding: "position" or "trade"
    """
    position_capacity = capacity_from_adv(median_adv, typical_weight, max_participation)

    if daily_turnover_fraction is None or daily_turnover_fraction <= 0.0:
        # No turnover constraint: position binds by default.
        return CapacityResult(value=position_capacity, binding="position")

    trade_capacity = capacity_from_turnover(
        median_adv, typical_weight, max_participation, daily_turnover_fraction
    )

    # The binding constraint is the tighter (smaller) one.
    if position_capacity <= trade_capacity:
        return CapacityResult(value=position_capacity, binding="position")
    else:
        return CapacityResult(value=trade_capacity, binding="trade")
