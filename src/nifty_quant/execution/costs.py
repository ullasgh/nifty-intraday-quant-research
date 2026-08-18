"""Transaction cost models and slippage estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FillBatch:
    """Vectorized batch of executed fills. All arrays must be the same length."""

    notional: np.ndarray  # float64, >= 0, absolute value of price*qty per fill
    is_buy: np.ndarray  # bool
    n_orders: np.ndarray | None = None  # orders represented by each fill row

    def __post_init__(self) -> None:
        notional = np.asarray(self.notional, dtype=np.float64)
        is_buy = np.asarray(self.is_buy, dtype=bool)

        if notional.ndim != 1 or is_buy.ndim != 1:
            raise ValueError("notional and is_buy must be 1-D arrays")
        if notional.shape != is_buy.shape:
            raise ValueError("notional and is_buy must have the same length")
        if np.any(notional < 0):
            raise ValueError("notional must be non-negative")
        if not np.all(np.isfinite(notional)):
            raise ValueError("notional must be finite")

        if self.n_orders is None:
            n_orders = np.ones_like(notional, dtype=np.float64)
        else:
            n_orders = np.asarray(self.n_orders, dtype=np.float64)
            if n_orders.ndim != 1:
                raise ValueError("n_orders must be a 1-D array")
            if n_orders.shape != notional.shape:
                raise ValueError("n_orders must have the same length as notional")
            if np.any(n_orders <= 0):
                raise ValueError("n_orders must be strictly positive")
            if not np.all(np.isfinite(n_orders)):
                raise ValueError("n_orders must be finite")

        object.__setattr__(self, "notional", notional)
        object.__setattr__(self, "is_buy", is_buy)
        object.__setattr__(self, "n_orders", n_orders)


@dataclass(frozen=True)
class ChargesTotal:
    """Scalar (summed) version of Charges — same fields, python floats."""

    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    ipft: float
    stamp_duty: float
    gst: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.ipft
            + self.stamp_duty
            + self.gst
        )


@dataclass(frozen=True)
class Charges:
    """Vectorized per-fill charge breakdown.

    All arrays float64, same length as the input FillBatch.
    """

    brokerage: np.ndarray
    stt: np.ndarray
    exchange_txn: np.ndarray
    sebi: np.ndarray
    ipft: np.ndarray
    stamp_duty: np.ndarray
    gst: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Elementwise sum of all seven components."""
        return np.asarray(
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.ipft
            + self.stamp_duty
            + self.gst,
            dtype=np.float64,
        )

    def sum(self) -> ChargesTotal:
        """Sum every component (and total) across the batch into scalar floats."""
        return ChargesTotal(
            brokerage=float(np.sum(self.brokerage)),
            stt=float(np.sum(self.stt)),
            exchange_txn=float(np.sum(self.exchange_txn)),
            sebi=float(np.sum(self.sebi)),
            ipft=float(np.sum(self.ipft)),
            stamp_duty=float(np.sum(self.stamp_duty)),
            gst=float(np.sum(self.gst)),
        )

    def as_bps_of(self, notional: np.ndarray) -> np.ndarray:
        """Elementwise total / notional * 1e4, safe against divide-by-zero."""
        notional_arr = np.asarray(notional, dtype=np.float64)
        out = np.zeros_like(notional_arr, dtype=np.float64)
        np.divide(self.total, notional_arr, out=out, where=notional_arr != 0.0)
        return out * 10_000.0


class CostModel(Protocol):
    def charges(self, fills: FillBatch) -> Charges: ...


@dataclass(frozen=True)
class NSEIntradayEquityCosts:
    """NSE intraday (MIS) equity cost model. Implements CostModel structurally."""

    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003
    stt_sell_pct: float = 0.00025
    exch_txn_pct: float = 0.0000297
    sebi_pct: float = 0.000001
    ipft_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18

    def charges(self, fills: FillBatch) -> Charges:
        """Vectorized charges.

        Brokerage = min(brokerage_flat, brokerage_pct * per_order_notional)
        per order, multiplied by n_orders for the fill row.
        """
        n_orders = fills.n_orders
        if n_orders is None:
            raise ValueError("FillBatch.n_orders is None; this should not happen after validation")

        per_order_notional = fills.notional / n_orders
        per_order_brokerage = np.minimum(
            self.brokerage_flat, self.brokerage_pct * per_order_notional
        )
        brokerage = per_order_brokerage * n_orders

        exchange_txn = self.exch_txn_pct * fills.notional
        sebi = self.sebi_pct * fills.notional
        stt = np.where(fills.is_buy, 0.0, self.stt_sell_pct * fills.notional)
        stamp_duty = np.where(fills.is_buy, self.stamp_buy_pct * fills.notional, 0.0)
        ipft = self.ipft_pct * fills.notional
        gst = self.gst_pct * (brokerage + exchange_txn + sebi)

        return Charges(
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            sebi=sebi,
            ipft=ipft,
            stamp_duty=stamp_duty,
            gst=gst,
        )

    def round_trip_bps(self, notional_per_leg: float) -> float:
        """Round-trip cost in bps for one buy fill and one sell fill."""
        if notional_per_leg <= 0.0:
            raise ValueError("notional_per_leg must be positive")

        fills = FillBatch(
            notional=np.array([notional_per_leg, notional_per_leg], dtype=np.float64),
            is_buy=np.array([True, False], dtype=bool),
        )
        total_charges = self.charges(fills).sum()
        return total_charges.total / notional_per_leg * 10_000.0


@dataclass(frozen=True)
class NSEDeliveryEquityCosts:
    """Supplementary delivery cost model for force-carried overnight positions.

    STT is charged on both sides. The flat per-scrip DP charge is folded into
    ``stt`` because ``Charges`` has no dedicated DP field; for sell fill rows
    the ``stt`` component therefore contains both STT and the per-scrip DP
    charge.
    """

    stt_pct: float = 0.001
    dp_charge_per_scrip: float = 15.93

    def charges(self, fills: FillBatch) -> Charges:
        zero = np.zeros_like(fills.notional, dtype=np.float64)
        stt = self.stt_pct * fills.notional
        stt = np.where(fills.is_buy, stt, stt + self.dp_charge_per_scrip)

        return Charges(
            brokerage=zero.copy(),
            stt=stt,
            exchange_txn=zero.copy(),
            sebi=zero.copy(),
            ipft=zero.copy(),
            stamp_duty=zero.copy(),
            gst=zero.copy(),
        )


@dataclass(frozen=True)
class ZeroCost:
    """All charges zero. Used to prove costs are additive and not entangled with fills."""

    def charges(self, fills: FillBatch) -> Charges:
        zero = np.zeros_like(fills.notional, dtype=np.float64)
        return Charges(
            brokerage=zero.copy(),
            stt=zero.copy(),
            exchange_txn=zero.copy(),
            sebi=zero.copy(),
            ipft=zero.copy(),
            stamp_duty=zero.copy(),
            gst=zero.copy(),
        )


@dataclass(frozen=True)
class FixedBpsCost:
    """Sensitivity-sweep-ONLY flat bps cost model.

    Deliberately unrealistic: no cap or floor, scales linearly with notional.
    Use only for cost-sensitivity sweeps, never as a substitute for
    NSEIntradayEquityCosts.
    """

    bps: float = 0.0

    def charges(self, fills: FillBatch) -> Charges:
        brokerage = self.bps / 10_000.0 * fills.notional
        zero = np.zeros_like(fills.notional, dtype=np.float64)

        return Charges(
            brokerage=brokerage,
            stt=zero.copy(),
            exchange_txn=zero.copy(),
            sebi=zero.copy(),
            ipft=zero.copy(),
            stamp_duty=zero.copy(),
            gst=zero.copy(),
        )


def _sharpe_sign(gross_returns: np.ndarray, turnover: np.ndarray, cost_bps: float) -> float:
    """Return sign of net Sharpe ratio for a given round-trip cost bps."""
    net = gross_returns - (cost_bps / 10_000.0) * turnover
    mean = float(np.mean(net))
    std = float(np.std(net, ddof=1))

    if std == 0.0:
        if mean > 0.0:
            return 1.0
        if mean < 0.0:
            return -1.0
        return 0.0

    sharpe = mean / std
    if sharpe > 0.0:
        return 1.0
    if sharpe < 0.0:
        return -1.0
    return 0.0


def breakeven_cost_bps(
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    expansion_factor: float = 10.0,
) -> float:
    """Round-trip cost in bps that drives the Sharpe ratio of net returns to zero.

    net_returns = gross_returns - (cost_bps / 1e4) * turnover.
    Sharpe uses sample standard deviation (ddof=1) and no annualization.

    Returns
    -------
    float
        The breakeven round-trip cost, if a positive breakeven exists. If the
        strategy has no positive gross edge (``mean(gross_returns) <= 0``), returns
        ``0.0``: there is no positive cost the strategy can sustain because it is
        already losing before costs. If turnover is zero/degenerate
        (``mean(turnover) <= 0``), returns ``float('nan')``: the breakeven cost is
        undefined because turnover never imposes a meaningful cost. The function
        does not raise for either of those two cases.
    """
    gross = np.asarray(gross_returns, dtype=np.float64)
    turn = np.asarray(turnover, dtype=np.float64)

    if gross.ndim != 1 or turn.ndim != 1:
        raise ValueError("gross_returns and turnover must be 1-D arrays")
    if gross.shape != turn.shape:
        raise ValueError("gross_returns and turnover must have the same length")
    if gross.size < 2:
        raise ValueError("need at least 2 observations for sample standard deviation")
    if not np.all(np.isfinite(gross)) or not np.all(np.isfinite(turn)):
        raise ValueError("gross_returns and turnover must be finite")
    if np.any(turn < 0):
        raise ValueError("turnover must be non-negative")
    if tol <= 0 or max_iter <= 0 or expansion_factor <= 1:
        raise ValueError("tol must be positive, max_iter positive, expansion_factor > 1")

    mean_gross = float(np.mean(gross))
    mean_turn = float(np.mean(turn))

    if mean_gross <= 0:
        return 0.0
    if mean_turn <= 0:
        return float("nan")

    root_guess = mean_gross * 10_000.0 / mean_turn
    upper = max(1.0, 2.0 * root_guess)
    upper_sign = _sharpe_sign(gross, turn, upper)

    for _ in range(max_iter):
        if upper_sign <= 0.0:
            break
        # Unreachable, and algebraically so rather than merely untested: `upper` is seeded
        # at max(1.0, 2 * root_guess) and root_guess is the cost (in bps) at which mean net
        # return is exactly zero. Both branches of the max exceed root_guess -- if
        # root_guess < 0.5 then 1.0 > root_guess, otherwise 2 * root_guess > root_guess --
        # so the seed always overshoots the root and `upper_sign <= 0.0` breaks on the first
        # iteration. The expansion loop therefore never runs a second pass. Kept because it
        # is the correct behaviour if the seeding heuristic is ever changed.
        upper *= expansion_factor  # pragma: no cover
        if not np.isfinite(upper):  # pragma: no cover
            # Invariant: `upper` is seeded at `2 * root_guess`, which brackets the root
            # for any input where `mean_gross > 0` and `mean_turn > 0` (both guaranteed by
            # lines 323–326). The expansion sequence grows upper toward infinity while the
            # root is finite, so upper eventually overshoots (upper_sign becomes <= 0).
            # This guard protects against a future change to the seeding heuristic.
            raise ValueError("could not bracket a sign change in net Sharpe ratio")
        upper_sign = _sharpe_sign(gross, turn, upper)
    else:  # pragma: no cover
        # Invariant: as described above, upper_sign must become <= 0 within max_iter
        # iterations. This guard protects against a future seeding heuristic change.
        raise ValueError("could not bracket a sign change in net Sharpe ratio")

    if upper_sign == 0.0:  # pragma: no cover
        # Invariant: same as above. If upper_sign == 0, the seeded upper is the exact
        # root, which is extremely rare. This guard is kept for robustness.
        return upper

    lower = 0.0
    lower_sign = _sharpe_sign(gross, turn, lower)
    if lower_sign <= 0.0:  # pragma: no cover
        # Invariant: by line 323–324, we have mean_gross > 0. At cost=0, net returns =
        # gross returns, so mean(net) = mean_gross > 0 => lower_sign > 0. This guard
        # protects against a future change to the early-return logic.
        raise ValueError("net Sharpe at zero cost is already non-positive")

    for _ in range(max_iter):
        if upper - lower < tol:
            break

        mid = (lower + upper) / 2.0
        mid_sign = _sharpe_sign(gross, turn, mid)
        if mid_sign == 0.0:
            return mid

        if lower_sign * mid_sign > 0.0:
            lower = mid
            lower_sign = mid_sign
        else:
            upper = mid

    return (lower + upper) / 2.0
