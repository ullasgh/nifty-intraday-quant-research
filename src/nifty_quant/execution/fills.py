"""Order fill simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SlippageComponents:
    """The two components a blended slippage `bps()` number is built from -- kept
    separate so a TCA record can recover spread vs. impact rather than only their
    sum (spec `tca_record.md`, "Interface change in fills.py")."""

    spread_bps: np.ndarray
    impact_bps: np.ndarray


class SlippageModel(Protocol):
    def bps(self, notional: np.ndarray, bar_traded_value: np.ndarray) -> np.ndarray: ...

    def components(
        self, notional: np.ndarray, bar_traded_value: np.ndarray
    ) -> SlippageComponents: ...


@dataclass(frozen=True)
class SqrtImpactSlippage:
    """`half_spread_bps=1.5` and `impact_coef=10.0` are an ASSUMED baseline, not a
    measured one: a prior calibration attempt regressing this data returned the
    assumed constants back (intercept 1.5044 vs 1.5, slope 9.9828 vs 10.0), so OHLCV
    cannot recover them. Rule 8 is not satisfied here -- do not present these as
    calibrated."""

    half_spread_bps: float = 1.5
    impact_coef: float = 10.0

    def components(
        self, notional: np.ndarray, bar_traded_value: np.ndarray
    ) -> SlippageComponents:
        notional = np.asarray(notional, dtype=np.float64)
        bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
        notional, bar_traded_value = np.broadcast_arrays(notional, bar_traded_value)

        valid = (
            np.isfinite(notional)
            & (notional >= 0)
            & np.isfinite(bar_traded_value)
            & (bar_traded_value > 0)
        )
        safe_btv = np.where(valid, bar_traded_value, 1.0)
        ratio = np.full_like(notional, 0.0, dtype=np.float64)
        impact = np.full_like(notional, 0.0, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(notional, safe_btv, out=ratio, where=valid)
            np.sqrt(np.maximum(ratio, 0.0), out=impact, where=valid)

        # Invalid entries get +inf so they can never fill -- put the entire penalty
        # on spread_bps (impact_bps stays a finite 0.0 there) so
        # spread_bps + impact_bps == bps() holds unconditionally, including on the
        # invalid branch.
        spread_bps = np.where(valid, self.half_spread_bps, np.inf)
        impact_bps = np.where(valid, self.impact_coef * impact, 0.0)
        return SlippageComponents(spread_bps=spread_bps, impact_bps=impact_bps)

    def bps(self, notional: np.ndarray, bar_traded_value: np.ndarray) -> np.ndarray:
        """Redefined in terms of `components()`, computed once, so the blended
        number can never drift from its own parts."""
        c = self.components(notional, bar_traded_value)
        return np.asarray(c.spread_bps + c.impact_bps, dtype=np.float64)


@dataclass(frozen=True)
class ZeroSlippage:
    def components(
        self, notional: np.ndarray, bar_traded_value: np.ndarray
    ) -> SlippageComponents:
        shape = np.broadcast_shapes(
            np.asarray(notional).shape, np.asarray(bar_traded_value).shape
        )
        zeros = np.zeros(shape, dtype=np.float64)
        return SlippageComponents(spread_bps=zeros, impact_bps=zeros.copy())

    def bps(self, notional: np.ndarray, bar_traded_value: np.ndarray) -> np.ndarray:
        c = self.components(notional, bar_traded_value)
        return np.asarray(c.spread_bps + c.impact_bps, dtype=np.float64)


@dataclass(frozen=True)
class FillResult:
    filled_qty: np.ndarray
    fill_price: np.ndarray
    rejected: np.ndarray
    unfilled_notional: np.ndarray


@dataclass(frozen=True)
class FillModel:
    slippage: SlippageModel
    max_participation: float = 0.02

    def fill(
        self,
        orders: np.ndarray,
        prices: np.ndarray,
        bar_traded_value: np.ndarray,
        tradable: np.ndarray,
    ) -> FillResult:
        """Partial-fill orders against bar capacity without carrying a shortfall."""
        orders = np.asarray(orders, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
        tradable = np.asarray(tradable, dtype=bool)

        arrays_1d = (
            orders.ndim == 1
            and prices.ndim == 1
            and bar_traded_value.ndim == 1
            and tradable.ndim == 1
        )
        if not arrays_1d:
            raise ValueError("orders, prices, bar_traded_value, tradable must be 1-D")
        if not (orders.shape == prices.shape == bar_traded_value.shape == tradable.shape):
            raise ValueError("all fill inputs must have the same length")

        shape = orders.shape
        has_order = orders != 0
        valid_orders = np.isfinite(orders)
        valid_prices = np.isfinite(prices) & (prices > 0)
        valid_btv = np.isfinite(bar_traded_value) & (bar_traded_value > 0)

        safe_btv = np.where(valid_btv, bar_traded_value, 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            safe_prices = np.where(valid_prices, prices, 0.0)
            abs_orders = np.where(valid_orders, np.abs(orders), 0.0)
            desired_notional = abs_orders * safe_prices
            max_notional = self.max_participation * safe_btv

        desired_notional = np.where(np.isfinite(desired_notional), desired_notional, 0.0)
        desired_notional = np.maximum(desired_notional, 0.0)
        max_notional = np.where(np.isfinite(max_notional), max_notional, 0.0)

        eligible = has_order & tradable & valid_orders & valid_prices & valid_btv
        capped_notional = np.where(eligible, np.minimum(desired_notional, max_notional), 0.0)

        slip_notional = np.where(eligible, capped_notional, 0.0)
        slip_btv = np.where(eligible, bar_traded_value, 1.0)
        raw_slip = self.slippage.bps(slip_notional, slip_btv)
        finite_slip = np.isfinite(raw_slip)
        eligible = eligible & finite_slip
        capped_notional = np.where(eligible, capped_notional, 0.0)
        slip_bps = np.where(eligible, raw_slip, 0.0)

        buy_mask = orders > 0
        sell_mask = orders < 0
        price_mult = np.ones(shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            slip_factor = slip_bps / 10_000.0
            price_mult = np.where(buy_mask, 1.0 + slip_factor, price_mult)
            price_mult = np.where(sell_mask, 1.0 - slip_factor, price_mult)
            raw_fill_price = prices * price_mult

        valid_fill_price = valid_prices & np.isfinite(raw_fill_price) & (raw_fill_price > 0)
        fill_price = np.where(valid_fill_price, raw_fill_price, 0.0)

        fill_fraction = np.zeros(shape, dtype=np.float64)
        frac_valid = eligible & (desired_notional > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(capped_notional, desired_notional, out=fill_fraction, where=frac_valid)
        fill_fraction = np.clip(fill_fraction, 0.0, 1.0)
        fill_fraction = np.where(np.isfinite(fill_fraction), fill_fraction, 0.0)

        filled_qty = orders * fill_fraction
        filled_qty = np.where(eligible & (fill_price > 0), filled_qty, 0.0)
        filled_qty = np.where(np.isfinite(filled_qty), filled_qty, 0.0)

        unfilled_notional = np.where(has_order, desired_notional - capped_notional, 0.0)
        unfilled_notional = np.maximum(unfilled_notional, 0.0)

        rejected = has_order & ~eligible

        return FillResult(
            filled_qty=filled_qty,
            fill_price=fill_price,
            rejected=rejected,
            unfilled_notional=unfilled_notional,
        )
