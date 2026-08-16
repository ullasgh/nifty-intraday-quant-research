"""Portfolio state accounting and P&L tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Portfolio:
    """Single-book portfolio state.

    cum_costs accumulates only modelled charges, never slippage. cum_pnl satisfies
    cum_pnl == cash + sum(shares * prices) + cum_costs - initial_capital.
    """

    shares: np.ndarray
    cash: float
    initial_capital: float
    cum_costs: float = 0.0
    cum_pnl: float = 0.0

    def equity(self, prices: np.ndarray) -> float:
        """Return cash plus mark-to-market value, float64."""
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape != self.shares.shape or prices.ndim != 1:
            raise ValueError("prices must be 1-D and match shares")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Mask zero-share columns to 0.0 so a NaN price in an absent or flat
            # symbol contributes exactly zero, never NaN, to mark-to-market.
            safe_prices = np.where(self.shares != 0, prices, 0.0)
            mtm = float(np.dot(self.shares, safe_prices))
        return float(self.cash + mtm)

    def apply_fills(self, qty: np.ndarray, price: np.ndarray, charges: float) -> None:
        """Apply fills, updating cash, shares, cum_costs; cum_pnl is untouched."""
        qty = np.asarray(qty, dtype=np.float64)
        price = np.asarray(price, dtype=np.float64)
        if (
            qty.ndim != 1
            or price.ndim != 1
            or qty.shape != price.shape
            or qty.shape != self.shares.shape
        ):
            raise ValueError("qty and price must be 1-D and match shares")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Mask zero-fill columns the same way: a NaN fill price at a zero-qty
            # index must not make the cash delta non-finite.
            safe_price = np.where(qty != 0, price, 0.0)
            fill_value = float(np.dot(qty, safe_price))

        self.cash -= fill_value + charges
        self.shares = self.shares + qty
        self.cum_costs += charges

    def mark(self, prices: np.ndarray) -> None:
        """Recompute cum_pnl from the current snapshot and prices."""
        self.cum_pnl = float(self.equity(prices) + self.cum_costs - self.initial_capital)


@dataclass(frozen=True)
class GrossNotionalSizer:
    """Convert weights to share orders with dust and participation guards."""

    gross: float = 1.0
    max_weight: float = 0.10
    min_trade_notional: float = 25_000.0
    whole_shares: bool = True

    def to_shares(
        self, weights: np.ndarray, prices: np.ndarray, capital: float
    ) -> np.ndarray:
        """Return clean signed share counts; never NaN or inf."""
        weights = np.asarray(weights, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        if weights.shape != prices.shape:
            raise ValueError("weights and prices must have the same shape")

        valid_price = np.isfinite(prices) & (prices > 0)
        valid_weight = np.isfinite(weights)
        safe_weight = np.where(valid_weight, weights, 0.0)
        clipped_weight = np.clip(safe_weight, -self.max_weight, self.max_weight)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            target_notional = clipped_weight * self.gross * capital
        target_notional = np.where(np.isfinite(target_notional), target_notional, 0.0)
        target_notional = np.where(valid_price, target_notional, 0.0)

        safe_prices = np.where(valid_price, prices, 1.0)
        raw_shares = np.zeros_like(target_notional)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(target_notional, safe_prices, out=raw_shares, where=valid_price)

        if self.whole_shares:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                abs_shares = np.abs(raw_shares)
                signs = np.where(
                    raw_shares > 0, 1.0, np.where(raw_shares < 0, -1.0, 0.0)
                )
                floor_abs = np.floor(abs_shares)
            floor_abs = np.where(np.isfinite(floor_abs), floor_abs, 0.0)
            raw_shares = signs * floor_abs

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            notional = np.abs(raw_shares * prices)
        notional = np.where(np.isfinite(notional), notional, 0.0)
        raw_shares = np.where(notional < self.min_trade_notional, 0.0, raw_shares)

        return np.nan_to_num(raw_shares, nan=0.0, posinf=0.0, neginf=0.0)
