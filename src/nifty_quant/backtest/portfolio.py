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
class SizingResult:
    """Result of capacity-aware sizing (returned only when bar_traded_value is given).

    shares: clean signed share counts, same contract as the legacy plain-array return.
    unmet_gross_pct: fraction (0.0-1.0) of the intended gross notional (sum of
        abs(target notional) before capacity capping) that could not be sized because
        every name with spare room was already at its own capacity cap. 0.0 means
        capacity was not binding (or was fully absorbed by renormalisation).
    """

    shares: np.ndarray
    unmet_gross_pct: float


@dataclass(frozen=True)
class GrossNotionalSizer:
    """Convert weights to share orders with dust and participation guards."""

    gross: float = 1.0
    max_weight: float = 0.10
    min_trade_notional: float = 25_000.0
    whole_shares: bool = True

    def _finalize_shares(
        self, notional: np.ndarray, prices: np.ndarray, valid_price: np.ndarray
    ) -> np.ndarray:
        """Convert clean notional to share counts with rounding and dust filtering."""
        safe_prices = np.where(valid_price, prices, 1.0)
        raw_shares = np.zeros_like(notional)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(notional, safe_prices, out=raw_shares, where=valid_price)

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

        result: np.ndarray = np.nan_to_num(raw_shares, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def to_shares(
        self,
        weights: np.ndarray,
        prices: np.ndarray,
        capital: float,
        *,
        bar_traded_value: np.ndarray | None = None,
        max_participation: float = 0.02,
    ) -> np.ndarray | SizingResult:
        """Return clean signed share counts; optionally capacity-aware.

        When bar_traded_value is None, behaviour is bit-identical to the original,
        capacity-unaware implementation and the return type is a plain np.ndarray.
        When bar_traded_value is given, each name's target notional is capped at
        max_participation * bar_traded_value (default matches FillModel.max_participation
        in nifty_quant.execution.fills -- this is a documented coupling, not a shared
        constant, to avoid a circular import; callers that must stay in sync should pass
        max_participation=config.fill_model.max_participation explicitly), the shortfall
        is renormalised into names with spare capacity (mirroring the engine's existing
        absent-symbol renormalisation, which lets survivors exceed their own original
        target), and the result is returned as a SizingResult exposing unmet_gross_pct.
        """
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

        if bar_traded_value is None:
            return self._finalize_shares(target_notional, prices, valid_price)

        bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
        if bar_traded_value.shape != weights.shape:
            raise ValueError("bar_traded_value must have the same shape as weights/prices")

        desired_abs = np.abs(target_notional)
        sign = np.sign(target_notional)
        valid_btv = np.isfinite(bar_traded_value) & (bar_traded_value > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            capacity = np.where(valid_btv, max_participation * bar_traded_value, 0.0)
            capacity = np.where(np.isfinite(capacity) & (capacity > 0), capacity, 0.0)

        final_abs = np.minimum(desired_abs, capacity)
        orig_gross = float(np.sum(desired_abs))

        # Iterative water-filling: redistribute the shortfall created by capped names
        # into names that still have spare room under their OWN capacity (not their own
        # original desired_abs -- survivors may exceed their original per-name target,
        # exactly like the engine's absent-symbol renormalisation). Only names with
        # desired_abs > 0 participate. At most len(weights) rounds: each round either
        # saturates at least one more name or fully closes the shortfall.
        for _ in range(len(weights)):
            achieved = float(np.sum(final_abs))
            shortfall = orig_gross - achieved
            if shortfall <= 1e-9 * max(orig_gross, 1.0):
                break
            free_mask = (desired_abs > 0) & (final_abs < capacity)
            if not np.any(free_mask):
                break
            free_share = np.where(free_mask, final_abs, 0.0)
            total_free_share = float(np.sum(free_share))
            if total_free_share <= 0.0:
                break
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                add = np.where(
                    free_mask,
                    shortfall * free_share / total_free_share,
                    0.0,
                )
            final_abs = final_abs + add
            final_abs = np.minimum(final_abs, capacity)

        achieved_gross = float(np.sum(final_abs))
        if orig_gross <= 0.0:
            unmet_gross_pct = 0.0
        else:
            unmet_gross_pct = max(0.0, (orig_gross - achieved_gross) / orig_gross)
        if not np.isfinite(unmet_gross_pct):
            unmet_gross_pct = 0.0

        capacity_adjusted_notional = sign * final_abs
        capacity_adjusted_notional = np.where(
            np.isfinite(capacity_adjusted_notional), capacity_adjusted_notional, 0.0
        )
        shares = self._finalize_shares(capacity_adjusted_notional, prices, valid_price)
        return SizingResult(shares=shares, unmet_gross_pct=float(unmet_gross_pct))
