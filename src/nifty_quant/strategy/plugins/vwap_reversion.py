"""Intraday VWAP band reversion.

VWAP is the benchmark institutional execution desks are judged against, so large orders
get worked *toward* VWAP over the session. The hypothesis under test (NOT an established
edge) is that a price stretched far from the session VWAP on modest volume is a temporary
displacement that reverts as benchmark-driven flow arrives.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from nifty_quant.calendar import ist_label
from nifty_quant.features.core import parkinson_volatility
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.registry import register


class VwapReversionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    typical_price: Literal["close", "hlc3"] = "hlc3"
    session_start_time: str = "09:16"
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    normalize: bool = True
    vol_window: int = 30
    min_bars_since_open: int = 15
    square_off_time: str = "15:20"
    target_vol_ann: float = 0.15
    sigma_floor: float = 1e-5
    max_weight: float = 0.10
    gross: float = 1.0

    @field_validator("session_start_time")
    @classmethod
    def _validate_session_start_time(cls, v: str) -> str:
        """Reject "09:15" and earlier: the 09:15 bar is measured broken (pre-open call
        auction leaking in, close > high in every observed case) and must never
        contribute to VWAP."""
        try:
            parsed = datetime.strptime(v, "%H:%M")
        except ValueError as exc:
            raise ValueError("session_start_time must be HH:MM") from exc
        minute = parsed.hour * 60 + parsed.minute
        if minute <= 555:
            raise ValueError("session_start_time must be after 09:15")
        return v


@register
class VwapReversionStrategy(Strategy):
    """VWAP band reversion strategy with vol-normalized deviation and vol-targeted sizing."""

    name = "vwap_reversion"
    Params = VwapReversionParams

    def data_request(self) -> DataRequest:
        """Return the data request necessary for precompute."""
        p = self.params
        return DataRequest(
            freq="1",
            fields=("high", "low", "close", "volume"),
            lookback_bars=p.vol_window + 10,
        )

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        """Fully vectorized whole-panel feature computation.

        Session-anchored cumulative VWAP is computed via a global cumsum minus the
        cumsum value at each session start, so it resets exactly at day_offsets
        boundaries without assuming a fixed bar count per session. Bars before
        `session_start_time` (default 09:16, excluding the broken 09:15 bar) and
        missing bars (NaN at rest, never forward-filled) contribute zero to both the
        numerator and denominator rather than propagating NaN through the rest of
        the session.

        `dev` is a horizon-scaled z-score: the raw fractional VWAP deviation is a
        quantity accumulated since session open, so under a random walk it scales like
        `sigma_per_bar * sqrt(bars_since_open)`. Dividing by the per-bar `sigma` alone
        (with no horizon scaling) is NOT stationary within a session -- it drifts
        upward roughly `sqrt(elapsed_bars)` from open to close, so a fixed threshold
        would mean something different at 09:20 than at 15:15. Scaling `sigma` by
        `sqrt(max(bars_since_open, 1))` before dividing makes `dev` a real z-score.
        """
        p = self.params

        close = panel.field("close").astype(np.float64)
        high = panel.field("high").astype(np.float64)
        low = panel.field("low").astype(np.float64)
        volume = panel.field("volume").astype(np.float64)

        if p.typical_price == "hlc3":
            typical_price = (high + low + close) / 3.0
        else:
            typical_price = close

        valid = np.isfinite(typical_price) & np.isfinite(volume)
        minute_of_day = panel.minute_of_day()
        parsed_start = datetime.strptime(p.session_start_time, "%H:%M")
        start_minute = parsed_start.hour * 60 + parsed_start.minute
        excluded_by_time = minute_of_day < start_minute
        contributing = valid & ~excluded_by_time[:, None]

        safe_tp = np.where(valid, typical_price, 0.0)
        safe_vol = np.where(valid, volume, 0.0)
        num_contrib = np.where(contributing, safe_tp * safe_vol, 0.0)
        vol_contrib = np.where(contributing, safe_vol, 0.0)

        day_offsets = panel.day_offsets
        day_lengths = np.diff(day_offsets)

        def _session_reset_cumsum(contrib: np.ndarray) -> np.ndarray:
            cum_global = np.cumsum(contrib, axis=0)
            padded = np.vstack(
                [np.zeros((1, contrib.shape[1]), dtype=contrib.dtype), cum_global]
            )
            session_base = padded[day_offsets[:-1]]
            base_per_row = np.repeat(session_base, day_lengths, axis=0)
            return np.asarray(cum_global - base_per_row)

        cum_num = _session_reset_cumsum(num_contrib)
        cum_vol = _session_reset_cumsum(vol_contrib)

        vwap = np.full_like(cum_num, np.nan)
        valid_vwap = cum_vol > 0
        vwap[valid_vwap] = cum_num[valid_vwap] / cum_vol[valid_vwap]

        dev_raw = np.full_like(close, np.nan)
        good_vwap = valid_vwap & np.isfinite(vwap) & (vwap > 0) & np.isfinite(close)
        dev_raw[good_vwap] = (close[good_vwap] - vwap[good_vwap]) / vwap[good_vwap]

        sigma = parkinson_volatility(high, low, p.vol_window, day_offsets=day_offsets)

        # bars_since_open is needed for horizon scaling in the normalization step below.
        n_rows, n_sym = typical_price.shape
        rows = np.arange(n_rows, dtype=np.int64)
        day_start_rows = np.repeat(day_offsets[:-1], day_lengths)
        bars_once = rows - day_start_rows
        bars_since_open = np.tile(bars_once[:, None], (1, n_sym)).astype(np.float64)

        if p.normalize:
            finite_sigma = np.isfinite(sigma) & (sigma > 0)
            safe_sigma = np.where(finite_sigma, sigma, np.inf)
            horizon = np.sqrt(np.maximum(bars_since_open, 1.0))
            scaled_sigma = safe_sigma * horizon
            sigma_guard = np.maximum(scaled_sigma, p.sigma_floor)
            dev = np.full_like(dev_raw, np.nan)
            finite_dev_raw = np.isfinite(dev_raw)
            dev[finite_dev_raw] = dev_raw[finite_dev_raw] / sigma_guard[finite_dev_raw]
        else:
            dev = dev_raw

        return {
            "dev": dev,
            "sigma": sigma,
            "bars_since_open": bars_since_open,
        }

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        """Build a target portfolio for the current decision bar.

        The ``signals`` mapping contains the same keys returned by
        :meth:`precompute`, already sliced to a one-dimensional row for this
        decision time.
        """
        p = self.params
        n = len(view.symbols)

        shares = np.asarray(state.shares, dtype=np.float64)
        dev = np.asarray(signals["dev"], dtype=np.float64)
        sigma = np.asarray(signals["sigma"], dtype=np.float64)
        bars_since_open = np.asarray(signals["bars_since_open"])
        tradable = np.asarray(view.tradable, dtype=bool)

        trading_open = bool(ist_label(view.ts) < p.square_off_time)

        pos_sign = np.sign(shares)
        is_open = shares != 0
        finite_dev = np.isfinite(dev)
        exit_reversion = finite_dev & (np.abs(dev) < p.exit_threshold)
        force_exit = is_open & (not trading_open or exit_reversion)
        sign = np.where(force_exit, 0, pos_sign).astype(np.int8)

        if trading_open:
            flat = shares == 0
            eligible = (
                flat
                & tradable
                & finite_dev
                & (bars_since_open >= p.min_bars_since_open)
            )
            new_short = eligible & (dev > p.entry_threshold)
            new_long = eligible & (dev < -p.entry_threshold) & ~new_short
            sign[new_short] = -1
            sign[new_long] = 1

        # Volatility-targeted sizing. Proportionality is scale-invariant;
        # annualization would cancel during cross-sectional normalization.
        safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
        sigma_guard = np.maximum(safe_sigma, p.sigma_floor)

        weights = np.zeros(n, dtype=np.float64)
        active = sign != 0
        if np.any(active):
            raw = sign[active].astype(np.float64) / sigma_guard[active]
            clipped = np.clip(raw, -p.max_weight, p.max_weight)
            abs_sum = float(np.sum(np.abs(clipped)))
            if abs_sum > p.gross:
                clipped = clipped * (p.gross / abs_sum)
            weights[active] = clipped

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        meta = {
            "gross": float(np.sum(np.abs(weights))),
            "long_count": float(np.sum(weights > 0)),
            "short_count": float(np.sum(weights < 0)),
            "target_vol_ann": float(p.target_vol_ann),
        }
        return TargetPortfolio(weights=weights, meta=meta)
