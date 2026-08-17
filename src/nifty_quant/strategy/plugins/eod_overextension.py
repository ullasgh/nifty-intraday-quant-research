"""End-of-Day Over-Extension / overnight gap strategy.

Index and mutual funds face end-of-day execution mandates; concentrated high-volume
directional pressure into the close creates a temporary imbalance. The hypothesis under
test is whether it resolves in the overnight gap. The horizon is deliberately one where
one minute of latency does not destroy the edge, unlike ``volume_breakout``.

Positions are held **overnight** (delivery, not intraday). Running a backtest of this
strategy MUST use ``NSEDeliveryEquityCosts`` (STT 0.1% both sides, DP ~Rs 15.93/scrip)
from ``nifty_quant.execution.costs``, NOT ``NSEIntradayEquityCosts``.

The 09:15 bar is structurally broken (close > high in every observed case - a pre-open
call-auction leak). This strategy must never read or trade on it; ``precompute`` masks
that bar and the parameter validator rejects 09:15 as a configured time.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from nifty_quant.features.core import (
    cross_sectional_rank,
    parkinson_volatility,
    rolling_max,
    rolling_min,
    volume_zscore,
)
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.registry import register

_FORBIDDEN_MINUTE = 9 * 60 + 15


def _parse_hhmm(value: str) -> int:
    hour_str, minute_str = value.split(":")
    return int(hour_str) * 60 + int(minute_str)


def _anchor_row_per_day(
    target_minute: int,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
) -> np.ndarray:
    """Resolve, for each session, the first row >= ``target_minute``."""
    n_days = len(day_offsets) - 1
    anchor_row = np.empty(n_days, dtype=np.int64)
    for d in range(n_days):
        start = int(day_offsets[d])
        end = int(day_offsets[d + 1])
        session_minutes = minute_of_day[start:end]
        pos = int(np.searchsorted(session_minutes, target_minute, side="left"))
        if pos < session_minutes.shape[0]:
            idx = start + pos
        else:
            idx = end - 1
        if minute_of_day[idx] == _FORBIDDEN_MINUTE:
            idx = min(idx + 1, end - 1)
        anchor_row[d] = idx
    return anchor_row


def _ffill_true_within_day(raw: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Forward-fill True flags within each session, never across session boundaries."""
    out = np.zeros_like(raw, dtype=bool)
    n_days = len(day_offsets) - 1
    for d in range(n_days):
        start = int(day_offsets[d])
        end = int(day_offsets[d + 1])
        out[start:end] = np.maximum.accumulate(
            raw[start:end].astype(np.int8), axis=0
        ).astype(bool)
    return out


class EODOverExtensionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_open_time: str = "09:16"
    signal_time: str = "15:10"
    entry_time: str = "15:20"
    exit_time: str = "09:16"
    entry_threshold: float = 0.01
    volume_z_threshold: float = 2.0
    volume_window: int = 30
    vol_window: int = 30
    n_max: int = 10
    direction: Literal["long", "short", "both"] = "both"
    sigma_floor: float = 1e-5
    max_weight: float = 0.10
    gross: float = 1.0

    @field_validator("session_open_time", "signal_time", "entry_time", "exit_time")
    @classmethod
    def _validate_time_not_0915(cls, v: str) -> str:
        if v == "09:15":
            raise ValueError(
                "09:15 is the broken pre-open auction bar and cannot be a strategy time"
            )
        return v

    @field_validator("entry_threshold")
    @classmethod
    def _validate_entry_threshold_ge_zero(cls, v: float) -> float:
        if v < 0:
            raise ValueError("entry_threshold must be >= 0")
        return v

    @field_validator("volume_window", "vol_window", "n_max")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("value must be >= 1")
        return v

    @field_validator("sigma_floor", "max_weight", "gross")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value must be > 0")
        return v


@register
class EODOverExtensionStrategy(Strategy):
    name = "eod_overextension"
    Params = EODOverExtensionParams

    def data_request(self) -> DataRequest:
        p = self.params
        lookback = max(p.volume_window, p.vol_window)
        return DataRequest(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            lookback_bars=lookback + 5,
            lookback_days=5,
        )

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        p = self.params

        close = panel.field("close").astype(np.float64, copy=False)
        high = panel.field("high").astype(np.float64, copy=False)
        low = panel.field("low").astype(np.float64, copy=False)
        volume = panel.field("volume").astype(np.float64, copy=False)
        minute_of_day = panel.minute_of_day().astype(np.int64, copy=False)
        day_offsets = panel.day_offsets

        n_rows, n_sym = close.shape

        valid_bar = minute_of_day != _FORBIDDEN_MINUTE
        close_m = np.where(valid_bar[:, None], close, np.nan)
        high_m = np.where(valid_bar[:, None], high, np.nan)
        low_m = np.where(valid_bar[:, None], low, np.nan)
        volume_m = np.where(valid_bar[:, None], volume, np.nan)

        row_idx = np.arange(n_rows)
        day_index = np.searchsorted(day_offsets, row_idx, side="right") - 1

        session_open_row = _anchor_row_per_day(
            _parse_hhmm(p.session_open_time), minute_of_day, day_offsets
        )
        signal_row = _anchor_row_per_day(
            _parse_hhmm(p.signal_time), minute_of_day, day_offsets
        )
        entry_row = _anchor_row_per_day(
            _parse_hhmm(p.entry_time), minute_of_day, day_offsets
        )
        exit_row = _anchor_row_per_day(
            _parse_hhmm(p.exit_time), minute_of_day, day_offsets
        )

        session_open_price_row = close_m[session_open_row[day_index]]
        row_after_open = row_idx >= session_open_row[day_index]
        safe_open = np.where(
            np.isfinite(session_open_price_row) & (session_open_price_row > 0),
            session_open_price_row,
            np.nan,
        )
        session_return = np.where(
            row_after_open[:, None] & np.isfinite(safe_open) & np.isfinite(close_m),
            close_m / safe_open - 1.0,
            np.nan,
        )

        session_high = rolling_max(
            high_m, window=max(n_rows, 1), min_count=1, day_offsets=day_offsets
        )
        session_low = rolling_min(
            low_m, window=max(n_rows, 1), min_count=1, day_offsets=day_offsets
        )
        rng = session_high - session_low
        safe_rng = np.where(np.isfinite(rng) & (rng > 0), rng, np.nan)
        position_in_range = (close_m - session_low) / safe_rng

        vol_z = volume_zscore(
            volume_m,
            minute_of_day,
            p.volume_window,
            deseasonalize=True,
            day_offsets=day_offsets,
        )
        sigma = parkinson_volatility(
            high_m,
            low_m,
            p.vol_window,
            day_offsets=day_offsets,
        )

        passes_return = np.isfinite(session_return) & (
            np.abs(session_return) >= p.entry_threshold
        )
        passes_vol = np.isfinite(vol_z) & (vol_z >= p.volume_z_threshold)
        eligible = passes_return & passes_vol
        score_for_rank = np.where(eligible, -np.abs(session_return), np.inf)
        rank_asc = cross_sectional_rank(score_for_rank, pct=False, min_names=1)
        top_n_mask = eligible & np.isfinite(rank_asc) & (rank_asc <= p.n_max)
        raw_long_level = top_n_mask & (session_return > 0)
        raw_short_level = top_n_mask & (session_return < 0)

        if p.direction == "long":
            raw_short_level = np.zeros_like(raw_short_level, dtype=bool)
        elif p.direction == "short":
            raw_long_level = np.zeros_like(raw_long_level, dtype=bool)
        elif p.direction == "both":
            pass
        else:
            raise ValueError(f"Unsupported direction: {p.direction!r}")

        is_signal_row = row_idx == signal_row[day_index]
        raw_long_signal_row = raw_long_level & is_signal_row[:, None]
        raw_short_signal_row = raw_short_level & is_signal_row[:, None]

        long_ffill = _ffill_true_within_day(raw_long_signal_row, day_offsets)
        short_ffill = _ffill_true_within_day(raw_short_signal_row, day_offsets)

        entry_cursor_row = np.clip(entry_row - 1, 0, n_rows - 1)
        entry_long = np.zeros((n_rows, n_sym), dtype=bool)
        entry_long[entry_cursor_row, :] = long_ffill[entry_cursor_row, :]
        entry_short = np.zeros((n_rows, n_sym), dtype=bool)
        entry_short[entry_cursor_row, :] = short_ffill[entry_cursor_row, :]

        exit_cursor_row = np.clip(exit_row - 1, 0, n_rows - 1)
        exit_due = np.zeros((n_rows, n_sym), dtype=bool)
        exit_due[exit_cursor_row, :] = True
        if n_rows >= 2:
            exit_due[n_rows - 2, :] = True

        return {
            "entry_long": entry_long,
            "entry_short": entry_short,
            "exit_due": exit_due,
            "sigma": sigma,
            "session_return": session_return,
            "position_in_range": position_in_range,
            "vol_z": vol_z,
        }

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        p = self.params
        n = len(view.symbols)
        shares = np.asarray(state.shares, dtype=np.float64)
        tradable = np.asarray(view.tradable, dtype=bool)
        exit_due = np.asarray(signals["exit_due"], dtype=bool)
        entry_long = np.asarray(signals["entry_long"], dtype=bool)
        entry_short = np.asarray(signals["entry_short"], dtype=bool)
        sigma = np.asarray(signals["sigma"], dtype=np.float64)

        if not (np.any(exit_due) or np.any(entry_long) or np.any(entry_short)):
            return None

        holding = shares != 0
        sign = np.zeros(n, dtype=np.int8)
        sign[holding] = np.sign(shares[holding]).astype(np.int8)

        new_sign = sign.copy()
        exit_now = exit_due & holding
        new_sign[exit_now] = 0

        flat_after_exit = new_sign == 0
        can_enter = flat_after_exit & tradable
        go_long = can_enter & entry_long
        go_short = can_enter & entry_short & ~go_long
        new_sign[go_long] = 1
        new_sign[go_short] = -1

        weights = np.zeros(n, dtype=np.float64)
        active = new_sign != 0
        if np.any(active):
            safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
            sigma_guard = np.maximum(safe_sigma, p.sigma_floor)
            raw = new_sign[active].astype(np.float64) / sigma_guard[active]
            clipped = np.clip(raw, -p.max_weight, p.max_weight)
            abs_sum = np.sum(np.abs(clipped))
            if abs_sum > p.gross:
                clipped = clipped * (p.gross / abs_sum)
            weights[active] = clipped

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        meta = {
            "gross": float(np.sum(np.abs(weights))),
            "long_count": float(np.sum(weights > 0)),
            "short_count": float(np.sum(weights < 0)),
            "exit_count": float(np.sum(exit_now)),
        }
        return TargetPortfolio(weights=weights, meta=meta)
