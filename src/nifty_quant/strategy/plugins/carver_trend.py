"""
Robert Carver multi-speed volatility-scaled trend.

CAVEAT: this is Robert Carver's DAILY trend system for FUTURES, applied here to
1-minute NSE cash equities as an UNTESTED TRANSPLANT, not an endorsement — this
plugin exists to test that hypothesis.
"""
from __future__ import annotations

import math
import warnings
from typing import ClassVar, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nifty_quant.calendar import ist_label
from nifty_quant.features.core import log_returns, rolling_std
from nifty_quant.guards import check_day_offsets
from nifty_quant.settings import REGULAR_SESSION_BARS
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.registry import register

# Annualization convention only -- NOT a session-stride assumption. Every rolling/EMA
# window elsewhere in this module is bounded by `day_offsets` (session-aware), so nothing
# here depends on this constant for row indexing; irregular sessions (60-375+ bars) are
# handled correctly regardless of this value. This is purely the fixed scalar used to
# express a per-bar volatility estimate in the same "annualized fraction" units as
# `target_vol_ann` (e.g. 0.15 == 15%/year), mirroring the sqrt(252) daily-annualization
# convention used throughout quant finance. 252 nominal trading days/year *
# REGULAR_SESSION_BARS (375, the settings.py NOMINAL regular-session bar count, itself
# documented there as never to be used for row indexing) gives a standard NSE cash-equity
# nominal bars-per-year figure for this scaling purpose only.
_TRADING_DAYS_PER_YEAR = 252
_NOMINAL_BARS_PER_YEAR = _TRADING_DAYS_PER_YEAR * REGULAR_SESSION_BARS
_ANNUALIZATION_FACTOR = math.sqrt(_NOMINAL_BARS_PER_YEAR)


class CarverTrendParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speeds: list[tuple[int, int]] = Field(
        default_factory=lambda: [(8, 32), (16, 64), (32, 128)]
    )
    # each (fast, slow) EMA span pair in bars; fast < slow, both positive ints
    vol_window: int = 36  # rolling_std window (bars) over 1-bar log returns
    forecast_cap: float = 20.0
    forecast_scalar: float | None = None  # None => auto-fit per symbol per speed
    target_abs_forecast: float = 10.0
    scalar_min_periods: int = 200
    scalar_floor: float = 1e-6
    sigma_floor: float = 1e-6
    target_vol_ann: float = 0.15
    max_weight: float = 0.10
    gross: float = 1.0
    # Carver's own convention (Systematic Trading) sizes the no-trade buffer at roughly 10% of
    # a typical/average absolute position size. This system's position-size scale is
    # `max_weight` (default 0.10), so the default buffer here is 10% of that: 0.01.
    rebalance_buffer: float = 0.01
    square_off_time: str = "15:20"

    @field_validator("speeds")
    @classmethod
    def _validate_speeds(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not v:
            raise ValueError("speeds must be a non-empty list")
        for fast, slow in v:
            if fast <= 0:
                raise ValueError(f"fast span must be > 0, got {fast}")
            if slow <= fast:
                raise ValueError(f"slow span must be > fast, got slow={slow}, fast={fast}")
        return v

    @field_validator("vol_window")
    @classmethod
    def _validate_vol_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vol_window must be > 0")
        return v

    @field_validator("target_vol_ann")
    @classmethod
    def _validate_target_vol_ann(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("target_vol_ann must be > 0")
        return v

    @field_validator("rebalance_buffer")
    @classmethod
    def _validate_rebalance_buffer(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rebalance_buffer must be >= 0")
        return v

    @field_validator("scalar_min_periods")
    @classmethod
    def _validate_scalar_min_periods(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("scalar_min_periods must be > 0")
        return v


def _ema_by_session(x: np.ndarray, span: int, day_offsets: np.ndarray) -> np.ndarray:
    """Session-bounded EMA that resets at the start of every session.

    Rationale: bridging the overnight gap would let stale trend estimated from the
    prior session's closing dynamics leak into the next session's open across a
    many-hour gap that the EMA's implicit time-decay assumption does not represent;
    resetting per session matches the day-bounded convention `rolling_std`/
    `log_returns` already use elsewhere in this codebase. The loop below iterates
    over sessions (a few thousand at most for the whole dataset), never over
    individual rows.
    """
    check_day_offsets(day_offsets, x.shape[0])
    out = np.full_like(x, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    starts = day_offsets[:-1].tolist()
    ends = day_offsets[1:].tolist()
    for start, end in zip(starts, ends):
        # Unreachable: `check_day_offsets` runs as the FIRST statement of this function and
        # raises ContractViolation unless every consecutive pair in `day_offsets` is strictly
        # increasing. `starts`/`ends` are built from those exact consecutive pairs
        # (`day_offsets[:-1]` / `day_offsets[1:]`), so `end > start` always holds once that
        # guard has passed. Kept rather than deleted because it would fire — rather than
        # silently producing an empty slice — if the guard were ever moved or weakened.
        if end <= start:  # pragma: no cover - check_day_offsets invariant, see above
            continue
        ema = (
            pd.DataFrame(x[start:end])
            .ewm(alpha=alpha, adjust=False, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        out[start:end] = ema
    return out


def _expanding_forecast_scalar(
    abs_raw: np.ndarray,
    target_abs_forecast: float,
    min_periods: int,
    floor: float,
) -> np.ndarray:
    """Per-symbol causal expanding mean of `abs_raw`.

    This is intentionally GLOBAL (spans all sessions in chronological order, not
    reset per session) -- it's a long-run calibration statistic, analogous to how
    Carver fits forecast scalars on years of history, not a single day. This is
    NOT the same causal scope as `_ema_by_session` (deliberately different, don't
    conflate them). Rows before `min_periods` finite observations are NaN -- there
    is no silent fallback to a partial or full-sample estimate.
    """
    finite = np.isfinite(abs_raw)
    vals = np.where(finite, abs_raw, 0.0)
    cum_sum = np.cumsum(vals, axis=0)
    cum_count = np.cumsum(finite, axis=0)

    mean_abs = np.full_like(abs_raw, np.nan, dtype=np.float64)
    enough = cum_count >= min_periods
    np.divide(cum_sum, cum_count, out=mean_abs, where=enough & (cum_count > 0))

    denom = np.maximum(mean_abs, floor)  # floor only applies where mean_abs is finite
    scalar = np.full_like(abs_raw, np.nan, dtype=np.float64)
    np.divide(target_abs_forecast, denom, out=scalar, where=enough)
    return scalar


def _within_rebalance_buffer(
    target_weight: np.ndarray, current_weight: np.ndarray, buffer: float
) -> bool:
    """Carver's no-trade buffer, applied PORTFOLIO-WIDE: True only if every symbol's target
    has stayed within `buffer` of the reconstructed current weight.

    This is deliberately whole-portfolio, not per-symbol-partial. `current_weight` is
    reconstructed from `state.shares * price / state.equity`, which is only APPROXIMATE --
    the sizer floors to whole shares and (under the engine's default `compound=False`)
    converts weight to shares using a fixed capital basis the strategy has no direct access
    to, so `state.shares` does not exactly round-trip back to weight space. Using that
    approximate reconstruction only for a threshold comparison (this function) is safe;
    using it as an OUTPUT weight (the previous, buggy per-symbol design) is not -- it fed
    reconstruction noise back into the sizer and defeated the buffer almost entirely
    (measured: ~1% trade-count reduction instead of the 36-66% a correct whole-portfolio
    hold achieves on the same data). A true per-instrument partial hold would need an exact
    weight reconstruction this interface cannot guarantee; portfolio-wide buffering avoids
    that noise class entirely, at the cost of being coarser than Carver's per-instrument
    description.

    `current_weight` may contain NaN where the current position's weight can't be
    reconstructed (unknown/non-finite price or equity); `NaN <= buffer` is False in numpy,
    so any such symbol forces a full portfolio refresh rather than silently freezing on a
    position that can't be verified -- the conservative default. `buffer <= 0` always
    returns False (never hold), which is the backward-compatibility escape hatch.
    """
    if buffer <= 0.0:
        return False
    return bool(np.all(np.abs(target_weight - current_weight) <= buffer))


@register
class CarverTrendStrategy(Strategy):
    """Multi-speed EMA-crossover trend, volatility-normalized and vol-targeted.

    Stateless: the target weight is recomputed fresh every bar purely from that
    bar's sliced `signals`, unlike strategies that track hold-bars/cooldown state.
    """

    name: ClassVar[str] = "carver_trend"
    Params: ClassVar[type[BaseModel]] = CarverTrendParams

    def data_request(self) -> DataRequest:
        """Return the data request necessary for precompute."""
        p = self.params
        lookback_bars = max(max(slow for _, slow in p.speeds), p.vol_window) + 10
        return DataRequest(freq="1", fields=("close",), lookback_bars=lookback_bars)

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        """Fully vectorized whole-panel feature computation.

        Row t of every returned array depends only on rows <= t. Returns
        `"vol"`, `"raw_i"`/`"scalar_i"`/`"scaled_i"` per speed index i, and
        `"net_forecast"` -- all float64 arrays of shape (n_rows, n_symbols).
        """
        p = self.params
        day_offsets = panel.day_offsets
        close = panel.field("close").astype(np.float64)
        check_day_offsets(day_offsets, close.shape[0])

        # The 09:15 bar leaks the pre-open call-auction print and is corrupted in
        # this dataset. Masking it prevents it from seeding any EMA and also
        # protects the second row's log-return (which divides by the first bar's
        # close).
        close_masked = close.copy()
        close_masked[day_offsets[:-1]] = np.nan

        returns = log_returns(close_masked, day_offsets=day_offsets)
        vol = rolling_std(returns, p.vol_window, day_offsets=day_offsets)

        vol_guard = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
        vol_guard = np.maximum(vol_guard, p.sigma_floor)

        safe_close = np.where(
            np.isfinite(close_masked) & (close_masked > 0), close_masked, np.nan
        )
        denom = safe_close * vol_guard

        signals: dict[str, np.ndarray] = {"vol": vol_guard}
        clipped_list: list[np.ndarray] = []

        for i, (fast, slow) in enumerate(p.speeds):
            ema_fast = _ema_by_session(close_masked, fast, day_offsets)
            ema_slow = _ema_by_session(close_masked, slow, day_offsets)

            raw = np.full_like(denom, np.nan, dtype=np.float64)
            valid = (
                np.isfinite(ema_fast)
                & np.isfinite(ema_slow)
                & np.isfinite(denom)
                & (denom != 0)
            )
            np.divide(ema_fast - ema_slow, denom, out=raw, where=valid)
            signals[f"raw_{i}"] = raw

            if p.forecast_scalar is not None:
                scalar = np.full_like(raw, p.forecast_scalar, dtype=np.float64)
            else:
                scalar = _expanding_forecast_scalar(
                    np.abs(raw), p.target_abs_forecast, p.scalar_min_periods, p.scalar_floor
                )
            signals[f"scalar_{i}"] = scalar

            scaled = raw * scalar
            signals[f"scaled_{i}"] = scaled
            clipped_list.append(np.clip(scaled, -p.forecast_cap, p.forecast_cap))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            net_forecast = np.nanmean(np.stack(clipped_list, axis=0), axis=0)
        net_forecast = np.clip(net_forecast, -p.forecast_cap, p.forecast_cap)
        signals["net_forecast"] = net_forecast
        return signals

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        """Build a target portfolio for the current decision bar.

        `signals` arrives pre-sliced to 1-D (n_symbols,) per the base class
        contract, aligned to `view.symbols`.
        """
        p = self.params
        net_forecast = np.asarray(signals["net_forecast"], dtype=np.float64)
        vol = np.asarray(signals["vol"], dtype=np.float64)
        tradable = np.asarray(view.tradable, dtype=bool)
        trading_open = ist_label(view.ts) < p.square_off_time

        vol_guard = np.where(np.isfinite(vol) & (vol > 0), vol, np.inf)
        vol_guard = np.maximum(vol_guard, p.sigma_floor)

        vol_ann_guard = vol_guard * _ANNUALIZATION_FACTOR
        raw_weight = np.where(
            np.isfinite(net_forecast),
            (net_forecast / p.forecast_cap) * (p.target_vol_ann / vol_ann_guard),
            0.0,
        )
        raw_weight = np.where(tradable, raw_weight, 0.0)

        clipped = np.clip(raw_weight, -p.max_weight, p.max_weight)
        abs_sum = np.sum(np.abs(clipped))
        if abs_sum > p.gross:
            clipped = clipped * (p.gross / abs_sum)

        weights = np.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        if not trading_open:
            weights = np.zeros_like(weights)
        elif p.rebalance_buffer > 0.0:
            price = np.asarray(view.last("close"), dtype=np.float64)
            shares = np.asarray(state.shares, dtype=np.float64)
            if np.isfinite(state.equity) and state.equity > 0:
                current_weight = np.where(
                    np.isfinite(price) & (price > 0),
                    shares * price / state.equity,
                    np.nan,
                )
            else:
                current_weight = np.full_like(weights, np.nan)
            if _within_rebalance_buffer(weights, current_weight, p.rebalance_buffer):
                return None

        meta = {
            "gross": float(np.sum(np.abs(weights))),
            "long_count": float(np.sum(weights > 0)),
            "short_count": float(np.sum(weights < 0)),
        }
        return TargetPortfolio(weights=weights, meta=meta)
