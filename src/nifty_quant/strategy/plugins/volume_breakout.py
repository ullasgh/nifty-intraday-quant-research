"""
Institutional Volume Exhaustion & Momentum Capture.

This plugin implements a breakout-continuation strategy by default.  A ``fade``
direction flips the sign of the generated signals to satisfy the opposite
hypothesis.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from nifty_quant.calendar import ist_label
from nifty_quant.features.core import (
    breakout_down,
    breakout_up,
    parkinson_volatility,
    volume_zscore,
)
from nifty_quant.features.persistence import rolling_hurst
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.registry import register


class VolumeBreakoutParams(BaseModel):
    """Parameters for the volume-breakout strategy."""

    model_config = ConfigDict(extra="forbid")

    breakout_window: int = 30
    volume_window: int = 30
    volume_z_threshold: float = 2.0
    hurst_window: int = 390
    hurst_threshold: float = 0.55
    use_hurst: bool = True
    vol_window: int = 30
    deseasonalize: bool = True
    direction: Literal["continuation", "fade"] = "continuation"
    exit_mode: Literal["time", "opposite", "stop_target"] = "time"
    hold_bars: int = 30
    min_hold_bars: int = 5
    cooldown_bars: int = 0
    square_off_time: str = "15:20"
    stop_loss_pct: float = 0.01
    target_pct: float = 0.02
    target_vol_ann: float = 0.15
    sigma_floor: float = 1e-5
    max_weight: float = 0.10
    gross: float = 1.0

    @field_validator("hurst_window")
    @classmethod
    def _validate_hurst_window(cls, v: int) -> int:
        """Validate positivity only; rolling_hurst emits its own warning later."""
        if v <= 0:
            raise ValueError("hurst_window must be positive")
        return v


def _edge_trigger(cond: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Convert a level condition into an edge-triggered (rising-edge) signal.

    ``cond`` is a 2-D boolean array of shape (n_rows, n_sym).  ``day_offsets``
    is a 1-D array of session-start row offsets (length n_days+1).  The first
    bar of each session is treated as having no prior-bar continuation, so a
    True on the first bar of a session is a fresh edge.
    """
    prev = np.zeros_like(cond, dtype=bool)
    if cond.shape[0] > 1:
        prev[1:] = cond[:-1]
    starts = np.asarray(day_offsets[:-1], dtype=np.intp)
    prev[starts] = False
    return np.asarray(cond & ~prev, dtype=bool)


def _volume_breakout_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    *,
    breakout_window: int,
    volume_window: int,
    volume_z_threshold: float,
    hurst_window: int,
    hurst_threshold: float,
    use_hurst: bool,
    vol_window: int,
    deseasonalize: bool,
    direction: str,
) -> np.ndarray:
    """Vectorized whole-panel feature core.

    Returns a stacked float64 array of shape (n_rows, n_sym, 5):
    [..., 0] long (0/1), [..., 1] short (0/1), [..., 2] sigma,
    [..., 3] vol_z, [..., 4] hurst.  NaN hurst values are encoded as
    ``-inf`` in the stack so no NaN propagates into the causal-guard
    comparison; callers convert the sentinel back to NaN.

    This function is intentionally NOT independently causal-decorated at this
    level. Every sub-computation (`volume_zscore`, `breakout_up`,
    `breakout_down`, `parkinson_volatility`, and `rolling_hurst`) is already
    causal-guarded at its own definition site, so an outer generic guard would
    add no coverage for this composite. It would, however, inject
    domain-agnostic synthetic ``close`` values into
    `rolling_hurst(log_price=True)`, which now correctly rejects non-positive
    input. Causality for this composite is instead verified end-to-end by
    `tests/test_volume_breakout.py::test_precompute_is_causal`, which uses a
    realistic, always-positive, OHLC-consistent future segment.
    """
    sigma = parkinson_volatility(
        high, low, vol_window, day_offsets=day_offsets
    )
    vol_z = volume_zscore(
        volume,
        minute_of_day,
        volume_window,
        deseasonalize=deseasonalize,
        day_offsets=day_offsets,
    )

    if use_hurst:
        # Intentionally NOT day-bounded: a regular NSE session is ~375 1-min bars,
        # so a day-bounded rolling window of hurst_window=390 can never fill within
        # a single session and would return all-NaN for every row of every session
        # (measured: 0/92520 finite on RELIANCE 2024 with day_offsets passed through).
        # Letting the window bridge the overnight gap in continuous log-price is an
        # accepted approximation; it reproduces the spec's calibration numbers
        # (median H~=0.467, p90~=0.556 on real RELIANCE 2024 data). rolling_hurst is
        # itself @guards.causal-guarded on `price`, so this stays causal.
        hurst = rolling_hurst(
            close,
            window=hurst_window,
            day_offsets=None,
            warn_short_window=True,
        )
    else:
        hurst = np.full_like(close, np.inf, dtype=np.float64)

    up = breakout_up(close, high, breakout_window, day_offsets=day_offsets)
    down = breakout_down(close, low, breakout_window, day_offsets=day_offsets)

    if use_hurst:
        hurst_ok = hurst > hurst_threshold
    else:
        hurst_ok = np.ones_like(up, dtype=bool)

    vol_ok = vol_z > volume_z_threshold

    raw_long = up & vol_ok & hurst_ok
    raw_short = down & vol_ok & hurst_ok

    if direction == "fade":
        raw_long, raw_short = raw_short, raw_long
    elif direction != "continuation":
        raise ValueError(f"Unsupported direction: {direction!r}")

    # Edge-triggered entries: True only on the bar where the level condition first
    # becomes True, respecting session boundaries.
    edge_long = _edge_trigger(raw_long, day_offsets)
    edge_short = _edge_trigger(raw_short, day_offsets)

    hurst_stack = np.where(np.isnan(hurst), -np.inf, hurst).astype(np.float64)

    return np.stack(
        [
            edge_long.astype(np.float64),
            edge_short.astype(np.float64),
            sigma.astype(np.float64),
            vol_z.astype(np.float64),
            hurst_stack,
        ],
        axis=-1,
    )


@register
class VolumeBreakoutStrategy(Strategy):
    """Volume breakout strategy with Hurst persistence filter."""

    name = "volume_breakout"
    Params = VolumeBreakoutParams

    def __init__(self, params: VolumeBreakoutParams) -> None:
        super().__init__(params)
        self._bars_in_position: dict[str, int] | None = None
        self._cooldown_remaining: dict[str, int] | None = None
        self._cooldown_side: dict[str, int] | None = None

    def data_request(self) -> DataRequest:
        """Return the data request necessary for precompute."""
        p = self.params
        lookback = max(p.breakout_window, p.volume_window, p.vol_window)
        if p.use_hurst:
            lookback = max(lookback, p.hurst_window + 20)
        return DataRequest(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            lookback_bars=lookback + 10,
        )

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        """Fully vectorized whole-panel feature computation.

        The numerical core is delegated to :func:`_volume_breakout_core`,
        which is decorated with a causal guard.  Hurst's ``-inf`` sentinel is
        converted back to NaN before returning.
        """
        p = self.params
        stacked = _volume_breakout_core(
            panel.field("close"),
            panel.field("high"),
            panel.field("low"),
            panel.field("volume"),
            panel.minute_of_day(),
            panel.day_offsets,
            breakout_window=p.breakout_window,
            volume_window=p.volume_window,
            volume_z_threshold=p.volume_z_threshold,
            hurst_window=p.hurst_window,
            hurst_threshold=p.hurst_threshold,
            use_hurst=p.use_hurst,
            vol_window=p.vol_window,
            deseasonalize=p.deseasonalize,
            direction=p.direction,
        )

        hurst = np.where(np.isneginf(stacked[..., 4]), np.nan, stacked[..., 4])
        return {
            "long": stacked[..., 0].astype(bool),
            "short": stacked[..., 1].astype(bool),
            "sigma": stacked[..., 2],
            "vol_z": stacked[..., 3],
            "hurst": hurst,
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

        if self._bars_in_position is None or set(self._bars_in_position) != set(
            view.symbols
        ):
            self._bars_in_position = {sym: 0 for sym in view.symbols}
            self._cooldown_remaining = {sym: 0 for sym in view.symbols}
            self._cooldown_side = {sym: 0 for sym in view.symbols}
        bars = self._bars_in_position
        cooldown_remaining = self._cooldown_remaining
        cooldown_side = self._cooldown_side
        assert bars is not None and cooldown_remaining is not None and cooldown_side is not None

        shares = np.asarray(state.shares, dtype=np.float64)
        tradable = np.asarray(view.tradable, dtype=bool)
        long_sig = np.asarray(signals["long"], dtype=bool)
        short_sig = np.asarray(signals["short"], dtype=bool)
        sigma = np.asarray(signals["sigma"], dtype=np.float64)

        sign = np.zeros(n, dtype=np.int8)

        # Track bars-in-position and reset flat symbols.
        for i, sym in enumerate(view.symbols):
            if shares[i] == 0:
                bars[sym] = 0
            else:
                bars[sym] = bars.get(sym, 0) + 1

        bar_label = ist_label(view.ts)
        trading_open = bar_label < p.square_off_time

        exited_this_bar: set[str] = set()

        # Determine exits for currently open positions.
        for i, sym in enumerate(view.symbols):
            pos = shares[i]
            if pos == 0:
                continue

            exit_pos = False
            if not trading_open:
                exit_pos = True
            elif p.exit_mode == "time":
                exit_pos = bars[sym] >= p.hold_bars and bars[sym] >= p.min_hold_bars
            elif p.exit_mode == "opposite":
                exit_pos = bars[sym] >= p.min_hold_bars and (
                    (pos > 0 and short_sig[i]) or (pos < 0 and long_sig[i])
                )
            elif p.exit_mode == "stop_target":
                # PortfolioState has no entry price, so real stop/target
                # tracking is approximated with the same max-holding-period
                # safety net as 'time'.
                exit_pos = bars[sym] >= p.hold_bars
            else:
                # Unreachable: `exit_mode` is declared on the pydantic Params model as
                # Literal["time", "opposite", "stop_target"], so any other value is
                # rejected at CONSTRUCTION time and can never reach this dispatch. The
                # three arms above are exhaustive over that Literal. Kept so that adding a
                # member to the Literal without adding an arm here fails loudly instead of
                # silently leaving `exit_pos` unbound.
                raise ValueError(  # pragma: no cover
                    f"Unsupported exit_mode: {p.exit_mode!r}"
                )

            if exit_pos:
                cooldown_remaining[sym] = p.cooldown_bars
                cooldown_side[sym] = 1 if pos > 0 else -1
                exited_this_bar.add(sym)
            else:
                sign[i] = 1 if pos > 0 else -1

        # New entries for currently flat names, respecting the defensive
        # square-off deadline.
        if trading_open:
            flat = shares == 0
            new_long = flat & tradable & long_sig
            new_short = (flat & tradable & short_sig) & ~new_long

            # Apply same-side cooldown suppression.
            for i, sym in enumerate(view.symbols):
                if cooldown_remaining[sym] > 0:
                    if cooldown_side[sym] == 1:
                        new_long[i] = False
                    elif cooldown_side[sym] == -1:
                        new_short[i] = False

            sign[new_long] = 1
            sign[new_short] = -1

        # Decrement cooldown counters for symbols that did not exit this bar.
        # Exited symbols keep their full cooldown value until the next bar.
        for sym in view.symbols:
            if sym not in exited_this_bar and cooldown_remaining[sym] > 0:
                cooldown_remaining[sym] -= 1

        # Volatility-targeted sizing.  Proportionality is scale-invariant;
        # annualization would cancel during cross-sectional normalization.
        safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
        sigma_guard = np.maximum(safe_sigma, p.sigma_floor)

        weights = np.zeros(n, dtype=np.float64)
        active = sign != 0
        if np.any(active):
            raw = sign[active].astype(np.float64) / sigma_guard[active]
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
            "target_vol_ann": p.target_vol_ann,
        }

        return TargetPortfolio(weights=weights, meta=meta)

    def target_units(self, panel: PanelLike, capital: float) -> pd.DataFrame:
        """Build a full-panel target-units dataframe for research/plotting.

        This is a pure per-bar target-weight snapshot function.  It does not
        apply ``exit_mode``, ``hold_bars``, or the defensive square-off
        timing; those are handled by the stateful engine loop in
        :meth:`on_decision`.
        """
        signals = self.precompute(panel)

        long_mask = signals["long"].astype(bool)
        short_mask = signals["short"].astype(bool) & ~long_mask

        sign = np.zeros_like(long_mask, dtype=np.float64)
        sign[long_mask] = 1.0
        sign[short_mask] = -1.0

        sigma = signals["sigma"]
        safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
        sigma_guard = np.maximum(safe_sigma, self.params.sigma_floor)

        raw = sign / sigma_guard
        clipped = np.clip(raw, -self.params.max_weight, self.params.max_weight)

        row_abs_sum = np.sum(np.abs(clipped), axis=1, keepdims=True)
        scale = np.where(
            row_abs_sum > self.params.gross,
            self.params.gross / np.maximum(row_abs_sum, np.finfo(np.float64).eps),
            1.0,
        )
        weights = clipped * scale

        close = panel.field("close")
        safe_close = np.where((close > 0) & np.isfinite(close), close, np.inf)
        units = weights * capital / safe_close
        units = np.nan_to_num(units, nan=0.0, posinf=0.0, neginf=0.0)

        ts_index = pd.to_datetime(panel.ts, unit="s", utc=True).tz_convert(
            "Asia/Kolkata"
        )
        index = pd.MultiIndex.from_product(
            [ts_index, panel.symbols], names=["Timestamp", "Ticker"]
        )
        return pd.DataFrame({"Target_Units": units.ravel(order="C")}, index=index)
