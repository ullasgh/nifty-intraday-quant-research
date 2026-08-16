"""Cross-sectional z-score of the entry-to-decision return.

The strategy computes the log return from ``entry_time`` to ``decision_time`` for
every symbol, standardizes that return across symbols at each decision point, and
builds a long/short portfolio from the extreme z-scores.

Design notes:

- ``entry_time`` defaults to ``"09:16"``, not ``"09:15"``.  The 09:15 bar's open
  is the pre-open call-auction price and is not reliably obtainable.
- ``mode="reversal"`` is the default.  The blueprint's original rule (long the
  extreme winners, short the extreme losers) is ``mode="momentum"``.
- Fills are assumed to occur at the decision bar's open.  ``precompute`` therefore
  uses the open price at both the entry and decision checkpoints.  Using the
  decision bar's close would leak the rest of that bar's price action into the
  signal.
- ``precompute`` returns one row per session that has both checkpoints, aligned to
  the panel's full row index.  All other rows are NaN.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nifty_quant import guards
from nifty_quant.calendar import to_ist
from nifty_quant.features.core import cross_sectional_zscore
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.registry import register


def _parse_hhmm(value: str) -> int:
    """Parse ``HH:MM`` into minutes since midnight.

    Raises ``ValueError`` with a clear message on malformed input.  Pydantic wraps
    this into a ``ValidationError`` when raised inside a model validator.
    """
    try:
        hour_str, minute_str = value.split(":")
    except ValueError as exc:
        raise ValueError(
            f"Invalid HH:MM time {value!r}; expected exactly 'HH:MM'"
        ) from exc

    if (
        len(hour_str) != 2
        or len(minute_str) != 2
        or not hour_str.isdigit()
        or not minute_str.isdigit()
    ):
        raise ValueError(
            f"Invalid HH:MM time {value!r}; expected exactly 'HH:MM' with two-digit "
            "hour and minute"
        )

    hour = int(hour_str)
    minute = int(minute_str)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(
            f"Invalid HH:MM time {value!r}; hour must be 00-23 and minute 00-59"
        )

    return hour * 60 + minute


class XSecZScoreParams(BaseModel):
    """Parameters for :class:`XSecZScoreStrategy`."""

    model_config = ConfigDict(extra="forbid")

    entry_time: str = "09:16"
    decision_time: str = "11:00"
    exit_time: str = "15:20"
    long_threshold: float = 1.5
    short_threshold: float = -1.5
    mode: Literal["momentum", "reversal"] = "reversal"
    n_max_side: int | None = Field(default=None, gt=0)
    min_names: int = Field(default=20, ge=1)
    robust: bool = False
    neutral: Literal["dollar", "none"] = "dollar"
    max_weight: float = Field(default=0.10, gt=0.0)
    gross: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_times(self) -> "XSecZScoreParams":
        """Validate checkpoint ordering and warn on the 09:15 call auction."""
        entry_minutes = _parse_hhmm(self.entry_time)
        decision_minutes = _parse_hhmm(self.decision_time)
        exit_minutes = _parse_hhmm(self.exit_time)

        if decision_minutes <= entry_minutes:
            raise ValueError("decision_time must be strictly after entry_time")
        if exit_minutes <= decision_minutes:
            raise ValueError("exit_time must be strictly after decision_time")

        if self.entry_time == "09:15":
            warnings.warn(
                "entry_time 09:15 uses the pre-open call-auction open, which is not "
                "reliably obtainable; prefer 09:16 or later.",
                UserWarning,
                stacklevel=2,
            )

        return self


def _cap_top_k(mask: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Keep only the ``k`` True entries of ``mask`` with the highest ``score``.

    Parameters are both 1-D arrays of the same shape.  ``k <= 0`` returns an all-False
    mask. If fewer than ``k`` entries are True, the mask is returned unchanged (never
    promotes an entry that was not already True).
    """
    mask = np.asarray(mask, dtype=bool)
    score = np.asarray(score, dtype=np.float64)

    if k <= 0:
        return np.zeros_like(mask, dtype=bool)

    idx = np.flatnonzero(mask)
    if idx.size <= k:
        return mask.copy()

    top = idx[np.argsort(-score[idx])[:k]]
    result = np.zeros_like(mask, dtype=bool)
    result[top] = True
    return result


def _select_and_weight(
    z_row: np.ndarray,
    *,
    mode: str,
    long_threshold: float,
    short_threshold: float,
    n_max_side: int | None,
    max_weight: float,
    gross: float,
    neutral: str,
    tradable: np.ndarray | None,
) -> np.ndarray:
    """Select extreme z-scores and convert them to target weights.

    Bucket weighting is equal-weight by design, not proportional to ``|z|``, to
    avoid single-name concentration risk from an outlier z-score. ``max_weight``
    clipping happens after the gross-target allocation, so ``sum(abs(w))`` can end
    up strictly less than ``gross`` whenever clipping binds; this is intentional and
    expected, never renormalized back up.
    """
    z = np.asarray(z_row, dtype=np.float64)
    finite = np.isfinite(z)

    if not finite.any():
        return np.zeros(z.shape[0], dtype=np.float64)

    selection_z = z if mode == "momentum" else -z

    long_mask = finite & (selection_z > long_threshold)
    short_mask = finite & (selection_z < short_threshold)

    if n_max_side is not None:
        long_mask = _cap_top_k(long_mask, selection_z, n_max_side)
        short_mask = _cap_top_k(short_mask, -selection_z, n_max_side)

    if tradable is not None:
        tradable_bool = np.asarray(tradable, dtype=bool)
        long_mask = long_mask & tradable_bool
        short_mask = short_mask & tradable_bool

    w = np.zeros(z.shape[0], dtype=np.float64)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    if neutral == "dollar":
        if n_long > 0:
            w[long_mask] = (gross / 2.0) / n_long
        if n_short > 0:
            w[short_mask] = -(gross / 2.0) / n_short
    else:
        n_sel = n_long + n_short
        if n_sel > 0:
            per = gross / n_sel
            w[long_mask] = per
            w[short_mask] = -per

    np.clip(w, -max_weight, max_weight, out=w)
    w = np.where(np.isfinite(w), w, 0.0)
    return w


@guards.causal(row_arg=0, n_probes=3)
def _cross_sectional_return_zscore(
    returns: np.ndarray, *, min_names: int, robust: bool
) -> np.ndarray:
    """Compute per-session cross-sectional z-scores.

    ``returns`` has shape ``(n_common_sessions, n_symbols)``, one row per session
    that has both the entry-time and decision-time checkpoints.  The computation
    is row-independent by construction (``cross_sectional_zscore`` never looks
    across rows), so this guard is a structural proof, not just documentation,
    that no session's z-score can depend on another session's prices.
    """
    return cross_sectional_zscore(returns, min_names=min_names, robust=robust)


@register
class XSecZScoreStrategy(Strategy):
    """Cross-sectional entry-to-decision z-score strategy."""

    name = "xsec_zscore"
    Params = XSecZScoreParams

    def data_request(self) -> DataRequest:
        return DataRequest(
            freq="1",
            fields=("open", "close"),
            decision_times=(self.params.decision_time, self.params.exit_time),
        )

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        """Compute full-panel signal and return matrices.

        The actual row-wise numeric step lives in
        :func:`_cross_sectional_return_zscore`, which is decorated with
        ``@guards.causal(row_arg=0)``.  The guard cannot be placed directly on
        ``precompute`` because ``guards.causal`` requires its ``row_arg`` to resolve
        to an actual ndarray argument, while ``precompute`` only receives a
        ``PanelLike`` object.
        """
        p = self.params
        n_rows = len(panel.ts)
        n_sym = len(panel.symbols)

        entry_rows = panel.rows_at_time(p.entry_time)
        decision_rows = panel.rows_at_time(p.decision_time)

        day_index = (
            np.searchsorted(panel.day_offsets, np.arange(n_rows), side="right") - 1
        )
        entry_days = day_index[entry_rows]
        decision_days = day_index[decision_rows]

        common_days, entry_pos, decision_pos = np.intersect1d(
            entry_days,
            decision_days,
            assume_unique=True,
            return_indices=True,
        )

        ret_full = np.full((n_rows, n_sym), np.nan, dtype=np.float64)
        z_full = np.full((n_rows, n_sym), np.nan, dtype=np.float64)

        if common_days.size > 0:
            open_field = np.asarray(panel.field("open"), dtype=np.float64)
            entry_price = open_field[entry_rows[entry_pos]]
            decision_price = open_field[decision_rows[decision_pos]]

            valid = (
                np.isfinite(entry_price)
                & (entry_price > 0)
                & np.isfinite(decision_price)
                & (decision_price > 0)
            )

            ratio = np.full_like(entry_price, np.nan)
            np.divide(decision_price, entry_price, out=ratio, where=valid)

            ret_common = np.full_like(entry_price, np.nan)
            np.log(ratio, out=ret_common, where=valid)

            if common_days.size < 2:
                # The causal guard's row_arg shape[0]>=2 requirement bypasses the
                # decorator when there is only one common session.
                z_common = cross_sectional_zscore(
                    ret_common, min_names=p.min_names, robust=p.robust
                )
            else:
                z_common = _cross_sectional_return_zscore(
                    ret_common, min_names=p.min_names, robust=p.robust
                )

            target_rows = decision_rows[decision_pos]
            ret_full[target_rows] = ret_common
            z_full[target_rows] = z_common

        return {"zscore": z_full, "ret": ret_full}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        """Produce a target portfolio at the current decision bar.

        ``signals[key]`` here is a 1-D array of shape ``(len(view.symbols),)`` -
        the precompute-output row for the current decision bar.  The engine slices
        ``precompute(panel)[key][row_index_of(view.ts)]`` before calling this
        method.  This method always returns an explicit target portfolio, including
        an all-zero portfolio for "no trade this session"; ``None`` would mean
        "hold current position", which is not the right semantics for a strategy
        that flattens intraday.
        """
        p = self.params
        z_row = np.asarray(signals["zscore"], dtype=np.float64)

        if z_row.shape != (len(view.symbols),):
            raise ValueError(
                f"zscore signal shape {z_row.shape} does not match view symbols "
                f"({len(view.symbols)},)"
            )

        tradable = (
            np.asarray(view.tradable, dtype=bool)
            if view.tradable is not None
            else None
        )

        w = _select_and_weight(
            z_row,
            mode=p.mode,
            long_threshold=p.long_threshold,
            short_threshold=p.short_threshold,
            n_max_side=p.n_max_side,
            max_weight=p.max_weight,
            gross=p.gross,
            neutral=p.neutral,
            tradable=tradable,
        )

        n_long = int(np.sum(w > 0))
        n_short = int(np.sum(w < 0))

        return TargetPortfolio(
            weights=w,
            meta={"n_long": float(n_long), "n_short": float(n_short)},
        )

    def target_units(self, panel: PanelLike, capital: float) -> pd.DataFrame:
        """Vectorized diagnostic/analysis helper.

        Tradability is not available here because ``Panel`` carries no per-bar
        tradable mask; only :meth:`on_decision` via ``MarketView.tradable`` applies
        tradability filtering.
        """
        p = self.params
        signals = self.precompute(panel)
        z_full = signals["zscore"]

        decision_rows = panel.rows_at_time(p.decision_time)
        n_sym = len(panel.symbols)

        weights = np.empty((len(decision_rows), n_sym), dtype=np.float64)
        for i, row in enumerate(decision_rows):
            weights[i] = _select_and_weight(
                z_full[row],
                mode=p.mode,
                long_threshold=p.long_threshold,
                short_threshold=p.short_threshold,
                n_max_side=p.n_max_side,
                max_weight=p.max_weight,
                gross=p.gross,
                neutral=p.neutral,
                tradable=None,
            )

        open_field = np.asarray(panel.field("open"), dtype=np.float64)
        price = open_field[decision_rows]
        dollar = weights * float(capital)

        valid_price = np.isfinite(price) & (price > 0)
        units = np.full_like(dollar, np.nan)
        np.divide(dollar, price, out=units, where=valid_price)
        units = np.where(valid_price, units, 0.0)

        ts_index = to_ist(panel.ts[decision_rows])
        index = pd.MultiIndex.from_product(
            [ts_index, panel.symbols], names=["Timestamp", "Ticker"]
        )

        return pd.DataFrame(
            {"Target_Units": units.reshape(-1).astype(np.float64)},
            index=index,
        )
