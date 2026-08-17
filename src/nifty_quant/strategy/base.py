"""Base strategy interface and lifecycle hooks."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
from pydantic import BaseModel

from nifty_quant.settings import REGULAR_SESSION_BARS


@dataclass(frozen=True)
class DataRequest:
    freq: Literal["1", "D"] = "1"
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    lookback_bars: int = 0
    lookback_days: int = 0
    decision_times: tuple[str, ...] | None = None
    extra_series: tuple[str, ...] = ()
    needs_intrabar_risk: bool = False

    def warmup_bars(self) -> int:
        """Upper-bound estimate of warmup bars needed for buffers.

        Uses REGULAR_SESSION_BARS (375) only as an upper-bound estimate of bars per
        session. Never use this value for actual row indexing into a panel; real
        session bar counts vary (e.g. half-days).
        """
        return max(self.lookback_bars, self.lookback_days * REGULAR_SESSION_BARS)


class MarketView(Protocol):
    """Backward-only accessors over a bar panel as of a decision cursor.

    There is structurally no way to express a forward index through this interface.
    """

    ts: int
    session_date: date
    symbols: tuple[str, ...]
    tradable: np.ndarray

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        ...

    def window(self, field: str, n: int, *, ffill: bool = False) -> np.ndarray:
        ...

    def at_time(self, field: str, hhmm: str, days_back: int = 0) -> np.ndarray:
        ...

    def daily(self, field: str, n_days: int, symbol: str | None = None) -> np.ndarray:
        ...


class LookaheadError(RuntimeError):
    """Raised when an accessor is asked for data that would require looking forward."""


class PanelLike(Protocol):
    """Structural interface satisfied by the real Panel type in nifty_quant.data.panel."""

    symbols: tuple[str, ...]
    day_offsets: np.ndarray
    ts: np.ndarray

    def field(self, name: str) -> np.ndarray:
        ...

    def minute_of_day(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TargetPortfolio:
    weights: np.ndarray
    meta: dict[str, float] = field(default_factory=dict)

    def validate(self, n_symbols: int, *, max_gross: float = 1.0) -> None:
        """Raise ValueError if weights are invalid.

        Checks shape, finiteness, and gross exposure <= max_gross * (1 + 1e-9).
        """
        if self.weights.shape != (n_symbols,):
            raise ValueError(
                f"weights shape {self.weights.shape} != expected {(n_symbols,)}"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain non-finite values (NaN/inf)")
        if np.sum(np.abs(self.weights)) > max_gross * (1 + 1e-9):
            raise ValueError("gross exposure exceeds max_gross limit")


@dataclass(frozen=True)
class PortfolioState:
    shares: np.ndarray
    cash: float
    equity: float
    ts: int


def _ffill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 0 for each column independently.

    Only meaningful for floating/complex arrays; integer/bool arrays are returned
    unchanged. The fill is backward-only in the time sense: values propagate from
    earlier rows to later rows within the supplied slice.
    """
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        return arr.copy()

    out = arr.copy()
    mask = np.isnan(arr)
    n_rows, n_cols = arr.shape

    for col in range(n_cols):
        if not mask[:, col].any():
            continue
        valid_positions = np.where(~mask[:, col])[0]
        if valid_positions.size == 0:
            continue

        row_idx = np.arange(n_rows)
        last_valid = np.where(mask[:, col], -1, row_idx)
        last_valid = np.maximum.accumulate(last_valid)

        valid_values = arr[:, col]
        fill_values = np.where(last_valid >= 0, valid_values[last_valid.clip(0)], np.nan)
        out[:, col] = np.where(mask[:, col], fill_values, valid_values)

    return out


class Strategy(ABC):
    """Abstract base class for all strategies.

    `on_decision` signals contract: `signals` is the mapping returned by
    `precompute(panel)`, but the engine slices every array in it down to the
    current decision cursor row before calling `on_decision`. By the time a
    strategy sees it, every value in `signals` is a 1-D array of shape
    `(n_symbols,)`, aligned index-for-index with `view.symbols` -- that is,
    `signals[key][i]` corresponds to `view.symbols[i]`. This is the mirror
    image of what `precompute` returns (full 2-D `(n_rows, n_symbols)`
    arrays), so strategies must not assume `signals` is 2-D.
    """

    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None:
        self.params = params

    @abstractmethod
    def data_request(self) -> DataRequest:
        ...

    @abstractmethod
    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        """Vectorized whole-panel feature computation.

        Causal: row t of every returned array may depend only on rows <= t.
        Implementations must not use per-row Python loops.
        """

    @abstractmethod
    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        """Return target weights, or None to hold current position.

        `signals` arrives pre-sliced to 1-D `(n_symbols,)` per the class
        docstring, aligned to `view.symbols`.
        """

    def on_session_start(self, session_date: date) -> None:
        """Optional hook. Default: no-op."""

    def on_session_end(self, session_date: date) -> None:
        """Optional hook. Default: no-op."""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Strategy":
        """Build an instance of this class from a raw config mapping.

        Expects cfg to contain a "params" key with a dict. Params are validated
        through cls.Params(**cfg["params"]).
        """
        params_dict = cfg["params"]
        if not isinstance(params_dict, Mapping):
            raise TypeError("'params' key must be a mapping")
        params = cls.Params(**dict(params_dict))
        return cls(params=params)


class ArrayMarketView:
    """Concrete MarketView over a mapping of full panel arrays and a cursor index."""

    def __init__(
        self,
        panel_arrays: Mapping[str, np.ndarray],
        cursor: int,
        symbols: tuple[str, ...],
        ts_array: np.ndarray,
        tradable: np.ndarray,
        session_date: date,
        minute_of_day: np.ndarray,
        day_offsets: np.ndarray,
    ) -> None:
        """Initialize from full panel arrays.

        `day_offsets` is the canonical session-start row offset array of length
        `n_days + 1`: rows for day `i` are
        `[day_offsets[i], day_offsets[i+1])`. This is the same array as
        `Panel.day_offsets` — do not confuse it with a per-row day-index array.
        The per-row day-index array is derived internally and stored as
        `self._day_index`.
        """
        self._panel_arrays = panel_arrays
        self._cursor = cursor
        self.symbols = symbols
        self.tradable = tradable
        self.session_date = session_date
        self._minute_of_day = minute_of_day
        self._day_offsets = day_offsets

        if cursor < 0 or cursor >= len(ts_array):
            raise IndexError("cursor out of range for ts_array")
        if tradable.shape != (len(symbols),):
            raise ValueError("tradable shape does not match symbols")
        if minute_of_day.shape != ts_array.shape:
            raise ValueError("minute_of_day shape does not match ts_array")

        n_rows = len(ts_array)
        if day_offsets.ndim != 1:
            raise ValueError("day_offsets must be 1-D")
        if len(day_offsets) == 0:
            raise ValueError("day_offsets must not be empty")
        if day_offsets[0] != 0:
            raise ValueError("day_offsets[0] must be 0")
        if day_offsets[-1] != n_rows:
            raise ValueError("day_offsets[-1] must equal len(ts_array)")
        if np.any(np.diff(day_offsets) <= 0):
            raise ValueError("day_offsets must be strictly increasing")

        self._day_index = np.searchsorted(day_offsets, np.arange(n_rows), side="right") - 1

        self.ts = int(ts_array[cursor])

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        """Return the field values at cursor - offset.

        Never looks forward. If ffill=True, NaNs are forward-filled from the most
        recent non-NaN value at or before the requested row for each symbol.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        idx = self._cursor - offset
        if idx < 0:
            raise LookaheadError("Not enough history for requested offset")

        if ffill:
            history = self._panel_arrays[field][: idx + 1]
            filled = _ffill_2d(history)
            return np.asarray(filled[-1])

        return np.asarray(self._panel_arrays[field][idx])

    def window(self, field: str, n: int, *, ffill: bool = False) -> np.ndarray:
        """Return the last n rows ending at the cursor, shape (n, n_sym)."""
        if n <= 0:
            raise ValueError("n must be positive")
        start = self._cursor - n + 1
        if start < 0:
            raise ValueError("Not enough history for requested window")

        arr = self._panel_arrays[field][start : self._cursor + 1]
        if ffill:
            return _ffill_2d(arr)
        return arr

    def at_time(self, field: str, hhmm: str, days_back: int = 0) -> np.ndarray:
        """Return field values at a specific HH:MM label in current or prior session.

        The label is parsed as hour*60 + minute, e.g. "10:59" -> 659. Raises
        LookaheadError if the resolved row is not strictly before the cursor.
        """
        parts = hhmm.split(":")
        if len(parts) != 2:
            raise ValueError("hhmm must be HH:MM")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            raise ValueError("hhmm must be HH:MM with integer components") from None
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("hhmm out of range")

        label = hour * 60 + minute
        current_day = int(self._day_index[self._cursor])
        target_day = current_day - days_back
        if target_day < 0:
            raise LookaheadError("Requested day is before first session")

        indices = np.where(
            (self._day_index == target_day) & (self._minute_of_day == label)
        )[0]
        if len(indices) == 0:
            raise LookaheadError(f"No bar at {hhmm} for target session")
        row = int(indices[0])
        if row >= self._cursor:
            raise LookaheadError("Requested time is at or after cursor")

        return np.asarray(self._panel_arrays[field][row])

    def daily(
        self, field: str, n_days: int, symbol: str | None = None
    ) -> np.ndarray:
        """Return prior-session-only data, one observation per session.

        For each prior session, uses the LAST row of that session as the daily
        observation. Sessions are ordered chronologically ascending, so returned
        shape is (n_days, n_sym) if symbol is None, else (n_days,). The session
        containing the cursor is never included. Raises ValueError if insufficient
        history or n_days <= 0.
        """
        if n_days <= 0:
            raise ValueError("n_days must be positive")

        current_day = int(self._day_index[self._cursor])
        unique_days = np.unique(self._day_index)
        prior_days = unique_days[unique_days < current_day]
        if len(prior_days) < n_days:
            raise ValueError("Insufficient prior sessions")

        selected_days = prior_days[-n_days:]
        rows = np.empty(n_days, dtype=np.int64)
        for i, day in enumerate(selected_days):
            day_indices = np.where(self._day_index == day)[0]
            if day_indices.size == 0:  # pragma: no cover - unreachable class invariant
                # `selected_days` is a slice of `prior_days`, which is filtered from
                # `np.unique(self._day_index)`. Every value there is by construction a day
                # index that occurs at least once, so this branch cannot be reached without a
                # prior violation of the `_day_index` invariant established in `__init__`.
                # Kept as a guard against a future refactor of that derivation; excluded from
                # coverage rather than deleted so the invariant stays asserted in the code.
                raise ValueError("No rows for prior session")
            rows[i] = day_indices[-1]

        data = self._panel_arrays[field][rows]
        if symbol is None:
            return data

        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
        sym_idx = self.symbols.index(symbol)
        return data[:, sym_idx]
