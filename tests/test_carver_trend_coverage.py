"""Coverage tests for CarverTrendStrategy."""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nifty_quant.strategy.base import PortfolioState, TargetPortfolio
from nifty_quant.strategy.plugins.carver_trend import (
    CarverTrendParams,
    CarverTrendStrategy,
    _within_rebalance_buffer,
)


@dataclass
class _PanelStub:
    symbols: tuple[str, ...]
    day_offsets: np.ndarray
    ts: np.ndarray
    arrays: dict[str, np.ndarray]
    minute_array: np.ndarray

    def field(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def minute_of_day(self) -> np.ndarray:
        return self.minute_array


@dataclass
class _ViewStub:
    ts: int
    session_date: datetime.date
    symbols: tuple[str, ...]
    tradable: np.ndarray
    close: np.ndarray

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        if field == "close" and offset == 0:
            return self.close
        raise NotImplementedError("not needed by on_decision in these tests")


def _make_ts(n_rows: int) -> np.ndarray:
    base = pd.Timestamp("2024-01-01 09:15:00", tz="Asia/Kolkata")
    return (base.timestamp() + np.arange(n_rows, dtype=np.float64) * 60.0).astype(np.int64)


def _make_panel(
    close: np.ndarray,
    day_offsets: np.ndarray,
    symbols: tuple[str, ...] | None = None,
) -> _PanelStub:
    close = np.array(close, dtype=np.float32, copy=True)
    n_rows, n_symbols = close.shape
    if symbols is None:
        symbols = tuple(f"S{i}" for i in range(n_symbols))
    arrays = {
        "close": close,
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "volume": np.ones((n_rows, n_symbols), dtype=np.float32),
    }
    return _PanelStub(
        symbols=symbols,
        day_offsets=np.asarray(day_offsets, dtype=np.int64),
        ts=_make_ts(n_rows),
        arrays=arrays,
        minute_array=np.arange(n_rows, dtype=np.int32),
    )


def _flat_state(n_symbols: int, ts: int, equity: float = 1_000_000.0) -> PortfolioState:
    return PortfolioState(
        shares=np.zeros(n_symbols, dtype=np.float64),
        cash=0.0,
        equity=equity,
        ts=ts,
    )


def _view(
    ts: int,
    symbols: tuple[str, ...],
    tradable: np.ndarray,
    close: np.ndarray,
) -> _ViewStub:
    return _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 1),
        symbols=symbols,
        tradable=np.asarray(tradable, dtype=bool),
        close=np.asarray(close, dtype=np.float64),
    )


def _random_close(
    n_rows: int,
    n_symbols: int,
    seed: int,
    sigma: float = 0.002,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=(n_rows - 1, n_symbols))
    log_path = 4.60517 + np.cumsum(
        np.vstack((np.zeros((1, n_symbols)), steps)),
        axis=0,
    )
    return np.exp(log_path).astype(np.float32)


def test_speeds_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match=r"speeds must be a non-empty list"):
        CarverTrendParams(speeds=[])


def test_fast_span_must_be_positive() -> None:
    with pytest.raises(ValidationError, match=r"fast span must be > 0"):
        CarverTrendParams(speeds=[(0, 10)])


def test_slow_span_must_exceed_fast() -> None:
    with pytest.raises(ValidationError, match=r"slow span must be > fast"):
        CarverTrendParams(speeds=[(10, 10)])


def test_vol_window_must_be_positive() -> None:
    with pytest.raises(ValidationError, match=r"vol_window must be > 0"):
        CarverTrendParams(vol_window=0)


def test_target_vol_ann_must_be_positive() -> None:
    with pytest.raises(ValidationError, match=r"target_vol_ann must be > 0"):
        CarverTrendParams(target_vol_ann=0.0)
    with pytest.raises(ValidationError, match=r"target_vol_ann must be > 0"):
        CarverTrendParams(target_vol_ann=-0.1)


def test_rebalance_buffer_must_be_nonnegative() -> None:
    with pytest.raises(ValidationError, match=r"rebalance_buffer must be >= 0"):
        CarverTrendParams(rebalance_buffer=-0.01)


def test_scalar_min_periods_must_be_positive() -> None:
    with pytest.raises(ValidationError, match=r"scalar_min_periods must be > 0"):
        CarverTrendParams(scalar_min_periods=0)


def test_params_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        CarverTrendParams(not_a_real_field=1)


def test_within_rebalance_buffer_short_circuits_non_positive_buffer() -> None:
    target_weight = np.array([0.05, -0.05])
    current_weight = target_weight.copy()
    assert _within_rebalance_buffer(target_weight, current_weight, 0.0) is False
    assert _within_rebalance_buffer(target_weight, current_weight, -0.5) is False


def test_on_decision_nonpositive_equity_uses_nan_current_weights_and_rebalances() -> None:
    symbols = ("S0", "S1")
    view = _view(
        _make_ts(2)[0],
        symbols,
        tradable=np.array([True, True]),
        close=np.array([100.0, 200.0]),
    )
    signals = {
        "net_forecast": np.array([1.0, -0.5]),
        "vol": np.array([0.01, 0.02]),
    }
    strategy = CarverTrendStrategy(params=CarverTrendParams())

    for bad_equity in (0.0, float("nan")):
        state = _flat_state(len(symbols), view.ts, equity=bad_equity)
        result = strategy.on_decision(view, signals, state)
        assert isinstance(result, TargetPortfolio)
        assert result.weights.shape == (len(symbols),)


def test_irregular_session_375_bars_then_muhurat_60_bars() -> None:
    n_symbols = 3
    symbols = ("S0", "S1", "S2")
    close = _random_close(435, n_symbols, seed=42, sigma=0.003)
    day_offsets = np.array([0, 375, 435], dtype=np.int64)
    panel = _make_panel(close, day_offsets, symbols=symbols)

    strategy = CarverTrendStrategy(params=CarverTrendParams())
    signals = strategy.precompute(panel)

    for name, arr in signals.items():
        assert arr.shape == (435, n_symbols), name

    session1 = signals["net_forecast"][:375]
    session2 = signals["net_forecast"][375:435]
    assert np.any(np.isfinite(session1))
    assert np.any(np.isfinite(session2))

    row_signals = {name: arr[434] for name, arr in signals.items()}
    view = _view(
        panel.ts[434],
        symbols,
        tradable=np.ones(n_symbols, dtype=bool),
        close=panel.field("close")[434],
    )
    state = _flat_state(n_symbols, view.ts)

    decision = strategy.on_decision(view, row_signals, state)
    assert decision is None or isinstance(decision, TargetPortfolio)
    if decision is not None:
        assert decision.weights.shape == (n_symbols,)
