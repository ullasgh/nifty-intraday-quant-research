from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping

import numpy as np
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import sharpe_ratio
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import ZeroCost
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy import registry
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from tests.contract_fixtures import minimal_contract

from .test_causality import assert_precompute_is_causal

_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _make_ts(session_date: dt.date, hhmm: str) -> int:
    hour_str, minute_str = hhmm.split(":")
    return int(
        dt.datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            int(hour_str),
            int(minute_str),
            tzinfo=_IST,
        ).timestamp()
    )


class ObviousLeakParams(BaseModel):
    K: int = 1


class OffByOneParams(BaseModel):
    pass


def _trade_on_signal(
    view: MarketView,
    signals: Mapping[str, np.ndarray],
    state: PortfolioState,
) -> TargetPortfolio | None:
    del state
    signal = np.asarray(signals["peek"], dtype=np.float64)
    n_symbols = len(view.symbols)
    weights = np.zeros(n_symbols, dtype=np.float64)
    valid = np.isfinite(signal) & (signal != 0.0)
    weights[valid] = 0.1 * np.sign(signal[valid])
    return TargetPortfolio(
        weights=weights,
        meta={"gross": float(np.sum(np.abs(weights)))},
    )


class ObviousLeakStrategy(Strategy):
    name = "_obvious_leak_v2"
    Params = ObviousLeakParams

    def __init__(self, params: ObviousLeakParams) -> None:
        super().__init__(params)
        self.params: ObviousLeakParams = params

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        close = panel.field("close").astype(np.float64)
        n_rows, n_symbols = close.shape
        k = self.params.K
        peek = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
        if 0 < k < n_rows:
            peek[:-k, :] = close[k:, :] - close[:-k, :]
        return {"peek": peek}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return _trade_on_signal(view, signals, state)


class NoLeakTwinStrategy(Strategy):
    name = "_no_leak_obvious_twin_v2"
    Params = ObviousLeakParams

    def __init__(self, params: ObviousLeakParams) -> None:
        super().__init__(params)
        self.params: ObviousLeakParams = params

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        close = panel.field("close").astype(np.float64)
        n_rows, n_symbols = close.shape
        k = self.params.K
        peek = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
        if 0 < k < n_rows:
            peek[k:, :] = close[k:, :] - close[:-k, :]
        return {"peek": peek}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return _trade_on_signal(view, signals, state)


class OffByOneRowLeakStrategy(Strategy):
    name = "_off_by_one_leak_v2"
    Params = OffByOneParams

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        open_array = panel.field("open").astype(np.float64)
        n_rows, n_symbols = open_array.shape
        peek = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
        if n_rows > 1:
            peek[:-1, :] = (open_array[1:, :] - open_array[:-1, :]) * 2.0
        return {"peek": peek}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return _trade_on_signal(view, signals, state)


class NoLeakOffByOneTwinStrategy(Strategy):
    name = "_no_leak_off_by_one_twin_v2"
    Params = OffByOneParams

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        open_array = panel.field("open").astype(np.float64)
        n_rows, n_symbols = open_array.shape
        peek = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
        if n_rows > 1:
            peek[1:, :] = open_array[1:, :] - open_array[:-1, :]
        return {"peek": peek}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return _trade_on_signal(view, signals, state)


@pytest.fixture(autouse=True)
def _register_leaky_strategies() -> Iterator[None]:
    registry.register(ObviousLeakStrategy)
    registry.register(OffByOneRowLeakStrategy)
    try:
        yield
    finally:
        registry_dict: dict[str, type[Strategy]] = getattr(
            registry, "_REGISTRY"
        )
        for cls in (ObviousLeakStrategy, OffByOneRowLeakStrategy):
            registry_dict.pop(cls.name, None)


def _build_simple_panel(n_rows: int = 20) -> Panel:
    symbols = ("SIMPLE",)
    session_date = dt.date(2024, 1, 2)
    times = [f"09:{15 + i:02d}" for i in range(n_rows)]
    rng = np.random.default_rng(20240102)
    close = np.empty((n_rows, 1), dtype=np.float64)
    price = 100.0
    for idx in range(n_rows):
        price += rng.normal(0.0, 0.5)
        close[idx, 0] = max(price, 1.0)
    open_array = np.empty_like(close)
    open_array[0] = 100.0
    open_array[1:] = close[:-1]
    wick = 0.01 * close + 0.02
    high = np.maximum(open_array, close) + wick
    low = np.minimum(open_array, close) - wick
    low = np.maximum(low, 0.01)
    volume = np.full_like(close, 100_000.0)

    ts_list = [_make_ts(session_date, hhmm) for hhmm in times]
    fields = {
        "open": open_array.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=np.array(ts_list, dtype=np.int64),
        day_offsets=np.array([0, n_rows], dtype=np.int32),
        dates=np.array([session_date], dtype=object),
    )


def _trend_panel(
    n_sessions: int = 5, bars_per_session: int = 80, leg_len: int = 20
) -> Panel:
    symbols = ("TREND",)
    base_date = dt.date(2024, 2, 5)
    specs: list[tuple[dt.date, list[str]]] = []
    for session_idx in range(n_sessions):
        session_date = base_date + dt.timedelta(days=session_idx * 7)
        times: list[str] = []
        for bar_idx in range(bars_per_session):
            total_minutes = 9 * 60 + 15 + bar_idx
            times.append(f"{total_minutes // 60:02d}:{total_minutes % 60:02d}")
        specs.append((session_date, times))

    n_rows = n_sessions * bars_per_session
    close = np.empty((n_rows, 1), dtype=np.float64)
    price = 100.0
    direction = 1.0
    for idx in range(n_rows):
        if idx > 0 and idx % leg_len == 0:
            direction *= -1.0
        price = price + direction * 0.01
        close[idx, 0] = price

    open_array = np.empty_like(close)
    open_array[0] = 100.0
    open_array[1:] = close[:-1]
    wick = 0.01 * close + 0.02
    high = np.maximum(open_array, close) + wick
    low = np.minimum(open_array, close) - wick
    low = np.maximum(low, 0.01)
    volume = np.full_like(close, 100_000.0)

    ts_list: list[int] = []
    day_offsets: list[int] = [0]
    dates_list: list[dt.date] = []
    for session_date, times in specs:
        for hhmm in times:
            ts_list.append(_make_ts(session_date, hhmm))
        day_offsets.append(len(ts_list))
        dates_list.append(session_date)

    fields = {
        "open": open_array.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=np.array(ts_list, dtype=np.int64),
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=np.array(dates_list, dtype=object),
    )


def test_v1_helper_catches_obvious_peek() -> None:
    panel = _build_simple_panel()
    strategy = ObviousLeakStrategy(ObviousLeakParams(K=1))
    with pytest.raises(AssertionError, match="Strategy _obvious_leak"):
        assert_precompute_is_causal(strategy, panel, 5)


def test_v1_helper_catches_off_by_one_peek() -> None:
    panel = _build_simple_panel()
    strategy = OffByOneRowLeakStrategy(OffByOneParams())
    with pytest.raises(AssertionError, match="Strategy _off_by_one"):
        assert_precompute_is_causal(strategy, panel, 5)


def test_leaky_peek_survives_engine_lag_and_beats_causal_twin() -> None:
    panel = _trend_panel()
    leaky = ObviousLeakStrategy(ObviousLeakParams(K=5))
    twin = NoLeakTwinStrategy(ObviousLeakParams(K=5))

    config = BacktestConfig(
        decision_latency_bars=0,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    leaky_result = run_backtest(leaky, panel, config, contract=minimal_contract())
    twin_result = run_backtest(twin, panel, config, contract=minimal_contract())

    leaky_sharpe = float(sharpe_ratio(leaky_result.returns))
    twin_sharpe = float(sharpe_ratio(twin_result.returns))

    assert np.isfinite(leaky_sharpe)
    assert leaky_sharpe > 5.0, f"leaky Sharpe too low: {leaky_sharpe:.3f}"
    assert leaky_sharpe > twin_sharpe + 5.0, (
        f"leaky Sharpe {leaky_sharpe:.3f} not sufficiently above "
        f"causal twin Sharpe {twin_sharpe:.3f}"
    )
