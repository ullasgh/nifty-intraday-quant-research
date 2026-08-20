import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import ZeroCost
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from tests.contract_fixtures import minimal_contract
from tests.test_engine import (
    BARS_PER_DAY,
    N_DAYS,
    SYMBOLS,
    ConstantWeightStrategy,
    flat_close,
    make_panel,
)

_DENSE_SYMBOLS = ("AAA", "BBB")
_DENSE_N_DAYS = 1
_DENSE_BARS_PER_DAY = 8
_DENSE_N_ROWS = _DENSE_N_DAYS * _DENSE_BARS_PER_DAY
_DENSE_N_SYM = len(_DENSE_SYMBOLS)
_IST = ZoneInfo("Asia/Kolkata")


def _dense_session_grid():
    start = dt.date(2024, 1, 2)
    day_start = pd.Timestamp(start.year, start.month, start.day, 9, 15, tz=_IST)
    idx = pd.date_range(day_start, periods=_DENSE_BARS_PER_DAY, freq="1min")
    idx_utc = idx.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    ts = secs.astype(np.int64)
    day_offsets = np.array([0, _DENSE_BARS_PER_DAY], dtype=np.int32)
    dates_arr = np.array([start], dtype=object)
    return ts, day_offsets, dates_arr


def make_dense_panel() -> Panel:
    ts, day_offsets, dates = _dense_session_grid()
    t = np.arange(_DENSE_N_ROWS)[:, None].astype(np.float64)
    sym = np.arange(_DENSE_N_SYM)[None, :].astype(np.float64)
    close = 100.0 + sym * 10.0 + 0.5 * t
    open_arr = close.copy()
    high_arr = close.copy()
    low_arr = close.copy()
    vol_arr = np.full((_DENSE_N_ROWS, _DENSE_N_SYM), 1_000_000.0, dtype=np.float64)
    fields = {
        "open": open_arr.astype(np.float32),
        "high": high_arr.astype(np.float32),
        "low": low_arr.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": vol_arr.astype(np.float32),
    }
    return Panel(fields=fields, symbols=_DENSE_SYMBOLS, ts=ts, day_offsets=day_offsets, dates=dates)


class DenseConstantWeightStrategy(Strategy):
    name = "dense_constant_weight"

    class Params(BaseModel):
        weights: tuple[float, ...]

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        return TargetPortfolio(weights=np.array(self.params.weights, dtype=np.float64))


def _dense_config(latency: int) -> BacktestConfig:
    return BacktestConfig(
        decision_latency_bars=latency,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )


def test_latency_zero_behaviour_unchanged():
    result = run_backtest(
        DenseConstantWeightStrategy(
            DenseConstantWeightStrategy.Params(weights=(0.05, -0.05))
        ),
        make_dense_panel(),
        _dense_config(0),
        contract=minimal_contract(),
    )

    expected_equity = np.array([
        10000000.0,
        10000000.0,
        10000227.5,
        10000453.0,
        10000676.0,
        10000897.0,
        10001115.5,
        10001115.5,
    ])
    expected_positions = np.array([
        [0.0, 0.0],
        [5000.0, -4545.0],
        [4975.0, -4524.0],
        [4950.0, -4504.0],
        [4926.0, -4484.0],
        [4901.0, -4464.0],
        [4878.0, -4444.0],
        [0.0, 0.0],
    ])

    assert np.array_equal(result.equity_curve, expected_equity)
    assert np.array_equal(result.positions, expected_positions)
    assert result.n_trades == 14


def test_no_double_ordering_at_latency_one():
    result = run_backtest(
        DenseConstantWeightStrategy(
            DenseConstantWeightStrategy.Params(weights=(0.05, -0.05))
        ),
        make_dense_panel(),
        _dense_config(1),
        contract=minimal_contract(),
    )

    max_abs = np.max(np.abs(result.positions))
    assert max_abs == pytest.approx(5000.0, abs=1.0)
    # Pre-fix formula (target - portfolio, no in_flight) fails here: max was 9975.0.
    assert max_abs < 7000.0


def test_position_magnitude_does_not_grow_with_latency():
    max_abs_by_latency = []
    for latency in (0, 1, 2):
        result = run_backtest(
            DenseConstantWeightStrategy(
                DenseConstantWeightStrategy.Params(weights=(0.05, -0.05))
            ),
            make_dense_panel(),
            _dense_config(latency),
            contract=minimal_contract(),
        )
        max_abs_by_latency.append(np.max(np.abs(result.positions)))

    for max_abs in max_abs_by_latency:
        assert max_abs == pytest.approx(5000.0, abs=1.0)
    # Pre-fix grows from 5000.0 to 9975.0 to 14925.0 across latencies 0, 1, 2.


def test_partial_fill_removes_dropped_remainder_from_in_flight():
    close = flat_close(100.0)
    volume = np.full_like(close, 1_000_000.0)
    volume[1:4, 0] = 100.0  # Cap AAA volume on early bars so the first fill is partial.

    panel = make_panel(close, volume=volume)

    config = BacktestConfig(
        decision_latency_bars=1,
        fill_model=FillModel(slippage=ZeroSlippage(), max_participation=0.1),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(
        DenseConstantWeightStrategy(
            DenseConstantWeightStrategy.Params(weights=(0.05, 0.0, 0.0))
        ),
        panel,
        config,
        contract=minimal_contract(),
    )

    aaa_trades = result.trades[
        (result.trades["symbol"] == SYMBOLS[0]) & (result.trades["qty"] != 0)
    ]
    assert len(aaa_trades) >= 2

    # After the partial fill resolves, subsequent orders continue to build the position.
    # A phantom in_flight residual would leave it frozen near the initial partial fill.
    max_aaa_position = np.max(result.positions[:, 0])
    assert max_aaa_position > 4000.0


def test_rejected_order_clears_in_flight():
    close = flat_close(100.0)
    volume = np.full_like(close, 1_000_000.0)
    panel = make_panel(close, volume=volume)

    tradable = np.ones_like(close, dtype=bool)
    tradable[1:3, 0] = False  # AAA not tradable for the early fill bars.

    config = BacktestConfig(
        decision_latency_bars=1,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(
        DenseConstantWeightStrategy(
            DenseConstantWeightStrategy.Params(weights=(0.05, 0.0, 0.0))
        ),
        panel,
        config,
        tradable=tradable,
        contract=minimal_contract(),
    )

    max_aaa_position = np.max(result.positions[:, 0])
    assert max_aaa_position > 4000.0


def test_eod_liquidation_still_flattens_at_latency():
    close = flat_close(100.0)
    volume = np.full_like(close, 1_000_000.0)

    for latency in (0, 1, 2):
        panel = make_panel(close, volume=volume)
        config = BacktestConfig(
            decision_latency_bars=latency,
            fill_model=FillModel(slippage=ZeroSlippage()),
            cost_model=ZeroCost(),
            sizer=GrossNotionalSizer(),
        )

        result = run_backtest(
            DenseConstantWeightStrategy(
                DenseConstantWeightStrategy.Params(weights=(0.05, 0.0, 0.0))
            ),
            panel,
            config,
            contract=minimal_contract(),
        )

        for day in range(N_DAYS):
            last_row = (day + 1) * BARS_PER_DAY - 1
            assert np.all(result.positions[last_row] == 0)


def test_trade_count_does_not_explode_with_latency():
    close = flat_close(100.0)
    volume = np.full_like(close, 1_000_000.0)

    results = []
    for latency in (0, 1, 2):
        panel = make_panel(close, volume=volume)
        config = BacktestConfig(
            decision_latency_bars=latency,
            fill_model=FillModel(slippage=ZeroSlippage()),
            cost_model=ZeroCost(),
            sizer=GrossNotionalSizer(),
        )

        results.append(
            run_backtest(
                ConstantWeightStrategy(
                    ConstantWeightStrategy.Params(
                        weights=(0.05, 0.0, 0.0), decision_time="10:00"
                    )
                ),
                panel,
                config,
                contract=minimal_contract(),
            )
        )

    lat0, lat1, lat2 = (r.n_trades for r in results)

    # With one decision per day there is at most one order in flight regardless of latency,
    # so post-fix counts should be nearly identical. The 3x bound is deliberately loose to
    # catch genuine explosive growth without pinning an exact ratio.
    assert lat1 <= 3 * lat0
    assert lat2 <= 3 * lat0
