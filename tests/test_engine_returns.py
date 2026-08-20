import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    _compute_returns,
    run_backtest,
)
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts, ZeroCost
from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from tests.contract_fixtures import minimal_contract

SYMBOLS = ("AAA", "BBB", "CCC")
N_DAYS = 5
BARS_PER_DAY = 375
N_ROWS = N_DAYS * BARS_PER_DAY
N_SYM = len(SYMBOLS)
_IST = ZoneInfo("Asia/Kolkata")


def _session_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = dt.date(2024, 1, 2)
    dates: list[dt.date] = []
    d = start
    while len(dates) < N_DAYS:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)

    ts_chunks = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=BARS_PER_DAY, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.arange(0, (N_DAYS + 1) * BARS_PER_DAY, BARS_PER_DAY, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def make_panel(
    close: np.ndarray,
    *,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> Panel:
    ts, day_offsets, dates = _session_grid()
    close = np.asarray(close, dtype=np.float64)
    assert close.shape == (N_ROWS, N_SYM)
    open_arr = close.copy() if open_ is None else np.asarray(open_, dtype=np.float64)
    high_arr = (
        np.maximum(open_arr, close) if high is None else np.asarray(high, dtype=np.float64)
    )
    low_arr = (
        np.minimum(open_arr, close) if low is None else np.asarray(low, dtype=np.float64)
    )
    vol_arr = (
        np.full((N_ROWS, N_SYM), 1_000_000.0, dtype=np.float64)
        if volume is None
        else np.asarray(volume, dtype=np.float64)
    )
    fields = {
        "open": open_arr.astype(np.float32),
        "high": high_arr.astype(np.float32),
        "low": low_arr.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": vol_arr.astype(np.float32),
    }
    return Panel(fields=fields, symbols=SYMBOLS, ts=ts, day_offsets=day_offsets, dates=dates)


def flat_close(price: float = 100.0) -> np.ndarray:
    return np.full((N_ROWS, N_SYM), price, dtype=np.float64)


def row_at(day: int, hhmm: str) -> int:
    hour, minute = (int(p) for p in hhmm.split(":"))
    minute_of_day = hour * 60 + minute
    return day * BARS_PER_DAY + (minute_of_day - 9 * 60 - 15)


class ConstantWeightStrategy(Strategy):
    name = "constant_weight"

    class Params(BaseModel):
        weights: tuple[float, ...]
        decision_time: str = "10:00"

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=(self.params.decision_time,))

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        return TargetPortfolio(weights=np.array(self.params.weights, dtype=np.float64))


def _varying_panel() -> Panel:
    rng = np.random.default_rng(11)
    base = 100.0 * (np.arange(N_SYM)[None, :] + 1.0)
    wave = 1.0 + 0.01 * np.sin(np.arange(N_ROWS)[:, None] / 37.0 + np.arange(N_SYM)[None, :])
    close = base * wave + rng.normal(0.0, 0.01, size=(N_ROWS, N_SYM))
    return make_panel(close=np.maximum(close, 1.0))


def _sharpe(returns: np.ndarray) -> float:
    std = np.std(returns)
    if std > 0.0:
        return float(np.mean(returns) / std)
    return 0.0


def _flat_panel_with_open_bump(fill_row: int, fill_open: float) -> Panel:
    close = flat_close(100.0)
    open_arr = close.copy()
    open_arr[fill_row, 0] = fill_open
    return make_panel(close=close, open_=open_arr)


def _constant_weight_strategy(
    weights: tuple[float, ...], decision_time: str = "10:00"
) -> ConstantWeightStrategy:
    return ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=weights, decision_time=decision_time)
    )


def _trade_at_fill_row(
    result: BacktestResult, panel: Panel, fill_row: int, symbol: str
) -> pd.Series:
    ts = panel.ts[fill_row]
    trades = result.trades[(result.trades["symbol"] == symbol) & (result.trades["ts"] == ts)]
    assert len(trades) > 0
    return trades.iloc[0]


def test_compute_returns_normal_path() -> None:
    returns, first_ruin_index = _compute_returns(np.array([110.0, 121.0]), 100.0)
    assert returns == pytest.approx([0.10, 0.10])
    assert first_ruin_index == -1


def test_compute_returns_flags_negative_denominator() -> None:
    returns, first_ruin_index = _compute_returns(
        np.array([100.0, 50.0, -10.0, -20.0]), 100.0
    )
    assert returns[3] == 0.0
    # F3: ruin is flagged AT the crash row, not one row late.
    assert first_ruin_index == 2
    assert np.all(np.isfinite(returns))


def test_compute_returns_flags_zero_equity() -> None:
    returns, first_ruin_index = _compute_returns(np.array([100.0, 0.0, 50.0]), 100.0)
    assert returns[2] == 0.0
    # F3: ruin is flagged AT the crash row, not one row late.
    assert first_ruin_index == 1
    assert np.all(np.isfinite(returns))


def test_ruin_is_sticky() -> None:
    returns, first_ruin_index = _compute_returns(
        np.array([100.0, -5.0, 200.0, 300.0]), 100.0
    )
    # F3: ruin is flagged AT the crash row, not one row late.
    assert first_ruin_index == 1
    assert returns[2] == 0.0
    assert returns[3] == 0.0


def test_returns_reconcile_with_equity_through_ruin() -> None:
    equity = np.array([1e7, 5e6, 0.0, 5e6])
    returns, ruin_index = _compute_returns(equity, 1e7)
    # the return INTO ruin is well defined: a good positive denominator (5e6) and a
    # finite numerator (0.0). It is exactly -100% and must NOT be zeroed.
    assert returns[2] == pytest.approx(-1.0)
    assert ruin_index == 2
    # compounding the returns must reproduce the equity curve down to the wipeout
    reconstructed = 1e7 * np.cumprod(1.0 + returns)
    assert reconstructed[2] == pytest.approx(0.0)


def test_compute_returns_flags_nan_numerator() -> None:
    returns, first_ruin_index = _compute_returns(
        np.array([100.0, np.nan, 120.0]), 100.0
    )
    assert np.all(np.isfinite(returns))
    assert first_ruin_index == 1
    assert returns[1] == 0.0
    assert returns[2] == 0.0


def test_gross_sharpe_ge_net_sharpe_by_construction() -> None:
    panel = _varying_panel()
    strategy = _constant_weight_strategy((0.05, -0.03, 0.02))
    config = BacktestConfig(
        fill_model=FillModel(slippage=SqrtImpactSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.total_costs > 0.0
    assert not result.ruined

    # observed bug: gross Sharpe -0.369 vs net Sharpe -0.058 was IMPOSSIBLE
    # and is what this test guards against.
    gross_sharpe = _sharpe(result.gross_returns)
    net_sharpe = _sharpe(result.returns)
    assert gross_sharpe >= net_sharpe - 1e-9


def test_backtest_result_exposes_ruin_fields() -> None:
    result = BacktestResult(
        equity_curve=np.array([100.0]),
        returns=np.array([0.0]),
        positions=np.zeros((1, N_SYM), dtype=np.float64),
        trades=pd.DataFrame(),
        gross_returns=np.array([0.0]),
        total_costs=0.0,
        n_trades=0,
        turnover=np.array([0.0]),
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
    )

    assert result.ruined is False
    assert result.ruin_index == -1

    result_dict = result.to_dict()
    assert "ruined" in result_dict
    assert "ruin_index" in result_dict
    assert result_dict["ruined"] is False
    assert result_dict["ruin_index"] == -1


def test_trades_frame_has_shortfall_columns() -> None:
    panel = make_panel(close=flat_close(100.0))
    strategy = _constant_weight_strategy((0.05, 0.0, 0.0))
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.n_trades > 0
    expected = {"decision_price", "fill_price", "shortfall_bps", "participation", "filled_frac"}
    assert expected.issubset(set(result.trades.columns))
    assert np.all(np.isfinite(result.trades["shortfall_bps"].to_numpy()))
    assert np.all(np.isfinite(result.trades["participation"].to_numpy()))
    assert np.all(np.isfinite(result.trades["filled_frac"].to_numpy()))


def test_shortfall_sign_convention() -> None:
    decision_time = "10:00"
    fill_row = row_at(0, decision_time) + 1

    buy_panel = _flat_panel_with_open_bump(fill_row=fill_row, fill_open=101.0)
    buy_strategy = _constant_weight_strategy((0.05, 0.0, 0.0))
    buy_config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    buy_result = run_backtest(buy_strategy, buy_panel, buy_config, contract=minimal_contract())
    buy_trade = _trade_at_fill_row(buy_result, buy_panel, fill_row, "AAA")
    assert buy_trade["qty"] > 0.0
    assert buy_trade["shortfall_bps"] > 0.0

    sell_panel = _flat_panel_with_open_bump(fill_row=fill_row, fill_open=99.0)
    sell_strategy = _constant_weight_strategy((-0.05, 0.0, 0.0))
    sell_config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    sell_result = run_backtest(sell_strategy, sell_panel, sell_config, contract=minimal_contract())
    sell_trade = _trade_at_fill_row(sell_result, sell_panel, fill_row, "AAA")
    assert sell_trade["qty"] < 0.0
    assert sell_trade["shortfall_bps"] > 0.0


def test_decision_price_is_cursor_close_not_fill_price() -> None:
    decision_time = "10:00"
    fill_row = row_at(0, decision_time) + 1

    panel = _flat_panel_with_open_bump(fill_row=fill_row, fill_open=101.0)
    strategy = _constant_weight_strategy((0.05, 0.0, 0.0))
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    trade = _trade_at_fill_row(result, panel, fill_row, "AAA")
    assert trade["decision_price"] == pytest.approx(100.0)
    assert trade["fill_price"] == pytest.approx(101.0)
    assert trade["decision_price"] != trade["fill_price"]
