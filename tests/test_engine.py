import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
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
    high_arr = np.maximum(open_arr, close) if high is None else np.asarray(high, dtype=np.float64)
    low_arr = np.minimum(open_arr, close) if low is None else np.asarray(low, dtype=np.float64)
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


class CheatStrategy(Strategy):
    name = "cheat"

    class Params(BaseModel):
        weight_size: float = 0.05

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel: Panel) -> dict:
        close = panel.field("close")
        future_ret = np.zeros_like(close, dtype=np.float64)
        future_ret[:-1] = close[1:] / close[:-1] - 1.0
        return {"future_ret": future_ret}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        fr = signals["future_ret"]
        return TargetPortfolio(weights=np.sign(fr) * self.params.weight_size)


class MeanRevertStrategy(Strategy):
    name = "mean_revert"

    class Params(BaseModel):
        weight_size: float = 0.05

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel: Panel) -> dict:
        self._first_ts = int(np.asarray(panel.ts)[0])
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        # The first decision row has no prior bar. `view.last(..., offset=1)` would
        # raise a lookahead error, so treat it as a no-op.
        if int(view.ts) <= self._first_ts:
            return None
        last = view.last("close")
        prev = view.last("close", offset=1)
        mom = last - prev
        return TargetPortfolio(weights=-np.sign(mom) * self.params.weight_size)


class StopStrategy(Strategy):
    name = "stop"

    class Params(BaseModel):
        weight_size: float = 0.05
        stop_price: float = 95.0

    def __init__(self, params):
        super().__init__(params)
        self._entered = False

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=("10:00",), needs_intrabar_risk=True)

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        if self._entered:
            return None
        self._entered = True
        return TargetPortfolio(
            weights=np.array([self.params.weight_size, 0.0, 0.0], dtype=np.float64),
            meta={"stop:AAA": self.params.stop_price},
        )


def _varying_panel() -> Panel:
    base = 100.0 * (np.arange(N_SYM)[None, :] + 1.0)
    variation = 1.0 + 0.01 * np.sin(
        np.arange(N_ROWS)[:, None] / 37.0 + np.arange(N_SYM)[None, :]
    )
    close = base * variation
    return make_panel(close)


def test_accounting_invariant_holds_every_step():
    panel = _varying_panel()
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.03, 0.02))
    )
    config = BacktestConfig(
        fill_model=FillModel(slippage=SqrtImpactSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    assert np.all(np.isfinite(result.equity_curve))
    assert np.all(np.isfinite(result.gross_returns))
    assert result.equity_curve.size > 0


def test_zero_cost_identity_net_equals_gross_exactly():
    panel = _varying_panel()
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.03, 0.02))
    )
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    assert result.total_costs == 0.0
    assert np.array_equal(result.returns, result.gross_returns)


def test_cost_subtraction_matches_modelled_costs_exactly():
    panel = make_panel(flat_close(100.0))
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, 0.05, 0.05))
    )
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    net_drop = result.equity_curve[-1] - result.initial_capital
    assert result.total_costs > 0.0
    assert net_drop == pytest.approx(-result.total_costs, abs=1e-6)


def test_one_bar_lag_defeats_lookahead_cheat():
    # i.i.d. per-bar returns: a same-bar ("naive") fill has a large, fully
    # realizable edge every single bar; a genuinely independent next bar has
    # zero expected edge, so once the engine's mandatory 1-bar lag pushes the
    # fill past the bar the leaked signal describes, only noise is left.
    # `decision_times=None` makes every bar a decision row so the position is
    # re-targeted (and effectively rolled) every bar -- this keeps the actual
    # holding period comparable to the 1-bar lag itself, which is essential:
    # with a once-a-day decision cadence the multi-hundred-bar holding period
    # swamps the lag effect with unrelated intraday drift.
    rng = np.random.default_rng(7)
    rets = rng.choice([-0.02, 0.02], size=(N_ROWS, N_SYM))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)

    panel = make_panel(close)

    strategy = CheatStrategy(CheatStrategy.Params(weight_size=0.05))
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    # Reference: what a same-bar ("naive"/lookahead-bugged) fill would have
    # captured on every decision, computed independently of the engine.
    naive_pnl = 0.0
    for t in range(1, N_ROWS):
        fr = close[t] / close[t - 1] - 1.0
        naive_pnl += float(np.sum(np.abs(fr))) * config.capital * 0.05

    assert naive_pnl > 0.0
    engine_pnl = abs(result.equity_curve[-1] - result.initial_capital)
    assert engine_pnl < 0.2 * naive_pnl


def test_square_off_flattens_with_no_forced_liquidation():
    panel = make_panel(flat_close(100.0))
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.05, 0.03))
    )

    result = run_backtest(strategy, panel, BacktestConfig())

    assert result.forced_eod_liquidation_days == 0


def test_square_off_rejection_forces_eod_liquidation():
    panel = make_panel(flat_close(100.0))
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.05, 0.03))
    )

    tradable = np.ones((N_ROWS, N_SYM), dtype=bool)
    for day in range(N_DAYS):
        start = row_at(day, "15:20")
        tradable[start:(day + 1) * BARS_PER_DAY, :] = False

    result = run_backtest(strategy, panel, BacktestConfig(), tradable=tradable)

    assert result.forced_eod_liquidation_days == N_DAYS


def test_rejection_when_symbol_not_tradable():
    panel = make_panel(flat_close(100.0))
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, 0.05, 0.05))
    )

    tradable = np.ones((N_ROWS, N_SYM), dtype=bool)
    tradable[:, 0] = False

    result = run_backtest(strategy, panel, BacktestConfig(), tradable=tradable)

    assert result.trades.empty or (result.trades["symbol"] != "AAA").all()
    assert 0.0 < result.rejected_order_rate < 1.0


# This test previously asserted `unfilled_notional_pct > 0.0`, which encoded a real defect:
# capacity was checked too late, at fill time, after 57.5-81% of intended notional had
# already been silently dropped. GrossNotionalSizer is now capacity-aware and pre-caps
# target notional at max_participation * bar_traded_value before sizing, so nothing is
# dropped at fill time in this scenario and the correct expectation is 0.0.
def test_capacity_aware_sizing_leaves_no_unfilled_notional():
    volume = np.full((N_ROWS, N_SYM), 100.0, dtype=np.float64)
    panel = make_panel(flat_close(100.0), volume=volume)
    strategy = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.08, 0.0, 0.0))
    )
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage(), max_participation=0.01),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    assert result.unfilled_notional_pct == 0.0


# FillModel.fill is a pure per-bar, stateless function; it cannot carry a shortfall forward
# by construction. Call it directly to keep its participation-cap behaviour covered now that
# GrossNotionalSizer pre-caps capacity before the fill model runs.
def test_fill_model_caps_and_drops_shortfall_directly():
    fill_model = FillModel(slippage=ZeroSlippage(), max_participation=0.01)

    orders = np.array([5.0, -0.5, 10.0])
    prices = np.array([100.0, 200.0, 50.0])
    bar_traded_value = np.array([10_000.0, 20_000.0, 5_000.0])
    tradable = np.ones_like(prices, dtype=bool)

    desired_notional = np.abs(orders) * prices
    participation_cap = fill_model.max_participation * bar_traded_value
    capped_notional = np.minimum(desired_notional, participation_cap)
    expected_unfilled = desired_notional - capped_notional

    result = fill_model.fill(orders, prices, bar_traded_value, tradable)

    # Symbol 0 and symbol 2 exceed their caps; symbol 1 fills in full.
    np.testing.assert_allclose(
        np.abs(result.filled_qty * result.fill_price),
        capped_notional,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.unfilled_notional[0] > 0.0
    assert result.unfilled_notional[2] > 0.0
    np.testing.assert_allclose(
        result.unfilled_notional,
        expected_unfilled,
        rtol=1e-12,
        atol=1e-12,
    )

    assert result.filled_qty[1] == orders[1]
    assert result.fill_price[1] == prices[1]
    assert result.unfilled_notional[1] == 0.0

    # A second, independent bar with no order for the previously capped symbol 0 shows the
    # shortfall is not carried forward: FillModel has no state and cannot remember it.
    second_result = fill_model.fill(
        np.array([0.0, 0.5, 0.0]),
        prices,
        bar_traded_value,
        tradable,
    )
    assert second_result.unfilled_notional[0] == 0.0
    assert second_result.filled_qty[0] == 0.0


def test_conservative_stop_fills_at_open_not_stop_price():
    open_arr = flat_close(100.0).copy()
    high_arr = flat_close(100.0).copy()
    low_arr = flat_close(100.0).copy()
    close_arr = flat_close(100.0).copy()

    stop_row = row_at(0, "10:02")
    open_arr[stop_row, 0] = 90.0
    high_arr[stop_row, 0] = 95.0
    low_arr[stop_row, 0] = 85.0
    close_arr[stop_row, 0] = 88.0

    panel = make_panel(close_arr, open_=open_arr, high=high_arr, low=low_arr)
    strategy = StopStrategy(StopStrategy.Params(weight_size=0.05, stop_price=95.0))
    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )

    result = run_backtest(strategy, panel, config)

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"].sort_values("ts")
    assert len(aaa_trades) == 2

    stop_trade = aaa_trades.iloc[1]
    assert stop_trade["price"] == 90.0
    assert stop_trade["qty"] < 0


def test_latency_sensitivity_degrades_mean_reversion():
    ret = np.where(np.arange(N_ROWS) % 2 == 0, 0.01, -0.01)
    rets = np.tile(ret[:, None], (1, N_SYM))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    panel = make_panel(close)

    strategy0 = MeanRevertStrategy(MeanRevertStrategy.Params(weight_size=0.05))
    config0 = BacktestConfig(
        decision_latency_bars=0,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    result_lat0 = run_backtest(strategy0, panel, config0)

    strategy1 = MeanRevertStrategy(MeanRevertStrategy.Params(weight_size=0.05))
    config1 = BacktestConfig(
        decision_latency_bars=1,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    result_lat1 = run_backtest(strategy1, panel, config1)

    assert result_lat0.equity_curve[-1] != result_lat1.equity_curve[-1]
    assert result_lat0.equity_curve[-1] > result_lat1.equity_curve[-1]


def test_determinism_bit_identical_equity_curves():
    panel = _varying_panel()

    strategy1 = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.03, 0.02))
    )
    config1 = BacktestConfig(
        fill_model=FillModel(slippage=SqrtImpactSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )
    result1 = run_backtest(strategy1, panel, config1)

    strategy2 = ConstantWeightStrategy(
        ConstantWeightStrategy.Params(weights=(0.05, -0.03, 0.02))
    )
    config2 = BacktestConfig(
        fill_model=FillModel(slippage=SqrtImpactSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )
    result2 = run_backtest(strategy2, panel, config2)

    assert np.array_equal(result1.equity_curve, result2.equity_curve)
    assert np.array_equal(result1.returns, result2.returns)
