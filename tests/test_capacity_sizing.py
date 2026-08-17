import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.portfolio import GrossNotionalSizer, SizingResult
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

SYMBOLS = ("AAA", "BBB", "CCC")
N_DAYS = 1
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

    def precompute(self, panel: Panel) -> dict[str, object]:
        return {}

    def on_decision(
        self,
        view: MarketView,
        signals: dict[str, object],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return TargetPortfolio(weights=np.array(self.params.weights, dtype=np.float64))


def test_backward_compatible_when_no_capacity_supplied() -> None:
    """Capacity-aware sizing must not alter the legacy plain-array return contract."""
    sizer = GrossNotionalSizer()
    weights = np.array([0.05, -0.03, 0.02])
    prices = np.array([100.0, 200.0, 50.0])
    capital = 1e7

    result = sizer.to_shares(weights, prices, capital)

    assert isinstance(result, np.ndarray)
    assert not isinstance(result, SizingResult)
    # Hand-derived: 500000/100=5000, -300000/200=-1500, 200000/50=4000.
    assert np.array_equal(result, np.array([5000.0, -1500.0, 4000.0]))


def test_target_notional_capped_by_participation() -> None:
    """A single name has no renormalisation target, so the full shortfall is reported."""
    sizer = GrossNotionalSizer()
    weights = np.array([0.05])
    prices = np.array([100.0])
    capital = 1e7
    bar_traded_value = np.array([2_500_000.0])

    result = sizer.to_shares(weights, prices, capital, bar_traded_value=bar_traded_value)

    assert isinstance(result, SizingResult)
    # capacity = 0.02 * 2_500_000 = 50_000; desired = 500_000 -> 10x too large.
    assert result.shares[0] == pytest.approx(500.0, abs=1.0)
    assert result.unmet_gross_pct == pytest.approx(0.9)


def test_renormalisation_preserves_gross_when_capacity_allows() -> None:
    """Shortfall from one capacity-bound name is absorbed by another with spare room."""
    sizer = GrossNotionalSizer()
    weights = np.array([0.05, 0.03])
    prices = np.array([100.0, 100.0])
    capital = 1e7
    bar_traded_value = np.array([2_500_000.0, 100_000_000.0])

    result = sizer.to_shares(weights, prices, capital, bar_traded_value=bar_traded_value)

    assert isinstance(result, SizingResult)
    # Name A capped at 50_000, name B absorbs the 450_000 shortfall -> 750_000.
    np.testing.assert_allclose(result.shares, [500.0, 7500.0], atol=1.0)
    gross = float(np.sum(np.abs(result.shares) * prices))
    assert gross == pytest.approx(800_000.0, rel=1e-6)
    assert result.unmet_gross_pct == pytest.approx(0.0, abs=1e-6)


def test_gross_falls_short_when_capacity_insufficient() -> None:
    """If all capacity is exhausted the unmet fraction must be reported, not absorbed."""
    sizer = GrossNotionalSizer()
    weights = np.array([0.05, 0.03])
    prices = np.array([100.0, 100.0])
    capital = 1e7
    bar_traded_value = np.array([2_500_000.0, 2_500_000.0])

    result = sizer.to_shares(weights, prices, capital, bar_traded_value=bar_traded_value)

    assert isinstance(result, SizingResult)
    # Both names capped at 50_000 -> 500 shares each, achieved gross = 100_000.
    np.testing.assert_allclose(result.shares, [500.0, 500.0], atol=1.0)
    gross = float(np.sum(np.abs(result.shares) * prices))
    assert gross <= 800_000.0 + 1e-6
    # Unmet = (800_000 - 100_000) / 800_000 = 0.875.
    assert result.unmet_gross_pct == pytest.approx(0.875, abs=1e-6)


def test_zero_or_nan_bar_traded_value_is_safe() -> None:
    """Zero or NaN bar traded value must be treated as zero capacity, never propagate NaNs."""
    sizer = GrossNotionalSizer()
    weights = np.array([0.05, 0.05, 0.05])
    prices = np.array([100.0, 100.0, 100.0])
    capital = 1e7
    bar_traded_value = np.array([0.0, np.nan, 1_000_000_000.0])

    result = sizer.to_shares(weights, prices, capital, bar_traded_value=bar_traded_value)

    assert isinstance(result, SizingResult)
    assert result.shares[0] == 0.0
    assert result.shares[1] == 0.0
    # Name C absorbs the full 1_000_000 shortfall from A and B -> 1_500_000 notional.
    assert result.shares[2] == pytest.approx(15000.0, abs=1.0)
    assert np.all(np.isfinite(result.shares))
    assert np.isfinite(result.unmet_gross_pct)


def test_unfilled_notional_pct_drops_end_to_end() -> None:
    """The D2 seam: capacity-aware sizing removes the fill-model rejection spike."""
    weights = np.array([0.08, -0.08, 0.05])
    prices = np.array([100.0, 100.0, 100.0])
    capital = 1e7
    volume = np.array([2_000.0, 2_000.0, 2_000.0])  # thin bar
    bar_traded_value = prices * volume  # 200_000 each
    tradable = np.array([True, True, True])

    sizer = GrossNotionalSizer()
    fill_model = FillModel(slippage=ZeroSlippage())

    naive_shares = sizer.to_shares(weights, prices, capital)
    naive_result = fill_model.fill(naive_shares, prices, bar_traded_value, tradable)
    naive_desired = float(np.sum(np.abs(naive_shares) * prices))
    naive_unfilled_pct = float(np.sum(naive_result.unfilled_notional)) / naive_desired

    aware = sizer.to_shares(
        weights,
        prices,
        capital,
        bar_traded_value=bar_traded_value,
        max_participation=fill_model.max_participation,
    )
    aware_result = fill_model.fill(aware.shares, prices, bar_traded_value, tradable)
    aware_desired = float(np.sum(np.abs(aware.shares) * prices))
    aware_unfilled_pct = (
        float(np.sum(aware_result.unfilled_notional)) / aware_desired
        if aware_desired > 0.0
        else 0.0
    )

    # Naive sizing asks for 2.1 million gross; the bar caps each name at 4 thousand.
    assert naive_unfilled_pct > 0.5
    # Capacity-aware sizing already caps each name at max_participation * bar_traded_value,
    # so fill time should drop nothing. 1% headroom covers whole-share rounding only.
    assert aware_unfilled_pct < 0.01


def test_capacity_uses_cursor_bar_not_decision_bar() -> None:
    """Sizing must never see the bar it is about to trade into (one-bar lookahead guard)."""
    decision_row = row_at(0, "10:00")
    baseline_volume = np.full((N_ROWS, N_SYM), 1_000_000.0, dtype=np.float64)
    poisoned_volume = baseline_volume.copy()
    poisoned_volume[decision_row, :] = 1.0

    panel_baseline = make_panel(flat_close(100.0), volume=baseline_volume)
    panel_poisoned = make_panel(flat_close(100.0), volume=poisoned_volume)

    config = BacktestConfig(
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=ZeroCost(),
        sizer=GrossNotionalSizer(),
    )
    strategy_baseline = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(
            weights=(0.05, -0.03, 0.02), decision_time="10:00"
        )
    )
    strategy_poisoned = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(
            weights=(0.05, -0.03, 0.02), decision_time="10:00"
        )
    )

    result_baseline = run_backtest(strategy_baseline, panel_baseline, config)
    result_poisoned = run_backtest(strategy_poisoned, panel_poisoned, config)

    assert np.array_equal(result_baseline.positions, result_poisoned.positions)
    assert len(result_baseline.trades) == len(result_poisoned.trades)
    np.testing.assert_array_equal(
        result_baseline.trades["qty"], result_poisoned.trades["qty"]
    )


def test_whole_shares_and_min_trade_notional_still_enforced() -> None:
    """Capacity awareness must not bypass legacy guards: dust filter and whole-share rounding."""
    sizer = GrossNotionalSizer()
    capital = 1e7
    ample_capacity = np.array([1_000_000_000.0])

    # Case 1: desired notional 24_900 is below the 25_000 min-trade-notional dust filter.
    dust_weights = np.array([0.00249])
    dust_prices = np.array([100.0])
    result_dust = sizer.to_shares(
        dust_weights,
        dust_prices,
        capital,
        bar_traded_value=ample_capacity,
        max_participation=0.02,
    )
    assert isinstance(result_dust, SizingResult)
    assert result_dust.shares[0] == 0.0

    # Case 2: desired notional 537_000, price 97 -> raw shares 5536.08... floor to 5536.
    round_weights = np.array([0.0537])
    round_prices = np.array([97.0])
    result_round = sizer.to_shares(
        round_weights,
        round_prices,
        capital,
        bar_traded_value=ample_capacity,
        max_participation=0.02,
    )
    assert isinstance(result_round, SizingResult)
    assert result_round.shares[0] == 5536.0
