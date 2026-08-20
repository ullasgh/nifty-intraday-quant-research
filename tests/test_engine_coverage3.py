"""Tests closing the last reachable coverage gaps in `src/nifty_quant/backtest/engine.py`.

Two originally-listed gaps are provably unreachable under the current codebase and are
therefore NOT tested here (no `# pragma: no cover` is used anywhere):

GAP A -- line 564 (`if fill_row in pending_orders: in_flight -= pending_orders[fill_row]`
inside the DECISION block). For this branch's True arm to execute, some earlier-processed
event must already have written `pending_orders[X]` where X is the fill_row a later
decision computes. Two decisions can never collide: for a fixed `decision_latency_bars` L,
fill_row(t) = t + 1 + L is strictly increasing in the decision row t, so no two distinct
decisions ever produce the same fill_row. A decision (row t_dec) finding a square-off's
earlier entry requires solving t_dec + 1 + L == t_sq + 1, i.e. t_dec == t_sq - L. Since a
decision reaching this line must be processed AFTER the square-off already wrote its entry
(loop runs in increasing row order), we need t_dec > t_sq, but t_dec == t_sq - L <= t_sq
for any L >= 0. Contradiction -- impossible for any non-negative latency. The MIRROR case
(decision writes first at fill_row X, a LATER same-day square-off computes the same X) is
real and reachable -- but it hits the square-off block's OWN check at line 589-590, a
different, already-covered pair of lines. That is the only direction physically possible.
Conclusion: line 564 is dead code under the current fill_row formula.

GAP B -- branch 456->466 (the `elif position < 0:` check falling through to `if not
triggered:` without entering either the `if position > 0` or `elif position < 0` body).
This requires `position` (== portfolio.shares[sym_idx]) to be neither > 0 nor < 0 while
also not being caught by the `== 0` guard at line 436 two lines earlier -- i.e. `position`
must be NaN (IEEE-754 NaN compares False to every relational operator including ==, >, <).
The only way `portfolio.shares` can ever hold a NaN is via a non-finite `qty` argument to
`Portfolio.apply_fills`. But BOTH callers of apply_fills (`_execute_model_fill` and
`_execute_direct_fill`) now construct a `FillBatch` from `abs(qty * price)` BEFORE calling
apply_fills, and `FillBatch.__post_init__` unconditionally raises `ValueError` if that
notional array contains any non-finite entry. Since any non-finite `qty` combined with ANY
`price` (finite, or even 0 -- NaN*0 is still NaN in IEEE-754) always produces a non-finite
notional, the ValueError fires before shares can ever be corrupted. `GrossNotionalSizer`
also forces `target_shares` finite via `np.nan_to_num` before it ever reaches an order.
There is therefore no code path, real or via a duck-typed test-double fill_model/sizer,
that can leave `portfolio.shares` non-finite under the current codebase.
Conclusion: branch 456->466 is unreachable under the current FillBatch validation.
"""

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.orders import OrderIntent
from nifty_quant.data.panel import Panel
from nifty_quant.execution.fills import FillModel, FillResult, ZeroSlippage
from nifty_quant.strategy.base import DataRequest, Strategy, TargetPortfolio
from tests.contract_fixtures import minimal_contract

SYMBOLS = ("AAA", "BBB", "CCC")
_IST = ZoneInfo("Asia/Kolkata")


def _make_grid(day_lengths: list[int], start=dt.date(2024, 1, 2)):
    dates = []
    d = start
    while len(dates) < len(day_lengths):
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    ts_chunks = []
    for day, n in zip(dates, day_lengths):
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.concatenate([[0], np.cumsum(day_lengths)]).astype(np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _make_panel(ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr):
    fields = {
        "open": open_arr.astype(np.float32),
        "high": high_arr.astype(np.float32),
        "low": low_arr.astype(np.float32),
        "close": close_arr.astype(np.float32),
        "volume": volume_arr.astype(np.float32),
    }
    return Panel(fields=fields, symbols=SYMBOLS, ts=ts, day_offsets=day_offsets, dates=dates_arr)


class _EmptyParams(BaseModel):
    pass


class _EnterOnceStrategy(Strategy):
    name = "EnterOnce"
    Params = _EmptyParams

    def __init__(self, params, entry_row_ts: int, weight: float = 0.1, symbol_idx: int = 0):
        super().__init__(params)
        self.entry_row_ts = entry_row_ts
        self.weight = weight
        self.symbol_idx = symbol_idx

    def data_request(self):
        return DataRequest()

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        if state.ts == self.entry_row_ts:
            weights = np.zeros(len(SYMBOLS), dtype=np.float64)
            weights[self.symbol_idx] = self.weight
            return TargetPortfolio(weights=weights)
        return None


class _ZeroWeightWithStopStrategy(Strategy):
    """Registers a stop for AAA via meta on a single decision row while never holding any
    AAA position (weight always 0). A non-None target ALWAYS queues an order -- even an
    all-zero one -- into pending_orders for fill_row = t+1+decision_latency_bars, and that
    fill's bookkeeping always calls `_remove_zero_stops()`, which purges a zero-position
    stop the instant it runs. To let the intrabar risk clock observe the still-registered,
    still-zero-position stop on an intermediate row (proving it correctly no-ops rather
    than that the stop simply never existed), the caller must supply
    `decision_latency_bars >= 1` so there is a real gap between the decision row (where the
    stop is set) and fill_row (the first row a purge could occur)."""

    name = "ZeroWeightWithStop"
    Params = _EmptyParams

    def __init__(self, params, entry_row_ts: int, stop_price: float = 95.0):
        super().__init__(params)
        self.entry_row_ts = entry_row_ts
        self.stop_price = stop_price

    def data_request(self):
        return DataRequest(needs_intrabar_risk=True)

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        if state.ts == self.entry_row_ts:
            return TargetPortfolio(
                weights=np.zeros(len(SYMBOLS), dtype=np.float64),
                meta={"stop:AAA": self.stop_price},
            )
        return None


class _ShortWithStopStrategy(Strategy):
    name = "ShortWithStop"
    Params = _EmptyParams

    def __init__(self, params, entry_row_ts: int, stop_price: float = 105.0):
        super().__init__(params)
        self.entry_row_ts = entry_row_ts
        self.stop_price = stop_price

    def data_request(self):
        return DataRequest(needs_intrabar_risk=True)

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        if state.ts == self.entry_row_ts:
            weights = np.zeros(len(SYMBOLS), dtype=np.float64)
            weights[0] = -0.1
            return TargetPortfolio(weights=weights, meta={"stop:AAA": self.stop_price})
        return None


class _EnterAtSpecificRowStrategy(Strategy):
    name = "EnterAtSpecificRow"
    Params = _EmptyParams

    def __init__(self, params, entry_row_ts: int, weight: float = 0.1, symbol_idx: int = 0):
        super().__init__(params)
        self.entry_row_ts = entry_row_ts
        self.weight = weight
        self.symbol_idx = symbol_idx

    def data_request(self):
        return DataRequest()

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        if state.ts == self.entry_row_ts:
            weights = np.zeros(len(SYMBOLS), dtype=np.float64)
            weights[self.symbol_idx] = self.weight
            return TargetPortfolio(weights=weights)
        return None


@dataclass
class _GhostFillModel:
    """Wraps a real FillModel but overrides ONE symbol's reported fill to a fixed,
    finite (qty, price) pair regardless of what was actually ordered for it -- used to
    prove _record_trade cannot divide by zero when a pluggable fill_model reports a
    fill for a symbol whose own order was 0 (`desired_qty == 0`)."""

    ghost_idx: int
    ghost_qty: float
    ghost_price: float
    max_participation: float = 0.02
    _inner: FillModel = field(default_factory=lambda: FillModel(slippage=ZeroSlippage()))

    def fill(self, orders, prices, bar_traded_value, tradable):
        result = self._inner.fill(orders, prices, bar_traded_value, tradable)
        filled_qty = result.filled_qty.copy()
        fill_price = result.fill_price.copy()
        filled_qty[self.ghost_idx] = self.ghost_qty
        fill_price[self.ghost_idx] = self.ghost_price
        return FillResult(
            filled_qty=filled_qty,
            fill_price=fill_price,
            rejected=result.rejected,
            unfilled_notional=result.unfilled_notional,
        )


def test_shortfall_bps_forced_zero_when_decision_price_nonfinite():
    """TEST 1: shortfall_bps forced to 0.0 when decision_price is non-finite (line 286)."""
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    close_arr[1, 0] = np.nan

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(decision_latency_bars=0, fill_model=FillModel(slippage=ZeroSlippage()))
    strategy = _EnterOnceStrategy(_EmptyParams(), entry_row_ts=ts[0], weight=0.1, symbol_idx=0)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    buy_trades = aaa_trades[aaa_trades["qty"] > 0]
    assert len(buy_trades) == 1
    row = buy_trades.iloc[0]
    assert np.isnan(row["decision_price"])
    assert row["shortfall_bps"] == 0.0
    assert np.isfinite(row["fill_price"])
    assert row["fill_price"] == pytest.approx(100.0)


def test_participation_forced_zero_when_bar_traded_value_zero_at_direct_fill():
    """TEST 2: participation forced to 0.0 when bar_traded_value <= 0 at a DIRECT fill.

    (line 296)
    """
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    volume_arr[4, 0] = 0.0

    day_start = pd.Timestamp(2024, 1, 2, 9, 15, tz=_IST)
    square_off_time = (day_start + pd.Timedelta(minutes=4)).strftime("%H:%M")

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(
        decision_latency_bars=0,
        square_off_time=square_off_time,
        fill_model=FillModel(slippage=ZeroSlippage()),
    )
    strategy = _EnterOnceStrategy(_EmptyParams(), entry_row_ts=ts[0], weight=0.1, symbol_idx=0)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    sell_trades = aaa_trades[aaa_trades["qty"] < 0]
    assert len(sell_trades) == 1
    row = sell_trades.iloc[0]
    assert row["ts"] == panel.ts[4]
    assert row["participation"] == 0.0
    assert result.forced_eod_liquidation_days == 0
    assert result.positions[-1, 0] == 0.0


def test_filled_frac_forced_zero_for_phantom_fill_with_zero_desired_order():
    """TEST 3: filled_frac forced to 0.0 for a phantom fill with zero desired order (line 303)."""
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(
        decision_latency_bars=0,
        fill_model=_GhostFillModel(ghost_idx=0, ghost_qty=37.0, ghost_price=100.0),
    )
    strategy = _EnterOnceStrategy(_EmptyParams(), entry_row_ts=ts[0], weight=0.1, symbol_idx=1)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # The ghost model's phantom fill is a REAL fill from the portfolio's point of view (it
    # goes through apply_fills like any other), so the resulting 37-share AAA position is
    # itself squared off at session end -- that second (real, ordered) close is a distinct
    # trade with its own, non-zero desired_qty and is not part of this assertion.
    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    phantom_trades = aaa_trades[aaa_trades["ts"] == panel.ts[2]]
    assert len(phantom_trades) == 1
    aaa_row = phantom_trades.iloc[0]
    assert aaa_row["qty"] == pytest.approx(37.0)
    assert aaa_row["filled_frac"] == 0.0
    assert np.isfinite(aaa_row["filled_frac"])

    bbb_trades = result.trades[result.trades["symbol"] == "BBB"]
    bbb_entry_trades = bbb_trades[bbb_trades["ts"] == panel.ts[2]]
    assert len(bbb_entry_trades) == 1
    bbb_row = bbb_entry_trades.iloc[0]
    assert bbb_row["filled_frac"] == pytest.approx(1.0)


def test_intrabar_stop_skipped_zero_position():
    """TEST 4: intrabar risk clock is a no-op for a symbol currently at zero position.

    (line 437)

    `decision_latency_bars=2` is required here (see `_ZeroWeightWithStopStrategy`'s
    docstring): it opens a gap between the decision row that registers the stop (row 1)
    and fill_row (row 4, the day's last row), during which rows 2 and 3 exercise the
    risk clock with the stop still registered and shares still at 0.
    """
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    low_arr[2, 0] = 1.0
    low_arr[3, 0] = 1.0

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(decision_latency_bars=2, fill_model=FillModel(slippage=ZeroSlippage()))
    strategy = _ZeroWeightWithStopStrategy(_EmptyParams(), entry_row_ts=ts[0], stop_price=95.0)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.n_trades == 0
    assert result.forced_eod_liquidation_days == 0
    assert np.all(result.positions == 0.0)


def test_intrabar_short_stop_nonfinite_open():
    """TEST 6: short intrabar stop fills at the raw stop price when open[t] is non-finite.

    (line 464)
    """
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    high_arr[2, 0] = 100.0
    high_arr[3, 0] = 110.0
    open_arr[3, 0] = np.nan

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(decision_latency_bars=0, fill_model=FillModel(slippage=ZeroSlippage()))
    strategy = _ShortWithStopStrategy(_EmptyParams(), entry_row_ts=ts[0], stop_price=105.0)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    cover_trades = aaa_trades[aaa_trades["qty"] > 0]
    assert len(cover_trades) == 1
    row = cover_trades.iloc[0]
    assert row["ts"] == panel.ts[3]
    assert row["qty"] == pytest.approx(10000.0)
    assert row["fill_price"] == pytest.approx(105.0)
    assert result.positions[-1, 0] == 0.0
    assert result.forced_eod_liquidation_days == 0


def test_square_off_dropped_when_position_opens_on_last_row():
    """TEST 7a (LEAD REWRITE, justified by `specs/order_lifecycle.md` section D): the
    original premise -- a decision opening a position at/after square-off, which then
    needed its own square-off intent dropped -- is now categorically disallowed. The
    decision block is gated: `t < square_off_row_for_day[day_idx]` must hold before
    `strategy.on_decision` is even called, so no ENTRY can ever be queued at or after
    the square-off row in the first place; there is nothing left to "drop".

    Hand-traced: square_off_time is set 3 minutes after the 09:15 session open, and
    with day 0 running only 6 rows (all before the literal 15:20 default), that clamps
    `square_off_row_for_day[0] == 3` exactly. The scripted entry's cursor ts is
    `ts[3]`, so its decision would fire at row `t=4` (cursor of decision row t is
    `t-1`) -- which is `>= 3`, so it is gated off before `on_decision` is ever called.
    The corrected contract is therefore: no ENTRY is queued at or after the
    square-off row, and the session ends flat with nothing to liquidate at all.

    (originally: branch 587->595)
    """
    ts, day_offsets, dates_arr = _make_grid([6, 4])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    day_start = pd.Timestamp(2024, 1, 2, 9, 15, tz=_IST)
    square_off_time = (day_start + pd.Timedelta(minutes=3)).strftime("%H:%M")

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(
        decision_latency_bars=0,
        square_off_time=square_off_time,
        fill_model=FillModel(slippage=ZeroSlippage()),
    )
    strategy = _EnterAtSpecificRowStrategy(
        _EmptyParams(), entry_row_ts=ts[3], weight=0.1, symbol_idx=0
    )

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # No ENTRY ever queued: the scripted decision's cursor ts (ts[3]) fires at
    # decision row t=4, which is >= square_off_row_for_day[0]==3, so the decision
    # block gate (spec item D) suppresses the call to `on_decision` entirely.
    entry_trades = result.trades[
        result.trades["intent"] == OrderIntent.ENTRY.value
    ]
    assert len(entry_trades) == 0
    assert result.n_trades == 0

    # Nothing to liquidate: the book was never opened, so there is nothing for
    # square-off or the terminal safety net to do.
    assert result.forced_eod_liquidation_days == 0
    assert result.positions[5, 0] == 0.0

    # Nothing carried into day 2 either.
    assert result.positions[6, 0] == 0.0


def test_muhurat_short_session_uses_immediate_square_off():
    """TEST 7b: a genuinely short (Muhurat-style) session that ends at/before the DEFAULT
    square_off_time="15:20" always takes the "immediate" branch, never the dropped-queue path."""
    ts, day_offsets, dates_arr = _make_grid([8])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(decision_latency_bars=0, fill_model=FillModel(slippage=ZeroSlippage()))
    strategy = _EnterOnceStrategy(_EmptyParams(), entry_row_ts=ts[0], weight=0.1, symbol_idx=0)

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.forced_eod_liquidation_days == 0
    assert result.positions[-1, 0] == 0.0
    assert result.n_trades == 2


def test_present_does_not_imply_tradable():
    """TEST 8: present vs tradable must never be merged (repo invariant, not a coverage gap).

    AAA's open price at row 2 is perfectly finite (the bar is `present`), but the caller
    supplies an external `tradable` mask marking it untradable there (e.g. a halt). If the
    engine ever collapsed `tradable` into `present` (using `isfinite(open)` alone to gate
    fills instead of the caller-supplied mask), this order would wrongly fill.
    """
    ts, day_offsets, dates_arr = _make_grid([5])
    n_rows = len(ts)
    n_sym = len(SYMBOLS)

    open_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    high_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    low_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    close_arr = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volume_arr = np.full((n_rows, n_sym), 1e6, dtype=np.float32)

    tradable = np.ones((n_rows, n_sym), dtype=bool)
    tradable[2, 0] = False

    panel = _make_panel(
        ts, day_offsets, dates_arr, open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    config = BacktestConfig(decision_latency_bars=0, fill_model=FillModel(slippage=ZeroSlippage()))
    strategy = _EnterOnceStrategy(_EmptyParams(), entry_row_ts=ts[0], weight=0.1, symbol_idx=0)

    result = run_backtest(
        strategy, panel, config, tradable=tradable, contract=minimal_contract()
    )

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa_trades) == 0
    assert result.rejected_order_rate > 0.0
    assert result.positions[-1, 0] == 0.0
