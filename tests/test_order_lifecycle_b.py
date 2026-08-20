# SPEC DEFECTS / AMBIGUITIES FOUND
#
# 1. The `row_hook` test seam for item 5 is not specified anywhere in the spec, but item 5's
#    wording ("assert directly, not infer... on every row") is impossible to satisfy from
#    outside `run_backtest` without SOME internal-state hook. This test assumes a
#    `row_hook: Callable[[int, np.ndarray, dict], None] | None = None` keyword-only parameter
#    will be added, called once per row after that row's in_flight bookkeeping settles. This is
#    the test author's assumption, not a spec requirement, and the sibling test author may have
#    assumed something different — flag as a genuine divergence risk.
#    LEAD CORRECTION (AMENDMENT 1 item 1): the flagged divergence risk was real -- suite A
#    assumed a different seam. Neither survives: the invariant is now a FULL-strictness guard
#    inside the engine, not a `row_hook`. `test_05_in_flight_invariant_every_row` is rewritten
#    to a positive control (normal run completes without `ContractViolation`) plus a new
#    `test_05b_in_flight_guard_fires_on_injected_desync` negative control (AMENDMENT 2 item 4)
#    that monkeypatches `PendingOrder` inside `nifty_quant.backtest.engine`'s namespace.
#
# 2. Known conflicting sibling tests (expected to fail once B.1 append-not-overwrite lands):
#    - tests/test_engine_coverage2.py:462  test_pending_order_fill_row_collision_overwrites
#    - tests/test_engine_coverage.py:2225  test_pending_orders_collision_same_fill_row
#    These encode the OLD overwrite contract and are expected to fail once B.1 is implemented.
#    Do not edit them.
#
# 3. Item 8 says to assert the forced-EOD charge "component-wise" against
#    NSEIntradayEquityCosts + NSEDeliveryEquityCosts, but the blotter's `charges` column
#    (engine.py `_record_trade`) is a single already-summed scalar per fill -- there is no
#    per-component (brokerage/stt/exchange/.../gst) breakdown recorded per trade anywhere in
#    `BacktestResult`. This test therefore reconstructs a `FillBatch` from the recorded trade's
#    own qty/price and calls the cost models itself to get a component breakdown, rather than
#    reading components off the blotter -- the "component-wise" comparison is done against an
#    independently-computed Charges object, not against per-component blotter data (which does
#    not exist). If the lead wants a literal per-component blotter assertion, `Charges`/the
#    trade schema would need a component-wise column, which is a bigger schema change than this
#    spec asks for.
#
# 4. Item 8's own worked formula hardcodes `NSEIntradayEquityCosts()` as the "config.cost_model"
#    side of the composition, even though spec section E states the fix should compose
#    WHATEVER `config.cost_model` is configured (not necessarily NSEIntradayEquityCosts
#    specifically) with `NSEDeliveryEquityCosts()`. This test relies on `BacktestConfig`'s
#    default `cost_model=NSEIntradayEquityCosts()` to make the two consistent, but that means
#    neither test author's item-8 test (written from this spec) is likely to cover the fully
#    generic case of a NON-default `config.cost_model` composing correctly -- a coverage gap
#    inherited directly from how the spec phrased the required test, not something either author
#    can fix without contradicting the spec's own worked example.
#
# 5. No other genuine ambiguities or defects found beyond the above.

from __future__ import annotations

import datetime as dt
import importlib
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

import nifty_quant.guards as guards
from nifty_quant.backtest.engine import (
    BacktestConfig,
    run_backtest,
)
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import (
    FillBatch,
    NSEDeliveryEquityCosts,
    NSEIntradayEquityCosts,
    ZeroCost,
)
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from tests.contract_fixtures import minimal_contract

# Tolerant imports for new symbols that do not exist yet (RED).
try:
    from nifty_quant.backtest.engine import OrderIntent, PendingOrder
except ImportError:  # pragma: no cover - fallback path for future layout
    from nifty_quant.backtest.orders import OrderIntent, PendingOrder  # type: ignore

try:
    from nifty_quant.execution.costs import CompositeCostModel
except ImportError:  # pragma: no cover - not yet implemented
    CompositeCostModel = None  # type: ignore


_IST = ZoneInfo("Asia/Kolkata")
BARS_PER_DAY = 375
SYMBOLS = ("AAA", "BBB", "CCC")


def _session_grid(n_days: int):
    start = dt.date(2024, 1, 2)
    dates = []
    d = start
    while len(dates) < n_days:
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
    day_offsets = np.arange(0, (n_days + 1) * BARS_PER_DAY, BARS_PER_DAY, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _make_panel(
    n_days: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> Panel:
    ts, day_offsets, dates_arr = _session_grid(n_days)
    fields = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(fields=fields, symbols=SYMBOLS, ts=ts, day_offsets=day_offsets, dates=dates_arr)


def _flat_panel(
    n_days: int,
    *,
    price: float = 100.0,
    volume: float = 1_000_000.0,
) -> Panel:
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), price, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), volume, dtype=np.float32)
    return _make_panel(n_days, prices, prices, prices, prices, volumes)


class ConstantWeightStrategy(Strategy):
    """Holds constant weights from the first decision row onward."""

    name = "ConstantWeightStrategy"

    class Params(BaseModel):
        pass

    def __init__(self, weights: Mapping[str, float]):
        self._weights = np.array([weights.get(s, 0.0) for s in SYMBOLS], dtype=np.float64)
        super().__init__(self.Params())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel) -> dict[str, np.ndarray]:
        return {}

    def on_decision(self, view, signals, state: PortfolioState) -> TargetPortfolio | None:
        return TargetPortfolio(weights=self._weights.copy())


class ScheduledWeightStrategy(Strategy):
    """Weight schedule keyed by row index (t)."""

    name = "ScheduledWeightStrategy"

    class Params(BaseModel):
        pass

    def __init__(self, schedule: Mapping[int, Mapping[str, float]]):
        self._schedule = schedule
        super().__init__(self.Params())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel) -> dict[str, np.ndarray]:
        return {}

    def on_decision(self, view, signals, state: PortfolioState) -> TargetPortfolio | None:
        weights = np.zeros(len(SYMBOLS), dtype=np.float64)
        for t, wmap in sorted(self._schedule.items()):
            if view.ts >= t:
                for sym, w in wmap.items():
                    weights[SYMBOLS.index(sym)] = w
        return TargetPortfolio(weights=weights)


class StopRiskStrategy(Strategy):
    """Constant weight with an intrabar stop meta key that triggers a RISK_EXIT."""

    name = "StopRiskStrategy"

    class Params(BaseModel):
        pass

    def __init__(self, weights: Mapping[str, float], stop_price: float = 95.0):
        self._weights = np.array([weights.get(s, 0.0) for s in SYMBOLS], dtype=np.float64)
        self._stop_price = stop_price
        super().__init__(self.Params())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None, needs_intrabar_risk=True)

    def precompute(self, panel) -> dict[str, np.ndarray]:
        return {}

    def on_decision(self, view, signals, state: PortfolioState) -> TargetPortfolio | None:
        meta = {}
        for i, sym in enumerate(SYMBOLS):
            if self._weights[i] != 0:
                meta[f"stop:{sym}"] = self._stop_price
        return TargetPortfolio(weights=self._weights.copy(), meta=meta)


def _square_off_row(day_offsets: np.ndarray, day_idx: int, square_off_time: str = "15:20") -> int:
    hh, mm = map(int, square_off_time.split(":"))
    target_minute = hh * 60 + mm
    day_start = day_offsets[day_idx]
    day_end = day_offsets[day_idx + 1]
    for t in range(day_start, day_end):
        minute = 555 + (t - day_start)
        if minute >= target_minute:
            return t
    return day_end - 1


def test_01_partial_square_off_retries_to_flat():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    # AAA thin in square-off window [365:375)
    volumes[365:375, 0] = 100_000.0
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.forced_eod_liquidation_days == 0
    eod_exit_fills = result.trades[result.trades["intent"] == OrderIntent.EOD_EXIT]
    assert len(eod_exit_fills) > 1


def test_02_repeated_partial_square_off_monotonic():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    # AAA very thin in square-off window
    volumes[365:375, 0] = 50_000.0
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.forced_eod_liquidation_days == 0
    eod_fills = result.trades[result.trades["intent"] == OrderIntent.EOD_EXIT]
    assert len(eod_fills) >= 3

    # CORRECTED BEHAVIOUR: `cumulative` as originally written summed only the
    # (all-negative) EOD_EXIT qtys, which is cumulative SHARES SOLD -- a quantity
    # that necessarily grows in magnitude fill over fill, so `abs(cumulative)`
    # could never decrease and the original assertion was unsatisfiable by
    # construction. The quantity the test intended to track is the REMAINING
    # position: entry_shares + cumulative(EOD_EXIT qty), which does decrease
    # monotonically toward zero as each retry liquidates more of the book.
    entries = result.trades[
        (result.trades["intent"] == OrderIntent.ENTRY)
        & (result.trades["symbol"] == "AAA")
    ]
    assert len(entries) == 1
    entry_shares = float(entries.iloc[0]["qty"])

    aaa_eod = eod_fills[eod_fills["symbol"] == "AAA"].sort_values("ts")
    cumulative = 0.0
    prev_remaining = None
    for _, row in aaa_eod.iterrows():
        cumulative += row["qty"]
        remaining = abs(entry_shares + cumulative)
        if prev_remaining is not None:
            assert remaining < prev_remaining
        prev_remaining = remaining
    assert prev_remaining == 0.0


def test_03_forced_eod_when_cannot_finish():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    # AAA extremely thin in square-off window
    volumes[365:375, 0] = 1_000.0
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.forced_eod_liquidation_days == 1
    eod_fills = result.trades[result.trades["intent"] == OrderIntent.EOD_EXIT]
    forced_fills = result.trades[result.trades["intent"] == OrderIntent.FORCED_EOD]
    assert len(eod_fills) >= 1
    assert len(forced_fills) >= 1


def test_04_no_reopening_after_square_off():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    # ScheduledWeightStrategy compares `view.ts >= threshold`, and `view.ts` at decision
    # row t is `panel.ts[t - 1]` (the cursor), NOT the raw row index t. Keying the schedule
    # by row index (as the naive `{t: ...}` per-row dict would) is a mismatch: view.ts is a
    # huge epoch-seconds value, so `view.ts >= t` is true for every integer row-index key,
    # and the LAST schedule entry wins from the very first decision -- silently applying the
    # flipped weight starting at row 1 instead of at the square-off row. Fix: key the
    # schedule by actual panel ts values (a two-entry step function), and set the flip
    # threshold to `panel.ts[364]` -- the cursor value seen by the decision at row t=365 --
    # so the flip takes effect exactly at (and only at) the square-off row's decision, which
    # is precisely the row `run_backtest` must gate off per spec item D.
    schedule = {
        panel.ts[0]: {"AAA": 0.05},
        panel.ts[364]: {"AAA": -0.05},
    }

    strategy = ScheduledWeightStrategy(schedule)
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    sq_row = _square_off_row(panel.day_offsets, 0)
    entry_trades = result.trades[result.trades["intent"] == OrderIntent.ENTRY]
    # No ENTRY queued at or after square-off row
    late_entries = entry_trades[entry_trades["ts"] >= panel.ts[sq_row]]
    assert len(late_entries) == 0
    assert result.forced_eod_liquidation_days == 0


def test_05_in_flight_invariant_every_row():
    # LEAD REWRITE (AMENDMENT 1 item 1): `run_backtest` takes no `row_hook` -- a
    # test-only parameter on a production entry point is exactly the seam the
    # amendment rejects. The invariant is instead a FULL-strictness guard enforced
    # inside the engine on every row (same site/style as `guards.check_accounting`).
    # This becomes a two-part test: a positive control (a normal multi-symbol,
    # multi-session run completes without `ContractViolation`, which it can only do
    # if the guard held on every single row) and a negative control below that
    # deliberately desynchronises `in_flight` from the pending-order book and
    # asserts the guard fires.
    n_days = 2
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05, "BBB": 0.02})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")

    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.n_trades > 0


def test_05b_in_flight_guard_fires_on_injected_desync(monkeypatch):
    # Negative control for the FULL-strictness in-flight guard (AMENDMENT 1 item 1,
    # AMENDMENT 2 item 4). No production seam exists to desynchronise `in_flight`
    # from the pending-order book, so this test intrudes on the private
    # `nifty_quant.backtest.engine` module namespace -- monkeypatching `PendingOrder`
    # so every queued order's recorded `shares` is doubled relative to what actually
    # got added to `in_flight`. That is the accepted, test-only intrusion the
    # amendment sanctions in place of a production `row_hook`.
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")

    real_pending_order = PendingOrder

    def corrupt_pending_order(*args, **kwargs):
        order = real_pending_order(*args, **kwargs)
        corrupted_shares = np.asarray(order.shares, dtype=np.float64).copy() * 2.0
        object.__setattr__(order, "shares", corrupted_shares)
        return order

    engine_module = importlib.import_module("nifty_quant.backtest.engine")
    orders_module = importlib.import_module("nifty_quant.backtest.orders")

    patched_any = False
    for module in (engine_module, orders_module):
        if hasattr(module, "PendingOrder"):
            monkeypatch.setattr(module, "PendingOrder", corrupt_pending_order)
            patched_any = True

    # This negative control intentionally depends on PendingOrder being resolved
    # inside engine.py under one of the two names patched above. A third alias
    # would make the injection inert -- a known limitation of the amendment's
    # mandated best-effort technique, not a silent pass.
    assert patched_any

    with pytest.raises(guards.ContractViolation):
        run_backtest(strategy, panel, config, contract=minimal_contract())


def test_06_append_not_overwrite_both_intents():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    # Same ts-vs-row-index fix as test_04: key by panel ts, not raw row index. The bump
    # threshold `panel.ts[359]` is the cursor value seen by the decision at row t=360, so
    # AAA's weight is 0.005 for decisions t=1..359 and 0.05 from t=360 onward.
    schedule = {
        panel.ts[0]: {"AAA": 0.005, "BBB": 0.05},
        panel.ts[359]: {"AAA": 0.05, "BBB": 0.05},
    }

    strategy = ScheduledWeightStrategy(schedule)
    config = BacktestConfig(capital=1e7, square_off_time="15:20", decision_latency_bars=5)
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # Find the fill row where ENTRY and EOD_EXIT collide
    sq_row = _square_off_row(panel.day_offsets, 0)
    fill_row = sq_row + 1
    fill_ts = panel.ts[fill_row]

    row_trades = result.trades[result.trades["ts"] == fill_ts]
    aaa_trade = row_trades[row_trades["symbol"] == "AAA"]
    bbb_trade = row_trades[row_trades["symbol"] == "BBB"]

    assert len(aaa_trade) == 1
    assert len(bbb_trade) == 1

    # Compute expected quantities
    # AAA: ENTRY from t=360 decision (latency 5 -> fill at 366) for 0.05 weight = 5000 shares
    #      EOD_EXIT at sq_row+1 = 366 for -5000 shares (current position)
    #      Net = 0? No — the ENTRY is for additional shares beyond current position.
    #      At t=360, position is 500 shares (0.005 weight). Target 0.05 = 5000 shares.
    #      Order = 5000 - 500 - in_flight. in_flight at t=360 is 0 (previous order filled).
    #      So ENTRY order = 4500 shares.
    #      EOD_EXIT at t=365 queues -5000 shares (current position after ENTRY fills at 366).
    #      But wait — ENTRY fills at 366, EOD_EXIT also fills at 366.
    #      At t=365, position is still 500 shares (ENTRY not yet filled).
    #      EOD_EXIT queues -500 shares.
    #      Net at fill row 366: ENTRY +4500 + EOD_EXIT -500 = +4000.
    #      So AAA executed quantity = 4000.
    aaa_expected = 4000.0
    bbb_expected = 0.0  # BBB: ENTRY 0 (no change) + EOD_EXIT -5000 = -5000? No.
    # BBB: constant 0.05 from t=1. At t=365, position = 5000 shares.
    # EOD_EXIT queues -5000. No ENTRY pending (no weight change).
    # Net = -5000.
    bbb_expected = -5000.0

    assert aaa_trade.iloc[0]["qty"] == pytest.approx(aaa_expected)
    assert bbb_trade.iloc[0]["qty"] == pytest.approx(bbb_expected)

    # Intent convention: largest absolute contribution per symbol
    # AAA: ENTRY +4500 vs EOD_EXIT -500 -> ENTRY wins
    # BBB: EOD_EXIT -5000 only -> EOD_EXIT
    assert aaa_trade.iloc[0]["intent"] == OrderIntent.ENTRY
    assert bbb_trade.iloc[0]["intent"] == OrderIntent.EOD_EXIT


def test_07_intent_tagging_valid_and_risk_exit():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = StopRiskStrategy({"AAA": 0.05}, stop_price=95.0)
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # Every trade has a valid OrderIntent
    valid_intents = set(OrderIntent)
    assert all(intent in valid_intents for intent in result.trades["intent"])

    # At least one RISK_EXIT (stop triggers when price drops below 95)
    # With flat price 100, stop won't trigger. Need to force a price drop.
    # Rebuild with a price drop below stop.
    prices_drop = prices.copy()
    prices_drop[100:110, 0] = 90.0  # Drop below stop price
    panel_drop = _make_panel(n_days, prices_drop, prices_drop, prices_drop, prices_drop, volumes)
    result_drop = run_backtest(strategy, panel_drop, config, contract=minimal_contract())
    assert any(intent == OrderIntent.RISK_EXIT for intent in result_drop.trades["intent"])

    # Normal session produces EOD_EXIT and zero FORCED_EOD
    assert any(intent == OrderIntent.EOD_EXIT for intent in result.trades["intent"])
    assert not any(intent == OrderIntent.FORCED_EOD for intent in result.trades["intent"])


def test_08_composed_forced_eod_costs():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    # Zero volume from square-off row through last row
    volumes[365:375, :] = 0.0
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    forced_trades = result.trades[result.trades["intent"] == OrderIntent.FORCED_EOD]
    assert len(forced_trades) >= 1

    for _, trade in forced_trades.iterrows():
        qty = trade["qty"]
        price = trade["price"]
        notional = abs(qty * price)
        is_buy = qty > 0
        batch = FillBatch(notional=np.array([notional]), is_buy=np.array([is_buy]))

        intraday_charges = NSEIntradayEquityCosts().charges(batch)
        delivery_charges = NSEDeliveryEquityCosts().charges(batch)

        expected_total = intraday_charges.total[0] + delivery_charges.total[0]
        assert trade["charges"] == pytest.approx(expected_total)

        # Contrast: delivery-only has zero brokerage and GST
        assert intraday_charges.brokerage[0] > 0
        assert intraday_charges.gst[0] > 0
        assert delivery_charges.brokerage[0] == 0.0
        assert delivery_charges.gst[0] == 0.0


def test_09_ordinary_path_unchanged():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(
        capital=1e7,
        square_off_time="23:59",
        cost_model=ZeroCost(),
        fill_model=FillModel(slippage=ZeroSlippage()),
    )
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # Price never moves, costs zero -> equity invariant at capital
    assert np.allclose(result.equity_curve, 1e7)
    # One entry fill, one square-off fill
    assert result.trades.shape[0] == 2


def test_10_accounting_invariant_throughout():
    n_days = 1
    n_rows = n_days * BARS_PER_DAY
    n_sym = len(SYMBOLS)
    prices = np.full((n_rows, n_sym), 100.0, dtype=np.float32)
    volumes = np.full((n_rows, n_sym), 1_000_000.0, dtype=np.float32)
    # Thin volume at square-off to force retries
    volumes[365:375, 0] = 100_000.0
    panel = _make_panel(n_days, prices, prices, prices, prices, volumes)

    strategy = ConstantWeightStrategy({"AAA": 0.05})
    config = BacktestConfig(capital=1e7, square_off_time="15:20")
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    assert result.forced_eod_liquidation_days == 0

    # Reconstruct final cash from trades
    total_cash_flow = 0.0
    for _, trade in result.trades.iterrows():
        total_cash_flow -= trade["qty"] * trade["price"]
        total_cash_flow -= trade["charges"]
    final_cash = 1e7 + total_cash_flow
    assert final_cash == pytest.approx(result.equity_curve[-1])
