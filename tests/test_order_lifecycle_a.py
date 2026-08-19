"""SPEC DEFECTS / AMBIGUITIES FOUND

1. Required test 5 originally demanded a per-row assertion over the engine-local
   ``in_flight`` and ``pending_orders`` objects, but there was no public observation
   seam.  The accepted amendment moves this invariant into an engine-side FULL
   strictness guard and restates the test as a positive normal run plus a negative
   fault-injection control.  The negative control still has an unresolved limitation:
   this file can patch only ``nifty_quant.backtest.engine.PendingOrder`` and/or
   ``nifty_quant.backtest.orders.PendingOrder``.  If engine.py binds the class under
   a third alias, the injection is inert and cannot prove that the guard fires.

2. Required test 6's example of a stop-out colliding with a square-off is not reachable
   under the rest of the specified architecture.  Intrabar stops execute immediately
   through ``_execute_direct_fill`` in their trigger row and never enter
   ``pending_orders``.  This file therefore tests the reachable ENTRY+EOD_EXIT
   collision instead.  Re-architecting RISK_EXIT orders onto the pending queue would
   be an undocumented behavior change.

3. The specification permits OrderIntent in either engine.py or orders.py, does not
   prescribe the module for PendingOrder or CompositeCostModel, and does not specify
   whether the blotter stores an enum member, enum value, or enum name.  This file
   chooses:
       from nifty_quant.backtest.orders import OrderIntent, PendingOrder
       from nifty_quant.execution.costs import CompositeCostModel
   It assumes the blotter stores OrderIntent enum members themselves.  A different
   implementation choice requires only the corresponding import/comparison updates.
   LEAD CORRECTION (AMENDMENT 2 item 2): the blotter stores `.value` as a plain
   `str`, not the enum member. `test_item7`'s `_assert_intents_are_enum_members`
   is re-pointed to value-equality against `OrderIntent` values accordingly.

4. A single blotter ``intent`` value cannot literally represent both source intents
   when multiple orders are netted into one per-symbol fill row.  This file follows
   the explicit largest-absolute-contribution convention: in the collision scenario
   the row is tagged EOD_EXIT because its -5000 contribution is larger than the
   +3000 ENTRY contribution.  Both intents are still checked across the complete
   blotter.

5. The two pre-existing collision tests named by the specification retain the old
   overwrite contract.  They are intentionally not modified here; they are lead-owned
   contract updates.

6. The engine can leak an order decided late in one session into the next session
   because pending orders are not session-bounded; this file adds a regression test
   requiring that order to be dropped and never executed. The report does not specify
   how a drop is "counted", so this test assumes reuse of the existing
   `n_rejected`/`rejected_order_rate` mechanism -- a genuine guess; a different
   mechanism, such as a dedicated field, would require only a one-line assertion
   change rather than a redesign.
   LEAD CORRECTION (AMENDMENT 2 item 3): a dedicated field,
   `BacktestResult.n_orders_dropped_at_session_end`, was added, and folding drops
   into `rejected_order_rate` is explicitly forbidden (a rejection means the market
   refused the order; a drop means the engine cancelled it). `test_item11a` is
   re-pointed to the dedicated counter.
"""

import datetime as dt
import importlib

import numpy as np
import pytest
from pydantic import BaseModel

import nifty_quant.guards as guards
from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.orders import OrderIntent, PendingOrder
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import (
    CompositeCostModel,
    FillBatch,
    NSEDeliveryEquityCosts,
    NSEIntradayEquityCosts,
    ZeroCost,
)
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy.base import DataRequest, Strategy, TargetPortfolio


BARS_PER_DAY = 375
CAPITAL = 10_000_000.0
PRICE = 100.0
AMPLE_VOLUME = 1_000_000.0
THIN_VOLUME = 20_000.0
MAX_PARTICIPATION = 0.02
CHARGE_FIELDS = (
    "brokerage",
    "stt",
    "exchange_txn",
    "sebi",
    "ipft",
    "stamp_duty",
    "gst",
)


def _session_grid(n_days: int):
    """Build the deliberately regular synthetic session grid used only here."""
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    start = dt.date(2024, 1, 2)
    dates = []
    while len(dates) < n_days:
        if start.weekday() < 5:
            dates.append(start)
        start += dt.timedelta(days=1)

    ts_parts = []
    for session_date in dates:
        session_start = dt.datetime.combine(
            session_date, dt.time(9, 15)
        ).replace(tzinfo=ist)
        ts_parts.extend(
            int((session_start + dt.timedelta(minutes=minute)).timestamp())
            for minute in range(BARS_PER_DAY)
        )

    ts = np.asarray(ts_parts, dtype=np.int64)
    day_offsets = (
        np.arange(n_days + 1, dtype=np.int32) * BARS_PER_DAY
    )
    return ts, day_offsets, np.asarray(dates, dtype=object)


def make_panel(
    close,
    n_days: int,
    *,
    symbols=None,
    open_=None,
    high=None,
    low=None,
    volume=None,
) -> Panel:
    close64 = np.asarray(close, dtype=np.float64)
    if close64.ndim != 2:
        raise ValueError("close must be a two-dimensional array")
    if close64.shape[0] != n_days * BARS_PER_DAY:
        raise ValueError("close has the wrong number of synthetic rows")

    n_symbols = close64.shape[1]
    if symbols is None:
        symbols = tuple(
            "AAA" if index == 0 else f"SYM{index}"
            for index in range(n_symbols)
        )
    symbols = tuple(symbols)
    if len(symbols) != n_symbols:
        raise ValueError("symbols and close have inconsistent widths")

    open64 = close64.copy() if open_ is None else np.asarray(open_, dtype=np.float64)
    high64 = (
        np.maximum(open64, close64)
        if high is None
        else np.asarray(high, dtype=np.float64)
    )
    low64 = (
        np.minimum(open64, close64)
        if low is None
        else np.asarray(low, dtype=np.float64)
    )
    volume64 = (
        np.full(close64.shape, AMPLE_VOLUME, dtype=np.float64)
        if volume is None
        else np.asarray(volume, dtype=np.float64)
    )

    ts, day_offsets, dates = _session_grid(n_days)
    return Panel(
        fields={
            "open": np.asarray(open64, dtype=np.float32),
            "high": np.asarray(high64, dtype=np.float32),
            "low": np.asarray(low64, dtype=np.float32),
            "close": np.asarray(close64, dtype=np.float32),
            "volume": np.asarray(volume64, dtype=np.float32),
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates,
    )


def row_at(panel: Panel, day: int, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    offset = hour * 60 + minute - (9 * 60 + 15)
    start = int(panel.day_offsets[day])
    end = int(panel.day_offsets[day + 1])
    if not 0 <= offset < end - start:
        raise ValueError("requested time is outside the synthetic session")
    return start + offset


def _flat_panel(
    n_days: int = 1,
    *,
    symbols=("AAA",),
    thin_by_day=None,
) -> Panel:
    ts, day_offsets, _ = _session_grid(n_days)
    n_rows = int(day_offsets[-1])
    close = np.full((n_rows, len(symbols)), PRICE, dtype=np.float64)
    volume = np.full_like(close, AMPLE_VOLUME, dtype=np.float64)

    for day, by_symbol in (thin_by_day or {}).items():
        for symbol_index, offsets in by_symbol.items():
            for offset in offsets:
                row = int(day_offsets[day]) + int(offset)
                volume[row, symbol_index] = THIN_VOLUME

    return Panel(
        fields={
            "open": np.asarray(close, dtype=np.float32),
            "high": np.asarray(close, dtype=np.float32),
            "low": np.asarray(close, dtype=np.float32),
            "close": np.asarray(close, dtype=np.float32),
            "volume": np.asarray(volume, dtype=np.float32),
        },
        symbols=tuple(symbols),
        ts=ts,
        day_offsets=day_offsets,
        dates=_session_grid(n_days)[2],
    )


class ConstantWeightStrategy(Strategy):
    class Params(BaseModel):
        weights: tuple[float, ...]
        decision_time: str = "10:00"

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=(self.params.decision_time,))

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        return TargetPortfolio(
            weights=np.asarray(self.params.weights, dtype=np.float64)
        )


class AlwaysTargetStrategy(Strategy):
    class Params(BaseModel):
        weights: tuple[float, ...]
        decision_times: tuple[str, ...] = ("10:00",)

    def __init__(self, params):
        super().__init__(params)
        self.decision_calls = 0

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=self.params.decision_times)

    def precompute(self, panel):
        return {}

    def on_session_start(self, session_date) -> None:
        self.decision_calls = 0

    def on_decision(self, view, signals, state):
        self.decision_calls += 1
        return TargetPortfolio(
            weights=np.asarray(self.params.weights, dtype=np.float64)
        )


class ChangingWeightStrategy(Strategy):
    class Params(BaseModel):
        first_weight: float
        second_weight: float
        decision_times: tuple[str, ...] = ("10:00", "15:18")

    def __init__(self, params):
        super().__init__(params)
        self._decision_calls = 0

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=self.params.decision_times)

    def precompute(self, panel):
        return {}

    def on_session_start(self, session_date) -> None:
        self._decision_calls = 0

    def on_decision(self, view, signals, state):
        weight = (
            self.params.first_weight
            if self._decision_calls == 0
            else self.params.second_weight
        )
        self._decision_calls += 1
        return TargetPortfolio(
            weights=np.asarray((weight,), dtype=np.float64)
        )


class StopTargetStrategy(Strategy):
    class Params(BaseModel):
        weights: tuple[float, ...]
        decision_time: str = "10:00"
        stop_price: float = 99.0

    def data_request(self) -> DataRequest:
        return DataRequest(
            decision_times=(self.params.decision_time,),
            needs_intrabar_risk=True,
        )

    def precompute(self, panel):
        return {}

    def on_decision(self, view, signals, state):
        return TargetPortfolio(
            weights=np.asarray(self.params.weights, dtype=np.float64),
            meta={"stop:AAA": float(self.params.stop_price)},
        )


def _run(
    strategy,
    panel,
    *,
    cost_model=None,
    fill_model=None,
    decision_latency_bars: int = 0,
    capital: float = CAPITAL,
):
    if cost_model is None:
        cost_model = NSEIntradayEquityCosts()
    if fill_model is None:
        fill_model = FillModel(slippage=ZeroSlippage())
    config = BacktestConfig(
        capital=capital,
        cost_model=cost_model,
        fill_model=fill_model,
        decision_latency_bars=decision_latency_bars,
    )
    return run_backtest(strategy, panel, config)


def _expected_target_shares(
    weight: float,
    *,
    capital: float = CAPITAL,
    price: float = PRICE,
) -> float:
    clipped_weight = float(np.clip(weight, -0.10, 0.10))
    return float(
        np.sign(clipped_weight)
        * np.floor(abs(clipped_weight) * capital / price)
    )


def _thin_fill_shares() -> float:
    return float(MAX_PARTICIPATION * THIN_VOLUME * PRICE / PRICE)


def _intent_rows(result, intent: OrderIntent):
    mask = result.trades["intent"].map(lambda value: value == intent)
    return result.trades.loc[mask].sort_values("ts").reset_index(drop=True)


def _assert_intents_are_enum_members(trades) -> None:
    # AMENDMENT 2 item 2 pins the blotter's `intent` column to the enum's `.value`
    # as a plain `str` (parquet-safe), not the enum member itself, so the isinstance
    # check this file's docstring flagged as a guess is re-pointed to value-equality
    # against the valid `OrderIntent` values.
    values = trades["intent"].tolist()
    assert len(values) == len(trades)
    valid_values = {member.value for member in OrderIntent}
    assert all(value in valid_values for value in values)


def _assert_flat(result, n_symbols: int) -> None:
    positions = np.asarray(result.positions, dtype=np.float64)
    np.testing.assert_allclose(
        positions[-1],
        np.zeros(n_symbols, dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def _forced_case(cost_model):
    panel = _flat_panel(
        thin_by_day={0: {0: tuple(range(366, 375))}},
    )
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(
            weights=(0.05,),
            decision_time="10:00",
        )
    )
    result = _run(
        strategy,
        panel,
        cost_model=cost_model,
        fill_model=FillModel(slippage=ZeroSlippage()),
    )
    return panel, result


def _forced_trade(result):
    forced = _intent_rows(result, OrderIntent.FORCED_EOD)
    assert len(forced) == 1
    return forced.iloc[0]


def _batch_from_trade(row) -> FillBatch:
    qty = float(row["qty"])
    price = float(row["price"])
    return FillBatch(
        notional=np.asarray([abs(qty) * price], dtype=np.float64),
        is_buy=np.asarray([bool(row["is_buy"])], dtype=bool),
    )


def test_item1_partial_square_off_retries_to_flat():
    panel = _flat_panel(thin_by_day={0: {0: (366,)}})
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    result = _run(strategy, panel)

    entry_shares = _expected_target_shares(0.05)
    thin_shares = _thin_fill_shares()
    eod = _intent_rows(result, OrderIntent.EOD_EXIT)

    assert len(eod) == 2
    assert result.n_trades == 3
    assert result.forced_eod_liquidation_days == 0
    np.testing.assert_allclose(
        np.asarray(eod["qty"], dtype=np.float64),
        np.asarray(
            (-thin_shares, -(entry_shares - thin_shares)),
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-10,
    )
    assert float(eod["qty"].sum()) == pytest.approx(-entry_shares, abs=1e-10)
    # LEAD DERIVATION (hand-traced, cross-checked against sibling test_item2's
    # identical retry mechanism): square_off_row_for_day[0] = row_at("15:20") = 365.
    # At t=365 the square-off block queues EOD_EXIT #1 for fill_row=366. Row 366 is
    # the ONLY thin row, so it partial-fills there (offset 366, "15:21") and leaves a
    # remainder. Because that fill removes the pending order from the queue before
    # the square-off block re-evaluates at t=366, `_eod_exit_pending()` is False
    # again, so the block re-arms and queues EOD_EXIT #2 for fill_row=367 -- row 367
    # is back to ample volume, so it fills there in full. The LAST EOD_EXIT therefore
    # lands two rows after the square-off row (365 + 2 = 367, "15:22"), exactly the
    # same "square_off_row + n_eod_exit_fills" pattern test_item2 asserts explicitly
    # (`row_at(panel, 0, "15:20") + 5` for its 5 fills). 366 ("15:21") is only the
    # FIRST (partial) fill, not the last.
    assert int(eod.iloc[-1]["ts"]) == int(panel.ts[row_at(panel, 0, "15:20") + 2])
    assert int(eod.iloc[-1]["ts"]) < int(panel.ts[-1])
    _assert_flat(result, 1)


def test_item2_repeated_partial_square_off():
    panel = _flat_panel(
        thin_by_day={0: {0: tuple(range(366, 370))}},
    )
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    result = _run(strategy, panel)

    entry_shares = _expected_target_shares(0.05)
    thin_shares = _thin_fill_shares()
    eod = _intent_rows(result, OrderIntent.EOD_EXIT)
    remaining_after_fill = entry_shares + np.cumsum(
        np.asarray(eod["qty"], dtype=np.float64)
    )

    assert len(eod) == 5
    assert result.n_trades == 6
    assert result.forced_eod_liquidation_days == 0
    np.testing.assert_allclose(
        np.asarray(eod["qty"], dtype=np.float64),
        np.asarray(
            (-thin_shares, -thin_shares, -thin_shares, -thin_shares, -3400.0),
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        remaining_after_fill,
        np.asarray((4600.0, 4200.0, 3800.0, 3400.0, 0.0)),
        rtol=0.0,
        atol=1e-10,
    )
    assert np.all(np.diff(remaining_after_fill) < 0.0)
    assert int(eod.iloc[-1]["ts"]) == int(panel.ts[row_at(panel, 0, "15:20") + 5])
    assert int(eod.iloc[-1]["ts"]) < int(panel.ts[-1])
    _assert_flat(result, 1)


def test_item3_square_off_that_cannot_finish_uses_terminal_safety_net():
    panel = _flat_panel(
        thin_by_day={0: {0: tuple(range(366, 375))}},
    )
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    result = _run(strategy, panel)

    entry_shares = _expected_target_shares(0.05)
    thin_shares = _thin_fill_shares()
    remaining_shares = entry_shares - 9.0 * thin_shares
    eod = _intent_rows(result, OrderIntent.EOD_EXIT)
    forced = _intent_rows(result, OrderIntent.FORCED_EOD)

    assert len(eod) == 9
    assert len(forced) == 1
    assert result.n_trades == 11
    assert result.forced_eod_liquidation_days == 1
    np.testing.assert_allclose(
        np.asarray(eod["qty"], dtype=np.float64),
        np.full(9, -thin_shares, dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )
    assert float(forced.iloc[0]["qty"]) == pytest.approx(
        -remaining_shares,
        abs=1e-10,
    )
    assert float(forced.iloc[0]["price"]) == pytest.approx(PRICE, abs=1e-10)
    assert int(forced.iloc[0]["ts"]) == int(panel.ts[-1])
    _assert_flat(result, 1)


def test_item4_no_reopening_after_square_off():
    panel = _flat_panel()
    strategy = AlwaysTargetStrategy(
        params=AlwaysTargetStrategy.Params(
            weights=(0.05,),
            decision_times=("10:00", "15:21"),
        )
    )
    result = _run(strategy, panel)

    entries = _intent_rows(result, OrderIntent.ENTRY)
    eod = _intent_rows(result, OrderIntent.EOD_EXIT)
    late_entries = entries.loc[
        entries["ts"] >= int(panel.ts[row_at(panel, 0, "15:20")])
    ]

    assert strategy.decision_calls == 1
    assert len(entries) == 1
    assert len(late_entries) == 0
    assert len(eod) == 1
    assert result.n_trades == 2
    assert result.forced_eod_liquidation_days == 0
    assert int(entries.iloc[0]["ts"]) == int(panel.ts[row_at(panel, 0, "10:01")])
    _assert_flat(result, 1)


def test_item5a_in_flight_guard_holds_on_normal_run():
    panel = _flat_panel(
        n_days=2,
        symbols=("AAA", "BBB"),
        thin_by_day={0: {0: tuple(range(366, 375))}},
    )
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(
            weights=(0.05, 0.04),
        )
    )

    # run_backtest enables FULL strictness and checks the in-flight/pending-order
    # invariant internally on every row; a successful return is the positive control.
    result = _run(strategy, panel)

    assert result.n_trades == 17
    assert len(result.trades) == 17
    assert result.forced_eod_liquidation_days == 1
    assert len(_intent_rows(result, OrderIntent.ENTRY)) == 4
    assert len(_intent_rows(result, OrderIntent.EOD_EXIT)) == 12
    assert len(_intent_rows(result, OrderIntent.FORCED_EOD)) == 1
    _assert_flat(result, 2)


def test_item5b_in_flight_guard_fires_on_injected_desync(monkeypatch):
    panel = _flat_panel()
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    real_pending_order = PendingOrder

    def corrupt_pending_order(*args, **kwargs):
        order = real_pending_order(*args, **kwargs)
        corrupted_shares = (
            np.asarray(order.shares, dtype=np.float64).copy() * 2.0
        )
        object.__setattr__(order, "shares", corrupted_shares)
        return order

    engine_module = importlib.import_module("nifty_quant.backtest.engine")
    orders_module = importlib.import_module("nifty_quant.backtest.orders")

    patched_names = []
    for module in (engine_module, orders_module):
        if hasattr(module, "PendingOrder"):
            monkeypatch.setattr(
                module,
                "PendingOrder",
                corrupt_pending_order,
            )
            patched_names.append(f"{module.__name__}.PendingOrder")

    # This negative control intentionally depends on PendingOrder being resolved
    # inside engine.py under one of the two exact names patched above.  A third
    # alias would make the injection inert, which is a known limitation of the
    # amendment's mandated best-effort technique.
    assert patched_names == [
        "nifty_quant.backtest.engine.PendingOrder",
        "nifty_quant.backtest.orders.PendingOrder",
    ] or patched_names == ["nifty_quant.backtest.engine.PendingOrder"] or patched_names == [
        "nifty_quant.backtest.orders.PendingOrder"
    ]

    with pytest.raises(guards.ContractViolation):
        _run(strategy, panel)


def test_item6_append_not_overwrite_nets_entry_and_eod_exit():
    panel = _flat_panel()
    strategy = ChangingWeightStrategy(
        params=ChangingWeightStrategy.Params(
            first_weight=0.05,
            second_weight=0.08,
            decision_times=("10:00", "15:18"),
        )
    )
    result = _run(
        strategy,
        panel,
        decision_latency_bars=2,
    )

    collision_row = row_at(panel, 0, "15:21")
    collision = result.trades.loc[
        (result.trades["ts"] == int(panel.ts[collision_row]))
        & (result.trades["symbol"] == "AAA")
    ].reset_index(drop=True)

    assert len(collision) == 1
    assert float(collision.iloc[0]["qty"]) == pytest.approx(-2000.0, abs=1e-10)
    assert float(collision.iloc[0]["notional"]) == pytest.approx(
        200_000.0,
        abs=1e-8,
    )
    assert not bool(collision.iloc[0]["is_buy"])
    assert collision.iloc[0]["intent"] == OrderIntent.EOD_EXIT

    # The +3000 ENTRY and -5000 EOD_EXIT are summed into one -2000 fill.
    # The single collision row consequently carries only the largest
    # contribution's intent, EOD_EXIT; both intents occur in the full blotter.
    assert len(_intent_rows(result, OrderIntent.ENTRY)) == 1
    assert len(_intent_rows(result, OrderIntent.EOD_EXIT)) == 2
    assert result.n_trades == 3
    assert result.forced_eod_liquidation_days == 0
    _assert_flat(result, 1)


def test_item7_intent_tagging_for_risk_and_normal_eod_paths():
    close = np.full((BARS_PER_DAY, 1), PRICE, dtype=np.float64)
    low = close.copy()
    _, day_offsets, _ = _session_grid(1)
    low[int(day_offsets[0]) + 50, 0] = 98.0
    risk_panel = make_panel(
        close,
        1,
        symbols=("AAA",),
        low=low,
        volume=np.full_like(close, AMPLE_VOLUME, dtype=np.float64),
    )
    risk_strategy = StopTargetStrategy(
        params=StopTargetStrategy.Params(
            weights=(0.05,),
            stop_price=99.0,
        )
    )
    risk_result = _run(risk_strategy, risk_panel)
    _assert_intents_are_enum_members(risk_result.trades)

    normal_panel = _flat_panel()
    normal_strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    normal_result = _run(normal_strategy, normal_panel)
    _assert_intents_are_enum_members(normal_result.trades)

    assert risk_result.n_trades == 2
    assert len(_intent_rows(risk_result, OrderIntent.ENTRY)) == 1
    assert len(_intent_rows(risk_result, OrderIntent.RISK_EXIT)) == 1
    assert len(_intent_rows(risk_result, OrderIntent.EOD_EXIT)) == 0
    assert len(_intent_rows(risk_result, OrderIntent.FORCED_EOD)) == 0
    assert risk_result.forced_eod_liquidation_days == 0
    assert float(
        _intent_rows(risk_result, OrderIntent.RISK_EXIT).iloc[0]["qty"]
    ) == pytest.approx(-_expected_target_shares(0.05), abs=1e-10)

    assert normal_result.n_trades == 2
    assert len(_intent_rows(normal_result, OrderIntent.ENTRY)) == 1
    assert len(_intent_rows(normal_result, OrderIntent.EOD_EXIT)) == 1
    assert len(_intent_rows(normal_result, OrderIntent.FORCED_EOD)) == 0
    assert normal_result.forced_eod_liquidation_days == 0
    _assert_flat(normal_result, 1)


def test_item8_composite_cost_model_direct():
    fills = FillBatch(
        notional=np.asarray((10_000.0, 25_000.0), dtype=np.float64),
        is_buy=np.asarray((True, False), dtype=bool),
    )
    intraday = NSEIntradayEquityCosts()
    delivery = NSEDeliveryEquityCosts()
    composite = CompositeCostModel(models=(intraday, delivery))

    actual = composite.charges(fills)
    expected_intraday = intraday.charges(fills)
    expected_delivery = delivery.charges(fills)

    for field in CHARGE_FIELDS:
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected_intraday, field)
            + getattr(expected_delivery, field),
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        actual.total,
        expected_intraday.total + expected_delivery.total,
        rtol=0.0,
        atol=1e-12,
    )


def test_item8_forced_eod_composed_costs_default_cost_model():
    cost_model = NSEIntradayEquityCosts()
    panel, result = _forced_case(cost_model)
    forced = _forced_trade(result)
    batch = _batch_from_trade(forced)

    base_charges = cost_model.charges(batch)
    delivery_charges = NSEDeliveryEquityCosts().charges(batch)
    composed_charges = CompositeCostModel(
        models=(cost_model, NSEDeliveryEquityCosts())
    ).charges(batch)
    expected_total = (
        base_charges.total[0] + delivery_charges.total[0]
    )

    remaining_shares = _expected_target_shares(0.05) - 9.0 * _thin_fill_shares()
    assert result.n_trades == 11
    assert result.forced_eod_liquidation_days == 1
    assert float(forced["qty"]) == pytest.approx(-remaining_shares, abs=1e-10)
    assert float(forced["notional"]) == pytest.approx(
        remaining_shares * PRICE,
        abs=1e-8,
    )
    assert float(forced["charges"]) == pytest.approx(
        expected_total,
        abs=1e-8,
    )

    for field in CHARGE_FIELDS:
        np.testing.assert_allclose(
            getattr(composed_charges, field),
            getattr(base_charges, field)
            + getattr(delivery_charges, field),
            rtol=0.0,
            atol=1e-12,
        )
    assert float(base_charges.brokerage[0]) > 0.0
    assert float(base_charges.gst[0]) > 0.0
    assert float(composed_charges.brokerage[0]) > 0.0
    assert float(composed_charges.gst[0]) > 0.0
    _assert_flat(result, 1)


def test_item8_forced_eod_composed_costs_zero_cost_model():
    cost_model = ZeroCost()
    panel, result = _forced_case(cost_model)
    forced = _forced_trade(result)
    batch = _batch_from_trade(forced)

    base_charges = cost_model.charges(batch)
    delivery_charges = NSEDeliveryEquityCosts().charges(batch)
    composed_charges = CompositeCostModel(
        models=(cost_model, NSEDeliveryEquityCosts())
    ).charges(batch)
    expected_total = delivery_charges.total[0]

    assert result.n_trades == 11
    assert result.forced_eod_liquidation_days == 1
    assert float(forced["charges"]) == pytest.approx(
        expected_total,
        abs=1e-8,
    )
    np.testing.assert_allclose(
        base_charges.total,
        np.zeros(1, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        composed_charges.brokerage,
        np.zeros(1, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        composed_charges.gst,
        np.zeros(1, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        composed_charges.total,
        delivery_charges.total,
        rtol=0.0,
        atol=1e-12,
    )
    _assert_flat(result, 1)


def test_item9_ordinary_path_matches_self_derived_regression():
    panel = _flat_panel()
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )
    cost_model = NSEIntradayEquityCosts()
    fill_model = FillModel(slippage=ZeroSlippage())
    result = _run(
        strategy,
        panel,
        cost_model=cost_model,
        fill_model=fill_model,
        capital=CAPITAL,
    )

    shares = _expected_target_shares(0.05)
    entry_batch = FillBatch(
        notional=np.asarray([shares * PRICE], dtype=np.float64),
        is_buy=np.asarray([True], dtype=bool),
    )
    exit_batch = FillBatch(
        notional=np.asarray([shares * PRICE], dtype=np.float64),
        is_buy=np.asarray([False], dtype=bool),
    )
    entry_charges = cost_model.charges(entry_batch).total[0]
    exit_charges = cost_model.charges(exit_batch).total[0]
    expected_final_equity = CAPITAL - entry_charges - exit_charges
    expected_equity_curve = np.asarray(
        (CAPITAL, expected_final_equity),
        dtype=np.float64,
    )

    assert result.n_trades == 2
    assert len(result.trades) == 2
    assert result.forced_eod_liquidation_days == 0
    assert len(_intent_rows(result, OrderIntent.ENTRY)) == 1
    assert len(_intent_rows(result, OrderIntent.EOD_EXIT)) == 1
    assert len(_intent_rows(result, OrderIntent.FORCED_EOD)) == 0
    np.testing.assert_allclose(
        np.asarray(result.equity_curve, dtype=np.float64),
        expected_equity_curve,
        rtol=0.0,
        atol=1e-6,
    )
    assert result.total_costs == pytest.approx(
        entry_charges + exit_charges,
        abs=1e-8,
    )
    _assert_flat(result, 1)


def test_item10_accounting_invariant_holds_including_retry_rows():
    panel = _flat_panel(
        thin_by_day={0: {0: tuple(range(366, 370))}},
    )
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.05,))
    )

    # The engine's unconditional accounting guard runs on every row, including
    # rows on which a partial EOD retry is filled.  Returning normally therefore
    # verifies the invariant throughout; the aggregate checks below verify the
    # resulting accounting independently at the end.
    result = _run(strategy, panel)

    eod = _intent_rows(result, OrderIntent.EOD_EXIT)
    trade_costs = float(
        np.asarray(result.trades["charges"], dtype=np.float64).sum()
    )
    expected_final_equity = CAPITAL - trade_costs

    assert result.n_trades == 6
    assert len(eod) == 5
    assert result.forced_eod_liquidation_days == 0
    assert result.total_costs == pytest.approx(trade_costs, abs=1e-8)
    assert result.equity_curve[-1] == pytest.approx(
        expected_final_equity,
        abs=1e-6,
    )
    _assert_flat(result, 1)


def test_item11a_decision_that_cannot_fill_within_its_session_is_dropped_and_counted():
    panel = _flat_panel(n_days=2)
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.10,), decision_time="14:00")
    )

    result = _run(strategy, panel, decision_latency_bars=100)

    aaa_positive_qty = result.trades[
        (result.trades["symbol"] == "AAA") & (result.trades["qty"] > 0)
    ]
    next_session_trades = result.trades[
        result.trades["ts"] >= panel.ts[panel.day_offsets[1]]
    ]

    assert len(aaa_positive_qty) == 0
    assert result.n_trades == 0
    assert len(next_session_trades) == 0
    assert result.forced_eod_liquidation_days == 0
    _assert_flat(result, 1)
    # AMENDMENT 2 item 3: a session-end drop is NOT a rejection (the market did not
    # refuse the order; the engine cancelled it to avoid an overnight leak), and
    # `rejected_order_rate` explicitly must not fold in drops. Re-pointed to the
    # dedicated `n_orders_dropped_at_session_end` counter. `_flat_panel(n_days=2)`
    # decides once per day at "14:00" (decision_time is a daily wall-clock trigger,
    # not a one-shot), and `decision_latency_bars=100` pushes each of the two
    # daily decisions' fill row past its own session end, so both are dropped.
    assert result.n_orders_dropped_at_session_end == 2


def test_item11b_control_same_decision_fills_normally_within_session():
    panel = _flat_panel(n_days=2)
    strategy = ConstantWeightStrategy(
        params=ConstantWeightStrategy.Params(weights=(0.10,), decision_time="14:00")
    )

    result = _run(strategy, panel, decision_latency_bars=0)

    entry_rows = _intent_rows(result, OrderIntent.ENTRY)

    assert result.n_trades >= 1
    assert len(entry_rows) >= 1
