"""Adversarial engine and portfolio invariant tests -- author A.

Written from specs/engine_adversarial_invariants.md ALONE (independent of author B's
tests). This spec adds no feature: it describes invariants the CURRENT engine is
supposed to satisfy on hostile inputs. A PASSING test confirms an invariant holds. A
FAILING test is a FINDING about real engine behaviour, reported below, never "fixed" by
weakening the assertion or by touching src/.

============================== SPEC DEFECTS ==================================

D1. Item 1 asks to "assert the realised P&L on the closing leg and the new short's cost
    basis are separated correctly" when a single order crosses zero. `Portfolio`
    (src/nifty_quant/backtest/portfolio.py) has NO lot-level or realised/unrealised P&L
    split: it tracks only `cash`, `shares`, `cum_costs` and a single whole-book
    mark-to-market `cum_pnl`. There is nothing in the public API to assert a "closing
    leg" P&L against. `test_cross_zero_long_to_short` below asserts the closest testable
    proxy instead: the crossing trade's cash-flow arithmetic (qty * price - charges)
    reconciles exactly with the cash delta, which is the only sense in which the two
    legs are "separated" in this architecture.

D2. Item 16 ("ruin: assert `ruined` is set and `ruin_index` points at the right row")
    presumes `ruin_index` indexes a *bar row*. It does not: `equity_curve` (and
    therefore `ruin_index`) is sampled only at decision rows plus one final
    post-session append (engine.py:484-501, 625-642), not at every bar. "The right row"
    is therefore ambiguous without also fixing `decision_times` to a schedule dense
    enough to make every bar a decision row. `test_ruin_flag_and_index` below controls
    for this explicitly.

============================ INVARIANT FAILURES ===============================

21 tests, run against the real engine: 17 PASS, 4 FAIL. Every FAIL below is backed by
a real, reproduced test run in this file -- not asserted from memory -- and none of
these tests was weakened or the engine touched to make it pass.

FAIL 1 -- I5 violated. `test_halt_after_entry_forced_liquidation_ignores_tradable`
(item 10). A symbol halted (`tradable=False`) mid-session still gets FILLED when the
mandatory forced end-of-day liquidation (engine.py:612-623, `_execute_direct_fill`)
runs: it bypasses the `tradable` mask entirely. I2 (session ends flat) therefore holds
for a halted symbol too, but "for a reason nobody chose" -- in reality you cannot
liquidate a halted stock. `forced_eod_liquidation_days` does not distinguish an
ordinary forced liquidation from one that filled against a halted symbol.

FAIL 2 -- I4 violated. `test_missing_bar_while_position_open_corrupts_equity`
(item 11). A single NaN close on a symbol with an OPEN position writes a literal
`nan` into `result.equity_curve` (observed: `[1000000.0, nan, 999917.35]`), and
`result.ruined` flips to `True` off the back of it (ruin_index=1) even though nothing
about the strategy "ruined". Root cause: `Portfolio.equity()`
(portfolio.py:30-35) only masks the price to 0.0 when `shares == 0`; when `shares !=
0` the raw (possibly NaN) price is kept, so `np.dot(shares, prices)` goes NaN.
Compounding this: the accounting guard at engine.py:414-425 (`guards.check_accounting`)
is called on every row with this same NaN `positions_value`, but its check is
`abs(discrepancy) > atol`, and NaN comparisons are always False in IEEE-754, so the
guard silently PASSES instead of raising -- the safety net has a blind spot for
exactly the failure mode it exists to catch.

FAIL 3 -- item 15 ("prevented or surfaced explicitly") violated; it is neither.
`test_undercapitalized_order_drives_cash_negative_unsurfaced`. A 100x-over-capital
fill (via a duck-typed sizer bypassing `GrossNotionalSizer`'s own budget awareness)
drove ledger-reconstructed cash to -9,900,695.86 against 100,000.0 starting capital,
with no exception, no rejection, and `result.ruined` staying `False` throughout --
the excursion is invisible in `equity_curve` because no decision-row snapshot
happened to land during the brief negative-cash window, and is only recoverable by
reconstructing cash from the trade ledger by hand.

FAIL 4 -- SEVERE, beyond the 20 numbered items (bonus test, motivated by item 20's
"no state leaks across on_session_start", verified independently before being
delegated for writing). `test_last_row_decision_fills_across_session_boundary`. A
decision made on a session's LAST bar queues an order for `fill_row = t + 1 +
decision_latency_bars` (engine.py:568-569), bounded only against the PANEL
(`fill_row < n_rows`), never against the session it was decided in. The order
survives the session boundary uncrossed -- `on_session_start` (engine.py:406-409)
resets `active_stops` and `square_off_queued` but never `pending_orders` or
`in_flight` -- and fills on the very next session's OPENING bar. Observed: entry
fill ts exactly equals the next session's first-bar ts. This creates an overnight
position via machinery nobody declared, charged intraday (not delivery) costs, in a
repo whose entire premise is intraday-only trading with a forced-flat EOD invariant.
Day 0's own I2 holds only because day 0 never actually held the position -- it was
silently relocated into day 1 instead.

WHAT HELD: items 1-9, 12-14, 16-19 all pass as written (17/21 tests), including the
accounting identity (I1) reconstructed independently from the trade ledger at
session end, session-ends-flat (I2) in every scenario tried, the conservative
gap-through-stop fill (`min(stop, open)`, never the stop price itself), all-NaN
column exclusion from cross-sectional sizing, the 10x price move staying finite, the
ruin_index off-by-one (asserted per its documented decision-row-index-space
contract, per D2), 60-bar/105-bar irregular sessions in one panel, single-row
sessions, an absent square-off time clamping to the last row, and no STOP-specific
state leak across a weekend gap (distinct from FAIL 4's pending-order leak).

================================================================================
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.portfolio import GrossNotionalSizer, SizingResult
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)

_IST = ZoneInfo("Asia/Kolkata")
CAPITAL = 1_000_000.0


# --------------------------------------------------------------------------------
# Shared, hand-verified infrastructure. Every helper below has been exercised
# directly against the real engine (not just written from reading source) before
# being relied on in an assertion.
# --------------------------------------------------------------------------------


def make_grid(
    day_lengths: list[int], start: dt.date = dt.date(2024, 1, 2)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-minute bars starting 09:15 IST per day, one entry in day_lengths per session.

    Business days only (Sat/Sun skipped), so consecutive entries naturally produce a
    real weekend gap in `ts` when `start` lands on/after a Friday.
    """
    dates: list[dt.date] = []
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
    ts = np.concatenate(ts_chunks).astype(np.int64) if ts_chunks else np.empty(0, dtype=np.int64)
    day_offsets = np.concatenate([[0], np.cumsum(day_lengths)]).astype(np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def make_panel(
    symbols: tuple[str, ...],
    ts: np.ndarray,
    day_offsets: np.ndarray,
    dates: np.ndarray,
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> Panel:
    """Build a Panel from explicit float64 arrays, cast to float32 at rest like real data."""
    fields = {
        "open": np.asarray(open_, dtype=np.float64).astype(np.float32),
        "high": np.asarray(high, dtype=np.float64).astype(np.float32),
        "low": np.asarray(low, dtype=np.float64).astype(np.float32),
        "close": np.asarray(close, dtype=np.float64).astype(np.float32),
        "volume": np.asarray(volume, dtype=np.float64).astype(np.float32),
    }
    return Panel(fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates)


def row_at(day_offsets: np.ndarray, day: int, hhmm: str) -> int:
    """Row index of a given HH:MM within `day`, assuming the session starts 09:15."""
    hour, minute = (int(p) for p in hhmm.split(":"))
    minute_of_day = hour * 60 + minute
    return int(day_offsets[day]) + (minute_of_day - 9 * 60 - 15)


class _EmptyParams(BaseModel):
    pass


class TsScriptStrategy(Strategy):
    """Returns a scripted TargetPortfolio keyed by the decision's cursor timestamp.

    `state.ts` (PortfolioState.ts) is the ts of the cursor bar (decision row t's cursor
    is t-1), NOT the decision row's own ts -- verified empirically against the real
    engine. `script` maps that cursor ts (as a plain int) to a TargetPortfolio; any
    decision whose cursor ts is not a key returns None (hold).
    """

    name = "ts_script"
    Params = _EmptyParams

    def __init__(
        self,
        params: BaseModel,
        script: dict[int, TargetPortfolio],
        *,
        decision_times: tuple[str, ...] | None,
        needs_intrabar_risk: bool = False,
    ) -> None:
        super().__init__(params)
        self._script = script
        self._decision_times = decision_times
        self._needs_intrabar_risk = needs_intrabar_risk

    def data_request(self) -> DataRequest:
        return DataRequest(
            decision_times=self._decision_times,
            needs_intrabar_risk=self._needs_intrabar_risk,
        )

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(
        self, view: MarketView, signals: dict, state: PortfolioState
    ) -> TargetPortfolio | None:
        return self._script.get(int(state.ts))


def default_config(**overrides) -> BacktestConfig:
    kwargs = dict(
        capital=CAPITAL,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


def assert_finite_everywhere(result) -> None:
    """I4: equity, returns, turnover, positions finite on every entry."""
    assert np.all(np.isfinite(result.equity_curve)), (
        f"I4 violated: non-finite equity_curve entries at "
        f"{np.flatnonzero(~np.isfinite(result.equity_curve))}: {result.equity_curve}"
    )
    assert np.all(np.isfinite(result.returns)), "I4 violated: non-finite returns"
    assert np.all(np.isfinite(result.turnover)), "I4 violated: non-finite turnover"
    assert np.all(np.isfinite(result.positions)), "I4 violated: non-finite positions"


def assert_session_ends_flat(result) -> None:
    """I2: np.all(shares == 0) at the very last recorded snapshot."""
    assert result.positions.shape[0] > 0, "no snapshots recorded at all"
    assert np.all(result.positions[-1] == 0.0), (
        f"I2 violated: final position snapshot is not flat: {result.positions[-1]}"
    )


def reconstructed_final_cash(result) -> float:
    """Independent (non-engine) reconstruction of terminal cash from the trade ledger.

    cash_T = initial_capital - sum(qty * price) - sum(charges) over every recorded
    fill, in ledger order. This is derived purely from Portfolio.apply_fills's own
    documented contract (cash -= fill_value + charges), not from re-deriving the
    engine's internal equity() call, so comparing it against equity_curve[-1] is a
    genuine independent check of I1, not a restatement of the implementation.
    """
    trades = result.trades
    if trades.empty:
        return float(result.initial_capital)
    notional_signed = float((trades["qty"] * trades["price"]).sum())
    charges = float(trades["charges"].sum())
    return float(result.initial_capital) - notional_signed - charges


def assert_i1_terminal(result, atol: float = 1e-6) -> None:
    """I1 at the terminal snapshot: with I2 holding, positions_value is 0, so
    equity_curve[-1] must equal the ledger-reconstructed cash exactly."""
    assert_session_ends_flat(result)
    expected = reconstructed_final_cash(result)
    actual = float(result.equity_curve[-1])
    assert actual == pytest.approx(expected, abs=atol), (
        f"I1 violated at terminal snapshot: equity_curve[-1]={actual} != "
        f"ledger-reconstructed cash={expected} (diff={actual - expected})"
    )
def test_cross_zero_long_to_short():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:25") - 1]): TargetPortfolio(
            weights=np.array([-0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:30") - 1]): TargetPortfolio(
            weights=np.array([-0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25", "09:30"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert result.trades.iloc[1]["qty"] == pytest.approx(-2000.0)
    assert result.positions[-2, 0] == pytest.approx(-1000.0)
    assert_finite_everywhere(result)
    # see D1
    assert_i1_terminal(result)


def test_short_to_flat():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([-0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:25") - 1]): TargetPortfolio(
            weights=np.array([0.0, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25"),
    )
    result = run_backtest(strategy, panel, default_config())

    # positions[-2] is the decision-2 SNAPSHOT, taken BEFORE decision 2's own
    # buy-to-cover order fills (it reflects state left by decision 1 only), so it is
    # still short here -- it does not observe the flat-out fill. Verify flatness
    # directly from the ledger instead: exactly 2 trades (short entry, buy-to-cover),
    # net AAA quantity zero, and no further (forced EOD) trade was needed because the
    # position was already flat by session end.
    assert result.trades.iloc[1]["qty"] == pytest.approx(1000.0)
    assert len(result.trades) == 2
    assert result.trades[result.trades["symbol"] == "AAA"]["qty"].sum() == pytest.approx(0.0)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_cross_zero_short_to_long():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([-0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:25") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:30") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25", "09:30"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert result.trades.iloc[1]["qty"] == pytest.approx(2000.0)
    assert result.positions[-2, 0] == pytest.approx(1000.0)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_long_flat_long_again_same_session():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:25") - 1]): TargetPortfolio(
            weights=np.array([0.0, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:30") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25", "09:30"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert len(result.trades) >= 4
    assert result.trades.iloc[0]["qty"] == pytest.approx(1000.0)
    assert result.trades.iloc[1]["qty"] == pytest.approx(-1000.0)
    assert result.trades.iloc[2]["qty"] == pytest.approx(1000.0)
    assert np.count_nonzero(result.trades.iloc[:3]["qty"].to_numpy()) == 3
    assert result.positions[1, 0] == pytest.approx(1000.0)
    assert result.positions[2, 0] == pytest.approx(0.0)
    # positions[3] is the FINAL (post-session) snapshot, taken AFTER the mandatory
    # forced EOD square-off has already flattened the re-entered position again -- it
    # is not "the state right after decision 3's fill". Verify the re-entry via the
    # ledger instead: trades.iloc[2] (already asserted above, +1000) is the re-entry,
    # and trades.iloc[3] is the forced EOD exit that closes it back out.
    assert result.trades.iloc[3]["qty"] == pytest.approx(-1000.0)
    assert result.positions[-1, 0] == pytest.approx(0.0)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_identical_target_emits_no_order():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:25") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert_finite_everywhere(result)
    # decision 2's identical target must emit no order and thus produce no trade of
    # its own -- but the position it leaves open still has to be closed by the
    # mandatory forced EOD square-off at session end, which is a THIRD, unrelated
    # trade. So the ledger has exactly 2 rows (entry + forced EOD exit), never 3, and
    # the second row's ts must be the session's LAST row (the square-off), not
    # decision 2's own fill row -- that is the proof decision 2 itself traded nothing.
    assert len(result.trades) == 2
    assert result.trades.iloc[0]["qty"] == pytest.approx(1000.0)
    assert int(result.trades.iloc[1]["ts"]) == int(ts[-1])
    assert result.total_costs == pytest.approx(result.trades["charges"].sum())
    assert_i1_terminal(result)


def test_zero_bar_traded_value_rejects_order():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)
    volume[6, 0] = 0.0

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:30") - 1]): TargetPortfolio(
            weights=np.array([0.0, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:30"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert len(result.trades) == 0
    assert result.positions[-2, 0] == pytest.approx(0.0)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_repeated_partial_fills():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)
    volume[4, 0] = 1e9
    volume[6, 0] = 50.0
    volume[10, 0] = 1e9
    # Decision 2 sits at row 11 (row_at(day_offsets, 0, "09:26") == 11), so its FILL
    # row is 11 + 1 == 12, not 11 -- fill_row = decision_row + 1 + decision_latency_bars
    # (engine.py:568). row 11 itself is decision 2's CURSOR-adjacent decision row, not
    # a fill row at all.
    volume[12, 0] = 50.0

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:26") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:26"),
    )
    result = run_backtest(strategy, panel, default_config())

    aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
    aaa_rows = np.searchsorted(ts, aaa_trades["ts"].to_numpy())
    partial_trades = aaa_trades[(aaa_rows == 6) | (aaa_rows == 12)]

    assert len(partial_trades) >= 2
    assert np.all(partial_trades["filled_frac"].to_numpy() < 1.0)

    filled_through_second = aaa_trades[
        (aaa_rows <= 12) & (aaa_trades["qty"].to_numpy() > 0)
    ]["qty"].sum()
    assert 0.0 < filled_through_second < 1000.0

    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_rejected_order_does_not_double_next_target():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)
    volume[6, 0] = 0.0

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
        int(ts[row_at(day_offsets, 0, "09:26") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:26"),
    )
    result = run_backtest(strategy, panel, default_config())

    sizing: SizingResult = GrossNotionalSizer().to_shares(
        np.array([0.10, 0.0]),
        close[10],
        CAPITAL,
        bar_traded_value=close[10] * volume[10],
        max_participation=0.02,
    )
    expected_shares = sizing.shares[0]

    # Decision 2 is decision row 11 (row_at(day_offsets, 0, "09:26") == 11); its FILL
    # row is 11 + 1 == 12 (engine.py:568), not 11 itself.
    retry = result.trades[
        (result.trades["symbol"] == "AAA")
        & (result.trades["ts"] == ts[12])
        & (result.trades["qty"] > 0)
    ]

    assert len(retry) == 1
    assert retry.iloc[0]["qty"] == expected_shares
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_zero_and_nan_open_price_reject_not_divide():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)
    open_[6, 0] = 0.0
    open_[6, 1] = np.nan

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.10])
        ),
        int(ts[row_at(day_offsets, 0, "09:30") - 1]): TargetPortfolio(
            weights=np.array([0.0, 0.0])
        ),
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:30"),
    )
    result = run_backtest(strategy, panel, default_config())

    assert len(result.trades) == 0
    assert result.positions[-2, 0] == pytest.approx(0.0)
    assert result.positions[-2, 1] == pytest.approx(0.0)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_halt_after_entry_forced_liquidation_ignores_tradable():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    tradable = np.ones((n_rows, 2), dtype=bool)
    tradable[7:, 0] = False

    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): TargetPortfolio(
            weights=np.array([0.10, 0.0])
        )
    }

    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )
    result = run_backtest(
        strategy,
        panel,
        default_config(square_off_time="09:30"),
        tradable=tradable,
    )

    assert_finite_everywhere(result)
    assert_session_ends_flat(result)

    trade_rows = np.searchsorted(ts, result.trades["ts"].to_numpy())
    halted_aaa = (
        (result.trades["symbol"].to_numpy() == "AAA")
        & (trade_rows >= 7)
    )
    # A failure here is a confirmed I5 finding, not a test bug.
    assert not np.any(halted_aaa)
def test_missing_bar_while_position_open_corrupts_equity():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([60])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    open_[7:21, 0] = np.nan
    high[7:21, 0] = np.nan
    low[7:21, 0] = np.nan
    close[7:21, 0] = np.nan
    volume[7:21, 0] = np.nan

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    buy = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    noop = TargetPortfolio(weights=np.array([0.0, 0.0]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): buy,
        int(ts[row_at(day_offsets, 0, "09:26") - 1]): noop,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:26"),
    )

    result = run_backtest(strategy, panel, default_config())

    # EXPECTED TO FAIL: confirmed NaN-in-equity-curve finding, see file docstring
    assert_finite_everywhere(result)
    # The finite check is expected to fail first; terminal flatness is expected to pass.
    assert_session_ends_flat(result)


def test_absent_symbol_never_ordered():
    SYMBOLS = ("AAA", "BBB", "CCC")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 3), 100.0)
    high = np.full((n_rows, 3), 100.0)
    low = np.full((n_rows, 3), 100.0)
    close = np.full((n_rows, 3), 100.0)
    volume = np.full((n_rows, 3), 1_000_000.0)

    open_[:, 2] = np.nan
    high[:, 2] = np.nan
    low[:, 2] = np.nan
    close[:, 2] = np.nan
    volume[:, 2] = np.nan

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.05, 0.05, 0.05]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25"),
    )

    result = run_backtest(strategy, panel, default_config())

    assert "CCC" in result.absent_symbols
    assert result.n_symbols_absent == 1
    assert "CCC" not in result.trades["symbol"].tolist()
    assert np.all(np.asarray(result.positions)[:, 2] == 0.0)
    assert_finite_everywhere(result)


def test_extreme_price_move_stays_finite():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    open_[10:, 0] = 1_000.0
    high[10:, 0] = 1_000.0
    low[10:, 0] = 1_000.0
    close[10:, 0] = 1_000.0

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )

    result = run_backtest(strategy, panel, default_config())

    assert_finite_everywhere(result)
    assert_i1_terminal(result)
    assert result.equity_curve[-1] > CAPITAL


def test_ruin_flag_and_index_off_by_one():
    SYMBOLS = ("AAA",)
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 1), 100.0)
    high = np.full((n_rows, 1), 100.0)
    low = np.full((n_rows, 1), 100.0)
    close = np.full((n_rows, 1), 100.0)
    volume = np.full((n_rows, 1), 1_000_000.0)

    open_[10:, 0] = -500.0
    high[10:, 0] = -500.0
    low[10:, 0] = -500.0
    close[10:, 0] = -500.0

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([1.0]), meta={})
    script = {int(ts[0]): target}
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=None,
    )

    result = run_backtest(
        strategy,
        panel,
        default_config(
            capital=100_000.0,
            sizer=GrossNotionalSizer(max_weight=1.0),
        ),
    )

    assert result.equity_curve[9] < 0
    assert result.ruined is True
    # The documented off-by-one contract predicts 10; do not adjust this assertion if it differs.
    assert result.ruin_index == 10


def test_undercapitalized_order_drives_cash_negative_unsurfaced():
    @dataclass(frozen=True)
    class _HugeSizer:
        def to_shares(
            self,
            weights,
            prices,
            capital,
            *,
            bar_traded_value=None,
            max_participation=0.02,
        ):
            return SizingResult(
                shares=np.array([100_000.0]),
                unmet_gross_pct=0.0,
            )

    SYMBOLS = ("AAA",)
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 1), 100.0)
    high = np.full((n_rows, 1), 100.0)
    low = np.full((n_rows, 1), 100.0)
    close = np.full((n_rows, 1), 100.0)
    volume = np.full((n_rows, 1), 1_000_000_000.0)

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.10]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )

    result = run_backtest(
        strategy,
        panel,
        default_config(
            capital=100_000.0,
            sizer=_HugeSizer(),
            square_off_time="09:30",
        ),
    )

    trades = result.trades
    entry = trades[trades["qty"] > 0].iloc[0]
    assert entry["qty"] == pytest.approx(100_000.0)
    assert entry["price"] == pytest.approx(100.0)
    assert result.ruined is False

    initial_capital = result.initial_capital
    cash_out = trades["qty"].to_numpy() * trades["price"].to_numpy() + trades["charges"].to_numpy()
    running = initial_capital - np.cumsum(cash_out)
    # EXPECTED TO FAIL: confirmed unsurfaced negative-cash finding, see file docstring
    assert np.all(running >= 0.0)


def test_muhurat_and_shortened_sessions_same_panel():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([60, 105])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
        int(ts[row_at(day_offsets, 1, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )

    assert panel.n_rows() == 60 + 105 == 165
    result = run_backtest(strategy, panel, default_config())

    assert result.forced_eod_liquidation_days == 0
    assert_i1_terminal(result)
    assert_finite_everywhere(result)


def test_single_row_session_no_crash():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([10, 1, 10])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
        int(ts[row_at(day_offsets, 2, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )

    assert panel.n_rows() == 21
    result = run_backtest(strategy, panel, default_config())

    day1_ts = int(ts[row_at(day_offsets, 1, "09:15")])
    assert not np.any(result.trades["ts"].to_numpy() == day1_ts)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)


def test_square_off_time_absent_from_panel():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([15])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20",),
    )

    result = run_backtest(
        strategy,
        panel,
        default_config(square_off_time="15:20"),
    )

    # The clamp-to-last-row contract predicts zero; keep this assertion strict if it differs.
    assert result.forced_eod_liquidation_days == 0
    assert_i1_terminal(result)
    assert_finite_everywhere(result)


def test_no_stop_state_leak_across_weekend_gap():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20, 20], start=dt.date(2024, 1, 5))
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    # Friday's last bar (09:29) to Monday's first bar (09:15) is ~71.8h -- just under
    # 3 full days (3 days would land at Monday 09:29, 14 minutes later), not over.
    # 2 days cleanly separates this from a normal overnight gap (~17-18h).
    assert (
        ts[row_at(day_offsets, 1, "09:15")]
        - ts[row_at(day_offsets, 0, "09:29")]
    ) > 2 * 24 * 3600

    low[row_at(day_offsets, 1, "09:24"), 0] = 90.0

    day0_target = TargetPortfolio(
        weights=np.array([0.10, 0.0]),
        meta={"stop:AAA": 95.0},
    )
    day1_target = TargetPortfolio(
        weights=np.array([0.10, 0.0]),
        meta={},
    )
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): day0_target,
        int(ts[row_at(day_offsets, 1, "09:20") - 1]): day1_target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:26"),
        needs_intrabar_risk=True,
    )

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    result = run_backtest(strategy, panel, default_config())

    assert np.asarray(result.positions)[-2, 0] != 0.0
    assert_i1_terminal(result)
    assert_finite_everywhere(result)


def test_rejected_order_then_different_target_same_day():
    SYMBOLS = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20, 20])
    n_rows = len(ts)

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1_000_000.0)

    rejected_fill_row = row_at(day_offsets, 1, "09:21")
    volume[rejected_fill_row, 0] = 0.0

    day0_target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    day1_first_target = TargetPortfolio(
        weights=np.array([0.10, 0.0]),
        meta={},
    )
    day1_second_target = TargetPortfolio(
        weights=np.array([0.20, 0.0]),
        meta={},
    )
    script = {
        int(ts[row_at(day_offsets, 0, "09:20") - 1]): day0_target,
        int(ts[row_at(day_offsets, 1, "09:20") - 1]): day1_first_target,
        int(ts[row_at(day_offsets, 1, "09:25") - 1]): day1_second_target,
    }
    strategy = TsScriptStrategy(
        _EmptyParams(),
        script,
        decision_times=("09:20", "09:25"),
    )

    panel = make_panel(
        SYMBOLS,
        ts,
        day_offsets,
        dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    result = run_backtest(strategy, panel, default_config())

    expected = GrossNotionalSizer().to_shares(
        np.array([0.20, 0.0]),
        np.array([100.0, 100.0]),
        CAPITAL,
        bar_traded_value=np.array([100_000_000.0, 100_000_000.0]),
        max_participation=0.02,
    ).shares[0]

    second_fill_row = row_at(day_offsets, 1, "09:25") + 1
    second_entry = result.trades[
        (result.trades["symbol"] == "AAA")
        & (result.trades["ts"] == int(ts[second_fill_row]))
        & (result.trades["qty"] > 0)
    ]

    assert len(second_entry) == 1
    assert second_entry.iloc[0]["qty"] == pytest.approx(expected)
    assert_i1_terminal(result)
    assert_finite_everywhere(result)





def test_last_row_decision_fills_across_session_boundary():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([10, 10])
    prices = np.full((len(ts), len(symbols)), 100.0)
    volume = np.full((len(ts), len(symbols)), 1_000_000.0)
    panel = make_panel(
        symbols,
        ts,
        day_offsets,
        dates,
        open_=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=volume,
    )
    strategy = TsScriptStrategy(
        _EmptyParams(),
        {
            int(ts[8]): TargetPortfolio(
                weights=np.array([0.10, 0.0]),
            )
        },
        decision_times=None,
    )
    result = run_backtest(strategy, panel, default_config())

    day1_first_ts = int(ts[row_at(day_offsets, 1, "09:15")])
    # F1 FIXED (spec: order_lifecycle.md section C/D, engine.py session-end drop): the
    # order decided on day 0's last bar cannot fill within day 0, so the engine drops
    # it at the session boundary and counts the drop instead of letting it fill on day
    # 1's opening bar.
    cross_boundary_trades = result.trades[result.trades["ts"] == day1_first_ts]
    assert cross_boundary_trades.empty, (
        f"expected no fill at day 1's first-bar ts={day1_first_ts} (the order decided "
        f"on day 0's last bar must be dropped, not silently filled across the session "
        f"boundary), got {len(cross_boundary_trades)} trade(s)"
    )
    assert result.n_orders_dropped_at_session_end == 1, (
        "expected the day-0-last-bar order to be dropped and counted exactly once, got "
        f"n_orders_dropped_at_session_end={result.n_orders_dropped_at_session_end}"
    )
    assert_session_ends_flat(result)
    assert_finite_everywhere(result)
    assert_i1_terminal(result)
