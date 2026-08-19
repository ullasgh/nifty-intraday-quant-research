"""Adversarial engine/portfolio invariant tests (author B), from
specs/engine_adversarial_invariants.md.

This spec adds no feature. It asserts invariants the CURRENT engine is claimed to
satisfy under hostile inputs. A test that PASSES confirms the invariant; a test that
FAILS is a FINDING about real behaviour. Findings are reported here, never "fixed" by
weakening an assertion or touching src/. See the bottom of this docstring for the
findings recorded after actually running the suite (`pytest -q`).

============================================================================
SPEC DEFECTS (author B's independent reading, before implementation existed)
============================================================================
- Item 12 says "Per engine.py:9-11 the fill must be at the conservative price ...
  min(stop_price, open[t])". The line numbers are stale relative to the checked-out
  engine.py (the rule now lives in the module docstring around lines 9-11 AND is
  re-implemented at the intrabar-risk block around line 453); the described behaviour
  itself is unambiguous and testable, so this is cosmetic, not a blocker.
- I1's docstring pointer ("guards.py:655-677 ... engine.py:412-419") is accurate for
  this checkout, but note it only covers the check called at bar-open on close[t-1].
  The spec explicitly says these tests must go further (post-fill, post-stop,
  post-square-off, post-liquidation); doing that black-box (this file cannot hook the
  engine internals) requires re-deriving cash from the public `trades` ledger rather
  than re-invoking `guards.check_accounting`, which would just be a tautological
  re-assertion of a formula (`Portfolio.mark` sets cum_pnl so the identity holds by
  construction wherever `mark()` is called). See `assert_accounting_identity` below.
- The spec does not say whether "ruin_index points at the right row" means the row
  where equity first breaches zero, or the row whose *return* is undefined because the
  *prior* equity was <= 0 (which is one row later, per `_compute_returns`'s own
  docstring). Both readings are defensible; this file tests the natural reading
  (ruin_index == the row where equity itself first goes non-positive) and reports the
  mismatch as a finding below.
- Item 19 conflates "square-off time absent from panel" with several distinct code
  paths in the engine (direct fill on the actual last row; a queued fill for a
  following row that doesn't exist and is silently dropped; a genuinely-missing time
  that always resolves to the last row per `np.searchsorted(..., side="left")`). The
  spec only asks for one; this file tests the "always resolves to the last row" case,
  which is the one guaranteed whenever `square_off_time` is later than the session
  (the common real case: Muhurat/DR sessions end well before 15:20).

============================================================================
INVARIANT FINDINGS -- from the actual `pytest -q` run of this file, not predicted
in advance. Do not trust this summary over the individual test failures/docstrings.
============================================================================
PASS: items 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18, 19 (I1/I2 hold as tested).

FAIL (genuine findings; see the named test's assert message for the exact numbers):
  - Item 10: I2 (square-off/terminal liquidation) still reaches flat even though the
    symbol has been untradable since row 3 -- because the engine's forced square-off
    and EOD-liquidation paths call `_execute_direct_fill`, which does NOT consult the
    `tradable` mask at all (only the normal decision-order path does). So I2 holds,
    but only because the very halt check that governs ordinary trading is silently
    bypassed for forced liquidation. Not scored as a hard FAIL of I2 itself (I2 does
    hold), but flagged here because it is an asymmetry the spec's "if it cannot [reach
    flat], that is a finding" framing did not anticipate in the other direction.
  - Item 11 (I4): `Portfolio.equity`'s `np.where(shares != 0, prices, 0.0)` keeps the
    RAW price (NaN and all) wherever a position is open, so a single NaN close on a
    held symbol poisons that row's `np.dot` to NaN. `result.equity_curve` contains an
    explicit NaN entry. As a side effect this also flips `result.ruined=True` with
    `ruin_index` pointing at the NaN row itself (via `_compute_returns`'s own
    non-finite check), unlike item 16's crash below.
  - Item 15: cash goes negative (-90.83, from transaction charges on a 100%-of-capital
    entry) while `result.ruined` stays `False` and equity stays strongly positive.
    Negative cash is neither prevented nor surfaced anywhere in `BacktestResult`.
  - Item 16: `ruin_index` is one row LATER than the row where equity itself first
    reaches 0.0. `_compute_returns` only flags a row "bad" when its *prior* row's
    equity was <= 0 (or the row's own equity is non-finite -- see item 11), so the
    crash row itself is never flagged; only the row after it is.
  - Item 20: a decision made on a session's FINAL row queues an order for
    fill_row = t+1, which lands on the FOLLOWING session's first bar. Nothing in
    `on_session_start` (which only clears `active_stops`/`square_off_queued`) cancels
    or re-evaluates this pending order across the boundary, so it silently fills on
    Monday's open as a direct consequence of a Friday decision -- state (a pending
    order) does cross the session boundary.
"""

from __future__ import annotations

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
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Panel construction helpers -- synthetic, deterministic, never touches data/.
# ---------------------------------------------------------------------------


def _epoch_seconds(day: dt.date, hhmm: str) -> int:
    hour, minute = (int(p) for p in hhmm.split(":"))
    ts_ist = pd.Timestamp(day.year, day.month, day.day, hour, minute, tz=_IST)
    return int(ts_ist.tz_convert("UTC").timestamp())


def minute_labels(start: str, n: int) -> list[str]:
    hour, minute = (int(p) for p in start.split(":"))
    base = hour * 60 + minute
    return [f"{(base + i) // 60:02d}:{(base + i) % 60:02d}" for i in range(n)]


def build_panel(
    symbols: tuple[str, ...],
    day_minutes: list[tuple[dt.date, list[str]]],
    field_values: dict[str, np.ndarray],
) -> Panel:
    """Build a Panel from explicit per-day minute labels and full field arrays.

    `field_values` maps field name -> full (n_rows, n_symbols) array, NaN allowed,
    never forward-filled here. Rows are never a fixed count per day (rule 5).
    """
    ts_list: list[int] = []
    offsets = [0]
    dates: list[dt.date] = []
    for day, minutes in day_minutes:
        for hhmm in minutes:
            ts_list.append(_epoch_seconds(day, hhmm))
        offsets.append(offsets[-1] + len(minutes))
        dates.append(day)
    ts = np.array(ts_list, dtype=np.int64)
    n_rows = ts.shape[0]
    for name, arr in field_values.items():
        assert arr.shape == (n_rows, len(symbols)), (name, arr.shape, n_rows)
    fields = {name: np.asarray(arr, dtype=np.float32) for name, arr in field_values.items()}
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=np.array(offsets, dtype=np.int32),
        dates=np.array(dates, dtype=object),
    )


def make_fields(
    open_: np.ndarray,
    close: np.ndarray,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    open_arr = np.asarray(open_, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    high_arr = (
        np.asarray(high, dtype=np.float64) if high is not None else np.fmax(open_arr, close_arr)
    )
    low_arr = (
        np.asarray(low, dtype=np.float64) if low is not None else np.fmin(open_arr, close_arr)
    )
    vol_arr = (
        np.asarray(volume, dtype=np.float64)
        if volume is not None
        else np.full_like(open_arr, 1_000_000.0)
    )
    return {
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": vol_arr,
    }


def make_config(
    capital: float = 1_000_000.0,
    *,
    square_off_time: str = "15:20",
    cost_model=None,
    fill_model: FillModel | None = None,
    sizer: GrossNotionalSizer | None = None,
) -> BacktestConfig:
    """Deterministic, arithmetic-clean default config: no slippage, no cost, full
    capacity, whole-capital sizer. Individual tests override what they need to hit
    the specific mechanism under test (participation caps, real cost models, etc).
    """
    return BacktestConfig(
        capital=capital,
        square_off_time=square_off_time,
        cost_model=cost_model if cost_model is not None else ZeroCost(),
        fill_model=(
            fill_model
            if fill_model is not None
            else FillModel(slippage=ZeroSlippage(), max_participation=1.0)
        ),
        sizer=(
            sizer
            if sizer is not None
            else GrossNotionalSizer(max_weight=1.0, min_trade_notional=0.0)
        ),
    )


def assert_accounting_identity(
    trades: pd.DataFrame,
    equity: float,
    shares: np.ndarray,
    prices_now: np.ndarray,
    capital: float,
    *,
    up_to_ts: int | None = None,
    atol: float = 1e-6,
) -> None:
    """I1, re-derived independently from the public trade ledger (not by re-invoking
    `guards.check_accounting`, which would be tautological -- `Portfolio.mark` sets
    `cum_pnl` so the identity holds by construction wherever `mark()` runs).

    cash = capital - sum(signed fill notional) - sum(charges), purely from `trades`.
    equity = cash + positions_value(shares, prices_now).

    `up_to_ts` MUST be passed whenever `equity`/`shares` reflect a point in time
    strictly before the backtest's end -- otherwise later trades (e.g. the forced
    square-off) leak into the derived cash and the check is meaningless.
    """
    df = trades if up_to_ts is None else trades[trades["ts"] <= up_to_ts]
    signed_notional = float(np.sum(np.where(df["is_buy"], df["notional"], -df["notional"])))
    total_charges = float(df["charges"].sum())
    expected_cash = capital - signed_notional - total_charges
    positions_value = float(np.dot(shares, np.where(shares != 0, prices_now, 0.0)))
    expected_equity = expected_cash + positions_value
    assert abs(expected_equity - equity) <= atol, (
        f"I1 accounting identity violated: expected_equity={expected_equity}, "
        f"reported equity={equity}, discrepancy={expected_equity - equity}"
    )


def assert_flat_at_session_end(panel: Panel, positions: np.ndarray) -> None:
    """I2: shares == 0 at every session's last row.

    Valid only for strategies using DataRequest(decision_times=None), where decision
    rows are exactly t=1..n_rows-1: `positions[k]` (0-indexed) is the snapshot taken
    INSIDE row t=k+1's own decision block, which runs BEFORE that same row's
    square-off/forced-liquidation code later in the same iteration. So `positions[t-1]`
    is the PRE-square-off state for row t, not the post-square-off state. The
    post-square-off state for a day's last row `L` is only ever observable at array
    index `L`: for a non-final day that is day+1's first decision-row append (which
    runs strictly after day L's own square-off, chronologically); for the panel's
    final day it is the extra snapshot the engine appends once after the whole loop
    finishes (`positions` has exactly `n_rows` entries in total). Both cases resolve
    to the same formula: `pos_idx = last_row`.
    """
    for day_idx in range(len(panel.day_offsets) - 1):
        last_row = int(panel.day_offsets[day_idx + 1]) - 1
        pos_idx = last_row
        if pos_idx < positions.shape[0]:
            assert np.all(positions[pos_idx] == 0.0), (
                f"I2 violated: nonzero shares at day {day_idx} last row {last_row}: "
                f"{positions[pos_idx]}"
            )


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


class _EmptyParams(BaseModel):
    pass


class ScriptedStrategy(Strategy):
    """Returns a scripted target on chosen decision-call indices (0-based, across
    ALL decision calls for the whole backtest, not per-row). `repeat_last=True`
    reuses the most recent scheduled entry forever after (a "persistent target").
    """

    name = "scripted"
    Params = _EmptyParams

    def __init__(
        self,
        schedule: dict[int, tuple[np.ndarray, dict]],
        *,
        needs_intrabar_risk: bool = False,
        repeat_last: bool = False,
        decision_times: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(params=self.Params())
        self._schedule = dict(schedule)
        self._repeat_last = repeat_last
        self._step = 0
        self._needs_intrabar_risk = needs_intrabar_risk
        self._decision_times = decision_times

    def data_request(self) -> DataRequest:
        return DataRequest(
            decision_times=self._decision_times, needs_intrabar_risk=self._needs_intrabar_risk
        )

    def precompute(self, panel):
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        step = self._step
        self._step += 1
        entry = self._schedule.get(step)
        if entry is None and self._repeat_last and self._schedule:
            last_key = max(k for k in self._schedule if k <= step)
            entry = self._schedule[last_key]
        if entry is None:
            return None
        weights, meta = entry
        return TargetPortfolio(weights=np.array(weights, dtype=np.float64), meta=dict(meta))


def one_shot(weights, meta=None, *, needs_intrabar_risk: bool = False) -> ScriptedStrategy:
    return ScriptedStrategy(
        {0: (weights, meta or {})}, needs_intrabar_risk=needs_intrabar_risk, repeat_last=False
    )


def persistent(weights, meta=None, *, needs_intrabar_risk: bool = False) -> ScriptedStrategy:
    return ScriptedStrategy(
        {0: (weights, meta or {})}, needs_intrabar_risk=needs_intrabar_risk, repeat_last=True
    )


D0 = dt.date(2024, 1, 2)  # Tuesday
D_FRI = dt.date(2024, 1, 5)
D_MON = dt.date(2024, 1, 8)


# ===========================================================================
# 1. Long -> short crossing zero in one order
# ===========================================================================


def test_01_long_to_short_crossing_zero():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 6)
    day_minutes = [(D0, minutes)]
    # step0 (t=1): weight +1 -> fills row2 @ 100 (buy 10_000).
    # step1 (t=2): weight -1 -> fills row3 @ 110 (sell 20_000, crossing to -10_000).
    open_ = np.array([[100.0], [100.0], [100.0], [110.0], [120.0], [120.0]])
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = ScriptedStrategy(
        {0: (np.array([1.0]), {}), 1: (np.array([-1.0]), {})}, repeat_last=False
    )
    config = make_config()
    result = run_backtest(strategy, panel, config)

    # Manual ledger: buy 10_000 @ 100 (cash 1_000_000 -> 0), sell 20_000 @ 110
    # (cash 0 -> +2_200_000), realising a 100_000 gain on the closing long leg and
    # opening a fresh -10_000 short at basis 110.
    positions = result.positions
    # decision rows: t=1,2,3,4,5 -> positions indices 0..4 (+1 final duplicate of t=5)
    assert positions[1][0] == pytest.approx(10_000.0)  # after row2 fill, before flip
    assert positions[2][0] == pytest.approx(-10_000.0)  # after row3 crossing fill

    idx3 = 2  # equity_curve/positions index for t=3
    expected_cash_after_cross = 2_200_000.0
    expected_equity_row3 = expected_cash_after_cross + (-10_000.0) * 110.0  # mark at close[3]
    assert result.equity_curve[idx3] == pytest.approx(expected_equity_row3, abs=1e-6)

    assert_accounting_identity(
        result.trades,
        float(result.equity_curve[idx3]),
        positions[idx3],
        close[3],
        config.capital,
        up_to_ts=int(panel.ts[3]),
    )
    assert_flat_at_session_end(panel, positions)
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 2. Short -> flat
# ===========================================================================


def test_02_short_to_flat():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 6)
    day_minutes = [(D0, minutes)]
    open_ = np.array([[100.0], [100.0], [100.0], [90.0], [90.0], [90.0]])
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = ScriptedStrategy(
        {0: (np.array([-1.0]), {}), 1: (np.array([0.0]), {})}, repeat_last=False
    )
    config = make_config()
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    assert positions[1][0] == pytest.approx(-10_000.0)
    assert positions[2][0] == pytest.approx(0.0)

    # Short 10_000 @ 100 (cash 1_000_000 -> 2_000_000), buy back 10_000 @ 90
    # (cash -> 1_100_000), realising a 100_000 gain, flat.
    idx3 = 2
    assert result.equity_curve[idx3] == pytest.approx(1_100_000.0, abs=1e-6)  # cash == equity, flat
    assert_flat_at_session_end(panel, positions)


# ===========================================================================
# 3. Short -> long crossing zero
# ===========================================================================


def test_03_short_to_long_crossing_zero():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 6)
    day_minutes = [(D0, minutes)]
    open_ = np.array([[100.0], [100.0], [100.0], [90.0], [90.0], [90.0]])
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = ScriptedStrategy(
        {0: (np.array([-1.0]), {}), 1: (np.array([1.0]), {})}, repeat_last=False
    )
    config = make_config()
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    assert positions[1][0] == pytest.approx(-10_000.0)
    assert positions[2][0] == pytest.approx(10_000.0)

    # Short 10_000 @ 100 (cash -> 2_000_000), buy 20_000 @ 90 crossing to +10_000
    # (cash -> 2_000_000 - 1_800_000 = 200_000).
    idx3 = 2
    expected_cash = 200_000.0
    expected_equity = expected_cash + 10_000.0 * 90.0
    assert result.equity_curve[idx3] == pytest.approx(expected_equity, abs=1e-6)
    assert_flat_at_session_end(panel, positions)
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 4. Long -> flat -> long again within one session
# ===========================================================================


def test_04_long_flat_long_within_session():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 7)
    day_minutes = [(D0, minutes)]
    open_ = np.full((7, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = ScriptedStrategy(
        {
            0: (np.array([1.0]), {}),  # t=1 -> fill row2, long
            1: (np.array([0.0]), {}),  # t=2 -> fill row3, flat
            2: (np.array([1.0]), {}),  # t=3 -> fill row4, long again
        },
        repeat_last=False,
    )
    config = make_config()
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    assert positions[1][0] == pytest.approx(10_000.0)  # after row2
    assert positions[2][0] == pytest.approx(0.0)  # after row3
    assert positions[3][0] == pytest.approx(10_000.0)  # after row4

    assert_flat_at_session_end(panel, positions)
    idx = 4  # index for t=5: quiescent, strictly before the forced square-off at t=6
    assert_accounting_identity(
        result.trades,
        float(result.equity_curve[idx]),
        positions[idx],
        close[5],
        config.capital,
        up_to_ts=int(panel.ts[5]),
    )
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 5. Target identical to current holding: no order, no cost
# ===========================================================================


def test_05_identical_target_emits_no_order():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 6)
    day_minutes = [(D0, minutes)]
    open_ = np.full((6, 1), 100.0)  # perfectly flat, so target_shares is bit-identical
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([0.5]))  # same weight every decision call
    config = make_config(cost_model=NSEIntradayEquityCosts())
    result = run_backtest(strategy, panel, config)

    trades = result.trades
    aaa_trades = trades[trades["symbol"] == "AAA"]
    # Exactly one entry trade (row2 fill) and one forced-flat trade at session end;
    # the repeated identical target on every other decision row must emit nothing.
    assert len(aaa_trades) == 2, (
        f"expected exactly 2 trades (entry + square-off), got:\n{aaa_trades}"
    )
    assert_flat_at_session_end(panel, result.positions)


# ===========================================================================
# 6. Zero fill (bar traded value zero): rejected, unchanged, no charge, I1 holds
# ===========================================================================


def test_06_zero_bar_traded_value_rejects_fill():
    symbols = ("AAA",)
    minutes = minute_labels("09:15", 5)
    day_minutes = [(D0, minutes)]
    open_ = np.full((5, 1), 100.0)
    close = open_.copy()
    volume = np.full((5, 1), 1_000_000.0)
    volume[2, 0] = 0.0  # fill row (t=2) has zero traded value
    fields = make_fields(open_, close, volume=volume)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = one_shot(np.array([1.0]))
    config = make_config(cost_model=NSEIntradayEquityCosts())
    result = run_backtest(strategy, panel, config)

    assert result.n_trades == 0
    assert result.total_costs == pytest.approx(0.0)
    assert np.all(result.positions == 0.0)
    assert result.rejected_order_rate > 0.0
    assert_flat_at_session_end(panel, result.positions)
    assert np.all(np.isfinite(result.equity_curve))
    assert result.equity_curve[-1] == pytest.approx(config.capital, abs=1e-6)


# ===========================================================================
# 7. Repeated partial fills across consecutive rows
# ===========================================================================


def test_07_repeated_partial_fills_across_rows():
    symbols = ("AAA",)
    n = 8
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    # Ramping volume: capacity (participation-cap notional) grows every bar, so a
    # constant full-capital target keeps producing fresh, nonzero, capacity-limited
    # orders on consecutive decision rows instead of reaching the target in one fill.
    volume = np.array([[5_000.0 + 3_000.0 * i] for i in range(n)])
    fields = make_fields(open_, close, volume=volume)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))  # full-capital target, every decision row
    config = make_config(fill_model=FillModel(slippage=ZeroSlippage(), max_participation=0.02))
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    abs_shares = np.abs(positions[:, 0])
    # At least 3 consecutive decision rows with strictly increasing position size:
    # the target is never reached in a single bar because of the participation cap.
    growing = [i for i in range(1, len(abs_shares)) if abs_shares[i] > abs_shares[i - 1] > 0]
    assert len(growing) >= 3, f"expected >=3 consecutive growth steps, got shares={abs_shares}"

    full_target_shares = config.capital / 100.0  # weight=1.0, price=100
    # Position never reaches the full uncapped target while capacity keeps binding
    # early on -- confirms the fills genuinely are partial relative to desired size.
    assert abs_shares[1] < full_target_shares

    trades = result.trades
    buys = trades[trades["is_buy"]]
    assert len(buys) >= 3
    assert np.all(buys["filled_frac"] > 0.0)

    assert_flat_at_session_end(panel, positions)
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 8. Rejected order does not corrupt in_flight; the target is re-ordered whole,
#    not doubled and not suppressed, on the next decision row.
# ===========================================================================


def test_08_rejected_order_does_not_corrupt_in_flight():
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    volume = np.full((n, 1), 1_000_000.0)
    volume[2, 0] = 0.0  # row2's fill is rejected (zero bar-traded-value)
    fields = make_fields(open_, close, volume=volume)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))  # same full-capital target every call
    config = make_config()
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    assert positions[1][0] == pytest.approx(0.0)  # row2 fill rejected: still flat
    # row3's fill must land on the FULL original target (10_000), neither doubled
    # (an in_flight bug that fails to net out the rejected order) nor suppressed
    # (an in_flight bug that treats the rejected order as still outstanding).
    full_target_shares = config.capital / 100.0
    assert positions[2][0] == pytest.approx(full_target_shares)

    assert_flat_at_session_end(panel, positions)


# ===========================================================================
# 9. Fill at a price of exactly zero, and at a NaN price: both rejected, no
#    divide-by-zero corruption.
# ===========================================================================


def test_09_zero_and_nan_fill_prices_are_rejected_not_divided_by():
    symbols = ("AAA", "BBB")
    n = 5
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 2), 100.0)
    close = open_.copy()
    open_[2, 0] = 0.0  # AAA's fill row: exact zero price
    open_[2, 1] = np.nan  # BBB's fill row: NaN price
    close[2, :] = open_[2, :]
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = one_shot(np.array([0.5, 0.5]))
    config = make_config()
    with np.errstate(all="raise"):
        result = run_backtest(strategy, panel, config)

    assert np.all(result.positions == 0.0)
    assert result.n_trades == 0
    assert np.all(np.isfinite(result.equity_curve))
    assert result.equity_curve[-1] == pytest.approx(config.capital, abs=1e-6)


# ===========================================================================
# 10. Halt after entry: symbol becomes untradable while a position is open.
# ===========================================================================


def test_10_halt_after_entry_no_stale_mark_and_square_off_still_flat():
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = np.array([[100.0], [100.0], [100.0], [105.0], [110.0], [115.0]])
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    tradable = np.ones((n, 1), dtype=bool)
    tradable[3:, 0] = False  # AAA halts from row3 onward, price data keeps moving

    strategy = one_shot(np.array([1.0]))
    config = make_config()
    result = run_backtest(strategy, panel, config, tradable=tradable)

    positions = result.positions
    assert positions[1][0] == pytest.approx(10_000.0)  # entered before the halt

    # Not marked stale: equity at the halted rows must track the REAL close price,
    # not freeze at the last tradable price.
    idx3 = 2  # t=3, first halted row
    idx4 = 3  # t=4
    expected_equity_idx3 = 0.0 + 10_000.0 * close[3, 0]  # cash is 0 after full-capital buy
    expected_equity_idx4 = 0.0 + 10_000.0 * close[4, 0]
    assert result.equity_curve[idx3] == pytest.approx(expected_equity_idx3, abs=1e-6)
    assert result.equity_curve[idx4] == pytest.approx(expected_equity_idx4, abs=1e-6)
    assert result.equity_curve[idx3] != pytest.approx(result.equity_curve[idx4])

    # I2: the engine's own forced square-off/EOD liquidation must still reach flat,
    # even though AAA has been untradable since row3 (see the file's top docstring
    # for why: `_execute_direct_fill` bypasses the tradable mask entirely).
    assert_flat_at_session_end(panel, positions)
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 11. Missing bar (NaN OHLCV) after entry: I4 (finite everywhere).
# ===========================================================================


def test_11_nan_close_after_entry_must_not_produce_nonfinite_equity():
    """FINDING, confirmed: `Portfolio.equity`'s `np.where(shares != 0, prices, 0.0)`
    keeps the raw (possibly NaN) price wherever a position is open, so a single NaN
    close on a held symbol poisons that row's mark-to-market to NaN via the dot
    product. This test asserts the actual I4 claim (finite everywhere) and FAILS,
    which is itself the reportable finding -- see the file's top docstring. Per rule
    9 the assertion is on a single decisive row, not a coverage sweep.
    """
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = np.array([[100.0], [100.0], [100.0], [np.nan], [100.0], [100.0]])
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))
    config = make_config()
    result = run_backtest(strategy, panel, config)

    assert np.all(np.isfinite(result.equity_curve)), (
        "I4 violated: a NaN close on a held symbol produced a non-finite equity_curve "
        f"entry: {result.equity_curve}"
    )


# ===========================================================================
# 12. Gap through a stop: fill at min(stop_price, open[t]) for a long, never at
#     the stop price when the open is worse.
# ===========================================================================


def test_12_gap_through_stop_fills_at_conservative_price():
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    open_[2, 0] = 70.0  # row2's open gaps well below the stop
    close = open_.copy()
    low = open_.copy()
    low[2, 0] = 65.0  # low is even lower, so the stop is definitely triggered
    high = np.maximum(open_, close)
    fields = make_fields(open_, close, high=high, low=low)
    panel = build_panel(symbols, day_minutes, fields)

    stop_price = 90.0
    strategy = one_shot(np.array([1.0]), meta={"stop:AAA": stop_price}, needs_intrabar_risk=True)
    config = make_config()
    result = run_backtest(strategy, panel, config)

    trades = result.trades
    sells = trades[~trades["is_buy"]]
    assert len(sells) == 1, f"expected exactly one stop-triggered sell, got:\n{trades}"
    fill_price = float(sells.iloc[0]["price"])
    assert fill_price == pytest.approx(70.0), (
        f"stop must fill at min(stop_price={stop_price}, open={70.0}) = 70.0, "
        f"got {fill_price} (would be {stop_price} if the conservative-price rule "
        "were violated)"
    )
    assert_flat_at_session_end(panel, result.positions)


# ===========================================================================
# 13. All-NaN symbol column: never enters a cross-sectional computation, never
#     receives an order.
# ===========================================================================


def test_13_absent_symbol_never_ordered_and_is_reported_absent():
    symbols = ("AAA", "BBB")
    n = 5
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 2), 100.0)
    close = open_.copy()
    open_[:, 1] = np.nan  # BBB entirely absent
    close[:, 1] = np.nan
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    # Equal target weights for both symbols; BBB must be masked out and AAA
    # renormalised to absorb the full gross, per the engine's own absent-symbol
    # handling (engine.py ~550-556).
    strategy = persistent(np.array([0.5, 0.5]))
    config = make_config()
    result = run_backtest(strategy, panel, config)

    assert result.n_symbols_absent == 1
    assert result.absent_symbols == ("BBB",)

    trades = result.trades
    assert (trades["symbol"] == "BBB").sum() == 0
    assert np.all(result.positions[:, 1] == 0.0)
    # AAA absorbs the full gross: it received the entire 1.0 weight, not 0.5.
    full_target_shares = config.capital / 100.0
    assert result.positions[1][0] == pytest.approx(full_target_shares)

    assert_flat_at_session_end(panel, result.positions)
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 14. Extreme price move (10x in one bar): finite equity everywhere.
# ===========================================================================


def test_14_extreme_price_move_stays_finite():
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = np.array([[100.0], [100.0], [100.0], [1_000.0], [1_000.0], [1_000.0]])
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))
    config = make_config()
    result = run_backtest(strategy, panel, config)

    assert np.all(np.isfinite(result.equity_curve))
    assert np.all(np.isfinite(result.returns))
    assert np.all(np.isfinite(result.turnover))
    assert_flat_at_session_end(panel, result.positions)


# ===========================================================================
# 15. Negative cash: prevented, or explicitly surfaced via `ruined`.
# ===========================================================================


def test_15_negative_cash_is_prevented_or_surfaced():
    symbols = ("AAA",)
    n = 5
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    # Full-capital entry with a REAL cost model: buying with 100% of capital leaves
    # cash short by exactly the transaction charges, driving cash negative even
    # though the position's notional roughly matches capital.
    strategy = one_shot(np.array([1.0]))
    config = make_config(cost_model=NSEIntradayEquityCosts())
    result = run_backtest(strategy, panel, config)

    positions = result.positions
    idx2 = 1  # t=2, right after the entry fill
    derived_cash = float(result.equity_curve[idx2]) - float(positions[idx2][0]) * close[2, 0]

    if derived_cash < -1e-9:
        assert result.ruined, (
            f"cash went negative ({derived_cash}) but `ruined` was never set -- "
            "neither prevented nor surfaced, per I1's capital group (item 15)"
        )
    assert_flat_at_session_end(panel, positions)


# ===========================================================================
# 16. Ruin: equity reaches zero. `ruined` and `ruin_index` must be correct.
# ===========================================================================


def test_16_ruin_sets_ruined_and_ruin_index_at_the_crash_row():
    symbols = ("AAA",)
    n = 5
    minutes = minute_labels("09:15", n)
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = np.array([[100.0], [100.0], [0.0], [50.0], [50.0]])
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))
    config = make_config()  # ZeroCost: cash is exactly 0 after the full-capital buy
    result = run_backtest(strategy, panel, config)

    equity = result.equity_curve
    # t=1 (idx0): pre-fill, equity == capital.
    # t=2 (idx1): fill executes @ open[2]=100 (cash -> 0, shares -> 10_000), THEN
    #   marked @ close[2]=0.0 in the SAME row's decision block -> equity == 0.0
    #   exactly: the crash row.
    idx_crash = 1
    assert equity[idx_crash] == pytest.approx(0.0, abs=1e-6), f"equity_curve={equity}"

    assert result.ruined is True
    # Natural reading of "ruin_index points at the right row": the row where equity
    # itself first reaches <= 0, i.e. idx_crash. `_compute_returns` actually flags a
    # row "bad" (and thus ruined) only once its PRIOR row's equity is <= 0 -- see the
    # function's own docstring -- so the reported index is expected to be idx_crash+1,
    # one row later than the crash itself. This is the FINDING; do not weaken the
    # assertion to match it.
    assert result.ruin_index == idx_crash, (
        f"ruin_index={result.ruin_index}, expected {idx_crash} (the row where equity "
        f"itself first reaches <= 0). equity_curve={equity}"
    )


# ===========================================================================
# 17. 60-bar Muhurat-like and 105-bar DR-like sessions in the same panel.
# ===========================================================================


def test_17_variable_session_lengths_60_and_105_bars():
    symbols = ("AAA",)
    day1 = dt.date(2024, 1, 2)
    day2 = dt.date(2024, 1, 3)
    minutes1 = minute_labels("09:15", 60)
    minutes2 = minute_labels("09:15", 105)
    day_minutes = [(day1, minutes1), (day2, minutes2)]
    n = 60 + 105
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    assert panel.day_offsets.tolist() == [0, 60, 165]

    strategy = persistent(np.array([0.5]))
    config = make_config()
    result = run_backtest(strategy, panel, config)

    assert result.forced_eod_liquidation_days == 0
    assert_flat_at_session_end(panel, result.positions)
    assert np.all(np.isfinite(result.equity_curve))
    assert len(result.equity_curve) == n  # one per decision row (t=1..n-1) + 1 final


# ===========================================================================
# 18. A single-row session.
# ===========================================================================


def test_18_single_row_session():
    symbols = ("AAA",)
    day_minutes = [(D0, ["09:15"])]
    open_ = np.full((1, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = persistent(np.array([1.0]))  # never actually gets a chance to decide
    config = make_config()
    result = run_backtest(strategy, panel, config)

    assert result.n_trades == 0
    assert len(result.equity_curve) == 1
    assert result.equity_curve[0] == pytest.approx(config.capital)
    assert np.all(result.positions == 0.0)


# ===========================================================================
# 19. Square-off time absent from the panel entirely.
# ===========================================================================


def test_19_square_off_time_absent_from_panel_still_flattens_gracefully():
    symbols = ("AAA",)
    n = 6
    minutes = minute_labels("09:15", n)  # session ends 09:20, nowhere near 15:20
    day_minutes = [(D0, minutes)]
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    strategy = one_shot(np.array([1.0]))
    config = make_config(square_off_time="15:20", cost_model=NSEIntradayEquityCosts())
    result = run_backtest(strategy, panel, config)

    assert_flat_at_session_end(panel, result.positions)
    # Must resolve via the adaptive last-row square-off path (normal intraday
    # costs), not the punitive forced_eod_liquidation delivery-cost fallback.
    assert result.forced_eod_liquidation_days == 0
    assert np.all(np.isfinite(result.equity_curve))


# ===========================================================================
# 20. Two adjacent sessions with a weekend gap: no state leaks across
#     on_session_start.
# ===========================================================================


def test_20_weekend_gap_no_order_carries_into_next_session():
    """FINDING, confirmed: a decision made on session 1's LAST row queues a fill for
    fill_row = t+1, which is session 2's FIRST bar. `on_session_start` resets
    `active_stops`/`square_off_queued` but nothing cancels or re-evaluates a pending
    order across the boundary, so it silently fills on the following session's open.
    This test asserts the natural reading of "no state leaks" (no fill should
    execute in session 2 as a result of a decision taken in session 1) and FAILS --
    that failure is the finding, see the file's top docstring.
    """
    symbols = ("AAA",)
    fri_minutes = minute_labels("09:15", 4)  # t=0..3
    mon_minutes = minute_labels("09:15", 4)  # t=4..7
    day_minutes = [(D_FRI, fri_minutes), (D_MON, mon_minutes)]
    n = 8
    open_ = np.full((n, 1), 100.0)
    close = open_.copy()
    fields = make_fields(open_, close)
    panel = build_panel(symbols, day_minutes, fields)

    assert panel.day_offsets.tolist() == [0, 4, 8]
    # Real weekend gap in the raw ts: Friday's 4th bar (09:18) to Monday's first bar
    # (09:15) is just under 3 calendar days but comfortably more than 2.
    assert panel.ts[4] - panel.ts[3] > 2 * 86400

    # Fire only on the 3rd decision call, i.e. t=3 -- Friday's own LAST row.
    strategy = ScriptedStrategy({2: (np.array([1.0]), {})}, repeat_last=False)
    config = make_config()
    result = run_backtest(strategy, panel, config)

    monday_first_ts = int(panel.ts[4])
    trades = result.trades
    monday_open_trades = trades[trades["ts"] == monday_first_ts]
    assert monday_open_trades.empty, (
        "state leaked across on_session_start: a decision made on Friday's last row "
        f"produced a fill on Monday's first bar:\n{monday_open_trades}"
    )
    # Friday itself must still be flat at its own session end (I2), independent of
    # whatever happens to the pending order.
    fri_last_row = int(panel.day_offsets[1]) - 1
    fri_pos_idx = fri_last_row - 1
    assert np.all(result.positions[fri_pos_idx] == 0.0)
