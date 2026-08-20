"""
Defects / ambiguities / contradictions found in the spec:
- `DailyResult.dates` is declared as `np.ndarray  # int64 epoch-seconds`, but the spec also says
  dates are "the session dates from `panel.dates`", which in the repo are `datetime.date` objects
  (dtype=object). Item 9 explicitly says "not derived from row timestamps" and "panel's session
  dates", which is incompatible with int64 epoch-seconds. Tests will assert equality to
  `panel.dates` (object array of `datetime.date`), since that is the stated behavioural intent.
- First session with no decision row: spec does not say what equity to carry forward when there
  is no prior day. Rule 6 forbids silent forward-fill, but this is exactly a forward-fill. A
  reasonable invariant is to use `initial_capital`; tests will assert that.
- Item 7 requires comparing against the OLD reconstruction, but that old reconstruction lives in
  `cli.py` private helpers which this spec deletes. Reimplementing it in the test could diverge
  from the real old logic, especially around handling of the final row and duplicate rows.
- The old reconstruction's final row mapping is ambiguous if the final row is also a decision row
  (duplicate row index). This can cause length mismatch/shape issues in
  `aggregate_returns_by_group`.
- Rule 6 says no fixed 375-bar stride, but item 4 explicitly mentions "regular 375-bar sessions"
  mixed with shorter ones. This is not necessarily a contradiction, but it suggests the rule is
  about never assuming a constant length; tests must not rely on 375 anywhere.
- Item 2 example says "two returns of +1% yield 1.01*1.01-1". With the engine starting in cash,
  the first decision row return is likely 0, so getting two +1% rows requires at least three
  decision rows. Test will instead verify daily return equals manual compounding of the actual
  per-row returns for a day, and separately assert daily turnover sums to exact expected values.
- `gross_returns` aggregation rule is not specified beyond the existence of the field; tests
  assume it compounds like `returns` (since gross returns are also returns).
- Turnover final row inclusion: if the unconditional final row is appended at the end of the very
  last session and has nonzero turnover, daily turnover for the last day could be overstated.
  Tests avoid such ambiguity for intermediate days and use separate days where possible.
"""

import math
from dataclasses import asdict
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import BaseModel

# `DailyResult` is unused today: importing it is the intentional TDD-red probe --
# it must raise ImportError until `nifty_quant.backtest.daily` exists. Kept
# deliberately (see the module-level defect note above), not a ruff F401 miss.
from nifty_quant.backtest.daily import DailyResult, build_daily  # noqa: F401
from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import (
    PerformanceMetrics,
    aggregate_returns_by_group,
    compute_metrics,
)
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.calendar import SessionGrid
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

# ---------------------------------------------------------------------------
# Date/time helpers. Rule 4: ts is int64 epoch-seconds, no timezone attached.
# We follow existing repo tests: treat wall-clock times as UTC (naive epoch secs).
# ---------------------------------------------------------------------------
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")


def _make_epoch(date_obj: date, hour: int, minute: int) -> int:
    """hour:minute is an IST wall-clock time (rows_at_time/minute_of_day resolve in IST)."""
    dt = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, tzinfo=IST)
    return int(dt.timestamp())


def _session_ts(
    date_obj: date,
    n_bars: int,
    start_hour: int = 9,
    start_min: int = 15,
) -> np.ndarray:
    """Generate `n_bars` 1-minute bars, left-labelled, starting at start_hour:start_min."""
    start = _make_epoch(date_obj, start_hour, start_min)
    return (start + np.arange(n_bars, dtype=np.int64) * 60).astype(np.int64)


def _make_panel(
    sessions,
    symbols=("A",),
    prices=None,
) -> Panel:
    """
    Build an in-memory Panel from a list of (date, n_bars) sessions.
    `prices` optionally supplies a per-row close (and all OHLC) array.
    """
    all_ts = []
    for d, n in sessions:
        all_ts.append(_session_ts(d, n))
    ts = np.concatenate(all_ts).astype(np.int64)
    n_rows = ts.shape[0]
    n_symbols = len(symbols)

    if prices is None:
        prices = np.full((n_rows, n_symbols), 100.0, dtype=np.float32)
    else:
        prices = np.asarray(prices, dtype=np.float32)
        if prices.ndim == 1:
            prices = np.broadcast_to(prices.reshape(-1, 1), (n_rows, n_symbols)).copy()
        else:
            prices = prices.reshape(n_rows, n_symbols)

    fields = {}
    for name in ("open", "high", "low", "close", "volume"):
        if name == "volume":
            # Non-zero on purpose: FillModel.max_participation defaults to 0.02
            # (execution/fills.py:64), and the engine caps each order's fill
            # notional at max_participation * (volume * price) (engine.py:317-324,
            # portfolio.py GrossNotionalSizer.to_shares). With volume == 0 that cap
            # is 0 and every order fills for zero shares regardless of the target
            # weight, which is what left turnover/returns/equity vacuously zero in
            # this suite until now (masked earlier by the shape crash in bug 1).
            # Worst case in this suite: GrossNotionalSizer's own max_weight=0.10
            # clip (portfolio.py) bounds any single-symbol order to
            # 0.10 * gross(1.0) * capital(1e7, `_run_bt`'s fixed non-compounding
            # capital) = 1_000_000 notional. Breakeven bar_traded_value to clear the
            # 2% participation cap is 1_000_000 / 0.02 = 50_000_000 (the "50x" the
            # cap implies). We give it ~500x that notional instead of the bare
            # breakeven: volume=5_000_000 shares/bar at price~100 is a traded value
            # of ~5e8, so participation stays ~0.2% -- 10x inside the cap, not
            # balanced on top of it.
            fields[name] = np.full((n_rows, n_symbols), 5_000_000.0, dtype=np.float32)
        else:
            fields[name] = prices.copy()

    grid = SessionGrid.from_timestamps(ts)
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _make_empty_panel() -> Panel:
    """An empty Panel: zero rows, zero days."""
    ts = np.array([], dtype=np.int64)
    n_rows = 0
    n_symbols = 1
    fields = {
        name: np.empty((n_rows, n_symbols), dtype=np.float32)
        for name in ("open", "high", "low", "close", "volume")
    }
    return Panel(
        fields=fields,
        symbols=("A",),
        ts=ts,
        day_offsets=np.array([0], dtype=np.int32),
        dates=np.array([], dtype=object),
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class _TimeWeightParams(BaseModel):
    weights: dict[str, float]


class _TimeWeightStrategy(Strategy):
    name = "TimeWeightStrategy"
    Params = _TimeWeightParams

    def __init__(self, params: _TimeWeightParams):
        self.params = params

    def data_request(self) -> DataRequest:
        return DataRequest(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            decision_times=tuple(sorted(self.params.weights.keys())),
        )

    def precompute(self, panel: Panel):
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        # engine.py builds `view` at cursor = t - 1 (the last fully closed bar) by
        # design, never at the decision row t itself -- see the engine's own
        # "row t is the bar the strategy is not allowed to see yet" comment. Add
        # one bar (60s) back to recover the IST wall-clock time of the actual
        # decision row; safe here because no decision_time used in this suite is
        # a session's opening bar, so cursor never crosses a session boundary.
        hhmm = datetime.fromtimestamp(view.ts + 60, tz=IST).strftime("%H:%M")
        w = self.params.weights.get(hhmm, 0.0)
        return TargetPortfolio(weights=np.array([w], dtype=float), meta={})


class _ConstantWeightParams(BaseModel):
    weight: float = 1.0


class _ConstantWeightStrategy(Strategy):
    name = "ConstantWeightStrategy"
    Params = _ConstantWeightParams

    def __init__(self, params: _ConstantWeightParams = _ConstantWeightParams()):
        self.params = params

    def data_request(self) -> DataRequest:
        return DataRequest(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            decision_times=None,
        )

    def precompute(self, panel: Panel):
        return {}

    def on_decision(
        self, view: MarketView, signals, state: PortfolioState
    ) -> TargetPortfolio | None:
        return TargetPortfolio(
            weights=np.array([self.params.weight], dtype=float), meta={}
        )


# ---------------------------------------------------------------------------
# Old reconstruction reimplementation (test-local, not from cli.py)
# ---------------------------------------------------------------------------
def _old_row_indices(strategy: Strategy, panel: Panel) -> np.ndarray:
    """Reconstruct old cli.py row selection: decision rows + final row."""
    decision_times = strategy.data_request().decision_times
    if decision_times is None:
        decision_rows = np.arange(panel.n_rows(), dtype=np.int64)
    else:
        rows = []
        for t in decision_times:
            rows.append(panel.rows_at_time(t))
        decision_rows = np.sort(np.concatenate(rows)) if rows else np.array([], dtype=np.int64)

    final_row = panel.n_rows() - 1
    row_indices = np.append(decision_rows, final_row)
    row_indices = np.sort(row_indices)
    return row_indices


def _old_daily_returns(strategy: Strategy, panel: Panel, result_returns: np.ndarray) -> np.ndarray:
    """Old cli.py _daily_returns equivalent using rows_at_time + searchsorted."""
    row_indices = _old_row_indices(strategy, panel)
    if len(row_indices) != len(result_returns):
        raise ValueError(
            f"old reconstruction length mismatch: "
            f"rows={len(row_indices)} returns={len(result_returns)}"
        )
    group_ids = np.searchsorted(panel.day_offsets, row_indices, side="right") - 1
    return aggregate_returns_by_group(result_returns, group_ids)


def _run_bt(strategy, panel):
    """Run a backtest with zero cost and zero slippage for exact numeric tests."""
    # `max_weight=1.0`: GrossNotionalSizer's default max_weight=0.10 silently
    # clips every target weight above 10%, which this suite's strategies exceed
    # (weights up to 1.0). Raising the cap removes the clip; it never changes
    # sizing for weights already <= 0.10, so it cannot alter any test that
    # currently passes under the default.
    config = BacktestConfig(
        capital=1e7,
        sizer=GrossNotionalSizer(max_weight=1.0),
        cost_model=ZeroCost(),
        fill_model=FillModel(slippage=ZeroSlippage()),
    )
    return run_backtest(strategy, panel, config, contract=minimal_contract())


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _metrics_equal(a: PerformanceMetrics, b: PerformanceMetrics) -> bool:
    da = asdict(a)
    db = asdict(b)
    for key in da:
        va, vb = da[key], db[key]
        if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
            continue
        if va != vb:
            raise AssertionError(f"metrics field {key!r} differ: {va!r} != {vb!r}")
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_item_1_daily_returns_match_old_reconstruction():
    sessions = [(date(2022, 1, 3), 375), (date(2022, 1, 4), 375), (date(2022, 1, 5), 375)]
    # A flat 100.0 price makes every per-row return exactly 0.0 (buy-and-hold at an
    # unchanged price nets zero P&L), which made this test pass even under a
    # stubbed-to-all-zeros build_daily. Ramp price across the whole panel so real
    # per-row returns are genuinely nonzero.
    ts = np.concatenate([_session_ts(day, n) for day, n in sessions])
    n_rows = len(ts)
    prices = np.linspace(100.0, 120.0, n_rows, dtype=np.float32)
    panel = _make_panel(sessions, prices=prices)
    strategy = _TimeWeightStrategy(
        _TimeWeightParams(weights={"10:00": 0.2, "11:00": 0.8})
    )
    result = _run_bt(strategy, panel)

    old_returns = _old_daily_returns(strategy, panel, result.returns)

    assert len(old_returns) == len(panel.dates)
    assert len(result.daily.returns) == len(panel.dates)
    np.testing.assert_array_equal(result.daily.returns, old_returns)

    # gross_returns compounds within the session exactly like returns (Amendment 1
    # item 3) -- the whole suite otherwise only ever checks its length, so a broken
    # or always-zero gross_returns aggregation would go undetected. Reuse the same
    # old-reconstruction helper against the gross series.
    old_gross_returns = _old_daily_returns(strategy, panel, result.gross_returns)
    assert len(result.daily.gross_returns) == len(panel.dates)
    np.testing.assert_array_equal(result.daily.gross_returns, old_gross_returns)


def test_item_2_turnover_sums_returns_compound():
    """This test verifies the per-day attribution of turnover (day 0's square-off).

    The sums-vs-compounds contract is verified more directly by
    test_item_2b_build_daily_turnover_sums_and_returns_compound, which does not
    depend on engine attribution and cannot be vacuous.
    """
    # --- Turnover sums: two decision rows of turnover 0.1 and 0.2 => 0.6 total ---
    sessions = [(date(2022, 1, 3), 375), (date(2022, 1, 4), 375)]
    panel_const = _make_panel(sessions)
    strategy_turnover = _TimeWeightStrategy(
        _TimeWeightParams(weights={"10:00": 0.1, "10:30": 0.3})
    )
    result_turnover = _run_bt(strategy_turnover, panel_const)

    # Day 0 has two decision rows; final row is on day 1.
    # Daily turnover day 0 should be sum of the two decision-row turnovers plus
    # the mandatory end-of-day square-off: 0.1 (10:00) + 0.2 (10:30) + 0.3 (square-off).
    assert result_turnover.daily.turnover[0] == 0.6
    # The per-row turnovers now reflect the F8 fix: the 10:00 decision (row 0)
    # shows 0.0 (position change deferred), and the 10:30 decision (row 1)
    # combines with the end-of-day square-off to show 0.6 total.
    assert result_turnover.turnover[0] == 0.0
    assert result_turnover.turnover[1] == 0.6

    # --- Returns compound: daily return = (1+r1)*(1+r2)-1 for day 0 ---
    # Build a panel with two sessions and rising prices so the first day has
    # two decision rows with non-zero returns.
    sessions2 = [(date(2022, 1, 5), 375), (date(2022, 1, 6), 375)]
    # Generate prices: rise from 100 to 101 between 10:00 and 10:30 on day 0,
    # then stay at 101 for remainder of day 0 and all of day 1.
    ts = []
    for d, n in sessions2:
        ts.append(_session_ts(d, n))
    ts_all = np.concatenate(ts).astype(np.int64)
    n_rows = ts_all.shape[0]
    prices = np.full(n_rows, 100.0, dtype=np.float32)
    # Day 0 offsets: 10:00 is index 45, 10:30 is index 75.
    prices[45:76] = 100.0 + (np.arange(45, 76) - 45) / 30.0 * 1.0  # 100 to 101
    prices[76:] = 101.0

    panel_rising = _make_panel(sessions2, prices=prices)
    strategy_returns = _TimeWeightStrategy(
        _TimeWeightParams(weights={"10:00": 1.0, "10:30": 1.0})
    )
    result_returns = _run_bt(strategy_returns, panel_rising)

    # Identify rows for day 0 using old reconstruction, excluding final row.
    row_indices = _old_row_indices(strategy_returns, panel_rising)
    group_ids = np.searchsorted(panel_rising.day_offsets, row_indices, side="right") - 1
    day0_rows = row_indices[group_ids == 0]
    # Remove final row if it somehow landed in day 0 (it shouldn't: final is day 1).
    day0_rows = day0_rows[day0_rows != panel_rising.n_rows() - 1]

    # There should be exactly two decision rows in day 0: 10:00 and 10:30.
    assert len(day0_rows) == 2

    # Get their positions in result.returns
    pos = np.where(np.isin(row_indices, day0_rows))[0]
    r1 = result_returns.returns[pos[0]]
    r2 = result_returns.returns[pos[1]]

    expected_compound = (1.0 + r1) * (1.0 + r2) - 1.0
    assert np.isclose(result_returns.daily.returns[0], expected_compound, atol=1e-15)


def test_item_2b_build_daily_turnover_sums_and_returns_compound():
    """Direct unit test of the asymmetry: turnover SUMS, returns COMPOUND.

    The per-row values [0.1, 0.2] are chosen so that sum (0.3) and compound
    (0.32) are clearly distinguishable. This contract is verified here at the
    aggregation level, independent of engine attribution.
    """
    row_day_index = np.array([0, 0], dtype=np.int64)
    turnover = np.array([0.1, 0.2], dtype=np.float64)
    returns = np.array([0.1, 0.2], dtype=np.float64)
    gross_returns = np.array([0.1, 0.2], dtype=np.float64)
    # Equity values after applying the returns: 1e7 * (1.1) * (1.2) = 1.32e7
    equity = np.array([1.1e7, 1.32e7], dtype=np.float64)

    # UTC-midnight epoch for 2022-01-03 (arbitrary choice; it's a single day)
    date_2022_01_03 = int(
        datetime(2022, 1, 3, tzinfo=timezone.utc).timestamp()
    )
    dates = np.array([date_2022_01_03], dtype=np.int64)

    daily = build_daily(
        row_day_index=row_day_index,
        equity=equity,
        returns=returns,
        gross_returns=gross_returns,
        turnover=turnover,
        dates=dates,
        initial_capital=1e7,
    )

    # Assert turnover SUMS, not compounds
    expected_turnover_sum = 0.1 + 0.2  # 0.3
    expected_turnover_compound = (1.0 + 0.1) * (1.0 + 0.2) - 1.0  # 0.32
    assert np.isclose(daily.turnover[0], expected_turnover_sum, atol=1e-15)
    assert not np.isclose(daily.turnover[0], expected_turnover_compound, atol=1e-15)

    # Assert returns COMPOUND, not sum
    expected_returns_compound = (1.0 + 0.1) * (1.0 + 0.2) - 1.0  # 0.32
    expected_returns_sum = 0.1 + 0.2  # 0.3
    assert np.isclose(daily.returns[0], expected_returns_compound, atol=1e-15)
    assert not np.isclose(daily.returns[0], expected_returns_sum, atol=1e-15)


def test_item_3_equity_is_end_of_day():
    sessions = [(date(2022, 1, 10), 375), (date(2022, 1, 11), 375)]
    # Flat price makes buy-then-hold-unchanged net to exactly the same equity by
    # construction. Ramp price between the 10:00 and 10:30 decision rows (offsets
    # 45 and 75 within a 09:15-start session) so equity genuinely differs.
    ts = np.concatenate([_session_ts(day, n) for day, n in sessions])
    n_rows = len(ts)
    prices = np.full(n_rows, 100.0, dtype=np.float32)
    prices[45:76] = np.linspace(100.0, 103.0, 31, dtype=np.float32)
    prices[76:] = 103.0
    panel = _make_panel(sessions, prices=prices)
    strategy = _TimeWeightStrategy(
        _TimeWeightParams(weights={"10:00": 0.5, "10:30": 0.7})
    )
    result = _run_bt(strategy, panel)

    # Day 0 has two decision rows. Daily equity for day 0 should equal the
    # equity_curve entry corresponding to the *last* decision row in day 0,
    # i.e. the 10:30 row, not the 10:00 row.
    row_indices = _old_row_indices(strategy, panel)
    group_ids = np.searchsorted(panel.day_offsets, row_indices, side="right") - 1
    day0_rows = row_indices[group_ids == 0]
    # Exclude final row (if somehow day 0 were last, but it isn't)
    day0_decision_rows = day0_rows[day0_rows != panel.n_rows() - 1]
    last_day0_row = day0_decision_rows[-1]

    # Find its position in result.equity_curve
    pos = np.where(row_indices == last_day0_row)[0][0]

    assert result.daily.equity[0] == result.equity_curve[pos]
    # Ensure it's not picking the first row's equity (a broken impl might).
    first_pos = np.where(row_indices == day0_decision_rows[0])[0][0]
    assert result.daily.equity[0] != result.equity_curve[first_pos]


def test_item_4_irregular_sessions():
    sessions = [
        (date(2022, 2, 7), 375),
        (date(2022, 2, 8), 60),   # shortened session (e.g. Muhurat)
        (date(2022, 2, 9), 105),  # disaster-recovery session
        (date(2022, 2, 10), 375),
    ]
    panel = _make_panel(sessions)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"10:00": 0.5}))
    result = _run_bt(strategy, panel)

    assert result.daily.n_days == len(panel.dates) == 4
    assert len(result.daily.returns) == 4
    assert len(result.daily.equity) == 4
    assert len(result.daily.gross_returns) == 4
    assert len(result.daily.turnover) == 4
    assert len(result.daily.dates) == 4

    # Dates must be in ascending order and equal to UTC-midnight epoch keys for panel.dates.
    expected_dates = [
        int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        for d in panel.dates
    ]
    assert result.daily.dates.tolist() == expected_dates


def test_item_5_session_with_no_decision_row():
    sessions = [
        (date(2022, 3, 1), 375),  # has 11:00
        (date(2022, 3, 2), 60),   # 09:15-10:14, no 11:00
        (date(2022, 3, 3), 375),  # has 11:00
    ]
    panel = _make_panel(sessions)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"11:00": 0.5}))
    result = _run_bt(strategy, panel)

    assert result.daily.n_days == len(panel.dates) == 3

    # Day 1 is the missing-checkpoint session.
    assert result.daily.returns[1] == 0.0
    assert result.daily.turnover[1] == 0.0
    # Equity carried forward from previous day's end-of-day.
    assert result.daily.equity[1] == result.daily.equity[0]
    # Dates include the missing session, as UTC-midnight epoch-seconds
    # (Amendment 1 item 2 of specs/daily_results.md), not a raw datetime.date.
    expected_date_1 = int(
        datetime(
            panel.dates[1].year, panel.dates[1].month, panel.dates[1].day, tzinfo=timezone.utc
        ).timestamp()
    )
    assert result.daily.dates[1] == expected_date_1


def test_item_6_n_days_equals_len_panel_dates():
    sessions = [
        (date(2022, 4, 4), 375),
        (date(2022, 4, 5), 375),
        (date(2022, 4, 6), 375),
        (date(2022, 4, 7), 375),
    ]
    panel = _make_panel(sessions)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"10:15": 0.4}))
    result = _run_bt(strategy, panel)

    assert result.daily.n_days == len(panel.dates)


def test_item_7_metrics_unchanged():
    sessions = [(date(2022, 5, 2), 375), (date(2022, 5, 3), 375), (date(2022, 5, 4), 375)]
    # Same fix as test_item_1: a flat 100.0 price makes every per-row return
    # exactly 0.0, so compute_metrics(all-zeros) == compute_metrics(all-zeros)
    # trivially, regardless of whether the daily aggregation is correct. Ramp
    # price across the whole panel so the metrics are computed from real values.
    ts = np.concatenate([_session_ts(day, n) for day, n in sessions])
    n_rows = len(ts)
    prices = np.linspace(100.0, 120.0, n_rows, dtype=np.float32)
    panel = _make_panel(sessions, prices=prices)
    strategy = _TimeWeightStrategy(
        _TimeWeightParams(weights={"10:00": 0.3, "10:30": 0.9})
    )
    result = _run_bt(strategy, panel)

    old_returns = _old_daily_returns(strategy, panel, result.returns)
    assert len(old_returns) == len(result.daily.returns)

    m_old = compute_metrics(old_returns)
    m_new = compute_metrics(result.daily.returns)

    assert _metrics_equal(m_old, m_new)


def test_item_8_empty_panel_empty_returns():
    panel = _make_empty_panel()
    strategy = _ConstantWeightStrategy(_ConstantWeightParams(weight=1.0))
    result = _run_bt(strategy, panel)

    assert result.daily.n_days == 0
    assert result.daily.returns.size == 0
    assert result.daily.equity.size == 0
    assert result.daily.gross_returns.size == 0
    assert result.daily.turnover.size == 0
    assert result.daily.dates.size == 0


def test_item_9_dates_are_panel_session_dates():
    sessions = [
        (date(2022, 6, 6), 375),
        (date(2022, 6, 7), 375),
    ]
    panel = _make_panel(sessions)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"10:00": 0.2}))
    result = _run_bt(strategy, panel)

    # The DailyResult dates must be UTC-midnight epoch-seconds for the panel's
    # session dates, not raw datetime.date objects or epoch-seconds derived
    # from row timestamps.
    expected_dates = [
        int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        for d in panel.dates
    ]
    assert result.daily.dates.tolist() == expected_dates


def test_item_10_first_session_with_no_decision_row_equity_initial_capital():
    # The spec did not explicitly address this edge. We decide: if the very
    # first session has no decision row, its carried-forward equity is the
    # initial capital (there is nothing else to carry forward).
    sessions = [
        (date(2022, 7, 1), 60),   # no 11:00
        (date(2022, 7, 4), 375),  # has 11:00
    ]
    # Session 0 stays flat at 100.0 throughout (must have no decision row and no
    # trade, so its own assertions below keep holding). Session 1's "11:00" is
    # row offset 105 within that session, i.e. global row 60+105=165; its fill
    # lands at global row 166. Ramp price from there to the end of session 1 so
    # the position entered at 11:00 experiences real movement.
    ts = np.concatenate([_session_ts(day, n) for day, n in sessions])
    n_rows = len(ts)
    prices = np.full(n_rows, 100.0, dtype=np.float32)
    prices[166:] = np.linspace(100.0, 105.0, n_rows - 166, dtype=np.float32)
    panel = _make_panel(sessions, prices=prices)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"11:00": 0.5}))
    result = _run_bt(strategy, panel)

    assert result.daily.n_days == 2
    assert result.daily.returns[0] == 0.0
    assert result.daily.turnover[0] == 0.0
    assert result.daily.equity[0] == result.initial_capital
    # And day 1 has normal activity.
    assert result.daily.returns[1] != 0.0 or result.daily.turnover[1] != 0.0


def test_item_11_daily_lengths_consistency():
    sessions = [
        (date(2022, 8, 1), 375),
        (date(2022, 8, 2), 60),
        (date(2022, 8, 3), 105),
        (date(2022, 8, 4), 375),
    ]
    panel = _make_panel(sessions)
    strategy = _TimeWeightStrategy(_TimeWeightParams(weights={"10:00": 0.6}))
    result = _run_bt(strategy, panel)

    n = result.daily.n_days
    assert len(result.daily.dates) == n
    assert len(result.daily.equity) == n
    assert len(result.daily.returns) == n
    assert len(result.daily.gross_returns) == n
    assert len(result.daily.turnover) == n
    # Ensure n_days equals panel length, as required by item 6.
    assert n == len(panel.dates)
