"""Spec defects / ambiguities found

- DailyResult.dates is specified as int64 epoch-seconds, while Panel.dates contains
  datetime.date objects with no time-of-day. The required behaviour does not state the
  conversion convention; UTC midnight is implied by the concrete date test, but is not
  stated in section B itself.
- The meaning of n_days for an empty panel, a panel with zero-width sessions, and an empty
  returns vector is not fully specified. Item 6 counts panel.dates, while item 8 asks for an
  empty DailyResult, and section D says every session without a decision row should appear.
- Section D specifies carried-forward equity for a session with no engine row but does not
  specify the equity value to use if the first session has no row.
- gross_returns is listed as a DailyResult field but has no aggregation rule. It is unclear
  whether it should compound, sum, take the last value, or use a separate gross-P&L rule.
- The unconditional final engine row can be the only row in the final session, but the
  interaction between that row, section D's no-row rule, and the end-of-day equity rule is
  only implicit.
- Section A says day and row indices are recorded internally, but does not specify their
  names, storage, or a public construction/aggregation API for DailyResult.
- The empty-panel construction path is unspecified: it is unclear whether run_backtest itself
  must handle zero rows or whether only a lower-level daily aggregation path must do so.
- Section C says published metrics must not change, while section D adds zero-return periods
  for sessions omitted by the old reconstruction. On a backtest with missing decision rows,
  those extra periods can change annualization and other metrics unless the old comparison is
  also padded.
- Panel.dates and panel.ts are described separately, but the contract does not say what should
  happen if their calendar dates are intentionally not in lockstep; section B says output must
  follow panel.dates, whereas row selection follows timestamps.
- Adding daily to the frozen BacktestResult dataclass may affect positional construction and
  default-field ordering, but backward-compatibility requirements are not stated.
- The item-2 test guidance permits testing aggregation directly because exact engine-generated
  1% returns and 0.1/0.2 turnover are difficult to guarantee under the cost and fill models,
  but no named public aggregation entry point is specified for that direct test.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
from nifty_quant.backtest.daily import DailyResult
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import aggregate_returns_by_group, compute_metrics
from nifty_quant.data.panel import Panel
from nifty_quant.strategy.base import (
    DataRequest,
    Strategy,
    TargetPortfolio,
)

_IST = ZoneInfo("Asia/Kolkata")


def _session_grid(
    bars_per_day: tuple[int, ...] | list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build timestamps, explicit offsets, and dates for irregular sessions."""
    bars_per_day = tuple(int(n) for n in bars_per_day)

    dates: list[dt.date] = []
    current = dt.date(2024, 1, 2)
    while len(dates) < len(bars_per_day):
        if current.weekday() < 5:
            dates.append(current)
        current += dt.timedelta(days=1)

    ts_chunks: list[np.ndarray] = []
    for session_date, n_bars in zip(dates, bars_per_day):
        session_start = dt.datetime.combine(
            session_date,
            dt.time(9, 15),
            tzinfo=_IST,
        )
        ts_chunks.append(
            np.asarray(
                [
                    int(
                        (
                            session_start + dt.timedelta(minutes=bar_number)
                        )
                        .astimezone(dt.timezone.utc)
                        .timestamp()
                    )
                    for bar_number in range(n_bars)
                ],
                dtype=np.int64,
            )
        )

    if ts_chunks:
        ts = np.concatenate(ts_chunks).astype(np.int64, copy=False)
    else:
        ts = np.empty(0, dtype=np.int64)

    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.asarray(bars_per_day, dtype=np.int64)),
        )
    ).astype(np.int32)

    return ts, offsets, np.asarray(dates, dtype=object)


def _make_panel(
    bars_per_day: tuple[int, ...] | list[int],
) -> Panel:
    """Create float32-at-rest synthetic OHLCV data with varied price paths."""
    ts, day_offsets, dates = _session_grid(bars_per_day)

    close_chunks: list[np.ndarray] = []
    for session_index, n_bars in enumerate(bars_per_day):
        bar_number = np.arange(int(n_bars), dtype=np.float64)
        direction = 1.0 if session_index % 2 == 0 else -1.0
        close_chunks.append(
            100.0
            + 2.0 * session_index
            + direction * 0.08 * bar_number
            + 0.7 * np.sin((bar_number + 11.0 * session_index) / 19.0)
        )

    if close_chunks:
        close = np.concatenate(close_chunks).astype(np.float64, copy=False)
    else:
        close = np.empty(0, dtype=np.float64)
    close = close.reshape(-1, 1)

    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.full(close.shape, 1_000_000.0, dtype=np.float64)

    fields = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(
        fields=fields,
        symbols=("AAA",),
        ts=ts,
        day_offsets=day_offsets,
        dates=dates,
    )


class _TestParams(BaseModel):
    pass


class _FixedWeightStrategy(Strategy):
    name = "daily-results-test-fixed-weight"
    Params = _TestParams

    def __init__(
        self,
        decision_times: tuple[str, ...] | None,
        weights: tuple[float, ...] = (0.5,),
    ) -> None:
        super().__init__(_TestParams())
        self._decision_times = (
            None if decision_times is None else tuple(decision_times)
        )
        self._weights = np.asarray(weights, dtype=np.float64)

    def data_request(self) -> DataRequest:
        return DataRequest(
            freq="1",
            decision_times=self._decision_times,
        )

    def precompute(self, panel: Panel) -> dict[str, np.ndarray]:
        return {}

    def on_decision(self, view, signals, state) -> TargetPortfolio:
        return TargetPortfolio(weights=self._weights.copy())


def _run_fixed(
    panel: Panel,
    decision_times: tuple[str, ...] | None,
) -> tuple[object, _FixedWeightStrategy]:
    strategy = _FixedWeightStrategy(decision_times)
    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(capital=1_000_000.0),
    )
    return result, strategy


def _decision_and_final_rows(
    panel: Panel,
    decision_times: tuple[str, ...] | None,
) -> np.ndarray:
    """Reproduce the old CLI row-selection rule without importing cli.py."""
    n_rows = panel.n_rows()
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    if decision_times is None:
        decision_rows = np.arange(n_rows, dtype=np.int64)
    else:
        per_time = [
            np.asarray(panel.rows_at_time(time_string), dtype=np.int64)
            for time_string in decision_times
        ]
        if per_time:
            decision_rows = np.unique(np.concatenate(per_time))
        else:
            decision_rows = np.empty(0, dtype=np.int64)

    decision_rows = decision_rows[decision_rows != 0]
    final_row = np.asarray([n_rows - 1], dtype=np.int64)
    # cli.py's real helper concatenates the final row unconditionally (not a set union):
    # if the panel's absolute last row is also a decision row, the real engine appends
    # its equity/turnover TWICE (once inside the decision-row branch of the loop, once
    # more unconditionally after the loop). Deduping here would silently under-count
    # relative to the real `result.returns`/`result.turnover` length in that case.
    return np.concatenate((decision_rows, final_row)).astype(np.int64, copy=False)


def _old_reconstruction(
    returns: np.ndarray,
    panel: Panel,
    decision_times: tuple[str, ...] | None,
) -> np.ndarray:
    """Compound result.returns using the old reconstructed row/day mapping."""
    returns = np.asarray(returns, dtype=np.float64)
    row_indices = _decision_and_final_rows(panel, decision_times)
    if row_indices.size != returns.size:
        raise AssertionError(
            "The reconstructed row count does not match result.returns"
        )

    day_indices = (
        np.searchsorted(
            panel.day_offsets,
            row_indices,
            side="right",
        )
        - 1
    ).astype(np.int64, copy=False)
    return aggregate_returns_by_group(returns, day_indices)


def _expected_epoch_dates(panel: Panel) -> np.ndarray:
    """Convert panel session dates independently to UTC-midnight epoch seconds."""
    return np.asarray(
        [
            int(
                dt.datetime.combine(
                    session_date,
                    dt.time.min,
                    tzinfo=dt.timezone.utc,
                ).timestamp()
            )
            for session_date in panel.dates
        ],
        dtype=np.int64,
    )


def test_1_daily_returns_match_old_reconstruction() -> None:
    panel = _make_panel((375, 375, 375))
    result, strategy = _run_fixed(panel, ("10:00",))

    expected = _old_reconstruction(
        result.returns,
        panel,
        strategy.data_request().decision_times,
    )

    np.testing.assert_allclose(
        result.daily.returns,
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_2_turnover_sums_returns_compound() -> None:
    # Test the aggregation rule directly. Exact engine-generated 1% legs and
    # 0.1/0.2 turnover are not guaranteed under the real cost/fill models, and
    # the specification explicitly permits a DailyResult-shaped aggregation
    # scenario for this item.
    decision_returns = np.asarray([0.01, 0.01], dtype=np.float64)
    decision_turnover = np.asarray([0.1, 0.2], dtype=np.float64)
    group_ids = np.zeros(2, dtype=np.int64)

    compounded_returns = aggregate_returns_by_group(
        decision_returns,
        group_ids,
    )
    summed_turnover = np.add.reduceat(
        decision_turnover,
        np.asarray([0], dtype=np.int64),
    )

    daily = DailyResult(
        dates=np.asarray([0], dtype=np.int64),
        equity=np.asarray([1.0201], dtype=np.float64),
        returns=compounded_returns.astype(np.float64, copy=False),
        gross_returns=np.asarray([0.0], dtype=np.float64),
        turnover=summed_turnover.astype(np.float64, copy=False),
        n_days=1,
    )

    np.testing.assert_allclose(
        daily.returns,
        np.asarray([1.01 * 1.01 - 1.0], dtype=np.float64),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        daily.turnover,
        np.asarray([0.3], dtype=np.float64),
        rtol=0,
        atol=1e-15,
    )

    assert daily.returns[0] != decision_returns.sum()
    assert daily.turnover[0] != (
        (1.0 + decision_turnover[0])
        * (1.0 + decision_turnover[1])
        - 1.0
    )


def test_3_daily_equity_is_end_of_day() -> None:
    panel = _make_panel((375, 375, 375, 375))
    result, strategy = _run_fixed(panel, ("10:00",))

    row_indices = _decision_and_final_rows(
        panel,
        strategy.data_request().decision_times,
    )
    day_indices = (
        np.searchsorted(
            panel.day_offsets,
            row_indices,
            side="right",
        )
        - 1
    ).astype(np.int64, copy=False)

    for session_index in range(len(panel.dates)):
        result_positions = np.flatnonzero(day_indices == session_index)
        assert result_positions.size > 0

        expected_equity = result.equity_curve[result_positions[-1]]
        actual_equity = result.daily.equity[session_index]
        np.testing.assert_allclose(
            np.asarray([actual_equity], dtype=np.float64),
            np.asarray([expected_equity], dtype=np.float64),
            rtol=0,
            atol=1e-12,
        )


def test_4_irregular_sessions_do_not_use_fixed_stride() -> None:
    bars_per_day = (375, 375, 60, 105, 375)
    panel = _make_panel(bars_per_day)
    result, strategy = _run_fixed(panel, ("10:00",))

    assert len(result.daily.dates) == 5
    assert len(result.daily.returns) == 5
    assert len(result.daily.gross_returns) == 5
    assert len(result.daily.equity) == 5
    assert len(result.daily.turnover) == 5

    expected_dates = _expected_epoch_dates(panel)
    np.testing.assert_array_equal(result.daily.dates, expected_dates)
    np.testing.assert_array_equal(
        result.daily.dates,
        np.sort(result.daily.dates),
    )

    expected_returns = _old_reconstruction(
        result.returns,
        panel,
        strategy.data_request().decision_times,
    )
    np.testing.assert_allclose(
        result.daily.returns,
        expected_returns,
        rtol=0,
        atol=1e-12,
    )


def test_5_missing_session_has_zero_and_carried_equity() -> None:
    # Both irregular sessions are deliberately present.  15:00 is covered by
    # the regular 375-bar sessions but by neither the 60-bar nor 105-bar one.
    panel = _make_panel((375, 60, 105, 375))
    result, strategy = _run_fixed(panel, ("15:00",))

    assert len(result.daily.dates) == len(panel.dates)
    assert len(result.daily.returns) == len(panel.dates)
    assert len(result.daily.gross_returns) == len(panel.dates)
    assert len(result.daily.equity) == len(panel.dates)
    assert len(result.daily.turnover) == len(panel.dates)

    row_indices = _decision_and_final_rows(
        panel,
        strategy.data_request().decision_times,
    )
    day_indices = (
        np.searchsorted(
            panel.day_offsets,
            row_indices,
            side="right",
        )
        - 1
    ).astype(np.int64, copy=False)
    np.testing.assert_array_equal(
        np.unique(day_indices),
        np.asarray([0, 3], dtype=np.int64),
    )

    for session_index in (1, 2):
        assert result.daily.returns[session_index] == 0.0
        assert result.daily.turnover[session_index] == 0.0
        assert (
            result.daily.equity[session_index]
            == result.daily.equity[session_index - 1]
        )


def test_6_daily_n_days_equals_panel_dates() -> None:
    panel = _make_panel((375, 375, 375, 375, 375))
    result, _ = _run_fixed(panel, ("10:00",))

    assert result.daily.n_days == len(panel.dates)
    assert result.daily.n_days == len(result.daily.dates)


def test_7_daily_metrics_unchanged_from_old_reconstruction() -> None:
    panel = _make_panel((375, 375, 375, 375, 375, 375, 375))
    result, strategy = _run_fixed(panel, ("10:00", "14:00"))

    old_reconstructed_returns = _old_reconstruction(
        result.returns,
        panel,
        strategy.data_request().decision_times,
    )

    daily_metrics = compute_metrics(result.daily.returns)
    old_metrics = compute_metrics(old_reconstructed_returns)

    assert daily_metrics == old_metrics


def test_8_empty_panel_and_empty_returns_produce_empty_daily_result() -> None:
    # The stated Panel invariants permit a truly empty panel with no sessions:
    # day_offsets has its required initial/final zero and dates is empty.  Use
    # run_backtest so the empty-result path, rather than a private constructor
    # invented by this test, is exercised.
    empty_panel = Panel(
        fields={
            "open": np.empty((0, 1), dtype=np.float32),
            "high": np.empty((0, 1), dtype=np.float32),
            "low": np.empty((0, 1), dtype=np.float32),
            "close": np.empty((0, 1), dtype=np.float32),
            "volume": np.empty((0, 1), dtype=np.float32),
        },
        symbols=("AAA",),
        ts=np.empty(0, dtype=np.int64),
        day_offsets=np.asarray([0], dtype=np.int32),
        dates=np.empty(0, dtype=object),
    )

    result, _ = _run_fixed(empty_panel, ("10:00",))

    assert np.asarray(result.returns).size == 0
    assert result.daily.n_days == 0
    assert result.daily.dates.size == 0
    assert result.daily.equity.size == 0
    assert result.daily.returns.size == 0
    assert result.daily.gross_returns.size == 0
    assert result.daily.turnover.size == 0
    assert result.daily.dates.dtype == np.dtype(np.int64)
    for values in (
        result.daily.equity,
        result.daily.returns,
        result.daily.gross_returns,
        result.daily.turnover,
    ):
        assert values.dtype == np.dtype(np.float64)


def test_9_daily_dates_are_panel_session_dates() -> None:
    panel = _make_panel((375, 60, 105, 375))
    result, _ = _run_fixed(panel, ("15:00",))

    expected_dates = _expected_epoch_dates(panel)
    assert result.daily.dates.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result.daily.dates, expected_dates)

    # The missing middle sessions make this distinguishable from the deleted
    # repeat-day-zero fallback.  With ordinary synthetic sessions, deriving a
    # date from a row timestamp would otherwise share the same calendar date;
    # panel metadata plus the no-duplicates assertion tests the required path.
    assert np.unique(result.daily.dates).size == len(panel.dates)
    np.testing.assert_array_equal(
        result.daily.dates,
        np.sort(result.daily.dates),
    )
    assert result.daily.dates[0] == expected_dates[0]

