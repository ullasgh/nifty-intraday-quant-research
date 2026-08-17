"""Tests for the walkforward pooled-turnover-basis defect (cli.py DEFECT 1).

Before the fix, `walkforward` pooled `_daily_returns`-aggregated (per trading day) gross/net
return series together with RAW per-decision-row `res.turnover`, so `pooled_turnover` and
`pooled_gross`/`pooled_net` had different lengths. `breakeven_cost_bps(pooled_gross,
pooled_turnover)` then raised `ValueError: gross_returns and turnover must have the same
length`, and the command crashed on any real (unstubbed) `_daily_returns`/`_daily_turnover`
run. It was masked in the rest of the suite only because `tests/test_cli_tradable.py`
monkeypatches `cli._daily_returns` to a fixed-length stub.

These tests exercise the REAL `_daily_returns`/`_daily_turnover` reconstruction end to end
(only `run_backtest` is stubbed with a deterministic, correctly-sized fake result), so they
would fail against the pre-fix code.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import _daily_returns, _daily_turnover, app
from nifty_quant.data.panel import Panel
from nifty_quant.universe.static import Universe

runner = CliRunner()


def _session_ts(
    session_date: date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(session_date, time(*start_hhmm), tzinfo=ist)
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _make_panel(
    session_dates: list[date],
    bars_per_session: list[int],
    symbols: tuple[str, ...] = ("A", "B"),
    close_price: float = 100.5,
    volume_value: float = 1000.0,
) -> Panel:
    ts = np.concatenate(
        [_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    fields = {
        "open": np.full((n_rows, n_symbols), close_price - 0.5, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), close_price + 0.5, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), close_price - 1.0, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), close_price, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), volume_value, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


def _fake_result(n_rows: int, seed: int = 0) -> BacktestResult:
    """Deterministic fake `BacktestResult` correctly sized to `panel.n_rows()`.

    Unlike `tests/test_cli_tradable.py`'s fixed-length stub, this always matches the real
    per-decision-row reconstruction, so the real (unstubbed) `_daily_returns`/
    `_daily_turnover` in `cli.py` can run against it without a length mismatch.
    """
    rng = np.random.default_rng(seed)
    gross = rng.normal(loc=0.0006, scale=0.004, size=n_rows).astype(np.float64)
    turnover = np.full(n_rows, 1.0, dtype=np.float64)
    returns = (gross - 0.0002 * turnover).astype(np.float64)
    return BacktestResult(
        equity_curve=np.cumprod(1.0 + returns) * 1e7,
        returns=returns,
        positions=np.zeros((n_rows, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
        ruined=False,
        ruin_index=-1,
    )


def _capturing_run_backtest(calls: list[dict[str, Any]]) -> Callable[..., BacktestResult]:
    def _stub(strat: Any, panel: Panel, config: Any, **kwargs: Any) -> BacktestResult:
        result = _fake_result(panel.n_rows())
        calls.append({"panel": panel, "result": result})
        return result

    return _stub


class _FakeCalendar:
    def __init__(self, dates: list[date]) -> None:
        self._dates = list(dates)

    def session_dates(
        self,
        start: date | None = None,
        end: date | None = None,
        *,
        usable_only: bool = True,
    ) -> list[date]:
        return [
            d
            for d in self._dates
            if (start is None or d >= start) and (end is None or d <= end)
        ]


def _walkforward_args(all_dates: list[date]) -> list[str]:
    return [
        "walkforward",
        "--strategy",
        "volume_breakout",
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--start",
        all_dates[0].isoformat(),
        "--end",
        all_dates[-1].isoformat(),
        "--train-years",
        "0.012",
        "--test-years",
        "0.012",
    ]


def _run_walkforward_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    all_dates: list[date],
) -> tuple[Any, list[dict[str, Any]]]:
    """Invoke `nq walkforward` for real, with REAL `_daily_returns`/`_daily_turnover`.

    Only `run_backtest`, `load_panel`, `load_universe`, `TradingCalendar.from_index_bars`,
    and `settings.RESULTS_ROOT` are stubbed -- everything else, including the day-basis
    reconstruction under test, runs unmodified.
    """
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="ab", symbols=panel.symbols),
    )
    monkeypatch.setattr(engine_mod, "run_backtest", _capturing_run_backtest(calls))
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)),
    )

    result = runner.invoke(app, _walkforward_args(all_dates))
    return result, calls


def _all_dates_280() -> list[date]:
    return pd.bdate_range("2020-01-01", periods=280).date.tolist()


def _split_calls(calls: list[dict[str, Any]], full_panel: Panel) -> list[dict[str, Any]]:
    """Filter `run_backtest` calls down to the walk-forward split evaluations.

    `walkforward` also calls `run_backtest` three more times, on the FULL, untrimmed
    panel, for its latency-sensitivity block -- those calls must not be mistaken for
    pooled splits. `Panel.sub(...)` (used to build each split's `test_panel`) always
    returns a NEW object when given date bounds, so identity comparison against the
    full panel reliably distinguishes the two.
    """
    return [call for call in calls if call["panel"] is not full_panel]


def test_daily_turnover_sums_within_session_while_daily_returns_compound() -> None:
    """Unit-level: turnover sums per day, returns compound per day, on the same day basis.

    Two sessions (3 bars, 4 bars); decision_times=None reconstructs rows 1..6 plus the
    duplicated final row 6, mapping rows {1,2} to day 0 and {3,4,5,6,6} to day 1. Expected
    values below are computed independently of `aggregate_returns_by_group`/`_daily_turnover`
    by hand, so this is not a tautological restatement of the implementation.
    """
    panel = _make_panel([date(2024, 1, 2), date(2024, 1, 3)], [3, 4])
    returns = np.array([0.01, 0.02, 0.03, -0.01, 0.005, 0.0, 0.02], dtype=np.float64)
    turnover = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)

    daily_ret = _daily_returns(returns, panel, None)
    daily_to = _daily_turnover(turnover, panel, None)

    assert daily_ret.shape == daily_to.shape == (2,)

    expected_ret_day0 = (1.01 * 1.02) - 1.0
    expected_ret_day1 = (1.03 * 0.99 * 1.005 * 1.0 * 1.02) - 1.0
    expected_to_day0 = 0.1 + 0.2
    expected_to_day1 = 0.3 + 0.4 + 0.5 + 0.6 + 0.7

    assert daily_ret[0] == pytest.approx(expected_ret_day0, abs=1e-9)
    assert daily_ret[1] == pytest.approx(expected_ret_day1, abs=1e-9)
    assert daily_to[0] == pytest.approx(expected_to_day0, abs=1e-9)
    assert daily_to[1] == pytest.approx(expected_to_day1, abs=1e-9)

    # Turnover must NOT be compounded (that would be meaningless for an additive quantity):
    # a naive compound of the day-1 turnover values would differ from the correct sum.
    naive_compound_day1 = (1.3 * 1.4 * 1.5 * 1.6 * 1.7) - 1.0
    assert daily_to[1] != pytest.approx(naive_compound_day1, abs=1e-9)


def test_walkforward_exits_zero_with_real_daily_returns_on_multi_session_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The core regression check: pre-fix, this crashes with a length-mismatch ValueError
    from `breakeven_cost_bps(pooled_gross, pooled_turnover)` as soon as `_daily_returns` is
    NOT stubbed to a fixed length -- i.e. on any real data.
    """
    all_dates = _all_dates_280()
    panel = _make_panel(all_dates, [2] * len(all_dates))

    result, calls = _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    assert result.exit_code == 0, result.output
    split_calls = _split_calls(calls, panel)
    assert len(split_calls) >= 2, "expected at least two walk-forward splits for this date range"


def test_pooled_gross_and_pooled_turnover_have_equal_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reconstructs, independently of `walkforward`'s internals, what each split's daily
    return and daily turnover series would be, and asserts they are always equal length --
    which is exactly the invariant `pooled_gross`/`pooled_turnover` need to hold after
    concatenation for `breakeven_cost_bps` to not crash.
    """
    all_dates = _all_dates_280()
    panel = _make_panel(all_dates, [2] * len(all_dates))

    result, calls = _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    assert result.exit_code == 0, result.output
    split_calls = _split_calls(calls, panel)
    assert len(split_calls) >= 2

    total_days = 0
    for call in split_calls:
        split_panel = call["panel"]
        split_result = call["result"]
        split_daily_net = _daily_returns(split_result.returns, split_panel, None)
        split_daily_turnover = _daily_turnover(split_result.turnover, split_panel, None)
        assert split_daily_net.shape == split_daily_turnover.shape
        total_days += split_daily_net.size

    match = re.search(r"\[daily returns, n_days=(\d+)\]", result.output)
    assert match is not None, result.output
    assert int(match.group(1)) == total_days


def test_pooled_mean_turnover_is_reported_on_daily_basis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The printed `mean turnover:` figure must be the mean of day-SUMMED turnover, not the
    mean of raw per-decision-row turnover (a different, smaller basis since each day sums
    several decision rows).
    """
    all_dates = _all_dates_280()
    panel = _make_panel(all_dates, [2] * len(all_dates))

    result, calls = _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    assert result.exit_code == 0, result.output
    split_calls = _split_calls(calls, panel)
    assert len(split_calls) >= 2

    daily_turnovers = [
        _daily_turnover(call["result"].turnover, call["panel"], None) for call in split_calls
    ]
    expected_pooled_turnover = np.concatenate(daily_turnovers)
    expected_mean = float(np.mean(expected_pooled_turnover))

    # Sanity check the two bases really differ here (otherwise this test would not
    # distinguish a correct fix from the pre-fix raw-per-row basis).
    raw_turnovers = np.concatenate([call["result"].turnover for call in split_calls])
    assert expected_mean != pytest.approx(float(np.mean(raw_turnovers)), abs=1e-9)

    match = re.search(r"mean turnover: ([\-0-9.]+)", result.output)
    assert match is not None, result.output
    reported_mean = float(match.group(1))
    assert reported_mean == pytest.approx(expected_mean, abs=5e-5)


def test_multi_split_pooling_concatenates_correctly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With several splits, the pooled series length must equal the sum of each split's
    own daily length, in split order -- not truncated, not double-counted, not reordered.
    """
    all_dates = _all_dates_280()
    panel = _make_panel(all_dates, [2] * len(all_dates))

    result, calls = _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    assert result.exit_code == 0, result.output
    split_calls = _split_calls(calls, panel)
    assert len(split_calls) >= 2, "need multiple splits to test pooling concatenation"

    per_split_daily_net_sizes = [
        _daily_returns(call["result"].returns, call["panel"], None).size for call in split_calls
    ]
    per_split_daily_turnover_sizes = [
        _daily_turnover(call["result"].turnover, call["panel"], None).size
        for call in split_calls
    ]
    assert per_split_daily_net_sizes == per_split_daily_turnover_sizes

    expected_total = sum(per_split_daily_net_sizes)
    match = re.search(r"\[daily returns, n_days=(\d+)\]", result.output)
    assert match is not None, result.output
    assert int(match.group(1)) == expected_total
