from __future__ import annotations

import datetime
import re
import types

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.daily import DailyResult
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import _tradable_mask_summary, app
from nifty_quant.data.panel import Panel
from nifty_quant.universe.static import Universe

runner = CliRunner()


def _session_ts(session_date, n_bars, start_hhmm=(9, 15)):
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(
        session_date, datetime.time(*start_hhmm), tzinfo=ist
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _make_panel(
    session_dates,
    bars_per_session,
    symbols=("A", "B"),
    close_price=100.5,
    volume_value=1000.0,
    nan_cells=(),
):
    """Build a synthetic multi-session panel."""
    ts = np.concatenate(
        [_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.cumsum(bars_per_session).astype(np.int32),
        ]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    fields = {
        "open": np.full(
            (n_rows, n_symbols), close_price - 0.5, dtype=np.float64
        ),
        "high": np.full(
            (n_rows, n_symbols), close_price + 0.5, dtype=np.float64
        ),
        "low": np.full(
            (n_rows, n_symbols), close_price - 1.0, dtype=np.float64
        ),
        "close": np.full((n_rows, n_symbols), close_price, dtype=np.float64),
        "volume": np.full(
            (n_rows, n_symbols), volume_value, dtype=np.float64
        ),
    }
    for row, col in nan_cells:
        for name in fields:
            fields[name][row, col] = np.nan
    return Panel(
        fields,
        symbols,
        ts,
        day_offsets,
        np.asarray(session_dates, dtype=object),
    )


def _fake_result(n=3):
    rng = np.random.default_rng(0)
    gross = rng.normal(-0.001, 0.005, size=n)
    turnover = np.full(n, 1.0)
    returns = gross - 0.0005 * turnover
    equity_curve = np.cumprod(1.0 + returns) * 1e7
    # `result.daily` is what the CLI now reads for its printed Sharpe/turnover
    # figures and for the `returns.parquet` artifact (it no longer calls a
    # `_daily_returns`/`_daily_turnover` reconstruction helper). This test suite
    # only asserts on tradable-mask plumbing, not on these numeric values, so a
    # trivial one-row-per-decision-row `DailyResult` is enough to keep the CLI's
    # code paths that read `result.daily.*` from raising.
    daily = DailyResult(
        dates=np.arange(n, dtype=np.int64),
        equity=equity_curve,
        returns=returns,
        gross_returns=gross,
        turnover=turnover,
        n_days=n,
    )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=np.zeros((n, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
        daily=daily,
    )


def _capturing_run_backtest(calls):
    def _stub(strat, panel, config, **kwargs):
        calls.append({"panel": panel, "tradable": kwargs.get("tradable")})
        return _fake_result()

    return _stub


class _FakeCalendar:
    def __init__(self, dates):
        self._dates = list(dates)

    def session_dates(self, start=None, end=None, *, usable_only=True):
        return [
            d
            for d in self._dates
            if (start is None or d >= start) and (end is None or d <= end)
        ]


ALL_DATES = pd.bdate_range("2024-01-01", periods=12).date.tolist()


def _backtest_args() -> list[str]:
    return [
        "backtest",
        "--strategy",
        "volume_breakout",
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
    ]


def _walkforward_args() -> list[str]:
    return [
        "walkforward",
        "--strategy",
        "volume_breakout",
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--start",
        ALL_DATES[0].isoformat(),
        "--end",
        ALL_DATES[-1].isoformat(),
        "--train-years",
        "0.012",
        "--test-years",
        "0.012",
    ]


def _patch_common(
    monkeypatch: pytest.MonkeyPatch, tmp_path, panel, calls
) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="ab", symbols=("A", "B")),
    )
    monkeypatch.setattr(
        engine_mod, "run_backtest", _capturing_run_backtest(calls)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


def _patch_walkforward(
    monkeypatch: pytest.MonkeyPatch, tmp_path, panel, calls
) -> None:
    import nifty_quant.calendar as calendar_mod

    _patch_common(monkeypatch, tmp_path, panel, calls)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(
            lambda cls, symbol="NIFTY50": _FakeCalendar(ALL_DATES)
        ),
    )


def test_tradable_mask_summary_reports_distinct_present_and_excluded_pcts() -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1)],
        [4],
        symbols=("A", "B"),
    )
    close = panel.field("close")
    close[1, 0] = np.nan
    close[3, 1] = np.nan

    present = ~np.isnan(close)
    mask = np.ones(close.shape, dtype=bool)
    mask[~present] = False
    mask[2, 0] = False

    assert not np.any(mask & np.isnan(close))

    total = present.size
    present_count = int(np.count_nonzero(present))
    tradable_count = int(np.count_nonzero(mask))
    excluded_count = total - tradable_count
    present_but_excluded = present_count - tradable_count
    excluded_pct = 100.0 * excluded_count / total
    present_pct = 100.0 * present_count / total
    present_but_excluded_pct = 100.0 * present_but_excluded / present_count

    summary = _tradable_mask_summary(panel, mask)

    for value in (
        excluded_pct,
        present_pct,
        present_but_excluded_pct,
    ):
        assert f"{value:.2f}%" in summary
    assert excluded_pct != present_pct
    assert present_but_excluded_pct > 0.0


def test_tradable_mask_summary_empty_panel_is_na() -> None:
    panel = types.SimpleNamespace(field=lambda name: np.empty((0, 0)))
    mask = np.empty((0, 0), dtype=bool)

    assert _tradable_mask_summary(panel, mask) == (
        "tradable filter: n/a (empty panel)"
    )


def test_backtest_default_passes_mask_to_every_run_backtest_call(
    monkeypatch, tmp_path
) -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [5, 5],
        symbols=("A", "B"),
        nan_cells=[(1, 0)],
    )
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    result = runner.invoke(app, _backtest_args())

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    for entry in calls:
        assert entry["tradable"] is not None
        assert entry["tradable"].shape == (
            entry["panel"].n_rows(),
            entry["panel"].n_symbols(),
        )
    assert "tradable filter: excluded" in result.output
    assert int(calls[0]["tradable"].sum()) == 0


def test_backtest_no_tradable_filter_passes_none(monkeypatch, tmp_path) -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [5, 5],
        symbols=("A", "B"),
        nan_cells=[(1, 0)],
    )
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    result = runner.invoke(
        app, [*_backtest_args(), "--no-tradable-filter"]
    )

    assert result.exit_code == 0, result.output
    for entry in calls:
        assert entry["tradable"] is None
    assert "tradable filter:" not in result.output


def test_backtest_min_adv_inr_is_honoured(monkeypatch, tmp_path) -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [5, 5],
        symbols=("A", "B"),
        nan_cells=[(1, 0)],
    )
    calls_default_threshold = []
    _patch_common(
        monkeypatch, tmp_path, panel, calls_default_threshold
    )

    result_default = runner.invoke(app, _backtest_args())

    import nifty_quant.backtest.engine as engine_mod

    calls_low_threshold = []
    monkeypatch.setattr(
        engine_mod,
        "run_backtest",
        _capturing_run_backtest(calls_low_threshold),
    )
    result_low = runner.invoke(
        app, [*_backtest_args(), "--min-adv-inr", "0"]
    )

    assert result_default.exit_code == 0, result_default.output
    assert result_low.exit_code == 0, result_low.output
    assert (
        calls_low_threshold[0]["tradable"].sum()
        > calls_default_threshold[0]["tradable"].sum()
    )
    assert calls_default_threshold[0]["tradable"].sum() == 0
    assert calls_low_threshold[0]["tradable"].sum() > 0


def test_walkforward_passes_distinct_masks_per_panel(
    monkeypatch, tmp_path
) -> None:
    panel = _make_panel(
        ALL_DATES,
        [5] * 12,
        symbols=("A", "B"),
    )
    calls = []
    _patch_walkforward(monkeypatch, tmp_path, panel, calls)

    result = runner.invoke(app, _walkforward_args())

    assert result.exit_code == 0, result.output
    assert len(calls) == 4
    for entry in calls:
        assert entry["tradable"] is not None
        assert entry["tradable"].shape == (
            entry["panel"].n_rows(),
            entry["panel"].n_symbols(),
        )
    assert calls[0]["panel"].n_rows() != calls[1]["panel"].n_rows()
    assert calls[0]["tradable"].shape != calls[1]["tradable"].shape
    assert calls[1]["panel"] is calls[2]["panel"] is calls[3]["panel"]
    assert (
        calls[1]["tradable"]
        is calls[2]["tradable"]
        is calls[3]["tradable"]
    )
    assert result.output.count("tradable filter: excluded") == 1


def test_walkforward_no_tradable_filter_passes_none(
    monkeypatch, tmp_path
) -> None:
    panel = _make_panel(
        ALL_DATES,
        [5] * 12,
        symbols=("A", "B"),
    )
    calls = []
    _patch_walkforward(monkeypatch, tmp_path, panel, calls)

    result = runner.invoke(
        app, [*_walkforward_args(), "--no-tradable-filter"]
    )

    assert result.exit_code == 0, result.output
    for entry in calls:
        assert entry["tradable"] is None
    assert "tradable filter:" not in result.output


def test_present_and_tradable_are_not_conflated_in_cli_output(
    monkeypatch, tmp_path
) -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [5, 5],
        symbols=("A", "B"),
        nan_cells=[(1, 0)],
    )
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    result = runner.invoke(app, _backtest_args())

    assert result.exit_code == 0, result.output
    summary = next(
        line
        for line in result.output.splitlines()
        if line.startswith("tradable filter: excluded ")
    )
    excluded_match = re.search(r"excluded ([0-9.]+)%", summary)
    present_match = re.search(r"present=([0-9.]+)%", summary)
    assert excluded_match is not None
    assert present_match is not None
    assert float(excluded_match.group(1)) != float(present_match.group(1))
