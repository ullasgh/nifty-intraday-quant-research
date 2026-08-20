"""Coverage for the `--min-history-sessions` point-in-time eligibility gate
wired into `nq backtest` and `nq walkforward` (specs/pit_universe.md item C).

Mirrors the monkeypatch pattern of tests/test_cli_tradable.py: `run_backtest`
is stubbed to capture the `tradable` array it was actually called with, so
these tests check the real CLI wiring (`_apply_pit_eligibility` -> the
Typer commands) rather than re-testing `compute_eligibility` itself (already
covered by tests/test_pit_universe_a.py, tests/test_pit_universe_b.py).
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.daily import DailyResult
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.universe.pit import compute_eligibility, eligibility_mask_to_bars
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


def _make_panel(session_dates, bars_per_session, symbols=("A", "B"), absent_before=None):
    """`absent_before`: dict symbol -> session index; that symbol's bars are
    NaN for every session strictly before that index (a synthetic new
    listing), matching the `_build_panel` convention used by
    tests/test_pit_universe_b.py."""
    absent_before = absent_before or {}
    ts = np.concatenate(
        [_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    close = np.full((n_rows, n_symbols), 100.5, dtype=np.float64)
    volume = np.full((n_rows, n_symbols), 1.0e7, dtype=np.float64)

    for col, sym in enumerate(symbols):
        first_session = absent_before.get(sym, 0)
        for session_idx in range(first_session):
            start = int(day_offsets[session_idx])
            end = int(day_offsets[session_idx + 1])
            close[start:end, col] = np.nan
            volume[start:end, col] = np.nan

    fields = {
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


def _fake_result(n=3):
    rng = np.random.default_rng(0)
    gross = rng.normal(-0.001, 0.005, size=n)
    turnover = np.full(n, 1.0)
    returns = gross - 0.0005 * turnover
    equity_curve = np.cumprod(1.0 + returns) * 1e7
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


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path, panel, calls, universe) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(universe_mod, "load_universe", lambda name: universe)
    monkeypatch.setattr(engine_mod, "run_backtest", _capturing_run_backtest(calls))
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


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
        "2024-01-03",
    ]


def test_backtest_without_flag_leaves_tradable_mask_unchanged(monkeypatch, tmp_path) -> None:
    session_dates = [datetime.date(2024, 1, i) for i in (1, 2, 3)]
    panel = _make_panel(session_dates, [5, 5, 5], absent_before={"B": 2})
    universe = Universe(name="ab", symbols=("A", "B"))
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls, universe)

    result = runner.invoke(app, _backtest_args())

    assert result.exit_code == 0, result.output
    for entry in calls:
        # Default tradable_filter=True path: unaffected by eligibility, since
        # --min-history-sessions was never given (opt-in only, rule 8).
        assert entry["tradable"] is not None
    assert "point-in-time eligibility" not in result.output


def test_backtest_min_history_sessions_excludes_late_listing_from_tradable(
    monkeypatch, tmp_path
) -> None:
    session_dates = [datetime.date(2024, 1, i) for i in (1, 2, 3)]
    bars_per_session = np.asarray([5, 5, 5])
    # B has no bars in session 0 -- a synthetic new listing.
    panel = _make_panel(session_dates, bars_per_session, absent_before={"B": 1})
    universe = Universe(name="ab", symbols=("A", "B"))
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls, universe)

    result = runner.invoke(
        app,
        [
            *_backtest_args(),
            "--no-tradable-filter",
            "--min-history-sessions",
            "1",
            "--min-adv-inr",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "point-in-time eligibility" in result.output

    expected_eligibility = compute_eligibility(
        panel, universe, min_history_sessions=1, min_adv_inr=0.0
    )
    expected_bars = eligibility_mask_to_bars(panel, expected_eligibility)

    assert calls, "run_backtest was never invoked"
    for entry in calls:
        assert entry["tradable"] is not None
        np.testing.assert_array_equal(entry["tradable"], expected_bars)

    # B is genuinely excluded (not merely present-with-zero-weight) on session 0:
    b_col = panel.symbols.index("B")
    start, end = int(panel.day_offsets[0]), int(panel.day_offsets[1])
    assert not np.any(calls[0]["tradable"][start:end, b_col])


def test_backtest_min_history_sessions_intersects_with_tradable_filter(
    monkeypatch, tmp_path
) -> None:
    session_dates = [datetime.date(2024, 1, i) for i in (1, 2, 3)]
    bars_per_session = np.asarray([5, 5, 5])
    panel = _make_panel(session_dates, bars_per_session, absent_before={"B": 1})
    universe = Universe(name="ab", symbols=("A", "B"))
    calls = []
    _patch_common(monkeypatch, tmp_path, panel, calls, universe)

    result = runner.invoke(
        app,
        [
            *_backtest_args(),
            "--min-history-sessions",
            "1",
            "--min-adv-inr",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output

    from nifty_quant.data.validate import tradable_mask as real_tradable_mask

    expected_tradable = real_tradable_mask(panel, min_adv_inr=0.0)
    expected_eligibility = compute_eligibility(
        panel, universe, min_history_sessions=1, min_adv_inr=0.0
    )
    expected_bars = eligibility_mask_to_bars(panel, expected_eligibility)
    expected_combined = expected_tradable & expected_bars

    for entry in calls:
        np.testing.assert_array_equal(entry["tradable"], expected_combined)
