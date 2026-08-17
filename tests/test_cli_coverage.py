"""Coverage-closing tests for ``nifty_quant.cli``.

Companion to ``tests/test_cli.py`` and ``tests/test_cli_tradable.py``; targets the
remaining lines/branches those two files don't exercise (error paths, cache
commands, the daily-return/turnover/timestamp reconstruction helpers' edge
cases, and the ``build_panel``/``validate``/``strategies``/``sweep`` command
bodies). Never touches real ``data/`` -- every loader that would otherwise read
real market data is monkeypatched.
"""

from __future__ import annotations

import datetime
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import (
    _daily_return_ts,
    _daily_returns,
    _daily_turnover,
    _decision_and_final_rows,
    _parse_date,
    app,
)
from nifty_quant.data.panel import Panel
from nifty_quant.universe.static import Universe

runner = CliRunner()


def _make_panel(
    n_sessions: int = 3,
    bars_per_session: int = 2,
    dates: tuple[datetime.date, ...] | None = None,
) -> Panel:
    if dates is None:
        base = datetime.date(2024, 1, 1)
        dates = tuple(base + datetime.timedelta(days=i) for i in range(n_sessions))
    ts_list: list[int] = []
    day_offsets = [0]
    for i in range(n_sessions):
        for j in range(bars_per_session):
            ts_list.append(1_700_000_000 + i * 86_400 + j * 60)
        day_offsets.append(len(ts_list))
    return Panel(
        fields={},
        symbols=(),
        ts=np.asarray(ts_list, dtype=np.int64),
        day_offsets=np.asarray(day_offsets, dtype=np.int64),
        dates=np.asarray(dates, dtype=object),
    )


def test_parse_date_invalid() -> None:
    with pytest.raises(Exception):
        _parse_date("not-a-date", "--start")


def test_decision_and_final_rows_empty_panel() -> None:
    panel = Panel(
        fields={},
        symbols=(),
        ts=np.empty(0, dtype=np.int64),
        day_offsets=np.array([0], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    result = _decision_and_final_rows(panel, None)
    assert result.size == 0


def test_decision_and_final_rows_decision_times_nonempty() -> None:
    panel = _make_panel(n_sessions=3, bars_per_session=2)
    result = _decision_and_final_rows(panel, ("09:15",))
    assert result.size >= 1
    assert result[-1] == panel.ts.shape[0] - 1


def test_decision_and_final_rows_decision_times_empty_tuple() -> None:
    panel = _make_panel(n_sessions=3, bars_per_session=2)
    result = _decision_and_final_rows(panel, ())
    assert result.size == 1
    assert result[0] == panel.ts.shape[0] - 1


def test_daily_returns_empty_returns() -> None:
    panel = _make_panel()
    result = _daily_returns(np.empty(0, dtype=np.float64), panel, None)
    assert result.size == 0


def test_daily_returns_empty_panel_nonempty_returns() -> None:
    panel = Panel(
        fields={},
        symbols=(),
        ts=np.empty(0, dtype=np.int64),
        day_offsets=np.array([0], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    returns = np.array([0.01, -0.02], dtype=np.float64)
    result = _daily_returns(returns, panel, None)
    assert result.size == 0


def test_daily_returns_length_mismatch_raises() -> None:
    panel = _make_panel(n_sessions=2, bars_per_session=2)
    returns = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    with pytest.raises(ValueError, match="length mismatch"):
        _daily_returns(returns, panel, None)


def test_daily_turnover_empty_turnover() -> None:
    panel = _make_panel()
    result = _daily_turnover(np.empty(0, dtype=np.float64), panel, None)
    assert result.size == 0


def test_daily_turnover_empty_panel_nonempty_turnover() -> None:
    panel = Panel(
        fields={},
        symbols=(),
        ts=np.empty(0, dtype=np.int64),
        day_offsets=np.array([0], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    turnover = np.array([0.1, 0.2], dtype=np.float64)
    result = _daily_turnover(turnover, panel, None)
    assert result.size == 0


def test_daily_turnover_length_mismatch_raises() -> None:
    panel = _make_panel(n_sessions=2, bars_per_session=2)
    turnover = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="length mismatch"):
        _daily_turnover(turnover, panel, None)


def test_daily_return_ts_n_days_zero() -> None:
    panel = _make_panel()
    result = _daily_return_ts(panel, None, 0)
    assert result.size == 0


def test_daily_return_ts_empty_panel_nonzero_n_days() -> None:
    panel = Panel(
        fields={},
        symbols=(),
        ts=np.empty(0, dtype=np.int64),
        day_offsets=np.array([0], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    result = _daily_return_ts(panel, None, 3)
    assert result.size == 0


def test_daily_return_ts_row_indices_empty_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.cli as cli_mod

    panel = _make_panel(n_sessions=3, bars_per_session=2)
    monkeypatch.setattr(
        cli_mod,
        "_decision_and_final_rows",
        lambda panel, decision_times: np.empty(0, dtype=np.int64),
    )
    result = _daily_return_ts(panel, None, 3)
    assert result.size == 3


def test_daily_return_ts_fallback_no_padding() -> None:
    panel = _make_panel(n_sessions=5, bars_per_session=1)
    result = _daily_return_ts(panel, None, 3)
    assert result.size == 3


def test_daily_return_ts_fallback_padding_empty_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nifty_quant.cli as cli_mod

    panel = Panel(
        fields={},
        symbols=(),
        ts=np.array([1_700_000_000, 1_700_000_060], dtype=np.int64),
        day_offsets=np.array([0, 1, 2], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    monkeypatch.setattr(
        cli_mod,
        "_decision_and_final_rows",
        lambda panel, decision_times: np.array([0, 1], dtype=np.int64),
    )
    result = _daily_return_ts(panel, None, 5)
    assert result.size == 5
    assert np.all(result == 0)


def test_daily_return_ts_final_empty_dates_returns_zeros() -> None:
    panel = Panel(
        fields={},
        symbols=(),
        ts=np.array([1_700_000_000, 1_700_000_060], dtype=np.int64),
        day_offsets=np.array([0, 1, 2], dtype=np.int64),
        dates=np.empty(0, dtype=object),
    )
    result = _daily_return_ts(panel, None, 2)
    assert result.size == 2
    assert np.all(result == 0)


def test_symbols_command_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.universe.static as static_mod

    def raise_error() -> tuple[str, ...]:
        raise RuntimeError("boom")

    monkeypatch.setattr(static_mod, "all_data_symbols", raise_error)
    result = runner.invoke(app, ["symbols"])
    assert result.exit_code != 0
    assert "symbols: boom" in result.output


def test_symbols_command_no_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.manifest as manifest_mod
    import nifty_quant.universe.static as static_mod

    fake_manifest = types.SimpleNamespace(coverage={})
    monkeypatch.setattr(manifest_mod.Manifest, "load", classmethod(lambda cls: fake_manifest))
    monkeypatch.setattr(static_mod, "all_data_symbols", lambda: ("RELIANCE", "TCS"))
    monkeypatch.setattr(static_mod, "equity_symbols", lambda: ("RELIANCE", "TCS"))
    result = runner.invoke(app, ["symbols"])
    assert result.exit_code == 0
    assert "coverage: no data" in result.output


def test_build_panel_invalid_years() -> None:
    result = runner.invoke(app, ["build-panel", "--years", "abc"])
    assert result.exit_code != 0
    assert "Invalid --years" in result.output


def test_build_panel_valid_no_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    captured: dict[str, object] = {}

    def fake_build_panel(**kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return [Path("/fake/2023"), Path("/fake/2024")]

    monkeypatch.setattr(pb_mod, "build_panel", fake_build_panel)
    result = runner.invoke(app, ["build-panel", "--years", "2023,2024"])
    assert result.exit_code == 0
    assert captured["symbols"] is None
    assert "/fake/2023" in result.output
    assert "/fake/2024" in result.output
    assert "built 2 year-cache(s)" in result.output


def test_build_panel_with_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    captured: dict[str, object] = {}

    def fake_build_panel(**kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return [Path("/fake/2023")]

    monkeypatch.setattr(pb_mod, "build_panel", fake_build_panel)
    result = runner.invoke(app, ["build-panel", "--years", "2023", "--symbols", "RELIANCE,TCS"])
    assert result.exit_code == 0
    assert captured["symbols"] == ("RELIANCE", "TCS")


def test_build_panel_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    def raise_error(**kwargs: object) -> list[Path]:
        raise RuntimeError("build boom")

    monkeypatch.setattr(pb_mod, "build_panel", raise_error)
    result = runner.invoke(app, ["build-panel", "--years", "2023"])
    assert result.exit_code != 0
    assert "build-panel failed: build boom" in result.output


def test_build_panel_empty_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    monkeypatch.setattr(pb_mod, "build_panel", lambda **kwargs: [])
    result = runner.invoke(app, ["build-panel", "--years", "2023"])
    assert result.exit_code == 0
    assert "built 0 year-cache(s)" in result.output


def _fake_validate_report() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        findings=(),
        n_rows=100,
        n_symbols=2,
        n_sessions=5,
        errors=lambda: (),
        to_json=lambda: '{"findings": []}',
        summary=lambda: "fake summary",
    )


def _patch_validate_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nifty_quant.calendar as cal_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.data.validate as validate_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: object())
    monkeypatch.setattr(
        cal_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": object()),
    )
    monkeypatch.setattr(
        validate_mod,
        "validate_panel",
        lambda panel, calendar: _fake_validate_report(),
    )


def test_validate_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validate_deps(monkeypatch)
    result = runner.invoke(app, ["validate", "--year", "2024"])
    assert result.exit_code == 0
    assert "checks run: " in result.output
    assert "check_timestamps" in result.output
    assert "check_session_lengths" in result.output
    assert "check_ohlc_consistency" in result.output
    assert "check_zero_or_negative_prices" in result.output
    assert "check_stale_bars" in result.output
    assert "check_unexplained_gaps" in result.output
    assert "check_all_nan_columns" in result.output
    assert "check_volume_sanity" in result.output
    assert "fake summary" in result.output
    assert "errors: 0 / findings: 0" in result.output


def test_validate_with_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validate_deps(monkeypatch)
    result = runner.invoke(app, ["validate", "--year", "2024", "--symbols", "RELIANCE,TCS"])
    assert result.exit_code == 0


def test_validate_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validate_deps(monkeypatch)
    result = runner.invoke(app, ["validate", "--year", "2024", "--json"])
    assert result.exit_code == 0
    assert result.output.strip() == '{"findings": []}'
    assert "checks run:" not in result.output


def test_validate_out_writes_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_validate_deps(monkeypatch)
    out_dir = tmp_path / "reports"
    result = runner.invoke(app, ["validate", "--year", "2024", "--out", str(out_dir)])
    assert result.exit_code == 0
    output_file = out_dir / "dq_2024.json"
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == '{"findings": []}'
    assert str(output_file) in result.output


def test_validate_load_panel_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel as panel_mod

    def raise_error(spec: object) -> object:
        raise RuntimeError("load boom")

    monkeypatch.setattr(panel_mod, "load_panel", raise_error)
    result = runner.invoke(app, ["validate", "--year", "2024"])
    assert result.exit_code != 0
    assert "validate failed: load boom" in result.output


def test_validate_out_write_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_validate_deps(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--year", "2024", "--out", str(blocker)])
    assert result.exit_code != 0
    assert "could not write validation report" in result.output


def test_strategies_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.strategy.registry as registry_mod

    monkeypatch.setattr(
        registry_mod,
        "available",
        lambda: (_ for _ in ()).throw(RuntimeError("strat boom")),
    )
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code != 0
    assert "strategies: strat boom" in result.output


def test_cache_info_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.manifest as manifest_mod

    def raise_error(path: object = None) -> object:
        raise FileNotFoundError("missing manifest")

    monkeypatch.setattr(manifest_mod.Manifest, "load", classmethod(raise_error))
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code != 0
    assert "cache info: missing manifest" in result.output


def test_cache_info_panel_root_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import nifty_quant.data.manifest as manifest_mod
    import nifty_quant.settings as settings_mod

    fake_manifest = types.SimpleNamespace(
        cache_key=lambda: "fake-key",
    )
    monkeypatch.setattr(manifest_mod.Manifest, "load", classmethod(lambda cls: fake_manifest))
    monkeypatch.setattr(settings_mod, "CACHE_ROOT", tmp_path)
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "cache_key: fake-key" in result.output


def test_cache_info_skips_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import nifty_quant.data.manifest as manifest_mod
    import nifty_quant.settings as settings_mod

    fake_manifest = types.SimpleNamespace(
        cache_key=lambda: "fake-key",
    )
    monkeypatch.setattr(manifest_mod.Manifest, "load", classmethod(lambda cls: fake_manifest))
    monkeypatch.setattr(settings_mod, "CACHE_ROOT", tmp_path)
    panel_root = tmp_path / "panel" / f"v{settings_mod.PANEL_VERSION}"
    panel_root.mkdir(parents=True)
    (panel_root / "real-dir").mkdir()
    (panel_root / "plain-file.txt").write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "real-dir" in result.output
    assert "plain-file.txt" not in result.output


def test_cache_info_iterdir_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import nifty_quant.data.manifest as manifest_mod
    import nifty_quant.settings as settings_mod

    fake_manifest = types.SimpleNamespace(
        cache_key=lambda: "fake-key",
    )
    monkeypatch.setattr(manifest_mod.Manifest, "load", classmethod(lambda cls: fake_manifest))
    monkeypatch.setattr(settings_mod, "CACHE_ROOT", tmp_path)
    panel_root = tmp_path / "panel" / f"v{settings_mod.PANEL_VERSION}"
    panel_root.parent.mkdir(parents=True)
    panel_root.write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code != 0
    assert "cache info: " in result.output


def test_cache_gc_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    def raise_error(dry_run: bool = False) -> list[Path]:
        raise RuntimeError("gc boom")

    monkeypatch.setattr(pb_mod, "gc_orphans", raise_error)
    result = runner.invoke(app, ["cache", "gc"])
    assert result.exit_code != 0
    assert "cache gc failed: gc boom" in result.output


def test_cache_gc_no_orphans(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    monkeypatch.setattr(pb_mod, "gc_orphans", lambda dry_run=False: [])
    result = runner.invoke(app, ["cache", "gc"])
    assert result.exit_code == 0
    assert result.output.strip() == "no orphaned cache dirs"


def test_cache_gc_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    monkeypatch.setattr(
        pb_mod,
        "gc_orphans",
        lambda dry_run=False: [Path("/fake/a"), Path("/fake/b")],
    )
    result = runner.invoke(app, ["cache", "gc", "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] would remove: /fake/a" in result.output
    assert "[dry-run] would remove: /fake/b" in result.output
    assert "would remove 2 dir(s)" in result.output


def test_cache_gc_remove(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.data.panel_builder as pb_mod

    monkeypatch.setattr(
        pb_mod,
        "gc_orphans",
        lambda dry_run=False: [Path("/fake/a"), Path("/fake/b")],
    )
    result = runner.invoke(app, ["cache", "gc"])
    assert result.exit_code == 0
    assert "removed: /fake/a" in result.output
    assert "removed: /fake/b" in result.output
    assert "removed 2 dir(s)" in result.output


def test_dunder_main_invokes_app() -> None:
    import runpy
    import sys

    argv = sys.argv
    sys.argv = ["nq", "--version"]
    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("nifty_quant.cli", run_name="__main__")
    finally:
        sys.argv = argv
    assert exc_info.value.code == 0


def _session_ts(session_date, n_bars, start_hhmm=(9, 15)):
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(session_date, datetime.time(*start_hhmm), tzinfo=ist)
    return np.asarray([int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64)


def _make_ohlcv_panel(session_dates, bars_per_session, symbols=("A", "B")):
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
        "open": np.full((n_rows, n_symbols), 100.0, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), 101.0, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), 99.0, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), 100.5, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


def _fake_result(n=3, *, gross=None, net=None, turnover=None):
    if gross is None:
        gross = np.full(n, -0.001, dtype=np.float64)
    if turnover is None:
        turnover = np.full(gross.size, 1.0, dtype=np.float64)
    if net is None:
        net = gross - 0.0005 * turnover
    return BacktestResult(
        equity_curve=np.cumprod(1.0 + net) * 1e7,
        returns=net,
        positions=np.zeros((net.size, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
    )


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


def test_backtest_strategy_config_mismatch_fails_before_loaders(monkeypatch) -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "xsec_zscore",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
        ],
    )
    assert result.exit_code != 0
    assert "does not match" in result.output


def test_backtest_breakeven_nan_no_trades_branch(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    fake = _fake_result(n=1, gross=np.array([0.001]), net=np.array([0.0005]))

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(
        engine_mod, "run_backtest", lambda strat, panel, config, *, tradable=None: fake
    )
    monkeypatch.setattr(
        cli_mod, "_daily_returns", lambda returns, panel, decision_times: np.array([0.0005, 0.0004])
    )
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code == 0
    assert "breakeven cost: n/a (no trades / zero turnover)" in result.output


def test_backtest_latency_sensitivity_note_signal_dies(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])

    def stub_run_backtest(strat, panel, config, *, tradable=None):
        if config.decision_latency_bars == 0:
            return _fake_result(
                n=3, gross=np.array([0.02, 0.015, 0.025]), net=np.array([0.02, 0.015, 0.025])
            )
        return _fake_result(
            n=3, gross=np.array([-0.02, -0.015, -0.025]), net=np.array([-0.02, -0.015, -0.025])
        )

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", stub_run_backtest)
    monkeypatch.setattr(cli_mod, "_daily_returns", lambda returns, panel, decision_times: returns)
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.ones_like(turnover)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code == 0
    assert "NOTE: signal dies at one minute of latency" in result.output


def test_backtest_latency_sensitivity_exception_branch(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])

    def stub_run_backtest(strat, panel, config, *, tradable=None):
        if config.decision_latency_bars != 0:
            raise RuntimeError("latency boom")
        return _fake_result(
            n=3, gross=np.array([0.02, 0.015, 0.025]), net=np.array([0.02, 0.015, 0.025])
        )

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", stub_run_backtest)
    monkeypatch.setattr(cli_mod, "_daily_returns", lambda returns, panel, decision_times: returns)
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.ones_like(turnover)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code == 0
    assert "latency sensitivity: n/a (latency boom)" in result.output
    assert "gross Sharpe:" in result.output
    assert "net Sharpe:" in result.output


def test_backtest_manifest_load_failure_is_nonfatal(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.manifest as manifest_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    fake = _fake_result(
        n=3, gross=np.array([0.01, 0.012, 0.011]), net=np.array([0.01, 0.012, 0.011])
    )

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(
        engine_mod, "run_backtest", lambda strat, panel, config, *, tradable=None: fake
    )
    monkeypatch.setattr(cli_mod, "_daily_returns", lambda returns, panel, decision_times: returns)
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.ones_like(turnover)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        manifest_mod.Manifest,
        "load",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("manifest boom"))),
    )

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code == 0
    assert "gross Sharpe:" in result.output


def test_backtest_engine_failure_fallback_record_succeeds(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])

    def stub_run_backtest(strat, panel, config, *, tradable=None):
        raise RuntimeError("engine boom")

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", stub_run_backtest)
    monkeypatch.setattr(
        cli_mod, "_daily_returns", lambda returns, panel, decision_times: np.array([0.001, 0.002])
    )
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code != 0
    assert "backtest failed: engine boom" in result.output


def test_backtest_engine_failure_fallback_record_also_fails(monkeypatch, tmp_path) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.research.registry as registry_reg_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])

    def stub_run_backtest(strat, panel, config, *, tradable=None):
        raise RuntimeError("engine boom")

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", stub_run_backtest)
    monkeypatch.setattr(
        cli_mod, "_daily_returns", lambda returns, panel, decision_times: np.array([0.001, 0.002])
    )
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        registry_reg_mod,
        "TrialRegistry",
        lambda path: (_ for _ in ()).throw(RuntimeError("registry boom")),
    )

    result = runner.invoke(app, _backtest_args())
    assert result.exit_code != 0
    assert "backtest failed: engine boom" in result.output


def _session_ts(session_date, n_bars, start_hhmm=(9, 15)):
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(session_date, datetime.time(*start_hhmm), tzinfo=ist)
    return np.asarray([int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64)


def _make_ohlcv_panel(session_dates, bars_per_session, symbols=("A", "B")):
    ts = np.concatenate(
        [_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    fields = {
        "open": np.full((n_rows, n_symbols), 100.0, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), 101.0, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), 99.0, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), 100.5, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


class _FakeCalendar:
    def __init__(self, dates):
        self._dates = list(dates)

    def session_dates(self, start=None, end=None, *, usable_only=True):
        return [
            d for d in self._dates if (start is None or d >= start) and (end is None or d <= end)
        ]


ALL_DATES = pd.bdate_range("2024-01-01", periods=12).date.tolist()


def _fake_result(n=3, *, gross=None, net=None, turnover=None):
    if gross is None:
        gross = np.full(n, -0.001, dtype=np.float64)
    if turnover is None:
        turnover = np.full(gross.size, 1.0, dtype=np.float64)
    if net is None:
        net = gross - 0.0005 * turnover
    return BacktestResult(
        equity_curve=np.cumprod(1.0 + net) * 1e7,
        returns=net,
        positions=np.zeros((net.size, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
    )


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


def _patch_walkforward_common(
    monkeypatch, tmp_path, panel, run_backtest_stub, daily_returns_stub, daily_turnover_stub
):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", run_backtest_stub)
    monkeypatch.setattr(cli_mod, "_daily_returns", daily_returns_stub)
    monkeypatch.setattr(cli_mod, "_daily_turnover", daily_turnover_stub)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(ALL_DATES)),
    )


def test_walkforward_unknown_strategy_without_end() -> None:
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "not_a_real_strategy",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            ALL_DATES[0].isoformat(),
        ],
    )
    assert result.exit_code != 0
    assert "Unknown strategy" in result.output


def test_walkforward_start_after_end() -> None:
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-02-01",
            "--end",
            "2024-01-01",
        ],
    )
    assert result.exit_code != 0
    assert "after" in result.output


def test_walkforward_config_not_found() -> None:
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "/nonexistent/path/does_not_exist.yaml",
            "--start",
            ALL_DATES[0].isoformat(),
            "--end",
            ALL_DATES[-1].isoformat(),
        ],
    )
    assert result.exit_code != 0
    assert "Config file not found" in result.output


def test_walkforward_strategy_mismatch() -> None:
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "xsec_zscore",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            ALL_DATES[0].isoformat(),
            "--end",
            ALL_DATES[-1].isoformat(),
        ],
    )
    assert result.exit_code != 0
    assert "does not match" in result.output


def test_walkforward_calendar_raises() -> None:
    import nifty_quant.calendar as calendar_mod

    def _raise_calendar(cls, symbol="NIFTY50"):
        raise RuntimeError("calendar boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        calendar_mod.TradingCalendar, "from_index_bars", classmethod(_raise_calendar)
    )
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            ALL_DATES[0].isoformat(),
        ],
    )
    monkeypatch.undo()
    assert result.exit_code != 0
    assert "walkforward setup failed: calendar boom" in result.output


def test_walkforward_start_after_calendar_end() -> None:
    import nifty_quant.calendar as calendar_mod

    early_dates = pd.bdate_range("2023-01-01", periods=5).date.tolist()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(early_dates)),
    )
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-06-01",
        ],
    )
    monkeypatch.undo()
    assert result.exit_code != 0
    assert "after" in result.output


def test_walkforward_split_setup_failure(monkeypatch, tmp_path) -> None:
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.research.splits as splits_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(ALL_DATES)),
    )
    monkeypatch.setattr(
        splits_mod.WalkForwardSplitter,
        "split",
        lambda self, trading_dates: (_ for _ in ()).throw(RuntimeError("split boom")),
    )
    result = runner.invoke(
        app,
        [
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
        ],
    )
    assert result.exit_code != 0
    assert "walkforward split setup failed: split boom" in result.output


def test_walkforward_no_splits_fit(monkeypatch, tmp_path) -> None:
    import nifty_quant.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(ALL_DATES)),
    )
    args = _walkforward_args()
    args[args.index("--train-years") + 1] = "999"
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "no walk-forward splits fit" in result.output


def test_walkforward_manifest_load_raises_nonfatal(monkeypatch, tmp_path) -> None:
    import nifty_quant.data.manifest as manifest_mod

    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))
    monkeypatch.setattr(
        manifest_mod.Manifest,
        "load",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("manifest boom"))),
    )
    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda strat, panel, config, *, tradable=None: _fake_result(),
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code == 0
    assert "gross Sharpe:" in result.output


def test_walkforward_breakeven_warning_printed(monkeypatch, tmp_path) -> None:
    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))
    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda strat, panel, config, *, tradable=None: _fake_result(
            gross=np.full(3, -0.02, dtype=np.float64), turnover=np.full(3, 1.0, dtype=np.float64)
        ),
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code == 0
    assert "WARNING: breakeven cost" in result.output
    assert "does not survive its own costs" in result.output


def test_walkforward_breakeven_warning_not_printed(monkeypatch, tmp_path) -> None:
    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))
    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda strat, panel, config, *, tradable=None: _fake_result(
            gross=np.full(3, 0.05, dtype=np.float64), turnover=np.full(3, 1.0, dtype=np.float64)
        ),
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code == 0
    assert "WARNING: breakeven cost" not in result.output


def test_walkforward_latency_signal_dies(monkeypatch, tmp_path) -> None:
    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))

    def _latency_stub(strat, panel, config, *, tradable=None):
        lat = getattr(config, "decision_latency_bars", 0)
        if lat == 0:
            return _fake_result(
                gross=np.array([0.02, 0.015, 0.025], dtype=np.float64),
                turnover=np.full(3, 1.0, dtype=np.float64),
            )
        elif lat == 1:
            return _fake_result(
                gross=np.array([-0.02, -0.015, -0.025], dtype=np.float64),
                turnover=np.full(3, 1.0, dtype=np.float64),
            )
        else:
            return _fake_result(
                gross=np.array([-0.02, -0.015, -0.025], dtype=np.float64),
                turnover=np.full(3, 1.0, dtype=np.float64),
            )

    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        _latency_stub,
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code == 0
    assert "NOTE: signal dies at one minute of latency" in result.output


def test_walkforward_pooled_net_size_lt_2(monkeypatch, tmp_path) -> None:
    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))
    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda strat, panel, config, *, tradable=None: _fake_result(),
        lambda x, panel, decision_times: np.empty(0, dtype=np.float64),
        lambda x, panel, decision_times: np.empty(0, dtype=np.float64),
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code == 0


def test_walkforward_engine_raises_fallback_succeeds(monkeypatch, tmp_path) -> None:
    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))

    def _raise_engine(strat, panel, config, *, tradable=None):
        raise RuntimeError("wf engine boom")

    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        _raise_engine,
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code != 0
    assert "walkforward failed: wf engine boom" in result.output


def test_walkforward_engine_raises_fallback_fails(monkeypatch, tmp_path) -> None:
    import nifty_quant.research.registry as registry_mod

    panel = _make_ohlcv_panel(ALL_DATES, [5] * len(ALL_DATES))

    def _raise_engine(strat, panel, config, *, tradable=None):
        raise RuntimeError("wf engine boom")

    _patch_walkforward_common(
        monkeypatch,
        tmp_path,
        panel,
        _raise_engine,
        lambda x, panel, decision_times: x,
        lambda x, panel, decision_times: x,
    )
    real_trial_registry_cls = registry_mod.TrialRegistry
    call_count = {"n": 0}

    def _flaky_trial_registry(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_trial_registry_cls(path)
        raise RuntimeError("registry boom")

    monkeypatch.setattr(registry_mod, "TrialRegistry", _flaky_trial_registry)
    result = runner.invoke(app, _walkforward_args())
    assert result.exit_code != 0
    assert "walkforward failed: wf engine boom" in result.output


_VALID_VOLUME_BREAKOUT_PARAMS = {
    "breakout_window": 30,
    "volume_window": 30,
    "volume_z_threshold": 2.0,
    "hurst_window": 390,
    "hurst_threshold": 0.55,
    "use_hurst": False,
    "vol_window": 30,
    "deseasonalize": True,
    "direction": "continuation",
    "exit_mode": "time",
    "hold_bars": 30,
    "min_hold_bars": 5,
    "cooldown_bars": 0,
    "square_off_time": "15:20",
    "stop_loss_pct": 0.01,
    "target_pct": 0.02,
    "target_vol_ann": 0.15,
    "sigma_floor": 1.0e-5,
    "max_weight": 0.10,
    "gross": 1.0,
}


def _write_sweep_yaml(tmp_path, *, sweep_values=(2.0, 2.5)) -> Path:
    import yaml

    sweep_path = tmp_path / "sweep.yaml"
    sweep_config = {
        "strategy": "volume_breakout",
        "base_params": dict(_VALID_VOLUME_BREAKOUT_PARAMS),
        "sweep": {"volume_z_threshold": list(sweep_values)},
    }
    with open(sweep_path, "w") as f:
        yaml.safe_dump(sweep_config, f)
    return sweep_path


def _sweep_monkeypatch_set(monkeypatch, tmp_path, panel, *, run_backtest_stub):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.cli as cli_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", run_backtest_stub)
    monkeypatch.setattr(cli_mod, "_daily_returns", lambda returns, panel, decision_times: returns)
    monkeypatch.setattr(
        cli_mod, "_daily_turnover", lambda turnover, panel, decision_times: turnover
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


def _fake_calendar(dates):
    class _FakeCalendar:
        def __init__(self, dates):
            self._dates = list(dates)

        def session_dates(self, start=None, end=None, *, usable_only=True):
            return self._dates

    return _FakeCalendar(dates)


def test_sweep_no_end_nonexistent_config_covers_calendar_branch(monkeypatch, tmp_path) -> None:
    import nifty_quant.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, name: _fake_calendar([datetime.date(2024, 1, 1)])),
    )
    result = runner.invoke(
        app,
        ["sweep", "--config", str(tmp_path / "missing.yaml"), "--start", "2024-01-01"],
    )
    assert result.exit_code != 0
    assert "could not load sweep config" in result.output


def test_sweep_start_after_end_fails(monkeypatch, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(tmp_path / "any.yaml"),
            "--start",
            "2024-02-01",
            "--end",
            "2024-01-01",
        ],
    )
    assert result.exit_code != 0
    assert "after" in result.output


def test_sweep_calendar_raises_when_end_omitted(monkeypatch, tmp_path) -> None:
    import nifty_quant.calendar as calendar_mod

    def _boom(cls, name):
        raise RuntimeError("calendar boom")

    monkeypatch.setattr(calendar_mod.TradingCalendar, "from_index_bars", classmethod(_boom))
    result = runner.invoke(
        app,
        ["sweep", "--config", str(tmp_path / "any.yaml"), "--start", "2024-01-01"],
    )
    assert result.exit_code != 0
    assert "sweep could not resolve calendar: calendar boom" in result.output


def test_sweep_calendar_resolves_dates_before_start(monkeypatch, tmp_path) -> None:
    import nifty_quant.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, name: _fake_calendar([datetime.date(2023, 12, 1)])),
    )
    result = runner.invoke(
        app,
        ["sweep", "--config", str(tmp_path / "any.yaml"), "--start", "2024-01-01"],
    )
    assert result.exit_code != 0
    assert "after" in result.output


def test_sweep_unknown_strategy(monkeypatch, tmp_path) -> None:
    import yaml

    payload = {"strategy": "not_a_real_strategy", "base_params": {}, "sweep": {}}
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = runner.invoke(
        app,
        ["sweep", "--config", str(config_path), "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code != 0
    assert "Unknown strategy" in result.output


def test_sweep_manifest_load_raises_nonfatal(monkeypatch, tmp_path) -> None:
    import nifty_quant.data.manifest as manifest_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    config_path = _write_sweep_yaml(tmp_path)
    _sweep_monkeypatch_set(
        monkeypatch,
        tmp_path,
        panel,
        run_backtest_stub=lambda strat, panel, cfg: _fake_result(),
    )
    monkeypatch.setattr(
        manifest_mod.Manifest,
        "load",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("manifest boom"))),
    )
    result = runner.invoke(
        app,
        ["sweep", "--config", str(config_path), "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code == 0
    assert "sweep complete:" in result.output


def test_sweep_per_param_failure_records_and_continues(monkeypatch, tmp_path) -> None:
    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    config_path = _write_sweep_yaml(tmp_path, sweep_values=(2.0, 2.5))

    def _stub(strat, panel, cfg):
        raise RuntimeError("param boom")

    _sweep_monkeypatch_set(monkeypatch, tmp_path, panel, run_backtest_stub=_stub)
    result = runner.invoke(
        app,
        ["sweep", "--config", str(config_path), "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code == 0
    assert "FAILED: param boom" in result.output
    assert "sweep complete: 0 ok, 2 failed, 2 total" in result.output


def test_sweep_per_param_failure_record_also_raises(monkeypatch, tmp_path) -> None:
    import nifty_quant.research.registry as registry_reg_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    config_path = _write_sweep_yaml(tmp_path, sweep_values=(2.0, 2.5))

    def _stub(strat, panel, cfg):
        raise RuntimeError("param boom")

    _sweep_monkeypatch_set(monkeypatch, tmp_path, panel, run_backtest_stub=_stub)
    monkeypatch.setattr(
        registry_reg_mod.TrialRegistry,
        "record",
        lambda self, rec: (_ for _ in ()).throw(RuntimeError("record boom")),
    )
    result = runner.invoke(
        app,
        ["sweep", "--config", str(config_path), "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code == 0
    assert "FAILED: param boom" in result.output
    assert "sweep complete:" in result.output


def test_sweep_outer_exception_fails(monkeypatch, tmp_path) -> None:
    import nifty_quant.universe.static as universe_mod

    session_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    config_path = _write_sweep_yaml(tmp_path)
    _sweep_monkeypatch_set(
        monkeypatch,
        tmp_path,
        panel,
        run_backtest_stub=lambda strat, panel, cfg: _fake_result(),
    )
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: (_ for _ in ()).throw(RuntimeError("universe boom")),
    )
    result = runner.invoke(
        app,
        ["sweep", "--config", str(config_path), "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code != 0
    assert "sweep failed: universe boom" in result.output
