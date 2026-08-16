"""Tests for the ``nq`` CLI against the real repo data."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nifty_quant.cli import app

runner = CliRunner()


def test_version_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "nifty_quant" in result.output


def test_info_exits_zero_and_prints_data_root() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "DATA_ROOT" in result.output


def test_symbols_reports_expected_counts() -> None:
    result = runner.invoke(app, ["symbols"])
    assert result.exit_code == 0
    assert "153 total" in result.output
    assert "149 equity" in result.output


def test_strategies_lists_both_plugins() -> None:
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    assert "volume_breakout" in result.output
    assert "xsec_zscore" in result.output


def test_cache_info_prints_cache_key() -> None:
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "cache_key:" in result.output


@pytest.mark.slow
def test_validate_reports_ohlc_check() -> None:
    result = runner.invoke(
        app,
        ["validate", "--year", "2024", "--symbols", "RELIANCE"],
    )
    assert result.exit_code == 0
    assert "check_ohlc_consistency" in result.output


def test_backtest_unknown_strategy_fails_cleanly() -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "not_a_real_strategy",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown strategy" in result.output


def test_backtest_missing_config_fails_cleanly() -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            "/nonexistent/path/does_not_exist.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )
    assert result.exit_code != 0
    assert "Config file not found" in result.output


def test_backtest_start_after_end_fails_cleanly() -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
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


def _fake_negative_edge_result():
    """A canned BacktestResult with a strictly negative gross edge."""
    import numpy as np
    import pandas as pd

    from nifty_quant.backtest.engine import BacktestResult

    rng = np.random.default_rng(0)
    n = 100
    gross_returns = rng.normal(loc=-0.001, scale=0.005, size=n)
    turnover = np.full(n, 1.0)
    returns = gross_returns - 0.0005 * turnover  # net is also negative

    return BacktestResult(
        equity_curve=np.cumprod(1.0 + returns) * 1e7,
        returns=returns,
        positions=np.zeros((n, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross_returns,
        total_costs=1234.0,
        n_trades=10,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
    )


def _small_universe():
    from nifty_quant.universe.static import Universe

    return Universe(name="test_small", symbols=("RELIANCE",))


def test_backtest_negative_edge_exits_zero_with_verdict(monkeypatch, tmp_path) -> None:
    """A strategy with no gross edge must still exit 0 and print a plain-language
    verdict instead of aborting the whole command."""
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(universe_mod, "load_universe", lambda name: _small_universe())
    monkeypatch.setattr(
        engine_mod, "run_backtest", lambda strat, panel, config, **kw: _fake_negative_edge_result()
    )

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "gross Sharpe:" in result.output
    assert "net Sharpe:" in result.output
    assert "NEGATIVE GROSS EDGE" in result.output


def test_backtest_derived_stat_failure_degrades_to_na(monkeypatch, tmp_path) -> None:
    """A failure in a derived statistic (here: breakeven) must degrade that line to
    n/a rather than aborting the command; primary results must still print."""
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.execution.costs as costs_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated derived-stat failure")

    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(universe_mod, "load_universe", lambda name: _small_universe())
    monkeypatch.setattr(
        engine_mod, "run_backtest", lambda strat, panel, config, **kw: _fake_negative_edge_result()
    )
    monkeypatch.setattr(costs_mod, "breakeven_cost_bps", _boom)

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "gross Sharpe:" in result.output
    assert "net Sharpe:" in result.output
    assert "breakeven cost: n/a" in result.output
    assert "simulated derived-stat failure" in result.output
