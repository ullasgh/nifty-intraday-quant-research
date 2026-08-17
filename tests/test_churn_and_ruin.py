from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import nifty_quant.backtest.engine as engine_mod
import nifty_quant.settings as settings_mod
import nifty_quant.universe.static as universe_mod
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.backtest.metrics import verdict_line
from nifty_quant.cli import app
from nifty_quant.features.core import breakout_down, breakout_up, volume_zscore
from nifty_quant.strategy.base import PortfolioState
from nifty_quant.strategy.plugins.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
    _edge_trigger,
)


@dataclass
class _PanelStub:
    symbols: tuple[str, ...]
    day_offsets: np.ndarray
    ts: np.ndarray
    arrays: dict[str, np.ndarray]
    minute_array: np.ndarray

    def field(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def minute_of_day(self) -> np.ndarray:
        return self.minute_array


@dataclass
class _ViewStub:
    ts: int
    session_date: datetime.date
    symbols: tuple[str, ...]
    tradable: np.ndarray

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        raise NotImplementedError("not needed by on_decision in these tests")


def test_enters_on_edge_not_on_level() -> None:
    n_rows = 45
    symbol = "EDGE"
    close = (100.0 + 2.0 * np.arange(n_rows, dtype=np.float32)).reshape(n_rows, 1)
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    volume = np.empty((n_rows, 1), dtype=np.float32)
    for i in range(n_rows):
        if i < 15:
            volume[i, 0] = 999.0 if i % 2 == 0 else 1001.0
        else:
            volume[i, 0] = 1_000_000.0 * (3.0 ** (i - 15))

    panel = _PanelStub(
        symbols=(symbol,),
        day_offsets=np.array([0, n_rows], dtype=np.intp),
        ts=np.arange(n_rows, dtype=np.int64),
        arrays={"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        minute_array=np.arange(n_rows, dtype=np.int64) % 375,
    )
    params = VolumeBreakoutParams(
        use_hurst=False,
        breakout_window=5,
        volume_window=10,
        volume_z_threshold=2.0,
        deseasonalize=False,
    )
    strategy = VolumeBreakoutStrategy(params)
    signals = strategy.precompute(panel)

    up = breakout_up(close, high, 5, day_offsets=panel.day_offsets)
    vol_z = volume_zscore(
        volume,
        panel.minute_of_day(),
        10,
        deseasonalize=False,
        day_offsets=panel.day_offsets,
    )
    raw_long = up & (vol_z > 2.0)
    assert int(np.sum(raw_long)) >= 20

    assert int(np.sum(signals["long"])) == 1
    assert int(np.sum(signals["short"])) == 0


def test_edge_detection_resets_at_session_boundary() -> None:
    day_offsets = np.array([0, 60, 165], dtype=np.intp)
    cond = np.zeros((165, 1), dtype=bool)
    cond[59, 0] = True
    cond[60, 0] = True

    result = _edge_trigger(cond, day_offsets)

    assert result[59, 0]
    assert result[60, 0]
    assert int(result.sum()) == 2

    cond[100, 0] = True
    cond[101, 0] = True

    result = _edge_trigger(cond, day_offsets)

    assert result[100, 0]
    assert not result[101, 0]


def test_trade_count_collapses_versus_prefix() -> None:
    symbol = "CHURN"
    breakout_window = 5
    volume_window = 10
    volume_z_threshold = 2.0
    warmup = 15
    burst_len = 20
    gap_len = 12
    cycles = 20
    n_rows = warmup + cycles * (burst_len + gap_len)

    close = (100.0 + 2.0 * np.arange(n_rows, dtype=np.float32)).reshape(n_rows, 1)
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    volume = np.empty((n_rows, 1), dtype=np.float32)
    for i in range(n_rows):
        volume[i, 0] = 999.0 if i % 2 == 0 else 1001.0
    for c in range(cycles):
        burst_start = warmup + c * (burst_len + gap_len)
        for k in range(burst_len):
            volume[burst_start + k, 0] = 1_000_000.0 * (3.0**k)

    day_offsets = np.array([0, n_rows], dtype=np.intp)
    panel = _PanelStub(
        symbols=(symbol,),
        day_offsets=day_offsets,
        ts=np.arange(n_rows, dtype=np.int64),
        arrays={"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        minute_array=np.arange(n_rows, dtype=np.int64) % 375,
    )
    params = VolumeBreakoutParams(
        use_hurst=False,
        breakout_window=breakout_window,
        volume_window=volume_window,
        volume_z_threshold=volume_z_threshold,
        deseasonalize=False,
    )
    strategy = VolumeBreakoutStrategy(params)
    signals = strategy.precompute(panel)
    edge_total = int(np.sum(signals["long"])) + int(np.sum(signals["short"]))

    up = breakout_up(
        close,
        high,
        breakout_window,
        day_offsets=day_offsets,
    )
    down = breakout_down(
        close,
        low,
        breakout_window,
        day_offsets=day_offsets,
    )
    vol_z = volume_zscore(
        volume,
        panel.minute_of_day(),
        volume_window,
        deseasonalize=False,
        day_offsets=day_offsets,
    )
    vol_ok = vol_z > volume_z_threshold
    raw_long = up & vol_ok
    raw_short = down & vol_ok
    raw_level_count = int(np.sum(raw_long)) + int(np.sum(raw_short))

    assert raw_level_count >= 200
    assert edge_total * 10 <= raw_level_count


def test_config_default_is_use_hurst_false() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "strategies"
        / "volume_breakout.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["params"]["use_hurst"] is False


def test_min_hold_bars_blocks_early_exit() -> None:
    min_hold = 3
    params = VolumeBreakoutParams(
        exit_mode="opposite",
        min_hold_bars=min_hold,
        hold_bars=30,
        cooldown_bars=0,
        use_hurst=False,
    )
    strategy = VolumeBreakoutStrategy(params)
    symbol = "RELIANCE"
    symbols = (symbol,)

    def view_at(ts: int) -> _ViewStub:
        return _ViewStub(
            ts=ts,
            session_date=datetime.date(2024, 1, 15),
            symbols=symbols,
            tradable=np.array([True]),
        )

    def state_with(shares: float, ts: int) -> PortfolioState:
        return PortfolioState(shares=np.array([shares]), cash=0.0, equity=0.0, ts=ts)

    def signals(*, long: bool, short: bool) -> dict[str, np.ndarray]:
        return {
            "long": np.array([long], dtype=bool),
            "short": np.array([short], dtype=bool),
            "sigma": np.array([0.02], dtype=np.float64),
            "vol_z": np.zeros(1, dtype=np.float64),
            "hurst": np.full(1, np.inf, dtype=np.float64),
        }

    ts0 = int(pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Kolkata").timestamp())
    entry = strategy.on_decision(
        view_at(ts0), signals(long=True, short=False), state_with(0.0, ts0)
    )
    assert entry is not None
    assert entry.weights[0] > 0.0

    for call in range(1, min_hold + 1):
        ts = ts0 + call * 60
        target = strategy.on_decision(
            view_at(ts), signals(long=False, short=True), state_with(1.0, ts)
        )
        assert target is not None
        if call < min_hold:
            assert target.weights[0] > 0.0
        else:
            assert target.weights[0] == 0.0


def test_cooldown_blocks_immediate_reentry() -> None:
    cooldown = 2
    params = VolumeBreakoutParams(
        exit_mode="time",
        hold_bars=30,
        min_hold_bars=1,
        cooldown_bars=cooldown,
        use_hurst=False,
    )
    strategy = VolumeBreakoutStrategy(params)
    symbol = "RELIANCE"
    symbols = (symbol,)

    normal_ts = int(pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Kolkata").timestamp())
    square_off_ts = int(pd.Timestamp("2024-01-15 15:20:00", tz="Asia/Kolkata").timestamp())

    def view_at(ts: int) -> _ViewStub:
        return _ViewStub(
            ts=ts,
            session_date=datetime.date(2024, 1, 15),
            symbols=symbols,
            tradable=np.array([True]),
        )

    def state_with(shares: float, ts: int) -> PortfolioState:
        return PortfolioState(shares=np.array([shares]), cash=0.0, equity=0.0, ts=ts)

    def long_signals() -> dict[str, np.ndarray]:
        return {
            "long": np.array([True], dtype=bool),
            "short": np.array([False], dtype=bool),
            "sigma": np.array([0.02], dtype=np.float64),
            "vol_z": np.zeros(1, dtype=np.float64),
            "hurst": np.full(1, np.inf, dtype=np.float64),
        }

    exit_target = strategy.on_decision(
        view_at(square_off_ts), long_signals(), state_with(1.0, square_off_ts)
    )
    assert exit_target is not None
    assert exit_target.weights[0] == 0.0

    for lag in range(1, cooldown + 1):
        ts = normal_ts + lag * 60
        target = strategy.on_decision(view_at(ts), long_signals(), state_with(0.0, ts))
        assert target is not None
        assert target.weights[0] == 0.0

    ts_allowed = normal_ts + (cooldown + 1) * 60
    target = strategy.on_decision(
        view_at(ts_allowed), long_signals(), state_with(0.0, ts_allowed)
    )
    assert target is not None
    assert target.weights[0] > 0.0


def test_no_reentry_while_positioned_same_side() -> None:
    params = VolumeBreakoutParams(
        exit_mode="time",
        hold_bars=100,
        min_hold_bars=1,
        cooldown_bars=0,
        use_hurst=False,
    )
    symbol = "RELIANCE"
    symbols = (symbol,)
    ts = int(pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Kolkata").timestamp())
    view = _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 15),
        symbols=symbols,
        tradable=np.array([True]),
    )
    state = PortfolioState(shares=np.array([1.0]), cash=0.0, equity=0.0, ts=ts)

    def signals(long: bool) -> dict[str, np.ndarray]:
        return {
            "long": np.array([long], dtype=bool),
            "short": np.array([False], dtype=bool),
            "sigma": np.array([0.02], dtype=np.float64),
            "vol_z": np.zeros(1, dtype=np.float64),
            "hurst": np.full(1, np.inf, dtype=np.float64),
        }

    with_signal_strategy = VolumeBreakoutStrategy(params)
    without_signal_strategy = VolumeBreakoutStrategy(params)

    target_with = with_signal_strategy.on_decision(view, signals(True), state)
    target_without = without_signal_strategy.on_decision(view, signals(False), state)

    assert target_with is not None
    assert target_without is not None
    assert target_with.weights[0] == target_without.weights[0]
    assert target_with.weights[0] > 0.0


def test_stop_target_squareoff_still_work() -> None:
    hold_bars = 5
    params = VolumeBreakoutParams(
        exit_mode="stop_target",
        min_hold_bars=1,
        hold_bars=hold_bars,
        cooldown_bars=0,
        use_hurst=False,
    )
    strategy = VolumeBreakoutStrategy(params)
    symbol = "RELIANCE"
    symbols = (symbol,)

    def view_at(ts: int) -> _ViewStub:
        return _ViewStub(
            ts=ts,
            session_date=datetime.date(2024, 1, 15),
            symbols=symbols,
            tradable=np.array([True]),
        )

    def state_with(shares: float, ts: int) -> PortfolioState:
        return PortfolioState(shares=np.array([shares]), cash=0.0, equity=0.0, ts=ts)

    def signals(long: bool = True, short: bool = False) -> dict[str, np.ndarray]:
        return {
            "long": np.array([long], dtype=bool),
            "short": np.array([short], dtype=bool),
            "sigma": np.array([0.02], dtype=np.float64),
            "vol_z": np.zeros(1, dtype=np.float64),
            "hurst": np.full(1, np.inf, dtype=np.float64),
        }

    ts0 = int(pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Kolkata").timestamp())
    entry = strategy.on_decision(
        view_at(ts0), signals(long=True, short=False), state_with(0.0, ts0)
    )
    assert entry is not None
    assert entry.weights[0] > 0.0

    for held in range(1, hold_bars):
        ts = ts0 + held * 60
        target = strategy.on_decision(
            view_at(ts), signals(long=False, short=False), state_with(1.0, ts)
        )
        assert target is not None
        assert target.weights[0] > 0.0

    ts_exit = ts0 + hold_bars * 60
    exit_target = strategy.on_decision(
        view_at(ts_exit), signals(long=False, short=False), state_with(1.0, ts_exit)
    )
    assert exit_target is not None
    assert exit_target.weights[0] == 0.0

    square_off_params = VolumeBreakoutParams(
        exit_mode="time",
        hold_bars=30,
        min_hold_bars=5,
        cooldown_bars=0,
        use_hurst=False,
    )
    square_off_strategy = VolumeBreakoutStrategy(square_off_params)
    square_off_ts = int(pd.Timestamp("2024-01-15 15:20:00", tz="Asia/Kolkata").timestamp())
    square_off_view = _ViewStub(
        ts=square_off_ts,
        session_date=datetime.date(2024, 1, 15),
        symbols=symbols,
        tradable=np.array([True]),
    )
    square_off_state = PortfolioState(
        shares=np.array([1.0]), cash=0.0, equity=0.0, ts=square_off_ts
    )
    square_off_target = square_off_strategy.on_decision(
        square_off_view, signals(long=False, short=False), square_off_state
    )
    assert square_off_target is not None
    assert square_off_target.weights[0] == 0.0


def _fake_ruined_result() -> BacktestResult:
    rng = np.random.default_rng(0)
    # 1500 matches the real 4-session x 375-bar RELIANCE panel for
    # 2024-01-02..2024-01-05 that the backtest command's new daily-return-aggregation
    # reconstruction requires an exact row-count match against.
    n = 1500
    gross_returns = rng.normal(loc=-0.001, scale=0.005, size=n)
    turnover = np.full(n, 1.0)
    returns = gross_returns - 0.0005 * turnover

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
        ruined=True,
        ruin_index=42,
    )


def test_cli_reports_ruined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda *args, **kwargs: universe_mod.Universe(name="test_small", symbols=("RELIANCE",)),
    )
    monkeypatch.setattr(
        engine_mod,
        "run_backtest",
        lambda strat, panel, config, **kw: _fake_ruined_result(),
    )

    result = CliRunner().invoke(
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
    assert "ruined: True" in result.output
    assert "ruin_index: 42" in result.output


def test_verdict_line_refuses_sharpe_when_ruined() -> None:
    line = verdict_line(
        sharpe_net=2.5,
        sr_se=0.1,
        n_trials=5,
        n_eff=5.0,
        sr0=0.0,
        dsr=0.99,
        pbo=0.1,
        ruined=True,
    )
    assert "RUINED" in line
    assert "Sharpe=2.500" not in line


def test_verdict_line_flags_low_trade_count() -> None:
    line_low = verdict_line(
        sharpe_net=1.0,
        sr_se=0.1,
        n_trials=10,
        n_eff=5.0,
        sr0=0.0,
        dsr=0.99,
        pbo=0.1,
        n_trades=5,
    )
    assert "WARNING" in line_low
    assert "5" in line_low

    line_default = verdict_line(
        sharpe_net=1.0,
        sr_se=0.1,
        n_trials=10,
        n_eff=5.0,
        sr0=0.0,
        dsr=0.99,
        pbo=0.1,
    )
    assert "WARNING" not in line_default
