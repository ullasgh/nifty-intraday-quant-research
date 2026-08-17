"""TDD red tests for returns-persistence spec (specs/returns_persistence.md).

The producer feature does not exist yet: these tests are intentionally RED.
They monkeypatch only data/engine plumbing and exercise the real CLI, real
`_daily_returns`, and real registry/metrics consumer code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.backtest.metrics import pbo_cscv
from nifty_quant.cli import _daily_returns, app
from nifty_quant.data.panel import Panel
from nifty_quant.research.registry import TrialRegistry
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


def _fake_result(
    n_rows: int,
    *,
    ruined: bool = False,
    ruin_index: int = -1,
    zero_from: int | None = None,
    seed: int = 0,
) -> BacktestResult:
    """Build a deterministic, seeded fake BacktestResult.

    ``zero_from`` independently zeroes raw returns from that position onward WITHOUT
    setting ``ruined``/``ruin_index`` -- used to construct a genuinely-flat-but-not-ruined
    series for test 18, which must be indistinguishable from a ruined series by return
    values alone. ``BacktestResult`` is frozen, so all mutation must happen before
    construction, never on the built instance.
    """
    rng = np.random.default_rng(seed)
    gross = rng.normal(loc=0.0006, scale=0.004, size=n_rows).astype(np.float64)
    turnover = np.full(n_rows, 1.0, dtype=np.float64)
    returns = (gross - 0.0002 * turnover).astype(np.float64)
    if ruined and 0 <= ruin_index < n_rows:
        returns = returns.copy()
        returns[ruin_index:] = 0.0
    if zero_from is not None and 0 <= zero_from < n_rows:
        returns = returns.copy()
        returns[zero_from:] = 0.0
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
        ruined=ruined,
        ruin_index=ruin_index if ruined else -1,
    )


def _capturing_run_backtest(
    calls: list[dict[str, Any]],
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> Callable[..., BacktestResult]:
    def _stub(strat: Any, panel: Panel, config: Any, **kwargs: Any) -> BacktestResult:
        result = (
            result_factory(panel)
            if result_factory is not None
            else _fake_result(panel.n_rows())
        )
        calls.append({"panel": panel, "tradable": kwargs.get("tradable"), "result": result})
        return result

    return _stub


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    calls: list[dict[str, Any]],
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="ab", symbols=panel.symbols),
    )
    monkeypatch.setattr(
        engine_mod,
        "run_backtest",
        _capturing_run_backtest(calls, result_factory),
    )
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
        "2024-01-02",
    ]


def _run_backtest_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    *,
    result_factory: Callable[[Panel], BacktestResult] | None = None,
    extra_args: tuple[str, ...] = (),
) -> tuple[Path, list[dict[str, Any]]]:
    """Invoke `nq backtest` against a synthetic panel; return (trial_dir, calls)."""
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls, result_factory)
    result = runner.invoke(app, [*_backtest_args(), *extra_args])
    assert result.exit_code == 0, result.output
    trial_dirs = sorted(p for p in (tmp_path / "trials").iterdir() if p.is_dir())
    assert len(trial_dirs) == 1, f"expected exactly one trial dir, found {trial_dirs}"
    return trial_dirs[0], calls


def _panel_4_sessions_5_bars() -> Panel:
    return _make_panel(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        [5, 5, 5, 5],
        symbols=("A", "B"),
    )


def _panel_20_sessions_3_bars() -> Panel:
    dates = pd.bdate_range("2020-01-01", periods=20).date.tolist()
    return _make_panel(dates, [3] * len(dates), symbols=("A", "B"))


def _write_sweep_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "sweep.yaml"
    config = {
        "strategy": "volume_breakout",
        "base_params": {
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
        },
        "sweep": {"hold_bars": [20, 30]},
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def _run_sweep_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    sweep_yaml_path: Path,
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls, result_factory)
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(sweep_yaml_path),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )
    assert result.exit_code == 0, result.output
    return calls


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
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> list[dict[str, Any]]:
    import nifty_quant.calendar as calendar_mod

    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls, result_factory)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)),
    )
    runner.invoke(app, _walkforward_args(all_dates))
    # Do NOT assert exit_code == 0 here. `walkforward`'s pooled-summary step calls
    # `breakeven_cost_bps(pooled_gross, pooled_turnover)` where `pooled_gross` is
    # DAY-aggregated (via the real `_daily_returns`) but `pooled_turnover` is the raw
    # per-decision-row `res.turnover` concatenated across splits -- a pre-existing
    # shape-mismatch bug in `cli.py` that is orthogonal to the returns-persistence
    # feature under test here (it is masked in the existing test suite only because
    # `cli._daily_returns` is monkeypatched there to a fixed-length stub). Every
    # per-split `TrialRecord` is written to the registry inside the loop, before this
    # pooled step runs, so split-level artifacts are still observable even though the
    # command ultimately exits non-zero. Out of scope to fix (cli.py is off-limits).
    return calls


def test_backtest_writes_returns_parquet_next_to_result_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    assert (trial_dir / "result.parquet").is_file()
    assert (trial_dir / "returns.parquet").is_file()


def test_returns_parquet_has_exactly_ts_and_return_columns_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    assert list(frame.columns) == ["ts", "return"]


def test_ts_is_int64_epoch_seconds_utc_at_midnight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    ts = frame["ts"].to_numpy()
    assert ts.dtype == np.int64

    session_dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    expected = {
        int(datetime.combine(d, time(0, 0), tzinfo=timezone.utc).timestamp())
        for d in session_dates
    }
    assert all(int(v) % 86400 == 0 for v in ts)
    assert {int(v) for v in ts} == expected


def test_return_is_float64(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    assert frame["return"].to_numpy().dtype == np.float64


def test_ts_is_strictly_increasing_and_unique(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    ts = pd.read_parquet(trial_dir / "returns.parquet")["ts"].to_numpy()
    assert np.all(np.diff(ts) > 0)


def test_one_row_per_trading_day_not_per_decision_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2024-01-01", periods=3).date.tolist()
    panel = _make_panel(dates, [50] * len(dates), symbols=("A", "B"))
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    # Guard against silently breaking 252-annualization: 149 decision rows total,
    # but the persisted artifact must aggregate to exactly 3 trading days.
    assert len(frame) == 3


def test_stored_series_equals_cli_daily_returns_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, calls = _run_backtest_cli(monkeypatch, tmp_path, panel)
    assert len(calls) >= 1
    expected = _daily_returns(calls[0]["result"].returns, calls[0]["panel"], None)
    stored = pd.read_parquet(trial_dir / "returns.parquet")["return"].to_numpy()
    assert np.array_equal(stored, expected)


def test_irregular_session_produces_one_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _make_panel([date(2024, 1, 2)], [60], symbols=("A", "B"))
    trial_dir, _ = _run_backtest_cli(monkeypatch, tmp_path, panel)
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    assert len(frame) == 1


def test_build_trial_matrix_accepts_written_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    _run_backtest_cli(monkeypatch, tmp_path, panel)
    registry = TrialRegistry(tmp_path / "trials.db")
    matrix = registry.build_trial_matrix(strategy="volume_breakout")
    assert matrix.n_dropped == 0, matrix.explain()


def test_pbo_cscv_returns_finite_value_on_written_trials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_20_sessions_3_bars()
    yaml_path = _write_sweep_yaml(tmp_path)
    _run_sweep_cli(monkeypatch, tmp_path, panel, yaml_path)
    trial_matrix = TrialRegistry(tmp_path / "trials.db").build_trial_matrix(
        strategy="volume_breakout"
    ).matrix
    result = pbo_cscv(trial_matrix)
    assert math.isfinite(result) and 0.0 <= result <= 1.0


def test_walkforward_sets_result_path_on_every_trial_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_dates = pd.bdate_range("2020-01-01", periods=280).date.tolist()
    panel = _make_panel(all_dates, [2] * len(all_dates))
    _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    records = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")
    split_records = [r for r in records if r.split_id != "pooled"]
    assert len(split_records) >= 1
    assert all(r.result_path is not None for r in split_records)


def test_walkforward_writes_one_returns_parquet_per_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_dates = pd.bdate_range("2020-01-01", periods=280).date.tolist()
    panel = _make_panel(all_dates, [2] * len(all_dates))
    _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    records = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")
    split_records = [r for r in records if r.split_id != "pooled"]
    assert len(split_records) >= 2
    for r in split_records:
        assert (Path(r.result_path) / "returns.parquet").is_file()


def test_walkforward_split_ts_ranges_do_not_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_dates = pd.bdate_range("2020-01-01", periods=280).date.tolist()
    panel = _make_panel(all_dates, [2] * len(all_dates))
    _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)

    records = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")
    split_records = [r for r in records if r.split_id != "pooled"]
    assert all(r.result_path is not None for r in split_records), (
        "walkforward split records must have result_path before comparing ts ranges"
    )

    ranges: list[tuple[int, int]] = []
    for r in split_records:
        path = Path(r.result_path) / "returns.parquet"
        assert path.is_file(), f"missing returns.parquet for {r}"
        ts = pd.read_parquet(path)["ts"].to_numpy()
        ranges.append((int(ts.min()), int(ts.max())))

    ranges.sort(key=lambda x: x[0])
    for earlier, later in zip(ranges, ranges[1:]):
        assert earlier[1] < later[0]


def test_sweep_sets_result_path_on_every_trial_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_20_sessions_3_bars()
    yaml_path = _write_sweep_yaml(tmp_path)
    _run_sweep_cli(monkeypatch, tmp_path, panel, yaml_path)

    records = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")
    assert len(records) == 2
    assert all(r.result_path is not None for r in records)


def test_sweep_writes_returns_parquet_per_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_20_sessions_3_bars()
    yaml_path = _write_sweep_yaml(tmp_path)
    _run_sweep_cli(monkeypatch, tmp_path, panel, yaml_path)

    records = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")
    assert len(records) == 2
    for r in records:
        assert (Path(r.result_path) / "returns.parquet").is_file()


def test_ruined_run_still_writes_returns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    trial_dir, _ = _run_backtest_cli(
        monkeypatch,
        tmp_path,
        panel,
        result_factory=lambda p: _fake_result(p.n_rows(), ruined=True, ruin_index=14),
    )

    assert (trial_dir / "returns.parquet").is_file()
    frame = pd.read_parquet(trial_dir / "returns.parquet")
    # ruin_index=14 zeroes raw-return positions 14..19, which map entirely to the
    # fourth and final session (day_offsets[3] == 15), so its daily return is 0.0.
    assert len(frame) == 4
    assert frame["return"].iloc[-1] == 0.0


def test_ruined_flag_and_index_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    _run_backtest_cli(
        monkeypatch,
        tmp_path,
        panel,
        result_factory=lambda p: _fake_result(p.n_rows(), ruined=True, ruin_index=14),
    )

    _MISSING = object()
    record = TrialRegistry(tmp_path / "trials.db").all(strategy="volume_breakout")[0]

    # SPEC-AMBIGUITY: spec allows "TrialRecord (or artifact metadata)" for where
    # ruined/ruin_index are recorded; asserting against TrialRecord fields since
    # build_trial_matrix reads trials from SQLite rows without opening per-trial
    # artifact files.
    assert getattr(record, "ruined", _MISSING) is True
    assert getattr(record, "ruin_index", _MISSING) == 14


def test_consumer_can_distinguish_ruined_from_flat_series(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    ruined_dir = tmp_path / "ruined"
    flat_dir = tmp_path / "flat"
    ruined_dir.mkdir(parents=True)
    flat_dir.mkdir(parents=True)

    ruined_trial_dir, _ = _run_backtest_cli(
        monkeypatch,
        ruined_dir,
        panel,
        result_factory=lambda p: _fake_result(p.n_rows(), ruined=True, ruin_index=14),
    )

    def flat_factory(p: Panel) -> BacktestResult:
        # Not ruined: returns[14:] happen to be zero for an unrelated, legitimate
        # reason (e.g. no position held), constructed pre-freeze via zero_from.
        return _fake_result(p.n_rows(), zero_from=14)

    flat_trial_dir, _ = _run_backtest_cli(
        monkeypatch,
        flat_dir,
        panel,
        result_factory=flat_factory,
    )

    ruined_returns = pd.read_parquet(ruined_trial_dir / "returns.parquet")["return"]
    flat_returns = pd.read_parquet(flat_trial_dir / "returns.parquet")["return"]

    assert ruined_returns.iloc[-1] == 0.0
    assert flat_returns.iloc[-1] == 0.0
    assert ruined_returns.iloc[-1] == flat_returns.iloc[-1]

    ruined_record = TrialRegistry(ruined_dir / "trials.db").all(strategy="volume_breakout")[0]
    flat_record = TrialRegistry(flat_dir / "trials.db").all(strategy="volume_breakout")[0]

    _MISSING = object()
    assert getattr(ruined_record, "ruined", _MISSING) is True
    assert getattr(flat_record, "ruined", _MISSING) is False


def test_two_identical_runs_write_identical_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel = _panel_4_sessions_5_bars()
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    trial_dir_a, _ = _run_backtest_cli(
        monkeypatch,
        run_a,
        panel,
        result_factory=lambda p: _fake_result(p.n_rows()),
    )
    trial_dir_b, _ = _run_backtest_cli(
        monkeypatch,
        run_b,
        panel,
        result_factory=lambda p: _fake_result(p.n_rows()),
    )

    assert (trial_dir_a / "returns.parquet").read_bytes() == (
        trial_dir_b / "returns.parquet"
    ).read_bytes()
