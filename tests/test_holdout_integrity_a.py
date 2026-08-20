"""SPEC DEFECTS / AMBIGUITIES FOUND

(a) Item 8 specifies check_cash_non_negative(cash, *, floor=0.0), but provides no
    parameter carrying "the row" despite requiring the raised message to name it;
    that part is therefore untestable as specified.

(b) Item 6 names no exception class for a stored-boundary disagreement, so this test
    asserts against Exception broadly and checks the required date strings.

(c) Item 7 (section D) names no callable for reconciliation/annotation. This test
    assumes HoldoutLock.annotate_legacy_reads(reason: str) -> int, inferred only from
    the naming convention of the two existing public methods on the class.

(d) The boundary rules imply that a caller window ending before a stored boundary is
    a prefix/window query, whereas a caller calendar extending beyond the stored
    holdout end is a disagreement; the tests use that distinction.
"""

from __future__ import annotations

import inspect
import json
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import nifty_quant.guards as guards
import nifty_quant.research.tilt as tilt_module
from nifty_quant.backtest.engine import BacktestResult, build_daily
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.research.splits import HoldoutLock, Split, WalkForwardSplitter
from nifty_quant.research.tilt import TiltConfig, run_tilt
from nifty_quant.universe.static import Universe
from tests.contract_fixtures import minimal_contract

runner = CliRunner()


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
    "sigma_floor": 1.0e-5,
    "max_weight": 0.10,
    "gross": 1.0,
}


def _make_trading_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


class _FakeCalendar:
    def __init__(self, dates):
        self._dates = list(dates)

    def session_dates(self, start=None, end=None, *, usable_only=True):
        return [
            d
            for d in self._dates
            if (start is None or d >= start) and (end is None or d <= end)
        ]


def _fake_result(n=3, *, gross=None, net=None, turnover=None, daily=None):
    if gross is None:
        gross = np.full(n, -0.001, dtype=np.float64)
    if turnover is None:
        turnover = np.full(gross.size, 1.0, dtype=np.float64)
    if net is None:
        net = gross - 0.0005 * turnover
    equity_curve = np.cumprod(1.0 + net) * 1e7
    if daily is None:
        n_rows = net.size
        daily = build_daily(
            np.arange(n_rows, dtype=np.int64),
            equity_curve,
            net,
            gross,
            turnover,
            np.arange(n_rows, dtype=np.int64),
            initial_capital=1e7,
        )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=net,
        positions=np.zeros((net.size, 1)),
        trades=pd.DataFrame(
            {"symbol": [], "side": [], "qty": [], "price": []}
        ),
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


def _session_ts(session_date, n_bars, start_hhmm=(9, 15)):
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(
        session_date, dtime(*start_hhmm), tzinfo=ist
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _make_ohlcv_panel(
    session_dates, bars_per_session, symbols=("A", "B")
):
    ts = np.concatenate(
        [
            _session_ts(d, n)
            for d, n in zip(session_dates, bars_per_session)
        ]
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
        "volume": np.full(
            (n_rows, n_symbols), 1_000_000.0, dtype=np.float64
        ),
    }
    return Panel(
        fields,
        symbols,
        ts,
        day_offsets,
        np.asarray(session_dates, dtype=object),
    )


def _seed_lock(
    path: Path,
    *,
    count: int,
    holdout_start: date,
    holdout_end: date,
    log=None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "count": count,
                "log": log or [],
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _make_split(
    split_id: str,
    train_dates: list[date],
    test_dates: list[date],
) -> Split:
    # NOTE: `Split`'s real fields are `id`, `train`, `test`, `embargo_days` (no
    # default on embargo_days) -- confirmed against
    # `nifty_quant.research.splits.Split` directly, not guessed.
    return Split(
        id=split_id,
        train=(train_dates[0], train_dates[-1]),
        test=(test_dates[0], test_dates[-1]),
        embargo_days=5,
    )


def _patch_common(
    monkeypatch,
    tmp_path,
    panel,
    run_backtest_stub,
    calendar_dates,
):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="ab", symbols=("A", "B")),
    )
    monkeypatch.setattr(engine_mod, "run_backtest", run_backtest_stub)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(
            lambda cls, *args, **kwargs: _FakeCalendar(calendar_dates)
        ),
    )


def _walkforward_args(start: date, end: date) -> list[str]:
    return [
        "walkforward",
        "--strategy",
        "volume_breakout",
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--train-years",
        "0.1",
        "--test-years",
        "0.03",
        "--no-tradable-filter",
    ]


# Item 1
def test_item_1_holdout_boundary_is_reused_for_short_window(tmp_path):
    full_dates = _make_trading_dates(
        date(2020, 1, 1), date(2026, 6, 30)
    )
    short_dates = full_dates[:40]
    lock = HoldoutLock(path=tmp_path / "lock.json")

    window_full = lock.holdout_range(full_dates)
    # must fail against current code: HoldoutLock.holdout_range ignores the
    # file entirely and always recomputes from its argument.
    window_short = lock.holdout_range(short_dates)

    assert window_short == window_full


# Item 2
def test_item_2_walkforward_uses_preseeded_far_future_boundary(
    monkeypatch, tmp_path
):
    calendar_dates = _make_trading_dates(
        date(2024, 1, 1), date(2024, 3, 1)
    )
    panel = _make_ohlcv_panel(
        calendar_dates, [1] * len(calendar_dates)
    )
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda *args, **kwargs: _fake_result(),
        calendar_dates,
    )

    late_test_dates = calendar_dates[-5:]
    monkeypatch.setattr(
        WalkForwardSplitter,
        "split",
        lambda self, *args, **kwargs: [
            _make_split(
                "rolling_000",
                calendar_dates[:20],
                late_test_dates,
            )
        ],
    )

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        count=0,
        holdout_start=date(2099, 1, 1),
        holdout_end=date(2099, 12, 31),
    )

    result = runner.invoke(
        app,
        _walkforward_args(date(2024, 1, 1), date(2024, 3, 1)),
    )

    assert result.exit_code == 0
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    # today this records 1: current code ignores the pre-seeded far-future
    # boundary and manufactures a fake holdout inside the short 2024 window
    # instead.
    assert state["count"] == 0


# Item 3
def test_item_3_walkforward_refuses_then_allows_holdout(
    monkeypatch, tmp_path
):
    calendar_dates = _make_trading_dates(
        date(2024, 1, 1), date(2024, 3, 15)
    )
    panel = _make_ohlcv_panel(
        calendar_dates, [1] * len(calendar_dates)
    )
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda *args, **kwargs: _fake_result(),
        calendar_dates,
    )

    late_feb_dates = [
        d for d in calendar_dates if date(2024, 2, 20) <= d <= date(2024, 2, 23)
    ]
    monkeypatch.setattr(
        WalkForwardSplitter,
        "split",
        lambda self, *args, **kwargs: [
            _make_split(
                "rolling_000",
                calendar_dates[:20],
                late_feb_dates,
            )
        ],
    )

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        count=0,
        holdout_start=date(2024, 2, 1),
        holdout_end=date(2025, 2, 1),
    )

    args = _walkforward_args(date(2024, 1, 1), date(2024, 3, 15))
    refused = runner.invoke(app, args)

    # today this succeeds (exit_code 0): current code never refuses, only
    # counts.
    assert refused.exit_code != 0
    assert "holdout" in refused.output.lower()
    refused_state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert refused_state["count"] == 0

    allowed = runner.invoke(app, args + ["--allow-holdout"])

    # today this fails at argument parsing: --allow-holdout does not exist yet.
    assert allowed.exit_code == 0
    allowed_state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert allowed_state["count"] == 1


# Item 4
def test_item_4_backtest_refuses_intersecting_holdout(
    monkeypatch, tmp_path
):
    calendar_dates = _make_trading_dates(
        date(2024, 6, 1), date(2024, 6, 10)
    )
    panel = _make_ohlcv_panel(
        calendar_dates, [1] * len(calendar_dates)
    )
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda *args, **kwargs: _fake_result(),
        calendar_dates,
    )

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        count=0,
        holdout_start=date(2024, 1, 1),
        holdout_end=date(2025, 1, 1),
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
            "2024-06-01",
            "--end",
            "2024-06-10",
            "--no-tradable-filter",
        ],
    )

    # today this succeeds: `nq backtest` never imports or consults HoldoutLock
    # at all.
    assert result.exit_code != 0
    assert "holdout" in result.output.lower()


# Item 4
def test_item_4_sweep_refuses_intersecting_holdout(monkeypatch, tmp_path):
    calendar_dates = _make_trading_dates(
        date(2024, 6, 1), date(2024, 6, 10)
    )
    panel = _make_ohlcv_panel(
        calendar_dates, [1] * len(calendar_dates)
    )
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda *args, **kwargs: _fake_result(),
        calendar_dates,
    )

    sweep_config = tmp_path / "sweep.yaml"
    sweep_config.write_text(
        yaml.safe_dump(
            {
                "strategy": "volume_breakout",
                "base_params": _VALID_VOLUME_BREAKOUT_PARAMS,
                "sweep": {"volume_z_threshold": [2.0, 2.5]},
            }
        ),
        encoding="utf-8",
    )

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        count=0,
        holdout_start=date(2024, 1, 1),
        holdout_end=date(2025, 1, 1),
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(sweep_config),
            "--start",
            "2024-06-01",
            "--end",
            "2024-06-10",
        ],
    )

    # today this succeeds: `nq sweep` never consults HoldoutLock either.
    assert result.exit_code != 0
    assert "holdout" in result.output.lower()


# Item 5
def test_item_5_tilt_uses_shared_holdout_lock_path(monkeypatch, tmp_path):
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.settings as settings_mod

    calendar_dates = _make_trading_dates(
        date(2020, 1, 1), date(2026, 1, 1)
    )
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(
            lambda cls, *args, **kwargs: _FakeCalendar(calendar_dates)
        ),
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    real_holdout_lock = HoldoutLock
    recorded_paths: list[Path] = []

    class _SpyHoldoutLock(real_holdout_lock):
        def __init__(self, *args, **kwargs):
            if "path" in kwargs:
                recorded_paths.append(Path(kwargs["path"]))
            elif args:
                recorded_paths.append(Path(args[0]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tilt_module, "HoldoutLock", _SpyHoldoutLock)

    panel_dates = [date(2020, 1, 1)]
    panel = _make_ohlcv_panel(panel_dates, [1])

    with pytest.raises(ValueError):
        run_tilt(
            panel=panel,
            config=TiltConfig(
                start=date(2020, 1, 1),
                end=date(2025, 9, 1),
            ),
            contract=minimal_contract(),
        )

    # today this records Path('/tmp/nifty_quant_holdout_lock.json').
    assert recorded_paths
    assert recorded_paths[0] == (
        settings_mod.RESULTS_ROOT / "holdout_lock.json"
    )


def test_item_5_tilt_source_has_no_tmp_holdout_path():
    source = inspect.getsource(tilt_module)

    assert "/tmp" not in source


# Item 6
def test_item_6_holdout_boundary_disagreement_is_reported(tmp_path):
    dates_v1 = _make_trading_dates(
        date(2020, 1, 1), date(2024, 6, 30)
    )
    lock = HoldoutLock(path=tmp_path / "lock.json")
    stored_window = lock.holdout_range(dates_v1)

    dates_v2 = _make_trading_dates(
        date(2020, 1, 1), date(2025, 1, 31)
    )

    with pytest.raises(Exception) as excinfo:
        lock.holdout_range(dates_v2)

    # must fail against current code: holdout_range() never persists or
    # compares against a stored boundary at all, so no disagreement is ever
    # detected or raised.
    message = str(excinfo.value)
    assert stored_window[1].isoformat() in message
    assert dates_v2[-1].isoformat() in message


# Item 7
def test_item_7_annotates_legacy_reads_without_changing_count_or_log(
    tmp_path,
):
    # ASSUMED API: the spec names no callable for section D's reconciliation
    # operation. This test assumes annotate_legacy_reads(reason: str) -> int,
    # inferred from HoldoutLock.read_count and HoldoutLock.record_read.
    entries = [
        {
            "ts": f"2026-01-01T00:00:0{i}Z",
            "reason": f"walkforward split rolling_{i:03d}",
        }
        for i in range(6)
    ]
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps({"count": 6, "log": entries}),
        encoding="utf-8",
    )

    lock = HoldoutLock(path=path)
    n = lock.annotate_legacy_reads(
        reason="false positive: manufactured holdout, F10"
    )

    assert n == 6
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["count"] == 6
    assert len(state["log"]) == 6
    for entry in state["log"]:
        assert any(
            "false positive" in str(value)
            for value in entry.values()
        )


# Item 8
def test_item_8_check_cash_non_negative_strictness_nan_and_floor():
    with (
        guards.strictness(guards.Strictness.FULL),
        pytest.raises(guards.ContractViolation) as excinfo,
    ):
        guards.check_cash_non_negative(-100.0)
    assert "-100" in str(excinfo.value)

    # the F6 lesson -- a naive `abs(cash) > tol`-style magnitude check would
    # let this NaN pass silently, since `abs(nan) > 0.0` is False in Python;
    # an explicit `not np.isfinite(cash)` branch is required.
    with (
        guards.strictness(guards.Strictness.FULL),
        pytest.raises(guards.ContractViolation),
    ):
        guards.check_cash_non_negative(float("nan"))

    with guards.strictness(guards.Strictness.CHEAP):
        guards.check_cash_non_negative(-100.0)

    with guards.strictness(guards.Strictness.OFF):
        guards.check_cash_non_negative(-100.0)

    with guards.strictness(guards.Strictness.FULL):
        guards.check_cash_non_negative(-50.0, floor=-100.0)
        with pytest.raises(guards.ContractViolation):
            guards.check_cash_non_negative(-150.0, floor=-100.0)


# Item 9
def test_item_9_allow_holdout_records_one_read_for_three_splits(
    monkeypatch, tmp_path
):
    calendar_dates = _make_trading_dates(
        date(2023, 6, 1), date(2025, 12, 31)
    )
    panel = _make_ohlcv_panel(
        calendar_dates, [1] * len(calendar_dates)
    )
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        lambda *args, **kwargs: _fake_result(),
        calendar_dates,
    )

    split_tests = [
        _make_trading_dates(date(2024, 1, 8), date(2024, 1, 12)),
        _make_trading_dates(date(2024, 6, 3), date(2024, 6, 7)),
        _make_trading_dates(date(2025, 11, 3), date(2025, 11, 7)),
    ]
    monkeypatch.setattr(
        WalkForwardSplitter,
        "split",
        lambda self, *args, **kwargs: [
            _make_split(
                f"rolling_{i:03d}",
                calendar_dates[:100],
                test_dates,
            )
            for i, test_dates in enumerate(split_tests)
        ],
    )

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        count=0,
        holdout_start=date(2024, 1, 1),
        holdout_end=date(2025, 12, 31),
    )

    result = runner.invoke(
        app,
        _walkforward_args(date(2024, 1, 1), date(2025, 12, 31))
        + ["--allow-holdout"],
    )

    # ADJUDICATED by the lead, specs/holdout_integrity.md AMENDMENT 2 item 1. This suite
    # originally asserted `count == 1` (one read per RUN); the independent suite asserted
    # one per intersecting split. No implementation can satisfy both, so the disagreement
    # was escalated rather than averaged, and PER-SPLIT won on scientific grounds:
    #
    #   the counter records how many times out-of-sample data has been LOOKED AT, because
    #   each look is a chance to select on it. Three intersecting splits produce three
    #   out-of-sample observations; calling that "1" understates the multiple-testing
    #   exposure -- the same error as `effective_n_trials` reporting 3.000 where the
    #   honest value is 1.055.
    #
    # Per-split also keeps the count consistent with the per-split audit log.
    assert result.exit_code == 0
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == len(split_tests) == 3
