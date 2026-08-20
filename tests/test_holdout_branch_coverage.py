"""Deliberate coverage-restoration tests for the three modules whose holdout branches
regressed once `tests/conftest.py` grew an autouse fixture that redirects every test
onto a far-future (2099) holdout boundary (see conftest.py `_isolated_holdout_lock`).

Every test here asserts real behaviour at the point it forces execution through --
never a bare "the line ran" call. Where a listed branch turned out to be structurally
unreachable, that is reported in the module docstring rather than forced with a hack.

FINDING (not fixed here -- source is out of scope): `HoldoutLock.holdout_range`
(src/nifty_quant/research/splits.py line 248, branch 237->248) is unreachable through
any external caller. The block guarding it is only entered when
`candidate_end > stored_end` (the early return at line 229 already handles
`candidate_end <= stored_end`). The equality check at line 237,
`(candidate_start, candidate_end) != (stored_start, stored_end)`, can therefore never
be False inside that block, because its second element (`candidate_end`) is, by the
guard that got us there, strictly greater than `stored_end`. Every path into this
block ends in the `raise HoldoutBoundaryError(...)` at line 238; the `return
stored_start, stored_end` at line 248 cannot be reached without either editing the
source or fabricating a caller that violates `candidate_end = trading_dates[-1]`
(which would mean testing something other than this method). Not forced.
"""

from __future__ import annotations

import datetime
import json
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import nifty_quant.guards as guards
from nifty_quant.backtest.engine import BacktestResult, build_daily
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.guards import ContractViolation, check_cash_non_negative
from nifty_quant.research.splits import HoldoutLock
from nifty_quant.universe.static import Universe

runner = CliRunner()

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Shared helpers (deliberately self-contained -- not imported from other test
# files, per the "do not modify any other test file" constraint).
# ---------------------------------------------------------------------------


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


def _seed_lock(path: Path, *, count: int, holdout_start: date, holdout_end: date, log=None) -> None:
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


def _patch_common(monkeypatch, tmp_path, panel, run_backtest_stub, calendar_dates):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(engine_mod, "run_backtest", run_backtest_stub)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, *args, **kwargs: _FakeCalendar(calendar_dates)),
    )


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


def _write_strategy_config(path: Path, *, volume_z_threshold: float) -> None:
    params = dict(_VALID_VOLUME_BREAKOUT_PARAMS)
    params["volume_z_threshold"] = volume_z_threshold
    path.write_text(
        yaml.safe_dump({"strategy": "volume_breakout", "params": params}), encoding="utf-8"
    )


def _walkforward_args(config_path: str, start: date, end: date) -> list[str]:
    return [
        "walkforward",
        "--strategy",
        "volume_breakout",
        "--config",
        config_path,
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


# ---------------------------------------------------------------------------
# splits.py -- HoldoutLock.holdout_range / annotate_legacy_reads
# ---------------------------------------------------------------------------


def test_holdout_range_non_dict_json_is_treated_as_missing_state(tmp_path):
    """Covers branch 210->213: `loaded` parses but is not a dict (e.g. a JSON array),
    so `state` must fall back to the default `{"count": 0, "log": []}` rather than
    crashing or silently adopting the array. That default lacks "holdout_start", so
    the method must then treat this as first-ever initialisation and persist a fresh
    boundary -- proving the fallback actually took effect, not just that no
    exception was raised.
    """
    path = tmp_path / "lock.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    lock = HoldoutLock(path=path)
    dates = _make_trading_dates(date(2024, 1, 1), date(2024, 6, 30))

    start, end = lock.holdout_range(dates)

    assert end == dates[-1]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["holdout_start"] == start.isoformat()
    assert persisted["holdout_end"] == end.isoformat()
    # count/log were not in the malformed file; the default must have been used,
    # not an attempt to preserve the (nonexistent) prior count/log.
    assert persisted["count"] == 0
    assert persisted["log"] == []


def test_annotate_legacy_reads_missing_path_returns_zero(tmp_path):
    """Covers line 315-316: no lock file exists at all -- must return 0 without
    creating one (annotate is a read-then-maybe-write operation, never a creator)."""
    path = tmp_path / "does_not_exist.json"
    lock = HoldoutLock(path=path)

    result = lock.annotate_legacy_reads(reason="audit")

    assert result == 0
    assert not path.exists()


def test_annotate_legacy_reads_non_dict_json_treated_as_empty(tmp_path):
    """Covers branch 321->324: the file exists but its JSON root is not a dict, so
    the log must be treated as empty (0 annotated) rather than raising or reading
    through a non-dict `.get`."""
    path = tmp_path / "lock.json"
    path.write_text(json.dumps("not-a-dict"), encoding="utf-8")
    lock = HoldoutLock(path=path)

    result = lock.annotate_legacy_reads(reason="audit")

    assert result == 0


def test_annotate_legacy_reads_skips_already_annotated_entries(tmp_path):
    """Covers branch 327->326 (loop continues past an entry that already carries an
    'annotation', per the documented idempotence contract) together with the
    annotated>0 direction of 331->334. One of two log entries is pre-annotated; only
    the other gets annotated, and the pre-annotated one is left untouched."""
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "count": 2,
                "log": [
                    {"ts": "2024-01-01T00:00:00+00:00", "reason": "old", "annotation": "prior"},
                    {"ts": "2024-01-02T00:00:00+00:00", "reason": "new"},
                ],
                "holdout_start": "2024-06-01",
                "holdout_end": "2025-06-01",
            }
        ),
        encoding="utf-8",
    )
    lock = HoldoutLock(path=path)

    annotated = lock.annotate_legacy_reads(reason="fresh audit")

    assert annotated == 1
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["log"][0]["annotation"] == "prior"  # untouched
    assert state["log"][1]["annotation"] == "fresh audit"  # newly annotated
    assert state["count"] == 2  # count/log length never mutated by annotation


def test_annotate_legacy_reads_all_already_annotated_is_a_noop(tmp_path):
    """Covers the False direction of branch 331->334: when every existing entry
    already carries an annotation, `annotated` stays 0 and `_atomic_write` must NOT
    be called (verified via mtime, since a write with identical content would
    otherwise be invisible to a value-only assertion)."""
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "count": 1,
                "log": [
                    {
                        "ts": "2024-01-01T00:00:00+00:00",
                        "reason": "old",
                        "annotation": "prior",
                    }
                ],
                "holdout_start": "2024-06-01",
                "holdout_end": "2025-06-01",
            }
        ),
        encoding="utf-8",
    )
    lock = HoldoutLock(path=path)
    mtime_before = path.stat().st_mtime_ns

    annotated = lock.annotate_legacy_reads(reason="second pass")

    assert annotated == 0
    assert path.stat().st_mtime_ns == mtime_before


# ---------------------------------------------------------------------------
# guards.py -- check_cash_non_negative
# ---------------------------------------------------------------------------


def test_check_cash_non_negative_resolves_unset_strictness_and_names_row(monkeypatch):
    """Covers branch 1102->1103 (module-level `_strictness` cache is None, so
    `get_strictness()` must be called to resolve it) and branch 1108->1109 (a `row`
    was supplied, so the raised message must name it). Forces FULL strictness via
    `NQ_STRICT=2` so resolution actually reaches the cash check rather than
    returning early."""
    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")

    with pytest.raises(ContractViolation) as exc_info:
        check_cash_non_negative(-5.0, row=42, floor=0.0)

    msg = str(exc_info.value)
    assert "row 42" in msg
    assert "cash=-5.0" in msg
    # get_strictness() must have run and cached a concrete level (not left None).
    assert guards._strictness is not None


# ---------------------------------------------------------------------------
# cli.py -- backtest holdout cluster (391-415)
# ---------------------------------------------------------------------------


def test_backtest_calendar_resolution_failure_is_reported(monkeypatch):
    """Covers lines 391-392: TradingCalendar.from_index_bars raising must be
    reported through `_fail`, naming the underlying exception, not propagate as an
    uncaught traceback."""
    import nifty_quant.calendar as calendar_mod

    def _raise(cls, symbol="NIFTY50"):
        raise RuntimeError("calendar boom")

    monkeypatch.setattr(calendar_mod.TradingCalendar, "from_index_bars", classmethod(_raise))

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-05",
        ],
    )

    assert result.exit_code != 0
    assert "backtest could not resolve calendar" in result.output
    assert "calendar boom" in result.output


def test_backtest_holdout_boundary_disagreement_is_reported(monkeypatch, tmp_path):
    """Covers lines 404-405: a stored boundary that disagrees with a fresh
    recomputation over the (fake) calendar must raise HoldoutBoundaryError, caught
    and reported via `_fail`, naming both boundaries."""
    calendar_dates = _make_trading_dates(date(2024, 1, 1), date(2024, 6, 10))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2020, 1, 1), holdout_end=date(2024, 1, 1))

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

    assert result.exit_code != 0
    assert "disagrees with a recomputation" in result.output
    assert "2020-01-01" in result.output
    assert "2024-01-01" in result.output


def test_backtest_allow_holdout_records_one_read(monkeypatch, tmp_path):
    """Covers branch 414->415 (holdout_intersects AND allow_holdout): with
    --allow-holdout the run proceeds and the lock's count is incremented exactly
    once."""
    calendar_dates = _make_trading_dates(date(2024, 6, 1), date(2024, 6, 10))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2024, 1, 1), holdout_end=date(2025, 1, 1))

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
            "--allow-holdout",
        ],
    )

    assert result.exit_code == 0, result.output
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == 1
    assert len(state["log"]) == 1


# ---------------------------------------------------------------------------
# cli.py -- walkforward holdout cluster (902-941) and DSR n_eff note (1204-1217)
# ---------------------------------------------------------------------------


def test_walkforward_feature_lookback_estimation_failure_falls_back_to_zero(monkeypatch, tmp_path):
    """Covers lines 902-903: when --feature-lookback is not given, the CLI estimates
    it from `registry.build(cfg).data_request().warmup_bars()`; if THAT raises, the
    fallback is 0.0 rather than propagating. Only the first `registry.build` call
    (the estimation one) is made to fail -- later calls (building the strategy to
    actually run each split) must still succeed, or this test could not observe the
    fallback taking effect versus the whole command just failing for an unrelated
    reason.
    """
    import nifty_quant.strategy.registry as registry_mod

    calendar_dates = _make_trading_dates(date(2024, 1, 1), date(2024, 3, 1))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2099, 1, 1), holdout_end=date(2099, 12, 31))

    real_build = registry_mod.build
    call_count = {"n": 0}

    def _flaky_build(cfg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("warmup estimation boom")
        return real_build(cfg)

    monkeypatch.setattr(registry_mod, "build", _flaky_build)

    cfg_path = tmp_path / "cfg.yaml"
    _write_strategy_config(cfg_path, volume_z_threshold=2.0)

    result = runner.invoke(
        app, _walkforward_args(str(cfg_path), date(2024, 1, 1), date(2024, 3, 1))
    )

    assert result.exit_code == 0, result.output
    # The estimation call really did fail and fall back, and later calls recovered:
    # the run must have completed (multiple registry.build calls happened) rather
    # than the whole command failing on the first raise.
    assert call_count["n"] >= 2


def test_walkforward_holdout_boundary_disagreement_is_reported(monkeypatch, tmp_path):
    """Covers lines 940-941: same disagreement guard as backtest's 404-405, but on
    the walkforward command's own HoldoutLock.holdout_range call."""
    calendar_dates = _make_trading_dates(date(2024, 1, 1), date(2024, 6, 10))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2020, 1, 1), holdout_end=date(2024, 1, 1))

    cfg_path = tmp_path / "cfg.yaml"
    _write_strategy_config(cfg_path, volume_z_threshold=2.0)

    result = runner.invoke(
        app, _walkforward_args(str(cfg_path), date(2024, 1, 1), date(2024, 6, 10))
    )

    assert result.exit_code != 0
    assert "disagrees with a recomputation" in result.output
    assert "2020-01-01" in result.output
    assert "2024-01-01" in result.output


def test_walkforward_effective_n_trials_below_two_reports_single_trial(monkeypatch, tmp_path):
    """Covers branch 1204->1211 and lines 1211-1212: two DIFFERENT trial configs
    (distinct config_hash) whose exploration artifacts, once aligned, are perfectly
    correlated (`_fake_result()`'s stub is a constant returns array regardless of
    split/config) must report effective_n_trials < 2 and skip
    expected_max_sharpe(), rather than silently inventing multiple-testing breadth.
    Two separate CLI invocations, sharing one tmp_path/trials.db, are required: a
    single run's own splits all share ONE config_hash, so build_trial_matrix keeps
    only one column per hash -- two distinct configs are needed for a second column.
    """
    calendar_dates = _make_trading_dates(date(2024, 1, 1), date(2024, 3, 1))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2099, 1, 1), holdout_end=date(2099, 12, 31))

    cfg1 = tmp_path / "cfg1.yaml"
    cfg2 = tmp_path / "cfg2.yaml"
    _write_strategy_config(cfg1, volume_z_threshold=2.0)
    _write_strategy_config(cfg2, volume_z_threshold=2.5)

    first = runner.invoke(app, _walkforward_args(str(cfg1), date(2024, 1, 1), date(2024, 3, 1)))
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, _walkforward_args(str(cfg2), date(2024, 1, 1), date(2024, 3, 1)))
    assert second.exit_code == 0, second.output
    assert "honest effective_n_trials=" in second.output
    assert "SR0 is reported as 0.0 instead of calling expected_max_sharpe()" in second.output


# ---------------------------------------------------------------------------
# cli.py -- sweep holdout cluster (1382-1406)
# ---------------------------------------------------------------------------


def _sweep_config_path(tmp_path: Path) -> Path:
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
    return sweep_config


def test_sweep_calendar_resolution_failure_for_holdout_check_is_reported(monkeypatch, tmp_path):
    """Covers lines 1382-1383: sweep's SECOND calendar resolution (the one used
    purely for the holdout check) must be reported distinctly from the first (the
    one used to default --end). --end is passed explicitly here so the first
    resolution is skipped entirely and only the holdout-check resolution can be the
    one that fails."""
    import nifty_quant.calendar as calendar_mod

    def _raise(cls, symbol="NIFTY50"):
        raise RuntimeError("calendar boom for holdout check")

    monkeypatch.setattr(calendar_mod.TradingCalendar, "from_index_bars", classmethod(_raise))

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(_sweep_config_path(tmp_path)),
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-10",
        ],
    )

    assert result.exit_code != 0
    assert "sweep could not resolve calendar for holdout check" in result.output
    assert "calendar boom for holdout check" in result.output


def test_sweep_holdout_boundary_disagreement_is_reported(monkeypatch, tmp_path):
    """Covers lines 1395-1396."""
    calendar_dates = _make_trading_dates(date(2024, 1, 1), date(2024, 6, 10))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2020, 1, 1), holdout_end=date(2024, 1, 1))

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(_sweep_config_path(tmp_path)),
            "--start",
            "2024-06-01",
            "--end",
            "2024-06-10",
        ],
    )

    assert result.exit_code != 0
    assert "disagrees with a recomputation" in result.output
    assert "2020-01-01" in result.output
    assert "2024-01-01" in result.output


def test_sweep_allow_holdout_records_one_read(monkeypatch, tmp_path):
    """Covers branch 1405->1406 and line 1406: --allow-holdout on an intersecting
    sweep must record exactly one read."""
    calendar_dates = _make_trading_dates(date(2024, 6, 1), date(2024, 6, 10))
    panel = _make_ohlcv_panel(calendar_dates, [1] * len(calendar_dates))
    _patch_common(monkeypatch, tmp_path, panel, lambda *a, **kw: _fake_result(), calendar_dates)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, count=0, holdout_start=date(2024, 1, 1), holdout_end=date(2025, 1, 1))

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(_sweep_config_path(tmp_path)),
            "--start",
            "2024-06-01",
            "--end",
            "2024-06-10",
            "--allow-holdout",
        ],
    )

    assert result.exit_code == 0, result.output
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == 1
    assert len(state["log"]) == 1


# ---------------------------------------------------------------------------
# cli.py -- tilt (1679-1680, 1724->exit, 1730-1733)
# ---------------------------------------------------------------------------

_TILT_N_SYMBOLS = 5
_TILT_SYMBOLS = tuple(f"SYM{i}" for i in range(_TILT_N_SYMBOLS))


def _ts_for(day: date, hhmm: str) -> int:
    hour, minute = (int(p) for p in hhmm.split(":"))
    stamp = pd.Timestamp(day.year, day.month, day.day, hour, minute, tz=_IST)
    return int(stamp.tz_convert("UTC").timestamp())


def _build_tilt_panel(sessions, symbols=_TILT_SYMBOLS) -> Panel:
    sessions_sorted = sorted(sessions, key=lambda item: item[0])
    ts_list: list[int] = []
    price_rows: list[np.ndarray] = []
    day_offsets = [0]
    row = 0
    for day, bars in sessions_sorted:
        for hhmm in sorted(bars.keys()):
            ts_list.append(_ts_for(day, hhmm))
            price_rows.append(np.asarray(bars[hhmm], dtype=np.float64))
            row += 1
        day_offsets.append(row)
    ts = np.array(ts_list, dtype=np.int64)
    price_arr = np.stack(price_rows).astype(np.float32)
    volume_arr = np.full(price_arr.shape, 1_000_000.0, dtype=np.float32)
    dates_arr = np.array([d for d, _ in sessions_sorted], dtype=object)
    return Panel(
        fields={"open": price_arr, "close": price_arr.copy(), "volume": volume_arr},
        symbols=symbols,
        ts=ts,
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=dates_arr,
    )


def _flat_row(base: float = 100.0, n: int = _TILT_N_SYMBOLS) -> np.ndarray:
    return np.array([base + i * 0.01 for i in range(n)], dtype=np.float64)


def _business_days(start: date, n: int) -> list[date]:
    out: list[date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _complete_tilt_panel(dates: list[date], *, warmup_day: date | None = None) -> Panel:
    """Every session has both checkpoint bars present -- zero missing-checkpoint
    warnings, so `result.warnings` is empty and the CLI's post-run warnings block is
    skipped entirely (branch 1724->exit).

    `warmup_day`, if given, is an EXTRA session dated before `dates[0]` that is
    included in the panel but not in the requested --start/--end range. tilt.py
    builds its overnight feature ("d-1 close") on the FULL panel before slicing to
    the requested window (see `run_tilt`, "Build overnight feature on full panel
    (so left-edge session has valid d-1)"), so without a day before the window's
    first requested date, that first date always has zero valid d-1 closes and is
    unconditionally warned/skipped -- unrelated to the branch under test here.
    """
    sessions = []
    all_dates = ([warmup_day] if warmup_day is not None else []) + list(dates)
    for day in all_dates:
        entry_row = _flat_row(100.0)
        exit_row = entry_row + 0.05
        sessions.append((day, {"09:16": entry_row, "15:20": exit_row}))
    return _build_tilt_panel(sessions)


def test_tilt_start_after_end_fails():
    """Covers line 1680."""
    result = runner.invoke(app, ["tilt", "--start", "2024-02-01", "--end", "2024-01-01"])
    assert result.exit_code != 0
    assert "is after" in result.output


def test_tilt_without_warnings_reaches_normal_exit(monkeypatch, tmp_path):
    """Covers branch 1724->exit: a fully-populated panel produces zero warnings, so
    the `if result.warnings:` block must be skipped and the command must still exit
    0 (proving the skip, not just the absence of a crash)."""
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.universe.static as universe_mod

    dates = _business_days(date(2022, 1, 3), 14)
    panel = _complete_tilt_panel(dates, warmup_day=date(2021, 12, 31))

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="all_equity", symbols=_TILT_SYMBOLS),
    )

    result = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
            "--tilt",
            "mild",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Warnings:" not in result.output


def test_tilt_value_error_from_run_tilt_is_reported(monkeypatch, tmp_path):
    """Covers lines 1730-1731: run_tilt's own ValueError ("No sessions with >= 5
    valid names", raised for a universe with too few symbols) must be caught and
    reported via `_fail(str(exc))`, distinct from the generic Exception handler."""
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.universe.static as universe_mod

    d0, d1 = date(2022, 1, 3), date(2022, 1, 4)
    sessions = [
        (d0, {"09:16": _flat_row(100.0, n=3), "15:20": _flat_row(100.0, n=3) + 0.05}),
        (d1, {"09:16": _flat_row(100.0, n=3), "15:20": _flat_row(100.0, n=3) + 0.05}),
    ]
    panel = _build_tilt_panel(sessions, symbols=("X0", "X1", "X2"))

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="tiny", symbols=("X0", "X1", "X2")),
    )

    result = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            d0.isoformat(),
            "--end",
            d1.isoformat(),
            "--tilt",
            "aggressive",
            "--smoothing",
            "1.0",
        ],
    )

    assert result.exit_code != 0
    assert "No sessions with >= 5 valid names" in result.output
    assert "Error running tilt" not in result.output


def test_tilt_unexpected_exception_is_reported(monkeypatch, tmp_path):
    """Covers lines 1732-1733: a non-ValueError exception raised inside the try
    block (here, from run_tilt itself) must be caught by the generic Exception
    handler and prefixed distinctly from the ValueError path."""
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.research.tilt as tilt_mod
    import nifty_quant.universe.static as universe_mod

    dates = _business_days(date(2022, 1, 3), 5)
    panel = _complete_tilt_panel(dates)

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="all_equity", symbols=_TILT_SYMBOLS),
    )

    def _boom(panel, config):
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(tilt_mod, "run_tilt", _boom)

    result = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
            "--tilt",
            "mild",
        ],
    )

    assert result.exit_code != 0
    assert "Error running tilt: unexpected boom" in result.output
