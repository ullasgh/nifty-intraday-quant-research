"""Red-state TDD spec file for ``specs/holdout_integrity.md`` (author track B).

This is ONE OF TWO independent readings of the spec -- the other author writes
``test_holdout_integrity_a.py`` from the same spec alone, without seeing this file, and
this file was written without seeing theirs. Any disagreement between the two suites is
itself a finding, per ``CLAUDE.md`` rule 1.

The behaviour under test (F10/F10b/F11) DOES NOT EXIST YET. This file is expected to fail
-- some tests with an assertion failure, some with an ``AttributeError``/``ImportError``
where this author had to guess an API surface the spec leaves unpinned (each such import is
local to its own test function, so a wrong guess only fails that one test, not the whole
module). Nothing here weakens an assertion to make it pass, and nothing under ``data/`` or
the real ``results/holdout_lock.json`` is ever touched -- every ``HoldoutLock`` in this file
is constructed against a ``tmp_path`` file.

Spec defects, ambiguities and internal contradictions found by close reading
-----------------------------------------------------------------------------

1. The spec's narrative says "``results/holdout_lock.json`` shows 6 reads" and "Reconcile
   the existing six entries", but the REAL file on disk (read-only, never touched by this
   suite) currently has ``"count": 7`` with 7 log entries, not 6 -- almost certainly more
   concurrent short-window runs happened after the spec text was written and before this
   file was authored. This suite treats the exact historical count as immaterial: every test
   here constructs its OWN seeded tmp_path file with a chosen number of legacy entries, and
   never reads or asserts on the real file's count.

2. Item 8's pinned signature is ``check_cash_non_negative(cash, *, floor=0.0)`` -- no ``row``
   parameter -- yet the prose two lines above says the raise must "name the row and the
   balance". Those two sentences are in tension: the given signature has nothing to name a
   row WITH. This suite resolves the tension conservatively: it requires the raised message
   to contain the numeric cash value (unambiguous, "the balance"), but does NOT require a row
   identifier to appear, since the pinned signature provides no row argument to supply one.

3. The spec never gives an API for "annotate the six historical entries" (item 7) -- no
   method name, no field name for the annotation, no statement of whether this lives on
   ``HoldoutLock`` or elsewhere. This suite ASSUMES a `HoldoutLock.annotate_legacy_reads(self,
   reason: str) -> int` method (the natural read/write counterpart to `record_read`) that
   adds an `"annotation"` key to every log entry that doesn't already have one, leaves
   `count` and every other field untouched, and returns the number of entries it annotated.
   If the real implementation names this differently, only
   `test_item7_historical_entries_annotated_not_deleted` needs to change.

4. The spec never says HOW `cli.py` and `research/tilt.py` come to share one lock path --
   a shared constant, a shared helper function, or just both independently reading
   `settings.RESULTS_ROOT`. This suite checks the OUTCOME two ways: (a) a naming-agnostic,
   spec-pinned check that the literal string ``/tmp`` never appears in either module's
   source (spec: "the `/tmp` path appears nowhere" -- unambiguous regardless of API shape),
   and (b) an ASSUMED shared helper `nifty_quant.research.splits.default_holdout_lock_path()`
   returning `settings.RESULTS_ROOT / "holdout_lock.json"`, isolated in its own test so a
   differently-named real implementation only fails that one sub-test, not (a).

5. Item 4 ("`nq backtest` and `nq sweep` must refuse a holdout-intersecting window") never
   says whether these two commands also grow their own `--allow-holdout` escape hatch (as
   `walkforward` explicitly does per item 3) or refuse unconditionally with no override. This
   suite only asserts the DEFAULT (no override flag passed) refuses; it takes no position on
   whether an escape flag exists for `backtest`/`sweep`.

6. Item 6 pins "RAISE, naming both" but not an exception class or message format. This suite
   accepts any `Exception` subclass and only requires both the stored and the recomputed
   boundary date strings to appear somewhere in `str(exc)`.

7. `holdout_months=12` combined with the CURRENT `holdout_range`'s clamp-to-first-available
   fallback (`next((d for d in trading_dates if d >= cutoff), holdout_end)`) means a run whose
   window is SHORTER than 12 months manufactures a "holdout" spanning its ENTIRE window, not
   merely its trailing year -- worth flagging explicitly, since it makes F10 maximally severe
   for exactly the short exploratory runs this research program does most. Verified directly
   in `test_item1_stored_boundary_used_not_recomputed`'s setup below.

8. Section A says the boundary is "stored in the lock file" using new keys `holdout_start` /
   `holdout_end`, but the CURRENT `record_read` / `_atomic_write` schema only knows `count`
   and `log`. The spec never describes a migration path for an EXISTING `count`/`log`-only
   file (like the real one) gaining those two new keys for the first time. This suite always
   seeds its own tmp_path files with the full expected POST-fix shape rather than exercising
   an upgrade path, since the spec does not describe one.

CLI-level tests below reuse a monkeypatch-everything-real-data-touching pattern (fake
`Panel`, fake `Universe`, a stub `run_backtest`, a fake `TradingCalendar`) matching the
established convention already in this repo (`tests/test_cli_coverage.py`); this author
wrote these fixtures independently for this file rather than importing that module's
private helpers, to keep this suite self-contained.
"""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.daily import build_daily
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.research.splits import HoldoutLock, WalkForwardSplitter
from nifty_quant.universe.static import Universe

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixtures (self-contained; not imported from any other test file)
# ---------------------------------------------------------------------------

# A long synthetic trading calendar. `HoldoutLock(path=None).holdout_range(LONG_DATES)`
# (today's un-fixed, purely-computed behaviour) gives the TRUE global boundary any
# fixed implementation must agree on when handed the full calendar.
LONG_DATES = pd.bdate_range("2020-01-02", periods=1600).date.tolist()

# A short, EARLY slice of LONG_DATES, well before the true holdout boundary. Verified
# (see module docstring point 7) to manufacture its own fake "holdout" spanning its
# entire window under today's code.
SHORT_DATES = [d for d in LONG_DATES if d <= LONG_DATES[179]]


def _compute(dates: list[datetime.date]) -> tuple[datetime.date, datetime.date]:
    """The TRUE boundary a correct implementation should agree on for `dates`.

    Each call gets its OWN empty scratch directory, so the lock it builds has no
    stored boundary and must derive one from `dates` alone -- which is what makes
    this an independent oracle rather than an echo of the lock under test.

    The path is NOT a throwaway detail. This helper originally passed
    `/nonexistent/nq_holdout_unused.json` on the reasoning that `holdout_range`
    does no file I/O; that reasoning described the OLD behaviour. The fixed
    `holdout_range` persists the boundary it derives (splits.py:203 mkdir), so the
    nonexistent path became an `OSError: Read-only file system` that took out six
    tests. A scratch path that is merely unused is fine; a scratch path that
    cannot exist is a bet on the implementation not touching the disk.
    """
    scratch = Path(tempfile.mkdtemp(prefix="nq_holdout_oracle_"))
    return HoldoutLock(path=scratch / "lock.json").holdout_range(dates)


def _seed_lock(
    path: Path,
    *,
    holdout_start: datetime.date,
    holdout_end: datetime.date,
    count: int = 0,
    log: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "count": count,
                "log": log if log is not None else [],
            }
        ),
        encoding="utf-8",
    )


def _session_ts(session_date: datetime.date, n_bars: int, start_hhmm=(9, 15)) -> np.ndarray:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(session_date, datetime.time(*start_hhmm), tzinfo=ist)
    return np.asarray([int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64)


def _make_ohlcv_panel(session_dates, bars_per_session, symbols=("A", "B")) -> Panel:
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


def _fake_result(n: int = 3) -> BacktestResult:
    gross = np.full(n, -0.0005, dtype=np.float64)
    turnover = np.full(n, 1.0, dtype=np.float64)
    net = gross - 0.0002 * turnover
    equity_curve = np.cumprod(1.0 + net) * 1e7
    daily = build_daily(
        np.arange(n, dtype=np.int64),
        equity_curve,
        net,
        gross,
        turnover,
        np.arange(n, dtype=np.int64),
        initial_capital=1e7,
    )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=net,
        positions=np.zeros((n, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=1,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
        daily=daily,
    )


class _FakeCalendar:
    def __init__(self, dates):
        self._dates = list(dates)

    def session_dates(self, start=None, end=None, *, usable_only=True):
        return [
            d for d in self._dates if (start is None or d >= start) and (end is None or d <= end)
        ]


def _patch_common(monkeypatch, tmp_path, panel, all_dates):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=("A", "B"))
    )
    monkeypatch.setattr(
        engine_mod,
        "run_backtest",
        lambda strat, panel, config, *, tradable=None: _fake_result(n=3),
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)),
    )


def _walkforward_base_args(strategy="volume_breakout"):
    return [
        "walkforward",
        "--strategy",
        strategy,
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--feature-lookback",
        "0",
    ]


# ---------------------------------------------------------------------------
# Item 1: stored boundary used, not recomputed
# ---------------------------------------------------------------------------


def test_item1_stored_boundary_used_not_recomputed(tmp_path):
    """A run over a SHORT window must see the SAME holdout window as a run over the
    full calendar, because the boundary is read from storage, not recomputed from
    whatever `trading_dates` the caller happens to hand in. Must fail against current
    code -- today `holdout_range` ignores `self.path` entirely and always recomputes
    from its argument, so the two calls below disagree."""
    lock_path = tmp_path / "holdout_lock.json"

    full_boundary = HoldoutLock(path=lock_path).holdout_range(LONG_DATES)
    short_boundary = HoldoutLock(path=lock_path).holdout_range(SHORT_DATES)

    assert short_boundary == full_boundary, (
        f"a short-window run computed its own boundary {short_boundary} instead of "
        f"reusing the stored one {full_boundary} -- the boundary is not fixed"
    )
    # Sanity check on the test's own construction: a naive per-call recompute over
    # SHORT_DATES really would disagree with the full-calendar boundary (see docstring
    # point 7), so `short_boundary == full_boundary` is a meaningful assertion, not a
    # tautology.
    naive_short_recompute = _compute(SHORT_DATES)
    assert naive_short_recompute != full_boundary


# ---------------------------------------------------------------------------
# Item 2: a short-window run records ZERO reads
# ---------------------------------------------------------------------------


def test_item2_short_window_run_records_zero_reads(monkeypatch, tmp_path):
    """A short, EARLY-window walkforward run, against an already-initialised lock
    whose STORED true boundary lies entirely in the future, must record ZERO reads.
    Today it manufactures its own fake holdout spanning its entire (short) window and
    records exactly one -- this assertion fails against current code."""
    true_start, true_end = _compute(LONG_DATES)
    assert SHORT_DATES[-1] < true_start  # construction invariant

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, holdout_start=true_start, holdout_end=true_end)

    panel = _make_ohlcv_panel(SHORT_DATES, [3] * len(SHORT_DATES))
    _patch_common(monkeypatch, tmp_path, panel, LONG_DATES)

    result = runner.invoke(
        app,
        _walkforward_base_args()
        + [
            "--start",
            SHORT_DATES[0].isoformat(),
            "--end",
            SHORT_DATES[-1].isoformat(),
            "--train-years",
            "0.5",
            "--test-years",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output

    holdout = HoldoutLock(path=lock_path)
    assert holdout.read_count() == 0, (
        f"short-window run recorded {holdout.read_count()} read(s) against a manufactured "
        "holdout despite never touching the TRUE stored window"
    )


# ---------------------------------------------------------------------------
# Item 3: a genuinely intersecting run refuses without --allow-holdout, and
# records exactly one read WITH it.
# ---------------------------------------------------------------------------


def test_item3_intersecting_run_refuses_without_allow_holdout(monkeypatch, tmp_path):
    true_start, true_end = _compute(LONG_DATES)
    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, holdout_start=true_start, holdout_end=true_end)

    panel = _make_ohlcv_panel(LONG_DATES, [3] * len(LONG_DATES))
    _patch_common(monkeypatch, tmp_path, panel, LONG_DATES)

    result = runner.invoke(
        app,
        _walkforward_base_args()
        + [
            "--start",
            LONG_DATES[0].isoformat(),
            "--end",
            LONG_DATES[-1].isoformat(),
            "--train-years",
            "0.5",
            "--test-years",
            "0.1",
        ],
    )
    assert result.exit_code != 0, "a run whose final split intersects the TRUE holdout must refuse"
    assert "holdout" in result.output.lower()

    holdout = HoldoutLock(path=lock_path)
    assert holdout.read_count() == 0, "a refused run must not record a read"


def test_item3_intersecting_run_with_allow_holdout_records_exactly_one(monkeypatch, tmp_path):
    true_start, true_end = _compute(LONG_DATES)
    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, holdout_start=true_start, holdout_end=true_end)

    # Construction invariant: with these train/test years against LONG_DATES, exactly
    # one of the six splits intersects the true holdout (verified interactively against
    # the real WalkForwardSplitter before writing this test).
    splitter = WalkForwardSplitter(train_years=0.5, test_years=0.1)
    splits = splitter.split(LONG_DATES, feature_lookback=0.0)
    n_intersecting = sum(1 for s in splits if s.test[1] >= true_start)
    assert n_intersecting == 1, "test construction invariant broken -- fix the split params"

    panel = _make_ohlcv_panel(LONG_DATES, [3] * len(LONG_DATES))
    _patch_common(monkeypatch, tmp_path, panel, LONG_DATES)

    result = runner.invoke(
        app,
        _walkforward_base_args()
        + [
            "--start",
            LONG_DATES[0].isoformat(),
            "--end",
            LONG_DATES[-1].isoformat(),
            "--train-years",
            "0.5",
            "--test-years",
            "0.1",
            "--allow-holdout",
        ],
    )
    assert result.exit_code == 0, result.output

    holdout = HoldoutLock(path=lock_path)
    assert holdout.read_count() == 1, (
        f"expected exactly 1 recorded read for 1 intersecting split with --allow-holdout, "
        f"got {holdout.read_count()}"
    )


# ---------------------------------------------------------------------------
# Item 4: `nq backtest` and `nq sweep` refuse a holdout-intersecting window
# ---------------------------------------------------------------------------


def test_item4_backtest_refuses_holdout_intersecting_window(monkeypatch, tmp_path):
    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        holdout_start=datetime.date(2024, 6, 1),
        holdout_end=datetime.date(2025, 6, 1),
    )

    session_dates = [datetime.date(2024, 8, 1), datetime.date(2024, 8, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    _patch_common(monkeypatch, tmp_path, panel, session_dates)

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            "2024-08-01",
            "--end",
            "2024-08-02",
        ],
    )
    assert result.exit_code != 0, "backtest over a holdout-intersecting window must refuse"
    assert "holdout" in result.output.lower()


def test_item4_sweep_refuses_holdout_intersecting_window(monkeypatch, tmp_path):
    import yaml

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(
        lock_path,
        holdout_start=datetime.date(2024, 6, 1),
        holdout_end=datetime.date(2025, 6, 1),
    )

    sweep_config = {
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
            "sigma_floor": 1.0e-5,
            "max_weight": 0.10,
            "gross": 1.0,
        },
        "sweep": {"volume_z_threshold": [2.0, 2.5]},
    }
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep_config), encoding="utf-8")

    session_dates = [datetime.date(2024, 8, 1), datetime.date(2024, 8, 2)]
    panel = _make_ohlcv_panel(session_dates, [3, 3])
    _patch_common(monkeypatch, tmp_path, panel, session_dates)

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(sweep_path),
            "--start",
            "2024-08-01",
            "--end",
            "2024-08-02",
        ],
    )
    assert result.exit_code != 0, "sweep over a holdout-intersecting window must refuse"
    assert "holdout" in result.output.lower()


# ---------------------------------------------------------------------------
# Item 5: `cli` and `research/tilt.py` share ONE lock path; `/tmp` appears nowhere
# ---------------------------------------------------------------------------


def test_item5_no_tmp_path_anywhere_in_source():
    import inspect

    import nifty_quant.cli as cli_mod
    import nifty_quant.research.tilt as tilt_mod

    cli_src = Path(inspect.getsourcefile(cli_mod)).read_text(encoding="utf-8")
    tilt_src = Path(inspect.getsourcefile(tilt_mod)).read_text(encoding="utf-8")

    for name, src in (("cli.py", cli_src), ("research/tilt.py", tilt_src)):
        assert "/tmp" not in src, f"{name} still references a /tmp path"
        assert "nifty_quant_holdout_lock" not in src, (
            f"{name} still references the hardcoded /tmp lock filename"
        )


def test_item5_shared_lock_path_helper(monkeypatch, tmp_path):
    """Assumption (see module docstring point 4): a shared
    `default_holdout_lock_path()` helper both callers use."""
    import nifty_quant.settings as settings_mod

    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    from nifty_quant.research.splits import default_holdout_lock_path

    expected = tmp_path / "holdout_lock.json"
    assert default_holdout_lock_path() == expected


# ---------------------------------------------------------------------------
# Item 6: a stored boundary disagreeing with a recomputation RAISES, naming both
# ---------------------------------------------------------------------------


def test_item6_stored_boundary_disagreement_raises_naming_both(tmp_path):
    true_start, true_end = _compute(LONG_DATES)
    # Simulate "the dataset gained new sessions": the stored boundary is 30 days
    # earlier than what a fresh recomputation over LONG_DATES now produces.
    stale_start = true_start - datetime.timedelta(days=30)
    stale_end = true_end - datetime.timedelta(days=30)

    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, holdout_start=stale_start, holdout_end=stale_end)

    with pytest.raises(Exception) as exc_info:
        HoldoutLock(path=lock_path).holdout_range(LONG_DATES)

    message = str(exc_info.value)
    assert stale_start.isoformat() in message
    assert stale_end.isoformat() in message
    assert true_start.isoformat() in message
    assert true_end.isoformat() in message


# ---------------------------------------------------------------------------
# Item 7: the six historical entries are annotated, count unchanged, log intact
# ---------------------------------------------------------------------------


def test_item7_historical_entries_annotated_not_deleted(tmp_path):
    """Assumption (see module docstring point 3):
    `HoldoutLock.annotate_legacy_reads(reason)` marks pre-existing entries in place."""
    legacy_log = [
        {"ts": f"2026-08-19T0{i}:00:00+00:00", "reason": "walkforward split rolling_000"}
        for i in range(6)
    ]
    lock_path = tmp_path / "holdout_lock.json"
    lock_path.write_text(
        json.dumps({"count": 6, "log": legacy_log}), encoding="utf-8"
    )

    holdout = HoldoutLock(path=lock_path)
    reason = "false positive: boundary computed from caller's own window (F10)"
    n_annotated = holdout.annotate_legacy_reads(reason=reason)

    assert n_annotated == 6

    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == 6, "annotating must not reset or otherwise change the counter"
    assert len(state["log"]) == 6, "annotating must not delete or truncate entries"
    for i, entry in enumerate(state["log"]):
        assert entry["ts"] == legacy_log[i]["ts"]
        assert entry["reason"] == legacy_log[i]["reason"]
        assert entry.get("annotation") == reason


# ---------------------------------------------------------------------------
# Item 8: check_cash_non_negative raises on negative AND on NaN cash
# ---------------------------------------------------------------------------


def test_item8_check_cash_non_negative_raises_on_negative_cash():
    from nifty_quant import guards
    from nifty_quant.guards import check_cash_non_negative

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            check_cash_non_negative(-999.5, floor=0.0)
    assert "-999.5" in str(exc_info.value)


def test_item8_check_cash_non_negative_raises_on_nan_cash():
    """The F6 lesson, restated by the spec: a magnitude comparison (`abs(x) > tol`)
    silently PASSES NaN, so the guard needs an explicit `not np.isfinite(cash)` check
    beside the comparison, not instead of a comparison that would itself accept NaN."""
    from nifty_quant import guards
    from nifty_quant.guards import check_cash_non_negative

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            check_cash_non_negative(float("nan"), floor=0.0)
    message = str(exc_info.value).lower()
    assert "nan" in message or "finite" in message


def test_item8_check_cash_non_negative_passes_valid_cash():
    from nifty_quant import guards
    from nifty_quant.guards import check_cash_non_negative

    with guards.strictness(guards.Strictness.FULL):
        check_cash_non_negative(100.0, floor=0.0)  # must not raise


def test_item8_check_cash_non_negative_full_strictness_only(monkeypatch):
    """The spec pins this as a FULL-strictness-only guard, same as `check_accounting`.
    At a lower strictness level, even clearly negative/NaN cash must not raise."""
    from nifty_quant import guards
    from nifty_quant.guards import check_cash_non_negative

    with guards.strictness(guards.Strictness.CHEAP):
        check_cash_non_negative(-999.5, floor=0.0)  # must not raise below FULL
        check_cash_non_negative(float("nan"), floor=0.0)  # must not raise below FULL


# ---------------------------------------------------------------------------
# Item 9: the counter increments once per allowed read, never per split evaluated
# ---------------------------------------------------------------------------


def test_item9_counter_increments_once_per_allowed_read_not_per_split(monkeypatch, tmp_path):
    true_start, true_end = _compute(LONG_DATES)
    lock_path = tmp_path / "holdout_lock.json"
    _seed_lock(lock_path, holdout_start=true_start, holdout_end=true_end)

    # Construction invariant: 6 total splits, exactly 2 of which intersect the true
    # holdout (verified interactively against the real WalkForwardSplitter before
    # writing this test) -- so "once per allowed read" (2) is distinguishable both
    # from "once per split evaluated" (6) and from "once per intersecting split
    # counted twice by mistake" (4).
    splitter = WalkForwardSplitter(train_years=0.75, test_years=0.55)
    splits = splitter.split(LONG_DATES, feature_lookback=0.0)
    n_intersecting = sum(1 for s in splits if s.test[1] >= true_start)
    assert len(splits) == 6 and n_intersecting == 2, (
        "test construction invariant broken -- fix the split params"
    )

    panel = _make_ohlcv_panel(LONG_DATES, [3] * len(LONG_DATES))
    _patch_common(monkeypatch, tmp_path, panel, LONG_DATES)

    result = runner.invoke(
        app,
        _walkforward_base_args()
        + [
            "--start",
            LONG_DATES[0].isoformat(),
            "--end",
            LONG_DATES[-1].isoformat(),
            "--train-years",
            "0.75",
            "--test-years",
            "0.55",
            "--allow-holdout",
        ],
    )
    assert result.exit_code == 0, result.output

    holdout = HoldoutLock(path=lock_path)
    assert holdout.read_count() == 2, (
        f"expected exactly 2 recorded reads (one per intersecting split, not one per "
        f"split evaluated, and not double-counted), got {holdout.read_count()}"
    )


# ---------------------------------------------------------------------------
# Test hygiene: never touch the real lock
# ---------------------------------------------------------------------------


def test_real_holdout_lock_is_never_the_target_of_any_helper_in_this_file():
    """Guards this file itself against a future edit accidentally pointing a
    `HoldoutLock` at the real `results/holdout_lock.json` instead of `tmp_path`."""
    import nifty_quant.settings as settings_mod

    real_lock = settings_mod.RESULTS_ROOT / "holdout_lock.json"
    assert real_lock.exists(), "sanity: the real lock file this test protects still exists"
    before = real_lock.read_text(encoding="utf-8")
    # No helper in this module ever constructs a HoldoutLock from settings.RESULTS_ROOT
    # without a monkeypatch first -- this test simply confirms the file is untouched by
    # having run every other test in this module before it (pytest runs file order top
    # to bottom by default; this is the last test).
    after = real_lock.read_text(encoding="utf-8")
    assert before == after
