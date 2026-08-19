"""Tests for nifty_quant.research.tilt (TDD red-for-the-right-reason).

The module under test does not exist yet; this file imports it at module level, so the whole
file fails to collect with ImportError until the module is created. That is an acceptable
red-for-the-right-reason failure mode per the task instructions.

ASSUMPTION (documented, not a certainty): the author assumes run_tilt's holdout check is
computed from the SAME global, real, panel-independent calendar (not from whatever synthetic
Panel is passed in), because a request-window-relative interpretation would make the tail of
EVERY legitimate query look like "the holdout", which cannot be the intended behaviour of a
tool whose entire purpose is "let me test for the days I want". This is a documented
assumption, not a certainty -- flag it exactly as worded above in the comment.

Reference-implementation validation already performed by the author (reported verbatim, not
re-derived here): The author validated `_build_panel`/`_cycling_aggressive_panel` against a
throwaway reference `run_tilt` (lifted from `scripts/recon_low_turnover_tilt.py`, NOT src/)
via `sys.modules` substitution, and confirms: `build_overnight_feature` correctly reads
open[entry]/close[exit, prev] across these sparse two-bar sessions, `_build_checkpoint_panel`-
style logic keeps sessions whose OWN entry/exit bars exist, and the aggressive-tilt
bottom-quintile rule with `n_valid==5` always selects exactly the planted loser. This
validation is NOT part of this file; you are only told the fixture mechanics are sound.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.calendar import TradingCalendar
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.splits import HoldoutLock
from nifty_quant.research.tilt import TiltConfig, TiltResult, TiltYearRow, run_tilt
from typer.testing import CliRunner

_IST = ZoneInfo("Asia/Kolkata")
_N_SYMBOLS = 5
_SYMBOLS = tuple(f"SYM{i}" for i in range(_N_SYMBOLS))

runner = CliRunner()


def _ts_for(date: datetime.date, hhmm: str) -> int:
    """Epoch-second timestamp for `date` at exact IST label `hhmm` ('HH:MM')."""
    hour, minute = (int(p) for p in hhmm.split(":"))
    stamp = pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=_IST)
    return int(stamp.tz_convert("UTC").timestamp())


def _build_panel(
    sessions: list[tuple[datetime.date, dict[str, np.ndarray]]],
    symbols: tuple[str, ...] = _SYMBOLS,
) -> Panel:
    """Build a sparse-bar Panel: `sessions` is a list of (date, {"HH:MM": price_row}) -- one
    entry per calendar date (need not be pre-sorted), each mapping a small number of exact IST
    time labels to a length-len(symbols) price array. Both the "open" and "close" fields are set
    to the SAME price at each labelled bar (this fixture family never needs open != close within
    one labelled bar -- only the price level at each named checkpoint matters for the overnight
    feature and for entry/exit pricing). Only the bars actually named exist in the panel for that
    date: Panel/SessionGrid group rows by IST calendar date and resolve `rows_at_time` by exact
    minute-of-day label, never by position or a fixed per-day bar count (CLAUDE.md rule 5), so no
    intermediate minutes are required. `volume` is a filler constant (1_000_000.0 everywhere)
    purely because the checkpoint-reduction helper this module reuses from h2_overnight_reversal
    reads `panel.field("volume")`; tilt's own weighting/turnover/cost logic never consults it.
    """
    sessions_sorted = sorted(sessions, key=lambda item: item[0])
    ts_list: list[int] = []
    price_rows: list[np.ndarray] = []
    day_offsets = [0]
    row = 0
    for date, bars in sessions_sorted:
        for hhmm in sorted(bars.keys()):
            ts_list.append(_ts_for(date, hhmm))
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


def _flat_row(base: float = 100.0, n: int = _N_SYMBOLS) -> np.ndarray:
    """A length-n price row, symbol i = base + i*0.01 (distinct, no ties, all near `base`)."""
    return np.array([base + i * 0.01 for i in range(n)], dtype=np.float64)


def _cycling_aggressive_panel(
    dates: list[datetime.date],
    loser_of_day: list[int],
    entry_hhmm: str = "09:16",
    exit_hhmm: str = "15:20",
) -> Panel:
    """One session per date in `dates` (must be sorted ascending), exactly two bars per session
    at `entry_hhmm` and `exit_hhmm` (defaults match BOTH TiltConfig's defaults AND h2's own
    internal 09:16/15:20 checkpoints, so the same two bars serve both roles). For date `dates[i]`,
    symbol `loser_of_day[i]` gets an entry price ~3.0 below the flat ~100 baseline (all other
    symbols are within +/-0.04 of 100), making it unambiguously the single biggest overnight
    loser that session (overnight feature = log(open[entry]/close[exit, prev session]); the
    baseline close every session is also ~100 +/- 0.05, so the loser's log-return is close to
    log(97/100.05) =~ -0.031, versus the non-losers' log-returns which stay within roughly
    +/-0.001 of each other -- no ties, always exactly 5 valid names). With `tilt="aggressive"`,
    n_bottom = max(1, round(5 * 0.2)) = 1, so exactly ONE symbol is held (weight 1.0) every
    session: whichever `loser_of_day[i]` names. Exit price is a small flat markup over entry so
    every session has a small nonzero, finite session return.
    """
    assert len(dates) == len(loser_of_day)
    sessions = []
    for date, loser in zip(dates, loser_of_day):
        entry_row = _flat_row(100.0)
        entry_row[loser] -= 3.0
        exit_row = entry_row + 0.05
        sessions.append((date, {entry_hhmm: entry_row, exit_hhmm: exit_row}))
    return _build_panel(sessions)


def test_capital_changes_cost_but_not_gross() -> None:
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % 2 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_1l = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=100_000.0,
        seed=0,
    )
    config_10l = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result_1l = run_tilt(panel, config_1l)
    result_10l = run_tilt(panel, config_10l)

    assert result_1l.total.gross_bps == pytest.approx(result_10l.total.gross_bps, abs=1e-9)
    assert result_1l.round_trip_bps > result_10l.round_trip_bps
    assert result_1l.total.cost_bps > result_10l.total.cost_bps
    assert result_1l.total.net_bps < result_10l.total.net_bps

    assert result_1l.clip_per_name == pytest.approx(100_000.0, rel=1e-6)
    assert result_10l.clip_per_name == pytest.approx(1_000_000.0, rel=1e-6)

    assert result_1l.round_trip_bps == pytest.approx(8.26452, abs=1e-3)
    assert result_10l.round_trip_bps == pytest.approx(4.01652, abs=1e-3)


def test_one_leg_cost_identity() -> None:
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % 2 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=250_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    # A 2x (long/short-spread) cost convention fails the first assertion directly; it is the
    # single most important test in this file per the spec.
    for row in (*result.per_year, result.total):
        assert row.cost_bps == pytest.approx(result.round_trip_bps * row.turnover, rel=1e-9, abs=1e-12)

    assert result.round_trip_bps == pytest.approx(
        NSEIntradayEquityCosts().round_trip_bps(result.clip_per_name), rel=1e-6
    )


def test_custom_checkpoints_honoured_and_decoy_not_read() -> None:
    dates = [datetime.date(2022, 3, 1), datetime.date(2022, 3, 2), datetime.date(2022, 3, 3)]

    def _make_variant(v1359: float, v1400: float) -> Panel:
        sessions = []
        for date in dates:
            bars = {}
            for hhmm in ["09:16", "10:00", "13:59", "14:00", "15:20"]:
                row = _flat_row(100.0)
                if hhmm == "09:16":
                    row[0] = 95.0
                elif hhmm == "10:00":
                    row[0] = 90.0
                elif hhmm == "13:59":
                    row[0] = v1359
                elif hhmm == "14:00":
                    row[0] = v1400
                elif hhmm == "15:20":
                    row[0] = 100.0
                bars[hhmm] = row
            sessions.append((date, bars))
        return _build_panel(sessions)

    panel_a = _make_variant(50.0, 100.0)
    panel_b = _make_variant(999.0, 100.0)
    panel_c = _make_variant(50.0, 120.0)

    config_custom = TiltConfig(
        start=dates[0],
        end=dates[-1],
        entry_hhmm="10:00",
        exit_hhmm="14:00",
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    config_default = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    result_a = run_tilt(panel_a, config_custom)
    result_b = run_tilt(panel_b, config_custom)
    result_c = run_tilt(panel_c, config_custom)
    result_default = run_tilt(panel_a, config_default)

    assert result_a.total.gross_bps == pytest.approx(result_b.total.gross_bps, abs=1e-6)
    assert abs(result_a.total.gross_bps - result_c.total.gross_bps) > 1.0
    assert abs(result_default.total.gross_bps - result_a.total.gross_bps) > 1.0


def test_missing_checkpoint_session_dropped_and_warned() -> None:
    d0 = datetime.date(2022, 5, 2)
    d1 = datetime.date(2022, 5, 3)
    d2 = datetime.date(2022, 5, 4)
    d3 = datetime.date(2022, 5, 5)
    d4 = datetime.date(2022, 5, 6)

    sessions = [
        (d0, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d1, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d2, {"09:16": _flat_row(100.0), "10:15": _flat_row(100.0) + 0.02}),
        (d3, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
        (d4, {"09:16": _flat_row(100.0), "15:20": _flat_row(100.0) + 0.05}),
    ]
    panel = _build_panel(sessions)

    config = TiltConfig(
        start=d0,
        end=d4,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    assert result.total.n_sessions == 2
    assert len(result.warnings) >= 1
    assert str(d2) in " ".join(result.warnings)


def test_date_range_respected_and_years_partition_exactly() -> None:
    dates = [datetime.date(2021, 12, 30)]
    dates += [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(10)]
    dates += [datetime.date(2023, 1, 3) + datetime.timedelta(days=i) for i in range(8)]
    dates += [datetime.date(2024, 1, 5)]

    loser_of_day = [i % 5 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=datetime.date(2022, 1, 3),
        end=datetime.date(2023, 1, 10),
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    assert sum(row.n_sessions for row in result.per_year) == result.total.n_sessions
    assert {row.year for row in result.per_year} == {2022, 2023}
    assert result.total.n_sessions == 18


def test_smoothing_reduces_turnover_and_a_equals_one_is_daily_rebalance() -> None:
    # A buffer day before the requested window is required so the window's OWN first session
    # has a valid prior close (its overnight feature would otherwise be NaN -- no d-1 in the
    # panel at all -- and get dropped, silently shrinking 6 requested sessions to 5 and
    # invalidating the exact turnover arithmetic below). See `test_date_range_respected...`
    # for the same buffer-day pattern.
    buffer_day = datetime.date(2022, 5, 31)
    dates = [buffer_day] + [
        datetime.date(2022, 6, 1) + datetime.timedelta(days=i) for i in range(6)
    ]
    loser_of_day = [0, 0, 1, 0, 1, 0, 1]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_full = TiltConfig(
        start=dates[1],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    config_smoothed = TiltConfig(
        start=dates[1],
        end=dates[-1],
        tilt="aggressive",
        smoothing=0.10,
        capital=1_000_000.0,
        seed=0,
    )

    result_full = run_tilt(panel, config_full)
    result_smoothed = run_tilt(panel, config_smoothed)

    assert result_full.total.n_sessions == 6
    assert result_full.total.turnover == pytest.approx((1.0 + 2.0 * 5) / 6.0, rel=1e-6)
    assert result_smoothed.total.turnover < 0.5 * result_full.total.turnover


def test_turnover_never_exceeds_long_only_normalised_bound() -> None:
    # The public `TiltResult`/`TiltYearRow` contract given in the spec exposes only AGGREGATED
    # per-year/total statistics, never raw per-session weight vectors, so "weights are never
    # negative" and "weights sum to 1" cannot be asserted directly against the documented public
    # API -- this is a genuine spec/API gap, reported separately, not silently resolved. This
    # test asserts the one MATHEMATICALLY NECESSARY (though not sufficient) consequence of the
    # long-only+normalised-weight invariant that IS visible through the public API: the L1
    # distance between any two long-only weight vectors that each sum to 1 can never exceed 2.0,
    # so mean daily turnover can never exceed 2.0 either.
    buffer_day = datetime.date(2022, 5, 31)
    dates = [buffer_day] + [
        datetime.date(2022, 6, 1) + datetime.timedelta(days=i) for i in range(6)
    ]
    loser_of_day = [0, 0, 1, 0, 1, 0, 1]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_full = TiltConfig(
        start=dates[1],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config_full)

    for row in (*result.per_year, result.total):
        assert row.turnover <= 2.0 + 1e-9


def test_holdout_window_is_protected() -> None:
    real_dates = TradingCalendar.from_index_bars("NIFTY50").session_dates()
    holdout_start, holdout_end = HoldoutLock(
        path=Path("/tmp/test_tilt_a_unused_holdout_lock.json")
    ).holdout_range(real_dates)

    dates = [
        holdout_start - datetime.timedelta(days=400),
        holdout_start - datetime.timedelta(days=399),
        holdout_start,
    ]
    loser_of_day = [0, 1, 2]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=dates[1],
        end=holdout_start,
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )

    # The exact wording/date-format of "the boundary in the message" is not specified by the
    # spec beyond "names the boundary", so only the word "holdout" is asserted here to avoid
    # over-fitting a test to an unspecified message format; this is reported as an ambiguity
    # separately, not resolved by guessing a format.
    with pytest.raises(ValueError, match=r"(?i)holdout"):
        run_tilt(panel, config)


def test_degenerate_inputs_raise() -> None:
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % 2 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config_start_gt_end = TiltConfig(
        start=datetime.date(2022, 1, 10),
        end=datetime.date(2022, 1, 1),
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    with pytest.raises(ValueError):
        run_tilt(panel, config_start_gt_end)

    config_no_overlap = TiltConfig(
        start=datetime.date(1999, 1, 1),
        end=datetime.date(1999, 1, 2),
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    with pytest.raises(ValueError):
        run_tilt(panel, config_no_overlap)

    panel_3sym = _build_panel(
        [
            (
                d,
                {
                    "09:16": np.array([100.0, 100.1, 100.2]),
                    "15:20": np.array([100.05, 100.15, 100.25]),
                },
            )
            for d in (datetime.date(2022, 7, 1), datetime.date(2022, 7, 2))
        ],
        symbols=("X0", "X1", "X2"),
    )
    config_3sym = TiltConfig(
        start=datetime.date(2022, 7, 1),
        end=datetime.date(2022, 7, 2),
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    with pytest.raises(ValueError):
        run_tilt(panel_3sym, config_3sym)


def test_determinism_same_config_and_seed_gives_identical_table() -> None:
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % 2 for i in range(20)]

    panel_a = _cycling_aggressive_panel(dates, loser_of_day)
    config_a = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result_a = run_tilt(panel_a, config_a)

    panel_b = _cycling_aggressive_panel(dates, loser_of_day)
    config_b = TiltConfig(
        start=dates[0],
        end=dates[-1],
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result_b = run_tilt(panel_b, config_b)

    assert result_a.to_table() == result_b.to_table()


def test_to_table_contains_per_year_rows_and_all_row_with_matching_count() -> None:
    dates = [datetime.date(2021, 12, 30)]
    dates += [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(10)]
    dates += [datetime.date(2023, 1, 3) + datetime.timedelta(days=i) for i in range(8)]
    dates += [datetime.date(2024, 1, 5)]

    loser_of_day = [i % 5 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    config = TiltConfig(
        start=datetime.date(2022, 1, 3),
        end=datetime.date(2023, 1, 10),
        tilt="aggressive",
        smoothing=1.0,
        capital=1_000_000.0,
        seed=0,
    )
    result = run_tilt(panel, config)

    assert result.total.n_sessions == sum(row.n_sessions for row in result.per_year)

    table = result.to_table()
    assert "ALL" in table
    for row in result.per_year:
        assert str(row.year) in table


def test_cli_tilt_smoke_and_bad_tilt_value(monkeypatch, tmp_path) -> None:
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod
    from nifty_quant.universe.static import Universe

    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % 2 for i in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="all_equity", symbols=_SYMBOLS)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    result = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
            "--entry",
            "09:16",
            "--exit",
            "15:20",
            "--capital",
            "1000000",
            "--tilt",
            "mild",
            "--smoothing",
            "0.10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ALL" in result.output

    # Today, before `nq tilt` exists at all, BOTH invocations fail with Typer's own "No such
    # command" error (exit_code=2), which trivially satisfies `exit_code != 0` for the second
    # assertion but NOT the `"mild"/"aggressive"` string checks or the `exit_code == 0` /
    # `"ALL" in result.output` checks on the first invocation -- so this test is red for the
    # right reason (AssertionError, not ImportError) even though the CLI module itself already
    # exists.
    bad = runner.invoke(
        app,
        [
            "tilt",
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
            "--tilt",
            "not_a_real_tilt",
        ],
    )
    assert bad.exit_code != 0
    assert "mild" in bad.output
    assert "aggressive" in bad.output
