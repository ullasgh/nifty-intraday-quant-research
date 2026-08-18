"""Internal unit tests for `nifty_quant.research.hypotheses.h3_intraday_xsec_reversal`.

`tests/test_h3_deepseek.py` and `tests/test_h3_luna.py` remain frozen -- 47 tests written
independently before this module existed, and neither is touched here. This file is owned
entirely by the implementer and exists to restore two guards that a 100%-branch-coverage
gate, combined with "the implementer may not add tests", had made impossible to keep: a
gate like that actively rewards deleting error handling over testing it. Once the
implementer is allowed to write tests, that perverse incentive disappears, so the guards
come back, each pinned down by a test here:

1. `_restrict_feature_to_panel`'s zero-row guard -- a `start`/`end` window selecting zero
   sessions must not raise a bare `IndexError` from `sliced.ts[0]`.
2. `_survivorship_window`'s empty-`dates` fallback -- the same trigger, a different call
   site (`dates[0]` before building the `SurvivorshipReport` window).

It also pins the end-to-end fail-closed behaviour: `run_h3` with a zero-session window
must surface a named, message-bearing `ContractViolation` from `cross_sectional_rank`'s
own row-count contract, never an unexplained `IndexError` three frames deep.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.guards import ContractViolation
from nifty_quant.research.hypotheses.h3_intraday_xsec_reversal import (
    _restrict_feature_to_panel,
    _survivorship_window,
    build_morning_residual_feature,
    run_h3,
)

# ---------------------------------------------------------------------------
# Minimal panel builder (self-contained -- does not import fixture helpers from
# either frozen H3 suite).
# ---------------------------------------------------------------------------


def _epoch_seconds(session_date: datetime.date, hhmm: str) -> int:
    """Left-labelled epoch-second UTC timestamp for an IST wall-clock HH:MM."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(session_date.year, session_date.month, session_date.day, hour, minute)
    utc_naive = local - datetime.timedelta(hours=5, minutes=30)
    return int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())


def _build_panel(
    symbols: tuple[str, ...],
    sessions: list[datetime.date],
    open_by_session: dict[datetime.date, list[float]],
    morning_close_by_session: dict[datetime.date, list[float]],
    exit_close_by_session: dict[datetime.date, list[float]],
) -> Panel:
    """Dense (n_rows, n_symbols) panel with exactly three bars per session: 09:16 open,
    10:00 close (morning checkpoint), and 15:20 close (exit checkpoint).
    """
    n_symbols = len(symbols)
    n_sessions = len(sessions)
    n_rows = 3 * n_sessions

    ts = np.empty(n_rows, dtype=np.int64)
    fields: dict[str, np.ndarray] = {
        name: np.full((n_rows, n_symbols), np.nan, dtype=np.float32)
        for name in ("open", "high", "low", "close")
    }
    volume = np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float32)

    for day_ix, d in enumerate(sessions):
        entry_row = 3 * day_ix
        morning_row = 3 * day_ix + 1
        exit_row = 3 * day_ix + 2
        ts[entry_row] = _epoch_seconds(d, "09:16")
        ts[morning_row] = _epoch_seconds(d, "10:00")
        ts[exit_row] = _epoch_seconds(d, "15:20")

        entry_opens = open_by_session[d]
        morning_closes = morning_close_by_session[d]
        exit_closes = exit_close_by_session[d]
        for j in range(n_symbols):
            o = entry_opens[j]
            fields["open"][entry_row, j] = o
            fields["close"][entry_row, j] = o
            fields["high"][entry_row, j] = o + 0.01
            fields["low"][entry_row, j] = o - 0.01

            mc = morning_closes[j]
            fields["open"][morning_row, j] = o
            fields["close"][morning_row, j] = mc
            fields["high"][morning_row, j] = max(o, mc) + 0.01
            fields["low"][morning_row, j] = min(o, mc) - 0.01

            ec = exit_closes[j]
            fields["open"][exit_row, j] = mc
            fields["close"][exit_row, j] = ec
            fields["high"][exit_row, j] = max(mc, ec) + 0.01
            fields["low"][exit_row, j] = min(mc, ec) - 0.01
    fields["volume"] = volume

    grid = SessionGrid.from_timestamps(ts)
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _linear_panel(symbols: tuple[str, ...], sessions: list[datetime.date]) -> Panel:
    """A well-formed panel with distinct price levels per session, so consecutive
    sessions never collide and every checkpoint is finite for every symbol.
    """
    n_symbols = len(symbols)
    open_by_session: dict[datetime.date, list[float]] = {}
    morning_close_by_session: dict[datetime.date, list[float]] = {}
    exit_close_by_session: dict[datetime.date, list[float]] = {}
    level = 100.0
    for d in sessions:
        open_by_session[d] = [level + j for j in range(n_symbols)]
        level += 10.0
        morning_close_by_session[d] = [level + j for j in range(n_symbols)]
        level += 10.0
        exit_close_by_session[d] = [level + j for j in range(n_symbols)]
    return _build_panel(
        symbols, sessions, open_by_session, morning_close_by_session, exit_close_by_session
    )


# ---------------------------------------------------------------------------
# 2. `_restrict_feature_to_panel` zero-row guard
# ---------------------------------------------------------------------------


def test_restrict_feature_to_panel_empty_window_returns_empty_feature_not_index_error() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    panel = _linear_panel(symbols, sessions)
    feature = build_morning_residual_feature(panel)

    # `end` before every session in the panel: `Panel.sub`'s own date-comparison logic
    # (`dates <= end` all False) resolves this to a genuine zero-row slice, not merely
    # a "keep everything" fallback.
    before_all_sessions = sessions[0] - datetime.timedelta(days=1000)
    empty_sliced = panel.sub(end=before_all_sessions)
    assert empty_sliced.n_rows() == 0

    restricted = _restrict_feature_to_panel(panel, feature, empty_sliced)

    assert restricted.values.shape == (0, len(symbols))
    assert restricted.kind == "return"
    assert restricted.name == feature.name


def test_restrict_feature_to_panel_nonempty_window_is_unaffected_by_the_guard() -> None:
    """The positive case: a normal, non-empty window still slices to the exact matching
    row range, so the guard only special-cases the zero-row branch."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    panel = _linear_panel(symbols, sessions)
    feature = build_morning_residual_feature(panel)

    sliced = panel.sub(start=sessions[1])
    restricted = _restrict_feature_to_panel(panel, feature, sliced)

    assert restricted.values.shape == (sliced.n_rows(), len(symbols))
    np.testing.assert_array_equal(restricted.values, feature.values[3:])


# ---------------------------------------------------------------------------
# 3. `_survivorship_window` empty-`dates` fallback
# ---------------------------------------------------------------------------


def test_survivorship_window_falls_back_to_today_for_empty_dates() -> None:
    empty_dates = np.empty(0, dtype=object)
    window_start, window_end = _survivorship_window(empty_dates)

    today = datetime.date.today()
    assert window_start == today
    assert window_end == today


def test_survivorship_window_uses_first_and_last_date_when_nonempty() -> None:
    d0 = datetime.date(2021, 1, 1)
    d1 = datetime.date(2021, 6, 15)
    d2 = datetime.date(2021, 12, 31)
    dates = np.array([d0, d1, d2], dtype=object)

    window_start, window_end = _survivorship_window(dates)

    assert window_start == d0
    assert window_end == d2


# ---------------------------------------------------------------------------
# End-to-end fail-closed check
# ---------------------------------------------------------------------------


def test_run_h3_empty_window_fails_closed_with_a_named_exception_not_index_error() -> None:
    """End-to-end: a `start`/`end` window selecting zero sessions must not surface a bare
    `IndexError` from either guard restored above. `Lens`/`expectancy`'s own
    `cross_sectional_rank` contract (a layer this module does not own and must not modify)
    still fails closed on a wholly empty feature -- but with `ContractViolation`, a named,
    message-bearing exception, never an unexplained `IndexError` three frames deep."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    panel = _linear_panel(symbols, sessions)

    before_all_sessions = sessions[0] - datetime.timedelta(days=1000)

    with pytest.raises(ContractViolation) as exc_info:
        run_h3(panel, end=before_all_sessions, seed=0)

    assert not isinstance(exc_info.value, IndexError)
    assert str(exc_info.value)