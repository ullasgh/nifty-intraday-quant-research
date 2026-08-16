import datetime as dt

import numpy as np
import pytest

from nifty_quant.calendar import (
    SessionGrid,
    SessionKind,
    TradingCalendar,
    ist_label,
    to_ist,
)


@pytest.fixture(scope="module")
def calendar() -> TradingCalendar:
    return TradingCalendar.from_index_bars("NIFTY50")


def test_ist_label_and_date():
    assert ist_label(1704080700) == "09:15"
    assert to_ist(np.array([1704080700]))[0].date() == dt.date(2024, 1, 1)


def test_total_session_count(calendar: TradingCalendar):
    assert len(calendar.session_dates(usable_only=False)) == 2250


@pytest.mark.parametrize(
    "d",
    [
        dt.date(2018, 11, 7),
        dt.date(2019, 10, 27),
        dt.date(2020, 11, 14),
        dt.date(2022, 10, 24),
        dt.date(2023, 11, 12),
        dt.date(2024, 11, 1),
        dt.date(2025, 10, 21),
    ],
)
def test_classify_muhurat(calendar: TradingCalendar, d: dt.date):
    assert calendar.classify(d) is SessionKind.MUHURAT


def test_classify_other_kinds(calendar: TradingCalendar):
    assert calendar.classify(dt.date(2024, 3, 2)) is SessionKind.SPECIAL_SESSION
    assert calendar.classify(dt.date(2020, 3, 13)) is SessionKind.HALTED
    assert calendar.classify(dt.date(2017, 10, 18)) is SessionKind.CORRUPT


def test_is_usable(calendar: TradingCalendar):
    assert calendar.is_usable(dt.date(2024, 11, 1)) is False
    assert calendar.is_usable(dt.date(2024, 1, 1)) is True


def test_irregular_session_counts(calendar: TradingCalendar):
    all_sessions = calendar.sessions(usable_only=False)
    n_irregular_bars = sum(1 for s in all_sessions if s.n_bars != 375)
    n_not_regular = sum(1 for s in all_sessions if s.kind is not SessionKind.REGULAR)
    assert n_irregular_bars == 105
    assert n_not_regular == 105


def test_2018_onward_counts(calendar: TradingCalendar):
    dates = calendar.session_dates(dt.date(2018, 1, 1), dt.date(2026, 8, 14), usable_only=False)
    assert len(dates) == 2136

    sessions = calendar.sessions(dt.date(2018, 1, 1), dt.date(2026, 8, 14), usable_only=False)
    n_not_regular = sum(1 for s in sessions if s.kind is not SessionKind.REGULAR)
    assert n_not_regular == 32


@pytest.fixture(scope="module")
def muhurat_span_grid(calendar: TradingCalendar) -> SessionGrid:
    start, end = dt.date(2024, 10, 28), dt.date(2024, 11, 5)
    dates = to_ist(calendar.ts).date
    mask = (dates >= start) & (dates <= end)
    return SessionGrid.from_timestamps(calendar.ts[mask])


def test_session_grid_muhurat_span(muhurat_span_grid: SessionGrid):
    grid = muhurat_span_grid
    diffs = np.diff(grid.day_offsets)

    assert np.all(diffs > 0)
    assert grid.day_offsets[-1] == len(grid.ts)
    assert 60 in diffs

    rows = grid.rows_at_time("09:15")
    assert len(rows) > 0
    for row in rows:
        assert ist_label(int(grid.ts[row])) == "09:15"

    row_dates = to_ist(grid.ts[rows]).date
    assert dt.date(2024, 11, 1) not in row_dates


def test_session_grid_day_offsets_match_n_bars(calendar: TradingCalendar):
    grid = SessionGrid.from_timestamps(calendar.ts)
    diffs = np.diff(grid.day_offsets)
    sessions_by_date = {s.date: s for s in calendar.sessions(usable_only=False)}

    assert len(diffs) == len(grid.dates)
    for date, n_bars_in_grid in zip(grid.dates, diffs):
        assert n_bars_in_grid == sessions_by_date[date].n_bars
