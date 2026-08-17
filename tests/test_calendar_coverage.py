from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from nifty_quant.calendar import SessionGrid, TradingCalendar


def _session_ts(d: datetime.date, start_hhmm: str, n_bars: int) -> np.ndarray:
    """Generate n_bars consecutive 1-minute-spaced left-labelled epoch-second timestamps
    (int64) starting at IST wall-clock time start_hhmm on IST calendar date d."""
    hour, minute = (int(p) for p in start_hhmm.split(":"))
    start = datetime.datetime(
        d.year,
        d.month,
        d.day,
        hour,
        minute,
        tzinfo=ZoneInfo("Asia/Kolkata"),
    )
    return np.array(
        [int((start + datetime.timedelta(minutes=i)).timestamp()) for i in range(n_bars)],
        dtype=np.int64,
    )


@pytest.fixture
def three_session_ts() -> np.ndarray:
    ts_a = _session_ts(datetime.date(2018, 1, 2), "09:15", 375)
    ts_b = _session_ts(datetime.date(2018, 11, 7), "09:15", 60)
    ts_c = _session_ts(datetime.date(2018, 11, 8), "09:15", 375)
    return np.concatenate([ts_a, ts_b, ts_c])


def test_trading_calendar_empty_timestamps_no_sessions() -> None:
    calendar = TradingCalendar.from_timestamps(np.array([], dtype=np.int64))
    assert calendar.ts.size == 0
    assert calendar.session_dates(usable_only=False) == []


def test_trading_calendar_from_index_bars_missing_symbol_raises() -> None:
    with pytest.raises(FileNotFoundError, match="__NO_SUCH_SYMBOL_XYZ__"):
        TradingCalendar.from_index_bars(symbol="__NO_SUCH_SYMBOL_XYZ__")


def test_sessions_end_filter_excludes_later_session(
    three_session_ts: np.ndarray,
) -> None:
    calendar = TradingCalendar.from_timestamps(three_session_ts)
    dates = [
        session.date
        for session in calendar.sessions(
            end=datetime.date(2018, 11, 7),
            usable_only=False,
        )
    ]
    assert dates == [
        datetime.date(2018, 1, 2),
        datetime.date(2018, 11, 7),
    ]


def test_sessions_default_usable_only_excludes_muhurat(
    three_session_ts: np.ndarray,
) -> None:
    calendar = TradingCalendar.from_timestamps(three_session_ts)
    dates = [session.date for session in calendar.sessions()]
    assert dates == [
        datetime.date(2018, 1, 2),
        datetime.date(2018, 11, 8),
    ]


def test_classify_unknown_date_raises_keyerror(
    three_session_ts: np.ndarray,
) -> None:
    calendar = TradingCalendar.from_timestamps(three_session_ts)
    with pytest.raises(KeyError, match="2099-01-01"):
        calendar.classify(datetime.date(2099, 1, 1))


def test_is_usable_unknown_date_raises_keyerror(
    three_session_ts: np.ndarray,
) -> None:
    calendar = TradingCalendar.from_timestamps(three_session_ts)
    with pytest.raises(KeyError, match="2099-01-01"):
        calendar.is_usable(datetime.date(2099, 1, 1))


def test_session_grid_from_timestamps_empty() -> None:
    grid = SessionGrid.from_timestamps(np.array([], dtype=np.int64))
    assert grid.ts.size == 0
    assert grid.n_days() == 0
    assert np.array_equal(grid.day_offsets, np.array([0], dtype=np.int32))


def test_session_grid_day_slice_found_and_not_found(
    three_session_ts: np.ndarray,
) -> None:
    grid = SessionGrid.from_timestamps(three_session_ts)
    assert grid.day_slice(datetime.date(2018, 1, 2)) == slice(0, 375)
    assert grid.day_slice(datetime.date(2018, 11, 7)) == slice(375, 435)
    assert grid.day_slice(datetime.date(2018, 11, 8)) == slice(435, 810)
    with pytest.raises(KeyError, match="2099-01-01"):
        grid.day_slice(datetime.date(2099, 1, 1))


def test_rows_at_time_malformed_hhmm_raises_valueerror() -> None:
    ts = _session_ts(datetime.date(2018, 1, 2), "09:15", 3)
    grid = SessionGrid.from_timestamps(ts)
    with pytest.raises(ValueError, match="Expected HH:MM, got"):
        grid.rows_at_time("not-a-time")
