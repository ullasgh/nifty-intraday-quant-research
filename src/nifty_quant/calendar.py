"""
Bars are left-labelled: a bar timestamped `ts` covers the minute starting at `ts`.
`ts` is int64 epoch seconds UTC with no tz attached; convert via `to_ist()`.
NEVER assume a fixed 375-bar stride per session — irregular sessions exist
(Muhurat, special sessions, halts, corrupt data, minor gaps). Always derive
session boundaries from actual bar timestamps, e.g. via TradingCalendar / SessionGrid.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_RESEARCH_START = datetime.date(2018, 1, 1)


def to_ist(ts: np.ndarray) -> pd.DatetimeIndex:
    """Convert int64 epoch-second UTC array to a tz-aware Asia/Kolkata DatetimeIndex."""
    ts_array = np.asarray(ts, dtype=np.int64)
    return pd.to_datetime(ts_array, unit="s", utc=True).tz_convert(IST)


def ist_label(ts: int) -> str:
    """Return 'HH:MM' IST label for a single epoch-second timestamp."""
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(IST).strftime("%H:%M")


class SessionKind(StrEnum):
    REGULAR = "regular"
    MUHURAT = "muhurat"
    MUHURAT_MERGED = "muhurat_merged"
    SPECIAL_SESSION = "special_session"
    HALTED = "halted"
    CORRUPT = "corrupt"
    MINOR = "minor"


_MUHURAT_DATES: frozenset[datetime.date] = frozenset(
    {
        datetime.date(2018, 11, 7),
        datetime.date(2019, 10, 27),
        datetime.date(2020, 11, 14),
        datetime.date(2022, 10, 24),
        datetime.date(2023, 11, 12),
        datetime.date(2024, 11, 1),
        datetime.date(2025, 10, 21),
    }
)

_MUHURAT_MERGED_DATES: frozenset[datetime.date] = frozenset(
    {
        datetime.date(2019, 10, 25),
        datetime.date(2021, 11, 4),
    }
)

_SPECIAL_SESSION_DATES: frozenset[datetime.date] = frozenset(
    {
        datetime.date(2024, 3, 2),
        datetime.date(2024, 5, 18),
    }
)

_HALTED_DATES: frozenset[datetime.date] = frozenset(
    {
        datetime.date(2020, 3, 13),
        datetime.date(2020, 3, 23),
        datetime.date(2021, 2, 24),
        datetime.date(2018, 2, 8),
        datetime.date(2018, 3, 22),
    }
)

_CORRUPT_DATES: frozenset[datetime.date] = frozenset(
    {
        datetime.date(2017, 10, 18),
        datetime.date(2017, 10, 23),
    }
)


def _classify_session(d: datetime.date, n_bars: int) -> SessionKind:
    if d in _MUHURAT_DATES:
        return SessionKind.MUHURAT
    if d in _MUHURAT_MERGED_DATES:
        return SessionKind.MUHURAT_MERGED
    if d in _SPECIAL_SESSION_DATES:
        return SessionKind.SPECIAL_SESSION
    if d in _HALTED_DATES:
        return SessionKind.HALTED
    if d in _CORRUPT_DATES:
        return SessionKind.CORRUPT
    if n_bars != 375:
        return SessionKind.MINOR
    return SessionKind.REGULAR


@dataclass(frozen=True)
class Session:
    date: datetime.date
    kind: SessionKind
    n_bars: int
    first_label: datetime.time
    last_label: datetime.time

    @property
    def is_research_usable(self) -> bool:
        """False for MUHURAT, MUHURAT_MERGED, CORRUPT, SPECIAL_SESSION. True otherwise
        (REGULAR, HALTED, MINOR)."""
        return self.kind not in {
            SessionKind.MUHURAT,
            SessionKind.MUHURAT_MERGED,
            SessionKind.CORRUPT,
            SessionKind.SPECIAL_SESSION,
        }


class TradingCalendar:
    """Session calendar derived from actual bar timestamps, with a hardcoded
    classification table for known irregular sessions (Muhurat, special sessions,
    halts, corrupt data). See module docstring: never assume a fixed 375-bar stride."""

    ts: np.ndarray

    def __init__(self, ts: np.ndarray) -> None:
        ts_array = np.unique(np.asarray(ts, dtype=np.int64))
        self.ts = ts_array
        self._sessions: dict[datetime.date, Session] = {}

        if ts_array.size == 0:
            return

        ist_index = to_ist(ts_array)
        dates = ist_index.date

        change = np.empty(len(dates), dtype=bool)
        change[0] = True
        change[1:] = dates[1:] != dates[:-1]
        starts = np.flatnonzero(change)

        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(ts_array)
            d = dates[start]
            n_bars = end - start
            first_label = ist_index[start].time()
            last_label = ist_index[end - 1].time()
            kind = _classify_session(d, n_bars)
            self._sessions[d] = Session(
                date=d,
                kind=kind,
                n_bars=n_bars,
                first_label=first_label,
                last_label=last_label,
            )

    @classmethod
    def from_index_bars(cls, symbol: str = "NIFTY50") -> "TradingCalendar":
        """Read all `data/bars/1/{symbol}/*.parquet` files (repo root resolved from
        this file's location), concatenate the `ts` column, and build from that."""
        repo_root = Path(__file__).resolve().parents[2]
        bars_dir = repo_root / "data" / "bars" / "1" / symbol
        parquet_files = sorted(bars_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {bars_dir}")

        frames = [pd.read_parquet(path, columns=["ts"]) for path in parquet_files]
        ts = pd.concat(frames, ignore_index=True)["ts"].to_numpy(dtype=np.int64)
        return cls(ts)

    @classmethod
    def from_timestamps(cls, ts: np.ndarray) -> "TradingCalendar":
        """Build directly from a raw int64 epoch-second timestamp array."""
        return cls(ts)

    def sessions(
        self,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
        *,
        usable_only: bool = True,
    ) -> list[Session]:
        """Sessions with date in [start, end] inclusive (None = unbounded on that side),
        sorted by date. If usable_only, keep only sessions where is_research_usable."""
        result: list[Session] = []
        for date in sorted(self._sessions):
            if start is not None and date < start:
                continue
            if end is not None and date > end:
                continue
            session = self._sessions[date]
            if usable_only and not session.is_research_usable:
                continue
            result.append(session)
        return result

    def session_dates(
        self,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
        *,
        usable_only: bool = True,
    ) -> list[datetime.date]:
        """Same filtering as sessions(), returning just the dates."""
        return [
            session.date
            for session in self.sessions(start, end, usable_only=usable_only)
        ]

    def classify(self, d: datetime.date) -> SessionKind:
        """Raise KeyError if d is not a known session date."""
        if d not in self._sessions:
            raise KeyError(f"Unknown session date: {d}")
        return self._sessions[d].kind

    def is_usable(self, d: datetime.date) -> bool:
        """Convenience: Session(...).is_research_usable for date d. Raise KeyError if
        d is not a known session date."""
        if d not in self._sessions:
            raise KeyError(f"Unknown session date: {d}")
        return self._sessions[d].is_research_usable


@dataclass(frozen=True)
class SessionGrid:
    """Canonical row index for a panel built from raw bar timestamps. NEVER assume a
    fixed 375-bar stride between days — derive everything from actual timestamps."""

    ts: np.ndarray            # int64 epoch sec, sorted unique, all sessions in range
    day_offsets: np.ndarray   # int32, len = n_days+1; rows for day i are
                               # [day_offsets[i], day_offsets[i+1])
    dates: np.ndarray         # object array of datetime.date, len n_days

    @classmethod
    def from_timestamps(cls, ts: np.ndarray) -> "SessionGrid":
        """Build a grid from a raw (not necessarily sorted/deduplicated) int64
        epoch-second timestamp array, grouping rows into sessions by IST calendar date."""
        ts_sorted = np.unique(np.asarray(ts, dtype=np.int64))

        if ts_sorted.size == 0:
            return cls(
                ts=ts_sorted,
                day_offsets=np.array([0], dtype=np.int32),
                dates=np.empty(0, dtype=object),
            )

        ist_index = to_ist(ts_sorted)
        dates_raw = ist_index.date

        change = np.empty(len(dates_raw), dtype=bool)
        change[0] = True
        change[1:] = dates_raw[1:] != dates_raw[:-1]
        starts = np.flatnonzero(change)

        day_offsets = np.empty(len(starts) + 1, dtype=np.int32)
        day_offsets[:-1] = starts
        day_offsets[-1] = len(ts_sorted)

        dates_arr = np.array(dates_raw[starts], dtype=object)

        return cls(ts=ts_sorted, day_offsets=day_offsets, dates=dates_arr)

    def n_days(self) -> int:
        return len(self.dates)

    def day_slice(self, d: datetime.date) -> slice:
        """Raise KeyError if d is not in self.dates."""
        matches = np.flatnonzero(self.dates == d)
        if len(matches) == 0:
            raise KeyError(f"Date not found in grid: {d}")

        idx = int(matches[0])
        return slice(int(self.day_offsets[idx]), int(self.day_offsets[idx + 1]))

    def rows_at_time(self, hhmm: str) -> np.ndarray:
        """int64 row-index array: for every session (day) in this grid, if that day HAS
        a bar whose IST label equals hhmm (format 'HH:MM'), include that bar's absolute
        row index in self.ts; sessions lacking that exact label are simply omitted (no
        placeholder, no error). MUST be computed by vectorized comparison of each row's
        actual IST minute-of-day against the target minute-of-day — MUST NOT be
        implemented as day_start_offset + fixed_row_offset, since day lengths vary
        (e.g. a 60-bar Muhurat day has no 09:15 bar at all)."""
        try:
            parsed_time = datetime.datetime.strptime(hhmm, "%H:%M").time()
        except ValueError as exc:
            raise ValueError(f"Expected HH:MM, got {hhmm!r}") from exc

        target_minute = parsed_time.hour * 60 + parsed_time.minute
        return np.flatnonzero(self.minute_of_day() == target_minute)

    def minute_of_day(self) -> np.ndarray:
        """int16 array, len(self.ts): IST minutes since midnight for each row."""
        ist_index = to_ist(self.ts)
        minutes = ist_index.hour * 60 + ist_index.minute
        return np.asarray(minutes, dtype=np.int16)
