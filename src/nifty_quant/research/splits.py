"""Data splitting utilities for walk-forward and holdout research."""

import calendar
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Split:
    id: str
    train: tuple[date, date]
    test: tuple[date, date]
    embargo_days: int


class EmbargoTooShortError(ValueError):
    """Raised when the embargo is shorter than the maximum lookback."""


def _subtract_months(value: date, months: int) -> date:
    """Subtract ``months`` calendar months from ``value``, clamping the day."""
    total_month = value.year * 12 + (value.month - 1)
    target_total = total_month - months
    target_year = target_total // 12
    target_month_0 = target_total % 12
    target_month = target_month_0 + 1
    target_day = min(value.day, calendar.monthrange(target_year, target_month)[1])
    return date(target_year, target_month, target_day)


@contextmanager
def _locked_path(path: Path) -> Iterator[None]:
    """Best-effort advisory lock for atomic JSON read-modify-write cycles."""
    if fcntl is None:
        yield
        return

    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass(frozen=True)
class WalkForwardSplitter:
    scheme: Literal["rolling", "anchored"] = "rolling"
    train_years: float = 3.0
    test_years: float = 1.0
    step_years: float = 1.0
    embargo_days: int = 5  # in TRADING days

    def split(
        self, trading_dates: list[date], *, max_lookback_days: int = 0
    ) -> list[Split]:
        if max_lookback_days > 0 and self.embargo_days < max_lookback_days:
            raise EmbargoTooShortError(
                f"embargo_days={self.embargo_days} is shorter than "
                f"max_lookback_days={max_lookback_days}; increase embargo to avoid "
                "train/test data leakage"
            )

        if self.train_years <= 0 or self.test_years <= 0 or self.step_years <= 0:
            raise ValueError("train_years, test_years, and step_years must be > 0")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")

        n = len(trading_dates)
        if n == 0:
            return []

        train_days = round(self.train_years * TRADING_DAYS_PER_YEAR)
        test_days = round(self.test_years * TRADING_DAYS_PER_YEAR)
        step_days = round(self.step_years * TRADING_DAYS_PER_YEAR)

        if train_days <= 0 or test_days <= 0 or step_days <= 0:
            raise ValueError("window lengths must be at least one trading day")

        splits: list[Split] = []

        if self.scheme == "rolling":
            cursor = 0
            split_idx = 0
            while True:
                train_start_idx = cursor
                train_end_idx = cursor + train_days - 1
                if train_end_idx >= n:
                    break

                test_start_idx = train_end_idx + 1 + self.embargo_days
                test_end_idx = test_start_idx + test_days - 1
                if test_end_idx >= n:
                    break

                splits.append(
                    Split(
                        id=f"{self.scheme}_{split_idx:03d}",
                        train=(
                            trading_dates[train_start_idx],
                            trading_dates[train_end_idx],
                        ),
                        test=(
                            trading_dates[test_start_idx],
                            trading_dates[test_end_idx],
                        ),
                        embargo_days=self.embargo_days,
                    )
                )
                cursor += step_days
                split_idx += 1
        else:  # anchored
            split_idx = 0
            while True:
                train_end_idx = train_days - 1 + split_idx * step_days
                if train_end_idx >= n:
                    break

                test_start_idx = train_end_idx + 1 + self.embargo_days
                test_end_idx = test_start_idx + test_days - 1
                if test_end_idx >= n:
                    break

                splits.append(
                    Split(
                        id=f"{self.scheme}_{split_idx:03d}",
                        train=(trading_dates[0], trading_dates[train_end_idx]),
                        test=(
                            trading_dates[test_start_idx],
                            trading_dates[test_end_idx],
                        ),
                        embargo_days=self.embargo_days,
                    )
                )
                split_idx += 1

        return splits


@dataclass(frozen=True)
class HoldoutLock:
    path: Path
    holdout_months: int = 12

    def holdout_range(self, trading_dates: list[date]) -> tuple[date, date]:
        if not trading_dates:
            raise ValueError("trading_dates must not be empty")

        holdout_end = trading_dates[-1]
        cutoff = _subtract_months(holdout_end, self.holdout_months)
        holdout_start = next(
            (day for day in trading_dates if day >= cutoff),
            holdout_end,
        )
        return holdout_start, holdout_end

    def read_count(self) -> int:
        if not self.path.exists():
            return 0

        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            return 0

        return int(data.get("count", 0))

    def _atomic_write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def record_read(self, reason: str) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _locked_path(self.path):
            state: dict[str, Any] = {"count": 0, "log": []}
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    state = loaded

            count = int(state.get("count", 0))
            log = list(state.get("log", []))
            log.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reason": reason,
                }
            )

            new_state = {**state, "count": count + 1, "log": log}
            self._atomic_write(new_state)

        return count + 1
