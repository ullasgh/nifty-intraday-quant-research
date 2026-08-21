"""Data splitting utilities for walk-forward and holdout research."""

import calendar
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal

fcntl: ModuleType | None
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
# NOTE: `nifty_quant/concurrency.py::locked_path` duplicates this fcntl.flock pattern ON
# PURPOSE -- see the long comment there. `tests/test_splits_coverage.py` pins this module's
# `fcntl` attribute and this function as live module attributes, so this copy cannot be
# replaced by a re-export. Change one, change the other.
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
        self,
        trading_dates: list[date],
        *,
        max_lookback_days: int = 0,
        feature_lookback: float = 0.0,
        label_horizon: float = 0.0,
        holding_period: float = 0.0,
        execution_horizon: float = 0.0,
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

        from nifty_quant.research.embargo import EmbargoComponents

        components = EmbargoComponents(
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        required = math.ceil(components.required_sessions())
        if self.embargo_days < required:
            dominant = components.dominant_term()
            raise EmbargoTooShortError(
                f"embargo_days={self.embargo_days} is shorter than the required "
                f"embargo of {required} sessions (dominant term: "
                f"{dominant}={getattr(components, dominant)}); increase embargo to "
                "avoid train/test data leakage"
            )

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


class HoldoutBoundaryError(ValueError):
    """Raised when a stored holdout boundary disagrees with a fresh recomputation."""


@dataclass(frozen=True)
class HoldoutLock:
    path: Path
    holdout_months: int = 12

    def holdout_range(self, trading_dates: list[date]) -> tuple[date, date]:
        if not trading_dates:
            raise ValueError("trading_dates must not be empty")

        candidate_end = trading_dates[-1]
        cutoff = _subtract_months(candidate_end, self.holdout_months)
        candidate_start = next(
            (day for day in trading_dates if day >= cutoff),
            candidate_end,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _locked_path(self.path):
            state: dict[str, Any] = {"count": 0, "log": []}
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    state = loaded

            if "holdout_start" not in state or "holdout_end" not in state:
                # First-ever initialisation, OR a migration of a legacy
                # count/log-only file: persist the candidate, leaving count/log
                # (already inside `state`) untouched.
                self._atomic_write(
                    {
                        **state,
                        "holdout_start": candidate_start.isoformat(),
                        "holdout_end": candidate_end.isoformat(),
                    }
                )
                return candidate_start, candidate_end

            stored_start = date.fromisoformat(state["holdout_start"])
            stored_end = date.fromisoformat(state["holdout_end"])

            if candidate_end <= stored_end:
                # A prefix/window query: this caller's calendar does not extend
                # past what is already known, so it asserts nothing new about the
                # calendar. Trust the stored boundary; do not recompute/compare.
                return stored_start, stored_end

            # The calendar has gained sessions past the previously known point, so the
            # boundary WOULD move. That is never acceptable silently.
            #
            # There is deliberately no equality check here. The `candidate_end <= stored_end`
            # guard above already returned, so `candidate_end > stored_end` strictly holds --
            # which means `(candidate_start, candidate_end)` can never equal
            # `(stored_start, stored_end)`. The former guard was
            # `if (candidate_start, candidate_end) != (stored_start, stored_end): raise`,
            # whose condition was vacuously True and whose trailing
            # `return stored_start, stored_end` was unreachable by any caller that does not
            # violate `candidate_end = trading_dates[-1]`. Removed as dead code rather than
            # marked with a pragma, so the control flow states what actually happens.
            raise HoldoutBoundaryError(
                f"stored holdout boundary "
                f"[{stored_start.isoformat()}, {stored_end.isoformat()}] "
                f"disagrees with a recomputation over the current calendar "
                f"[{candidate_start.isoformat()}, {candidate_end.isoformat()}]; "
                "a holdout boundary must never move -- if the dataset genuinely "
                "gained new sessions this must be a deliberate, recorded decision, "
                "not a silent shift"
            )

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

    def annotate_legacy_reads(self, reason: str) -> int:
        """Mark pre-existing log entries as false positives, in place.

        Does NOT reset ``count`` and does NOT truncate ``log`` -- the audit trail
        must record that these reads happened and why they did not count.
        Idempotent: an entry that already carries an ``"annotation"`` is left
        alone, so re-running this after new legitimate reads have been recorded
        only annotates the entries that still lack one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _locked_path(self.path):
            if not self.path.exists():
                return 0

            state: dict[str, Any] = {"count": 0, "log": []}
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                state = loaded

            log = state.get("log", [])
            annotated = 0
            for entry in log:
                if "annotation" not in entry:
                    entry["annotation"] = reason
                    annotated += 1

            if annotated:
                self._atomic_write(state)

            return annotated


def default_holdout_lock_path() -> Path:
    """The single shared lock path every caller must use.

    ``settings.RESULTS_ROOT / "holdout_lock.json"`` -- imported locally to avoid
    any top-of-file circular-import risk, matching this repo's lazy-import
    convention elsewhere (e.g. ``nifty_quant.cli``).
    """
    from nifty_quant import settings

    return settings.RESULTS_ROOT / "holdout_lock.json"
