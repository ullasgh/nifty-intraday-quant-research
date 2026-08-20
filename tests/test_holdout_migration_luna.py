"""Migration-path test for `specs/holdout_integrity.md` AMENDMENT 1 item 3.

Neither of the two independently-authored contract suites
(`test_holdout_integrity_a.py`, `test_holdout_integrity_b.py`) exercises the
upgrade path for a lock file that only ever had `count`/`log` (the shape the
real, pre-fix `results/holdout_lock.json` has today) -- both suites seed their
fixtures with the full post-fix shape instead. The amendment explicitly calls
this out: "Add it -- a migration that has never been run is a migration that
does not work." This file supplies that missing coverage. Written by
`worker-luna` (gpt-5.6-luna); this file is NOT part of the two locked
contract suites and may be edited/extended.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from nifty_quant.research.splits import HoldoutLock


def _make_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def test_migration_from_legacy_count_log_only_file_persists_boundary(tmp_path: Path) -> None:
    """A legacy file with only count/log (today's real shape) gains a boundary on
    first use, and count/log are left byte-for-byte untouched."""
    legacy_log = [
        {"ts": f"2026-08-19T0{i}:00:00+00:00", "reason": "walkforward split rolling_000"}
        for i in range(7)
    ]
    lock_path = tmp_path / "holdout_lock.json"
    lock_path.write_text(
        json.dumps({"count": 7, "log": legacy_log}), encoding="utf-8"
    )

    dates = _make_dates(date(2020, 1, 1), date(2026, 8, 14))
    lock = HoldoutLock(path=lock_path)
    start, end = lock.holdout_range(dates)

    assert (start, end) == lock.holdout_range(dates[:5])  # now fixed/reused

    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == 7, "migration must not reset the counter"
    assert state["log"] == legacy_log, "migration must not touch existing log entries"
    assert state["holdout_start"] == start.isoformat()
    assert state["holdout_end"] == end.isoformat()


def test_migration_is_idempotent_across_instances(tmp_path: Path) -> None:
    """A second HoldoutLock instance against the same migrated path reuses the
    now-stored boundary rather than re-deriving it from a different window."""
    lock_path = tmp_path / "holdout_lock.json"
    lock_path.write_text(json.dumps({"count": 3, "log": []}), encoding="utf-8")

    full_dates = _make_dates(date(2019, 1, 1), date(2026, 8, 14))
    short_dates = full_dates[:30]

    first = HoldoutLock(path=lock_path).holdout_range(full_dates)
    second = HoldoutLock(path=lock_path).holdout_range(short_dates)

    assert first == second
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["count"] == 3
    assert state["log"] == []


def test_real_lock_file_migrates_cleanly_without_mutating_the_original(tmp_path: Path) -> None:
    """Sanity-checks the migration logic against a byte-for-byte COPY of the real
    production lock's `count`/`log`, reduced to the legacy (pre-migration,
    count/log-only) shape this spec's migration path must handle, proving the
    fix handles the actual file's real history -- WITHOUT ever touching the real
    file itself (a fresh tmp_path copy is used throughout).

    Deliberately does NOT assert on whether the real file itself currently has a
    stored boundary already -- that is an external, mutable fact (this repo has
    concurrent agents that may migrate it for real between runs; the count/log
    values ONLY are read here, as evidence of a plausible legacy log to migrate
    from, not to gate this test's pass/fail on the real file's own transient
    migration status).
    """
    from nifty_quant import settings

    real_lock_path = settings.RESULTS_ROOT / "holdout_lock.json"
    assert real_lock_path.exists(), "sanity: the real lock file still exists"
    real_state = json.loads(real_lock_path.read_text(encoding="utf-8"))
    legacy_shape = {"count": real_state["count"], "log": real_state["log"]}

    copy_path = tmp_path / "holdout_lock.json"
    copy_path.write_text(json.dumps(legacy_shape), encoding="utf-8")

    dates = _make_dates(date(2020, 1, 1), date(2026, 8, 14))
    HoldoutLock(path=copy_path).holdout_range(dates)

    migrated = json.loads(copy_path.read_text(encoding="utf-8"))
    assert migrated["count"] == legacy_shape["count"]
    assert migrated["log"] == legacy_shape["log"]
    assert "holdout_start" in migrated and "holdout_end" in migrated
