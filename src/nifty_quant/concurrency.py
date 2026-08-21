"""Shared cross-process locking primitive.

Extracted from ``nifty_quant.research.splits`` so that other modules needing
a cross-process mutual-exclusion lock (e.g. ``nifty_quant.data.panel``) reuse
the same implementation instead of growing a second copy. See
``config_hash`` (three copies, repo history) for why duplication of this
kind of thing is worth avoiding.
"""

# DELIBERATE DUPLICATION, do not "consolidate" without reading this.
#
# `research/splits.py::_locked_path` implements the same fcntl.flock pattern. They were merged
# once and it broke two tests: `tests/test_splits_coverage.py` pins `splits.fcntl` and
# `splits._locked_path` as LIVE MODULE ATTRIBUTES -- one monkeypatches `splits.fcntl = None` to
# exercise the no-fcntl fallback, another deletes the module from `sys.modules` and re-imports it
# to prove the `except ImportError: fcntl = None` branch works.
#
# That is not an accident of test style: HoldoutLock guards a one-shot out-of-sample read, and its
# fallback behaviour when fcntl is unavailable is part of its contract, so the test pins the import
# itself. Re-exporting from here would remove the very import the test inspects.
#
# So: ~15 lines are duplicated on purpose. If you change the locking semantics in one file, change
# them in the other, and keep this comment in both.


from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

fcntl: ModuleType | None
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


@contextmanager
def locked_path(path: Path) -> Iterator[None]:
    """Best-effort advisory cross-process lock keyed on ``path``.

    Blocks until the lock is acquired (no spin loop: the wait happens inside
    the kernel via ``flock``), so a waiter never busy-polls. On platforms
    without ``fcntl`` this degrades to a no-op (single-process only).
    """
    if fcntl is None:
        yield
        return

    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
