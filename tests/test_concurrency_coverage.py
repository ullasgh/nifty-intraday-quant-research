"""Targeted coverage for `concurrency.py` (new module, ungated) and
`data/panel.py`'s waiter-re-check-under-lock branch (regressed from 100%).

Every load in this file that touches a `PanelSpec` points `settings.CACHE_ROOT` at
`tmp_path` (never the real `cache/` directory) and restricts the requested symbols/date
range to keep the materialised cache tiny. Reads real committed `data/bars/1/...`
parquet (read-only; nothing under `data/` is ever written).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

import nifty_quant.concurrency as concurrency
import nifty_quant.data.panel as panel_module
from nifty_quant.data.manifest import Manifest
from nifty_quant.data.panel import PanelSpec, load_panel

# ---------------------------------------------------------------------------
# `concurrency.locked_path` -- the no-`fcntl` fallback (lines 50-51).
# ---------------------------------------------------------------------------


def test_locked_path_is_a_pure_noop_when_fcntl_unavailable(tmp_path: Path, monkeypatch):
    """On a platform without `fcntl` (or with it monkeypatched away, mirroring the
    sibling test for `research.splits.locked_path`), `locked_path` must still run the
    wrapped body -- but must NOT create the advisory lock file it would otherwise
    create, since there is no lock to hold."""
    monkeypatch.setattr(concurrency, "fcntl", None)

    target = tmp_path / "resource"
    body_ran: list[str] = []
    with concurrency.locked_path(target):
        body_ran.append("ran")

    assert body_ran == ["ran"]
    lock_path = target.with_name(target.name + ".lock")
    assert not lock_path.exists(), (
        "fcntl=None fallback must be a true no-op: it must never create the "
        "advisory .lock file"
    )


def test_locked_path_creates_lock_file_with_real_fcntl_for_contrast(tmp_path: Path):
    """Contrast case: with real `fcntl` available, the same call DOES create the
    advisory lock file, so the no-op assertion above is pinned to the `fcntl is None`
    condition specifically rather than being true unconditionally."""
    target = tmp_path / "resource2"
    with concurrency.locked_path(target):
        lock_path = target.with_name(target.name + ".lock")
        assert lock_path.exists()


# ---------------------------------------------------------------------------
# `data/panel.py`'s materialization race: a waiter that misses the fast-path check,
# acquires the lock, and finds the lock-holder already published (line 564) -- must
# use the freshly-published data and must NEVER rebuild.
# ---------------------------------------------------------------------------


def test_waiter_uses_materialization_published_between_outer_check_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    try:
        manifest = Manifest.load()
    except FileNotFoundError:
        pytest.skip("Real data/MANIFEST.json not found; skipping.")

    year = 2018
    candidates = sorted(
        sym for sym, cov in manifest.coverage.items() if year in cov.years
    )
    if len(candidates) < 1:
        pytest.skip(f"Need >=1 symbol with {year} data, found {len(candidates)}.")
    symbols = (candidates[0],)
    field = "close"
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 1, 5)

    cache_root = tmp_path / "cache"
    monkeypatch.setattr("nifty_quant.settings.CACHE_ROOT", cache_root)

    spec = PanelSpec(freq="1", fields=(field,), symbols=symbols, start=start, end=end)

    # Build (and materialize) for real once, so a genuine, complete materialization
    # exists on disk before the race is simulated.
    warm = load_panel(spec, memmap=True)
    warm_arr = np.array(warm.field(field), dtype=np.float32)

    real_try_open = panel_module._try_open_materialized
    call_count = {"n": 0}

    def fake_try_open(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the outer, no-lock fast-path check missing the (real, already
            # complete) materialization.
            return None
        return real_try_open(*args, **kwargs)

    def _rebuild_must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "the rebuild path (_build_materialized) must never run when the waiter's "
            "re-check under the lock finds a materialization the lock-holder already "
            "published"
        )

    monkeypatch.setattr(panel_module, "_try_open_materialized", fake_try_open)
    monkeypatch.setattr(panel_module, "_build_materialized", _rebuild_must_not_be_called)

    result = load_panel(spec, memmap=True)

    # Fast-path check (miss), then the re-check under the lock (hit): exactly 2 calls.
    assert call_count["n"] == 2
    result_arr = np.array(result.field(field), dtype=np.float32)
    np.testing.assert_array_equal(result_arr, warm_arr)
