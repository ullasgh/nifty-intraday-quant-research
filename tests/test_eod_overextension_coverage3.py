from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nifty_quant.data.panel import Panel
from nifty_quant.strategy.plugins.eod_overextension import (
    EODOverExtensionParams,
    EODOverExtensionStrategy,
    _anchor_row_per_day,
)

_IST = ZoneInfo("Asia/Kolkata")


def _one_symbol_panel(periods: int, first_minute: int = 9 * 60 + 15) -> Panel:
    """Build a single-symbol, single-session panel with consecutive 1-minute bars."""
    day = dt.date(2024, 1, 2)
    day_start = pd.Timestamp(
        day.year,
        day.month,
        day.day,
        first_minute // 60,
        first_minute % 60,
        tz=_IST,
    )
    idx = pd.date_range(day_start, periods=periods, freq="1min")
    idx_utc = idx.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)

    base = np.arange(100.0, 100.0 + periods, dtype=np.float64).reshape(periods, 1)
    open_arr = base.astype(np.float32)
    high_arr = (base + 1.0).astype(np.float32)
    low_arr = (base - 0.5).astype(np.float32)
    close_arr = (base + 0.25).astype(np.float32)
    volume_arr = np.full((periods, 1), 1000.0, dtype=np.float32)

    return Panel(
        fields={
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": volume_arr,
        },
        symbols=("SYM",),
        ts=secs,
        day_offsets=np.array([0, periods], dtype=np.int32),
        dates=np.array([day], dtype=object),
    )


def test_anchor_row_skips_forbidden_bar_when_target_precedes_0915() -> None:
    """White-box contract test for the anchor-helper's forbidden-bar bump.

    ``_anchor_row_per_day`` must never anchor on the masked 09:15 bar.  A
    target before 09:15 (e.g. 09:00) lands on the first bar, which usually is
    09:15; the helper must step forward one row to 09:16.  The companion
    09:16 target already lands above the forbidden bar and therefore must not
    move further.
    """
    minute_of_day = np.array([555, 556, 557, 558, 559], dtype=np.int64)
    day_offsets = np.array([0, 5], dtype=np.int32)

    anchor_pre_open = _anchor_row_per_day(9 * 60 + 0, minute_of_day, day_offsets)
    assert anchor_pre_open[0] == 1  # 555 is forbidden, bump to row index 1 (09:16)

    anchor_0916 = _anchor_row_per_day(9 * 60 + 16, minute_of_day, day_offsets)
    assert anchor_0916[0] == 1  # already lands on non-forbidden 09:16, no extra bump


def test_precompute_with_early_session_open_avoids_forbidden_anchor_row() -> None:
    """Early configured session-open time still avoids the masked 09:15 row.

    Using ``session_open_time="09:00"`` causes the raw search to land on the
    forbidden 09:15 row.  The anchor helper bumps it to 09:16, so session
    return from 09:16 onward is computed from a real close instead of becoming
    all-NaN due to a masked anchor row.
    """
    panel = _one_symbol_panel(periods=5)
    params = EODOverExtensionParams(session_open_time="09:00")
    strategy = EODOverExtensionStrategy(params)

    signals = strategy.precompute(panel)

    session_return = signals["session_return"]
    assert session_return.shape == (5, 1)
    assert np.isnan(session_return[0, 0])  # still before the anchored session open
    assert session_return[1, 0] == 0.0  # 09:16 row used as open, so return is exactly 0


def test_precompute_single_row_panel_skips_forced_end_of_panel_exit() -> None:
    """Single-row whole-panel guard skips only the redundant forced exit.

    For ``n_rows == 1`` the ``n_rows >= 2`` guard in ``precompute`` is false,
    so the ``exit_due[n_rows - 2, :] = True`` safety valve is not executed.
    Without the guard, the index would wrap to row ``-1`` (the same row) for
    one row, silently doing no harm; for ``n_rows == 0`` it would raise by
    attempting ``exit_due[-2, :]``.  The ordinary per-session exit cursor is
    applied independently and must still mark the only available row.
    """
    panel = _one_symbol_panel(periods=1)
    strategy = EODOverExtensionStrategy(EODOverExtensionParams())

    signals = strategy.precompute(panel)

    exit_due = signals["exit_due"]
    assert exit_due.shape == (1, 1)
    assert bool(exit_due[0, 0]) is True  # per-session exit cursor still reached row 0
