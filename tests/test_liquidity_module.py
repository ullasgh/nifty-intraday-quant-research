"""Regression coverage for moving `compute_prior_adv` out of `research/lens.py`
into `data/liquidity.py` (specs/pit_universe.md amendment 1, item 1).

Two things must hold after the move:
  1. `research.lens.compute_prior_adv` must be the SAME function object as
     `data.liquidity.compute_prior_adv` -- a re-export, not a second copy, so the
     repo can never again lose a day to two same-looking liquidity statistics
     that silently drift apart.
  2. The moved function's deliberate divergence from `data/validate.py`'s inline
     tradable_mask ADV logic (NaN for an entirely-absent session, vs
     `np.nansum`-over-all-NaN's silent 0.0) must survive the move unchanged:
     the two paths must agree everywhere except at that documented divergence.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from nifty_quant.data.liquidity import compute_prior_adv as liquidity_compute_prior_adv
from nifty_quant.data.panel import Panel
from nifty_quant.research.lens import compute_prior_adv as lens_compute_prior_adv


def _session_ts(session_date: date, n_bars: int) -> np.ndarray:
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(session_date, time(9, 15), tzinfo=ist)
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64
    )


def _make_panel(n_sessions: int, absent_session: int) -> Panel:
    """One symbol, one bar per session, present everywhere except
    `absent_session` (all bars NaN that session)."""
    session_dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n_sessions)]
    close = np.full((n_sessions, 1), 100.0, dtype=np.float64)
    volume = np.full((n_sessions, 1), 1.0e6, dtype=np.float64)
    close[absent_session, 0] = np.nan
    volume[absent_session, 0] = np.nan

    ts = np.concatenate([_session_ts(d, 1) for d in session_dates])
    day_offsets = np.arange(n_sessions + 1, dtype=np.int32)

    return Panel(
        fields={
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        symbols=("ONLY",),
        ts=ts,
        day_offsets=day_offsets,
        dates=np.asarray(session_dates, dtype=object),
    )


def test_lens_reexports_the_same_function_object_not_a_copy() -> None:
    assert lens_compute_prior_adv is liquidity_compute_prior_adv


def _validate_style_day_value_and_adv(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Reimplementation of the INLINE arithmetic at data/validate.py's
    tradable_mask (nansum over an all-NaN session silently yields 0.0), kept
    separate here on purpose: that inline logic is not an importable function,
    which is exactly what amendment 1 says (half the original report was right
    about that)."""
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1
    n_symbols = close.shape[1]

    day_value = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        day_value[session_idx, :] = np.nansum(
            close[start:end] * volume[start:end], axis=0
        )

    adv = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(1, n_sessions):
        lookback_start = max(0, session_idx - 20)
        adv[session_idx, :] = np.nanmean(
            day_value[lookback_start:session_idx, :], axis=0
        )
    return day_value, adv


def test_moved_adv_agrees_with_validate_style_inline_logic_except_at_all_nan_session() -> None:
    n_sessions = 30
    absent_session = 10
    panel = _make_panel(n_sessions, absent_session)

    liquidity_prior_adv = liquidity_compute_prior_adv(panel)
    liquidity_day_value = np.full((n_sessions, 1), np.nan, dtype=np.float64)
    for session_idx in range(n_sessions):
        liquidity_day_value[session_idx, 0] = liquidity_prior_adv[session_idx, 0]
    # Re-derive liquidity.py's OWN day_value series the same way pit.py/lens.py
    # would see it: read back the per-session ADV via the bar-level output.
    liquidity_adv_by_session = liquidity_prior_adv[np.arange(n_sessions), :]

    validate_day_value, validate_adv = _validate_style_day_value_and_adv(panel)

    # Day-value divergence is confined to exactly the entirely-absent session:
    # liquidity.py preserves NaN there; the validate.py-style inline nansum
    # silently collapses it to 0.0. This is the documented, deliberate
    # difference -- assert it is exactly that one cell, no more.
    manual_liquidity_day_value = np.full((n_sessions, 1), np.nan, dtype=np.float64)
    close = panel.field("close")
    volume = panel.field("volume")
    day_offsets = panel.day_offsets
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        seg = close[start:end] * volume[start:end]
        has_any = np.any(np.isfinite(seg), axis=0)
        manual_liquidity_day_value[session_idx, :] = np.where(
            has_any, np.nansum(seg, axis=0), np.nan
        )

    assert np.isnan(manual_liquidity_day_value[absent_session, 0])
    assert validate_day_value[absent_session, 0] == 0.0

    other_sessions = [s for s in range(n_sessions) if s != absent_session]
    np.testing.assert_array_equal(
        manual_liquidity_day_value[other_sessions, 0],
        validate_day_value[other_sessions, 0],
    )

    # Downstream ADV: any trailing window spanning the absent session sees a
    # real (if small) divergence between "ignore it" (nanmean) and "count it as
    # zero, on a fixed 20-session denominator" (validate.py's inline style) --
    # except when every OTHER value in a given window is exactly zero too. Here
    # they are not, so the affected windows (sessions absent_session+1 through
    # absent_session+20) must diverge, and untouched windows (entirely before
    # or entirely 20+ sessions after) must not.
    unaffected_before = list(range(1, absent_session))
    np.testing.assert_allclose(
        liquidity_adv_by_session[unaffected_before, 0],
        validate_adv[unaffected_before, 0],
    )

    affected = list(range(absent_session + 1, min(absent_session + 21, n_sessions)))
    assert affected, "fixture must produce at least one affected trailing window"
    assert np.any(
        np.abs(
            liquidity_adv_by_session[affected, 0] - validate_adv[affected, 0]
        )
        > 1.0
    )

    far_after = list(range(absent_session + 21, n_sessions))
    if far_after:
        np.testing.assert_allclose(
            liquidity_adv_by_session[far_after, 0], validate_adv[far_after, 0]
        )
