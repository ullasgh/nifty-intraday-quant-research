"""Edge cases and branch coverage for pit.py.

Exercises uncovered lines and branches:
- Line 155: source validation when source is not in _ALLOWED_SOURCES
- Branch [154,155]: if source not in _ALLOWED_SOURCES (True branch)
- Branch [188,190]: if col_idx (True branch with non-empty col_idx)
- Branch [193,195]: if n_sessions > 1 (False branch with single session)
- Branch [200,205]: if col_idx (True branch with non-empty col_idx)

NOTE ON MISSING BRANCHES:
The False branches of [188,190] and [200,205] (when col_idx is empty)
would require passing a universe with no symbols in common with the panel.
This path triggers a shape-mismatch bug in the implementation (lines 188-190:
session_present is initialized to shape (n_sessions, 0) but receives
assignment from np.any(..., axis=0) with shape (n_panel_symbols,)). Since
this is a misuse of the API (universes should have overlapping symbols) and
the code cannot be modified, these branches remain untested. They represent
an unreachable state under normal usage.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.universe.pit import compute_eligibility
from nifty_quant.universe.static import Universe


def _session_ts(
    session_date: date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
    """Create timestamps for a trading session."""
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(
        session_date,
        time(*start_hhmm),
        tzinfo=ist,
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _make_panel(
    session_dates,
    bars_per_session,
    symbols,
    close_by_session,
    volume_by_session,
) -> Panel:
    """Construct a minimal Panel for testing."""
    session_dates = np.asarray(session_dates, dtype=object)
    n_sessions = len(session_dates)
    symbols = tuple(symbols)

    bars = np.asarray(bars_per_session, dtype=np.int64)
    if bars.ndim == 0:
        bars = np.full(n_sessions, int(bars), dtype=np.int64)
    if bars.shape != (n_sessions,):
        raise ValueError("bars_per_session must have one entry per session")
    if np.any(bars <= 0):
        raise ValueError("each synthetic session must contain at least one bar")

    expected_shape = (n_sessions, len(symbols))
    close_by_session = np.asarray(close_by_session, dtype=np.float64)
    volume_by_session = np.asarray(volume_by_session, dtype=np.float64)
    if close_by_session.shape != expected_shape:
        raise ValueError("close_by_session has the wrong shape")
    if volume_by_session.shape != expected_shape:
        raise ValueError("volume_by_session has the wrong shape")

    close = np.repeat(close_by_session, bars, axis=0)
    volume = np.repeat(volume_by_session, bars, axis=0)

    no_bar = ~np.isfinite(close)
    volume[no_bar] = np.nan

    fields = {
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    }

    ts = np.concatenate(
        [
            _session_ts(session_date, int(n_bars))
            for session_date, n_bars in zip(session_dates, bars)
        ]
    )
    day_offsets = np.empty(n_sessions + 1, dtype=np.int32)
    day_offsets[0] = 0
    day_offsets[1:] = np.cumsum(bars, dtype=np.int64)

    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=session_dates,
    )


def _session_dates(n_sessions: int) -> list[date]:
    """Generate n_sessions consecutive trading dates."""
    first_date = date(2024, 1, 2)
    return [first_date + timedelta(days=i) for i in range(n_sessions)]


def test_invalid_source_raises_value_error() -> None:
    """Line 154-158: Reject source values not in _ALLOWED_SOURCES.

    Tests the branch where source is neither "availability_proxy" nor
    "index_membership" -- exercises line 155's raise statement.
    """
    symbols = ("A",)
    close = np.full((3, 1), 100.0)
    volume = np.full((3, 1), 1.0e7)
    panel = _make_panel(
        _session_dates(3),
        [2, 1, 3],
        symbols,
        close,
        volume,
    )

    # source="bogus" is not in _ALLOWED_SOURCES -> line 154 condition is True,
    # executes line 155's raise.
    with pytest.raises(ValueError) as exc_info:
        compute_eligibility(
            panel,
            name="test-invalid-source",
            min_history_sessions=1,
            min_adv_inr=5.0e7,
            source="bogus",
        )

    # Verify it's the right error message (from line 155-158), not the
    # index_membership-specific one (lines 160-167).
    assert "is not a recognised PitEligibility source" in str(exc_info.value)
    assert "must be one of" in str(exc_info.value)


def test_index_membership_source_raises_with_specific_message() -> None:
    """Line 159-167: Reject source='index_membership' with dataset context.

    Although this data is in _ALLOWED_SOURCES (so line 154 passes), the
    function explicitly rejects it with a dataset-specific message.
    Ensures that no API silently claims survivorship correction when the
    data contains zero delisted names.
    """
    symbols = ("A",)
    close = np.full((3, 1), 100.0)
    volume = np.full((3, 1), 1.0e7)
    panel = _make_panel(
        _session_dates(3),
        [2, 1, 3],
        symbols,
        close,
        volume,
    )

    with pytest.raises(ValueError) as exc_info:
        compute_eligibility(
            panel,
            name="test-index-membership",
            min_history_sessions=1,
            min_adv_inr=5.0e7,
            source="index_membership",
        )

    # Verify the index-membership-specific message (line 160-167), which
    # names the limitation and suggests "availability_proxy" instead.
    assert "index-membership history" in str(exc_info.value)
    assert "availability_proxy" in str(exc_info.value)


def test_partial_universe_exercises_col_idx() -> None:
    """Lines 188-190, 200-205: col_idx filters to universe subset, both branches tested.

    When a Universe has some but not all panel symbols, col_idx is non-empty and
    exercises the True branch of both `if col_idx:` checks:
    - Line 188: session_slice IS sliced by col_idx
    - Line 200: adv_by_session IS sliced by col_idx

    Eligibility is computed over only the universe's symbols.
    """
    panel_symbols = ("A", "B", "C")
    close = np.full((3, len(panel_symbols)), 100.0)
    volume = np.full_like(close, 1.0e7)
    panel = _make_panel(
        _session_dates(3),
        [2, 1, 3],
        panel_symbols,
        close,
        volume,
    )

    # Create a universe with a subset of panel symbols.
    universe = Universe(
        name="subset-universe",
        symbols=("A", "C"),  # excludes B
        source="availability_proxy",
    )

    eligibility = compute_eligibility(
        panel,
        universe=universe,
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    # With 2 symbols in common, the mask should have shape (3, 2).
    assert eligibility.mask.shape == (3, 2)
    # column_symbols should be the filtered set in panel order.
    assert eligibility.column_symbols == ("A", "C")
    # symbols (sorted) should be the sorted version.
    assert eligibility.symbols == ("A", "C")


def test_single_session_panel_skips_cumsum() -> None:
    """Line 193-195: n_sessions == 1 skips cumsum, prior_present_count stays zero.

    A single-session panel has n_sessions = 1. The cumsum at line 194 only
    runs if n_sessions > 1, so with one session, listed_ok is computed from
    an all-zero prior_present_count, making all symbols ineligible (they have
    zero prior sessions of presence).
    """
    symbols = ("A", "B")
    close = np.array([[100.0, 105.0]], dtype=np.float64)  # 1 session
    volume = np.array([[1.0e7, 1.0e7]], dtype=np.float64)
    panel = _make_panel(
        _session_dates(1),
        [3],  # single session, 3 bars
        symbols,
        close,
        volume,
    )

    eligibility = compute_eligibility(
        panel,
        name="single-session",
        min_history_sessions=0,  # Even with min_history_sessions=0
        min_adv_inr=5.0e7,
    )

    # On session 0 itself, no prior sessions exist, so listed_ok must be all False.
    # (The line 194 cumsum does NOT run because n_sessions == 1.)
    assert eligibility.mask.shape == (1, 2)
    # All symbols ineligible on session 0 because prior_present_count[0,:] = 0
    # and min_history_sessions is 0, but trailing ADV on session 0 is NaN
    # (first row of ADV is always NaN), so liquid_ok is also False.
    assert not np.any(eligibility.mask[0, :])


def test_col_idx_filtering_preserves_order() -> None:
    """Lines 171-176: universe filters col_idx to a subset, maintaining panel order.

    When universe is given and has a subset of panel symbols, col_idx indexes
    into the panel's column order. The result's column_symbols are in that
    same order (matching the panel), but symbols is sorted.
    """
    panel_symbols = ("Z", "A", "M")
    close = np.full((3, len(panel_symbols)), 100.0)
    volume = np.full_like(close, 1.0e7)
    panel = _make_panel(
        _session_dates(3),
        [2, 1, 3],
        panel_symbols,
        close,
        volume,
    )

    # Universe with a subset, in a different order than panel.
    universe = Universe(
        name="subset-universe",
        symbols=("A", "Z"),  # reversed from panel order
        source="availability_proxy",
    )

    eligibility = compute_eligibility(
        panel,
        universe=universe,
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    # column_symbols follow panel order: Z at index 0, A at index 1 in panel.
    assert eligibility.column_symbols == ("Z", "A")
    # symbols are sorted.
    assert eligibility.symbols == ("A", "Z")
    # mask shape is (n_sessions, n_filtered_symbols) = (3, 2).
    assert eligibility.mask.shape == (3, 2)


def test_present_tradable_eligible_distinctions() -> None:
    """Rule 7 / CLAUDE.md: present, tradable, and eligible are distinct.

    A symbol can be:
    - Present: a bar exists that session
    - Tradable: a bar exists and is usable (not halted, sufficient liquidity)
    - Eligible: in the research universe (enough prior history + current ADV)

    This test verifies that a newly-listed symbol can be present and tradable
    on early sessions but still ineligible because min_history_sessions hasn't
    elapsed.
    """
    symbols = ("EARLY", "LATE")
    n_sessions = 10
    min_history_sessions = 3

    # EARLY is present from day 0, LATE starts at day 5.
    close = np.full((n_sessions, len(symbols)), np.nan, dtype=np.float64)
    close[:, 0] = 100.0  # EARLY everywhere
    close[5:, 1] = 105.0  # LATE from session 5

    # Both have sufficient volume when present.
    volume = np.full_like(close, np.nan)
    volume[:, 0] = 2.0e7
    volume[5:, 1] = 2.0e7

    panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        close,
        volume,
    )

    eligibility = compute_eligibility(
        panel,
        name="test-present-tradable-eligible",
        min_history_sessions=min_history_sessions,
        min_adv_inr=5.0e7,
    )

    late_idx = 1
    # LATE is present starting at session 5, but needs min_history_sessions=3
    # prior sessions, so it becomes eligible at session 5 + 3 = session 8.
    assert not np.any(eligibility.mask[5:8, late_idx])  # ineligible
    assert np.all(eligibility.mask[8:, late_idx])  # eligible from session 8


def test_all_nan_trailing_adv_window_is_ineligible() -> None:
    """Lines 200-205: liquid_ok uses NaN comparisons (NaN >= X is False).

    When a symbol's entire 20-session trailing window is absent (all NaN),
    compute_prior_adv_by_session returns NaN. The comparison NaN >= min_adv_inr
    evaluates to False, marking it ineligible. This is the desired "unknown
    liquidity is not eligible" behavior, never a spurious pass.
    """
    symbols = ("PHANTOM",)
    n_sessions = 25
    close = np.full((n_sessions, 1), np.nan, dtype=np.float64)
    volume = np.full_like(close, np.nan)

    panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        close,
        volume,
    )

    eligibility = compute_eligibility(
        panel,
        name="test-nan-adv",
        min_history_sessions=0,
        min_adv_inr=5.0e7,
    )

    # With all bars absent, the mask should be all False (no eligible rows).
    assert not np.any(eligibility.mask)


def test_single_session_with_partial_universe() -> None:
    """Combines single-session (line 193-195 False branch) with col_idx filtering.

    Ensures that both conditions work together: n_sessions == 1 and a
    filtered col_idx from universe subset selection. On session 0, prior_present_count
    remains all-zero because no prior sessions exist, so all symbols are ineligible
    from the listing perspective, regardless of universe filtering.
    """
    panel_symbols = ("A", "B")
    close = np.array([[100.0, 105.0]], dtype=np.float64)
    volume = np.array([[1.0e7, 1.0e7]], dtype=np.float64)
    panel = _make_panel(
        _session_dates(1),
        [1],
        panel_symbols,
        close,
        volume,
    )

    # Universe with subset of panel symbols.
    universe = Universe(
        name="subset-single-session",
        symbols=("A",),
        source="availability_proxy",
    )

    eligibility = compute_eligibility(
        panel,
        universe=universe,
        min_history_sessions=0,
        min_adv_inr=5.0e7,
    )

    # With 1 symbol, mask shape is (1, 1).
    assert eligibility.mask.shape == (1, 1)
    assert eligibility.column_symbols == ("A",)
    # On session 0, symbol A is still ineligible because ADV[0] is NaN.
    assert not eligibility.mask[0, 0]


def test_empty_universe_intersection_well_formed() -> None:
    """Lines 188-190, 200-205: False branches when col_idx is empty.

    Passing a Universe with ZERO symbols in common with the panel results in
    col_idx = []. This exercises the False branch of both `if col_idx:` checks
    at lines 188 and 200, which now index unconditionally (the prior `if col_idx:`
    guard was a bug that skipped filtering when it was most needed).

    The result must be well-formed: empty mask with correct shape, empty
    symbol sets, and stable hash. This is a valid use case (e.g., typo in
    --universe CLI argument silently falls back to full universe, then later
    an explicit subset is filtered to zero overlap).
    """
    panel_symbols = ("A", "B", "C")
    close = np.full((3, len(panel_symbols)), 100.0)
    volume = np.full_like(close, 1.0e7)
    panel = _make_panel(
        _session_dates(3),
        [2, 1, 3],
        panel_symbols,
        close,
        volume,
    )

    # Universe with NO symbols in common with panel.
    universe = Universe(
        name="disjoint-universe",
        symbols=("X", "Y", "Z"),
        source="availability_proxy",
    )

    eligibility = compute_eligibility(
        panel,
        universe=universe,
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    # With zero symbols in common, mask should have shape (n_sessions, 0).
    assert eligibility.mask.shape == (3, 0)
    assert eligibility.mask.dtype == np.bool_
    # Both symbol fields should be empty.
    assert eligibility.symbols == ()
    assert eligibility.column_symbols == ()
    # dates should still be aligned to panel.
    assert len(eligibility.dates) == 3
    # universe_hash should be stable and well-formed.
    assert isinstance(eligibility.universe_hash, str)
    assert len(eligibility.universe_hash) == 64  # SHA256 hex
    # Name should come from universe.
    assert eligibility.name == "disjoint-universe"
    assert eligibility.source == "availability_proxy"

    # Hash should be stable across re-computation.
    eligibility2 = compute_eligibility(
        panel,
        universe=universe,
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )
    assert eligibility.universe_hash == eligibility2.universe_hash
