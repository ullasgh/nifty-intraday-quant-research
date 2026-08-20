"""SPEC DEFECTS / AMBIGUITIES FOUND

1. The spec names no module path, class, or function signature for the per-session eligibility
   mask, the source validator, or universe_hash. This file assumes the documented
   ``nifty_quant.universe.pit`` interface; the other independent test author received the same
   spec but not this assumed interface, so their divergent guess is itself a finding.

2. Item A.1's "at least min_history_sessions prior sessions with a present bar" is a COUNT over
   history, with no requirement that those sessions be contiguous or include the very first
   session. Item 8's reference control (continuous_coverage_mask) instead checks ONLY the first
   and last session's presence, ignoring the count in between entirely. These definitions diverge
   whenever a symbol has interior gaps (for example, present on day 0 and day N-1 but missing
   10 interior days: it passes the old first/last control but may fail a strict count threshold).
   The spec claims the new mask "generalises" the old control without addressing this case. The
   fixture for item 8 sidesteps the ambiguity with zero interior gaps rather than assume a
   resolution.

3. Section A.2 says to reuse ``data/validate.py:736-739`` using a ``compute_prior_adv``-style
   calculation, but no function named ``compute_prior_adv`` exists there today. Those lines are
   inline arithmetic inside ``tradable_mask``, not an extractable named helper. Reuse therefore
   implies extracting a new helper, which the spec does not say explicitly.

4. The trailing-ADV window uses ``max(0, session_idx - 20)`` as its lower bound, silently
   shrinking near the start of a panel instead of being undefined until a full 20-session
   trailing window exists. The spec's "eligible ... min_history_sessions sessions after"
   language implies a hard cutoff for the listing leg but says nothing about whether the liquidity
   leg should require a full window or shrink. These tests only assert liquidity outcomes on
   sessions with a full 20-session trailing history.

5. Item 5 requires constructing an ``index_membership`` result to raise but does not name the
   exception type. The test accepts either ``ValueError`` or ``NotImplementedError``.

6. Item 6's ``universe_hash`` implies that the eligibility result carries a name and construction
   parameters (``min_history_sessions`` and ``min_adv_inr``) as first-class fields, even though
   Section A only describes the return value as a boolean mask plus a ``source`` field. The
   hash's stated inputs are therefore richer than the described return shape.

7. The reference ADV initializes session zero's ADV to NaN, so no symbol can be eligible on the
   first panel session under a normal finite threshold. The causality fixture treats a stable
   symbol as eligible throughout the evaluable portion of the baseline panel.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.universe.pit import PitEligibility, compute_eligibility
from nifty_quant.universe.static import Universe, equity_symbols, survivorship_report


def _session_ts(
    session_date: date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
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
    first_date = date(2024, 1, 2)
    return [first_date + timedelta(days=i) for i in range(n_sessions)]


def test_item1_newly_listed_name_waits_for_prior_sessions() -> None:
    n_sessions = 40
    first_present = 18
    min_history_sessions = 2
    symbols = ("CORE", "LATE")

    close = np.full((n_sessions, len(symbols)), np.nan, dtype=np.float64)
    close[:, 0] = 100.0
    close[first_present:, 1] = 100.0

    volume = np.full_like(close, np.nan)
    volume[:, 0] = 1.0e7
    volume[first_present:, 1] = 1.0e7

    panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        close,
        volume,
    )
    eligibility = compute_eligibility(
        panel,
        name="item1-pit",
        min_history_sessions=min_history_sessions,
        min_adv_inr=5.0e7,
    )

    late_idx = panel.symbols.index("LATE")
    eligible_from = first_present + min_history_sessions

    assert np.array_equal(
        eligibility.mask[:eligible_from, late_idx],
        np.zeros(eligible_from, dtype=bool),
    )
    assert np.array_equal(
        eligibility.mask[eligible_from:, late_idx],
        np.ones(n_sessions - eligible_from, dtype=bool),
    )


def test_item2_causality_preserves_all_symbol_prefixes_in_both_directions() -> None:
    n_sessions = 32
    pivot = 22
    symbols = ("ABSENT", "STABLE")
    absent_idx = symbols.index("ABSENT")
    stable_idx = symbols.index("STABLE")

    baseline_close = np.full((n_sessions, len(symbols)), np.nan, dtype=np.float64)
    baseline_close[:, stable_idx] = 100.0
    baseline_volume = np.full_like(baseline_close, np.nan)
    baseline_volume[:, stable_idx] = 2.0e7

    baseline_panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        baseline_close,
        baseline_volume,
    )

    added_close = baseline_close.copy()
    added_close[pivot:, absent_idx] = 100.0
    added_volume = baseline_volume.copy()
    added_volume[pivot:, absent_idx] = 2.0e7
    added_panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        added_close,
        added_volume,
    )

    removed_close = baseline_close.copy()
    removed_close[pivot:, stable_idx] = np.nan
    removed_volume = baseline_volume.copy()
    removed_volume[pivot:, stable_idx] = np.nan
    removed_panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        removed_close,
        removed_volume,
    )

    kwargs = {
        "name": "item2-pit",
        "min_history_sessions": 1,
        "min_adv_inr": 5.0e7,
    }
    mask_baseline = compute_eligibility(baseline_panel, **kwargs).mask
    mask_added = compute_eligibility(added_panel, **kwargs).mask
    mask_removed = compute_eligibility(removed_panel, **kwargs).mask

    assert np.array_equal(mask_baseline[:pivot, :], mask_added[:pivot, :])
    assert np.array_equal(mask_baseline[:pivot, :], mask_removed[:pivot, :])

    assert not np.any(mask_baseline[:, absent_idx])
    assert np.all(mask_baseline[pivot:, stable_idx])

    assert not bool(mask_added[pivot, absent_idx])
    assert np.all(mask_added[pivot + 1 :, absent_idx])

    # Removing STABLE's presence FROM session `pivot` onward must not retroactively
    # change eligibility exactly AT `pivot`: eligibility on session d depends only on
    # data strictly before d, and session `pivot` itself is not "before pivot", so its
    # own (now-removed) bar is never part of that computation. This is the sharpest
    # single-cell demonstration of direction (b): the causal boundary is exclusive of
    # d itself, not merely "history vs. future".
    assert bool(mask_removed[pivot, stable_idx])


def test_item3_illiquid_name_is_excluded_before_cross_sectional_ranking() -> None:
    n_sessions = 42
    symbols = ("A", "B", "C")

    close = np.full((n_sessions, len(symbols)), 100.0, dtype=np.float64)
    volume = np.full((n_sessions, len(symbols)), 1.0e7, dtype=np.float64)
    volume[:20, 1] = 1.0

    panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        close,
        volume,
    )
    eligibility = compute_eligibility(
        panel,
        name="item3-pit",
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    session_idx = 20
    eligible_row = eligibility.mask[session_idx, :]
    assert np.array_equal(
        eligible_row,
        np.asarray([True, False, True], dtype=bool),
    )
    assert np.array_equal(
        eligibility.mask[41, :],
        np.asarray([True, True, True], dtype=bool),
    )

    raw_values = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)

    compressed_values = np.compress(eligible_row, raw_values)
    compressed_ranks = pd.Series(compressed_values).rank()

    omitted_values = np.asarray(
        [
            raw_values[index]
            for index in range(len(raw_values))
            if eligible_row[index]
        ]
    )
    omitted_ranks = pd.Series(omitted_values).rank()

    assert compressed_ranks.equals(omitted_ranks)

    zero_filled_values = raw_values.copy()
    zero_filled_values[~eligible_row] = 0.0
    zero_filled_ranks = pd.Series(zero_filled_values).rank()

    assert np.any(
        zero_filled_ranks.to_numpy()[eligible_row]
        != compressed_ranks.to_numpy()
    )


def test_item4_mask_shape_uses_session_count_with_irregular_bar_counts() -> None:
    symbols = ("A", "B")
    bars_per_session = (375, 60, 105, 375)
    close = np.full((len(bars_per_session), len(symbols)), 100.0)
    volume = np.full_like(close, 1.0e7)

    panel = _make_panel(
        _session_dates(len(bars_per_session)),
        bars_per_session,
        symbols,
        close,
        volume,
    )

    assert np.array_equal(
        panel.day_offsets,
        np.asarray([0, 375, 435, 540, 915], dtype=np.int32),
    )
    assert panel.n_rows() == 915
    assert panel.n_days() == 4

    eligibility = compute_eligibility(
        panel,
        name="item4-pit",
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    assert eligibility.mask.shape == (4, 2)
    assert eligibility.mask.dtype == np.bool_
    assert eligibility.dates.shape == (4,)
    assert np.array_equal(eligibility.dates, panel.dates)


def test_item5_source_defaults_to_availability_proxy_and_rejects_membership() -> None:
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

    default_result = compute_eligibility(
        panel,
        name="item5-pit",
        min_history_sessions=1,
        min_adv_inr=5.0e7,
    )

    assert isinstance(default_result, PitEligibility)
    assert default_result.source == "availability_proxy"

    with pytest.raises((ValueError, NotImplementedError)):
        compute_eligibility(
            panel,
            name="item5-pit",
            min_history_sessions=1,
            min_adv_inr=5.0e7,
            source="index_membership",
        )


def test_item6_universe_hash_is_deterministic_and_sensitive_to_one_flip() -> None:
    # Mutating the LAST session's own data is causally inert (there is no later
    # session whose trailing window could observe it), so the flip has to land on a
    # session strictly BEFORE the panel's end. We mutate the second-to-last session
    # (index -2) and use a volume computed so that including vs. excluding that one
    # session's contribution to the 20-session trailing-ADV window at the LAST session
    # (index -1) straddles min_adv_inr=5e7: the real compute_prior_adv uses
    # np.nanmean (divides by count of present entries, not a fixed denominator), so we
    # need VARYING values: 19 sessions at 5.1e7 and one session (index -2) at 0.5e7.
    # With -2 absent (NaN): nanmean = (19*5.1e7)/19 = 5.1e7 (2% above, liquid at -1).
    # With -2 present (0.5e7): nanmean = (19*5.1e7+0.5e7)/20 = 4.87e7 (2.6% below,
    # illiquid at -1). Every other cell is unaffected because its trailing window never
    # spans index -2 at the position where a flip occurs (it only enters windows ending
    # at or after index -1, and this panel's last valid session IS -1).
    n_sessions = 22
    symbols = ("X", "Y")
    v_high = 5.1e7
    v_low = 0.5e7

    baseline_close = np.full((n_sessions, len(symbols)), 1.0)
    baseline_close[-2, 0] = np.nan
    baseline_volume = np.full((n_sessions, len(symbols)), v_high)
    baseline_volume[-2, 0] = np.nan
    baseline_volume[:, 1] = 1.0e8

    mutated_close = baseline_close.copy()
    mutated_close[-2, 0] = 1.0
    mutated_volume = baseline_volume.copy()
    mutated_volume[-2, 0] = v_low

    baseline_panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        baseline_close,
        baseline_volume,
    )
    mutated_panel = _make_panel(
        _session_dates(n_sessions),
        [1] * n_sessions,
        symbols,
        mutated_close,
        mutated_volume,
    )

    kwargs = {
        "name": "item6-pit",
        "min_history_sessions": 1,
        "min_adv_inr": 5.0e7,
    }
    baseline_result = compute_eligibility(baseline_panel, **kwargs)
    baseline_result_again = compute_eligibility(baseline_panel, **kwargs)
    mutated_result = compute_eligibility(mutated_panel, **kwargs)

    difference = baseline_result.mask != mutated_result.mask
    assert np.count_nonzero(difference) == 1
    assert bool(difference[-1, 0])
    assert not np.any(difference[:-1, :])
    assert not np.any(difference[:, 1])

    baseline_hash = baseline_result.universe_hash()
    baseline_hash_again = baseline_result_again.universe_hash()

    assert isinstance(baseline_hash, str)
    assert baseline_hash == baseline_hash_again
    assert baseline_hash != mutated_result.universe_hash()


def test_item7_survivorship_report_retains_later_listing_boundary() -> None:
    symbols = equity_symbols()
    universe = Universe(
        name="equity_symbols",
        symbols=symbols,
    )

    report = survivorship_report(
        universe,
        date(2017, 1, 1),
        date(2026, 12, 31),
    )
    warning_line = report.warning_line()

    assert (
        "Missing names are later listings; this dataset has no delistings."
        in warning_line
    )


def test_item8_count_mask_matches_continuous_coverage_control_on_gapless_fixture() -> None:
    def continuous_coverage_mask_reference(panel: Panel) -> np.ndarray:
        close = panel.field("close")
        first_day_rows = slice(int(panel.day_offsets[0]), int(panel.day_offsets[1]))
        last_day_rows = slice(int(panel.day_offsets[-2]), int(panel.day_offsets[-1]))
        first_day_ok = np.any(np.isfinite(close[first_day_rows, :]), axis=0)
        last_day_ok = np.any(np.isfinite(close[last_day_rows, :]), axis=0)
        return first_day_ok & last_day_ok

    n_sessions = 22
    symbols = ("CORE_A", "CORE_B", "LATE")

    close = np.full((n_sessions, len(symbols)), np.nan, dtype=np.float64)
    close[:, 0] = 100.0
    close[:, 1] = 100.0
    close[-1, 2] = 100.0

    volume = np.full_like(close, np.nan)
    volume[np.isfinite(close)] = 1.0e7

    panel = _make_panel(
        _session_dates(n_sessions),
        [2] * n_sessions,
        symbols,
        close,
        volume,
    )

    eligibility = compute_eligibility(
        panel,
        name="item8-pit",
        min_history_sessions=n_sessions - 1,
        min_adv_inr=5.0e7,
    )

    continuous_reference = continuous_coverage_mask_reference(panel)

    assert np.array_equal(
        continuous_reference,
        np.asarray([True, True, False], dtype=bool),
    )
    assert np.array_equal(
        eligibility.mask[-1, :],
        continuous_reference,
    )
    assert eligibility.mask[:-1, :].sum() == 0
