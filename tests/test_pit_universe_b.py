"""SPEC DEFECTS / AMBIGUITIES FOUND

1. Entry-point naming: The spec does not name the entry point; the assumed API contract invents
   `compute_eligibility` in `nifty_quant/universe/pit.py`. This is a genuine ambiguity in the
   spec text itself.

2. Current-session presence requirement: The spec says eligibility requires "at least
   `min_history_sessions` PRIOR sessions with a present bar" but does not explicitly state
   whether the symbol must ALSO be present on session `d` itself. The phrase "using only data
   strictly before `d`" suggests the current session's presence is not part of the eligibility
   computation, but a symbol absent on session `d` would still be marked eligible if it has
   sufficient history. This is ambiguous.

3. "compute_prior_adv-style reuse": The spec mentions this phrase but no such named function
   exists in the codebase (or at least is not referenced elsewhere in the spec). It is unclear
   whether this refers to an existing internal helper or is a suggestion for implementation.

4. "min_history_sessions value that spans the whole window": The spec says "largest value for
   which any session can still be eligible within a window of `n_sessions` sessions (i.e.
   `n_sessions - 1`)" but this is only true if sessions are indexed 0..n-1 and the first
   eligible session is index `min_history_sessions`. The phrasing is confusing because it
   conflates "number of prior sessions required" with "session index at which eligibility first
   becomes possible."

5. Liquidity threshold semantics: The spec says "trailing ADV (20-session trailing mean of
   sum(close*volume) per session, current session excluded)" but does not specify whether the
   20-session window is a rolling window of the most recent 20 sessions (including sessions
   where the symbol was absent) or the most recent 20 sessions where the symbol was present.
   This affects the ADV computation for symbols with gaps.

6. Hash stability across column order: The spec requires the hash to be invariant to symbol
   column order, but the `PointInTimeEligibility` dataclass stores `symbols` as a tuple in a
   specific order. The hash must therefore be computed from a sorted representation, but the
   spec does not state whether the `symbols` field itself should be sorted or preserve input
   order.

7. Interaction with `Universe.as_of`: The spec references `Universe.as_of(d)` and its
   `source="availability_proxy"` but does not specify whether `compute_eligibility` should
   internally call `Universe.as_of` or receive a `Universe` object directly. The assumed API
   contract passes a `Universe` object, but the relationship to `as_of` is unclear.

8. Delisting boundary wording: The spec says "zero delisted names (every symbol in the manifest
   runs to the end of the window)" but then says the docstring must state "the delisting
   boundary explicitly." It is unclear what "delisting boundary" means when there are no
   delistings in the data.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.universe.static import Universe, survivorship_report


def _session_ts(
    session_date: datetime.date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(
        session_date, datetime.time(start_hhmm[0], start_hhmm[1]), tzinfo=ist
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64
    )


def _build_panel(
    session_dates: list[datetime.date],
    bars_per_session: list[int],
    symbols: tuple[str, ...],
    *,
    present_from_session: dict[str, int] | None = None,
    absent_sessions: dict[str, set[int]] | None = None,
    close_price: float = 100.0,
    volume_value: float = 1_000_000.0,
) -> Panel:
    """Build a synthetic Panel. Symbols are present (non-NaN close/volume) for every bar of a
    session unless that session index is before `present_from_session[symbol]` or is listed in
    `absent_sessions[symbol]`, in which case ALL bars of that symbol for that session are NaN."""
    present_from_session = present_from_session or {}
    absent_sessions = absent_sessions or {}

    n_sessions = len(session_dates)
    n_symbols = len(symbols)

    # Concatenate timestamps across sessions
    all_ts = []
    for i, n_bars in enumerate(bars_per_session):
        all_ts.extend(_session_ts(session_dates[i], n_bars))
    ts = np.asarray(all_ts, dtype=np.int64)

    # Build day_offsets via cumsum of bars_per_session prefixed with 0
    day_offsets = np.cumsum([0] + bars_per_session, dtype=np.int64)

    n_rows = len(ts)

    # Build field arrays
    close = np.full((n_rows, n_symbols), close_price, dtype=np.float64)
    open_ = np.full((n_rows, n_symbols), close_price, dtype=np.float64)
    high = np.full((n_rows, n_symbols), close_price, dtype=np.float64)
    low = np.full((n_rows, n_symbols), close_price, dtype=np.float64)
    volume = np.full((n_rows, n_symbols), volume_value, dtype=np.float64)

    # Apply presence rules
    for col_idx, symbol in enumerate(symbols):
        first_session = present_from_session.get(symbol, 0)
        extra_absent = absent_sessions.get(symbol, set())

        for session_idx in range(n_sessions):
            if session_idx < first_session or session_idx in extra_absent:
                start_row = day_offsets[session_idx]
                end_row = day_offsets[session_idx + 1]
                close[start_row:end_row, col_idx] = np.nan
                open_[start_row:end_row, col_idx] = np.nan
                high[start_row:end_row, col_idx] = np.nan
                low[start_row:end_row, col_idx] = np.nan
                volume[start_row:end_row, col_idx] = np.nan

    fields = {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
    }

    return Panel(
        fields,
        symbols,
        ts,
        day_offsets,
        np.asarray(session_dates, dtype=object),
    )


def _rank_values(values: np.ndarray) -> np.ndarray:
    """Manual rank helper: rank 1 is highest value. Ties broken by first occurrence."""
    return np.argsort(np.argsort(-values)) + 1


@pytest.fixture
def sample_panel() -> Panel:
    """A simple 10-session panel with 3 symbols, all present from session 0."""
    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(10)
    ]
    bars_per_session = [375] * 10
    symbols = ("AAA", "BBB", "CCC")
    return _build_panel(session_dates, bars_per_session, symbols)


@pytest.fixture
def sample_universe() -> Universe:
    return Universe(name="test_universe", symbols=("AAA", "BBB", "CCC"))


# 1. A name whose first present bar is mid-window is INELIGIBLE before that date and eligible only
#    `min_history_sessions` sessions after it.
def test_mid_window_listing_eligibility_boundary() -> None:
    from nifty_quant.universe.pit import compute_eligibility

    n_sessions = 20
    min_history = 5
    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_sessions)
    ]
    bars_per_session = [375] * n_sessions
    symbols = ("EARLY", "LATE")

    # LATE first present at session 7
    panel = _build_panel(
        session_dates,
        bars_per_session,
        symbols,
        present_from_session={"EARLY": 0, "LATE": 7},
    )
    universe = Universe(name="test", symbols=symbols)

    result = compute_eligibility(
        panel, universe, min_history_sessions=min_history, min_adv_inr=0.0
    )

    # LATE should be ineligible before session 7 + min_history = 12
    for session_idx in range(12):
        late_col = symbols.index("LATE")
        assert not result.mask[session_idx, late_col], (
            f"LATE should be ineligible at session {session_idx}"
        )

    # LATE should be eligible at session 12 (7 + 5 prior sessions: 7,8,9,10,11)
    late_col = symbols.index("LATE")
    assert result.mask[12, late_col], "LATE should be eligible at session 12"

    # EARLY should be eligible from session min_history onward
    early_col = symbols.index("EARLY")
    for session_idx in range(min_history):
        assert not result.mask[session_idx, early_col], (
            f"EARLY should be ineligible at session {session_idx}"
        )
    for session_idx in range(min_history, n_sessions):
        assert result.mask[session_idx, early_col], (
            f"EARLY should be eligible at session {session_idx}"
        )


# 2. Eligibility on session `d` is unchanged by mutating any data at or after `d` (causality /
#    no lookahead).
def test_causality_no_lookahead() -> None:
    from nifty_quant.universe.pit import compute_eligibility

    n_sessions = 15
    min_history = 3
    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_sessions)
    ]
    bars_per_session = [375] * n_sessions
    symbols = ("AAA", "BBB", "CCC")
    universe = Universe(name="test", symbols=symbols)

    # Panel A: all present from session 0
    panel_a = _build_panel(session_dates, bars_per_session, symbols)
    result_a = compute_eligibility(
        panel_a, universe, min_history_sessions=min_history, min_adv_inr=0.0
    )

    # Panel B: identical to A for every bar strictly before `cutoff_session`; mutated at and
    # after it -- AAA's presence is removed and BBB's volume is zeroed from that point on.
    # `Panel.field(name)` returns the live backing array (not a copy), so mutating it in place
    # after construction is the intended way to get two panels that are byte-identical up to a
    # cutoff and diverge only after it.
    cutoff_session = 8
    panel_b = _build_panel(session_dates, bars_per_session, symbols)
    aaa_col = symbols.index("AAA")
    bbb_col = symbols.index("BBB")
    for session_idx in range(cutoff_session, n_sessions):
        start_row = int(panel_b.day_offsets[session_idx])
        end_row = int(panel_b.day_offsets[session_idx + 1])
        panel_b.field("close")[start_row:end_row, aaa_col] = np.nan
        panel_b.field("open")[start_row:end_row, aaa_col] = np.nan
        panel_b.field("high")[start_row:end_row, aaa_col] = np.nan
        panel_b.field("low")[start_row:end_row, aaa_col] = np.nan
        panel_b.field("volume")[start_row:end_row, aaa_col] = np.nan
        panel_b.field("volume")[start_row:end_row, bbb_col] = 0.0

    result_b = compute_eligibility(
        panel_b, universe, min_history_sessions=min_history, min_adv_inr=0.0
    )

    # Assert masks agree on all sessions strictly before cutoff
    for session_idx in range(cutoff_session):
        np.testing.assert_array_equal(
            result_a.mask[session_idx],
            result_b.mask[session_idx],
            err_msg=f"Masks differ at session {session_idx} (before cutoff)",
        )


# 3. A name that fails the trailing-liquidity test on session `d` is excluded from that session's
#    cross-section such that the remaining names' ranks are computed as if it were absent -- not
#    as if it had zero weight. "Zero weight" means the ineligible name's own factor VALUE is
#    still fed into the ranking population and only its downstream portfolio weight is forced to
#    zero afterward -- that still changes N (and therefore the rank) for every name whose true
#    value sits on the other side of the ineligible name's true value. Exclusion (dropping the
#    row before ranking, as the eligibility mask must do) does not have that effect. This must
#    be demonstrated with a case where the ineligible name's value is BETWEEN two eligible names'
#    values, otherwise dropping vs zero-weighting-only coincidentally produce the same order.
def test_ineligible_symbol_excluded_from_ranking() -> None:
    # One session's cross-section: AAA=30, BBB=20, CCC=10. BBB fails the trailing-liquidity
    # test -- and BBB's true value (20) sits strictly between AAA's and CCC's.
    values = np.array([30.0, 20.0, 10.0])  # AAA, BBB, CCC
    eligible = np.array([True, False, True])  # this session's eligibility-mask row

    # Correct: BBB is dropped from the cross-section entirely before ranking (as if absent).
    dropped_ranks = _rank_values(values[eligible])  # ranks over [AAA=30, CCC=10]
    assert dropped_ranks.tolist() == [1, 2], "AAA=1st, CCC=2nd of 2 when BBB is dropped"

    # Wrong: BBB stays in the ranking population with its TRUE value (only its portfolio
    # weight would be zeroed afterward, which does not change N or the ranking computation).
    kept_ranks_all = _rank_values(values)  # ranks over [AAA=30, BBB=20, CCC=10]
    kept_ranks_for_eligible = kept_ranks_all[eligible]  # AAA, CCC ranks with BBB still in N
    assert kept_ranks_for_eligible.tolist() == [1, 3], (
        "AAA=1st, CCC=3rd of 3 when BBB (mid-value) is merely zero-weighted, not dropped"
    )

    # Same two names, same true relative order -- but CCC's rank is 2 (of 2) when the
    # ineligible name is properly excluded, and 3 (of 3) when it is merely zero-weighted while
    # still occupying a slot in the cross-section. This is exactly the distinction the spec
    # draws between "excluded ... entirely" and "given zero weight", and it is why the
    # eligibility mask must be applied by dropping rows before any rank/z-score computation,
    # not by post-hoc zeroing a computed weight.
    assert not np.array_equal(dropped_ranks, kept_ranks_for_eligible)


# 4. The eligibility mask has shape `(n_sessions, n_symbols)` and no fixed session-length
#    assumption.
def test_mask_shape_with_mixed_session_lengths() -> None:
    from nifty_quant.universe.pit import compute_eligibility

    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(4)
    ]
    bars_per_session = [375, 60, 105, 375]
    symbols = ("AAA", "BBB", "CCC")

    panel = _build_panel(session_dates, bars_per_session, symbols)
    universe = Universe(name="test", symbols=symbols)

    result = compute_eligibility(
        panel, universe, min_history_sessions=1, min_adv_inr=0.0
    )

    assert result.mask.shape == (4, 3), (
        f"Expected mask shape (4, 3), got {result.mask.shape}"
    )


# 5. `source` is `"availability_proxy"` on all current data; constructing with
#    `source="index_membership"` explicitly requested must raise `ValueError`.
def test_source_field_and_index_membership_raises(
    sample_panel: Panel, sample_universe: Universe
) -> None:
    from nifty_quant.universe.pit import compute_eligibility

    result = compute_eligibility(
        sample_panel, sample_universe, min_history_sessions=1, min_adv_inr=0.0
    )
    assert result.source == "availability_proxy", (
        f"Expected source='availability_proxy', got '{result.source}'"
    )

    with pytest.raises(ValueError, match="index_membership"):
        compute_eligibility(
            sample_panel,
            sample_universe,
            min_history_sessions=1,
            min_adv_inr=0.0,
            source="index_membership",
        )


# 6. `universe_hash` is stable across two independent calls with identical panel/params, and
#    changes when the eligible set changes on any single session. Also invariant to symbol
#    column order.
def test_universe_hash_stability_and_sensitivity(
    sample_panel: Panel, sample_universe: Universe
) -> None:
    from nifty_quant.universe.pit import compute_eligibility

    result1 = compute_eligibility(
        sample_panel, sample_universe, min_history_sessions=1, min_adv_inr=0.0
    )
    result2 = compute_eligibility(
        sample_panel, sample_universe, min_history_sessions=1, min_adv_inr=0.0
    )

    assert result1.universe_hash == result2.universe_hash, (
        "Hash should be stable across identical calls"
    )

    # Change the eligible set on (at least) one session via the LIQUID condition, not the
    # LISTED/presence condition -- deliberately avoids depending on whether current-session
    # presence is itself required for eligibility (the spec is ambiguous about that, see
    # defect #2 above), by only ever touching trailing history, never session `d` itself.
    #
    # All sessions have day_value = close * volume * n_bars = 100 * 1e6 * 375 = 3.75e10 in the
    # baseline. Dropping AAA's volume to ~0 for exactly session index 3 drags its expanding-
    # window trailing ADV (mean of all prior day_values, current session excluded) below
    # 3.3e10 for several sessions starting at session 4, while the unmodified panel's ADV for
    # AAA stays at 3.75e10 (>= 3.3e10) throughout. min_history_sessions=1 keeps the LISTED
    # condition satisfied from session 1 onward for both panels, isolating the LIQUID effect.
    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(10)
    ]
    bars_per_session = [375] * 10
    symbols = ("AAA", "BBB", "CCC")
    min_adv_threshold = 3.3e10

    panel_orig = _build_panel(session_dates, bars_per_session, symbols)
    result_orig = compute_eligibility(
        panel_orig,
        sample_universe,
        min_history_sessions=1,
        min_adv_inr=min_adv_threshold,
    )

    panel_modified = _build_panel(session_dates, bars_per_session, symbols)
    aaa_col = symbols.index("AAA")
    bad_start = int(panel_modified.day_offsets[3])
    bad_end = int(panel_modified.day_offsets[4])
    panel_modified.field("volume")[bad_start:bad_end, aaa_col] = 1.0

    result3 = compute_eligibility(
        panel_modified,
        sample_universe,
        min_history_sessions=1,
        min_adv_inr=min_adv_threshold,
    )

    assert not np.array_equal(result_orig.mask, result3.mask), (
        "Expected the eligibility mask to actually change after dropping one session's volume"
    )
    assert result_orig.universe_hash != result3.universe_hash, (
        "Hash should change when the eligible set changes on any session"
    )

    # Test invariance to symbol column order
    symbols_reordered = ("CCC", "AAA", "BBB")
    panel_reordered = _build_panel(session_dates, bars_per_session, symbols_reordered)
    universe_reordered = Universe(name="test_universe", symbols=symbols_reordered)

    result4 = compute_eligibility(
        panel_reordered, universe_reordered, min_history_sessions=1, min_adv_inr=0.0
    )

    assert result1.universe_hash == result4.universe_hash, (
        "Hash should be invariant to symbol column order"
    )


# 7. Survivorship reporting still fires and still communicates its boundary.
def test_survivorship_report_boundary_wording(monkeypatch: pytest.MonkeyPatch) -> None:
    import nifty_quant.universe.static as static_module

    # Clear lru_cache before monkeypatching
    static_module._get_manifest.cache_clear()

    synthetic_manifest = {
        "coverage": {
            "OLD": {"years": [2020, 2021, 2022]},
            "NEW": {"years": [2022]},
        }
    }

    def mock_get_manifest():
        return synthetic_manifest

    # `monkeypatch.setattr` restores the original (lru_cache-wrapped) `_get_manifest` on test
    # teardown automatically -- do not call `.cache_clear()` on it while still patched, since
    # the patched-in plain function has no `.cache_clear` attribute at all.
    monkeypatch.setattr(static_module, "_get_manifest", mock_get_manifest)

    universe = Universe(name="test", symbols=("OLD", "NEW"))
    report = survivorship_report(
        universe, datetime.date(2020, 1, 1), datetime.date(2022, 12, 31)
    )
    warning = report.warning_line()
    assert "Missing names are later listings; this dataset has no delistings." in warning, (
        f"Warning line does not contain expected boundary wording: {warning}"
    )


# 8. Setting `min_history_sessions` to `n_sessions - 1` collapses to continuous coverage.
def test_min_history_sessions_collapses_to_continuous_coverage() -> None:
    from nifty_quant.universe.pit import compute_eligibility

    n_sessions = 20
    min_history = n_sessions - 1  # 19
    session_dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_sessions)
    ]
    bars_per_session = [375] * n_sessions
    symbols = ("OLD", "LATE")

    # OLD present from session 0, LATE present from session 5
    panel = _build_panel(
        session_dates,
        bars_per_session,
        symbols,
        present_from_session={"OLD": 0, "LATE": 5},
    )
    universe = Universe(name="test", symbols=symbols)

    result = compute_eligibility(
        panel, universe, min_history_sessions=min_history, min_adv_inr=0.0
    )

    old_col = symbols.index("OLD")
    late_col = symbols.index("LATE")

    # (a) OLD is eligible on session 19
    assert result.mask[19, old_col], "OLD should be eligible on session 19"

    # (b) LATE is NEVER eligible on any session
    assert not np.any(result.mask[:, late_col]), (
        "LATE should never be eligible with min_history_sessions=19"
    )

    # (c) The set of symbols EVER eligible equals exactly {"OLD"}
    ever_eligible = set()
    for session_idx in range(n_sessions):
        for col_idx, symbol in enumerate(symbols):
            if result.mask[session_idx, col_idx]:
                ever_eligible.add(symbol)

    assert ever_eligible == {"OLD"}, (
        f"Expected ever-eligible set to be {{'OLD'}}, got {ever_eligible}"
    )
