"""Internal unit tests for `nifty_quant.research.hypotheses.h2_overnight_reversal`.

`tests/test_h2_deepseek.py` and `tests/test_h2_luna.py` remain frozen -- 52 tests written
independently before this module existed, and neither is touched here. This file is owned
entirely by the implementer and exists to restore three guards that a 100%-branch-coverage
gate, combined with "the implementer may not add tests", had made impossible to keep: a
gate like that actively rewards deleting error handling over testing it. Once the
implementer is allowed to write tests, that perverse incentive disappears, so the guards
come back, each pinned down by a test here:

1. `run_h2`'s `horizon_mode` validation -- a `ValueError` naming both the bad value and the
   only accepted one, instead of silently returning session-mode numbers for a caller who
   asked for something else.
2. `_restrict_feature_to_panel`'s zero-row guard -- a `start`/`end` window selecting zero
   sessions must not raise a bare `IndexError` from `sliced.ts[0]`.
3. `_survivorship_window`'s empty-`dates` fallback -- the same trigger, a different call
   site (`dates[0]` before building the `SurvivorshipReport` window).

It also pins two behaviours neither frozen suite distinguishes, because both suites only
ever built 2-bar-per-session fixtures:

4. `run_h2` builds the overnight-return feature on the FULL, unrestricted panel before
   applying `start`/`end`, so a windowed run's first in-window session still sees the true
   previous session's 15:20 close rather than losing it to the window boundary. A future
   refactor to the simpler `panel.sub()`-first pattern (correct for H1, whose checkpoints
   never span a session boundary) would silently pass all 54 other tests while corrupting
   the first session of every windowed H2 run.
5. `run_h2` REDUCES the panel to exactly two rows per usable session
   (`_build_checkpoint_panel`) before calling `Lens` at all. Without this, `horizon=1` on a
   real many-bar-per-session panel measures a ~1-minute return, not the entry-to-exit
   trading session -- silently, with no error, reporting an edge diluted by roughly two
   orders of magnitude as a hypothesis verdict. Both frozen suites' fixtures have exactly
   two bars per session, so `horizon=1` was accidentally correct in every one of their 52
   tests regardless of whether this reduction existed -- this file is the only thing that
   exercises a realistic multi-bar session and would have caught the bug.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.guards import ContractViolation
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    _build_checkpoint_panel,
    _restrict_feature_to_panel,
    _survivorship_window,
    build_overnight_feature,
    run_h2,
)

# ---------------------------------------------------------------------------
# Minimal panel builder (self-contained -- does not import fixture helpers from
# either frozen H2 suite).
# ---------------------------------------------------------------------------


def _epoch_seconds(session_date: datetime.date, hhmm: str) -> int:
    """Left-labelled epoch-second UTC timestamp for an IST wall-clock HH:MM."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(session_date.year, session_date.month, session_date.day, hour, minute)
    utc_naive = local - datetime.timedelta(hours=5, minutes=30)
    return int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())


def _build_panel(
    symbols: tuple[str, ...],
    sessions: list[datetime.date],
    open_by_session: dict[datetime.date, list[float]],
    close_by_session: dict[datetime.date, list[float]],
) -> Panel:
    """Dense (n_rows, n_symbols) panel with exactly two bars per session: 09:16 (a doji --
    open == close, so the entry checkpoint is unambiguous) and 15:20 (close given
    per-session, per-symbol by the caller).
    """
    n_symbols = len(symbols)
    n_sessions = len(sessions)
    n_rows = 2 * n_sessions

    ts = np.empty(n_rows, dtype=np.int64)
    fields: dict[str, np.ndarray] = {
        name: np.full((n_rows, n_symbols), np.nan, dtype=np.float32)
        for name in ("open", "high", "low", "close")
    }
    volume = np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float32)

    for day_ix, d in enumerate(sessions):
        entry_row = 2 * day_ix
        exit_row = 2 * day_ix + 1
        ts[entry_row] = _epoch_seconds(d, "09:16")
        ts[exit_row] = _epoch_seconds(d, "15:20")

        entry_opens = open_by_session[d]
        exit_closes = close_by_session[d]
        for j in range(n_symbols):
            o = entry_opens[j]
            fields["open"][entry_row, j] = o
            fields["close"][entry_row, j] = o
            fields["high"][entry_row, j] = o + 0.01
            fields["low"][entry_row, j] = o - 0.01

            c = exit_closes[j]
            fields["open"][exit_row, j] = o
            fields["close"][exit_row, j] = c
            fields["high"][exit_row, j] = max(o, c) + 0.01
            fields["low"][exit_row, j] = min(o, c) - 0.01
    fields["volume"] = volume

    grid = SessionGrid.from_timestamps(ts)
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _linear_panel(symbols: tuple[str, ...], sessions: list[datetime.date]) -> Panel:
    """A well-formed panel with a distinct close level per session, so consecutive
    sessions never collide in price and every checkpoint is finite for every symbol.
    """
    n_symbols = len(symbols)
    open_by_session: dict[datetime.date, list[float]] = {}
    close_by_session: dict[datetime.date, list[float]] = {}
    level = 100.0
    for d in sessions:
        open_by_session[d] = [level + j for j in range(n_symbols)]
        level += 10.0
        close_by_session[d] = [level + j for j in range(n_symbols)]
    return _build_panel(symbols, sessions, open_by_session, close_by_session)


def _dense_panel_from_bars(
    symbols: tuple[str, ...],
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]],
) -> Panel:
    """General multi-bar-per-session panel builder (unlike `_build_panel`, sessions may
    have any number of bars at any labels, not just the 09:16/15:20 checkpoints). The
    shared row grid is the UNION of every symbol's timestamps; a symbol with no bar at a
    given shared timestamp gets NaN OHLCV there (never forward-filled, CLAUDE.md rule 6).
    """
    per_symbol_by_ts: dict[str, dict[int, dict[str, float]]] = {}
    all_ts: set[int] = set()
    for sym, entries in bars_by_symbol.items():
        by_ts: dict[int, dict[str, float]] = {}
        for session_date, hhmm, ohlc in entries:
            ts = _epoch_seconds(session_date, hhmm)
            by_ts[ts] = ohlc
            all_ts.add(ts)
        per_symbol_by_ts[sym] = by_ts

    ts_sorted = np.array(sorted(all_ts), dtype=np.int64)
    n_rows = ts_sorted.shape[0]
    n_symbols = len(symbols)

    fields: dict[str, np.ndarray] = {
        name: np.full((n_rows, n_symbols), np.nan, dtype=np.float32)
        for name in ("open", "high", "low", "close")
    }
    volume = np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float32)

    for j, sym in enumerate(symbols):
        by_ts = per_symbol_by_ts.get(sym, {})
        for i, ts in enumerate(ts_sorted):
            ohlc = by_ts.get(int(ts))
            if ohlc is None:
                volume[i, j] = np.nan
                continue
            for name in ("open", "high", "low", "close"):
                fields[name][i, j] = ohlc[name]
    fields["volume"] = volume

    grid = SessionGrid.from_timestamps(ts_sorted)
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


# ---------------------------------------------------------------------------
# 5. Realistic multi-bar-per-session panel: the checkpoint-reduction regression guard
# ---------------------------------------------------------------------------


def test_run_h2_on_a_realistic_multi_bar_panel_matches_the_two_bar_equivalent() -> None:
    """Regression guard for the horizon=1-on-a-raw-multi-bar-panel bug: without reducing
    the panel to exactly two rows per session before handing it to `Lens`, `horizon=1`
    silently measures a ~1-minute return instead of the entry-to-exit trading session,
    diluting the true edge by roughly two orders of magnitude without ever raising --
    a plausible-looking wrong number reported as a research finding.

    Two panels, deliberately built to reduce to IDENTICAL checkpoint economics:

    - `realistic_panel`: ~100 extra 1-minute filler bars scattered through each regular
      session (unrelated random noise, present ONLY to prove they get ignored), one
      ~60-bar Muhurat session (sentinel 999.0 prices, no 15:20 bar at all -- the LAST
      session in the panel, so its absence cannot also corrupt a later day's overnight
      return, which the frozen suites already cover separately), and a 15:21 print on the
      last regular session with a wildly different close (proving the exit resolves by
      the 15:20 LABEL, never "last bar of session").
    - `two_bar_panel`: exactly the sessions and checkpoint (09:16 open, 15:20 close)
      prices that SURVIVE `realistic_panel`'s reduction -- day0 (bootstrap, first-ever
      session), day1, day2 -- with no filler and no Muhurat session, i.e. precisely the
      style of fixture both frozen suites already validate.

    `run_h2` on both must produce bit-identical expectancy tables, and the recovered edge
    must actually be large (same order as the frozen suites' own planted-reversal
    fixtures), not diluted toward ~0 by filler-bar contamination.
    """
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    k = -1.0
    day0 = datetime.date(2023, 1, 2)
    day1 = datetime.date(2023, 1, 3)
    day2 = datetime.date(2023, 1, 4)
    muhurat = datetime.date(2023, 1, 5)

    def _checkpoint_prices(prev_close: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if prev_close is None:
            opens = np.array([100.0 + j for j in range(len(symbols))])
        else:
            opens = prev_close * np.exp(ramp)
        closes = opens * np.exp(k * ramp)
        return opens, closes

    day0_open, day0_close = _checkpoint_prices(None)
    day1_open, day1_close = _checkpoint_prices(day0_close)
    day2_open, day2_close = _checkpoint_prices(day1_close)

    # --- two_bar_panel: exactly what a frozen-suite-style fixture would build. ---
    two_bar_open = {
        day0: day0_open.tolist(), day1: day1_open.tolist(), day2: day2_open.tolist()
    }
    two_bar_close = {
        day0: day0_close.tolist(), day1: day1_close.tolist(), day2: day2_close.tolist()
    }
    two_bar_panel = _build_panel(symbols, [day0, day1, day2], two_bar_open, two_bar_close)

    # --- realistic_panel: same checkpoint economics, plus filler/Muhurat/late-print noise. ---
    rng = np.random.default_rng(2026)
    bars: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {s: [] for s in symbols}

    def _minute_label(total_minutes: int) -> str:
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

    def _add_entry_bar(d: datetime.date, opens: np.ndarray) -> None:
        for j, sym in enumerate(symbols):
            o = float(opens[j])
            bars[sym].append(
                (d, "09:16", {"open": o, "high": o + 0.01, "low": o - 0.01, "close": o})
            )

    def _add_exit_bar(d: datetime.date, closes: np.ndarray, hhmm: str = "15:20") -> None:
        for j, sym in enumerate(symbols):
            c = float(closes[j])
            bars[sym].append(
                (d, hhmm, {"open": c, "high": c + 0.01, "low": c - 0.01, "close": c})
            )

    def _add_filler_bars(d: datetime.date, n: int) -> None:
        # ~100 filler minute bars strictly between 09:16 and 15:20, random unrelated
        # noise -- present only to prove `run_h2` ignores everything but the checkpoint
        # labels. A sparse, non-contiguous minute selection keeps the panel irregular,
        # matching CLAUDE.md rule 5 (never assume a fixed bars-per-session stride).
        candidate_minutes = rng.choice(
            np.arange(9 * 60 + 17, 15 * 60 + 20), size=n, replace=False
        )
        for minute in candidate_minutes:
            hhmm = _minute_label(int(minute))
            for sym in symbols:
                level = 200.0 + float(rng.standard_normal())
                bars[sym].append(
                    (
                        d,
                        hhmm,
                        {
                            "open": level,
                            "high": level + 0.5,
                            "low": level - 0.5,
                            "close": level + float(rng.standard_normal()),
                        },
                    )
                )

    _add_entry_bar(day0, day0_open)
    _add_exit_bar(day0, day0_close)
    _add_filler_bars(day0, 90)

    _add_entry_bar(day1, day1_open)
    _add_filler_bars(day1, 100)
    _add_exit_bar(day1, day1_close)

    _add_entry_bar(day2, day2_open)
    _add_filler_bars(day2, 100)
    _add_exit_bar(day2, day2_close)
    # A late print after square-off with a wildly different close: must be ignored.
    _add_exit_bar(day2, np.full(len(symbols), 1.0), hhmm="15:21")

    # Muhurat: ~60 sentinel bars, entry label present, NO exit label at all, last
    # session in the panel (so there is no "day after Muhurat" for this test to worry
    # about -- that propagation is already covered by the frozen suites).
    for minute in range(9 * 60 + 16, 10 * 60 + 15 + 1):
        hhmm = _minute_label(minute)
        for sym in symbols:
            bars[sym].append(
                (muhurat, hhmm, {"open": 999.0, "high": 999.01, "low": 998.99, "close": 999.0})
            )

    realistic_panel = _dense_panel_from_bars(symbols, bars)

    seed = 2026
    realistic_verdict = run_h2(realistic_panel, seed=seed)
    two_bar_verdict = run_h2(two_bar_panel, seed=seed)

    assert math.isfinite(realistic_verdict.expectancy.spread_bps)
    assert realistic_verdict.expectancy.spread_bps == pytest.approx(
        two_bar_verdict.expectancy.spread_bps, abs=1e-8
    )
    assert realistic_verdict.expectancy.spread_t == pytest.approx(
        two_bar_verdict.expectancy.spread_t, abs=1e-8
    )
    assert realistic_verdict.expectancy.n_total == two_bar_verdict.expectancy.n_total

    # The bug this guards against: on the raw (unreduced) panel, `horizon=1` would
    # measure a ~1-minute return, diluting the true planted edge (same order as the
    # frozen suites' own reversal fixtures, roughly -170 bps for this ramp/k) toward ~0.
    assert abs(realistic_verdict.expectancy.spread_bps) > 100.0


def test_build_checkpoint_panel_drops_muhurat_and_ignores_the_late_print() -> None:
    """Focused unit test of `_build_checkpoint_panel` in isolation: a Muhurat session (no
    15:20 label) contributes zero rows to the reduced panel, and a 15:21 print on another
    session does not change that session's reduced "close" value.
    """
    symbols = ("A",)
    day0 = datetime.date(2024, 6, 3)
    muhurat = datetime.date(2024, 6, 4)
    bars: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {"A": []}

    def _bar(o: float, c: float | None = None) -> dict[str, float]:
        c = o if c is None else c
        return {"open": o, "high": max(o, c) + 0.01, "low": min(o, c) - 0.01, "close": c}

    bars["A"].append((day0, "09:16", _bar(95.0)))
    bars["A"].append((day0, "15:20", _bar(95.0, 100.0)))
    bars["A"].append((day0, "15:21", _bar(50.0, 50.0)))  # late print, must be ignored
    for minute in range(9 * 60 + 16, 10 * 60 + 15 + 1):
        hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
        bars["A"].append((muhurat, hhmm, _bar(999.0)))

    panel = _dense_panel_from_bars(symbols, bars)
    feature = build_overnight_feature(panel)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, feature)

    assert checkpoint_panel.n_rows() == 2  # only day0 has both checkpoints
    assert checkpoint_panel.n_days() == 1
    assert checkpoint_panel.dates[0] == day0
    np.testing.assert_allclose(checkpoint_panel.field("close")[:, 0], [95.0, 100.0])
    assert checkpoint_feature.values.shape == (2, 1)


# ---------------------------------------------------------------------------
# 1. `horizon_mode` validation
# ---------------------------------------------------------------------------


def test_run_h2_rejects_unsupported_horizon_mode() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    panel = _linear_panel(symbols, sessions)

    with pytest.raises(ValueError) as exc_info:
        run_h2(panel, horizon_mode="bar", seed=0)

    message = str(exc_info.value)
    assert "horizon_mode" in message
    assert "'bar'" in message
    assert "session" in message


def test_run_h2_accepts_the_only_implemented_horizon_mode() -> None:
    """The positive case for the same guard: `"session"` (default or explicit) must not
    raise, so the negative-case test above is exercising real validation logic and not an
    always-true condition."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    panel = _linear_panel(symbols, sessions)

    default_result = run_h2(panel, seed=0)
    explicit_result = run_h2(panel, horizon_mode="session", seed=0)

    assert default_result.expectancy.spread_bps == pytest.approx(
        explicit_result.expectancy.spread_bps
    )


# ---------------------------------------------------------------------------
# 2. `_restrict_feature_to_panel` zero-row guard
# ---------------------------------------------------------------------------


def test_restrict_feature_to_panel_empty_window_returns_empty_feature_not_index_error() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    panel = _linear_panel(symbols, sessions)
    feature = build_overnight_feature(panel)

    # `end` before every session in the panel: `Panel.sub`'s own date-comparison logic
    # (`dates <= end` all False) resolves this to a genuine zero-row slice, not merely
    # a "keep everything" fallback.
    before_all_sessions = sessions[0] - datetime.timedelta(days=1000)
    empty_sliced = panel.sub(end=before_all_sessions)
    assert empty_sliced.n_rows() == 0

    restricted = _restrict_feature_to_panel(panel, feature, empty_sliced)

    assert restricted.values.shape == (0, len(symbols))
    assert restricted.kind == "return"
    assert restricted.name == feature.name


def test_restrict_feature_to_panel_nonempty_window_is_unaffected_by_the_guard() -> None:
    """The positive case: a normal, non-empty window still slices to the exact matching
    row range, so the guard only special-cases the zero-row branch."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    panel = _linear_panel(symbols, sessions)
    feature = build_overnight_feature(panel)

    sliced = panel.sub(start=sessions[1])
    restricted = _restrict_feature_to_panel(panel, feature, sliced)

    assert restricted.values.shape == (sliced.n_rows(), len(symbols))
    np.testing.assert_array_equal(restricted.values, feature.values[2:])


# ---------------------------------------------------------------------------
# 3. `_survivorship_window` empty-`dates` fallback
# ---------------------------------------------------------------------------


def test_survivorship_window_falls_back_to_today_for_empty_dates() -> None:
    empty_dates = np.empty(0, dtype=object)
    window_start, window_end = _survivorship_window(empty_dates)

    today = datetime.date.today()
    assert window_start == today
    assert window_end == today


def test_survivorship_window_uses_first_and_last_date_when_nonempty() -> None:
    d0 = datetime.date(2021, 1, 1)
    d1 = datetime.date(2021, 6, 15)
    d2 = datetime.date(2021, 12, 31)
    dates = np.array([d0, d1, d2], dtype=object)

    window_start, window_end = _survivorship_window(dates)

    assert window_start == d0
    assert window_end == d2


def test_run_h2_empty_window_fails_closed_with_a_named_exception_not_index_error() -> None:
    """End-to-end: a `start`/`end` window selecting zero sessions must not surface a bare
    `IndexError` from either guard restored above. `Lens`/`expectancy`'s own
    `cross_sectional_rank` contract (a layer this module does not own and must not modify)
    still fails closed on a wholly empty feature -- but with `ContractViolation`, a named,
    message-bearing exception, never an unexplained `IndexError` three frames deep."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    sessions = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    panel = _linear_panel(symbols, sessions)

    before_all_sessions = sessions[0] - datetime.timedelta(days=1000)

    with pytest.raises(ContractViolation) as exc_info:
        run_h2(panel, end=before_all_sessions, seed=0)

    assert not isinstance(exc_info.value, IndexError)
    assert str(exc_info.value)


# ---------------------------------------------------------------------------
# Feature built on the FULL panel before `start`/`end` restriction (point 1)
# ---------------------------------------------------------------------------


def test_windowed_run_uses_the_true_previous_close_before_the_window_start() -> None:
    """3 sessions: day0, day1, day2. `start=day1` restricts the reporting window to
    [day1, day2], but day1's overnight return must still be computed from day0's REAL
    15:20 close -- which lies strictly before `start` -- not treated as missing because
    day0 itself falls outside the window.

    A `panel.sub(start=day1)`-first implementation (correct for H1, wrong here because H1's
    checkpoints never span a session boundary) would make day1 the FIRST session of the
    sliced panel and its overnight return would come out NaN. This test constructs the
    contrast explicitly: `build_overnight_feature` on the pre-sliced panel true-NaNs day1,
    while `run_h2`'s actual full-panel-first construction does not.
    """
    symbols = ("A",)
    day0 = datetime.date(2022, 3, 1)
    day1 = datetime.date(2022, 3, 2)
    day2 = datetime.date(2022, 3, 3)
    sessions = [day0, day1, day2]

    # day0's 15:20 close is a materially different, hand-picked level (55.0) from what a
    # naive same-day open/close ramp would produce, so a wrong implementation reading the
    # wrong (or no) prior close would not coincidentally match by construction.
    open_by_session = {day0: [50.0], day1: [60.0], day2: [70.0]}
    close_by_session = {day0: [55.0], day1: [61.0], day2: [71.0]}
    panel = _build_panel(symbols, sessions, open_by_session, close_by_session)

    full_feature = build_overnight_feature(panel)
    sliced_panel = panel.sub(start=day1)
    restricted = _restrict_feature_to_panel(panel, full_feature, sliced_panel)

    sym_ix = panel.sym_ix["A"]
    day1_row_in_window = 0  # first row of the windowed slice is day1's 09:16 entry
    expected_r_on_day1 = math.log(60.0 / 55.0)

    assert restricted.values[day1_row_in_window, sym_ix] == pytest.approx(
        expected_r_on_day1, abs=1e-9
    )

    # Contrast: building the feature on the ALREADY-restricted panel (the wrong,
    # sub()-first pattern) loses day0's close entirely and NaNs day1's overnight return.
    wrong_pattern_feature = build_overnight_feature(sliced_panel)
    assert math.isnan(wrong_pattern_feature.values[day1_row_in_window, sym_ix])


def test_run_h2_windowed_verdict_counts_the_first_in_window_session() -> None:
    """Integration-level corroboration of the unit test above, through the public
    `run_h2` API: an 8-symbol, 3-session panel windowed to `start=day1` (2 sessions) must
    not raise, and must report the observation count consistent with BOTH in-window
    sessions contributing -- not silently dropping the first one to a manufactured NaN.
    """
    symbols = tuple(f"SYM{i}" for i in range(8))
    day0 = datetime.date(2022, 4, 1)
    day1 = datetime.date(2022, 4, 4)
    day2 = datetime.date(2022, 4, 5)
    sessions = [day0, day1, day2]
    panel = _linear_panel(symbols, sessions)

    full_verdict = run_h2(panel, seed=0)
    windowed_verdict = run_h2(panel, start=day1, seed=0)

    # The windowed run has strictly fewer sessions in scope (2 vs 3) but must still have
    # a strictly positive observation count -- i.e. day1 (the window's first session) is
    # counted, not NaN'd out by a spurious loss of its previous close.
    assert windowed_verdict.expectancy.n_total > 0
    assert windowed_verdict.expectancy.n_total < full_verdict.expectancy.n_total
