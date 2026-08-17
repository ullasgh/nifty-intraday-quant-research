"""TDD red-state suite (deepseek variant) for `tradable_overnight_return`.

Independent of `tests/test_tradable_overnight_luna.py` — written from
`specs/tradable_overnight_return.md` alone, without reading the sibling suite.

`tradable_overnight_return` does not exist in `src/nifty_quant/features/market.py` yet; the
import below fails at collection time (ImportError), which is the expected red state for this
whole file until the implementation lands.

Minute-of-day labels used throughout (IST minutes since midnight):
    09:15 -> 555   09:16 -> 556   09:20 -> 560
    15:00 -> 900   15:20 -> 920   15:29 -> 929
"""

from __future__ import annotations

import numpy as np

from nifty_quant.features.market import overnight_return, tradable_overnight_return
from nifty_quant.guards import Strictness, strictness


def _two_session_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2 sessions x 3 rows, labels 09:15/09:16/15:20 in each session.

    Session 0 (rows 0-2) is the first session -> always NaN (no prior).
    Session 1 (rows 3-5) draws its entry (09:16) from row 4 and its exit (15:20) from
    session 0's row 2.
    """
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 555, 556, 920], dtype=np.int16)
    open_ = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float32)
    close = np.array([13.0, 14.0, 15.0, 23.0, 24.0, 25.0], dtype=np.float32)
    return day_offsets, minute_of_day, open_, close


def test_hand_computed_two_session_panel() -> None:
    day_offsets, minute_of_day, open_, close = _two_session_fixture()

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.isnan(result[0:3]).all()
    expected = np.log(np.float64(open_[4]) / np.float64(close[2]))
    np.testing.assert_allclose(result[3:6], expected, rtol=1e-9)


def test_differs_from_overnight_return_on_realistic_bars() -> None:
    """MOST IMPORTANT TEST — regression test for the whole defect.

    Session 0 has distinct 09:15/09:16/.../15:20/15:29 closes, so the label-resolved
    exit (15:20) and the positional exit (last row = 15:29) diverge materially.
    Session 1 has a distinct 09:15 open/close vs 09:16 open, so the label-resolved
    entry (09:16 open) and the positional entry (first row = 09:15 close) also diverge.
    `tradable_overnight_return` and `overnight_return` MUST disagree on session 1.
    """
    day_offsets = np.array([0, 6, 9], dtype=np.int32)
    minute_of_day = np.array(
        [555, 556, 557, 558, 920, 929, 555, 556, 557], dtype=np.int16
    )
    open_ = np.array(
        [100.0, 100.5, 101.0, 101.5, 102.0, 103.0, 199.0, 201.0, 202.0],
        dtype=np.float32,
    )
    close = np.array(
        [100.0, 100.5, 101.0, 101.5, 102.0, 105.0, 199.0, 199.5, 200.0],
        dtype=np.float32,
    )

    tradable = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    legacy = overnight_return(close, day_offsets)

    # Label-resolved: entry = 09:16 open of session 1 (row 7), exit = 15:20 close of
    # session 0 (row 4).
    expected_tradable = np.log(np.float64(open_[7]) / np.float64(close[4]))
    # Positional (existing behaviour): first row of session 1 (row 6, label 09:15) over
    # last row of session 0 (row 5, label 15:29).
    expected_legacy = np.log(np.float64(close[6]) / np.float64(close[5]))

    assert abs(expected_tradable - expected_legacy) > 1e-3, "fixture failed to separate"

    np.testing.assert_allclose(tradable[6:9], expected_tradable, rtol=1e-9)
    np.testing.assert_allclose(legacy[6:9], expected_legacy, rtol=1e-9)
    assert not np.allclose(tradable[6:9], legacy[6:9])


def test_ignores_the_0915_bar_entirely() -> None:
    """Corrupting the 09:15 close (the real observed `close > high` defect) must not
    move the result at all: entry is 09:16, exit is the PRIOR session's 15:20, neither
    of which is the 09:15 bar."""
    day_offsets, minute_of_day, open_, close = _two_session_fixture()
    baseline = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    corrupted_close = close.copy()
    corrupted_close[3] = np.float32(1e6)  # 09:15 row of session 1 -- absurd outlier
    corrupted = tradable_overnight_return(open_, corrupted_close, day_offsets, minute_of_day)

    assert np.array_equal(baseline, corrupted, equal_nan=True)


def test_ignores_bars_after_exit_hhmm() -> None:
    """A 15:29 bar (the session's true final print) with a wildly different close must
    not affect the result: exit_hhmm=15:20 is used, not the last bar of the session."""
    day_offsets = np.array([0, 4, 7], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 929, 555, 556, 920], dtype=np.int16)
    open_ = np.array([10.0, 11.0, 12.0, 12.5, 20.0, 21.0, 22.0], dtype=np.float32)
    close_baseline = np.array(
        [13.0, 14.0, 15.0, 15.5, 23.0, 24.0, 25.0], dtype=np.float32
    )
    close_outlier = close_baseline.copy()
    close_outlier[3] = np.float32(999.0)  # 15:29 row of session 0 -- absurd outlier

    baseline = tradable_overnight_return(open_, close_baseline, day_offsets, minute_of_day)
    outlier = tradable_overnight_return(open_, close_outlier, day_offsets, minute_of_day)

    assert np.array_equal(baseline, outlier, equal_nan=True)
    # Sanity: the fixture is actually exercising a finite session, not two all-NaN arrays.
    assert np.isfinite(baseline[4:7]).all()


def test_custom_labels_honoured() -> None:
    day_offsets = np.array([0, 4, 7], dtype=np.int32)
    minute_of_day = np.array([555, 556, 560, 900, 555, 556, 560], dtype=np.int16)
    open_ = np.array([10.0, 11.0, 12.0, 12.5, 20.0, 21.0, 22.0], dtype=np.float32)
    close = np.array([13.0, 14.0, 15.0, 16.0, 23.0, 24.0, 25.0], dtype=np.float32)

    result = tradable_overnight_return(
        open_, close, day_offsets, minute_of_day, entry_hhmm="09:20", exit_hhmm="15:00"
    )

    # entry = 09:20 open of session 1 (row 6), exit = 15:00 close of session 0 (row 3).
    expected = np.log(np.float64(open_[6]) / np.float64(close[3]))
    np.testing.assert_allclose(result[4:7], expected, rtol=1e-9)


def test_session_missing_entry_label_is_nan_no_reach_back() -> None:
    """Session 1 has no 09:16 bar at all (labels skip 555 -> 557). Its own result must
    be entirely NaN -- it must not reach back to an earlier/other session's entry."""
    day_offsets = np.array([0, 3, 6, 9], dtype=np.int32)
    minute_of_day = np.array(
        [555, 556, 920, 555, 557, 920, 555, 556, 920], dtype=np.int16
    )
    open_ = np.array(
        [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0], dtype=np.float32
    )
    close = np.array(
        [13.0, 14.0, 15.0, 23.0, 24.0, 25.0, 33.0, 34.0, 35.0], dtype=np.float32
    )

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.isnan(result[3:6]).all()
    assert np.isnan(result[3:6]).sum() == 3  # counted: the whole session, not a partial hit


def test_muhurat_session_makes_next_session_nan_not_bridged_to_older_close() -> None:
    """Session 1 is a ~60-row Muhurat-shaped session with no 15:20 bar. Session 2's
    result must be entirely NaN -- it must NOT silently bridge back to session 0's
    15:20 close."""
    n_muhurat = 60
    day_offsets = np.array([0, 3, 3 + n_muhurat, 3 + n_muhurat + 3], dtype=np.int32)

    session0_minutes = np.array([555, 556, 920], dtype=np.int16)
    session1_minutes = (555 + np.arange(n_muhurat)).astype(np.int16)  # 09:15..10:14, no 15:20
    session2_minutes = np.array([555, 556, 920], dtype=np.int16)
    minute_of_day = np.concatenate([session0_minutes, session1_minutes, session2_minutes])

    rng = np.random.default_rng(2024)
    n_rows = int(day_offsets[-1])
    open_ = rng.uniform(50.0, 150.0, size=n_rows).astype(np.float32)
    close = rng.uniform(50.0, 150.0, size=n_rows).astype(np.float32)

    close0_1520 = np.float64(close[2])  # session 0's 15:20 close

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    session2_start = int(day_offsets[2])
    session2_end = int(day_offsets[3])
    entry_row = session2_start + 1  # 09:16 is the 2nd row of session 2
    assert np.isnan(result[session2_start:session2_end]).all()

    illegal_bridge = np.log(np.float64(open_[entry_row]) / close0_1520)
    assert not np.isclose(result[session2_start], illegal_bridge, equal_nan=True)


def test_per_symbol_independence() -> None:
    """2 symbols, same session: A has both endpoints, B is missing its entry open.
    A must be finite, B must be NaN, in the SAME session."""
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 555, 556, 920], dtype=np.int16)
    open_a = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float32)
    open_b = np.array([10.0, 11.0, 12.0, 20.0, np.nan, 22.0], dtype=np.float32)
    open_ = np.stack([open_a, open_b], axis=1)
    close_col = np.array([13.0, 14.0, 15.0, 23.0, 24.0, 25.0], dtype=np.float32)
    close = np.stack([close_col, close_col], axis=1)

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    expected_a = np.log(np.float64(open_a[4]) / np.float64(close_col[2]))
    np.testing.assert_allclose(result[3:6, 0], expected_a, rtol=1e-9)
    assert np.isnan(result[3:6, 1]).all()


def test_first_session_is_nan() -> None:
    day_offsets, minute_of_day, open_, close = _two_session_fixture()
    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    assert np.isnan(result[0:3]).all()


def test_broadcast_to_every_row_of_its_session() -> None:
    day_offsets, minute_of_day, open_, close = _two_session_fixture()
    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    session1 = result[3:6]
    assert np.isfinite(session1).all()
    assert len(np.unique(session1)) == 1
    expected = np.log(np.float64(open_[4]) / np.float64(close[2]))
    np.testing.assert_allclose(session1, expected, rtol=1e-9)


def test_nan_input_propagates_without_forward_fill() -> None:
    """Session 1's OWN 15:20 close is NaN. Session 1 (which depends on session 0's
    15:20, which IS finite) must remain finite. Session 2 (which depends on session
    1's 15:20) must be NaN. The NaN must not "forward-fill" backward into session 1."""
    day_offsets = np.array([0, 3, 6, 9], dtype=np.int32)
    minute_of_day = np.array(
        [555, 556, 920, 555, 556, 920, 555, 556, 920], dtype=np.int16
    )
    open_ = np.array(
        [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0], dtype=np.float32
    )
    close = np.array(
        [13.0, 14.0, 15.0, 23.0, 24.0, np.nan, 33.0, 34.0, 35.0], dtype=np.float32
    )

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.isfinite(result[3:6]).all()
    expected_session1 = np.log(np.float64(open_[4]) / np.float64(close[2]))
    np.testing.assert_allclose(result[3:6], expected_session1, rtol=1e-9)
    assert np.isnan(result[6:9]).all()


def test_float32_input_to_float64_output() -> None:
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 555, 556, 920], dtype=np.int16)
    open_ = np.array(
        [100.25, 100.5, 100.75, 205.75, 206.25, 206.5], dtype=np.float32
    )
    close = np.array(
        [101.25, 101.5, 101.75, 207.25, 207.5, 207.75], dtype=np.float32
    )

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert result.dtype == np.float64
    # Upcasting float32 -> float64 is exact (no precision loss), so this is a tight check.
    expected = np.log(np.float64(open_[4]) / np.float64(close[2]))
    np.testing.assert_allclose(result[3:6], expected, rtol=1e-9)


def test_irregular_sessions_375_then_60_bar_muhurat() -> None:
    day_offsets = np.array([0, 375, 435], dtype=np.int32)
    minute_of_day = np.concatenate(
        [
            (555 + np.arange(375)).astype(np.int16),  # continuous 09:15..15:29
            (555 + np.arange(60)).astype(np.int16),  # Muhurat: 09:15..10:14
        ]
    )
    rng = np.random.default_rng(1301)
    n_rows = 435
    open_ = rng.uniform(100.0, 300.0, size=n_rows).astype(np.float32)
    close = rng.uniform(100.0, 300.0, size=n_rows).astype(np.float32)

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.isnan(result[0:375]).all()

    entry_row = 375 + 1  # 09:16 is the 2nd row of the Muhurat session
    exit_row = 920 - 555  # 15:20 position within the 375-bar session (row 365)
    expected = np.log(np.float64(open_[entry_row]) / np.float64(close[exit_row]))

    session1 = result[375:435]
    assert np.isfinite(session1).all()
    assert len(np.unique(session1)) == 1
    np.testing.assert_allclose(session1, expected, rtol=1e-9)


def test_1d_and_2d_inputs_both_supported() -> None:
    day_offsets, minute_of_day, open_1d, close_1d = _two_session_fixture()

    result_1d = tradable_overnight_return(open_1d, close_1d, day_offsets, minute_of_day)
    assert result_1d.shape == (6,)

    open_2d = np.stack([open_1d, open_1d], axis=1)
    close_2d = np.stack([close_1d, close_1d], axis=1)
    result_2d = tradable_overnight_return(open_2d, close_2d, day_offsets, minute_of_day)
    assert result_2d.shape == (6, 2)

    np.testing.assert_allclose(result_2d[:, 0], result_1d, equal_nan=True)
    np.testing.assert_allclose(result_2d[:, 1], result_1d, equal_nan=True)


def test_causal_probe_passes_at_full_strictness() -> None:
    """Fixture where the entry-labelled row IS the first row of every session and the
    exit-labelled row IS the last row of the previous session -- structurally causal-safe
    regardless of which of `open_`/`close` the real implementation names as its
    `@causal(row_arg=...)` target, matching how `overnight_return` itself is causal-safe."""
    day_offsets = np.array([0, 3, 6, 9], dtype=np.int32)
    minute_of_day = np.array(
        [556, 557, 920, 556, 557, 920, 556, 557, 920], dtype=np.int16
    )
    rng = np.random.default_rng(77)
    open_ = rng.uniform(50.0, 150.0, size=9).astype(np.float32)
    close = rng.uniform(50.0, 150.0, size=9).astype(np.float32)

    with strictness(Strictness.FULL):
        result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert result.shape == (9,)


def test_non_existent_label_yields_nan_not_raise() -> None:
    """DECISION (documented here): a syntactically valid HH:MM label that never occurs
    in the data (e.g. the market never trades at 03:00) yields an all-NaN result with NO
    exception raised. This matches the module's "NaN means absent" philosophy and is the
    same outcome as a session simply missing that label (see
    test_session_missing_entry_label_is_nan_no_reach_back)."""
    day_offsets, minute_of_day, open_, close = _two_session_fixture()

    result = tradable_overnight_return(
        open_, close, day_offsets, minute_of_day, entry_hhmm="03:00"
    )

    assert np.isnan(result).all()


def test_determinism() -> None:
    day_offsets, minute_of_day, open_, close = _two_session_fixture()
    result1 = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    result2 = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    assert np.array_equal(result1, result2, equal_nan=True)


def test_overnight_return_docstring_points_at_tradable_variant() -> None:
    assert overnight_return.__doc__ is not None
    assert "tradable_overnight_return" in overnight_return.__doc__
