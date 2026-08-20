"""Tests for uncovered branches in cv.py and embargo.py Phase B code.

These tests cover three specific branches that are rarely exercised in normal usage:
1. _embargo_width_rows line 37: when day_offsets does not end at n_rows
2. bars_to_sessions line 62: when n_bars <= 0
3. bars_to_sessions line 68: when day_offsets has no sessions (empty)
"""

import numpy as np

from nifty_quant.research.cv import _embargo_width_rows
from nifty_quant.research.embargo import bars_to_sessions


class TestEmbargoWidthRowsIncompletePanel:
    """Test _embargo_width_rows when day_offsets does not end at n_rows.

    This tests cv.py line 37: `day_offsets = np.append(day_offsets, n_rows)`
    Branch [36, 37] is taken only when day_offsets[-1] != n_rows, which happens
    when a panel's final session boundary is not yet appended to day_offsets.
    """

    def test_embargo_width_with_incomplete_day_offsets(self):
        """Test embargo width calculation when day_offsets does not end at n_rows.

        Scenario: 3 sessions with lengths 60, 105, and 80 bars. The day_offsets
        array only contains the first two boundaries [0, 60, 165], but the actual
        total row count is 245 (60 + 105 + 80). The function should append 245
        to day_offsets and proceed with embargo calculation.
        """
        # Three sessions: 60 bars, then 105 bars, then 80 bars
        # Total n_rows = 245
        day_offsets = np.array([0, 60, 165], dtype=np.int64)
        n_rows = 245  # Total rows (60 + 105 + 80)

        # Start at row 0 (beginning of first session), embargo 2 sessions forward
        # Sessions are: [0, 60), [60, 165), [165, 245)
        # 2 sessions from the start = rows [0, 165) = 165 rows
        result = _embargo_width_rows(
            start_row=0,
            n_rows=n_rows,
            embargo_sessions=2,
            day_offsets=day_offsets,
        )

        assert result == 165, (
            f"Expected 165 rows (covering 2 sessions from start), got {result}. "
            "The function should append n_rows to day_offsets when day_offsets[-1] != n_rows."
        )

    def test_embargo_width_with_incomplete_offsets_middle_start(self):
        """Test embargo width when starting mid-panel with incomplete day_offsets."""
        # Four sessions: 60, 105, 80, 120 bars
        # Total = 365 rows
        day_offsets = np.array([0, 60, 165, 245], dtype=np.int64)
        n_rows = 365  # 60 + 105 + 80 + 120

        # Start at row 60 (beginning of session 1), embargo 2 sessions forward
        # Sessions 1 and 2: [60, 165) and [165, 245) = 185 rows
        result = _embargo_width_rows(
            start_row=60,
            n_rows=n_rows,
            embargo_sessions=2,
            day_offsets=day_offsets,
        )

        assert result == 185, (
            f"Expected 185 rows, got {result}. "
            "Should cover 2 sessions starting from row 60."
        )

    def test_embargo_width_single_session_incomplete_offsets(self):
        """Test embargo width with 1 session and incomplete day_offsets."""
        # Single session of 100 bars, but incomplete day_offsets
        day_offsets = np.array([0], dtype=np.int64)
        n_rows = 100

        # Starting at row 0, embargo 1 session = 100 rows
        result = _embargo_width_rows(
            start_row=0,
            n_rows=n_rows,
            embargo_sessions=1,
            day_offsets=day_offsets,
        )

        assert result == 100, (
            f"Expected 100 rows (single incomplete session), got {result}."
        )


class TestBarsToSessionsNonPositiveInput:
    """Test bars_to_sessions when n_bars <= 0.

    This tests embargo.py line 62: `if n_bars <= 0: return 0`
    This early return handles degenerate inputs (requesting 0 or negative bars).
    """

    def test_bars_to_sessions_zero_bars(self):
        """Test bars_to_sessions with n_bars=0.

        Requesting 0 bars should return 0 sessions, as there are no bars to cover.
        """
        day_offsets = np.array([0, 60, 165, 245], dtype=np.int64)
        result = bars_to_sessions(0, day_offsets)

        assert result == 0, (
            "Requesting 0 bars should return 0 sessions (no bars to cover)."
        )

    def test_bars_to_sessions_negative_bars(self):
        """Test bars_to_sessions with n_bars < 0.

        Requesting negative bars is nonsensical but should safely return 0 sessions.
        """
        day_offsets = np.array([0, 60, 165, 245], dtype=np.int64)
        result = bars_to_sessions(-10, day_offsets)

        assert result == 0, (
            "Requesting negative bars should return 0 sessions."
        )

    def test_bars_to_sessions_one_bar(self):
        """Boundary test: requesting exactly 1 bar should return 1 session.

        This confirms the n_bars <= 0 check is not too broad.
        """
        day_offsets = np.array([0, 60, 165], dtype=np.int64)
        result = bars_to_sessions(1, day_offsets)

        assert result == 1, (
            "Requesting 1 bar should return 1 session (the last/most-recent session)."
        )


class TestBarsToSessionsEmptyPanel:
    """Test bars_to_sessions when day_offsets has no sessions.

    This tests embargo.py line 68: `if n_sessions == 0: return 0`
    This happens when day_offsets is empty or has only one element (which would
    mean zero session boundaries, thus zero session lengths via np.diff).
    """

    def test_bars_to_sessions_empty_day_offsets(self):
        """Test bars_to_sessions with empty day_offsets array.

        An empty day_offsets means there are no sessions at all. Requesting any
        number of bars should return 0 sessions.
        """
        day_offsets = np.array([], dtype=np.int64)
        result = bars_to_sessions(100, day_offsets)

        assert result == 0, (
            "Empty day_offsets (no sessions) should return 0, regardless of bar request."
        )

    def test_bars_to_sessions_single_element_offsets(self):
        """Test bars_to_sessions with single-element day_offsets.

        A single element [0] in day_offsets means the panel ends at row 0 with
        zero sessions (np.diff([0]) is empty). Requesting bars should return 0.
        """
        day_offsets = np.array([0], dtype=np.int64)
        result = bars_to_sessions(100, day_offsets)

        assert result == 0, (
            "Single-element day_offsets means zero sessions; should return 0."
        )

    def test_bars_to_sessions_empty_with_zero_bars(self):
        """Test bars_to_sessions with empty day_offsets and zero bars requested.

        Both conditions are degenerate: no sessions AND no bars requested.
        Should return 0 (result from either condition is correct).
        """
        day_offsets = np.array([], dtype=np.int64)
        result = bars_to_sessions(0, day_offsets)

        assert result == 0, (
            "Empty panel and zero bars requested should return 0."
        )
