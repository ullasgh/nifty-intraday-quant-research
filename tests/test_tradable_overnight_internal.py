"""Internal tests for tradable_overnight_return error paths.

These tests cover input validation and error conditions that are not exercised
by the contract test suites (test_tradable_overnight_deepseek.py,
test_tradable_overnight_luna.py), ensuring 100% line and branch coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features.market import tradable_overnight_return


def test_malformed_exit_hhmm_raises_value_error() -> None:
    """Malformed exit_hhmm must raise ValueError naming exit_hhmm.

    The entry_hhmm path is tested by the contract suites; this covers the
    symmetric exit_hhmm path, which has identical parsing and error handling.
    """
    open_ = np.array([49.9, 50.05, 51.9, 52.4, 59.8, 60.15, 60.9, 61.7])
    close = np.array([50.0, 50.1, 52.0, 52.5, 60.0, 60.2, 61.0, 61.8])
    day_offsets = np.array([0, 4, 8], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 929, 555, 556, 920, 929])

    with pytest.raises(ValueError) as exc_info:
        tradable_overnight_return(
            open_,
            close,
            day_offsets,
            minute_of_day,
            exit_hhmm="25:99",
        )

    assert "exit_hhmm" in str(exc_info.value)


def test_3d_open_array_raises_value_error() -> None:
    """3-D open_ array must raise ValueError naming the dimension problem.

    np.asarray converts a wide range of inputs; this validates that the
    function rejects arrays outside the documented 1-D/2-D contract.
    """
    open_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    day_offsets = np.array([0, 4, 8], dtype=np.int32)
    minute_of_day = np.array([555, 556, 920, 929, 555, 556, 920, 929])

    with pytest.raises(ValueError) as exc_info:
        tradable_overnight_return(
            open_3d,
            close,
            day_offsets,
            minute_of_day,
        )

    error_msg = str(exc_info.value)
    assert "1-D" in error_msg or "2-D" in error_msg or "array" in error_msg


def test_non_1d_minute_of_day_raises_value_error() -> None:
    """2-D minute_of_day array must raise ValueError naming the dimension problem.

    The function accepts bare arrays, not Panel objects; callers might pass
    minute_of_day with wrong shape. This validates the contract check.
    """
    open_ = np.array([49.9, 50.05, 51.9, 52.4, 59.8, 60.15, 60.9, 61.7])
    close = np.array([50.0, 50.1, 52.0, 52.5, 60.0, 60.2, 61.0, 61.8])
    day_offsets = np.array([0, 4, 8], dtype=np.int32)
    minute_of_day_2d = np.array([[555, 556], [920, 929], [555, 556], [920, 929]])

    with pytest.raises(ValueError) as exc_info:
        tradable_overnight_return(
            open_,
            close,
            day_offsets,
            minute_of_day_2d,
        )

    error_msg = str(exc_info.value)
    assert "1-D" in error_msg or "minute_of_day" in error_msg or "array" in error_msg
