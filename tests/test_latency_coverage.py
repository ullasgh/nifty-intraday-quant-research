"""Coverage tests for `nifty_quant.research.latency` — validation raises and branch gaps.

These tests close the coverage gap in latency.py: validation error paths that the
main suite (test_latency_profile.py) doesn't exercise, plus the untaken branch
where session_len <= k in session_aware_shift's loop.

Each test is written to hit exactly one missing line/branch, with hand-computed
expected values (not just "it ran") for valid-input tests per repo audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.research.latency import latency_profile, session_aware_shift


def _day_offsets(bars_per_session: list[int]) -> np.ndarray:
    """Helper: construct day_offsets array from list of bars per session."""
    offsets = [0]
    for n in bars_per_session:
        offsets.append(offsets[-1] + n)
    return np.array(offsets, dtype=np.int64)


# ---------------------------------------------------------------------------
# session_aware_shift validation raises (lines 54, 56)
# ---------------------------------------------------------------------------


def test_session_aware_shift_x_not_2d_raises() -> None:
    """Line 54: x.ndim != 2 raises ValueError."""
    x_1d = np.array([1.0, 2.0, 3.0])  # 1-D, not 2-D
    day_offsets = _day_offsets([3])
    k = 1

    with pytest.raises(ValueError, match=r"x must be 2-D"):
        session_aware_shift(x_1d, k, day_offsets)


def test_session_aware_shift_k_negative_raises() -> None:
    """Line 56: k < 0 raises ValueError."""
    x = np.random.default_rng(0).standard_normal((10, 5))
    day_offsets = _day_offsets([10])
    k = -1

    with pytest.raises(ValueError, match=r"k must be >= 0"):
        session_aware_shift(x, k, day_offsets)


# ---------------------------------------------------------------------------
# latency_profile validation raises (lines 112, 114, 116, 135)
# ---------------------------------------------------------------------------


def test_latency_profile_feature_not_2d_raises() -> None:
    """Line 112: feature.ndim != 2 raises ValueError."""
    feature_1d = np.array([1.0, 2.0, 3.0])  # 1-D, not 2-D
    close = np.random.default_rng(1).standard_normal((10, 5))
    day_offsets = _day_offsets([10])

    with pytest.raises(ValueError, match=r"feature must be 2-D"):
        latency_profile(feature_1d, close, day_offsets, horizon=1)


def test_latency_profile_close_not_2d_raises() -> None:
    """Line 114: close.ndim != 2 raises ValueError."""
    feature = np.random.default_rng(2).standard_normal((10, 5))
    close_1d = np.array([1.0, 2.0, 3.0])  # 1-D, not 2-D
    day_offsets = _day_offsets([10])

    with pytest.raises(ValueError, match=r"close must be 2-D"):
        latency_profile(feature, close_1d, day_offsets, horizon=1)


def test_latency_profile_shape_mismatch_raises() -> None:
    """Line 116: feature and close shapes don't match raises ValueError."""
    feature = np.random.default_rng(3).standard_normal((10, 5))
    close = np.random.default_rng(4).standard_normal((10, 3))  # Different n_symbols
    day_offsets = _day_offsets([10])

    with pytest.raises(ValueError, match=r"feature shape .* must match close shape"):
        latency_profile(feature, close, day_offsets, horizon=1)


def test_latency_profile_negative_lag_raises() -> None:
    """Line 135: negative lag in lags tuple raises ValueError."""
    n_rows, n_symbols = 20, 6
    rng = np.random.default_rng(5)
    feature = rng.standard_normal((n_rows, n_symbols))
    close = np.exp(np.cumsum(0.001 * rng.standard_normal((n_rows, n_symbols)), axis=0))
    day_offsets = _day_offsets([10, 10])
    lags = (0, 1, -1)  # Contains negative lag

    with pytest.raises(ValueError, match=r"every lag must be >= 0"):
        latency_profile(feature, close, day_offsets, horizon=1, lags=lags)


# ---------------------------------------------------------------------------
# Branch 71->69: session_len <= k (untaken branch in shift loop)
# ---------------------------------------------------------------------------


def test_session_aware_shift_short_session_untaken_branch() -> None:
    """Branch [71, 69]: session shorter than k stays NaN (loop condition False).

    When session_len <= k, the condition `if session_len > k:` on line 71 is False,
    so the assignment on line 72 is skipped. The output for that session remains
    all NaN.

    This test constructs a two-session example: the first session is shorter than k,
    the second is longer. We verify:
    - First session output is all NaN (loop condition was False)
    - Second session is properly shifted (loop condition was True)
    """
    # First session: 3 bars (shorter than k)
    # Second session: 10 bars (longer than k)
    day_offsets = _day_offsets([3, 10])
    n_rows = 13
    n_symbols = 2
    k = 5

    # Create a simple feature matrix: [t, s] = t + 0.1*s (easy to verify)
    feature = np.zeros((n_rows, n_symbols), dtype=np.float64)
    for t in range(n_rows):
        for s in range(n_symbols):
            feature[t, s] = float(t) + 0.1 * float(s)

    shifted = session_aware_shift(feature, k, day_offsets)

    # First session (rows 0-2): all NaN (session_len=3 < k=5)
    assert np.all(np.isnan(shifted[0:3])), (
        "First session should be all NaN since session_len (3) <= k (5)"
    )

    # Second session (rows 3-12): starts with k NaN rows (causal), then filled
    # Rows 3-7: should be NaN (first k rows of the session)
    assert np.all(np.isnan(shifted[3:8])), (
        "First k=5 rows of second session should be NaN (causal)"
    )

    # Rows 8-12: should be filled with feature[3:7] + feature[4:8] etc
    # shifted[8, s] = feature[3, s] (shift back k=5 bars within the session)
    # feature[3, s] = 3 + 0.1*s
    np.testing.assert_array_almost_equal(shifted[8], feature[3], decimal=10)

    # shifted[9, s] = feature[4, s] = 4 + 0.1*s
    np.testing.assert_array_almost_equal(shifted[9], feature[4], decimal=10)

    # shifted[10, s] = feature[5, s] = 5 + 0.1*s
    np.testing.assert_array_almost_equal(shifted[10], feature[5], decimal=10)

    # Verify the pattern for all filled rows
    for t in range(8, 13):
        expected = feature[t - k]
        np.testing.assert_array_almost_equal(shifted[t], expected, decimal=10)


def test_session_aware_shift_zero_length_session_edge_case() -> None:
    """Edge case: verify k=0 returns identity (branch always taken for non-zero sessions).

    This is a secondary check that k=0 (lag 0) always returns the input unchanged.
    """
    day_offsets = _day_offsets([5, 8])
    feature = np.random.default_rng(6).standard_normal((13, 3))
    k = 0

    shifted = session_aware_shift(feature, k, day_offsets)

    np.testing.assert_array_equal(shifted, feature, err_msg="k=0 should return identity")


def test_session_aware_shift_exactly_equal_length_session() -> None:
    """Edge case: session_len == k (condition `session_len > k` is False).

    When session_len == k, the loop condition is False (not >), so the assignment
    is skipped and the session remains all NaN. This is another way to hit branch 71->69.
    """
    day_offsets = _day_offsets([5])
    n_rows = 5
    n_symbols = 2
    k = 5  # k equals session length

    feature = np.ones((n_rows, n_symbols), dtype=np.float64)
    shifted = session_aware_shift(feature, k, day_offsets)

    # Entire session should be NaN (session_len == k, so condition is False)
    assert np.all(np.isnan(shifted)), (
        "When session_len == k, the output should be all NaN (condition False)"
    )
