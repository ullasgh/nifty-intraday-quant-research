"""Tests to reach 100% coverage of _rolling_beta_reference input validation.

Tests for the PRIVATE _rolling_beta_reference function's input validation
raises (lines 122, 126, 128, 132 in src/nifty_quant/features/market.py).
"""

import numpy as np
import pytest

from nifty_quant.features.market import _rolling_beta_reference


def test_rolling_beta_reference_returns_not_2d() -> None:
    """Test that _rolling_beta_reference raises ValueError if returns is not 2-D."""
    # Pass 1-D array for returns (should be 2-D)
    returns = np.array([0.01, 0.02], dtype=np.float64)
    market_returns = np.array([0.01, 0.02], dtype=np.float64)
    window = 2

    with pytest.raises(ValueError, match="returns must be a 2-D array"):
        _rolling_beta_reference(returns, market_returns, window)


def test_rolling_beta_reference_market_returns_not_1d() -> None:
    """Test that _rolling_beta_reference raises ValueError if market_returns is not 1-D."""
    # Pass 2-D array for market_returns (should be 1-D)
    returns = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    market_returns = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    window = 2

    with pytest.raises(ValueError, match="market_returns must be a 1-D array"):
        _rolling_beta_reference(returns, market_returns, window)


def test_rolling_beta_reference_length_mismatch() -> None:
    """Test that _rolling_beta_reference raises ValueError on length mismatch."""
    # returns has 5 rows, market_returns has 2 rows
    returns = np.array(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08], [0.09, 0.10]],
        dtype=np.float64,
    )
    market_returns = np.array([0.01, 0.02], dtype=np.float64)
    window = 2

    with pytest.raises(ValueError, match="returns and market_returns must have same length"):
        _rolling_beta_reference(returns, market_returns, window)


def test_rolling_beta_reference_window_not_positive() -> None:
    """Test that _rolling_beta_reference raises ValueError if window <= 0."""
    returns = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    market_returns = np.array([0.01, 0.02], dtype=np.float64)
    window = 0

    with pytest.raises(ValueError, match="window must be positive"):
        _rolling_beta_reference(returns, market_returns, window)


def test_rolling_beta_reference_window_negative() -> None:
    """Test that _rolling_beta_reference raises ValueError if window < 0."""
    returns = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    market_returns = np.array([0.01, 0.02], dtype=np.float64)
    window = -1

    with pytest.raises(ValueError, match="window must be positive"):
        _rolling_beta_reference(returns, market_returns, window)
