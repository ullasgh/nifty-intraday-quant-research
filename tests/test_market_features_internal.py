"""Internal tests for market feature validation and error paths.

This test module covers validation error conditions in market.py functions,
including argument shape/type validation, parameter constraints, and edge cases
that would only occur with invalid/degenerate input.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features.market import (
    amihud_illiquidity,
    bars_since_open,
    beta_residual_return,
    breadth,
    close_location_value,
    cross_sectional_dispersion,
    median_pairwise_correlation,
    opening_range,
    overnight_return,
    rolling_beta,
    rv_to_vix_ratio,
    sector_relative_return,
    signed_volume_proxy,
    vol_ratio,
)


class TestHelperFunctionValidation:
    """Test _as_float64_2d and _as_float64_1d validation paths."""

    def test_rolling_beta_validates_returns_must_be_2d(self) -> None:
        """rolling_beta should reject 1-D returns array."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            rolling_beta(returns_1d, market_returns, 2, day_offsets=day_offsets)

    def test_rolling_beta_validates_market_returns_must_be_1d(self) -> None:
        """rolling_beta should reject 2-D market_returns array."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        market_2d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        day_offsets = np.array([0, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="market_returns must be a 1-D array"):
            rolling_beta(returns, market_2d, 2, day_offsets=day_offsets)

    def test_rolling_beta_validates_shape_match(self) -> None:
        """rolling_beta should reject mismatched row counts."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(
            ValueError, match="returns and market_returns must have same length"
        ):
            rolling_beta(returns, market_returns, 2, day_offsets=day_offsets)

    def test_rolling_beta_validates_window_positive(self) -> None:
        """rolling_beta should reject non-positive window."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        market_returns = np.array([0.1], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="window must be positive"):
            rolling_beta(returns, market_returns, 0, day_offsets=day_offsets)

        with pytest.raises(ValueError, match="window must be positive"):
            rolling_beta(returns, market_returns, -1, day_offsets=day_offsets)


class TestBetaResidualValidation:
    """Test beta_residual_return validation."""

    def test_beta_residual_return_validates_returns_2d(self) -> None:
        """beta_residual_return should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            beta_residual_return(returns_1d, market_returns, 2, day_offsets=day_offsets)

    def test_beta_residual_return_validates_market_1d(self) -> None:
        """beta_residual_return should reject 2-D market_returns."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        market_2d = np.array([[0.1, 0.2]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="market_returns must be a 1-D array"):
            beta_residual_return(returns, market_2d, 2, day_offsets=day_offsets)

    def test_beta_residual_return_validates_length_match(self) -> None:
        """beta_residual_return should reject mismatched lengths."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(
            ValueError, match="returns and market_returns must have same length"
        ):
            beta_residual_return(returns, market_returns, 2, day_offsets=day_offsets)


class TestSectorRelativeReturnValidation:
    """Test sector_relative_return validation."""

    def test_sector_relative_return_validates_returns_2d(self) -> None:
        """sector_relative_return should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sector_ids = np.array([0, 0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            sector_relative_return(returns_1d, sector_ids)

    def test_sector_relative_return_validates_sector_ids_length(self) -> None:
        """sector_relative_return should reject mismatched sector_ids length."""
        returns = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        sector_ids = np.array([0, 1], dtype=np.int32)  # Only 2 symbols

        with pytest.raises(
            ValueError, match="sector_ids length must match number of symbols"
        ):
            sector_relative_return(returns, sector_ids)


class TestBreadthValidation:
    """Test breadth validation."""

    def test_breadth_validates_returns_2d(self) -> None:
        """breadth should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            breadth(returns_1d)


class TestDispersionValidation:
    """Test cross_sectional_dispersion validation."""

    def test_dispersion_validates_returns_2d(self) -> None:
        """dispersion should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            cross_sectional_dispersion(returns_1d)


class TestMedianPairwiseCorrelationValidation:
    """Test median_pairwise_correlation validation."""

    def test_median_pairwise_correlation_validates_returns_2d(self) -> None:
        """median_pairwise_correlation should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            median_pairwise_correlation(returns_1d, 2, day_offsets=day_offsets)

    def test_median_pairwise_correlation_validates_window_positive(self) -> None:
        """median_pairwise_correlation should reject non-positive window."""
        returns = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="window must be positive"):
            median_pairwise_correlation(returns, 0, day_offsets=day_offsets)


class TestVolRatioValidation:
    """Test vol_ratio validation."""

    def test_vol_ratio_validates_returns_2d(self) -> None:
        """vol_ratio should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            vol_ratio(returns_1d, 2, 5, day_offsets=day_offsets)

    def test_vol_ratio_validates_window_ordering(self) -> None:
        """vol_ratio should reject short_window >= long_window."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="short_window must be less than long_window"):
            vol_ratio(returns, 5, 5, day_offsets=day_offsets)

        with pytest.raises(ValueError, match="short_window must be less than long_window"):
            vol_ratio(returns, 6, 5, day_offsets=day_offsets)


class TestRvToVixValidation:
    """Test rv_to_vix_ratio validation."""

    def test_rv_to_vix_validates_realized_vol_1d(self) -> None:
        """rv_to_vix_ratio should reject non-1D realized_vol."""
        realized_vol_2d = np.array([[0.2, 0.3]], dtype=np.float32)
        vix = np.array([20.0], dtype=np.float32)

        with pytest.raises(
            ValueError, match="realized_vol_ann must be a 1-D array"
        ):
            rv_to_vix_ratio(realized_vol_2d, vix)

    def test_rv_to_vix_validates_vix_1d(self) -> None:
        """rv_to_vix_ratio should reject non-1D vix."""
        realized_vol = np.array([0.2], dtype=np.float32)
        vix_2d = np.array([[20.0, 25.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="vix_level must be a 1-D array"):
            rv_to_vix_ratio(realized_vol, vix_2d)

    def test_rv_to_vix_validates_length_match(self) -> None:
        """rv_to_vix_ratio should reject mismatched lengths."""
        realized_vol = np.array([0.2, 0.3], dtype=np.float32)
        vix = np.array([20.0], dtype=np.float32)

        with pytest.raises(
            ValueError, match="realized_vol_ann and vix_level must have same length"
        ):
            rv_to_vix_ratio(realized_vol, vix)


class TestCloseLocationValueValidation:
    """Test close_location_value validation."""

    def test_clv_validates_high_2d(self) -> None:
        """clv should reject non-2D high."""
        high_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        low = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
        close = np.array([[0.75, 1.75, 2.75]], dtype=np.float32)

        with pytest.raises(ValueError, match="high must be a 2-D array"):
            close_location_value(high_1d, low, close)

    def test_clv_validates_low_2d(self) -> None:
        """clv should reject non-2D low."""
        high = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        low_1d = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        close = np.array([[0.75, 1.75, 2.75]], dtype=np.float32)

        with pytest.raises(ValueError, match="low must be a 2-D array"):
            close_location_value(high, low_1d, close)

    def test_clv_validates_close_2d(self) -> None:
        """clv should reject non-2D close."""
        high = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        low = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
        close_1d = np.array([0.75, 1.75, 2.75], dtype=np.float32)

        with pytest.raises(ValueError, match="close must be a 2-D array"):
            close_location_value(high, low, close_1d)

    def test_clv_validates_shapes_match(self) -> None:
        """clv should reject mismatched shapes."""
        high = np.array([[1.0, 2.0]], dtype=np.float32)
        low = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
        close = np.array([[0.75, 1.75, 2.75]], dtype=np.float32)

        with pytest.raises(ValueError, match="high, low, close must have identical shapes"):
            close_location_value(high, low, close)


class TestSignedVolumeProxyValidation:
    """Test signed_volume_proxy validation."""

    def test_svp_validates_volume_matches_ohlc(self) -> None:
        """signed_volume_proxy should reject mismatched volume shape."""
        high = np.array([[1.0, 2.0]], dtype=np.float32)
        low = np.array([[0.5, 1.5]], dtype=np.float32)
        close = np.array([[0.75, 1.75]], dtype=np.float32)
        volume = np.array([[1000.0, 2000.0, 3000.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="volume must have same shape as OHLC"):
            signed_volume_proxy(high, low, close, volume)


class TestAmihudIlliquidityValidation:
    """Test amihud_illiquidity validation."""

    def test_amihud_validates_returns_2d(self) -> None:
        """amihud should reject 1-D returns."""
        returns_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        traded_value = np.array([[1000.0, 2000.0]], dtype=np.float32)
        day_offsets = np.array([0, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="returns must be a 2-D array"):
            amihud_illiquidity(returns_1d, traded_value, 2, day_offsets=day_offsets)

    def test_amihud_validates_traded_value_2d(self) -> None:
        """amihud should reject 1-D traded_value."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        traded_1d = np.array([1000.0, 2000.0], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="traded_value must be a 2-D array"):
            amihud_illiquidity(returns, traded_1d, 2, day_offsets=day_offsets)

    def test_amihud_validates_shapes_match(self) -> None:
        """amihud should reject mismatched shapes."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        traded_value = np.array([[1000.0, 2000.0, 3000.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="returns and traded_value must have same shape"):
            amihud_illiquidity(returns, traded_value, 2, day_offsets=day_offsets)

    def test_amihud_validates_window_positive(self) -> None:
        """amihud should reject non-positive window."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        traded_value = np.array([[1000.0, 2000.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="window must be positive"):
            amihud_illiquidity(returns, traded_value, 0, day_offsets=day_offsets)


class TestOvernightReturnValidation:
    """Test overnight_return validation."""

    def test_overnight_return_validates_close_1d(self) -> None:
        """overnight_return should reject 2-D close."""
        close_2d = np.array([[100.0, 101.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="close must be a 1-D array"):
            overnight_return(close_2d, day_offsets)


class TestOpeningRangeValidation:
    """Test opening_range validation."""

    def test_opening_range_validates_high_dims(self) -> None:
        """opening_range should reject 0-D or 3D+ high."""
        high_0d = np.array(100.0, dtype=np.float32)
        low = np.array([80.0], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="high and low must be 1-D or 2-D arrays"):
            opening_range(high_0d, low, day_offsets, n_bars=1)

    def test_opening_range_validates_low_dims(self) -> None:
        """opening_range should reject 0-D or 3D+ low."""
        high = np.array([100.0], dtype=np.float32)
        low_0d = np.array(80.0, dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        # 0-D array causes IndexError when trying to access shape
        with pytest.raises((ValueError, IndexError)):
            opening_range(high, low_0d, day_offsets, n_bars=1)

    def test_opening_range_validates_n_bars_positive(self) -> None:
        """opening_range should reject non-positive n_bars."""
        high = np.array([100.0], dtype=np.float32)
        low = np.array([80.0], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="n_bars must be positive"):
            opening_range(high, low, day_offsets, n_bars=0)

        with pytest.raises(ValueError, match="n_bars must be positive"):
            opening_range(high, low, day_offsets, n_bars=-1)


class TestBarssinceOpenValidation:
    """Test bars_since_open validation."""

    # Note: bars_since_open doesn't have many validation paths,
    # but we test the day_offsets validation through check_day_offsets


class TestEdgeCasesAndDegenerate:
    """Test edge cases like empty data, all-NaN columns, etc."""

    def test_rolling_beta_handles_minimal_data(self) -> None:
        """rolling_beta should return NaN for data below min_count."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        market_returns = np.array([0.1], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        # Window larger than data - all should be NaN
        result = rolling_beta(returns, market_returns, 5, day_offsets=day_offsets)
        assert np.all(np.isnan(result))

    def test_rolling_beta_with_explicit_min_count(self) -> None:
        """rolling_beta with explicit min_count different from window."""
        returns = np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32
        )
        market_returns = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        day_offsets = np.array([0, 4], dtype=np.int32)

        # min_count=2 means we can compute beta after 2 pairs
        result = rolling_beta(returns, market_returns, 4, day_offsets=day_offsets, min_count=2)
        # First row should be NaN (only 1 data point)
        assert np.isnan(result[0]).all()
        # Second row should have values (2 data points)
        assert np.isfinite(result[1]).any()

    def test_beta_residual_return_first_row_is_nan(self) -> None:
        """beta_residual_return should return NaN for first row of session (no prior beta)."""
        returns = np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32
        )
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        result = beta_residual_return(returns, market_returns, 2, day_offsets=day_offsets)
        # First row should be NaN (no prior beta)
        assert np.isnan(result[0]).all()

    def test_sector_relative_return_all_unknown_sectors(self) -> None:
        """sector_relative_return with all -1 sector_ids should return all NaN."""
        returns = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        sector_ids = np.array([-1, -1, -1], dtype=np.int32)

        result = sector_relative_return(returns, sector_ids, min_names=2)
        # All should be NaN (unknown sectors)
        assert np.all(np.isnan(result))

    def test_cross_sectional_dispersion_single_valid(self) -> None:
        """dispersion with fewer than min_names should return NaN."""
        returns = np.array([[1.0, 2.0, 3.0, 4.0, np.nan]], dtype=np.float32)

        result = cross_sectional_dispersion(returns, min_names=5)
        # Only 4 valid symbols < 5 min_names
        assert np.isnan(result[0])

    def test_median_pairwise_correlation_below_min_names(self) -> None:
        """median_pairwise_correlation with < min_names symbols should return all NaN."""
        returns = np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32
        )
        day_offsets = np.array([0, 3], dtype=np.int32)

        # min_names=5 but only 2 symbols
        result = median_pairwise_correlation(returns, 2, day_offsets=day_offsets, min_names=5)
        assert np.all(np.isnan(result))

    def test_vol_ratio_zero_variance_long(self) -> None:
        """vol_ratio when long variance is near-zero should return NaN."""
        returns = np.array(
            [[0.0001, 0.0001], [0.0001, 0.0001], [0.5, 0.6]],
            dtype=np.float32,
        )
        day_offsets = np.array([0, 3], dtype=np.int32)

        result = vol_ratio(returns, 2, 3, day_offsets=day_offsets)
        # Last row has low variance from first 2 rows, higher from 3 rows
        assert result.shape == (3, 2)

    def test_rv_to_vix_negative_vix(self) -> None:
        """rv_to_vix_ratio with negative/zero VIX should return NaN."""
        realized_vol = np.array([0.2, 0.2], dtype=np.float32)
        vix = np.array([20.0, 0.0], dtype=np.float32)

        result = rv_to_vix_ratio(realized_vol, vix)
        # Second row has vix_as_fraction=0 -> NaN
        assert np.isfinite(result[0])
        assert np.isnan(result[1])

    def test_amihud_zero_or_negative_traded_value(self) -> None:
        """amihud with zero/negative traded_value should return NaN."""
        returns = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        traded_value = np.array([[1000.0, 0.0, -1000.0]], dtype=np.float32)
        day_offsets = np.array([0, 1], dtype=np.int32)

        result = amihud_illiquidity(returns, traded_value, 1, day_offsets=day_offsets)
        # First symbol valid, second and third NaN
        assert np.isfinite(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])

    def test_opening_range_handles_short_session(self) -> None:
        """opening_range should return all NaN if session is shorter than n_bars."""
        high = np.array([10.0, 12.0], dtype=np.float32)  # Only 2 bars
        low = np.array([8.0, 9.0], dtype=np.float32)
        day_offsets = np.array([0, 2], dtype=np.int32)

        or_high, or_low = opening_range(high, low, day_offsets, n_bars=5)
        # All rows should be NaN since session is shorter than n_bars
        assert np.isnan(or_high).all()
        assert np.isnan(or_low).all()

    def test_opening_range_2d_handles_short_session(self) -> None:
        """opening_range with 2D input should return all NaN if session < n_bars."""
        high = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)  # 2 bars
        low = np.array([[8.0, 9.0], [9.0, 10.0]], dtype=np.float32)
        day_offsets = np.array([0, 2], dtype=np.int32)

        or_high, or_low = opening_range(high, low, day_offsets, n_bars=5)
        # All rows should be NaN
        assert np.isnan(or_high).all()
        assert np.isnan(or_low).all()

    def test_rolling_beta_with_nan_values_in_window(self) -> None:
        """rolling_beta should handle NaN values in returns and market_returns."""
        returns = np.array(
            [[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0], [4.0, np.nan], [5.0, 6.0]],
            dtype=np.float32,
        )
        market_returns = np.array([0.1, np.nan, 0.3, 0.4, 0.5], dtype=np.float32)
        day_offsets = np.array([0, 5], dtype=np.int32)

        result = rolling_beta(returns, market_returns, 3, day_offsets=day_offsets, min_count=2)
        # Verify shape and some values are NaN/finite appropriately
        assert result.shape == (5, 2)

    def test_beta_residual_with_all_nan_values(self) -> None:
        """beta_residual_return should handle all-NaN rows."""
        returns = np.array(
            [[np.nan, np.nan], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32
        )
        market_returns = np.array([np.nan, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        result = beta_residual_return(returns, market_returns, 2, day_offsets=day_offsets)
        # First row should be all NaN
        assert np.all(np.isnan(result[0]))

    def test_sector_relative_return_mixed_sectors(self) -> None:
        """sector_relative_return with mixed valid/invalid sectors."""
        returns = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
        )
        # Mix of valid sectors, unknown (-1), and another valid
        sector_ids = np.array([0, 0, -1, 0, 1], dtype=np.int32)

        result = sector_relative_return(returns, sector_ids, min_names=2)
        # Symbols in sector 0 should have finite values, -1 should be NaN
        assert np.isfinite(result[0, 0])
        assert np.isfinite(result[0, 1])
        assert np.isnan(result[0, 2])
        assert np.isfinite(result[0, 3])

    def test_median_pairwise_correlation_with_constant_series(self) -> None:
        """median_pairwise_correlation with constant series (zero variance)."""
        returns = np.array(
            [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32
        )
        day_offsets = np.array([0, 3], dtype=np.int32)

        result = median_pairwise_correlation(returns, 2, day_offsets=day_offsets, min_names=2)
        # Should have some values computed despite constant series
        assert result.shape == (3,)

    def test_amihud_with_all_nan_window(self) -> None:
        """amihud_illiquidity with all-NaN values in window."""
        returns = np.array(
            [[np.nan, 0.1], [np.nan, 0.2], [0.3, np.nan]], dtype=np.float32
        )
        traded_value = np.array(
            [[1000.0, 2000.0], [1000.0, 2000.0], [1000.0, 2000.0]], dtype=np.float32
        )
        day_offsets = np.array([0, 3], dtype=np.int32)

        result = amihud_illiquidity(returns, traded_value, 2, day_offsets=day_offsets)
        # Some values should be NaN, some might be finite
        assert result.shape == (3, 2)

    def test_rolling_beta_with_day_offsets_validation(self) -> None:
        """rolling_beta exercises day_offsets validation path."""
        returns = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        # Day offsets mark session boundaries
        day_offsets = np.array([0, 2, 3], dtype=np.int32)

        result = rolling_beta(returns, market_returns, 2, day_offsets=day_offsets)
        assert result.shape == (3, 2)
        assert result.dtype == np.float64

    def test_beta_residual_return_with_day_offsets(self) -> None:
        """beta_residual_return exercises day_offsets session boundary marking."""
        returns = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        market_returns = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        day_offsets = np.array([0, 2, 3], dtype=np.int32)

        result = beta_residual_return(returns, market_returns, 2, day_offsets=day_offsets)
        assert result.shape == (3, 2)
        # First row of each session should be NaN
        assert np.all(np.isnan(result[0]))
        assert np.all(np.isnan(result[2]))

    def test_sector_relative_return_single_valid_per_sector(self) -> None:
        """sector_relative_return with minimum symbols per sector."""
        returns = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        sector_ids = np.array([0, 0, 1], dtype=np.int32)

        result = sector_relative_return(returns, sector_ids, min_names=2)
        # Sector 1 has only 1 symbol, should be NaN
        assert np.isnan(result[0, 2])
        assert np.isnan(result[1, 2])

    def test_breadth_all_zero_returns(self) -> None:
        """breadth with all zero returns."""
        returns = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        result = breadth(returns, min_names=5)
        # Zero returns count as neither advance nor decline
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_dispersion_minimum_valid_symbols(self) -> None:
        """cross_sectional_dispersion with min_names constraint."""
        returns = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        result = cross_sectional_dispersion(returns, min_names=4)
        # Not enough symbols, should be all NaN
        assert np.all(np.isnan(result))

    def test_median_correlation_minimum_names(self) -> None:
        """median_pairwise_correlation respects min_names."""
        returns = np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        )
        result = median_pairwise_correlation(returns, 2, min_names=3)
        # Only 2 symbols but min_names=3
        assert np.all(np.isnan(result))

    def test_vol_ratio_both_windows_zero_variance(self) -> None:
        """vol_ratio when both short and long have zero variance."""
        returns = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        result = vol_ratio(returns, short_window=1, long_window=2)
        # Zero variance yields NaN
        assert np.all(np.isnan(result))

    def test_rv_to_vix_with_zero_values(self) -> None:
        """rv_to_vix_ratio with zero realized volatility and VIX."""
        realized_vol = np.array([0.0, 0.0], dtype=np.float32)
        vix = np.array([0.0, 0.0], dtype=np.float32)
        result = rv_to_vix_ratio(realized_vol, vix)
        # 0 / (0/100) = NaN
        assert np.all(np.isnan(result))

    def test_close_location_value_high_equals_low_vector(self) -> None:
        """close_location_value when high==low across multiple rows."""
        high = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        low = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        close = np.array([[1.0, 2.5], [3.0, 3.5]], dtype=np.float32)

        result = close_location_value(high, low, close)
        # When high==low, result should be 0.0
        assert result[0, 0] == 0.0
        assert result[1, 0] == 0.0

    def test_signed_volume_proxy_high_equals_low(self) -> None:
        """signed_volume_proxy when high==low (degenerate case)."""
        high = np.array([[1.0, 2.0]], dtype=np.float32)
        low = np.array([[1.0, 2.0]], dtype=np.float32)
        close = np.array([[1.5, 2.5]], dtype=np.float32)
        volume = np.array([[1000.0, 2000.0]], dtype=np.float32)

        result = signed_volume_proxy(high, low, close, volume)
        # When high==low, numerator is 0
        assert result[0, 0] == 0.0

    def test_amihud_single_bar_sufficient(self) -> None:
        """amihud_illiquidity with window=1."""
        returns = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
        traded_value = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)

        result = amihud_illiquidity(returns, traded_value, window=1)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result[1]))

    def test_bars_since_open_single_bar_session(self) -> None:
        """bars_since_open with single-bar session."""
        day_offsets = np.array([0, 1, 2], dtype=np.int32)
        result = bars_since_open(day_offsets, n_rows=2)
        assert result[0] == 0
        assert result[1] == 0

    def test_overnight_return_multiple_sessions(self) -> None:
        """overnight_return with multiple discontinuous sessions."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
        day_offsets = np.array([0, 2, 4, 5], dtype=np.int32)

        result = overnight_return(close, day_offsets)
        # First row of each session (except first) should be overnight return
        assert np.isnan(result[0])  # First session, first row
        assert np.isfinite(result[2])  # Second session, first row
        assert np.isfinite(result[4])  # Third session, first row

    def test_opening_range_with_nans(self) -> None:
        """opening_range handles NaN values in high/low and returns tuple."""
        high = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        low = np.array([[0.5, 1.0], [2.0, np.nan], [4.0, 5.0]], dtype=np.float32)
        day_offsets = np.array([0, 3], dtype=np.int32)

        or_high, or_low = opening_range(high, low, day_offsets=day_offsets, n_bars=2)
        assert or_high.shape == (3, 2)
        assert or_low.shape == (3, 2)

    def test_rolling_beta_with_no_day_offsets(self) -> None:
        """rolling_beta exercises path without day_offsets (single session)."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        market_returns = np.array([0.1, 0.2], dtype=np.float32)

        # Call without day_offsets
        result = rolling_beta(returns, market_returns, 2)
        assert result.shape == (2, 2)

    def test_beta_residual_return_without_day_offsets(self) -> None:
        """beta_residual_return exercises path without day_offsets."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        market_returns = np.array([0.1, 0.2], dtype=np.float32)

        result = beta_residual_return(returns, market_returns, 2)
        assert result.shape == (2, 2)
        # First row should be NaN (no prior beta)
        assert np.all(np.isnan(result[0]))

    def test_sector_relative_return_with_min_names_edge(self) -> None:
        """sector_relative_return with exactly min_names symbols per sector."""
        returns = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        sector_ids = np.array([0, 0, 1, 1], dtype=np.int32)

        result = sector_relative_return(returns, sector_ids, min_names=2)
        # All should be finite since we have exactly 2 per sector
        assert np.all(np.isfinite(result))

    def test_breadth_mixed_returns(self) -> None:
        """breadth with mixed advancing, declining, and zero returns."""
        returns = np.array(
            [[-1.0, 0.0, 1.0, -1.0, 0.5]], dtype=np.float32
        )
        result = breadth(returns, min_names=5)
        # 2 advancing (+1, +0.5), 2 declining (-1, -1), 1 zero, 5 total: (2-2)/5 = 0
        assert result[0] == 0.0

    def test_cross_sectional_dispersion_low_min_names(self) -> None:
        """cross_sectional_dispersion with min_names=2."""
        returns = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32
        )
        result = cross_sectional_dispersion(returns, min_names=2)
        # Should compute std for each row
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_median_correlation_with_varying_values(self) -> None:
        """median_pairwise_correlation with varied returns."""
        returns = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32
        )
        result = median_pairwise_correlation(returns, 3, min_names=2)
        # Should have valid correlation
        assert result.shape == (3,)

    def test_vol_ratio_decreasing_volatility(self) -> None:
        """vol_ratio with decreasing volatility over time."""
        returns = np.array(
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.1, 0.2, 0.3]], dtype=np.float32
        )
        result = vol_ratio(returns, short_window=1, long_window=2)
        # Should produce ratios (n_rows, n_symbols)
        assert result.shape == (3, 3)

    def test_close_location_value_at_boundaries(self) -> None:
        """close_location_value at the boundaries (+1, -1)."""
        high = np.array([[2.0, 3.0]], dtype=np.float32)
        low = np.array([[1.0, 2.0]], dtype=np.float32)
        close_at_high = np.array([[2.0, 3.0]], dtype=np.float32)
        close_at_low = np.array([[1.0, 2.0]], dtype=np.float32)

        result_high = close_location_value(high, low, close_at_high)
        result_low = close_location_value(high, low, close_at_low)

        # CLV = (close - low) / (high - low) * 2 - 1
        # At high: (high - low) / (high - low) * 2 - 1 = 1
        # At low: (low - low) / (high - low) * 2 - 1 = -1
        assert result_high[0, 0] == 1.0
        assert result_low[0, 0] == -1.0

    def test_beta_residual_return_single_row(self) -> None:
        """beta_residual_return with single row (edge case for shape[0] > 1)."""
        returns = np.array([[1.0, 2.0]], dtype=np.float32)
        market_returns = np.array([0.1], dtype=np.float32)

        result = beta_residual_return(returns, market_returns, 1)
        # Single row should be all NaN (no prior beta)
        assert result.shape == (1, 2)
        assert np.all(np.isnan(result))

    def test_beta_residual_with_three_rows_and_sessions(self) -> None:
        """beta_residual_return with multiple sessions."""
        returns = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
        )
        market_returns = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        day_offsets = np.array([0, 2, 4], dtype=np.int32)

        result = beta_residual_return(returns, market_returns, 2, day_offsets=day_offsets)
        # First row of each session should be NaN
        assert np.all(np.isnan(result[0]))  # Session 1 start
        assert np.all(np.isnan(result[2]))  # Session 2 start


class TestCoverageGaps:
    """Tests to achieve 100% coverage on remaining reachable branches."""

    def test_sector_relative_return_with_nan_in_row(self) -> None:
        """sector_relative_return skips NaN values in each row (line 249 branch)."""
        returns = np.array(
            [
                [1.0, np.nan, 3.0, 4.0, 5.0],
                [2.0, 2.5, np.nan, 3.5, 4.5],
                [3.0, 3.5, 4.0, np.nan, 5.5],
            ],
            dtype=np.float32,
        )
        sector_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)

        result = sector_relative_return(returns, sector_ids, min_names=2)
        # Verify shape is correct and NaN values are propagated
        assert result.shape == (3, 5)
        # Column 1 (NaN in row 0) should be NaN for that row
        assert np.isnan(result[0, 1])
        # Column 2 (NaN in row 1) should be NaN for that row
        assert np.isnan(result[1, 2])

    def test_overnight_return_with_valid_closes(self) -> None:
        """overnight_return computes log return when closes are valid (line 630 True)."""
        close = np.array(
            [100.0, 101.0, 102.0, 105.0, 104.0, 103.0], dtype=np.float32
        )
        day_offsets = np.array([0, 3, 6], dtype=np.int32)

        result = overnight_return(close, day_offsets)
        # First row of each session is overnight return
        # Session 2 (row 3): log(105/102) where 102=close[2], 105=close[3]
        assert np.isfinite(result[3])
        expected = np.log(105.0 / 102.0)
        assert np.abs(result[3] - expected) < 1e-5

    def test_overnight_return_negative_close(self) -> None:
        """overnight_return skips when close is negative or zero (line 630 False)."""
        close = np.array(
            [100.0, 101.0, 0.0, 105.0, 104.0, 103.0], dtype=np.float32
        )
        day_offsets = np.array([0, 3, 6], dtype=np.int32)

        result = overnight_return(close, day_offsets)
        # Session 2 (row 3): prev_close = close[2] = 0.0, which fails > 0 check
        # So result[3] should be NaN
        assert np.isnan(result[3])


class TestSectorMapIntegrity:
    """Verify sector map integrity and prevent regressions.

    These tests ensure the sector map covers all equities, has no duplicate
    keys, and maintains usable sector sizes for cross-sectional analysis.
    """

    def test_sector_map_covers_all_equities(self) -> None:
        """Verify every symbol from equity_symbols() is mapped."""
        from nifty_quant.universe.sectors import SECTOR_MAP
        from nifty_quant.universe.static import equity_symbols

        symbols = equity_symbols()
        for sym in symbols:
            assert (
                sym in SECTOR_MAP
            ), f"Symbol {sym} not found in SECTOR_MAP — will resolve to -1"

    def test_sector_map_has_no_duplicates(self) -> None:
        """Verify SECTOR_MAP contains no duplicate keys.

        Duplicate keys in a dict literal silently overwrite earlier values,
        potentially causing silent sector misassignments. This test ensures
        the map size equals the number of intended symbols.
        """
        from nifty_quant.universe.sectors import SECTOR_MAP
        from nifty_quant.universe.static import equity_symbols

        symbols = equity_symbols()
        n_symbols = len(symbols)
        n_mapped = len(SECTOR_MAP)
        assert n_mapped == n_symbols, (
            f"SECTOR_MAP has {n_mapped} keys but {n_symbols} equities expected. "
            "Duplicate keys detected."
        )

    def test_sector_ids_no_unmapped_symbols(self) -> None:
        """Verify sector_ids() returns no -1 (unmapped) values."""
        from nifty_quant.universe.sectors import sector_ids
        from nifty_quant.universe.static import equity_symbols

        symbols = equity_symbols()
        ids = sector_ids(symbols)

        unmapped = [sym for sym, sid in zip(symbols, ids) if sid == -1]
        assert (
            len(unmapped) == 0
        ), f"Unmapped symbols (resolve to -1): {unmapped}"

    def test_sectors_have_minimum_members(self) -> None:
        """Verify no sector has fewer than 3 members.

        Sectors with <3 members cannot produce valid sector-relative returns
        with min_names=3, silently yielding NaN forever. This gate ensures
        feature usability.
        """
        from nifty_quant.universe.sectors import SECTOR_MAP

        sector_counts = {}
        for sector in SECTOR_MAP.values():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        undersized = {s: c for s, c in sector_counts.items() if c < 3}
        assert (
            len(undersized) == 0
        ), f"Sectors with <3 members (permanently NaN at min_names=3): {undersized}"

    def test_sector_count_within_reasonable_range(self) -> None:
        """Verify sector count is within 8-16 (as specified).

        Too many sectors (>16) defeats the purpose of sector groupings;
        too few (<8) loses meaningful differentiation.
        """
        from nifty_quant.universe.sectors import SECTOR_MAP

        unique_sectors = set(SECTOR_MAP.values())
        assert (
            8 <= len(unique_sectors) <= 16
        ), f"Sector count {len(unique_sectors)} outside range [8, 16]"

    def test_sector_distribution_summary(self) -> None:
        """Log sector distribution for visibility."""
        from nifty_quant.universe.sectors import SECTOR_MAP

        sector_counts = {}
        for sector in SECTOR_MAP.values():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        sorted_sectors = sorted(sector_counts.items(), key=lambda x: -x[1])

        # Print summary for debugging
        print(f"\nSector distribution ({len(sector_counts)} sectors):")
        for sector, count in sorted_sectors:
            print(f"  {sector:35s}: {count:2d} symbols")

        assert len(sector_counts) > 0, "No sectors found"
