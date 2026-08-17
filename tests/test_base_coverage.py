"""100% coverage tests for nifty_quant.strategy.base module."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from nifty_quant.strategy.base import (
    ArrayMarketView,
    DataRequest,
    LookaheadError,
    PortfolioState,
    Strategy,
    TargetPortfolio,
    _ffill_2d,
)


class TestDataRequest:
    """Tests for DataRequest.warmup_bars()."""

    def test_warmup_bars_lookback_bars_only(self) -> None:
        req = DataRequest(lookback_bars=100, lookback_days=0)
        assert req.warmup_bars() == 100

    def test_warmup_bars_lookback_days_only(self) -> None:
        req = DataRequest(lookback_bars=0, lookback_days=3)
        assert req.warmup_bars() == 3 * 375

    def test_warmup_bars_lookback_both_uses_max(self) -> None:
        req = DataRequest(lookback_bars=100, lookback_days=1)
        assert req.warmup_bars() == max(100, 1 * 375)

    def test_warmup_bars_both_large_days(self) -> None:
        req = DataRequest(lookback_bars=100, lookback_days=2)
        assert req.warmup_bars() == 2 * 375


class TestTargetPortfolioValidate:
    """Tests for TargetPortfolio.validate()."""

    def test_validate_correct_shape(self) -> None:
        weights = np.array([0.5, -0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        tp.validate(n_symbols=2)

    def test_validate_wrong_shape(self) -> None:
        weights = np.array([0.5, -0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="weights shape .* != expected"):
            tp.validate(n_symbols=3)

    def test_validate_wrong_shape_too_many(self) -> None:
        weights = np.array([0.5, -0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="weights shape .* != expected"):
            tp.validate(n_symbols=1)

    def test_validate_contains_nan(self) -> None:
        weights = np.array([0.5, np.nan], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="weights contain non-finite values"):
            tp.validate(n_symbols=2)

    def test_validate_contains_inf(self) -> None:
        weights = np.array([0.5, np.inf], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="weights contain non-finite values"):
            tp.validate(n_symbols=2)

    def test_validate_contains_neginf(self) -> None:
        weights = np.array([0.5, -np.inf], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="weights contain non-finite values"):
            tp.validate(n_symbols=2)

    def test_validate_gross_exposure_within_limit(self) -> None:
        weights = np.array([0.3, -0.3], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        tp.validate(n_symbols=2, max_gross=1.0)

    def test_validate_gross_exposure_at_limit_with_tolerance(self) -> None:
        weights = np.array([0.5, 0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        tp.validate(n_symbols=2, max_gross=1.0)

    def test_validate_gross_exposure_exceeds_limit(self) -> None:
        weights = np.array([0.6, 0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="gross exposure exceeds max_gross limit"):
            tp.validate(n_symbols=2, max_gross=1.0)

    def test_validate_gross_exposure_with_custom_limit(self) -> None:
        weights = np.array([0.4, 0.4], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        tp.validate(n_symbols=2, max_gross=0.8)

    def test_validate_gross_exposure_exceeds_custom_limit(self) -> None:
        weights = np.array([0.5, 0.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="gross exposure exceeds max_gross limit"):
            tp.validate(n_symbols=2, max_gross=0.9)

    def test_validate_single_symbol(self) -> None:
        weights = np.array([1.0], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        tp.validate(n_symbols=1, max_gross=1.0)

    def test_validate_single_symbol_exceeds_limit(self) -> None:
        weights = np.array([1.5], dtype=np.float64)
        tp = TargetPortfolio(weights=weights)
        with pytest.raises(ValueError, match="gross exposure exceeds max_gross limit"):
            tp.validate(n_symbols=1, max_gross=1.0)


class TestFfill2d:
    """Tests for _ffill_2d forward-fill behavior."""

    def test_ffill_float_array_basic(self) -> None:
        arr = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]], dtype=np.float64)
        result = _ffill_2d(arr)
        assert result[1, 0] == 1.0
        assert result[2, 1] == 3.0
        assert result[2, 0] == 4.0

    def test_ffill_float_array_leading_nan_stays_nan(self) -> None:
        arr = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float64)
        result = _ffill_2d(arr)
        assert np.isnan(result[0, 0])
        assert result[1, 1] == 1.0
        assert result[1, 0] == 2.0

    def test_ffill_float_array_all_nan_column(self) -> None:
        arr = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float64)
        result = _ffill_2d(arr)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])

    def test_ffill_float_array_no_nan(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _ffill_2d(arr)
        assert np.array_equal(result, arr, equal_nan=True)

    def test_ffill_integer_array_returned_unchanged(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result = _ffill_2d(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == np.int64

    def test_ffill_bool_array_returned_unchanged(self) -> None:
        arr = np.array([[True, False], [False, True]], dtype=bool)
        result = _ffill_2d(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == bool

    def test_ffill_complex_array(self) -> None:
        arr = np.array([[1.0 + 1j, 2.0], [np.nan + 1j, 3.0]], dtype=np.complex128)
        result = _ffill_2d(arr)
        assert result[1, 0] == (1.0 + 1j)

    def test_ffill_creates_copy(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _ffill_2d(arr)
        result[0, 0] = 999.0
        assert arr[0, 0] == 1.0

    def test_ffill_float32_array(self) -> None:
        arr = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float32)
        result = _ffill_2d(arr)
        assert result[1, 0] == 1.0
        assert result.dtype == np.float32

    def test_ffill_string_array_returned_unchanged(self) -> None:
        arr = np.array([["a", "b"], ["c", "d"]], dtype=object)
        result = _ffill_2d(arr)
        assert np.array_equal(result, arr)
        assert result is not arr


class TestArrayMarketViewInit:
    """Tests for ArrayMarketView.__init__ validation."""

    def _make_minimal_arrays(
        self, n_rows: int = 5, n_symbols: int = 2
    ) -> dict[str, np.ndarray]:
        return {
            "close": np.arange(n_rows * n_symbols, dtype=np.float32).reshape(
                n_rows, n_symbols
            ),
            "open": np.zeros((n_rows, n_symbols), dtype=np.float32),
        }

    def _make_view(
        self,
        n_rows: int = 5,
        n_symbols: int = 2,
        cursor: int = 2,
        day_offsets: np.ndarray | None = None,
    ) -> ArrayMarketView:
        if day_offsets is None:
            day_offsets = np.array([0, 5], dtype=np.int32)
        return ArrayMarketView(
            panel_arrays=self._make_minimal_arrays(n_rows, n_symbols),
            cursor=cursor,
            symbols=("SYM1", "SYM2")[:n_symbols],
            ts_array=np.arange(n_rows, dtype=np.int64),
            tradable=np.ones(n_symbols, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(n_rows, dtype=np.int64),
            day_offsets=day_offsets,
        )

    def test_init_valid(self) -> None:
        view = self._make_view()
        assert view.ts == 2
        assert view.session_date == date(2024, 1, 10)

    def test_init_cursor_negative(self) -> None:
        with pytest.raises(IndexError, match="cursor out of range"):
            self._make_view(cursor=-1)

    def test_init_cursor_at_length(self) -> None:
        with pytest.raises(IndexError, match="cursor out of range"):
            self._make_view(cursor=5)

    def test_init_tradable_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="tradable shape does not match symbols"):
            ArrayMarketView(
                panel_arrays=self._make_minimal_arrays(5, 2),
                cursor=2,
                symbols=("SYM1", "SYM2", "SYM3"),
                ts_array=np.arange(5, dtype=np.int64),
                tradable=np.ones(2, dtype=bool),
                session_date=date(2024, 1, 10),
                minute_of_day=np.arange(5, dtype=np.int64),
                day_offsets=np.array([0, 5], dtype=np.int32),
            )

    def test_init_minute_of_day_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="minute_of_day shape does not match"):
            ArrayMarketView(
                panel_arrays=self._make_minimal_arrays(5, 2),
                cursor=2,
                symbols=("SYM1", "SYM2"),
                ts_array=np.arange(5, dtype=np.int64),
                tradable=np.ones(2, dtype=bool),
                session_date=date(2024, 1, 10),
                minute_of_day=np.arange(3, dtype=np.int64),
                day_offsets=np.array([0, 5], dtype=np.int32),
            )

    def test_init_day_offsets_not_1d(self) -> None:
        with pytest.raises(ValueError, match="day_offsets must be 1-D"):
            self._make_view(day_offsets=np.array([[0, 5]], dtype=np.int32))

    def test_init_day_offsets_empty(self) -> None:
        with pytest.raises(ValueError, match="day_offsets must not be empty"):
            self._make_view(day_offsets=np.array([], dtype=np.int32))

    def test_init_day_offsets_first_not_zero(self) -> None:
        with pytest.raises(ValueError, match="day_offsets\\[0\\] must be 0"):
            self._make_view(day_offsets=np.array([1, 5], dtype=np.int32))

    def test_init_day_offsets_last_not_n_rows(self) -> None:
        with pytest.raises(ValueError, match="day_offsets\\[-1\\] must equal"):
            self._make_view(day_offsets=np.array([0, 4], dtype=np.int32))

    def test_init_day_offsets_not_strictly_increasing(self) -> None:
        with pytest.raises(ValueError, match="day_offsets must be strictly increasing"):
            self._make_view(day_offsets=np.array([0, 5, 5], dtype=np.int32))

    def test_init_day_offsets_decreasing(self) -> None:
        with pytest.raises(ValueError, match="day_offsets must be strictly increasing"):
            ArrayMarketView(
                panel_arrays=self._make_minimal_arrays(10, 2),
                cursor=2,
                symbols=("SYM1", "SYM2"),
                ts_array=np.arange(10, dtype=np.int64),
                tradable=np.ones(2, dtype=bool),
                session_date=date(2024, 1, 10),
                minute_of_day=np.arange(10, dtype=np.int64),
                day_offsets=np.array([0, 5, 3, 10], dtype=np.int32),
            )

    def test_init_multiple_sessions(self) -> None:
        view = self._make_view(n_rows=10, day_offsets=np.array([0, 5, 10], dtype=np.int32))
        assert view.ts == 2


class TestArrayMarketViewLast:
    """Tests for ArrayMarketView.last() accessor."""

    def _make_view(self, n_rows: int = 5, n_symbols: int = 2) -> ArrayMarketView:
        return ArrayMarketView(
            panel_arrays={
                "close": np.arange(n_rows * n_symbols, dtype=np.float64).reshape(
                    n_rows, n_symbols
                ),
            },
            cursor=3,
            symbols=("SYM1", "SYM2")[:n_symbols],
            ts_array=np.arange(n_rows, dtype=np.int64),
            tradable=np.ones(n_symbols, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(n_rows, dtype=np.int64),
            day_offsets=np.array([0, 5], dtype=np.int32),
        )

    def test_last_at_cursor(self) -> None:
        view = self._make_view()
        result = view.last("close", offset=0)
        expected = np.array([6.0, 7.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_last_one_back(self) -> None:
        view = self._make_view()
        result = view.last("close", offset=1)
        expected = np.array([4.0, 5.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_last_max_offset(self) -> None:
        view = self._make_view()
        result = view.last("close", offset=3)
        expected = np.array([0.0, 1.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_last_negative_offset_raises_value_error(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="offset must be non-negative"):
            view.last("close", offset=-1)

    def test_last_offset_exceeds_history_raises_lookahead(self) -> None:
        view = self._make_view()
        with pytest.raises(LookaheadError, match="Not enough history"):
            view.last("close", offset=4)

    def test_last_with_ffill_propagates_values(self) -> None:
        panel_arrays = {
            "close": np.array(
                [[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan], [4.0, 40.0]],
                dtype=np.float64,
            )
        }
        view = ArrayMarketView(
            panel_arrays=panel_arrays,
            cursor=3,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(4, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(4, dtype=np.int64),
            day_offsets=np.array([0, 4], dtype=np.int32),
        )
        result = view.last("close", offset=0, ffill=True)
        assert result[0] == 4.0
        assert result[1] == 40.0

    def test_last_with_ffill_preserves_leading_nan(self) -> None:
        panel_arrays = {
            "close": np.array(
                [[np.nan, 10.0], [1.0, np.nan], [2.0, 20.0]],
                dtype=np.float64,
            )
        }
        view = ArrayMarketView(
            panel_arrays=panel_arrays,
            cursor=2,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(3, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(3, dtype=np.int64),
            day_offsets=np.array([0, 3], dtype=np.int32),
        )
        result = view.last("close", offset=2, ffill=True)
        assert np.isnan(result[0])


class TestArrayMarketViewWindow:
    """Tests for ArrayMarketView.window() accessor."""

    def _make_view(self, n_rows: int = 5, n_symbols: int = 2) -> ArrayMarketView:
        return ArrayMarketView(
            panel_arrays={
                "close": np.arange(n_rows * n_symbols, dtype=np.float64).reshape(
                    n_rows, n_symbols
                ),
            },
            cursor=4,
            symbols=("SYM1", "SYM2")[:n_symbols],
            ts_array=np.arange(n_rows, dtype=np.int64),
            tradable=np.ones(n_symbols, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(n_rows, dtype=np.int64),
            day_offsets=np.array([0, 5], dtype=np.int32),
        )

    def test_window_one_row(self) -> None:
        view = self._make_view()
        result = view.window("close", n=1)
        expected = np.array([[8.0, 9.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_window_multiple_rows(self) -> None:
        view = self._make_view()
        result = view.window("close", n=3)
        expected = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_window_full_history(self) -> None:
        view = self._make_view()
        result = view.window("close", n=5)
        assert result.shape == (5, 2)

    def test_window_zero_raises_value_error(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="n must be positive"):
            view.window("close", n=0)

    def test_window_negative_raises_value_error(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="n must be positive"):
            view.window("close", n=-1)

    def test_window_exceeds_history_raises_value_error(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="Not enough history"):
            view.window("close", n=6)

    def test_window_with_ffill(self) -> None:
        panel_arrays = {
            "close": np.array(
                [[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]],
                dtype=np.float64,
            )
        }
        view = ArrayMarketView(
            panel_arrays=panel_arrays,
            cursor=2,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(3, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=np.arange(3, dtype=np.int64),
            day_offsets=np.array([0, 3], dtype=np.int32),
        )
        result = view.window("close", n=3, ffill=True)
        assert result[1, 0] == 1.0
        assert result[2, 1] == 20.0


class TestArrayMarketViewAtTime:
    """Tests for ArrayMarketView.at_time() accessor."""

    def _make_multi_session_view(self) -> ArrayMarketView:
        n_rows = 12
        n_symbols = 2
        minute_of_day = np.array([600, 615, 630, 645, 600, 615, 630, 645, 600, 615, 630, 645])
        day_offsets = np.array([0, 4, 8, 12], dtype=np.int32)
        return ArrayMarketView(
            panel_arrays={
                "close": np.arange(n_rows * n_symbols, dtype=np.float64).reshape(
                    n_rows, n_symbols
                ),
            },
            cursor=11,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(n_rows, dtype=np.int64),
            tradable=np.ones(n_symbols, dtype=bool),
            session_date=date(2024, 1, 12),
            minute_of_day=minute_of_day,
            day_offsets=day_offsets,
        )

    def test_at_time_valid_hhmm_format(self) -> None:
        view = self._make_multi_session_view()
        result = view.at_time("close", "10:00", days_back=0)
        expected = np.array([16.0, 17.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_at_time_prior_session(self) -> None:
        view = self._make_multi_session_view()
        result = view.at_time("close", "10:00", days_back=1)
        expected = np.array([8.0, 9.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_at_time_invalid_hhmm_format_no_colon(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm must be HH:MM"):
            view.at_time("close", "1000", days_back=0)

    def test_at_time_invalid_hhmm_non_numeric(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm must be HH:MM with integer components"):
            view.at_time("close", "ab:cd", days_back=0)

    def test_at_time_hour_out_of_range_high(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm out of range"):
            view.at_time("close", "24:00", days_back=0)

    def test_at_time_hour_out_of_range_negative(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm out of range"):
            view.at_time("close", "-1:00", days_back=0)

    def test_at_time_minute_out_of_range_high(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm out of range"):
            view.at_time("close", "10:60", days_back=0)

    def test_at_time_minute_out_of_range_negative(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="hhmm out of range"):
            view.at_time("close", "10:-1", days_back=0)

    def test_at_time_days_back_before_first_session(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(LookaheadError, match="Requested day is before first session"):
            view.at_time("close", "10:00", days_back=10)

    def test_at_time_no_bar_at_label(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(LookaheadError, match="No bar at .* for target session"):
            view.at_time("close", "09:59", days_back=0)

    def test_at_time_at_or_after_cursor(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(LookaheadError, match="Requested time is at or after cursor"):
            view.at_time("close", "10:45", days_back=0)

    def test_at_time_boundary_hour_23(self) -> None:
        minute_of_day = np.array([0, 1380, 0, 1380])
        day_offsets = np.array([0, 2, 4], dtype=np.int32)
        view = ArrayMarketView(
            panel_arrays={"close": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])},
            cursor=3,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(4, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=minute_of_day,
            day_offsets=day_offsets,
        )
        result = view.at_time("close", "23:00", days_back=1)
        np.testing.assert_array_equal(result, np.array([3.0, 4.0]))

    def test_at_time_boundary_minute_59(self) -> None:
        minute_of_day = np.array([0, 59, 0, 59])
        day_offsets = np.array([0, 2, 4], dtype=np.int32)
        view = ArrayMarketView(
            panel_arrays={"close": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])},
            cursor=3,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(4, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 10),
            minute_of_day=minute_of_day,
            day_offsets=day_offsets,
        )
        result = view.at_time("close", "00:59", days_back=1)
        np.testing.assert_array_equal(result, np.array([3.0, 4.0]))


class TestArrayMarketViewDaily:
    """Tests for ArrayMarketView.daily() accessor."""

    def _make_multi_session_view(self, irregular: bool = False) -> ArrayMarketView:
        if irregular:
            n_rows = 10
            minute_of_day = np.array([600, 615, 630, 645, 600, 615, 630, 600, 615, 630])
            day_offsets = np.array([0, 4, 7, 10], dtype=np.int32)
        else:
            n_rows = 12
            minute_of_day = np.array([600, 615, 630, 645, 600, 615, 630, 645, 600, 615, 630, 645])
            day_offsets = np.array([0, 4, 8, 12], dtype=np.int32)

        return ArrayMarketView(
            panel_arrays={
                "close": np.arange(n_rows * 2, dtype=np.float64).reshape(n_rows, 2),
            },
            cursor=n_rows - 1,
            symbols=("SYM1", "SYM2"),
            ts_array=np.arange(n_rows, dtype=np.int64),
            tradable=np.ones(2, dtype=bool),
            session_date=date(2024, 1, 12),
            minute_of_day=minute_of_day,
            day_offsets=day_offsets,
        )

    def test_daily_one_prior_session(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=1)
        assert result.shape == (1, 2)
        expected = np.array([[14.0, 15.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_daily_multiple_prior_sessions(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=2)
        assert result.shape == (2, 2)
        expected = np.array([[6.0, 7.0], [14.0, 15.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_daily_with_symbol_filter(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=1, symbol="SYM1")
        expected = np.array([14.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_daily_with_symbol_filter_second_symbol(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=1, symbol="SYM2")
        expected = np.array([15.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_daily_zero_days_raises_value_error(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="n_days must be positive"):
            view.daily("close", n_days=0)

    def test_daily_negative_days_raises_value_error(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="n_days must be positive"):
            view.daily("close", n_days=-1)

    def test_daily_insufficient_history(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="Insufficient prior sessions"):
            view.daily("close", n_days=10)

    def test_daily_exact_prior_sessions_available(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=2)
        assert result.shape == (2, 2)

    def test_daily_unknown_symbol(self) -> None:
        view = self._make_multi_session_view()
        with pytest.raises(ValueError, match="Unknown symbol"):
            view.daily("close", n_days=1, symbol="NONEXISTENT")

    def test_daily_irregular_sessions(self) -> None:
        view = self._make_multi_session_view(irregular=True)
        result = view.daily("close", n_days=2)
        assert result.shape == (2, 2)

    def test_daily_excludes_current_session(self) -> None:
        view = self._make_multi_session_view()
        result = view.daily("close", n_days=1)
        assert result[0, 0] == 14.0


class MinimalParams(BaseModel):
    alpha: float = 0.5


class MinimalStrategy(Strategy):
    name = "minimal"
    Params = MinimalParams

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: Any) -> dict[str, np.ndarray]:
        return {}

    def on_decision(self, view: Any, signals: Any, state: Any) -> TargetPortfolio | None:
        return None


class TestStrategyFromConfig:
    """Tests for Strategy.from_config() class method."""

    def test_from_config_valid(self) -> None:
        config = {"params": {"alpha": 0.7}}
        strategy = MinimalStrategy.from_config(config)
        assert strategy.params.alpha == 0.7

    def test_from_config_missing_params_key(self) -> None:
        config = {"other": {}}
        with pytest.raises(KeyError):
            MinimalStrategy.from_config(config)

    def test_from_config_params_not_mapping(self) -> None:
        config = {"params": [1, 2, 3]}
        with pytest.raises(TypeError, match="'params' key must be a mapping"):
            MinimalStrategy.from_config(config)

    def test_from_config_params_not_a_list(self) -> None:
        config = {"params": "not_a_dict"}
        with pytest.raises(TypeError, match="'params' key must be a mapping"):
            MinimalStrategy.from_config(config)

    def test_from_config_params_string_not_mapping(self) -> None:
        config = {"params": "invalid"}
        with pytest.raises(TypeError, match="'params' key must be a mapping"):
            MinimalStrategy.from_config(config)

    def test_from_config_default_params(self) -> None:
        config = {"params": {}}
        strategy = MinimalStrategy.from_config(config)
        assert strategy.params.alpha == 0.5


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""

    def test_portfolio_state_creation(self) -> None:
        state = PortfolioState(
            shares=np.array([1.0, 2.0], dtype=np.float64),
            cash=10000.0,
            equity=10000.0,
            ts=1234567890,
        )
        assert state.cash == 10000.0
        assert state.ts == 1234567890

    def test_portfolio_state_frozen(self) -> None:
        state = PortfolioState(
            shares=np.array([1.0, 2.0], dtype=np.float64),
            cash=10000.0,
            equity=10000.0,
            ts=1234567890,
        )
        with pytest.raises(Exception):
            state.cash = 20000.0
