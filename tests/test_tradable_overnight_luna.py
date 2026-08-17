"""TDD red-suite for tradable_overnight_return.
Independently authored from specs/tradable_overnight_return.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.features.market import overnight_return, tradable_overnight_return
from nifty_quant.guards import Strictness, strictness


def _fixture_a() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minute_of_day = np.array(
        [555, 556, 920, 929, 555, 556, 920, 929],
        dtype=np.int64,
    )
    close = np.array(
        [50.0, 50.1, 52.0, 52.5, 60.0, 60.2, 61.0, 61.8],
        dtype=np.float64,
    )
    open_ = np.array(
        [49.9, 50.05, 51.9, 52.4, 59.8, 60.15, 60.9, 61.7],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 8], dtype=np.int32)
    return open_, close, day_offsets, minute_of_day


def _fixture_f() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minute_of_day = np.array(
        [
            555,
            556,
            920,
            929,
            555,
            556,
            920,
            929,
            555,
            556,
            920,
            929,
        ],
        dtype=np.int64,
    )
    close_clean = np.array(
        [
            50.0,
            50.1,
            52.0,
            52.5,
            60.0,
            60.2,
            61.0,
            61.8,
            70.0,
            70.2,
            71.0,
            71.8,
        ],
        dtype=np.float64,
    )
    open_ = np.array(
        [
            49.9,
            50.05,
            51.9,
            52.4,
            59.8,
            60.15,
            60.9,
            61.7,
            69.9,
            70.15,
            70.9,
            71.7,
        ],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 8, 12], dtype=np.int32)
    return open_, close_clean, day_offsets, minute_of_day


def test_tradable_overnight_return_uses_entry_open_and_prior_exit_close_by_label() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    expected = np.log(open_[5] / close[2])

    np.testing.assert_allclose(result[4:8], np.full(4, expected))
    assert np.all(np.isnan(result[0:4]))


def test_tradable_overnight_return_differs_from_overnight_return_on_realistic_bars() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    old = overnight_return(close, day_offsets)
    new = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert old[4] == pytest.approx(np.log(60.0 / 52.5))
    assert new[4] == pytest.approx(np.log(60.15 / 52.0))
    assert abs(old[4] - new[4]) > 1e-4


def test_tradable_overnight_return_ignores_09_15_bar_even_when_corrupt() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()
    original = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    close_corrupt = close.copy()
    close_corrupt[4] = 999999.0
    changed = tradable_overnight_return(
        open_,
        close_corrupt,
        day_offsets,
        minute_of_day,
    )

    assert np.array_equal(changed, original, equal_nan=True)


def test_tradable_overnight_return_ignores_bars_after_exit_label() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()
    original = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    close_corrupt = close.copy()
    close_corrupt[3] = 999999.0
    changed = tradable_overnight_return(
        open_,
        close_corrupt,
        day_offsets,
        minute_of_day,
    )

    assert np.array_equal(changed, original, equal_nan=True)


def test_tradable_overnight_return_supports_custom_labels() -> None:
    minute_of_day = np.array(
        [555, 560, 900, 929, 555, 560, 900, 929],
        dtype=np.int64,
    )
    close = np.array(
        [70.0, 70.3, 71.0, 71.9, 80.0, 80.4, 81.0, 81.9],
        dtype=np.float64,
    )
    open_ = np.array(
        [69.9, 70.25, 70.9, 71.8, 79.8, 80.35, 80.9, 81.8],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 8], dtype=np.int32)

    result = tradable_overnight_return(
        open_,
        close,
        day_offsets,
        minute_of_day,
        entry_hhmm="09:20",
        exit_hhmm="15:00",
    )
    expected = np.log(open_[5] / close[2])

    np.testing.assert_allclose(result[4:8], np.full(4, expected))
    assert np.all(np.isnan(result[0:4]))


def test_tradable_overnight_return_missing_entry_label_yields_nan_no_reach_back() -> None:
    minute_of_day = np.array(
        [
            555,
            556,
            920,
            929,
            555,
            920,
            929,
            555,
            556,
            920,
            929,
        ],
        dtype=np.int64,
    )
    close = np.array(
        [
            50.0,
            50.1,
            52.0,
            52.5,
            65.0,
            66.0,
            66.5,
            80.0,
            80.1,
            82.0,
            82.5,
        ],
        dtype=np.float64,
    )
    open_ = np.array(
        [
            49.9,
            50.05,
            51.9,
            52.4,
            64.9,
            65.9,
            66.4,
            79.9,
            80.05,
            81.9,
            82.4,
        ],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 7, 11], dtype=np.int32)

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.all(np.isnan(result[0:4]))
    assert np.all(np.isnan(result[4:7]))
    assert np.isnan(result[4:7]).sum() == 3

    expected2 = np.log(open_[8] / close[5])
    np.testing.assert_allclose(result[7:11], np.full(4, expected2))
    assert np.all(np.isfinite(result[7:11]))
    assert not np.isclose(result[7], np.log(open_[8] / close[2]))


def test_tradable_overnight_return_muhurat_missing_exit_no_reach_back_past_it() -> None:
    minute_of_day = np.array(
        [
            555,
            556,
            920,
            929,
            1095,
            1096,
            1097,
            1098,
            1099,
            555,
            556,
            920,
            929,
        ],
        dtype=np.int64,
    )
    close = np.array(
        [
            90.0,
            90.2,
            91.0,
            91.8,
            200.0,
            201.0,
            202.0,
            203.0,
            204.0,
            110.0,
            110.2,
            111.0,
            111.9,
        ],
        dtype=np.float64,
    )
    open_ = np.array(
        [
            89.9,
            90.15,
            90.9,
            91.7,
            199.0,
            200.0,
            201.0,
            202.0,
            203.0,
            109.9,
            110.15,
            110.9,
            111.8,
        ],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 9, 13], dtype=np.int32)

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.all(np.isnan(result[0:4]))
    assert np.all(np.isnan(result[4:9]))
    # Regression guard: do not reach back to session 0's close[2] when session 1 lacks exit.
    assert np.all(np.isnan(result[9:13]))


def test_tradable_overnight_return_per_symbol_independence() -> None:
    minute_of_day = np.array(
        [555, 556, 920, 929, 555, 556, 920, 929],
        dtype=np.int64,
    )
    close = np.array(
        [
            [50.0, 150.0],
            [50.1, 150.1],
            [52.0, 152.0],
            [52.5, 152.5],
            [60.0, 160.0],
            [60.2, 160.2],
            [61.0, 161.0],
            [61.8, 161.8],
        ],
        dtype=np.float64,
    )
    open_ = np.array(
        [
            [49.9, 149.9],
            [50.05, 150.05],
            [51.9, 151.9],
            [52.4, 152.4],
            [59.8, 159.8],
            [60.15, np.nan],
            [60.9, 160.9],
            [61.7, 161.7],
        ],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 4, 8], dtype=np.int32)

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert result.shape == (8, 2)
    assert np.all(np.isnan(result[0:4, :]))

    expected_a = np.log(open_[5, 0] / close[2, 0])
    np.testing.assert_allclose(result[4:8, 0], np.full(4, expected_a))
    assert np.all(np.isnan(result[4:8, 1]))
    assert np.all(np.isfinite(result[4:8, 0]))
    assert np.all(np.isnan(result[4:8, 1]))


def test_tradable_overnight_return_first_session_is_nan() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.all(np.isnan(result[0:4]))


def test_tradable_overnight_return_broadcasts_every_row() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.isfinite(result[4])
    np.testing.assert_allclose(result[4:8], np.full(4, result[4]))


def test_tradable_overnight_return_nan_propagates_no_forward_fill() -> None:
    open_, close_clean, day_offsets, minute_of_day = _fixture_f()
    close_nan = close_clean.copy()
    close_nan[6] = np.nan

    result = tradable_overnight_return(
        open_,
        close_nan,
        day_offsets,
        minute_of_day,
    )

    assert np.all(np.isnan(result[0:4]))

    expected1 = np.log(open_[5] / close_clean[2])
    np.testing.assert_allclose(result[4:8], np.full(4, expected1))
    assert np.all(np.isfinite(result[4:8]))

    assert np.all(np.isnan(result[8:12]))
    not_bridged = np.log(open_[9] / close_clean[2])
    assert np.isnan(result[8])
    assert not np.isclose(result[8], not_bridged)


def test_tradable_overnight_return_float32_input_upcasts_to_float64() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()
    open32 = open_.astype(np.float32)
    close32 = close.astype(np.float32)

    result = tradable_overnight_return(
        open32,
        close32,
        day_offsets,
        minute_of_day,
    )

    assert result.dtype == np.float64
    expected = np.log(np.float64(open32[5]) / np.float64(close32[2]))
    np.testing.assert_allclose(result[4:8], expected, rtol=1e-9)


def test_tradable_overnight_return_irregular_session_lengths() -> None:
    def _normal_session_minutes() -> np.ndarray:
        return np.arange(555, 930, dtype=np.int64)

    def _muhurat_session_minutes(n: int = 60) -> np.ndarray:
        return np.arange(1095, 1095 + n, dtype=np.int64)

    minute_of_day = np.concatenate(
        [
            _normal_session_minutes(),
            _muhurat_session_minutes(),
            _normal_session_minutes(),
            _normal_session_minutes(),
        ]
    )
    day_offsets = np.array([0, 375, 435, 810, 1185], dtype=np.int32)
    n_rows = 1185
    close = 100.0 + 0.01 * np.arange(n_rows, dtype=np.float64)
    open_ = close - 0.005

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)

    assert np.all(np.isnan(result[0:375]))
    assert np.all(np.isnan(result[375:435]))
    assert np.all(np.isnan(result[435:810]))

    # These offsets require explicit day_offsets, never a fixed 375-row stride.
    expected3 = np.log(open_[811] / close[800])
    np.testing.assert_allclose(result[810:1185], np.full(375, expected3))
    assert np.all(np.isfinite(result[810:1185]))


def test_tradable_overnight_return_supports_1d_and_2d() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    result = tradable_overnight_return(open_, close, day_offsets, minute_of_day)
    assert result.shape == (8,)

    open_2d = open_[:, None]
    close_2d = close[:, None]
    result_2d = tradable_overnight_return(
        open_2d,
        close_2d,
        day_offsets,
        minute_of_day,
    )

    assert result_2d.shape == (8, 1)
    np.testing.assert_allclose(result_2d[:, 0], result, equal_nan=True)


def test_tradable_overnight_return_is_causal() -> None:
    open_, close_clean, day_offsets, minute_of_day = _fixture_f()
    rng = np.random.default_rng(42)

    close_perturbed = close_clean.copy()
    open_perturbed = open_.copy()
    close_perturbed[8:12] = rng.normal(5000, 100, size=4)
    open_perturbed[8:12] = rng.normal(5000, 100, size=4)

    baseline = tradable_overnight_return(
        open_,
        close_clean,
        day_offsets,
        minute_of_day,
    )
    changed = tradable_overnight_return(
        open_perturbed,
        close_perturbed,
        day_offsets,
        minute_of_day,
    )
    np.testing.assert_allclose(changed[0:8], baseline[0:8], equal_nan=True)

    open_a, close_a, day_offsets_a, minute_of_day_a = _fixture_a()
    with strictness(Strictness.FULL):
        result = tradable_overnight_return(
            open_a,
            close_a,
            day_offsets_a,
            minute_of_day_a,
        )
    assert result.shape == (8,)


def test_tradable_overnight_return_malformed_label_raises_value_error() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    # Deliberately lock in rows_at_time's malformed-label ValueError contract.
    with pytest.raises(ValueError):
        tradable_overnight_return(
            open_,
            close,
            day_offsets,
            minute_of_day,
            entry_hhmm="25:99",
        )


def test_tradable_overnight_return_is_deterministic() -> None:
    open_, close, day_offsets, minute_of_day = _fixture_a()

    result_one = tradable_overnight_return(
        open_,
        close,
        day_offsets,
        minute_of_day,
    )
    result_two = tradable_overnight_return(
        open_,
        close,
        day_offsets,
        minute_of_day,
    )

    assert np.array_equal(result_one, result_two, equal_nan=True)


def test_overnight_return_docstring_points_to_tradable_variant() -> None:
    assert "tradable_overnight_return" in overnight_return.__doc__