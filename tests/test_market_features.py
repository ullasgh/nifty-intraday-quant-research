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
from nifty_quant.guards import Strictness, strictness
from nifty_quant.universe.sectors import SECTOR_MAP, sector_ids


def _make_returns(n_rows: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.001, size=(n_rows, n_symbols)).astype(np.float32)


def _make_market(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.001, size=n_rows).astype(np.float32)


def _make_ohlc2d(
    n_rows: int, n_symbols: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = rng.normal(100, 1, size=(n_rows, n_symbols)).astype(np.float32)
    high = close + np.abs(
        rng.normal(0.5, 0.1, size=(n_rows, n_symbols))
    ).astype(np.float32)
    low = close - np.abs(
        rng.normal(0.5, 0.1, size=(n_rows, n_symbols))
    ).astype(np.float32)
    volume = rng.uniform(1000, 10000, size=(n_rows, n_symbols)).astype(np.float32)
    return high, low, close, volume


def _make_close1d(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=n_rows)))
    ).astype(np.float32)


def _day_offsets_three_sessions() -> np.ndarray:
    return np.array([0, 10, 20, 30], dtype=np.int32)


def test_rolling_beta_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 3, 10)
    market_returns = _make_market(n_rows, 11)
    day_offsets = _day_offsets_three_sessions()
    with strictness(Strictness.FULL):
        result = rolling_beta(
            returns, market_returns, 5, day_offsets=day_offsets, min_count=3
        )
    assert result.shape == (n_rows, 3)


def test_beta_residual_return_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 3, 12)
    market_returns = _make_market(n_rows, 13)
    day_offsets = _day_offsets_three_sessions()
    with strictness(Strictness.FULL):
        result = beta_residual_return(
            returns, market_returns, 5, day_offsets=day_offsets
        )
    assert result.shape == (n_rows, 3)


def test_sector_relative_return_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 3, 14)
    sector_ids = np.array([0, 0, 1], dtype=np.int32)
    with strictness(Strictness.FULL):
        result = sector_relative_return(returns, sector_ids, min_names=2)
    assert result.shape == (n_rows, 3)


def test_breadth_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 5, 15)
    with strictness(Strictness.FULL):
        result = breadth(returns, min_names=3)
    assert result.shape == (n_rows,)


def test_cross_sectional_dispersion_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 5, 16)
    with strictness(Strictness.FULL):
        result = cross_sectional_dispersion(returns, min_names=3)
    assert result.shape == (n_rows,)


def test_median_pairwise_correlation_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 5, 17)
    day_offsets = _day_offsets_three_sessions()
    with strictness(Strictness.FULL):
        result = median_pairwise_correlation(
            returns, 5, day_offsets=day_offsets, min_names=3
        )
    assert result.shape == (n_rows,)


def test_vol_ratio_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 3, 18)
    day_offsets = _day_offsets_three_sessions()
    with strictness(Strictness.FULL):
        result = vol_ratio(returns, 2, 5, day_offsets=day_offsets)
    assert result.shape == (n_rows, 3)


def test_rv_to_vix_ratio_is_causal() -> None:
    n_rows = 30
    rng = np.random.default_rng(19)
    realized_vol_ann = rng.normal(0.2, 0.05, size=n_rows).astype(np.float32)
    vix_level = rng.uniform(15, 25, size=n_rows).astype(np.float32)
    with strictness(Strictness.FULL):
        result = rv_to_vix_ratio(realized_vol_ann, vix_level)
    assert result.shape == (n_rows,)


def test_close_location_value_is_causal() -> None:
    high, low, close, _ = _make_ohlc2d(30, 3, 20)
    with strictness(Strictness.FULL):
        result = close_location_value(high, low, close)
    assert result.shape == (30, 3)


def test_signed_volume_proxy_is_causal() -> None:
    high, low, close, volume = _make_ohlc2d(30, 3, 21)
    with strictness(Strictness.FULL):
        result = signed_volume_proxy(high, low, close, volume)
    assert result.shape == (30, 3)


def test_amihud_illiquidity_is_causal() -> None:
    n_rows = 30
    returns = _make_returns(n_rows, 3, 22)
    rng = np.random.default_rng(23)
    traded_value = rng.uniform(1000, 10000, size=(n_rows, 3)).astype(np.float32)
    day_offsets = _day_offsets_three_sessions()
    with strictness(Strictness.FULL):
        result = amihud_illiquidity(
            returns, traded_value, 5, day_offsets=day_offsets
        )
    assert result.shape == (n_rows, 3)


def test_bars_since_open_is_causal() -> None:
    day_offsets = _day_offsets_three_sessions()
    result = bars_since_open(day_offsets, 30)
    assert result.shape == (30,)
    assert result.dtype == np.int32


def test_overnight_return_is_causal() -> None:
    day_offsets = _day_offsets_three_sessions()
    close = _make_close1d(30, 24)
    baseline = overnight_return(close, day_offsets)
    rng = np.random.default_rng(25)
    perturbed = close.copy()
    perturbed[20:] = rng.normal(2000, 10, size=10).astype(np.float32)
    changed = overnight_return(perturbed, day_offsets)
    np.testing.assert_allclose(changed[:20], baseline[:20], equal_nan=True)
    with strictness(Strictness.FULL):
        result = overnight_return(close, day_offsets)
    assert result.shape == (30,)


def test_beta_residual_uses_prior_beta_not_contemporaneous() -> None:
    n_rows = 60
    day_offsets = np.array([0, 30, n_rows], dtype=np.int32)
    rng = np.random.default_rng(26)
    market_returns = rng.normal(0, 0.01, size=n_rows).astype(np.float32)
    dynamic_beta = np.linspace(0.5, 1.5, n_rows, dtype=np.float32)
    noise = rng.normal(0, 0.0002, size=(n_rows, 2)).astype(np.float32)
    returns = (dynamic_beta[:, None] * market_returns[:, None] + noise).astype(
        np.float32
    )
    beta = rolling_beta(
        returns, market_returns, 5, day_offsets=day_offsets, min_count=5
    )
    residual = beta_residual_return(
        returns, market_returns, 5, day_offsets=day_offsets
    )
    expected_prior = returns - np.roll(beta, 1, axis=0) * market_returns[:, None]
    wrong_contemporaneous = returns - beta * market_returns[:, None]

    found_difference = False
    for session_start in (0, 30):
        for idx in range(session_start + 5, session_start + 30):
            if not (
                np.isfinite(expected_prior[idx]).all()
                and np.isfinite(residual[idx]).all()
            ):
                continue
            np.testing.assert_allclose(
                residual[idx], expected_prior[idx], rtol=1e-6, atol=1e-6
            )
            if np.max(np.abs(residual[idx] - wrong_contemporaneous[idx])) > 1e-6:
                found_difference = True
    assert found_difference


def test_opening_range_is_nan_before_range_completes() -> None:
    high = np.array([10, 12, 11, 9, 8, 13], dtype=np.float32)
    low = np.array([8, 9, 10, 7, 6, 11], dtype=np.float32)
    day_offsets = np.array([0, 6], dtype=np.int32)
    or_high, or_low = opening_range(high, low, day_offsets, n_bars=5)
    assert np.isnan(or_high[:5]).all()
    assert np.isnan(or_low[:5]).all()
    assert or_high[5] == pytest.approx(high[0:5].max(), rel=1e-9)
    assert or_low[5] == pytest.approx(low[0:5].min(), rel=1e-9)


def test_all_functions_respect_irregular_sessions() -> None:
    n_rows = 435
    day_offsets = np.array([0, 375, n_rows], dtype=np.int32)
    rng = np.random.default_rng(101)
    returns = rng.normal(0, 0.001, size=(n_rows, 3)).astype(np.float32)
    market_returns = rng.normal(0, 0.001, size=n_rows).astype(np.float32)
    high, low, close, volume = _make_ohlc2d(n_rows, 3, 102)
    sector_ids = np.array([0, 0, 1], dtype=np.int32)
    traded_value = rng.uniform(1000, 10000, size=(n_rows, 3)).astype(np.float32)
    realized_vol_ann = rng.uniform(0.1, 0.3, size=n_rows).astype(np.float32)
    vix_level = rng.uniform(10, 30, size=n_rows).astype(np.float32)
    close_1d = close[:, 0].copy()

    calls = [
        (
            rolling_beta,
            (returns, market_returns),
            {
                "window": 20,
                "day_offsets": day_offsets,
                "min_count": 10,
            },
        ),
        (
            beta_residual_return,
            (returns, market_returns),
            {"window": 20, "day_offsets": day_offsets},
        ),
        (
            sector_relative_return,
            (returns, sector_ids),
            {"min_names": 2},
        ),
        (breadth, (returns,), {"min_names": 3}),
        (cross_sectional_dispersion, (returns,), {"min_names": 3}),
        (
            median_pairwise_correlation,
            (returns,),
            {"window": 20, "day_offsets": day_offsets, "min_names": 3},
        ),
        (
            vol_ratio,
            (returns,),
            {"short_window": 5, "long_window": 20, "day_offsets": day_offsets},
        ),
        (rv_to_vix_ratio, (realized_vol_ann, vix_level), {}),
        (close_location_value, (high, low, close), {}),
        (signed_volume_proxy, (high, low, close, volume), {}),
        (
            amihud_illiquidity,
            (returns, traded_value),
            {"window": 20, "day_offsets": day_offsets},
        ),
        (bars_since_open, (day_offsets, n_rows), {}),
        (overnight_return, (close_1d, day_offsets), {}),
        (opening_range, (high, low, day_offsets), {"n_bars": 5}),
    ]

    for func, args, kwargs in calls:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            for arr in result:
                assert arr.shape[0] == n_rows
        else:
            assert result.shape[0] == n_rows


def test_nothing_spans_a_session_boundary_except_overnight_return() -> None:
    n_rows_session1 = 10
    n_rows_session2 = 5
    n_rows = n_rows_session1 + n_rows_session2
    day_offsets = np.array([0, n_rows_session1, n_rows], dtype=np.int32)
    rng = np.random.default_rng(103)
    returns = rng.normal(0, 0.001, size=(n_rows, 2)).astype(np.float32)
    returns[n_rows_session1:] += 100.0
    market_returns = rng.normal(0, 0.001, size=n_rows).astype(np.float32)
    market_returns[n_rows_session1:] += 100.0
    beta = rolling_beta(
        returns, market_returns, 4, day_offsets=day_offsets, min_count=4
    )
    assert np.isnan(beta[n_rows_session1:n_rows_session1 + 3, :]).all()

    traded_value = rng.uniform(1000, 10000, size=(n_rows, 2)).astype(np.float32)
    amihud = amihud_illiquidity(
        returns, traded_value, 4, day_offsets=day_offsets
    )
    assert np.isnan(amihud[n_rows_session1:n_rows_session1 + 3, :]).all()


def test_overnight_return_first_session_is_nan() -> None:
    day_offsets = np.array([0, 10, 25], dtype=np.int32)
    close = _make_close1d(25, 104)
    result = overnight_return(close, day_offsets)
    assert np.isnan(result[:10]).all()


def test_bars_since_open_restarts_each_session() -> None:
    day_offsets = np.array([0, 375, 435, 810], dtype=np.int32)
    result = bars_since_open(day_offsets, 810)
    expected = np.concatenate(
        [
            np.arange(375, dtype=np.int32),
            np.arange(60, dtype=np.int32),
            np.arange(375, dtype=np.int32),
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_rolling_beta_matches_hand_computed_cov_over_var() -> None:
    returns = np.array(
        [[1.0], [2.0], [3.0], [2.0], [1.0], [4.0]], dtype=np.float32
    )
    market_returns = np.array(
        [0.0, 0.5, -0.2, 1.0, 0.3, 0.7], dtype=np.float32
    )
    day_offsets = np.array([0, 6], dtype=np.int32)
    result = rolling_beta(
        returns, market_returns, 6, day_offsets=day_offsets, min_count=6
    )
    expected = (
        np.cov(returns[:, 0], market_returns, ddof=1)[0, 1]
        / np.var(market_returns, ddof=1)
    )
    assert np.isnan(result[0:5, 0]).all()
    assert result[5, 0] == pytest.approx(expected, rel=1e-9)


def test_beta_of_market_on_itself_is_one() -> None:
    n_rows = 6
    market_returns = np.array(
        [0.0, 0.5, -0.2, 1.0, 0.3, 0.7], dtype=np.float32
    )
    returns = np.tile(market_returns[:, None], (1, 3)).astype(np.float32)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    result = rolling_beta(
        returns, market_returns, 3, day_offsets=day_offsets, min_count=3
    )
    assert np.allclose(result[2:], 1.0, rtol=1e-6)


def test_breadth_counts_advancing_minus_declining_over_present() -> None:
    returns = np.array([[0.01, -0.02, 0.03, 0.01, -0.005]], dtype=np.float32)
    result = breadth(returns, min_names=5)
    assert result[0] == pytest.approx((3 - 2) / 5, rel=1e-9)


def test_breadth_zero_returns_count_as_neither() -> None:
    returns = np.array([[0.02, 0.0, -0.01, 0.03, -0.02]], dtype=np.float32)
    result = breadth(returns, min_names=5)
    assert result[0] == pytest.approx(0.0, rel=1e-9)


def test_dispersion_matches_numpy_std_ddof_one() -> None:
    row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = cross_sectional_dispersion(row, min_names=5)
    expected = np.nanstd(row[0], ddof=1)
    assert result[0] == pytest.approx(expected, rel=1e-9)


def test_clv_equals_plus_one_when_close_equals_high() -> None:
    high = np.array([[100.0]], dtype=np.float32)
    low = np.array([[90.0]], dtype=np.float32)
    close = np.array([[100.0]], dtype=np.float32)
    result = close_location_value(high, low, close)
    assert result[0, 0] == pytest.approx(1.0, rel=1e-9)


def test_clv_equals_minus_one_when_close_equals_low() -> None:
    high = np.array([[100.0]], dtype=np.float32)
    low = np.array([[90.0]], dtype=np.float32)
    close = np.array([[90.0]], dtype=np.float32)
    result = close_location_value(high, low, close)
    assert result[0, 0] == pytest.approx(-1.0, rel=1e-9)


def test_clv_is_zero_when_high_equals_low() -> None:
    high = np.array([[100.0]], dtype=np.float32)
    low = np.array([[100.0]], dtype=np.float32)

    close_eq = np.array([[100.0]], dtype=np.float32)
    result_eq = close_location_value(high, low, close_eq)
    assert result_eq[0, 0] == 0.0
    assert np.isfinite(result_eq[0, 0])

    close_diff = np.array([[105.0]], dtype=np.float32)
    result_diff = close_location_value(high, low, close_diff)
    assert result_diff[0, 0] == 0.0
    assert np.isfinite(result_diff[0, 0])


def test_clv_bounded_in_minus_one_to_one() -> None:
    rng = np.random.default_rng(24)
    n_rows, n_symbols = 50, 5
    close = rng.normal(100, 5, size=(n_rows, n_symbols)).astype(np.float32)
    high = close + np.abs(
        rng.normal(0.5, 0.2, size=(n_rows, n_symbols))
    ).astype(np.float32)
    low = close - np.abs(
        rng.normal(0.5, 0.2, size=(n_rows, n_symbols))
    ).astype(np.float32)
    result = close_location_value(high, low, close)
    assert np.all(np.isfinite(result))
    assert np.all((result >= -1.0 - 1e-6) & (result <= 1.0 + 1e-6))


def test_vol_ratio_greater_than_one_on_expanding_vol() -> None:
    n_rows = 40
    rng = np.random.default_rng(25)
    low_vol = rng.normal(0, 0.001, size=(20, 1)).astype(np.float32)
    high_vol = rng.normal(0, 0.05, size=(20, 1)).astype(np.float32)
    returns = np.concatenate([low_vol, high_vol], axis=0)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    result = vol_ratio(returns, 5, 20, day_offsets=day_offsets)
    assert result[29, 0] > 1.0


def test_vol_ratio_rejects_short_ge_long() -> None:
    returns = _make_returns(20, 2, 27)
    day_offsets = np.array([0, 20], dtype=np.int32)
    with pytest.raises(ValueError):
        vol_ratio(returns, 10, 10, day_offsets=day_offsets)
    with pytest.raises(ValueError):
        vol_ratio(returns, 11, 10, day_offsets=day_offsets)


def test_rv_to_vix_divides_vix_by_100() -> None:
    realized_vol_ann = np.array([0.20], dtype=np.float64)
    vix_level = np.array([20.0], dtype=np.float64)
    result = rv_to_vix_ratio(realized_vol_ann, vix_level)
    assert result[0] == pytest.approx(1.0, rel=1e-9)


def test_amihud_higher_for_illiquid_symbol() -> None:
    n_rows = 10
    rng = np.random.default_rng(28)
    abs_returns = np.abs(rng.normal(0, 0.01, size=(n_rows, 2))).astype(np.float32)
    traded_value = np.empty((n_rows, 2), dtype=np.float32)
    traded_value[:, 0] = 1_000_000.0
    traded_value[:, 1] = 1_000.0
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    result = amihud_illiquidity(
        abs_returns, traded_value, 5, day_offsets=day_offsets
    )
    assert result[-1, 1] > result[-1, 0]


def test_amihud_non_positive_traded_value_yields_nan_not_inf() -> None:
    returns = np.array([[0.01]] * 5, dtype=np.float32)
    traded_value = np.array(
        [[0.0], [-5.0], [0.0], [-5.0], [0.0]], dtype=np.float32
    )
    day_offsets = np.array([0, 5], dtype=np.int32)
    result = amihud_illiquidity(returns, traded_value, 3, day_offsets=day_offsets)
    assert not np.any(np.isinf(result))
    assert np.isnan(result[2:5, 0]).all()


def test_median_pairwise_correlation_of_identical_series_is_one() -> None:
    n_rows = 30
    rng = np.random.default_rng(29)
    series = rng.normal(0, 0.001, size=n_rows).astype(np.float32)
    returns = np.tile(series[:, None], (1, 5)).astype(np.float32)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    result = median_pairwise_correlation(
        returns, 10, day_offsets=day_offsets, min_names=5
    )
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    assert np.allclose(finite, 1.0, atol=1e-6)


def test_median_pairwise_correlation_of_independent_series_is_near_zero() -> None:
    n_rows = 200
    returns = _make_returns(n_rows, 10, 30)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    result = median_pairwise_correlation(
        returns, 30, day_offsets=day_offsets, min_names=5
    )
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    assert np.all(np.abs(finite) < 0.3)


def test_sector_relative_return_sums_to_zero_within_sector() -> None:
    returns = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    sector_ids = np.array([0, 0, 0, 0], dtype=np.int32)
    result = sector_relative_return(returns, sector_ids, min_names=3)
    assert result.shape == (1, 4)
    assert result[0, :].sum() == pytest.approx(0.0, abs=1e-6)


def test_sector_relative_unknown_sector_is_nan() -> None:
    returns = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    sector_ids = np.array([0, 0, -1], dtype=np.int32)
    result = sector_relative_return(returns, sector_ids, min_names=2)
    assert np.isnan(result[0, 2])
    assert np.isfinite(result[0, 0:2]).all()


def test_sector_below_min_names_is_nan() -> None:
    returns = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    sector_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    result = sector_relative_return(returns, sector_ids, min_names=3)
    assert np.isnan(result[0, 0:2]).all()
    assert np.isfinite(result[0, 2:5]).all()


def test_opening_range_matches_first_n_bars_high_low() -> None:
    high = np.array([10, 12, 11, 9, 8, 13], dtype=np.float32)
    low = np.array([8, 9, 10, 7, 6, 11], dtype=np.float32)
    day_offsets = np.array([0, 6], dtype=np.int32)
    or_high, or_low = opening_range(high, low, day_offsets, n_bars=3)
    expected_high = high[0:3].max()
    expected_low = low[0:3].min()
    np.testing.assert_allclose(or_high[3:], expected_high, rtol=0)
    np.testing.assert_allclose(or_low[3:], expected_low, rtol=0)


def test_nan_bars_propagate_and_are_not_filled() -> None:
    high = np.arange(1, 7, dtype=np.float32).reshape(6, 1)
    low = np.arange(0, 6, dtype=np.float32).reshape(6, 1)
    close = np.arange(0.5, 6.5, dtype=np.float32).reshape(6, 1)
    high[3, 0] = np.nan
    result_clv = close_location_value(high, low, close)
    assert np.isnan(result_clv[3, 0])

    returns_all_nan = _make_returns(6, 5, 31)
    returns_all_nan[3, :] = np.nan
    result_breadth = breadth(returns_all_nan, min_names=3)
    result_disp = cross_sectional_dispersion(returns_all_nan, min_names=3)
    assert np.isnan(result_breadth[3])
    assert np.isnan(result_disp[3])

    returns_beta = _make_returns(6, 1, 32)
    market_returns = _make_market(6, 33)
    returns_beta[4, 0] = np.nan
    day_offsets = np.array([0, 6], dtype=np.int32)
    result_beta = rolling_beta(
        returns_beta, market_returns, 3, day_offsets=day_offsets, min_count=3
    )
    assert np.isnan(result_beta[4, 0])
    assert np.isnan(result_beta[5, 0])

    returns_amihud = _make_returns(6, 1, 34)
    returns_amihud[4, 0] = np.nan
    traded_value = np.full((6, 1), 1000.0, dtype=np.float32)
    result_amihud = amihud_illiquidity(
        returns_amihud,
        traded_value,
        3,
        day_offsets=np.array([0, 6], dtype=np.int32),
    )
    assert np.isnan(result_amihud[4, 0])


def test_all_nan_symbol_never_poisons_a_cross_sectional_statistic() -> None:
    returns = np.array([[0.01, -0.02, 0.03, np.nan]], dtype=np.float32)
    broad_result = breadth(returns, min_names=3)
    disp_result = cross_sectional_dispersion(returns, min_names=3)
    expected_breadth = (2 - 1) / 3
    expected_disp = np.nanstd(returns[0, :3], ddof=1)
    assert broad_result[0] == pytest.approx(expected_breadth, rel=1e-9)
    assert disp_result[0] == pytest.approx(expected_disp, rel=1e-9)


def test_min_names_gates_breadth_dispersion_and_correlation() -> None:
    returns_few = np.array([[0.01, -0.02, 0.03, 0.01]], dtype=np.float32)
    breadth_few = breadth(returns_few, min_names=5)
    disp_few = cross_sectional_dispersion(returns_few, min_names=5)
    assert np.isnan(breadth_few[0])
    assert np.isnan(disp_few[0])

    returns_enough = np.array(
        [[0.01, -0.02, 0.03, 0.01, -0.005]], dtype=np.float32
    )
    breadth_enough = breadth(returns_enough, min_names=5)
    disp_enough = cross_sectional_dispersion(returns_enough, min_names=5)
    assert np.isfinite(breadth_enough[0])
    assert np.isfinite(disp_enough[0])

    rng = np.random.default_rng(35)
    day_offsets = np.array([0, 20], dtype=np.int32)
    returns_corr_few = rng.normal(0, 0.001, size=(20, 3)).astype(np.float32)
    corr_few = median_pairwise_correlation(
        returns_corr_few, 5, day_offsets=day_offsets, min_names=5
    )
    assert np.isnan(corr_few).all()

    returns_corr_enough = rng.normal(0, 0.001, size=(20, 5)).astype(np.float32)
    corr_enough = median_pairwise_correlation(
        returns_corr_enough, 5, day_offsets=day_offsets, min_names=5
    )
    assert not np.isnan(corr_enough).all()


def test_constant_series_zero_variance_degenerate_cases() -> None:
    n_rows = 5
    day_offsets = np.array([0, n_rows], dtype=np.int32)

    returns_const_symbol = np.ones((n_rows, 1), dtype=np.float32)
    market_varying = np.linspace(-0.01, 0.01, n_rows, dtype=np.float32)
    result_const_symbol = rolling_beta(
        returns_const_symbol, market_varying, 3, day_offsets=day_offsets
    )
    assert np.all(~np.isinf(result_const_symbol))

    returns_varying = _make_returns(n_rows, 1, 36)
    market_const = np.full(n_rows, 0.02, dtype=np.float32)
    result_const_market = rolling_beta(
        returns_varying, market_const, 3, day_offsets=day_offsets
    )
    assert np.all(~np.isinf(result_const_market))
    assert np.isnan(result_const_market).any()

    returns_identical_row = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    disp = cross_sectional_dispersion(returns_identical_row, min_names=3)
    assert disp[0] == 0.0


def test_all_outputs_are_float64() -> None:
    n_rows = 12
    n_symbols = 3
    returns = _make_returns(n_rows, n_symbols, 37)
    market = _make_market(n_rows, 38)
    high, low, close, volume = _make_ohlc2d(n_rows, n_symbols, 39)
    sector_ids = np.array([0, 0, 1], dtype=np.int32)
    traded_value = np.full((n_rows, n_symbols), 1000.0, dtype=np.float32)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    realized_vol = np.full(n_rows, 0.2, dtype=np.float32)
    vix = np.full(n_rows, 20.0, dtype=np.float32)
    close_1d = close[:, 0].copy()

    float_results = [
        rolling_beta(
            returns, market, 3, day_offsets=day_offsets, min_count=3
        ),
        beta_residual_return(returns, market, 3, day_offsets=day_offsets),
        sector_relative_return(returns, sector_ids, min_names=2),
        breadth(returns, min_names=3),
        cross_sectional_dispersion(returns, min_names=3),
        median_pairwise_correlation(
            returns, 3, day_offsets=day_offsets, min_names=3
        ),
        vol_ratio(returns, 2, 3, day_offsets=day_offsets),
        rv_to_vix_ratio(realized_vol, vix),
        close_location_value(high, low, close),
        signed_volume_proxy(high, low, close, volume),
        amihud_illiquidity(
            returns, traded_value, 3, day_offsets=day_offsets
        ),
        overnight_return(close_1d, day_offsets),
    ]
    for result in float_results:
        assert result.dtype == np.float64

    assert bars_since_open(day_offsets, n_rows).dtype == np.int32

    or_high, or_low = opening_range(high, low, day_offsets, n_bars=3)
    assert or_high.dtype == np.float64
    assert or_low.dtype == np.float64


def test_output_shapes_match_contract() -> None:
    n_rows = 12
    n_symbols = 3
    returns = _make_returns(n_rows, n_symbols, 40)
    market = _make_market(n_rows, 41)
    high, low, close, volume = _make_ohlc2d(n_rows, n_symbols, 42)
    sector_ids = np.array([0, 0, 1], dtype=np.int32)
    traded_value = np.full((n_rows, n_symbols), 1000.0, dtype=np.float32)
    day_offsets = np.array([0, n_rows], dtype=np.int32)
    realized_vol = np.full(n_rows, 0.2, dtype=np.float32)
    vix = np.full(n_rows, 20.0, dtype=np.float32)
    close_1d = close[:, 0].copy()

    per_symbol = [
        rolling_beta(
            returns, market, 3, day_offsets=day_offsets, min_count=3
        ),
        beta_residual_return(returns, market, 3, day_offsets=day_offsets),
        sector_relative_return(returns, sector_ids, min_names=2),
        vol_ratio(returns, 2, 3, day_offsets=day_offsets),
    ]
    for result in per_symbol:
        assert result.shape == (n_rows, n_symbols)

    market_level = [
        breadth(returns, min_names=3),
        cross_sectional_dispersion(returns, min_names=3),
        median_pairwise_correlation(
            returns, 3, day_offsets=day_offsets, min_names=3
        ),
        rv_to_vix_ratio(realized_vol, vix),
        bars_since_open(day_offsets, n_rows),
        overnight_return(close_1d, day_offsets),
    ]
    for result in market_level:
        assert result.shape == (n_rows,)

    ohlcv = [
        close_location_value(high, low, close),
        signed_volume_proxy(high, low, close, volume),
        amihud_illiquidity(
            returns, traded_value, 3, day_offsets=day_offsets
        ),
    ]
    for result in ohlcv:
        assert result.shape == (n_rows, n_symbols)

    or_high, or_low = opening_range(high, low, day_offsets, n_bars=3)
    assert or_high.shape == high.shape
    assert or_low.shape == high.shape


def test_sector_ids_maps_unknown_to_minus_one() -> None:
    result = sector_ids(["NOT_A_REAL_SYMBOL_XYZ"])
    assert result[0] == -1
    assert result.dtype == np.int32

    keys = list(SECTOR_MAP.keys())[:2]
    if keys:
        known = sector_ids(keys)
        assert known.shape == (len(keys),)
        assert np.all(known >= 0)
        assert np.issubdtype(known.dtype, np.integer)
