import math

import numpy as np
import pytest

from nifty_quant.backtest.metrics import (
    aggregate_returns_by_group,
    sharpe_ratio,
    sharpe_standard_error,
)


def test_1_daily_aggregation_matches_manual() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005, 0.01])
    group_ids = np.array([0, 0, 0, 1, 1])

    # Session 0: (1 + 0.01) * (1 - 0.02) * (1 + 0.03) - 1
    # Session 1: (1 + 0.005) * (1 + 0.01) - 1
    expected = np.array([
        (1 + 0.01) * (1 - 0.02) * (1 + 0.03) - 1,
        (1 + 0.005) * (1 + 0.01) - 1,
    ])

    result = aggregate_returns_by_group(returns, group_ids)

    assert len(result) == 2
    assert result == pytest.approx(expected)


def test_2_bar_and_checkpoint_strategies_yield_same_daily_period_count() -> None:
    n_sessions = 20
    bar_rows_per_session = 375
    checkpoint_rows_per_session = 3

    group_ids_bar = np.repeat(np.arange(n_sessions), bar_rows_per_session)
    group_ids_checkpoint = np.repeat(np.arange(n_sessions), checkpoint_rows_per_session)

    rng = np.random.default_rng(0)
    bar_returns = rng.normal(0.0, 0.001, size=group_ids_bar.size)
    checkpoint_returns = rng.normal(0.0, 0.001, size=group_ids_checkpoint.size)

    daily_bar = aggregate_returns_by_group(bar_returns, group_ids_bar)
    daily_checkpoint = aggregate_returns_by_group(checkpoint_returns, group_ids_checkpoint)

    assert len(daily_bar) == n_sessions
    assert len(daily_checkpoint) == n_sessions
    assert len(daily_bar) == len(daily_checkpoint)

    # Both now have exactly 20 daily observations, so sharpe_ratio(daily, periods_per_year=252)
    # annualizes them on the same frequency basis. Before the fix, bar-frequency rows were
    # incorrectly treated as daily observations.
    assert math.isfinite(sharpe_ratio(daily_bar, periods_per_year=252))
    assert math.isfinite(sharpe_ratio(daily_checkpoint, periods_per_year=252))


def test_3_autocorrelated_series_not_inflated_by_naive_scaling() -> None:
    n_sessions = 252
    rows_per_session = 15
    total_rows = n_sessions * rows_per_session

    rng = np.random.default_rng(0)

    # Same AR(1)-on-innovations recurrence as test_10 in test_metrics.py
    # (x[i] = rho * x[i-1] + innov[i]), with strong persistence (rho=0.95) and a
    # deterministic per-row drift that dominates the innovation scale so the sign of
    # the realized mean is robust across seeds.
    rho = 0.95
    innov = rng.normal(0.0, 0.001, size=total_rows)
    x = np.empty(total_rows)
    x[0] = innov[0]
    for i in range(1, total_rows):
        x[i] = rho * x[i - 1] + innov[i]
    minute_returns = x + 0.0015

    group_ids = np.repeat(np.arange(n_sessions), rows_per_session)
    daily = aggregate_returns_by_group(minute_returns, group_ids)

    naive_sharpe = sharpe_ratio(minute_returns, periods_per_year=total_rows)
    daily_sharpe = sharpe_ratio(daily, periods_per_year=252)

    # Naive sqrt(total_rows) annualization treats minute-scale rows as independent.
    # Under strong positive autocorrelation it overstates the Sharpe by more than 3x
    # relative to same-day compounded daily returns.
    assert naive_sharpe > 3.0 * daily_sharpe
    assert math.isfinite(naive_sharpe)
    assert math.isfinite(daily_sharpe)


def test_4_irregular_sessions_aggregate_correctly() -> None:
    rng = np.random.default_rng(0)

    # 60-bar Muhurat session followed by a 105-bar disaster-recovery Saturday session.
    muhurat_bars = 60
    disaster_recovery_bars = 105
    total_rows = muhurat_bars + disaster_recovery_bars

    returns = rng.normal(0.0, 0.001, size=total_rows)
    group_ids = np.array([0] * muhurat_bars + [1] * disaster_recovery_bars, dtype=np.int64)

    result = aggregate_returns_by_group(returns, group_ids)

    expected = np.array([
        np.prod(1.0 + returns[:muhurat_bars]) - 1.0,
        np.prod(1.0 + returns[muhurat_bars:]) - 1.0,
    ])

    assert len(result) == 2
    assert result == pytest.approx(expected)


def test_5_annualized_se_scales_correctly() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=300)

    for adjust_autocorr in (True, False):
        per_period = sharpe_standard_error(
            returns,
            annualized=False,
            adjust_autocorr=adjust_autocorr,
        )
        annualized = sharpe_standard_error(
            returns,
            annualized=True,
            periods_per_year=252,
            adjust_autocorr=adjust_autocorr,
        )

        assert annualized == pytest.approx(per_period * math.sqrt(252))


def test_6_group_ids_must_be_sorted_raises() -> None:
    returns = np.array([0.01, -0.01, 0.02])

    with pytest.raises(ValueError):
        aggregate_returns_by_group(returns, np.array([0, 1, 0]))

    with pytest.raises(ValueError):
        aggregate_returns_by_group(np.array([0.01, -0.01]), np.array([0]))


def test_7_zero_or_single_return_is_safe() -> None:
    empty = aggregate_returns_by_group(np.array([]), np.array([]))
    assert empty.size == 0

    single_returns = np.array([0.02])
    single_groups = np.array([0])
    single = aggregate_returns_by_group(single_returns, single_groups)

    assert len(single) == 1
    assert single[0] == pytest.approx(0.02)
    assert math.isfinite(single[0])

    for length in (1, 2):
        short_returns = np.linspace(0.01, 0.02, num=length)
        for annualized in (True, False):
            se = sharpe_standard_error(
                short_returns,
                annualized=annualized,
                periods_per_year=252,
            )
            assert math.isnan(se)
