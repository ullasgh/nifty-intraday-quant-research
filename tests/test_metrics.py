import math

import numpy as np
import pytest
from scipy import stats

from nifty_quant.backtest.metrics import (
    compute_metrics,
    deflated_sharpe,
    drawdown_series,
    effective_n_trials,
    expected_max_sharpe,
    max_drawdown,
    pbo_cscv,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    turnover,
    verdict_line,
)


def test_1_analytic_sharpe() -> None:
    constant = np.full(100, 0.001, dtype=np.float64)
    assert math.isnan(compute_metrics(constant).sharpe)
    assert math.isnan(sharpe_ratio(constant))

    rng = np.random.default_rng(0)
    mu, sigma, t = 0.0008, 0.01, 100_000
    returns = rng.normal(mu, sigma, size=t)
    expected = (mu / sigma) * math.sqrt(252)
    assert sharpe_ratio(returns, periods_per_year=252) == pytest.approx(
        expected, rel=0.05
    )


def test_2_max_drawdown_hand_computed() -> None:
    equity = np.array([100.0, 120.0, 90.0, 130.0, 80.0])
    depth, peak_idx, trough_idx = max_drawdown(equity)

    assert depth == pytest.approx((80.0 - 130.0) / 130.0)
    assert peak_idx == 3
    assert trough_idx == 4
    assert not (peak_idx == 1 and trough_idx == 2)


def test_3_drawdown_series_running_max() -> None:
    equity = np.array([100.0, 120.0, 90.0, 130.0, 80.0])
    dd = drawdown_series(equity)

    assert np.all(dd <= 1e-12)
    running_max = np.maximum.accumulate(equity)
    at_running_max = np.isclose(equity, running_max, atol=1e-12)
    assert np.allclose(dd[at_running_max], 0.0, atol=1e-10)


def test_4_sortino_gt_sharpe_positively_skewed() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_exponential(2000) * 0.01 - 0.003

    assert stats.skew(returns, bias=False) > 0.0

    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)

    assert math.isfinite(sharpe)
    assert math.isfinite(sortino)
    assert sortino > sharpe


def test_5_turnover_alternating() -> None:
    weights = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    result = turnover(weights)

    assert np.allclose(result[1:], 1.0, atol=1e-12)


def test_6_expected_max_sharpe_monotonicity() -> None:
    assert expected_max_sharpe(2, 1.0) < expected_max_sharpe(1000, 1.0)
    assert expected_max_sharpe(100, 0.0) == 0.0

    with pytest.raises(ValueError):
        expected_max_sharpe(1, 1.0)


def test_7_deflated_sharpe() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.002, 0.01, size=2000)

    dsr_strong = deflated_sharpe(returns, sr0=0.0)
    assert 0.0 <= dsr_strong <= 1.0
    assert dsr_strong > 0.95

    dsr_weak = deflated_sharpe(returns, sr0=0.5)
    assert 0.0 <= dsr_weak <= 1.0
    assert dsr_weak < 0.5

    sr0_values = [0.0, 0.05, 0.5]
    dsr_values = [deflated_sharpe(returns, sr0=sr0) for sr0 in sr0_values]
    for i in range(len(dsr_values) - 1):
        assert dsr_values[i] >= dsr_values[i + 1]


def test_8_effective_n_trials() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=500)
    identical = np.tile(base.reshape(-1, 1), (1, 10))
    assert effective_n_trials(identical) == pytest.approx(1.0, abs=0.1)

    independent = rng.normal(0.0, 0.01, size=(500, 10))
    assert effective_n_trials(independent) == pytest.approx(10.0, abs=2.0)


def test_9_pbo_cscv() -> None:
    # 10 trials (not 5): an even trial count avoids the discrete-rank tie at the
    # median that would otherwise bias the "at/below median" count for odd n_trials.
    rng = np.random.default_rng(0)
    strong_trial = rng.normal(0.01, 0.01, size=960)
    noise_trials = rng.normal(0.0, 0.01, size=(960, 9))
    strong_matrix = np.column_stack((strong_trial, noise_trials))
    assert pbo_cscv(strong_matrix, n_splits=16) < 0.2

    rng_noise = np.random.default_rng(27)
    noise_matrix = rng_noise.normal(0.0, 0.01, size=(2400, 10))
    pbo_noise = pbo_cscv(noise_matrix, n_splits=16)
    assert 0.3 <= pbo_noise <= 0.7


def test_10_sharpe_standard_error_autocorr() -> None:
    rng = np.random.default_rng(0)
    t = 1000
    innov = rng.normal(size=t)
    x = np.empty(t)
    x[0] = innov[0]
    for i in range(1, t):
        x[i] = 0.5 * x[i - 1] + innov[i]

    se_auto = sharpe_standard_error(x, adjust_autocorr=True)
    se_iid = sharpe_standard_error(x, adjust_autocorr=False)
    assert se_auto > se_iid


def test_11_verdict_line() -> None:
    line_est = verdict_line(
        sharpe_net=1.0,
        sr_se=0.1,
        n_trials=10,
        n_eff=5.0,
        sr0=0.0,
        dsr=0.99,
        pbo=0.1,
    )
    assert line_est.endswith("ESTABLISHED")
    assert not line_est.endswith("NOT ESTABLISHED")

    line_not_est = verdict_line(
        sharpe_net=0.2,
        sr_se=0.1,
        n_trials=10,
        n_eff=5.0,
        sr0=0.0,
        dsr=0.5,
        pbo=0.6,
    )
    assert line_not_est.endswith("NOT ESTABLISHED")


def test_12_no_inf_sweep() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=1000)

    metric_values = list(compute_metrics(returns).to_dict().values())
    scalars = [
        sharpe_ratio(returns),
        sortino_ratio(returns),
        sharpe_standard_error(returns, adjust_autocorr=True),
        expected_max_sharpe(10, 1.0),
        deflated_sharpe(returns, sr0=0.0),
        effective_n_trials(rng.normal(0.0, 0.01, size=(200, 3))),
        pbo_cscv(rng.normal(0.0, 0.01, size=(240, 3)), n_splits=8),
    ]

    all_values = metric_values + scalars
    for value in all_values:
        assert not math.isinf(value)
