"""Tests for nifty_quant.features.persistence."""

import time
import warnings

import numpy as np
import pytest

from nifty_quant import guards
from nifty_quant.features import persistence


def test_hurst_weights_orthogonality():
    """Requirement 1: default Hurst weights satisfy the OLS constraints."""
    lags, weights = persistence.hurst_weights()

    assert lags.shape == weights.shape
    assert np.abs(np.sum(weights)) < 1e-12
    assert np.abs(np.sum(weights * np.log(lags.astype(float))) - 1.0) < 1e-12


def test_rolling_hurst_matches_naive_reference():
    """Requirement 2: vectorized rolling_hurst equals the per-window OLS estimate."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((2000, 3)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    vectorized = persistence.rolling_hurst(price, window=390, warn_short_window=False)

    lags, _ = persistence.hurst_weights()

    for c in range(price.shape[1]):
        for i in (389, 500, 1000, 1999):
            w = np.log(price[i - 390 + 1 : i + 1, c])
            log_tau = []
            for lag in lags:
                diff = w[lag:] - w[:-lag]
                tau_L = np.std(diff, ddof=0)
                log_tau.append(np.log(tau_L))

            naive = np.polyfit(np.log(lags.astype(float)), log_tau, 1)[0]
            assert abs(vectorized[i, c] - naive) < 1e-9


def test_rolling_hurst_large_panel_speed():
    """Requirement 3: large-panel rolling_hurst finishes in reasonable time."""
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((50_000, 20)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    start = time.perf_counter()
    persistence.rolling_hurst(price, window=390)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0


def test_fbm_hurst_recovery_acceptance_gate():
    """Requirement 4: fBm paths recover their Hurst exponent within tolerance."""
    # ACCEPTANCE GATE
    for H in (0.3, 0.5, 0.7):
        estimates = []
        base_seed = int(round(H * 10))
        for k in range(10):
            seed = 100 * base_seed + k
            path = persistence.fbm(20_000, H, seed=seed)
            h_hat = persistence.hurst_static(path, log_price=False)
            assert np.isfinite(h_hat)
            estimates.append(h_hat)

        assert abs(np.mean(estimates) - H) < 0.05


def test_null_distribution_short_window_bias():
    """Requirement 5: null_distribution reproduces the documented window bias."""
    results_30 = persistence.null_distribution(
        persistence.rolling_hurst, window=30, n_draws=2000, seed=0
    )
    finite_30 = results_30[~np.isnan(results_30)]
    assert finite_30.size > 0
    assert 0.02 <= np.nanmean(results_30) <= 0.20
    assert np.mean(finite_30 > 0.55) < 0.01

    results_1000 = persistence.null_distribution(
        persistence.rolling_hurst, window=1000, n_draws=500, seed=0
    )
    finite_1000 = results_1000[~np.isnan(results_1000)]
    assert finite_1000.size > 0
    assert 0.45 <= np.nanmean(results_1000) <= 0.55


def test_rolling_hurst_warns_on_short_window_only():
    """Requirement 6: warn below HURST_MIN_WINDOW, no warning at/above it."""
    rng = np.random.default_rng(7)
    returns = rng.standard_normal((500, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    with pytest.warns(UserWarning):
        persistence.rolling_hurst(price, window=30)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        persistence.rolling_hurst(price, window=390)


def test_variance_ratio_family():
    """Requirement 7: variance_ratio / lo_mackinlay_z behavior on known regimes."""
    # (a) IID increments: VR(2) close to 1 relative to its asymptotic SE.
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(2000) * 0.001)
    vr = persistence.variance_ratio(x, 2)
    se = np.sqrt(2 * (2 - 1) / (3 * 2 * 2000))
    assert abs(vr - 1.0) < 3 * se

    # (b) Empirical size: at least 95% of IID |z| values below 3.
    z_values = []
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        x = np.cumsum(rng.standard_normal(2000) * 0.001)
        z_values.append(persistence.lo_mackinlay_z(x, 2, robust=True))

    z_values = np.asarray(z_values)
    assert np.mean(np.abs(z_values) < 3) >= 0.95

    # (c) Positive-autocorrelation AR(1) increments -> VR(2) > 1.
    rng = np.random.default_rng(3)
    eps = rng.standard_normal(2000) * 0.001
    incr = np.zeros(2000)
    incr[0] = eps[0]
    for t in range(1, 2000):
        incr[t] = 0.3 * incr[t - 1] + eps[t]
    assert persistence.variance_ratio(np.cumsum(incr), 2) > 1.0

    # (d) Negative-autocorrelation AR(1) increments -> VR(2) < 1.
    rng = np.random.default_rng(4)
    eps = rng.standard_normal(2000) * 0.001
    incr = np.zeros(2000)
    incr[0] = eps[0]
    for t in range(1, 2000):
        incr[t] = -0.3 * incr[t - 1] + eps[t]
    assert persistence.variance_ratio(np.cumsum(incr), 2) < 1.0


def test_nan_discipline_and_day_offsets():
    """Requirement 8: NaN propagation and day-boundary behavior."""
    rng = np.random.default_rng(5)
    returns = rng.standard_normal((800, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))
    price[400, 0] = np.nan

    out = persistence.rolling_hurst(price, window=60, warn_short_window=False)
    assert np.isnan(out[400, 0])
    assert np.all(np.isnan(out[400:400 + 60, 0]))
    assert not np.isinf(out).any()

    rng = np.random.default_rng(6)
    returns = rng.standard_normal((750, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Two ~375-bar sessions. window=390 would make each session all-NaN by
    # construction, so use a smaller window to exercise day_offsets.
    day_offsets = np.array([0, 375, 750])
    result = persistence.rolling_hurst(
        price,
        window=60,
        day_offsets=day_offsets,
        warn_short_window=False,
    )

    assert not np.isinf(result).any()
    assert np.all(np.isfinite(result) | np.isnan(result))
    assert np.all(np.isnan(result[0:59, :]))
    assert np.all(np.isnan(result[375:375 + 59, :]))


def test_dtype_discipline():
    """Requirement 9: all public numerical outputs have the expected dtypes."""
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((100, 2)) * 0.001
    price = 100.0 * np.exp(np.cumsum(returns, axis=0))

    result = persistence.rolling_hurst(price, window=30, warn_short_window=False)
    assert result.dtype == np.float64

    rng = np.random.default_rng(9)
    x2d = np.cumsum(rng.standard_normal((100, 3)) * 0.001, axis=0)

    vr2d = persistence.variance_ratio(x2d, 2)
    assert vr2d.dtype == np.float64

    z2d = persistence.lo_mackinlay_z(x2d, 2, robust=True)
    assert z2d.dtype == np.float64

    rvr = persistence.rolling_variance_ratio(x2d, window=20, q=2)
    assert rvr.dtype == np.float64

    nd = persistence.null_distribution(
        persistence.rolling_hurst, window=30, n_draws=5, seed=0
    )
    assert nd.dtype == np.float64

    fbm_path = persistence.fbm(100, 0.5, seed=0)
    assert fbm_path.dtype == np.float64

    lags, weights = persistence.hurst_weights()
    # lags is int64 by design; only weights needs the float64 dtype check here.
    assert lags.dtype == np.int64
    assert weights.dtype == np.float64


def test_rolling_hurst_causal_probe_respects_positive_domain():
    rng = np.random.default_rng(42)
    returns = rng.standard_normal((450, 3)) * 0.001
    price = 1000.0 * np.exp(np.cumsum(returns, axis=0))

    with guards.strictness(guards.Strictness.FULL):
        result = persistence.rolling_hurst(
            price,
            window=390,
            log_price=True,
            warn_short_window=False,
        )

    assert np.isfinite(result[389:]).any()


def test_rolling_hurst_rejects_non_positive_with_log_price():
    rng = np.random.default_rng(43)
    returns = rng.standard_normal((450, 3)) * 0.001
    price = 1000.0 * np.exp(np.cumsum(returns, axis=0))
    price[100, 1] = -1.0

    with guards.strictness(guards.Strictness.CHEAP):
        with pytest.raises(ValueError):
            persistence.rolling_hurst(
                price,
                window=390,
                log_price=True,
                warn_short_window=False,
            )


def test_hurst_static_log_price_requires_positive_without_warning():
    """Requirement 10: non-positive log_price input raises ValueError cleanly."""
    x = np.array([1.0, 2.0, -3.0, 4.0, 5.0] * 10)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            persistence.hurst_static(x, log_price=True)
