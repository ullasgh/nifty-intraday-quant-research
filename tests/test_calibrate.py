from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_quant.features.calibrate import (
    NullCalibration,
    calibrate_empirical,
    calibrate_null,
    calibration_report,
    suggest_threshold,
)
from nifty_quant.features.persistence import hurst_static


def _random_walk_hurst_draws(window: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_draws):
        log_price = np.cumsum(rng.standard_normal(window))
        path = np.exp(log_price)
        values.append(hurst_static(path))
    return np.asarray(values, dtype=np.float64)


def test_reproduces_measured_bias() -> None:
    cal30 = calibrate_null(hurst_static, window=30, n_draws=1500, seed=0)
    assert 0.02 <= cal30.mean <= 0.20
    assert cal30.threshold_at(90) < 0.50

    cal1000 = calibrate_null(hurst_static, window=1000, n_draws=500, seed=0)
    assert 0.44 <= cal1000.mean <= 0.56


def test_suggested_threshold_fpr() -> None:
    window = 390
    threshold = suggest_threshold(
        hurst_static,
        window=window,
        target_fpr=0.10,
        n_draws=2000,
        seed=1,
    )

    fresh_draws = _random_walk_hurst_draws(window, n_draws=2000, seed=99)
    fpr = float(np.mean(fresh_draws > threshold))
    assert fpr == pytest.approx(0.10, abs=0.03)


def test_p_value_monotone_right_tail() -> None:
    cal: NullCalibration = calibrate_null(
        hurst_static, window=390, n_draws=1500, seed=2
    )

    lo = cal.threshold_at(10)
    hi = cal.threshold_at(90)
    observed = np.linspace(lo, hi, 7)

    p_values = [cal.p_value(float(x), tail="right") for x in observed]
    assert all(0.0 <= p <= 1.0 for p in p_values)
    assert all(p_i >= p_j for p_i, p_j in zip(p_values, p_values[1:]))


def test_threshold_at_median() -> None:
    cal = calibrate_null(hurst_static, window=390, n_draws=300, seed=3)
    assert cal.threshold_at(50) == pytest.approx(cal.percentiles[50.0])


def test_seed_reproducibility() -> None:
    a = calibrate_null(hurst_static, window=390, n_draws=300, seed=7)
    b = calibrate_null(hurst_static, window=390, n_draws=300, seed=7)
    c = calibrate_null(hurst_static, window=390, n_draws=300, seed=8)

    assert a.mean == b.mean
    assert a.std == b.std
    assert a.percentiles == b.percentiles
    assert a.mean != c.mean


def test_calibration_report_window_ordering() -> None:
    df = calibration_report(
        "hurst_static", (30, 390), estimator=hurst_static, n_draws=400
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert list(df["window"]) == [30, 390]

    p90_30 = df.loc[df["window"] == 30, "p90"].item()
    p90_390 = df.loc[df["window"] == 390, "p90"].item()
    assert p90_390 > p90_30


def test_calibrate_empirical() -> None:
    values = np.arange(1, 101, dtype=float)

    cal = calibrate_empirical(values)
    assert cal.percentiles[50.0] == pytest.approx(np.percentile(values, 50))
    assert cal.percentiles[10.0] == pytest.approx(np.percentile(values, 10))
    assert cal.percentiles[90.0] == pytest.approx(np.percentile(values, 90))

    values_with_nan = np.append(values, np.nan)
    cal_nan = calibrate_empirical(values_with_nan)

    assert cal_nan.percentiles == cal.percentiles
    assert cal_nan.n_draws == 100
