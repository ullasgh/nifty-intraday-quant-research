"""Integration tests for the guard layer wired into feature, persistence, and panel APIs.

The bare ``check_*`` helpers in :mod:`nifty_quant.guards` (including
``check_day_offsets`` and ``check_sorted_unique``) are always on. The
``strictness`` context manager gates only decorator-based checks such as
``finite_output``, ``panel_contract``, and ``causal``.
"""

import datetime

import numpy as np
import pytest

from nifty_quant import guards
from nifty_quant.data.panel import Panel
from nifty_quant.features import core, persistence


@pytest.fixture(autouse=True)
def _restore_strictness():
    saved = guards.get_strictness()
    yield
    guards.set_strictness(saved)


# 1. Per-row day-index arrays are rejected by every public entry point that
# accepts day_offsets. This pins the migration from the old length-n_rows
# convention to the canonical n_days+1 session-boundary convention.
def test_per_row_day_offsets_rejected():
    n_rows = 30
    day_offsets = np.repeat([0, 1, 2], n_rows // 3)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((n_rows, 3))
    price = np.exp(np.cumsum(rng.standard_normal((n_rows, 3)), axis=0))

    with pytest.raises(guards.ContractViolation):
        core.rolling_mean(x, 5, day_offsets=day_offsets)

    with pytest.raises(guards.ContractViolation):
        core.rolling_zscore(x, 5, day_offsets=day_offsets)

    with pytest.raises(guards.ContractViolation):
        persistence.rolling_hurst(
            price,
            5,
            warn_short_window=False,
            day_offsets=day_offsets,
        )


# 2. Valid canonical boundaries, including a short middle session, are accepted
# by the same public entry points and produce expected output shapes.
def test_valid_canonical_day_offsets_accepted():
    day_offsets = np.array([0, 375, 435, 810])
    n_rows = 810

    rng = np.random.default_rng(2)
    x = rng.standard_normal((n_rows, 3))
    price = np.exp(np.cumsum(rng.standard_normal((n_rows, 3)), axis=0))

    mean = core.rolling_mean(x, 20, day_offsets=day_offsets)
    zscore = core.rolling_zscore(x, 20, day_offsets=day_offsets)

    assert isinstance(mean, np.ndarray)
    assert mean.shape == x.shape
    assert isinstance(zscore, np.ndarray)
    assert zscore.shape == x.shape

    hurst = persistence.rolling_hurst(
        price,
        60,
        warn_short_window=False,
        day_offsets=day_offsets,
    )
    assert isinstance(hurst, np.ndarray)
    assert hurst.shape == price.shape


# 3. Malformed canonical boundary arrays are caught directly by
# check_day_offsets before any computation happens.
def test_nonmonotonic_day_offsets_rejected():
    x = np.random.default_rng(3).standard_normal((810, 3))
    day_offsets = np.array([0, 375, 300, 810])

    with pytest.raises(guards.ContractViolation):
        core.rolling_mean(x, 20, day_offsets=day_offsets)


def test_day_offsets_must_start_at_zero():
    x = np.random.default_rng(4).standard_normal((810, 3))
    day_offsets = np.array([5, 375, 810])

    with pytest.raises(guards.ContractViolation):
        core.rolling_mean(x, 20, day_offsets=day_offsets)


def test_day_offsets_must_end_at_n_rows():
    x = np.random.default_rng(5).standard_normal((810, 3))
    day_offsets = np.array([0, 375, 750])

    with pytest.raises(guards.ContractViolation):
        core.rolling_mean(x, 20, day_offsets=day_offsets)


# 4. Panel construction now enforces sorted, duplicate-free timestamps. These
# tests pin both failure modes of check_sorted_unique on direct construction.
def test_panel_rejects_unsorted_ts():
    ts = np.array([10, 0, 20, 30, 40], dtype=np.int64)
    day_offsets = np.array([0, ts.shape[0]])
    dates = np.array([datetime.date(2024, 1, 1)], dtype=object)

    with pytest.raises(guards.ContractViolation):
        Panel(fields={}, symbols=(), ts=ts, day_offsets=day_offsets, dates=dates)


def test_panel_rejects_duplicate_ts():
    ts = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    day_offsets = np.array([0, ts.shape[0]])
    dates = np.array([datetime.date(2024, 1, 1)], dtype=object)

    with pytest.raises(guards.ContractViolation):
        Panel(fields={}, symbols=(), ts=ts, day_offsets=day_offsets, dates=dates)


# 5. The finite_output discipline across the feature module yields finite-or-NaN
# outputs, never inf. A zero-variance rolling window is a natural divide-by-zero
# probe for rolling_zscore.
def test_finite_output_never_inf_rolling_zscore_constant_window():
    n, k, window = 20, 3, 5
    x = np.full((n, k), 5.0, dtype=np.float64)

    result = core.rolling_zscore(x, window)

    # Early warmup rows are NaN; every valid window is a constant window with
    # zero variance, so every valid output must also be NaN-not inf.
    valid_rows = slice(window - 1, n)
    assert np.all(np.isnan(result[valid_rows]))
    assert not np.isinf(result).any()


def test_finite_output_sweep_random_features_never_inf():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((50, 5))
    mask = rng.random((50, 5)) < 0.08
    x[mask] = np.nan

    outputs = [
        core.rolling_mean(x, 5),
        core.rolling_std(x, 5),
        core.rolling_zscore(x, 5),
        core.cross_sectional_zscore(x, min_names=2),
        core.cross_sectional_rank(x, pct=True, min_names=2),
    ]

    for result in outputs:
        assert not np.isinf(result).any()
        assert np.all(np.isfinite(result) | np.isnan(result))


# 6. Strictness.OFF is a real switch for decorator-based guards. Unlike the
# always-on check_* helpers, finite_output is strictness-gated and must become
# a no-op under Strictness.OFF.
def test_strictness_off_gates_decorator_guard():
    @guards.finite_output(allow_nan=False)
    def _bad_finite():
        return np.array([np.nan])

    # Under CHEAP (or the default) the decorator enforces its contract.
    with guards.strictness(guards.Strictness.CHEAP):
        with pytest.raises(guards.ContractViolation):
            _bad_finite()

    # Under OFF it permits the NaN through.
    with guards.strictness(guards.Strictness.OFF):
        result = _bad_finite()

    assert np.isnan(result[0])
