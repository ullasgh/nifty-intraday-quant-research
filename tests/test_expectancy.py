from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from nifty_quant import guards
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research import expectancy

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _single_session_day_offsets(n_rows: int) -> np.ndarray:
    return np.array([0, n_rows], dtype=int)


def _multi_session_day_offsets(lengths: list[int]) -> np.ndarray:
    return np.cumsum([0] + lengths, dtype=int)


def _driftless_gbm_close(
    rng: np.random.Generator,
    n_rows: int,
    n_symbols: int,
    sigma: float,
    close0: float = 100.0,
) -> np.ndarray:
    log_rets = rng.normal(0.0, sigma, size=(n_rows - 1, n_symbols))
    close = np.empty((n_rows, n_symbols), dtype=np.float64)
    close[0, :] = close0
    close[1:, :] = close0 * np.exp(np.cumsum(log_rets, axis=0))
    return close


def _close_from_one_step_log_returns(
    one_step_log_returns: np.ndarray,
    close0: float = 1.0,
) -> np.ndarray:
    cum = np.cumsum(one_step_log_returns, axis=0)
    log_close = np.vstack(
        [np.zeros((1, one_step_log_returns.shape[1]), dtype=np.float64), cum]
    )
    return close0 * np.exp(log_close)


def _build_horizon_one_fwd_with_effect(
    n_rows: int,
    n_symbols: int,
    effect_top_bps: float,
    effect_bottom_bps: float,
    rng: np.random.Generator,
    sigma: float = 0.002,
) -> tuple[expectancy.ForwardReturns, np.ndarray]:
    """Build a horizon-1 ForwardReturns object plus a per-row feature array.

    The feature array is ``np.tile(np.arange(n_symbols), (n_rows, 1))``.  With
    ``method="cross_sectional_rank"`` this guarantees that, for every row, the
    lowest feature symbol falls into bucket 0 and the highest into bucket
    ``n_buckets - 1``.  The one-step log returns are constructed with mean
    ``effect_top_bps/1e4`` for the top symbol and ``effect_bottom_bps/1e4``
    for the bottom symbol; the middle symbols have zero mean.
    """
    feature = np.tile(np.arange(n_symbols, dtype=np.float64), (n_rows, 1))
    one_step = np.empty((n_rows - 1, n_symbols), dtype=np.float64)
    for sym in range(n_symbols):
        if sym == n_symbols - 1:
            mean_log = effect_top_bps / 1e4
        elif sym == 0:
            mean_log = effect_bottom_bps / 1e4
        else:
            mean_log = 0.0
        one_step[:, sym] = rng.normal(mean_log, sigma, size=n_rows - 1)
    close = _close_from_one_step_log_returns(one_step)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon=1)
    return fwd, feature


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------

def test_forward_return_is_log_ratio_h_bars_ahead() -> None:
    close = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [4.0, 8.0],
            [8.0, 16.0],
            [16.0, 32.0],
        ],
        dtype=np.float64,
    )
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(5), horizon=2)
    expected = np.log(4.0)
    assert fwd.values[0, 0] == pytest.approx(expected)
    assert fwd.values[0, 1] == pytest.approx(expected)
    assert fwd.values[1, 0] == pytest.approx(expected)
    assert fwd.values[1, 1] == pytest.approx(expected)
    assert fwd.values[2, 0] == pytest.approx(expected)
    assert fwd.values[2, 1] == pytest.approx(expected)
    assert np.isnan(fwd.values[3]).all()
    assert np.isnan(fwd.values[4]).all()


def test_never_crosses_session_boundary() -> None:
    day_offsets = _multi_session_day_offsets([6, 4])
    n_rows = 10
    n_symbols = 2
    close = np.tile(
        np.arange(1, n_rows + 1, dtype=np.float64)[:, None], (1, n_symbols)
    )
    fwd = expectancy.forward_returns(close, day_offsets, horizon=2)
    assert np.isnan(fwd.values[4:6]).all()
    assert np.isnan(fwd.values[8:10]).all()
    assert not np.isnan(fwd.values[:4]).any()
    assert not np.isnan(fwd.values[6:8]).any()


def test_irregular_sessions_handled() -> None:
    lengths = [375, 60]
    day_offsets = _multi_session_day_offsets(lengths)
    total = 435
    n_symbols = 3
    close = np.tile(
        np.arange(1, total + 1, dtype=np.float64)[:, None], (1, n_symbols)
    )
    horizon = 10
    fwd = expectancy.forward_returns(close, day_offsets, horizon)
    assert np.isnan(fwd.values[365:375]).all()
    assert np.isnan(fwd.values[425:435]).all()
    assert fwd.n_defined == (375 - 10) + (60 - 10)
    assert fwd.n_nan_tail == 10 + 10
    assert not np.isnan(fwd.values[:365]).any()
    assert not np.isnan(fwd.values[375:425]).any()


def test_nan_close_propagates_and_is_not_filled() -> None:
    n_rows = 10
    n_symbols = 2
    close = np.tile(
        np.arange(1, n_rows + 1, dtype=np.float64)[:, None], (1, n_symbols)
    )
    close[3, 0] = np.nan
    horizon = 2
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(n_rows), horizon)

    assert np.isnan(fwd.values[1, 0])
    assert np.isnan(fwd.values[3, 0])

    for t in (0, 2, 4, 5, 6, 7):
        assert not np.isnan(fwd.values[t, 0])

    for t in range(8):
        assert not np.isnan(fwd.values[t, 1])

    assert np.isnan(fwd.values[8:10]).all()


def test_horizon_one_equals_next_bar_log_return() -> None:
    close = np.array(
        [
            [1.0, 3.0],
            [2.0, 6.0],
            [4.0, 9.0],
            [8.0, 27.0],
        ],
        dtype=np.float64,
    )
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(4), horizon=1)
    assert fwd.values[0, 0] == pytest.approx(np.log(2.0))
    assert fwd.values[1, 0] == pytest.approx(np.log(2.0))
    assert fwd.values[2, 0] == pytest.approx(np.log(2.0))
    assert fwd.values[0, 1] == pytest.approx(np.log(2.0))
    assert fwd.values[1, 1] == pytest.approx(np.log(1.5))
    assert fwd.values[2, 1] == pytest.approx(np.log(3.0))
    assert np.isnan(fwd.values[3]).all()


@pytest.mark.parametrize("horizon", [0, -1, -5])
def test_horizon_zero_and_negative_raise_value_error(horizon: int) -> None:
    close = np.ones((10, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        expectancy.forward_returns(close, _single_session_day_offsets(10), horizon)


def test_horizon_longer_than_session_yields_all_nan_for_that_session() -> None:
    close = np.ones((3, 2), dtype=np.float64)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(3), horizon=4)
    assert np.isnan(fwd.values).all()


def test_n_nan_tail_counts_horizon_truncation_correctly() -> None:
    lengths = [10, 8]
    day_offsets = _multi_session_day_offsets(lengths)
    total = 18
    close = np.tile(
        np.arange(1, total + 1, dtype=np.float64)[:, None], (1, 2)
    )
    horizon = 3
    fwd = expectancy.forward_returns(close, day_offsets, horizon)
    expected_tail = horizon * len(lengths)
    assert fwd.n_nan_tail == expected_tail
    assert fwd.n_defined == total - expected_tail
    assert np.isnan(fwd.values[7:10]).all()
    assert np.isnan(fwd.values[15:18]).all()


def test_dtype_is_float64() -> None:
    close = np.ones((10, 2), dtype=np.float32)
    fwd = expectancy.forward_returns(close, _single_session_day_offsets(10), horizon=2)
    assert fwd.values.dtype == np.float64


# ---------------------------------------------------------------------------
# causal_buckets
# ---------------------------------------------------------------------------

def test_expanding_quantile_uses_only_prior_rows() -> None:
    rng = np.random.default_rng(10)
    n_rows, n_symbols = 200, 3
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)
    min_history = 50

    base = expectancy.causal_buckets(
        feature,
        day_offsets,
        n_buckets=5,
        method="expanding_quantile",
        min_history=min_history,
    )

    cut = 120
    rng2 = np.random.default_rng(999)
    perturbed = feature.copy()
    perturbed[cut + 1 :, :] = rng2.normal(
        loc=10.0, scale=5.0, size=(n_rows - cut - 1, n_symbols)
    )
    pert = expectancy.causal_buckets(
        perturbed,
        day_offsets,
        n_buckets=5,
        method="expanding_quantile",
        min_history=min_history,
    )

    assert np.array_equal(base.labels[: cut + 1], pert.labels[: cut + 1])


def test_full_sample_quantile_would_differ() -> None:
    rng = np.random.default_rng(11)
    n_rows = 200
    feature = np.empty((n_rows, 1), dtype=np.float64)
    feature[:100, 0] = rng.normal(0.0, 1.0, size=100)
    feature[100:, 0] = rng.normal(10.0, 1.0, size=100)
    day_offsets = _single_session_day_offsets(n_rows)

    causal = expectancy.causal_buckets(
        feature,
        day_offsets,
        n_buckets=5,
        method="expanding_quantile",
        min_history=50,
    )
    causal_labels = causal.labels[:, 0]

    qs = np.quantile(feature[:, 0], [0.2, 0.4, 0.6, 0.8])
    naive = np.zeros(n_rows, dtype=np.int8)
    for i, val in enumerate(feature[:, 0]):
        naive[i] = np.searchsorted(qs, val, side="right")

    differs = any(
        causal_labels[i] != naive[i] for i in range(50, n_rows)
    )
    assert differs


def test_rows_before_min_history_are_unassigned_not_bucketed() -> None:
    rng = np.random.default_rng(12)
    n_rows, n_symbols = 100, 2
    feature = rng.normal(size=(n_rows, n_symbols))
    min_history = 50
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="expanding_quantile",
        min_history=min_history,
    )
    assert np.all(result.labels[:min_history] == -1)
    assert np.all(result.labels[min_history:] >= 0)


def test_cross_sectional_rank_is_causal_by_construction() -> None:
    rng = np.random.default_rng(13)
    n_rows, n_symbols = 30, 5
    feature = rng.normal(size=(n_rows, n_symbols))
    day_offsets = _single_session_day_offsets(n_rows)

    base = expectancy.causal_buckets(
        feature,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        min_history=0,
    )
    t = 15
    other_rows = np.arange(n_rows) != t
    perturbed = feature.copy()
    perturbed[other_rows, :] = rng.normal(
        loc=20.0, scale=10.0, size=(n_rows - 1, n_symbols)
    )
    pert = expectancy.causal_buckets(
        perturbed,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        min_history=0,
    )
    assert np.array_equal(base.labels[t], pert.labels[t])


def test_bucket_labels_are_balanced_asymptotically() -> None:
    rng = np.random.default_rng(14)
    n_rows = 3000
    feature = rng.uniform(size=(n_rows, 1))
    n_buckets = 5
    min_history = 50
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=n_buckets,
        method="expanding_quantile",
        min_history=min_history,
    )
    raw = result.labels[min_history:, 0]
    assigned = raw[raw >= 0]
    counts = np.bincount(assigned, minlength=n_buckets)
    expected = len(assigned) / n_buckets
    for b in range(n_buckets):
        assert counts[b] >= expected * 0.7
        assert counts[b] <= expected * 1.3


def test_constant_feature_degenerate_case() -> None:
    n_rows, n_symbols = 100, 2
    feature = np.full((n_rows, n_symbols), 42.0, dtype=np.float64)
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(n_rows),
        n_buckets=5,
        method="expanding_quantile",
        min_history=10,
    )
    assert result.labels.dtype == np.int8
    assert result.labels.shape == feature.shape
    first = result.labels[0, 0]
    assert np.all(result.labels == first)
    assert first == -1 or first >= 0


def test_all_nan_feature_yields_all_unassigned() -> None:
    feature = np.full((50, 2), np.nan, dtype=np.float64)
    result = expectancy.causal_buckets(
        feature,
        _single_session_day_offsets(50),
        n_buckets=5,
        method="expanding_quantile",
        min_history=20,
    )
    assert np.all(result.labels == -1)


def test_causal_decorator_probe_passes_at_full_strictness() -> None:
    rng = np.random.default_rng(17)
    n_rows, n_symbols = 300, 3
    feature = rng.normal(size=(n_rows, n_symbols))
    with guards.strictness(guards.Strictness.FULL):
        result = expectancy.causal_buckets(
            feature,
            _single_session_day_offsets(n_rows),
            n_buckets=5,
            method="expanding_quantile",
            min_history=50,
        )
    assert result.labels.shape == feature.shape


# ---------------------------------------------------------------------------
# overlap correction
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_naive_se_is_inflated_relative_to_block_bootstrap() -> None:
    rng = np.random.default_rng(18)
    n_rows, n_symbols = 8000, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=30)
    feature = rng.normal(size=(n_rows, n_symbols))

    table_naive = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        method="cross_sectional_rank",
        se_method="naive",
        n_boot=500,
        seed=18,
    )
    table_boot = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=500,
        seed=18,
    )
    ratio = abs(table_naive.spread_t) / abs(table_boot.spread_t)
    assert 2.0 <= ratio <= 9.0


@pytest.mark.parametrize("se_method", ["block_bootstrap", "non_overlapping"])
def test_n_effective_less_than_n_obs_when_horizon_gt_one(se_method: str) -> None:
    rng = np.random.default_rng(19)
    n_rows, n_symbols = 500, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=10)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method=se_method,
        n_boot=200,
        seed=19,
    )
    for stat in table.buckets:
        assert stat.n_effective < stat.n_obs


def test_non_overlapping_subsample_has_no_shared_bars() -> None:
    rng = np.random.default_rng(20)
    n_rows, n_symbols = 800, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    horizon = 10
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=horizon)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="non_overlapping",
        n_boot=100,
        seed=20,
    )
    for stat in table.buckets:
        assert stat.n_effective <= np.ceil(stat.n_obs / horizon) + 1e-9
        assert stat.n_effective > 0


@pytest.mark.slow
def test_block_bootstrap_blocks_never_straddle_a_session() -> None:
    rng = np.random.default_rng(21)
    lengths = [50, 60, 40, 70]
    day_offsets = _multi_session_day_offsets(lengths)
    total = sum(lengths)
    n_symbols = 8
    sigma = 0.002
    close = _driftless_gbm_close(rng, total, n_symbols, sigma)
    horizon = 30
    fwd = expectancy.forward_returns(close, day_offsets, horizon=horizon)
    feature = rng.normal(size=(total, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=200,
        seed=21,
    )
    for bucket in table.buckets:
        if bucket.n_obs > 0:
            assert np.isfinite(bucket.ci_low_bps)
            assert np.isfinite(bucket.ci_high_bps)
            assert np.isfinite(bucket.t_stat)


@pytest.mark.slow
def test_block_bootstrap_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(22)
    n_rows, n_symbols = 1000, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=5)
    feature = rng.normal(size=(n_rows, n_symbols))

    kw = dict(
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=300,
        seed=123,
    )
    t1 = expectancy.conditional_expectancy(feature, fwd, day_offsets, **kw)
    t2 = expectancy.conditional_expectancy(feature, fwd, day_offsets, **kw)
    pd.testing.assert_frame_equal(t1.to_frame(), t2.to_frame())

    kw2 = dict(kw, seed=456)
    t3 = expectancy.conditional_expectancy(feature, fwd, day_offsets, **kw2)
    df1 = t1.to_frame()
    df3 = t3.to_frame()
    assert not np.allclose(df1["ci_low_bps"].to_numpy(), df3["ci_low_bps"].to_numpy())
    assert not np.allclose(df1["ci_high_bps"].to_numpy(), df3["ci_high_bps"].to_numpy())


# SPEC-AMBIGUITY: spec says n_effective < n_obs only for horizon > 1; for horizon == 1 the
# strictest reading is n_effective == n_obs exactly
@pytest.mark.parametrize("se_method", ["block_bootstrap", "non_overlapping"])
def test_horizon_one_needs_no_correction(se_method: str) -> None:
    rng = np.random.default_rng(23)
    n_rows, n_symbols = 300, 6
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method=se_method,
        n_boot=100,
        seed=23,
    )
    for stat in table.buckets:
        assert stat.n_effective == pytest.approx(float(stat.n_obs))


# ---------------------------------------------------------------------------
# null acceptance
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pure_noise_yields_no_significant_bucket() -> None:
    rng = np.random.default_rng(24)
    n_rows, n_symbols = 6000, 8
    sigma = 0.002
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    feature_rng = np.random.default_rng(2401)
    feature = feature_rng.normal(size=(n_rows, n_symbols))

    finite = fwd.values[np.isfinite(fwd.values)]
    mean_ret = float(np.mean(finite))
    se = sigma / np.sqrt(finite.size)
    assert abs(mean_ret) < 4.0 * se

    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=500,
        seed=24,
    )
    # Per-bucket CIs for all 5 buckets are a multiple-comparisons error: on
    # correct code each CI contains zero with probability 0.95, so all five
    # do only with probability ~0.95^5 ~= 0.77. Assert instead on the spread,
    # the single tradeable quantity and a single statistical comparison.
    top = table.buckets[-1]
    bottom = table.buckets[0]
    top_se = top.std_bps / np.sqrt(max(1.0, top.n_effective))
    bottom_se = bottom.std_bps / np.sqrt(max(1.0, bottom.n_effective))
    spread_se = np.sqrt(top_se**2 + bottom_se**2)
    spread_ci_low = table.spread_bps - 1.96 * spread_se
    spread_ci_high = table.spread_bps + 1.96 * spread_se
    assert spread_ci_low <= 0.0 <= spread_ci_high
    assert table.survives_costs is False


@pytest.mark.slow
def test_null_false_positive_rate_is_near_nominal() -> None:
    """Strictly stronger null test than a single-seed bucket-containment check.

    A real leak, such as lookahead, inflates the CI-excludes-zero rate far above
    the nominal 5%. Across many independent seeds this is visible in the
    aggregate rate, while any single seed conflates ordinary sampling variation
    ("bad luck") with an actual defect and cannot statistically distinguish the
    two.
    """
    n_buckets = 5
    total_buckets = 0
    exclusions = 0

    for seed in range(1000, 1030):
        n_rows = 1000
        n_symbols = 8
        sigma = 0.002

        rng = np.random.default_rng(seed)
        close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma)
        day_offsets = _single_session_day_offsets(n_rows)
        fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

        feature_rng = np.random.default_rng(seed + 100000)
        feature = feature_rng.normal(size=(n_rows, n_symbols))

        table = expectancy.conditional_expectancy(
            feature,
            fwd,
            day_offsets,
            n_buckets=n_buckets,
            method="cross_sectional_rank",
            se_method="block_bootstrap",
            n_boot=200,
            seed=seed,
        )

        for bucket in table.buckets:
            if bucket.ci_low_bps > 0.0 or bucket.ci_high_bps < 0.0:
                exclusions += 1
            total_buckets += 1

    rate = exclusions / total_buckets
    assert 0.0 <= rate <= 0.12


def test_planted_effect_is_recovered() -> None:
    """Test planted top-bottom spread is recovered in a single combined CI.

    The previous version asserted CI-containment separately for the top and
    bottom bucket means. Each individual CI has ~95% coverage under correct
    code, so two checks would pass jointly only ~0.95^2 ~= 0.9025 of the time.
    This is a milder version of the same multiple-comparisons flaw already
    fixed for the pure-noise test elsewhere in this file, where all five
    bucket CIs were checked separately.

    Instead, this test constructs a single 95% CI for the recovered spread and
    asserts that it contains the true planted spread of 100.0 bps. At this
    sample size the spread CI half-width is roughly 1 bps (each edge bucket's
    own half-width is ~0.55 bps, combined in quadrature), so a CI centered
    near zero cannot reach up to 100 bps. Thus the assertion would still
    catch a gross regression, e.g. a no-op conditional_expectancy returning
    spread_bps near zero, or any module regression that recovers less than
    roughly half the planted 100 bps spread.
    """
    rng = np.random.default_rng(25)
    n_rows, n_symbols = 5000, 5
    sigma = 0.002
    fwd, feature = _build_horizon_one_fwd_with_effect(
        n_rows,
        n_symbols,
        effect_top_bps=50.0,
        effect_bottom_bps=-50.0,
        rng=rng,
        sigma=sigma,
    )
    day_offsets = _single_session_day_offsets(n_rows)
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=300,
        seed=25,
    )
    top = table.buckets[-1]
    bottom = table.buckets[0]
    top_se = top.std_bps / np.sqrt(max(1.0, top.n_effective))
    bottom_se = bottom.std_bps / np.sqrt(max(1.0, bottom.n_effective))
    spread_se = np.sqrt(top_se**2 + bottom_se**2)
    spread_ci_low = table.spread_bps - 1.96 * spread_se
    spread_ci_high = table.spread_bps + 1.96 * spread_se
    assert spread_ci_low <= 100.0 <= spread_ci_high


def test_planted_effect_below_cost_hurdle_does_not_survive_costs() -> None:
    rng = np.random.default_rng(26)
    n_rows, n_symbols = 10000, 5
    sigma = 0.002
    fwd, feature = _build_horizon_one_fwd_with_effect(
        n_rows,
        n_symbols,
        effect_top_bps=5.0,
        effect_bottom_bps=-5.0,
        rng=rng,
        sigma=sigma,
    )
    day_offsets = _single_session_day_offsets(n_rows)
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=300,
        seed=26,
        cost_hurdle_bps=8.3,
    )
    assert table.survives_costs is False
    top = table.buckets[-1]
    bottom = table.buckets[0]
    assert top.ci_low_bps > 0.0
    assert bottom.ci_high_bps < 0.0


# ---------------------------------------------------------------------------
# ExpectancyTable
# ---------------------------------------------------------------------------

# SPEC-AMBIGUITY: empty-bucket representation unspecified; construct fixture guaranteeing
# non-empty buckets to sidestep the ambiguity
def test_spread_is_top_minus_bottom_bucket() -> None:
    rng = np.random.default_rng(27)
    n_rows, n_symbols = 1000, 5
    sigma = 0.002
    fwd, feature = _build_horizon_one_fwd_with_effect(
        n_rows,
        n_symbols,
        effect_top_bps=10.0,
        effect_bottom_bps=-10.0,
        rng=rng,
        sigma=sigma,
    )
    day_offsets = _single_session_day_offsets(n_rows)
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
        seed=27,
    )
    assert [b.bucket for b in table.buckets] == list(range(5))
    assert table.spread_bps == pytest.approx(
        table.buckets[-1].mean_bps - table.buckets[0].mean_bps
    )


@pytest.mark.parametrize(
    "spread_bps, cost_hurdle_bps, expected",
    [(15.0, 10.0, False), (30.0, 10.0, True)],
)
def test_survives_costs_requires_two_x_hurdle(
    spread_bps: float, cost_hurdle_bps: float, expected: bool
) -> None:
    rng = np.random.default_rng(28)
    n_rows, n_symbols = 2000, 5
    sigma = 0.001
    fwd, feature = _build_horizon_one_fwd_with_effect(
        n_rows,
        n_symbols,
        effect_top_bps=spread_bps / 2.0,
        effect_bottom_bps=-spread_bps / 2.0,
        rng=rng,
        sigma=sigma,
    )
    day_offsets = _single_session_day_offsets(n_rows)
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
        cost_hurdle_bps=cost_hurdle_bps,
    )
    assert table.survives_costs is expected


def test_cost_hurdle_defaults_to_nse_intraday_round_trip() -> None:
    rng = np.random.default_rng(29)
    n_rows, n_symbols = 500, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
        cost_hurdle_bps=None,
    )
    expected = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert table.cost_hurdle_bps == pytest.approx(expected)


def test_explain_names_se_method_overlap_correction_and_hurdle() -> None:
    rng = np.random.default_rng(30)
    n_rows, n_symbols = 200, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=50,
        cost_hurdle_bps=12.5,
    )
    text = table.explain()
    lowered = text.lower()
    assert "block_bootstrap" in lowered
    assert "overlap" in lowered or "correction" in lowered
    assert "12.5" in text
    assert "hurdle" in lowered
    assert "bps" in lowered


def test_explain_warns_loudly_when_se_method_is_naive() -> None:
    rng = np.random.default_rng(31)
    n_rows, n_symbols = 200, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
    )
    text = table.explain()
    # Check for at least one loud marker; implementation may choose wording.
    loud_markers = ["NAIVE", "WARNING", "UNCORRECTED"]
    assert any(marker in text for marker in loud_markers)


def test_to_frame_schema_and_row_order() -> None:
    rng = np.random.default_rng(32)
    n_rows, n_symbols = 300, 5
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(n_rows, n_symbols))
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
    )
    df = table.to_frame()
    assert isinstance(df, pd.DataFrame)
    required = {
        "bucket",
        "n_obs",
        "n_effective",
        "mean_bps",
        "median_bps",
        "std_bps",
        "t_stat",
        "se_method",
        "ci_low_bps",
        "ci_high_bps",
    }
    assert required.issubset(df.columns)
    assert df["bucket"].tolist() == sorted(df["bucket"].tolist())
    assert len(df) == len(table.buckets)


# ---------------------------------------------------------------------------
# stability decomposition
# ---------------------------------------------------------------------------

# SPEC-AMBIGUITY: expectancy_by_year's full signature (dates array, dict[int, ExpectancyTable]
# return keyed by calendar year) is not given verbatim by the spec and was inferred.
def test_expectancy_by_year_partitions_observations_exactly() -> None:
    rng = np.random.default_rng(33)
    sessions = [300, 300, 300, 300]
    day_offsets = _multi_session_day_offsets(sessions)
    total = sum(sessions)
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 7, 1),
        datetime.date(2025, 1, 1),
        datetime.date(2025, 7, 1),
    ]
    dates_arr = np.array(dates, dtype=object)
    n_symbols = 5
    sigma = 0.002
    close = _driftless_gbm_close(rng, total, n_symbols, sigma)
    horizon = 2
    fwd = expectancy.forward_returns(close, day_offsets, horizon)
    feature = rng.normal(size=(total, n_symbols))

    by_year = expectancy.expectancy_by_year(
        feature,
        fwd,
        day_offsets,
        dates_arr,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
    )
    overall = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="naive",
    )
    assert sum(t.n_total for t in by_year.values()) == overall.n_total


def test_effect_confined_to_one_year_is_visible_in_by_year() -> None:
    rng = np.random.default_rng(34)
    sessions = [300, 300, 300, 300]
    day_offsets = _multi_session_day_offsets(sessions)
    total = sum(sessions)
    n_symbols = 5
    sigma = 0.002
    feature = np.tile(np.arange(n_symbols, dtype=np.float64), (total, 1))
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 7, 1),
        datetime.date(2025, 1, 1),
        datetime.date(2025, 7, 1),
    ]
    dates_arr = np.array(dates, dtype=object)

    session_idx = np.searchsorted(day_offsets[1:], np.arange(total - 1), side="right")
    year_of_row = np.array([dates_arr[i].year for i in session_idx], dtype=int)

    one_step = np.empty((total - 1, n_symbols), dtype=np.float64)
    for sym in range(n_symbols):
        if sym == n_symbols - 1:
            mean = np.where(year_of_row == 2025, 50.0 / 1e4, 0.0)
        elif sym == 0:
            mean = np.where(year_of_row == 2025, -50.0 / 1e4, 0.0)
        else:
            mean = np.zeros(total - 1, dtype=np.float64)
        one_step[:, sym] = rng.normal(loc=mean, scale=sigma, size=total - 1)
    close = _close_from_one_step_log_returns(one_step)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    by_year = expectancy.expectancy_by_year(
        feature,
        fwd,
        day_offsets,
        dates_arr,
        n_buckets=5,
        method="cross_sectional_rank",
        se_method="block_bootstrap",
        n_boot=200,
        seed=34,
    )

    assert 2025 in by_year
    assert 2024 in by_year
    t2025 = by_year[2025]
    assert t2025.survives_costs is True
    assert t2025.buckets[-1].ci_low_bps > 0.0

    t2024 = by_year[2024]
    t2024_top = t2024.buckets[-1]
    t2024_bottom = t2024.buckets[0]
    t2024_top_se = t2024_top.std_bps / np.sqrt(max(1.0, t2024_top.n_effective))
    t2024_bottom_se = t2024_bottom.std_bps / np.sqrt(max(1.0, t2024_bottom.n_effective))
    t2024_spread_se = np.sqrt(t2024_top_se**2 + t2024_bottom_se**2)
    t2024_spread_ci_low = t2024.spread_bps - 1.96 * t2024_spread_se
    t2024_spread_ci_high = t2024.spread_bps + 1.96 * t2024_spread_se
    # Replaces a 5-way per-bucket multiple-comparisons check with a single spread-CI check,
    # matching tests 24 and 25 elsewhere in the file.
    assert t2024_spread_ci_low <= 0.0 <= t2024_spread_ci_high


# SPEC-AMBIGUITY: expectancy_by_time_of_day's full signature (minute_of_day array,
# time_bucket_minutes, dict[int, ExpectancyTable] return keyed by bucket floor) is not given
# verbatim by the spec and was inferred.
def test_by_time_of_day_uses_minute_of_day_not_row_index() -> None:
    rng = np.random.default_rng(35)
    n_rows1, n_rows2 = 200, 200
    total = n_rows1 + n_rows2
    day_offsets = _multi_session_day_offsets([n_rows1, n_rows2])
    n_symbols = 5
    sigma = 0.002
    close = _driftless_gbm_close(rng, total, n_symbols, sigma)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(total, n_symbols))

    minute_of_day = np.empty(total, dtype=int)
    for i in range(total):
        if i < n_rows1:
            minute_of_day[i] = 555 + i
        else:
            minute_of_day[i] = 1080 + (i - n_rows1)

    result = expectancy.expectancy_by_time_of_day(
        feature,
        fwd,
        day_offsets,
        minute_of_day,
        n_buckets=5,
        time_bucket_minutes=60,
        method="cross_sectional_rank",
        se_method="naive",
        seed=35,
    )
    keys = set(result.keys())
    assert 540 in keys
    assert 1080 in keys
    assert 0 not in keys
    for k in keys:
        assert k % 60 == 0


def test_by_liquidity_decile_uses_strictly_prior_adv() -> None:
    # SPEC-AMBIGUITY: expectancy_by_liquidity_decile's full signature (prior_adv array,
    # dict[int, ExpectancyTable] return keyed by decile 0-9) is not given verbatim by the
    # spec and was inferred; "strictly prior" is treated as a caller-side data contract
    # (the function trusts prior_adv as already lagged) rather than something the function
    # itself re-derives or causal-probes.
    #
    # SPEC-AMBIGUITY: returned decile tables aggregate all sessions; row-level assignments
    # are not exposed. Build prior_adv with unique values per row so each of 10 deciles
    # receives one symbol per row, then verify aggregate counts.
    rng = np.random.default_rng(36)
    n_symbols = 10
    n_rows1, n_rows2 = 150, 150
    total = n_rows1 + n_rows2
    day_offsets = _multi_session_day_offsets([n_rows1, n_rows2])
    prior_adv = np.tile(np.arange(n_symbols, dtype=np.float64), (total, 1))

    sigma = 0.002
    close = _driftless_gbm_close(rng, total, n_symbols, sigma)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature = rng.normal(size=(total, n_symbols))

    result = expectancy.expectancy_by_liquidity_decile(
        feature,
        fwd,
        day_offsets,
        prior_adv,
        n_buckets=10,
        method="cross_sectional_rank",
        se_method="naive",
        seed=36,
    )
    assert set(result.keys()) == set(range(10))
    defined_rows = total - 2  # two sessions each lose last row to horizon=1 tail
    for k in range(10):
        assert result[k].n_total == defined_rows
    total_observations = defined_rows * n_symbols
    assert sum(result[k].n_total for k in range(10)) == total_observations

    # Perturb later session prior_adv by shuffling values within each of its rows.
    prior_adv_pert = prior_adv.copy()
    for r in range(n_rows1, total):
        prior_adv_pert[r, :] = rng.permutation(prior_adv_pert[r, :])
    result_pert = expectancy.expectancy_by_liquidity_decile(
        feature,
        fwd,
        day_offsets,
        prior_adv_pert,
        n_buckets=10,
        method="cross_sectional_rank",
        se_method="naive",
        seed=36,
    )
    assert set(result_pert.keys()) == set(range(10))
    for k in range(10):
        assert result_pert[k].n_total == defined_rows


# ---------------------------------------------------------------------------
# double_sort
# ---------------------------------------------------------------------------

# SPEC-AMBIGUITY: double_sort's and SortResult's full signatures (n_buckets_a/b,
# thin_cell_threshold, cells as a 2-D tuple of BucketStat, thin_cells as (i, j) index pairs)
# are not given verbatim by the spec and were inferred.
# SPEC-AMBIGUITY: empty-cell representation unspecified; construct fixture where every cell
# is guaranteed non-empty
def test_double_sort_cell_counts_sum_to_total() -> None:
    rng = np.random.default_rng(37)
    n_rows, n_symbols = 500, 9
    n_buckets_a = 3
    n_buckets_b = 3
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    # Feature values guarantee each of the 9 (a,b) combinations receives one symbol per row.
    feature_a = np.tile(np.arange(n_symbols, dtype=np.float64), (n_rows, 1))
    feature_b = np.tile(
        np.array([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=np.float64), (n_rows, 1)
    )

    result = expectancy.double_sort(
        feature_a,
        feature_b,
        fwd,
        day_offsets,
        n_buckets_a=n_buckets_a,
        n_buckets_b=n_buckets_b,
        method="cross_sectional_rank",
        se_method="naive",
        thin_cell_threshold=30,
    )
    total_obs = sum(cell.n_obs for row in result.cells for cell in row)
    assert total_obs == result.n_total
    assert result.n_buckets_a == n_buckets_a
    assert result.n_buckets_b == n_buckets_b
    assert len(result.cells) == n_buckets_a
    for row in result.cells:
        assert len(row) == n_buckets_b


def test_double_sort_reports_thin_cells() -> None:
    rng = np.random.default_rng(38)
    n_rows, n_symbols = 200, 100
    n_buckets_a = 3
    n_buckets_b = 3
    close = _driftless_gbm_close(rng, n_rows, n_symbols, sigma=0.002)
    day_offsets = _single_session_day_offsets(n_rows)
    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)

    base = rng.normal(size=(n_rows, n_symbols))
    feature_a = base
    feature_b = base + rng.normal(scale=0.01, size=(n_rows, n_symbols))

    result = expectancy.double_sort(
        feature_a,
        feature_b,
        fwd,
        day_offsets,
        n_buckets_a=n_buckets_a,
        n_buckets_b=n_buckets_b,
        method="cross_sectional_rank",
        se_method="naive",
        thin_cell_threshold=30,
    )
    assert len(result.thin_cells) > 0
    for (i, j) in result.thin_cells:
        assert result.cells[i][j].n_obs < result.thin_cell_threshold
