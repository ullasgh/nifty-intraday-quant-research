from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.features import core as core_features
from nifty_quant.research import expectancy
from nifty_quant.research.lens import (
    Feature,
    FeatureKindError,
    HypothesisVerdict,
    Lens,
    StabilityReport,
)

_IST = ZoneInfo("Asia/Kolkata")
_N_SYMBOLS = 5
_SYMBOLS = ("S00", "S01", "S02", "S03", "S04")
_DEFAULT_VOLUME = (1e3, 1e4, 1e5, 1e6, 1e7)
_ALL_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
_ALL_SIGNAL_YEARS = set(_ALL_YEARS)
_PASS_LATENCY = {0: 1.0, 1: 0.6, 2: 0.55}
_FAIL_LATENCY = {0: 1.0, 1: 0.6, 2: 0.4}

# AMENDMENT 2 (specs/lens_verdict_integrity.md): criterion 7 needs at least two
# COMPLETE calendar years (all 12 months present), which `_build_panel`'s plain
# "N sessions in January" date grid can never produce regardless of how many
# distinct years it spans. These two years get one session per month instead
# (see `full_year_span` on `_build_panel`) so criterion 7 actually evaluates.
_FULL_YEAR_COMPLETE_YEARS: frozenset[int] = frozenset({2024, 2025})
_FULL_YEAR_SESSIONS_PER_YEAR: dict[int, int] = {
    year: (12 if year in _FULL_YEAR_COMPLETE_YEARS else 2) for year in _ALL_YEARS
}

# AMENDMENT 2: deterministic strategy_returns for criterion 6 (deflated Sharpe).
# sr0 = expected_max_sharpe(2, var_trial_sharpes=1.0) ~= 0.5198; this array's
# raw per-period SR is ~= 4.3, so deflated_sharpe saturates to 1.0 -- well
# above DSR_SIGNIFICANCE=0.95 -- deterministically (fixed seed), not by luck.
_PASS_STRATEGY_RETURNS: np.ndarray = np.random.default_rng(0).normal(
    0.002, 0.0006, size=60
)


def _session_grid(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_chunks: list[np.ndarray] = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bars_per_session, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    n_days = len(dates)
    day_offsets = np.arange(0, (n_days + 1) * bars_per_session, bars_per_session, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _signal_means(effect: float, n_symbols: int = _N_SYMBOLS) -> np.ndarray:
    # linspace(-effect, effect, 5) == [-effect, -effect/2, 0, effect/2, effect],
    # i.e. this is an exact generalization of the original 5-symbol literal.
    return np.linspace(-effect, effect, n_symbols, dtype=np.float64)


def _close_prices_from_log_returns(returns: np.ndarray) -> np.ndarray:
    cum = np.cumsum(returns, axis=0)
    log_close = np.vstack([np.zeros((1, returns.shape[1]), dtype=np.float64), cum])
    close_all = np.exp(log_close)
    return close_all[:-1, :]


def _build_panel(
    years: tuple[int, ...],
    sessions_per_year: int | dict[int, int] = 2,
    bars_per_session: int = 25,
    signal_years: set[int] | None = None,
    *,
    effect: float = 0.01,
    sigma: float = 0.0005,
    flat_sigma: float = 0.0,
    effect_window_bars: tuple[int, int] | None = None,
    volume_profile: tuple[float, ...] = _DEFAULT_VOLUME,
    seed: int = 0,
    symbols: tuple[str, ...] = _SYMBOLS,
    full_year_span: frozenset[int] | set[int] | None = None,
) -> Panel:
    """Build a synthetic Panel fixture.

    `full_year_span` (AMENDMENT 2, specs/lens_verdict_integrity.md): a set of
    years for which sessions are placed one-per-month (12 sessions required for
    that year) instead of consecutive January days, so criterion 7's
    "two complete calendar years" check can actually evaluate. Years not in
    this set keep the original January-only placement -- existing callers that
    never pass `full_year_span` get byte-identical dates to before this change.
    """
    if signal_years is None:
        signal_years = set(years)

    n_symbols = len(symbols)
    if len(volume_profile) != n_symbols:
        raise ValueError(
            f"volume_profile has {len(volume_profile)} entries, expected "
            f"{n_symbols} to match symbols"
        )

    if isinstance(sessions_per_year, int):
        year_sessions = {year: sessions_per_year for year in years}
    else:
        year_sessions = dict(sessions_per_year)

    dates: list[dt.date] = []
    for year in years:
        n_sessions = year_sessions[year]
        if full_year_span and year in full_year_span:
            if n_sessions != 12:
                raise ValueError(
                    f"full_year_span requires exactly 12 sessions for year "
                    f"{year} (one per month), got {n_sessions}"
                )
            for month in range(1, 13):
                dates.append(dt.date(year, month, 2))
        else:
            for session_idx in range(n_sessions):
                dates.append(dt.date(year, 1, 2 + session_idx))

    n_rows = len(dates) * bars_per_session
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_rows, n_symbols), dtype=np.float64)

    row = 0
    for date in dates:
        year = date.year
        if year in signal_years:
            means = _signal_means(effect, n_symbols)
            if effect_window_bars is None:
                block_means = np.tile(means, (bars_per_session, 1))
            else:
                start, end = effect_window_bars
                block_means = np.zeros((bars_per_session, n_symbols), dtype=np.float64)
                if end > start:
                    block_means[start:end, :] = means
            block_sigma = sigma
        else:
            block_means = np.zeros((bars_per_session, n_symbols), dtype=np.float64)
            block_sigma = flat_sigma

        noise = rng.normal(0.0, block_sigma, size=(bars_per_session, n_symbols))
        returns[row : row + bars_per_session] = block_means + noise
        row += bars_per_session

    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.array(volume_profile, dtype=np.float64), (n_rows, 1))

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _build_small_panel(seed: int = 0) -> Panel:
    return _build_panel(
        (2024,),
        sessions_per_year=5,
        bars_per_session=30,
        signal_years={2024},
        effect=0.001,
        sigma=0.002,
        seed=seed,
    )


def _random_feature_values(n_rows: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(n_rows, _N_SYMBOLS))


def _call_verdict(
    panel: Panel,
    hypothesis_id: str = "H001_close_reversion",
    *,
    latency_profile: dict[int, float] | None = _PASS_LATENCY,
    # AMENDMENT 2: default to a real, PASS-producing criterion-6 evaluation
    # (real effective_n_trials + real strategy_returns) instead of the old
    # effective_n_trials=1/strategy_returns=None combination that always left
    # criterion 6 NOT_EVALUATED. Callers exercising criterion 6 directly still
    # override both explicitly.
    effective_n_trials: int = 2,
    strategy_returns: np.ndarray | None = _PASS_STRATEGY_RETURNS,
    seed: int = 0,
    horizon: int = 1,
) -> HypothesisVerdict:
    lens = Lens(panel, seed=seed)
    return lens.verdict(
        hypothesis_id,
        "return_1",
        horizon,
        latency_profile=latency_profile,
        effective_n_trials=effective_n_trials,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )


def _assert_tables_summary_equal(
    table_lens: expectancy.ExpectancyTable, table_direct: expectancy.ExpectancyTable
) -> None:
    assert table_lens.horizon == table_direct.horizon
    assert table_lens.feature_name == table_direct.feature_name
    assert table_lens.n_total == table_direct.n_total
    np.testing.assert_allclose(table_lens.cost_hurdle_bps, table_direct.cost_hurdle_bps)
    np.testing.assert_allclose(table_lens.spread_bps, table_direct.spread_bps)
    np.testing.assert_allclose(table_lens.spread_t, table_direct.spread_t)
    assert table_lens.survives_costs == table_direct.survives_costs
    assert len(table_lens.buckets) == len(table_direct.buckets)


def _assert_bucket_equal(
    bucket_lens: expectancy.BucketStat, bucket_direct: expectancy.BucketStat
) -> None:
    assert bucket_lens.bucket == bucket_direct.bucket
    assert bucket_lens.n_obs == bucket_direct.n_obs
    np.testing.assert_allclose(bucket_lens.n_effective, bucket_direct.n_effective)
    np.testing.assert_allclose(bucket_lens.mean_bps, bucket_direct.mean_bps)
    np.testing.assert_allclose(bucket_lens.median_bps, bucket_direct.median_bps)
    np.testing.assert_allclose(bucket_lens.std_bps, bucket_direct.std_bps)
    np.testing.assert_allclose(bucket_lens.t_stat, bucket_direct.t_stat)
    assert bucket_lens.se_method == bucket_direct.se_method
    np.testing.assert_allclose(bucket_lens.ci_low_bps, bucket_direct.ci_low_bps)
    np.testing.assert_allclose(bucket_lens.ci_high_bps, bucket_direct.ci_high_bps)
    assert bucket_lens.block_indices == bucket_direct.block_indices


def _all_pass_panel(seed: int = 0) -> Panel:
    return _build_panel(
        _ALL_YEARS,
        sessions_per_year=_FULL_YEAR_SESSIONS_PER_YEAR,
        bars_per_session=240,
        signal_years=_ALL_SIGNAL_YEARS,
        effect=0.01,
        sigma=0.0005,
        seed=seed,
        full_year_span=_FULL_YEAR_COMPLETE_YEARS,
    )


def _make_all_pass_verdict() -> HypothesisVerdict:
    return _call_verdict(_all_pass_panel(), hypothesis_id="H001_close_reversion")


def _make_killed_verdict() -> HypothesisVerdict:
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    return _call_verdict(panel, hypothesis_id="H_KILLED")


# Construction / wiring


def test_construction_wiring_explicit_universe_and_cost_model() -> None:
    # AMENDMENT 2 (specs/lens_verdict_integrity.md): the original version of
    # this test asserted `lens.panel is panel` and `lens.symbols ==
    # panel.symbols` while passing a restricted `universe` -- i.e. it asserted
    # that restriction had NO effect, which is the L4 defect as intended
    # behaviour. Rewritten to assert restriction DID take effect, using >= 5
    # symbols in both the full (8) and restricted (5) sets per the min_names=5
    # trap (cross_sectional_rank silently returns all-NaN below 5 names).
    full_symbols = tuple(f"S{i:02d}" for i in range(8))
    restricted_universe = tuple(f"S{i:02d}" for i in (1, 2, 3, 4, 5))
    volume_profile = tuple(1e4 for _ in range(8))
    panel = _build_panel(
        (2024,),
        sessions_per_year=5,
        bars_per_session=30,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
        seed=0,
        symbols=full_symbols,
        volume_profile=volume_profile,
    )
    cost_model = NSEIntradayEquityCosts(brokerage_flat=15.0)
    lens_full = Lens(panel, cost_model=cost_model, seed=42)
    lens_restricted = Lens(panel, universe=restricted_universe, cost_model=cost_model, seed=42)

    # Restriction happened once at construction (L4): the restricted Lens no
    # longer holds the original unrestricted panel, and its recorded symbol
    # count and universe reflect the restriction.
    assert lens_restricted.panel is not panel
    assert lens_restricted.n_symbols_used == 5
    assert lens_full.n_symbols_used == 8
    assert lens_restricted.symbols == restricted_universe
    assert lens_restricted.universe == restricted_universe
    assert lens_restricted.seed == 42
    assert lens_restricted.cost_model is cost_model
    # Row/day structure is unaffected by a symbol-axis restriction.
    assert np.array_equal(lens_restricted.day_offsets, panel.day_offsets)
    assert np.array_equal(lens_restricted.minute_of_day, panel.minute_of_day())

    # The restriction is not merely recorded, it visibly changes the computed
    # answer: excluding S00, S06, S07 (which carry the most extreme signal
    # means) from the top/bottom quintile buckets changes spread_bps. A test
    # that passed whether or not restriction happened would prove nothing.
    kwargs = dict(method="cross_sectional_rank", n_buckets=5, n_boot=100)
    table_full = lens_full.expectancy("return_1", 1, **kwargs)
    table_restricted = lens_restricted.expectancy("return_1", 1, **kwargs)
    assert table_full.spread_bps != table_restricted.spread_bps


def test_construction_defaults_universe_cost_model_seed() -> None:
    panel = _build_small_panel(seed=1)
    lens = Lens(panel)

    assert lens.seed == 0
    assert lens.universe == panel.symbols
    assert isinstance(lens.cost_model, NSEIntradayEquityCosts)
    assert lens.panel is panel
    assert np.array_equal(lens.day_offsets, panel.day_offsets)
    assert np.array_equal(lens.minute_of_day, panel.minute_of_day())


def test_available_features_non_empty_and_contains_builtins() -> None:
    lens = Lens(_build_small_panel(seed=2))
    available = lens.available_features()

    assert isinstance(available, tuple)
    assert len(available) > 0
    assert {"close", "return_1", "volume_zscore"}.issubset(set(available))

    with pytest.raises(KeyError, match="not_a_feature"):
        lens.feature("not_a_feature")


def test_feature_memoization_identical_params_same_object_and_distinct_params_differ() -> None:
    lens = Lens(_build_small_panel(seed=3))

    f1 = lens.feature("return_1")
    f2 = lens.feature("return_1")
    assert f1 is f2

    f_v1 = lens.feature("volume_zscore", window=20)
    f_v2 = lens.feature("volume_zscore", window=20)
    assert f_v1 is f_v2

    f_v3 = lens.feature("volume_zscore", window=30)
    assert f_v1 is not f_v3
    assert not np.array_equal(f_v1.values, f_v3.values)


# Feature objects


def test_feature_close_kind_and_warmup() -> None:
    panel = _build_small_panel(seed=4)
    lens = Lens(panel)

    feature = lens.feature("close")

    assert feature.name == "close"
    assert feature.kind == "level"
    assert feature.warmup_bars == 0
    assert feature.params == {}
    np.testing.assert_array_equal(feature.values, panel.field("close").astype(np.float64))


def test_feature_return_1_kind_warmup_and_values() -> None:
    panel = _build_small_panel(seed=5)
    lens = Lens(panel)

    feature = lens.feature("return_1")

    assert feature.name == "return_1"
    assert feature.kind == "return"
    assert feature.warmup_bars == 1
    assert feature.params == {}
    expected = core_features.log_returns(
        panel.field("close").astype(np.float64), day_offsets=panel.day_offsets
    )
    np.testing.assert_array_equal(feature.values, expected)


def test_feature_volume_zscore_params_and_values() -> None:
    panel = _build_small_panel(seed=6)
    lens = Lens(panel)

    feature = lens.feature("volume_zscore", window=20)

    assert feature.name == "volume_zscore"
    assert feature.kind == "ratio"
    assert feature.warmup_bars == 20
    assert feature.params == {"window": 20, "deseasonalize": True}

    expected = core_features.volume_zscore(
        panel.field("volume").astype(np.float64),
        panel.minute_of_day(),
        window=20,
        deseasonalize=True,
        day_offsets=panel.day_offsets,
    )
    np.testing.assert_array_equal(feature.values, expected)


def test_feature_explain_names_params() -> None:
    lens = Lens(_build_small_panel(seed=7))
    feature = lens.feature("volume_zscore", window=20)

    text = feature.explain()

    assert "volume_zscore" in text
    assert "ratio" in text
    assert "window" in text
    assert "20" in text


def test_lens_expectancy_level_feature_raises_feature_kind_error() -> None:
    lens = Lens(_build_small_panel(seed=8))
    with pytest.raises(FeatureKindError) as excinfo:
        lens.expectancy(lens.feature("close"), horizon=1)

    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


def test_lens_stability_level_feature_raises_feature_kind_error() -> None:
    lens = Lens(_build_small_panel(seed=9))
    with pytest.raises(FeatureKindError) as excinfo:
        lens.stability(lens.feature("close"), horizon=1)

    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


def test_lens_verdict_level_feature_raises_feature_kind_error() -> None:
    lens = Lens(_build_small_panel(seed=10))
    with pytest.raises(FeatureKindError) as excinfo:
        lens.verdict("H_lev", lens.feature("close"), 1)

    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


# Delegation


def test_lens_expectancy_delegates_with_default_kwargs() -> None:
    panel = _build_small_panel(seed=11)
    lens = Lens(panel)
    feature = Feature(
        name="return_1",
        values=_random_feature_values(panel.n_rows(), seed=1234),
        kind="return",
        warmup_bars=1,
        params={},
    )

    table_lens = lens.expectancy(
        feature, horizon=1, method="cross_sectional_rank", n_buckets=5, se_method="naive"
    )

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    table_direct = expectancy.conditional_expectancy(
        feature.values,
        fwd,
        panel.day_offsets,
        method="cross_sectional_rank",
        n_buckets=5,
        se_method="naive",
        feature_name=feature.name,
    )

    _assert_tables_summary_equal(table_lens, table_direct)


def test_lens_expectancy_delegates_with_custom_kwargs() -> None:
    panel = _build_small_panel(seed=12)
    lens = Lens(panel)
    feature = Feature(
        name="return_1",
        values=_random_feature_values(panel.n_rows(), seed=1235),
        kind="return",
        warmup_bars=1,
        params={},
    )

    kwargs = {
        "method": "cross_sectional_rank",
        "n_buckets": 5,
        "se_method": "naive",
        "seed": 7,
        "n_boot": 100,
    }

    table_lens = lens.expectancy(feature, horizon=1, **kwargs)

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    table_direct = expectancy.conditional_expectancy(
        feature.values, fwd, panel.day_offsets, feature_name=feature.name, **kwargs
    )

    _assert_tables_summary_equal(table_lens, table_direct)


def test_lens_expectancy_honors_explicit_cost_hurdle_bps() -> None:
    panel = _build_small_panel(seed=13)
    lens = Lens(panel)
    feature = Feature(
        name="return_1",
        values=_random_feature_values(panel.n_rows(), seed=1236),
        kind="return",
        warmup_bars=1,
        params={},
    )

    table_lens = lens.expectancy(
        feature,
        horizon=1,
        method="cross_sectional_rank",
        n_buckets=5,
        se_method="naive",
        cost_hurdle_bps=12.3,
    )

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    table_direct = expectancy.conditional_expectancy(
        feature.values,
        fwd,
        panel.day_offsets,
        method="cross_sectional_rank",
        n_buckets=5,
        se_method="naive",
        cost_hurdle_bps=12.3,
        feature_name=feature.name,
    )

    np.testing.assert_allclose(table_lens.cost_hurdle_bps, 12.3)
    np.testing.assert_allclose(table_direct.cost_hurdle_bps, 12.3)
    _assert_tables_summary_equal(table_lens, table_direct)


def test_lens_expectancy_bucket_stat_fields_match_direct() -> None:
    panel = _build_small_panel(seed=14)
    lens = Lens(panel)
    feature = Feature(
        name="return_1",
        values=_random_feature_values(panel.n_rows(), seed=1237),
        kind="return",
        warmup_bars=1,
        params={},
    )

    table_lens = lens.expectancy(
        feature, horizon=1, method="cross_sectional_rank", n_buckets=5, se_method="naive"
    )

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    table_direct = expectancy.conditional_expectancy(
        feature.values,
        fwd,
        panel.day_offsets,
        method="cross_sectional_rank",
        n_buckets=5,
        se_method="naive",
    )

    assert len(table_lens.buckets) == len(table_direct.buckets)
    for bucket_lens, bucket_direct in zip(table_lens.buckets, table_direct.buckets):
        _assert_bucket_equal(bucket_lens, bucket_direct)


# Stability


def test_stability_returns_all_decompositions_with_year_keys() -> None:
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=2,
        bars_per_session=25,
        signal_years=_ALL_SIGNAL_YEARS,
        effect=0.01,
        sigma=0.0005,
    )
    lens = Lens(panel)

    report = lens.stability(lens.feature("return_1"), horizon=1)

    assert isinstance(report, StabilityReport)
    assert len(report.by_year) == len(_ALL_YEARS)
    assert set(report.by_year) == set(_ALL_YEARS)
    assert len(report.by_time_of_day) > 0
    assert len(report.by_liquidity_decile) > 0


@pytest.mark.parametrize(
    "signal_years, expected_stable",
    [
        ({2018, 2019, 2020, 2021, 2022, 2023}, True),
        ({2018, 2019, 2020}, False),
    ],
    ids=["six_of_eight_stable", "three_of_eight_unstable"],
)
def test_stability_sign_rule_6_of_8_years(signal_years: set[int], expected_stable: bool) -> None:
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=2,
        bars_per_session=25,
        signal_years=signal_years,
        effect=0.01,
        sigma=0.0005,
    )
    lens = Lens(panel)

    report = lens.stability(lens.feature("return_1"), horizon=1)

    assert report.sign_stable is expected_stable
    if expected_stable:
        assert report.n_years_sign_consistent >= 6
    else:
        assert report.n_years_sign_consistent < 6


def test_stability_single_year_effect_reported_unstable() -> None:
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    lens = Lens(panel)

    report = lens.stability(lens.feature("return_1"), horizon=1)

    assert report.n_years_total == 1
    assert report.sign_stable is False


def test_stability_thin_year_included_without_raising() -> None:
    year_sessions = {year: 20 for year in _ALL_YEARS}
    year_sessions[2021] = 2

    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=year_sessions,
        bars_per_session=25,
        signal_years=set(),
        effect=0.0,
        sigma=0.0,
    )
    lens = Lens(panel)

    report = lens.stability(lens.feature("return_1"), horizon=1)

    assert 2021 in report.by_year


# Verdict criteria


@pytest.mark.parametrize(
    "panel_kwargs, expected_token",
    [
        (dict(effect=0.01, sigma=0.0005), "PASS"),
        (dict(effect=0.0005, sigma=0.0005), "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_1_edge_vs_costs(
    panel_kwargs: dict[str, float], expected_token: str
) -> None:
    # AMENDMENT 2: full_year_span + _call_verdict's new defaults make criteria
    # 6 and 7 genuinely evaluate, so the PASS case is a real seven-of-seven
    # SURVIVED rather than a five-of-seven verdict with two criteria silently
    # dropped (specs/lens_verdict_integrity.md AMENDMENT 2).
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=_FULL_YEAR_SESSIONS_PER_YEAR,
        bars_per_session=240,
        signal_years=_ALL_SIGNAL_YEARS,
        full_year_span=_FULL_YEAR_COMPLETE_YEARS,
        **panel_kwargs,
    )
    verdict = _call_verdict(panel)

    assert expected_token in verdict.reasons[0]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


@pytest.mark.parametrize(
    "years, sessions_per_year, signal_years, full_year_span, expected_token",
    [
        (_ALL_YEARS, _FULL_YEAR_SESSIONS_PER_YEAR, _ALL_SIGNAL_YEARS,
         _FULL_YEAR_COMPLETE_YEARS, "PASS"),
        ((2024,), 2, {2024}, frozenset(), "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_2_sign_stability(
    years: tuple[int, ...],
    sessions_per_year: int | dict[int, int],
    signal_years: set[int],
    full_year_span: frozenset[int],
    expected_token: str,
) -> None:
    panel = _build_panel(
        years,
        sessions_per_year=sessions_per_year,
        bars_per_session=240,
        signal_years=signal_years,
        effect=0.01,
        sigma=0.0005,
        full_year_span=full_year_span,
    )
    verdict = _call_verdict(panel)

    assert expected_token in verdict.reasons[1]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


@pytest.mark.parametrize(
    "effect, sigma, expected_token",
    [
        (0.01, 0.0005, "PASS"),
        (0.0, 0.0005, "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_3_overlap_correction(
    effect: float, sigma: float, expected_token: str
) -> None:
    # horizon=10 (> 1) so consecutive forward returns genuinely overlap and the
    # block-bootstrap SE differs from a naive SE -- at horizon=1 there is no overlap
    # to correct for, so this criterion could not distinguish "corrected" from
    # "uncorrected". The FAIL case has zero injected effect (not merely a smaller one),
    # since a real edge tends to stay significant even after correction: mean forward
    # return grows linearly with horizon while its noise only grows as sqrt(horizon),
    # so a genuine edge's t-stat rises with horizon rather than falling. The
    # exact spread_t/spread_bps this comment historically quoted are no longer
    # reproduced here verbatim: AMENDMENT 2's full_year_span changes the date
    # grid (and therefore the row count) so criteria 6/7 evaluate, which
    # shifts those exact figures without changing the FAIL outcome they
    # supported.
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=_FULL_YEAR_SESSIONS_PER_YEAR,
        bars_per_session=240,
        signal_years=_ALL_SIGNAL_YEARS,
        effect=effect,
        sigma=sigma,
        full_year_span=_FULL_YEAR_COMPLETE_YEARS,
    )
    verdict = _call_verdict(panel, horizon=10)

    assert expected_token in verdict.reasons[2]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


@pytest.mark.parametrize(
    "effect_window, expected_token",
    [
        (None, "PASS"),
        ((0, 45), "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_4_concentration(
    effect_window: tuple[int, int] | None, expected_token: str
) -> None:
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=_FULL_YEAR_SESSIONS_PER_YEAR,
        bars_per_session=240,
        signal_years=_ALL_SIGNAL_YEARS,
        effect=0.01,
        sigma=0.0005,
        effect_window_bars=effect_window,
        full_year_span=_FULL_YEAR_COMPLETE_YEARS,
    )
    verdict = _call_verdict(panel)

    assert expected_token in verdict.reasons[3]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


@pytest.mark.parametrize(
    "latency_profile, expected_token",
    [
        (_PASS_LATENCY, "PASS"),
        (_FAIL_LATENCY, "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_5_latency_profile(
    latency_profile: dict[int, float], expected_token: str
) -> None:
    panel = _all_pass_panel()
    verdict = _call_verdict(panel, latency_profile=latency_profile)

    assert expected_token in verdict.reasons[4]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


@pytest.mark.parametrize(
    "seed, mean, expected_token",
    [
        (1, 0.01, "PASS"),
        (2, 0.0, "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_verdict_criterion_6_deflated_sharpe(
    seed: int, mean: float, expected_token: str
) -> None:
    """Criterion 6 now deflates the supplied `strategy_returns`, never
    `fwd.values` (specs/lens_criteria_6_7_repair.md Defect 1 required
    behaviour: "If strategy_returns is supplied AND effective_n_trials >= 2,
    deflate CORRECTLY"). With no strategy_returns the criterion is
    NOT_EVALUATED (see test_verdict_criterion5_not_evaluated_without_latency_
    profile's criterion-5 analogue, and tests/test_lens_criteria_repair_a.py),
    so a PASS/FAIL comparison must be driven by an explicit strategy_returns
    array with effective_n_trials >= 2, not merely by varying the trial count
    against fwd.values as the old (buggy) test did.

    effect/sigma are unchanged from the original test's panel, so criteria
    1-5 still all PASS on their own. AMENDMENT 2 (specs/lens_verdict_
    integrity.md): criterion 7 now ALSO evaluates via `full_year_span` --
    under the corrected tri-state contract a NOT_EVALUATED criterion no
    longer vanishes from the conjunction, so leaving criterion 7 unevaluated
    here (as the January-only grid used to do) would make `verdict.survived`
    INCONCLUSIVE-false rather than a real PASS/FAIL comparison on criterion 6.

    Measured directly against `deflated_sharpe`/`expected_max_sharpe`
    (nifty_quant.backtest.metrics, both pre-existing/untouched):
    `expected_max_sharpe(2, var_trial_sharpes=1.0) ~= 0.5198`.
    - seed=1, normal(mean=0.01, sigma=0.001, n=500): raw per-period SR ~= 10,
      so deflated_sharpe(arr, sr0=0.5198) saturates to 1.0 -> PASS under any
      reasonable significance level.
    - seed=2, normal(mean=0.0, sigma=0.001, n=500): raw per-period SR ~= -0.05,
      so deflated_sharpe(arr, sr0=0.5198) is numerically zero -> FAIL under
      any reasonable significance level.
    """
    panel = _build_panel(
        _ALL_YEARS,
        sessions_per_year=_FULL_YEAR_SESSIONS_PER_YEAR,
        bars_per_session=240,
        signal_years=_ALL_SIGNAL_YEARS,
        effect=0.005,
        sigma=0.010,
        full_year_span=_FULL_YEAR_COMPLETE_YEARS,
    )
    strategy_returns = np.random.default_rng(seed).normal(mean, 0.001, size=500)
    verdict = _call_verdict(
        panel, effective_n_trials=2, strategy_returns=strategy_returns
    )

    assert expected_token in verdict.reasons[5]
    assert verdict.survived is (expected_token == "PASS")
    assert verdict.outcome == ("SURVIVED" if expected_token == "PASS" else "KILLED")


def test_verdict_reports_all_criteria_even_multiple_fail() -> None:
    """Criterion 6 must be genuinely evaluated (not merely defaulted) for
    "PASS" in reasons[5] to hold, since specs/lens_criteria_6_7_repair.md
    requires strategy_returns to be supplied AND effective_n_trials>=2 before
    criterion 6 can PASS or FAIL at all (Defect 1). A saturating-positive
    strategy_returns array (identical construction to
    test_verdict_criterion_6_deflated_sharpe's PASS case, seed=1, mean=0.01)
    keeps every other criterion's PASS/FAIL exactly as before."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    strategy_returns = np.random.default_rng(1).normal(0.01, 0.001, size=500)
    verdict = _call_verdict(
        panel,
        latency_profile=_FAIL_LATENCY,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    # Moved 6 -> 7 when criterion 7 (recent-years cost gate) was added to the spec.
    assert len(verdict.reasons) == 7
    assert "FAIL" in verdict.reasons[1]
    assert "FAIL" in verdict.reasons[3]
    assert "FAIL" in verdict.reasons[4]
    assert "PASS" in verdict.reasons[0]
    assert "PASS" in verdict.reasons[2]
    assert "PASS" in verdict.reasons[5]


def test_verdict_criterion5_not_evaluated_without_latency_profile() -> None:
    # AMENDMENT 2 (specs/lens_verdict_integrity.md): this is the L1 regression
    # test. This call deliberately omits BOTH latency_profile (criterion 5)
    # and strategy_returns (criterion 6, since it calls lens.verdict directly
    # rather than through _call_verdict) -- criteria 1-4 and 7 still PASS on
    # `_all_pass_panel`. Under the corrected tri-state contract, one or more
    # NOT_EVALUATED criteria with no FAIL is INCONCLUSIVE, and
    # `survived is False`, never True: an incomplete evaluation must never
    # read as a pass.
    panel = _all_pass_panel()
    lens = Lens(panel)

    verdict = lens.verdict(
        "H001",
        "return_1",
        1,
        latency_profile=None,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )

    assert "NOT_EVALUATED" in verdict.reasons[4]
    assert "PASS" not in verdict.reasons[4]
    assert "NOT_EVALUATED" in verdict.reasons[5]
    assert verdict.outcome == "INCONCLUSIVE"
    assert verdict.survived is False


def test_verdict_all_evaluated_pass_and_flip_one_criterion() -> None:
    # AMENDMENT 2: no longer overrides effective_n_trials=1 (which forced
    # criterion 6 to NOT_EVALUATED) -- both calls now use _call_verdict's
    # default (effective_n_trials=2, real strategy_returns), so pass_verdict
    # is a genuine seven-of-seven SURVIVED, not five PASSes with two criteria
    # silently dropped.
    panel = _all_pass_panel()

    pass_verdict = _call_verdict(panel, latency_profile=_PASS_LATENCY)
    assert pass_verdict.survived is True
    assert pass_verdict.outcome == "SURVIVED"
    assert "PASS" in pass_verdict.reasons[4]

    flip_verdict = _call_verdict(panel, latency_profile=_FAIL_LATENCY)
    assert flip_verdict.survived is False
    assert flip_verdict.outcome == "KILLED"
    assert "FAIL" in flip_verdict.reasons[4]


# Explain / markdown


def test_to_markdown_contains_hypothesis_id() -> None:
    verdict = _make_all_pass_verdict()
    assert "H001_close_reversion" in verdict.to_markdown()


def test_to_markdown_contains_all_reason_lines() -> None:
    verdict = _make_all_pass_verdict()
    markdown = verdict.to_markdown()
    for reason in verdict.reasons:
        assert reason in markdown


def test_to_markdown_contains_cost_hurdle_and_se_method() -> None:
    verdict = _make_all_pass_verdict()
    markdown = verdict.to_markdown()

    assert "cost" in markdown.lower()
    assert "hurdle" in markdown.lower()
    assert "block_bootstrap" in markdown


def test_explain_contains_hypothesis_id() -> None:
    verdict = _make_all_pass_verdict()
    assert "H001_close_reversion" in verdict.explain()


def test_markdown_survived_and_killed_tokens() -> None:
    # AMENDMENT 2: _make_all_pass_verdict now goes through a genuinely
    # seven-of-seven-evaluated verdict (full_year_span + real strategy_returns
    # via _call_verdict's defaults), so this is a real SURVIVED, not a
    # five-of-seven verdict that happened to render "SURVIVED".
    passed_verdict = _make_all_pass_verdict()
    assert passed_verdict.outcome == "SURVIVED"
    passed_markdown = passed_verdict.to_markdown()
    assert "SURVIVED" in passed_markdown

    killed_verdict = _make_killed_verdict()
    assert killed_verdict.outcome == "KILLED"
    killed_markdown = killed_verdict.to_markdown()
    assert "KILLED" in killed_markdown
    assert "SURVIVED" not in killed_markdown


# Determinism


def test_verdict_determinism_same_inputs() -> None:
    panel = _all_pass_panel(seed=123)
    lens = Lens(panel, seed=0)

    kwargs = {
        "latency_profile": _PASS_LATENCY,
        "effective_n_trials": 1,
        "method": "cross_sectional_rank",
        "n_buckets": 5,
        "n_boot": 100,
    }
    first = lens.verdict("H_det", "return_1", 1, **kwargs)
    second = lens.verdict("H_det", "return_1", 1, **kwargs)

    assert first.survived == second.survived
    assert first.reasons == second.reasons
    np.testing.assert_allclose(first.cost_hurdle_bps, second.cost_hurdle_bps)


def test_seed_recorded_in_markdown() -> None:
    panel = _all_pass_panel()
    lens = Lens(panel, seed=42)

    verdict = lens.verdict(
        "H_seed",
        "return_1",
        1,
        latency_profile=_PASS_LATENCY,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )

    assert "42" in verdict.to_markdown()
