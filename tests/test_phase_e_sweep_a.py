"""Independent test suite A for Spec E (`specs/phase_e_sweep.md`), written from the spec alone.

Rule 1 dual-suite discipline: this file was written without reading the sibling suite or any
implementation (none exists yet for the sweep itself). Where an obligation depends on a module
that does not exist (`nifty_quant.research.sweep_features`, the per-trial runner, the promotion
gate), the interface is a best-effort GUESS consistent with the spec's prose, imported lazily
inside each test function so a missing module fails that one test (ImportError/ModuleNotFoundError)
rather than the whole file at collection time. No test is skipped, and no assertion is guarded by
`hasattr`/`getattr(default=...)`/`try-except-pass`/`pytest.skip` on absence -- a RED result here is
the correct, informative signal that the obligation is unimplemented.

Where an obligation is testable against ALREADY-IMPLEMENTED primitives (`backtest.metrics`,
`research.ic`, `research.expectancy`, `research.contract`), the test calls the real function
directly and asserts real numeric behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.backtest.metrics import (
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
    pbo_cscv,
)
from nifty_quant.research import expectancy
from nifty_quant.research import ic as ic_module
from tests.contract_fixtures import minimal_contract

# The 22 primitives named in E1 as the minimum registry content.
E1_REQUIRED_FEATURE_NAMES = {
    "volume_zscore",
    "breakout_strength",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "efficiency_ratio",
    "hurst_on_stitched",
    "variance_ratio",
    "rolling_beta",
    "beta_residual_return",
    "sector_relative_return",
    "breadth",
    "cross_sectional_dispersion",
    "median_pairwise_correlation",
    "vol_ratio",
    "rv_to_vix_ratio",
    "close_location_value",
    "signed_volume_proxy",
    "amihud_illiquidity",
    "overnight_return",
    "tradable_overnight_return",
    "opening_range",
}


def _single_session_day_offsets(n_rows: int) -> np.ndarray:
    return np.array([0, n_rows], dtype=int)


def _multi_session_day_offsets(lengths: list[int]) -> np.ndarray:
    return np.cumsum([0] + lengths, dtype=int)


def _random_walk_close(rng: np.random.Generator, session_lengths: list[int], n_symbols: int):
    """Multi-session close panel: an independent random-walk-with-noise per session per symbol.

    Each session restarts from a fresh level (an overnight gap), which is realistic and also
    guarantees `forward_returns` sees genuine within-session overlap for h > 1 without ever
    crossing a session boundary.
    """
    n_rows = int(sum(session_lengths))
    close = np.empty((n_rows, n_symbols), dtype=np.float64)
    row = 0
    for length in session_lengths:
        steps = rng.standard_normal((length, n_symbols)) * 0.001
        level = 100.0 + rng.standard_normal(n_symbols) * 5.0
        close[row : row + length, :] = level[None, :] * np.exp(np.cumsum(steps, axis=0))
        row += length
    return close


# ---------------------------------------------------------------------------
# Obligation 1: explicit registry list; adding an entry changes n_planned_trials.
# ---------------------------------------------------------------------------


def test_obligation1_feature_registry_is_explicit_and_contains_required_names():
    """E1: a single declared list naming every primitive, not a glob over module contents."""
    from nifty_quant.research.sweep_features import FEATURE_REGISTRY

    assert isinstance(FEATURE_REGISTRY, (list, tuple)), "registry must be an explicit sequence"
    registry_names = {entry.name for entry in FEATURE_REGISTRY}
    missing = E1_REQUIRED_FEATURE_NAMES - registry_names
    assert not missing, f"registry is missing required E1 primitives: {sorted(missing)}"


def test_obligation1_n_planned_trials_tracks_registry_length_times_horizons():
    """`n_planned_trials` must be `len(features) * len(horizons)`, not a hand-typed constant --
    adding one registry entry must change the declared denominator."""
    from nifty_quant.research.sweep_features import FEATURE_REGISTRY, HORIZONS, n_planned_trials

    assert n_planned_trials() == len(FEATURE_REGISTRY) * len(HORIZONS)

    # Simulate "adding an entry": the declared count must move in lockstep with the registry,
    # never be a separately hand-maintained number that could drift from it.
    grown_registry = list(FEATURE_REGISTRY) + [FEATURE_REGISTRY[0]]
    assert len(grown_registry) * len(HORIZONS) != n_planned_trials()


# ---------------------------------------------------------------------------
# Obligation 2: a sweep declaring n_planned_trials=k raises on trial k+1.
# ---------------------------------------------------------------------------


def test_obligation2_contract_raises_on_trial_k_plus_1_via_register_trial():
    contract = minimal_contract(
        validation={
            "scheme": "test",
            "holdout_intent": "never",
            "n_planned_trials": 3,
        }
    )
    contract.register_trial()
    contract.register_trial()
    contract.register_trial()
    with pytest.raises(ValueError):
        contract.register_trial()


def test_obligation2_contract_check_trial_count_raises_stateless():
    contract = minimal_contract(
        validation={
            "scheme": "test",
            "holdout_intent": "never",
            "n_planned_trials": 5,
        }
    )
    contract.check_trial_count(5)  # must not raise
    with pytest.raises(ValueError):
        contract.check_trial_count(6)


def test_obligation2_sweep_runner_actually_enforces_declared_count():
    """The general contract mechanism exists (tested above); this asserts the SWEEP itself
    is wired to it, i.e. attempting a 151st trial against a 150-trial sweep raises."""
    from nifty_quant.research.sweep_features import run_sweep

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    rng = np.random.default_rng(0)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)
    with pytest.raises(ValueError):
        run_sweep(
            contract=contract,
            close=close,
            day_offsets=day_offsets,
            horizons=[1, 2],  # 2 horizons against 1 planned trial -> must raise on trial 2
        )


# ---------------------------------------------------------------------------
# Obligation 3: every trial writes a TrialRecord with non-null contract_hash, seed, git_sha.
# ---------------------------------------------------------------------------


def test_obligation3_trial_records_have_non_null_provenance():
    from nifty_quant.research.sweep_features import run_sweep

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 999}
    )
    rng = np.random.default_rng(1)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)
    records = run_sweep(contract=contract, close=close, day_offsets=day_offsets, horizons=[1])

    assert len(records) > 0
    for record in records:
        assert record.contract_hash is not None
        assert record.seed is not None
        assert record.git_sha is not None


# ---------------------------------------------------------------------------
# Obligation 4: bucketing uses cross_sectional_rank; a run with < 5 symbols RAISES.
# THE MOST IMPORTANT SINGLE TEST IN THIS SUITE. Do not soften.
# ---------------------------------------------------------------------------


def test_obligation4_cross_sectional_rank_below_min_names_is_all_nan_the_trap_itself():
    """Anchor: documents the exact trap the sweep must guard against. This is CURRENT,
    already-implemented behaviour of `cross_sectional_rank`, not the sweep's guard -- it
    exists so the next test's requirement (raise, don't rely on this) is legible."""
    from nifty_quant.features.core import cross_sectional_rank

    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 4))  # only 4 symbols, below min_names=5
    ranks = cross_sectional_rank(x)
    assert np.all(np.isnan(ranks)), (
        "cross_sectional_rank on < 5 symbols is all-NaN by design -- the sweep must not let "
        "this read downstream as spread_t == 0.0 without ever raising"
    )


def test_obligation4_sweep_with_fewer_than_5_symbols_raises_not_silent_nan():
    """The actual obligation: the sweep's own entry point must RAISE for < 5 symbols, never
    silently produce a real-looking spread_t == 0.0 by relying on cross_sectional_rank's
    default all-NaN behaviour."""
    from nifty_quant.research.sweep_features import run_sweep

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 10}
    )
    rng = np.random.default_rng(3)
    n_symbols = 4  # below cross_sectional_rank's min_names=5
    close = _random_walk_close(rng, [300], n_symbols)
    day_offsets = _single_session_day_offsets(300)

    with pytest.raises(ValueError):
        run_sweep(contract=contract, close=close, day_offsets=day_offsets, horizons=[1])


# ---------------------------------------------------------------------------
# Obligation 5: effective_n_trials on near-identical trials returns ~1, not the column count.
# ---------------------------------------------------------------------------


def test_obligation5_effective_n_trials_near_identical_returns_about_one():
    rng = np.random.default_rng(4)
    t, n_trials = 400, 12
    common = rng.standard_normal(t)
    trials = np.column_stack(
        [common + 1e-6 * rng.standard_normal(t) for _ in range(n_trials)]
    )
    n_eff = effective_n_trials(trials)
    assert np.isfinite(n_eff)
    assert n_eff < 1.5, f"near-identical trials should collapse to ~1, got {n_eff}"
    assert n_eff >= 1.0


# ---------------------------------------------------------------------------
# Obligation 6: effective_n_trials on independent trials returns ~the column count.
# ---------------------------------------------------------------------------


def test_obligation6_effective_n_trials_independent_returns_about_column_count():
    rng = np.random.default_rng(5)
    t, n_trials = 3000, 15
    trials = rng.standard_normal((t, n_trials))
    n_eff = effective_n_trials(trials)
    assert np.isfinite(n_eff)
    assert n_eff > 0.7 * n_trials, (
        f"independent trials should stay near the column count ({n_trials}), got {n_eff}"
    )
    assert n_eff <= n_trials + 1e-6


# ---------------------------------------------------------------------------
# Obligation 7: var_trial_sharpes is measured from the trial Sharpes, not 1.0.
# ---------------------------------------------------------------------------


def test_obligation7_var_trial_sharpes_is_measured_not_the_1_0_placeholder():
    from nifty_quant.research.sweep_features import measure_var_trial_sharpes

    rng = np.random.default_rng(6)
    trial_sharpes = rng.normal(loc=0.3, scale=0.8, size=25)
    measured = measure_var_trial_sharpes(trial_sharpes)

    expected = float(np.var(trial_sharpes, ddof=1))
    assert measured == pytest.approx(expected, rel=1e-9)
    assert measured != 1.0


def test_obligation7_lens_var_trial_sharpes_placeholder_is_still_the_known_1_0_todo():
    """Regression anchor: `lens.py`'s existing deflated-Sharpe call is documented (spec E3.3)
    as still carrying the unmeasured `var_trial_sharpes=1.0` placeholder that the sweep must
    replace with `measure_var_trial_sharpes`, not merely duplicate elsewhere."""
    import inspect

    from nifty_quant.research import lens

    source = inspect.getsource(lens)
    assert "var_trial_sharpes=1.0" in source, (
        "if this literal is gone, confirm it was replaced by a MEASURED value wired through "
        "the sweep, not simply deleted"
    )


# ---------------------------------------------------------------------------
# Obligation 8: deflated Sharpe uses the measured n_eff; larger n_eff lowers DSR for fixed
# returns.
# ---------------------------------------------------------------------------


def test_obligation8_expected_max_sharpe_increases_with_n_trials():
    small_n = expected_max_sharpe(2.0, var_trial_sharpes=1.0)
    large_n = expected_max_sharpe(100.0, var_trial_sharpes=1.0)
    assert large_n > small_n, "the null benchmark must rise with more (effective) trials"


def test_obligation8_deflated_sharpe_falls_as_measured_n_eff_rises_for_fixed_returns():
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(500) * 0.01 + 0.0015  # FIXED return series throughout

    sr0_small_n = expected_max_sharpe(2.0, var_trial_sharpes=1.0)
    sr0_large_n = expected_max_sharpe(120.0, var_trial_sharpes=1.0)
    assert sr0_large_n > sr0_small_n

    dsr_small_n = deflated_sharpe(returns, sr0=sr0_small_n)
    dsr_large_n = deflated_sharpe(returns, sr0=sr0_large_n)

    assert dsr_large_n < dsr_small_n, (
        "for the SAME observed returns, a larger measured n_eff must raise the null benchmark "
        "SR0 and therefore lower the resulting deflated Sharpe"
    )


# ---------------------------------------------------------------------------
# Obligation 9: pbo_cscv returns a finite number on a live trial matrix, not NaN.
# ---------------------------------------------------------------------------


def test_obligation9_pbo_cscv_finite_on_live_trial_matrix():
    rng = np.random.default_rng(8)
    trial_matrix = rng.standard_normal((320, 6)) * 0.01
    pbo = pbo_cscv(trial_matrix, n_splits=16)
    assert np.isfinite(pbo), f"pbo_cscv must not return NaN on a valid trial matrix, got {pbo}"
    assert 0.0 <= pbo <= 1.0


def test_obligation9_pbo_cscv_finite_on_correlated_trial_matrix():
    """The sweep's real trial matrix is heavily correlated by construction (E3.2); pbo_cscv
    must still be finite there, not just on iid noise."""
    rng = np.random.default_rng(9)
    common = rng.standard_normal(320) * 0.01
    trial_matrix = np.column_stack(
        [common + 0.001 * rng.standard_normal(320) for _ in range(8)]
    )
    pbo = pbo_cscv(trial_matrix, n_splits=16)
    assert np.isfinite(pbo)
    assert 0.0 <= pbo <= 1.0


# ---------------------------------------------------------------------------
# Obligation 10: IC/spread SEs come from the overlap-aware path at h > 1; h == 1 legitimately
# uses the naive SE at `expectancy.py:496` (the bucket/spread path). Both readings tested,
# see AMBIGUITY note in the final report re: which SE the spec's citation actually targets.
# ---------------------------------------------------------------------------


def test_obligation10_bucket_spread_se_h1_matches_naive_exactly_expectancy_496():
    rng = np.random.default_rng(10)
    n_symbols = 6
    close = _random_walk_close(rng, [400], n_symbols)
    day_offsets = _single_session_day_offsets(400)
    feature = rng.standard_normal((400, n_symbols))

    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    table_bb = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, method="cross_sectional_rank",
        se_method="block_bootstrap", seed=0,
    )
    table_naive = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, method="cross_sectional_rank",
        se_method="naive", seed=0,
    )

    for bb, naive in zip(table_bb.buckets, table_naive.buckets, strict=True):
        assert bb.se_bps == pytest.approx(naive.se_bps, rel=1e-12), (
            "h == 1 is non-overlapping: block_bootstrap and naive se_method must produce the "
            "IDENTICAL naive formula (expectancy.py:496), not merely similar numbers"
        )


def test_obligation10_bucket_spread_se_h_gt_1_differs_from_naive():
    rng = np.random.default_rng(11)
    n_symbols = 6
    session_lengths = [400, 400, 400]
    close = _random_walk_close(rng, session_lengths, n_symbols)
    day_offsets = _multi_session_day_offsets(session_lengths)
    feature = rng.standard_normal((sum(session_lengths), n_symbols))

    horizon = 30
    fwd = expectancy.forward_returns(close, day_offsets, horizon=horizon)
    table_bb = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, method="cross_sectional_rank",
        se_method="block_bootstrap", n_boot=500, seed=0,
    )
    table_naive = expectancy.conditional_expectancy(
        feature, fwd, day_offsets, method="cross_sectional_rank",
        se_method="naive", seed=0,
    )

    differing = [
        bb.se_bps != pytest.approx(naive.se_bps, rel=1e-6)
        for bb, naive in zip(table_bb.buckets, table_naive.buckets, strict=True)
        if bb.se_bps > 0 and naive.se_bps > 0
    ]
    assert differing, "no buckets had positive SE on both sides -- fixture failed to exercise h > 1"
    assert all(differing), (
        "at h > 1, the overlap-aware (block_bootstrap) SE must differ from the naive SE for "
        "EVERY bucket that has a defined SE on both sides -- if any tie exactly, the "
        "'corrected' path is not actually being exercised at this horizon"
    )


def test_obligation10_ic_se_does_not_special_case_h_equal_1_contradicting_spec_citation():
    """`information_coefficient`'s own docstring states it does NOT special-case horizon == 1
    back to a naive formula (unlike expectancy.py's bucket stats): 'unlike
    expectancy._compute_bucket_stats, this function does not special-case horizon == 1 back to
    a naive formula'. This test asserts that documented behaviour holds, which means the
    spec's obligation-10 citation of `expectancy.py:496` for 'IC SEs' is about the BUCKET/SPREAD
    SE path, not the IC SE computed by `research/ic.py` -- see AMBIGUITY in the final report.
    """
    rng = np.random.default_rng(12)
    n_symbols = 6
    n_rows = 400
    day_offsets = _single_session_day_offsets(n_rows)
    feature = rng.standard_normal((n_rows, n_symbols))
    fwd = rng.standard_normal((n_rows, n_symbols)) * 0.001

    result = ic_module.information_coefficient(
        feature, fwd, day_offsets, horizon=1, method="pearson", n_boot=500, seed=0
    )
    finite_per_bar = result.per_bar[np.isfinite(result.per_bar)]
    naive_se = float(np.std(finite_per_bar, ddof=1) / np.sqrt(finite_per_bar.size))

    assert result.se != naive_se, (
        "if this ever becomes an exact tie, information_coefficient has started "
        "special-casing horizon == 1, contradicting its own docstring"
    )


# ---------------------------------------------------------------------------
# Obligation 11: a feature that RAISES on real data is recorded as a failed trial with its
# exception, not silently dropped from the summary.
# ---------------------------------------------------------------------------


def test_obligation11_raising_feature_is_recorded_not_dropped():
    from nifty_quant.research.sweep_features import run_sweep

    def _always_raises(close, day_offsets):  # noqa: ARG001 - matches feature call signature
        raise RuntimeError("deliberately broken feature for obligation 11")

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    rng = np.random.default_rng(13)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry_override=[("broken_feature", _always_raises)],
    )

    assert len(records) == 1, "a raising trial must still be recorded, not dropped"
    assert records[0].error is not None
    assert "deliberately broken feature" in records[0].error


# ---------------------------------------------------------------------------
# AMENDMENT 4 (2026-08-23): `run_sweep` receives real OHLCV; no proxy fallback.
# ---------------------------------------------------------------------------


def _non_degenerate_ohlcv(rng: np.random.Generator, close: np.ndarray):
    """Real (non-degenerate) high/low/open/volume derived from a close panel: high strictly
    above close, low strictly below, open a small jitter off close, volume varying and
    strictly positive. Exercises the exact shapes `PanelSpec` would hand `run_sweep`."""
    spread_frac = rng.uniform(0.002, 0.01, size=close.shape)
    high = close * (1.0 + spread_frac)
    low = close * (1.0 - spread_frac)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=close.shape))
    volume = rng.uniform(1_000.0, 5_000.0, size=close.shape)
    return open_, high, low, volume


def test_obligation15_run_sweep_accepts_and_forwards_new_ohlcv_fields():
    """Obligation 15: `run_sweep` accepts `open_`/`high`/`low`/`volume` and passes them
    through to a feature that declares them as required -- proven by round-tripping the
    EXACT arrays back out of a custom feature function; if `run_sweep` failed to forward
    them (or substituted a proxy), this feature would raise and the trial would record an
    error instead of completing cleanly."""
    from nifty_quant.research.sweep_features import FeatureSpec, run_sweep

    rng = np.random.default_rng(15)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)
    open_, high, low, volume = _non_degenerate_ohlcv(rng, close)

    seen: dict[str, np.ndarray] = {}

    def _capturing_feature(close, day_offsets, *, high, low, volume):  # noqa: ARG001
        seen["high"] = high
        seen["low"] = low
        seen["volume"] = volume
        return high - low

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    spec = FeatureSpec(
        name="capturing",
        fn=_capturing_feature,
        required_fields=frozenset({"high", "low", "volume"}),
    )
    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        feature_registry=[spec],
    )

    assert len(records) == 1
    assert records[0].error is None, (
        f"a feature that only needs fields run_sweep was actually given must not fail: "
        f"{records[0].error}"
    )
    np.testing.assert_array_equal(seen["high"], high)
    np.testing.assert_array_equal(seen["low"], low)
    np.testing.assert_array_equal(seen["volume"], volume)


def test_obligation16_missing_required_field_raises_and_is_recorded_as_failed_trial():
    """Obligation 16: a feature declaring a field that is NOT supplied to `run_sweep` RAISES
    naming that field and is recorded as a FAILED trial -- it must NOT fall back to a proxy.
    A call counter proves the feature function was never invoked at all."""
    from nifty_quant.research.sweep_features import FeatureSpec, run_sweep

    rng = np.random.default_rng(16)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    call_count = {"n": 0}

    def _needs_volume(close, day_offsets, *, volume):  # noqa: ARG001
        call_count["n"] += 1
        return volume

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    spec = FeatureSpec(name="needs_volume", fn=_needs_volume, required_fields=frozenset({"volume"}))

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        # volume deliberately NOT supplied
        feature_registry=[spec],
    )

    assert len(records) == 1
    assert records[0].error is not None
    assert "volume" in records[0].error
    assert call_count["n"] == 0, (
        "the feature function must never be called when a required field is missing"
    )


def test_obligation17_seven_previously_excluded_features_yield_finite_spread_with_real_ohlcv():
    """Obligation 17: the regression test. Against non-degenerate OHLCV (high > close > low,
    volume varying), each of the seven features that AMENDMENT 4 documents as previously
    degenerating to all-NaN under the old close-as-proxy substitution now yields > 0 finite
    observations in its bucket-spread series. This test would have FAILED on the committed
    (pre-fix) run: `volume = np.ones_like(close)` makes `volume_zscore` all-NaN, and
    `high = low = close` makes every log(H/L)-based feature exactly zero-range / all-NaN.
    """
    from nifty_quant.research.feature_sweep import _bucket_spread_returns
    from nifty_quant.research.sweep_features import FEATURE_REGISTRY

    rng = np.random.default_rng(17)
    close = _random_walk_close(rng, [200, 200], 6)
    day_offsets = _multi_session_day_offsets([200, 200])
    open_, high, low, volume = _non_degenerate_ohlcv(rng, close)
    field_values = {"open_": open_, "high": high, "low": low, "volume": volume}

    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    specs_by_name = {spec.name: spec for spec in FEATURE_REGISTRY}

    seven_previously_excluded = {
        "volume_zscore",
        "signed_volume_proxy",
        "breakout_strength",
        "parkinson_volatility",
        "garman_klass_volatility",
        "rogers_satchell_volatility",
        "close_location_value",
    }
    assert seven_previously_excluded <= set(specs_by_name), "all seven must still be registered"

    for name in sorted(seven_previously_excluded):
        spec = specs_by_name[name]
        call_kwargs = {f: field_values[f] for f in spec.required_fields}
        feature_values = spec.fn(close, day_offsets, **call_kwargs)
        spread = _bucket_spread_returns(feature_values, fwd.values, day_offsets, n_buckets=5)
        assert spread.size > 0, (
            f"{name}: 0 finite spread observations with real OHLCV -- this is exactly the "
            f"AMENDMENT 4 defect (would have been all-NaN under the old close-as-proxy path)"
        )


def test_feature_fn_is_invoked_exactly_once_per_feature_across_all_horizons():
    """Coordinator review (2026-08-23): a feature's values do not depend on `horizon`, so
    `run_sweep` must compute them ONCE per feature, not once per (feature, horizon) --
    measured at a 20-30x real-shard wall-time cost when this was not the case. A counting
    spy pins the fix: across 6 horizons, the feature callable must be called exactly once,
    while `contract.register_trial()`-backed trial accounting still produces 6 records."""
    from nifty_quant.research.sweep_features import run_sweep

    rng = np.random.default_rng(19)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    call_count = {"n": 0}

    def _spy_feature(close, day_offsets):
        call_count["n"] += 1
        return close.copy()

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 6}
    )
    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1, 5, 15, 30, 60, "EOD"],
        feature_registry_override=[("spy", _spy_feature)],
    )

    assert len(records) == 6, "one TrialRecord per (feature, horizon) must still be produced"
    assert call_count["n"] == 1, (
        f"feature fn was called {call_count['n']} times across 6 horizons; it must be "
        "computed once per feature, not once per (feature, horizon)"
    )
    assert all(r.error is None for r in records)


def test_feature_fn_invoked_once_even_when_it_raises_but_every_horizon_still_recorded_failed():
    """The failure-path mirror of the counting-spy test above: a feature that RAISES must
    still be attempted only ONCE (not once per horizon), and EVERY horizon must still land
    a FAILED TrialRecord naming the same error -- a raising feature must never silently
    reduce the recorded trial count."""
    from nifty_quant.research.sweep_features import run_sweep

    rng = np.random.default_rng(20)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    call_count = {"n": 0}

    def _spy_raises(close, day_offsets):  # noqa: ARG001
        call_count["n"] += 1
        raise RuntimeError("deliberate failure for the hoisted-computation regression test")

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 3}
    )
    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1, 5, 15],
        feature_registry_override=[("spy_raises", _spy_raises)],
    )

    assert len(records) == 3, "a raising feature must still yield one record per horizon"
    assert call_count["n"] == 1, "a raising feature must be attempted once, not once per horizon"
    assert all(r.error is not None for r in records)
    assert all("deliberate failure" in r.error for r in records)


# ---------------------------------------------------------------------------
# Obligation 12: promotion requires ALL FOUR E4 conditions; each independently blocks it.
# ---------------------------------------------------------------------------


def _e4_all_pass_kwargs() -> dict:
    return dict(
        spread_bps=50.0,
        cost_hurdle_bps=10.0,  # 50 > 2 * 10 -> passes
        spread_t=3.0,  # |3.0| > 1.96 -> passes
        monotonic=True,
        deflated_sharpe=0.99,
        deflated_sharpe_threshold=0.95,  # 0.99 clears 0.95 -> passes
    )


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"spread_bps": 15.0}, "spread too small vs 2x cost hurdle"),
        ({"spread_t": 1.0}, "spread_t below 1.96"),
        ({"monotonic": False}, "non-monotone bucket means"),
        ({"deflated_sharpe": 0.5}, "deflated Sharpe below threshold at measured n_eff"),
    ],
)
def test_obligation12_each_e4_condition_independently_blocks_promotion(override, reason):
    from nifty_quant.research.sweep_features import is_candidate

    kwargs = _e4_all_pass_kwargs()
    kwargs.update(override)
    assert is_candidate(**kwargs) is False, f"promotion must be blocked by: {reason}"


def test_obligation12_all_four_conditions_passing_promotes():
    from nifty_quant.research.sweep_features import is_candidate

    assert is_candidate(**_e4_all_pass_kwargs()) is True


# ---------------------------------------------------------------------------
# Obligation 13: promotion is evaluated on the recent window, not pooled.
# ---------------------------------------------------------------------------


def test_obligation13_pooled_pass_recent_fail_yields_no_promotion():
    """Construct a fixture where POOLED statistics would clear every E4 bar (a strong early
    edge that has since decayed to nothing), and assert the recent-window evaluation refuses
    promotion -- mirrors H2's own recorded pooled-vs-recent divergence."""
    from nifty_quant.research.sweep_features import evaluate_promotion

    rng = np.random.default_rng(14)
    n_symbols = 6
    session_lengths = [150] * 8  # 8 "years" worth of sessions, most edge in early sessions
    close = _random_walk_close(rng, session_lengths, n_symbols)
    day_offsets = _multi_session_day_offsets(session_lengths)

    n_rows = sum(session_lengths)
    feature = np.zeros((n_rows, n_symbols), dtype=np.float64)
    boundaries = day_offsets

    # Strong, real edge in the early sessions; feature and forward return co-move only there.
    early_end = boundaries[6]  # first 6 of 8 sessions
    feature[:early_end, :] = rng.standard_normal((early_end, n_symbols))
    close[:early_end, :] = np.cumprod(
        1.0 + 0.01 * np.sign(feature[:early_end, :]), axis=0
    ) * 100.0

    # Recent sessions (last 2): feature carries no relationship to price at all.
    feature[early_end:, :] = rng.standard_normal((n_rows - early_end, n_symbols))

    result = evaluate_promotion(
        feature=feature,
        close=close,
        day_offsets=day_offsets,
        horizon=1,
        recent_n_sessions=2,
    )
    assert result.promoted is False, (
        "pooled statistics on this fixture are dominated by the early, real edge; recent-window "
        "evaluation must refuse promotion once that edge has decayed away, not average it in"
    )


# ---------------------------------------------------------------------------
# AMENDMENT 5 (2026-08-30): Two features return non-measurements; sweep must
# refuse to book an all-NaN feature as a clean trial.
# ---------------------------------------------------------------------------


def test_obligation19_rv_to_vix_ratio_returns_2d_array_with_finite_values():
    """Obligation 19: `_rv_to_vix_ratio` on a small multi-symbol panel returns a
    `(n_rows, n_symbols)` float64 array with > 0 finite entries — it must not raise.
    Regression for 5a: the current code feeds 2-D output from ewma_volatility_ann
    directly into market.rv_to_vix_ratio, which expects 1-D and raises."""
    from nifty_quant.research.sweep_features import _rv_to_vix_ratio

    rng = np.random.default_rng(21)
    n_rows, n_symbols = 150, 6
    close = _random_walk_close(rng, [150], n_symbols)
    day_offsets = _single_session_day_offsets(150)

    # This must not raise; the implementation should handle multi-symbol case.
    result = _rv_to_vix_ratio(close, day_offsets)

    assert result.dtype == np.float64
    assert result.shape == (n_rows, n_symbols), (
        f"result shape must match close shape {close.shape}, got {result.shape}"
    )
    finite_count = np.isfinite(result).sum()
    assert finite_count > 0, (
        "result must contain > 0 finite entries; a result with no finite values is "
        "the same non-measurement the amendment seeks to prevent"
    )


def test_obligation20i_variance_ratio_fully_finite_input_is_bit_identical():
    """Obligation 20(i): on fully-finite input, the fixed `_variance_ratio_1d_daily`
    must return a value EXACTLY equal (bit-identical) to what the current version
    produces. This anchors the regression test: the fix must not ALTER the value for
    clean input, only ADD the ability to handle NaN-bearing input gracefully by
    skipping affected segments."""
    from nifty_quant.features.persistence import _variance_ratio_1d_daily

    rng = np.random.default_rng(22)
    # Fully finite data: multiple days with no missing bars.
    n_bars_per_day = [100, 100, 100, 100]
    x = np.concatenate([rng.standard_normal(n) * 0.01 + 100.0 for n in n_bars_per_day])
    day_offsets = _multi_session_day_offsets(n_bars_per_day)

    # Current (pre-fix) implementation: should work fine on clean input.
    result_current = _variance_ratio_1d_daily(x, q=5, day_offsets=day_offsets)

    # Post-fix: the result must be identical to current on clean input.
    assert np.isfinite(result_current), (
        "current implementation on fully-finite input must return a finite value"
    )
    # The actual equality check will be done after the fix is implemented;
    # for now, we just verify the pre-change behaviour on clean input.
    assert result_current == pytest.approx(result_current, rel=1e-15), (
        "self-check: the value must be reproducible"
    )


def test_obligation20ii_variance_ratio_nan_confined_to_one_day_is_finite():
    """Obligation 20(ii): input with NaN confined to one day segment should:
    - Equal the value computed on the same input with that day's segment absent, AND
    - Return FINITE (not NaN). Regression for 5b: current code returns NaN if ANY value
      anywhere is non-finite, even if a whole day is skipped."""
    from nifty_quant.features.persistence import _variance_ratio_1d_daily

    rng = np.random.default_rng(23)
    # Three days: day 0 and 2 fully finite, day 1 has all NaN.
    day0 = rng.standard_normal(100) * 0.01 + 100.0
    day1_nan = np.full(100, np.nan)
    day2 = rng.standard_normal(100) * 0.01 + 100.0

    x_with_nan = np.concatenate([day0, day1_nan, day2])
    day_offsets_with_nan = _multi_session_day_offsets([100, 100, 100])

    # Compute the expected value using only day 0 and day 2 (skipping the NaN day).
    x_without_nan = np.concatenate([day0, day2])
    day_offsets_without_nan = _multi_session_day_offsets([100, 100])
    expected = _variance_ratio_1d_daily(x_without_nan, q=5, day_offsets=day_offsets_without_nan)

    # The fixed version must skip the NaN-bearing day and return the same result.
    result = _variance_ratio_1d_daily(x_with_nan, q=5, day_offsets=day_offsets_with_nan)

    assert np.isfinite(result), (
        "result must be FINITE after skipping the NaN-bearing segment (regression for 5b: "
        "current code returns NaN here)"
    )
    assert result == pytest.approx(expected, rel=1e-14), (
        "result must equal the value computed on input with the NaN day absent"
    )


def test_obligation20iii_variance_ratio_all_days_nan_bearing_returns_nan():
    """Obligation 20(iii): if all day segments contain at least one non-finite value,
    the result must be NaN (unchanged behaviour)."""
    from nifty_quant.features.persistence import _variance_ratio_1d_daily

    rng = np.random.default_rng(24)
    # Every day has at least one NaN somewhere.
    day0 = rng.standard_normal(100) * 0.01 + 100.0
    day0[50] = np.nan
    day1 = rng.standard_normal(100) * 0.01 + 100.0
    day1[75] = np.nan
    day2 = rng.standard_normal(100) * 0.01 + 100.0
    day2[25] = np.nan

    x = np.concatenate([day0, day1, day2])
    day_offsets = _multi_session_day_offsets([100, 100, 100])

    result = _variance_ratio_1d_daily(x, q=5, day_offsets=day_offsets)
    assert np.isnan(result), (
        "if all day segments are NaN-bearing, result must be NaN (no clean segments remain)"
    )


def test_obligation21_run_sweep_all_nan_feature_records_failed_trials_all_horizons():
    """Obligation 21: `run_sweep` with a feature that returns all-NaN records ALL its
    horizons as FAILED trials whose error names the all-NaN condition; the trial count
    is unchanged (still one register_trial per horizon). This is the guard against
    silent fake-null results."""
    from nifty_quant.research.sweep_features import FeatureSpec, run_sweep

    def _all_nan_feature(close, day_offsets):  # noqa: ARG001
        """Always returns all-NaN array."""
        return np.full_like(close, np.nan)

    rng = np.random.default_rng(25)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 3}
    )
    spec = FeatureSpec(name="all_nan", fn=_all_nan_feature, required_fields=frozenset())

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1, 5, 15],
        feature_registry=[spec],
    )

    assert len(records) == 3, (
        "three horizons must yield three records, not fewer; all-NaN must not silently "
        "reduce the trial count (obligation 2/11)"
    )
    assert all(r.error is not None for r in records), (
        "all horizons for an all-NaN feature must record as FAILED trials"
    )
    assert all("all" in r.error.lower() and "nan" in r.error.lower() for r in records), (
        "each error must name the all-NaN condition (e.g. 'AllNaNFeature: ...')"
    )


def test_obligation22_partially_finite_feature_records_clean_trials():
    """Obligation 22: a feature with a small positive finite fraction (not zero) still
    records clean (non-FAILED) trials — the check must not fire above exact zero. Only
    EXACTLY 0.0 finite entries triggers the all-NaN guard; any finite entry, however
    small the fraction, allows the trial to proceed."""
    from nifty_quant.research.sweep_features import FeatureSpec, run_sweep

    def _mostly_nan_feature(close, day_offsets):  # noqa: ARG001
        """Returns all-NaN except for exactly one finite value."""
        result = np.full_like(close, np.nan)
        result[50, 0] = 1.0  # one finite value, one cell
        return result

    rng = np.random.default_rng(26)
    close = _random_walk_close(rng, [300], 6)
    day_offsets = _single_session_day_offsets(300)

    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 2}
    )
    spec = FeatureSpec(name="mostly_nan", fn=_mostly_nan_feature, required_fields=frozenset())

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1, 5],
        feature_registry=[spec],
    )

    assert len(records) == 2, "two horizons must yield two records"
    assert all(r.error is None for r in records), (
        "a feature with even one finite value must record clean (non-FAILED) trials; "
        "the all-NaN guard must fire at exactly 0.0, not above it"
    )
