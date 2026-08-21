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
