"""Independent test suite for the Phase E conditional-analysis sweep.

Written from specs/phase_e_sweep.md, without reading any sibling test file. Originally
written before the Phase E implementation landed, when this suite had to GUESS at module
names for the not-yet-built runner and promotion function; the coordinator's report
(after both independent suites' guesses were compared) pinned the real names via
AMENDMENT 2/3, and this revision repoints every guessed import at them:

    research/sweep_features.py     FEATURE_REGISTRY, HORIZONS, n_planned_trials()
                                    (also exposes FEATURES/N_PLANNED_TRIALS aliases, which
                                    this suite originally guessed at -- both resolve to the
                                    same objects, so this file uses the pinned names)
    research/feature_sweep.py      run_sweep(...), measure_var_trial_sharpes(...),
                                    is_candidate(...), evaluate_promotion(...)

`research/sweep.py` (this suite's original obligation-2/3/7/11/12/13 guess) is a
pre-existing, UNRELATED grid-parameter/constraint-parsing module -- a real naming
collision this suite flagged and that AMENDMENT 2 explicitly worked around by putting the
Phase E runner in `feature_sweep.py` instead.

Obligation 4 keeps its original ANCHOR test unchanged: `conditional_expectancy` itself is
confirmed (independently, by the coordinator) to still silently return
`spread_bps=0.0, spread_t=0.0` for a 3-symbol fixture, by design -- the guard belongs in
`run_sweep`, not in that shared primitive, so changing the primitive's contract would ripple
through every other caller. A second test now asserts the real obligation directly against
`run_sweep`.

Obligation 14 (new, AMENDMENT 3) is added below: `run_sweep` must refuse to run if its
contract's `holdout_intent` is not `'never'`, or if `contract.data['end']` falls on or after
the live holdout boundary.

Obligations 5, 6, 8, 9, 10 test `nifty_quant.backtest.metrics` and `nifty_quant.research.ic`
directly and are unaffected by any of the above; they are unchanged from the original
submission.

No test in this file guards an assertion behind hasattr/getattr(default)/
try-except-AttributeError/pytest.skip on an absent attribute or missing import. A
not-yet-implemented obligation is left to fail outright.
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
from nifty_quant.research.expectancy import conditional_expectancy, forward_returns
from nifty_quant.research.feature_sweep import (
    evaluate_promotion,
    is_candidate,
    measure_var_trial_sharpes,
    run_sweep,
)
from nifty_quant.research.ic import information_coefficient
from nifty_quant.research.sweep_features import HORIZONS, n_planned_trials
from tests.contract_fixtures import minimal_contract


def _make_close(n_symbols: int, n_rows: int = 40, seed: int = 0):
    """A real, float64 close-price panel and a session-BOUNDARY day_offsets array
    ([0, ..., n_rows], strictly increasing -- guards.check_day_offsets), single session.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.01, size=(n_rows, n_symbols)).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    return close, day_offsets


def _dummy_feature(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """A trivial (close, day_offsets) -> feature callable for tests that only care about
    the sweep's bookkeeping (trial counting, provenance, symbol-count guard), not any
    particular feature's numeric content."""
    return close.copy()


def _raising_feature(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    raise RuntimeError("synthetic failure for obligation 11")


# ---------------------------------------------------------------------------
# Obligation 1 -- explicit registry, denominator changes with the list.
# ---------------------------------------------------------------------------


def test_obligation_1_registry_is_explicit_list_and_defines_n_planned_trials():
    from nifty_quant.research.sweep_features import FEATURE_REGISTRY

    # E1: "not a glob over module contents -- an explicit list" (a tuple, here, but a
    # fixed, enumerable, non-introspected sequence either way, which is the substance of
    # the obligation).
    assert isinstance(FEATURE_REGISTRY, (list, tuple))
    assert len(FEATURE_REGISTRY) > 0
    assert len(HORIZONS) > 0

    n_planned = n_planned_trials()
    assert n_planned == len(FEATURE_REGISTRY) * len(HORIZONS)

    # "a registry that silently grows changes the denominator" -- one more declared
    # feature must change the formula n_planned_trials() is computed from.
    assert (len(FEATURE_REGISTRY) + 1) * len(HORIZONS) != n_planned


# ---------------------------------------------------------------------------
# Obligation 2 -- a declared n_planned_trials raises on the (k+1)th trial.
# ---------------------------------------------------------------------------


def test_obligation_2_sweep_raises_on_trial_k_plus_one():
    close, day_offsets = _make_close(n_symbols=5)

    # `validation` is a WHOLE-section override (contract_fixtures.minimal_contract
    # replaces the section, it does not merge into it), so holdout_intent='never' must be
    # restated here alongside n_planned_trials or _assert_holdout_intent_never fires first.
    contract = minimal_contract(
        validation={
            "scheme": "test",
            "holdout_intent": "never",
            "n_planned_trials": 1,
        }
    )

    # Two (feature, horizon) combinations attempted against a sweep that declared only
    # one planned trial must raise partway through, not silently run both.
    with pytest.raises(ValueError, match="n_planned_trials"):
        run_sweep(
            contract=contract,
            close=close,
            day_offsets=day_offsets,
            horizons=[1, 5],
            feature_registry_override=[("dummy", _dummy_feature)],
        )


# ---------------------------------------------------------------------------
# Obligation 3 -- every trial writes a TrialRecord with non-null provenance.
# ---------------------------------------------------------------------------


def test_obligation_3_every_trial_writes_trial_record_with_hash_seed_git_sha():
    close, day_offsets = _make_close(n_symbols=5)
    contract = minimal_contract()

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry_override=[("dummy", _dummy_feature)],
    )

    assert len(records) == 1
    for record in records:
        assert record.contract_hash is not None
        assert record.contract_hash == contract.contract_hash
        assert record.seed is not None
        assert record.git_sha is not None


# ---------------------------------------------------------------------------
# Obligation 4 -- THE anti-silent-failure test.
# ---------------------------------------------------------------------------


def test_obligation_4_anchor_conditional_expectancy_does_not_raise_below_five_symbols_today():
    """ANCHOR test: documents the primitive's sharp edge as it exists today, unchanged
    on purpose (per the coordinator: changing `conditional_expectancy`'s return contract
    would ripple through every existing caller, so the guard lives in `run_sweep`, not
    here). `cross_sectional_rank` enforces `min_names=5`; below it, `causal_buckets`
    leaves every row unassigned (label -1), so `conditional_expectancy` returns an EMPTY
    bucket table with `spread_bps=0.0, spread_t=0.0` -- a real-looking zero, not an
    exception. This test asserts that TODAY'S behaviour, so a future change to this
    primitive's contract is a deliberate, visible decision, not an accidental regression.
    """
    close, day_offsets = _make_close(n_symbols=3)  # deliberately below min_names=5
    feature = np.random.default_rng(4).normal(size=close.shape).astype(np.float64)
    fwd = forward_returns(close, day_offsets, horizon=1)

    table = conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=5,
        method="cross_sectional_rank",
        cost_hurdle_bps=10.0,
        feature_name="obligation_4_anchor",
    )

    assert table.buckets == ()
    assert table.spread_bps == 0.0
    assert table.spread_t == 0.0


def test_obligation_4_run_sweep_raises_for_fewer_than_five_symbols():
    """The real obligation: the SWEEP RUNNER, not the shared primitive, must refuse to
    run below `MIN_NAMES=5` symbols rather than let the anchor test's silent zero out
    into a trial record that reads as a real result.
    """
    close, day_offsets = _make_close(n_symbols=3)  # deliberately below min_names=5
    contract = minimal_contract()

    with pytest.raises(ValueError, match="5"):
        run_sweep(
            contract=contract,
            close=close,
            day_offsets=day_offsets,
            horizons=[1],
            feature_registry_override=[("dummy", _dummy_feature)],
        )


# ---------------------------------------------------------------------------
# Obligation 5 -- effective_n_trials on near-identical trials ~= 1.
# ---------------------------------------------------------------------------


def test_obligation_5_effective_n_trials_near_identical_columns_reports_about_one():
    rng = np.random.default_rng(5)
    t = 500
    n_trials = 10

    base = rng.normal(0.0, 1.0, size=t).astype(np.float64)
    tiny_noise = rng.normal(0.0, 1e-3, size=(t, n_trials)).astype(np.float64)
    trial_returns = base[:, None] + tiny_noise  # all columns near-identical

    n_eff = effective_n_trials(trial_returns)

    assert isinstance(n_eff, float)
    # Not exactly 1.0 because of the tiny independent noise, but must be far
    # below the naive column count -- pins the estimator from the low end.
    assert n_eff < 2.0
    assert n_eff < n_trials


# ---------------------------------------------------------------------------
# Obligation 6 -- effective_n_trials on independent trials ~= column count.
# ---------------------------------------------------------------------------


def test_obligation_6_effective_n_trials_independent_columns_reports_column_count():
    rng = np.random.default_rng(6)
    t = 500
    n_trials = 10

    trial_returns = rng.normal(0.0, 1.0, size=(t, n_trials)).astype(np.float64)
    n_eff = effective_n_trials(trial_returns)

    # Sampling noise means an iid Gaussian correlation matrix is never
    # exactly identity, so a 0.6 * n_trials floor is a stable, conservative
    # lower bound -- pins the estimator from the high end.
    assert n_eff > n_trials * 0.6


# ---------------------------------------------------------------------------
# Obligation 7 -- var_trial_sharpes is MEASURED, not the 1.0 placeholder.
# ---------------------------------------------------------------------------


def test_obligation_7_var_trial_sharpes_is_measured_not_the_placeholder_one():
    # Concrete trial Sharpes a real sweep would produce, and the concrete
    # derivation that SHOULD be applied to them (sample variance, ddof=1).
    trial_sharpes = np.array([0.18, 0.24, 0.31, 0.27, 0.35], dtype=np.float64)
    expected_var = float(np.var(trial_sharpes, ddof=1))

    measured = measure_var_trial_sharpes(trial_sharpes)

    assert measured == pytest.approx(expected_var)
    assert measured != 1.0  # the known unmeasured placeholder (lens.py:815) must be gone


# ---------------------------------------------------------------------------
# Obligation 8 -- deflated Sharpe uses n_eff; larger n_eff lowers the DSR.
# ---------------------------------------------------------------------------


def test_obligation_8_deflated_sharpe_uses_measured_n_eff_and_larger_n_lowers_dsr():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0015, 0.02, size=200).astype(np.float64)

    variance_trials = 0.25
    small_n = 2
    large_n = 150

    em_small = expected_max_sharpe(small_n, variance_trials)
    em_large = expected_max_sharpe(large_n, variance_trials)
    assert em_large > em_small  # more trials -> harder expected null-max hurdle

    dsr_small = deflated_sharpe(returns, sr0=em_small)
    dsr_large = deflated_sharpe(returns, sr0=em_large)

    # Same realised returns, larger measured n_eff -> a harder threshold ->
    # a strictly lower DSR.
    assert dsr_small > dsr_large


# ---------------------------------------------------------------------------
# Obligation 9 -- pbo_cscv returns a finite number on a live trial matrix.
# ---------------------------------------------------------------------------


def test_obligation_9_pbo_cscv_returns_finite_value_on_live_trial_matrix():
    rng = np.random.default_rng(9)
    t = 512
    n_trials = 10

    trial_matrix = rng.normal(0.0, 1.0, size=(t, n_trials)).astype(np.float64)
    pbo = pbo_cscv(trial_matrix, n_splits=16)

    assert np.isfinite(pbo)
    assert 0.0 <= pbo <= 1.0


# ---------------------------------------------------------------------------
# Obligation 10 -- IC SE is overlap-aware; differs from a naive iid SE at h>1.
# ---------------------------------------------------------------------------


def test_obligation_10_ic_se_overlap_aware_differs_from_naive_iid_se_at_h_gt_1():
    """`information_coefficient` (research/ic.py) never special-cases horizon==1
    back to a naive formula -- its SE always comes from the overlap-aware
    block bootstrap. We build an autocorrelated common factor so the per-bar
    IC series itself is autocorrelated at h=5 (h > 1), and assert the
    reported SE differs materially from a naive iid std/sqrt(n) computed
    directly on the same per-bar series.

    Note (per the task brief, NOT re-tested here as a pass/fail assertion):
    this is distinct from expectancy.py's bucket-level spread SE, where
    horizon == 1 legitimately falls back to the naive formula
    (expectancy.py ~line 496) because non-overlapping h=1 forward returns
    carry no such correction to make. That legitimate special case belongs
    to `conditional_expectancy`'s bucket statistics, not to
    `information_coefficient`, and asserting they behave identically would
    be testing the wrong function.
    """
    t = 250
    n_symbols = 6
    horizon = 5
    rng = np.random.default_rng(10)

    # Bars are left-labelled; day_offsets is a session-BOUNDARY array
    # ([0, ..., n_rows], strictly increasing -- guards.check_day_offsets),
    # built explicitly rather than assuming any fixed session stride
    # (CLAUDE.md rule 5) -- one session here.
    day_offsets = np.array([0, t], dtype=np.int64)

    total = t + horizon
    factor = np.empty(total, dtype=np.float64)
    factor[0] = 0.0
    for idx in range(1, total):
        factor[idx] = 0.95 * factor[idx - 1] + rng.normal(0.0, 0.10)

    loadings = np.linspace(-1.0, 1.0, n_symbols, dtype=np.float64)

    feature = (
        factor[:t, None] * loadings[None, :]
        + rng.normal(0.0, 0.20, size=(t, n_symbols)).astype(np.float64)
    )
    fwd = (
        factor[horizon : t + horizon, None] * loadings[None, :]
        + rng.normal(0.0, 0.20, size=(t, n_symbols)).astype(np.float64)
    )

    result = information_coefficient(
        feature=feature,
        fwd=fwd,
        day_offsets=day_offsets,
        horizon=horizon,
        method="pearson",
        seed=11,
    )

    finite = result.per_bar[np.isfinite(result.per_bar)]
    assert finite.size > 10

    naive_se = float(finite.std(ddof=1) / np.sqrt(finite.size))

    assert np.isfinite(result.se)
    assert result.se > 0.0
    assert not np.isclose(result.se, naive_se, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Obligation 11 -- a raising feature is a recorded failed trial, not dropped.
# ---------------------------------------------------------------------------


def test_obligation_11_raising_feature_recorded_as_failed_trial_with_exception():
    close, day_offsets = _make_close(n_symbols=5)
    contract = minimal_contract()

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry_override=[("raises", _raising_feature)],
    )

    assert len(records) == 1
    assert records[0].error is not None
    assert "RuntimeError" in records[0].error


# ---------------------------------------------------------------------------
# Obligation 12 -- promotion requires ALL FOUR E4 conditions independently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_metric, bad_value",
    [
        ("spread_bps", 0.5),        # fails: |spread_bps| > 2 * cost_hurdle_bps (2*2.0=4.0)
        ("spread_t", 1.0),          # fails: |spread_t| > 1.96
        ("monotonic", False),       # fails: E[R | bucket] monotone across buckets
        ("deflated_sharpe", 0.1),   # fails: deflated Sharpe clears its threshold
    ],
)
def test_obligation_12_each_e4_condition_independently_blocks_promotion(
    bad_metric, bad_value
):
    metrics = {
        "spread_bps": 12.0,
        "cost_hurdle_bps": 2.0,
        "spread_t": 2.3,
        "monotonic": True,
        "deflated_sharpe": 1.5,
        "deflated_sharpe_threshold": 0.8,
    }
    # All four PASS by construction; flip exactly one so the test isolates
    # that single condition as the blocker.
    metrics[bad_metric] = bad_value

    assert is_candidate(**metrics) is False


def test_obligation_12_all_four_conditions_passing_does_promote():
    metrics = {
        "spread_bps": 12.0,
        "cost_hurdle_bps": 2.0,
        "spread_t": 2.3,
        "monotonic": True,
        "deflated_sharpe": 1.5,
        "deflated_sharpe_threshold": 0.8,
    }
    assert is_candidate(**metrics) is True


# ---------------------------------------------------------------------------
# Obligation 13 -- promotion is evaluated on the recent window, not pooled.
# ---------------------------------------------------------------------------


def test_obligation_13_promotion_uses_recent_window_not_pooled():
    """Real fixture, real `evaluate_promotion` call: a fixed per-symbol feature rank
    (identical every row, so bucket assignment is stable) predicts forward returns
    STRONGLY across the first 12 (of 15) sessions but carries NO signal in the last 3 --
    mirroring H2's own recorded pooled-vs-recent divergence (sign-stable pooled edge,
    KILLED on the recent-years cost gate, `killed-hypotheses.md`). Pooled statistics
    (dominated by the 12 edge-carrying sessions) must clear every E4 bar; the recent
    window (the last 3, no-edge sessions) must not, and `promoted` must be False.
    """
    rng = np.random.default_rng(13)
    n_symbols = 6
    n_sessions_early = 12
    n_sessions_recent = 3
    rows_per_session = 20
    n_rows = (n_sessions_early + n_sessions_recent) * rows_per_session
    early_rows = n_sessions_early * rows_per_session

    day_offsets = np.arange(0, n_rows + rows_per_session, rows_per_session, dtype=np.int64)
    assert day_offsets[-1] == n_rows

    # Fixed per-symbol rank, identical every row -> deterministic bucket assignment
    # under cross_sectional_rank (rank is a within-row statistic, so a constant column
    # ordering produces the same top/bottom bucket membership on every row).
    symbol_rank = np.arange(n_symbols, dtype=np.float64)
    feature = np.broadcast_to(symbol_rank[None, :], (n_rows, n_symbols)).astype(np.float64).copy()

    centered_rank = symbol_rank - symbol_rank.mean()
    edge_slope = 0.01
    noise_sd = 0.001
    log_returns = np.empty((n_rows, n_symbols), dtype=np.float64)
    for t in range(n_rows):
        if t < early_rows:
            log_returns[t, :] = edge_slope * centered_rank + rng.normal(0.0, noise_sd, n_symbols)
        else:
            log_returns[t, :] = rng.normal(0.0, noise_sd, n_symbols)  # no edge, recent window

    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    result = evaluate_promotion(
        feature=feature,
        close=close,
        day_offsets=day_offsets,
        horizon=1,
        recent_n_sessions=n_sessions_recent,
        n_buckets=5,
    )

    # Pooled clears every E4 bar (the fixture is dominated by 12 strong-edge sessions).
    assert abs(result.pooled["spread_bps"]) > 2.0 * result.pooled["cost_hurdle_bps"]
    assert abs(result.pooled["spread_t"]) > 1.96
    assert result.pooled["monotonic"] is True

    # The recent window (last 3, no-edge sessions) is what decides promotion, and it must
    # NOT promote even though pooled would have.
    assert result.promoted is False


# ---------------------------------------------------------------------------
# Obligation 14 (AMENDMENT 3) -- run_sweep is structurally unable to reach the holdout.
# ---------------------------------------------------------------------------


def test_obligation_14_holdout_intent_must_be_never():
    close, day_offsets = _make_close(n_symbols=5)
    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "after_conditions_close"}
    )

    with pytest.raises(ValueError, match="holdout_intent"):
        run_sweep(
            contract=contract,
            close=close,
            day_offsets=day_offsets,
            horizons=[1],
            feature_registry_override=[("dummy", _dummy_feature)],
        )


@pytest.mark.holdout_aware
@pytest.mark.parametrize(
    "end_date, expect_raise",
    [
        ("2020-12-31", False),  # safely before the holdout
        ("2025-08-14", True),   # exact boundary -- must still refuse
        ("2026-08-01", True),   # inside the holdout window
    ],
)
def test_obligation_14_data_window_must_end_before_holdout_start(end_date, expect_raise):
    """Marked `holdout_aware` (tests/conftest.py) so this test sees the REAL production
    holdout boundary rather than the suite-wide far-future stand-in every other test in
    this repo gets -- this test is specifically ABOUT the holdout guard.
    """
    close, day_offsets = _make_close(n_symbols=5)
    contract = minimal_contract(
        data={
            "panel_id": "test-panel",
            "panel_hash": "test",
            "universe_name": "test",
            "start": "2020-01-01",
            "end": end_date,
        }
    )

    call = lambda: run_sweep(  # noqa: E731 - local, single-use, clearer than a def here
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry_override=[("dummy", _dummy_feature)],
    )

    if expect_raise:
        with pytest.raises(ValueError, match="holdout"):
            call()
    else:
        records = call()
        assert len(records) == 1
