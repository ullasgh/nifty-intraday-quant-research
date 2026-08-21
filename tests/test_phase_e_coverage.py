"""Targeted coverage for `research/feature_sweep.py` (new module, ungated) and
`research/expectancy.py` (regressed from 100%).

Every test asserts a real, documented contract (the EOD forward-return arithmetic, the
data-window/registry/2-D guards, the deflated-Sharpe NaN-on-insufficient-data behaviour,
the expanding-quantile bucketer's "no prior data yet" and "constant feature" branches, and
the chunked block-bootstrap's session-too-short / block-index-budget / all-NaN-chunk
branches) rather than merely reaching a line. No test touches `data/`, `results/
holdout_lock.json`, or any file under `src/`.
"""

from __future__ import annotations

import numpy as np
import pytest

import nifty_quant.research.feature_sweep as feature_sweep_module
from nifty_quant.research import expectancy
from nifty_quant.research.feature_sweep import (
    _eod_forward_returns,
    _forward_returns_for_horizon,
    _is_monotonic,
    evaluate_promotion,
    run_sweep,
)
from nifty_quant.research.sweep_features import FeatureSpec
from tests.contract_fixtures import minimal_contract

# ---------------------------------------------------------------------------
# Helpers (self-contained; no import from a sibling test file).
# ---------------------------------------------------------------------------


def _dummy_feature(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """A trivial (close, day_offsets) -> feature callable: bookkeeping tests only,
    never asserted on for numeric content."""
    return close.copy()


def _make_close(n_symbols: int, n_rows: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.01, size=(n_rows, n_symbols)).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    return close, day_offsets


# ---------------------------------------------------------------------------
# `_eod_forward_returns` / `_forward_returns_for_horizon` -- the "EOD" horizon
# (feature_sweep.py lines 58, 72-86).
# ---------------------------------------------------------------------------


def test_eod_forward_returns_uses_last_bar_of_each_session_and_never_crosses_it():
    """Two sessions with very different price levels: the EOD return for every row
    must reference ITS OWN session's final close, never the other session's."""
    close = np.array(
        [
            [100.0],
            [110.0],
            [121.0],  # session 1's last bar
            [50.0],
            [55.0],  # session 2's last bar
        ]
    )
    day_offsets = np.array([0, 3, 5], dtype=np.int64)

    fwd = _eod_forward_returns(close, day_offsets)

    expected_s1 = np.log(np.array([121.0 / 100.0, 121.0 / 110.0, 121.0 / 121.0]))
    expected_s2 = np.log(np.array([55.0 / 50.0, 55.0 / 55.0]))
    np.testing.assert_allclose(fwd.values[:3, 0], expected_s1)
    np.testing.assert_allclose(fwd.values[3:, 0], expected_s2)

    # The last bar of every session compares against itself: log(1) == 0.
    assert fwd.values[2, 0] == pytest.approx(0.0, abs=1e-12)
    assert fwd.values[4, 0] == pytest.approx(0.0, abs=1e-12)

    assert fwd.session_bounded is True
    assert fwd.n_defined == 5
    assert fwd.n_nan_tail == 0
    # `horizon` is recorded as the median session length (3 and 2 bars -> 2.5 -> int 2).
    assert fwd.horizon == int(np.median(np.diff(day_offsets)))
    assert fwd.horizon == 2


def test_forward_returns_for_horizon_dispatches_eod_string_to_eod_helper():
    close, day_offsets = _make_close(n_symbols=1, n_rows=10, seed=1)
    direct = _eod_forward_returns(close, day_offsets)
    dispatched = _forward_returns_for_horizon(close, day_offsets, "EOD")

    np.testing.assert_array_equal(dispatched.values, direct.values)
    assert dispatched.horizon == direct.horizon
    assert dispatched.n_defined == direct.n_defined


def test_forward_returns_for_horizon_dispatches_int_to_fixed_shift():
    close, day_offsets = _make_close(n_symbols=1, n_rows=10, seed=2)
    dispatched = _forward_returns_for_horizon(close, day_offsets, 3)
    direct = expectancy.forward_returns(close, day_offsets, horizon=3)
    np.testing.assert_array_equal(dispatched.values, direct.values)


# ---------------------------------------------------------------------------
# `run_sweep` guards.
# ---------------------------------------------------------------------------


def test_run_sweep_rejects_non_2d_close():
    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    close_1d = np.zeros(10, dtype=np.float64)
    day_offsets = np.array([0, 10], dtype=np.int64)

    with pytest.raises(ValueError, match="2-D"):
        run_sweep(
            contract=contract,
            close=close_1d,
            day_offsets=day_offsets,
            horizons=[1],
            feature_registry_override=[("dummy", _dummy_feature)],
        )


def test_run_sweep_uses_feature_registry_argument_when_no_override_given():
    """`feature_registry=` (as distinct from `feature_registry_override=`) must be
    used verbatim, replacing the module default -- not merged with it."""
    close, day_offsets = _make_close(n_symbols=5, n_rows=30, seed=3)
    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1}
    )
    custom_registry = [FeatureSpec(name="custom_only_feature", fn=_dummy_feature)]

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry=custom_registry,
    )

    assert len(records) == 1
    assert records[0].strategy == "sweep::custom_only_feature"


def test_run_sweep_no_op_when_contract_data_has_no_end_key(monkeypatch):
    """Obligation 14 part 2's own docstring: a contract whose `data` section carries
    no `end` makes the window-vs-holdout check a no-op. Pinned here by making
    `HoldoutLock` explode if constructed -- proving the guard genuinely never reaches
    the holdout machinery, rather than merely returning early for some other reason."""
    close, day_offsets = _make_close(n_symbols=5, n_rows=20, seed=4)
    contract = minimal_contract(
        data={
            "panel_id": "test-panel",
            "panel_hash": "test",
            "universe_name": "test",
            "start": "2020-01-01",
            # deliberately no "end"
        },
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 1},
    )

    def _boom(*args, **kwargs):
        raise AssertionError(
            "HoldoutLock must never be constructed when contract.data has no 'end'"
        )

    monkeypatch.setattr(feature_sweep_module, "HoldoutLock", _boom)

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1],
        feature_registry_override=[("dummy", _dummy_feature)],
    )
    assert len(records) == 1
    assert records[0].error is None


# ---------------------------------------------------------------------------
# `_is_monotonic` -- fewer than two finite values is not monotonic (line 308).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1.0],
        [float("nan"), 2.0],
        [float("nan"), float("nan")],
    ],
)
def test_is_monotonic_false_with_fewer_than_two_finite_values(values):
    assert _is_monotonic(values) is False


def test_is_monotonic_true_and_false_on_real_sequences():
    assert _is_monotonic([1.0, 2.0, 3.0]) is True
    assert _is_monotonic([3.0, 2.0, 1.0]) is True
    assert _is_monotonic([1.0, 3.0, 2.0]) is False


# ---------------------------------------------------------------------------
# `evaluate_promotion` / `_e4_metrics` -- deflated Sharpe is NaN, not a fabricated
# number, when the recent window yields fewer than 4 spread observations (line 361).
# ---------------------------------------------------------------------------


def test_evaluate_promotion_reports_nan_dsr_on_too_few_recent_observations():
    n_symbols = 5
    rng = np.random.default_rng(5)
    # Session 1: 30 rows of real data (gives the pooled window plenty of history).
    # Session 2: 2 rows only -- the RECENT window this test asks for.
    s1 = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, size=(30, n_symbols)).astype(np.float64), axis=0)
    )
    s2 = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, size=(2, n_symbols)).astype(np.float64), axis=0)
    )
    close = np.concatenate([s1, s2], axis=0)
    day_offsets = np.array([0, 30, 32], dtype=np.int64)
    feature = rng.normal(size=close.shape).astype(np.float64)

    result = evaluate_promotion(
        feature=feature,
        close=close,
        day_offsets=day_offsets,
        horizon=1,
        recent_n_sessions=1,  # only the 2-row session
        n_buckets=5,
    )

    assert np.isnan(result.recent["deflated_sharpe"])
    assert bool(result.conditions["deflated_sharpe"]) is False
    assert result.promoted is False


# ---------------------------------------------------------------------------
# `expectancy.causal_buckets(method="expanding_quantile")`:
#   - a symbol with no finite prior data yet at time t (152->150 branch)
#   - a genuinely constant symbol backfilling its pre-min_history rows (173-174)
# ---------------------------------------------------------------------------


def test_expanding_quantile_skips_bucketing_with_no_prior_finite_data():
    """Symbol 0 has real data throughout; symbol 1 is NaN for its first 6 rows (so at
    t=3 -- inside the loop, min_history=3 -- there is no prior finite data for symbol 1
    at all) and only becomes finite from row 6 onward."""
    n_rows = 12
    feature = np.full((n_rows, 2), np.nan, dtype=np.float64)
    feature[:, 0] = np.arange(n_rows, dtype=np.float64)
    feature[6:, 1] = np.arange(n_rows - 6, dtype=np.float64) + 1.0
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    result = expectancy.causal_buckets(
        feature, day_offsets, n_buckets=3, method="expanding_quantile", min_history=3
    )

    # Symbol 1: every row before it ever has ANY prior finite value must stay
    # unassigned (-1), since `finite_vals` is empty for those (t, s=1) pairs.
    assert np.all(result.labels[3:6, 1] == -1)
    # Once prior finite data exists for symbol 1 (from row 7 onward it has row 6
    # behind it), it gets bucketed.
    assert result.labels[7, 1] != -1
    # Symbol 0 has data the whole time, so it is bucketed from min_history onward.
    assert np.all(result.labels[3:, 0] != -1)


def test_expanding_quantile_backfills_constant_symbol_before_min_history():
    """A symbol whose feature value never changes: quantile edges collapse, but the
    module's degenerate-constant-feature correction must still assign every row
    (including rows before `min_history`) to a single, consistent bucket."""
    n_rows = 10
    n_symbols = 2
    feature = np.empty((n_rows, n_symbols), dtype=np.float64)
    feature[:, 0] = 7.0  # constant for every row, every t
    feature[:, 1] = np.arange(n_rows, dtype=np.float64)  # varying, for contrast
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    result = expectancy.causal_buckets(
        feature, day_offsets, n_buckets=4, method="expanding_quantile", min_history=4
    )

    # The main loop only bucket-assigns rows >= min_history; the constant-symbol
    # correction must have backfilled rows [0, min_history) to match.
    assert np.all(result.labels[:4, 0] == result.labels[4, 0])
    assert result.labels[4, 0] != -1
    # The varying symbol is untouched by the constant-feature correction: it stays
    # unassigned before min_history.
    assert np.all(result.labels[:4, 1] == -1)


# ---------------------------------------------------------------------------
# `_block_bootstrap_means_chunked` -- the chunked block-bootstrap internals
# (expectancy.py lines 570-571, 578->580, 584-597, 621->626, 628->602).
# ---------------------------------------------------------------------------


def test_chunked_bootstrap_falls_back_to_iid_rows_when_every_session_too_short():
    """`block_length = horizon + BLOCK_LENGTH_EXTRA_BARS` (5 here, horizon=1 -> 6).
    Four sessions of 3 rows each are all shorter than that, so EVERY session must be
    skipped (n_sessions_skipped == 4) and the function must fall back to the
    iid-row resampling path (block_indices stays empty: no block was ever drawn).
    Also exercises the explicit-`chunk_size` branch (skips auto-derivation)."""
    n_rows = 12
    day_offsets = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    values_2d = np.arange(n_rows, dtype=np.float64).reshape(n_rows, 1) + 1.0

    boot_means, block_indices, n_sessions_skipped = expectancy._block_bootstrap_means_chunked(
        values_2d, day_offsets, horizon=1, n_boot=20, seed=7, chunk_size=6
    )

    assert n_sessions_skipped == 4
    assert block_indices == ()
    assert boot_means.shape == (20,)
    # Every input value is finite, so the iid-row fallback must produce finite means.
    assert np.all(np.isfinite(boot_means))


def test_chunked_bootstrap_iid_fallback_reports_nan_for_all_nan_chunks():
    """Same too-short-session fallback as above, but with entirely NaN input: every
    resampled row in every chunk is NaN, so `np.any(valid_rows)` must be False for
    every chunk -- `boot_means` must come back entirely NaN, not zero or fabricated."""
    n_rows = 12
    day_offsets = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    values_2d = np.full((n_rows, 1), np.nan, dtype=np.float64)

    boot_means, block_indices, n_sessions_skipped = expectancy._block_bootstrap_means_chunked(
        values_2d, day_offsets, horizon=1, n_boot=20, seed=8, chunk_size=6
    )

    assert n_sessions_skipped == 4
    assert block_indices == ()
    assert np.all(np.isnan(boot_means))


def test_chunked_bootstrap_caps_block_indices_and_skips_all_nan_chunks(monkeypatch):
    """Monkeypatches the module's own diagnostic-retention budget down to a size
    reachable without materialising hundreds of thousands of blocks (the real
    constant, `_BLOCK_INDICES_MAX_RETAINED`, is measured against a 64 MB budget and
    is deliberately far too large to exhaust honestly in a unit test). With the
    budget patched to 1 entry, the SECOND chunk onward must skip retention
    (remaining_capacity <= 0). All input values are NaN, so no chunk ever has a
    finite row, exercising the "no valid rows this chunk" branch on every
    iteration."""
    monkeypatch.setattr(expectancy, "_BLOCK_INDICES_MAX_RETAINED", 1)

    n_rows = 20
    day_offsets = np.array([0, n_rows], dtype=np.int64)  # one long session
    values_2d = np.full((n_rows, 1), np.nan, dtype=np.float64)

    boot_means, block_indices, n_sessions_skipped = expectancy._block_bootstrap_means_chunked(
        values_2d, day_offsets, horizon=1, n_boot=4, seed=11, chunk_size=1
    )

    assert n_sessions_skipped == 0
    # Retention capped at the patched budget, even though 4 chunks each drew blocks.
    assert len(block_indices) == 1
    # No chunk had any finite row (values_2d is entirely NaN): every mean is NaN.
    assert np.all(np.isnan(boot_means))


# ---------------------------------------------------------------------------
# `_bucket_row_means` -- a bucket-returns panel with no finite cell anywhere must
# come back entirely NaN, not zero (expectancy.py line 669->671).
# ---------------------------------------------------------------------------


def test_bucket_row_means_all_nan_when_no_row_has_any_valid_data():
    from nifty_quant.research.expectancy import _bucket_row_means

    bucket_returns_2d = np.full((6, 3), np.nan, dtype=np.float64)
    row_means = _bucket_row_means(bucket_returns_2d)

    assert row_means.shape == (6,)
    assert np.all(np.isnan(row_means))


def test_bucket_row_means_real_values_when_some_rows_have_data():
    from nifty_quant.research.expectancy import _bucket_row_means

    bucket_returns_2d = np.array(
        [
            [1.0, np.nan, 3.0],
            [np.nan, np.nan, np.nan],
            [2.0, 4.0, np.nan],
        ]
    )
    row_means = _bucket_row_means(bucket_returns_2d)

    np.testing.assert_allclose(row_means[0], 2.0)
    assert np.isnan(row_means[1])
    np.testing.assert_allclose(row_means[2], 3.0)


# ---------------------------------------------------------------------------
# `_compute_bucket_stats` block_bootstrap: se_bps falls back to 0.0 when fewer than
# two finite bootstrap means come back (expectancy.py line 775) -- distinct from
# fabricating a nonzero SE off a single noisy draw.
# ---------------------------------------------------------------------------


def test_compute_bucket_stats_block_bootstrap_se_zero_with_single_boot_mean():
    from nifty_quant.research.expectancy import _compute_bucket_stats

    n_rows, n_symbols = 40, 2
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    rng = np.random.default_rng(13)
    bucket_returns = rng.normal(0.0, 0.001, size=(n_rows, n_symbols)).flatten()

    stat = _compute_bucket_stats(
        bucket_returns,
        horizon=5,
        day_offsets=day_offsets,
        se_method="block_bootstrap",
        n_boot=1,  # exactly one replicate -> len(valid_boot_means) <= 1
        seed=13,
    )

    assert stat is not None
    assert stat.se_bps == 0.0
