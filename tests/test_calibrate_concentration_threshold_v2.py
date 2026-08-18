"""Tests for scripts/calibrate_concentration_threshold_v2.py's pure helper logic.

Authored under ``scripts/`` (not ``tests/``) because the task that produced
``calibrate_concentration_threshold_v2.py`` was explicitly scoped to touch nothing under
``tests/`` while v1's calibration run was in flight against the same panel -- see
``tests/test_calibrate_concentration_threshold.py``'s header for the identical precedent on
v1. ``pyproject.toml`` sets ``testpaths = ["tests"]``, so this file is not auto-collected;
Moved into ``tests/`` at review, matching v1's file, so pytest actually collects it.

Focus: the pieces that actually CHANGED from v1 -- ``_prior_session_mean`` (strictly-prior
per-symbol session lag) and ``_liquidity_deciles`` (causal cross-sectional-rank bucketing of
rupee turnover, replacing v1's full-sample pooled-quantile bucketing of raw share volume).
Two tests import v1's ``_liquidity_deciles`` directly and assert it does NOT have the
causal (-1 at session 0) property v2 requires -- i.e. they are proven to fail if v2's
function were reverted to just delegate to v1's, which is exactly the "pre-change code"
mutation this file guards against. The unchanged-logic pieces (``_concentration_stat``,
``_permute_within_sessions``, ``_decile_spreads`` vs ``_decile_spreads_fast``) are also
covered, re-run against the v2 module object to confirm the copy-and-modify didn't disturb
them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


calib_v2 = _load("calibrate_concentration_threshold_v2", "calibrate_concentration_threshold_v2.py")
calib_v1 = _load(
    "calibrate_concentration_threshold_v1_for_mutation_check",
    "calibrate_concentration_threshold.py",
)


# ---------------------------------------------------------------------------
# _prior_session_mean -- copied from scripts/recon_h2_liquidity_units.py
# ---------------------------------------------------------------------------


def test_prior_session_mean_is_strictly_prior_and_session_0_is_nan():
    # 3 sessions x 2 rows, 2 symbols.
    values = np.array(
        [
            [1.0, 10.0],  # session 0
            [3.0, 20.0],  # session 0
            [5.0, 2.0],  # session 1
            [7.0, 4.0],  # session 1
            [100.0, 100.0],  # session 2 (must NOT leak into its own prior)
            [100.0, 100.0],  # session 2
        ]
    )
    day_offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    prior = calib_v2._prior_session_mean(values, day_offsets)

    # Session 0: no prior sessions -> NaN.
    assert np.all(np.isnan(prior[0:2]))
    # Session 1: prior = session 0's per-day mean = [2.0, 15.0].
    np.testing.assert_allclose(prior[2], [2.0, 15.0])
    np.testing.assert_allclose(prior[3], [2.0, 15.0])
    # Session 2: prior = mean of session-0 and session-1 per-day means
    #   symbol0: mean(2.0, 6.0) = 4.0 ; symbol1: mean(15.0, 3.0) = 9.0
    np.testing.assert_allclose(prior[4], [4.0, 9.0])
    np.testing.assert_allclose(prior[5], [4.0, 9.0])


# ---------------------------------------------------------------------------
# _liquidity_deciles (v2) -- causal cross-sectional-rank bucketing of rupee
# turnover, replacing v1's full-sample pooled-quantile bucketing of raw volume.
# ---------------------------------------------------------------------------


def test_liquidity_deciles_v2_is_undefined_at_session_0():
    # session 0 has no prior sessions, so every cell must be -1 regardless of how
    # "liquid" that session's own (unlagged) turnover looks -- this is the causality
    # fix v2 makes over v1, which bucketed every finite-volume row including the first.
    turnover = np.array(
        [
            [100.0, 200.0, 300.0, 400.0, 500.0],  # session 0
            [1.0, 1.0, 1.0, 1.0, 1.0],  # session 1, row 0 (values irrelevant to this test)
            [1.0, 1.0, 1.0, 1.0, 1.0],  # session 1, row 1
        ]
    )
    day_offsets = np.array([0, 1, 3], dtype=np.int32)
    deciles = calib_v2._liquidity_deciles(turnover, day_offsets)
    assert np.all(deciles[0] == -1)


def test_liquidity_deciles_v2_buckets_by_causal_cross_sectional_rank():
    # Session 0 (single row) establishes distinct turnover values for 5 symbols; session
    # 1's rows must be bucketed by the CROSS-SECTIONAL RANK of session 0's mean (the
    # strictly-prior quantity), not by any full-sample quantile of raw values.
    # 5 symbols, ascending turnover -> pct ranks 0, .25, .5, .75, 1.0 -> deciles 0,2,5,7,9
    # (clip(int(pct*10), 0, 9)).
    turnover = np.array(
        [
            [100.0, 200.0, 300.0, 400.0, 500.0],  # session 0
            [np.nan] * 5,  # session 1, row 0 (own-row values irrelevant to bucketing)
            [np.nan] * 5,  # session 1, row 1
        ]
    )
    day_offsets = np.array([0, 1, 3], dtype=np.int32)
    deciles = calib_v2._liquidity_deciles(turnover, day_offsets)
    expected = np.array([0, 2, 5, 7, 9], dtype=np.int8)
    np.testing.assert_array_equal(deciles[1], expected)
    np.testing.assert_array_equal(deciles[2], expected)


def test_v1_liquidity_deciles_does_not_have_the_causal_session_0_property():
    """Mutation-check: proves the two properties above are a real behavioural change, not
    an artifact of the test. Feeding v1's (pre-change) ``_liquidity_deciles`` the same
    finite, non-degenerate session-0 volume it would see if v2's function were reverted to
    just call v1's directly does NOT produce an all -1 first session -- v1 has no notion of
    "session" or "prior" at all; it buckets every finite cell from full-sample quantiles.
    So a revert of v2 to v1's logic fails ``test_liquidity_deciles_v2_is_undefined_at_session_0``.
    """
    volume = np.array(
        [
            [100.0, 200.0, 300.0, 400.0, 500.0],  # "session 0" by row position only
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    deciles = calib_v1._liquidity_deciles(volume)
    # v1 assigns real (non -1) deciles to every finite cell, including row 0 -- the
    # opposite of the causal property v2 requires.
    assert not np.all(deciles[0] == -1)
    assert np.all(np.isin(deciles[0], np.arange(10)))


# ---------------------------------------------------------------------------
# _concentration_stat / ConcentrationRecord -- unchanged from v1, re-verified on
# the v2 module object (mirrors research/lens.py 645-672, including the UPPER
# median convention this whole calibration exists to get right).
# ---------------------------------------------------------------------------


def test_concentration_stat_median_index_matches_production_even_length_rule():
    # sorted(|nonzero|) = [10, 20, 30, 40] (4 elements) -> production median = sorted[2] = 30
    # (the UPPER median), NOT np.median's 25.0.
    spreads = {0: -40.0, 1: 10.0, 2: 20.0, 3: -30.0}
    record = calib_v2._concentration_stat(spreads)
    assert record.ratio == pytest.approx(40.0 / 30.0)


def test_concentration_stat_fires_when_ratio_exceeds_threshold_and_bottom_is_argmax():
    spreads = {0: -30.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0,
               8: 10.0, 9: 10.0}
    record = calib_v2._concentration_stat(spreads)
    assert record.ratio == pytest.approx(3.0)
    assert record.fired(2.0) is True


def test_concentration_stat_does_not_fire_when_bottom_is_not_argmax():
    spreads = {0: 5.0, 1: 5.0, 2: 50.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 5.0,
               8: 5.0, 9: 5.0}
    record = calib_v2._concentration_stat(spreads)
    assert record.bottom_is_argmax is False
    assert record.fired(0.0) is False


# ---------------------------------------------------------------------------
# _permute_within_sessions -- unchanged from v1, re-verified on the v2 module.
# ---------------------------------------------------------------------------


def test_permute_within_sessions_uses_one_permutation_per_session_not_per_row():
    feature = np.array(
        [[0.0, 1.0, 2.0, 3.0],
         [10.0, 11.0, 12.0, 13.0],
         [20.0, 21.0, 22.0, 23.0]],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 3], dtype=np.int32)
    rng = np.random.default_rng(5)
    permuted = calib_v2._permute_within_sessions(feature, day_offsets, rng)
    col0_targets = {int(np.where(permuted[r] == feature[r, 0])[0][0]) for r in range(3)}
    assert len(col0_targets) == 1


# ---------------------------------------------------------------------------
# _decile_spreads vs _decile_spreads_fast -- unchanged logic, re-verified on the
# v2 module (the performance path the 500-replicate loop actually relies on).
# ---------------------------------------------------------------------------


def test_decile_spreads_fast_matches_reference_slow_form_v2():
    from nifty_quant.research import expectancy

    n_rows, n_symbols = 60, 12
    day_offsets = np.array([0, 25, 60], dtype=np.int32)
    rng = np.random.default_rng(123)

    liquidity_deciles = rng.integers(0, 3, size=(n_rows, n_symbols)).astype(np.int8)
    liquidity_deciles[rng.random((n_rows, n_symbols)) < 0.15] = -1

    feature = rng.normal(size=(n_rows, n_symbols))
    fwd_values = rng.normal(scale=0.01, size=(n_rows, n_symbols))
    fwd = expectancy.ForwardReturns(
        values=fwd_values, horizon=1, session_bounded=True, n_defined=n_rows, n_nan_tail=0
    )

    reference = calib_v2._decile_spreads(feature, fwd, liquidity_deciles, day_offsets, horizon=1)
    layouts = calib_v2._precompute_decile_layouts(liquidity_deciles, fwd)
    fast = calib_v2._decile_spreads_fast(feature, layouts, day_offsets, horizon=1)

    assert set(reference) == set(fast)
    for d in reference:
        assert fast[d] == pytest.approx(reference[d], abs=1e-9)
