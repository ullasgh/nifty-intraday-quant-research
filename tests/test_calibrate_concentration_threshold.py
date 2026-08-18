"""Tests for scripts/calibrate_concentration_threshold.py's pure helper logic.

Originally written under ``scripts/`` because the authoring task was scoped to touch
nothing under ``tests/``. Moved here at review: ``pyproject.toml`` sets
``testpaths = ["tests"]``, so a file left in ``scripts/`` is never collected and therefore
protects nothing.

Covers the three pieces of this script that are NOT a direct passthrough to already-tested
production code: the criterion-4 mirror (``_concentration_stat`` /
``ConcentrationRecord.fired``, since that logic lives inline inside ``Lens.verdict()`` and
has no standalone test target to compare against), the vectorized liquidity-decile port
(``_liquidity_deciles``, checked against a brute-force reference identical in form to
``research/lens.py``'s original per-cell loop), and the within-session permutation
(``_permute_within_sessions``) that defines the null construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "calibrate_concentration_threshold.py"
)
_spec = importlib.util.spec_from_file_location("calibrate_concentration_threshold", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calib = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = calib
_spec.loader.exec_module(calib)


# ---------------------------------------------------------------------------
# ConcentrationRecord / _concentration_stat -- mirrors research/lens.py 645-672
# ---------------------------------------------------------------------------


def test_fires_when_ratio_exceeds_threshold_and_bottom_is_argmax():
    # Bottom decile (0) has the largest |spread|, 3x the median of the rest.
    spreads = {0: -30.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0,
               8: 10.0, 9: 10.0}
    record = calib._concentration_stat(spreads)
    assert record.ratio == pytest.approx(3.0)
    assert record.bottom_is_argmax is True
    assert record.fired(2.0) is True


def test_does_not_fire_when_ratio_at_or_below_threshold():
    # Bottom is argmax but only exactly 2x the median -> strict ">" must not fire.
    spreads = {0: -20.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0,
               8: 10.0, 9: 10.0}
    record = calib._concentration_stat(spreads)
    assert record.ratio == pytest.approx(2.0)
    assert record.fired(2.0) is False
    assert record.fired(1.9) is True


def test_does_not_fire_when_bottom_is_not_argmax():
    # A middle decile is the true outlier; bottom is unremarkable.
    spreads = {0: 5.0, 1: 5.0, 2: 50.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 5.0,
               8: 5.0, 9: 5.0}
    record = calib._concentration_stat(spreads)
    assert record.bottom_is_argmax is False
    assert record.fired(2.0) is False
    assert record.fired(0.0) is False  # even a permissive threshold can't fire without argmax


def test_never_fires_when_bottom_decile_missing_or_zero():
    spreads_missing = {1: 40.0, 2: 5.0, 3: 5.0}
    record = calib._concentration_stat(spreads_missing)
    assert record.fired(0.0) is False

    spreads_zero_bottom = {0: 0.0, 1: 40.0, 2: 5.0, 3: 5.0}
    record = calib._concentration_stat(spreads_zero_bottom)
    assert record.fired(0.0) is False


def test_only_bottom_nonzero_fires_at_any_threshold():
    spreads = {0: -12.0, 1: 0.0, 2: 0.0, 3: 0.0}
    record = calib._concentration_stat(spreads)
    assert record.only_bottom_nonzero is True
    assert record.fired(2.0) is True
    assert record.fired(1e9) is True  # matches lens.py: this branch ignores the ratio gate


def test_median_index_matches_production_even_length_rule():
    # sorted(|nonzero|) = [10, 20, 30, 40] (4 elements, bottom is the 40).
    # Production uses sorted[len // 2] = sorted[2] = 30, NOT the true median (25.0).
    spreads = {0: -40.0, 1: 10.0, 2: 20.0, 3: -30.0}
    record = calib._concentration_stat(spreads)
    assert record.ratio == pytest.approx(40.0 / 30.0)


# ---------------------------------------------------------------------------
# _liquidity_deciles -- vectorized port of research/lens.py 470-484
# ---------------------------------------------------------------------------


def _brute_force_deciles(volume: np.ndarray) -> np.ndarray:
    """Independent, deliberately non-vectorized reimplementation of research/lens.py's
    original per-cell loop (lines 470-484), used only as a test oracle.
    """
    finite = np.isfinite(volume)
    quantiles = np.quantile(volume[finite], np.linspace(0, 1, 11))
    out = np.full(volume.shape, -1, dtype=np.int8)
    for i in range(volume.shape[0]):
        for s in range(volume.shape[1]):
            vol = volume[i, s]
            if np.isfinite(vol):
                out[i, s] = np.searchsorted(quantiles[1:-1], vol, side="right")
    return out


def test_liquidity_deciles_matches_brute_force_with_nans_and_ties():
    rng = np.random.default_rng(7)
    volume = rng.exponential(scale=1000.0, size=(60, 9))
    volume[3, 2] = np.nan
    volume[10, :] = np.nan
    volume[0, 0] = volume[1, 0]  # a tie, to exercise side="right"
    got = calib._liquidity_deciles(volume)
    want = _brute_force_deciles(volume)
    np.testing.assert_array_equal(got, want)


def test_liquidity_deciles_uses_side_right_on_exact_quantile_edges():
    # 11 evenly spaced integers 0..10: np.quantile(..., linspace(0, 1, 11)) lands
    # EXACTLY on the data values (quantile_k == k), so decile edges are 1..9 exactly,
    # coinciding with real data values -- the only way to force the side="right" vs
    # side="left" distinction (research/lens.py line 482 uses side="right") to actually
    # matter, since with continuous data quantile edges almost never exactly equal an
    # observation.
    volume = np.arange(11, dtype=np.float64).reshape(1, 11)
    deciles = calib._liquidity_deciles(volume)[0]
    # side="right": searchsorted([1..9], v, side="right") -- v==1 falls in decile 1
    # (after the edge), not decile 0. side="left" would instead put v==1 in decile 0.
    assert deciles[1] == 1
    assert deciles[9] == 9
    # And it must still agree with the same brute-force oracle used elsewhere.
    np.testing.assert_array_equal(deciles, _brute_force_deciles(volume)[0])


def test_check_liquidity_deciles_raises_on_real_mismatch():
    volume = np.abs(np.random.default_rng(3).normal(size=(20, 4))) + 1.0
    deciles = calib._liquidity_deciles(volume)
    tampered = deciles.copy()
    tampered[0, 0] = (tampered[0, 0] + 1) % 10
    with pytest.raises(AssertionError):
        calib._check_liquidity_deciles(volume, tampered, n_check_rows=20)
    # The real (untampered) deciles must pass.
    calib._check_liquidity_deciles(volume, deciles, n_check_rows=20)


# ---------------------------------------------------------------------------
# _permute_within_sessions -- the null construction
# ---------------------------------------------------------------------------


def test_permute_within_sessions_is_a_within_row_permutation():
    # Two sessions of 3 rows each, 4 symbols; each cell has a unique value so any
    # reordering is detectable.
    feature = np.arange(24, dtype=np.float64).reshape(6, 4)
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    rng = np.random.default_rng(11)
    permuted = calib._permute_within_sessions(feature, day_offsets, rng)

    # Every row's value set is unchanged (same multiset, just reassigned columns).
    for row in range(6):
        np.testing.assert_array_equal(
            np.sort(permuted[row]), np.sort(feature[row])
        )
    # It actually did something (not the identity) for this seed.
    assert not np.array_equal(permuted, feature)


def test_permute_within_sessions_uses_one_permutation_per_session_not_per_row():
    # Within a session, the SAME column permutation is applied to every row -- so the
    # column that receives symbol 0's original value must be identical across all rows
    # of that session.
    feature = np.array(
        [[0.0, 1.0, 2.0, 3.0],
         [10.0, 11.0, 12.0, 13.0],
         [20.0, 21.0, 22.0, 23.0]],
        dtype=np.float64,
    )
    day_offsets = np.array([0, 3], dtype=np.int32)
    rng = np.random.default_rng(5)
    permuted = calib._permute_within_sessions(feature, day_offsets, rng)

    # Column 0's original values were 0, 10, 20; find which output column now holds them.
    col0_targets = {int(np.where(permuted[r] == feature[r, 0])[0][0]) for r in range(3)}
    assert len(col0_targets) == 1  # same destination column in every row of the session


def test_permute_within_sessions_does_not_cross_session_boundary():
    feature = np.arange(16, dtype=np.float64).reshape(4, 4)
    day_offsets = np.array([0, 1, 4], dtype=np.int32)  # session 0: row 0; session 1: rows 1-3
    rng = np.random.default_rng(2)
    permuted = calib._permute_within_sessions(feature, day_offsets, rng)

    # Session 0 has one row/one possible permutation of 4 columns: still row 0's own values.
    np.testing.assert_array_equal(np.sort(permuted[0]), np.sort(feature[0]))
    # Session 1 rows must each be permutations of THEIR OWN row, never row 0's values.
    for row in (1, 2, 3):
        np.testing.assert_array_equal(np.sort(permuted[row]), np.sort(feature[row]))


def test_permute_within_sessions_is_seed_deterministic():
    feature = np.random.default_rng(1).normal(size=(9, 5))
    day_offsets = np.array([0, 4, 9], dtype=np.int32)
    out1 = calib._permute_within_sessions(feature, day_offsets, np.random.default_rng(99))
    out2 = calib._permute_within_sessions(feature, day_offsets, np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# _decile_spreads -- integration with the real (imported, not reimplemented)
# nifty_quant.research.expectancy.conditional_expectancy
# ---------------------------------------------------------------------------


def test_decile_spreads_isolates_deciles_and_recovers_planted_effect():
    from nifty_quant.research import expectancy

    n_rows, n_symbols = 40, 10
    day_offsets = np.array([0, 20, 40], dtype=np.int32)
    rng = np.random.default_rng(0)

    # Two liquidity deciles (0 and 1), 5 symbols each. Decile 0 has NO feature-return
    # relationship (pure noise); decile 1 has a strong monotone one.
    volume_deciles = np.zeros((n_rows, n_symbols), dtype=np.int8)
    volume_deciles[:, 5:] = 1

    feature = rng.normal(size=(n_rows, n_symbols))
    fwd_values = rng.normal(scale=1e-6, size=(n_rows, n_symbols))
    # Plant a strong positive feature -> return relationship only in decile 1's columns.
    fwd_values[:, 5:] += 0.01 * feature[:, 5:]

    fwd = expectancy.ForwardReturns(
        values=fwd_values,
        horizon=1,
        session_bounded=True,
        n_defined=n_rows,
        n_nan_tail=0,
    )

    spreads = calib._decile_spreads(feature, fwd, volume_deciles, day_offsets, horizon=1)

    assert set(spreads) == {0, 1}
    assert abs(spreads[1]) > abs(spreads[0])  # planted decile shows the larger spread
    assert spreads[1] > 0  # positive feature -> positive return, so top-minus-bottom > 0


def test_decile_spreads_fast_matches_reference_slow_form():
    """`_decile_spreads_fast` (the precomputed-layout, narrow-array path used for the
    actual 500+-replicate loop) must produce IDENTICAL spread_bps to `_decile_spreads`
    (the full-width, lens.py-faithful reference) -- this is the equivalence the whole
    performance optimization depends on.
    """
    from nifty_quant.research import expectancy

    n_rows, n_symbols = 60, 12
    day_offsets = np.array([0, 25, 60], dtype=np.int32)
    rng = np.random.default_rng(123)

    # Uneven, row-varying liquidity decile membership (some cells NaN/-1 = no decile),
    # to exercise the "not every row has the same member count" path that makes `width`
    # meaningful (rather than every row having identical membership).
    volume_deciles = rng.integers(0, 3, size=(n_rows, n_symbols)).astype(np.int8)
    # Scatter in some "no decile assigned" cells (-1), matching real non-finite-volume rows.
    volume_deciles[rng.random((n_rows, n_symbols)) < 0.15] = -1

    feature = rng.normal(size=(n_rows, n_symbols))
    fwd_values = rng.normal(scale=0.01, size=(n_rows, n_symbols))
    fwd = expectancy.ForwardReturns(
        values=fwd_values, horizon=1, session_bounded=True, n_defined=n_rows, n_nan_tail=0
    )

    reference = calib._decile_spreads(feature, fwd, volume_deciles, day_offsets, horizon=1)
    layouts = calib._precompute_decile_layouts(volume_deciles, fwd)
    fast = calib._decile_spreads_fast(feature, layouts, day_offsets, horizon=1)

    assert set(reference) == set(fast)
    for decile in reference:
        assert fast[decile] == pytest.approx(reference[decile], abs=1e-9), decile


def test_decile_layout_arrays_are_compacted_to_width_not_full_symbol_count():
    """`_DecileLayout.sort_idx`/`valid`/`compact_fwd` must be pre-truncated to `width`
    columns, not the full symbol count -- this is a structural (non-flaky) stand-in for
    the actual perf requirement (a timing assertion would be flaky per CLAUDE.md rule 9):
    if this regresses back to full width, `_decile_spreads_fast` stays numerically correct
    but silently loses the entire point of precomputing the layout.
    """
    from nifty_quant.research import expectancy

    n_rows, n_symbols = 50, 20
    rng = np.random.default_rng(4)
    # Decile 0 membership is deliberately narrow (~15% of columns per row).
    volume_deciles = (rng.random((n_rows, n_symbols)) < 0.15).astype(np.int8) - 1
    volume_deciles[volume_deciles == -1] = 5  # everything else in an unrelated decile
    volume_deciles[rng.random((n_rows, n_symbols)) < 0.15] = 0
    fwd_values = rng.normal(scale=0.01, size=(n_rows, n_symbols))
    fwd = expectancy.ForwardReturns(
        values=fwd_values, horizon=1, session_bounded=True, n_defined=n_rows, n_nan_tail=0
    )
    layouts = calib._precompute_decile_layouts(volume_deciles, fwd)
    for decile, layout in layouts.items():
        assert layout.sort_idx.shape == (n_rows, layout.width)
        assert layout.valid.shape == (n_rows, layout.width)
        assert layout.compact_fwd.shape == (n_rows, layout.width)
        assert layout.width < n_symbols, (
            f"decile {decile} layout was not compacted below the full symbol count"
        )


def test_decile_spreads_fast_layout_is_reused_correctly_across_different_features():
    """A single precomputed layout (built from one `fwd`/`volume_deciles` pair) must give
    correct, differing results when fed two different (permuted) feature arrays -- this is
    exactly how the real replicate loop reuses one layout across many permutations.
    """
    from nifty_quant.research import expectancy

    n_rows, n_symbols = 30, 8
    day_offsets = np.array([0, 15, 30], dtype=np.int32)
    rng = np.random.default_rng(9)
    volume_deciles = rng.integers(0, 2, size=(n_rows, n_symbols)).astype(np.int8)
    fwd_values = rng.normal(scale=0.01, size=(n_rows, n_symbols))
    fwd = expectancy.ForwardReturns(
        values=fwd_values, horizon=1, session_bounded=True, n_defined=n_rows, n_nan_tail=0
    )
    layouts = calib._precompute_decile_layouts(volume_deciles, fwd)

    feature_a = rng.normal(size=(n_rows, n_symbols))
    feature_b = rng.normal(size=(n_rows, n_symbols))
    result_a = calib._decile_spreads_fast(feature_a, layouts, day_offsets, horizon=1)
    result_b = calib._decile_spreads_fast(feature_b, layouts, day_offsets, horizon=1)

    # Each individually matches the slow reference computed with its own feature.
    ref_a = calib._decile_spreads(feature_a, fwd, volume_deciles, day_offsets, horizon=1)
    ref_b = calib._decile_spreads(feature_b, fwd, volume_deciles, day_offsets, horizon=1)
    for decile in ref_a:
        assert result_a[decile] == pytest.approx(ref_a[decile], abs=1e-9)
    for decile in ref_b:
        assert result_b[decile] == pytest.approx(ref_b[decile], abs=1e-9)
    # And the two different features generally give different results (sanity: the reused
    # layout is not silently caching stale output).
    assert any(
        abs(result_a[d] - result_b[d]) > 1e-9 for d in result_a if d in result_b
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
