"""Red-state TDD spec file for `nifty_quant.research.cv`.

The module under test does not exist yet; importing it here makes the whole
file fail at collection with an ImportError, which is the expected red state.
"""

import itertools
import math

import numpy as np
import pytest

from nifty_quant.backtest.metrics import pbo_cscv
from nifty_quant.research.cv import CombinatorialPurgedCV, Fold, PurgedKFold


def _assert_train_test_disjoint(fold: Fold) -> None:
    assert np.intersect1d(fold.train_idx, fold.test_idx, assume_unique=True).size == 0


def _assert_no_forward_label_window_overlap(fold: Fold, label_horizon: int) -> None:
    test_set = set(fold.test_idx.tolist())
    for i in fold.train_idx.tolist():
        for j in range(i, i + label_horizon + 1):
            assert j not in test_set


@pytest.mark.parametrize("n_splits", [2, 3, 5, 8])
def test_train_and_test_are_always_disjoint(n_splits: int) -> None:
    folds = PurgedKFold(n_splits=n_splits).split(240)
    for fold in folds:
        _assert_train_test_disjoint(fold)


def test_no_train_label_window_overlaps_test_block() -> None:
    kf = PurgedKFold(n_splits=5, label_horizon=10)
    folds = kf.split(500)
    for fold in folds:
        _assert_no_forward_label_window_overlap(fold, label_horizon=10)


def test_label_horizon_zero_reduces_to_contiguous_kfold() -> None:
    n_rows = 500
    # embargo_frac=0.0 always produced width=0 regardless of n_rows; the absolute
    # equivalent is simply embargo_sessions=0.
    kf = PurgedKFold(n_splits=5, label_horizon=0, embargo_sessions=0)
    folds = kf.split(n_rows)
    for fold in folds:
        complement = np.setdiff1d(np.arange(n_rows), fold.test_idx, assume_unique=True)
        assert np.array_equal(fold.train_idx, complement)
        assert fold.n_purged == 0
        assert fold.n_embargoed == 0


def test_larger_label_horizon_purges_strictly_more() -> None:
    # embargo_frac=0.0 always produced width=0 regardless of n_rows; the absolute
    # equivalent is simply embargo_sessions=0.
    kf5 = PurgedKFold(n_splits=5, label_horizon=5, embargo_sessions=0)
    kf20 = PurgedKFold(n_splits=5, label_horizon=20, embargo_sessions=0)
    total5 = sum(fold.n_purged for fold in kf5.split(500))
    total20 = sum(fold.n_purged for fold in kf20.split(500))
    assert total20 > total5


def test_embargo_applies_after_test_block_not_before() -> None:
    n_rows = 500
    # embargo_frac=0.02 on n_rows=500 produced width = ceil(0.02 * 500) = 10 rows.
    # split() below is called without day_offsets, so embargo_sessions maps 1:1
    # onto rows (per cv._embargo_width_rows).
    embargo_sessions = 10
    width = embargo_sessions
    folds = PurgedKFold(
        n_splits=5, label_horizon=0, embargo_sessions=embargo_sessions
    ).split(n_rows)
    fold = next(
        f for f in folds if f.test_idx[0] > 0 and f.test_idx[-1] + 1 + width <= n_rows
    )
    test_start = int(fold.test_idx[0])
    test_end = int(fold.test_idx[-1]) + 1

    assert test_start - 1 in fold.train_idx
    for i in range(test_end, test_end + width):
        assert i not in fold.train_idx


def test_embargo_sessions_zero_purges_nothing_extra() -> None:
    # embargo_frac=0.0 always produced width=0 regardless of n_rows; the absolute
    # equivalent is simply embargo_sessions=0.
    folds = PurgedKFold(n_splits=5, label_horizon=0, embargo_sessions=0).split(500)
    for fold in folds:
        assert fold.n_embargoed == 0
        assert fold.n_purged == 0


def test_n_purged_and_n_embargoed_are_reported_accurately() -> None:
    n_rows = 500
    n_splits = 5
    label_horizon = 5
    # embargo_frac=0.02 on n_rows=500 produced width = ceil(0.02 * 500) = 10 rows.
    # split() below is called without day_offsets, so embargo_sessions maps 1:1
    # onto rows (per cv._embargo_width_rows).
    embargo_sessions = 10
    width = embargo_sessions

    folds = PurgedKFold(
        n_splits=n_splits,
        label_horizon=label_horizon,
        embargo_sessions=embargo_sessions,
    ).split(n_rows)

    fold = next(
        f
        for f in folds
        if f.test_idx[0] >= label_horizon and f.test_idx[-1] + 1 + width <= n_rows
    )

    expected_purged = label_horizon
    expected_embargoed = width

    assert fold.n_purged == expected_purged
    assert fold.n_embargoed == expected_embargoed
    assert len(fold.train_idx) + fold.n_purged + fold.n_embargoed == (
        n_rows - len(fold.test_idx)
    )


def test_test_blocks_are_contiguous_and_in_time_order() -> None:
    folds = PurgedKFold(n_splits=5).split(500)
    for fold in folds:
        test_idx = fold.test_idx
        assert test_idx[-1] - test_idx[0] + 1 == len(test_idx)
        assert np.all(np.diff(test_idx) == 1)

    for left, right in zip(folds, folds[1:]):
        assert left.test_idx[0] < right.test_idx[0]


def test_test_blocks_cover_all_rows_exactly_once() -> None:
    n_rows = 500
    folds = PurgedKFold(n_splits=5).split(n_rows)
    all_test = np.concatenate([fold.test_idx for fold in folds])
    assert np.array_equal(np.sort(all_test), np.arange(n_rows))
    assert len(np.unique(all_test)) == n_rows


def test_folds_snap_to_session_boundaries_when_day_offsets_given() -> None:
    day_offsets = np.array([0, 100, 200, 300, 400, 500])
    folds = PurgedKFold(n_splits=5).split(500, day_offsets=day_offsets)
    offset_set = set(day_offsets.tolist())
    for fold in folds:
        start = int(fold.test_idx[0])
        end = int(fold.test_idx[-1]) + 1
        assert start in offset_set
        assert end in offset_set


def test_irregular_sessions_do_not_break_snapping() -> None:
    day_offsets = np.array([0, 375, 435])
    folds = PurgedKFold(n_splits=2).split(435, day_offsets=day_offsets)
    offset_set = set(day_offsets.tolist())
    for fold in folds:
        start = int(fold.test_idx[0])
        end = int(fold.test_idx[-1]) + 1
        assert start in offset_set
        assert end in offset_set


def test_never_shuffles() -> None:
    # embargo_frac=0.02 on n_rows=500 (the split() calls below) produced width =
    # ceil(0.02 * 500) = 10 rows; split() is called without day_offsets, so
    # embargo_sessions maps 1:1 onto rows (per cv._embargo_width_rows).
    kf = PurgedKFold(n_splits=5, label_horizon=10, embargo_sessions=10)
    folds1 = kf.split(500)
    folds2 = kf.split(500)

    assert len(folds1) == len(folds2)
    for left, right in zip(folds1, folds2):
        assert left.id == right.id
        assert np.array_equal(left.train_idx, right.train_idx)
        assert np.array_equal(left.test_idx, right.test_idx)

    assert all(
        folds1[i].test_idx[0] < folds1[i + 1].test_idx[0]
        for i in range(len(folds1) - 1)
    )


def test_rejects_n_splits_below_two() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)


def test_rejects_negative_label_horizon() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(label_horizon=-1)


def test_rejects_negative_embargo_sessions() -> None:
    # embargo_frac validated both bounds (0 <= frac < 1). embargo_sessions is an
    # absolute session count with no natural upper bound (unlike a row fraction,
    # nothing caps how many sessions may be embargoed), so __post_init__
    # (src/nifty_quant/research/cv.py:79-80) validates only the lower bound.
    # No equivalent upper-bound check exists to migrate; reported, not invented.
    with pytest.raises(ValueError):
        PurgedKFold(embargo_sessions=-1)


def test_rejects_n_rows_too_small_for_n_splits() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=10).split(5)


@pytest.mark.parametrize(
    "n_groups,n_test_groups",
    [(6, 2), (5, 1), (7, 3), (4, 2)],
)
def test_n_paths_matches_combinatorial_formula(n_groups: int, n_test_groups: int) -> None:
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)
    expected = math.comb(n_groups, n_test_groups) * n_test_groups // n_groups
    assert cv.n_paths() == expected


def test_number_of_folds_equals_n_choose_k() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    folds = cv.split(600)
    assert len(folds) == math.comb(6, 2)


def test_every_group_appears_in_test_the_same_number_of_times() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    folds = cv.split(600)
    group_size = 100
    counts = np.zeros(6, dtype=int)

    for fold in folds:
        groups = np.unique(fold.test_idx // group_size)
        assert len(groups) == 2
        for group in groups:
            counts[group] += 1

    assert np.all(counts == math.comb(5, 1))


def test_combinations_are_lexicographic_and_deterministic() -> None:
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
    folds = cv.split(500)
    observed = []

    for fold in folds:
        groups = tuple(sorted(np.unique(fold.test_idx // 100).tolist()))
        observed.append(groups)

    expected = list(itertools.combinations(range(5), 2))
    assert observed == expected


def test_cpcv_folds_also_purged_and_embargoed() -> None:
    # embargo_frac=0.01 on n_rows=600 (the split() call below) produced width =
    # ceil(0.01 * 600) = 6 rows; split() is called without day_offsets, so
    # embargo_sessions maps 1:1 onto rows (per cv._embargo_width_rows).
    cv = CombinatorialPurgedCV(
        n_groups=6,
        n_test_groups=2,
        label_horizon=10,
        embargo_sessions=6,
    )
    folds = cv.split(600)

    assert any(fold.n_purged > 0 for fold in folds)
    for fold in folds:
        _assert_train_test_disjoint(fold)
        _assert_no_forward_label_window_overlap(fold, label_horizon=10)


def test_rejects_n_test_groups_ge_n_groups() -> None:
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=5).n_paths()
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=6).n_paths()


def test_paths_matrix_shape_is_periods_by_n_paths() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    folds = cv.split(400)
    rng = np.random.default_rng(123)
    fold_results = [rng.normal(0.0, 0.01, size=100) for _ in folds]
    result = cv.paths(fold_results)
    assert result.shape == (100, 4)


def test_paths_matrix_contains_no_nan() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    folds = cv.split(400)
    rng = np.random.default_rng(456)
    fold_results = [rng.normal(0.0, 0.01, size=100) for _ in folds]
    result = cv.paths(fold_results)
    assert np.isnan(result).sum() == 0


def test_paths_raises_on_gaps_rather_than_filling() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    folds = cv.split(400)
    rng = np.random.default_rng(789)

    gap_results = [rng.normal(0.0, 0.01, size=100) for _ in folds]
    gap_results[1] = np.concatenate([gap_results[1], [0.0]])
    with pytest.raises(ValueError):
        cv.paths(gap_results)

    nan_results = [rng.normal(0.0, 0.01, size=100) for _ in folds]
    nan_results[1][0] = np.nan
    with pytest.raises(ValueError):
        cv.paths(nan_results)


def test_pbo_cscv_accepts_a_cpcv_paths_matrix() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=1)
    folds = cv.split(960)
    block_len = len(folds[0].test_idx)
    assert block_len == 160

    trial_columns = []
    for seed in [11, 22, 33, 44, 55]:
        rng = np.random.default_rng(seed)
        fold_results = [rng.normal(0.0, 0.01, size=block_len) for _ in folds]
        per_trial_paths = cv.paths(fold_results)
        trial_columns.append(per_trial_paths.reshape(-1))

    trial_matrix = np.column_stack(trial_columns)
    assert trial_matrix.shape == (960, 5)

    pbo = pbo_cscv(trial_matrix, n_splits=16)
    assert np.isfinite(pbo)
    assert 0.0 <= pbo <= 1.0


def test_pbo_is_high_for_overfit_trials_and_low_for_genuine() -> None:
    n_periods = 2400
    n_splits = 16

    rng_genuine = np.random.default_rng(0)
    strong_trial = rng_genuine.normal(0.01, 0.01, size=n_periods)
    noise_trials_genuine = rng_genuine.normal(0.0, 0.01, size=(n_periods, 9))
    genuine_matrix = np.column_stack((strong_trial, noise_trials_genuine))

    rng_overfit = np.random.default_rng(27)
    overfit_matrix = rng_overfit.normal(0.0, 0.01, size=(n_periods, 10))

    pbo_genuine = pbo_cscv(genuine_matrix, n_splits=n_splits)
    pbo_overfit = pbo_cscv(overfit_matrix, n_splits=n_splits)

    assert pbo_overfit > pbo_genuine


def test_explain_reports_purge_and_embargo_counts() -> None:
    # embargo_frac=0.02 on n_rows=500 produced width = ceil(0.02 * 500) = 10 rows;
    # split() is called without day_offsets, so embargo_sessions maps 1:1 onto
    # rows (per cv._embargo_width_rows).
    folds = PurgedKFold(n_splits=5, label_horizon=5, embargo_sessions=10).split(500)
    fold = next(f for f in folds if f.n_purged > 0 and f.n_embargoed > 0)

    explanation = fold.explain()
    assert isinstance(explanation, str)
    assert str(fold.n_purged) in explanation
    assert str(fold.n_embargoed) in explanation


def test_identical_inputs_give_identical_folds() -> None:
    # embargo_frac=0.02 on n_rows=500 (the split() calls below) produced width =
    # ceil(0.02 * 500) = 10 rows; split() is called without day_offsets, so
    # embargo_sessions maps 1:1 onto rows (per cv._embargo_width_rows).
    kf = PurgedKFold(n_splits=5, label_horizon=10, embargo_sessions=10)
    folds1 = kf.split(500)
    folds2 = kf.split(500)

    assert len(folds1) == len(folds2)
    for left, right in zip(folds1, folds2):
        assert left.id == right.id
        assert np.array_equal(left.train_idx, right.train_idx)
        assert np.array_equal(left.test_idx, right.test_idx)
        assert left.n_purged == right.n_purged
        assert left.n_embargoed == right.n_embargoed


def test_day_offsets_can_be_incomplete() -> None:
    """Line 192: day_offsets[-1] != n_rows triggers append."""
    day_offsets = np.array([0, 100, 200])  # Doesn't end at n_rows
    kf = PurgedKFold(n_splits=2)
    folds = kf.split(300, day_offsets=day_offsets)
    # If line 192 executed, folds should partition 300 rows into 2 blocks
    assert len(folds) == 2
    assert folds[0].test_idx[-1] + 1 + len(folds[1].test_idx) == 300


def test_fewer_days_than_splits_uses_fallback() -> None:
    """Lines 199-204: n_days < n_splits triggers fallback (each day is a block)."""
    day_offsets = np.array([0, 100])  # 2 days
    kf = PurgedKFold(n_splits=5)  # 5 splits > 2 days
    folds = kf.split(200, day_offsets=day_offsets)
    # Fallback: each day becomes its own test block, so 2 folds not 5
    assert len(folds) == 2
    assert folds[0].test_idx[-1] - folds[0].test_idx[0] == 99  # Day 0: 100 rows
    assert folds[1].test_idx[-1] - folds[1].test_idx[0] == 99  # Day 1: 100 rows


def test_split_rejects_invalid_n_test_groups() -> None:
    """Line 288: split() also validates n_test_groups >= n_groups, not just n_paths()."""
    cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=3)
    with pytest.raises(ValueError, match="n_test_groups must be < n_groups"):
        cv.split(300)


def test_paths_handles_empty_fold_results() -> None:
    """Line 383: paths() handles empty fold_results."""
    cv = CombinatorialPurgedCV(n_groups=2, n_test_groups=1)
    result = cv.paths([])
    assert result.shape == (0, 0)


def test_paths_rejects_mismatched_total_length() -> None:
    """Line 401: fold_results total length must divide evenly by appearances."""
    cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=1)
    # Create fold_results where total_len % appearances != 0
    # C(2, 0) = 1, so appearances_per_row = 1
    # If all 3 folds have length 100, total_len = 300
    # n_rows = 300 / 1 = 300, folds = split(300) creates 3 folds of 100 each
    # This is valid. To make it invalid, create a list with odd structure.
    # Actually, the modulo check is: total_len % C(n_groups-1, n_test_groups-1) != 0
    # With n_groups=3, n_test_groups=1: C(2,0)=1, so total_len % 1 == 0 always
    # With n_groups=4, n_test_groups=2: C(3,1)=3, so total_len % 3 must == 0
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2)
    # Each fold tests 2 of 4 groups, so each fold tests 200 rows out of 400
    # There are C(4,2)=6 folds, so total_len = 6*200 = 1200
    # appearances_per_row = C(3,1) = 3
    # n_rows = 1200 / 3 = 400 ✓
    # To trigger line 401, create fold_results with total_len % 3 != 0
    # Example: 5 arrays of 200 each: total_len = 1000, 1000 % 3 = 1
    fold_results = [np.random.normal(0, 0.01, size=200) for _ in range(5)]
    with pytest.raises(ValueError, match="fold result lengths incompatible"):
        cv.paths(fold_results)


def test_paths_rejects_fold_count_mismatch() -> None:
    """Line 407: number of fold_results must match number of folds."""
    cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=1)
    # C(3, 1) = 3 folds, each tests 1 of 3 groups (n_rows=300, each group 100)
    # Create fold_results with wrong count (4 instead of 3)
    fold_results = [np.random.normal(0, 0.01, size=100) for _ in range(4)]
    # First check: consistent length ✓
    # n_rows = 400 / 1 = 400, but split(400) expects 400 rows / 3 groups
    # Actually the issue is different. Let me reconsider.
    # If we pass 4 arrays of 100 each, total_len=400, appearances=1, n_rows=400
    # split(400) creates 3 folds. So len(folds)=3 != 4
    with pytest.raises(ValueError, match="number of fold results"):
        cv.paths(fold_results)


def test_paths_verifies_fold_result_sizes_line_413() -> None:
    """Line 413-414: the loop verifies each fold_result matches its fold.test_idx.

    The algorithm is self-consistent: with valid fold_results, sizes always match.
    This test verifies that valid results don't trigger the error at 414.
    A true gap (mismatched sizes) would be caught here, but such a gap would
    require inputs where the algorithm's self-consistency breaks, which cannot
    happen with valid inputs to split().
    """
    cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=2)
    folds = cv.split(150)
    # Create fold_results from the actual folds
    fold_results = [np.random.normal(0, 0.01, size=len(f.test_idx)) for f in folds]
    # This should succeed because sizes match
    result = cv.paths(fold_results)
    # Verify the result shape
    assert result.shape == (len(fold_results[0]), len(folds))
