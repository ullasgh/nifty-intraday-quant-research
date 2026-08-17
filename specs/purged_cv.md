# Spec: `nifty_quant.research.cv` — purged K-fold and CPCV

**Status:** contract for TDD. Tests written from this document alone.

## Why this exists

`research/splits.py` has `WalkForwardSplitter` (rolling/anchored, with an embargo) and
`HoldoutLock`. It has **no purged K-fold and no CPCV**, so the repo cannot do the two things
Lopez de Prado's machinery requires:

1. **Purging.** A label at time `t` built from an `h`-bar forward return *overlaps* rows
   `t..t+h`. If any of those land in the test fold while `t` is in train, the model has seen the
   test period's outcome. Ordinary K-fold on time series leaks this way silently.
2. **CPCV** (Combinatorial Purged CV). Standard walk-forward yields ONE backtest path, so
   `pbo_cscv` has nothing to compare. CPCV generates many paths by choosing `k` test groups from
   `N`, giving the distribution PBO needs.

Phase 3 tests five hypothesis families. Without these, multiple-testing control is unavailable
and every verdict is a single-path point estimate.

## Public API

```python
@dataclass(frozen=True)
class Fold:
    id: int
    train_idx: np.ndarray      # int64, sorted, unique
    test_idx: np.ndarray       # int64, sorted, unique
    n_purged: int              # train observations dropped for label overlap
    n_embargoed: int           # train observations dropped by the embargo
    def explain(self) -> str: ...


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 5
    label_horizon: int = 0     # bars a label looks forward; drives purging
    embargo_frac: float = 0.01 # fraction of total rows embargoed AFTER each test block
    def __post_init__(self) -> None:
        """ValueError on n_splits < 2, label_horizon < 0, embargo_frac < 0 or >= 1."""
    def split(self, n_rows: int, *, day_offsets: np.ndarray | None = None) -> list[Fold]:
        """Contiguous test blocks in time order (NEVER shuffled -- shuffling destroys the
        temporal structure purging exists to protect).

        Purge: drop any train index `i` where [i, i + label_horizon] intersects the test block.
        Embargo: additionally drop `ceil(embargo_frac * n_rows)` train indices immediately
        AFTER the test block (not before -- the leak direction is train-follows-test).

        If `day_offsets` is given, test-block boundaries snap to session boundaries so a fold
        never splits a trading day.
        """


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    n_groups: int = 6
    n_test_groups: int = 2
    label_horizon: int = 0
    embargo_frac: float = 0.01
    def n_paths(self) -> int:
        """Number of distinct backtest paths = C(n_groups, n_test_groups) * n_test_groups
        / n_groups. Document the derivation in the docstring."""
    def split(self, n_rows, *, day_offsets=None) -> list[Fold]:
        """One Fold per combination of `n_test_groups` chosen from `n_groups` contiguous
        groups. Same purge + embargo rules. Deterministic ordering (combinations in
        lexicographic order) so runs are reproducible."""
    def paths(self, fold_results: Sequence[np.ndarray]) -> np.ndarray:
        """Assemble per-fold OOS return series into a (n_periods, n_paths) matrix suitable for
        `backtest.metrics.pbo_cscv`. Must contain NO NaN -- pbo_cscv cannot accept them.
        Document exactly how gaps are handled and raise rather than silently filling."""
```

## Hard rules

- **Never shuffle.** Test blocks are contiguous and in time order.
- Purge is symmetric in intent but the embargo is **one-sided (after the test block)**; state why.
- `label_horizon=0` means no purging — assert that reduces to plain contiguous K-fold.
- Train and test indices must be **disjoint** in every fold, always.
- Reuse `WalkForwardSplitter`'s trading-day index arithmetic where applicable; do not invent a
  second convention. `TRADING_DAYS_PER_YEAR = 252` already exists.
- Deterministic: identical inputs give identical folds. No RNG.
- Vectorized index maths; a loop over folds is fine, a loop over rows is not.

## Required tests (`tests/test_purged_cv.py`)

**PurgedKFold — the leak guarantees:**
1. `test_train_and_test_are_always_disjoint` — parametrized over n_splits
2. `test_no_train_label_window_overlaps_test_block` — the core purge property: for every train
   index `i`, `[i, i+label_horizon]` must not intersect the test block
3. `test_label_horizon_zero_reduces_to_contiguous_kfold`
4. `test_larger_label_horizon_purges_strictly_more`
5. `test_embargo_applies_after_test_block_not_before`
6. `test_embargo_frac_zero_purges_nothing_extra`
7. `test_n_purged_and_n_embargoed_are_reported_accurately`
8. `test_test_blocks_are_contiguous_and_in_time_order`
9. `test_test_blocks_cover_all_rows_exactly_once`
10. `test_folds_snap_to_session_boundaries_when_day_offsets_given`
11. `test_irregular_sessions_do_not_break_snapping` — 375-bar then 60-bar Muhurat day
12. `test_never_shuffles` — folds are reproducible and ordered

**Validation:**
13-16. `test_rejects_n_splits_below_two`, `test_rejects_negative_label_horizon`,
`test_rejects_embargo_frac_out_of_range`, `test_rejects_n_rows_too_small_for_n_splits`

**CPCV:**
17. `test_n_paths_matches_combinatorial_formula` — parametrized over (n_groups, n_test_groups)
18. `test_number_of_folds_equals_n_choose_k`
19. `test_every_group_appears_in_test_the_same_number_of_times` — the symmetry property
20. `test_combinations_are_lexicographic_and_deterministic`
21. `test_cpcv_folds_also_purged_and_embargoed`
22. `test_rejects_n_test_groups_ge_n_groups`
23. `test_paths_matrix_shape_is_periods_by_n_paths`
24. `test_paths_matrix_contains_no_nan`
25. `test_paths_raises_on_gaps_rather_than_filling`

**Integration — the reason this module exists:**
26. `test_pbo_cscv_accepts_a_cpcv_paths_matrix` — build folds, synthesise per-fold OOS returns,
    assemble via `paths()`, feed to `backtest.metrics.pbo_cscv`, assert finite and in [0, 1].
27. `test_pbo_is_high_for_overfit_trials_and_low_for_genuine` — construct a trial set where the
    best in-sample performer is random (expect high PBO) and one where it is genuinely best
    (expect lower PBO). This is the test that proves the machinery discriminates rather than
    merely runs.
28. `test_explain_reports_purge_and_embargo_counts`

**Determinism:**
29. `test_identical_inputs_give_identical_folds`
