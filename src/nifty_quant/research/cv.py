"""Purged K-fold and Combinatorial Purged Cross-Validation for time-series data.

This module implements Lopez de Prado's purging and embedding techniques to prevent
forward-looking bias in time-series backtests. A label at time `t` built from an `h`-bar
forward return overlaps rows `t..t+h`. If any of those land in the test fold while `t` is
in train, the model has already seen the test period's outcome, silently. This module
removes that leakage via purging and provides CPCV for multiple-testing control.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nifty_quant.research.embargo import required_embargo_sessions


def _embargo_width_rows(
    start_row: int,
    n_rows: int,
    embargo_sessions: int,
    day_offsets: np.ndarray | None,
) -> int:
    """Convert an absolute session count into a row-width, starting at `start_row`
    (which must itself be a session boundary when `day_offsets` is given) and
    walking FORWARD through the next `embargo_sessions` sessions. Without
    `day_offsets`, sessions and rows are treated 1:1. This is an ABSOLUTE size --
    it must never scale with `n_rows`."""
    if embargo_sessions <= 0:
        return 0
    if day_offsets is None:
        return embargo_sessions
    day_offsets = np.asarray(day_offsets, dtype=np.int64)
    if day_offsets[-1] != n_rows:
        day_offsets = np.append(day_offsets, n_rows)
    idx = int(np.searchsorted(day_offsets, start_row))
    end_idx = min(idx + embargo_sessions, len(day_offsets) - 1)
    return int(day_offsets[end_idx] - start_row)


def _resolve_embargo_sessions(
    *,
    configured_label_horizon: int,
    configured_embargo_sessions: int,
    feature_lookback: float,
    label_horizon: float,
    holding_period: float,
    execution_horizon: float,
) -> int:
    """Resolve the RIGHT-side embargo width, in sessions, shared by `PurgedKFold`
    and `CombinatorialPurgedCV`.

    Two decisions, both from `specs/embargo_sizing.md` Amendment 2 clause 6, made
    explicit here so neither is left ambiguous:

    1. `configured_embargo_sessions` (the constructor's `embargo_sessions` field)
       is an explicit override. When > 0 it wins outright over the derived value
       -- a caller who names a number means it. When 0 (the default), the width
       is derived from the four dependence-horizon components via
       `required_embargo_sessions`, the SAME helper `WalkForwardSplitter.split`
       uses, so the two mechanisms size the embargo identically for identical
       components.
    2. `label_horizon` here (a `split()`-time float, in sessions) is a second
       source for the same concept `configured_label_horizon` (the constructor's
       `label_horizon`, in bars) already holds, and is used ONLY for the LEFT
       purge -- never for this embargo-sum. To avoid two live sources for one
       value: if the `split()`-time float is left at its default (0.0), the
       constructor's value is reused (cast to float) as the embargo-sum's
       label_horizon term. If the `split()`-time float is passed non-zero, it
       overrides the constructor's value for the embargo-sum ONLY; the
       constructor field alone still, and always, drives the left purge.
    """
    if configured_embargo_sessions > 0:
        return configured_embargo_sessions
    effective_label_horizon = (
        label_horizon if label_horizon != 0.0 else float(configured_label_horizon)
    )
    return required_embargo_sessions(
        feature_lookback, effective_label_horizon, holding_period, execution_horizon
    )


@dataclass(frozen=True)
class Fold:
    """A single train/test split with purge and embargo tracking."""

    id: int
    train_idx: np.ndarray  # int64, sorted, unique
    test_idx: np.ndarray  # int64, sorted, unique
    n_purged: int  # train observations dropped for label overlap
    n_embargoed: int  # train observations dropped by the embargo

    def explain(self) -> str:
        """Return a human-readable explanation of the fold's purge and embargo."""
        return (
            f"Fold {self.id}: purged {self.n_purged} rows, embargoed {self.n_embargoed} rows. "
            f"Train size: {len(self.train_idx)}, Test size: {len(self.test_idx)}"
        )


@dataclass(frozen=True)
class PurgedKFold:
    """Contiguous K-fold with purging and one-sided embargo.

    Never shuffles. Test blocks are contiguous and in time order. Purging removes any
    train row `i` where [i, i + label_horizon] intersects the test block, because a
    label built from an h-bar forward return overlaps rows [t, t+h]. The embargo is
    one-sided (AFTER the test block) because the leak direction is train-follows-test.
    """

    n_splits: int = 5
    label_horizon: int = 0  # bars a label looks forward; drives purging (LEFT side only)
    embargo_sessions: int = 0  # explicit override for the RIGHT-side embargo width, in
    # sessions. When > 0 it wins over the width derived from split()'s four
    # dependence-horizon components (a caller who names a number means it). When left
    # at the default 0, the width is derived -- see split().

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")
        if self.embargo_sessions < 0:
            raise ValueError("embargo_sessions must be >= 0")

    def split(
        self,
        n_rows: int,
        *,
        day_offsets: np.ndarray | None = None,
        feature_lookback: float = 0.0,
        label_horizon: float = 0.0,
        holding_period: float = 0.0,
        execution_horizon: float = 0.0,
    ) -> list[Fold]:
        """Contiguous test blocks in time order, with purging and embargo.

        Parameters
        ----------
        n_rows : int
            Total number of rows to split.
        day_offsets : np.ndarray | None
            If provided, test-block boundaries snap to session boundaries (rows
            that start a new day/session). This ensures a fold never splits a
            trading day.
        feature_lookback, label_horizon, holding_period, execution_horizon : float
            The four dependence-horizon components, in SESSIONS, used to derive
            the RIGHT-side embargo width via the same
            `nifty_quant.research.embargo.required_embargo_sessions` helper the
            walk-forward splitter uses (spec `embargo_sizing.md` section A,
            Amendment 2 clause 6). They do NOT affect the LEFT purge -- that
            stays governed solely by the constructor's `label_horizon` (in bars),
            unchanged. Purging removes label overlap on the left; the embargo
            removes serial dependence on the right; the spec explicitly forbids
            unifying the two, so a fold's left and right widths may legitimately
            differ.

            `label_horizon` here is a SECOND source, in sessions, for the same
            concept the constructor's `label_horizon` field holds in bars.
            Resolution (documented so two sources for one value is never
            ambiguous): if this keyword is left at its default (0.0), the
            constructor's `label_horizon` is reused (cast to float) as the
            embargo-sum's label_horizon term, so a caller states the number only
            once. If this keyword is passed non-zero, it OVERRIDES the
            constructor's value for the embargo-sum contribution ONLY -- the
            constructor field alone still drives the left purge, always.

        Returns
        -------
        list[Fold]
            List of folds with disjoint test indices covering all rows exactly once.

        Raises
        ------
        ValueError
            If n_rows is too small for n_splits.
        """
        if n_rows < self.n_splits:
            raise ValueError(
                f"n_rows ({n_rows}) must be at least n_splits ({self.n_splits})"
            )

        embargo_sessions = _resolve_embargo_sessions(
            configured_label_horizon=self.label_horizon,
            configured_embargo_sessions=self.embargo_sessions,
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )

        folds: list[Fold] = []

        # If day_offsets provided, snap test blocks to session boundaries.
        if day_offsets is not None:
            day_offsets = np.asarray(day_offsets, dtype=np.int64)
            test_block_boundaries = self._snap_to_days(n_rows, day_offsets, self.n_splits)
        else:
            # Simple contiguous split: each block gets roughly n_rows // n_splits rows.
            test_block_boundaries = self._contiguous_split(n_rows, self.n_splits)

        for fold_id, (test_start, test_end) in enumerate(test_block_boundaries):
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            # All rows except test are candidates for training.
            candidate_train = np.concatenate(
                [
                    np.arange(0, test_start, dtype=np.int64),
                    np.arange(test_end, n_rows, dtype=np.int64),
                ]
            )

            # Purge: drop any train row i where [i, i + label_horizon] intersects test block.
            # [i, i + label_horizon] intersects [test_start, test_end) iff:
            #   i < test_end AND i + label_horizon >= test_start
            purge_start = test_start - self.label_horizon
            purge_end = test_end
            purge_mask = (candidate_train < purge_start) | (candidate_train >= purge_end)
            train_after_purge = candidate_train[purge_mask]
            n_purged = len(candidate_train) - len(train_after_purge)

            # Embargo: drop embargo_width rows immediately AFTER test block. Width is
            # an absolute session count converted to rows via day_offsets -- it must
            # never scale with n_rows.
            embargo_start = test_end
            embargo_width = _embargo_width_rows(
                embargo_start, n_rows, embargo_sessions, day_offsets
            )
            embargo_end = min(embargo_start + embargo_width, n_rows)
            embargo_mask = (train_after_purge < embargo_start) | (
                train_after_purge >= embargo_end
            )
            train_idx = train_after_purge[embargo_mask]
            n_embargoed = len(train_after_purge) - len(train_idx)

            folds.append(
                Fold(
                    id=fold_id,
                    train_idx=np.sort(train_idx),
                    test_idx=test_idx,
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )

        return folds

    @staticmethod
    def _contiguous_split(n_rows: int, n_splits: int) -> list[tuple[int, int]]:
        """Partition n_rows into n_splits contiguous test blocks.

        Each block covers test_end - test_start rows, with blocks ordered in time.

        Returns
        -------
        list[tuple[int, int]]
            List of (test_start, test_end) pairs. [test_start, test_end) is the test block.
        """
        block_size = n_rows // n_splits
        boundaries: list[tuple[int, int]] = []
        for i in range(n_splits):
            start = i * block_size
            if i == n_splits - 1:
                # Last block takes all remaining rows.
                end = n_rows
            else:
                end = (i + 1) * block_size
            boundaries.append((start, end))
        return boundaries

    @staticmethod
    def _snap_to_days(
        n_rows: int, day_offsets: np.ndarray, n_splits: int
    ) -> list[tuple[int, int]]:
        """Partition n_rows into n_splits test blocks snapped to session boundaries.

        day_offsets are the row indices where new sessions start (e.g., [0, 375, 435]).
        Test-block boundaries must align with these offsets.

        Parameters
        ----------
        n_rows : int
            Total number of rows.
        day_offsets : np.ndarray
            Session start row indices (must be sorted and include 0 and n_rows).
        n_splits : int
            Number of splits.

        Returns
        -------
        list[tuple[int, int]]
            List of (test_start, test_end) snapped to day_offsets.
        """
        day_offsets = np.asarray(day_offsets, dtype=np.int64)

        # Ensure day_offsets ends at n_rows.
        if day_offsets[-1] != n_rows:
            day_offsets = np.append(day_offsets, n_rows)

        # Number of "days" (sessions).
        n_days = len(day_offsets) - 1

        if n_days < n_splits:
            # Fewer days than splits: each day is its own test block.
            boundaries: list[tuple[int, int]] = []
            for i in range(n_days):
                start = int(day_offsets[i])
                end = int(day_offsets[i + 1])
                boundaries.append((start, end))
            return boundaries

        # Distribute days across splits.
        days_per_split = n_days // n_splits
        extra_days = n_days % n_splits

        boundaries = []
        day_idx = 0
        for split_idx in range(n_splits):
            # This split gets days_per_split days, plus one extra if split_idx < extra_days.
            n_days_in_split = days_per_split + (1 if split_idx < extra_days else 0)
            start_day = day_idx
            end_day = day_idx + n_days_in_split

            start_row = int(day_offsets[start_day])
            end_row = int(day_offsets[end_day])
            boundaries.append((start_row, end_row))

            day_idx = end_day

        return boundaries


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation for hypothesis testing.

    Generates C(n_groups, n_test_groups) distinct backtest paths by choosing
    n_test_groups contiguous groups as the test set and the rest as train.
    This provides the diversity pbo_cscv needs to discriminate overfitting.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    label_horizon: int = 0  # bars a label looks forward; drives purging (LEFT side only)
    embargo_sessions: int = 0  # explicit override for the RIGHT-side embargo width, in
    # sessions -- same override semantics as `PurgedKFold.embargo_sessions`; see
    # `_resolve_embargo_sessions`.

    def n_paths(self) -> int:
        """Number of distinct backtest paths.

        Derivation: each combination of n_test_groups chosen from n_groups
        appears in exactly C(n_groups-1, n_test_groups-1) folds (by the hockey-stick
        identity). Since there are C(n_groups, n_test_groups) such combinations,
        and each yields one path, the total is:

            C(n_groups, n_test_groups) * n_test_groups / n_groups

        This simplifies to the identity above when averaged per-group.

        Returns
        -------
        int
            Number of distinct test paths = C(n_groups, n_test_groups) * n_test_groups / n_groups.

        Raises
        ------
        ValueError
            If n_test_groups >= n_groups.
        """
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be < n_groups")

        n_combos = math.comb(self.n_groups, self.n_test_groups)
        return n_combos * self.n_test_groups // self.n_groups

    def split(
        self,
        n_rows: int,
        *,
        day_offsets: np.ndarray | None = None,
        feature_lookback: float = 0.0,
        label_horizon: float = 0.0,
        holding_period: float = 0.0,
        execution_horizon: float = 0.0,
    ) -> list[Fold]:
        """Combinatorial test-group selection with purge and embargo.

        Each fold tests one contiguous group (or group combination) and trains on the rest.
        Purging and embargo apply to each fold independently.

        Parameters
        ----------
        n_rows : int
            Total number of rows.
        day_offsets : np.ndarray | None
            If provided, group boundaries snap to session boundaries.
        feature_lookback, label_horizon, holding_period, execution_horizon : float
            Same meaning, units and constructor-vs-split-time `label_horizon`
            resolution as `PurgedKFold.split` (see its docstring and
            `_resolve_embargo_sessions`): these size the RIGHT-side embargo only,
            never the LEFT purge, which stays governed by the constructor's
            `label_horizon` (bars).

        Returns
        -------
        list[Fold]
            Folds for all C(n_groups, n_test_groups) combinations, in lexicographic order.
        """
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be < n_groups")

        embargo_sessions = _resolve_embargo_sessions(
            configured_label_horizon=self.label_horizon,
            configured_embargo_sessions=self.embargo_sessions,
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )

        # Use PurgedKFold to partition into n_groups groups.
        kf = PurgedKFold(
            n_splits=self.n_groups,
            label_horizon=self.label_horizon,
            embargo_sessions=self.embargo_sessions,
        )
        base_folds = kf.split(n_rows, day_offsets=day_offsets)

        # Extract group boundaries from base_folds.
        group_ranges = [(f.test_idx[0], f.test_idx[-1] + 1) for f in base_folds]

        folds: list[Fold] = []
        fold_id = 0

        # Iterate over all combinations of n_test_groups groups (lexicographic).
        for test_groups in itertools.combinations(range(self.n_groups), self.n_test_groups):
            # Collect all test rows for this combination.
            test_idx_list = []
            for group_idx in test_groups:
                start, end = group_ranges[group_idx]
                test_idx_list.append(np.arange(start, end, dtype=np.int64))
            test_idx = np.concatenate(test_idx_list)

            # All rows except test are candidates for training.
            test_set = set(test_idx.tolist())
            candidate_train = np.array(
                [i for i in range(n_rows) if i not in test_set], dtype=np.int64
            )

            # Purge: drop any train row i where [i, i + label_horizon] intersects test block.
            purge_mask = np.ones(len(candidate_train), dtype=bool)
            for i_idx, i in enumerate(candidate_train):
                # Check if [i, i + label_horizon] intersects test_idx.
                label_end = i + self.label_horizon
                if i <= test_idx[-1] and label_end >= test_idx[0]:
                    # More precise: check if any test row falls in [i, label_end].
                    if np.any((test_idx >= i) & (test_idx <= label_end)):
                        purge_mask[i_idx] = False

            train_after_purge = candidate_train[purge_mask]
            n_purged = len(candidate_train) - len(train_after_purge)

            # Embargo: drop embargo_sessions worth of rows immediately AFTER test
            # block. Width is an absolute session count converted to rows via
            # day_offsets -- it must never scale with n_rows.
            embargo_start = int(test_idx[-1]) + 1
            embargo_width = _embargo_width_rows(
                embargo_start, n_rows, embargo_sessions, day_offsets
            )
            embargo_end = min(embargo_start + embargo_width, n_rows)
            embargo_mask = (train_after_purge < embargo_start) | (
                train_after_purge >= embargo_end
            )
            train_idx = train_after_purge[embargo_mask]
            n_embargoed = len(train_after_purge) - len(train_idx)

            folds.append(
                Fold(
                    id=fold_id,
                    train_idx=np.sort(train_idx),
                    test_idx=np.sort(test_idx),
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )
            fold_id += 1

        return folds

    def paths(self, fold_results: Sequence[np.ndarray]) -> np.ndarray:
        """Assemble per-fold OOS return series into a (n_periods, n_paths) matrix.

        Each element of fold_results is a 1-D array of per-fold out-of-sample returns.
        The fold order matches the output of self.split() (lexicographic combinations).
        All fold_results must have the same length (no gaps) and must not contain NaN.

        Parameters
        ----------
        fold_results : Sequence[np.ndarray]
            Per-fold OOS return arrays. Must contain NO NaN. All arrays must have
            the same length. Length must match the number of folds from self.split().

        Returns
        -------
        np.ndarray
            Shape (fold_size, n_folds), where n_folds = C(n_groups, n_test_groups).
            Each column is one fold's OOS returns. No NaN values.

        Raises
        ------
        ValueError
            If fold_results contain NaN, if fold sizes are inconsistent (gap),
            or if the fold result lengths don't match the test set sizes.
        """
        fold_results_list = [np.asarray(r, dtype=np.float64) for r in fold_results]

        if not fold_results_list:
            return np.empty((0, 0), dtype=np.float64)

        # Check for NaN
        for i, r in enumerate(fold_results_list):
            if np.any(np.isnan(r)):
                raise ValueError(f"fold_results[{i}] contains NaN")

        # Check consistent length
        lengths = [len(r) for r in fold_results_list]
        if len(set(lengths)) > 1:
            raise ValueError("fold_results have inconsistent lengths (gap detected)")

        # Infer n_rows from the fold results
        total_len = sum(lengths)
        appearances_per_row = math.comb(
            self.n_groups - 1, self.n_test_groups - 1
        )
        if total_len % appearances_per_row != 0:
            raise ValueError("fold result lengths incompatible with CPCV parameters")
        n_rows = total_len // appearances_per_row

        # Verify folds match
        folds = self.split(n_rows)
        if len(folds) != len(fold_results_list):
            raise ValueError(
                f"number of fold results ({len(fold_results_list)}) does not match "
                f"number of folds ({len(folds)})"
            )

        for i, (fold, result) in enumerate(zip(folds, fold_results_list)):
            # Unreachable, and NOT for the combinatorial-identity reason first written here.
            # The operative invariant is that `n_rows` is DERIVED from these very results
            # (`total_len // appearances_per_row`), and an earlier check has already rejected
            # any input whose per-fold lengths differ from one another. So by the time we get
            # here every result has the same length L, `n_rows` is an exact multiple of it,
            # and `split(n_rows)` therefore partitions into groups of exactly L -- the
            # comparison below cannot differ.
            #
            # Genuine gaps ARE still caught: a ragged input (e.g. lengths 100, 101, 100, 100)
            # raises "fold_results have inconsistent lengths (gap detected)" at the earlier
            # check. Verified by direct call, not assumed. This arm is retained only so the
            # guarantee stays asserted if `n_rows` ever stops being derived from the results.
            if len(result) != len(fold.test_idx):  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    f"fold {i} result length ({len(result)}) does not match "
                    f"test_idx length ({len(fold.test_idx)}) — gap detected"
                )

        # Stack fold results as columns
        return np.column_stack(fold_results_list)
