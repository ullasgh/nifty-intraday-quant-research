"""Regression tests for the row-mean block-bootstrap fix.

`_compute_bucket_stats`'s `block_bootstrap` branch used to resample the FULL 2-D
`(n_rows, n_symbols)` bucket panel per replicate before reducing each replicate to
one number via a cell-level `nanmean`. That is `n_symbols` times more memory/work
than necessary: the block bootstrap exists to respect dependence ALONG ROWS, not
across symbols within a row, so the fix reduces to a length-`n_rows` per-row
bucket-mean series ONCE (`_bucket_row_means`) and resamples THAT instead
(`research/ic.py`'s `_overlap_aware_se` already does the analogous thing for the IC
series).

These tests are new (this file did not exist before the fix) and are written against
the NEW behaviour; they are not modifications of any pre-existing test file.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.research import expectancy
from nifty_quant.research.expectancy import (
    _block_bootstrap_means_chunked,
    _block_bootstrap_resampling_2d,
    _bucket_row_means,
    _compute_bucket_stats,
)


class TestBucketRowMeans:
    def test_row_with_no_valid_symbol_is_nan(self) -> None:
        values = np.array(
            [
                [1.0, np.nan, 3.0],
                [np.nan, np.nan, np.nan],
                [2.0, 4.0, np.nan],
            ]
        )
        row_means = _bucket_row_means(values)
        assert row_means[0] == pytest.approx(2.0)  # mean(1.0, 3.0)
        assert np.isnan(row_means[1])
        assert row_means[2] == pytest.approx(3.0)  # mean(2.0, 4.0)

    def test_matches_manual_nanmean_per_row(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(size=(50, 8))
        mask = rng.random((50, 8)) < 0.7
        values = np.where(mask, values, np.nan)
        row_means = _bucket_row_means(values)
        for t in range(50):
            row = values[t]
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                assert np.isnan(row_means[t])
            else:
                assert row_means[t] == pytest.approx(np.mean(finite))


class TestComputeBucketStatsResamplesRowMeanSeries:
    """`_compute_bucket_stats` must bootstrap the per-row bucket-MEAN series, not the
    raw 2-D panel. On a fixture with heterogeneous per-row valid-symbol counts these
    two give MATERIALLY different se_bps (row-weighting vs cell-weighting), so this
    directly discriminates the new code path from the pre-fix one.
    """

    def _fixture(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n_rows, n_symbols = 400, 6
        day_offsets = np.array([0, 100, 200, 300, 400])
        values = rng.normal(0, 0.001, size=(n_rows, n_symbols))
        mask = rng.random((n_rows, n_symbols)) < 0.6
        values = np.where(mask, values, np.nan)
        return values, day_offsets

    @pytest.mark.parametrize("seed", [0, 1, 7, 123])
    def test_se_bps_equals_row_mean_bootstrap_not_cell_bootstrap(self, seed: int) -> None:
        values, day_offsets = self._fixture()
        horizon = 5
        n_boot = 200
        flat = values.flatten()

        stat = _compute_bucket_stats(
            flat, horizon, day_offsets, se_method="block_bootstrap", n_boot=n_boot, seed=seed
        )
        assert stat is not None

        # Expected: bootstrap the REDUCED per-row bucket-mean series.
        n_rows = values.shape[0]
        row_means = _bucket_row_means(values).reshape((n_rows, 1))
        expected_boot_means, _, _ = _block_bootstrap_means_chunked(
            row_means, day_offsets, horizon, n_boot, seed
        )
        expected_valid = expected_boot_means[np.isfinite(expected_boot_means)]
        expected_se_bps = float(np.std(expected_valid * 1e4, ddof=1))
        assert stat.se_bps == pytest.approx(expected_se_bps, rel=0, abs=1e-12)

        # Contrast: bootstrapping the RAW 2-D panel (the pre-fix behaviour) gives a
        # materially different se_bps on this fixture (heterogeneous per-row
        # coverage) -- confirming the two code paths are not interchangeable and
        # that `stat.se_bps` above tracks the row-mean path, not this one.
        cell_boot_means, _, _ = _block_bootstrap_means_chunked(
            values, day_offsets, horizon, n_boot, seed
        )
        cell_valid = cell_boot_means[np.isfinite(cell_boot_means)]
        cell_se_bps = float(np.std(cell_valid * 1e4, ddof=1))
        assert stat.se_bps != pytest.approx(cell_se_bps, rel=0, abs=1e-9)


class TestBlockIndicesDiagnosticCap:
    """`block_indices` is a diagnostic; retention must be capped so it cannot grow
    unbounded across all replicates on a production-sized panel, without changing
    which blocks are drawn or how the resampled series/means are computed.
    """

    def test_cap_bounds_retained_block_indices_2d(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(expectancy, "_BLOCK_INDICES_MAX_RETAINED", 5)
        rng = np.random.default_rng(3)
        n_rows, n_symbols = 200, 2
        day_offsets = np.array([0, 200])
        values = rng.normal(size=(n_rows, n_symbols))
        horizon = 2
        n_boot = 50  # draws far more than 5 blocks total across all replicates

        resampled, block_indices, n_skipped = _block_bootstrap_resampling_2d(
            values, day_offsets, horizon, n_boot, seed=1
        )
        assert n_skipped == 0
        assert resampled.shape == (n_boot, n_rows, n_symbols)
        assert len(block_indices) <= 5

    def test_cap_bounds_retained_block_indices_chunked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(expectancy, "_BLOCK_INDICES_MAX_RETAINED", 5)
        rng = np.random.default_rng(3)
        n_rows, n_symbols = 200, 2
        day_offsets = np.array([0, 200])
        values = rng.normal(size=(n_rows, n_symbols))
        horizon = 2
        n_boot = 50

        boot_means, block_indices, n_skipped = _block_bootstrap_means_chunked(
            values, day_offsets, horizon, n_boot, seed=1
        )
        assert n_skipped == 0
        assert boot_means.shape == (n_boot,)
        assert len(block_indices) <= 5

    def test_cap_does_not_change_which_blocks_are_drawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Capping retention must not change the actual resampled values: the same
        seed must give bit-identical `boot_means`/`resampled` regardless of the cap.
        """
        rng = np.random.default_rng(3)
        n_rows, n_symbols = 200, 2
        day_offsets = np.array([0, 200])
        values = rng.normal(size=(n_rows, n_symbols))
        horizon = 2
        n_boot = 50

        resampled_full, _, _ = _block_bootstrap_resampling_2d(
            values, day_offsets, horizon, n_boot, seed=1
        )
        monkeypatch.setattr(expectancy, "_BLOCK_INDICES_MAX_RETAINED", 1)
        resampled_capped, block_indices_capped, _ = _block_bootstrap_resampling_2d(
            values, day_offsets, horizon, n_boot, seed=1
        )
        np.testing.assert_array_equal(resampled_full, resampled_capped)
        assert len(block_indices_capped) <= 1
