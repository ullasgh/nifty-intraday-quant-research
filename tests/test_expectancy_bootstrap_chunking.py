"""Regression tests for the memory-bounded block-bootstrap chunking fix.

The `block_bootstrap` branch of `_compute_bucket_stats` used to materialise a full
`(n_boot, n_rows, n_symbols)` array via `_block_bootstrap_resampling_2d` before
reducing it to one mean per replicate. On the real panel (701_863 rows x 149
symbols, n_boot=1000) that is ~0.8 TB and gets SIGKILLed. The fix processes
replicates in memory-bounded chunks via `_block_bootstrap_means_chunked` /
`_default_bootstrap_chunk_size`, and MUST produce bit-identical `se_bps` (and
`boot_means`) for a given seed regardless of `chunk_size` -- chunking only changes
how many already-decided replicates are held in memory at once, never what is
drawn or in what order.

Uses two sessions of UNEQUAL length (per CLAUDE.md rule 5: never assume a fixed
per-day row-count stride).
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.research.expectancy import (
    _block_bootstrap_means_chunked,
    _block_bootstrap_resampling_2d,
    _compute_bucket_stats,
    _default_bootstrap_chunk_size,
)


def _fixture(seed_for_data: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two unequal-length sessions, some NaN gaps, some symbols entirely NaN."""
    rng = np.random.default_rng(seed_for_data)
    day_offsets = np.array([0, 130, 400])  # session lengths 130 and 270 -- unequal
    n_rows, n_symbols = 400, 6
    values = rng.normal(0, 0.001, size=(n_rows, n_symbols))
    mask = rng.random((n_rows, n_symbols)) < 0.6
    values = np.where(mask, values, np.nan)
    values[:, -1] = np.nan  # one symbol entirely absent
    return values, day_offsets


class TestDefaultChunkSizeIsMemoryBounded:
    def test_real_panel_shape_resolves_to_chunk_size_one(self) -> None:
        # 701_863 rows x 149 symbols: one replicate alone is ~0.78 GB, already over
        # the 256 MB target, so chunk_size must floor at 1 (never 0, never negative).
        chunk = _default_bootstrap_chunk_size(701_863, 149)
        assert chunk == 1

    def test_small_panel_allows_multiple_replicates_per_chunk(self) -> None:
        chunk = _default_bootstrap_chunk_size(400, 6)
        bytes_per_replicate = 400 * 6 * 8
        assert chunk >= 1
        assert chunk * bytes_per_replicate <= 256 * 1024 * 1024

    def test_chunk_size_never_zero_or_negative(self) -> None:
        for n_rows, n_symbols in [(1, 1), (10**9, 10**9), (701_863, 149)]:
            assert _default_bootstrap_chunk_size(n_rows, n_symbols) >= 1


class TestChunkedMatchesFullArrayReference:
    """The chunked path must reproduce the full-array path's boot means exactly."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 123])
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 37, 1000])
    def test_boot_means_identical_to_full_array_reduction(
        self, seed: int, chunk_size: int
    ) -> None:
        values, day_offsets = _fixture()
        horizon = 5
        n_boot = 40

        # Reference: the OLD, unchunked path -- full array, then reduce.
        resampled, ref_block_indices, ref_skipped = _block_bootstrap_resampling_2d(
            values, day_offsets, horizon, n_boot, seed
        )
        resampled_flat = resampled.reshape((n_boot, -1))
        valid_rows = np.isfinite(resampled_flat).any(axis=1)
        ref_boot_means = np.full(n_boot, np.nan, dtype=np.float64)
        if np.any(valid_rows):
            ref_boot_means[valid_rows] = np.nanmean(resampled_flat[valid_rows], axis=1)

        # New: chunked path.
        boot_means, block_indices, n_skipped = _block_bootstrap_means_chunked(
            values, day_offsets, horizon, n_boot, seed, chunk_size=chunk_size
        )

        np.testing.assert_array_equal(boot_means, ref_boot_means)
        assert block_indices == ref_block_indices
        assert n_skipped == ref_skipped

    def test_different_chunk_sizes_agree_with_each_other(self) -> None:
        values, day_offsets = _fixture()
        horizon = 3
        n_boot = 25
        seed = 99

        results = [
            _block_bootstrap_means_chunked(
                values, day_offsets, horizon, n_boot, seed, chunk_size=cs
            )
            for cs in (1, 4, 25, 1000)
        ]
        reference_means = results[0][0]
        reference_blocks = results[0][1]
        for boot_means, block_indices, _skipped in results[1:]:
            np.testing.assert_array_equal(boot_means, reference_means)
            assert block_indices == reference_blocks


class TestComputeBucketStatsSeBpsUnchangedByChunking:
    """End-to-end: `_compute_bucket_stats`'s se_bps must not depend on chunk_size."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 123])
    def test_se_bps_identical_across_chunk_sizes(self, seed: int) -> None:
        values, day_offsets = _fixture()
        flat = values.flatten()
        horizon = 5
        n_boot = 40

        stat_default = _compute_bucket_stats(
            flat, horizon, day_offsets, se_method="block_bootstrap", n_boot=n_boot, seed=seed
        )
        assert stat_default is not None

        for chunk_size in (1, 2, 5, 40, 1000):
            stat = _compute_bucket_stats(
                flat,
                horizon,
                day_offsets,
                se_method="block_bootstrap",
                n_boot=n_boot,
                seed=seed,
                chunk_size=chunk_size,
            )
            assert stat is not None
            assert stat.se_bps == stat_default.se_bps
            assert stat.mean_bps == stat_default.mean_bps
            assert stat.t_stat == stat_default.t_stat

    def test_se_bps_matches_recorded_row_weighted_baseline(self) -> None:
        """Baseline `se_bps` for the ROW-WEIGHTED estimator on this exact fixture,
        seeds 0/1/7/123, n_boot=200, horizon=5.

        RE-BASELINED 2026-08-21 (specs/overlap_se.md AMENDMENT 9). The previous values
        pinned the CELL-weighted estimator, which averaged over stock-days: a bar with 30
        eligible names carried ten times the weight of a bar with 3. The bootstrap now
        reduces to per-row bucket MEANS first, so it averages over BARS -- which is what a
        strategy actually earns, since a bar is one decision and one P&L event however many
        names happened to qualify.

        On this deliberately patchy fixture (6 symbols, 60% coverage) the two estimators
        differ by 7.3%-14.7%; on realistic geometry (149 symbols, 95% coverage) they differ
        by 0.06%-0.25%. The gap is heterogeneous per-row coverage, not a defect in either.

        The test's PURPOSE is unchanged and is why it was not deleted: an unpinned SE is how
        the L7 and L8 defects survived for the life of this repo. It now pins the estimator
        we chose rather than the one we replaced.
        """
        rng = np.random.default_rng(42)
        n_rows, n_symbols = 400, 6
        day_offsets = np.array([0, 100, 200, 300, 400])
        horizon = 5
        bucket_returns = rng.normal(0, 0.001, size=(n_rows, n_symbols))
        mask = rng.random((n_rows, n_symbols)) < 0.6
        bucket_returns = np.where(mask, bucket_returns, np.nan)
        flat = bucket_returns.flatten()

        expected = {
            0: 0.3272566677326099,
            1: 0.32371609850266153,
            7: 0.3230864171117964,
            123: 0.2984809699964152,
        }
        for seed, expected_se_bps in expected.items():
            stat = _compute_bucket_stats(
                flat,
                horizon,
                day_offsets,
                se_method="block_bootstrap",
                n_boot=200,
                seed=seed,
            )
            assert stat is not None
            assert stat.se_bps == pytest.approx(expected_se_bps, rel=0, abs=1e-12)
