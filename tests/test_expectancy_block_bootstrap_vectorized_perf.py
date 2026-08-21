"""Performance regression test for the vectorized block-bootstrap draw in
`_block_bootstrap_means_chunked`.

Context: the pre-vectorization implementation drew block start positions and gathered
each block with a per-replicate, per-block Python loop (~70M iterations on the real
panel geometry: 701_863 rows x 149 symbols, horizon=5, n_boot=1000 -- measured at
~471 seconds for one `_compute_bucket_stats` call). The fix replaces that loop with a
single batched `rng.integers` draw per chunk plus broadcasting-based gather indices
(see `_draw_block_start_positions` / `_blocks_to_gather_indices` in
`nifty_quant/research/expectancy.py`), while remaining bit-identical to the old
per-block-loop RNG draw sequence for the same seed (verified separately across 40
seeds and 4 chunk sizes against a verbatim transcription of the pre-fix loop; see the
correctness contract already pinned by `tests/test_expectancy_bootstrap_chunking.py`,
which this file does not duplicate).

This file exists to catch a REGRESSION back to a per-block Python loop, not to pin an
exact number: the fixture and threshold below were calibrated (see this file's git
history / task report) so that the OLD per-block-loop implementation clearly BLOWS the
budget (~6.6s measured) while the vectorized implementation clearly clears it with
margin (~0.9s measured) on the same hardware -- a >=7x spread, not a hairline margin,
per CLAUDE.md rule 9 (prefer a large effect-to-noise ratio over a tight threshold that
flakes on correct code).
"""

from __future__ import annotations

import time

import numpy as np

from nifty_quant.research.expectancy import _block_bootstrap_means_chunked

# Generous budget: measured ~0.9s for the vectorized implementation and ~6.6s for the
# pre-fix per-block Python loop on the same fixture (see module docstring). 3.0s
# leaves > 2x headroom above the vectorized measurement and is still comfortably below
# half of the pre-fix measurement, so ordinary CI slowdown cannot flip the verdict.
_TIME_BUDGET_SECONDS = 3.0


class TestBlockBootstrapMeansChunkedIsVectorized:
    def test_moderate_panel_completes_within_generous_time_budget(self) -> None:
        n_rows, n_symbols = 500_000, 1
        day_offsets = np.array([0, n_rows], dtype=np.int64)
        horizon = 5
        n_boot = 700

        rng = np.random.default_rng(5)
        values = rng.normal(size=(n_rows, n_symbols))

        t0 = time.perf_counter()
        boot_means, _block_indices, n_sessions_skipped = _block_bootstrap_means_chunked(
            values, day_offsets, horizon, n_boot, seed=0, chunk_size=1
        )
        elapsed = time.perf_counter() - t0

        # Sanity: still a real, non-degenerate result (guards against a "fast because
        # it stopped computing anything" false pass).
        assert n_sessions_skipped == 0
        assert boot_means.shape == (n_boot,)
        assert np.isfinite(boot_means).all()

        assert elapsed < _TIME_BUDGET_SECONDS, (
            f"_block_bootstrap_means_chunked took {elapsed:.2f}s, over the "
            f"{_TIME_BUDGET_SECONDS}s budget -- this is the signature of a regression "
            "back to a per-block Python loop (see module docstring)."
        )
