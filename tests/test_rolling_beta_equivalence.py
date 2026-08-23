"""Equivalence check: vectorized `rolling_beta` vs. the private `np.cov`-based reference.

`rolling_beta` was rewritten from a per-(row, symbol) `np.cov`/`np.var` loop to a
NaN-aware cumulative-sum reformulation for a ~480x speedup (see the docstring on
`rolling_beta` in `nifty_quant.features.market` for the full measured evidence: the
downstream bucket-assignment / spread_bps acceptance test on real data showed 0
reassignments). The two implementations are NOT bit-identical (summation-order
differs whenever a window contains any NaN), so this test guards the SIZE of that
deviation on every run rather than letting the claim live only in a comment.

`_rolling_beta_reference` is kept private in `market.py` for exactly this purpose.
"""

from __future__ import annotations

import numpy as np

from nifty_quant.features.market import _rolling_beta_reference, rolling_beta


def _returns_with_gaps(n_rows: int, n_symbols: int, seed: int, nan_frac: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.001, size=(n_rows, n_symbols))
    gaps = rng.random(size=(n_rows, n_symbols)) < nan_frac
    x[gaps] = np.nan
    return x


def _market_with_gaps(n_rows: int, seed: int, nan_frac: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.001, size=n_rows)
    gaps = rng.random(size=n_rows) < nan_frac
    x[gaps] = np.nan
    return x


def test_rolling_beta_matches_reference_within_measured_tolerance() -> None:
    """NaN mask must match EXACTLY; values must match to well within the measured
    1.458e-12 (window=60, real data) deviation -- generous enough for synthetic data
    of a different scale, tight enough to catch a real algebraic bug (e.g. wrong ddof,
    wrong window alignment), which produces O(1) not O(1e-9) differences."""
    n_rows, n_symbols, window = 800, 12, 60
    day_offsets = np.array([0, 250, 500, n_rows], dtype=np.int64)

    returns = _returns_with_gaps(n_rows, n_symbols, seed=1, nan_frac=0.05)
    market = _market_with_gaps(n_rows, seed=2, nan_frac=0.03)

    fast = rolling_beta(returns, market, window, day_offsets=day_offsets)
    ref = _rolling_beta_reference(returns, market, window, day_offsets=day_offsets)

    fast_mask = np.isfinite(fast)
    ref_mask = np.isfinite(ref)
    assert np.array_equal(fast_mask, ref_mask), "NaN mask must be identical (hard requirement)"

    both = fast_mask & ref_mask
    assert both.any(), "fixture produced no comparable finite betas"
    max_abs_diff = float(np.max(np.abs(fast[both] - ref[both])))
    assert max_abs_diff < 1e-8, f"max abs diff {max_abs_diff:.3e} exceeds tolerance"
    assert np.allclose(fast[both], ref[both], atol=1e-9, rtol=1e-6)


def test_rolling_beta_matches_reference_no_gaps() -> None:
    """With no NaNs at all, both implementations sum over the exact same, full-length
    window every time -- this should be much closer to bit-identical than the
    gap-containing case."""
    n_rows, n_symbols, window = 500, 8, 30
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.001, size=(n_rows, n_symbols))
    market = rng.normal(0, 0.001, size=n_rows)

    fast = rolling_beta(returns, market, window, day_offsets=day_offsets)
    ref = _rolling_beta_reference(returns, market, window, day_offsets=day_offsets)

    both = np.isfinite(fast) & np.isfinite(ref)
    assert np.array_equal(np.isfinite(fast), np.isfinite(ref))
    assert both.any()
    max_abs_diff = float(np.max(np.abs(fast[both] - ref[both])))
    assert max_abs_diff < 1e-9, f"max abs diff {max_abs_diff:.3e} exceeds tolerance"


def test_rolling_beta_float32_input_takes_the_exact_reference_path() -> None:
    """`rolling_beta`'s fast (cumulative-sum) path is gated to float64-only input (see
    its docstring's "dtype gate" section): float32 input must be BIT-IDENTICAL to
    `_rolling_beta_reference`, not merely close, because it takes the exact same code
    path (this is what `test_rolling_beta_matches_hand_computed_cov_over_var` in
    `test_market_features.py` also depends on, via a float32 fixture)."""
    n_rows, n_symbols, window = 60, 4, 10
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.001, size=(n_rows, n_symbols)).astype(np.float32)
    market = rng.normal(0, 0.001, size=n_rows).astype(np.float32)

    fast = rolling_beta(returns, market, window, day_offsets=day_offsets)
    ref = _rolling_beta_reference(returns, market, window, day_offsets=day_offsets)

    assert np.array_equal(np.isfinite(fast), np.isfinite(ref))
    both = np.isfinite(fast) & np.isfinite(ref)
    assert both.any()
    assert np.array_equal(fast[both], ref[both]), "float32 input must skip the fast path entirely"
