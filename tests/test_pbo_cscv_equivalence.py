"""Equivalence tests for the fast pbo_cscv optimization using per-block sufficient statistics.

The fast implementation precomputes per-block sum, sum-of-squares, and count statistics once,
then each combination is an n_splits-length reduction instead of a full-matrix pass per
combination. This test ensures that per-trial Sharpes computed from statistics exactly match
those from the reference implementation.

Floating-point sufficiency-stats summation reorders operations, so Sharpes may differ at
~1e-12 level, but PBO depends on RANKS which are stable away from exact ties.
"""

import time

import numpy as np
import pytest

from nifty_quant.backtest.metrics import _pbo_cscv_reference, pbo_cscv


class TestPboCscvEquivalence:
    """Equivalence tests for pbo_cscv fast path against reference implementation."""

    # Lead split (review-derived): the full 270-case sweep costs ~13 minutes, almost all
    # of it in the deliberately slow reference, so it is `slow`-marked. The unmarked
    # subset below runs in the gate on every push, keeping the reference exercised (its
    # coverage floor) and the bitwise pin alive; the full sweep still runs under
    # `pytest -m slow` and in any release pass.
    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize(
        "t,n_trials,scale",
        [(64, 7, 1.0), (257, 40, 10.0), (1000, 2, 1.0)],
    )
    def test_pbo_cscv_exact_equivalence_fast_subset(
        self, t: int, n_trials: int, scale: float, seed: int
    ) -> None:
        """Gate-speed subset of the exhaustive sweep below -- same assertion, 9 cases.

        n_splits=8 (C(8,4)=70 combinations vs C(16,8)=12,870) keeps the gate cost in
        seconds while exercising the identical block/partition code paths; the slow
        sweep pins n_splits=16.
        """
        rng = np.random.default_rng(seed)
        trial_matrix = rng.normal(0.0, 0.01 * scale, size=(t, n_trials))
        assert pbo_cscv(trial_matrix, n_splits=8) == _pbo_cscv_reference(
            trial_matrix, n_splits=8
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", range(30))
    @pytest.mark.parametrize(
        "t,n_trials,scale",
        [
            (64, 2, 1.0),
            (64, 7, 1.0),
            (64, 40, 1.0),
            (257, 2, 1.0),
            (257, 7, 0.1),
            (257, 40, 10.0),
            (1000, 2, 1.0),
            (1000, 7, 1.0),
            (1000, 40, 1.0),
        ],
    )
    def test_pbo_cscv_exact_equivalence_across_shapes_and_scales(
        self, t: int, n_trials: int, scale: float, seed: int
    ) -> None:
        """Test that fast and reference implementations agree exactly (bitwise) on random matrices.

        Matrices span shapes (T in {64, 257, 1000}, n_trials in {2, 7, 40}) with mixed scales,
        seeded 0..29. The exact equivalence (not just approx) is required because PBO ranking
        must be stable.
        """
        rng = np.random.default_rng(seed)
        trial_matrix = rng.normal(0.0, 0.01 * scale, size=(t, n_trials))

        # Fast and reference implementations must agree exactly (n_splits=16, the
        # production setting -- the gate-speed subset above covers n_splits=8)
        pbo_fast = pbo_cscv(trial_matrix, n_splits=16)
        pbo_ref = _pbo_cscv_reference(trial_matrix, n_splits=16)

        # Bitwise equality check
        assert pbo_fast == pbo_ref, (
            f"Seed {seed}, shape ({t}, {n_trials}), scale {scale}: "
            f"fast={pbo_fast}, reference={pbo_ref}, diff={abs(pbo_fast - pbo_ref)}"
        )

    def test_pbo_cscv_near_tied_matrix(self) -> None:
        """Test a deliberately near-tied matrix (two identical columns).

        When two trials have nearly identical OOS Sharpes, rankdata's 'average' method
        is tested. Both implementations must handle ties identically.
        """
        rng = np.random.default_rng(42)
        t = 500
        n_trials = 10

        # Create a matrix with two nearly identical columns
        base_trial = rng.normal(0.0, 0.01, size=t)
        other_trials = rng.normal(0.0, 0.01, size=(t, n_trials - 2))

        # Make two trials nearly identical (with tiny noise to avoid exact ties)
        tied_trial_1 = base_trial.copy()
        tied_trial_2 = base_trial + rng.normal(0.0, 1e-10, size=t)

        trial_matrix = np.column_stack([tied_trial_1, tied_trial_2, other_trials])

        pbo_fast = pbo_cscv(trial_matrix, n_splits=8)
        pbo_ref = _pbo_cscv_reference(trial_matrix, n_splits=8)

        assert pbo_fast == pbo_ref, (
            f"Near-tied matrix: fast={pbo_fast}, reference={pbo_ref}"
        )

    def test_pbo_cscv_nan_handling(self) -> None:
        """Test that NaN values are handled identically by both implementations.

        The current contract allows NaN in the input and treats it as missing data.
        """
        rng = np.random.default_rng(51)
        t = 300
        n_trials = 5

        trial_matrix = rng.normal(0.0, 0.01, size=(t, n_trials))

        # Inject some NaNs
        nan_mask = rng.uniform(size=(t, n_trials)) < 0.05
        trial_matrix[nan_mask] = np.nan

        pbo_fast = pbo_cscv(trial_matrix, n_splits=8)
        pbo_ref = _pbo_cscv_reference(trial_matrix, n_splits=8)

        assert pbo_fast == pbo_ref, (
            f"NaN-handling matrix: fast={pbo_fast}, reference={pbo_ref}"
        )

    @pytest.mark.slow
    def test_pbo_cscv_timing_fast_path(self) -> None:
        """Timing sanity check: print wall-clock time comparison (not asserted).

        Measures the fast path on a synthetic 50k x 114 matrix at n_splits=16.
        On the real Phase E merge (514,070 x 114, n_splits=16), the reference
        implementation took ~37 minutes. This sanity line allows manual inspection
        of speedup without making wall-clock assertions (which flake).
        """
        # Create a representative test matrix (50k x 114 is 1/10 the Phase E size)
        rng = np.random.default_rng(99)
        t = 50_000
        n_trials = 114
        trial_matrix = rng.normal(0.0, 0.01, size=(t, n_trials))

        # Time the fast path
        start_fast = time.perf_counter()
        pbo_fast = pbo_cscv(trial_matrix, n_splits=16)
        elapsed_fast = time.perf_counter() - start_fast

        # Time the reference path
        start_ref = time.perf_counter()
        pbo_ref = _pbo_cscv_reference(trial_matrix, n_splits=16)
        elapsed_ref = time.perf_counter() - start_ref

        # Verify they still match
        assert pbo_fast == pbo_ref

        # Print timing info (not asserted, for manual inspection)
        speedup_factor = elapsed_ref / elapsed_fast if elapsed_fast > 0 else float('inf')
        print(f"\nTiming on {t} x {n_trials} matrix (n_splits=16):")
        print(f"  Reference (old): {elapsed_ref:.3f} seconds")
        print(f"  Fast (new):      {elapsed_fast:.3f} seconds")
        print(f"  Speedup factor:  {speedup_factor:.1f}x")
        print(f"  Estimated Phase E speedup (514k rows): {speedup_factor:.1f}x")


class TestPboCscvReferenceMaintainsCoverage:
    """Tests that the reference implementation is still exercised by the new test suite.

    Like _rolling_beta_reference, _pbo_cscv_reference must remain covered by tests
    to ensure the equivalence check stays runnable.
    """

    def test_reference_is_exercised_by_equivalence_suite(self) -> None:
        """Smoke test: calling _pbo_cscv_reference from this module confirms it's exercised."""
        rng = np.random.default_rng(123)
        trial_matrix = rng.normal(0.0, 0.01, size=(100, 5))

        # Simply calling it ensures coverage
        result = _pbo_cscv_reference(trial_matrix, n_splits=4)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
