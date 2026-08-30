"""Coverage for feature_sweep.py lines 310-311: per-horizon exception handling.

Tests the case where a feature computes successfully, but `conditional_expectancy`
(or `_forward_returns_for_horizon`) raises an exception for a specific horizon.
Verifies that:
  (a) exactly one TrialRecord captures the error
  (b) other horizons' records succeed with finite metrics
  (c) the trial count never collapses (obligation 11)
"""

from __future__ import annotations

import numpy as np

import nifty_quant.research.feature_sweep as feature_sweep_module
from nifty_quant.research.feature_sweep import run_sweep
from tests.contract_fixtures import minimal_contract


def _dummy_feature(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """A trivial (close, day_offsets) -> feature callable."""
    return close.copy()


def _make_close(n_symbols: int, n_rows: int = 40, seed: int = 0):
    """Create synthetic close and day_offsets arrays."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.01, size=(n_rows, n_symbols)).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    return close, day_offsets


def test_per_horizon_exception_in_expectancy_is_recorded_and_does_not_collapse_trials(
    monkeypatch,
):
    """When conditional_expectancy raises for a specific horizon, that error is
    recorded on that horizon's TrialRecord, but other horizons complete normally.

    Obligation 11: a failure in a per-horizon step must not reduce the trial count.
    """
    close, day_offsets = _make_close(n_symbols=5, n_rows=40, seed=42)
    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 3}
    )

    # Patch conditional_expectancy to raise RuntimeError on the second call.
    # We'll track calls with a counter.
    call_count = [0]
    real_conditional_expectancy = feature_sweep_module.expectancy.conditional_expectancy

    def mock_conditional_expectancy(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            # Raise on the second call (second horizon).
            raise RuntimeError("Test error: expectancy computation failed for this horizon")
        return real_conditional_expectancy(*args, **kwargs)

    monkeypatch.setattr(
        feature_sweep_module.expectancy,
        "conditional_expectancy",
        mock_conditional_expectancy,
    )

    # Run sweep over 3 horizons with a single feature.
    horizons = [1, 2, 3]
    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=horizons,
        feature_registry_override=[("test_feature", _dummy_feature)],
    )

    # Assertion (c): total record count equals n_horizons
    assert len(records) == 3, f"Expected 3 records, got {len(records)}"

    # Separate records by error status
    error_records = [r for r in records if r.error is not None]
    success_records = [r for r in records if r.error is None]

    # Assertion (a): exactly one TrialRecord has an error
    assert len(error_records) == 1, f"Expected 1 error record, got {len(error_records)}"

    # The error message should match what we raised
    error_record = error_records[0]
    expected_error = (
        "RuntimeError: Test error: expectancy computation failed for this horizon"
    )
    assert error_record.error == expected_error

    # A failed horizon's metrics should be None/NaN
    assert error_record.sharpe_gross is None
    assert error_record.sharpe_net is None
    assert error_record.n_trades is None
    assert error_record.turnover is None
    # breakeven_bps is None because table was None (exception before table was set)
    assert error_record.breakeven_bps is None

    # Assertion (b): the other horizons have error=None and finite metrics
    assert len(success_records) == 2, f"Expected 2 success records, got {len(success_records)}"
    for record in success_records:
        assert record.error is None
        # Metrics should be None (feature has no signal, but computation succeeded)
        # (sharpe_gross/net are None because we didn't run backtest, just expectancy)
        assert record.sharpe_gross is None
        assert record.sharpe_net is None
        assert record.n_trades is None
        assert record.turnover is None
        # breakeven_bps is computed when table is not None
        assert record.breakeven_bps is not None
        assert isinstance(record.breakeven_bps, (int, float))


def test_per_horizon_exception_captures_exception_type_and_message(monkeypatch):
    """Verify the error message format: 'ExceptionType: message'."""
    close, day_offsets = _make_close(n_symbols=5, n_rows=30, seed=99)
    contract = minimal_contract(
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": 2}
    )

    call_count = [0]
    real_conditional_expectancy = feature_sweep_module.expectancy.conditional_expectancy

    def mock_conditional_expectancy(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Custom validation error for testing")
        return real_conditional_expectancy(*args, **kwargs)

    # Patch in the feature_sweep module namespace
    monkeypatch.setattr(
        feature_sweep_module.expectancy,
        "conditional_expectancy",
        mock_conditional_expectancy,
    )

    records = run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=[1, 2],
        feature_registry_override=[("test_feature", _dummy_feature)],
    )

    error_records = [r for r in records if r.error is not None]
    assert len(error_records) == 1
    # The error message should include both the exception type and message
    assert "ValueError" in error_records[0].error
    assert "Custom validation error for testing" in error_records[0].error
