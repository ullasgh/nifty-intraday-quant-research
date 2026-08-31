"""Test the wiring of VAR_TRIAL_SHARPES_PHASE_E into the criterion-6 deflated Sharpe path.

Tests obligations (i) and (ii) from the amendment to specs/lens_criteria_6_7_repair.md:

(i) the criterion-6 path calls expected_max_sharpe with VAR_TRIAL_SHARPES_PHASE_E, not 1.0.
    Verify by monkeypatching expected_max_sharpe with a recording wrapper, drive the
    criterion-6 branch via the lens's public entry, assert the recorded kwarg equals
    the constant.

(ii) VAR_TRIAL_SHARPES_PHASE_E equals the value parsed from the committed report file
     results/phase_e_sweep_report_2026-08-30.txt (parse the "var_trial_sharpes (MEASURED)="
     line). This pins the constant to its recorded derivation so it cannot silently drift.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nifty_quant.research.lens import VAR_TRIAL_SHARPES_PHASE_E
from tests.test_lens_criteria_repair_a import (
    _PASS_LATENCY,
    _call_verdict,
    _isolate_c6_panel,
)


def test_criterion6_calls_expected_max_sharpe_with_var_trial_sharpes_phase_e() -> None:
    """Obligation (i): The criterion-6 path calls expected_max_sharpe with the module
    constant VAR_TRIAL_SHARPES_PHASE_E, not 1.0.

    Uses a monkeypatch recording wrapper around expected_max_sharpe to capture
    the keyword argument passed when the criterion-6 branch executes.
    """
    panel = _isolate_c6_panel()
    good_returns = np.random.default_rng(1).normal(0.01, 0.001, size=500)

    # Record calls to expected_max_sharpe
    recorded_calls: list[dict] = []

    def recording_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        recorded_calls.append({"args": args, "kwargs": kwargs})
        # Call the original implementation to get a real return value
        from nifty_quant.backtest.metrics import expected_max_sharpe as original
        return original(*args, **kwargs)

    with patch(
        "nifty_quant.research.lens.expected_max_sharpe", side_effect=recording_wrapper
    ):
        _call_verdict(
            panel,
            latency_profile=_PASS_LATENCY,
            strategy_returns=good_returns,
            effective_n_trials=5,
        )

    # expected_max_sharpe should have been called at least once with
    # var_trial_sharpes=VAR_TRIAL_SHARPES_PHASE_E
    assert len(recorded_calls) > 0, "expected_max_sharpe was not called"
    found_call = False
    for call in recorded_calls:
        if call["kwargs"].get("var_trial_sharpes") == VAR_TRIAL_SHARPES_PHASE_E:
            found_call = True
            break

    assert found_call, (
        f"expected_max_sharpe was never called with var_trial_sharpes={VAR_TRIAL_SHARPES_PHASE_E}. "
        f"Calls: {recorded_calls}"
    )


def test_var_trial_sharpes_phase_e_matches_sweep_report() -> None:
    """Obligation (ii): VAR_TRIAL_SHARPES_PHASE_E equals the value recorded in the
    committed sweep report results/phase_e_sweep_report_2026-08-30.txt.

    Parses the "var_trial_sharpes (MEASURED)=" line from the report and asserts
    the constant matches exactly. This pins the constant to its recorded derivation
    so it cannot silently drift from the report.
    """
    report_path = (
        Path(__file__).parent.parent
        / "results"
        / "phase_e_sweep_report_2026-08-30.txt"
    )
    assert report_path.exists(), f"Report file not found at {report_path}"

    with open(report_path) as f:
        content = f.read()

    # Parse the "var_trial_sharpes (MEASURED)=" line.
    # Example line from report: "var_trial_sharpes (MEASURED)= 0.0389957 ..."
    match = re.search(
        r"var_trial_sharpes\s*\(MEASURED\)\s*=\s*([\d.]+)", content
    )
    assert match, "Could not parse 'var_trial_sharpes (MEASURED)=' from report file"

    reported_value = float(match.group(1))

    assert VAR_TRIAL_SHARPES_PHASE_E == reported_value, (
        f"VAR_TRIAL_SHARPES_PHASE_E ({VAR_TRIAL_SHARPES_PHASE_E}) does not match "
        f"the reported value from the sweep ({reported_value})"
    )
