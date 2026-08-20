"""Cover the `deflated_sharpe(...) is nan` arm of criterion 6 (`lens.py:848-850`).

`deflated_sharpe` (nifty_quant.backtest.metrics) returns `NaN` when the sample
standard deviation is negligible relative to the data's scale (see
`_is_negligible_std`): a constant `strategy_returns` series has `std == 0`, which
is exactly that condition. This is a DIFFERENT gate from the earlier
`finite_returns.size < 4` short-circuit -- a constant array of length >= 4 clears
that gate cleanly and still reaches `deflated_sharpe`, which must then itself
report NaN.

Confirmed directly against `deflated_sharpe` before writing the verdict-level
test below (probe, not guessed):

    >>> import numpy as np
    >>> from nifty_quant.backtest.metrics import deflated_sharpe, expected_max_sharpe
    >>> const_returns = np.full(10, 0.001, dtype=np.float64)
    >>> exp_max = expected_max_sharpe(5, var_trial_sharpes=1.0)
    >>> deflated_sharpe(const_returns, sr0=exp_max)
    nan

Reuses `_isolate_c6_panel`, `_call_verdict`, `_criterion_token`, and
`_PASS_LATENCY` from `tests.test_lens_criteria_repair_a` rather than
reinventing panel-construction fixtures.
"""

from __future__ import annotations

import numpy as np

from tests.test_lens_criteria_repair_a import (
    _PASS_LATENCY,
    _call_verdict,
    _criterion_token,
    _isolate_c6_panel,
)


def test_criterion6_constant_returns_dsr_nan_reports_not_evaluated_not_fail() -> None:
    """A NaN deflated Sharpe (constant `strategy_returns`, zero std) must report
    NOT_EVALUATED -- never FAIL -- and the reason string must carry the `dsr=nan`
    detail plus the trial count. This is the exact contract `lens.py:848-850`
    exists to honour: a numerically-undefined DSR is silence, not a verdict.

    `_isolate_c6_panel()` makes every OTHER criterion genuinely PASS, so this
    isolates criterion 6 the same way the sibling precedent test in
    `test_lens_criteria_repair_a.py` does.
    """
    panel = _isolate_c6_panel()
    # 10 identical values: >= 4 finite survivors (clears the finite-count gate),
    # but zero sample std (fails deflated_sharpe's own negligible-std guard).
    constant_returns = np.full(10, 0.001, dtype=np.float64)

    verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=constant_returns,
        effective_n_trials=2,
    )

    reason = verdict.reasons[5]
    token = _criterion_token(reason)
    assert token == "NOT_EVALUATED"
    assert token != "FAIL", "a NaN dsr must never be reported as a FAIL"
    assert "dsr=nan" in reason
    assert "trials=2" in reason


def test_criterion6_constant_returns_dsr_nan_marks_verdict_INCONCLUSIVE() -> None:
    """The dsr=nan NOT_EVALUATED result must propagate to `any_not_evaluated` and
    `explain()`'s INCOMPLETE marker, and -- mirroring criterion 5's established
    precedent -- must NOT block `survived`: every other criterion genuinely
    PASSes on `_isolate_c6_panel()`, so `survived` stays True even though
    criterion 6 could not be evaluated.
    """
    panel = _isolate_c6_panel()
    constant_returns = np.full(10, 0.001, dtype=np.float64)

    verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=constant_returns,
        effective_n_trials=2,
    )

    assert _criterion_token(verdict.reasons[5]) == "NOT_EVALUATED"
    for i in (0, 1, 2, 3, 4, 6):
        assert _criterion_token(verdict.reasons[i]) == "PASS", (
            f"criterion {i + 1} must genuinely PASS to isolate criterion 6: "
            f"{verdict.reasons[i]!r}"
        )

    assert verdict.any_not_evaluated is True
    assert "INCOMPLETE" in verdict.explain()

    # ADJUDICATED, specs/lens_verdict_integrity.md L1. This previously asserted
    # `survived is True`, reasoning that "a NOT_EVALUATED criterion 6 must not block
    # survival, same as criterion 5" -- which is precisely the defect: unevaluated
    # criteria were DROPPED from the conjunction, so `survived` could be True with
    # kill criteria never run. A criterion that vanishes when unsupplied is not a
    # kill criterion. The verdict is now tri-state and this case is INCONCLUSIVE:
    # nothing FAILED, but the screen did not finish.
    assert verdict.outcome == "INCONCLUSIVE"
    assert verdict.survived is False


def test_criterion6_well_behaved_returns_on_same_fixture_gives_real_verdict() -> None:
    """Contrast case: the SAME panel and `effective_n_trials`, but with a
    well-behaved (non-constant, strongly positive) `strategy_returns` series,
    must produce a genuine PASS -- proving the NaN-dsr path above is what
    changed the outcome, not some defect in the shared fixture.
    """
    panel = _isolate_c6_panel()
    good_returns = np.random.default_rng(1).normal(0.01, 0.001, size=500)

    verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=good_returns,
        effective_n_trials=2,
    )

    reason = verdict.reasons[5]
    assert _criterion_token(reason) == "PASS"
    assert "dsr=nan" not in reason
    assert verdict.any_not_evaluated is False
    assert "INCOMPLETE" not in verdict.explain()
    assert verdict.survived is True
