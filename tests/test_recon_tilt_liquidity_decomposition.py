"""Tests for scripts/recon_tilt_liquidity_decomposition.py's criterion-4 mirror.

The script's job is to run already-tested production machinery (`Lens.stability`,
`compute_prior_adv`, `expectancy.causal_buckets`) on the tilt's signal and then REPORT the
verdict. Almost all of it is a passthrough. The one piece that is not -- and the piece the
published PASS/FAIL rests on -- is `report_lens_criterion_4`, which re-implements the
concentration comparison that lives inline inside `Lens.verdict()` (research/lens.py:703-777)
so the intermediate max/median ratio is visible instead of collapsed into a string.

A mirror of inline logic has no standalone target to diff against, so it is tested directly
here against synthetic `StabilityReport`s: uniform deciles must PASS, a bottom decile that
dominates beyond the measured 4.8695 threshold must FAIL, and -- the discriminating case --
a MIDDLE decile that dominates by the same margin must still PASS, because criterion 4 is
about the bottom (illiquid) decile specifically, not about dispersion in general.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from nifty_quant.research.expectancy import ExpectancyTable
from nifty_quant.research.lens import CONCENTRATION_RATIO_THRESHOLD, StabilityReport

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "recon_tilt_liquidity_decomposition.py"
)
_spec = importlib.util.spec_from_file_location(
    "recon_tilt_liquidity_decomposition", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
recon = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = recon
_spec.loader.exec_module(recon)


def _table(spread_bps: float, n_total: int = 1867) -> ExpectancyTable:
    """A minimal ExpectancyTable carrying only the fields criterion 4 reads."""
    return ExpectancyTable(
        buckets=(),
        horizon=1,
        feature_name="overnight_gap",
        n_total=n_total,
        cost_hurdle_bps=1.0,
        spread_bps=spread_bps,
        spread_t=spread_bps,
        survives_costs=False,
    )


def _report(
    decile_spreads: dict[int, float],
    tod_spreads: dict[str, float] | None = None,
) -> StabilityReport:
    if tod_spreads is None:
        # Two buckets, same sign, ratio 1.0 -- the time-of-day leg cannot fire, so the
        # liquidity leg is isolated.
        tod_spreads = {"open": -20.0, "mid": -20.0}
    return StabilityReport(
        by_year={},
        by_time_of_day={k: _table(v) for k, v in tod_spreads.items()},
        by_liquidity_decile={k: _table(v) for k, v in decile_spreads.items()},
        n_years_total=8,
        n_years_sign_consistent=8,
        dominant_sign="-",
    )


def test_uniform_deciles_pass() -> None:
    """Ten deciles of equal magnitude: max/median == 1.0, nothing fires."""
    report = _report({d: -20.0 for d in range(10)})
    result, reasons, spreads = recon.report_lens_criterion_4(report)
    assert result == "PASS"
    assert reasons == []
    assert len(spreads) == 10


def test_bottom_decile_domination_fails() -> None:
    """Bottom decile beyond the measured threshold, and the argmax: criterion 4 FAILS."""
    spreads = {d: -10.0 for d in range(10)}
    spreads[0] = -10.0 * (CONCENTRATION_RATIO_THRESHOLD + 1.0)
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "FAIL"
    assert any("bottom liquidity decile" in r for r in reasons)


def test_middle_decile_domination_still_passes() -> None:
    """Same dispersion, but decile 5 dominates instead of decile 0: must still PASS.

    This is the case that separates "concentrated in the illiquid tail" from "dispersed".
    A mirror that tested only the ratio and forgot `bottom_is_argmax` would fail here.
    """
    spreads = {d: -10.0 for d in range(10)}
    spreads[5] = -10.0 * (CONCENTRATION_RATIO_THRESHOLD + 1.0)
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "PASS"
    assert reasons == []


def test_bottom_decile_ratio_just_below_threshold_passes() -> None:
    """Strictly-greater-than comparison: a ratio at 99% of the threshold does not fire."""
    spreads = {d: -10.0 for d in range(10)}
    spreads[0] = -10.0 * CONCENTRATION_RATIO_THRESHOLD * 0.99
    result, _, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "PASS"


def test_only_bottom_decile_has_edge_fails() -> None:
    """If the bottom decile is the ONLY decile with any edge, that is concentration."""
    spreads = {d: 0.0 for d in range(10)}
    spreads[0] = -30.0
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "FAIL"
    assert any("only decile with edge" in r for r in reasons)


def test_time_of_day_sign_disagreement_fails() -> None:
    """Mixed-sign time-of-day buckets fire the time-of-day leg independently."""
    result, reasons, _ = recon.report_lens_criterion_4(
        _report({d: -20.0 for d in range(10)}, tod_spreads={"open": -20.0, "mid": +20.0})
    )
    assert result == "FAIL"
    assert any("sign" in r for r in reasons)


def test_single_time_of_day_bucket_fails() -> None:
    """One bucket with data is concentration by definition, per lens.py:741-744."""
    result, reasons, _ = recon.report_lens_criterion_4(
        _report({d: -20.0 for d in range(10)}, tod_spreads={"open": -20.0})
    )
    assert result == "FAIL"
    assert any("single time-of-day" in r for r in reasons)


@pytest.mark.parametrize("attr,expected", [("SEED", 0), ("HORIZON", 1), ("N_DECILES", 10)])
def test_pre_registered_constants(attr: str, expected: int) -> None:
    """The run is deterministic: seed and geometry are fixed in the file, not inferred."""
    assert getattr(recon, attr) == expected
