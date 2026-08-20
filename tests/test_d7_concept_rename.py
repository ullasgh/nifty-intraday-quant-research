"""D7 (specs/feature_layer.md) -- volume z-score CONCEPT layer rename.

`volume_zscore` keeps its function name (features/core.py, owned by another agent
and not touched here). What must change is prose that frames a 1-minute volume
spike as proven institutional flow / smart money / accumulation, since the data
has no order-level attribution to support that mechanism claim -- see the
`volume_breakout` KILLED verdict (gross Sharpe -0.048 / net -0.233).

This test regression-guards the one production plugin docstring that made that
claim: strategy/plugins/volume_breakout.py's module docstring used to open with
"Institutional Volume Exhaustion & Momentum Capture."
"""
from __future__ import annotations

import inspect

from nifty_quant.strategy.plugins import volume_breakout


def test_volume_breakout_docstring_does_not_claim_institutional_flow() -> None:
    """The module docstring must not assert institutional-flow / smart-money framing."""
    doc = volume_breakout.__doc__ or ""
    lowered = doc.lower()
    assert "institutional" not in lowered, (
        "volume_breakout module docstring still frames the volume z-score signal as "
        "'institutional' flow -- D7 requires the concept layer say abnormal ACTIVITY, "
        "not a proven institutional mechanism the data cannot support"
    )
    assert "smart money" not in lowered
    assert "abnormal" in lowered, (
        "expected the corrected docstring to describe the signal as abnormal volume "
        "activity (spec D7's replacement concept name)"
    )


def test_volume_breakout_source_has_no_institutional_flow_prose() -> None:
    """Belt-and-braces: scan the whole module source, not just __doc__, for the claim."""
    src = inspect.getsource(volume_breakout)
    assert "Institutional Volume Exhaustion" not in src
