"""One shared minimal `ResearchContract` for tests that are not ABOUT contracts.

`run_backtest()` and `run_tilt()` now require a `ResearchContract` (spec C4,
`specs/research_contract.md`). Around a hundred pre-existing tests across ten files call those
entry points to exercise engine mechanics, order lifecycle, tilt behaviour and CLI plumbing --
none of them is about pre-registration, and none should be asserting anything about a contract.

This module exists so those ten files share ONE definition instead of inventing ten. This repo has
already produced a broken build from two agents implementing opposite sides of a single interface,
and a contract built ten slightly different ways is the same hazard spread wider: a later change to
the contract's shape would then have ten places to miss.

Tests that ARE about contracts (`tests/test_research_contract_{a,b}.py`) build their own and must
NOT import this -- they are testing the construction rules themselves.
"""

from __future__ import annotations

from nifty_quant.research.contract import ResearchContract

# Deliberately inert values. Nothing here should make a test pass or fail: if a test's outcome
# changes when these change, that test is accidentally depending on contract content and is
# mis-scoped.
_MINIMAL_SECTIONS: dict[str, dict[str, object]] = {
    "data": {"panel_id": "test-panel", "panel_hash": "test", "universe_name": "test"},
    "features": {"ids": (), "feature_version": "test"},
    "label": {"horizon_bars": 1, "overlapping": False},
    "execution": {"cost_model_id": "test", "slippage_model_id": "test"},
    "portfolio": {"sizing": "test"},
    "validation": {"scheme": "test", "holdout_intent": "never"},
}


def minimal_contract(**overrides: object) -> ResearchContract:
    """A valid, inert contract for tests that merely need to satisfy the gate.

    Pass `**overrides` to replace a whole section (or `seed`) when a test genuinely needs a
    specific value -- for example a holdout-intent test overriding `validation`.
    """
    sections: dict[str, object] = {
        name: dict(body) for name, body in _MINIMAL_SECTIONS.items()
    }
    sections["seed"] = 0
    sections.update(overrides)
    return ResearchContract(**sections)  # type: ignore[arg-type]
