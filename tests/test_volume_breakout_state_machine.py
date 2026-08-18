# tests/test_volume_breakout_state_machine.py
"""Targeted state-machine coverage for volume_breakout.on_decision.

Covers, with direct behavioral assertions (not incidental hits):

  - lines 338-339: the SHORT-side cooldown suppression arm
    (``elif cooldown_side[sym] == -1: new_short[i] = False``), which mirrors
    the already-covered LONG arm at 336-337.
  - line 362: the gross-exposure rescale
    (``clipped = clipped * (p.gross / abs_sum)``) when the clipped raw
    weights' L1 sum exceeds ``p.gross``.

``on_decision`` is called directly (bypassing ``precompute``/panel
construction entirely) so every input -- signals, prior shares, decision
timestamp -- is exact and reproducible, per ``tests/test_volume_breakout.py``'s
existing ``on_decision``-direct pattern (see e.g. ``test_weights_are_bounded_and_finite``).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nifty_quant.strategy.base import PortfolioState
from nifty_quant.strategy.plugins.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
)


@dataclass
class _ViewStub:
    ts: int
    session_date: datetime.date
    symbols: tuple[str, ...]
    tradable: np.ndarray

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        raise NotImplementedError("not needed by on_decision in these tests")

    def window(self, field: str, n: int, *, ffill: bool = False) -> np.ndarray:
        raise NotImplementedError("not needed by on_decision in these tests")


def _make_view(
    symbols: tuple[str, ...], hhmmss: str, tradable: np.ndarray | None = None
) -> _ViewStub:
    ts = int(pd.Timestamp(f"2024-01-01 {hhmmss}", tz="Asia/Kolkata").timestamp())
    n = len(symbols)
    return _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 1),
        symbols=symbols,
        tradable=np.ones(n, dtype=bool) if tradable is None else tradable.astype(bool),
    )


def _make_state(shares: np.ndarray, ts: int) -> PortfolioState:
    return PortfolioState(
        shares=np.asarray(shares, dtype=np.float64),
        cash=0.0,
        equity=1_000_000.0,
        ts=ts,
    )


def _make_signals(
    n: int,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "long": np.asarray(long_mask, dtype=bool),
        "short": np.asarray(short_mask, dtype=bool),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "vol_z": np.zeros(n, dtype=np.float64),
        "hurst": np.full(n, np.inf, dtype=np.float64),
    }


def test_short_entry_exit_sets_cooldown_side_negative_one() -> None:
    """Establishes (and directly verifies) the precondition lines 320-321 must
    produce before the short-cooldown arm (338-339) can ever fire: a SHORT
    exit must set ``cooldown_side[sym] == -1``, not leave it at the LONG
    convention (``1``) or the initial value (``0``).
    """
    symbols = ("AAA", "BBB")
    params = VolumeBreakoutParams(
        exit_mode="time",
        hold_bars=1,
        min_hold_bars=1,
        cooldown_bars=5,
        max_weight=0.5,
        gross=10.0,
        sigma_floor=1e-5,
    )
    strategy = VolumeBreakoutStrategy(params)

    # Bar 0: flat -> AAA gets a fresh SHORT entry signal, BBB stays flat.
    view0 = _make_view(symbols, "10:00:00")
    signals0 = _make_signals(
        2,
        long_mask=np.array([False, False]),
        short_mask=np.array([True, False]),
        sigma=np.array([0.1, 0.1]),
    )
    state0 = _make_state(np.array([0.0, 0.0]), view0.ts)
    target0 = strategy.on_decision(view0, signals0, state0)
    assert target0 is not None
    assert target0.weights[0] < 0.0, "AAA must enter short on the signal bar"
    assert target0.weights[1] == 0.0

    # Bar 1: engine has filled the short (shares now negative for AAA). With
    # hold_bars=min_hold_bars=1 the position is now old enough to exit on
    # this very bar, which must set cooldown_side["AAA"] = -1.
    view1 = _make_view(symbols, "10:01:00")
    signals1 = _make_signals(
        2,
        long_mask=np.array([False, False]),
        short_mask=np.array([False, False]),
        sigma=np.array([0.1, 0.1]),
    )
    state1 = _make_state(np.array([-100.0, 0.0]), view1.ts)
    target1 = strategy.on_decision(view1, signals1, state1)
    assert target1 is not None
    assert target1.weights[0] == 0.0, "AAA target must go flat on the exit bar"
    assert strategy._cooldown_side is not None
    assert strategy._cooldown_side["AAA"] == -1
    assert strategy._cooldown_remaining is not None
    assert strategy._cooldown_remaining["AAA"] == 5


def test_short_cooldown_suppresses_same_side_reentry_but_not_other_symbol() -> None:
    """Lines 338-339: a symbol in SHORT cooldown must have a fresh short
    signal suppressed (``new_short[i] = False``), while a *different* symbol
    with no active cooldown must enter normally in the same call -- the
    contrast that proves suppression is symbol-scoped, not global.

    This also drives branches 336->338 (cooldown_side != 1, so the long arm
    is skipped) and 338->339 (cooldown_side == -1, so the short arm fires)
    for real, on a fresh short signal that would otherwise have entered.
    """
    symbols = ("AAA", "BBB")
    params = VolumeBreakoutParams(
        exit_mode="time",
        hold_bars=1,
        min_hold_bars=1,
        cooldown_bars=5,
        max_weight=0.5,
        gross=10.0,
        sigma_floor=1e-5,
    )
    strategy = VolumeBreakoutStrategy(params)

    # Bar 0: AAA enters short.
    view0 = _make_view(symbols, "10:00:00")
    signals0 = _make_signals(
        2,
        long_mask=np.array([False, False]),
        short_mask=np.array([True, False]),
        sigma=np.array([0.1, 0.1]),
    )
    state0 = _make_state(np.array([0.0, 0.0]), view0.ts)
    target0 = strategy.on_decision(view0, signals0, state0)
    assert target0 is not None
    assert target0.weights[0] < 0.0

    # Bar 1: AAA's short exits (hold_bars=min_hold_bars=1), cooldown starts.
    view1 = _make_view(symbols, "10:01:00")
    signals1 = _make_signals(
        2,
        long_mask=np.array([False, False]),
        short_mask=np.array([False, False]),
        sigma=np.array([0.1, 0.1]),
    )
    state1 = _make_state(np.array([-100.0, 0.0]), view1.ts)
    target1 = strategy.on_decision(view1, signals1, state1)
    assert target1 is not None
    assert target1.weights[0] == 0.0

    # Bar 2: engine reports AAA flat again (position closed). A fresh short
    # signal fires for BOTH AAA (still in cooldown) and BBB (never in
    # cooldown). AAA must be suppressed; BBB must enter normally.
    view2 = _make_view(symbols, "10:02:00")
    signals2 = _make_signals(
        2,
        long_mask=np.array([False, False]),
        short_mask=np.array([True, True]),
        sigma=np.array([0.1, 0.1]),
    )
    state2 = _make_state(np.array([0.0, 0.0]), view2.ts)
    target2 = strategy.on_decision(view2, signals2, state2)
    assert target2 is not None
    assert target2.weights[0] == 0.0, (
        "AAA's fresh short signal must be suppressed by the same-side cooldown "
        "(line 339: new_short[i] = False)"
    )
    assert target2.weights[1] < 0.0, (
        "BBB has no active cooldown and must enter short normally in the "
        "same call -- proves suppression is per-symbol, not a global block"
    )
    assert strategy._cooldown_remaining is not None
    # Not in exited_this_bar this call, so the trailing decrement loop fires.
    assert strategy._cooldown_remaining["AAA"] == 4


def test_cooldown_side_neither_arm_falls_through_without_suppressing() -> None:
    """White-box probe for the elif's implicit "no match" fall-through arc.

    NAMED INVARIANT: at every call site that writes ``cooldown_side`` (lines
    328-329, ``cooldown_side[sym] = 1 if pos > 0 else -1``), it is written in
    the same statement as ``cooldown_remaining[sym] = p.cooldown_bars``. There
    is no other write site, and the two dicts are always replaced together
    (never partially) on the reinit path (``set(self._bars_in_position) !=
    set(view.symbols)``). Consequently, for every state reachable through the
    public ``on_decision`` API, ``cooldown_remaining[sym] > 0`` implies
    ``cooldown_side[sym] in (1, -1)`` -- ``cooldown_side[sym] == 0`` can never
    coincide with an active cooldown.

    That makes the elif's own "neither side matched" fall-through (the arc
    coverage.py labels ``346->342``: the elif condition is False and control
    returns directly to the ``for`` loop, as opposed to falling through the
    body at line 347) UNREACHABLE via any sequence of on_decision calls --
    confirmed empirically with an isolated 8-line reproduction of this exact
    if/elif-in-for-loop shape: two symbols both taking the True elif branch
    leaves that arc reported missing (``6->2``) by ``coverage run --branch``,
    while a symbol with a "neither" side value covers it. It is exercised
    here only by directly priming the strategy's own persistent state dicts
    to a combination the state machine itself can never produce, in order to
    verify the elif is a genuine no-op fall-through (does not accidentally
    suppress) rather than a claim that this combination occurs in practice.
    """
    symbols = ("AAA",)
    params = VolumeBreakoutParams(exit_mode="time", max_weight=0.5, gross=10.0)
    strategy = VolumeBreakoutStrategy(params)

    # Call 0: any flat, no-signal bar, purely to initialize the per-symbol
    # state dicts for this exact symbol set (matches the reinit condition).
    view0 = _make_view(symbols, "10:00:00")
    signals0 = _make_signals(
        1,
        long_mask=np.array([False]),
        short_mask=np.array([False]),
        sigma=np.array([0.1]),
    )
    state0 = _make_state(np.array([0.0]), view0.ts)
    strategy.on_decision(view0, signals0, state0)

    assert strategy._cooldown_remaining is not None
    assert strategy._cooldown_side is not None
    # Prime the state machine's own dicts to the off-nominal combination the
    # write sites can never produce (see the docstring invariant above).
    strategy._cooldown_remaining["AAA"] = 5
    strategy._cooldown_side["AAA"] = 0

    view1 = _make_view(symbols, "10:01:00")
    signals1 = _make_signals(
        1,
        long_mask=np.array([False]),
        short_mask=np.array([True]),
        sigma=np.array([0.1]),
    )
    state1 = _make_state(np.array([0.0]), view1.ts)
    target1 = strategy.on_decision(view1, signals1, state1)

    assert target1 is not None
    assert target1.weights[0] < 0.0, (
        "an out-of-domain cooldown_side (neither 1 nor -1) must fall through "
        "the elif without suppressing the entry -- the elif is exhaustive "
        "over the two real cooldown sides and has no else clause"
    )


def test_gross_weight_clipping_rescales_exactly_to_gross_when_saturated() -> None:
    """Line 362: when the L1 sum of max-weight-clipped raw weights exceeds
    ``p.gross``, every active weight is rescaled by ``p.gross / abs_sum`` so
    that the realized gross exposure is exactly ``p.gross`` (float64
    tolerance) -- the documented sizing guarantee, not merely "less than".

    Three symbols saturate at max_weight=0.5 (sigma tiny relative to
    sigma_floor forces the clip), giving abs_sum = 1.5 > gross = 1.0.
    """
    symbols = ("AAA", "BBB", "CCC")
    params = VolumeBreakoutParams(
        exit_mode="time",
        max_weight=0.5,
        gross=1.0,
        sigma_floor=1e-5,
    )
    strategy = VolumeBreakoutStrategy(params)

    view = _make_view(symbols, "10:00:00")
    signals = _make_signals(
        3,
        long_mask=np.array([True, True, True]),
        short_mask=np.array([False, False, False]),
        # Well below sigma_floor: sigma_guard clamps to sigma_floor, so
        # raw = sign / sigma_guard saturates far past max_weight for all three.
        sigma=np.array([1e-6, 1e-6, 1e-6]),
    )
    state = _make_state(np.array([0.0, 0.0, 0.0]), view.ts)

    target = strategy.on_decision(view, signals, state)
    assert target is not None
    weights = target.weights

    assert np.all(np.isfinite(weights))
    expected = params.max_weight * (params.gross / (3 * params.max_weight))
    assert weights == pytest.approx(np.full(3, expected), rel=1e-9)
    assert np.sum(np.abs(weights)) == pytest.approx(params.gross, abs=1e-9), (
        "documented invariant: after the line-362 rescale, "
        "sum(|weights|) == p.gross exactly (within float64 tolerance)"
    )
