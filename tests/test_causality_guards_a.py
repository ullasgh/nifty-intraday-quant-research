"""Tests for the proposed universe- and execution-causality decorators.

SPEC DEFECTS / AMBIGUITIES:
* The signatures of ``universe_causal`` and ``execution_causal`` are unspecified:
  in particular, it is unclear whether they accept ``row_arg``, named arguments,
  multiple row arguments, or infer inputs from the wrapped callable.
* The mapping between sessions, rows, names, and panel axes is unspecified, as is
  how a name appearing or disappearing should be represented in an array.
* The execution requirement names volume, spread, high, low, and close, while the
  supplied ``FillModel.fill`` signature exposes orders, prices,
  ``bar_traded_value``, and ``tradable``; ``FillResult`` also has no charges field.
  These tests use one packed row-wise array and treat prices as close and
  ``bar_traded_value`` as the volume-derived cap input.
* It is unspecified whether guards support arbitrary return types such as
  ``FillResult``, how equality is applied to structured outputs, and whether the
  guard may mutate arguments in place.
* "Several" sampled sessions and the exact required first/last/boundary probe
  schedule are not quantified.
* The required ContractViolation message contents are described semantically but
  no exact format is specified.
* The operational meaning of documenting a blind spot for seeded stochastic
  functions is unspecified, especially for functions using global rather than
  explicitly supplied RNG state.
"""

import re

import numpy as np
import pytest

from nifty_quant import guards
from nifty_quant.execution.fills import FillModel, ZeroSlippage

_MAX_PARTICIPATION = 0.25


def _present_panel() -> np.ndarray:
    """Eight sessions with both first-presence and last-presence boundaries."""
    present = np.ones((8, 4), dtype=bool)
    present[:5, 2] = False  # Symbol 2 first has a bar at session 5.
    present[7, 3] = False  # Symbol 3 has bars through session 6 only.
    return present


def leaky_eligibility(present: np.ndarray) -> np.ndarray:
    return np.tile(np.any(present, axis=0), (present.shape[0], 1))


def causal_eligibility(present: np.ndarray) -> np.ndarray:
    # Eligibility is based only on whether the name has a bar on this session.
    return present.copy()


def _fill_panel() -> np.ndarray:
    # Columns are orders, execution price/close, bar traded value, and tradable.
    # Values vary by row so a future-row read changes the deliberately leaky model.
    # The final column is tradability, not presence; no missing bars are ffilled.
    return np.array(
        [
            [3.0, 100.0, 200.0, 1.0],
            [-2.0, 101.0, 400.0, 1.0],
            [1.0, 99.0, 80.0, 0.0],
            [4.0, 102.0, 600.0, 1.0],
        ],
        dtype=np.float64,
    )


def leaky_fill(bars: np.ndarray) -> np.ndarray:
    orders = bars[:, 0]
    prices = bars[:, 1]
    bar_traded_value = bars[:, 2]
    tradable = bars[:, 3].astype(bool)

    next_btv = np.empty_like(bar_traded_value)
    # Deliberate leak: row t reads the next bar's traded value.  The last row
    # sensibly falls back to its own value rather than wrapping around.
    next_btv[:-1] = bar_traded_value[np.arange(1, len(bar_traded_value))]
    next_btv[-1] = bar_traded_value[-1]

    cap_qty = _MAX_PARTICIPATION * next_btv / prices
    filled_qty = np.where(
        tradable,
        np.sign(orders) * np.minimum(np.abs(orders), cap_qty),
        0.0,
    )
    fill_price = np.where(filled_qty != 0.0, prices, 0.0)
    charges = np.abs(filled_qty) * prices * 0.001  # Fixture-only charge rate.
    return np.column_stack((filled_qty, fill_price, charges))


def _assert_row_and_quantity(message: str, quantity_name: str) -> None:
    text = str(message).lower()
    assert quantity_name.lower() in text
    assert re.search(r"(?:row|session)[^0-9]{0,30}[0-9]", text)


def test_universe_causal_catches_full_history_any_leak_canary():
    present = _present_panel()
    raw_mask = leaky_eligibility(present)

    # This establishes the leak independently of the guard: symbol 2 is absent
    # on row 0, but whole-panel np.any incorrectly marks it eligible.
    # Truthiness, not `is True`: these are numpy bool scalars, and
    # `np.bool_(True) is True` is False. `== True` says the right thing but trips
    # ruff E712, so assert truthiness -- identical semantics, lint-clean.
    assert raw_mask[0, 2]
    assert not present[0, 2]

    @guards.universe_causal(row_arg=0)
    def guarded(p):
        return leaky_eligibility(p)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(present)


def test_universe_causal_passes_correct_eligibility():
    present = _present_panel()

    @guards.universe_causal(row_arg=0)
    def guarded(p):
        return causal_eligibility(p)

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(present)

    np.testing.assert_array_equal(result, present)


def test_universe_causal_both_perturbation_directions_and_one_directional_check_misses_canary():
    present = _present_panel()

    # Explicitly exercise both meaningful panel transformations.  The first
    # introduces a later appearance; the second removes later bars from a name
    # already present on row 0.
    appears_later = present.copy()
    appears_later[1:5, 2] = True
    disappears_later = present.copy()
    disappears_later[1:7, 3] = False
    assert np.any(~present[1:5, 2] & appears_later[1:5, 2])
    assert np.any(present[1:7, 3] & ~disappears_later[1:7, 3])

    def disappearance_only_checker(fn, panel):
        baseline = fn(panel)
        # Cuts include the first and last sessions and both sides of the
        # symbol-2 first-presence boundary (sessions 4 and 5).
        for d in (0, 4, 5, 6, 7):
            probe = panel.copy()
            already_present = panel[d]
            probe[d + 1 :, already_present] = False
            assert not np.any(probe & ~panel)  # no False -> True perturbation
            if not np.array_equal(baseline[d], fn(probe)[d]):
                return True
        return False

    assert disappearance_only_checker(leaky_eligibility, present) is False

    @guards.universe_causal(row_arg=0)
    def guarded(p):
        return leaky_eligibility(p)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(present)


def test_execution_causal_catches_bar_traded_value_t_plus_1_leak():
    bars = _fill_panel()
    changed = bars.copy()
    changed[1, 2] *= 2.0  # A future cap input is changed, while row 0 is intact.
    assert leaky_fill(changed)[0, 0] != leaky_fill(bars)[0, 0]

    @guards.execution_causal(row_arg=0)
    def guarded(b):
        return leaky_fill(b)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(bars)


def test_execution_causal_real_fill_model_passes():
    bars = _fill_panel()
    model = FillModel(
        slippage=ZeroSlippage(),
        max_participation=_MAX_PARTICIPATION,
    )

    @guards.execution_causal(row_arg=0)
    def guarded(b):
        result = model.fill(
            b[:, 0],
            b[:, 1],
            b[:, 2],
            b[:, 3].astype(bool),
        )
        # ZeroSlippage supplies a zero-charge projection; retaining the actual
        # fill quantity and price keeps the guarded result numeric and row-wise.
        charges = np.zeros_like(result.filled_qty, dtype=np.float64)
        return np.column_stack((result.filled_qty, result.fill_price, charges))

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(bars)

    assert result.shape == (len(bars), 3)


@pytest.mark.parametrize("level", [guards.Strictness.OFF, guards.Strictness.CHEAP])
def test_universe_and_execution_causal_noop_at_off_and_cheap(level):
    present = _present_panel()
    bars = _fill_panel()

    @guards.universe_causal(row_arg=0)
    def guarded_universe(p):
        return leaky_eligibility(p)

    @guards.execution_causal(row_arg=0)
    def guarded_execution(b):
        return leaky_fill(b)

    expected_universe = leaky_eligibility(present)
    expected_execution = leaky_fill(bars)
    with guards.strictness(level):
        actual_universe = guarded_universe(present)
        actual_execution = guarded_execution(bars)

    np.testing.assert_array_equal(actual_universe, expected_universe)
    np.testing.assert_allclose(
        actual_execution, expected_execution, equal_nan=True
    )


def test_universe_and_execution_causal_active_at_full():
    present = _present_panel()
    bars = _fill_panel()

    @guards.universe_causal(row_arg=0)
    def guarded_universe(p):
        return leaky_eligibility(p)

    @guards.execution_causal(row_arg=0)
    def guarded_execution(b):
        return leaky_fill(b)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded_universe(present)
        with pytest.raises(guards.ContractViolation):
            guarded_execution(bars)


def test_universe_and_execution_causal_violation_message_names_row_and_quantity():
    present = _present_panel()
    bars = _fill_panel()

    @guards.universe_causal(row_arg=0)
    def guarded_universe(p):
        return leaky_eligibility(p)

    @guards.execution_causal(row_arg=0)
    def guarded_execution(b):
        return leaky_fill(b)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as universe_error:
            guarded_universe(present)
        with pytest.raises(guards.ContractViolation) as execution_error:
            guarded_execution(bars)

    _assert_row_and_quantity(str(universe_error.value), "leaky_eligibility")
    _assert_row_and_quantity(str(execution_error.value), "leaky_fill")


def test_universe_and_execution_causal_do_not_fire_on_seed_dependent_non_lookahead():
    present = _present_panel()

    def seeded_eligibility(p, seed):
        # The data values are never read; only the explicit shape and seed are
        # used, so this is stochastic but not a future-row dependency.
        return np.random.default_rng(seed).random(p.shape) > 0.5

    @guards.universe_causal(row_arg=0)
    def guarded(p, seed):
        return seeded_eligibility(p, seed)

    # Honest blind-spot note: if the guard cannot preserve the explicit seed
    # across its re-runs and flags this function, that false positive documents
    # the section-D limitation rather than being hidden by weakening the test.
    with guards.strictness(guards.Strictness.FULL):
        first = guarded(present, seed=19)
        second = guarded(present, seed=19)

    np.testing.assert_array_equal(first, second)
