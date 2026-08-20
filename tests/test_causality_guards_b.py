"""Independent test suite B for `specs/causality_guards.md` (Phase B track B4).

Written from the spec ALONE: no sibling test file and no implementation exist yet.
`universe_causal` and `execution_causal` are not present in `nifty_quant.guards` as of
this writing, so this whole file is expected to be RED -- most of it will fail at
collection/call time with an `AttributeError`/`ImportError` rather than a clean
assertion failure, until Track B4 lands the two guards. That is the correct state for
a spec with no implementation yet, not a mistake in this file.

SPEC DEFECTS / AMBIGUITIES FOUND (read before touching this file)
-------------------------------------------------------------------------------------
1. No concrete signature is given for `universe_causal`/`execution_causal`, unlike
   `causal` (guards.py:267), which already documents `row_arg`, `n_probes`, `seed`,
   `domain`. This file ASSUMES, by analogy and "in the same style as @causal":
     - `universe_causal(*, panel_arg=0, n_probes=3, seed=0)` wraps a function
       `f(panel: np.ndarray) -> np.ndarray` (bool mask, same leading two axes as
       `panel`).
     - `execution_causal(*, row_args=("prices", "bar_traded_value"), n_probes=3,
       seed=0)` wraps a function of `(orders, prices, bar_traded_value, tradable)`.
   If the real implementation names its parameters differently, every test below
   fails at the call boundary (`TypeError`), which looks identical to "the guard is
   broken" without this note.

2. Section B says "Perturb volume, high, low and close", but the concrete "fill path"
   in this repo, `FillModel.fill()` (fills.py:66), takes only `prices` and
   `bar_traded_value` -- there is no `high`/`low` argument anywhere in the signature,
   and `volume` is already folded into `bar_traded_value` by the caller
   (engine.py:475: `bar_traded_value = volume_safe[t] * prices`). This file perturbs
   `prices` and `bar_traded_value` only, since that is the actual perturbable surface
   of the actual function the spec claims to describe.

3. `execution_causal` is required to check "fill quantity, fill price AND charges"
   together, but `FillModel.fill()` returns `FillResult` (filled_qty, fill_price,
   rejected, unfilled_notional) -- there is no charges field. Charges come from a
   separate `CostModel.charges(FillBatch)` call that section B never mentions
   composing with the fill step. This file composes them in a test-local
   `_fill_and_charge` helper into one stacked ndarray column-wise, matching
   `@causal`'s existing "output must be an ndarray" convention -- an assumption, since
   section B never states `execution_causal`'s output-type contract at all.

4. Section A.3's directionality claim doesn't actually hold for the item-1 canary
   itself: `np.any(present, axis=0)` is direction-symmetric and (as this file's own
   construction shows -- see `test_add_only_probe_misses_the_remove_direction_canary`
   and its docstring) is independently catchable by EITHER an add-only or a
   remove-only probe alone, depending on the fixture. The claim only becomes true for
   a *naive* probe that behaves like today's `@causal` row-redraw: it turns a future
   NaN into a value (so it exercises "appears later") but never turns a future value
   into NaN (so it can NEVER exercise "disappears later"). The spec conflates "the
   item-1 canary" with "the case that needs both directions"; this file therefore
   builds a SECOND, direction-isolating pair of fixtures for item 3, rather than
   reusing the item-1 canary, and adds a small test-local (non-production) probe to
   make the "checking only one direction misses a real leak" claim checkable at all.

5. Section A never defines "panel" (shape, dtype, presence convention) despite
   CLAUDE.md rule 6 ("NaN means no bar occurred") being the obvious candidate. This
   file assumes a 2-D float64 array shaped (session, symbol) with NaN = absent. An
   implementer who instead threads a separate boolean `present` array (which is
   arguably MORE faithful to CLAUDE.md rule 7's "present and tradable must never be
   conflated") would break every test here at the call boundary, not the assertion
   boundary.

6. Item 6 ("no-ops at OFF and CHEAP") doesn't say whether "no-op" means the wrapped
   function still runs unguarded (this file's assumption, matching `@causal`'s and
   `deterministic`'s existing OFF/CHEAP behaviour in guards.py) or is skipped/stubbed
   out entirely. Assumed the former, since the spec says "in the same style".

7. Item 8's "or if they cannot distinguish the two, that limitation is documented" is
   an escape hatch this file cannot verify by itself -- documentation lives in
   `src/`, which this file must not touch. This file (a) proves the case it believes
   IS distinguishable (per-row-seeded, future-independent noise) does NOT false-fire,
   and (b) separately pins a DIFFERENT, real blind spot that mirrors
   `research/expectancy.py:939` exactly (a leak baked into row t's OWN value, not
   into a row after it) and asserts the guard has nothing to say about it. This file
   can only confirm the blind spot is real, not that `src/` documents it -- that is a
   review-time check, not a test-time one.
-------------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant import guards
from nifty_quant.execution.costs import FillBatch, NSEIntradayEquityCosts
from nifty_quant.execution.fills import FillModel, ZeroSlippage


@pytest.fixture(autouse=True)
def _restore_strictness():
    saved = guards.get_strictness()
    yield
    guards.set_strictness(saved)


# ---------------------------------------------------------------------------------
# Shared fixtures / canaries -- universe causality
# ---------------------------------------------------------------------------------


def _staggered_panel() -> np.ndarray:
    """8 sessions x 4 symbols, NaN = absent (assumption 5 above).

    Symbol 0: present every session (no boundary).
    Symbol 1: first appears at session 3 -- a "first present bar" boundary.
    Symbol 2: present ONLY at session 5, isolated -- the case that specifically
        needs the REMOVAL perturbation direction (item 3).
    Symbol 3: present through session 4, then disappears -- a "last present bar"
        boundary in the opposite direction.
    """
    panel = np.full((8, 4), np.nan)
    panel[:, 0] = np.arange(8) + 100.0
    panel[3:, 1] = np.arange(5) + 200.0
    panel[5, 2] = 300.0
    panel[:5, 3] = np.arange(5) + 400.0
    return panel


def _leaky_any_over_panel(panel: np.ndarray) -> np.ndarray:
    """Required test 1 canary: `np.any(present, axis=0)` broadcast to every row --
    the textbook static-universe leak named verbatim by the spec (lines 37-38)."""
    present = np.isfinite(panel)
    ever = np.any(present, axis=0)
    return np.broadcast_to(ever, panel.shape).copy()


def _correct_causal_eligibility(panel: np.ndarray) -> np.ndarray:
    """A genuinely causal mask: eligible at session d iff present at or before d.
    Required test 2: decorating this must PASS, not just "not obviously fail"."""
    present = np.isfinite(panel)
    return np.cumsum(present, axis=0) > 0


def _leaky_forward_any(panel: np.ndarray) -> np.ndarray:
    """Leaky in a way orthogonal to the item-1 canary: eligible at d if present at d
    OR AT ANY LATER session. Used to build direction-isolating fixtures for item 3
    (see assumption/defect 4)."""
    present = np.isfinite(panel)
    n = panel.shape[0]
    out = np.zeros_like(present, dtype=bool)
    for d in range(n):
        out[d] = np.any(present[d:], axis=0)
    return out


def _causal_eligibility_with_per_row_noise(panel: np.ndarray) -> np.ndarray:
    """Causal AND genuinely non-deterministic for reasons that have nothing to do
    with lookahead: each row's coin flip is seeded by that ROW'S OWN INDEX alone
    (never by future data, never by the panel's total length). Required test item 8:
    a guard unable to tell "depends on the future" apart from "depends on an
    independent per-row seed" would false-fire here.
    """
    present = np.isfinite(panel)
    base = np.cumsum(present, axis=0) > 0
    out = base.copy()
    for d in range(panel.shape[0]):
        rng = np.random.default_rng(d)
        flip = rng.random(panel.shape[1]) > 0.97
        out[d] = base[d] ^ flip
    return out


# ---------------------------------------------------------------------------------
# Required test 1 + 2 -- universe_causal catches the canary, passes correct code
# ---------------------------------------------------------------------------------


def test_universe_causal_catches_full_panel_any_canary():
    """Required test 1: `np.any(present, axis=0)` is caught. If this passes the
    guard, the guard is broken (spec line 66)."""
    panel = _staggered_panel()
    guarded = guards.universe_causal(panel_arg=0)(_leaky_any_over_panel)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(panel)


def test_universe_causal_passes_correct_eligibility():
    """Required test 2: a genuinely causal eligibility function must PASS. A guard
    tight enough to also reject correct code is as useless as one that misses bad
    code (task brief) -- this is the counterpart to test 1, not optional."""
    panel = _staggered_panel()
    guarded = guards.universe_causal(panel_arg=0)(_correct_causal_eligibility)
    with guards.strictness(guards.Strictness.FULL):
        result = guarded(panel)
    np.testing.assert_array_equal(result, _correct_causal_eligibility(panel))


# ---------------------------------------------------------------------------------
# Required test 3 -- both perturbation directions
# ---------------------------------------------------------------------------------


def test_universe_causal_catches_remove_direction_leak():
    """Required test 3, remove direction. Symbol present ONLY at session 4; probing
    session 0 (always probed per spec A.4's "include the first session") can only
    reveal the leak by REMOVING that future presence -- adding presence elsewhere
    changes nothing, since session 4 is already present in the baseline."""
    panel = np.full((6, 1), np.nan)
    panel[4, 0] = 999.0
    guarded = guards.universe_causal(panel_arg=0)(_leaky_forward_any)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(panel)


def test_universe_causal_catches_add_direction_leak():
    """Required test 3, add direction. Symbol present NOWHERE in the baseline; the
    leak (retroactive eligibility) can only be exposed by ADDING presence to a
    future row -- there is nothing already present to remove."""
    panel = np.full((6, 1), np.nan)
    guarded = guards.universe_causal(panel_arg=0)(_leaky_forward_any)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(panel)


def _naive_add_only_probe_misses(fn, panel: np.ndarray, cut: int, seed: int) -> bool:
    """TEST-ONLY scaffold -- NOT a second implementation of `universe_causal`, and
    NOT touching src/. Exists solely to make the spec's directionality claim (lines
    35-38) checkable: this probe redraws future NaN cells into finite values (like
    `@causal`'s existing row-redraw does today) but never turns an existing finite
    future cell back into NaN. Returns True iff row `cut` is unchanged, i.e. the
    naive, add-only probe found nothing.
    """
    rng = np.random.default_rng(seed)
    baseline = fn(panel)
    perturbed = panel.copy()
    future = perturbed[cut + 1 :]
    was_absent = ~np.isfinite(future)
    future[was_absent] = rng.standard_normal(size=int(was_absent.sum()))
    return bool(np.array_equal(baseline[cut], fn(perturbed)[cut], equal_nan=True))


def test_add_only_probe_misses_the_remove_direction_canary():
    """Pins WHY item 3 requires both directions (defect 4): the naive, add-only
    probe above finds nothing on the exact remove-only leak that
    `test_universe_causal_catches_remove_direction_leak` requires the REAL
    `universe_causal` to catch. A guard built the naive way would silently pass a
    provably leaky function.
    """
    panel = np.full((6, 1), np.nan)
    panel[4, 0] = 999.0
    assert _naive_add_only_probe_misses(_leaky_forward_any, panel, cut=0, seed=0)


# ---------------------------------------------------------------------------------
# Required test 4 + 5 -- execution_causal catches the canary, passes the real model
# ---------------------------------------------------------------------------------


def _leaky_fill_reads_future_btv(orders, prices, bar_traded_value, tradable):
    """Required test 4 canary: reads `bar_traded_value[t+1]` when computing the
    participation cap -- spec's own example of "the one input where reading the
    wrong row is both easy and invisible" (section B, referencing fills.py:101)."""
    orders = np.asarray(orders, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    btv = np.asarray(bar_traded_value, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    btv_next = np.empty_like(btv)
    btv_next[:-1] = btv[1:]
    btv_next[-1] = btv[-1]

    desired_notional = np.abs(orders) * prices
    max_notional = 0.02 * btv_next
    filled_notional = np.minimum(desired_notional, max_notional)
    safe_prices = np.where(prices > 0, prices, 1.0)
    filled_qty = np.sign(orders) * filled_notional / safe_prices
    filled_qty = np.where(tradable, filled_qty, 0.0)
    return np.stack([filled_qty, prices], axis=-1)


def _fill_and_charge(orders, prices, bar_traded_value, tradable):
    """Test-local composition of the REAL `FillModel` + REAL cost model (defect 3):
    `execution_causal` needs a single ndarray to compare row-for-row, and neither
    `FillResult` nor `Charges` is one on its own. Columns: filled_qty, fill_price,
    total_charges.
    """
    model = FillModel(slippage=ZeroSlippage())
    result = model.fill(orders, prices, bar_traded_value, tradable)
    notional = np.abs(result.filled_qty) * result.fill_price
    batch = FillBatch(notional=notional, is_buy=np.asarray(orders) > 0)
    charges = NSEIntradayEquityCosts().charges(batch)
    return np.stack([result.filled_qty, result.fill_price, charges.total], axis=-1)


def test_execution_causal_catches_future_btv_read():
    """Required test 4."""
    orders = np.array([10.0, -5.0, 20.0, 8.0, -12.0, 6.0])
    prices = np.full(6, 100.0)
    tradable = np.ones(6, dtype=bool)
    btv = np.array([1e6, 2e6, 5e5, 3e6, 1.5e6, 4e6])

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _leaky_fill_reads_future_btv
    )
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation):
            guarded(orders, prices, btv, tradable)


def test_execution_causal_passes_for_real_fill_model():
    """Required test 5: the real `FillModel` (plus real cost model) must PASS, the
    execution-side counterpart to `test_universe_causal_passes_correct_eligibility`.
    A false positive on genuinely causal production code is as damaging as a missed
    leak (task brief)."""
    rng = np.random.default_rng(7)
    n = 10
    orders = rng.standard_normal(n) * 50.0
    prices = 100.0 + rng.standard_normal(n)
    tradable = np.ones(n, dtype=bool)
    btv = np.abs(rng.standard_normal(n)) * 1e6 + 1e5

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _fill_and_charge
    )
    with guards.strictness(guards.Strictness.FULL):
        result = guarded(orders, prices, btv, tradable)

    expected = _fill_and_charge(orders, prices, btv, tradable)
    np.testing.assert_allclose(result, expected)


def test_execution_causal_passes_when_participation_cap_binds():
    """Required test 5, participation-cap-specific: section B explicitly singles
    out `fills.py:101` (`max_participation * bar_traded_value`) as the highest-risk
    input, so exercise a fixture where the cap actually BINDS (desired notional >
    cap for at least one row), not only an always-under-cap case."""
    n = 8
    prices = np.full(n, 50.0)
    tradable = np.ones(n, dtype=bool)
    btv = np.full(n, 1e4)
    orders = np.array([1000.0, -800.0, 1500.0, 600.0, -1200.0, 900.0, -500.0, 1100.0])

    model = FillModel(slippage=ZeroSlippage(), max_participation=0.02)
    desired_notional = np.abs(orders) * prices
    cap = model.max_participation * btv
    assert np.any(desired_notional > cap), "fixture must actually bind the cap"

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _fill_and_charge
    )
    with guards.strictness(guards.Strictness.FULL):
        result = guarded(orders, prices, btv, tradable)

    expected = _fill_and_charge(orders, prices, btv, tradable)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------------
# Required test 6 -- no-op at OFF/CHEAP, active at FULL
# ---------------------------------------------------------------------------------


def test_universe_causal_noop_below_full():
    """Required test 6, universe side."""
    panel = _staggered_panel()
    guarded = guards.universe_causal(panel_arg=0)(_leaky_any_over_panel)
    expected = _leaky_any_over_panel(panel)
    for level in (guards.Strictness.OFF, guards.Strictness.CHEAP):
        with guards.strictness(level):
            result = guarded(panel)
        np.testing.assert_array_equal(result, expected)


def test_execution_causal_noop_below_full():
    """Required test 6, execution side."""
    orders = np.array([10.0, -5.0, 20.0])
    prices = np.array([100.0, 100.0, 100.0])
    tradable = np.ones(3, dtype=bool)
    btv = np.array([1e6, 1e6, 1e6])

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _leaky_fill_reads_future_btv
    )
    expected = _leaky_fill_reads_future_btv(orders, prices, btv, tradable)
    for level in (guards.Strictness.OFF, guards.Strictness.CHEAP):
        with guards.strictness(level):
            result = guarded(orders, prices, btv, tradable)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------------
# Required test 7 -- failure message names the row/session and the quantity
# ---------------------------------------------------------------------------------


def test_universe_causal_message_names_session_and_symbol():
    """Required test 7, universe side: message must name the row/session that moved
    and the quantity (eligibility/mask), not just "something changed"."""
    panel = _staggered_panel()
    guarded = guards.universe_causal(panel_arg=0)(_leaky_any_over_panel)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as excinfo:
            guarded(panel)

    message = str(excinfo.value)
    assert any(str(d) in message for d in range(panel.shape[0])), (
        f"message must name the session/row index that moved: {message!r}"
    )
    assert any(
        term in message.lower() for term in ("eligib", "mask", "universe", "session")
    ), f"message must name the quantity that moved: {message!r}"


def test_execution_causal_message_names_row_and_quantity():
    """Required test 7, execution side: message must name the row/session index and
    which quantity moved (qty/price/charges), not just "something changed"."""
    orders = np.array([10.0, -5.0, 20.0, 8.0, -12.0, 6.0])
    prices = np.full(6, 100.0)
    tradable = np.ones(6, dtype=bool)
    btv = np.array([1e6, 2e6, 5e5, 3e6, 1.5e6, 4e6])

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _leaky_fill_reads_future_btv
    )
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as excinfo:
            guarded(orders, prices, btv, tradable)

    message = str(excinfo.value)
    assert any(str(i) in message for i in range(len(orders))), (
        f"message must name the row/session index that moved: {message!r}"
    )
    assert any(
        term in message.lower() for term in ("qty", "quantity", "price", "fill")
    ), f"message must name the quantity that moved: {message!r}"


# ---------------------------------------------------------------------------------
# Required test 8 -- no false-fire on legitimate non-lookahead non-determinism, and
# the documented blind spot (section D), mirroring research/expectancy.py:939.
# ---------------------------------------------------------------------------------


def test_universe_causal_does_not_fire_on_row_seeded_noise():
    """Required test 8, first clause: per-row-seeded noise that does not depend on
    the future must not be flagged."""
    panel = _staggered_panel()
    guarded = guards.universe_causal(panel_arg=0)(
        _causal_eligibility_with_per_row_noise
    )
    with guards.strictness(guards.Strictness.FULL):
        result = guarded(panel)
    np.testing.assert_array_equal(
        result, _causal_eligibility_with_per_row_noise(panel)
    )


def test_execution_causal_cannot_see_a_leak_already_baked_into_row_t():
    """Required test 8, second clause / section D: pins a real blind spot, mirroring
    `research/expectancy.py:939` exactly ("prior_adv must already be lagged ...
    @causal cannot detect a leak in it"). `execution_causal` can only prove row t's
    output is unaffected by rows strictly AFTER t; it cannot prove row t's OWN
    `bar_traded_value` was honestly computed from data <= t before ever reaching
    `fill()`. Simulate a caller that centred `bar_traded_value` on a forward-looking
    2-bar window -- contaminating row t itself, not a row after it -- and show the
    guard raises nothing even though the contamination demonstrably changed row 0's
    result relative to the honest input.
    """
    n = 6
    raw_btv = np.array([1e6, 2e6, 5e5, 3e6, 1.5e6, 4e6])
    contaminated_btv = raw_btv.copy()
    contaminated_btv[:-1] = 0.5 * (raw_btv[:-1] + raw_btv[1:])

    orders = np.array([250.0, -5.0, 20.0, 8.0, -12.0, 6.0])
    prices = np.full(n, 100.0)
    tradable = np.ones(n, dtype=bool)

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        _fill_and_charge
    )
    with guards.strictness(guards.Strictness.FULL):
        result = guarded(orders, prices, contaminated_btv, tradable)  # must NOT raise

    clean_result = _fill_and_charge(orders, prices, raw_btv, tradable)
    assert not np.array_equal(result[0], clean_result[0]), (
        "fixture must actually leak at row 0 for the blind spot to be real, not just "
        "assumed"
    )


# ---------------------------------------------------------------------------------
# Constraints section -- guards sample rows, they do not check every row (and do not
# silently check zero either). Pinned so a future change can't silently regress
# either way without a test noticing.
# ---------------------------------------------------------------------------------


def test_universe_causal_perturbs_a_sample_not_every_row():
    """An exhaustive guard would call the wrapped function ~500 times on a
    500-session panel (one perturbation per session); a guard that silently checks
    nothing would call it exactly once (baseline only, zero probes). Pin a small,
    bounded call count strictly between those two failure modes."""
    calls = {"n": 0}

    def eligibility(panel: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        present = np.isfinite(panel)
        return np.cumsum(present, axis=0) > 0

    rng = np.random.default_rng(3)
    n_sessions = 500
    big_panel = rng.standard_normal((n_sessions, 5))
    guarded = guards.universe_causal(panel_arg=0)(eligibility)

    with guards.strictness(guards.Strictness.FULL):
        guarded(big_panel)

    assert 2 <= calls["n"] <= 10, (
        f"expected a small, bounded number of probe calls (a sample, not every "
        f"session, and not zero either), got {calls['n']} calls for a "
        f"{n_sessions}-session panel"
    )


def test_execution_causal_perturbs_a_sample_not_every_row():
    """Same pin as above, execution side: FULL strictness must stay usable, so
    `execution_causal` must not re-run the fill path once per row on a large book."""
    calls = {"n": 0}

    def fill_fn(orders, prices, bar_traded_value, tradable):
        calls["n"] += 1
        model = FillModel(slippage=ZeroSlippage())
        result = model.fill(orders, prices, bar_traded_value, tradable)
        return np.stack([result.filled_qty, result.fill_price], axis=-1)

    rng = np.random.default_rng(9)
    n_rows = 1_000
    orders = rng.standard_normal(n_rows) * 20.0
    prices = 100.0 + rng.standard_normal(n_rows)
    tradable = np.ones(n_rows, dtype=bool)
    btv = np.abs(rng.standard_normal(n_rows)) * 1e6 + 1e5

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(fill_fn)

    with guards.strictness(guards.Strictness.FULL):
        guarded(orders, prices, btv, tradable)

    assert 2 <= calls["n"] <= 10, (
        f"expected a small, bounded number of probe calls (a sample, not every "
        f"row), got {calls['n']} calls for a {n_rows}-row book"
    )
