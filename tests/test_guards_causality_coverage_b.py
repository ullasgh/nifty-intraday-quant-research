"""Coverage-gap tests for `universe_causal` / `execution_causal` in
`src/nifty_quant/guards.py` (guards.py fell from 100% to 94.7% when Phase B added
these two decorators; the coverage floor is a ratchet -- this file is tests, not a
lowered floor).

This is ONE OF TWO independent authors closing the same gap list (the other writes
`test_guards_causality_coverage_a.py`); by design this file was written without
reading that one.

Every test here calls the real decorator and asserts on OBSERVED behaviour (an
exception type + message, a returned value, or a call count) -- never a mere
line-touch. Where a guard's own source-recovery helper (`_source_snippet`) needs to
hit its except-branch, a function is compiled from a string with a synthetic
filename so `inspect.getsource` genuinely fails (verified directly below), rather
than mocking anything.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from nifty_quant import guards


def _make_sourceless_func(name: str, params: str, body: str):
    """Compile a real function object from a string with a synthetic filename, so
    `inspect.signature` works normally but `inspect.getsource` cannot find a
    backing file and raises OSError -- exercising `_source_snippet`'s except
    branch (guards.py:401-402) with a genuine callable, not a mock."""
    src = f"def {name}({params}):\n{body}\n"
    code = compile(src, f"<synthetic:{name}>", "exec")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 -- deliberate, isolated, no external input
    return ns[name]


# ---------------------------------------------------------------------------
# `_source_snippet` except-branch (401-402), exercised via a genuine leak catch.
# ---------------------------------------------------------------------------


def test_source_snippet_omitted_when_getsource_fails_on_a_real_leak():
    """A function compiled from a string has no backing source file, so
    `inspect.getsource` raises OSError inside `_source_snippet`. The helper must
    swallow that and return `""` rather than propagating it through the
    `ContractViolation` raise path -- and the guard must still correctly detect
    the leak itself, since that's the only way this line executes at all."""
    leaky = _make_sourceless_func(
        "leaky",
        "panel",
        "    import numpy as np\n"
        "    present = panel if panel.dtype == np.bool_ else np.isfinite(panel)\n"
        "    total = int(present.sum())\n"
        "    return np.full(panel.shape[0], total, dtype=np.int64)\n",
    )
    with pytest.raises(OSError):
        inspect.getsource(leaky)  # sanity: confirm the premise before testing the guard

    guarded = guards.universe_causal(panel_arg=0)(leaky)
    # session 0's future (rows 1, 2) has absent cells that the add-direction probe
    # will force to present, changing the (leaky, whole-panel) total count.
    panel = np.array([[True, True], [False, True], [True, False]])
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(panel)
    msg = str(exc.value)
    assert "eligibility mask row/session 0 changed" in msg
    assert "\nsource:" not in msg  # _source_snippet returned "", not a traceback
    assert "leaky" in msg  # func.__name__ is still present, unaffected


# ---------------------------------------------------------------------------
# `universe_causal`
# ---------------------------------------------------------------------------


def test_universe_causal_full_strictness_via_env_fallback_when_cache_unset(monkeypatch):
    """When the module-level `_strictness` cache is unset (None), the wrapper
    must derive the level itself via `get_strictness()` (guards.py:447-448)
    rather than assume OFF/CHEAP. Proven by observation: FULL is derived purely
    from `NQ_STRICT`, with no `strictness()` context manager involved, and the
    guard genuinely catches a leak as a result."""

    def leaky(panel):
        present = panel if panel.dtype == np.bool_ else np.isfinite(panel)
        return np.full(panel.shape[0], int(present.sum()), dtype=np.int64)

    guarded = guards.universe_causal(panel_arg=0)(leaky)
    panel = np.array([[True, False], [False, True], [True, True]])

    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")
    with pytest.raises(guards.ContractViolation):
        guarded(panel)


def test_universe_causal_rejects_non_2d_panel_argument():
    """Argument-validation raise (guards.py:453-457): message must name the
    resolved argument and what was actually received."""

    def f(panel):
        return np.zeros(panel.shape[0])

    guarded = guards.universe_causal(panel_arg=0)(f)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(np.array([True, False, True]))  # 1-D, not 2-D
    msg = str(exc.value)
    assert "universe_causal argument 'panel' must be a 2-D ndarray, got ndarray" in msg


def test_universe_causal_rejects_non_ndarray_output():
    """Output-contract raise (guards.py:460-464)."""

    def f(panel):
        return [True, False, True]  # not an ndarray

    guarded = guards.universe_causal(panel_arg=0)(f)
    panel = np.ones((3, 2), dtype=bool)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(panel)
    msg = str(exc.value)
    assert "universe_causal output must be an ndarray with shape[0] == 3" in msg
    assert "got list" in msg


def test_universe_causal_noop_for_single_session_panel():
    """`n_sessions < 2` early return (guards.py:466-468), proved by OBSERVATION
    (a call counter), not by reading the source: with one session there is
    nothing to probe, so the wrapped function must be called exactly once."""
    calls = {"n": 0}

    def eligibility(panel):
        calls["n"] += 1
        present = panel if panel.dtype == np.bool_ else np.isfinite(panel)
        return present.any(axis=1)

    guarded = guards.universe_causal(panel_arg=0)(eligibility)
    panel = np.array([[True, False, True]])
    with guards.strictness(guards.Strictness.FULL):
        out = guarded(panel)
    np.testing.assert_array_equal(out, np.array([True]))
    assert calls["n"] == 1


def test_universe_causal_add_direction_shape_mismatch_raises():
    """Add-direction probe shape-contract raise (guards.py:507-511). `n_probes=1`
    with a 2-session panel makes `probe_set == {0, 1}` already, so the top-up
    sampling branch is skipped too (guards.py:488->493, budget <= 0)."""
    calls = {"n": 0}

    def flaky(panel):
        calls["n"] += 1
        if calls["n"] == 1:
            return np.ones(panel.shape[0], dtype=bool)  # baseline: correct shape
        return np.ones(panel.shape[0] - 1, dtype=bool)  # every probe: wrong shape

    guarded = guards.universe_causal(panel_arg=0, n_probes=1)(flaky)
    panel = np.ones((2, 2), dtype=bool)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(panel)
    msg = str(exc.value)
    assert "universe_causal add-direction probe at session 0 returned shape" in msg
    assert "baseline was" in msg


def test_universe_causal_remove_direction_shape_mismatch_raises():
    """Remove-direction probe shape-contract raise (guards.py:537-541): the
    add-direction probe must pass first (matching baseline exactly), so only the
    remove-direction call is made to return a bad shape."""
    calls = {"n": 0}

    def flaky(panel):
        calls["n"] += 1
        if calls["n"] <= 2:  # baseline, then the add-direction probe
            return np.ones(panel.shape[0], dtype=bool)
        return np.ones(panel.shape[0] - 1, dtype=bool)  # remove-direction: wrong shape

    guarded = guards.universe_causal(panel_arg=0, n_probes=1)(flaky)
    panel = np.ones((2, 2), dtype=bool)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(panel)
    msg = str(exc.value)
    assert "universe_causal remove-direction probe at session 0 returned shape" in msg
    assert "baseline was" in msg


# ---------------------------------------------------------------------------
# `execution_causal` -- decoration-time argument wiring
# ---------------------------------------------------------------------------


def test_execution_causal_rejects_conflicting_row_arg_and_row_args():
    """`ValueError` for conflicting arguments (guards.py:598-601), raised at
    decoration time. Assert the exact message, since that is what tells a caller
    which argument was wrong."""

    def f(x):
        return x

    with pytest.raises(ValueError) as exc:
        guards.execution_causal(row_arg=0, row_args=("x",))(f)
    assert str(exc.value) == "execution_causal: specify exactly one of row_arg or row_args"


def test_execution_causal_default_convention_when_neither_arg_given():
    """When NEITHER `row_arg` nor `row_args` is given, `effective_row_arg`
    defaults to `0` (guards.py:602-604) -- the single-ndarray, positional-arg-0
    convention. Proved both ways: a causally-correct function passes cleanly,
    and a genuinely leaky one is caught, so this isn't just "didn't crash"."""

    def correct_fill(x):
        # x columns: [orders, price, bar_traded_value, tradable] -- participation
        # cap uses only row t's own inputs.
        cap = np.minimum(x[:, 0], 0.1 * x[:, 2])
        return cap * x[:, 3]

    def leaky_fill(x):
        shifted_btv = np.empty(x.shape[0])
        shifted_btv[:-1] = x[1:, 2]
        shifted_btv[-1] = x[-1, 2]
        return x[:, 0] + shifted_btv  # reads tomorrow's bar_traded_value

    x = np.array(
        [
            [10.0, 100.0, 500.0, 1.0],
            [12.0, 101.0, 480.0, 1.0],
            [8.0, 102.0, 510.0, 1.0],
            [9.0, 103.0, 495.0, 1.0],
        ]
    )
    guarded_correct = guards.execution_causal()(correct_fill)
    guarded_leaky = guards.execution_causal()(leaky_fill)

    with guards.strictness(guards.Strictness.FULL):
        out = guarded_correct(x)
        np.testing.assert_array_equal(out, correct_fill(x))

        with pytest.raises(guards.ContractViolation) as exc:
            guarded_leaky(x)
    msg = str(exc.value)
    assert "execution_causal violation: fill quantity/price/charges at row 0 changed" in msg


# ---------------------------------------------------------------------------
# `execution_causal` -- single-ndarray (`row_arg`) wrapper
# ---------------------------------------------------------------------------


def test_execution_causal_row_arg_full_strictness_via_env_fallback(monkeypatch):
    """`get_strictness()` fallback for the `row_arg` wrapper (guards.py:614-616),
    proved by observation with the module cache reset to None."""

    def leaky(x):
        shifted = np.empty(x.shape[0])
        shifted[:-1] = x[1:, 0]
        shifted[-1] = x[-1, 0]
        return x[:, 0] + shifted

    guarded = guards.execution_causal(row_arg=0)(leaky)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")
    with pytest.raises(guards.ContractViolation):
        guarded(x)


def test_execution_causal_row_arg_rejects_non_2d_argument():
    """Argument-validation raise (guards.py:621-625)."""

    def f(x):
        return np.zeros(x.shape[0])

    guarded = guards.execution_causal(row_arg=0)(f)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(np.array([1.0, 2.0, 3.0]))  # 1-D
    msg = str(exc.value)
    assert "execution_causal argument 'x' must be a 2-D ndarray, got ndarray" in msg


def test_execution_causal_row_arg_rejects_bad_output_shape():
    """Output-contract raise (guards.py:628-633)."""

    def f(x):
        return np.zeros(x.shape[0] - 1)

    guarded = guards.execution_causal(row_arg=0)(f)
    x = np.ones((4, 2))
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(x)
    msg = str(exc.value)
    assert "execution_causal output must be an ndarray with shape[0] == 4" in msg


def test_execution_causal_row_arg_noop_for_single_row():
    """`n_rows < 2` early return (guards.py:635-637), proved by call count."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x[:, 0] * 2

    guarded = guards.execution_causal(row_arg=0)(f)
    x = np.array([[5.0, 6.0]])
    with guards.strictness(guards.Strictness.FULL):
        out = guarded(x)
    np.testing.assert_array_equal(out, np.array([10.0]))
    assert calls["n"] == 1


def test_execution_causal_row_arg_probe_shape_mismatch_raises():
    """Probe shape-contract raise (guards.py:658-663). `n_probes=1` with 3 rows
    makes `budget == 0`, so the top-up sampling branch is also skipped
    (guards.py:643->648) and `probe_set == {0}` deterministically."""
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        n = x.shape[0]
        if calls["n"] == 1:
            return np.zeros(n)
        return np.zeros(n - 1)

    guarded = guards.execution_causal(row_arg=0, n_probes=1)(flaky)
    x = np.ones((3, 2))
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(x)
    msg = str(exc.value)
    assert "execution_causal probe at row 0 returned shape" in msg
    assert "baseline was" in msg


# ---------------------------------------------------------------------------
# `execution_causal` -- multi-argument (`row_args`) wrapper
# ---------------------------------------------------------------------------


def test_execution_causal_row_args_rejects_non_1d_entry():
    """Argument-validation raise for a non-1-D `row_args` entry (guards.py:
    692-697); message must name the specific argument that was wrong."""

    def f(prices, bar_traded_value):
        return prices  # never reached

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(f)
    prices = np.ones((3, 2))  # 2-D, invalid for a row_args entry
    btv = np.ones(3)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(prices, btv)
    msg = str(exc.value)
    assert "execution_causal row_args entry 'prices' must be a 1-D ndarray" in msg


def test_execution_causal_row_args_rejects_mismatched_lengths():
    """Argument-validation raise for mismatched `row_args` lengths (guards.py:
    700-705); message must name both lengths and the offending argument."""

    def f(prices, bar_traded_value):
        return prices  # never reached

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(f)
    prices = np.ones(4)
    btv = np.ones(3)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(prices, btv)
    msg = str(exc.value)
    assert "execution_causal row_args entries must share one length, got 4 and 3" in msg
    assert "'bar_traded_value'" in msg


def test_execution_causal_row_args_noop_when_fewer_than_two_rows():
    """`n_rows < 2` early return (guards.py:708-709), normal case: the length
    comes from real 1-element arrays."""
    calls = {"n": 0}

    def fill(prices, bar_traded_value):
        calls["n"] += 1
        return prices * 0 + bar_traded_value

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(fill)
    prices = np.array([100.0])
    btv = np.array([500.0])
    with guards.strictness(guards.Strictness.FULL):
        out = guarded(prices, btv)
    np.testing.assert_array_equal(out, np.array([500.0]))
    assert calls["n"] == 1


def test_execution_causal_empty_row_args_raises_at_decoration():
    """F13, fixed: `row_args=()` used to make `static_names == []`, so the
    validation loop (guards.py:690-706) never ran, `n_rows` was never assigned
    (stayed `None`), and the `n_rows is None` arm of the early return at
    guards.py:708-709 called straight through to `func` with ZERO validation --
    at ANY strictness including FULL, a deliberately leaky function (row t
    reading row t+1) passed completely undetected.

    Fixed at DECORATION time: an empty container is a caller error ("nothing to
    perturb"), not "nothing to do" -- the same family as F12's `if col_idx:`
    treating an empty list like `None`. This test now pins the fix: applying
    the decorator to a leaky function must raise `ValueError` immediately, the
    function must never even be called, and the message must name `row_args`.
    The leaky fixture is kept because it's what makes the stakes of "silently
    disabled" concrete -- a bare `pytest.raises` wouldn't explain why this
    matters."""
    calls = {"n": 0}

    def leaky_if_it_ever_ran(prices, bar_traded_value):
        calls["n"] += 1
        shifted = np.empty_like(bar_traded_value)
        shifted[:-1] = bar_traded_value[1:]
        shifted[-1] = bar_traded_value[-1]
        return prices + shifted

    with pytest.raises(ValueError) as exc:
        guards.execution_causal(row_args=())(leaky_if_it_ever_ran)
    assert "row_args" in str(exc.value)
    assert calls["n"] == 0  # never reached -- the raise happens at decoration time


def test_execution_causal_row_args_full_strictness_via_env_fallback(monkeypatch):
    """`get_strictness()` fallback for the `row_args` wrapper (guards.py:
    682-684), proved by observation with the module cache reset to None."""

    def leaky(prices, bar_traded_value):
        shifted = np.empty_like(bar_traded_value)
        shifted[:-1] = bar_traded_value[1:]
        shifted[-1] = bar_traded_value[-1]
        return prices + shifted

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(leaky)
    prices = np.array([100.0, 101.0, 102.0])
    btv = np.array([500.0, 10.0, 700.0])

    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")
    with pytest.raises(guards.ContractViolation):
        guarded(prices, btv)


def test_execution_causal_row_args_rejects_bad_baseline_output():
    """Output-contract raise for the `row_args` wrapper's baseline call
    (guards.py:712-716)."""

    def f(prices, bar_traded_value):
        return list(prices)  # not an ndarray

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(f)
    prices = np.array([1.0, 2.0, 3.0])
    btv = np.array([1.0, 2.0, 3.0])
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(prices, btv)
    msg = str(exc.value)
    assert "execution_causal output must be an ndarray with shape[0] == 3" in msg


def test_execution_causal_row_args_handles_degenerate_scale_column():
    """Degenerate-scale branch (guards.py:742): when a `row_args` column has
    fewer than 2 finite-positive entries (here, `bar_traded_value` is all zero,
    so 0), the log-normal jitter falls back to `scale=1.0, sigma=0.01` rather
    than dividing by/logging a near-zero median. The perturbation this produces
    is still large enough relative to the degenerate baseline (0 -> ~1) to catch
    a real leak that reads the next row's `bar_traded_value` -- proving the
    fallback constants are not just non-crashing but actually functional."""

    def leaky_uses_next_btv(prices, bar_traded_value):
        shifted = np.empty_like(bar_traded_value)
        shifted[:-1] = bar_traded_value[1:]
        shifted[-1] = bar_traded_value[-1]
        return prices + shifted

    guarded = guards.execution_causal(row_args=("prices", "bar_traded_value"))(
        leaky_uses_next_btv
    )
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    bar_traded_value = np.zeros(5)  # zero finite-positive entries -> degenerate scale
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(prices, bar_traded_value)
    msg = str(exc.value)
    assert "execution_causal violation: fill quantity/price/charges at row 0 changed" in msg


def test_execution_causal_row_args_probe_shape_mismatch_raises():
    """Probe shape-contract raise for the `row_args` wrapper (guards.py:
    769-774). `n_probes=1` with 3 rows makes `budget == 0`, so the top-up
    sampling branch is also skipped (guards.py:722->727)."""
    calls = {"n": 0}

    def flaky(prices, bar_traded_value):
        calls["n"] += 1
        n = prices.shape[0]
        if calls["n"] == 1:
            return np.zeros(n)
        return np.zeros(n - 1)

    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"), n_probes=1
    )(flaky)
    prices = np.full(3, 100.0)
    btv = np.full(3, 1000.0)
    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc:
            guarded(prices, btv)
    msg = str(exc.value)
    assert "execution_causal probe at row 0 returned shape" in msg
    assert "baseline was" in msg
