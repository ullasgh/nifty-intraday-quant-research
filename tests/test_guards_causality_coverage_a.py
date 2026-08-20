"""Coverage-gap closure for `universe_causal`/`execution_causal`/`_source_snippet`.

Targets the specific lines/branches left uncovered when Phase B added these two
decorators (see `guards.py` around lines 396-785). Each test asserts real observed
behaviour (an exception's type/message content, a returned value, or a call count)
rather than merely executing a line, per CLAUDE.md rule 1.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from nifty_quant import guards


def _assert_violation_message(
    exc_info: pytest.ExceptionInfo,
    function_name: str,
    index: int | None = None,
    *terms: str,
) -> None:
    message = str(exc_info.value)
    assert function_name in message
    if index is not None:
        assert re.search(
            rf"\b(?:row|session)\D*{index}\b",
            message,
            flags=re.IGNORECASE,
        )
    for term in terms:
        assert term.lower() in message.lower()


# --- Item 1: _source_snippet source-recovery failures and success


def test_source_snippet_handles_builtin_exec_function_and_local_function():
    assert guards._source_snippet(len) == ""

    namespace: dict[str, object] = {}
    exec(
        "def generated_without_source(value):\n"
        "    return value + 1\n",
        namespace,
    )
    generated = namespace["generated_without_source"]
    assert callable(generated)
    assert guards._source_snippet(generated) == ""

    def ordinary_local_function(value):
        return value

    snippet = guards._source_snippet(ordinary_local_function)
    assert snippet
    assert snippet.startswith("\nsource: ")


# --- Item 2: strictness fallback through get_strictness for all conventions


def test_universe_causal_uses_environment_strictness_fallback(monkeypatch):
    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")

    def fallback_universe_leak(panel):
        whole_universe = np.any(panel, axis=0)
        return np.tile(whole_universe, (panel.shape[0], 1))

    panel = np.array(
        [[True, False], [True, False], [True, False]],
        dtype=bool,
    )
    guarded = guards.universe_causal()(fallback_universe_leak)

    with pytest.raises(guards.ContractViolation) as exc_info:
        guarded(panel)

    _assert_violation_message(
        exc_info,
        "fallback_universe_leak",
        0,
        "eligibility mask",
    )


def test_execution_causal_row_arg_uses_environment_strictness_fallback(monkeypatch):
    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")

    def fallback_row_arg_leak(bars):
        value = 1.0 if bars[1, 2] == 8.0 else 0.0
        return np.full(bars.shape[0], value)

    bars = np.array(
        [[1.0, 100.0, 7.0], [1.0, 100.0, 8.0], [1.0, 100.0, 9.0]],
        dtype=np.float64,
    )
    guarded = guards.execution_causal(row_arg=0)(fallback_row_arg_leak)

    with pytest.raises(guards.ContractViolation) as exc_info:
        guarded(bars)

    _assert_violation_message(
        exc_info,
        "fallback_row_arg_leak",
        0,
        "quantity",
    )


def test_execution_causal_row_args_uses_environment_strictness_fallback(monkeypatch):
    monkeypatch.setattr(guards, "_strictness", None)
    monkeypatch.setenv("NQ_STRICT", "2")

    def fallback_row_args_leak(prices, bar_traded_value):
        value = 1.0 if prices[1] == 20.0 else 0.0
        return np.full(prices.shape[0], value)

    prices = np.array([10.0, 20.0, 30.0])
    bar_traded_value = np.array([100.0, 200.0, 300.0])
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(fallback_row_args_leak)

    with pytest.raises(guards.ContractViolation) as exc_info:
        guarded(prices, bar_traded_value)

    _assert_violation_message(
        exc_info,
        "fallback_row_args_leak",
        0,
        "prices",
    )


# --- Item 3: universe_causal panel argument type and dimensionality


def test_universe_causal_rejects_non_2d_panel_argument():
    def invalid_universe_panel(panel):
        return np.zeros((1, 1), dtype=bool)

    guarded = guards.universe_causal()(invalid_universe_panel)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(np.array([True, False], dtype=bool))

    message = str(exc_info.value)
    assert "invalid_universe_panel" in message
    assert "panel" in message.lower()
    assert "2-d ndarray" in message.lower()


# --- Item 4: universe_causal baseline output shape validation


def test_universe_causal_rejects_wrong_baseline_output_shape():
    def invalid_universe_output(panel):
        return [True]

    panel = np.ones((2, 1), dtype=bool)
    guarded = guards.universe_causal()(invalid_universe_output)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(panel)

    message = str(exc_info.value)
    assert "invalid_universe_output" in message
    assert "shape[0]" in message
    assert "2" in message


# --- Item 5: universe_causal one-session early return


def test_universe_causal_returns_one_session_result_without_probing():
    calls: list[int] = []

    def one_session_universe(panel):
        calls.append(1)
        return np.tile(np.any(panel, axis=0), (panel.shape[0], 1))

    panel = np.array([[True, False]], dtype=bool)
    guarded = guards.universe_causal()(one_session_universe)

    expected = one_session_universe(panel)
    calls.clear()

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(panel)

    assert np.array_equal(result, expected)
    assert len(calls) == 1


# --- Item 6: universe_causal sampling top-up skipped


def test_universe_causal_still_probes_when_sampling_top_up_is_skipped():
    def all_present_universe_leak(panel):
        whole_universe = np.all(panel, axis=0)
        return np.tile(whole_universe, (panel.shape[0], 1))

    panel = np.ones((3, 2), dtype=bool)
    guarded = guards.universe_causal(n_probes=1)(all_present_universe_leak)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(panel)

    _assert_violation_message(
        exc_info,
        "all_present_universe_leak",
        0,
        "eligibility mask",
    )


# --- Item 7: universe_causal add-direction probe shape mismatch


def test_universe_causal_rejects_add_probe_shape_mismatch():
    def universe_add_shape_change(panel):
        if np.isnan(panel).any():
            return np.ones(panel.shape[0])
        return np.ones(panel.shape[0] - 1)

    panel = np.array([[1.0], [np.nan]], dtype=np.float64)
    guarded = guards.universe_causal()(universe_add_shape_change)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(panel)

    _assert_violation_message(
        exc_info,
        "universe_add_shape_change",
        0,
        "add-direction",
        "shape",
    )


# --- Item 8: universe_causal remove-direction probe shape mismatch


def test_universe_causal_rejects_remove_probe_shape_mismatch():
    def universe_remove_shape_change(panel):
        if np.any(~panel):
            return np.ones(panel.shape[0] - 1)
        return np.ones(panel.shape[0])

    panel = np.ones((2, 1), dtype=bool)
    guarded = guards.universe_causal()(universe_remove_shape_change)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(panel)

    _assert_violation_message(
        exc_info,
        "universe_remove_shape_change",
        0,
        "remove-direction",
        "shape",
    )


# --- Item 9: execution_causal mutually exclusive row arguments


def test_execution_causal_rejects_both_row_argument_conventions():
    def both_row_conventions(bars, other):
        return bars, other

    with pytest.raises(ValueError) as exc_info:
        guards.execution_causal(
            row_arg=0,
            row_args=("bars", "other"),
        )(both_row_conventions)

    message = str(exc_info.value).lower()
    assert "exactly one" in message
    assert "row_arg" in message
    assert "row_args" in message


# --- Item 10: execution_causal default row_arg path


def test_execution_causal_defaults_to_first_positional_row_argument():
    def default_execution_leak(bars):
        value = 1.0 if bars[1, 2] == 8.0 else 0.0
        return np.full(bars.shape[0], value)

    bars = np.array(
        [[1.0, 100.0, 7.0], [1.0, 100.0, 8.0], [1.0, 100.0, 9.0]],
        dtype=np.float64,
    )
    guarded = guards.execution_causal()(default_execution_leak)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(bars)

    _assert_violation_message(
        exc_info,
        "default_execution_leak",
        0,
        "quantity",
    )


# --- Item 11: execution_causal row_arg dimensionality check


def test_execution_causal_row_arg_requires_2d_ndarray():
    def row_arg_not_2d(bars):
        return np.zeros(bars.shape[0])

    guarded = guards.execution_causal(row_arg=0)(row_arg_not_2d)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(np.ones(3, dtype=np.float64))

    message = str(exc_info.value)
    assert "row_arg_not_2d" in message
    assert "bars" in message.lower()
    assert "2-d ndarray" in message.lower()


# --- Item 12: execution_causal row_arg baseline output shape


def test_execution_causal_row_arg_rejects_wrong_baseline_output_shape():
    def invalid_execution_output(bars):
        return np.ones(2)

    bars = np.ones((3, 3), dtype=np.float64)
    guarded = guards.execution_causal(row_arg=0)(invalid_execution_output)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(bars)

    message = str(exc_info.value)
    assert "invalid_execution_output" in message
    assert "shape[0]" in message
    assert "3" in message


# --- Item 13: execution_causal row_arg one-row early return


def test_execution_causal_row_arg_returns_one_row_result_without_probing():
    calls: list[int] = []

    def one_row_execution(bars):
        calls.append(1)
        return bars[:, :1]

    bars = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
    guarded = guards.execution_causal(row_arg=0)(one_row_execution)

    expected = one_row_execution(bars)
    calls.clear()

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(bars)

    assert np.array_equal(result, expected)
    assert len(calls) == 1


# --- Item 14: execution_causal row_arg sampling top-up skipped


def test_execution_causal_row_arg_still_probes_two_rows_without_top_up():
    def two_row_execution_leak(bars):
        value = 1.0 if bars[1, 2] == 9.0 else 0.0
        return np.array([value, 0.0])

    bars = np.array(
        [[1.0, 100.0, 8.0], [1.0, 100.0, 9.0]],
        dtype=np.float64,
    )
    guarded = guards.execution_causal(row_arg=0)(two_row_execution_leak)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(bars)

    _assert_violation_message(
        exc_info,
        "two_row_execution_leak",
        0,
        "quantity",
    )


# --- Item 15: execution_causal row_arg probe shape mismatch


def test_execution_causal_row_arg_rejects_probe_shape_mismatch():
    def row_arg_probe_shape_change(bars):
        if np.isnan(bars).any():
            return np.ones(bars.shape[0])
        return np.ones(bars.shape[0] - 1)

    bars = np.array([[1.0], [np.nan]], dtype=np.float64)
    guarded = guards.execution_causal(row_arg=0)(row_arg_probe_shape_change)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(bars)

    _assert_violation_message(
        exc_info,
        "row_arg_probe_shape_change",
        0,
        "shape",
    )


# --- Item 16: row_args strictness fallback (covered by Item 2)


# --- Item 17: execution_causal row_args one-dimensional entry check


def test_execution_causal_row_args_requires_one_dimensional_entries():
    def row_args_not_1d(prices, bar_traded_value):
        return prices + bar_traded_value

    prices = np.ones((2, 2), dtype=np.float64)
    bar_traded_value = np.ones(2, dtype=np.float64)
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(row_args_not_1d)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    message = str(exc_info.value)
    assert "row_args_not_1d" in message
    assert "prices" in message
    assert "1-d ndarray" in message.lower()


# --- Item 18: execution_causal row_args mismatched lengths


def test_execution_causal_row_args_reject_mismatched_lengths():
    def row_args_length_mismatch(prices, bar_traded_value):
        return prices

    prices = np.ones(5, dtype=np.float64)
    bar_traded_value = np.ones(4, dtype=np.float64)
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(row_args_length_mismatch)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    message = str(exc_info.value)
    assert "row_args_length_mismatch" in message
    assert "5" in message
    assert "4" in message
    assert "bar_traded_value" in message


# --- Item 19: execution_causal row_args one-row early return


def test_execution_causal_row_args_returns_one_row_result_without_y0_probe():
    calls: list[int] = []

    def one_row_row_args(prices, bar_traded_value):
        calls.append(1)
        return [float(prices[0] + bar_traded_value[0])]

    prices = np.array([3.0])
    bar_traded_value = np.array([2.0])
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(one_row_row_args)

    expected = one_row_row_args(prices, bar_traded_value)
    calls.clear()

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(prices, bar_traded_value)

    assert result == expected
    assert len(calls) == 1


# --- Item 20: execution_causal row_args baseline output shape


def test_execution_causal_row_args_rejects_wrong_baseline_output_shape():
    def invalid_row_args_output(prices, bar_traded_value):
        return np.ones((1, 1))

    prices = np.array([10.0, 20.0])
    bar_traded_value = np.array([100.0, 200.0])
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(invalid_row_args_output)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    message = str(exc_info.value)
    assert "invalid_row_args_output" in message
    assert "shape[0]" in message
    assert "2" in message


# --- Item 21: execution_causal row_args sampling top-up skipped


def test_execution_causal_row_args_still_probes_two_rows_without_top_up():
    def two_row_row_args_leak(prices, bar_traded_value):
        value = 1.0 if prices[1] == 9.0 else 0.0
        return np.array([value, 0.0])

    prices = np.array([1.0, 9.0])
    bar_traded_value = np.array([2.0, 3.0])
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(two_row_row_args_leak)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    _assert_violation_message(
        exc_info,
        "two_row_row_args_leak",
        0,
        "prices",
    )


# --- Item 22: row_args sigma fallback for degenerate non-positive data


def test_execution_causal_row_args_zero_values_remain_causal():
    def current_row_only(prices, bar_traded_value):
        return prices + bar_traded_value

    prices = np.array([10.0, 20.0, 30.0])
    bar_traded_value = np.zeros(3)
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(current_row_only)

    expected = current_row_only(prices, bar_traded_value)

    with guards.strictness(guards.Strictness.FULL):
        result = guarded(prices, bar_traded_value)

    assert np.array_equal(result, expected)


def test_execution_causal_row_args_zero_values_still_detect_future_leak():
    def degenerate_row_args_leak(prices, bar_traded_value):
        return np.full(prices.shape[0], bar_traded_value[1])

    prices = np.array([10.0, 20.0])
    bar_traded_value = np.zeros(2)
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(degenerate_row_args_leak)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    _assert_violation_message(
        exc_info,
        "degenerate_row_args_leak",
        0,
        "bar_traded_value",
    )


# --- Item 23: execution_causal row_args probe shape mismatch


def test_execution_causal_row_args_rejects_probe_shape_mismatch():
    def row_args_probe_shape_change(prices, bar_traded_value):
        if np.isnan(prices).any():
            return np.ones(prices.shape[0])
        return np.ones(prices.shape[0] - 1)

    prices = np.array([10.0, np.nan])
    bar_traded_value = np.array([1.0, 2.0])
    guarded = guards.execution_causal(
        row_args=("prices", "bar_traded_value"),
    )(row_args_probe_shape_change)

    with guards.strictness(guards.Strictness.FULL):
        with pytest.raises(guards.ContractViolation) as exc_info:
            guarded(prices, bar_traded_value)

    _assert_violation_message(
        exc_info,
        "row_args_probe_shape_change",
        0,
        "shape",
    )
