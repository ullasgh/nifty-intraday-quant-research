from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from nifty_quant.strategy.base import Strategy
from nifty_quant.strategy.registry import (
    _REGISTRY,
    _to_serializable,
    available,
    build,
    config_hash,
    get,
    register,
)


class _DummyParams(BaseModel):
    threshold: float = 1.0


class _DummyStrategy(Strategy):
    name = "__coverage_test_strategy_a__"
    Params = _DummyParams

    def data_request(self) -> Any:
        return None

    def precompute(self, panel: Any) -> dict[str, Any]:
        return {}

    def on_decision(self, view: Any, signals: Any, state: Any) -> Any:
        return None


class _DummyStrategyAA(_DummyStrategy):
    name = "__coverage_test_strategy_aa__"


class _DummyStrategyZZ(_DummyStrategy):
    name = "__coverage_test_strategy_zz__"


class _DummyStrategyCollide(_DummyStrategy):
    name = "__coverage_test_strategy_a__"


class _FreshDummyStrategy(_DummyStrategy):
    name = "__coverage_test_strategy_fresh__"


class _FreshDummyStrategy2(_DummyStrategy):
    name = "__coverage_test_strategy_fresh_2__"


class _UnsupportedCustom:
    pass


@pytest.fixture(autouse=True)
def _register_dummy_strategies() -> Iterator[None]:
    before = set(_REGISTRY)
    register(_DummyStrategy)
    register(_DummyStrategyAA)
    register(_DummyStrategyZZ)
    yield
    for key in set(_REGISTRY) - before:
        del _REGISTRY[key]


def test_register_fresh_name_succeeds_and_is_retrievable() -> None:
    cls = _FreshDummyStrategy
    register(cls)
    assert get(cls.name) is cls
    assert cls.name in available()


def test_register_duplicate_name_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Strategy name '__coverage_test_strategy_a__' already registered"),
    ):
        register(_DummyStrategyCollide)


def test_register_returns_class_unchanged() -> None:
    cls = _FreshDummyStrategy2
    assert register(cls) is cls


def test_get_unknown_strategy_key_error_and_cause_chaining() -> None:
    name = "some_never_registered_name_zzz"
    with pytest.raises(KeyError) as exc_info:
        get(name)
    assert type(exc_info.value) is KeyError
    assert exc_info.value.__cause__ is None
    assert str(exc_info.value) == repr(f"Unknown strategy: {name}")


def test_get_registered_name_returns_class_identity() -> None:
    assert get(_DummyStrategy.name) is _DummyStrategy


def test_available_returns_sorted_list_with_registered_names() -> None:
    names = available()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert names == sorted(names)
    assert "__coverage_test_strategy_aa__" in names
    assert "__coverage_test_strategy_zz__" in names
    assert names.index("__coverage_test_strategy_aa__") < names.index(
        "__coverage_test_strategy_zz__"
    )


def test_build_missing_strategy_key_plain_key_error() -> None:
    with pytest.raises(KeyError):
        build({"params": {}})


@pytest.mark.parametrize("bad", [1, None])
def test_build_strategy_key_not_string_type_error(bad: Any) -> None:
    with pytest.raises(TypeError, match=re.escape("'strategy' key must be a string")):
        build({"strategy": bad, "params": {}})


def test_build_unknown_strategy_propagates_key_error() -> None:
    name = "some_never_registered_name_build_zzz"
    with pytest.raises(KeyError, match=re.escape(repr(f"Unknown strategy: {name}"))):
        build({"strategy": name, "params": {}})


def test_build_valid_config_returns_instance_with_params() -> None:
    cfg = {"strategy": "__coverage_test_strategy_a__", "params": {"threshold": 2.5}}
    strategy = build(cfg)
    assert isinstance(strategy, _DummyStrategy)
    assert strategy.params.threshold == 2.5


def test_build_missing_params_key_propagates_plain_key_error() -> None:
    with pytest.raises(KeyError):
        build({"strategy": "__coverage_test_strategy_a__"})


def test_build_params_not_mapping_type_error() -> None:
    with pytest.raises(TypeError, match=re.escape("'params' key must be a mapping")):
        build({"strategy": "__coverage_test_strategy_a__", "params": [1, 2]})


def test_to_serializable_dict_recursively_converts() -> None:
    value = {
        "a": 1,
        "b": {"c": [2, 3]},
        "d": (4, 5),
    }
    assert _to_serializable(value) == {
        "a": 1,
        "b": {"c": [2, 3]},
        "d": [4, 5],
    }


def test_to_serializable_list_and_tuple_same_result() -> None:
    assert _to_serializable([1, 2, 3]) == [1, 2, 3]
    assert _to_serializable((1, 2, 3)) == [1, 2, 3]


def test_to_serializable_set_deterministic_order() -> None:
    assert _to_serializable({3, 1, 2}) == [1, 2, 3]
    assert _to_serializable({2, 3, 1}) == [1, 2, 3]
    assert _to_serializable({"b", "a", "c"}) == ["a", "b", "c"]


def test_to_serializable_primitives_pass_through() -> None:
    for value in ("x", 1, 1.5, True, None):
        assert _to_serializable(value) == value


def test_to_serializable_unsupported_types_raise_type_error() -> None:
    with pytest.raises(TypeError, match=re.escape("Unsupported config value type: bytes")):
        _to_serializable(b"abc")
    with pytest.raises(
        TypeError,
        match=re.escape("Unsupported config value type: _UnsupportedCustom"),
    ):
        _to_serializable(_UnsupportedCustom())


def test_to_serializable_nested_unsupported_raises() -> None:
    with pytest.raises(TypeError, match=re.escape("Unsupported config value type: bytes")):
        _to_serializable({"a": [1, (2, b"x")]})


def test_config_hash_deterministic() -> None:
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"a": 1, "b": 2}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_key_order_independent() -> None:
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"b": 2, "a": 1}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_list_and_tuple_equivalent() -> None:
    assert config_hash({"x": [1, 2, 3]}) == config_hash({"x": (1, 2, 3)})


def test_config_hash_different_values_produce_different_hashes() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_format_is_16_lowercase_hex() -> None:
    digest = config_hash({"a": 1})
    assert isinstance(digest, str)
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_hash_does_not_mutate_input() -> None:
    original = {"a": [3, 2, 1], "b": {"c": (1, 2)}}
    expected = {"a": [3, 2, 1], "b": {"c": (1, 2)}}
    config_hash(original)
    assert original == expected


def test_config_hash_set_value_end_to_end() -> None:
    digest = config_hash({"x": {3, 1, 2}})
    assert isinstance(digest, str)
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)
    assert config_hash({"x": {3, 1, 2}}) == config_hash({"x": {2, 3, 1}})
