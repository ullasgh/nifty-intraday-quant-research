"""100% coverage tests for nifty_quant.research.sweep module."""

from __future__ import annotations

from pathlib import Path

import pytest

from nifty_quant.research.sweep import (
    _evaluate_constraint,
    _parse_literal,
    canonical_json,
    config_hash,
    expand,
    load_sweep_yaml,
)


class TestParseLiteral:
    """Tests for _parse_literal() token parser."""

    def test_parse_literal_integer_positive(self) -> None:
        result = _parse_literal("42")
        assert result == 42
        assert isinstance(result, int)

    def test_parse_literal_integer_negative(self) -> None:
        result = _parse_literal("-42")
        assert result == -42
        assert isinstance(result, int)

    def test_parse_literal_integer_zero(self) -> None:
        result = _parse_literal("0")
        assert result == 0

    def test_parse_literal_float_with_decimal(self) -> None:
        result = _parse_literal("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_literal_float_negative(self) -> None:
        result = _parse_literal("-3.14")
        assert result == -3.14

    def test_parse_literal_float_with_exponent(self) -> None:
        result = _parse_literal("1e-2")
        assert result == 0.01

    def test_parse_literal_float_with_positive_exponent(self) -> None:
        result = _parse_literal("1e2")
        assert result == 100.0

    def test_parse_literal_float_with_explicit_positive_exponent(self) -> None:
        result = _parse_literal("1e+2")
        assert result == 100.0

    def test_parse_literal_float_with_decimal_and_exponent(self) -> None:
        result = _parse_literal("2.5e-1")
        assert result == 0.25

    def test_parse_literal_string_single_quote(self) -> None:
        result = _parse_literal("'hello'")
        assert result == "hello"
        assert isinstance(result, str)

    def test_parse_literal_string_double_quote(self) -> None:
        result = _parse_literal('"hello"')
        assert result == "hello"

    def test_parse_literal_string_empty_single_quote(self) -> None:
        result = _parse_literal("''")
        assert result == ""

    def test_parse_literal_string_with_spaces(self) -> None:
        result = _parse_literal("'hello world'")
        assert result == "hello world"

    def test_parse_literal_identifier_lowercase(self) -> None:
        result = _parse_literal("x")
        assert result == "x"

    def test_parse_literal_identifier_uppercase(self) -> None:
        result = _parse_literal("ALPHA")
        assert result == "ALPHA"

    def test_parse_literal_identifier_with_underscore(self) -> None:
        result = _parse_literal("max_value")
        assert result == "max_value"

    def test_parse_literal_identifier_with_numbers(self) -> None:
        result = _parse_literal("param1")
        assert result == "param1"

    def test_parse_literal_identifier_starts_with_underscore(self) -> None:
        result = _parse_literal("_private")
        assert result == "_private"

    def test_parse_literal_invalid_quoted_mismatch(self) -> None:
        with pytest.raises(ValueError, match="invalid literal in constraint"):
            _parse_literal("'hello\"")

    def test_parse_literal_invalid_single_char_quoted(self) -> None:
        with pytest.raises(ValueError, match="invalid literal in constraint"):
            _parse_literal("'")

    def test_parse_literal_invalid_starts_with_number(self) -> None:
        with pytest.raises(ValueError, match="invalid literal in constraint"):
            _parse_literal("1param")

    def test_parse_literal_invalid_special_char(self) -> None:
        with pytest.raises(ValueError, match="invalid literal in constraint"):
            _parse_literal("@invalid")

    def test_parse_literal_invalid_hyphen_not_at_start(self) -> None:
        with pytest.raises(ValueError, match="invalid literal in constraint"):
            _parse_literal("3-5")

    def test_parse_literal_float_starting_with_decimal(self) -> None:
        result = _parse_literal("0.5")
        assert result == 0.5


class TestEvaluateConstraint:
    """Tests for _evaluate_constraint() expression evaluator."""

    def test_constraint_equality_integers(self) -> None:
        result = _evaluate_constraint("x == 5", {"x": 5})
        assert result is True

    def test_constraint_equality_integers_false(self) -> None:
        result = _evaluate_constraint("x == 5", {"x": 6})
        assert result is False

    def test_constraint_inequality_integers(self) -> None:
        result = _evaluate_constraint("x != 5", {"x": 6})
        assert result is True

    def test_constraint_inequality_integers_false(self) -> None:
        result = _evaluate_constraint("x != 5", {"x": 5})
        assert result is False

    def test_constraint_greater_than_integers(self) -> None:
        result = _evaluate_constraint("x > 5", {"x": 6})
        assert result is True

    def test_constraint_greater_than_integers_false(self) -> None:
        result = _evaluate_constraint("x > 5", {"x": 5})
        assert result is False

    def test_constraint_greater_equal_integers(self) -> None:
        result = _evaluate_constraint("x >= 5", {"x": 5})
        assert result is True

    def test_constraint_greater_equal_integers_false(self) -> None:
        result = _evaluate_constraint("x >= 5", {"x": 4})
        assert result is False

    def test_constraint_less_than_integers(self) -> None:
        result = _evaluate_constraint("x < 5", {"x": 4})
        assert result is True

    def test_constraint_less_than_integers_false(self) -> None:
        result = _evaluate_constraint("x < 5", {"x": 5})
        assert result is False

    def test_constraint_less_equal_integers(self) -> None:
        result = _evaluate_constraint("x <= 5", {"x": 5})
        assert result is True

    def test_constraint_less_equal_integers_false(self) -> None:
        result = _evaluate_constraint("x <= 5", {"x": 6})
        assert result is False

    def test_constraint_both_literals_integers(self) -> None:
        result = _evaluate_constraint("5 == 5", {})
        assert result is True

    def test_constraint_both_literals_integers_false(self) -> None:
        result = _evaluate_constraint("5 == 6", {})
        assert result is False

    def test_constraint_float_comparison(self) -> None:
        result = _evaluate_constraint("x > 1.5", {"x": 2.0})
        assert result is True

    def test_constraint_string_comparison(self) -> None:
        result = _evaluate_constraint("mode == 'aggressive'", {"mode": "aggressive"})
        assert result is True

    def test_constraint_string_comparison_false(self) -> None:
        result = _evaluate_constraint("mode == 'aggressive'", {"mode": "conservative"})
        assert result is False

    def test_constraint_mixed_param_and_literal(self) -> None:
        result = _evaluate_constraint("threshold > 0.5", {"threshold": 0.7})
        assert result is True

    def test_constraint_literal_on_left_param_on_right(self) -> None:
        result = _evaluate_constraint("5 < x", {"x": 6})
        assert result is True

    def test_constraint_whitespace_handling(self) -> None:
        result = _evaluate_constraint("   x   ==   5   ", {"x": 5})
        assert result is True

    def test_constraint_type_error_incomparable(self) -> None:
        with pytest.raises(ValueError, match="cannot compare .* with operator"):
            _evaluate_constraint("x > 'string'", {"x": 5})

    def test_constraint_invalid_format_no_operator(self) -> None:
        with pytest.raises(ValueError, match="invalid constraint"):
            _evaluate_constraint("x 5", {"x": 5})

    def test_constraint_invalid_format_extra_tokens(self) -> None:
        with pytest.raises(ValueError, match="invalid constraint"):
            _evaluate_constraint("x == 5 extra", {"x": 5})

    def test_constraint_missing_param_key(self) -> None:
        result = _evaluate_constraint("x == 5", {})
        assert result is False

    def test_constraint_both_identifiers_as_literals(self) -> None:
        result = _evaluate_constraint("x == y", {})
        assert result is False

    def test_constraint_equality_floats(self) -> None:
        result = _evaluate_constraint("x == 3.14", {"x": 3.14})
        assert result is True

    def test_constraint_malformed_operator_between_spaces(self) -> None:
        with pytest.raises(ValueError, match="invalid constraint"):
            _evaluate_constraint("x = = 5", {"x": 5})


class TestExpand:
    """Tests for expand() cartesian product with filtering."""

    def test_expand_single_param_single_value(self) -> None:
        result = expand({"base": 0}, {"x": [1]})
        assert len(result) == 1
        assert result[0] == {"base": 0, "x": 1}

    def test_expand_single_param_multiple_values(self) -> None:
        result = expand({"base": 0}, {"x": [1, 2, 3]})
        assert len(result) == 3
        assert all("base" in r and r["base"] == 0 for r in result)

    def test_expand_multiple_params_cartesian_product(self) -> None:
        result = expand({}, {"x": [1, 2], "y": [10, 20]})
        assert len(result) == 4
        expected_combos = [
            {"x": 1, "y": 10},
            {"x": 1, "y": 20},
            {"x": 2, "y": 10},
            {"x": 2, "y": 20},
        ]
        assert result == expected_combos

    def test_expand_base_params_preserved(self) -> None:
        base = {"alpha": 0.5, "beta": 0.3}
        result = expand(base, {"x": [1, 2]})
        assert len(result) == 2
        assert result[0]["alpha"] == 0.5
        assert result[0]["beta"] == 0.3

    def test_expand_base_params_overridden_by_sweep(self) -> None:
        base = {"x": 0}
        result = expand(base, {"x": [1, 2]})
        assert len(result) == 2
        assert result[0]["x"] == 1
        assert result[1]["x"] == 2

    def test_expand_with_constraint_filter(self) -> None:
        base = {}
        sweep = {"x": [1, 2, 3], "y": [2, 3, 4]}
        constraints = ["x < y"]
        result = expand(base, sweep, constraints)
        assert len(result) == 6
        assert all(r["x"] < r["y"] for r in result)

    def test_expand_with_constraint_all_filtered(self) -> None:
        base = {}
        sweep = {"x": [1, 2], "y": [1, 2]}
        constraints = ["x > y"]
        result = expand(base, sweep, constraints)
        assert len(result) == 1
        assert result[0] == {"x": 2, "y": 1}

    def test_expand_with_multiple_constraints(self) -> None:
        base = {}
        sweep = {"x": [1, 2, 3], "y": [1, 2, 3]}
        constraints = ["x < y", "y > 1"]
        result = expand(base, sweep, constraints)
        for r in result:
            assert r["x"] < r["y"]
            assert r["y"] > 1

    def test_expand_empty_sweep(self) -> None:
        result = expand({"base": 0}, {})
        assert len(result) == 1
        assert result[0] == {"base": 0}

    def test_expand_empty_base_params(self) -> None:
        result = expand({}, {"x": [1, 2]})
        assert len(result) == 2
        assert all("x" in r for r in result)

    def test_expand_three_params_cartesian(self) -> None:
        result = expand({}, {"a": [1], "b": [2], "c": [3, 4]})
        assert len(result) == 2
        assert result[0] == {"a": 1, "b": 2, "c": 3}
        assert result[1] == {"a": 1, "b": 2, "c": 4}

    def test_expand_constraint_with_string_values(self) -> None:
        result = expand({}, {"mode": ["fast", "slow"]}, ["mode == 'fast'"])
        assert len(result) == 1
        assert result[0]["mode"] == "fast"

    def test_expand_constraint_invalid_raises_error(self) -> None:
        with pytest.raises(ValueError):
            expand({}, {"x": [1, 2]}, ["malformed constraint"])

    def test_expand_preserves_order(self) -> None:
        result = expand({}, {"x": [3, 1, 2]})
        assert result[0]["x"] == 3
        assert result[1]["x"] == 1
        assert result[2]["x"] == 2

    def test_expand_float_sweep_values(self) -> None:
        result = expand({}, {"alpha": [0.1, 0.5, 0.9]})
        assert len(result) == 3
        assert result[0]["alpha"] == 0.1

    def test_expand_mixed_type_sweep_values(self) -> None:
        result = expand({}, {"x": [1, 2.5, "three"]})
        assert len(result) == 3
        assert result[0]["x"] == 1
        assert result[1]["x"] == 2.5
        assert result[2]["x"] == "three"


class TestLoadSweepYaml:
    """Tests for load_sweep_yaml() configuration loader."""

    def test_load_sweep_yaml_valid(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params:
  alpha: 0.5
  beta: 0.3
sweep:
  x: [1, 2]
  y: [10, 20]
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert strategy == "my_strategy"
        assert len(configs) == 4

    def test_load_sweep_yaml_with_constraints(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep:
  x: [1, 2, 3]
  y: [1, 2, 3]
constraints:
  - x < y
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert len(configs) == 3

    def test_load_sweep_yaml_missing_strategy_key(self, tmp_path: Path) -> None:
        yaml_content = """
base_params: {}
sweep: {}
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="missing required top-level keys"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_missing_base_params_key(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
sweep: {}
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="missing required top-level keys"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_missing_sweep_key(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="missing required top-level keys"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_all_keys_missing(self, tmp_path: Path) -> None:
        yaml_content = """
other: value
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="missing required top-level keys"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_not_a_dict(self, tmp_path: Path) -> None:
        yaml_content = """
- item1
- item2
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="sweep YAML must contain a top-level mapping"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_strategy_not_string(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: 123
base_params: {}
sweep: {}
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="strategy must be a string"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_base_params_not_dict(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: "not a dict"
sweep: {}
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="base_params must be a mapping"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_sweep_not_dict(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep: "not a dict"
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="sweep must be a mapping"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_constraints_not_list(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep: {}
constraints: "not a list"
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="constraints must be a list of strings"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_constraints_not_strings(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep: {}
constraints:
  - x > 1
  - 123
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="constraints must be a list of strings"):
            load_sweep_yaml(yaml_file)

    def test_load_sweep_yaml_constraints_null_becomes_empty_list(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep:
  x: [1, 2]
constraints:
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert len(configs) == 2

    def test_load_sweep_yaml_no_constraints_key(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep:
  x: [1, 2]
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert len(configs) == 2

    def test_load_sweep_yaml_preserves_base_params(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params:
  alpha: 0.5
  beta: 0.3
sweep:
  x: [1]
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert configs[0]["alpha"] == 0.5
        assert configs[0]["beta"] == 0.3

    def test_load_sweep_yaml_empty_base_params(self, tmp_path: Path) -> None:
        yaml_content = """
strategy: my_strategy
base_params: {}
sweep:
  x: [1, 2]
"""
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text(yaml_content)
        strategy, configs = load_sweep_yaml(yaml_file)
        assert len(configs) == 2


class TestCanonicalJson:
    """Tests for canonical_json() serialization."""

    def test_canonical_json_simple_dict(self) -> None:
        obj = {"a": 1, "b": 2}
        result = canonical_json(obj)
        assert result == '{"a":1,"b":2}'

    def test_canonical_json_sorted_keys(self) -> None:
        obj = {"z": 1, "a": 2}
        result = canonical_json(obj)
        assert result == '{"a":2,"z":1}'

    def test_canonical_json_nested_dict(self) -> None:
        obj = {"b": {"x": 1}, "a": 2}
        result = canonical_json(obj)
        assert '{"a":2,"b":{"x":1}}' == result

    def test_canonical_json_list(self) -> None:
        obj = {"items": [1, 2, 3]}
        result = canonical_json(obj)
        assert result == '{"items":[1,2,3]}'

    def test_canonical_json_string_values(self) -> None:
        obj = {"name": "test", "type": "strategy"}
        result = canonical_json(obj)
        assert '"name":"test"' in result
        assert '"type":"strategy"' in result

    def test_canonical_json_no_spaces(self) -> None:
        obj = {"a": 1}
        result = canonical_json(obj)
        assert " " not in result

    def test_canonical_json_custom_object_with_str(self) -> None:
        class CustomObj:
            def __str__(self) -> str:
                return "custom_string"

        obj = {"custom": CustomObj()}
        result = canonical_json(obj)
        assert "custom_string" in result


class TestConfigHash:
    """Tests for config_hash() hashing."""

    def test_config_hash_simple_dict(self) -> None:
        cfg = {"a": 1}
        result = config_hash(cfg)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_config_hash_all_hex_chars(self) -> None:
        cfg = {"a": 1}
        result = config_hash(cfg)
        assert all(c in "0123456789abcdef" for c in result)

    def test_config_hash_deterministic(self) -> None:
        cfg = {"a": 1, "b": 2}
        hash1 = config_hash(cfg)
        hash2 = config_hash(cfg)
        assert hash1 == hash2

    def test_config_hash_key_order_irrelevant(self) -> None:
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_config_hash_different_values_different_hashes(self) -> None:
        cfg1 = {"a": 1}
        cfg2 = {"a": 2}
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_config_hash_different_keys_different_hashes(self) -> None:
        cfg1 = {"a": 1}
        cfg2 = {"b": 1}
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_config_hash_nested_dict(self) -> None:
        cfg = {"params": {"alpha": 0.5}}
        result = config_hash(cfg)
        assert len(result) == 16

    def test_config_hash_list_values(self) -> None:
        cfg = {"values": [1, 2, 3]}
        result = config_hash(cfg)
        assert len(result) == 16

    def test_config_hash_consistent_with_canonical_json(self) -> None:
        cfg = {"z": 1, "a": 2}
        hash1 = config_hash(cfg)
        cfg_reordered = {"a": 2, "z": 1}
        hash2 = config_hash(cfg_reordered)
        assert hash1 == hash2
