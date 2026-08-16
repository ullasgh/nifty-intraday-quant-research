"""Parameter sweep and experiment orchestration."""

from __future__ import annotations

import hashlib
import itertools
import json
import operator
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}

_CONSTRAINT_RE = re.compile(
    r"^\s*(?P<lhs>\S+)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<rhs>\S+)\s*$"
)

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_literal(token: str) -> Any:
    if _INT_RE.fullmatch(token):
        return int(token)

    if _FLOAT_RE.fullmatch(token):
        return float(token)

    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]

    if _IDENTIFIER_RE.fullmatch(token):
        return token

    raise ValueError(f"invalid literal in constraint: {token!r}")


def _evaluate_constraint(constraint: str, params: Mapping[str, Any]) -> bool:
    match = _CONSTRAINT_RE.match(constraint)
    if match is None:
        raise ValueError(f"invalid constraint: {constraint!r}")

    lhs_token = match.group("lhs")
    op_token = match.group("op")
    rhs_token = match.group("rhs")

    lhs = params[lhs_token] if lhs_token in params else _parse_literal(lhs_token)
    rhs = params[rhs_token] if rhs_token in params else _parse_literal(rhs_token)

    comparator = _OPERATORS[op_token]
    try:
        return bool(comparator(lhs, rhs))
    except TypeError as exc:
        raise ValueError(
            f"cannot compare {lhs!r} and {rhs!r} with operator '{op_token}'"
        ) from exc


def expand(
    base_params: Mapping[str, Any],
    sweep: Mapping[str, Sequence[Any]],
    constraints: Sequence[str] = (),
) -> list[dict[str, Any]]:
    keys = list(sweep.keys())
    value_lists = [list(values) for values in sweep.values()]

    results: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        params = dict(base_params)
        params.update(zip(keys, combo))

        if all(_evaluate_constraint(constraint, params) for constraint in constraints):
            results.append(params)

    return results


def load_sweep_yaml(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("sweep YAML must contain a top-level mapping")

    missing = [key for key in ("strategy", "base_params", "sweep") if key not in data]
    if missing:
        raise ValueError(f"missing required top-level keys: {', '.join(missing)}")

    strategy = data["strategy"]
    base_params = data["base_params"]
    sweep = data["sweep"]
    constraints = data.get("constraints", [])

    if not isinstance(strategy, str):
        raise ValueError("strategy must be a string")
    if not isinstance(base_params, dict):
        raise ValueError("base_params must be a mapping")
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping")
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list) or not all(
        isinstance(constraint, str) for constraint in constraints
    ):
        raise ValueError("constraints must be a list of strings")

    return strategy, expand(base_params, sweep, constraints)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Any) -> str:
    return hashlib.blake2s(
        canonical_json(cfg).encode("utf-8"), digest_size=8
    ).hexdigest()
