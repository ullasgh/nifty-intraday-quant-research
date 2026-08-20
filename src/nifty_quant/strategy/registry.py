"""Strategy registration and lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nifty_quant.strategy.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Decorator to register a strategy class under its name."""
    if cls.name in _REGISTRY:
        raise ValueError(f"Strategy name {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Strategy]:
    """Look up a registered strategy class by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown strategy: {name}") from None


def available() -> list[str]:
    """Return a sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def build(cfg: Mapping[str, Any]) -> Strategy:
    """Build a strategy instance from a config mapping.

    cfg must contain a "strategy" key (name) and a "params" key (dict).
    """
    name = cfg["strategy"]
    if not isinstance(name, str):
        raise TypeError("'strategy' key must be a string")
    cls = get(name)
    return cls.from_config(cfg)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert config values to JSON-serializable deterministic form."""
    if isinstance(obj, Mapping):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, set):
        # Sort by repr for deterministic ordering independent of set iteration order.
        return [_to_serializable(item) for item in sorted(obj, key=repr)]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"Unsupported config value type: {type(obj).__name__}")


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Return a stable 16-hex-char hash of an order-independent canonical JSON form.

    Thin wrapper delegating to `nifty_quant.research.contract.canonical_hash`
    (specs/research_contract.md: "REPLACES the three config_hash functions").
    NOTE (spec P2): this function only ever sees whatever mapping the caller
    passes it (typically `{"strategy": ..., "params": ...}`); it cannot see
    universe/dates/costs/seed unless the caller includes them, so callers that
    need a run's true identity (`walkforward`, `sweep`) must hash a
    `ResearchContract`'s `contract_hash`, not this function's output.
    """
    serializable = _to_serializable(dict(cfg))
    from nifty_quant.research.contract import canonical_hash

    return canonical_hash(serializable)
