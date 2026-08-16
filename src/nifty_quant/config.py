"""Reproducible backtest run configuration: YAML loading, validation, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class RunConfig(BaseModel):
    """Reproducible backtest run configuration, loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    params: dict[str, Any] = {}
    universe: str = "all_equity"
    start: date
    end: date
    capital: float = 1e7
    costs: str = "nse_intraday_default"
    square_off_time: str = "15:20"
    decision_latency_bars: int = 0
    seed: int = 0


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict via yaml.safe_load.

    Raises FileNotFoundError if `path` does not exist, or ValueError if the parsed
    YAML is not a mapping at the top level.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping/dict")
    return data


def load_run_config(path: Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file at `path`."""
    data = load_yaml(path)
    return RunConfig(**data)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing/comparison purposes.

    Dict keys are sorted recursively, separators are compact (no whitespace), and
    `date`/`datetime` objects are rendered as ISO-format strings. If `obj` is a
    pydantic BaseModel it is first converted via `model_dump(mode="python")` so
    nested date fields remain real `date` objects before serialization.
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="python")

    def _default(o: Any) -> str:
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def config_hash(cfg: Any) -> str:
    """Return a deterministic 16-hex-char blake2s hash of `cfg`'s canonical JSON."""
    return hashlib.blake2s(canonical_json(cfg).encode("utf-8"), digest_size=8).hexdigest()
