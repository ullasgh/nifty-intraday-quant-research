"""ResearchContract: the declared-before-run schema, specs/research_contract.md (Spec C4).

A contract is a declared-before-run schema covering `data`, `features`, `label`,
`execution`, `portfolio` and `validation`, plus a top-level `seed`
(AMENDMENT 1 item 1 -- `seed` cuts across sections, so it is not filed under any
single one). The research entry points `run_backtest()` and `run_tilt()` refuse to
run without one.

`contract_hash` REPLACES the three pre-existing `config_hash` readings
(`config.py:72`, `strategy/registry.py:62`, and the now-deleted dead copy in
`research/sweep.py`): `config.py`/`strategy/registry.py`'s `config_hash` now
delegate to `canonical_hash()` below -- the same canonicalisation + hashing
mechanism `contract_hash` applies specifically to a `ResearchContract`'s six
sections plus seed. Delegating to the exact six-section structure was not possible
for those two functions without breaking their pre-existing generic
"hash any mapping" call sites (`tests/test_config.py`, `tests/test_research.py`,
and every `strategy_config_hash({"strategy": ..., "params": ...})` call site in
`cli.py`), so the shared *mechanism* is what is unified, not the input shape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_REQUIRED_SECTIONS: tuple[str, ...] = (
    "data",
    "features",
    "label",
    "execution",
    "portfolio",
    "validation",
)
_HOLDOUT_INTENTS: tuple[str, ...] = ("never", "after_conditions_close", "reading_now")


def _canonicalize(value: Any) -> Any:
    """Recursively normalise `value` so `json.dumps(sort_keys=True)` renders the
    same output regardless of the original mapping's key insertion order, at any
    nesting depth (obligation 2: "insensitive to key ordering within a section")."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize(item) for item in sorted(value, key=repr)]
    return value


def canonical_json(payload: Any) -> str:
    """Deterministic, key-order-insensitive JSON serialisation used for hashing."""
    return json.dumps(
        _canonicalize(payload), sort_keys=True, separators=(",", ":"), default=str
    )


def canonical_hash(payload: Any) -> str:
    """16-hex-char blake2s digest of `payload`'s canonical JSON form.

    The single hashing mechanism `config.py:config_hash` and
    `strategy/registry.py:config_hash` now delegate to; `contract_hash` below is
    this same mechanism applied to a `ResearchContract`'s six sections + seed.
    """
    return hashlib.blake2s(canonical_json(payload).encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True, kw_only=True, eq=False)
class ResearchContract:
    """A declared-before-run research contract.

    Every field defaults to `None` so a whole section omitted at the call site and
    a whole section explicitly passed as `None` fail identically, in
    `__post_init__`, with one message naming the field -- Python's own
    missing-keyword-argument `TypeError` cannot express that once every field has a
    default (which it must, so both cases share one code path). The spec's
    "supplied or explicitly declared absent" allowance is for FIELDS within a
    section, not for a whole section: a whole section can never be validly absent.
    """

    data: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    label: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    portfolio: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in _REQUIRED_SECTIONS:
            value = getattr(self, name)
            if value is None or not isinstance(value, dict):
                raise TypeError(
                    f"ResearchContract is missing required section {name!r} "
                    "(a section must be supplied as a non-None mapping)"
                )
        if self.seed is None:
            raise TypeError(
                "ResearchContract is missing required top-level field 'seed' "
                "(AMENDMENT 1: seed is a top-level field, not a section member)"
            )
        assert self.validation is not None
        holdout_intent = self.validation.get("holdout_intent")
        if holdout_intent not in _HOLDOUT_INTENTS:
            raise ValueError(
                f"validation['holdout_intent'] must be one of {_HOLDOUT_INTENTS}, "
                f"got {holdout_intent!r}"
            )

    def __hash__(self) -> int:
        return hash(self.contract_hash)

    @property
    def contract_hash(self) -> str:
        return contract_hash(self)

    def register_trial(self) -> int:
        """Stateful trial counter: each call registers one more realised trial
        against this contract's declared `n_planned_trials`, raising once the
        (k+1)th trial is attempted (obligation 8)."""
        assert self.validation is not None
        planned = self.validation.get("n_planned_trials")
        current: int = self.__dict__.get("_trial_count", 0)
        next_count = current + 1
        if planned is not None and next_count > planned:
            raise ValueError(
                f"attempted trial {next_count} exceeds declared "
                f"n_planned_trials={planned}"
            )
        object.__setattr__(self, "_trial_count", next_count)
        return next_count

    def check_trial_count(self, attempted_trial_number: int) -> None:
        """Stateless companion to `register_trial`: raises if
        `attempted_trial_number` exceeds the declared `n_planned_trials`, without
        mutating any counter."""
        assert self.validation is not None
        planned = self.validation.get("n_planned_trials")
        if planned is not None and attempted_trial_number > planned:
            raise ValueError(
                f"attempted trial {attempted_trial_number} exceeds declared "
                f"n_planned_trials={planned}"
            )

    def check_holdout_intent(self, *, allow_holdout: bool) -> None:
        """Refuse any disagreement between the declared tri-state intent and the
        `--allow-holdout` flag actually passed, in EITHER direction (obligation 9):
        only `holdout_intent='reading_now'` may reach the holdout, and it must
        match the flag."""
        assert self.validation is not None
        intent = self.validation.get("holdout_intent")
        if intent == "reading_now" and not allow_holdout:
            raise ValueError(
                "validation.holdout_intent='reading_now' requires --allow-holdout"
            )
        if intent != "reading_now" and allow_holdout:
            raise ValueError(
                f"--allow-holdout was passed but validation.holdout_intent={intent!r} "
                "(only 'reading_now' may reach the holdout)"
            )


def contract_hash(contract: ResearchContract) -> str:
    """16-hex-char, deterministic, order-insensitive hash of all six sections plus
    `seed`. REPLACES config.py/strategy_registry.py's separate config_hash readings
    (spec P2) -- see `canonical_hash()` above, which both now delegate to."""
    payload = {
        "data": contract.data,
        "features": contract.features,
        "label": contract.label,
        "execution": contract.execution,
        "portfolio": contract.portfolio,
        "validation": contract.validation,
        "seed": contract.seed,
    }
    return canonical_hash(payload)
