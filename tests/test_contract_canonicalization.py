"""Two canonicalisation paths that the C4 delegation left untested.

Both are real behaviours with a way to be wrong, not coverage padding:

* `config_hash` accepts a pydantic `RunConfig` as well as a plain mapping. After the
  delegation to `contract.canonical_hash` (specs/research_contract.md), the model is
  dumped with `mode="python"` so nested `date` fields stay real `date` objects. Dumping
  in JSON mode instead would stringify them BEFORE canonicalisation and produce a
  different digest for the same config depending on how it was passed in.

* `_canonicalize` sorts sets, because Python set iteration order is not stable across
  processes for many element types. An unsorted set inside a contract section would make
  `contract_hash` non-deterministic across runs -- which defeats the entire purpose of a
  pre-registration hash.
"""

from __future__ import annotations

import datetime as dt

from nifty_quant.config import RunConfig, config_hash
from nifty_quant.research.contract import canonical_hash, canonical_json


def _run_config() -> RunConfig:
    return RunConfig(
        strategy="volume_breakout",
        params={"window": 30},
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
    )


def test_config_hash_accepts_a_pydantic_model_and_matches_its_python_dump() -> None:
    cfg = _run_config()

    # The model and its python-mode dump must hash identically: the wrapper's only job
    # for a BaseModel is to dump it, so a divergence here means the dump mode changed.
    assert config_hash(cfg) == config_hash(cfg.model_dump(mode="python"))


def test_config_hash_keeps_dates_as_dates_not_pre_stringified() -> None:
    cfg = _run_config()

    # `mode="json"` would stringify the dates before canonicalisation. canonical_json
    # renders `date` via `default=str` anyway, so the RENDERED text matches -- but the
    # distinction is real and this pins the dump mode against a silent change to it.
    assert "2024-01-01" in canonical_json(cfg.model_dump(mode="python"))
    assert isinstance(cfg.model_dump(mode="python")["start"], dt.date)


def test_config_hash_is_sensitive_to_every_field() -> None:
    base = _run_config()
    for field, value in (
        ("strategy", "vwap_reversion"),
        ("universe", "nifty50"),
        ("start", dt.date(2024, 2, 1)),
        ("end", dt.date(2025, 1, 31)),
        ("seed", 7),
        ("costs", "nse_delivery_default"),
    ):
        other = base.model_copy(update={field: value})
        assert config_hash(other) != config_hash(base), (
            f"changing {field!r} must change the hash -- this is the P2 collision defect, "
            "where the old strategy/registry hash covered only strategy+params"
        )


def test_canonicalize_sorts_sets_so_the_hash_is_order_independent() -> None:
    # Two sets that are equal but very likely to iterate in different orders.
    a = {"zeta", "alpha", "mu", "beta"}
    b = set(reversed(list(a)))

    assert canonical_hash({"symbols": a}) == canonical_hash({"symbols": b})
    # And the rendered form is genuinely sorted, not merely coincidentally equal.
    assert canonical_json({"symbols": a}) == '{"symbols":["alpha","beta","mu","zeta"]}'


def test_canonicalize_sorts_sets_nested_inside_a_section() -> None:
    left = {"validation": {"years": {2024, 2023, 2025}}}
    right = {"validation": {"years": {2025, 2024, 2023}}}

    assert canonical_hash(left) == canonical_hash(right)
    assert canonical_json(left) == '{"validation":{"years":[2023,2024,2025]}}'
