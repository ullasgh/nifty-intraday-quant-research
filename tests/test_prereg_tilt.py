"""Tests for `scripts/prereg_tilt.py`, the tilt's pre-registration artifact.

These tests never load a panel and never touch `data/`: everything asserted here
is about the hash binding, the record's provenance completeness, and the holdout
tripwire, all of which are pure functions of module constants.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_prereg_module() -> Any:
    """Import `scripts/prereg_tilt.py` (not an installed package) by path."""
    module_path = REPO_ROOT / "scripts" / "prereg_tilt.py"
    spec = importlib.util.spec_from_file_location("prereg_tilt", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prereg_tilt"] = module
    spec.loader.exec_module(module)
    return module


prereg = _load_prereg_module()


# ---------------------------------------------------------------------------
# 1. The hash binds the configuration.
# ---------------------------------------------------------------------------


def test_config_hash_is_deterministic() -> None:
    assert prereg.prereg_config_hash() == prereg.prereg_config_hash()
    assert len(prereg.prereg_config_hash()) == 16


def test_config_hash_is_key_order_insensitive() -> None:
    reversed_params = dict(reversed(list(prereg.FROZEN_PARAMS.items())))
    assert list(reversed_params) != list(prereg.FROZEN_PARAMS)
    assert prereg.prereg_config_hash(reversed_params) == prereg.prereg_config_hash()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("entry_hhmm", "09:20"),
        ("exit_hhmm", "15:15"),
        ("capital", 2_000_000.0),
        ("tilt", "aggressive"),
        ("smoothing", 1.0),
        ("rebalance_every", 2),
        ("universe", "nifty50"),
        ("continuous_only", True),
        ("seed", 1),
    ],
)
def test_every_frozen_param_changes_hash(name: str, value: Any) -> None:
    """Changing ANY tilt parameter must change `config_hash`.

    This is the property that makes the pre-registration mean something: a hash
    that does not move when `smoothing` goes 0.10 -> 1.0 (the difference between
    +3.22 and -9.41 bps/day) binds nothing.
    """
    assert prereg.FROZEN_PARAMS[name] != value, "perturbation must differ from frozen"
    mutated = dict(prereg.FROZEN_PARAMS)
    mutated[name] = value
    assert prereg.prereg_config_hash(mutated) != prereg.prereg_config_hash()


def test_holdout_window_is_bound_into_the_hash() -> None:
    shifted = prereg.prereg_config_hash(
        holdout_end=prereg.HOLDOUT_END + datetime.timedelta(days=1)
    )
    assert shifted != prereg.prereg_config_hash()
    shifted_start = prereg.prereg_config_hash(
        holdout_start=prereg.HOLDOUT_START + datetime.timedelta(days=1)
    )
    assert shifted_start != prereg.prereg_config_hash()


def test_research_window_is_not_bound_into_the_hash() -> None:
    """The holdout run uses different dates by construction; binding the research
    window would guarantee the holdout run can never match its pre-registration."""
    payload = json.loads(_hashed_payload_json())
    assert "research_start" not in payload
    assert "research_end" not in payload
    assert payload["params"] == prereg.FROZEN_PARAMS


def _hashed_payload_json() -> str:
    """Reconstruct the payload `prereg_config_hash` hashes, for inspection."""
    from nifty_quant.research.contract import canonical_json

    return canonical_json(
        {
            "schema": prereg.PREREG_SCHEMA,
            "strategy": prereg.STRATEGY,
            "params": dict(prereg.FROZEN_PARAMS),
            "holdout_start": prereg.HOLDOUT_START.isoformat(),
            "holdout_end": prereg.HOLDOUT_END.isoformat(),
        }
    )


def test_binding_report_shows_every_parameter_changing() -> None:
    report = prereg.binding_report()
    assert "NOT BOUND" not in report
    for name in prereg.FROZEN_PARAMS:
        assert name in report


# ---------------------------------------------------------------------------
# 2. What is hashed is what is executed.
# ---------------------------------------------------------------------------


def test_frozen_params_cover_every_tilt_config_field() -> None:
    """`FROZEN_PARAMS` must be exactly `TiltConfig`'s fields minus start/end.

    Without this, a newly added `TiltConfig` parameter would be executed but not
    pre-registered -- a silent hole in the freeze.
    """
    from nifty_quant.research.tilt import TiltConfig

    tilt_fields = {f.name for f in dataclasses.fields(TiltConfig)}
    assert set(prereg.FROZEN_PARAMS) == tilt_fields - {"start", "end"}


def test_frozen_tilt_config_carries_the_hashed_params() -> None:
    config = prereg.frozen_tilt_config()
    for name, value in prereg.FROZEN_PARAMS.items():
        assert getattr(config, name) == value
    assert config.start == prereg.RESEARCH_START
    assert config.end == prereg.RESEARCH_END


# ---------------------------------------------------------------------------
# 3. The holdout is not spent.
# ---------------------------------------------------------------------------


def test_research_window_is_outside_the_holdout() -> None:
    prereg._assert_research_window_outside_holdout()
    assert prereg.RESEARCH_END < prereg.HOLDOUT_START


def test_holdout_tripwire_refuses_a_research_window_touching_the_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "RESEARCH_END", prereg.HOLDOUT_START)
    with pytest.raises(ValueError, match="holdout"):
        prereg._assert_research_window_outside_holdout()


def test_declared_holdout_window_matches_the_lock_file() -> None:
    lock_path = REPO_ROOT / "results" / "holdout_lock.json"
    if not lock_path.exists():  # pragma: no cover - lock file is committed
        pytest.skip("holdout_lock.json absent")
    state = json.loads(lock_path.read_text(encoding="utf-8"))
    assert state["holdout_start"] == prereg.HOLDOUT_START.isoformat()
    assert state["holdout_end"] == prereg.HOLDOUT_END.isoformat()


# ---------------------------------------------------------------------------
# 4. The record itself: provenance completeness.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _FakeYearRow:
    year: int = -1
    n_sessions: int = 388
    gross_bps: float = 4.57
    turnover: float = 0.1272
    cost_bps: float = 1.35
    net_bps: float = 3.22
    ann_net_pct: float = 8.10


@dataclasses.dataclass(frozen=True)
class _FakeResult:
    config: Any
    per_year: tuple[_FakeYearRow, ...] = (_FakeYearRow(year=2024),)
    total: _FakeYearRow = _FakeYearRow()
    clip_per_name: float = 13_513.5
    round_trip_bps: float = 10.62
    breakeven_turnover: float = 0.43
    n_symbols: int = 149
    warnings: tuple[str, ...] = ()
    min_weight_seen: float = 0.0
    max_weight_sum_deviation: float = 1e-15
    max_n_held: int = 74


def _fake_contract() -> Any:
    from nifty_quant.research.contract import ResearchContract

    return ResearchContract(
        data={"panel_id": "p", "panel_hash": "ph"},
        features={"feature_ids": ["h2_overnight_reversal"]},
        label={"horizon_bars": 1},
        execution={"cost_model_id": prereg.COST_MODEL_ID},
        portfolio={"sizing_scheme": "long_only_weight_tilt"},
        validation={"holdout_intent": "after_conditions_close", "n_planned_trials": 1},
        seed=0,
    )


def _build_record() -> Any:
    return prereg.build_prereg_record(
        result=_FakeResult(config=prereg.frozen_tilt_config()),
        contract=_fake_contract(),
        panel_hash="panel-hash-value",
        universe_hash="universe-hash-value",
        data_fingerprint="fingerprint-value",
        result_path="results/prereg/tilt_x.json",
        wall_s=12.5,
    )


#: The only columns a tilt pre-registration may leave NULL, each with a stated
#: reason in `build_prereg_record`'s docstring. Anything else being NULL is a bug.
_ALLOWED_NULL_FIELDS = {
    "sharpe_gross",  # TiltResult exposes pooled means only, never a daily series
    "sharpe_net",  # ditto
    "n_trades",  # a smoothed weight book has no discrete round trips
    "ruin_index",  # nothing was ruined
    "parent_trial_id",  # this is a root pre-registration, not a child of a sweep
    "error",  # the run succeeded
}


def test_every_provenance_field_is_populated() -> None:
    record = _build_record()
    nulls = {
        name
        for name, value in dataclasses.asdict(record).items()
        if value is None or value == ""
    }
    assert nulls == _ALLOWED_NULL_FIELDS


def test_git_sha_is_real_and_not_none() -> None:
    """`git_sha` was hard-coded `None` at all 7 historical write sites."""
    record = _build_record()
    assert record.git_sha is not None
    assert record.git_sha not in ("", "None", "null")
    base = record.git_sha.removesuffix("-dirty")
    assert base == "no-git" or (len(base) == 40 and all(
        c in "0123456789abcdef" for c in base
    ))


def test_purpose_is_a_preregistration_not_a_sweep_row() -> None:
    record = _build_record()
    assert record.purpose == "preregistration"
    assert record.purpose not in ("exploration", "confirmation")


def test_config_hash_column_binds_the_parameters_not_the_contract() -> None:
    """The recorded `config_hash` must move when a tilt parameter moves.

    `run_tilt` sets `config_hash = contract.contract_hash`, which contains no
    TiltConfig parameter at all -- so this asserts the pre-registration does NOT
    inherit that behaviour.
    """
    record = _build_record()
    assert record.config_hash == prereg.prereg_config_hash()
    assert record.contract_hash == _fake_contract().contract_hash
    mutated = dict(prereg.FROZEN_PARAMS)
    mutated["smoothing"] = 1.0
    assert record.config_hash != prereg.prereg_config_hash(mutated)


def test_record_carries_the_research_period_result() -> None:
    record = _build_record()
    params = json.loads(record.params_json)
    assert params["research_start"] == prereg.RESEARCH_START.isoformat()
    assert params["research_end"] == prereg.RESEARCH_END.isoformat()
    assert params["research_result"]["n_sessions"] == 388
    assert params["research_result"]["net_bps"] == pytest.approx(3.22)
    assert params["frozen_params"] == prereg.FROZEN_PARAMS
    assert record.turnover == pytest.approx(0.1272)
    assert record.breakeven_bps == pytest.approx(4.57)
    assert record.start == prereg.RESEARCH_START.isoformat()
    assert record.end == prereg.RESEARCH_END.isoformat()


def test_embargo_components_are_the_tilt_s_real_horizons() -> None:
    record = _build_record()
    embargo = json.loads(record.embargo_components)
    assert embargo == {
        "feature_lookback": 1.0,
        "label_horizon": 0.0,
        "holding_period": 1.0,
        "execution_horizon": 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Registry round-trip.
# ---------------------------------------------------------------------------


def test_record_round_trips_through_the_registry(tmp_path: Path) -> None:
    from nifty_quant.research.registry import TrialRegistry

    registry = TrialRegistry(tmp_path / "trials.db")
    record = _build_record()
    registry.record(record)

    rows = registry.all(strategy="tilt")
    assert len(rows) == 1
    assert rows[0] == record

    # Append-only + idempotent: re-running the script must not duplicate the row.
    registry.record(record)
    assert len(registry.all(strategy="tilt")) == 1


def test_prereg_row_is_not_counted_as_an_exploration_trial(tmp_path: Path) -> None:
    """A frozen pre-registration must not inflate the sweep count that PBO/DSR
    corrections are computed against."""
    from nifty_quant.research.registry import TrialRegistry

    registry = TrialRegistry(tmp_path / "trials.db")
    registry.record(_build_record())
    assert registry.n_trials(strategy="tilt") == 0
    assert registry.n_trials(strategy="tilt", purpose="preregistration") == 1


def test_split_id_names_the_holdout_window() -> None:
    assert prereg.split_id() == "prereg_holdout_2025-08-14_2026-08-14"
