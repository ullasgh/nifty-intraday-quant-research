"""Independent test suite for `ResearchContract`, spec `specs/research_contract.md`.

TDD "red" suite: `nifty_quant.research.contract` does not exist yet, so most tests below are
EXPECTED to fail on ImportError/AttributeError/TypeError. That is the correct, intended outcome
-- do not weaken assertions or wrap anything in try/except to dodge a failure.

ASSUMPTION (documented, not certainty): the implementation lives at
`nifty_quant.research.contract`, exposing `ResearchContract` (a frozen, hashable dataclass) and a
module-level `contract_hash(contract) -> str` function (16-hex-char, deterministic), mirroring the
existing `config_hash` functions in `config.py` / `strategy/registry.py`.

ASSUMPTION (documented, not certainty): `ResearchContract`'s constructor takes the six required
section mappings (`data`, `features`, `label`, `execution`, `portfolio`, `validation`) as
keyword-only dict arguments, PLUS a top-level `seed: int` keyword argument. `seed` is placed
outside all six sections deliberately: the spec's six-section table gives it no home, and it
cuts across sections (it drives bootstrap resampling, permutation nulls, CSCV split assignment,
and any stochastic strategy tie-break), so filing it under one section would assert something
false about its scope.

ASSUMPTION (documented, not certainty): omitting any of the six named sections raises `TypeError`
whose message names the missing section (the standard dataclass missing-argument message already
does this, e.g. "missing 1 required positional argument: 'data'").

ASSUMPTION (documented, not certainty): for obligation 8, the sweep gate is exposed as
`contract.check_trial_count(attempted_trial_number)`, called immediately before each attempt; it
raises once `attempted_trial_number` exceeds `contract.validation["n_planned_trials"]`. The
signature of an actual sweep-running callable is not pinned by the spec, so this tests the gate
method directly rather than guessing a CLI/function signature that does not yet exist.

ASSUMPTION (documented, not certainty): for obligation 9, `contract.check_holdout_intent(
allow_holdout: bool) -> None` raises `ValueError` when `validation["holdout_intent"]` disagrees
with the passed flag (`"never"` + `allow_holdout=True`, or `"reading_now"` + `allow_holdout=False`).

Obligation 10 goes through the real `nq walkforward` CLI path (per explicit direction: do not
invent a `run_walkforward_to_registry(...)` seam that exists only to be tested). `load_panel`,
`load_universe`, and `settings.RESULTS_ROOT` are monkeypatched so no real `data/` is touched, and
`TrialRegistry.record` is wrapped to capture every record written, including the pooled one.
Because the exact walkforward CLI flag names are not pinned by the spec, `_walkforward_cli_args`
introspects the Click command's registered parameters (via `typer.main.get_command`) and supplies
plausible values by name rather than guessing a single fixed flag list.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from nifty_quant.research.contract import ResearchContract, contract_hash
from pydantic import BaseModel

from nifty_quant.data.panel import Panel

_DEFAULT_SEED = 137


def _full_sections(**overrides: dict[str, object]) -> dict[str, dict[str, object]]:
    """The six required sections, WITHOUT `seed` (a top-level field, see module docstring)."""
    sections: dict[str, dict[str, object]] = {
        "data": {
            "panel_id": "synthetic_intraday",
            "panel_hash": "panel-hash-001",
            "start": "2022-01-03",
            "end": "2022-01-22",
            "bar_interval": "1min",
            "universe_name": "all_equity",
            "universe_hash": "universe-hash-001",
        },
        "features": {
            "features": [
                {"id": "overnight_return", "params": {"lookback_bars": 5}},
                {"id": "realized_volatility", "params": {"window_bars": 20}},
            ],
            "feature_version": "features-v1",
        },
        "label": {
            "horizon_bars": 5,
            "construction": "forward_close_to_close_return",
            "overlapping": False,
        },
        "execution": {
            "cost_model_id": "nse_intraday_equity_v1",
            "slippage_model_id": "sqrt_impact_v1",
            "decision_latency_bars": 1,
            "participation_cap": 0.10,
        },
        "portfolio": {
            "sizing_scheme": "volatility_scaled",
            "gross_clip": 1.0,
            "max_weight_clip": 0.20,
            "target_volatility": None,
        },
        "validation": {
            "split_scheme": "walkforward",
            "purge_width": 5,
            "embargo_width": 3,
            "n_planned_trials": 2,
            "holdout_intent": "never",
        },
    }

    for section_name, changes in overrides.items():
        sections[section_name] = {**sections[section_name], **changes}

    return sections


def _contract(seed: int = _DEFAULT_SEED, **overrides: dict[str, object]) -> ResearchContract:
    return ResearchContract(**_full_sections(**overrides), seed=seed)


def _reverse_mapping(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_mapping(item) for key, item in reversed(tuple(value.items()))}
    if isinstance(value, list):
        return [_reverse_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_reverse_mapping(item) for item in value)
    return value


def _hash_in_fresh_process(sections: dict[str, dict[str, object]], seed: int) -> str:
    """Compute `contract_hash` in a brand-new Python process, to test obligation 2's
    "stable across process restarts" requirement for real (not merely "same value twice in this
    interpreter", which would not catch e.g. reliance on `id()`, insertion-ordered non-canonical
    hashing, or process-local salts)."""
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    python_path = [str(repository_root / "src"), str(repository_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    source = (
        "import json, sys\n"
        "from nifty_quant.research.contract import ResearchContract, contract_hash\n"
        "payload = json.loads(sys.argv[1])\n"
        "sections = payload['sections']\n"
        "seed = payload['seed']\n"
        "print(contract_hash(ResearchContract(**sections, seed=seed)))\n"
    )
    payload = json.dumps({"sections": sections, "seed": seed}, separators=(",", ":"))
    return subprocess.check_output(
        [sys.executable, "-c", source, payload],
        cwd=repository_root,
        env=environment,
        text=True,
    ).strip()


# --------------------------------------------------------------------------------------
# Obligation 1: missing any of the six required sections raises, naming the section.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_section",
    ("data", "features", "label", "execution", "portfolio", "validation"),
)
def test_obligation1_missing_required_section_is_rejected(missing_section: str) -> None:
    sections = _full_sections()
    del sections[missing_section]

    with pytest.raises(TypeError) as exc_info:
        ResearchContract(**sections, seed=_DEFAULT_SEED)

    assert missing_section in str(exc_info.value)


def test_edge_missing_top_level_seed_is_rejected() -> None:
    """`seed` is a required top-level field (see module docstring), not part of any of the six
    named sections, so it is not covered by obligation 1's parametrization above -- but the same
    "no silent default for a research choice" principle applies to it."""
    sections = _full_sections()

    with pytest.raises(TypeError) as exc_info:
        ResearchContract(**sections)  # no seed=...

    assert "seed" in str(exc_info.value)


# --------------------------------------------------------------------------------------
# Obligation 2: contract_hash stable across process restarts, insensitive to key order.
# --------------------------------------------------------------------------------------


def test_obligation2_contract_hash_is_stable_across_process_restarts() -> None:
    sections = _full_sections()
    expected = contract_hash(ResearchContract(**sections, seed=_DEFAULT_SEED))

    restarted_hashes = [
        _hash_in_fresh_process(sections, _DEFAULT_SEED),
        _hash_in_fresh_process(sections, _DEFAULT_SEED),
    ]

    assert re.fullmatch(r"[0-9a-f]{16}", expected)
    assert restarted_hashes == [expected, expected]


def test_obligation2_contract_hash_ignores_mapping_key_order() -> None:
    sections = _full_sections()
    reordered_sections = {
        section_name: _reverse_mapping(section)
        for section_name, section in reversed(tuple(sections.items()))
    }

    original_hash = contract_hash(ResearchContract(**sections, seed=_DEFAULT_SEED))
    reordered_hash = contract_hash(ResearchContract(**reordered_sections, seed=_DEFAULT_SEED))

    assert reordered_hash == original_hash


# --------------------------------------------------------------------------------------
# Obligation 3: contract_hash CHANGES for each of universe / start / end / cost model id /
# seed / embargo width -- one assertion per field, per spec's explicit instruction.
# --------------------------------------------------------------------------------------


def test_obligation3_hash_changes_for_universe() -> None:
    original = _contract()
    changed = _contract(data={"universe_name": "nifty_500"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation3_hash_changes_for_start() -> None:
    original = _contract()
    changed = _contract(data={"start": "2022-01-04"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation3_hash_changes_for_end() -> None:
    original = _contract()
    changed = _contract(data={"end": "2022-01-23"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation3_hash_changes_for_cost_model_id() -> None:
    original = _contract()
    changed = _contract(execution={"cost_model_id": "nse_equity_costs_v2"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation3_hash_changes_for_seed() -> None:
    original = _contract(seed=137)
    changed = _contract(seed=138)

    assert contract_hash(changed) != contract_hash(original)


def test_obligation3_hash_changes_for_embargo_width() -> None:
    original = _contract()
    changed = _contract(validation={"embargo_width": 4})

    assert contract_hash(changed) != contract_hash(original)


# --------------------------------------------------------------------------------------
# Obligation 4: regression test for P2 -- fields the OLD strategy/registry.py:62 hash
# ignored (universe, dates, costs, seed) must change contract_hash.
# --------------------------------------------------------------------------------------


def test_obligation4_hash_includes_universe_ignored_by_old_registry_hash() -> None:
    original = _contract()
    changed = _contract(data={"universe_name": "nifty_500"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation4_hash_includes_dates_ignored_by_old_registry_hash() -> None:
    original = _contract()
    changed = _contract(data={"start": "2022-01-04"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation4_hash_includes_costs_ignored_by_old_registry_hash() -> None:
    original = _contract()
    changed = _contract(execution={"cost_model_id": "nse_equity_costs_v2"})

    assert contract_hash(changed) != contract_hash(original)


def test_obligation4_hash_includes_seed_ignored_by_old_registry_hash() -> None:
    original = _contract(seed=137)
    changed = _contract(seed=138)

    assert contract_hash(changed) != contract_hash(original)


def test_obligation4_old_registry_hash_actually_collided_on_these_fields() -> None:
    """Sanity anchor for the regression framing: the OLD `strategy/registry.py:62` `config_hash`
    only hashes `{"strategy": ..., "params": ...}`-shaped mappings, so two configs that differ
    only in universe/dates/costs/seed genuinely DO collide under it -- this is the defect
    `contract_hash` (tested above) must not repeat."""
    from nifty_quant.strategy.registry import config_hash as old_registry_config_hash

    cfg_a = {"strategy": "constant_weight", "params": {"weight": 0.05}}
    cfg_b = {"strategy": "constant_weight", "params": {"weight": 0.05}}

    assert old_registry_config_hash(cfg_a) == old_registry_config_hash(cfg_b)


# --------------------------------------------------------------------------------------
# Positive control / immutability / hashability -- reasonable inferences from "frozen,
# hashable declaration" that the spec implies but does not spell out as numbered obligations.
# --------------------------------------------------------------------------------------


def test_positive_control_full_contract_constructs() -> None:
    contract = _contract()

    assert isinstance(contract, ResearchContract)
    assert re.fullmatch(r"[0-9a-f]{16}", contract_hash(contract))


def test_contract_is_immutable() -> None:
    contract = _contract()

    with pytest.raises(FrozenInstanceError):
        contract.data = {}  # type: ignore[misc]


def test_contract_is_hashable() -> None:
    contract = _contract()

    hash_value = hash(contract)
    keyed = {contract: "declared"}

    assert isinstance(hash_value, int)
    assert keyed[contract] == "declared"


# --------------------------------------------------------------------------------------
# Shared synthetic-panel fixtures for the run_backtest/run_tilt/walkforward obligations
# (5, 6, 7, 10). Pattern mirrors the sibling `tests/test_tilt_a.py` fixture family so a
# genuinely valid call is made -- these must be red because the ENFORCEMENT is missing,
# not because the inputs are broken.
# --------------------------------------------------------------------------------------

_IST = ZoneInfo("Asia/Kolkata")
_N_SYMBOLS = 5
_SYMBOLS = tuple(f"SYM{i}" for i in range(_N_SYMBOLS))


def _ts_for(date: dt.date, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    stamp = pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=_IST)
    return int(stamp.tz_convert("UTC").timestamp())


def _build_panel(
    sessions: list[tuple[dt.date, dict[str, np.ndarray]]],
    symbols: tuple[str, ...] = _SYMBOLS,
) -> Panel:
    sessions_sorted = sorted(sessions, key=lambda item: item[0])
    ts_list: list[int] = []
    price_rows: list[np.ndarray] = []
    day_offsets = [0]
    row = 0

    for date, bars in sessions_sorted:
        for hhmm in sorted(bars):
            ts_list.append(_ts_for(date, hhmm))
            price_rows.append(np.asarray(bars[hhmm], dtype=np.float64))
            row += 1
        day_offsets.append(row)

    ts = np.array(ts_list, dtype=np.int64)
    price_arr = np.stack(price_rows).astype(np.float32)
    volume_arr = np.full(price_arr.shape, 1_000_000.0, dtype=np.float32)
    dates_arr = np.array([date for date, _ in sessions_sorted], dtype=object)

    return Panel(
        fields={"open": price_arr, "close": price_arr.copy(), "volume": volume_arr},
        symbols=symbols,
        ts=ts,
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=dates_arr,
    )


def _flat_row(base: float = 100.0, n: int = _N_SYMBOLS) -> np.ndarray:
    return np.array([base + index * 0.01 for index in range(n)], dtype=np.float64)


def _cycling_aggressive_panel(
    dates: list[dt.date],
    loser_of_day: list[int],
    entry_hhmm: str = "09:16",
    exit_hhmm: str = "15:20",
) -> Panel:
    sessions: list[tuple[dt.date, dict[str, np.ndarray]]] = []
    for date, loser in zip(dates, loser_of_day):
        entry_row = _flat_row(100.0)
        entry_row[loser] -= 3.0
        exit_row = entry_row + 0.05
        sessions.append((date, {entry_hhmm: entry_row, exit_hhmm: exit_row}))
    return _build_panel(sessions)


# --------------------------------------------------------------------------------------
# Obligation 5: run_backtest() without a contract raises.
# --------------------------------------------------------------------------------------


def test_obligation5_run_backtest_without_contract_raises() -> None:
    from nifty_quant.backtest.engine import BacktestConfig, run_backtest
    from nifty_quant.backtest.portfolio import GrossNotionalSizer
    from nifty_quant.execution.costs import NSEIntradayEquityCosts
    from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
    from nifty_quant.strategy.base import (
        DataRequest,
        MarketView,
        PortfolioState,
        Strategy,
        TargetPortfolio,
    )

    symbols = ("AAA", "BBB", "CCC")
    n_days = 2
    bars_per_day = 375
    n_rows = n_days * bars_per_day
    n_symbols = len(symbols)

    start = dt.date(2024, 1, 2)
    dates: list[dt.date] = []
    current = start
    while len(dates) < n_days:
        if current.weekday() < 5:
            dates.append(current)
        current += dt.timedelta(days=1)

    timestamp_chunks = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        index = pd.date_range(day_start, periods=bars_per_day, freq="1min")
        index_utc = index.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        seconds = ((index_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        timestamp_chunks.append(seconds)

    timestamps = np.concatenate(timestamp_chunks).astype(np.int64)
    day_offsets = np.arange(0, (n_days + 1) * bars_per_day, bars_per_day, dtype=np.int32)
    close = np.full((n_rows, n_symbols), 100.0, dtype=np.float64)
    panel = Panel(
        fields={
            "open": close.astype(np.float32),
            "high": close.astype(np.float32),
            "low": close.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float32),
        },
        symbols=symbols,
        ts=timestamps,
        day_offsets=day_offsets,
        dates=np.array(dates, dtype=object),
    )

    class ConstantWeightStrategy(Strategy):
        name = "constant_weight"

        class Params(BaseModel):
            weights: tuple[float, ...]
            decision_time: str = "10:00"

        def data_request(self) -> DataRequest:
            return DataRequest(decision_times=(self.params.decision_time,))

        def precompute(self, panel: Panel) -> dict:
            return {}

        def on_decision(self, view: MarketView, signals, state: PortfolioState):
            return TargetPortfolio(weights=np.array(self.params.weights, dtype=np.float64))

    strategy = ConstantWeightStrategy(ConstantWeightStrategy.Params(weights=(0.05, -0.03, 0.02)))
    config = BacktestConfig(
        fill_model=FillModel(slippage=SqrtImpactSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )

    # No `contract=` kwarg at all, with otherwise fully valid inputs. TODAY this succeeds (the
    # enforcement plainly does not exist) so `pytest.raises` correctly fails now -- red for the
    # right reason, not an accidental TypeError from a made-up kwarg.
    with pytest.raises(TypeError):
        run_backtest(strategy, panel, config)


# --------------------------------------------------------------------------------------
# Obligation 6 (MOST IMPORTANT): run_tilt() without a contract raises. Regression test
# for P1 -- today `research/tilt.py` has zero references to TrialRegistry/TrialRecord/
# run_backtest, so a fully valid call below currently SUCCEEDS, which is exactly why
# this test is expected to be red right now.
# --------------------------------------------------------------------------------------


def test_obligation6_run_tilt_without_contract_raises() -> None:
    from nifty_quant.research.tilt import TiltConfig, run_tilt

    dates = [dt.date(2022, 1, 3) + dt.timedelta(days=index) for index in range(20)]
    loser_of_day = [index % 2 for index in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)
    config = TiltConfig(start=dates[0], end=dates[-1])

    with pytest.raises(TypeError):
        run_tilt(panel, config)


# --------------------------------------------------------------------------------------
# Obligation 7: a completed run_tilt() writes a TrialRecord whose contract_hash matches
# the declared contract, with non-null seed and non-null git_sha.
# --------------------------------------------------------------------------------------


def test_obligation7_run_tilt_writes_contract_hash_seed_and_git_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nifty_quant.research.registry import TrialRecord, TrialRegistry
    from nifty_quant.research.tilt import TiltConfig, run_tilt

    dates = [dt.date(2022, 1, 3) + dt.timedelta(days=index) for index in range(20)]
    loser_of_day = [index % 2 for index in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)
    config = TiltConfig(start=dates[0], end=dates[-1], seed=137)
    contract = _contract(
        seed=config.seed,
        data={"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
    )
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    written: list[TrialRecord] = []
    original_record = registry.record

    def capture(record: TrialRecord) -> None:
        written.append(record)
        original_record(record)

    monkeypatch.setattr(registry, "record", capture)

    run_tilt(panel, config, contract=contract, registry=registry)

    assert written
    record = written[-1]
    assert isinstance(record, TrialRecord)
    assert record.contract_hash == contract_hash(contract)
    assert record.seed is not None
    assert record.git_sha is not None


# --------------------------------------------------------------------------------------
# Obligation 8: a sweep declaring n_planned_trials=k that attempts trial k+1 raises.
# --------------------------------------------------------------------------------------


def _attempt_declared_sweep(contract: ResearchContract, attempted_trials: int) -> None:
    for trial_number in range(1, attempted_trials + 1):
        contract.check_trial_count(trial_number)


def test_obligation8_sweep_trial_k_plus_one_is_rejected() -> None:
    contract = _contract()
    planned_trials = contract.validation["n_planned_trials"]

    _attempt_declared_sweep(contract, planned_trials)

    with pytest.raises(ValueError):
        _attempt_declared_sweep(contract, planned_trials + 1)


# --------------------------------------------------------------------------------------
# Obligation 9: holdout intent tri-state must match --allow-holdout.
# --------------------------------------------------------------------------------------


def test_obligation9_never_holdout_with_allow_flag_is_refused() -> None:
    contract = _contract(validation={"holdout_intent": "never"})

    with pytest.raises(ValueError):
        contract.check_holdout_intent(allow_holdout=True)


def test_obligation9_reading_now_without_allow_flag_is_refused() -> None:
    contract = _contract(validation={"holdout_intent": "reading_now"})

    with pytest.raises(ValueError):
        contract.check_holdout_intent(allow_holdout=False)


def test_obligation9_matching_intent_and_flag_do_not_raise() -> None:
    """Edge case implied by the spec's "disagreement ... is refused" wording: agreement must NOT
    be refused, else the whole tri-state gate would be indistinguishable from an unconditional
    refusal."""
    never_contract = _contract(validation={"holdout_intent": "never"})
    reading_now_contract = _contract(validation={"holdout_intent": "reading_now"})

    never_contract.check_holdout_intent(allow_holdout=False)
    reading_now_contract.check_holdout_intent(allow_holdout=True)


# --------------------------------------------------------------------------------------
# Obligation 10: regression test for P4 -- every TrialRecord written by walkforward,
# INCLUDING the pooled record, has all 12 Amendment-1 provenance fields populated.
# Goes through the real `nq walkforward` CLI path per explicit direction.
# --------------------------------------------------------------------------------------


def _walkforward_cli_args(app, start: dt.date, end: dt.date, contract_path: Path) -> list[str]:
    import click
    from typer.main import get_command

    root_command = get_command(app)
    walkforward_command = root_command.commands["walkforward"]
    unset = object()
    args = ["walkforward"]

    for parameter in walkforward_command.params:
        name = parameter.name.replace("-", "_")
        value: object = unset

        if name in {"start", "start_date", "from_date"}:
            value = start.isoformat()
        elif name in {"end", "end_date", "to_date"}:
            value = end.isoformat()
        elif name in {"contract", "contract_path", "contract_file"}:
            value = str(contract_path)
        elif name in {"universe", "universe_name"}:
            value = "all_equity"
        elif name in {"panel", "panel_id", "panel_spec", "data"}:
            value = "synthetic"
        elif name in {
            "train_days", "training_days", "train_window", "train_window_days", "train_size",
        }:
            value = "10"
        elif name in {"test_days", "testing_days", "test_window", "test_window_days", "test_size"}:
            value = "5"
        elif name in {"step_days", "step", "step_size", "rolling_step_days"}:
            value = "5"
        elif name in {"n_splits", "splits", "num_splits"}:
            value = "2"
        elif name in {"purge_days", "purge_width"}:
            value = "1"
        elif name in {"embargo_days", "embargo_width"}:
            value = "3"
        elif name in {"seed", "random_seed"}:
            value = "137"
        elif name in {"allow_holdout"}:
            value = False
        elif name in {"params", "params_json", "parameters"}:
            value = "{}"
        elif name in {"strategy", "strategy_name", "strategy_id"}:
            if parameter.required or parameter.default is None:
                choices = getattr(parameter.type, "choices", ())
                value = next(iter(choices), "constant_weight")
        elif parameter.required:
            type_name = getattr(parameter.type, "name", "")
            if type_name in {"integer", "int"}:
                value = "1"
            elif type_name in {"float"}:
                value = "1.0"
            elif type_name in {"boolean", "bool"}:
                value = False
            else:
                value = "synthetic"

        if value is unset:
            continue

        if isinstance(parameter, click.Option):
            option_names = [option for option in parameter.opts if option.startswith("--")]
            option_name = option_names[0] if option_names else parameter.opts[0]
            if parameter.is_flag:
                if bool(value):
                    args.append(option_name)
            else:
                args.extend([option_name, str(value)])
        else:
            args.append(str(value))

    return args


def test_obligation10_walkforward_populates_provenance_on_split_and_pooled_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod
    from nifty_quant.cli import app
    from nifty_quant.research.registry import TrialRecord, TrialRegistry
    from nifty_quant.universe.static import Universe

    dates = [dt.date(2022, 1, 3) + dt.timedelta(days=index) for index in range(20)]
    loser_of_day = [index % 2 for index in range(20)]
    panel = _cycling_aggressive_panel(dates, loser_of_day)

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="all_equity", symbols=_SYMBOLS)
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps({**_full_sections(), "seed": _DEFAULT_SEED}), encoding="utf-8"
    )

    records: list[TrialRecord] = []
    original_record = TrialRegistry.record

    def capture(record_registry: TrialRegistry, record: TrialRecord) -> None:
        records.append(record)
        original_record(record_registry, record)

    monkeypatch.setattr(TrialRegistry, "record", capture)

    runner = CliRunner()
    result = runner.invoke(app, _walkforward_cli_args(app, dates[0], dates[-1], contract_path))

    assert result.exit_code == 0, result.output
    assert records
    assert any(record.split_id.casefold() == "pooled" for record in records)
    assert any(record.split_id.casefold() != "pooled" for record in records)

    provenance_fields = (
        "seed",
        "universe_name",
        "universe_hash",
        "panel_hash",
        "start",
        "end",
        "cost_model_id",
        "slippage_model_id",
        "fill_model_id",
        "embargo_components",
        "parent_trial_id",
        "feature_version",
    )

    for record in records:
        for field_name in provenance_fields:
            value = getattr(record, field_name)
            assert value is not None, field_name
            if isinstance(value, str):
                assert value not in {"", "{}"}, field_name


# --------------------------------------------------------------------------------------
# Obligation 11: research/sweep.py's dead-code config_hash no longer exists.
# --------------------------------------------------------------------------------------


def test_obligation11_research_sweep_has_no_config_hash() -> None:
    import nifty_quant.research.sweep as sweep_mod

    assert not hasattr(sweep_mod, "config_hash")
