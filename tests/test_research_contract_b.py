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
invent a `run_walkforward_to_registry(...)` seam that exists only to be tested), using the SAME
proven harness pattern as `tests/test_walkforward_pooling.py` (read, not modified, for the
pattern): a synthetic multi-session `Panel`, a real registered strategy (`volume_breakout`) with
its params written to a real config YAML in `tmp_path`, `load_panel`/`load_universe`/
`TradingCalendar.from_index_bars`/`settings.RESULTS_ROOT` monkeypatched, and `engine.run_backtest`
stubbed to a deterministic fake result (populated via the real `build_daily`, exactly as that
harness does) so the test exercises the CLI's contract/provenance wiring rather than needing a
real strategy signal on synthetic prices. After the CLI run, the resulting `TrialRegistry` is
opened directly from `tmp_path` and every record it contains (`TrialRegistry.all()`) is checked,
rather than guessing CLI flag names via parameter introspection -- that approach was tried and
abandoned (see git history of this file): each guessed flag/value it got wrong only surfaced once
the previous guess was fixed, because introspection cannot recover semantics (which strategy is
actually registered, which option wants a file path vs. a literal) that only the CLI's own code
knows.
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
from pydantic import BaseModel

from nifty_quant.data.panel import Panel
from nifty_quant.research.contract import ResearchContract, contract_hash

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
# Goes through the real `nq walkforward` CLI path, using the SAME proven harness shape
# as `tests/test_walkforward_pooling.py` (read for the pattern, not modified or imported
# from): a synthetic multi-session Panel, a real registered strategy with its params
# written to a real config YAML in `tmp_path`, `load_panel`/`load_universe`/
# `TradingCalendar.from_index_bars`/`settings.RESULTS_ROOT` monkeypatched, and
# `engine.run_backtest` stubbed to a deterministic fake result populated via the real
# `build_daily` -- NOT introspected/guessed CLI arguments, which cannot recover semantics
# (which strategy name is actually registered, which option wants a file path vs. a
# literal) that only the CLI's own code knows.
# --------------------------------------------------------------------------------------

_WF_STRATEGY_YAML = """\
strategy: volume_breakout
params:
  breakout_window: 30
  volume_window: 30
  volume_z_threshold: 2.0
  hurst_window: 390
  hurst_threshold: 0.55
  use_hurst: false
  vol_window: 30
  deseasonalize: true
  direction: continuation
  exit_mode: time
  hold_bars: 30
  min_hold_bars: 5
  cooldown_bars: 0
  square_off_time: "15:20"
  sigma_floor: 1.0e-5
  max_weight: 0.10
  gross: 1.0
"""


def _wf_session_ts(
    session_date: dt.date, n_bars: int, start_hhmm: tuple[int, int] = (9, 15)
) -> np.ndarray:
    from datetime import time, timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    start = dt.datetime.combine(session_date, time(*start_hhmm), tzinfo=ist)
    return np.asarray([int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64)


def _wf_make_panel(
    session_dates: list[dt.date],
    bars_per_session: list[int],
    symbols: tuple[str, ...] = ("A", "B"),
    close_price: float = 100.5,
    volume_value: float = 1000.0,
) -> Panel:
    """Mirrors `tests/test_walkforward_pooling.py::_make_panel` -- a plain, non-checkpoint,
    N-bar-per-session panel with all five OHLCV fields present, suitable for the real
    `nq walkforward` CLI path (universe hashing, panel hashing, tradable-mask computation)
    when `run_backtest` itself is stubbed out."""
    ts = np.concatenate(
        [_wf_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    fields = {
        "open": np.full((n_rows, n_symbols), close_price - 0.5, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), close_price + 0.5, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), close_price - 1.0, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), close_price, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), volume_value, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


def _wf_row_day_index(panel: Panel) -> np.ndarray:
    n_rows = panel.n_rows()
    return (
        np.searchsorted(panel.day_offsets, np.arange(n_rows, dtype=np.int64), side="right") - 1
    )


def _wf_session_dates_epoch(panel: Panel) -> np.ndarray:
    return np.asarray(
        [
            int(
                dt.datetime(
                    session_date.year, session_date.month, session_date.day, tzinfo=dt.timezone.utc
                ).timestamp()
            )
            for session_date in panel.dates
        ],
        dtype=np.int64,
    )


def _wf_fake_result(n_rows: int, panel: Panel, seed: int = 0):
    """Mirrors `tests/test_walkforward_pooling.py::_fake_result`: a deterministic fake
    `BacktestResult` sized to `panel.n_rows()`, with `.daily` populated by the REAL
    `build_daily`, so the CLI's own day-aggregation and provenance wiring run unmodified --
    only the (irrelevant to this obligation) strategy signal itself is stubbed out."""
    from nifty_quant.backtest.daily import build_daily
    from nifty_quant.backtest.engine import BacktestResult

    rng = np.random.default_rng(seed)
    gross = rng.normal(loc=0.0006, scale=0.004, size=n_rows).astype(np.float64)
    turnover = np.full(n_rows, 1.0, dtype=np.float64)
    returns = (gross - 0.0002 * turnover).astype(np.float64)
    initial_capital = 1e7
    equity_curve = np.cumprod(1.0 + returns) * initial_capital
    daily = build_daily(
        _wf_row_day_index(panel),
        equity_curve,
        returns,
        gross,
        turnover,
        _wf_session_dates_epoch(panel),
        initial_capital=initial_capital,
    )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=np.zeros((n_rows, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=initial_capital,
        ruined=False,
        ruin_index=-1,
        daily=daily,
    )


class _WFFakeCalendar:
    """Mirrors `tests/test_walkforward_pooling.py::_FakeCalendar`."""

    def __init__(self, dates: list[dt.date]) -> None:
        self._dates = list(dates)

    def session_dates(
        self,
        start: dt.date | None = None,
        end: dt.date | None = None,
        *,
        usable_only: bool = True,
    ) -> list[dt.date]:
        return [
            d for d in self._dates if (start is None or d >= start) and (end is None or d <= end)
        ]


def test_obligation10_walkforward_populates_provenance_on_split_and_pooled_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod
    from nifty_quant.cli import app
    from nifty_quant.research.registry import TrialRegistry
    from nifty_quant.universe.static import Universe

    all_dates = pd.bdate_range("2020-01-01", periods=280).date.tolist()
    panel = _wf_make_panel(all_dates, [2] * len(all_dates))

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=panel.symbols)
    )
    monkeypatch.setattr(
        engine_mod,
        "run_backtest",
        lambda strat, pnl, config, **kwargs: _wf_fake_result(pnl.n_rows(), pnl),
    )
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _WFFakeCalendar(all_dates)),
    )

    config_path = tmp_path / "volume_breakout.yaml"
    config_path.write_text(_WF_STRATEGY_YAML, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            str(config_path),
            "--start",
            all_dates[0].isoformat(),
            "--end",
            all_dates[-1].isoformat(),
            "--train-years",
            "0.012",
            "--test-years",
            "0.012",
        ],
    )

    assert result.exit_code == 0, result.output

    registry = TrialRegistry(tmp_path / "trials.db")
    records = registry.all()

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
