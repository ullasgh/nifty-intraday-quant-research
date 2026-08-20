"""Independent test suite A for specs/research_contract.md (Spec C4 -- ResearchContract).

Written from the spec ALONE, before any implementation exists. `ResearchContract` does not
exist yet, `run_backtest()` and `run_tilt()` do not yet require a contract argument, and
`research/sweep.py:config_hash` has not yet been deleted. This file is EXPECTED to be RED
(ImportError / AttributeError / signature mismatches) until that implementation lands. That is
the correct and intended state for a tests-first suite -- do not weaken any assertion to make
it pass early.

Updated per specs/research_contract.md AMENDMENT 1 (2026-08-20), which resolves three
ambiguities this suite's first draft reported rather than silently guessed around:

1. `seed` is a REQUIRED TOP-LEVEL field of `ResearchContract` (sibling to the six sections,
   included in `contract_hash`), NOT a member of `execution`. The first draft filed it under
   `execution` as a documented assumption; the amendment rejects that reading because seed
   cuts across sections (bootstrap/permutation nulls and CSCV split assignment in
   `validation`, stochastic tie-breaks in `portfolio`/`execution`) and a single-section home
   would misstate which stage it belongs to.
2. The module is `nifty_quant.research.contract`, confirmed by the amendment (matches
   `research/tilt.py`, `research/registry.py`, `research/splits.py`).
3. Obligation 10 is tested through the CLI (`nq walkforward` via `CliRunner`), not through an
   invented `run_walkforward_to_registry(...)` seam -- walkforward writes its records inline
   in `cli.py` and exposes no callable seam, and the amendment is explicit that one must not
   be added purely to be testable.

Section contents are modelled as plain dicts (the spec's table gives contents as bullet lists,
not a formal schema), keyed by the field names spelled out in the spec's table verbatim.
"""

from __future__ import annotations

import datetime
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nifty_quant.research.sweep as sweep_module

# --- everything else needs the not-yet-existing contract module --------------------------

try:
    from nifty_quant.research.contract import ResearchContract

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via _IMPORT_ERROR checks below
    ResearchContract = None  # type: ignore[assignment, misc]
    _IMPORT_ERROR = exc


def _require_contract() -> None:
    if ResearchContract is None:
        pytest.fail(
            f"nifty_quant.research.contract.ResearchContract does not exist yet "
            f"(import failed with: {_IMPORT_ERROR!r}). This is the expected RED state "
            f"of a tests-first suite -- implement ResearchContract to turn this green."
        )


# ---------------------------------------------------------------------------------------
# Section builders -- one canonical valid value per section, per the spec's table.
# ---------------------------------------------------------------------------------------


def _data_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "panel_id": "nifty50_1m_v1",
        "panel_hash": "a" * 16,
        "start": "2022-01-03",
        "end": "2022-06-30",
        "bar_interval_s": 60,
        "universe_name": "all_equity",
        "universe_hash": "b" * 16,
    }
    base.update(overrides)
    return base


def _features_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "feature_ids": [{"id": "overnight_gap", "params": {"lookback": 1}}],
        "feature_version": "v1",
    }
    base.update(overrides)
    return base


def _label_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "horizon_bars": 1,
        "construction": "forward_return",
        "overlapping": False,
    }
    base.update(overrides)
    return base


def _execution_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "cost_model_id": "nse_intraday_default",
        "slippage_model_id": "none",
        "decision_latency_bars": 0,
        "participation_cap": 0.1,
    }
    base.update(overrides)
    return base


def _portfolio_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "sizing_scheme": "clipped_rank",
        "gross_clip": 1.0,
        "max_weight": 0.2,
        "target_vol": None,
    }
    base.update(overrides)
    return base


def _validation_section(**overrides: Any) -> dict[str, Any]:
    base = {
        "split_scheme": "rolling",
        "purge_width_bars": 5,
        "embargo_width_bars": 5,
        "n_planned_trials": 10,
        "holdout_intent": "never",
    }
    base.update(overrides)
    return base


def _valid_sections(**section_overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = {
        "data": _data_section(),
        "features": _features_section(),
        "label": _label_section(),
        "execution": _execution_section(),
        "portfolio": _portfolio_section(),
        "validation": _validation_section(),
    }
    sections.update(section_overrides)
    return sections


def _make_contract(seed: int = 0, **section_overrides: dict[str, Any]) -> Any:
    """Build a valid contract. `seed` is a TOP-LEVEL field per AMENDMENT 1 point 1 -- it is
    NOT part of any of the six section dicts, so it is a dedicated keyword here rather than
    something callers stuff into `execution` or `validation`."""
    _require_contract()
    return ResearchContract(seed=seed, **_valid_sections(**section_overrides))


# ---------------------------------------------------------------------------------------
# Obligation 1: constructing a contract with any of the six sections missing raises; the
# message names the missing section. One test per section (6 sub-cases in one parametrised
# test, per the "one assertion each" spirit of obligation 3's instruction, applied here too
# since it is the same "don't collapse into one case" concern).
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_section",
    ["data", "features", "label", "execution", "portfolio", "validation"],
)
def test_missing_section_raises_and_names_it(missing_section: str) -> None:
    _require_contract()
    sections = _valid_sections()
    del sections[missing_section]
    with pytest.raises((ValueError, TypeError)) as excinfo:
        ResearchContract(**sections)
    assert missing_section in str(excinfo.value), (
        f"error message must name the missing section {missing_section!r}, "
        f"got: {excinfo.value!r}"
    )


def test_missing_section_via_explicit_none_also_raises() -> None:
    """A section explicitly passed as None is still "missing" -- the spec says every field
    must be supplied or explicitly declared absent, but that concerns FIELDS within a
    section, not whole sections. A whole section cannot be "explicitly absent"."""
    _require_contract()
    sections = _valid_sections(data=None)  # type: ignore[arg-type]
    with pytest.raises((ValueError, TypeError)) as excinfo:
        ResearchContract(**sections)
    assert "data" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# Obligation 2: contract_hash is stable across process restarts and insensitive to key
# ordering within a section.
# ---------------------------------------------------------------------------------------


def test_contract_hash_is_deterministic_across_independent_constructions() -> None:
    """Stands in for "stable across process restarts": two independently built contracts
    from freshly-constructed section dicts (no shared object identity) must hash equal."""
    contract_1 = _make_contract()
    contract_2 = _make_contract()
    assert contract_1.contract_hash == contract_2.contract_hash


def test_contract_hash_insensitive_to_key_ordering_within_a_section() -> None:
    forward_order = {
        "cost_model_id": "nse_intraday_default",
        "slippage_model_id": "none",
        "decision_latency_bars": 0,
        "participation_cap": 0.1,
    }
    reversed_order = dict(reversed(list(forward_order.items())))
    assert list(forward_order.keys()) != list(reversed_order.keys())

    contract_forward = _make_contract(execution=forward_order)
    contract_reversed = _make_contract(execution=reversed_order)
    assert contract_forward.contract_hash == contract_reversed.contract_hash


def test_contract_hash_is_a_stable_string_not_recomputed_randomly() -> None:
    """Calling contract_hash twice on the SAME instance must return the identical value
    (guards against a hash that salts itself with e.g. an uncontrolled timestamp)."""
    contract = _make_contract()
    first = contract.contract_hash
    second = contract.contract_hash
    assert first == second
    assert isinstance(first, str)
    assert len(first) > 0


# ---------------------------------------------------------------------------------------
# Obligation 3: contract_hash CHANGES when any one of: universe, start, end, cost model id,
# seed, or embargo width changes. One test per field, six separate tests -- this is the P2
# regression and the spec is explicit that one combined case is not acceptable.
# ---------------------------------------------------------------------------------------


def test_contract_hash_changes_with_universe() -> None:
    base = _make_contract()
    changed = _make_contract(data=_data_section(universe_name="banking_only"))
    assert base.contract_hash != changed.contract_hash


def test_contract_hash_changes_with_start() -> None:
    base = _make_contract()
    changed = _make_contract(data=_data_section(start="2022-02-01"))
    assert base.contract_hash != changed.contract_hash


def test_contract_hash_changes_with_end() -> None:
    base = _make_contract()
    changed = _make_contract(data=_data_section(end="2022-07-31"))
    assert base.contract_hash != changed.contract_hash


def test_contract_hash_changes_with_cost_model_id() -> None:
    base = _make_contract()
    changed = _make_contract(execution=_execution_section(cost_model_id="delivery_default"))
    assert base.contract_hash != changed.contract_hash


def test_contract_hash_changes_with_seed() -> None:
    """seed is a TOP-LEVEL field per AMENDMENT 1 point 1, not a section member."""
    base = _make_contract(seed=0)
    changed = _make_contract(seed=1)
    assert base.contract_hash != changed.contract_hash


def test_contract_hash_changes_with_embargo_width() -> None:
    base = _make_contract()
    changed = _make_contract(validation=_validation_section(embargo_width_bars=10))
    assert base.contract_hash != changed.contract_hash


# ---------------------------------------------------------------------------------------
# Obligation 4: two contracts differing ONLY in a field the old strategy/registry.py:62 hash
# ignored (universe, dates, costs, seed) produce DIFFERENT hashes. Regression test for P2,
# phrased as the old hash would have seen it: strategy+params identical, everything else
# varies one at a time.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_section, overrides, description",
    [
        ("data", {"universe_name": "banking_only"}, "universe"),
        ("data", {"start": "2022-03-01"}, "start date"),
        ("data", {"end": "2022-08-31"}, "end date"),
        ("execution", {"cost_model_id": "delivery_default"}, "cost model"),
    ],
)
def test_p2_regression_old_registry_hash_blind_spots_now_change_the_hash(
    field_section: str, overrides: dict[str, Any], description: str
) -> None:
    """Old `strategy/registry.py:62` config_hash only covered strategy+params, so it would
    have collided on every one of these variations. contract_hash must not repeat that bug."""
    section_builder = {
        "data": _data_section,
        "execution": _execution_section,
    }[field_section]

    base = _make_contract()
    changed = _make_contract(**{field_section: section_builder(**overrides)})
    assert base.contract_hash != changed.contract_hash, (
        f"contract_hash must be sensitive to {description}, reproducing the P2 collision "
        f"if it is not"
    )


def test_p2_regression_seed_field_now_changes_the_hash() -> None:
    """Same P2 regression as above, for `seed`. Kept as its own test rather than folded into
    the parametrised case above because seed is a TOP-LEVEL field per AMENDMENT 1 point 1,
    not a section member -- it needs the top-level `seed=` keyword, not a section override."""
    base = _make_contract(seed=0)
    changed = _make_contract(seed=42)
    assert base.contract_hash != changed.contract_hash, (
        "contract_hash must be sensitive to seed, reproducing the P2 collision if it is not"
    )


# ---------------------------------------------------------------------------------------
# Obligation 5: run_backtest() without a contract raises.
# ---------------------------------------------------------------------------------------


def test_run_backtest_signature_requires_contract_argument() -> None:
    """A required-and-defaultless `contract` parameter on run_backtest is the mechanism by
    which calling it without a contract raises (TypeError: missing required argument).
    Checked via introspection rather than a full synchronous run, because run_backtest needs
    a fully constructed Panel/Strategy/BacktestConfig that are orthogonal to what this
    obligation is actually testing (contract enforcement, not backtest mechanics)."""
    from nifty_quant.backtest.engine import run_backtest

    sig = inspect.signature(run_backtest)
    assert "contract" in sig.parameters, (
        "run_backtest must accept a `contract` parameter -- spec section 'Enforcement', "
        "gate (1)"
    )
    assert sig.parameters["contract"].default is inspect.Parameter.empty, (
        "`contract` must be required (no default) so omitting it raises immediately, "
        "per obligation 5"
    )


def test_run_backtest_called_without_contract_keyword_raises() -> None:
    """Behavioural companion to the signature test: actually calling run_backtest with the
    contract explicitly withheld must raise before doing any backtest work. We pass garbage
    for the other positional arguments deliberately -- if contract enforcement fires FIRST
    (as it must), the garbage is never touched and any TypeError/ValueError is acceptable."""
    from nifty_quant.backtest.engine import run_backtest

    with pytest.raises((TypeError, ValueError)):
        run_backtest(strategy=None, panel=None, config=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# Obligation 6 (MOST IMPORTANT): run_tilt() without a contract raises. This is the
# regression test for P1 -- tilt is the only construction that has beaten the index net of
# costs, and it currently bypasses the entire research spine (zero references to
# TrialRegistry/TrialRecord/run_backtest in research/tilt.py). Both a static (signature) and
# a dynamic (actual call, with a real minimal panel) check are given: the dynamic one proves
# the enforcement fires BEFORE any tilt computation happens, using a real panel that would
# otherwise produce a valid TiltResult if the gate were bypassed.
# ---------------------------------------------------------------------------------------


def _flat_row(base: float, n: int) -> np.ndarray:
    return np.array([base + i * 0.01 for i in range(n)], dtype=np.float64)


def _minimal_tilt_panel() -> Any:
    """A tiny, real, valid Panel: 5 symbols, 20 sessions, two bars/session (entry 09:16,
    exit 15:20, IST, matching TiltConfig's defaults), sufficient for run_tilt to actually
    execute a full computation if contract enforcement did NOT fire. Bars are left-labelled
    per CLAUDE.md rule 4; sessions are indexed by explicit day offsets, never a fixed
    row-count assumption, per rule 5."""
    from zoneinfo import ZoneInfo

    import pandas as pd

    from nifty_quant.data.panel import Panel

    ist = ZoneInfo("Asia/Kolkata")
    n_symbols = 5
    symbols = tuple(f"SYM{i}" for i in range(n_symbols))
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(20)]

    def ts_for(date: datetime.date, hhmm: str) -> int:
        hour, minute = (int(p) for p in hhmm.split(":"))
        stamp = pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=ist)
        return int(stamp.tz_convert("UTC").timestamp())

    ts_list: list[int] = []
    price_rows: list[np.ndarray] = []
    day_offsets = [0]
    row = 0
    for date in dates:
        entry_row = _flat_row(100.0, n_symbols)
        exit_row = entry_row + 0.05
        for hhmm, price_row in (("09:16", entry_row), ("15:20", exit_row)):
            ts_list.append(ts_for(date, hhmm))
            price_rows.append(price_row)
            row += 1
        day_offsets.append(row)

    ts = np.array(ts_list, dtype=np.int64)
    price_arr = np.stack(price_rows).astype(np.float32)
    volume_arr = np.full(price_arr.shape, 1_000_000.0, dtype=np.float32)
    dates_arr = np.array(dates, dtype=object)
    return Panel(
        fields={"open": price_arr, "close": price_arr.copy(), "volume": volume_arr},
        symbols=symbols,
        ts=ts,
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=dates_arr,
    ), dates


def test_run_tilt_signature_requires_contract_argument() -> None:
    from nifty_quant.research.tilt import run_tilt

    sig = inspect.signature(run_tilt)
    assert "contract" in sig.parameters, (
        "run_tilt must accept a `contract` parameter -- spec section 'Enforcement', "
        "gate (2). research/tilt.py currently has ZERO references to the research spine "
        "(P1); this is the fix."
    )
    assert sig.parameters["contract"].default is inspect.Parameter.empty, (
        "`contract` must be required (no default) so omitting it raises immediately, "
        "per obligation 6"
    )


def test_run_tilt_without_contract_raises_before_computing_a_result() -> None:
    """The regression test for P1. Uses a real, valid, minimal panel + TiltConfig that WOULD
    produce a normal TiltResult if the contract gate were bypassed -- so a pass here proves
    enforcement, not an accident of malformed input."""
    from nifty_quant.research.tilt import TiltConfig, run_tilt

    panel, dates = _minimal_tilt_panel()
    config = TiltConfig(start=dates[0], end=dates[-1], tilt="aggressive", smoothing=1.0)

    with pytest.raises((TypeError, ValueError)):
        run_tilt(panel, config)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------------------
# Obligation 7: a completed run_tilt() writes a TrialRecord whose contract_hash matches the
# declared contract, with a non-null seed and a non-null git_sha.
# ---------------------------------------------------------------------------------------


def test_run_tilt_with_contract_writes_matching_trial_record(tmp_path: Path) -> None:
    from nifty_quant.research.registry import TrialRegistry
    from nifty_quant.research.tilt import TiltConfig, run_tilt

    panel, dates = _minimal_tilt_panel()
    config = TiltConfig(
        start=dates[0], end=dates[-1], tilt="aggressive", smoothing=1.0, seed=7
    )
    contract = _make_contract(seed=7)

    registry_path = tmp_path / "trials.sqlite"
    registry = TrialRegistry(registry_path)

    run_tilt(  # type: ignore[call-arg]
        panel, config, contract=contract, registry=registry
    )

    records = registry.all()
    assert len(records) >= 1, "run_tilt must write at least one TrialRecord"
    matching = [r for r in records if r.contract_hash == contract.contract_hash]  # type: ignore[attr-defined]
    assert matching, (
        f"no TrialRecord's contract_hash matches the declared contract "
        f"({contract.contract_hash}); got hashes: "
        f"{[getattr(r, 'contract_hash', None) for r in records]}"
    )
    for record in matching:
        assert record.seed is not None, "TrialRecord.seed must be non-null"
        assert record.git_sha is not None, "TrialRecord.git_sha must be non-null"


# ---------------------------------------------------------------------------------------
# Obligation 8: a sweep declaring n_planned_trials=k that attempts trial k+1 raises.
# ---------------------------------------------------------------------------------------


def test_sweep_exceeding_n_planned_trials_raises() -> None:
    """Whatever mechanism the sweep uses to enforce this (a method on ResearchContract, or a
    counter object it returns), the contract itself is the declared source of truth for
    n_planned_trials, so this suite exercises it at that level: attempting to register a
    (k+1)th trial against a contract declaring n_planned_trials=k must raise."""
    _require_contract()
    contract = _make_contract(validation=_validation_section(n_planned_trials=2))
    assert contract.validation["n_planned_trials"] == 2  # type: ignore[attr-defined]

    register = getattr(contract, "register_trial", None)
    assert register is not None, (
        "ResearchContract must expose a way to check/register realised trials against "
        "n_planned_trials (e.g. a `register_trial()` method) -- obligation 8 requires "
        "attempting trial k+1 to raise, and nothing else in the spec names the mechanism"
    )
    register()
    register()
    with pytest.raises(ValueError):
        register()


# ---------------------------------------------------------------------------------------
# Obligation 9: holdout intent = never combined with --allow-holdout is refused, and so is
# reading_now without the flag.
# ---------------------------------------------------------------------------------------


def test_holdout_intent_never_with_allow_holdout_flag_is_refused() -> None:
    _require_contract()
    contract = _make_contract(validation=_validation_section(holdout_intent="never"))
    check = getattr(contract, "check_holdout_intent", None)
    assert check is not None, (
        "ResearchContract must expose a way to validate holdout_intent against the "
        "--allow-holdout flag (e.g. check_holdout_intent(allow_holdout: bool))"
    )
    with pytest.raises(ValueError):
        check(allow_holdout=True)


def test_holdout_intent_reading_now_without_allow_holdout_flag_is_refused() -> None:
    _require_contract()
    contract = _make_contract(
        validation=_validation_section(holdout_intent="reading_now")
    )
    check = getattr(contract, "check_holdout_intent", None)
    assert check is not None
    with pytest.raises(ValueError):
        check(allow_holdout=False)


def test_holdout_intent_matching_flag_is_accepted() -> None:
    """Companion sanity check: the tri-state is not simply rejecting everything. Both
    matching combinations (never/False and reading_now/True) must NOT raise."""
    _require_contract()
    never_contract = _make_contract(
        validation=_validation_section(holdout_intent="never")
    )
    reading_now_contract = _make_contract(
        validation=_validation_section(holdout_intent="reading_now")
    )
    never_contract.check_holdout_intent(allow_holdout=False)  # type: ignore[attr-defined]
    reading_now_contract.check_holdout_intent(allow_holdout=True)  # type: ignore[attr-defined]


def test_holdout_intent_rejects_unknown_value() -> None:
    """The spec names exactly three states: never, after_conditions_close, reading_now. A
    fourth string is not a silently-accepted extension."""
    with pytest.raises((ValueError, TypeError)):
        _make_contract(
            validation=_validation_section(holdout_intent="sometimes_maybe")
        )


# ---------------------------------------------------------------------------------------
# Obligation 10: every TrialRecord written by walkforward, including the POOLED record, has
# all 12 Amendment-1 provenance fields populated. Regression test for P4.
# ---------------------------------------------------------------------------------------

_AMENDMENT_1_PROVENANCE_FIELDS = (
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


def test_amendment_1_field_list_has_exactly_twelve_fields() -> None:
    """Sanity-checks this suite's own list against TrialRecord's actual dataclass fields
    (VERIFIED FACTS: 31 fields total, 19 original + 12 provenance). If TrialRecord's field
    set ever drifts from this list, this test (not the walkforward one below) is the one
    that explains why."""
    from nifty_quant.research.registry import TrialRecord

    field_names = {f.name for f in __import__("dataclasses").fields(TrialRecord)}
    assert len(_AMENDMENT_1_PROVENANCE_FIELDS) == 12
    for name in _AMENDMENT_1_PROVENANCE_FIELDS:
        assert name in field_names, (
            f"expected Amendment-1 provenance field {name!r} on TrialRecord, "
            f"got fields: {sorted(field_names)}"
        )


def _wf_session_ts(session_date: datetime.date, n_bars: int) -> np.ndarray:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(
        session_date, datetime.time(9, 15), tzinfo=ist
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64
    )


def _wf_panel(all_dates: list[datetime.date], bars_per_session: list[int]) -> Any:
    """A synthetic panel spanning `all_dates`, `bars_per_session[i]` left-labelled bars per
    session, indexed by explicit day offsets (CLAUDE.md rule 5, never a fixed row count).
    Modelled after tests/test_walkforward_pooling.py's `_make_panel`, self-contained here
    rather than imported, per the task's "do not modify/import any other test file" rule."""
    from nifty_quant.data.panel import Panel

    ts = np.concatenate(
        [_wf_session_ts(d, n) for d, n in zip(all_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    n_rows = int(ts.size)
    symbols = ("A", "B")
    n_symbols = len(symbols)
    close_price = 100.5
    fields = {
        "open": np.full((n_rows, n_symbols), close_price - 0.5, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), close_price + 0.5, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), close_price - 1.0, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), close_price, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), 1000.0, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.asarray(all_dates, dtype=object))


def _wf_row_day_index(panel: Any) -> np.ndarray:
    n_rows = panel.n_rows()
    return (
        np.searchsorted(panel.day_offsets, np.arange(n_rows, dtype=np.int64), side="right")
        - 1
    )


def _wf_session_dates_epoch(panel: Any) -> np.ndarray:
    return np.asarray(
        [
            int(
                datetime.datetime(
                    d.year, d.month, d.day, tzinfo=datetime.timezone.utc
                ).timestamp()
            )
            for d in panel.dates
        ],
        dtype=np.int64,
    )


def _wf_fake_backtest_result(panel: Any, seed: int = 0) -> Any:
    """A deterministic fake BacktestResult, correctly sized to `panel.n_rows()`, with a real
    (not stubbed) `daily` populated via `build_daily` -- only `run_backtest` itself is
    faked, so everything downstream (the day-basis pooling under test) runs unmodified."""
    from nifty_quant.backtest.daily import build_daily
    from nifty_quant.backtest.engine import BacktestResult

    n_rows = panel.n_rows()
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
    import pandas as pd

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


def _run_walkforward_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, panel: Any, all_dates: list[datetime.date]
) -> Any:
    """Invoke `nq walkforward` end to end via CliRunner, per AMENDMENT 1 point 3: this
    exercises the exact path production takes (inline record-writing in cli.py), rather
    than an invented injectable-registry seam. Only `run_backtest`, `load_panel`,
    `load_universe`, `TradingCalendar.from_index_bars` are stubbed; `settings.RESULTS_ROOT`
    is already redirected to `tmp_path` by the suite-wide autouse fixture in conftest.py,
    and is repeated here explicitly for robustness/clarity."""
    from typer.testing import CliRunner

    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod
    from nifty_quant.universe.static import Universe

    class _FakeCalendar:
        def __init__(self, dates: list[datetime.date]) -> None:
            self._dates = list(dates)

        def session_dates(
            self,
            start: datetime.date | None = None,
            end: datetime.date | None = None,
            *,
            usable_only: bool = True,
        ) -> list[datetime.date]:
            return [
                d
                for d in self._dates
                if (start is None or d >= start) and (end is None or d <= end)
            ]

    def _stub_run_backtest(strat: Any, pnl: Any, config: Any, **kwargs: Any) -> Any:
        return _wf_fake_backtest_result(pnl)

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=panel.symbols)
    )
    monkeypatch.setattr(engine_mod, "run_backtest", _stub_run_backtest)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)),
    )

    from nifty_quant.cli import app

    runner = CliRunner()
    return runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
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


def test_walkforward_pooled_record_has_all_amendment_1_fields_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for P4, per AMENDMENT 1 point 3: `cli.py`'s pooled record
    (~1268-1289) currently populates NONE of the 12 Amendment-1 provenance fields, even
    though it represents the corrected, validated verdict a reader would trust most. This
    invokes the REAL `nq walkforward` command end to end (only `run_backtest`/panel/universe/
    calendar loading are stubbed) and inspects the pooled row (`split_id == "pooled"`) it
    writes to `settings.RESULTS_ROOT / "trials.db"`.
    """
    import pandas as pd

    from nifty_quant.research.registry import TrialRegistry

    all_dates = pd.bdate_range("2020-01-01", periods=280).date.tolist()
    panel = _wf_panel(all_dates, [2] * len(all_dates))

    result = _run_walkforward_cli(monkeypatch, tmp_path, panel, all_dates)
    assert result.exit_code == 0, result.output

    registry = TrialRegistry(tmp_path / "trials.db")
    records = registry.all()
    pooled_records = [r for r in records if getattr(r, "split_id", None) == "pooled"]
    assert pooled_records, (
        f"expected at least one pooled TrialRecord (split_id == 'pooled'); "
        f"got split_ids: {[getattr(r, 'split_id', None) for r in records]}"
    )

    for record in pooled_records:
        for field_name in _AMENDMENT_1_PROVENANCE_FIELDS:
            value = getattr(record, field_name)
            assert value not in (None, "", "{}"), (
                f"pooled record's {field_name!r} is unpopulated ({value!r}) -- "
                f"this is the P4 regression"
            )
        assert record.seed is not None, "pooled record's seed must be non-null"
        assert record.git_sha is not None, "pooled record's git_sha must be non-null"


# ---------------------------------------------------------------------------------------
# Obligation 11: research/sweep.py:config_hash no longer exists.
# ---------------------------------------------------------------------------------------


def test_sweep_config_hash_no_longer_exists() -> None:
    """`research/sweep.py`'s config_hash (line ~124) is dead code -- never imported per the
    spec's verified-current-state table -- and must be deleted rather than left as a fourth
    reading of contract_hash's idea."""
    assert not hasattr(sweep_module, "config_hash"), (
        "nifty_quant.research.sweep.config_hash must be deleted (spec: 'sweep.py's dead "
        "copy is deleted rather than left as a fourth reading of the same idea')"
    )
