"""Tests for specs/run_provenance.md (Phase B track B3), authored from the spec ALONE
by worker-luna (author A of two independent authors; author B's suite is not read).

Written by hand after two `gpt-5.6-luna` API calls on this same prompt both failed to
produce usable output (first: `finish=length` at max_tokens=16000 with the entire
budget consumed by reasoning and zero visible completion tokens; the retry at
max_tokens=40000 was abandoned mid-flight per an explicit orchestrator instruction to
stop polling a backgrounded call and write the file directly instead). No text from
either API call was read or used below.

The implementation this spec describes DOES NOT EXIST YET. Every test here is expected
to fail (RED), either at import/construction time (new `TrialRecord` fields, a new
`nifty_quant.research.provenance` module) or at assertion time (`cli.py` still
hard-codes `git_sha=None` and swallows registry-write exceptions). That is correct.

AMENDMENT 1 (2026-08-20) resolved nine ambiguities in the original spec and WINS over
the body wherever the two disagree; this file was written against the amended spec.

================ SPEC DEFECTS / AMBIGUITIES SURVIVING THE AMENDMENT ================

1. No function/module names are specified for the new computations (git-sha reading,
   panel-hash, model-id canonicalisation). Per the amendment's own item 5/6 fixes, most
   of these are now testable purely as OBSERVABLE OUTPUTS on `TrialRecord` /
   `metrics.json` via the real `nq backtest` / `nq sweep` CLI commands, which this file
   does wherever possible -- avoiding invented internal APIs entirely. The one
   exception is item 2's "dirty tree -> `-dirty` suffix" / "no git -> `no-git`"
   sub-claims, which cannot be exercised through this repo's own (real, and
   non-deterministically dirty/clean) working tree without corrupting it. For those
   two sub-tests only, this file assumes a `nifty_quant.research.provenance.get_git_sha
   (repo_root: Path | None = None) -> str` function exists; a different independent
   author may reasonably invent a different name, and that disagreement is itself a
   finding for the lead to reconcile, per this repo's dual-authorship design.

2. Amendment item 5 ("log the exception at WARNING with the trial id") does not pin an
   exact log message format or logger name, only that a WARNING-level record naming the
   failure is emitted. This file asserts a WARNING record exists and contains the
   injected exception's text; it does NOT independently verify "with the trial id" is
   present, since the spec does not say whether "the trial id" means `config_hash`, the
   `trial_dir` path, or something else, and pinning a specific one would test this
   file's invention rather than the spec.

3. Amendment item 8 ("[sweep's parent_trial_id] points at the sweep's base trial")
   defines "base trial" only as "the trial `base_params` alone would represent" (i.e.
   `strategy_config_hash({"strategy": ..., "params": base_params})`), not as an ACTUAL
   row `nq sweep` is required to write. If the implementation never records a trial for
   `base_params` itself, `parent_trial_id` is a dangling reference to a hash with no
   matching row, which would make "the chain is walkable" (item 8's own requirement,
   now three generations per the amendment) fail in practice for a sweep-derived trial
   even though its literal `parent_trial_id` value is "correct" per this narrower
   definition. This file tests only the literal, spec-mandated claim (the hash value
   matches), not resolvability of that hash to a row, and flags this as a real gap the
   lead should close by requiring sweep to record its base_params trial too.

4. Amendment item 4 lists SIX "same slice" inputs for `panel_hash` (symbol list;
   first+last `ts` as one pair; row count; field names; adjustment settings;
   `PANEL_VERSION`), but `cli.py`'s `backtest`/`sweep` commands expose no
   `--adjustments` flag today, so that axis cannot be exercised through the real CLI
   entry points this file otherwise uses throughout (deliberately, to avoid inventing
   internal APIs -- see point 1). This file exercises the date-range axis (the
   load-bearing one, item 3) and the symbol-list axis (item 3 supplementary) through
   the CLI; `adjustments` and `PANEL_VERSION` sensitivity are left untested here as an
   explicit, reported gap rather than invented around.

5. The spec's own cited `cli.py` line numbers for the seven `git_sha=None` write sites
   (732, 766, 977, 1144, 1332, 1358 -- six numbers given for "seven" sites) are STALE
   against current HEAD. Grepping `git_sha=None` today finds exactly seven occurrences,
   at entirely different line numbers (540, 574, 781, 943, 977, 1127, 1153), confirming
   the count but not the locations. Test-by-symbol, not test-by-line-number, applies
   here too: this file locates the defect by CLI *behaviour* (an actual `nq backtest`
   run's recorded `git_sha`), never by grepping a line number.

6. Amendment item 6's exact 16-key `metrics.json["provenance"]` set is a strict SUBSET
   of `TrialRecord`'s fields (it excludes `strategy`, `params_json`, `split_id`,
   `purpose`, the five performance/outcome fields, `wall_s`, `result_path`, `error`,
   `ruined`, `ruin_index`) -- i.e. "provenance" means identity/lineage only, not a full
   record dump. Not a defect, just noted so the exact-key-set assertion below isn't
   mistaken for an oversight.

7. Since this spec was written, `TrialRecord` gained a `contract_hash` field (spec C4,
   `specs/research_contract.md`) distinct from `config_hash`, and it is written into
   `metrics.json["provenance"]` alongside the original 16 keys. Verified populated
   (non-null) on a real recorded run, not merely present-but-None on the dataclass, so
   `PROVENANCE_KEYS` below is 17 keys, not 16.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from nifty_quant.backtest.daily import build_daily
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import app
from nifty_quant.data.manifest import Manifest
from nifty_quant.data.panel import Panel
from nifty_quant.research.registry import TrialRecord, TrialRegistry
from nifty_quant.strategy.registry import config_hash as strategy_config_hash
from nifty_quant.universe.static import Universe

runner = CliRunner()
_IST = ZoneInfo("Asia/Kolkata")

# The exact 17-key provenance set, amendment item 6, verbatim, plus contract_hash
# (TrialRecord gained this field distinct from config_hash; verified populated below).
PROVENANCE_KEYS = frozenset(
    {
        "config_hash",
        "contract_hash",
        "git_sha",
        "code_version",
        "data_fingerprint",
        "panel_hash",
        "universe_name",
        "universe_hash",
        "seed",
        "start",
        "end",
        "cost_model_id",
        "slippage_model_id",
        "fill_model_id",
        "embargo_components",
        "parent_trial_id",
        "feature_version",
    }
)
NULLABLE_PROVENANCE_KEYS = frozenset({"seed", "parent_trial_id"})

_VALID_VOLUME_BREAKOUT_PARAMS = {
    "breakout_window": 30,
    "volume_window": 30,
    "volume_z_threshold": 2.0,
    "hurst_window": 390,
    "hurst_threshold": 0.55,
    "use_hurst": False,
    "vol_window": 30,
    "deseasonalize": True,
    "direction": "continuation",
    "exit_mode": "time",
    "hold_bars": 30,
    "min_hold_bars": 5,
    "cooldown_bars": 0,
    "square_off_time": "15:20",
    "sigma_floor": 1.0e-5,
    "max_weight": 0.10,
    "gross": 1.0,
}


# ================ shared fixtures / helpers (self-contained; nothing imported
# ================ from other test files, and no bar data under data/ is read) ================


def _session_grid(
    start: dt.date, n_days: int, bars_per_day: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates: list[dt.date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    ts_chunks = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bars_per_day, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.arange(0, (n_days + 1) * bars_per_day, bars_per_day, dtype=np.int32)
    return ts, day_offsets, np.array(dates, dtype=object)


def _tiny_panel(
    start: dt.date,
    *,
    n_days: int = 2,
    bars_per_day: int = 3,
    symbols: tuple[str, ...] = ("A", "B"),
) -> Panel:
    """A small, fast, self-contained Panel. Values are constants -- panel_hash is
    metadata-only (rule: never hash bar data), so what matters is ts/day_offsets/
    dates/symbols/field-names, not the numbers inside the arrays."""
    ts, day_offsets, dates = _session_grid(start, n_days, bars_per_day)
    n_rows = ts.shape[0]
    fields = {
        name: np.full((n_rows, len(symbols)), 100.0, dtype=np.float32)
        for name in ("open", "high", "low", "close", "volume")
    }
    return Panel(fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates)


def _make_fake_result(n: int = 2) -> BacktestResult:
    gross = np.full(n, -0.001, dtype=np.float64)
    turnover = np.full(n, 1.0, dtype=np.float64)
    net = gross - 0.0005 * turnover
    equity = np.cumprod(1.0 + net) * 1e7
    daily = build_daily(
        np.arange(n, dtype=np.int64),
        equity,
        net,
        gross,
        turnover,
        np.arange(n, dtype=np.int64),
        initial_capital=1e7,
    )
    return BacktestResult(
        equity_curve=equity,
        returns=net,
        positions=np.zeros((n, 1)),
        trades=pd.DataFrame({"symbol": [], "side": [], "qty": [], "price": []}),
        gross_returns=gross,
        total_costs=1.0,
        n_trades=5,
        turnover=turnover,
        rejected_order_rate=0.0,
        unfilled_notional_pct=0.0,
        forced_eod_liquidation_days=0,
        initial_capital=1e7,
        daily=daily,
    )


def _write_backtest_config(tmp_path: Path, *, params: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "strategy.yaml"
    cfg = {
        "strategy": "volume_breakout",
        "params": dict(_VALID_VOLUME_BREAKOUT_PARAMS if params is None else params),
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_sweep_config(
    tmp_path: Path, *, base_params: dict, sweep_values=(2.0, 2.5)
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_cfg = {
        "strategy": "volume_breakout",
        "base_params": dict(base_params),
        "sweep": {"volume_z_threshold": list(sweep_values)},
    }
    sweep_path.write_text(yaml.safe_dump(sweep_cfg), encoding="utf-8")
    return sweep_path


def _apply_common_mocks(monkeypatch, tmp_path: Path, panel: Panel) -> None:
    """Route `nq backtest`/`nq sweep` through a fixed synthetic panel and a canned
    backtest result, never touching real 121M-row bar data, and isolate RESULTS_ROOT
    (and hence trials.db) to `tmp_path` so tests never share state."""
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=panel.symbols)
    )
    monkeypatch.setattr(engine_mod, "run_backtest", lambda *a, **kw: _make_fake_result())
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


def _run_backtest_cli(
    monkeypatch, tmp_path: Path, panel: Panel, *, start: str, end: str, config: Path | None = None
) -> None:
    _apply_common_mocks(monkeypatch, tmp_path, panel)
    cfg_path = config if config is not None else _write_backtest_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            str(cfg_path),
            "--start",
            start,
            "--end",
            end,
            "--no-tradable-filter",
        ],
    )
    assert result.exit_code == 0, result.output


def _run_sweep_cli(
    monkeypatch, tmp_path: Path, panel: Panel, *, sweep_config: Path, start: str, end: str
) -> None:
    _apply_common_mocks(monkeypatch, tmp_path, panel)
    result = runner.invoke(
        app,
        ["sweep", "--config", str(sweep_config), "--start", start, "--end", end],
    )
    assert result.exit_code == 0, result.output


def _all_trials(tmp_path: Path) -> list[TrialRecord]:
    return TrialRegistry(tmp_path / "trials.db").all()


def _one_trial(tmp_path: Path) -> TrialRecord:
    trials = _all_trials(tmp_path)
    assert len(trials) == 1, f"expected exactly one trial, found {len(trials)}"
    return trials[0]


def _one_metrics_json(tmp_path: Path) -> dict:
    paths = list((tmp_path / "trials").glob("*/metrics.json"))
    assert len(paths) == 1, f"expected exactly one metrics.json, found {len(paths)}"
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _full_trial_record(**overrides) -> TrialRecord:
    """A TrialRecord with every field (old + spec-section-A new) populated with a
    distinctive, non-default value, for round-trip/chain tests. All new fields are
    passed as keywords, independent of wherever the real dataclass ends up ordering
    them (constructing a frozen dataclass by field ORDER would be fragile against a
    reordering the implementer might need to satisfy Python's "no required field after
    a defaulted one" rule)."""
    fields = dict(
        config_hash="cfg-abc123",
        ts="2026-08-20T10:00:00+00:00",
        strategy="volume_breakout",
        params_json=json.dumps({"volume_z_threshold": 2.0}, sort_keys=True),
        split_id="full",
        purpose="exploration",
        sharpe_gross=1.0 / 3.0,
        sharpe_net=-2.0 / 7.0,
        n_trades=42,
        turnover=0.987654321,
        breakeven_bps=12.5,
        git_sha="a1b2c3d4e5f6-dirty",
        data_fingerprint="d7f0f02a6a5ee405",
        code_version="0.1.0",
        wall_s=3.14159,
        result_path="/tmp/results/trials/cfg-abc123",
        error=None,
        ruined=False,
        ruin_index=-1,
        seed=7,
        universe_name="all_equity",
        universe_hash="u" * 16,
        panel_hash="p" * 16,
        start="2024-01-02",
        end="2024-01-05",
        cost_model_id="NSEIntradayEquityCosts()",
        slippage_model_id="SqrtImpactSlippage()",
        fill_model_id="FillModel()",
        embargo_components=json.dumps(
            {
                "execution_horizon": 0.0,
                "feature_lookback": 1.0,
                "holding_period": 10.0,
                "label_horizon": 2.0,
            },
            sort_keys=True,
        ),
        parent_trial_id=None,
        feature_version="v1",
    )
    fields.update(overrides)
    return TrialRecord(**fields)


# ================================ item 1 ================================
# "Every field round-trips through the registry and back." Amendment item 9: BYTE-FOR-
# BYTE for strings, exact for ints, repr-stable for floats -- not recomputed
# equivalence.


def test_item1_all_fields_round_trip_byte_for_byte(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials.db")
    original = _full_trial_record()
    registry.record(original)
    retrieved = _one_trial(tmp_path)

    for f in dataclasses.fields(TrialRecord):
        original_value = getattr(original, f.name)
        retrieved_value = getattr(retrieved, f.name)
        assert retrieved_value == original_value, (
            f"field {f.name!r} did not round-trip: wrote {original_value!r}, "
            f"read back {retrieved_value!r}"
        )

    # Explicit repr-stability for the float fields (amendment item 9's specific
    # wording), not just `==` (which would also pass for a value that happened to
    # be numerically equivalent but reconstructed rather than stored verbatim).
    assert repr(retrieved.sharpe_net) == repr(original.sharpe_net)
    assert repr(retrieved.turnover) == repr(original.turnover)
    assert repr(retrieved.wall_s) == repr(original.wall_s)

    # Byte-for-byte for a string field that would be silently corrupted by any
    # canonicalisation/re-serialisation pass.
    assert retrieved.embargo_components == original.embargo_components
    assert retrieved.git_sha == "a1b2c3d4e5f6-dirty"


# ================================ item 2 ================================
# "git_sha is never None after a write; a dirty tree yields a `-dirty` suffix."


def test_item2_git_sha_never_none_after_cli_backtest_write(monkeypatch, tmp_path: Path) -> None:
    panel = _tiny_panel(dt.date(2024, 5, 6))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-05-06", end="2024-05-07")
    record = _one_trial(tmp_path)
    assert record.git_sha is not None
    assert record.git_sha != "None"
    assert isinstance(record.git_sha, str) and len(record.git_sha) > 0


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)


def test_item2_git_sha_dirty_tree_gets_dirty_suffix(tmp_path: Path) -> None:
    from nifty_quant.research.provenance import get_git_sha

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "f.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    clean_sha = get_git_sha(repo_root=repo)
    assert clean_sha is not None and not clean_sha.endswith("-dirty")

    (repo / "f.txt").write_text("v2 -- dirties the tree", encoding="utf-8")
    dirty_sha = get_git_sha(repo_root=repo)
    assert dirty_sha == f"{clean_sha}-dirty"


def test_item2_git_sha_outside_a_repo_returns_no_git_sentinel(tmp_path: Path) -> None:
    from nifty_quant.research.provenance import get_git_sha

    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    assert get_git_sha(repo_root=not_a_repo) == "no-git"


# ================================ item 3 (LOAD-BEARING) ================================
# "panel_hash differs for two runs on different date ranges of the same dataset, where
# data_fingerprint is identical." Anti-coincidence: the two panels below are DISJOINT
# in calendar date (so first/last ts genuinely differ) but have the SAME row count and
# SAME symbols/fields (so a hash that only reacted to row count, or to symbols/fields,
# would wrongly agree here) -- isolating ts as the discriminator, verified explicitly
# below rather than merely asserted.


def test_item3_panel_hash_differs_for_disjoint_date_ranges_same_data_fingerprint(
    monkeypatch, tmp_path: Path
) -> None:
    panel_a = _tiny_panel(dt.date(2024, 1, 2))  # Tue+Wed, Jan 2026-01-02/03
    panel_b = _tiny_panel(dt.date(2024, 2, 6))  # Tue+Wed, disjoint week

    # Confirm the anti-coincidence setup: equal row count, genuinely different ts.
    assert panel_a.n_rows() == panel_b.n_rows()
    assert panel_a.symbols == panel_b.symbols
    assert int(panel_a.ts[0]) != int(panel_b.ts[0])
    assert int(panel_a.ts[-1]) != int(panel_b.ts[-1])

    tmp_a, tmp_b = tmp_path / "a", tmp_path / "b"
    _run_backtest_cli(monkeypatch, tmp_a, panel_a, start="2024-01-02", end="2024-01-03")
    _run_backtest_cli(monkeypatch, tmp_b, panel_b, start="2024-02-06", end="2024-02-07")

    record_a = _one_trial(tmp_a)
    record_b = _one_trial(tmp_b)

    real_fingerprint = Manifest.load().fingerprint
    assert record_a.data_fingerprint == real_fingerprint
    assert record_b.data_fingerprint == real_fingerprint
    assert record_a.data_fingerprint == record_b.data_fingerprint, (
        "data_fingerprint is whole-dataset and must be identical across both runs -- "
        "the whole reason panel_hash exists is that this field alone cannot tell them "
        "apart"
    )
    assert record_a.panel_hash != record_b.panel_hash, (
        "panel_hash must differ for two disjoint date-range slices of the same "
        "dataset even though data_fingerprint is identical -- this is item 3's "
        "specific, load-bearing failure mode"
    )


def test_item3_panel_hash_differs_for_different_symbol_lists_same_ts_and_row_count(
    monkeypatch, tmp_path: Path
) -> None:
    """Supplementary: isolates the "sorted symbol list" input of the six named in
    amendment item 4, using the SAME ts/day_offsets/dates for both panels so only the
    symbol axis varies."""
    ts, day_offsets, dates = _session_grid(dt.date(2024, 3, 4), n_days=2, bars_per_day=3)
    n_rows = ts.shape[0]

    def _panel(symbols: tuple[str, ...]) -> Panel:
        fields = {
            name: np.full((n_rows, len(symbols)), 100.0, dtype=np.float32)
            for name in ("open", "high", "low", "close", "volume")
        }
        return Panel(fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates)

    panel_ab = _panel(("A", "B"))
    panel_abc = _panel(("A", "B", "C"))
    assert np.array_equal(panel_ab.ts, panel_abc.ts)
    assert panel_ab.n_rows() == panel_abc.n_rows()

    tmp_a, tmp_b = tmp_path / "ab", tmp_path / "abc"
    _run_backtest_cli(monkeypatch, tmp_a, panel_ab, start="2024-03-04", end="2024-03-05")
    _run_backtest_cli(monkeypatch, tmp_b, panel_abc, start="2024-03-04", end="2024-03-05")

    record_ab = _one_trial(tmp_a)
    record_abc = _one_trial(tmp_b)
    assert record_ab.data_fingerprint == record_abc.data_fingerprint
    assert record_ab.panel_hash != record_abc.panel_hash


# ================================ item 4 ================================
# "panel_hash is stable across runs on the same slice."


def test_item4_panel_hash_stable_across_independent_runs_same_slice(
    monkeypatch, tmp_path: Path
) -> None:
    # Two INDEPENDENTLY constructed Panel objects with identical content (not the
    # same object) -- proves the hash is a function of content, not identity.
    panel_1 = _tiny_panel(dt.date(2024, 6, 3))
    panel_2 = _tiny_panel(dt.date(2024, 6, 3))
    assert panel_1 is not panel_2
    assert np.array_equal(panel_1.ts, panel_2.ts)

    tmp_a, tmp_b = tmp_path / "run1", tmp_path / "run2"
    _run_backtest_cli(monkeypatch, tmp_a, panel_1, start="2024-06-03", end="2024-06-04")
    _run_backtest_cli(monkeypatch, tmp_b, panel_2, start="2024-06-03", end="2024-06-04")

    record_1 = _one_trial(tmp_a)
    record_2 = _one_trial(tmp_b)
    assert record_1.panel_hash == record_2.panel_hash


# ================================ item 5 ================================
# "cost_model_id distinguishes NSEIntradayEquityCosts() from one with a non-default
# field, and is identical for two default instances." Amendment item 1 pins the exact
# canonical string, so this is testable with exact-value assertions, not just
# inequality.


def test_item5_cost_model_id_default_matches_canonical_form(monkeypatch, tmp_path: Path) -> None:
    panel = _tiny_panel(dt.date(2024, 7, 1))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-07-01", end="2024-07-02")
    record = _one_trial(tmp_path)
    assert record.cost_model_id == "NSEIntradayEquityCosts()"


def test_item5_cost_model_id_identical_for_two_default_instances(
    monkeypatch, tmp_path: Path
) -> None:
    panel = _tiny_panel(dt.date(2024, 7, 8))
    tmp_a, tmp_b = tmp_path / "a", tmp_path / "b"
    _run_backtest_cli(monkeypatch, tmp_a, panel, start="2024-07-08", end="2024-07-09")
    _run_backtest_cli(monkeypatch, tmp_b, panel, start="2024-07-08", end="2024-07-09")
    assert _one_trial(tmp_a).cost_model_id == _one_trial(tmp_b).cost_model_id == (
        "NSEIntradayEquityCosts()"
    )


def test_item5_cost_model_id_includes_nondefault_field(monkeypatch, tmp_path: Path) -> None:
    import nifty_quant.execution.costs as costs_mod

    real_cls = costs_mod.NSEIntradayEquityCosts
    monkeypatch.setattr(costs_mod, "NSEIntradayEquityCosts", lambda: real_cls(brokerage_flat=10.0))

    panel = _tiny_panel(dt.date(2024, 7, 15))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-07-15", end="2024-07-16")
    record = _one_trial(tmp_path)
    assert record.cost_model_id == "NSEIntradayEquityCosts(brokerage_flat=10.0)"
    assert record.cost_model_id != "NSEIntradayEquityCosts()"


# ================================ item 6 ================================
# Amendment item 5, made concrete: on registry-write failure, (1) log at WARNING and
# never swallow, (2) metrics.json["registry_write_failed"] is True, (3) the artifact
# dir is still written and the command still exits 0. And: False on every normal run.


def test_item6_registry_write_failure_surfaced_not_swallowed(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    import nifty_quant.research.registry as registry_mod

    def _raise(self, rec) -> None:
        raise RuntimeError("registry boom")

    monkeypatch.setattr(registry_mod.TrialRegistry, "record", _raise)

    panel = _tiny_panel(dt.date(2024, 8, 5))
    _apply_common_mocks(monkeypatch, tmp_path, panel)
    cfg_path = _write_backtest_config(tmp_path)

    caplog.set_level(logging.WARNING)
    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            "volume_breakout",
            "--config",
            str(cfg_path),
            "--start",
            "2024-08-05",
            "--end",
            "2024-08-06",
            "--no-tradable-filter",
        ],
    )

    assert result.exit_code == 0, (
        "a registry write failure must not abort the backtest -- "
        f"got exit_code={result.exit_code}, output={result.output!r}"
    )

    metrics = _one_metrics_json(tmp_path)
    assert metrics.get("registry_write_failed") is True

    assert any(
        record.levelno == logging.WARNING and "registry boom" in record.getMessage()
        for record in caplog.records
    ), "the swallowed exception must be logged at WARNING, not silently dropped"


def test_item6_registry_write_failed_is_false_on_normal_run(monkeypatch, tmp_path: Path) -> None:
    panel = _tiny_panel(dt.date(2024, 8, 12))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-08-12", end="2024-08-13")
    metrics = _one_metrics_json(tmp_path)
    assert metrics.get("registry_write_failed") is False, (
        "registry_write_failed must be an explicit False on a normal run, not merely "
        "absent -- amendment item 5: 'a field with two observable states'"
    )


# ================================ item 7 ================================
# Amendment item 6: metrics.json["provenance"] has EXACTLY the 16 keys named by the
# amendment plus the later-added `contract_hash` (17 total, see docstring note 7),
# null permitted only for seed/parent_trial_id; and the 39 existing performance keys
# must be untouched (rule: provenance ADDS keys, never renames/removes).


def test_item7_metrics_json_provenance_block_exact_key_set(monkeypatch, tmp_path: Path) -> None:
    panel = _tiny_panel(dt.date(2024, 9, 2))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-09-02", end="2024-09-03")
    metrics = _one_metrics_json(tmp_path)

    assert "provenance" in metrics
    provenance = metrics["provenance"]
    assert isinstance(provenance, dict)
    assert set(provenance.keys()) == PROVENANCE_KEYS, (
        "metrics.json['provenance'] must contain EXACTLY the 16 spec keys -- no more, "
        f"no fewer. Got: {sorted(provenance.keys())}"
    )
    for key in PROVENANCE_KEYS - NULLABLE_PROVENANCE_KEYS:
        assert provenance[key] is not None, f"provenance[{key!r}] must not be null"


def test_item7_metrics_json_provenance_does_not_disturb_existing_performance_keys(
    monkeypatch, tmp_path: Path
) -> None:
    panel = _tiny_panel(dt.date(2024, 9, 9))
    _run_backtest_cli(monkeypatch, tmp_path, panel, start="2024-09-09", end="2024-09-10")
    metrics = _one_metrics_json(tmp_path)

    # Two of the five recently-added diagnostic counters, plus a long-standing key --
    # must remain present, top-level, and unrenamed by the provenance addition.
    for key in ("breakeven_bps", "n_stale_marks", "n_rows_negative_cash", "gross_sharpe"):
        assert key in metrics, f"existing performance key {key!r} must not be removed/renamed"
    assert "provenance" not in {"breakeven_bps", "n_stale_marks"}  # sanity: keys distinct


# ================================ item 8 ================================
# Amendment item 7: a THREE-generation chain (grandparent -> parent -> child) must be
# walkable.


def test_item8_parent_trial_id_three_generation_chain_is_walkable(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "trials.db")
    grandparent = _full_trial_record(config_hash="gen0", parent_trial_id=None)
    parent = _full_trial_record(config_hash="gen1", parent_trial_id="gen0")
    child = _full_trial_record(config_hash="gen2", parent_trial_id="gen1")
    for record in (grandparent, parent, child):
        registry.record(record)

    by_hash = {record.config_hash: record for record in _all_trials(tmp_path)}
    assert set(by_hash) == {"gen0", "gen1", "gen2"}

    chain: list[str] = []
    current: TrialRecord | None = by_hash["gen2"]
    while current is not None:
        chain.append(current.config_hash)
        current = by_hash.get(current.parent_trial_id) if current.parent_trial_id else None

    assert chain == ["gen2", "gen1", "gen0"], f"chain did not walk three generations: {chain}"
    assert by_hash["gen0"].parent_trial_id is None, "the root of the chain must terminate at None"


# ================================ item 9 ================================
# Amendment item 8: a sweep-derived trial's parent_trial_id is non-null and equals the
# hash `base_params` alone would produce; and sweep must not leave a field null that
# backtest fills.


def test_item9_sweep_trial_parent_points_at_base_params_hash(monkeypatch, tmp_path: Path) -> None:
    base_params = dict(_VALID_VOLUME_BREAKOUT_PARAMS)
    expected_base_hash = strategy_config_hash(
        {"strategy": "volume_breakout", "params": base_params}
    )

    sweep_cfg = _write_sweep_config(tmp_path, base_params=base_params, sweep_values=(2.0, 2.5))
    panel = _tiny_panel(dt.date(2024, 10, 1))
    _run_sweep_cli(
        monkeypatch, tmp_path, panel, sweep_config=sweep_cfg, start="2024-10-01", end="2024-10-02"
    )

    sweep_trials = _all_trials(tmp_path)
    assert len(sweep_trials) == 2, f"expected 2 sweep trials (2.0, 2.5), found {len(sweep_trials)}"
    for trial in sweep_trials:
        assert trial.parent_trial_id is not None
        assert trial.parent_trial_id == expected_base_hash, (
            f"sweep trial {trial.config_hash!r} parent_trial_id "
            f"{trial.parent_trial_id!r} != base_params hash {expected_base_hash!r}"
        )


def test_item9_sweep_provenance_no_field_left_null_that_backtest_fills(
    monkeypatch, tmp_path: Path
) -> None:
    base_params = dict(_VALID_VOLUME_BREAKOUT_PARAMS)
    checked_keys = PROVENANCE_KEYS - {"seed", "parent_trial_id"}

    tmp_bt, tmp_sw = tmp_path / "backtest", tmp_path / "sweep"
    panel = _tiny_panel(dt.date(2024, 10, 8))

    backtest_cfg = _write_backtest_config(tmp_bt, params=base_params)
    _run_backtest_cli(
        monkeypatch, tmp_bt, panel, start="2024-10-08", end="2024-10-09", config=backtest_cfg
    )
    backtest_trial = _one_trial(tmp_bt)
    assert backtest_trial.parent_trial_id is None, "a root nq backtest trial has no parent"

    sweep_cfg = _write_sweep_config(tmp_sw, base_params=base_params, sweep_values=(2.0,))
    _run_sweep_cli(
        monkeypatch, tmp_sw, panel, sweep_config=sweep_cfg, start="2024-10-08", end="2024-10-09"
    )
    sweep_trials = _all_trials(tmp_sw)
    assert len(sweep_trials) == 1
    sweep_trial = sweep_trials[0]

    for key in checked_keys:
        bt_value = getattr(backtest_trial, key)
        sw_value = getattr(sweep_trial, key)
        assert bt_value is not None, f"backtest trial field {key!r} unexpectedly null (bad fixture)"
        assert sw_value is not None, (
            f"sweep left field {key!r} null even though nq backtest fills it -- "
            "amendment item 8: 'no field may be left null by the sweep path that the "
            "backtest path fills'"
        )
