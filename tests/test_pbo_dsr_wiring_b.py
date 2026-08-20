"""Tests for specs/pbo_dsr_wiring.md, authored from the spec ALONE by author B of two
independent authors (author A's suite, `test_pbo_dsr_wiring_a.py`, is not read).

The wiring this spec describes DOES NOT EXIST YET (`nq sweep` never calls `pbo_cscv`,
`effective_n_trials`, or `build_trial_matrix`; `TrialMatrix.explain()` has no per-trial
path/ts/split_id; `cli.py` still sets `n_eff = float(n_trials)`; there is no "too few
periods" message anywhere). Most tests here are expected to FAIL (RED). Two tests (item
5 and half of item 8) exercise `effective_n_trials`/`deflated_sharpe` directly, which the
spec says are ALREADY correct -- those are expected to PASS today, as regression guards.

================ SPEC DEFECTS / AMBIGUITIES FOUND ================

1. **Item 2 has no pinned output shape.** The spec says `explain()` must "name the
   resolved artifact path, ts and split_id for each trial" but does not say whether this
   lives in new `TrialMatrix` dataclass fields, a dict, or purely in the `.explain()`
   string. Adding fields would be a structural change I cannot predict the name of from
   the spec alone, so this file tests only the OBSERABLE CONTRACT: the three values
   appear as substrings of `.explain()`'s returned string. A different but reasonable
   implementation (e.g. storing them in a new field and never rendering them into the
   string) would satisfy the literal field name but fail this test; I judge the spec's
   own wording ("explain() must additionally report") to require the string form.

2. **Item 4 offers the implementer two incompatible fixes and the orchestrator's own
   instruction picks one.** The spec body says "either use `min(16, T - (T % 2))` or
   REFUSE below 16 sessions", but the "Required tests" section and this task's explicit
   instruction both describe only the REFUSE branch ("REFUSES with a message naming both
   numbers, rather than letting a bare ValueError escape"). This file tests the REFUSE
   branch only, per the Required-tests text and the explicit task instruction, and flags
   the body's silent-degrade alternative as a live disagreement for the lead to resolve
   against author A's reading.

3. **Item 6's "no matrix can be assembled" is easy to conflate with "a 1-column matrix".**
   `effective_n_trials` already returns a legitimate `1.0` for a single trial (see
   `metrics.py`, `if n_trials == 1: return 1.0`) -- that is NOT "unavailable", it is a
   correct answer. This file therefore does not use a single-successful-trial sweep to
   exercise item 6 (that would coincidentally show `eff=1.000`, indistinguishable from
   the honest computation for n=1). Instead it forces EVERY trial in the sweep to fail
   (mocked `run_backtest` raising), so zero trials produce an artifact at all and
   `build_trial_matrix` legitimately cannot assemble anything -- the unambiguous "no
   matrix" case the spec is describing.

4. **No output wording is pinned anywhere in the spec** for how PBO, DSR, or "n_eff
   unavailable" should be printed by `nq sweep`. This file uses permissive,
   case-insensitive regexes (`PBO[:=]<number>`, and an OR of "unavailable"/"not
   available"/"n/a"/"cannot be computed"/"no matrix" near "n_eff"/"effective ... trials")
   rather than a single hardcoded string, and documents that a differently-worded but
   equally honest message could fail this file's regex -- a risk inherent to testing
   CLI prose against a spec that never pins it.

5. **Item 7's own illustrative numbers are internally inconsistent with `expected_max_sharpe`'s
   contract.** The spec's D3 section quotes "a nine-trial exploration set: honest value
   near 1" -- but `expected_max_sharpe` raises `ValueError` for `n_trials < 2`
   (`metrics.py` line ~434), so feeding an honest n_eff of ~1 into it, as item 7
   literally asks ("feeding the honest n_eff into expected_max_sharpe"), would crash, not
   produce a lower sr0. This file's own item 7 fixture is deliberately engineered to a
   MODERATE correlation structure (honest n_eff ~3.6 against a raw count of 9) rather
   than reproducing the spec's own near-1 illustration, specifically so the test can
   exercise the successful comparison item 7 asks for instead of a `ValueError` the spec
   text does not mention how to handle. This is flagged as a real gap for the lead: the
   eventual wiring needs a defined behaviour for honest n_eff < 2 (floor it? report sr0
   unavailable too?) that the spec never states.

6. **Item 8 does not name "the caller".** `deflated_sharpe`'s `T < 4 -> nan` behaviour is
   already wired into exactly one call site pre-spec: `nq walkforward`'s pooled-DSR
   calculation (`cli.py`, ~line 1128) -- `nq sweep` has no DSR call site at all yet (that
   would be new code this spec doesn't describe). This file therefore targets `nq
   walkforward`, the only site that already exists independent of whatever new sweep
   wiring the implementer builds for items 2-4/6-7. A `nq sweep`-side DSR message would
   also satisfy the spirit of item 8 if the implementer adds one there instead or as
   well; this file does not test that path.

7. **Rule 8 compliance note, not a spec defect**: item 4's expected refusal message is
   checked against `n_splits`'s ACTUAL default read via `inspect.signature(pbo_cscv)`
   rather than a hardcoded `16`, so this test does not itself become the kind of
   unmeasured constant CLAUDE.md rule 8 forbids production code from having.

8. **Fixture facts, not thresholds** (rule 8 sanity note): "22 trading days in Jan 2024"
   and "10 trading days in 2024-01-01..2024-01-12" (both `reliance_only` universe) are
   MEASURED from the real trading calendar via a throwaway `nq sweep` run during
   authoring, not chosen a priori; they are asserted as fixture preconditions (self-
   checking) rather than assumed silently, so a calendar change would fail loudly here
   rather than mask a broken test.
"""

from __future__ import annotations

import inspect
import math
import re
from datetime import date, datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from nifty_quant.backtest.daily import build_daily
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.backtest.metrics import (
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
    pbo_cscv,
)
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.research.registry import TrialRecord, TrialRegistry
from nifty_quant.universe.static import Universe

runner = CliRunner()

_IST = timezone(timedelta(hours=5, minutes=30))

_BASE_PARAMS = {
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


# ================ shared helpers ================


def _write_returns_parquet(path: Path, ts: np.ndarray, returns: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts": np.asarray(ts, dtype=np.int64),
            "return": np.asarray(returns, dtype=np.float64),
        }
    ).to_parquet(path, index=False)


def _trial_record(
    *, config_hash: str, ts: str, split_id: str, result_path: str
) -> TrialRecord:
    """A minimally-filled but fully-valid `TrialRecord`; only the fields the collision
    logic cares about (config_hash, ts, split_id, result_path) vary between tests."""
    return TrialRecord(
        config_hash=config_hash,
        ts=ts,
        strategy="volume_breakout",
        params_json="{}",
        split_id=split_id,
        purpose="exploration",
        sharpe_gross=0.1,
        sharpe_net=0.05,
        n_trades=10,
        turnover=1.0,
        breakeven_bps=5.0,
        git_sha="deadbeef",
        data_fingerprint="fp",
        code_version="0.0.0-test",
        wall_s=1.0,
        result_path=result_path,
        error=None,
    )


def _write_sweep_config(tmp_path: Path, *, sweep_values: tuple[float, ...]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "strategy": "volume_breakout",
        "base_params": dict(_BASE_PARAMS),
        "sweep": {"volume_z_threshold": list(sweep_values)},
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ================ items 1-2: TrialRegistry.build_trial_matrix / .explain() ================
# These are pure registry-level tests: synthetic artifacts on disk, no CLI, no real data.


def _collision_fixture(tmp_path: Path) -> tuple[TrialRegistry, str, Path, Path]:
    """Two rows sharing one config_hash: an OLDER row pointing at a 3-period
    walk-forward-style artifact, and a NEWER row pointing at a 22-period sweep-style
    artifact -- exactly the D1 scenario in the spec."""
    registry = TrialRegistry(tmp_path / "trials.db")
    chash = "collide_config_hash_0001"

    old_dir = tmp_path / "trials" / "stale_walkforward_3day"
    old_ts = np.array([1704067200, 1704153600, 1704240000], dtype=np.int64)  # 3 days
    _write_returns_parquet(
        old_dir / "returns.parquet", old_ts, np.array([0.001, -0.002, 0.0005])
    )

    new_dir = tmp_path / "trials" / "fresh_sweep_22day"
    new_ts = old_ts[0] + np.arange(22, dtype=np.int64) * 86400
    new_returns = np.linspace(-0.01, 0.01, 22)
    _write_returns_parquet(new_dir / "returns.parquet", new_ts, new_returns)

    registry.record(
        _trial_record(
            config_hash=chash,
            ts="2024-01-01T00:00:00+00:00",
            split_id="walkforward_test_slice",
            result_path=str(old_dir),
        )
    )
    registry.record(
        _trial_record(
            config_hash=chash,
            ts="2024-06-01T00:00:00+00:00",
            split_id="sweep_full_result",
            result_path=str(new_dir),
        )
    )
    return registry, chash, old_dir, new_dir


def test_item1_collision_prefers_newest_artifact_with_result_path(tmp_path: Path) -> None:
    """The load-bearing test (spec item 1): on a config_hash collision between a stale
    3-period artifact and a fresh 22-period one, the matrix must reflect the 22-period
    artifact. Must FAIL against current code (`ORDER BY ts, rowid LIMIT 1` picks the
    OLDEST row, i.e. the 3-period one)."""
    registry, chash, _old_dir, _new_dir = _collision_fixture(tmp_path)

    matrix = registry.build_trial_matrix(trial_ids=[chash])

    assert matrix.trial_ids == (chash,)
    assert matrix.matrix.shape == (22, 1), (
        "collision must resolve to the NEWER 22-period sweep artifact, not the stale "
        f"3-period one; got matrix shape {matrix.matrix.shape}"
    )
    assert matrix.period_index.size == 22


def test_item2_explain_names_resolved_artifact_path_ts_and_split_id(tmp_path: Path) -> None:
    """`explain()` must name WHICH artifact path, ts, and split_id each kept trial
    resolved to. Must FAIL against current code, whose `.explain()` has no such
    information to report at all (per D1: "It has no way to say I loaded a stale
    artifact")."""
    registry, chash, _old_dir, new_dir = _collision_fixture(tmp_path)

    matrix = registry.build_trial_matrix(trial_ids=[chash])
    summary = matrix.explain()

    assert str(new_dir) in summary, (
        f"explain() must name the resolved artifact's path ({new_dir}); got:\n{summary}"
    )
    assert "2024-06-01T00:00:00+00:00" in summary, (
        f"explain() must name the resolved row's ts; got:\n{summary}"
    )
    assert "sweep_full_result" in summary, (
        f"explain() must name the resolved row's split_id; got:\n{summary}"
    )


# ================ items 3-4: nq sweep, real backtests, end to end ================
# reliance_only universe, Jan 2024 -- measured at authoring time: 4-5s for a 2-value
# sweep. See spec-defect note 8 above on why these day counts are asserted, not assumed.


def test_item3_real_sweep_reports_finite_pbo_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    """Spec item 3: a real >= 2-param sweep produces a FINITE PBO in [0, 1], driven from
    the CLI end to end. Jan 2024 on `reliance_only` gives 22 trading sessions
    (>= the pbo_cscv default n_splits of 16), matching the spec's own D1 worked example.
    Must FAIL against current code: `nq sweep` never calls `pbo_cscv` at all today."""
    import nifty_quant.settings as settings_mod

    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    cfg_path = _write_sweep_config(tmp_path, sweep_values=(1.5, 2.0))
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(cfg_path),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--universe",
            "reliance_only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "n_days=22" in result.output, (
        "fixture assumption broken: expected 22 trading sessions for reliance_only in "
        f"Jan 2024; got:\n{result.output}"
    )

    match = re.search(
        r"(?i)PBO\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|nan)",
        result.output,
    )
    assert match is not None, f"expected a PBO value printed by `nq sweep`; got:\n{result.output}"

    pbo_value = float(match.group(1))
    assert math.isfinite(pbo_value), f"PBO must be finite, got {pbo_value}"
    assert 0.0 <= pbo_value <= 1.0, f"PBO must lie in [0, 1], got {pbo_value}"


def test_item4_sweep_refuses_when_periods_below_n_splits(
    tmp_path: Path, monkeypatch
) -> None:
    """Spec item 4: a sweep with T < n_splits REFUSES with a message naming both
    numbers, rather than a bare ValueError escaping from inside pbo_cscv.
    2024-01-01..2024-01-12 on reliance_only gives T=10 trading sessions, below
    pbo_cscv's own default n_splits (read from the function signature, not hardcoded,
    per rule 8). Must FAIL against current code: no refusal exists at all today, because
    nothing calls pbo_cscv from `nq sweep`."""
    import nifty_quant.settings as settings_mod

    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    n_splits_default = inspect.signature(pbo_cscv).parameters["n_splits"].default

    cfg_path = _write_sweep_config(tmp_path, sweep_values=(1.5, 2.0))
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(cfg_path),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-12",
            "--universe",
            "reliance_only",
        ],
    )

    assert result.exception is None, (
        f"a T < n_splits sweep must not let a bare exception escape: {result.exception!r}"
    )
    assert "n_days=10" in result.output, (
        "fixture assumption broken: expected 10 trading sessions for reliance_only in "
        f"2024-01-01..2024-01-12; got:\n{result.output}"
    )
    assert "ValueError" not in result.output, (
        f"a bare ValueError must not leak into the sweep's own output; got:\n{result.output}"
    )

    period_re = re.compile(r"\b10\b")
    splits_re = re.compile(rf"\b{re.escape(str(n_splits_default))}\b")
    refusal_keyword_re = re.compile(
        r"(?i)(refus|cannot|can't|requires? at least|insufficient|too few|not enough|"
        r"below|fewer than)"
    )
    matching_lines = [
        line
        for line in result.output.splitlines()
        if period_re.search(line) and splits_re.search(line) and refusal_keyword_re.search(line)
    ]
    assert matching_lines, (
        f"expected a refusal message naming both the actual period count (10) and "
        f"n_splits ({n_splits_default}); got:\n{result.output}"
    )


# ================ item 5: effective_n_trials, pure function-level ================
# Already-correct arithmetic per the spec; this is a regression guard, expected GREEN.


def test_item5_effective_n_trials_near_one_for_highly_correlated_trials() -> None:
    """Three near-identical trials must report an honest n_eff near 1, strictly less
    than the raw count of 3 -- the entire point of the statistic. Deterministic
    fixture, no RNG (rule 9): a shared base series with a tiny fixed sinusoidal
    perturbation per column."""
    t = np.arange(30)
    base = np.linspace(-0.01, 0.01, 30)
    noise = 1e-6
    matrix = np.column_stack(
        [
            base,
            base + noise * np.sin(t),
            base - noise * np.cos(t),
        ]
    )
    raw_count = matrix.shape[1]

    n_eff = effective_n_trials(matrix)

    assert n_eff < raw_count, (
        f"honest n_eff ({n_eff}) must be strictly less than the raw trial count "
        f"({raw_count}) for near-duplicate trials"
    )
    assert n_eff < 1.5, f"three near-identical trials should give n_eff near 1, got {n_eff}"


# ================ item 6: n_eff unavailable when no matrix can be assembled ================
# Mocked sweep, every backtest fails -> zero artifacts -> genuinely no matrix.


def _tiny_panel(dates: list[date], symbols: tuple[str, ...] = ("RELIANCE",)) -> Panel:
    ts_list = []
    for d in dates:
        start = datetime.combine(d, dt_time(9, 15), tzinfo=_IST)
        ts_list.append(np.array([int(start.timestamp())], dtype=np.int64))
    ts = np.concatenate(ts_list)
    day_offsets = np.arange(0, len(dates) + 1, dtype=np.int32)
    n_rows = ts.shape[0]
    fields = {
        name: np.full((n_rows, len(symbols)), 100.0, dtype=np.float32)
        for name in ("open", "high", "low", "close", "volume")
    }
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=np.array(dates, dtype=object),
    )


def test_item6_n_eff_reported_unavailable_when_no_matrix_can_be_assembled(
    tmp_path: Path, monkeypatch
) -> None:
    """Spec item 6: where no matrix can be assembled, n_eff must be reported as
    unavailable, NOT silently substituted by a raw trial count. Every backtest in this
    sweep fails (mocked `run_backtest` raises), so zero trials produce an artifact and
    `build_trial_matrix` cannot assemble anything at all -- the unambiguous case (see
    spec-defect note 3 above on why a single-successful-trial sweep would NOT test
    this). Must FAIL against current code: `nq sweep` prints nothing about n_eff today."""
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    panel = _tiny_panel(dates)

    def _always_fail(*_args: object, **_kwargs: object) -> BacktestResult:
        raise RuntimeError("synthetic backtest failure for item6 fixture")

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="r", symbols=panel.symbols)
    )
    monkeypatch.setattr(engine_mod, "run_backtest", _always_fail)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)

    cfg_path = _write_sweep_config(tmp_path, sweep_values=(1.5, 2.0))
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(cfg_path),
            "--start",
            dates[0].isoformat(),
            "--end",
            dates[-1].isoformat(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0 ok, 2 failed" in result.output, (
        f"fixture assumption broken: expected both trials to fail; got:\n{result.output}"
    )

    assert not re.search(r"(?i)eff\s*=\s*[0-9]", result.output), (
        "n_eff must not be silently reported as a numeric value when no matrix could "
        f"be assembled (zero trials produced an artifact); got:\n{result.output}"
    )
    unavailable_re = re.compile(
        r"(n_eff|effective[^.\n]{0,20}trials?)[^.\n]{0,60}"
        r"(unavailable|not available|n/a|cannot be computed|could not be computed|no matrix)"
        r"|(unavailable|not available|n/a|cannot be computed|could not be computed|"
        r"no matrix)[^.\n]{0,60}(n_eff|effective[^.\n]{0,20}trials?)",
        re.IGNORECASE,
    )
    assert unavailable_re.search(result.output), (
        f"expected an explicit n_eff-unavailable marker; got:\n{result.output}"
    )


# ================ item 7: honest n_eff -> lower sr0 -> higher DSR, direction pin ================
# Pure function-level test; deterministic fixture engineered to keep honest n_eff >= 2
# (see spec-defect note 5 above on why this diverges from the spec's own ~1 illustration).


def test_item7_honest_n_eff_yields_lower_sr0_and_higher_dsr_than_raw_count() -> None:
    t = np.arange(40)
    fa = np.linspace(-0.02, 0.02, 40)
    fb = np.array([0.015 * ((-1) ** i) for i in t])
    fc = np.array([0.012 * np.sin(2 * np.pi * i / 9) for i in t])
    fd = np.array([0.01 * np.cos(2 * np.pi * i / 13) for i in t])
    weights = [
        (1.0, 0.05, 0.02, 0.01),
        (0.95, 0.1, 0.02, 0.02),
        (0.9, 0.05, 0.05, 0.02),
        (0.1, 1.0, 0.05, 0.02),
        (0.05, 0.95, 0.05, 0.05),
        (0.05, 0.05, 1.0, 0.05),
        (0.05, 0.05, 0.95, 0.1),
        (0.05, 0.05, 0.05, 1.0),
        (0.1, 0.05, 0.05, 0.95),
    ]
    matrix = np.column_stack(
        [wa * fa + wb * fb + wc * fc + wd * fd for wa, wb, wc, wd in weights]
    )
    raw_count = matrix.shape[1]

    honest_n_eff = effective_n_trials(matrix)
    assert 2.0 <= honest_n_eff < raw_count, (
        f"fixture assumption broken: honest n_eff ({honest_n_eff}) must be in "
        f"[2, {raw_count}) for expected_max_sharpe to accept it directly"
    )

    var_trial_sharpes = 0.04
    sr0_raw = expected_max_sharpe(raw_count, var_trial_sharpes)
    sr0_honest = expected_max_sharpe(honest_n_eff, var_trial_sharpes)
    assert sr0_honest < sr0_raw, (
        f"honest n_eff ({honest_n_eff:.3f}) must yield a LOWER sr0 than the raw count "
        f"({raw_count}): got sr0_honest={sr0_honest}, sr0_raw={sr0_raw}"
    )

    pooled = np.random.default_rng(7).normal(loc=0.0009, scale=0.01, size=60)
    dsr_raw = deflated_sharpe(pooled, sr0=sr0_raw)
    dsr_honest = deflated_sharpe(pooled, sr0=sr0_honest)
    assert math.isfinite(dsr_raw) and math.isfinite(dsr_honest)
    assert dsr_honest > dsr_raw, (
        "under-deflation direction pin: an honest sr0 must produce a HIGHER DSR than "
        f"the raw-count sr0 on the same pooled returns; got dsr_honest={dsr_honest}, "
        f"dsr_raw={dsr_raw}"
    )


# ================ item 8: deflated_sharpe(T=3) is nan; caller says "too few periods"
# ================


def test_item8_deflated_sharpe_returns_nan_at_t3_unit_level() -> None:
    """Regression guard (already-correct arithmetic, must not change): T=3 is below the
    function's own T>=4 floor for skew/kurtosis, so it returns nan."""
    t3_returns = np.array([0.001, -0.0004, 0.0009])
    assert math.isnan(deflated_sharpe(t3_returns, sr0=0.0))


def _wf_session_ts(session_date: date) -> np.ndarray:
    start = datetime.combine(session_date, dt_time(9, 15), tzinfo=_IST)
    return np.array([int(start.timestamp())], dtype=np.int64)


def _wf_panel(dates: list[date], symbols: tuple[str, ...] = ("A", "B")) -> Panel:
    ts = np.concatenate([_wf_session_ts(d) for d in dates]).astype(np.int64)
    day_offsets = np.arange(0, len(dates) + 1, dtype=np.int32)
    n_rows = ts.shape[0]
    fields = {
        "open": np.full((n_rows, len(symbols)), 100.0, dtype=np.float64),
        "high": np.full((n_rows, len(symbols)), 100.5, dtype=np.float64),
        "low": np.full((n_rows, len(symbols)), 99.5, dtype=np.float64),
        "close": np.full((n_rows, len(symbols)), 100.0, dtype=np.float64),
        "volume": np.full((n_rows, len(symbols)), 1000.0, dtype=np.float64),
    }
    return Panel(fields, symbols, ts, day_offsets, np.array(dates, dtype=object))


def _wf_row_day_index(panel: Panel) -> np.ndarray:
    return (
        np.searchsorted(panel.day_offsets, np.arange(panel.n_rows(), dtype=np.int64), side="right")
        - 1
    )


def _wf_session_dates_epoch(panel: Panel) -> np.ndarray:
    return np.asarray(
        [
            int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            for d in panel.dates
        ],
        dtype=np.int64,
    )


def _wf_fake_result(n_rows: int, panel: Panel, seed: int = 0) -> BacktestResult:
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


class _FakeCalendar:
    def __init__(self, dates: list[date]) -> None:
        self._dates = list(dates)

    def session_dates(
        self, start: date | None = None, end: date | None = None, *, usable_only: bool = True
    ) -> list[date]:
        return [
            d for d in self._dates if (start is None or d >= start) and (end is None or d <= end)
        ]


def _run_walkforward_cli(monkeypatch, tmp_path: Path, all_dates: list[date]):
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    panel = _wf_panel(all_dates)

    def _capturing(strat, test_panel, config, **kwargs):
        return _wf_fake_result(test_panel.n_rows(), test_panel)

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod, "load_universe", lambda name: Universe(name="ab", symbols=panel.symbols)
    )
    monkeypatch.setattr(engine_mod, "run_backtest", _capturing)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)),
    )

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


def test_item8_walkforward_reports_too_few_periods_when_dsr_is_nan(
    tmp_path: Path, monkeypatch
) -> None:
    """Spec item 8, caller half: a walk-forward pooled test window of exactly T=3
    sessions (round(0.012 * 252) == 3 -- see the `train-years`/`test-years` choice) makes
    `deflated_sharpe` return nan; the caller must say "too few periods" rather than
    presenting a bare `DSR=nan` as if it were an ordinary computation. Must FAIL against
    current code: `nq walkforward` already prints `DSR=nan` today with no explanation at
    all (confirmed at authoring time with an unmodified checkout)."""
    all_dates = pd.bdate_range("2020-01-01", periods=20).date.tolist()
    result = _run_walkforward_cli(monkeypatch, tmp_path, all_dates)

    assert result.exit_code == 0, result.output
    assert "n_days=3" in result.output, (
        "fixture assumption broken: expected exactly one 3-session walk-forward test "
        f"split; got:\n{result.output}"
    )
    assert "DSR=nan" in result.output, (
        f"fixture assumption broken: expected DSR=nan at T=3; got:\n{result.output}"
    )
    assert re.search(r"(?i)too few periods", result.output), (
        "a nan DSR caused by T < 4 periods must be explained as 'too few periods', not "
        f"presented as a bare computation failure; got:\n{result.output}"
    )
