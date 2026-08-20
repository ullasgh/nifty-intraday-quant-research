"""Tests for the PBO/DSR wiring specification.

Specification defects, ambiguities, and contradictions noticed:

* D2 offers two alternative fixes for the split-count guard — either shrink
  n_splits via ``min(16, T - (T % 2))``, or refuse below 16 sessions — but
  Required Test 4 ("A sweep with T < n_splits REFUSES...") only accepts the
  refuse branch; if an implementer picks the shrink-alternative, item 4 is
  unsatisfiable for any T in [2, 16).
* Item 2 says that explain() names the resolved artifact path, timestamp, and
  split ID, but it does not specify the required explanatory format or whether
  the path must be absolute, relative, or merely represented by its basename.
  Its requirement that the older split ID not occur also appears to prohibit
  explaining discarded candidates, although the current explain() summary is
  candidate-oriented.
* Item 6 says n_eff should be reported as unavailable when no matrix can be
  assembled, but does not specify whether that means NaN, text, a sentinel, or
  a change to the verdict-line format. The requested regex only partially
  constrains the acceptable output.
* Item 8 says the caller reports "too few periods", but does not identify the
  caller or specify the exact capitalization, punctuation, or surrounding
  wording.
* Item 4 discusses the current pbo_cscv error text and asks that it not be the
  whole story, but it is ambiguous whether that literal text must remain in
  the final caller error or whether only the two numeric counts are required.
* Item 3 requires a finite PBO value from a two-column sweep but does not
  specify whether the value must be printed by sweep itself, by a helper
  summary, or in stdout versus stderr; the regex searches the combined CLI
  output.
* Item 7 uses the float effective trial count as the first argument to
  expected_max_sharpe(), whose documented signature calls that argument an
  int, without explicitly requiring that the implementation accept non-integer
  effective counts.
* Item 6 pre-seeds trials with result_path=None and simultaneously refers to
  assembling a zero-column matrix; the registry API does not specify the
  exact shape or period axis returned for an empty matrix.
* AMENDMENT 1 (added to the spec after this file's first draft) restates item
  7 and adds Required Test 9: honest n_eff < 2 must report sr0 = 0.0 with an
  explicit single-trial warning rather than raising ValueError from
  expected_max_sharpe (which requires n_trials >= 2). The amendment does not
  specify the exact wording of the warning beyond "note that the trials are
  effectively a SINGLE trial"; test_9 below accepts a loose paraphrase.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.daily import build_daily
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.backtest.metrics import (
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
)
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.research.registry import TrialRecord, TrialRegistry
from nifty_quant.research.splits import Split
from nifty_quant.universe.static import Universe

runner = CliRunner()


def _session_ts(
    session_date: date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(session_date, time(*start_hhmm), tzinfo=ist)
    # 60 seconds is an arbitrary test-fixture bar spacing, not a claim about
    # real markets.
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _make_panel(
    session_dates: list[date],
    bars_per_session: list[int],
    symbols: tuple[str, ...] = ("A", "B"),
    close_price: float = 100.5,
    volume_value: float = 1000.0,
) -> Panel:
    # These price and volume values are arbitrary test-fixture constants,
    # chosen to keep the synthetic panel valid, not claims about real markets.
    ts = np.concatenate(
        [_session_ts(d, n) for d, n in zip(session_dates, bars_per_session)]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.cumsum(bars_per_session).astype(np.int32),
        ]
    )
    n_rows = int(ts.size)
    n_symbols = len(symbols)
    fields = {
        "open": np.full(
            (n_rows, n_symbols), close_price - 0.5, dtype=np.float64
        ),
        "high": np.full(
            (n_rows, n_symbols), close_price + 0.5, dtype=np.float64
        ),
        "low": np.full(
            (n_rows, n_symbols), close_price - 1.0, dtype=np.float64
        ),
        "close": np.full(
            (n_rows, n_symbols), close_price, dtype=np.float64
        ),
        "volume": np.full(
            (n_rows, n_symbols), volume_value, dtype=np.float64
        ),
    }
    return Panel(
        fields,
        symbols,
        ts,
        day_offsets,
        np.asarray(session_dates, dtype=object),
    )


def _row_day_index(panel: Panel) -> np.ndarray:
    n_rows = panel.n_rows()
    return np.searchsorted(
        panel.day_offsets,
        np.arange(n_rows, dtype=np.int64),
        side="right",
    ) - 1


def _session_dates_epoch(panel: Panel) -> np.ndarray:
    return np.asarray(
        [
            int(
                datetime(
                    d.year,
                    d.month,
                    d.day,
                    tzinfo=timezone.utc,
                ).timestamp()
            )
            for d in panel.dates
        ],
        dtype=np.int64,
    )


def _fake_result(
    n_rows: int,
    panel: Panel,
    *,
    seed: int = 0,
    daily_returns: np.ndarray | None = None,
) -> BacktestResult:
    # Seed 0 and these return/cost constants are arbitrary test-fixture
    # constants, chosen to produce deterministic valid synthetic results, not
    # claims about real markets.
    rng = np.random.default_rng(seed)
    gross = rng.normal(
        loc=0.0006,
        scale=0.004,
        size=n_rows,
    ).astype(np.float64)
    turnover = np.full(n_rows, 1.0, dtype=np.float64)
    returns = (gross - 0.0002 * turnover).astype(np.float64)
    initial_capital = 1e7
    equity_curve = np.cumprod(1.0 + returns) * initial_capital
    daily = build_daily(
        _row_day_index(panel),
        equity_curve,
        returns,
        gross,
        turnover,
        _session_dates_epoch(panel),
        initial_capital=initial_capital,
    )
    if daily_returns is not None:
        daily = build_daily(
            _row_day_index(panel),
            equity_curve,
            returns,
            gross,
            turnover,
            _session_dates_epoch(panel),
            initial_capital=initial_capital,
        )
        object.__setattr__(
            daily,
            "returns",
            np.asarray(daily_returns, dtype=np.float64),
        )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=np.zeros((n_rows, 1)),
        trades=pd.DataFrame(
            {"symbol": [], "side": [], "qty": [], "price": []}
        ),
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


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    calls: list[dict[str, Any]],
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> None:
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    def _stub(strat: Any, panel_arg: Panel, config: Any, **kwargs: Any) -> BacktestResult:
        result = (
            result_factory(panel_arg)
            if result_factory is not None
            else _fake_result(panel_arg.n_rows(), panel_arg)
        )
        calls.append({"panel": panel_arg, "result": result})
        return result

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="ab", symbols=panel.symbols),
    )
    monkeypatch.setattr(engine_mod, "run_backtest", _stub)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


def _write_sweep_yaml(tmp_path: Path, hold_bars: list[int]) -> Path:
    import yaml

    yaml_path = tmp_path / "sweep.yaml"
    config = {
        "strategy": "volume_breakout",
        "base_params": {
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
        },
        "sweep": {"hold_bars": hold_bars},
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def _write_returns_parquet(
    artifact_dir: Path,
    n_rows: int,
    returns: np.ndarray | float,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 1704153600 and 0.001 are arbitrary test-fixture constants used for a
    # deterministic daily artifact, not claims about real markets.
    start_epoch = 1704153600
    if np.isscalar(returns):
        values = np.full(n_rows, float(returns), dtype=np.float64)
    else:
        values = np.asarray(returns, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "ts": (
                start_epoch + np.arange(n_rows, dtype=np.int64) * 86400
            ).astype(np.int64),
            "return": values,
        }
    )
    frame.to_parquet(artifact_dir / "returns.parquet", index=False)


def _record(
    *,
    config_hash: str,
    ts: str,
    split_id: str,
    strategy: str = "volume_breakout",
    sharpe_net: float | None = 0.1,
    result_path: str | None = None,
) -> TrialRecord:
    return TrialRecord(
        config_hash=config_hash,
        ts=ts,
        strategy=strategy,
        params_json="{}",
        split_id=split_id,
        purpose="exploration",
        sharpe_gross=None,
        sharpe_net=sharpe_net,
        n_trades=None,
        turnover=None,
        breakeven_bps=None,
        git_sha=None,
        data_fingerprint=None,
        code_version=None,
        wall_s=None,
        result_path=result_path,
        error=None,
    )


def _collision_fixture(tmp_path: Path) -> tuple[TrialRegistry, Path, Path]:
    # Three and twenty-two are arbitrary test-fixture row counts selected to
    # expose stale-artifact resolution, not claims about real markets.
    old_dir = tmp_path / "old_artifact_3d"
    new_dir = tmp_path / "new_artifact_22d"
    _write_returns_parquet(old_dir, 3, 0.001)
    _write_returns_parquet(new_dir, 22, 0.001)

    registry = TrialRegistry(tmp_path / "registry.sqlite")
    registry.record(
        _record(
            config_hash="collision_hash_abc123",
            ts="2024-01-01T00:00:00+00:00",
            split_id="wf_test_0",
            result_path=str(old_dir),
        )
    )
    registry.record(
        _record(
            config_hash="collision_hash_abc123",
            ts="2024-06-01T00:00:00+00:00",
            split_id="sweep_full",
            result_path=str(new_dir),
        )
    )
    return registry, old_dir, new_dir


def _make_correlated_fixture() -> tuple[np.ndarray, np.ndarray]:
    # This is the verified arbitrary test-fixture constant, chosen by running
    # effective_n_trials/expected_max_sharpe/deflated_sharpe on it directly and
    # confirming it lands in the needed range; not a claim about real markets.
    rng = np.random.default_rng(42)
    T = 30
    common = rng.normal(0.0008, 0.01, size=T)

    def _make_correlated_matrix(
        n_trials: int,
        common_weight: float,
        idio_scale: float,
        seed: int,
    ) -> np.ndarray:
        r = np.random.default_rng(seed)
        cols = []
        for _ in range(n_trials):
            idio = r.normal(0.0, idio_scale, size=T)
            cols.append(common_weight * common + (1 - common_weight) * idio)
        return np.stack(cols, axis=1)

    m3 = _make_correlated_matrix(
        3,
        common_weight=0.999,
        idio_scale=0.01,
        seed=7,
    )
    m5 = _make_correlated_matrix(
        5,
        common_weight=0.5,
        idio_scale=0.01,
        seed=123,
    )
    return m3, m5


def _pooled_for_dsr_fixture() -> np.ndarray:
    # Seed 999, T=40, and these distribution parameters are arbitrary
    # test-fixture constants, chosen by running the metrics directly and
    # confirming the needed range; not a claim about real markets.
    rng2 = np.random.default_rng(999)
    return rng2.normal(0.0009, 0.008, size=40)


def _route_cli_registry(
    monkeypatch: pytest.MonkeyPatch,
    registry: TrialRegistry,
) -> None:
    import nifty_quant.cli as cli_mod
    import nifty_quant.research.registry as registry_mod

    def factory(path: Any) -> TrialRegistry:
        return registry

    monkeypatch.setattr(registry_mod, "TrialRegistry", factory)
    if hasattr(cli_mod, "TrialRegistry"):
        monkeypatch.setattr(cli_mod, "TrialRegistry", factory)


class _FakeCalendar:
    def __init__(self, dates: list[date]) -> None:
        self._dates = list(dates)

    def session_dates(
        self,
        start: date | None = None,
        end: date | None = None,
        *,
        usable_only: bool = True,
    ) -> list[date]:
        return [
            d
            for d in self._dates
            if (start is None or d >= start)
            and (end is None or d <= end)
        ]


def _patch_walkforward(
    monkeypatch: pytest.MonkeyPatch,
    all_dates: list[date],
    split: Split,
) -> None:
    import nifty_quant.calendar as calendar_mod
    import nifty_quant.research.splits as splits_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(
            lambda cls, symbol="NIFTY50": _FakeCalendar(all_dates)
        ),
    )
    monkeypatch.setattr(
        splits_mod.WalkForwardSplitter,
        "split",
        lambda self, trading_dates, **kwargs: [split],
    )


def _walkforward_args(all_dates: list[date]) -> list[str]:
    return [
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
    ]


def test_trial_matrix_prefers_newest_artifact_for_config_hash(
    tmp_path: Path,
) -> None:
    registry, old_dir, new_dir = _collision_fixture(tmp_path)

    matrix = registry.build_trial_matrix(
        trial_ids=["collision_hash_abc123"]
    )

    assert matrix.matrix.shape[0] == 22
    assert matrix.period_index.size == 22


def test_trial_matrix_explain_names_resolved_newer_artifact_and_not_old_split(
    tmp_path: Path,
) -> None:
    registry, old_dir, new_dir = _collision_fixture(tmp_path)

    matrix = registry.build_trial_matrix(
        trial_ids=["collision_hash_abc123"]
    )
    explanation = matrix.explain()

    assert str(new_dir) in explanation or new_dir.name in explanation
    assert "2024-06-01T00:00:00+00:00" in explanation
    assert "sweep_full" in explanation
    assert "wf_test_0" not in explanation


def test_sweep_reports_finite_pbo_for_two_trials_and_twenty_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Twenty sessions and three bars per session are arbitrary test-fixture
    # constants chosen to provide a valid finite CSCV input, not claims about
    # real markets.
    session_dates = pd.bdate_range(
        "2020-01-01",
        periods=20,
    ).date.tolist()
    panel = _make_panel(session_dates, [3] * 20)
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    yaml_path = _write_sweep_yaml(tmp_path, [20, 30])
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(yaml_path),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )

    assert result.exit_code == 0
    match = re.search(
        r"PBO[=:\s]+([0-9]*\.?[0-9]+)",
        result.output,
        re.IGNORECASE,
    )
    assert match is not None, result.output
    pbo = float(match.group(1))
    assert math.isfinite(pbo)
    assert 0.0 <= pbo <= 1.0


def test_sweep_refuses_one_session_with_explicit_one_and_sixteen_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One session is an arbitrary test-fixture constant chosen to force the
    # split-count guard, not a claim about real markets.
    session_dates = pd.bdate_range(
        "2020-01-01",
        periods=1,
    ).date.tolist()
    panel = _make_panel(session_dates, [3])
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    yaml_path = _write_sweep_yaml(tmp_path, [20, 30])
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(yaml_path),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )

    match = re.search(
        r"PBO[=:\s]+([0-9]*\.?[0-9]+)",
        result.output,
        re.IGNORECASE,
    )
    if match is not None:
        pbo = float(match.group(1))
        assert not (math.isfinite(pbo) and 0.0 <= pbo <= 1.0)

    assert "1" in result.output
    assert "16" in result.output


def test_effective_n_trials_is_near_one_for_three_near_identical_trials() -> None:
    m3, _ = _make_correlated_fixture()

    n_eff = effective_n_trials(m3)

    assert n_eff < 3.0
    assert n_eff < 1.5


def test_walkforward_reports_n_eff_unavailable_when_no_artifact_matrix_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two failed exploration trials and five deterministic sessions are
    # arbitrary test-fixture constants selected to exercise the no-matrix
    # branch, not claims about real markets.
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    for i, sharpe in enumerate((0.1, 0.2)):
        registry.record(
            _record(
                config_hash=f"failed_trial_{i}",
                ts=f"2024-01-0{i + 1}T00:00:00+00:00",
                split_id=f"failed_split_{i}",
                sharpe_net=sharpe,
                result_path=None,
            )
        )

    session_dates = pd.bdate_range(
        "2020-01-01",
        periods=5,
    ).date.tolist()
    panel = _make_panel(session_dates, [3] * 5)
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls)
    _route_cli_registry(monkeypatch, registry)

    split = Split(
        id="fixed_no_matrix",
        train=(session_dates[0], session_dates[0]),
        test=(session_dates[0], session_dates[-1]),
        embargo_days=0,
    )
    _patch_walkforward(monkeypatch, session_dates, split)

    result = runner.invoke(app, _walkforward_args(session_dates))
    match = re.search(r"eff=([^)|]*)\)", result.output)
    assert match is not None, result.output

    field = match.group(1).strip()
    # raw_count is 3, not the 2 pre-seeded failed trials: cli.py's per-split loop
    # records walkforward's OWN new trial (purpose="exploration") for the one
    # patched Split *inside* the loop, before the pooled-summary n_trials() count
    # runs -- so n_trials sees the 2 pre-seeded rows plus this run's own 1 row.
    # Verified against the actual current (buggy) CLI output: "trials=3 (eff=3.000)".
    raw_count = 3
    raw_formatted = f"{float(raw_count):.3f}"
    assert field != raw_formatted

    unavailable = False
    try:
        parsed = float(field)
    except ValueError:
        parsed = None
    if parsed is not None and math.isnan(parsed):
        unavailable = True
    if any(
        token in field.lower()
        for token in ("not available", "unavailable", "n/a", "na", "no matrix")
    ):
        unavailable = True
    assert unavailable, result.output


def test_honest_effective_trial_count_lowers_sr0_and_raises_dsr() -> None:
    _, m5 = _make_correlated_fixture()
    pooled_for_dsr_check = _pooled_for_dsr_fixture()

    per_trial_sharpes = m5.mean(axis=0) / m5.std(axis=0, ddof=1)
    var = np.var(per_trial_sharpes, ddof=0)
    n_eff = effective_n_trials(m5)
    sr0_honest = expected_max_sharpe(n_eff, var)
    sr0_raw = expected_max_sharpe(5, var)

    # These are cross-checks for the verified arbitrary fixture, not alternate
    # ways of deriving the values used by the assertions.
    assert sr0_honest == pytest.approx(0.08711515639119406)
    assert sr0_raw == pytest.approx(0.13687142037575528)
    assert sr0_honest < sr0_raw

    dsr_honest = deflated_sharpe(
        pooled_for_dsr_check,
        sr0=sr0_honest,
    )
    dsr_raw = deflated_sharpe(
        pooled_for_dsr_check,
        sr0=sr0_raw,
    )
    assert dsr_honest > dsr_raw


def test_walkforward_uses_honest_n_eff_for_sr0_and_dsr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 7(b), CLI-level. Ground truth is read back from the SAME sqlite file
    the CLI itself writes to (`settings.RESULTS_ROOT / "trials.db"`, patched to
    `tmp_path / "trials.db"`), using the REAL `build_trial_matrix` /
    `effective_n_trials` / `expected_max_sharpe` / `deflated_sharpe` on the
    REGISTRY'S ACTUAL POST-RUN STATE -- this deliberately does not try to
    hand-simulate cli.py's internal trial bookkeeping (which, verified against
    the actual current code, ALSO records the walk-forward split's own new
    trial as a 6th "exploration" row, inside the per-split loop, before the
    pooled-summary n_trials()/var_trial_sharpes() count runs). Pre-seeding the
    5-trial `m5` fixture on the FIRST 30 of the panel's 40 real session dates
    (as real UTC-midnight epoch seconds, matching how cli.py writes
    `res.daily.dates` for its own trial) keeps the inner-join period axis
    non-empty once that unavoidable 6th trial is added: verified directly
    (I ran this exact registry-assembly against the real registry/metrics code)
    to give a (30, 6) matrix, n_eff ~= 3.377 (>= 2, so expected_max_sharpe
    accepts it), sr0_honest ~= 2.048 < sr0_raw ~= 2.840, and
    dsr_honest ~= 3.27e-40 > dsr_raw ~= 2.89e-74 for this exact fixture
    (`net_sharpe` values here are ANNUALIZED, per `sharpe_ratio()`'s
    `* sqrt(252)`, matching what cli.py actually records).
    """
    _, m5 = _make_correlated_fixture()
    pooled_for_dsr_check = _pooled_for_dsr_fixture()

    # Forty sessions and three bars per session are arbitrary test-fixture
    # constants selected to pin the pooled DSR input, not claims about markets.
    session_dates = pd.bdate_range(
        "2020-01-01",
        periods=40,
    ).date.tolist()
    panel = _make_panel(session_dates, [3] * 40)

    # The pre-seeded m5 columns are written on the FIRST 30 of these 40 real
    # session dates (not an arbitrary fixed epoch) so that the walk-forward
    # run's own new trial -- which necessarily gets recorded on all 40 real
    # dates -- still overlaps with them under build_trial_matrix's inner join.
    first_30_ts = np.asarray(
        [
            int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            for d in session_dates[:30]
        ],
        dtype=np.int64,
    )

    registry = TrialRegistry(tmp_path / "trials.db")
    for i in range(5):
        artifact_dir = tmp_path / f"exploration_artifact_{i}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"ts": first_30_ts, "return": m5[:, i].astype(np.float64)}
        ).to_parquet(artifact_dir / "returns.parquet", index=False)
        # `sharpe_ratio()` (what cli.py actually stores as `net_sharpe`) is the
        # ANNUALIZED per-period Sharpe (`* sqrt(252)`, rf=0 default), not the raw
        # per-period ratio -- match it here so `var_trial_sharpes` mixes the 5
        # pre-seeded rows with the walk-forward run's own 6th row on the SAME
        # basis, rather than silently understating their variance.
        realized_sharpe = float(m5[:, i].mean() / m5[:, i].std(ddof=1)) * math.sqrt(252)
        registry.record(
            _record(
                config_hash=f"exploration_hash_{i}",
                ts=f"2024-01-{i + 1:02d}T00:00:00+00:00",
                split_id=f"exploration_split_{i}",
                sharpe_net=realized_sharpe,
                result_path=str(artifact_dir),
            )
        )

    calls: list[dict[str, Any]] = []

    def result_factory(panel_arg: Panel) -> BacktestResult:
        # cli.py's walkforward calls run_backtest 4 times for a single split:
        # once for the split's own test window (the only call whose
        # `.daily.returns` feeds `pooled_net_list`/`pooled_net`), plus 3 more
        # for its post-loop latency-sensitivity diagnostic (latency 0/1/2),
        # which are NOT pooled and NOT re-recorded. This override always
        # returns the SAME fixed `daily_returns=pooled_for_dsr_check`
        # regardless of which of those 4 calls it is, so `pooled_net` inside
        # cli.py is deterministically exactly `pooled_for_dsr_check`.
        return _fake_result(
            panel_arg.n_rows(),
            panel_arg,
            daily_returns=pooled_for_dsr_check,
        )

    # settings.RESULTS_ROOT is patched to tmp_path here, so the CLI's own
    # `TrialRegistry(settings.RESULTS_ROOT / "trials.db")` call opens the SAME
    # sqlite file `registry` above just seeded into -- no registry-routing
    # monkeypatch needed.
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        calls,
        result_factory=result_factory,
    )

    split = Split(
        id="fixed_dsr",
        train=(session_dates[0], session_dates[0]),
        test=(session_dates[0], session_dates[-1]),
        embargo_days=0,
    )
    _patch_walkforward(monkeypatch, session_dates, split)

    result = runner.invoke(app, _walkforward_args(session_dates))
    assert result.exit_code == 0, result.output

    post_registry = TrialRegistry(tmp_path / "trials.db")
    ground_matrix = post_registry.build_trial_matrix(
        strategy="volume_breakout", purpose="exploration"
    )
    ground_raw_n = post_registry.n_trials(strategy="volume_breakout")
    ground_var = post_registry.var_trial_sharpes(strategy="volume_breakout")
    ground_n_eff = effective_n_trials(ground_matrix.matrix)
    assert ground_n_eff >= 2.0, (
        "fixture precondition: honest n_eff must be >= 2 for expected_max_sharpe "
        f"to accept it directly -- got {ground_n_eff} from matrix "
        f"{ground_matrix.matrix.shape}, explain(): {ground_matrix.explain()}"
    )

    sr0_honest = expected_max_sharpe(ground_n_eff, ground_var)
    sr0_raw = expected_max_sharpe(ground_raw_n, ground_var)
    assert sr0_honest < sr0_raw

    dsr_honest = deflated_sharpe(pooled_for_dsr_check, sr0=sr0_honest)
    dsr_raw = deflated_sharpe(pooled_for_dsr_check, sr0=sr0_raw)
    assert dsr_honest > dsr_raw

    sr0_match = re.search(r"SR0=(-?[0-9.]+)", result.output)
    dsr_match = re.search(r"DSR=(-?[0-9.]+)", result.output)
    assert sr0_match is not None, result.output
    assert dsr_match is not None, result.output

    printed_sr0 = float(sr0_match.group(1))
    printed_dsr = float(dsr_match.group(1))
    # The load-bearing assertions: today's buggy wiring prints sr0_raw/dsr_raw
    # (raw count fed into expected_max_sharpe); a correctly-wired CLI must
    # instead print sr0_honest/dsr_honest (from the assembled matrix's honest
    # effective_n_trials).
    assert abs(printed_sr0 - sr0_honest) < 1e-3, result.output
    assert not math.isclose(printed_sr0, sr0_raw, abs_tol=1e-3), result.output
    assert printed_dsr > dsr_raw, result.output


def test_deflated_sharpe_is_nan_for_three_periods() -> None:
    # These three fixed values are an arbitrary deterministic test fixture,
    # chosen solely to exercise the documented minimum-period behavior, not a
    # claim about real markets.
    returns = np.asarray([0.001, -0.002, 0.003], dtype=np.float64)

    assert math.isnan(deflated_sharpe(returns, sr0=0.0))


def test_walkforward_reports_too_few_periods_for_nan_dsr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three sessions and three bars per session are arbitrary test-fixture
    # constants chosen to exercise the caller's short-period explanation, not
    # claims about real markets.
    session_dates = pd.bdate_range(
        "2020-01-01",
        periods=3,
    ).date.tolist()
    panel = _make_panel(session_dates, [3] * 3)
    calls: list[dict[str, Any]] = []
    _patch_common(monkeypatch, tmp_path, panel, calls)

    split = Split(
        id="fixed_short_dsr",
        train=(session_dates[0], session_dates[0]),
        test=(session_dates[0], session_dates[-1]),
        embargo_days=0,
    )
    _patch_walkforward(monkeypatch, session_dates, split)

    result = runner.invoke(app, _walkforward_args(session_dates))

    assert "DSR=nan" in result.output
    # Deliberately "too few period" (singular stem), not just "too few": this
    # fixture's n_trades=5 (< min_trades_for_sharpe=30) already triggers the
    # UNRELATED, pre-existing "too few trades" warning today -- verified this is
    # a real false-positive risk by running this exact fixture against current
    # code and confirming "too few" already matches that trades warning even
    # though no "too few periods" explanation exists anywhere in the output.
    # Matching "too few period" specifically is what actually pins item 8's
    # requirement and is what is genuinely absent from current output.
    assert "too few period" in result.output.lower(), result.output


def test_walkforward_reports_sr0_zero_and_single_trial_warning_for_low_n_eff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Required Test 9 (AMENDMENT 1, added to the spec after this file's first
    draft): honest n_eff < 2 must report sr0 = 0.0 with an explicit single-trial
    warning, and must NOT call expected_max_sharpe (which raises ValueError for
    n_trials < 2) or crash. Fixture verified directly against the real
    TrialRegistry/effective_n_trials: 3 near-identical trials (m3) plus the
    walk-forward run's own unavoidable 4th "exploration" row (recorded inside
    the per-split loop, before the pooled-summary step -- see the note on this
    in test_walkforward_uses_honest_n_eff_for_sr0_and_dsr above) assemble into a
    (30, 4) matrix with effective_n_trials == 1.5817416339577108, i.e. < 2.
    Expected RED against current code: n_eff = float(n_trials) today is a raw
    count (== 4 here, always an integer >= 1), so expected_max_sharpe(4, var) is
    called unconditionally and prints some nonzero SR0, never "0.000", and never
    any single-trial warning.
    """
    n_sessions = 40  # Arbitrary test-fixture constant, not a claim about real markets.
    n_overlap = 30  # Arbitrary test-fixture constant, not a claim about real markets.
    n_seed_trials = 3  # Arbitrary test-fixture constant, not a claim about real markets.
    session_dates = pd.bdate_range(
        "2020-01-01", periods=n_sessions
    ).date.tolist()
    # Arbitrary test-fixture constant, not a claim about real markets.
    bars_per_session = [3] * n_sessions
    panel = _make_panel(session_dates, bars_per_session)

    # Arbitrary test-fixture constant, not a claim about real markets.
    rng = np.random.default_rng(42)
    common = rng.normal(
        0.0008,  # Arbitrary test-fixture constant, not a claim about real markets.
        0.01,  # Arbitrary test-fixture constant, not a claim about real markets.
        size=n_overlap,
    )
    common_weight = 0.999  # Arbitrary test-fixture constant, not a claim about real markets.
    idio_scale = 0.01  # Arbitrary test-fixture constant, not a claim about real markets.
    correlated_rng = np.random.default_rng(
        7
    )  # Arbitrary test-fixture constant, not a claim about real markets.
    columns = []
    for _ in range(n_seed_trials):
        idio = correlated_rng.normal(0.0, idio_scale, size=n_overlap)
        columns.append(
            common_weight * common + (1 - common_weight) * idio
        )
    m3 = np.stack(columns, axis=1)

    # int64 UTC-midnight epoch seconds on the FIRST n_overlap of the panel's real
    # session dates (not an arbitrary fixed epoch) -- build_trial_matrix's inner
    # join needs these to overlap the walk-forward run's own new trial, which is
    # always written on all n_sessions real dates. `ts` must be integer dtype or
    # build_trial_matrix drops the row as "ts column has non-integer dtype".
    first_30_ts = np.asarray(
        [
            int(
                datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
            )
            for d in session_dates[:n_overlap]
        ],
        dtype=np.int64,
    )
    registry = TrialRegistry(tmp_path / "trials.db")
    periods_per_year = 252  # matches sharpe_ratio()'s default, not an arbitrary pick.
    for trial_idx in range(n_seed_trials):
        trial_returns = m3[:, trial_idx]
        returns_path = tmp_path / f"seed_{trial_idx}.parquet"
        pd.DataFrame(
            {"ts": first_30_ts, "return": trial_returns.astype(np.float64)}
        ).to_parquet(returns_path, index=False)
        # Annualized, matching what cli.py's own net_sharpe recording actually
        # stores (sharpe_ratio() * sqrt(periods_per_year)) -- see the same note
        # in test_walkforward_uses_honest_n_eff_for_sr0_and_dsr above.
        sharpe_net = float(
            np.mean(trial_returns)
            / np.std(trial_returns, ddof=1)
            * np.sqrt(periods_per_year)
        )
        registry.record(
            _record(
                config_hash=f"seed-config-{trial_idx}",
                ts=f"2020-01-01T00:00:{trial_idx:02d}+00:00",
                split_id=f"seed-split-{trial_idx}",
                sharpe_net=sharpe_net,
                result_path=str(returns_path),
            )
        )

    fixed_returns = np.random.default_rng(999).normal(
        0.0009,  # Arbitrary test-fixture constant, not a claim about real markets.
        0.008,  # Arbitrary test-fixture constant, not a claim about real markets.
        size=n_sessions,
    )
    calls: list[dict[str, Any]] = []
    _patch_common(
        monkeypatch,
        tmp_path,
        panel,
        calls,
        result_factory=lambda p: _fake_result(
            p.n_rows(), p, daily_returns=fixed_returns
        ),
    )

    split = Split(
        id="fixture-split",
        train=(session_dates[0], session_dates[0]),
        test=(session_dates[0], session_dates[-1]),
        embargo_days=0,  # Arbitrary test-fixture constant, not a claim about real markets.
    )
    _patch_walkforward(monkeypatch, session_dates, split)

    result = runner.invoke(app, _walkforward_args(session_dates))
    # Item 9's core requirement: this must NOT raise/crash, even though honest
    # n_eff < 2 would make expected_max_sharpe(n_eff, var) raise ValueError if
    # called directly on it.
    assert result.exit_code == 0, result.output

    sr0_match = re.search(r"SR0=(-?[0-9.]+)", result.output)
    assert sr0_match is not None, result.output
    zero_tolerance = 1e-9
    assert float(sr0_match.group(1)) == pytest.approx(
        0.0, abs=zero_tolerance
    ), result.output

    output_lower = result.output.lower()
    assert (
        ("single" in output_lower and "trial" in output_lower)
        or ("n_eff" in output_lower and "trial" in output_lower)
    ), result.output
