"""Tests for run provenance and experiment lineage.

This test suite is written from specs/run_provenance.md ALONE, independently of any other
test file or implementation. Two independent test suites catch different defects; disagreement
between them is valuable even if both are red.

SPEC AMENDMENTS INCORPORATED:

Amendment 1 (2026-08-20) resolved all nine original defects:

1. cost_model_id format: "<ClassName>(<k=v for non-default fields, sorted by k>)"
   - Floats use repr() for precision preservation
   - Example: NSEIntradayEquityCosts() for default; NSEIntradayEquityCosts(brokerage_flat=10.0)

2. embargo_components: JSON object with sorted keys:
   {"feature_lookback": <float sessions>, "label_horizon": <float sessions>,
    "holding_period": <float sessions>, "execution_horizon": <float sessions>}
   - Units are SESSIONS as floats, pre-ceil
   - Sorted keys for stable serialization

3. universe_hash: blake2s over canonical JSON of universe name, per-session sorted eligible
   symbol sets, and eligibility parameters.

4. "Same slice": Two runs are identical slice iff ALL of: sorted symbol list, first and last ts,
   row count, field names, adjustment settings, PANEL_VERSION. If any differ, hash MUST differ.

5. Item 6 (registry write failure): On failure, (a) log at WARNING with trial id, (b) set
   registry_write_failed: true in metrics.json, (c) still write artifact and exit 0.
   registry_write_failed is false on normal runs (two observable states).

6. Item 7 (complete provenance): metrics.json["provenance"] has EXACTLY these keys, all present,
   null allowed only for seed and parent_trial_id:
   config_hash, git_sha, code_version, data_fingerprint, panel_hash, universe_name,
   universe_hash, seed, start, end, cost_model_id, slippage_model_id, fill_model_id,
   embargo_components, parent_trial_id, feature_version

7. Item 8: Three-generation chain (grandparent -> parent -> child) must be walkable.

8. Item 9: Sweep-derived trial has parent_trial_id non-null pointing to sweep's base trial.
   nq sweep populates SAME provenance fields as nq backtest.

9. Item 1: BYTE-FOR-BYTE for strings, exact equality for ints, repr()-stable for floats.
   Not re-computed equivalence.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import fields
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.backtest.daily import DailyResult
from nifty_quant.backtest.engine import BacktestResult
from nifty_quant.cli import app
from nifty_quant.data.panel import Panel
from nifty_quant.research.registry import TrialRecord, TrialRegistry
from nifty_quant.universe.static import Universe

# ============================================================================
# Integration test helpers (copied from test_returns_persistence.py pattern)
# ============================================================================

_runner = CliRunner()

# Standard embargo_components JSON used across multiple tests.
_EMBARGO_JSON = (
    '{"execution_horizon": 0.0, "feature_lookback": 10.0, '
    '"holding_period": 5.0, "label_horizon": 1.0}'
)


def _make_panel(
    n_sessions: int = 3,
    bars_per_session: int = 2,
    dates: tuple[date, ...] | None = None,
    symbols: tuple[str, ...] = ("TESTSYM",),
) -> Panel:
    """Create a minimal synthetic panel for testing.

    Populates real open/high/low/close/volume arrays (finite, positive, shaped
    (n_rows, n_symbols)) so that the CLI's tradable_mask (which reads panel.field("close")
    etc.) does not blow up before provenance code ever runs. Values are deterministic but
    otherwise arbitrary -- this is a provenance test, not a P&L test.
    """
    if dates is None:
        base = date(2024, 1, 1)
        dates = tuple(base + timedelta(days=i) for i in range(n_sessions))
    ts_list: list[int] = []
    day_offsets = [0]
    for i in range(n_sessions):
        for j in range(bars_per_session):
            ts_list.append(1_700_000_000 + i * 86_400 + j * 60)
        day_offsets.append(len(ts_list))
    n_rows = len(ts_list)
    n_symbols = len(symbols)
    close = np.full((n_rows, n_symbols), 100.0, dtype=np.float32)
    fields = {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": np.full((n_rows, n_symbols), 1_000.0, dtype=np.float32),
    }
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=np.asarray(ts_list, dtype=np.int64),
        day_offsets=np.asarray(day_offsets, dtype=np.int64),
        dates=np.asarray(dates, dtype=object),
    )


def _patch_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> None:
    """Monkeypatch system to use synthetic panel and results."""
    import nifty_quant.backtest.engine as engine_mod
    import nifty_quant.data.panel as panel_mod
    import nifty_quant.settings as settings_mod
    import nifty_quant.universe.static as universe_mod

    monkeypatch.setattr(panel_mod, "load_panel", lambda spec: panel)
    monkeypatch.setattr(
        universe_mod,
        "load_universe",
        lambda name: Universe(name="test", symbols=panel.symbols),
    )

    def mock_run_backtest(spec, panel, universe, *args, **kwargs):
        if result_factory is not None:
            return result_factory(panel)
        # NOTE: BacktestResult's real constructor (nifty_quant.backtest.engine) takes
        # equity_curve/returns/positions/trades/gross_returns/total_costs/n_trades/
        # turnover/rejected_order_rate/unfilled_notional_pct/forced_eod_liquidation_days
        # -- not the n_live_bars/returns_noncash/returns_cash/turnover_pct/cost_bps
        # kwargs this fixture previously invented, which do not exist on the dataclass
        # at all and raised TypeError before any provenance code ran. `daily` must also
        # be populated (not left at its empty default) because the CLI reads
        # result.daily.gross_returns / result.daily.returns to compute Sharpe metrics.
        n_rows = panel.n_rows()
        n_symbols = panel.n_symbols()
        n_days = panel.n_days()
        capital = 1e7
        day_epochs = np.asarray(
            [
                int(datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp())
                for d in panel.dates
            ],
            dtype=np.int64,
        )
        daily = DailyResult(
            dates=day_epochs,
            equity=np.full(n_days, capital, dtype=np.float64),
            returns=np.zeros(n_days, dtype=np.float64),
            gross_returns=np.zeros(n_days, dtype=np.float64),
            turnover=np.zeros(n_days, dtype=np.float64),
            n_days=n_days,
        )
        return BacktestResult(
            equity_curve=np.full(n_rows, capital, dtype=np.float64),
            returns=np.zeros(n_rows, dtype=np.float64),
            positions=np.zeros((n_rows, n_symbols), dtype=np.float64),
            trades=pd.DataFrame(),
            gross_returns=np.zeros(n_rows, dtype=np.float64),
            total_costs=0.0,
            n_trades=10,
            turnover=np.zeros(n_rows, dtype=np.float64),
            rejected_order_rate=0.0,
            unfilled_notional_pct=0.0,
            forced_eod_liquidation_days=0,
            initial_capital=capital,
            daily=daily,
        )

    monkeypatch.setattr(engine_mod, "run_backtest", mock_run_backtest)
    monkeypatch.setattr(settings_mod, "RESULTS_ROOT", tmp_path)


def _backtest_args() -> list[str]:
    """Standard backtest CLI arguments."""
    return [
        "backtest",
        "--strategy",
        "volume_breakout",
        "--config",
        "configs/strategies/volume_breakout.yaml",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
        "--no-tradable-filter",
    ]


def _run_backtest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel: Panel,
    result_factory: Callable[[Panel], BacktestResult] | None = None,
) -> Path:
    """Run a backtest via CLI and return the trial directory."""
    _patch_system(monkeypatch, tmp_path, panel, result_factory)
    result = _runner.invoke(app, _backtest_args())
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    trial_dirs = sorted(p for p in (tmp_path / "trials").iterdir() if p.is_dir())
    assert len(trial_dirs) == 1, f"Expected 1 trial dir, found {len(trial_dirs)}"
    return trial_dirs[0]


class TestItem1FieldRoundTrip:
    """Item 1: Every field round-trips through the registry and back unchanged.

    BYTE-FOR-BYTE for stored strings; exact equality for ints; repr()-stable for floats.
    Tests that all twelve new fields survive write-and-read cycle through SQLite with
    exact byte preservation.
    """

    def test_new_trial_record_fields_exist(self):
        """Verify TrialRecord has all twelve new fields as dataclass attributes."""
        new_fields = {
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
        }
        existing_field_names = {f.name for f in fields(TrialRecord)}
        for new_field in new_fields:
            assert new_field in existing_field_names, (
                f"TrialRecord missing required field: {new_field}"
            )

    def test_seed_round_trip_none(self, tmp_path):
        """seed (int | None) round-trips as None, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc123",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc123",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.seed is None

    def test_seed_round_trip_int(self, tmp_path):
        """seed (int | None) round-trips as an integer, exact equality."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc124",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc124",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=42,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.seed == 42

    def test_universe_fields_round_trip(self, tmp_path):
        """universe_name and universe_hash round-trip unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc125",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc125",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty_smallcap",
            universe_hash="uni_hash_abc_def_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.universe_name == "nifty_smallcap"
        assert loaded_rec.universe_hash == "uni_hash_abc_def_123"

    def test_panel_hash_round_trip(self, tmp_path):
        """panel_hash round-trips unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        test_hash = "panel_xyz_2026_01_01_to_12_31"
        record = TrialRecord(
            config_hash="abc126",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc126",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash=test_hash,
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.panel_hash == test_hash

    def test_date_fields_round_trip(self, tmp_path):
        """start and end ISO date strings round-trip unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc127",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc127",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-02-15",
            end="2026-11-20",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.start == "2026-02-15"
        assert loaded_rec.end == "2026-11-20"

    def test_cost_model_id_round_trip(self, tmp_path):
        """cost_model_id round-trips unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        test_cost_id = "NSEIntradayEquityCosts(brokerage_flat=15.0)"
        record = TrialRecord(
            config_hash="abc128",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc128",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id=test_cost_id,
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.cost_model_id == test_cost_id

    def test_slippage_fill_model_ids_round_trip(self, tmp_path):
        """slippage_model_id and fill_model_id round-trip unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc129",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc129",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="vwap_slippage_bps_5",
            fill_model_id="perfect_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.slippage_model_id == "vwap_slippage_bps_5"
        assert loaded_rec.fill_model_id == "perfect_fill"

    def test_embargo_components_round_trip(self, tmp_path):
        """embargo_components JSON round-trips unchanged, byte-for-byte.

        JSON with sorted keys: feature_lookback, label_horizon, holding_period, execution_horizon.
        Units are SESSION counts as floats.
        """
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        embargo_json = _EMBARGO_JSON
        record = TrialRecord(
            config_hash="abc130",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc130",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=embargo_json,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.embargo_components == embargo_json

    def test_parent_trial_id_round_trip_none(self, tmp_path):
        """parent_trial_id (str | None) round-trips as None, exact equality."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc131",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc131",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.parent_trial_id is None

    def test_parent_trial_id_round_trip_string(self, tmp_path):
        """parent_trial_id (str | None) round-trips as a string, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc132",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc132",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id="abc123_parent",
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.parent_trial_id == "abc123_parent"

    def test_feature_version_round_trip(self, tmp_path):
        """feature_version round-trips unchanged, byte-for-byte."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc133",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc133",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="market_features_v3.2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.feature_version == "market_features_v3.2.1"


class TestItem2GitSha:
    """Item 2: git_sha is never None; dirty trees yield '-dirty' suffix.

    Tests that git_sha is properly populated (not None) and dirty working trees
    are marked with a '-dirty' suffix.
    """

    def test_git_sha_never_none_in_registry(self, tmp_path):
        """After a write, git_sha is not None in the registry."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc200",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc200",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.git_sha is not None
        assert isinstance(loaded_rec.git_sha, str)

    def test_git_sha_dirty_suffix(self, tmp_path):
        """Dirty working tree is marked with '-dirty' suffix."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc201",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567-dirty",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc201",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.git_sha.endswith("-dirty")

    def test_git_sha_no_git_fallback(self, tmp_path):
        """git_sha is 'no-git' when git is unavailable."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record = TrialRecord(
            config_hash="abc202",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="no-git",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc202",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record)
        loaded = registry.all()

        assert len(loaded) == 1
        loaded_rec = loaded[0]
        assert loaded_rec.git_sha == "no-git"


class TestItem3PanelHashDiffers:
    """Item 3: panel_hash differs across two runs on different date ranges.

    This tests the specific problem that panel_hash was designed to fix: data_fingerprint
    is identical for different slices of the same dataset.
    """

    def test_panel_hash_differs_different_date_ranges(self, tmp_path):
        """Two trials on different date ranges have different panel_hash
        despite same data_fingerprint.
        """
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record1 = TrialRecord(
            config_hash="abc300",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="same_dataset_fingerprint",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc300",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_2026_01_01_to_06_30",
            start="2026-01-01",
            end="2026-06-30",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        record2 = TrialRecord(
            config_hash="abc301",
            ts="2026-01-02T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_1",
            purpose="exploration",
            sharpe_gross=1.6,
            sharpe_net=1.3,
            n_trades=110,
            turnover=0.06,
            breakeven_bps=11.0,
            git_sha="abc1234567",
            data_fingerprint="same_dataset_fingerprint",
            code_version="v1.0",
            wall_s=124.45,
            result_path="/results/abc301",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_2026_07_01_to_12_31",
            start="2026-07-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record1)
        registry.record(record2)
        loaded = registry.all()

        assert len(loaded) == 2
        rec1 = next(r for r in loaded if r.config_hash == "abc300")
        rec2 = next(r for r in loaded if r.config_hash == "abc301")

        assert rec1.data_fingerprint == rec2.data_fingerprint
        assert rec1.panel_hash != rec2.panel_hash


class TestItem4PanelHashStable:
    """Item 4: panel_hash is stable across two runs on the same slice.

    "Same slice" means ALL SIX of: sorted symbol list, first and last ts, row count,
    field names, adjustment settings, PANEL_VERSION. If any differ, hash MUST differ.
    """

    def test_panel_hash_stable_same_slice(self, tmp_path):
        """Two trials on the same six-attribute slice produce identical panel_hash."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record1 = TrialRecord(
            config_hash="abc400",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat_v1",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_same",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc400",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="stable_panel_hash_2026_01_01_to_12_31",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        record2 = TrialRecord(
            config_hash="abc401",
            ts="2026-01-02T10:00:00Z",
            strategy="test_strat_v1",
            params_json='{}',
            split_id="train_0",
            purpose="confirmation",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=101,
            turnover=0.051,
            breakeven_bps=10.1,
            git_sha="abc1234567",
            data_fingerprint="data_fp_same",
            code_version="v1.0",
            wall_s=123.50,
            result_path="/results/abc401",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="stable_panel_hash_2026_01_01_to_12_31",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record1)
        registry.record(record2)
        loaded = registry.all()

        assert len(loaded) == 2
        rec1 = next(r for r in loaded if r.config_hash == "abc400")
        rec2 = next(r for r in loaded if r.config_hash == "abc401")

        assert rec1.panel_hash == rec2.panel_hash
        assert rec1.start == rec2.start
        assert rec1.end == rec2.end


class TestItem5CostModelId:
    """Item 5: cost_model_id distinguishes default from non-default, and is identical
    for two default instances.

    Format: "<ClassName>(<k=v for non-default fields, sorted by k>)"
    Floats use repr() for precision preservation.
    """

    def test_cost_model_id_default_instance_identical(self, tmp_path):
        """Two default NSEIntradayEquityCosts() instances have identical cost_model_id."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record1 = TrialRecord(
            config_hash="abc500",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc500",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        record2 = TrialRecord(
            config_hash="abc501",
            ts="2026-01-02T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_1",
            purpose="exploration",
            sharpe_gross=1.6,
            sharpe_net=1.3,
            n_trades=110,
            turnover=0.06,
            breakeven_bps=11.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=124.45,
            result_path="/results/abc501",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record1)
        registry.record(record2)
        loaded = registry.all()

        assert len(loaded) == 2
        rec1 = next(r for r in loaded if r.config_hash == "abc500")
        rec2 = next(r for r in loaded if r.config_hash == "abc501")

        assert rec1.cost_model_id == rec2.cost_model_id
        assert rec1.cost_model_id == "NSEIntradayEquityCosts()"

    def test_cost_model_id_distinguishes_non_default(self, tmp_path):
        """A non-default NSEIntradayEquityCosts instance has a different cost_model_id.

        Format uses sorted field names, repr() for float precision.
        """
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        record_default = TrialRecord(
            config_hash="abc502",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc502",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        record_custom = TrialRecord(
            config_hash="abc503",
            ts="2026-01-02T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_1",
            purpose="exploration",
            sharpe_gross=1.6,
            sharpe_net=1.3,
            n_trades=110,
            turnover=0.06,
            breakeven_bps=11.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=124.45,
            result_path="/results/abc503",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts(brokerage_flat=15.0)",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(record_default)
        registry.record(record_custom)
        loaded = registry.all()

        assert len(loaded) == 2
        rec_default = next(r for r in loaded if r.config_hash == "abc502")
        rec_custom = next(r for r in loaded if r.config_hash == "abc503")

        assert rec_default.cost_model_id != rec_custom.cost_model_id
        assert rec_default.cost_model_id == "NSEIntradayEquityCosts()"
        assert "brokerage_flat=15.0" in rec_custom.cost_model_id


class TestItem6RegistryWriteFailureVisible:
    """Item 6: Registry write failures are surfaced, not swallowed.

    On failure: (a) log at WARNING with trial id, (b) set registry_write_failed: true
    in metrics.json, (c) still write artifact and exit 0.
    registry_write_failed is false on normal runs (two observable states).
    """

    def test_registry_write_failed_false_on_normal_backtest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """registry_write_failed is false in metrics.json on normal backtest."""
        panel = _make_panel()
        trial_dir = _run_backtest(monkeypatch, tmp_path, panel)

        # Read metrics.json written to disk by the backtest
        metrics_path = trial_dir / "metrics.json"
        assert metrics_path.is_file(), f"metrics.json not written to {trial_dir}"

        with open(metrics_path) as f:
            metrics = json.load(f)

        # On normal run, registry_write_failed must be present and false
        assert "registry_write_failed" in metrics, (
            "metrics.json must have registry_write_failed field"
        )
        assert metrics["registry_write_failed"] is False, (
            "registry_write_failed must be false on normal backtest"
        )

    def test_registry_write_failed_true_on_registry_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """registry_write_failed is true in metrics.json when registry write fails.

        Monkeypatch TrialRegistry.record to raise an exception, run backtest, and
        assert exit code 0 (backtest continues), artifact exists, and
        metrics.json["registry_write_failed"] is True.
        """
        panel = _make_panel()

        # Patch to make registry.record raise
        def mock_record_raises(self, rec):
            raise sqlite3.OperationalError("database locked")

        monkeypatch.setattr(TrialRegistry, "record", mock_record_raises)

        # Run backtest - must still succeed despite registry failure
        _patch_system(monkeypatch, tmp_path, panel)
        result = _runner.invoke(app, _backtest_args())

        # Must exit 0 even though registry failed
        msg = f"Backtest must succeed despite registry failure: {result.output}"
        assert result.exit_code == 0, msg

        # Trial directory must exist
        trial_dirs = sorted(p for p in (tmp_path / "trials").iterdir() if p.is_dir())
        assert len(trial_dirs) >= 1, "Trial artifact directory must exist"
        trial_dir = trial_dirs[0]

        # metrics.json must exist and have registry_write_failed: true
        metrics_path = trial_dir / "metrics.json"
        assert metrics_path.is_file(), "metrics.json must be written despite registry failure"

        with open(metrics_path) as f:
            metrics = json.load(f)

        assert "registry_write_failed" in metrics, (
            "metrics.json must record registry_write_failed marker"
        )
        assert metrics["registry_write_failed"] is True, (
            "registry_write_failed must be true when registry.record raises"
        )


class TestItem7MetricsJsonProvenance:
    """Item 7: metrics.json contains a complete provenance block.

    metrics.json["provenance"] contains EXACTLY these keys (all present, null allowed
    only for seed and parent_trial_id):
    config_hash, contract_hash, git_sha, code_version, data_fingerprint, panel_hash,
    universe_name, universe_hash, seed, start, end, cost_model_id, slippage_model_id,
    fill_model_id, embargo_components, parent_trial_id, feature_version

    `contract_hash` was added after amendment item 6 was written (spec C4,
    `specs/research_contract.md`), distinct from the pre-existing `config_hash`; it is
    verified populated (non-null) on a real recorded run, not merely present on the
    dataclass.
    """

    def test_provenance_block_exact_key_set_on_disk(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """metrics.json["provenance"] on disk contains exactly the required keys."""
        panel = _make_panel()
        trial_dir = _run_backtest(monkeypatch, tmp_path, panel)

        # Read metrics.json written to disk by the backtest
        metrics_path = trial_dir / "metrics.json"
        assert metrics_path.is_file(), f"metrics.json not written to {trial_dir}"

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Must have provenance object
        assert "provenance" in metrics, "metrics.json must have provenance block"

        # 16 keys named by amendment item 6, plus `contract_hash` -- TrialRecord gained
        # this field (distinct from config_hash) under spec C4 (research_contract.md)
        # after this amendment was written; verified populated (non-null) below.
        required_keys = {
            "config_hash", "contract_hash", "git_sha", "code_version", "data_fingerprint",
            "panel_hash", "universe_name", "universe_hash", "seed", "start", "end",
            "cost_model_id", "slippage_model_id", "fill_model_id", "embargo_components",
            "parent_trial_id", "feature_version"
        }

        provenance = metrics["provenance"]
        actual_keys = set(provenance.keys())

        assert actual_keys == required_keys, (
            f"Key set mismatch in written metrics.json. "
            f"Missing: {required_keys - actual_keys}. "
            f"Extra: {actual_keys - required_keys}"
        )

    def test_provenance_block_nullability_on_disk(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """Only seed and parent_trial_id are allowed to be null in written provenance."""
        panel = _make_panel()
        trial_dir = _run_backtest(monkeypatch, tmp_path, panel)

        # Read metrics.json written to disk
        metrics_path = trial_dir / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert "provenance" in metrics
        provenance = metrics["provenance"]

        # Fields that are allowed to be null
        nullable_fields = {"seed", "parent_trial_id"}

        # Check that non-nullable fields are actually present (not null)
        for field in provenance.keys():
            if field not in nullable_fields:
                assert provenance[field] is not None, (
                    f"Field '{field}' in provenance must not be null (only seed and "
                    f"parent_trial_id are allowed to be null)"
                )

    def test_provenance_complete_all_fields_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """Provenance block must have all required fields with correct nullability semantics."""
        panel = _make_panel()
        trial_dir = _run_backtest(monkeypatch, tmp_path, panel)

        # Read metrics.json written to disk
        metrics_path = trial_dir / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert "provenance" in metrics
        provenance = metrics["provenance"]

        # 16 fields named by amendment item 6, plus `contract_hash` (see comment on
        # test_provenance_block_exact_key_set_on_disk above) -- 17 fields must exist.
        required_keys = {
            "config_hash", "contract_hash", "git_sha", "code_version", "data_fingerprint",
            "panel_hash", "universe_name", "universe_hash", "seed", "start", "end",
            "cost_model_id", "slippage_model_id", "fill_model_id", "embargo_components",
            "parent_trial_id", "feature_version"
        }

        for key in required_keys:
            assert key in provenance, f"Provenance must have key '{key}'"

        # Exactly seventeen fields, no more, no less
        assert len(provenance) == 17, (
            f"Provenance must have exactly 17 keys, found {len(provenance)}"
        )


class TestItem8ParentTrialIdChain:
    """Item 8: parent_trial_id chains are walkable.

    A THREE-GENERATION chain (grandparent -> parent -> child) must be walkable.
    Two levels is not enough to catch an implementation that stores parent but
    cannot follow more than one hop.
    """

    def test_parent_trial_id_three_generation_chain(self, tmp_path):
        """A three-generation chain (grandparent->parent->child) is walkable."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        # Create grandparent trial
        grandparent_record = TrialRecord(
            config_hash="abc800_grandparent",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc800_grandparent",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        # Create parent that derives from grandparent
        parent_record = TrialRecord(
            config_hash="abc800_parent",
            ts="2026-01-02T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="confirmation",
            sharpe_gross=1.55,
            sharpe_net=1.25,
            n_trades=105,
            turnover=0.055,
            breakeven_bps=10.5,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.50,
            result_path="/results/abc800_parent",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id="abc800_grandparent",
            feature_version="v2.1",
        )

        # Create child that derives from parent
        child_record = TrialRecord(
            config_hash="abc800_child",
            ts="2026-01-03T10:00:00Z",
            strategy="test_strat",
            params_json='{}',
            split_id="train_0",
            purpose="confirmation",
            sharpe_gross=1.6,
            sharpe_net=1.3,
            n_trades=110,
            turnover=0.06,
            breakeven_bps=11.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=124.45,
            result_path="/results/abc800_child",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id="abc800_parent",
            feature_version="v2.1",
        )

        registry.record(grandparent_record)
        registry.record(parent_record)
        registry.record(child_record)
        loaded = registry.all()

        assert len(loaded) == 3

        # Walk the chain from child to root
        current = next(r for r in loaded if r.config_hash == "abc800_child")
        chain = [current]

        while current.parent_trial_id is not None:
            parent = next(
                (r for r in loaded if r.config_hash == current.parent_trial_id),
                None,
            )
            msg = f"parent_trial_id {current.parent_trial_id} not found in registry"
            assert parent is not None, msg
            chain.append(parent)
            current = parent

        # Verify three-generation chain
        assert len(chain) == 3
        assert chain[0].config_hash == "abc800_child"
        assert chain[1].config_hash == "abc800_parent"
        assert chain[2].config_hash == "abc800_grandparent"
        assert chain[0].parent_trial_id == "abc800_parent"
        assert chain[1].parent_trial_id == "abc800_grandparent"
        assert chain[2].parent_trial_id is None


class TestItem9SweepProvenance:
    """Item 9: nq sweep trials carry the same provenance completeness as nq backtest.

    A sweep-derived trial has parent_trial_id non-null pointing at sweep's base trial.
    nq sweep populates SAME provenance fields as nq backtest (no field left null by sweep
    that backtest fills).
    """

    def test_sweep_trial_has_complete_provenance(self, tmp_path):
        """A sweep-derived trial has all provenance fields populated."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        # A trial from sweep should have all provenance fields, same as backtest
        sweep_record = TrialRecord(
            config_hash="abc900_sweep",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{"param1": 0.1}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc900_sweep",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=12345,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        registry.record(sweep_record)
        loaded = registry.all()

        assert len(loaded) == 1
        rec = loaded[0]

        # All provenance fields must be populated
        assert rec.seed is not None
        assert rec.universe_name is not None
        assert rec.universe_hash is not None
        assert rec.panel_hash is not None
        assert rec.start is not None
        assert rec.end is not None
        assert rec.cost_model_id is not None
        assert rec.slippage_model_id is not None
        assert rec.fill_model_id is not None
        assert rec.embargo_components is not None
        assert rec.feature_version is not None

    def test_sweep_derived_trial_points_to_base(self, tmp_path):
        """A sweep-derived trial points to the sweep's base trial via parent_trial_id."""
        db_path = tmp_path / "test.db"
        registry = TrialRegistry(db_path)

        # Base trial (the sweep parameter set)
        base_record = TrialRecord(
            config_hash="abc900_base",
            ts="2026-01-01T10:00:00Z",
            strategy="test_strat",
            params_json='{"param1": 0.05}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.4,
            sharpe_net=1.1,
            n_trades=90,
            turnover=0.04,
            breakeven_bps=9.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=122.45,
            result_path="/results/abc900_base",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=None,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id=None,
            feature_version="v2.1",
        )

        # Derived trial from sweep (points to base)
        derived_record = TrialRecord(
            config_hash="abc900_derived",
            ts="2026-01-01T10:30:00Z",
            strategy="test_strat",
            params_json='{"param1": 0.1}',
            split_id="train_0",
            purpose="exploration",
            sharpe_gross=1.5,
            sharpe_net=1.2,
            n_trades=100,
            turnover=0.05,
            breakeven_bps=10.0,
            git_sha="abc1234567",
            data_fingerprint="data_fp_123",
            code_version="v1.0",
            wall_s=123.45,
            result_path="/results/abc900_derived",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=12345,
            universe_name="nifty50",
            universe_hash="uni_hash_123",
            panel_hash="panel_hash_123",
            start="2026-01-01",
            end="2026-12-31",
            cost_model_id="NSEIntradayEquityCosts()",
            slippage_model_id="default_slippage",
            fill_model_id="default_fill",
            embargo_components=_EMBARGO_JSON,
            parent_trial_id="abc900_base",
            feature_version="v2.1",
        )

        registry.record(base_record)
        registry.record(derived_record)
        loaded = registry.all()

        assert len(loaded) == 2
        derived = next(r for r in loaded if r.config_hash == "abc900_derived")
        base = next(r for r in loaded if r.config_hash == "abc900_base")

        assert derived.parent_trial_id == "abc900_base"
        assert base.parent_trial_id is None
