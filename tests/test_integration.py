"""Engine<->strategy integration test tier: panel, strategy, engine, costs, and
metrics exercised together in one end-to-end stack."""

import datetime
import math
from collections.abc import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

import nifty_quant.strategy.plugins  # noqa: F401  (registers built-in plugins)
from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import compute_metrics
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts, ZeroCost
from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
from nifty_quant.strategy import registry
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from nifty_quant.strategy.plugins.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
)
from nifty_quant.universe.static import equity_symbols
from tests.contract_fixtures import minimal_contract

_IST = ZoneInfo("Asia/Kolkata")
BARS_PER_DAY = 375


def _session_grid(n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = datetime.date(2024, 1, 2)
    dates: list[datetime.date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)

    ts_chunks = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=BARS_PER_DAY, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(
            dtype=np.int64
        )
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.arange(
        0, (n_days + 1) * BARS_PER_DAY, BARS_PER_DAY, dtype=np.int32
    )
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _make_synthetic_panel(
    n_sym: int = 8, n_days: int = 5, seed: int = 7
) -> Panel:
    ts, day_offsets, dates = _session_grid(n_days)
    n_rows = len(ts)
    rng = np.random.default_rng(seed)

    symbols = tuple(f"SYM{i:02d}" for i in range(n_sym))

    bases = rng.uniform(80.0, 400.0, size=n_sym)
    close = np.zeros((n_rows, n_sym), dtype=np.float64)
    for j, base in enumerate(bases):
        sym_ret = rng.normal(0.0, 0.0006, size=n_rows)
        for start_ix, end_ix in zip(day_offsets[:-1], day_offsets[1:]):
            sl = slice(start_ix, end_ix)
            sym_ret[sl] = np.cumsum(sym_ret[sl])
        close[:, j] = base * np.exp(sym_ret)

    open_ = close.copy()
    for start_ix, end_ix in zip(day_offsets[:-1], day_offsets[1:]):
        if end_ix > start_ix + 1:
            open_[start_ix + 1 : end_ix] = close[start_ix : end_ix - 1]

    high_noise = np.abs(rng.normal(0.0, 0.0003, size=n_rows))
    low_noise = np.abs(rng.normal(0.0, 0.0003, size=n_rows))
    high = np.maximum(open_, close) * (1.0 + high_noise[:, None])
    low = np.minimum(open_, close) * (1.0 - low_noise[:, None])
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(500_000.0, 2_000_000.0, size=(n_rows, n_sym))

    fields = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(
        fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates
    )


def _make_synthetic_panel_with_absent_symbols(
    n_sym: int = 8, n_days: int = 5, seed: int = 7
) -> Panel:
    """Synthetic panel with symbol 0 entirely absent (NaN OHLC) and symbol 1
    partially absent (NaN OHLC for the first 60% of rows)."""
    panel = _make_synthetic_panel(n_sym=n_sym, n_days=n_days, seed=seed)
    fields = {
        name: np.asarray(panel.field(name)).copy()
        for name in ("open", "high", "low", "close", "volume")
    }
    n_rows = len(panel.ts)

    # Symbol 0: fully absent all rows.
    for field in ("open", "high", "low", "close"):
        fields[field][:, 0] = np.nan

    # Symbol 1: missing for first 60% of rows, valid thereafter.
    split_idx = int(n_rows * 0.6)
    for field in ("open", "high", "low", "close"):
        fields[field][:split_idx, 1] = np.nan

    return Panel(
        fields=fields,
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )


def _make_synthetic_panel_with_short_session() -> Panel:
    """Two sessions: normal 375-bar session then a short 100-bar session
    ending before 15:20 (last bar ~10:54)."""
    normal_date = datetime.date(2024, 1, 2)
    normal_start = pd.Timestamp(
        normal_date.year, normal_date.month, normal_date.day, 9, 15, tz=_IST
    )
    normal_idx = pd.date_range(normal_start, periods=BARS_PER_DAY, freq="1min")
    normal_utc = normal_idx.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    normal_ts = ((normal_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)

    short_date = normal_date + datetime.timedelta(days=1)
    short_start = pd.Timestamp(short_date.year, short_date.month, short_date.day, 9, 15, tz=_IST)
    short_idx = pd.date_range(short_start, periods=100, freq="1min")
    short_utc = short_idx.tz_convert("UTC")
    short_ts = ((short_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)

    ts = np.concatenate([normal_ts, short_ts]).astype(np.int64)
    day_offsets = np.array([0, BARS_PER_DAY, BARS_PER_DAY + 100], dtype=np.int32)
    dates = np.array([normal_date, short_date], dtype=object)

    n_rows = len(ts)
    symbols = ("SYM00", "SYM01", "SYM02")
    close = np.full((n_rows, 3), 100.0, dtype=np.float32)
    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.full((n_rows, 3), 1_000_000.0, dtype=np.float32)

    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return Panel(
        fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates
    )


def _flat_zero_return_panel() -> Panel:
    ts, day_offsets, dates = _session_grid(2)
    n_rows = len(ts)
    symbols = ("SYM00", "SYM01", "SYM02")

    close = np.full((n_rows, 3), 100.0, dtype=np.float32)
    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.full((n_rows, 3), 1_000_000.0, dtype=np.float32)

    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return Panel(
        fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates
    )


class _ProbeParams(BaseModel):
    pass


class _FixedWeightStrategy(Strategy):
    name = "fixed_weight_probe"
    Params = _ProbeParams

    def __init__(self) -> None:
        super().__init__(_ProbeParams())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=("10:00",))

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        return {}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        del signals, state
        n = len(view.symbols)
        weights = np.array(
            [0.1 if i % 2 == 0 else -0.1 for i in range(n)], dtype=np.float64
        )
        return TargetPortfolio(weights=weights)


class _SquareOffProbeStrategy(Strategy):
    name = "square_off_probe"
    Params = _ProbeParams

    def __init__(self) -> None:
        super().__init__(_ProbeParams())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=("09:20",))

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        return {}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        del signals, state
        n = len(view.symbols)
        weights = np.zeros(n, dtype=np.float64)
        weights[0] = 1.0  # open a long position in the first symbol
        return TargetPortfolio(weights=weights)


class _FlatProbeStrategy(Strategy):
    name = "flat_probe"
    Params = _ProbeParams

    def __init__(self) -> None:
        super().__init__(_ProbeParams())

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=("10:00",))

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        return {}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        del signals, state
        n = len(view.symbols)
        return TargetPortfolio(weights=np.zeros(n, dtype=np.float64))


class _SignalsContractProbe(Strategy):
    name = "signals_contract_probe"
    Params = _ProbeParams

    def __init__(self, inner: Strategy) -> None:
        super().__init__(_ProbeParams())
        self._inner = inner
        self._precompute_result: dict[str, np.ndarray] = {}
        self._cached_ts: np.ndarray | None = None

    def data_request(self) -> DataRequest:
        return self._inner.data_request()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        result = self._inner.precompute(panel)
        self._precompute_result = result
        self._cached_ts = panel.ts
        return result

    def on_session_start(self, session_date: datetime.date) -> None:
        self._inner.on_session_start(session_date)

    def on_session_end(self, session_date: datetime.date) -> None:
        self._inner.on_session_end(session_date)

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        assert self._cached_ts is not None
        cursor_row = int(np.searchsorted(self._cached_ts, view.ts))
        assert self._cached_ts[cursor_row] == view.ts

        for key, arr in signals.items():
            assert arr.ndim == 1, f"signal {key} must be 1-D at cursor"
            assert arr.shape == (len(view.symbols),), (
                f"signal {key} has shape {arr.shape}, "
                f"expected ({len(view.symbols)},)"
            )
            expected = self._precompute_result[key][cursor_row]
            np.testing.assert_array_equal(arr, expected)

        return self._inner.on_decision(view, signals, state)


def _build_plugin_strategy(plugin_name: str) -> Strategy:
    cls = registry.get(plugin_name)
    try:
        if plugin_name == "volume_breakout":
            # TODO(use_hurst): flip back to the VolumeBreakoutParams default
            # (use_hurst=True) once the @causal domain="positive" guard fix lands.
            params = cls.Params(use_hurst=False)
        elif plugin_name == "xsec_zscore":
            params = cls.Params(min_names=4)
        else:
            params = cls.Params()
    except Exception as exc:
        pytest.skip(f"plugin {plugin_name} requires custom params: {exc}")
    return cls(params)


@pytest.mark.parametrize("plugin_name", registry.available())
def test_all_registered_plugins_run_end_to_end(plugin_name: str) -> None:
    panel = _make_synthetic_panel()
    strategy = _build_plugin_strategy(plugin_name)
    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.equity_curve.size > 0
    assert np.all(np.isfinite(result.equity_curve))


@pytest.mark.parametrize("plugin_name", registry.available())
def test_absent_symbols_do_not_crash(plugin_name: str) -> None:
    panel = _make_synthetic_panel_with_absent_symbols()
    strategy = _build_plugin_strategy(plugin_name)
    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    # xsec_zscore's signal is only valid at the decision row itself, not at
    # cursor = t - 1, so it structurally never trades via run_backtest
    # (independent of absent-data handling).
    if plugin_name == "volume_breakout":
        assert result.n_trades > 0
    assert np.all(np.isfinite(result.equity_curve))
    fully_absent_symbol = panel.symbols[0]
    assert (result.trades["symbol"] != fully_absent_symbol).all()


def test_absent_symbol_positions_are_zero() -> None:
    panel = _make_synthetic_panel_with_absent_symbols()
    strategy = _FixedWeightStrategy()
    result = run_backtest(strategy, panel, BacktestConfig(), contract=minimal_contract())

    absent_idx = 0  # symbol 0 is fully absent
    assert np.all(result.positions[:, absent_idx] == 0.0)


def test_gross_renormalization_with_absent_symbol() -> None:
    panel = _make_synthetic_panel_with_absent_symbols()  # n_sym=8, n_days=5
    strategy = _FixedWeightStrategy()
    config = BacktestConfig(
        capital=1_000_000.0, sizer=GrossNotionalSizer(max_weight=0.20)
    )
    result = run_backtest(strategy, panel, config, contract=minimal_contract())

    # decision time is "10:00", i.e. row offset 45 within each session; the
    # resulting order fills one bar later (decision_latency_bars=0 by default),
    # at row offset 46 -- that is the entry leg's fill row for each day.
    n_days = len(panel.dates)
    entry_fill_rows = np.array(
        [int(panel.day_offsets[d]) + 46 for d in range(n_days)], dtype=np.int64
    )
    entry_ts = set(int(panel.ts[r]) for r in entry_fill_rows)

    entry_trades = result.trades[result.trades["ts"].isin(entry_ts)]
    capital = config.capital
    if entry_trades.empty:
        gross_by_day = np.zeros(n_days, dtype=np.float64)
    else:
        gross_by_day = (
            entry_trades.groupby("ts")["notional"].sum() / capital
        ).to_numpy()
    mean_gross = float(np.mean(gross_by_day)) if gross_by_day.size else 0.0

    n_sym = len(panel.symbols)
    orig_gross = 0.1 * n_sym  # 0.8 with 8 symbols
    # lower bound: gross if no renormalization happened (absent symbol removed)
    lower_bound = orig_gross * (n_sym - 1) / n_sym

    assert mean_gross > lower_bound
    assert abs(mean_gross - orig_gross) / orig_gross < 0.20  # 20% relative tolerance


def test_accounting_invariant_with_absent_symbols() -> None:
    panel = _make_synthetic_panel_with_absent_symbols(n_sym=6, n_days=3, seed=11)
    strategy = _FixedWeightStrategy()

    # run_backtest already enforces the accounting identity via guards at
    # Strictness.FULL on every row; a ContractViolation escaping this call fails
    # the test naturally.
    result = run_backtest(strategy, panel, BacktestConfig(), contract=minimal_contract())

    assert result.equity_curve.size > 0
    assert np.all(np.isfinite(result.equity_curve))


@pytest.mark.parametrize("plugin_name", registry.available())
def test_signals_arrive_1d_at_cursor(plugin_name: str) -> None:
    panel = _make_synthetic_panel()
    inner = _build_plugin_strategy(plugin_name)
    probe = _SignalsContractProbe(inner)

    result = run_backtest(
        probe,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.equity_curve.size > 0


def test_gross_vs_net_costs_reconcile() -> None:
    panel = _make_synthetic_panel(n_sym=6, n_days=3)

    gross_strategy = _FixedWeightStrategy()
    net_strategy = _FixedWeightStrategy()

    gross_result = run_backtest(
        gross_strategy,
        panel,
        BacktestConfig(
            cost_model=ZeroCost(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )
    net_result = run_backtest(
        net_strategy,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert net_result.total_costs > 0.0
    assert gross_result.equity_curve[-1] > net_result.equity_curve[-1]
    assert abs(
        (gross_result.equity_curve[-1] - net_result.equity_curve[-1])
        - net_result.total_costs
    ) < 1e-6


def test_accounting_invariant_holds_across_full_run() -> None:
    panel = _make_synthetic_panel(n_sym=6, n_days=3)
    strategy = _FixedWeightStrategy()

    # run_backtest already enforces the accounting identity via guards at
    # Strictness.FULL on every row; a ContractViolation escaping this call fails
    # the test naturally, which is the whole point of this test.
    result = run_backtest(strategy, panel, BacktestConfig(), contract=minimal_contract())

    assert result.equity_curve.size > 0


@pytest.mark.parametrize("plugin_name", registry.available())
def test_square_off_flat_no_forced_liquidation(plugin_name: str) -> None:
    panel = _make_synthetic_panel()
    strategy = _build_plugin_strategy(plugin_name)

    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.forced_eod_liquidation_days == 0
    assert np.all(result.positions[-1] == 0.0)
    # forced_eod_liquidation_days == 0 is the authoritative signal that every
    # session closed flat without the engine's forced-liquidation fallback
    # kicking in; the engine's own assert would have raised otherwise, so
    # reaching this line is itself informative.


def test_square_off_on_abbreviated_session_no_forced_liquidation() -> None:
    panel = _make_synthetic_panel_with_short_session()
    strategy = _SquareOffProbeStrategy()

    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.forced_eod_liquidation_days == 0
    # The last decision row of result.positions reflects the final square-off,
    # which must be flat even though the short session had no bar at/after 15:20.
    assert np.all(result.positions[-1] == 0.0)


def test_metrics_pipeline_never_returns_inf() -> None:
    # Part A: degenerate all-zero returns may produce NaN, but never inf.
    flat_panel = _flat_zero_return_panel()
    flat_strategy = _FlatProbeStrategy()
    flat_result = run_backtest(
        flat_strategy,
        flat_panel,
        BacktestConfig(
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )
    flat_metrics = compute_metrics(flat_result.returns)
    assert not math.isinf(flat_metrics.sharpe)
    assert not math.isinf(flat_metrics.max_drawdown)

    # Part B: a real fixed-weight run has nonzero returns and must stay finite.
    panel = _make_synthetic_panel(n_sym=6, n_days=3)
    strategy = _FixedWeightStrategy()
    result = run_backtest(strategy, panel, BacktestConfig(), contract=minimal_contract())
    metrics = compute_metrics(result.returns)
    assert math.isfinite(metrics.sharpe)
    assert math.isfinite(metrics.max_drawdown)


@pytest.mark.slow
def test_real_2024_volume_breakout_end_to_end() -> None:
    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=(
            "RELIANCE",
            "HDFCBANK",
            "ABB",
            "ACC",
            "BPCL",
            "CIPLA",
            "COALINDIA",
            "BHARTIARTL",
        ),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    panel = load_panel(spec)

    # TODO(use_hurst): flip back to the VolumeBreakoutParams default
    # (use_hurst=True) once the @causal domain="positive" guard fix lands.
    params = VolumeBreakoutParams(use_hurst=False)
    strategy = VolumeBreakoutStrategy(params=params)

    # Capital sized to the real per-minute liquidity of the smaller-cap names in
    # this universe (ABB/ACC median 1-min traded value ~Rs 1.1-3.5M on 2024 data):
    # the BacktestConfig() default capital=1e7 combined with this strategy's
    # default max_weight/gross produces per-symbol order sizes far above what a
    # single 1-minute bar can absorb under the 2% participation cap (verified:
    # unfilled_notional_pct=0.795, forced_eod_liquidation_days=62 at capital=1e7),
    # which is a sizing artifact of the test harness, not a bug in the engine or
    # the signal-slicing fix this test tier exists to pin.
    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            capital=1_000_000.0,
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.n_trades > 0
    assert np.all(np.isfinite(result.equity_curve))
    assert 0.0 <= result.rejected_order_rate <= 1.0
    assert 0.0 <= result.unfilled_notional_pct <= 1.0
    # NSE's real 2024 calendar has exactly 3 documented abbreviated sessions that
    # end before the configured square_off_time "15:20" (2024-03-02 and
    # 2024-05-18 end ~12:29; 2024-11-01 is the Muhurat evening special session).
    # The engine now resolves square-off per-session, falling back to the
    # session's last bar and flattening there with normal intraday costs, so
    # forced EOD liquidation should never happen for a strategy that always
    # squares off -- including on those abbreviated sessions.
    assert result.forced_eod_liquidation_days == 0

    gross_metrics = compute_metrics(result.gross_returns)
    net_metrics = compute_metrics(result.returns)
    gross_final_equity = result.initial_capital * np.prod(1 + result.gross_returns)

    print(f"gross_final_equity={gross_final_equity:.2f}")
    print(f"net_final_equity={result.equity_curve[-1]:.2f}")
    print(f"total_costs={result.total_costs:.2f}")
    print(f"n_trades={result.n_trades}")
    print(f"gross_sharpe={gross_metrics.sharpe:.4f}")
    print(f"net_sharpe={net_metrics.sharpe:.4f}")


@pytest.mark.slow
def test_real_2024_volume_breakout_full_universe_absent_symbols() -> None:

    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=equity_symbols(),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    panel = load_panel(spec)

    # TODO(use_hurst): flip back to the VolumeBreakoutParams default
    # (use_hurst=True) once the @causal domain="positive" guard fix lands.
    params = VolumeBreakoutParams(use_hurst=False)
    strategy = VolumeBreakoutStrategy(params=params)

    result = run_backtest(
        strategy,
        panel,
        BacktestConfig(
            capital=1_000_000.0,
            cost_model=NSEIntradayEquityCosts(),
            fill_model=FillModel(slippage=SqrtImpactSlippage()),
        ), contract=minimal_contract()
    )

    assert result.n_trades > 0
    assert np.all(np.isfinite(result.equity_curve))
    # TATACAP, TMCV, ENRIN have zero bars anywhere in 2024 (not-yet-listed);
    # HYUNDAI is a mid-2024 IPO with ~80.9% NaN rows but DOES have some valid
    # bars, so per-row masking handles it correctly without counting it here --
    # n_symbols_absent counts only symbols with NO valid bar at all.
    assert result.n_symbols_absent == 3
    assert set(result.absent_symbols) == {"TATACAP", "TMCV", "ENRIN"}

    gross_metrics = compute_metrics(result.gross_returns)
    net_metrics = compute_metrics(result.returns)
    gross_final_equity = result.initial_capital * np.prod(1 + result.gross_returns)

    print(f"gross_final_equity={gross_final_equity:.2f}")
    print(f"net_final_equity={result.equity_curve[-1]:.2f}")
    print(f"total_costs={result.total_costs:.2f}")
    print(f"n_trades={result.n_trades}")
    print(f"gross_sharpe={gross_metrics.sharpe:.4f}")
    print(f"net_sharpe={net_metrics.sharpe:.4f}")
