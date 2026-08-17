from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.data.panel import Panel
from nifty_quant.strategy.base import PortfolioState, TargetPortfolio
from nifty_quant.strategy.plugins.carver_trend import CarverTrendParams, CarverTrendStrategy
from nifty_quant.strategy.registry import build, get


@dataclass
class _PanelStub:
    symbols: tuple[str, ...]
    day_offsets: np.ndarray
    ts: np.ndarray
    arrays: dict[str, np.ndarray]
    minute_array: np.ndarray

    def field(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def minute_of_day(self) -> np.ndarray:
        return self.minute_array


@dataclass
class _ViewStub:
    ts: int
    session_date: datetime.date
    symbols: tuple[str, ...]
    tradable: np.ndarray
    close: np.ndarray

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        if field == "close" and offset == 0:
            return self.close
        raise NotImplementedError("not needed by on_decision in these tests")


def _make_ts(n_rows: int) -> np.ndarray:
    base = pd.Timestamp("2024-01-01 09:15:00", tz="Asia/Kolkata")
    return (base.timestamp() + np.arange(n_rows, dtype=np.float64) * 60.0).astype(np.int64)


def _make_panel(
    close: np.ndarray,
    day_offsets: np.ndarray,
    symbols: tuple[str, ...] | None = None,
) -> _PanelStub:
    close = np.array(close, dtype=np.float32, copy=True)
    n_rows, n_symbols = close.shape

    if symbols is None:
        symbols = tuple(f"S{i}" for i in range(n_symbols))

    arrays = {
        "close": close,
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "volume": np.ones((n_rows, n_symbols), dtype=np.float32),
    }

    return _PanelStub(
        symbols=symbols,
        day_offsets=np.asarray(day_offsets, dtype=np.int64),
        ts=_make_ts(n_rows),
        arrays=arrays,
        minute_array=np.arange(n_rows, dtype=np.int32),
    )


def _flat_state(n_symbols: int, ts: int) -> PortfolioState:
    return PortfolioState(
        shares=np.zeros(n_symbols, dtype=np.float64),
        cash=0.0,
        equity=1_000_000.0,
        ts=ts,
    )


def _view(
    ts: int,
    symbols: tuple[str, ...],
    tradable: np.ndarray,
    close: np.ndarray,
) -> _ViewStub:
    return _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 1),
        symbols=symbols,
        tradable=np.asarray(tradable, dtype=bool),
        close=np.asarray(close, dtype=np.float64),
    )


def _random_close(
    n_rows: int,
    n_symbols: int,
    seed: int,
    sigma: float = 0.002,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=(n_rows - 1, n_symbols))
    log_path = 4.60517 + np.cumsum(
        np.vstack((np.zeros((1, n_symbols)), steps)),
        axis=0,
    )
    return np.exp(log_path).astype(np.float32)


def test_registered() -> None:
    assert get("carver_trend") is CarverTrendStrategy

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "strategies" / "carver_trend.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text())
    built = build(cfg)
    assert built.name == "carver_trend"


def test_precompute_2d_and_on_decision_1d() -> None:
    n_rows = 300
    n_symbols = 3
    symbols = ("A", "B", "C")

    close = _random_close(n_rows, n_symbols, seed=7)
    panel = _make_panel(close, np.array([0, n_rows], dtype=np.int64), symbols=symbols)

    strategy = CarverTrendStrategy(params=CarverTrendParams())
    signals = strategy.precompute(panel)

    for arr in signals.values():
        assert arr.ndim == 2
        assert arr.shape == (n_rows, n_symbols)

    row_index = 250
    sliced = {key: value[row_index] for key, value in signals.items()}

    for arr in sliced.values():
        assert arr.ndim == 1
        assert arr.shape == (n_symbols,)

    ts = int(panel.ts[row_index])
    view = _view(
        ts,
        symbols,
        np.ones(n_symbols, dtype=bool),
        close=panel.field("close")[row_index].astype(np.float64),
    )
    target = strategy.on_decision(view, sliced, _flat_state(n_symbols, ts))

    assert isinstance(target, TargetPortfolio)
    assert target.weights.shape == (n_symbols,)


def test_no_lookahead() -> None:
    n_rows = 512
    n_symbols = 2
    symbols = ("X", "Y")
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    close_base = _random_close(n_rows, n_symbols, seed=123, sigma=0.003)
    panel_base = _make_panel(close_base, day_offsets, symbols=symbols)

    strategy = CarverTrendStrategy(params=CarverTrendParams())
    base_signals = strategy.precompute(panel_base)

    k = 256

    spike_close = close_base.copy()
    spike_close[k + 1 :] *= 1e6

    nan_close = close_base.copy()
    nan_close[k + 1 :] = np.nan

    for name, mutated_close in {"spike": spike_close, "nan": nan_close}.items():
        panel_mut = _make_panel(
            mutated_close.astype(np.float32),
            day_offsets,
            symbols=symbols,
        )
        mutated_signals = strategy.precompute(panel_mut)

        for key in base_signals:
            assert np.array_equal(
                base_signals[key][: k + 1],
                mutated_signals[key][: k + 1],
                equal_nan=True,
            ), f"key={key}, variant={name}"


def test_forecast_scalar_is_expanding_not_full_sample() -> None:
    n_rows = 3000
    half = n_rows // 2
    rng = np.random.default_rng(42)

    small_sigma = 0.001
    large_sigma = 0.008

    steps = np.empty(n_rows, dtype=np.float64)
    steps[:half] = rng.normal(0.0, small_sigma, size=half)
    steps[half:] = rng.normal(0.0, large_sigma, size=n_rows - half)

    log_close = 4.60517 + np.cumsum(steps)
    close = np.exp(log_close).astype(np.float32)

    panel = _make_panel(close.reshape(-1, 1), np.array([0, n_rows]), symbols=("SOLO",))
    strategy = CarverTrendStrategy(params=CarverTrendParams())
    signals = strategy.precompute(panel)

    scalar_0 = signals["scalar_0"]
    early = scalar_0[400, 0]
    late = scalar_0[-1, 0]

    assert np.isfinite(early)
    assert np.isfinite(late)
    assert abs(early - late) / early > 0.1
    assert np.nanstd(scalar_0) > 0


def test_fitted_scalar_targets_expected_abs_forecast() -> None:
    n_rows = 5000
    rng = np.random.default_rng(11)

    t = np.arange(n_rows, dtype=np.float64)
    drift = 0.0004 * np.sin(2 * np.pi * t / 350.0)
    steps = rng.normal(0.0, 0.004, size=n_rows) + drift

    log_close = 4.60517 + np.cumsum(steps)
    close = np.exp(log_close).astype(np.float32)

    panel = _make_panel(close.reshape(-1, 1), np.array([0, n_rows]), symbols=("PERSIST",))
    strategy = CarverTrendStrategy(params=CarverTrendParams())
    signals = strategy.precompute(panel)

    warmup = strategy.params.scalar_min_periods * 3
    scaled_0 = signals["scaled_0"][warmup:, 0]
    finite_scaled = scaled_0[np.isfinite(scaled_0)]

    assert finite_scaled.size > 0

    mean_abs_forecast = float(np.nanmean(np.abs(finite_scaled)))

    # Asymptotic expanding-mean calibration on finite, non-stationary synthetic data.
    # Carver's own live systems show abs-forecast averages only approximately track the target.
    assert strategy.params.target_abs_forecast * 0.5 <= mean_abs_forecast
    assert mean_abs_forecast <= strategy.params.target_abs_forecast * 2.0


def test_forecast_is_clipped_to_cap() -> None:
    n_rows = 1000
    t = np.arange(n_rows, dtype=np.float64)
    close = 100.0 * np.exp(0.02 * t).astype(np.float32)

    panel = _make_panel(close.reshape(-1, 1), np.array([0, n_rows]), symbols=("ROCKET",))
    strategy = CarverTrendStrategy(params=CarverTrendParams())
    signals = strategy.precompute(panel)

    net_forecast = signals["net_forecast"]
    finite_net = net_forecast[np.isfinite(net_forecast)]

    assert finite_net.size > 0
    assert np.all(np.abs(finite_net) <= 20.0 + 1e-9)


def test_uptrend_positive_downtrend_negative() -> None:
    n_rows = 1200
    t = np.arange(n_rows, dtype=np.float64)

    close_up = 100.0 * np.exp(0.0008 * t)
    close_dn = 100.0 * np.exp(-0.0008 * t)
    close = np.column_stack([close_up, close_dn]).astype(np.float32)

    symbols = ("UP", "DOWN")
    panel = _make_panel(close, np.array([0, n_rows]), symbols=symbols)
    strategy = CarverTrendStrategy(params=CarverTrendParams(rebalance_buffer=0.0))
    signals = strategy.precompute(panel)

    row_index = -1
    sliced = {key: value[row_index] for key, value in signals.items()}
    ts = int(panel.ts[row_index])
    view = _view(
        ts,
        symbols,
        np.array([True, True]),
        close=panel.field("close")[row_index].astype(np.float64),
    )

    target = strategy.on_decision(view, sliced, _flat_state(len(symbols), ts))

    assert target.weights[0] > 0
    assert target.weights[1] < 0


def test_vol_scaling_inverse() -> None:
    # The old workaround params (forecast_cap=200.0, max_weight=10.0, gross=10.0) are
    # gone: pre-Fix-1 `(net_forecast/cap)/vol_guard` had no target-vol numerator and
    # saturated to max_weight on almost any nonzero forecast, so the only way to reveal
    # the 1/vol relationship was to push the clip far away. The fixed annualized-vol
    # formula `(net_forecast/cap) * (target_vol_ann / vol_ann)` is genuinely
    # proportional, so the default cap/max_weight/gross show the same 2:1 ratio without
    # clipping.
    params = CarverTrendParams(rebalance_buffer=0.0)
    strategy = CarverTrendStrategy(params=params)

    signals = {
        "net_forecast": np.array([10.0, 10.0]),
        "vol": np.array([0.01, 0.02]),
    }

    view = _view(
        0,
        ("A", "B"),
        np.array([True, True]),
        close=np.array([100.0, 100.0]),
    )
    target = strategy.on_decision(view, signals, _flat_state(2, 0))

    assert target.weights[0] == pytest.approx(2.0 * target.weights[1], rel=1e-6)
    assert target.weights[0] > 0
    assert target.weights[1] > 0
    assert abs(target.weights[0]) < params.max_weight
    assert abs(target.weights[1]) < params.max_weight


def test_weights_respect_gross_and_max_weight() -> None:
    # The fixed annualized-vol formula produces a natural post-clip gross of ~0.50 for
    # these signals (every symbol clips to max_weight), so gross=1.0 no longer binds.
    # Tightening gross to 0.25 guarantees the rescale branch is actually exercised.
    params = CarverTrendParams(max_weight=0.10, gross=0.25, rebalance_buffer=0.0)
    strategy = CarverTrendStrategy(params=params)

    n_symbols = 5
    signals = {
        "net_forecast": np.array([15.0, -18.0, 12.0, -25.0, 20.0], dtype=np.float64),
        "vol": np.array([0.001, 0.002, 0.0015, 0.003, 0.002], dtype=np.float64),
    }

    view = _view(
        0,
        tuple(f"S{i}" for i in range(n_symbols)),
        np.ones(n_symbols, dtype=bool),
        close=np.full(n_symbols, 100.0),
    )
    target = strategy.on_decision(view, signals, _flat_state(n_symbols, 0))

    target.validate(n_symbols=n_symbols, max_gross=params.gross)
    assert np.all(np.abs(target.weights) <= params.max_weight + 1e-9)


def test_irregular_session_length() -> None:
    day_offsets = np.array([0, 60, 165], dtype=np.int64)
    n_symbols = 2

    close = _random_close(165, n_symbols, seed=2024, sigma=0.003)
    panel = _make_panel(close, day_offsets, symbols=("M", "D"))

    strategy = CarverTrendStrategy(params=CarverTrendParams(forecast_scalar=1.0))
    signals = strategy.precompute(panel)

    for arr in signals.values():
        assert arr.shape == (165, n_symbols)

    assert np.isfinite(signals["net_forecast"]).any()


def test_never_reads_0915_bar() -> None:
    n_per_session = 400
    n_symbols = 2
    day_offsets = np.array([0, n_per_session, n_per_session * 2], dtype=np.int64)
    symbols = ("P", "Q")

    close_base = _random_close(n_per_session * 2, n_symbols, seed=99, sigma=0.003)
    panel_base = _make_panel(close_base, day_offsets, symbols=symbols)

    strategy = CarverTrendStrategy(params=CarverTrendParams())
    base_signals = strategy.precompute(panel_base)

    spike_close = close_base.copy()
    spike_close[day_offsets[:-1]] *= 1e6

    nan_close = close_base.copy()
    nan_close[day_offsets[:-1]] = np.nan

    mask = np.ones(n_per_session * 2, dtype=bool)
    mask[day_offsets[:-1]] = False

    for name, mutated_close in {"spike": spike_close, "nan": nan_close}.items():
        panel_mut = _make_panel(
            mutated_close.astype(np.float32),
            day_offsets,
            symbols=symbols,
        )
        mutated_signals = strategy.precompute(panel_mut)

        for key in base_signals:
            assert np.array_equal(
                base_signals[key][mask],
                mutated_signals[key][mask],
                equal_nan=True,
            ), f"key={key}, variant={name}"


def test_weights_not_universally_saturated() -> None:
    n_rows = 1500
    n_symbols = 4
    symbols = tuple(f"S{i}" for i in range(n_symbols))

    # sigma=0.003 keeps typical net_forecast/vol weights comfortably inside max_weight;
    # with the fixed formula a smaller sigma (e.g. 0.0015) would clip most observations
    # and reproduce the very bang-bang behaviour this test guards against.
    close = _random_close(n_rows, n_symbols, seed=31, sigma=0.003)
    panel = _make_panel(close, np.array([0, n_rows], dtype=np.int64), symbols=symbols)

    params = CarverTrendParams(rebalance_buffer=0.0)
    strategy = CarverTrendStrategy(params=params)
    signals = strategy.precompute(panel)

    all_weights: list[float] = []
    for row in range(300, n_rows, 5):
        sliced = {key: value[row] for key, value in signals.items()}
        ts = int(panel.ts[row])
        view = _view(
            ts,
            symbols,
            np.ones(n_symbols, dtype=bool),
            close=panel.field("close")[row].astype(np.float64),
        )
        target = strategy.on_decision(view, sliced, _flat_state(n_symbols, ts))
        all_weights.extend(target.weights)

    weights = np.array(all_weights, dtype=np.float64)
    nonzero = weights[np.abs(weights) > 0]
    assert nonzero.size > 0

    saturated = np.isclose(np.abs(nonzero), params.max_weight)
    # Most bar/symbol observations on ordinary non-extreme-trend data should show
    # genuine graduated sizing, with only occasional high-conviction low-vol moments
    # clipping. A saturated fraction below 0.5 is a conservative floor: if every
    # nonzero weight were saturated, that would be exactly the pre-Fix-1 bang-bang bug.
    assert not np.all(saturated)
    assert float(np.mean(saturated)) < 0.5


def test_higher_vol_gives_smaller_weight_when_unsaturated() -> None:
    n_rows = 1800
    drift = 0.012

    rng = np.random.default_rng(11)
    log_close_low_vol = (
        4.60517
        + drift * np.arange(n_rows, dtype=np.float64)
        + np.cumsum(rng.normal(0.0, 0.006, n_rows))
    )
    log_close_high_vol = (
        4.60517
        + drift * np.arange(n_rows, dtype=np.float64)
        + np.cumsum(rng.normal(0.0, 0.030, n_rows))
    )
    close = np.column_stack([np.exp(log_close_low_vol), np.exp(log_close_high_vol)]).astype(
        np.float32
    )

    # Both symbols share the exact same deterministic positive drift; HIGH_VOL adds 5x
    # the per-bar noise of LOW_VOL, so its realized vol is ~5x larger. Because the
    # fixed formula divides by annualized vol, the low-vol symbol must receive the
    # strictly larger weight, and neither weight may be clipped for the comparison to
    # be informative.
    symbols = ("LOW_VOL", "HIGH_VOL")
    panel = _make_panel(close, np.array([0, n_rows], dtype=np.int64), symbols=symbols)

    params = CarverTrendParams(rebalance_buffer=0.0)
    strategy = CarverTrendStrategy(params=params)
    signals = strategy.precompute(panel)

    row = n_rows - 1
    sliced = {key: value[row] for key, value in signals.items()}
    ts = int(panel.ts[row])
    view = _view(
        ts,
        symbols,
        np.ones(2, dtype=bool),
        close=panel.field("close")[row].astype(np.float64),
    )
    target = strategy.on_decision(view, sliced, _flat_state(2, ts))

    assert target.weights[0] > 0
    assert target.weights[1] > 0
    assert target.weights[0] > target.weights[1]
    assert abs(target.weights[0]) < params.max_weight - 1e-9
    assert abs(target.weights[1]) < params.max_weight - 1e-9


def test_rebalance_buffer_reduces_trade_count() -> None:
    n_sessions = 9
    bars_per_session = 200
    n_symbols = 5
    n_rows = n_sessions * bars_per_session

    symbols = tuple(f"S{i}" for i in range(n_symbols))
    close = _random_close(n_rows, n_symbols, seed=44, sigma=0.0015)

    high = (close * 1.001).astype(np.float32)
    low = (close * 0.999).astype(np.float32)
    volume = np.full((n_rows, n_symbols), 1_000_000.0, dtype=np.float32)

    arrays = {
        "open": close.astype(np.float32),
        "high": high,
        "low": low,
        "close": close.astype(np.float32),
        "volume": volume,
    }

    day_offsets = np.arange(0, n_rows + 1, bars_per_session, dtype=np.int32)
    start_date = datetime.date(2024, 1, 1)
    dates = np.array(
        [start_date + datetime.timedelta(days=i) for i in range(n_sessions)],
        dtype=object,
    )

    ts_parts: list[np.ndarray] = []
    for session_index in range(n_sessions):
        session_date = start_date + datetime.timedelta(days=session_index)
        session_start = pd.Timestamp(
            f"{session_date.isoformat()} 09:15:00", tz="Asia/Kolkata"
        )
        ts_parts.append(
            (
                session_start.timestamp()
                + np.arange(bars_per_session, dtype=np.float64) * 60.0
            ).astype(np.int64)
        )
    ts = np.concatenate(ts_parts)

    panel = Panel(
        symbols=symbols,
        dates=dates,
        ts=ts,
        day_offsets=day_offsets,
        fields=arrays,
    )

    unbuffered = CarverTrendStrategy(params=CarverTrendParams(rebalance_buffer=0.0))
    buffered = CarverTrendStrategy(params=CarverTrendParams(rebalance_buffer=0.01))

    result_unbuffered = run_backtest(unbuffered, panel, BacktestConfig())
    result_buffered = run_backtest(buffered, panel, BacktestConfig())

    assert result_unbuffered.n_trades > 0
    # With buffer=0 the strategy re-targets on most bars, so a modest default buffer
    # should cut a clear fraction of churn. 0.7x is a conservative floor that fails
    # loudly if the buffer branch stops doing anything, without requiring a particular
    # buffer aggressiveness.
    assert result_buffered.n_trades <= 0.7 * result_unbuffered.n_trades


def test_buffer_zero_reproduces_unbuffered_behaviour() -> None:
    symbols = ("A", "B")
    close = np.array([100.0, 250.0], dtype=np.float64)

    signals = {
        "net_forecast": np.array([12.0, -8.0]),
        "vol": np.array([0.01, 0.02]),
    }

    view = _view(0, symbols, np.ones(2, dtype=bool), close=close)

    # Learn the raw unbuffered target from a flat starting state.
    raw_params = CarverTrendParams(rebalance_buffer=0.0)
    raw_strategy = CarverTrendStrategy(params=raw_params)
    raw_target = raw_strategy.on_decision(view, signals, _flat_state(2, 0))
    raw_weights = raw_target.weights

    # Build a deliberately non-flat state whose current weights sit close to (but not
    # exactly at) the raw target: raw_weights - 0.3 * max_weight * sign(raw_weights).
    step = 0.3 * raw_params.max_weight * np.sign(raw_weights)
    current_weight = raw_weights - step
    shares = current_weight * 1_000_000.0 / close
    state = PortfolioState(shares=shares, cash=0.0, equity=1_000_000.0, ts=0)

    # Buffered: every symbol is within the 0.05 buffer of the raw target, so the
    # whole-portfolio hold policy returns None instead of a TargetPortfolio.
    buffered_params = CarverTrendParams(rebalance_buffer=0.05)
    buffered_strategy = CarverTrendStrategy(params=buffered_params)
    buffered_target = buffered_strategy.on_decision(view, signals, state)

    assert buffered_target is None

    # A narrower buffer than the 0.03 gap triggers a full-portfolio rebalance.
    # The whole-portfolio policy emits the complete fresh target whenever any symbol
    # is outside its buffer.
    narrow_buffer_params = CarverTrendParams(rebalance_buffer=0.01)
    narrow_buffer_strategy = CarverTrendStrategy(params=narrow_buffer_params)
    narrow_buffer_target = narrow_buffer_strategy.on_decision(view, signals, state)

    assert narrow_buffer_target is not None
    assert np.allclose(narrow_buffer_target.weights, raw_weights, atol=1e-9)

    # Escape hatch: buffer <= 0 reproduces the raw target exactly, regardless of the
    # nonzero current position.
    zero_buffer_params = CarverTrendParams(rebalance_buffer=0.0)
    zero_buffer_strategy = CarverTrendStrategy(params=zero_buffer_params)
    zero_buffer_target = zero_buffer_strategy.on_decision(view, signals, state)

    assert zero_buffer_target is not None
    assert np.allclose(zero_buffer_target.weights, raw_weights, atol=1e-12)
