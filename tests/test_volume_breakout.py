# tests/test_volume_breakout.py
from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from nifty_quant import guards
from nifty_quant.strategy.base import PortfolioState
from nifty_quant.strategy.plugins.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
)
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

    def last(self, field: str, offset: int = 0, *, ffill: bool = False) -> np.ndarray:
        raise NotImplementedError("not needed by on_decision in these tests")


def _make_panel(
    n_rows: int = 450,
    n_syms: int = 3,
    seed: int = 0,
    volume_spikes: bool = False,
) -> _PanelStub:
    """Create a single-session synthetic panel with positive prices."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.0025, size=(n_rows, n_syms))
    close = 1000.0 * np.exp(np.cumsum(rets, axis=0))
    high = close + rng.uniform(0.5, 3.0, size=(n_rows, n_syms))
    low = close - rng.uniform(0.5, 3.0, size=(n_rows, n_syms))
    open_ = close - rng.normal(0.0, 0.25, size=(n_rows, n_syms))
    low = np.maximum(low, 0.01)
    open_ = np.maximum(open_, 0.01)

    volume = rng.lognormal(mean=8.0, sigma=1.0, size=(n_rows, n_syms))
    if volume_spikes:
        spike_idx = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
        volume[spike_idx, :] *= 20.0

    day_offsets = np.array([0, n_rows], dtype=np.int64)
    minute = (np.arange(n_rows) % 375).astype(np.int64)
    ts = (
        pd.Timestamp("2024-01-01 09:15:00", tz="Asia/Kolkata").timestamp()
        + np.arange(n_rows) * 60
    ).astype(np.int64)

    arrays = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return _PanelStub(
        symbols=tuple(f"S{i}" for i in range(n_syms)),
        day_offsets=day_offsets,
        ts=ts,
        arrays=arrays,
        minute_array=minute,
    )


def _copy_panel_with_future_noise(
    panel: _PanelStub, k: int, seed: int = 123
) -> _PanelStub:
    """Return a panel copy with all rows after ``k`` replaced by random values.

    The future OHLC path is generated as a realistic, strictly positive
    multiplicative continuation of the real close at ``k`` so that the scramble
    is representative of plausible future prices and always preserves the
    OHLC relation low <= min(open, close) <= max(open, close) <= high.
    """
    rng = np.random.default_rng(seed)
    arrays = {name: arr.copy() for name, arr in panel.arrays.items()}
    n_rows, n_syms = panel.arrays["close"].shape
    n_future = n_rows - k - 1

    if n_future > 0:
        close_k = arrays["close"][k]

        close_future = close_k * np.exp(
            np.cumsum(rng.normal(0.0, 0.01, size=(n_future, n_syms)), axis=0)
        )
        open_future = np.empty_like(close_future)
        open_future[0] = close_k
        open_future[1:] = close_future[:-1]

        oc_min = np.minimum(open_future, close_future)
        oc_max = np.maximum(open_future, close_future)
        high_future = oc_max + rng.uniform(0.1, 1.0, size=(n_future, n_syms))
        low_future = np.maximum(
            oc_min - rng.uniform(0.1, 1.0, size=(n_future, n_syms)),
            0.01,
        )

        arrays["open"][k + 1 :] = open_future.astype(arrays["open"].dtype, copy=False)
        arrays["high"][k + 1 :] = high_future.astype(arrays["high"].dtype, copy=False)
        arrays["low"][k + 1 :] = low_future.astype(arrays["low"].dtype, copy=False)
        arrays["close"][k + 1 :] = close_future.astype(arrays["close"].dtype, copy=False)
        arrays["volume"][k + 1 :] = rng.lognormal(
            mean=8.0, sigma=1.0, size=(n_future, n_syms)
        ).astype(arrays["volume"].dtype, copy=False)

    return _PanelStub(
        symbols=panel.symbols,
        day_offsets=panel.day_offsets.copy(),
        ts=panel.ts.copy(),
        arrays=arrays,
        minute_array=panel.minute_array.copy(),
    )


def _make_view(n_syms: int, tradable: np.ndarray | None = None) -> _ViewStub:
    ts = int(pd.Timestamp("2024-01-01 10:00:00", tz="Asia/Kolkata").timestamp())
    return _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 1),
        symbols=tuple(f"S{i}" for i in range(n_syms)),
        tradable=np.ones(n_syms, dtype=bool)
        if tradable is None
        else tradable.astype(bool),
    )


def test_precompute_is_causal() -> None:
    # Use the strategy's default configuration (use_hurst=True, deseasonalize=True)
    # so the test exercises the real default path, including Hurst causality.
    params = VolumeBreakoutParams()
    strategy = VolumeBreakoutStrategy(params)
    panel = _make_panel(n_rows=450, n_syms=3, seed=0)
    k = panel.day_offsets[-1] // 2
    noisy = _copy_panel_with_future_noise(panel, k)

    with guards.strictness(guards.Strictness.FULL):
        base = strategy.precompute(panel)
        perturbed = strategy.precompute(noisy)

    for key in ("long", "short", "sigma", "vol_z", "hurst"):
        assert np.array_equal(
            base[key][: k + 1],
            perturbed[key][: k + 1],
            equal_nan=True,
        )


def test_zero_trade_regression_pinned() -> None:
    params = VolumeBreakoutParams(use_hurst=False, deseasonalize=False)
    strategy = VolumeBreakoutStrategy(params)
    panel = _make_panel(n_rows=450, n_syms=3, seed=42, volume_spikes=True)
    signals = strategy.precompute(panel)
    assert int(signals["long"].sum() + signals["short"].sum()) > 0

    close = panel.field("close")
    high = panel.field("high")
    rolling_high = (
        pd.DataFrame(high)
        .rolling(params.breakout_window)
        .max()
        .to_numpy()
    )
    unshifted = close > rolling_high
    assert int(unshifted.sum()) == 0


def test_hurst_window_guard_warns() -> None:
    n_rows, n_syms = 200, 3
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0, 0.0005, size=(n_rows, n_syms))
    close = 1000.0 * np.exp(np.cumsum(rets, axis=0))
    high = close + rng.uniform(0.1, 0.5, size=(n_rows, n_syms))
    low = close - rng.uniform(0.1, 0.5, size=(n_rows, n_syms))
    open_ = close.copy()
    low = np.maximum(low, 0.01)
    volume = rng.lognormal(mean=8.0, sigma=1.0, size=(n_rows, n_syms))
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    minute = (np.arange(n_rows) % 375).astype(np.int64)
    ts = (
        pd.Timestamp("2024-01-01 09:15:00", tz="Asia/Kolkata").timestamp()
        + np.arange(n_rows) * 60
    ).astype(np.int64)
    panel = _PanelStub(
        symbols=tuple(f"S{i}" for i in range(n_syms)),
        day_offsets=day_offsets,
        ts=ts,
        arrays={
            "open": open_.astype(np.float32),
            "high": high.astype(np.float32),
            "low": low.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        minute_array=minute,
    )

    short_params = VolumeBreakoutParams(
        hurst_window=30,
        use_hurst=True,
        volume_z_threshold=-10.0,
        deseasonalize=False,
    )
    short_strategy = VolumeBreakoutStrategy(short_params)
    with pytest.warns(UserWarning, match="window=30 < HURST_MIN_WINDOW"):
        short_signals = short_strategy.precompute(panel)
    total_short = int(short_signals["long"].sum() + short_signals["short"].sum())
    assert total_short <= 0.02 * n_rows * n_syms

    long_params = VolumeBreakoutParams(
        hurst_window=390,
        use_hurst=True,
        volume_z_threshold=-10.0,
        deseasonalize=False,
    )
    long_strategy = VolumeBreakoutStrategy(long_params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        long_strategy.precompute(panel)
    hurst_warnings = [w for w in caught if "window=" in str(w.message)]
    assert len(hurst_warnings) == 0


def test_weights_are_bounded_and_finite() -> None:
    params = VolumeBreakoutParams(
        exit_mode="time",
        gross=1.0,
        max_weight=0.10,
        sigma_floor=1e-5,
    )
    strategy = VolumeBreakoutStrategy(params)
    view = _make_view(3)
    signals = {
        "long": np.array([True, True, True]),
        "short": np.array([False, False, False]),
        "sigma": np.array([0.05, 1e-8, 1.0], dtype=np.float64),
        "vol_z": np.zeros(3, dtype=np.float64),
        "hurst": np.full(3, np.inf, dtype=np.float64),
    }
    state = PortfolioState(
        shares=np.zeros(3, dtype=np.float64),
        cash=0.0,
        equity=1_000_000.0,
        ts=view.ts,
    )

    target = strategy.on_decision(view, signals, state)
    assert target is not None
    weights = target.weights
    assert np.all(np.isfinite(weights))
    assert np.sum(np.abs(weights)) <= params.gross * (1 + 1e-9)
    assert np.all(np.abs(weights) <= params.max_weight + 1e-9)
    assert abs(weights[1]) <= params.max_weight + 1e-9


def test_vol_targeted_sizing_inverse_proportional() -> None:
    params = VolumeBreakoutParams(
        exit_mode="time",
        max_weight=10.0,
        gross=10.0,
    )
    strategy = VolumeBreakoutStrategy(params)
    view = _make_view(2)
    signals = {
        "long": np.array([True, True]),
        "short": np.array([False, False]),
        "sigma": np.array([0.2, 0.4], dtype=np.float64),
        "vol_z": np.zeros(2, dtype=np.float64),
        "hurst": np.full(2, 0.6, dtype=np.float64),
    }
    state = PortfolioState(
        shares=np.zeros(2, dtype=np.float64),
        cash=0.0,
        equity=1_000_000.0,
        ts=view.ts,
    )

    target = strategy.on_decision(view, signals, state)
    assert target is not None
    weights = target.weights
    assert weights[0] == pytest.approx(2.0 * weights[1], rel=1e-2)


def test_direction_fade_flips_sign() -> None:
    cont_params = VolumeBreakoutParams(
        use_hurst=False,
        deseasonalize=False,
        direction="continuation",
    )
    fade_params = VolumeBreakoutParams(
        use_hurst=False,
        deseasonalize=False,
        direction="fade",
    )
    cont_strategy = VolumeBreakoutStrategy(cont_params)
    fade_strategy = VolumeBreakoutStrategy(fade_params)

    panel = _make_panel(n_rows=450, n_syms=3, seed=42, volume_spikes=True)
    cont_signals = cont_strategy.precompute(panel)
    fade_signals = fade_strategy.precompute(panel)

    assert np.array_equal(cont_signals["long"], fade_signals["short"])
    assert np.array_equal(cont_signals["short"], fade_signals["long"])


def test_target_units_contract() -> None:
    params = VolumeBreakoutParams(use_hurst=False, deseasonalize=False)
    strategy = VolumeBreakoutStrategy(params)
    panel = _make_panel(n_rows=120, n_syms=2, seed=7, volume_spikes=True)

    result = strategy.target_units(panel, capital=1_000_000.0)

    assert isinstance(result, pd.DataFrame)
    assert list(result.index.names) == ["Timestamp", "Ticker"]
    assert "Target_Units" in result.columns
    assert result["Target_Units"].dtype == np.float64
    assert np.all(np.isfinite(result["Target_Units"].to_numpy()))


def test_registry_and_config() -> None:
    assert get("volume_breakout") is VolumeBreakoutStrategy

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "strategies"
        / "volume_breakout.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["strategy"] == "volume_breakout"

    built = build(cfg)
    assert isinstance(built, VolumeBreakoutStrategy)

    bad_params = {**cfg["params"], "not_a_real_param": 1}
    with pytest.raises(ValidationError):
        VolumeBreakoutParams(**bad_params)


@pytest.mark.slow
def test_real_data_signal_count() -> None:
    panel_mod = pytest.importorskip("nifty_quant.data.panel")
    PanelSpec = panel_mod.PanelSpec
    load_panel = panel_mod.load_panel

    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=("RELIANCE",),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )

    try:
        panel = load_panel(spec, memmap=False)
    except FileNotFoundError:
        pytest.skip("RELIANCE 2024 parquet data not available")
    if panel is None or panel.ts is None or len(panel.ts) == 0:
        pytest.skip("RELIANCE 2024 panel is empty")

    strategy = VolumeBreakoutStrategy(VolumeBreakoutParams())
    signals = strategy.precompute(panel)
    total = int(signals["long"].sum() + signals["short"].sum())
    assert 100 <= total <= 600

    no_hurst_params = VolumeBreakoutParams(use_hurst=False)
    no_hurst_signals = VolumeBreakoutStrategy(no_hurst_params).precompute(panel)
    total_no_hurst = int(
        no_hurst_signals["long"].sum() + no_hurst_signals["short"].sum()
    )
    assert total_no_hurst >= 3 * total
