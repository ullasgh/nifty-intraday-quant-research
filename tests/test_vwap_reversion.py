from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from nifty_quant.strategy.base import PortfolioState, TargetPortfolio
from nifty_quant.strategy.plugins.vwap_reversion import VwapReversionParams, VwapReversionStrategy
from nifty_quant.strategy.registry import build, get

_DEFAULT_TS = int(pd.Timestamp("2024-01-01 10:00:00", tz="Asia/Kolkata").timestamp())


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


def _make_multisession_minutes(lengths: list[int], start_minutes: list[int]) -> np.ndarray:
    parts = []
    for length, start_minute in zip(lengths, start_minutes):
        parts.append(np.arange(start_minute, start_minute + length, dtype=np.int64))
    return np.concatenate(parts)


def _make_panel(
    arrays: dict[str, np.ndarray],
    *,
    day_offsets: np.ndarray | None = None,
    minute_array: np.ndarray | None = None,
    symbols: tuple[str, ...] | None = None,
) -> _PanelStub:
    n_rows = arrays["close"].shape[0]
    n_symbols = arrays["close"].shape[1] if arrays["close"].ndim == 2 else 1
    if symbols is None:
        symbols = tuple(f"S{i}" for i in range(n_symbols))
    if day_offsets is None:
        day_offsets = np.array([0, n_rows], dtype=np.int64)
    if minute_array is None:
        minute_array = np.arange(556, 556 + n_rows, dtype=np.int64)
    return _PanelStub(
        symbols=symbols,
        day_offsets=day_offsets,
        ts=np.arange(n_rows, dtype=np.int64),
        arrays=arrays,
        minute_array=minute_array,
    )


def _make_view(
    n_syms: int,
    tradable: np.ndarray | None = None,
    ts: int = _DEFAULT_TS,
) -> _ViewStub:
    symbols = tuple(f"S{i}" for i in range(n_syms))
    if tradable is None:
        tradable = np.ones(n_syms, dtype=bool)
    return _ViewStub(
        ts=ts,
        session_date=datetime.date(2024, 1, 1),
        symbols=symbols,
        tradable=tradable,
    )


def _make_state(shares: np.ndarray | None = None, equity: float = 1e6) -> PortfolioState:
    if shares is None:
        shares = np.zeros(1, dtype=np.float64)
    return PortfolioState(
        shares=np.asarray(shares, dtype=np.float64),
        cash=equity,
        equity=equity,
        ts=_DEFAULT_TS,
    )


def _assert_array_prefix_equal(actual: np.ndarray, expected: np.ndarray, stop: int) -> None:
    actual_prefix = actual[: stop + 1]
    expected_prefix = expected[: stop + 1]
    np.testing.assert_array_equal(np.isnan(actual_prefix), np.isnan(expected_prefix))
    not_nan = ~np.isnan(actual_prefix)
    np.testing.assert_array_equal(actual_prefix[not_nan], expected_prefix[not_nan])


def test_registered() -> None:
    assert get("vwap_reversion") is VwapReversionStrategy

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "strategies" / "vwap_reversion.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text())
    strategy = build(cfg)
    assert isinstance(strategy, VwapReversionStrategy)


def test_vwap_matches_hand_computed_micro_data() -> None:
    close = np.array([[100.0], [200.0], [150.0]], dtype=np.float64)
    volume = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    # high/low deliberately differ from close so hlc3 would give a different typical price
    # than "close" -- proves the code path actually reads typical_price="close" below.
    high = np.array([[101.0], [201.0], [151.0]], dtype=np.float64)
    low = np.array([[99.0], [199.0], [149.0]], dtype=np.float64)

    panel = _make_panel({"high": high, "low": low, "close": close, "volume": volume})
    strategy = VwapReversionStrategy(
        VwapReversionParams(typical_price="close", normalize=False, vol_window=1)
    )

    signals = strategy.precompute(panel)

    # vwap[0] = 100*10/10 = 100.0
    # vwap[1] = (100*10 + 200*20)/(10+20) = 5000/30 = 166.66667
    # vwap[2] = (5000 + 150*30)/(30+30) = 9500/60 = 158.33333
    vwap = np.array([100.0, 5000.0 / 30.0, 9500.0 / 60.0])
    expected_dev = (close[:, 0] - vwap) / vwap

    np.testing.assert_allclose(signals["dev"][:, 0], expected_dev, rtol=1e-6, atol=1e-12)


def test_vwap_resets_each_session() -> None:
    close = np.array(
        [[100.0], [101.0], [102.0], [103.0], [104.0], [200.0], [202.0], [204.0], [206.0]],
        dtype=np.float64,
    )
    volume = np.full((9, 1), 10.0, dtype=np.float64)
    high = close + 2.0
    low = close - 2.0
    minute_array = _make_multisession_minutes([5, 4], [556, 556])
    day_offsets = np.array([0, 5, 9], dtype=np.int64)

    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        day_offsets=day_offsets,
        minute_array=minute_array,
    )
    strategy = VwapReversionStrategy(
        VwapReversionParams(normalize=False, vol_window=1, typical_price="close")
    )
    signals = strategy.precompute(panel)

    # Session 2's first row (index 5) is its own only contributor so far, so its VWAP
    # equals its own typical price and the raw deviation is exactly zero.
    assert signals["dev"][5, 0] == pytest.approx(0.0, abs=1e-12)

    close_mutated = close.copy()
    close_mutated[4, 0] = 1e9  # poison session 1's last bar
    mutated_panel = _make_panel(
        {"high": high, "low": low, "close": close_mutated, "volume": volume},
        day_offsets=day_offsets,
        minute_array=minute_array,
    )
    mutated_signals = strategy.precompute(mutated_panel)

    np.testing.assert_array_equal(mutated_signals["dev"][5:6, :], signals["dev"][5:6, :])


def test_irregular_session_lengths() -> None:
    lengths = [60, 105, 375]  # Muhurat-like, DR-like, and a regular session
    rng = np.random.default_rng(2024)
    close_parts = []
    for length in lengths:
        walk = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=(length, 1)), axis=0)
        close_parts.append(np.clip(walk, 1e-3, None))
    close = np.concatenate(close_parts, axis=0)

    delta = rng.uniform(0.1, 1.0, size=close.shape)
    high = close + delta
    low = np.clip(close - delta, 1e-3, None)
    volume = rng.uniform(100.0, 1000.0, size=close.shape)

    minute_array = _make_multisession_minutes(lengths, [556, 556, 556])
    day_offsets = np.array([0, 60, 165, 540], dtype=np.int64)

    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        day_offsets=day_offsets,
        minute_array=minute_array,
    )
    strategy = VwapReversionStrategy(VwapReversionParams(normalize=False, vol_window=1))
    signals = strategy.precompute(panel)

    for start in (0, 60, 165):
        assert signals["dev"][start, 0] == pytest.approx(0.0, abs=1e-9)


def test_nan_bars_do_not_poison_session_vwap() -> None:
    close = np.array([[100.0], [110.0], [np.nan], [121.0], [132.0]])
    volume = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
    high = np.array([[101.0], [111.0], [121.0], [122.0], [133.0]])
    low = np.array([[99.0], [109.0], [119.0], [120.0], [131.0]])

    minute_array = _make_multisession_minutes([5], [556])
    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        minute_array=minute_array,
    )
    strategy = VwapReversionStrategy(
        VwapReversionParams(typical_price="close", normalize=False, vol_window=1)
    )
    signals = strategy.precompute(panel)

    assert np.isnan(signals["dev"][2, 0])

    # Independently recompute the expected VWAP, skipping the NaN bar entirely (not
    # forward-filled, not NaN-propagated forward).
    close_flat = close[:, 0]
    volume_flat = volume[:, 0]
    expected_vwap = np.full(5, np.nan, dtype=np.float64)
    for i in range(5):
        if np.isfinite(close_flat[i]):
            prefix_close = close_flat[: i + 1]
            prefix_volume = volume_flat[: i + 1]
            valid = np.isfinite(prefix_close)
            expected_vwap[i] = np.sum(prefix_close[valid] * prefix_volume[valid]) / np.sum(
                prefix_volume[valid]
            )

    for i in (3, 4):
        expected_dev = (close_flat[i] - expected_vwap[i]) / expected_vwap[i]
        assert signals["dev"][i, 0] == pytest.approx(expected_dev, rel=1e-6, abs=1e-12)


def test_never_reads_0915_bar() -> None:
    normal_close = np.array([[100.0], [101.0], [102.0], [103.0]])
    normal_volume = np.full((4, 1), 10.0, dtype=np.float64)
    normal_high = normal_close + 1.0
    normal_low = normal_close - 1.0

    # Poisoned 09:15 bar: close > high, exactly the documented real-world corruption.
    poisoned_close = np.array([[1e9], [100.0], [101.0], [102.0], [103.0]])
    poisoned_volume = np.full((5, 1), 10.0, dtype=np.float64)
    poisoned_high = np.array([[99.0], [101.0], [102.0], [103.0], [104.0]])
    poisoned_low = np.array([[98.0], [99.0], [100.0], [101.0], [102.0]])

    with_minutes = _make_multisession_minutes([1, 4], [555, 556])
    without_minutes = _make_multisession_minutes([4], [556])

    with_panel = _make_panel(
        {
            "high": poisoned_high,
            "low": poisoned_low,
            "close": poisoned_close,
            "volume": poisoned_volume,
        },
        day_offsets=np.array([0, 5], dtype=np.int64),
        minute_array=with_minutes,
    )
    without_panel = _make_panel(
        {
            "high": normal_high,
            "low": normal_low,
            "close": normal_close,
            "volume": normal_volume,
        },
        day_offsets=np.array([0, 4], dtype=np.int64),
        minute_array=without_minutes,
    )

    strategy = VwapReversionStrategy(
        VwapReversionParams(normalize=False, vol_window=1, typical_price="close")
    )
    with_signals = strategy.precompute(with_panel)
    without_signals = strategy.precompute(without_panel)

    np.testing.assert_array_equal(with_signals["dev"][1:, :], without_signals["dev"])

    with pytest.raises(ValidationError):
        VwapReversionParams(session_start_time="09:15")


def test_no_lookahead() -> None:
    n_symbols = 2
    lengths = [100, 110, 90]
    day_offsets = np.array([0, 100, 210, 300], dtype=np.int64)

    rng = np.random.default_rng(7)
    close_parts = []
    for length in lengths:
        steps = rng.normal(0.0, 0.3, size=(length, n_symbols))
        close_part = 100.0 + np.cumsum(steps, axis=0)
        close_parts.append(np.clip(close_part, 1e-3, None))
    close = np.concatenate(close_parts, axis=0)

    delta = rng.uniform(0.1, 0.8, size=close.shape)
    high = close + delta
    low = np.clip(close - delta, 1e-3, None)
    volume = rng.uniform(100.0, 1000.0, size=close.shape)

    minute_array = _make_multisession_minutes(lengths, [556, 556, 556])
    arrays = {"high": high, "low": low, "close": close, "volume": volume}
    panel = _make_panel(arrays, day_offsets=day_offsets, minute_array=minute_array)

    strategy = VwapReversionStrategy(VwapReversionParams())
    original_signals = strategy.precompute(panel)
    cut = 150

    for variant in ("multiply", "nan"):
        mutated_arrays = {
            "high": high.copy(),
            "low": low.copy(),
            "close": close.copy(),
            "volume": volume.copy(),
        }
        if variant == "multiply":
            for key in mutated_arrays:
                mutated_arrays[key][cut + 1 :] *= 1e6
        else:
            for key in mutated_arrays:
                mutated_arrays[key][cut + 1 :] = np.nan

        mutated_panel = _make_panel(
            mutated_arrays, day_offsets=day_offsets, minute_array=minute_array
        )
        mutated_signals = strategy.precompute(mutated_panel)

        for key in original_signals:
            _assert_array_prefix_equal(original_signals[key], mutated_signals[key], cut)


def test_precompute_2d_and_on_decision_1d() -> None:
    n_rows = 10
    n_symbols = 3
    rng = np.random.default_rng(99)
    close = rng.lognormal(mean=4.6, sigma=0.1, size=(n_rows, n_symbols))
    delta = rng.uniform(0.1, 0.5, size=close.shape)
    high = close + delta
    low = close - delta
    volume = rng.uniform(100.0, 1000.0, size=close.shape)

    minute_array = _make_multisession_minutes([n_rows], [556])
    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        minute_array=minute_array,
    )
    strategy = VwapReversionStrategy(VwapReversionParams(normalize=True, vol_window=1))
    signals = strategy.precompute(panel)

    for value in signals.values():
        assert value.shape == (n_rows, n_symbols)

    row = 5
    signals_1d = {key: value[row] for key, value in signals.items()}
    view = _make_view(n_symbols, tradable=np.ones(n_symbols, dtype=bool))
    state = _make_state(np.zeros(n_symbols, dtype=np.float64))

    target = strategy.on_decision(view, signals_1d, state)

    assert target is not None
    assert isinstance(target, TargetPortfolio)
    assert target.weights.shape == (n_symbols,)
    target.validate(n_symbols)


def test_signal_direction() -> None:
    strategy = VwapReversionStrategy(VwapReversionParams())
    view = _make_view(2, tradable=np.ones(2, dtype=bool))
    state = _make_state(np.zeros(2, dtype=np.float64))
    signals = {
        "dev": np.array([20.0, -20.0]),
        "sigma": np.array([0.1, 0.2]),
        "bars_since_open": np.array([20, 20]),
    }

    target = strategy.on_decision(view, signals, state)

    assert target is not None
    assert target.weights[0] < 0.0  # far above VWAP -> short
    assert target.weights[1] > 0.0  # far below VWAP -> long
    target.validate(2)


def test_min_bars_since_open_blocks_early_trades() -> None:
    strategy = VwapReversionStrategy(VwapReversionParams())
    view = _make_view(1, tradable=np.ones(1, dtype=bool))
    state = _make_state(np.zeros(1, dtype=np.float64))
    signals = {
        "dev": np.array([20.0]),
        "sigma": np.array([0.1]),
        "bars_since_open": np.array([10]),
    }

    blocked = strategy.on_decision(view, signals, state)

    assert blocked is not None
    assert blocked.weights[0] == 0.0

    allowed_signals = {**signals, "bars_since_open": np.array([20])}
    allowed = strategy.on_decision(view, allowed_signals, state)

    assert allowed is not None
    assert allowed.weights[0] != 0.0


def test_weights_respect_gross_and_max_weight() -> None:
    n_symbols = 4
    strategy = VwapReversionStrategy(VwapReversionParams(gross=0.5, max_weight=0.2))
    view = _make_view(n_symbols, tradable=np.ones(n_symbols, dtype=bool))
    state = _make_state(np.zeros(n_symbols, dtype=np.float64))
    signals = {
        "dev": np.array([20.0, -20.0, 15.0, -15.0]),
        "sigma": np.array([0.1, 0.2, 0.15, 0.3]),
        "bars_since_open": np.array([20, 20, 20, 20]),
    }

    target = strategy.on_decision(view, signals, state)

    assert target is not None
    target.validate(n_symbols)
    assert np.all(np.isfinite(target.weights))
    assert np.sum(np.abs(target.weights)) <= 0.5 * (1 + 1e-9)


def test_deviation_zscore_is_stationary_across_session() -> None:
    """THE TEST FOR THE UNITS-MISMATCH BUG.

    `dev` is a displacement accumulated since session open, so under a random walk it
    scales like `sigma_per_bar * sqrt(bars_since_open)`. Dividing by an unscaled
    per-bar `sigma` (the pre-fix behavior) is therefore NOT a stationary z-score: it
    drifts upward across the session. Horizon-scaling `sigma` by
    `sqrt(max(bars_since_open, 1))` before dividing (the fix) makes it stationary.

    Many symbols (200) are used so per-band statistics average out sampling noise.
    Bands are chosen well past the vol_window=30 warmup and away from the panel's
    edges. Tolerances: empirically measured on this exact construction (different
    seed) the fixed ratio was ~1.09 and the unscaled ratio was ~2.01, matching the
    predicted sqrt(elapsed-bar-ratio) growth for the old version; [0.5, 2.0] for the
    fixed statistic gives headroom for RNG-seed sensitivity while still being tight
    enough to catch a regression back toward the old sqrt-growth (which would push the
    ratio to ~2.0 or beyond), and `> 1.5` for the old statistic is comfortably below
    its measured ~2.01 so the "old version fails stationarity" assertion is robust.
    """
    n_rows, n_symbols = 300, 200
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.0, 0.0015, size=(n_rows, n_symbols))
    close = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    delta = 0.05 + 0.02 * rng.random(close.shape)
    high = close + delta
    low = close - delta
    volume = rng.uniform(500.0, 1500.0, size=close.shape)

    minute_array = np.full(n_rows, 556, dtype=np.int64)
    day_offsets = np.array([0, n_rows], dtype=np.int64)
    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        day_offsets=day_offsets,
        minute_array=minute_array,
    )

    strat_new = VwapReversionStrategy(VwapReversionParams(vol_window=30, normalize=True))
    signals_new = strat_new.precompute(panel)
    dev_z_new = signals_new["dev"]
    sigma = signals_new["sigma"]

    strat_raw = VwapReversionStrategy(VwapReversionParams(vol_window=30, normalize=False))
    signals_raw = strat_raw.precompute(panel)
    dev_raw = signals_raw["dev"]

    # Independently reproduce the OLD (pre-fix) unscaled-per-bar-sigma normalization,
    # without calling any strategy internals, to prove the fix actually matters.
    sigma_guard_old = np.maximum(
        np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf),
        VwapReversionParams().sigma_floor,
    )
    dev_z_old = np.where(np.isfinite(dev_raw), dev_raw / sigma_guard_old, np.nan)

    first_band = slice(40, 100)
    last_band = slice(200, 260)

    def mean_abs_dev(arr: np.ndarray, band: slice) -> float:
        return float(np.nanmean(np.abs(arr[band, :])))

    ratio_new = mean_abs_dev(dev_z_new, last_band) / mean_abs_dev(dev_z_new, first_band)
    assert 0.5 <= ratio_new <= 2.0, (
        f"Fixed z-score not stationary: late/early ratio {ratio_new:.3f} outside [0.5, 2.0]"
    )

    ratio_old = mean_abs_dev(dev_z_old, last_band) / mean_abs_dev(dev_z_old, first_band)
    assert ratio_old > 1.5, (
        f"Old unscaled version should drift, but ratio {ratio_old:.3f} <= 1.5"
    )


def test_entry_threshold_default_is_two_sigma() -> None:
    """Pins the default to a true 2-sigma threshold, not a tuned fudge factor."""
    assert VwapReversionParams().entry_threshold == 2.0


def test_irregular_session_bars_since_open() -> None:
    """bars_since_open must reset for each session even with irregular lengths."""
    lengths = [60, 105]  # Muhurat-like, DR-Saturday-like
    minute_array = _make_multisession_minutes(lengths, [556, 556])
    day_offsets = np.array([0, 60, 165], dtype=np.int64)

    n_rows = 165
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(n_rows, 1)), axis=0))
    high = close + 0.01
    low = close - 0.01
    volume = rng.uniform(500.0, 1000.0, size=close.shape)

    panel = _make_panel(
        {"high": high, "low": low, "close": close, "volume": volume},
        day_offsets=day_offsets,
        minute_array=minute_array,
    )

    strategy = VwapReversionStrategy(VwapReversionParams())
    signals = strategy.precompute(panel)
    bars = signals["bars_since_open"]

    assert bars[0, 0] == 0  # session 1's first row
    assert bars[59, 0] == 59  # session 1's last row
    assert bars[60, 0] == 0  # session 2's first row -- reset, not 60
    assert bars[164, 0] == 104  # session 2's last row (105th bar of session 2)
