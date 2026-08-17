"""Tests for :class:`EODOverExtensionStrategy`."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import nifty_quant.strategy.plugins  # noqa: F401
from nifty_quant import guards
from nifty_quant.data.panel import Panel
from nifty_quant.strategy import registry
from nifty_quant.strategy.base import ArrayMarketView, PortfolioState
from nifty_quant.strategy.plugins.eod_overextension import (
    EODOverExtensionParams,
    EODOverExtensionStrategy,
)

_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_FIELD_NAMES = ("open", "high", "low", "close", "volume")


def _make_ts(session_date: dt.date, hhmm: str) -> int:
    hour_str, minute_str = hhmm.split(":")
    return int(
        dt.datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            int(hour_str),
            int(minute_str),
            tzinfo=_IST,
        ).timestamp()
    )


def _regular_times() -> list[str]:
    times = []
    current = 9 * 60 + 15
    end = 15 * 60 + 29
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _muhurat_times() -> list[str]:
    times = []
    current = 18 * 60
    end = 18 * 60 + 59
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _short_dr_times() -> list[str]:
    times = []
    current = 9 * 60 + 15
    end = 10 * 60 + 59
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _build_panel(
    specs: list[tuple[dt.date, list[str]]],
    symbols: tuple[str, ...],
    close: np.ndarray,
    volume: np.ndarray,
    rng: np.random.Generator,
) -> Panel:
    ts_parts = []
    dates = []
    day_offsets = [0]
    for session_date, times in specs:
        for hhmm in times:
            ts_parts.append(_make_ts(session_date, hhmm))
        day_offsets.append(day_offsets[-1] + len(times))
        dates.append(session_date)

    close_f = np.asarray(close, dtype=np.float64)
    volume_f = np.asarray(volume, dtype=np.float64)
    open_ = np.empty_like(close_f, dtype=np.float64)
    open_[0] = 100.0
    open_[1:] = close_f[:-1]

    wick = 0.01 * close_f + rng.uniform(0.0005, 0.002, size=close_f.shape)
    high = np.maximum(open_, close_f) + wick
    low = np.minimum(open_, close_f) - wick
    np.maximum(low, 0.01, out=low)

    fields = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close_f.astype(np.float32),
        "volume": volume_f.astype(np.float32),
    }
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=np.asarray(ts_parts, dtype=np.int64),
        day_offsets=np.asarray(day_offsets, dtype=np.int32),
        dates=np.asarray(dates, dtype=object),
    )


def _day_index_for_cursor(panel: Panel, cursor: int) -> int:
    day_idx = int(np.searchsorted(panel.day_offsets, cursor, side="right") - 1)
    return int(min(max(day_idx, 0), panel.n_days() - 1))


def _build_view(panel: Panel, cursor: int, tradable: np.ndarray) -> ArrayMarketView:
    panel_arrays = {f: panel.field(f) for f in _FIELD_NAMES}
    return ArrayMarketView(
        panel_arrays,
        cursor,
        symbols=panel.symbols,
        ts_array=panel.ts,
        tradable=tradable,
        session_date=panel.dates[_day_index_for_cursor(panel, cursor)],
        minute_of_day=panel.minute_of_day(),
        day_offsets=panel.day_offsets,
    )


def _nan_mask(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "b":
        return np.zeros_like(arr, dtype=bool)
    return np.isnan(arr)


def _assert_equal_nan_aware(expected: np.ndarray, actual: np.ndarray) -> None:
    assert expected.shape == actual.shape
    expected_nan = _nan_mask(expected)
    actual_nan = _nan_mask(actual)
    np.testing.assert_array_equal(expected_nan, actual_nan)
    if expected.dtype.kind == "b":
        np.testing.assert_array_equal(expected, actual)
    else:
        np.testing.assert_array_equal(expected[~expected_nan], actual[~actual_nan])


def _random_close_volume(
    n_rows: int, n_symbols: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    close = np.empty((n_rows, n_symbols), dtype=np.float64)
    close[0] = np.linspace(100.0, 100.0 + 10.0 * (n_symbols - 1), n_symbols)
    for i in range(1, n_rows):
        close[i] = close[i - 1] + rng.normal(0.0, 0.3, size=n_symbols)
    volume = rng.uniform(90_000.0, 110_000.0, size=(n_rows, n_symbols))
    return close, volume


def _mutate_panel_future(panel: Panel, cut_row: int, mode: str) -> Panel:
    fields = {}
    future_mask = np.arange(panel.n_rows()) > cut_row
    for name in _FIELD_NAMES:
        data = np.array(panel.field(name), dtype=np.float64, copy=True)
        if mode == "1e6":
            data[future_mask, :] *= 1e6
        elif mode == "nan":
            data[future_mask, :] = np.nan
        else:
            raise ValueError(f"Unknown mutation mode: {mode!r}")
        fields[name] = data.astype(np.float32)
    return Panel(
        fields=fields,
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )


def _build_minimal_view(symbols: tuple[str, ...]) -> ArrayMarketView:
    n_syms = len(symbols)
    panel_arrays = {
        field: np.zeros((1, n_syms), dtype=np.float32) for field in _FIELD_NAMES
    }
    return ArrayMarketView(
        panel_arrays,
        cursor=0,
        symbols=symbols,
        ts_array=np.array([0], dtype=np.int64),
        tradable=np.ones(n_syms, dtype=bool),
        session_date=dt.date(2024, 1, 10),
        minute_of_day=np.array([600], dtype=np.int64),
        day_offsets=np.array([0, 1], dtype=np.int32),
    )


def _build_signal_fixture(
    dates: tuple[dt.date, ...],
) -> tuple[Panel, dict[str, np.ndarray], EODOverExtensionStrategy, int, int, int]:
    symbols = ("TARGET", "FLAT1", "FLAT2")
    specs = [(d, _regular_times()) for d in dates]
    regular = _regular_times()
    n_sessions = len(specs)
    n_syms = len(symbols)
    n_rows = n_sessions * len(regular)

    close = np.zeros((n_rows, n_syms), dtype=np.float64)
    volume = np.full((n_rows, n_syms), 100_000.0, dtype=np.float64)

    signal_local = regular.index("15:10")
    entry_local = regular.index("15:20") - 1
    exit_local = regular.index("09:16") - 1
    length = len(regular)

    for d in range(n_sessions):
        start = d * length
        end = start + length
        if d == 0:
            close[start:end, :] = 100.0
        elif d == 1:
            trend = np.linspace(100.0, 104.0, signal_local + 1)
            close[start : start + signal_local + 1, 0] = trend
            close[start + signal_local + 1 : end, 0] = 104.0
            close[start:end, 1] = 100.0
            close[start:end, 2] = 100.0
            volume[start + signal_local, 0] = 5_000_000.0
        else:
            close[start:end, 0] = 104.0
            close[start:end, 1] = 100.0
            close[start:end, 2] = 100.0

    rng = np.random.default_rng(20240101)
    panel = _build_panel(specs, symbols, close, volume, rng)
    strategy = registry.build(
        {
            "strategy": "eod_overextension",
            "params": {
                "session_open_time": "09:16",
                "signal_time": "15:10",
                "entry_time": "15:20",
                "exit_time": "09:16",
                "entry_threshold": 0.02,
                "volume_z_threshold": 1.0,
                "volume_window": 10,
                "vol_window": 10,
                "n_max": 10,
                "direction": "long",
                "max_weight": 1.0,
                "gross": 1.0,
            },
        }
    )
    with guards.strictness(guards.Strictness.FULL):
        signals = strategy.precompute(panel)

    entry_cursor = int(panel.day_offsets[1]) + entry_local
    exit_cursor = int(panel.day_offsets[2]) + exit_local
    return panel, signals, strategy, 0, entry_cursor, exit_cursor


def _run_overnight_hold_test(dates: tuple[dt.date, ...]) -> None:
    panel, signals, strategy, target_idx, entry_cursor, exit_cursor = (
        _build_signal_fixture(dates)
    )
    n_syms = panel.n_symbols()
    tradable = np.ones(n_syms, dtype=bool)

    assert signals["entry_long"][entry_cursor, target_idx]

    entry_view = _build_view(panel, entry_cursor, tradable)
    entry_signals = {key: value[entry_cursor] for key, value in signals.items()}
    entry_state = PortfolioState(
        shares=np.zeros(n_syms, dtype=np.float64),
        cash=100_000.0,
        equity=100_000.0,
        ts=int(panel.ts[entry_cursor]),
    )
    with guards.strictness(guards.Strictness.FULL):
        target = strategy.on_decision(entry_view, entry_signals, entry_state)
    assert target is not None
    assert target.weights[target_idx] != 0.0
    others = np.ones(n_syms, dtype=bool)
    others[target_idx] = False
    assert np.all(target.weights[others] == 0.0)

    shares = np.sign(target.weights).astype(np.float64)
    mid_cursor = entry_cursor + 5
    assert mid_cursor < int(panel.day_offsets[2])
    mid_view = _build_view(panel, mid_cursor, tradable)
    mid_signals = {key: value[mid_cursor] for key, value in signals.items()}
    mid_state = PortfolioState(
        shares=shares,
        cash=100_000.0,
        equity=100_000.0,
        ts=int(panel.ts[mid_cursor]),
    )
    with guards.strictness(guards.Strictness.FULL):
        mid_target = strategy.on_decision(mid_view, mid_signals, mid_state)
    assert mid_target is None

    exit_view = _build_view(panel, exit_cursor, tradable)
    exit_signals = {key: value[exit_cursor] for key, value in signals.items()}
    exit_state = PortfolioState(
        shares=shares,
        cash=100_000.0,
        equity=100_000.0,
        ts=int(panel.ts[exit_cursor]),
    )
    with guards.strictness(guards.Strictness.FULL):
        exit_target = strategy.on_decision(exit_view, exit_signals, exit_state)
    assert exit_target is not None
    assert np.all(exit_target.weights == 0.0)


def test_registered() -> None:
    assert "eod_overextension" in registry.available()
    assert registry.get("eod_overextension") is EODOverExtensionStrategy


def test_precompute_2d_and_on_decision_1d() -> None:
    dates = [dt.date(2024, 1, 8), dt.date(2024, 1, 9), dt.date(2024, 1, 10)]
    n_syms = 4
    specs = [(d, _regular_times()) for d in dates]
    rng = np.random.default_rng(42)
    n_rows = sum(len(times) for _, times in specs)
    close, volume = _random_close_volume(n_rows, n_syms, rng)
    panel = _build_panel(specs, tuple(f"S{i}" for i in range(n_syms)), close, volume, rng)

    strategy = registry.build({"strategy": "eod_overextension", "params": {}})
    with guards.strictness(guards.Strictness.FULL):
        signals = strategy.precompute(panel)

    expected_dtypes = {
        "entry_long": np.dtype("bool"),
        "entry_short": np.dtype("bool"),
        "exit_due": np.dtype("bool"),
        "sigma": np.dtype("float64"),
        "session_return": np.dtype("float64"),
        "position_in_range": np.dtype("float64"),
        "vol_z": np.dtype("float64"),
    }
    assert set(signals) == set(expected_dtypes)
    for key, array in signals.items():
        assert array.shape == (panel.n_rows(), panel.n_symbols())
        assert array.dtype == expected_dtypes[key]

    cursor = int(panel.day_offsets[1])
    assert cursor != 0
    assert np.any(signals["exit_due"][cursor])

    signals_row = {key: value[cursor] for key, value in signals.items()}
    for value in signals_row.values():
        assert value.shape == (n_syms,)

    view = _build_view(panel, cursor, np.ones(n_syms, dtype=bool))
    state = PortfolioState(
        shares=np.zeros(n_syms, dtype=np.float64),
        cash=100_000.0,
        equity=100_000.0,
        ts=int(panel.ts[cursor]),
    )
    with guards.strictness(guards.Strictness.FULL):
        target = strategy.on_decision(view, signals_row, state)
    assert target is not None
    assert target.weights.shape == (n_syms,)
    target.validate(n_syms)


def test_no_lookahead() -> None:
    dates = [
        dt.date(2024, 1, 8),
        dt.date(2024, 1, 9),
        dt.date(2024, 1, 10),
        dt.date(2024, 1, 11),
    ]
    n_syms = 3
    specs = [(d, _regular_times()) for d in dates]
    rng = np.random.default_rng(7)
    n_rows = sum(len(times) for _, times in specs)
    close, volume = _random_close_volume(n_rows, n_syms, rng)
    panel = _build_panel(specs, tuple(f"S{i}" for i in range(n_syms)), close, volume, rng)

    strategy = registry.build({"strategy": "eod_overextension", "params": {}})
    with guards.strictness(guards.Strictness.FULL):
        baseline = strategy.precompute(panel)

    cut_rows = [
        int(panel.day_offsets[1]) + 180,
        int(panel.day_offsets[2]) + 200,
        int(panel.day_offsets[3]) - 1,
    ]

    for mode in ("1e6", "nan"):
        for cut_row in cut_rows:
            mutated_panel = _mutate_panel_future(panel, cut_row, mode)
            with guards.strictness(guards.Strictness.FULL):
                mutated = strategy.precompute(mutated_panel)
            for key in baseline:
                left = baseline[key][: cut_row + 1]
                right = mutated[key][: cut_row + 1]
                _assert_equal_nan_aware(left, right)


def test_never_reads_0915_bar() -> None:
    dates = [dt.date(2024, 1, 8), dt.date(2024, 1, 9)]
    n_syms = 3
    specs = [(d, _regular_times()) for d in dates]
    rng = np.random.default_rng(11)
    n_rows = sum(len(times) for _, times in specs)
    close, volume = _random_close_volume(n_rows, n_syms, rng)
    panel = _build_panel(specs, tuple(f"S{i}" for i in range(n_syms)), close, volume, rng)

    strategy = registry.build({"strategy": "eod_overextension", "params": {}})
    with guards.strictness(guards.Strictness.FULL):
        baseline = strategy.precompute(panel)

    fields = {
        name: np.array(panel.field(name), dtype=np.float32, copy=True)
        for name in _FIELD_NAMES
    }
    poison_rows = np.flatnonzero(panel.minute_of_day() == 555)
    assert poison_rows.size >= 1
    for row in poison_rows:
        fields["close"][row, :] = np.float32(1e9)
        assert np.all(
            fields["close"][row, :].astype(np.float64)
            > fields["high"][row, :].astype(np.float64)
        )

    poisoned_panel = Panel(
        fields=fields,
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )
    with guards.strictness(guards.Strictness.FULL):
        poisoned_signals = strategy.precompute(poisoned_panel)

    assert set(poisoned_signals) == set(baseline)
    for key in baseline:
        _assert_equal_nan_aware(baseline[key], poisoned_signals[key])


def test_rejects_0915_config() -> None:
    for field in ("session_open_time", "signal_time", "entry_time", "exit_time"):
        with pytest.raises(Exception):
            EODOverExtensionParams(**{field: "09:15"})


def test_holds_overnight_then_exits_next_session() -> None:
    _run_overnight_hold_test(
        (dt.date(2024, 1, 8), dt.date(2024, 1, 9), dt.date(2024, 1, 10))
    )


def test_does_not_carry_across_calendar_gap_incorrectly() -> None:
    _run_overnight_hold_test(
        (dt.date(2024, 1, 10), dt.date(2024, 1, 11), dt.date(2024, 1, 15))
    )


def test_irregular_session_length() -> None:
    specs = [
        (dt.date(2024, 1, 10), _regular_times()),
        (dt.date(2024, 1, 11), _short_dr_times()),
        (dt.date(2024, 1, 12), _muhurat_times()),
        (dt.date(2024, 1, 15), _regular_times()),
    ]
    n_syms = 3
    rng = np.random.default_rng(4321)
    n_rows = sum(len(times) for _, times in specs)
    close, volume = _random_close_volume(n_rows, n_syms, rng)
    panel = _build_panel(specs, tuple(f"S{i}" for i in range(n_syms)), close, volume, rng)

    strategy = registry.build({"strategy": "eod_overextension", "params": {}})
    with guards.strictness(guards.Strictness.FULL):
        signals = strategy.precompute(panel)

    for array in signals.values():
        assert array.shape == (panel.n_rows(), panel.n_symbols())

    for key in ("exit_due", "entry_long", "entry_short"):
        for day_idx in range(panel.n_days()):
            start = int(panel.day_offsets[day_idx])
            end = int(panel.day_offsets[day_idx + 1])
            segment = signals[key][start:end]
            assert segment.shape == (end - start, n_syms)
            assert np.all(np.flatnonzero(segment) < segment.size)


def test_vol_targeted_sizing_inverse_to_vol() -> None:
    symbols = ("A", "B")
    n_syms = len(symbols)
    panel_arrays = {
        field: np.zeros((1, n_syms), dtype=np.float32) for field in _FIELD_NAMES
    }
    view = ArrayMarketView(
        panel_arrays,
        cursor=0,
        symbols=symbols,
        ts_array=np.array([0], dtype=np.int64),
        tradable=np.ones(n_syms, dtype=bool),
        session_date=dt.date(2024, 1, 10),
        minute_of_day=np.array([600], dtype=np.int64),
        day_offsets=np.array([0, 1], dtype=np.int32),
    )
    signals_row = {
        "entry_long": np.array([True, True]),
        "entry_short": np.array([False, False]),
        "exit_due": np.array([False, False]),
        "sigma": np.array([0.01, 0.02], dtype=np.float64),
    }
    strategy = registry.build(
        {
            "strategy": "eod_overextension",
            "params": {"direction": "long", "max_weight": 1000.0, "gross": 1.0},
        }
    )
    state = PortfolioState(
        shares=np.zeros(n_syms, dtype=np.float64),
        cash=100_000.0,
        equity=100_000.0,
        ts=0,
    )
    with guards.strictness(guards.Strictness.FULL):
        target = strategy.on_decision(view, signals_row, state)
    assert target is not None
    target.validate(n_syms)
    assert abs(target.weights[0]) > abs(target.weights[1]) > 0
    assert not np.any(np.abs(target.weights) == strategy.params.max_weight)


def test_entry_signal_causal_across_session_boundary() -> None:
    panel, baseline_signals, strategy, target_idx, entry_cursor, _exit_cursor = (
        _build_signal_fixture(
            (dt.date(2024, 1, 8), dt.date(2024, 1, 9), dt.date(2024, 1, 10))
        )
    )

    assert baseline_signals["entry_long"][entry_cursor, target_idx]

    cut_row = entry_cursor

    for mode in ("1e6", "nan"):
        mutated_panel = _mutate_panel_future(panel, cut_row, mode)
        with guards.strictness(guards.Strictness.FULL):
            mutated_signals = strategy.precompute(mutated_panel)

        for key in baseline_signals:
            _assert_equal_nan_aware(
                baseline_signals[key][: cut_row + 1],
                mutated_signals[key][: cut_row + 1],
            )

        assert (
            mutated_signals["entry_long"][entry_cursor, target_idx]
            == baseline_signals["entry_long"][entry_cursor, target_idx]
        )

    regular = _regular_times()
    session2_entry_cursor = int(panel.day_offsets[2]) + (regular.index("15:20") - 1)

    assert not baseline_signals["entry_long"][session2_entry_cursor, target_idx]
    assert not baseline_signals["entry_short"][session2_entry_cursor, target_idx]


def test_weights_respect_gross_and_max_weight() -> None:
    symbols = tuple(f"S{i}" for i in range(5))
    n_syms = len(symbols)
    view = _build_minimal_view(symbols)
    sigma = np.array([0.02, 0.03, 0.04, 0.05, 20.0], dtype=np.float64)
    signals_row = {
        "entry_long": np.ones(n_syms, dtype=bool),
        "entry_short": np.zeros(n_syms, dtype=bool),
        "exit_due": np.zeros(n_syms, dtype=bool),
        "sigma": sigma,
    }
    strategy = registry.build(
        {
            "strategy": "eod_overextension",
            "params": {"direction": "long", "max_weight": 0.1, "gross": 0.3},
        }
    )
    state = PortfolioState(
        shares=np.zeros(n_syms, dtype=np.float64),
        cash=100_000.0,
        equity=100_000.0,
        ts=0,
    )
    with guards.strictness(guards.Strictness.FULL):
        target = strategy.on_decision(view, signals_row, state)
    assert target is not None
    target.validate(n_syms)
    params = strategy.params
    assert float(np.sum(np.abs(target.weights))) <= params.gross + 1e-9
    assert np.all(np.abs(target.weights) <= params.max_weight + 1e-9)
