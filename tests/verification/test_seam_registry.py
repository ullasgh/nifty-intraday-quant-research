from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml

import nifty_quant.strategy.plugins  # noqa: F401
from nifty_quant import guards
from nifty_quant.data.panel import Panel
from nifty_quant.strategy import registry
from nifty_quant.strategy.base import (
    ArrayMarketView,
    DataRequest,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)

_ROOT = Path(__file__).resolve().parents[2]
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_SYMBOLS: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(25))
_REGISTERED_STRATEGY_NAMES: tuple[str, ...] = tuple(registry.available())


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
    times: list[str] = []
    current = 9 * 60 + 15
    end = 15 * 60 + 29
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _short_dr_times() -> list[str]:
    times: list[str] = []
    current = 9 * 60 + 15
    end = 10 * 60 + 59
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _muhurat_times() -> list[str]:
    times: list[str] = []
    current = 18 * 60
    end = 18 * 60 + 59
    while current <= end:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += 1
    return times


def _default_session_specs() -> list[tuple[dt.date, list[str]]]:
    return [
        (dt.date(2024, 1, 2), _regular_times()),
        (dt.date(2024, 1, 3), _short_dr_times()),
        (dt.date(2024, 1, 4), _muhurat_times()),
        (dt.date(2024, 1, 5), _regular_times()),
    ]


def _random_walk_close(
    n_rows: int, n_symbols: int, rng: np.random.Generator
) -> np.ndarray:
    returns = rng.normal(loc=0.0002, scale=0.002, size=(n_rows - 1, n_symbols))
    log_prices = np.empty((n_rows, n_symbols), dtype=np.float64)
    log_prices[0] = np.log(100.0)
    log_prices[1:] = log_prices[0] + np.cumsum(returns, axis=0)
    return np.exp(log_prices).astype(np.float64)


def _build_panel_from_close_and_volume(
    specs: list[tuple[dt.date, list[str]]],
    symbols: tuple[str, ...],
    close: np.ndarray,
    volume: np.ndarray,
    rng: np.random.Generator,
) -> Panel:
    ts_list: list[int] = []
    day_offsets: list[int] = [0]
    dates_list: list[dt.date] = []
    for session_date, times in specs:
        for hhmm in times:
            ts_list.append(_make_ts(session_date, hhmm))
        day_offsets.append(len(ts_list))
        dates_list.append(session_date)

    open_array = np.empty_like(close)
    open_array[0] = 100.0
    open_array[1:] = close[:-1]

    wick = 0.01 * close + rng.uniform(
        0.0005, 0.002, size=close.shape
    ).astype(close.dtype)
    high = np.maximum(open_array, close) + wick
    low = np.minimum(open_array, close) - wick
    low = np.maximum(low, 0.01)

    field_arrays = {
        "open": open_array.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    return Panel(
        fields=field_arrays,
        symbols=symbols,
        ts=np.array(ts_list, dtype=np.int64),
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=np.array(dates_list, dtype=object),
    )


def build_irregular_panel() -> Panel:
    specs = _default_session_specs()
    rng = np.random.default_rng(20240102)
    n_rows = sum(len(times) for _, times in specs)
    n_symbols = len(_SYMBOLS)

    close = _random_walk_close(n_rows, n_symbols, rng)
    volume = np.exp(
        rng.normal(loc=11.0, scale=0.3, size=(n_rows, n_symbols))
    ).astype(np.float64)

    return _build_panel_from_close_and_volume(
        specs, _SYMBOLS, close, volume, rng
    )


def _build_alignment_panel(
    symbols: tuple[str, ...], outlier_index: int
) -> tuple[Panel, int]:
    base_date = dt.date(2024, 2, 5)
    specs: list[tuple[dt.date, list[str]]] = []
    for idx in range(10):
        specs.append(
            (base_date + dt.timedelta(days=idx * 7), _regular_times())
        )
    specs.append((base_date + dt.timedelta(days=70), _muhurat_times()))
    specs.append((base_date + dt.timedelta(days=77), _short_dr_times()))

    target_times = specs[4][1]
    rows_before_target = sum(len(times) for _, times in specs[:4])
    target_row = rows_before_target + target_times.index("11:00")

    n_rows = sum(len(times) for _, times in specs)
    close = np.full((n_rows, len(symbols)), 100.0, dtype=np.float64)
    close[:, outlier_index] = 100.0 * np.exp(
        np.arange(n_rows, dtype=np.float64) * 0.0005
    )

    volume = np.full((n_rows, len(symbols)), 100_000.0, dtype=np.float64)
    volume[target_row, outlier_index] = 1_000_000.0
    volume[target_row - 1, outlier_index] = 1_000_000.0

    rng = np.random.default_rng(5678)
    panel = _build_panel_from_close_and_volume(
        specs, symbols, close, volume, rng
    )
    return panel, target_row


def _load_strategy(name: str) -> Strategy:
    config_path = _ROOT / "configs" / "strategies" / f"{name}.yaml"
    strategy_cls = registry.get(name)

    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        if not isinstance(cfg, Mapping):
            pytest.fail(
                f"Strategy {name}: config file {config_path} is not a mapping"
            )
        return registry.build(cfg)

    try:
        params = strategy_cls.Params()
    except Exception as exc:
        pytest.fail(
            f"Strategy {name}: cannot construct default Params(): {exc}"
        )
    return strategy_cls(params=params)


def _pick_valid_cursor(panel: Panel) -> int:
    rows = panel.rows_at_time("09:20")
    if rows.size > 0:
        return int(rows[0])
    if panel.n_rows() > 2:
        return 1
    return int(panel.n_rows() - 2)


def _day_index_for_cursor(panel: Panel, cursor: int) -> int:
    day_idx = int(
        np.searchsorted(panel.day_offsets, cursor, side="right") - 1
    )
    return int(min(max(day_idx, 0), panel.n_days() - 1))


@pytest.fixture(scope="module")
def irregular_panel() -> Panel:
    return build_irregular_panel()


@pytest.mark.parametrize("name", _REGISTERED_STRATEGY_NAMES)
def test_plugin_engine_seam(name: str, irregular_panel: Panel) -> None:
    strategy = _load_strategy(name)

    request = strategy.data_request()
    assert isinstance(request, DataRequest)

    with guards.strictness(guards.Strictness.FULL):
        signals_full = strategy.precompute(irregular_panel)

    signal_dict = dict(signals_full)
    expected_shape = (irregular_panel.n_rows(), irregular_panel.n_symbols())
    for key, arr in signal_dict.items():
        assert isinstance(arr, np.ndarray), (
            f"Strategy {name}: signal {key} is not an ndarray"
        )
        assert arr.shape == expected_shape, (
            f"Strategy {name}: signal {key} shape {arr.shape} != "
            f"expected {expected_shape}"
        )

    cursor = _pick_valid_cursor(irregular_panel)
    panel_arrays: dict[str, np.ndarray] = {
        field_name: irregular_panel.field(field_name)
        for field_name in dict.fromkeys(
            (*request.fields, *request.extra_series)
        )
    }
    tradable = np.ones(irregular_panel.n_symbols(), dtype=bool)
    view = ArrayMarketView(
        panel_arrays,
        cursor,
        symbols=irregular_panel.symbols,
        ts_array=irregular_panel.ts,
        tradable=tradable,
        session_date=irregular_panel.dates[
            _day_index_for_cursor(irregular_panel, cursor)
        ],
        minute_of_day=irregular_panel.minute_of_day(),
        day_offsets=irregular_panel.day_offsets,
    )

    signals_row = {
        key: arr[cursor] for key, arr in signal_dict.items()
    }
    for key, arr in signals_row.items():
        assert arr.shape == (irregular_panel.n_symbols(),), (
            f"Strategy {name}: sliced signal {key} shape {arr.shape} != "
            f"{(irregular_panel.n_symbols(),)}"
        )

    state = PortfolioState(
        shares=np.zeros(irregular_panel.n_symbols(), dtype=np.float64),
        cash=1e7,
        equity=1e7,
        ts=int(irregular_panel.ts[cursor]),
    )

    with guards.strictness(guards.Strictness.FULL):
        target = strategy.on_decision(view, signals_row, state)

    if target is not None:
        assert isinstance(target, TargetPortfolio)
        assert target.weights.shape == (irregular_panel.n_symbols(),)
        target.validate(irregular_panel.n_symbols())


def test_outlier_alignment_fires_only_for_outlier_symbol() -> None:
    outlier_index = 7
    panel, target_row = _build_alignment_panel(
        _SYMBOLS, outlier_index
    )
    cursor = target_row - 1

    any_fired = False
    for name in _REGISTERED_STRATEGY_NAMES:
        strategy = _load_strategy(name)
        request = strategy.data_request()

        with guards.strictness(guards.Strictness.FULL):
            signals_full = strategy.precompute(panel)

        signal_dict = dict(signals_full)
        panel_arrays: dict[str, np.ndarray] = {
            field_name: panel.field(field_name)
            for field_name in dict.fromkeys(
                (*request.fields, *request.extra_series)
            )
        }

        tradable = np.ones(panel.n_symbols(), dtype=bool)
        view = ArrayMarketView(
            panel_arrays,
            cursor,
            symbols=panel.symbols,
            ts_array=panel.ts,
            tradable=tradable,
            session_date=panel.dates[_day_index_for_cursor(panel, cursor)],
            minute_of_day=panel.minute_of_day(),
            day_offsets=panel.day_offsets,
        )
        signals_row = {
            key: arr[cursor] for key, arr in signal_dict.items()
        }
        state = PortfolioState(
            shares=np.zeros(panel.n_symbols(), dtype=np.float64),
            cash=1e7,
            equity=1e7,
            ts=int(panel.ts[cursor]),
        )

        with guards.strictness(guards.Strictness.FULL):
            target = strategy.on_decision(view, signals_row, state)

        if target is None:
            continue

        weights = target.weights
        if np.any(weights != 0.0):
            any_fired = True
            nonzero = np.flatnonzero(weights != 0.0)
            assert set(nonzero.tolist()) == {outlier_index}, (
                f"Strategy {name} produced nonzero weights at indices "
                f"{nonzero.tolist()}; expected only outlier index "
                f"{outlier_index}"
            )

    assert any_fired, (
        "No registered strategy produced a nonzero weight on the outlier "
        "alignment panel; the seam alignment check exercised no real signal"
    )
