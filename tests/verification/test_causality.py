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
from nifty_quant.strategy.base import DataRequest, Strategy

_ROOT = Path(__file__).resolve().parents[2]
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_SYMBOLS: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(25))
_DEFAULT_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


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


def _session_specs(n_repeats: int) -> list[tuple[dt.date, list[str]]]:
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    base_specs = _default_session_specs()
    if n_repeats == 1:
        return base_specs

    specs = list(base_specs)
    last_date = base_specs[-1][0]

    for _ in range(1, n_repeats):
        for _, times in base_specs:
            next_date = last_date + dt.timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += dt.timedelta(days=1)
            specs.append((next_date, times))
            last_date = next_date

    return specs


def _random_walk_close(
    n_rows: int,
    n_symbols: int,
    rng: np.random.Generator,
    *,
    drift: float = 0.0,
) -> np.ndarray:
    returns = rng.normal(loc=drift, scale=0.002, size=(n_rows - 1, n_symbols))
    log_prices = np.empty((n_rows, n_symbols), dtype=np.float64)
    log_prices[0] = np.log(100.0)
    log_prices[1:] = log_prices[0] + np.cumsum(returns, axis=0)
    return np.exp(log_prices).astype(np.float32)


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


def build_irregular_panel(*, drift: float = 0.0, n_repeats: int = 1, seed: int = 20240102) -> Panel:
    specs = _session_specs(n_repeats)
    rng = np.random.default_rng(seed)
    n_rows = sum(len(times) for _, times in specs)
    n_symbols = len(_SYMBOLS)

    close = _random_walk_close(n_rows, n_symbols, rng, drift=drift)
    volume = np.exp(
        rng.normal(loc=11.0, scale=0.3, size=(n_rows, n_symbols))
    ).astype(np.float64)

    return _build_panel_from_close_and_volume(
        specs, _SYMBOLS, close, volume, rng
    )


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


def assert_precompute_is_causal(
    strategy: Strategy,
    panel: Panel,
    cut_row: int,
    *,
    fields: tuple[str, ...] = _DEFAULT_FIELDS,
) -> None:
    """Assert that precompute's outputs for rows <= cut_row do not depend on rows > cut_row."""
    n_rows = panel.n_rows()
    if not 0 <= cut_row <= n_rows - 2:
        raise ValueError(
            f"cut_row {cut_row} must satisfy 0 <= cut_row <= {n_rows - 2}"
        )

    def _run(p: Panel) -> dict[str, np.ndarray]:
        with guards.strictness(guards.Strictness.FULL):
            return dict(strategy.precompute(p))

    y0 = _run(panel)

    mutable_names: list[str] = []
    for field_name in fields:
        try:
            panel.field(field_name)
            mutable_names.append(field_name)
        except KeyError:
            continue

    for variant in ("1e6", "nan"):
        mutated_fields: dict[str, np.ndarray] = {}
        for field_name in mutable_names:
            arr = panel.field(field_name).copy()
            if variant == "1e6":
                arr[cut_row + 1 :] = arr[cut_row + 1 :] * 1e6
            else:
                arr[cut_row + 1 :] = np.nan
            mutated_fields[field_name] = arr

        mutated_panel = Panel(
            fields=mutated_fields,
            symbols=panel.symbols,
            ts=panel.ts,
            day_offsets=panel.day_offsets,
            dates=panel.dates,
        )
        y1 = _run(mutated_panel)

        assert y1.keys() == y0.keys(), (
            f"Strategy {strategy.name}, variant {variant}, cut_row {cut_row}: "
            f"precompute keys changed from {set(y0)} to {set(y1)}"
        )

        for key in y0:
            arr0 = y0[key]
            arr1 = y1[key]
            assert arr0.shape == arr1.shape, (
                f"Strategy {strategy.name}, variant {variant}, cut_row {cut_row}: "
                f"signal {key} shape {arr1.shape} != baseline {arr0.shape}"
            )

            prefix0 = arr0[: cut_row + 1]
            prefix1 = arr1[: cut_row + 1]
            nan0 = np.isnan(prefix0)
            nan1 = np.isnan(prefix1)

            if not np.array_equal(nan0, nan1):
                diff = nan0 != nan1
                if diff.ndim == 1:
                    bad_rows = np.flatnonzero(diff)
                else:
                    bad_rows = np.flatnonzero(np.any(diff, axis=1))
                first_row = int(bad_rows[0])
                raise AssertionError(
                    f"Strategy {strategy.name}, variant {variant}, "
                    f"cut_row {cut_row}, signal {key}: NaN mask mismatch; "
                    f"first mismatching row index {first_row}"
                )

            finite0 = prefix0[~nan0]
            finite1 = prefix1[~nan1]
            if not np.array_equal(finite0, finite1):
                eq = prefix0 == prefix1
                if eq.ndim == 1:
                    bad_rows = np.flatnonzero(~eq)
                else:
                    bad_rows = np.flatnonzero(~np.all(eq | nan0, axis=1))
                first_row = int(bad_rows[0])
                raise AssertionError(
                    f"Strategy {strategy.name}, variant {variant}, "
                    f"cut_row {cut_row}, signal {key}: finite-value mismatch; "
                    f"first mismatching row index {first_row}"
                )


def _cut_rows_to_sweep(strategy: Strategy, panel: Panel) -> list[int]:
    n_rows = panel.n_rows()
    cut_rows: set[int] = set()

    for day_idx in range(1, panel.n_days()):
        boundary = int(panel.day_offsets[day_idx])
        if 0 < boundary < n_rows - 1:
            cut_rows.add(boundary - 1)
            cut_rows.add(boundary)

    request: DataRequest = strategy.data_request()
    if request.decision_times is not None:
        for decision_time in request.decision_times:
            for row in panel.rows_at_time(decision_time):
                for delta in (1, 2):
                    candidate = int(row) - delta
                    if 0 <= candidate <= n_rows - 2:
                        cut_rows.add(candidate)

    rng = np.random.default_rng(12345)
    if n_rows > 1:
        n_random = min(8, n_rows - 1)
        candidates = rng.integers(0, n_rows - 1, size=n_random)
        for candidate in candidates:
            cut_rows.add(int(min(max(candidate, 0), n_rows - 2)))

    return sorted(cut_rows)


def test_fixture_is_driftless() -> None:
    panel = build_irregular_panel()
    log_returns = np.diff(
        np.log(panel.field("close").astype(np.float64)),
        axis=0,
    )
    mean = float(np.mean(log_returns))
    standard_error = 0.002 / np.sqrt(log_returns.size)
    # 4-sigma is generous enough not to be flaky for the fixed seed 20240102, while a
    # reintroduced drift of the previous magnitude (0.0002 is roughly 15x this standard
    # error over the full panel) would fail by a wide margin.
    assert abs(mean) < 4.0 * standard_error, (
        f"Expected |mean log return| < 4 * standard_error; "
        f"got mean={mean:.6g}, standard_error={standard_error:.6g}"
    )


@pytest.fixture(scope="module")
def irregular_panel() -> Panel:
    return build_irregular_panel()


@pytest.mark.parametrize("name", registry.available())
def test_precompute_is_causal_for_every_registered_strategy(
    name: str, irregular_panel: Panel
) -> None:
    strategy = _load_strategy(name)
    for cut_row in _cut_rows_to_sweep(strategy, irregular_panel):
        assert_precompute_is_causal(strategy, irregular_panel, cut_row)
