import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from nifty_quant.strategy import registry
from nifty_quant.strategy.base import (
    ArrayMarketView,
    DataRequest,
    LookaheadError,
    MarketView,
    PanelLike,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)


class EmptyParams(BaseModel):
    pass


class CausalRollingMeanParams(BaseModel):
    pass


class LeakyFullSampleMeanParams(BaseModel):
    pass


@dataclass
class PanelStub:
    symbols: tuple[str, ...]
    day_offsets: np.ndarray
    ts: np.ndarray
    field_arrays: Mapping[str, np.ndarray]
    minute_of_day_array: np.ndarray

    def field(self, name: str) -> np.ndarray:
        return self.field_arrays[name]

    def minute_of_day(self) -> np.ndarray:
        return self.minute_of_day_array


def _make_strategy_class(strategy_name: str) -> type[Strategy]:
    class _Registered(Strategy):
        name: ClassVar[str] = strategy_name
        Params: ClassVar[type[BaseModel]] = EmptyParams

        def data_request(self) -> DataRequest:
            return DataRequest()

        def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
            return {"close": panel.field("close")}

        def on_decision(
            self,
            view: MarketView,
            signals: Mapping[str, np.ndarray],
            state: PortfolioState,
        ) -> TargetPortfolio | None:
            return None

    return _Registered


def _make_view(
    close: np.ndarray,
    cursor: int,
    *,
    symbols: tuple[str, ...] = ("A", "B"),
    ts_array: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
    minute_of_day: np.ndarray | None = None,
    day_offsets: np.ndarray | None = None,
    session_date: date = date(2024, 1, 2),
) -> ArrayMarketView:
    n_rows = close.shape[0]

    if ts_array is None:
        ts_array = np.arange(n_rows, dtype=np.int64)
    if tradable is None:
        tradable = np.ones(len(symbols), dtype=bool)
    if minute_of_day is None:
        minute_of_day = np.arange(n_rows) % 375
    if day_offsets is None:
        day_offsets = np.array([0, n_rows], dtype=np.int64)

    return ArrayMarketView(
        panel_arrays={"close": close},
        cursor=cursor,
        symbols=symbols,
        ts_array=ts_array,
        tradable=tradable,
        session_date=session_date,
        minute_of_day=minute_of_day,
        day_offsets=day_offsets,
    )


class CausalRollingMean(Strategy):
    name: ClassVar[str] = "causal_rolling_mean_strategy_test"
    Params: ClassVar[type[BaseModel]] = CausalRollingMeanParams

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        close = panel.field("close")
        window = 5

        csum = np.cumsum(close, axis=0)
        csum_shift = np.zeros_like(csum)
        if close.shape[0] > window:
            csum_shift[window:] = csum[:-window]

        counts = np.minimum(window, np.arange(1, close.shape[0] + 1))[:, None].astype(
            close.dtype
        )

        return {"rolling_mean_5": (csum - csum_shift) / counts}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return None


class LeakyFullSampleMean(Strategy):
    name: ClassVar[str] = "leaky_full_sample_mean_strategy_test"
    Params: ClassVar[type[BaseModel]] = LeakyFullSampleMeanParams

    def data_request(self) -> DataRequest:
        return DataRequest()

    def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
        x = panel.field("close")
        return {"centered": x - np.nanmean(x, axis=0)}

    def on_decision(
        self,
        view: MarketView,
        signals: Mapping[str, np.ndarray],
        state: PortfolioState,
    ) -> TargetPortfolio | None:
        return None


def test_last_rejects_negative_offset_and_returns_cursor_row() -> None:
    close = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    view = _make_view(close, cursor=2)

    with pytest.raises(ValueError):
        view.last("close", -1)

    np.testing.assert_array_equal(view.last("close", 0), close[2])


def test_window_returns_exactly_n_rows_ending_at_cursor() -> None:
    close = np.arange(12, dtype=float).reshape(6, 2)
    view = _make_view(close, cursor=4)

    result = view.window("close", 3)

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, close[2:5])
    np.testing.assert_array_equal(result[-1], view.last("close", 0))


def test_at_time_raises_for_cursor_or_future_but_returns_earlier() -> None:
    close = np.arange(12, dtype=float).reshape(6, 2)
    minute_of_day = np.array([555, 560, 565, 555, 560, 565], dtype=np.int64)
    day_offsets = np.array([0, 3, 6], dtype=np.int64)

    view = _make_view(
        close,
        cursor=4,
        minute_of_day=minute_of_day,
        day_offsets=day_offsets,
    )

    np.testing.assert_array_equal(view.at_time("close", "09:15"), close[3])

    with pytest.raises(LookaheadError):
        view.at_time("close", "09:20")

    with pytest.raises(LookaheadError):
        view.at_time("close", "09:25")


def test_future_row_scrambling_detects_precompute_leak_but_not_causal() -> None:
    rng = np.random.default_rng(2024_05_20)
    n_rows, n_symbols, k = 30, 4, 15

    close0 = rng.normal(size=(n_rows, n_symbols))
    close1 = close0.copy()
    close1[k + 1 :] = rng.normal(
        loc=100.0,
        scale=1.0,
        size=(n_rows - k - 1, n_symbols),
    )

    def make_panel(close: np.ndarray) -> PanelStub:
        return PanelStub(
            symbols=("A", "B", "C", "D"),
            day_offsets=np.array([0, n_rows], dtype=np.int64),
            ts=np.arange(n_rows, dtype=np.int64),
            field_arrays={"close": close},
            minute_of_day_array=np.arange(n_rows) % 375,
        )

    panel0 = make_panel(close0)
    panel1 = make_panel(close1)

    causal = CausalRollingMean(params=CausalRollingMeanParams())
    y0 = causal.precompute(panel0)
    y1 = causal.precompute(panel1)

    for key in y0:
        assert np.array_equal(
            y0[key][: k + 1],
            y1[key][: k + 1],
            equal_nan=True,
        )

    leaky = LeakyFullSampleMean(params=LeakyFullSampleMeanParams())
    y0_leak = leaky.precompute(panel0)
    y1_leak = leaky.precompute(panel1)

    for key in y0_leak:
        assert not np.array_equal(
            y0_leak[key][: k + 1],
            y1_leak[key][: k + 1],
            equal_nan=True,
        )


def test_registry_rejects_duplicate_names_lists_and_get_unknown() -> None:
    name = f"repeated_{uuid.uuid4().hex[:8]}"
    class_a = _make_strategy_class(name)
    class_b = _make_strategy_class(name)

    registry.register(class_a)

    with pytest.raises(ValueError):
        registry.register(class_b)

    assert name in registry.available()
    assert registry.get(name) is class_a

    with pytest.raises(KeyError):
        registry.get(f"nope_{uuid.uuid4().hex[:8]}")


def test_config_hash_stable_unique_and_hex_length() -> None:
    cfg1 = {
        "strategy": "hash_sentinel",
        "params": {
            "a": [1, 2],
            "b": {"c": 3, "d": [4, 5]},
            "e": None,
        },
    }
    cfg2 = {
        "params": {
            "e": None,
            "b": {"d": [4, 5], "c": 3},
            "a": [1, 2],
        },
        "strategy": "hash_sentinel",
    }

    hash1 = registry.config_hash(cfg1)
    hash2 = registry.config_hash(cfg2)

    assert hash1 == hash2
    assert len(hash1) == 16
    assert all(char in "0123456789abcdef" for char in hash1)

    cfg3 = {
        "strategy": "hash_sentinel",
        "params": {
            "a": [1, 2],
            "b": {"c": 4, "d": [4, 5]},
            "e": None,
        },
    }
    assert registry.config_hash(cfg3) != hash1


def test_target_portfolio_validate_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        TargetPortfolio(weights=np.array([0.5, 0.5])).validate(3)

    with pytest.raises(ValueError):
        TargetPortfolio(weights=np.array([0.5, np.nan, 0.0])).validate(3)

    with pytest.raises(ValueError):
        TargetPortfolio(weights=np.array([0.5, np.inf, 0.0])).validate(3)

    with pytest.raises(ValueError):
        TargetPortfolio(weights=np.array([0.5, 0.6, 0.0])).validate(3)


def test_registry_build_configures_instance_and_rejects_invalid_params() -> None:
    strategy_name = f"build_strategy_{uuid.uuid4().hex[:8]}"

    class BuildParams(BaseModel):
        lookback: int

    class BuildStrategy(Strategy):
        name: ClassVar[str] = strategy_name
        Params: ClassVar[type[BaseModel]] = BuildParams

        def data_request(self) -> DataRequest:
            return DataRequest()

        def precompute(self, panel: PanelLike) -> dict[str, np.ndarray]:
            return {"close": panel.field("close")}

        def on_decision(
            self,
            view: MarketView,
            signals: Mapping[str, np.ndarray],
            state: PortfolioState,
        ) -> TargetPortfolio | None:
            return None

    registry.register(BuildStrategy)

    built = registry.build(
        {"strategy": strategy_name, "params": {"lookback": 42}}
    )
    assert isinstance(built, BuildStrategy)
    assert built.params == BuildParams(lookback=42)

    with pytest.raises(ValidationError):
        registry.build(
            {"strategy": strategy_name, "params": {"lookback": "not-an-int"}}
        )


def test_data_request_warmup_bars_takes_max_of_lookbacks() -> None:
    bars_dominate = DataRequest(lookback_bars=500, lookback_days=1)
    assert bars_dominate.warmup_bars() == 500

    days_dominate = DataRequest(lookback_bars=100, lookback_days=3)
    assert days_dominate.warmup_bars() == 1125


def test_day_index_canonical_offsets_with_short_muhurat_session() -> None:
    close = np.zeros((810, 2), dtype=float)
    view = _make_view(
        close,
        cursor=0,
        day_offsets=np.array([0, 375, 435, 810], dtype=np.int64),
    )

    expected = np.concatenate(
        [
            np.zeros(375, dtype=np.int64),
            np.ones(60, dtype=np.int64),
            np.full(375, 2, dtype=np.int64),
        ]
    )

    assert view._day_index[0] == 0
    assert view._day_index[374] == 0
    assert view._day_index[375] == 1
    assert view._day_index[434] == 1
    assert view._day_index[435] == 2
    assert view._day_index[809] == 2

    np.testing.assert_array_equal(view._day_index, expected)


def test_bad_day_offsets_shape_is_rejected() -> None:
    close = np.zeros((810, 2), dtype=float)
    bad_offsets = np.arange(810, dtype=np.int64)

    with pytest.raises(ValueError):
        _make_view(close, cursor=0, day_offsets=bad_offsets)
