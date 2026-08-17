from __future__ import annotations

import datetime
from typing import Callable, Mapping

import numpy as np
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    build_overnight_feature,
    run_h2,
)
from nifty_quant.research.lens import Feature, FeatureKindError, HypothesisVerdict, Lens

_IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
_EPOCH = datetime.datetime(1970, 1, 1)


def _epoch(day: datetime.date, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(day.year, day.month, day.day, hour, minute)
    return int((local - _IST_OFFSET - _EPOCH).total_seconds())


def build_h2_panel(
    sessions: Mapping[
        datetime.date,
        Mapping[str, Mapping[str, Mapping[str, float]]],
    ],
    symbols: tuple[str, ...],
) -> Panel:
    """Build a float32-at-rest H2 panel from per-symbol labelled checkpoint bars."""
    all_dates = sorted(sessions)
    rows: list[tuple[int, datetime.date, Mapping[str, Mapping[str, float]]]] = []
    for day in all_dates:
        for hhmm, symbol_values in sessions[day].items():
            rows.append((_epoch(day, hhmm), day, symbol_values))
    rows.sort(key=lambda row: row[0])

    field_names = {"open", "close", "volume"}
    for _, _, symbol_values in rows:
        for values in symbol_values.values():
            field_names.update(values)

    n_rows = len(rows)
    n_symbols = len(symbols)
    fields = {
        name: np.full((n_rows, n_symbols), np.nan, dtype=np.float32)
        for name in field_names
    }
    for row_ix, (_, _, symbol_values) in enumerate(rows):
        for symbol_ix, symbol in enumerate(symbols):
            values = symbol_values.get(symbol, {})
            for name, value in values.items():
                fields[name][row_ix, symbol_ix] = np.float32(value)
            if "volume" not in values:
                fields["volume"][row_ix, symbol_ix] = np.float32(1_000.0 + symbol_ix)

    offsets = [0]
    for day in all_dates:
        offsets.append(offsets[-1] + sum(row_day == day for _, row_day, _ in rows))

    return Panel(
        fields=fields,
        symbols=symbols,
        ts=np.asarray([row[0] for row in rows], dtype=np.int64),
        day_offsets=np.asarray(offsets, dtype=np.int32),
        dates=np.asarray(all_dates, dtype=object),
    )


def _session_rows(panel: Panel, day: datetime.date) -> np.ndarray:
    day_ix = next(ix for ix, value in enumerate(panel.dates) if value == day)
    return np.arange(panel.day_offsets[day_ix], panel.day_offsets[day_ix + 1])


def _feature_session_values(
    feature: Feature,
    panel: Panel,
    day: datetime.date,
    symbol: str,
) -> np.ndarray:
    return feature.values[_session_rows(panel, day), panel.sym_ix[symbol]]


def _clone_sessions(
    sessions: Mapping[
        datetime.date,
        Mapping[str, Mapping[str, Mapping[str, float]]],
    ],
) -> dict[datetime.date, dict[str, dict[str, dict[str, float]]]]:
    return {
        day: {
            hhmm: {symbol: dict(values) for symbol, values in symbol_values.items()}
            for hhmm, symbol_values in day_values.items()
        }
        for day, day_values in sessions.items()
    }


def _business_dates(start: datetime.date, count: int) -> list[datetime.date]:
    dates: list[datetime.date] = []
    day = start
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day)
        day += datetime.timedelta(days=1)
    return dates


def _signal_panel(
    dates: list[datetime.date],
    *,
    symbols: tuple[str, ...],
    relation: float | Callable[[datetime.date, int], float],
    seed: int,
    overnight_scale: float = 0.01,
    noise_scale: float = 0.0002,
) -> Panel:
    rng = np.random.default_rng(seed)
    prior_close = np.asarray([100.0 + 2.0 * ix for ix in range(len(symbols))])
    sessions: dict[datetime.date, dict[str, dict[str, dict[str, float]]]] = {}

    for day_ix, day in enumerate(dates):
        overnight = rng.normal(0.0, overnight_scale, size=len(symbols))
        noise = rng.normal(0.0, noise_scale, size=len(symbols))
        day_open = prior_close * np.exp(overnight)
        day_relation = (
            relation(day, day_ix) if callable(relation) else relation
        )
        intraday = day_relation * overnight + noise
        day_close = day_open * np.exp(intraday)

        open_bars: dict[str, dict[str, float]] = {}
        close_bars: dict[str, dict[str, float]] = {}
        for symbol_ix, symbol in enumerate(symbols):
            volume = 10_000.0 + 1_000.0 * symbol_ix + 17.0 * day_ix
            open_bars[symbol] = {
                "open": float(day_open[symbol_ix]),
                "close": float(day_open[symbol_ix]),
                "volume": volume,
            }
            close_bars[symbol] = {
                "open": float(day_close[symbol_ix]),
                "close": float(day_close[symbol_ix]),
                "volume": volume + 3.0,
            }

        sessions[day] = {"09:16": open_bars, "15:20": close_bars}
        prior_close = day_close

    return build_h2_panel(sessions, symbols)


def test_round_trip_cost_is_resolved_from_the_cost_model() -> None:
    costs = NSEIntradayEquityCosts()
    round_trip = costs.round_trip_bps(1e5)

    assert round_trip == pytest.approx(8.26452, abs=1e-5)
    assert 2.0 * round_trip == pytest.approx(16.52904, abs=1e-5)


def test_overnight_return_uses_previous_close_and_current_0916_open() -> None:
    symbols = ("S00",)
    dates = (datetime.date(2024, 1, 4), datetime.date(2024, 1, 5))
    sessions = {
        dates[0]: {
            "09:15": {"S00": {"open": 90.0, "close": 91.0, "high": 92.0}},
            "09:16": {"S00": {"open": 90.0, "close": 91.0}},
            "12:00": {"S00": {"open": 95.0, "close": 96.0}},
            "15:20": {"S00": {"open": 99.0, "close": 100.0}},
        },
        dates[1]: {
            "09:15": {"S00": {"open": 108.0, "close": 109.0, "high": 110.0}},
            "09:16": {"S00": {"open": 110.0, "close": 111.0}},
            "12:00": {"S00": {"open": 112.0, "close": 113.0}},
            "15:20": {"S00": {"open": 114.0, "close": 115.0}},
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    expected = np.log(
        np.float64(np.float32(110.0)) / np.float64(np.float32(100.0))
    )
    values = _feature_session_values(feature, panel, dates[1], "S00")

    assert values == pytest.approx(expected, abs=1e-5)


def test_corrupting_0915_does_not_change_the_overnight_feature() -> None:
    symbols = ("S00", "S01")
    dates = (datetime.date(2024, 2, 1), datetime.date(2024, 2, 2))
    sessions = {
        dates[0]: {
            "09:15": {
                "S00": {"open": 90.0, "close": 91.0, "high": 92.0},
                "S01": {"open": 91.0, "close": 92.0, "high": 93.0},
            },
            "09:16": {
                "S00": {"open": 90.0, "close": 91.0},
                "S01": {"open": 91.0, "close": 92.0},
            },
            "12:00": {
                "S00": {"open": 95.0, "close": 96.0},
                "S01": {"open": 96.0, "close": 97.0},
            },
            "15:20": {
                "S00": {"open": 99.0, "close": 100.0},
                "S01": {"open": 100.0, "close": 101.0},
            },
        },
        dates[1]: {
            "09:15": {
                "S00": {"open": 108.0, "close": 109.0, "high": 110.0},
                "S01": {"open": 109.0, "close": 110.0, "high": 111.0},
            },
            "09:16": {
                "S00": {"open": 110.0, "close": 111.0},
                "S01": {"open": 111.0, "close": 112.0},
            },
            "12:00": {
                "S00": {"open": 112.0, "close": 113.0},
                "S01": {"open": 113.0, "close": 114.0},
            },
            "15:20": {
                "S00": {"open": 114.0, "close": 115.0},
                "S01": {"open": 115.0, "close": 116.0},
            },
        },
    }
    clean_panel = build_h2_panel(sessions, symbols)
    corrupted = _clone_sessions(sessions)
    for day in dates:
        for symbol in symbols:
            corrupted[day]["09:15"][symbol]["close"] = 10_000.0
            corrupted[day]["09:15"][symbol]["high"] = 1.0
    corrupted_panel = build_h2_panel(corrupted, symbols)

    clean = build_overnight_feature(clean_panel)
    changed = build_overnight_feature(corrupted_panel)

    np.testing.assert_array_equal(clean.values, changed.values)


def test_first_session_has_nan_overnight_return_for_every_symbol() -> None:
    symbols = ("S00", "S01", "S02")
    dates = (datetime.date(2024, 3, 1), datetime.date(2024, 3, 4))
    sessions = {
        dates[0]: {
            "09:16": {
                symbol: {"open": 100.0 + ix, "close": 100.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {"open": 101.0 + ix, "close": 101.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
        dates[1]: {
            "09:16": {
                symbol: {"open": 103.0 + ix, "close": 103.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {"open": 104.0 + ix, "close": 104.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    first = feature.values[_session_rows(panel, dates[0])]
    second = feature.values[_session_rows(panel, dates[1])]

    assert np.isnan(first).all()
    assert np.isfinite(second).all()


def test_previous_usable_session_bridges_a_weekend_gap() -> None:
    symbols = ("S00",)
    friday = datetime.date(2024, 1, 5)
    monday = datetime.date(2024, 1, 8)
    sessions = {
        friday: {
            "09:16": {"S00": {"open": 98.0, "close": 98.0}},
            "15:20": {"S00": {"open": 100.0, "close": 100.0}},
        },
        monday: {
            "09:16": {"S00": {"open": 110.0, "close": 110.0}},
            "15:20": {"S00": {"open": 111.0, "close": 111.0}},
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    expected = np.log(
        np.float64(np.float32(110.0)) / np.float64(np.float32(100.0))
    )
    assert _feature_session_values(feature, panel, monday, "S00") == pytest.approx(
        expected,
        abs=1e-5,
    )


def test_missing_muhurat_close_does_not_reach_back_to_an_older_session() -> None:
    symbols = ("S00",)
    dates = (
        datetime.date(2024, 11, 1),
        datetime.date(2024, 11, 2),
        datetime.date(2024, 11, 4),
    )
    sessions = {
        dates[0]: {
            "09:16": {"S00": {"open": 98.0, "close": 98.0}},
            "15:20": {"S00": {"open": 100.0, "close": 100.0}},
        },
        dates[1]: {
            "09:16": {"S00": {"open": 105.0, "close": 105.0}},
            "12:00": {"S00": {"open": 106.0, "close": 106.0}},
        },
        dates[2]: {
            "09:16": {"S00": {"open": 110.0, "close": 110.0}},
            "15:20": {"S00": {"open": 111.0, "close": 111.0}},
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    middle = _feature_session_values(feature, panel, dates[1], "S00")
    next_day = _feature_session_values(feature, panel, dates[2], "S00")

    assert np.isfinite(middle).all()
    assert np.isnan(next_day).all()


def test_overnight_return_is_resolved_per_symbol_when_one_symbol_has_a_gap() -> None:
    symbols = ("S00", "S01")
    dates = (datetime.date(2024, 4, 1), datetime.date(2024, 4, 2))
    sessions = {
        dates[0]: {
            "09:16": {
                "S00": {"open": 98.0, "close": 98.0},
                "S01": {"open": np.nan, "close": np.nan},
            },
            "15:20": {
                "S00": {"open": 100.0, "close": 100.0},
                "S01": {"open": np.nan, "close": np.nan},
            },
        },
        dates[1]: {
            "09:16": {
                "S00": {"open": 110.0, "close": 110.0},
                "S01": {"open": 120.0, "close": 120.0},
            },
            "15:20": {
                "S00": {"open": 111.0, "close": 111.0},
                "S01": {"open": 121.0, "close": 121.0},
            },
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    values = feature.values[_session_rows(panel, dates[1])]
    assert np.isfinite(values[:, panel.sym_ix["S00"]]).all()
    assert np.isnan(values[:, panel.sym_ix["S01"]]).all()


def test_nan_does_not_forward_fill_across_sessions_or_symbols() -> None:
    symbols = ("S00", "S01")
    dates = (
        datetime.date(2024, 5, 1),
        datetime.date(2024, 5, 2),
        datetime.date(2024, 5, 3),
    )
    sessions = {
        dates[0]: {
            "09:16": {
                "S00": {"open": 98.0, "close": 98.0},
                "S01": {"open": 98.0, "close": 98.0},
            },
            "15:20": {
                "S00": {"open": np.nan, "close": np.nan},
                "S01": {"open": 100.0, "close": 100.0},
            },
        },
        dates[1]: {
            "09:16": {
                "S00": {"open": 110.0, "close": 110.0},
                "S01": {"open": np.nan, "close": np.nan},
            },
            "15:20": {
                "S00": {"open": 105.0, "close": 105.0},
                "S01": {"open": 105.0, "close": 105.0},
            },
        },
        dates[2]: {
            "09:16": {
                "S00": {"open": 112.0, "close": 112.0},
                "S01": {"open": 110.0, "close": 110.0},
            },
            "15:20": {
                "S00": {"open": 113.0, "close": 113.0},
                "S01": {"open": 111.0, "close": 111.0},
            },
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    second = feature.values[_session_rows(panel, dates[1])]
    third = feature.values[_session_rows(panel, dates[2])]

    assert np.isnan(second[:, panel.sym_ix["S00"]]).all()
    assert np.isnan(second[:, panel.sym_ix["S01"]]).all()
    assert np.isfinite(third).all()


def test_overnight_feature_is_broadcast_to_every_row_and_has_return_kind() -> None:
    symbols = ("S00", "S01")
    day_one = datetime.date(2024, 6, 3)
    day_two = datetime.date(2024, 6, 4)
    sessions = {
        day_one: {
            "09:15": {
                symbol: {"open": 90.0 + ix, "close": 91.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "09:16": {
                symbol: {"open": 90.0 + ix, "close": 91.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "12:00": {
                symbol: {"open": 95.0 + ix, "close": 96.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {"open": 99.0 + ix, "close": 100.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
        day_two: {
            "09:15": {
                symbol: {"open": 108.0 + ix, "close": 109.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "09:16": {
                symbol: {"open": 110.0 + ix, "close": 111.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "12:00": {
                symbol: {"open": 112.0 + ix, "close": 113.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {"open": 114.0 + ix, "close": 115.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    day_values = feature.values[_session_rows(panel, day_two)]
    for symbol_ix in range(len(symbols)):
        assert np.all(day_values[:, symbol_ix] == day_values[0, symbol_ix])
    assert feature.kind == "return"


def test_overnight_feature_has_panel_shape_and_float64_values() -> None:
    symbols = ("S00", "S01", "S02")
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 7, 1), 4),
        symbols=symbols,
        relation=-1.0,
        seed=11,
    )
    feature = build_overnight_feature(panel)

    assert feature.values.shape == (panel.n_rows(), panel.n_symbols())
    assert feature.values.dtype == np.float64


def test_reversal_relationship_produces_a_negative_top_minus_bottom_spread() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 45),
        symbols=symbols,
        relation=-1.4,
        seed=12,
        overnight_scale=0.01,
        noise_scale=0.00015,
    )
    verdict = run_h2(panel, seed=12)

    assert isinstance(verdict, HypothesisVerdict)
    assert verdict.expectancy.spread_bps < 0.0
    assert verdict.expectancy.spread_t < -1.96
    assert "negative" in verdict.explain().lower()


def test_momentum_relationship_is_positive_and_reported_as_momentum() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 45),
        symbols=symbols,
        relation=1.4,
        seed=13,
        overnight_scale=0.01,
        noise_scale=0.00015,
    )
    verdict = run_h2(panel, seed=13)

    assert verdict.expectancy.spread_bps > 0.0
    assert verdict.expectancy.spread_t > 1.96
    explanation = verdict.explain().lower()
    assert "positive" in explanation
    assert "momentum" in explanation


def test_verified_driftless_pure_noise_is_not_significant_or_cost_surviving() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 70),
        symbols=symbols,
        relation=0.0,
        seed=4,
        overnight_scale=0.01,
        noise_scale=0.003,
    )
    verdict = run_h2(panel, seed=4)

    # Seed 4 is checked for this exact construction: the driftless draw is not an
    # unlucky significant realization before the closed-survival assertion below.
    assert np.isfinite(verdict.expectancy.spread_bps)
    assert abs(verdict.expectancy.spread_bps) < verdict.cost_hurdle_bps
    assert abs(verdict.expectancy.spread_t) < 1.96
    assert verdict.expectancy.survives_costs is False
    assert verdict.survived is False


def test_statistically_significant_reversal_below_model_cost_hurdle_is_killed() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 90),
        symbols=symbols,
        relation=-0.02,
        seed=14,
        overnight_scale=0.0005,
        noise_scale=0.0000001,
    )
    verdict = run_h2(panel, seed=14)
    hurdle = 2.0 * NSEIntradayEquityCosts().round_trip_bps(1e5)

    assert abs(verdict.expectancy.spread_t) > 1.96
    assert abs(verdict.expectancy.spread_bps) < hurdle
    assert verdict.expectancy.survives_costs is False
    assert verdict.survived is False


def test_run_h2_returns_all_six_criteria_after_a_failure() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 35),
        symbols=symbols,
        relation=0.0,
        seed=15,
        noise_scale=0.0001,
    )
    verdict = run_h2(panel, seed=15)

    assert isinstance(verdict, HypothesisVerdict)
    assert len(verdict.reasons) >= 6
    for criterion_ix in range(1, 7):
        assert verdict.reasons[criterion_ix - 1].startswith(f"{criterion_ix}.")
    assert "FAIL" in verdict.reasons[0]


def test_latency_criterion_is_not_evaluated_without_a_latency_profile() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 20),
        symbols=symbols,
        relation=-1.0,
        seed=16,
    )
    verdict = run_h2(panel, seed=16)

    assert "NOT_EVALUATED" in verdict.reasons[4]
    assert "PASS" not in verdict.reasons[4]


def test_run_h2_expectancy_matches_direct_cross_sectional_lens_expectancy() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 30),
        symbols=symbols,
        relation=-1.0,
        seed=17,
    )
    seed = 17
    verdict = run_h2(panel, seed=seed)
    feature = build_overnight_feature(panel)

    # H2 must explicitly select cross_sectional_rank; expanding quantiles cannot
    # bucket this once-per-session signal in a small fixture.
    direct = Lens(panel, seed=seed).expectancy(
        feature,
        horizon=1,
        method="cross_sectional_rank",
    )
    actual = verdict.expectancy

    assert actual.spread_bps == pytest.approx(direct.spread_bps, abs=1e-8)
    assert actual.spread_t == pytest.approx(direct.spread_t, abs=1e-8)
    assert actual.n_total == direct.n_total
    assert actual.cost_hurdle_bps == pytest.approx(direct.cost_hurdle_bps, abs=1e-8)
    assert actual.horizon == direct.horizon
    assert actual.feature_name == direct.feature_name
    assert len(actual.buckets) == len(direct.buckets)
    assert actual.buckets == direct.buckets


def test_default_hurdle_resolves_from_the_cost_model() -> None:
    # HypothesisVerdict.cost_hurdle_bps mirrors ExpectancyTable.cost_hurdle_bps, which is
    # the RAW round-trip cost (~8.26452), not the doubled survival gate -- the "2x" only
    # appears inline in criterion 1's PASS/FAIL comparison text. Asserting 2x here would
    # fail a spec-correct implementation that reuses Lens.verdict() without reimplementing
    # it, which is exactly what this spec requires.
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=18,
    )
    verdict = run_h2(panel, seed=18)
    expected = NSEIntradayEquityCosts().round_trip_bps(1e5)

    assert verdict.cost_hurdle_bps == pytest.approx(expected, abs=1e-8)


def test_survivorship_warning_line_is_in_verdict_output() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 15),
        symbols=symbols,
        relation=0.0,
        seed=19,
    )
    verdict = run_h2(panel, seed=19)

    assert any(line.startswith("UNIVERSE ") for line in verdict.reasons)
    assert "UNIVERSE " in verdict.explain()


def test_stability_partitions_observations_by_calendar_year_exactly_once() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = (
        _business_dates(datetime.date(2019, 1, 2), 8)
        + _business_dates(datetime.date(2020, 1, 2), 8)
        + _business_dates(datetime.date(2021, 1, 4), 8)
    )
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-0.8,
        seed=20,
    )
    verdict = run_h2(panel, seed=20)

    assert set(verdict.stability.by_year) == {2019, 2020, 2021}
    assert sum(table.n_total for table in verdict.stability.by_year.values()) == (
        verdict.expectancy.n_total
    )


def test_one_year_only_effect_is_not_reported_as_sign_stable_across_all_years() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = (
        _business_dates(datetime.date(2019, 1, 2), 12)
        + _business_dates(datetime.date(2020, 1, 2), 12)
        + _business_dates(datetime.date(2021, 1, 4), 12)
    )

    def relationship(day: datetime.date, _: int) -> float:
        if day.year == 2019:
            return -1.5
        return 0.25

    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=relationship,
        seed=21,
        overnight_scale=0.01,
        noise_scale=0.0001,
    )
    verdict = run_h2(panel, seed=21)

    assert verdict.stability.n_years_total == 3
    assert verdict.stability.n_years_sign_consistent < verdict.stability.n_years_total


def test_start_and_end_restrict_the_holdout_window() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = _business_dates(datetime.date(2024, 1, 2), 30)
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-1.0,
        seed=22,
    )
    full = run_h2(panel, seed=22)
    restricted = run_h2(
        panel,
        start=dates[10],
        end=dates[20],
        seed=22,
    )

    assert restricted.expectancy.n_total < full.expectancy.n_total


def test_identical_inputs_and_seed_produce_identical_verdict_statistics() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 25),
        symbols=symbols,
        relation=-0.7,
        seed=23,
    )
    first = run_h2(panel, seed=23)
    second = run_h2(panel, seed=23)

    assert first.seed == second.seed == 23
    assert first.reasons == second.reasons
    assert first.expectancy.spread_bps == pytest.approx(second.expectancy.spread_bps)
    assert first.expectancy.spread_t == pytest.approx(second.expectancy.spread_t)
    assert first.stability == second.stability


def test_seed_is_recorded_in_verdict_and_explanation() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 10),
        symbols=symbols,
        relation=0.0,
        seed=24,
    )
    verdict = run_h2(panel, seed=24)

    assert verdict.seed == 24
    assert "24" in verdict.explain()


def test_degenerate_days_are_nan_excluded_and_do_not_raise() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = _business_dates(datetime.date(2024, 8, 1), 16)
    sessions: dict[datetime.date, dict[str, dict[str, dict[str, float]]]] = {}
    previous = np.asarray([100.0 + ix for ix in range(len(symbols))])

    for day_ix, day in enumerate(dates):
        is_full_day = day_ix % 4 in (3, 0) and day_ix > 0
        active = range(len(symbols)) if is_full_day else range(3)
        opens = previous * np.exp(0.001 * (day_ix + 1))
        closes = opens * np.exp(-0.0005 * (day_ix + 1))
        open_values: dict[str, dict[str, float]] = {}
        close_values: dict[str, dict[str, float]] = {}
        for symbol_ix, symbol in enumerate(symbols):
            if symbol_ix in active:
                open_values[symbol] = {
                    "open": float(opens[symbol_ix]),
                    "close": float(opens[symbol_ix]),
                    "volume": 10_000.0 + symbol_ix,
                }
                close_values[symbol] = {
                    "open": float(closes[symbol_ix]),
                    "close": float(closes[symbol_ix]),
                    "volume": 10_001.0 + symbol_ix,
                }
            else:
                open_values[symbol] = {
                    "open": np.nan,
                    "close": np.nan,
                    "volume": np.nan,
                }
                close_values[symbol] = {
                    "open": np.nan,
                    "close": np.nan,
                    "volume": np.nan,
                }
        sessions[day] = {"09:16": open_values, "15:20": close_values}
        previous = closes

    panel = build_h2_panel(sessions, symbols)
    verdict = run_h2(panel, seed=25)

    assert isinstance(verdict, HypothesisVerdict)
    assert len(verdict.reasons) >= 6
    assert verdict.expectancy.n_total >= 0
    assert not np.isinf(verdict.expectancy.spread_bps)


def test_lens_rejects_a_non_return_feature() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 9, 2), 4),
        symbols=symbols,
        relation=0.0,
        seed=26,
    )
    level = Feature(
        name="not_a_return",
        values=np.ones((panel.n_rows(), panel.n_symbols()), dtype=np.float64),
        kind="level",
        warmup_bars=0,
        params={},
    )

    with pytest.raises(FeatureKindError):
        Lens(panel).expectancy(level, horizon=1, method="cross_sectional_rank")


def test_building_feature_does_not_mutate_panel_fields() -> None:
    symbols = ("S00", "S01")
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 10, 1), 5),
        symbols=symbols,
        relation=-0.5,
        seed=27,
    )
    original_open = panel.field("open").copy()
    original_close = panel.field("close").copy()

    build_overnight_feature(panel)

    np.testing.assert_array_equal(panel.field("open"), original_open)
    np.testing.assert_array_equal(panel.field("close"), original_close)


def test_explicit_session_horizon_matches_the_default_horizon_mode() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 10, 1), 18),
        symbols=symbols,
        relation=-0.9,
        seed=28,
    )
    implicit = run_h2(panel, seed=28)
    explicit = run_h2(panel, horizon_mode="session", seed=28)

    assert explicit.expectancy.spread_bps == pytest.approx(
        implicit.expectancy.spread_bps,
        abs=1e-8,
    )
    assert explicit.expectancy.spread_t == pytest.approx(
        implicit.expectancy.spread_t,
        abs=1e-8,
    )
    assert explicit.expectancy.n_total == implicit.expectancy.n_total


def test_missing_checkpoint_on_current_day_is_nan_for_only_that_symbol() -> None:
    symbols = ("S00", "S01", "S02", "S03", "S04")
    day_one = datetime.date(2024, 11, 11)
    day_two = datetime.date(2024, 11, 12)
    sessions = {
        day_one: {
            "09:16": {
                symbol: {"open": 100.0 + ix, "close": 100.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {"open": 101.0 + ix, "close": 101.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
        day_two: {
            "09:16": {
                symbol: {"open": 110.0 + ix, "close": 110.0 + ix}
                for ix, symbol in enumerate(symbols[:-1])
            },
            "15:20": {
                symbol: {"open": 111.0 + ix, "close": 111.0 + ix}
                for ix, symbol in enumerate(symbols)
            },
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)
    values = feature.values[_session_rows(panel, day_two)]

    assert np.isfinite(values[:, :4]).all()
    assert np.isnan(values[:, 4]).all()


def test_overnight_feature_uses_1520_close_not_the_last_arbitrary_bar() -> None:
    symbols = ("S00",)
    day_one = datetime.date(2024, 12, 2)
    day_two = datetime.date(2024, 12, 3)
    sessions = {
        day_one: {
            "09:16": {"S00": {"open": 95.0, "close": 95.0}},
            "12:00": {"S00": {"open": 97.0, "close": 97.0}},
            "15:20": {"S00": {"open": 100.0, "close": 100.0}},
            "15:21": {"S00": {"open": 50.0, "close": 50.0}},
        },
        day_two: {
            "09:15": {"S00": {"open": 108.0, "close": 109.0, "high": 110.0}},
            "09:16": {"S00": {"open": 110.0, "close": 110.0}},
            "15:20": {"S00": {"open": 111.0, "close": 111.0}},
        },
    }
    panel = build_h2_panel(sessions, symbols)
    feature = build_overnight_feature(panel)

    value = _feature_session_values(feature, panel, day_two, "S00")[0]
    expected = np.log(
        np.float64(np.float32(110.0)) / np.float64(np.float32(100.0))
    )

    assert value == pytest.approx(expected, abs=1e-5)
