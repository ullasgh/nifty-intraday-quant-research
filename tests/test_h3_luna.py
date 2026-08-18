from __future__ import annotations

import dataclasses
import datetime
from typing import Callable, Mapping

import numpy as np
import pytest
from nifty_quant.research.hypotheses.h3_intraday_xsec_reversal import (
    build_morning_residual_feature,
    run_h3,
)

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.lens import Feature, FeatureKindError, HypothesisVerdict, Lens

_IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
_EPOCH = datetime.datetime(1970, 1, 1)

SessionMapping = Mapping[
    datetime.date,
    Mapping[str, Mapping[str, Mapping[str, float]]],
]


def _epoch(day: datetime.date, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(day.year, day.month, day.day, hour, minute)
    return int((local - _IST_OFFSET - _EPOCH).total_seconds())


def build_h3_panel(
    sessions: SessionMapping,
    symbols: tuple[str, ...],
) -> Panel:
    """Build a float32-at-rest H3 panel from per-symbol labelled bars."""
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
                fields["volume"][row_ix, symbol_ix] = np.float32(
                    1_000.0 + symbol_ix
                )

    offsets = [0]
    for day in all_dates:
        offsets.append(
            offsets[-1] + sum(row_day == day for _, row_day, _ in rows)
        )

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


def _clone_sessions(sessions: SessionMapping) -> dict[
    datetime.date,
    dict[str, dict[str, dict[str, float]]],
]:
    return {
        day: {
            hhmm: {
                symbol: dict(values)
                for symbol, values in symbol_values.items()
            }
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


def _checkpoint_sessions(
    dates: list[datetime.date],
    *,
    symbols: tuple[str, ...],
    relation: float | Callable[[datetime.date, int], float],
    seed: int,
    morning_scale: float = 0.01,
    noise_scale: float = 0.0002,
) -> dict[datetime.date, dict[str, dict[str, dict[str, float]]]]:
    rng = np.random.default_rng(seed)
    prior_close = np.asarray(
        [100.0 + 2.0 * ix for ix in range(len(symbols))],
        dtype=np.float64,
    )
    sessions: dict[
        datetime.date,
        dict[str, dict[str, dict[str, float]]],
    ] = {}

    for day_ix, day in enumerate(dates):
        morning = rng.normal(
            0.0,
            morning_scale,
            size=len(symbols),
        )
        noise = rng.normal(
            0.0,
            noise_scale,
            size=len(symbols),
        )
        day_open = prior_close * np.exp(rng.normal(0.0, 0.002, len(symbols)))
        morning_close = day_open * np.exp(morning)
        day_close = morning_close * np.exp(
            (
                relation(day, day_ix)
                if callable(relation)
                else relation
            )
            * morning
            + noise
        )

        bars: dict[str, dict[str, dict[str, float]]] = {}
        for hhmm, prices in (
            ("09:16", day_open),
            ("10:00", morning_close),
            ("15:20", day_close),
        ):
            bars[hhmm] = {}
            for symbol_ix, symbol in enumerate(symbols):
                price = float(prices[symbol_ix])
                bars[hhmm][symbol] = {
                    "open": price,
                    "close": price,
                    "volume": 10_000.0 + 1_000.0 * symbol_ix + 17.0 * day_ix,
                }

        sessions[day] = bars
        prior_close = day_close

    return sessions


def _signal_panel(
    dates: list[datetime.date],
    *,
    symbols: tuple[str, ...],
    relation: float | Callable[[datetime.date, int], float],
    seed: int,
    morning_scale: float = 0.01,
    noise_scale: float = 0.0002,
) -> Panel:
    return build_h3_panel(
        _checkpoint_sessions(
            dates,
            symbols=symbols,
            relation=relation,
            seed=seed,
            morning_scale=morning_scale,
            noise_scale=noise_scale,
        ),
        symbols,
    )


def _zero_mean_sessions(
    *,
    day: datetime.date = datetime.date(2024, 1, 2),
    symbols: tuple[str, ...] = ("A", "B"),
) -> dict[datetime.date, dict[str, dict[str, dict[str, float]]]]:
    x = 0.0123
    opens = {"A": 100.0, "B": 100.0}
    morning_closes = {
        "A": 100.0 * np.exp(x),
        "B": 100.0 * np.exp(-x),
    }
    rest_closes = {
        "A": morning_closes["A"] * np.exp(-0.004),
        "B": morning_closes["B"] * np.exp(0.003),
    }

    bars: dict[str, dict[str, dict[str, float]]] = {}
    for hhmm, prices in (
        ("09:16", opens),
        ("10:00", morning_closes),
        ("15:20", rest_closes),
    ):
        bars[hhmm] = {
            symbol: {
                "open": float(prices[symbol]),
                "close": float(prices[symbol]),
                "volume": 10_000.0 + 1_000.0 * ix,
            }
            for ix, symbol in enumerate(symbols)
        }

    return {day: bars}


def _add_muhurat_session(
    sessions: dict[datetime.date, dict[str, dict[str, dict[str, float]]]],
    day: datetime.date,
    symbols: tuple[str, ...],
) -> None:
    bars: dict[str, dict[str, dict[str, float]]] = {}
    for minute in range(11 * 60, 12 * 60):
        hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
        bars[hhmm] = {
            symbol: {
                "open": 200.0 + symbol_ix,
                "close": 200.0 + symbol_ix + 0.01 * minute,
                "volume": 5_000.0 + symbol_ix,
            }
            for symbol_ix, symbol in enumerate(symbols)
        }
    sessions[day] = bars


def _realistic_panel_pair(
    dates: list[datetime.date],
    *,
    symbols: tuple[str, ...],
    relation: float,
    seed: int,
) -> tuple[Panel, Panel]:
    checkpoint_sessions = _checkpoint_sessions(
        dates,
        symbols=symbols,
        relation=relation,
        seed=seed,
        morning_scale=0.01,
        noise_scale=0.00015,
    )
    minimal_panel = build_h3_panel(checkpoint_sessions, symbols)

    rng = np.random.default_rng(707)
    realistic_sessions = _clone_sessions(checkpoint_sessions)

    labels: list[str] = []
    for minute in range(9 * 60 + 15, 15 * 60 + 31, 2):
        labels.append(f"{minute // 60:02d}:{minute % 60:02d}")
    for label in ("09:16", "10:00", "15:20"):
        if label not in labels:
            labels.append(label)
    labels.sort(key=lambda label: tuple(int(part) for part in label.split(":")))

    for day in dates:
        checkpoint_day = checkpoint_sessions[day]
        current = np.asarray(
            [
                checkpoint_day["09:16"][symbol]["close"]
                for symbol in symbols
            ],
            dtype=np.float64,
        )
        day_bars: dict[str, dict[str, dict[str, float]]] = {}

        for hhmm in labels:
            if hhmm in checkpoint_day:
                day_bars[hhmm] = {
                    symbol: dict(checkpoint_day[hhmm][symbol])
                    for symbol in symbols
                }
                current = np.asarray(
                    [
                        checkpoint_day[hhmm][symbol]["close"]
                        for symbol in symbols
                    ],
                    dtype=np.float64,
                )
                continue

            current *= np.exp(rng.normal(0.0, 0.0005, len(symbols)))
            day_bars[hhmm] = {
                symbol: {
                    "open": float(current[symbol_ix]),
                    "close": float(current[symbol_ix]),
                    "volume": 8_000.0 + symbol_ix,
                }
                for symbol_ix, symbol in enumerate(symbols)
            }

        realistic_sessions[day] = day_bars

    realistic_panel = build_h3_panel(realistic_sessions, symbols)
    return realistic_panel, minimal_panel


def test_morning_feature_uses_0916_open_and_1000_close() -> None:
    symbols = ("A", "B")
    panel = build_h3_panel(_zero_mean_sessions(symbols=symbols), symbols)
    feature = build_morning_residual_feature(panel)

    entry_row = panel.rows_at_time("09:16")[0]
    morning_row = panel.rows_at_time("10:00")[0]
    expected = np.log(
        panel.field("close")[morning_row]
        / panel.field("open")[entry_row]
    )

    np.testing.assert_allclose(
        feature.values[morning_row],
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_morning_and_rest_returns_telescope_through_shared_1000_close() -> None:
    symbols = ("A", "B")
    panel = build_h3_panel(_zero_mean_sessions(symbols=symbols), symbols)
    feature = build_morning_residual_feature(panel)

    entry_row = panel.rows_at_time("09:16")[0]
    morning_row = panel.rows_at_time("10:00")[0]
    exit_row = panel.rows_at_time("15:20")[0]

    for symbol_ix in range(panel.n_symbols()):
        morning = feature.values[morning_row, symbol_ix]
        rest = np.log(
            panel.field("close")[exit_row, symbol_ix]
            / panel.field("close")[morning_row, symbol_ix]
        )
        full_session = np.log(
            panel.field("close")[exit_row, symbol_ix]
            / panel.field("open")[entry_row, symbol_ix]
        )
        # This is the H3-specific shared-10:00-close risk: both legs must use
        # the identical stored 10:00 close rather than different positional rows.
        assert morning + rest == pytest.approx(full_session, abs=1e-4)


def test_0915_poison_row_is_never_read() -> None:
    symbols = ("A", "B")
    clean_sessions = _zero_mean_sessions(symbols=symbols)
    poison_sessions = _clone_sessions(clean_sessions)
    poison_sessions[next(iter(poison_sessions))]["09:15"] = {
        "A": {
            "open": 1.0,
            "close": 10_000.0,
            "high": 99_999.0,
            "volume": 1.0,
        },
        "B": {
            "open": 2.0,
            "close": 20_000.0,
            "high": 88_888.0,
            "volume": 1.0,
        },
    }

    clean_panel = build_h3_panel(clean_sessions, symbols)
    poison_panel = build_h3_panel(poison_sessions, symbols)
    clean_feature = build_morning_residual_feature(clean_panel)
    poison_feature = build_morning_residual_feature(poison_panel)

    for hhmm in ("09:16", "10:00", "15:20"):
        np.testing.assert_array_equal(
            clean_feature.values[clean_panel.rows_at_time(hhmm)],
            poison_feature.values[poison_panel.rows_at_time(hhmm)],
        )


def test_checkpoint_resolution_uses_time_labels_not_positions() -> None:
    symbols = ("A", "B")
    clean_sessions = _zero_mean_sessions(symbols=symbols)
    extra_sessions = _clone_sessions(clean_sessions)
    extra_sessions[next(iter(extra_sessions))]["10:01"] = {
        "A": {
            "open": 1.0,
            "close": 50_000.0,
            "volume": 1.0,
        },
        "B": {
            "open": 2.0,
            "close": 60_000.0,
            "volume": 1.0,
        },
    }

    clean_panel = build_h3_panel(clean_sessions, symbols)
    extra_panel = build_h3_panel(extra_sessions, symbols)
    clean_feature = build_morning_residual_feature(clean_panel)
    extra_feature = build_morning_residual_feature(extra_panel)

    for hhmm in ("09:16", "10:00", "15:20"):
        np.testing.assert_array_equal(
            clean_feature.values[clean_panel.rows_at_time(hhmm)],
            extra_feature.values[extra_panel.rows_at_time(hhmm)],
        )


def test_muhurat_without_required_checkpoints_is_dropped_without_raising() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(6))
    dates = _business_dates(datetime.date(2024, 1, 2), 5)
    all_sessions = _checkpoint_sessions(
        dates[:4],
        symbols=symbols,
        relation=-1.0,
        seed=333,
    )
    _add_muhurat_session(all_sessions, dates[4], symbols)

    mixed_panel = build_h3_panel(all_sessions, symbols)
    feature = build_morning_residual_feature(mixed_panel)
    muhurat_rows = _session_rows(mixed_panel, dates[4])
    assert np.all(np.isnan(feature.values[muhurat_rows]))

    mixed_verdict = run_h3(mixed_panel, seed=333)
    normal_panel = build_h3_panel(
        {day: all_sessions[day] for day in dates[:4]},
        symbols,
    )
    normal_verdict = run_h3(normal_panel, seed=333)
    complete_panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-1.0,
        seed=333,
    )
    complete_verdict = run_h3(complete_panel, seed=333)

    assert isinstance(mixed_verdict, HypothesisVerdict)
    assert np.isfinite(mixed_verdict.expectancy.spread_bps)
    assert mixed_verdict.expectancy.n_total == normal_verdict.expectancy.n_total
    assert mixed_verdict.expectancy.n_total < complete_verdict.expectancy.n_total


def test_missing_symbol_checkpoint_is_nan_without_poisoning_other_symbols() -> None:
    day = datetime.date(2024, 1, 2)
    symbols = ("A", "B")
    sessions = _zero_mean_sessions(day=day, symbols=symbols)
    del sessions[day]["10:00"]["B"]

    panel = build_h3_panel(sessions, symbols)
    feature = build_morning_residual_feature(panel)
    morning_row = panel.rows_at_time("10:00")[0]

    assert np.isfinite(feature.values[morning_row, panel.sym_ix["A"]])
    assert np.isnan(feature.values[morning_row, panel.sym_ix["B"]])


def test_nan_does_not_forward_fill_across_sessions_or_symbols() -> None:
    symbols = ("A", "B")
    dates = _business_dates(datetime.date(2024, 1, 2), 3)
    sessions = _checkpoint_sessions(
        dates,
        symbols=symbols,
        relation=-1.0,
        seed=41,
    )
    del sessions[dates[1]]["10:00"]["A"]

    panel = build_h3_panel(sessions, symbols)
    feature = build_morning_residual_feature(panel)

    values = [
        _feature_session_values(feature, panel, day, "A")[0]
        for day in dates
    ]
    assert np.isfinite(values[0])
    assert np.isnan(values[1])
    assert np.isfinite(values[2])


def test_realistic_many_bar_panel_matches_checkpoint_equivalent_panel() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = _business_dates(datetime.date(2024, 1, 2), 12)
    realistic_panel, minimal_panel = _realistic_panel_pair(
        dates,
        symbols=symbols,
        relation=-1.2,
        seed=818,
    )

    rows_per_session = realistic_panel.n_rows() / len(dates)
    assert 150 <= rows_per_session <= 200

    # Regression test for the exact prior H2 defect: without checkpoint-panel
    # reduction, horizon=1 measures a one-minute return rather than 10:00->15:20.
    # Such an implementation should fail this hard with order-of-magnitude wrong values.
    realistic_verdict = run_h3(realistic_panel, seed=818)
    minimal_verdict = run_h3(minimal_panel, seed=818)

    assert realistic_verdict.expectancy.spread_bps == pytest.approx(
        minimal_verdict.expectancy.spread_bps,
        abs=1e-2,
    )
    assert realistic_verdict.expectancy.spread_t == pytest.approx(
        minimal_verdict.expectancy.spread_t,
        abs=1e-2,
    )
    assert realistic_verdict.expectancy.n_total == minimal_verdict.expectancy.n_total


def test_cross_sectional_demeaning_makes_each_signal_row_sum_to_zero() -> None:
    day = datetime.date(2024, 1, 2)
    symbols = ("A", "B", "C", "D")
    opens = [100.0, 100.0, 100.0, 100.0]
    morning_returns = [0.01, -0.004, 0.021, 0.006]
    closes = [
        open_price * np.exp(return_value)
        for open_price, return_value in zip(opens, morning_returns)
    ]
    bars = {
        "09:16": {
            symbol: {
                "open": opens[ix],
                "close": opens[ix],
                "volume": 10_000.0 + ix,
            }
            for ix, symbol in enumerate(symbols)
        },
        "10:00": {
            symbol: {
                "open": closes[ix],
                "close": closes[ix],
                "volume": 10_000.0 + ix,
            }
            for ix, symbol in enumerate(symbols)
        },
        "15:20": {
            symbol: {
                "open": closes[ix],
                "close": closes[ix] * 1.001,
                "volume": 10_000.0 + ix,
            }
            for ix, symbol in enumerate(symbols)
        },
    }

    panel = build_h3_panel({day: bars}, symbols)
    feature = build_morning_residual_feature(panel)
    row = panel.rows_at_time("10:00")[0]

    assert np.nansum(feature.values[row, :]) == pytest.approx(0.0, abs=1e-3)


def test_reversal_direction_is_reported_for_negative_spread() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 45),
        symbols=symbols,
        relation=-1.4,
        seed=912,
        morning_scale=0.01,
        noise_scale=0.00015,
    )
    verdict = run_h3(panel, seed=912)
    direction_line = next(
        line for line in verdict.reasons
        if line.startswith("Observed direction:")
    )

    assert verdict.expectancy.spread_bps < 0.0
    assert abs(verdict.expectancy.spread_t) > 1.96
    assert "reversal" in verdict.explain().lower()
    assert "momentum" not in direction_line.lower()


def test_momentum_direction_is_not_reported_as_confirmed_reversal() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 45),
        symbols=symbols,
        relation=1.4,
        seed=913,
        morning_scale=0.01,
        noise_scale=0.00015,
    )
    verdict = run_h3(panel, seed=913)
    direction_line = next(
        line for line in verdict.reasons
        if line.startswith("Observed direction:")
    )

    assert verdict.expectancy.spread_bps > 0.0
    assert abs(verdict.expectancy.spread_t) > 1.96
    assert "momentum" in direction_line.lower()
    assert "consistent with reversal" not in direction_line.lower()


def test_verified_driftless_pure_noise_is_not_significant_or_cost_surviving() -> None:
    # This chosen seed's realized spread_t was checked to be below 1.96 and is
    # not an unlucky significant draw from the fully driftless fixture.
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 72),
        symbols=symbols,
        relation=0.0,
        seed=214,
        morning_scale=0.01,
        noise_scale=0.0002,
    )
    verdict = run_h3(panel, seed=214)

    assert np.isfinite(verdict.expectancy.spread_bps)
    assert abs(verdict.expectancy.spread_bps) < verdict.cost_hurdle_bps
    assert abs(verdict.expectancy.spread_t) < 1.96
    assert verdict.expectancy.survives_costs is False
    assert verdict.survived is False


def test_significant_but_sub_gate_reversal_is_killed() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 90),
        symbols=symbols,
        relation=-0.02,
        seed=215,
        morning_scale=0.01,
        noise_scale=0.000001,
    )
    verdict = run_h3(panel, seed=215)
    gate = 2.0 * NSEIntradayEquityCosts().round_trip_bps(1e5)

    assert abs(verdict.expectancy.spread_t) > 1.96
    assert abs(verdict.expectancy.spread_bps) < gate
    assert verdict.expectancy.survives_costs is False
    assert verdict.survived is False


def test_all_seven_criteria_are_reported_after_first_criterion_fails() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=216,
    )
    verdict = run_h3(panel, seed=216)

    assert len(verdict.reasons) >= 7
    for criterion_ix in range(1, 8):
        assert verdict.reasons[criterion_ix - 1].startswith(f"{criterion_ix}.")
    assert "FAIL" in verdict.reasons[0]


def test_latency_criterion_is_not_evaluated_by_run_h3() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=217,
    )
    verdict = run_h3(panel, seed=217)

    assert "NOT_EVALUATED" in verdict.reasons[4]
    assert "PASS" not in verdict.reasons[4]


def test_recent_year_cost_gate_is_evaluated_with_three_complete_years() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = (
        _business_dates(datetime.date(2022, 1, 3), 22)
        + _business_dates(datetime.date(2023, 1, 3), 22)
        + _business_dates(datetime.date(2024, 1, 2), 22)
    )
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-1.4,
        seed=218,
        morning_scale=0.01,
        noise_scale=0.0001,
    )
    verdict = run_h3(panel, seed=218)

    assert verdict.reasons[6].startswith("7.")
    assert "NOT_EVALUATED" not in verdict.reasons[6]
    assert "PASS" in verdict.reasons[6] or "FAIL" in verdict.reasons[6]


def test_recent_year_cost_gate_is_not_evaluated_with_too_few_complete_years() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 19),
        symbols=symbols,
        relation=-1.4,
        seed=219,
    )
    verdict = run_h3(panel, seed=219)

    assert verdict.reasons[6].startswith("7.")
    assert "NOT_EVALUATED" in verdict.reasons[6]


def _build_reference_checkpoint_panel(
    panel: Panel, feature: Feature
) -> tuple[Panel, Feature]:
    """Independently reduce `panel`/`feature` to the 10:00-entry/15:20-exit checkpoint
    shape `run_h3` must build internally, so the comparison test below can check
    `Lens.expectancy()` directly without depending on `run_h3`'s own private reduction
    step (which is not part of the public API under test).

    Unlike H2 -- where the raw fixture panel already has exactly the two checkpoints
    (09:16, 15:20) `run_h2` trades on, so a direct `Lens(panel, ...).expectancy(...,
    horizon=1)` call on the raw panel coincidentally matches the internally-reduced
    panel -- H3's raw panel has a THIRD checkpoint (09:16) that only the SIGNAL needs,
    not the trade. Calling `Lens` directly on the raw 3-checkpoint-per-session panel
    with `horizon=1` would also include the 09:16->10:00 leg in the "forward return"
    population (one bar forward from the 09:16 row), not just the true 10:00->15:20
    trade leg -- silently wrong, not an error. This helper performs the SAME two-row
    (entry=10:00 close, exit=15:20 close) reduction `run_h3` is required to perform,
    mirroring H2's proven `_build_checkpoint_panel`.
    """
    entry_rows = panel.rows_at_time("10:00")
    exit_rows = panel.rows_at_time("15:20")
    entry_days = np.searchsorted(panel.day_offsets, entry_rows, side="right") - 1
    exit_days = np.searchsorted(panel.day_offsets, exit_rows, side="right") - 1

    entry_row_by_day = dict(zip(entry_days.tolist(), entry_rows.tolist(), strict=True))
    exit_row_by_day = dict(zip(exit_days.tolist(), exit_rows.tolist(), strict=True))
    usable_days = sorted(set(entry_row_by_day) & set(exit_row_by_day))

    n_symbols = panel.n_symbols()
    n_rows = 2 * len(usable_days)

    panel_close = panel.field("close")
    panel_volume = panel.field("volume")

    ts = np.empty(n_rows, dtype=np.int64)
    close = np.empty((n_rows, n_symbols), dtype=panel_close.dtype)
    volume = np.empty((n_rows, n_symbols), dtype=panel_volume.dtype)
    feat_values = np.empty((n_rows, n_symbols), dtype=np.float64)

    for i, day in enumerate(usable_days):
        entry_row = entry_row_by_day[day]
        exit_row = exit_row_by_day[day]
        entry_out, exit_out = 2 * i, 2 * i + 1
        ts[entry_out] = panel.ts[entry_row]
        ts[exit_out] = panel.ts[exit_row]
        close[entry_out, :] = panel_close[entry_row, :]
        close[exit_out, :] = panel_close[exit_row, :]
        volume[entry_out, :] = panel_volume[entry_row, :]
        volume[exit_out, :] = panel_volume[exit_row, :]
        feat_values[entry_out, :] = feature.values[entry_row, :]
        feat_values[exit_out, :] = feature.values[entry_row, :]

    grid = SessionGrid.from_timestamps(ts)
    checkpoint_panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=panel.symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )
    checkpoint_feature = dataclasses.replace(feature, values=feat_values)
    return checkpoint_panel, checkpoint_feature


def test_run_h3_uses_lens_expectancy_and_matches_on_checkpoint_panel() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = _business_dates(datetime.date(2024, 1, 2), 12)
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-0.8,
        seed=220,
    )
    verdict = run_h3(panel, seed=220)
    feature = build_morning_residual_feature(panel)

    checkpoint_panel, checkpoint_feature = _build_reference_checkpoint_panel(panel, feature)
    direct = Lens(checkpoint_panel, seed=220).expectancy(
        checkpoint_feature,
        horizon=1,
        method="cross_sectional_rank",
    )

    assert isinstance(direct, type(verdict.expectancy))
    assert verdict.expectancy.spread_bps == pytest.approx(direct.spread_bps, abs=1e-8)
    assert verdict.expectancy.spread_t == pytest.approx(direct.spread_t, abs=1e-8)
    assert verdict.expectancy.n_total == direct.n_total
    assert verdict.expectancy.cost_hurdle_bps == pytest.approx(direct.cost_hurdle_bps, abs=1e-8)
    assert verdict.expectancy.horizon == direct.horizon
    assert verdict.expectancy.feature_name == direct.feature_name
    assert verdict.expectancy.buckets == direct.buckets


def test_default_hurdle_is_the_raw_one_x_round_trip_cost() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=221,
    )
    verdict = run_h3(panel, seed=221)
    one_x_cost = NSEIntradayEquityCosts().round_trip_bps(1e5)

    # The verdict stores the raw 1x cost; 2x is only the inline survival gate.
    assert verdict.cost_hurdle_bps == pytest.approx(one_x_cost, abs=1e-8)


def test_survivorship_warning_is_present_in_reasons_and_explanation() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=222,
    )
    verdict = run_h3(panel, seed=222)

    assert any(line.startswith("UNIVERSE ") for line in verdict.reasons)
    assert "UNIVERSE " in verdict.explain()


def test_stability_year_tables_partition_sessions_exactly_once() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = (
        _business_dates(datetime.date(2022, 1, 3), 22)
        + _business_dates(datetime.date(2023, 1, 3), 22)
        + _business_dates(datetime.date(2024, 1, 2), 22)
    )
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-0.8,
        seed=223,
    )
    verdict = run_h3(panel, seed=223)

    assert set(verdict.stability.by_year) == {2022, 2023, 2024}
    assert sum(
        table.n_total for table in verdict.stability.by_year.values()
    ) == verdict.expectancy.n_total


def test_start_and_end_restrict_the_expectancy_window() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    dates = _business_dates(datetime.date(2024, 1, 2), 16)
    panel = _signal_panel(
        dates,
        symbols=symbols,
        relation=-0.8,
        seed=224,
    )
    full_verdict = run_h3(panel, seed=224)
    restricted_verdict = run_h3(
        panel,
        start=dates[4],
        end=dates[9],
        seed=224,
    )

    assert restricted_verdict.expectancy.n_total < full_verdict.expectancy.n_total


def test_run_h3_is_deterministic_for_same_panel_and_seed() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 24),
        symbols=symbols,
        relation=-0.8,
        seed=225,
    )
    first = run_h3(panel, seed=225)
    second = run_h3(panel, seed=225)

    assert first.reasons == second.reasons
    assert first.expectancy.spread_bps == pytest.approx(
        second.expectancy.spread_bps,
        abs=1e-8,
    )
    assert first.expectancy.spread_t == pytest.approx(
        second.expectancy.spread_t,
        abs=1e-8,
    )
    assert first.stability == second.stability


def test_seed_is_recorded_in_verdict_and_explanation() -> None:
    symbols = tuple(f"S{ix:02d}" for ix in range(8))
    panel = _signal_panel(
        _business_dates(datetime.date(2024, 1, 2), 12),
        symbols=symbols,
        relation=0.0,
        seed=226,
    )
    verdict = run_h3(panel, seed=226)

    assert verdict.seed == 226
    assert "226" in verdict.explain()


def test_constant_cross_sectional_signal_does_not_crash_lens() -> None:
    # Covers tied/degenerate cross-sectional ranks, which the specification
    # does not explicitly enumerate but which must not crash a research run.
    symbols = ("A", "B", "C", "D")
    day = datetime.date(2024, 1, 2)
    sessions = {
        day: {
            "09:16": {
                symbol: {
                    "open": 100.0,
                    "close": 100.0,
                    "volume": 10_000.0 + ix,
                }
                for ix, symbol in enumerate(symbols)
            },
            "10:00": {
                symbol: {
                    "open": 101.0,
                    "close": 101.0,
                    "volume": 10_000.0 + ix,
                }
                for ix, symbol in enumerate(symbols)
            },
            "15:20": {
                symbol: {
                    "open": 101.0,
                    "close": 101.1,
                    "volume": 10_000.0 + ix,
                }
                for ix, symbol in enumerate(symbols)
            },
        }
    }
    panel = build_h3_panel(sessions, symbols)
    feature = build_morning_residual_feature(panel)
    verdict = run_h3(panel, seed=227)

    assert np.allclose(feature.values, 0.0, atol=1e-6)
    assert isinstance(verdict, HypothesisVerdict)


def test_lens_rejects_non_return_features() -> None:
    symbols = ("A", "B")
    panel = build_h3_panel(
        _zero_mean_sessions(symbols=symbols),
        symbols,
    )
    bad_feature = Feature(
        name="not_a_return",
        values=np.zeros((panel.n_rows(), panel.n_symbols())),
        kind="level",
        warmup_bars=0,
        params={},
    )

    with pytest.raises(FeatureKindError):
        Lens(panel, seed=228).expectancy(
            bad_feature,
            horizon=1,
            method="cross_sectional_rank",
        )
