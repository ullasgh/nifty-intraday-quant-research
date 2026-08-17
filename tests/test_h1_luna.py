from __future__ import annotations

# The statistical requirements require a pre-holdout date range, while the public API
# signature omits it.  These tests therefore assume run_h1 gains inclusive keyword-only
# parameters start: date | None = None and end: date | None = None, defaulting to full range.
import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h1_market_intraday_momentum import (
    H1Result,
    SessionObservations,
    extract_session_observations,
    run_h1,
)
from nifty_quant.research.splits import HoldoutLock

_IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
_EPOCH = datetime.datetime(1970, 1, 1)


def _epoch(day: datetime.date, hhmm: str) -> int:
    """Return the UTC epoch-second timestamp for an IST wall-clock time."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(day.year, day.month, day.day, hour, minute)
    return int((local - _IST_OFFSET - _EPOCH).total_seconds())


def build_panel(
    sessions: Mapping[datetime.date, Mapping[str, Mapping[str, float]]],
    symbols: tuple[str, ...] = ("NIFTY50",),
) -> Panel:
    """Build a sparse float32-at-rest Panel from labelled synthetic bars."""
    all_dates = sorted(sessions)
    rows: list[tuple[int, datetime.date, dict[str, float]]] = []

    for day in all_dates:
        for hhmm, field_values in sessions[day].items():
            rows.append((_epoch(day, hhmm), day, dict(field_values)))

    rows.sort(key=lambda row: row[0])
    ts = np.asarray([row[0] for row in rows], dtype=np.int64)
    row_dates = [row[1] for row in rows]
    field_names = sorted({name for _, _, values in rows for name in values})
    n_symbols = len(symbols)

    fields: dict[str, np.ndarray] = {
        name: np.full((len(rows), n_symbols), np.nan, dtype=np.float32)
        for name in field_names
    }
    for row_idx, (_, _, field_values) in enumerate(rows):
        for name, value in field_values.items():
            fields[name][row_idx, :] = value

    offsets = [0]
    for day in all_dates:
        offsets.append(offsets[-1] + sum(row_day == day for row_day in row_dates))

    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=np.asarray(offsets, dtype=np.int32),
        dates=np.asarray(all_dates, dtype=object),
    )


def _returns_session(
    day: datetime.date,
    r_first: float,
    r_mid: float,
    r_last: float,
) -> dict[datetime.date, dict[str, dict[str, float]]]:
    """Create one labelled session whose checkpoint returns are prescribed."""
    open_first = 100.0
    close_first = open_first * float(np.exp(r_first))
    open_last = close_first * float(np.exp(r_mid))
    close_last = open_last * float(np.exp(r_last))

    return {
        day: {
            "09:16": {"open": open_first, "close": open_first},
            "09:45": {"open": close_first, "close": close_first},
            "14:50": {"open": open_last, "close": open_last},
            "15:20": {"open": close_last, "close": close_last},
        }
    }


def _panel_for_returns(
    r_first: np.ndarray,
    r_mid: np.ndarray,
    r_last: np.ndarray,
    *,
    start: datetime.date = datetime.date(2018, 1, 2),
) -> Panel:
    """Build one synthetic session per return triple."""
    if not (len(r_first) == len(r_mid) == len(r_last)):
        raise ValueError("return arrays must have equal lengths")

    sessions: dict[datetime.date, dict[str, dict[str, float]]] = {}
    for index, (first, mid, last) in enumerate(zip(r_first, r_mid, r_last, strict=True)):
        day = start + datetime.timedelta(days=index)
        sessions.update(_returns_session(day, float(first), float(mid), float(last)))
    return build_panel(sessions)


def _one_result(
    r_first: np.ndarray,
    r_last: np.ndarray,
    *,
    r_mid: np.ndarray | None = None,
    cost_hurdle_bps: float | None = None,
) -> H1Result:
    """Run H1 on return arrays, supplying a nonconstant middle return by default."""
    if r_mid is None:
        r_mid = np.linspace(-0.001, 0.001, len(r_first), dtype=np.float64)
    panel = _panel_for_returns(r_first, r_mid, r_last)
    if cost_hurdle_bps is None:
        return run_h1(panel, "NIFTY50")
    return run_h1(panel, "NIFTY50", cost_hurdle_bps=cost_hurdle_bps)


def test_checkpoint_returns_use_exact_labels_and_ignore_corrupt_0915() -> None:
    day = datetime.date(2024, 1, 2)
    sessions = {
        day: {
            "09:15": {"open": 1.0e9, "close": -1.0e9},
            "09:16": {"open": 100.0, "close": 101.0},
            "09:45": {"open": 107.0, "close": 110.0},
            "14:50": {"open": 104.0, "close": 105.0},
            "15:20": {"open": 106.0, "close": 108.0},
        }
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert isinstance(observations, SessionObservations)
    assert observations.n_sessions_dropped == 0
    assert observations.r_first[0] == pytest.approx(np.log(110.0 / 100.0), rel=1e-7)
    # r_last = log(close[15:20] / open[14:50]); open[14:50] is 104.0, NOT close[14:50]=105.0.
    assert observations.r_last[0] == pytest.approx(np.log(108.0 / 104.0), rel=1e-7)
    assert observations.r_first[0] != pytest.approx(np.log(110.0 / 1.0e9))
    assert observations.r_first.dtype == np.float64
    assert observations.r_mid.dtype == np.float64
    assert observations.r_last.dtype == np.float64
    assert "drop" in observations.explain().lower()


def test_return_decomposition_uses_first_close_and_last_open() -> None:
    day = datetime.date(2024, 2, 1)
    sessions = {
        day: {
            "09:16": {"open": 100.0, "close": 101.0},
            "09:45": {"open": 105.0, "close": 110.0},
            "14:50": {"open": 104.0, "close": 105.0},
            "15:20": {"open": 106.0, "close": 108.0},
        }
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")
    expected_total = np.log(108.0 / 100.0)

    # r_first = log(close[09:45] / open[09:16]) -> denominator is OPEN (100.0), not close (101.0).
    assert observations.r_first[0] == pytest.approx(np.log(110.0 / 100.0), rel=1e-7)
    # r_mid = log(open[14:50] / close[09:45]) -> numerator is OPEN (104.0), not close (105.0).
    assert observations.r_mid[0] == pytest.approx(np.log(104.0 / 110.0), rel=1e-7)
    # r_last = log(close[15:20] / open[14:50]) -> denominator is OPEN (104.0), not close (105.0).
    assert observations.r_last[0] == pytest.approx(np.log(108.0 / 104.0), rel=1e-7)
    assert np.isclose(
        observations.r_first[0] + observations.r_mid[0] + observations.r_last[0],
        expected_total,
        rtol=1e-9,
        atol=1e-12,
    )


def test_missing_1450_is_dropped_and_counted() -> None:
    valid_day = datetime.date(2024, 3, 1)
    missing_day = datetime.date(2024, 3, 4)
    sessions = {
        valid_day: _returns_session(valid_day, 0.001, 0.0, 0.002)[valid_day],
        missing_day: {
            "09:16": {"open": 100.0, "close": 101.0},
            "09:45": {"open": 101.0, "close": 102.0},
            "15:20": {"open": 103.0, "close": 104.0},
        },
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert observations.n_sessions_dropped == 1
    assert observations.dates.tolist() == [valid_day]
    # Resolved 2026-08-17: exactly one key increments per dropped session (first missing
    # checkpoint in order 09:16, 09:45, 14:50, 15:20), so this dict is exactly one entry.
    assert observations.drop_reasons == {"missing_14:50": 1}
    assert sum(observations.drop_reasons.values()) == observations.n_sessions_dropped


def test_muhurat_session_drops_naturally_for_missing_late_checkpoints() -> None:
    regular_day = datetime.date(2024, 11, 4)
    muhurat_day = datetime.date(2024, 11, 1)
    muhurat_bars: dict[str, dict[str, float]] = {}

    for minute_offset in range(60):
        minute = 16 + minute_offset
        hour = 9 + minute // 60
        minute_in_hour = minute % 60
        hhmm = f"{hour:02d}:{minute_in_hour:02d}"
        muhurat_bars[hhmm] = {"open": 100.0, "close": 100.1}

    sessions = {
        regular_day: _returns_session(regular_day, 0.001, 0.0, 0.001)[regular_day],
        muhurat_day: muhurat_bars,
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert observations.n_sessions_dropped == 1
    assert observations.dates.tolist() == [regular_day]
    # Muhurat bars run 09:16..10:15: both 09:16 and 09:45 are present, so the first missing
    # checkpoint in order is 14:50 -- exactly one drop_reasons key increments.
    assert observations.drop_reasons == {"missing_14:50": 1}
    assert sum(observations.drop_reasons.values()) == observations.n_sessions_dropped


def test_nan_at_required_checkpoint_is_not_forward_filled() -> None:
    bad_day = datetime.date(2024, 4, 1)
    good_day = datetime.date(2024, 4, 2)
    sessions = {
        bad_day: {
            "09:16": {"open": 100.0, "close": 101.0},
            "09:45": {"open": 101.0, "close": np.nan},
            "14:50": {"open": 102.0, "close": 102.0},
            "15:20": {"open": 103.0, "close": 104.0},
        },
        good_day: _returns_session(good_day, 0.001, 0.0, 0.001)[good_day],
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert observations.n_sessions_dropped == 1
    assert observations.dates.tolist() == [good_day]
    # 09:16 is fully present; 09:45's close is NaN, so 09:45 is the first missing checkpoint.
    assert observations.drop_reasons == {"missing_09:45": 1}
    assert sum(observations.drop_reasons.values()) == observations.n_sessions_dropped
    assert np.isfinite(observations.r_first).all()


def test_sparse_late_start_and_mid_session_gap_resolve_by_time_label() -> None:
    day = datetime.date(2024, 5, 6)
    sessions = {
        day: {
            "09:16": {"open": 200.0, "close": 200.0},
            "09:45": {"open": 210.0, "close": 220.0},
            "10:30": {"open": 220.0, "close": 220.0},
            "12:17": {"open": 220.0, "close": 220.0},
            "14:50": {"open": 200.0, "close": 200.0},
            "15:20": {"open": 204.0, "close": 208.0},
        }
    }

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert observations.n_sessions_dropped == 0
    assert observations.dates.tolist() == [day]
    assert observations.r_first[0] == pytest.approx(np.log(220.0 / 200.0), rel=1e-7)
    assert observations.r_mid[0] == pytest.approx(np.log(200.0 / 220.0), rel=1e-7)
    assert observations.r_last[0] == pytest.approx(np.log(208.0 / 200.0), rel=1e-7)


def test_dates_are_sorted_unique_and_return_arrays_have_equal_lengths() -> None:
    days = [
        datetime.date(2024, 6, 5),
        datetime.date(2024, 6, 3),
        datetime.date(2024, 6, 4),
    ]
    sessions: dict[datetime.date, dict[str, dict[str, float]]] = {}
    for index, day in enumerate(days):
        sessions.update(
            _returns_session(day, 0.001 * (index + 1), 0.0, -0.001 * (index + 1))
        )

    observations = extract_session_observations(build_panel(sessions), "NIFTY50")

    assert observations.dates.dtype == object
    assert observations.dates.tolist() == sorted(set(days))
    assert len(observations.dates) == len(observations.r_first)
    assert len(observations.dates) == len(observations.r_mid)
    assert len(observations.dates) == len(observations.r_last)


def test_beta_matches_hand_computed_ols_slope_and_t_stat_sign() -> None:
    r_first = np.asarray([-0.004, -0.002, 0.001, 0.003, 0.006, 0.009], dtype=np.float64)
    r_mid = np.asarray([0.002, -0.001, 0.003, -0.002, 0.001, -0.003], dtype=np.float64)
    r_last = np.asarray([-0.0025, -0.0005, 0.0015, 0.001, 0.004, 0.005], dtype=np.float64)

    result = _one_result(r_first, r_last, r_mid=r_mid)
    x = r_first
    y = r_last
    expected_beta = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))

    assert result.beta == pytest.approx(expected_beta, rel=1e-5, abs=1e-9)
    assert np.sign(result.t_stat) == np.sign(result.beta)
    assert isinstance(result, H1Result)


def test_planted_half_coefficient_is_recovered() -> None:
    rng = np.random.default_rng(20260817)
    r_first = rng.normal(loc=0.0, scale=0.003, size=160)
    r_mid = rng.normal(loc=0.0, scale=0.002, size=160)
    noise = rng.normal(loc=0.0, scale=0.00035, size=160)
    r_last = 0.5 * r_first + noise

    result = _one_result(r_first, r_last, r_mid=r_mid)

    assert result.beta == pytest.approx(0.5, abs=0.08)
    assert result.t_stat > 1.96


def test_verified_driftless_pure_noise_is_not_significant_or_cost_surviving() -> None:
    # Seed pre-verified offline: with seed=20260817 this exact pure-noise fixture happens to
    # realize a spuriously "significant" t-stat (~2.34, chance tail event), which would make a
    # spec-correct implementation fail this test for reasons unrelated to correctness -- exactly
    # the kind of shared-fixture drift/luck this repo has been burned by before. seed=4 is
    # verified driftless AND non-significant for this exact construction.
    seed = 4
    rng = np.random.default_rng(seed)
    first_draw = rng.normal(loc=0.0, scale=0.003, size=512)
    last_draw = rng.normal(loc=0.0, scale=0.003, size=512)

    # Check the realized generator output before centering, so accidental drift in this
    # shared fixture fails here rather than masquerading as a statistical finding.
    assert abs(float(first_draw.mean())) < 0.0003
    assert abs(float(last_draw.mean())) < 0.0003

    r_first = first_draw - first_draw.mean()
    r_last = last_draw - last_draw.mean()
    r_mid = rng.normal(loc=0.0, scale=0.002, size=512)
    assert abs(float(r_first.mean())) < 1e-15
    assert abs(float(r_last.mean())) < 1e-15

    result = _one_result(r_first, r_last, r_mid=r_mid)

    assert abs(result.t_stat) < 1.96
    assert result.survives_costs is False


def test_mean_edge_is_the_sign_rule_quantity() -> None:
    rng = np.random.default_rng(314159)
    r_first = rng.normal(loc=0.0, scale=0.003, size=64)
    r_mid = rng.normal(loc=0.0, scale=0.002, size=64)
    r_last = 0.25 * r_first + rng.normal(loc=0.0, scale=0.0005, size=64)

    result = _one_result(r_first, r_last, r_mid=r_mid)
    panel = _panel_for_returns(r_first, r_mid, r_last)
    observations = extract_session_observations(panel, "NIFTY50")
    expected = float(1e4 * np.mean(np.sign(observations.r_first) * observations.r_last))

    assert result.mean_edge_bps == pytest.approx(expected, abs=1e-9)


def test_hit_rate_excludes_zero_first_return() -> None:
    r_first = np.asarray([0.01, -0.01, 0.0, 0.02], dtype=np.float64)
    r_mid = np.asarray([0.0, 0.001, -0.001, 0.002], dtype=np.float64)
    r_last = np.asarray([0.001, -0.001, 0.005, -0.001], dtype=np.float64)

    result = _one_result(r_first, r_last, r_mid=r_mid)

    assert result.hit_rate == pytest.approx(2.0 / 3.0)
    assert result.n_sessions == 4


def test_hit_rate_is_nan_when_no_session_trades() -> None:
    # Resolved 2026-08-17: if NO session trades (every r_first == 0), hit_rate must be nan,
    # not 0.0 -- 0.0 would falsely read as "traded and always wrong".
    # NOTE: an all-zero r_first is unavoidably a zero-variance regressor for beta/t_stat too
    # (this is a genuinely degenerate corner the spec does not otherwise address), so this
    # test intentionally asserts ONLY on hit_rate and leaves beta/t_stat/r_squared behaviour
    # on this input unspecified.
    r_first = np.zeros(3, dtype=np.float64)
    r_mid = np.asarray([-0.0005, 0.0002, 0.0009], dtype=np.float64)
    r_last = np.asarray([0.001, -0.002, 0.0015], dtype=np.float64)

    result = _one_result(r_first, r_last, r_mid=r_mid)

    assert np.isnan(result.hit_rate)
    assert result.n_sessions == 3


def test_survives_costs_requires_strictly_more_than_twice_hurdle() -> None:
    hurdle = 8.26452
    threshold = 2.0 * hurdle
    r_first = np.asarray([0.003, -0.003, 0.002, -0.002], dtype=np.float64)
    r_mid = np.asarray([0.001, -0.001, 0.002, -0.002], dtype=np.float64)

    below_edge_bps = threshold - 0.01
    above_edge_bps = threshold + 0.01
    below = _one_result(
        r_first,
        np.sign(r_first) * below_edge_bps / 1e4,
        r_mid=r_mid,
        cost_hurdle_bps=hurdle,
    )
    above = _one_result(
        r_first,
        np.sign(r_first) * above_edge_bps / 1e4,
        r_mid=r_mid,
        cost_hurdle_bps=hurdle,
    )

    # BPS_TOL calibrated against the measured float32-at-rest noise floor, not tightened back:
    # `below_edge_bps`/`above_edge_bps` are raw Python-float targets that never touch the
    # panel, but `mean_edge_bps` is read back off prices that were round-tripped through the
    # panel's float32-at-rest storage (CLAUDE.md rule 3). Measured on this exact fixture:
    # pure float64 target 16.53904 vs. panel-round-tripped 16.53918879797452 (delta ~1.5e-4
    # absolute on a ~16.5 bps value). abs=1e-4 sits inside that noise and is not a safe bound;
    # abs=1e-2 has real margin above it.
    BPS_TOL = 1e-2
    assert below.mean_edge_bps == pytest.approx(below_edge_bps, abs=BPS_TOL)
    assert above.mean_edge_bps == pytest.approx(above_edge_bps, abs=BPS_TOL)
    assert below.survives_costs is False
    assert above.survives_costs is True


# Statistical requirements: "a statistically significant beta with a sub-cost edge is a kill".
# seed=1 pre-verified offline (t_stat ~15.5, mean_edge_bps ~1.07), not a borderline/flaky draw.
def test_significant_beta_with_sub_cost_edge_is_a_kill() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    r_first = rng.normal(loc=0.0, scale=0.003, size=n)
    beta_true = 0.05
    noise = rng.normal(loc=0.0, scale=0.0004, size=n)
    r_last = beta_true * r_first + noise

    result = _one_result(r_first, r_last)

    assert abs(result.t_stat) > 1.96
    assert result.mean_edge_bps < 16.52904
    assert result.survives_costs is False
    assert result.beta > 0
    assert result.beta == pytest.approx(beta_true, abs=0.02)


def test_default_cost_hurdle_uses_nse_round_trip_cost_at_one_lakh() -> None:
    rng = np.random.default_rng(20260817)
    r_first = rng.normal(loc=0.0, scale=0.003, size=32)
    r_mid = rng.normal(loc=0.0, scale=0.002, size=32)
    r_last = rng.normal(loc=0.0, scale=0.003, size=32)

    result = _one_result(r_first, r_last, r_mid=r_mid)
    expected_hurdle = NSEIntradayEquityCosts().round_trip_bps(100_000.0)

    assert expected_hurdle == pytest.approx(8.26452, rel=1e-6)
    assert result.cost_hurdle_bps == pytest.approx(expected_hurdle, rel=1e-6)


def test_mid_only_generation_collapses_controlled_first_coefficient() -> None:
    # n=1000 at this seed was verified offline to give a huge margin (ratio ~0.0009 against
    # the 0.15 threshold below) -- the original n=500 draw only cleared it by ~5.6%, a thin,
    # seed-dependent margin that would look like a regression the first time the fixture
    # changed. n=1000 is comfortably robust, not just deterministically lucky.
    rng = np.random.default_rng(8675309)
    r_mid = rng.normal(loc=0.0, scale=0.003, size=1000)
    r_first = r_mid + rng.normal(loc=0.0, scale=0.0008, size=1000)
    r_last = 0.7 * r_mid + rng.normal(loc=0.0, scale=0.001, size=1000)

    result = _one_result(r_first, r_last, r_mid=r_mid)

    assert abs(result.beta) > 0.3
    assert abs(result.beta_controlling_for_mid) < 0.15 * abs(result.beta)
    assert abs(result.t_stat_controlling_for_mid) < abs(result.t_stat)


def test_first_only_generation_survives_mid_control_and_matches_bivariate_ols() -> None:
    rng = np.random.default_rng(424242)
    r_first = rng.normal(loc=0.0, scale=0.003, size=500)
    r_mid = rng.normal(loc=0.0, scale=0.003, size=500)
    r_last = 0.7 * r_first + rng.normal(loc=0.0, scale=0.0005, size=500)

    result = _one_result(r_first, r_last, r_mid=r_mid)
    panel = _panel_for_returns(r_first, r_mid, r_last)
    observations = extract_session_observations(panel, "NIFTY50")
    design = np.column_stack(
        [
            np.ones(len(observations.r_first)),
            observations.r_first,
            observations.r_mid,
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, observations.r_last, rcond=None)

    assert result.beta_controlling_for_mid == pytest.approx(float(coefficients[1]), rel=1e-6)
    assert abs(result.beta_controlling_for_mid) > 0.5 * abs(result.beta)
    assert result.t_stat_controlling_for_mid > 1.96


def test_by_year_partitions_sessions_once_and_counts_dominant_sign() -> None:
    sessions: dict[datetime.date, dict[str, dict[str, float]]] = {}
    year_edges = {
        2018: 20.0,
        2019: 15.0,
        2020: 10.0,
        2021: 12.0,
        2022: 8.0,
        2023: 18.0,
        2024: -5.0,
        2025: 11.0,
    }

    for year, edge_bps in year_edges.items():
        for index in range(4):
            day = datetime.date(year, 1, 2) + datetime.timedelta(days=index)
            first = 0.003 if index % 2 == 0 else -0.003
            last = np.sign(first) * edge_bps / 1e4
            sessions.update(_returns_session(day, first, 0.0005 * (index - 1), last))

    panel = build_panel(sessions)
    observations = extract_session_observations(panel, "NIFTY50")
    result = run_h1(panel, "NIFTY50")

    assert result.n_sessions == 32
    assert set(result.by_year) == set(year_edges)
    assert sum(result.by_year.values()) != 0.0
    assert result.years_sign_consistent == 7

    for year in year_edges:
        indices = np.asarray([day.year == year for day in observations.dates])
        expected = float(
            1e4
            * np.mean(np.sign(observations.r_first[indices]) * observations.r_last[indices])
        )
        assert result.by_year[year] == pytest.approx(expected, abs=1e-5)


def test_effect_present_in_only_one_year_is_visible_in_consistency_count() -> None:
    sessions: dict[datetime.date, dict[str, dict[str, float]]] = {}

    for year in range(2018, 2026):
        edge_bps = 20.0 if year == 2018 else 0.0
        for index in range(3):
            day = datetime.date(year, 2, 1) + datetime.timedelta(days=index)
            first = 0.002 if index % 2 == 0 else -0.002
            last = np.sign(first) * edge_bps / 1e4
            sessions.update(_returns_session(day, first, 0.0007 * (index - 1), last))

    result = run_h1(build_panel(sessions), "NIFTY50")

    assert set(result.by_year) == set(range(2018, 2026))
    assert result.by_year[2018] > 0.0
    for year in range(2019, 2026):
        assert result.by_year[year] == pytest.approx(0.0, abs=1e-6)
    assert result.years_sign_consistent == 1


def test_explain_and_markdown_report_all_kill_criteria_and_status() -> None:
    rng = np.random.default_rng(20260817)
    r_first = rng.normal(loc=0.0, scale=0.003, size=80)
    r_mid = rng.normal(loc=0.0, scale=0.002, size=80)
    r_last = 0.1 * r_first + rng.normal(loc=0.0, scale=0.001, size=80)
    result = _one_result(r_first, r_last, r_mid=r_mid)

    report = f"{result.explain()}\n{result.to_markdown()}"
    lower = report.lower()

    assert "nifty50" in lower
    assert str(result.n_sessions) in report
    assert "cost" in lower
    assert "hurdle" in lower
    assert "year" in lower
    assert "t-stat" in lower or "t stat" in lower or "t_stat" in lower
    assert "1.96" in lower
    assert "mid" in lower or "control" in lower or "confound" in lower
    assert "50%" in report or "50 %" in report or "0.5" in lower
    assert "SURVIVED" in report or "KILLED" in report


def test_run_h1_is_deterministic_and_records_the_fixture_seed() -> None:
    fixture_seed = 20260817
    rng = np.random.default_rng(fixture_seed)
    r_first = rng.normal(loc=0.0, scale=0.003, size=96)
    r_mid = rng.normal(loc=0.0, scale=0.002, size=96)
    r_last = 0.2 * r_first + rng.normal(loc=0.0, scale=0.001, size=96)

    panel = _panel_for_returns(r_first, r_mid, r_last)
    result_a = run_h1(panel, "NIFTY50")
    result_b = run_h1(panel, "NIFTY50")

    # H1Result has no seed field in the specified public API; the literal seed is recorded
    # alongside this deterministic fixture, and identical inputs must produce equal results.
    assert fixture_seed == 20260817
    assert result_a == result_b
    assert result_a.explain() == result_b.explain()
    assert result_a.to_markdown() == result_b.to_markdown()


def test_run_h1_docstring_states_why_no_overlap_correction_is_used() -> None:
    # This guards against adding a block-bootstrap or horizon correction: one observation
    # per session means the session observations do not overlap.
    docstring = run_h1.__doc__ or ""
    lower = docstring.lower()

    assert "overlap" in lower
    assert "session" in lower
    assert "observation" in lower


def test_run_h1_respects_pre_holdout_date_range_without_reading_holdout(
    tmp_path: Path,
) -> None:
    trading_dates = [
        datetime.date(year, month, 1)
        for year in range(2018, 2021)
        for month in range(1, 13)
    ]
    # r_mid must not be constant across sessions: a constant column makes the bivariate
    # r_last ~ a + b1*r_first + b2*r_mid regression singular (LinAlgError) for any correct
    # implementation, so give it small seeded per-session variation instead of a fixed 0.0002.
    mid_rng = np.random.default_rng(2026)
    mid_noise = mid_rng.normal(loc=0.0002, scale=0.00005, size=len(trading_dates))
    sessions: dict[datetime.date, dict[str, dict[str, float]]] = {}
    for index, day in enumerate(trading_dates):
        first = 0.001 if index % 2 == 0 else -0.001
        sessions.update(
            _returns_session(day, first, float(mid_noise[index]), 0.0005 * np.sign(first))
        )

    panel = build_panel(sessions)
    lock = HoldoutLock(path=tmp_path / "h1-luna-holdout-lock.json", holdout_months=12)
    holdout_start, holdout_end = lock.holdout_range(trading_dates)
    pre_holdout_end = holdout_start - datetime.timedelta(days=1)
    pre_dates = [day for day in trading_dates if day <= pre_holdout_end]

    result = run_h1(
        panel,
        "NIFTY50",
        start=trading_dates[0],
        end=pre_holdout_end,
    )
    pre_observations = extract_session_observations(
        build_panel({day: sessions[day] for day in pre_dates}),
        "NIFTY50",
    )

    assert holdout_start <= holdout_end
    assert result.n_sessions == len(pre_dates)
    assert all(day < holdout_start for day in pre_observations.dates)
    assert all(day > holdout_end or day < holdout_start for day in pre_observations.dates)
    assert set(result.by_year) == {day.year for day in pre_dates}
