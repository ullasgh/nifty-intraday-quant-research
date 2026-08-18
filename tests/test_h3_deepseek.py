"""TDD red tests for H3 (cross-sectional intraday reversal), from
specs/h3_intraday_xsec_reversal.md, before src/nifty_quant/research/hypotheses/
h3_intraday_xsec_reversal.py exists.

Written independently of tests/test_h3_luna.py, per the spec's instruction that two
independent readings are the point -- this file must not import from, read, or match
against that sibling suite.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research import expectancy as expectancy_module
from nifty_quant.research.hypotheses.h3_intraday_xsec_reversal import (
    build_morning_residual_feature,
    run_h3,
)
from nifty_quant.research.lens import HypothesisVerdict


def _epoch_seconds(session_date: datetime.date, hhmm: str) -> int:
    """Left-labelled epoch-second UTC timestamp for an IST wall-clock HH:MM."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.datetime(
        session_date.year, session_date.month, session_date.day, hour, minute
    )
    utc_naive = local - datetime.timedelta(hours=5, minutes=30)
    return int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())


def _bar(
    open_: float,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
) -> dict[str, float]:
    c = open_ if close is None else close
    with np.errstate(invalid="ignore"):
        h = max(open_, c) + 0.01 if high is None else high
        lo = min(open_, c) - 0.01 if low is None else low
    return {"open": open_, "high": h, "low": lo, "close": c}


def _build_panel(
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]],
    default_volume: float = 1_000_000.0,
) -> Panel:
    symbols = tuple(bars_by_symbol.keys())
    per_symbol_by_ts: dict[str, dict[int, dict[str, float]]] = {}
    all_ts: set[int] = set()
    for sym, bars in bars_by_symbol.items():
        by_ts: dict[int, dict[str, float]] = {}
        for session_date, hhmm, ohlc in bars:
            ts = _epoch_seconds(session_date, hhmm)
            by_ts[ts] = ohlc
            all_ts.add(ts)
        per_symbol_by_ts[sym] = by_ts

    ts_sorted = np.array(sorted(all_ts), dtype=np.int64)
    n_rows = ts_sorted.shape[0]
    n_symbols = len(symbols)

    fields: dict[str, np.ndarray] = {
        name: np.full((n_rows, n_symbols), np.nan, dtype=np.float32)
        for name in ("open", "high", "low", "close")
    }
    volume = np.full((n_rows, n_symbols), default_volume, dtype=np.float32)

    for j, sym in enumerate(symbols):
        by_ts = per_symbol_by_ts[sym]
        for i, ts in enumerate(ts_sorted):
            ohlc = by_ts.get(int(ts))
            if ohlc is None:
                volume[i, j] = np.nan
                continue
            for name in ("open", "high", "low", "close"):
                fields[name][i, j] = ohlc[name]
    fields["volume"] = volume

    grid = SessionGrid.from_timestamps(ts_sorted)
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _checkpoint_ramp_panel(
    symbols: tuple[str, ...],
    n_sessions: int,
    start_date: datetime.date,
    ramp: np.ndarray,
    k: float,
    noise_scale: float,
    seed: int,
) -> Panel:
    """n_sessions independent 3-bar (09:16 open, 10:00 close, 15:20 close) sessions, one
    calendar day apart. Every session's r_morning[j] == ramp[j] EXACTLY (the 10:00 close
    is derived from a constant 100.0 09:16 open via ramp[j]) -- unlike H2's overnight
    feature, H3 has no cross-day carry, so there is no "first session is NaN" special
    case. The afternoon leg is planted as r_rest[j] = k * ramp[j] + noise, so k's sign
    controls the top-minus-bottom spread sign deterministically: k<0 -> reversal
    (negative spread), k>0 -> momentum (positive spread). With exactly 2 checkpoint rows
    per session after reduction, horizon=1 from the 10:00 row is exactly the 15:20
    close, so spread_bps is hand/numerically-verifiable given ramp/k, and only the
    SE/t-stat depend on noise_scale."""
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    for day_i in range(n_sessions):
        d = start_date + datetime.timedelta(days=day_i)
        open_0916 = {sym: 100.0 for sym in symbols}
        noise = rng.standard_normal(n_symbols) * noise_scale
        for j, sym in enumerate(symbols):
            o = open_0916[sym]
            c1000 = o * math.exp(float(ramp[j]))
            r_rest = k * float(ramp[j]) + float(noise[j])
            c1520 = c1000 * math.exp(r_rest)
            bars_by_symbol[sym].append((d, "09:16", _bar(o)))
            bars_by_symbol[sym].append((d, "10:00", _bar(open_=c1000, close=c1000)))
            bars_by_symbol[sym].append((d, "15:20", _bar(open_=c1000, close=c1520)))
    return _build_panel(bars_by_symbol)


def _yearly_reversal_panel(
    symbols: tuple[str, ...],
    ramp: np.ndarray,
    year_k: dict[int, float],
    sessions_per_year: int,
    noise_scale: float,
    seed: int,
) -> Panel:
    """Same 3-bar-per-session construction as `_checkpoint_ramp_panel`, but with a
    per-year reversal coefficient `year_k[year]`, one continuous rng stream across the
    whole panel, weekly sessions starting Jan 15 of each year."""
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    for year in sorted(year_k):
        k = year_k[year]
        for j in range(sessions_per_year):
            d = datetime.date(year, 1, 15) + datetime.timedelta(days=7 * j)
            open_0916 = {sym: 100.0 for sym in symbols}
            noise = rng.standard_normal(n_symbols) * noise_scale
            for i, sym in enumerate(symbols):
                o = open_0916[sym]
                c1000 = o * math.exp(float(ramp[i]))
                r_rest = k * float(ramp[i]) + float(noise[i])
                c1520 = c1000 * math.exp(r_rest)
                bars_by_symbol[sym].append((d, "09:16", _bar(o)))
                bars_by_symbol[sym].append((d, "10:00", _bar(open_=c1000, close=c1000)))
                bars_by_symbol[sym].append((d, "15:20", _bar(open_=c1000, close=c1520)))
    return _build_panel(bars_by_symbol)


def _realistic_dense_session_bars(
    d: datetime.date,
    symbols: tuple[str, ...],
    open_0916: dict[str, float],
    close_1000: dict[str, float],
    close_1520: dict[str, float],
    rng: np.random.Generator,
    drop_frac: float,
) -> dict[str, list[tuple[datetime.date, str, dict[str, float]]]]:
    """One session's DENSE 1-minute bars from 09:15 to 15:20 inclusive (366 minutes,
    squarely in the real ~100-375 bars/session range), per symbol. Only the 09:16,
    10:00, and 15:20 minutes carry the session's true checkpoint prices; every other
    minute is a flat filler bar (doji at the most recently established checkpoint
    price) -- present so a POSITIONALLY-resolving implementation reads the wrong bar,
    but carrying no information of its own, so it cannot manufacture a spurious signal.
    `drop_frac` of the non-checkpoint minutes are removed entirely (no bar for any
    symbol at that minute -- a real missing print), varying the realized bar count per
    session and proving resolution is by IST time label, never a fixed row count."""
    out: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    all_minutes = list(range(9 * 60 + 15, 15 * 60 + 20 + 1))
    checkpoint_minutes = {9 * 60 + 16: "0916", 10 * 60 + 0: "1000", 15 * 60 + 20: "1520"}
    droppable = [m for m in all_minutes if m not in checkpoint_minutes]
    n_drop = int(len(droppable) * drop_frac)
    dropped: set[int] = (
        set(rng.choice(droppable, size=n_drop, replace=False).tolist()) if n_drop else set()
    )
    for sym in symbols:
        current_price = open_0916[sym]
        for m in all_minutes:
            if m in dropped:
                continue
            hour, minute = m // 60, m % 60
            hhmm = f"{hour:02d}:{minute:02d}"
            if m in checkpoint_minutes:
                tag = checkpoint_minutes[m]
                price = {
                    "0916": open_0916[sym], "1000": close_1000[sym], "1520": close_1520[sym],
                }[tag]
                out[sym].append((d, hhmm, _bar(price)))
                current_price = price
            else:
                out[sym].append((d, hhmm, _bar(current_price)))
    return out


def _realistic_and_checkpoint_panels(
    symbols: tuple[str, ...],
    n_sessions: int,
    start_date: datetime.date,
    ramp: np.ndarray,
    k: float,
    noise_scale: float,
    seed: int,
) -> tuple[Panel, Panel]:
    """Returns (realistic_panel, minimal_checkpoint_panel) built from the IDENTICAL
    underlying r_morning=ramp[j], r_rest=k*ramp[j]+noise economics (same rng draw order
    as `_checkpoint_ramp_panel` with the same seed), so a CORRECT run_h3 must produce
    numerically identical verdicts on both -- the realistic panel additionally carries
    ~250-370 filler/gap bars per session (verified: 258-366 bars/session at drop_frac
    0.3 on 1/3 of sessions) that a correct BY-TIME-LABEL checkpoint reduction must
    ignore entirely. Verified numerically before adoption at
    (symbols=8, n_sessions=150, ramp=linspace(-0.01,0.01,8), k=-1.0, noise_scale=0.002,
    seed=111): both panels give spread_bps ~ -172.73 bps, t ~ -89.0, n_total=150,
    identical to >= 1e-6 bps between the two panels."""
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    realistic_bars: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    minimal_bars: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    for day_i in range(n_sessions):
        d = start_date + datetime.timedelta(days=day_i)
        open_0916 = {sym: 100.0 for sym in symbols}
        noise = rng.standard_normal(n_symbols) * noise_scale
        close_1000: dict[str, float] = {}
        close_1520: dict[str, float] = {}
        for j, sym in enumerate(symbols):
            c1000 = open_0916[sym] * math.exp(float(ramp[j]))
            r_rest = k * float(ramp[j]) + float(noise[j])
            c1520 = c1000 * math.exp(r_rest)
            close_1000[sym] = c1000
            close_1520[sym] = c1520
            minimal_bars[sym].append((d, "09:16", _bar(open_0916[sym])))
            minimal_bars[sym].append((d, "10:00", _bar(open_=c1000, close=c1000)))
            minimal_bars[sym].append((d, "15:20", _bar(open_=c1000, close=c1520)))
        drop_frac = 0.3 if day_i % 3 == 0 else 0.0
        session_bars = _realistic_dense_session_bars(
            d, symbols, open_0916, close_1000, close_1520, rng, drop_frac
        )
        for sym in symbols:
            realistic_bars[sym].extend(session_bars[sym])
    return _build_panel(realistic_bars), _build_panel(minimal_bars)


# ---------------------------------------------------------------------------
# Checkpoint construction
# ---------------------------------------------------------------------------


def test_r_morning_hand_computed_via_symmetric_two_symbol_fixture() -> None:
    d0 = datetime.date(2021, 1, 1)
    raw_a = math.log(102.0 / 100.0)
    close_b_1000 = 50.0 * math.exp(-raw_a)
    bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=102.0, close=102.0)),
        ],
        "B": [
            (d0, "09:16", _bar(50.0)),
            (d0, "10:00", _bar(open_=close_b_1000, close=close_b_1000)),
        ],
    }
    panel = _build_panel(bars)
    feat = build_morning_residual_feature(panel)
    a_ix = panel.sym_ix["A"]
    row = int(panel.day_offsets[0])

    # abs=1e-6, not 1e-9: prices round-trip through the panel's float32-at-rest
    # storage before this comparison, so a ~1e-7 relative rounding error on a ~100.0
    # price level is expected here, not a bug.
    assert feat.values[row, a_ix] == pytest.approx(raw_a, abs=1e-6)


def test_telescoping_identity_r_morning_plus_r_rest_equals_full_day_log_return() -> None:
    d0 = datetime.date(2021, 1, 1)
    raw_a = math.log(102.0 / 100.0)
    close_b_1000 = 50.0 * math.exp(-raw_a)
    close_a_1520 = 105.0
    bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=102.0, close=102.0)),
            (d0, "15:20", _bar(open_=102.0, close=close_a_1520)),
        ],
        "B": [
            (d0, "09:16", _bar(50.0)),
            (d0, "10:00", _bar(open_=close_b_1000, close=close_b_1000)),
        ],
    }
    panel = _build_panel(bars)
    feat = build_morning_residual_feature(panel)
    a_ix = panel.sym_ix["A"]
    row = int(panel.day_offsets[0])

    feature_value_a = feat.values[row, a_ix]
    r_rest_a = math.log(close_a_1520 / 102.0)
    # Both legs share the 10:00 close by construction. abs=1e-6, not 1e-9: prices
    # round-trip through the panel's float32-at-rest storage before this comparison.
    assert feature_value_a + r_rest_a == pytest.approx(
        math.log(close_a_1520 / 100.0), abs=1e-6
    )


def test_0915_never_read_bit_identical_when_0915_present_and_corrupt() -> None:
    d0 = datetime.date(2020, 1, 1)
    raw_a = math.log(102.0 / 100.0)
    close_b_1000 = 50.0 * math.exp(-raw_a)

    baseline_bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=102.0, close=102.0)),
        ],
        "B": [
            (d0, "09:16", _bar(50.0)),
            (d0, "10:00", _bar(open_=close_b_1000, close=close_b_1000)),
        ],
    }
    corrupted_bars = {
        "A": [
            *baseline_bars["A"],
            (d0, "09:15", _bar(open_=50.0, high=50.2, low=49.8, close=95.0)),
        ],
        "B": baseline_bars["B"],
    }

    baseline_panel = _build_panel(baseline_bars)
    corrupted_panel = _build_panel(corrupted_bars)

    baseline_feat = build_morning_residual_feature(baseline_panel)
    corrupted_feat = build_morning_residual_feature(corrupted_panel)

    baseline_val = baseline_feat.values[
        int(baseline_panel.day_offsets[0]), baseline_panel.sym_ix["A"]
    ]
    corrupted_val = corrupted_feat.values[
        int(corrupted_panel.day_offsets[0]), corrupted_panel.sym_ix["A"]
    ]
    assert corrupted_val == baseline_val
    assert not math.isnan(baseline_val)


def test_1001_wild_bar_ignored_resolution_by_time_label() -> None:
    d0 = datetime.date(2020, 1, 1)
    raw_a = math.log(102.0 / 100.0)
    close_b_1000 = 50.0 * math.exp(-raw_a)

    baseline_bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=102.0, close=102.0)),
        ],
        "B": [
            (d0, "09:16", _bar(50.0)),
            (d0, "10:00", _bar(open_=close_b_1000, close=close_b_1000)),
        ],
    }
    wild_bars = {
        "A": [
            *baseline_bars["A"],
            (d0, "10:01", _bar(open_=6000.0, close=6000.0)),
        ],
        "B": baseline_bars["B"],
    }

    baseline_panel = _build_panel(baseline_bars)
    wild_panel = _build_panel(wild_bars)

    baseline_feat = build_morning_residual_feature(baseline_panel)
    wild_feat = build_morning_residual_feature(wild_panel)

    baseline_val = baseline_feat.values[
        int(baseline_panel.day_offsets[0]), baseline_panel.sym_ix["A"]
    ]
    wild_val = wild_feat.values[
        int(wild_panel.day_offsets[0]), wild_panel.sym_ix["A"]
    ]
    assert wild_val == baseline_val


def test_muhurat_session_drops_out_of_checkpoint_reduction() -> None:
    d0 = datetime.date(2020, 1, 1)  # normal
    muhurat = datetime.date(2020, 1, 2)  # ~60 bars, no 15:20
    d2 = datetime.date(2020, 1, 3)  # normal

    bars_list: list[tuple[datetime.date, str, dict[str, float]]] = [
        (d0, "09:16", _bar(100.0)),
        (d0, "10:00", _bar(open_=101.0, close=101.0)),
        (d0, "15:20", _bar(open_=101.0, close=102.0)),
    ]
    for minute in range(9 * 60 + 16, 10 * 60 + 15 + 1):
        hour, minute_part = minute // 60, minute % 60
        bars_list.append((muhurat, f"{hour:02d}:{minute_part:02d}", _bar(999.0)))
    bars_list += [
        (d2, "09:16", _bar(200.0)),
        (d2, "10:00", _bar(open_=201.0, close=201.0)),
        (d2, "15:20", _bar(open_=201.0, close=202.0)),
    ]

    panel = _build_panel({"SYM": bars_list})
    result: HypothesisVerdict = run_h3(panel, seed=0)

    assert result.expectancy.n_total == 2


def test_per_symbol_nan_independent_of_other_symbols_same_session() -> None:
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)
    d2 = datetime.date(2020, 1, 3)
    bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=101.0, close=101.0)),
            (d1, "09:16", _bar(102.0)),
            (d1, "10:00", _bar(open_=103.0, close=103.0)),
            (d2, "09:16", _bar(104.0)),
            (d2, "10:00", _bar(open_=105.0, close=105.0)),
        ],
        "B": [
            (d0, "09:16", _bar(200.0)),
            (d0, "10:00", _bar(open_=201.0, close=201.0)),
            # B has no bars at all on d1.
            (d2, "09:16", _bar(210.0)),
            (d2, "10:00", _bar(open_=211.0, close=211.0)),
        ],
    }
    panel = _build_panel(bars)
    feat = build_morning_residual_feature(panel)
    a_ix = panel.sym_ix["A"]
    b_ix = panel.sym_ix["B"]
    d1_row = int(panel.day_offsets[1])

    assert not math.isnan(feat.values[d1_row, a_ix])
    assert math.isnan(feat.values[d1_row, b_ix])


def test_nan_checkpoint_propagates_without_forward_fill() -> None:
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)
    d2 = datetime.date(2020, 1, 3)
    bars = {
        "SYM": [
            (d0, "09:16", _bar(100.0)),
            (d0, "10:00", _bar(open_=101.0, close=101.0)),
            (d0, "15:20", _bar(open_=101.0, close=102.0)),
            (d1, "09:16", _bar(open_=float("nan"))),
            (d1, "10:00", _bar(open_=103.0, close=103.0)),
            (d1, "15:20", _bar(open_=103.0, close=104.0)),
            (d2, "09:16", _bar(105.0)),
            (d2, "10:00", _bar(open_=106.0, close=106.0)),
            (d2, "15:20", _bar(open_=106.0, close=107.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_morning_residual_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    d0_row = int(panel.day_offsets[0])
    d1_row = int(panel.day_offsets[1])
    d2_row = int(panel.day_offsets[2])

    assert not math.isnan(feat.values[d0_row, sym_ix])
    assert math.isnan(feat.values[d1_row, sym_ix])
    assert not math.isnan(feat.values[d2_row, sym_ix])


# ---------------------------------------------------------------------------
# Realistic session (the critical regression test)
# ---------------------------------------------------------------------------


def test_realistic_multi_bar_session_matches_two_bar_checkpoint_equivalent() -> None:
    """This is the H2 regression: without checkpoint-panel reduction, horizon=1 on the
    realistic (many-bar) panel would measure a ~1-minute return almost everywhere instead
    of the 10:00->15:20 session return, and would silently return a plausible-looking but
    wrong (near-zero) number rather than error -- verified numerically before adoption
    (see the fixture's own docstring for the exact numbers)."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    realistic_panel, minimal_panel = _realistic_and_checkpoint_panels(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-1.0, noise_scale=0.002, seed=111,
    )
    realistic_result: HypothesisVerdict = run_h3(realistic_panel, seed=1)
    minimal_result: HypothesisVerdict = run_h3(minimal_panel, seed=1)

    assert realistic_result.expectancy.spread_bps == pytest.approx(
        minimal_result.expectancy.spread_bps, abs=1e-2
    )
    assert realistic_result.expectancy.spread_t == pytest.approx(
        minimal_result.expectancy.spread_t, abs=1e-1
    )
    assert realistic_result.expectancy.n_total == minimal_result.expectancy.n_total
    assert realistic_result.expectancy.n_total == 150


# ---------------------------------------------------------------------------
# Cross-sectional behaviour
# ---------------------------------------------------------------------------


def test_demeaning_makes_each_row_sum_to_approximately_zero() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=10, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=42,
    )
    feat = build_morning_residual_feature(panel)
    first_session_row = int(panel.day_offsets[0])

    assert np.nansum(feat.values[first_session_row, :]) == pytest.approx(0.0, abs=1e-9)


def test_planted_reversal_recovered_with_negative_spread() -> None:
    """Verified numerically before adoption: spread ~ -172.86 bps, t ~ -86.3."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-1.0, noise_scale=0.002, seed=111,
    )
    result: HypothesisVerdict = run_h3(panel, seed=1)

    assert result.expectancy.spread_bps < 0.0
    assert abs(result.expectancy.spread_t) > 1.96


def test_planted_momentum_yields_positive_spread_and_explain_states_direction() -> None:
    """Verified numerically before adoption: spread ~ +170.12 bps. H3's real empirical
    answer is a POSITIVE (momentum) spread and criterion 1 gates on abs(spread), so both
    directions must be nameable, not just momentum."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))

    momentum_panel = _checkpoint_ramp_panel(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=1.0, noise_scale=0.002, seed=222,
    )
    momentum_result: HypothesisVerdict = run_h3(momentum_panel, seed=2)
    assert momentum_result.expectancy.spread_bps > 0.0
    assert "momentum" in momentum_result.explain().lower()

    reversal_panel = _checkpoint_ramp_panel(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-1.0, noise_scale=0.002, seed=111,
    )
    reversal_result: HypothesisVerdict = run_h3(reversal_panel, seed=1)
    assert "reversal" in reversal_result.explain().lower()


def test_pure_noise_yields_insignificant_spread_and_not_survived() -> None:
    """A driftless generator can still hand an unlucky draw, so the realized t-stat was
    checked before being asserted, not assumed. Verified numerically before adoption at
    this exact seed pair: t ~ 0.28."""
    n_sessions, n_symbols, scale = 200, 8, 0.01
    rng_morning = np.random.default_rng(90001)
    rng_rest = np.random.default_rng(90002)
    r_morning = rng_morning.standard_normal((n_sessions, n_symbols)) * scale
    r_rest = rng_rest.standard_normal((n_sessions, n_symbols)) * scale

    se_morning = r_morning.std(ddof=1) / np.sqrt(r_morning.size)
    se_rest = r_rest.std(ddof=1) / np.sqrt(r_rest.size)
    assert abs(r_morning.mean()) < 5 * se_morning
    assert abs(r_rest.mean()) < 5 * se_rest

    symbols = tuple(f"SYM{i}" for i in range(n_symbols))
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    start_date = datetime.date(2021, 1, 1)
    for day_i in range(n_sessions):
        d = start_date + datetime.timedelta(days=day_i)
        for j, sym in enumerate(symbols):
            o = 100.0
            c1000 = o * math.exp(float(r_morning[day_i, j]))
            c1520 = c1000 * math.exp(float(r_rest[day_i, j]))
            bars_by_symbol[sym].append((d, "09:16", _bar(o)))
            bars_by_symbol[sym].append((d, "10:00", _bar(open_=c1000, close=c1000)))
            bars_by_symbol[sym].append((d, "15:20", _bar(open_=c1000, close=c1520)))
    panel = _build_panel(bars_by_symbol)

    result: HypothesisVerdict = run_h3(panel, seed=3)

    assert abs(result.expectancy.spread_t) < 1.96
    assert result.survived is False


def test_planted_reversal_below_hurdle_is_killed_despite_significance() -> None:
    """Verified numerically before adoption: spread ~ 14.08 bps (< 2x hurdle ~16.53 bps),
    t ~ 47.8."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=300, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=0.08, noise_scale=0.0005, seed=333,
    )
    result: HypothesisVerdict = run_h3(panel, seed=4)

    hurdle = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert abs(result.expectancy.spread_bps) < 2 * hurdle
    assert abs(result.expectancy.spread_t) > 1.96
    assert result.survived is False


# ---------------------------------------------------------------------------
# Lens integration
# ---------------------------------------------------------------------------


def test_verdict_reports_all_seven_criteria_even_after_a_failure() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=60, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=0.0, noise_scale=0.002, seed=101,
    )
    result: HypothesisVerdict = run_h3(panel, seed=0)

    for n in range(1, 8):
        assert any(r.startswith(f"{n}.") for r in result.reasons), f"missing criterion {n}"


def test_criterion_5_is_not_evaluated_and_never_silently_passes() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=60, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=102,
    )
    result: HypothesisVerdict = run_h3(panel, seed=0)

    c5_lines = [r for r in result.reasons if r.startswith("5.")]
    assert len(c5_lines) == 1
    assert "NOT_EVALUATED" in c5_lines[0]
    assert "PASS" not in c5_lines[0]


def test_criterion_7_recent_years_gate_present_and_evaluated() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    year_k = {y: (-0.6 if y % 2 == 0 else -0.5) for y in range(2018, 2026)}
    panel = _yearly_reversal_panel(
        symbols, ramp, year_k, sessions_per_year=25, noise_scale=0.001, seed=42,
    )
    result: HypothesisVerdict = run_h3(panel, seed=0)

    c7_lines = [r for r in result.reasons if r.startswith("7.")]
    assert len(c7_lines) == 1
    assert "NOT_EVALUATED" not in c7_lines[0]


def test_run_h3_expectancy_equals_direct_conditional_expectancy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=60, start_date=datetime.date(2021, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=77,
    )

    calls: list[expectancy_module.ExpectancyTable] = []
    real_fn = expectancy_module.conditional_expectancy

    def _spy(*args: object, **kwargs: object) -> expectancy_module.ExpectancyTable:
        table = real_fn(*args, **kwargs)  # type: ignore[arg-type]
        calls.append(table)
        return table

    monkeypatch.setattr(expectancy_module, "conditional_expectancy", _spy)

    result: HypothesisVerdict = run_h3(panel, seed=3)

    assert len(calls) >= 1
    assert result.expectancy == calls[0]


# ---------------------------------------------------------------------------
# Reporting and hygiene
# ---------------------------------------------------------------------------


def test_output_contains_survivorship_warning_line() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=40, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=103,
    )
    result: HypothesisVerdict = run_h3(panel, seed=0)
    assert "UNIVERSE" in result.explain()


def test_start_end_restricts_window_to_strictly_fewer_sessions() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=80, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=104,
    )
    full: HypothesisVerdict = run_h3(panel, seed=0)
    cutoff = panel.dates[40]
    restricted: HypothesisVerdict = run_h3(panel, seed=0, end=cutoff)

    assert restricted.expectancy.n_total < full.expectancy.n_total


def test_run_h3_is_deterministic_and_records_seed() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=50, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=105,
    )
    result_a: HypothesisVerdict = run_h3(panel, seed=42)
    result_b: HypothesisVerdict = run_h3(panel, seed=42)

    assert result_a.seed == 42
    assert result_b.seed == 42
    assert result_a.survived == result_b.survived
    assert result_a.expectancy.spread_bps == result_b.expectancy.spread_bps
    assert result_a.expectancy.spread_t == result_b.expectancy.spread_t
    assert result_a.expectancy.n_total == result_b.expectancy.n_total
    assert (
        result_a.stability.n_years_sign_consistent
        == result_b.stability.n_years_sign_consistent
    )
    assert result_a.reasons == result_b.reasons


def test_default_cost_hurdle_resolved_from_cost_model_not_hardcoded() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _checkpoint_ramp_panel(
        symbols, n_sessions=40, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=106,
    )
    result: HypothesisVerdict = run_h3(panel, seed=0)

    expected_hurdle = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert result.cost_hurdle_bps == pytest.approx(expected_hurdle, rel=1e-9)
    assert result.expectancy.cost_hurdle_bps == pytest.approx(expected_hurdle, rel=1e-9)
    assert result.cost_hurdle_bps == pytest.approx(8.26452, abs=1e-2)
