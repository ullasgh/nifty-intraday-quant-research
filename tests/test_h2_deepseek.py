"""TDD red tests for H2 (cross-sectional overnight reversal), from
specs/h2_overnight_reversal.md, before src/nifty_quant/research/hypotheses/
h2_overnight_reversal.py exists.

Written independently of tests/test_h2_luna.py, per the spec's instruction that two
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
from nifty_quant.features.market import overnight_return
from nifty_quant.research import expectancy as expectancy_module
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    build_overnight_feature,
    run_h2,
)
from nifty_quant.research.lens import HypothesisVerdict

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


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
    """Build an OHLC dict; high/low default to bracket open/close, close defaults to
    open (a doji), so the caller only has to spell out what actually matters."""
    c = open_ if close is None else close
    with np.errstate(invalid="ignore"):
        h = max(open_, c) + 0.01 if high is None else high
        lo = min(open_, c) - 0.01 if low is None else low
    return {"open": open_, "high": h, "low": lo, "close": c}


def _build_panel(
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]],
    default_volume: float = 1_000_000.0,
) -> Panel:
    """Multi-symbol dense panel builder. The shared row grid is the UNION of every
    symbol's timestamps; a symbol with no bar at a given shared timestamp gets NaN
    OHLCV at that row -- this is how a per-symbol gap (symbol B absent on a day symbol
    A traded) is represented in a dense (n_rows, n_symbols) panel (CLAUDE.md rules 6/7:
    `present` is per-symbol, never forward-filled)."""
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


def _ramp_panel(
    symbols: tuple[str, ...],
    n_sessions: int,
    start_date: datetime.date,
    ramp: np.ndarray,
    k: float,
    noise_scale: float,
    seed: int,
) -> Panel:
    """`n_sessions` consecutive 2-bar (09:16 open, 15:20 close) sessions, one calendar
    day apart. Every session after the first has `r_on[j] == ramp[j]` EXACTLY -- each
    day's 09:16 open is derived from the PRIOR day's 15:20 close via `ramp[j]`, so the
    overnight-return feature's cross-sectional rank ordering across symbols is
    identical, session to session, by construction. The intraday leg is planted as
    `r_id[j] = k * ramp[j] + noise`, so `k`'s sign controls the top-minus-bottom spread
    sign deterministically: k<0 -> reversal (negative spread), k>0 -> momentum
    (positive spread), k=0 -> no planted relationship. Session 0's r_on is undefined
    (no prior session), exactly like the first session in real data -- not a fixture
    artifact.

    With exactly 2 bars/session, the horizon from the 09:16 row to the 15:20 row is 1
    bar for every session, which is what makes `horizon_mode="session"` hand-verifiable:
    horizon=1 disables the block-bootstrap SE path entirely (an analytic SE is used
    instead), so bucket means/spreads are fully deterministic given `ramp`/`k`, and only
    the SE/t-stat depend on `noise_scale`."""
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    prev_close: dict[str, float] | None = None
    for day_i in range(n_sessions):
        d = start_date + datetime.timedelta(days=day_i)
        if prev_close is None:
            opens = {sym: 100.0 for sym in symbols}
        else:
            opens = {
                sym: prev_close[sym] * math.exp(float(ramp[j]))
                for j, sym in enumerate(symbols)
            }
        noise = rng.standard_normal(n_symbols) * noise_scale
        closes: dict[str, float] = {}
        for j, sym in enumerate(symbols):
            o = opens[sym]
            r_id = k * float(ramp[j]) + float(noise[j])
            c = o * math.exp(r_id)
            bars_by_symbol[sym].append((d, "09:16", _bar(o)))
            bars_by_symbol[sym].append((d, "15:20", _bar(open_=o, close=c)))
            closes[sym] = c
        prev_close = closes
    return _build_panel(bars_by_symbol)


def _yearly_reversal_panel(
    symbols: tuple[str, ...],
    ramp: np.ndarray,
    year_k: dict[int, float],
    sessions_per_year: int,
    noise_scale: float,
    seed: int,
) -> Panel:
    """8 years (or however many keys `year_k` has) x `sessions_per_year` 2-bar sessions,
    one continuous rng stream across the whole panel (matches `_ramp_panel`'s
    determinism story). Each year gets its OWN reversal/momentum coefficient
    `year_k[year]`, so each year's expectancy sign is independently controllable --
    verified numerically (not just asserted) to be robust to the noise draw before this
    fixture was adopted; see the docstring on the test that consumes it."""
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    prev_close: dict[str, float] | None = None
    for year in sorted(year_k):
        k = year_k[year]
        for j in range(sessions_per_year):
            d = datetime.date(year, 1, 15) + datetime.timedelta(days=7 * j)
            if prev_close is None:
                opens = {sym: 100.0 for sym in symbols}
            else:
                opens = {
                    sym: prev_close[sym] * math.exp(float(ramp[i]))
                    for i, sym in enumerate(symbols)
                }
            noise = rng.standard_normal(n_symbols) * noise_scale
            closes: dict[str, float] = {}
            for i, sym in enumerate(symbols):
                o = opens[sym]
                r_id = k * float(ramp[i]) + float(noise[i])
                c = o * math.exp(r_id)
                bars_by_symbol[sym].append((d, "09:16", _bar(o)))
                bars_by_symbol[sym].append((d, "15:20", _bar(open_=o, close=c)))
                closes[sym] = c
            prev_close = closes
    return _build_panel(bars_by_symbol)


def _day_index_of_row(day_offsets: np.ndarray, row: int) -> int:
    """Which session (0-based day index) a global row index belongs to."""
    return int(np.searchsorted(day_offsets[1:], row, side="right"))


# ---------------------------------------------------------------------------
# Overnight return construction (highest risk)
# ---------------------------------------------------------------------------


def test_r_on_uses_prev_1520_close_and_today_0916_open_hand_computed() -> None:
    d0 = datetime.date(2021, 1, 1)
    d1 = datetime.date(2021, 1, 2)
    bars = {
        "SYM": [
            (d0, "09:16", _bar(50.0)),
            (d0, "15:20", _bar(open_=50.0, close=55.0)),
            (d1, "09:16", _bar(60.0)),
            (d1, "15:20", _bar(open_=60.0, close=61.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]
    d1_row = int(panel.day_offsets[1])

    expected = math.log(60.0 / 55.0)
    assert feat.values[d1_row, sym_ix] == pytest.approx(expected, abs=1e-9)


def test_0915_never_read_r_on_bit_identical_when_0915_present_and_corrupt() -> None:
    """All 61/61 observed 2024 OHLC violations sit in the 09:15 bar (close > high,
    pre-open call-auction leakage). r_on for the session that gained a 09:15 bar must be
    bit-for-bit unaffected by that bar's presence, let alone its corruption."""
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)

    baseline_bars = {
        "SYM": [
            (d0, "09:16", _bar(100.0)),
            (d0, "15:20", _bar(open_=100.0, close=102.0)),
            (d1, "09:16", _bar(103.0)),
            (d1, "15:20", _bar(open_=103.0, close=104.0)),
        ]
    }
    corrupted_bars = {
        "SYM": [
            *baseline_bars["SYM"],
            # Corrupt 09:15 bar on d1: close > high, open/close far from the true
            # 09:16 open of 103.0 -- if this were ever read as "the session's first
            # close", r_on[d1] would come out very different.
            (d1, "09:15", _bar(open_=50.0, high=50.2, low=49.8, close=95.0)),
        ]
    }

    baseline_panel = _build_panel(baseline_bars)
    corrupted_panel = _build_panel(corrupted_bars)

    baseline_feat = build_overnight_feature(baseline_panel)
    corrupted_feat = build_overnight_feature(corrupted_panel)

    baseline_val = baseline_feat.values[
        int(baseline_panel.day_offsets[1]), baseline_panel.sym_ix["SYM"]
    ]
    corrupted_val = corrupted_feat.values[
        int(corrupted_panel.day_offsets[1]), corrupted_panel.sym_ix["SYM"]
    ]
    assert corrupted_val == baseline_val
    assert not math.isnan(baseline_val)


def test_first_session_has_no_prior_close_and_is_nan_not_dropped() -> None:
    """The first session in the whole panel has no d-1 at all -> NaN, and the array
    shape is preserved (the row is present with a NaN value, never silently removed)."""
    d0 = datetime.date(2022, 1, 1)
    d1 = datetime.date(2022, 1, 2)
    bars = {
        "SYM": [
            (d0, "09:16", _bar(100.0)),
            (d0, "15:20", _bar(open_=100.0, close=101.0)),
            (d1, "09:16", _bar(102.0)),
            (d1, "15:20", _bar(open_=102.0, close=103.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    assert feat.values.shape[0] == panel.n_rows()
    first_day = slice(int(panel.day_offsets[0]), int(panel.day_offsets[1]))
    assert np.all(np.isnan(feat.values[first_day, sym_ix]))
    # The second session's r_on IS defined, so the NaN above is specifically about
    # "no prior session", not a blanket failure of the fixture.
    assert not math.isnan(feat.values[int(panel.day_offsets[1]), sym_ix])


def test_gap_bridges_to_previous_usable_session_not_calendar_day() -> None:
    """d-1 is the previous session IN THE PANEL, never `date - 1 day`: Friday's close
    must feed Monday's r_on across the weekend gap (no Saturday/Sunday rows exist at
    all -- they are not degenerate sessions, they simply are not in the panel)."""
    friday = datetime.date(2020, 1, 3)
    monday = datetime.date(2020, 1, 6)
    bars = {
        "SYM": [
            (friday, "09:16", _bar(100.0)),
            (friday, "15:20", _bar(open_=100.0, close=110.0)),
            (monday, "09:16", _bar(115.0)),
            (monday, "15:20", _bar(open_=115.0, close=116.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    assert panel.n_days() == 2
    monday_row = int(panel.day_offsets[1])
    expected = math.log(115.0 / 110.0)
    assert feat.values[monday_row, sym_ix] == pytest.approx(expected, abs=1e-9)


def test_muhurat_no_1520_makes_next_day_r_on_nan_not_stale() -> None:
    """A ~60-bar Muhurat session (09:16..10:15 inclusive) has no 15:20 bar at all, so it
    is NOT a usable d-1: per the spec's own drop rule ("missing the 15:20 close on d-1"
    -> dropped), the day after Muhurat must be NaN, not silently bridge past Muhurat to
    the last REGULAR session's close. Muhurat's own bars are all set to a distinctive
    sentinel price (999.0) so that if a buggy implementation ever read one of them as a
    'close', the result would be visibly, not coincidentally, wrong."""
    d0 = datetime.date(2020, 1, 1)  # normal
    muhurat = datetime.date(2020, 1, 2)  # ~60 bars, no 15:20
    d2 = datetime.date(2020, 1, 3)  # normal

    bars_list: list[tuple[datetime.date, str, dict[str, float]]] = [
        (d0, "09:16", _bar(100.0)),
        (d0, "15:20", _bar(open_=100.0, close=105.0)),
    ]
    for minute in range(9 * 60 + 16, 10 * 60 + 15 + 1):
        hour, minute_part = minute // 60, minute % 60
        bars_list.append((muhurat, f"{hour:02d}:{minute_part:02d}", _bar(999.0)))
    bars_list += [
        (d2, "09:16", _bar(200.0)),
        (d2, "15:20", _bar(open_=200.0, close=201.0)),
    ]

    panel = _build_panel({"SYM": bars_list})
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    assert panel.n_days() == 3
    d2_row = int(panel.day_offsets[2])
    assert math.isnan(feat.values[d2_row, sym_ix])


def test_per_symbol_gap_is_independent_not_panel_wide() -> None:
    """Symbol A trades on d-1, symbol B does not (no bars at all for B that day) -- B's
    r_on on d must be NaN while A's is finite. This is per-symbol, not per-panel: a
    panel-level drop would silently discard A's perfectly good observation too."""
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)
    d2 = datetime.date(2020, 1, 3)
    bars = {
        "A": [
            (d0, "09:16", _bar(100.0)),
            (d0, "15:20", _bar(open_=100.0, close=101.0)),
            (d1, "09:16", _bar(102.0)),
            (d1, "15:20", _bar(open_=102.0, close=103.0)),
            (d2, "09:16", _bar(104.0)),
            (d2, "15:20", _bar(open_=104.0, close=105.0)),
        ],
        "B": [
            (d0, "09:16", _bar(200.0)),
            (d0, "15:20", _bar(open_=200.0, close=201.0)),
            # B has no bars at all on d1.
            (d2, "09:16", _bar(210.0)),
            (d2, "15:20", _bar(open_=210.0, close=211.0)),
        ],
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    a_ix = panel.sym_ix["A"]
    b_ix = panel.sym_ix["B"]
    d2_row = int(panel.day_offsets[2])

    expected_a = math.log(104.0 / 103.0)
    assert feat.values[d2_row, a_ix] == pytest.approx(expected_a, abs=1e-9)
    assert math.isnan(feat.values[d2_row, b_ix])


def test_nan_checkpoint_propagates_without_forward_fill() -> None:
    """A corrupt (NaN) 09:16 open on day d-1... err, on the MIDDLE day makes that day's
    OWN r_on undefined, but must not corrupt the FOLLOWING day's r_on, which needs the
    middle day's 15:20 close (a real, present value) -- never forward-filled from an
    earlier or later session (CLAUDE.md rule 6)."""
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)
    d2 = datetime.date(2020, 1, 3)
    bars = {
        "SYM": [
            (d0, "09:16", _bar(100.0)),
            (d0, "15:20", _bar(open_=100.0, close=101.0)),
            (d1, "09:16", _bar(open_=float("nan"))),
            (d1, "15:20", _bar(open_=102.0, close=103.0)),
            (d2, "09:16", _bar(104.0)),
            (d2, "15:20", _bar(open_=104.0, close=105.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    d1_row = int(panel.day_offsets[1])
    d2_row = int(panel.day_offsets[2])
    assert math.isnan(feat.values[d1_row, sym_ix])

    expected_d2 = math.log(104.0 / 103.0)
    assert feat.values[d2_row, sym_ix] == pytest.approx(expected_d2, abs=1e-9)


def test_feature_broadcast_across_session_and_kind_is_return() -> None:
    d0 = datetime.date(2020, 1, 1)
    d1 = datetime.date(2020, 1, 2)
    bars = {
        "SYM": [
            (d0, "09:16", _bar(100.0)),
            (d0, "11:00", _bar(open_=100.5, close=100.5)),
            (d0, "15:20", _bar(open_=100.5, close=102.0)),
            (d1, "09:16", _bar(103.0)),
            (d1, "13:00", _bar(open_=103.2, close=103.2)),
            (d1, "15:20", _bar(open_=103.2, close=104.0)),
        ]
    }
    panel = _build_panel(bars)
    feat = build_overnight_feature(panel)
    sym_ix = panel.sym_ix["SYM"]

    assert feat.kind == "return"
    d1_slice = slice(int(panel.day_offsets[1]), int(panel.day_offsets[2]))
    values = feat.values[d1_slice, sym_ix]
    assert np.all(values == values[0])
    assert not math.isnan(values[0])


def test_build_overnight_feature_matches_market_overnight_return_directly() -> None:
    """In a fixture with NO 09:15 bar and open == close at 09:16 (a doji, via `_bar`'s
    default), the panel's natural day_offsets already point at 09:16, and
    `overnight_return`'s own 'first close of session' IS exactly the spec's 'open of
    09:16' -- so an implementation that truly reuses (never reimplements)
    `features.market.overnight_return` must match it bit-for-bit here."""
    bars = {
        "SYM": [
            (datetime.date(2020, 1, 1), "09:16", _bar(100.0)),
            (datetime.date(2020, 1, 1), "15:20", _bar(open_=100.0, close=103.0)),
            (datetime.date(2020, 1, 2), "09:16", _bar(104.0)),
            (datetime.date(2020, 1, 2), "15:20", _bar(open_=104.0, close=102.0)),
            (datetime.date(2020, 1, 3), "09:16", _bar(101.0)),
            (datetime.date(2020, 1, 3), "15:20", _bar(open_=101.0, close=101.5)),
        ]
    }
    panel = _build_panel(bars)
    sym_ix = panel.sym_ix["SYM"]

    direct = overnight_return(panel.field("close")[:, sym_ix], panel.day_offsets)
    via_feature = build_overnight_feature(panel).values[:, sym_ix]

    np.testing.assert_array_equal(direct, via_feature)


# ---------------------------------------------------------------------------
# Reversal detection
# ---------------------------------------------------------------------------


def test_planted_reversal_recovered_with_negative_spread() -> None:
    """r_id = -k * r_on + noise, k=1.0 (a full reversal): the top overnight-gain bucket
    must show the most negative intraday return and vice versa, so top-minus-bottom is
    NEGATIVE. Verified numerically before adoption: spread ~ -172.8 bps, |t| >> 1.96."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-1.0, noise_scale=0.002, seed=111,
    )
    lens_result: HypothesisVerdict = run_h2(panel, seed=1)

    assert lens_result.expectancy.spread_bps < 0.0
    assert abs(lens_result.expectancy.spread_t) > 1.96


def test_planted_momentum_yields_positive_spread_and_explain_states_direction() -> None:
    """The mirror-image plant: r_id = +k * r_on + noise. Criterion 1 uses abs(spread),
    so this positive-spread MOMENTUM relationship clears the same statistical gate a
    reversal would -- the guard against silently reporting it as confirmed reversal is
    that `explain()` must name the observed direction. Verified numerically before
    adoption: spread ~ +170.2 bps."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=150, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=1.0, noise_scale=0.002, seed=222,
    )
    result: HypothesisVerdict = run_h2(panel, seed=2)

    assert result.expectancy.spread_bps > 0.0
    explanation = result.explain().lower()
    # The observed direction (momentum, not reversal) must be named explicitly: a
    # positive spread clears the same abs()-gated criterion 1 a real reversal would,
    # so silence here would let momentum masquerade as a confirmed H2 result.
    assert "momentum" in explanation


def test_pure_noise_yields_insignificant_spread_and_not_survived() -> None:
    """r_on and r_id drawn from two INDEPENDENT rng streams, zero shared structure.
    Driftlessness of both raw generators is verified BEFORE any significance claim
    (CLAUDE.md-adjacent lesson from H1: a shared-fixture generator once baked in an
    ~8x-dominant drift), and the realized spread t-stat at this exact seed pair is
    verified to be insignificant before being asserted -- a driftless generator can
    still hand an unlucky draw (a previous fixture produced t=+2.345 on pure noise at
    seed 20260817). Verified numerically before adoption: t ~ 0.276."""
    n_sessions, n_symbols, scale = 200, 8, 0.01
    rng_on = np.random.default_rng(90001)
    rng_id = np.random.default_rng(90002)
    r_on = rng_on.standard_normal((n_sessions, n_symbols)) * scale
    r_id = rng_id.standard_normal((n_sessions, n_symbols)) * scale

    se_on = r_on.std(ddof=1) / np.sqrt(r_on.size)
    se_id = r_id.std(ddof=1) / np.sqrt(r_id.size)
    assert abs(r_on.mean()) < 5 * se_on
    assert abs(r_id.mean()) < 5 * se_id

    symbols = tuple(f"SYM{i}" for i in range(n_symbols))
    bars_by_symbol: dict[str, list[tuple[datetime.date, str, dict[str, float]]]] = {
        s: [] for s in symbols
    }
    prev_close: dict[str, float] | None = None
    start_date = datetime.date(2021, 1, 1)
    for day_i in range(n_sessions):
        d = start_date + datetime.timedelta(days=day_i)
        if prev_close is None:
            opens = {sym: 100.0 for sym in symbols}
        else:
            opens = {
                sym: prev_close[sym] * math.exp(float(r_on[day_i, j]))
                for j, sym in enumerate(symbols)
            }
        closes: dict[str, float] = {}
        for j, sym in enumerate(symbols):
            o = opens[sym]
            c = o * math.exp(float(r_id[day_i, j]))
            bars_by_symbol[sym].append((d, "09:16", _bar(o)))
            bars_by_symbol[sym].append((d, "15:20", _bar(open_=o, close=c)))
            closes[sym] = c
        prev_close = closes
    panel = _build_panel(bars_by_symbol)

    result: HypothesisVerdict = run_h2(panel, seed=3)

    assert abs(result.expectancy.spread_t) < 1.96
    assert result.survived is False


def test_planted_reversal_below_hurdle_is_killed_despite_significance() -> None:
    """A small but real reversal (k=0.08) with low noise and many sessions: the spread
    is statistically significant (large N, low variance) but stays under the 2x cost
    hurdle -- criterion 1 must fail and the overall verdict must be KILLED, regardless
    of criterion 3's significance passing. Verified numerically before adoption:
    spread ~ 14.08 bps (< 2x hurdle ~16.53 bps), t ~ 47.7."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=300, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=0.08, noise_scale=0.0005, seed=333,
    )
    result: HypothesisVerdict = run_h2(panel, seed=4)

    hurdle = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert abs(result.expectancy.spread_bps) < 2 * hurdle
    assert abs(result.expectancy.spread_t) > 1.96
    assert result.survived is False


# ---------------------------------------------------------------------------
# Integration with Lens
# ---------------------------------------------------------------------------


def test_verdict_reports_all_six_criteria_even_after_a_failure() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    # k=0: no planted relationship at all -- criterion 1 (edge) fails for certain.
    panel = _ramp_panel(
        symbols, n_sessions=60, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=0.0, noise_scale=0.002, seed=101,
    )
    result: HypothesisVerdict = run_h2(panel, seed=0)

    for n in range(1, 7):
        assert any(r.startswith(f"{n}.") for r in result.reasons), f"missing criterion {n}"


def test_criterion_5_is_not_evaluated_and_never_silently_passes() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=60, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=102,
    )
    result: HypothesisVerdict = run_h2(panel, seed=0)

    c5_lines = [r for r in result.reasons if r.startswith("5.")]
    assert len(c5_lines) == 1
    assert "NOT_EVALUATED" in c5_lines[0]
    assert "PASS" not in c5_lines[0]


def test_run_h2_expectancy_equals_direct_conditional_expectancy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_h2 must not reimplement expectancy maths: spy on the real
    `expectancy.conditional_expectancy` (the function `Lens.expectancy` delegates to)
    and confirm run_h2's returned table IS the one that call produced, not a
    separately-computed lookalike."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
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

    result: HypothesisVerdict = run_h2(panel, seed=3)

    assert len(calls) >= 1
    assert result.expectancy == calls[0]


# ---------------------------------------------------------------------------
# Reporting and hygiene
# ---------------------------------------------------------------------------


def test_output_contains_survivorship_warning_line() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=40, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=103,
    )
    result: HypothesisVerdict = run_h2(panel, seed=0)
    assert "UNIVERSE" in result.explain()


def test_isolated_one_year_effect_shows_low_years_sign_consistent() -> None:
    """8 years, 5 symbols mapped 1:1 into the 5 cross-sectional buckets (deterministic
    rank order every session), one dominant year (2018, k=+5.0) and the rest split 5
    negative / 3 positive at a small magnitude -- verified numerically before adoption
    (seeds 42/7/2026/12345 all reproduce sign pattern {-1: 5, 1: 3}) to be robust to the
    noise draw. years_sign_consistent must land on the modal sign's count (5), well
    under the 6/8 threshold for `sign_stable`."""
    symbols = tuple(f"SYM{i}" for i in range(5))
    ramp = np.linspace(-0.008, 0.008, len(symbols))
    year_k = {
        2018: 5.0,
        2019: -0.3,
        2020: -0.3,
        2021: -0.3,
        2022: 0.3,
        2023: 0.3,
        2024: -0.3,
        2025: -0.3,
    }
    panel = _yearly_reversal_panel(
        symbols, ramp, year_k, sessions_per_year=6, noise_scale=0.0004, seed=42,
    )
    result: HypothesisVerdict = run_h2(panel, seed=0)

    assert result.stability.n_years_total == 8
    assert result.stability.n_years_sign_consistent == 5
    assert result.stability.sign_stable is False


def test_start_end_restricts_window_to_strictly_fewer_sessions() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=80, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=104,
    )
    full: HypothesisVerdict = run_h2(panel, seed=0)
    cutoff = panel.dates[40]
    restricted: HypothesisVerdict = run_h2(panel, seed=0, end=cutoff)

    assert restricted.expectancy.n_total < full.expectancy.n_total


def test_run_h2_is_deterministic_and_records_seed() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=50, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=105,
    )
    result_a: HypothesisVerdict = run_h2(panel, seed=42)
    result_b: HypothesisVerdict = run_h2(panel, seed=42)

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


def test_degenerate_day_below_min_names_yields_no_raise_and_finite_result() -> None:
    """A day with fewer than 5 usable symbols (the cross-sectional min_names default
    used throughout this repo's ranking machinery) must be excluded from that day's
    cross-sectional ranking -- yielding NaN for that day's contribution, never raising
    and never corrupting the rest of the (otherwise well-formed) run."""
    symbols = tuple(f"SYM{i}" for i in range(8))
    n_sessions = 12
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=n_sessions, start_date=datetime.date(2020, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=555,
    )

    # Blank out all but 2 symbols' bars on session index 5, leaving that day with only
    # 2 usable names -- well below the min_names=5 cross-sectional gate.
    degenerate_day = 5
    row_slice = slice(
        int(panel.day_offsets[degenerate_day]), int(panel.day_offsets[degenerate_day + 1])
    )
    for j in range(2, len(symbols)):
        for field_name in ("open", "high", "low", "close"):
            panel.field(field_name)[row_slice, j] = np.nan

    result: HypothesisVerdict = run_h2(panel, seed=1)

    assert math.isfinite(result.expectancy.spread_bps)
    assert math.isfinite(result.expectancy.spread_t)


# ---------------------------------------------------------------------------
# Extra: cost hurdle provenance (constraint, not a numbered spec item, but explicitly
# called out as high-risk: "Resolve it FROM the model, never hard-code")
# ---------------------------------------------------------------------------


def test_default_cost_hurdle_resolved_from_cost_model_not_hardcoded() -> None:
    symbols = tuple(f"SYM{i}" for i in range(8))
    ramp = np.linspace(-0.01, 0.01, len(symbols))
    panel = _ramp_panel(
        symbols, n_sessions=40, start_date=datetime.date(2019, 1, 1),
        ramp=ramp, k=-0.6, noise_scale=0.001, seed=106,
    )
    result: HypothesisVerdict = run_h2(panel, seed=0)

    expected_hurdle = NSEIntradayEquityCosts().round_trip_bps(1e5)
    assert result.cost_hurdle_bps == pytest.approx(expected_hurdle, rel=1e-9)
    assert result.expectancy.cost_hurdle_bps == pytest.approx(expected_hurdle, rel=1e-9)
    assert result.cost_hurdle_bps == pytest.approx(8.26452, abs=1e-2)
