from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import pytest

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h1_market_intraday_momentum import (
    H1Result,
    SessionObservations,
    extract_session_observations,
    run_h1,
)
from nifty_quant.research.splits import HoldoutLock

R_FIRST_SMALL: np.ndarray = np.array(
    [0.001, -0.002, 0.003, -0.0015, 0.0005, 0.0022, -0.0008]
)
R_LAST_SMALL: np.ndarray = np.array(
    [0.0004, -0.0011, 0.0016, -0.0009, 0.0002, 0.0013, -0.0003]
)

# H1's checkpoint prices are derived via `base * exp(return)` and stored in the panel as
# float32 (rest-of-repo convention). That round trip introduces relative noise of order
# 1e-5 to 1e-4 in reconstructed log-returns and downstream OLS statistics, which is far
# looser than pytest.approx's tight default tolerance. BPS_TOL / tighter-but-still-safe
# tolerances below are calibrated with margin against that measured noise floor.
BPS_TOL = 1e-2


def _background_r_mid(n: int, seed: int = 909) -> np.ndarray:
    """Small, non-degenerate r_mid values for fixtures where r_mid's specific value is
    irrelevant to the assertions but must have nonzero variance: a constant (e.g. all-
    zero) r_mid column makes the mandatory `beta_controlling_for_mid` regression's
    design matrix singular, since every session's r_mid then explains no variance."""
    return np.random.default_rng(seed).standard_normal(n) * 0.0005


def _epoch_seconds(session_date: datetime.date, hhmm: str) -> int:
    """Left-labelled epoch-second UTC timestamp for an IST wall-clock HH:MM on session_date."""
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
    """Build an OHLC dict; high/low default to bracket open/close, close defaults to open."""
    c = open_ if close is None else close
    h = max(open_, c) + 0.01 if high is None else high
    lo = min(open_, c) - 0.01 if low is None else low
    return {"open": open_, "high": h, "low": lo, "close": c}


def _build_panel(
    bars: list[tuple[datetime.date, str, dict[str, float]]],
    symbol: str = "NIFTY50",
) -> Panel:
    """bars: list of (date, 'HH:MM', ohlc_dict). Order does not matter; rows are sorted by
    timestamp internally. One symbol only."""
    rows = sorted(
        ((_epoch_seconds(d, hhmm), ohlc) for d, hhmm, ohlc in bars),
        key=lambda r: r[0],
    )
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    fields: dict[str, np.ndarray] = {}
    for field_name in ("open", "high", "low", "close"):
        fields[field_name] = np.array(
            [[r[1][field_name]] for r in rows], dtype=np.float32
        )
    grid = SessionGrid.from_timestamps(ts)
    return Panel(
        fields=fields,
        symbols=(symbol,),
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _planted_panel(
    r_first: np.ndarray,
    r_mid: np.ndarray,
    r_last: np.ndarray,
    start_date: datetime.date,
    symbol: str = "NIFTY50",
) -> Panel:
    """Build a synthetic panel whose OHLC checkpoints imply exactly the supplied returns."""
    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    base = 100.0
    for i, (rf, rm, rl) in enumerate(zip(r_first, r_mid, r_last)):
        d = start_date + datetime.timedelta(days=i)
        open_0916 = base
        close_0945 = base * math.exp(float(rf))
        open_1450 = close_0945 * math.exp(float(rm))
        close_1520 = open_1450 * math.exp(float(rl))
        bars.append((d, "09:16", _bar(open_0916)))
        bars.append((d, "09:45", _bar(open_=close_0945, close=close_0945)))
        bars.append((d, "14:50", _bar(open_=open_1450, close=open_1450)))
        bars.append((d, "15:20", _bar(open_=open_1450, close=close_1520)))
    return _build_panel(bars, symbol=symbol)


def _small_panel() -> Panel:
    """r_mid gets small nonzero background noise (irrelevant to the univariate stats
    this fixture is used to check) so the mandatory beta_controlling_for_mid regression
    run_h1 always computes is never degenerate."""
    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    base = 100.0
    r_mid_bg = _background_r_mid(len(R_FIRST_SMALL))
    for i, (rf, rl) in enumerate(zip(R_FIRST_SMALL, R_LAST_SMALL)):
        d = datetime.date(2019, 1, 1 + i)
        open_0916 = base
        close_0945 = base * math.exp(float(rf))
        open_1450 = close_0945 * math.exp(float(r_mid_bg[i]))
        close_1520 = open_1450 * math.exp(float(rl))
        bars.append((d, "09:16", _bar(open_0916)))
        bars.append((d, "09:45", _bar(open_=close_0945, close=close_0945)))
        bars.append((d, "14:50", _bar(open_=open_1450, close=open_1450)))
        bars.append((d, "15:20", _bar(open_=open_1450, close=close_1520)))
    return _build_panel(bars)


def _simple_valid_bars(
    d: datetime.date,
) -> list[tuple[datetime.date, str, dict[str, float]]]:
    return [
        (d, "09:16", _bar(100.0)),
        (d, "09:45", _bar(open_=100.0, close=101.0)),
        (d, "14:50", _bar(open_=101.0, close=101.0)),
        (d, "15:20", _bar(open_=101.0, close=102.0)),
    ]


def _yearly_r_last_panel(year_r_last: dict[int, float]) -> Panel:
    """8 years x 3 sessions/year; r_first strictly positive & distinct (non-degenerate OLS),
    r_last a per-year constant so each year's mean edge is analytically known. r_mid gets
    small nonzero background noise (irrelevant to by_year/years_sign_consistent, which
    depend only on sign(r_first)*r_last) so the mandatory beta_controlling_for_mid
    regression is never degenerate."""
    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    r_mid_bg = _background_r_mid(24)
    k = 0
    for year in range(2018, 2026):
        rl = year_r_last[year]
        for j in range(3):
            d = datetime.date(year, 1, 15 + j)
            rf = 0.0005 * (k + 1)
            open_0916 = 100.0
            close_0945 = open_0916 * math.exp(rf)
            open_1450 = close_0945 * math.exp(float(r_mid_bg[k]))
            close_1520 = open_1450 * math.exp(rl)
            bars.append((d, "09:16", _bar(open_0916)))
            bars.append((d, "09:45", _bar(open_=close_0945, close=close_0945)))
            bars.append((d, "14:50", _bar(open_=open_1450, close=open_1450)))
            bars.append((d, "15:20", _bar(open_=open_1450, close=close_1520)))
            k += 1
    return _build_panel(bars)


def _planted_multiyear_panel(
    r_first: np.ndarray,
    r_mid: np.ndarray,
    r_last: np.ndarray,
    start_year: int,
    n_years: int,
    sessions_per_year: int,
    symbol: str = "NIFTY50",
) -> Panel:
    """Like _planted_panel, but spreads len(r_first) == n_years * sessions_per_year
    sessions evenly across n_years distinct calendar years (needed for by_year /
    years_sign_consistent fixtures with a real edge, rather than the per-year-constant
    shortcut _yearly_r_last_panel uses). Each year's dates run from Jan 1 onward by
    session index so they never cross into the next calendar year even for a large
    sessions_per_year."""
    assert len(r_first) == n_years * sessions_per_year
    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    base = 100.0
    k = 0
    for y in range(n_years):
        year = start_year + y
        for j in range(sessions_per_year):
            d = datetime.date(year, 1, 1) + datetime.timedelta(days=j)
            rf = float(r_first[k])
            rm = float(r_mid[k])
            rl = float(r_last[k])
            open_0916 = base
            close_0945 = base * math.exp(rf)
            open_1450 = close_0945 * math.exp(rm)
            close_1520 = open_1450 * math.exp(rl)
            bars.append((d, "09:16", _bar(open_0916)))
            bars.append((d, "09:45", _bar(open_=close_0945, close=close_0945)))
            bars.append((d, "14:50", _bar(open_=open_1450, close=open_1450)))
            bars.append((d, "15:20", _bar(open_=open_1450, close=close_1520)))
            k += 1
    return _build_panel(bars, symbol=symbol)


# ---------------------------------------------------------------------------
# Checkpoint extraction (highest-risk area)
# ---------------------------------------------------------------------------


def test_r_first_uses_0916_open_and_0945_close_hand_computed() -> None:
    d = datetime.date(2020, 1, 1)
    bars = [
        (d, "09:16", _bar(100.0)),
        (d, "09:45", _bar(open_=100.0, close=101.0)),
        (d, "14:50", _bar(open_=101.0, close=101.0)),
        (d, "15:20", _bar(open_=101.0, close=101.0)),
    ]
    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")
    assert obs.r_first[0] == pytest.approx(math.log(101.0 / 100.0))


def test_0915_is_never_referenced() -> None:
    """All 61/61 observed 2024 OHLC violations are at 09:15 (close > high, pre-open call
    auction leakage). r_first must be bit-for-bit unaffected by that bar's presence or
    corruption."""
    d1 = datetime.date(2020, 1, 1)
    d2 = datetime.date(2020, 1, 2)
    d3 = datetime.date(2020, 1, 3)

    baseline_bars = _simple_valid_bars(d1) + [
        (d1, "09:15", _bar(open_=90.0, high=91.0, low=89.0, close=90.5))
    ]
    no_0915_bars = _simple_valid_bars(d2)
    corrupt_bars = _simple_valid_bars(d3) + [
        (d3, "09:15", _bar(open_=90.0, high=90.2, low=89.8, close=95.0))
    ]

    baseline = extract_session_observations(_build_panel(baseline_bars), "NIFTY50")
    no_0915 = extract_session_observations(_build_panel(no_0915_bars), "NIFTY50")
    corrupt = extract_session_observations(_build_panel(corrupt_bars), "NIFTY50")

    assert baseline.r_first[0] == no_0915.r_first[0]
    assert baseline.r_first[0] == corrupt.r_first[0]


def test_r_last_uses_1450_open_and_1520_close_hand_computed() -> None:
    d = datetime.date(2020, 1, 1)
    bars = [
        (d, "09:16", _bar(100.0)),
        (d, "09:45", _bar(open_=100.0, close=101.0)),
        (d, "14:50", _bar(open_=200.0, close=200.0)),
        (d, "15:20", _bar(open_=200.0, close=198.0)),
    ]
    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")
    assert obs.r_last[0] == pytest.approx(math.log(198.0 / 200.0))


def test_decomposition_identity() -> None:
    d = datetime.date(2020, 1, 1)
    bars = [
        (d, "09:16", _bar(100.0)),
        (d, "09:45", _bar(open_=100.0, close=101.0)),
        (d, "14:50", _bar(open_=200.0, close=200.0)),
        (d, "15:20", _bar(open_=200.0, close=198.0)),
    ]
    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert (
        obs.r_first[0] + obs.r_mid[0] + obs.r_last[0]
        == pytest.approx(math.log(198.0 / 100.0), abs=1e-12)
    )


def test_session_missing_1450_is_dropped() -> None:
    gapped_date = datetime.date(2020, 1, 1)
    control_date = datetime.date(2020, 1, 2)

    bars: list[tuple[datetime.date, str, dict[str, float]]] = [
        (gapped_date, "09:16", _bar(100.0)),
        (gapped_date, "09:45", _bar(open_=100.0, close=101.0)),
        (gapped_date, "14:49", _bar(open_=101.0, close=101.0)),
        (gapped_date, "14:51", _bar(open_=101.0, close=101.0)),
        (gapped_date, "15:20", _bar(open_=101.0, close=102.0)),
    ]
    bars.extend(_simple_valid_bars(control_date))

    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert obs.n_sessions_dropped == 1
    # AMBIGUITIES RESOLVED 2026-08-17: exactly ONE key increments per dropped session (the
    # first missing checkpoint in order 09:16, 09:45, 14:50, 15:20), so this is exact, not
    # just ">= 1", and sum(drop_reasons.values()) == n_sessions_dropped is assertable.
    assert obs.drop_reasons == {"missing_14:50": 1}
    assert sum(obs.drop_reasons.values()) == obs.n_sessions_dropped
    assert control_date in obs.dates
    assert gapped_date not in obs.dates


def test_muhurat_session_dropped() -> None:
    """A ~60-bar session (09:16..10:15 inclusive) has no 14:50 bar at all and must drop
    out naturally, never assuming a fixed 375-bar stride."""
    muhurat_date = datetime.date(2020, 1, 1)
    control_date = datetime.date(2020, 1, 2)

    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    for minute in range(9 * 60 + 16, 10 * 60 + 15 + 1):
        hour = minute // 60
        minute_part = minute % 60
        hhmm = f"{hour:02d}:{minute_part:02d}"
        bars.append((muhurat_date, hhmm, _bar(100.0)))
    bars.extend(_simple_valid_bars(control_date))

    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert obs.n_sessions_dropped == 1
    assert obs.drop_reasons == {"missing_14:50": 1}
    assert sum(obs.drop_reasons.values()) == obs.n_sessions_dropped
    assert muhurat_date not in obs.dates
    assert control_date in obs.dates


def test_nan_at_required_checkpoint_drops_session_no_forward_fill() -> None:
    """NaN at the 09:45 close is the first-in-order missing checkpoint for this session
    (09:16 present, 09:45 NaN), so the session is dropped and counted under
    'missing_09:45' -- not forward-filled from a neighbouring bar or session."""
    nan_date = datetime.date(2020, 1, 1)
    control_date = datetime.date(2020, 1, 2)

    bars: list[tuple[datetime.date, str, dict[str, float]]] = [
        (nan_date, "09:16", _bar(100.0)),
        (nan_date, "09:45", _bar(open_=100.0, close=float("nan"))),
        (nan_date, "14:50", _bar(open_=101.0, close=101.0)),
        (nan_date, "15:20", _bar(open_=101.0, close=102.0)),
    ]
    bars.extend(_simple_valid_bars(control_date))

    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert obs.n_sessions_dropped == 1
    assert obs.drop_reasons == {"missing_09:45": 1}
    assert sum(obs.drop_reasons.values()) == obs.n_sessions_dropped
    assert nan_date not in obs.dates
    assert control_date in obs.dates
    assert not np.isnan(obs.r_first).any()
    assert not np.isnan(obs.r_mid).any()
    assert not np.isnan(obs.r_last).any()


def test_drop_reasons_sum_equals_n_dropped_across_mixed_reasons() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: a session missing several checkpoints increments
    exactly ONE key (the first missing checkpoint in order 09:16, 09:45, 14:50, 15:20).
    With three sessions each dropped for a DIFFERENT first-missing checkpoint, the sum of
    drop_reasons.values() must equal n_sessions_dropped exactly."""
    control_date = datetime.date(2020, 1, 1)
    missing_0916_date = datetime.date(2020, 1, 2)
    missing_1450_date = datetime.date(2020, 1, 3)
    missing_1520_date = datetime.date(2020, 1, 4)

    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    bars.extend(_simple_valid_bars(control_date))

    # No 09:16 bar at all; 09:45/14:50/15:20 present.
    bars.extend(
        [
            (missing_0916_date, "09:45", _bar(open_=100.0, close=101.0)),
            (missing_0916_date, "14:50", _bar(open_=101.0, close=101.0)),
            (missing_0916_date, "15:20", _bar(open_=101.0, close=102.0)),
        ]
    )
    # 09:16/09:45 present, no 14:50 bar, 15:20 present.
    bars.extend(
        [
            (missing_1450_date, "09:16", _bar(100.0)),
            (missing_1450_date, "09:45", _bar(open_=100.0, close=101.0)),
            (missing_1450_date, "15:20", _bar(open_=101.0, close=102.0)),
        ]
    )
    # 09:16/09:45/14:50 present, no 15:20 bar.
    bars.extend(
        [
            (missing_1520_date, "09:16", _bar(100.0)),
            (missing_1520_date, "09:45", _bar(open_=100.0, close=101.0)),
            (missing_1520_date, "14:50", _bar(open_=101.0, close=101.0)),
        ]
    )

    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert obs.n_sessions_dropped == 3
    assert obs.drop_reasons == {
        "missing_09:16": 1,
        "missing_14:50": 1,
        "missing_15:20": 1,
    }
    assert sum(obs.drop_reasons.values()) == obs.n_sessions_dropped
    assert control_date in obs.dates
    for dropped in (missing_0916_date, missing_1450_date, missing_1520_date):
        assert dropped not in obs.dates


def test_time_label_resolution_survives_row_offset_shift() -> None:
    """Session B has an extra decoy bar between 09:16 and 09:45, shifting the row offset
    of every later checkpoint relative to day start. Checkpoint resolution must use
    Panel.rows_at_time (time-label lookup), never day_start + fixed_row_offset."""
    d1 = datetime.date(2020, 1, 1)
    d2 = datetime.date(2020, 1, 2)

    bars_a = _simple_valid_bars(d1)
    bars_b = _simple_valid_bars(d2) + [
        (d2, "09:30", _bar(open_=100.5, close=100.5))
    ]

    panel = _build_panel(bars_a + bars_b)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert obs.dates[0] == d1
    assert obs.dates[1] == d2
    assert obs.r_first[0] == pytest.approx(obs.r_first[1])
    assert obs.r_mid[0] == pytest.approx(obs.r_mid[1])
    assert obs.r_last[0] == pytest.approx(obs.r_last[1])


def test_dates_ascending_unique_and_shared_lengths() -> None:
    dates = [
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 4),
    ]

    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    for d in reversed(dates):
        bars.extend(_simple_valid_bars(d))

    panel = _build_panel(bars)
    obs: SessionObservations = extract_session_observations(panel, "NIFTY50")

    assert len(obs.dates) == 4
    assert len(obs.dates) == len(obs.r_first) == len(obs.r_mid) == len(obs.r_last)
    date_array = obs.dates.astype("datetime64[D]")
    assert np.all(np.diff(date_array) > np.timedelta64(0, "D"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_small_ols_against_hand_computed() -> None:
    panel = _small_panel()
    result: H1Result = run_h1(panel, "NIFTY50")

    # Hand-computed OLS-with-intercept on R_FIRST_SMALL/R_LAST_SMALL: beta=0.54537475,
    # t_stat=22.629871, r_squared=0.990331. Tolerances are widened past pytest.approx's
    # tight default because the fixture round-trips through float32 panel storage.
    assert result.beta == pytest.approx(0.54537475, rel=1e-3)
    assert result.t_stat == pytest.approx(22.629871, rel=1e-2)
    assert result.r_squared == pytest.approx(0.990331, rel=1e-3)


def test_t_stat_sign_matches_beta_sign() -> None:
    result: H1Result = run_h1(_small_panel(), "NIFTY50")
    assert (result.beta > 0) == (result.t_stat > 0)


def test_planted_relationship_recovered() -> None:
    rng = np.random.default_rng(1201)
    n = 80
    r_first = rng.standard_normal(n) * 0.006
    noise = rng.standard_normal(n) * 0.001
    r_last = 0.5 * r_first + noise
    r_mid = _background_r_mid(n)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2020, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    assert 0.3 <= result.beta <= 0.7
    assert abs(result.t_stat) > 1.96


def test_pure_noise_is_driftless_and_killed() -> None:
    """Two INDEPENDENT rng streams so there is zero shared structure between r_first and
    r_last. Driftlessness of the raw generator is asserted BEFORE any significance claim
    (this repo's shared fixture once baked an ~8x-dominant drift into 'noise')."""
    rng1 = np.random.default_rng(4001)
    rng2 = np.random.default_rng(4002)
    n = 300
    r_first = rng1.standard_normal(n) * 0.005
    r_last = rng2.standard_normal(n) * 0.005

    se_first = r_first.std(ddof=1) / np.sqrt(n)
    se_last = r_last.std(ddof=1) / np.sqrt(n)
    assert abs(r_first.mean()) < 5 * se_first
    assert abs(r_last.mean()) < 5 * se_last

    r_mid = _background_r_mid(n)
    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2021, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    assert abs(result.t_stat) < 1.96
    assert result.survives_costs is False


def test_mean_edge_bps_matches_definition_exactly() -> None:
    result: H1Result = run_h1(_small_panel(), "NIFTY50")
    expected = 1e4 * np.mean(np.sign(R_FIRST_SMALL) * R_LAST_SMALL)
    assert result.mean_edge_bps == pytest.approx(expected, abs=BPS_TOL)


def test_hit_rate_counts_sign_agreement_and_excludes_zero_r_first() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: zero-r_first (flat, no trade) sessions are
    excluded from BOTH the numerator and the denominator --
    traded = r_first != 0; hit_rate = mean(sign(r_last[traded]) == sign(r_first[traded]))."""
    r_first = np.array([0.001, 0.002, -0.0015, 0.0, -0.002], dtype=np.float64)
    r_last = np.array([0.001, -0.001, -0.0005, 0.003, 0.0007], dtype=np.float64)
    r_mid = _background_r_mid(5)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2020, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    # sessions: (+,+) agree, (+,-) disagree, (-,-) agree, (0,+) EXCLUDED, (-,+) disagree
    # -> 2 of the 4 TRADED sessions agree -> 2/4, not 2/5.
    assert result.hit_rate == pytest.approx(0.5)


def test_hit_rate_and_degenerate_regressions_are_nan_when_r_first_is_constant() -> None:
    """DEGENERATE REGRESSIONS RETURN NaN, THEY NEVER RAISE (spec, added 2026-08-17).
    An all-zero r_first column means: (1) no session trades, so hit_rate must be nan
    (not 0.0 -- a zero edge would misleadingly read as "measured and found flat" rather
    than "not measurable"); (2) r_first has zero variance, so the univariate regression
    (beta, t_stat, r_squared) is unidentified -> nan; (3) the bivariate design matrix
    [1, r_first, r_mid] is singular for the same reason (the r_first column is a
    multiple of the intercept column) -> beta_controlling_for_mid and its t_stat are
    also nan. The load-bearing assertion is that `run_h1` returns normally at all: a
    `numpy.linalg.LinAlgError` must never escape, because this harness will be pointed
    at 149 symbols x 8 years where some slice will inevitably be this degenerate, and an
    uncaught exception would abort the whole run rather than failing just this slice
    closed (a nan edge cannot clear the cost hurdle; a crash loses every other symbol)."""
    n = 6
    r_first = np.zeros(n, dtype=np.float64)
    r_mid = _background_r_mid(n)
    r_last = np.array([0.001, -0.002, 0.0015, -0.0009, 0.0002, -0.0013], dtype=np.float64)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2020, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    assert math.isnan(result.hit_rate)
    assert math.isnan(result.beta)
    assert math.isnan(result.t_stat)
    assert math.isnan(result.r_squared)
    assert math.isnan(result.beta_controlling_for_mid)
    assert math.isnan(result.t_stat_controlling_for_mid)


def test_survives_costs_just_below_threshold() -> None:
    n = 5
    r_first = np.array([0.001 * (i + 1) for i in range(n)], dtype=np.float64)
    r_mid = _background_r_mid(n)
    r_last = np.full(n, 19.9 / 1e4, dtype=np.float64)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2024, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50", cost_hurdle_bps=10.0)

    assert result.mean_edge_bps == pytest.approx(19.9, abs=BPS_TOL)
    assert result.survives_costs is False


def test_survives_costs_just_above_threshold() -> None:
    n = 5
    r_first = np.array([0.001 * (i + 1) for i in range(n)], dtype=np.float64)
    r_mid = _background_r_mid(n)
    r_last = np.full(n, 20.1 / 1e4, dtype=np.float64)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2024, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50", cost_hurdle_bps=10.0)

    assert result.mean_edge_bps == pytest.approx(20.1, abs=BPS_TOL)
    assert result.survives_costs is True


def test_default_cost_hurdle_uses_nse_costs() -> None:
    result: H1Result = run_h1(_small_panel(), "NIFTY50")
    assert result.cost_hurdle_bps == pytest.approx(
        NSEIntradayEquityCosts().round_trip_bps(1e5)
    )


# ---------------------------------------------------------------------------
# Confound (mandatory)
# ---------------------------------------------------------------------------


def test_confound_r_last_from_r_mid_requires_control() -> None:
    """r_last is driven ONLY by r_mid; a shared day-level drift factor makes the
    univariate beta look spuriously non-zero. Once r_mid is controlled for, b1 must
    collapse -- this is the difference between day-level drift and real intraday
    momentum."""
    rng = np.random.default_rng(6001)
    n = 150
    drift = rng.standard_normal(n) * 0.005
    eps1 = rng.standard_normal(n) * 0.0015
    eps2 = rng.standard_normal(n) * 0.0015
    eps3 = rng.standard_normal(n) * 0.0008
    r_first = drift + eps1
    r_mid = drift + eps2
    r_last = 0.7 * r_mid + eps3

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2022, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    assert abs(result.beta) > 0.1
    assert abs(result.beta_controlling_for_mid) < 0.5 * abs(result.beta)


def test_confound_r_last_from_r_first_survives_control() -> None:
    rng = np.random.default_rng(5001)
    n = 120
    r_first = rng.standard_normal(n) * 0.006
    r_mid = rng.standard_normal(n) * 0.004
    noise = rng.standard_normal(n) * 0.0008
    r_last = 0.55 * r_first + noise

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2023, 1, 1))
    result: H1Result = run_h1(panel, "NIFTY50")

    assert abs(result.beta_controlling_for_mid) >= 0.5 * abs(result.beta)
    assert abs(result.t_stat_controlling_for_mid) > 1.96


# ---------------------------------------------------------------------------
# Stability and reporting
# ---------------------------------------------------------------------------


def test_by_year_partition_correctness() -> None:
    year_r_last = {
        2018: 0.003,
        2019: 0.003,
        2020: 0.003,
        2021: 0.003,
        2022: -0.003,
        2023: 0.003,
        2024: 0.003,
        2025: -0.003,
    }
    panel = _yearly_r_last_panel(year_r_last)
    result: H1Result = run_h1(panel, "NIFTY50")

    assert set(result.by_year.keys()) == set(range(2018, 2026))
    positive_years = {2018, 2019, 2020, 2021, 2023, 2024}
    negative_years = {2022, 2025}
    for year in positive_years:
        assert result.by_year[year] == pytest.approx(30.0, abs=BPS_TOL)
    for year in negative_years:
        assert result.by_year[year] == pytest.approx(-30.0, abs=BPS_TOL)


def test_years_sign_consistent_majority() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: years_sign_consistent counts years holding the
    MODAL NON-ZERO sign (discard exactly-zero-edge years, take the sign held by the most
    remaining years). No years are zero-edge here, so this is unaffected by that
    resolution: 6 of 8 years positive -> modal sign positive, count 6."""
    year_r_last = {
        2018: 0.003,
        2019: 0.003,
        2020: 0.003,
        2021: 0.003,
        2022: -0.003,
        2023: 0.003,
        2024: 0.003,
        2025: -0.003,
    }
    result: H1Result = run_h1(_yearly_r_last_panel(year_r_last), "NIFTY50")
    assert result.years_sign_consistent == 6


def test_isolated_one_year_effect_is_low_years_sign_consistent() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: counting years (not magnitude-weighting) is what
    makes this robust to a single dominant year -- exactly the shape this fixture
    exercises. 2018 is a huge positive outlier (+50 bps) but only 3 of 8 years
    (2018, 2020, 2023) are individually positive; the modal non-zero sign among the 8
    years is NEGATIVE (5 of 8), so years_sign_consistent counts the 5 negative years --
    now assertable exactly rather than as a loose upper bound."""
    year_r_last = {
        2018: 0.05,
        2019: -0.001,
        2020: 0.0005,
        2021: -0.0008,
        2022: -0.0003,
        2023: 0.0002,
        2024: -0.0006,
        2025: -0.0004,
    }
    result: H1Result = run_h1(_yearly_r_last_panel(year_r_last), "NIFTY50")
    assert result.years_sign_consistent == 5


def test_years_sign_consistent_discards_zero_edge_years() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: years whose mean edge is exactly zero are
    discarded from the vote entirely (they neither count toward the total nor toward
    either sign's tally). 2 of 8 years here have an exactly-zero edge; among the
    remaining 6, 4 are positive and 2 are negative, so the modal sign is positive with
    count 4 -- not 4/8 nor 4/6 expressed as a fraction, just the raw count of years
    holding the modal sign."""
    year_r_last = {
        2018: 0.003,
        2019: 0.003,
        2020: 0.003,
        2021: 0.003,
        2022: -0.003,
        2023: -0.003,
        2024: 0.0,
        2025: 0.0,
    }
    result: H1Result = run_h1(_yearly_r_last_panel(year_r_last), "NIFTY50")
    assert result.years_sign_consistent == 4


def test_years_sign_consistent_exact_tie_is_mixed_and_counts_zero() -> None:
    """AMBIGUITIES RESOLVED 2026-08-17: on an exact tie between positive and negative
    years, dominant_sign is 'mixed' and years_sign_consistent counts the years holding
    that dominant sign -- no individual year's own sign is ever 'mixed', so the count is
    0. 4 positive years, 4 negative years, none zero."""
    year_r_last = {
        2018: 0.003,
        2019: 0.003,
        2020: 0.003,
        2021: 0.003,
        2022: -0.003,
        2023: -0.003,
        2024: -0.003,
        2025: -0.003,
    }
    result: H1Result = run_h1(_yearly_r_last_panel(year_r_last), "NIFTY50")
    assert result.years_sign_consistent == 0


def test_verdict_is_survived_when_all_four_kill_criteria_pass() -> None:
    """No fixture elsewhere in this suite produces an edge that clears the cost hurdle
    (the highest is ~8.3 bps against a ~16.53 bps gate), so the SURVIVED branch of
    H1Result's verdict has never been exercised. Plant r_last = 1.0 * r_first + small
    noise across 8 distinct calendar years (10 sessions/year, well above the 3/year floor
    for a year to vote) so ALL FOUR kill criteria pass comfortably, not marginally:
    (1) mean_edge_bps clears 2x the cost hurdle by a wide margin, (2) every one of the 8
    years is individually positive (8/8 >= 6/8), (3) the univariate t-stat is enormous
    (r_last is deterministically driven by r_first), (4) beta_controlling_for_mid barely
    moves once r_mid (independent background noise) is added, since r_last never
    depended on r_mid in the first place.

    Values below were computed with the same OLS-with-intercept / mean(sign(r_first) *
    r_last) formulas the implementation uses, BEFORE running against the implementation:
    mean_edge_bps ~= 52.36, beta ~= 1.0007, t_stat ~= 73.0, all 8 years positive
    (individual year edges roughly 34-66 bps), beta_controlling_for_mid retains ~99.96%
    of the univariate beta with t_stat_controlling_for_mid ~= 72.4."""
    n_years = 8
    sessions_per_year = 10
    n = n_years * sessions_per_year

    rng = np.random.default_rng(7001)
    r_first = rng.standard_normal(n) * 0.006
    noise = rng.standard_normal(n) * 0.0008
    r_last = 1.0 * r_first + noise
    r_mid = np.random.default_rng(7002).standard_normal(n) * 0.0005

    panel = _planted_multiyear_panel(r_first, r_mid, r_last, 2018, n_years, sessions_per_year)
    result: H1Result = run_h1(panel, "NIFTY50")

    # Pinned numerically (not just "greater than"): confirms the fixture landed where
    # calibration predicted, with float32-panel-storage margin (BPS_TOL), not a fluke.
    assert result.mean_edge_bps == pytest.approx(52.3628, abs=1.0)
    assert result.mean_edge_bps > 2 * result.cost_hurdle_bps
    assert result.survives_costs is True
    assert abs(result.t_stat) > 1.96
    assert result.years_sign_consistent >= 6
    assert abs(result.beta_controlling_for_mid) >= 0.5 * abs(result.beta)
    assert abs(result.t_stat_controlling_for_mid) > 1.96

    assert "SURVIVED" in result.explain()
    assert "SURVIVED" in result.to_markdown()
    assert "KILLED" not in result.explain()


def test_verdict_is_killed_when_only_the_cost_gate_fails() -> None:
    """Mirror of the SURVIVED test above: pins the cost gate itself, not just the happy
    path. Same r_last = k * r_first + noise construction, but with k small enough that
    mean_edge_bps lands just under the 2x cost hurdle (~16.53 bps) while the OTHER three
    kill criteria still pass comfortably (this is the canonical H1 failure shape the spec
    describes: 'a statistically significant beta with a sub-cost edge is a kill'). Uses
    more sessions/year (40 vs. 10) so the smaller effect still has enough statistical
    power to be significant and to keep every year's sign the same.

    Values below were computed the same way as the SURVIVED fixture, before running
    against the implementation: mean_edge_bps ~= 14.49 (vs. the ~16.53 bps gate), beta
    ~= 0.3236, t_stat ~= 40.4, all 8 years positive (individual year edges roughly
    12-16 bps), beta_controlling_for_mid retains ~99.85% of the univariate beta with
    t_stat_controlling_for_mid ~= 40.3."""
    n_years = 8
    sessions_per_year = 40
    n = n_years * sessions_per_year

    rng = np.random.default_rng(8001)
    r_first = rng.standard_normal(n) * 0.006
    noise = rng.standard_normal(n) * 0.0008
    r_last = 0.335 * r_first + noise
    r_mid = np.random.default_rng(8002).standard_normal(n) * 0.0005

    panel = _planted_multiyear_panel(r_first, r_mid, r_last, 2018, n_years, sessions_per_year)
    result: H1Result = run_h1(panel, "NIFTY50")

    assert result.mean_edge_bps == pytest.approx(14.4891, abs=1.0)
    assert result.mean_edge_bps < 2 * result.cost_hurdle_bps
    assert result.survives_costs is False

    # The other three criteria still pass -- this is a cost-gate-only failure, not a
    # statistically weak fixture masquerading as one.
    assert abs(result.t_stat) > 1.96
    assert result.years_sign_consistent >= 6
    assert abs(result.beta_controlling_for_mid) >= 0.5 * abs(result.beta)
    assert abs(result.t_stat_controlling_for_mid) > 1.96

    assert "KILLED" in result.explain()
    assert "KILLED" in result.to_markdown()
    assert "SURVIVED" not in result.explain()


def test_explain_and_to_markdown_contain_required_content() -> None:
    result: H1Result = run_h1(_small_panel(), "NIFTY50")

    explanation = result.explain()
    assert result.symbol in explanation
    assert str(result.n_sessions) in explanation
    assert "SURVIVED" in explanation or "KILLED" in explanation

    lowered = explanation.lower()
    assert "edge" in lowered
    assert "year" in lowered
    assert "t_stat" in lowered or "t-stat" in lowered
    assert "controlling" in lowered

    markdown = result.to_markdown()
    assert result.symbol in markdown
    assert "|" in markdown or "#" in markdown


def test_run_h1_is_deterministic() -> None:
    rng = np.random.default_rng(1201)
    n = 80
    r_first = rng.standard_normal(n) * 0.006
    noise = rng.standard_normal(n) * 0.001
    r_last = 0.5 * r_first + noise
    r_mid = _background_r_mid(n)

    panel = _planted_panel(r_first, r_mid, r_last, datetime.date(2020, 1, 1))
    result_a: H1Result = run_h1(panel, "NIFTY50")
    result_b: H1Result = run_h1(panel, "NIFTY50")

    assert result_a.n_sessions == result_b.n_sessions
    assert result_a.beta == result_b.beta
    assert result_a.t_stat == result_b.t_stat
    assert result_a.r_squared == result_b.r_squared
    assert result_a.mean_edge_bps == result_b.mean_edge_bps
    assert result_a.hit_rate == result_b.hit_rate
    assert result_a.cost_hurdle_bps == result_b.cost_hurdle_bps
    assert result_a.survives_costs == result_b.survives_costs
    assert result_a.years_sign_consistent == result_b.years_sign_consistent
    assert result_a.beta_controlling_for_mid == result_b.beta_controlling_for_mid
    assert result_a.t_stat_controlling_for_mid == result_b.t_stat_controlling_for_mid
    assert dict(result_a.by_year) == dict(result_b.by_year)


def test_run_h1_only_sees_pre_holdout_window_via_end_kwarg(
    tmp_path: Path,
) -> None:
    """AMENDED 2026-08-17: run_h1 now takes keyword-only `start`/`end` (mirroring
    Panel.sub) specifically so the caller can pass the pre-holdout window without
    pre-slicing the panel themselves. This test exercises that kwarg directly: run_h1
    on the full panel with `end=pre_holdout_end` must see strictly fewer sessions than
    run_h1 on the same panel with no range, and exactly the sessions that fall on or
    before pre_holdout_end. 26 monthly sessions (2 years + 2 months) ensures the
    pre-holdout window has enough sessions (>= 2) for a non-degenerate OLS fit."""
    n = 26
    # Slight per-session variation (not a constant column) avoids a degenerate
    # zero-variance OLS design in either the full or the sliced pre-holdout panel.
    r_first = np.array([0.001 * (1 + 0.01 * i) for i in range(n)], dtype=np.float64)
    r_mid = _background_r_mid(n)
    r_last = np.array([0.001 * (1 + 0.01 * i) for i in range(n)], dtype=np.float64)

    dates: list[datetime.date] = []
    year = 2022
    month = 1
    for _ in range(n):
        dates.append(datetime.date(year, month, 15))
        month += 1
        if month > 12:
            month = 1
            year += 1

    bars: list[tuple[datetime.date, str, dict[str, float]]] = []
    for d, rf, rm, rl in zip(dates, r_first, r_mid, r_last):
        open_0916 = 100.0
        close_0945 = open_0916 * math.exp(float(rf))
        open_1450 = close_0945 * math.exp(float(rm))
        close_1520 = open_1450 * math.exp(float(rl))
        bars.append((d, "09:16", _bar(open_0916)))
        bars.append((d, "09:45", _bar(open_=close_0945, close=close_0945)))
        bars.append((d, "14:50", _bar(open_=open_1450, close=open_1450)))
        bars.append((d, "15:20", _bar(open_=open_1450, close=close_1520)))

    panel = _build_panel(bars)
    holdout_lock = HoldoutLock(path=tmp_path / "holdout.json", holdout_months=12)
    holdout_start, _holdout_end = holdout_lock.holdout_range(list(panel.dates))

    pre_holdout_dates = [d for d in panel.dates if d < holdout_start]
    assert len(pre_holdout_dates) >= 2, "need >= 2 pre-holdout sessions for a non-degenerate fit"
    pre_holdout_end = max(pre_holdout_dates)

    full_result: H1Result = run_h1(panel, "NIFTY50")
    result: H1Result = run_h1(panel, "NIFTY50", end=pre_holdout_end)

    assert result.n_sessions == len(pre_holdout_dates)
    assert result.n_sessions < full_result.n_sessions == len(panel.dates)
