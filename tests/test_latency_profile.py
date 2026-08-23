"""Tests for `nifty_quant.research.latency.latency_profile` (spec: task brief,
"latency profile" -- criterion 5 of `research.lens.Lens.verdict` has been
NOT_EVALUATED for every caller in the repo because nothing produced its input).

Each test is written from the spec alone; no implementation was read before
these were drafted.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.features import core as core_features
from nifty_quant.research import lens as lens_module
from nifty_quant.research.latency import latency_profile, session_aware_shift

_IST = "Asia/Kolkata"


def _day_offsets(bars_per_session: list[int]) -> np.ndarray:
    offsets = [0]
    for n in bars_per_session:
        offsets.append(offsets[-1] + n)
    return np.array(offsets, dtype=np.int64)


def _decaying_signal_fixture(
    seed: int = 0,
    n_sessions: int = 4,
    bars_per_session: int = 120,
    n_symbols: int = 6,
    rho: float = 0.85,
    effect: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """feature/close/day_offsets where the feature's OWN autocorrelation decays
    at rate `rho` per bar, and the forward return depends on the CONTEMPORANEOUS
    feature value only. A stale (lagged) copy of the feature therefore predicts
    the current return with correlation ~= rho**k at lag k, so the conditional
    bucket spread must shrink as lag grows.
    """
    rng = np.random.default_rng(seed)
    day_offsets = _day_offsets([bars_per_session] * n_sessions)
    n_rows = n_sessions * bars_per_session

    feature = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    close = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)

    for s in range(n_sessions):
        start = s * bars_per_session
        x = np.zeros((bars_per_session, n_symbols), dtype=np.float64)
        x[0] = rng.standard_normal(n_symbols)
        innovation_scale = np.sqrt(1.0 - rho**2)
        for t in range(1, bars_per_session):
            x[t] = rho * x[t - 1] + innovation_scale * rng.standard_normal(n_symbols)
        feature[start : start + bars_per_session] = x

        ret_true = effect * x
        c = np.ones((bars_per_session, n_symbols), dtype=np.float64)
        for t in range(bars_per_session - 1):
            c[t + 1] = c[t] * np.exp(ret_true[t])
        close[start : start + bars_per_session] = c

    return feature, close, day_offsets


def _noise_fixture(
    seed: int = 0,
    n_sessions: int = 4,
    bars_per_session: int = 120,
    n_symbols: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """feature and forward returns are independent pure noise: no lag should
    show a real edge."""
    rng = np.random.default_rng(seed)
    day_offsets = _day_offsets([bars_per_session] * n_sessions)
    n_rows = n_sessions * bars_per_session

    feature = rng.standard_normal((n_rows, n_symbols))
    ret_noise = 0.001 * rng.standard_normal((n_rows, n_symbols))

    close = np.ones((n_rows, n_symbols), dtype=np.float64)
    for s in range(n_sessions):
        start = s * bars_per_session
        for t in range(bars_per_session - 1):
            row = start + t
            close[row + 1] = close[row] * np.exp(ret_noise[row])

    return feature, close, day_offsets


def _session_grid(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_chunks: list[np.ndarray] = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bars_per_session, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    n_days = len(dates)
    day_offsets = np.arange(
        0, (n_days + 1) * bars_per_session, bars_per_session, dtype=np.int32
    )
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _build_verdict_panel(
    n_sessions: int = 4, bars_per_session: int = 30, n_symbols: int = 6, seed: int = 0
) -> Panel:
    dates = [dt.date(2024, 1, 2 + i) for i in range(n_sessions)]
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    n_rows = n_sessions * bars_per_session

    rng = np.random.default_rng(seed)
    log_rets = 0.0008 * rng.standard_normal((n_rows, n_symbols))
    close = np.ones((n_rows, n_symbols), dtype=np.float32)
    for s in range(n_sessions):
        start = s * bars_per_session
        for t in range(bars_per_session - 1):
            row = start + t
            close[row + 1] = close[row] * np.exp(log_rets[row])
    volume = np.full((n_rows, n_symbols), 100_000.0, dtype=np.float32)

    symbols = tuple(f"SYM{i}" for i in range(n_symbols))
    return Panel(
        fields={"close": close, "volume": volume},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


# ---------------------------------------------------------------------------
# 1. Decaying signal -> spread shrinks monotonically across lags 0, 1, 2.
# ---------------------------------------------------------------------------


def test_latency_profile_decaying_signal_shrinks_monotonically() -> None:
    feature, close, day_offsets = _decaying_signal_fixture()

    profile = latency_profile(feature, close, day_offsets, horizon=1)

    assert set(profile.keys()) == {0, 1, 2}
    spread0 = abs(profile[0])
    spread1 = abs(profile[1])
    spread2 = abs(profile[2])
    assert spread0 > spread1 > spread2


# ---------------------------------------------------------------------------
# 2. Pure noise -> all three lags indistinguishable from zero.
# ---------------------------------------------------------------------------


def test_latency_profile_noise_fixture_all_lags_near_zero() -> None:
    feature, close, day_offsets = _noise_fixture()

    profile = latency_profile(feature, close, day_offsets, horizon=1)

    for k, spread_bps in profile.items():
        assert abs(spread_bps) < 5.0, f"lag {k} spread {spread_bps} not near zero"


# ---------------------------------------------------------------------------
# 3. The shift never crosses a session boundary (irregular 60/105/375 fixture).
# ---------------------------------------------------------------------------


def test_session_aware_shift_never_crosses_session_boundary() -> None:
    session_lengths = [60, 105, 375]
    day_offsets = _day_offsets(session_lengths)
    n_rows = int(day_offsets[-1])
    n_symbols = 5

    rng = np.random.default_rng(1)
    feature = rng.standard_normal((n_rows, n_symbols))

    for k in (1, 2):
        shifted = session_aware_shift(feature, k, day_offsets)
        for start, end in zip(day_offsets[:-1], day_offsets[1:]):
            start, end = int(start), int(end)
            # First k rows of the session must be NaN: no valid lagged value
            # exists without reaching into the PRECEDING session.
            assert np.all(np.isnan(shifted[start : start + k]))
            # And the shift is causal: the first defined row equals the
            # session's own first row (drawn from k bars ago, never from a
            # neighbouring session).
            if end - start > k:
                np.testing.assert_array_equal(shifted[start + k], feature[start])


def test_session_aware_shift_zero_lag_is_identity() -> None:
    day_offsets = _day_offsets([60, 105, 375])
    n_rows = int(day_offsets[-1])
    feature = np.random.default_rng(2).standard_normal((n_rows, 5))

    shifted = session_aware_shift(feature, 0, day_offsets)

    np.testing.assert_array_equal(shifted, feature)


# ---------------------------------------------------------------------------
# 4. Fewer than 5 symbols raises, rather than silently returning zeros.
# ---------------------------------------------------------------------------


def test_latency_profile_fewer_than_five_symbols_raises() -> None:
    n_symbols = 3
    day_offsets = _day_offsets([50])
    n_rows = int(day_offsets[-1])
    rng = np.random.default_rng(3)
    feature = rng.standard_normal((n_rows, n_symbols))
    close = np.exp(np.cumsum(0.001 * rng.standard_normal((n_rows, n_symbols)), axis=0))

    with pytest.raises(ValueError):
        latency_profile(feature, close, day_offsets, horizon=1)


# ---------------------------------------------------------------------------
# 5. Shape compatibility with lens.verdict's criterion 5 (the point of the task).
# ---------------------------------------------------------------------------


def test_latency_profile_shape_matches_lens_verdict_contract() -> None:
    feature, close, day_offsets = _decaying_signal_fixture()
    lags = (0, 1, 2)

    profile = latency_profile(feature, close, day_offsets, horizon=1, lags=lags)

    assert list(profile.keys()) == list(lags)
    for k in lags:
        assert isinstance(k, int)
    for value in profile.values():
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_latency_profile_makes_lens_criterion_5_not_not_evaluated() -> None:
    panel = _build_verdict_panel()
    close = panel.field("close").astype(np.float64)
    feature_values = core_features.log_returns(close, day_offsets=panel.day_offsets)

    profile = latency_profile(feature_values, close, panel.day_offsets, horizon=1)

    lens = lens_module.Lens(panel, seed=0)
    verdict = lens.verdict(
        "H_TEST_latency",
        "return_1",
        1,
        latency_profile=profile,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=50,
    )

    criterion_5_reason = verdict.reasons[4]
    assert criterion_5_reason.startswith("5. Latency profile criterion:")
    assert "NOT_EVALUATED" not in criterion_5_reason
    assert ("PASS" in criterion_5_reason) or ("FAIL" in criterion_5_reason)
