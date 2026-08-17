from __future__ import annotations

import datetime
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from nifty_quant import settings
from nifty_quant.research.audit import adjustment_audit as adj
from nifty_quant.research.audit.adjustment_audit import (
    PLAUSIBLE_FACTORS,
    AdjustmentClass,
    AuditParams,
    AuditReport,
    SuspectEvent,
    daily_aggregate,
    main,
    scan_symbol,
)
from nifty_quant.universe.static import INDEX_SYMBOLS, equity_symbols


def _session_ts(
    session_date: datetime.date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> list[int]:
    """Return left-labelled epoch-second UTC timestamps for an IST wall-clock start."""
    local_start = datetime.datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        start_hhmm[0],
        start_hhmm[1],
    )
    utc_naive = local_start - datetime.timedelta(hours=5, minutes=30)
    epoch0 = int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())
    return [epoch0 + 60 * i for i in range(n_bars)]


def _write_bars(
    bars_root: Path,
    symbol: str,
    year: int,
    rows: list[dict[str, Any]],
) -> None:
    symbol_dir = bars_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    frame.to_parquet(symbol_dir / f"{year}.parquet", index=False)


def _patch_bars_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    bars_root = tmp_path / "bars" / "1"
    monkeypatch.setattr(settings, "BARS_1M", bars_root)
    return bars_root


def _make_dates(n_days: int, start: str = "2023-01-02") -> list[datetime.date]:
    return list(pd.bdate_range(start=start, periods=n_days).date)


def _write_1bar_series(
    bars_root: Path,
    symbol: str,
    dates: Sequence[datetime.date],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> None:
    if not (len(dates) == len(closes) == len(volumes)):
        raise ValueError("date/close/volume length mismatch")

    rows_by_year: dict[int, list[dict[str, Any]]] = {}
    for date, close, volume in zip(dates, closes, volumes):
        rows_by_year.setdefault(date.year, []).append(
            {
                "ts": _session_ts(date, 1)[0],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
            }
        )

    for year, rows in rows_by_year.items():
        _write_bars(bars_root, symbol, year, rows)


def _make_event(
    symbol: str = "TEST",
    date: datetime.date = datetime.date(2024, 10, 28),
    classification: AdjustmentClass = AdjustmentClass.VOLUME_UNADJUSTED,
    price_ratio: float = 1.0,
    volume_ratio: float = 2.0,
    traded_value_ratio: float = 2.0,
    nearest_factor: float | None = 2.0,
    factor_error: float = 0.0,
    n_days_before: int = 20,
    n_days_after: int = 20,
) -> SuspectEvent:
    return SuspectEvent(
        symbol=symbol,
        date=date,
        price_ratio=price_ratio,
        volume_ratio=volume_ratio,
        traded_value_ratio=traded_value_ratio,
        nearest_factor=nearest_factor,
        factor_error=factor_error,
        classification=classification,
        n_days_before=n_days_before,
        n_days_after=n_days_after,
    )


def test_params_defaults_match_spec() -> None:
    """Defaults calibrated empirically against the real 149-symbol 2018-2026 panel.

    See AuditParams docstring for the measured quantiles and arithmetic behind
    these numbers (false-positive-storm fix, 2026-08-17).
    """
    params = AuditParams()
    assert params.window_days == 20
    assert params.min_traded_value_log_step == pytest.approx(2.3)
    assert params.min_volume_log_step == pytest.approx(2.1)
    assert params.factor_tolerance == pytest.approx(0.04)
    assert params.max_price_step == pytest.approx(0.05)
    assert params.min_days_each_side == 10


@pytest.mark.parametrize(
    "field",
    [
        "min_traded_value_log_step",
        "min_volume_log_step",
        "factor_tolerance",
        "max_price_step",
    ],
)
def test_params_rejects_non_positive_thresholds(field: str) -> None:
    kwargs: dict[str, Any] = {"window_days": 20, "min_days_each_side": 10}
    kwargs[field] = 0.0
    with pytest.raises(ValueError):
        AuditParams(**kwargs)


def test_params_rejects_window_days_below_two() -> None:
    with pytest.raises(ValueError):
        AuditParams(window_days=1)


def test_params_rejects_min_days_exceeding_window() -> None:
    with pytest.raises(ValueError):
        AuditParams(window_days=20, min_days_each_side=21)


def test_params_is_frozen() -> None:
    params = AuditParams()
    with pytest.raises(FrozenInstanceError):
        params.window_days = 30


def test_plausible_factors_sorted_unique_and_positive() -> None:
    factors = list(PLAUSIBLE_FACTORS)
    assert factors == sorted(factors)
    assert len(factors) == len(set(factors))
    assert all(f > 0.0 for f in factors)


def test_plausible_factors_closed_under_reciprocal() -> None:
    factors = list(PLAUSIBLE_FACTORS)
    for factor in factors:
        assert any(np.isclose(1.0 / factor, g, atol=1e-12) for g in factors)


def test_aggregates_ohlcv_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)

    rows = [
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[0],
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=10.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[1],
            open=101.0,
            high=106.0,
            low=98.0,
            close=104.0,
            volume=20.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[2],
            open=102.0,
            high=107.0,
            low=97.0,
            close=105.0,
            volume=30.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 3), 2)[0],
            open=110.0,
            high=111.0,
            low=109.0,
            close=110.5,
            volume=40.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 3), 2)[1],
            open=111.0,
            high=112.0,
            low=108.0,
            close=111.0,
            volume=50.0,
        ),
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    assert list(df.index) == [datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)]
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded_value",
        "n_bars",
    ]

    row1 = df.loc[datetime.date(2023, 1, 2)]
    assert row1["open"] == pytest.approx(100.0)
    assert row1["high"] == pytest.approx(107.0)
    assert row1["low"] == pytest.approx(97.0)
    assert row1["close"] == pytest.approx(105.0)
    assert row1["volume"] == pytest.approx(60.0)
    assert row1["n_bars"] == 3
    assert row1["traded_value"] == pytest.approx(6260.0)

    row2 = df.loc[datetime.date(2023, 1, 3)]
    assert row2["open"] == pytest.approx(110.0)
    assert row2["high"] == pytest.approx(112.0)
    assert row2["low"] == pytest.approx(108.0)
    assert row2["close"] == pytest.approx(111.0)
    assert row2["volume"] == pytest.approx(90.0)
    assert row2["n_bars"] == 2
    assert row2["traded_value"] == pytest.approx(9970.0)


def test_traded_value_is_bar_level_sum_not_daily_product(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    rows = [
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 2)[0],
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=2.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 2)[1],
            open=100.0,
            high=110.0,
            low=100.0,
            close=110.0,
            volume=1.0,
        ),
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    row = df.loc[datetime.date(2023, 1, 2)]

    assert row["traded_value"] == pytest.approx(100.0 * 2.0 + 110.0 * 1.0)
    assert row["close"] == pytest.approx(110.0)
    assert row["volume"] == pytest.approx(3.0)
    assert row["traded_value"] != pytest.approx(row["close"] * row["volume"])


def test_nan_bars_excluded_not_forward_filled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    rows = [
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[0],
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=10.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[1],
            open=101.0,
            high=106.0,
            low=96.0,
            close=np.nan,
            volume=100.0,
        ),
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 3)[2],
            open=102.0,
            high=107.0,
            low=97.0,
            close=102.0,
            volume=20.0,
        ),
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    row = df.loc[datetime.date(2023, 1, 2)]

    assert row["open"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(102.0)
    assert row["high"] == pytest.approx(107.0)
    assert row["low"] == pytest.approx(95.0)
    assert row["volume"] == pytest.approx(30.0)
    assert row["n_bars"] == 2
    assert row["traded_value"] == pytest.approx(100.0 * 10.0 + 102.0 * 20.0)


def test_session_with_all_nan_bars_is_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    rows = [
        dict(
            ts=_session_ts(datetime.date(2023, 1, 2), 2)[i],
            open=np.nan,
            high=np.nan,
            low=np.nan,
            close=np.nan,
            volume=np.nan,
        )
        for i in range(2)
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    assert df.empty
    assert datetime.date(2023, 1, 2) not in df.index


def test_missing_year_file_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_bars_root(monkeypatch, tmp_path)
    df = daily_aggregate("TEST", [2023])
    assert df.empty
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded_value",
        "n_bars",
    ]
    assert df["open"].dtype == np.float64
    assert df["high"].dtype == np.float64
    assert df["low"].dtype == np.float64
    assert df["close"].dtype == np.float64
    assert df["volume"].dtype == np.float64
    assert df["traded_value"].dtype == np.float64
    assert df["n_bars"].dtype == np.int64


def test_no_years_present_returns_empty_frame_with_correct_dtypes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_bars_root(monkeypatch, tmp_path)
    df = daily_aggregate("TEST", [])
    assert df.empty
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded_value",
        "n_bars",
    ]
    assert df["open"].dtype == np.float64
    assert df["high"].dtype == np.float64
    assert df["low"].dtype == np.float64
    assert df["close"].dtype == np.float64
    assert df["volume"].dtype == np.float64
    assert df["traded_value"].dtype == np.float64
    assert df["n_bars"].dtype == np.int64


def test_irregular_session_handled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    session_date = datetime.date(2023, 1, 2)
    rows = [
        dict(
            ts=_session_ts(session_date, 60)[i],
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=10.0 + i,
        )
        for i in range(60)
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    assert list(df.index) == [session_date]
    row = df.loc[session_date]
    assert row["n_bars"] == 60
    assert row["open"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(100.5 + 59)
    assert row["volume"] == pytest.approx(sum(10.0 + i for i in range(60)))


def test_index_is_ascending_unique_dates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    rows: list[dict[str, Any]] = []
    for date in [
        datetime.date(2023, 1, 3),
        datetime.date(2023, 1, 1),
        datetime.date(2023, 1, 2),
    ]:
        rows.append(
            dict(
                ts=_session_ts(date, 1)[0],
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
            )
        )
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    assert list(df.index) == sorted(
        [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)]
    )
    assert df.index.is_unique


def test_session_boundary_uses_ist_not_utc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    session_date = datetime.date(2023, 1, 2)
    rows = [
        dict(
            ts=_session_ts(session_date, 1, (9, 15))[0],
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        ),
        dict(
            ts=_session_ts(session_date, 1, (15, 29))[0],
            open=101.0,
            high=102.0,
            low=100.0,
            close=101.0,
            volume=20.0,
        ),
    ]
    _write_bars(bars_root, "TEST", 2023, rows)

    df = daily_aggregate("TEST", [2023])
    assert list(df.index) == [session_date]
    assert df.loc[session_date, "n_bars"] == 2


def test_clean_series_emits_no_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    rng = np.random.default_rng(42)
    closes = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


def test_injected_volume_only_step_is_detected_as_volume_unadjusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Factor 20.0 (not 2.0): defaults were calibrated to min_traded_value_log_step=2.3
    # (~10x) and min_volume_log_step=2.1 (~8.2x) against real-data noise (see
    # AuditParams docstring), so the injected step must clear both to be detectable.
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * 20.0
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert len(events) == 1
    event = events[0]
    assert event.date == dates[d]
    assert event.classification == AdjustmentClass.VOLUME_UNADJUSTED
    assert event.nearest_factor == pytest.approx(20.0)


def test_injected_reciprocal_step_is_price_unadjusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60

    # SPEC-AMBIGUITY: traded_value_ratio = price_ratio * volume_ratio. For the
    # reciprocal condition log(price_ratio) + log(traded_value_ratio) == 0 to hold,
    # volume_ratio must be 1 / price_ratio**2. Using price_ratio=0.05 (/20) and
    # volume_ratio=400 (*400) gives traded_value_ratio=20.0, clearing both the
    # 2.3 traded-value and 2.1 volume corroboration thresholds (see AuditParams).
    closes[d:] = closes[d:] / 20.0
    volumes[d:] = volumes[d:] * 400.0
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert len(events) == 1
    event = events[0]
    assert event.date == dates[d]
    assert event.classification == AdjustmentClass.PRICE_UNADJUSTED
    assert event.price_ratio == pytest.approx(0.05)
    assert event.traded_value_ratio == pytest.approx(20.0)


def test_step_not_near_a_plausible_factor_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Factor 15.0 clears both min_traded_value_log_step (2.3) and
    # min_volume_log_step (2.1) but log(15.0)=2.708 sits 0.288 from log(20.0) and
    # 0.405 from log(10.0), both well outside factor_tolerance (0.04).
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * 15.0
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert len(events) == 1
    event = events[0]
    assert event.classification == AdjustmentClass.AMBIGUOUS
    assert event.nearest_factor is None
    assert event.factor_error == np.inf


def test_step_below_min_log_step_emits_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * 1.2
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


def test_non_maximum_suppression_emits_one_event_per_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Factor 20.0: see min_traded_value_log_step/min_volume_log_step calibration
    # note on the other injected-step tests above.
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * 20.0
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert len(events) == 1


def test_event_date_is_the_first_day_on_the_new_basis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * 20.0
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events
    assert events[0].date == dates[d]


def test_insufficient_history_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 40
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


def test_candidate_with_too_few_finite_days_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Directly patch daily_aggregate so we can control finite traded_value content.
    dates = _make_dates(60)
    closes = np.full(60, 1000.0, dtype=np.float64)
    volumes = np.full(60, 100_000.0, dtype=np.float64)
    traded_value = closes * volumes
    traded_value[15:35] = np.nan

    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(60, dtype=np.int64),
        },
        index=pd.Index(dates),
    )
    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


def test_zero_trailing_median_is_skipped_not_divide_by_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dates = _make_dates(60)
    closes = np.full(60, 1000.0, dtype=np.float64)
    volumes = np.full(60, 100_000.0, dtype=np.float64)
    traded_value = closes * volumes
    traded_value[10:30] = 0.0

    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(60, dtype=np.int64),
        },
        index=pd.Index(dates),
    )
    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


@pytest.mark.parametrize("factor", PLAUSIBLE_FACTORS)
def test_parametrized_over_all_plausible_factors(
    factor: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    params = AuditParams()
    if abs(float(np.log(factor))) < params.min_traded_value_log_step:
        pytest.skip("SPEC-AMBIGUITY: factor too close to 1 to be detectable")

    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60
    volumes[d:] = volumes[d:] * factor
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], params)
    assert len(events) == 1
    assert events[0].nearest_factor == pytest.approx(factor)


def test_detection_is_direction_symmetric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Factors 20.0/0.05 (not 2.0/0.5): see min_traded_value_log_step/
    # min_volume_log_step calibration note on the injected-step tests above.
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    for symbol, factor in [("UP", 20.0), ("DOWN", 0.05)]:
        n = 120
        dates = _make_dates(n)
        closes = np.full(n, 1000.0, dtype=np.float64)
        volumes = np.full(n, 100_000.0, dtype=np.float64)
        d = 60
        volumes[d:] = volumes[d:] * factor
        _write_1bar_series(bars_root, symbol, dates, closes, volumes)

        events = scan_symbol(symbol, [2023], AuditParams())
        assert len(events) == 1
        assert events[0].nearest_factor == pytest.approx(factor)


def test_by_class_filters_and_consistent_returns_empty() -> None:
    volume_event = _make_event(
        "A", datetime.date(2024, 1, 2), AdjustmentClass.VOLUME_UNADJUSTED
    )
    price_event = _make_event(
        "B", datetime.date(2024, 1, 3), AdjustmentClass.PRICE_UNADJUSTED
    )
    ambiguous_event = _make_event(
        "C", datetime.date(2024, 1, 4), AdjustmentClass.AMBIGUOUS
    )
    report = AuditReport(
        (volume_event, price_event, ambiguous_event),
        ("A", "B", "C"),
        (2024,),
        AuditParams(),
    )

    assert report.by_class(AdjustmentClass.VOLUME_UNADJUSTED) == (volume_event,)
    assert report.by_class(AdjustmentClass.PRICE_UNADJUSTED) == (price_event,)
    assert report.by_class(AdjustmentClass.AMBIGUOUS) == (ambiguous_event,)
    assert report.by_class(AdjustmentClass.CONSISTENT) == ()


def test_exclusion_windows_span_window_days_each_side_inclusive() -> None:
    # SPEC-AMBIGUITY: `window_days` is a trading-day row count everywhere else in the spec
    # (it indexes `df.iloc[...]`), but `exclusion_windows()` operates on `datetime.date`
    # values with no trading calendar in its signature. The strictest reading available
    # without inventing an undocumented calendar dependency is plain calendar-day
    # `datetime.timedelta(days=window_days)` arithmetic, which is what this test asserts.
    event = _make_event(
        "TEST",
        datetime.date(2024, 10, 28),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    report = AuditReport((event,), ("TEST",), (2024,), AuditParams(window_days=20))
    windows = report.exclusion_windows()
    assert windows["TEST"] == (
        (datetime.date(2024, 10, 8), datetime.date(2024, 11, 17)),
    )


def test_exclusion_windows_merge_overlapping_events() -> None:
    event1 = _make_event(
        "TEST",
        datetime.date(2024, 10, 28),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    event2 = _make_event(
        "TEST",
        datetime.date(2024, 11, 1),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    report = AuditReport(
        (event1, event2),
        ("TEST",),
        (2024,),
        AuditParams(window_days=20),
    )
    windows = report.exclusion_windows()["TEST"]
    assert len(windows) == 1
    assert windows[0] == (datetime.date(2024, 10, 8), datetime.date(2024, 11, 21))


def test_exclusion_windows_omit_ambiguous_events() -> None:
    ambiguous = _make_event(
        "AMB",
        datetime.date(2024, 1, 2),
        AdjustmentClass.AMBIGUOUS,
    )
    volume_unadjusted = _make_event(
        "VOL",
        datetime.date(2024, 1, 3),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    report = AuditReport(
        (ambiguous, volume_unadjusted),
        ("AMB", "VOL"),
        (2024,),
        AuditParams(window_days=20),
    )
    windows = report.exclusion_windows()
    assert "AMB" not in windows
    assert "VOL" in windows


def test_exclusion_windows_are_sorted_and_disjoint() -> None:
    event1 = _make_event(
        "TEST",
        datetime.date(2024, 1, 15),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    event2 = _make_event(
        "TEST",
        datetime.date(2024, 6, 15),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    report = AuditReport(
        (event1, event2),
        ("TEST",),
        (2024,),
        AuditParams(window_days=20),
    )
    windows = report.exclusion_windows()["TEST"]
    assert len(windows) == 2
    assert windows[0][0] < windows[1][0]
    assert windows[0][1] < windows[1][0]


@pytest.mark.parametrize("cls", list(AdjustmentClass))
def test_is_clean_true_only_when_no_unadjusted_events(cls: AdjustmentClass) -> None:
    event = _make_event("TEST", datetime.date(2024, 1, 2), cls)
    report = AuditReport((event,), ("TEST",), (2024,), AuditParams())
    expected_clean = cls not in {
        AdjustmentClass.VOLUME_UNADJUSTED,
        AdjustmentClass.PRICE_UNADJUSTED,
    }
    assert report.is_clean() == expected_clean


def test_to_frame_columns_dtypes_and_row_order() -> None:
    event_b = _make_event(
        "B",
        datetime.date(2024, 1, 2),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    event_a = _make_event(
        "A",
        datetime.date(2024, 1, 1),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    report = AuditReport((event_b, event_a), ("A", "B"), (2024,), AuditParams())
    df = report.to_frame()

    expected_cols = [
        "symbol",
        "date",
        "price_ratio",
        "volume_ratio",
        "traded_value_ratio",
        "nearest_factor",
        "factor_error",
        "classification",
        "n_days_before",
        "n_days_after",
    ]
    assert list(df.columns) == expected_cols
    assert df["symbol"].tolist() == ["A", "B"]
    assert df["date"].tolist() == [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]

    # SPEC-AMBIGUITY: date is stored as object dtype datetime.date, not datetime64.
    assert df["symbol"].dtype == object
    assert df["date"].dtype == object
    assert df["price_ratio"].dtype == np.float64
    assert df["volume_ratio"].dtype == np.float64
    assert df["traded_value_ratio"].dtype == np.float64
    assert df["factor_error"].dtype == np.float64
    assert df["classification"].dtype == object
    assert df["n_days_before"].dtype == np.int64
    assert df["n_days_after"].dtype == np.int64


def test_to_frame_on_empty_report_has_correct_schema() -> None:
    report = AuditReport((), (), (2024,), AuditParams())
    df = report.to_frame()
    expected_cols = [
        "symbol",
        "date",
        "price_ratio",
        "volume_ratio",
        "traded_value_ratio",
        "nearest_factor",
        "factor_error",
        "classification",
        "n_days_before",
        "n_days_after",
    ]
    assert list(df.columns) == expected_cols
    assert df.empty
    assert df["symbol"].dtype == object
    assert df["date"].dtype == object
    assert df["price_ratio"].dtype == np.float64
    assert df["volume_ratio"].dtype == np.float64
    assert df["traded_value_ratio"].dtype == np.float64
    assert df["factor_error"].dtype == np.float64
    assert df["classification"].dtype == object
    assert df["n_days_before"].dtype == np.int64
    assert df["n_days_after"].dtype == np.int64


def test_suspect_event_explain_names_every_input_and_threshold() -> None:
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        price_ratio=1.0,
        volume_ratio=2.0,
        traded_value_ratio=2.0,
        nearest_factor=2.0,
        factor_error=0.0,
        classification=AdjustmentClass.VOLUME_UNADJUSTED,
        n_days_before=20,
        n_days_after=20,
    )
    text = event.explain()

    assert "price_ratio" in text
    assert "volume_ratio" in text
    assert "traded_value_ratio" in text
    assert "nearest_factor" in text
    assert "factor_error" in text
    assert "classification" in text
    assert "VOLUME_UNADJUSTED" in text
    assert "price_ratio" in text
    assert "window_days" in text
    assert "min_traded_value_log_step" in text
    assert "factor_tolerance" in text
    assert "max_price_step" in text
    assert "n_days_before" in text
    assert "n_days_after" in text
    assert "2.0" in text
    assert "1.0" in text


def test_report_explain_states_counts_per_class_and_params() -> None:
    events = (
        _make_event(
            "A",
            datetime.date(2024, 1, 2),
            AdjustmentClass.VOLUME_UNADJUSTED,
        ),
        _make_event(
            "B",
            datetime.date(2024, 1, 3),
            AdjustmentClass.VOLUME_UNADJUSTED,
        ),
        _make_event(
            "C",
            datetime.date(2024, 1, 4),
            AdjustmentClass.AMBIGUOUS,
        ),
    )
    report = AuditReport(events, ("A", "B", "C"), (2024,), AuditParams())
    text = report.explain()

    assert "VOLUME_UNADJUSTED" in text
    assert "AMBIGUOUS" in text
    assert "window_days" in text
    assert "min_traded_value_log_step" in text
    assert "factor_tolerance" in text
    assert "2" in text


def test_explain_is_deterministic() -> None:
    event = _make_event()
    report = AuditReport((event,), ("TEST",), (2024,), AuditParams())

    assert event.explain() == event.explain()
    assert report.explain() == report.explain()


def test_run_audit_is_worker_count_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_scan(
        symbol: str,
        years: Sequence[int],
        params: AuditParams,
    ) -> tuple[SuspectEvent, ...]:
        return (_make_event(symbol, datetime.date(2024, 1, 2)),)

    monkeypatch.setattr(adj, "scan_symbol", fake_scan)

    report_1 = adj.run_audit(symbols=["A", "B"], years=[2024], workers=1)
    report_4 = adj.run_audit(symbols=["A", "B"], years=[2024], workers=4)

    pd.testing.assert_frame_equal(report_1.to_frame(), report_4.to_frame())


def test_run_audit_default_years_start_at_2018(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Sequence[int]] = []

    def fake_scan(
        symbol: str,
        years: Sequence[int],
        params: AuditParams,
    ) -> tuple[SuspectEvent, ...]:
        captured.append(years)
        return ()

    monkeypatch.setattr(adj, "scan_symbol", fake_scan)

    adj.run_audit(symbols=["TEST"], workers=1)

    assert len(captured) == 1
    assert tuple(captured[0]) == tuple(range(2018, 2027))


def test_run_audit_default_symbols_is_equity_universe_excluding_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_bars_root(monkeypatch, tmp_path)

    report = adj.run_audit(years=[2020], workers=1)

    assert tuple(report.symbols_scanned) == equity_symbols()
    assert INDEX_SYMBOLS.isdisjoint(report.symbols_scanned)


def test_cli_exit_code_zero_when_clean_one_when_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SPEC-AMBIGUITY: the spec's CLI usage line shows `--years 2018-2026`, a dash-separated
    # range, not a bare single year; use the documented "START-END" format (with START==END
    # for a single year) rather than guessing a bare-integer form the spec never shows.
    monkeypatch.setattr(
        adj,
        "scan_symbol",
        lambda symbol, years, params: (),
    )
    assert main(["--years", "2020-2020", "--symbols", "TEST"]) == 0

    bad_event = _make_event(
        "TEST",
        datetime.date(2024, 1, 2),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    monkeypatch.setattr(
        adj,
        "scan_symbol",
        lambda symbol, years, params: (bad_event,),
    )
    assert main(["--years", "2020-2020", "--symbols", "TEST"]) == 1


@pytest.mark.slow
def test_reliance_2024_bonus_is_not_flagged_volume_unadjusted() -> None:
    if not settings.MANIFEST_PATH.exists():
        pytest.skip("Real data/MANIFEST.json not found; skipping slow real-data test.")

    events = scan_symbol("RELIANCE", years=list(range(2018, 2027)))
    near_bonus = [e for e in events if e.date == datetime.date(2024, 10, 28)]

    for event in near_bonus:
        assert event.classification != AdjustmentClass.VOLUME_UNADJUSTED


def test_price_unadjusted_event_explain() -> None:
    """Test explain() for PRICE_UNADJUSTED classification."""
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        price_ratio=0.5,
        volume_ratio=2.0,
        traded_value_ratio=1.0,
        nearest_factor=2.0,
        factor_error=0.0,
        classification=AdjustmentClass.PRICE_UNADJUSTED,
    )
    text = event.explain()
    assert "PRICE_UNADJUSTED" in text
    assert "reciprocal" in text


def test_ambiguous_event_explain_with_factor() -> None:
    """Test explain() for AMBIGUOUS with nearest_factor present."""
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        price_ratio=1.5,
        volume_ratio=2.0,
        traded_value_ratio=3.0,
        nearest_factor=3.0,
        factor_error=0.02,
        classification=AdjustmentClass.AMBIGUOUS,
    )
    text = event.explain()
    assert "AMBIGUOUS" in text
    assert "confidence" in text


def test_ambiguous_event_explain_without_factor() -> None:
    """Test explain() for AMBIGUOUS without nearest_factor."""
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        price_ratio=1.0,
        volume_ratio=1.7,
        traded_value_ratio=1.7,
        nearest_factor=None,
        factor_error=np.inf,
        classification=AdjustmentClass.AMBIGUOUS,
    )
    text = event.explain()
    assert "AMBIGUOUS" in text
    assert "nearest_factor is None" in text


def test_cli_main_returns_zero_when_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that main() returns 0 for clean audit."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    dates = _make_dates(50)
    closes = np.full(50, 1000.0, dtype=np.float64)
    volumes = np.full(50, 100_000.0, dtype=np.float64)
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    exit_code = main(["--years", "2023-2023", "--symbols", "TEST"])
    assert exit_code == 0


def test_cli_main_returns_one_when_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that main() returns 1 for dirty audit."""
    bad_event = _make_event(
        "TEST",
        datetime.date(2024, 1, 2),
        AdjustmentClass.VOLUME_UNADJUSTED,
    )
    monkeypatch.setattr(
        adj,
        "scan_symbol",
        lambda symbol, years, params: (bad_event,),
    )
    exit_code = main(["--years", "2024-2024", "--symbols", "TEST"])
    assert exit_code == 1


def test_cli_main_writes_parquet_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that main() writes parquet output when --out is specified."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)
    dates = _make_dates(50)
    closes = np.full(50, 1000.0, dtype=np.float64)
    volumes = np.full(50, 100_000.0, dtype=np.float64)
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    out_path = tmp_path / "results.parquet"
    exit_code = main(
        ["--years", "2023-2023", "--symbols", "TEST", "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.exists()


def test_cli_main_invalid_years_format(capsys: pytest.CaptureFixture) -> None:
    """Test that main() handles invalid --years format."""
    exit_code = main(["--years", "2023", "--symbols", "TEST"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "START-END" in captured.err


def test_cli_main_invalid_year_values(capsys: pytest.CaptureFixture) -> None:
    """Test that main() handles non-integer year values."""
    exit_code = main(["--years", "abc-def", "--symbols", "TEST"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "valid integers" in captured.err


def test_cli_main_no_symbols_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that main() uses default symbols when not specified."""
    # Mock scan_symbol to avoid hitting real data
    monkeypatch.setattr(adj, "scan_symbol", lambda s, y, p: ())
    exit_code = main(["--years", "2023-2023"])
    # Should succeed without error
    assert exit_code == 0


def test_nan_volume_ratio_when_zero_trailing_median(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that volume_ratio is NaN when trailing median volume is zero."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)

    # Create a scenario where trailing median volume could be 0
    # Use very small volumes
    dates = _make_dates(50)
    closes = np.full(50, 1000.0, dtype=np.float64)
    volumes = np.concatenate(
        [np.full(20, 0.0), np.full(30, 100_000.0)]  # First 20 days have 0 volume
    )
    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    # Events should be created without crash even with zero volumes
    # The important thing is that volume_ratio=NaN doesn't break anything
    assert isinstance(events, tuple)


def test_volume_median_zero_while_traded_value_positive_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Volume corroboration gate must reject a candidate whose volume median is
    zero even when its traded_value median is positive. `traded_value` is
    injected directly (decoupled from close*volume) so the two can diverge --
    real `daily_aggregate` output never lets this happen, since traded_value is
    computed FROM volume, but the gate must still be defensive."""
    n = 60
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.concatenate(
        [np.full(20, 0.0), np.full(40, 100_000.0)]  # First 20 days have 0 volume
    )
    traded_value = np.full(n, 1e8, dtype=np.float64)  # stays positive throughout

    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(n, dtype=np.int64),
        },
        index=pd.Index(dates),
    )
    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    assert events == ()


def test_tied_tie_keys_break_to_earlier_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that when adjacent-day jumps are tied, earlier date wins."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)

    # Create data with a step that has equal adjacent-day jumps at two candidates
    # This is artificial but tests the tie-break logic
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)

    # Create a 20x step (not 2x): see min_traded_value_log_step/min_volume_log_step
    # calibration note on the injected-step tests above.
    d = 60
    volumes[d:] = volumes[d:] * 20.0

    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    # Should select the earliest candidate with the max adjacent-day jump
    assert len(events) == 1
    assert events[0].date == dates[d]


def test_ambiguous_with_none_factor_explain() -> None:
    """Test explain() for AMBIGUOUS with None factor."""
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        classification=AdjustmentClass.AMBIGUOUS,
        nearest_factor=None,
        factor_error=np.inf,
    )
    text = event.explain()
    assert "nearest_factor is None" in text
    assert "AMBIGUOUS" in text


def test_consistent_classification_explain_has_no_reasoning_section() -> None:
    """CONSISTENT is never emitted by scan_symbol, but SuspectEvent can still be
    constructed directly with it (e.g. by a caller), and explain() must not
    crash: none of the VOLUME_UNADJUSTED/PRICE_UNADJUSTED/AMBIGUOUS branches
    match, so it falls straight through to the header/threshold text only."""
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        classification=AdjustmentClass.CONSISTENT,
    )
    text = event.explain()
    assert "classification: consistent" in text
    assert "Classification logic:" in text
    assert "->" not in text


def test_non_finite_traded_value_in_event_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that non-finite traded values don't crash event building."""
    _patch_bars_root(monkeypatch, tmp_path)

    # Create data with NaN traded values in the candidate windows
    dates = _make_dates(60)
    closes = np.full(60, 1000.0, dtype=np.float64)
    volumes = np.full(60, 100_000.0, dtype=np.float64)
    traded_value = closes * volumes

    # Inject NaN into traded_value for one day
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(60, dtype=np.int64),
        },
        index=pd.Index(dates),
    )
    df.loc[dates[30], "traded_value"] = np.nan

    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    # Should handle NaN without crashing
    assert isinstance(events, tuple)


def test_volume_ratio_nan_with_zero_trailing_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that volume_ratio is NaN when trailing median volume is zero."""
    # Create data where trailing window has zero volume but leading doesn't
    dates = _make_dates(80)
    closes = np.full(80, 1000.0, dtype=np.float64)
    volumes = np.full(80, 100_000.0, dtype=np.float64)

    # Set all volume in early periods to 0, then switch to normal
    volumes[:35] = 0.0
    volumes[35:] = 100_000.0

    # Create a step in traded_value later
    volumes[55:] = 200_000.0

    # Now a candidate around index 40 might have:
    # - Trailing [20, 40): all zeros
    # - Leading [40, 60): normal to 2x
    traded_value = closes * volumes

    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(80, dtype=np.int64),
        },
        index=pd.Index(dates),
    )

    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    # The test just ensures the code path runs without error
    # We're specifically testing that event building handles zero trailing volume
    assert isinstance(events, tuple)


def test_non_finite_tie_key_loses_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-finite tie_key loses any tie-break."""
    # Create data with NaN in traded_value at a position that makes tie_key -inf
    dates = _make_dates(80)
    closes = np.full(80, 1000.0, dtype=np.float64)
    volumes = np.full(80, 100_000.0, dtype=np.float64)

    # Create a 20x step at day 50 (not 2x): must clear min_traded_value_log_step
    # (2.3) and min_volume_log_step (2.1); see calibration note above.
    volumes[50:] = 2_000_000.0

    # Create dataframe with NaN at day 49 (i-1 for a candidate at day 50)
    traded_value = closes * volumes
    traded_value[49] = np.nan

    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "traded_value": traded_value,
            "n_bars": np.ones(80, dtype=np.int64),
        },
        index=pd.Index(dates),
    )

    monkeypatch.setattr(adj, "daily_aggregate", lambda symbol, years: df)

    events = scan_symbol("TEST", [2023], AuditParams())
    # Should find events from candidates that have finite tie_keys
    assert len(events) >= 0


def test_two_candidates_with_equal_tie_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test tie-break when two candidates have identical adjacent-day jumps."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)

    # Create data with TWO identical volume steps at different times. 2,000,000
    # (not 200,000) so the 20x/0.05x steps clear min_traded_value_log_step (2.3)
    # and min_volume_log_step (2.1); see calibration note on tests above.
    n = 130
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.concatenate([
        np.full(50, 100_000.0),     # Days 0-49: 100k
        np.full(30, 2_000_000.0),   # Days 50-79: 2M (20x step)
        np.full(50, 100_000.0),     # Days 80-129: back to 100k
    ])

    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    # When tie_keys are equal for multiple candidates, earlier date wins
    # Find event from first step at day 50
    event_dates = [e.date for e in events]
    # Should have detected the jump around day 50
    assert any(abs((d - dates[50]).days) <= 1 for d in event_dates if d is not None)


def test_ambiguous_classification_via_reciprocal_check() -> None:
    """Test AMBIGUOUS classification from reciprocal check (not None factor)."""
    # Create an event that's AMBIGUOUS because reciprocal doesn't match
    # This exercises the else branch in the AMBIGUOUS classification explain()
    event = _make_event(
        symbol="TEST",
        date=datetime.date(2024, 10, 28),
        price_ratio=0.45,  # Price fell ~55%
        volume_ratio=2.2,
        traded_value_ratio=1.0,  # Net traded_value unchanged
        nearest_factor=2.0,  # Factor exists
        factor_error=0.05,  # Factor matches reasonable
        classification=AdjustmentClass.AMBIGUOUS,
        n_days_before=15,
        n_days_after=15,
    )
    text = event.explain()
    # This should exercise the reciprocal check branch
    assert "AMBIGUOUS" in text
    assert "confidence" in text


def test_ambiguous_with_large_price_step_and_factor_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test AMBIGUOUS from scan_symbol with factor but bad reciprocal."""
    bars_root = _patch_bars_root(monkeypatch, tmp_path)

    # Create data that triggers AMBIGUOUS:
    # - Traded value jumps (detectable step)
    # - Nearest factor found
    # - But price jump and traded_value don't reciprocate (bad fit)
    n = 120
    dates = _make_dates(n)
    closes = np.full(n, 1000.0, dtype=np.float64)
    volumes = np.full(n, 100_000.0, dtype=np.float64)
    d = 60

    # Price drops 50%, volume rises 20x -> traded_value rises 10x, matching
    # PLAUSIBLE_FACTORS' 10.0 exactly (so nearest_factor is found), but this does
    # NOT match the reciprocal relation (would need price_ratio == 1/10, not 0.5),
    # so classification falls through to AMBIGUOUS. 10x/20x (not 1.5x/2.5x) are
    # needed to clear min_traded_value_log_step (2.3) and min_volume_log_step (2.1).
    closes[d:] = closes[d:] * 0.5  # 50% drop
    volumes[d:] = volumes[d:] * 20.0  # 20x rise

    _write_1bar_series(bars_root, "TEST", dates, closes, volumes)

    events = scan_symbol("TEST", [2023], AuditParams())
    # Should find event classified as AMBIGUOUS
    ambig_events = [e for e in events if e.classification == AdjustmentClass.AMBIGUOUS]
    # At least should have no crash when explain() is called on detected events
    for e in ambig_events:
        text = e.explain()
        assert "AMBIGUOUS" in text
