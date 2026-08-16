from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

from nifty_quant.calendar import TradingCalendar
from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.data.validate import (
    Severity,
    check_all_nan_columns,
    check_ohlc_consistency,
    check_session_lengths,
    check_stale_bars,
    check_timestamps,
    check_unexplained_gaps,
    check_zero_or_negative_prices,
    circuit_locked_mask,
    tradable_mask,
    validate_panel,
)


def _session_ts(
    session_date: datetime.date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> np.ndarray:
    """One timestamp per minute for an IST session start."""
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start = datetime.datetime.combine(
        session_date,
        datetime.time(start_hhmm[0], start_hhmm[1]),
        tzinfo=ist,
    )
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)],
        dtype=np.int64,
    )


def _default_fields(
    n_rows: int,
    n_symbols: int,
    *,
    open_price: float = 100.0,
    high_price: float = 101.0,
    low_price: float = 99.0,
    close_price: float = 100.5,
    volume_value: float = 1000.0,
) -> dict[str, np.ndarray]:
    return {
        "open": np.full((n_rows, n_symbols), open_price, dtype=np.float64),
        "high": np.full((n_rows, n_symbols), high_price, dtype=np.float64),
        "low": np.full((n_rows, n_symbols), low_price, dtype=np.float64),
        "close": np.full((n_rows, n_symbols), close_price, dtype=np.float64),
        "volume": np.full((n_rows, n_symbols), volume_value, dtype=np.float64),
    }


def _make_panel(
    session_dates: list[datetime.date],
    bars_per_session: list[int] | None = None,
    symbols: tuple[str, ...] = ("A", "B"),
) -> Panel:
    if bars_per_session is None:
        bars_per_session = [375] * len(session_dates)
    if len(bars_per_session) != len(session_dates):
        raise ValueError("bars_per_session must match session_dates")

    ts = np.concatenate(
        [
            _session_ts(session_date, n_bars)
            for session_date, n_bars in zip(session_dates, bars_per_session)
        ]
    ).astype(np.int64)
    day_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(bars_per_session).astype(np.int32)]
    )
    fields = _default_fields(int(ts.size), len(symbols))
    return Panel(fields, symbols, ts, day_offsets, np.asarray(session_dates, dtype=object))


def test_check_timestamps_non_monotonic() -> None:
    # `Panel.__init__` now enforces strictly-increasing/unique `ts` via
    # `guards.check_sorted_unique`, so a corrupted panel can no longer be built through
    # the public Panel constructor. `check_timestamps` only reads `.ts`, `.day_offsets`,
    # and `.dates` off its argument, so a duck-typed stand-in exercises the check
    # in isolation without going through Panel's contract guard.
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    s1 = _session_ts(dates[0], 375)
    s2 = _session_ts(dates[1], 375)
    ts = np.concatenate([s1, s2]).astype(np.int64)
    ts[400] = ts[399]  # duplicate inside second session

    day_offsets = np.array([0, 375, 750], dtype=np.int32)
    panel = SimpleNamespace(ts=ts, day_offsets=day_offsets, dates=np.asarray(dates, dtype=object))

    findings = check_timestamps(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_timestamps"
    assert finding.severity is Severity.ERROR
    assert finding.symbol is None
    assert finding.session == datetime.date(2024, 1, 2)
    assert finding.count == 1


def test_check_ohlc_consistency_high_below_low() -> None:
    panel = _make_panel([datetime.date(2024, 1, 1)], [375])
    panel.field("high")[4, 0] = 98.5

    findings = check_ohlc_consistency(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_ohlc_consistency"
    assert finding.severity is Severity.ERROR
    assert finding.symbol == "A"
    assert finding.count == 1
    assert check_zero_or_negative_prices(panel) == []


def test_check_zero_price_not_double_flagged() -> None:
    panel = _make_panel([datetime.date(2024, 1, 1)], [375])
    panel.field("low")[7, 1] = 0.0

    findings = check_zero_or_negative_prices(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_zero_or_negative_prices"
    assert finding.severity is Severity.ERROR
    assert finding.symbol == "B"
    assert finding.count == 1
    assert check_ohlc_consistency(panel) == []


def test_check_stale_bars_12_bar_run() -> None:
    panel = _make_panel([datetime.date(2024, 1, 1)], [375])
    for field_name in ("open", "high", "low", "close"):
        panel.field(field_name)[10:22, 1] = 100.0
    panel.field("volume")[10:22, 1] = 0.0

    findings = check_stale_bars(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_stale_bars"
    assert finding.severity is Severity.WARN
    assert finding.symbol == "B"
    assert finding.session == datetime.date(2024, 1, 1)
    assert finding.count == 12


def test_check_unexplained_gaps_threshold() -> None:
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    panel = _make_panel(dates, [375, 375])
    close = panel.field("close")
    open_ = panel.field("open")
    close[374, 0] = 100.0
    close[374, 1] = 100.0
    open_[375, 0] = 130.0
    open_[375, 1] = 100.0

    findings = check_unexplained_gaps(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_unexplained_gaps"
    assert finding.severity is Severity.WARN
    assert finding.symbol == "A"
    assert finding.session == datetime.date(2024, 1, 2)
    assert finding.count == 1

    small_gap_panel = _make_panel(dates, [375, 375])
    close_small = small_gap_panel.field("close")
    open_small = small_gap_panel.field("open")
    close_small[374, 0] = 100.0
    close_small[374, 1] = 100.0
    open_small[375, 0] = 102.0
    open_small[375, 1] = 100.0
    assert check_unexplained_gaps(small_gap_panel) == []


def test_check_all_nan_columns_volume_for_symbol() -> None:
    panel = _make_panel([datetime.date(2024, 1, 1)], [375])
    panel.field("volume")[:, 0] = np.nan

    findings = check_all_nan_columns(panel)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "check_all_nan_columns"
    assert finding.severity is Severity.ERROR
    assert finding.symbol == "A"
    assert "volume" in finding.detail


def test_check_session_lengths_muhurat_classification() -> None:
    panel = _make_panel([datetime.date(2024, 11, 1)], [60])
    calendar = TradingCalendar(panel.ts)

    with_calendar = check_session_lengths(panel, calendar=calendar)
    assert len(with_calendar) == 1
    assert with_calendar[0].severity is Severity.INFO

    without_calendar = check_session_lengths(panel, calendar=None)
    assert len(without_calendar) == 1
    assert without_calendar[0].severity is Severity.WARN


def test_circuit_locked_mask_bands() -> None:
    high = np.array([[105.0], [103.0], [105.0]])
    low = np.array([[105.0], [98.0], [100.0]])
    close = np.array([[101.0], [101.0], [101.0]])
    prev_close = np.array([[100.0], [100.0], [100.0]])

    mask = circuit_locked_mask(high, low, close, prev_close)
    assert mask[0, 0]
    assert not mask[1, 0]
    assert not mask[2, 0]


def test_tradable_mask_is_subset_of_present() -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [3, 3],
    )
    panel.field("close")[1, 0] = np.nan

    present = ~np.isnan(panel.field("close"))
    tradable = tradable_mask(panel)

    assert not np.any(tradable & ~present)
    assert np.all(tradable <= present)


def test_validate_panel_end_to_end() -> None:
    panel = _make_panel(
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        [375, 375],
    )
    panel.field("low")[3, 0] = 0.0
    for field_name in ("open", "high", "low", "close"):
        panel.field(field_name)[20:32, 1] = 100.0
    panel.field("volume")[20:32, 1] = 0.0

    report = validate_panel(panel)
    counts = report.by_check()
    assert counts["check_zero_or_negative_prices"] >= 1
    assert counts["check_stale_bars"] >= 1
    assert len(report.findings) == 2
    assert report.n_rows == panel.n_rows()
    assert report.n_symbols == panel.n_symbols()
    assert report.n_sessions == panel.n_days()

    parsed = json.loads(report.to_json())
    assert parsed["n_rows"] == report.n_rows
    assert parsed["n_symbols"] == report.n_symbols
    assert parsed["n_sessions"] == report.n_sessions
    assert len(parsed["findings"]) == len(report.findings)


@pytest.mark.slow
def test_validate_panel_real_2024() -> None:
    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    panel = load_panel(spec)
    calendar = TradingCalendar(panel.ts)

    validate_panel(panel, calendar=calendar)

    findings = check_session_lengths(panel, calendar=calendar)
    for session_date in (
        datetime.date(2024, 3, 2),
        datetime.date(2024, 5, 18),
        datetime.date(2024, 11, 1),
    ):
        matches = [f for f in findings if f.session == session_date]
        assert len(matches) == 1
        finding = matches[0]
        assert finding.severity is Severity.INFO
        assert finding.symbol is None
        assert finding.check == "check_session_lengths"
