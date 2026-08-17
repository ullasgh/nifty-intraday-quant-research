"""Coverage gap tests for xsec_zscore strategy plugin."""

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pydantic import ValidationError

from nifty_quant.data.panel import Panel
from nifty_quant.strategy.base import PortfolioState
from nifty_quant.strategy.plugins.xsec_zscore import (
    XSecZScoreParams,
    XSecZScoreStrategy,
    _select_and_weight,
)

IST = ZoneInfo("Asia/Kolkata")


def _make_ts(date_obj: dt.date, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = dt.datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, tzinfo=IST)
    return int(local.timestamp())


def _build_panel(sessions, symbols, fields=("open", "close")) -> Panel:
    """Build a Panel directly from sparse per-session bar dictionaries.

    Each session dict is {"date": datetime.date, "bars": {"HH:MM": {symbol: price}}}.
    Bars are sorted by timestamp within each session; sessions are sorted by date.
    A session dict's "bars" need not include every minute -- only the labels you list.
    """
    symbols = tuple(symbols)
    sorted_sessions = sorted(sessions, key=lambda s: s["date"])

    ts_list = []
    row_counts = []
    dates_list = []
    field_arrays = {field: [] for field in fields}

    for sess in sorted_sessions:
        date = sess["date"]
        bars = sess["bars"]
        bar_items = sorted(bars.items(), key=lambda kv: _make_ts(date, kv[0]))

        for hhmm, bar_prices in bar_items:
            ts_list.append(_make_ts(date, hhmm))
            for field in fields:
                field_arrays[field].append([bar_prices.get(sym, np.nan) for sym in symbols])
        row_counts.append(len(bar_items))
        dates_list.append(date)

    ts = np.array(ts_list, dtype=np.int64)
    day_offsets = np.array([0] + list(np.cumsum(row_counts)), dtype=np.int32)
    dates = np.array(dates_list, dtype=object)

    field_arrs = {
        field: np.array(values, dtype=np.float32)
        for field, values in field_arrays.items()
    }
    return Panel(
        fields=field_arrs,
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates,
    )


@dataclass
class _ViewStub:
    ts: int
    session_date: object
    symbols: tuple[str, ...]
    tradable: np.ndarray | None

    def last(self, field, offset=0, *, ffill=False):
        raise NotImplementedError

    def window(self, field, n, *, ffill=False):
        raise NotImplementedError

    def at_time(self, field, hhmm, days_back=0):
        raise NotImplementedError

    def daily(self, field, n_days, symbol=None):
        raise NotImplementedError


def test_parse_hhmm_malformed_colon_count() -> None:
    with pytest.raises(ValidationError, match="expected exactly 'HH:MM'"):
        XSecZScoreParams(entry_time="0916")
    with pytest.raises(ValidationError, match="expected exactly 'HH:MM'"):
        XSecZScoreParams(decision_time="11:00:00")


def test_parse_hhmm_two_digit_validation() -> None:
    with pytest.raises(ValidationError, match="expected exactly 'HH:MM' with two-digit"):
        XSecZScoreParams(entry_time="9:16")
    with pytest.raises(ValidationError, match="expected exactly 'HH:MM' with two-digit"):
        XSecZScoreParams(decision_time="11:5")
    with pytest.raises(ValidationError, match="expected exactly 'HH:MM' with two-digit"):
        XSecZScoreParams(entry_time="0a:16")


def test_parse_hhmm_range_check() -> None:
    with pytest.raises(ValidationError, match="hour must be 00-23"):
        XSecZScoreParams(entry_time="24:00")
    with pytest.raises(ValidationError, match="minute 00-59"):
        XSecZScoreParams(decision_time="11:60")


def test_select_and_weight_neutral_none_same_per_name_magnitude() -> None:
    z_row = np.array([3.0, 2.5, -3.0, 0.0, 0.0])
    gross = 1.0

    w_none = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.5,
        gross=gross,
        neutral="none",
        tradable=None,
    )

    # 2 longs, 1 short -> every selected name gets gross / 3.
    expected = gross / 3.0
    assert np.allclose(w_none[:2], expected)
    assert np.allclose(w_none[2], -expected)
    assert np.allclose(w_none[3:], 0.0)

    w_dollar = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.5,
        gross=gross,
        neutral="dollar",
        tradable=None,
    )

    # In dollar mode, long side and short side each get gross/2.  With 2 longs and
    # 1 short the per-name magnitudes are different, proving "none" is not side-balanced.
    assert w_dollar[0] == 0.25   # (1.0 / 2) / 2
    assert w_dollar[2] == -0.5   # -(1.0 / 2) / 1
    assert w_dollar[0] != abs(w_dollar[2])


def test_select_and_weight_neutral_none_empty_selection_is_zero() -> None:
    # No z-value crosses either threshold, so neutral="none" hits n_sel == 0 and
    # skips straight past the per-name weighting without dividing by zero.
    z_row = np.array([0.5, -0.5, 0.0, 1.0, -1.0])
    w = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.5,
        gross=1.0,
        neutral="none",
        tradable=None,
    )
    np.testing.assert_array_equal(w, np.zeros_like(z_row))


def test_select_and_weight_dollar_short_only() -> None:
    z_row = np.array([-3.0, -2.5, 0.0, 0.5])
    w = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.5,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )

    # No valid longs -> long weights all zero.
    assert np.all(w[z_row >= 0] == 0.0)
    assert w[0] < 0 and w[1] < 0

    short_sum = np.sum(w[w < 0])
    # Dollar neutral allocates gross/2 to the short side when there are no longs.
    assert np.isclose(short_sum, -1.0 / 2.0)


def test_on_decision_signal_shape_mismatch() -> None:
    view = _ViewStub(
        ts=0,
        session_date=dt.date(2024, 1, 1),
        symbols=("A", "B", "C"),
        tradable=None,
    )
    state = PortfolioState(shares=np.zeros(3), cash=0.0, equity=0.0, ts=0)
    strategy = XSecZScoreStrategy(params=XSecZScoreParams())

    with pytest.raises(ValueError, match="does not match view symbols"):
        strategy.on_decision(
            view=view,
            signals={"zscore": np.array([1.0, 2.0])},
            state=state,
        )


def test_precompute_no_entry_bars_returns_all_nan() -> None:
    panel = _build_panel(
        [
            {"date": dt.date(2024, 1, 1), "bars": {"11:00": {"A": 100.0, "B": 200.0}}},
            {"date": dt.date(2024, 1, 2), "bars": {"11:00": {"A": 101.0, "B": 199.0}}},
        ],
        ("A", "B"),
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams())
    signals = strategy.precompute(panel)

    z = signals["zscore"]
    ret = signals["ret"]
    assert z.shape == (2, 2)
    assert ret.shape == (2, 2)

    # This single call covers both branch edges: entry_rows is empty so common_days
    # is empty at the first if (290->302).  Since the filter block is skipped, it is
    # still empty at the second if (302->345), returning all-NaN without price reads.
    assert np.all(np.isnan(z))
    assert np.all(np.isnan(ret))


def test_target_units_session_decision_row_first_bar_zero_weights() -> None:
    symbols = ("A", "B", "C", "D", "E")
    entry = {"A": 100.0, "B": 110.0, "C": 90.0, "D": 105.0, "E": 95.0}
    mid = {"A": 112.0, "B": 108.0, "C": 95.0, "D": 100.0, "E": 90.0}
    decision = {"A": 102.0, "B": 111.0, "C": 92.0, "D": 106.0, "E": 97.0}

    session1_bars = {
        "09:16": entry,
        "10:59": mid,
        "11:00": decision,
    }

    panel = _build_panel(
        [
            {"date": dt.date(2024, 1, 1), "bars": session1_bars},
            {"date": dt.date(2024, 1, 2), "bars": {"11:00": {s: 999.0 for s in symbols}}},
        ],
        symbols,
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))
    result = strategy.target_units(panel, capital=1_000_000.0)

    assert len(result) == 2 * len(symbols)

    # Result rows follow decision_rows order: session 1's symbols first, then session 2's.
    session1_units = result.iloc[:5]["Target_Units"]
    assert not np.all(session1_units == 0.0)

    session2_units = result.iloc[5:]["Target_Units"]
    # session2's decision row is also its session start, so target_units writes zeros
    # and continues before touching z_full[row - 1] or the poisoned 999.0 price.
    assert np.all(session2_units == 0.0)


def test_irregular_session_375_bars_then_muhurat_60_bars_no_crash() -> None:
    symbols = ("A", "B", "C", "D", "E")
    entry_prices = {"A": 100.0, "B": 110.0, "C": 90.0, "D": 105.0, "E": 95.0}
    mid_prices = {"A": 112.0, "B": 108.0, "C": 95.0, "D": 100.0, "E": 90.0}
    decision_prices = {"A": 102.0, "B": 111.0, "C": 92.0, "D": 106.0, "E": 97.0}

    # Session 1: 375 regular one-minute bars from 09:15 to 15:29 inclusive.
    base = dt.datetime(2024, 3, 1, 9, 15)
    bars1 = {}
    for i in range(375):
        dt_i = base + dt.timedelta(minutes=i)
        hhmm = dt_i.strftime("%H:%M")
        if hhmm == "10:59":
            price = mid_prices
        elif hhmm == "11:00":
            price = decision_prices
        else:
            price = entry_prices
        bars1[hhmm] = {sym: price[sym] for sym in symbols}

    # Session 2: 60 Muhurat-style bars with no 09:16 or 11:00 labels.
    base2 = dt.datetime(2024, 3, 2, 18, 15)
    bars2 = {}
    for i in range(60):
        dt_i = base2 + dt.timedelta(minutes=i)
        hhmm = dt_i.strftime("%H:%M")
        bars2[hhmm] = {sym: 100.0 for sym in symbols}

    panel = _build_panel(
        [
            {"date": dt.date(2024, 3, 1), "bars": bars1},
            {"date": dt.date(2024, 3, 2), "bars": bars2},
        ],
        symbols,
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))
    signals = strategy.precompute(panel)

    z = signals["zscore"]
    ret = signals["ret"]
    assert z.shape == (435, len(symbols))
    assert ret.shape == (435, len(symbols))

    # Session 1's cursor row (10:59) has finite z-scores.
    cursor_rows = panel.rows_at_time("10:59")
    assert len(cursor_rows) == 1
    cr = cursor_rows[0]
    assert np.any(np.isfinite(z[cr]))

    # All 60 rows of session 2 (Muhurat) are entirely NaN in both arrays.
    session2_z = z[375:]
    session2_ret = ret[375:]
    assert np.all(np.isnan(session2_z))
    assert np.all(np.isnan(session2_ret))
