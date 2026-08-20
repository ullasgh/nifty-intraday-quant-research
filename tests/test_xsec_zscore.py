"""Tests for the cross-sectional z-score strategy plugin."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from nifty_quant import guards
from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.strategy import registry
from nifty_quant.strategy.plugins.xsec_zscore import (
    XSecZScoreParams,
    XSecZScoreStrategy,
    _cap_top_k,
    _cross_sectional_return_zscore,
    _select_and_weight,
)
from tests.contract_fixtures import minimal_contract

IST = ZoneInfo("Asia/Kolkata")


def _make_ts(date_obj: dt.date, hhmm: str) -> int:
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = dt.datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, tzinfo=IST)
    return int(local.timestamp())


def _build_panel(sessions, symbols, fields=("open", "close")) -> Panel:
    """Build a Panel directly from sparse per-session bar dictionaries.

    Each session dict is {"date": datetime.date, "bars": {"HH:MM": {symbol: price}}}.
    The bars are sorted by timestamp within each session; sessions are sorted by date.
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


def _build_dense_ohlcv_panel(dates, symbols, *, seed=0, bars_per_day=375) -> Panel:
    """Build a dense regular panel with open/high/low/close/volume fields.

    Each day has bars from 09:15 through 15:29 left-labelled, containing entry_time,
    decision_time, cursor row, and exit_time. The 09:16 open and 10:59 close are set
    per symbol with enough cross-sectional spread for the Z-score strategy to select
    names under min_names=4.
    """
    symbols = tuple(symbols)
    n_sym = len(symbols)

    ts_list = []
    row_counts = []
    dates_arr = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
        for i in range(bars_per_day):
            ts_list.append(base + dt.timedelta(minutes=i))
        row_counts.append(bars_per_day)
        dates_arr.append(d)

    ts = np.array([int(t.timestamp()) for t in ts_list], dtype=np.int64)
    day_offsets = np.array([0] + list(np.cumsum(row_counts)), dtype=np.int32)
    dates_arr = np.array(dates_arr, dtype=object)

    n_rows = len(ts_list)
    shape = (n_rows, n_sym)
    open_arr = np.full(shape, np.nan, dtype=np.float32)
    high_arr = np.full(shape, np.nan, dtype=np.float32)
    low_arr = np.full(shape, np.nan, dtype=np.float32)
    close_arr = np.full(shape, np.nan, dtype=np.float32)
    volume_arr = np.full(shape, 1e6, dtype=np.float32)

    entry_prices = np.array([100.0 + i * 5.0 for i in range(n_sym)], dtype=np.float64)
    decision_closes = np.array([110.0 + i * 10.0 for i in range(n_sym)], dtype=np.float64)

    for day_idx in range(len(dates)):
        start = day_idx * bars_per_day
        idx_1059 = start + 104
        idx_1100 = start + 105
        idx_1520 = start + 365

        for i in range(n_sym):
            entry = entry_prices[i]
            dec = decision_closes[i]

            segment = slice(start, start + bars_per_day)
            open_arr[segment, i] = entry
            close_arr[segment, i] = entry
            high_arr[segment, i] = entry + 0.2
            low_arr[segment, i] = entry - 0.2

            open_arr[idx_1059, i] = dec
            close_arr[idx_1059, i] = dec
            high_arr[idx_1059, i] = dec + 0.2
            low_arr[idx_1059, i] = dec - 0.2

            open_arr[idx_1100, i] = dec + 0.5
            close_arr[idx_1100, i] = dec + 0.5
            high_arr[idx_1100, i] = dec + 1.0
            low_arr[idx_1100, i] = dec + 0.1

            open_arr[idx_1520, i] = entry + 1.0
            close_arr[idx_1520, i] = entry + 1.0
            high_arr[idx_1520, i] = entry + 1.2
            low_arr[idx_1520, i] = entry + 0.8

    return Panel(
        fields={
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": volume_arr,
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def test_causal_precompute():
    symbols = ("A", "B", "C", "D", "E", "F")
    sessions = []
    for day in range(1, 7):
        d = dt.date(2024, 1, day)
        sessions.append(
            {
                "date": d,
                "bars": {
                    "09:16": {sym: 100.0 + i * 10 + day * 0.1 for i, sym in enumerate(symbols)},
                    "11:00": {sym: 105.0 + i * 5 + day * 0.3 for i, sym in enumerate(symbols)},
                },
            }
        )

    panel = _build_panel(sessions, symbols, fields=("open", "close"))
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=6))
    y0 = strategy.precompute(panel)

    for k in (1, 3):
        open_arr = panel.field("open").copy()
        close_arr = panel.field("close").copy()

        start = int(panel.day_offsets[k + 1])
        rng = np.random.default_rng(42 + k)
        open_arr[start:] = rng.uniform(50.0, 200.0, open_arr[start:].shape).astype(np.float32)
        close_arr[start:] = rng.uniform(50.0, 200.0, close_arr[start:].shape).astype(np.float32)

        scrambled = Panel(
            fields={"open": open_arr, "close": close_arr},
            symbols=panel.symbols,
            ts=panel.ts,
            day_offsets=panel.day_offsets,
            dates=panel.dates,
        )

        with guards.strictness(guards.Strictness.FULL):
            y1 = strategy.precompute(scrambled)

        row_end = int(panel.day_offsets[k + 1])
        assert np.array_equal(y0["zscore"][:row_end], y1["zscore"][:row_end], equal_nan=True)
        assert np.array_equal(y0["ret"][:row_end], y1["ret"][:row_end], equal_nan=True)

    returns = np.arange(20, dtype=np.float64).reshape(5, 4) * 0.01
    with guards.strictness(guards.Strictness.FULL):
        out = _cross_sectional_return_zscore(returns, min_names=2, robust=False)

    assert isinstance(out, np.ndarray)
    assert out.shape == returns.shape


def test_xsec_zscore_and_momentum_reversal_sign_flip():
    symbols = ("A", "B", "C", "D", "E")
    entry = {sym: 100.0 for sym in symbols}
    decision = {
        "A": 110.0,
        "B": 105.0,
        "C": 100.0,
        "D": 95.0,
        "E": 90.0,
    }

    panel = _build_panel(
        [
            {
                "date": dt.date(2024, 1, 1),
                "bars": {"09:16": entry, "10:59": decision, "11:00": decision},
            }
        ],
        symbols,
        fields=("open", "close"),
    )

    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))
    signals = strategy.precompute(panel)
    row = panel.rows_at_time("11:00")[0]
    # The engine reads the precomputed signal at cursor = decision_row - 1 (see
    # engine.py); precompute() writes there, not at the decision-time row itself.
    z_row = signals["zscore"][row - 1]

    r = np.log(np.array([110.0, 105.0, 100.0, 95.0, 90.0]) / 100.0)
    expected_z = (r - r.mean()) / r.std(ddof=0)
    np.testing.assert_allclose(z_row, expected_z, atol=1e-6, rtol=1e-6)

    z_manual = np.array([3.0, 2.0, -3.0, -2.0, 0.0, 0.0])
    kwargs = dict(
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.5,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )
    w_momentum = _select_and_weight(z_manual, mode="momentum", **kwargs)
    w_reversal = _select_and_weight(z_manual, mode="reversal", **kwargs)

    assert np.allclose(w_momentum, -w_reversal)
    assert np.any(w_momentum != 0.0)


def test_missing_checkpoint_no_borrowing():
    symbols = ("A", "B", "C", "D")
    sessions = [
        {
            "date": dt.date(2024, 1, 1),
            "bars": {
                "09:16": {sym: 100.0 for sym in symbols},
                "11:00": {"A": 105.0, "B": 95.0, "C": 110.0, "D": 90.0},
            },
        },
        {
            "date": dt.date(2024, 1, 2),
            "bars": {
                "09:16": {sym: 100.0 for sym in symbols},
                "10:15": {"A": 101.0, "B": 100.0, "C": 99.0, "D": 102.0},
            },
        },
    ]

    panel = _build_panel(sessions, symbols, fields=("open", "close"))
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=4))
    signals = strategy.precompute(panel)

    b_date = dt.date(2024, 1, 2)
    b_idx = np.flatnonzero(panel.dates == b_date)[0]
    row_start = int(panel.day_offsets[b_idx])
    row_end = int(panel.day_offsets[b_idx + 1])

    assert np.all(np.isnan(signals["zscore"][row_start:row_end]))
    assert np.all(np.isnan(signals["ret"][row_start:row_end]))

    w = _select_and_weight(
        signals["zscore"][row_start],
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.1,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )
    np.testing.assert_array_equal(w, np.zeros(len(symbols), dtype=np.float64))


def test_min_names_blanks_small_finite_cross_section():
    symbols = tuple(f"S{i:02d}" for i in range(25))
    entry = {sym: 100.0 + (i + 1) * 0.01 for i, sym in enumerate(symbols)}
    decision = {symbols[i]: 110.0 + i for i in range(3)}

    panel = _build_panel(
        [
            {
                "date": dt.date(2024, 1, 1),
                "bars": {"09:16": entry, "10:59": decision, "11:00": decision},
            }
        ],
        symbols,
        fields=("open", "close"),
    )

    strategy = XSecZScoreStrategy(params=XSecZScoreParams())
    signals = strategy.precompute(panel)
    row = panel.rows_at_time(strategy.params.decision_time)[0]
    # Signal lives at cursor = decision_row - 1 (the row the engine reads).
    z_row = signals["zscore"][row - 1]

    assert np.all(np.isnan(z_row))

    w = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.1,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )
    np.testing.assert_array_equal(w, np.zeros(len(symbols), dtype=np.float64))


def test_dollar_neutrality():
    z_row = np.array([3.0, 2.5, 2.0, 1.7, -3.0, -2.5, -2.0, -1.7, 0.5, -0.5, 1.0, -1.0])
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

    pos_sum = float(np.sum(w[w > 0]))
    neg_sum = float(np.sum(w[w < 0]))
    assert abs(pos_sum + neg_sum) < 1e-9
    assert np.sum(np.abs(w)) <= 1.0 + 1e-9
    assert np.isclose(np.sum(np.abs(w)), 1.0)


def test_weight_caps_and_cap_top_k_regression():
    # 6a: n_max_side selects exactly the n most extreme long names.
    z_row = np.array([3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 0.0, 0.0, 0.0, 0.0])
    w = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=3,
        max_weight=0.5,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )

    assert np.sum(w > 0) == 3
    np.testing.assert_array_equal(np.flatnonzero(w > 0), np.array([0, 1, 2]))
    assert np.all(w[3:] == 0.0)

    # 6b: max_weight clipping is applied after equal-weight sizing.
    z_row_2 = np.array([2.0, 1.7, 0.0, 0.0])
    w_2 = _select_and_weight(
        z_row_2,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=None,
        max_weight=0.05,
        gross=1.0,
        neutral="dollar",
        tradable=None,
    )
    assert np.all(np.abs(w_2) <= 0.05 + 1e-12)
    assert np.all(np.isfinite(w_2))
    assert np.isclose(w_2[0], 0.05)
    assert np.isclose(w_2[1], 0.05)

    # 6c: regression for _cap_top_k fix — do not promote False masks.
    mask = np.array([False, True, False, True, False])
    score = np.array([10.0, 1.0, 20.0, 2.0, 30.0])
    result = _cap_top_k(mask, score, 5)
    np.testing.assert_array_equal(result, mask)

    zero_result = _cap_top_k(mask, score, 0)
    np.testing.assert_array_equal(zero_result, np.zeros_like(mask, dtype=bool))

    normal_mask = np.array([True, False, True, False, True, False, True, True])
    normal_score = np.array([10.0, 0.0, 8.0, 0.0, 6.0, 0.0, 4.0, 2.0])
    normal_result = _cap_top_k(normal_mask, normal_score, 3)
    expected = np.array([True, False, True, False, True, False, False, False])
    np.testing.assert_array_equal(normal_result, expected)


def test_tradability_does_not_promote_lower_ranked_name():
    z_row = np.array([9.0, 8.0, 7.0, 6.0, 0.0, 0.0])
    tradable = np.ones(len(z_row), dtype=bool)
    tradable[0] = False

    w = _select_and_weight(
        z_row,
        mode="momentum",
        long_threshold=1.5,
        short_threshold=-1.5,
        n_max_side=3,
        max_weight=0.5,
        gross=1.0,
        neutral="dollar",
        tradable=tradable,
    )

    assert w[0] == 0.0
    assert np.sum(w > 0) == 2


def test_no_lookahead_precompute_is_causal():
    # THIS TEST MUST FAIL against the pre-fix code (open[t] leaking into row t-1)
    # and pass after the fix.
    symbols = tuple(f"S{i}" for i in range(6))
    sessions = []
    for day in range(1, 5):
        d = dt.date(2024, 1, day)
        entry = {sym: 100.0 + i for i, sym in enumerate(symbols)}
        decision_close = {sym: 110.0 + i * 3 for i, sym in enumerate(symbols)}
        decision_open = {sym: 999.0 + i for i, sym in enumerate(symbols)}
        sessions.append(
            {
                "date": d,
                "bars": {
                    "09:16": entry,
                    "10:59": decision_close,
                    "11:00": decision_open,
                },
            }
        )

    panel = _build_panel(sessions, symbols, fields=("open", "close"))
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=6))
    original_signals = strategy.precompute(panel)
    decision_rows = panel.rows_at_time(strategy.params.decision_time)

    for t in decision_rows:
        cursor = t - 1
        for variant in ("scale", "nan"):
            open_arr = panel.field("open").copy()
            close_arr = panel.field("close").copy()
            if variant == "scale":
                open_arr[t:] *= 1e6
                close_arr[t:] *= 1e6
            else:
                open_arr[t:] = np.nan
                close_arr[t:] = np.nan

            mutated = Panel(
                fields={"open": open_arr, "close": close_arr},
                symbols=panel.symbols,
                ts=panel.ts,
                day_offsets=panel.day_offsets,
                dates=panel.dates,
            )
            mutated_signals = strategy.precompute(mutated)
            np.testing.assert_array_equal(
                original_signals["zscore"][cursor],
                mutated_signals["zscore"][cursor],
            )
            np.testing.assert_array_equal(
                original_signals["ret"][cursor],
                mutated_signals["ret"][cursor],
            )


def test_strategy_actually_opens_positions_through_run_backtest():
    symbols = tuple(f"S{i}" for i in range(8))
    dates = [dt.date(2024, 1, d) for d in range(5, 10)]
    panel = _build_dense_ohlcv_panel(dates, symbols, seed=42)
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=4))
    result = run_backtest(strategy, panel, BacktestConfig(), contract=minimal_contract())
    assert result.n_trades > 0


def test_signal_present_at_engine_cursor_row():
    symbols = ("A", "B", "C", "D", "E")
    entry = {sym: 100.0 for sym in symbols}
    decision = {
        "A": 110.0,
        "B": 105.0,
        "C": 100.0,
        "D": 95.0,
        "E": 90.0,
    }
    panel = _build_panel(
        [
            {
                "date": dt.date(2024, 1, 1),
                "bars": {"09:16": entry, "10:59": decision, "11:00": decision},
            }
        ],
        symbols,
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))
    signals = strategy.precompute(panel)
    decision_row = panel.rows_at_time(strategy.params.decision_time)[0]
    cursor = decision_row - 1
    z_row = signals["zscore"][cursor]
    assert np.any(np.isfinite(z_row))
    assert np.any(z_row != 0.0)


def test_decision_row_zero_does_not_wrap():
    # Session 1 (date1) has a SINGLE bar, stamped with the decision-time label
    # ("11:00"), placed at absolute row 0 of the whole panel -- day_offsets[0] is
    # always 0, so this row is simultaneously "decision row is absolute row 0" and
    # "decision row is the first row of its own session". It has no entry-time bar
    # (impossible to have one earlier, since row 0 is the panel's first row), so it
    # can never appear in `common_days` and must get no signal. Session 2 (date2,
    # strictly later so `ts` stays monotonic) has a real 09:16 -> 10:59 -> 11:00
    # triple, giving precompute a legitimate common day whose cursor row (10:59) is
    # NOT the entry row, so its return is non-degenerate. The 11:00 bar's own price
    # is poisoned to confirm it is never read (causal fix under test).
    symbols = ("A", "B")
    date1 = dt.date(2024, 1, 1)
    date2 = dt.date(2024, 1, 2)

    panel = _build_panel(
        [
            {"date": date1, "bars": {"11:00": {"A": 100.0, "B": 200.0}}},
            {
                "date": date2,
                "bars": {
                    "09:16": {"A": 100.0, "B": 200.0},
                    "10:59": {"A": 110.0, "B": 210.0},
                    "11:00": {"A": 999.0, "B": 999.0},
                },
            },
        ],
        symbols,
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=2))
    signals = strategy.precompute(panel)

    # Session 1's only row (absolute row 0, also the panel's LAST-row wraparound
    # target under -1 indexing) gets no signal: no entry bar existed for it.
    assert np.all(np.isnan(signals["zscore"][0]))
    assert np.all(np.isnan(signals["ret"][0]))

    # Session 2's real signal lands at its cursor row (the 10:59 bar, row 2), not
    # at its entry row (row 1) and not at its own decision row (row 3, the panel's
    # actual last row).
    assert np.all(np.isfinite(signals["zscore"][2]))
    assert np.all(np.isfinite(signals["ret"][2]))
    assert np.all(np.isnan(signals["zscore"][1]))
    assert np.all(np.isnan(signals["ret"][1]))

    # The panel's actual final row (session 2's own decision row, holding the
    # poisoned 999.0 price) must stay NaN: nothing wrapped a row-0-adjacent value
    # into it, and the decision row itself is never a write target.
    assert np.all(np.isnan(signals["zscore"][-1]))
    assert np.all(np.isnan(signals["ret"][-1]))


def test_decision_row_at_session_start_is_skipped():
    # Session 1 (date1) has a real 09:16 -> 10:59 -> 11:00 triple, so precompute
    # legitimately writes a signal at its cursor row (10:59, not its own first row).
    # Session 2 (date2, strictly later) has a SINGLE bar stamped with the
    # decision-time label ("11:00") as its own first (and only) row -- the
    # "decision row is the first bar of its session" shape, with no entry bar in
    # that session, sitting immediately after session 1. Its 11:00 price is
    # poisoned to make any accidental read/leak obvious.
    symbols = ("A", "B")
    date1 = dt.date(2024, 1, 1)
    date2 = dt.date(2024, 1, 2)

    panel = _build_panel(
        [
            {
                "date": date1,
                "bars": {
                    "09:16": {"A": 100.0, "B": 200.0},
                    "10:59": {"A": 110.0, "B": 210.0},
                    "11:00": {"A": 999.0, "B": 999.0},
                },
            },
            {"date": date2, "bars": {"11:00": {"A": 999.0, "B": 999.0}}},
        ],
        symbols,
        fields=("open", "close"),
    )
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=2))
    signals = strategy.precompute(panel)

    # Session 1's real signal lands at its cursor row (10:59, row 1), unaffected by
    # session 2 existing at all.
    assert np.all(np.isfinite(signals["zscore"][1]))
    assert np.all(np.isfinite(signals["ret"][1]))

    # Session 1's own entry row (row 0) and decision row (row 2) are not write
    # targets.
    assert np.all(np.isnan(signals["zscore"][0]))
    assert np.all(np.isnan(signals["ret"][0]))
    assert np.all(np.isnan(signals["zscore"][2]))
    assert np.all(np.isnan(signals["ret"][2]))

    # Session 2's decision-at-session-start row (row 3, the panel's last row, and
    # also the poisoned price) never gets a signal: it has no entry pairing and is
    # its own session's first row.
    assert np.all(np.isnan(signals["zscore"][3]))
    assert np.all(np.isnan(signals["ret"][3]))


def test_never_reads_0915_bar():
    symbols = tuple(f"S{i}" for i in range(6))
    entry = {sym: 100.0 + i for i, sym in enumerate(symbols)}
    decision = {sym: 110.0 + (i % 3) * 5 for i, sym in enumerate(symbols)}
    poison = {sym: 1e9 for sym in symbols}

    panel_with_0915 = _build_panel(
        [
            {
                "date": dt.date(2024, 1, 1),
                "bars": {
                    "09:15": poison,
                    "09:16": entry,
                    "10:59": decision,
                    "11:00": decision,
                },
            }
        ],
        symbols,
        fields=("open", "close"),
    )
    panel_without_0915 = _build_panel(
        [
            {
                "date": dt.date(2024, 1, 1),
                "bars": {
                    "09:16": entry,
                    "10:59": decision,
                    "11:00": decision,
                },
            }
        ],
        symbols,
        fields=("open", "close"),
    )

    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=6))
    sig_with = strategy.precompute(panel_with_0915)
    sig_without = strategy.precompute(panel_without_0915)

    cursor_with = panel_with_0915.rows_at_time("11:00")[0] - 1
    cursor_without = panel_without_0915.rows_at_time("11:00")[0] - 1

    np.testing.assert_array_equal(
        sig_with["zscore"][cursor_with],
        sig_without["zscore"][cursor_without],
    )
    np.testing.assert_array_equal(
        sig_with["ret"][cursor_with],
        sig_without["ret"][cursor_without],
    )

    ret_finite = sig_with["ret"][cursor_with]
    finite_vals = ret_finite[np.isfinite(ret_finite)]
    assert np.all(np.abs(finite_vals) < 1e6)


def test_params_validation():
    with pytest.raises(ValidationError):
        XSecZScoreParams(entry_time="11:00", decision_time="11:00")

    with pytest.raises(ValidationError):
        XSecZScoreParams(entry_time="11:00", decision_time="10:00")

    with pytest.raises(ValidationError):
        XSecZScoreParams(exit_time="11:00")

    with pytest.warns(UserWarning):
        p = XSecZScoreParams(entry_time="09:15")
    assert p.entry_time == "09:15"


def test_two_configs_one_class():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs" / "strategies"

    cfg1 = yaml.safe_load((config_dir / "xsec_zscore.yaml").read_text())
    cfg2 = yaml.safe_load((config_dir / "xsec_zscore_afternoon.yaml").read_text())

    strat1 = registry.build(cfg1)
    strat2 = registry.build(cfg2)

    assert isinstance(strat1, XSecZScoreStrategy)
    assert isinstance(strat2, XSecZScoreStrategy)

    assert strat1.params.entry_time == "09:16"
    assert strat2.params.entry_time == "11:00"
    assert strat1.params.decision_time == "11:00"
    assert strat2.params.decision_time == "14:30"
    assert strat1.params.long_threshold == 1.5
    assert strat2.params.long_threshold == 1.8

    symbols = tuple(f"S{i:02d}" for i in range(20))
    rM = np.linspace(-0.04, 0.04, num=20)
    rA = np.sin(np.arange(20)) * 0.03

    sessions = []
    for d in (dt.date(2024, 1, 10), dt.date(2024, 1, 11), dt.date(2024, 1, 12)):
        p09 = {sym: 100.0 for sym in symbols}
        p11 = {}
        p14 = {}
        p15 = {}
        for i, sym in enumerate(symbols):
            p11_i = 100.0 * np.exp(rM[i])
            p14_i = p11_i * np.exp(rA[i])
            p15_i = p14_i * 1.01
            p11[sym] = p11_i
            p14[sym] = p14_i
            p15[sym] = p15_i
        sessions.append(
            {
                "date": d,
                "bars": {
                    "09:16": p09,
                    "10:59": p11,
                    "11:00": p11,
                    "14:29": p14,
                    "14:30": p14,
                    "15:20": p15,
                },
            }
        )

    panel = _build_panel(sessions, symbols, fields=("open", "close"))
    z1 = strat1.precompute(panel)["zscore"]
    z2 = strat2.precompute(panel)["zscore"]

    assert np.any(np.isfinite(z1))
    assert np.any(np.isfinite(z2))
    assert not np.array_equal(z1, z2, equal_nan=True)


def test_target_units_contract():
    symbols = ("A", "B", "C", "D", "E")
    sessions = []
    for day in (1, 2):
        d = dt.date(2024, 2, day)
        sessions.append(
            {
                "date": d,
                "bars": {
                    "09:16": {sym: 100.0 + i for i, sym in enumerate(symbols)},
                    "11:00": {sym: 105.0 + i for i, sym in enumerate(symbols)},
                },
            }
        )

    panel = _build_panel(sessions, symbols, fields=("open", "close"))
    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))

    result = strategy.target_units(panel, capital=1_000_000.0)

    assert list(result.index.names) == ["Timestamp", "Ticker"]
    assert list(result.columns) == ["Target_Units"]
    assert result["Target_Units"].dtype == np.float64
    assert len(result) == len(panel.rows_at_time("11:00")) * len(symbols)


@pytest.mark.slow
def test_real_data_slow() -> None:
    from nifty_quant.data import panel_builder
    from nifty_quant.data.manifest import Manifest

    try:
        manifest = Manifest.load()
    except FileNotFoundError:
        pytest.skip("Real data/MANIFEST.json not found; skipping slow real-data test.")

    candidates = sorted(sym for sym, cov in manifest.coverage.items() if 2024 in cov.years)
    if len(candidates) < 20:
        pytest.skip(f"Need at least 20 symbols with 2024 data, found {len(candidates)}.")
    selected = candidates[:20]

    panel_builder.build_panel(freq="1", years=[2024], symbols=selected, progress=False)
    spec = PanelSpec(
        freq="1",
        fields=("open", "close"),
        symbols=tuple(selected),
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
    )
    panel = load_panel(spec)
    strategy = XSecZScoreStrategy(params=XSecZScoreParams())
    signals = strategy.precompute(panel)
    z = signals["zscore"]

    decision_rows = panel.rows_at_time(strategy.params.decision_time)
    cursor_rows = decision_rows - 1
    n_sessions_with_signal = int(np.sum(np.any(np.isfinite(z[cursor_rows]), axis=1)))
    assert n_sessions_with_signal > 200

    for row in cursor_rows:
        row_z = z[row]
        finite = row_z[np.isfinite(row_z)]
        if finite.size >= strategy.params.min_names:
            assert abs(float(np.mean(finite))) < 1e-6
            assert abs(float(np.std(finite)) - 1.0) < 1e-6

    assert not np.any(np.isinf(z))
