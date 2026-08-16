"""Tests for the cross-sectional z-score strategy plugin."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from nifty_quant import guards
from nifty_quant.data.panel import Panel, PanelSpec, load_panel
from nifty_quant.strategy import registry
from nifty_quant.strategy.plugins.xsec_zscore import (
    XSecZScoreParams,
    XSecZScoreStrategy,
    _cap_top_k,
    _cross_sectional_return_zscore,
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
        open_arr[start:] = rng.uniform(50.0, 200.0, size=open_arr[start:].shape).astype(np.float32)
        close_arr[start:] = rng.uniform(50.0, 200.0, size=close_arr[start:].shape).astype(np.float32)

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
                "bars": {"09:16": entry, "11:00": decision},
            }
        ],
        symbols,
        fields=("open", "close"),
    )

    strategy = XSecZScoreStrategy(params=XSecZScoreParams(min_names=5))
    signals = strategy.precompute(panel)
    row = panel.rows_at_time("11:00")[0]
    z_row = signals["zscore"][row]

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
                "bars": {"09:16": entry, "11:00": decision},
            }
        ],
        symbols,
        fields=("open", "close"),
    )

    strategy = XSecZScoreStrategy(params=XSecZScoreParams())
    signals = strategy.precompute(panel)
    row = panel.rows_at_time(strategy.params.decision_time)[0]
    z_row = signals["zscore"][row]

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
                    "11:00": p11,
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
    n_sessions_with_signal = int(np.sum(np.any(np.isfinite(z[decision_rows]), axis=1)))
    assert n_sessions_with_signal > 200

    for row in decision_rows:
        row_z = z[row]
        finite = row_z[np.isfinite(row_z)]
        if finite.size >= strategy.params.min_names:
            assert abs(float(np.mean(finite))) < 1e-6
            assert abs(float(np.std(finite)) - 1.0) < 1e-6

    assert not np.any(np.isinf(z))
