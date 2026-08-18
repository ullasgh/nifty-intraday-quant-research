from __future__ import annotations

import datetime as dt
import re
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import test_lens as tl

from nifty_quant.data.panel import Panel
from nifty_quant.research import expectancy
from nifty_quant.research import lens as lens_module
from nifty_quant.research.lens import (
    Lens,
    StabilityReport,
)

_IST = ZoneInfo("Asia/Kolkata")


def _make_session_grid_2_symbols(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-symbol session grid with uniform bars_per_session.

    Returns (ts, day_offsets, dates_arr) matching the Panel constructor's expectations.
    """
    n_sessions = len(dates)
    n_rows = n_sessions * bars_per_session
    ts = np.zeros(n_rows, dtype=np.int64)
    day_offsets = np.zeros(n_sessions + 1, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)

    row = 0
    for d, date in enumerate(dates):
        day_offsets[d] = row
        # IST trading day starts 9:15, 1-min bars
        start = dt.datetime.combine(date, dt.time(9, 15), tzinfo=_IST)
        for b in range(bars_per_session):
            ts[row] = int(start.timestamp()) + b * 60
            row += 1
    day_offsets[n_sessions] = row

    return ts, day_offsets, dates_arr


def _make_session_grid_irregular(
    dates: list[dt.date], bars_per_session_list: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a session grid with irregular per-session bar counts.

    Returns (ts, day_offsets, dates_arr) matching the Panel constructor's expectations.
    """
    n_sessions = len(dates)
    n_rows = sum(bars_per_session_list)
    ts = np.zeros(n_rows, dtype=np.int64)
    day_offsets = np.zeros(n_sessions + 1, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)

    row = 0
    for d, (date, n_bars) in enumerate(zip(dates, bars_per_session_list)):
        day_offsets[d] = row
        start = dt.datetime.combine(date, dt.time(9, 15), tzinfo=_IST)
        for b in range(n_bars):
            ts[row] = int(start.timestamp()) + b * 60
            row += 1
    day_offsets[n_sessions] = row

    return ts, day_offsets, dates_arr


def _make_panel_2_symbols(
    close: np.ndarray,
    volume: np.ndarray,
    symbols: tuple[str, ...],
    dates: list[dt.date],
    bars_per_session: int,
) -> Panel:
    """Build a 2-symbol Panel from close/volume arrays."""
    ts, day_offsets, dates_arr = _make_session_grid_2_symbols(dates, bars_per_session)
    return Panel(
        fields={
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _make_panel_irregular(
    close: np.ndarray,
    volume: np.ndarray,
    symbols: tuple[str, ...],
    dates: list[dt.date],
    bars_per_session_list: list[int],
) -> Panel:
    """Build a Panel with irregular per-session bar counts."""
    ts, day_offsets, dates_arr = _make_session_grid_irregular(dates, bars_per_session_list)
    return Panel(
        fields={
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _find_concentration_threshold() -> float | None:
    """Locate a module-level numeric constant in lens.py whose name suggests it is the
    criterion-4 concentration-ratio threshold. Returns None if not found (current code has
    only an inline literal `2`, no named constant -- see test_concentration_threshold_is_named
    below, which is the RED signal for this item)."""
    pattern = re.compile(
        r"concentration.*(threshold|ratio|ratio_cutoff|cutoff)|"
        r"(threshold|cutoff).*concentration",
        re.IGNORECASE,
    )
    hits = {
        name: value
        for name, value in vars(lens_module).items()
        if pattern.search(name)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    if len(hits) == 1:
        return float(next(iter(hits.values())))
    return None


def _make_expectancy_table(spread_bps: float) -> expectancy.ExpectancyTable:
    """Build a minimal ExpectancyTable with only spread_bps set meaningfully."""
    return expectancy.ExpectancyTable(
        buckets=(),
        horizon=1,
        feature_name="test",
        n_total=0,
        cost_hurdle_bps=0.0,
        spread_bps=spread_bps,
        spread_t=0.0,
        survives_costs=False,
    )


def test_prior_adv_uses_turnover_not_share_volume() -> None:
    """Item 1: UNITS regression guard (primary — exact prior_adv equality).

    S_HIGH has 100x lower share volume but 100x higher price than S_LOW, so their
    rupee turnover is identical. The fixed implementation must compute prior_adv
    from close*volume, not raw volume.
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    bars_per_session = 3
    n_rows = len(dates) * bars_per_session
    n_symbols = 2

    close = np.zeros((n_rows, n_symbols), dtype=np.float64)
    volume = np.zeros((n_rows, n_symbols), dtype=np.float64)

    # S_HIGH: close=100.0, volume=10.0 -> turnover=1000.0
    close[:, 0] = 100.0
    volume[:, 0] = 10.0

    # S_LOW: close=1.0, volume=1000.0 -> turnover=1000.0
    close[:, 1] = 1.0
    volume[:, 1] = 1000.0

    panel = _make_panel_2_symbols(
        close, volume, ("S_HIGH", "S_LOW"), dates, bars_per_session
    )
    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")

    with patch.object(
        expectancy,
        "expectancy_by_liquidity_decile",
        wraps=expectancy.expectancy_by_liquidity_decile,
    ) as mock_ebld:
        lens.stability(feature_obj, horizon=1)

        assert mock_ebld.called, (
            "Lens.stability() must delegate to expectancy.expectancy_by_liquidity_decile(); "
            "the current implementation still uses its own inline np.quantile loop and never "
            "calls this function at all."
        )

        call = mock_ebld.call_args
        prior_adv = call.kwargs.get("prior_adv")
        if prior_adv is None and len(call.args) > 3:
            prior_adv = call.args[3]

        assert prior_adv is not None, "prior_adv must be passed to expectancy_by_liquidity_decile"
        assert prior_adv.shape == (9, 2), f"Expected shape (9, 2), got {prior_adv.shape}"

        # Session 0 (rows 0-2): all NaN
        assert np.all(np.isnan(prior_adv[0:3, :])), "Session 0 prior_adv must be all NaN"

        # Session 1 (rows 3-5): prior = session 0 only, both columns == 1000.0
        np.testing.assert_allclose(prior_adv[3:6, :], 1000.0, rtol=1e-5)

        # Session 2 (rows 6-8): prior = sessions 0-1, both columns == 1000.0
        np.testing.assert_allclose(prior_adv[6:9, :], 1000.0, rtol=1e-5)

        # Key assertion: S_HIGH and S_LOW prior_adv are elementwise equal
        np.testing.assert_allclose(prior_adv[3:, 0], prior_adv[3:, 1], rtol=1e-5)


def test_end_to_end_same_decile_for_tied_turnover() -> None:
    """Item 1b: UNITS regression guard (secondary — end-to-end SAME decile, no mock).

    S_HIGH and S_LOW have identical turnover (50_000.0) despite very different
    share volumes and prices. Under the fixed implementation they must land in the
    same liquidity decile, giving exactly 4 distinct populated deciles.
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5)]
    bars_per_session = 5
    n_rows = len(dates) * bars_per_session
    n_symbols = 5

    close = np.zeros((n_rows, n_symbols), dtype=np.float64)
    volume = np.zeros((n_rows, n_symbols), dtype=np.float64)

    # S_HIGH: close=1000.0, volume=50.0 -> turnover=50_000.0
    close[:, 0] = 1000.0
    volume[:, 0] = 50.0

    # S_LOW: close=50.0, volume=1000.0 -> turnover=50_000.0
    close[:, 1] = 50.0
    volume[:, 1] = 1000.0

    # A1: close=1.0, volume=5_000.0 -> turnover=5_000.0
    close[:, 2] = 1.0
    volume[:, 2] = 5_000.0

    # A2: close=1.0, volume=500_000.0 -> turnover=500_000.0
    close[:, 3] = 1.0
    volume[:, 3] = 500_000.0

    # A3: close=1.0, volume=50_000_000.0 -> turnover=50_000_000.0
    close[:, 4] = 1.0
    volume[:, 4] = 50_000_000.0

    ts, day_offsets, dates_arr = _make_session_grid_2_symbols(dates, bars_per_session)
    # Need to extend to 5 symbols
    ts_5 = ts
    day_offsets_5 = day_offsets
    dates_arr_5 = dates_arr

    panel = Panel(
        fields={
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=("S_HIGH", "S_LOW", "A1", "A2", "A3"),
        ts=ts_5,
        day_offsets=day_offsets_5,
        dates=dates_arr_5,
    )

    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")
    stab_report = lens.stability(feature_obj, horizon=1)

    assert len(stab_report.by_liquidity_decile) == 4, (
        f"Expected 4 distinct populated deciles (S_HIGH and S_LOW tied), "
        f"got {len(stab_report.by_liquidity_decile)}"
    )


def test_prior_adv_causality_and_session_nan() -> None:
    """Items 2, 3, 4: causality guard + session-0-NaN + prior excludes current session.

    X jumps from 10.0 to 10_000_000.0 in session 3. The prior_adv for session 3
    must still be 10.0 (mean of sessions 0-2), not the current session's value.
    Session 0 must be all NaN. Session 4's prior_adv must be the mean of sessions
    0-3: (10+10+10+10_000_000)/4 = 2_500_007.5.
    """
    dates = [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
        dt.date(2024, 1, 8),
    ]
    bars_per_session = 4
    n_rows = len(dates) * bars_per_session
    n_symbols = 2

    close = np.ones((n_rows, n_symbols), dtype=np.float64)
    volume = np.zeros((n_rows, n_symbols), dtype=np.float64)

    # Session 0: X=10.0, Y=500.0
    volume[0:4, 0] = 10.0
    volume[0:4, 1] = 500.0

    # Session 1: X=10.0, Y=500.0
    volume[4:8, 0] = 10.0
    volume[4:8, 1] = 500.0

    # Session 2: X=10.0, Y=500.0
    volume[8:12, 0] = 10.0
    volume[8:12, 1] = 500.0

    # Session 3: X=10_000_000.0, Y=500.0
    volume[12:16, 0] = 10_000_000.0
    volume[12:16, 1] = 500.0

    # Session 4: X=10_000_000.0, Y=500.0
    volume[16:20, 0] = 10_000_000.0
    volume[16:20, 1] = 500.0

    panel = _make_panel_2_symbols(close, volume, ("X", "Y"), dates, bars_per_session)
    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")

    with patch.object(
        expectancy,
        "expectancy_by_liquidity_decile",
        wraps=expectancy.expectancy_by_liquidity_decile,
    ) as mock_ebld:
        lens.stability(feature_obj, horizon=1)

        assert mock_ebld.called, (
            "Lens.stability() must delegate to expectancy.expectancy_by_liquidity_decile(); "
            "the current implementation still uses its own inline np.quantile loop and never "
            "calls this function at all."
        )

        call = mock_ebld.call_args
        prior_adv = call.kwargs.get("prior_adv")
        if prior_adv is None and len(call.args) > 3:
            prior_adv = call.args[3]

        assert prior_adv is not None, "prior_adv must be passed to expectancy_by_liquidity_decile"
        assert prior_adv.shape == (20, 2), f"Expected shape (20, 2), got {prior_adv.shape}"

        # Item 3: Session 0 (rows 0-3): all NaN
        assert np.all(np.isnan(prior_adv[0:4, :])), "Session 0 prior_adv must be all NaN"

        # Session 1 (rows 4-7): X=10.0, Y=500.0
        np.testing.assert_allclose(prior_adv[4:8, 0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(prior_adv[4:8, 1], 500.0, rtol=1e-5)

        # Session 2 (rows 8-11): X=10.0, Y=500.0
        np.testing.assert_allclose(prior_adv[8:12, 0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(prior_adv[8:12, 1], 500.0, rtol=1e-5)

        # Item 2 & 4: Session 3 (rows 12-15): X=10.0 (NOT 10_000_000.0), Y=500.0
        np.testing.assert_allclose(prior_adv[12:16, 0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(prior_adv[12:16, 1], 500.0, rtol=1e-5)

        # Session 4 (rows 16-19): X=2_500_007.5, Y=500.0
        np.testing.assert_allclose(prior_adv[16:20, 0], 2_500_007.5, rtol=1e-5)
        np.testing.assert_allclose(prior_adv[16:20, 1], 500.0, rtol=1e-5)


def test_prior_adv_irregular_sessions_no_fixed_stride() -> None:
    """Item 5: irregular sessions, no fixed bar-count stride.

    Session 0 has 60 bars, session 1 has 200 bars. The prior_adv for session 1
    must be derived from session 0's 60 bars (all 1000.0), not any assumed fixed
    stride like 375 or 200.
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    bars_per_session_list = [60, 200]
    n_rows = sum(bars_per_session_list)
    n_symbols = 1

    close = np.ones((n_rows, n_symbols), dtype=np.float64)
    volume = np.zeros((n_rows, n_symbols), dtype=np.float64)

    # Session 0: volume=1000.0
    volume[0:60, 0] = 1000.0

    # Session 1: volume=5000.0 (value doesn't matter for the assertion)
    volume[60:260, 0] = 5000.0

    panel = _make_panel_irregular(
        close, volume, ("S00",), dates, bars_per_session_list
    )
    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")

    with patch.object(
        expectancy,
        "expectancy_by_liquidity_decile",
        wraps=expectancy.expectancy_by_liquidity_decile,
    ) as mock_ebld:
        lens.stability(feature_obj, horizon=1)

        assert mock_ebld.called, (
            "Lens.stability() must delegate to expectancy.expectancy_by_liquidity_decile(); "
            "the current implementation still uses its own inline np.quantile loop and never "
            "calls this function at all."
        )

        call = mock_ebld.call_args
        prior_adv = call.kwargs.get("prior_adv")
        if prior_adv is None and len(call.args) > 3:
            prior_adv = call.args[3]

        assert prior_adv is not None, "prior_adv must be passed to expectancy_by_liquidity_decile"
        assert prior_adv.shape[0] == 260, (
            f"Expected 260 rows (60+200), got {prior_adv.shape[0]}"
        )

        # Session 0 (rows 0-59): all NaN
        assert np.all(np.isnan(prior_adv[0:60, 0])), "Session 0 prior_adv must be all NaN"

        # Session 1 (rows 60-259): all == 1000.0
        np.testing.assert_allclose(prior_adv[60:260, 0], 1000.0, rtol=1e-5)


def test_prior_adv_nan_turnover_propagates() -> None:
    """Item 6: NaN turnover propagates, never forward-filled or zero-filled.

    Session 0 has one NaN volume bar. The prior_adv for session 1 must be the
    NaN-aware mean of the 3 finite bars: (100+100+100)/3 = 100.0, NOT 75.0
    (which would result from zero-filling the NaN).
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    bars_per_session = 4
    n_rows = len(dates) * bars_per_session
    n_symbols = 1

    close = np.ones((n_rows, n_symbols), dtype=np.float64)
    volume = np.zeros((n_rows, n_symbols), dtype=np.float64)

    # Session 0: [100.0, NaN, 100.0, 100.0]
    volume[0, 0] = 100.0
    volume[1, 0] = np.nan
    volume[2, 0] = 100.0
    volume[3, 0] = 100.0

    # Session 1: 100.0 constant
    volume[4:8, 0] = 100.0

    panel = _make_panel_2_symbols(close, volume, ("Z",), dates, bars_per_session)
    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")

    with patch.object(
        expectancy,
        "expectancy_by_liquidity_decile",
        wraps=expectancy.expectancy_by_liquidity_decile,
    ) as mock_ebld:
        lens.stability(feature_obj, horizon=1)

        assert mock_ebld.called, (
            "Lens.stability() must delegate to expectancy.expectancy_by_liquidity_decile(); "
            "the current implementation still uses its own inline np.quantile loop and never "
            "calls this function at all."
        )

        call = mock_ebld.call_args
        prior_adv = call.kwargs.get("prior_adv")
        if prior_adv is None and len(call.args) > 3:
            prior_adv = call.args[3]

        assert prior_adv is not None, "prior_adv must be passed to expectancy_by_liquidity_decile"

        # Session 1 (rows 4-7): NaN-aware mean of session 0's finite bars = 100.0
        np.testing.assert_allclose(prior_adv[4:8, 0], 100.0, rtol=1e-5)

        # Explicitly assert NOT 75.0 (zero-filled mean)
        assert not np.isclose(prior_adv[4, 0], 75.0), (
            "prior_adv must be NaN-aware mean (100.0), not zero-filled mean (75.0)"
        )


def test_method_cross_sectional_rank_mandatory_part_a() -> None:
    """Item 7 Part A: confirm the default method trap exists in the dependency.

    With fewer than 5000 rows, the default method="expanding_quantile" silently
    produces all-zero spreads. This confirms the trap is real.
    """
    panel = tl._build_panel((2024,), sessions_per_year=3, bars_per_session=10)
    lens = Lens(panel, seed=0)
    feature_obj = lens.feature("return_1")
    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, horizon=1)

    prior_adv = np.where(np.isfinite(feature_obj.values), 1.0, np.nan)

    tables = expectancy.expectancy_by_liquidity_decile(
        feature_obj.values,
        fwd,
        panel.day_offsets,
        prior_adv,
    )

    for table in tables.values():
        assert table.spread_bps == 0.0, (
            "Default method='expanding_quantile' should produce all-zero spreads "
            "on a panel with fewer than 5000 rows"
        )


def test_method_cross_sectional_rank_mandatory_part_b() -> None:
    """Item 7 Part B: Lens must pass method='cross_sectional_rank' explicitly.

    2 sessions. Session 0 is "quiet": close=1.0 and volume=10_000.0 EXACTLY, for
    every bar, every symbol -- turnover is therefore bit-identical across all 5
    symbols (both operands are exactly representable in float32, so the product
    is exact too, with no accumulated rounding to break the tie). Session 1 is
    the "signal" session: close is differentiated per symbol using the same
    graduated per-symbol signal-mean construction as test_lens.py's
    `_signal_means`/`_build_panel` (`[-effect, -effect/2, 0, +effect/2, +effect]`
    plus small noise), so return_1 carries a real, non-degenerate cross-sectional
    edge there.

    Session 1's OWN close/volume never affect its OWN decile assignment -- only
    session 0 (strictly prior) does -- so session 1's rows all land in the SAME
    liquidity decile (prior_adv tied at exactly 10_000.0 for all 5 symbols),
    giving `cross_sectional_rank`'s `min_names=5` inner feature-bucketing >= 5
    finite values per row and letting it compute a real, non-zero spread from
    the differentiated session-1 signal. (An earlier attempt using a flat
    constant volume atop test_lens.py's multi-session, always-signal
    `_build_panel` failed here: turnover = close * volume is NOT exactly tied
    when close itself varies bar-to-bar across all sessions, and float32
    round-trip noise of order 1e-5 broke `rankdata`'s exact-equality tie
    detection, splitting the 5 symbols into 5 different singleton deciles and
    producing all-zero spreads for the wrong reason.)
    """
    n_symbols = 5
    bars_per_session = 60
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    ts, day_offsets, dates_arr = tl._session_grid(dates, bars_per_session)

    rng = np.random.default_rng(11)
    n_rows = 2 * bars_per_session
    close = np.ones((n_rows, n_symbols), dtype=np.float64)
    volume = np.full((n_rows, n_symbols), 10_000.0, dtype=np.float64)

    means = tl._signal_means(0.02)
    noise = rng.normal(0.0, 0.0005, size=(bars_per_session, n_symbols))
    rets = np.tile(means, (bars_per_session, 1)) + noise
    cum = np.cumsum(rets, axis=0)
    close[bars_per_session:, :] = np.exp(cum)

    symbols = tuple(f"S0{i}" for i in range(n_symbols))
    panel = Panel(
        fields={
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel, seed=11)
    feature_obj = lens.feature("return_1")
    stab_report = lens.stability(feature_obj, horizon=1)

    has_nonzero_spread = False
    for table in stab_report.by_liquidity_decile.values():
        if np.isfinite(table.spread_bps) and table.spread_bps != 0.0:
            has_nonzero_spread = True
            break

    assert has_nonzero_spread, (
        "At least one ExpectancyTable must have a finite, non-zero spread_bps; "
        "a suite that would pass against ten silent 0.0 spreads has tested nothing."
    )


def test_concentration_threshold_is_named_module_constant() -> None:
    """Item 8 Test 1: concentration threshold must be a named module-level constant."""
    threshold = _find_concentration_threshold()
    assert threshold is not None, (
        "criterion 4's concentration ratio threshold must be a named module-level "
        "constant in lens.py, not an inline literal `2`"
    )


def test_concentration_fires_above_threshold_not_below() -> None:
    """Item 8 Test 2: concentration check fires above threshold, not below.

    Uses the runtime-discovered threshold constant. Three scenarios:
    (a) fires unconditionally (bottom is argmax, ratio > threshold)
    (b) does not fire (bottom is argmax, ratio < threshold)
    (c) does not fire (bottom is NOT argmax)
    """
    threshold = _find_concentration_threshold()
    if threshold is None:
        pytest.skip(
            "module constant not yet defined; see "
            "test_concentration_threshold_is_named_module_constant"
        )

    if not (1.0 <= threshold <= 5.0):
        pytest.skip(
            "threshold outside the assumed calibration range 1.0-5.0; "
            "fixture below is only proven algebraically correct in that range -- "
            "widen the fixture design if recalibration lands outside it"
        )

    if not (1.3 < threshold <= 5.0):
        pytest.skip(
            "threshold must be > 1.3 for scenario (b) argmax property to hold"
        )

    panel = tl._build_panel((2024,), sessions_per_year=4, bars_per_session=10)
    lens = Lens(panel, seed=0)

    # Scenario (a): FIRES
    by_liquidity_decile_a = {
        0: _make_expectancy_table(-(threshold + 1.0) * 100.0),
        1: _make_expectancy_table(100.0),
        2: _make_expectancy_table(90.0),
    }
    by_time_of_day_a = {
        "open": _make_expectancy_table(5.0),
        "close": _make_expectancy_table(5.5),
    }
    stab_report_a = StabilityReport(
        by_year={},
        by_time_of_day=by_time_of_day_a,
        by_liquidity_decile=by_liquidity_decile_a,
        n_years_total=1,
        n_years_sign_consistent=1,
        dominant_sign="+",
    )

    with patch.object(Lens, "stability", return_value=stab_report_a):
        verdict_a = lens.verdict(
            "H001_close_reversion",
            "return_1",
            1,
            latency_profile=tl._PASS_LATENCY,
            method="cross_sectional_rank",
            n_buckets=5,
            n_boot=50,
        )

    assert "FAIL" in verdict_a.reasons[3], (
        f"Scenario (a) should FAIL, got: {verdict_a.reasons[3]}"
    )
    assert "concentrated in bottom liquidity decile" in verdict_a.reasons[3], (
        f"Scenario (a) should mention concentration, got: {verdict_a.reasons[3]}"
    )

    # Scenario (b): DOES NOT FIRE (ratio below threshold, bottom still argmax)
    by_liquidity_decile_b = {
        0: _make_expectancy_table(-(threshold - 0.3) * 100.0),
        1: _make_expectancy_table(100.0),
        2: _make_expectancy_table(99.0),
    }
    by_time_of_day_b = {
        "open": _make_expectancy_table(5.0),
        "close": _make_expectancy_table(5.5),
    }
    stab_report_b = StabilityReport(
        by_year={},
        by_time_of_day=by_time_of_day_b,
        by_liquidity_decile=by_liquidity_decile_b,
        n_years_total=1,
        n_years_sign_consistent=1,
        dominant_sign="+",
    )

    with patch.object(Lens, "stability", return_value=stab_report_b):
        verdict_b = lens.verdict(
            "H001_close_reversion",
            "return_1",
            1,
            latency_profile=tl._PASS_LATENCY,
            method="cross_sectional_rank",
            n_buckets=5,
            n_boot=50,
        )

    assert "PASS" in verdict_b.reasons[3], (
        f"Scenario (b) should PASS, got: {verdict_b.reasons[3]}"
    )

    # Scenario (c): DOES NOT FIRE (bottom decile NOT the argmax)
    by_liquidity_decile_c = {
        0: _make_expectancy_table(-5.0),
        1: _make_expectancy_table((threshold * 1000.0) * 10.0),
        2: _make_expectancy_table(9.0),
    }
    by_time_of_day_c = {
        "open": _make_expectancy_table(5.0),
        "close": _make_expectancy_table(5.5),
    }
    stab_report_c = StabilityReport(
        by_year={},
        by_time_of_day=by_time_of_day_c,
        by_liquidity_decile=by_liquidity_decile_c,
        n_years_total=1,
        n_years_sign_consistent=1,
        dominant_sign="+",
    )

    with patch.object(Lens, "stability", return_value=stab_report_c):
        verdict_c = lens.verdict(
            "H001_close_reversion",
            "return_1",
            1,
            latency_profile=tl._PASS_LATENCY,
            method="cross_sectional_rank",
            n_buckets=5,
            n_boot=50,
        )

    assert "PASS" in verdict_c.reasons[3], (
        f"Scenario (c) should PASS, got: {verdict_c.reasons[3]}"
    )


def test_all_seven_criteria_reported_in_order() -> None:
    """Item 9: all seven criteria still reported, in order."""
    panel = tl._build_panel(
        tl._ALL_YEARS,
        sessions_per_year=2,
        bars_per_session=50,
        effect=0.01,
        sigma=0.0005,
        seed=0,
    )
    lens = Lens(panel, seed=0)
    verdict = lens.verdict(
        "H001_close_reversion",
        "return_1",
        1,
        latency_profile=tl._PASS_LATENCY,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=50,
    )

    assert len(verdict.reasons) == 7, (
        f"Expected 7 reasons, got {len(verdict.reasons)}"
    )
    assert [r.split(".")[0] for r in verdict.reasons] == ["1", "2", "3", "4", "5", "6", "7"], (
        f"Reasons must be numbered 1-7 in order, got: {[r.split('.')[0] for r in verdict.reasons]}"
    )


def test_determinism() -> None:
    """Item 10: determinism — identical inputs produce identical outputs."""
    panel = tl._build_panel(
        tl._ALL_YEARS,
        sessions_per_year=2,
        bars_per_session=50,
        effect=0.01,
        sigma=0.0005,
        seed=0,
    )

    lens_a = Lens(panel, seed=0)
    verdict_a = lens_a.verdict(
        "H001_close_reversion",
        "return_1",
        1,
        latency_profile=tl._PASS_LATENCY,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=50,
    )

    lens_b = Lens(panel, seed=0)
    verdict_b = lens_b.verdict(
        "H001_close_reversion",
        "return_1",
        1,
        latency_profile=tl._PASS_LATENCY,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=50,
    )

    assert verdict_a.reasons == verdict_b.reasons, (
        "Reasons must be deterministic across identical calls"
    )
    assert verdict_a.survived == verdict_b.survived, (
        "survived flag must be deterministic across identical calls"
    )
