from __future__ import annotations

import datetime as dt
from typing import Callable
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.research import expectancy
from nifty_quant.research import lens as lens_module
from nifty_quant.research.lens import Lens

_IST = ZoneInfo("Asia/Kolkata")


def _session_ts(
    date: dt.date, n_bars: int, start_time: dt.time = dt.time(9, 15)
) -> np.ndarray:
    start = dt.datetime.combine(date, start_time, tzinfo=_IST)
    return np.array(
        [int(start.timestamp()) + b * 60 for b in range(n_bars)], dtype=np.int64
    )


def _session_grid(
    dates: list[dt.date],
    bars_per_session: list[int],
    start_time: dt.time = dt.time(9, 15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sessions = len(dates)
    day_offsets = np.zeros(n_sessions + 1, dtype=np.int32)
    ts_chunks: list[np.ndarray] = []
    row = 0
    for d, (date, n_bars) in enumerate(zip(dates, bars_per_session)):
        day_offsets[d] = row
        ts_chunks.append(_session_ts(date, n_bars, start_time=start_time))
        row += n_bars
    day_offsets[n_sessions] = row
    ts = np.concatenate(ts_chunks)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _dates_for(n_sessions: int, year: int = 2024) -> list[dt.date]:
    return [dt.date(year, 1, 2 + i) for i in range(n_sessions)]


def _generate_arrays(
    n_sessions: int,
    bars_per_session: int | list[int],
    symbols: tuple[str, ...],
    close_fn: Callable[[int, int], float],
    volume_fn: Callable[[int, int], float],
    return_mean_fn: Callable[[int, int], float] | None,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pure-numpy close/volume construction (no Panel yet, so callers may inject
    NaN bars before building the Panel).

    volume is held CONSTANT across every bar of a session (volume_fn(s, j)), so
    close*volume summed over a session is exactly session_bars * mean(close) *
    volume -- easy to reference-check against expectancy's own nansum formula.

    close is NEVER perfectly flat within a session: when return_mean_fn is None,
    close ramps by +0.01 per bar offset so the return_1 feature is not ITSELF
    session-constant (a flat return_1 would be indistinguishable, to the
    session-constant detector below, from the liquidity/ADV array under test).
    When return_mean_fn is given, close follows a persistent per-symbol mean
    log-return (same trick as tests/test_lens.py's _signal_means/_build_panel)
    plus N(0, sigma) noise, reset to close_fn's starting level at the start of
    every session (so turnover stays close to volume_fn(s, j) * bars_per_session
    * close_fn(s, j), not compounding across sessions).
    """
    bars_list = (
        [bars_per_session] * n_sessions
        if isinstance(bars_per_session, int)
        else list(bars_per_session)
    )
    n_symbols = len(symbols)
    n_rows = sum(bars_list)
    rng = np.random.default_rng(seed)

    close = np.empty((n_rows, n_symbols), dtype=np.float64)
    volume = np.empty((n_rows, n_symbols), dtype=np.float64)

    row = 0
    for s, n_bars in enumerate(bars_list):
        for j in range(n_symbols):
            vol = volume_fn(s, j)
            volume[row : row + n_bars, j] = vol
            start_close = close_fn(s, j)
            if return_mean_fn is None:
                close[row : row + n_bars, j] = start_close + 0.01 * np.arange(n_bars)
            else:
                mean = return_mean_fn(s, j)
                noise = rng.normal(0.0, sigma, size=n_bars)
                log_close = np.log(start_close) + np.cumsum(mean + noise)
                close[row : row + n_bars, j] = np.exp(log_close)
        row += n_bars

    return close, volume, bars_list


def _panel_from_arrays(
    close: np.ndarray,
    volume: np.ndarray,
    symbols: tuple[str, ...],
    bars_per_session: list[int],
    year: int = 2024,
    start_time: dt.time = dt.time(9, 15),
) -> Panel:
    dates = _dates_for(len(bars_per_session), year=year)
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session, start_time=start_time)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _reference_prior_adv_sessions(
    close: np.ndarray, volume: np.ndarray, day_offsets: np.ndarray
) -> np.ndarray:
    """Independently reimplements data/validate.py:727-739's ADV convention:
    per-session nansum(close*volume), then a trailing 20-session, strictly
    prior nanmean. Session 0 is NaN. Returns (n_sessions, n_symbols)."""
    n_sessions = len(day_offsets) - 1
    n_symbols = close.shape[1]
    day_value = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for s in range(n_sessions):
        start, end = int(day_offsets[s]), int(day_offsets[s + 1])
        with np.errstate(invalid="ignore"):
            day_value[s, :] = np.nansum(close[start:end] * volume[start:end], axis=0)
    adv = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for s in range(1, n_sessions):
        lookback_start = max(0, s - 20)
        with np.errstate(invalid="ignore"):
            adv[s, :] = np.nanmean(day_value[lookback_start:s, :], axis=0)
    return adv


def _broadcast_to_rows(
    session_values: np.ndarray, day_offsets: np.ndarray, n_rows: int
) -> np.ndarray:
    n_symbols = session_values.shape[1]
    out = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    for s in range(session_values.shape[0]):
        start, end = int(day_offsets[s]), int(day_offsets[s + 1])
        out[start:end, :] = session_values[s, :]
    return out


def _is_session_constant(arr: np.ndarray, day_offsets: np.ndarray) -> bool:
    """True if, in every session, every symbol column's finite values are all
    equal (a session-level statistic broadcast over bars looks like this; a
    per-bar-varying feature does not)."""
    for s in range(len(day_offsets) - 1):
        start, end = int(day_offsets[s]), int(day_offsets[s + 1])
        block = arr[start:end, :]
        for j in range(block.shape[1]):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            if not np.allclose(finite, finite[0]):
                return False
    return True


def _find_session_constant_call(
    call_args_list: list, day_offsets: np.ndarray
) -> np.ndarray | None:
    """Among every recorded expectancy.causal_buckets(...) call, find the one
    whose bucketed array is session-constant per _is_session_constant -- this
    identifies the ADV/liquidity call regardless of whether the implementer
    delegated to expectancy.expectancy_by_liquidity_decile or kept the decile
    loop in lens.py and called causal_buckets directly (spec: both are
    acceptable); both paths call expectancy.causal_buckets(prior_adv, ...,
    method="cross_sectional_rank") with a row-broadcast prior_adv array whose
    shape is (n_rows, n_symbols) with n_rows == day_offsets[-1]."""
    n_rows = int(day_offsets[-1])
    for call in call_args_list:
        arr = call.kwargs.get("feature")
        if arr is None and call.args:
            arr = call.args[0]
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] != n_rows:
            continue
        if _is_session_constant(arr, day_offsets):
            return arr
    return None


def _capture_causal_buckets(fn: Callable[[], object]) -> tuple[object, list]:
    with patch.object(
        expectancy, "causal_buckets", wraps=expectancy.causal_buckets
    ) as spy:
        result = fn()
    return result, list(spy.call_args_list)


_N_PER_DECILE = 5
_N_DECILES = 10
_TURNOVER_BASE_EXP = 2  # decile k -> turnover level 10**(k + _TURNOVER_BASE_EXP)


def _decile_symbols() -> tuple[str, ...]:
    return tuple(f"D{k}_{j}" for k in range(_N_DECILES) for j in range(_N_PER_DECILE))


def _decile_ladder_volume_fn(k: int) -> float:
    return 10.0 ** (k + _TURNOVER_BASE_EXP)


def _build_decile_ladder(
    n_sessions: int,
    bars_per_session: int,
    effect_by_decile: dict[int, float],
    default_effect: float,
    sigma: float,
    seed: int,
    extra_symbols: list[tuple[str, float, float, float]] | None = None,
    start_time: dt.time = dt.time(9, 15),
) -> tuple[Panel, tuple[str, ...]]:
    """50 base symbols in 10 groups of 5 (D{k}_{j}, k=liquidity decile 0..9,
    j=0..4 quintile slot within the group). Every base symbol has CONSTANT
    close=10.0 and CONSTANT volume=10**(k+2) across all sessions, so rupee
    turnover is exponentially separated by decile (10x per step) and stable --
    robust against the small price drift `return_mean_fn` noise introduces.
    Each base symbol's persistent return_1 mean is
    effect_by_decile.get(k, default_effect) * (j - 2) / 2, i.e. -e, -e/2, 0,
    +e/2, +e across j=0..4 (same "persistent offset" trick as
    tests/test_lens.py's _signal_means/_build_panel) -- this makes symbols with
    j=4 rank at the top quintile within their decile's cross-sectional feature
    ranking, and j=0 at the bottom, reliably.

    extra_symbols: list of (name, close_level, volume_level, effect) appended
    after the 50 base symbols, each held at a CONSTANT close/volume (hence
    constant turnover) across every session, with a single persistent return_1
    mean offset (no quintile spread -- there is only one of each)."""
    base_symbols = _decile_symbols()
    extra = extra_symbols or []
    extra_names = tuple(name for name, *_ in extra)
    symbols = base_symbols + extra_names
    n_base = len(base_symbols)

    def close_fn(s: int, sym_idx: int) -> float:
        if sym_idx < n_base:
            return 10.0
        return extra[sym_idx - n_base][1]

    def volume_fn(s: int, sym_idx: int) -> float:
        if sym_idx < n_base:
            k = sym_idx // _N_PER_DECILE
            return _decile_ladder_volume_fn(k)
        return extra[sym_idx - n_base][2]

    def return_mean_fn(s: int, sym_idx: int) -> float:
        if sym_idx < n_base:
            k, j = divmod(sym_idx, _N_PER_DECILE)
            effect = effect_by_decile.get(k, default_effect)
            return effect * (j - 2) / 2.0
        return extra[sym_idx - n_base][3]

    close, volume, bars_list = _generate_arrays(
        n_sessions, bars_per_session, symbols, close_fn, volume_fn,
        return_mean_fn, sigma, seed,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list, start_time=start_time)
    return panel, symbols


def _build_causality_flip_panel(
    n_sessions: int,
    bars_per_session: int,
    transition_session: int,
    flip_effect: float,
    flip_effect_sessions: set[int],
    default_effect: float,
    sigma: float,
    seed: int,
) -> tuple[Panel, tuple[str, ...]]:
    """Builds a 51-symbol panel: 50 base decile-ladder symbols plus one extra
    symbol S_FLIP whose volume jumps from tiny (1.0) to enormous (1e12) at
    transition_session. S_FLIP's return effect is flip_effect only in
    flip_effect_sessions, else 0.0."""
    base_symbols = _decile_symbols()
    symbols = base_symbols + ("S_FLIP",)
    n_base = len(base_symbols)

    def close_fn(s: int, sym_idx: int) -> float:
        return 10.0

    def volume_fn(s: int, sym_idx: int) -> float:
        if sym_idx < n_base:
            k = sym_idx // _N_PER_DECILE
            return _decile_ladder_volume_fn(k)
        return 1.0 if s < transition_session else 1e12

    def return_mean_fn(s: int, sym_idx: int) -> float:
        if sym_idx < n_base:
            k, j = divmod(sym_idx, _N_PER_DECILE)
            return default_effect * (j - 2) / 2.0
        return flip_effect if s in flip_effect_sessions else 0.0

    close, volume, bars_list = _generate_arrays(
        n_sessions, bars_per_session, symbols, close_fn, volume_fn,
        return_mean_fn, sigma, seed,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    return panel, symbols


def test_units_regression_guard() -> None:
    """Spec item 1: rupee turnover (close*volume) bucketing, not raw share count.

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (share-count, full-sample bucketing).
    """
    panel, _ = _build_decile_ladder(
        n_sessions=15,
        bars_per_session=6,
        effect_by_decile={},
        default_effect=0.001,
        sigma=0.0008,
        seed=1,
        extra_symbols=[("S_HIGH", 10000.0, 100.0, 0.5), ("S_LOW", 1e-5, 1e11, 0.5)],
    )
    lens = Lens(panel, seed=1)
    feature = lens.feature("return_1")
    stab = lens.stability(feature, horizon=1)

    # S_HIGH and S_LOW both have rupee turnover 1e6, matching decile 4's level.
    # Under correct turnover bucketing they join decile 4 and boost its spread.
    # Under share-count bucketing they land in deciles 0 and 9 respectively.
    # The 0.5 effect vs 0.001 baseline gives ~1670bps spread in the boosted decile.
    # 500 is safely below the true effect and safely above the ~20bps baseline.
    assert np.isfinite(stab.by_liquidity_decile[4].spread_bps)
    assert abs(stab.by_liquidity_decile[4].spread_bps) > 500.0

    # Deciles 0 and 9 should NOT get the boost under correct bucketing.
    # 200 is a loose bound distinguishing baseline noise from the 0.5-effect symbol.
    for decile in (0, 9):
        if decile in stab.by_liquidity_decile:
            assert abs(stab.by_liquidity_decile[decile].spread_bps) < 200.0


def test_causality_guard() -> None:
    """Spec item 2: prior_adv must be strictly prior (no lookahead).

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (full-sample np.quantile lookahead).
    """
    panel, _ = _build_causality_flip_panel(
        n_sessions=22,
        bars_per_session=6,
        transition_session=20,
        flip_effect=3.0,
        flip_effect_sessions={20},
        default_effect=0.001,
        sigma=0.0008,
        seed=2,
    )
    lens = Lens(panel, seed=2)
    feature = lens.feature("return_1")
    stab = lens.stability(feature, horizon=1)

    # At session 20, trailing 20-session prior-ADV for S_FLIP is all pre-transition
    # (tiny), so causal bucketing puts it in decile 0. Full-sample bucketing would
    # put it in decile 9. The 3.0 effect concentrated in session 20 gives a large
    # spread in the correct decile. 150 is safely above baseline and below the
    # ~5000bps-scale effect.
    assert np.isfinite(stab.by_liquidity_decile[0].spread_bps)
    assert abs(stab.by_liquidity_decile[0].spread_bps) > 150.0

    if 9 in stab.by_liquidity_decile:
        assert abs(stab.by_liquidity_decile[9].spread_bps) < 200.0


def test_session_0_excluded() -> None:
    """Spec item 3: session 0 has no prior turnover -> NaN, not decile 0.

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (no causal_buckets call for liquidity decomposition at all).
    """
    symbols = tuple(f"S{j}" for j in range(6))
    close, volume, bars_list = _generate_arrays(
        n_sessions=5,
        bars_per_session=4,
        symbols=symbols,
        close_fn=lambda s, j: 10.0 + j,
        volume_fn=lambda s, j: 1000.0 * (s + 1) + 10.0 * j,
        return_mean_fn=None,
        sigma=0.0,
        seed=3,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    lens = Lens(panel, seed=3)
    feature = lens.feature("return_1")

    _, call_args_list = _capture_causal_buckets(
        lambda: lens.stability(feature, horizon=1)
    )
    captured = _find_session_constant_call(call_args_list, panel.day_offsets)

    # Current code never calls expectancy.causal_buckets for liquidity decomposition,
    # so finding no session-constant call is the expected RED signal.
    assert captured is not None

    # Session 0 (rows 0:4) must be all-NaN: no prior data.
    assert np.all(np.isnan(captured[0:4, :]))
    # Session 1 (rows 4:8) must have real values.
    assert not np.all(np.isnan(captured[4:8, :]))


def test_prior_adv_excludes_current_session() -> None:
    """Spec item 4: prior_adv excludes session s itself (hand-checkable).

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (no causal_buckets call for liquidity decomposition).
    """
    symbols = ("A", "B")
    close, volume, bars_list = _generate_arrays(
        n_sessions=5,
        bars_per_session=3,
        symbols=symbols,
        close_fn=lambda s, j: 10.0 + j,
        volume_fn=lambda s, j: 500.0 if (s == 2 and j == 0) else 5.0,
        return_mean_fn=None,
        sigma=0.0,
        seed=4,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    lens = Lens(panel, seed=4)
    feature = lens.feature("return_1")

    _, call_args_list = _capture_causal_buckets(
        lambda: lens.stability(feature, horizon=1)
    )
    captured = _find_session_constant_call(call_args_list, panel.day_offsets)

    expected_sessions = _reference_prior_adv_sessions(
        close, volume, panel.day_offsets
    )
    expected = _broadcast_to_rows(
        expected_sessions, panel.day_offsets, close.shape[0]
    )

    assert captured is not None
    # This equality check catches an off-by-one that would include session s's own
    # spike in its own prior_adv.
    assert np.allclose(captured, expected, equal_nan=True)


def test_irregular_sessions_no_fixed_stride() -> None:
    """Spec item 5: irregular sessions (Muhurat), no fixed stride assumed.

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (no causal_buckets call for liquidity decomposition).
    """
    symbols = tuple(f"S{j}" for j in range(6))
    bars_per_session = [12, 12, 12, 12, 6, 12, 12, 12, 12]
    close, volume, bars_list = _generate_arrays(
        n_sessions=9,
        bars_per_session=bars_per_session,
        symbols=symbols,
        close_fn=lambda s, j: 20.0 + j,
        volume_fn=lambda s, j: 300.0 * (j + 1),
        return_mean_fn=None,
        sigma=0.0,
        seed=5,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    lens = Lens(panel, seed=5)
    feature = lens.feature("return_1")

    _, call_args_list = _capture_causal_buckets(
        lambda: lens.stability(feature, horizon=1)
    )
    captured = _find_session_constant_call(call_args_list, panel.day_offsets)

    expected_sessions = _reference_prior_adv_sessions(
        close, volume, panel.day_offsets
    )
    expected = _broadcast_to_rows(
        expected_sessions, panel.day_offsets, close.shape[0]
    )

    assert captured is not None
    # This check fails if the implementation assumed any fixed rows-per-session
    # stride, because the Muhurat session's shorter length would misalign the
    # broadcast boundaries.
    assert np.allclose(captured, expected, equal_nan=True)


def test_nan_turnover_propagates() -> None:
    """Spec item 6: NaN turnover (missing bar) propagates, never forward-filled
    or zero-filled.

    EXPECTED TO FAIL against current src/nifty_quant/research/lens.py
    (no causal_buckets call for liquidity decomposition).
    """
    symbols = ("A", "B")
    close, volume, bars_list = _generate_arrays(
        n_sessions=3,
        bars_per_session=4,
        symbols=symbols,
        close_fn=lambda s, j: 10.0 + j,
        volume_fn=lambda s, j: 50.0,
        return_mean_fn=None,
        sigma=0.0,
        seed=6,
    )
    # Mutate one interior bar: session 1 (index 1), 2nd bar within it (offset 1),
    # symbol 0.
    row = int(4 + 1)  # session 1 starts at row 4, offset 1
    close[row, 0] = np.nan
    volume[row, 0] = np.nan

    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    lens = Lens(panel, seed=6)
    feature = lens.feature("return_1")

    _, call_args_list = _capture_causal_buckets(
        lambda: lens.stability(feature, horizon=1)
    )
    captured = _find_session_constant_call(call_args_list, panel.day_offsets)

    expected_sessions = _reference_prior_adv_sessions(
        close, volume, panel.day_offsets
    )
    expected = _broadcast_to_rows(
        expected_sessions, panel.day_offsets, close.shape[0]
    )

    assert captured is not None
    # If the implementation forward-filled the missing bar's price, day_value for
    # session 1 would be inflated relative to this nansum-based reference.
    assert np.allclose(captured, expected, equal_nan=True)


def test_cross_sectional_rank_used() -> None:
    """Spec item 7: cross_sectional_rank actually used, non-zero finite spreads.

    NOT required to be RED against current src/nifty_quant/research/lens.py --
    the current per-decile FEATURE bucketing already passes
    method="cross_sectional_rank" explicitly (only the OUTER decile assignment
    is broken there). This test is still required by the spec: it is the
    regression guard against reverting to the expanding_quantile default
    (which would silently zero out every decile's spread) once the fix lands.
    """
    panel, _ = _build_decile_ladder(
        n_sessions=15,
        bars_per_session=6,
        effect_by_decile={},
        default_effect=0.01,
        sigma=0.0008,
        seed=7,
    )
    lens = Lens(panel, seed=7)
    feature = lens.feature("return_1")
    stab = lens.stability(feature, horizon=1)

    assert len(stab.by_liquidity_decile) >= 1
    # With method defaulting to expanding_quantile, a once-per-session ADV signal
    # can never accumulate min_history ROWS, so every decile would silently read
    # spread_bps == 0.0. cross_sectional_rank must actually be used.
    assert any(
        np.isfinite(table.spread_bps) and table.spread_bps != 0.0
        for table in stab.by_liquidity_decile.values()
    )


@pytest.mark.parametrize(
    "effect_by_decile,default_effect,seed,expected_fail",
    [
        ({0: 0.05}, 0.002, 81, True),
        ({9: 0.05}, 0.002, 82, False),
        ({}, 0.01, 83, False),
    ],
)
def test_criterion_4_firing_logic(
    effect_by_decile: dict[int, float],
    default_effect: float,
    seed: int,
    expected_fail: bool,
) -> None:
    """Spec item 8: criterion 4 firing logic, parameterized on module constant.

    NOT required to be RED against current src/nifty_quant/research/lens.py --
    this fixture's base decile ladder has a CONSTANT close price across every
    symbol (only volume varies), so the units defect is invisible to it, and
    turnover does not vary over time, so the causality defect is invisible to
    it too; `getattr(..., 2.0)` also reproduces the current inline literal `2`
    numerically. The CONCENTRATION_THRESHOLD constant does not exist yet in
    current code, but that has no observable effect here. This test is still
    required by the spec: it pins down the PASS/FAIL decision logic itself
    (argmax + ratio-vs-threshold), independent of items 1/2's units/causality
    regressions.
    """
    threshold = float(getattr(lens_module, "CONCENTRATION_THRESHOLD", 2.0))
    assert threshold > 0.0
    # A session starting at 10:27 with 6 one-minute bars covers minutes
    # 627,628,629 (still "open", < 630) and 630,631,632 ("mid", >= 630) -- a
    # balanced 3/3 split across TWO time-of-day buckets, using the SAME small
    # bars_per_session as every other decile-ladder fixture (large
    # bars_per_session would instead make the per-bar persistent return_1
    # effect COMPOUND across many bars into a huge session-long price drift,
    # corrupting the close*volume turnover this fixture depends on). This is
    # needed so criterion 4's SEPARATE time-of-day concentration check (which
    # independently FAILs whenever only one time-of-day bucket has any data at
    # all) does not confound the liquidity-decile check this test targets.
    panel, _ = _build_decile_ladder(
        n_sessions=15,
        bars_per_session=6,
        effect_by_decile=effect_by_decile,
        default_effect=default_effect,
        sigma=0.0008,
        seed=seed,
        start_time=dt.time(10, 27),
    )
    lens = Lens(panel, seed=seed)
    feature = lens.feature("return_1")
    verdict = lens.verdict(
        "H_TEST",
        feature,
        horizon=1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=1,
    )
    criterion_line = next(r for r in verdict.reasons if r.startswith("4."))

    # This is a MAGNITUDE/qualitative check (big margin vs threshold), not an
    # attempt to hit threshold exactly. The exact numeric threshold is still
    # pending calibration per the spec, so these fixtures are deliberately far
    # from the boundary in both directions.
    if expected_fail:
        assert "FAIL" in criterion_line
        assert "concentrated in bottom liquidity decile" in criterion_line
        # Genuine parameterization, not just a qualitative label check: the
        # boosted decile's spread must exceed `threshold` times a representative
        # "other decile" spread (decile 5, an unboosted decile picked as a
        # stand-in -- NOT a replica of lens.py's own upper-median convention,
        # which this test deliberately does not reimplement). The fixture's
        # ~28x true ratio (see debug measurement) is far above any plausible
        # calibrated threshold, so this stays valid once CONCENTRATION_THRESHOLD
        # is substituted with its measured value.
        decile0 = abs(verdict.stability.by_liquidity_decile[0].spread_bps)
        decile5 = abs(verdict.stability.by_liquidity_decile[5].spread_bps)
        assert decile0 > threshold * decile5
    else:
        assert "PASS" in criterion_line


def test_all_seven_criteria_reported_in_order() -> None:
    """Spec item 9: all seven criteria still reported, in order.

    NOT required to be RED against current src/nifty_quant/research/lens.py --
    reporting all 7 criteria in order is unrelated to the units/causality/
    threshold-naming defects this spec repairs, and current code already does
    it. This is a regression guard: the repair must not accidentally drop or
    reorder a criterion.
    """
    symbols = tuple(f"S{j}" for j in range(6))
    close, volume, bars_list = _generate_arrays(
        n_sessions=6,
        bars_per_session=6,
        symbols=symbols,
        close_fn=lambda s, j: 10.0 + j + 0.05 * s,
        volume_fn=lambda s, j: 100.0 * (j + 1),
        return_mean_fn=lambda s, j: 0.002 * (j - 2.5),
        sigma=0.001,
        seed=8,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)
    lens = Lens(panel, seed=8)
    feature = lens.feature("return_1")
    verdict = lens.verdict(
        "H_TEST",
        feature,
        horizon=1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=1,
    )

    assert len(verdict.reasons) == 7
    for i in range(1, 8):
        assert verdict.reasons[i - 1].startswith(f"{i}.")


def test_determinism() -> None:
    """Spec item 10: determinism across independent Lens constructions.

    NOT required to be RED against current src/nifty_quant/research/lens.py --
    determinism is unrelated to the units/causality/threshold-naming defects
    this spec repairs, and current code is already deterministic. This is a
    regression guard: the repair (in particular the new prior_adv computation
    and causal_buckets delegation) must not introduce nondeterminism.
    """
    symbols = tuple(f"S{j}" for j in range(6))
    close, volume, bars_list = _generate_arrays(
        n_sessions=6,
        bars_per_session=6,
        symbols=symbols,
        close_fn=lambda s, j: 10.0 + j + 0.05 * s,
        volume_fn=lambda s, j: 100.0 * (j + 1),
        return_mean_fn=lambda s, j: 0.002 * (j - 2.5),
        sigma=0.001,
        seed=9,
    )
    panel = _panel_from_arrays(close, volume, symbols, bars_list)

    lens_a = Lens(panel, seed=9)
    lens_b = Lens(panel, seed=9)
    feature_a = lens_a.feature("return_1")
    feature_b = lens_b.feature("return_1")

    verdict_a = lens_a.verdict(
        "H_TEST",
        feature_a,
        horizon=1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=1,
    )
    verdict_b = lens_b.verdict(
        "H_TEST",
        feature_b,
        horizon=1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=1,
    )

    assert verdict_a.reasons == verdict_b.reasons
    assert verdict_a.survived == verdict_b.survived
    assert verdict_a.explain() == verdict_b.explain()
