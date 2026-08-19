"""Suite B targets the pending kill-criterion-4 repair in ``Lens.stability``.
It is based solely on ``specs/lens_criterion_4_repair.md``.
This suite ASSUMES the fix adds two currently absent names to
``nifty_quant.research.lens``: ``compute_prior_adv(panel: Panel) -> np.ndarray``,
which computes trailing-20-session strictly-prior rupee-turnover ADV,
broadcast at bar level with session 0 as NaN, and the float constant
``CONCENTRATION_RATIO_THRESHOLD``, replacing the hardcoded criterion-4 ratio.
Every test needing either assumed name fetches it through ``_require_attr``.
Thus an API mismatch produces an ``AssertionError`` rather than collection aborting.
``cross_sectional_rank`` returns all-NaN for rows with fewer than five finite values.
The delegated fix applies that behavior twice: for prior-ADV deciles and
for the feature within each decile. Therefore spread values read from
``by_liquidity_decile`` require at least 50 symbols, or every decile is 0.0.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import nifty_quant.research.lens as lens_module
from nifty_quant.data.panel import Panel
from nifty_quant.research import expectancy
from nifty_quant.research.lens import Lens
from tests.test_lens import (
    _FAIL_LATENCY,
    _PASS_LATENCY,
    _all_pass_panel,
    _build_panel,
    _call_verdict,
)

_IST = ZoneInfo("Asia/Kolkata")


def _require_attr(obj: object, name: str, context: str) -> Any:
    """Fetch obj.name, converting a missing attribute into an AssertionError."""
    assert hasattr(obj, name), (
        f"{context}: expected nifty_quant.research.lens.{name} to exist "
        f"(see specs/lens_criterion_4_repair.md); not found on current lens.py"
    )
    return getattr(obj, name)


def _ist_session_grid(
    dates: list[dt.date], bars_per_session: int | list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an IST 09:15 one-minute session grid."""
    bars_list = (
        [bars_per_session] * len(dates)
        if isinstance(bars_per_session, int)
        else list(bars_per_session)
    )
    assert len(bars_list) == len(dates)
    ts_chunks: list[np.ndarray] = []
    day_offsets_list = [0]
    row = 0
    for date, n_bars in zip(dates, bars_list):
        day_start = pd.Timestamp(date.year, date.month, date.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n_bars, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
        row += n_bars
        day_offsets_list.append(row)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    dates_arr = np.array(dates, dtype=object)
    day_offsets = np.array(day_offsets_list, dtype=np.int32)
    return ts, day_offsets, dates_arr


def _panel_from_symbol_series(
    symbols: tuple[str, ...],
    symbol_returns: np.ndarray,  # (n_symbols,) float64, constant per-bar log return
    symbol_volume: np.ndarray,  # (n_sessions, n_symbols) float64, per-session volume
    bars_per_session: int | list[int],
    *,
    symbol_base_price: np.ndarray | float = 1.0,
    start_date: dt.date = dt.date(2024, 1, 2),
) -> Panel:
    """Build a synthetic Panel with an exactly hand-computable price path.

    Symbol k's close at global row t is
    ``symbol_base_price[k] * exp(t * symbol_returns[k])``.  The constant
    per-bar log return makes horizon-one log and forward returns exactly
    ``symbol_returns[k]`` at every valid non-boundary row.

    Session volume is constant across that session and may vary by session.
    Prices and volumes are stored as float32, matching production storage.
    """
    symbol_returns = np.asarray(symbol_returns, dtype=np.float64)
    symbol_volume = np.asarray(symbol_volume, dtype=np.float64)
    n_symbols = len(symbols)
    n_sessions = symbol_volume.shape[0]
    assert symbol_returns.shape == (n_symbols,)
    assert symbol_volume.shape == (n_sessions, n_symbols)

    if isinstance(symbol_base_price, (int, float)):
        base_price = np.full(n_symbols, float(symbol_base_price), dtype=np.float64)
    else:
        base_price = np.asarray(symbol_base_price, dtype=np.float64)
    assert base_price.shape == (n_symbols,)

    dates = [start_date + dt.timedelta(days=i) for i in range(n_sessions)]
    ts, day_offsets, dates_arr = _ist_session_grid(dates, bars_per_session)
    n_rows = len(ts)

    row_idx = np.arange(n_rows, dtype=np.float64)
    close = base_price[None, :] * np.exp(np.outer(row_idx, symbol_returns))

    volume = np.empty((n_rows, n_symbols), dtype=np.float64)
    for s in range(n_sessions):
        start, end = int(day_offsets[s]), int(day_offsets[s + 1])
        volume[start:end, :] = symbol_volume[s, :]

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


def _reference_liquidity_decile_tables(
    panel: Panel,
    feature_values: np.ndarray,
    day_offsets: np.ndarray,
    prior_adv: np.ndarray,
    *,
    horizon: int = 1,
    n_liquidity_buckets: int = 10,
    n_feature_buckets: int = 5,
) -> dict[int, expectancy.ExpectancyTable]:
    """Correct-by-construction ORACLE for the 10-liquidity x 5-feature geometry, built
    ONLY from already-trusted `nifty_quant.research.expectancy` primitives (never from
    `Lens`) -- this is exactly the "keep the decile loop, call causal_buckets +
    conditional_expectancy directly" option the spec allows as an alternative to extending
    `expectancy_by_liquidity_decile` (which cannot express 10x5, since it feeds one
    `n_buckets` to both the ADV and feature dimensions).
    """
    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, day_offsets, horizon)
    adv_bucketing = expectancy.causal_buckets(
        prior_adv, day_offsets, n_buckets=n_liquidity_buckets, method="cross_sectional_rank"
    )
    tables: dict[int, expectancy.ExpectancyTable] = {}
    for decile in range(n_liquidity_buckets):
        mask = adv_bucketing.labels == decile
        if not np.any(mask):
            continue
        feat_masked = np.where(mask, feature_values, np.nan)
        fwd_masked = np.where(mask, fwd.values, np.nan)
        n_defined = int(np.sum(np.any(np.isfinite(fwd_masked), axis=1)))
        fwd_obj = expectancy.ForwardReturns(
            values=fwd_masked,
            horizon=horizon,
            session_bounded=True,
            n_defined=n_defined,
            n_nan_tail=int(np.sum(~np.isfinite(fwd_masked))),
        )
        tables[decile] = expectancy.conditional_expectancy(
            feat_masked,
            fwd_obj,
            day_offsets,
            n_buckets=n_feature_buckets,
            method="cross_sectional_rank",
            feature_name="reference_feature",
        )
    return tables


def _reference_prior_adv(
    close: np.ndarray, volume: np.ndarray, day_offsets: np.ndarray
) -> np.ndarray:
    """Independent, hand-written reference implementation of the `prior_adv` formula
    (nifty_quant/data/validate.py:727-738's existing ADV convention, reused here as-is,
    NOT re-derived differently): per-session TOTAL rupee turnover via `nansum`, then a
    trailing 20-session STRICTLY PRIOR `nanmean`, session 0 NaN, bar-level broadcast to
    shape (n_rows, n_symbols). Used in test bodies as the independent "expected" oracle
    to compare against `compute_prior_adv` (the assumed real implementation, fetched via
    `_require_attr`) -- NOT the same code path, so the comparison is non-tautological.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)
    n_rows = close.shape[0]
    n_symbols = close.shape[1]
    n_sessions = len(day_offsets) - 1

    day_value = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    adv = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(n_sessions):
            start, end = int(day_offsets[s]), int(day_offsets[s + 1])
            day_value[s, :] = np.nansum(close[start:end] * volume[start:end], axis=0)

        for s in range(1, n_sessions):
            lookback_start = max(0, s - 20)
            adv[s, :] = np.nanmean(day_value[lookback_start:s, :], axis=0)

    out = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    for s in range(n_sessions):
        start, end = int(day_offsets[s]), int(day_offsets[s + 1])
        out[start:end, :] = adv[s, :]
    return out


def test_units_guard_equal_turnover_lands_in_same_decile() -> None:
    # B's price/volume were originally (1.0, 1_000_000.0): equal turnover to A, but
    # ALSO an accidentally-similar RAW SHARE VOLUME to the near-fillers (~1e6), so the
    # current buggy (share-volume) bucketing put B in the bottom decile too, by
    # coincidence -- discovered by running this fixture against unfixed src/, per the
    # task's validation step, not by re-deriving it on paper. B's price is now driven
    # far lower (0.001) so its RAW SHARE VOLUME (turnover/price = 1e9) sits far ABOVE
    # every general filler's share volume, landing B at the OPPOSITE end under
    # share-volume bucketing while its rupee turnover is still exactly equal to A's.
    symbols = tuple(f"S{i:02d}" for i in range(52))

    symbol_base_price = np.full(52, 10.0, dtype=np.float64)
    symbol_base_price[0] = 100_000.0
    symbol_base_price[1] = 0.001

    symbol_volume_by_symbol = np.empty(52, dtype=np.float64)
    symbol_volume_by_symbol[0] = 10.0
    symbol_volume_by_symbol[1] = 1_000_000_000.0
    symbol_volume_by_symbol[2:6] = np.array(
        [1_000_000.0 + 1000.0 * (i - 2) for i in range(2, 6)],
        dtype=np.float64,
    )
    symbol_volume_by_symbol[6:] = np.linspace(5_000_000.0, 100_000_000.0, 46)
    symbol_volume = np.tile(symbol_volume_by_symbol, (20, 1))

    symbol_returns = np.zeros(52, dtype=np.float64)
    symbol_returns[0] = -0.0008
    symbol_returns[1] = 0.0008

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        15,
        symbol_base_price=symbol_base_price,
    )

    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    prior_adv_correct = _reference_prior_adv(close, volume, panel.day_offsets)
    reference = _reference_liquidity_decile_tables(
        panel,
        feature.values,
        panel.day_offsets,
        prior_adv_correct,
        horizon=1,
    )
    actual = lens_obj.stability(feature, horizon=1).by_liquidity_decile

    assert 0 in reference
    assert np.isfinite(reference[0].spread_bps)
    assert abs(reference[0].spread_bps) > 5.0
    assert 0 in actual
    # This must FAIL against the current (unfixed) lens.py, which buckets by raw share
    # volume via a full-sample np.quantile instead of rupee turnover via strictly-prior
    # cross_sectional_rank—an entirely different partition of symbols into deciles.
    assert np.isclose(
        actual[0].spread_bps,
        reference[0].spread_bps,
        rtol=1e-6,
        atol=1e-6,
    )


def test_causality_guard_strictly_prior_wins_over_full_sample() -> None:
    # Design note (found by running this fixture, not by hand-derivation alone):
    # giving T a CONSTANT per-bar return for its whole life cannot discriminate causal
    # from non-causal bucketing on its own -- conditional_expectancy's bucket MEAN is
    # unaffected by *how many* of T's rows land in a bucket, only by T's (constant)
    # VALUE while present. The volume jump is also 20,000x (10 -> 1e6), so the trailing
    # 20-session average is swamped within ONE session of the jump entering the
    # window -- there is no multi-session "lag" to exploit here, only the ONE session
    # (session 10 itself) that `prior_adv[10]` provably excludes by construction
    # (spec item 4). So the signal is placed ONLY on session 10's bars: T is flat
    # (return 0.0) everywhere else. Verified empirically: under strictly-prior/causal
    # bucketing, T is still bottom-decile AT session 10 (prior_adv[10] reflects only
    # its 10 prior tiny-volume sessions), so decile 0 captures T's session-10 spike;
    # under the current buggy bucketing (raw share volume, reacts to the value in that
    # very row), T's session-10 volume is ALREADY the post-jump value, so it is
    # excluded from the bottom decile at that row and the spike is invisible there.
    symbols = tuple(f"T{i:02d}" for i in range(55))
    n_sessions = 30
    bars_per_session = 15
    jump_session = 10
    spike_return = 0.05

    symbol_base_price = np.full(55, 10.0, dtype=np.float64)
    symbol_base_price[0] = 100.0

    symbol_returns = np.zeros(55, dtype=np.float64)  # T's spike is injected below

    symbol_volume = np.empty((n_sessions, 55), dtype=np.float64)
    symbol_volume[:jump_session, 0] = 50.0  # tiny, below the anchors (500)
    symbol_volume[jump_session:, 0] = 1_000_000.0  # huge, post-jump
    symbol_volume[:, 1:6] = 500.0  # 5 anchors: always near the bottom decile
    symbol_volume[:, 6:] = np.linspace(50_000.0, 2_000_000.0, 49)

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        bars_per_session,
        symbol_base_price=symbol_base_price,
    )

    # Inject a constant spike_return log-return into T's (column 0) session-10 bars
    # only, by rebuilding that session's close sub-path from its own starting value.
    # T's other sessions, and every other symbol, are untouched (flat, return 0.0).
    close = panel.field("close").copy()
    start10 = int(panel.day_offsets[jump_session])
    end10 = int(panel.day_offsets[jump_session + 1])
    n_bars10 = end10 - start10
    base_val = float(close[start10, 0])
    close[start10:end10, 0] = base_val * np.exp(
        spike_return * np.arange(n_bars10, dtype=np.float64)
    ).astype(np.float32)
    panel = Panel(
        fields={"close": close, "volume": panel.field("volume")},
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )

    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    close64 = panel.field("close").astype(np.float64)
    volume64 = panel.field("volume").astype(np.float64)
    prior_adv_correct = _reference_prior_adv(close64, volume64, panel.day_offsets)
    reference = _reference_liquidity_decile_tables(
        panel,
        feature.values,
        panel.day_offsets,
        prior_adv_correct,
        horizon=1,
    )
    actual = lens_obj.stability(feature, horizon=1).by_liquidity_decile

    assert 0 in reference
    assert np.isfinite(reference[0].spread_bps)
    assert abs(reference[0].spread_bps) > 100.0
    assert 0 in actual
    # This must FAIL against the current (unfixed) lens.py: measured, it reports
    # by_liquidity_decile[0].spread_bps == 0.0 (T's session-10 spike is invisible to
    # it), while the reference/causal construction reports ~500 bps -- see the design
    # note above and the task's validation step.
    assert np.isclose(
        actual[0].spread_bps,
        reference[0].spread_bps,
        rtol=1e-6,
        atol=1e-6,
    )


def test_prior_adv_session_zero_is_nan() -> None:
    symbols = tuple(f"P{i}" for i in range(6))
    symbol_volume = np.array(
        [
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            [110.0, 210.0, 310.0, 410.0, 510.0, 610.0],
            [120.0, 220.0, 320.0, 420.0, 520.0, 620.0],
        ]
    )
    panel = _panel_from_symbol_series(
        symbols,
        np.zeros(6),
        symbol_volume,
        bars_per_session=4,
        symbol_base_price=10.0,
    )
    compute_prior_adv = _require_attr(
        lens_module,
        "compute_prior_adv",
        "prior_adv session-0 NaN (spec item 3)",
    )
    prior_adv = compute_prior_adv(panel)

    assert bool(
        np.all(
            np.isnan(
                prior_adv[
                    panel.day_offsets[0] : panel.day_offsets[1],
                    :,
                ]
            )
        )
    )
    assert bool(
        np.any(
            np.isfinite(
                prior_adv[
                    panel.day_offsets[1] : panel.day_offsets[2],
                    :,
                ]
            )
        )
    )


def test_prior_adv_excludes_own_session() -> None:
    symbols = ("Q0", "Q1")
    symbol_volume = np.array(
        [
            [100.0, 200.0],
            [300.0, 400.0],
            [10_000_000.0, 20_000_000.0],
        ]
    )
    panel = _panel_from_symbol_series(
        symbols,
        np.zeros(2),
        symbol_volume,
        bars_per_session=3,
        symbol_base_price=1.0,
    )
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    expected_prior_adv = _reference_prior_adv(
        close,
        volume,
        panel.day_offsets,
    )
    compute_prior_adv = _require_attr(
        lens_module,
        "compute_prior_adv",
        "prior_adv excludes own session (spec item 4)",
    )
    actual_prior_adv = compute_prior_adv(panel)

    start2 = int(panel.day_offsets[2])
    assert bool(np.all(expected_prior_adv[start2, :] < 1000.0))
    np.testing.assert_allclose(
        actual_prior_adv,
        expected_prior_adv,
        equal_nan=True,
    )
    assert bool(np.all(actual_prior_adv[start2, :] < 1000.0))


def test_prior_adv_irregular_sessions_no_fixed_stride() -> None:
    symbols = ("R0", "R1", "R2")
    symbol_volume = np.array(
        [
            [100.0, 150.0, 200.0],
            [110.0, 160.0, 210.0],
            [120.0, 170.0, 220.0],
        ]
    )
    panel = _panel_from_symbol_series(
        symbols,
        np.zeros(3),
        symbol_volume,
        bars_per_session=[375, 60, 105],
        symbol_base_price=1.0,
    )
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    expected_prior_adv = _reference_prior_adv(
        close,
        volume,
        panel.day_offsets,
    )
    compute_prior_adv = _require_attr(
        lens_module,
        "compute_prior_adv",
        "prior_adv irregular sessions (spec item 5)",
    )
    actual_prior_adv = compute_prior_adv(panel)

    np.testing.assert_allclose(
        actual_prior_adv,
        expected_prior_adv,
        equal_nan=True,
    )

    start, end = int(panel.day_offsets[2]), int(panel.day_offsets[3])
    column = actual_prior_adv[start:end, 0]
    finite_vals = column[np.isfinite(column)]
    assert finite_vals.size > 0
    assert bool(np.all(finite_vals == finite_vals[0]))

    start, end = int(panel.day_offsets[1]), int(panel.day_offsets[2])
    column = actual_prior_adv[start:end, 1]
    finite_vals = column[np.isfinite(column)]
    assert finite_vals.size > 0
    assert bool(np.all(finite_vals == finite_vals[0]))

    lens_obj = Lens(panel)
    report = lens_obj.stability(lens_obj.feature("return_1"), horizon=1)
    assert report is not None


def test_prior_adv_nan_propagates_never_filled() -> None:
    symbols = tuple(f"N{i}" for i in range(6))
    symbol_volume = np.tile(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
        (4, 1),
    ) * (1.0 + 0.05 * np.arange(4))[:, None]
    panel = _panel_from_symbol_series(
        symbols,
        np.zeros(6),
        symbol_volume,
        bars_per_session=5,
        symbol_base_price=1.0,
    )

    close0 = panel.field("close").copy()
    volume0 = panel.field("volume").copy()
    baseline_close = panel.field("close").astype(np.float64)
    baseline_volume = panel.field("volume").astype(np.float64)
    target_row = int(panel.day_offsets[1])
    close0[target_row, 2] = np.nan
    volume0[target_row, 2] = np.nan
    mutated_panel = Panel(
        fields={"close": close0, "volume": volume0},
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )

    start1, end1 = int(panel.day_offsets[1]), int(panel.day_offsets[2])
    baseline_day_value = np.nansum(
        baseline_close[start1:end1, 2] * baseline_volume[start1:end1, 2]
    )
    mutated_close = mutated_panel.field("close").astype(np.float64)
    mutated_volume = mutated_panel.field("volume").astype(np.float64)
    mutated_day_value = np.nansum(
        mutated_close[start1:end1, 2] * mutated_volume[start1:end1, 2]
    )
    removed_bar_contribution = (
        baseline_close[target_row, 2] * baseline_volume[target_row, 2]
    )

    assert bool(np.isfinite(mutated_day_value))
    assert bool(
        np.isclose(
            mutated_day_value,
            baseline_day_value - removed_bar_contribution,
            rtol=1e-5,
        )
    )

    close1 = panel.field("close").copy()
    volume1 = panel.field("volume").copy()
    start0, end0 = int(panel.day_offsets[0]), int(panel.day_offsets[1])
    close1[start0:end0, 4] = np.nan
    volume1[start0:end0, 4] = np.nan
    mutated_panel_b = Panel(
        fields={"close": close1, "volume": volume1},
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )
    compute_prior_adv = _require_attr(
        lens_module,
        "compute_prior_adv",
        "prior_adv NaN propagation (spec item 6)",
    )
    prior_adv_b = compute_prior_adv(mutated_panel_b)

    # An ENTIRELY-ABSENT session must yield NaN, not 0.0.
    #
    # `np.nansum` over an all-NaN slice returns 0.0, and this assertion originally
    # pinned that raw numpy behaviour. The spec was amended (commit 42ba072) to FORBID
    # it: a symbol that did not trade at all on a session is not a zero-turnover name.
    # Recording it as 0.0 would rank it as maximally illiquid and drop it into decile 0
    # -- the exact decile whose spread decides criterion 4, and therefore H2 -- instead
    # of excluding it. That is a rule-6 violation: NaN means "no bar", never zero.
    #
    # Note `validate.py`'s `tradable_mask` DOES use a bare nansum, deliberately and
    # correctly for its own purpose (absent -> 0 ADV -> not tradable). The two uses
    # legitimately differ; they must not be "harmonised".
    #
    # Corrected by the lead as spec author, NOT by an implementer making a test pass:
    # the contract changed after this suite was authored. Symbol 4's only session in the
    # trailing window is entirely absent, so its prior_adv is undefined, not zero.
    assert bool(np.isnan(prior_adv_b[start1, 4]))
    assert bool(np.isfinite(prior_adv_b[start1, 3]))


def test_method_cross_sectional_rank_produces_nonzero_spread() -> None:
    # N=50 (not 55): with 10 causal_buckets deciles, cross_sectional_rank's percentile-
    # rank edges land EXACTLY on multiples of 5 only when N=50 (verified empirically:
    # decile sizes are precisely [5]*10). At N=55 the 6th-lowest-turnover symbol (a
    # flat filler, return 0.0) leaks into decile 0 alongside the 5 signal members and
    # ties with the signal member whose return is also 0.0, corrupting the exact
    # bucket-edge arithmetic this test relies on -- discovered by running this fixture,
    # not by re-deriving the edge formula on paper, per the task's validation step.
    symbols = tuple(f"M{i:02d}" for i in range(50))
    symbol_returns = np.concatenate(
        [np.linspace(-0.01, 0.01, 5), np.zeros(45, dtype=float)]
    )
    symbol_volume = np.tile(
        np.concatenate(
            [
                np.full(5, 1000.0),
                np.linspace(50_000.0, 5_000_000.0, 45),
            ]
        ),
        (15, 1),
    )
    symbol_base_price = np.full(50, 10.0)

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        bars_per_session=10,
        symbol_base_price=symbol_base_price,
    )
    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    actual = lens_obj.stability(feature, horizon=1).by_liquidity_decile

    assert 0 in actual
    assert np.isfinite(actual[0].spread_bps)
    assert actual[0].spread_bps != 0.0
    expected_spread_bps = (0.01 - (-0.01)) * 1e4
    assert np.isclose(
        actual[0].spread_bps,
        expected_spread_bps,
        rtol=1e-6,
        atol=1e-6,
    )


def test_criterion4_fires_when_ratio_exceeds_threshold_and_bottom_is_argmax() -> None:
    threshold_for_fixture = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    symbols = tuple(f"K{i:02d}" for i in range(50))
    spread_targets = {
        # ratio = spread_targets[0] / m_bps(=5.0) must exceed `threshold`; multiplying
        # by 5.0 here (not just `threshold * 4.0`) is what actually encodes that ratio
        # -- an earlier draft omitted this and produced ratio=1.6 < threshold=2.0,
        # found by running the fixture, not by re-deriving the ratio algebra on paper.
        0: 5.0 * threshold_for_fixture * 4.0,
        4: 5.0,
        8: 5.0,
    }
    return_parts = []
    volume_parts = []
    for tier in range(10):
        count = 5  # N=50: 10 tiers x 5 -> clean decile boundaries (verified empirically)
        if tier in spread_targets:
            half = spread_targets[tier] / 2.0 / 1e4
            return_parts.append(np.linspace(-half, half, count))
        else:
            return_parts.append(np.zeros(count, dtype=float))
        volume_parts.append(np.full(count, 10.0 ** (tier + 1)))

    symbol_returns = np.concatenate(return_parts)
    symbol_volume = np.tile(np.concatenate(volume_parts), (3, 1))
    symbol_base_price = np.full(50, 1.0)

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        bars_per_session=76,  # 75 "open" bars (555-629) + 1 "mid" bar (630): >= 2
        # by_time_of_day buckets get populated, avoiding criterion 4's UNRELATED
        # single-time-of-day-bucket auto-FAIL branch, which would otherwise mask
        # the liquidity-decile behavior this test targets.
        symbol_base_price=symbol_base_price,
    )
    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    threshold = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    assert threshold > 1.0
    verdict = lens_obj.verdict(
        "H_c4",
        feature,
        1,
        latency_profile=None,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=10,
    )

    assert verdict.reasons[3].startswith("4.")
    assert "FAIL" in verdict.reasons[3]
    assert "concentrated in bottom liquidity decile" in verdict.reasons[3]


def test_criterion4_does_not_fire_when_ratio_below_threshold() -> None:
    threshold_for_fixture = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    ratio_below = 1.0 + (threshold_for_fixture - 1.0) * 0.5
    symbols = tuple(f"K{i:02d}" for i in range(50))
    spread_targets = {
        0: 5.0 * ratio_below,
        4: 5.0,
        8: 5.0,
    }
    return_parts = []
    volume_parts = []
    for tier in range(10):
        count = 5  # N=50: 10 tiers x 5 -> clean decile boundaries (verified empirically)
        if tier in spread_targets:
            half = spread_targets[tier] / 2.0 / 1e4
            return_parts.append(np.linspace(-half, half, count))
        else:
            return_parts.append(np.zeros(count, dtype=float))
        volume_parts.append(np.full(count, 10.0 ** (tier + 1)))

    symbol_returns = np.concatenate(return_parts)
    symbol_volume = np.tile(np.concatenate(volume_parts), (3, 1))
    symbol_base_price = np.full(50, 1.0)

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        bars_per_session=76,  # 75 "open" bars (555-629) + 1 "mid" bar (630): >= 2
        # by_time_of_day buckets get populated, avoiding criterion 4's UNRELATED
        # single-time-of-day-bucket auto-FAIL branch, which would otherwise mask
        # the liquidity-decile behavior this test targets.
        symbol_base_price=symbol_base_price,
    )
    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    threshold = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    assert threshold > 1.0
    verdict = lens_obj.verdict(
        "H_c4",
        feature,
        1,
        latency_profile=None,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=10,
    )

    assert verdict.reasons[3].startswith("4.")
    assert "PASS" in verdict.reasons[3]


def test_criterion4_does_not_fire_when_bottom_is_not_argmax() -> None:
    threshold_for_fixture = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    symbols = tuple(f"K{i:02d}" for i in range(50))
    spread_targets = {
        0: 5.0,
        4: 5.0 * threshold_for_fixture * 3.0,
        8: 5.0,
    }
    return_parts = []
    volume_parts = []
    for tier in range(10):
        count = 5  # N=50: 10 tiers x 5 -> clean decile boundaries (verified empirically)
        if tier in spread_targets:
            half = spread_targets[tier] / 2.0 / 1e4
            return_parts.append(np.linspace(-half, half, count))
        else:
            return_parts.append(np.zeros(count, dtype=float))
        volume_parts.append(np.full(count, 10.0 ** (tier + 1)))

    symbol_returns = np.concatenate(return_parts)
    symbol_volume = np.tile(np.concatenate(volume_parts), (3, 1))
    symbol_base_price = np.full(50, 1.0)

    panel = _panel_from_symbol_series(
        symbols,
        symbol_returns,
        symbol_volume,
        bars_per_session=76,  # 75 "open" bars (555-629) + 1 "mid" bar (630): >= 2
        # by_time_of_day buckets get populated, avoiding criterion 4's UNRELATED
        # single-time-of-day-bucket auto-FAIL branch, which would otherwise mask
        # the liquidity-decile behavior this test targets.
        symbol_base_price=symbol_base_price,
    )
    lens_obj = Lens(panel)
    feature = lens_obj.feature("return_1")
    threshold = _require_attr(
        lens_module,
        "CONCENTRATION_RATIO_THRESHOLD",
        "criterion-4 firing threshold must be a named module constant (spec item 5)",
    )
    assert threshold > 1.0
    verdict = lens_obj.verdict(
        "H_c4",
        feature,
        1,
        latency_profile=None,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=10,
    )

    assert verdict.reasons[3].startswith("4.")
    assert "PASS" in verdict.reasons[3]


def test_all_seven_criteria_reported_in_order() -> None:
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    verdict = _call_verdict(
        panel,
        latency_profile=_FAIL_LATENCY,
        effective_n_trials=1,
    )

    assert len(verdict.reasons) == 7
    for i, reason in enumerate(verdict.reasons):
        assert reason.startswith(f"{i + 1}.")


def test_verdict_determinism_same_inputs_and_seed() -> None:
    panel = _all_pass_panel(seed=123)
    lens_obj = Lens(panel, seed=0)
    kwargs = dict(
        latency_profile=_PASS_LATENCY,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=50,
    )
    first = lens_obj.verdict("H_det", "return_1", 1, **kwargs)
    second = lens_obj.verdict("H_det", "return_1", 1, **kwargs)

    assert first.survived == second.survived
    assert first.reasons == second.reasons
    np.testing.assert_allclose(first.cost_hurdle_bps, second.cost_hurdle_bps)
