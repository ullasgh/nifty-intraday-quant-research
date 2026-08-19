"""Core feature computation primitives.

All functions upcast float32 inputs to float64 internally and return float64
(except the breakout_* functions, which return bool). NaN means "no bar" and is
never forward-filled; rolling windows use a min_periods-style min_count semantic.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata

from nifty_quant.guards import causal, check_day_offsets, finite_output, panel_contract


def _as_float64_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return np.where(np.isfinite(arr), arr, np.nan)


def _validate_window(window: int) -> None:
    if window <= 0:
        raise ValueError("window must be a positive integer")


def _resolve_min_count(min_count: int | None, window: int) -> int:
    resolved = window if min_count is None else min_count
    if resolved < 0:
        raise ValueError("min_count must be non-negative")
    return resolved


def _apply_by_session(
    func: Callable[..., np.ndarray],
    *arrays: np.ndarray,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    if not arrays:
        raise ValueError("_apply_by_session requires at least one array")

    first = arrays[0]
    n_rows = first.shape[0]

    if day_offsets is None:
        return func(*arrays)

    offsets = np.asarray(day_offsets)
    if offsets.ndim != 1:
        raise ValueError("day_offsets must be 1-D")
    if offsets.shape[0] < 2:
        raise ValueError("day_offsets must have at least 2 entries (start and end)")
    if offsets[0] != 0 or offsets[-1] != n_rows:
        raise ValueError("day_offsets must start at 0 and end at n_rows")
    if n_rows == 0:
        return np.empty_like(first, dtype=np.float64)

    starts = offsets[:-1]
    ends = offsets[1:]

    out = np.empty(first.shape, dtype=np.float64)
    for start, end in zip(starts.tolist(), ends.tolist()):
        session_arrays = tuple(a[start:end] for a in arrays)
        out[start:end] = func(*session_arrays)
    return out


def _rolling_finite_counts(x: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(x)
    cum = np.cumsum(finite, axis=0, dtype=np.float64)
    if x.shape[0] > window:
        counts = cum.copy()
        counts[window:] = cum[window:] - cum[:-window]
        return counts
    return cum


def _rolling_sum_count_sq(
    x: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x)
    x_zero = np.where(finite, x, 0.0)

    cum_sum = np.cumsum(x_zero, axis=0, dtype=np.float64)
    cum_sq = np.cumsum(x_zero * x_zero, axis=0, dtype=np.float64)
    cum_count = np.cumsum(finite, axis=0, dtype=np.float64)

    if x.shape[0] > window:
        counts = cum_count.copy()
        sums = cum_sum.copy()
        sumsq = cum_sq.copy()

        counts[window:] = cum_count[window:] - cum_count[:-window]
        sums[window:] = cum_sum[window:] - cum_sum[:-window]
        sumsq[window:] = cum_sq[window:] - cum_sq[:-window]
        return sums, sumsq, counts

    return cum_sum, cum_sq, cum_count


def _rolling_mean_no_day(x: np.ndarray, window: int, min_count: int) -> np.ndarray:
    sums, _, counts = _rolling_sum_count_sq(x, window)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = (counts >= min_count) & (counts > 0)
    np.divide(sums, counts, out=out, where=valid)
    return out


def _rolling_std_no_day(x: np.ndarray, window: int, min_count: int, ddof: int) -> np.ndarray:
    sums, sumsq, counts = _rolling_sum_count_sq(x, window)
    out = np.full_like(x, np.nan, dtype=np.float64)

    valid = (counts >= min_count) & (counts > ddof)

    square_mean = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums * sums, counts, out=square_mean, where=valid)

    var_num = np.maximum(sumsq - square_mean, 0.0)
    denom = counts - ddof

    var = np.full_like(x, np.nan, dtype=np.float64)
    np.divide(var_num, denom, out=var, where=valid)
    np.sqrt(var, out=out, where=valid)
    return out


def _rolling_max_min_no_day(
    x: np.ndarray, window: int, min_count: int, use_max: bool
) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] == 0:
        return out

    counts = _rolling_finite_counts(x, window)

    if window > 1:
        pad = np.full((window - 1, x.shape[1]), np.nan, dtype=np.float64)
        x_padded = np.concatenate((pad, x), axis=0)
    else:
        x_padded = x

    # sliding_window_view over axis=0 appends the window as the LAST axis, so the
    # view has shape (n_rows, n_cols, window) - reduce over axis=-1, not axis=1
    # (axis=1 would incorrectly reduce across symbols instead of across time).
    view = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=window, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rolled = np.nanmax(view, axis=-1) if use_max else np.nanmin(view, axis=-1)

    valid = (counts >= min_count) & (counts > 0)
    out = np.where(valid, rolled, np.nan)
    return out


def _log_returns_no_day(close: np.ndarray) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if close.shape[0] <= 1:
        return out

    prev = close[:-1]
    curr = close[1:]
    valid = (prev > 0) & (curr > 0)

    ratio = np.full_like(curr, np.nan, dtype=np.float64)
    np.divide(curr, prev, out=ratio, where=valid)

    logged = np.full_like(curr, np.nan, dtype=np.float64)
    np.log(ratio, out=logged, where=valid)
    out[1:] = logged
    return out


def _shift_one_bar(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.shape[0] > 1:
        out[1:] = x[:-1]
    return out


@finite_output(allow_nan=True)
def log_returns(close: np.ndarray, *, day_offsets: np.ndarray | None = None) -> np.ndarray:
    """(n,k) log(close_t / close_{t-1}); first row of each session is NaN.

    close <= 0 at either t or t-1 makes that row NaN.
    """
    close64 = _as_float64_2d(close, "close")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), close64.shape[0])
    return _apply_by_session(_log_returns_no_day, close64, day_offsets=day_offsets)


@finite_output(allow_nan=True)
def rolling_mean(
    x: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Session-aware rolling mean with min_count-style handling of NaNs."""
    x64 = _as_float64_2d(x, "x")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), x64.shape[0])
    _validate_window(window)
    resolved_min_count = _resolve_min_count(min_count, window)

    return _apply_by_session(
        lambda arr: _rolling_mean_no_day(arr, window, resolved_min_count),
        x64,
        day_offsets=day_offsets,
    )


@finite_output(allow_nan=True)
def rolling_std(
    x: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    ddof: int = 1,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Session-aware rolling standard deviation with nan-aware cumsum stats."""
    x64 = _as_float64_2d(x, "x")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), x64.shape[0])
    _validate_window(window)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    resolved_min_count = _resolve_min_count(min_count, window)

    return _apply_by_session(
        lambda arr: _rolling_std_no_day(arr, window, resolved_min_count, ddof),
        x64,
        day_offsets=day_offsets,
    )


@finite_output(allow_nan=True)
def rolling_max(
    x: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Session-aware rolling max using sliding windows and finite counts."""
    x64 = _as_float64_2d(x, "x")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), x64.shape[0])
    _validate_window(window)
    resolved_min_count = _resolve_min_count(min_count, window)

    return _apply_by_session(
        lambda arr: _rolling_max_min_no_day(arr, window, resolved_min_count, True),
        x64,
        day_offsets=day_offsets,
    )


@finite_output(allow_nan=True)
def rolling_min(
    x: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Session-aware rolling min using sliding windows and finite counts."""
    x64 = _as_float64_2d(x, "x")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), x64.shape[0])
    _validate_window(window)
    resolved_min_count = _resolve_min_count(min_count, window)

    return _apply_by_session(
        lambda arr: _rolling_max_min_no_day(arr, window, resolved_min_count, False),
        x64,
        day_offsets=day_offsets,
    )


@finite_output(allow_nan=True)
def rolling_zscore(
    x: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    ddof: int = 1,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """(x - rolling_mean) / rolling_std. Zero or NaN std yields NaN."""
    x64 = _as_float64_2d(x, "x")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), x64.shape[0])
    mean = rolling_mean(x64, window, min_count=min_count, day_offsets=day_offsets)
    std = rolling_std(x64, window, min_count=min_count, ddof=ddof, day_offsets=day_offsets)

    out = np.full_like(x64, np.nan, dtype=np.float64)
    valid = (std > 0) & np.isfinite(std)
    np.divide(x64 - mean, std, out=out, where=valid)
    return out


@finite_output(allow_nan=True)
def parkinson_volatility(
    high: np.ndarray,
    low: np.ndarray,
    window: int,
    *,
    min_count: int | None = None,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """sqrt(rolling_mean(ln(high/low)^2) / (4 ln 2)) per bar.

    Invalid high/low pairs become NaN log-range terms and count as missing.
    """
    high64 = _as_float64_2d(high, "high")
    low64 = _as_float64_2d(low, "low")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), high64.shape[0])
    if high64.shape != low64.shape:
        raise ValueError("high and low must have identical shapes")

    valid = (high64 > 0) & (low64 > 0) & (high64 >= low64)

    ratio = np.full_like(high64, np.nan, dtype=np.float64)
    np.divide(high64, low64, out=ratio, where=valid)

    log_ratio = np.full_like(ratio, np.nan, dtype=np.float64)
    np.log(ratio, out=log_ratio, where=valid)

    log_range_sq = np.square(log_ratio)
    mean_log_range_sq = rolling_mean(
        log_range_sq, window, min_count=min_count, day_offsets=day_offsets
    )
    return np.sqrt(mean_log_range_sq / (4.0 * np.log(2.0)))


def annualization_factor(day_offsets: np.ndarray, *, sessions_per_year: int = 252) -> float:
    """sqrt(sessions_per_year * median_bars_per_session).

    Bars per session is the MEDIAN of np.diff(day_offsets) over every session
    present -- median, not mean, so a 60-bar Muhurat session or a 105-bar
    shortened/disaster-recovery session does not drag the estimate (rule 5: never
    assume a fixed 375-bar session stride). No `window` parameter and no trailing
    slice: bars-per-session is a CALENDAR property, not a price property, so
    deriving it from the whole panel introduces no lookahead -- knowing a future
    session was a 60-bar Muhurat session tells you nothing about any return
    (specs/portfolio_vol_target.md, amendment 3).

    For N sessions, day_offsets carries N+1 entries (N starts plus the final end).
    Raises ValueError on a ZERO-session panel (len(day_offsets) < 2) rather than
    returning NaN: a NaN annualization factor would propagate silently into every
    downstream sigma, weight and share count, surfacing as an empty book rather
    than as an error (amendment 4 addendum).
    """
    offsets = np.asarray(day_offsets)
    if offsets.ndim != 1:
        raise ValueError("day_offsets must be 1-D")
    session_lengths = np.diff(offsets)
    if session_lengths.size == 0:
        raise ValueError(
            "annualization_factor: day_offsets has zero sessions "
            f"(shape={offsets.shape}); at least a start and an end are required"
        )
    bars_per_session = float(np.median(session_lengths))
    return float(np.sqrt(sessions_per_year * bars_per_session))


@finite_output(allow_nan=True)
def ewma_volatility_ann(
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    halflife: float,
    sessions_per_year: int = 252,
) -> np.ndarray:
    """Causal, session-bounded EWMA volatility of 1-bar log returns, annualized.

    `halflife` (in bars) is a REQUIRED, un-defaulted parameter (rule 8): it is a
    genuine research choice governing how price history is weighted, and Phase E
    measures which value best predicts realised forward volatility
    (specs/portfolio_vol_target.md, amendments 1 and 3). A caller must pass one
    explicitly -- there is no silently-picked default, and passing `halflife=None`
    raises just as omitting it does.

    Returns are session-bounded via `log_returns` (the first bar of every session
    is NaN -- no overnight/cross-session return is ever formed, so no overnight gap
    poisons the vol estimate). The EWMA VARIANCE STATE is NOT reset at session
    boundaries: at a session's first bar there is no new return to observe, so the
    running estimate simply carries its prior level forward rather than restarting
    from an uninformative zero, which would otherwise force every session to "warm
    up" from scratch. Squared log returns update the state with decay
    `alpha = 1 - 0.5 ** (1 / halflife)`.

    Annualized via `annualization_factor(day_offsets, sessions_per_year=...)` --
    the median bars-per-session over the whole panel; see that function's
    docstring for why no trailing window is needed here.
    """
    if halflife is None:
        raise ValueError("halflife is required and cannot be None")
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError("halflife must be a finite positive number")

    close64 = _as_float64_2d(close, "close")
    offsets = np.asarray(day_offsets)
    check_day_offsets(offsets, close64.shape[0])

    returns = log_returns(close64, day_offsets=offsets)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)

    r2 = returns * returns
    n_rows, n_cols = r2.shape
    var = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    state = np.full(n_cols, np.nan, dtype=np.float64)
    for t in range(n_rows):
        row = r2[t]
        has_obs = np.isfinite(row)
        state_valid = np.isfinite(state)
        updated = alpha * row + (1.0 - alpha) * state
        state = np.where(
            has_obs & state_valid,
            updated,
            np.where(has_obs & ~state_valid, row, state),
        )
        var[t] = state

    sigma_bar = np.sqrt(var)
    factor = annualization_factor(offsets, sessions_per_year=sessions_per_year)
    return sigma_bar * factor


@panel_contract(ndim=2, dtype_out=np.float64)
@finite_output(allow_nan=True)
def cross_sectional_zscore(
    x: np.ndarray, *, min_names: int = 5, robust: bool = False
) -> np.ndarray:
    """Standardize across symbols within each row, ignoring NaNs."""
    x64 = _as_float64_2d(x, "x")
    finite_count = np.isfinite(x64).sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if robust:
            center = np.nanmedian(x64, axis=1)
            spread = np.nanmedian(np.abs(x64 - center[:, None]), axis=1)
            scale = 1.4826 * spread
        else:
            center = np.nanmean(x64, axis=1)
            scale = np.nanstd(x64, axis=1, ddof=0)

    valid_row = (finite_count >= min_names) & (scale > 0)

    out = np.full_like(x64, np.nan, dtype=np.float64)
    np.divide(
        x64 - center[:, None],
        scale[:, None],
        out=out,
        where=valid_row[:, None],
    )
    return out


@panel_contract(ndim=2, dtype_out=np.float64)
@finite_output(allow_nan=True)
def cross_sectional_rank(x: np.ndarray, *, pct: bool = True, min_names: int = 5) -> np.ndarray:
    """Rank across symbols within each row; NaNs stay NaN and are excluded."""
    x64 = _as_float64_2d(x, "x")
    finite_mask = np.isfinite(x64)
    finite_count = finite_mask.sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # nan_policy="omit" excludes NaNs from ranking on a per-row basis while
        # leaving their output as NaN; "propagate" (scipy default) would instead
        # turn an ENTIRE row NaN whenever a single NaN is present, which is wrong.
        ranks = rankdata(x64, method="average", axis=1, nan_policy="omit")

    if pct:
        denom = finite_count[:, None] - 1.0
        scaled = np.full_like(ranks, np.nan, dtype=np.float64)

        valid = (finite_count[:, None] > 1) & finite_mask
        np.divide(ranks - 1.0, denom, out=scaled, where=valid)

        single = (finite_count == 1)[:, None] & finite_mask
        scaled[single] = 0.0
        ranks = scaled

    valid_rows = finite_count >= min_names
    return np.where(valid_rows[:, None] & finite_mask, ranks, np.nan)


@finite_output(allow_nan=True)
def time_of_day_mean(x: np.ndarray, minute_of_day: np.ndarray) -> np.ndarray:
    """In-sample per-minute-of-day column means, broadcast back over rows.

    Warning: this is an in-sample statistic. Fit on training data only to avoid
    lookahead bias.
    """
    x64 = _as_float64_2d(x, "x")
    minute_arr = np.asarray(minute_of_day)

    if minute_arr.ndim != 1 or minute_arr.shape[0] != x64.shape[0]:
        raise ValueError("minute_of_day must be 1-D with length equal to n_rows")

    unique_mins, inverse = np.unique(minute_arr, return_inverse=True)
    n_buckets = unique_mins.shape[0]

    sums = np.zeros((n_buckets, x64.shape[1]), dtype=np.float64)
    counts = np.zeros((n_buckets, x64.shape[1]), dtype=np.float64)

    finite_mask = np.isfinite(x64)
    np.add.at(sums, inverse, np.where(finite_mask, x64, 0.0))
    np.add.at(counts, inverse, finite_mask.astype(np.float64))

    means = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=(counts > 0))
    return means[inverse]


@finite_output(allow_nan=True)
def time_of_day_mean_expanding(
    x: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    *,
    min_sessions: int = 1,
) -> np.ndarray:
    """Causal per-minute-of-day column means: strictly-prior-session expanding mean.

    For row r in session s at minute-of-day m, returns the mean of finite values
    at minute m over sessions [0, s) only -- the current session and every future
    session are excluded, so this is safe to use as a live/backtest signal input
    (unlike ``time_of_day_mean``, which is a whole-array in-sample statistic).
    Sessions before the ``min_sessions``-th prior observation at that minute
    yield NaN -- there is no silent fallback to an in-sample or partial estimate.

    Vectorized: no Python loop over rows. Groups rows into (session, minute)
    buckets, computes a per-bucket sum/count via ``np.add.at``, takes a
    cumulative sum over the session axis, and shifts it by one session so each
    row only ever sees strictly earlier sessions.
    """
    x64 = _as_float64_2d(x, "x")
    minute_arr = np.asarray(minute_of_day)
    if minute_arr.ndim != 1 or minute_arr.shape[0] != x64.shape[0]:
        raise ValueError("minute_of_day must be 1-D with length equal to n_rows")

    n_rows, n_cols = x64.shape
    offs = np.asarray(day_offsets)
    check_day_offsets(offs, n_rows)
    n_days = offs.shape[0] - 1
    row_session_id = np.repeat(np.arange(n_days), np.diff(offs))

    unique_mins, inverse = np.unique(minute_arr, return_inverse=True)
    n_buckets = unique_mins.shape[0]

    finite_mask = np.isfinite(x64)
    flat_idx = row_session_id * n_buckets + inverse

    per_sum = np.zeros((n_days * n_buckets, n_cols), dtype=np.float64)
    per_count = np.zeros((n_days * n_buckets, n_cols), dtype=np.float64)
    np.add.at(per_sum, flat_idx, np.where(finite_mask, x64, 0.0))
    np.add.at(per_count, flat_idx, finite_mask.astype(np.float64))

    per_sum = per_sum.reshape(n_days, n_buckets, n_cols)
    per_count = per_count.reshape(n_days, n_buckets, n_cols)

    cum_sum = np.cumsum(per_sum, axis=0)
    cum_count = np.cumsum(per_count, axis=0)

    zero = np.zeros((1, n_buckets, n_cols), dtype=np.float64)
    expanding_sum = np.concatenate([zero, cum_sum[:-1]], axis=0)
    expanding_count = np.concatenate([zero, cum_count[:-1]], axis=0)

    means = np.full((n_days, n_buckets, n_cols), np.nan, dtype=np.float64)
    valid = expanding_count >= min_sessions
    np.divide(expanding_sum, expanding_count, out=means, where=valid)

    return means[row_session_id, inverse]


@finite_output(allow_nan=True)
@causal(row_arg="x")
def deseasonalize_by_time_of_day(
    x: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
    mode: str = "expanding",
    min_sessions: int = 1,
) -> np.ndarray:
    """x / <per-minute-of-day mean>(x), with tiny denominator guard.

    mode="expanding" (default): causal. Uses ``time_of_day_mean_expanding`` --
    each row's normaliser only sees strictly prior sessions, so this is safe as
    a live/backtest signal input. If ``day_offsets`` is None, the whole array is
    treated as a single session, so every row has zero prior sessions and the
    output is entirely NaN (no silent lookahead fallback -- pass day_offsets to
    get a real normaliser).
    mode="in_sample": the OLD, NON-CAUSAL whole-array in-sample mean
    (``time_of_day_mean``). Full lookahead bias; research-only, never for
    backtest/live signal inputs. Emits a UserWarning.
    """
    x64 = _as_float64_2d(x, "x")

    if mode == "in_sample":
        warnings.warn(
            "deseasonalize_by_time_of_day(mode='in_sample') uses a whole-array "
            "in-sample mean and has full lookahead bias; it is research-only and "
            "must never feed a backtest or live signal. Use mode='expanding' "
            "(the default) for a causal normaliser.",
            UserWarning,
            stacklevel=2,
        )
        tod_mean = time_of_day_mean(x64, minute_of_day)
    elif mode == "expanding":
        offs = day_offsets if day_offsets is not None else np.array([0, x64.shape[0]])
        tod_mean = time_of_day_mean_expanding(
            x64, minute_of_day, offs, min_sessions=min_sessions
        )
    else:
        raise ValueError(f"mode must be 'expanding' or 'in_sample', got {mode!r}")

    out = np.full_like(x64, np.nan, dtype=np.float64)
    np.divide(x64, tod_mean, out=out, where=(np.abs(tod_mean) > eps))
    return out


@finite_output(allow_nan=True)
@causal(row_arg="volume")
def volume_zscore(
    volume: np.ndarray,
    minute_of_day: np.ndarray,
    window: int,
    *,
    deseasonalize: bool = True,
    min_count: int | None = None,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Deseasonalize, then rolling zscore; institutional block-volume feature."""
    volume64 = _as_float64_2d(volume, "volume")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), volume64.shape[0])

    if deseasonalize:
        volume64 = deseasonalize_by_time_of_day(volume64, minute_of_day, day_offsets)

    return rolling_zscore(volume64, window, min_count=min_count, day_offsets=day_offsets)


@causal(row_arg="close")
def breakout_up(
    close: np.ndarray,
    high: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """True where close exceeds prior-window rolling max of high.

    The rolling max is shifted one bar so bar t is excluded from the comparison
    window. Session boundaries are respected for both rolling and shifting.
    """
    close64 = _as_float64_2d(close, "close")
    high64 = _as_float64_2d(high, "high")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), close64.shape[0])
    if close64.shape != high64.shape:
        raise ValueError("close and high must have identical shapes")

    rolled_max = rolling_max(high64, window, min_count=window, day_offsets=day_offsets)
    shifted_max = _apply_by_session(_shift_one_bar, rolled_max, day_offsets=day_offsets)
    return close64 > shifted_max


@causal(row_arg="close")
def breakout_down(
    close: np.ndarray,
    low: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """True where close is below prior-window rolling min of low."""
    close64 = _as_float64_2d(close, "close")
    low64 = _as_float64_2d(low, "low")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), close64.shape[0])
    if close64.shape != low64.shape:
        raise ValueError("close and low must have identical shapes")

    rolled_min = rolling_min(low64, window, min_count=window, day_offsets=day_offsets)
    shifted_min = _apply_by_session(_shift_one_bar, rolled_min, day_offsets=day_offsets)
    return close64 < shifted_min
