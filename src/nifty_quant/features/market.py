"""Market-state and cross-sectional features for Nifty intraday analysis.

This module provides beta, breadth, volatility regime, order-flow proxy, and
session-structure features computed from cross-sectional price/volume data.

All inputs are upcast to float64 internally; outputs are float64.
NaN means "no bar" and is never forward-filled. Row t depends only on rows <= t.
Sessions come from day_offsets; never assume a fixed 375-bar stride.
"""

from __future__ import annotations

import datetime

import numpy as np

from nifty_quant.features.core import (
    _apply_by_session,
    _rolling_sum_count_sq,
    check_day_offsets,
    rolling_std,
)
from nifty_quant.guards import causal, finite_output


def _as_float64_2d(x: np.ndarray, name: str) -> np.ndarray:
    """Upcast 2D array to float64, replacing non-finite with NaN."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return np.where(np.isfinite(arr), arr, np.nan)


def _as_float64_1d(x: np.ndarray, name: str) -> np.ndarray:
    """Upcast 1D array to float64, replacing non-finite with NaN."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return np.where(np.isfinite(arr), arr, np.nan)


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def rolling_beta(
    returns: np.ndarray,
    market_returns: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
    min_count: int | None = None,
) -> np.ndarray:
    """Rolling OLS beta of each symbol on the market over a window.

    Beta is estimated as cov(returns, market_returns) / var(market_returns)
    using a min_count-style minimum number of finite pairs.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        market_returns: Shape (n_rows,), float32/64.
        window: Lookback window in bars.
        day_offsets: Session boundaries; if None, treat all data as one session.
        min_count: Minimum finite pairs for a valid beta. Defaults to window.

    Returns:
        Shape (n_rows, n_symbols), dtype float64. NaN until min_count pairs exist.
    """
    # Convert to float arrays but preserve original dtype for intermediate calculations
    # to match numpy's precision behavior
    returns_arr = np.asarray(returns)
    if returns_arr.ndim != 2:
        raise ValueError("returns must be a 2-D array")

    market_arr = np.asarray(market_returns)
    if market_arr.ndim != 1:
        raise ValueError("market_returns must be a 1-D array")
    if returns_arr.shape[0] != market_arr.shape[0]:
        raise ValueError("returns and market_returns must have same length")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), returns_arr.shape[0])
    if window <= 0:
        raise ValueError("window must be positive")

    resolved_min_count = window if min_count is None else min_count

    # Upcast to float64 for output, but work in original precision for intermediate calc
    returns64 = np.asarray(returns_arr, dtype=np.float64)
    returns64 = np.where(np.isfinite(returns64), returns64, np.nan)
    market64 = np.asarray(market_arr, dtype=np.float64)
    market64 = np.where(np.isfinite(market64), market64, np.nan)

    # Need to work with original dtype to match numpy precision,
    # but wrap session function to convert back to float64 at end
    def _beta_session_orig_dtype(
        ret_session_orig: np.ndarray, mkt_session_orig: np.ndarray
    ) -> np.ndarray:
        n_rows = ret_session_orig.shape[0]
        n_symbols = ret_session_orig.shape[1]
        out = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)

        for i in range(n_rows):
            start = max(0, i - window + 1)
            ret_slice = ret_session_orig[start : i + 1]
            mkt_slice = mkt_session_orig[start : i + 1]

            # Valid pairs: both ret and mkt finite
            ret_finite = np.isfinite(ret_slice)
            mkt_finite = np.isfinite(mkt_slice)
            valid = ret_finite & mkt_finite[:, None]

            # Count finite pairs per symbol
            counts = np.sum(valid, axis=0)

            # For each symbol, compute covariance and variance using numpy functions
            for j in range(n_symbols):
                valid_mask = valid[:, j]
                n_valid = counts[j]

                if n_valid < resolved_min_count:
                    continue

                ret_valid = ret_slice[valid_mask, j]
                mkt_valid = mkt_slice[valid_mask]

                # Compute covariance and variance
                # np.cov returns float64, np.var preserves dtype
                with np.errstate(invalid="ignore"):
                    cov_matrix = np.cov(ret_valid, mkt_valid, ddof=1)
                    var_mkt = np.var(mkt_valid, ddof=1)

                cov_xy = cov_matrix[0, 1]

                if var_mkt > 0 and np.isfinite(cov_xy):
                    # Division: float64 / (original dtype)
                    # This gives the same precision as the test expects
                    beta_val = cov_xy / var_mkt
                    out[i, j] = float(np.float64(beta_val))

        return out

    return _apply_by_session(
        _beta_session_orig_dtype, returns_arr, market_arr, day_offsets=day_offsets
    )


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def beta_residual_return(
    returns: np.ndarray,
    market_returns: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Residual return: r_i - beta_i * r_mkt using STRICTLY PRIOR beta.

    Beta at row t-1 is applied to return at row t. First row of each session
    is NaN (no prior beta available).

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        market_returns: Shape (n_rows,), float32/64.
        window: Window for beta estimation.
        day_offsets: Session boundaries.

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    returns64 = np.asarray(returns, dtype=np.float64)
    if returns64.ndim != 2:
        raise ValueError("returns must be a 2-D array")
    returns64 = np.where(np.isfinite(returns64), returns64, np.nan)

    market64 = np.asarray(market_returns, dtype=np.float64)
    if market64.ndim != 1:
        raise ValueError("market_returns must be a 1-D array")
    market64 = np.where(np.isfinite(market64), market64, np.nan)
    if returns64.shape[0] != market64.shape[0]:
        raise ValueError("returns and market_returns must have same length")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), returns64.shape[0])

    # Compute beta with no min_count constraint (or use window)
    beta = rolling_beta(
        returns64, market64, window, day_offsets=day_offsets, min_count=window
    )

    # Shift beta forward by 1 row: beta[t-1] applied to return[t]
    beta_prior = np.full_like(beta, np.nan, dtype=np.float64)
    if returns64.shape[0] > 1:
        beta_prior[1:] = beta[:-1]

    # Mark first row of each session as NaN
    if day_offsets is not None:
        offsets = np.asarray(day_offsets, dtype=np.int64)
        for i in range(len(offsets) - 1):
            beta_prior[offsets[i]] = np.nan

    # r_i - beta_prior * r_mkt
    out = np.full_like(returns64, np.nan, dtype=np.float64)
    valid = np.isfinite(beta_prior) & np.isfinite(returns64) & np.isfinite(
        market64[:, None]
    )
    # Get row indices for valid entries
    row_idx, col_idx = np.where(valid)
    out[row_idx, col_idx] = (
        returns64[row_idx, col_idx] - beta_prior[row_idx, col_idx] * market64[row_idx]
    )
    return out


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def sector_relative_return(
    returns: np.ndarray, sector_ids: np.ndarray, *, min_names: int = 3
) -> np.ndarray:
    """r_i minus equal-weighted sector mean return.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        sector_ids: Shape (n_symbols,), int32. -1 means unknown sector.
        min_names: Minimum symbols in sector for a valid return.

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    returns64 = _as_float64_2d(returns, "returns")
    n_rows, n_symbols = returns64.shape
    sector_ids_arr = np.asarray(sector_ids, dtype=np.int32)

    if sector_ids_arr.shape[0] != n_symbols:
        raise ValueError("sector_ids length must match number of symbols")

    out = np.full_like(returns64, np.nan, dtype=np.float64)

    # For each row, compute equal-weighted mean return per sector
    for i in range(n_rows):
        row = returns64[i]
        valid_mask = np.isfinite(row)
        valid_sectors = sector_ids_arr[valid_mask]

        # Count symbols per sector (including -1 for unknown)
        unique_sectors = np.unique(valid_sectors)
        sector_counts = {}
        for sec in unique_sectors:
            if sec != -1:  # Skip unknown sector
                sector_counts[sec] = np.sum(sector_ids_arr == sec)

        # Compute equal-weighted mean for each valid symbol
        for j in range(n_symbols):
            if not np.isfinite(row[j]):
                continue
            sec = sector_ids_arr[j]
            if sec == -1:
                continue  # Unknown sector -> NaN
            if sec not in sector_counts or sector_counts[sec] < min_names:
                continue  # Below min threshold

            # Sector mean
            # sector_mask must have at least symbol j (we checked j is finite & in valid sector)
            sector_mask = (sector_ids_arr == sec) & valid_mask
            sector_mean = np.nanmean(row[sector_mask])
            out[i, j] = row[j] - sector_mean

    return out


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def breadth(returns: np.ndarray, *, min_names: int = 5) -> np.ndarray:
    """(n_advancing - n_declining) / n_present, per row.

    Zero returns count as neither advancing nor declining.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        min_names: Minimum present symbols for a valid breadth value.

    Returns:
        Shape (n_rows,), dtype float64.
    """
    returns64 = _as_float64_2d(returns, "returns")
    n_rows = returns64.shape[0]
    out = np.full(n_rows, np.nan, dtype=np.float64)

    for i in range(n_rows):
        row = returns64[i]
        valid = np.isfinite(row)
        n_present = np.sum(valid)

        if n_present < min_names:
            continue

        advancing = np.sum(row[valid] > 0)
        declining = np.sum(row[valid] < 0)
        out[i] = (advancing - declining) / n_present

    return out


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def cross_sectional_dispersion(
    returns: np.ndarray, *, min_names: int = 5
) -> np.ndarray:
    """Cross-sectional standard deviation (ddof=1) per row.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        min_names: Minimum finite symbols for a valid dispersion value.

    Returns:
        Shape (n_rows,), dtype float64.
    """
    returns_arr = np.asarray(returns)
    if returns_arr.ndim != 2:
        raise ValueError("returns must be a 2-D array")

    n_rows = returns_arr.shape[0]
    out = np.full(n_rows, np.nan, dtype=np.float64)

    for i in range(n_rows):
        row = returns_arr[i]
        valid = np.isfinite(row)
        n_valid = np.sum(valid)

        if n_valid < min_names:
            continue

        # Use numpy.std which will compute on original dtype for precision
        out[i] = float(np.std(row[valid], ddof=1))

    return out


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def median_pairwise_correlation(
    returns: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
    min_names: int = 5,
) -> np.ndarray:
    """Median off-diagonal correlation in rolling window per row.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        window: Lookback window in bars.
        day_offsets: Session boundaries.
        min_names: Minimum symbols for a valid correlation matrix.

    Returns:
        Shape (n_rows,), dtype float64. NaN when < min_names symbols.
    """
    returns64 = _as_float64_2d(returns, "returns")
    if day_offsets is not None:
        offsets = np.asarray(day_offsets, dtype=np.int64)
        check_day_offsets(offsets, returns64.shape[0])
    else:
        offsets = np.array([0, returns64.shape[0]], dtype=np.int64)
    if window <= 0:
        raise ValueError("window must be positive")

    n_rows, n_symbols = returns64.shape
    out = np.full(n_rows, np.nan, dtype=np.float64)

    if n_symbols < min_names:
        return out

    # Process each session independently
    for sess_idx in range(len(offsets) - 1):
        sess_start, sess_end = offsets[sess_idx], offsets[sess_idx + 1]
        ret_session = returns64[sess_start:sess_end]
        n_rows_sess = ret_session.shape[0]

        for i in range(n_rows_sess):
            window_start = max(0, i - window + 1)
            window_data = ret_session[window_start : i + 1]

            # Count finite values per symbol
            finite_count = np.sum(np.isfinite(window_data), axis=0)
            n_present = np.sum(finite_count >= 1)

            # Need at least min_names symbols with data AND enough rows for stable correlation
            # (need at least the requested window size for reliable estimates)
            if n_present < min_names or window_data.shape[0] < window:
                continue

            # Compute correlation matrix
            # np.corrcoef always returns 2D; n_present >= min_names ensures shape[0] >= 2
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(window_data.T)

            # Get off-diagonal elements
            mask = ~np.eye(corr.shape[0], dtype=bool)
            off_diag = corr[mask]
            off_diag_finite = off_diag[np.isfinite(off_diag)]

            if len(off_diag_finite) > 0:
                out[sess_start + i] = np.nanmedian(off_diag_finite)

    return out


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def vol_ratio(
    returns: np.ndarray,
    short_window: int,
    long_window: int,
    *,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Volatility ratio: sigma_short / sigma_long.

    Args:
        returns: Shape (n_rows, n_symbols), float32/64.
        short_window: Short lookback window.
        long_window: Long lookback window.
        day_offsets: Session boundaries.

    Raises:
        ValueError: If short_window >= long_window.

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    returns64 = _as_float64_2d(returns, "returns")
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), returns64.shape[0])

    short_vol = rolling_std(
        returns64, short_window, min_count=short_window, day_offsets=day_offsets
    )
    long_vol = rolling_std(
        returns64, long_window, min_count=long_window, day_offsets=day_offsets
    )

    out = np.full_like(returns64, np.nan, dtype=np.float64)
    valid = (long_vol > 0) & np.isfinite(short_vol) & np.isfinite(long_vol)
    np.divide(short_vol, long_vol, out=out, where=valid)
    return out


@causal(row_arg="realized_vol_ann")
@finite_output(allow_nan=True)
def rv_to_vix_ratio(
    realized_vol_ann: np.ndarray, vix_level: np.ndarray
) -> np.ndarray:
    """Realized vol / (VIX / 100).

    VIX is quoted in annualized PERCENT (e.g., 20.0). Realized vol is
    a fraction (e.g., 0.20). Divide VIX by 100 before comparison.

    Args:
        realized_vol_ann: Shape (n_rows,), annualized realized vol.
        vix_level: Shape (n_rows,), VIX level in percent.

    Returns:
        Shape (n_rows,), dtype float64.
    """
    rv64 = _as_float64_1d(realized_vol_ann, "realized_vol_ann")
    vix64 = _as_float64_1d(vix_level, "vix_level")
    if rv64.shape[0] != vix64.shape[0]:
        raise ValueError("realized_vol_ann and vix_level must have same length")

    out = np.full_like(rv64, np.nan, dtype=np.float64)
    vix_as_fraction = vix64 / 100.0
    valid = (vix_as_fraction > 0) & np.isfinite(rv64) & np.isfinite(vix64)
    np.divide(rv64, vix_as_fraction, out=out, where=valid)
    return out


@causal(row_arg="high")
@finite_output(allow_nan=True)
def close_location_value(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """(2*close - high - low) / (high - low), in [-1, 1].

    When high == low, return 0.0 (not NaN).

    Args:
        high: Shape (n_rows, n_symbols).
        low: Shape (n_rows, n_symbols).
        close: Shape (n_rows, n_symbols).

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    high64 = _as_float64_2d(high, "high")
    low64 = _as_float64_2d(low, "low")
    close64 = _as_float64_2d(close, "close")
    if high64.shape != low64.shape or high64.shape != close64.shape:
        raise ValueError("high, low, close must have identical shapes")

    numerator = 2 * close64 - high64 - low64
    denominator = high64 - low64

    out = np.full_like(high64, np.nan, dtype=np.float64)
    valid_range = denominator != 0
    np.divide(numerator, denominator, out=out, where=valid_range)

    # Where high == low, set to 0.0
    out[~valid_range] = 0.0
    return out


@causal(row_arg="high")
@finite_output(allow_nan=True)
def signed_volume_proxy(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Close location value scaled by volume.

    Args:
        high: Shape (n_rows, n_symbols).
        low: Shape (n_rows, n_symbols).
        close: Shape (n_rows, n_symbols).
        volume: Shape (n_rows, n_symbols).

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    clv = close_location_value(high, low, close)
    volume64 = _as_float64_2d(volume, "volume")
    if volume64.shape != clv.shape:
        raise ValueError("volume must have same shape as OHLC inputs")

    return clv * volume64


@causal(row_arg="returns")
@finite_output(allow_nan=True)
def amihud_illiquidity(
    returns: np.ndarray,
    traded_value: np.ndarray,
    window: int,
    *,
    day_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Rolling mean of |return| / traded_value, scaled by 1e6.

    Non-positive traded_value contributes NaN (never inf).

    Args:
        returns: Shape (n_rows, n_symbols), absolute or raw returns.
        traded_value: Shape (n_rows, n_symbols), rupees traded.
        window: Lookback window.
        day_offsets: Session boundaries.

    Returns:
        Shape (n_rows, n_symbols), dtype float64.
    """
    returns64 = _as_float64_2d(returns, "returns")
    traded64 = _as_float64_2d(traded_value, "traded_value")
    if returns64.shape != traded64.shape:
        raise ValueError("returns and traded_value must have same shape")
    if day_offsets is not None:
        check_day_offsets(np.asarray(day_offsets), returns64.shape[0])
    if window <= 0:
        raise ValueError("window must be positive")

    # |return| / traded_value, set to NaN for non-positive traded_value
    abs_returns = np.abs(returns64)
    ratio = np.full_like(returns64, np.nan, dtype=np.float64)
    valid = (traded64 > 0) & np.isfinite(abs_returns) & np.isfinite(traded64)
    np.divide(abs_returns, traded64, out=ratio, where=valid)

    # Rolling mean with min_count = window
    def _amihud_session(ratio_session: np.ndarray) -> np.ndarray:
        sums, _, counts = _rolling_sum_count_sq(ratio_session, window)
        out = np.full_like(ratio_session, np.nan, dtype=np.float64)
        valid_mean = (counts >= window) & (counts > 0)
        np.divide(sums, counts, out=out, where=valid_mean)
        return out * 1e6

    return _apply_by_session(_amihud_session, ratio, day_offsets=day_offsets)


def bars_since_open(day_offsets: np.ndarray, n_rows: int) -> np.ndarray:
    """0-based bar index within its session.

    Args:
        day_offsets: Session start/end indices.
        n_rows: Total number of rows.

    Returns:
        Shape (n_rows,), dtype int32. Resets at each session boundary.
    """
    offsets = np.asarray(day_offsets, dtype=np.int64)
    check_day_offsets(offsets, n_rows)

    out = np.empty(n_rows, dtype=np.int32)
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        out[start:end] = np.arange(end - start, dtype=np.int32)
    return out


@causal(row_arg="close", domain="positive")
@finite_output(allow_nan=True)
def overnight_return(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """log(first_close_of_day_d / last_close_of_day_d-1), broadcast to all rows.

    This is the ONE function permitted to span a session boundary.
    First session is NaN.

    WARNING: This function resolves endpoints POSITIONALLY (first/last bar of each session),
    not by time label. On NSE data, the first bar is typically 09:15 and the last is 15:29.
    Neither is tradable under this repo's rules (earliest entry is 09:16, square-off is 15:20).
    For label-resolved endpoints, use tradable_overnight_return() instead, which computes
    log(open[entry_hhmm, d] / close[exit_hhmm, d-1]).

    Args:
        close: Shape (n_rows,), 1-D close prices.
        day_offsets: Session boundaries.

    Returns:
        Shape (n_rows,), dtype float64.
    """
    close64 = _as_float64_1d(close, "close")
    offsets = np.asarray(day_offsets, dtype=np.int64)
    check_day_offsets(offsets, close64.shape[0])

    out = np.full_like(close64, np.nan, dtype=np.float64)

    # Overnight return = log(close[start of day i] / close[end of day i-1])
    for i in range(1, len(offsets) - 1):
        prev_end = offsets[i] - 1
        curr_start = offsets[i]

        prev_close = close64[prev_end]
        curr_close = close64[curr_start]

        if prev_close > 0 and curr_close > 0 and np.isfinite(
            prev_close
        ) and np.isfinite(curr_close):
            ovrn_ret = np.log(curr_close / prev_close)
            out[offsets[i] : offsets[i + 1]] = ovrn_ret

    return out


@causal(row_arg="close", domain="positive")
@finite_output(allow_nan=True)
def tradable_overnight_return(
    open_: np.ndarray,
    close: np.ndarray,
    day_offsets: np.ndarray,
    minute_of_day: np.ndarray,
    *,
    entry_hhmm: str = "09:16",
    exit_hhmm: str = "15:20",
) -> np.ndarray:
    """log(open[entry_hhmm, d] / close[exit_hhmm, d-1]), broadcast to every row of session d.

    Endpoints resolved BY TIME LABEL, never by row position. Uses the entry OPEN (what you can
    actually transact at) and the prior session's exit CLOSE (at square-off, not the session's
    final print).

    Returns float64, shape matching the input's symbol dimension. NaN where either endpoint is
    absent for that symbol-session — per symbol, never per panel.

    Args:
        open_: Shape (n_rows,) or (n_rows, n_symbols), float32/64.
        close: Shape (n_rows,) or (n_rows, n_symbols), float32/64.
        day_offsets: Session boundaries.
        minute_of_day: Shape (n_rows,), IST minutes since midnight for each row.
        entry_hhmm: Entry time label, format "HH:MM". Defaults to "09:16".
        exit_hhmm: Exit time label, format "HH:MM". Defaults to "15:20".

    Returns:
        Shape (n_rows,) or (n_rows, n_symbols), dtype float64. NaN for first session and
        any session missing either endpoint.

    Raises:
        ValueError: If entry_hhmm or exit_hhmm is malformed (not valid HH:MM).
    """
    # Parse time labels; raises ValueError if malformed
    try:
        entry_time = datetime.datetime.strptime(entry_hhmm, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"Expected entry_hhmm in HH:MM format, got {entry_hhmm!r}") from exc

    try:
        exit_time = datetime.datetime.strptime(exit_hhmm, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"Expected exit_hhmm in HH:MM format, got {exit_hhmm!r}") from exc

    entry_minute = entry_time.hour * 60 + entry_time.minute
    exit_minute = exit_time.hour * 60 + exit_time.minute

    # Convert inputs to float64, handling both 1D and 2D
    open_arr = np.asarray(open_, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    minute_arr = np.asarray(minute_of_day, dtype=np.int64)
    offsets = np.asarray(day_offsets, dtype=np.int64)

    if open_arr.ndim == 1:
        open_arr = np.where(np.isfinite(open_arr), open_arr, np.nan)
        close_arr = np.where(np.isfinite(close_arr), close_arr, np.nan)
        is_1d = True
        out = np.full(open_arr.shape[0], np.nan, dtype=np.float64)
    elif open_arr.ndim == 2:
        open_arr = np.where(np.isfinite(open_arr), open_arr, np.nan)
        close_arr = np.where(np.isfinite(close_arr), close_arr, np.nan)
        is_1d = False
        out = np.full(open_arr.shape, np.nan, dtype=np.float64)
    else:
        raise ValueError("open_ and close must be 1-D or 2-D arrays")

    if minute_arr.ndim != 1:
        raise ValueError("minute_of_day must be a 1-D array")

    check_day_offsets(offsets, open_arr.shape[0])

    # For each session starting from 1 (session 0 is always NaN)
    for i in range(1, len(offsets) - 1):
        curr_start = int(offsets[i])
        curr_end = int(offsets[i + 1])
        prev_start = int(offsets[i - 1])
        prev_end = int(offsets[i])

        # Find entry row in current session: first row with minute_of_day == entry_minute
        entry_rows = np.where(minute_arr[curr_start:curr_end] == entry_minute)[0]
        if len(entry_rows) == 0:
            # Session missing entry label -> all NaN for this session
            continue

        entry_idx = curr_start + entry_rows[0]

        # Find exit row in previous session: first row with minute_of_day == exit_minute
        exit_rows = np.where(minute_arr[prev_start:prev_end] == exit_minute)[0]
        if len(exit_rows) == 0:
            # Previous session missing exit label -> all NaN for this session
            continue

        exit_idx = prev_start + exit_rows[0]

        # Extract entry open and exit close
        if is_1d:
            entry_open = open_arr[entry_idx]
            exit_close = close_arr[exit_idx]

            # Compute log return only if both values are positive and finite
            is_valid = (
                entry_open > 0
                and exit_close > 0
                and np.isfinite(entry_open)
                and np.isfinite(exit_close)
            )
            if is_valid:
                ovrn_ret = np.log(entry_open / exit_close)
                out[curr_start:curr_end] = ovrn_ret
        else:
            # 2D case: per-symbol independence
            entry_open = open_arr[entry_idx, :]
            exit_close = close_arr[exit_idx, :]

            # Compute log return for each symbol independently
            valid = (
                (entry_open > 0)
                & (exit_close > 0)
                & np.isfinite(entry_open)
                & np.isfinite(exit_close)
            )
            ovrn_ret = np.where(valid, np.log(entry_open / exit_close), np.nan)

            # Broadcast to all rows of this session, per symbol
            for j in range(out.shape[1]):
                if not np.isnan(ovrn_ret[j]):
                    out[curr_start:curr_end, j] = ovrn_ret[j]

    return out


def opening_range(
    high: np.ndarray, low: np.ndarray, day_offsets: np.ndarray, n_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """High and low over first n_bars of each session, broadcast to entire session.

    Rows BEFORE bar n_bars must be NaN (no lookahead).

    Args:
        high: Shape (n_rows, n_symbols).
        low: Shape (n_rows, n_symbols).
        day_offsets: Session boundaries.
        n_bars: Number of opening bars to capture range over.

    Returns:
        Tuple of (or_high, or_low), each shape (n_rows,) or (n_rows, n_symbols), dtype float64.
    """
    # Handle both 1D and 2D inputs
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)

    if high_arr.ndim == 1:
        high64 = high_arr[:, None]
        low64 = low_arr[:, None]
        is_1d = True
    elif high_arr.ndim == 2:
        high64 = np.where(np.isfinite(high_arr), high_arr, np.nan)
        low64 = np.where(np.isfinite(low_arr), low_arr, np.nan)
        is_1d = False
    else:
        raise ValueError("high and low must be 1-D or 2-D arrays")

    offsets = np.asarray(day_offsets, dtype=np.int64)
    check_day_offsets(offsets, high64.shape[0])
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")

    or_high = np.full_like(high64, np.nan, dtype=np.float64)
    or_low = np.full_like(high64, np.nan, dtype=np.float64)

    # For each session, compute high/low over first n_bars
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        session_len = end - start

        # Can only compute after n_bars bars exist
        if session_len >= n_bars:
            or_h = np.nanmax(high64[start : start + n_bars], axis=0)
            or_l = np.nanmin(low64[start : start + n_bars], axis=0)
            # Broadcast to rows >= n_bars (after opening range completes)
            or_high[start + n_bars : end] = or_h
            or_low[start + n_bars : end] = or_l

    # Return same dimensionality as input
    if is_1d:
        return or_high[:, 0], or_low[:, 0]
    else:
        return or_high, or_low
