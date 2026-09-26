"""Per-bar and per-session features for the VWAP +/- k*ADR fade study.

All functions take plain numpy arrays: per-bar arrays are ``(n_rows, n_sym)``, per-session
arrays are ``(n_days, n_sym)``. Session ``d`` covers rows ``day_offsets[d] :
day_offsets[d + 1]``; sessions have arbitrary lengths and no fixed stride is ever assumed.
A bar is present iff its ``close`` is finite. Nothing here fills, back-fills, or
interpolates any input. All arithmetic is float64 (float32 inputs are upcast first).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 09:16 IST. The 09:15 bar is excluded everywhere: it is measured to carry pre-open
# call-auction leakage (close > high in 61/61 observed OHLC violations).
SESSION_START_MINUTE = 556
# 15:20 IST. NSE MIS square-off.
SQUARE_OFF_MINUTE = 920


def _as_2d_float64(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, n_sym), got shape {out.shape}")
    return out


def _check_same_shape(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    shapes = {name: a.shape for name, a in arrays.items()}
    first = next(iter(shapes.values()))
    if any(s != first for s in shapes.values()):
        raise ValueError(f"per-bar array shapes disagree: {shapes}")
    return first[0], first[1]


def _validate_minute_of_day(minute_of_day: np.ndarray, n_rows: int) -> np.ndarray:
    mod = np.asarray(minute_of_day)
    if mod.ndim != 1 or mod.shape[0] != n_rows:
        raise ValueError(
            f"minute_of_day must be 1-D of length n_rows={n_rows}, got shape {mod.shape}"
        )
    return mod


def _validate_day_offsets(day_offsets: np.ndarray, n_rows: int) -> np.ndarray:
    offs = np.asarray(day_offsets)
    if offs.ndim != 1 or offs.shape[0] < 1:
        raise ValueError("day_offsets must be a non-empty 1-D array")
    if not np.issubdtype(offs.dtype, np.integer):
        raise ValueError(f"day_offsets must have an integer dtype, got {offs.dtype}")
    offs = offs.astype(np.int64)
    if offs[0] != 0:
        raise ValueError("day_offsets[0] must be 0")
    if offs[-1] != n_rows:
        raise ValueError(f"day_offsets[-1] must equal n_rows={n_rows}, got {offs[-1]}")
    if np.any(np.diff(offs) <= 0):
        raise ValueError("day_offsets must be strictly increasing")
    return offs


def session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    start_minute: int = SESSION_START_MINUTE,
) -> np.ndarray:
    """Session-anchored cumulative VWAP of typical price ``(high + low + close) / 3``.

    A row contributes iff ``minute_of_day >= start_minute`` and high/low/close/volume are
    all finite with ``volume >= 0``. Sums reset at every ``day_offsets`` boundary. The
    output is ``cum_pv / cum_v`` where ``cum_v > 0``, the bar is present (finite close)
    and the row is at/after ``start_minute``; NaN elsewhere. Non-contributing rows add
    exactly 0.0, so a later row's VWAP equals the VWAP computed with that row deleted.

    Each session is cumulated from zero (a loop over sessions, vectorized over rows and
    symbols inside) rather than via a global cumsum minus a session base, which would
    lose precision to cancellation on long panels.
    """
    h = _as_2d_float64("high", high)
    lo = _as_2d_float64("low", low)
    c = _as_2d_float64("close", close)
    v = _as_2d_float64("volume", volume)
    n_rows, _ = _check_same_shape({"high": h, "low": lo, "close": c, "volume": v})
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)

    after_start = (mod >= start_minute)[:, None]
    finite = np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & np.isfinite(v)
    contributing = finite & (v >= 0) & after_start

    tp = (h + lo + c) / 3.0
    pv = np.where(contributing, tp * v, 0.0)
    vol = np.where(contributing, v, 0.0)

    cum_pv = np.empty_like(pv)
    cum_v = np.empty_like(vol)
    for start, stop in zip(offs[:-1], offs[1:]):
        cum_pv[start:stop] = np.cumsum(pv[start:stop], axis=0)
        cum_v[start:stop] = np.cumsum(vol[start:stop], axis=0)

    ok = (cum_v > 0) & np.isfinite(c) & after_start
    out = np.full(c.shape, np.nan, dtype=np.float64)
    out[ok] = cum_pv[ok] / cum_v[ok]
    return out


def session_range_pct(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    start_minute: int = SESSION_START_MINUTE,
) -> np.ndarray:
    """Per-session range ``(max(high) - min(low)) / last close``, shape ``(n_days, n_sym)``.

    Only rows with ``minute_of_day >= start_minute`` and finite high/low/close are used;
    the divisor is the close of the last such row in the session. NaN if the session has
    no such rows or that close is <= 0.
    """
    h = _as_2d_float64("high", high)
    lo = _as_2d_float64("low", low)
    c = _as_2d_float64("close", close)
    n_rows, n_sym = _check_same_shape({"high": h, "low": lo, "close": c})
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    if n_days == 0:
        return np.full((0, n_sym), np.nan, dtype=np.float64)

    usable = (
        np.isfinite(h) & np.isfinite(lo) & np.isfinite(c) & (mod >= start_minute)[:, None]
    )
    starts = offs[:-1]
    hi_max = np.maximum.reduceat(np.where(usable, h, -np.inf), starts, axis=0)
    lo_min = np.minimum.reduceat(np.where(usable, lo, np.inf), starts, axis=0)

    row_idx = np.broadcast_to(np.arange(n_rows, dtype=np.int64)[:, None], (n_rows, n_sym))
    last_row = np.maximum.reduceat(np.where(usable, row_idx, -1), starts, axis=0)
    has_rows = last_row >= starts[:, None]
    sym_idx = np.broadcast_to(np.arange(n_sym)[None, :], (n_days, n_sym))
    last_close = np.where(has_rows, c[np.where(has_rows, last_row, 0), sym_idx], np.nan)

    ok = has_rows & (last_close > 0)
    out = np.full((n_days, n_sym), np.nan, dtype=np.float64)
    out[ok] = (hi_max[ok] - lo_min[ok]) / last_close[ok]
    return out


def adr_pct(range_pct: np.ndarray, n: int) -> np.ndarray:
    """Causal average daily range: ``out[d] = mean(range_pct[d - n : d])``.

    Only the ``n`` sessions strictly before ``d`` are used; session ``d`` itself never is.
    NaN when ``d < n`` or when any value in the window is NaN (no skipping). Raises
    ``ValueError`` if ``n < 1``.
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    n = int(n)
    r = np.asarray(range_pct, dtype=np.float64)
    if r.ndim < 1:
        raise ValueError("range_pct must be at least 1-D (n_days, ...)")
    n_days = r.shape[0]
    out = np.full(r.shape, np.nan, dtype=np.float64)
    if n_days <= n:
        return out
    # windows[i] = range_pct[i : i + n]; out[d] uses window d - n for d in [n, n_days).
    windows = sliding_window_view(r, n, axis=0)
    out[n:] = windows[: n_days - n].mean(axis=-1)
    return out


def max_excursion(
    close: np.ndarray,
    vwap: np.ndarray,
    adr_day: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    start_minute: int = SESSION_START_MINUTE,
    end_minute: int = SQUARE_OFF_MINUTE,
) -> np.ndarray:
    """Per-session max of ``|close - vwap| / (vwap * adr_day[d])``, shape ``(n_days, n_sym)``.

    Rows used: ``start_minute <= minute_of_day < end_minute`` with finite close and vwap
    and ``vwap > 0``. NaN if there are no such rows or ``adr_day[d]`` is not finite or
    <= 0. This is the measured distribution the k grid is drawn from (CLAUDE.md rule 8).
    """
    c = _as_2d_float64("close", close)
    vw = _as_2d_float64("vwap", vwap)
    adr = _as_2d_float64("adr_day", adr_day)
    n_rows, n_sym = _check_same_shape({"close": c, "vwap": vw})
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    if adr.shape != (n_days, n_sym):
        raise ValueError(
            f"adr_day must have shape (n_days, n_sym)={(n_days, n_sym)}, got {adr.shape}"
        )
    if n_days == 0:
        return np.full((0, n_sym), np.nan, dtype=np.float64)

    adr_ok = np.isfinite(adr) & (adr > 0)
    adr_rows = np.repeat(np.where(adr_ok, adr, np.nan), np.diff(offs), axis=0)
    in_window = ((mod >= start_minute) & (mod < end_minute))[:, None]
    usable = (
        in_window
        & np.isfinite(c)
        & np.isfinite(vw)
        & (vw > 0)
        & np.repeat(adr_ok, np.diff(offs), axis=0)
    )

    ratio = np.full(c.shape, -np.inf, dtype=np.float64)
    ratio[usable] = np.abs(c[usable] - vw[usable]) / (vw[usable] * adr_rows[usable])
    best = np.maximum.reduceat(ratio, offs[:-1], axis=0)

    return np.where(adr_ok & np.isfinite(best), best, np.nan)
