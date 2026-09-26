"""Per-bar and per-session features for the stress-gated residual liquidity-provision fade.

Spec: ``specs/liqprov_fade.md`` section 1. Conventions follow
``nifty_quant.research.vwap_adr.features``: per-bar stock arrays are ``(n_rows, n_sym)``,
index arrays (NIFTY50, INDIAVIX) are 1-D ``(n_rows,)`` aligned to the same rows, and
per-session arrays are ``(n_days,)`` or ``(n_days, n_sym)``. Session ``d`` covers rows
``day_offsets[d] : day_offsets[d + 1]``; sessions have arbitrary lengths and no fixed
stride is ever assumed. A bar is present iff its ``close`` is finite. Nothing here fills,
back-fills or interpolates any input. All arithmetic is float64 (float32 inputs are
upcast first).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from nifty_quant.research.vwap_adr.features import (
    SESSION_START_MINUTE,
    SQUARE_OFF_MINUTE,
)

__all__ = [
    "index_twap",
    "prior_session_beta",
    "index_dev",
    "residual_stretch",
    "vix_change",
    "session_extreme",
    "rolling_session_quantile",
    "stress_gate",
]


# --------------------------------------------------------------------------- validation


def _as_1d_float64(name: str, arr: np.ndarray, n: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {out.shape[0]}")
    return out


def _as_2d_float64(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, n_sym), got shape {out.shape}")
    return out


def _as_1d_bool(name: str, arr: np.ndarray, n: int) -> np.ndarray:
    raw = np.asarray(arr)
    if raw.ndim != 1 or raw.shape[0] != n:
        raise ValueError(f"{name} must be 1-D of length {n}, got shape {raw.shape}")
    if raw.dtype != np.bool_:
        raise ValueError(f"{name} must have a bool dtype, got {raw.dtype}")
    return raw


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


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _session_id(offs: np.ndarray) -> np.ndarray:
    """Session index of every row, length ``n_rows``."""
    return np.repeat(np.arange(offs.shape[0] - 1, dtype=np.int64), np.diff(offs))


# --------------------------------------------------------------------------- 1.1


def index_twap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    start_minute: int = SESSION_START_MINUTE,
) -> np.ndarray:
    """Session-anchored time-weighted average of the index typical price, ``(n_rows,)``.

    A row contributes iff ``minute_of_day >= start_minute`` and high/low/close are all
    finite. ``twap[t]`` is the mean of ``(h + l + c) / 3`` over contributing rows of
    ``t``'s session up to and including ``t``. NaN if the index bar at ``t`` is absent
    (non-finite close), ``minute_of_day[t] < start_minute``, or nothing has contributed
    yet. The mean resets at every session boundary. Each session is cumulated from zero
    (no global cumsum minus a session base, which loses precision to cancellation).
    """
    c = _as_1d_float64("close", close)
    n_rows = c.shape[0]
    h = _as_1d_float64("high", high, n_rows)
    lo = _as_1d_float64("low", low, n_rows)
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)

    after_start = mod >= start_minute
    contributing = after_start & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    tp = np.where(contributing, (h + lo + c) / 3.0, 0.0)
    cnt = contributing.astype(np.float64)

    cum_tp = np.empty(n_rows, dtype=np.float64)
    cum_n = np.empty(n_rows, dtype=np.float64)
    for start, stop in zip(offs[:-1], offs[1:]):
        cum_tp[start:stop] = np.cumsum(tp[start:stop])
        cum_n[start:stop] = np.cumsum(cnt[start:stop])

    ok = after_start & np.isfinite(c) & (cum_n > 0)
    out = np.full(n_rows, np.nan, dtype=np.float64)
    out[ok] = cum_tp[ok] / cum_n[ok]
    return out


# --------------------------------------------------------------------------- 1.2


def prior_session_beta(
    stock_close: np.ndarray,
    index_close: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    usable: np.ndarray,
    lookback: int = 20,
    start_minute: int = SESSION_START_MINUTE,
) -> np.ndarray:
    """Causal pooled OLS beta of 1-min stock log returns on index log returns, ``(n_days, n_sym)``.

    A pair exists at row ``t`` iff ``t - 1`` is the immediately preceding row of the same
    session, ``minute_of_day[t - 1] >= start_minute``, and all four closes
    (``S[t-1], S[t], I[t-1], I[t]``) are finite and > 0; a gap is never bridged. The pair
    is ``x = ln(I_t / I_{t-1})``, ``y = ln(S_t / S_{t-1})`` and belongs to ``t``'s session.

    ``beta[d]`` pools ``n, Sx, Sy, Sxy, Sxx`` over the ``lookback`` most recent sessions
    ``j < d`` with ``usable[j]`` (session ``d`` itself never), and is
    ``(n Sxy - Sx Sy) / (n Sxx - Sx^2)``. NaN if fewer than ``lookback`` usable prior
    sessions exist, if the symbol has zero pairs in any of them, or if the denominator is
    <= 0. Raises ``ValueError`` if ``lookback < 2``.
    """
    lookback = _check_int("lookback", lookback, 2)
    s = _as_2d_float64("stock_close", stock_close)
    n_rows, n_sym = s.shape
    ic = _as_1d_float64("index_close", index_close, n_rows)
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    use = _as_1d_bool("usable", usable, n_days)

    out = np.full((n_days, n_sym), np.nan, dtype=np.float64)
    if n_days == 0 or n_rows < 2:
        return out

    # Row t has a candidate predecessor iff t is not the first row of its session and
    # the predecessor is at/after start_minute.
    has_prev = np.ones(n_rows, dtype=bool)
    has_prev[offs[:-1]] = False
    has_prev[1:] &= mod[:-1] >= start_minute
    has_prev[0] = False

    i_ok = np.isfinite(ic) & (ic > 0)
    idx_pair = np.zeros(n_rows, dtype=bool)
    idx_pair[1:] = has_prev[1:] & i_ok[1:] & i_ok[:-1]
    lx = np.zeros(n_rows, dtype=np.float64)
    lx[1:][idx_pair[1:]] = np.log(ic[1:][idx_pair[1:]] / ic[:-1][idx_pair[1:]])

    # ly[t] = ln(S_t / S_{t-1}); NaN wherever either close is missing or <= 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(np.where(s > 0, s, np.nan))
    ly = np.zeros((n_rows, n_sym), dtype=np.float64)
    ly[1:] = log_s[1:] - log_s[:-1]
    del log_s
    ok = np.isfinite(ly) & idx_pair[:, None]
    okf = ok.astype(np.float64)
    ly[~ok] = 0.0
    del ok

    # Per-session sums. The index return is shared across symbols, so the x-moments are
    # BLAS vector-matrix products per session instead of full (n_rows, n_sym) temporaries.
    lx2 = lx * lx
    n_d = np.empty((n_days, n_sym), dtype=np.float64)
    sx_d = np.empty((n_days, n_sym), dtype=np.float64)
    sxx_d = np.empty((n_days, n_sym), dtype=np.float64)
    sxy_d = np.empty((n_days, n_sym), dtype=np.float64)
    sy_d = np.empty((n_days, n_sym), dtype=np.float64)
    for d, (start, stop) in enumerate(zip(offs[:-1], offs[1:])):
        o = okf[start:stop]
        y = ly[start:stop]
        n_d[d] = o.sum(axis=0)
        sx_d[d] = lx[start:stop] @ o
        sxx_d[d] = lx2[start:stop] @ o
        sxy_d[d] = lx[start:stop] @ y
        sy_d[d] = y.sum(axis=0)
    del okf, ly

    # Pool over sliding windows of `lookback` consecutive usable sessions.
    u_idx = np.flatnonzero(use)
    m = u_idx.shape[0]
    if m < lookback:
        return out

    def _win(a: np.ndarray, reduce: str = "sum") -> np.ndarray:
        view = sliding_window_view(a[u_idx], lookback, axis=0)  # (m - L + 1, n_sym, L)
        res: np.ndarray = view.min(axis=-1) if reduce == "min" else view.sum(axis=-1)
        return res

    n_min = _win(n_d, "min")
    n_w = _win(n_d).astype(np.float64)
    sx_w = _win(sx_d)
    sy_w = _win(sy_d)
    sxy_w = _win(sxy_d)
    sxx_w = _win(sxx_d)

    num = n_w * sxy_w - sx_w * sy_w
    den = n_w * sxx_w - sx_w * sx_w
    good = (n_min > 0) & (den > 0)
    beta_w = np.full(num.shape, np.nan, dtype=np.float64)
    beta_w[good] = num[good] / den[good]

    # Number of usable sessions strictly before d; window w = k - lookback covers the
    # usable sessions u_idx[k - lookback : k].
    k = np.searchsorted(u_idx, np.arange(n_days), side="left")
    have = k >= lookback
    out[have] = beta_w[k[have] - lookback]
    return out


# --------------------------------------------------------------------------- 1.3


def index_dev(index_close: np.ndarray, twap: np.ndarray) -> np.ndarray:
    """``index_close / twap - 1``, ``(n_rows,)``; NaN where either is non-finite or twap <= 0."""
    ic = _as_1d_float64("index_close", index_close)
    tw = _as_1d_float64("twap", twap, ic.shape[0])
    ok = np.isfinite(ic) & np.isfinite(tw) & (tw > 0)
    out = np.full(ic.shape, np.nan, dtype=np.float64)
    out[ok] = ic[ok] / tw[ok] - 1.0
    return out


# --------------------------------------------------------------------------- 1.4


def residual_stretch(
    close: np.ndarray,
    vwap: np.ndarray,
    idx_dev: np.ndarray,
    beta_day: np.ndarray,
    adr_day: np.ndarray,
    day_offsets: np.ndarray,
) -> np.ndarray:
    """Beta-adjusted stretch below/above VWAP in ADR units, ``(n_rows, n_sym)``.

    ``z[t, s] = ((close / vwap - 1) - beta_day[d, s] * idx_dev[t]) / adr_day[d, s]`` with
    ``d`` the session of ``t``. NaN if any input is non-finite, ``vwap <= 0`` or
    ``adr_day <= 0``.
    """
    c = _as_2d_float64("close", close)
    vw = _as_2d_float64("vwap", vwap)
    if vw.shape != c.shape:
        raise ValueError(f"vwap shape {vw.shape} != close shape {c.shape}")
    n_rows, n_sym = c.shape
    dev = _as_1d_float64("idx_dev", idx_dev, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    beta = _as_2d_float64("beta_day", beta_day)
    adr = _as_2d_float64("adr_day", adr_day)
    for name, a in (("beta_day", beta), ("adr_day", adr)):
        if a.shape != (n_days, n_sym):
            raise ValueError(
                f"{name} must have shape (n_days, n_sym)={(n_days, n_sym)}, got {a.shape}"
            )

    # Invalid per-session / per-row inputs become NaN once at low resolution, so the
    # (n_rows, n_sym) pass is pure arithmetic plus one final finiteness mask.
    counts = np.diff(offs)
    beta_ok = np.where(np.isfinite(beta), beta, np.nan)
    adr_ok = np.where(np.isfinite(adr) & (adr > 0), adr, np.nan)
    dev_ok = np.where(np.isfinite(dev), dev, np.nan)[:, None]
    vw_ok = np.where(np.isfinite(vw) & (vw > 0), vw, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = c / vw_ok
        out -= 1.0
        out -= np.repeat(beta_ok, counts, axis=0) * dev_ok
        out /= np.repeat(adr_ok, counts, axis=0)
    out[~np.isfinite(out)] = np.nan
    return out


# --------------------------------------------------------------------------- 1.5


def vix_change(
    vix_close: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    start_minute: int = SESSION_START_MINUTE,
) -> np.ndarray:
    """``vix_close[t] / prev - 1`` with ``prev`` the previous session's last finite close.

    ``prev`` is taken over any minute of session ``d - 1`` (the 09:15 bar included).
    NaN for session 0, when ``prev`` is missing or <= 0, when the VIX bar at ``t`` is
    absent, or when ``minute_of_day[t] < start_minute``.
    """
    v = _as_1d_float64("vix_close", vix_close)
    n_rows = v.shape[0]
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    out = np.full(n_rows, np.nan, dtype=np.float64)
    if n_days < 2:
        return out

    finite = np.isfinite(v)
    rows = np.arange(n_rows, dtype=np.int64)
    last_row = np.maximum.reduceat(np.where(finite, rows, -1), offs[:-1])
    last_close = np.where(last_row >= offs[:-1], v[np.maximum(last_row, 0)], np.nan)

    prev_day = np.full(n_days, np.nan, dtype=np.float64)
    prev_day[1:] = last_close[:-1]
    prev = np.repeat(prev_day, np.diff(offs))

    ok = finite & (mod >= start_minute) & np.isfinite(prev) & (prev > 0)
    out[ok] = v[ok] / prev[ok] - 1.0
    return out


# --------------------------------------------------------------------------- 1.6


def session_extreme(
    x: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    how: str,
    start_minute: int = SESSION_START_MINUTE,
    end_minute: int = SQUARE_OFF_MINUTE,
) -> np.ndarray:
    """Per-session max or min of ``x`` over ``start_minute <= minute < end_minute``.

    Only finite values are used. NaN when a session has no such rows. Accepts
    ``(n_rows,)`` or ``(n_rows, n_sym)`` and returns ``(n_days,)`` or ``(n_days, n_sym)``.
    Raises ``ValueError`` unless ``how`` is ``"max"`` or ``"min"``.
    """
    if how not in ("max", "min"):
        raise ValueError(f"how must be 'max' or 'min', got {how!r}")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {a.shape}")
    n_rows = a.shape[0]
    mod = _validate_minute_of_day(minute_of_day, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    out_shape = (n_days,) + a.shape[1:]
    if n_days == 0:
        return np.full(out_shape, np.nan, dtype=np.float64)

    window = (mod >= start_minute) & (mod < end_minute)
    if a.ndim == 2:
        window = window[:, None]
    mask = window & np.isfinite(a)
    starts = offs[:-1]
    if how == "max":
        best = np.maximum.reduceat(np.where(mask, a, -np.inf), starts, axis=0)
    else:
        best = np.minimum.reduceat(np.where(mask, a, np.inf), starts, axis=0)
    has = np.add.reduceat(mask.astype(np.int64), starts, axis=0) > 0
    return np.where(has, best, np.nan)


# --------------------------------------------------------------------------- 1.7


def rolling_session_quantile(
    per_session: np.ndarray,
    usable: np.ndarray,
    q: float,
    window: int = 250,
    min_sessions: int = 120,
) -> np.ndarray:
    """Causal rolling quantile over prior usable sessions, ``(n_days,)``.

    ``out[d] = np.quantile(vals, q)`` (numpy default "linear" method), where ``vals`` are
    the most recent ``window`` values ``per_session[j]`` with ``j < d``, ``usable[j]`` and
    ``per_session[j]`` finite. NaN if fewer than ``min_sessions`` such values exist.
    Raises ``ValueError`` unless ``0 < q < 1`` and ``window >= min_sessions >= 1``.
    """
    qf = float(q)
    if not (0.0 < qf < 1.0):
        raise ValueError(f"q must satisfy 0 < q < 1, got {q!r}")
    min_sessions = _check_int("min_sessions", min_sessions, 1)
    window = _check_int("window", window, 1)
    if window < min_sessions:
        raise ValueError(f"window ({window}) must be >= min_sessions ({min_sessions})")
    vals_all = _as_1d_float64("per_session", per_session)
    n_days = vals_all.shape[0]
    use = _as_1d_bool("usable", usable, n_days)

    valid_idx = np.flatnonzero(use & np.isfinite(vals_all))
    valid_vals = vals_all[valid_idx]
    # k[d] = number of valid sessions strictly before d.
    k = np.searchsorted(valid_idx, np.arange(n_days), side="left")
    out = np.full(n_days, np.nan, dtype=np.float64)
    last_k = -1
    last_val = np.nan
    for d in np.flatnonzero(k >= min_sessions):
        kd = int(k[d])
        if kd != last_k:  # the slice only changes after a valid session
            last_val = float(np.quantile(valid_vals[max(0, kd - window) : kd], qf))
            last_k = kd
        out[d] = last_val
    return out


# --------------------------------------------------------------------------- 1.8


def stress_gate(
    vix_chg: np.ndarray,
    idx_dev_adr: np.ndarray,
    v_day: np.ndarray,
    m_day: np.ndarray,
    day_offsets: np.ndarray,
) -> np.ndarray:
    """Market-liquidation gate, bool ``(n_rows,)``.

    True at row ``t`` of session ``d`` iff ``vix_chg[t] >= v_day[d]`` and
    ``idx_dev_adr[t] <= m_day[d]`` with all four values finite; False otherwise. The
    thresholds are measured, causal rolling quantiles set by the runner (CLAUDE.md
    rule 8); see spec section 1.8.
    """
    vc = _as_1d_float64("vix_chg", vix_chg)
    n_rows = vc.shape[0]
    da = _as_1d_float64("idx_dev_adr", idx_dev_adr, n_rows)
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.shape[0] - 1
    v = _as_1d_float64("v_day", v_day, n_days)
    m = _as_1d_float64("m_day", m_day, n_days)

    counts = np.diff(offs)
    v_r = np.repeat(v, counts)
    m_r = np.repeat(m, counts)
    finite = np.isfinite(vc) & np.isfinite(da) & np.isfinite(v_r) & np.isfinite(m_r)
    with np.errstate(invalid="ignore"):
        gate: np.ndarray = finite & (vc >= v_r) & (da <= m_r)
    return gate
