"""Shared trailing-liquidity (ADV) computation for nifty_quant.

Lives in ``data/`` because both ``research/lens.py`` (liquidity-decile
bucketing) and ``universe/pit.py`` (point-in-time eligibility) need the same
trailing-ADV statistic, and ``universe`` is the lower layer -- it must not
import from ``research``. This module is the single shared home so a third
copy of the same computation never appears (the repo has already lost a day
to two same-looking liquidity statistics that were not, in fact, the same
statistic).
"""

from __future__ import annotations

import numpy as np

from nifty_quant.data.panel import Panel


def compute_prior_adv(panel: Panel) -> np.ndarray:
    """Trailing 20-session, strictly-prior rupee ADV (close*volume turnover),
    broadcast to bar level; shape (n_rows, n_symbols) float64.

    Session 0 is entirely NaN (no prior session exists). Liquidity is rupee
    turnover, not raw share count, matching data/validate.py's tradable_mask
    convention (per-session nansum, then trailing-20-session strictly-prior
    nanmean) -- EXCEPT for one deliberate difference: an entirely-absent
    session (a symbol with zero finite bars that session) yields NaN for that
    session's turnover total here, never 0.0. `np.nansum` over an all-NaN slice
    silently returns 0.0, which validate.py's tradable_mask correctly treats as
    "0 ADV -> not tradable" for ITS purpose; for THIS purpose (liquidity-decile
    bucketing, and point-in-time eligibility) it would instead misclassify a
    non-trading symbol as the most illiquid possible name, dropping it into
    decile 0 -- a rule-6 violation (NaN means "no bar", never zero). Session
    boundaries always come from `panel.day_offsets`, never a fixed bars-per-
    session stride (Muhurat/shortened sessions vary).
    """
    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    day_offsets = panel.day_offsets

    n_sessions = len(day_offsets) - 1
    n_symbols = close.shape[1]

    day_value = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        day_slice = slice(start, end)
        session_turnover = close[day_slice] * volume[day_slice]
        has_any = np.any(np.isfinite(session_turnover), axis=0)
        # Unlike data/validate.py's tradable_mask, an entirely absent session
        # must remain NaN here so it cannot be classified into liquidity decile 0.
        day_value[session_idx, :] = np.where(
            has_any, np.nansum(session_turnover, axis=0), np.nan
        )

    adv = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(1, n_sessions):
        lookback_start = max(0, session_idx - 20)
        adv[session_idx, :] = np.nanmean(
            day_value[lookback_start:session_idx, :], axis=0
        )

    prior_adv = np.full(close.shape, np.nan, dtype=np.float64)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        prior_adv[start:end, :] = adv[session_idx, :]

    return prior_adv


def compute_prior_adv_by_session(panel: Panel) -> np.ndarray:
    """Session-level view of `compute_prior_adv`: shape (n_sessions, n_symbols).

    `compute_prior_adv` broadcasts one value per session to every bar of that
    session; this returns exactly that one value per session, by reading the
    first row of each session off the bar-level array. Avoids a second
    trailing-ADV implementation for callers (e.g. point-in-time eligibility)
    that need a per-session rather than per-bar shape.
    """
    prior_adv = compute_prior_adv(panel)
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1
    first_rows = day_offsets[:n_sessions]
    return prior_adv[first_rows, :]
