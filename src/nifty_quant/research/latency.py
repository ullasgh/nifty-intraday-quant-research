"""Latency profile: does the edge survive acting on the signal late?

`research.lens.Lens.verdict`'s criterion 5 asks whether the conditional-expectancy
spread a feature produces at decision lag 0 (act immediately) is retained at lags
1 and 2 (act one/two bars late, i.e. on stale information). Nothing in the repo
produced that input before this module -- both committed H2 and H3 verdicts record
criterion 5 as NOT_EVALUATED for exactly that reason.

`cli.py`'s existing latency machinery computes SHARPES for plugin strategies, which
is the wrong quantity (`verdict()` wants bucket-spread EDGES in bps) for the wrong
scope (plugin strategies only, not every feature/hypothesis this criterion should
gate). This module fills that gap directly: for each lag, shift the feature forward
in a session-aware, causal way and recompute the same conditional-expectancy spread
`research.expectancy.conditional_expectancy` already produces for lag 0.

Per CLAUDE.md rule 8, the RETENTION THRESHOLD this profile is judged against
(`lens.py`'s `0.5`, itself marked UNCALIBRATED) is out of scope here: this module's
job is to PRODUCE the profile, not to judge it.
"""

from __future__ import annotations

import numpy as np

from nifty_quant.guards import causal, check_day_offsets
from nifty_quant.research import expectancy

# `features.core.cross_sectional_rank`'s own `min_names` floor: below this many
# symbols with finite data in a row, it returns an all-NaN row rather than a rank.
# `conditional_expectancy` then sees no bucketed observations and reports
# `spread_bps=0.0` -- a real-looking "no edge" that is actually "could not compute
# an edge at all". This has fooled the repo three times already (see the module
# docstring in `research/expectancy.py` / CLAUDE.md rule 8); raise instead of
# silently returning zeros.
_MIN_NAMES_CROSS_SECTIONAL_RANK = 5


@causal(row_arg="x")
def session_aware_shift(x: np.ndarray, k: int, day_offsets: np.ndarray) -> np.ndarray:
    """Shift `x` forward by `k` bars, session-aware and causal.

    `out[t] = x[t - k]` within the session containing `t`; `out[t]` is NaN for the
    first `k` rows of every session (no valid lagged value exists there -- never
    forward-filled, per CLAUDE.md rule 6) and NEVER pulls a value from the
    preceding session (rule 5: sessions vary in length -- Muhurat 60 bars,
    disaster-recovery 105 bars -- so boundaries always come from `day_offsets`,
    never a fixed stride).

    This models "the decision at bar t uses the feature value observed k bars
    ago" -- i.e. acting on a stale signal.
    """
    x64 = np.asarray(x, dtype=np.float64)
    if x64.ndim != 2:
        raise ValueError("x must be 2-D (n_rows, n_symbols)")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    n_rows = x64.shape[0]
    offsets = np.asarray(day_offsets, dtype=np.int64)
    check_day_offsets(offsets, n_rows)

    out = np.full_like(x64, np.nan, dtype=np.float64)
    if k == 0:
        out[:, :] = x64
        return out

    starts = offsets[:-1]
    ends = offsets[1:]
    for start, end in zip(starts.tolist(), ends.tolist()):
        session_len = end - start
        if session_len > k:
            out[start + k : end] = x64[start : end - k]
    return out


def latency_profile(
    feature: np.ndarray,
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    horizon: int,
    lags: tuple[int, ...] = (0, 1, 2),
    n_buckets: int = 5,
    method: str = "cross_sectional_rank",
    seed: int = 0,
) -> dict[int, float]:
    """Conditional-expectancy bucket spread (bps) at each decision lag.

    For every `k` in `lags`, the SIGNAL is shifted forward by `k` bars via
    `session_aware_shift` -- the decision at bar t uses the feature value from bar
    t-k, i.e. acts on information that is `k` bars stale -- and the resulting
    conditional bucket spread is computed with
    `research.expectancy.conditional_expectancy` against the SAME `horizon`-bar
    forward return used at lag 0. The forward return itself never shifts: only the
    signal used to bucket it does.

    Returns `{k: spread_bps}` for every requested lag, the exact shape
    `lens.Lens.verdict(latency_profile=...)` consumes for its criterion 5.

    Raises
    ------
    ValueError
        If `feature`/`close` are not 2-D, their shapes disagree, any lag is
        negative, or (for `method="cross_sectional_rank"`, the default) there are
        fewer than `_MIN_NAMES_CROSS_SECTIONAL_RANK` symbol columns -- below that
        floor `cross_sectional_rank` silently returns all-NaN rows rather than
        raising, which would otherwise read downstream as a real "no edge" 0.0.
    """
    feature64 = np.asarray(feature, dtype=np.float64)
    close64 = np.asarray(close, dtype=np.float64)
    if feature64.ndim != 2:
        raise ValueError("feature must be 2-D (n_rows, n_symbols)")
    if close64.ndim != 2:
        raise ValueError("close must be 2-D (n_rows, n_symbols)")
    if feature64.shape != close64.shape:
        raise ValueError(
            f"feature shape {feature64.shape} must match close shape {close64.shape}"
        )

    n_rows, n_symbols = feature64.shape
    offsets = np.asarray(day_offsets, dtype=np.int64)
    check_day_offsets(offsets, n_rows)

    if method == "cross_sectional_rank" and n_symbols < _MIN_NAMES_CROSS_SECTIONAL_RANK:
        raise ValueError(
            "latency_profile: method='cross_sectional_rank' requires >= "
            f"{_MIN_NAMES_CROSS_SECTIONAL_RANK} symbol columns, got {n_symbols}. "
            "Below this floor cross_sectional_rank returns an all-NaN row rather "
            "than a rank, which conditional_expectancy would report as a "
            "real-looking 0.0 spread; raising instead of returning a silent zero."
        )

    for k in lags:
        if k < 0:
            raise ValueError(f"every lag must be >= 0, got {k}")

    fwd = expectancy.forward_returns(close64, offsets, horizon)

    profile: dict[int, float] = {}
    for k in lags:
        shifted_feature = session_aware_shift(feature64, k, offsets)
        table = expectancy.conditional_expectancy(
            shifted_feature,
            fwd,
            offsets,
            n_buckets=n_buckets,
            method=method,
            seed=seed,
            feature_name=f"latency_lag_{k}",
        )
        profile[int(k)] = float(table.spread_bps)

    return profile
