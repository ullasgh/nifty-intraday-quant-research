"""Information coefficient: cross-sectional, per-bar, then aggregated across bars.

D5 of `specs/feature_layer.md` (plus AMENDMENT 1 item 5 and AMENDMENT 2). IC answers
"does the feature rank-order (or linearly order) the forward return cross-sectionally, on a
given bar" -- it is NOT the correlation of the feature and the forward return pooled across
time and symbols. Pooling conflates a cross-sectional signal with a time-series one, and this
program's live tilt candidate is cross-sectional (see `research/lens.py`,
`research/tilt.py`). Every IC in this module is computed one bar at a time, across symbols,
and the per-bar series is then aggregated (mean, and an overlap-aware SE) across bars.

AMENDMENT 2 pins the SE machinery: reuse the moving-block resampler
(`expectancy._block_bootstrap_resampling_2d`, its derived block length
`horizon + BLOCK_LENGTH_EXTRA_BARS`, its within-session block drawing, its skipped-session
accounting) -- NOT `expectancy._compute_bucket_stats`, which is bucket-and-bps specific and
was shown to hand back essentially the naive SE when correlation values were forced through
it. This module builds its own aggregator directly on the resampler: resample the per-bar
cross-sectional IC series, take the mean per replicate, take the standard deviation across
replicates.

The resampler (`_block_bootstrap_resampling_2d`) and its block-length rule
(`_derive_block_length`) are already module-level functions in `expectancy.py`, not nested
inside `_compute_bucket_stats` -- so they are already importable independently, with a single
implementation. No extraction was needed; this module imports them directly rather than
copying them, so the two block bootstraps (bucket-stats SE and IC SE) cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]  # no scipy-stubs package installed

from nifty_quant.research.expectancy import (
    _block_bootstrap_resampling_2d,
    forward_returns,
)

# Cross-sectional gate: a bar with fewer than this many finite (feature, fwd) pairs across
# symbols produces an undefined/unstable correlation and is excluded from the aggregate, NaN
# in the per-bar series. Matches the existing cross-sectional convention used throughout the
# codebase (`features/core.py:cross_sectional_zscore`/`cross_sectional_rank`,
# `features/market.py:breadth`/`dispersion`/`median_pairwise_correlation`), all `min_names=5`.
# Reused here rather than re-derived: it is the same cross-sectional-validity question (how
# many names does a single-bar cross-sectional statistic need to be meaningful), not a new
# threshold in the rule-8 sense.
MIN_NAMES = 5

Method = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class ICResult:
    """Cross-sectional information coefficient, aggregated across bars.

    `mean`/`se` are pinned by `specs/feature_layer.md` AMENDMENT 1 item 5. `per_bar` is an
    additional field (not in the pinned set) carrying the per-bar cross-sectional IC series,
    NaN where a bar had fewer than `MIN_NAMES` valid pairs or zero cross-sectional variance;
    it is exposed because obligation 13 needs the per-bar series to compute the naive SE it
    compares against.
    """

    mean: float
    se: float
    n_bars: int
    method: Method
    horizon: int
    per_bar: np.ndarray


@dataclass(frozen=True)
class ICDecay:
    """IC at each horizon in a grid, plus the fitted half-life. Never a single number."""

    horizons: tuple[int, ...]
    ic: np.ndarray
    se: np.ndarray
    half_life: float


def _per_bar_cross_sectional_ic(
    feature: np.ndarray,
    fwd: np.ndarray,
    *,
    method: Method,
    min_names: int = MIN_NAMES,
) -> np.ndarray:
    """One correlation per bar, across symbols. NaN where undefined."""
    n_rows = feature.shape[0]
    per_bar = np.full(n_rows, np.nan, dtype=np.float64)
    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    for t in range(n_rows):
        x_row = feature[t, :]
        y_row = fwd[t, :]
        valid = np.isfinite(x_row) & np.isfinite(y_row)
        if int(valid.sum()) < min_names:
            continue
        x_valid = x_row[valid]
        y_valid = y_row[valid]
        # Zero cross-sectional variance in either leg makes the correlation undefined
        # (division by zero / NaN from scipy); exclude rather than let it manufacture 0 or NaN
        # silently downstream.
        if np.std(x_valid) == 0.0 or np.std(y_valid) == 0.0:
            continue
        rho, _ = corr_fn(x_valid, y_valid)
        per_bar[t] = rho

    return per_bar


def _overlap_aware_se(
    per_bar_ic: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
    *,
    n_boot: int,
    seed: int,
) -> float:
    """Block-bootstrap SE of the mean of a (possibly NaN-containing) 1-D series.

    Reuses `expectancy._block_bootstrap_resampling_2d` directly -- same derived block length,
    same within-session block drawing, same skipped-session accounting as the overlap_se.md
    machinery used for bps-scaled returns. Only the aggregation on top differs (mean per
    replicate across the correlation series, not a bps mean/CI/t-stat).
    """
    n_rows = per_bar_ic.shape[0]
    values_2d = per_bar_ic.reshape(n_rows, 1).astype(np.float64)

    resampled, _block_indices, _n_sessions_skipped = _block_bootstrap_resampling_2d(
        values_2d, day_offsets, horizon, n_boot, seed
    )
    resampled_flat = resampled.reshape(n_boot, n_rows)

    valid_rows = np.isfinite(resampled_flat).any(axis=1)
    boot_means = np.full(n_boot, np.nan, dtype=np.float64)
    if np.any(valid_rows):
        boot_means[valid_rows] = np.nanmean(resampled_flat[valid_rows], axis=1)

    finite_boot_means = boot_means[np.isfinite(boot_means)]
    if finite_boot_means.size > 1:
        return float(np.std(finite_boot_means, ddof=1))
    return 0.0


def information_coefficient(
    feature: np.ndarray,
    fwd: np.ndarray,
    day_offsets: np.ndarray,
    *,
    horizon: int,
    method: Method = "pearson",
    n_boot: int = 1000,
    seed: int = 0,
) -> ICResult:
    """Cross-sectional IC, per bar, aggregated across bars.

    `horizon` identifies the forward-return window `fwd` was built with, and is required to
    derive the block-bootstrap block length for the overlap-aware SE (`horizon +
    BLOCK_LENGTH_EXTRA_BARS`, see `expectancy._derive_block_length`); it defaults to 1 (no
    assumed overlap) when the caller does not know or does not need to state it.

    `se` MUST come from the overlap-aware block-bootstrap path (AMENDMENT 1 item 5 /
    AMENDMENT 2), unconditionally -- unlike `expectancy._compute_bucket_stats`, this function
    does not special-case `horizon == 1` back to a naive formula; the resampler's own derived
    block length (`1 + BLOCK_LENGTH_EXTRA_BARS = 6`) already captures short-range dependence
    in the per-bar IC series even when the caller states no explicit overlap.
    """
    feature = np.asarray(feature, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)

    per_bar = _per_bar_cross_sectional_ic(feature, fwd, method=method)
    finite = per_bar[np.isfinite(per_bar)]
    mean_ic = float(np.mean(finite)) if finite.size else float("nan")

    se = _overlap_aware_se(per_bar, day_offsets, horizon, n_boot=n_boot, seed=seed)

    return ICResult(
        mean=mean_ic,
        se=se,
        n_bars=int(feature.shape[0]),
        method=method,
        horizon=int(horizon),
        per_bar=per_bar,
    )


def _fit_half_life(horizons: np.ndarray, ic_values: np.ndarray) -> float:
    """Fit `ic(h) = A * 0.5**(h / half_life)` via a linear regression of log|ic| on h.

    `half_life = ln(0.5) / slope`. Needs >= 2 finite, strictly positive IC values (log of a
    non-positive value is undefined); returns NaN if the fit is not identifiable.
    """
    valid = np.isfinite(ic_values) & (ic_values > 0.0)
    if int(valid.sum()) < 2:
        return float("nan")

    log_ic = np.log(ic_values[valid])
    h = horizons[valid]
    slope, _intercept = np.polyfit(h, log_ic, 1)

    if slope >= 0.0:
        # Not decaying (flat or increasing) -- half-life is undefined/infinite.
        return float("inf")

    return float(np.log(0.5) / slope)


def ic_decay(
    feature: np.ndarray,
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    horizons: list[int] | tuple[int, ...],
    n_boot: int = 1000,
    seed: int = 0,
) -> ICDecay:
    """IC at each horizon in `horizons`, plus the fitted half-life. Returns the full grid.

    Forward returns at each horizon are built via `expectancy.forward_returns`
    (session-bounded log return, NaN where the horizon would cross a session boundary --
    never a fixed-stride assumption, CLAUDE.md rule 5), reused rather than re-derived. Each
    horizon's cross-sectional IC uses Pearson correlation: `ic_decay` fits an exponential
    decay curve, and the log-linear half-life fit is only exact under a linear correlation
    measure (Spearman on a rank transform would not reproduce the exact `rho**h` construction
    used in `specs/feature_layer.md` obligation 12's test fixture).
    """
    feature = np.asarray(feature, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)

    horizons_tuple = tuple(int(h) for h in horizons)
    ic_values = np.full(len(horizons_tuple), np.nan, dtype=np.float64)
    se_values = np.full(len(horizons_tuple), np.nan, dtype=np.float64)

    for i, h in enumerate(horizons_tuple):
        fwd_h = forward_returns(close, day_offsets, h).values
        result = information_coefficient(
            feature, fwd_h, day_offsets, horizon=h, method="pearson", n_boot=n_boot, seed=seed
        )
        ic_values[i] = result.mean
        se_values[i] = result.se

    half_life = _fit_half_life(np.asarray(horizons_tuple, dtype=np.float64), ic_values)

    return ICDecay(horizons=horizons_tuple, ic=ic_values, se=se_values, half_life=half_life)
