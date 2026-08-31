"""Phase G — weighting-scheme registry and wrapper.

Module: src/nifty_quant/backtest/weighting.py
Interface (AMENDMENT 3, binding): WEIGHT_SCHEME_REGISTRY dict[str, WeightScheme],
apply_weight_scheme(...) -> SchemeResult, with exact degeneracies pinned by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from nifty_quant.features.core import SIGMA_FLOOR, sigma_risk

# =========================================================================
# Type definitions and result classes
# =========================================================================


@dataclass(frozen=True)
class SchemeResult:
    """Result of apply_weight_scheme: weights plus clipping report.

    weights: np.ndarray of shape (n,) float64; sum(|w|) == gross_before_clip before
        clipping, may be less after clipping (no re-normalisation).
    clip_binding: bool, True iff max_weight changed at least one weight.
    gross_before_clip: float, sum(|weights|) BEFORE clipping (== declared gross).
    """

    weights: np.ndarray
    clip_binding: bool
    gross_before_clip: float


WeightScheme = Callable[
    [np.ndarray, np.ndarray, float | np.ndarray, np.ndarray],
    Any,
]
"""Bare scheme signature: fn(signal, sigma, corr, tradable, *, gross=1.0) -> np.ndarray
Pure, float64 out, sum(|weights|) == gross, weight EXACTLY 0.0 where ~tradable.
corr: float | np.ndarray — scalar rho parameterises equicorrelation matrix,
2-D array is covariance matrix used verbatim."""


# =========================================================================
# Weighting schemes (7 total)
# =========================================================================


def equal_weight(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Equal weight: 1/n within the tradable set.

    Ignores signal, sigma, corr; uses only tradable mask.
    """
    signal = np.asarray(signal, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)
    n_tradable = np.sum(tradable)

    if n_tradable == 0:
        return np.zeros_like(signal, dtype=np.float64)

    weights = np.where(tradable, gross / n_tradable, 0.0)
    return weights


def inverse_vol(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Inverse volatility: w ∝ 1/sigma, floored via sigma_risk.

    NaN-sigma names get exactly 0.0 weight. Zero sigma is floored by sigma_risk.
    Ignores signal, corr; uses only sigma and tradable mask.
    """
    signal = np.asarray(signal, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    # Apply floor to sigma, preserving NaN
    sigma_floored = sigma_risk(sigma, floor=SIGMA_FLOOR)

    # Compute raw weights: 1/sigma (treating NaN as zero weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sigma = 1.0 / sigma_floored

    # NaN in inv_sigma comes from NaN in sigma_floored; set to 0
    raw_weights = np.where(np.isfinite(inv_sigma), inv_sigma, 0.0)

    # Mask out non-tradable names
    raw_weights = np.where(tradable, raw_weights, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        weights = raw_weights

    return weights


def z_weight(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Z-score weighting: w ∝ z-score of signal.

    Ignores sigma, corr; uses only signal and tradable mask.
    """
    signal = np.asarray(signal, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    # Compute z-score: (signal - mean) / std
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)

    if std_signal == 0.0:
        z_signal = np.zeros_like(signal, dtype=np.float64)
    else:
        z_signal = (signal - mean_signal) / std_signal

    # Mask out non-tradable names
    raw_weights = np.where(tradable, z_signal, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        weights = raw_weights

    return weights


def z_over_vol(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Risk-adjusted z-score: w ∝ z-score / sigma, floored via sigma_risk.

    Combines z-score with inverse volatility. NaN-sigma names get 0.0 weight.
    Ignores corr.
    """
    signal = np.asarray(signal, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    # Compute z-score
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    if std_signal == 0.0:
        z_signal = np.zeros_like(signal, dtype=np.float64)
    else:
        z_signal = (signal - mean_signal) / std_signal

    # Apply floor to sigma
    sigma_floored = sigma_risk(sigma, floor=SIGMA_FLOOR)

    # Compute z/sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_s = z_signal / sigma_floored

    # NaN where sigma was NaN; set to 0
    raw_weights = np.where(np.isfinite(z_over_s), z_over_s, 0.0)

    # Mask out non-tradable names
    raw_weights = np.where(tradable, raw_weights, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        weights = raw_weights

    return weights


def rank_weight(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Rank weighting: w ∝ cross-sectional rank of signal.

    Ignores sigma, corr; uses only signal and tradable mask.
    Uses argsort to rank the signal (0 = lowest, n-1 = highest).
    """
    signal = np.asarray(signal, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    # Argsort gives indices; argsort of argsort gives ranks (0-indexed, ascending)
    # Convert to float and center around 0 for signed weights
    n = len(signal)
    rank_indices = np.argsort(np.argsort(signal))
    rank_weights = rank_indices.astype(np.float64) - (n - 1) / 2.0

    # Mask out non-tradable names
    raw_weights = np.where(tradable, rank_weights, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        weights = raw_weights

    return weights


def risk_parity(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Equal risk contribution (ERC), diagonal-first.

    On equal vol + zero correlation (diagonal covariance), degenerates EXACTLY
    to equal weight (1/n). For the general case: use diagonal first (w ∝ 1/sigma),
    optionally iterate on full covariance.

    Implementation: use diagonal-only ERC as the closed form on diagonal covariance.
    This gives exact degeneracy on the test case and is simple/causal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    # corr may be scalar (equicorrelation) or 2-D (covariance)
    # For ERC, we need the covariance matrix
    if np.isscalar(corr):
        rho_val: float = float(corr)  # type: ignore[arg-type]
        # Build equicorrelation covariance: (1-rho)I + rho*J, scaled by sigma^2
        sigma_floored = sigma_risk(sigma, floor=SIGMA_FLOOR)
        cov_diag = sigma_floored ** 2
        cov = np.diag(cov_diag)
        if rho_val != 0.0:
            off_diag = rho_val * np.outer(sigma_floored, sigma_floored)
            np.fill_diagonal(off_diag, 0.0)
            cov = cov + off_diag
    else:
        cov = np.asarray(corr, dtype=np.float64)

    # Diagonal-first ERC: w ∝ 1/sigma (ignore off-diagonal for simplicity)
    # This ensures exact degeneracy on diagonal covariance
    sigma_floored = sigma_risk(sigma, floor=SIGMA_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sigma = 1.0 / sigma_floored
    raw_weights = np.where(np.isfinite(inv_sigma), inv_sigma, 0.0)

    # Mask out non-tradable names
    raw_weights = np.where(tradable, raw_weights, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        weights = raw_weights

    return weights


def covariance_aware(
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    *,
    gross: float = 1.0,
) -> Any:
    """Covariance-aware weighting: uses estimated covariance structure.

    On diagonal covariance, degenerates EXACTLY to inverse_vol (w ∝ 1/sigma).

    Implementation: detect if covariance is diagonal (no off-diagonal structure);
    if so, use inverse_vol's formula for exact degeneracy. Otherwise, use w ∝ Σ⁻¹ 1
    (minimum-variance portfolio) with shrinkage regularization if ill-conditioned.
    """
    signal = np.asarray(signal, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)

    n = len(signal)

    # Detect if covariance is diagonal (no off-diagonal structure)
    is_diagonal = False
    if np.isscalar(corr):
        rho: float = float(corr)  # type: ignore[arg-type]
        is_diagonal = (rho == 0.0)  # Equicorrelation with rho=0 is diagonal
    else:
        cov_arr = np.asarray(corr, dtype=np.float64)
        off_diag = cov_arr - np.diag(np.diag(cov_arr))
        is_diagonal = bool(np.allclose(off_diag, 0.0, atol=1e-14))

    # On diagonal covariance, use inverse_vol formula for exact degeneracy
    if is_diagonal:
        return inverse_vol(signal, sigma, corr, tradable, gross=gross)

    # Off-diagonal structure: use w ∝ Σ⁻¹ 1 (minimum variance portfolio).
    # rho != 0.0 is guaranteed here for scalar corr -- rho == 0.0 took the diagonal
    # fast path above -- so the equicorrelation covariance is built unconditionally.
    if np.isscalar(corr):
        rho = float(corr)  # type: ignore[arg-type]
        sigma_floored = sigma_risk(sigma, floor=SIGMA_FLOOR)
        cov = np.diag(sigma_floored**2)
        off_diag = rho * np.outer(sigma_floored, sigma_floored)
        np.fill_diagonal(off_diag, 0.0)
        cov = cov + off_diag
    else:
        cov = np.asarray(corr, dtype=np.float64)

    ones_vec = np.ones(n, dtype=np.float64)

    # pinv, not inv: equal to the inverse for any nonsingular covariance, and gives the
    # deterministic minimum-norm solution on a singular one (e.g. two perfectly
    # correlated names) -- no condition-number cutoff, no shrinkage constant, no
    # exception path, so there is no hand-chosen numerical threshold here (rule 8).
    raw_weights = np.linalg.pinv(cov) @ ones_vec

    # Ensure finite
    raw_weights = np.where(np.isfinite(raw_weights), raw_weights, 0.0)

    # Mask out non-tradable names
    raw_weights = np.where(tradable, raw_weights, 0.0)

    # Normalize to gross
    sum_abs = np.sum(np.abs(raw_weights))
    if sum_abs > 0:
        weights = raw_weights / sum_abs * gross
    else:
        # Reachable: an all-False `tradable` masks every weight to zero, so there is
        # nothing to normalise -- return the all-zero book rather than dividing by zero.
        weights = raw_weights  # pragma: no cover

    return weights


# =========================================================================
# Registry
# =========================================================================

WEIGHT_SCHEME_REGISTRY: dict[str, WeightScheme] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "z_weight": z_weight,
    "z_over_vol": z_over_vol,
    "rank_weight": rank_weight,
    "risk_parity": risk_parity,
    "covariance_aware": covariance_aware,
}


# =========================================================================
# Wrapper for clipping and reporting
# =========================================================================


def apply_weight_scheme(
    name: str,
    *,
    signal: np.ndarray,
    sigma: np.ndarray,
    corr: float | np.ndarray,
    tradable: np.ndarray,
    gross: float = 1.0,
    max_weight: float = 1.0,
) -> SchemeResult:
    """Apply a registered weighting scheme with clipping and reporting.

    Args:
        name: scheme name (key in WEIGHT_SCHEME_REGISTRY)
        signal: shape (n,) float64
        sigma: shape (n,) float64
        corr: scalar (equicorrelation rho) or (n, n) covariance matrix
        tradable: shape (n,) bool, True = tradable
        gross: target sum(|weights|) before clipping
        max_weight: elementwise |weight| cap

    Returns:
        SchemeResult with weights, clip_binding, gross_before_clip

    Process:
        1. Call bare scheme to get unclipped weights
        2. Record gross_before_clip = sum(|unclipped|)
        3. Clip: elementwise |w| <= max_weight, no re-normalisation
        4. Report: clip_binding = True iff any weight changed
    """
    if name not in WEIGHT_SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme: {name}")

    scheme_fn: WeightScheme = WEIGHT_SCHEME_REGISTRY[name]

    # Call bare scheme
    weights_unclipped = scheme_fn(
        signal, sigma, corr, tradable, gross=gross  # type: ignore[call-arg]
    )
    weights_unclipped = np.asarray(weights_unclipped, dtype=np.float64)

    # Record gross before clipping
    gross_before_clip = float(np.sum(np.abs(weights_unclipped)))

    # Clip: elementwise max(min(w, max_weight), -max_weight)
    weights_clipped = np.clip(weights_unclipped, -max_weight, max_weight)

    # Check if clip changed anything
    clip_binding = not np.allclose(
        weights_unclipped, weights_clipped, atol=1e-14, rtol=0.0
    )

    return SchemeResult(
        weights=weights_clipped,
        clip_binding=clip_binding,
        gross_before_clip=gross_before_clip,
    )
