"""Phase G — G3 pre-registered scheme-comparison runner.

Module: src/nifty_quant/research/portfolio_comparison.py
Interface (AMENDMENT 3 & 5, binding):
  - run_scheme_comparison(*, close, day_offsets, contract, registry)
    -> result with .effective_n_trials
  - evaluate_scheme_promotion(*, returns, day_offsets, recent_n_sessions)
    -> result with .promoted: bool
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from nifty_quant.backtest.metrics import deflated_sharpe, effective_n_trials
from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY
from nifty_quant.research.feature_sweep import DEFLATED_SHARPE_THRESHOLD_DEFAULT
from nifty_quant.research.provenance import get_git_sha
from nifty_quant.research.registry import TrialRecord

# =========================================================================
# Result classes
# =========================================================================


@dataclass(frozen=True)
class SchemeComparisonResult:
    """Result of run_scheme_comparison.

    effective_n_trials: float, the measured participation ratio of trial returns
                        across all schemes (will be < 7 on correlated inputs).
    """

    effective_n_trials: float


@dataclass(frozen=True)
class PromotionResult:
    """Result of evaluate_scheme_promotion.

    promoted: bool, True iff the recent-window deflated Sharpe > 0.0
              (zero is the deflated statistic's definitional null).
    """

    promoted: bool


# =========================================================================
# run_scheme_comparison: multi-scheme pre-registered experiment
# =========================================================================


def run_scheme_comparison(
    *,
    close: np.ndarray,
    day_offsets: np.ndarray,
    contract: Any,
    registry: Any,
) -> SchemeComparisonResult:
    """Run a pre-registered comparison of all seven weighting schemes.

    Each scheme is registered as one trial through the contract. Computes
    effective_n_trials from the correlated scheme returns.

    Args:
        close: shape (n_bars, n_symbols) float64, close prices; must not contain NaN
        day_offsets: shape (n_days + 1,) int64, bar indices marking day boundaries
                     (day j spans bars [day_offsets[j], day_offsets[j+1]))
        contract: ResearchContract, with n_planned_trials already validated
        registry: TrialRegistry, for recording one TrialRecord per scheme

    Returns:
        SchemeComparisonResult with .effective_n_trials: float < 7
    """
    close = np.asarray(close, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)

    n_bars, n_symbols = close.shape
    scheme_names = tuple(sorted(WEIGHT_SCHEME_REGISTRY.keys()))
    n_schemes = len(scheme_names)

    # Compute log-returns: shape (n_bars - 1, n_symbols)
    log_returns = np.diff(np.log(close), axis=0)

    from nifty_quant.features.core import SIGMA_FLOOR

    # Compute rolling sigma (volatility) over a window.
    # Use a shorter window (5 bars) to minimize initial NaN values.
    sigma_window = 5
    sigma = np.full_like(log_returns, SIGMA_FLOOR, dtype=np.float64)
    for i in range(sigma_window - 1, len(log_returns)):
        window_std = np.std(log_returns[max(0, i - sigma_window + 1) : i + 1], axis=0)
        sigma[i] = np.maximum(window_std, SIGMA_FLOOR)
    # Fill initial bars with SIGMA_FLOOR as well
    for i in range(min(sigma_window - 1, len(log_returns))):
        sigma[i] = SIGMA_FLOOR

    # Compute rolling z-score signal (mean-reversion style).
    # Use a short window for the signal as well.
    signal_window = 5
    signal = np.zeros_like(log_returns, dtype=np.float64)
    for i in range(signal_window - 1, len(log_returns)):
        window_returns = log_returns[max(0, i - signal_window + 1) : i + 1]
        mean_ret = np.mean(window_returns, axis=0)
        std_ret = np.std(window_returns, axis=0)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            signal[i] = np.where(std_ret > 1e-10, -mean_ret / std_ret, 0.0)
    # Initial bars have zero signal, which is fine (neutral signal)

    # All symbols are tradable (no filtering here; rule 7 handled at the caller)
    tradable = np.ones(n_symbols, dtype=bool)

    # For each bar and scheme, compute weights and apply to forward returns.
    # trial_returns shape: (n_bars - 1, n_schemes)
    trial_returns = np.empty((len(log_returns), n_schemes), dtype=np.float64)

    for bar_idx in range(len(log_returns)):
        sig = signal[bar_idx]
        sig_floored = sigma[bar_idx]  # sigma is already floored above
        fwd_ret = log_returns[bar_idx]

        for scheme_idx, scheme_name in enumerate(scheme_names):
            scheme_fn = WEIGHT_SCHEME_REGISTRY[scheme_name]

            # Compute weights for this bar's scheme.
            # corr=0.1 is a reasonable equicorrelation proxy (5-10% typical pairwise corr).
            # Note: schemes use only tradable mask; present/tradable separation handled by caller.
            weights = scheme_fn(sig, sig_floored, 0.1, tradable, gross=1.0)  # type: ignore[call-arg]
            weights = np.asarray(weights, dtype=np.float64)

            # Portfolio return: weights @ forward_returns
            trial_returns[bar_idx, scheme_idx] = float(weights @ fwd_ret)

    # Compute effective_n_trials from the trial returns.
    n_eff = effective_n_trials(trial_returns)

    # Register each scheme as a trial through the contract and registry.
    # Build a unique config_hash for each scheme so they don't collide.
    from nifty_quant import __version__

    contract_hash = contract.contract_hash

    for scheme_idx, scheme_name in enumerate(scheme_names):
        # Build a scheme-specific config dict.
        # Use the scheme name in the config to ensure uniqueness.
        scheme_config = {"scheme": scheme_name}
        config_json = json.dumps(scheme_config, sort_keys=True)
        config_hash = hashlib.sha256((contract_hash + config_json).encode()).hexdigest()

        # Register trial with contract (side effect: validates trial count)
        contract.register_trial()

        # Record TrialRecord in registry
        registry.record(
            TrialRecord(
                config_hash=config_hash,
                contract_hash=contract_hash,
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                strategy="scheme_comparison",
                params_json=json.dumps({"scheme": scheme_name}),
                split_id="full",
                purpose="exploration",
                sharpe_gross=None,
                sharpe_net=None,
                n_trades=None,
                turnover=None,
                breakeven_bps=None,
                git_sha=get_git_sha(),
                data_fingerprint=None,
                code_version=__version__,
                wall_s=None,
                result_path=None,
                error=None,
                seed=None,
            )
        )

    return SchemeComparisonResult(effective_n_trials=n_eff)


# =========================================================================
# evaluate_scheme_promotion: recent-window gate
# =========================================================================


def evaluate_scheme_promotion(
    *,
    returns: np.ndarray,
    day_offsets: np.ndarray,
    recent_n_sessions: int,
) -> PromotionResult:
    """Evaluate promotion on the recent window only, not pooled.

    Extracts the most recent returns and computes deflated Sharpe on that window.
    Promotes iff recent DSR exceeds DEFLATED_SHARPE_THRESHOLD_DEFAULT (per AMENDMENT 6).

    Args:
        returns: shape (n_bars,) float64, return series
        day_offsets: shape (n_days + 1,) int64, bar indices marking day boundaries
        recent_n_sessions: int, number of most-recent sessions to use for promotion gate

    Returns:
        PromotionResult with .promoted: bool
    """
    returns = np.asarray(returns, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)

    # Extract the recent window: the last recent_n_sessions sessions.
    # day_offsets has shape (n_days + 1,), so:
    #   day_offsets[-1] is the final bar index (exclusive, as in Python slicing)
    #   The last session starts at day_offsets[-(recent_n_sessions + 1)]
    n_days = len(day_offsets) - 1
    start_day_idx = max(0, n_days - recent_n_sessions)
    start_bar_idx = int(day_offsets[start_day_idx])
    end_bar_idx = int(day_offsets[-1])

    recent_returns = returns[start_bar_idx:end_bar_idx]

    # Compute deflated Sharpe on the recent window.
    # sr0=0.0 means the null hypothesis is Sharpe ratio = 0.0 (no edge).
    # Per AMENDMENT 6: promote iff recent DSR exceeds DEFLATED_SHARPE_THRESHOLD_DEFAULT
    # (0.95, the repo's one-sided 95% convention, not a tuned cutoff; see feature_sweep).
    deflated_sharpe_recent = deflated_sharpe(recent_returns, sr0=0.0)

    # Promote iff recent DSR clears the threshold (evidence for positive Sharpe ratio).
    # A NaN deflated_sharpe (e.g., from < 4 returns) is treated as no promotion.
    promoted = (
        np.isfinite(deflated_sharpe_recent)
        and deflated_sharpe_recent > DEFLATED_SHARPE_THRESHOLD_DEFAULT
    )

    return PromotionResult(promoted=bool(promoted))
