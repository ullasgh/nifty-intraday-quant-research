"""Performance and overfitting metrics for an intraday NSE equity backtest.

All computations use float64.  The trading calendar constant 252 is only used for
annualization and never for indexing or stride assumptions.

Deflated Sharpe (DSR) must be computed on a POOLED out-of-sample return series across all
folds/periods, never per-split — a single ~1-year split has T~250 observations and lacks the
statistical power to establish significance for any realistic Sharpe ratio. Always
concatenate OOS returns across splits before calling `deflated_sharpe`.
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats


def _is_negligible_std(std: float, arr: np.ndarray) -> bool:
    """True if std should be treated as zero relative to the scale of `arr`."""
    if not np.isfinite(std):
        return True

    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(a))) if a.size > 0 else 1.0)
    return float(std) <= 1e-9 * scale


def aggregate_returns_by_group(returns: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Aggregate per-period returns into one compounded return per group.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of simple per-period returns. ``1 + r`` is the growth factor.
        Converted to float64 and flattened.
    group_ids : np.ndarray
        1-D integer array of same length as *returns*. Entries with the same group
        id must be contiguous, and *group_ids* must be non-decreasing.

    Returns
    -------
    np.ndarray
        1-D float64 array of compounded per-group returns. Empty input returns an
        empty array. NaN values in *returns* propagate to the group result.

    Raises
    ------
    ValueError
        If *returns* and *group_ids* have different lengths, or if *group_ids* is
        not non-decreasing.
    """
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)

    if returns.size != group_ids.size:
        raise ValueError("returns and group_ids must have the same length")
    if np.any(np.diff(group_ids) < 0):
        raise ValueError("group_ids must be non-decreasing")

    if returns.size == 0:
        return returns

    starts = np.r_[0, np.flatnonzero(np.diff(group_ids) != 0) + 1]
    compounded = np.multiply.reduceat(1.0 + returns, starts) - 1.0

    group_sizes = np.diff(np.r_[starts, group_ids.size])
    single_row_groups = group_sizes == 1
    compounded[single_row_groups] = returns[starts[single_row_groups]]

    return compounded


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration_days: int
    var_95: float
    es_95: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_periods: int
    skew: float
    kurtosis: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a flat dict with float values."""
        return {
            "total_return": float(self.total_return),
            "cagr": float(self.cagr),
            "ann_volatility": float(self.ann_volatility),
            "sharpe": float(self.sharpe),
            "sortino": float(self.sortino),
            "calmar": float(self.calmar),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_duration_days": float(self.max_drawdown_duration_days),
            "var_95": float(self.var_95),
            "es_95": float(self.es_95),
            "hit_rate": float(self.hit_rate),
            "avg_win": float(self.avg_win),
            "avg_loss": float(self.avg_loss),
            "profit_factor": float(self.profit_factor),
            "n_periods": float(self.n_periods),
            "skew": float(self.skew),
            "kurtosis": float(self.kurtosis),
        }


def compute_metrics(
    returns: np.ndarray, *, periods_per_year: int = 252, rf: float = 0.0
) -> PerformanceMetrics:
    """Compute standard performance, risk, and overfitting statistics.

    ``returns`` are simple per-period returns (1 + return is a growth factor), not log
    returns and not annualized.  ``rf`` is an ANNUAL risk-free rate.  It is converted to
    per-period units using the additive approximation ``rf / periods_per_year``.

    Parameters
    ----------
    returns:
        Simple per-period returns.
    periods_per_year:
        Number of periods per year; used only for annualization.
    rf:
        Annual risk-free rate.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = len(arr)

    if n == 0:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            ann_volatility=np.nan,
            sharpe=np.nan,
            sortino=np.nan,
            calmar=np.nan,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            var_95=np.nan,
            es_95=np.nan,
            hit_rate=np.nan,
            avg_win=np.nan,
            avg_loss=np.nan,
            profit_factor=np.nan,
            n_periods=0,
            skew=np.nan,
            kurtosis=np.nan,
        )

    period_std = float(np.std(arr, ddof=1)) if n >= 2 else np.nan

    total_return = float(np.prod(1.0 + arr) - 1.0)
    growth = 1.0 + total_return
    cagr = (
        float(growth ** (periods_per_year / n) - 1.0)
        if growth >= 0.0
        else np.nan
    )
    ann_vol = (
        float(period_std * math.sqrt(periods_per_year)) if n >= 2 else np.nan
    )

    equity_cumprod = np.cumprod(1.0 + arr)
    # Seed the cumprod curve with 1.0 so drawdowns starting in the first period are measured.
    equity_seeded = np.concatenate(([1.0], equity_cumprod))
    max_dd_depth, peak_idx, trough_idx = max_drawdown(equity_seeded)
    max_dd_duration = int(trough_idx - peak_idx)

    calmar = cagr / abs(max_dd_depth) if abs(max_dd_depth) > 1e-12 else np.nan

    p5 = float(np.percentile(arr, 5.0))
    var_95 = -p5

    tail = arr[arr <= p5]
    es_95 = -float(np.mean(tail)) if tail.size > 0 else np.nan

    hit_rate = float(np.mean(arr > 0.0))

    pos = arr[arr > 0.0]
    neg = arr[arr < 0.0]
    avg_win = float(np.mean(pos)) if pos.size > 0 else np.nan
    avg_loss = float(np.mean(neg)) if neg.size > 0 else np.nan

    gains = float(np.sum(pos))
    losses = float(np.sum(neg))
    profit_factor = gains / abs(losses) if losses != 0.0 else np.nan

    if n < 3 or _is_negligible_std(period_std, arr):
        skew = np.nan
    else:
        skew = float(stats.skew(arr, bias=False))

    if n < 4 or _is_negligible_std(period_std, arr):
        kurtosis = np.nan
    else:
        kurtosis = float(stats.kurtosis(arr, fisher=False, bias=False))

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        ann_volatility=ann_vol,
        sharpe=sharpe_ratio(arr, periods_per_year=periods_per_year, rf=rf),
        sortino=sortino_ratio(arr, periods_per_year=periods_per_year, rf=rf),
        calmar=calmar,
        max_drawdown=max_dd_depth,
        max_drawdown_duration_days=max_dd_duration,
        var_95=var_95,
        es_95=es_95,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_periods=n,
        skew=skew,
        kurtosis=kurtosis,
    )


def sharpe_ratio(
    returns: np.ndarray, *, periods_per_year: int = 252, rf: float = 0.0
) -> float:
    """Return the annualized Sharpe ratio of simple per-period returns.

    ``rf`` is an annual risk-free rate; the per-period rate used is
    ``rf / periods_per_year``.  The per-period Sharpe ratio is computed with sample
    standard deviation (``ddof=1``) and then annualized by ``sqrt(periods_per_year)``.
    Returns ``NaN`` if fewer than two observations or zero sample standard deviation.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = len(arr)
    if n < 2:
        return np.nan

    std = float(np.std(arr, ddof=1))
    if _is_negligible_std(std, arr):
        return np.nan

    rf_period = rf / periods_per_year
    excess = float(np.mean(arr)) - rf_period
    return float((excess / std) * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    *,
    periods_per_year: int = 252,
    rf: float = 0.0,
    target: float = 0.0,
) -> float:
    """Return the annualized Sortino ratio.

    Downside deviation uses every period in the denominator and only squared negative
    deviations in the numerator:
        sqrt(mean(min(returns - target, 0)^2)).
    ``rf`` is an annual risk-free rate converted as ``rf / periods_per_year``.
    Returns ``NaN`` if downside deviation is zero or fewer than two observations.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = len(arr)
    if n < 2:
        return np.nan

    rf_period = rf / periods_per_year
    downside_sq = np.minimum(arr - target, 0.0) ** 2
    downside_dev = math.sqrt(float(np.mean(downside_sq)))
    if _is_negligible_std(downside_dev, arr):
        return np.nan

    numerator_per_period = float(np.mean(arr)) - rf_period
    numerator_ann = numerator_per_period * periods_per_year
    denominator_ann = downside_dev * math.sqrt(periods_per_year)
    return float(numerator_ann / denominator_ann)


def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]:
    """Return (depth, peak_idx, trough_idx) of the worst drawdown.

    Depth is the most negative running-max normalized value:
        (equity[t] - running_max[t]) / running_max[t].
    The trough is the global argmin of that series, and the peak is the argmax of the
    equity curve before and including that trough.  Returns ``(0.0, 0, 0)`` for trivial
    input without raising.
    """
    eq = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    if eq.size <= 1:
        return (0.0, 0, 0)

    running_max = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_denom = np.where(running_max != 0.0, running_max, 1.0)
        dd = np.where(running_max != 0.0, (eq - running_max) / safe_denom, 0.0)

    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq[: trough_idx + 1]))
    depth = float(dd[trough_idx])
    return (depth, peak_idx, trough_idx)


def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Return running-max drawdowns, vectorized and <= 0 for every point.

    Running max uses ``np.maximum.accumulate``.  Values are exactly zero at running-max
    points.
    """
    eq = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    if eq.size == 0:
        return np.empty(0, dtype=np.float64)

    running_max = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.divide(
            eq - running_max,
            running_max,
            out=np.zeros_like(eq),
            where=running_max != 0.0,
        )
    return dd


def sharpe_standard_error(
    returns: np.ndarray,
    *,
    adjust_autocorr: bool = True,
    annualized: bool = False,
    periods_per_year: int = 252,
) -> float:
    """Return the standard error of a Sharpe ratio estimate.

    Base iid formula: ``sqrt((1 + 0.5 * SR^2) / T)``.  ``SR`` here is the raw
    per-period Sharpe ratio, not annualized and without an rf adjustment.  If
    ``adjust_autocorr``, apply Lo (2002) scaling using sample autocorrelations up to
    lag ``min(T-2, 10)`` with a biased covariance denominator.  Returns ``NaN`` when
    ``T < 3`` or sample standard deviation is zero.

    By default (``annualized=False``), the returned value is the same per-period,
    non-annualized standard error this function has always returned.  If
    ``annualized=True``, the final standard error is multiplied by
    ``sqrt(periods_per_year)`` so it can be compared directly against this repo's
    annualized ``sharpe_ratio``.  If ``annualized=True`` and ``periods_per_year`` is
    non-finite or ``<= 0``, returns ``NaN`` instead of letting ``math.sqrt`` propagate
    a bad value.
    """
    if annualized and (not math.isfinite(periods_per_year) or periods_per_year <= 0.0):
        return float("nan")

    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    t = len(arr)
    if t < 3:
        return np.nan

    std = float(np.std(arr, ddof=1))
    if _is_negligible_std(std, arr):
        return np.nan

    sr = float(np.mean(arr) / std)
    se = math.sqrt((1.0 + 0.5 * sr**2) / t)

    if not adjust_autocorr:
        per_period_se = float(se)
    else:
        max_lag = min(t - 2, 10)
        if max_lag <= 0:
            per_period_se = float(se)  # pragma: no cover - t >= 3 invariant
            # The t < 3 early return has already fired otherwise, so t >= 3 here, which
            # implies max_lag = min(t - 2, 10) >= 1. The condition max_lag <= 0 is never
            # true. Kept rather than removed because a zero or negative max_lag would make
            # the autocorrelation adjustment loop silently skip all lags, producing an
            # understated standard error without raising.
        else:
            x = arr - float(np.mean(arr))
            var_biased = float(np.mean(x * x))

            weighted_rho_sum = 0.0
            for k in range(1, max_lag + 1):
                cov_biased = float(np.mean(x[k:] * x[:-k]))
                rho_k = cov_biased / var_biased
                weighted_rho_sum += ((t - k) / t) * rho_k

            autocorr_scale = 1.0 + 2.0 * weighted_rho_sum
            autocorr_scale = max(autocorr_scale, 1e-8)
            per_period_se = float(se * math.sqrt(autocorr_scale))

    if annualized:
        return float(per_period_se * math.sqrt(periods_per_year))
    return per_period_se


def turnover(weights: np.ndarray) -> np.ndarray:
    """Return one-way per-period turnover for a 1-D or 2-D weight array.

    For shape ``(T, n_assets)``, turnover[t] = sum(abs(weights[t] - weights[t-1])) / 2.
    A 1-D input is treated as one asset.  The first period uses a prior of all zeros,
    so ``turnover[0] = sum(abs(weights[0])) / 2``.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    elif w.ndim != 2:
        raise ValueError("weights must be a 1-D or 2-D array")

    t = w.shape[0]
    if t == 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(t, dtype=np.float64)
    out[0] = 0.5 * float(np.sum(np.abs(w[0])))
    if t > 1:
        diffs = np.abs(np.diff(w, axis=0))
        one_way = np.sum(diffs, axis=1)
        out[1:] = 0.5 * one_way
    return out


def expected_max_sharpe(n_trials: int | float, var_trial_sharpes: float) -> float:
    """Return Bailey & Lopez de Prado's expected maximum Sharpe ratio under null.

    Formula:
        sqrt(var_trial_sharpes) *
        ((1 - gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e))),
    where ``gamma`` is Euler-Mascheroni and ``e`` is Euler's number.
    Raises ``ValueError`` for fewer than two trials and returns zero when variance is
    exactly zero. ``n_trials`` accepts a fractional ``float`` (e.g. an honest
    ``effective_n_trials()`` result, which is a correlation-based EFFECTIVE count, not
    an integer draw count) as well as a raw integer trial count -- the formula itself
    is continuous in ``N`` and does not require an integer.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be at least 2")
    if var_trial_sharpes < 0.0:
        raise ValueError("var_trial_sharpes must be non-negative")
    if var_trial_sharpes == 0.0:
        return 0.0

    sigma = math.sqrt(float(var_trial_sharpes))
    gamma = float(np.euler_gamma)
    inv_n = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    inv_ne = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(sigma * ((1.0 - gamma) * inv_n + gamma * inv_ne))


def deflated_sharpe(
    returns: np.ndarray, *, sr0: float, periods_per_year: int = 252
) -> float:
    """Return the Deflated Sharpe Ratio PSR for a pooled return series.

    Both ``sr0`` and the computed SR are per-period, non-annualized quantities;
    ``periods_per_year`` here is used ONLY for API consistency and currently has no
    effect on the computation.  Passing an annualized SR0 against a per-period SR is the
    classic DSR bug and will silently produce a wrong, usually inflated, DSR.

    The denominator is the PSR correction term:
        sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2).
    ``kurt`` is non-excess, raw kurtosis.  ``T`` is ``len(returns)``.  Values of ``T``
    below two return ``NaN``; because the formula needs reliable higher moments,
    fewer than four observations also return ``NaN``.  A non-positive denominator is
    floored at ``1e-12`` defensively, and the final CDF value is clipped to ``[0, 1]``.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    t = len(arr)
    if t < 2:
        return np.nan

    std = float(np.std(arr, ddof=1))
    if _is_negligible_std(std, arr):
        return np.nan

    sr = float(np.mean(arr) / std)

    if t < 3:
        skew = np.nan
    else:
        skew = float(stats.skew(arr, bias=False))

    if t < 4:
        kurt = np.nan
    else:
        kurt = float(stats.kurtosis(arr, fisher=False, bias=False))

    if not (np.isfinite(skew) and np.isfinite(kurt)):
        return np.nan

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    denom = max(float(denom), 1e-12)

    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)
    dsr = float(stats.norm.cdf(z))
    return float(np.clip(dsr, 0.0, 1.0))


def effective_n_trials(trial_returns: np.ndarray) -> float:
    """Return the participation ratio of the trial correlation matrix.

    ``trial_returns`` is shape ``(T, n_trials)``.  Eigenvalues are clipped at zero to
    guard tiny negative numerical values.  A single trial returns ``1.0``.
    """
    arr = np.asarray(trial_returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_trials = arr.shape[1]
    if n_trials == 1:
        return 1.0

    if arr.shape[0] < 2:
        return np.nan

    corr = np.corrcoef(arr, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return np.nan

    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)

    sum_eig = float(np.sum(eigvals))
    sum_sq = float(np.sum(eigvals**2))
    if sum_sq == 0.0:
        return 0.0  # pragma: no cover - correlation trace invariant
        # corr is a correlation matrix from np.corrcoef, which has 1.0 on its diagonal by
        # construction. Since eigenvalues are clipped at 0 but the trace (sum of eigenvalues)
        # equals the sum of the diagonal, which is n_trials >= 2 here (the n_trials == 1 case
        # already returned above), sum_eig > 0, and since eigvals are real and not all zero,
        # the sum of squared eigenvalues is strictly positive for any valid correlation
        # matrix. Kept as a guard against a future refactor that changes the eigenvalue
        # clipping or the correlation input source.

    return float((sum_eig**2) / sum_sq)


def _per_trial_sharpe(return_matrix: np.ndarray) -> np.ndarray:
    """Return non-annualized per-trial Sharpe ratios, using -inf for degenerate columns."""
    if return_matrix.ndim != 2:
        raise ValueError("return_matrix must be a 2-D array")

    t, n_trials = return_matrix.shape
    if t < 2:
        return np.full(n_trials, -np.inf, dtype=np.float64)

    means = np.mean(return_matrix, axis=0)
    stds = np.std(return_matrix, axis=0, ddof=1)
    out = np.full(n_trials, -np.inf, dtype=np.float64)

    for j in range(n_trials):
        std_j = float(stds[j])
        if not _is_negligible_std(std_j, return_matrix[:, j]):
            out[j] = float(means[j]) / std_j

    return out


def pbo_cscv(trial_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Return Combinatorially Symmetric Cross-Validation PBO.

    The rows are split into ``n_splits`` contiguous blocks.  Every balanced combination
    of ``n_splits // 2`` blocks is used once as in-sample (IS); the complement is
    out-of-sample (OOS).  Per-trial non-annualized Sharpe ratios are computed in both
    halves.  For the best IS trial, its relative OOS rank is mapped to ``(0,1)`` via
    ``omega = rank / (n_trials + 1)``, where ``rank`` is 1-indexed and ascending from
    ``stats.rankdata(..., method="average")``.  The result is logit-transformed, and the
    PBO is the fraction of partitions where the logit is ``< 0``.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")

    mat = np.asarray(trial_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("trial_matrix must be a 2-D array")
    if mat.shape[1] < 2:
        raise ValueError("trial_matrix must have at least two trial columns")
    if n_splits > mat.shape[0]:
        raise ValueError("n_splits must not exceed the number of return rows")

    blocks = np.array_split(mat, n_splits, axis=0)
    n_trials = mat.shape[1]

    count_lambda_lt_zero = 0
    partitions = 0

    for is_idx_tuple in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_idx_tuple)
        is_blocks = [blocks[i] for i in is_idx_tuple]
        oos_blocks = [blocks[i] for i in range(n_splits) if i not in is_set]

        is_mat = np.concatenate(is_blocks, axis=0)
        oos_mat = np.concatenate(oos_blocks, axis=0)

        is_sharpes = _per_trial_sharpe(is_mat)
        oos_sharpes = _per_trial_sharpe(oos_mat)

        best_is_idx = int(np.argmax(is_sharpes))

        ranks = stats.rankdata(oos_sharpes, method="average")
        rank_of_best = ranks[best_is_idx]
        omega_bar = rank_of_best / (n_trials + 1.0)
        omega_bar = float(np.clip(omega_bar, 1e-12, 1.0 - 1e-12))

        lambda_c = math.log(omega_bar / (1.0 - omega_bar))
        if lambda_c < 0.0:
            count_lambda_lt_zero += 1

        partitions += 1

    return float(count_lambda_lt_zero / partitions)


def verdict_line(
    sharpe_net: float,
    sr_se: float,
    n_trials: int,
    n_eff: float,
    sr0: float,
    dsr: float,
    pbo: float,
    *,
    ruined: bool = False,
    n_trades: int | None = None,
    min_trades_for_sharpe: int = 30,
) -> str:
    """Return a one-line summary with an ESTABLISHED/NOT ESTABLISHED verdict.

    A ruined run produces a hard capital-destroyed warning instead: the ruin guard forces
    post-ruin returns to zero, which flatters the computed Sharpe and makes it unreliable.
    If *n_trades* is set below *min_trades_for_sharpe*, a small-sample warning is appended.
    The default floor of 30 trades is a common quant/statistics rule-of-thumb below which a
    Sharpe/t-stat is too dominated by small-sample noise to be considered approximately
    normal.
    """
    if ruined:
        return (
            "RUINED: capital destroyed (ruin guard tripped) | Sharpe is not meaningful "
            "because post-ruin returns are forced to 0.0, which flatters it | NOT ESTABLISHED"
        )

    verdict = "ESTABLISHED" if dsr >= 0.95 else "NOT ESTABLISHED"
    # SR0/DSR use general (`g`) formatting, not fixed-point, because a genuine
    # deflation effect can move DSR/SR0 by many orders of magnitude (e.g. a
    # correctly-deflated Sharpe under a large SR0 can land at 1e-40 rather than
    # 1e-74) -- fixed 3-decimal formatting collapses both to the same "0.000" and
    # silently destroys exactly the honest-vs-raw distinction this wiring exists
    # to preserve.
    line = (
        f"Sharpe={sharpe_net:.3f} (SE={sr_se:.3f}) | trials={n_trials} "
        f"(eff={n_eff:.3f}) | SR0={sr0:.4g} | DSR={dsr:.4g} | PBO={pbo:.3f} | {verdict}"
    )
    if n_trades is not None and n_trades < min_trades_for_sharpe:
        line += (
            f" | WARNING: too few trades for the Sharpe to be statistically meaningful "
            f"(n_trades={n_trades}, min_trades_for_sharpe={min_trades_for_sharpe})"
        )
    return line
