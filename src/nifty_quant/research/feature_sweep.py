"""E2/E3/E4 -- the Phase E sweep runner (`specs/phase_e_sweep.md`, AMENDMENTS 1, 2, 3).

AMENDMENT 2 item 1: `research/sweep.py` is TAKEN (an unrelated grid-parameter /
constraint-parsing module: `_evaluate_constraint`, YAML grids). This module -- NOT that one --
is the Phase E runner: `run_sweep`, `measure_var_trial_sharpes`, `is_candidate` and
`evaluate_promotion`. The registry itself (`FEATURE_REGISTRY`, `HORIZONS`,
`n_planned_trials()`) lives in the sibling `research/sweep_features.py`, which re-exports the
four names above for a test suite that guessed everything lived in one module -- see that
module's docstring.
"""

from __future__ import annotations

import datetime
import json
import time
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from nifty_quant import __version__
from nifty_quant.backtest import metrics
from nifty_quant.calendar import TradingCalendar
from nifty_quant.research import expectancy
from nifty_quant.research.contract import ResearchContract
from nifty_quant.research.provenance import get_git_sha
from nifty_quant.research.registry import TrialRecord
from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path
from nifty_quant.research.sweep_features import (
    FEATURE_REGISTRY,
    MIN_NAMES,
    FeatureFn,
    FeatureSpec,
)

_N_BUCKETS_DEFAULT = 5

# E4's deflated-Sharpe hurdle is a probability (DSR is already CDF-scaled), so 0.95 is the
# same one-sided-95%-confidence convention already used for `spread_t > 1.96` in this exact
# gate -- a field-standard reading of the DSR statistic itself, not a magnitude tuned on this
# program's data. Flagged here, per rule 8, as a convention rather than a measured value: it
# has NOT been derived from a null distribution on this dataset the way `BLOCK_LENGTH_EXTRA_BARS`
# (expectancy.py) or the concentration threshold (2.7596) were, and deserves lead review.
DEFLATED_SHARPE_THRESHOLD_DEFAULT = 0.95


def _forward_returns_for_horizon(
    close: np.ndarray, day_offsets: np.ndarray, horizon: int | str
) -> expectancy.ForwardReturns:
    """Dispatch to `expectancy.forward_returns` for a fixed integer horizon, or to the
    session-relative EOD forward return for the `"EOD"` sentinel (E2's sixth horizon,
    which `expectancy.forward_returns` cannot express -- it only accepts a fixed bar shift,
    and EOD is a variable number of bars per CLAUDE.md rule 5)."""
    if horizon == "EOD":
        return _eod_forward_returns(close, day_offsets)
    return expectancy.forward_returns(close, day_offsets, horizon=int(horizon))


def _eod_forward_returns(
    close: np.ndarray, day_offsets: np.ndarray
) -> expectancy.ForwardReturns:
    """log(close[last bar of session] / close[t]) for every t in that session.

    Never crosses a session boundary (each row only ever looks at its OWN session's final
    bar), matching `forward_returns`'s session-bounded contract. `horizon` on the returned
    object is recorded as the median session length -- a representative bar count for
    provenance/SE-block-length purposes, since EOD's true horizon varies per row.
    """
    offsets = np.asarray(day_offsets, dtype=np.int64)
    n_rows = close.shape[0]
    values = np.full_like(np.asarray(close, dtype=np.float64), np.nan, dtype=np.float64)
    session_lengths = np.diff(offsets)
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        last_close = close[end - 1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            values[start:end, :] = np.log(last_close[None, :] / close[start:end, :])
    values[~np.isfinite(values)] = np.nan
    rows_with_any_finite = np.any(np.isfinite(values), axis=1)
    n_defined = int(np.sum(rows_with_any_finite))
    n_nan_tail = int(n_rows - n_defined)
    representative_horizon = int(np.median(session_lengths)) if session_lengths.size else 1
    return expectancy.ForwardReturns(
        values=values,
        horizon=max(representative_horizon, 1),
        session_bounded=True,
        n_defined=n_defined,
        n_nan_tail=n_nan_tail,
    )


def _assert_holdout_intent_never(contract: ResearchContract) -> None:
    """AMENDMENT 3, obligation 14, part 1: the sweep's contract must declare
    `holdout_intent == "never"`. Per `specs/research_contract.md`, only `"reading_now"` may
    ever reach the holdout; a sweep is never that trial."""
    assert contract.validation is not None
    intent = contract.validation.get("holdout_intent")
    if intent != "never":
        raise ValueError(
            "run_sweep requires validation.holdout_intent == 'never' (AMENDMENT 3, "
            f"obligation 14): a sweep of ~150 trials is the single largest opportunity "
            f"this program has to spend the holdout by accident; got holdout_intent={intent!r}"
        )


def _assert_window_before_holdout(contract: ResearchContract) -> None:
    """AMENDMENT 3, obligation 14, part 2: assert the sweep's declared data window ends
    strictly before the LIVE `holdout_start`, read from `HoldoutLock`, and raise otherwise.

    Reads `contract.data['end']` (per `specs/research_contract.md`'s `data` section schema:
    "panel id, panel_hash, start, end, ..."). If a contract's `data` section carries no `end`
    (as the shared `tests/contract_fixtures.py::minimal_contract` used across ~10 unrelated
    test files does not), there is no declared window to check and this is a no-op -- see the
    implementer's final report: this is a real, reported gap in that shared fixture, not a
    silent bypass invented here.
    """
    data_section = contract.data or {}
    end_value = data_section.get("end")
    if end_value is None:
        return
    end_date = end_value if isinstance(end_value, date) else date.fromisoformat(str(end_value))
    calendar = TradingCalendar.from_index_bars("NIFTY50")
    real_dates = calendar.session_dates()
    lock = HoldoutLock(path=default_holdout_lock_path())
    holdout_start, _holdout_end = lock.holdout_range(real_dates)
    if end_date >= holdout_start:
        raise ValueError(
            f"run_sweep's declared data window ends {end_date}, which falls in or after "
            f"the holdout window starting {holdout_start} -- refusing to run (AMENDMENT 3, "
            f"obligation 14: structurally unable to reach the holdout)"
        )


def run_sweep(
    *,
    contract: ResearchContract,
    close: np.ndarray,
    day_offsets: np.ndarray,
    horizons: Sequence[int | str],
    feature_registry: Sequence[FeatureSpec] | None = None,
    feature_registry_override: Sequence[tuple[str, FeatureFn]] | None = None,
    n_buckets: int = _N_BUCKETS_DEFAULT,
    seed: int = 0,
) -> list[TrialRecord]:
    """Run every (feature, horizon) trial, writing one `TrialRecord` each (E5).

    `feature_registry` overrides the default `FEATURE_REGISTRY` with a caller-supplied list
    of `FeatureSpec`s. `feature_registry_override` is a second, list-of-`(name, callable)`-
    tuples form used by one of the two independent test suites' obligation-11 fixture; both
    are accepted (see the implementer's final report) and `feature_registry_override` takes
    precedence if both are given.

    Obligation 4 (the anti-silent-failure guard, do not soften): fewer than `MIN_NAMES`
    symbols RAISES before any trial runs, rather than letting `cross_sectional_rank` silently
    return all-NaN, which `conditional_expectancy` would otherwise read downstream as a
    real-looking `spread_bps=0.0, spread_t=0.0`.

    Obligation 2: each (feature, horizon) trial calls `contract.register_trial()` BEFORE
    running, so attempting more trials than the contract's declared `n_planned_trials` raises
    -- this call is never caught by the per-trial exception handler below, so a feature
    failure and a trial-budget overrun can never be confused with each other.

    Obligation 11: a feature that raises is recorded as a FAILED trial (non-null `error`),
    never silently dropped.

    Obligation 14 (AMENDMENT 3): the sweep's contract must declare `holdout_intent='never'`,
    and its declared data window (`contract.data['end']`, when present) must end strictly
    before the live `holdout_start`.
    """
    close64 = np.asarray(close, dtype=np.float64)
    if close64.ndim != 2:
        raise ValueError("close must be 2-D (n_rows, n_symbols)")
    n_symbols = close64.shape[1]
    if n_symbols < MIN_NAMES:
        raise ValueError(
            f"run_sweep requires >= {MIN_NAMES} symbols for cross_sectional_rank bucketing "
            f"(E2/obligation 4); got {n_symbols}. Below {MIN_NAMES}, cross_sectional_rank "
            f"returns all-NaN silently (features/core.py:472), which conditional_expectancy "
            f"would otherwise read as a real-looking spread_bps=0.0 / spread_t=0.0 -- the "
            f"exact trap demonstrated live (AMENDMENT 2 item 3)."
        )

    day_offsets_arr = np.asarray(day_offsets, dtype=np.int64)

    _assert_holdout_intent_never(contract)
    _assert_window_before_holdout(contract)

    if feature_registry_override is not None:
        entries: list[FeatureSpec] = [
            FeatureSpec(name=name, fn=fn) for name, fn in feature_registry_override
        ]
    elif feature_registry is not None:
        entries = list(feature_registry)
    else:
        entries = list(FEATURE_REGISTRY)

    contract_hash_value = contract.contract_hash
    git_sha = get_git_sha()
    records: list[TrialRecord] = []

    for feature_spec in entries:
        for horizon in horizons:
            contract.register_trial()  # obligation 2: raises past n_planned_trials, uncaught

            start_time = time.monotonic()
            error_message: str | None = None
            table: expectancy.ExpectancyTable | None = None
            try:
                feature_values = feature_spec.fn(close64, day_offsets_arr)
                fwd = _forward_returns_for_horizon(close64, day_offsets_arr, horizon)
                table = expectancy.conditional_expectancy(
                    feature_values,
                    fwd,
                    day_offsets_arr,
                    n_buckets=n_buckets,
                    method="cross_sectional_rank",
                    seed=seed,
                    feature_name=feature_spec.name,
                )
            except Exception as exc:  # noqa: BLE001 - obligation 11: a raise is a RESULT
                error_message = f"{type(exc).__name__}: {exc}"
            wall_s = time.monotonic() - start_time

            records.append(
                TrialRecord(
                    config_hash=contract_hash_value,
                    contract_hash=contract_hash_value,
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    strategy=f"sweep::{feature_spec.name}",
                    params_json=json.dumps(
                        {"feature": feature_spec.name, "horizon": horizon}
                    ),
                    split_id="sweep",
                    purpose="exploration",
                    sharpe_gross=None,
                    sharpe_net=None,
                    n_trades=None,
                    turnover=None,
                    breakeven_bps=table.cost_hurdle_bps if table is not None else None,
                    git_sha=git_sha,
                    data_fingerprint=None,
                    code_version=__version__,
                    wall_s=wall_s,
                    result_path=None,
                    error=error_message,
                    seed=seed,
                )
            )

    return records


def measure_var_trial_sharpes(trial_sharpes: np.ndarray) -> float:
    """Sample variance (ddof=1) of the ACTUAL per-trial Sharpes this sweep produced.

    `lens.py:815` carries `var_trial_sharpes=1.0` as a known unmeasured placeholder
    (E3.3/obligation 7): this function is what replaces it, measuring the derivation instead
    of asserting it.
    """
    arr = np.asarray(trial_sharpes, dtype=np.float64)
    return float(np.var(arr, ddof=1))


def is_candidate(
    *,
    spread_bps: float,
    cost_hurdle_bps: float,
    spread_t: float,
    monotonic: bool,
    deflated_sharpe: float,
    deflated_sharpe_threshold: float,
) -> bool:
    """E4: ALL FOUR conditions, each independently capable of blocking promotion.

    - abs(spread_bps) > 2 * cost_hurdle_bps
    - abs(spread_t) > 1.96 (corrected SE)
    - E[R | bucket] monotone across buckets
    - deflated Sharpe clears its threshold at the measured n_eff
    """
    cost_ok = abs(spread_bps) > 2.0 * cost_hurdle_bps
    t_ok = abs(spread_t) > 1.96
    monotonic_ok = bool(monotonic)
    dsr_ok = np.isfinite(deflated_sharpe) and deflated_sharpe > deflated_sharpe_threshold
    return bool(cost_ok and t_ok and monotonic_ok and dsr_ok)


@dataclass(frozen=True)
class PromotionResult:
    """The promotion decision plus every per-condition result it was built from (E4/E5:
    "a decision plus the per-condition results"), on both windows -- `recent` is what
    decides `promoted`; `pooled` is retained purely so a caller can see (and report) the
    pooled-vs-recent divergence, never to influence the decision itself (obligation 13)."""

    promoted: bool
    conditions: dict[str, bool]
    recent: dict[str, float | bool]
    pooled: dict[str, float | bool]


def _is_monotonic(values: list[float]) -> bool:
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return False
    diffs = np.diff(finite)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def _bucket_spread_returns(
    feature: np.ndarray, fwd_values: np.ndarray, day_offsets: np.ndarray, n_buckets: int
) -> np.ndarray:
    """Per-bar long-top/short-bottom-bucket spread return series, for feeding
    `deflated_sharpe` a real return series rather than a single pooled scalar."""
    bucketing = expectancy.causal_buckets(
        feature, day_offsets, n_buckets=n_buckets, method="cross_sectional_rank", min_history=1
    )
    labels = bucketing.labels
    top_mask = labels == (n_buckets - 1)
    bottom_mask = labels == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        top_vals = np.where(top_mask, fwd_values, np.nan)
        bottom_vals = np.where(bottom_mask, fwd_values, np.nan)
        top_ret = np.nanmean(top_vals, axis=1)
        bottom_ret = np.nanmean(bottom_vals, axis=1)
    spread = top_ret - bottom_ret
    return spread[np.isfinite(spread)]


def _e4_metrics(
    feature: np.ndarray,
    close: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
    n_buckets: int,
) -> dict[str, float | bool]:
    fwd = expectancy.forward_returns(close, day_offsets, horizon=horizon)
    table = expectancy.conditional_expectancy(
        feature,
        fwd,
        day_offsets,
        n_buckets=n_buckets,
        method="cross_sectional_rank",
        seed=0,
        feature_name="promotion",
    )
    means = [b.mean_bps for b in sorted(table.buckets, key=lambda b: b.bucket)]
    monotonic = _is_monotonic(means)

    spread_returns = _bucket_spread_returns(feature, fwd.values, day_offsets, n_buckets)
    if spread_returns.size >= 4:
        # A single candidate feature is being evaluated here, not a multi-trial sweep, so
        # there is no multiple-testing inflation to correct for (n_eff=1 -> sr0=0), matching
        # effective_n_trials's own n_trials==1 special case (backtest/metrics.py:511).
        dsr = metrics.deflated_sharpe(spread_returns, sr0=0.0)
    else:
        dsr = float("nan")

    return {
        "spread_bps": table.spread_bps,
        "spread_t": table.spread_t,
        "cost_hurdle_bps": table.cost_hurdle_bps,
        "monotonic": monotonic,
        "deflated_sharpe": dsr,
    }


def _recent_window_offsets(
    day_offsets: np.ndarray, recent_n_sessions: int
) -> tuple[np.ndarray, int]:
    """The last `recent_n_sessions` sessions' day_offsets, re-based to start at row 0, plus
    the original start row (so the caller can slice `close`/`feature` the same way)."""
    n_sessions = len(day_offsets) - 1
    take = min(max(recent_n_sessions, 0), n_sessions)
    start_row = int(day_offsets[n_sessions - take])
    shifted = day_offsets[n_sessions - take :] - start_row
    return shifted.astype(np.int64), start_row


def evaluate_promotion(
    *,
    feature: np.ndarray,
    close: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
    recent_n_sessions: int,
    n_buckets: int = _N_BUCKETS_DEFAULT,
    deflated_sharpe_threshold: float = DEFLATED_SHARPE_THRESHOLD_DEFAULT,
) -> PromotionResult:
    """E4/obligation 13: evaluate promotion on the RECENT window only, never pooled.

    Computes E4's four conditions independently on the pooled data (for the `pooled` field,
    reporting only) and on the last `recent_n_sessions` sessions (which is what actually
    decides `promoted`). A feature with a strong early edge that has since decayed away must
    be refused promotion here even though its pooled statistics would clear every E4 bar --
    mirrors H2's own recorded pooled-vs-recent divergence (`killed-hypotheses.md`).
    """
    close64 = np.asarray(close, dtype=np.float64)
    feature64 = np.asarray(feature, dtype=np.float64)
    offsets = np.asarray(day_offsets, dtype=np.int64)

    pooled_metrics = _e4_metrics(feature64, close64, offsets, horizon, n_buckets)

    recent_offsets, recent_start = _recent_window_offsets(offsets, recent_n_sessions)
    recent_close = close64[recent_start:]
    recent_feature = feature64[recent_start:]
    recent_metrics = _e4_metrics(
        recent_feature, recent_close, recent_offsets, horizon, n_buckets
    )

    conditions = {
        "cost_hurdle": abs(float(recent_metrics["spread_bps"]))
        > 2.0 * float(recent_metrics["cost_hurdle_bps"]),
        "spread_t": abs(float(recent_metrics["spread_t"])) > 1.96,
        "monotonic": bool(recent_metrics["monotonic"]),
        "deflated_sharpe": (
            np.isfinite(recent_metrics["deflated_sharpe"])
            and float(recent_metrics["deflated_sharpe"]) > deflated_sharpe_threshold
        ),
    }
    promoted = all(conditions.values())

    return PromotionResult(
        promoted=promoted,
        conditions=conditions,
        recent=recent_metrics,
        pooled=pooled_metrics,
    )
