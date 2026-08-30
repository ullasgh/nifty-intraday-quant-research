"""E1 -- the Phase E feature registry (`specs/phase_e_sweep.md`, AMENDMENT 1 + 2).

A single explicit list, `FEATURE_REGISTRY`, naming every feature primitive this program
sweeps against `HORIZONS`. Deliberately NOT a glob over module contents: `n_planned_trials()`
is derived from this list's length, and a registry that silently grows via introspection
would change that denominator without anyone noticing (E1).

AMENDMENT 2 pins module ownership: THIS file owns the registry (`FEATURE_REGISTRY`,
`HORIZONS`, `n_planned_trials()`). The runner (`run_sweep`), the measured-variance helper
(`measure_var_trial_sharpes`) and the promotion gate (`is_candidate`/`evaluate_promotion`)
are OWNED by `research/feature_sweep.py` -- see that module's docstring. They are
re-exported at the bottom of this file (after the registry itself is fully defined) purely
for call-site convenience: one of the two independently-written test suites for this spec
(`tests/test_phase_e_sweep_a.py`) guessed that every Phase-E name lives in this module, since
the spec text names only `research/sweep_features.py` and never mentions a second module by
name until AMENDMENT 2. Both import paths resolve to the SAME function object; nothing is
duplicated. See the implementer's final report for the full list of places the two suites'
guesses disagree.

AMENDMENT 4 (2026-08-23): `run_sweep`'s signature now carries real `open_`/`high`/`low`/
`volume` panels (keyword-only, default `None`), not just `close`/`day_offsets`. Before this
amendment, every registry entry that needed more than a close panel silently synthesised a
PROXY from `close` alone (`volume = np.ones_like(close)`, `high = low = open_ = close`) --
which made a z-score of constant volume NaN and `log(H/L)` with `H == L` exactly zero,
degenerating SEVEN features to all-NaN output that was then excluded from the trial matrix
as if it were a measured null. **It was not measured at all.** Each such wrapper below now
takes its real field(s) as keyword-only arguments and declares them on its `FeatureSpec` via
`required_fields`; `run_sweep` validates every declared field is supplied BEFORE calling the
wrapper and raises (recorded as a FAILED trial naming the field, reusing the obligation-11
mechanism) rather than falling back to a proxy -- the whole point of the fix is that the
silent substitution must not simply move somewhere else. Features that genuinely need only
`close`/`day_offsets` are unchanged. Proxies that are NOT one of these four OHLCV fields
(the `_minute_of_day_proxy` synthetic timestamp, `sector_relative_return`'s all-one sector
id, `rv_to_vix_ratio`'s flat VIX level, `amihud_illiquidity`'s `|close|` traded-value
notional) are out of this amendment's scope -- `run_sweep` was never given real sector
metadata, a real VIX series, or real traded value/notional to substitute in their place, so
there is nothing to wire through yet; they remain documented sharp edges of this registry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from nifty_quant.features import core as _core
from nifty_quant.features import market as _market
from nifty_quant.features import persistence as _persistence

# Same cross-sectional-validity floor `cross_sectional_rank`/`breadth`/`dispersion`/
# `median_pairwise_correlation` already use (features/core.py:472 et al.) -- reused, not
# re-derived, because it is the identical question (how many names does a single-bar
# cross-sectional statistic need to be meaningful), not a new rule-8 threshold.
MIN_NAMES = 5

FeatureFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class FeatureSpec:
    """One registry entry: a name, a `(close, day_offsets, **fields) -> feature array`
    callable, and the OHLCV fields (beyond `close`/`day_offsets`, which every callable
    always receives positionally) it REQUIRES.

    AMENDMENT 4: `run_sweep` validates that every name in `required_fields` was supplied
    (non-`None`) to it before calling `fn`, then calls `fn` with EXACTLY those fields as
    keyword arguments -- e.g. `required_fields=frozenset({"high", "low"})` is called as
    `fn(close, day_offsets, high=..., low=...)`. The default, empty `required_fields`,
    is called exactly `fn(close, day_offsets)`, unchanged from before this amendment --
    every feature that never needed OHLCV keeps its original call signature verbatim.
    """

    name: str
    fn: FeatureFn
    required_fields: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Proxy construction shared by several wrappers below.
# ---------------------------------------------------------------------------

_OPEN_MINUTE_IST = 555  # 09:15 IST in minutes-since-midnight -- NSE's real session open.
_DEFAULT_WINDOW = 30  # bars; a lookback convention shared by every windowed wrapper here.
_DEFAULT_HALFLIFE_BARS = 20.0
_DEFAULT_OPENING_RANGE_BARS = 15
_DEFAULT_VARIANCE_RATIO_Q = 5
_VIX_PROXY_LEVEL = 20.0  # flat placeholder level; no real VIX series reaches this registry.


def _minute_of_day_proxy(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Synthetic IST minute-of-day, assuming 1-minute bars from a 09:15 open.

    A proxy: `run_sweep` carries no real bar timestamps, only `day_offsets` (session
    boundaries). Consistent with the repo's real convention (left-labelled 1-minute bars
    starting 09:15) but not measured from actual timestamps here.
    """
    n_rows = close.shape[0]
    bso = _market.bars_since_open(day_offsets, n_rows)
    return (_OPEN_MINUTE_IST + bso).astype(np.int64)


def _returns_and_market(
    close: np.ndarray, day_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-symbol log returns, plus an equal-weight cross-sectional mean as a market-index
    return proxy (no real index series reaches `run_sweep`)."""
    returns = _core.log_returns(close, day_offsets=day_offsets)
    with warnings.catch_warnings():
        # Every session's first row is all-NaN by construction (log_returns' own
        # session-start convention) -- nanmean's "empty slice" warning on that row is
        # expected, not a defect, and its NaN result is the correct output.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        market_returns = np.nanmean(returns, axis=1)
    return returns, market_returns


def _broadcast_row_stat(stat_1d: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Broadcast a market-wide per-bar statistic (shape (n_rows,)) to every symbol's
    column, matching the sweep's uniform (n_rows, n_symbols) feature convention -- every
    name shares the identical value on a given bar, which is what a market-level regime
    feature legitimately is."""
    return np.broadcast_to(stat_1d[:, None], close.shape).astype(np.float64).copy()


# ---------------------------------------------------------------------------
# Registry wrappers -- one per E1-named primitive, uniform (close, day_offsets) signature.
# ---------------------------------------------------------------------------


def _volume_zscore(
    close: np.ndarray, day_offsets: np.ndarray, *, volume: np.ndarray
) -> np.ndarray:
    minute_of_day = _minute_of_day_proxy(close, day_offsets)
    return _core.volume_zscore(volume, minute_of_day, _DEFAULT_WINDOW, day_offsets=day_offsets)


def _breakout_strength(
    close: np.ndarray, day_offsets: np.ndarray, *, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    return _core.breakout_strength(close, high, low, _DEFAULT_WINDOW, day_offsets=day_offsets)


def _parkinson_volatility(
    close: np.ndarray, day_offsets: np.ndarray, *, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    del close  # unused: parkinson_volatility is purely a high/low range statistic
    return _core.parkinson_volatility(high, low, _DEFAULT_WINDOW, day_offsets=day_offsets)


def _garman_klass_volatility(
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    return _core.garman_klass_volatility(
        open_, high, low, close, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _rogers_satchell_volatility(
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    return _core.rogers_satchell_volatility(
        open_, high, low, close, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _efficiency_ratio(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    return _core.efficiency_ratio(close, _DEFAULT_WINDOW, day_offsets=day_offsets)


def _hurst_on_stitched(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    return _persistence.hurst_on_stitched(close, day_offsets)


def _variance_ratio(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    # variance_ratio raises NotImplementedError for 2-D x combined with day_offsets (its own
    # docstring), so it is applied per symbol on the 1-D log-price series and broadcast --
    # a time-invariant per-symbol persistence characteristic, not a time-varying signal.
    n_rows, n_symbols = close.shape
    log_close = np.log(close)
    per_symbol = np.array(
        [
            _persistence.variance_ratio(
                log_close[:, s], _DEFAULT_VARIANCE_RATIO_Q, day_offsets=day_offsets
            )
            for s in range(n_symbols)
        ],
        dtype=np.float64,
    )
    return np.broadcast_to(per_symbol[None, :], (n_rows, n_symbols)).astype(np.float64).copy()


def _rolling_beta(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, market_returns = _returns_and_market(close, day_offsets)
    return _market.rolling_beta(
        returns, market_returns, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _beta_residual_return(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, market_returns = _returns_and_market(close, day_offsets)
    return _market.beta_residual_return(
        returns, market_returns, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _sector_relative_return(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    n_symbols = close.shape[1]
    sector_ids = np.zeros(n_symbols, dtype=np.int64)  # proxy: no sector metadata in run_sweep
    return _market.sector_relative_return(returns, sector_ids)


def _breadth(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    return _broadcast_row_stat(_market.breadth(returns), close)


def _cross_sectional_dispersion(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    return _broadcast_row_stat(_market.cross_sectional_dispersion(returns), close)


def _median_pairwise_correlation(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    stat = _market.median_pairwise_correlation(
        returns, _DEFAULT_WINDOW, day_offsets=day_offsets
    )
    return _broadcast_row_stat(stat, close)


def _vol_ratio(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    return _market.vol_ratio(
        returns, _DEFAULT_WINDOW // 3, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _rv_to_vix_ratio(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    rv_ann = _core.ewma_volatility_ann(
        close, day_offsets, halflife=_DEFAULT_HALFLIFE_BARS
    )
    n_rows, n_symbols = rv_ann.shape
    out = np.empty((n_rows, n_symbols), dtype=np.float64)
    vix_proxy_1d = np.full(n_rows, _VIX_PROXY_LEVEL)  # proxy: no VIX series in run_sweep
    for s in range(n_symbols):
        out[:, s] = _market.rv_to_vix_ratio(rv_ann[:, s], vix_proxy_1d)
    return out


def _close_location_value(
    close: np.ndarray, day_offsets: np.ndarray, *, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    del day_offsets
    return _market.close_location_value(high, low, close)


def _signed_volume_proxy(
    close: np.ndarray,
    day_offsets: np.ndarray,
    *,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    del day_offsets
    return _market.signed_volume_proxy(high, low, close, volume)


def _amihud_illiquidity(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    returns, _market_returns = _returns_and_market(close, day_offsets)
    traded_value = np.abs(close)  # proxy notional: no real traded value in run_sweep
    return _market.amihud_illiquidity(
        returns, traded_value, _DEFAULT_WINDOW, day_offsets=day_offsets
    )


def _overnight_return(close: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    n_rows, n_symbols = close.shape
    out = np.empty((n_rows, n_symbols), dtype=np.float64)
    for s in range(n_symbols):
        out[:, s] = _market.overnight_return(close[:, s], day_offsets)
    return out


def _tradable_overnight_return(
    close: np.ndarray, day_offsets: np.ndarray, *, open_: np.ndarray
) -> np.ndarray:
    minute_of_day = _minute_of_day_proxy(close, day_offsets)
    return _market.tradable_overnight_return(open_, close, day_offsets, minute_of_day)


def _opening_range(
    close: np.ndarray, day_offsets: np.ndarray, *, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    del close  # unused: opening_range is purely a high/low range statistic
    # reduce the (or_high, or_low) pair opening_range returns to a single range-width array.
    high_range, low_range = _market.opening_range(
        high, low, day_offsets, _DEFAULT_OPENING_RANGE_BARS
    )
    return np.asarray(high_range - low_range, dtype=np.float64)


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec("volume_zscore", _volume_zscore, required_fields=frozenset({"volume"})),
    FeatureSpec(
        "breakout_strength", _breakout_strength, required_fields=frozenset({"high", "low"})
    ),
    FeatureSpec(
        "parkinson_volatility",
        _parkinson_volatility,
        required_fields=frozenset({"high", "low"}),
    ),
    FeatureSpec(
        "garman_klass_volatility",
        _garman_klass_volatility,
        required_fields=frozenset({"open_", "high", "low"}),
    ),
    FeatureSpec(
        "rogers_satchell_volatility",
        _rogers_satchell_volatility,
        required_fields=frozenset({"open_", "high", "low"}),
    ),
    FeatureSpec("efficiency_ratio", _efficiency_ratio),
    FeatureSpec("hurst_on_stitched", _hurst_on_stitched),
    FeatureSpec("variance_ratio", _variance_ratio),
    FeatureSpec("rolling_beta", _rolling_beta),
    FeatureSpec("beta_residual_return", _beta_residual_return),
    FeatureSpec("sector_relative_return", _sector_relative_return),
    FeatureSpec("breadth", _breadth),
    FeatureSpec("cross_sectional_dispersion", _cross_sectional_dispersion),
    FeatureSpec("median_pairwise_correlation", _median_pairwise_correlation),
    FeatureSpec("vol_ratio", _vol_ratio),
    FeatureSpec("rv_to_vix_ratio", _rv_to_vix_ratio),
    FeatureSpec(
        "close_location_value", _close_location_value, required_fields=frozenset({"high", "low"})
    ),
    FeatureSpec(
        "signed_volume_proxy",
        _signed_volume_proxy,
        required_fields=frozenset({"high", "low", "volume"}),
    ),
    FeatureSpec("amihud_illiquidity", _amihud_illiquidity),
    FeatureSpec("overnight_return", _overnight_return),
    FeatureSpec(
        "tradable_overnight_return",
        _tradable_overnight_return,
        required_fields=frozenset({"open_"}),
    ),
    FeatureSpec("opening_range", _opening_range, required_fields=frozenset({"high", "low"})),
)

# E2: horizons in {1, 5, 15, 30, 60 bars, EOD}. "EOD" is a distinguished sentinel handled by
# `feature_sweep._forward_returns_for_horizon` (log return from bar t to the LAST bar of t's
# own session), since `expectancy.forward_returns` only accepts a fixed integer bar shift.
HORIZONS: tuple[int | str, ...] = (1, 5, 15, 30, 60, "EOD")


def n_planned_trials() -> int:
    """`len(FEATURE_REGISTRY) * len(HORIZONS)` -- NEVER a hand-typed constant (E1): adding
    one registry entry or one horizon must change this denominator automatically."""
    return len(FEATURE_REGISTRY) * len(HORIZONS)


# `tests/test_phase_e_sweep_b.py` independently guessed a different registry shape:
# `FEATURES` (plain list[str]) and `N_PLANNED_TRIALS` (a precomputed constant) rather than
# `FEATURE_REGISTRY` (list[FeatureSpec]) and `n_planned_trials()` (a callable). Both are
# harmless to expose side by side in this module -- they name different things and neither
# shadows the other -- so both suites' guesses are satisfied without contradiction.
FEATURES: list[str] = [entry.name for entry in FEATURE_REGISTRY]
N_PLANNED_TRIALS: int = n_planned_trials()


# `feature_sweep.py` imports FEATURE_REGISTRY/HORIZONS/MIN_NAMES/FeatureSpec from THIS module
# at its own top -- an eager re-export here (`from ...feature_sweep import run_sweep`) would
# therefore be a genuine module-level circular import whose success depends on which of the
# two modules a caller happens to import first (verified: importing `feature_sweep` first
# raises ImportError, importing `sweep_features` first does not). PEP 562 module
# `__getattr__` defers the cross-import to first ACCESS instead of module-definition time, so
# `from nifty_quant.research.sweep_features import run_sweep` (one test suite's guess) works
# regardless of import order, without feature_sweep.py needing to duplicate any logic.
_REEXPORTED_FROM_FEATURE_SWEEP = frozenset(
    {"evaluate_promotion", "is_candidate", "measure_var_trial_sharpes", "run_sweep"}
)


def __getattr__(name: str) -> object:
    if name in _REEXPORTED_FROM_FEATURE_SWEEP:
        from nifty_quant.research import feature_sweep

        return getattr(feature_sweep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# `evaluate_promotion`/`is_candidate`/`measure_var_trial_sharpes`/`run_sweep` are resolvable
# via this module's `__getattr__` above but deliberately excluded from `__all__`: they are not
# bound names in this module's namespace (ruff F822 correctly flags a static `__all__` entry
# with no matching binding), and `feature_sweep.py` is their canonical, statically-checkable
# home per AMENDMENT 2.
__all__ = [
    "FEATURE_REGISTRY",
    "FEATURES",
    "FeatureSpec",
    "HORIZONS",
    "MIN_NAMES",
    "N_PLANNED_TRIALS",
    "n_planned_trials",
]
