"""Causality coverage for the float64 `rolling_beta` FAST PATH, and a power check on
the `@causal` probe budget that protects it.

Why this file exists. `rolling_beta` has a dtype gate: float64 input takes
`_beta_session_vectorized`, anything else delegates to `_rolling_beta_reference`. Every
causality fixture in `tests/test_market_features.py` is float32 (`_make_returns` /
`_make_market` both end in `.astype(np.float32)`), so `test_rolling_beta_is_causal` and
`test_beta_residual_return_is_causal` exercise the REFERENCE path only -- while
`sweep_features._returns_and_market` yields float64, so production runs the fast path.
Verified by spying on `_beta_session_vectorized`: 0 calls for the float32 fixtures, 3
(one per session) for float64. A real lookahead injected into the fast path left the
whole suite green.

The float32 fixtures are deliberately NOT changed -- they legitimately pin the reference
path. This file adds the missing float64 half, so both paths stay covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant import guards
from nifty_quant.features import market
from nifty_quant.features.market import beta_residual_return, rolling_beta
from nifty_quant.guards import ContractViolation, Strictness, strictness

N_ROWS = 30
N_SYMBOLS = 3
WINDOW = 5
MIN_COUNT = 3


def _day_offsets() -> np.ndarray:
    """Three 10-row sessions -- explicit offsets, never a fixed stride (rule 5)."""
    return np.array([0, 10, 20, 30], dtype=np.int32)


def _returns64(seed: int) -> np.ndarray:
    """float64 returns, matching what `sweep_features._returns_and_market` hands over."""
    return np.random.default_rng(seed).normal(0.0, 0.001, size=(N_ROWS, N_SYMBOLS))


def _market64(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.001, size=N_ROWS)


def _leaky_session_beta(original):
    """The exact lookahead that the float32-only fixtures failed to catch.

    Fills each NaN warm-up slot from the row BELOW it, so output row t depends on input
    row t+1. Applied to `_beta_session_vectorized`'s return, per session.
    """

    def wrapper(
        ret_session: np.ndarray,
        mkt_session: np.ndarray,
        window: int,
        resolved_min_count: int,
    ) -> np.ndarray:
        out = original(ret_session, mkt_session, window, resolved_min_count)
        out[:-1] = np.where(np.isnan(out[:-1]), out[1:], out[:-1])
        return out

    return wrapper


# ---------------------------------------------------------------------------
# Defect 1: the fast path is reached, and is causal.
# ---------------------------------------------------------------------------


def test_float64_input_takes_the_vectorized_fast_path() -> None:
    """Pin the dtype gate itself, so the fixtures below cannot silently stop covering it.

    Without this, a future dtype change would quietly route the float64 causality tests
    back through the reference path and the fast path would lose coverage again with no
    test turning red -- which is precisely how the gap arose.
    """
    calls: list[int] = []
    original = market._beta_session_vectorized

    def spy(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    market._beta_session_vectorized = spy  # type: ignore[assignment]
    try:
        rolling_beta(
            _returns64(10).astype(np.float32),
            _market64(11).astype(np.float32),
            WINDOW,
            day_offsets=_day_offsets(),
            min_count=MIN_COUNT,
        )
        assert calls == [], "float32 must delegate to the reference path"

        rolling_beta(
            _returns64(10),
            _market64(11),
            WINDOW,
            day_offsets=_day_offsets(),
            min_count=MIN_COUNT,
        )
        assert len(calls) == 3, "float64 must take the fast path, once per session"
    finally:
        market._beta_session_vectorized = original  # type: ignore[assignment]


def test_rolling_beta_float64_is_causal() -> None:
    with strictness(Strictness.FULL):
        result = rolling_beta(
            _returns64(10),
            _market64(11),
            WINDOW,
            day_offsets=_day_offsets(),
            min_count=MIN_COUNT,
        )
    assert result.shape == (N_ROWS, N_SYMBOLS)
    assert result.dtype == np.float64


def test_beta_residual_return_float64_is_causal() -> None:
    with strictness(Strictness.FULL):
        result = beta_residual_return(
            _returns64(12),
            _market64(13),
            WINDOW,
            day_offsets=_day_offsets(),
        )
    assert result.shape == (N_ROWS, N_SYMBOLS)


# ---------------------------------------------------------------------------
# Defect 1, the part that matters: the guard must FAIL on an injected lookahead.
# A causality test that cannot fail on injected lookahead is worse than none.
# ---------------------------------------------------------------------------


def test_injected_fast_path_lookahead_is_detected_exhaustively() -> None:
    """Deterministic power check: with every cut probed, the leak is always caught.

    Uses `n_probes = N_ROWS - 1`, which `cut_count` clamps to the full candidate set, so
    detection is a certainty rather than a rate -- no seed dependence, nothing to tune.
    This is the assertion that would have caught the injected lookahead outright.
    """
    original = market._beta_session_vectorized
    market._beta_session_vectorized = _leaky_session_beta(original)  # type: ignore[assignment]
    try:
        probed = guards.causal(row_arg=0, n_probes=N_ROWS - 1, seed=0)(
            rolling_beta.__wrapped__
        )
        with strictness(Strictness.FULL):
            with pytest.raises(ContractViolation, match="causal violation"):
                probed(
                    _returns64(10),
                    _market64(11),
                    WINDOW,
                    day_offsets=_day_offsets(),
                    min_count=MIN_COUNT,
                )
    finally:
        market._beta_session_vectorized = original  # type: ignore[assignment]


def test_clean_fast_path_survives_exhaustive_probing() -> None:
    """The companion no-leak control: exhaustive probing must NOT fire on clean code.

    Without this, the test above would also pass if `@causal` raised unconditionally.
    """
    probed = guards.causal(row_arg=0, n_probes=N_ROWS - 1, seed=0)(
        rolling_beta.__wrapped__
    )
    with strictness(Strictness.FULL):
        result = probed(
            _returns64(10),
            _market64(11),
            WINDOW,
            day_offsets=_day_offsets(),
            min_count=MIN_COUNT,
        )
    assert result.shape == (N_ROWS, N_SYMBOLS)


# ---------------------------------------------------------------------------
# Defect 2: the DEFAULT probe budget must have real power against this leak.
# ---------------------------------------------------------------------------

# Seeds and bound, per CLAUDE.md rule 9. A single-seed "the guard caught it" assertion is
# the flaky coverage-check shape rule 9 bans: at the default's true detection rate it
# would fail ~1 run in 10 on correct code. So this asserts a RATE across many seeds, at a
# large effect-to-SE ratio.
#
# The leak geometry is enumerated exactly (see the derivation on `guards.causal`): the
# detecting cuts are {1, 11, 21}, 3 of 29 candidates, so the true detection rate at the
# default m=15 is the hypergeometric 1 - C(26,15)/C(29,15) = 0.9004 (measured 0.925 over
# 120 seeds). With 60 seeds the binomial SE is sqrt(.9*.1/60) = 0.039, so a floor of 0.70
# sits 5.2 SE below the true rate -- effectively deterministic, and it still fails
# overwhelmingly at the old default of 3, whose true rate is 0.2885 (10.7 SE ABOVE its
# own mean would be needed to reach 0.70). Neither the seed count nor the floor was
# tuned to make this pass: both are read off the closed form above.
DETECTION_SEEDS = 60
DETECTION_FLOOR = 0.70


def _detection_rate(n_probes: int, n_seeds: int = DETECTION_SEEDS) -> float:
    """Fraction of guard seeds at which the injected fast-path lookahead is caught."""
    original = market._beta_session_vectorized
    market._beta_session_vectorized = _leaky_session_beta(original)  # type: ignore[assignment]
    returns = _returns64(10)
    market_returns = _market64(11)
    day_offsets = _day_offsets()
    hits = 0
    try:
        with strictness(Strictness.FULL):
            for seed in range(n_seeds):
                probed = guards.causal(row_arg=0, n_probes=n_probes, seed=seed)(
                    rolling_beta.__wrapped__
                )
                try:
                    probed(
                        returns,
                        market_returns,
                        WINDOW,
                        day_offsets=day_offsets,
                        min_count=MIN_COUNT,
                    )
                except ContractViolation:
                    hits += 1
    finally:
        market._beta_session_vectorized = original  # type: ignore[assignment]
    return hits / n_seeds


def test_default_n_probes_detects_the_localised_leak_at_rate() -> None:
    """The derived default must have real power against the leak it was derived from.

    This is the regression that pins CLAUDE.md rule 8 for `n_probes`: lowering the
    default back toward 3 drops the measured rate under the floor and turns this red.
    """
    default_probes = guards.causal.__kwdefaults__["n_probes"]
    rate = _detection_rate(default_probes)
    assert rate >= DETECTION_FLOOR, (
        f"@causal default n_probes={default_probes} detected the localised "
        f"fast-path lookahead in only {rate:.3f} of {DETECTION_SEEDS} guard seeds "
        f"(floor {DETECTION_FLOOR})"
    )


def test_detection_rate_is_monotone_in_probe_budget() -> None:
    """More probes must not detect less -- the curve the default was read off.

    Guards against a probe-selection change that spends budget without buying power (the
    measured failure mode of the rejected deterministic-hybrid strategy). Compares a
    single wide spread, not per-point coverage, per rule 9.
    """
    low = _detection_rate(2)
    high = _detection_rate(20)
    assert high > low + 0.30, (
        f"detection rate barely moved with the probe budget: "
        f"m=2 -> {low:.3f}, m=20 -> {high:.3f}"
    )
