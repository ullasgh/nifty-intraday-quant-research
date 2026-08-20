"""Size embargoes over the full train/test dependence horizon (feature lookback +
label horizon + holding period + execution lag)."""

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class EmbargoComponents:
    feature_lookback: float = 0.0
    label_horizon: float = 0.0
    holding_period: float = 0.0
    execution_horizon: float = 0.0

    def required_sessions(self) -> float:
        """Sum of the four components, in sessions. Not rounded here -- rounding
        happens exactly once, at the point a caller compares against an integer
        embargo budget (see `required_embargo_sessions`)."""
        return float(
            self.feature_lookback
            + self.label_horizon
            + self.holding_period
            + self.execution_horizon
        )

    def dominant_term(self) -> str:
        """Name of whichever component currently holds the largest value."""
        return max(
            fields(self),
            key=lambda field: getattr(self, field.name),
        ).name


def effective_memory_sessions(a: float) -> float:
    """Effective memory, in sessions, of an EMA with decay weight `a`.

    For `w_t = (1-a) w_{t-1} + a * target_t`, the memory is `1/a` sessions to a
    factor. Raises ValueError for `a <= 0` or `a > 1` (undefined/infinite memory
    must never be silently returned as 0 or inf).
    """
    if not 0 < a <= 1:
        raise ValueError(f"invalid EMA parameter a={a!r}; expected 0 < a <= 1")
    return 1.0 / a


# Alias: a second caller of this module independently expects this name for the
# same concept. Both resolve to the identical function object.
ewma_effective_memory_sessions = effective_memory_sessions


def bars_to_sessions(n_bars: int, day_offsets: np.ndarray) -> int:
    """Smallest `m` such that the LAST `m` sessions (walking backward from the end
    of `day_offsets`) together cover at least `n_bars` bars.

    `day_offsets` is a sorted array of row indices marking each session's start,
    with `day_offsets[-1]` equal to the total row count. Never assumes a fixed
    bars-per-session stride -- session lengths are read from `day_offsets` (rule 5).
    """
    if n_bars <= 0:
        return 0

    day_offsets = np.asarray(day_offsets)
    session_lengths = np.diff(day_offsets)[::-1]
    n_sessions = len(session_lengths)
    if n_sessions == 0:
        return 0

    cumulative_bars = np.cumsum(session_lengths)
    required = np.searchsorted(cumulative_bars, n_bars, side="left") + 1
    return min(int(required), n_sessions)


def required_embargo_sessions(
    feature_lookback: float,
    label_horizon: float,
    holding_period: float,
    execution_horizon: float,
) -> int:
    """math.ceil of the sum of the four components, rounded ONCE at the end (never
    per-component, which would over-count by up to one session per term)."""
    return math.ceil(
        feature_lookback
        + label_horizon
        + holding_period
        + execution_horizon
    )
