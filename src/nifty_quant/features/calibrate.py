"""Calibrate statistical-estimator thresholds against their own null distribution.

Thresholds on statistical estimators (e.g. a rolling Hurst exponent) must be set as
percentiles of the estimator's OWN null distribution at the SAME window length used in
production, not textbook absolutes copied from asymptotic theory. Measured evidence
(from `nifty_quant.features.persistence`, using the naive per-window lag-variance Hurst
estimator on a pure random walk with true H=0.5): at window=30, H = 0.090 +/- 0.165 (so a
`H > 0.55` filter can never fire - P(H>0.55) = 0.0%); at window=390 (`HURST_MIN_WINDOW`),
median H ~= 0.467 and p90 ~= 0.556, which makes `H > 0.55` a sensible ~90th-percentile cut.
The window was the bug, not the threshold - this module makes that finding reproducible on
demand via `calibration_report`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nifty_quant.features.persistence import fbm as _fbm

PERCENTILE_LEVELS: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
)

_REFERENCE_THRESHOLD: float = 0.55


@dataclass(frozen=True)
class NullCalibration:
    """Summary of a feature estimator's null distribution."""

    estimator: str
    window: int
    n_draws: int
    generator: str
    mean: float
    std: float
    percentiles: dict[float, float]

    def threshold_at(self, pct: float) -> float:
        level = float(pct)
        try:
            return self.percentiles[level]
        except KeyError:
            available = ", ".join(str(k) for k in sorted(self.percentiles))
            raise KeyError(
                f"Percentile {pct} not calibrated. Available levels: {available}"
            ) from None

    def p_value(
        self,
        observed: float,
        *,
        tail: Literal["right", "left"] = "right",
    ) -> float:
        if tail not in {"right", "left"}:
            raise ValueError("tail must be 'right' or 'left'")

        pairs = sorted(self.percentiles.items(), key=lambda item: item[1])
        if not pairs:
            raise ValueError("percentiles are empty")

        values: list[float] = []
        probs: list[float] = []
        for p, v in pairs:
            prob = p / 100.0
            if values and v == values[-1]:
                probs[-1] = max(probs[-1], prob)
            else:
                values.append(v)
                probs.append(prob)

        xp = np.asarray(values, dtype=np.float64)
        fp = np.asarray(probs, dtype=np.float64)

        cdf = float(np.interp(float(observed), xp, fp, left=fp[0], right=fp[-1]))
        if tail == "right":
            return float(np.clip(1.0 - cdf, 0.0, 1.0))
        return float(np.clip(cdf, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "window": self.window,
            "n_draws": self.n_draws,
            "generator": self.generator,
            "mean": self.mean,
            "std": self.std,
            "percentiles": {
                str(level): value for level, value in self.percentiles.items()
            },
        }


def _draw_null(
    estimator: Callable[[np.ndarray], float],
    window: int,
    *,
    n_draws: int,
    seed: int,
    generator: Literal["random_walk", "iid", "ar1", "fbm"],
    generator_kwargs: dict | None = None,
) -> np.ndarray:
    if window < 3:
        raise ValueError("window must be >= 3")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    if generator not in {"random_walk", "iid", "ar1", "fbm"}:
        raise ValueError(f"unknown generator: {generator}")

    rng = np.random.default_rng(seed)
    kwargs = dict(generator_kwargs or {})

    draws: list[float] = []
    for _ in range(n_draws):
        if generator == "random_walk":
            scale = float(kwargs.get("scale", 1.0))
            increments = rng.standard_normal(window) * scale
            log_price = np.cumsum(increments)
            path = np.exp(log_price)

        elif generator == "iid":
            scale = float(kwargs.get("scale", 1.0))
            path = np.exp(rng.standard_normal(window) * scale)

        elif generator == "ar1":
            phi = float(kwargs.get("phi", 0.0))
            scale = float(kwargs.get("scale", 1.0))
            eps = rng.standard_normal(window) * scale
            r = np.empty(window, dtype=np.float64)
            r[0] = eps[0]
            for i in range(1, window):
                r[i] = phi * r[i - 1] + eps[i]
            log_price = np.cumsum(r)
            path = np.exp(log_price)

        elif generator == "fbm":
            h = float(kwargs.get("H", 0.5))
            if not 0.0 < h < 1.0:
                raise ValueError("fbm H must be in (0, 1)")
            draw_seed = int(rng.integers(0, 2**31 - 1))
            log_price = _fbm(window, h, seed=draw_seed)
            path = np.exp(log_price)

        else:  # pragma: no cover - guarded above
            raise AssertionError("unreachable generator branch")

        value = float(estimator(path))
        if np.isfinite(value):
            draws.append(value)

    return np.asarray(draws, dtype=np.float64)


def calibrate_null(
    estimator: Callable[[np.ndarray], float],
    window: int,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    generator: Literal["random_walk", "iid", "ar1", "fbm"] = "random_walk",
    generator_kwargs: dict | None = None,
) -> NullCalibration:
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    if window < 3:
        raise ValueError("window must be >= 3")

    draws = _draw_null(
        estimator,
        window,
        n_draws=n_draws,
        seed=seed,
        generator=generator,
        generator_kwargs=generator_kwargs,
    )
    if draws.size == 0:
        raise ValueError("No finite draws were produced")

    percentiles = {
        level: float(np.percentile(draws, level)) for level in PERCENTILE_LEVELS
    }

    return NullCalibration(
        estimator=getattr(estimator, "__name__", "estimator"),
        window=window,
        n_draws=int(draws.size),
        generator=generator,
        mean=float(np.mean(draws)),
        std=float(np.std(draws, ddof=0)),
        percentiles=percentiles,
    )


def calibrate_empirical(values: np.ndarray) -> NullCalibration:
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite values")

    percentiles = {
        level: float(np.percentile(finite, level)) for level in PERCENTILE_LEVELS
    }

    return NullCalibration(
        estimator="empirical",
        window=0,
        n_draws=int(finite.size),
        generator="empirical",
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=0)),
        percentiles=percentiles,
    )


def suggest_threshold(
    estimator: Callable[[np.ndarray], float],
    window: int,
    target_fpr: float = 0.10,
    **kw,
) -> float:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")

    n_draws = int(kw.get("n_draws", 2000))
    seed = int(kw.get("seed", 0))
    generator = kw.get("generator", "random_walk")
    generator_kwargs = kw.get("generator_kwargs")

    draws = _draw_null(
        estimator,
        window,
        n_draws=n_draws,
        seed=seed,
        generator=generator,
        generator_kwargs=generator_kwargs,
    )
    if draws.size == 0:
        raise ValueError("No finite draws were produced")

    return float(np.quantile(draws, 1.0 - target_fpr))


def calibration_report(
    estimator_name: str,
    windows: Sequence[int],
    *,
    estimator: Callable[[np.ndarray], float],
    n_draws: int = 1000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for w in windows:
        cal = calibrate_null(
            estimator,
            w,
            n_draws=n_draws,
            seed=0,
            generator="random_walk",
        )
        draws = _draw_null(
            estimator,
            w,
            n_draws=n_draws,
            seed=0,
            generator="random_walk",
            generator_kwargs=None,
        )
        p_exceed = (
            float(np.mean(draws > _REFERENCE_THRESHOLD)) if draws.size else 0.0
        )

        rows.append(
            {
                "estimator": estimator_name,
                "window": w,
                "mean": cal.mean,
                "std": cal.std,
                "p90": cal.threshold_at(90),
                "p95": cal.threshold_at(95),
                "p_exceed_ref": p_exceed,
            }
        )

    columns = [
        "estimator",
        "window",
        "mean",
        "std",
        "p90",
        "p95",
        "p_exceed_ref",
    ]
    return pd.DataFrame(rows, columns=columns)
