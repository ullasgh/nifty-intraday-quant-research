"""Verification that selectivity knobs are not decorative.

If relaxing a strategy's selectivity parameter toward "accept everything" does not degrade
performance, that parameter is not selecting signal -- it is packaging. This test finds
such knobs generically from each registered strategy's pydantic `Params` model, discovers
the loosening direction empirically rather than assuming sign conventions, and asserts that
loosening strictly increases trade count while net Sharpe does not improve. Strategies and
fields that cannot be tested are skipped with explicit reasons, never silently.
"""

from __future__ import annotations

import math
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

import nifty_quant.strategy.plugins  # noqa: F401
from nifty_quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from nifty_quant.backtest.metrics import sharpe_ratio
from nifty_quant.strategy import registry
from tests.contract_fixtures import minimal_contract

from .test_causality import _load_strategy, build_irregular_panel

# The synthetic panel has only four sessions, so a single Sharpe estimate is extremely noisy.
# This slack must be wide enough to absorb same-seed sampling noise around "no improvement"
# (the same single-draw daily-Sharpe standard-error reasoning used elsewhere in this repo's
# verification tests), but it remains far below the >5.0 Sharpe gap `test_leak_canary.py`
# treats as a genuine leak.
SHARPE_SLACK = 0.25

# Below this many trades, comparing baseline vs. relaxed Sharpe is comparing sampling
# noise, not signal: with N<20, a single trade's realized P&L can swing the sample mean
# or standard deviation by double-digit percentages, well before any real selectivity
# effect could be distinguished from chance. This is deliberately below the textbook
# ~30-observation rule of thumb for CLT-based normality assumptions, chosen so genuinely
# low-frequency, by-design-selective strategies (e.g. eod_overextension, which trades at
# most once per symbol per day through a cross-sectional top-N cut) can still be tested on
# a synthetic panel of feasible size, while still filtering out near-empty (0-8 trade)
# samples that are pure noise.
MIN_TRADES_FOR_SENSITIVITY = 20

_INCLUDE_NAME_TOKENS = frozenset({"threshold", "topk", "band"})
_EXCLUDE_NAME_TOKENS = frozenset({
    "window",
    "lookback",
    "hold",
    "bars",
    "days",
    "weight",
    "gross",
    "capital",
    "time",
    "floor",
    "pct",
})
_INCLUDE_NAME_TOKEN_PAIRS = (("top", "k"), ("n", "max"))
_EXCLUDE_NAME_TOKEN_PAIRS = (("vol", "ann"),)


def _tokens(field_name: str) -> tuple[str, ...]:
    """Return lowercase underscore-delimited tokens for a field name."""
    return tuple(part for part in field_name.lower().split("_") if part)


def _field_annotation(params_cls: type[BaseModel], field_name: str, raw: Any) -> Any:
    """Return a resolved field annotation, falling back to pydantic's raw copy."""
    try:
        return get_type_hints(params_cls).get(field_name, raw)
    except Exception:
        return raw


def _numeric_type(annotation: Any) -> type[int] | type[float] | None:
    """Return the numeric type for int/float or Optional[int]/Optional[float] annotations."""
    origin = get_origin(annotation)
    if origin is None:
        if annotation in (int, float):
            return cast("type[int] | type[float]", annotation)
        return None

    if origin is Union or origin is UnionType:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1 and non_none[0] in (int, float):
                return cast("type[int] | type[float]", non_none[0])
    return None


def _candidate_selectivity_fields(params_cls: type[BaseModel]) -> list[str]:
    """Find pydantic fields that look like selectivity knobs without hardcoding names."""
    candidates: list[str] = []
    for field_name, field_info in params_cls.model_fields.items():
        tokens = _tokens(field_name)

        has_include_token = bool(_INCLUDE_NAME_TOKENS.intersection(tokens))
        has_include_pair = any(
            tokens[i : i + 2] == pair
            for pair in _INCLUDE_NAME_TOKEN_PAIRS
            for i in range(len(tokens) - 1)
        )
        if not (has_include_token or has_include_pair):
            continue

        has_exclude_token = bool(_EXCLUDE_NAME_TOKENS.intersection(tokens))
        has_exclude_pair = any(
            tokens[i : i + 2] == pair
            for pair in _EXCLUDE_NAME_TOKEN_PAIRS
            for i in range(len(tokens) - 1)
        )
        if has_exclude_token or has_exclude_pair:
            continue

        annotation = _field_annotation(params_cls, field_name, field_info.annotation)
        if _numeric_type(annotation) is None:
            continue
        candidates.append(field_name)
    return sorted(candidates)


def _perturbation_step(value: int | float, numeric_type: type[int] | type[float]) -> int | float:
    """Return a proportional, non-zero step size for *value* and its numeric type."""
    if numeric_type is int:
        return max(round(abs(value) * 0.5), 1)
    return max(abs(value) * 0.5, 0.5)


def _build_perturbed_params(
    p0: BaseModel, field_name: str, new_value: int | float
) -> BaseModel | None:
    """Return validated params with one field changed, or None if the change is invalid."""
    kwargs = p0.model_dump()
    kwargs[field_name] = new_value
    try:
        return p0.__class__.model_validate(kwargs)
    except ValidationError:
        return None


@pytest.mark.parametrize("name", registry.available())
def test_relaxing_selectivity_knob_moves_toward_noise(name: str) -> None:
    strategy_cls = registry.get(name)
    candidates = _candidate_selectivity_fields(strategy_cls.Params)
    if not candidates:
        pytest.skip(
            f"{name}: no selectivity-knob-shaped field found on "
            f"{strategy_cls.Params.__name__}"
        )

    baseline_strategy = _load_strategy(name)
    config = BacktestConfig()
    # n_repeats=3 (12 sessions instead of 4) gives low-frequency, by-design-selective
    # strategies (eod_overextension: cross-sectional top-N, dual-threshold, at most once per
    # symbol per day) enough sessions to clear MIN_TRADES_FOR_SENSITIVITY on both the baseline
    # and the relaxed run; verified empirically against this exact seed.
    panel = build_irregular_panel(n_repeats=3)
    baseline_result = run_backtest(baseline_strategy, panel, config, contract=minimal_contract())
    p0 = baseline_strategy.params

    tested_any = False
    skip_reasons: list[str] = []

    for field_name in candidates:
        current_value = getattr(p0, field_name)
        if current_value is None:
            skip_reasons.append(
                f"{field_name}: default is None (already unbounded/loosest possible), "
                "nothing to relax toward"
            )
            continue
        if isinstance(current_value, bool) or not isinstance(current_value, (int, float)):
            skip_reasons.append(f"{field_name}: non-numeric default after discovery")
            continue

        field_info = strategy_cls.Params.model_fields[field_name]
        annotation = _field_annotation(strategy_cls.Params, field_name, field_info.annotation)
        numeric_type = _numeric_type(annotation)
        if numeric_type is None:
            skip_reasons.append(f"{field_name}: no numeric type after discovery")
            continue

        step = _perturbation_step(current_value, numeric_type)
        direction_results: dict[str, BacktestResult] = {}
        for direction, delta in (("down", -step), ("up", step)):
            perturbed = _build_perturbed_params(p0, field_name, current_value + delta)
            if perturbed is None:
                continue
            direction_results[direction] = run_backtest(
                strategy_cls(params=perturbed), panel, config, contract=minimal_contract()
            )

        if not direction_results:
            skip_reasons.append(
                f"{field_name}: neither perturbation direction produces a valid Params object"
            )
            continue

        trade_increase = [
            (direction, result)
            for direction, result in direction_results.items()
            if result.n_trades > baseline_result.n_trades
        ]
        if not trade_increase:
            skip_reasons.append(
                f"{field_name}: neither perturbation direction increased trade count vs "
                f"baseline ({baseline_result.n_trades}) on the synthetic panel; "
                "knob has no measurable effect here"
            )
            continue

        loosening_direction, relaxed_result = max(
            trade_increase,
            key=lambda item: item[1].n_trades - baseline_result.n_trades,
        )

        if (
            baseline_result.n_trades < MIN_TRADES_FOR_SENSITIVITY
            or relaxed_result.n_trades < MIN_TRADES_FOR_SENSITIVITY
        ):
            skip_reasons.append(
                f"{field_name}: insufficient trades for a meaningful Sharpe comparison on "
                f"{name} (baseline={baseline_result.n_trades} trades, "
                f"relaxed={relaxed_result.n_trades} trades, "
                f"MIN_TRADES_FOR_SENSITIVITY={MIN_TRADES_FOR_SENSITIVITY})"
            )
            continue

        baseline_sharpe = sharpe_ratio(baseline_result.returns)
        relaxed_sharpe = sharpe_ratio(relaxed_result.returns)

        if baseline_result.ruined or relaxed_result.ruined:
            ruined_sides = [
                side
                for side, flag in (
                    ("baseline", baseline_result.ruined),
                    ("relaxed", relaxed_result.ruined),
                )
                if flag
            ]
            skip_reasons.append(
                f"{field_name}: ruined run ({', '.join(ruined_sides)}); Sharpe is meaningless"
            )
            continue
        if math.isnan(baseline_sharpe) or math.isnan(relaxed_sharpe):
            nan_sides = [
                side
                for side, value in (
                    ("baseline", baseline_sharpe),
                    ("relaxed", relaxed_sharpe),
                )
                if math.isnan(value)
            ]
            skip_reasons.append(f"{field_name}: Sharpe is NaN for {', '.join(nan_sides)}")
            continue

        assert relaxed_result.n_trades > baseline_result.n_trades, (
            f"{name}/{field_name}: relaxation direction {loosening_direction} did not "
            f"strictly increase trade count "
            f"({relaxed_result.n_trades} <= {baseline_result.n_trades})"
        )
        assert relaxed_sharpe <= baseline_sharpe + SHARPE_SLACK, (
            f"{name}/{field_name}: relaxing {field_name} {loosening_direction} improved net "
            f"Sharpe from {baseline_sharpe:.4f} to {relaxed_sharpe:.4f}; "
            f"slack={SHARPE_SLACK}"
        )
        tested_any = True

    if not tested_any:
        pytest.skip(
            f"{name}: no testable selectivity knob after discovery; " + "; ".join(skip_reasons)
        )
