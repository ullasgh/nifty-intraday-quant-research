"""Lightweight universe filters with no dependency on panel/data-loading code."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from nifty_quant.settings import MANIFEST_PATH


def _load_coverage() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    coverage = manifest.get("coverage", {})
    return coverage if isinstance(coverage, dict) else {}


def coverage_filter(
    symbols: Sequence[str], start: date, end: date, *, min_coverage: float = 0.98
) -> tuple[str, ...]:
    """Keep symbols whose manifest years cover at least min_coverage of the range."""
    if end < start:
        return ()
    total_years = end.year - start.year + 1
    if total_years <= 0:
        return ()  # pragma: no cover - date comparison invariant
        # datetime.date comparison is lexicographic on (year, month, day), and the earlier
        # if end < start: return () has already fired otherwise. So by this point end >=
        # start holds as full dates, which necessarily implies end.year >= start.year -- a
        # strictly smaller year would make the whole date smaller regardless of month/day.
        # Therefore total_years = end.year - start.year + 1 >= 1 always. Kept as a guard
        # against a future refactor that changes the date comparison logic.

    coverage = _load_coverage()
    kept: list[str] = []
    for symbol in symbols:
        entry = coverage.get(symbol)
        if not isinstance(entry, dict):
            continue
        raw_years = entry.get("years")
        if not isinstance(raw_years, list):
            continue
        symbol_years = {y for y in raw_years if isinstance(y, int)}
        covered = sum(
            1 for y in range(start.year, end.year + 1) if y in symbol_years
        )
        if (covered / total_years) >= min_coverage:
            kept.append(symbol)

    return tuple(kept)


def liquidity_filter(
    symbols: Sequence[str],
    adv_by_symbol: Mapping[str, float],
    *,
    min_adv_inr: float = 5e7,
) -> tuple[str, ...]:
    """Keep symbols whose ADV in INR is at least min_adv_inr.

    Symbols absent from `adv_by_symbol` are treated as unknown/zero liquidity
    and dropped. Result preserves input order.
    """
    kept: list[str] = []
    for symbol in symbols:
        adv = adv_by_symbol.get(symbol)
        if adv is not None and adv >= min_adv_inr:
            kept.append(symbol)
    return tuple(kept)


def combine(*masks: np.ndarray) -> np.ndarray:
    """Logical AND of any number of compatible boolean arrays.

    Raises ValueError if no masks are supplied or shapes do not match.
    """
    if not masks:
        raise ValueError("combine() requires at least one mask")

    shape = masks[0].shape
    if any(m.shape != shape for m in masks[1:]):
        raise ValueError("all masks must have the same shape")

    result = masks[0].astype(bool, copy=True)
    for mask in masks[1:]:
        np.logical_and(result, mask, out=result)
    return result
