"""Static universe definitions and survivorship diagnostics for nifty_quant."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import yaml

from nifty_quant.settings import BARS_1M, MANIFEST_PATH, REPO_ROOT


@lru_cache(maxsize=1)
def _get_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


INDEX_SYMBOLS: frozenset[str] = frozenset(
    {"NIFTY50", "NIFTY100", "NIFTYBANK", "INDIAVIX"}
)


@dataclass(frozen=True)
class Universe:
    name: str
    symbols: tuple[str, ...]
    as_of_date: date | None = None
    source: str = "static"

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbols

    def as_of(self, d: date) -> "Universe":
        """Point-in-time availability proxy for this universe.

        This method filters `self.symbols` to those whose coverage entry in
        ``data/MANIFEST.json`` contains `d` (inclusive on both ends, using the
        date parts of the ``first`` and ``last`` ISO timestamps).

        This is a **data availability** proxy, not real index membership. A
        symbol having bars on date `d` does not mean that it was actually a
        constituent of any index on that date, because this repository has no
        point-in-time membership history for NIFTY or any other index.
        Callers must not treat the result as historically accurate index
        membership.

        Symbols with no manifest coverage entry are dropped, as are symbols
        whose ``first``/``last`` timestamps cannot be parsed.
        """
        manifest = _get_manifest()
        coverage = manifest.get("coverage", {})
        if not isinstance(coverage, dict):
            coverage = {}

        kept: list[str] = []
        for symbol in self.symbols:
            entry = coverage.get(symbol)
            if not isinstance(entry, dict):
                continue
            first_raw = entry.get("first")
            last_raw = entry.get("last")
            if not isinstance(first_raw, str) or not isinstance(last_raw, str):
                continue
            try:
                first = datetime.fromisoformat(first_raw).date()
                last = datetime.fromisoformat(last_raw).date()
            except ValueError:
                continue
            if first <= d <= last:
                kept.append(symbol)

        return Universe(
            name=self.name,
            symbols=tuple(kept),
            as_of_date=d,
            source="availability_proxy",
        )

    def mask(self, all_symbols: Sequence[str]) -> np.ndarray:
        """Bool array aligned to all_symbols, True where that symbol is a member."""
        members = set(self.symbols)
        return np.fromiter(
            (s in members for s in all_symbols),
            dtype=bool,
            count=len(all_symbols),
        )


def all_data_symbols() -> tuple[str, ...]:
    """Sorted tuple of every symbol subdirectory name under BARS_1M.

    This is the de-facto universe: everything for which we have any bar data
    at all.
    """
    return tuple(sorted(p.name for p in BARS_1M.iterdir() if p.is_dir()))


def equity_symbols() -> tuple[str, ...]:
    """All data symbols minus the index symbols, sorted.

    Indices are not tradable instruments and must never enter a
    cross-sectional equity ranking.
    """
    return tuple(s for s in all_data_symbols() if s not in INDEX_SYMBOLS)


def load_universe(name: str) -> Universe:
    """Load a Universe from ``configs/universe/<name>.yaml``.

    If the file does not exist or its shape is invalid, fall back to the full
    equity universe with source ``all_data_symbols_fallback``.
    """
    path = REPO_ROOT / "configs" / "universe" / f"{name}.yaml"
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return Universe(
            name=name,
            symbols=equity_symbols(),
            source="all_data_symbols_fallback",
        )

    if not isinstance(data, dict) or not isinstance(data.get("symbols"), list):
        return Universe(
            name=name,
            symbols=equity_symbols(),
            source="all_data_symbols_fallback",
        )

    symbols = tuple(str(s) for s in data["symbols"])
    return Universe(name=str(data.get("name", name)), symbols=symbols)


@dataclass(frozen=True)
class SurvivorshipReport:
    universe_name: str
    as_of_date: date | None
    source: str
    n_symbols: int
    per_year: dict[int, dict[str, int | float]]
    missing_by_year: dict[int, tuple[str, ...]]

    def warning_line(self) -> str:
        """One-line summary of the worst-coverage year in this report."""
        if not self.per_year:
            return f"UNIVERSE {self.universe_name}: no survivorship data to report."

        worst_year = min(
            self.per_year,
            key=lambda y: self.per_year[y]["pct_active"],
        )
        info = self.per_year[worst_year]
        missing = int(info["missing"])
        total = self.n_symbols
        basis = (
            f"AS OF {self.as_of_date.isoformat()}"
            if self.as_of_date is not None
            else "WITH NO AS-OF DATE"
        )
        base = f"UNIVERSE {self.universe_name} {basis}; {total} names"

        if missing == 0:
            return f"{base}; all {total} names had data in {worst_year}."
        return (
            f"{base}; {missing} of {total} names had no data in {worst_year}. "
            f"Returns before {worst_year} are survivorship-inflated."
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-safe representation: dates as ISO strings, tuples as lists."""
        return {
            "universe_name": self.universe_name,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "source": self.source,
            "n_symbols": self.n_symbols,
            "per_year": self.per_year,
            "missing_by_year": {
                year: list(symbols)
                for year, symbols in self.missing_by_year.items()
            },
        }


def survivorship_report(
    universe: Universe, start: date, end: date
) -> SurvivorshipReport:
    """Cheap per-year coverage report using each symbol's manifest years list."""
    manifest = _get_manifest()
    coverage = manifest.get("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}

    symbol_year_sets: dict[str, set[int]] = {}
    for sym in universe.symbols:
        entry = coverage.get(sym)
        years: set[int] = set()
        if isinstance(entry, dict):
            raw_years = entry.get("years")
            if isinstance(raw_years, list):
                years = {y for y in raw_years if isinstance(y, int)}
        symbol_year_sets[sym] = years

    n_symbols = len(universe.symbols)
    per_year: dict[int, dict[str, int | float]] = {}
    missing_by_year: dict[int, tuple[str, ...]] = {}

    for year in range(start.year, end.year + 1):
        active = sum(
            1
            for sym in universe.symbols
            if year in symbol_year_sets.get(sym, set())
        )
        missing = n_symbols - active
        pct_active = active / n_symbols if n_symbols else 0.0
        per_year[year] = {
            "active": active,
            "missing": missing,
            "pct_active": pct_active,
        }
        missing_by_year[year] = tuple(
            sorted(
                sym
                for sym in universe.symbols
                if year not in symbol_year_sets.get(sym, set())
            )
        )

    return SurvivorshipReport(
        universe_name=universe.name,
        as_of_date=universe.as_of_date,
        source=universe.source,
        n_symbols=n_symbols,
        per_year=per_year,
        missing_by_year=missing_by_year,
    )
