from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nifty_quant.settings import MANIFEST_PATH, PANEL_VERSION


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    years: tuple[int, ...]
    rows: int
    first: datetime.datetime
    last: datetime.datetime


@dataclass(frozen=True)
class Manifest:
    resolution: str
    written_at: datetime.datetime
    n_symbols: int
    total_rows: int
    adjustments: str
    fingerprint: str
    coverage: Mapping[str, SymbolCoverage]

    @classmethod
    def load(cls, path: Path | None = None) -> "Manifest":
        """Load and parse data/MANIFEST.json."""
        if path is None:
            path = MANIFEST_PATH
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        coverage = {
            sym: SymbolCoverage(
                symbol=sym,
                years=tuple(info["years"]),
                rows=info["rows"],
                first=datetime.datetime.fromisoformat(info["first"]),
                last=datetime.datetime.fromisoformat(info["last"]),
            )
            for sym, info in raw["coverage"].items()
        }

        return cls(
            resolution=raw["resolution"],
            written_at=datetime.datetime.fromisoformat(raw["written_at"]),
            n_symbols=raw["symbols"],
            total_rows=raw["total_rows"],
            adjustments=raw["adjustments"],
            fingerprint=raw["fingerprint"],
            coverage=coverage,
        )

    def symbols(self) -> tuple[str, ...]:
        """Sorted tuple of all symbol names in coverage."""
        return tuple(sorted(self.coverage))

    def symbols_covering(
        self, start: datetime.date, end: datetime.date
    ) -> tuple[str, ...]:
        """Sorted tuple of symbols whose coverage overlaps [start, end]."""
        return tuple(
            sorted(
                sym
                for sym, cov in self.coverage.items()
                if cov.first.date() <= end and cov.last.date() >= start
            )
        )

    def cache_key(self) -> str:
        """Cache-invalidation key from fingerprint, adjustments, resolution, and panel version."""
        key = f"{self.fingerprint}|{self.adjustments}|{self.resolution}|{PANEL_VERSION}"
        return hashlib.blake2s(key.encode("ascii"), digest_size=8).hexdigest()
