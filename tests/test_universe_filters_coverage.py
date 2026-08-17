"""Additional coverage-focused tests for nifty_quant.universe.filters.

This file intentionally targets branches not covered by tests/test_universe.py.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nifty_quant.universe import filters as filters_module


def _write_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(filters_module, "MANIFEST_PATH", manifest_path)


def test_coverage_filter_end_before_start_returns_empty() -> None:
    result = filters_module.coverage_filter(
        ("S1",), date(2021, 1, 1), date(2020, 1, 1)
    )

    assert result == ()


def test_coverage_filter_skips_missing_malformed_and_keeps_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "coverage": {
            "BAD_ENTRY": "oops",
            "BAD_YEARS": {"years": "2020"},
            "GOOD": {"years": [2021, 2022]},
        }
    }
    _write_manifest(tmp_path, monkeypatch, payload)

    result = filters_module.coverage_filter(
        ("MISSING", "BAD_ENTRY", "BAD_YEARS", "GOOD"),
        date(2021, 1, 1),
        date(2022, 1, 1),
        min_coverage=1.0,
    )

    assert result == ("GOOD",)


def test_liquidity_filter_drops_symbol_missing_from_adv() -> None:
    symbols = ("S1", "MISSING", "S2")
    adv_by_symbol = {"S1": 100.0, "S2": 200.0}

    result = filters_module.liquidity_filter(
        symbols, adv_by_symbol, min_adv_inr=50.0
    )

    assert result == ("S1", "S2")
