from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nifty_quant.universe import static as static_module
from nifty_quant.universe.static import SurvivorshipReport, Universe, survivorship_report


@pytest.fixture
def report_missing() -> SurvivorshipReport:
    return SurvivorshipReport(
        universe_name="mini_universe",
        as_of_date=None,
        source="static",
        n_symbols=3,
        per_year={
            2018: {"active": 1, "missing": 2, "pct_active": 1 / 3},
            2019: {"active": 3, "missing": 0, "pct_active": 1.0},
        },
        missing_by_year={2018: ("B", "C"), 2019: ()},
    )


@pytest.fixture
def report_all_present() -> SurvivorshipReport:
    return SurvivorshipReport(
        universe_name="full_universe",
        as_of_date=None,
        source="static",
        n_symbols=3,
        per_year={
            2018: {"active": 3, "missing": 0, "pct_active": 1.0},
            2019: {"active": 3, "missing": 0, "pct_active": 1.0},
        },
        missing_by_year={2018: (), 2019: ()},
    )


@pytest.fixture
def fake_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_data = {
        "coverage": {
            "AAA": {"years": [2018, 2019, 2020]},
            "BBB": {"years": [2019, 2020]},
            "CCC": {"years": [2018, 2019, 2020]},
        }
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    monkeypatch.setattr(static_module, "MANIFEST_PATH", manifest_path)
    static_module._get_manifest.cache_clear()
    yield
    static_module._get_manifest.cache_clear()


def test_warning_line_missing_names_omits_survivorship_inflated(report_missing: SurvivorshipReport) -> None:
    warning = report_missing.warning_line()
    assert "survivorship-inflated" not in warning.lower()


def test_warning_line_missing_names_keeps_factual_counts(report_missing: SurvivorshipReport) -> None:
    warning = report_missing.warning_line()
    assert "UNIVERSE mini_universe" in warning
    assert "2 of 3 names had no data in 2018" in warning
    assert "later listing" in warning.lower()


def test_warning_line_all_present_unchanged(report_all_present: SurvivorshipReport) -> None:
    warning = report_all_present.warning_line()
    expected = "UNIVERSE full_universe WITH NO AS-OF DATE; 3 names; all 3 names had data in 2018."
    assert warning == expected


def test_warning_line_basis_with_as_of_date_set(report_missing: SurvivorshipReport) -> None:
    report = SurvivorshipReport(
        universe_name=report_missing.universe_name,
        as_of_date=date(2022, 6, 15),
        source=report_missing.source,
        n_symbols=report_missing.n_symbols,
        per_year=report_missing.per_year,
        missing_by_year=report_missing.missing_by_year,
    )
    warning = report.warning_line()
    assert "AS OF 2022-06-15" in warning
    assert "WITH NO AS-OF DATE" not in warning


def test_warning_line_basis_with_as_of_date_none(report_missing: SurvivorshipReport) -> None:
    warning = report_missing.warning_line()
    assert "WITH NO AS-OF DATE" in warning


def test_survivorship_report_consumer_regression(fake_manifest: None) -> None:
    universe = Universe(
        name="test_universe",
        symbols=("AAA", "BBB", "CCC"),
    )
    report = survivorship_report(universe, date(2018, 1, 1), date(2020, 1, 1))
    warning = report.warning_line()
    assert "survivorship-inflated" not in warning.lower()
    assert "UNIVERSE test_universe" in warning
