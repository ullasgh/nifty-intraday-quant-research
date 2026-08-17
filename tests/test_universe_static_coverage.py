"""Coverage-gap tests for static universe definitions in nifty_quant."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nifty_quant.universe import static as static_module
from nifty_quant.universe.static import (
    Universe,
    equity_symbols,
    load_universe,
    survivorship_report,
)


@pytest.fixture()
def fake_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _write(payload: dict) -> Path:
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(static_module, "MANIFEST_PATH", manifest_path)
        static_module._get_manifest.cache_clear()
        return manifest_path

    yield _write
    static_module._get_manifest.cache_clear()


def test_universe_len_matches_symbols() -> None:
    universe = Universe(name="len_test", symbols=("A", "B", "C"))
    assert len(universe) == 3
    assert len(universe) == len(universe.symbols)


def test_universe_contains_checks_membership() -> None:
    universe = Universe(name="contains_test", symbols=("A", "B"))
    assert "A" in universe
    assert "X" not in universe


def test_as_of_coverage_not_dict_drops_all(fake_manifest) -> None:
    fake_manifest({"coverage": "not-a-dict"})
    universe = Universe(name="as_of_fallback", symbols=("A", "B"))
    as_of = universe.as_of(date(2020, 1, 1))

    assert as_of.symbols == ()
    assert as_of.source == "availability_proxy"
    assert as_of.as_of_date == date(2020, 1, 1)


def test_as_of_drops_broken_symbols_keeps_valid(fake_manifest) -> None:
    fake_manifest(
        {
            "coverage": {
                "VALID": {
                    "first": "2019-01-01T00:00:00",
                    "last": "2021-01-01T00:00:00",
                },
                "NOT_DICT": "oops",
                "INT_TS": {"first": 2019, "last": 2021},
                "BAD_ISO": {
                    "first": "not-a-date",
                    "last": "2021-01-01T00:00:00",
                },
            }
        }
    )
    universe = Universe(
        name="broken_entries",
        symbols=("VALID", "ABSENT", "NOT_DICT", "INT_TS", "BAD_ISO"),
    )

    as_of = universe.as_of(date(2020, 1, 1))

    assert as_of.symbols == ("VALID",)


def test_load_universe_missing_file_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(static_module, "REPO_ROOT", tmp_path)

    loaded = load_universe("does_not_exist")

    assert loaded.source == "all_data_symbols_fallback"
    assert loaded.symbols == equity_symbols()
    assert loaded.name == "does_not_exist"


def test_load_universe_malformed_yaml_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs" / "universe"
    config_dir.mkdir(parents=True)
    (config_dir / "bad_yaml.yaml").write_text("a: [1, 2", encoding="utf-8")
    monkeypatch.setattr(static_module, "REPO_ROOT", tmp_path)

    loaded = load_universe("bad_yaml")

    assert loaded.source == "all_data_symbols_fallback"
    assert loaded.symbols == equity_symbols()


def test_load_universe_top_level_list_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs" / "universe"
    config_dir.mkdir(parents=True)
    (config_dir / "list.yaml").write_text("- A\n- B\n", encoding="utf-8")
    monkeypatch.setattr(static_module, "REPO_ROOT", tmp_path)

    loaded = load_universe("list")

    assert loaded.source == "all_data_symbols_fallback"
    assert loaded.symbols == equity_symbols()


def test_load_universe_missing_symbols_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs" / "universe"
    config_dir.mkdir(parents=True)
    (config_dir / "missing_symbols.yaml").write_text(
        "name: test\n", encoding="utf-8"
    )
    monkeypatch.setattr(static_module, "REPO_ROOT", tmp_path)

    loaded = load_universe("missing_symbols")

    assert loaded.source == "all_data_symbols_fallback"
    assert loaded.symbols == equity_symbols()


def test_load_universe_symbols_not_list_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs" / "universe"
    config_dir.mkdir(parents=True)
    (config_dir / "symbols_not_list.yaml").write_text(
        "symbols: A\n", encoding="utf-8"
    )
    monkeypatch.setattr(static_module, "REPO_ROOT", tmp_path)

    loaded = load_universe("symbols_not_list")

    assert loaded.source == "all_data_symbols_fallback"
    assert loaded.symbols == equity_symbols()


def test_warning_line_empty_per_year(fake_manifest) -> None:
    fake_manifest({"coverage": {}})
    universe = Universe(name="empty_report", symbols=("A",))

    report = survivorship_report(
        universe, date(2023, 1, 1), date(2020, 1, 1)
    )

    assert report.per_year == {}
    assert report.warning_line() == (
        "UNIVERSE empty_report: no survivorship data to report."
    )


def test_warning_line_all_covered_no_inflation(fake_manifest) -> None:
    fake_manifest(
        {
            "coverage": {
                "A": {"years": [2020, 2021]},
                "B": {"years": [2020, 2021]},
            }
        }
    )
    universe = Universe(name="covered", symbols=("A", "B"))

    report = survivorship_report(
        universe, date(2020, 1, 1), date(2021, 1, 1)
    )
    warning = report.warning_line()

    assert "all 2 names had data in 2020." in warning
    assert "survivorship-inflated" not in warning


def test_survivorship_report_to_dict_json_safe(fake_manifest) -> None:
    fake_manifest(
        {
            "coverage": {
                "A": {"years": [2020]},
                "B": {"years": [2020]},
            }
        }
    )

    report_none = survivorship_report(
        Universe(name="none_date", symbols=("A", "B")),
        date(2020, 1, 1),
        date(2020, 1, 1),
    )
    result_none = report_none.to_dict()
    assert result_none["as_of_date"] is None
    assert all(
        isinstance(v, list) for v in result_none["missing_by_year"].values()
    )

    report_dated = survivorship_report(
        Universe(
            name="dated",
            symbols=("A",),
            as_of_date=date(2020, 1, 1),
        ),
        date(2020, 1, 1),
        date(2020, 1, 1),
    )
    result_dated = report_dated.to_dict()
    assert result_dated["as_of_date"] == "2020-01-01"
    assert isinstance(result_dated["as_of_date"], str)
    assert all(
        isinstance(v, list) for v in result_dated["missing_by_year"].values()
    )


def test_survivorship_report_coverage_not_dict_all_missing(
    fake_manifest,
) -> None:
    fake_manifest({"coverage": "not-a-dict"})
    universe = Universe(name="bad_coverage", symbols=("A", "B"))

    report = survivorship_report(
        universe, date(2020, 1, 1), date(2020, 1, 1)
    )

    assert report.per_year[2020]["active"] == 0
    assert report.per_year[2020]["missing"] == 2
    assert report.missing_by_year[2020] == ("A", "B")


def test_survivorship_report_broken_years_entries(fake_manifest) -> None:
    fake_manifest(
        {
            "coverage": {
                "GOOD": {"years": [2020]},
                "YEARS_BAD": {"years": "not-list"},
            }
        }
    )
    universe = Universe(name="broken_years", symbols=("GOOD", "ABSENT", "YEARS_BAD"))

    report = survivorship_report(
        universe, date(2020, 1, 1), date(2020, 1, 1)
    )

    assert report.per_year[2020]["active"] == 1
    assert report.per_year[2020]["missing"] == 2
    assert report.missing_by_year[2020] == ("ABSENT", "YEARS_BAD")
