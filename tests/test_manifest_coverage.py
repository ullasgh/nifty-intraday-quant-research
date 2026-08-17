import datetime
import json
from dataclasses import replace
from pathlib import Path

import pytest

import nifty_quant.data.manifest as manifest_module
from nifty_quant.data.manifest import Manifest, SymbolCoverage


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _sym(
    symbol: str,
    years: tuple[int, ...],
    rows: int,
    first: datetime.datetime,
    last: datetime.datetime,
) -> SymbolCoverage:
    return SymbolCoverage(
        symbol=symbol,
        years=years,
        rows=rows,
        first=first,
        last=last,
    )


def test_load_explicit_path_reads_manifest(tmp_path: Path) -> None:
    path = tmp_path / "custom" / "MANIFEST.json"
    payload = {
        "resolution": "1",
        "written_at": "2024-01-01T00:00:00+05:30",
        "symbols": 2,
        "total_rows": 100,
        "adjustments": "none",
        "fingerprint": "fp_abc",
        "coverage": {
            "AAA": {
                "years": [2024],
                "rows": 50,
                "first": "2024-01-01T09:15:00+05:30",
                "last": "2024-01-02T15:30:00+05:30",
            },
            "BBB": {
                "years": [2024],
                "rows": 50,
                "first": "2024-01-01T09:15:00+05:30",
                "last": "2024-01-02T15:30:00+05:30",
            },
        },
    }
    _write_manifest(path, payload)

    manifest = Manifest.load(path=path)

    assert manifest.resolution == "1"
    assert manifest.n_symbols == 2
    assert manifest.total_rows == 100
    assert manifest.adjustments == "none"
    assert manifest.fingerprint == "fp_abc"
    assert set(manifest.coverage) == {"AAA", "BBB"}
    assert manifest.coverage["AAA"].years == (2024,)
    assert manifest.coverage["AAA"].rows == 50
    assert manifest.coverage["AAA"].first == datetime.datetime.fromisoformat(
        "2024-01-01T09:15:00+05:30"
    )
    assert manifest.coverage["AAA"].last == datetime.datetime.fromisoformat(
        "2024-01-02T15:30:00+05:30"
    )


def test_load_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        Manifest.load(path=missing)
    assert str(missing) in str(exc_info.value)


def test_symbols_returns_sorted_tuple() -> None:
    cov_a = _sym(
        "AAA",
        (2024,),
        10,
        datetime.datetime(2024, 1, 1, 9, 15),
        datetime.datetime(2024, 1, 1, 9, 20),
    )
    cov_m = _sym(
        "MMM",
        (2024,),
        10,
        datetime.datetime(2024, 1, 1, 9, 15),
        datetime.datetime(2024, 1, 1, 9, 20),
    )
    cov_z = _sym(
        "ZEE",
        (2024,),
        10,
        datetime.datetime(2024, 1, 1, 9, 15),
        datetime.datetime(2024, 1, 1, 9, 20),
    )
    manifest = Manifest(
        resolution="1",
        written_at=datetime.datetime(2024, 1, 1, 12, 0),
        n_symbols=3,
        total_rows=30,
        adjustments="none",
        fingerprint="fp",
        coverage={"ZEE": cov_z, "AAA": cov_a, "MMM": cov_m},
    )
    assert manifest.symbols() == ("AAA", "MMM", "ZEE")


def test_symbols_covering_boundaries_and_exclusions() -> None:
    start = datetime.date(2024, 6, 10)
    end = datetime.date(2024, 6, 20)

    sym_before = _sym(
        "SYM_BEFORE",
        (2024,),
        5,
        datetime.datetime(2024, 1, 1, 9, 15),
        datetime.datetime(2024, 1, 31, 15, 30),
    )
    sym_after = _sym(
        "SYM_AFTER",
        (2024,),
        5,
        datetime.datetime(2024, 12, 1, 9, 15),
        datetime.datetime(2024, 12, 31, 15, 30),
    )
    sym_overlap = _sym(
        "SYM_OVERLAP",
        (2024,),
        5,
        datetime.datetime(2024, 6, 1, 9, 15),
        datetime.datetime(2024, 6, 30, 15, 30),
    )
    sym_first_eq_end = _sym(
        "SYM_FIRST_EQ_END",
        (2024,),
        5,
        datetime.datetime(2024, 6, 20, 9, 15),
        datetime.datetime(2024, 6, 25, 15, 30),
    )
    sym_last_eq_start = _sym(
        "SYM_LAST_EQ_START",
        (2024,),
        5,
        datetime.datetime(2024, 6, 1, 9, 15),
        datetime.datetime(2024, 6, 10, 15, 30),
    )
    sym_off_by_one = _sym(
        "SYM_OFFBYONE",
        (2024,),
        5,
        datetime.datetime(2024, 6, 1, 9, 15),
        datetime.datetime(2024, 6, 9, 15, 30),
    )

    manifest = Manifest(
        resolution="1",
        written_at=datetime.datetime(2024, 1, 1, 12, 0),
        n_symbols=6,
        total_rows=30,
        adjustments="none",
        fingerprint="fp",
        coverage={
            "SYM_OVERLAP": sym_overlap,
            "SYM_BEFORE": sym_before,
            "SYM_AFTER": sym_after,
            "SYM_FIRST_EQ_END": sym_first_eq_end,
            "SYM_LAST_EQ_START": sym_last_eq_start,
            "SYM_OFFBYONE": sym_off_by_one,
        },
    )

    result = manifest.symbols_covering(start, end)
    assert result == (
        "SYM_FIRST_EQ_END",
        "SYM_LAST_EQ_START",
        "SYM_OVERLAP",
    )


def test_load_malformed_json_raises_json_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        Manifest.load(path=path)


def test_load_missing_coverage_key_raises_keyerror(tmp_path: Path) -> None:
    path = tmp_path / "missing_coverage.json"
    payload = {
        "resolution": "1",
        "written_at": "2024-01-01T00:00:00+05:30",
        "symbols": 0,
        "total_rows": 0,
        "adjustments": "none",
        "fingerprint": "fp",
    }
    _write_manifest(path, payload)
    with pytest.raises(KeyError) as exc_info:
        Manifest.load(path=path)
    assert exc_info.value.args[0] == "coverage"


def test_cache_key_deterministic_and_depends_only_on_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cov_base = _sym(
        "AAA",
        (2024,),
        10,
        datetime.datetime(2024, 1, 1, 9, 15),
        datetime.datetime(2024, 1, 1, 9, 20),
    )
    base = Manifest(
        resolution="1",
        written_at=datetime.datetime(2024, 1, 1, 12, 0),
        n_symbols=1,
        total_rows=10,
        adjustments="none",
        fingerprint="fp_base",
        coverage={"AAA": cov_base},
    )
    base_key = base.cache_key()

    # Coverage does NOT affect cache_key: it is derived only from fingerprint,
    # adjustments, resolution, and PANEL_VERSION per the docstring.
    cov_other = _sym(
        "AAA",
        (2025,),
        20,
        datetime.datetime(2025, 1, 1, 9, 15),
        datetime.datetime(2025, 1, 1, 9, 20),
    )
    changed_coverage = replace(base, coverage={"AAA": cov_other})
    assert changed_coverage.cache_key() == base_key

    # Fingerprint changes key.
    changed_fingerprint = replace(base, fingerprint="fp_other")
    assert changed_fingerprint.cache_key() != base_key

    # Adjustments changes key.
    changed_adjustments = replace(base, adjustments="adjusted")
    assert changed_adjustments.cache_key() != base_key

    # Resolution changes key.
    changed_resolution = replace(base, resolution="2")
    assert changed_resolution.cache_key() != base_key

    # PANEL_VERSION changes key.
    monkeypatch.setattr(manifest_module, "PANEL_VERSION", 999)
    assert base.cache_key() != base_key

    # Determinism: same manifest, same key on repeated calls.
    monkeypatch.undo()
    assert base.cache_key() == base_key
