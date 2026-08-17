"""Additional test coverage for src/nifty_quant/data/panel_builder.py.

Targets the gaps left by tests/test_panel.py and tests/test_panel_coverage.py
(which exercise panel_builder.build_panel only incidentally, via
nifty_quant.data.panel.load_panel). This file drives panel_builder's public
surface (build_panel, gc_orphans, panel_cache_dir, is_cached) and its small
private helpers (_available_years, _read_symbol_arrays, _read_symbols_json)
directly, with synthetic parquet trees under tmp_path.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nifty_quant import settings
from nifty_quant.data import panel_builder
from nifty_quant.data.manifest import Manifest
from nifty_quant.guards import ContractViolation

FIELDS = ("open", "high", "low", "close", "volume")


def _session_ts(
    session_date: datetime.date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> list[int]:
    """Return left-labelled epoch-second UTC timestamps for an IST wall-clock start."""
    local_start = datetime.datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        start_hhmm[0],
        start_hhmm[1],
    )
    utc_naive = local_start - datetime.timedelta(hours=5, minutes=30)
    epoch0 = int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())
    return [epoch0 + 60 * i for i in range(n_bars)]


def _session_rows(
    session_date: datetime.date, n_bars: int, price0: float = 100.0
) -> list[tuple[int, float, float, float, float, float]]:
    """Deterministic OHLCV rows for a synthetic session, in chronological order."""
    return [
        (ts, price0 + i, price0 + 1.0 + i, price0 - 1.0 + i, price0 + 0.5 + i, 1000.0 + i)
        for i, ts in enumerate(_session_ts(session_date, n_bars))
    ]


def _write_symbol_parquet(
    bars_root: Path, symbol: str, year: int, rows: list[tuple[Any, ...]]
) -> Path:
    symbol_dir = bars_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["ts", *FIELDS])
    path = symbol_dir / f"{year}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _write_empty_symbol_parquet(bars_root: Path, symbol: str, year: int) -> Path:
    symbol_dir = bars_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ts": pd.Series([], dtype="int64"),
            **{field: pd.Series([], dtype="float32") for field in FIELDS},
        }
    )
    path = symbol_dir / f"{year}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _write_manifest(
    data_root: Path, *, fingerprint: str = "fp1", adjustments: str = "adj1"
) -> Path:
    manifest = {
        "resolution": "1",
        "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "symbols": 0,
        "total_rows": 0,
        "adjustments": adjustments,
        "fingerprint": fingerprint,
        "coverage": {},
    }
    manifest_path = data_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _setup_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fingerprint: str = "fp1",
) -> dict[str, Path]:
    """Point settings at a fresh synthetic data/cache tree under tmp_path."""
    data_root = tmp_path / "data"
    bars_root = data_root / "bars" / "1"
    bars_root.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    manifest_path = _write_manifest(data_root, fingerprint=fingerprint)

    monkeypatch.setattr(settings, "DATA_ROOT", data_root)
    monkeypatch.setattr(settings, "BARS_1M", bars_root)
    monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    monkeypatch.setattr("nifty_quant.data.manifest.MANIFEST_PATH", manifest_path)

    return {
        "data_root": data_root,
        "bars_root": bars_root,
        "cache_root": cache_root,
        "manifest_path": manifest_path,
    }


# ---------------------------------------------------------------------------
# _available_years
# ---------------------------------------------------------------------------


def test_available_years_missing_root_returns_empty_list(tmp_path: Path) -> None:
    """A source root that does not exist on disk yields no years, not an error."""
    missing_root = tmp_path / "does-not-exist"
    assert panel_builder._available_years(missing_root) == []


def test_available_years_filters_non_digit_stems_and_sorts(tmp_path: Path) -> None:
    """Only <SYMBOL>/<digits>.parquet files count; result is sorted ascending."""
    source_root = tmp_path / "bars" / "1"
    (source_root / "AAA").mkdir(parents=True)
    (source_root / "BBB").mkdir(parents=True)
    (source_root / "AAA" / "2023.parquet").touch()
    (source_root / "AAA" / "2021.parquet").touch()
    (source_root / "BBB" / "2022.parquet").touch()
    # Non-digit stem must be ignored, not raise ValueError from int().
    (source_root / "BBB" / "manifest.parquet").touch()

    assert panel_builder._available_years(source_root) == [2021, 2022, 2023]


# ---------------------------------------------------------------------------
# _read_symbol_arrays
# ---------------------------------------------------------------------------


def test_read_symbol_arrays_zero_row_parquet_returns_typed_empty_arrays(
    tmp_path: Path,
) -> None:
    """A structurally valid but empty parquet file yields correctly typed empty arrays."""
    bars_root = tmp_path / "bars" / "1"
    path = _write_empty_symbol_parquet(bars_root, "AAA", 2024)

    result = panel_builder._read_symbol_arrays(path)

    assert set(result.keys()) == {"ts", *FIELDS}
    assert result["ts"].dtype == np.int64
    assert result["ts"].shape == (0,)
    for field in FIELDS:
        assert result[field].dtype == np.float32
        assert result[field].shape == (0,)


# ---------------------------------------------------------------------------
# _read_symbols_json
# ---------------------------------------------------------------------------


def test_read_symbols_json_missing_file_returns_none(tmp_path: Path) -> None:
    """A missing symbols.json (OSError on read) is treated as 'no cached symbols'."""
    missing = tmp_path / "symbols.json"
    assert panel_builder._read_symbols_json(missing) is None


def test_read_symbols_json_malformed_text_returns_none(tmp_path: Path) -> None:
    """Non-JSON file content (JSONDecodeError) is treated as 'no cached symbols'."""
    path = tmp_path / "symbols.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert panel_builder._read_symbols_json(path) is None


def test_read_symbols_json_wrong_shape_returns_none(tmp_path: Path) -> None:
    """Valid JSON that is not a list[str] (dict, or list with non-str elements) is rejected."""
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"AAA": 1}), encoding="utf-8")
    assert panel_builder._read_symbols_json(dict_path) is None

    mixed_list_path = tmp_path / "mixed.json"
    mixed_list_path.write_text(json.dumps(["AAA", 123]), encoding="utf-8")
    assert panel_builder._read_symbols_json(mixed_list_path) is None


# ---------------------------------------------------------------------------
# build_panel: default years / symbols discovery
# ---------------------------------------------------------------------------


def test_build_panel_years_none_discovers_available_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """years=None must fall back to every year found under source_root."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="years-none")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2023, _session_rows(day, 5))
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    output = panel_builder.build_panel(
        freq="1", years=None, symbols=["AAA"], force=False, progress=False
    )

    years_built = sorted(int(p.name) for p in output)
    assert years_built == [2023, 2024]


def test_build_panel_symbols_none_discovers_present_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbols=None must discover every symbol with a parquet file for that year."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="symbols-none")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))
    _write_symbol_parquet(bars_root, "BBB", 2024, _session_rows(day, 5, price0=200.0))

    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=None, force=False, progress=False
    )

    assert len(output) == 1
    cache_dir = output[0]
    symbols_json = json.loads((cache_dir / "symbols.json").read_text(encoding="utf-8"))
    assert symbols_json == ["AAA", "BBB"]
    for field in FIELDS:
        arr = np.load(cache_dir / f"{field}.f32.npy")
        assert arr.dtype == np.float32
        assert arr.shape == (5, 2)


# ---------------------------------------------------------------------------
# build_panel: cache reuse vs. rebuild
# ---------------------------------------------------------------------------


def test_build_panel_reuse_subset_with_progress_prints_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cached year is reused (not rebuilt) when requested symbols are a subset,
    and progress=True prints a 'reusing' message."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="reuse-subset")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))
    _write_symbol_parquet(bars_root, "BBB", 2024, _session_rows(day, 5, price0=200.0))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA", "BBB"], force=False, progress=False
    )
    cache_dir = first[0]
    field_path = cache_dir / "open.f32.npy"
    mtime_before = field_path.stat().st_mtime_ns

    capsys.readouterr()
    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=True
    )

    assert second == [cache_dir]
    assert field_path.stat().st_mtime_ns == mtime_before
    captured = capsys.readouterr()
    assert "reusing" in captured.out
    assert str(cache_dir) in captured.out


def test_build_panel_superset_request_triggers_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting a symbol not present in the cached symbols.json invalidates the cache,
    even though force=False and meta.json is still on disk."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="superset")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))
    _write_symbol_parquet(bars_root, "BBB", 2024, _session_rows(day, 5, price0=200.0))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = first[0]
    assert json.loads((cache_dir / "symbols.json").read_text(encoding="utf-8")) == ["AAA"]

    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA", "BBB"], force=False, progress=False
    )

    assert second == [cache_dir]
    assert json.loads((cache_dir / "symbols.json").read_text(encoding="utf-8")) == [
        "AAA",
        "BBB",
    ]


def test_build_panel_missing_symbols_json_triggers_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If symbols.json is unreadable (deleted) while meta.json still exists, the year
    is treated as not cached and rebuilt from scratch rather than raising."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="missing-symjson")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = first[0]
    (cache_dir / "symbols.json").unlink()

    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )

    assert second == [cache_dir]
    assert (cache_dir / "symbols.json").is_file()
    assert json.loads((cache_dir / "symbols.json").read_text(encoding="utf-8")) == ["AAA"]


def test_build_panel_corrupt_symbols_json_triggers_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbols.json with valid JSON but the wrong shape (not list[str]) is treated
    as uncached and the year is rebuilt rather than raising."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="wrong-shape-symjson")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = first[0]
    (cache_dir / "symbols.json").write_text(json.dumps({"AAA": 1}), encoding="utf-8")

    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )

    assert second == [cache_dir]
    assert json.loads((cache_dir / "symbols.json").read_text(encoding="utf-8")) == ["AAA"]


# ---------------------------------------------------------------------------
# build_panel: no matching source data for requested symbols
# ---------------------------------------------------------------------------


def test_build_panel_missing_symbol_with_stale_cache_dir_deletes_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a later request finds none of its symbols on disk, but a cache_dir with a
    meta.json already exists (from a prior, different request), the stale meta.json is
    deleted so the year no longer reports as cached."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="stale-meta")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = first[0]
    assert (cache_dir / "meta.json").is_file()

    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["ZZZ"], force=False, progress=False
    )

    assert second == []
    assert not (cache_dir / "meta.json").exists()
    manifest = Manifest.load()
    assert not panel_builder.is_cached(manifest.cache_key(), "1", 2024)


def test_build_panel_missing_symbol_without_existing_cache_dir_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting a symbol with no parquet file, for a year never built before, does
    nothing: no cache_dir is created, no exception is raised."""
    _setup_settings(tmp_path, monkeypatch, fingerprint="never-built")
    manifest = Manifest.load()
    cache_dir = panel_builder.panel_cache_dir(manifest.cache_key(), "1", 2099)
    assert not cache_dir.exists()

    output = panel_builder.build_panel(
        freq="1", years=[2099], symbols=["ZZZ"], force=False, progress=False
    )

    assert output == []
    assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# build_panel: force=True rebuild
# ---------------------------------------------------------------------------


def test_build_panel_force_rebuild_overwrites_stale_meta_and_refreshes_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=True must bypass any cache reuse, delete the pre-existing meta.json, and
    rewrite the cache from the current source parquet."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="force-rebuild")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5, price0=100.0))

    first = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = first[0]
    original_open = np.load(cache_dir / "open.f32.npy").copy()
    original_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))

    # Change the underlying source data so a rebuild is observable.
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5, price0=900.0))

    second = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=True, progress=False
    )

    assert second == [cache_dir]
    rebuilt_open = np.load(cache_dir / "open.f32.npy")
    assert not np.array_equal(original_open, rebuilt_open)
    expected_open = np.array([900.0, 901.0, 902.0, 903.0, 904.0], dtype=np.float32)
    assert np.allclose(rebuilt_open[:, 0], expected_open)
    rebuilt_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    assert rebuilt_meta["built_at"] != original_meta["built_at"]


# ---------------------------------------------------------------------------
# build_panel: zero-row parquet handling
# ---------------------------------------------------------------------------


def test_build_panel_all_symbols_zero_rows_skips_year_no_cache_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every requested-and-found symbol has zero rows for a year, the union of
    timestamps is empty and the year is skipped entirely: no meta.json, no npy files
    considered built, nothing in the returned list."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="all-empty")
    bars_root = paths["bars_root"]
    _write_empty_symbol_parquet(bars_root, "AAA", 2024)
    _write_empty_symbol_parquet(bars_root, "BBB", 2024)

    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA", "BBB"], force=False, progress=False
    )

    assert output == []
    manifest = Manifest.load()
    cache_dir = panel_builder.panel_cache_dir(manifest.cache_key(), "1", 2024)
    assert not (cache_dir / "meta.json").exists()


def test_build_panel_one_symbol_zero_rows_raises_contract_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol whose parquet file exists but has zero rows contributes no timestamps
    to the union, so its column ends up entirely NaN alongside symbols that do have
    data. That trips the check_no_all_nan_columns canary (a real symbol with no rows
    at all this year looks identical, at the array level, to a column-misalignment
    bug) and build_panel must raise rather than silently emit a bad column."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="one-empty")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_empty_symbol_parquet(bars_root, "AAA", 2024)
    _write_symbol_parquet(bars_root, "BBB", 2024, _session_rows(day, 5, price0=200.0))

    with pytest.raises(ContractViolation) as excinfo:
        panel_builder.build_panel(
            freq="1", years=[2024], symbols=["AAA", "BBB"], force=False, progress=False
        )

    assert "AAA" in str(excinfo.value)
    assert "all-NaN" in str(excinfo.value)


# ---------------------------------------------------------------------------
# build_panel: progress printing on a fresh build
# ---------------------------------------------------------------------------


def test_build_panel_progress_true_prints_build_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """progress=True on a genuine (non-reuse) build prints a 'rows x symbols' summary."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="progress-build")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    capsys.readouterr()
    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=True
    )

    captured = capsys.readouterr()
    assert "5 rows x 1 symbols" in captured.out
    assert str(output[0]) in captured.out


# ---------------------------------------------------------------------------
# build_panel: corrupt source parquet propagates rather than being swallowed
# ---------------------------------------------------------------------------


def test_build_panel_corrupt_parquet_raises_and_identifies_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parquet file that exists but is not valid parquet must raise loudly, not be
    silently skipped or turned into an all-NaN column."""
    import pyarrow.lib as pa_lib

    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="corrupt-parquet")
    bars_root = paths["bars_root"]
    symbol_dir = bars_root / "AAA"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "2024.parquet").write_bytes(b"not a parquet file at all")

    with pytest.raises(pa_lib.ArrowInvalid) as excinfo:
        panel_builder.build_panel(
            freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
        )

    assert "parquet" in str(excinfo.value).lower() or "footer" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# build_panel: unsorted / duplicate timestamps within a single symbol's source
# ---------------------------------------------------------------------------


def test_build_panel_unsorted_timestamps_land_at_correct_sorted_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol's raw parquet rows need not be time-sorted: build_panel must place
    each row at its correct position in the sorted union grid, keyed by timestamp
    value, not by row order in the source file."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="unsorted-ts")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    t0, t1, t2 = _session_ts(day, 3)
    # Rows written out of chronological order; close price uniquely identifies
    # which timestamp each row belongs to.
    rows = [
        (t2, 320.0, 321.0, 319.0, 320.5, 3000.0),
        (t0, 300.0, 301.0, 299.0, 300.5, 1000.0),
        (t1, 310.0, 311.0, 309.0, 310.5, 2000.0),
    ]
    _write_symbol_parquet(bars_root, "AAA", 2024, rows)

    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = output[0]

    ts_arr = np.load(cache_dir / "ts.i64.npy")
    assert list(ts_arr) == [t0, t1, t2]
    close_arr = np.load(cache_dir / "close.f32.npy")
    assert np.allclose(close_arr[:, 0], np.array([300.5, 310.5, 320.5], dtype=np.float32))


def test_build_panel_duplicate_timestamps_within_symbol_documented_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol's raw parquet with a duplicate timestamp does not raise or get
    deduplicated with an error: the union timestamp grid deduplicates (one row for
    that timestamp), and the written value follows plain numpy fancy-index-assign
    semantics for the duplicate row indices (last row in file order wins). This test
    documents that actual, current behavior rather than asserting it is desirable."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="dup-ts")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    (t0,) = _session_ts(day, 1)
    rows = [
        (t0, 100.0, 101.0, 99.0, 111.0, 1000.0),
        (t0, 200.0, 201.0, 199.0, 222.0, 2000.0),
    ]
    _write_symbol_parquet(bars_root, "AAA", 2024, rows)

    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], force=False, progress=False
    )
    cache_dir = output[0]

    ts_arr = np.load(cache_dir / "ts.i64.npy")
    assert list(ts_arr) == [t0]  # union dedupes the timestamp itself

    close_arr = np.load(cache_dir / "close.f32.npy")
    expected = np.full(1, np.nan, dtype=np.float32)
    expected[np.array([0, 0])] = np.array([111.0, 222.0], dtype=np.float32)
    assert close_arr[0, 0] == expected[0]


# ---------------------------------------------------------------------------
# build_panel: workers fan-out
# ---------------------------------------------------------------------------


def test_build_panel_workers_fanout_matches_single_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading symbol parquet files concurrently (workers > 1) must produce byte-
    identical arrays to a single-worker (sequential) build, since column placement is
    keyed by the deterministic `read_paths` order, not by thread completion order."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="workers-fanout")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for i, sym in enumerate(symbols):
        _write_symbol_parquet(bars_root, sym, 2024, _session_rows(day, 5, price0=100.0 + 10 * i))

    single = panel_builder.build_panel(
        freq="1", years=[2024], symbols=symbols, workers=1, force=False, progress=False
    )
    cache_dir = single[0]
    single_close = np.load(cache_dir / "close.f32.npy").copy()
    single_ts = np.load(cache_dir / "ts.i64.npy").copy()

    fanout = panel_builder.build_panel(
        freq="1", years=[2024], symbols=symbols, workers=8, force=True, progress=False
    )

    assert fanout == [cache_dir]
    fanout_close = np.load(cache_dir / "close.f32.npy")
    fanout_ts = np.load(cache_dir / "ts.i64.npy")
    assert np.array_equal(single_ts, fanout_ts)
    assert np.array_equal(single_close, fanout_close)


def test_build_panel_non_positive_workers_clamped_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """workers<=0 must not raise (ThreadPoolExecutor requires max_workers>=1): the
    builder clamps it to at least 1 worker."""
    paths = _setup_settings(tmp_path, monkeypatch, fingerprint="workers-clamped")
    bars_root = paths["bars_root"]
    day = datetime.date(2024, 1, 2)
    _write_symbol_parquet(bars_root, "AAA", 2024, _session_rows(day, 5))

    output = panel_builder.build_panel(
        freq="1", years=[2024], symbols=["AAA"], workers=0, force=False, progress=False
    )

    assert len(output) == 1
    assert (output[0] / "meta.json").is_file()


# ---------------------------------------------------------------------------
# gc_orphans
# ---------------------------------------------------------------------------


def test_gc_orphans_missing_base_dir_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No panel cache has ever been built: base dir absent, no orphans, no error."""
    _setup_settings(tmp_path, monkeypatch, fingerprint="gc-no-base")
    assert panel_builder.gc_orphans() == []
    assert panel_builder.gc_orphans(dry_run=False) == []


def test_gc_orphans_dry_run_default_lists_without_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run=True (the default) reports orphan cache_key dirs but leaves them on disk."""
    _setup_settings(tmp_path, monkeypatch, fingerprint="gc-current")
    current_key = Manifest.load().cache_key()
    base = settings.CACHE_ROOT / "panel" / f"v{settings.PANEL_VERSION}"
    (base / current_key).mkdir(parents=True)
    orphan_a = base / "orphan-aaa"
    orphan_b = base / "orphan-bbb"
    orphan_a.mkdir()
    orphan_b.mkdir()

    result = panel_builder.gc_orphans()

    assert result == [orphan_a, orphan_b]
    assert orphan_a.exists()
    assert orphan_b.exists()
    assert (base / current_key).exists()


def test_gc_orphans_dry_run_false_deletes_orphan_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run=False actually removes every non-current cache_key dir."""
    _setup_settings(tmp_path, monkeypatch, fingerprint="gc-delete")
    current_key = Manifest.load().cache_key()
    base = settings.CACHE_ROOT / "panel" / f"v{settings.PANEL_VERSION}"
    (base / current_key).mkdir(parents=True)
    orphan = base / "orphan-zzz"
    orphan.mkdir()
    (orphan / "meta.json").write_text("{}", encoding="utf-8")

    result = panel_builder.gc_orphans(dry_run=False)

    assert result == [orphan]
    assert not orphan.exists()
    assert (base / current_key).exists()


def test_gc_orphans_dry_run_false_with_no_orphans_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run=False with only the current cache_key present deletes nothing and
    returns an empty list (the deletion loop over an empty orphans list is a no-op)."""
    _setup_settings(tmp_path, monkeypatch, fingerprint="gc-only-current")
    current_key = Manifest.load().cache_key()
    base = settings.CACHE_ROOT / "panel" / f"v{settings.PANEL_VERSION}"
    (base / current_key).mkdir(parents=True)

    result = panel_builder.gc_orphans(dry_run=False)

    assert result == []
    assert (base / current_key).exists()
