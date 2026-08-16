from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nifty_quant import settings
from nifty_quant.calendar import to_ist
from nifty_quant.data import panel_builder
from nifty_quant.data.manifest import Manifest
from nifty_quant.data.panel import (
    Panel,
    PanelSpec,
    PanelTooLargeError,
    _materialized_dir,
    load_panel,
)

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


def _ist_scalar(ts_epoch: int | np.integer) -> pd.Timestamp:
    """Return a scalar tz-aware Timestamp for a single epoch second."""
    result = to_ist(ts_epoch)
    if isinstance(result, pd.DatetimeIndex):
        result = result[0]
    return pd.Timestamp(result)


def _create_panel_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_bars: dict[str, set[int]] | None = None,
    build_and_load: bool = True,
) -> dict[str, Any]:
    """Build the synthetic 3-symbol 2024 tree and return a loaded Panel plus raw pieces."""
    symbols = ("AAA", "BBB", "CCC")
    year = 2024
    dates = [datetime.date(2024, 1, d) for d in range(1, 10)]
    short_date = datetime.date(2024, 1, 10)

    sym_frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        rows = []
        for d in dates:
            for i, ts in enumerate(_session_ts(d, 375)):
                rows.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0 + i))
        for i, ts in enumerate(_session_ts(short_date, 60)):
            rows.append((ts, 200.0 + i, 201.0 + i, 199.0 + i, 200.5 + i, 2000.0 + i))
        frame = pd.DataFrame(rows, columns=["ts", *FIELDS])
        if missing_bars and sym in missing_bars:
            frame = frame.loc[~frame["ts"].isin(missing_bars[sym])].reset_index(drop=True)
        sym_frames[sym] = frame

    data_root = tmp_path / "data"
    bars_root = data_root / "bars" / "1"
    bars_root.mkdir(parents=True)
    for sym, frame in sym_frames.items():
        symbol_dir = bars_root / sym
        symbol_dir.mkdir()
        frame.to_parquet(symbol_dir / f"{year}.parquet", index=False)

    coverage = {}
    for sym, frame in sym_frames.items():
        coverage[sym] = {
            "years": [year],
            "rows": len(frame),
            "first": pd.to_datetime(int(frame["ts"].min()), unit="s", utc=True)
            .tz_convert("Asia/Kolkata")
            .isoformat(),
            "last": pd.to_datetime(int(frame["ts"].max()), unit="s", utc=True)
            .tz_convert("Asia/Kolkata")
            .isoformat(),
        }

    manifest = {
        "resolution": "1",
        "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "symbols": len(symbols),  # int, matching real MANIFEST.json shape
        "total_rows": sum(len(frame) for frame in sym_frames.values()),
        "adjustments": "testadj",
        "fingerprint": "testfp",
        "coverage": coverage,
    }
    manifest_path = data_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache_root = tmp_path / "cache"

    monkeypatch.setattr(settings, "DATA_ROOT", data_root)
    monkeypatch.setattr(settings, "BARS_1M", bars_root)
    monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    # manifest.py imported MANIFEST_PATH directly; patch its module global as well.
    monkeypatch.setattr("nifty_quant.data.manifest.MANIFEST_PATH", manifest_path)

    spec = PanelSpec(
        freq="1",
        fields=FIELDS,
        symbols=symbols,
        start=datetime.date(2024, 1, 1),
        end=short_date,
    )

    if build_and_load:
        panel = load_panel(spec, memmap=True)
    else:
        panel = None

    return {
        "symbols": symbols,
        "dates": dates,
        "short_date": short_date,
        "frames": sym_frames,
        "data_root": data_root,
        "bars_root": bars_root,
        "manifest_path": manifest_path,
        "cache_root": cache_root,
        "spec": spec,
        "panel": panel,
    }


@pytest.fixture
def panel_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Shared synthetic fixture used by all fast tests."""
    return _create_panel_scenario(tmp_path, monkeypatch)


def test_timestamp_contract() -> None:
    """Epoch 1704080700 must map to 2024-01-01 09:15 IST."""
    assert _ist_scalar(1704080700) == pd.Timestamp("2024-01-01 09:15", tz="Asia/Kolkata")


def test_round_trip_fidelity(panel_fixture: dict[str, Any]) -> None:
    """Panel values match fresh parquet reads after a float32 cast."""
    panel = panel_fixture["panel"]
    frames = panel_fixture["frames"]
    rng = np.random.default_rng(42)
    samples = []

    # 25 rows per symbol = 75 total: enough to cover regular and short sessions.
    for sym, frame in frames.items():
        idx = rng.choice(len(frame), size=25, replace=False)
        for i in idx:
            row = frame.iloc[i]
            samples.append((sym, int(row["ts"]), row))

    assert 50 <= len(samples) <= 100

    for sym, ts, row in samples:
        col = panel.sym_ix[sym]
        row_indices = np.flatnonzero(panel.ts == ts)
        assert row_indices.size == 1
        row_idx = int(row_indices[0])
        for field in FIELDS:
            expected = np.float32(row[field])
            actual = panel.field(field)[row_idx, col]
            assert actual == expected, f"{sym} ts={ts} field={field}"


def test_day_offsets_integrity(panel_fixture: dict[str, Any]) -> None:
    """day_offsets must be strictly increasing and capture the short 60-bar session."""
    panel = panel_fixture["panel"]
    offsets = panel.day_offsets

    assert offsets[0] == 0
    assert offsets[-1] == panel.n_rows()
    assert len(offsets) == panel.n_days() + 1
    assert np.all(np.diff(offsets) > 0)
    assert 60 in np.diff(offsets)


def test_rows_at_time_0915_labels_exact(panel_fixture: dict[str, Any]) -> None:
    """Regression: rows_at_time('09:15') must only return bars whose label is 09:15."""
    panel = panel_fixture["panel"]
    rows = panel.rows_at_time("09:15")
    assert rows.size > 0
    minutes = panel.minute_of_day()[rows]
    assert np.all(minutes == 9 * 60 + 15)


def test_rows_at_time_1020_excludes_short_session(panel_fixture: dict[str, Any]) -> None:
    """10:20 exists in regular sessions only, so exactly 9 rows and no short-session row."""
    panel = panel_fixture["panel"]
    rows = panel.rows_at_time("10:20")
    assert len(rows) == 9

    short_date = panel_fixture["short_date"]
    for row in rows:
        ist_ts = _ist_scalar(panel.ts[row])
        assert ist_ts.date() != short_date

    # The naive fixed-offset formula would incorrectly include the short session.
    naive = panel.day_offsets[:-1] + 65
    assert not np.array_equal(rows, naive)


def test_no_forward_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing interior bars stay NaN, not forward-filled."""
    first_session_ts = _session_ts(datetime.date(2024, 1, 1), 375)
    missing_ts = set(first_session_ts[50:53])

    scenario = _create_panel_scenario(
        tmp_path,
        monkeypatch,
        missing_bars={"AAA": missing_ts},
    )
    panel = scenario["panel"]
    col = panel.sym_ix["AAA"]
    close = panel.field("close")

    missing_rows = np.flatnonzero(np.isin(panel.ts, list(missing_ts)))
    assert missing_rows.size == len(missing_ts)
    assert np.all(np.isnan(close[missing_rows, col]))

    expected_present = 9 * 375 + 60 - len(missing_ts)
    actual_present = int(np.sum(~np.isnan(close[:, col])))
    assert actual_present == expected_present
    assert int(np.isnan(close[:, col]).sum()) == panel.n_rows() - expected_present


def test_no_all_nan_columns(panel_fixture: dict[str, Any]) -> None:
    """With full synthetic data, no field may contain an all-NaN symbol column."""
    panel = panel_fixture["panel"]
    for field in FIELDS:
        arr = panel.field(field)
        assert not np.any(np.all(np.isnan(arr), axis=0))


def test_dtype_and_long_dtype(panel_fixture: dict[str, Any]) -> None:
    """Panel arrays are float32; to_long returns float64 columns."""
    panel = panel_fixture["panel"]
    assert panel.field("close").dtype == np.float32

    long_df = panel.to_long()
    assert all(dtype == np.float64 for dtype in long_df.dtypes)


def test_to_long_from_long_roundtrip(panel_fixture: dict[str, Any]) -> None:
    """to_long/from_long must preserve index names, IST timestamps, and field values."""
    panel = panel_fixture["panel"]
    long_df = panel.to_long()

    assert list(long_df.index.names) == ["Timestamp", "Ticker"]
    ts_idx = long_df.index.get_level_values("Timestamp")
    assert isinstance(ts_idx, pd.DatetimeIndex)
    assert str(ts_idx.tz) == "Asia/Kolkata"

    p2 = Panel.from_long(long_df)
    for field in panel_fixture["spec"].fields:
        assert np.allclose(p2.field(field), panel.field(field), equal_nan=True)


def test_memory_guard() -> None:
    """A spec whose estimated size exceeds the limit must raise PanelTooLargeError."""
    spec = PanelSpec(
        freq="1",
        fields=("open", "close"),
        symbols=("AAA", "BBB"),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )

    with pytest.raises(PanelTooLargeError) as excinfo:
        load_panel(spec, memory_limit_gb=0.000001)
    assert "memmap" in str(excinfo.value)


def test_cache_invalidation() -> None:
    """Different manifest fingerprints must produce different cache keys and dirs."""
    base = Manifest(
        resolution="1",
        written_at=datetime.datetime.now(datetime.UTC),
        n_symbols=3,
        total_rows=100,
        adjustments="testadj",
        fingerprint="fp1",
        coverage={},
    )
    changed = replace(base, fingerprint="fp2")

    key_a = base.cache_key()
    key_b = changed.cache_key()

    assert key_a != key_b
    assert panel_builder.panel_cache_dir(key_a, "1", 2024) != panel_builder.panel_cache_dir(
        key_b, "1", 2024
    )


def test_memmap_backing(panel_fixture: dict[str, Any]) -> None:
    """memmap=True yields memmapped field arrays; memmap=False yields ordinary ndarrays."""
    panel = panel_fixture["panel"]
    memmap_close = panel.field("close")
    assert isinstance(memmap_close, np.memmap) or isinstance(memmap_close.base, np.memmap)

    non_memmap_panel = load_panel(panel_fixture["spec"], memmap=False)
    assert not isinstance(non_memmap_panel.field("close"), np.memmap)


@pytest.mark.slow
def test_real_data_slow() -> None:
    """Build a 2024 panel for 5 real symbols and verify completeness against parquet rows."""
    try:
        manifest = Manifest.load()
    except FileNotFoundError:
        pytest.skip("Real data/MANIFEST.json not found; skipping slow real-data test.")

    candidates = sorted(
        sym for sym, cov in manifest.coverage.items() if 2024 in cov.years
    )
    if len(candidates) < 5:
        pytest.skip(f"Need at least 5 symbols with 2024 data, found {len(candidates)}.")

    if "RELIANCE" in manifest.coverage and 2024 in manifest.coverage["RELIANCE"].years:
        selected = ["RELIANCE"] + [sym for sym in candidates if sym != "RELIANCE"][:4]
    else:
        selected = candidates[:5]
    selected = selected[:5]

    panel_builder.build_panel(freq="1", years=[2024], symbols=selected, progress=False)

    spec = PanelSpec(
        freq="1",
        fields=("close",),
        symbols=tuple(selected),
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    panel = load_panel(spec, memmap=False)
    close = panel.field("close")

    for i, sym in enumerate(panel.symbols):
        nan_count = int(np.isnan(close[:, i]).sum())
        assert nan_count == 0, f"Symbol {sym} has {nan_count} NaN rows for 2024."
        assert close[:, i].shape[0] == panel.n_rows()

    # Derive the expected row count from a real parquet file instead of hardcoding it.
    reliance_parquet = settings.BARS_1M / "RELIANCE" / "2024.parquet"
    if "RELIANCE" in panel.symbols and reliance_parquet.exists():
        expected_rows = len(pd.read_parquet(reliance_parquet, columns=["ts"]))
        reliance_spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("RELIANCE",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 12, 31),
        )
        reliance_panel = load_panel(reliance_spec, memmap=False)
        assert reliance_panel.n_rows() == expected_rows
        assert panel.n_rows() == expected_rows
    else:
        first_sym = panel.symbols[0]
        parquet_path = settings.BARS_1M / first_sym / "2024.parquet"
        if not parquet_path.exists():
            pytest.skip(f"Parquet file for {first_sym} 2024 is missing.")
        expected_rows = len(pd.read_parquet(parquet_path, columns=["ts"]))
        assert panel.n_rows() == expected_rows


_RACE_WORKER_SCRIPT = """
import datetime
import hashlib
import json
import sys
import time

import numpy as np

from nifty_quant.data.panel import PanelSpec, load_panel


def main() -> None:
    symbols = tuple(sys.argv[1].split(","))
    field = sys.argv[2]
    start = datetime.date.fromisoformat(sys.argv[3])
    end = datetime.date.fromisoformat(sys.argv[4])
    pre_delay_s = float(sys.argv[5])
    hold_open_s = float(sys.argv[6])
    out_path = sys.argv[7]

    if pre_delay_s > 0:
        time.sleep(pre_delay_s)

    spec = PanelSpec(freq="1", fields=(field,), symbols=symbols, start=start, end=end)
    panel = load_panel(spec, memmap=True)
    arr = panel.field(field)

    if hold_open_s > 0:
        t_end = time.time() + hold_open_s
        while time.time() < t_end:
            _ = arr[0, 0]  # keep actively touching the live mapping
            time.sleep(0.001)

    snapshot = np.array(arr, dtype=np.float32)
    checksum = hashlib.sha256(snapshot.tobytes()).hexdigest()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"checksum": checksum, "shape": list(snapshot.shape)}, fh)


if __name__ == "__main__":
    main()
"""


def test_reuse_does_not_rewrite_materialized_files(panel_fixture: dict[str, Any]) -> None:
    """A second load_panel() call for the same spec must not rewrite the cached arrays."""
    spec = panel_fixture["spec"]
    cache_key = Manifest.load().cache_key()
    materialized_dir = _materialized_dir(cache_key, spec)
    close_path = materialized_dir / "close.f32.npy"
    assert close_path.exists()

    mtime_before = close_path.stat().st_mtime_ns
    load_panel(spec, memmap=True)

    assert close_path.stat().st_mtime_ns == mtime_before


def test_atomic_materialization_leaves_no_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-materialization failure must leave no canonical dir and no temp dirs."""
    scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)
    spec = scenario["spec"]

    real_open_memmap = np.lib.format.open_memmap
    call_count = 0

    def fake_open_memmap(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            raise RuntimeError("synthetic failure mid-materialization")
        return real_open_memmap(*args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", fake_open_memmap)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        load_panel(spec, memmap=True)

    materialized_dir = _materialized_dir(Manifest.load().cache_key(), spec)
    assert not materialized_dir.exists()

    if materialized_dir.parent.exists():
        leftover_tmp_dirs = list(
            materialized_dir.parent.glob(f"{materialized_dir.name}.tmp.*")
        )
        assert leftover_tmp_dirs == []


def test_force_rebuild_changes_mtime_and_data_stays_correct(
    panel_fixture: dict[str, Any],
) -> None:
    """force=True must rebuild (new mtime) and still produce correct data."""
    spec = panel_fixture["spec"]
    materialized_dir = _materialized_dir(Manifest.load().cache_key(), spec)
    close_path = materialized_dir / "close.f32.npy"

    mtime_before = close_path.stat().st_mtime_ns
    rebuilt = load_panel(spec, memmap=True, force=True)

    assert close_path.stat().st_mtime_ns != mtime_before

    reference = load_panel(spec, memmap=False)
    assert np.allclose(rebuilt.field("close"), reference.field("close"), equal_nan=True)


def test_materialized_array_is_read_only(panel_fixture: dict[str, Any]) -> None:
    """The returned memmap-backed array must reject in-place mutation."""
    panel = panel_fixture["panel"]
    close = panel.field("close")

    with pytest.raises(ValueError, match="read-only"):
        close[0, 0] = 999.0


@pytest.mark.slow
def test_concurrent_load_panel_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the concurrent materialization race.

    Two independent OS processes materializing the same PanelSpec concurrently used
    to race on ``np.lib.format.open_memmap(path, mode="w+")`` truncating the shared
    canonical path while the other process still had it memory-mapped open. Pre-fix,
    this can SIGBUS-kill the reading process outright - an uncatchable hardware fault,
    not a Python exception - so both workers here run as separate `subprocess`
    children (never `ProcessPoolExecutor`/`multiprocessing`, whose workers are still
    children of the pytest runner and whose death can take the runner down with them).
    A crash therefore only shows up as a non-zero/negative child exit code, which is
    asserted on below, instead of crashing the test process itself.

    Uses real repository fixture data (like test_real_data_slow) because settings
    monkeypatches do not propagate across a `subprocess` boundary; only CACHE_ROOT
    (env-var-driven) is overridden, so nothing is ever written under `data/`.
    """
    try:
        manifest = Manifest.load()
    except FileNotFoundError:
        pytest.skip("Real data/MANIFEST.json not found; skipping slow real-data test.")

    candidates = sorted(sym for sym, cov in manifest.coverage.items() if 2024 in cov.years)
    if len(candidates) < 2:
        pytest.skip(f"Need at least 2 symbols with 2024 data, found {len(candidates)}.")
    symbols = tuple(candidates[:2])
    field = "close"
    start = datetime.date(2024, 1, 1)
    end = datetime.date(2024, 12, 31)

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)

    spec = PanelSpec(freq="1", fields=(field,), symbols=symbols, start=start, end=end)

    # Pre-warm the per-year panel_builder cache AND publish an initial materialization
    # sequentially, in-process, isolated to cache_root. This makes the race below a
    # "someone already has this open, someone else rebuilds it" scenario - matching the
    # real bug report - rather than a first-ever-build race (panel_builder's own
    # per-year cache build has a separate, out-of-scope race not covered by this fix).
    load_panel(spec, memmap=True)

    worker_script = tmp_path / "_race_worker.py"
    worker_script.write_text(_RACE_WORKER_SCRIPT, encoding="utf-8")

    env = dict(os.environ)
    env["NQ_CACHE_ROOT"] = str(cache_root)

    out_reader = tmp_path / "reader_out.json"
    out_writer = tmp_path / "writer_out.json"

    common_args = [",".join(symbols), field, start.isoformat(), end.isoformat()]

    reader_proc = subprocess.Popen(
        [sys.executable, str(worker_script), *common_args, "0", "0.5", str(out_reader)],
        env=env,
    )
    writer_proc = subprocess.Popen(
        [sys.executable, str(worker_script), *common_args, "0.05", "0", str(out_writer)],
        env=env,
    )

    def _wait(proc: subprocess.Popen, label: str) -> int:
        try:
            return proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(f"{label} subprocess timed out and was killed")

    reader_returncode = _wait(reader_proc, "reader")
    writer_returncode = _wait(writer_proc, "writer")

    assert reader_returncode == 0, (
        f"reader subprocess crashed/failed with exit code {reader_returncode} "
        "(a SIGBUS from a truncated live mmap shows up as a negative code on POSIX)"
    )
    assert writer_returncode == 0, f"writer subprocess failed with exit code {writer_returncode}"

    reader_result = json.loads(out_reader.read_text(encoding="utf-8"))
    writer_result = json.loads(out_writer.read_text(encoding="utf-8"))

    reference_panel = load_panel(spec, memmap=False)
    reference = np.array(reference_panel.field(field), dtype=np.float32)
    assert np.all(np.isfinite(reference))
    reference_checksum = hashlib.sha256(reference.tobytes()).hexdigest()

    assert reader_result["checksum"] == reference_checksum
    assert writer_result["checksum"] == reference_checksum
