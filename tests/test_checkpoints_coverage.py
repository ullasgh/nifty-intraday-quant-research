import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nifty_quant.data.manifest as manifest_module
import nifty_quant.settings as settings
from nifty_quant.calendar import SessionGrid
from nifty_quant.data.checkpoints import (
    CheckpointPanel,
    cached_checkpoint_view,
    checkpoint_view,
)
from nifty_quant.data.panel import Panel


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


def _build_small_panel() -> Panel:
    """3 sessions, 5 bars/day, 09:15-09:19, symbols AAA,BBB,CCC."""
    symbols = ("AAA", "BBB", "CCC")
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    all_ts: list[int] = []
    for day in dates:
        all_ts.extend(_session_ts(day, 5))
    ts = np.asarray(all_ts, dtype=np.int64)
    grid = SessionGrid.from_timestamps(ts)
    n_rows = len(ts)
    n_symbols = len(symbols)
    fields: dict[str, np.ndarray] = {}
    for field_name in ("close", "open"):
        arr = np.zeros((n_rows, n_symbols), dtype=np.float32)
        for sym_idx in range(n_symbols):
            if field_name == "close":
                arr[:, sym_idx] = (
                    np.arange(n_rows, dtype=np.float32) * 100.0 + sym_idx
                )
            else:
                arr[:, sym_idx] = (
                    np.arange(n_rows, dtype=np.float32) * 100.0 + sym_idx + 0.5
                )
        fields[field_name] = arr
    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _build_mixed_length_panel() -> Panel:
    """3 sessions at REALISTIC scale: two full 375-bar regular sessions around one
    60-bar Muhurat-length session (index 1). Guards against a fixed-stride
    day-index assumption that a minimal uniform-length fixture would conceal:
    day_offsets are [0, 375, 435, 810], not a multiple of any constant stride."""
    symbols = ("AAA", "BBB", "CCC")
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(3)]
    session_lengths = (375, 60, 375)
    all_ts: list[int] = []
    for day, n_bars in zip(dates, session_lengths):
        all_ts.extend(_session_ts(day, n_bars))
    ts = np.asarray(all_ts, dtype=np.int64)
    grid = SessionGrid.from_timestamps(ts)
    n_rows = len(ts)
    n_symbols = len(symbols)
    close = np.zeros((n_rows, n_symbols), dtype=np.float32)
    for sym_idx in range(n_symbols):
        close[:, sym_idx] = 100.0 + sym_idx * 10.0 + np.arange(n_rows, dtype=np.float32) * 0.01
    return Panel(
        fields={"close": close},
        symbols=symbols,
        ts=ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


def _write_minimal_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolution": "1",
        "written_at": "2024-01-01T00:00:00+05:30",
        "symbols": 0,
        "total_rows": 0,
        "adjustments": "none",
        "fingerprint": "fp_min",
        "coverage": {},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def _write_pipeline_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal parquet tree plus MANIFEST.json, return (data_root, cache_root)."""
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    bars_dir = data_root / "bars" / "1"
    symbols = ("AAA", "BBB")
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(3)]

    all_ts: list[int] = []
    for day in dates:
        all_ts.extend(_session_ts(day, 5))
    ts = np.asarray(all_ts, dtype=np.int64)

    for symbol in symbols:
        sym_dir = bars_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "ts": ts,
                "open": np.arange(len(ts), dtype=np.float64) * 0.01 + 100.0,
                "high": np.arange(len(ts), dtype=np.float64) * 0.01 + 101.0,
                "low": np.arange(len(ts), dtype=np.float64) * 0.01 + 99.0,
                "close": np.arange(len(ts), dtype=np.float64) * 0.01 + 100.5,
                "volume": np.full(len(ts), 1000, dtype=np.int64),
            }
        )
        df.to_parquet(sym_dir / "2024.parquet")

    manifest_path = data_root / "MANIFEST.json"
    manifest = {
        "resolution": "1",
        "written_at": "2024-01-01T00:00:00+05:30",
        "symbols": len(symbols),
        "total_rows": len(ts) * len(symbols),
        "adjustments": "none",
        "fingerprint": "fp_test",
        "coverage": {
            sym: {
                "years": [2024],
                "rows": len(ts),
                "first": "2024-01-01T09:15:00+05:30",
                "last": "2024-01-03T09:19:00+05:30",
            }
            for sym in symbols
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    return data_root, cache_root


def test_get_missing_field_raises_keyerror() -> None:
    cp = CheckpointPanel(
        times=("09:16", "09:17"),
        fields=("close", "open"),
        symbols=("AAA", "BBB"),
        dates=np.array([datetime.date(2024, 1, 1)], dtype=object),
        _data={"close": np.zeros((1, 2, 2), dtype=np.float64)},
    )
    with pytest.raises(KeyError) as exc_info:
        cp.get("open", "09:16")
    message = str(exc_info.value)
    assert "open" in message
    assert "close" in message


def test_get_missing_time_raises_keyerror_with_existing_field() -> None:
    cp = CheckpointPanel(
        times=("09:16", "09:17"),
        fields=("close",),
        symbols=("AAA", "BBB"),
        dates=np.array([datetime.date(2024, 1, 1)], dtype=object),
        _data={"close": np.zeros((1, 2, 2), dtype=np.float64)},
    )
    with pytest.raises(KeyError) as exc_info:
        cp.get("close", "09:18")
    message = str(exc_info.value)
    assert "09:18" in message
    assert "09:16" in message
    assert "09:17" in message


def test_array_missing_field_raises_keyerror() -> None:
    cp = CheckpointPanel(
        times=("09:16", "09:17"),
        fields=("close", "open"),
        symbols=("AAA", "BBB"),
        dates=np.array([datetime.date(2024, 1, 1)], dtype=object),
        _data={"close": np.zeros((1, 2, 2), dtype=np.float64)},
    )
    with pytest.raises(KeyError) as exc_info:
        cp.array("open")
    assert "open" in str(exc_info.value)


def test_checkpoint_view_never_present_time_is_all_nan() -> None:
    panel = _build_small_panel()
    cp = checkpoint_view(panel, times=("09:16", "23:59"), fields=("close",))
    assert cp.times == ("09:16", "23:59")
    data = cp._data["close"]
    assert data.shape == (3, 2, 3)

    # 23:59 never occurs in any session: the whole column stays NaN.
    assert np.isnan(data[:, 1, :]).all()

    # 09:16 (offset 1 in each 5-bar 09:15-09:19 session) matches the source panel.
    expected = panel.field("close")[[1, 6, 11], :]
    np.testing.assert_allclose(data[:, 0, :], expected)


def test_checkpoint_view_realistic_mixed_length_sessions_no_stride_leakage() -> None:
    """Regression guard: day-index resolution must come from actual day_offsets
    (which are irregular here: 375, 60, 375 bars), never from a fixed
    bars-per-session stride. A uniform-length fixture cannot expose a stride bug
    because every day's offset would also be a multiple of that stride."""
    panel = _build_mixed_length_panel()
    muhurat_day_idx = 1

    cp = checkpoint_view(panel, times=("09:16", "14:30"), fields=("close",))

    # "09:16" exists in every session, including the 60-bar Muhurat day.
    row_0916 = [1, 376, 436]
    expected_0916 = panel.field("close")[row_0916, :]
    np.testing.assert_allclose(cp.get("close", "09:16"), expected_0916)

    # "14:30" (minute offset 315) is beyond the 60-bar Muhurat session's last bar
    # (offset 59) so it must be NaN there, while the two 375-bar days resolve to
    # their true absolute rows (315 and 750) -- not day_idx * a fixed stride.
    checkpoint_1430 = cp.get("close", "14:30")
    assert np.isnan(checkpoint_1430[muhurat_day_idx, :]).all()

    row_1430_regular = [315, 750]
    expected_1430_regular = panel.field("close")[row_1430_regular, :]
    np.testing.assert_allclose(
        checkpoint_1430[[0, 2], :], expected_1430_regular
    )


def test_cached_checkpoint_view_corrupted_cache_hit_missing_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, cache_root = _write_pipeline_data(tmp_path)
    manifest_path = data_root / "MANIFEST.json"

    monkeypatch.setattr(settings, "DATA_ROOT", data_root)
    monkeypatch.setattr(settings, "BARS_1M", data_root / "bars" / "1")
    monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(manifest_module, "MANIFEST_PATH", manifest_path)

    args = {
        "times": ("09:16", "09:17"),
        "fields": ("close", "open"),
        "symbols": ("AAA", "BBB"),
        "start": datetime.date(2024, 1, 1),
        "end": datetime.date(2024, 1, 3),
    }
    cached_checkpoint_view(**args)

    cache_dir = cache_root / "checkpoints"
    cache_files = list(cache_dir.rglob("*.npz"))
    assert len(cache_files) == 1
    cache_path = cache_files[0]

    # Overwrite the cache with the same metadata but missing field__open.
    with np.load(cache_path, allow_pickle=True) as data:
        reduced = {name: data[name] for name in data.files if name != "field__open"}
    np.savez(cache_path, **reduced)

    with pytest.raises(KeyError) as exc_info:
        cached_checkpoint_view(**args)
    assert "field__open" in str(exc_info.value)


def test_cached_checkpoint_view_miss_symbols_none_raises_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_minimal_manifest(manifest_path)
    cache_root = tmp_path / "empty_cache"

    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(manifest_module, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError) as exc_info:
        cached_checkpoint_view(
            times=("09:16",),
            fields=("close",),
            symbols=None,
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 2),
        )
    assert "symbols" in str(exc_info.value)


def test_cached_checkpoint_view_miss_start_end_none_raises_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_minimal_manifest(manifest_path)
    cache_root = tmp_path / "empty_cache"

    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(manifest_module, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError) as exc_info:
        cached_checkpoint_view(
            times=("09:16",),
            fields=("close",),
            symbols=("AAA",),
            start=None,
            end=None,
        )
    message = str(exc_info.value)
    assert "start" in message
    assert "end" in message
