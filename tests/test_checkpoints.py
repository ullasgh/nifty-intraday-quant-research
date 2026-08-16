from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nifty_quant.data.manifest as manifest_module
import nifty_quant.settings as settings
from nifty_quant.calendar import SessionGrid
from nifty_quant.data.checkpoints import (
    cached_checkpoint_view,
    checkpoint_view,
)
from nifty_quant.data.manifest import Manifest
from nifty_quant.data.panel import Panel


def _session_ts(
    session_date: datetime.date,
    n_bars: int,
    start_hhmm: tuple[int, int] = (9, 15),
) -> list[int]:
    """Return left-labelled epoch-second UTC timestamps for an IST wall-clock start."""
    local_start = datetime.datetime(
        session_date.year, session_date.month, session_date.day,
        start_hhmm[0], start_hhmm[1],
    )
    utc_naive = local_start - datetime.timedelta(hours=5, minutes=30)
    epoch0 = int((utc_naive - datetime.datetime(1970, 1, 1)).total_seconds())
    return [epoch0 + 60 * i for i in range(n_bars)]


def _build_in_memory_panel() -> Panel:
    """10 sessions: all full 375-bar sessions except the 5th, which has 60 bars."""
    symbols = ("AAA", "BBB", "CCC")
    dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i)
        for i in range(10)
    ]
    short_idx = 4

    all_ts: list[int] = []
    for i, day in enumerate(dates):
        n_bars = 60 if i == short_idx else 375
        all_ts.extend(_session_ts(day, n_bars))

    ts = np.asarray(all_ts, dtype=np.int64)
    grid = SessionGrid.from_timestamps(ts)

    n_rows = len(ts)
    n_symbols = len(symbols)

    fields: dict[str, np.ndarray] = {}
    for field_name in ("close", "open"):
        arr = np.zeros((n_rows, n_symbols), dtype=np.float32)
        for sym_idx in range(n_symbols):
            base = 100.0 + sym_idx * 10.0
            if field_name == "open":
                base += 0.5
            arr[:, sym_idx] = base + np.arange(n_rows, dtype=np.float32) * 0.01
        fields[field_name] = arr

    return Panel(
        fields=fields,
        symbols=symbols,
        ts=ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )


@pytest.fixture
def panel() -> Panel:
    return _build_in_memory_panel()


def test_checkpoint_view_11_00_matches_source_panel(panel: Panel) -> None:
    cp = checkpoint_view(panel, ("11:00",), ("close",))
    rows = panel.rows_at_time("11:00")
    day_indices = np.searchsorted(panel.day_offsets, rows, side="right") - 1

    assert len(rows) == 9
    assert len(day_indices) == 9

    for day_idx, row in zip(day_indices, rows):
        assert panel.dates[int(day_idx)] != datetime.date(2024, 1, 5)
        for sym_idx in range(panel.n_symbols()):
            got = cp.get("close", "11:00")[int(day_idx), sym_idx]
            expected = panel.field("close")[int(row), sym_idx]
            assert np.isclose(got, expected)


def test_short_session_missing_checkpoint_is_nan(panel: Panel) -> None:
    cp = checkpoint_view(panel, ("11:00",), ("close",))
    short_day_idx = 4
    short_close = cp.get("close", "11:00")[short_day_idx, :]

    assert np.isnan(short_close).all()

    short_date = panel.dates[short_day_idx]
    short_slice = panel.day_slice(short_date)
    last_short_row = short_slice.stop - 1

    for sym_idx in range(panel.n_symbols()):
        # Nan must not equal any neighbouring day's 11:00 value.
        assert short_close[sym_idx] != cp.get("close", "11:00")[short_day_idx - 1, sym_idx]
        assert short_close[sym_idx] != cp.get("close", "11:00")[short_day_idx + 1, sym_idx]
        # Nan must not equal the short session's own last bar close at 10:14.
        assert short_close[sym_idx] != panel.field("close")[last_short_row, sym_idx]


def test_entry_checkpoint_warnings(panel: Panel) -> None:
    with pytest.warns(UserWarning):
        checkpoint_view(panel, ("09:15",), ("close",))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        checkpoint_view(panel, ("09:16",), ("close",))

    assert not any(issubclass(w.category, UserWarning) for w in record)


def test_array_shape_and_time_equality(panel: Panel) -> None:
    times = ("09:16", "11:00", "14:30")
    cp = checkpoint_view(panel, times, ("close",))

    arr = cp.array("close")
    assert arr.shape == (10, len(times), 3)

    for time_idx, time_label in enumerate(times):
        assert np.allclose(
            arr[:, time_idx, :], cp.get("close", time_label), equal_nan=True
        )


def test_nbytes_formula_and_float64(panel: Panel) -> None:
    times = ("09:16", "11:00", "14:30", "15:20")
    cp = checkpoint_view(panel, times, ("close",))

    n_days = len(panel.dates)
    n_times = len(times)
    n_symbols = panel.n_symbols()
    n_fields = len(cp.fields)

    expected = n_days * n_times * n_symbols * 8 * n_fields
    assert cp.nbytes() == expected
    assert cp.nbytes() < 10 * 1024

    assert cp.array("close").dtype == np.float64
    assert cp.get("close", "09:16").dtype == np.float64


def _write_real_pipeline_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal parquet tree plus MANIFEST.json, return (data_root, cache_root)."""
    data_root = tmp_path / "data"
    bars_root = data_root / "bars" / "1"
    cache_root = tmp_path / "cache"
    bars_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    symbols = ("AAA", "BBB", "CCC")
    dates = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=i)
        for i in range(5)
    ]

    coverage: dict[str, dict] = {}
    total_rows = 0

    for sym in symbols:
        sym_dir = bars_root / sym
        sym_dir.mkdir(parents=True, exist_ok=True)

        ts_list: list[int] = []
        for day in dates:
            ts_list.extend(_session_ts(day, 375))

        n = len(ts_list)
        total_rows += n

        close = (100.0 + np.arange(n, dtype=np.float32) * 0.01).astype(np.float32)
        df = pd.DataFrame(
            {
                "ts": np.asarray(ts_list, dtype=np.int64),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(n, dtype=np.float32),
            }
        )
        df.to_parquet(sym_dir / "2024.parquet", index=False)

        coverage[sym] = {
            "symbol": sym,
            "years": [2024],
            "rows": n,
            "first": datetime.datetime(2024, 1, 1, 9, 15).isoformat(),
            "last": datetime.datetime(2024, 1, 5, 15, 29).isoformat(),
        }

    manifest = {
        "resolution": "1",
        "written_at": datetime.datetime(2024, 1, 5, 0, 0).isoformat(),
        "symbols": len(symbols),
        "total_rows": total_rows,
        "adjustments": "none",
        "fingerprint": "testfingerprint",
        "coverage": coverage,
    }

    manifest_path = data_root / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return data_root, cache_root


def test_cached_checkpoint_view_real_pipeline_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, cache_root = _write_real_pipeline_data(tmp_path)

    monkeypatch.setattr(settings, "DATA_ROOT", data_root)
    monkeypatch.setattr(settings, "BARS_1M", data_root / "bars" / "1")
    monkeypatch.setattr(settings, "MANIFEST_PATH", data_root / "MANIFEST.json")
    monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(
        manifest_module, "MANIFEST_PATH", data_root / "MANIFEST.json"
    )

    start = datetime.date(2024, 1, 1)
    end = datetime.date(2024, 1, 5)
    args = {
        "times": ("11:00",),
        "fields": ("close",),
        "symbols": ("AAA", "BBB", "CCC"),
        "start": start,
        "end": end,
    }

    cp_first = cached_checkpoint_view(**args)

    default_cache_key = Manifest.load().cache_key()
    checkpoint_dir = cache_root / "checkpoints" / default_cache_key
    assert checkpoint_dir.exists()

    npz_before = list(checkpoint_dir.glob("*.npz"))
    assert len(npz_before) == 1
    mtime_before = npz_before[0].stat().st_mtime_ns

    cp_second = cached_checkpoint_view(**args)
    assert np.allclose(
        cp_first.array("close"), cp_second.array("close"), equal_nan=True
    )

    npz_after = list(checkpoint_dir.glob("*.npz"))
    assert len(npz_after) == 1
    assert npz_after[0].stat().st_mtime_ns == mtime_before

    cp_other = cached_checkpoint_view(**args, cache_key="other")
    other_dir = cache_root / "checkpoints" / "other"
    assert other_dir.exists()
    assert len(list(other_dir.glob("*.npz"))) == 1
    assert other_dir != checkpoint_dir
    assert np.allclose(
        cp_first.array("close"), cp_other.array("close"), equal_nan=True
    )


def test_to_long_index_names(panel: Panel) -> None:
    times = ("09:16", "11:00")
    cp = checkpoint_view(panel, times, ("close",))

    long_df = cp.to_long()
    assert list(long_df.index.names) == ["Timestamp", "Ticker"]
    assert len(long_df) == 10 * len(times) * 3
    assert list(long_df.columns) == list(cp.fields)


def test_get_and_array_dtype_float64(panel: Panel) -> None:
    cp = checkpoint_view(panel, ("09:16",), ("close",))

    assert cp.array("close").dtype == np.float64
    assert cp.get("close", "09:16").dtype == np.float64
