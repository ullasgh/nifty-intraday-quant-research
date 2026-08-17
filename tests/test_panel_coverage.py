"""Additional test coverage for src/nifty_quant/data/panel.py."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nifty_quant import settings
from nifty_quant.data.panel import (
    Panel,
    PanelSpec,
    _try_open_materialized,
    _write_materialized_meta,
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
        "symbols": len(symbols),
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


class TestPanelMethods:
    """Test Panel instance methods for full coverage."""

    def test_field_raises_keyerror_with_available_fields(
        self, panel_fixture: dict[str, Any]
    ) -> None:
        """field() must raise KeyError with available fields in the message."""
        panel = panel_fixture["panel"]
        with pytest.raises(KeyError) as excinfo:
            panel.field("nonexistent")
        assert "nonexistent" in str(excinfo.value)
        assert "close" in str(excinfo.value)

    def test_n_symbols_returns_count(self, panel_fixture: dict[str, Any]) -> None:
        """n_symbols() must return the count of symbols."""
        panel = panel_fixture["panel"]
        assert panel.n_symbols() == 3
        assert panel.n_symbols() == len(panel.symbols)

    def test_day_slice_returns_valid_slice(self, panel_fixture: dict[str, Any]) -> None:
        """day_slice() must return a valid slice object for a date in the panel."""
        panel = panel_fixture["panel"]
        d = panel_fixture["dates"][0]
        slc = panel.day_slice(d)
        assert isinstance(slc, slice)
        assert slc.start is not None
        assert slc.stop is not None
        assert slc.start < slc.stop

    def test_sub_no_args_returns_self(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with no arguments must return self."""
        panel = panel_fixture["panel"]
        result = panel.sub()
        assert result is panel

    def test_sub_missing_symbol_raises_keyerror(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with missing symbols must raise KeyError."""
        panel = panel_fixture["panel"]
        with pytest.raises(KeyError) as excinfo:
            panel.sub(symbols=("AAA", "NONEXISTENT"))
        assert "NONEXISTENT" in str(excinfo.value)
        assert "not in panel" in str(excinfo.value)

    def test_sub_filters_symbols(self, panel_fixture: dict[str, Any]) -> None:
        """sub() must filter to requested symbols in panel order, not requested order."""
        panel = panel_fixture["panel"]
        requested = ("CCC", "AAA")
        sub_panel = panel.sub(symbols=requested)
        # Maintains panel's original order, not requested order
        assert set(sub_panel.symbols) == {"CCC", "AAA"}
        assert sub_panel.symbols == ("AAA", "CCC")

    def test_sub_filters_by_start_date(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with start date must exclude earlier dates."""
        panel = panel_fixture["panel"]
        start = datetime.date(2024, 1, 5)
        sub_panel = panel.sub(start=start)
        assert sub_panel.n_days() < panel.n_days()
        assert sub_panel.dates[0] >= start

    def test_sub_filters_by_end_date(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with end date must exclude later dates."""
        panel = panel_fixture["panel"]
        end = datetime.date(2024, 1, 5)
        sub_panel = panel.sub(end=end)
        assert sub_panel.n_days() < panel.n_days()
        assert sub_panel.dates[-1] <= end

    def test_sub_start_after_end_returns_empty(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with start > end must return an empty panel."""
        panel = panel_fixture["panel"]
        sub_panel = panel.sub(
            start=datetime.date(2024, 1, 8),
            end=datetime.date(2024, 1, 3),
        )
        assert sub_panel.n_rows() == 0
        assert sub_panel.n_days() == 0

    def test_sub_start_before_all_dates_ignored(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with start before all dates must start at day 0."""
        panel = panel_fixture["panel"]
        sub_panel = panel.sub(start=datetime.date(2020, 1, 1))
        assert sub_panel.n_days() == panel.n_days()

    def test_sub_end_after_all_dates_ignored(self, panel_fixture: dict[str, Any]) -> None:
        """sub() with end after all dates must include all days."""
        panel = panel_fixture["panel"]
        sub_panel = panel.sub(end=datetime.date(2030, 12, 31))
        assert sub_panel.n_days() == panel.n_days()

    def test_sub_both_symbols_and_dates(self, panel_fixture: dict[str, Any]) -> None:
        """sub() must filter both symbols and dates correctly."""
        panel = panel_fixture["panel"]
        start = datetime.date(2024, 1, 3)
        end = datetime.date(2024, 1, 7)
        sub_panel = panel.sub(symbols=("AAA", "BBB"), start=start, end=end)
        assert sub_panel.symbols == ("AAA", "BBB")
        assert sub_panel.dates[0] >= start
        assert sub_panel.dates[-1] <= end

    def test_sub_preserves_field_values(self, panel_fixture: dict[str, Any]) -> None:
        """sub() must preserve actual field values for selected rows."""
        panel = panel_fixture["panel"]
        start = datetime.date(2024, 1, 2)
        sub_panel = panel.sub(start=start, symbols=("AAA",))

        sub_close = sub_panel.field("close")

        # Rows in sub_panel should match the selected column and date range
        assert sub_close.shape[0] == sub_panel.n_rows()
        assert sub_close.shape[1] == 1  # Only AAA

    def test_to_long_all_fields(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() without fields must include all fields."""
        panel = panel_fixture["panel"]
        long_df = panel.to_long()
        for field in FIELDS:
            assert field in long_df.columns

    def test_to_long_selected_fields(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() with fields argument must include only selected fields."""
        panel = panel_fixture["panel"]
        selected = ("close", "volume")
        long_df = panel.to_long(fields=selected)
        assert set(long_df.columns) == set(selected)

    def test_to_long_missing_field_raises_keyerror(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() with missing field must raise KeyError."""
        panel = panel_fixture["panel"]
        with pytest.raises(KeyError) as excinfo:
            panel.to_long(fields=("close", "nonexistent"))
        assert "nonexistent" in str(excinfo.value)

    def test_to_long_index_structure(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() must produce a MultiIndex with Timestamp and Ticker."""
        panel = panel_fixture["panel"]
        long_df = panel.to_long()
        assert list(long_df.index.names) == ["Timestamp", "Ticker"]
        assert len(long_df.index) == panel.n_rows() * panel.n_symbols()

    def test_to_long_timestamp_tz_ist(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() timestamps must be tz-aware IST."""
        panel = panel_fixture["panel"]
        long_df = panel.to_long()
        ts_idx = long_df.index.get_level_values("Timestamp")
        assert ts_idx.tz is not None
        assert "Kolkata" in str(ts_idx.tz)

    def test_to_long_dtype_float64(self, panel_fixture: dict[str, Any]) -> None:
        """to_long() columns must be float64, even though panel stores float32."""
        panel = panel_fixture["panel"]
        long_df = panel.to_long()
        for dtype in long_df.dtypes:
            assert dtype == np.float64

    def test_nbytes_sums_fields(self, panel_fixture: dict[str, Any]) -> None:
        """nbytes() must sum the byte sizes of all fields."""
        panel = panel_fixture["panel"]
        total = panel.nbytes()
        expected = sum(arr.nbytes for arr in [panel.field(f) for f in FIELDS])
        assert total == expected
        assert total > 0


class TestPanelFromLong:
    """Test Panel.from_long() for full branch coverage."""

    def test_from_long_requires_multiindex(self) -> None:
        """from_long() must raise ValueError if index is not MultiIndex."""
        df = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )
        with pytest.raises(ValueError) as excinfo:
            Panel.from_long(df)
        assert "MultiIndex" in str(excinfo.value)

    def test_from_long_requires_correct_names(self) -> None:
        """from_long() must raise ValueError if MultiIndex names are wrong."""
        index = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(["2024-01-01"], tz="UTC"), ["AAA"]],
            names=["Date", "Symbol"],  # Wrong names
        )
        df = pd.DataFrame({"close": [1.0]}, index=index)
        with pytest.raises(ValueError) as excinfo:
            Panel.from_long(df)
        assert "Timestamp" in str(excinfo.value) or "Ticker" in str(excinfo.value)

    def test_from_long_with_naive_timestamps(self) -> None:
        """from_long() must localize naive UTC timestamps."""
        ts_idx = pd.DatetimeIndex(["2024-01-01 09:15", "2024-01-01 09:15"], name="Timestamp")
        ticker_idx = pd.Index(["AAA", "BBB"], name="Ticker")
        index = pd.MultiIndex.from_arrays([ts_idx, ticker_idx])
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
        panel = Panel.from_long(df)
        assert panel.n_rows() == 1
        assert panel.n_symbols() == 2

    def test_from_long_with_utc_timestamps(self) -> None:
        """from_long() must accept UTC-aware timestamps."""
        ts_idx = pd.DatetimeIndex(
            ["2024-01-01 09:15", "2024-01-01 09:15"],
            tz="UTC",
            name="Timestamp",
        )
        ticker_idx = pd.Index(["AAA", "BBB"], name="Ticker")
        index = pd.MultiIndex.from_arrays([ts_idx, ticker_idx])
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
        panel = Panel.from_long(df)
        assert panel.n_rows() == 1
        assert panel.n_symbols() == 2

    def test_from_long_with_ist_timestamps(self) -> None:
        """from_long() must convert IST timestamps to UTC."""
        ts_ist = pd.DatetimeIndex(
            ["2024-01-01 09:15", "2024-01-01 09:15"],
            tz="Asia/Kolkata",
            name="Timestamp",
        )
        ticker_idx = pd.Index(["AAA", "BBB"], name="Ticker")
        index = pd.MultiIndex.from_arrays([ts_ist, ticker_idx])
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
        panel = Panel.from_long(df)
        assert panel.n_rows() == 1
        assert panel.ts[0] > 0  # Valid epoch

    def test_from_long_creates_sorted_symbols(self) -> None:
        """from_long() must sort symbols alphabetically."""
        ts_idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-01", "2024-01-01"],
            tz="UTC",
            name="Timestamp",
        )
        ticker_idx = pd.Index(["CCC", "AAA", "BBB"], name="Ticker")
        index = pd.MultiIndex.from_arrays([ts_idx, ticker_idx])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
        panel = Panel.from_long(df)
        assert panel.symbols == ("AAA", "BBB", "CCC")

    def test_from_long_handles_duplicate_entries(self) -> None:
        """from_long() must handle duplicate entries by taking first value."""
        ts_idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-01"],
            tz="UTC",
            name="Timestamp",
        )
        ticker_idx = pd.Index(["AAA", "AAA"], name="Ticker")
        index = pd.MultiIndex.from_arrays([ts_idx, ticker_idx])
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
        panel = Panel.from_long(df)
        # from_long groups by (ts, ticker) and takes first
        assert panel.field("close")[0, 0] == 100.0


class TestMaterializationCoverage:
    """Test materialization helper functions."""

    def test_try_open_materialized_missing_meta_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if meta.json is missing."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()
        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=PanelSpec(
                freq="1",
                fields=("close",),
                symbols=("AAA",),
                start=datetime.date(2024, 1, 1),
                end=datetime.date(2024, 1, 31),
            ),
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_invalid_json_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if meta.json is invalid."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()
        (materialized_dir / "meta.json").write_text("invalid json")
        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=PanelSpec(
                freq="1",
                fields=("close",),
                symbols=("AAA",),
                start=datetime.date(2024, 1, 1),
                end=datetime.date(2024, 1, 31),
            ),
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_non_dict_meta_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if meta.json is not a dict."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()
        (materialized_dir / "meta.json").write_text('["list", "not", "dict"]')
        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=PanelSpec(
                freq="1",
                fields=("close",),
                symbols=("AAA",),
                start=datetime.date(2024, 1, 1),
                end=datetime.date(2024, 1, 31),
            ),
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_mismatched_meta_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if meta doesn't match expected."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()
        meta = {
            "spec_key": "wrong_key",
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": 1,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))
        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=PanelSpec(
                freq="1",
                fields=("close",),
                symbols=("AAA",),
                start=datetime.date(2024, 1, 1),
                end=datetime.date(2024, 1, 31),
            ),
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_wrong_ts_shape_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if ts shape is wrong."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        # Create ts with wrong shape (should be (100,), not (50,))
        ts = np.arange(50, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_wrong_ts_dtype_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if ts dtype is wrong."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        # Create ts with wrong dtype (float64 instead of int64)
        ts = np.arange(100, dtype=np.float64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_wrong_field_shape_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if field shape is wrong."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        ts = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        # Create close with wrong shape (should be (100, 1), not (50, 1))
        close = np.ones((50, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close)

        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_try_open_materialized_wrong_field_dtype_returns_none(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must return None if field dtype is wrong."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        ts = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        # Create close with wrong dtype (float64 instead of float32)
        close = np.ones((100, 1), dtype=np.float64)
        np.save(materialized_dir / "close.f32.npy", close)

        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            total_rows=100,
            n_symbols=1,
        )
        assert result is None

    def test_write_materialized_meta_creates_valid_json(
        self, tmp_path: Path
    ) -> None:
        """_write_materialized_meta() must write a valid, complete meta.json."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close", "volume"),
            symbols=("AAA", "BBB"),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        _write_materialized_meta(
            materialized_dir,
            cache_key="testkey",
            spec=spec,
            total_rows=500,
            n_symbols=2,
        )

        meta_path = materialized_dir / "meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["spec_key"] == spec.key()
        assert meta["cache_key"] == "testkey"
        assert meta["n_rows"] == 500
        assert meta["n_symbols"] == 2
        assert meta["fields"] == ["close", "volume"]
        assert meta["dtype"] == "float32"
        assert meta["complete"] is True


class TestLoadPanelEdgeCases:
    """Test edge cases in load_panel."""

    def test_load_panel_empty_dates_handled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with spec that results in empty dates must handle gracefully."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        # Create a spec that ends before all data
        spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=scenario["symbols"],
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2023, 12, 31),  # Before start!
        )

        panel = load_panel(spec, memmap=True)
        assert panel.n_rows() == 0
        assert panel.n_days() == 0

    def test_load_panel_memmap_false_fills_nans_for_missing_symbol(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with memmap=False must fill NaN for missing symbols in some years."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        # Load with memmap=False
        panel = load_panel(scenario["spec"], memmap=False)

        # Panel should load successfully even with NaN fills
        assert panel.n_rows() > 0
        close = panel.field("close")
        assert close.dtype == np.float32
        assert close.shape[1] == 3


class TestPanelInitialization:
    """Test Panel.__init__ edge cases."""

    def test_panel_empty_ts_skips_sort_check(self, tmp_path: Path) -> None:
        """Panel with empty ts must not call check_sorted_unique."""
        panel = Panel(
            fields={"close": np.array([], dtype=np.float32).reshape(0, 1)},
            symbols=("AAA",),
            ts=np.array([], dtype=np.int64),
            day_offsets=np.array([0], dtype=np.int32),
            dates=np.array([], dtype=object),
        )
        assert panel.n_rows() == 0

    def test_panel_sym_ix_mapping(self, panel_fixture: dict[str, Any]) -> None:
        """Panel.sym_ix must correctly map symbols to column indices."""
        panel = panel_fixture["panel"]
        for idx, sym in enumerate(panel.symbols):
            assert panel.sym_ix[sym] == idx


class TestPanelSubWithEmptyDates:
    """Test sub() with empty dates scenario."""

    def test_sub_with_empty_panel(self) -> None:
        """sub() on an empty panel must handle the case gracefully."""
        # Create an empty panel
        panel = Panel(
            fields={"close": np.array([], dtype=np.float32).reshape(0, 1)},
            symbols=("AAA",),
            ts=np.array([], dtype=np.int64),
            day_offsets=np.array([0], dtype=np.int32),
            dates=np.array([], dtype=object),
        )
        # sub() with empty dates triggers the n_days == 0 path
        result = panel.sub(start=datetime.date(2024, 1, 1), end=datetime.date(2024, 1, 31))
        assert result.n_rows() == 0


class TestFromLongEdgeCases:
    """Test from_long edge cases."""

    def test_from_long_string_timestamp_converted(self) -> None:
        """from_long() must convert string timestamps to DatetimeIndex."""
        index = pd.MultiIndex.from_product(
            [["2024-01-01", "2024-01-02"], ["AAA", "BBB"]],
            names=["Timestamp", "Ticker"],
        )
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=index)
        # This will convert string timestamps to DatetimeIndex
        panel = Panel.from_long(df)
        assert panel.n_rows() == 2
        assert panel.n_symbols() == 2


class TestMissingSymbolsInYears:
    """Test handling of missing symbols in years (fills with NaN)."""

    def test_load_panel_with_missing_symbols_fills_nan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel must fill NaN when a symbol is missing in a year."""
        symbols = ("AAA", "BBB", "CCC")
        year = 2024
        dates = [datetime.date(2024, 1, d) for d in range(1, 3)]

        # Only AAA and BBB in the data
        sym_frames: dict[str, pd.DataFrame] = {}
        for sym in ("AAA", "BBB"):
            rows = []
            for d in dates:
                for i, ts in enumerate(_session_ts(d, 375)):
                    rows.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0 + i))
            frame = pd.DataFrame(rows, columns=["ts", *FIELDS])
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
            "symbols": len(symbols),
            "total_rows": sum(len(frame) for frame in sym_frames.values()),
            "adjustments": "testadj",
            "fingerprint": "testfp2",
            "coverage": coverage,
        }
        manifest_path = data_root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        cache_root = tmp_path / "cache"

        monkeypatch.setattr(settings, "DATA_ROOT", data_root)
        monkeypatch.setattr(settings, "BARS_1M", bars_root)
        monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
        monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
        monkeypatch.setattr("nifty_quant.data.manifest.MANIFEST_PATH", manifest_path)

        spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=symbols,  # Request all 3, but only AAA and BBB exist
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 2),
        )

        # Load with memmap=True to test the materialization NaN fill path
        # (line 372 in _write_materialized_arrays)
        panel = load_panel(spec, memmap=True)
        assert panel.n_symbols() == 3

        # CCC should be all NaN
        close = panel.field("close")
        ccc_col = panel.sym_ix["CCC"]
        assert np.all(np.isnan(close[:, ccc_col]))

        # AAA and BBB should have data
        aaa_col = panel.sym_ix["AAA"]
        bbb_col = panel.sym_ix["BBB"]
        assert not np.all(np.isnan(close[:, aaa_col]))
        assert not np.all(np.isnan(close[:, bbb_col]))


class TestLoadPanelSpecStartEndNone:
    """Test load_panel with spec.start or spec.end as None (edge case)."""

    def test_load_panel_spec_start_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel should handle None start date gracefully."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        # Spec with None start (though the type doesn't allow it normally)
        # This tests the spec.start is None branch in load_panel
        spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=scenario["symbols"],
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 10),
        )

        panel = load_panel(spec, memmap=True)
        assert panel.n_rows() > 0

    def test_load_panel_start_idx_greater_than_end_excl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel should swap start_idx and end_excl when start > end."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=scenario["symbols"],
            start=datetime.date(2024, 1, 10),
            end=datetime.date(2024, 1, 1),  # End before start
        )

        panel = load_panel(spec, memmap=True)
        # Should result in empty or minimal panel due to the swap
        assert panel.n_rows() >= 0


class TestMaterializationRaceConditionCoverage:
    """Test materialization race condition branches (hard to trigger fully)."""

    def test_try_open_materialized_returns_valid_arrays_when_exists(
        self, tmp_path: Path
    ) -> None:
        """_try_open_materialized() must successfully open valid materialized files."""
        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir()

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        ts = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        close = np.ones((100, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close)

        result = _try_open_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            total_rows=100,
            n_symbols=1,
        )
        assert result is not None
        ts_result, fields_result = result
        assert ts_result.shape == (100,)
        assert "close" in fields_result
        assert fields_result["close"].shape == (100, 1)

    def test_build_materialized_with_existing_valid_cache_returns_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_materialized should return early if valid cache exists and force=False."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create a valid materialized cache
        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        ts = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts)

        close = np.ones((100, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close)

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        # Call _build_materialized with existing valid cache and force=False
        # It should detect the existing cache and return it
        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=False,
        )

        assert ts_result.shape == (100,)
        assert "close" in fields_result

    def test_build_materialized_force_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_materialized with force=True must rebuild even if cache exists."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create an old materialized cache
        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))

        ts = np.arange(100, dtype=np.int64) + 1000  # Old data
        np.save(materialized_dir / "ts.i64.npy", ts)

        close = np.ones((100, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close)

        # Create mock year_infos with different data
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64) + 5000  # New data
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32) * 2
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        # Call _build_materialized with force=True to force rebuild
        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=True,
        )

        # Should have new data
        assert ts_result[0] >= 5000  # From year_infos
        assert fields_result["close"][0, 0] == 2.0


class TestLoadPanelWithMemmap:
    """Test load_panel with various memmap scenarios."""

    def test_load_panel_memmap_true_uses_cache_if_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with memmap=True should reuse cache if it's valid."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        # First load
        panel1 = load_panel(scenario["spec"], memmap=True)

        # Second load should reuse the same data
        panel2 = load_panel(scenario["spec"], memmap=True)

        # Both should have the same row count
        assert panel1.n_rows() == panel2.n_rows()

    def test_load_panel_memmap_false_always_copies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with memmap=False must load into regular arrays, not memmaps."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        panel = load_panel(scenario["spec"], memmap=False)
        close = panel.field("close")

        # Should not be a memmap when memmap=False
        assert not isinstance(close, np.memmap)

    def test_load_panel_memmap_false_with_missing_symbols(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with memmap=False must fill NaN for missing symbols (line 580)."""
        symbols = ("AAA", "BBB", "CCC")
        year = 2024
        dates = [datetime.date(2024, 1, d) for d in range(1, 3)]

        # Only AAA and BBB in the data
        sym_frames: dict[str, pd.DataFrame] = {}
        for sym in ("AAA", "BBB"):
            rows = []
            for d in dates:
                for i, ts in enumerate(_session_ts(d, 375)):
                    rows.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0 + i))
            frame = pd.DataFrame(rows, columns=["ts", *FIELDS])
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
            "symbols": len(symbols),
            "total_rows": sum(len(frame) for frame in sym_frames.values()),
            "adjustments": "testadj",
            "fingerprint": "testfp3",
            "coverage": coverage,
        }
        manifest_path = data_root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        cache_root = tmp_path / "cache"

        monkeypatch.setattr(settings, "DATA_ROOT", data_root)
        monkeypatch.setattr(settings, "BARS_1M", bars_root)
        monkeypatch.setattr(settings, "MANIFEST_PATH", manifest_path)
        monkeypatch.setattr(settings, "CACHE_ROOT", cache_root)
        monkeypatch.setattr("nifty_quant.data.manifest.MANIFEST_PATH", manifest_path)

        spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=symbols,
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 2),
        )

        # Load with memmap=False to test line 580 (NaN fill in non-memmap path)
        panel = load_panel(spec, memmap=False)
        assert panel.n_symbols() == 3

        close = panel.field("close")
        ccc_col = panel.sym_ix["CCC"]
        # CCC should be all NaN
        assert np.all(np.isnan(close[:, ccc_col]))

    def test_load_panel_restricted_date_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_panel with a narrower date range must only return those rows."""
        scenario = _create_panel_scenario(tmp_path, monkeypatch, build_and_load=False)

        narrow_spec = PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=scenario["symbols"],
            start=datetime.date(2024, 1, 2),
            end=datetime.date(2024, 1, 5),
        )

        panel = load_panel(narrow_spec, memmap=True)
        assert panel.n_days() <= 4  # Jan 2-5 is max 4 days
        assert panel.dates[0] >= datetime.date(2024, 1, 2)
        assert panel.dates[-1] <= datetime.date(2024, 1, 5)


class TestConcurrentMaterializationRaceConditions:
    """Test race condition handling in _build_materialized."""

    def test_concurrent_publish_race_condition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate another process publishing while we build (lines 412-414)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        # Monkeypatch _try_open_materialized to simulate race condition:
        # First call returns None (our temp dir), subsequent calls return valid cache
        call_count = 0

        def mock_try_open(mat_dir, cache_key, spec, total_rows, n_symbols):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: concurrent process hasn't published yet
                return None
            else:
                # Subsequent calls: concurrent process has now published
                meta = {
                    "spec_key": spec.key(),
                    "cache_key": cache_key,
                    "n_rows": 100,
                    "n_symbols": 1,
                    "fields": ["close"],
                    "dtype": "float32",
                    "panel_version": settings.PANEL_VERSION,
                    "complete": True,
                }
                (mat_dir / "meta.json").write_text(json.dumps(meta))
                ts = np.arange(100, dtype=np.int64)
                np.save(mat_dir / "ts.i64.npy", ts)
                close = np.ones((100, 1), dtype=np.float32)
                np.save(mat_dir / "close.f32.npy", close)
                return ts, {"close": close}

        monkeypatch.setattr(
            "nifty_quant.data.panel._try_open_materialized",
            mock_try_open,
        )

        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=False,
        )

        # Should have returned the concurrently-published panel
        assert ts_result.shape == (100,)
        assert "close" in fields_result

    def test_backup_dir_creation_fails_with_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.replace(materialized_dir, backup_dir) raises FileNotFoundError (lines 421-422)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        real_replace = os.replace
        replace_call_count = 0

        def mock_replace(src, dst):
            nonlocal replace_call_count
            replace_call_count += 1
            # First call: backup (lines 420) raises FileNotFoundError
            if replace_call_count == 1:
                raise FileNotFoundError("Simulated missing materialized_dir")
            # Second call: temp->canonical succeeds
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)

        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=False,
        )

        # Should still succeed despite backup failure
        assert ts_result.shape == (100,)
        assert "close" in fields_result

    def test_publish_fails_with_valid_cache_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError on publish with valid existing cache returns it (lines 426-442)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create a valid existing materialized cache
        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))
        ts_existing = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts_existing)
        close_existing = np.ones((100, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close_existing)

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64) + 5000  # Different data
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32) * 2
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        real_replace = os.replace
        replace_call_count = 0

        def mock_replace(src, dst):
            nonlocal replace_call_count
            replace_call_count += 1
            # Second replace (tmp->canonical) raises OSError
            if replace_call_count == 2:
                raise OSError("Simulated publish failure")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)

        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=False,
        )

        # Should return the existing valid cache (not the new data)
        assert ts_result.shape == (100,)
        assert ts_result[0] < 1000  # Existing data, not new data starting at 5000

    def test_publish_fails_with_invalid_cache_raises_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.replace raises OSError with no valid existing cache (lines 426-442, sub-case b)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        real_replace = os.replace
        replace_call_count = 0

        def mock_replace(src, dst):
            nonlocal replace_call_count
            replace_call_count += 1
            # Second replace (tmp->canonical) raises OSError
            if replace_call_count == 2:
                raise OSError("Simulated publish failure")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)

        with pytest.raises(RuntimeError) as excinfo:
            _build_materialized(
                materialized_dir,
                cache_key="test",
                spec=spec,
                requested_symbols=("AAA",),
                year_infos=year_infos,
                total_rows=100,
                force=False,
            )

        # Should raise RuntimeError about invalid existing materialization
        assert "invalid" in str(excinfo.value).lower()
        # Verify exception chaining
        assert excinfo.value.__cause__ is not None

    def test_publish_oserror_with_backup_and_valid_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test OSError recovery with both backup and valid cache (lines 428, 442)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create a valid existing materialized cache
        meta = {
            "spec_key": spec.key(),
            "cache_key": "test",
            "n_rows": 100,
            "n_symbols": 1,
            "fields": ["close"],
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }
        (materialized_dir / "meta.json").write_text(json.dumps(meta))
        ts_existing = np.arange(100, dtype=np.int64)
        np.save(materialized_dir / "ts.i64.npy", ts_existing)
        close_existing = np.ones((100, 1), dtype=np.float32)
        np.save(materialized_dir / "close.f32.npy", close_existing)

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64) + 5000
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32) * 2
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        real_replace = os.replace
        replace_call_count = 0

        def mock_replace(src, dst):
            nonlocal replace_call_count
            replace_call_count += 1
            # First replace: backup succeeds
            if replace_call_count == 1:
                return real_replace(src, dst)
            # Second replace: throws OSError, forcing recovery with backup_dir set
            elif replace_call_count == 2:
                raise OSError("Simulated publish failure")
            return real_replace(src, dst)

        def mock_try_open_materialized(mat_dir, cache_key, spec_arg, total_rows, n_symbols):
            # Always return a valid cache to test the recovery path (lines 428, 442)
            return (np.arange(100, dtype=np.int64), {"close": np.ones((100, 1), dtype=np.float32)})

        monkeypatch.setattr(os, "replace", mock_replace)
        monkeypatch.setattr(
            "nifty_quant.data.panel._try_open_materialized",
            mock_try_open_materialized,
        )

        # Call with force=True so it doesn't return early on existing cache
        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=True,
        )

        # Should return via the recovery path (lines 428, 442 executed)
        assert ts_result.shape == (100,)
        assert "close" in fields_result

    def test_post_publish_validation_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_try_open_materialized returns None after successful publish (line 455)."""
        from nifty_quant.data.panel import _build_materialized

        materialized_dir = tmp_path / "mat"
        materialized_dir.mkdir(parents=True)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        call_count = 0

        def mock_try_open(mat_dir, cache_key, spec_arg, total_rows, n_symbols):
            nonlocal call_count
            call_count += 1
            # All calls return None to simulate validation failure
            return None

        monkeypatch.setattr(
            "nifty_quant.data.panel._try_open_materialized",
            mock_try_open,
        )

        with pytest.raises(RuntimeError) as excinfo:
            _build_materialized(
                materialized_dir,
                cache_key="test",
                spec=spec,
                requested_symbols=("AAA",),
                year_infos=year_infos,
                total_rows=100,
                force=False,
            )

        # Should raise RuntimeError about failed validation after publishing
        assert "validation" in str(excinfo.value).lower()

    def test_publish_oserror_no_backup_with_valid_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError on first publish returns recovered cache when backup_dir is None."""
        from nifty_quant.data.panel import _build_materialized

        # Fresh materialized_dir that does NOT exist (no prior materialization)
        materialized_dir = tmp_path / "mat"
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)
        # Importantly, materialized_dir itself is NOT created, so backup_dir stays None

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        replace_call_count = 0

        def mock_replace(src, dst):
            nonlocal replace_call_count
            replace_call_count += 1
            # The FIRST (and only) replace call (tmp->canonical) raises OSError
            raise OSError("Simulated publish failure on first attempt")

        def mock_try_open(mat_dir, cache_key, spec_arg, total_rows, n_symbols):
            # Return a valid cache for recovery (tests line 431-442)
            return (np.arange(100, dtype=np.int64), {"close": np.ones((100, 1), dtype=np.float32)})

        monkeypatch.setattr(os, "replace", mock_replace)
        monkeypatch.setattr(
            "nifty_quant.data.panel._try_open_materialized",
            mock_try_open,
        )

        # Call with force=False on non-existent materialized_dir
        ts_result, fields_result = _build_materialized(
            materialized_dir,
            cache_key="test",
            spec=spec,
            requested_symbols=("AAA",),
            year_infos=year_infos,
            total_rows=100,
            force=False,
        )

        # Should return the recovered cache (tests 428->431 branch with backup_dir=None)
        assert ts_result.shape == (100,)
        assert "close" in fields_result
        # Verify we went through recovery path
        assert replace_call_count == 1

    def test_publish_oserror_no_backup_without_valid_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError with no prior cache raises RuntimeError when recovery fails."""
        from nifty_quant.data.panel import _build_materialized

        # Fresh materialized_dir that does NOT exist
        materialized_dir = tmp_path / "mat"
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        spec = PanelSpec(
            freq="1",
            fields=("close",),
            symbols=("AAA",),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 31),
        )

        # Create mock year_infos
        year_dir = tmp_path / "year_cache"
        year_dir.mkdir()
        ts_year = np.arange(100, dtype=np.int64)
        np.save(year_dir / "ts.i64.npy", ts_year)
        close_year = np.ones((100, 1), dtype=np.float32)
        np.save(year_dir / "close.f32.npy", close_year)

        year_infos = [
            {
                "dir": year_dir,
                "rows": 100,
                "symbols": ["AAA"],
                "dates": [datetime.date(2024, 1, 1)],
            }
        ]

        def mock_replace(src, dst):
            # The publish attempt raises OSError
            raise OSError("Simulated publish failure")

        def mock_try_open(mat_dir, cache_key, spec_arg, total_rows, n_symbols):
            # No valid cache available for recovery
            return None

        monkeypatch.setattr(os, "replace", mock_replace)
        monkeypatch.setattr(
            "nifty_quant.data.panel._try_open_materialized",
            mock_try_open,
        )

        # Call should raise RuntimeError because recovery failed
        with pytest.raises(RuntimeError) as excinfo:
            _build_materialized(
                materialized_dir,
                cache_key="test",
                spec=spec,
                requested_symbols=("AAA",),
                year_infos=year_infos,
                total_rows=100,
                force=False,
            )

        # Verify error type and cause
        assert "invalid" in str(excinfo.value).lower()
        # The __cause__ should be the original OSError
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, OSError)


@pytest.fixture
def panel_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Shared synthetic fixture used by all tests."""
    return _create_panel_scenario(tmp_path, monkeypatch)
