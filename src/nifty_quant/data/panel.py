from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from nifty_quant import settings
from nifty_quant.calendar import SessionGrid, to_ist
from nifty_quant.data.manifest import Manifest
from nifty_quant.guards import check_day_offsets, check_sorted_unique


@dataclass(frozen=True)
class PanelSpec:
    freq: Literal["1", "D"]
    fields: tuple[str, ...]
    symbols: tuple[str, ...]
    start: datetime.date
    end: datetime.date

    def estimated_bytes(self) -> int:
        """Rough upper-bound byte estimate for the pre-allocation memory guard."""
        calendar_days = max(1, (self.end - self.start).days + 1)
        approx_trading_days = max(1, math.ceil(calendar_days * 5 / 7))
        bars_per_day = 375 if self.freq == "1" else 1
        return int(approx_trading_days * bars_per_day * len(self.symbols) * len(self.fields) * 4)

    def key(self) -> str:
        """Stable blake2s hash of the spec, independent of tuple order and process."""
        payload = (
            self.freq
            + "|"
            + ",".join(sorted(self.fields))
            + "|"
            + ",".join(sorted(self.symbols))
            + "|"
            + self.start.isoformat()
            + "|"
            + self.end.isoformat()
        )
        return hashlib.blake2s(payload.encode("ascii"), digest_size=8).hexdigest()


class PanelTooLargeError(MemoryError):
    """Raised when a PanelSpec would exceed the configured memory limit."""


class Panel:
    """Dense aligned multi-symbol panel: fields are (n_rows, n_symbols) float32 arrays."""

    ts: np.ndarray
    day_offsets: np.ndarray
    dates: np.ndarray
    symbols: tuple[str, ...]
    sym_ix: Mapping[str, int]

    def __init__(
        self,
        fields: Mapping[str, np.ndarray],
        symbols: tuple[str, ...],
        ts: np.ndarray,
        day_offsets: np.ndarray,
        dates: np.ndarray,
    ) -> None:
        self._fields = dict(fields)
        self.symbols = tuple(symbols)
        self.ts = np.asarray(ts, dtype=np.int64)
        if self.ts.shape[0] > 0:
            check_sorted_unique(self.ts)
        self.day_offsets = np.asarray(day_offsets, dtype=np.int32)
        check_day_offsets(self.day_offsets, n_rows=self.ts.shape[0])
        self.dates = np.asarray(dates, dtype=object)
        self.sym_ix = {sym: idx for idx, sym in enumerate(self.symbols)}
        self._grid = SessionGrid(
            ts=self.ts,
            day_offsets=self.day_offsets,
            dates=self.dates,
        )

    def field(self, name: str) -> np.ndarray:
        try:
            return self._fields[name]
        except KeyError as exc:
            available = ", ".join(self._fields.keys())
            raise KeyError(f"Field {name!r} not found. Available fields: {available}") from exc

    def n_rows(self) -> int:
        return len(self.ts)

    def n_days(self) -> int:
        return len(self.dates)

    def n_symbols(self) -> int:
        return len(self.symbols)

    def day_slice(self, d: datetime.date) -> slice:
        return self._grid.day_slice(d)

    def rows_at_time(self, hhmm: str) -> np.ndarray:
        return self._grid.rows_at_time(hhmm)

    def minute_of_day(self) -> np.ndarray:
        return self._grid.minute_of_day()

    def sub(
        self,
        symbols: tuple[str, ...] | None = None,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> "Panel":
        if symbols is None and start is None and end is None:
            return self

        if symbols is None:
            selected_symbols = self.symbols
            col_indices = None
        else:
            missing = [s for s in symbols if s not in self.sym_ix]
            if missing:
                raise KeyError(f"Symbols not in panel: {missing}")
            requested_set = set(symbols)
            selected_symbols = tuple(s for s in self.symbols if s in requested_set)
            col_indices = [self.sym_ix[s] for s in selected_symbols]

        dates = self.dates
        offsets = self.day_offsets
        n_days = len(dates)

        if n_days == 0:
            start_idx = 0
            end_excl = 0
        else:
            if start is None:
                start_idx = 0
            else:
                idx = np.flatnonzero(dates >= start)
                start_idx = 0 if len(idx) == 0 else int(idx[0])

            if end is None:
                end_excl = n_days
            else:
                idx = np.flatnonzero(dates <= end)
                end_excl = 0 if len(idx) == 0 else int(idx[-1]) + 1

        if start_idx > end_excl:
            start_idx = end_excl

        row_start = int(offsets[start_idx])
        row_end = int(offsets[end_excl])
        sliced_ts = self.ts[row_start:row_end]
        grid = SessionGrid.from_timestamps(sliced_ts)

        fields_sliced = {}
        for name, arr in self._fields.items():
            view = arr[row_start:row_end]
            fields_sliced[name] = view if col_indices is None else view[:, col_indices]

        return Panel(
            fields=fields_sliced,
            symbols=tuple(selected_symbols),
            ts=grid.ts,
            day_offsets=grid.day_offsets,
            dates=grid.dates,
        )

    def to_long(self, fields: tuple[str, ...] | None = None) -> pd.DataFrame:
        if fields is None:
            field_names = list(self._fields.keys())
        else:
            field_names = []
            for name in fields:
                if name not in self._fields:
                    available = ", ".join(self._fields.keys())
                    raise KeyError(f"Field {name!r} not found. Available fields: {available}")
                field_names.append(name)

        ts_index = to_ist(self.ts)
        index = pd.MultiIndex.from_product(
            [ts_index, self.symbols],
            names=["Timestamp", "Ticker"],
        )

        data = {
            name: np.asarray(self._fields[name], dtype=np.float64).reshape(-1)
            for name in field_names
        }
        return pd.DataFrame(data, index=index)

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "Panel":
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("from_long requires a MultiIndex named ['Timestamp', 'Ticker']")
        if list(df.index.names) != ["Timestamp", "Ticker"]:
            raise ValueError("from_long requires a MultiIndex named ['Timestamp', 'Ticker']")

        ts_index = df.index.get_level_values("Timestamp")
        if not isinstance(ts_index, pd.DatetimeIndex):
            ts_index = pd.to_datetime(ts_index)

        if ts_index.tz is None:
            ts_index = ts_index.tz_localize("UTC")
        else:
            ts_index = ts_index.tz_convert("UTC")

        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        delta = ts_index - epoch
        long_ts = delta.astype("timedelta64[s]").astype(np.int64).to_numpy(dtype=np.int64)

        ticker_level = df.index.get_level_values("Ticker")
        tickers = np.asarray(ticker_level, dtype=object)
        symbols = tuple(sorted(pd.unique(ticker_level)))
        ts_sorted = np.unique(long_ts)

        work_index = pd.MultiIndex.from_arrays(
            [long_ts, tickers],
            names=["ts", "Ticker"],
        )

        fields: dict[str, np.ndarray] = {}
        for col in df.columns:
            s = df[col].copy(deep=False)
            s.index = work_index
            s = s.groupby(level=[0, 1]).first()
            wide = s.unstack(level="Ticker")
            wide = wide.reindex(index=ts_sorted, columns=symbols)
            fields[str(col)] = wide.to_numpy(dtype=np.float32)

        grid = SessionGrid.from_timestamps(ts_sorted)
        return cls(
            fields=fields,
            symbols=symbols,
            ts=grid.ts,
            day_offsets=grid.day_offsets,
            dates=grid.dates,
        )

    def nbytes(self) -> int:
        return sum(arr.nbytes for arr in self._fields.values())


def _materialized_dir(cache_key: str, spec: PanelSpec):
    return (
        settings.CACHE_ROOT
        / "panel"
        / f"v{settings.PANEL_VERSION}"
        / cache_key
        / spec.freq
        / "_materialized"
        / spec.key()
    )


def _write_materialized_meta(
    materialized_dir,
    cache_key: str,
    spec: PanelSpec,
    total_rows: int,
    n_symbols: int,
) -> None:
    meta = {
        "spec_key": spec.key(),
        "cache_key": cache_key,
        "n_rows": total_rows,
        "n_symbols": n_symbols,
        "fields": list(spec.fields),
        "dtype": "float32",
        "panel_version": settings.PANEL_VERSION,
        "complete": True,
    }
    (materialized_dir / "meta.json").write_text(
        json.dumps(meta, sort_keys=True),
        encoding="utf-8",
    )


def _try_open_materialized(
    materialized_dir,
    cache_key: str,
    spec: PanelSpec,
    total_rows: int,
    n_symbols: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    try:
        meta_path = materialized_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        expected_meta = {
            "spec_key": spec.key(),
            "cache_key": cache_key,
            "n_rows": total_rows,
            "n_symbols": n_symbols,
            "fields": list(spec.fields),
            "dtype": "float32",
            "panel_version": settings.PANEL_VERSION,
            "complete": True,
        }

        if not isinstance(meta, dict):
            return None
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                return None

        ts = np.lib.format.open_memmap(materialized_dir / "ts.i64.npy", mode="r")
        if ts.shape != (total_rows,) or ts.dtype != np.int64:
            return None

        fields = {}
        for field in spec.fields:
            arr = np.lib.format.open_memmap(materialized_dir / f"{field}.f32.npy", mode="r")
            if arr.shape != (total_rows, n_symbols) or arr.dtype != np.float32:
                return None
            fields[field] = arr

        return ts, fields
    except Exception:
        return None


def _write_materialized_arrays(
    tmp_dir,
    spec: PanelSpec,
    requested_symbols: tuple[str, ...],
    year_infos,
    total_rows: int,
) -> None:
    ts_final = np.lib.format.open_memmap(
        tmp_dir / "ts.i64.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )
    try:
        offset = 0
        for info in year_infos:
            rows = info["rows"]
            ts_year = np.load(info["dir"] / "ts.i64.npy", mmap_mode="r")
            ts_final[offset : offset + rows] = ts_year
            offset += rows
            del ts_year
        ts_final.flush()
    finally:
        del ts_final

    for field in spec.fields:
        arr = np.lib.format.open_memmap(
            tmp_dir / f"{field}.f32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, len(requested_symbols)),
        )
        try:
            offset = 0
            for info in year_infos:
                rows = info["rows"]
                src = np.load(info["dir"] / f"{field}.f32.npy", mmap_mode="r")
                year_sym_to_idx = {sym: idx for idx, sym in enumerate(info["symbols"])}
                dest = arr[offset : offset + rows, :]

                for col_idx, sym in enumerate(requested_symbols):
                    year_idx = year_sym_to_idx.get(sym)
                    if year_idx is None:
                        dest[:, col_idx] = np.nan
                    else:
                        dest[:, col_idx] = src[:, year_idx]

                offset += rows
                del src
            arr.flush()
        finally:
            del arr


def _build_materialized(
    materialized_dir,
    cache_key: str,
    spec: PanelSpec,
    requested_symbols: tuple[str, ...],
    year_infos,
    total_rows: int,
    *,
    force: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    materialized_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{materialized_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_dir = materialized_dir.parent / tmp_name
    tmp_dir.mkdir()

    backup_dir = None
    try:
        _write_materialized_arrays(tmp_dir, spec, requested_symbols, year_infos, total_rows)
        _write_materialized_meta(tmp_dir, cache_key, spec, total_rows, len(requested_symbols))

        if materialized_dir.exists():
            if not force:
                existing = _try_open_materialized(
                    materialized_dir,
                    cache_key,
                    spec,
                    total_rows,
                    len(requested_symbols),
                )
                if existing is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return existing

            backup_dir = materialized_dir.parent / (
                f"{materialized_dir.name}.old.{os.getpid()}.{uuid.uuid4().hex}"
            )
            try:
                os.replace(materialized_dir, backup_dir)
            except FileNotFoundError:
                backup_dir = None

        try:
            os.replace(tmp_dir, materialized_dir)
        except OSError as replace_exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if backup_dir is not None:
                shutil.rmtree(backup_dir, ignore_errors=True)

            cached = _try_open_materialized(
                materialized_dir,
                cache_key,
                spec,
                total_rows,
                len(requested_symbols),
            )
            if cached is None:
                raise RuntimeError(
                    "Failed to publish materialized panel; existing materialization is invalid."
                ) from replace_exc
            return cached

        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)

        cached = _try_open_materialized(
            materialized_dir,
            cache_key,
            spec,
            total_rows,
            len(requested_symbols),
        )
        if cached is None:
            raise RuntimeError("Materialized panel failed validation immediately after publishing.")
        return cached
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def load_panel(
    spec: PanelSpec,
    *,
    memmap: bool = True,
    memory_limit_gb: float | None = None,
    force: bool = False,
) -> Panel:
    """Build or load a Panel from the per-year cache, honouring the memory guard."""
    limit = memory_limit_gb if memory_limit_gb is not None else settings.MEMORY_LIMIT_GB
    estimated_bytes = spec.estimated_bytes()
    if estimated_bytes / 1e9 > limit:
        raise PanelTooLargeError(
            f"Estimated panel size {estimated_bytes / 1e9:.2f} GB exceeds limit {limit:.2f} GB. "
            "Narrow the year range, reduce fields, or set memmap=True."
        )

    manifest = Manifest.load()
    cache_key = manifest.cache_key()

    from nifty_quant.data import panel_builder

    years = sorted({year for year in range(spec.start.year, spec.end.year + 1)})
    panel_builder.build_panel(
        freq=spec.freq,
        years=years,
        symbols=spec.symbols,
        force=False,
        progress=False,
    )

    requested_symbols = tuple(sorted(spec.symbols))
    year_infos = []
    total_rows = 0

    for year in years:
        cdir = panel_builder.panel_cache_dir(cache_key, spec.freq, year)

        raw_symbols = json.loads((cdir / "symbols.json").read_text(encoding="utf-8"))
        with (cdir / "dates.json").open("r", encoding="utf-8") as fh:
            raw_dates = json.load(fh)
        dates = [datetime.date.fromisoformat(d) for d in raw_dates]
        with (cdir / "meta.json").open("r", encoding="utf-8") as fh:
            json.load(fh)

        ts_probe = np.load(cdir / "ts.i64.npy", mmap_mode="r")
        rows = ts_probe.shape[0]
        del ts_probe

        total_rows += rows
        year_infos.append(
            {
                "dir": cdir,
                "rows": rows,
                "symbols": raw_symbols,
                "dates": dates,
            }
        )

    if memmap:
        materialized_dir = _materialized_dir(cache_key, spec)
        materialized_dir.parent.mkdir(parents=True, exist_ok=True)

        if not force:
            cached = _try_open_materialized(
                materialized_dir,
                cache_key,
                spec,
                total_rows,
                len(requested_symbols),
            )
            if cached is not None:
                ts_final, fields_final = cached
            else:
                ts_final, fields_final = _build_materialized(
                    materialized_dir,
                    cache_key,
                    spec,
                    requested_symbols,
                    year_infos,
                    total_rows,
                    force=False,
                )
        else:
            ts_final, fields_final = _build_materialized(
                materialized_dir,
                cache_key,
                spec,
                requested_symbols,
                year_infos,
                total_rows,
                force=True,
            )
    else:
        ts_final = np.empty(total_rows, dtype=np.int64)

        offset = 0
        for info in year_infos:
            rows = info["rows"]
            ts_year = np.load(info["dir"] / "ts.i64.npy")
            ts_final[offset : offset + rows] = ts_year
            offset += rows
            del ts_year

        fields_final = {}
        for field in spec.fields:
            arr = np.empty((total_rows, len(requested_symbols)), dtype=np.float32)

            offset = 0
            for info in year_infos:
                rows = info["rows"]
                src = np.load(info["dir"] / f"{field}.f32.npy")
                year_sym_to_idx = {sym: idx for idx, sym in enumerate(info["symbols"])}
                dest = arr[offset : offset + rows, :]

                for col_idx, sym in enumerate(requested_symbols):
                    year_idx = year_sym_to_idx.get(sym)
                    if year_idx is None:
                        dest[:, col_idx] = np.nan
                    else:
                        dest[:, col_idx] = src[:, year_idx]

                offset += rows
                del src

            fields_final[field] = arr

    full_grid = SessionGrid.from_timestamps(ts_final)
    dates_full = full_grid.dates
    offsets_full = full_grid.day_offsets
    n_days_full = len(dates_full)

    if n_days_full == 0:
        start_idx = 0
        end_excl = 0
    else:
        # `spec.start` / `spec.end` are non-Optional `datetime.date` (see PanelSpec). The
        # former `is None` branches here were vestigial and unreachable: `key()`,
        # `estimated_bytes()` and the `range(spec.start.year, ...)` above all dereference
        # them first, so a None would have raised long before this point. They are removed
        # rather than pragma'd because they advertised an unbounded-range load that the rest
        # of the module cannot actually perform.
        idx = np.flatnonzero(dates_full >= spec.start)
        start_idx = 0 if len(idx) == 0 else int(idx[0])

        idx = np.flatnonzero(dates_full <= spec.end)
        end_excl = 0 if len(idx) == 0 else int(idx[-1]) + 1

    if start_idx > end_excl:
        start_idx = end_excl

    row_start = int(offsets_full[start_idx])
    row_end = int(offsets_full[end_excl])

    ts_final = ts_final[row_start:row_end]
    fields_sliced = {name: arr[row_start:row_end] for name, arr in fields_final.items()}
    grid = SessionGrid.from_timestamps(ts_final)

    return Panel(
        fields=fields_sliced,
        symbols=requested_symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )