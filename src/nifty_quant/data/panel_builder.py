from __future__ import annotations

import datetime
import json
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

import nifty_quant.settings as settings
from nifty_quant.calendar import SessionGrid
from nifty_quant.data.manifest import Manifest
from nifty_quant.guards import check_no_all_nan_columns

FIELDS = ("open", "high", "low", "close", "volume")


def _available_years(source_root: Path) -> list[int]:
    """Return sorted years found under source_root/<SYMBOL>/<YEAR>.parquet."""
    if not source_root.exists():
        return []
    years = {
        int(p.stem)
        for p in source_root.glob("*/*.parquet")
        if p.stem.isdigit()
    }
    return sorted(years)


def _read_symbol_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read one symbol-year parquet file into float32/int64 column arrays."""
    columns = ["ts", *FIELDS]
    df = pd.read_parquet(path, columns=columns)
    if len(df) == 0:
        return {
            "ts": np.empty(0, dtype=np.int64),
            **{field: np.empty(0, dtype=np.float32) for field in FIELDS},
        }
    return {
        "ts": df["ts"].to_numpy(dtype=np.int64),
        **{field: df[field].to_numpy(dtype=np.float32) for field in FIELDS},
    }


def _read_symbols_json(path: Path) -> list[str] | None:
    """Read symbols.json and validate it is a list of strings."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list) and all(isinstance(s, str) for s in data):
        return data
    return None


def panel_cache_dir(cache_key: str, freq: str, year: int) -> Path:
    """settings.CACHE_ROOT / 'panel' / f'v{PANEL_VERSION}' / cache_key / freq / str(year)."""
    return (
        settings.CACHE_ROOT
        / "panel"
        / f"v{settings.PANEL_VERSION}"
        / cache_key
        / freq
        / str(year)
    )


def is_cached(cache_key: str, freq: str, year: int) -> bool:
    """True iff meta.json exists in the year cache directory."""
    return (panel_cache_dir(cache_key, freq, year) / "meta.json").is_file()


def build_panel(
    freq: str = "1",
    years: Sequence[int] | None = None,
    symbols: Sequence[str] | None = None,
    *,
    workers: int = 8,
    force: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Build (or reuse) a memmapped .npy cache from raw parquet bars, one year at a time."""
    manifest = Manifest.load()
    cache_key = manifest.cache_key()
    source_root = settings.BARS_1M if freq == "1" else settings.BARS_D

    if years is None:
        years = _available_years(source_root)
    else:
        years = sorted(set(years))

    output: list[Path] = []

    for year in years:
        cache_dir = panel_cache_dir(cache_key, freq, year)

        if symbols is None:
            requested_symbols = sorted(
                p.parent.name for p in source_root.glob(f"*/{year}.parquet")
                if p.is_file()
            )
        else:
            requested_symbols = sorted(set(symbols))

        if not force and is_cached(cache_key, freq, year):
            cached_symbols = _read_symbols_json(cache_dir / "symbols.json")
            if cached_symbols is not None and set(requested_symbols).issubset(
                cached_symbols
            ):
                if progress:
                    print(f"[panel_builder] {freq} {year}: reusing {cache_dir}")
                output.append(cache_dir)
                continue

        symbol_paths = [
            (sym, source_root / sym / f"{year}.parquet")
            for sym in requested_symbols
        ]
        symbols_found = [sym for sym, path in symbol_paths if path.is_file()]

        if not symbols_found:
            if cache_dir.exists() and (cache_dir / "meta.json").exists():
                (cache_dir / "meta.json").unlink()
            continue

        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "meta.json"
        if meta_path.exists():
            meta_path.unlink()

        read_paths = [path for _, path in symbol_paths if path.is_file()]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            symbol_data_list = list(executor.map(_read_symbol_arrays, read_paths))

        all_ts = [data["ts"] for data in symbol_data_list]
        union_ts = np.unique(np.concatenate(all_ts)) if all_ts else np.empty(0, dtype=np.int64)
        if union_ts.size == 0:
            continue

        n_rows = union_ts.size
        n_sym = len(symbols_found)

        field_arrays: dict[str, np.ndarray] = {}
        for field in FIELDS:
            path = cache_dir / f"{field}.f32.npy"
            arr = open_memmap(path, mode="w+", dtype=np.float32, shape=(n_rows, n_sym))
            arr.fill(np.nan)
            field_arrays[field] = arr

        ts_arr = open_memmap(
            cache_dir / "ts.i64.npy", mode="w+", dtype=np.int64, shape=(n_rows,)
        )
        ts_arr[:] = union_ts

        grid = SessionGrid.from_timestamps(union_ts)
        day_offsets_arr = open_memmap(
            cache_dir / "day_offsets.i32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(grid.day_offsets.size,),
        )
        day_offsets_arr[:] = grid.day_offsets

        for j, data in enumerate(symbol_data_list):
            row_idx = np.searchsorted(union_ts, data["ts"])
            for field in FIELDS:
                field_arrays[field][row_idx, j] = data[field]

        # Canary for silent column misalignment (e.g. a bad searchsorted index write
        # that scrambles which physical column holds which symbol's data): every
        # symbol in `symbols_found` has at least one row of real data in this year's
        # source parquet, so a fully-NaN column here means something went wrong in the
        # fill above. Restricted to `symbols_found` (not the full universe) so symbols
        # that are legitimately absent from a given year don't trigger noisy false
        # positives.
        for field in FIELDS:
            check_no_all_nan_columns(field_arrays[field], names=symbols_found)

        dates = grid.dates
        (cache_dir / "dates.json").write_text(
            json.dumps([d.isoformat() for d in dates]), encoding="utf-8"
        )
        (cache_dir / "symbols.json").write_text(
            json.dumps(symbols_found), encoding="utf-8"
        )

        for arr in field_arrays.values():
            arr.flush()
        ts_arr.flush()
        day_offsets_arr.flush()
        del field_arrays
        del ts_arr, day_offsets_arr

        meta = {
            "panel_version": settings.PANEL_VERSION,
            "fingerprint": manifest.fingerprint,
            "adjustments": manifest.adjustments,
            "freq": freq,
            "year": year,
            "n_rows": n_rows,
            "n_symbols": n_sym,
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        if progress:
            print(
                f"[panel_builder] {freq} {year}: {n_rows} rows x {n_sym} symbols -> {cache_dir}"
            )
        output.append(cache_dir)

    return output


def gc_orphans(dry_run: bool = True) -> list[Path]:
    """Remove panel cache_key dirs that do not match the current manifest fingerprint."""
    base = settings.CACHE_ROOT / "panel" / f"v{settings.PANEL_VERSION}"
    if not base.exists():
        return []

    current_key = Manifest.load().cache_key()
    orphans = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name != current_key),
        key=lambda p: p.name,
    )
    if not dry_run:
        for path in orphans:
            shutil.rmtree(path)
    return orphans
