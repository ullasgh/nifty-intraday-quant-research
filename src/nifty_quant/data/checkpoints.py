"""Checkpoint views are a derived slice of the dense panel at specific IST time
labels declared by a strategy's `decision_times`. This module is strategy-agnostic
and never hardcodes a checkpoint list.

The 09:15 bar's open is the pre-open call-auction price and is not reliably
obtainable, so the earliest honest ENTRY checkpoint is 09:16. Requesting
"09:15" as an entry checkpoint emits a UserWarning; "09:16" does not.
"""

from __future__ import annotations

import datetime
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from nifty_quant import settings
from nifty_quant.calendar import IST
from nifty_quant.data.manifest import Manifest
from nifty_quant.data.panel import Panel, PanelSpec, load_panel


@dataclass(frozen=True)
class CheckpointPanel:
    times: tuple[str, ...]            # "HH:MM", sorted ascending
    fields: tuple[str, ...]
    symbols: tuple[str, ...]
    dates: np.ndarray                 # object array of datetime.date, len n_days
    _data: Mapping[str, np.ndarray] = field(
        repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        times = tuple(sorted(set(self.times)))
        fields = tuple(sorted(set(self.fields)))
        symbols = tuple(self.symbols)
        dates = np.asarray(self.dates, dtype=object)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "dates", dates)

    def get(self, field: str, time: str) -> np.ndarray:
        """(n_days, n_symbols) float64. KeyError if field or time missing."""
        if field not in self._data:
            available = tuple(sorted(self._data))
            raise KeyError(
                f"field {field!r} is not available; available fields: {available}"
            )
        if time not in self.times:
            raise KeyError(
                f"time {time!r} is not available; available times: {self.times}"
            )
        time_idx = self.times.index(time)
        return self._data[field][:, time_idx, :]

    def array(self, field: str) -> np.ndarray:
        """(n_days, n_times, n_symbols) float64, axis 1 ordered per self.times."""
        if field not in self._data:
            available = tuple(sorted(self._data))
            raise KeyError(
                f"field {field!r} is not available; available fields: {available}"
            )
        return self._data[field]

    def nbytes(self) -> int:
        """Total bytes across all stored field arrays."""
        return sum(arr.nbytes for arr in self._data.values())

    def to_long(self) -> pd.DataFrame:
        """MultiIndex ['Timestamp', 'Ticker'], Timestamp is TZ-aware Asia/Kolkata."""
        timestamps: list[pd.Timestamp] = []
        tickers: list[str] = []
        columns = {f: [] for f in self.fields}

        time_index = {t: i for i, t in enumerate(self.times)}

        for day_idx, day in enumerate(self.dates):
            for time_label in self.times:
                hh, mm = map(int, time_label.split(":"))
                naive = datetime.datetime(day.year, day.month, day.day, hh, mm)
                ts = pd.Timestamp(naive, tz=IST)
                time_idx = time_index[time_label]
                for sym_idx, sym in enumerate(self.symbols):
                    timestamps.append(ts)
                    tickers.append(sym)
                    for field_name in self.fields:
                        columns[field_name].append(
                            self._data[field_name][day_idx, time_idx, sym_idx]
                        )

        idx = pd.MultiIndex.from_arrays(
            [timestamps, tickers], names=["Timestamp", "Ticker"]
        )
        return pd.DataFrame(columns, index=idx)


def checkpoint_view(
    panel: Panel,
    times: Sequence[str],
    fields: Sequence[str] = ("close",),
) -> CheckpointPanel:
    """Derive a CheckpointPanel from a dense Panel at specific IST labels."""
    for label in times:
        if label == "09:15":
            warnings.warn(
                "checkpoint '09:15' is the pre-open call-auction price and is not "
                "reliably obtainable; use '09:16' or later",
                UserWarning,
            )

    times_tuple = tuple(sorted(set(times)))
    fields_tuple = tuple(sorted(set(fields)))

    n_days = panel.n_days()
    n_symbols = panel.n_symbols()
    n_times = len(times_tuple)

    arrays: dict[str, np.ndarray] = {
        f: np.full((n_days, n_times, n_symbols), np.nan, dtype=np.float64)
        for f in fields_tuple
    }

    for time_idx, label in enumerate(times_tuple):
        rows = panel.rows_at_time(label)
        if rows.size == 0:
            continue
        day_idx = np.searchsorted(panel.day_offsets, rows, side="right") - 1
        for field_name in fields_tuple:
            arrays[field_name][day_idx, time_idx, :] = panel.field(field_name)[
                rows, :
            ].astype(np.float64)

    return CheckpointPanel(
        times=times_tuple,
        fields=fields_tuple,
        symbols=panel.symbols,
        dates=panel.dates,
        _data=arrays,
    )


def cached_checkpoint_view(
    times: Sequence[str],
    fields: Sequence[str] = ("close",),
    symbols: Sequence[str] | None = None,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    *,
    cache_key: str | None = None,
) -> CheckpointPanel:
    """Memoized wrapper around `checkpoint_view`.

    Caches results as `.npz` files under
    `settings.CACHE_ROOT / "checkpoints" / cache_key / request_hash.npz`.
    """
    times_tuple = tuple(sorted(set(times)))
    fields_tuple = tuple(sorted(set(fields)))
    symbols_tuple = tuple(sorted(set(symbols))) if symbols is not None else None

    payload = (
        times_tuple,
        fields_tuple,
        symbols_tuple,
        start.isoformat() if start is not None else None,
        end.isoformat() if end is not None else None,
    )
    request_hash = hashlib.blake2s(
        str(payload).encode("ascii"), digest_size=8
    ).hexdigest()

    active_cache_key = (
        Manifest.load().cache_key() if cache_key is None else cache_key
    )
    path = (
        settings.CACHE_ROOT
        / "checkpoints"
        / active_cache_key
        / f"{request_hash}.npz"
    )

    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            times_loaded = tuple(str(x) for x in data["times"].tolist())
            fields_loaded = tuple(str(x) for x in data["fields"].tolist())
            symbols_loaded = tuple(str(x) for x in data["symbols"].tolist())
            dates_loaded = data["dates"]

            arrays: dict[str, np.ndarray] = {}
            for field_name in fields_loaded:
                key = f"field__{field_name}"
                if key not in data.files:
                    raise KeyError(f"cache file {path} is missing array {key!r}")
                arrays[field_name] = data[key]

        return CheckpointPanel(
            times=times_loaded,
            fields=fields_loaded,
            symbols=symbols_loaded,
            dates=dates_loaded,
            _data=arrays,
        )

    if symbols is None:
        raise ValueError(
            "`symbols` is required to build a new cached checkpoint view; "
            "pass an explicit list of symbols"
        )
    if start is None or end is None:
        raise ValueError(
            "`start` and `end` are required to build a new cached checkpoint view"
        )

    spec = PanelSpec(
        freq="1",
        fields=fields_tuple,
        symbols=symbols_tuple,
        start=start,
        end=end,
    )
    panel = load_panel(spec)
    cp = checkpoint_view(panel, times_tuple, fields_tuple)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, np.ndarray] = {
        "times": np.asarray(cp.times, dtype=str),
        "fields": np.asarray(cp.fields, dtype=str),
        "symbols": np.asarray(cp.symbols, dtype=str),
        "dates": cp.dates,
    }
    for field_name in cp.fields:
        save_payload[f"field__{field_name}"] = cp._data[field_name]

    np.savez(path, **save_payload)
    return cp
