"""VWAP +/- k*ADR fade simulator (spec `specs/vwap_adr_fade.md`, section 2).

Each symbol is simulated independently, session by session, with at most one open
position. The per-row conditions of the entry/exit state machine are precomputed with
numpy for one symbol column at a time; the stateful walk then only visits event rows,
jumping between them with ``bisect`` over sorted index lists, so its cost scales with the
number of trades rather than the number of rows.

Nothing here forward-fills: a NaN close means "no bar" and such a row can never signal,
trigger a VWAP exit or be a ``no_bar`` exit row. ``tradable`` is kept separate from
presence and is consulted only for entry fills.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from nifty_quant.execution.costs import CostModel, FillBatch

OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "day_idx",
    "side",
    "signal_row",
    "entry_row",
    "entry_price",
    "exit_row",
    "exit_price",
    "exit_reason",
    "qty",
    "gross_pnl",
    "costs",
    "net_pnl",
    "gross_bps",
    "net_bps",
    "holding_bars",
)

_INT_COLUMNS = ("day_idx", "side", "signal_row", "entry_row", "exit_row", "holding_bars")

_REASON_VWAP = 0
_REASON_SQUARE_OFF = 1
_REASON_NO_BAR = 2
_REASON_NAMES = np.array(["vwap", "square_off", "no_bar"], dtype=object)


class _Slippage(Protocol):
    def bps(self, notional: np.ndarray, bar_traded_value: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FadeParams:
    """Parameters of the fade. ``k`` is in ADR units; ``notional`` in rupees per trade."""

    k: float
    notional: float
    start_minute: int = 556
    square_off_minute: int = 920

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 0):
            raise ValueError(f"k must be finite and > 0, got {self.k!r}")
        if not (math.isfinite(self.notional) and self.notional > 0):
            raise ValueError(f"notional must be finite and > 0, got {self.notional!r}")
        if not self.square_off_minute > self.start_minute:
            raise ValueError(
                f"square_off_minute ({self.square_off_minute}) must be > "
                f"start_minute ({self.start_minute})"
            )


def _empty_frame() -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for col in OUTPUT_COLUMNS:
        if col in ("symbol", "exit_reason"):
            data[col] = pd.Series([], dtype=object)
        elif col in _INT_COLUMNS:
            data[col] = pd.Series([], dtype=np.int64)
        else:
            data[col] = pd.Series([], dtype=np.float64)
    return pd.DataFrame(data, columns=list(OUTPUT_COLUMNS))


def _validate_day_offsets(day_offsets: np.ndarray, n_rows: int) -> np.ndarray:
    offs = np.asarray(day_offsets)
    if offs.ndim != 1 or offs.size < 1:
        raise ValueError("day_offsets must be a non-empty 1-D array")
    if not np.issubdtype(offs.dtype, np.integer):
        raise ValueError("day_offsets must be an integer array")
    offs = offs.astype(np.int64)
    if offs[0] != 0:
        raise ValueError("day_offsets[0] must be 0")
    if offs[-1] != n_rows:
        raise ValueError(f"day_offsets[-1] must equal n_rows ({n_rows}), got {offs[-1]}")
    if offs.size > 1 and not np.all(np.diff(offs) > 0):
        raise ValueError("day_offsets must be strictly increasing")
    return offs


def simulate_fade(
    open_: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vwap: np.ndarray,
    adr_day: np.ndarray,
    tradable: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    symbols: Sequence[str],
    params: FadeParams,
    *,
    cost_model: CostModel | None = None,
    slippage: _Slippage | None = None,
) -> pd.DataFrame:
    """Simulate the VWAP +/- k*ADR fade; one output row per completed trade."""
    close_arr = np.asarray(close)
    if close_arr.ndim != 2:
        raise ValueError("close must be a 2-D (n_rows, n_sym) array")
    n_rows, n_sym = close_arr.shape
    shape = (n_rows, n_sym)
    per_bar = {
        "open_": np.asarray(open_),
        "volume": np.asarray(volume),
        "vwap": np.asarray(vwap),
        "tradable": np.asarray(tradable),
    }
    for name, arr in per_bar.items():
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    minute_arr = np.asarray(minute_of_day)
    if minute_arr.ndim != 1 or minute_arr.shape[0] != n_rows:
        raise ValueError(f"minute_of_day must be 1-D of length n_rows ({n_rows})")
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.size - 1
    adr_arr = np.asarray(adr_day)
    if adr_arr.shape != (n_days, n_sym):
        raise ValueError(f"adr_day has shape {adr_arr.shape}, expected {(n_days, n_sym)}")
    symbols_list = [str(s) for s in symbols]
    if len(symbols_list) != n_sym:
        raise ValueError(f"len(symbols) is {len(symbols_list)}, expected n_sym ({n_sym})")

    k = float(params.k)
    notional = float(params.notional)
    start_min = int(params.start_minute)
    sq_min = int(params.square_off_minute)

    minute = minute_arr.astype(np.int64)
    day_of_row = np.repeat(np.arange(n_days, dtype=np.int64), np.diff(offs))
    sess_end_of_row = offs[1:][day_of_row] if n_rows else np.zeros(0, dtype=np.int64)
    # Row r+1 exists and is in the same session as row r.
    has_next_same = np.zeros(n_rows, dtype=bool)
    if n_rows:
        has_next_same = np.arange(n_rows, dtype=np.int64) + 1 < sess_end_of_row
    in_window = (minute >= start_min) & (minute < sq_min)
    next_before_sq = np.zeros(n_rows, dtype=bool)
    next_before_sq[:-1] = minute[1:] < sq_min
    at_or_after_sq = minute >= sq_min
    sess_end_list: list[int] = sess_end_of_row.tolist()

    # Per-trade records, gathered across symbols.
    rec_sym: list[int] = []
    rec_side: list[int] = []
    rec_sig: list[int] = []
    rec_entry: list[int] = []
    rec_exit: list[int] = []
    rec_reason: list[int] = []
    rec_entry_px: list[float] = []
    rec_exit_px: list[float] = []
    rec_exit_btv: list[float] = []
    rec_entry_btv: list[float] = []

    tradable_all = per_bar["tradable"]
    with np.errstate(invalid="ignore", over="ignore"):
        for s in range(n_sym):
            c = np.asarray(close_arr[:, s], dtype=np.float64)
            o = np.asarray(per_bar["open_"][:, s], dtype=np.float64)
            v = np.asarray(per_bar["volume"][:, s], dtype=np.float64)
            vw = np.asarray(per_bar["vwap"][:, s], dtype=np.float64)
            trd = np.asarray(tradable_all[:, s], dtype=bool)
            adr_row = np.asarray(adr_arr[:, s], dtype=np.float64)[day_of_row]

            present = np.isfinite(c)
            vwap_ok = np.isfinite(vw)
            vol_pos = np.isfinite(v) & (v > 0)
            fillable = np.isfinite(o) & (o > 0) & vol_pos
            no_bar_ok = present & (c > 0) & vol_pos

            # Entry: signal at r, filled at r+1 only if r+1 qualifies; otherwise dropped.
            sig_base = (
                present & vwap_ok & (vw > 0) & np.isfinite(adr_row) & (adr_row > 0) & in_window
            )
            upper = vw * (1.0 + k * adr_row)
            lower = vw * (1.0 - k * adr_row)
            next_ok = np.zeros(n_rows, dtype=bool)
            next_ok[:-1] = fillable[1:] & trd[1:]
            fill_ok = has_next_same & next_before_sq & next_ok
            short_sig = sig_base & (c > upper) & fill_ok
            long_sig = sig_base & (c < lower) & fill_ok
            entry_rows: list[int] = np.flatnonzero(short_sig | long_sig).tolist()
            if not entry_rows:
                continue
            is_long_row = long_sig  # rows in entry_rows are exactly one of long/short

            # Exit event rows.
            sq_rows: list[int] = np.flatnonzero(at_or_after_sq & fillable).tolist()
            vx_base = present & vwap_ok
            vwap_long_rows: list[int] = np.flatnonzero(vx_base & (c >= vw)).tolist()
            vwap_short_rows: list[int] = np.flatnonzero(vx_base & (c <= vw)).tolist()
            fillable_rows: list[int] = np.flatnonzero(fillable).tolist()
            no_bar_rows: list[int] = np.flatnonzero(no_bar_ok).tolist()

            n_entry = len(entry_rows)
            n_sq = len(sq_rows)
            n_fill = len(fillable_rows)
            pos = 0  # first row at which entry signals are evaluated
            while True:
                i = bisect_left(entry_rows, pos)
                if i >= n_entry:
                    break
                r = entry_rows[i]
                e = r + 1
                side = 1 if is_long_row[r] else -1
                end = sess_end_list[e]

                # Rule 1 candidate: first square-off-eligible fillable row >= e.
                j = bisect_left(sq_rows, e)
                q = sq_rows[j] if j < n_sq and sq_rows[j] < end else end
                # Rule 2 candidate: first VWAP-reversion row >= e.
                vrows = vwap_long_rows if side == 1 else vwap_short_rows
                j = bisect_left(vrows, e)
                vr = vrows[j] if j < len(vrows) and vrows[j] < end else end

                if q < end and q <= vr:
                    x = q
                    reason = _REASON_SQUARE_OFF
                    exit_px = float(o[x])
                else:
                    x = end
                    if vr < end:
                        j = bisect_left(fillable_rows, vr + 1)
                        if j < n_fill and fillable_rows[j] < end:
                            x = fillable_rows[j]
                    if x < end:
                        reason = _REASON_VWAP
                        exit_px = float(o[x])
                    else:
                        j = bisect_left(no_bar_rows, end) - 1
                        if j < 0 or no_bar_rows[j] < e:
                            raise ValueError(
                                f"symbol {symbols_list[s]!r}: position entered at row {e} "
                                "has no row in its session with finite positive close and "
                                "positive volume for a no_bar exit (entry bar is fillable "
                                "and tradable but has no usable close)"
                            )
                        x = no_bar_rows[j]
                        reason = _REASON_NO_BAR
                        exit_px = float(c[x])

                rec_sym.append(s)
                rec_side.append(side)
                rec_sig.append(r)
                rec_entry.append(e)
                rec_exit.append(x)
                rec_reason.append(reason)
                rec_entry_px.append(float(o[e]))
                rec_exit_px.append(exit_px)
                rec_entry_btv.append(float(c[e] * v[e]))
                rec_exit_btv.append(float(c[x] * v[x]))
                pos = x

    n_tr = len(rec_sym)
    if n_tr == 0:
        return _empty_frame()

    side_a = np.asarray(rec_side, dtype=np.int64)
    side_f = side_a.astype(np.float64)
    entry_px = np.asarray(rec_entry_px, dtype=np.float64)
    exit_px_a = np.asarray(rec_exit_px, dtype=np.float64)
    entry_row = np.asarray(rec_entry, dtype=np.int64)
    exit_row = np.asarray(rec_exit, dtype=np.int64)

    qty = notional / entry_px
    gross = side_f * qty * (exit_px_a - entry_px)
    entry_notional = qty * entry_px
    exit_notional = qty * exit_px_a
    entry_is_buy = side_a == 1

    # One batch of 2*n fills: [entries..., exits...]. Both cost models are elementwise,
    # so per-trade sums equal the per-trade computation.
    fill_notional = np.concatenate([entry_notional, exit_notional])
    fill_is_buy = np.concatenate([entry_is_buy, ~entry_is_buy])
    if cost_model is None:
        charges = np.zeros(n_tr, dtype=np.float64)
    else:
        tot = np.asarray(
            cost_model.charges(FillBatch(notional=fill_notional, is_buy=fill_is_buy)).total,
            dtype=np.float64,
        )
        charges = tot[:n_tr] + tot[n_tr:]
    if slippage is None:
        slip_cost = np.zeros(n_tr, dtype=np.float64)
    else:
        fill_btv = np.concatenate(
            [np.asarray(rec_entry_btv, dtype=np.float64), np.asarray(rec_exit_btv, np.float64)]
        )
        bps = np.asarray(slippage.bps(fill_notional, fill_btv), dtype=np.float64)
        per_fill = fill_notional * bps / 1e4
        slip_cost = per_fill[:n_tr] + per_fill[n_tr:]
    costs = charges + slip_cost
    net = gross - costs

    sym_idx = np.asarray(rec_sym, dtype=np.int64)
    sym_names = np.array(symbols_list, dtype=object)[sym_idx]
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(sym_names, dtype=object),
            "day_idx": day_of_row[entry_row],
            "side": side_a,
            "signal_row": np.asarray(rec_sig, dtype=np.int64),
            "entry_row": entry_row,
            "entry_price": entry_px,
            "exit_row": exit_row,
            "exit_price": exit_px_a,
            "exit_reason": pd.Series(
                _REASON_NAMES[np.asarray(rec_reason, dtype=np.int64)], dtype=object
            ),
            "qty": qty,
            "gross_pnl": gross,
            "costs": costs,
            "net_pnl": net,
            "gross_bps": gross / notional * 1e4,
            "net_bps": net / notional * 1e4,
            "holding_bars": exit_row - entry_row,
        },
        columns=list(OUTPUT_COLUMNS),
    )
    frame = frame.sort_values(["entry_row", "symbol"], kind="mergesort").reset_index(drop=True)
    return frame
