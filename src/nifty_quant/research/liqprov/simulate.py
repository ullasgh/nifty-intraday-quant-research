"""Stress-gated residual liquidity-provision fade simulator (`specs/liqprov_fade.md` §2).

Long only. The band is ``vwap * (1 + beta_day * idx_dev - k * adr_day)``; a long is taken
either passively (a buy limit at ``band[t-1]`` resting from the close of ``t-1``, filled on
row ``t`` only on a strict one-tick trade-through) or at the market (a ``close <= band``
signal on a gated row, filled at the next row's open). Exits are the ``vwap_adr`` state
machine for a long (``specs/vwap_adr_fade.md`` §2.2 rules 1-3).

As in ``vwap_adr.simulate``, per-row conditions are precomputed with numpy one symbol
column at a time and the stateful walk only visits event rows, jumping between them with
``bisect``; its cost scales with the number of trades, not rows. Nothing is forward-filled
and ``tradable`` (consulted only for entry fills) is kept distinct from presence.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_quant.execution.costs import CostModel, FillBatch
from nifty_quant.research.vwap_adr.simulate import (
    _REASON_NAMES,
    _REASON_NO_BAR,
    _REASON_SQUARE_OFF,
    _REASON_VWAP,
    OUTPUT_COLUMNS,
    _empty_frame,
    _Slippage,
    _validate_day_offsets,
)

__all__ = ["LiqParams", "OUTPUT_COLUMNS", "simulate_liqprov"]

_MODES = ("passive", "market")


@dataclass(frozen=True)
class LiqParams:
    """``k`` in ADR units; ``notional`` in rupees per trade; ``tick`` in rupees."""

    k: float
    notional: float
    mode: str
    tick: float = 0.05
    start_minute: int = 556
    square_off_minute: int = 920

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 0):
            raise ValueError(f"k must be finite and > 0, got {self.k!r}")
        if not (math.isfinite(self.notional) and self.notional > 0):
            raise ValueError(f"notional must be finite and > 0, got {self.notional!r}")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if not (math.isfinite(self.tick) and self.tick > 0):
            raise ValueError(f"tick must be finite and > 0, got {self.tick!r}")
        if not self.square_off_minute > self.start_minute:
            raise ValueError(
                f"square_off_minute ({self.square_off_minute}) must be > "
                f"start_minute ({self.start_minute})"
            )


def simulate_liqprov(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vwap: np.ndarray,
    idx_dev: np.ndarray,
    beta_day: np.ndarray,
    adr_day: np.ndarray,
    gate: np.ndarray,
    tradable: np.ndarray,
    minute_of_day: np.ndarray,
    day_offsets: np.ndarray,
    symbols: Sequence[str],
    params: LiqParams,
    *,
    cost_model: CostModel | None = None,
    slippage: _Slippage | None = None,
) -> pd.DataFrame:
    """Simulate the long-only liquidity-provision fade; one output row per trade."""
    close_arr = np.asarray(close)
    if close_arr.ndim != 2:
        raise ValueError("close must be a 2-D (n_rows, n_sym) array")
    n_rows, n_sym = close_arr.shape
    shape = (n_rows, n_sym)
    per_bar = {
        "open_": np.asarray(open_),
        "high": np.asarray(high),
        "low": np.asarray(low),
        "volume": np.asarray(volume),
        "vwap": np.asarray(vwap),
        "tradable": np.asarray(tradable),
    }
    for name, arr in per_bar.items():
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    per_row = {
        "idx_dev": np.asarray(idx_dev),
        "gate": np.asarray(gate),
        "minute_of_day": np.asarray(minute_of_day),
    }
    for name, arr in per_row.items():
        if arr.ndim != 1 or arr.shape[0] != n_rows:
            raise ValueError(f"{name} must be 1-D of length n_rows ({n_rows})")
    offs = _validate_day_offsets(day_offsets, n_rows)
    n_days = offs.size - 1
    per_day = {"beta_day": np.asarray(beta_day), "adr_day": np.asarray(adr_day)}
    for name, arr in per_day.items():
        if arr.shape != (n_days, n_sym):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(n_days, n_sym)}")
    symbols_list = [str(s) for s in symbols]
    if len(symbols_list) != n_sym:
        raise ValueError(f"len(symbols) is {len(symbols_list)}, expected n_sym ({n_sym})")

    k = float(params.k)
    notional = float(params.notional)
    tick = float(params.tick)
    passive = params.mode == "passive"
    start_min = int(params.start_minute)
    sq_min = int(params.square_off_minute)

    minute = per_row["minute_of_day"].astype(np.int64)
    idx = per_row["idx_dev"].astype(np.float64)
    gate_arr = per_row["gate"].astype(bool)
    day_of_row = np.repeat(np.arange(n_days, dtype=np.int64), np.diff(offs))
    sess_end_of_row = offs[1:][day_of_row]
    sess_end_list: list[int] = sess_end_of_row.tolist()
    at_or_after_sq = minute >= sq_min
    before_sq = minute < sq_min
    in_window = (minute >= start_min) & before_sq
    # prev_ok[t]: row t-1 exists in t's session, t-1 is at/after start, t before square-off,
    # and the order resting from t-1's close was placed on a gated row.
    prev_ok = np.zeros(n_rows, dtype=bool)
    # next_ok[r]: row r+1 exists in r's session and is before square-off.
    next_ok = np.zeros(n_rows, dtype=bool)
    if n_rows > 1:
        same = day_of_row[1:] == day_of_row[:-1]
        prev_ok[1:] = same & (minute[:-1] >= start_min) & before_sq[1:] & gate_arr[:-1]
        next_ok[:-1] = same & before_sq[1:]

    rec_sym: list[int] = []
    rec_sig: list[int] = []
    rec_entry: list[int] = []
    rec_exit: list[int] = []
    rec_reason: list[int] = []
    rec_entry_px: list[float] = []
    rec_exit_px: list[float] = []
    rec_entry_btv: list[float] = []
    rec_exit_btv: list[float] = []

    with np.errstate(invalid="ignore", over="ignore"):
        for s in range(n_sym):
            c = np.asarray(close_arr[:, s], dtype=np.float64)
            o = np.asarray(per_bar["open_"][:, s], dtype=np.float64)
            lo = np.asarray(per_bar["low"][:, s], dtype=np.float64)
            v = np.asarray(per_bar["volume"][:, s], dtype=np.float64)
            vw = np.asarray(per_bar["vwap"][:, s], dtype=np.float64)
            trd = np.asarray(per_bar["tradable"][:, s], dtype=bool)
            beta_row = np.asarray(per_day["beta_day"][:, s], dtype=np.float64)[day_of_row]
            adr_row = np.asarray(per_day["adr_day"][:, s], dtype=np.float64)[day_of_row]

            present = np.isfinite(c)
            vol_pos = np.isfinite(v) & (v > 0)
            fillable = np.isfinite(o) & (o > 0) & vol_pos
            entry_fill_ok = fillable & trd

            band = vw * (1.0 + beta_row * idx - k * adr_row)
            band_ok = (
                np.isfinite(vw)
                & (vw > 0)
                & np.isfinite(idx)
                & np.isfinite(beta_row)
                & np.isfinite(adr_row)
                & (adr_row > 0)
                & np.isfinite(band)
                & (band > 0)
            )

            if passive:
                # Candidate fill rows t; the signal row is t-1 and the limit is band[t-1].
                lim = np.full(n_rows, np.nan)
                lim[1:] = band[:-1]
                lim_ok = np.zeros(n_rows, dtype=bool)
                lim_ok[1:] = band_ok[:-1]
                cand = prev_ok & lim_ok & entry_fill_ok & np.isfinite(lo) & (lo <= lim - tick)
                entry_rows: list[int] = np.flatnonzero(cand).tolist()
            else:
                # Signal rows r; fill at r+1.
                nxt = np.zeros(n_rows, dtype=bool)
                nxt[:-1] = entry_fill_ok[1:]
                cand = present & gate_arr & in_window & band_ok & (c <= band) & next_ok & nxt
                entry_rows = np.flatnonzero(cand).tolist()
            if not entry_rows:
                continue

            sq_rows: list[int] = np.flatnonzero(at_or_after_sq & fillable).tolist()
            vwap_rows: list[int] = np.flatnonzero(present & np.isfinite(vw) & (c >= vw)).tolist()
            fillable_rows: list[int] = np.flatnonzero(fillable).tolist()
            no_bar_rows: list[int] = np.flatnonzero(present & (c > 0) & vol_pos).tolist()

            n_entry = len(entry_rows)
            n_sq = len(sq_rows)
            n_vw = len(vwap_rows)
            n_fill = len(fillable_rows)
            pos = 0  # first candidate row (fill row if passive, signal row if market)
            while True:
                i = bisect_left(entry_rows, pos)
                if i >= n_entry:
                    break
                if passive:
                    e = entry_rows[i]
                    r = e - 1
                    entry_px = min(float(band[r]), float(o[e]))
                else:
                    r = entry_rows[i]
                    e = r + 1
                    entry_px = float(o[e])
                end = sess_end_list[e]

                # Rule 1 candidate: first square-off-eligible fillable row >= e.
                j = bisect_left(sq_rows, e)
                q = sq_rows[j] if j < n_sq and sq_rows[j] < end else end
                # Rule 2 candidate: first VWAP-reversion row >= e.
                j = bisect_left(vwap_rows, e)
                vr = vwap_rows[j] if j < n_vw and vwap_rows[j] < end else end

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
                                "positive volume for a no_bar exit"
                            )
                        x = no_bar_rows[j]
                        reason = _REASON_NO_BAR
                        exit_px = float(c[x])

                rec_sym.append(s)
                rec_sig.append(r)
                rec_entry.append(e)
                rec_exit.append(x)
                rec_reason.append(reason)
                rec_entry_px.append(entry_px)
                rec_exit_px.append(exit_px)
                rec_entry_btv.append(float(c[e] * v[e]))
                rec_exit_btv.append(float(c[x] * v[x]))
                # Passive: the next order rests from x's close, so it can fill at x+1.
                # Market: x's own close may signal again.
                pos = x + 1 if passive else x

    n_tr = len(rec_sym)
    if n_tr == 0:
        return _empty_frame()

    entry_px_a = np.asarray(rec_entry_px, dtype=np.float64)
    exit_px_a = np.asarray(rec_exit_px, dtype=np.float64)
    entry_row = np.asarray(rec_entry, dtype=np.int64)
    exit_row = np.asarray(rec_exit, dtype=np.int64)

    qty = notional / entry_px_a
    gross = qty * (exit_px_a - entry_px_a)
    entry_notional = qty * entry_px_a
    exit_notional = qty * exit_px_a

    if cost_model is None:
        charges = np.zeros(n_tr, dtype=np.float64)
    else:
        # One batch of 2n fills [entry buys..., exit sells...]; the models are elementwise.
        tot = np.asarray(
            cost_model.charges(
                FillBatch(
                    notional=np.concatenate([entry_notional, exit_notional]),
                    is_buy=np.concatenate([np.ones(n_tr, bool), np.zeros(n_tr, bool)]),
                )
            ).total,
            dtype=np.float64,
        )
        charges = tot[:n_tr] + tot[n_tr:]
    if slippage is None:
        slip_cost = np.zeros(n_tr, dtype=np.float64)
    else:
        exit_btv = np.asarray(rec_exit_btv, dtype=np.float64)
        exit_bps = np.asarray(slippage.bps(exit_notional, exit_btv), dtype=np.float64)
        slip_cost = exit_notional * exit_bps / 1e4
        if not passive:
            entry_btv = np.asarray(rec_entry_btv, dtype=np.float64)
            entry_bps = np.asarray(slippage.bps(entry_notional, entry_btv), dtype=np.float64)
            slip_cost = entry_notional * entry_bps / 1e4 + slip_cost
    costs = charges + slip_cost
    net = gross - costs

    sym_names = np.array(symbols_list, dtype=object)[np.asarray(rec_sym, dtype=np.int64)]
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(sym_names, dtype=object),
            "day_idx": day_of_row[entry_row],
            "side": np.ones(n_tr, dtype=np.int64),
            "signal_row": np.asarray(rec_sig, dtype=np.int64),
            "entry_row": entry_row,
            "entry_price": entry_px_a,
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
    return frame.sort_values(["entry_row", "symbol"], kind="mergesort").reset_index(drop=True)
