"""Runner for the VWAP +/- k*ADR fade study (spec: ``specs/vwap_adr_fade.md``, section 5).

End-to-end grid over (k, N, notional) on real 1-minute bars, window resolved from the
holdout lock (never hand-typed). Writes ``REPORT.md``, ``grid.csv``, per-cell trade
parquet files and ``k_grid.json`` under ``--out``.

Usage (defaults per spec):
    uv run python scripts/run_vwap_adr_backtest.py

Never calls ``HoldoutLock.record_read``; the lock's read count is printed before and
after and the run aborts if it changed. Nothing under ``data/`` is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant import settings
from nifty_quant.backtest.metrics import (
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
    pbo_cscv,
    sharpe_ratio,
)
from nifty_quant.calendar import TradingCalendar
from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.data.validate import tradable_mask
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.execution.fills import SqrtImpactSlippage
from nifty_quant.research.splits import (
    HoldoutBoundaryError,
    HoldoutLock,
    default_holdout_lock_path,
)
from nifty_quant.research.vwap_adr.features import (
    SESSION_START_MINUTE,
    SQUARE_OFF_MINUTE,
    adr_pct,
    max_excursion,
    session_range_pct,
    session_vwap,
)
from nifty_quant.research.vwap_adr.report import daily_pnl, summarize
from nifty_quant.research.vwap_adr.simulate import FadeParams, simulate_fade
from nifty_quant.universe.static import load_universe, survivorship_report

OHLCV = ("open", "high", "low", "close", "volume")
K_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
K_LITERAL = 3.0
SUMMARY_KEYS = (
    "trades",
    "trades_per_day",
    "win_rate",
    "pct_exit_vwap",
    "pct_exit_square_off",
    "pct_exit_no_bar",
    "mean_gross_bps",
    "mean_net_bps",
    "median_holding_bars",
    "total_gross_pnl",
    "total_net_pnl",
    "total_costs",
    "sharpe_net",
    "sharpe_gross",
    "max_concurrent",
    "worst_trade_net_bps",
    "worst_day_net_pnl",
    "long_trades",
    "short_trades",
)


class Abort(RuntimeError):
    """A spec-defined abort condition."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_ist(ts: np.ndarray | pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s", utc=True).tz_convert(
        "Asia/Kolkata"
    )


def half_year_label(d: dt.date) -> str:
    return f"H{1 if d.month <= 6 else 2}-{d.year}"


def fmt(x: object, nd: int = 3) -> str:
    if x is None:
        return "NaN"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    xf = float(x)  # type: ignore[arg-type]
    if not math.isfinite(xf):
        return "NaN"
    if xf.is_integer() and abs(xf) < 1e15:
        return str(int(xf))
    return f"{xf:.{nd}f}"


def md_table(header: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c if isinstance(c, str) else fmt(c) for c in r) + " |")
    return "\n".join(out)


def k_label(k: float) -> str:
    return f"{k:.3f}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe", default="nifty50")
    p.add_argument("--years", type=int, default=2, help="window length in calendar years")
    p.add_argument("--adr-n", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--notional", type=float, nargs="+", default=[100000.0, 1000000.0])
    p.add_argument(
        "--warmup-days",
        type=int,
        default=60,
        help="calendar days of extra history loaded only so ADR exists on day one",
    )
    p.add_argument("--out", default="results/vwap_adr")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- steps


def resolve_universe(name: str):
    universe = load_universe(name)
    if universe.source == "all_data_symbols_fallback":
        raise Abort(
            f"universe {name!r} resolved to the silent fallback "
            "(all_data_symbols_fallback); configs/universe/<name>.yaml missing or invalid"
        )
    missing = [s for s in universe.symbols if not (settings.BARS_1M / s).is_dir()]
    if missing:
        raise Abort(f"universe {name!r}: no data/bars/1/<sym> directory for {missing}")
    return universe


def resolve_window(years: int, lock: HoldoutLock, calendar: TradingCalendar):
    usable_dates = calendar.session_dates()
    try:
        holdout_start, holdout_end = lock.holdout_range(usable_dates)
    except HoldoutBoundaryError as exc:
        raise Abort(f"holdout boundary error: {exc}") from exc
    all_dates = calendar.session_dates(usable_only=False)
    before = [d for d in all_dates if d < holdout_start]
    if not before:
        raise Abort(f"no session date before holdout_start {holdout_start}")
    end = before[-1]
    if end >= holdout_start:  # defensive; cannot happen given the filter above
        raise Abort(f"window end {end} >= holdout_start {holdout_start}")
    try:
        lower = end.replace(year=end.year - years)
    except ValueError:  # 29 Feb
        lower = end.replace(year=end.year - years, day=28)
    lower = lower + dt.timedelta(days=1)
    start = next((d for d in all_dates if d >= lower), None)
    if start is None or start > end:
        raise Abort(f"no session date in [{lower}, {end}]")
    return start, end, holdout_start, holdout_end


def build_k_grid(excursion_window: np.ndarray) -> dict:
    vals = excursion_window[np.isfinite(excursion_window)]
    if vals.size == 0:
        raise Abort("max_excursion has no finite values in the window; cannot derive k grid")
    q = {f"{qq:.2f}": float(np.quantile(vals, qq)) for qq in K_QUANTILES}
    grid = sorted({round(v, 3) for v in q.values()} | {K_LITERAL})
    grid = [k for k in grid if k > 0]
    return {
        "quantiles": q,
        "quantiles_rounded": {kk: round(v, 3) for kk, v in q.items()},
        "literal": K_LITERAL,
        "grid": grid,
        "sample_size": int(vals.size),
        "total_pairs": int(excursion_window.size),
        "frac_ge_3": float(np.mean(vals >= K_LITERAL)),
        "describe": {
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "max": float(vals.max()),
        },
    }


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = settings.REPO_ROOT / out
    adr_ns = list(args.adr_n)
    if not adr_ns or any(n < 1 for n in adr_ns):
        raise Abort("--adr-n values must be >= 1")
    n_mid = adr_ns[len(adr_ns) // 2]
    notionals = list(args.notional)
    if 100000.0 not in notionals:
        raise Abort("--notional must include 100000 (the overfitting/best-cell reference)")
    ref_notional = 100000.0

    # 1. Universe
    universe = resolve_universe(args.universe)
    symbols = tuple(universe.symbols)
    log(f"universe {universe.name}: {len(symbols)} symbols (source={universe.source})")

    # 2. Window from the holdout
    lock = HoldoutLock(path=default_holdout_lock_path())
    reads_before = lock.read_count()
    log(f"holdout lock {lock.path}: read_count before = {reads_before}")
    calendar = TradingCalendar.from_index_bars("NIFTY50")
    win_start, win_end, ho_start, ho_end = resolve_window(args.years, lock, calendar)
    log(
        f"holdout [{ho_start}, {ho_end}]; research window [{win_start}, {win_end}] "
        f"({args.years}y); warmup {args.warmup_days} calendar days"
    )
    surv = survivorship_report(universe, win_start, win_end)

    # 3. Panel
    load_start = win_start - dt.timedelta(days=args.warmup_days)
    spec = PanelSpec(freq="1", fields=OHLCV, symbols=symbols, start=load_start, end=win_end)
    tp = time.time()
    panel = load_panel(spec)
    log(
        f"panel loaded in {time.time() - tp:.1f}s: {panel.n_rows()} rows x "
        f"{panel.n_symbols()} symbols, {panel.n_days()} sessions "
        f"[{panel.dates[0]} .. {panel.dates[-1]}]"
    )
    if set(panel.symbols) != set(symbols) or len(panel.symbols) != len(symbols):
        raise Abort(f"panel symbols {panel.symbols} differ from universe {symbols}")
    symbols = tuple(panel.symbols)  # panel column order is authoritative
    dates = [d for d in panel.dates]
    n_days_panel = len(dates)
    if dates[-1] > win_end or dates[-1] >= ho_start:
        raise Abort(f"panel extends to {dates[-1]}, past window end {win_end}")

    offs = np.asarray(panel.day_offsets, dtype=np.int64)
    mod = np.asarray(panel.minute_of_day(), dtype=np.int64)
    ts = np.asarray(panel.ts, dtype=np.int64)
    o = np.asarray(panel.field("open"), dtype=np.float64)
    h = np.asarray(panel.field("high"), dtype=np.float64)
    lo = np.asarray(panel.field("low"), dtype=np.float64)
    c = np.asarray(panel.field("close"), dtype=np.float64)
    v = np.asarray(panel.field("volume"), dtype=np.float64)

    # Session classification: in-window & usable.
    unknown_dates: list[dt.date] = []
    usable = np.zeros(n_days_panel, dtype=bool)
    for i, d in enumerate(dates):
        try:
            usable[i] = calendar.is_usable(d)
        except KeyError:
            unknown_dates.append(d)
            usable[i] = False
    in_window = np.array([win_start <= d <= win_end for d in dates], dtype=bool)
    session_ok = in_window & usable
    ok_days = np.flatnonzero(session_ok)
    n_days = int(ok_days.size)
    excluded_in_window = [dates[i] for i in np.flatnonzero(in_window & ~usable)]
    if n_days == 0:
        raise Abort("no usable in-window sessions")
    # day_map[panel_day] -> index into the in-window usable series, -1 elsewhere.
    day_map = np.full(n_days_panel, -1, dtype=np.int64)
    day_map[ok_days] = np.arange(n_days, dtype=np.int64)
    ok_dates = [dates[i] for i in ok_days]
    log(
        f"sessions: {n_days_panel} in panel, {int(in_window.sum())} in window, "
        f"{n_days} in-window usable; excluded in-window: {excluded_in_window}; "
        f"unknown to calendar: {unknown_dates}"
    )

    tradable_raw = tradable_mask(panel)
    row_session_ok = np.repeat(session_ok, np.diff(offs))
    tradable = tradable_raw & row_session_ok[:, None]
    present = np.isfinite(c)
    log(
        f"tradable: {int(tradable_raw.sum())} raw tradable bars, {int(tradable.sum())} after "
        f"session filter; present bars {int(present.sum())} "
        f"(tradable/present in-window = "
        f"{tradable.sum() / max(1, (present & row_session_ok[:, None]).sum()):.4f})"
    )

    # 4. Features (once per N)
    tf = time.time()
    vwap = session_vwap(h, lo, c, v, mod, offs)
    rng = session_range_pct(h, lo, c, mod, offs)
    adr = {n: adr_pct(rng, n) for n in adr_ns}
    log(f"features computed in {time.time() - tf:.1f}s (N={adr_ns})")
    # ADR availability on the first in-window day.
    for n in adr_ns:
        first = ok_days[0]
        n_fin = int(np.isfinite(adr[n][first]).sum())
        log(f"  ADR N={n}: finite on first window session for {n_fin}/{len(symbols)} symbols")

    # 5. k grid (before any P&L)
    exc = max_excursion(c, vwap, adr[n_mid], mod, offs)
    exc_window = exc[session_ok]
    kg = build_k_grid(exc_window)
    kg.update({"adr_n": n_mid, "window": [str(win_start), str(win_end)], "n_sessions": n_days})
    k_grid = kg["grid"]
    log(
        f"k grid (N={n_mid}, sample={kg['sample_size']}): quantiles "
        + ", ".join(f"q{kk}={vv:.4f}" for kk, vv in kg["quantiles"].items())
        + f"; frac>=3.0={kg['frac_ge_3']:.4f}; grid={k_grid}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "k_grid.json").write_text(json.dumps(kg, indent=2))

    # 6. Grid run
    cost_model = NSEIntradayEquityCosts()
    slippage = SqrtImpactSlippage()
    rows: list[dict] = []
    trades_by_cell: dict[tuple[float, int, float], pd.DataFrame] = {}
    daily_by_cell: dict[tuple[float, int, float], np.ndarray] = {}
    n_cells = len(k_grid) * len(adr_ns) * len(notionals)
    i_cell = 0
    for n in adr_ns:
        for k in k_grid:
            for notional in notionals:
                i_cell += 1
                tc = time.time()
                params = FadeParams(
                    k=k,
                    notional=notional,
                    start_minute=SESSION_START_MINUTE,
                    square_off_minute=SQUARE_OFF_MINUTE,
                )
                tr = simulate_fade(
                    o,
                    c,
                    v,
                    vwap,
                    adr[n],
                    tradable,
                    mod,
                    offs,
                    symbols,
                    params,
                    cost_model=cost_model,
                    slippage=slippage,
                )
                panel_day = tr["day_idx"].to_numpy(dtype=np.int64)
                before_window = np.array([dates[d] < win_start for d in panel_day], dtype=bool)
                if before_window.any():
                    raise Abort(
                        f"cell k={k} N={n}: {int(before_window.sum())} trades before window "
                        "start (impossible by construction: tradable is False there)"
                    )
                mapped = day_map[panel_day]
                if (mapped < 0).any():
                    raise Abort(
                        f"cell k={k} N={n}: {int((mapped < 0).sum())} trades on sessions "
                        "outside the in-window usable set"
                    )
                tr = tr.rename(columns={"day_idx": "panel_day_idx"})
                tr.insert(1, "day_idx", mapped)
                tr["entry_ts"] = ts[tr["entry_row"].to_numpy(dtype=np.int64)]
                tr["exit_ts"] = ts[tr["exit_row"].to_numpy(dtype=np.int64)]
                tr["session_date"] = [ok_dates[d] for d in tr["day_idx"]]
                summ = summarize(tr, n_days)
                row = {"k": k, "adr_n": n, "notional": notional, **summ}
                rows.append(row)
                key = (k, n, notional)
                trades_by_cell[key] = tr
                daily_by_cell[key] = daily_pnl(tr, n_days, "net_pnl")
                pq = tr.copy()
                pq["session_date"] = pq["session_date"].astype(str)
                pq.to_parquet(out / f"trades_{k_label(k)}_{n}_{int(notional)}.parquet")
                log(
                    f"cell {i_cell}/{n_cells} k={k_label(k)} N={n} notional={int(notional)}: "
                    f"{int(summ['trades'])} trades, win {fmt(summ['win_rate'])}, "
                    f"mean_net_bps {fmt(summ['mean_net_bps'], 2)}, "
                    f"sharpe_net {fmt(summ['sharpe_net'], 2)} ({time.time() - tc:.1f}s)"
                )
    grid = pd.DataFrame(rows, columns=["k", "adr_n", "notional", *SUMMARY_KEYS])
    grid.to_csv(out / "grid.csv", index=False)

    # 7. Overfitting controls at the reference notional
    cells_ref = [(k, n) for n in adr_ns for k in k_grid]
    mat = np.column_stack(
        [daily_by_cell[(k, n, ref_notional)] / ref_notional for k, n in cells_ref]
    )
    col_std = mat.std(axis=0, ddof=1)
    varying = col_std > 0
    of: dict = {"n_cells": len(cells_ref), "T": int(mat.shape[0])}
    of["constant_cells"] = [
        f"k={k_label(k)},N={n}" for (k, n), ok in zip(cells_ref, varying) if not ok
    ]
    mat_v = mat[:, varying]
    cells_v = [cn for cn, ok in zip(cells_ref, varying) if ok]
    of["n_eff"] = effective_n_trials(mat_v) if mat_v.shape[1] >= 1 else float("nan")
    per_sr = np.array(
        [float(np.mean(col) / np.std(col, ddof=1)) for col in mat_v.T], dtype=np.float64
    )
    of["var_trial_sharpes"] = float(np.var(per_sr, ddof=1)) if per_sr.size >= 2 else float("nan")
    ann = np.array([sharpe_ratio(col, periods_per_year=252) for col in mat_v.T])
    if per_sr.size:
        b = int(np.nanargmax(ann))
        best_kn = cells_v[b]
        of["best_cell"] = f"k={k_label(best_kn[0])},N={best_kn[1]}"
        of["best_sharpe_ann"] = float(ann[b])
        of["best_sharpe_period"] = float(per_sr[b])
        n_eff = of["n_eff"]
        if not (math.isfinite(n_eff) and n_eff >= 2.0):
            of["sr0"] = 0.0
            of["sr0_note"] = (
                f"effective_n_trials={fmt(n_eff, 4)} < 2 (or NaN): no multiple-testing "
                "penalty applied, sr0=0.0 (expected_max_sharpe requires n >= 2)"
            )
        else:
            of["sr0"] = expected_max_sharpe(n_eff, of["var_trial_sharpes"])
            of["sr0_note"] = (
                "sr0 = expected_max_sharpe(effective_n_trials, var of per-period trial "
                "Sharpes), both per-period (non-annualised)"
            )
        of["dsr"] = deflated_sharpe(mat_v[:, b], sr0=of["sr0"])
    else:
        best_kn = None
        of["dsr"] = float("nan")
    try:
        of["pbo"] = pbo_cscv(mat_v, n_splits=16)
    except ValueError as exc:
        of["pbo"] = float("nan")
        of["pbo_error"] = str(exc)
    log(
        f"overfitting: n_eff={fmt(of['n_eff'], 3)} of {len(cells_ref)} cells, best "
        f"{of.get('best_cell')} sharpe_ann={fmt(of.get('best_sharpe_ann'), 3)}, "
        f"sr0={fmt(of.get('sr0'), 5)}, DSR={fmt(of['dsr'], 4)}, PBO={fmt(of['pbo'], 4)}"
    )

    # Holdout read count check
    reads_after = lock.read_count()
    log(f"holdout lock read_count after = {reads_after}")
    if reads_after != reads_before:
        raise Abort(f"holdout read count changed: {reads_before} -> {reads_after}")

    # 8. Report
    report = build_report(
        args=args,
        universe=universe,
        surv=surv,
        win=(win_start, win_end),
        holdout=(ho_start, ho_end),
        reads=(reads_before, reads_after),
        load_start=load_start,
        n_days=n_days,
        excluded_in_window=excluded_in_window,
        unknown_dates=unknown_dates,
        kg=kg,
        grid=grid,
        adr_ns=adr_ns,
        k_grid=k_grid,
        notionals=notionals,
        ref_notional=ref_notional,
        trades_by_cell=trades_by_cell,
        daily_by_cell=daily_by_cell,
        ok_dates=ok_dates,
        of=of,
        best_kn=best_kn,
        symbols=symbols,
        runtime=time.time() - t0,
    )
    (out / "REPORT.md").write_text(report)
    log(f"wrote {out / 'REPORT.md'}, grid.csv, k_grid.json, {len(trades_by_cell)} parquet files")
    log(f"total runtime {time.time() - t0:.1f}s")
    return 0


# --------------------------------------------------------------------------- report


def build_report(**kw) -> str:
    args = kw["args"]
    win_start, win_end = kw["win"]
    ho_start, ho_end = kw["holdout"]
    kg = kw["kg"]
    grid: pd.DataFrame = kw["grid"]
    adr_ns = kw["adr_ns"]
    k_grid = kw["k_grid"]
    notionals = kw["notionals"]
    ref = kw["ref_notional"]
    trades_by_cell = kw["trades_by_cell"]
    daily_by_cell = kw["daily_by_cell"]
    ok_dates = kw["ok_dates"]
    of = kw["of"]
    n_days = kw["n_days"]
    L: list[str] = []
    excl_str = [str(d) for d in kw["excluded_in_window"]] or "none"
    L.append("# VWAP +/- k*ADR fade: backtest report")
    L.append("")
    L.append(
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by "
        f"scripts/run_vwap_adr_backtest.py (runtime {kw['runtime']:.0f}s). "
        f"Spec: specs/vwap_adr_fade.md."
    )
    L.append("")

    # Caveats first
    L.append("## Caveats (read first)")
    L.append("")
    L.append(
        f"- Survivorship: universe `{kw['universe'].name}` (source `{kw['universe'].source}`) "
        "is the CURRENT constituent list applied to the whole window; not point-in-time "
        f"membership. {kw['surv'].warning_line()}"
    )
    L.append(
        f"- Holdout untouched: holdout = [{ho_start}, {ho_end}] from the lock file. Research "
        f"window = [{win_start}, {win_end}] ({args.years} calendar years ending at the last "
        f"session strictly before holdout_start). Bars loaded from {kw['load_start']} "
        f"({args.warmup_days} calendar days warmup, used only for ADR/ADV history; no trade "
        f"before {win_start}). Holdout read count before/after: "
        f"{kw['reads'][0]}/{kw['reads'][1]} (record_read never called)."
    )
    L.append(
        "- No stop loss: positions exit only on VWAP reversion (next fillable bar's open) or "
        "15:20 square-off. Tail losses are unbounded intraday."
    )
    L.append(
        "- Fills: entry at the NEXT bar's open after the signal bar's close (never at the "
        "band or signal close); the next bar must be present, have volume > 0 and be "
        "tradable, else the signal is dropped. Fractional share quantities."
    )
    L.append(
        "- Slippage: SqrtImpactSlippage() defaults (half_spread 1.5 bps + 10*sqrt(notional / "
        "bar traded value)) are ASSUMED, not calibrated against real fills. Costs: "
        "NSEIntradayEquityCosts() defaults."
    )
    L.append(
        "- Tradable mask: nifty_quant.data.validate.tradable_mask with its default ADV floor "
        "min_adv_inr=5e7 (Rs 5 crore, 20-session trailing mean of close*volume). This floor "
        "is a pre-existing repo constant, NOT derived from a measured null (CLAUDE.md rule 8 "
        "not satisfied for it); it also excludes circuit-locked and stale bars."
    )
    L.append(
        f"- Sessions: {n_days} in-window usable sessions (TradingCalendar.is_usable). "
        f"In-window sessions excluded as unusable: {excl_str}. "
        f"Panel dates unknown to the NIFTY50 calendar (treated unusable): "
        f"{[str(d) for d in kw['unknown_dates']] or 'none'}. The 09:15 bar is excluded from "
        "VWAP, range and signals."
    )
    L.append(
        f"- Multiple testing: {len(k_grid) * len(adr_ns)} (k, N) cells x {len(notionals)} "
        "notionals were run; the best cell is selected in-sample. See overfitting stats."
    )
    L.append(
        "- Daily P&L / Sharpe series include zero-P&L days; P&L is summed in rupees across "
        "symbols (no capital constraint; max_concurrent shows peak simultaneous positions)."
    )
    L.append("")

    # k-grid derivation
    L.append("## k-grid derivation (CLAUDE.md rule 8)")
    L.append("")
    L.append(
        f"Measured before any P&L: `max_excursion` = per (symbol, session) max over "
        f"09:16 <= t < 15:20 of |close - VWAP| / (VWAP * ADR_N), ADR_N with N = {kg['adr_n']} "
        f"(middle of --adr-n), over the {n_days} in-window usable sessions. Finite sample "
        f"size: {kg['sample_size']} of {kg['total_pairs']} (symbol, session) pairs. "
        f"Min {kg['describe']['min']:.4f}, mean {kg['describe']['mean']:.4f}, "
        f"max {kg['describe']['max']:.4f}."
    )
    L.append("")
    L.append(
        md_table(
            ["quantile", "value", "rounded (3 dp)"],
            [
                [q, f"{val:.6f}", f"{kg['quantiles_rounded'][q]:.3f}"]
                for q, val in kg["quantiles"].items()
            ],
        )
    )
    L.append("")
    L.append(
        f"Fraction of (symbol, session) pairs with excursion >= 3.0: {kg['frac_ge_3']:.6f}. "
        f"Grid = rounded quantiles plus literal 3.0, deduplicated and sorted: "
        f"{[k_label(k) for k in k_grid]}. The same grid is used for every N."
    )
    L.append("")

    # Summary tables
    for notional in notionals:
        L.append(f"## Summary at notional Rs {int(notional):,}")
        L.append("")
        sub = grid[grid["notional"] == notional]
        header = ["k", "N", *SUMMARY_KEYS]
        trows = [
            [k_label(r["k"]), int(r["adr_n"]), *[r[key] for key in SUMMARY_KEYS]]
            for _, r in sub.iterrows()
        ]
        L.append(md_table(header, trows))
        L.append("")

    # Per-half-year breakdown
    halves = sorted({half_year_label(d) for d in ok_dates}, key=lambda s: (s[3:], s[:2]))
    half_of_day = np.array([half_year_label(d) for d in ok_dates], dtype=object)
    for notional in notionals:
        L.append(f"## Per-half-year breakdown at Rs {int(notional):,}")
        L.append("")
        L.append(
            "Per cell and half-year: sessions, trades, mean net bps, total net P&L (Rs), "
            "annualised Sharpe of daily net P&L within that half."
        )
        L.append("")
        trows = []
        for n in adr_ns:
            for k in k_grid:
                tr = trades_by_cell[(k, n, notional)]
                dly = daily_by_cell[(k, n, notional)]
                tr_half = (
                    half_of_day[tr["day_idx"].to_numpy(dtype=np.int64)] if len(tr) else np.array([])
                )
                for hv in halves:
                    dm = half_of_day == hv
                    tm = tr_half == hv
                    nb = tr["net_bps"].to_numpy()[tm] if len(tr) else np.array([])
                    trows.append(
                        [
                            k_label(k),
                            n,
                            hv,
                            int(dm.sum()),
                            int(tm.sum()),
                            float(nb.mean()) if nb.size else float("nan"),
                            float(dly[dm].sum()),
                            sharpe_ratio(dly[dm], periods_per_year=252),
                        ]
                    )
        L.append(
            md_table(
                ["k", "N", "half", "sessions", "trades", "mean_net_bps", "net_pnl", "sharpe_net"],
                trows,
            )
        )
        L.append("")

    # Long/short breakdown
    for notional in notionals:
        L.append(f"## Long/short breakdown at Rs {int(notional):,}")
        L.append("")
        trows = []
        for n in adr_ns:
            for k in k_grid:
                tr = trades_by_cell[(k, n, notional)]
                for side, name in ((1, "long"), (-1, "short")):
                    s = tr[tr["side"] == side]
                    dly = daily_pnl(s, n_days, "net_pnl")
                    trows.append(
                        [
                            k_label(k),
                            n,
                            name,
                            len(s),
                            float((s["net_pnl"] > 0).mean()) if len(s) else float("nan"),
                            float(s["gross_bps"].mean()) if len(s) else float("nan"),
                            float(s["net_bps"].mean()) if len(s) else float("nan"),
                            float(s["net_pnl"].sum()),
                            sharpe_ratio(dly, periods_per_year=252),
                        ]
                    )
        L.append(
            md_table(
                [
                    "k",
                    "N",
                    "side",
                    "trades",
                    "win_rate",
                    "mean_gross_bps",
                    "mean_net_bps",
                    "net_pnl",
                    "sharpe_net",
                ],
                trows,
            )
        )
        L.append("")

    # Per-symbol
    sub_ref = grid[grid["notional"] == ref]
    fin = sub_ref[np.isfinite(sub_ref["sharpe_net"])]
    best_row = fin.loc[fin["sharpe_net"].idxmax()] if len(fin) else None
    per_sym_cells: list[tuple[str, float, int]] = []
    if best_row is not None:
        per_sym_cells.append(
            ("best cell by sharpe_net at Rs 1L", float(best_row["k"]), int(best_row["adr_n"]))
        )
    for n in adr_ns:
        if K_LITERAL in k_grid:
            per_sym_cells.append(("k=3.0 cell", K_LITERAL, n))
    L.append("## Per-symbol net P&L")
    L.append("")
    for label, k, n in per_sym_cells:
        L.append(f"### {label}: k={k_label(k)}, N={n}")
        L.append("")
        frames = {nt: trades_by_cell[(k, n, nt)] for nt in notionals}
        trows = []
        for sym in kw["symbols"]:
            r: list[object] = [sym]
            t_ref = frames[ref][frames[ref]["symbol"] == sym]
            r.append(len(t_ref))
            r.append(float(t_ref["net_bps"].mean()) if len(t_ref) else float("nan"))
            for nt in notionals:
                f = frames[nt]
                r.append(float(f.loc[f["symbol"] == sym, "net_pnl"].sum()))
            trows.append(r)
        trows.sort(key=lambda r: r[3 + notionals.index(ref)])
        L.append(
            md_table(
                [
                    "symbol",
                    "trades",
                    "mean_net_bps (1L)",
                    *[f"net_pnl Rs {int(nt):,}" for nt in notionals],
                ],
                trows,
            )
        )
        L.append("")

    # Overfitting
    L.append("## Overfitting statistics (Rs 1L notional)")
    L.append("")
    L.append(
        f"Trial matrix: daily net P&L / notional, {of['T']} in-window usable sessions x "
        f"{of['n_cells']} (k, N) cells."
        + (
            f" Zero-variance cells excluded: {of['constant_cells']}."
            if of["constant_cells"]
            else ""
        )
    )
    L.append("")
    L.append(
        md_table(
            ["statistic", "value"],
            [
                ["effective_n_trials", of.get("n_eff")],
                [
                    "var of per-period trial Sharpes",
                    f"{of.get('var_trial_sharpes', float('nan')):.6g}",
                ],
                ["best cell (annualised sharpe_net)", str(of.get("best_cell"))],
                ["best annualised Sharpe", of.get("best_sharpe_ann")],
                ["best per-period Sharpe", of.get("best_sharpe_period")],
                ["sr0 (per-period)", f"{of.get('sr0', float('nan')):.6g}"],
                ["deflated Sharpe (probability)", of.get("dsr")],
                ["PBO (CSCV, 16 splits)", of.get("pbo")],
            ],
        )
    )
    L.append("")
    L.append(f"sr0 note: {of.get('sr0_note', 'n/a')}.")
    if "pbo_error" in of:
        L.append(f"PBO error: {of['pbo_error']}.")
    L.append("")

    # Worst trades at 1L (distinct trades across cells)
    L.append("## 10 worst trades at Rs 1L (distinct trades across all cells)")
    L.append("")
    allt = []
    for n in adr_ns:
        for k in k_grid:
            t = trades_by_cell[(k, n, ref)]
            if len(t):
                t = t.assign(k=k, adr_n=n)
                allt.append(t)
    if allt:
        at = pd.concat(allt, ignore_index=True)
        key_cols = ["symbol", "side", "entry_row", "exit_row"]
        at["n_cells"] = at.groupby(key_cols)["net_bps"].transform("size")
        at = at.sort_values(["net_bps", "k", "adr_n"], kind="mergesort")
        at = at.drop_duplicates(subset=key_cols, keep="first").head(10)
        ent = to_ist(at["entry_ts"])
        ext = to_ist(at["exit_ts"])
        trows = []
        for (_, r), e, x in zip(at.iterrows(), ent, ext):
            trows.append(
                [
                    r["symbol"],
                    e.strftime("%Y-%m-%d %H:%M"),
                    x.strftime("%Y-%m-%d %H:%M"),
                    "long" if r["side"] == 1 else "short",
                    f"{r['entry_price']:.2f}",
                    f"{r['exit_price']:.2f}",
                    r["exit_reason"],
                    f"{r['net_bps']:.1f}",
                    f"k={k_label(r['k'])},N={int(r['adr_n'])}",
                    int(r["n_cells"]),
                ]
            )
        L.append(
            md_table(
                [
                    "symbol",
                    "entry (IST)",
                    "exit (IST)",
                    "side",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "net_bps",
                    "first cell",
                    "cells containing it",
                ],
                trows,
            )
        )
    else:
        L.append("No trades.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Abort as exc:
        print(f"ABORT: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
