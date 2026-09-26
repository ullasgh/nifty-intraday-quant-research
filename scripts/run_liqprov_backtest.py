"""Runner for the stress-gated residual liquidity-provision fade (spec section 5.2).

Spec: ``specs/liqprov_fade.md``. Runs the 12 pre-registered trials (q x k x mode) at
Rs 10L and Rs 1L on 2018-01-01 .. 2023-08-13 and writes ``REPORT.md``, ``grid.csv`` and
per-trial trade parquet files under ``--out``.

Refuses to run unless ``results/liqprov/prereg.json`` exists, its ``config_hash``
recomputes, and a FRESH recomputation of the whole pre-registered body (k grid, gate
thresholds and frequencies, window, universe, trials, ...) matches it. The fresh
recomputation uses the functions in ``scripts/prereg_liqprov.py`` itself.

Usage:
    uv run python scripts/run_liqprov_backtest.py
    # smoke test only (warm-up period, outside the test window, 1 trial, scratch dir):
    uv run python scripts/run_liqprov_backtest.py --smoke --out /tmp/claude-0/liqprov_smoke

Never calls ``HoldoutLock.record_read``; the read count is checked before and after and
the run aborts if it moved. Nothing under ``data/`` is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prereg_liqprov as pre  # noqa: E402

from nifty_quant import settings  # noqa: E402
from nifty_quant.backtest.metrics import (  # noqa: E402
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
    pbo_cscv,
    sharpe_ratio,
)
from nifty_quant.calendar import TradingCalendar  # noqa: E402
from nifty_quant.data.validate import tradable_mask  # noqa: E402
from nifty_quant.execution.costs import NSEIntradayEquityCosts  # noqa: E402
from nifty_quant.execution.fills import SqrtImpactSlippage  # noqa: E402
from nifty_quant.research.liqprov.evaluate import kill_criteria, yearly_net  # noqa: E402
from nifty_quant.research.liqprov.simulate import LiqParams, simulate_liqprov  # noqa: E402
from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path  # noqa: E402
from nifty_quant.research.vwap_adr.features import (  # noqa: E402
    SESSION_START_MINUTE,
    SQUARE_OFF_MINUTE,
)
from nifty_quant.research.vwap_adr.report import daily_pnl, summarize  # noqa: E402
from nifty_quant.universe.static import survivorship_report  # noqa: E402

Abort = pre.Abort
log = pre.log

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
)
KILL_KEYS = (
    "mean_net_bps",
    "day_t",
    "positive_years",
    "n_years",
    "mean_net_bps_ex_top5",
    "c1_mean_net_positive",
    "c2_day_t_gt_2",
    "c3_positive_years_ge_4",
    "c4_ex_top5_positive",
    "passes",
)
SMOKE_BOUNDS = (pre.LOAD_START, dt.date(2017, 12, 31))
#: Smoke-only rolling-quantile history: the pre-registered min_sessions=120 cannot be met
#: inside the ~115-session warm-up, so the smoke run shortens it purely to exercise code.
SMOKE_Q_WINDOW = 60
SMOKE_Q_MIN_SESSIONS = 40


def to_ist(ts: np.ndarray | pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s", utc=True).tz_convert(
        "Asia/Kolkata"
    )


def fmt(x: object, nd: int = 3) -> str:
    if x is None:
        return "NaN"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
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


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prereg", default="results/liqprov/prereg.json")
    p.add_argument("--out", default="results/liqprov")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="code smoke test: 1 trial on a date range inside the 2017 warm-up period "
        "(outside the test window); skips the prereg recomputation match; requires --out "
        "outside results/",
    )
    p.add_argument("--smoke-start", type=dt.date.fromisoformat, default=dt.date(2017, 11, 1))
    p.add_argument("--smoke-end", type=dt.date.fromisoformat, default=dt.date(2017, 12, 29))
    p.add_argument("--smoke-trial", type=int, default=0, help="index into prereg trials")
    return p.parse_args(argv)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else settings.REPO_ROOT / p


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    args = parse_args(argv)
    out = _abs(args.out)
    prereg_path = _abs(args.prereg)

    # 1. Pre-registration: exists and its hash recomputes.
    doc = pre.load_prereg(prereg_path)
    log(f"prereg {prereg_path}: config_hash {doc['config_hash']} recomputes")
    if doc.get("schema") != pre.SCHEMA:
        raise Abort(f"prereg schema {doc.get('schema')!r} != {pre.SCHEMA!r}")
    if pre.build_trials(doc["k_grid"]) != doc["trials"]:
        raise Abort("prereg trials are not the q x k x mode product of its own k grid")
    notionals = [
        float(doc["execution"]["notional"]),
        float(doc["execution"]["notional_sensitivity"]),
    ]
    ref_notional = notionals[0]
    years = [int(y) for y in doc["kill_criteria"]["years"]]
    dsr_bar = float(doc["kill_criteria"]["dsr_bar"])
    pbo_splits = int(doc["kill_criteria"]["pbo_splits"])
    tick = float(doc["execution"]["tick"])

    if args.smoke:
        results_root = (settings.REPO_ROOT / "results").resolve()
        if out.resolve() == results_root or results_root in out.resolve().parents:
            raise Abort(f"--smoke output {out} must be outside {results_root}")
        s0, s1 = args.smoke_start, args.smoke_end
        if not (SMOKE_BOUNDS[0] <= s0 <= s1 <= SMOKE_BOUNDS[1] and s1 < pre.WINDOW_START):
            raise Abort(f"smoke range [{s0}, {s1}] must lie inside {SMOKE_BOUNDS}")
        if not 0 <= args.smoke_trial < len(doc["trials"]):
            raise Abort(f"--smoke-trial {args.smoke_trial} out of range")
        win_start, win_end = s0, s1
        trials = [doc["trials"][args.smoke_trial]]
    else:
        win_start, win_end = pre.WINDOW_START, pre.WINDOW_END
        trials = list(doc["trials"])

    universe = pre.resolve_universe()
    lock = HoldoutLock(path=default_holdout_lock_path())
    reads_before = lock.read_count()
    log(f"holdout lock {lock.path}: read_count before = {reads_before}")
    calendar = TradingCalendar.from_index_bars(pre.INDEX_SYMBOL)
    ho_start, ho_end = pre.resolve_holdout(lock, calendar)
    if pre.WINDOW_END >= ho_start or win_end >= ho_start:
        raise Abort(f"window end {win_end} >= holdout start {ho_start}")
    log(f"holdout [{ho_start}, {ho_end}]; trading window [{win_start}, {win_end}]")

    # 2. Panel and features (the same code path as the prereg).
    ctx = pre.load_context(universe, calendar, pre.LOAD_START, win_start, win_end)
    if len(ctx.stock_syms) != 50:
        raise Abort(f"expected 50 stock columns, got {len(ctx.stock_syms)}")
    if args.smoke:
        pre.compute_features(ctx, q_window=SMOKE_Q_WINDOW, q_min_sessions=SMOKE_Q_MIN_SESSIONS)
        log("SMOKE: prereg recomputation match skipped (different window by construction)")
    else:
        pre.compute_features(ctx)
        fresh = pre.build_body(ctx, universe, (ho_start, ho_end), (reads_before, reads_before))
        stored = {k: v for k, v in doc.items() if k not in pre.HASH_EXCLUDED_KEYS}
        diffs = pre.diff_bodies(stored, fresh)
        if diffs:
            raise Abort("fresh recomputation differs from prereg:\n  " + "\n  ".join(diffs[:50]))
        log("fresh recomputation of k grid, gate thresholds/frequencies matches prereg")

    offs = ctx.offs
    dates = ctx.dates
    ok_days = ctx.ok_days
    n_days = int(ok_days.size)
    day_map = np.full(len(dates), -1, dtype=np.int64)
    day_map[ok_days] = np.arange(n_days, dtype=np.int64)
    ok_dates = [dates[i] for i in ok_days]

    stock_panel = ctx.panel.sub(symbols=ctx.stock_syms)
    if tuple(stock_panel.symbols) != ctx.stock_syms:
        raise Abort("stock sub-panel column order differs from the stock column mapping")
    tradable_raw = tradable_mask(stock_panel)
    del stock_panel
    row_ok = np.repeat(ctx.session_ok, np.diff(offs))
    tradable = tradable_raw & row_ok[:, None]
    present = np.isfinite(ctx.stock["close"])
    log(
        f"tradable: {int(tradable_raw.sum())} raw, {int(tradable.sum())} after session filter; "
        f"tradable/present in-window = "
        f"{tradable.sum() / max(1, (present & row_ok[:, None]).sum()):.4f}"
    )

    # 3. Trials.
    f = ctx.feats
    s = ctx.stock
    sym_col = {sym: i for i, sym in enumerate(ctx.stock_syms)}
    idx_open = ctx.index["open"]
    cost_model = NSEIntradayEquityCosts()
    slippage = SqrtImpactSlippage()
    rows: list[dict] = []
    trades: dict[tuple[str, float], pd.DataFrame] = {}
    daily: dict[tuple[str, float], np.ndarray] = {}
    kills: dict[tuple[str, float], dict] = {}
    by_year: dict[tuple[str, float], dict[int, float]] = {}
    march: dict[tuple[str, float], float] = {}
    decomp: dict[tuple[str, float], dict] = {}
    for i_t, trial in enumerate(trials, 1):
        q = float(trial["q"])
        gate = f["thresholds"][q]["gate"]
        for notional in notionals:
            tc = time.time()
            params = LiqParams(
                k=float(trial["k"]),
                notional=notional,
                mode=str(trial["mode"]),
                tick=tick,
                start_minute=SESSION_START_MINUTE,
                square_off_minute=SQUARE_OFF_MINUTE,
            )
            tr = simulate_liqprov(
                s["open"],
                s["high"],
                s["low"],
                s["close"],
                s["volume"],
                f["vwap"],
                f["idx_dev"],
                f["beta"],
                f["adr"],
                gate,
                tradable,
                ctx.mod,
                offs,
                ctx.stock_syms,
                params,
                cost_model=cost_model,
                slippage=slippage,
            )
            if (tr["side"] != 1).any():
                raise Abort(f"{trial['id']}: non-long trade produced")
            panel_day = tr["day_idx"].to_numpy(dtype=np.int64)
            mapped = day_map[panel_day]
            if (mapped < 0).any():
                raise Abort(
                    f"{trial['id']}: {int((mapped < 0).sum())} trades on sessions outside "
                    "the in-window usable set (impossible: tradable is False there)"
                )
            tr = tr.rename(columns={"day_idx": "panel_day_idx"})
            tr.insert(1, "day_idx", mapped)
            entry_row = tr["entry_row"].to_numpy(dtype=np.int64)
            exit_row = tr["exit_row"].to_numpy(dtype=np.int64)
            cols = np.array([sym_col[x] for x in tr["symbol"]], dtype=np.int64)
            tr["entry_ts"] = ctx.ts[entry_row]
            tr["exit_ts"] = ctx.ts[exit_row]
            tr["session_date"] = [ok_dates[d] for d in tr["day_idx"]]
            beta_d = f["beta"][panel_day, cols] if len(tr) else np.empty(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mkt = beta_d * (idx_open[exit_row] / idx_open[entry_row] - 1.0) * 1e4
            tr["beta_day"] = beta_d
            tr["market_bps"] = mkt
            tr["residual_bps"] = tr["gross_bps"].to_numpy(dtype=np.float64) - mkt

            key = (trial["id"], notional)
            summ = summarize(tr, n_days)
            kc = kill_criteria(tr, ok_dates, years)
            yn = yearly_net(tr, ok_dates)
            mar = (
                float(
                    tr.loc[
                        [d.year == 2020 and d.month == 3 for d in tr["session_date"]], "net_pnl"
                    ].sum()
                )
                if len(tr)
                else 0.0
            )
            fin = np.isfinite(mkt)
            dc = {
                "mean_gross_bps": float(tr["gross_bps"].mean()) if len(tr) else float("nan"),
                "mean_market_bps": float(np.mean(mkt[fin])) if fin.any() else float("nan"),
                "mean_residual_bps": (
                    float(np.mean(tr["residual_bps"].to_numpy()[fin]))
                    if fin.any()
                    else float("nan")
                ),
                "n_decomposed": int(fin.sum()),
                "n_undecomposed": int((~fin).sum()),
            }
            dly = daily_pnl(tr, n_days, "net_pnl")
            worst_d = int(np.argmin(dly)) if n_days else -1
            trades[key], daily[key], kills[key] = tr, dly, kc
            by_year[key], march[key], decomp[key] = yn, mar, dc
            rows.append(
                {
                    "trial": trial["id"],
                    "q": q,
                    "k": float(trial["k"]),
                    "k_name": trial["k_name"],
                    "mode": trial["mode"],
                    "notional": notional,
                    **{k_: summ[k_] for k_ in SUMMARY_KEYS},
                    "worst_day_date": str(ok_dates[worst_d]) if worst_d >= 0 else "",
                    **{f"kill_{k_}": kc[k_] for k_ in KILL_KEYS},
                    "march2020_net_pnl": mar,
                    **{f"net_{y}": yn.get(y, 0.0) for y in years},
                    **{f"decomp_{k_}": v for k_, v in dc.items()},
                }
            )
            log(
                f"trial {i_t}/{len(trials)} {trial['id']} @ {int(notional)}: "
                f"{len(tr)} trades, mean_net_bps {fmt(summ['mean_net_bps'], 2)}, "
                f"day_t {fmt(kc['day_t'], 2)}, passes {kc['passes']} "
                f"({time.time() - tc:.1f}s)"
            )
    grid = pd.DataFrame(rows)

    # 4. Overfitting controls at the reference notional (Rs 10L).
    of = overfitting(trials, daily, ref_notional, pbo_splits)
    log(
        f"overfitting: n_eff={fmt(of['n_eff'], 3)} of {of['n_cells']}, best "
        f"{of.get('best_cell')} sharpe_ann={fmt(of.get('best_sharpe_ann'), 3)}, "
        f"sr0={fmt(of.get('sr0'), 5)}, DSR={fmt(of.get('dsr'), 4)}, PBO={fmt(of.get('pbo'), 4)}"
    )

    reads_after = lock.read_count()
    log(f"holdout lock read_count after = {reads_after}")
    if reads_after != reads_before:
        raise Abort(f"holdout read count changed: {reads_before} -> {reads_after}")

    # 5. Outputs.
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv", index=False)
    for (tid, notional), tr in trades.items():
        pq = tr.copy()
        pq["session_date"] = pq["session_date"].astype(str)
        pq.to_parquet(out / f"trades_{tid}_{int(notional)}.parquet")
    report = build_report(
        smoke=args.smoke,
        doc=doc,
        prereg_path=prereg_path,
        universe=universe,
        surv=survivorship_report(universe, win_start, win_end),
        win=(win_start, win_end),
        holdout=(ho_start, ho_end),
        reads=(reads_before, reads_after),
        ctx=ctx,
        n_days=n_days,
        trials=trials,
        notionals=notionals,
        ref=ref_notional,
        years=years,
        grid=grid,
        trades=trades,
        kills=kills,
        by_year=by_year,
        march=march,
        decomp=decomp,
        of=of,
        dsr_bar=dsr_bar,
        runtime=time.time() - t0,
    )
    (out / "REPORT.md").write_text(report)
    log(f"wrote {out / 'REPORT.md'}, grid.csv, {len(trades)} parquet files")
    log(f"total runtime {time.time() - t0:.1f}s")
    return 0


def overfitting(
    trials: list[dict],
    daily: dict[tuple[str, float], np.ndarray],
    ref: float,
    pbo_splits: int,
) -> dict:
    ids = [t["id"] for t in trials]
    mat = np.column_stack([daily[(tid, ref)] / ref for tid in ids])
    of: dict = {"n_cells": len(ids), "T": int(mat.shape[0])}
    col_std = mat.std(axis=0, ddof=1) if mat.shape[0] >= 2 else np.zeros(mat.shape[1])
    varying = col_std > 0
    of["constant_cells"] = [tid for tid, ok in zip(ids, varying) if not ok]
    mat_v = mat[:, varying]
    ids_v = [tid for tid, ok in zip(ids, varying) if ok]
    nan = float("nan")
    of["n_eff"] = effective_n_trials(mat_v) if mat_v.shape[1] >= 1 else nan
    per_sr = np.array([float(np.mean(c) / np.std(c, ddof=1)) for c in mat_v.T], dtype=np.float64)
    of["var_trial_sharpes"] = float(np.var(per_sr, ddof=1)) if per_sr.size >= 2 else nan
    ann = np.array([sharpe_ratio(c, periods_per_year=252) for c in mat_v.T], dtype=np.float64)
    if ann.size and np.isfinite(ann).any():
        b = int(np.nanargmax(ann))
        of["best_cell"] = ids_v[b]
        of["best_sharpe_ann"] = float(ann[b])
        of["best_sharpe_period"] = float(per_sr[b])
        n_eff = of["n_eff"]
        var_sr = of["var_trial_sharpes"]
        if not (math.isfinite(n_eff) and n_eff >= 2.0 and math.isfinite(var_sr)):
            of["sr0"] = 0.0
            of["sr0_note"] = (
                f"effective_n_trials={fmt(n_eff, 4)} < 2 (or NaN), or fewer than 2 varying "
                "trials: no multiple-testing penalty applied, sr0=0.0 "
                "(expected_max_sharpe requires n >= 2)"
            )
        else:
            of["sr0"] = expected_max_sharpe(n_eff, var_sr)
            of["sr0_note"] = (
                "sr0 = expected_max_sharpe(effective_n_trials, var of per-period trial "
                "Sharpes), both per-period (non-annualised)"
            )
        of["dsr"] = deflated_sharpe(mat_v[:, b], sr0=of["sr0"])
    else:
        of["best_cell"] = None
        of["dsr"] = nan
        of["sr0"] = nan
        of["sr0_note"] = "no trial has a finite Sharpe (no varying daily P&L)"
    try:
        of["pbo"] = pbo_cscv(mat_v, n_splits=pbo_splits)
    except ValueError as exc:
        of["pbo"] = nan
        of["pbo_error"] = str(exc)
    return of


# --------------------------------------------------------------------------- report


def build_report(**kw) -> str:  # noqa: C901 - linear report assembly
    doc = kw["doc"]
    ctx = kw["ctx"]
    win_start, win_end = kw["win"]
    ho_start, ho_end = kw["holdout"]
    trials = kw["trials"]
    notionals = kw["notionals"]
    ref = kw["ref"]
    years = kw["years"]
    of = kw["of"]
    kills = kw["kills"]
    grid: pd.DataFrame = kw["grid"]
    surv = kw["surv"]
    kg = doc["k_grid"]
    gates = doc["gates"]
    L: list[str] = []
    L.append("# Stress-gated residual liquidity-provision fade: backtest report")
    L.append("")
    if kw["smoke"]:
        L.append(
            f"SMOKE TEST ONLY. Code check on [{win_start}, {win_end}] (2017 warm-up period, "
            "outside the test window), 1 trial, rolling-quantile window/min_sessions shortened "
            f"to {SMOKE_Q_WINDOW}/{SMOKE_Q_MIN_SESSIONS}. No number below is evidence."
        )
        L.append("")
    L.append(
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by "
        f"scripts/run_liqprov_backtest.py (runtime {kw['runtime']:.0f}s). Spec: "
        f"specs/liqprov_fade.md. Pre-registration: {kw['prereg_path'].name}, config_hash "
        f"`{doc['config_hash']}` (git sha at prereg: "
        f"{doc.get('provenance', {}).get('git_sha')})."
    )
    L.append("")

    # Caveats first.
    missing_2018 = surv.missing_by_year.get(win_start.year, ())
    late = {}
    for y in sorted(surv.missing_by_year):
        for sym in surv.missing_by_year[y]:
            late[sym] = y
    late_str = ", ".join(f"{s} (no data through {y})" for s, y in late.items()) or "none"
    L.append("## Caveats (read first)")
    L.append("")
    L.append(
        f"- Survivorship: universe `{kw['universe'].name}` is the CURRENT Nifty 50 constituent "
        "list applied to the whole window, not point-in-time membership. Names that were "
        "added to the index during or after the window are traded before they joined it, "
        "and names that left the index are absent. Listed mid-window (no data in some "
        f"window years): {late_str}. "
        f"Missing in {win_start.year}: {list(missing_2018) or 'none'}. {surv.warning_line()}"
    )
    L.append(
        "- Test-window rationale: the hypothesis was generated from exploratory diagnostics on "
        "2023-08 .. 2025-08 (the base VWAP+/-k*ADR fade), so that window is not evidence. "
        f"The test window [{pre.WINDOW_START}, {pre.WINDOW_END}] was never used for this "
        "family. Bars from 2017-07-17 are loaded only as warm-up for ADR, beta and the "
        "rolling gate quantiles; no trade occurs outside usable in-window sessions."
    )
    L.append(
        f"- Holdout untouched: holdout = [{ho_start}, {ho_end}] from the lock file; the panel "
        f"ends at {ctx.dates[-1]}. Holdout read count before/after: "
        f"{kw['reads'][0]}/{kw['reads'][1]} (record_read never called)."
    )
    L.append(
        "- Slippage: SqrtImpactSlippage() defaults (half_spread 1.5 bps + 10*sqrt(notional / "
        "bar traded value)) are ASSUMED, not calibrated (CLAUDE.md rule 8 not satisfied for "
        "them). Passive entries pay no slippage and are assumed filled on a one-tick "
        "trade-through, with no queue position modelled. Costs: NSEIntradayEquityCosts()."
    )
    L.append(
        "- Tradable mask: tradable_mask on stock columns only, default ADV floor "
        "min_adv_inr=5e7 (Rs 5 crore, 20-session trailing mean). That floor is a pre-existing "
        "repo CONSTANT, not derived from a measured null (rule 8 not satisfied for it)."
    )
    L.append(
        f"- Tick: {doc['execution']['tick']} rupees (NSE equity tick for 2018-2023) is used for "
        "the passive trade-through test."
    )
    L.append(
        "- No stop loss: positions exit only on VWAP reversion (next fillable bar's open), "
        "15:20 square-off, or the last bar's close. Long only. Fractional quantities; P&L "
        "is summed across symbols with no capital constraint (see max_concurrent)."
    )
    first_fin = {qk: g["first_session_with_finite_thresholds"] for qk, g in gates["per_q"].items()}
    L.append(
        "- Gate warm-up: the rolling gate thresholds need "
        f"{doc['features']['quantile_min_sessions']} prior usable sessions, so the gate "
        f"cannot fire before {sorted(set(first_fin.values()))} (first in-window session with "
        "finite thresholds); those early sessions are in the denominators as zero-P&L days."
    )
    L.append(
        "- Gate sparsity: the gate fires on "
        + ", ".join(f"{g['sessions_fired']} (q={qk})" for qk, g in gates["per_q"].items())
        + f" of {gates['n_sessions']} usable in-window sessions. Kill criterion c3 (>= 4 "
        "positive years out of 6) and the day-clustered t are therefore evaluated on very "
        "few trade-days for the stricter gates."
    )
    L.append(
        f"- Unusable sessions (TradingCalendar.is_usable) are untradable: "
        f"{[str(d) for d in ctx.excluded_in_window] or 'none'}. The 09:15 bar is excluded "
        "from VWAP, TWAP, range, beta and signals."
    )
    L.append(
        f"- Multiple testing: {doc['n_trials']} pre-registered trials (3 q x 2 k x 2 modes) "
        "at 2 notionals; the best trial is selected in-sample. See the overfitting section."
    )
    L.append(
        "- Beta/residual decomposition: market part = beta_d * (NIFTY50 open at exit row / "
        "open at entry row - 1) in bps; residual = gross_bps - market part. Trades whose "
        "index bar is absent at either row are left undecomposed (counted)."
    )
    L.append("")

    # Pre-registration.
    L.append("## Pre-registered parameters (measured, no P&L)")
    L.append("")
    L.append(
        f"k grid: {kg['definition']}. Sample {kg['sample_size']} of {kg['total_pairs']} pairs."
    )
    L.append("")
    L.append(
        md_table(
            ["name", "quantile of minz", "value", "k (3 dp)"],
            [
                [n, f"{v['q']:.2f}", f"{v['minz_quantile']:.6f}", f"{kg['k'][n]:.3f}"]
                for n, v in kg["quantiles"].items()
            ],
        )
    )
    L.append("")
    L.append(f"Gate: {gates['definition']}")
    L.append("")
    L.append(
        md_table(
            ["q", "sessions fired", "fraction", "median v_day", "median m_day", "first finite"],
            [
                [
                    qk,
                    g["sessions_fired"],
                    f"{g['frac_sessions_fired']:.4f}",
                    f"{g['v_day_describe'].get('median', float('nan')):.4f}",
                    f"{g['m_day_describe'].get('median', float('nan')):.4f}",
                    str(g["first_session_with_finite_thresholds"]),
                ]
                for qk, g in gates["per_q"].items()
            ],
        )
    )
    L.append("")
    L.append("Kill criteria (pre-registered text):")
    L.append("")
    for t in doc["kill_criteria"]["text"]:
        L.append(f"- {t}")
    L.append("")

    # Kill criteria per trial.
    n_pass = {nt: sum(bool(kills[(t["id"], nt)]["passes"]) for t in trials) for nt in notionals}
    for nt in notionals:
        tag = "primary" if nt == ref else "sensitivity"
        L.append(f"## Kill criteria per trial at Rs {int(nt):,} ({tag})")
        L.append("")
        L.append(
            md_table(
                ["trial", *KILL_KEYS],
                [[t["id"], *[kills[(t["id"], nt)][k] for k in KILL_KEYS]] for t in trials],
            )
        )
        L.append("")
        L.append(f"Trials passing c1-c4: {n_pass[nt]} of {len(trials)}.")
        L.append("")

    # Summary tables.
    for nt in notionals:
        L.append(f"## Summary at Rs {int(nt):,}")
        L.append("")
        sub = grid[grid["notional"] == nt]
        L.append(
            md_table(
                ["trial", *SUMMARY_KEYS, "worst_day_date"],
                [
                    [r["trial"], *[r[k] for k in SUMMARY_KEYS], r["worst_day_date"]]
                    for _, r in sub.iterrows()
                ],
            )
        )
        L.append("")

    # March 2020 and per-year.
    L.append("## March-2020 net P&L per trial (Rs)")
    L.append("")
    L.append(
        md_table(
            ["trial", *[f"Rs {int(nt):,}" for nt in notionals]],
            [[t["id"], *[kw["march"][(t["id"], nt)] for nt in notionals]] for t in trials],
        )
    )
    L.append("")
    for nt in notionals:
        L.append(f"## Per-year net P&L at Rs {int(nt):,} (Rs)")
        L.append("")
        L.append(
            md_table(
                ["trial", *[str(y) for y in years]],
                [
                    [t["id"], *[kw["by_year"][(t["id"], nt)].get(y, 0.0) for y in years]]
                    for t in trials
                ],
            )
        )
        L.append("")

    # Decomposition.
    L.append(f"## Beta / residual decomposition at Rs {int(ref):,} (mean bps per trade)")
    L.append("")
    L.append(
        md_table(
            [
                "trial",
                "mean_gross_bps",
                "mean_market_bps",
                "mean_residual_bps",
                "n_decomposed",
                "n_undecomposed",
            ],
            [
                [
                    t["id"],
                    *[
                        kw["decomp"][(t["id"], ref)][k]
                        for k in (
                            "mean_gross_bps",
                            "mean_market_bps",
                            "mean_residual_bps",
                            "n_decomposed",
                            "n_undecomposed",
                        )
                    ],
                ]
                for t in trials
            ],
        )
    )
    L.append("")

    # Overfitting.
    L.append(f"## Overfitting statistics (Rs {int(ref):,})")
    L.append("")
    L.append(
        f"Trial matrix: daily net P&L / notional, {of['T']} usable in-window sessions x "
        f"{of['n_cells']} trials (zero days included)."
        + (
            f" Zero-variance trials excluded: {of['constant_cells']}."
            if of["constant_cells"]
            else ""
        )
    )
    L.append("")
    dsr = of.get("dsr", float("nan"))
    L.append(
        md_table(
            ["statistic", "value"],
            [
                ["effective_n_trials", of.get("n_eff")],
                [
                    "var of per-period trial Sharpes",
                    f"{of.get('var_trial_sharpes', float('nan')):.6g}",
                ],
                ["best trial (annualised sharpe_net)", str(of.get("best_cell"))],
                ["best annualised Sharpe", of.get("best_sharpe_ann")],
                ["best per-period Sharpe", of.get("best_sharpe_period")],
                ["sr0 (per-period)", f"{of.get('sr0', float('nan')):.6g}"],
                ["deflated Sharpe (probability)", dsr],
                [
                    f"DSR bar ({kw['dsr_bar']}) met",
                    bool(math.isfinite(dsr) and dsr >= kw["dsr_bar"]),
                ],
                [f"PBO (CSCV, {doc['kill_criteria']['pbo_splits']} splits)", of.get("pbo")],
            ],
        )
    )
    L.append("")
    L.append(f"sr0 note: {of.get('sr0_note', 'n/a')}.")
    if "pbo_error" in of:
        L.append(f"PBO error: {of['pbo_error']}.")
    L.append("")
    supported = n_pass[ref] > 0 and math.isfinite(dsr) and dsr >= kw["dsr_bar"]
    L.append(
        f"Verdict against the pre-registration: {n_pass[ref]} of {len(trials)} trials pass "
        f"c1-c4 at Rs {int(ref):,}; DSR of the best trial = {fmt(dsr, 4)} vs bar "
        f"{kw['dsr_bar']}. Hypothesis {'SUPPORTED' if supported else 'NOT supported (killed)'}."
    )
    L.append("")

    # 10 worst trades.
    L.append(f"## 10 worst trades at Rs {int(ref):,} (distinct trades across trials)")
    L.append("")
    allt = [
        kw["trades"][(t["id"], ref)].assign(trial=t["id"])
        for t in trials
        if len(kw["trades"][(t["id"], ref)])
    ]
    if allt:
        at = pd.concat(allt, ignore_index=True)
        key_cols = ["symbol", "entry_row", "exit_row"]
        at["n_trials"] = at.groupby(key_cols)["net_bps"].transform("size")
        at = at.sort_values(["net_bps", "trial"], kind="mergesort")
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
                    f"{r['entry_price']:.2f}",
                    f"{r['exit_price']:.2f}",
                    r["exit_reason"],
                    f"{r['gross_bps']:.1f}",
                    f"{r['net_bps']:.1f}",
                    f"{r['market_bps']:.1f}" if math.isfinite(r["market_bps"]) else "NaN",
                    r["trial"],
                    int(r["n_trials"]),
                ]
            )
        L.append(
            md_table(
                [
                    "symbol",
                    "entry (IST)",
                    "exit (IST)",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "gross_bps",
                    "net_bps",
                    "market_bps",
                    "first trial",
                    "trials containing it",
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
