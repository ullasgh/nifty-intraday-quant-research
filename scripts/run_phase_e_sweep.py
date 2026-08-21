#!/usr/bin/env python3
"""Phase E conditional-analysis sweep -- PARALLEL SHARDED runner.

A single serial process running the full 22-feature x 6-horizon = 132-trial sweep was
SIGKILLed by the OS (exit 137): the full-panel `close` array alone is ~0.78 GB in float64,
and materialising it once per feature inside one long-lived process compounds with whatever
else that process retains over 132 trials. This script instead partitions the feature
registry into disjoint shards, runs each shard in its OWN subprocess with its OWN memmap'd
panel (so peak RSS per process stays near the ~0.78 GB panel-load floor, not a multiple of
it), and merges the shards' independent output files afterwards.

Two modes
---------
Worker/shard mode (what each subprocess runs):
    .venv/bin/python scripts/run_phase_e_sweep.py --shard 0 --n-shards 4
    Runs only ``sweep_features.FEATURE_REGISTRY[0::4]`` x every horizon, and writes ITS OWN
    ``shard_0_of_4.pkl`` under ``--output-dir``. Shards never share an output path -- a
    single path with two concurrent writers is how results get silently lost.

Parent/parallel mode (what a human runs):
    .venv/bin/python scripts/run_phase_e_sweep.py --parallel --n-shards 4 --workers 4
    Launches N shard subprocesses (never more than ``--workers`` running at once -- RAM,
    not CPU, is the constraint here: 4 workers x ~0.78 GB/panel-load is already most of the
    measured ~6.2 GB free), waits for all of them, then merges every shard file into one
    report.

Merge / report (parent mode only), in this order (do not reorder -- a ranked table without
its denominator above it is the exact failure mode this script exists to avoid):
    1. MEASURED ``effective_n_trials`` (backtest.metrics.effective_n_trials) on the merged
       (T, n_trials) trial matrix, next to the contract's own PLANNED trial count.
    2. PBO (backtest.metrics.pbo_cscv) on that same matrix.
    3. ``var_trial_sharpes`` MEASURED from the actual per-trial Sharpes this run produced
       (feature_sweep.measure_var_trial_sharpes) -- never the ``lens.py`` unmeasured
       ``var_trial_sharpes=1.0`` placeholder.
    4. The per-trial table, sorted by deflated Sharpe (using the corrected
       ``expected_max_sharpe(n_eff, var_trial_sharpes=<measured>)`` as DSR's ``sr0``, the
       same correction ``lens.py`` itself documents as missing at its own placeholder site).
    5. Any trial that RAISED, listed with its exception -- a feature failing on real data is
       a RESULT (E5/obligation 11), never silently dropped from the report.

Non-negotiable safety
----------------------
The contract this script builds declares ``holdout_intent="never"`` and a data window ending
strictly before the live holdout boundary (2025-08-14). ``feature_sweep.run_sweep`` asserts
both itself and RAISES otherwise; this script does not add, weaken, catch, or duplicate that
check -- it is enforced exactly once, inside `run_sweep`, and left alone.

IMPORTANT: this script's plumbing has been verified ONLY on a small slice (a handful of
sessions, >= 5 symbols, 2 features, 1 horizon) -- see the worker's final report for the
command used. **The full 132-trial sweep has NOT been executed by this script.**
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import recon_low_turnover_tilt as base  # noqa: E402  (exposes START, END, load_universe, PanelSpec, load_panel)

from nifty_quant.backtest import metrics  # noqa: E402
from nifty_quant.research import expectancy  # noqa: E402
from nifty_quant.research import feature_sweep as fs  # noqa: E402
from nifty_quant.research import sweep_features as sf  # noqa: E402
from nifty_quant.research.contract import ResearchContract  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Options a parent process must forward verbatim to every shard subprocess it launches, so
# every shard analyses the exact same universe/window/horizons/buckets/seed -- only the
# feature partition (--shard/--n-shards) differs between them.
_PASSTHROUGH_OPTS = (
    "output_dir",
    "universe",
    "start",
    "end",
    "horizons",
    "n_buckets",
    "seed",
    "max_symbols",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shard", type=int, default=0, help="this process's shard index (worker mode)")
    p.add_argument(
        "--n-shards", type=int, default=4, help="total number of shards; features run i::N"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="max CONCURRENT shard subprocesses in --parallel mode. Measured: RAM 16 GB, "
        "~6.2 GB free, ~0.78 GB per memmap'd panel load -- do not raise above 4 without "
        "re-measuring free RAM.",
    )
    p.add_argument(
        "--parallel", action="store_true", help="parent mode: launch shards, wait, merge, report"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "phase_e_shards",
        help="each shard writes its OWN file here (shard_<i>_of_<n>.pkl); never a shared path",
    )
    p.add_argument("--universe", type=str, default="all_equity")
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="override the data window start (YYYY-MM-DD). TESTING ONLY -- default is "
        "recon_low_turnover_tilt.START (2018-01-01), the pre-registered Phase E window.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="override the data window end (YYYY-MM-DD). TESTING ONLY -- default is "
        "recon_low_turnover_tilt.END (2025-07-31). run_sweep RAISES if this reaches the "
        "live holdout boundary; that check is never bypassed here.",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="comma list, e.g. '1,5,EOD'. Default: the full sweep_features.HORIZONS.",
    )
    p.add_argument(
        "--n-buckets",
        type=int,
        default=5,
        help="bucket count for cross_sectional_rank (matches run_sweep's own default)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="TESTING ONLY: slice the universe to the first N symbols for a quick smoke test.",
    )
    return p


def _parse_horizons(spec: str | None) -> list[int | str]:
    if spec is None:
        return list(sf.HORIZONS)
    out: list[int | str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.append("EOD" if tok.upper() == "EOD" else int(tok))
    return out


# ---------------------------------------------------------------------------
# Worker (shard) mode
# ---------------------------------------------------------------------------


def _shard_output_path(output_dir: Path, shard: int, n_shards: int) -> Path:
    return output_dir / f"shard_{shard}_of_{n_shards}.pkl"


def shard_features_for(shard: int, n_shards: int) -> list[sf.FeatureSpec]:
    """This shard's slice of `sweep_features.FEATURE_REGISTRY`: features[shard::n_shards].

    Every shard uses this SAME function with its own index, so the N slices are pairwise
    disjoint and their union is the whole registry by construction (Python slice semantics),
    never a hand-partitioned list that could drift out of sync with the registry.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if not (0 <= shard < n_shards):
        raise ValueError(
            f"shard must satisfy 0 <= shard < n_shards; got shard={shard}, n_shards={n_shards}"
        )
    return list(sf.FEATURE_REGISTRY)[shard::n_shards]


def _raw_spread_series(
    feature_values: np.ndarray,
    fwd_values: np.ndarray,
    day_offsets: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """Per-bar long-top/short-bottom-bucket spread return series, WITHOUT the final
    isfinite filter that ``feature_sweep._bucket_spread_returns`` applies.

    Mirrors that function's exact bucketing call (``cross_sectional_rank``, ``min_history=1``
    -- the same config `feature_sweep.py` already uses for this purpose, not a new choice)
    but keeps every row (NaN where undefined) so multiple trials' series stay ROW-ALIGNED and
    can be stacked into a single (T, n_trials) matrix by the merge step. `run_sweep`'s own
    `TrialRecord` carries no return series at all (`sharpe_gross`/`sharpe_net` are always
    `None` for this sweep -- see `feature_sweep.run_sweep`), so this is recomputed here from
    the same public/semi-public primitives, not read off a record.
    """
    bucketing = expectancy.causal_buckets(
        feature_values,
        day_offsets,
        n_buckets=n_buckets,
        method="cross_sectional_rank",
        min_history=1,
    )
    labels = bucketing.labels
    top_mask = labels == (n_buckets - 1)
    bottom_mask = labels == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        top_ret = np.nanmean(np.where(top_mask, fwd_values, np.nan), axis=1)
        bottom_ret = np.nanmean(np.where(bottom_mask, fwd_values, np.nan), axis=1)
    return (top_ret - bottom_ret).astype(np.float64)


def run_shard(args: argparse.Namespace) -> Path:
    """Run this shard's features x every horizon, write ONE pickle, return its path."""
    horizons = _parse_horizons(args.horizons)

    symbols_all = base.load_universe(args.universe).symbols
    symbols = tuple(symbols_all[: args.max_symbols]) if args.max_symbols else tuple(symbols_all)

    start = dt.date.fromisoformat(args.start) if args.start else base.START
    end = dt.date.fromisoformat(args.end) if args.end else base.END

    spec = base.PanelSpec(
        freq="1",
        # `run_sweep` consumes ONLY `close` (its signature takes close + day_offsets); the
        # registry synthesises OHLC proxies internally. Loading all five fields cost 5x the
        # memory and, with 4 concurrent shards, caused memmap contention -- `ValueError: mmap
        # length is greater than file size` and a SIGBUS. Load only what is used.
        fields=("close",),
        symbols=symbols,
        start=start,
        end=end,
    )
    print(
        f"[shard {args.shard}/{args.n_shards}] loading panel {start}..{end} "
        f"({len(symbols)} symbols), memmap=True",
        flush=True,
    )
    # memmap=False: with only `close` loaded this is ~0.39 GB (float32) per shard, so 4
    # concurrent shards fit comfortably in the ~5.8 GB free -- and a private in-process array
    # cannot race other shards on a shared cache file the way the memmap path did.
    panel = base.load_panel(spec, memmap=False)
    close = np.asarray(panel.field("close"), dtype=np.float64)
    day_offsets = np.asarray(panel.day_offsets)
    print(
        f"[shard {args.shard}/{args.n_shards}] panel: {panel.n_rows()} rows, "
        f"{panel.n_days()} days, {panel.n_symbols()} symbols",
        flush=True,
    )

    shard_features = shard_features_for(args.shard, args.n_shards)
    n_planned_shard = len(shard_features) * len(horizons)
    print(
        f"[shard {args.shard}/{args.n_shards}] {len(shard_features)} features x {len(horizons)} "
        f"horizons = {n_planned_shard} trials: {[f.name for f in shard_features]}",
        flush=True,
    )

    contract = ResearchContract(
        data={
            "panel_id": args.universe,
            "panel_hash": f"phase_e_shard{args.shard}_of_{args.n_shards}",
            "universe_name": args.universe,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        features={"ids": tuple(f.name for f in shard_features), "feature_version": "phase_e_v1"},
        label={"horizons": tuple(str(h) for h in horizons), "overlapping": True},
        execution={"cost_model_id": "nse_intraday_default", "slippage_model_id": "sqrt_impact"},
        portfolio={"sizing": "bucket_spread"},
        validation={
            "scheme": "conditional_sweep",
            "holdout_intent": "never",
            "n_planned_trials": n_planned_shard,
        },
        seed=args.seed,
    )
    print(
        f"[shard {args.shard}/{args.n_shards}] contract_hash = {contract.contract_hash}", flush=True
    )

    records = fs.run_sweep(
        contract=contract,
        close=close,
        day_offsets=day_offsets,
        horizons=horizons,
        feature_registry=shard_features,
        n_buckets=args.n_buckets,
        seed=args.seed,
    )
    n_failed = sum(1 for r in records if r.error is not None)
    print(
        f"[shard {args.shard}/{args.n_shards}] run_sweep returned {len(records)} records "
        f"({n_failed} raised)",
        flush=True,
    )

    # Recompute the row-aligned raw spread series for every trial that DIDN'T raise (obligation
    # 11 already recorded the raises above; nothing further to compute for those).
    spread_series: dict[tuple[str, str], np.ndarray] = {}
    for feat in shard_features:
        try:
            feature_values = feat.fn(close, day_offsets)
        except Exception:
            continue
        for horizon in horizons:
            try:
                fwd = fs._forward_returns_for_horizon(close, day_offsets, horizon)
                spread_series[(feat.name, str(horizon))] = _raw_spread_series(
                    feature_values, fwd.values, day_offsets, args.n_buckets
                )
            except Exception:
                continue

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _shard_output_path(args.output_dir, args.shard, args.n_shards)
    with open(out_path, "wb") as fh:
        pickle.dump(
            {
                "shard": args.shard,
                "n_shards": args.n_shards,
                "records": records,
                "spread_series": spread_series,
                "n_rows": close.shape[0],
                "horizons": horizons,
            },
            fh,
        )
    print(f"[shard {args.shard}/{args.n_shards}] wrote {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Parent mode: launch + merge + report
# ---------------------------------------------------------------------------


def launch_shards(args: argparse.Namespace) -> None:
    """Launch N shard subprocesses, never more than `--workers` concurrently."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_shards):
        stale = _shard_output_path(args.output_dir, i, args.n_shards)
        if stale.exists():
            stale.unlink()  # never merge a leftover file from a previous, differently-shaped run

    passthrough: list[str] = []
    for opt in _PASSTHROUGH_OPTS:
        val = getattr(args, opt)
        if val is None:
            continue
        passthrough += [f"--{opt.replace('_', '-')}", str(val)]

    base_cmd = [sys.executable, str(Path(__file__).resolve())]
    pending = list(range(args.n_shards))
    running: dict[int, subprocess.Popen] = {}
    exit_codes: dict[int, int] = {}

    while pending or running:
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            cmd = base_cmd + ["--shard", str(i), "--n-shards", str(args.n_shards)] + passthrough
            print(
                f"[parallel] launching shard {i}/{args.n_shards} (pid pending): {' '.join(cmd)}",
                flush=True,
            )
            running[i] = subprocess.Popen(cmd)
        time.sleep(0.5)
        for i in list(running):
            ret = running[i].poll()
            if ret is not None:
                exit_codes[i] = ret
                del running[i]
                print(f"[parallel] shard {i} exited with code {ret}", flush=True)

    failed = {i: c for i, c in exit_codes.items() if c != 0}
    if failed:
        raise RuntimeError(f"shard subprocess(es) failed, aborting merge: {failed}")


def _load_shard(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# The mechanical floor `effective_n_trials`/`pbo_cscv`/`expected_max_sharpe` themselves
# require to be DEFINED at all (corrcoef needs >= 2 rows; pbo_cscv needs n_splits <= T and
# n_splits even, floored at 2) -- not a rule-8 research threshold, just the smallest T those
# callees can accept without raising or silently returning nan.
_MIN_T_FOR_STATS = 2


def _classify_trial_series(series: np.ndarray) -> str | None:
    """Return None if `series` is USABLE for the trial matrix, else the reason it must be
    EXCLUDED -- never filled, interpolated, or forward-filled to keep it alive (rule 6).

    A single all-NaN column (e.g. `rolling_hurst` at window=390 against ~375-bar sessions,
    measured 0/92520 finite on RELIANCE 2024) would otherwise make the "rows where ALL
    trials are finite" intersection empty for the WHOLE matrix, regardless of how many other
    trials are good. Excluding the degenerate column, not the analysis, is a RESULT worth
    reporting (a feature that cannot produce a usable series on real data), same as a raise.
    """
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return "all-NaN (0 finite observations)"
    if finite.size == 1:
        return "insufficient finite observations (1 < 2)"
    if np.std(finite, ddof=1) == 0.0:
        return "zero-variance (all finite observations identical)"
    return None


def merge_shards(output_dir: Path, n_shards: int) -> dict:
    shard_data = [_load_shard(_shard_output_path(output_dir, i, n_shards)) for i in range(n_shards)]

    all_records = []
    spread_series: dict[tuple[str, str], np.ndarray] = {}
    for sd in shard_data:
        all_records.extend(sd["records"])
        spread_series.update(sd["spread_series"])  # keys are disjoint: features partition i::N

    n_planned_run = len(all_records)  # sum of every shard's own declared n_planned_trials

    trial_keys = sorted(spread_series.keys())

    # Step 1 (required hardening): exclude all-NaN / zero-variance / degenerate trials
    # BEFORE intersecting rows, so one unusable feature can never collapse T to 0 for
    # every other trial.
    excluded_trials: dict[tuple[str, str], str] = {}
    usable_keys: list[tuple[str, str]] = []
    for k in trial_keys:
        reason = _classify_trial_series(spread_series[k])
        if reason is not None:
            excluded_trials[k] = reason
        else:
            usable_keys.append(k)

    n_rows_before_intersection: int | None = None
    if usable_keys:
        stacked = np.column_stack([spread_series[k] for k in usable_keys])  # (n_rows, n_usable)
        n_rows_before_intersection = stacked.shape[0]
        finite_row_mask = np.all(np.isfinite(stacked), axis=1)
        matrix = stacked[finite_row_mask]
    elif trial_keys:
        # Every trial was excluded, but at least one existed -- still report the starting
        # row count rather than leaving it silently unknown.
        n_rows_before_intersection = next(iter(spread_series.values())).shape[0]
        matrix = np.empty((0, 0))
    else:
        matrix = np.empty((0, 0))

    n_rows_after_intersection = matrix.shape[0]
    rows_dropped = (
        n_rows_before_intersection - n_rows_after_intersection
        if n_rows_before_intersection is not None
        else None
    )
    rows_dropped_pct = (
        (rows_dropped / n_rows_before_intersection * 100.0)
        if n_rows_before_intersection
        else None
    )

    t, n_usable_trials = matrix.shape

    # Step 2: effective_n_trials, with an explicit reason string whenever it can't be a real
    # measurement, rather than a bare, unexplained nan.
    n_eff_note: str | None = None
    if n_usable_trials == 0:
        n_eff = float("nan")
        n_eff_note = (
            "NOT COMPUTED: 0 usable trial columns after exclusions (see excluded list below)"
        )
    elif n_usable_trials == 1:
        n_eff = metrics.effective_n_trials(matrix)  # trivially 1.0 by that function's own contract
        n_eff_note = "trivial: only 1 usable trial column survived exclusions"
    elif t < _MIN_T_FOR_STATS:
        n_eff = metrics.effective_n_trials(matrix)  # nan, per effective_n_trials' own contract
        n_eff_note = (
            f"NOT COMPUTED (nan): only T={t} row(s) survive the finite-row intersection "
            f"across {n_usable_trials} usable columns (need >= {_MIN_T_FOR_STATS})"
        )
    else:
        n_eff = metrics.effective_n_trials(matrix)

    # Step 3: PBO, same explicit-reason discipline.
    pbo_note: str | None = None
    if n_usable_trials < 2:
        pbo = float("nan")
        pbo_note = (
            f"NOT COMPUTED: pbo_cscv requires >= 2 trial columns, have {n_usable_trials} usable"
        )
    elif t < _MIN_T_FOR_STATS:
        pbo = float("nan")
        pbo_note = f"NOT COMPUTED: pbo_cscv requires >= {_MIN_T_FOR_STATS} aligned rows, have T={t}"
    else:
        n_splits = min(16, t)
        n_splits -= n_splits % 2
        pbo = metrics.pbo_cscv(matrix, n_splits=n_splits)

    # MEASURED per-trial Sharpe from each trial's OWN finite observations (not the
    # row-intersected matrix -- a trial with a long warm-up NaN run still contributes its
    # real observations here), feeding feature_sweep.measure_var_trial_sharpes (never the
    # lens.py var_trial_sharpes=1.0 placeholder). Excluded (all-NaN/degenerate) trials
    # correctly fall out as nan below and are reported separately, not silently mixed in.
    per_trial_sharpe: dict[tuple[str, str], float] = {}
    for k in trial_keys:
        finite = spread_series[k][np.isfinite(spread_series[k])]
        if finite.size >= 2 and np.std(finite, ddof=1) > 0:
            per_trial_sharpe[k] = float(np.mean(finite) / np.std(finite, ddof=1))
        else:
            per_trial_sharpe[k] = float("nan")

    finite_sharpes = np.array([v for v in per_trial_sharpe.values() if np.isfinite(v)])
    var_trial_sharpes = (
        fs.measure_var_trial_sharpes(finite_sharpes) if finite_sharpes.size >= 2 else float("nan")
    )

    exp_max_sharpe_note: str | None = None
    if not np.isfinite(n_eff):
        exp_max_sharpe = float("nan")
        exp_max_sharpe_note = "NOT COMPUTED: effective_n_trials is not finite (see above)"
    elif n_eff < 2:
        exp_max_sharpe = float("nan")
        exp_max_sharpe_note = (
            f"NOT COMPUTED: expected_max_sharpe requires n_trials >= 2, "
            f"effective_n_trials={n_eff:.4f}"
        )
    elif not np.isfinite(var_trial_sharpes):
        exp_max_sharpe = float("nan")
        exp_max_sharpe_note = (
            "NOT COMPUTED: var_trial_sharpes is not finite "
            "(fewer than 2 trials had a measurable Sharpe)"
        )
    else:
        exp_max_sharpe = metrics.expected_max_sharpe(n_eff, var_trial_sharpes=var_trial_sharpes)

    per_trial_dsr: dict[tuple[str, str], float] = {}
    for k in trial_keys:
        finite = spread_series[k][np.isfinite(spread_series[k])]
        if finite.size >= 4 and np.isfinite(exp_max_sharpe):
            per_trial_dsr[k] = metrics.deflated_sharpe(finite, sr0=exp_max_sharpe)
        else:
            per_trial_dsr[k] = float("nan")

    failed_records = [r for r in all_records if r.error is not None]

    return {
        "records": all_records,
        "n_planned_run": n_planned_run,
        "matrix_shape": matrix.shape,
        "n_rows_before_intersection": n_rows_before_intersection,
        "n_rows_after_intersection": n_rows_after_intersection,
        "rows_dropped": rows_dropped,
        "rows_dropped_pct": rows_dropped_pct,
        "excluded_trials": excluded_trials,
        "n_eff": n_eff,
        "n_eff_note": n_eff_note,
        "pbo": pbo,
        "pbo_note": pbo_note,
        "var_trial_sharpes": var_trial_sharpes,
        "exp_max_sharpe": exp_max_sharpe,
        "exp_max_sharpe_note": exp_max_sharpe_note,
        "per_trial_sharpe": per_trial_sharpe,
        "per_trial_dsr": per_trial_dsr,
        "failed_records": failed_records,
    }


def _fmt(value: float, spec: str = ".4f") -> str:
    return format(value, spec) if np.isfinite(value) else "nan"


def print_report(merged: dict) -> None:
    n_eff = merged["n_eff"]
    pbo = merged["pbo"]
    var_ts = merged["var_trial_sharpes"]
    t, n_trials = merged["matrix_shape"]
    n_before = merged["n_rows_before_intersection"]
    n_after = merged["n_rows_after_intersection"]
    rows_dropped = merged["rows_dropped"]
    rows_dropped_pct = merged["rows_dropped_pct"]

    print("=" * 88)
    # Matrix shape FIRST -- if T collapsed, it is visible here, not merely inferable from a nan.
    print(f"TRIAL MATRIX SHAPE (T x n_trials) = {t} x {n_trials}")
    if n_before is not None:
        pct_str = f"{rows_dropped_pct:.2f}%" if rows_dropped_pct is not None else "n/a"
        print(
            f"  rows: {n_before} before the finite-row intersection -> {n_after} after "
            f"({rows_dropped} dropped, {pct_str})"
        )
    else:
        print("  rows: no usable trial series existed to intersect")
    print(
        f"MEASURED effective_n_trials = {_fmt(n_eff)}   "
        f"(this run's PLANNED trial count = {merged['n_planned_run']}; "
        f"full-registry production planned count = {sf.n_planned_trials()})"
    )
    if merged["n_eff_note"]:
        print(f"  {merged['n_eff_note']}")
    print(f"PBO (CSCV)                  = {_fmt(pbo)}")
    if merged["pbo_note"]:
        print(f"  {merged['pbo_note']}")
    print(
        f"var_trial_sharpes (MEASURED)= {_fmt(var_ts, '.6g')}   "
        f"(expected_max_sharpe used as DSR sr0 = {_fmt(merged['exp_max_sharpe'], '.6g')})"
    )
    if merged["exp_max_sharpe_note"]:
        print(f"  {merged['exp_max_sharpe_note']}")
    print("=" * 88)

    rows = []
    for key, dsr in merged["per_trial_dsr"].items():
        name, horizon = key
        rows.append((name, horizon, merged["per_trial_sharpe"][key], dsr))
    rows.sort(key=lambda r: (-r[3] if np.isfinite(r[3]) else float("inf")))

    print(f"{'feature':30s} {'horizon':>8s} {'sharpe':>10s} {'deflated_sharpe':>16s}")
    for name, horizon, sharpe, dsr in rows:
        print(f"{name:30s} {horizon:>8s} {sharpe:10.4f} {dsr:16.4f}")

    excluded = merged["excluded_trials"]
    if excluded:
        print("\nEXCLUDED FROM THE TRIAL MATRIX (all-NaN / zero-variance) -- a RESULT, not a bug:")
        for (name, horizon), reason in sorted(excluded.items()):
            print(f"  {name} (horizon={horizon}): {reason}")

    if merged["failed_records"]:
        print("\nFAILED TRIALS (raised -- a RESULT, not dropped, per E5/obligation 11):")
        for r in merged["failed_records"]:
            print(f"  {r.strategy} {r.params_json}: {r.error}")
    else:
        print("\nNo trials raised.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.parallel:
        t0 = time.monotonic()
        launch_shards(args)
        merged = merge_shards(args.output_dir, args.n_shards)
        print(
            f"[parallel] {args.n_shards} shard(s) + merge finished in "
            f"{time.monotonic() - t0:.1f}s",
            flush=True,
        )
        print_report(merged)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
