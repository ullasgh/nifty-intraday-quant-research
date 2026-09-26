"""Pre-registration for the stress-gated residual liquidity-provision fade.

Spec: ``specs/liqprov_fade.md`` section 5.1. NO P&L IS COMPUTED ANYWHERE IN THIS SCRIPT:
it measures the k grid and the stress-gate thresholds/frequencies on the test window's
feature distributions only, and freezes them (plus the full 12-trial grid, notionals,
kill criteria and DSR bar) into ``results/liqprov/prereg.json`` with a ``config_hash``.

Usage:
    uv run python scripts/prereg_liqprov.py

Deterministic: re-running reproduces the same ``config_hash``. The hash is
blake2s(digest_size=8) over the canonical JSON (``sort_keys=True``,
``separators=(",", ":")``) of every key except ``config_hash`` and ``provenance`` (the
latter holds only the generation timestamp and the git sha / dirty flag).

Never calls ``HoldoutLock.record_read``; the read count is checked before and after and
the script aborts if it moved. Nothing under ``data/`` is written.

``scripts/run_liqprov_backtest.py`` imports this module so that its fresh recomputation
of the k grid and gate thresholds uses byte-for-byte the same code path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nifty_quant import settings
from nifty_quant.calendar import TradingCalendar
from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.research.liqprov.features import (
    index_dev,
    index_twap,
    prior_session_beta,
    residual_stretch,
    rolling_session_quantile,
    session_extreme,
    stress_gate,
    vix_change,
)
from nifty_quant.research.splits import (
    HoldoutBoundaryError,
    HoldoutLock,
    default_holdout_lock_path,
)
from nifty_quant.research.vwap_adr.features import (
    SESSION_START_MINUTE,
    SQUARE_OFF_MINUTE,
    adr_pct,
    session_range_pct,
    session_vwap,
)
from nifty_quant.universe.static import load_universe, survivorship_report

# --------------------------------------------------------------------------- frozen config

SCHEMA = "liqprov_prereg_v1"
SPEC = "specs/liqprov_fade.md"
UNIVERSE = "nifty50"
INDEX_SYMBOL = "NIFTY50"
VIX_SYMBOL = "INDIAVIX"
OHLCV = ("open", "high", "low", "close", "volume")

#: Test window (spec section 0): never used for this strategy family.
WINDOW_START = dt.date(2018, 1, 1)
WINDOW_END = dt.date(2023, 8, 13)
#: Warm-up bars are loaded from here, used ONLY for ADR/beta/rolling-quantile history.
LOAD_START = dt.date(2017, 7, 17)

ADR_N = 10  # stock ADR and index ADR
INDEX_ADR_N = 10
BETA_LOOKBACK = 20
GATE_QS = (0.80, 0.90, 0.95)
#: rolling_session_quantile defaults (spec 1.7), recorded explicitly.
QUANTILE_WINDOW = 250
QUANTILE_MIN_SESSIONS = 120
#: k grid: k90 = -quantile(minz, 0.10), k95 = -quantile(minz, 0.05) (spec 5.1 step 5).
K_QUANTILES = {"k90": 0.10, "k95": 0.05}
MODES = ("passive", "market")
NOTIONAL = 1_000_000.0
NOTIONAL_SENSITIVITY = 100_000.0
TICK = 0.05
KILL_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
DSR_BAR = 0.95
PBO_SPLITS = 16

KILL_CRITERIA_TEXT = [
    "All criteria are evaluated per trial on net_bps / net_pnl at notional Rs 10L "
    "(Rs 1L is a reported sensitivity only), with NSEIntradayEquityCosts() + "
    "SqrtImpactSlippage(), over usable sessions in [2018-01-01, 2023-08-13].",
    "c1_mean_net_positive: mean net_bps over all trades > 0.",
    "c2_day_t_gt_2: day-clustered t of net_bps (trades collapsed to one mean per session, "
    "mean / (std(ddof=1) / sqrt(n_days))) > 2.",
    "c3_positive_years_ge_4: at least 4 of the 6 calendar years 2018-2023 have summed "
    "net_pnl > 0 (a year with no trades is not positive).",
    "c4_ex_top5_positive: mean net_bps after dropping all trades on the 5 best days by "
    "total net_pnl > 0.",
    "A trial passes only if c1-c4 all hold (any NaN fails).",
    f"Multiple-testing bar: the deflated Sharpe (per-period, sr0 = expected_max_sharpe("
    f"effective_n_trials, var of per-period trial Sharpes)) of the best trial at Rs 10L "
    f"must be >= {DSR_BAR}. PBO (CSCV, {PBO_SPLITS} splits) is reported alongside.",
    "The hypothesis is supported only if at least one trial passes c1-c4 AND the DSR bar "
    "is met. Otherwise the family is killed; no re-parameterisation on this window.",
]


class Abort(RuntimeError):
    """A spec-defined abort condition."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trial_id(q: float, k: float, mode: str) -> str:
    return f"q{q:.2f}_k{k:.3f}_{mode}"


# --------------------------------------------------------------------------- hashing


def _jsonable(x: Any) -> Any:
    """Plain-python, JSON-safe copy (NaN/inf -> None; numpy scalars -> python)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    return x


HASH_EXCLUDED_KEYS = ("config_hash", "provenance")


def canonical_json(body: dict) -> str:
    return json.dumps(_jsonable(body), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(doc: dict) -> str:
    """blake2s(8) over the canonical JSON of ``doc`` minus config_hash/provenance."""
    body = {k: v for k, v in doc.items() if k not in HASH_EXCLUDED_KEYS}
    return hashlib.blake2s(canonical_json(body).encode("utf-8"), digest_size=8).hexdigest()


def array_digest(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    # Canonicalise NaN payloads so the digest depends only on values.
    arr = np.where(np.isnan(arr), np.nan, arr)
    return hashlib.blake2s(arr.tobytes(), digest_size=8).hexdigest()


def git_provenance() -> dict:
    def _run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, cwd=settings.REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "git_sha": _run("git", "rev-parse", "HEAD") or None,
        "git_dirty": bool(_run("git", "status", "--porcelain", "--untracked-files=no")),
    }


# --------------------------------------------------------------------------- data


def resolve_universe(name: str = UNIVERSE):
    universe = load_universe(name)
    if universe.source == "all_data_symbols_fallback":
        raise Abort(
            f"universe {name!r} resolved to the silent fallback "
            "(all_data_symbols_fallback); configs/universe/<name>.yaml missing or invalid"
        )
    missing = [s for s in universe.symbols if not (settings.BARS_1M / s).is_dir()]
    if missing:
        raise Abort(f"universe {name!r}: no data/bars/1/<sym> directory for {missing}")
    for idx_sym in (INDEX_SYMBOL, VIX_SYMBOL):
        if idx_sym in universe.symbols:
            raise Abort(f"index symbol {idx_sym} inside stock universe {name!r}")
        if not (settings.BARS_1M / idx_sym).is_dir():
            raise Abort(f"no data/bars/1/{idx_sym} directory")
    return universe


def resolve_holdout(lock: HoldoutLock, calendar: TradingCalendar) -> tuple[dt.date, dt.date]:
    try:
        return lock.holdout_range(calendar.session_dates())
    except HoldoutBoundaryError as exc:
        raise Abort(f"holdout boundary error: {exc}") from exc


@dataclass
class Context:
    """Panel arrays + features shared by the prereg and the runner. All float64."""

    load_start: dt.date
    win_start: dt.date
    win_end: dt.date
    stock_syms: tuple[str, ...]
    panel: Any
    dates: list[dt.date]
    offs: np.ndarray
    mod: np.ndarray
    ts: np.ndarray
    usable: np.ndarray
    in_window: np.ndarray
    session_ok: np.ndarray
    unknown_dates: list[dt.date]
    excluded_in_window: list[dt.date]
    stock: dict[str, np.ndarray]
    index: dict[str, np.ndarray]
    vix_close: np.ndarray
    feats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok_days(self) -> np.ndarray:
        return np.flatnonzero(self.session_ok)


def load_context(
    universe,
    calendar: TradingCalendar,
    load_start: dt.date,
    win_start: dt.date,
    win_end: dt.date,
) -> Context:
    stocks = tuple(universe.symbols)
    all_syms = stocks + (INDEX_SYMBOL, VIX_SYMBOL)
    spec = PanelSpec(freq="1", fields=OHLCV, symbols=all_syms, start=load_start, end=win_end)
    tp = time.time()
    panel = load_panel(spec)
    log(
        f"panel loaded in {time.time() - tp:.1f}s: {panel.n_rows()} rows x "
        f"{panel.n_symbols()} symbols, {panel.n_days()} sessions "
        f"[{panel.dates[0]} .. {panel.dates[-1]}]"
    )
    if set(panel.symbols) != set(all_syms) or len(panel.symbols) != len(all_syms):
        raise Abort(f"panel symbols {panel.symbols} differ from requested {all_syms}")
    # load_panel returns its own (sorted) column order: map every column by NAME.
    stock_set = set(stocks)
    stock_syms = tuple(s for s in panel.symbols if s in stock_set)
    s_cols = np.array([panel.sym_ix[s] for s in stock_syms], dtype=np.int64)
    i_col = panel.sym_ix[INDEX_SYMBOL]
    v_col = panel.sym_ix[VIX_SYMBOL]

    dates = list(panel.dates)
    if dates[-1] > win_end:
        raise Abort(f"panel extends to {dates[-1]}, past window end {win_end}")
    if dates[0] < load_start:
        raise Abort(f"panel starts {dates[0]}, before load start {load_start}")

    offs = np.asarray(panel.day_offsets, dtype=np.int64)
    mod = np.asarray(panel.minute_of_day(), dtype=np.int64)
    ts = np.asarray(panel.ts, dtype=np.int64)

    stock: dict[str, np.ndarray] = {}
    index: dict[str, np.ndarray] = {}
    vix_close = np.empty(0)
    for f in OHLCV:
        raw = panel.field(f)
        stock[f] = np.asarray(raw[:, s_cols], dtype=np.float64)
        index[f] = np.asarray(raw[:, i_col], dtype=np.float64)
        if f == "close":
            vix_close = np.asarray(raw[:, v_col], dtype=np.float64)

    unknown: list[dt.date] = []
    usable = np.zeros(len(dates), dtype=bool)
    for i, d in enumerate(dates):
        try:
            usable[i] = calendar.is_usable(d)
        except KeyError:
            unknown.append(d)
    in_window = np.array([win_start <= d <= win_end for d in dates], dtype=bool)
    session_ok = in_window & usable
    if not session_ok.any():
        raise Abort("no usable in-window sessions")
    excluded = [dates[i] for i in np.flatnonzero(in_window & ~usable)]
    log(
        f"sessions: {len(dates)} in panel, {int(in_window.sum())} in window, "
        f"{int(session_ok.sum())} in-window usable; excluded in-window: "
        f"{[str(d) for d in excluded]}; unknown to calendar: {[str(d) for d in unknown]}"
    )
    return Context(
        load_start=load_start,
        win_start=win_start,
        win_end=win_end,
        stock_syms=stock_syms,
        panel=panel,
        dates=dates,
        offs=offs,
        mod=mod,
        ts=ts,
        usable=usable,
        in_window=in_window,
        session_ok=session_ok,
        unknown_dates=unknown,
        excluded_in_window=excluded,
        stock=stock,
        index=index,
        vix_close=vix_close,
    )


def compute_features(
    ctx: Context,
    qs: tuple[float, ...] = GATE_QS,
    q_window: int = QUANTILE_WINDOW,
    q_min_sessions: int = QUANTILE_MIN_SESSIONS,
) -> dict[str, Any]:
    """Stock/index ADR, beta, z, vix_change, idx_dev_adr and per-q thresholds + gates."""
    tf = time.time()
    s, ix, offs, mod = ctx.stock, ctx.index, ctx.offs, ctx.mod
    vwap = session_vwap(s["high"], s["low"], s["close"], s["volume"], mod, offs)
    rng = session_range_pct(s["high"], s["low"], s["close"], mod, offs)
    adr = adr_pct(rng, ADR_N)
    idx_rng = session_range_pct(
        ix["high"][:, None], ix["low"][:, None], ix["close"][:, None], mod, offs
    )
    idx_adr = adr_pct(idx_rng, INDEX_ADR_N)[:, 0]
    beta = prior_session_beta(s["close"], ix["close"], mod, offs, ctx.usable, BETA_LOOKBACK)
    twap = index_twap(ix["high"], ix["low"], ix["close"], mod, offs)
    idev = index_dev(ix["close"], twap)
    z = residual_stretch(s["close"], vwap, idev, beta, adr, offs)
    vchg = vix_change(ctx.vix_close, mod, offs)

    idx_adr_row = np.repeat(idx_adr, np.diff(offs))
    ok = np.isfinite(idev) & np.isfinite(idx_adr_row) & (idx_adr_row > 0)
    idev_adr = np.full(idev.shape, np.nan, dtype=np.float64)
    idev_adr[ok] = idev[ok] / idx_adr_row[ok]

    vmax = session_extreme(vchg, mod, offs, "max")
    dmin = session_extreme(idev_adr, mod, offs, "min")
    thresholds: dict[float, dict[str, np.ndarray]] = {}
    for q in qs:
        v_day = rolling_session_quantile(vmax, ctx.usable, q, q_window, q_min_sessions)
        m_day = rolling_session_quantile(dmin, ctx.usable, 1.0 - q, q_window, q_min_sessions)
        gate = stress_gate(vchg, idev_adr, v_day, m_day, offs)
        thresholds[q] = {"v_day": v_day, "m_day": m_day, "gate": gate}
    log(f"features computed in {time.time() - tf:.1f}s")
    feats = {
        "vwap": vwap,
        "adr": adr,
        "idx_adr": idx_adr,
        "beta": beta,
        "twap": twap,
        "idx_dev": idev,
        "z": z,
        "vix_chg": vchg,
        "idx_dev_adr": idev_adr,
        "vix_max_day": vmax,
        "idx_dev_adr_min_day": dmin,
        "thresholds": thresholds,
        "q_window": q_window,
        "q_min_sessions": q_min_sessions,
    }
    ctx.feats = feats
    return feats


def _describe(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(v.size),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "max": float(v.max()),
    }


def measure_k_grid(ctx: Context) -> dict:
    """Spec 5.1 step 5: quantiles of per-(symbol, session) min z over usable in-window days."""
    minz = session_extreme(ctx.feats["z"], ctx.mod, ctx.offs, "min")
    win = minz[ctx.session_ok]
    vals = win[np.isfinite(win)]
    if vals.size == 0:
        raise Abort("session min z has no finite values in the window; cannot derive k grid")
    out: dict[str, Any] = {
        "definition": (
            "minz = session_extreme(z, 'min') per (symbol, session) over 09:16 <= t < 15:20, "
            "restricted to usable in-window sessions and finite values; "
            "k90 = -quantile(minz, 0.10), k95 = -quantile(minz, 0.05) (numpy 'linear'), "
            "rounded to 3 dp"
        ),
        "sample_size": int(vals.size),
        "total_pairs": int(win.size),
        "n_sessions": int(ctx.session_ok.sum()),
        "n_symbols": len(ctx.stock_syms),
        "minz_describe": _describe(vals),
        "quantiles": {},
        "k": {},
    }
    for name, qq in K_QUANTILES.items():
        qv = float(np.quantile(vals, qq))
        out["quantiles"][name] = {"q": qq, "minz_quantile": qv}
        k = round(-qv, 3)
        if not (math.isfinite(k) and k > 0):
            raise Abort(f"{name} = {k} is not a positive finite k")
        out["k"][name] = k
    out["grid"] = sorted(set(out["k"].values()))
    return out


def measure_gates(ctx: Context) -> dict:
    """Spec 5.1 step 6 plus the threshold fingerprints the runner re-checks."""
    offs, mod = ctx.offs, ctx.mod
    ok = ctx.session_ok
    n_ok = int(ok.sum())
    starts = offs[:-1]
    trade_rows = (mod >= SESSION_START_MINUTE) & (mod < SQUARE_OFF_MINUTE)
    out: dict[str, Any] = {
        "definition": (
            "gate = stress_gate(vix_chg, idx_dev_adr, v_day, m_day); "
            "v_day = rolling_session_quantile(session_extreme(vix_chg,'max'), usable, q); "
            "m_day = rolling_session_quantile(session_extreme(idx_dev_adr,'min'), usable, 1-q); "
            f"window={ctx.feats['q_window']}, min_sessions={ctx.feats['q_min_sessions']}; "
            "idx_dev_adr = idx_dev / index ADR(N=10). sessions_fired counts in-window usable "
            "sessions with the gate True on at least one row (any minute); "
            "sessions_fired_trade_window restricts to rows 09:16 <= t < 15:20."
        ),
        "n_sessions": n_ok,
        "vix_max_day_describe": _describe(ctx.feats["vix_max_day"][ok]),
        "idx_dev_adr_min_day_describe": _describe(ctx.feats["idx_dev_adr_min_day"][ok]),
        "per_q": {},
    }
    for q, th in ctx.feats["thresholds"].items():
        gate = th["gate"]
        fired = np.add.reduceat(gate.astype(np.int64), starts) > 0
        fired_tw = np.add.reduceat((gate & trade_rows).astype(np.int64), starts) > 0
        both_finite = np.isfinite(th["v_day"]) & np.isfinite(th["m_day"])
        fin_days = np.flatnonzero(ok & both_finite)
        n_fired = int((fired & ok).sum())
        out["per_q"][f"{q:.2f}"] = {
            "q": q,
            "sessions_fired": n_fired,
            "frac_sessions_fired": n_fired / n_ok,
            "sessions_fired_trade_window": int((fired_tw & ok).sum()),
            "gated_rows": int((gate & np.repeat(ok, np.diff(offs))).sum()),
            "sessions_with_finite_thresholds": int(fin_days.size),
            "first_session_with_finite_thresholds": (
                str(ctx.dates[fin_days[0]]) if fin_days.size else None
            ),
            "v_day_describe": _describe(th["v_day"][ok]),
            "m_day_describe": _describe(th["m_day"][ok]),
            "v_day_digest": array_digest(th["v_day"][ok]),
            "m_day_digest": array_digest(th["m_day"][ok]),
            "fired_sessions_digest": hashlib.blake2s(
                np.ascontiguousarray((fired & ok)[ok]).tobytes(), digest_size=8
            ).hexdigest(),
        }
    return out


def build_trials(k_grid: dict) -> list[dict]:
    trials = []
    for q in GATE_QS:
        for kname in K_QUANTILES:
            k = k_grid["k"][kname]
            for mode in MODES:
                trials.append(
                    {"id": trial_id(q, k, mode), "q": q, "k_name": kname, "k": k, "mode": mode}
                )
    return trials


def build_body(
    ctx: Context,
    universe,
    holdout: tuple[dt.date, dt.date],
    holdout_reads: tuple[int, int],
) -> dict:
    """Everything that goes into config_hash (no timestamps, no git sha)."""
    k_grid = measure_k_grid(ctx)
    gates = measure_gates(ctx)
    surv = survivorship_report(universe, ctx.win_start, ctx.win_end)
    return {
        "schema": SCHEMA,
        "spec": SPEC,
        "strategy": "liqprov_fade (stress-gated residual liquidity-provision, long only)",
        "universe": {
            "name": universe.name,
            "source": universe.source,
            "n_symbols": len(ctx.stock_syms),
            "symbols": list(ctx.stock_syms),
            "index_symbol": INDEX_SYMBOL,
            "vix_symbol": VIX_SYMBOL,
            "survivorship": surv.to_dict(),
            "survivorship_warning": surv.warning_line(),
        },
        "window": {
            "test_start": str(ctx.win_start),
            "test_end": str(ctx.win_end),
            "load_start": str(ctx.load_start),
            "panel_first_date": str(ctx.dates[0]),
            "panel_last_date": str(ctx.dates[-1]),
            "panel_sessions": len(ctx.dates),
            "panel_rows": int(ctx.offs[-1]),
            "in_window_sessions": int(ctx.in_window.sum()),
            "in_window_usable_sessions": int(ctx.session_ok.sum()),
            "excluded_in_window_unusable": [str(d) for d in ctx.excluded_in_window],
            "unknown_to_calendar": [str(d) for d in ctx.unknown_dates],
            "usable_definition": "TradingCalendar.from_index_bars('NIFTY50').is_usable",
        },
        "holdout": {
            "start": str(holdout[0]),
            "end": str(holdout[1]),
            "read_count_before": holdout_reads[0],
            "read_count_after": holdout_reads[1],
            "record_read_called": False,
        },
        "features": {
            "start_minute": SESSION_START_MINUTE,
            "square_off_minute": SQUARE_OFF_MINUTE,
            "stock_adr_n": ADR_N,
            "index_adr_n": INDEX_ADR_N,
            "beta_lookback": BETA_LOOKBACK,
            "quantile_window": ctx.feats["q_window"],
            "quantile_min_sessions": ctx.feats["q_min_sessions"],
            "gate_qs": list(GATE_QS),
        },
        "k_grid": k_grid,
        "gates": gates,
        "trials": build_trials(k_grid),
        "n_trials": len(GATE_QS) * len(K_QUANTILES) * len(MODES),
        "execution": {
            "notional": NOTIONAL,
            "notional_sensitivity": NOTIONAL_SENSITIVITY,
            "tick": TICK,
            "cost_model": "NSEIntradayEquityCosts()",
            "slippage_model": "SqrtImpactSlippage() (half_spread 1.5 bps, impact_coef 10; "
            "ASSUMED, not calibrated); passive entries pay no slippage",
            "tradable": "tradable_mask(panel) on stock columns only (default min_adv_inr=5e7); "
            "unusable and out-of-window sessions untradable",
            "long_only": True,
            "stop_loss": None,
        },
        "kill_criteria": {
            "years": list(KILL_YEARS),
            "text": KILL_CRITERIA_TEXT,
            "dsr_bar": DSR_BAR,
            "pbo_splits": PBO_SPLITS,
        },
    }


def load_prereg(path: Path) -> dict:
    """Load prereg.json and verify its config_hash recomputes. Abort otherwise."""
    if not path.is_file():
        raise Abort(f"pre-registration {path} does not exist; run scripts/prereg_liqprov.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    stored = doc.get("config_hash")
    fresh = config_hash(doc)
    if stored != fresh:
        raise Abort(f"prereg config_hash mismatch: stored {stored}, recomputed {fresh}")
    return doc


def diff_bodies(a: Any, b: Any, path: str = "", rel_tol: float = 1e-9) -> list[str]:
    """Structural diff; floats compared with relative tolerance, everything else exactly."""
    a, b = _jsonable(a), _jsonable(b)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one")
            else:
                out.extend(diff_bodies(a[k], b[k], f"{path}/{k}", rel_tol))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_bodies(x, y, f"{path}[{i}]", rel_tol))
        return out
    if isinstance(a, float) and isinstance(b, float) and not isinstance(a, bool):
        if math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0):
            return []
        return [f"{path}: {a!r} != {b!r}"]
    if a != b or type(a) is not type(b):
        return [f"{path}: {a!r} != {b!r}"]
    return []


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="results/liqprov/prereg.json")
    args = p.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = settings.REPO_ROOT / out

    universe = resolve_universe()
    log(f"universe {universe.name}: {len(universe.symbols)} symbols (source={universe.source})")

    lock = HoldoutLock(path=default_holdout_lock_path())
    reads_before = lock.read_count()
    log(f"holdout lock {lock.path}: read_count before = {reads_before}")
    calendar = TradingCalendar.from_index_bars(INDEX_SYMBOL)
    ho_start, ho_end = resolve_holdout(lock, calendar)
    if WINDOW_END >= ho_start:
        raise Abort(f"test window end {WINDOW_END} >= holdout start {ho_start}")
    log(f"holdout [{ho_start}, {ho_end}]; test window [{WINDOW_START}, {WINDOW_END}]")

    ctx = load_context(universe, calendar, LOAD_START, WINDOW_START, WINDOW_END)
    if len(ctx.stock_syms) != 50:
        raise Abort(f"expected 50 stock columns, got {len(ctx.stock_syms)}")
    compute_features(ctx)

    reads_after = lock.read_count()
    log(f"holdout lock read_count after = {reads_after}")
    if reads_after != reads_before:
        raise Abort(f"holdout read count changed: {reads_before} -> {reads_after}")

    body = build_body(ctx, universe, (ho_start, ho_end), (reads_before, reads_after))
    doc = dict(body)
    doc["provenance"] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/prereg_liqprov.py",
        **git_provenance(),
    }
    doc["config_hash"] = config_hash(doc)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(doc), indent=2, sort_keys=True, allow_nan=False) + "\n")

    kg = body["k_grid"]
    log(
        f"k grid (sample {kg['sample_size']} of {kg['total_pairs']} pairs): "
        + ", ".join(
            f"{n}: minz q{v['q']:.2f}={v['minz_quantile']:.6f} -> k={kg['k'][n]:.3f}"
            for n, v in kg["quantiles"].items()
        )
    )
    for qk, g in body["gates"]["per_q"].items():
        first = g["first_session_with_finite_thresholds"]
        log(
            f"gate q={qk}: fired on {g['sessions_fired']}/{body['gates']['n_sessions']} "
            f"sessions ({g['frac_sessions_fired']:.4f}); finite thresholds on "
            f"{g['sessions_with_finite_thresholds']} (first {first})"
        )
    log(f"wrote {out}; config_hash={doc['config_hash']}; runtime {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Abort as exc:
        print(f"ABORT: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
