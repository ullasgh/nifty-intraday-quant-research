#!/usr/bin/env python3
"""Phase G3 experiment: 7-scheme comparison on the tilt candidate. FROZEN 2026-08-31.

With AMENDMENT 1 fixes for sigma annualization, NaN rho handling, and degeneracy refusal.
"""

from __future__ import annotations

import argparse
import datetime
import json
from typing import Any

import numpy as np

START = datetime.date(2018, 1, 1)
END = datetime.date(2025, 7, 31)

VOL_SIGMA_HALFLIFE_BARS = 20.0
VOL_CORR_WINDOW_BARS = 30
CAPITAL = 1_000_000.0
GROSS_WEIGHT = 1.0
MAX_WEIGHT = 0.05
RECENT_N_SESSIONS = 252

SCHEME_NAMES = [
    "equal_weight",
    "inverse_vol",
    "z_weight",
    "z_over_vol",
    "rank_weight",
    "risk_parity",
    "covariance_aware",
]


def _compute_mapped_signal(overnight_returns_valid: np.ndarray) -> np.ndarray:
    """Map overnight returns to [0, 1] rank-percentile (losers highest)."""
    n_valid = len(overnight_returns_valid)
    order = np.argsort(overnight_returns_valid, kind="stable")
    ranks = np.empty(n_valid, dtype=np.float64)
    ranks[order] = np.arange(n_valid, dtype=np.float64)
    rank_pct = ranks / max(1, n_valid - 1)
    return rank_pct.astype(np.float64)


def _compute_annualized_sigma(
    returns_hist_clean: list[float], sigma_floor: float
) -> float:
    """Compute annualized EWMA sigma from per-session returns.

    AMENDMENT 1: sigma_t = sqrt(EWMA of squared returns, halflife 20 sessions) * sqrt(252),
    then max(sigma_floor, .).
    """
    if len(returns_hist_clean) == 0:
        return np.nan

    alpha = 1.0 - np.exp(-np.log(2.0) / VOL_SIGMA_HALFLIFE_BARS)

    ewma_var = returns_hist_clean[0] ** 2
    for j in range(1, len(returns_hist_clean)):
        ewma_var = alpha * (returns_hist_clean[j] ** 2) + (1.0 - alpha) * ewma_var

    sigma_ann = np.sqrt(ewma_var) * np.sqrt(252.0)
    return float(max(sigma_ann, sigma_floor))


def _compute_median_correlation(returns_window: np.ndarray) -> float:
    """Compute median pairwise correlation from returns window."""
    returns_window = np.asarray(returns_window, dtype=np.float64)
    if returns_window.shape[0] < 2 or returns_window.shape[1] < 2:
        return np.nan

    corr_matrix = np.corrcoef(returns_window, rowvar=False)
    if not np.all(np.isfinite(corr_matrix)):
        return np.nan

    n = corr_matrix.shape[0]
    upper_tri = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri.append(corr_matrix[i, j])

    if len(upper_tri) == 0:
        return np.nan

    return float(np.median(upper_tri))


def main() -> None:
    """Run the phase G3 comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Phase G3 experiment: 7-scheme comparison on the tilt candidate."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print everything but write NO trials.",
    )
    args = parser.parse_args()

    print("Loading universe and panel...", flush=True)
    from nifty_quant import settings
    from nifty_quant.backtest.metrics import deflated_sharpe, effective_n_trials
    from nifty_quant.backtest.weighting import apply_weight_scheme
    from nifty_quant.calendar import TradingCalendar
    from nifty_quant.data.panel import PanelSpec, load_panel
    from nifty_quant.execution.costs import NSEIntradayEquityCosts
    from nifty_quant.features.core import SIGMA_FLOOR
    from nifty_quant.research.contract import ResearchContract, canonical_hash
    from nifty_quant.research.hypotheses.h2_overnight_reversal import (
        _build_checkpoint_panel,
        build_overnight_feature,
    )
    from nifty_quant.research.provenance import (
        FEATURE_VERSION,
        compute_panel_hash,
        compute_universe_hash,
        embargo_components_json,
        get_git_sha,
    )
    from nifty_quant.research.registry import TrialRecord, TrialRegistry
    from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path
    from nifty_quant.universe.static import load_universe

    cal = TradingCalendar.from_index_bars("NIFTY50")
    real_dates = cal.session_dates()
    holdout_lock = HoldoutLock(path=default_holdout_lock_path())
    holdout_start, holdout_end = holdout_lock.holdout_range(real_dates)
    if END >= holdout_start:
        raise ValueError(
            f"end date {END} falls in holdout window [{holdout_start}, {holdout_end}]"
        )

    universe = load_universe("all_equity")
    print(f"Universe: {len(universe.symbols)} symbols.", flush=True)

    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=universe.symbols,
        start=START,
        end=END,
    )
    print(f"Loading panel {START} .. {END}...", flush=True)
    panel = load_panel(spec, memmap=True)
    print(
        f"Panel: {panel.n_rows()} rows, {panel.n_days()} days, "
        f"{panel.n_symbols()} symbols.",
        flush=True,
    )

    print("Building overnight feature and checkpoint panel...", flush=True)
    feature = build_overnight_feature(panel)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, feature)
    print(
        f"Checkpoint panel: {checkpoint_panel.n_days()} sessions, "
        f"{checkpoint_panel.n_rows()} rows.",
        flush=True,
    )

    n_sessions = checkpoint_panel.n_days()
    n_symbols = panel.n_symbols()
    close_field = checkpoint_panel.field("close")

    print("Precomputing per-session metrics...", flush=True)

    session_data: list[
        tuple[
            datetime.date,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            int,
        ]
    ] = []

    returns_history_full: list[np.ndarray] = []
    returns_history_window: list[np.ndarray] = []

    n_warmup_sigma = max(int(2 * VOL_SIGMA_HALFLIFE_BARS), 2)
    n_warmup_corr = int(VOL_CORR_WINDOW_BARS)

    n_sessions_skipped_warmup = 0
    n_sessions_skipped_nan_rho = 0
    sigma_floor_count = 0
    sigma_total_count = 0

    for i in range(n_sessions):
        date = checkpoint_panel.dates[i]
        entry_row = 2 * i
        exit_row = 2 * i + 1

        feat = checkpoint_feature.values[entry_row, :].astype(np.float64)
        price_entry = close_field[entry_row, :].astype(np.float64)
        price_exit = close_field[exit_row, :].astype(np.float64)

        base_valid = (
            np.isfinite(feat)
            & np.isfinite(price_entry)
            & (price_entry > 0)
            & np.isfinite(price_exit)
        )

        r = np.full(n_symbols, np.nan, dtype=np.float64)
        r[base_valid] = price_exit[base_valid] / price_entry[base_valid] - 1.0

        returns_history_full.append(r.copy())
        returns_history_window.append(r.copy())
        if len(returns_history_window) > int(VOL_CORR_WINDOW_BARS):
            returns_history_window.pop(0)

        if len(returns_history_window) < n_warmup_corr:
            n_sessions_skipped_warmup += 1
            continue

        sigma_full = np.full(n_symbols, np.nan, dtype=np.float64)
        sigma_warmed_up = np.full(n_symbols, False, dtype=bool)

        for name_idx in range(n_symbols):
            returns_hist = [
                returns_history_full[j][name_idx] for j in range(len(returns_history_full))
            ]
            returns_hist_clean = [r for r in returns_hist if np.isfinite(r)]

            if len(returns_hist_clean) >= n_warmup_sigma:
                sigma_val = _compute_annualized_sigma(returns_hist_clean, SIGMA_FLOOR)
                sigma_full[name_idx] = sigma_val
                sigma_warmed_up[name_idx] = True
                sigma_total_count += 1
                if np.isclose(sigma_val, SIGMA_FLOOR):
                    sigma_floor_count += 1
            else:
                sigma_total_count += 1

        tradable = base_valid & sigma_warmed_up
        n_valid = int(tradable.sum())

        if n_valid < 5:
            n_sessions_skipped_warmup += 1
            continue

        r_valid = r[tradable]

        signal_valid = _compute_mapped_signal(-feat[tradable])
        signal_full = np.zeros(n_symbols, dtype=np.float64)
        signal_full[tradable] = signal_valid

        returns_window = np.array(
            [ret[tradable] for ret in returns_history_window],
            dtype=np.float64,
        )
        valid_rows = np.all(np.isfinite(returns_window), axis=1)
        if np.any(valid_rows) and np.sum(valid_rows) >= 2:
            returns_window_clean = returns_window[valid_rows, :]
            rho = _compute_median_correlation(returns_window_clean)
        else:
            rho = np.nan

        if not np.isfinite(rho):
            n_sessions_skipped_nan_rho += 1
            continue

        benchmark_return = float(np.mean(r_valid[np.isfinite(r_valid)]))

        session_data.append(
            (
                date,
                feat[tradable],
                signal_full.copy(),
                sigma_full.copy(),
                rho,
                tradable.copy(),
                r.copy(),
                r_valid.copy(),
                benchmark_return,
                n_valid,
            )
        )

    print(
        f"Sessions with >= 5 tradable names: {len(session_data)} "
        f"(skipped {n_sessions_skipped_warmup} for warm-up, "
        f"{n_sessions_skipped_nan_rho} for NaN rho)",
        flush=True,
    )

    if sigma_total_count > 0:
        sigma_floor_fraction = sigma_floor_count / sigma_total_count
        print(
            f"Sigma cells at floor: {sigma_floor_fraction * 100:.2f}% "
            f"({sigma_floor_count} / {sigma_total_count})",
            flush=True,
        )
        if sigma_floor_fraction > 0.10:
            print(
                f"ERROR: Sigma degeneracy exceeded 10% threshold. "
                f"SIGMA_FLOOR is a measured p1 (1%); {sigma_floor_fraction * 100:.2f}% "
                f"at floor means estimator is not measuring.",
                flush=True,
            )
            raise SystemExit(1)
    else:
        sigma_floor_fraction = 0.0

    print("Running calibration null...", flush=True)

    equal_weight_excess = []
    book = np.zeros(n_symbols, dtype=np.float64)

    for (
        date,
        _,
        signal_full,
        sigma_full,
        rho,
        tradable,
        r_full,
        r_valid,
        benchmark_return,
        n_valid,
    ) in session_data:
        result = apply_weight_scheme(
            "equal_weight",
            signal=signal_full,
            sigma=sigma_full,
            corr=rho,
            tradable=tradable,
            gross=GROSS_WEIGHT,
            max_weight=MAX_WEIGHT,
        )
        weights = result.weights

        weights_tradable = weights[tradable]
        if np.any(weights_tradable != 0):
            book_return = float(np.sum(weights_tradable * r_valid))
        else:
            book_return = 0.0

        excess_bps = (book_return - benchmark_return) * 10_000.0
        equal_weight_excess.append(float(excess_bps))

        book = weights.copy()

    ew_excess_clean = np.array(
        [x for x in equal_weight_excess if np.isfinite(x)],
        dtype=np.float64,
    )
    if len(ew_excess_clean) < 2:
        calibration_mean = np.nan
        calibration_se = np.nan
    else:
        calibration_mean = float(np.mean(ew_excess_clean))
        calibration_se = float(
            np.std(ew_excess_clean, ddof=1) / np.sqrt(len(ew_excess_clean))
        )

    print(
        f"Calibration null: equal_weight excess = "
        f"{calibration_mean:.2f} +/- {calibration_se:.2f} bps/day",
        flush=True,
    )
    if np.isfinite(calibration_mean) and np.isfinite(calibration_se):
        passes = abs(calibration_mean) < max(calibration_se, 1e-12)
        if not passes:
            print(
                f"ERROR: calibration null FAILED. "
                f"|mean| = {abs(calibration_mean):.2e}, SE = {calibration_se:.2e}",
                flush=True,
            )
            raise SystemExit(1)

    print("Running 7 schemes...", flush=True)

    scheme_results: dict[str, list[tuple[datetime.date, float, float, float, np.ndarray]]] = {
        name: [] for name in SCHEME_NAMES
    }

    for scheme_name in SCHEME_NAMES:
        print(f"  {scheme_name}...", flush=True)
        book = np.zeros(n_symbols, dtype=np.float64)

        for (
            date,
            _,
            signal_full,
            sigma_full,
            rho,
            tradable,
            r_full,
            r_valid,
            benchmark_return,
            n_valid,
        ) in session_data:
            try:
                result = apply_weight_scheme(
                    scheme_name,
                    signal=signal_full,
                    sigma=sigma_full,
                    corr=rho,
                    tradable=tradable,
                    gross=GROSS_WEIGHT,
                    max_weight=MAX_WEIGHT,
                )
            except Exception as e:
                print(
                    f"ERROR: {scheme_name} failed on {date}: {type(e).__name__}: {e}",
                    flush=True,
                )
                raise SystemExit(1)
            weights = result.weights
            clip_binding = 1.0 if result.clip_binding else 0.0

            turnover = float(np.sum(np.abs(weights - book)))

            weights_tradable = weights[tradable]
            if np.any(weights_tradable != 0):
                book_return = float(np.sum(weights_tradable * r_valid))
            else:
                book_return = 0.0
            excess_bps = (book_return - benchmark_return) * 10_000.0

            scheme_results[scheme_name].append(
                (date, excess_bps, turnover, clip_binding, weights.copy())
            )
            book = weights.copy()

    print("Computing per-scheme statistics...", flush=True)

    costs = NSEIntradayEquityCosts()
    scheme_stats: dict[str, dict[str, Any]] = {}

    for scheme_name in SCHEME_NAMES:
        records = scheme_results[scheme_name]
        excess_gross = np.array([r[1] for r in records], dtype=np.float64)
        turnovers = np.array([r[2] for r in records], dtype=np.float64)
        clip_bindings = np.array([r[3] for r in records], dtype=np.float64)

        n_held_counts = []
        for date_idx in range(len(records)):
            weights = scheme_results[scheme_name][date_idx][4]
            n_held = int(np.count_nonzero(weights))
            n_held_counts.append(n_held)

        n_held_typical = max(1, int(np.mean(n_held_counts))) if n_held_counts else 1
        clip_per_name = CAPITAL / n_held_typical
        round_trip_bps_val = costs.round_trip_bps(clip_per_name)

        cost_bps = round_trip_bps_val * turnovers
        excess_net = excess_gross - cost_bps

        gross_bps_mean = float(np.mean(excess_gross))
        net_bps_mean = float(np.mean(excess_net))
        turnover_mean = float(np.mean(turnovers))
        cost_bps_mean = float(np.mean(cost_bps))
        clip_binding_freq = float(np.mean(clip_bindings))

        recent_start_idx = max(0, len(records) - RECENT_N_SESSIONS)
        recent_excess_net = excess_net[recent_start_idx:]

        dsr_recent = deflated_sharpe(recent_excess_net, sr0=0.0)
        promoted = np.isfinite(dsr_recent) and dsr_recent > 0.95

        scheme_stats[scheme_name] = {
            "gross_bps_day": gross_bps_mean,
            "net_bps_day": net_bps_mean,
            "turnover_day": turnover_mean,
            "cost_bps_day": cost_bps_mean,
            "clip_binding_freq": clip_binding_freq,
            "dsr_recent": dsr_recent,
            "promoted": promoted,
            "n_sessions": len(records),
            "excess_net_series": excess_net,
        }

    print("Measuring effective_n_trials...", flush=True)

    trial_matrix = np.column_stack(
        [scheme_stats[name]["excess_net_series"] for name in SCHEME_NAMES]
    )
    n_eff = effective_n_trials(trial_matrix)

    print("Preparing output...", flush=True)

    calibration_passed = (
        abs(calibration_mean) < max(calibration_se, 1e-12)
        if np.isfinite(calibration_mean) and np.isfinite(calibration_se)
        else "N/A"
    )

    table_lines = [
        "# Phase G3 Comparison: 7-Scheme Results",
        "",
        "## Frozen Parameters",
        "",
        f"- Window: {START.isoformat()} to {END.isoformat()}",
        f"- Holdout boundary: {holdout_start.isoformat()}",
        f"- Vol sigma halflife (sessions): {VOL_SIGMA_HALFLIFE_BARS}",
        f"- Vol corr window (sessions): {VOL_CORR_WINDOW_BARS}",
        f"- Capital: Rs {CAPITAL:,.0f}",
        f"- Gross weight: {GROSS_WEIGHT}",
        f"- Max weight per name: {MAX_WEIGHT}",
        f"- Recent window (for DSR): {RECENT_N_SESSIONS} sessions",
        "- DSR promotion threshold: 0.95",
        "",
        "## Calibration Null",
        "",
        "Equal-weight scheme gross excess (must be ~0, |mean| < SE):",
        f"  Mean: {calibration_mean:.2f} bps/day",
        f"  SE: {calibration_se:.2f} bps/day",
        f"  PASSED: {calibration_passed}",
        "",
        "## Warm-up and Degeneracy",
        "",
        f"- Sessions skipped for warmup: {n_sessions_skipped_warmup}",
        f"- Sessions skipped for NaN rho (AMENDMENT 1): {n_sessions_skipped_nan_rho}",
        (
            f"- Sigma cells at floor: {sigma_floor_fraction * 100:.2f}% "
            f"({sigma_floor_count} / {sigma_total_count})"
        ),
        f"- Sessions processed: {len(session_data)}",
        "",
        "## Per-Scheme Statistics",
        "",
        (
            "| Scheme | Gross (bps/day) | Net (bps/day) | Turnover/day | "
            "Cost (bps/day) | Clip % | Recent DSR | Promoted |"
        ),
        "|--------|-----------|-----------|-----------|----------|------|---------|---------|",
    ]

    for scheme_name in SCHEME_NAMES:
        stats = scheme_stats[scheme_name]
        table_lines.append(
            f"| {scheme_name:<20} "
            f"| {stats['gross_bps_day']:>10.2f} "
            f"| {stats['net_bps_day']:>10.2f} "
            f"| {stats['turnover_day']:>10.4f} "
            f"| {stats['cost_bps_day']:>10.2f} "
            f"| {stats['clip_binding_freq'] * 100:>6.1f} "
            f"| {stats['dsr_recent']:>9.4f} "
            f"| {str(stats['promoted']):>9s} |"
        )

    table_lines.extend(
        [
            "",
            "## Effective Trials",
            "",
            f"- Effective n_trials: {n_eff:.4f}",
            f"- Sessions per scheme: {scheme_stats[SCHEME_NAMES[0]]['n_sessions']}",
            "",
        ]
    )

    report_text = "\n".join(table_lines)

    if not args.dry_run:
        print("Writing artifact...", flush=True)
        artifact_path = settings.RESULTS_ROOT / "phase_g3_comparison.md"
        artifact_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"Artifact written: {artifact_path}", flush=True)

    print("", flush=True)
    print(report_text, flush=True)

    if args.dry_run:
        print("", flush=True)
        print("--dry-run: nothing written to registry.", flush=True)
        return

    print("Registering trials...", flush=True)

    from nifty_quant.data.manifest import Manifest

    manifest = Manifest.load()
    panel_hash = compute_panel_hash(panel, adjustments=manifest.adjustments)
    universe_hash = compute_universe_hash(
        name="all_equity",
        symbols=universe.symbols,
        n_sessions=panel.n_days(),
    )
    git_sha = get_git_sha()
    if not git_sha:
        raise ValueError("refusing to register with an empty git_sha")

    contract = ResearchContract(
        data={
            "panel_id": str(spec),
            "panel_hash": panel_hash,
            "start": START.isoformat(),
            "end": END.isoformat(),
            "bar_interval_s": 60,
            "universe_name": "all_equity",
            "universe_hash": universe_hash,
        },
        features={
            "feature_ids": ["h2_overnight_reversal"],
            "feature_version": FEATURE_VERSION,
        },
        label={
            "horizon_bars": 1,
            "construction": "entry_to_exit_excess_over_equal_weight_index",
            "overlapping": False,
        },
        execution={
            "cost_model_id": "nse_intraday_default",
            "slippage_model_id": "none",
            "decision_latency_bars": 0,
            "participation_cap": 1.0,
        },
        portfolio={
            "sizing_scheme": "weighting_scheme_registry",
            "gross_clip": GROSS_WEIGHT,
            "max_weight": MAX_WEIGHT,
            "target_vol": None,
        },
        validation={
            "split_scheme": "none",
            "purge_width_bars": 0,
            "embargo_width_bars": 0,
            "n_planned_trials": len(SCHEME_NAMES),
            "holdout_intent": "never",
        },
        seed=0,
    )

    registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")

    for scheme_name in SCHEME_NAMES:
        stats = scheme_stats[scheme_name]

        payload = {
            "contract_hash": contract.contract_hash,
            "scheme_name": scheme_name,
            "frozen_params": {
                "window_start": START.isoformat(),
                "window_end": END.isoformat(),
                "vol_sigma_halflife_bars": VOL_SIGMA_HALFLIFE_BARS,
                "vol_corr_window_bars": VOL_CORR_WINDOW_BARS,
                "capital": CAPITAL,
                "gross_weight": GROSS_WEIGHT,
                "max_weight": MAX_WEIGHT,
            },
        }
        config_hash = canonical_hash(payload)

        record = TrialRecord(
            config_hash=config_hash,
            contract_hash=contract.contract_hash,
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            strategy="phase_g3_comparison",
            params_json=json.dumps(
                {
                    "scheme_name": scheme_name,
                    "frozen_params": payload["frozen_params"],
                    "result": {
                        "gross_bps_day": stats["gross_bps_day"],
                        "net_bps_day": stats["net_bps_day"],
                        "turnover_day": stats["turnover_day"],
                        "cost_bps_day": stats["cost_bps_day"],
                        "clip_binding_freq": stats["clip_binding_freq"],
                        "dsr_recent": (
                            float(stats["dsr_recent"])
                            if np.isfinite(stats["dsr_recent"])
                            else None
                        ),
                        "promoted": stats["promoted"],
                    },
                },
                sort_keys=True,
                default=str,
            ),
            split_id="full",
            purpose="exploration",  # type: ignore[assignment]
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=stats["turnover_day"],
            breakeven_bps=stats["gross_bps_day"],
            git_sha=git_sha,
            data_fingerprint=manifest.fingerprint,
            code_version="",
            wall_s=0.0,
            result_path="results/phase_g3_comparison.md",
            error=None,
            ruined=False,
            ruin_index=None,
            seed=0,
            universe_name="all_equity",
            universe_hash=universe_hash,
            panel_hash=panel_hash,
            start=START.isoformat(),
            end=END.isoformat(),
            cost_model_id="nse_intraday_default",
            slippage_model_id="none",
            fill_model_id="checkpoint_close_immediate_full",
            embargo_components=embargo_components_json(
                feature_lookback=1.0,
                label_horizon=0.0,
                holding_period=1.0,
                execution_horizon=0.0,
            ),
            parent_trial_id=None,
            feature_version=FEATURE_VERSION,
        )

        print(f"  Registering {scheme_name}...", flush=True)
        contract.register_trial()
        registry.record(record)

    print("Trials registered successfully.", flush=True)


if __name__ == "__main__":
    main()
