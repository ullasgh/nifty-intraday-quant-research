from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from nifty_quant import __version__

# specs/run_provenance.md item 6 (amendment item 5): a registry write failure must be
# logged at WARNING, never swallowed silently.
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - False at runtime by language definition
    # `typing.TYPE_CHECKING` is a hard-coded `False` in every CPython execution; only static
    # checkers (mypy/pyright) treat it as True, and they parse rather than execute. Combined
    # with `from __future__ import annotations` above, the annotations referencing `np`/`Panel`
    # are never evaluated at runtime either, so no call path can reach this block.
    import numpy as np

    from nifty_quant.data.panel import Panel
    from nifty_quant.universe.static import Universe

app = typer.Typer()

# Mirrors the fixed check registry in nifty_quant.data.validate.validate_panel; kept
# here so `nq validate` can report which checks ran even when a check found nothing
# for the requested symbols/year (validate_panel's report only lists checks that
# produced at least one Finding).
_VALIDATE_CHECK_NAMES = (
    "check_timestamps",
    "check_session_lengths",
    "check_ohlc_consistency",
    "check_zero_or_negative_prices",
    "check_stale_bars",
    "check_unexplained_gaps",
    "check_all_nan_columns",
    "check_volume_sanity",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nifty_quant {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Nifty quant research CLI."""


def _fail(msg: str) -> NoReturn:
    typer.echo(msg, err=True)
    raise typer.Exit(code=1)


def _parse_date(value: str, flag: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        _fail(f"Invalid date for {flag}: {value!r} (expected YYYY-MM-DD)")


def _ensure_plugins_loaded() -> None:
    import nifty_quant.strategy.plugins  # noqa: F401


def _build_research_contract(
    *,
    panel_id: str,
    panel_hash: str,
    start: date,
    end: date,
    universe_name: str,
    universe_hash: str,
    cost_model_id: str,
    slippage_model_id: str,
    decision_latency_bars: int,
    n_planned_trials: int,
    holdout_intent: str,
    seed: int,
    split_scheme: str = "none",
    purge_width_bars: int = 0,
    embargo_width_bars: int = 0,
    feature_version: str = "",
) -> Any:
    """Build the `ResearchContract` every `run_backtest()`/`run_tilt()` call site in
    this module now requires (specs/research_contract.md, Enforcement). There is no
    `--contract` CLI flag: the contract is declared from the same inputs the run
    already takes, matching AMENDMENT 1 point 3's direction not to invent a seam
    purely to be testable -- the CLI commands remain the single source of truth for
    what was run, and now additionally declare it as a contract before running.
    """
    from nifty_quant.research.contract import ResearchContract

    return ResearchContract(
        data={
            "panel_id": panel_id,
            "panel_hash": panel_hash,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bar_interval_s": 60,
            "universe_name": universe_name,
            "universe_hash": universe_hash,
        },
        features={"feature_ids": [], "feature_version": feature_version},
        label={
            "horizon_bars": 1,
            "construction": "forward_return",
            "overlapping": False,
        },
        execution={
            "cost_model_id": cost_model_id,
            "slippage_model_id": slippage_model_id,
            "decision_latency_bars": decision_latency_bars,
            "participation_cap": 1.0,
        },
        portfolio={
            "sizing_scheme": "gross_notional",
            "gross_clip": 1.0,
            "max_weight": 1.0,
            "target_vol": None,
        },
        validation={
            "split_scheme": split_scheme,
            "purge_width_bars": purge_width_bars,
            "embargo_width_bars": embargo_width_bars,
            "n_planned_trials": n_planned_trials,
            "holdout_intent": holdout_intent,
        },
        seed=seed,
    )


def _write_returns_parquet(
    path: Path,
    ts: "np.ndarray",
    daily_returns: "np.ndarray",
) -> None:
    """Write the two-column return artifact specified by ``specs/returns_persistence.md``."""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        {
            "ts": np.asarray(ts, dtype=np.int64),
            "return": np.asarray(daily_returns, dtype=np.float64),
        }
    )
    frame.to_parquet(path, index=False)


def _tradable_mask_summary(panel: "Panel", mask: "np.ndarray") -> str:
    """One-line debuggability summary of what the tradable filter excluded.

    Reports two distinct, never-conflated numbers: how many of the
    (row, symbol) cells simply have no bar (``present``), and -- separately --
    how many present bars were excluded by the tradable filter itself
    (liquidity / circuit-lock / staleness gating). ``mask`` is always a subset
    of ``present`` (tradable implies present), so this is a decomposition, not
    a merge of the two concepts.
    """
    import numpy as np

    close = panel.field("close")
    present = ~np.isnan(close)
    total = present.size
    if total == 0:
        return "tradable filter: n/a (empty panel)"
    present_count = int(np.count_nonzero(present))
    tradable_count = int(np.count_nonzero(mask))
    excluded_count = total - tradable_count
    present_but_excluded = present_count - tradable_count
    return (
        f"tradable filter: excluded {100.0 * excluded_count / total:.2f}% of "
        f"(row, symbol) cells overall "
        f"[present={100.0 * present_count / total:.2f}% of cells; "
        f"of present cells, "
        f"{100.0 * present_but_excluded / present_count if present_count else 0.0:.2f}% "
        f"excluded by liquidity/circuit-lock/staleness gating]"
    )


def _apply_pit_eligibility(
    panel: "Panel",
    universe: "Universe",
    tradable_full: "np.ndarray | None",
    *,
    min_history_sessions: int | None,
    min_adv_inr: float,
) -> "np.ndarray | None":
    """Combine point-in-time eligibility into a bar-level tradable mask.

    Returns `tradable_full` unchanged when `min_history_sessions is None`
    (the gate is opt-in only: CLAUDE.md rule 8 forbids a hand-chosen default
    for this threshold). Otherwise computes eligibility via
    `nifty_quant.universe.pit.compute_eligibility`, broadcasts it to bar
    level, and ANDs it into `tradable_full` -- EXCLUDING an ineligible name
    from that session's cross-section entirely (via the same `tradable`
    array strategies already use at decision time to build their
    cross-section), never merely zero-weighting it after the fact. Shared by
    `backtest` and `walkforward` so the two CLI paths cannot silently drift
    into two different eligibility behaviours.
    """
    if min_history_sessions is None:
        return tradable_full

    from nifty_quant.universe.pit import compute_eligibility, eligibility_mask_to_bars

    eligibility = compute_eligibility(
        panel,
        universe,
        min_history_sessions=min_history_sessions,
        min_adv_inr=min_adv_inr,
    )
    eligible_bars = eligibility_mask_to_bars(panel, eligibility)
    typer.echo(
        f"point-in-time eligibility: min_history_sessions={min_history_sessions}, "
        f"min_adv_inr={min_adv_inr:.2e} -- "
        f"{int(eligibility.mask[-1, :].sum())}/{eligibility.mask.shape[1]} names "
        f"eligible on the final session."
    )
    if tradable_full is None:
        return eligible_bars
    return tradable_full & eligible_bars


@app.command()
def info() -> None:
    """Print resolved settings paths and whether they exist."""
    from nifty_quant import settings

    paths = [
        ("DATA_ROOT", settings.DATA_ROOT),
        ("BARS_1M", settings.BARS_1M),
        ("BARS_D", settings.BARS_D),
        ("FUTURES_ROOT", settings.FUTURES_ROOT),
        ("EXTERNAL_ROOT", settings.EXTERNAL_ROOT),
        ("MANIFEST_PATH", settings.MANIFEST_PATH),
        ("CACHE_ROOT", settings.CACHE_ROOT),
        ("RESULTS_ROOT", settings.RESULTS_ROOT),
    ]

    for name, path in paths:
        status = "exists" if path.exists() else "missing"
        typer.echo(f"{name}: {path}  [{status}]")


@app.command()
def symbols(
    equity_only: bool = typer.Option(False, "--equity-only", help="Restrict to equity symbols"),
) -> None:
    """List universe size and manifest coverage."""
    from nifty_quant.data.manifest import Manifest
    from nifty_quant.universe.static import all_data_symbols, equity_symbols

    try:
        all_syms = all_data_symbols()
        eq_syms = equity_symbols()
        typer.echo(f"symbols: {len(all_syms)} total, {len(eq_syms)} equity")
        manifest = Manifest.load()
    except Exception as exc:
        _fail(f"symbols: {exc}")

    selected = tuple(s for s in (eq_syms if equity_only else all_syms) if s in manifest.coverage)
    if selected:
        first = min(manifest.coverage[s].first for s in selected)
        last = max(manifest.coverage[s].last for s in selected)
        typer.echo(f"coverage: {first.date().isoformat()} -> {last.date().isoformat()}")
    else:
        typer.echo("coverage: no data")


@app.command()
def build_panel(
    freq: str = typer.Option("1", "--freq"),
    years: str = typer.Option(..., "--years", help="Comma-separated years, e.g. 2023,2024"),
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build one-year panel cache directories."""
    try:
        years_list = sorted({int(y) for y in years.split(",") if y.strip()})
    except ValueError:
        _fail(f"Invalid --years: {years!r}")

    symbols_tuple = None
    if symbols is not None:
        symbols_tuple = tuple(s.strip() for s in symbols.split(",") if s.strip())

    from nifty_quant.data.panel_builder import build_panel as _build_panel

    try:
        paths = _build_panel(
            freq=freq,
            years=years_list,
            symbols=symbols_tuple,
            force=force,
            progress=True,
        )
    except Exception as exc:
        _fail(f"build-panel failed: {exc}")

    for path in paths:
        typer.echo(str(path))
    typer.echo(f"built {len(paths)} year-cache(s)")


@app.command()
def validate(
    year: int = typer.Option(..., "--year"),
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    out: Path | None = typer.Option(None, "--out", help="Output directory for JSON report"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON report"),
) -> None:
    """Run panel data-quality checks for a calendar year."""
    from nifty_quant.calendar import TradingCalendar
    from nifty_quant.data.panel import PanelSpec, load_panel
    from nifty_quant.data.validate import validate_panel
    from nifty_quant.universe.static import equity_symbols

    symbols_tuple = equity_symbols()
    if symbols is not None:
        symbols_tuple = tuple(s.strip() for s in symbols.split(",") if s.strip())

    start = date(year, 1, 1)
    end = date(year, 12, 31)

    try:
        spec = PanelSpec(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            symbols=symbols_tuple,
            start=start,
            end=end,
        )
        panel = load_panel(spec)
        calendar = TradingCalendar.from_index_bars("NIFTY50")
        report = validate_panel(panel, calendar=calendar)
    except Exception as exc:
        _fail(f"validate failed: {exc}")

    if json_output:
        typer.echo(report.to_json())
    else:
        # Always list every check that ran (not just ones with findings) so a clean
        # symbol/year still shows what was checked, e.g. check_ohlc_consistency.
        typer.echo("checks run: " + ", ".join(_VALIDATE_CHECK_NAMES))
        typer.echo(report.summary())
        typer.echo(f"errors: {len(report.errors())} / findings: {len(report.findings)}")

    if out is not None:
        try:
            out.mkdir(parents=True, exist_ok=True)
            output_path = out / f"dq_{year}.json"
            output_path.write_text(report.to_json(), encoding="utf-8")
            typer.echo(str(output_path))
        except Exception as exc:
            _fail(f"could not write validation report: {exc}")


@app.command()
def strategies() -> None:
    """List registered strategies and their parameter schemas."""
    _ensure_plugins_loaded()
    from nifty_quant.strategy import registry

    try:
        for name in registry.available():
            cls = registry.get(name)
            typer.echo(name)
            for field_name, field_info in cls.Params.model_fields.items():
                default = field_info.default if not field_info.is_required() else "<required>"
                typer.echo(f"    {field_name}: {field_info.annotation} = {default}")
    except Exception as exc:
        _fail(f"strategies: {exc}")


@app.command()
def backtest(
    strategy: str = typer.Option(..., "--strategy"),
    config: Path = typer.Option(..., "--config"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    universe_name: str = typer.Option("all_equity", "--universe"),
    tradable_filter: bool = typer.Option(
        True,
        "--tradable-filter/--no-tradable-filter",
        help=(
            "Gate on liquidity/circuit-lock/staleness via "
            "nifty_quant.data.validate.tradable_mask (enabled by default)."
        ),
    ),
    min_adv_inr: float = typer.Option(
        5e7,
        "--min-adv-inr",
        help=(
            "Minimum 20-session average daily traded value (INR) for the liquidity "
            "component of the tradable filter."
        ),
    ),
    min_history_sessions: int | None = typer.Option(
        None,
        "--min-history-sessions",
        help=(
            "Point-in-time eligibility gate (nifty_quant.universe.pit.compute_eligibility): "
            "a name must have this many PRIOR sessions with a present bar, plus clear "
            "--min-adv-inr on trailing 20-session ADV, to enter this session's cross-section. "
            "Disabled (no gate) unless explicitly set -- CLAUDE.md rule 8 forbids a "
            "hand-chosen default for this threshold, so it must be a deliberate choice per run."
        ),
    ),
    allow_holdout: bool = typer.Option(
        False,
        "--allow-holdout",
        help="Allow a deliberate, recorded read of the stored holdout window.",
    ),
) -> None:
    """Run a single full-sample backtest."""
    start_d = _parse_date(start, "--start")
    end_d = _parse_date(end, "--end")
    if start_d > end_d:
        _fail(f"--start {start_d} is after --end {end_d}")

    _ensure_plugins_loaded()
    from nifty_quant.strategy import registry

    if strategy not in registry.available():
        _fail(f"Unknown strategy {strategy!r}. Available: {', '.join(registry.available())}")

    from nifty_quant.config import load_yaml

    try:
        cfg = load_yaml(config)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    if cfg.get("strategy") != strategy:
        _fail(
            f"--strategy {strategy!r} does not match 'strategy: {cfg.get('strategy')!r}' "
            f"in {config}"
        )

    from nifty_quant.calendar import TradingCalendar

    try:
        calendar = TradingCalendar.from_index_bars("NIFTY50")
    except Exception as exc:
        _fail(f"backtest could not resolve calendar: {exc}")

    from nifty_quant.research.splits import (
        HoldoutBoundaryError,
        HoldoutLock,
        default_holdout_lock_path,
    )

    holdout = HoldoutLock(path=default_holdout_lock_path())
    full_dates = calendar.session_dates()
    try:
        holdout_start, holdout_end = holdout.holdout_range(full_dates)
    except HoldoutBoundaryError as exc:
        _fail(str(exc))

    holdout_intersects = end_d >= holdout_start
    if holdout_intersects and not allow_holdout:
        _fail(
            f"refusing: backtest end date {end_d} intersects the stored holdout "
            f"window [{holdout_start}, {holdout_end}]; pass --allow-holdout for a "
            "deliberate, recorded read"
        )
    if holdout_intersects and allow_holdout:
        holdout.record_read(reason=f"backtest {start_d}..{end_d}")

    try:
        import json
        import time
        from datetime import datetime, timezone

        import numpy as np
        import yaml

        from nifty_quant import settings
        from nifty_quant.backtest.engine import BacktestConfig, run_backtest
        from nifty_quant.backtest.metrics import (
            compute_metrics,
            sharpe_standard_error,
        )
        from nifty_quant.config import RunConfig
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.data.validate import tradable_mask
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
        from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
        from nifty_quant.research.provenance import (
            FEATURE_VERSION,
            canonical_model_id,
            compute_panel_hash,
            compute_universe_hash,
            embargo_components_json,
            get_git_sha,
        )
        from nifty_quant.research.registry import TrialRecord, TrialRegistry
        from nifty_quant.strategy.registry import config_hash as strategy_config_hash
        from nifty_quant.universe.static import load_universe, survivorship_report

        strat = registry.build(cfg)
        universe = load_universe(universe_name)
        typer.echo(survivorship_report(universe, start_d, end_d).warning_line())

        spec = PanelSpec(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            symbols=universe.symbols,
            start=start_d,
            end=end_d,
        )
        panel = load_panel(spec)

        tradable_full: np.ndarray | None
        if tradable_filter:
            tradable_full = tradable_mask(panel, min_adv_inr=min_adv_inr)
            typer.echo(_tradable_mask_summary(panel, tradable_full))
        else:
            tradable_full = None

        tradable_full = _apply_pit_eligibility(
            panel,
            universe,
            tradable_full,
            min_history_sessions=min_history_sessions,
            min_adv_inr=min_adv_inr,
        )

        cost_model = NSEIntradayEquityCosts()
        # Explicit, not left to BacktestConfig's default_factory, so cost_model_id /
        # slippage_model_id / fill_model_id (specs/run_provenance.md) can be derived
        # from the actual objects used -- same values BacktestConfig would default to,
        # so this changes no backtest numbers.
        slippage_model = SqrtImpactSlippage()
        fill_model = FillModel(slippage=slippage_model)

        # ---- specs/research_contract.md: provenance + ResearchContract, computed
        # BEFORE run_backtest() so the contract (gate 1) can be passed to it. Moved
        # earlier than this command's own downstream metrics/TrialRecord code so the
        # SAME hash values feed both the contract and the record written below. ----
        run_cfg = RunConfig(
            strategy=strategy,
            params=cfg.get("params", {}),
            universe=universe_name,
            start=start_d,
            end=end_d,
        )
        manifest_fingerprint = None
        manifest_adjustments = ""
        try:
            _manifest = Manifest.load()
            manifest_fingerprint = _manifest.fingerprint
            manifest_adjustments = _manifest.adjustments
        except Exception:
            manifest_fingerprint = None
            manifest_adjustments = ""

        git_sha = get_git_sha()
        panel_hash = compute_panel_hash(panel, adjustments=manifest_adjustments)
        universe_hash = compute_universe_hash(
            name=universe_name, symbols=universe.symbols, n_sessions=panel.n_days()
        )
        cost_model_id = canonical_model_id(cost_model)
        slippage_model_id = canonical_model_id(slippage_model)
        fill_model_id = canonical_model_id(fill_model)
        # `backtest` has no walk-forward split, so there is no embargo to size --
        # all four terms are 0.0 (not a hand-chosen threshold: it is the only value
        # describing "no embargo applies to a full-sample run").
        embargo_components = embargo_components_json(
            feature_lookback=0.0, label_horizon=0.0, holding_period=0.0, execution_horizon=0.0
        )
        contract = _build_research_contract(
            panel_id=str(spec),
            panel_hash=panel_hash,
            start=start_d,
            end=end_d,
            universe_name=universe_name,
            universe_hash=universe_hash,
            cost_model_id=cost_model_id,
            slippage_model_id=slippage_model_id,
            decision_latency_bars=0,
            n_planned_trials=1,
            holdout_intent="reading_now" if (holdout_intersects and allow_holdout) else "never",
            seed=run_cfg.seed,
            feature_version=FEATURE_VERSION,
        )

        t0 = time.monotonic()

        result = run_backtest(
            strat,
            panel,
            BacktestConfig(
                capital=1e7,
                square_off_time="15:20",
                decision_latency_bars=0,
                cost_model=cost_model,
                fill_model=fill_model,
            ),
            contract=contract,
            tradable=tradable_full,
        )

        daily_gross = result.daily.gross_returns
        daily_net = result.daily.returns

        gross = compute_metrics(daily_gross)
        net = compute_metrics(daily_net)
        net_se_ann = sharpe_standard_error(daily_net, annualized=True, periods_per_year=252)
        mean_turnover = float(np.mean(result.turnover)) if result.turnover.size else 0.0

        # Representative per-leg notional: capital, not realized per-trade notional.
        modelled_bps = cost_model.round_trip_bps(notional_per_leg=1e7)
        turnover_sum = float(np.sum(result.turnover))
        realized_cost_bps = (
            (result.total_costs / (turnover_sum * 1e7) * 10_000.0) if turnover_sum > 0 else 0.0
        )

        typer.echo(
            f"gross Sharpe: {gross.sharpe:.3f}    "
            f"net Sharpe: {net.sharpe:.3f} (ann. SE={net_se_ann:.3f})    "
            f"[daily returns, n_days={len(daily_net)}]"
        )
        typer.echo(f"mean turnover: {mean_turnover:.4f}")
        typer.echo(f"modelled round-trip cost: {modelled_bps:.2f} bps")
        typer.echo(f"realized total cost of turnover: {realized_cost_bps:.2f} bps")

        # Derived statistic: breakeven cost. A failure here must not lose the primary
        # results already printed above -- degrade to n/a instead of aborting.
        try:
            breakeven = (
                breakeven_cost_bps(result.gross_returns, result.turnover)
                if result.gross_returns.size >= 2
                else float("nan")
            )
            if np.isnan(breakeven):
                typer.echo("breakeven cost: n/a (no trades / zero turnover)")
            elif breakeven == 0.0:
                typer.echo(
                    "breakeven cost: 0.0 bps -- strategy has NEGATIVE GROSS EDGE; it "
                    "does not survive any transaction cost. Modelled cost: "
                    f"{modelled_bps:.2f} bps round trip."
                )
            elif breakeven < modelled_bps:
                typer.echo(
                    f"breakeven cost: {breakeven:.2f} bps vs modelled {modelled_bps:.2f} "
                    "bps -- DOES NOT SURVIVE ITS OWN COSTS"
                )
            else:
                typer.echo(f"breakeven cost: {breakeven:.2f} bps")
        except Exception as exc:
            breakeven = float("nan")
            typer.echo(f"breakeven cost: n/a ({exc})")

        # Derived statistic: latency sensitivity. Same degrade-to-n/a treatment.
        latency_ruined: dict[int, bool] = {0: bool(result.ruined)}
        try:
            latency_sharpes = {0: float(net.sharpe)}
            for latency_bars in (1, 2):
                latency_result = run_backtest(
                    strat,
                    panel,
                    BacktestConfig(
                        capital=1e7,
                        square_off_time="15:20",
                        decision_latency_bars=latency_bars,
                        cost_model=cost_model,
                        fill_model=fill_model,
                    ),
                    contract=contract,
                    tradable=tradable_full,
                )
                latency_sharpes[latency_bars] = float(
                    compute_metrics(latency_result.daily.returns).sharpe
                )
                latency_ruined[latency_bars] = bool(latency_result.ruined)
            ruined_suffix = {
                lat: " [RUINED]" if latency_ruined[lat] else "" for lat in (0, 1, 2)
            }
            typer.echo(
                f"latency sensitivity: net Sharpe lat0={latency_sharpes[0]:.3f}"
                f"{ruined_suffix[0]} "
                f"lat1={latency_sharpes[1]:.3f}{ruined_suffix[1]} "
                f"lat2={latency_sharpes[2]:.3f}{ruined_suffix[2]}"
            )
            if latency_sharpes[0] > 0 and (
                latency_sharpes[1] <= 0 or latency_sharpes[1] < 0.5 * latency_sharpes[0]
            ):
                typer.echo(
                    "NOTE: signal dies at one minute of latency -- this looks like the "
                    "bid-ask spread, not alpha."
                )
        except Exception as exc:
            latency_sharpes = {0: float(net.sharpe)}
            latency_ruined = {0: bool(result.ruined)}
            typer.echo(f"latency sensitivity: n/a ({exc})")

        typer.echo(f"rejected_order_rate: {float(result.rejected_order_rate):.3f}")
        typer.echo(f"unfilled_notional_pct: {float(result.unfilled_notional_pct):.3f}")
        typer.echo(f"ruined: {result.ruined}    ruin_index: {result.ruin_index}")

        chash = contract.contract_hash
        trial_dir = settings.RESULTS_ROOT / "trials" / chash
        trial_dir.mkdir(parents=True, exist_ok=True)

        (trial_dir / "config.yaml").write_text(
            yaml.safe_dump(run_cfg.model_dump(mode="json")), encoding="utf-8"
        )

        record = TrialRecord(
            config_hash=chash,
            contract_hash=chash,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            strategy=strategy,
            params_json=json.dumps(cfg.get("params", {})),
            split_id="full",
            purpose="exploration",
            sharpe_gross=gross.sharpe,
            sharpe_net=net.sharpe,
            n_trades=result.n_trades,
            turnover=mean_turnover,
            breakeven_bps=breakeven,
            git_sha=git_sha,
            data_fingerprint=manifest_fingerprint,
            code_version=__version__,
            wall_s=time.monotonic() - t0,
            result_path=str(trial_dir),
            ruined=bool(result.ruined),
            ruin_index=int(result.ruin_index),
            error=None,
            seed=run_cfg.seed,
            universe_name=universe_name,
            universe_hash=universe_hash,
            panel_hash=panel_hash,
            start=start_d.isoformat(),
            end=end_d.isoformat(),
            cost_model_id=cost_model_id,
            slippage_model_id=slippage_model_id,
            fill_model_id=fill_model_id,
            embargo_components=embargo_components,
            parent_trial_id=None,
            feature_version=FEATURE_VERSION,
        )

        trial_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
        registry_write_failed = False
        try:
            trial_registry.record(record)
        except Exception as registry_exc:
            registry_write_failed = True
            _logger.warning(
                "registry write failed for trial %s: %s", chash, registry_exc
            )

        metrics: dict[str, Any] = {}
        for key, value in gross.to_dict().items():
            metrics[f"gross_{key}"] = value
        for key, value in net.to_dict().items():
            metrics[f"net_{key}"] = value
        # Add diagnostic counters (F1, F2, F4, F7).
        n_forced_liq_nontradable = result.n_forced_liquidations_against_nontradable
        metrics.update(
            {
                "breakeven_bps": float(breakeven),
                "modelled_cost_bps": float(modelled_bps),
                "realized_cost_bps": float(realized_cost_bps),
                "rejected_order_rate": float(result.rejected_order_rate),
                "unfilled_notional_pct": float(result.unfilled_notional_pct),
                "latency_sharpes": latency_sharpes,
                "n_orders_dropped_at_session_end": result.n_orders_dropped_at_session_end,
                "n_forced_liquidations_against_nontradable": n_forced_liq_nontradable,
                "min_cash_seen": float(result.min_cash_seen),
                "n_rows_negative_cash": result.n_rows_negative_cash,
                "n_stale_marks": result.n_stale_marks,
                "registry_write_failed": registry_write_failed,
                "provenance": {
                    "config_hash": chash,
                    "contract_hash": chash,
                    "git_sha": git_sha,
                    "code_version": __version__,
                    "data_fingerprint": manifest_fingerprint,
                    "panel_hash": panel_hash,
                    "universe_name": universe_name,
                    "universe_hash": universe_hash,
                    "seed": run_cfg.seed,
                    "start": start_d.isoformat(),
                    "end": end_d.isoformat(),
                    "cost_model_id": cost_model_id,
                    "slippage_model_id": slippage_model_id,
                    "fill_model_id": fill_model_id,
                    "embargo_components": embargo_components,
                    "parent_trial_id": None,
                    "feature_version": FEATURE_VERSION,
                },
            }
        )
        (trial_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        result.trades.to_parquet(trial_dir / "result.parquet")
        _write_returns_parquet(trial_dir / "returns.parquet", result.daily.dates, daily_net)
    except Exception as exc:
        try:
            import json
            from datetime import datetime, timezone

            from nifty_quant import settings
            from nifty_quant.research.provenance import get_git_sha
            from nifty_quant.research.registry import TrialRecord, TrialRegistry
            from nifty_quant.strategy.registry import config_hash as strategy_config_hash

            fallback_hash = strategy_config_hash(cfg) if cfg is not None else "unknown"
            trial_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
            trial_registry.record(
                TrialRecord(
                    config_hash=fallback_hash,
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    strategy=strategy,
                    params_json=json.dumps(cfg.get("params", {})) if cfg is not None else "{}",
                    split_id="full",
                    purpose="exploration",
                    sharpe_gross=None,
                    sharpe_net=None,
                    n_trades=None,
                    turnover=None,
                    breakeven_bps=None,
                    git_sha=get_git_sha(),
                    data_fingerprint=None,
                    code_version=__version__,
                    wall_s=None,
                    result_path=None,
                    ruined=None,
                    ruin_index=None,
                    error=str(exc),
                )
            )
        except Exception:
            pass
        _fail(f"backtest failed: {exc}")


@app.command()
def walkforward(
    strategy: str = typer.Option(..., "--strategy"),
    config: Path = typer.Option(..., "--config"),
    train_years: float = typer.Option(3.0, "--train-years"),
    test_years: float = typer.Option(1.0, "--test-years"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    universe_name: str = typer.Option("all_equity", "--universe"),
    tradable_filter: bool = typer.Option(
        True,
        "--tradable-filter/--no-tradable-filter",
        help=(
            "Gate on liquidity/circuit-lock/staleness via "
            "nifty_quant.data.validate.tradable_mask (enabled by default)."
        ),
    ),
    min_adv_inr: float = typer.Option(
        5e7,
        "--min-adv-inr",
        help=(
            "Minimum 20-session average daily traded value (INR) for the liquidity "
            "component of the tradable filter."
        ),
    ),
    min_history_sessions: int | None = typer.Option(
        None,
        "--min-history-sessions",
        help=(
            "Point-in-time eligibility gate (nifty_quant.universe.pit.compute_eligibility): "
            "a name must have this many PRIOR sessions with a present bar, plus clear "
            "--min-adv-inr on trailing 20-session ADV, to enter this session's cross-section. "
            "Disabled (no gate) unless explicitly set -- CLAUDE.md rule 8 forbids a "
            "hand-chosen default for this threshold, so it must be a deliberate choice per run."
        ),
    ),
    feature_lookback: float | None = typer.Option(
        None,
        "--feature-lookback",
        help=(
            "Longest feature window (in sessions) any signal in this run reads, used "
            "to size the walk-forward embargo (research.embargo.EmbargoComponents). "
            "Defaults to a conservative estimate derived from the strategy's own "
            "declared DataRequest.warmup_bars() when not given explicitly."
        ),
    ),
    label_horizon: float = typer.Option(
        0.0,
        "--label-horizon",
        help="Forward-return label horizon (in sessions); one term of the embargo.",
    ),
    holding_period: float = typer.Option(
        0.0,
        "--holding-period",
        help=(
            "Expected position lifetime (in sessions); one term of the embargo. For "
            "an EMA-smoothed book, pass "
            "nifty_quant.research.embargo.effective_memory_sessions(a)."
        ),
    ),
    execution_horizon: float = typer.Option(
        0.0,
        "--execution-horizon",
        help="Decision-to-fill lag (in sessions); one term of the embargo.",
    ),
    allow_holdout: bool = typer.Option(
        False,
        "--allow-holdout",
        help="Permit a deliberate, recorded read of the holdout window.",
    ),
) -> None:
    """Run an out-of-sample walk-forward evaluation."""
    from nifty_quant.calendar import DEFAULT_RESEARCH_START

    start_str = start if start is not None else DEFAULT_RESEARCH_START.isoformat()
    start_d = _parse_date(start_str, "--start")
    end_d_given = None
    if end is not None:
        end_d_given = _parse_date(end, "--end")
        if start_d > end_d_given:
            _fail(f"--start {start_d} is after --end {end_d_given}")

    _ensure_plugins_loaded()
    from nifty_quant.strategy import registry

    if strategy not in registry.available():
        _fail(f"Unknown strategy {strategy!r}. Available: {', '.join(registry.available())}")

    from nifty_quant.config import load_yaml

    try:
        cfg = load_yaml(config)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    if cfg.get("strategy") != strategy:
        _fail(
            f"--strategy {strategy!r} does not match 'strategy: {cfg.get('strategy')!r}' "
            f"in {config}"
        )

    from nifty_quant.calendar import TradingCalendar
    from nifty_quant.research.splits import WalkForwardSplitter

    try:
        calendar = TradingCalendar.from_index_bars("NIFTY50")
        end_d = end_d_given if end_d_given is not None else calendar.session_dates()[-1]
    except Exception as exc:
        _fail(f"walkforward setup failed: {exc}")

    if start_d > end_d:
        _fail(f"--start {start_d} is after --end {end_d}")

    # Resolve feature_lookback so EmbargoTooShortError is reachable, rather than the
    # unarmed default of 0.0. feature_lookback falls back to a conservative estimate
    # derived from the strategy's own DataRequest.warmup_bars() when the caller does
    # not pass --feature-lookback explicitly. The four components (feature_lookback,
    # label_horizon, holding_period, execution_horizon) are passed to splitter.split()
    # separately below rather than pre-summed here.
    import math

    from nifty_quant.settings import REGULAR_SESSION_BARS

    resolved_feature_lookback: float
    if feature_lookback is None:
        try:
            resolved_feature_lookback = float(
                math.ceil(
                    registry.build(cfg).data_request().warmup_bars()
                    / REGULAR_SESSION_BARS
                )
            )
        except Exception:
            resolved_feature_lookback = 0.0
    else:
        resolved_feature_lookback = feature_lookback

    # Pass the four embargo components separately rather than pre-summing them
    # (see specs/embargo_sizing.md AMENDMENT 2 item 4): WalkForwardSplitter.split()
    # takes four components rather than one total so EmbargoTooShortError can name
    # WHICH term forced the embargo. Pre-summing through max_lookback_days destroys
    # that diagnostic at exactly the boundary a user sees it. max_lookback_days
    # remains supported on the splitter for backward compatibility but is no longer
    # the path the CLI uses.
    try:
        trading_dates = calendar.session_dates(start_d, end_d)
        splitter = WalkForwardSplitter(train_years=train_years, test_years=test_years)
        splits = splitter.split(
            trading_dates,
            feature_lookback=resolved_feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
    except Exception as exc:
        _fail(f"walkforward split setup failed: {exc}")

    if not splits:
        _fail("no walk-forward splits fit in the given date range/train/test years")

    from nifty_quant.research.splits import (
        HoldoutBoundaryError,
        HoldoutLock,
        default_holdout_lock_path,
    )

    holdout = HoldoutLock(path=default_holdout_lock_path())
    full_dates = calendar.session_dates()
    try:
        holdout_start, holdout_end = holdout.holdout_range(full_dates)
    except HoldoutBoundaryError as exc:
        _fail(str(exc))

    intersecting_splits = [s for s in splits if s.test[1] >= holdout_start]
    if intersecting_splits and not allow_holdout:
        _fail(
            f"refusing: {len(intersecting_splits)} split(s) intersect the stored holdout "
            f"window [{holdout_start}, {holdout_end}]; pass --allow-holdout for a "
            "deliberate, recorded read"
        )

    try:
        import json
        import math
        from datetime import datetime, timezone

        import numpy as np

        from nifty_quant import settings
        from nifty_quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
        from nifty_quant.backtest.metrics import (
            compute_metrics,
            deflated_sharpe,
            effective_n_trials,
            expected_max_sharpe,
            sharpe_standard_error,
            verdict_line,
        )
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.data.validate import tradable_mask
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
        from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
        from nifty_quant.research.provenance import (
            FEATURE_VERSION,
            canonical_model_id,
            compute_panel_hash,
            compute_universe_hash,
            embargo_components_json,
            get_git_sha,
        )
        from nifty_quant.research.registry import TrialRecord, TrialRegistry
        from nifty_quant.strategy.registry import config_hash as strategy_config_hash
        from nifty_quant.universe.static import load_universe, survivorship_report

        strat = registry.build(cfg)
        wf_chash = strategy_config_hash(cfg)
        _wf_git_sha = get_git_sha()
        universe = load_universe(universe_name)
        typer.echo(survivorship_report(universe, start_d, end_d).warning_line())

        spec = PanelSpec(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            symbols=universe.symbols,
            start=start_d,
            end=end_d,
        )
        panel = load_panel(spec)

        full_tradable_mask: np.ndarray | None
        if tradable_filter:
            full_tradable_mask = tradable_mask(panel, min_adv_inr=min_adv_inr)
            typer.echo(_tradable_mask_summary(panel, full_tradable_mask))
        else:
            full_tradable_mask = None

        full_tradable_mask = _apply_pit_eligibility(
            panel,
            universe,
            full_tradable_mask,
            min_history_sessions=min_history_sessions,
            min_adv_inr=min_adv_inr,
        )

        cost_model = NSEIntradayEquityCosts()
        # Provenance-only: BacktestConfig's own default_factory already builds an
        # equivalent fill/slippage model when none is passed below, so instantiating
        # these here (to derive their canonical ids for the contract/TrialRecords)
        # changes no backtest numbers -- same rationale as the `backtest` command.
        wf_slippage_model = SqrtImpactSlippage()
        wf_fill_model = FillModel(slippage=wf_slippage_model)

        manifest_fingerprint = None
        manifest_adjustments = ""
        try:
            _manifest = Manifest.load()
            manifest_fingerprint = _manifest.fingerprint
            manifest_adjustments = _manifest.adjustments
        except Exception:
            manifest_fingerprint = None
            manifest_adjustments = ""

        # ---- specs/research_contract.md: provenance + ResearchContract, computed
        # BEFORE any run_backtest() call in this command so the contract (gate 1)
        # can be passed to every one of them (per-split AND the latency-sensitivity
        # loop below). This is also the fix for P4: these are the SAME values now
        # populated onto every TrialRecord this command writes, split and pooled. ----
        wf_panel_hash = compute_panel_hash(panel, adjustments=manifest_adjustments)
        wf_universe_hash = compute_universe_hash(
            name=universe_name, symbols=universe.symbols, n_sessions=panel.n_days()
        )
        wf_cost_model_id = canonical_model_id(cost_model)
        wf_slippage_model_id = canonical_model_id(wf_slippage_model)
        wf_fill_model_id = canonical_model_id(wf_fill_model)
        wf_embargo_components = embargo_components_json(
            feature_lookback=resolved_feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        wf_contract = _build_research_contract(
            panel_id=str(spec),
            panel_hash=wf_panel_hash,
            start=start_d,
            end=end_d,
            universe_name=universe_name,
            universe_hash=wf_universe_hash,
            cost_model_id=wf_cost_model_id,
            slippage_model_id=wf_slippage_model_id,
            decision_latency_bars=0,
            n_planned_trials=1,
            holdout_intent=(
                "reading_now" if (intersecting_splits and allow_holdout) else "never"
            ),
            seed=0,
            split_scheme="walkforward",
            purge_width_bars=0,
            embargo_width_bars=int(
                math.ceil(
                    resolved_feature_lookback + label_horizon + holding_period
                    + execution_horizon
                )
            ),
            feature_version=FEATURE_VERSION,
        )

        trial_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")

        pooled_net_list: list[np.ndarray] = []
        pooled_gross_list: list[np.ndarray] = []
        pooled_turnover_list: list[np.ndarray] = []
        total_costs_list: list[float] = []

        for split in splits:
            test_panel = panel.sub(start=split.test[0], end=split.test[1])
            test_tradable_mask = (
                tradable_mask(test_panel, min_adv_inr=min_adv_inr) if tradable_filter else None
            )
            res = run_backtest(
                strat,
                test_panel,
                BacktestConfig(capital=1e7, cost_model=cost_model),
                contract=wf_contract,
                tradable=test_tradable_mask,
            )
            split_daily_gross = res.daily.gross_returns
            split_daily_net = res.daily.returns
            split_daily_turnover = res.daily.turnover
            split_trial_dir = settings.RESULTS_ROOT / "trials" / wf_chash / split.id
            split_trial_dir.mkdir(parents=True, exist_ok=True)
            _write_returns_parquet(
                split_trial_dir / "returns.parquet", res.daily.dates, split_daily_net
            )
            gross_sharpe = float(compute_metrics(split_daily_gross).sharpe)
            net_sharpe = float(compute_metrics(split_daily_net).sharpe)
            mean_turnover = float(np.mean(res.turnover)) if res.turnover.size else 0.0
            split_breakeven = (
                float(breakeven_cost_bps(res.gross_returns, res.turnover))
                if res.gross_returns.size >= 2
                else None
            )

            typer.echo(
                f"{split.id}  train={split.train[0]}..{split.train[1]}  "
                f"test={split.test[0]}..{split.test[1]}  n_trades={res.n_trades}  "
                f"gross_sharpe={gross_sharpe:.3f}  net_sharpe={net_sharpe:.3f}  "
                f"n_days={len(split_daily_net)}"
            )

            if split.test[1] >= holdout_start:
                holdout.record_read(reason=f"walkforward split {split.id}")

            trial_registry.record(
                TrialRecord(
                    config_hash=wf_chash,
                    contract_hash=wf_contract.contract_hash,
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    strategy=strategy,
                    params_json=json.dumps(cfg.get("params", {})),
                    split_id=split.id,
                    purpose="exploration",
                    sharpe_gross=gross_sharpe,
                    sharpe_net=net_sharpe,
                    n_trades=res.n_trades,
                    turnover=mean_turnover,
                    breakeven_bps=split_breakeven,
                    git_sha=_wf_git_sha,
                    data_fingerprint=manifest_fingerprint,
                    code_version=__version__,
                    wall_s=None,
                    result_path=str(split_trial_dir),
                    ruined=bool(res.ruined),
                    ruin_index=int(res.ruin_index),
                    error=None,
                    # ---- specs/research_contract.md AMENDMENT 1 / obligation 10:
                    # P4 regression fix -- these 12 fields were previously
                    # unpopulated on every split record this command wrote. ----
                    seed=wf_contract.seed,
                    universe_name=universe_name,
                    universe_hash=wf_universe_hash,
                    panel_hash=wf_panel_hash,
                    start=split.test[0].isoformat(),
                    end=split.test[1].isoformat(),
                    cost_model_id=wf_cost_model_id,
                    slippage_model_id=wf_slippage_model_id,
                    fill_model_id=wf_fill_model_id,
                    embargo_components=wf_embargo_components,
                    # Non-null per obligation 10: every split shares one parent --
                    # the strategy+params identity (`wf_chash`) all of this run's
                    # splits (and the pooled record below) were evaluated under.
                    parent_trial_id=wf_chash,
                    feature_version=FEATURE_VERSION,
                )
            )

            pooled_net_list.append(split_daily_net)
            pooled_gross_list.append(split_daily_gross)
            pooled_turnover_list.append(split_daily_turnover)
            total_costs_list.append(res.total_costs)

        pooled_net = np.concatenate(pooled_net_list)
        pooled_gross = np.concatenate(pooled_gross_list)
        pooled_turnover = np.concatenate(pooled_turnover_list)
        pooled_total_costs = float(np.sum(total_costs_list)) if total_costs_list else 0.0

        pooled_gross_metrics = compute_metrics(pooled_gross)
        pooled_net_metrics = compute_metrics(pooled_net)
        pooled_net_se_ann = sharpe_standard_error(
            pooled_net, annualized=True, periods_per_year=252
        )
        pooled_mean_turnover = (
            float(np.mean(pooled_turnover)) if pooled_turnover.size else 0.0
        )
        modelled_bps = cost_model.round_trip_bps(notional_per_leg=1e7)
        turnover_sum = float(np.sum(pooled_turnover))
        realized_cost_bps = (
            (pooled_total_costs / (turnover_sum * 1e7) * 10_000.0)
            if turnover_sum > 0
            else 0.0
        )
        pooled_breakeven = (
            breakeven_cost_bps(pooled_gross, pooled_turnover)
            if pooled_gross.size >= 2
            else float("nan")
        )

        typer.echo(
            f"gross Sharpe: {pooled_gross_metrics.sharpe:.3f}    "
            f"net Sharpe: {pooled_net_metrics.sharpe:.3f} (ann. SE={pooled_net_se_ann:.3f})    "
            f"[daily returns, n_days={len(pooled_net)}]"
        )
        typer.echo(f"mean turnover: {pooled_mean_turnover:.4f}")
        typer.echo(f"modelled round-trip cost: {modelled_bps:.2f} bps")
        typer.echo(f"realized total cost of turnover: {realized_cost_bps:.2f} bps")
        typer.echo(f"breakeven cost: {pooled_breakeven:.2f} bps")
        if pooled_breakeven < modelled_bps:
            typer.echo(
                f"WARNING: breakeven cost ({pooled_breakeven:.2f} bps) is below the modelled "
                f"cost ({modelled_bps:.2f} bps) -- this strategy does not survive its own costs."
            )

        # Latency sensitivity is evaluated once over the whole evaluation window,
        # rather than re-sweeping every split, to bound runtime.
        latency_results: dict[int, BacktestResult] = {}
        for latency_bars in (0, 1, 2):
            latency_results[latency_bars] = run_backtest(
                strat,
                panel,
                BacktestConfig(
                    capital=1e7,
                    square_off_time="15:20",
                    decision_latency_bars=latency_bars,
                    cost_model=cost_model,
                ),
                contract=wf_contract,
                tradable=full_tradable_mask,
            )
        latency_sharpes = {
            lat: float(compute_metrics(res.daily.returns).sharpe)
            for lat, res in latency_results.items()
        }
        latency_ruined: dict[int, bool] = {
            lat: bool(res.ruined) for lat, res in latency_results.items()
        }
        full_result = latency_results[0]

        ruined_suffix = {
            lat: " [RUINED]" if latency_ruined[lat] else "" for lat in (0, 1, 2)
        }
        typer.echo(
            f"latency sensitivity: net Sharpe lat0={latency_sharpes[0]:.3f}"
            f"{ruined_suffix[0]} "
            f"lat1={latency_sharpes[1]:.3f}{ruined_suffix[1]} "
            f"lat2={latency_sharpes[2]:.3f}{ruined_suffix[2]}"
        )
        if latency_sharpes[0] > 0 and (
            latency_sharpes[1] <= 0 or latency_sharpes[1] < 0.5 * latency_sharpes[0]
        ):
            typer.echo(
                "NOTE: signal dies at one minute of latency -- this looks like the "
                "bid-ask spread, not alpha."
            )

        typer.echo(f"rejected_order_rate: {float(full_result.rejected_order_rate):.3f}")
        typer.echo(f"unfilled_notional_pct: {float(full_result.unfilled_notional_pct):.3f}")
        typer.echo(f"ruined: {full_result.ruined}    ruin_index: {full_result.ruin_index}")

        n_trials = max(trial_registry.n_trials(strategy=strategy), 1)
        var_trial_sharpes = trial_registry.var_trial_sharpes(strategy=strategy)

        # specs/pbo_dsr_wiring.md D3: the registry now stores full per-trial return
        # series (result_path/returns.parquet), so effective_n_trials CAN be computed
        # honestly from an assembled matrix -- feeding the raw count into
        # expected_max_sharpe inflates sr0 and therefore UNDER-deflates the DSR
        # (the permissive, dangerous direction). Build that matrix from every
        # exploration-purpose trial this strategy has on record (this run's own new
        # split trial included, since it was just written above).
        wf_matrix = trial_registry.build_trial_matrix(strategy=strategy, purpose="exploration")
        typer.echo(wf_matrix.explain())

        if wf_matrix.matrix.shape[1] < 2:
            # Fewer than two trial columns could be assembled: effective_n_trials
            # is only a meaningful (correlation-based) statistic across MULTIPLE
            # trials, so report it as unavailable rather than either the raw count
            # or a technically-valid-but-misleading 1.0 -- a wrong number that
            # looks right is worse than a missing one.
            n_eff = float("nan")
            sr0 = 0.0
            typer.echo(
                "NOTE: n_eff unavailable -- fewer than two trial artifacts could be "
                f"assembled into a return matrix (raw trial count={n_trials}); SR0 "
                "is reported as 0.0 rather than substituting the raw count into "
                "expected_max_sharpe."
            )
        else:
            n_eff = effective_n_trials(wf_matrix.matrix)
            if n_eff < 2.0:
                # AMENDMENT 1: expected_max_sharpe raises for n_trials < 2, and an
                # honest n_eff this low is reachable (e.g. near-duplicate trials).
                # Do NOT clamp it up to 2 -- that would silently invent multiple-
                # testing breadth that was never there. n_eff near 1 means the
                # sweep explored one idea N ways, so there is no selection effect
                # to correct for: report sr0=0.0 and say so explicitly.
                sr0 = 0.0
                typer.echo(
                    f"NOTE: honest effective_n_trials={n_eff:.4f} < 2 -- these "
                    "trials are effectively a SINGLE trial, so there is no "
                    "multiple-testing penalty to deflate against; SR0 is reported "
                    "as 0.0 instead of calling expected_max_sharpe()."
                )
            else:
                sr0 = expected_max_sharpe(n_eff, var_trial_sharpes)

        if pooled_net.size >= 2:
            pooled_net_mean = float(np.mean(pooled_net))
            pooled_net_std = float(np.std(pooled_net, ddof=1))
            sr_period = pooled_net_mean / pooled_net_std if pooled_net_std > 0 else float("nan")
        else:
            sr_period = float("nan")

        sr_se = sharpe_standard_error(pooled_net) if pooled_net.size >= 2 else float("nan")
        dsr = deflated_sharpe(pooled_net, sr0=sr0) if pooled_net.size >= 2 else float("nan")
        if math.isnan(dsr):
            # deflated_sharpe silently returns nan below T=4 (needs reliable
            # skew/kurtosis) -- without this, a nan DSR looks like a computation
            # failure rather than "too few periods" to the caller.
            typer.echo(
                f"DSR is nan: too few periods (T={pooled_net.size} < 4) for "
                "reliable skew/kurtosis; see deflated_sharpe() docs."
            )

        # PBO needs >= 2 trial return columns over the same T, not available from a
        # single strategy run; do not call pbo_cscv with a fabricated matrix.
        pbo = float("nan")
        typer.echo(
            "PBO could not be computed: single-strategy run does not supply a "
            "trial return matrix."
        )

        typer.echo(
            verdict_line(
                sharpe_net=sr_period,
                sr_se=sr_se,
                n_trials=n_trials,
                n_eff=n_eff,
                sr0=sr0,
                dsr=dsr,
                pbo=pbo,
                ruined=bool(full_result.ruined),
                n_trades=int(full_result.n_trades),
            )
        )

        typer.echo(f"holdout reads: {holdout.read_count()}")

        pooled_final_breakeven = (
            float(breakeven_cost_bps(pooled_gross, pooled_turnover))
            if pooled_gross.size >= 2
            else None
        )
        trial_registry.record(
            TrialRecord(
                config_hash=strategy_config_hash(cfg),
                contract_hash=wf_contract.contract_hash,
                ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                strategy=strategy,
                params_json=json.dumps(cfg.get("params", {})),
                split_id="pooled",
                purpose="confirmation",
                sharpe_gross=pooled_gross_metrics.sharpe,
                sharpe_net=pooled_net_metrics.sharpe,
                n_trades=None,
                turnover=pooled_mean_turnover if pooled_turnover.size else None,
                breakeven_bps=pooled_final_breakeven,
                git_sha=_wf_git_sha,
                data_fingerprint=manifest_fingerprint,
                code_version=__version__,
                wall_s=None,
                result_path=None,
                ruined=bool(full_result.ruined),
                ruin_index=int(full_result.ruin_index),
                error=None,
                # ---- specs/research_contract.md AMENDMENT 1 / obligation 10: P4
                # regression fix -- the pooled record represents the corrected,
                # validated verdict a reader would trust most, and previously
                # carried the WEAKEST provenance of any record this command wrote
                # (none of these 12 fields). ----
                seed=wf_contract.seed,
                universe_name=universe_name,
                universe_hash=wf_universe_hash,
                panel_hash=wf_panel_hash,
                start=start_d.isoformat(),
                end=end_d.isoformat(),
                cost_model_id=wf_cost_model_id,
                slippage_model_id=wf_slippage_model_id,
                fill_model_id=wf_fill_model_id,
                embargo_components=wf_embargo_components,
                # Non-null per obligation 10: the pooled record's parent is the
                # same strategy+params identity its constituent splits share.
                parent_trial_id=wf_chash,
                feature_version=FEATURE_VERSION,
            )
        )
    except Exception as exc:
        try:
            import json
            from datetime import datetime, timezone

            from nifty_quant import settings
            from nifty_quant.research.provenance import get_git_sha
            from nifty_quant.research.registry import TrialRecord, TrialRegistry
            from nifty_quant.strategy.registry import config_hash as strategy_config_hash

            trial_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
            trial_registry.record(
                TrialRecord(
                    config_hash=strategy_config_hash(cfg),
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    strategy=strategy,
                    params_json=json.dumps(cfg.get("params", {})),
                    split_id="pooled",
                    purpose="confirmation",
                    sharpe_gross=None,
                    sharpe_net=None,
                    n_trades=None,
                    turnover=None,
                    breakeven_bps=None,
                    git_sha=get_git_sha(),
                    data_fingerprint=None,
                    code_version=__version__,
                    wall_s=None,
                    result_path=None,
                    ruined=None,
                    ruin_index=None,
                    error=str(exc),
                )
            )
        except Exception:
            pass
        _fail(f"walkforward failed: {exc}")


@app.command()
def sweep(
    config: Path = typer.Option(..., "--config"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    universe_name: str = typer.Option("all_equity", "--universe"),
    allow_holdout: bool = typer.Option(
        False,
        "--allow-holdout",
        help="Allow a deliberate, recorded read of the stored holdout window.",
    ),
) -> None:
    """Run a parameter-sweep across a config file."""
    from nifty_quant.calendar import DEFAULT_RESEARCH_START, TradingCalendar

    start_str = start if start is not None else DEFAULT_RESEARCH_START.isoformat()
    start_d = _parse_date(start_str, "--start")
    end_d_given = None
    if end is not None:
        end_d_given = _parse_date(end, "--end")
        if start_d > end_d_given:
            _fail(f"--start {start_d} is after --end {end_d_given}")

    if end_d_given is None:
        try:
            calendar = TradingCalendar.from_index_bars("NIFTY50")
            end_d = calendar.session_dates()[-1]
        except Exception as exc:
            _fail(f"sweep could not resolve calendar: {exc}")
    else:
        end_d = end_d_given

    if start_d > end_d:
        _fail(f"--start {start_d} is after --end {end_d}")

    _ensure_plugins_loaded()
    from nifty_quant.research.sweep import load_sweep_yaml
    from nifty_quant.strategy import registry

    try:
        strategy_name, param_dicts = load_sweep_yaml(config)
    except Exception as exc:
        _fail(f"could not load sweep config: {exc}")

    if strategy_name not in registry.available():
        _fail(
            f"Unknown strategy {strategy_name!r} in sweep config. "
            f"Available: {', '.join(registry.available())}"
        )

    try:
        calendar = TradingCalendar.from_index_bars("NIFTY50")
    except Exception as exc:
        _fail(f"sweep could not resolve calendar for holdout check: {exc}")

    from nifty_quant.research.splits import (
        HoldoutBoundaryError,
        HoldoutLock,
        default_holdout_lock_path,
    )

    holdout = HoldoutLock(path=default_holdout_lock_path())
    full_dates = calendar.session_dates()
    try:
        holdout_start, holdout_end = holdout.holdout_range(full_dates)
    except HoldoutBoundaryError as exc:
        _fail(str(exc))

    holdout_intersects = end_d >= holdout_start
    if holdout_intersects and not allow_holdout:
        _fail(
            f"refusing: sweep end date {end_d} intersects the stored holdout "
            f"window [{holdout_start}, {holdout_end}]; pass --allow-holdout for a "
            "deliberate, recorded read"
        )
    if holdout_intersects and allow_holdout:
        holdout.record_read(reason=f"sweep {start_d}..{end_d}")

    try:
        import inspect
        import json
        from datetime import datetime, timezone

        import numpy as np
        import yaml as yaml_mod

        from nifty_quant import settings
        from nifty_quant.backtest.engine import BacktestConfig, run_backtest
        from nifty_quant.backtest.metrics import (
            compute_metrics,
            effective_n_trials,
            pbo_cscv,
            sharpe_standard_error,
        )
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
        from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage
        from nifty_quant.research.provenance import (
            FEATURE_VERSION,
            canonical_model_id,
            compute_panel_hash,
            compute_universe_hash,
            embargo_components_json,
            get_git_sha,
        )
        from nifty_quant.research.registry import TrialRecord, TrialRegistry
        from nifty_quant.strategy.registry import config_hash as strategy_config_hash
        from nifty_quant.universe.static import load_universe, survivorship_report

        universe = load_universe(universe_name)
        typer.echo(survivorship_report(universe, start_d, end_d).warning_line())

        spec = PanelSpec(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            symbols=universe.symbols,
            start=start_d,
            end=end_d,
        )
        panel = load_panel(spec)
        cost_model = NSEIntradayEquityCosts()
        slippage_model = SqrtImpactSlippage()
        fill_model = FillModel(slippage=slippage_model)

        manifest_fingerprint = None
        manifest_adjustments = ""
        try:
            _manifest = Manifest.load()
            manifest_fingerprint = _manifest.fingerprint
            manifest_adjustments = _manifest.adjustments
        except Exception:
            manifest_fingerprint = None
            manifest_adjustments = ""

        # specs/run_provenance.md AMENDMENT 1 item 8: a sweep-derived trial's
        # parent_trial_id is the hash `base_params` alone would produce -- computed
        # from the raw YAML rather than `load_sweep_yaml`'s return (which only hands
        # back the already-EXPANDED param dicts, discarding `base_params` itself).
        # KNOWN GAP, reported rather than silently fixed: no row for this hash is
        # ever written to the registry (item9's own test pins `len(sweep_trials) == 2`
        # for a 2-value sweep, i.e. no extra base row), so a sweep-derived trial's
        # parent_trial_id can be a structurally dangling reference -- correct by value,
        # unwalkable in practice. See the task's own note on this; fixing it would
        # mean either writing an un-requested extra trial or changing what "sweep
        # trial count" means, neither of which this subtask's contract permits.
        _sweep_raw_cfg = yaml_mod.safe_load(config.read_text(encoding="utf-8"))
        base_params = (
            _sweep_raw_cfg.get("base_params", {}) if isinstance(_sweep_raw_cfg, dict) else {}
        )
        base_trial_id = strategy_config_hash(
            {"strategy": strategy_name, "params": base_params}
        )

        sweep_git_sha = get_git_sha()
        sweep_panel_hash = compute_panel_hash(panel, adjustments=manifest_adjustments)
        sweep_universe_hash = compute_universe_hash(
            name=universe_name, symbols=universe.symbols, n_sessions=panel.n_days()
        )
        sweep_cost_model_id = canonical_model_id(cost_model)
        sweep_slippage_model_id = canonical_model_id(slippage_model)
        sweep_fill_model_id = canonical_model_id(fill_model)
        # No walk-forward split in a sweep either -- see the identical rationale on
        # the `backtest` command above.
        sweep_embargo_components = embargo_components_json(
            feature_lookback=0.0, label_horizon=0.0, holding_period=0.0, execution_horizon=0.0
        )

        # ---- specs/research_contract.md: n_planned_trials is declared BEFORE the
        # sweep runs, from the already-expanded param_dicts (obligation 8). One
        # contract governs the whole sweep (data/costs/splits are fixed across
        # trials; only each trial's own strategy params vary, which is what
        # config_hash below still disambiguates per trial-directory). ----
        sweep_contract = _build_research_contract(
            panel_id=str(spec),
            panel_hash=sweep_panel_hash,
            start=start_d,
            end=end_d,
            universe_name=universe_name,
            universe_hash=sweep_universe_hash,
            cost_model_id=sweep_cost_model_id,
            slippage_model_id=sweep_slippage_model_id,
            decision_latency_bars=0,
            n_planned_trials=len(param_dicts),
            holdout_intent="reading_now" if (holdout_intersects and allow_holdout) else "never",
            seed=0,
            feature_version=FEATURE_VERSION,
        )

        registry_db = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
        n_ok = 0
        n_failed = 0
        # specs/pbo_dsr_wiring.md D2/D3: every config_hash this sweep run itself
        # attempts (whether it ultimately succeeds or fails), so the matrix
        # assembled below is built from exactly "the trials it just wrote" -- a
        # failed trial is still recorded (with result_path=None) and
        # build_trial_matrix correctly reports it as dropped rather than pretending
        # it does not exist.
        sweep_chashes: list[str] = []

        for i, params in enumerate(param_dicts):
            cfg = {"strategy": strategy_name, "params": params}
            chash = strategy_config_hash(cfg)
            sweep_chashes.append(chash)
            sweep_contract.check_trial_count(i + 1)
            try:
                strat = registry.build(cfg)
                res = run_backtest(
                    strat,
                    panel,
                    BacktestConfig(capital=1e7, cost_model=cost_model, fill_model=fill_model),
                    contract=sweep_contract,
                )
                sweep_daily_net = res.daily.returns
                trial_dir = settings.RESULTS_ROOT / "trials" / chash
                trial_dir.mkdir(parents=True, exist_ok=True)
                _write_returns_parquet(
                    trial_dir / "returns.parquet", res.daily.dates, sweep_daily_net
                )
                sweep_daily_gross = res.daily.gross_returns

                net = compute_metrics(sweep_daily_net)
                gross = compute_metrics(sweep_daily_gross)
                net_se_ann = sharpe_standard_error(
                    sweep_daily_net, annualized=True, periods_per_year=252
                )
                mean_turnover = float(np.mean(res.turnover)) if res.turnover.size else 0.0
                breakeven = (
                    float(breakeven_cost_bps(res.gross_returns, res.turnover))
                    if res.gross_returns.size >= 2
                    else None
                )

                typer.echo(
                    f"[{i + 1}/{len(param_dicts)}] {params} -> "
                    f"net_sharpe={net.sharpe:.3f} (ann. SE={net_se_ann:.3f})  "
                    f"n_days={len(sweep_daily_net)}"
                )

                registry_db.record(
                    TrialRecord(
                        config_hash=chash,
                        contract_hash=sweep_contract.contract_hash,
                        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        strategy=strategy_name,
                        params_json=json.dumps(params),
                        split_id="full",
                        purpose="exploration",
                        sharpe_gross=gross.sharpe,
                        sharpe_net=net.sharpe,
                        n_trades=res.n_trades,
                        turnover=mean_turnover,
                        breakeven_bps=breakeven,
                        git_sha=sweep_git_sha,
                        data_fingerprint=manifest_fingerprint,
                        code_version=__version__,
                        wall_s=None,
                        result_path=str(trial_dir),
                        ruined=bool(res.ruined),
                        ruin_index=int(res.ruin_index),
                        error=None,
                        seed=sweep_contract.seed,
                        universe_name=universe_name,
                        universe_hash=sweep_universe_hash,
                        panel_hash=sweep_panel_hash,
                        start=start_d.isoformat(),
                        end=end_d.isoformat(),
                        cost_model_id=sweep_cost_model_id,
                        slippage_model_id=sweep_slippage_model_id,
                        fill_model_id=sweep_fill_model_id,
                        embargo_components=sweep_embargo_components,
                        parent_trial_id=base_trial_id,
                        feature_version=FEATURE_VERSION,
                    )
                )
                n_ok += 1
            except Exception as exc:
                try:
                    registry_db.record(
                        TrialRecord(
                            config_hash=chash,
                            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            strategy=strategy_name,
                            params_json=json.dumps(params),
                            split_id="full",
                            purpose="exploration",
                            sharpe_gross=None,
                            sharpe_net=None,
                            n_trades=None,
                            turnover=None,
                            breakeven_bps=None,
                            git_sha=sweep_git_sha,
                            data_fingerprint=None,
                            code_version=__version__,
                            wall_s=None,
                            result_path=None,
                            ruined=None,
                            ruin_index=None,
                            error=str(exc),
                            parent_trial_id=base_trial_id,
                        )
                    )
                except Exception:
                    pass
                n_failed += 1
                typer.echo(f"[{i + 1}/{len(param_dicts)}] {params} -> FAILED: {exc}")

        typer.echo(f"sweep complete: {n_ok} ok, {n_failed} failed, {len(param_dicts)} total")

        # specs/pbo_dsr_wiring.md D2/D3: a sweep IS the multi-trial object PBO and
        # effective_n_trials were designed for -- assemble a matrix from exactly the
        # trials this run just wrote and report both, rather than leaving PBO
        # permanently nan and n_eff a raw, correlation-blind count.
        sweep_matrix = registry_db.build_trial_matrix(trial_ids=sweep_chashes)
        typer.echo(sweep_matrix.explain())

        n_splits_default = inspect.signature(pbo_cscv).parameters["n_splits"].default

        if sweep_matrix.matrix.shape[1] == 0:
            typer.echo(
                "n_eff: unavailable -- no trial return matrix could be assembled "
                f"from this sweep's {len(sweep_chashes)} trial(s); see explain() "
                "above. Not substituting the raw trial count."
            )
            typer.echo(
                "PBO: unavailable -- no trial return matrix could be assembled "
                "for this sweep."
            )
        else:
            sweep_n_eff = effective_n_trials(sweep_matrix.matrix)
            typer.echo(f"n_eff={sweep_n_eff:.3f}")

            if sweep_matrix.matrix.shape[1] < 2:
                typer.echo(
                    "PBO: unavailable -- pbo_cscv needs at least two trial "
                    f"columns; this sweep assembled {sweep_matrix.matrix.shape[1]}."
                )
            else:
                t_periods = sweep_matrix.matrix.shape[0]
                if t_periods < n_splits_default:
                    typer.echo(
                        f"PBO refused: T={t_periods} < n_splits={n_splits_default} "
                        "(need at least n_splits periods); cannot compute PBO for "
                        "this sweep. Silently lowering n_splits would change the "
                        "statistic being computed without saying so -- widen the "
                        "date range or n_splits instead."
                    )
                else:
                    sweep_pbo = pbo_cscv(sweep_matrix.matrix, n_splits=n_splits_default)
                    typer.echo(f"PBO={sweep_pbo:.4f}")
    except Exception as exc:
        _fail(f"sweep failed: {exc}")


@app.command()
def tilt(
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    entry: str = typer.Option("09:16", "--entry"),
    exit_time: str = typer.Option("15:20", "--exit"),
    capital: float = typer.Option(1_000_000.0, "--capital"),
    tilt_type: str = typer.Option("mild", "--tilt"),
    smoothing: float = typer.Option(0.10, "--smoothing"),
    rebalance_every: int = typer.Option(1, "--rebalance-every"),
    universe_name: str = typer.Option("all_equity", "--universe"),
    continuous_only: bool = typer.Option(
        False, "--continuous-only", help="Restrict to symbols with coverage"
    ),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run a parameterised tilt backtest."""
    start_d = _parse_date(start, "--start")
    end_d = _parse_date(end, "--end")
    if start_d > end_d:
        _fail(f"--start {start_d} is after --end {end_d}")

    if tilt_type not in ("mild", "aggressive"):
        _fail(
            f"Invalid --tilt value {tilt_type!r}. "
            f"Valid options: mild, aggressive"
        )

    try:
        from nifty_quant import settings
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.research.provenance import (
            FEATURE_VERSION,
            compute_panel_hash,
            compute_universe_hash,
        )
        from nifty_quant.research.registry import TrialRegistry
        from nifty_quant.research.tilt import TiltConfig, run_tilt
        from nifty_quant.universe.static import load_universe

        universe = load_universe(universe_name)
        typer.echo(f"Universe: {universe_name} ({len(universe.symbols)} symbols)")

        spec = PanelSpec(
            freq="1",
            fields=("open", "high", "low", "close", "volume"),
            symbols=universe.symbols,
            start=start_d,
            end=end_d,
        )
        panel = load_panel(spec)

        config = TiltConfig(
            start=start_d,
            end=end_d,
            entry_hhmm=entry,
            exit_hhmm=exit_time,
            capital=capital,
            tilt=tilt_type,
            smoothing=smoothing,
            rebalance_every=rebalance_every,
            universe=universe_name,
            continuous_only=continuous_only,
            seed=seed,
        )

        # ---- specs/research_contract.md, Enforcement gate 2 / P1: tilt is the only
        # construction in this repo that has beaten the index net of costs, and
        # previously bypassed the research spine entirely (zero references to
        # TrialRegistry/TrialRecord/run_backtest). It now requires a contract and
        # writes a TrialRecord, same as `backtest`/`walkforward`/`sweep`. `nq tilt`
        # has no --allow-holdout flag (run_tilt's own internal HoldoutLock check is
        # unconditional, with no bypass), so holdout_intent is always "never" here. ----
        manifest_adjustments = ""
        try:
            manifest_adjustments = Manifest.load().adjustments
        except Exception:
            manifest_adjustments = ""
        tilt_panel_hash = compute_panel_hash(panel, adjustments=manifest_adjustments)
        tilt_universe_hash = compute_universe_hash(
            name=universe_name, symbols=universe.symbols, n_sessions=panel.n_days()
        )
        contract = _build_research_contract(
            panel_id=str(spec),
            panel_hash=tilt_panel_hash,
            start=start_d,
            end=end_d,
            universe_name=universe_name,
            universe_hash=tilt_universe_hash,
            cost_model_id="nse_intraday_default",
            slippage_model_id="none",
            decision_latency_bars=0,
            n_planned_trials=1,
            holdout_intent="never",
            seed=seed,
            feature_version=FEATURE_VERSION,
        )
        tilt_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")

        result = run_tilt(panel, config, contract=contract, registry=tilt_registry)
        typer.echo("")
        typer.echo(result.to_table())
        typer.echo("")

        if result.warnings:
            typer.echo("Warnings:")
            for w in result.warnings:
                typer.echo(f"  {w}")
            typer.echo("")

    except ValueError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"Error running tilt: {exc}")


cache_app = typer.Typer()


@cache_app.command("info")
def cache_info() -> None:
    """Print manifest cache key and panel cache directories."""
    from nifty_quant import settings
    from nifty_quant.data.manifest import Manifest

    try:
        manifest = Manifest.load()
    except FileNotFoundError as exc:
        _fail(f"cache info: {exc}")

    typer.echo(f"cache_key: {manifest.cache_key()}")
    typer.echo(
        f"CACHE_ROOT: {settings.CACHE_ROOT}  "
        f"[{'exists' if settings.CACHE_ROOT.exists() else 'missing'}]"
    )

    panel_cache_root = settings.CACHE_ROOT / "panel" / f"v{settings.PANEL_VERSION}"
    if panel_cache_root.exists():
        try:
            for directory in panel_cache_root.iterdir():
                if not directory.is_dir():
                    continue
                total_bytes = sum(
                    f.stat().st_size for f in directory.rglob("*") if f.is_file()
                )
                current = "  [current]" if directory.name == manifest.cache_key() else ""
                typer.echo(f"  {directory.name}{current}: {total_bytes / 1e6:.1f} MB")
        except Exception as exc:
            _fail(f"cache info: {exc}")


@cache_app.command("gc")
def cache_gc(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview orphaned cache dirs"),
) -> None:
    """Remove orphaned panel cache directories."""
    from nifty_quant.data.panel_builder import gc_orphans

    try:
        orphans = gc_orphans(dry_run=dry_run)
    except Exception as exc:
        _fail(f"cache gc failed: {exc}")

    if not orphans:
        typer.echo("no orphaned cache dirs")
        return

    prefix = "[dry-run] would remove: " if dry_run else "removed: "
    for path in orphans:
        typer.echo(prefix + str(path))
    typer.echo(f"{'would remove' if dry_run else 'removed'} {len(orphans)} dir(s)")


app.add_typer(cache_app, name="cache")


if __name__ == "__main__":
    app()
