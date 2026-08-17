from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer

from nifty_quant import __version__

if TYPE_CHECKING:  # pragma: no cover - False at runtime by language definition
    # `typing.TYPE_CHECKING` is a hard-coded `False` in every CPython execution; only static
    # checkers (mypy/pyright) treat it as True, and they parse rather than execute. Combined
    # with `from __future__ import annotations` above, the annotations referencing `np`/`Panel`
    # are never evaluated at runtime either, so no call path can reach this block.
    import numpy as np

    from nifty_quant.data.panel import Panel

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


def _decision_and_final_rows(
    panel: "Panel",
    decision_times: tuple[str, ...] | None,
) -> "np.ndarray":
    """Reconstruct the strategy's decision rows and unconditional final row.

    This helper is shared by `_daily_returns`, `_daily_turnover`, and
    `_daily_return_ts` so none of them can reconstruct a different row set for
    the same backtest.
    """
    import numpy as np

    n_rows = panel.ts.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    if decision_times is None:
        decision_rows = np.arange(1, n_rows, dtype=np.int64)
    else:
        parts = [panel.rows_at_time(t) for t in decision_times]
        decision_rows = (
            np.unique(np.concatenate(parts)).astype(np.int64, copy=False)
            if parts
            else np.empty(0, dtype=np.int64)
        )
        decision_rows = decision_rows[decision_rows != 0]

    return np.concatenate((decision_rows, np.array([n_rows - 1], dtype=np.int64)))


def _daily_returns(
    returns: "np.ndarray",
    panel: "Panel",
    decision_times: tuple[str, ...] | None,
) -> "np.ndarray":
    """Aggregate decision-row returns into one compounded return per trading day.

    ``BacktestResult`` returns one entry per decision row plus an unconditional final
    row, not one entry per trading day.  Computing metrics directly on those rows with
    the default ``periods_per_year=252`` would annualize by the wrong factor for
    bar-frequency and checkpoint strategies.  This helper reconstructs the engine's
    decision-row selection from ``Strategy.data_request().decision_times`` and
    ``Panel.rows_at_time``, appends the unconditional final row, maps rows to days using
    ``Panel.day_offsets``, and compounds each day's returns with
    ``aggregate_returns_by_group``.

    Parameters
    ----------
    returns:
        1-D array of raw per-decision-row returns from a ``BacktestResult``.
    panel:
        Panel used for the backtest; its ``ts``, ``day_offsets``, and ``rows_at_time``
        are enough to reconstruct the engine's selection.
    decision_times:
        The strategy's decision times (``None`` means every row).

    Returns
    -------
    np.ndarray
        1-D float64 array of daily compounded returns. Empty input returns an empty
        array.

    Raises
    ------
    ValueError
        If reconstructed row count does not match ``returns`` exactly.
    """
    import numpy as np

    from nifty_quant.backtest.metrics import aggregate_returns_by_group

    if returns.size == 0:
        return np.empty(0, dtype=np.float64)

    n_rows = panel.ts.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.float64)

    row_indices = _decision_and_final_rows(panel, decision_times)

    if row_indices.size != returns.size:
        raise ValueError(
            "Daily-return reconstruction length mismatch: reconstructed "
            f"{row_indices.size} decision/final rows but backtest returns has "
            f"{returns.size} entries."
        )

    day_indices = np.searchsorted(panel.day_offsets, row_indices, side="right") - 1
    return aggregate_returns_by_group(returns, day_indices)


def _daily_turnover(
    turnover: "np.ndarray",
    panel: "Panel",
    decision_times: tuple[str, ...] | None,
) -> "np.ndarray":
    """Aggregate decision-row turnover into one additive total per trading day.

    Returns compound within a session, whereas turnover sums within a session because
    turnover is the notional churned and is therefore an additive quantity. Compounding
    turnover would be meaningless. Use this helper instead of raw per-decision-row
    turnover whenever pooling or concatenating it with day-aggregated returns, or the
    two series will end up on different bases.
    """
    import numpy as np

    if turnover.size == 0:
        return np.empty(0, dtype=np.float64)

    n_rows = panel.ts.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.float64)

    row_indices = _decision_and_final_rows(panel, decision_times)

    if row_indices.size != turnover.size:
        raise ValueError(
            "Daily-turnover reconstruction length mismatch: reconstructed "
            f"{row_indices.size} decision/final rows but turnover has "
            f"{turnover.size} entries."
        )

    day_indices = np.searchsorted(panel.day_offsets, row_indices, side="right") - 1
    turnover = np.asarray(turnover, dtype=np.float64).reshape(-1)
    starts = np.r_[0, np.flatnonzero(np.diff(day_indices) != 0) + 1]
    return np.add.reduceat(turnover, starts)


def _daily_return_ts(
    panel: "Panel",
    decision_times: tuple[str, ...] | None,
    n_days: int,
) -> "np.ndarray":
    """Return UTC-midnight timestamps for daily-return values.

    The int64 result has one epoch-second timestamp per requested day, ordered by the
    distinct reconstructed day groups in the same ascending order as `_daily_returns`.
    If ``n_days`` does not match those groups, a best-effort fallback uses trailing
    panel days and repeats day zero as needed; this is intended for callers using
    externally supplied or stubbed daily-return lengths and cannot recover an arbitrary
    true row-to-day mapping.
    """
    from datetime import datetime, timezone

    import numpy as np

    if n_days == 0:
        return np.empty(0, dtype=np.int64)

    if panel.ts.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    row_indices = _decision_and_final_rows(panel, decision_times)
    if row_indices.size == 0:
        unique_days = np.empty(0, dtype=np.int64)
    else:
        day_indices = np.searchsorted(panel.day_offsets, row_indices, side="right") - 1
        unique_days = np.unique(day_indices)

    if unique_days.size != n_days:
        # Unreachable when n_days comes from the real _daily_returns output because
        # step 4 derives that length from these same reconstructed day groups.
        unique_days = np.arange(
            max(0, len(panel.dates) - n_days),
            len(panel.dates),
            dtype=np.int64,
        )
        if unique_days.size < n_days:
            padding = n_days - unique_days.size
            if len(panel.dates) == 0:
                unique_days = np.zeros(n_days, dtype=np.int64)
            else:
                unique_days = np.concatenate(
                    (np.zeros(padding, dtype=np.int64), unique_days)
                )

    if len(panel.dates) == 0:
        return np.zeros(n_days, dtype=np.int64)

    session_dates = panel.dates[unique_days]
    return np.asarray(
        [
            int(
                datetime.combine(
                    session_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ).timestamp()
            )
            for session_date in session_dates
        ],
        dtype=np.int64,
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
        from nifty_quant.config import config_hash as run_config_hash
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.data.validate import tradable_mask
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
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

        cost_model = NSEIntradayEquityCosts()
        t0 = time.monotonic()

        result = run_backtest(
            strat,
            panel,
            BacktestConfig(
                capital=1e7,
                square_off_time="15:20",
                decision_latency_bars=0,
                cost_model=cost_model,
            ),
            tradable=tradable_full,
        )

        decision_times = strat.data_request().decision_times
        daily_gross = _daily_returns(result.gross_returns, panel, decision_times)
        daily_net = _daily_returns(result.returns, panel, decision_times)

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
                    ),
                    tradable=tradable_full,
                )
                latency_sharpes[latency_bars] = float(
                    compute_metrics(
                        _daily_returns(latency_result.returns, panel, decision_times)
                    ).sharpe
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

        run_cfg = RunConfig(
            strategy=strategy,
            params=cfg.get("params", {}),
            universe=universe_name,
            start=start_d,
            end=end_d,
        )
        chash = run_config_hash(run_cfg)
        trial_dir = settings.RESULTS_ROOT / "trials" / chash
        trial_dir.mkdir(parents=True, exist_ok=True)

        (trial_dir / "config.yaml").write_text(
            yaml.safe_dump(run_cfg.model_dump(mode="json")), encoding="utf-8"
        )

        metrics: dict[str, float] = {}
        for key, value in gross.to_dict().items():
            metrics[f"gross_{key}"] = value
        for key, value in net.to_dict().items():
            metrics[f"net_{key}"] = value
        metrics.update(
            {
                "breakeven_bps": float(breakeven),
                "modelled_cost_bps": float(modelled_bps),
                "realized_cost_bps": float(realized_cost_bps),
                "rejected_order_rate": float(result.rejected_order_rate),
                "unfilled_notional_pct": float(result.unfilled_notional_pct),
                "latency_sharpes": latency_sharpes,
            }
        )
        (trial_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        result.trades.to_parquet(trial_dir / "result.parquet")
        returns_ts = _daily_return_ts(panel, decision_times, daily_net.size)
        _write_returns_parquet(trial_dir / "returns.parquet", returns_ts, daily_net)

        manifest_fingerprint = None
        try:
            manifest_fingerprint = Manifest.load().fingerprint
        except Exception:
            manifest_fingerprint = None

        trial_registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
        trial_registry.record(
            TrialRecord(
                config_hash=chash,
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
                git_sha=None,
                data_fingerprint=manifest_fingerprint,
                code_version=__version__,
                wall_s=time.monotonic() - t0,
                result_path=str(trial_dir),
                ruined=bool(result.ruined),
                ruin_index=int(result.ruin_index),
                error=None,
            )
        )
    except Exception as exc:
        try:
            import json
            from datetime import datetime, timezone

            from nifty_quant import settings
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
                    git_sha=None,
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

    try:
        trading_dates = calendar.session_dates(start_d, end_d)
        splitter = WalkForwardSplitter(train_years=train_years, test_years=test_years)
        splits = splitter.split(trading_dates)
    except Exception as exc:
        _fail(f"walkforward split setup failed: {exc}")

    if not splits:
        _fail("no walk-forward splits fit in the given date range/train/test years")

    try:
        import json
        from datetime import datetime, timezone

        import numpy as np

        from nifty_quant import settings
        from nifty_quant.backtest.engine import BacktestConfig, run_backtest
        from nifty_quant.backtest.metrics import (
            compute_metrics,
            deflated_sharpe,
            expected_max_sharpe,
            sharpe_standard_error,
            verdict_line,
        )
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.data.validate import tradable_mask
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
        from nifty_quant.research.registry import TrialRecord, TrialRegistry
        from nifty_quant.research.splits import HoldoutLock
        from nifty_quant.strategy.registry import config_hash as strategy_config_hash
        from nifty_quant.universe.static import load_universe, survivorship_report

        strat = registry.build(cfg)
        wf_chash = strategy_config_hash(cfg)
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

        cost_model = NSEIntradayEquityCosts()

        holdout = HoldoutLock(path=settings.RESULTS_ROOT / "holdout_lock.json")
        holdout_start, _ = holdout.holdout_range(trading_dates)

        manifest_fingerprint = None
        try:
            manifest_fingerprint = Manifest.load().fingerprint
        except Exception:
            manifest_fingerprint = None

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
                tradable=test_tradable_mask,
            )
            decision_times = strat.data_request().decision_times
            split_daily_gross = _daily_returns(res.gross_returns, test_panel, decision_times)
            split_daily_net = _daily_returns(res.returns, test_panel, decision_times)
            split_daily_turnover = _daily_turnover(res.turnover, test_panel, decision_times)
            split_trial_dir = settings.RESULTS_ROOT / "trials" / wf_chash / split.id
            split_trial_dir.mkdir(parents=True, exist_ok=True)
            split_returns_ts = _daily_return_ts(
                test_panel, decision_times, split_daily_net.size
            )
            _write_returns_parquet(
                split_trial_dir / "returns.parquet", split_returns_ts, split_daily_net
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
                    git_sha=None,
                    data_fingerprint=manifest_fingerprint,
                    code_version=__version__,
                    wall_s=None,
                    result_path=str(split_trial_dir),
                    ruined=bool(res.ruined),
                    ruin_index=int(res.ruin_index),
                    error=None,
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
        latency_results: dict[int, object] = {}
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
                tradable=full_tradable_mask,
            )
        full_decision_times = strat.data_request().decision_times
        latency_sharpes = {
            lat: float(
                compute_metrics(
                    _daily_returns(res.returns, panel, full_decision_times)
                ).sharpe
            )
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
        sr0 = expected_max_sharpe(n_trials, var_trial_sharpes) if n_trials >= 2 else 0.0

        # This repo's TrialRegistry only stores summary Sharpes, not full per-trial
        # return series, so effective_n_trials cannot be computed honestly here.
        n_eff = float(n_trials)

        if pooled_net.size >= 2:
            pooled_net_mean = float(np.mean(pooled_net))
            pooled_net_std = float(np.std(pooled_net, ddof=1))
            sr_period = pooled_net_mean / pooled_net_std if pooled_net_std > 0 else float("nan")
        else:
            sr_period = float("nan")

        sr_se = sharpe_standard_error(pooled_net) if pooled_net.size >= 2 else float("nan")
        dsr = deflated_sharpe(pooled_net, sr0=sr0) if pooled_net.size >= 2 else float("nan")

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
                git_sha=None,
                data_fingerprint=manifest_fingerprint,
                code_version=__version__,
                wall_s=None,
                result_path=None,
                ruined=bool(full_result.ruined),
                ruin_index=int(full_result.ruin_index),
                error=None,
            )
        )
    except Exception as exc:
        try:
            import json
            from datetime import datetime, timezone

            from nifty_quant import settings
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
                    git_sha=None,
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
        import json
        from datetime import datetime, timezone

        import numpy as np

        from nifty_quant import settings
        from nifty_quant.backtest.engine import BacktestConfig, run_backtest
        from nifty_quant.backtest.metrics import (
            compute_metrics,
            sharpe_standard_error,
        )
        from nifty_quant.data.manifest import Manifest
        from nifty_quant.data.panel import PanelSpec, load_panel
        from nifty_quant.execution.costs import NSEIntradayEquityCosts, breakeven_cost_bps
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

        manifest_fingerprint = None
        try:
            manifest_fingerprint = Manifest.load().fingerprint
        except Exception:
            manifest_fingerprint = None

        registry_db = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
        n_ok = 0
        n_failed = 0

        for i, params in enumerate(param_dicts):
            cfg = {"strategy": strategy_name, "params": params}
            chash = strategy_config_hash(cfg)
            try:
                strat = registry.build(cfg)
                res = run_backtest(
                    strat,
                    panel,
                    BacktestConfig(capital=1e7, cost_model=cost_model),
                )
                sweep_decision_times = strat.data_request().decision_times
                sweep_daily_net = _daily_returns(res.returns, panel, sweep_decision_times)
                trial_dir = settings.RESULTS_ROOT / "trials" / chash
                trial_dir.mkdir(parents=True, exist_ok=True)
                sweep_returns_ts = _daily_return_ts(
                    panel, sweep_decision_times, sweep_daily_net.size
                )
                _write_returns_parquet(
                    trial_dir / "returns.parquet", sweep_returns_ts, sweep_daily_net
                )
                sweep_daily_gross = _daily_returns(res.gross_returns, panel, sweep_decision_times)

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
                        git_sha=None,
                        data_fingerprint=manifest_fingerprint,
                        code_version=__version__,
                        wall_s=None,
                        result_path=str(trial_dir),
                        ruined=bool(res.ruined),
                        ruin_index=int(res.ruin_index),
                        error=None,
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
                            git_sha=None,
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
                n_failed += 1
                typer.echo(f"[{i + 1}/{len(param_dicts)}] {params} -> FAILED: {exc}")

        typer.echo(f"sweep complete: {n_ok} ok, {n_failed} failed, {len(param_dicts)} total")
    except Exception as exc:
        _fail(f"sweep failed: {exc}")


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
