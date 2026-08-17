from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import pytest

import nifty_quant.strategy.plugins  # noqa: F401
from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import sharpe_ratio
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import ZeroCost
from nifty_quant.execution.fills import FillModel, ZeroSlippage
from nifty_quant.features.persistence import fbm
from nifty_quant.strategy import registry

from .test_causality import (
    _SYMBOLS,
    _build_panel_from_close_and_volume,
    _load_strategy,
    _muhurat_times,
    _regular_times,
    _short_dr_times,
)

_PriceGenerator = Callable[[int, int, int], np.ndarray]

N_DAYS: int = 20
_PANEL_BASE_DATE: dt.date = dt.date(2024, 1, 2)


def _null_session_specs(
    n_days: int, base_date: dt.date
) -> list[tuple[dt.date, list[str]]]:
    if n_days < 2:
        raise ValueError("n_days must be at least 2 to include both irregular sessions")
    specs: list[tuple[dt.date, list[str]]] = []
    for i in range(n_days):
        session_date = base_date + dt.timedelta(days=i)
        if i == 0:
            times = _short_dr_times()
        elif i == 1:
            times = _muhurat_times()
        else:
            times = _regular_times()
        specs.append((session_date, times))
    return specs


def _gbm_close(n_rows: int, n_symbols: int, seed: int) -> np.ndarray:
    """Driftless GBM: zero expected return per bar, so a trend-follower earns nothing
    pre-cost in expectation. A nonzero per-bar drift compounds enormously over thousands
    of bars (measured: loc=0.0002 over ~6915 bars gave a cumulative log-drift ~8x the
    total noise std, i.e. an obvious deterministic uptrend, not noise) and would make any
    trend-following strategy show a real (non-spurious) positive gross Sharpe -- that is
    not a lookahead leak, it is genuine information the null must not contain."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0, scale=0.002, size=(n_rows, n_symbols))
    log_price = np.cumsum(log_returns, axis=0)
    return 100.0 * np.exp(log_price)


def _fbm_close(n_rows: int, n_symbols: int, seed: int) -> np.ndarray:
    """Driftless fBm(H=0.5): fbm()'s own increments already have zero expected value by
    construction, so no separate drift term is added (see _gbm_close's docstring for why
    a nonzero drift would corrupt the "no edge" null)."""
    columns: list[np.ndarray] = []
    for col in range(n_symbols):
        col_seed = seed * 1000 + col
        raw_path = fbm(n_rows, H=0.5, seed=col_seed)
        scaled_path = 0.002 * raw_path
        columns.append(100.0 * np.exp(scaled_path))
    return np.column_stack(columns)


def _null_volume(
    n_rows: int, n_symbols: int, rng: np.random.Generator
) -> np.ndarray:
    return np.exp(rng.normal(loc=11.0, scale=0.3, size=(n_rows, n_symbols)))


def _build_null_panel(
    price_fn: _PriceGenerator, n_days: int, seed: int
) -> Panel:
    specs = _null_session_specs(n_days, _PANEL_BASE_DATE)
    n_rows = sum(len(times) for _, times in specs)
    close = price_fn(n_rows, len(_SYMBOLS), seed)
    volume = _null_volume(n_rows, len(_SYMBOLS), np.random.default_rng(seed))
    return _build_panel_from_close_and_volume(
        specs, _SYMBOLS, close, volume, np.random.default_rng(seed + 1)
    )


_SEEDS_PER_GENERATOR: int = 8

# A single run's annualized daily Sharpe estimated from ~18 daily observations has a
# standard error under the null of roughly sqrt(252 / 18) ~= 3.7 Sharpe units. Averaging
# over _SEEDS_PER_GENERATOR seeds per generator (16 runs per strategy) shrinks that to
# roughly 3.7 / sqrt(16) ~= 0.93. TOLERANCE below is comfortably above that, while still
# far below the >5.0 Sharpe gap test_leak_canary.py uses to flag a real leak.
# The first 8 seeds are used for GBM, the next 8 for fBm; disjoint sets prevent a
# cancellation across generators from masking a generator-specific leak.
SEEDS: tuple[int, ...] = (
    101, 102, 103, 104, 105, 106, 107, 108,
    201, 202, 203, 204, 205, 206, 207, 208,
)
_GBM_SEEDS: tuple[int, ...] = SEEDS[:_SEEDS_PER_GENERATOR]
_FBM_SEEDS: tuple[int, ...] = SEEDS[_SEEDS_PER_GENERATOR:]

TOLERANCE: float = 1.5

_GENERATORS: tuple[tuple[str, _PriceGenerator, tuple[int, ...]], ...] = (
    ("gbm", _gbm_close, _GBM_SEEDS),
    ("fbm", _fbm_close, _FBM_SEEDS),
)


# Null-test failure taxonomy:
# 1. ruined_runs assertion: "capital destroyed" -> forced liquidation / non-finite equity.
# 2. raw-Sharpe assertions: "probable lookahead leak" -> edge on pure noise with ZERO
#    cost and ZERO slippage. `raw_sharpes` is the ONLY frictionless alpha measure.
#    `.returns` under the default config is net of BOTH fees and market impact, and
#    `.gross_returns` under that same config is net of impact but NOT fees (a misleading
#    name for anyone assuming "gross" means frictionless -- this exact confusion caused
#    two earlier incorrect versions of this file).
# 3. net-Sharpe assertion: "broken cost/fill model" -> net expectancy positive with real
#    costs is impossible on pure noise. A large negative net Sharpe from churn is
#    EXPECTED and must never fail a magnitude bound.
@pytest.mark.slow
@pytest.mark.parametrize("name", registry.available())
def test_net_sharpe_straddles_zero_on_pure_noise(name: str) -> None:
    strategy = _load_strategy(name)

    raw_sharpes: list[float] = []
    with_slippage_sharpes: list[float] = []
    net_sharpes: list[float] = []
    ruined_runs: list[tuple[str, int, str]] = []
    skipped_nan_raw: list[tuple[str, int]] = []
    skipped_nan_with_slippage: list[tuple[str, int]] = []
    skipped_nan_net: list[tuple[str, int]] = []
    total_trades_raw = 0
    total_trades_costed = 0
    total_runs = 0

    for generator_name, price_fn, generator_seeds in _GENERATORS:
        for seed in generator_seeds:
            panel = _build_null_panel(price_fn, N_DAYS, seed)

            raw_config = BacktestConfig(
                cost_model=ZeroCost(),
                fill_model=FillModel(slippage=ZeroSlippage()),
            )
            raw_result = run_backtest(strategy, panel, raw_config)

            costed_config = BacktestConfig()
            costed_result = run_backtest(strategy, panel, costed_config)
            total_runs += 1

            if raw_result.ruined or costed_result.ruined:
                ruined_sides: list[str] = []
                if raw_result.ruined:
                    ruined_sides.append("raw")
                if costed_result.ruined:
                    ruined_sides.append("costed")
                ruined_runs.append((generator_name, seed, "+".join(ruined_sides)))
                continue

            total_trades_raw += int(raw_result.n_trades)
            total_trades_costed += int(costed_result.n_trades)

            raw_sharpe = sharpe_ratio(raw_result.returns)
            # costed_result.gross_returns is AFTER market-impact slippage but BEFORE
            # explicit fees/taxes; it is NOT frictionless and must never be labelled
            # "gross" as if it were.
            with_slippage_sharpe = sharpe_ratio(costed_result.gross_returns)
            net_sharpe = sharpe_ratio(costed_result.returns)

            if np.isnan(raw_sharpe):
                skipped_nan_raw.append((generator_name, seed))
            else:
                raw_sharpes.append(float(raw_sharpe))

            if np.isnan(with_slippage_sharpe):
                skipped_nan_with_slippage.append((generator_name, seed))
            else:
                with_slippage_sharpes.append(float(with_slippage_sharpe))

            if np.isnan(net_sharpe):
                skipped_nan_net.append((generator_name, seed))
            else:
                net_sharpes.append(float(net_sharpe))

    assert not ruined_runs, (
        f"Strategy {name} destroyed capital on pure-noise input for {ruined_runs}; "
        "this indicates a forced-liquidation/cost-model bug and must not be ignored"
    )

    if not raw_sharpes:
        if total_trades_raw == 0 and total_trades_costed == 0:
            reason = (
                f"strategy never submitted a single trade on any of {total_runs} null runs; "
                "nothing to assert about its raw/net Sharpe"
            )
        else:
            reason = (
                f"strategy never produced a computable raw (zero-cost, zero-slippage) "
                f"Sharpe on any of {total_runs} null runs despite "
                f"{total_trades_raw} raw-run and {total_trades_costed} costed-run trades; "
                "degenerate raw return variance across all runs"
            )
        if skipped_nan_raw:
            reason += f" (skipped {len(skipped_nan_raw)} NaN raw Sharpe observations)"
        pytest.skip(reason)

    mean_raw = float(np.mean(raw_sharpes))
    frac_raw_positive = float(np.mean(np.asarray(raw_sharpes) > 0.0))
    mean_with_slippage = (
        float(np.mean(with_slippage_sharpes)) if with_slippage_sharpes else float("nan")
    )
    mean_net = (
        float(np.mean(net_sharpes)) if net_sharpes else float("nan")
    )
    mean_costed_trades_per_run = (
        total_trades_costed / total_runs if total_runs else 0.0
    )

    assert abs(mean_raw) <= TOLERANCE, (
        f"Strategy {name}: mean RAW (zero-cost, zero-slippage) Sharpe {mean_raw:.3f} "
        f"over {len(raw_sharpes)} null runs exceeds null tolerance {TOLERANCE}; this is "
        f"the ONLY frictionless measure of alpha in this test, so a positive/large result "
        f"here is a probable lookahead leak, NOT a cost-model or slippage artifact. "
        f"Context: mean with-slippage Sharpe {mean_with_slippage:.3f}, mean net Sharpe "
        f"{mean_net:.3f}, avg {mean_costed_trades_per_run:.0f} costed trades/run. "
        f"Per-run raw Sharpes: {raw_sharpes}"
    )

    assert frac_raw_positive < 1.0, (
        f"Strategy {name}: RAW (zero-cost, zero-slippage) Sharpe was positive in ALL "
        f"{len(raw_sharpes)} pure-noise runs (mean {mean_raw:.3f}, avg "
        f"{mean_costed_trades_per_run:.0f} costed trades/run); this is the ONLY "
        f"frictionless measure of alpha in this test, so a consistently positive result is "
        f"a probable lookahead leak or data-generation bug. Context: mean with-slippage "
        f"Sharpe {mean_with_slippage:.3f}, mean net Sharpe {mean_net:.3f}. Per-run raw "
        f"Sharpes: {raw_sharpes}"
    )

    if not net_sharpes:
        reason = (
            f"strategy produced computable RAW (zero-cost, zero-slippage) Sharpe on "
            f"{len(raw_sharpes)} null runs but never a computable NET Sharpe; skipping "
            f"cost-model anomaly assertion"
        )
        if skipped_nan_net:
            reason += f" (skipped {len(skipped_nan_net)} NaN net Sharpe observations)"
        pytest.skip(reason)

    mean_net = float(np.mean(net_sharpes))

    # With real transaction costs applied to data with zero embedded edge, net expectancy
    # can only be zero or negative; a strategy that manages to be net-positive with real
    # costs on pure noise has a broken cost model or fill model, not skill. A large
    # negative value (heavy churn cost) is EXPECTED and must never fail this assertion --
    # only a positive mean fails it.
    assert mean_net <= 1e-9, (
        f"Strategy {name}: mean NET Sharpe {mean_net:.3f} exceeds 0.0 over "
        f"{len(net_sharpes)} null runs (avg {mean_costed_trades_per_run:.0f} costed "
        f"trades/run) with real costs applied. Net expectancy can only be zero or "
        f"negative on pure noise; a positive mean indicates a broken cost model or fill "
        f"model, not skill. Context: mean RAW (zero-cost, zero-slippage) Sharpe "
        f"{mean_raw:.3f}, mean with-slippage Sharpe {mean_with_slippage:.3f}. Per-run "
        f"net Sharpes: {net_sharpes}"
    )
