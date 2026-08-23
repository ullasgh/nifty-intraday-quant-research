#!/usr/bin/env python3
"""Freeze the tilt's holdout configuration as a pre-registration ``TrialRecord``.

Phase I step 4 requires the exact configuration that will be executed on the
holdout to be recorded and frozen BEFORE the holdout is spent. ``results/trials.db``
carried 23 rows and zero for ``tilt`` -- the only live candidate in this repo -- so
there was nothing to freeze and no fixed research-period number to compare a
holdout run against. This script writes exactly one row that supplies both.

What it does, in order:

1. Reads ``results/holdout_lock.json``'s ``count`` and remembers it.
2. Loads the panel for the RESEARCH window only (``2024-01-01 .. 2025-07-31``).
   The holdout window ``[2025-08-14, 2026-08-14]`` is never loaded; ``run_tilt``
   itself refuses any ``end`` at or past the holdout start, and
   ``_assert_research_window_outside_holdout`` refuses earlier still.
3. Runs the frozen configuration over the research window via the production
   ``run_tilt`` path (``registry=None``: this script writes the one row itself,
   rather than also leaving ``run_tilt``'s own ``purpose="exploration"`` row).
4. Writes the full research-period result to a dated JSON artifact under
   ``results/prereg/``.
5. Records ONE ``TrialRecord`` with ``purpose="preregistration"``.
6. Re-reads the holdout count and refuses to exit 0 if it moved.

Determinism: no randomness anywhere (``seed`` is carried for provenance only; the
tilt does not consume it). Re-running the script rewrites the same artifact and
is a no-op on the registry -- ``TrialRegistry.record`` is
``INSERT OR IGNORE`` on ``(config_hash, split_id)``.

Side effects: writes ``results/prereg/*.json`` and one row in
``results/trials.db``. Never modifies ``src/``, ``tests/`` or ``data/``.

WHY ``config_hash`` IS NOT ``contract_hash`` HERE
-------------------------------------------------
``run_tilt`` sets ``config_hash = contract_hash = contract.contract_hash``.
``ResearchContract.contract_hash`` hashes the contract's six sections plus seed --
which for ``tilt`` covers the panel, universe, dates, cost model and seed, but
**not a single TiltConfig parameter**: ``smoothing``, ``tilt``, ``capital``,
``entry_hhmm``, ``exit_hhmm``, ``rebalance_every`` and ``continuous_only`` all
live outside the contract, so flipping ``smoothing`` from 0.10 to 1.0 -- the
difference between +3.22 and -9.41 bps/day -- leaves ``contract_hash``
byte-identical. A pre-registration keyed on that hash would bind nothing.

So this script computes its own ``config_hash`` over the frozen tilt parameters
plus the declared holdout window (:func:`prereg_config_hash`), and stores the
contract's own hash in the ``contract_hash`` column unchanged. A
``record.contract_hash == contract.contract_hash`` check still succeeds; only a
legacy "look the row up by contract hash in the config_hash column" query does
not, and no such query exists for ``purpose="preregistration"`` rows.

The research start/end are deliberately EXCLUDED from ``config_hash``: the
holdout run executes these same parameters over a DIFFERENT window by
construction, so binding the research dates into the hash would guarantee the
holdout run could never match its own pre-registration. The declared holdout
window is bound instead.

Usage::

    NQ_CACHE_ROOT=/some/scratch python scripts/prereg_tilt.py            # write the row
    python scripts/prereg_tilt.py --dry-run                             # print, write nothing
    python scripts/prereg_tilt.py --show-binding                        # hash-binding proof
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from typing import Any

# ---------------------------------------------------------------------------
# The frozen configuration. Changing ANY value below changes `config_hash`, which
# is the whole point: a pre-registration that does not bind the parameters is
# theatre. See `tests/test_prereg_tilt.py::test_every_frozen_param_changes_hash`.
# ---------------------------------------------------------------------------

#: Every `TiltConfig` field EXCEPT `start`/`end` (see module docstring for why the
#: dates are excluded). `test_frozen_params_cover_every_tilt_config_field` asserts
#: this dict's keys are exactly `TiltConfig`'s fields minus {start, end}, so a new
#: TiltConfig parameter cannot silently escape the pre-registration.
FROZEN_PARAMS: dict[str, Any] = {
    "entry_hhmm": "09:16",
    "exit_hhmm": "15:20",
    "capital": 1_000_000.0,
    "tilt": "mild",
    "smoothing": 0.10,
    "rebalance_every": 1,
    "universe": "all_equity",
    "continuous_only": False,
    "seed": 0,
}

#: The holdout window, as stored in `results/holdout_lock.json`. Declared here as
#: intent only -- this script never loads a bar inside it.
HOLDOUT_START = datetime.date(2025, 8, 14)
HOLDOUT_END = datetime.date(2026, 8, 14)

#: The research window whose result is frozen as the comparison baseline. Chosen as
#: the window the recent-window significance verdict was measured on
#: (`scripts/recon_tilt_significance.py`, n=389) and the window immediately
#: preceding the holdout, so the holdout is the natural out-of-sample continuation.
RESEARCH_START = datetime.date(2024, 1, 1)
RESEARCH_END = datetime.date(2025, 7, 31)

#: `purpose` value for a frozen pre-registration. Deliberately NOT "exploration":
#: `TrialRegistry.n_trials`/`trial_sharpes`/`var_trial_sharpes`/`build_trial_matrix`
#: all default to `purpose="exploration"`, so a pre-registration row must not be
#: counted as another swept trial in the PBO/DSR machinery.
PREREG_PURPOSE = "preregistration"

#: Schema tag inside the hashed payload. Bumping it invalidates every previously
#: recorded pre-registration hash on purpose, rather than silently colliding.
PREREG_SCHEMA = "tilt_prereg_v1"

STRATEGY = "tilt"

#: Fixed cost/execution ids for the tilt path. `run_tilt` hardcodes
#: `NSEIntradayEquityCosts()` with no slippage model and no fill model: it prices
#: entry and exit at the checkpoint bars' closes with no latency and no partial
#: fills. Named rather than left "" so a reader is told what was assumed.
COST_MODEL_ID = "nse_intraday_default"
SLIPPAGE_MODEL_ID = "none"
FILL_MODEL_ID = "checkpoint_close_immediate_full"

#: `run_tilt` uses a same-session entry->exit holding period and a one-session
#: overnight feature lookback. There is no label horizon and no execution delay.
EMBARGO_FEATURE_LOOKBACK = 1.0
EMBARGO_LABEL_HORIZON = 0.0
EMBARGO_HOLDING_PERIOD = 1.0
EMBARGO_EXECUTION_HORIZON = 0.0


def split_id() -> str:
    """Stable `split_id` naming the window the pre-registration is for."""
    return f"prereg_holdout_{HOLDOUT_START.isoformat()}_{HOLDOUT_END.isoformat()}"


def _assert_research_window_outside_holdout() -> None:
    """Refuse before any data is touched if the research window reaches the holdout.

    `run_tilt` performs the same check against the GLOBAL calendar-derived
    boundary; this one is a cheap, local, constant-only tripwire so that an edit to
    `RESEARCH_END` fails in a unit test rather than only at panel-load time.
    """
    if RESEARCH_END >= HOLDOUT_START:
        raise ValueError(
            f"research window end {RESEARCH_END} reaches the holdout window "
            f"[{HOLDOUT_START}, {HOLDOUT_END}]; a pre-registration must never "
            "read the holdout"
        )
    if RESEARCH_START > RESEARCH_END:
        raise ValueError(
            f"research start {RESEARCH_START} is after end {RESEARCH_END}"
        )


def frozen_tilt_config(
    start: datetime.date = RESEARCH_START,
    end: datetime.date = RESEARCH_END,
) -> Any:
    """Build the frozen `TiltConfig` over `[start, end]`.

    The SAME `FROZEN_PARAMS` that go into `config_hash` are what is passed to
    `run_tilt`, so the hash cannot drift away from what actually executes.
    """
    from nifty_quant.research.tilt import TiltConfig

    return TiltConfig(start=start, end=end, **FROZEN_PARAMS)


def prereg_config_hash(
    params: dict[str, Any] | None = None,
    *,
    holdout_start: datetime.date = HOLDOUT_START,
    holdout_end: datetime.date = HOLDOUT_END,
) -> str:
    """Hash of the configuration that will actually be executed on the holdout.

    Binds every frozen tilt parameter plus the declared holdout window, via the
    repo's single canonicalisation mechanism (`research.contract.canonical_hash`),
    so the result is key-order insensitive and deterministic across processes.

    Deliberately does NOT bind the research window -- see the module docstring.
    """
    from nifty_quant.research.contract import canonical_hash

    payload = {
        "schema": PREREG_SCHEMA,
        "strategy": STRATEGY,
        "params": dict(FROZEN_PARAMS if params is None else params),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": holdout_end.isoformat(),
    }
    return canonical_hash(payload)


def binding_report() -> str:
    """Human-readable proof that `config_hash` binds every frozen parameter.

    Perturbs each parameter in turn and shows the hash moving. Used by
    `--show-binding` and asserted on in the test suite.
    """
    perturbations: dict[str, Any] = {
        "entry_hhmm": "09:20",
        "exit_hhmm": "15:15",
        "capital": 2_000_000.0,
        "tilt": "aggressive",
        "smoothing": 1.0,
        "rebalance_every": 2,
        "universe": "nifty50",
        "continuous_only": True,
        "seed": 1,
    }
    base = prereg_config_hash()
    lines = [
        "config_hash binding proof",
        f"  baseline (frozen config)                 {base}",
        "",
        "  perturbed parameter                       config_hash        changed?",
    ]
    for name, value in perturbations.items():
        mutated = dict(FROZEN_PARAMS)
        mutated[name] = value
        digest = prereg_config_hash(mutated)
        changed = "YES" if digest != base else "*** NO -- NOT BOUND ***"
        label = f"{name}={value!r}"
        lines.append(f"  {label:<40} {digest}  {changed}")

    shifted = prereg_config_hash(holdout_end=HOLDOUT_END + datetime.timedelta(days=1))
    lines.append("")
    lines.append(
        f"  {'holdout_end + 1 day':<40} {shifted}  "
        f"{'YES' if shifted != base else '*** NO -- NOT BOUND ***'}"
    )
    return "\n".join(lines)


def build_prereg_record(
    *,
    result: Any,
    contract: Any,
    panel_hash: str,
    universe_hash: str,
    data_fingerprint: str,
    result_path: str,
    wall_s: float,
    ts: str | None = None,
) -> Any:
    """Assemble the pre-registration `TrialRecord`.

    Every provenance column that has a real value is populated. The three that are
    left NULL, and why -- stated here rather than left for a reader to guess:

    - `sharpe_gross`, `sharpe_net`: `TiltResult` exposes per-year and pooled MEANS
      only; it never returns the daily excess-return series, and a Sharpe needs the
      daily standard deviation. Reconstructing the series outside `run_tilt` would
      be a second implementation of the smoothing recursion, which is exactly how
      two paths silently disagree. Recorded honestly as absent; the pooled
      `gross_bps`/`net_bps`/`ann_net_pct` live in the JSON artifact at
      `result_path`.
    - `n_trades`: the tilt holds a continuously-smoothed weight vector rather than
      discrete round trips, so a trade count is not defined for it. `turnover`
      (mean daily sum of |weight change|) is the meaningful analogue and IS
      recorded.

    `breakeven_bps` IS populated: it is the pooled gross excess in bps/day, i.e. the
    daily cost at which net reaches zero. `TiltResult.breakeven_turnover` is the
    same statement in turnover units and is in the artifact.
    """
    from nifty_quant import __version__
    from nifty_quant.research.provenance import (
        FEATURE_VERSION,
        embargo_components_json,
        get_git_sha,
    )
    from nifty_quant.research.registry import TrialRecord

    git_sha = get_git_sha()
    if not git_sha:  # pragma: no cover - get_git_sha never returns "" or None
        raise ValueError("refusing to pre-register with an empty git_sha")

    config = result.config
    return TrialRecord(
        config_hash=prereg_config_hash(),
        contract_hash=contract.contract_hash,
        ts=ts
        or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        strategy=STRATEGY,
        params_json=json.dumps(
            {
                "schema": PREREG_SCHEMA,
                "frozen_params": dict(FROZEN_PARAMS),
                "holdout_start": HOLDOUT_START.isoformat(),
                "holdout_end": HOLDOUT_END.isoformat(),
                "research_start": RESEARCH_START.isoformat(),
                "research_end": RESEARCH_END.isoformat(),
                "research_result": _result_summary(result),
                "contract_sections": {
                    "data": contract.data,
                    "features": contract.features,
                    "label": contract.label,
                    "execution": contract.execution,
                    "portfolio": contract.portfolio,
                    "validation": contract.validation,
                    "seed": contract.seed,
                },
            },
            sort_keys=True,
            default=str,
        ),
        split_id=split_id(),
        purpose=PREREG_PURPOSE,  # type: ignore[arg-type]
        sharpe_gross=None,
        sharpe_net=None,
        n_trades=None,
        turnover=float(result.total.turnover),
        breakeven_bps=float(result.total.gross_bps),
        git_sha=git_sha,
        data_fingerprint=data_fingerprint,
        code_version=__version__,
        wall_s=float(wall_s),
        result_path=result_path,
        error=None,
        ruined=False,
        ruin_index=None,
        seed=int(config.seed),
        universe_name=str(config.universe),
        universe_hash=universe_hash,
        panel_hash=panel_hash,
        start=RESEARCH_START.isoformat(),
        end=RESEARCH_END.isoformat(),
        cost_model_id=COST_MODEL_ID,
        slippage_model_id=SLIPPAGE_MODEL_ID,
        fill_model_id=FILL_MODEL_ID,
        embargo_components=embargo_components_json(
            feature_lookback=EMBARGO_FEATURE_LOOKBACK,
            label_horizon=EMBARGO_LABEL_HORIZON,
            holding_period=EMBARGO_HOLDING_PERIOD,
            execution_horizon=EMBARGO_EXECUTION_HORIZON,
        ),
        parent_trial_id=None,
        feature_version=FEATURE_VERSION,
    )


def _result_summary(result: Any) -> dict[str, Any]:
    """JSON-ready view of the frozen research-period result."""
    return {
        "n_sessions": int(result.total.n_sessions),
        "gross_bps": float(result.total.gross_bps),
        "turnover": float(result.total.turnover),
        "cost_bps": float(result.total.cost_bps),
        "net_bps": float(result.total.net_bps),
        "ann_net_pct": float(result.total.ann_net_pct),
        "clip_per_name": float(result.clip_per_name),
        "round_trip_bps": float(result.round_trip_bps),
        "breakeven_turnover": float(result.breakeven_turnover),
        "n_symbols": int(result.n_symbols),
        "max_n_held": int(result.max_n_held),
        "min_weight_seen": float(result.min_weight_seen),
        "max_weight_sum_deviation": float(result.max_weight_sum_deviation),
        "per_year": [dataclasses.asdict(row) for row in result.per_year],
        "n_warnings": len(result.warnings),
    }


def _holdout_count() -> int:
    from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path

    return HoldoutLock(path=default_holdout_lock_path()).read_count()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the backtest and print the record without writing anything.",
    )
    parser.add_argument(
        "--show-binding",
        action="store_true",
        help="Print the config_hash binding proof and exit without loading data.",
    )
    args = parser.parse_args(argv)

    if args.show_binding:
        print(binding_report())
        return 0

    import time

    from nifty_quant import settings
    from nifty_quant.data.manifest import Manifest
    from nifty_quant.data.panel import PanelSpec, load_panel
    from nifty_quant.research.contract import ResearchContract
    from nifty_quant.research.provenance import (
        FEATURE_VERSION,
        compute_panel_hash,
        compute_universe_hash,
    )
    from nifty_quant.research.registry import TrialRegistry
    from nifty_quant.research.tilt import run_tilt
    from nifty_quant.universe.static import load_universe

    _assert_research_window_outside_holdout()

    count_before = _holdout_count()
    print(f"holdout_lock count BEFORE: {count_before}", flush=True)
    print(binding_report(), flush=True)
    print("", flush=True)

    config = frozen_tilt_config()
    universe = load_universe(str(config.universe))
    print(
        f"Universe {config.universe}: {len(universe.symbols)} symbols. "
        f"Loading panel {RESEARCH_START} .. {RESEARCH_END} "
        f"(holdout {HOLDOUT_START} .. {HOLDOUT_END} is NOT loaded)...",
        flush=True,
    )
    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=universe.symbols,
        start=RESEARCH_START,
        end=RESEARCH_END,
    )
    panel = load_panel(spec)
    print(
        f"Panel: {panel.n_rows()} rows, {panel.n_days()} sessions, "
        f"{panel.n_symbols()} symbols.",
        flush=True,
    )

    manifest = Manifest.load()
    panel_hash = compute_panel_hash(panel, adjustments=manifest.adjustments)
    universe_hash = compute_universe_hash(
        name=str(config.universe),
        symbols=universe.symbols,
        n_sessions=panel.n_days(),
    )

    contract = ResearchContract(
        data={
            "panel_id": str(spec),
            "panel_hash": panel_hash,
            "start": RESEARCH_START.isoformat(),
            "end": RESEARCH_END.isoformat(),
            "bar_interval_s": 60,
            "universe_name": str(config.universe),
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
            "cost_model_id": COST_MODEL_ID,
            "slippage_model_id": SLIPPAGE_MODEL_ID,
            "decision_latency_bars": 0,
            "participation_cap": 1.0,
        },
        portfolio={
            "sizing_scheme": "long_only_weight_tilt",
            "gross_clip": 1.0,
            "max_weight": 1.0,
            "target_vol": None,
            "smoothing": FROZEN_PARAMS["smoothing"],
            "tilt": FROZEN_PARAMS["tilt"],
            "rebalance_every": FROZEN_PARAMS["rebalance_every"],
            "capital": FROZEN_PARAMS["capital"],
        },
        validation={
            "split_scheme": "holdout",
            "purge_width_bars": 0,
            "embargo_width_bars": 0,
            "n_planned_trials": 1,
            # The pre-registration DECLARES the intent to read the holdout once the
            # validation conditions close. It does not read it: "reading_now" is the
            # only value that may reach the holdout, and this is not that.
            "holdout_intent": "after_conditions_close",
        },
        seed=int(config.seed),
    )

    print("Running the frozen configuration over the research window...", flush=True)
    t0 = time.perf_counter()
    # registry=None: this script writes the single pre-registration row itself.
    # Letting run_tilt write as well would add a second, purpose="exploration" row
    # keyed on contract_hash, which binds none of the tilt parameters.
    result = run_tilt(panel, config, contract=contract, registry=None)
    wall_s = time.perf_counter() - t0
    print("", flush=True)
    print(result.to_table(), flush=True)
    print("", flush=True)

    config_hash = prereg_config_hash()
    artifact_dir = settings.RESULTS_ROOT / "prereg"
    artifact_path = artifact_dir / f"tilt_{config_hash}.json"
    rel_artifact = str(artifact_path.relative_to(settings.REPO_ROOT))

    record = build_prereg_record(
        result=result,
        contract=contract,
        panel_hash=panel_hash,
        universe_hash=universe_hash,
        data_fingerprint=manifest.fingerprint,
        result_path=rel_artifact,
        wall_s=wall_s,
    )

    if args.dry_run:
        print("--dry-run: nothing written. Record would be:", flush=True)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "config_hash": record.config_hash,
                    "contract_hash": record.contract_hash,
                    "ts": record.ts,
                    "git_sha": record.git_sha,
                    "code_version": record.code_version,
                    "purpose": record.purpose,
                    "split_id": record.split_id,
                    "frozen_params": dict(FROZEN_PARAMS),
                    "holdout_start": HOLDOUT_START.isoformat(),
                    "holdout_end": HOLDOUT_END.isoformat(),
                    "research_start": RESEARCH_START.isoformat(),
                    "research_end": RESEARCH_END.isoformat(),
                    "research_result": _result_summary(result),
                    "result_table": result.to_table(),
                    "warnings": list(result.warnings),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        registry = TrialRegistry(settings.RESULTS_ROOT / "trials.db")
        registry.record(record)
        print(f"Artifact: {artifact_path}", flush=True)
        print(f"Registry: {settings.RESULTS_ROOT / 'trials.db'}", flush=True)

    print("", flush=True)
    for name, value in dataclasses.asdict(record).items():
        print(f"  {name:<20} {value}", flush=True)

    count_after = _holdout_count()
    print("", flush=True)
    print(f"holdout_lock count AFTER: {count_after}", flush=True)
    if count_after != count_before:
        raise SystemExit(
            f"HOLDOUT WAS TOUCHED: count went {count_before} -> {count_after}. "
            "Pre-registration must never read the holdout."
        )
    print("Holdout untouched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
