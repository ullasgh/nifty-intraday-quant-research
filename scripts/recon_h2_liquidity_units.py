"""Re-measure H2's kill criterion 4 (liquidity concentration) with correct units and causality.

WHY THIS EXISTS
---------------
H2 (cross-sectional overnight reversal) is the largest effect in this program: pooled
-24.30 bps, t = -16.66, 8/8 years sign-consistent. Once clip size is accounted for it
clears criteria 1 and 7 at a Rs 10L clip, so criterion 4 is its SOLE remaining kill reason.

Criterion 4 is computed in `Lens.stability_report` (lens.py, ~line 468) as:

    volume = self.panel.field("volume").astype(np.float64)
    # Compute volume deciles (causal, per row)
    volume_quantiles = np.quantile(volume[np.isfinite(volume)], np.linspace(0, 1, 11))

Two apparent defects:

(a) WRONG QUANTITY -- `volume` is raw SHARE COUNT, not rupee turnover. A name trading at
    Rs 1 lakh/share has tiny share volume and large rupee turnover, so it lands in the
    "bottom liquidity decile" while being highly liquid.

(b) FULL-SAMPLE LOOKAHEAD -- `np.quantile` over the whole flattened array derives thresholds
    from the ENTIRE panel including the future, despite the comment claiming "causal, per
    row". The repo already has the correct machinery in
    `expectancy.expectancy_by_liquidity_decile`, whose `prior_adv` argument is documented
    "must already be lagged (strictly prior sessions)" and which buckets via
    `causal_buckets(..., method='cross_sectional_rank')`. Lens bypasses it.

This script re-measures criterion 4 three ways on the SAME checkpoint panel H2 uses:
    1. share volume, full-sample quantiles      (what Lens does today)
    2. rupee turnover, full-sample quantiles    (isolates defect (a))
    3. rupee turnover, strictly-prior causal    (the correct measurement)

Run: .venv/bin/python scripts/recon_h2_liquidity_units.py
"""

from __future__ import annotations

import datetime

import numpy as np

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.research import expectancy
from nifty_quant.research.hypotheses.h2_overnight_reversal import (
    _build_checkpoint_panel,
    build_overnight_feature,
)
from nifty_quant.universe.static import load_universe

START = datetime.date(2018, 1, 1)
END = datetime.date(2025, 7, 31)  # holdout (last 12 months) NOT read
N_DECILES = 10
SEED = 0


def _decile_spreads(
    feature_values: np.ndarray,
    fwd: expectancy.ForwardReturns,
    day_offsets: np.ndarray,
    liquidity: np.ndarray,
    *,
    causal: bool | str,
) -> dict[int, float]:
    """Return {decile: spread_bps} bucketing on `liquidity`.

    causal=True uses the repo's `expectancy_by_liquidity_decile`, which buckets via
    cross-sectional rank per row on a caller-supplied STRICTLY PRIOR quantity.
    causal=False reproduces Lens's current full-sample `np.quantile` bucketing.
    """
    if causal == "production_geometry":
        # Production uses 10 LIQUIDITY deciles x 5 FEATURE quintiles (lens.py:511 passes
        # n_buckets=5 to conditional_expectancy while looping range(10) over deciles).
        # `expectancy_by_liquidity_decile` couples BOTH to a single n_buckets, so it cannot
        # express that geometry -- hence this explicit path. Deciles are still bucketed
        # causally via cross_sectional_rank on the strictly-prior quantity.
        from nifty_quant.features.core import cross_sectional_rank

        ranks = cross_sectional_rank(liquidity)
        labels = np.full(liquidity.shape, -1, dtype=np.int64)
        ok = np.isfinite(ranks)
        labels[ok] = np.clip((ranks[ok] * N_DECILES).astype(np.int64), 0, N_DECILES - 1)

        out_pg: dict[int, float] = {}
        for d in range(N_DECILES):
            mask = labels == d
            if not np.any(mask):
                continue
            feat_d = np.where(mask, feature_values, np.nan)
            masked = np.where(mask, fwd.values, np.nan)
            fwd_d = expectancy.ForwardReturns(
                values=masked,
                horizon=fwd.horizon,
                session_bounded=fwd.session_bounded,
                n_defined=int(np.sum(np.isfinite(masked))),
                n_nan_tail=fwd.n_nan_tail,
            )
            table = expectancy.conditional_expectancy(
                feat_d, fwd_d, day_offsets, n_buckets=5,
                method="cross_sectional_rank", seed=SEED,
            )
            out_pg[d] = float(table.spread_bps)
        return out_pg

    if causal:
        # method="cross_sectional_rank" is MANDATORY. The default here is
        # "expanding_quantile", which needs min_history rows per column and can never
        # bucket a once-per-session signal -- it silently returns empty tables whose
        # spread_bps is 0.0, indistinguishable from "no effect". This is the same trap
        # that produced H2's 25,000x error and it is the DEFAULT on this function.
        tables = expectancy.expectancy_by_liquidity_decile(
            feature_values,
            fwd,
            day_offsets,
            liquidity,
            n_buckets=N_DECILES,
            method="cross_sectional_rank",
            seed=SEED,
        )
        return {d: float(t.spread_bps) for d, t in tables.items()}

    finite = liquidity[np.isfinite(liquidity)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, N_DECILES + 1))
    labels = np.full(liquidity.shape, -1, dtype=np.int64)
    ok = np.isfinite(liquidity)
    labels[ok] = np.searchsorted(edges[1:-1], liquidity[ok], side="right")

    out: dict[int, float] = {}
    for d in range(N_DECILES):
        mask = labels == d
        if not np.any(mask):
            continue
        feat_d = np.where(mask, feature_values, np.nan)
        masked_vals = np.where(mask, fwd.values, np.nan)
        fwd_d = expectancy.ForwardReturns(
            values=masked_vals,
            horizon=fwd.horizon,
            session_bounded=fwd.session_bounded,
            n_defined=int(np.sum(np.isfinite(masked_vals))),
            n_nan_tail=fwd.n_nan_tail,
        )
        table = expectancy.conditional_expectancy(
            feat_d, fwd_d, day_offsets, n_buckets=5,
            method="cross_sectional_rank", seed=SEED,
        )
        out[d] = float(table.spread_bps)
    return out


def _prior_session_mean(values: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Per-symbol mean of `values` over STRICTLY PRIOR sessions, broadcast to each row.

    Session s gets the mean over sessions [0, s); session 0 is NaN. This is the lag the
    `prior_adv` contract requires and that Lens never applies.
    """
    n_days = len(day_offsets) - 1
    per_day = np.full((n_days, values.shape[1]), np.nan, dtype=np.float64)
    for s in range(n_days):
        seg = values[day_offsets[s] : day_offsets[s + 1]]
        with np.errstate(invalid="ignore"):
            per_day[s] = np.nanmean(np.where(np.isfinite(seg), seg, np.nan), axis=0)

    prior = np.full_like(per_day, np.nan)
    for s in range(1, n_days):
        with np.errstate(invalid="ignore"):
            prior[s] = np.nanmean(per_day[:s], axis=0)

    out = np.full(values.shape, np.nan, dtype=np.float64)
    for s in range(n_days):
        out[day_offsets[s] : day_offsets[s + 1]] = prior[s]
    return out


def _report(name: str, spreads: dict[int, float]) -> None:
    print(f"\n--- {name} ---")
    for d in sorted(spreads):
        print(f"  decile {d}: {spreads[d]:+9.2f} bps")
    finite = {d: v for d, v in spreads.items() if np.isfinite(v) and v != 0.0}
    if not finite:
        print("  (no finite non-zero deciles)")
        return
    abs_vals = {d: abs(v) for d, v in finite.items()}
    argmax = max(abs_vals, key=lambda d: abs_vals[d])
    # PRODUCTION median, not np.median. `lens.py:665` uses
    # `sorted_liquidity_edges[len(sorted_liquidity_edges) // 2]` -- the UPPER median for an
    # even count, where np.median averages the 5th and 6th. The upper median is >= np.median,
    # so np.median inflates the ratio. With 10 deciles and a 0.5% decision margin that
    # difference alone can flip criterion 4. Both are reported below.
    sorted_abs = sorted(abs_vals.values())
    med_prod = float(sorted_abs[len(sorted_abs) // 2])
    med_np = float(np.median(sorted_abs))
    ratio = abs_vals[argmax] / med_prod if med_prod > 0 else float("inf")
    ratio_np = abs_vals[argmax] / med_np if med_np > 0 else float("inf")
    bottom_is_max = argmax == 0
    fires = bottom_is_max and ratio > 2.0
    print(f"  argmax |spread| = decile {argmax}")
    print(f"  max/median ratio, PRODUCTION median (upper): {ratio:.4f}")
    print(f"  max/median ratio, np.median (averaged):      {ratio_np:.4f}")
    print(f"  bottom-is-argmax = {bottom_is_max}   production ratio > 2 = {ratio > 2.0}")
    print(f"  ==> criterion 4 FIRES (production convention): {fires}")


def main() -> None:
    universe = load_universe("all_equity")
    spec = PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=universe.symbols,
        start=START,
        end=END,
    )
    panel = load_panel(spec)
    print(f"panel: {panel.n_rows()} rows x {panel.n_symbols()} symbols")

    feature = build_overnight_feature(panel)
    cp_panel, cp_feature = _build_checkpoint_panel(panel, feature)
    print(f"checkpoint panel: {cp_panel.n_rows()} rows ({cp_panel.n_rows() // 2} sessions)")

    # horizon=1 is correct ONLY because the checkpoint panel has exactly two rows per
    # session (09:16 entry, 15:20 exit) -- the same reduction run_h2 performs. On a raw
    # many-bar panel this would measure one MINUTE, which is the 25,000x H2 error.
    fwd = expectancy.forward_returns(
        cp_panel.field("close").astype(np.float64), cp_panel.day_offsets, 1
    )

    share_vol = cp_panel.field("volume").astype(np.float64)
    rupee_turnover = share_vol * cp_panel.field("close").astype(np.float64)
    prior_rupee = _prior_session_mean(rupee_turnover, cp_panel.day_offsets)

    # --- membership comparison -------------------------------------------------
    def bottom_names(liq: np.ndarray) -> set[str]:
        finite = liq[np.isfinite(liq)]
        edges = np.quantile(finite, np.linspace(0.0, 1.0, N_DECILES + 1))
        labels = np.full(liq.shape, -1, dtype=np.int64)
        ok = np.isfinite(liq)
        labels[ok] = np.searchsorted(edges[1:-1], liq[ok], side="right")
        frac = (labels == 0).sum(axis=0) / np.maximum((labels >= 0).sum(axis=0), 1)
        return {s for s, f in zip(cp_panel.symbols, frac) if f > 0.5}

    by_share = bottom_names(share_vol)
    by_rupee = bottom_names(rupee_turnover)
    print(f"\ndecile-0 by SHARE VOLUME  ({len(by_share)}): {sorted(by_share)}")
    print(f"decile-0 by RUPEE TURNOVER ({len(by_rupee)}): {sorted(by_rupee)}")
    print(f"overlap: {len(by_share & by_rupee)} / union {len(by_share | by_rupee)}")

    # --- three measurements ----------------------------------------------------
    _report(
        "1. SHARE VOLUME, full-sample quantiles  (what Lens does today)",
        _decile_spreads(cp_feature.values, fwd, cp_panel.day_offsets, share_vol, causal=False),
    )
    _report(
        "2. RUPEE TURNOVER, full-sample quantiles  (isolates the units defect)",
        _decile_spreads(
            cp_feature.values, fwd, cp_panel.day_offsets, rupee_turnover, causal=False
        ),
    )
    _report(
        "3. RUPEE TURNOVER, strictly-prior causal, 10 deciles x 10 feature buckets",
        _decile_spreads(
            cp_feature.values, fwd, cp_panel.day_offsets, prior_rupee, causal=True
        ),
    )
    _report(
        "4. RUPEE TURNOVER, strictly-prior causal, 10 deciles x 5 feature quintiles "
        "(PRODUCTION GEOMETRY -- this is the number that decides H2)",
        _decile_spreads(
            cp_feature.values,
            fwd,
            cp_panel.day_offsets,
            prior_rupee,
            causal="production_geometry",
        ),
    )


if __name__ == "__main__":
    main()
