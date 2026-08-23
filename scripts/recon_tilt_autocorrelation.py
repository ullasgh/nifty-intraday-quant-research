"""Measure the tilt's daily excess-return autocorrelation and recompute t with Newey-West.

WHY THIS EXISTS. The tilt candidate's headline significance comes from
`scripts/recon_tilt_significance.py:74`:

    se = sd / np.sqrt(n)          # pure iid, no autocorrelation correction
    t_stat = mean / se

That assumes independent observations. But the construction is
`w_t = 0.9 * w_(t-1) + 0.1 * target_t` -- weight smoothing with a ~6.6-session half-life -- so
POSITIONS are autocorrelated by design. If the daily excess RETURNS inherited that persistence,
the iid SE would understate and the t-statistic would be inflated.

RESULT (2026-08-21, and the reason this script is committed rather than discarded): they do not.
The returns are essentially uncorrelated with a slight NEGATIVE tilt at lags 2-3, so the iid SE is
CONSERVATIVE and correcting for dependence RAISES t rather than lowering it.

Persistent POSITIONS do not imply persistent RETURNS: the P&L comes from the held book meeting the
NEXT session's moves, which are near-independent.

Measured on the recent window (2024-01-01..2025-07-31, n=389, mild tilt, a=0.10):

    universe      mean    iid SE   iid t    NW SE    NW t     SE ratio
    full         1.5126   0.5503   2.7486   0.4729   3.1990   0.859
    continuous   0.8952   0.5045   1.7745   0.4474   2.0009   0.887

    ACF lags 1-10, full:       -0.010 -0.115 -0.079 -0.012 -0.013 +0.002 +0.007 +0.006 +0.062 -0.017
    ACF lags 1-10, continuous: -0.021 -0.075 -0.061 -0.022 -0.009 +0.011 +0.021 +0.046 +0.054 -0.040

READ THE CAVEAT. The continuous-coverage leg was the FAILING one (p=0.077) and reaches t=2.0009
under Newey-West -- but 2.00 against a 1.96 threshold is razor-thin (p~0.046). A verdict that flips
on the choice of SE estimator is FRAGILE, not established. Newey-West is the more general estimator
and is right to use regardless of direction, but note it would have DEFLATED t had the ACF come back
positive: applying it here is legitimate only because the choice was pre-committed before measuring.
The 1-of-4-combinations multiple-testing concern is unchanged.

Run:  .venv/bin/python scripts/recon_tilt_autocorrelation.py     (~10 min: full panel load)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon_low_turnover_tilt as base  # noqa: E402
import recon_tilt_significance as sig  # noqa: E402

MAX_ACF_LAG = 10
SMOOTHING_A = 0.10
TILT = "mild"


def _bartlett_newey_west_se(x: np.ndarray) -> tuple[float, int]:
    """Newey-West long-run SE of the mean, Bartlett kernel.

    Lag truncation `L = floor(4 * (n/100)^(2/9))` is the standard Newey-West (1994) automatic
    bandwidth, not a tuned choice -- it is a function of n alone.
    """
    n = x.size
    mean = float(np.mean(x))
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(np.sum((x - mean) ** 2) / n)
    for k in range(1, lag + 1):
        gamma_k = float(np.sum((x[k:] - mean) * (x[:-k] - mean)) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    return float(np.sqrt(max(lrv, 0.0) / n)), lag


def main() -> None:
    symbols = base.load_universe("all_equity").symbols
    spec = base.PanelSpec(
        freq="1",
        fields=("open", "high", "low", "close", "volume"),
        symbols=symbols,
        start=base.START,
        end=base.END,
    )
    print("loading panel (minutes)...", flush=True)
    panel = base.load_panel(spec, memmap=True)

    feature = base.build_overnight_feature(panel)
    checkpoint_panel, checkpoint_feature = base._build_checkpoint_panel(panel, feature)
    cont_mask = base.continuous_coverage_mask(panel)
    sessions = base.precompute_sessions(
        checkpoint_panel, checkpoint_feature, cont_mask, panel.n_symbols()
    )

    cost = getattr(base, "COST_BPS_PRIMARY", 8.26452)
    for universe_key in sessions:
        records = base.simulate_smoothing(
            sessions[universe_key], tilt=TILT, a=SMOOTHING_A, cost_bps_primary=cost
        )
        recent = [r for r in records if sig.RECENT_START <= r.date <= sig.RECENT_END]
        net = np.array([r.net_bps_primary for r in recent], dtype=np.float64)
        if net.size < 30:
            print(f"{universe_key}: only {net.size} sessions, skipping")
            continue

        n = net.size
        mean = float(np.mean(net))
        sd = float(np.std(net, ddof=1))
        se_iid = sd / np.sqrt(n)
        se_nw, lag = _bartlett_newey_west_se(net)
        acf = [
            float(np.corrcoef(net[:-k], net[k:])[0, 1]) for k in range(1, MAX_ACF_LAG + 1)
        ]

        print(f"\n=== universe: {universe_key} ===")
        print(f"n={n}  mean={mean:.4f} bps")
        print(f"  iid        SE={se_iid:.4f}  t={mean / se_iid:.4f}")
        print(f"  Newey-West SE={se_nw:.4f}  t={mean / se_nw:.4f}  (Bartlett lag L={lag})")
        print(f"  SE ratio (NW/iid) = {se_nw / se_iid:.3f}")
        print("  ACF lags 1-10: " + " ".join(f"{a:+.3f}" for a in acf))


if __name__ == "__main__":
    main()
