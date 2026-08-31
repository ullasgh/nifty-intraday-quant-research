# Phase G3 experiment design — 7-scheme comparison on the tilt candidate. FROZEN 2026-08-31.

Status: FROZEN. Author: lead. D1 decided by Ullas's delegation ("pick your best"): option (b),
the lead's on-record recommendation. This document is the pre-registration; the 7 trials it
declares are charged on top of the program's existing trial history.

## What runs

The 7 registered weighting schemes (`backtest.weighting.WEIGHT_SCHEME_REGISTRY`), each
replacing the tilt's own weighting step on the SAME inputs the tilt already uses:

- signal: the H2 overnight-reversal feature at the entry checkpoint
  (`build_overnight_feature`, exactly as `run_tilt` computes it),
- sigma: `sigma_risk`-floored EWMA at the entry checkpoint, halflife
  `BacktestConfig.vol_sigma_halflife_bars` (= 20.0, AMENDMENT 4),
- corr: scalar median pairwise correlation, window `vol_corr_window_bars` (= 30),
- tradable: the tilt's own universe/tradability mask (rule 7: `present` never enters),
- costs: `NSEIntradayEquityCosts` one-leg-on-turnover, identical to `run_tilt`,
- window: research window only; `holdout_intent="never"` + the window assertion, as run_sweep.

Registration: ONE ResearchContract, `n_planned_trials = 7`, one TrialRecord per scheme via
the fixed config_hash path. `effective_n_trials` measured across the 7 daily-excess series;
promotion per `evaluate_scheme_promotion` (recent window, 0.95 DSR bar, AMENDMENT 6).
Output feeds Phase H's per-liquidity-decile capacity work; G3 itself promotes nothing to
production (spec G: "What this phase must NOT do").

## D1 — OPEN: the long-only mapping (the decision that defines the comparison)

The tilt candidate is LONG-ONLY index-relative; schemes emit signed weights. Options:

  (a) Long-only projection: w := max(w, 0), renormalise to gross — distorts schemes whose
      information is in the short side; z_weight and rank_weight become sign-truncated.
  (b) Signal shift: rank-map the signal to [0, 1] before weighting (losers highest), so all
      schemes see a nonnegative-signal problem — closest to what the tilt itself does
      (clipped-rank), keeps the 7 schemes comparable, but changes z_weight's meaning.
  (c) Benchmark-relative: emit signed ACTIVE weights around the index book, floor total
      weight (index + active) at zero per name — truest to "index-relative tilt", but adds
      an index-book construction the tilt wrapper does not currently expose.

DECIDED: (b). It compares the seven schemes as alternative TRANSFORMS of the same
nonnegative tilt intent — the question G3 actually asks — and needs no new engine surface.
(c) remains the better experiment in principle and is recorded as Phase-H-adjacent follow-up,
a NEW registered trial set if ever run.

## Frozen parameters (completing the "What runs" section in checkpoint-panel units)

The tilt wrapper operates on a 2-rows-per-session checkpoint panel with no minute bars, so
the AMENDMENT-4 estimator constants are reinterpreted in SESSION units, stated explicitly:

- mapped signal at session t: cross-sectional rank-percentile of (-overnight_return) among
  tradable names, in [0, 1] — losers highest, the tilt's own reversal intent. NaN feature ->
  not tradable that session.
- sigma at session t: EWMA (halflife 20 SESSIONS) of per-session entry->exit log returns,
  `sigma_risk`-floored; sessions before the estimator warms up leave the name out of the book.
- corr at session t: median pairwise correlation of the same per-session returns over a
  trailing 30-SESSION window (consumed only by risk_parity / covariance_aware as the
  equicorrelation rho).
- schemes called with gross=1.0, max_weight=0.05 via apply_weight_scheme — 0.05 is the tilt
  program's existing clip convention (capital/clip discipline in the tilt wrapper), not a new
  constant; clip_binding is recorded per session per scheme.
- returns: session entry->exit, weights fixed within session, rebalanced daily (tilt's
  rebalance_every=1); excess = book return minus the equal-weight tradable-universe return.
- costs: `NSEIntradayEquityCosts` round-trip at clip = capital/n_held, capital Rs 1,000,000,
  ONE leg on turnover sum|w_t - w_{t-1}| — the tilt wrapper's own accounting, reused not
  reimplemented where the code allows.
- window: the tilt research window (recon_low_turnover_tilt START/END), ending before the
  holdout boundary; the runner asserts this (holdout_intent="never").
- CALIBRATION NULL, asserted by the runner before recording anything: the `equal_weight`
  scheme over all tradable names IS the equal-weight benchmark book, so its gross excess must
  be ~0 by construction (|mean| below its own SE). A runner failing this check has broken
  accounting and must refuse to write trials.

## Cost of running

7 tilt-length backtests plus statistics: minutes, not hours. The expensive budget is not
CPU; it is the 7 registered trials and the framing above.
