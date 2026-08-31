# Phase G3 experiment design — 7-scheme comparison on the tilt candidate. DRAFT, NOT FROZEN.

Status: DRAFT for review. Author: lead, 2026-08-31. **No trial may be registered against this
document until the DRAFT marker is removed** — freezing the design IS the pre-registration,
and this draft contains one open decision (D1) that determines what the experiment compares.

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

RECOMMENDATION: (b), because it compares the seven schemes as alternative TRANSFORMS of the
same nonnegative tilt intent — the question G3 actually asks — and needs no new engine
surface. (c) is the better experiment in principle but is Phase-H-adjacent scope. Decide,
record the decision here, remove DRAFT, then implement.

## Cost of running

7 tilt-length backtests plus statistics: minutes, not hours. The expensive budget is not
CPU; it is the 7 registered trials and the framing above.
