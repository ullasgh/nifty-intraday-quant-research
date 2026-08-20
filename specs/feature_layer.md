# Spec D — the feature layer

Status: spec, written before implementation. Author: lead. Date 2026-08-20.

## Context established by inventory, so the work is smaller than it looks

**14 of the 15 public functions in `features/market.py` are DEAD** — built, spec'd, tested, and
called by nothing. Only `tradable_overnight_return` is wired (`h2_overnight_reversal.py:50,78`,
reached via `research/tilt.py:380`). No plugin imports `features.market` at all.

So most of what the external review asked for already exists and is unwired. The genuinely new work
is small, and the largest single item is not a feature at all — it is the gap-stitched price path,
because without it the Hurst estimate cannot be made causal at the session boundary.

## D1 — the gap-stitched price path. This one is load-bearing.

**The problem, measured.** `rolling_hurst(price, window=390, ..., day_offsets=None)` is called by
`volume_breakout.py:132-146` with `day_offsets=None` **deliberately**: with day bounds and a
390-bar window against ~375-bar sessions, `_segment_bounds` confines every window to one session
and the result was `0/92520 finite on RELIANCE 2024`. Not a bug — an arithmetic impossibility. A
390-bar window cannot fit inside a 375-bar session.

So today the estimator sees overnight jumps as if they were one-minute returns, and the documented
workaround is to accept that. Neither branch is acceptable: day-bounded gives no data, unbounded
gives a contaminated estimate.

**The resolution.** Stitch the path, then estimate on the stitched path without day bounds:

    stitched[0]        = close[0]
    stitched[t]        = stitched[t-1] * (close[t] / close[t-1])       within a session
    stitched[t]        = stitched[t-1]                                  across a session boundary

i.e. the overnight return is set to zero rather than carried, so the series is continuous in level
and the cross-session log-difference is 0 instead of a jump. The window may then legitimately span
sessions, because what it spans is a series with the overnight jumps removed.

**This is NOT forward-filling and must not become it (rule 6).** A stitched value is only ever
produced where a real bar exists. `NaN` in `close` stays `NaN` in `stitched`; the stitch never
manufactures a bar, it only removes a jump between two real ones. The function must be named so
this is obvious (`stitch_overnight_gaps`, not `fill_*`), and it must be an explicit, visible call
at the call site — never applied silently inside another estimator.

Nothing like this exists today: zero `stitch` hits in `src/`. `strategy/base.py:105 _ffill_2d` is a
different thing (opt-in NaN carry) and must not be reused for this.

**Then** `hurst_on_stitched(close, day_offsets, window=390)` is the causal call path, and
`H > 0.55` stops being an assumption: D5 measures whether the relationship is monotonic at all.

## D2 — breakout strength, continuous

`breakout_up`/`breakout_down` (`core.py:660`, `:685`) return booleans. A boolean throws away how far
through the level the price went, which is the part that carries information.

    breakout_strength = (close - prior_window_high) / sigma

with `sigma` from the existing `parkinson_volatility` on the same window, and the prior window high
taken from `rolling_max` shifted one bar (the existing convention — do not change it). Negative
values are meaningful: they say how far BELOW the level price sits.

Keep the booleans for compatibility. They become `breakout_strength > 0`, and a test must assert
that identity so the two cannot drift apart.

## D3 — volatility estimator family

Today exactly three exist: `parkinson_volatility` (`core.py:311`), `ewma_volatility_ann`
(`core.py:377`, **zero callers**), and close-to-close assembled from `log_returns` + `rolling_std`.

Add **Garman-Klass** and **Rogers-Satchell**, both day-offset aware and NaN-propagating on the same
convention as `parkinson_volatility`. Add:

    sigma_risk = max(sigma_ewma, sigma_floor)

available to the sizer. `sigma_floor` is a THRESHOLD and rule 8 applies: derive it from the measured
distribution of realised per-symbol volatility on real data and record the derivation beside it. Do
not pick a round number. The floor exists to stop the inverse-vol sizer from taking an unbounded
position in a temporarily-still name, so the honest derivation is a low percentile of the measured
distribution, and which percentile must itself be justified by what position size it implies.

## D4 — efficiency ratio

    efficiency_ratio = abs(close[t] - close[t-n]) / sum(abs(diff(close))[t-n+1 : t+1])

Session-bounded via `day_offsets`. Complements `amihud_illiquidity` (which exists and is dead).
Ranges [0, 1]; 1 is a straight line, near 0 is pure churn.

## D5 — IC, rank IC, and decay. None of this exists.

Confirmed by grep: zero hits in `src/` for `spearman`, `rank_ic`, `information_coefficient`,
`decay`, or `half_life`. The only near-hits are unrelated (`sharpe_standard_error(adjust_autocorr=)`,
`ewma_volatility_ann(halflife=)`).

New module `research/ic.py`:

- `information_coefficient(feature, fwd, day_offsets, *, method="pearson"|"spearman")` — computed
  CROSS-SECTIONALLY per bar, then aggregated across bars. Not pooled: pooling across time and
  symbols conflates a cross-sectional signal with a time-series one, and this program's live
  candidate is cross-sectional.
- `ic_decay(feature, close, day_offsets, *, horizons)` — IC at each horizon, plus the fitted
  half-life. Returns the horizon grid and the IC at each, never a single summary number.

**The SE on any IC aggregate must come from the corrected overlap-aware machinery in
`specs/overlap_se.md`, not a naive iid formula.** Overlapping forward returns make the naive SE
wrong in exactly the way L7/L8 documents. This spec must not land before that one.

## D6 — wire `features/market.py` into the research harness

Beta-residual return, breadth, dispersion, signed volume, CLV and the VIX ratio become reachable
from the Phase E sweep. This is plumbing, not new computation — but note `vwap_reversion.py:149`
hand-rolls its own `bars_since_open` instead of importing `market.py`'s, and that duplication is
removed as part of the wiring.

Also note `median_pairwise_correlation` (`market.py:337`) is named by
`specs/portfolio_vol_target.md:57` as the rho source while `backtest/portfolio.py` takes rho as a
scalar parameter. That is a live seam between two specs and ONE agent must own both sides of it.

## D7 — rename the volume z-score concept layer

`volume_zscore` keeps its function name. The CONCEPT layer — docs, verdict text, any strategy
parameter naming — becomes `abnormal_volume_activity`.

Reason, and it is not cosmetic: without order-level data a 1-minute volume spike is abnormal
activity, not proven institutional flow. The current framing asserts a mechanism the data cannot
support, and this program has already killed one strategy built on that assertion.

## What must NOT happen

`features/calibrate.py` (`calibrate_null`, `suggest_threshold`) is **100% dead outside its own
tests** — every rule-8 threshold in production was instead derived by standalone
`scripts/calibrate_*.py`. Do not add a ninth threshold derived by yet another one-off script. Either
route new derivations through `features/calibrate.py`, or record explicitly why it does not fit
(its `estimator` is a 1-D price -> scalar callable, which is incompatible with the 2-D rolling
contract `persistence.null_distribution` uses — that mismatch is itself worth fixing).

## Test obligations

Dual independent suites per rule 1, from this spec alone.

1. `stitch_overnight_gaps` leaves within-session log-returns EXACTLY unchanged, bit for bit.
2. It sets the cross-session log-difference to exactly 0.0 at every session boundary.
3. `NaN` in `close` produces `NaN` in the stitched output at the same index — the stitch never
   manufactures a bar (rule 6). Assert on a fixture with interior NaNs.
4. `rolling_hurst` on the stitched path with `window=390` returns a MAJORITY-finite result on a
   multi-session fixture, where the same call day-bounded on the unstitched path returns
   all-NaN. Use IRREGULAR sessions (60, 105, 375) — this is the arithmetic-impossibility
   regression test and it must fail loudly if someone reintroduces day-bounding.
5. `breakout_strength > 0` is identical to `breakout_up` on random data, elementwise.
6. `breakout_strength` is signed and non-zero when price is below the level.
7. Garman-Klass and Rogers-Satchell match hand-computed values on a small fixture, propagate NaN
   on the `parkinson_volatility` convention, and respect `day_offsets`.
8. `sigma_floor` has a recorded derivation from measured data beside its value; the value is not a
   round number chosen by hand.
9. `efficiency_ratio` is exactly 1.0 on a monotone ramp and near 0 on a zig-zag with equal steps.
10. `information_coefficient` with `method="spearman"` on a perfectly rank-monotone fixture is
    exactly 1.0, and -1.0 when reversed.
11. IC is computed cross-sectionally per bar and NOT pooled — construct a fixture where the pooled
    and cross-sectional answers DIFFER, and assert the cross-sectional one. A fixture where both
    agree tests nothing.
12. `ic_decay` returns the full horizon grid, not a scalar, and its half-life on a synthetic
    exponentially-decaying IC matches the known value.
13. IC aggregate SEs come from the overlap-aware path — assert they differ from the naive iid SE
    on an overlapping-horizon fixture.
14. `vwap_reversion` uses `market.bars_since_open` and no longer defines its own.
