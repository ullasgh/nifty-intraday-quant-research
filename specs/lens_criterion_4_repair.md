# Spec: repair kill criterion 4 (liquidity concentration) in `Lens`

**Status:** contract for TDD. Two independent test suites written from this document ALONE.
Neither author reads the other's file, and no implementation exists yet.

Same file as `specs/lens_criteria_6_7_repair.md` (`src/nifty_quant/research/lens.py`). A SINGLE
implementer handles both specs in one pass — never two agents at this file concurrently.

**Why this matters more than the other two:** criterion 4 is the SOLE surviving kill reason for
H2, the largest measured effect in the program (pooled -24.30 bps, t = -16.66, 8/8 years).
Once clip size is accounted for, H2 clears criteria 1 and 7 at a Rs 10L clip. So this criterion
alone decides whether H2 is dead.

---

## Defect 1 — it buckets on SHARE COUNT, not rupee turnover

`Lens.stability_report`, around `lens.py:468`:

```python
volume = self.panel.field("volume").astype(np.float64)
```

`volume` is raw share count. A name trading at Rs 1 lakh/share has tiny share volume and large
rupee turnover, so it is classified "least liquid" while being highly liquid. **Measured**: the
share-volume bottom decile is `BAJAJHLDNG, BOSCHLTD, GLAXO, MRF, OFSS, PAGEIND, PGHH, SHREECEM,
SOLARINDS` — high-share-price blue chips with a median rupee ADV of ~Rs 22 crore/day. The
rupee-turnover bottom decile is `BAJAJHLDNG, EMAMILTD, GICRE, GLAXO, NIACL, PGHH, SOLARINDS`.
**Only 4 of 12 names overlap.**

Liquidity must be **rupee turnover = close * volume**, the quantity `tradable_mask` and capacity
sizing already use.

## Defect 2 — full-sample lookahead

```python
# Compute volume deciles (causal, per row)
volume_quantiles = np.quantile(volume[np.isfinite(volume)], np.linspace(0, 1, 11))
```

`np.quantile` over the whole flattened array derives thresholds from the ENTIRE panel including
the future. The comment claims "causal, per row"; it is not. A full-sample quantile is exactly
the lookahead this repo's `causal_buckets` exists to prevent.

The correct machinery already exists and is unused here:
`expectancy.expectancy_by_liquidity_decile(feature, fwd, day_offsets, prior_adv, ...)`, whose
docstring states `prior_adv` "must already be lagged (strictly prior sessions). The function
does not re-derive it, and @causal cannot detect a leak in it — so document that loudly at the
call site."

---

## Required behaviour

1. **Liquidity is rupee turnover**: `close * volume`, float64, NaN-preserving (NaN means "no
   bar" and must never be forward-filled or zero-filled).
2. **`prior_adv` is STRICTLY PRIOR**: for session `s`, the per-symbol mean rupee turnover over
   sessions `[0, s)`. Session 0 is NaN for every symbol. It must never include session `s`
   itself. Derive session boundaries from `panel.day_offsets` — **never** a fixed
   375-bar stride (Muhurat ~60 bars, shortened ~105).
3. **Delegate the bucketing** to `expectancy.expectancy_by_liquidity_decile` rather than the
   inline `np.quantile` + per-cell loop. Do not reimplement it.
4. **`method="cross_sectional_rank"` is MANDATORY at the call site.** That function DEFAULTS to
   `method="expanding_quantile"`, which needs `min_history` rows per column and can never
   bucket a once-per-session signal — it silently returns tables whose `spread_bps` is `0.0`,
   indistinguishable from "no effect". This is the same trap that produced H2's original
   25,000x error, and it is the DEFAULT. Passing it explicitly is part of the contract.
5. **The concentration threshold stays a named module-level constant** with its derivation
   recorded beside it (see below). It must not be an inline literal.
6. Behaviour when a decile is empty or all-NaN is unchanged: it contributes nothing, never 0.0,
   never a crash.

## The threshold — pending a measured null distribution

Criterion 4 currently fires when the bottom decile is the argmax of `|spread|` AND
`max > 2 * median`. **That `2` is a hand-chosen constant, which CLAUDE.md rule 8 forbids** —
every cutoff must be derived from a measured null distribution on real data, with the derivation
recorded next to the value. Rule 8 exists because a reasoned-not-measured threshold once shipped
at `0.35` and produced 8,217 false positives.

`scripts/calibrate_concentration_threshold.py` is deriving that null now (500 within-session
permutation replicates). **Implement the threshold as a named constant read from one place**, so
substituting the derived value is a one-line change. Tests must NOT hardcode `2.0` as the
expected cutoff — parameterise on the module constant, so the suites survive its replacement.

**Measured on real data (2018-01-01..2025-07-31, all_equity), for reference:**

    measurement                                decile-0   max/median   fires at 2.0
    share volume, full-sample (current code)    -55.33      2.6880        yes
    rupee turnover, full-sample                 -60.20      2.9990        yes
    rupee turnover, strictly-prior causal       -51.81      2.0104        yes

Note the correct measurement lands at **2.0104 against a threshold of 2.0** — a 0.5% margin.
That is why the null distribution matters.

## Required tests

Two INDEPENDENT suites: `tests/test_lens_criterion4_repair_a.py`, `..._b.py`.

1. **Units regression guard — the one that would have caught this.** Build a panel with two
   symbols: one HIGH price / LOW share count, one LOW price / HIGH share count, with equal
   rupee turnover. Under the correct implementation they must land in the SAME liquidity
   decile. Under share-count bucketing they land at opposite ends. This test must fail against
   the current implementation.
2. **Causality guard.** A symbol whose turnover is tiny early and huge late must be bucketed by
   its STRICTLY PRIOR turnover, not by its full-sample rank. Construct it so a full-sample
   quantile and a strictly-prior one disagree, and assert the prior one wins. Must also fail
   against the current implementation.
3. Session 0 has no prior turnover -> NaN -> those cells are unbucketed, not silently decile 0.
4. `prior_adv` for session `s` excludes session `s` — assert directly on the computed array for
   a hand-checkable fixture, not only through the verdict.
5. An irregular-session panel (~60-bar Muhurat alongside a full session) buckets correctly;
   nothing assumes a fixed bars-per-session stride.
6. NaN turnover (no bar) propagates and is never forward-filled or treated as zero.
7. `method="cross_sectional_rank"` is actually used: a once-per-session signal must produce
   NON-ZERO decile spreads. A suite that would pass against ten silent `0.0` spreads has not
   tested anything — assert at least one decile spread is non-zero and finite.
8. Criterion 4 fires when the bottom decile is the argmax AND ratio exceeds the module
   constant; does NOT fire when the bottom decile is not the argmax, however large the ratio;
   does NOT fire when the ratio is below the constant. Parameterise on the constant.
9. All SEVEN criteria still reported, in order, after this change.
10. Determinism: same inputs and seed -> identical verdict text.

## Constraints

100% line+branch with pragma exclusion DISABLED. Fully annotated; ruff clean. Never touch
`data/`. float64 in motion. `present` and `tradable` stay DISTINCT. Reuse existing `Lens`
fixtures rather than inventing parallel ones.
