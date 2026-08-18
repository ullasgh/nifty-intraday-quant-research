"""Calibrate the hard-coded ``2.0`` in ``Lens.verdict()``'s concentration kill criterion
(criterion 4, ``research/lens.py`` lines ~640-672) against a MEASURED null distribution,
per CLAUDE.md rule 8 ("no threshold is a hand-chosen constant") -- **v2, CORRECTED liquidity
bucketing**.

Why v2, not a patch to v1
--------------------------
``scripts/calibrate_concentration_threshold.py`` (v1) is a faithful port of the CURRENT
production ``_liquidity_deciles`` in ``research/lens.py`` lines 470-484: raw SHARE VOLUME,
bucketed by full-sample (whole-panel, including the future) ``np.quantile`` edges via
``searchsorted``. That is exactly the code ``scripts/recon_h2_liquidity_units.py`` shows to
be defective two ways: (a) share volume is the wrong quantity (rupee turnover is what
"liquid" means), and (b) full-sample quantiles are a lookahead violation dressed as
"causal, per row".

Critically, full-sample pooled deciles do NOT have equal per-row membership: a symbol whose
volume sits consistently above/below the panel-wide quantile edges can occupy the same
decile for the ENTIRE sample, so decile sizes (and hence the achievable dispersion of
decile-spread statistics) differ structurally from cross-sectional-rank deciles, which by
construction hold ~1/10 of that row's finite names every single row. The null of
``max(|decile spread|) / median(|decile spread|)`` is therefore a different object under
each bucketing rule. Applying v1's p95 to the corrected statistic would itself be a rule-8
violation (an unmeasured threshold) dressed as a rule-8 fix.

v1 is left completely untouched and is run separately for comparison; this script is a copy
with ONLY the liquidity-decile construction changed, per the task spec:

1. Liquidity = rupee turnover, ``close * volume``, float64, NaN-preserving (NaN means "no
   bar"; never forward-filled, never zero-filled -- CLAUDE.md rule 6).
2. Strictly prior: for session ``s``, the per-symbol MEAN rupee turnover over sessions
   ``[0, s)``; session 0 is NaN. Session bounds come from ``panel.day_offsets`` (CLAUDE.md
   rule 5 -- never a fixed 375-bar stride). ``_prior_session_mean`` below is copied verbatim
   from ``scripts/recon_h2_liquidity_units.py`` (not reimplemented).
3. Bucket causally by CROSS-SECTIONAL RANK of that strictly-prior quantity, not by pooled
   quantiles: ``nifty_quant.features.core.cross_sectional_rank`` then
   ``clip((rank * 10).astype(int), 0, 9)`` -- copied verbatim from
   ``scripts/recon_h2_liquidity_units.py``'s ``production_geometry`` branch. Decile 0 is
   still "bottom" (least liquid 10% by rank that row), matching production's semantics and
   v1's "bottom decile" labelling.

Everything else is IDENTICAL to v1: ``N_BUCKETS_PER_DECILE = 5`` (production is 10 liquidity
deciles x 5 feature quintiles), the same within-session cross-symbol permutation null, the
same replicate count and seeding convention, the same ``conditional_expectancy`` call
(``method="cross_sectional_rank"``, ``session_bounded=True``), and above all the same
PRODUCTION median convention: ``sorted_abs[len(sorted_abs) // 2]``, the UPPER median
(``research/lens.py`` line 665), NOT ``np.median`` (which averages the 5th/6th values of a
10-wide sorted list and inflates the ratio -- this exact substitution already produced a
wrong reading on this question, per ``scripts/recon_h2_liquidity_units.py``'s ``_report``).

Method
------
1. Load the real ``all_equity`` panel, 2018-01-01 .. 2025-07-31 (last 12 months held out).
2. Build ``return_1`` (the real feature) and its horizon-1 forward return.
3. Build rupee turnover, its strictly-prior per-symbol session mean, and the resulting
   causal cross-sectional-rank liquidity deciles (replicate-invariant -- computed once).
4. Generate the null exactly as v1 does: for each replicate, permute ``return_1`` ACROSS
   SYMBOL COLUMNS, independently per session (one random column permutation per session,
   applied to every row of that session). This destroys any association between a symbol's
   feature values and its liquidity decile while preserving session structure, the
   cross-sectional distribution of feature values at every row, and the real decile sizes.
5. For each replicate, compute the per-decile expectancy spread via
   ``nifty_quant.research.expectancy.conditional_expectancy`` (``n_buckets=5,
   method="cross_sectional_rank", session_bounded=True``) for each of the ten liquidity
   deciles, mirroring ``research/lens.py`` lines 485-515.
6. Apply the production criterion-4 logic (mirrored from ``research/lens.py`` lines
   645-672) with the threshold as a free parameter to get the null distribution of the
   ratio, the argmax-is-bottom rate, and the joint false-positive rate at any threshold.

Usage
-----
    .venv/bin/python scripts/calibrate_concentration_threshold_v2.py \\
        --n-replicates 500 --seed 42

All RNG draws come from a single ``numpy.random.default_rng(seed)`` seeded with a literal
int passed on the command line (default 42, reported in the output). float64 throughout
(CLAUDE.md rule 3).
"""

from __future__ import annotations

import argparse
import datetime
import time

import numpy as np

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.features import core as core_features
from nifty_quant.research import expectancy
from nifty_quant.universe.static import load_universe

RESEARCH_START = datetime.date(2018, 1, 1)
RESEARCH_END = datetime.date(2025, 7, 31)  # last 12 months held out; never read past this
HORIZON = 1
N_DECILES = 10
N_BUCKETS_PER_DECILE = 5
HARDCODED_THRESHOLD = 2.0  # the value under review
OBSERVED_H2_STATISTIC = 2.1189  # corrected observed ratio, reported by the caller


def _prior_session_mean(values: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Per-symbol mean of ``values`` over STRICTLY PRIOR sessions, broadcast to each row.

    Session ``s`` gets the mean over sessions ``[0, s)``; session 0 is NaN. Copied verbatim
    from ``scripts/recon_h2_liquidity_units.py::_prior_session_mean`` (not reimplemented),
    per the task spec. Session bounds come from ``day_offsets`` -- never a fixed
    bars-per-session stride (CLAUDE.md rule 5).
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


def _liquidity_deciles(rupee_turnover: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """CORRECTED liquidity-decile assignment: strictly-prior per-symbol mean rupee
    turnover, bucketed by causal cross-sectional rank -- copied (not reimplemented) from
    ``scripts/recon_h2_liquidity_units.py``'s ``production_geometry`` branch
    (``cross_sectional_rank`` then ``clip((rank * 10).astype(int), 0, 9)``). Decile 0 is
    "bottom" (least liquid 10% by rank, that row), matching production's semantics and v1's
    labelling. Returns ``-1`` wherever the strictly-prior quantity (hence the rank) is
    undefined -- session 0, or fewer than ``cross_sectional_rank``'s ``min_names`` finite
    names that row.
    """
    prior_rupee = _prior_session_mean(rupee_turnover, day_offsets)
    ranks = core_features.cross_sectional_rank(prior_rupee)
    deciles = np.full(ranks.shape, -1, dtype=np.int8)
    ok = np.isfinite(ranks)
    deciles[ok] = np.clip((ranks[ok] * N_DECILES).astype(np.int64), 0, N_DECILES - 1).astype(
        np.int8
    )
    return deciles


def _permute_within_sessions(
    feature: np.ndarray, day_offsets: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute ``feature`` across symbol columns, independently per session, using one
    random column permutation per session applied to every row of that session. Forward
    returns, liquidity (hence decile membership), and session boundaries are never touched
    -- only which symbol's feature value is read on a given row changes. This is the "no
    real concentration" null construction the task specifies: it destroys any
    symbol-liquidity association while preserving session structure, the cross-sectional
    distribution of feature values at each row, and the real decile sizes.
    """
    n_rows, n_symbols = feature.shape
    n_sessions = len(day_offsets) - 1
    permuted = np.empty_like(feature)
    session_perms = np.argsort(rng.random((n_sessions, n_symbols)), axis=1)
    for day in range(n_sessions):
        start, end = int(day_offsets[day]), int(day_offsets[day + 1])
        permuted[start:end, :] = feature[start:end, session_perms[day]]
    return permuted


def _decile_spreads(
    feature: np.ndarray,
    fwd: expectancy.ForwardReturns,
    liquidity_deciles: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
) -> dict[int, float]:
    """Per-liquidity-decile expectancy spread_bps, mirroring ``research/lens.py`` lines
    485-515 exactly: same masking scheme, same ``conditional_expectancy`` call with
    ``n_buckets=5, method="cross_sectional_rank"`` -- the real production statistic, not a
    reimplementation. Only deciles with at least one member row are included, matching
    ``Lens.stability()``'s ``if np.any(mask):`` guard.

    This is the reference (slow) form: it passes the full ``(n_rows, n_symbols)`` array to
    ``conditional_expectancy`` with non-members NaN'd out, exactly as ``Lens.stability()``
    does. ``_decile_spreads_fast`` below is a verified-equivalent, much cheaper form used
    for the actual replicate loop; this function is what it is checked against.
    """
    spreads: dict[int, float] = {}
    for decile in range(N_DECILES):
        mask = liquidity_deciles == decile
        if not np.any(mask):
            continue
        liq_feature = np.where(mask, feature, np.nan)
        liq_fwd_values = np.where(mask, fwd.values, np.nan)
        n_defined = int(np.sum(np.any(np.isfinite(liq_fwd_values), axis=1)))
        table = expectancy.conditional_expectancy(
            liq_feature,
            expectancy.ForwardReturns(
                values=liq_fwd_values,
                horizon=horizon,
                session_bounded=True,
                n_defined=n_defined,
                n_nan_tail=0,
            ),
            day_offsets,
            n_buckets=N_BUCKETS_PER_DECILE,
            method="cross_sectional_rank",
            feature_name="null_permuted_return_1",
        )
        spreads[decile] = table.spread_bps
    return spreads


class _DecileLayout:
    """Precomputed, per-decile, REPLICATE-INVARIANT compaction of a liquidity decile's
    membership down to its own (narrow) column width.

    Ranking within ``cross_sectional_rank`` (``research/expectancy.py``'s
    ``causal_buckets``) is computed per row purely from the finite values present in that
    row -- it does not depend on which column position a value sits in, nor on the total
    row width, only on the multiset of finite values and their count. Padding a decile's
    ~10%-of-149-symbols membership out to the full 149-column width (as
    ``Lens.stability()`` and ``_decile_spreads`` above do) therefore feeds
    ``conditional_expectancy`` ~10x more all-NaN column data than the ranking calculation
    ever uses, at a real cost: scipy's ``rankdata`` sort is dominated by row width, and this
    array is fed to it once per decile PER REPLICATE.

    Since the liquidity-decile membership mask and the (unpermuted) forward returns are
    identical across every replicate -- only the permuted feature changes -- the expensive
    part of this compaction (one ``argsort`` per decile establishing where each row's
    member columns are) is done ONCE here, not once per replicate. Each replicate then only
    needs a cheap ``take_along_axis`` gather of its own permuted feature into the
    precomputed layout before calling the same ``conditional_expectancy`` as
    ``_decile_spreads``, now on a ``(n_rows, width)`` array with ``width ~= n_symbols / 10``
    instead of ``(n_rows, n_symbols)``.

    Verified bit-identical to ``_decile_spreads`` (the padded, lens.py-faithful form) on a
    real sample by the ``main()`` cross-check before use in the replicate loop.
    """

    __slots__ = ("sort_idx", "width", "valid", "compact_fwd", "n_defined")

    def __init__(
        self,
        sort_idx: np.ndarray,
        width: int,
        valid: np.ndarray,
        compact_fwd: np.ndarray,
        n_defined: int,
    ) -> None:
        self.sort_idx = sort_idx  # (n_rows, width) int, argsort(~mask) per row, PRE-TRUNCATED
        # to the first `width` columns -- take_along_axis's cost scales with the indices
        # array's own shape, so truncating before every replicate's gather (rather than
        # gathering all n_symbols columns and slicing afterwards) is what actually realizes
        # this layout's speedup; see _precompute_decile_layouts.
        self.width = width  # max member count across rows for this decile
        self.valid = valid  # (n_rows, width) bool, True where the gathered column is a
        # real decile member (not padding from a non-member column further down the
        # sorted-by-mask order)
        self.compact_fwd = compact_fwd  # (n_rows, width), already NaN'd outside membership
        self.n_defined = n_defined


def _precompute_decile_layouts(
    liquidity_deciles: np.ndarray, fwd: expectancy.ForwardReturns
) -> dict[int, _DecileLayout]:
    """Build the replicate-invariant ``_DecileLayout`` for every liquidity decile that has
    at least one member row (matching ``Lens.stability()``'s ``if np.any(mask):`` guard).
    Call once, before the replicate loop.
    """
    layouts: dict[int, _DecileLayout] = {}
    for decile in range(N_DECILES):
        mask = liquidity_deciles == decile
        if not np.any(mask):
            continue
        width = int(mask.sum(axis=1).max())
        sort_idx_full = np.argsort(~mask, axis=1, kind="stable")
        sort_idx = sort_idx_full[:, :width]
        valid = np.take_along_axis(mask, sort_idx, axis=1)
        compact_fwd = np.take_along_axis(fwd.values, sort_idx, axis=1)
        compact_fwd = np.where(valid, compact_fwd, np.nan)
        n_defined = int(np.sum(np.any(np.isfinite(compact_fwd), axis=1)))
        layouts[decile] = _DecileLayout(sort_idx, width, valid, compact_fwd, n_defined)
    return layouts


def _decile_spreads_fast(
    feature: np.ndarray,
    layouts: dict[int, _DecileLayout],
    day_offsets: np.ndarray,
    horizon: int,
) -> dict[int, float]:
    """Same statistic as ``_decile_spreads`` (same ``conditional_expectancy`` call, same
    arguments), computed from the precomputed ``_DecileLayout``s instead of re-deriving the
    liquidity mask and re-sorting on every call. See ``_DecileLayout`` for why this is
    equivalent, not merely similar.
    """
    spreads: dict[int, float] = {}
    for decile, layout in layouts.items():
        gathered = np.take_along_axis(feature, layout.sort_idx, axis=1)
        # Columns beyond a row's true member count are gathered from non-member columns
        # further down the sorted-by-mask order (real, possibly finite, feature values
        # that do NOT belong to this decile) -- `layout.valid` is exactly the decile
        # membership mask in this compacted layout, so it must be applied here too, not
        # just to `compact_fwd`.
        compact_feature = np.where(layout.valid, gathered, np.nan)
        table = expectancy.conditional_expectancy(
            compact_feature,
            expectancy.ForwardReturns(
                values=layout.compact_fwd,
                horizon=horizon,
                session_bounded=True,
                n_defined=layout.n_defined,
                n_nan_tail=0,
            ),
            day_offsets,
            n_buckets=N_BUCKETS_PER_DECILE,
            method="cross_sectional_rank",
            feature_name="null_permuted_return_1",
        )
        spreads[decile] = table.spread_bps
    return spreads


class ConcentrationRecord:
    """Everything needed to recompute criterion 4's fired/not-fired verdict at ANY
    candidate threshold post-hoc, without re-deriving the per-decile spreads. Kept as a
    tiny plain object (not a raw bool) specifically so the joint false-positive rate can
    be evaluated at both 2.0 and the derived p95 threshold from a single pass over the
    replicates.
    """

    __slots__ = ("only_bottom_nonzero", "ratio", "bottom_is_argmax")

    def __init__(self, only_bottom_nonzero: bool, ratio: float | None, bottom_is_argmax: bool):
        self.only_bottom_nonzero = only_bottom_nonzero
        self.ratio = ratio
        self.bottom_is_argmax = bottom_is_argmax

    def fired(self, threshold: float) -> bool:
        """Mirrors ``Lens.verdict()``'s criterion-4 boolean (``research/lens.py`` lines
        652-672) at the given ``threshold`` in place of the hard-coded ``2.0``.
        """
        if self.only_bottom_nonzero:
            return True
        return self.ratio is not None and self.ratio > threshold and self.bottom_is_argmax


def _concentration_stat(spread_by_decile: dict[int, float]) -> ConcentrationRecord:
    """Mirrors ``Lens.verdict()``'s criterion-4 logic (``research/lens.py`` lines
    645-672) -- that logic lives inline inside ``verdict()``, not as an importable
    function, so it is reproduced here verbatim (with the hard-coded ``2.0`` lifted out
    into ``ConcentrationRecord.fired(threshold)`` so it can be swept post-hoc) rather than
    imported.

    ``ratio`` is max(|spread|) / median(|spread|) over nonzero-spread deciles (sorted
    ascending, ``[len // 2]`` index -- matching the production "median" exactly, including
    its even-length behaviour, i.e. the UPPER median, not ``np.median``); it is None
    whenever the production code could never define a ratio (fewer than 2 nonzero deciles,
    or a zero median).
    """
    bottom = spread_by_decile.get(0)
    if bottom is None or bottom == 0.0:
        return ConcentrationRecord(only_bottom_nonzero=False, ratio=None, bottom_is_argmax=False)

    edges = list(spread_by_decile.values())
    nonzero = [e for e in edges if e != 0.0]
    if len(nonzero) == 1:
        # Only the bottom decile shows any edge at all (mirrors lens.py's separate branch).
        return ConcentrationRecord(only_bottom_nonzero=True, ratio=None, bottom_is_argmax=True)

    sorted_abs = sorted(abs(e) for e in nonzero)
    max_edge = sorted_abs[-1]
    median_edge = sorted_abs[len(sorted_abs) // 2]  # UPPER median, matching lens.py:665
    bottom_is_argmax = abs(bottom) == max_edge
    ratio = max_edge / median_edge if median_edge > 0 else None
    return ConcentrationRecord(
        only_bottom_nonzero=False, ratio=ratio, bottom_is_argmax=bottom_is_argmax
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start", type=str, default=RESEARCH_START.isoformat(), help="YYYY-MM-DD"
    )
    parser.add_argument("--end", type=str, default=RESEARCH_END.isoformat(), help="YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)
    if end > RESEARCH_END:
        raise ValueError(
            f"--end {end} is past the holdout boundary {RESEARCH_END}; refusing to read it."
        )

    rng = np.random.default_rng(args.seed)

    universe = load_universe("all_equity")
    print(f"universe: {len(universe.symbols)} symbols")

    t0 = time.time()
    spec = PanelSpec(
        freq="1",
        fields=("close", "volume"),
        symbols=universe.symbols,
        start=start,
        end=end,
    )
    panel = load_panel(spec)
    print(
        f"panel loaded in {time.time() - t0:.1f}s: n_rows={panel.n_rows()} "
        f"n_symbols={panel.n_symbols()} {panel.dates[0]}..{panel.dates[-1]}"
    )

    close = panel.field("close").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    day_offsets = panel.day_offsets

    return_1 = core_features.log_returns(close, day_offsets=day_offsets)
    fwd = expectancy.forward_returns(close, day_offsets, HORIZON)

    # Rupee turnover: close * volume, float64, NaN-preserving (NaN wherever either input
    # is NaN, i.e. "no bar" -- never forward-filled or zero-filled, CLAUDE.md rule 6).
    rupee_turnover = close * volume
    del close, volume  # only needed to build return_1, fwd, and rupee_turnover

    liquidity_deciles = _liquidity_deciles(rupee_turnover, day_offsets)
    print(
        "liquidity deciles: rupee turnover, strictly-prior per-symbol session mean "
        "(_prior_session_mean, copied from recon_h2_liquidity_units.py), causal "
        "cross-sectional-rank bucketing (cross_sectional_rank + clip(rank*10,0,9), copied "
        "from recon_h2_liquidity_units.py's production_geometry branch)"
    )
    decile_counts = {d: int(np.sum(liquidity_deciles == d)) for d in range(N_DECILES)}
    print(f"liquidity-decile (row, symbol) cell counts: {decile_counts}")
    print(f"undefined (-1, e.g. session 0 / <min_names finite that row) cell count: "
          f"{int(np.sum(liquidity_deciles == -1))}")

    layouts = _precompute_decile_layouts(liquidity_deciles, fwd)
    widths = {d: layout.width for d, layout in layouts.items()}
    print(f"precomputed decile layouts (compacted column widths): {widths}")

    # Verify the fast (precomputed-layout) path against the slow, lens.py-faithful
    # reference on the FIRST replicate's actual permuted feature -- not just on synthetic
    # unit-test data -- before trusting it for the other (n_replicates - 1) replicates.
    first_permuted = _permute_within_sessions(return_1, day_offsets, rng)
    reference_spreads = _decile_spreads(
        first_permuted, fwd, liquidity_deciles, day_offsets, HORIZON
    )
    fast_spreads = _decile_spreads_fast(first_permuted, layouts, day_offsets, HORIZON)
    if set(reference_spreads) != set(fast_spreads) or any(
        abs(fast_spreads[d] - reference_spreads[d]) > 1e-9 for d in reference_spreads
    ):
        raise AssertionError(
            "_decile_spreads_fast does not match the reference _decile_spreads on real "
            f"data: reference={reference_spreads} fast={fast_spreads}"
        )
    print("fast decile-spread path verified bit-identical to the reference path "
          "on real data (replicate 1)")

    n_replicates = args.n_replicates
    records: list[ConcentrationRecord] = [_concentration_stat(reference_spreads)]

    t_loop = time.time()
    for rep in range(1, n_replicates):
        permuted = _permute_within_sessions(return_1, day_offsets, rng)
        spreads = _decile_spreads_fast(permuted, layouts, day_offsets, HORIZON)
        records.append(_concentration_stat(spreads))
        if (rep + 1) % max(1, n_replicates // 10) == 0:
            elapsed = time.time() - t_loop
            print(f"  replicate {rep + 1}/{n_replicates} ({elapsed:.1f}s elapsed)")

    print(f"\ncompleted {n_replicates} replicates in {time.time() - t_loop:.1f}s")

    ratios_arr = np.array(
        [r.ratio for r in records if r.ratio is not None], dtype=np.float64
    )
    n_ratio = len(ratios_arr)
    n_only_bottom_nonzero = sum(1 for r in records if r.only_bottom_nonzero)
    bottom_argmax_count = sum(1 for r in records if r.bottom_is_argmax)
    print(
        f"{n_ratio}/{n_replicates} replicates yielded a defined ratio; "
        f"{n_only_bottom_nonzero} had only the bottom decile nonzero (auto-fires at any "
        f"threshold); remainder had a zero-median nonzero set"
    )

    if n_ratio == 0:
        raise RuntimeError("no replicate produced a defined ratio; cannot calibrate")

    percentiles = [50, 75, 90, 95, 99]
    pct_values = {p: float(np.percentile(ratios_arr, p)) for p in percentiles}
    mean_ratio = float(np.mean(ratios_arr))
    median_ratio = float(np.median(ratios_arr))
    max_ratio = float(np.max(ratios_arr))
    p95_threshold = pct_values[95]

    pct_of_2 = float(100.0 * np.mean(ratios_arr <= HARDCODED_THRESHOLD))
    pct_of_observed = float(100.0 * np.mean(ratios_arr <= OBSERVED_H2_STATISTIC))

    joint_fp_rate_hardcoded = sum(r.fired(HARDCODED_THRESHOLD) for r in records) / n_replicates
    joint_fp_rate_p95 = sum(r.fired(p95_threshold) for r in records) / n_replicates

    print("\n=== [v2, corrected liquidity] Null distribution of "
          "max(|spread_bps|) / median(|spread_bps|) ===")
    print(f"seed={args.seed}  n_replicates={n_replicates}  n_with_defined_ratio={n_ratio}")
    print(f"mean={mean_ratio:.4f}  median={median_ratio:.4f}  max_observed={max_ratio:.4f}")
    for p in percentiles:
        print(f"  p{p} = {pct_values[p]:.4f}")

    print(f"\nhardcoded threshold 2.0 sits at the {pct_of_2:.1f}th percentile of the null "
          f"ratio distribution")
    print(f"observed H2 statistic {OBSERVED_H2_STATISTIC} sits at the {pct_of_observed:.1f}th "
          f"percentile of the null ratio distribution")
    print(f"derived p95 threshold (5% FP rate on the ratio alone) = {p95_threshold:.4f}")
    print(f"H2's observed {OBSERVED_H2_STATISTIC} "
          f"{'EXCEEDS' if OBSERVED_H2_STATISTIC > p95_threshold else 'DOES NOT EXCEED'} "
          f"the derived p95 threshold")
    print(f"bottom-decile-is-argmax rate under the null = "
          f"{100.0 * bottom_argmax_count / n_replicates:.1f}% (n={n_replicates}; "
          f"~10% expected by chance alone with 10 deciles)")
    print(f"JOINT false-positive rate at threshold=2.0 (ratio>2 AND bottom is argmax): "
          f"{joint_fp_rate_hardcoded:.4f} "
          f"({sum(r.fired(HARDCODED_THRESHOLD) for r in records)}/{n_replicates})")
    print(f"JOINT false-positive rate at threshold={p95_threshold:.4f} (derived p95): "
          f"{joint_fp_rate_p95:.4f} "
          f"({sum(r.fired(p95_threshold) for r in records)}/{n_replicates})")


if __name__ == "__main__":
    main()
