"""Calibrate the hard-coded ``2.0`` in ``Lens.verdict()``'s concentration kill criterion
(criterion 4, ``research/lens.py`` lines ~640-672) against a MEASURED null distribution,
per CLAUDE.md rule 8 ("no threshold is a hand-chosen constant").

Criterion 4 fires when, across the ten liquidity-decile expectancy tables
``Lens.stability()`` builds, the BOTTOM decile has both (a) the largest absolute spread
among deciles with a nonzero spread, and (b) that spread more than ``2x`` the median of
the nonzero deciles' absolute spreads. The ``2`` was reasoned, never measured -- exactly
the failure mode CLAUDE.md rule 8 calls out by name (the ``adjustment_audit`` 0.35
incident). This script measures what that ratio actually looks like when there is NO
real concentration, so the threshold can be picked from data instead of intuition.

Method
------
1. Load the real ``all_equity`` panel, 2018-01-01 .. 2025-07-31 (the last 12 months are
   held out and never read).
2. Build the real ``return_1`` feature and its horizon-1 forward return, and the same
   10-way liquidity-decile partition ``Lens.stability()`` builds from raw volume
   (``_liquidity_deciles`` below is a vectorized, verified-equivalent port of
   ``research/lens.py`` lines 470-484 -- the O(n_rows * n_symbols) pure-Python double loop
   there is infeasible to run once per replicate, let alone 500+ times; the vectorized
   form is checked bit-for-bit against the original loop on a real sample before use).
3. Generate the null: for each replicate, permute the ``return_1`` feature ACROSS SYMBOL
   COLUMNS, independently per trading session (one random column permutation per session,
   applied to every row of that session). This destroys any association between a
   symbol's feature values and its liquidity decile (a symbol's own volume history, hence
   its decile membership, is untouched) while preserving the session structure, the
   cross-sectional distribution of feature values at every row, and the real decile
   sizes -- i.e. a draw from "no concentration" by construction.
4. For each replicate, compute the per-decile expectancy spread via
   ``nifty_quant.research.expectancy.conditional_expectancy`` -- the exact function
   ``Lens.stability()`` calls (not reimplemented), with the identical arguments
   (``n_buckets=5``, ``method="cross_sectional_rank"``, ``session_bounded=True``) -- for
   each of the ten liquidity deciles, exactly mirroring ``research/lens.py`` lines
   485-515.
5. Apply the production criterion-4 logic (mirrored from ``research/lens.py`` lines
   645-672, since it lives inline inside ``Lens.verdict()`` and is not a standalone
   function to import) with the threshold as a free parameter, to get the null
   distribution of the ratio, the argmax-is-bottom rate, and the joint false-positive
   rate at any candidate threshold.

Usage
-----
    .venv/bin/python scripts/calibrate_concentration_threshold.py \\
        --n-replicates 500 --seed 42

All RNG draws come from a single ``numpy.random.default_rng(seed)`` seeded with a literal
int passed on the command line (default 42, reported in the output). float64 throughout
(CLAUDE.md rule 3); volume/close are cast up from the float32-at-rest panel fields before
any arithmetic.
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
N_BUCKETS_PER_DECILE = 5
HARDCODED_THRESHOLD = 2.0  # the value under review


def _liquidity_deciles(volume: np.ndarray) -> np.ndarray:
    """Vectorized port of ``Lens.stability()``'s liquidity-decile assignment
    (``research/lens.py`` lines 470-484): the same global (row, symbol)-pooled volume
    deciles (``np.quantile(volume[finite], linspace(0, 1, 11))``) and the same
    ``np.searchsorted(edges[1:-1], vol, side="right")`` edge rule, ``-1`` for non-finite
    volume. Only the per-cell Python ``for i / for s`` double loop is replaced by a single
    vectorized ``searchsorted`` call -- ``searchsorted`` applied to an array argument is
    the identical scalar rule applied elementwise, not a different algorithm. Verified
    bit-identical to the original loop on a real sample by ``_check_liquidity_deciles``.
    """
    volume64 = np.asarray(volume, dtype=np.float64)
    finite = np.isfinite(volume64)
    quantiles = np.quantile(volume64[finite], np.linspace(0, 1, 11))
    deciles = np.full(volume64.shape, -1, dtype=np.int8)
    deciles[finite] = np.searchsorted(
        quantiles[1:-1], volume64[finite], side="right"
    ).astype(np.int8)
    return deciles


def _check_liquidity_deciles(volume: np.ndarray, deciles: np.ndarray, n_check_rows: int) -> None:
    """Cross-check ``_liquidity_deciles`` against the original per-cell loop from
    ``research/lens.py`` lines 470-484, on the first ``n_check_rows`` rows of real data.
    Raises AssertionError on any mismatch. This is what makes the vectorization above a
    verified reuse of the production statistic rather than an unverified reimplementation.
    """
    volume64 = np.asarray(volume, dtype=np.float64)
    finite_all = np.isfinite(volume64)
    quantiles = np.quantile(volume64[finite_all], np.linspace(0, 1, 11))
    check_rows = min(n_check_rows, volume64.shape[0])
    reference = np.full((check_rows, volume64.shape[1]), -1, dtype=np.int8)
    for i in range(check_rows):
        for s in range(volume64.shape[1]):
            vol = volume64[i, s]
            if np.isfinite(vol):
                reference[i, s] = np.searchsorted(quantiles[1:-1], vol, side="right")
    if not np.array_equal(reference, deciles[:check_rows]):
        raise AssertionError(
            "_liquidity_deciles does not match the original per-cell loop from "
            "research/lens.py lines 470-484 -- vectorization is not equivalent."
        )


def _permute_within_sessions(
    feature: np.ndarray, day_offsets: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute ``feature`` across symbol columns, independently per session, using one
    random column permutation per session applied to every row of that session. Forward
    returns, volume (hence liquidity-decile membership), and session boundaries are never
    touched -- only which symbol's feature value is read on a given row changes. This is
    the "no real concentration" null construction the task specifies: it destroys any
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
    volume_deciles: np.ndarray,
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
    for the actual 500+-replicate loop; this function is what it is checked against.
    """
    spreads: dict[int, float] = {}
    for decile in range(10):
        mask = volume_deciles == decile
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
    real sample by ``_check_decile_layout_equivalence`` before use in the replicate loop.
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
    volume_deciles: np.ndarray, fwd: expectancy.ForwardReturns
) -> dict[int, _DecileLayout]:
    """Build the replicate-invariant ``_DecileLayout`` for every liquidity decile that has
    at least one member row (matching ``Lens.stability()``'s ``if np.any(mask):`` guard).
    Call once, before the replicate loop.
    """
    layouts: dict[int, _DecileLayout] = {}
    for decile in range(10):
        mask = volume_deciles == decile
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
    its even-length behaviour); it is None whenever the production code could never define
    a ratio (fewer than 2 nonzero deciles, or a zero median).
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
    median_edge = sorted_abs[len(sorted_abs) // 2]
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
    del close  # only needed to build return_1 and fwd; free before the replicate loop

    volume_deciles = _liquidity_deciles(volume)
    _check_liquidity_deciles(volume, volume_deciles, n_check_rows=3000)
    print("liquidity-decile vectorization verified against research/lens.py's loop (3000 rows)")
    decile_counts = {d: int(np.sum(volume_deciles == d)) for d in range(10)}
    print(f"liquidity-decile (row, symbol) cell counts: {decile_counts}")

    layouts = _precompute_decile_layouts(volume_deciles, fwd)
    widths = {d: layout.width for d, layout in layouts.items()}
    print(f"precomputed decile layouts (compacted column widths): {widths}")

    # Verify the fast (precomputed-layout) path against the slow, lens.py-faithful
    # reference on the FIRST replicate's actual permuted feature -- not just on synthetic
    # unit-test data -- before trusting it for the other (n_replicates - 1) replicates.
    first_permuted = _permute_within_sessions(return_1, day_offsets, rng)
    reference_spreads = _decile_spreads(first_permuted, fwd, volume_deciles, day_offsets, HORIZON)
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

    joint_fp_rate_hardcoded = sum(r.fired(HARDCODED_THRESHOLD) for r in records) / n_replicates
    joint_fp_rate_p95 = sum(r.fired(p95_threshold) for r in records) / n_replicates

    print("\n=== Null distribution of max(|spread_bps|) / median(|spread_bps|) ===")
    print(f"seed={args.seed}  n_replicates={n_replicates}  n_with_defined_ratio={n_ratio}")
    print(f"mean={mean_ratio:.4f}  median={median_ratio:.4f}  max_observed={max_ratio:.4f}")
    for p in percentiles:
        print(f"  p{p} = {pct_values[p]:.4f}")

    print(f"\nhardcoded threshold 2.0 sits at the {pct_of_2:.1f}th percentile of the null "
          f"ratio distribution")
    print(f"derived p95 threshold (5% FP rate on the ratio alone) = {p95_threshold:.4f}")
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
