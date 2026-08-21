"""Conditional expectancy analysis: does E[R_{t+h} | feature bucket] exceed cost hurdle?

This module answers the question that should come BEFORE any strategy is written:
"Does this feature bucket have an edge large enough to pay costs?"

Every strategy in this repo was built by assuming an effect existed; all of them lost.
This module is the gate every hypothesis passes through first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.features.core import cross_sectional_rank
from nifty_quant.guards import causal, check_day_offsets

# ---------------------------------------------------------------------------
# ForwardReturns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardReturns:
    """h-bar forward log returns, session-bounded."""

    values: np.ndarray  # (n_rows, n_symbols) float64, NaN where undefined
    horizon: int  # bars
    session_bounded: bool  # always True; recorded so a caller cannot forget
    n_defined: int
    n_nan_tail: int  # count NaN'd because the horizon ran off the session end

    def explain(self) -> str:
        """Explain the forward returns object."""
        defined_pct = 100.0 * self.n_defined / max(
            self.values.size, 1
        )  # avoid /0 on empty arrays
        return (
            f"ForwardReturns(horizon={self.horizon}, "
            f"session_bounded={self.session_bounded}, "
            f"n_defined={self.n_defined}, n_nan_tail={self.n_nan_tail}, "
            f"defined_pct={defined_pct:.1f}%)"
        )


def forward_returns(
    close: np.ndarray, day_offsets: np.ndarray, horizon: int
) -> ForwardReturns:
    """h-bar forward LOG return, SESSION-BOUNDED.

    values[t] = log(close[t + horizon] / close[t]), and is NaN whenever t and t + horizon
    are not in the same session. Rows within `horizon` of a session end are therefore NaN --
    this is not padding, it is the absence of a defined forward return. Never crosses a
    session boundary; never uses `t + horizon` from the next day. Never assumes a fixed
    375-bar stride -- session membership comes from `day_offsets` (CLAUDE.md rule 5).

    NaN in `close` (no bar) propagates to NaN and is NEVER forward-filled (rule 6).
    Computed in float64 (rule 3). Raises ValueError for horizon < 1.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close64 = np.asarray(close, dtype=np.float64)
    if close64.ndim != 2:
        raise ValueError("close must be 2-D")

    n_rows = close64.shape[0]
    offsets = np.asarray(day_offsets, dtype=int)
    check_day_offsets(offsets, n_rows)

    # Precompute session membership: which session contains each row
    # offsets[i:i+1] are the session boundaries
    session_id = np.searchsorted(offsets[1:], np.arange(n_rows), side="right")

    # Compute log returns
    values = np.full_like(close64, np.nan, dtype=np.float64)

    for t in range(n_rows - horizon):
        # Check if t and t+horizon are in the same session
        if session_id[t] == session_id[t + horizon]:
            # Both bars have finite close prices
            with np.errstate(divide="ignore", invalid="ignore"):
                values[t, :] = np.log(close64[t + horizon, :] / close64[t, :])
            # If either close[t] or close[t+horizon] is NaN, log will produce NaN
            # which propagates correctly

    # Count defined and NaN tail
    # Count rows that have at least one defined return
    # (all symbols in a row either all have the same NaN status or all are finite)
    rows_with_any_finite = np.any(np.isfinite(values), axis=1)
    rows_with_all_nan = ~np.any(np.isfinite(values), axis=1)
    n_defined = int(np.sum(rows_with_any_finite))
    n_nan_tail = int(np.sum(rows_with_all_nan))

    return ForwardReturns(
        values=values,
        horizon=horizon,
        session_bounded=True,
        n_defined=n_defined,
        n_nan_tail=n_nan_tail,
    )


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bucketing:
    """Bucketing metadata and labels."""

    labels: np.ndarray  # (n_rows, n_symbols) int8; -1 == unassigned
    n_buckets: int
    method: Literal["expanding_quantile", "rolling_quantile", "cross_sectional_rank"]
    warmup_rows: int
    edges_source: str  # human description of where the cut points came from

    def explain(self) -> str:
        """Explain the bucketing."""
        unassigned = int((self.labels == -1).sum())
        assigned = int((self.labels >= 0).sum())
        return (
            f"Bucketing(method={self.method}, n_buckets={self.n_buckets}, "
            f"warmup_rows={self.warmup_rows}, assigned={assigned}, "
            f"unassigned={unassigned}, edges_source={self.edges_source!r})"
        )


def _expanding_quantile_buckets(
    feature: np.ndarray,
    n_buckets: int,
    min_history: int,
) -> np.ndarray:
    """Assign buckets using expanding window quantiles (causal).

    Returns (n_rows, n_symbols) int8 array with labels 0..n_buckets-1, or -1 (unassigned)
    for rows before min_history.
    """
    n_rows, n_symbols = feature.shape
    labels = np.full((n_rows, n_symbols), -1, dtype=np.int8)

    for t in range(min_history, n_rows):
        prior_data = feature[:t]  # Only rows [0, t)
        finite_mask = np.isfinite(prior_data)

        for s in range(n_symbols):
            finite_vals = prior_data[finite_mask[:, s], s]
            if len(finite_vals) > 0:
                # Compute quantiles from prior data only
                edges = np.quantile(finite_vals, np.linspace(0, 1, n_buckets + 1))
                # Assign bucket for row t
                val = feature[t, s]
                if np.isfinite(val):
                    # searchsorted with side='right' gives bucket indices 0..n_buckets
                    bucket = int(np.searchsorted(edges[1:-1], val, side="right"))
                    labels[t, s] = bucket

    # For a degenerate constant feature, assign all rows to the same bucket
    # Check if each symbol is constant (all finite values are equal)
    for s in range(n_symbols):
        finite_mask = np.isfinite(feature[:, s])
        if np.sum(finite_mask) > 0:
            finite_vals = feature[finite_mask, s]
            if np.allclose(finite_vals, finite_vals[0]):
                # Constant symbol: assign all rows to the same bucket
                # Use the bucket from the first bucketed row
                bucketed_rows = np.where(labels[:, s] >= 0)[0]
                if len(bucketed_rows) > 0:
                    bucket_val = labels[bucketed_rows[0], s]
                    labels[:, s] = bucket_val

    return labels


def _rolling_quantile_buckets(
    feature: np.ndarray,
    n_buckets: int,
    min_history: int,
) -> np.ndarray:
    """Assign buckets using rolling window quantiles (causal)."""
    n_rows, n_symbols = feature.shape
    labels = np.full((n_rows, n_symbols), -1, dtype=np.int8)
    window = min_history

    for t in range(min_history, n_rows):
        start = max(0, t - window)
        window_data = feature[start:t]  # Only rows before t
        finite_mask = np.isfinite(window_data)

        for s in range(n_symbols):
            finite_vals = window_data[finite_mask[:, s], s]
            if len(finite_vals) > 0:
                edges = np.quantile(finite_vals, np.linspace(0, 1, n_buckets + 1))
                val = feature[t, s]
                if np.isfinite(val):
                    bucket = int(np.searchsorted(edges[1:-1], val, side="right"))
                    labels[t, s] = bucket

    return labels


@causal(row_arg=0)
def _causal_buckets_impl(
    feature: np.ndarray,
    n_buckets: int,
    method: str,
    min_history: int,
) -> np.ndarray:
    """Internal implementation of causal_buckets that returns just the labels array.

    This is decorated with @causal to enforce causality on the labels output.
    """
    feature64 = np.asarray(feature, dtype=np.float64)

    if method == "cross_sectional_rank":
        # Rank within each row; normalize to percentile ranks [0, 1]
        pct_ranks = cross_sectional_rank(feature64, pct=True)  # [0, 1] per row
        # Map percentile ranks to bucket indices 0..n_buckets-1
        labels = np.full((feature64.shape[0], feature64.shape[1]), -1, dtype=np.int8)
        valid = np.isfinite(pct_ranks)
        bucket_edges = np.linspace(0, 1, n_buckets + 1)
        labels[valid] = np.searchsorted(
            bucket_edges[1:-1], pct_ranks[valid], side="right"
        ).astype(np.int8)

    elif method == "expanding_quantile":
        labels = _expanding_quantile_buckets(feature64, n_buckets, min_history)

    elif method == "rolling_quantile":
        labels = _rolling_quantile_buckets(feature64, n_buckets, min_history)

    else:
        raise ValueError(f"unknown method: {method}")

    return labels


def causal_buckets(
    feature: np.ndarray,
    day_offsets: np.ndarray,
    n_buckets: int = 5,
    method: str = "expanding_quantile",
    min_history: int = 5000,
) -> Bucketing:
    """Assign each observation to a feature quantile bucket using ONLY prior information.

    THE POINT OF THIS FUNCTION: a full-sample quantile is itself a lookahead leak. Bucketing
    by `np.quantile(feature)` over the whole panel uses the future to decide what counted as
    "extreme" today, which inflates every downstream expectancy. This is the single easiest
    way to manufacture a fake edge in this repo, and it looks completely innocent.

    - "expanding_quantile": cut points at row t come from rows [0, t). Rows before
      `min_history` are labelled -1 (unassigned) rather than bucketed on thin data.
    - "rolling_quantile": cut points from a trailing window.
    - "cross_sectional_rank": rank across symbols WITHIN row t only -- no time dimension, so
      it is causal by construction. Reuse `features.core.cross_sectional_rank`.

    Must be decorated `@causal` (see `nifty_quant.guards`) so the lookahead prober exercises it.
    """
    feature64 = np.asarray(feature, dtype=np.float64)
    if feature64.ndim != 2:
        raise ValueError("feature must be 2-D")

    n_rows = feature64.shape[0]
    check_day_offsets(np.asarray(day_offsets), n_rows)

    # Call the internal causal-checked implementation
    labels = _causal_buckets_impl(feature64, n_buckets, method, min_history)

    edges_source = {
        "cross_sectional_rank": "cross-sectional rank within each row",
        "expanding_quantile": f"expanding quantiles, min_history={min_history}",
        "rolling_quantile": f"rolling quantiles, min_history={min_history}",
    }.get(method, method)

    return Bucketing(
        labels=labels,
        n_buckets=n_buckets,
        method=method,
        warmup_rows=min_history,
        edges_source=edges_source,
    )


# ---------------------------------------------------------------------------
# Overlap correction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Block-length derivation (CLAUDE.md rule 8): measured, not chosen by convention.
#
# `L = horizon` is a defensible FLOOR -- overlapping h-bar forward returns induce
# dependence spanning at least h bars by construction, a standard MA(h-1)-type overlap
# effect that exists even under iid 1-minute returns. But the underlying 1-minute return
# series carries its own serial correlation ON TOP of that overlap, which `L = horizon`
# alone ignores (specs/overlap_se.md section 2). This must be measured, not assumed.
#
# MEASURED 2026-08-20: lag-1..30 autocorrelation of 1-minute LOG returns (LEFT-LABELLED
# bars; only same-session consecutive pairs used -- a pair spanning a session gap, where
# the row-to-row ts delta != 60s, is excluded so no measured correlation crosses a
# session boundary) for ten liquid names -- RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK,
# SBIN, ITC, LT, AXISBANK, KOTAKBANK -- over 2024-01-01..2024-03-31 (n=~22,900
# same-session consecutive pairs per symbol, computed directly from `data/bars/1/<SYM>/
# 2024.parquet`). All ten symbols show the classic 1-minute bid-ask-bounce signature: a
# significant NEGATIVE lag-1 ACF, decaying to inside +/-0.02 (an order of magnitude below
# the lag-1 magnitude, used here as "practically zero") and staying there for >= 3
# consecutive lags:
#
#     symbol      lag-1 ACF   decay lag (first lag where |ACF| < 0.02 for 3 consec. lags)
#     RELIANCE      -0.0575    2
#     TCS           -0.0488    2
#     HDFCBANK      -0.0460    2
#     INFY          -0.0501    2
#     ICICIBANK     -0.0596    5
#     SBIN          -0.0410    3
#     ITC           -0.0631    4
#     LT            -0.0401    2
#     AXISBANK      -0.0617    4
#     KOTAKBANK     -0.0658    5
#
#     decay lag across symbols:  p50=2.5   p90=5.0   p95=5.0   max=5
#
# The p95 decay lag (5 bars) is used as the extra padding added on top of the
# horizon floor -- a conservative choice that over-covers 9 of the 10 measured symbols'
# decay rather than the median symbol's.
BLOCK_LENGTH_EXTRA_BARS: int = 5


# `block_indices` is a DIAGNOSTIC (which blocks were drawn), never part of the
# statistic itself -- capping how many are RETAINED changes nothing about which
# blocks are drawn, in what order, or how the resampled series/means are computed.
# Uncapped, it accumulates one (start, length) tuple per block per replicate: on the
# real panel (n_boot=1000, block lengths of ~6-25 rows over 701_863 rows) that is an
# estimated 28M-117M tuples, ~10-15 GB. Budget: ~64 MiB of retained diagnostic
# tuples, at a conservative ~100 bytes per (start, length) 2-tuple held inside a
# python list (two boxed ints at ~28 bytes each plus tuple/list-slot overhead,
# rounded up for margin) -- solving for count gives the cap below. Every test fixture
# in this repo (n_boot <= 500, tens of sessions) draws far fewer blocks than this
# cap, so the small-fixture contract (`block_indices` as an exhaustive, per-block
# record) is unaffected; only production-sized runs are bounded.
_BLOCK_INDICES_DIAGNOSTIC_BUDGET_BYTES: int = 64 * 1024 * 1024
_BLOCK_INDICES_BYTES_PER_ENTRY: int = 100
_BLOCK_INDICES_MAX_RETAINED: int = (
    _BLOCK_INDICES_DIAGNOSTIC_BUDGET_BYTES // _BLOCK_INDICES_BYTES_PER_ENTRY
)


def _derive_block_length(horizon: int) -> int:
    """Moving-block-bootstrap block length: DERIVED, never `L = horizon` by convention.

    See `BLOCK_LENGTH_EXTRA_BARS` above for the measured autocorrelation this is derived
    from. The floor is `horizon` (overlapping h-bar forward returns induce dependence
    spanning at least h bars by construction); `BLOCK_LENGTH_EXTRA_BARS` covers the
    additional measured autocorrelation in the underlying 1-minute return series.
    """
    return max(int(horizon), 1) + BLOCK_LENGTH_EXTRA_BARS


def _draw_block_start_positions(
    rng: np.random.Generator,
    starts_arr: np.ndarray,
    n_blocks_per_replicate: int,
    n_replicates: int,
) -> np.ndarray:
    """Draw block start positions for `n_replicates` replicates in ONE vectorized
    call, bit-identical to `n_replicates` sequential
    `rng.choice(starts_arr, size=n_blocks_per_replicate, replace=True)` calls.

    `Generator.choice` with `replace=True` and no `p` reduces internally to
    `starts_arr[rng.integers(0, len(starts_arr), size=shape)]`, and `Generator.integers`
    fills its output in the same flat, sequential draw order whether requested as one
    `size=(n_replicates, n_blocks_per_replicate)` call or as `n_replicates` separate
    `size=(n_blocks_per_replicate,)` calls made back-to-back on the same generator
    (verified empirically against both a 2-call and a variable-chunk-size,
    non-power-of-two-population case before relying on it here). Batching therefore
    costs nothing in RNG-stream fidelity, which is what lets the chunked path
    (`_block_bootstrap_means_chunked`) reproduce the full-array path
    (`_block_bootstrap_resampling_2d`) bit-for-bit for the same seed regardless of
    `chunk_size` -- a pre-existing contract this function must not break.

    Returns shape (n_replicates, n_blocks_per_replicate) of actual row positions
    (not indices into `starts_arr`).
    """
    idx = rng.integers(0, len(starts_arr), size=(n_replicates, n_blocks_per_replicate))
    return starts_arr[idx]


def _blocks_to_gather_indices(
    chosen_starts: np.ndarray, block_length: int, n_rows: int
) -> np.ndarray:
    """Vectorized equivalent of concatenating `values_2d[s : s + block_length]` pieces
    per replicate and truncating the concatenation to `n_rows`.

    Broadcasts each drawn start into `block_length` consecutive row indices
    (`starts[:, :, None] + arange(block_length)[None, None, :]`), flattens the
    per-replicate block sequence into one row-index series per replicate, and
    truncates each row to `n_rows` -- exactly what the per-block Python loop did,
    just built with broadcasting and a single fancy-index gather instead of a
    Python-level loop over blocks.

    Returns shape (n_replicates, n_rows) of row indices into the original panel.
    """
    within_block = np.arange(block_length)
    expanded = chosen_starts[:, :, None] + within_block[None, None, :]
    return expanded.reshape(chosen_starts.shape[0], -1)[:, :n_rows]


def _block_bootstrap_resampling_2d(
    values_2d: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], int]:
    """Moving-block bootstrap on 2-D array (n_rows, n_symbols), session-bounded.

    Standard moving-block bootstrap (specs/overlap_se.md section 1): each replicate draws
    blocks of `_derive_block_length(horizon)` rows WITH REPLACEMENT and concatenates them
    until the resampled series reaches `n_rows`, truncating the final block -- it resamples
    a SERIES, not a single block. No block straddles a session boundary: valid start
    positions are drawn only from within a single session via `day_offsets` (never assumes
    a fixed 375-bar stride, per CLAUDE.md rule 5). A session shorter than the block length
    contributes no start positions and is SKIPPED; the skip count is returned, not silently
    dropped.

    Returns:
      - resampled: (n_boot, n_rows, n_symbols) float64 array, one full-length series per
        replicate.
      - block_indices: FLAT tuple of (start, length) pairs, one entry per block actually
        drawn across ALL replicates (not grouped per replicate) -- every block has the
        same `length` (the derived block length), except none are ever truncated
        individually; only the concatenated series is truncated to `n_rows`.
      - n_sessions_skipped: count of sessions shorter than the block length in use.
    """
    n_rows, n_symbols = values_2d.shape
    offsets = np.asarray(day_offsets, dtype=int)
    n_sessions = len(offsets) - 1

    session_starts = offsets[:-1]
    session_ends = offsets[1:]

    block_length = _derive_block_length(horizon)

    # For each session, find all valid block starts (block must fit entirely inside it).
    valid_starts: list[int] = []
    n_sessions_skipped = 0
    for sess_idx in range(n_sessions):
        sess_start = int(session_starts[sess_idx])
        sess_end = int(session_ends[sess_idx])
        if sess_end - sess_start < block_length:
            n_sessions_skipped += 1
            continue
        valid_starts.extend(range(sess_start, sess_end - block_length + 1))

    rng = np.random.default_rng(seed)
    resampled = np.full((n_boot, n_rows, n_symbols), np.nan, dtype=np.float64)
    block_indices: list[tuple[int, int]] = []

    if not valid_starts:
        # No session is long enough to hold even one block at this block length. Fall
        # back to whole-row iid resampling so callers still get a usable (degenerate) SE
        # rather than an empty result; every session was already counted skipped above.
        for b in range(n_boot):
            row_idx = rng.choice(n_rows, size=n_rows, replace=True)
            resampled[b] = values_2d[row_idx]
        return resampled, tuple(block_indices), n_sessions_skipped

    starts_arr = np.array(valid_starts, dtype=int)
    n_blocks_per_replicate = -(-n_rows // block_length)  # ceil(n_rows / block_length)

    # Vectorized: draw every replicate's block starts in one call, expand each start
    # into `block_length` consecutive row indices via broadcasting, then gather with a
    # single fancy-index operation -- replaces what was previously a per-replicate,
    # per-block Python loop (see `_draw_block_start_positions` /
    # `_blocks_to_gather_indices` for the RNG-stream-fidelity argument).
    chosen_starts_all = _draw_block_start_positions(rng, starts_arr, n_blocks_per_replicate, n_boot)
    gather_idx = _blocks_to_gather_indices(chosen_starts_all, block_length, n_rows)
    # `.astype(..., copy=False)`: match the documented float64 return contract (the old
    # loop assigned into a pre-allocated float64 buffer, upcasting on assignment) without
    # an extra copy when `values_2d` is already float64.
    resampled = values_2d[gather_idx].astype(np.float64, copy=False)

    # `block_indices` retention: still capped at `_BLOCK_INDICES_MAX_RETAINED`, still in
    # the same flat per-replicate-then-per-block order the old loop produced (row-major
    # over `chosen_starts_all`, which is exactly (replicate, block) order).
    flat_starts = chosen_starts_all.ravel()
    take = min(_BLOCK_INDICES_MAX_RETAINED, flat_starts.size)
    block_indices = [(int(s), block_length) for s in flat_starts[:take]]

    return resampled, tuple(block_indices), n_sessions_skipped


# Target peak memory for one chunk of the resampled bootstrap buffer. Chosen as a
# round, conservative number well under typical per-process RAM headroom; see
# `_default_bootstrap_chunk_size` for how it is turned into a replicate count.
_BOOTSTRAP_CHUNK_TARGET_BYTES: int = 256 * 1024 * 1024


def _default_bootstrap_chunk_size(n_rows: int, n_symbols: int) -> int:
    """Derive a bootstrap chunk size bounding peak memory to ~256 MB (measured, not
    guessed -- see `_BOOTSTRAP_CHUNK_TARGET_BYTES`).

    One bootstrap replicate of the resampled series occupies
    `n_rows * n_symbols * 8` bytes (float64). Materialising `chunk_size` replicates
    at once therefore costs `chunk_size * n_rows * n_symbols * 8` bytes; solving for
    `chunk_size` against the target gives the formula below. Clamped to >= 1: a
    single replicate cannot be subdivided further because the row-level
    any-finite/nanmean reduction needs the full (n_rows, n_symbols) replicate
    materialised at once. Called directly with n_symbols=149 (the real panel's raw
    symbol count) this floor of 1 dominates -- one replicate alone is ~0.78 GB,
    already over the 256 MB target -- so `chunk_size` resolves to 1. In practice
    `_compute_bucket_stats`'s `block_bootstrap` branch no longer calls this against
    that shape: it reduces to a length-n_rows per-row bucket-mean series first (see
    `_bucket_row_means`) and passes THAT (n_rows, 1) shape here instead, so on the
    real panel `chunk_size` resolves to roughly `256 MB / (701_863 * 1 * 8 bytes)` =
    tens of replicates per chunk rather than flooring at 1. This function itself is
    unchanged and still tested standalone at n_symbols=149 for the floor case.
    """
    bytes_per_replicate = max(1, n_rows) * max(1, n_symbols) * 8
    return max(1, _BOOTSTRAP_CHUNK_TARGET_BYTES // bytes_per_replicate)


def _block_bootstrap_means_chunked(
    values_2d: np.ndarray,
    day_offsets: np.ndarray,
    horizon: int,
    n_boot: int,
    seed: int,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], int]:
    """Memory-bounded equivalent of `_block_bootstrap_resampling_2d` followed by the
    per-replicate any-finite/nanmean reduction that `_compute_bucket_stats` applies to
    its output. Returns per-replicate bootstrap MEANS directly (`boot_means`, shape
    `(n_boot,)`, NaN where a replicate had no finite values) instead of the full
    `(n_boot, n_rows, n_symbols)` array, so peak memory is
    `O(chunk_size * n_rows * n_symbols)` rather than `O(n_boot * n_rows * n_symbols)`
    (the latter is 0.8 TB on the real panel at n_boot=1000 and SIGKILLs the process).

    Bit-for-bit identical to the unchunked path for the same `seed`: exactly one
    `np.random.default_rng(seed)` is created and every replicate b = 0, 1, ..., n_boot-1
    draws its blocks from that SAME stream in that SAME order -- `chunk_size` only
    changes how many already-decided replicates are held in memory simultaneously
    before being reduced and discarded; it never changes what is drawn, in what
    order, or how the reduction is computed. `_block_bootstrap_resampling_2d` shares
    the same vectorized `_draw_block_start_positions` / `_blocks_to_gather_indices`
    helpers (its own direct tests still exercise the full-array contract on small
    fixtures); this function reimplements the chunked driving loop around those same
    helpers because that function's API requires returning the full array, which is
    exactly the allocation this function exists to avoid.
    """
    n_rows, n_symbols = values_2d.shape
    offsets = np.asarray(day_offsets, dtype=int)
    n_sessions = len(offsets) - 1
    session_starts = offsets[:-1]
    session_ends = offsets[1:]

    block_length = _derive_block_length(horizon)

    valid_starts: list[int] = []
    n_sessions_skipped = 0
    for sess_idx in range(n_sessions):
        sess_start = int(session_starts[sess_idx])
        sess_end = int(session_ends[sess_idx])
        if sess_end - sess_start < block_length:
            n_sessions_skipped += 1
            continue
        valid_starts.extend(range(sess_start, sess_end - block_length + 1))

    rng = np.random.default_rng(seed)
    boot_means = np.full(n_boot, np.nan, dtype=np.float64)
    block_indices: list[tuple[int, int]] = []

    if chunk_size is None:
        chunk_size = _default_bootstrap_chunk_size(n_rows, n_symbols)
    chunk_size = max(1, min(int(chunk_size), max(1, n_boot)))

    if not valid_starts:
        # Same iid-row fallback as `_block_bootstrap_resampling_2d`, chunked.
        for chunk_start in range(0, n_boot, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_boot)
            this_chunk = chunk_end - chunk_start
            resampled_chunk = np.full((this_chunk, n_rows, n_symbols), np.nan, dtype=np.float64)
            for local_b, b in enumerate(range(chunk_start, chunk_end)):
                row_idx = rng.choice(n_rows, size=n_rows, replace=True)
                resampled_chunk[local_b] = values_2d[row_idx]
            chunk_flat = resampled_chunk.reshape((this_chunk, -1))
            valid_rows = np.isfinite(chunk_flat).any(axis=1)
            if np.any(valid_rows):
                boot_means[chunk_start:chunk_end][valid_rows] = np.nanmean(
                    chunk_flat[valid_rows], axis=1
                )
        return boot_means, tuple(block_indices), n_sessions_skipped

    starts_arr = np.array(valid_starts, dtype=int)
    n_blocks_per_replicate = -(-n_rows // block_length)  # ceil(n_rows / block_length)

    for chunk_start in range(0, n_boot, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_boot)
        this_chunk = chunk_end - chunk_start

        # Vectorized per-chunk draw + gather (see `_draw_block_start_positions` /
        # `_blocks_to_gather_indices`): one `rng.integers` call per chunk instead of
        # `this_chunk * n_blocks_per_replicate` per-block Python-loop iterations.
        # Sequential chunks each drawing from the SAME generator reproduce exactly the
        # same flat draw order as one unchunked call over all `n_boot` replicates, so
        # this remains bit-identical to `_block_bootstrap_resampling_2d` for the same
        # seed regardless of `chunk_size` (see docstring above and the identity tests
        # in tests/test_expectancy_bootstrap_chunking.py).
        chosen_starts_chunk = _draw_block_start_positions(
            rng, starts_arr, n_blocks_per_replicate, this_chunk
        )
        gather_idx = _blocks_to_gather_indices(chosen_starts_chunk, block_length, n_rows)
        resampled_chunk = values_2d[gather_idx].astype(np.float64, copy=False)

        remaining_capacity = _BLOCK_INDICES_MAX_RETAINED - len(block_indices)
        if remaining_capacity > 0:
            flat_starts = chosen_starts_chunk.ravel()
            take = min(remaining_capacity, flat_starts.size)
            block_indices.extend((int(s), block_length) for s in flat_starts[:take])

        chunk_flat = resampled_chunk.reshape((this_chunk, -1))
        valid_rows = np.isfinite(chunk_flat).any(axis=1)
        if np.any(valid_rows):
            boot_means[chunk_start:chunk_end][valid_rows] = np.nanmean(
                chunk_flat[valid_rows], axis=1
            )

    return boot_means, tuple(block_indices), n_sessions_skipped


def _non_overlapping_subsample(values: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
    """Subsample every horizon-th observation (non-overlapping).

    Returns:
      - subsampled: 1-D array of subsampled values
      - n_effective: count of subsampled values
    """
    valid_mask = np.isfinite(values)
    valid_indices = np.where(valid_mask)[0]

    # Subsample every horizon-th valid observation
    subsampled_indices = valid_indices[::horizon]
    subsampled = values[subsampled_indices]

    return subsampled, len(subsampled)


def _bucket_row_means(bucket_returns_2d: np.ndarray) -> np.ndarray:
    """Reduce a masked (n_rows, n_symbols) bucket-returns panel to a length-`n_rows`
    series of per-row bucket means.

    Each entry is `nanmean` across symbols for that row; a row with no finite bucket
    member (no symbol in the bucket on that row, or all-NaN data) is NaN. The block
    bootstrap exists to respect dependence ALONG ROWS (sessions and bars), not across
    symbols within a row -- so resampling this reduced series instead of the full 2-D
    panel is statistically equivalent for the bucket MEAN (the only statistic the
    caller bootstraps) while being `n_symbols` times cheaper, and it is strictly
    better-behaved: resampling the 2-D array directly can tear a row's symbols apart
    across different drawn blocks, whereas resampling this series never can.
    """
    n_rows = bucket_returns_2d.shape[0]
    row_means = np.full(n_rows, np.nan, dtype=np.float64)
    row_has_valid = np.isfinite(bucket_returns_2d).any(axis=1)
    if np.any(row_has_valid):
        row_means[row_has_valid] = np.nanmean(bucket_returns_2d[row_has_valid], axis=1)
    return row_means


# ---------------------------------------------------------------------------
# Bucket statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketStat:
    """Statistics for a single bucket."""

    bucket: int
    n_obs: int
    # n_effective is REPORTING ONLY (specs/overlap_se.md section 3, AMENDMENT 1
    # obligation 8): n_effective = (std_bps / se_bps) ** 2. It never feeds back into
    # se_bps, t_stat or spread_t -- mutating it changes nothing downstream.
    n_effective: float
    mean_bps: float
    median_bps: float
    std_bps: float
    se_bps: float  # the bootstrap/analytic standard error IS the standard error (rule 8)
    t_stat: float
    se_method: Literal["block_bootstrap", "non_overlapping", "naive"]
    ci_low_bps: float
    ci_high_bps: float
    block_indices: tuple[tuple[int, int], ...] = ()  # (start, length) pairs for bootstrap


def _compute_bucket_stats(
    bucket_returns: np.ndarray,
    horizon: int,
    day_offsets: np.ndarray,
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    chunk_size: int | None = None,
) -> BucketStat | None:
    """Compute statistics for a single bucket.

    Returns None if the bucket has no observations.
    bucket_returns should be flattened (n_rows * n_symbols,).

    `chunk_size` bounds peak memory in the `block_bootstrap` branch: see
    `_default_bootstrap_chunk_size` for the derivation of its default (None ->
    auto-computed from `n_rows * n_symbols` against a ~256 MB target).
    """
    valid_mask = np.isfinite(bucket_returns)
    valid_values = bucket_returns[valid_mask]

    n_obs = len(valid_values)
    if n_obs == 0:
        return None

    # Convert to basis points
    values_bps = valid_values * 1e4

    mean_bps = float(np.mean(values_bps))
    median_bps = float(np.median(values_bps))
    std_bps = float(np.std(values_bps, ddof=1)) if n_obs > 1 else 0.0

    # Overlap correction: this block computes se_bps ONLY. n_effective is derived from
    # se_bps afterwards, as a REPORTING quantity -- it never feeds back into se_bps
    # (specs/overlap_se.md section 3; the old n_effective = n_obs / (std/se)**2 transform
    # here was the L8 defect and has been deleted).
    block_indices: tuple[tuple[int, int], ...] = ()
    if horizon == 1:
        # For horizon=1, no overlap correction is needed
        if std_bps > 0 and n_obs > 0:
            se_bps = std_bps / np.sqrt(n_obs)
        else:
            se_bps = 0.0

    elif se_method == "block_bootstrap":
        # For block bootstrap, we need to bootstrap at the row level
        # bucket_returns is flattened, so we need to reshape it first
        n_rows = day_offsets[-1]
        div_ok = bucket_returns.shape[0] % n_rows == 0
        n_symbols_in_data = bucket_returns.shape[0] // n_rows if div_ok else 0

        if n_symbols_in_data > 0:
            bucket_returns_2d = bucket_returns.reshape((n_rows, n_symbols_in_data))
            # Resample the length-n_rows per-row bucket-MEAN series, not the full
            # (n_rows, n_symbols) panel: the bucket statistic is a mean over all
            # (row, symbol) cells, and the block bootstrap exists to respect
            # dependence ALONG ROWS, not across symbols within a row -- so reducing
            # once via `_bucket_row_means` before resampling is `n_symbols_in_data`
            # times cheaper (and statistically preferable: it can never tear a row's
            # symbols apart across different blocks). This mirrors what
            # `research/ic.py`'s `_overlap_aware_se` already does for the IC series.
            # On the real panel this turns a 0.837 GB-per-replicate 2-D allocation
            # into a ~5.6 MB-per-replicate 1-D one, so `_block_bootstrap_means_chunked`
            # (still reused as-is below, just called on a 1-column array) auto-derives
            # a chunk_size far above 1 instead of flooring at it.
            row_means = _bucket_row_means(bucket_returns_2d).reshape((n_rows, 1))
            boot_means, block_indices, _n_sessions_skipped = _block_bootstrap_means_chunked(
                row_means, day_offsets, horizon, n_boot, seed, chunk_size
            )
            # Filter out NaN means (from empty or all-NaN samples)
            valid_boot_means = boot_means[np.isfinite(boot_means)]
            if len(valid_boot_means) > 1:
                boot_means_bps = valid_boot_means * 1e4
                se_bps = float(np.std(boot_means_bps, ddof=1))
            else:
                se_bps = 0.0
        else:
            se_bps = 0.0
            block_indices = ()

    elif se_method == "non_overlapping":
        _, n_eff_subsample = _non_overlapping_subsample(valid_values, horizon)
        se_bps = (
            std_bps / np.sqrt(max(1.0, n_eff_subsample))
            if std_bps > 0 and n_eff_subsample > 0
            else 0.0
        )

    elif se_method == "naive":
        se_bps = std_bps / np.sqrt(max(1.0, n_obs)) if std_bps > 0 and n_obs > 0 else 0.0

    else:
        raise ValueError(f"unknown se_method: {se_method}")

    # t-statistic and CI (95%) -- computed directly from se_bps, never through n_effective
    # (obligation 7's invariant: t_stat == mean_bps / se_bps exactly).
    t_stat = mean_bps / se_bps if se_bps > 0 else 0.0
    ci_half_width = 1.96 * se_bps  # ~95% CI
    ci_low_bps = mean_bps - ci_half_width
    ci_high_bps = mean_bps + ci_half_width

    # n_effective: REPORTING ONLY, derived from se_bps after the fact (obligation 8).
    n_effective = (std_bps / se_bps) ** 2 if se_bps > 0 else float(n_obs)

    return BucketStat(
        bucket=-1,  # Placeholder; set by caller
        n_obs=n_obs,
        n_effective=n_effective,
        mean_bps=mean_bps,
        median_bps=median_bps,
        std_bps=std_bps,
        se_bps=se_bps,
        t_stat=t_stat,
        se_method=se_method,
        ci_low_bps=ci_low_bps,
        ci_high_bps=ci_high_bps,
        block_indices=block_indices,
    )


# ---------------------------------------------------------------------------
# ExpectancyTable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectancyTable:
    """Expectancy table with bucket statistics."""

    buckets: tuple[BucketStat, ...]
    horizon: int
    feature_name: str
    n_total: int
    cost_hurdle_bps: float
    spread_bps: float  # top bucket mean minus bottom bucket mean
    spread_t: float
    survives_costs: bool  # abs(spread_bps) > 2 * cost_hurdle_bps

    def explain(self) -> str:
        """Full provenance: every stat, the SE method, the overlap correction, the cost
        hurdle used and where it came from, and the verdict with its reasoning."""
        lines = [
            f"ExpectancyTable(horizon={self.horizon}, feature_name={self.feature_name!r})",
            f"  n_total observations: {self.n_total}",
            f"  SE method: {self.buckets[0].se_method if self.buckets else 'N/A'}",
        ]

        if self.buckets and self.buckets[0].se_method == "naive":
            lines.append("  WARNING: NAIVE SE calculation, UNCORRECTED for overlap!")

        if self.buckets:
            lines.append(
                f"  Overlap correction: Using {self.buckets[0].se_method} method; "
                f"n_effective adjusts for serial correlation."
            )

        lines.extend(
            [
                f"  Cost hurdle: {self.cost_hurdle_bps} bps (from NSE intraday round-trip)",
                f"  Spread (top - bottom): {self.spread_bps:.2f} bps",
                f"  Spread t-statistic: {self.spread_t:.3f}",
            ]
        )

        if abs(self.spread_bps) > 2 * self.cost_hurdle_bps:
            lines.append(
                f"  SURVIVES costs: spread {abs(self.spread_bps):.2f} bps "
                f"> 2x hurdle {self.cost_hurdle_bps} bps"
            )
        else:
            lines.append(
                f"  DOES NOT survive costs: spread {abs(self.spread_bps):.2f} bps "
                f"<= 2x hurdle {self.cost_hurdle_bps} bps"
            )

        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        data = {
            "bucket": [b.bucket for b in self.buckets],
            "n_obs": [b.n_obs for b in self.buckets],
            "n_effective": [b.n_effective for b in self.buckets],
            "mean_bps": [b.mean_bps for b in self.buckets],
            "median_bps": [b.median_bps for b in self.buckets],
            "std_bps": [b.std_bps for b in self.buckets],
            "se_bps": [b.se_bps for b in self.buckets],
            "t_stat": [b.t_stat for b in self.buckets],
            "se_method": [b.se_method for b in self.buckets],
            "ci_low_bps": [b.ci_low_bps for b in self.buckets],
            "ci_high_bps": [b.ci_high_bps for b in self.buckets],
        }
        return pd.DataFrame(data)


def conditional_expectancy(
    feature: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    *,
    n_buckets: int = 5,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    cost_hurdle_bps: float | None = None,
    feature_name: str = "feature",
    chunk_size: int | None = None,
) -> ExpectancyTable:
    """Compute conditional expectancy table.

    `chunk_size` bounds peak memory of the `block_bootstrap` `se_method` -- see
    `_default_bootstrap_chunk_size`. Default `None` auto-derives it from the panel
    shape against a ~256 MB target; pass an explicit value to override.
    """
    feature64 = np.asarray(feature, dtype=np.float64)
    if feature64.ndim != 2:
        raise ValueError("feature must be 2-D")

    # Get bucketing
    bucketing = causal_buckets(
        feature64,
        day_offsets,
        n_buckets=n_buckets,
        method=method,
        min_history=5000 if method == "expanding_quantile" else 50,
    )

    # Compute stats for each bucket
    bucket_stats: list[BucketStat] = []
    for bucket_idx in range(n_buckets):
        # Find observations in this bucket
        in_bucket = bucketing.labels == bucket_idx
        bucket_returns = np.where(in_bucket, fwd.values, np.nan)
        bucket_returns_flat = bucket_returns.flatten()

        stat = _compute_bucket_stats(
            bucket_returns_flat,
            fwd.horizon,
            day_offsets,
            se_method=se_method,
            n_boot=n_boot,
            seed=seed,
            chunk_size=chunk_size,
        )

        if stat is not None:
            stat_with_bucket = BucketStat(
                bucket=bucket_idx,
                n_obs=stat.n_obs,
                n_effective=stat.n_effective,
                mean_bps=stat.mean_bps,
                median_bps=stat.median_bps,
                std_bps=stat.std_bps,
                se_bps=stat.se_bps,
                t_stat=stat.t_stat,
                se_method=stat.se_method,
                ci_low_bps=stat.ci_low_bps,
                ci_high_bps=stat.ci_high_bps,
                block_indices=stat.block_indices,
            )
            bucket_stats.append(stat_with_bucket)

    # Default cost hurdle
    if cost_hurdle_bps is None:
        cost_hurdle_bps = NSEIntradayEquityCosts().round_trip_bps(1e5)

    # Compute spread
    if len(bucket_stats) >= 2:
        top_mean = bucket_stats[-1].mean_bps
        bottom_mean = bucket_stats[0].mean_bps
        spread_bps = top_mean - bottom_mean

        # Spread SE from the bucket-level se_bps DIRECTLY (specs/overlap_se.md section 3):
        # the bootstrap SE already reported on each bucket, combined in quadrature. This is
        # the L8 fix at the table level -- the old code recomputed an SE via
        # std_bps / sqrt(n_effective), which routed through the now-deleted n_effective
        # over-correction and is exactly the defect AMENDMENT 2 traced spread_t through.
        top_se = bucket_stats[-1].se_bps
        bottom_se = bucket_stats[0].se_bps
        spread_se = np.sqrt(top_se**2 + bottom_se**2) if (top_se > 0 or bottom_se > 0) else 0.0
        spread_t = spread_bps / spread_se if spread_se > 0 else 0.0
    else:
        spread_bps = 0.0
        spread_t = 0.0

    survives_costs = abs(spread_bps) > 2 * cost_hurdle_bps

    # Count total observations
    n_total = fwd.n_defined

    return ExpectancyTable(
        buckets=tuple(bucket_stats),
        horizon=fwd.horizon,
        feature_name=feature_name,
        n_total=n_total,
        cost_hurdle_bps=cost_hurdle_bps,
        spread_bps=spread_bps,
        spread_t=spread_t,
        survives_costs=survives_costs,
    )


# ---------------------------------------------------------------------------
# Decomposition functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellStat:
    """Statistics for a single cell in a double-sort."""

    n_obs: int
    mean_bps: float
    median_bps: float
    std_bps: float
    t_stat: float
    se_method: Literal["block_bootstrap", "non_overlapping", "naive"]
    ci_low_bps: float
    ci_high_bps: float


@dataclass(frozen=True)
class SortResult:
    """Result of a double-sort."""

    cells: tuple[tuple[CellStat, ...], ...]  # ROW-MAJOR: (a-bucket outer, b-bucket inner)
    n_total: int
    n_buckets_a: int
    n_buckets_b: int
    thin_cell_threshold: int
    thin_cells: tuple[tuple[int, int], ...]  # (i, j) pairs of thin cells


def double_sort(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    *,
    n_buckets_a: int = 5,
    n_buckets_b: int = 5,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    seed: int = 0,
    thin_cell_threshold: int = 30,
) -> SortResult:
    """Perform a double-sort on two features."""
    feature_a64 = np.asarray(feature_a, dtype=np.float64)
    feature_b64 = np.asarray(feature_b, dtype=np.float64)

    # Get bucketing for both features
    buck_a = causal_buckets(
        feature_a64, day_offsets, n_buckets=n_buckets_a, method=method
    )
    buck_b = causal_buckets(
        feature_b64, day_offsets, n_buckets=n_buckets_b, method=method
    )

    # Compute stats for each cell
    cells: list[list[CellStat]] = []
    thin_cells: list[tuple[int, int]] = []

    for i in range(n_buckets_a):
        row = []
        for j in range(n_buckets_b):
            in_cell = (buck_a.labels == i) & (buck_b.labels == j)
            cell_returns = np.where(in_cell, fwd.values, np.nan).flatten()

            stat = _compute_bucket_stats(
                cell_returns,
                fwd.horizon,
                day_offsets,
                se_method=se_method,
                n_boot=100,
                seed=seed,
            )

            if stat is not None:
                cell_stat = CellStat(
                    n_obs=stat.n_obs,
                    mean_bps=stat.mean_bps,
                    median_bps=stat.median_bps,
                    std_bps=stat.std_bps,
                    t_stat=stat.t_stat,
                    se_method=stat.se_method,
                    ci_low_bps=stat.ci_low_bps,
                    ci_high_bps=stat.ci_high_bps,
                )
            else:
                # Empty cell
                cell_stat = CellStat(
                    n_obs=0,
                    mean_bps=0.0,
                    median_bps=0.0,
                    std_bps=0.0,
                    t_stat=0.0,
                    se_method=se_method,
                    ci_low_bps=0.0,
                    ci_high_bps=0.0,
                )

            row.append(cell_stat)

            # Check for thin cells (both empty and underfilled)
            if cell_stat.n_obs < thin_cell_threshold:
                thin_cells.append((i, j))

        cells.append(row)

    # n_total is the count of observations (cells), not rows
    # Each row contributes n_symbols cells (one per symbol)
    n_total = fwd.n_defined * feature_a.shape[1]  # rows * n_symbols

    return SortResult(
        cells=tuple(tuple(row) for row in cells),
        n_total=n_total,
        n_buckets_a=n_buckets_a,
        n_buckets_b=n_buckets_b,
        thin_cell_threshold=thin_cell_threshold,
        thin_cells=tuple(thin_cells),
    )


def expectancy_by_year(
    feature: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    dates: np.ndarray,
    *,
    n_buckets: int = 5,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    feature_name: str = "feature",
) -> dict[int, ExpectancyTable]:
    """Compute expectancy by calendar year."""
    feature64 = np.asarray(feature, dtype=np.float64)
    dates_arr = np.asarray(dates, dtype=object)
    n_rows = feature64.shape[0]

    # Map rows to years
    session_id = np.searchsorted(day_offsets[1:], np.arange(n_rows), side="right")
    years = np.array([dates_arr[min(i, len(dates_arr) - 1)].year for i in session_id])

    result = {}
    for year in np.unique(years):
        year_mask = years == year

        # Filter feature and fwd to this year
        feature_year = np.where(year_mask[:, None], feature64, np.nan)
        fwd_year_values = np.where(year_mask[:, None], fwd.values, np.nan)
        # Count defined rows (not cells)
        rows_with_any_finite_year = np.any(np.isfinite(fwd_year_values), axis=1)
        n_defined_year = int(np.sum(rows_with_any_finite_year))

        # Create a temporary ForwardReturns for this year
        fwd_year = ForwardReturns(
            values=fwd_year_values,
            horizon=fwd.horizon,
            session_bounded=fwd.session_bounded,
            n_defined=n_defined_year,
            n_nan_tail=int((~np.isfinite(fwd_year_values)).sum()),
        )

        # Compute expectancy for this year
        table_year = conditional_expectancy(
            feature_year,
            fwd_year,
            day_offsets,
            n_buckets=n_buckets,
            method=method,
            se_method=se_method,
            n_boot=n_boot,
            seed=seed,
            feature_name=feature_name,
        )

        result[year] = table_year

    return result


def expectancy_by_time_of_day(
    feature: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    minute_of_day: np.ndarray,
    *,
    n_buckets: int = 5,
    time_bucket_minutes: int = 60,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    feature_name: str = "feature",
) -> dict[int, ExpectancyTable]:
    """Compute expectancy by time-of-day bucket."""
    feature64 = np.asarray(feature, dtype=np.float64)
    minute_arr = np.asarray(minute_of_day, dtype=int)

    # Map rows to time buckets
    time_buckets = (minute_arr // time_bucket_minutes) * time_bucket_minutes

    result = {}
    for tb in np.unique(time_buckets):
        tb_mask = time_buckets == tb

        # Filter feature and fwd to this time bucket
        feature_tb = np.where(tb_mask[:, None], feature64, np.nan)
        fwd_tb_values = np.where(tb_mask[:, None], fwd.values, np.nan)
        # Count defined rows
        rows_with_any_finite_tb = np.any(np.isfinite(fwd_tb_values), axis=1)
        n_defined_tb = int(np.sum(rows_with_any_finite_tb))

        # Create a temporary ForwardReturns for this time bucket
        fwd_tb = ForwardReturns(
            values=fwd_tb_values,
            horizon=fwd.horizon,
            session_bounded=fwd.session_bounded,
            n_defined=n_defined_tb,
            n_nan_tail=int((~np.isfinite(fwd_tb_values)).sum()),
        )

        # Compute expectancy for this time bucket
        table_tb = conditional_expectancy(
            feature_tb,
            fwd_tb,
            day_offsets,
            n_buckets=n_buckets,
            method=method,
            se_method=se_method,
            n_boot=n_boot,
            seed=seed,
            feature_name=feature_name,
        )

        result[int(tb)] = table_tb

    return result


def expectancy_by_liquidity_decile(
    feature: np.ndarray,
    fwd: ForwardReturns,
    day_offsets: np.ndarray,
    prior_adv: np.ndarray,
    *,
    n_buckets: int = 10,
    method: str = "expanding_quantile",
    se_method: str = "block_bootstrap",
    n_boot: int = 1000,
    seed: int = 0,
    feature_name: str = "feature",
) -> dict[int, ExpectancyTable]:
    """Compute expectancy by liquidity decile.

    prior_adv must already be lagged (strictly prior sessions). The function does not
    re-derive it, and @causal cannot detect a leak in it — so document that loudly
    at the call site.
    """
    feature64 = np.asarray(feature, dtype=np.float64)
    prior_adv64 = np.asarray(prior_adv, dtype=np.float64)

    # Get bucketing for prior_adv (this is a caller contract: it's already prior)
    adv_bucketing = causal_buckets(
        prior_adv64, day_offsets, n_buckets=n_buckets, method="cross_sectional_rank"
    )

    result = {}
    for decile in range(n_buckets):
        decile_mask = adv_bucketing.labels == decile

        # Filter feature and fwd to this decile
        feature_decile = np.where(decile_mask, feature64, np.nan)
        fwd_decile_values = np.where(decile_mask, fwd.values, np.nan)
        # Count defined rows
        rows_with_any_finite_decile = np.any(np.isfinite(fwd_decile_values), axis=1)
        n_defined_decile = int(np.sum(rows_with_any_finite_decile))

        # Create a temporary ForwardReturns for this decile
        fwd_decile = ForwardReturns(
            values=fwd_decile_values,
            horizon=fwd.horizon,
            session_bounded=fwd.session_bounded,
            n_defined=n_defined_decile,
            n_nan_tail=int((~np.isfinite(fwd_decile_values)).sum()),
        )

        # Compute expectancy for this decile
        table_decile = conditional_expectancy(
            feature_decile,
            fwd_decile,
            day_offsets,
            n_buckets=n_buckets,
            method=method,
            se_method=se_method,
            n_boot=n_boot,
            seed=seed,
            feature_name=feature_name,
        )

        result[decile] = table_decile

    return result
