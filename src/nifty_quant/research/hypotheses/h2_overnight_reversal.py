"""H2: Cross-sectional overnight reversal (Berkman, Koch, Tuttle & Zhang 2012).

Stocks with the largest overnight gains (close[15:20, d-1] -> open[09:16, d]) are
hypothesized to underperform intraday, and stocks with the largest overnight losses to
outperform intraday, cross-sectionally across the universe. One round trip per name per
day: enter at 09:16, exit at the 15:20 square-off.

Unlike H1, this is a CROSS-SECTIONAL signal (ranked across symbols within a session, not
across time for one symbol), so it is built and evaluated entirely through
``research.lens.Lens`` -- this module does not reimplement bucketing, expectancy, overlap
correction, or the kill criteria, only:

1. Builds the overnight-return feature via ``features.market.tradable_overnight_return``,
   which resolves the entry/exit checkpoints BY TIME LABEL (09:16 open, 15:20 close),
   never by row position -- positional resolution would read the corrupt 09:15 print on
   the entry side and an arbitrary post-square-off print on the exit side.
1b. REDUCES the panel to exactly two rows per usable session before calling ``Lens`` at
    all (``_build_checkpoint_panel``). ``Lens.verdict(..., horizon=1, ...)`` measures a
    return one BAR ahead, positionally -- on a real many-bar-per-session panel, skipping
    this reduction would make ``horizon=1`` a ~1-minute return, not the entry-to-exit
    session, and would still return a plausible-looking (wrong) number rather than error.
2. Calls ``Lens.verdict()`` with ``method="cross_sectional_rank"``. This is mandatory, not
   a style choice: the default ``expanding_quantile`` bucketing needs ``min_history`` prior
   rows per column before assigning any bucket, and a once-per-session cross-sectional
   signal can never accumulate that -- the default would silently yield zero usable buckets
   and every verdict would come back empty, indistinguishable from "no effect".
   ``cross_sectional_rank`` ranks across symbols within a row and is therefore causal (and
   available) from the very first session.
3. Appends a stated observed direction (reversal vs. momentum -- criterion 1 uses
   ``abs(spread)``, so a positive spread from overnight MOMENTUM clears the same gate a
   reversal would; silence here would let it pass as a confirmed reversal) and the
   ``universe.static.survivorship_report(...).warning_line()`` to the verdict's reasons, so
   both are visible in ``HypothesisVerdict.explain()`` without touching the six kill
   criteria Lens itself already computed.

``HypothesisVerdict.cost_hurdle_bps`` is the RAW 1x round-trip cost (~8.26452 bps at
notional 1e5); the 2x survival gate is applied inline by ``Lens.verdict()``'s criterion 1
and is never stored doubled in this field.
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np

from nifty_quant.calendar import SessionGrid
from nifty_quant.data.panel import Panel
from nifty_quant.features.market import tradable_overnight_return
from nifty_quant.research.lens import Feature, HypothesisVerdict, Lens
from nifty_quant.universe.static import Universe, survivorship_report

_ENTRY_HHMM = "09:16"
_EXIT_HHMM = "15:20"
_HYPOTHESIS_ID = "H2_overnight_reversal"
_FEATURE_NAME = "h2_overnight_return"


def build_overnight_feature(panel: Panel) -> Feature:
    """(n_rows, n_symbols) overnight return, broadcast to every row of its session.

    ``r_on[i, d] = log(open[09:16, d, i] / close[15:20, d-1, i])``, delegated entirely to
    ``features.market.tradable_overnight_return`` (label-resolved endpoints, never
    positional): 09:15 is never read on the entry side, and a stray post-15:20 print is
    never read on the exit side. ``d-1`` is resolved by that function as the previous
    *session in the panel*, not the previous calendar day, and independently per symbol --
    a symbol missing either checkpoint is NaN for that session only, never forward-filled.
    The first session in the panel has no ``d-1`` and is NaN for every symbol.

    Does not mutate any panel field: the delegated function copies its inputs before
    transforming them.

    Returns:
        A ``Feature`` with ``kind="return"``, values shape ``(panel.n_rows(),
        panel.n_symbols())``, dtype float64.
    """
    values = tradable_overnight_return(
        panel.field("open"),
        panel.field("close"),
        panel.day_offsets,
        panel.minute_of_day(),
        entry_hhmm=_ENTRY_HHMM,
        exit_hhmm=_EXIT_HHMM,
    )
    return Feature(
        name=_FEATURE_NAME,
        values=values,
        kind="return",
        warmup_bars=0,
        params={"entry_hhmm": _ENTRY_HHMM, "exit_hhmm": _EXIT_HHMM},
    )


def _build_checkpoint_panel(panel: Panel, feature: Feature) -> tuple[Panel, Feature]:
    """Reduce `panel` (and the matching session-broadcast `feature`) to exactly two rows
    per USABLE session: the `_ENTRY_HHMM` open (stored in the reduced panel's "close"
    field, so that a `horizon=1` close-to-close forward return on the REDUCED panel is
    exactly the true intraday return) and the `_EXIT_HHMM` close. A session lacking
    EITHER label at all -- a ~60-bar Muhurat session has no 15:20 whatsoever; a session
    with a stray 15:21 print after square-off must still resolve 15:20, never the
    session's last row -- is dropped entirely, matching the drop rule already applied to
    the overnight feature itself.

    This reduction is why `Lens.verdict(..., horizon=1, ...)` is correct here: without
    it, `horizon=1` on a real ~375-bar session would measure the return over ONE MINUTE
    from whatever row a bucket happens to land on, not the entry-to-exit trading session
    -- an edge diluted by two orders of magnitude, returned without ever raising.

    Checkpoints are resolved by IST time label via `Panel.rows_at_time`, never by row
    position or a fixed bars-per-session assumption.
    """
    entry_rows = panel.rows_at_time(_ENTRY_HHMM)
    exit_rows = panel.rows_at_time(_EXIT_HHMM)
    entry_days = np.searchsorted(panel.day_offsets, entry_rows, side="right") - 1
    exit_days = np.searchsorted(panel.day_offsets, exit_rows, side="right") - 1

    entry_row_by_day = dict(zip(entry_days.tolist(), entry_rows.tolist(), strict=True))
    exit_row_by_day = dict(zip(exit_days.tolist(), exit_rows.tolist(), strict=True))
    usable_days = sorted(set(entry_row_by_day) & set(exit_row_by_day))

    n_symbols = panel.n_symbols()
    n_rows = 2 * len(usable_days)

    panel_open = panel.field("open")
    panel_close = panel.field("close")
    panel_volume = panel.field("volume")

    ts = np.empty(n_rows, dtype=np.int64)
    close = np.empty((n_rows, n_symbols), dtype=panel_close.dtype)
    volume = np.empty((n_rows, n_symbols), dtype=panel_volume.dtype)
    feat_values = np.empty((n_rows, n_symbols), dtype=np.float64)

    for i, day in enumerate(usable_days):
        entry_row = entry_row_by_day[day]
        exit_row = exit_row_by_day[day]
        entry_out, exit_out = 2 * i, 2 * i + 1
        ts[entry_out] = panel.ts[entry_row]
        ts[exit_out] = panel.ts[exit_row]
        close[entry_out, :] = panel_open[entry_row, :]
        close[exit_out, :] = panel_close[exit_row, :]
        volume[entry_out, :] = panel_volume[entry_row, :]
        volume[exit_out, :] = panel_volume[exit_row, :]
        feat_values[entry_out, :] = feature.values[entry_row, :]
        feat_values[exit_out, :] = feature.values[entry_row, :]

    grid = SessionGrid.from_timestamps(ts)
    checkpoint_panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=panel.symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )
    checkpoint_feature = dataclasses.replace(feature, values=feat_values)
    return checkpoint_panel, checkpoint_feature


def _restrict_feature_to_panel(full_panel: Panel, feature: Feature, sliced: Panel) -> Feature:
    """Slice `feature.values` to the row range `sliced` occupies within `full_panel`.

    `full_panel`/`feature` may be either the raw input panel or (as `run_h2` actually
    calls this) the reduced checkpoint panel from `_build_checkpoint_panel` -- this
    function only needs `sliced` to be a contiguous row range of `full_panel` (i.e.
    produced by `full_panel.sub(start=..., end=...)`, never a symbol restriction) --
    located by searching `full_panel.ts` (sorted, unique) for `sliced.ts[0]` rather than
    re-deriving the row range from `start`/`end` a second time, so there is exactly one
    place that maps dates to rows: `Panel.sub` itself.

    `run_h2` builds the overnight feature and reduces to checkpoints on the FULL,
    unrestricted panel before this slice is applied, so a restricted window's first
    session still sees the true previous session's 15:20 close even when that previous
    session falls before `start` -- restricting the reporting window must not manufacture
    a spurious NaN at its own left edge.

    A `start`/`end` window that selects zero sessions (`sliced.n_rows() == 0`) is a
    degenerate but legitimate caller input, not a bug: it returns an empty
    `(0, n_symbols)` feature rather than indexing `sliced.ts[0]` on an empty array, which
    would raise a bare `IndexError` three call frames inside a research harness -- exactly
    the opaque-crash failure mode that would abort an entire multi-symbol batch run.
    """
    n_rows_sliced = sliced.n_rows()
    if n_rows_sliced == 0:
        return dataclasses.replace(feature, values=feature.values[:0])
    row_start = int(np.searchsorted(full_panel.ts, sliced.ts[0]))
    row_end = row_start + n_rows_sliced
    return dataclasses.replace(feature, values=feature.values[row_start:row_end])


def _survivorship_window(dates: np.ndarray) -> tuple[datetime.date, datetime.date]:
    """First/last session date in `dates` for the survivorship report's `(start, end)`.

    A `start`/`end` restriction that selects zero sessions leaves `dates` empty; rather
    than raise a bare `IndexError` from `dates[0]` deep inside a batch run, this falls
    back to today's date for both endpoints -- a documented degenerate value (an empty
    window has no real reporting range to speak of) that keeps `survivorship_report`
    callable instead of aborting the whole run.
    """
    if len(dates) == 0:
        today = datetime.date.today()
        return today, today
    return dates[0], dates[-1]


def _direction_line(spread_bps: float, horizon: int, horizon_mode: str) -> str:
    """State the OBSERVED direction of the top-minus-bottom spread.

    Criterion 1 (the cost hurdle) is gated on `abs(spread_bps)`, so a positive spread from
    overnight MOMENTUM clears the identical statistical/cost bar a genuine reversal would.
    This line is what stops that from being silently reported as a confirmed reversal.
    """
    overlap_note = (
        f"horizon={horizon} bar(s) (horizon_mode={horizon_mode!r}): one observation per "
        "symbol-day, non-overlapping, so no block-bootstrap overlap correction is needed "
        "even though Lens defaults to requesting it."
    )
    if spread_bps < 0.0:
        return (
            f"Observed direction: NEGATIVE top-minus-bottom spread ({spread_bps:.4f} bps) "
            f"-- consistent with REVERSAL, the hypothesized direction. {overlap_note}"
        )
    return (
        f"Observed direction: POSITIVE top-minus-bottom spread ({spread_bps:.4f} bps) -- "
        f"this is MOMENTUM, not reversal. Do not report this as a confirmed reversal. "
        f"{overlap_note}"
    )


def run_h2(
    panel: Panel,
    *,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    horizon_mode: str = "session",
    cost_hurdle_bps: float | None = None,
    seed: int = 0,
) -> HypothesisVerdict:
    """Run the H2 cross-sectional overnight-reversal hypothesis test.

    Builds the overnight-return feature on the full `panel` (so that a restricted
    `start`/`end` window never loses its left-edge session's true previous close), then
    REDUCES panel and feature to exactly two rows per usable session --
    `_build_checkpoint_panel` -- before handing anything to `Lens`. This reduction is not
    optional: `Lens.verdict(..., horizon=1, ...)` measures a return `horizon` BARS ahead
    positionally, and on a real many-bar-per-session panel `horizon=1` without this
    reduction would silently measure a ~1-minute return, not the entry-to-exit trading
    session -- a plausible-looking edge diluted by roughly two orders of magnitude, with
    no error raised. After reduction, `horizon=1` from the entry row IS exactly the
    15:20 exit for every session, so bucket means/spreads are the true intraday return
    and the windows never overlap.

    `start`/`end` (inclusive, `Panel.sub` semantics) then restrict the REDUCED checkpoint
    panel before delegating everything else to `Lens.verdict()` with
    `method="cross_sectional_rank"` (mandatory -- see module docstring).

    This function never reads `research.splits.HoldoutLock` and never looks past whatever
    `start`/`end` the caller supplies -- the caller is responsible for locking out the
    holdout window.

    `"session"` is the only implemented `horizon_mode`: with exactly the two checkpoint
    bars 09:16/15:20 present per session, one bar forward from the entry row is exactly
    the 15:20 exit. Any other value is rejected outright -- silently falling back to
    session-mode behaviour for an unrecognized `horizon_mode` would be worse than raising,
    since a caller who thinks they asked for a different mode would get session-mode
    numbers back with no indication anything was ignored.

    Returns:
        A `HypothesisVerdict` with all six of `Lens.verdict()`'s kill criteria (criterion 5
        is `NOT_EVALUATED`, never a silent PASS, since no latency profile is supplied here),
        plus two appended reason lines: the observed reversal/momentum direction, and the
        `universe.static.survivorship_report(...).warning_line()` (starts with
        `"UNIVERSE "`). `cost_hurdle_bps` on the returned verdict is the raw 1x round-trip
        cost, matching `ExpectancyTable.cost_hurdle_bps` -- never the doubled survival gate.

    Raises:
        ValueError: If `horizon_mode` is not `"session"`.
    """
    if horizon_mode != "session":
        raise ValueError(
            f"Unsupported horizon_mode: {horizon_mode!r}; run_h2's horizon_mode parameter "
            "only implements 'session' (one observation per symbol-day, entry 09:16 -> "
            "exit 15:20)."
        )
    horizon = 1

    full_feature = build_overnight_feature(panel)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, full_feature)
    sliced_panel = checkpoint_panel.sub(start=start, end=end)
    sliced_feature = _restrict_feature_to_panel(checkpoint_panel, checkpoint_feature, sliced_panel)

    lens = Lens(sliced_panel, seed=seed)
    base_verdict = lens.verdict(
        _HYPOTHESIS_ID,
        sliced_feature,
        horizon,
        method="cross_sectional_rank",
        cost_hurdle_bps=cost_hurdle_bps,
    )

    window_start, window_end = _survivorship_window(sliced_panel.dates)

    universe = Universe(name="h2_panel", symbols=tuple(panel.symbols))
    report = survivorship_report(universe, window_start, window_end)

    extra_reasons = (
        _direction_line(base_verdict.expectancy.spread_bps, horizon, horizon_mode),
        report.warning_line(),
    )

    return dataclasses.replace(base_verdict, reasons=base_verdict.reasons + extra_reasons)
