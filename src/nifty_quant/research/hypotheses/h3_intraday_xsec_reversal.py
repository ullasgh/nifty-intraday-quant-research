"""H3: Cross-sectional intraday reversal from morning residual returns.

Stocks with the largest positive residual return over the first 44 minutes of the session
(09:16 open -> 10:00 close, cross-sectionally demeaned) are hypothesized to underperform
over the remainder of the session (10:00 -> 15:20), and stocks with the largest negative
residual to outperform, cross-sectionally across the universe. One round trip per name per
session: enter at 10:00, exit at the 15:20 square-off.

Unlike H1, this is a CROSS-SECTIONAL signal (ranked across symbols within a session, not
across time for one symbol), so it is built and evaluated entirely through
``research.lens.Lens`` -- this module does not reimplement bucketing, expectancy, overlap
correction, or the kill criteria, only:

1. Builds the morning-residual feature inline (no shared library function exists for this
   signal): per session, per symbol, ``r_morning = log(close[10:00] / open[09:16])``,
   then cross-sectionally demeaned within each session. Checkpoints are resolved BY TIME
   LABEL via ``minute_of_day()``/``day_offsets``, never by row position -- positional
   resolution would read the corrupt 09:15 print on the signal side and an arbitrary
   post-square-off print on the exit side.
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
   ``abs(spread)``, so a positive spread from morning MOMENTUM clears the same gate a
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
from nifty_quant.research.lens import Feature, HypothesisVerdict, Lens
from nifty_quant.universe.static import Universe, survivorship_report

_OPEN_HHMM = "09:16"
_MORNING_HHMM = "10:00"
_EXIT_HHMM = "15:20"
_HYPOTHESIS_ID = "H3_intraday_xsec_reversal"
_FEATURE_NAME = "h3_morning_residual_return"


def build_morning_residual_feature(panel: Panel) -> Feature:
    """(n_rows, n_symbols) morning residual return, broadcast to every row of its session.

    Per session, per symbol: ``r_morning = log(close[10:00] / open[09:16])``, then
    cross-sectionally demeaned within the session (subtract the mean of finite values
    across symbols). Checkpoints are resolved BY TIME LABEL via ``minute_of_day()``
    compared against ``9*60+16`` and ``10*60+0``, never by row position or a fixed
    bars-per-session assumption. A session lacking EITHER label entirely (no row at all
    for any symbol) is left NaN for every symbol. Per symbol, valid iff both prices are
    finite and strictly positive; otherwise NaN for that symbol only. If only one symbol
    is finite in a session, it demeans to exactly 0.0, not NaN. If zero symbols are
    finite, the session stays NaN.

    Does not mutate any panel field: copies inputs before transforming them.

    Returns:
        A ``Feature`` with ``kind="return"``, values shape ``(panel.n_rows(),
        panel.n_symbols())``, dtype float64.
    """
    n_rows = panel.n_rows()
    n_symbols = panel.n_symbols()
    values = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)

    panel_open = panel.field("open")
    panel_close = panel.field("close")
    minute_of_day = panel.minute_of_day()
    day_offsets = panel.day_offsets

    open_minute = 9 * 60 + 16
    morning_minute = 10 * 60 + 0

    for i in range(len(day_offsets) - 1):
        curr_start = day_offsets[i]
        curr_end = day_offsets[i + 1]

        session_minutes = minute_of_day[curr_start:curr_end]
        open_rows = np.where(session_minutes == open_minute)[0]
        morning_rows = np.where(session_minutes == morning_minute)[0]

        if len(open_rows) == 0 or len(morning_rows) == 0:
            continue

        open_row = curr_start + open_rows[0]
        morning_row = curr_start + morning_rows[0]

        entry_open = panel_open[open_row, :].astype(np.float64)
        morning_close = panel_close[morning_row, :].astype(np.float64)

        valid = (
            np.isfinite(entry_open)
            & np.isfinite(morning_close)
            & (entry_open > 0)
            & (morning_close > 0)
        )

        raw = np.full(n_symbols, np.nan, dtype=np.float64)
        raw[valid] = np.log(morning_close[valid] / entry_open[valid])

        finite_mask = np.isfinite(raw)
        if np.any(finite_mask):
            demeaned = raw.copy()
            demeaned[finite_mask] = raw[finite_mask] - np.mean(raw[finite_mask])
            values[curr_start:curr_end, :] = demeaned

    return Feature(
        name=_FEATURE_NAME,
        values=values,
        kind="return",
        warmup_bars=0,
        params={"entry_hhmm": _OPEN_HHMM, "morning_hhmm": _MORNING_HHMM},
    )


def _build_checkpoint_panel(panel: Panel, feature: Feature) -> tuple[Panel, Feature]:
    """Reduce `panel` (and the matching session-broadcast `feature`) to exactly two rows
    per USABLE session: the `_MORNING_HHMM` close (stored in the reduced panel's "close"
    field, so that a `horizon=1` close-to-close forward return on the REDUCED panel is
    exactly the true intraday return from 10:00 to 15:20) and the `_EXIT_HHMM` close. A
    session lacking EITHER label at all -- a ~60-bar Muhurat session has no 15:20
    whatsoever; a session with a stray 15:21 print after square-off must still resolve
    15:20, never the session's last row -- is dropped entirely, matching the drop rule
    already applied to the morning-residual feature itself.

    This reduction is why `Lens.verdict(..., horizon=1, ...)` is correct here: without
    it, `horizon=1` on a real ~375-bar session would measure the return over ONE MINUTE
    from whatever row a bucket happens to land on, not the entry-to-exit trading session
    -- an edge diluted by two orders of magnitude, returned without ever raising.

    Checkpoints are resolved by IST time label via `Panel.rows_at_time`, never by row
    position or a fixed bars-per-session assumption.
    """
    entry_rows = panel.rows_at_time(_MORNING_HHMM)
    exit_rows = panel.rows_at_time(_EXIT_HHMM)
    entry_days = np.searchsorted(panel.day_offsets, entry_rows, side="right") - 1
    exit_days = np.searchsorted(panel.day_offsets, exit_rows, side="right") - 1

    entry_row_by_day = dict(zip(entry_days.tolist(), entry_rows.tolist(), strict=True))
    exit_row_by_day = dict(zip(exit_days.tolist(), exit_rows.tolist(), strict=True))
    usable_days = sorted(set(entry_row_by_day) & set(exit_row_by_day))

    n_symbols = panel.n_symbols()
    n_rows = 2 * len(usable_days)

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
        close[entry_out, :] = panel_close[entry_row, :]
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

    `full_panel`/`feature` may be either the raw input panel or (as `run_h3` actually
    calls this) the reduced checkpoint panel from `_build_checkpoint_panel` -- this
    function only needs `sliced` to be a contiguous row range of `full_panel` (i.e.
    produced by `full_panel.sub(start=..., end=...)`, never a symbol restriction) --
    located by searching `full_panel.ts` (sorted, unique) for `sliced.ts[0]` rather than
    re-deriving the row range from `start`/`end` a second time, so there is exactly one
    place that maps dates to rows: `Panel.sub` itself.

    `run_h3` builds the morning-residual feature and reduces to checkpoints on the FULL,
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


def _direction_line(spread_bps: float) -> str:
    """State the OBSERVED direction of the top-minus-bottom spread.

    Criterion 1 (the cost hurdle) is gated on `abs(spread_bps)`, so a positive spread from
    morning MOMENTUM clears the identical statistical/cost bar a genuine reversal would.
    This line is what stops that from being silently reported as a confirmed reversal.
    """
    if spread_bps < 0.0:
        return (
            f"Observed direction: NEGATIVE top-minus-bottom spread ({spread_bps:.4f} bps) "
            f"-- consistent with REVERSAL, the hypothesized direction."
        )
    return (
        f"Observed direction: POSITIVE top-minus-bottom spread ({spread_bps:.4f} bps) -- "
        f"this is MOMENTUM, not reversal. Do not report this as a confirmed reversal."
    )


def run_h3(
    panel: Panel,
    *,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    cost_hurdle_bps: float | None = None,
    seed: int = 0,
) -> HypothesisVerdict:
    """Run the H3 cross-sectional intraday-reversal hypothesis test.

    Builds the morning-residual feature on the full `panel` (so that a restricted
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

    Returns:
        A `HypothesisVerdict` with all six of `Lens.verdict()`'s kill criteria (criterion 5
        is `NOT_EVALUATED`, never a silent PASS, since no latency profile is supplied here),
        plus two appended reason lines: the observed reversal/momentum direction, and the
        `universe.static.survivorship_report(...).warning_line()` (starts with
        `"UNIVERSE "`). `cost_hurdle_bps` on the returned verdict is the raw 1x round-trip
        cost, matching `ExpectancyTable.cost_hurdle_bps` -- never the doubled survival gate.
    """
    full_feature = build_morning_residual_feature(panel)
    checkpoint_panel, checkpoint_feature = _build_checkpoint_panel(panel, full_feature)
    sliced_panel = checkpoint_panel.sub(start=start, end=end)
    sliced_feature = _restrict_feature_to_panel(checkpoint_panel, checkpoint_feature, sliced_panel)

    lens = Lens(sliced_panel, seed=seed)
    base_verdict = lens.verdict(
        _HYPOTHESIS_ID,
        sliced_feature,
        1,
        method="cross_sectional_rank",
        cost_hurdle_bps=cost_hurdle_bps,
    )

    window_start, window_end = _survivorship_window(sliced_panel.dates)

    universe = Universe(name="h3_panel", symbols=tuple(panel.symbols))
    report = survivorship_report(universe, window_start, window_end)

    extra_reasons = (
        _direction_line(base_verdict.expectancy.spread_bps),
        report.warning_line(),
    )

    return dataclasses.replace(base_verdict, reasons=base_verdict.reasons + extra_reasons)