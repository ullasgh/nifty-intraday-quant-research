"""Parameterised long-only tilt backtest wrapper.

This module implements a reusable backtest wrapper for the long-only index-relative
tilt on H2's overnight-reversal signal. It supports configurable:
  - Date windows (start/end)
  - Time windows (entry_hhmm/exit_hhmm)
  - Capital (affects clip size and cost calculation)
  - Tilt type (mild = clipped-rank, aggressive = bottom quintile)
  - Smoothing (0.0 = no change, 1.0 = daily rebalance)
  - Rebalance frequency

The cost model charges ONE LEG (not 2x) on turnover only, making this the only
strategy in this repo that clears costs.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import defaultdict

import numpy as np

from nifty_quant.calendar import TradingCalendar
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research.hypotheses.h2_overnight_reversal import build_overnight_feature
from nifty_quant.research.splits import HoldoutLock, default_holdout_lock_path


@dataclasses.dataclass(frozen=True)
class TiltConfig:
    """Configuration for a tilt backtest.

    Attributes:
        start: First date to include (inclusive).
        end: Last date to include (inclusive).
        entry_hhmm: Entry time label (resolved by minute_of_day, not position).
        exit_hhmm: Exit time label (resolved by minute_of_day, not position).
        capital: Total book size in rupees; affects clip and thus cost.
        tilt: "mild" (clipped-rank) or "aggressive" (bottom quintile).
        smoothing: Weight-update parameter, 1.0 = daily rebalance, 0.0 = no drift.
        rebalance_every: Hold book k sessions between recomputes.
        universe: Universe name (used for reporting only).
        continuous_only: Restrict to symbols with coverage across whole window.
        seed: Random seed (unused by tilt, for consistency with other modules).
    """

    start: datetime.date
    end: datetime.date
    entry_hhmm: str = "09:16"
    exit_hhmm: str = "15:20"
    capital: float = 1_000_000.0
    tilt: str = "mild"
    smoothing: float = 0.10
    rebalance_every: int = 1
    universe: str = "all_equity"
    continuous_only: bool = False
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class TiltYearRow:
    """Per-year aggregated statistics.

    Attributes:
        year: Calendar year (-1 = "ALL" sentinel for pooled totals).
        n_sessions: Number of included sessions.
        gross_bps: Mean daily index-relative excess, before cost.
        turnover: Mean daily sum of |weight change|.
        cost_bps: Mean daily cost = round_trip_bps(clip) * turnover.
        net_bps: gross_bps - cost_bps.
        ann_net_pct: Annualized net return (net_bps * 252 / 100).
    """

    year: int
    n_sessions: int
    gross_bps: float
    turnover: float
    cost_bps: float
    net_bps: float
    ann_net_pct: float


@dataclasses.dataclass(frozen=True)
class TiltResult:
    """Complete backtest result.

    Attributes:
        config: The configuration used.
        per_year: Tuple of TiltYearRow, one per calendar year (sorted ascending).
        total: TiltYearRow with year=-1 representing pooled statistics.
        clip_per_name: capital / n_held; used to compute round_trip_bps.
        round_trip_bps: One-leg round-trip cost at clip_per_name.
        breakeven_turnover: gross_bps / round_trip_bps; inf if gross <= 0.
        n_symbols: Total symbols in universe.
        warnings: Tuple of warning messages (skipped sessions, etc.).
        min_weight_seen: Minimum weight over all sessions (should be >= 0.0).
        max_weight_sum_deviation: Max |sum(weights) - 1.0| over all sessions.
        max_n_held: Largest number of non-zero holdings in any session.
    """

    config: TiltConfig
    per_year: tuple[TiltYearRow, ...]
    total: TiltYearRow
    clip_per_name: float
    round_trip_bps: float
    breakeven_turnover: float
    n_symbols: int
    warnings: tuple[str, ...]
    min_weight_seen: float
    max_weight_sum_deviation: float
    max_n_held: int

    def to_table(self) -> str:
        """Return a human-readable summary table."""
        lines = []
        lines.append(
            f"Tilt backtest  {self.config.tilt} / smoothing {self.config.smoothing:.2f} / "
            f"capital Rs {self.config.capital:,.0f}"
        )
        lines.append(
            f"Universe {self.config.universe} ({self.n_symbols} names)   "
            f"{self.config.start} .. {self.config.end}   {self.total.n_sessions} sessions"
        )
        lines.append(
            f"Clip per name Rs {self.clip_per_name:,.0f}   "
            f"round-trip {self.round_trip_bps:.2f} bps   "
            f"breakeven turnover {self.breakeven_turnover:.2f}"
        )
        lines.append("")
        lines.append(
            "year   sessions   gross_bps   turnover   cost_bps   net_bps   ann_net%"
        )

        for row in self.per_year:
            lines.append(
                f"{row.year:<6} {row.n_sessions:<9} {row.gross_bps:>9.2f} "
                f"{row.turnover:>10.4f} {row.cost_bps:>9.2f} {row.net_bps:>8.2f} "
                f"{row.ann_net_pct:>8.2f}"
            )

        lines.append("-" * 68)
        total_row = self.total
        lines.append(
            f"{'ALL':<6} {total_row.n_sessions:<9} {total_row.gross_bps:>9.2f} "
            f"{total_row.turnover:>10.4f} {total_row.cost_bps:>9.2f} "
            f"{total_row.net_bps:>8.2f} {total_row.ann_net_pct:>8.2f}"
        )

        return "\n".join(lines)

    def explain(self) -> str:
        """Return provenance and methodology notes."""
        lines = []
        lines.append(f"Configuration: {self.config}")
        lines.append(
            f"Universe: {self.config.universe} ({self.n_symbols} symbols across window)"
        )
        lines.append(
            f"Window: {self.config.start} to {self.config.end} "
            f"({self.total.n_sessions} sessions included)"
        )
        lines.append(
            f"Entry/exit: {self.config.entry_hhmm} / {self.config.exit_hhmm} "
            f"(resolved by time label, not position)"
        )
        lines.append(
            f"Tilt: {self.config.tilt} "
            f"(clipped-rank with 0.5 cutoff or bottom quintile)"
        )
        lines.append(
            f"Smoothing: {self.config.smoothing} "
            f"(1.0 = daily rebalance, < 1.0 slows adaptation)"
        )
        lines.append(
            f"Capital: Rs {self.config.capital:,.0f} "
            f"(clip per name: Rs {self.clip_per_name:,.0f})"
        )
        lines.append(
            f"Cost: ONE LEG (not 2x). "
            f"Round-trip at clip: {self.round_trip_bps:.4f} bps. "
            f"Charged on turnover only."
        )
        lines.append(
            "Holdout boundary (GLOBAL): last 12 months of real trading calendar. "
            "Query dates constrained to trade before this boundary."
        )

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


def _build_checkpoint_panel_custom(
    panel: Panel, feature_values: np.ndarray, entry_hhmm: str, exit_hhmm: str
) -> tuple[Panel, np.ndarray, list[str]]:
    """Build checkpoint panel with custom entry/exit times.

    Mimics _build_checkpoint_panel from h2_overnight_reversal but accepts
    configurable entry/exit times. Resolves BOTH labels via minute_of_day(),
    stores entry-side value in reduced panel's close field so horizon=1
    close-to-close return is the true intraday return, and DROPS any session
    missing either label.

    Args:
        panel: Input panel (many bars per session).
        feature_values: (n_rows, n_symbols) feature values, broadcast to both
                       checkpoint rows for each session.
        entry_hhmm: Entry time label ("09:16", "10:00", etc).
        exit_hhmm: Exit time label ("15:20", "14:00", etc).

    Returns:
        (checkpoint_panel, checkpoint_feature_values, warnings) where warnings
        is a list of dates that were dropped due to missing checkpoints.
    """
    entry_rows = panel.rows_at_time(entry_hhmm)
    exit_rows = panel.rows_at_time(exit_hhmm)
    entry_days = np.searchsorted(panel.day_offsets, entry_rows, side="right") - 1
    exit_days = np.searchsorted(panel.day_offsets, exit_rows, side="right") - 1

    entry_row_by_day = dict(zip(entry_days.tolist(), entry_rows.tolist()))
    exit_row_by_day = dict(zip(exit_days.tolist(), exit_rows.tolist()))
    usable_days = sorted(set(entry_row_by_day) & set(exit_row_by_day))

    # Track which days were dropped due to missing checkpoints
    all_days = set(range(panel.n_days()))
    dropped_days = all_days - set(entry_row_by_day.keys()) | (
        all_days - set(exit_row_by_day.keys())
    )
    warnings = []
    for day in sorted(dropped_days):
        if day < len(panel.dates):
            warnings.append(
                f"{panel.dates[day]}: missing {entry_hhmm} or "
                f"{exit_hhmm} checkpoint, session skipped"
            )

    if not usable_days:
        # No usable sessions: return empty checkpoint panel
        n_symbols = panel.n_symbols()
        empty_ts = np.array([], dtype=np.int64)
        empty_close = np.zeros((0, n_symbols), dtype=panel.field("close").dtype)
        empty_volume = np.zeros((0, n_symbols), dtype=panel.field("volume").dtype)
        empty_feature = np.zeros((0, n_symbols), dtype=np.float64)
        empty_panel = Panel(
            fields={"close": empty_close, "volume": empty_volume},
            symbols=panel.symbols,
            ts=empty_ts,
            day_offsets=np.array([0], dtype=np.int32),
            dates=np.array([], dtype=object),
        )
        return empty_panel, empty_feature, warnings

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
        # Store entry-side value in "close" field so horizon=1 return is intraday
        close[entry_out, :] = panel_open[entry_row, :]
        close[exit_out, :] = panel_close[exit_row, :]
        volume[entry_out, :] = panel_volume[entry_row, :]
        volume[exit_out, :] = panel_volume[exit_row, :]
        # Broadcast feature to both rows
        feat_values[entry_out, :] = feature_values[entry_row, :]
        feat_values[exit_out, :] = feature_values[entry_row, :]

    from nifty_quant.calendar import SessionGrid

    grid = SessionGrid.from_timestamps(ts)
    checkpoint_panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=panel.symbols,
        ts=grid.ts,
        day_offsets=grid.day_offsets,
        dates=grid.dates,
    )
    return checkpoint_panel, feat_values, warnings


def _compute_mild_weights(feature_valid: np.ndarray) -> np.ndarray:
    """Compute mild-tilt weights: clipped-rank above 0.5 cutoff.

    Non-zero for the bottom 50% by overnight return (biggest losers),
    linearly increasing toward the loser.
    """
    n_valid = len(feature_valid)
    order = np.argsort(feature_valid, kind="stable")
    ranks = np.empty(n_valid, dtype=np.float64)
    ranks[order] = np.arange(n_valid, dtype=np.float64)
    rank_pct = ranks / (n_valid - 1)
    score = np.clip(0.5 - rank_pct, 0.0, None)
    score_sum = float(score.sum())
    if score_sum > 0.0:
        return score / score_sum
    else:
        return np.full(n_valid, 1.0 / n_valid, dtype=np.float64)


def _compute_aggressive_weights(feature_valid: np.ndarray) -> np.ndarray:
    """Compute aggressive-tilt weights: bottom quintile equal-weighted.

    Hold only the bottom 20% by overnight return, equal-weighted.
    """
    n_valid = len(feature_valid)
    order = np.argsort(feature_valid, kind="stable")
    n_bottom = max(1, int(round(n_valid * 0.2)))
    bottom_idx = order[:n_bottom]
    weights = np.zeros(n_valid, dtype=np.float64)
    weights[bottom_idx] = 1.0 / n_bottom
    return weights


def run_tilt(panel: Panel, config: TiltConfig) -> TiltResult:
    """Run the parameterised tilt backtest.

    Args:
        panel: The 1-minute OHLCV panel (many bars per session).
        config: TiltConfig with start, end, entry/exit times, capital, tilt type, etc.

    Returns:
        TiltResult with per-year rows, total aggregates, and diagnostics.

    Raises:
        ValueError: If start > end, no usable sessions, < 5 symbols throughout,
                   or end falls inside the holdout window.
    """
    # Validate input
    if config.start > config.end:
        raise ValueError(
            f"start ({config.start}) must be <= end ({config.end})"
        )

    if config.tilt not in ("mild", "aggressive"):
        raise ValueError(
            f"tilt must be 'mild' or 'aggressive', got {config.tilt!r}"
        )

    # Check holdout protection (GLOBAL, not panel-relative)
    try:
        cal = TradingCalendar.from_index_bars("NIFTY50")
        real_dates = cal.session_dates()
        holdout_lock = HoldoutLock(path=default_holdout_lock_path())
        holdout_start, holdout_end = holdout_lock.holdout_range(real_dates)
        if config.end >= holdout_start:
            raise ValueError(
                f"end date {config.end} falls in holdout window "
                f"[{holdout_start}, {holdout_end}]. "
                f"Holdout boundary is GLOBAL."
            )
    except ValueError:
        raise

    # Subset panel to requested date range
    sliced_panel = panel.sub(start=config.start, end=config.end)
    if sliced_panel.n_rows() == 0:
        raise ValueError(
            f"No data in panel for window [{config.start}, {config.end}]"
        )

    # Build overnight feature on full panel (so left-edge session has valid d-1)
    overnight_feature = build_overnight_feature(panel)

    # Restrict feature to the sliced window
    if sliced_panel.n_rows() > 0:
        row_start = int(np.searchsorted(panel.ts, sliced_panel.ts[0]))
        row_end = row_start + sliced_panel.n_rows()
        sliced_feature_values = overnight_feature.values[row_start:row_end]
    else:
        sliced_feature_values = overnight_feature.values[:0]

    # Build checkpoint panel with custom entry/exit times
    checkpoint_panel, checkpoint_feature, checkpoint_warnings = _build_checkpoint_panel_custom(
        sliced_panel, sliced_feature_values, config.entry_hhmm, config.exit_hhmm
    )

    if checkpoint_panel.n_rows() == 0:
        raise ValueError(
            f"No sessions with both {config.entry_hhmm} and {config.exit_hhmm} "
            f"checkpoints in window [{config.start}, {config.end}]"
        )

    n_sessions = checkpoint_panel.n_days()
    close_field = checkpoint_panel.field("close")

    # Track warnings
    warnings: list[str] = list(checkpoint_warnings)

    # Precompute per-session data
    # Type: (date, benchmark_return, r_full, target_full, raw_return_all, feat_valid)
    session_records: list[
        tuple[datetime.date, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    min_weight_global = 1.0
    max_weight_deviation_global = 0.0
    max_n_held_global = 0

    for day_idx in range(n_sessions):
        date = checkpoint_panel.dates[day_idx]
        entry_row = 2 * day_idx
        exit_row = 2 * day_idx + 1

        # Get values for this session
        feat = checkpoint_feature[entry_row, :].astype(np.float64)
        price_entry = close_field[entry_row, :].astype(np.float64)
        price_exit = close_field[exit_row, :].astype(np.float64)

        # Determine valid names
        valid = (
            np.isfinite(feat)
            & np.isfinite(price_entry)
            & np.isfinite(price_exit)
            & (price_entry > 0)
        )
        n_valid = int(valid.sum())

        # Skip if < 5 valid names
        if n_valid < 5:
            warnings.append(
                f"{date}: only {n_valid} valid names (need >= 5), session skipped"
            )
            continue

        # Compute returns and benchmark
        r_valid = price_exit[valid] / price_entry[valid] - 1.0
        benchmark_return = float(np.mean(r_valid))

        # Full-length return vector (zero for invalid names)
        r = np.zeros(len(price_entry), dtype=np.float64)
        r[valid] = r_valid

        # Compute target weights
        if config.tilt == "mild":
            mild_weights_valid = _compute_mild_weights(feat[valid])
            target_valid = mild_weights_valid
        else:  # aggressive
            aggressive_weights_valid = _compute_aggressive_weights(feat[valid])
            target_valid = aggressive_weights_valid

        # Full-length target weight vector
        target_full = np.zeros(len(price_entry), dtype=np.float64)
        target_full[valid] = target_valid

        # Track weight diagnostics
        min_weight_global = min(min_weight_global, float(np.min(target_full)))
        weight_sum = float(np.sum(target_full))
        max_weight_deviation_global = max(
            max_weight_deviation_global, abs(weight_sum - 1.0)
        )
        max_n_held_global = max(max_n_held_global, int(np.count_nonzero(target_full)))

        # Compute raw returns (for drift calculation)
        raw_return_all = price_exit / price_entry - 1.0
        raw_return_all[
            ~(np.isfinite(price_entry) & (price_entry > 0) & np.isfinite(price_exit))
        ] = np.nan

        session_records.append(
            (date, benchmark_return, r, target_full, raw_return_all, feat[valid])
        )

    if not session_records:
        raise ValueError(
            "No sessions with >= 5 valid names in window "
            f"[{config.start}, {config.end}]"
        )

    # Simulate with smoothing and rebalance frequency
    n_symbols = panel.n_symbols()
    book = np.zeros(n_symbols, dtype=np.float64)
    # Type: (date, excess_bps, turnover, n_held)
    daily_records: list[tuple[datetime.date, float, float, int]] = []
    n_held_counts: list[int] = []

    for session_idx, session_data in enumerate(session_records):
        date, benchmark_return, r, target_full, raw_return_all, feat_valid = session_data
        if session_idx % config.rebalance_every == 0:
            # Rebalance session: update book toward target
            book_start = (1.0 - config.smoothing) * book + config.smoothing * target_full
            turnover = float(np.sum(np.abs(book_start - book)))
        else:
            # Hold session: apply drift
            book_start = book.copy()
            turnover = 0.0
            # Apply drift using raw returns, filling NaNs with 0 for held positions
            held_mask = book_start != 0.0
            raw_filled = np.where(held_mask & np.isnan(raw_return_all), 0.0, raw_return_all)
            raw_filled = np.where(np.isnan(raw_filled), 0.0, raw_filled)
            denom = 1.0 + float(np.dot(book_start, raw_filled))
            if denom > 1e-6:
                book_start = book_start * (1.0 + raw_filled) / denom

        # Compute session return
        return_j = float(np.dot(book_start, r))
        excess_bps = (return_j - benchmark_return) * 10_000.0
        n_held = int(np.count_nonzero(book_start))

        daily_records.append((date, excess_bps, turnover, n_held))
        n_held_counts.append(n_held)

        # Update book for next session
        if session_idx % config.rebalance_every == 0:
            book = target_full.copy()
        else:
            book = book_start.copy()

    # Aggregate by year and compute costs
    costs = NSEIntradayEquityCosts()
    n_held_typical = max(1, int(np.mean(n_held_counts))) if n_held_counts else 1
    clip_per_name = config.capital / n_held_typical
    round_trip_bps_val = costs.round_trip_bps(clip_per_name)

    by_year: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for date, excess_bps, turnover, _ in daily_records:
        cost_bps = round_trip_bps_val * turnover
        net_bps = excess_bps - cost_bps
        by_year[date.year].append((excess_bps, turnover, cost_bps, net_bps))

    per_year_rows = []
    for year in sorted(by_year.keys()):
        records_year = by_year[year]
        n_sess = len(records_year)
        gross_bps = float(np.mean([r[0] for r in records_year]))
        turnover_mean = float(np.mean([r[1] for r in records_year]))
        cost_bps = float(np.mean([r[2] for r in records_year]))
        net_bps = float(np.mean([r[3] for r in records_year]))
        ann_net_pct = net_bps * 252.0 / 100.0

        per_year_rows.append(
            TiltYearRow(
                year=year,
                n_sessions=n_sess,
                gross_bps=gross_bps,
                turnover=turnover_mean,
                cost_bps=cost_bps,
                net_bps=net_bps,
                ann_net_pct=ann_net_pct,
            )
        )

    # Compute total aggregates
    total_records = [r for year_recs in by_year.values() for r in year_recs]
    total_n_sessions = len(total_records)
    total_gross_bps = float(np.mean([r[0] for r in total_records]))
    total_turnover = float(np.mean([r[1] for r in total_records]))
    total_cost_bps = float(np.mean([r[2] for r in total_records]))
    total_net_bps = float(np.mean([r[3] for r in total_records]))
    total_ann_net_pct = total_net_bps * 252.0 / 100.0

    total_row = TiltYearRow(
        year=-1,
        n_sessions=total_n_sessions,
        gross_bps=total_gross_bps,
        turnover=total_turnover,
        cost_bps=total_cost_bps,
        net_bps=total_net_bps,
        ann_net_pct=total_ann_net_pct,
    )

    # Compute breakeven turnover
    if total_gross_bps > 0:
        breakeven_turnover = total_gross_bps / round_trip_bps_val
    else:
        breakeven_turnover = float("inf")

    # Adjust weight diagnostics
    if min_weight_global > 0.99:
        min_weight_global = 0.0  # All weights were zeros

    return TiltResult(
        config=config,
        per_year=tuple(per_year_rows),
        total=total_row,
        clip_per_name=clip_per_name,
        round_trip_bps=round_trip_bps_val,
        breakeven_turnover=breakeven_turnover,
        n_symbols=n_symbols,
        warnings=tuple(warnings),
        min_weight_seen=min_weight_global,
        max_weight_sum_deviation=max_weight_deviation_global,
        max_n_held=max_n_held_global,
    )
