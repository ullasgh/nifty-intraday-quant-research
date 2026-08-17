"""Audit for corporate-action adjustment consistency in spot bars.

Detects mismatch between price and volume adjustments in spot bars by analyzing
traded-value steps via rolling medians. Emits events for splits, bonuses, and
other corporate actions where the adjustment of price and volume diverges.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from concurrent.futures import as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from nifty_quant import settings
from nifty_quant.universe.static import equity_symbols


class AdjustmentClass(StrEnum):
    """Classification of a suspected adjustment event."""

    CONSISTENT = "consistent"
    VOLUME_UNADJUSTED = "volume_unadjusted"
    PRICE_UNADJUSTED = "price_unadjusted"
    AMBIGUOUS = "ambiguous"


# Common NSE split/bonus factors and their inverses.
# 2, 3, 4, 5, 10, 20, 10/3, and their reciprocals; 3/2, 2/3, 5/2 removed
# as noise-indistinguishable. Sorted ascending, deduplicated.
PLAUSIBLE_FACTORS: Final[tuple[float, ...]] = (
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    1.0 / 3.0,
    0.5,
    2.0,
    3.0,
    10.0 / 3.0,
    4.0,
    5.0,
    10.0,
    20.0,
)


@dataclass(frozen=True)
class SuspectEvent:
    """A suspected corporate-action adjustment event."""

    symbol: str
    date: datetime.date
    price_ratio: float  # close[d] / close[d-1], raw adjacent-day
    volume_ratio: float  # median(volume[d : d+w]) / median(volume[d-w : d])
    traded_value_ratio: float  # same window medians, on close*volume
    nearest_factor: float | None  # closest entry in PLAUSIBLE_FACTORS, or None
    factor_error: float  # |log(traded_value_ratio) - log(nearest_factor)|; inf if None
    classification: AdjustmentClass
    n_days_before: int  # actual non-NaN days used in trailing window
    n_days_after: int  # actual non-NaN days used in leading window

    def explain(self) -> str:
        """Multi-line human-readable provenance of classification.

        Names every input, the thresholds applied, and why this classification
        was chosen over the others.
        """
        lines: list[str] = []
        lines.append(
            f"Event: {self.symbol} on {self.date.isoformat()}"
        )
        lines.append(
            f"  price_ratio (close[d]/close[d-1]): {self.price_ratio:.6f}"
        )
        lines.append(
            f"  volume_ratio (leading median / trailing median): {self.volume_ratio:.6f}"
        )
        lines.append(
            f"  traded_value_ratio (leading/trailing median): {self.traded_value_ratio:.6f}"
        )
        lines.append(
            f"  nearest_factor: {self.nearest_factor}"
        )
        lines.append(
            f"  factor_error: {self.factor_error:.6f}"
        )
        lines.append(
            f"  n_days_before (trailing): {self.n_days_before}"
        )
        lines.append(
            f"  n_days_after (leading): {self.n_days_after}"
        )
        lines.append("")
        lines.append(
            f"classification: {self.classification.value}"
        )
        lines.append("")
        lines.append("Thresholds applied:")
        lines.append("  window_days: 20 (static in explain)")
        lines.append("  min_traded_value_log_step: 2.3")
        lines.append("  min_volume_log_step: 2.1")
        lines.append("  factor_tolerance: 0.04")
        lines.append("  max_price_step: 0.05")
        lines.append("")
        lines.append("Classification logic:")

        price_step = abs(np.log(self.price_ratio))

        if self.classification == AdjustmentClass.VOLUME_UNADJUSTED:
            lines.append(
                f"  1. nearest_factor is not None (found {self.nearest_factor})"
            )
            lines.append(
                f"  2. price_step = |log(price_ratio)| = {price_step:.6f} <= max_price_step (0.05)"
            )
            lines.append(
                "  -> Volume step detected with no price step = VOLUME_UNADJUSTED"
            )
        elif self.classification == AdjustmentClass.PRICE_UNADJUSTED:
            lines.append(
                f"  1. nearest_factor is not None (found {self.nearest_factor})"
            )
            lines.append(
                f"  2. price_step = |log(price_ratio)| = {price_step:.6f} > max_price_step (0.05)"
            )
            recip_sum = abs(
                np.log(self.price_ratio) + np.log(self.traded_value_ratio)
            )
            sum_tol = "factor_tolerance (0.04)"
            lines.append(
                f"  3. |log(price_ratio) + log(traded_value_ratio)| = "
                f"{recip_sum:.6f} <= {sum_tol}"
            )
            lines.append(
                "  -> Price and traded_value steps are reciprocal = PRICE_UNADJUSTED"
            )
        elif self.classification == AdjustmentClass.AMBIGUOUS:
            if self.nearest_factor is None:
                lines.append(
                    f"  1. nearest_factor is None "
                    f"(factor_error {self.factor_error:.6f} > tol 0.04)"
                )
            else:
                lines.append(
                    f"  1. nearest_factor found: {self.nearest_factor}"
                )
                lines.append(
                    f"  2. price_step = |log(price_ratio)| = "
                    f"{price_step:.6f} > max_price_step (0.05)"
                )
                recip_sum = abs(
                    np.log(self.price_ratio)
                    + np.log(self.traded_value_ratio)
                )
                lines.append(
                    f"  3. |log(price_ratio) + log(traded_value_ratio)| = "
                    f"{recip_sum:.6f} > factor_tolerance (0.04)"
                )
            lines.append("  -> Cannot classify with confidence = AMBIGUOUS")

        return "\n".join(lines)


@dataclass(frozen=True)
class AuditParams:
    """Parameters controlling the audit algorithm.

    Calibration measured across the 149-symbol panel found traded-value
    abs(log(20-day median ratio)) quantiles of p50=0.231, p90=0.624,
    p99=1.280, p99.9=2.340, p99.99=3.117, and max=3.944; the volume-only
    p99.9 was 2.142. Thus thresholds of 2.3 and 2.1, with NMS, produce
    approximately 19-25 events rather than thousands. The revised factor list
    has a tightest adjacent log gap of 0.1054, so tolerance 0.04 is below
    half that gap (0.0527). Deliberately, ordinary per-day traded-value noise
    reaches about 1.87x at the 90th percentile, so this detector cannot
    reliably distinguish common 2x-5x factors from noise; it accepts blindness
    to roughly 2x-6x factors, leaving only factors with abs(log(factor)) >= 2.3
    (10x, 20x, and reciprocals 0.1, 0.05) reachable by default.
    """

    window_days: int = 20
    min_traded_value_log_step: float = 2.3
    min_volume_log_step: float = 2.1
    factor_tolerance: float = 0.04
    max_price_step: float = 0.05
    min_days_each_side: int = 10

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.window_days < 2:
            raise ValueError("window_days must be >= 2")
        if self.min_traded_value_log_step <= 0:
            raise ValueError("min_traded_value_log_step must be > 0")
        if self.min_volume_log_step <= 0:
            raise ValueError("min_volume_log_step must be > 0")
        if self.factor_tolerance <= 0:
            raise ValueError("factor_tolerance must be > 0")
        if self.max_price_step <= 0:
            raise ValueError("max_price_step must be > 0")
        if self.min_days_each_side > self.window_days:
            raise ValueError("min_days_each_side must be <= window_days")


@dataclass(frozen=True)
class AuditReport:
    """Report of a complete adjustment audit across symbols and years."""

    events: tuple[SuspectEvent, ...]
    symbols_scanned: tuple[str, ...]
    years: tuple[int, ...]
    params: AuditParams

    def by_class(self, cls: AdjustmentClass) -> tuple[SuspectEvent, ...]:
        """Filter events by classification.

        CONSISTENT returns empty tuple since it is never emitted as an event.
        """
        if cls == AdjustmentClass.CONSISTENT:
            return ()
        return tuple(e for e in self.events if e.classification == cls)

    def exclusion_windows(self) -> dict[str, tuple[tuple[datetime.date, datetime.date], ...]]:
        """Generate exclusion windows around problematic events.

        Only VOLUME_UNADJUSTED and PRICE_UNADJUSTED events produce windows.
        A window spans [event_date - params.window_days, event_date + params.window_days],
        inclusive on both ends. Overlapping windows for the same symbol are merged.
        """
        result: dict[str, list[tuple[datetime.date, datetime.date]]] = {}

        for event in self.events:
            # Skip AMBIGUOUS and CONSISTENT events
            if event.classification not in {
                AdjustmentClass.VOLUME_UNADJUSTED,
                AdjustmentClass.PRICE_UNADJUSTED,
            }:
                continue

            start = event.date - datetime.timedelta(days=self.params.window_days)
            end = event.date + datetime.timedelta(days=self.params.window_days)

            if event.symbol not in result:
                result[event.symbol] = []
            result[event.symbol].append((start, end))

        # Merge overlapping windows for each symbol
        merged: dict[str, tuple[tuple[datetime.date, datetime.date], ...]] = {}
        for symbol, windows in result.items():
            # Sort by start date
            sorted_windows = sorted(windows)

            merged_list: list[tuple[datetime.date, datetime.date]] = []
            for start, end in sorted_windows:
                if merged_list and start <= merged_list[-1][1]:
                    # Overlapping or adjacent, merge
                    merged_list[-1] = (merged_list[-1][0], max(merged_list[-1][1], end))
                else:
                    merged_list.append((start, end))

            merged[symbol] = tuple(merged_list)

        return merged

    def is_clean(self) -> bool:
        """True iff there are no VOLUME_UNADJUSTED and no PRICE_UNADJUSTED events."""
        return all(
            e.classification
            not in {
                AdjustmentClass.VOLUME_UNADJUSTED,
                AdjustmentClass.PRICE_UNADJUSTED,
            }
            for e in self.events
        )

    def explain(self) -> str:
        """Multi-line human-readable summary of the audit report."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("ADJUSTMENT AUDIT REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Symbols scanned: {len(self.symbols_scanned)}")
        year_range = (
            f"{min(self.years)} to {max(self.years)}"
            if self.years
            else "none to none"
        )
        lines.append(f"Years scanned: {year_range}")
        lines.append("")

        lines.append("Parameters:")
        lines.append(f"  window_days: {self.params.window_days}")
        lines.append(f"  min_traded_value_log_step: {self.params.min_traded_value_log_step}")
        lines.append(f"  min_volume_log_step: {self.params.min_volume_log_step}")
        lines.append(f"  factor_tolerance: {self.params.factor_tolerance}")
        lines.append(f"  max_price_step: {self.params.max_price_step}")
        lines.append(f"  min_days_each_side: {self.params.min_days_each_side}")
        lines.append("")

        lines.append("Event counts by classification:")
        for cls in AdjustmentClass:
            count = len(self.by_class(cls))
            lines.append(f"  {cls.name}: {count}")
        lines.append("")

        if self.is_clean():
            lines.append("STATUS: CLEAN (no volume or price unadjusted events)")
        else:
            lines.append("STATUS: DIRTY (contains volume or price unadjusted events)")
            lines.append("")
            lines.append("Detailed events:")
            for event in self.events:
                if event.classification in {
                    AdjustmentClass.VOLUME_UNADJUSTED,
                    AdjustmentClass.PRICE_UNADJUSTED,
                }:
                    lines.append(
                        f"  {event.symbol} {event.date.isoformat()} {event.classification.value}"
                    )

        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Convert events to a DataFrame.

        Columns: symbol, date, price_ratio, volume_ratio, traded_value_ratio,
                 nearest_factor, factor_error, classification, n_days_before, n_days_after
        Rows sorted by (symbol, date).
        """
        if not self.events:
            return pd.DataFrame(
                {
                    "symbol": pd.Series(dtype=object),
                    "date": pd.Series(dtype=object),
                    "price_ratio": pd.Series(dtype=np.float64),
                    "volume_ratio": pd.Series(dtype=np.float64),
                    "traded_value_ratio": pd.Series(dtype=np.float64),
                    "nearest_factor": pd.Series(dtype=object),
                    "factor_error": pd.Series(dtype=np.float64),
                    "classification": pd.Series(dtype=object),
                    "n_days_before": pd.Series(dtype=np.int64),
                    "n_days_after": pd.Series(dtype=np.int64),
                }
            )

        rows = []
        for event in sorted(self.events, key=lambda e: (e.symbol, e.date)):
            rows.append(
                {
                    "symbol": event.symbol,
                    "date": event.date,
                    "price_ratio": event.price_ratio,
                    "volume_ratio": event.volume_ratio,
                    "traded_value_ratio": event.traded_value_ratio,
                    "nearest_factor": event.nearest_factor,
                    "factor_error": event.factor_error,
                    "classification": event.classification.value,
                    "n_days_before": event.n_days_before,
                    "n_days_after": event.n_days_after,
                }
            )

        df = pd.DataFrame(rows)
        # Ensure correct dtypes: string columns should be object, not StringDtype
        df["symbol"] = df["symbol"].astype(object)
        df["date"] = df["date"].astype(object)
        df["classification"] = df["classification"].astype(object)
        return df


def daily_aggregate(symbol: str, years: Sequence[int]) -> pd.DataFrame:
    """Aggregate 1-minute bars to daily for one symbol.

    Reads data/bars/1/<symbol>/<year>.parquet. Returns a DataFrame indexed by
    `date` (datetime.date, ascending, unique) with float64 columns:
      open   -- first non-NaN open of the session
      high   -- max high
      low    -- min low
      close  -- last non-NaN close of the session
      volume -- sum of volume
      traded_value -- sum of (close * volume) over bars, NOT close_daily * volume_daily
      n_bars -- int64 count of bars present in the session

    Sessions are derived from the actual timestamps (IST calendar date), never from a fixed
    375-bar stride. NaN bars are excluded from the aggregation, never forward-filled; a
    session with zero non-NaN bars is omitted from the result entirely.
    """
    frames = []

    for year in years:
        parquet_path = settings.BARS_1M / symbol / f"{year}.parquet"
        if not parquet_path.exists():
            continue

        df_raw = pd.read_parquet(parquet_path)

        # Convert ts (int64 epoch-seconds UTC) to IST dates
        df_raw["ts"] = df_raw["ts"].astype(np.int64)
        df_raw["date"] = pd.to_datetime(df_raw["ts"], unit="s", utc=True).dt.tz_convert(
            "Asia/Kolkata"
        ).dt.date

        # Cast to float64 (float32 at rest, float64 in motion)
        for col in ["open", "high", "low", "close", "volume"]:
            df_raw[col] = df_raw[col].astype(np.float64)

        # Calculate traded_value as sum of close * volume for each bar
        df_raw["traded_value"] = df_raw["close"] * df_raw["volume"]

        # Group by date
        grouped = df_raw.groupby("date", as_index=False)

        agg_data = []
        for date_val, group in grouped:
            # Exclude NaN bars from aggregation
            group_clean = group.dropna(subset=["open", "high", "low", "close", "volume"])

            if len(group_clean) == 0:
                # Skip sessions with zero non-NaN bars
                continue

            row = {
                "date": date_val,
                "open": group_clean["open"].iloc[0],
                "high": group_clean["high"].max(),
                "low": group_clean["low"].min(),
                "close": group_clean["close"].iloc[-1],
                "volume": group_clean["volume"].sum(),
                "traded_value": group_clean["traded_value"].sum(),
                "n_bars": len(group_clean),
            }
            agg_data.append(row)

        if agg_data:
            frame = pd.DataFrame(agg_data)
            frame = frame.set_index("date")
            frames.append(frame)

    if frames:
        df = pd.concat(frames, axis=0)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]  # In case of overlap
    else:
        df = pd.DataFrame(
            {
                "open": pd.Series(dtype=np.float64),
                "high": pd.Series(dtype=np.float64),
                "low": pd.Series(dtype=np.float64),
                "close": pd.Series(dtype=np.float64),
                "volume": pd.Series(dtype=np.float64),
                "traded_value": pd.Series(dtype=np.float64),
                "n_bars": pd.Series(dtype=np.int64),
            }
        )
        df.index.name = "date"

    return df


def scan_symbol(
    symbol: str, years: Sequence[int], params: AuditParams = AuditParams()
) -> tuple[SuspectEvent, ...]:
    """Scan a single symbol for adjustment anomalies.

    Returns a tuple of SuspectEvent objects, sorted by date.
    """
    df = daily_aggregate(symbol, years)

    # Need at least 2 * window_days + 1 rows
    if len(df) < 2 * params.window_days + 1:
        return ()

    # (index, |log(traded_value_ratio)|, traded_value_ratio)
    candidates: list[tuple[int, float, float]] = []

    # Iterate over candidate indices
    for i in range(params.window_days, len(df) - params.window_days):
        # Trailing window: [i - window_days, i)
        # Leading window: [i, i + window_days)
        trailing = df.iloc[i - params.window_days : i]
        leading = df.iloc[i : i + params.window_days]

        # Count finite traded_value in each window
        n_before = trailing["traded_value"].notna().sum()
        n_after = leading["traded_value"].notna().sum()

        if n_before < params.min_days_each_side or n_after < params.min_days_each_side:
            continue

        # Calculate medians
        trailing_median_tv = trailing["traded_value"].median()
        leading_median_tv = leading["traded_value"].median()

        if not np.isfinite(trailing_median_tv) or trailing_median_tv <= 0:
            continue

        traded_value_ratio = leading_median_tv / trailing_median_tv

        # Calculate volume ratio for reporting
        trailing_median_vol = trailing["volume"].median()
        leading_median_vol = leading["volume"].median()
        if trailing_median_vol > 0:
            volume_ratio = leading_median_vol / trailing_median_vol
        else:
            volume_ratio = np.nan

        if not (
            np.isfinite(volume_ratio)
            and abs(np.log(volume_ratio)) >= params.min_volume_log_step
        ):
            continue

        # Price ratio (raw adjacent-day)
        price_ratio = df["close"].iloc[i] / df["close"].iloc[i - 1]

        # Check if |log(traded_value_ratio)| >= min_traded_value_log_step
        log_tv_step = abs(np.log(traded_value_ratio))
        if log_tv_step >= params.min_traded_value_log_step:
            candidates.append((i, log_tv_step, traded_value_ratio))

    # Non-maximum suppression: group candidates by gaps to the immediately
    # preceding candidate, rather than by span from the group's first candidate.
    # A single noisy regime-change run was observed to span 56 trading days in
    # ZYDUSLIFE 2019, so anchoring every group at its first candidate split one
    # underlying cluster into multiple events.
    if not candidates:
        return ()

    # Sort by index
    candidates.sort(key=lambda x: x[0])

    # Suppress: for each gap-to-previous contiguous group, keep only max
    # Tie-break via adjacent-day jump (see spec §4 AMENDED 2026-08-17)
    def _tie_key(idx: int) -> float:
        """Adjacent-day jump magnitude: abs(log(tv[i] / tv[i-1])).

        The windowed ratio (leading_median / trailing_median) is identical for
        all ~19 consecutive candidates around the true ex-date, because the
        median flips once >50% of the window rows land on the new basis. The
        adjacent-day jump, however, is uniquely maximized at the ex-date itself,
        where i-1 is the last old-basis day and i is the first new-basis day.
        This makes it the only quantity that can disambiguate within a tie.
        """
        tv_i = df["traded_value"].iloc[idx]
        tv_i_minus_1 = df["traded_value"].iloc[idx - 1]

        if (
            not np.isfinite(tv_i)
            or not np.isfinite(tv_i_minus_1)
            or tv_i_minus_1 <= 0
        ):
            return -np.inf  # Non-finite or bad: lose any tie

        return float(abs(np.log(tv_i / tv_i_minus_1)))

    kept_indices: list[int] = []
    i = 0
    while i < len(candidates):
        max_idx = i
        max_val = candidates[i][1]
        max_tie_key = _tie_key(candidates[i][0])
        j = i + 1
        # Look ahead while each consecutive candidate gap is within
        # window_days, allowing an arbitrarily wide contiguous run to remain
        # one suppression group.
        while j < len(candidates) and candidates[j][0] - candidates[j - 1][0] <= params.window_days:
            if candidates[j][1] > max_val:
                # Higher log_step: always prefer
                max_val = candidates[j][1]
                max_idx = j
                max_tie_key = _tie_key(candidates[j][0])
            elif candidates[j][1] == max_val:
                # Same log_step: use tie_key to pick the true ex-date. If
                # tie_key is ALSO tied, do nothing: candidates are visited in
                # increasing day-index order, so max_idx already points at the
                # earlier date, which is the desired tie-break for determinism
                # (candidates[j][0] > candidates[max_idx][0] always holds here,
                # so an explicit "prefer earlier" branch would be unreachable).
                tie_key_j = _tie_key(candidates[j][0])
                if tie_key_j > max_tie_key:
                    max_tie_key = tie_key_j
                    max_idx = j
            j += 1

        kept_indices.append(candidates[max_idx][0])
        i = j

    # Build events from kept indices
    events: list[SuspectEvent] = []

    for idx in kept_indices:
        trailing = df.iloc[idx - params.window_days : idx]
        leading = df.iloc[idx : idx + params.window_days]

        n_before = trailing["traded_value"].notna().sum()
        n_after = leading["traded_value"].notna().sum()

        trailing_median_tv = trailing["traded_value"].median()
        leading_median_tv = leading["traded_value"].median()
        traded_value_ratio = leading_median_tv / trailing_median_tv

        trailing_median_vol = trailing["volume"].median()
        leading_median_vol = leading["volume"].median()
        # idx only reaches this loop after passing the identical
        # min_volume_log_step gate on identical data in the candidate pass above,
        # which already required trailing_median_vol > 0 and a finite,
        # over-threshold volume_ratio; recomputing that branch here would be
        # provably dead code (same slice, same deterministic median), so this
        # is a plain division, not a re-guarded one.
        volume_ratio = leading_median_vol / trailing_median_vol

        price_ratio = df["close"].iloc[idx] / df["close"].iloc[idx - 1]

        # Find nearest factor
        log_tv = np.log(traded_value_ratio)
        factor_errors: list[tuple[float, float]] = []
        for factor in PLAUSIBLE_FACTORS:
            error = abs(log_tv - np.log(factor))
            factor_errors.append((error, factor))

        factor_errors.sort(key=lambda x: x[0])
        min_error, nearest_factor_cand = factor_errors[0]

        if min_error <= params.factor_tolerance:
            nearest_factor = nearest_factor_cand
            factor_error = min_error
        else:
            nearest_factor = None
            factor_error = np.inf

        # Classify
        price_step = abs(np.log(price_ratio))

        if nearest_factor is None:
            classification = AdjustmentClass.AMBIGUOUS
        elif price_step <= params.max_price_step:
            classification = AdjustmentClass.VOLUME_UNADJUSTED
        elif (
            abs(np.log(price_ratio) + np.log(traded_value_ratio))
            <= params.factor_tolerance
        ):
            classification = AdjustmentClass.PRICE_UNADJUSTED
        else:
            classification = AdjustmentClass.AMBIGUOUS

        event_date = df.index[idx]

        event = SuspectEvent(
            symbol=symbol,
            date=event_date,
            price_ratio=price_ratio,
            volume_ratio=volume_ratio,
            traded_value_ratio=traded_value_ratio,
            nearest_factor=nearest_factor,
            factor_error=factor_error,
            classification=classification,
            n_days_before=n_before,
            n_days_after=n_after,
        )
        events.append(event)

    return tuple(events)


def run_audit(
    symbols: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    params: AuditParams = AuditParams(),
    *,
    workers: int = 8,
) -> AuditReport:
    """Run a complete adjustment audit across symbols and years.

    symbols=None -> universe.static.equity_symbols()
    years=None -> 2018..2026 (DEFAULT_RESEARCH_START is 2018, and tests show 2018-2026)
    """
    if symbols is None:
        symbols = equity_symbols()
    if years is None:
        years = list(range(2018, 2027))

    symbols_tuple = tuple(symbols)
    years_tuple = tuple(years)

    # Scan symbols in parallel
    all_events: list[SuspectEvent] = []

    if workers == 1:
        # Single-threaded for determinism
        for symbol in symbols_tuple:
            events = scan_symbol(symbol, years_tuple, params)
            all_events.extend(events)
    else:
        # Multi-threaded using ThreadPoolExecutor (more reliable than ProcessPoolExecutor)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(scan_symbol, symbol, years_tuple, params): symbol
                for symbol in symbols_tuple
            }
            for future in as_completed(futures):
                events = future.result()
                all_events.extend(events)

    # Sort events by (symbol, date)
    all_events.sort(key=lambda e: (e.symbol, e.date))

    return AuditReport(
        events=tuple(all_events),
        symbols_scanned=symbols_tuple,
        years=years_tuple,
        params=params,
    )


def main(args: list[str] | None = None) -> int:
    """CLI entry point.

    Usage: python -m nifty_quant.research.audit.adjustment_audit \\
        --years START-END [--symbols A,B] [--out PATH]
    """
    parser = argparse.ArgumentParser(
        description="Audit spot bars for adjustment consistency."
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Year range in format START-END (e.g., 2018-2026)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: all equity symbols)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write results as parquet",
    )

    parsed = parser.parse_args(args)

    # Parse years
    year_parts = parsed.years.split("-")
    if len(year_parts) != 2:
        print("Error: --years must be in format START-END (e.g., 2018-2026)", file=sys.stderr)
        return 1

    try:
        year_start = int(year_parts[0])
        year_end = int(year_parts[1])
    except ValueError:
        print("Error: --years must contain valid integers", file=sys.stderr)
        return 1

    years = list(range(year_start, year_end + 1))

    # Parse symbols
    if parsed.symbols:
        symbols = [s.strip() for s in parsed.symbols.split(",")]
    else:
        symbols = None

    # Run audit
    report = run_audit(symbols=symbols, years=years)

    # Print report
    print(report.explain())

    # Write to file if requested
    if parsed.out:
        out_path = Path(parsed.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = report.to_frame()
        df.to_parquet(out_path, index=False)
        print(f"Results written to {out_path}")

    # Exit code based on cleanliness
    return 0 if report.is_clean() else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())  # pragma: no cover
