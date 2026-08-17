"""Validate and mask the already-adjusted spot one-minute panel.

This module never repairs, re-adjusts, or forward-fills bars. NaN means "no bar"
and stays NaN. A bar may be present without being tradable: tradable requires
the bar to exist, not be circuit-locked, not be part of a stale run, and to pass
a trailing ADV liquidity floor.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd

from nifty_quant import settings
from nifty_quant.calendar import SessionKind, TradingCalendar, to_ist
from nifty_quant.data.panel import Panel


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    symbol: str | None
    session: date | None
    count: int
    detail: str


@dataclass(frozen=True)
class DataQualityReport:
    findings: tuple[Finding, ...]
    n_rows: int
    n_symbols: int
    n_sessions: int

    def errors(self) -> tuple[Finding, ...]:
        """Return findings with severity ERROR."""
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    def by_check(self) -> dict[str, int]:
        """Count findings per check name."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.check] = counts.get(finding.check, 0) + 1
        return counts

    def to_json(self) -> str:
        """Serialize the report as sorted JSON."""
        findings = [
            {
                "check": f.check,
                "severity": f.severity.value,
                "symbol": f.symbol,
                "session": f.session.isoformat() if f.session else None,
                "count": f.count,
                "detail": f.detail,
            }
            for f in self.findings
        ]
        payload = {
            "n_rows": self.n_rows,
            "n_symbols": self.n_symbols,
            "n_sessions": self.n_sessions,
            "findings": findings,
        }
        return json.dumps(payload, sort_keys=True)

    def summary(self) -> str:
        """Return a short one-line-per-check summary of finding counts."""
        counts: dict[str, dict[str, int]] = {}
        for finding in self.findings:
            entry = counts.setdefault(
                finding.check, {"total": 0, "error": 0, "warn": 0, "info": 0}
            )
            entry["total"] += 1
            entry[finding.severity.value] += 1
        return "\n".join(
            f"{name}: total={c['total']} error={c['error']} "
            f"warn={c['warn']} info={c['info']}"
            for name, c in counts.items()
        )


def _true_run_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start, end, length) arrays for contiguous True runs in a 1-D mask."""
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if mask.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, empty

    padded = np.empty(mask.size + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = np.asarray(mask, dtype=bool)

    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1).astype(np.intp, copy=False)
    ends = np.flatnonzero(transitions == -1).astype(np.intp, copy=False)
    lengths = ends - starts
    return starts, ends, lengths


def validate_panel(
    panel: Panel,
    *,
    calendar: TradingCalendar | None = None,
    checks: Sequence[str] | None = None,
) -> DataQualityReport:
    """Dispatch to panel-based validation checks through an internal registry."""
    registry: dict[str, Callable[[], list[Finding]]] = {
        "check_timestamps": lambda: check_timestamps(panel, calendar=calendar),
        "check_session_lengths": lambda: check_session_lengths(panel, calendar=calendar),
        "check_ohlc_consistency": lambda: check_ohlc_consistency(panel),
        "check_zero_or_negative_prices": lambda: check_zero_or_negative_prices(panel),
        "check_stale_bars": lambda: check_stale_bars(panel),
        "check_unexplained_gaps": lambda: check_unexplained_gaps(panel),
        "check_all_nan_columns": lambda: check_all_nan_columns(panel),
        "check_volume_sanity": lambda: check_volume_sanity(panel),
    }

    if checks is None:
        names: list[str] = list(registry)
    else:
        names = list(checks)

    unknown = sorted(set(names) - set(registry))
    if unknown:
        available = ", ".join(registry)
        unknown_list = ", ".join(unknown)
        raise ValueError(
            f"Unknown check name(s): {unknown_list}. Available checks: {available}"
        )

    findings: list[Finding] = []
    for name in names:
        findings.extend(registry[name]())

    return DataQualityReport(
        findings=tuple(findings),
        n_rows=panel.n_rows(),
        n_symbols=panel.n_symbols(),
        n_sessions=panel.n_days(),
    )


def check_timestamps(
    panel: Panel, *, calendar: TradingCalendar | None = None
) -> list[Finding]:
    """Check per-session timestamp monotonicity and regular-session time bounds."""
    findings: list[Finding] = []
    for day_idx, session_date in enumerate(panel.dates):
        start = int(panel.day_offsets[day_idx])
        end = int(panel.day_offsets[day_idx + 1])
        ts = panel.ts[start:end]

        if ts.size > 1:
            diff = np.diff(ts)
            bad = int(np.count_nonzero(diff <= 0))
            if bad:
                findings.append(
                    Finding(
                        check="check_timestamps",
                        severity=Severity.ERROR,
                        symbol=None,
                        session=session_date,
                        count=bad,
                        detail=f"{bad} non-increasing/duplicate timestamp(s) in session",
                    )
                )

        if calendar is not None and ts.size > 0:
            try:
                kind = calendar.classify(session_date)
            except KeyError:
                continue

            if kind in (SessionKind.REGULAR, SessionKind.MINOR):
                ist_times = to_ist(np.asarray([ts[0], ts[-1]], dtype=np.int64))
                first_time = ist_times[0].time()
                last_time = ist_times[-1].time()
                if (
                    first_time < settings.SESSION_START
                    or first_time > settings.SESSION_END
                    or last_time < settings.SESSION_START
                    or last_time > settings.SESSION_END
                ):
                    findings.append(
                        Finding(
                            check="check_timestamps",
                            severity=Severity.ERROR,
                            symbol=None,
                            session=session_date,
                            count=1,
                            detail=(
                                f"session bars outside [{settings.SESSION_START},"
                                f"{settings.SESSION_END}]: first={first_time} "
                                f"last={last_time}"
                            ),
                        )
                    )

    return findings


def check_session_lengths(
    panel: Panel, *, calendar: TradingCalendar | None = None
) -> list[Finding]:
    """Check session bar counts against the regular session length."""
    findings: list[Finding] = []
    for day_idx, session_date in enumerate(panel.dates):
        start = int(panel.day_offsets[day_idx])
        end = int(panel.day_offsets[day_idx + 1])
        n_bars = end - start

        if n_bars == settings.REGULAR_SESSION_BARS:
            continue

        severity = Severity.WARN
        kind: SessionKind | None = None
        if calendar is not None:
            try:
                kind = calendar.classify(session_date)
                if kind is not SessionKind.MINOR:
                    severity = Severity.INFO
            except KeyError:
                kind = None

        suffix = f", classified as {kind.value}" if kind is not None else ""
        findings.append(
            Finding(
                check="check_session_lengths",
                severity=severity,
                symbol=None,
                session=session_date,
                count=n_bars,
                detail=f"{n_bars} bars (expected 375){suffix}",
            )
        )

    return findings


def check_ohlc_consistency(panel: Panel) -> list[Finding]:
    """Flag bars where positive OHLC values violate low<=open,close<=high."""
    findings: list[Finding] = []
    open_arr = panel.field("open")
    high_arr = panel.field("high")
    low_arr = panel.field("low")
    close_arr = panel.field("close")

    for sym, sym_ix in panel.sym_ix.items():
        o = open_arr[:, sym_ix]
        h = high_arr[:, sym_ix]
        lo = low_arr[:, sym_ix]
        c = close_arr[:, sym_ix]

        present_positive = (
            ~np.isnan(o)
            & ~np.isnan(h)
            & ~np.isnan(lo)
            & ~np.isnan(c)
            & (o > 0)
            & (h > 0)
            & (lo > 0)
            & (c > 0)
        )
        violations = present_positive & ((lo > o) | (o > h) | (lo > c) | (c > h))
        count = int(np.count_nonzero(violations))
        if count:
            findings.append(
                Finding(
                    check="check_ohlc_consistency",
                    severity=Severity.ERROR,
                    symbol=sym,
                    session=None,
                    count=count,
                    detail=f"{count} bar(s) violate low<=open,close<=high",
                )
            )

    return findings


def check_zero_or_negative_prices(panel: Panel) -> list[Finding]:
    """Flag non-NaN OHLC values that are zero or negative."""
    findings: list[Finding] = []
    open_arr = panel.field("open")
    high_arr = panel.field("high")
    low_arr = panel.field("low")
    close_arr = panel.field("close")

    for sym, sym_ix in panel.sym_ix.items():
        arrays = (
            open_arr[:, sym_ix],
            high_arr[:, sym_ix],
            low_arr[:, sym_ix],
            close_arr[:, sym_ix],
        )
        count = sum(
            int(np.count_nonzero(~np.isnan(arr) & (arr <= 0))) for arr in arrays
        )
        if count:
            findings.append(
                Finding(
                    check="check_zero_or_negative_prices",
                    severity=Severity.ERROR,
                    symbol=sym,
                    session=None,
                    count=count,
                    detail=f"{count} non-positive OHLC value(s)",
                )
            )

    return findings


def check_stale_bars(panel: Panel, *, min_run: int = 5) -> list[Finding]:
    """Flag symbols whose longest stale run reaches or exceeds ``min_run``."""
    findings: list[Finding] = []
    open_arr = panel.field("open")
    high_arr = panel.field("high")
    low_arr = panel.field("low")
    close_arr = panel.field("close")
    volume_arr = panel.field("volume")

    for sym, sym_ix in panel.sym_ix.items():
        o = open_arr[:, sym_ix]
        h = high_arr[:, sym_ix]
        lo = low_arr[:, sym_ix]
        c = close_arr[:, sym_ix]
        v = volume_arr[:, sym_ix]

        present = ~np.isnan(o) & ~np.isnan(h) & ~np.isnan(lo) & ~np.isnan(c) & ~np.isnan(v)
        flat = (o == h) & (h == lo) & (lo == c)
        stale_sym = present & flat & (v == 0.0)

        best_len = 0
        best_date: date | None = None

        for day_idx, session_date in enumerate(panel.dates):
            start = int(panel.day_offsets[day_idx])
            end = int(panel.day_offsets[day_idx + 1])
            _, _, lengths = _true_run_endpoints(stale_sym[start:end])
            if lengths.size == 0:
                continue
            pos = int(np.argmax(lengths))
            run_len = int(lengths[pos])
            if run_len > best_len:
                best_len = run_len
                best_date = session_date

        if best_len >= min_run and best_date is not None:
            findings.append(
                Finding(
                    check="check_stale_bars",
                    severity=Severity.WARN,
                    symbol=sym,
                    session=best_date,
                    count=best_len,
                    detail=f"longest stale run: {best_len} bars starting {best_date}",
                )
            )

    return findings


def check_unexplained_gaps(panel: Panel, *, threshold_pct: float = 20.0) -> list[Finding]:
    """Flag large overnight price moves that lack corporate-action explanation."""
    findings: list[Finding] = []
    open_arr = panel.field("open")
    close_arr = panel.field("close")

    for sym, sym_ix in panel.sym_ix.items():
        opens = open_arr[:, sym_ix]
        closes = close_arr[:, sym_ix]

        for day_idx in range(1, panel.n_days()):
            prev_start = int(panel.day_offsets[day_idx - 1])
            prev_end = int(panel.day_offsets[day_idx])
            curr_start = int(panel.day_offsets[day_idx])
            curr_end = int(panel.day_offsets[day_idx + 1])

            prev_close_block = closes[prev_start:prev_end]
            prev_present = np.flatnonzero(~np.isnan(prev_close_block))
            if prev_present.size == 0:
                continue
            prev_close = float(prev_close_block[prev_present[-1]])
            if prev_close <= 0:
                continue

            curr_open_block = opens[curr_start:curr_end]
            curr_present = np.flatnonzero(~np.isnan(curr_open_block))
            if curr_present.size == 0:
                continue
            this_open = float(curr_open_block[curr_present[0]])

            pct = abs(this_open / prev_close - 1.0) * 100.0
            if pct > threshold_pct:
                findings.append(
                    Finding(
                        check="check_unexplained_gaps",
                        severity=Severity.WARN,
                        symbol=sym,
                        session=panel.dates[day_idx],
                        count=1,
                        detail=(
                            f"overnight gap {pct:.1f}% "
                            f"(prev_close={prev_close:.2f} -> open={this_open:.2f}), "
                            "no corporate-action data available to explain it"
                        ),
                    )
                )

    return findings


def check_all_nan_columns(panel: Panel) -> list[Finding]:
    """Flag all-NaN fields on symbols that have data in another field."""
    findings: list[Finding] = []
    fields = ("open", "high", "low", "close", "volume")
    arrays = {field: panel.field(field) for field in fields}

    for sym, sym_ix in panel.sym_ix.items():
        all_nan = {
            field: bool(np.all(np.isnan(arrays[field][:, sym_ix]))) for field in fields
        }
        if not any(not flag for flag in all_nan.values()):
            continue

        for field in fields:
            if all_nan[field]:
                findings.append(
                    Finding(
                        check="check_all_nan_columns",
                        severity=Severity.ERROR,
                        symbol=sym,
                        session=None,
                        count=panel.n_rows(),
                        detail=(
                            f"field '{field}' is all-NaN for a symbol with data "
                            "elsewhere (alignment bug)"
                        ),
                    )
                )

    return findings


_INDEX_SYMBOLS = frozenset({"NIFTY50", "NIFTY100", "NIFTYBANK", "INDIAVIX"})


def check_volume_sanity(panel: Panel) -> list[Finding]:
    """Flag negative volume and sessions with no traded volume."""
    findings: list[Finding] = []
    volume_arr = panel.field("volume")
    close_arr = panel.field("close")

    for sym, sym_ix in panel.sym_ix.items():
        if sym in _INDEX_SYMBOLS:
            continue

        volume = volume_arr[:, sym_ix]
        present_volume = ~np.isnan(volume)
        negative_count = int(np.count_nonzero(present_volume & (volume < 0)))
        if negative_count:
            findings.append(
                Finding(
                    check="check_volume_sanity",
                    severity=Severity.ERROR,
                    symbol=sym,
                    session=None,
                    count=negative_count,
                    detail=f"{negative_count} negative volume value(s)",
                )
            )

        closes = close_arr[:, sym_ix]
        zero_sessions: list[date] = []
        for day_idx, session_date in enumerate(panel.dates):
            start = int(panel.day_offsets[day_idx])
            end = int(panel.day_offsets[day_idx + 1])
            close_present = ~np.isnan(closes[start:end])
            if not np.any(close_present):
                continue
            volume_block = volume[start:end]
            if np.all(volume_block[close_present] == 0.0):
                zero_sessions.append(session_date)

        if zero_sessions:
            dates_text = ",".join(d.isoformat() for d in zero_sessions[:5])
            if len(zero_sessions) > 5:
                dates_text += "..."
            findings.append(
                Finding(
                    check="check_volume_sanity",
                    severity=Severity.WARN,
                    symbol=sym,
                    session=None,
                    count=len(zero_sessions),
                    detail=(
                        f"{len(zero_sessions)} session(s) with zero volume "
                        f"throughout: {dates_text}"
                    ),
                )
            )

    return findings


def check_spot_vs_futures_adjustment(
    symbols: Sequence[str] | None = None, *, ratio_jump_pct: float = 5.0
) -> list[Finding]:
    """Detect one-sided spot-vs-futures corporate-action adjustment mismatches."""
    futures_path = settings.EXTERNAL_ROOT / "fo_bhavcopy" / "stock_futures_daily.parquet"
    futures = pd.read_parquet(futures_path, columns=["sym", "settle", "d"])
    futures["date"] = pd.DatetimeIndex(futures["d"]).normalize().date

    if symbols is None:
        target_symbols: list[str] = [str(s) for s in pd.unique(futures["sym"])]
    else:
        target_symbols = [str(s) for s in symbols]

    findings: list[Finding] = []

    for sym in target_symbols:
        sub = futures[futures["sym"] == sym]
        if sub.empty:
            continue

        futures_settle = pd.Series(
            sub["settle"].to_numpy(dtype=np.float64),
            index=sub["date"].to_numpy(),
        )
        futures_settle = futures_settle[~futures_settle.index.duplicated(keep="last")]
        futures_settle = futures_settle.dropna()
        if futures_settle.empty:
            continue

        min_year = min(d.year for d in futures_settle.index)
        max_year = max(d.year for d in futures_settle.index)
        year_paths = [
            settings.BARS_1M / sym / f"{year}.parquet"
            for year in range(min_year, max_year + 1)
        ]
        existing_paths = [path for path in year_paths if path.exists()]
        if not existing_paths:
            continue

        daily_parts: list[pd.Series] = []
        for path in existing_paths:
            try:
                bars = pd.read_parquet(path, columns=["ts", "close"])
            except OSError:
                continue
            if bars.empty:
                continue

            ist_index = to_ist(bars["ts"].to_numpy(dtype=np.int64))
            daily = pd.Series(
                bars["close"].to_numpy(dtype=np.float64),
                index=ist_index,
            ).groupby(ist_index.date).last()
            daily_parts.append(daily)

        if not daily_parts:
            continue

        spot_daily = pd.concat(daily_parts)
        spot_daily = spot_daily[~spot_daily.index.duplicated(keep="last")].sort_index()
        spot_daily = spot_daily.dropna()
        if spot_daily.empty:
            continue

        common = spot_daily.index.intersection(futures_settle.index)
        if len(common) < 2:
            continue

        spot = spot_daily.loc[common].sort_index()
        settle = futures_settle.loc[common].sort_index()
        ratio = spot / settle
        pct_change = ratio.pct_change() * 100.0

        for pos in range(1, len(ratio)):
            change = float(pct_change.iloc[pos])
            if not np.isfinite(change) or abs(change) <= ratio_jump_pct:
                continue

            current_date = ratio.index[pos]
            prev_ratio = float(ratio.iloc[pos - 1])
            current_ratio = float(ratio.iloc[pos])
            findings.append(
                Finding(
                    check="check_spot_vs_futures_adjustment",
                    severity=Severity.WARN,
                    symbol=sym,
                    session=current_date,
                    count=1,
                    detail=(
                        f"spot/futures ratio jumped {change:.1f}% on {current_date} "
                        f"(prev ratio={prev_ratio:.4f}, new ratio={current_ratio:.4f}) - "
                        "likely a one-sided corporate-action adjustment"
                    ),
                )
            )

    return findings


def circuit_locked_mask(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    prev_close: np.ndarray,
    *,
    bands: tuple[float, ...] = (0.05, 0.10, 0.20),
    tol: float = 1e-4,
) -> np.ndarray:
    """Return True where a bar looks circuit-locked at any of ``bands``."""
    _ = close

    locked = np.zeros_like(high, dtype=bool)
    with np.errstate(invalid="ignore"):
        for band in bands:
            upper = prev_close * (1.0 + band)
            lower = prev_close * (1.0 - band)
            near_upper = np.abs(high - upper) <= (tol * prev_close)
            near_lower = np.abs(low - lower) <= (tol * prev_close)
            flat = high == low
            locked |= (near_upper | near_lower) & flat

    return locked


def stale_mask(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    min_run: int = 5,
) -> np.ndarray:
    """Return True for rows inside per-column stale runs of length >= ``min_run``."""
    present = (
        ~np.isnan(open_)
        & ~np.isnan(high)
        & ~np.isnan(low)
        & ~np.isnan(close)
        & ~np.isnan(volume)
    )
    flat = (open_ == high) & (high == low) & (low == close)
    stale_bars = present & flat & (volume == 0.0)

    result = np.zeros_like(stale_bars, dtype=bool)
    for col_idx in range(stale_bars.shape[1]):
        starts, ends, lengths = _true_run_endpoints(stale_bars[:, col_idx])
        valid = lengths >= min_run
        for start, end in zip(starts[valid], ends[valid]):
            result[start:end, col_idx] = True

    return result


def tradable_mask(
    panel: Panel,
    *,
    min_adv_inr: float = 5e7,
    bands: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> np.ndarray:
    """Return a bool mask of present bars that are not locked, stale, or illiquid."""
    close = panel.field("close")
    open_ = panel.field("open")
    high = panel.field("high")
    low = panel.field("low")
    volume = panel.field("volume")

    present = ~np.isnan(close)
    n_rows = panel.n_rows()
    n_sessions = panel.n_days()
    n_symbols = panel.n_symbols()
    day_offsets = panel.day_offsets

    prev_close = np.full((n_rows, n_symbols), np.nan, dtype=close.dtype)
    for session_idx in range(1, n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        prev_ix = int(day_offsets[session_idx]) - 1
        # The False arm is unreachable for any constructible Panel and is excluded from
        # coverage: the loop starts at session_idx 1, and Panel.__init__ runs
        # check_day_offsets, which enforces day_offsets[0] == 0 and strict increase. So
        # start < end always holds, and prev_ix = day_offsets[session_idx] - 1 >= 0.
        # Kept rather than removed because a negative prev_ix would make close[prev_ix, :]
        # wrap around to the LAST row of the panel — silently seeding every session's
        # prev_close with future data instead of raising. That is a wrong-answer failure,
        # not a crash, so the guard stays.
        if start < end and prev_ix >= 0:  # pragma: no branch - Panel invariant, see above
            prev_close[start:end, :] = close[prev_ix, :]

    locked = circuit_locked_mask(
        high=high,
        low=low,
        close=close,
        prev_close=prev_close,
        bands=bands,
    )
    stale = stale_mask(
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    day_value = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        day_slice = slice(start, end)
        day_value[session_idx, :] = np.nansum(
            close[day_slice] * volume[day_slice], axis=0
        )

    adv = np.full((n_sessions, n_symbols), np.nan, dtype=np.float64)
    for session_idx in range(1, n_sessions):
        lookback_start = max(0, session_idx - 20)
        adv[session_idx, :] = np.nanmean(day_value[lookback_start:session_idx, :], axis=0)

    liquid_day = adv >= min_adv_inr
    liquid_rows = np.zeros((n_rows, n_symbols), dtype=bool)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        liquid_rows[start:end, :] = liquid_day[session_idx, :]

    return present & ~locked & ~stale & liquid_rows
