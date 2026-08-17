"""H1: Market Intraday Momentum (Gao, Han, Li & Zhou 2018), replicated on NIFTY50.

The first half-hour return of the day (``r_first``) is tested for predictive power over
the last half-hour return (``r_last``), controlling for the intervening day's drift
(``r_mid``). One round trip per session: enter at 14:50, exit at the 15:20 square-off.

Checkpoints are resolved by IST time label via ``Panel.rows_at_time`` / ``Panel.dates`` --
never by row arithmetic -- because sessions are irregular (Muhurat sessions run ~60 bars,
disaster-recovery sessions ~105). The 09:15 bar is never read: every observed 2024 OHLC
violation (``close > high``) sits there, an artefact of the pre-open call auction leaking
into the first minute bar. Bars are left-labelled and NaN means "no bar" -- a session
missing any required checkpoint is dropped and counted, never interpolated or
forward-filled. All return arithmetic is float64, even though the panel stores prices as
float32 at rest.

Degenerate regressions (a constant regressor, or fewer than three usable sessions) return
NaN rather than raising: this harness is meant to be pointed at ~149 symbols x 8 years,
where some slice will inevitably be degenerate, and a single uncaught
``numpy.linalg.LinAlgError`` would abort every other symbol's result along with it. A NaN
edge cannot clear the cost hurdle, so degenerate slices fail closed rather than being
silently reported as a (false) flat 0.0 edge.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts

_NAN = float("nan")


@dataclass(frozen=True)
class SessionObservations:
    """One (r_first, r_mid, r_last) triple per usable session, ascending by date."""

    dates: np.ndarray  # object array of datetime.date, ascending, unique
    r_first: np.ndarray  # float64
    r_mid: np.ndarray  # float64
    r_last: np.ndarray  # float64
    n_sessions_dropped: int
    drop_reasons: Mapping[str, int]

    def explain(self) -> str:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(self.drop_reasons.items()))
        return (
            f"SessionObservations: {len(self.dates)} sessions kept, "
            f"{self.n_sessions_dropped} sessions dropped (drop_reasons: {reasons})."
        )


def _checkpoint_values(
    panel: Panel, symbol: str, hhmm: str, field_name: str
) -> dict[datetime.date, float]:
    """date -> value of `field_name` at the bar labelled `hhmm`, resolved purely by IST
    time label (never row arithmetic), restricted to sessions where that bar exists AND
    the field is not NaN there. A session with no bar at `hhmm` at all, and a session
    whose bar at `hhmm` has a NaN `field_name`, are treated identically as "missing" --
    never forward-filled."""
    sym_idx = panel.sym_ix[symbol]
    rows = panel.rows_at_time(hhmm)
    day_idx = np.searchsorted(panel.day_offsets, rows, side="right") - 1
    values = panel.field(field_name)[rows, sym_idx].astype(np.float64)
    dates = panel.dates[day_idx]
    return {d: float(v) for d, v in zip(dates, values) if not np.isnan(v)}


def extract_session_observations(panel: Panel, symbol: str) -> SessionObservations:
    """Extract one checkpoint-return observation per session.

    r_first = log(close[09:45] / open[09:16])   -- first half hour, starting at 09:16
                                                     (09:15 is never read; see module docstring)
    r_mid   = log(open[14:50] / close[09:45])    -- the intervening day's drift
    r_last  = log(close[15:20] / open[14:50])    -- last half hour, ending at square-off

    so that r_first + r_mid + r_last telescopes exactly to
    log(close[15:20] / open[09:16]).

    A session missing any one of the four checkpoints (no bar at that time label, or a
    NaN field there) is dropped entirely and counted in `drop_reasons` under the FIRST
    missing checkpoint in the order 09:16, 09:45, 14:50, 15:20 -- so
    `sum(drop_reasons.values()) == n_sessions_dropped` always holds.
    """
    open_0916 = _checkpoint_values(panel, symbol, "09:16", "open")
    close_0945 = _checkpoint_values(panel, symbol, "09:45", "close")
    open_1450 = _checkpoint_values(panel, symbol, "14:50", "open")
    close_1520 = _checkpoint_values(panel, symbol, "15:20", "close")

    kept_dates: list[datetime.date] = []
    o1_list: list[float] = []
    c1_list: list[float] = []
    o2_list: list[float] = []
    c2_list: list[float] = []
    drop_reasons: dict[str, int] = {}

    for d in panel.dates:
        if d not in open_0916:
            drop_reasons["missing_09:16"] = drop_reasons.get("missing_09:16", 0) + 1
            continue
        if d not in close_0945:
            drop_reasons["missing_09:45"] = drop_reasons.get("missing_09:45", 0) + 1
            continue
        if d not in open_1450:
            drop_reasons["missing_14:50"] = drop_reasons.get("missing_14:50", 0) + 1
            continue
        if d not in close_1520:
            drop_reasons["missing_15:20"] = drop_reasons.get("missing_15:20", 0) + 1
            continue
        kept_dates.append(d)
        o1_list.append(open_0916[d])
        c1_list.append(close_0945[d])
        o2_list.append(open_1450[d])
        c2_list.append(close_1520[d])

    o1 = np.array(o1_list, dtype=np.float64)
    c1 = np.array(c1_list, dtype=np.float64)
    o2 = np.array(o2_list, dtype=np.float64)
    c2 = np.array(c2_list, dtype=np.float64)

    return SessionObservations(
        dates=np.array(kept_dates, dtype=object),
        r_first=np.log(c1 / o1),
        r_mid=np.log(o2 / c1),
        r_last=np.log(c2 / o2),
        n_sessions_dropped=len(panel.dates) - len(kept_dates),
        drop_reasons=MappingProxyType(drop_reasons),
    )


def _univariate_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS-with-intercept slope/t-stat/R^2 of y on x.

    Returns (nan, nan, nan) if the regression is not identified -- fewer than three
    usable sessions, or x has zero variance -- rather than raising. Standard errors use
    numpy float64 arithmetic throughout (never plain Python division), so an exactly
    zero residual variance or zero design variance produces IEEE-754 nan/inf, never a
    ZeroDivisionError.
    """
    n = len(x)
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if n < 3 or float(x64.max() - x64.min()) == 0.0:
        return _NAN, _NAN, _NAN

    xbar = x64.mean()
    ybar = y64.mean()
    sxx = np.sum((x64 - xbar) ** 2)
    sxy = np.sum((x64 - xbar) * (y64 - ybar))
    beta = sxy / sxx
    alpha = ybar - beta * xbar
    resid = y64 - (alpha + beta * x64)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y64 - ybar) ** 2)
    sigma2 = ss_res / np.float64(n - 2)
    se_beta = np.sqrt(sigma2 / sxx)
    t_stat = beta / se_beta
    r_squared = 1.0 - ss_res / ss_tot
    return float(beta), float(t_stat), float(r_squared)


def _bivariate_beta1(
    r_first: np.ndarray, r_mid: np.ndarray, r_last: np.ndarray
) -> tuple[float, float]:
    """Slope on r_first (and its t-stat) from r_last ~ a + b1*r_first + b2*r_mid.

    Returns (nan, nan) if the design matrix [1, r_first, r_mid] is not full column rank
    -- which covers both "fewer than 3 usable sessions" (rank is bounded by n) and "r_mid
    (or r_first) is constant" (its column is a multiple of the intercept column) in a
    single check. `lstsq` and `pinv` are used throughout, never a direct `solve`/`inv`,
    so a singular or ill-conditioned design never raises `numpy.linalg.LinAlgError`.
    """
    n = len(r_first)
    x1 = np.asarray(r_first, dtype=np.float64)
    x2 = np.asarray(r_mid, dtype=np.float64)
    y = np.asarray(r_last, dtype=np.float64)
    design = np.column_stack([np.ones(n, dtype=np.float64), x1, x2])

    if np.linalg.matrix_rank(design) < 3:
        return _NAN, _NAN

    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    beta1 = coeffs[1]
    resid = y - design @ coeffs
    ss_res = np.sum(resid**2)
    sigma2 = ss_res / np.float64(n - 3)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se_beta1 = np.sqrt(cov[1, 1])
    t_stat1 = beta1 / se_beta1
    return float(beta1), float(t_stat1)


def _mean_edge_bps(r_first: np.ndarray, r_last: np.ndarray) -> float:
    """1e4 * mean(sign(r_first) * r_last) -- the sign(r_first) trading rule's edge."""
    return float(1e4 * np.mean(np.sign(r_first) * r_last))


def _hit_rate(r_first: np.ndarray, r_last: np.ndarray) -> float:
    """Fraction of TRADED sessions (r_first != 0) where sign(r_last) == sign(r_first).

    A zero r_first is flat, not a trade, and is excluded from both the numerator and the
    denominator. If no session trades, the answer is nan, never 0.0 -- 0.0 would
    misleadingly read as "traded and always wrong" rather than "nothing to measure".
    """
    traded = r_first != 0.0
    if not np.any(traded):
        return _NAN
    agree = np.sign(r_last[traded]) == np.sign(r_first[traded])
    return float(np.mean(agree))


def _by_year_edges(dates: np.ndarray, r_first: np.ndarray, r_last: np.ndarray) -> dict[int, float]:
    """Per-calendar-year mean_edge_bps. A year with fewer than 3 sessions is nan (too few
    to estimate), matching the degenerate-regression convention elsewhere in this module."""
    years = np.array([d.year for d in dates], dtype=np.int64)
    result: dict[int, float] = {}
    for year in sorted(set(years.tolist())):
        mask = years == year
        if int(mask.sum()) < 3:
            result[year] = _NAN
        else:
            result[year] = _mean_edge_bps(r_first[mask], r_last[mask])
    return result


def _years_sign_consistent(by_year: Mapping[int, float]) -> int:
    """Count years holding the MODAL NON-ZERO sign of the per-year edge.

    Years whose edge is exactly zero, or nan (too few sessions to estimate), are
    discarded from the vote entirely -- they count toward neither total. On an exact tie
    between the number of positive and negative years, the dominant sign is "mixed" and
    this returns 0 (no individual year's sign is ever "mixed", so nothing holds it).
    Counting years rather than weighting by magnitude is deliberate: it is what makes the
    criterion robust to a single dominant year skewing a magnitude-weighted vote.
    """
    pos = 0
    neg = 0
    for edge in by_year.values():
        if math.isnan(edge) or edge == 0.0:
            continue
        if edge > 0:
            pos += 1
        else:
            neg += 1
    if pos > neg:
        return pos
    if neg > pos:
        return neg
    return 0


@dataclass(frozen=True)
class H1Result:
    """Result of one H1 run for a single symbol over some (sub-)range of `panel`."""

    symbol: str
    n_sessions: int
    beta: float
    t_stat: float
    r_squared: float
    mean_edge_bps: float
    hit_rate: float
    cost_hurdle_bps: float
    survives_costs: bool
    by_year: Mapping[int, float]
    years_sign_consistent: int
    beta_controlling_for_mid: float
    t_stat_controlling_for_mid: float

    def _survived(self) -> bool:
        """All four kill criteria: (1) mean_edge_bps clears 2x the cost hurdle, (2) the
        yearly edge sign is consistent in >= 6 of 8 years, (3) the univariate t-stat
        clears 1.96 in magnitude, and (4) beta_controlling_for_mid retains >= 50% of the
        univariate beta's magnitude and stays significant once r_mid is controlled for.
        Any nan statistic makes its comparison False, so a non-measurable slice fails
        closed rather than accidentally reading as a survival."""
        return (
            self.survives_costs
            and self.years_sign_consistent >= 6
            and abs(self.t_stat) > 1.96
            and abs(self.beta_controlling_for_mid) >= 0.5 * abs(self.beta)
            and abs(self.t_stat_controlling_for_mid) > 1.96
        )

    def _verdict(self) -> str:
        # A plain branch, deliberately. This was originally written as a dict lookup
        # (`{True: ..., False: ...}[...]`) because no fixture produced an above-hurdle edge, so an
        # `if/else` would have been structurally unreachable and shown as a coverage gap. That was
        # a workaround for a MISSING TEST, not a real need: the module's whole purpose is to say
        # "this survives costs", and that output path had none. Two fixtures now cover both
        # outcomes (a 52.36 bps SURVIVED case and a 14.49 bps KILLED mirror that fails only the
        # cost gate), so the branch is genuinely exercised and can be written plainly.
        if self._survived():
            return "SURVIVED"
        return "KILLED"

    def explain(self) -> str:
        lines = [
            f"H1 market intraday momentum -- symbol={self.symbol}, "
            f"n_sessions={self.n_sessions}",
            f"beta={self.beta:.6f} t_stat={self.t_stat:.4f} r_squared={self.r_squared:.4f}",
            f"beta_controlling_for_mid={self.beta_controlling_for_mid:.6f} "
            f"t_stat_controlling_for_mid={self.t_stat_controlling_for_mid:.4f} "
            "(confound control on r_mid)",
            f"mean_edge_bps={self.mean_edge_bps:.4f} hit_rate={self.hit_rate:.4f} "
            f"cost_hurdle_bps={self.cost_hurdle_bps:.5f} (2x-hurdle cost gate)",
            f"years_sign_consistent={self.years_sign_consistent} (needs >= 6 of 8 years)",
            "Kill criteria: (1) mean_edge_bps > 2x cost_hurdle_bps; "
            "(2) yearly edge sign consistent in >= 6 of 8 years; "
            "(3) abs(t_stat) > 1.96; "
            "(4) beta_controlling_for_mid retains >= 50% of the univariate beta "
            "magnitude AND abs(t_stat_controlling_for_mid) > 1.96.",
            f"Overall verdict: {self._verdict()}",
            "One observation per session, so the session observations do not overlap; "
            "a naive OLS t-stat is therefore valid here and no block-bootstrap overlap "
            "correction is applied.",
        ]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"# H1 Market Intraday Momentum -- {self.symbol}",
            "",
            "| metric | value |",
            "|---|---|",
            f"| n_sessions | {self.n_sessions} |",
            f"| beta | {self.beta:.6f} |",
            f"| t_stat | {self.t_stat:.4f} |",
            f"| r_squared | {self.r_squared:.4f} |",
            f"| mean_edge_bps | {self.mean_edge_bps:.4f} |",
            f"| hit_rate | {self.hit_rate:.4f} |",
            f"| cost_hurdle_bps | {self.cost_hurdle_bps:.5f} |",
            f"| beta_controlling_for_mid | {self.beta_controlling_for_mid:.6f} |",
            f"| t_stat_controlling_for_mid | {self.t_stat_controlling_for_mid:.4f} |",
            f"| years_sign_consistent | {self.years_sign_consistent} |",
            f"| verdict | {self._verdict()} |",
        ]
        return "\n".join(lines)


def run_h1(
    panel: Panel,
    symbol: str = "NIFTY50",
    *,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    cost_hurdle_bps: float | None = None,
) -> H1Result:
    """Run the H1 market-intraday-momentum hypothesis test on `panel` for `symbol`.

    One observation per trading session, so the session observations do not overlap with
    one another. That is what makes a naive OLS t-stat appropriate here with no
    block-bootstrap or Newey-West style overlap correction: each (r_first, r_mid, r_last)
    triple is drawn from a distinct calendar day, unlike most other signals in this repo
    that operate on overlapping intraday windows.

    `start`/`end` are inclusive and mirror `Panel.sub`, so the caller can restrict the
    panel to its pre-holdout window (`panel.sub(start=..., end=...)` semantics) without
    slicing the panel itself first. This function never reads
    `research.splits.HoldoutLock` and never reaches past whatever `start`/`end` the
    caller supplies.
    """
    sliced = panel.sub(start=start, end=end)
    obs = extract_session_observations(sliced, symbol)

    resolved_hurdle = (
        float(cost_hurdle_bps)
        if cost_hurdle_bps is not None
        else NSEIntradayEquityCosts().round_trip_bps(1e5)
    )

    beta, t_stat, r_squared = _univariate_ols(obs.r_first, obs.r_last)
    beta_mid, t_stat_mid = _bivariate_beta1(obs.r_first, obs.r_mid, obs.r_last)

    mean_edge_bps = _mean_edge_bps(obs.r_first, obs.r_last)
    hit_rate = _hit_rate(obs.r_first, obs.r_last)
    survives = bool(mean_edge_bps > 2.0 * resolved_hurdle)

    by_year = _by_year_edges(obs.dates, obs.r_first, obs.r_last)
    years_consistent = _years_sign_consistent(by_year)

    return H1Result(
        symbol=symbol,
        n_sessions=len(obs.dates),
        beta=beta,
        t_stat=t_stat,
        r_squared=r_squared,
        mean_edge_bps=mean_edge_bps,
        hit_rate=hit_rate,
        cost_hurdle_bps=resolved_hurdle,
        survives_costs=survives,
        by_year=MappingProxyType(by_year),
        years_sign_consistent=years_consistent,
        beta_controlling_for_mid=beta_mid,
        t_stat_controlling_for_mid=t_stat_mid,
    )
