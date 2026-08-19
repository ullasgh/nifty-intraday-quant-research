"""Pin two defects found by an adversarial critic in the just-landed `lens.py` repair
of `specs/lens_criteria_6_7_repair.md` (criteria 6/7) and `specs/lens_criterion_4_repair.md`
(criterion 4). Neither fix exists yet in `src/nifty_quant/research/lens.py`; every test
here must be RED against current `src/` for the right reason (an `AssertionError` on the
specific behaviour under test, never an `ImportError`/`AttributeError` from a missing
symbol).

Defect 1 -- criterion 7's `NOT_EVALUATED` branch (taken when fewer than two complete
calendar years exist) omits the `excluded partial years: ...` field that
`specs/lens_criteria_6_7_repair.md`'s 2026-08-19 ADJUDICATED ruling mandates for EVERY
criterion-7 reason string, not just the PASS/FAIL branch. `lens.py` (`:872-882`)
currently only prints `(complete_years=...)` there. Neither
`tests/test_lens_criteria_repair_a.py` nor `_b.py` checks for the field in this branch
(both suites' `NOT_EVALUATED` assertions check only the token and the absence of
PASS/FAIL, confirmed by reading both files directly).

Defect 2 -- criterion 4's time-of-day concentration check (`lens.py:783`,
`max_edge > median_edge * 2`) still compares against a bare literal `2` instead of the
named module constant `CONCENTRATION_RATIO_THRESHOLD` that the SAME max/median-ratio
statistic already uses at the liquidity-decile site three lines above (`:745`). That is
an unlabelled, hand-chosen cutoff -- a CLAUDE.md rule 8 violation -- and an inconsistency
within one criterion. Because the constant's current numeric value (2.0) happens to
equal the literal, a fixture built only around "does concentration fire" cannot
distinguish the literal from the constant: both give the same answer today. The tests
below therefore `monkeypatch` `CONCENTRATION_RATIO_THRESHOLD` to a value that DIFFERS
from 2.0 and build a fixture whose ratio lies strictly between the old literal (2.0)
and the patched value -- so the two implementations disagree, and only a genuine
`median_edge * CONCENTRATION_RATIO_THRESHOLD` read (rather than `* 2`) can pass. This
also satisfies the task's "robust to recalibration" requirement: nothing here hardcodes
2.0 as an expected outcome.

Fixtures and helpers reused per CLAUDE.md's "reuse existing fixtures" convention:
- Criterion-7 tests reuse `_build_dated_panel`, `_monthly_session_dates`, `_call_verdict`,
  `_criterion_token` from `tests.test_lens_criteria_repair_a` (the month-dense date
  builders named directly in the task).
- Criterion-4 tests reuse `_signal_means`, `_close_prices_from_log_returns`, `_SYMBOLS`,
  `_N_SYMBOLS` from `tests.test_lens` (same cross-test-module-import precedent that
  `tests/test_lens_criteria_repair_a.py`'s own docstring cites:
  `tests.test_engine` / `tests.test_pending_orders`).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.research import lens as lens_module
from nifty_quant.research.lens import Lens
from tests.test_lens import _N_SYMBOLS, _SYMBOLS, _close_prices_from_log_returns, _signal_means
from tests.test_lens_criteria_repair_a import (
    _build_dated_panel,
    _call_verdict,
    _criterion_token,
    _monthly_session_dates,
)

_IST = ZoneInfo("Asia/Kolkata")


# ===========================================================================
# Defect 1 -- criterion 7's NOT_EVALUATED branch must also carry the mandated
# "excluded partial years" field.
# ===========================================================================


def _excluded_partial_years_field(reason: str) -> str:
    """Extract the text following the mandated `excluded partial years:` marker,
    up to the next `)` (or end of string). Asserts the marker is present at all --
    this is the actual defect-1 pin: on unrepaired `src/`, the NOT_EVALUATED branch
    never emits this marker, so this assertion is what turns AssertionError (not
    ImportError/AttributeError) against current code.
    """
    marker = "excluded partial years:"
    lower = reason.lower()
    idx = lower.find(marker)
    assert idx != -1, (
        f"NOT_EVALUATED reason is missing the mandated 'excluded partial years' "
        f"field entirely: {reason!r}"
    )
    tail = reason[idx + len(marker) :]
    end = tail.find(")")
    if end != -1:
        tail = tail[:end]
    return tail.strip()


def test_criterion7_not_evaluated_zero_complete_years_names_all_partial_years() -> None:
    """Defect 1, sub-case A (zero complete years): 2023 and 2024 each have sessions
    in only 3 of their 12 calendar months, so neither is complete -- criterion 7 must
    report NOT_EVALUATED, and the mandated field must still name BOTH years as
    excluded (never silently read 'none' when years genuinely were excluded).
    """
    dates = _monthly_session_dates(2023, months=range(1, 4)) + _monthly_session_dates(
        2024, months=range(1, 4)
    )
    panel = _build_dated_panel(dates, 150, {2023, 2024}, effect=0.01, sigma=0.0005, seed=101)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert reason.startswith("7. Recent-years cost gate criterion:")
    assert _criterion_token(reason) == "NOT_EVALUATED"
    excluded = _excluded_partial_years_field(reason)
    assert "2023" in excluded
    assert "2024" in excluded
    assert "none" not in excluded.lower()


def test_criterion7_not_evaluated_one_complete_year_names_the_partial_year() -> None:
    """Defect 1, sub-case B (exactly one complete year + one named partial year):
    2024 has a session in every one of its 12 months (complete); 2025 has sessions
    only in Jan-Jun (partial). One complete year is still fewer than the two
    criterion 7 requires, so the result stays NOT_EVALUATED -- but the mandated
    field must name 2025 as excluded, and must NOT misname the genuinely-complete
    2024 as partial.
    """
    dates = _monthly_session_dates(2024) + _monthly_session_dates(2025, months=range(1, 7))
    panel = _build_dated_panel(dates, 150, {2024, 2025}, effect=0.01, sigma=0.0005, seed=102)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) == "NOT_EVALUATED"
    excluded = _excluded_partial_years_field(reason)
    assert "2025" in excluded
    assert "2024" not in excluded


# ===========================================================================
# Defect 2 -- criterion 4's time-of-day concentration check must read the named
# module constant, not a bare literal `2`.
# ===========================================================================


def _tod_concentration_constant() -> float:
    threshold = getattr(lens_module, "CONCENTRATION_RATIO_THRESHOLD", None)
    if threshold is None:
        threshold = getattr(lens_module, "CONCENTRATION_THRESHOLD", None)
    assert threshold is not None, (
        "expected a named concentration-ratio module constant "
        "(CONCENTRATION_RATIO_THRESHOLD or CONCENTRATION_THRESHOLD) on "
        "nifty_quant.research.lens"
    )
    return float(threshold)


def _build_time_of_day_panel(
    open_effect: float,
    mid_effect: float,
    close_effect: float,
    *,
    n_sessions: int,
    sigma: float,
    seed: int,
) -> Panel:
    """Build a panel with a controllable max/median time-of-day concentration
    ratio, isolated from every other kill criterion as far as possible.

    A full 375-bar session, starting 9:15 IST, spans exactly the three
    production time-of-day windows (`lens.py` 555-630 "open", 630-840 "mid",
    840-930 "close" minute-of-day) with no partial-window edge case. Within
    each window every bar gets a PERSISTENT per-symbol mean return via
    `_signal_means(effect)` (same convention as `tests/test_lens.py`'s
    `_build_panel`), so the resulting `by_time_of_day[window].spread_bps` (top
    quintile mean minus bottom quintile mean, in bps) is approximately
    `2 * effect * 1e4` for that window -- letting `open_effect` /
    `mid_effect` / `close_effect` set each window's edge independently and
    hence control the max/median ratio the concentration check reads.

    `_signal_means` always orders symbols S00 (most negative) .. S04 (most
    positive) regardless of the effect's magnitude, so cross-sectional-rank
    bucketing is consistent across all three windows even though their
    effect sizes differ -- and since every window's spread_bps is strictly
    POSITIVE by this same construction, the unrelated "mixed sign" branch of
    the concentration check never fires here.

    Volume is held identical across symbols (only 5 symbols -- exactly
    `min_names` for `cross_sectional_rank` -- spread across 10 liquidity
    deciles) so the liquidity-decile spreads collapse to 0.0 (verified
    empirically) and cannot confound the time-of-day-only signal this
    fixture targets: with `median_liquidity_edge == 0`, criterion 4's
    liquidity-concentration branch cannot fire (`median_liquidity_edge > 0`
    guards it), so any "4." FAIL/PASS decision below is driven purely by the
    time-of-day check.
    """
    bars_per_session = 375
    means_open = _signal_means(open_effect)
    means_mid = _signal_means(mid_effect)
    means_close = _signal_means(close_effect)
    block = np.vstack(
        [
            np.tile(means_open, (75, 1)),
            np.tile(means_mid, (210, 1)),
            np.tile(means_close, (90, 1)),
        ]
    )
    assert block.shape == (bars_per_session, _N_SYMBOLS)

    dates = list(pd.bdate_range("2024-01-02", periods=n_sessions).date)
    rng = np.random.default_rng(seed)
    returns_chunks: list[np.ndarray] = []
    for _ in dates:
        noise = rng.normal(0.0, sigma, size=(bars_per_session, _N_SYMBOLS))
        returns_chunks.append(block + noise)
    returns = np.concatenate(returns_chunks, axis=0)
    close = _close_prices_from_log_returns(returns)
    n_rows = returns.shape[0]
    volume = np.tile(np.full(_N_SYMBOLS, 1e5, dtype=np.float64), (n_rows, 1))

    ts_chunks: list[np.ndarray] = []
    day_offsets_list = [0]
    row = 0
    for date in dates:
        day_start = pd.Timestamp(date.year, date.month, date.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bars_per_session, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
        row += bars_per_session
        day_offsets_list.append(row)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    dates_arr = np.array(dates, dtype=object)
    day_offsets = np.array(day_offsets_list, dtype=np.int32)

    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _criterion4_line(panel: Panel, seed: int) -> str:
    lens = Lens(panel, seed=seed)
    feature = lens.feature("return_1")
    verdict = lens.verdict(
        "H_TOD",
        feature,
        horizon=1,
        latency_profile=None,
        effective_n_trials=1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=10,
    )
    line = verdict.reasons[3]
    assert line.startswith("4."), f"expected criterion-4 line at index 3, got {line!r}"
    return line


def test_criterion4_time_of_day_check_honours_patched_constant_suppressing_a_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ratio (~3.0, verified empirically) that fires under the OLD hardcoded `2`
    but must NOT fire once the check reads a PATCHED `CONCENTRATION_RATIO_THRESHOLD`
    of 5.0. This is decisive specifically because 3.0 lies strictly between the two:
    an implementation still comparing against the literal `2` reports FAIL here
    (3.0 > 2), while one that reads the (patched) module constant reports PASS
    (3.0 < 5.0). Deliberately does not merely reproduce the module constant's
    CURRENT numeric value (2.0), which would give the literal and the constant an
    identical answer and prove nothing.
    """
    monkeypatch.setattr(lens_module, "CONCENTRATION_RATIO_THRESHOLD", 5.0, raising=False)
    if hasattr(lens_module, "CONCENTRATION_THRESHOLD"):
        monkeypatch.setattr(lens_module, "CONCENTRATION_THRESHOLD", 5.0, raising=False)
    threshold = _tod_concentration_constant()
    assert threshold == 5.0

    panel = _build_time_of_day_panel(
        open_effect=0.006,
        mid_effect=0.002,
        close_effect=0.002,
        n_sessions=20,
        sigma=0.0003,
        seed=10,
    )
    line = _criterion4_line(panel, seed=10)

    assert "PASS" in line, (
        f"time-of-day ratio (~3.0) is below the patched threshold (5.0); a fix that "
        f"reads CONCENTRATION_RATIO_THRESHOLD must report PASS here, not FAIL from "
        f"a hardcoded 2: {line!r}"
    )
    assert "concentrated in single time-of-day bucket" not in line


def test_criterion4_time_of_day_check_honours_patched_constant_forcing_a_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: a ratio (~1.5, verified empirically) that does NOT fire
    under the OLD hardcoded `2` but MUST fire once the check reads a PATCHED
    `CONCENTRATION_RATIO_THRESHOLD` of 1.2. 1.5 lies strictly between the two, so
    an implementation still comparing against the literal `2` reports PASS here
    (1.5 < 2), while one that reads the (patched) module constant reports FAIL
    (1.5 > 1.2).
    """
    monkeypatch.setattr(lens_module, "CONCENTRATION_RATIO_THRESHOLD", 1.2, raising=False)
    if hasattr(lens_module, "CONCENTRATION_THRESHOLD"):
        monkeypatch.setattr(lens_module, "CONCENTRATION_THRESHOLD", 1.2, raising=False)
    threshold = _tod_concentration_constant()
    assert threshold == 1.2

    panel = _build_time_of_day_panel(
        open_effect=0.003,
        mid_effect=0.002,
        close_effect=0.002,
        n_sessions=20,
        sigma=0.0003,
        seed=11,
    )
    line = _criterion4_line(panel, seed=11)

    assert "FAIL" in line, (
        f"time-of-day ratio (~1.5) is above the patched threshold (1.2); a fix that "
        f"reads CONCENTRATION_RATIO_THRESHOLD must report FAIL here, not PASS from "
        f"a hardcoded 2: {line!r}"
    )
    assert "concentrated in single time-of-day bucket" in line
