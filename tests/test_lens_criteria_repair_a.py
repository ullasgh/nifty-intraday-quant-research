"""Independent test suite A for `specs/lens_criteria_6_7_repair.md`.

Written from the spec alone, before any implementation change to
`nifty_quant.research.lens.Lens.verdict()` exists. Do not read or reference
`tests/test_lens_criteria_repair_b.py` -- that is the second, independently
authored suite; per the spec, neither author may see the other's file.

Two defects are repaired here:

- Criterion 6 (deflated Sharpe) must be computed on `strategy_returns`, a new
  keyword-only parameter of `verdict()` carrying the strategy's own
  per-period P&L series -- never on `fwd.values` (the unconditional
  forward return of every symbol at every row, which is what the code
  currently misuses).
- Criterion 7 (recent-years cost gate) must treat a calendar year as
  "complete" iff `panel_first_date <= date(year, 1, 1)` and
  `panel_last_date >= date(year, 12, 31)`, using the panel's own first and
  last dates -- not the old `session_counts.get(year, 0) >= 20` threshold,
  which is itself a hand-chosen constant CLAUDE.md rule 8 forbids.

Fixtures and low-level helpers (`_SYMBOLS`, `_signal_means`,
`_close_prices_from_log_returns`, `_build_panel`, `_all_pass_panel`,
`_PASS_LATENCY`, etc.) are reused from `tests/test_lens.py` per the task's
"reuse existing fixtures" instruction; `tests.test_engine` /
`tests.test_pending_orders` already establish cross-test-module imports as
an accepted pattern in this repo.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.research.lens import HypothesisVerdict, Lens
from tests.test_lens import (
    _DEFAULT_VOLUME,
    _N_SYMBOLS,
    _PASS_LATENCY,
    _SYMBOLS,
    _all_pass_panel,
    _build_panel,
    _close_prices_from_log_returns,
    _signal_means,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Local helpers -- criterion 7 is about window EDGES (panel first/last date),
# which `tests.test_lens._build_panel` cannot exercise: its sessions always
# start on Jan 2/3 of each requested year and never reach Dec 31. This
# builder takes an explicit list of session dates (and, per CLAUDE.md rule 5,
# an optionally variable bar count per session -- never a fixed stride) so
# the panel's own first/last dates can be placed exactly on a test's chosen
# boundary.
# ---------------------------------------------------------------------------


def _build_dated_panel(
    dates: list[dt.date],
    bars_per_session: int | list[int],
    signal_years: set[int],
    *,
    effect: float = 0.01,
    sigma: float = 0.0005,
    flat_sigma: float = 0.0,
    seed: int = 0,
) -> Panel:
    """Build a Panel from explicit session dates, e.g. to place first/last
    panel dates exactly on a calendar-year boundary for criterion-7 tests.

    `bars_per_session` may be a single int (uniform stride) or a per-day list
    -- callers use the list form to include an irregular (e.g. Muhurat-like
    short) session, since CLAUDE.md rule 5 forbids assuming a fixed stride.
    """
    if isinstance(bars_per_session, int):
        bars_list = [bars_per_session] * len(dates)
    else:
        bars_list = list(bars_per_session)
        assert len(bars_list) == len(dates)

    rng = np.random.default_rng(seed)
    returns_chunks: list[np.ndarray] = []
    for date, n_bars in zip(dates, bars_list):
        if date.year in signal_years:
            means = _signal_means(effect)
            block_means = np.tile(means, (n_bars, 1))
            block_sigma = sigma
        else:
            block_means = np.zeros((n_bars, _N_SYMBOLS), dtype=np.float64)
            block_sigma = flat_sigma
        noise = rng.normal(0.0, block_sigma, size=(n_bars, _N_SYMBOLS))
        returns_chunks.append(block_means + noise)
    returns = np.concatenate(returns_chunks, axis=0)
    close = _close_prices_from_log_returns(returns)
    n_rows = returns.shape[0]
    volume = np.tile(np.array(_DEFAULT_VOLUME, dtype=np.float64), (n_rows, 1))

    ts_chunks: list[np.ndarray] = []
    day_offsets_list = [0]
    row = 0
    for date, n_bars in zip(dates, bars_list):
        day_start = pd.Timestamp(date.year, date.month, date.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n_bars, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
        row += n_bars
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


def _consecutive_weekdays(
    year: int, n_sessions: int, start_month: int, start_day: int
) -> list[dt.date]:
    """Return `n_sessions` consecutive weekday dates starting on/after the given day.

    Used only for the criterion-7 "matches spec scenario" regression guard, where a
    realistic (>= 20) session count per year is needed to exercise the OLD
    `session_counts >= 20` threshold the spec is repairing.
    """
    out: list[dt.date] = []
    d = dt.date(year, start_month, start_day)
    while len(out) < n_sessions:
        if d.weekday() < 5:
            out.append(d)
        d = d + dt.timedelta(days=1)
    return out


def _call_verdict(
    panel: Panel,
    *,
    hypothesis_id: str = "H_repair_a",
    latency_profile: dict[int, float] | None = _PASS_LATENCY,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = None,
    seed: int = 0,
    horizon: int = 1,
) -> HypothesisVerdict:
    """Call `Lens.verdict()` with the new `strategy_returns` keyword.

    This is the same shape as `tests.test_lens._call_verdict` extended with
    `strategy_returns`, per the spec's new keyword-only parameter.
    """
    lens = Lens(panel, seed=seed)
    return lens.verdict(
        hypothesis_id,
        "return_1",
        horizon,
        latency_profile=latency_profile,
        effective_n_trials=effective_n_trials,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )


def _criterion_token(reason: str) -> str:
    """Extract the PASS/FAIL/NOT_EVALUATED token from a reason line.

    Mirrors the exact parsing `Lens.verdict()` itself uses internally
    (`r.split(":")[1].strip().split()[0]`) to determine survival, so this
    helper asserts on the same thing the production code reads.
    """
    return reason.split(":")[1].strip().split()[0]


# ===========================================================================
# Criterion 6 -- deflated Sharpe must use strategy_returns, not fwd.values
# ===========================================================================


def test_criterion6_none_strategy_returns_not_evaluated_names_missing_input() -> None:
    """Spec test 1: strategy_returns=None -> NOT_EVALUATED, reason names what's missing.

    Asserts the exact criterion-6 line (reasons[5]), not just the overall verdict.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    verdict = _call_verdict(panel, strategy_returns=None)

    reason = verdict.reasons[5]
    assert reason.startswith("6. Deflated Sharpe criterion:")
    assert _criterion_token(reason) == "NOT_EVALUATED"
    assert "PASS" not in reason
    assert "FAIL" not in reason
    assert "strategy_returns" in reason


def test_criterion6_not_evaluated_is_excluded_from_overall_verdict() -> None:
    """Spec test 2: NOT_EVALUATED must not count as a PASS in the overall verdict.

    Mirrors the precedent already established for criterion 5
    (`test_verdict_criterion5_not_evaluated_without_latency_profile` in
    test_lens.py): every OTHER criterion passes on `_all_pass_panel()`, and
    criterion 6 is NOT_EVALUATED (strategy_returns=None) -- the overall
    verdict must still be SURVIVED, because NOT_EVALUATED is neither a
    PASS nor a FAIL, it is excluded from the aggregation entirely.

    To show this is not merely "criterion 6 is ignored" (a bug that would
    also make FAIL disappear), a second call on the SAME panel supplies
    strategy_returns that genuinely FAIL deflation and flips survived to
    False -- proving NOT_EVALUATED and an actual FAIL are treated
    differently, not both silently dropped.
    """
    panel = _all_pass_panel()

    not_evaluated_verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)
    for i in (0, 1, 2, 3, 4, 6):
        assert _criterion_token(not_evaluated_verdict.reasons[i]) == "PASS", (
            f"criterion {i + 1} must PASS for this precedent test to isolate criterion 6"
        )
    assert _criterion_token(not_evaluated_verdict.reasons[5]) == "NOT_EVALUATED"
    assert not_evaluated_verdict.survived is True

    failing_returns = np.random.default_rng(2).normal(0.0, 0.001, size=500)
    explicit_fail_verdict = _call_verdict(
        panel, strategy_returns=failing_returns, effective_n_trials=2
    )
    assert _criterion_token(explicit_fail_verdict.reasons[5]) == "FAIL"
    assert explicit_fail_verdict.survived is False


def test_criterion6_single_trial_not_evaluated_states_cannot_deflate() -> None:
    """Spec test 3: strategy_returns supplied with effective_n_trials=1 -> NOT_EVALUATED.

    A one-trial "deflated" Sharpe is just a Sharpe; the reason must say a single
    declared trial cannot be deflated (this is the exact misreport being fixed).
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    strategy_returns = np.random.default_rng(7).normal(0.001, 0.001, size=200)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=1)

    reason = verdict.reasons[5]
    assert _criterion_token(reason) == "NOT_EVALUATED"
    assert "trial" in reason.lower()
    assert "deflat" in reason.lower() or "single" in reason.lower()


@pytest.mark.parametrize(
    "seed, mean, expected_token",
    [
        (1, 0.01, "PASS"),
        (2, 0.0, "FAIL"),
    ],
    ids=["PASS", "FAIL"],
)
def test_criterion6_pass_and_fail_with_multiple_trials(
    seed: int, mean: float, expected_token: str
) -> None:
    """Spec test 4: with effective_n_trials>=2, criterion 6 both PASSes and FAILs.

    Measured directly against `deflated_sharpe`/`expected_max_sharpe` (both
    pre-existing, untouched utilities): at trials=2,
    expected_max_sharpe(2, var_trial_sharpes=1.0) ~= 0.5198.
    seed=1, normal(mean=0.01, sigma=0.001, n=500) -> DSR=1.0 (saturated) > 0.52 -> PASS.
    seed=2, normal(mean=0.0, sigma=0.001, n=500) -> DSR ~= 0.05 < 0.52 -> FAIL.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    strategy_returns = np.random.default_rng(seed).normal(mean, 0.001, size=500)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=2)

    reason = verdict.reasons[5]
    assert _criterion_token(reason) == expected_token


@pytest.mark.parametrize(
    "strategy_returns",
    [
        np.array([], dtype=np.float64),
        np.array([np.nan, np.nan, np.nan], dtype=np.float64),
        np.array([np.nan, 0.001, np.nan], dtype=np.float64),
    ],
    ids=["empty", "all_nan", "single_finite_value"],
)
def test_criterion6_degenerate_strategy_returns_not_evaluated(
    strategy_returns: np.ndarray,
) -> None:
    """Spec test 5: empty / all-NaN / single-finite-value strategy_returns -> NOT_EVALUATED.

    Must never crash and must never report 0.0-as-a-Sharpe.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=5)

    reason = verdict.reasons[5]
    assert _criterion_token(reason) == "NOT_EVALUATED"


def test_criterion6_two_or_three_finite_values_not_evaluated_not_crash() -> None:
    """Extra branch-coverage test beyond the spec's literal minimum (>= 2 finite is not
    enough for `deflated_sharpe` to return a non-NaN value: it needs skew (t>=3) AND
    kurtosis (t>=4) to be finite). A supplied strategy_returns with exactly 3 finite
    values clears the ">= 2 finite" gate but `deflated_sharpe` itself still returns NaN
    -- the implementation must catch this and report NOT_EVALUATED too, never crash and
    never silently treat NaN as FAIL or 0.0.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    strategy_returns = np.array([0.01, 0.02, -0.005], dtype=np.float64)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=5)

    assert _criterion_token(verdict.reasons[5]) == "NOT_EVALUATED"


def test_criterion6_regression_guard_positive_fwd_drift_does_not_force_pass() -> None:
    """Spec test 6, the regression guard for the actual bug.

    `_all_pass_panel()` drifts strongly POSITIVE (effect=0.01 injected in every
    symbol/year) -- exactly the shape of panel that made the OLD `fwd.values`-based
    criterion 6 report PASS at its default `effective_n_trials=1` (see spec: "the
    branch reduces to dsr > 0.0 -- i.e. did the universe drift up over the sample").

    Here `strategy_returns` is a clearly LOSING series (negative mean), independent of
    the panel's own drift. THIS test must fail against an implementation that still
    reads `fwd.values` for criterion 6 (even one that merely added the
    `strategy_returns` parameter to the signature without wiring it in): such an
    implementation would compute the deflated Sharpe from the panel's own positive
    drift and report PASS, exactly reproducing the historical defect. A correct
    implementation reads `strategy_returns` and reports FAIL.

    Measured (seed=3): strategy_returns = normal(mean=-0.005, sigma=0.001, n=500),
    DSR(strategy_returns) ~= 3.68e-218 (numerically zero) < expected_max_sharpe(2,
    var_trial_sharpes=1.0) ~= 0.5198 -> FAIL is the only correct outcome.
    """
    panel = _all_pass_panel()
    strategy_returns = np.random.default_rng(3).normal(-0.005, 0.001, size=500)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=2)

    reason = verdict.reasons[5]
    assert _criterion_token(reason) != "PASS", (
        "criterion 6 PASSed using the panel's own positive drift instead of the "
        "supplied (losing) strategy_returns -- this is the exact defect being repaired"
    )
    assert _criterion_token(reason) == "FAIL"


# ===========================================================================
# Criterion 7 -- a year is complete iff the panel window fully spans it
# ===========================================================================


def test_criterion7_window_ending_midyear_excludes_partial_year() -> None:
    """Spec test 7: a window ending mid-year (2025-07-31) excludes 2025 as partial.

    Panel spans 2023-01-01 .. 2025-07-31 with sessions in 2023, 2024, and 2025.
    complete(2023): first(2023-01-01) <= 2023-01-01 and last(2025-07-31) >= 2023-12-31.
    complete(2024): first <= 2024-01-01 and last >= 2024-12-31.
    complete(2025): last(2025-07-31) >= 2025-12-31 is FALSE -> excluded, partial.
    """
    dates = [
        dt.date(2023, 1, 1),
        dt.date(2023, 6, 15),
        dt.date(2024, 3, 10),
        dt.date(2024, 11, 20),
        dt.date(2025, 7, 31),
    ]
    panel = _build_dated_panel(dates, 40, {2023, 2024, 2025}, effect=0.01, sigma=0.0005, seed=10)

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert reason.startswith("7. Recent-years cost gate criterion:")
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2025" in reason
    assert "excluded" in reason.lower()
    assert "2023" in reason and "2024" in reason


def test_criterion7_window_ending_exactly_dec31_treats_year_complete() -> None:
    """Spec test 8: a window ending exactly 2024-12-31 treats 2024 as complete.

    Boundary case: panel_last_date == date(2024, 12, 31) satisfies `>=`, so 2024
    must be usable (not excluded as partial).
    """
    dates = [
        dt.date(2023, 1, 1),
        dt.date(2023, 6, 1),
        dt.date(2024, 6, 1),
        dt.date(2024, 12, 31),
    ]
    panel = _build_dated_panel(dates, 40, {2023, 2024}, effect=0.01, sigma=0.0005, seed=11)

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2024" in reason
    assert "2023" in reason


def test_criterion7_window_starting_midyear_excludes_partial_year() -> None:
    """Spec test 9: a window STARTING mid-year (2018-06-01) excludes 2018 as partial.

    The completeness rule is symmetric on both ends -- this is the start-side twin of
    test 7 (the end-side exclusion), and must be tested independently.

    Panel spans 2018-06-01 .. 2020-12-31 with sessions in 2018, 2019, 2020.
    complete(2018): first(2018-06-01) <= 2018-01-01 is FALSE -> excluded, partial.
    complete(2019), complete(2020): both hold.
    """
    dates = [
        dt.date(2018, 6, 1),
        dt.date(2018, 9, 1),
        dt.date(2019, 4, 1),
        dt.date(2019, 10, 1),
        dt.date(2020, 4, 1),
        dt.date(2020, 12, 31),
    ]
    panel = _build_dated_panel(dates, 40, {2018, 2019, 2020}, effect=0.01, sigma=0.0005, seed=12)

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2018" in reason
    assert "excluded" in reason.lower()
    assert "2019" in reason and "2020" in reason


@pytest.mark.parametrize(
    "dates, signal_years, seed",
    [
        (
            [dt.date(2024, 1, 1), dt.date(2024, 6, 1), dt.date(2024, 12, 31)],
            {2024},
            13,
        ),
        (
            [dt.date(2024, 6, 1), dt.date(2024, 8, 1)],
            {2024},
            14,
        ),
    ],
    ids=["one-complete-year", "one-partial-year"],
)
def test_criterion7_fewer_than_two_complete_years_not_evaluated(
    dates: list[dt.date], signal_years: set[int], seed: int
) -> None:
    """Spec test 10: fewer than two complete years -> NOT_EVALUATED, never PASS.

    Covers both sub-cases: a single YEAR that IS itself complete (only one exists, so
    the >= 2 requirement still fails) and a single year that is only PARTIAL (0
    complete years).
    """
    panel = _build_dated_panel(dates, 40, signal_years, effect=0.01, sigma=0.0005, seed=seed)

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) == "NOT_EVALUATED"
    assert "PASS" not in reason


def test_criterion7_uses_two_most_recent_complete_years_by_number() -> None:
    """Spec test 11: the two years used are the two most recent COMPLETE years,
    asserted explicitly by year number, not merely by the resulting figure.

    Panel spans 2021-01-01 .. 2024-12-31 with FOUR complete years available
    (2021, 2022, 2023, 2024). Only the two most recent (2023, 2024) may be named in
    the reason; 2021 and 2022 must not appear, even though they are also complete.
    """
    dates = [
        dt.date(2021, 1, 1),
        dt.date(2022, 6, 1),
        dt.date(2023, 6, 1),
        dt.date(2024, 12, 31),
    ]
    panel = _build_dated_panel(
        dates, 40, {2021, 2022, 2023, 2024}, effect=0.01, sigma=0.0005, seed=15
    )

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2023" in reason and "2024" in reason
    assert "2021" not in reason
    assert "2022" not in reason


def test_criterion7_regression_guard_matches_spec_scenario() -> None:
    """Regression guard reproducing the spec's own worked example almost exactly:
    "On the standard 2018-01-01..2025-07-31 window this changes criterion 7 from
    (2024, 2025) to (2023, 2024)."

    Unlike the other criterion-7 fixtures (which use a handful of sessions per year
    and would already read NOT_EVALUATED under the OLD `>= 20 sessions` threshold),
    this panel gives each year >= 20 real sessions -- enough for the OLD threshold to
    actually engage and exhibit the bug being repaired: a partial (~1-month) final
    year gets misreported as one of the two "complete" recent years. 2023 is anchored
    to include exactly 2023-01-01 so it, too, clears the new date-based rule.
    """
    dates_2023 = [dt.date(2023, 1, 1)] + _consecutive_weekdays(2023, 21, 1, 2)
    dates_2024 = _consecutive_weekdays(2024, 22, 1, 2)
    dates_2025 = _consecutive_weekdays(2025, 22, 1, 2)  # ends well before Dec 31: partial
    all_dates = sorted(dates_2023 + dates_2024 + dates_2025)
    panel = _build_dated_panel(
        all_dates, 5, {2023, 2024, 2025}, effect=0.02, sigma=0.001, seed=20
    )

    verdict = _call_verdict(panel, strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2023" in reason and "2024" in reason
    assert "excluded" in reason.lower() and "2025" in reason


# ===========================================================================
# Both criteria: full seven-line report, and determinism
# ===========================================================================


def test_all_seven_criteria_reported_in_order_with_irregular_session() -> None:
    """Spec test 12: every one of the seven criteria is reported, in order, even after
    some fail -- including one where criteria 6 and 7 are both NOT_EVALUATED.

    Includes an irregular-length session (60 bars then 15 bars, never assuming a
    fixed stride per CLAUDE.md rule 5) to confirm the day-offset handling that
    criteria 2 and 7 rely on does not silently assume equal-length sessions.
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    panel = _build_dated_panel(dates, [60, 15], {2024}, effect=0.01, sigma=0.0005, seed=21)

    verdict = _call_verdict(
        panel,
        latency_profile={0: 1.0, 1: 0.2, 2: 0.1},
        strategy_returns=None,
        effective_n_trials=1,
    )

    assert len(verdict.reasons) == 7
    for i, reason in enumerate(verdict.reasons):
        assert reason.startswith(f"{i + 1}."), f"reason {i} out of order: {reason!r}"
    # Mixed outcomes confirm this is a real exercise, not a trivially-all-pass panel:
    # criterion 2 (single year) fails sign stability, criteria 6 and 7 are
    # NOT_EVALUATED (no strategy_returns; fewer than two complete years).
    assert _criterion_token(verdict.reasons[1]) == "FAIL"
    assert _criterion_token(verdict.reasons[5]) == "NOT_EVALUATED"
    assert _criterion_token(verdict.reasons[6]) == "NOT_EVALUATED"


def test_verdict_determinism_same_inputs_and_seed() -> None:
    """Spec test 13: same inputs and seed -> identical verdict text.

    Exercises both repaired criteria together: strategy_returns supplied,
    effective_n_trials >= 2 (criterion 6 actually deflates), and a panel wide enough
    for criterion 7 to actually evaluate (not NOT_EVALUATED).
    """
    panel = _all_pass_panel(seed=99)
    strategy_returns = np.random.default_rng(42).normal(0.01, 0.001, size=300)

    first = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=3)
    second = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=3)

    assert first.reasons == second.reasons
    assert first.survived == second.survived
    np.testing.assert_allclose(first.cost_hurdle_bps, second.cost_hurdle_bps)
