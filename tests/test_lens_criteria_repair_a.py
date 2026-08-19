"""Independent test suite A for `specs/lens_criteria_6_7_repair.md` (AMENDED, commit
960b70e). Written from the amended spec alone, before any implementation change to
`nifty_quant.research.lens.Lens.verdict()` exists. Do not read or reference
`tests/test_lens_criteria_repair_b.py` -- that is the second, independently authored
suite; per the spec, neither author may see the other's file.

Two defects are repaired here:

- Criterion 6 (deflated Sharpe) must be computed on `strategy_returns`, a new
  keyword-only parameter of `verdict()` carrying the strategy's own per-period P&L
  series -- never on `fwd.values` (the unconditional forward return of every symbol at
  every row). It ALSO deflates against the correct benchmark: `deflated_sharpe`
  returns a PROBABILITY; `expected_max_sharpe` returns a SHARPE LEVEL. The correct
  form passes `sr0=expected_max_sharpe(n, var)` into `deflated_sharpe`, then compares
  the resulting probability against a named significance constant -- never
  `dsr > expected_max_sharpe(...)` directly (impossible for n >= 4). Any NaN `dsr`
  (t < 4, or negligible std) reports `NOT_EVALUATED`, never FAIL.
- Criterion 7 (recent-years cost gate) must treat a calendar year as "complete" iff
  the panel holds at least one session in EACH of that year's 12 calendar months --
  `{d.month for d in panel.dates if d.year == year} == {1..12}` -- not a date-span
  test and not the old `session_counts.get(year, 0) >= 20` threshold (itself a
  hand-chosen constant CLAUDE.md rule 8 forbids).
- `NOT_EVALUATED` never blocks `survived` (criterion 5's existing precedent, now
  explicit for every criterion), and `HypothesisVerdict.any_not_evaluated` /
  `explain()` must surface that the verdict is INCOMPLETE whenever it applies.

Fixtures and low-level helpers (`_SYMBOLS`, `_N_SYMBOLS`, `_DEFAULT_VOLUME`,
`_PASS_LATENCY`, `_signal_means`, `_close_prices_from_log_returns`, `_build_panel`,
`_all_pass_panel`) are reused from `tests/test_lens.py` per the task's "reuse existing
fixtures" instruction; `tests.test_engine` / `tests.test_pending_orders` already
establish cross-test-module imports as an accepted pattern in this repo.

`_all_pass_panel()` cannot be reused for criterion-7 PASS scenarios: its sessions are
always 2-per-year, both in January (`_build_panel`'s date generator places every
session on `date(year, 1, 2 + i)`), so no year in it ever has 12-month coverage under
the amended rule -- criterion 7 is `NOT_EVALUATED` on it regardless of which
implementation runs (verified directly: even the CURRENT, unrepaired code reports
`complete_years=none` on it, since 2 sessions/year also never clears the old `>= 20`
threshold). Criterion-7 fixtures here therefore use a dedicated dated-panel builder
that places one session per calendar month so completeness can be controlled exactly.
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

# Sentinel distinguishing "strategy_returns was not passed at all" (so the call site
# exercises whatever DEFAULT the implementation under test uses) from "strategy_returns
# was passed explicitly as None". Criterion-7-only tests use the former: on the
# current, unrepaired `Lens.verdict()` -- which has no `strategy_returns` parameter --
# omitting the keyword entirely lets the OLD criterion-6 codepath run to completion
# (rather than raising TypeError from an unrecognised kwarg swallowed into `**kw` and
# forwarded to `expectancy.conditional_expectancy()`), so a criterion-7 test's failure
# against old code comes from the criterion-7 text actually being wrong, not from an
# unrelated crash on an unrelated criterion.
_UNSET = object()


# ---------------------------------------------------------------------------
# Local helpers -- criterion 7 is about calendar-MONTH coverage per year, which
# `tests.test_lens._build_panel` cannot exercise (see module docstring). This builder
# takes an explicit list of session dates (and, per CLAUDE.md rule 5, an optionally
# variable bar count per session -- never a fixed stride) so a test can place sessions
# in exactly the months it needs.
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
    """Build a Panel from explicit session dates, e.g. to control which calendar
    months are covered for a given year (criterion 7 tests).

    `bars_per_session` may be a single int (uniform stride) or a per-day list --
    callers use the list form to include an irregular (e.g. Muhurat-like short)
    session, since CLAUDE.md rule 5 forbids assuming a fixed stride.

    Callers must keep `len(dates) * bars_per_session * effect` well under ~85: the
    close-price series is `exp(cumsum(returns))` with NO reset between sessions (a
    real, continuous log-price series), and float32 overflows past ~exp(88). This is
    an artifact of choosing `effect` and `bars_per_session` for a given fixture size,
    not a property of any criterion under test.
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


def _monthly_session_dates(
    year: int, day: int = 15, months: range = range(1, 13)
) -> list[dt.date]:
    """One session per requested calendar month of `year`, on a fixed day-of-month.

    The default `months=range(1, 13)` gives full 12-month coverage (a "complete"
    year under the amended rule); a restricted range gives a deliberately partial
    year, e.g. `months=range(1, 8)` for "Jan-Jul only".
    """
    return [dt.date(year, m, day) for m in months]


def _call_verdict(
    panel: Panel,
    *,
    hypothesis_id: str = "H_repair_a",
    latency_profile: dict[int, float] | None = _PASS_LATENCY,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = _UNSET,  # type: ignore[assignment]
    seed: int = 0,
    horizon: int = 1,
) -> HypothesisVerdict:
    """Call `Lens.verdict()`, optionally with the new `strategy_returns` keyword.

    `strategy_returns` defaults to the `_UNSET` sentinel: when left at that default,
    the keyword is NOT forwarded to `verdict()` at all, so criterion-7-only tests
    (which don't care about criterion 6) exercise whichever default the implementation
    under test provides rather than colliding with an implementation that has not yet
    added the parameter. Criterion-6 tests always pass `strategy_returns` explicitly
    (including `None`), since that is exactly what they are testing.
    """
    lens = Lens(panel, seed=seed)
    kwargs: dict[str, object] = dict(
        latency_profile=latency_profile,
        effective_n_trials=effective_n_trials,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )
    if strategy_returns is not _UNSET:
        kwargs["strategy_returns"] = strategy_returns
    return lens.verdict(hypothesis_id, "return_1", horizon, **kwargs)


def _criterion_token(reason: str) -> str:
    """Extract the PASS/FAIL/NOT_EVALUATED token from a reason line.

    Mirrors the exact parsing `Lens.verdict()` itself uses internally
    (`r.split(":")[1].strip().split()[0]`) to determine survival, so this helper
    asserts on the same thing the production code reads.
    """
    return reason.split(":")[1].strip().split()[0]


def _isolate_c6_panel() -> Panel:
    """A panel where every OTHER criterion genuinely PASSes (not merely
    NOT_EVALUATED), so criterion 6 alone can be isolated: 6 years (2019-2024, the
    hardcoded `n_years_sign_consistent >= 6` in `StabilityReport.sign_stable` needs
    exactly that many same-signed years), each with one session per calendar month
    (12-month coverage -> the two most recent, 2023 and 2024, are usable by criterion
    7), and bars_per_session=100 (enough bars per session to populate both the "open"
    and "mid" time-of-day buckets so criterion 4 does not read "concentrated in a
    single time-of-day bucket").
    """
    years = range(2019, 2025)
    dates: list[dt.date] = []
    for year in years:
        dates += _monthly_session_dates(year)
    return _build_dated_panel(
        dates, 100, set(years), effect=0.01, sigma=0.0005, seed=50
    )


# ===========================================================================
# Criterion 6 -- deflated Sharpe must use strategy_returns, correctly deflated
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
    test_lens.py, and now stated explicitly by the amended spec for every
    criterion): every OTHER criterion genuinely PASSES on `_isolate_c6_panel()`
    (verified directly below, not merely assumed), and criterion 6 is
    NOT_EVALUATED (strategy_returns=None) -- the overall verdict must still be
    SURVIVED, because NOT_EVALUATED is neither a PASS nor a FAIL, it is excluded
    from the aggregation entirely.

    To show this is not merely "criterion 6 is ignored" (a bug that would also make
    FAIL disappear), a second call on the SAME panel supplies strategy_returns that
    genuinely FAIL deflation and flips survived to False -- proving NOT_EVALUATED
    and an actual FAIL are treated differently, not both silently dropped.
    """
    panel = _isolate_c6_panel()

    not_evaluated_verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=None,
        effective_n_trials=1,
    )
    for i in (0, 1, 2, 3, 4, 6):
        assert _criterion_token(not_evaluated_verdict.reasons[i]) == "PASS", (
            f"criterion {i + 1} must genuinely PASS for this precedent test to "
            f"isolate criterion 6: {not_evaluated_verdict.reasons[i]!r}"
        )
    assert _criterion_token(not_evaluated_verdict.reasons[5]) == "NOT_EVALUATED"
    assert not_evaluated_verdict.survived is True

    failing_returns = np.random.default_rng(2).normal(0.0, 0.001, size=500)
    explicit_fail_verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=failing_returns,
        effective_n_trials=2,
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

    Uses the CORRECT deflation form: `deflated_sharpe(strategy_returns,
    sr0=expected_max_sharpe(n_trials, var))`, then compares the resulting
    PROBABILITY against a significance level -- never `dsr > expected_max_sharpe`
    directly (the second defect the amendment repairs).

    Measured directly against `deflated_sharpe`/`expected_max_sharpe` (both
    pre-existing, untouched utilities, `nifty_quant.backtest.metrics`):
    `expected_max_sharpe(2, var_trial_sharpes=1.0) ~= 0.5198`.
    - seed=1, normal(mean=0.01, sigma=0.001, n=500): raw per-period SR ~= 10, so
      `deflated_sharpe(arr, sr0=0.5198)` saturates to 1.0 -- PASS regardless of
      which reasonable significance level (e.g. anywhere in [0.90, 0.999]) the
      implementation under test picks.
    - seed=2, normal(mean=0.0, sigma=0.001, n=500): raw per-period SR ~= -0.05, so
      `deflated_sharpe(arr, sr0=0.5198)` is ~2e-37 (numerically zero) -- FAIL under
      any reasonable significance level.
    Both cases are deliberately built with extreme separation (saturated vs.
    numerically-zero probability) so the assertion does not depend on knowing the
    implementation's exact significance constant, per CLAUDE.md rule 9: driven by
    effect size, not by tuning a seed to land near an assumed threshold.
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


def test_criterion6_three_finite_values_not_evaluated_needs_at_least_four() -> None:
    """Amendment: "drop non-finite entries and evaluate on the remainder if >= 4
    finite values survive; otherwise NOT_EVALUATED." `deflated_sharpe` itself needs
    t >= 4 for a finite kurtosis estimate (`metrics.py:481-487`); a 3-finite-value
    array clears an ">= 2 finite" gate but not the amendment's actual ">= 4" gate --
    this pins that exact boundary, not just "some small N".
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    strategy_returns = np.array([0.01, 0.02, -0.005], dtype=np.float64)

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=5)

    assert _criterion_token(verdict.reasons[5]) == "NOT_EVALUATED"


def test_criterion6_partial_nan_with_four_finite_survivors_is_evaluated() -> None:
    """Amendment: partially-NaN strategy_returns must drop the non-finite entries
    and evaluate on the remainder once >= 4 finite values survive -- NOT_EVALUATED
    is for the case where too few survive, not for the mere presence of any NaN.

    Interleaves NaNs around 4 finite values with genuine variance (mean=0.6,
    sigma=0.001 -> raw per-period SR ~= 600, so deflated_sharpe saturates to ~1.0
    against `sr0=expected_max_sharpe(2, 1.0)~=0.52` regardless of implementation
    detail) so this is unambiguously evaluated and PASSing, not NOT_EVALUATED.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    finite = np.random.default_rng(5).normal(0.6, 0.001, size=4)
    strategy_returns = np.array(
        [np.nan, finite[0], finite[1], np.nan, finite[2], finite[3], np.nan],
        dtype=np.float64,
    )

    verdict = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=2)

    reason = verdict.reasons[5]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert _criterion_token(reason) == "PASS"


def test_criterion6_regression_guard_positive_fwd_drift_does_not_force_pass() -> None:
    """Spec test 6, the regression guard for the actual bug.

    `_all_pass_panel()` drifts strongly POSITIVE (effect=0.01 injected in every
    symbol/year) -- exactly the shape of panel that made the OLD `fwd.values`-based
    criterion 6 report PASS (the spec: "the branch reduces to dsr > 0.0 -- i.e. did
    the universe drift up over the sample").

    Here `strategy_returns` is a clearly LOSING series (negative mean), independent
    of the panel's own drift. THIS test fails against an implementation that still
    reads `fwd.values` for criterion 6 (even one that merely added the
    `strategy_returns` parameter to the signature without wiring it in): such an
    implementation would compute the deflated Sharpe from the panel's own positive
    drift and report PASS, exactly reproducing the historical defect. A correct
    implementation reads `strategy_returns` and reports FAIL.

    Measured (seed=3): strategy_returns = normal(mean=-0.005, sigma=0.001, n=500),
    raw per-period SR ~= -4.9, `deflated_sharpe(arr, sr0=0.5198)` ~= 7.4e-238
    (numerically zero) -- FAIL is the only correct outcome under any reasonable
    significance level.
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
# Criterion 7 -- a year is complete iff the panel covers all 12 of its months
# ===========================================================================


def test_criterion7_window_ending_midyear_excludes_partial_year() -> None:
    """Spec test 7: a window ending mid-year (2025, Jan-Jul only) excludes 2025 as
    partial. 2023 and 2024 each have a session in every one of their 12 months, so
    both are complete; 2025 has sessions only in months 1-7, so it is partial.
    """
    dates = (
        _monthly_session_dates(2023)
        + _monthly_session_dates(2024)
        + _monthly_session_dates(2025, months=range(1, 8))
    )
    panel = _build_dated_panel(dates, 240, {2023, 2024, 2025}, effect=0.01, sigma=0.0005, seed=10)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert reason.startswith("7. Recent-years cost gate criterion:")
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2025" in reason
    assert "excluded" in reason.lower()
    assert "2023" in reason and "2024" in reason


def test_criterion7_last_session_before_dec31_still_treats_year_complete() -> None:
    """Amendment's explicit required regression test: "a window ending 2023-12-31 has
    last session 2023-12-29 [under the real NSE calendar], so [the old date-span rule]
    marked a fully-covered year partial. Add a required test for exactly that case
    asserting 2023 is COMPLETE."

    2023's December session lands on the 29th, not the 31st -- the panel's own last
    date is 2023-12-29, strictly before date(2023, 12, 31). Under the OLD
    `panel_last_date >= date(year, 12, 31)` rule this would misclassify 2023 as
    partial; under the amended 12-MONTH-coverage rule, December has a session (the
    29th) so 2023 is complete regardless of which day of the month it falls on.
    """
    dates = _monthly_session_dates(2022) + _monthly_session_dates(
        2023, months=range(1, 12)
    ) + [dt.date(2023, 12, 29)]
    panel = _build_dated_panel(dates, 240, {2022, 2023}, effect=0.01, sigma=0.0005, seed=11)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2023" in reason
    assert "2022" in reason
    assert "excluded" not in reason.lower(), (
        f"2023 must be usable, not excluded as partial: {reason!r}"
    )


def test_criterion7_window_ending_exactly_dec31_treats_year_complete() -> None:
    """Spec test 8: a window ending exactly 2024-12-31 treats 2024 as complete.

    Literal boundary case named by the spec: the panel's last session date is
    exactly `date(2024, 12, 31)`, and 2024 has a session in every one of its 12
    months, so it must be usable (not excluded as partial).
    """
    dates = _monthly_session_dates(2023) + _monthly_session_dates(
        2024, months=range(1, 12)
    ) + [dt.date(2024, 12, 31)]
    panel = _build_dated_panel(dates, 240, {2023, 2024}, effect=0.01, sigma=0.0005, seed=16)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2024" in reason
    assert "2023" in reason


def test_criterion7_window_starting_midyear_excludes_partial_year() -> None:
    """Spec test 9: a window STARTING mid-year (2018, Jun-Dec only) excludes 2018 as
    partial -- the completeness rule is symmetric on both ends, and this is the
    start-side twin of the end-side exclusion tested above; both must be tested
    independently. 2019 and 2020 each cover all 12 months and remain usable.
    """
    dates = (
        _monthly_session_dates(2018, months=range(6, 13))
        + _monthly_session_dates(2019)
        + _monthly_session_dates(2020)
    )
    panel = _build_dated_panel(dates, 240, {2018, 2019, 2020}, effect=0.01, sigma=0.0005, seed=12)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2018" in reason
    assert "excluded" in reason.lower()
    assert "2019" in reason and "2020" in reason


@pytest.mark.parametrize(
    "dates, signal_years, seed",
    [
        (
            _monthly_session_dates(2024),
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
    complete years, since it only has sessions in June and August).
    """
    panel = _build_dated_panel(dates, 240, signal_years, effect=0.01, sigma=0.0005, seed=seed)

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) == "NOT_EVALUATED"
    assert "PASS" not in reason


def test_criterion7_uses_two_most_recent_complete_years_by_number() -> None:
    """Spec test 11: the two years used are the two most recent COMPLETE years,
    asserted explicitly by year number, not merely by the resulting figure.

    Four years (2021-2024), each with 12-month coverage, are all complete. Only the
    two most recent (2023, 2024) may be named in the reason; 2021 and 2022 must not
    appear, even though they too are complete.
    """
    dates = (
        _monthly_session_dates(2021)
        + _monthly_session_dates(2022)
        + _monthly_session_dates(2023)
        + _monthly_session_dates(2024)
    )
    panel = _build_dated_panel(
        dates, 150, {2021, 2022, 2023, 2024}, effect=0.01, sigma=0.0005, seed=15
    )

    verdict = _call_verdict(panel, effective_n_trials=1)

    reason = verdict.reasons[6]
    assert _criterion_token(reason) != "NOT_EVALUATED"
    assert "2023" in reason and "2024" in reason
    assert "2021" not in reason
    assert "2022" not in reason


# ===========================================================================
# Both criteria: full seven-line report, any_not_evaluated / explain(), and
# determinism
# ===========================================================================


def test_all_seven_criteria_reported_in_order_with_irregular_session() -> None:
    """Spec test 12: every one of the seven criteria is reported, in order, even
    after some fail -- including one where criteria 6 and 7 are both NOT_EVALUATED.

    Includes an irregular-length session (60 bars then 15 bars, never assuming a
    fixed stride per CLAUDE.md rule 5) to confirm the day-offset handling that
    criteria 2 and 7 rely on does not silently assume equal-length sessions.
    """
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    panel = _build_dated_panel(dates, [60, 15], {2024}, effect=0.01, sigma=0.0005, seed=21)

    verdict = _call_verdict(
        panel,
        latency_profile={0: 1.0, 1: 0.2, 2: 0.1},
        effective_n_trials=1,
    )

    assert len(verdict.reasons) == 7
    for i, reason in enumerate(verdict.reasons):
        assert reason.startswith(f"{i + 1}."), f"reason {i} out of order: {reason!r}"
    # Mixed outcomes confirm this is a real exercise, not a trivially-all-pass panel:
    # criterion 2 (single year) fails sign stability, criteria 6 and 7 are
    # NOT_EVALUATED (no strategy_returns; fewer than two complete years -- a single
    # year with sessions only in January cannot cover all 12 months).
    assert _criterion_token(verdict.reasons[1]) == "FAIL"
    assert _criterion_token(verdict.reasons[5]) == "NOT_EVALUATED"
    assert _criterion_token(verdict.reasons[6]) == "NOT_EVALUATED"


def test_any_not_evaluated_true_and_explain_states_incomplete() -> None:
    """Amendment item 5: `HypothesisVerdict` must expose a flag (e.g.
    `any_not_evaluated`) that is True when any criterion is NOT_EVALUATED, and
    `explain()` must state prominently that the verdict is INCOMPLETE -- not merely
    list NOT_EVALUATED among seven reason lines where a reader skims past it.
    """
    panel = _build_panel((2024,), sessions_per_year=5, bars_per_session=30, seed=0)
    verdict = _call_verdict(panel, strategy_returns=None)

    assert verdict.any_not_evaluated is True
    assert "INCOMPLETE" in verdict.explain()


def test_any_not_evaluated_false_when_every_criterion_evaluated() -> None:
    """The mirror case: when every one of the seven criteria is actually evaluated
    (none NOT_EVALUATED), `any_not_evaluated` must be False.
    """
    panel = _isolate_c6_panel()
    strategy_returns = np.random.default_rng(1).normal(0.01, 0.001, size=500)

    verdict = _call_verdict(
        panel,
        latency_profile=_PASS_LATENCY,
        strategy_returns=strategy_returns,
        effective_n_trials=2,
    )

    for reason in verdict.reasons:
        assert _criterion_token(reason) != "NOT_EVALUATED", reason
    assert verdict.any_not_evaluated is False


def test_verdict_determinism_same_inputs_and_seed() -> None:
    """Spec test 13: same inputs and seed -> identical verdict text.

    Exercises both repaired criteria together: strategy_returns supplied,
    effective_n_trials >= 2 (criterion 6 actually deflates), and a panel with two
    complete years (criterion 7 actually evaluates, not NOT_EVALUATED).
    """
    panel = _isolate_c6_panel()
    strategy_returns = np.random.default_rng(42).normal(0.01, 0.001, size=300)

    first = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=3)
    second = _call_verdict(panel, strategy_returns=strategy_returns, effective_n_trials=3)

    assert first.reasons == second.reasons
    assert first.survived == second.survived
    assert first.any_not_evaluated == second.any_not_evaluated
