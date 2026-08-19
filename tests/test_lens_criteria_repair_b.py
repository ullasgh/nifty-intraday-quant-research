"""Independent test suite (author B) for specs/lens_criteria_6_7_repair.md, AMENDED
version (commit 960b70e).

Written from the amended spec ALONE, before any implementation of the repair exists.
Does not read, import, or otherwise reference tests/test_lens_criteria_repair_a.py --
two independent readings of the same contract are the point; copying the other
author's tests would destroy it.

This file supersedes and REPLACES a prior draft written against the FIRST-DRAFT
(pre-amendment) spec, which used a date-span completeness rule for criterion 7 and a
comparison-preserving (still-broken) criterion 6 formula. The amended spec:

  1. Criterion 6 has a SECOND defect: deflated_sharpe returns a PROBABILITY in
     [0, 1], expected_max_sharpe returns a SHARPE LEVEL (~1.05 at n=4). Comparing
     them directly makes criterion 6 an automatic FAIL for n >= 4. The fix passes
     expected_max_sharpe(...) as deflated_sharpe's sr0 and compares the resulting
     PROBABILITY against a named significance constant.
  2. Criterion 7's completeness rule is 12-MONTH COVERAGE per calendar year, not a
     date-span or session-count check. A sparse fixture (e.g. two January sessions
     per year, as tests/test_lens.py's `_all_pass_panel` uses) never satisfies this
     -- every year in that fixture would be PARTIAL -- so fixtures asserting a real
     PASS/FAIL/years-used outcome for criterion 7 need genuine per-month density.

Reuses panel/fixture helpers from tests/test_lens.py where they fit. Helpers written
fresh here (none of this can be expressed by tests/test_lens.py's own fixtures,
which always anchor a year at Jan 2 and never reach other months):
  - month-dense per-year date lists, for criterion 7's completeness rule;
  - irregular per-day bar counts (test_lens._session_grid takes one fixed
    bars_per_session for the whole panel; CLAUDE.md rule 5 forbids assuming a fixed
    bars/session anyway);
  - a fixture with a controlled, positive-only cross-sectional drift, needed to
    reproduce the exact unconditional-drift bug criterion 6 must no longer report
    as PASS.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from test_lens import (
    _DEFAULT_VOLUME,
    _FAIL_LATENCY,
    _IST,
    _N_SYMBOLS,
    _PASS_LATENCY,
    _SYMBOLS,
    _build_panel,
    _close_prices_from_log_returns,
    _signal_means,
)

from nifty_quant.data.panel import Panel
from nifty_quant.research import expectancy
from nifty_quant.research.lens import HypothesisVerdict, Lens

# reasons[] is 0-indexed; criteria are numbered 1..7 in their text.
_C6_LINE = 5  # "6. Deflated Sharpe criterion: ..."
_C7_LINE = 6  # "7. Recent-years cost gate criterion: ..."


# ---------------------------------------------------------------------------
# Panel helpers specific to this suite
# ---------------------------------------------------------------------------


def _session_grid_irregular(
    dates: list[dt.date], bars_per_session: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like test_lens._session_grid, but bars_per_session varies PER DAY.

    Needed so criterion 7's window-edge fixtures can include an irregular session
    (e.g. a Muhurat-like 60-bar day) rather than assuming a fixed bar count for
    every session, per CLAUDE.md rule 5.
    """
    ts_chunks: list[np.ndarray] = []
    for day, n_bars in zip(dates, bars_per_session):
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n_bars, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.array([0, *np.cumsum(bars_per_session)], dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _dated_panel(
    dates: list[dt.date],
    bars_per_session: int | list[int],
    signal_years: set[int],
    *,
    effect: float = 0.01,
    sigma: float = 0.0005,
    seed: int = 0,
) -> Panel:
    """Build a panel spanning EXACTLY the given (possibly mid-year) dates.

    test_lens._build_panel always starts each year at Jan 2 and stops after
    `sessions_per_year` days, so its panel never exercises criterion 7's
    12-month-coverage rule in either direction (never reaches December, never
    covers every month). This lets a caller control the panel's exact session
    calendar, which is the entire subject under test.
    """
    n_days = len(dates)
    bars_list = (
        [bars_per_session] * n_days
        if isinstance(bars_per_session, int)
        else list(bars_per_session)
    )
    n_rows = sum(bars_list)
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_rows, _N_SYMBOLS), dtype=np.float64)

    row = 0
    for date, n_bars in zip(dates, bars_list):
        if date.year in signal_years:
            means = _signal_means(effect)
            block_means = np.tile(means, (n_bars, 1))
            block_sigma = sigma
        else:
            block_means = np.zeros((n_bars, _N_SYMBOLS), dtype=np.float64)
            block_sigma = 0.0
        noise = rng.normal(0.0, block_sigma, size=(n_bars, _N_SYMBOLS))
        returns[row : row + n_bars] = block_means + noise
        row += n_bars

    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.array(_DEFAULT_VOLUME, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid_irregular(dates, bars_list)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _monthly_dates(year: int, *, day: int = 10, months: range = range(1, 13)) -> list[dt.date]:
    """One session per requested calendar month of `year`.

    `day=10` is safe for every month (no Feb-29 edge case). The default `months`
    range gives full 12-month coverage -- a COMPLETE year under the amended
    criterion 7 rule. A restricted `months` range gives a PARTIAL year.
    """
    return [dt.date(year, m, day) for m in months]


def _positive_drift_panel(
    *,
    n_days: int = 40,
    bars_per_session: int = 30,
    drift: float = 0.02,
    noise_sigma: float = 0.0005,
    seed: int = 0,
    start: dt.date = dt.date(2022, 1, 3),
) -> Panel:
    """Panel where EVERY symbol drifts by the same strong positive amount.

    test_lens._signal_means (used by _dated_panel above) assigns 5 symbol means
    [-e, -e/2, 0, e/2, e], which sum to exactly zero -- fwd.values.flatten() over
    such a panel has mean 0 by construction, so it cannot demonstrate the bug.
    Here all symbols share one positive mean instead, so fwd.values (the
    UNCONDITIONAL forward return the old criterion 6 mistakenly deflated) drifts
    strongly positive while the strategy's own return series can still be flat
    or negative.
    """
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    bars_list = [bars_per_session] * n_days
    n_rows = n_days * bars_per_session
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, noise_sigma, size=(n_rows, _N_SYMBOLS))
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.array(_DEFAULT_VOLUME, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid_irregular(dates, bars_list)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _dense_all_pass_panel(seed: int = 0) -> Panel:
    """A panel with >= 6 sign-consistent years (criterion 2) and its two most
    recent years densely covering all 12 months each (criterion 7), so this suite
    can construct a case where every criterion is genuinely EVALUATED -- not
    NOT_EVALUATED for lack of qualifying data. 2018-2021 are sparse (2 sessions
    each, like test_lens._all_pass_panel) since criterion 2's n_years_total is
    explicitly unaffected by the completeness rule (spec: "criterion 2's
    n_years_total continues to count every year present, including partial
    ones"); only 2022 and 2023 need full 12-month density.
    """
    dates: list[dt.date] = []
    for year in (2018, 2019, 2020, 2021):
        dates.extend([dt.date(year, 1, 2), dt.date(year, 1, 3)])
    for year in (2022, 2023):
        dates.extend(_monthly_dates(year))
    return _dated_panel(
        sorted(dates),
        bars_per_session=240,
        signal_years={2018, 2019, 2020, 2021, 2022, 2023},
        effect=0.01,
        sigma=0.0005,
        seed=seed,
    )


def _call_verdict(
    panel: Panel,
    hypothesis_id: str = "H_REPAIR_B",
    *,
    latency_profile: dict[int, float] | None = _PASS_LATENCY,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = None,
    seed: int = 0,
    horizon: int = 1,
    n_boot: int = 30,
) -> HypothesisVerdict:
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
        n_boot=n_boot,
    )


# ---------------------------------------------------------------------------
# Criterion 6 -- deflated Sharpe on the STRATEGY's own returns
# ---------------------------------------------------------------------------


def test_criterion6_none_strategy_returns_is_not_evaluated_and_names_the_input() -> None:
    """Spec item 1: strategy_returns=None -> NOT_EVALUATED, reason names it."""
    verdict = _call_verdict(_dense_all_pass_panel(), strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason
    assert "strategy_returns" in reason


def test_criterion6_not_evaluated_does_not_count_as_pass_in_overall_verdict() -> None:
    """Spec item 2: NOT_EVALUATED must not silently count as PASS in the overall
    verdict -- mirrors criterion 5's existing precedent (test_lens.py's
    test_verdict_criterion5_not_evaluated_without_latency_profile: every other
    criterion PASSes, the NOT_EVALUATED criterion is skipped, and the hypothesis
    still survives). Requires a fixture where criteria 1-5 and 7 ALL genuinely
    PASS, which needs real 12-month density for criterion 7 -- a sparse
    January-only fixture would leave criterion 7 NOT_EVALUATED too and could not
    isolate criterion 6 as the sole unevaluated criterion this test requires."""
    verdict = _call_verdict(
        _dense_all_pass_panel(),
        latency_profile=_PASS_LATENCY,
        strategy_returns=None,
        effective_n_trials=1,
    )

    for i in range(7):
        if i == _C6_LINE:
            assert "NOT_EVALUATED" in verdict.reasons[i]
        else:
            assert "PASS" in verdict.reasons[i], verdict.reasons[i]

    assert verdict.survived is True


def test_criterion6_single_declared_trial_is_not_evaluated() -> None:
    """Spec item 3: strategy_returns supplied but effective_n_trials=1 ->
    NOT_EVALUATED. A one-trial "deflated" Sharpe is just a Sharpe; calling it
    deflated is the misreport being fixed."""
    strategy_returns = np.random.default_rng(42).normal(0.003, 0.001, size=300)

    verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=1
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "trial" in reason.lower()

    # Distinguishable from item 1's "no input at all" NOT_EVALUATED -- a reader
    # must be able to tell WHY it was not evaluated.
    none_reason = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=None, effective_n_trials=1
    ).reasons[_C6_LINE]
    assert reason != none_reason


def test_criterion6_second_defect_fix_passes_at_a_realistic_trial_count() -> None:
    """The most important regression in this suite for the SECOND defect: with
    the old (pre-amendment-spec) comparison `dsr > expected_max_sharpe(n, 1.0)`,
    dsr is a probability bounded by 1.0 while expected_max_sharpe(10, 1.0) ~=
    1.5746 -- so criterion 6 could NEVER pass once wired up at a realistic trial
    count (the spec's own words: "a realistic effective_n_trials is ~10").  This
    constructs a strategy_returns series with an overwhelming, unambiguous Sharpe
    ratio (mean/std = 20 per period over 500 observations) so the fixed
    comparison -- dsr = deflated_sharpe(returns, sr0=expected_max_sharpe(...))
    compared against a probability significance level -- passes regardless of
    the exact (rule-8-pending) significance constant chosen, while the
    old/unfixed comparison could not pass for ANY signal at this trial count."""
    strategy_returns = np.random.default_rng(11).normal(0.02, 0.001, size=500)

    verdict = _call_verdict(
        _dense_all_pass_panel(),
        strategy_returns=strategy_returns,
        effective_n_trials=10,
    )

    reason = verdict.reasons[_C6_LINE]
    assert "PASS" in reason
    assert "FAIL" not in reason
    assert "NOT_EVALUATED" not in reason


def test_criterion6_fails_at_the_same_realistic_trial_count_with_weak_signal() -> None:
    """Spec item 4's FAIL half, at the same realistic trial count used by the
    PASS test above -- isolates that PASS/FAIL is driven by the strategy's own
    signal, not merely by whether n_trials happens to be small."""
    strategy_returns = np.random.default_rng(23).normal(-0.001, 0.001, size=500)

    verdict = _call_verdict(
        _dense_all_pass_panel(),
        strategy_returns=strategy_returns,
        effective_n_trials=10,
    )

    reason = verdict.reasons[_C6_LINE]
    assert "FAIL" in reason
    assert "PASS" not in reason
    assert "NOT_EVALUATED" not in reason


def test_criterion6_same_signal_flips_pass_to_fail_as_trial_count_rises() -> None:
    """Spec item 4: a single strategy_returns series that PASSes at a small
    declared trial count must FAIL once the trial count is raised far enough --
    isolating effective_n_trials as the knob, holding the signal fixed. Raising
    n_trials raises expected_max_sharpe (the sr0 fed into deflated_sharpe), which
    lowers the resulting probability; this is the deflation actually working,
    not the units bug (defect 2), which the two tests above isolate
    separately."""
    strategy_returns = np.random.default_rng(42).normal(0.003, 0.001, size=300)

    pass_verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=2
    )
    fail_verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=100_000
    )

    assert "PASS" in pass_verdict.reasons[_C6_LINE]
    assert "FAIL" in fail_verdict.reasons[_C6_LINE]


@pytest.mark.parametrize(
    "strategy_returns",
    [
        np.array([], dtype=np.float64),
        np.full(10, np.nan, dtype=np.float64),
        np.array([0.01, np.nan, np.nan], dtype=np.float64),
    ],
    ids=["empty", "all_nan", "single_finite_value"],
)
def test_criterion6_degenerate_strategy_returns_are_not_evaluated_never_crash(
    strategy_returns: np.ndarray,
) -> None:
    """Spec item 5: empty / all-NaN / fewer than 2 finite values -> NOT_EVALUATED,
    never a crash and never a bare 0.0 masquerading as a real result. The call
    itself not raising is part of what this test checks."""
    verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=2
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason


def test_criterion6_three_finite_values_still_not_evaluated() -> None:
    """Spec: 'Partially-NaN strategy_returns: drop the non-finite entries and
    evaluate on the remainder if >= 4 finite values survive; otherwise
    NOT_EVALUATED.' Three finite values is below deflated_sharpe's own t<4
    floor (it needs reliable skew/kurtosis) -- distinct from item 5's
    single-finite-value corner, this exercises the exact >= 4 boundary named in
    the spec, not merely 'more than one'."""
    strategy_returns = np.array([0.01, 0.02, -0.01, np.nan, np.nan], dtype=np.float64)

    verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=2
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason


def test_criterion6_four_or_more_finite_values_with_nans_mixed_in_is_evaluated() -> None:
    """Mirror of the test above at the other side of the >= 4 finite boundary:
    once >= 4 finite values survive, criterion 6 must actually evaluate them
    (PASS/FAIL), dropping the NaNs rather than propagating them into
    NOT_EVALUATED or a crash."""
    rng = np.random.default_rng(5)
    strategy_returns = rng.normal(0.02, 0.001, size=500)
    strategy_returns[::50] = np.nan  # scatter NaNs through the series

    verdict = _call_verdict(
        _dense_all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=2
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "PASS" in reason or "FAIL" in reason


def test_criterion6_regression_guard_unconditional_drift_must_not_pass() -> None:
    """Spec item 6, the most important test in this suite for the FIRST defect: a
    panel whose UNCONDITIONAL forward returns (fwd.values) drift strongly
    positive, paired with a flat/negative strategy_returns series, must NOT
    report criterion 6 as PASS.

    This is exactly the defect: the old implementation deflates
    fwd.values.flatten() -- every symbol's raw forward return, not the
    strategy's P&L -- so a market that merely drifted up gets reported as
    "multiple testing was accounted for". This test must fail against the old
    implementation (it cannot even run there: strategy_returns is not a
    parameter it recognizes)."""
    panel = _positive_drift_panel(n_days=40, bars_per_session=30, drift=0.02, seed=0)

    # Sanity: the panel really does drift strongly positive -- this is exactly
    # the quantity the OLD implementation deflated.
    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    assert np.nanmean(fwd.values) > 0.01

    flat_negative_returns = np.random.default_rng(7).normal(-0.001, 0.0005, size=250)

    verdict = _call_verdict(
        panel,
        latency_profile=None,
        strategy_returns=flat_negative_returns,
        effective_n_trials=2,
    )

    reason = verdict.reasons[_C6_LINE]
    assert "PASS" not in reason
    assert "FAIL" in reason


# ---------------------------------------------------------------------------
# Criterion 6 -- any_not_evaluated flag and explain() INCOMPLETE statement
# ---------------------------------------------------------------------------


def test_any_not_evaluated_true_and_explain_states_incomplete_when_c6_unevaluated() -> None:
    """Spec: 'HypothesisVerdict must expose a boolean (e.g. any_not_evaluated)
    that is True when any criterion is NOT_EVALUATED, and explain() must state
    prominently that the verdict is INCOMPLETE -- not merely list NOT_EVALUATED
    among seven reason lines where a reader skims past it.' Reuses the same
    every-other-criterion-passes fixture as the survived-is-True test above, so
    this checks the flag on a verdict that is genuinely SURVIVED yet
    incomplete -- the exact case the spec calls out as uncomfortable."""
    verdict = _call_verdict(
        _dense_all_pass_panel(),
        latency_profile=_PASS_LATENCY,
        strategy_returns=None,
        effective_n_trials=1,
    )

    assert verdict.survived is True
    assert verdict.any_not_evaluated is True
    assert "INCOMPLETE" in verdict.explain()


def test_any_not_evaluated_false_when_all_seven_criteria_are_evaluated() -> None:
    """Corollary: once strategy_returns and enough complete years are supplied so
    every one of the seven criteria is actually evaluated (PASS or FAIL, never
    NOT_EVALUATED), the flag must be False and explain() must not carry the
    INCOMPLETE warning -- the flag is not simply True whenever the verdict
    exists, it tracks the real presence of an unevaluated criterion."""
    strategy_returns = np.random.default_rng(11).normal(0.02, 0.001, size=500)

    verdict = _call_verdict(
        _dense_all_pass_panel(),
        latency_profile=_PASS_LATENCY,
        strategy_returns=strategy_returns,
        effective_n_trials=10,
    )

    assert all("NOT_EVALUATED" not in reason for reason in verdict.reasons)
    assert verdict.any_not_evaluated is False
    assert "INCOMPLETE" not in verdict.explain()


# ---------------------------------------------------------------------------
# Criterion 7 -- recent-years cost gate, threshold-free completeness rule
# ---------------------------------------------------------------------------


def test_criterion7_excludes_partial_year_at_window_end_and_names_it() -> None:
    """Spec item 7: a window ending mid-year (2025-07-31, mirroring the real
    H2/H3 window described in the spec) excludes 2025 as partial, named in the
    reason. 2023 and 2024 are fully month-dense so they qualify as the two
    complete years actually used."""
    dates = (
        _monthly_dates(2023)
        + _monthly_dates(2024)
        + [dt.date(2025, m, 10) for m in range(1, 7)]
        + [dt.date(2025, 7, 31)]
    )
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2023, 2024, 2025})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2023,2024" in reason
    assert "2025" in reason


def test_criterion7_year_complete_when_window_ends_exactly_on_dec31_boundary() -> None:
    """Spec item 8 + item 11: a window ending EXACTLY 2024-12-31 treats 2024 as
    complete once every one of its 12 months has a session, and the two years
    actually used are named explicitly by year number. Includes a Muhurat-like
    60-bar irregular session on the final day: criterion 7 is about window
    EDGES, so a fixed bars/session cannot be assumed here (CLAUDE.md rule 5)."""
    dates = _monthly_dates(2023) + _monthly_dates(2024, months=range(1, 12)) + [
        dt.date(2024, 12, 31)
    ]
    bars_per_session = [20] * (len(dates) - 1) + [60]  # last session is Muhurat-like
    panel = _dated_panel(dates, bars_per_session, signal_years={2023, 2024})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2023,2024" in reason
    assert "excluded partial years: none" in reason


def test_criterion7_2023_is_complete_when_window_ends_2023_12_31() -> None:
    """Spec's own worked example, verified against the real NSE session calendar:
    'a window ending 2023-12-31 has panel_last_date = 2023-12-29 (the last NSE
    session), so 2023 would be marked PARTIAL despite being fully covered' under
    the REJECTED date-span rule -- and 'Verified ... 2022 and 2023 -> complete'
    under the amended 12-month rule. This constructs that exact boundary
    directly (2022 and 2023 both fully month-dense, panel ending 2023-12-31) and
    asserts 2023 is used as COMPLETE, not silently dropped as the old,
    superseded date-span rule would have done."""
    dates = _monthly_dates(2022) + _monthly_dates(2023, months=range(1, 12)) + [
        dt.date(2023, 12, 31)
    ]
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2022, 2023})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2022,2023" in reason
    assert "excluded partial years: none" in reason


def test_criterion7_excludes_partial_year_at_window_start_symmetric_rule() -> None:
    """Spec item 9: the completeness rule is symmetric -- a window STARTING
    mid-year (2018-06-01) excludes 2018 as partial too, not only a mid-year end
    (item 7)."""
    dates = (
        [dt.date(2018, 6, 1)]
        + [dt.date(2018, m, 10) for m in range(7, 13)]
        + _monthly_dates(2019)
        + _monthly_dates(2020)
    )
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2018, 2019, 2020})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2019,2020" in reason
    assert "2018" in reason


def test_criterion7_zero_complete_years_is_not_evaluated() -> None:
    """Spec item 10: with zero complete years, criterion 7 must report
    NOT_EVALUATED -- never a silent PASS. Both years here have sessions in only
    two of twelve months each."""
    dates = [
        dt.date(2023, 3, 1),
        dt.date(2023, 6, 15),
        dt.date(2024, 3, 1),
        dt.date(2024, 6, 30),
    ]
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2023, 2024})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason


def test_criterion7_one_complete_year_is_not_evaluated() -> None:
    """Spec item 10's other corner: exactly one complete year (2023, all 12
    months) plus one partial year (2024, one session) is still fewer than two
    complete years -> NOT_EVALUATED, never a silent PASS with a single year."""
    dates = _monthly_dates(2023) + [dt.date(2024, 6, 15)]
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2023, 2024})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason


def test_criterion7_uses_the_two_most_recent_complete_years_explicitly() -> None:
    """Spec item 11: with three complete years available (2019, 2020, 2021), the
    two years actually USED are the two most recent (2020, 2021) -- asserted by
    explicit year number, an earlier complete year must not silently sneak in
    just because it exists."""
    dates = _monthly_dates(2019) + _monthly_dates(2020) + _monthly_dates(2021)
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2019, 2020, 2021})
    verdict = _call_verdict(panel, latency_profile=None)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2020,2021" in reason
    assert "2019" not in reason


# ---------------------------------------------------------------------------
# Both criteria: all seven always reported, and determinism
# ---------------------------------------------------------------------------


def test_verdict_reports_all_seven_criteria_in_order_after_a_failure() -> None:
    """Spec item 12: all seven criteria are always reported, in numbered order,
    even once one (or several) of them fail."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    strategy_returns = np.random.default_rng(3).normal(0.001, 0.0005, size=50)

    verdict = _call_verdict(
        panel,
        latency_profile=_FAIL_LATENCY,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    assert len(verdict.reasons) == 7
    for i, reason in enumerate(verdict.reasons, start=1):
        assert reason.startswith(f"{i}."), reason
    # Not vacuous: at least one criterion genuinely failed or was skipped here.
    assert any("FAIL" in r or "NOT_EVALUATED" in r for r in verdict.reasons)


def test_verdict_determinism_same_inputs_and_seed_identical_text() -> None:
    """Spec item 13: same inputs and seed -> identical verdict text, for both the
    raw reasons and the rendered explain()/to_markdown() views."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=2,
        bars_per_session=25,
        signal_years={2024},
        effect=0.01,
        sigma=0.0005,
    )
    strategy_returns = np.random.default_rng(3).normal(0.001, 0.0005, size=50)

    lens = Lens(panel, seed=11)
    kwargs: dict[str, object] = dict(
        latency_profile=_FAIL_LATENCY,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )
    verdict_a = lens.verdict("H_DET", "return_1", 1, **kwargs)
    verdict_b = lens.verdict("H_DET", "return_1", 1, **kwargs)

    assert verdict_a.reasons == verdict_b.reasons
    assert verdict_a.explain() == verdict_b.explain()
    assert verdict_a.to_markdown() == verdict_b.to_markdown()
    assert verdict_a.survived == verdict_b.survived
    assert verdict_a.any_not_evaluated == verdict_b.any_not_evaluated
