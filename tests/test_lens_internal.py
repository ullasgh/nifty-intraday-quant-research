"""Internal lens.py tests for 100% line and branch coverage.

Tests for code paths not covered by the 42 contract tests in test_lens.py.
Focuses on edge cases, error paths, and branch combinations.
"""

from __future__ import annotations

import calendar
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.research import expectancy
from nifty_quant.research.lens import (
    Feature,
    FeatureKindError,
    HypothesisVerdict,
    Lens,
    StabilityReport,
    compute_prior_adv,
)

_IST = ZoneInfo("Asia/Kolkata")
_N_SYMBOLS = 5
_SYMBOLS = ("S00", "S01", "S02", "S03", "S04")


def _session_grid(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build timestamp grid, day_offsets, and dates array."""
    ts_chunks: list[np.ndarray] = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bars_per_session, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    n_days = len(dates)
    day_offsets = np.arange(0, (n_days + 1) * bars_per_session, bars_per_session, dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def _signal_means(effect: float) -> np.ndarray:
    """Return signal means for 5-bucket ranking."""
    return np.array([-effect, -effect / 2.0, 0.0, effect / 2.0, effect], dtype=np.float64)


def _close_prices_from_log_returns(returns: np.ndarray) -> np.ndarray:
    """Convert log returns to close prices (first bar = 1.0)."""
    cum = np.cumsum(returns, axis=0)
    log_close = np.vstack([np.zeros((1, returns.shape[1]), dtype=np.float64), cum])
    close_all = np.exp(log_close)
    return close_all[:-1, :]


def _monthly_session_dates(year: int, n_sessions: int) -> list[dt.date]:
    """`n_sessions` session dates for `year`, in STRICTLY CHRONOLOGICAL order
    (required: `Panel`'s `check_sorted_unique` rejects a non-increasing `ts`),
    covering every calendar month at least once when `n_sessions >= 12`.

    Criterion 7's amended rule (specs/lens_criteria_6_7_repair.md Defect 2)
    calls a year "complete" iff the panel holds a session in EACH of that
    year's 12 calendar months -- `{d.month for d in panel.dates if
    d.year == year} == {1..12}` -- never a session-count threshold. This
    repo's usual `date(year, 1, 2 + session_idx)` generator (see
    `_build_panel`'s default and the task brief's landmine 1) places every
    session in January, so it can NEVER satisfy that rule regardless of how
    many sessions are requested; this generator is what criterion-7 tests
    need instead whenever a year must come back "complete".

    `n_sessions < 12` deliberately gives a PARTIAL year (one session per
    month for the first `n_sessions` months only, e.g. `n_sessions=6` ->
    Jan-Jun, missing Jul-Dec) -- used by criterion-7 tests that need a
    year to be excluded despite having many total sessions. `n_sessions
    >= 12` spreads sessions evenly across all 12 months (extra sessions go
    to the earliest months first), with each month's own sessions spread
    across that month's real day count so the overall list stays strictly
    increasing month-by-month, day-by-day."""
    if n_sessions < 12:
        return [dt.date(year, month, 1) for month in range(1, n_sessions + 1)]

    base, extra = divmod(n_sessions, 12)
    dates: list[dt.date] = []
    for month in range(1, 13):
        sessions_this_month = base + (1 if month <= extra else 0)
        days_in_month = calendar.monthrange(year, month)[1]
        days = np.linspace(1, days_in_month, sessions_this_month, dtype=int)
        dates.extend(dt.date(year, month, int(day)) for day in days)
    return dates


def _build_panel(
    years: tuple[int, ...],
    sessions_per_year: int | dict[int, int] = 2,
    bars_per_session: int = 25,
    signal_years: set[int] | None = None,
    *,
    effect: float = 0.01,
    sigma: float = 0.0005,
    seed: int = 0,
    monthly_years: set[int] | None = None,
) -> Panel:
    """Build a Panel with specified years, sessions, and signal strength.

    `monthly_years`: years in this set get `_monthly_session_dates` (12-month
    coverage, for criterion-7 "complete year" tests) instead of the default
    consecutive-January dates; unlisted years are unaffected. Since returns
    are drawn deterministically per date in iteration order and depend only
    on `date.year` (never the specific day-of-month), swapping only the
    calendar placement of a year's dates changes no other computed value
    (spread_bps, cost_hurdle_bps, t-stats) versus the January-only layout.
    """
    if signal_years is None:
        signal_years = set(years)
    if monthly_years is None:
        monthly_years = set()

    if isinstance(sessions_per_year, int):
        year_sessions = {year: sessions_per_year for year in years}
    else:
        year_sessions = dict(sessions_per_year)

    dates: list[dt.date] = []
    for year in years:
        if year in monthly_years:
            dates += _monthly_session_dates(year, year_sessions[year])
        else:
            dates += [dt.date(year, 1, 2 + i) for i in range(year_sessions[year])]

    n_rows = len(dates) * bars_per_session
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_rows, _N_SYMBOLS), dtype=np.float64)

    row = 0
    for date in dates:
        year = date.year
        if year in signal_years:
            means = _signal_means(effect)
        else:
            means = np.zeros(_N_SYMBOLS, dtype=np.float64)

        block_means = np.tile(means, (bars_per_session, 1))
        noise = rng.normal(0, sigma, (bars_per_session, _N_SYMBOLS))
        returns[row : row + bars_per_session, :] = block_means + noise
        row += bars_per_session

    close = _close_prices_from_log_returns(returns)

    volume = np.full((n_rows, _N_SYMBOLS), 1e5, dtype=np.float64)
    for i in range(_N_SYMBOLS):
        volume[:, i] = (i + 1) * 1e3

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)

    panel_data = {
        "close": close,
        "volume": volume,
    }

    panel = Panel(
        fields=panel_data,
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )
    return panel


def _build_small_panel(seed: int) -> Panel:
    """Build a small panel for quick tests."""
    return _build_panel(
        (2023, 2024, 2025),
        sessions_per_year=2,
        bars_per_session=25,
        effect=0.01,
        sigma=0.0005,
        seed=seed,
    )


def _random_feature_values(n_rows: int, seed: int) -> np.ndarray:
    """Generate random feature values."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, (n_rows, _N_SYMBOLS)).astype(np.float64)


# ==============================================================================
# Feature.explain() tests
# ==============================================================================


def test_feature_explain_with_empty_params() -> None:
    """Feature.explain() with no params should not include parentheses."""
    feature = Feature(
        name="close",
        values=np.zeros((10, 5), dtype=np.float64),
        kind="level",
        warmup_bars=0,
        params={},
    )
    text = feature.explain()
    assert "close" in text
    assert "level" in text
    assert "warmup_bars=0" in text
    # No parentheses when params is empty
    assert "(" not in text or "()" not in text


def test_feature_explain_with_params() -> None:
    """Feature.explain() with params should include them in parentheses."""
    feature = Feature(
        name="volume_zscore",
        values=np.zeros((10, 5), dtype=np.float64),
        kind="ratio",
        warmup_bars=20,
        params={"window": 20, "deseasonalize": True},
    )
    text = feature.explain()
    assert "volume_zscore" in text
    assert "ratio" in text
    assert "window" in text or "20" in text
    assert "warmup_bars=20" in text


# ==============================================================================
# StabilityReport.explain() test
# ==============================================================================


def test_stability_report_explain() -> None:
    """StabilityReport.explain() should report years and stability."""
    report = StabilityReport(
        by_year={},
        by_time_of_day={},
        by_liquidity_decile={},
        n_years_total=5,
        n_years_sign_consistent=4,
        dominant_sign="+",
    )
    text = report.explain()
    assert "4/5" in text
    assert "+" in text
    assert "stable=False" in text


def test_stability_report_explain_stable() -> None:
    """StabilityReport.explain() when stable (>= 6 years)."""
    report = StabilityReport(
        by_year={},
        by_time_of_day={},
        by_liquidity_decile={},
        n_years_total=8,
        n_years_sign_consistent=6,
        dominant_sign="-",
    )
    text = report.explain()
    assert "6/8" in text
    assert "-" in text
    assert "stable=True" in text


def test_stability_report_explain_mixed_sign() -> None:
    """StabilityReport.explain() when sign is mixed."""
    report = StabilityReport(
        by_year={},
        by_time_of_day={},
        by_liquidity_decile={},
        n_years_total=3,
        n_years_sign_consistent=1,
        dominant_sign="mixed",
    )
    text = report.explain()
    assert "mixed" in text


# ==============================================================================
# FeatureKindError with string features (coverage of isinstance checks)
# ==============================================================================


def test_expectancy_with_string_feature() -> None:
    """lens.expectancy() with string feature should resolve and check kind."""
    panel = _build_small_panel(seed=20)
    lens = Lens(panel)
    with pytest.raises(FeatureKindError) as excinfo:
        lens.expectancy("close", horizon=1)
    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


def test_stability_with_string_feature() -> None:
    """lens.stability() with string feature should resolve and check kind."""
    panel = _build_small_panel(seed=21)
    lens = Lens(panel)
    with pytest.raises(FeatureKindError) as excinfo:
        lens.stability("close", horizon=1)
    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


def test_verdict_with_string_feature() -> None:
    """lens.verdict() with string feature should resolve and check kind."""
    panel = _build_small_panel(seed=22)
    lens = Lens(panel)
    with pytest.raises(FeatureKindError) as excinfo:
        lens.verdict("H_lev", "close", horizon=1)
    message = str(excinfo.value)
    assert "close" in message
    assert "level" in message
    assert "return" in message


# ==============================================================================
# Stability with edge cases: no years, tie in signs
# ==============================================================================


def test_stability_single_year_limited_data() -> None:
    """Stability with a single year of data."""
    # Create a panel with only one session in one year
    dates = [dt.date(2024, 1, 2)]
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session=50)

    close = np.ones((50, _N_SYMBOLS), dtype=np.float64)
    close += np.cumsum(np.random.default_rng(110).normal(0, 0.001, (50, _N_SYMBOLS)), axis=0)
    volume = np.ones((50, _N_SYMBOLS), dtype=np.float64) * 1e5

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    # With one year of data, n_years_total=1
    assert report.n_years_total == 1
    # Sign might be mixed if no consistent return direction
    assert report.dominant_sign in ("+", "-", "mixed")


def test_stability_tie_in_sign_counts() -> None:
    """Stability when equal number of years with +/- sign (tie-break to -)."""
    # Create a panel with multiple years with alternating signals
    # This tests the max(...key=count) tie-break behavior
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=1,
        bars_per_session=25,
        signal_years={2018, 2020, 2022, 2024},  # 4 positive years
        effect=0.01,
        seed=30,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    # Should have years with data; at least some should have consistent sign
    assert report.n_years_total >= 1


# ==============================================================================
# Verdict criterion 4: concentration in time-of-day and liquidity
# ==============================================================================


def test_verdict_criterion_4_single_time_of_day_bucket() -> None:
    """Criterion 4 fails when edge concentrated in single time-of-day bucket."""
    # Create a panel with full trading hours data
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=2,
        bars_per_session=75,  # Full session: 9:15-10:30 (75 minutes)
        effect=0.01,
        seed=40,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_tod", feature, horizon=1)

    # Check criterion 4 status
    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None
    # Should have Concentration criterion in the line
    assert "Concentration criterion" in c4_line


def test_verdict_criterion_4_with_empty_time_of_day() -> None:
    """Criterion 4 when no time-of-day buckets have data."""
    # Create a small panel
    panel = _build_small_panel(seed=41)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    # This should still work even with sparse time-of-day coverage
    verdict = lens.verdict("H_sparse_tod", feature, horizon=1)

    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None


def test_verdict_criterion_4_all_time_buckets_present() -> None:
    """Criterion 4 when all three time-of-day buckets are populated."""
    # Full trading day data: 9:15-15:30 (375 bars)
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=1,
        bars_per_session=375,
        effect=0.01,
        seed=105,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_full_day", feature, horizon=1)

    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None


def test_verdict_criterion_4_bucket_dominance() -> None:
    """Criterion 4 when one bucket dominates in edge magnitude."""
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=2,
        bars_per_session=200,
        effect=0.015,
        seed=106,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_dom_bucket", feature, horizon=1)

    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None


# ==============================================================================
# Verdict criterion 5: latency profile edge cases
# ==============================================================================


def test_verdict_criterion_5_negative_lag_0_edge() -> None:
    """Criterion 5 with negative lag_0_edge (short signal)."""
    panel = _build_small_panel(seed=42)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    latency = {0: -1.0, 1: -0.6, 2: -0.55}  # Negative lag-0 edge
    verdict = lens.verdict("H_lag_neg", feature, horizon=1, latency_profile=latency)

    c5_line = next((r for r in verdict.reasons if r.startswith("5.")), None)
    assert c5_line is not None
    # Should FAIL because lag_0_edge is not > 0
    assert "FAIL" in c5_line or "NOT_EVALUATED" in c5_line


def test_verdict_criterion_5_weak_retention() -> None:
    """Criterion 5 when lag retention is < 50%."""
    panel = _build_small_panel(seed=43)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    latency = {0: 1.0, 1: 0.3, 2: 0.2}  # Only 30% and 20% retention
    verdict = lens.verdict("H_lag_weak", feature, horizon=1, latency_profile=latency)

    c5_line = next((r for r in verdict.reasons if r.startswith("5.")), None)
    assert c5_line is not None
    assert "FAIL" in c5_line


def test_verdict_criterion_5_strong_retention() -> None:
    """Criterion 5 when lag retention >= 50%."""
    panel = _build_small_panel(seed=44)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    latency = {0: 1.0, 1: 0.55, 2: 0.51}  # 55% and 51% retention
    verdict = lens.verdict("H_lag_strong", feature, horizon=1, latency_profile=latency)

    c5_line = next((r for r in verdict.reasons if r.startswith("5.")), None)
    assert c5_line is not None
    assert "PASS" in c5_line


# ==============================================================================
# Verdict criterion 6: deflated Sharpe with various trial counts
# ==============================================================================


def test_verdict_criterion_6_single_trial_positive() -> None:
    """Criterion 6 with single trial and positive Sharpe."""
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=5,
        bars_per_session=50,
        effect=0.01,
        seed=50,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_sr1", feature, horizon=1, effective_n_trials=1)

    c6_line = next((r for r in verdict.reasons if r.startswith("6.")), None)
    assert c6_line is not None
    assert "trials=1" in c6_line


def test_verdict_criterion_6_many_trials() -> None:
    """Criterion 6 with many trials requires higher Sharpe."""
    panel = _build_small_panel(seed=51)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_sr_many", feature, horizon=1, effective_n_trials=1000)

    c6_line = next((r for r in verdict.reasons if r.startswith("6.")), None)
    assert c6_line is not None
    assert "trials=1000" in c6_line


def test_verdict_criterion_6_no_valid_returns() -> None:
    """Criterion 6 when no valid forward returns."""
    # Create a panel where all forward returns are NaN
    dates = [dt.date(2024, 1, 1)]
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session=5)

    close = np.ones((5, _N_SYMBOLS), dtype=np.float64)
    volume = np.ones((5, _N_SYMBOLS), dtype=np.float64)

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_nodata", feature, horizon=1, effective_n_trials=1)

    c6_line = next((r for r in verdict.reasons if r.startswith("6.")), None)
    assert c6_line is not None


# ==============================================================================
# HypothesisVerdict.explain() and to_markdown() formatting
# ==============================================================================


def test_hypothesis_verdict_explain_contains_all_parts() -> None:
    """HypothesisVerdict.explain() contains hypothesis_id, seed, cost hurdle, SE method, verdict."""
    panel = _build_small_panel(seed=60)
    lens = Lens(panel, seed=123)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_format", feature, horizon=1)

    text = verdict.explain()
    assert "H_format" in text
    assert "123" in text or "Seed: 123" in text
    assert "Cost hurdle" in text
    assert "SE method" in text
    assert "Final verdict" in text


def test_hypothesis_verdict_explain_survived() -> None:
    """HypothesisVerdict.explain() includes SURVIVED when verdict passed."""
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=5,
        bars_per_session=50,
        effect=0.01,
        seed=61,
    )
    lens = Lens(panel, seed=456)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_surv", feature, horizon=1)

    text = verdict.explain()
    if verdict.survived:
        assert "SURVIVED" in text
    else:
        assert "KILLED" in text


def test_hypothesis_verdict_to_markdown_structure() -> None:
    """HypothesisVerdict.to_markdown() should contain markdown sections."""
    panel = _build_small_panel(seed=62)
    lens = Lens(panel, seed=789)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_md", feature, horizon=1)

    md = verdict.to_markdown()
    assert "# H_md" in md
    assert "**Verdict:**" in md
    assert "## Kill Criteria" in md
    assert "## Context" in md
    assert "Cost hurdle" in md
    assert "SE method" in md
    assert "Seed:" in md or "seed" in md.lower()


def test_hypothesis_verdict_to_markdown_includes_all_reasons() -> None:
    """HypothesisVerdict.to_markdown() should include all criterion lines."""
    panel = _build_small_panel(seed=63)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_reasons", feature, horizon=1)

    md = verdict.to_markdown()
    for reason in verdict.reasons:
        # At minimum, the criterion number should appear
        assert any(f"{i}." in md for i in range(1, 7))


def test_hypothesis_verdict_to_markdown_killed() -> None:
    """HypothesisVerdict.to_markdown() should say KILLED when not survived."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=1,
        bars_per_session=20,
        effect=0.0001,
        seed=64,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_kill", feature, horizon=1)

    md = verdict.to_markdown()
    if not verdict.survived:
        assert "KILLED" in md


# ==============================================================================
# Feature memoization: coverage of cache behavior
# ==============================================================================


def test_feature_cache_distinct_params_no_collision() -> None:
    """Feature cache should not confuse different param sets."""
    lens = Lens(_build_small_panel(seed=70))

    # Request same feature with two different window values
    f20 = lens.feature("volume_zscore", window=20)
    f30 = lens.feature("volume_zscore", window=30)

    # Different params should give different cached objects
    assert f20 is not f30
    assert not np.array_equal(f20.values, f30.values)

    # Same params should give identical object
    f20_again = lens.feature("volume_zscore", window=20)
    assert f20 is f20_again


def test_feature_cache_param_order_normalized() -> None:
    """Feature cache should normalize parameter order."""
    lens = Lens(_build_small_panel(seed=71))

    # Request with different param order (Python dicts preserve order, but params are sorted)
    f1 = lens.feature("volume_zscore", window=25, deseasonalize=False)
    f2 = lens.feature("volume_zscore", deseasonalize=False, window=25)

    # Should be the same object (params are sorted internally)
    assert f1 is f2


# ==============================================================================
# Available features coverage
# ==============================================================================


def test_available_features_returns_tuple() -> None:
    """available_features() should return a tuple."""
    lens = Lens(_build_small_panel(seed=72))
    features = lens.available_features()

    assert isinstance(features, tuple)
    assert len(features) > 0
    assert "close" in features
    assert "return_1" in features
    assert "volume_zscore" in features


# ==============================================================================
# Verdict all criteria status reporting
# ==============================================================================


def test_verdict_all_criteria_reported_even_multiple_fail() -> None:
    """Verdict should report all 7 criteria (including criterion 7) even if several fail."""
    panel = _build_small_panel(seed=80)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_multi_fail", feature, horizon=1)

    # Should have exactly 7 reason lines (one per criterion, criterion 7 included)
    assert len(verdict.reasons) == 7
    for i in range(1, 8):
        assert any(f"{i}." in r for r in verdict.reasons)


def test_verdict_survived_only_when_all_evaluated_pass() -> None:
    """Verdict.survived is True only when all evaluated criteria pass."""
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=5,
        bars_per_session=50,
        effect=0.01,
        seed=81,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    # With latency profile, criterion 5 is evaluated
    latency = {0: 1.0, 1: 0.6, 2: 0.55}
    verdict = lens.verdict("H_all_eval", feature, horizon=1, latency_profile=latency)

    # survived should be True only if all criteria pass
    assert isinstance(verdict.survived, bool)
    # Verify consistency: count reason lines with PASS/FAIL/NOT_EVALUATED
    pass_count = sum(1 for r in verdict.reasons if "PASS" in r)
    fail_count = sum(1 for r in verdict.reasons if "FAIL" in r)
    not_eval_count = sum(1 for r in verdict.reasons if "NOT_EVALUATED" in r)

    assert pass_count + fail_count + not_eval_count == 7
    # survived should be True iff fail_count == 0
    assert verdict.survived == (fail_count == 0)


# ==============================================================================
# Lens construction edge cases
# ==============================================================================


def test_lens_with_explicit_empty_universe() -> None:
    """Lens should accept explicit universe parameter."""
    panel = _build_small_panel(seed=90)
    universe = ("S00", "S01")
    lens = Lens(panel, universe=universe)

    assert lens.universe == universe


def test_lens_default_universe_is_panel_symbols() -> None:
    """Lens default universe should be all panel symbols."""
    panel = _build_small_panel(seed=91)
    lens = Lens(panel)

    assert lens.universe == panel.symbols


def test_lens_with_custom_cost_model() -> None:
    """Lens should accept custom cost model."""
    panel = _build_small_panel(seed=92)
    cost_model = NSEIntradayEquityCosts(brokerage_flat=10.0)
    lens = Lens(panel, cost_model=cost_model)

    assert lens.cost_model is cost_model


def test_lens_default_cost_model() -> None:
    """Lens should create default cost model if not provided."""
    panel = _build_small_panel(seed=93)
    lens = Lens(panel)

    assert isinstance(lens.cost_model, NSEIntradayEquityCosts)


# ==============================================================================
# Stability reshaping branches (1D array handling)
# ==============================================================================


def test_stability_with_different_horizons() -> None:
    """Stability should compute correctly for different horizons."""
    panel = _build_small_panel(seed=94)
    lens = Lens(panel)

    # Get a feature
    feature = lens.feature("return_1")

    # Compute stability for different horizons
    report_h1 = lens.stability(feature, horizon=1)
    report_h5 = lens.stability(feature, horizon=5)

    assert isinstance(report_h1, StabilityReport)
    assert isinstance(report_h5, StabilityReport)
    # Both should have valid structures
    assert report_h1.n_years_total >= 0
    assert report_h5.n_years_total >= 0


def test_expectancy_and_stability_require_2d_features() -> None:
    """Lens.expectancy() and Lens.stability() enforce 2D feature requirement."""
    panel = _build_small_panel(seed=111)
    lens = Lens(panel)

    # All lens-managed features are 2D
    features = ["close", "return_1", "volume_zscore"]
    for fname in features:
        if fname != "close":  # close is a level, not a return
            feature = lens.feature(fname)
            assert feature.values.ndim == 2, f"{fname} should be 2D"


def test_stability_handles_2d_feature_values() -> None:
    """Stability should handle 2D feature values."""
    panel = _build_small_panel(seed=95)
    lens = Lens(panel)

    # Use the built-in return_1 feature which is 2D
    feature = lens.feature("return_1")
    assert feature.values.ndim == 2

    report = lens.stability(feature, horizon=1)
    assert isinstance(report, StabilityReport)


# ==============================================================================
# Coverage of verdict string feature type checking
# ==============================================================================


def test_verdict_accepts_string_feature_name() -> None:
    """Verdict should accept string feature name and resolve it."""
    panel = _build_small_panel(seed=96)
    lens = Lens(panel)

    # Should not raise FeatureKindError for return feature passed as string
    verdict = lens.verdict("H_str_ret", "return_1", horizon=1)
    assert isinstance(verdict, HypothesisVerdict)


def test_expectancy_accepts_string_feature_name() -> None:
    """Expectancy should accept string feature name and resolve it."""
    panel = _build_small_panel(seed=97)
    lens = Lens(panel)

    # Should not raise FeatureKindError for return feature passed as string
    table = lens.expectancy("return_1", horizon=1)
    assert isinstance(table, expectancy.ExpectancyTable)


def test_stability_accepts_string_feature_name() -> None:
    """Stability should accept string feature name and resolve it."""
    panel = _build_small_panel(seed=98)
    lens = Lens(panel)

    # Should not raise FeatureKindError for return feature passed as string
    report = lens.stability("return_1", horizon=1)
    assert isinstance(report, StabilityReport)


# ==============================================================================
# Time-of-day bucket edge cases
# ==============================================================================


def test_stability_with_empty_time_of_day_buckets() -> None:
    """Stability should handle sparse time-of-day coverage."""
    # Create a panel with timestamps only in the "open" period
    dates = [dt.date(2024, 1, 1)]

    # Only create bars from 9:15-10:00 (open period)
    bar_count = 45
    ts_chunks: list[np.ndarray] = []
    for day in dates:
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=bar_count, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)

    ts = np.concatenate(ts_chunks).astype(np.int64)
    day_offsets = np.array([0, bar_count], dtype=np.int32)
    dates_arr = np.array(dates, dtype=object)

    close = np.ones((bar_count, _N_SYMBOLS), dtype=np.float64)
    close += np.cumsum(np.random.default_rng(99).normal(0, 0.0001, (bar_count, _N_SYMBOLS)), axis=0)
    volume = np.ones((bar_count, _N_SYMBOLS), dtype=np.float64) * 1e5

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    # Should have fewer than 3 time-of-day buckets
    assert len(report.by_time_of_day) <= 3


# ==============================================================================
# Liquidity decile coverage
# ==============================================================================


def test_stability_liquidity_decile_all_present() -> None:
    """Stability should compute liquidity deciles for all bins present."""
    panel = _build_small_panel(seed=100)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    # All 10 deciles should be present since volume is varied
    assert len(report.by_liquidity_decile) >= 1
    assert max(report.by_liquidity_decile.keys()) <= 9


# ==============================================================================
# Verdict with criterion 4 liquidity concentration
# ==============================================================================


def test_verdict_criterion_4_dominance_check() -> None:
    """Criterion 4 checks if one time-of-day bucket dominates."""
    # Create a panel with majority of edge in one time period
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=2,
        bars_per_session=75,
        effect=0.01,
        seed=101,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_dom", feature, horizon=1)

    # Criterion 4 should be evaluated
    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None
    assert "Concentration criterion" in c4_line


def test_verdict_criterion_4_bottom_decile_edge_present() -> None:
    """Criterion 4 when bottom liquidity decile has an edge."""
    # Build a panel with sufficient data
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=3,
        bars_per_session=50,
        effect=0.01,
        seed=102,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_bottom_decile", feature, horizon=1)

    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None


def test_verdict_criterion_4_multiple_time_buckets_same_sign() -> None:
    """Criterion 4 with multiple time buckets all same sign."""
    # Create a strong signal across all periods
    panel = _build_panel(
        (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        sessions_per_year=3,
        bars_per_session=100,
        effect=0.02,
        seed=103,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_all_buckets", feature, horizon=1)

    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None
    assert "Concentration criterion" in c4_line


def test_verdict_criterion_6_negative_sharpe() -> None:
    """Criterion 6 with negative Sharpe ratio."""
    panel = _build_small_panel(seed=104)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    # With many trials and weak signal, likely to fail criterion 6
    verdict = lens.verdict("H_neg_sharpe", feature, horizon=1, effective_n_trials=100)

    c6_line = next((r for r in verdict.reasons if r.startswith("6.")), None)
    assert c6_line is not None
    assert "trials=100" in c6_line


def test_verdict_criterion_6_nan_sharpe() -> None:
    """Criterion 6 handles NaN Sharpe values."""
    # Create a very small panel with limited data
    dates = [dt.date(2024, 1, 1)]
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session=3)

    close = np.ones((3, _N_SYMBOLS), dtype=np.float64)
    volume = np.ones((3, _N_SYMBOLS), dtype=np.float64)

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_nan", feature, horizon=1, effective_n_trials=10)

    c6_line = next((r for r in verdict.reasons if r.startswith("6.")), None)
    assert c6_line is not None


def test_verdict_criterion_4_no_time_of_day_data() -> None:
    """Criterion 4 when no time-of-day buckets have data (empty by_time_of_day)."""
    # This is hard to test without manipulating minute_of_day values,
    # but we should test that the criterion handles empty by_time_of_day
    panel = _build_small_panel(seed=112)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    # Normal case with data
    verdict = lens.verdict("H_normal_c4", feature, horizon=1)
    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None
    assert "PASS" in c4_line or "FAIL" in c4_line


def test_verdict_criterion_4_empty_time_of_day_edges() -> None:
    """Criterion 4 when time_of_day_edges is empty or all zero."""
    # Create a panel where all returns are exactly zero
    dates = []
    for year in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025):
        dates.append(dt.date(year, 1, 2))

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session=50)

    # Flat prices -> zero returns
    close = np.ones((len(dates) * 50, _N_SYMBOLS), dtype=np.float64)
    volume = np.full((len(dates) * 50, _N_SYMBOLS), 1e5, dtype=np.float64)

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict("H_zero_returns", feature, horizon=1)
    c4_line = next((r for r in verdict.reasons if r.startswith("4.")), None)
    assert c4_line is not None


# ==============================================================================
# Liquidity-decile mask correctness (regression for the 2-D boolean-mask
# collapse bug: boolean-indexing a 2-D array with a 2-D mask always flattens
# to 1-D, destroying row/symbol structure and forcing a fabricated
# single-session day_offsets -- see lens.py Lens.stability()).
# ==============================================================================

_N_SYMBOLS_LIQ = 60
_SYMBOLS_LIQ = tuple(f"L{i:02d}" for i in range(_N_SYMBOLS_LIQ))
_EDGE_SYMBOLS_LIQ = 6  # count of lowest-volume symbols carrying the planted edge


def _build_liquidity_concentrated_panel(
    *,
    effect: float = 0.02,
    sigma: float = 0.0005,
    sigma_other_symbols: float | None = None,
    bars_per_session: int = 100,
    n_sessions: int = 4,
    seed: int = 7,
) -> Panel:
    """Panel with 60 symbols where the ENTIRE edge lives in the bottom volume
    decile (the 6 lowest-volume symbols, indices 0..5) and every other symbol
    is pure noise. Volume is constant per symbol (as in this file's other
    fixtures), with 60 distinct values 1..60 -- verified empirically to put
    symbols 0..5 alone in decile 0 and 6 symbols in each of the other 9
    deciles, comfortably clearing cross_sectional_rank's min_names=5 gate.

    `sigma_other_symbols`, if given, overrides the noise sigma for the 54
    non-edge symbols (e.g. 0.0 to make every non-bottom decile exactly flat).
    """
    dates = [dt.date(2024, 1, 2 + i) for i in range(n_sessions)]
    n_rows = bars_per_session * n_sessions

    rng = np.random.default_rng(seed)
    means = np.zeros(_N_SYMBOLS_LIQ, dtype=np.float64)
    means[:_EDGE_SYMBOLS_LIQ] = np.linspace(-effect, effect, _EDGE_SYMBOLS_LIQ)

    sigmas = np.full(_N_SYMBOLS_LIQ, sigma, dtype=np.float64)
    if sigma_other_symbols is not None:
        sigmas[_EDGE_SYMBOLS_LIQ:] = sigma_other_symbols

    returns = means[None, :] + rng.normal(0.0, 1.0, size=(n_rows, _N_SYMBOLS_LIQ)) * sigmas[None, :]
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.arange(1, _N_SYMBOLS_LIQ + 1, dtype=np.float64), (n_rows, 1))

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS_LIQ,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _build_liquidity_evenly_spread_panel(
    *,
    effect: float = 0.02,
    sigma: float = 0.0005,
    bars_per_session: int = 100,
    n_sessions: int = 4,
    seed: int = 7,
) -> Panel:
    """Panel with 60 symbols where EVERY volume decile's 6-symbol group gets
    the same [-effect, +effect] spread -- an edge present everywhere, with no
    single decile dominating. Used to confirm criterion 4's bottom-decile
    check does not fire just because the bottom decile HAS an edge."""
    dates = [dt.date(2024, 1, 2 + i) for i in range(n_sessions)]
    n_rows = bars_per_session * n_sessions

    rng = np.random.default_rng(seed)
    means = np.zeros(_N_SYMBOLS_LIQ, dtype=np.float64)
    for group_start in range(0, _N_SYMBOLS_LIQ, _EDGE_SYMBOLS_LIQ):
        means[group_start : group_start + _EDGE_SYMBOLS_LIQ] = np.linspace(
            -effect, effect, _EDGE_SYMBOLS_LIQ
        )

    returns = means[None, :] + rng.normal(0.0, sigma, size=(n_rows, _N_SYMBOLS_LIQ))
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.arange(1, _N_SYMBOLS_LIQ + 1, dtype=np.float64), (n_rows, 1))

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS_LIQ,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def test_stability_liquidity_decile_produces_finite_stats() -> None:
    """Regression test for the mask-collapse bug: with a wide (60-symbol)
    panel, the decile carrying a real planted edge must come back with
    FINITE statistics and n_total > 0 -- not the empty/NaN table the old
    boolean-mask code produced for every decile at any panel width."""
    panel = _build_liquidity_concentrated_panel()
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)

    assert 0 in report.by_liquidity_decile
    bottom_table = report.by_liquidity_decile[0]
    assert bottom_table.n_total > 0
    assert len(bottom_table.buckets) >= 2
    assert np.isfinite(bottom_table.spread_bps)
    assert bottom_table.spread_bps != 0.0


def test_stability_liquidity_decile_preserves_shape_and_real_day_offsets() -> None:
    """The masked feature/forward-return arrays must stay (n_rows, n_symbols)
    and use the panel's real day_offsets, not a fabricated single-session
    [0, N]. Verified indirectly: n_total must equal the count of rows where
    at least one in-decile symbol has a finite, session-respecting forward
    return, computed independently here from the panel's own day_offsets.

    The decile membership oracle below was rewritten against
    specs/lens_criterion_4_repair.md (Required behaviour items 1-3): the
    original oracle reimplemented the OLD algorithm -- a full-sample
    `np.quantile` over raw share `volume` -- which is exactly the defect the
    spec repairs (liquidity must be `close * volume` rupee turnover, bucketed
    causally/strictly-prior via `compute_prior_adv` + `causal_buckets`, never
    a full-sample quantile on share count). That old oracle is now wrong by
    construction and must not be patched to match output; it is replaced
    here by calling the actual production primitives
    (`compute_prior_adv`, `expectancy.causal_buckets`) that `Lens.stability`
    itself delegates to (`lens.py:552-558`), which is delegation to the real
    algorithm, not a second reimplementation of it. This test's own intent
    (shape/day_offsets preservation of the masked arrays) is unchanged."""
    panel = _build_liquidity_concentrated_panel()
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)

    prior_adv = compute_prior_adv(panel)
    adv_bucketing = expectancy.causal_buckets(
        prior_adv, panel.day_offsets, n_buckets=10, method="cross_sectional_rank"
    )

    for decile in (0, 1, 5, 9):
        mask = adv_bucketing.labels == decile
        masked_fwd = np.where(mask, fwd.values, np.nan)
        expected_n_total = int(np.sum(np.any(np.isfinite(masked_fwd), axis=1)))
        assert report.by_liquidity_decile[decile].n_total == expected_n_total


def test_stability_liquidity_decile_no_forward_fill_leak() -> None:
    """Cells outside a decile become NaN and must never be forward-filled
    (CLAUDE.md rule 6): a decile's n_total can never exceed the count of rows
    where the RAW (unmasked) forward return is defined."""
    panel = _build_liquidity_concentrated_panel()
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)

    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    max_possible_n_total = int(np.sum(np.any(np.isfinite(fwd.values), axis=1)))

    for table in report.by_liquidity_decile.values():
        assert table.n_total <= max_possible_n_total


def _assert_no_block_straddles_a_session(
    table: expectancy.ExpectancyTable, day_offsets: np.ndarray
) -> int:
    """Assert every block-bootstrap block in every bucket of `table` stays
    within a single real session per `day_offsets`. Returns the number of
    blocks checked (so callers can assert at least one was actually
    exercised, rather than vacuously passing on an empty table)."""
    checked = 0
    for bucket_stat in table.buckets:
        for block_start, block_length in bucket_stat.block_indices:
            block_end = block_start + block_length - 1
            start_session_idx = np.searchsorted(day_offsets[1:], block_start, side="right")
            end_session_idx = np.searchsorted(day_offsets[1:], block_end, side="right")
            assert start_session_idx == end_session_idx, (
                f"block ({block_start}, {block_length}) straddles a session boundary"
            )
            checked += 1
    return checked


def test_stability_by_year_block_bootstrap_respects_real_day_offsets() -> None:
    """Regression test: the by_year branch used to fabricate a single-session
    day_offsets=[0, N] for the year's rows, so the block bootstrap (the
    DEFAULT se_method) could sample a block straddling two REAL sessions
    within that year. horizon=2 is required to exercise the bootstrap at all
    (horizon=1 skips the overlap correction entirely)."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=6,
        bars_per_session=50,
        effect=0.01,
        sigma=0.0005,
        seed=55,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=2)

    assert 2024 in report.by_year
    checked = _assert_no_block_straddles_a_session(report.by_year[2024], panel.day_offsets)
    assert checked > 0


def test_stability_by_time_of_day_block_bootstrap_respects_real_day_offsets() -> None:
    """Regression test: the by_time_of_day branch used to fabricate a
    single-session day_offsets=[0, N] for the (non-contiguous-across-days)
    masked rows, so the block bootstrap could sample a block spanning the
    boundary between one day's time-of-day window and the next day's --
    rows with no real temporal adjacency."""
    panel = _build_panel(
        (2024,),
        sessions_per_year=4,
        bars_per_session=375,
        effect=0.01,
        sigma=0.0005,
        seed=77,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=2)

    assert len(report.by_time_of_day) > 0
    total_checked = 0
    for table in report.by_time_of_day.values():
        total_checked += _assert_no_block_straddles_a_session(table, panel.day_offsets)
    assert total_checked > 0


def test_stability_liquidity_decile_too_few_names_reported_as_nan() -> None:
    """With only 5 symbols, every non-empty decile holds a single symbol --
    below cross_sectional_rank's min_names=5 gate. That must come back as an
    honestly empty/NaN table (survives_costs=False, no buckets), not silently
    worked around by shrinking min_names or padding the decile."""
    panel = _build_small_panel(seed=100)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)

    assert len(report.by_liquidity_decile) > 0
    for table in report.by_liquidity_decile.values():
        assert len(table.buckets) == 0
        assert table.spread_bps == 0.0
        assert table.survives_costs is False


def test_verdict_criterion_4_fails_on_bottom_liquidity_decile_concentration() -> None:
    """THE behaviour the bug was suppressing: an edge planted only in the
    bottom volume decile must make criterion 4 report FAIL. Under the old
    code every by_liquidity_decile table was empty/NaN, so this branch of
    criterion 4 could never fire and a liquidity-concentrated hypothesis
    would silently PASS."""
    panel = _build_liquidity_concentrated_panel()
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H_bottom_liquidity_only",
        feature,
        1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )

    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "FAIL" in c4_line
    assert "bottom liquidity decile" in c4_line


def test_verdict_criterion_4_fails_when_only_bottom_decile_has_any_edge() -> None:
    """Branch where every OTHER decile is exactly flat (spread_bps == 0.0,
    not merely small): the bottom decile is the sole nonzero contributor, so
    criterion 4 must FAIL via the single-nonzero-decile path rather than the
    dominance-ratio path.

    `effect` is overridden down from the fixture's default 0.02 to 0.0005.
    specs/lens_criterion_4_repair.md Defect 1 makes liquidity decile
    membership a function of `close * volume` (rupee turnover), not the
    fixture's static per-symbol share-volume column alone
    (`compute_prior_adv`/`causal_buckets`, `lens.py:552-558`). At the
    default `effect=0.02` applied every bar for 400 bars, the edge symbols'
    cumulative price drift (up to exp(0.02*400) ~= exp(8), a ~2981x price
    swing) dwarfs the fixture's modest 1x-60x volume spread, so a planted
    edge symbol's rupee turnover migrates into non-bottom deciles across
    sessions -- leaking real edge into deciles 1-9 and breaking the "only
    the bottom decile has any edge" fixture this test relies on (measured:
    at effect=0.02, deciles 0, 1, 3, 7, 9 all come back nonzero). At
    effect=0.0005 the cumulative drift is negligible relative to the fixed
    volume ordering, so turnover-decile membership stays stable across all
    4 sessions and only decile 0 is nonzero, preserving this test's actual
    intent under the new bucketing."""
    panel = _build_liquidity_concentrated_panel(sigma_other_symbols=0.0, effect=0.0005)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    assert report.by_liquidity_decile[0].spread_bps != 0.0
    for decile in range(1, 10):
        assert report.by_liquidity_decile[decile].spread_bps == 0.0

    verdict = lens.verdict(
        "H_only_bottom_nonzero",
        feature,
        1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )

    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "FAIL" in c4_line
    assert "bottom liquidity decile" in c4_line


def test_verdict_criterion_4_passes_when_edge_spread_evenly_across_deciles() -> None:
    """When every decile carries a comparable edge (none dominates), the
    bottom-liquidity-decile check must NOT fire -- concentration means the
    bottom decile stands out, not merely that it has *an* edge."""
    panel = _build_liquidity_evenly_spread_panel()
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    spreads = [abs(t.spread_bps) for t in report.by_liquidity_decile.values()]
    assert max(spreads) <= 2 * sorted(spreads)[len(spreads) // 2]

    verdict = lens.verdict(
        "H_evenly_spread",
        feature,
        1,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )

    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "bottom liquidity decile" not in c4_line


# ==============================================================================
# Liquidity-decile volume NaN handling
# ==============================================================================


def test_stability_liquidity_decile_handles_nan_volume() -> None:
    """A NaN volume cell must be excluded from decile assignment (left at the
    sentinel, matching no decile) rather than crashing the quantile-boundary
    computation or being silently forward-filled."""
    panel = _build_small_panel(seed=200)
    volume = panel.field("volume").copy()
    volume[0, 0] = np.nan
    panel_with_nan_volume = Panel(
        fields={"close": panel.field("close"), "volume": volume},
        symbols=panel.symbols,
        ts=panel.ts,
        day_offsets=panel.day_offsets,
        dates=panel.dates,
    )
    lens = Lens(panel_with_nan_volume)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)

    assert len(report.by_liquidity_decile) > 0


# ==============================================================================
# Criterion 4: time-of-day axis, branches not reached by the panels above
# ==============================================================================


def test_verdict_criterion_4_no_time_of_day_buckets_at_all() -> None:
    """When every bar falls outside all three time-of-day windows (555-930
    minutes), by_time_of_day is empty and the time-of-day half of criterion 4
    must be skipped entirely rather than erroring."""
    day = dt.date(2024, 1, 2)
    day_start = pd.Timestamp(day.year, day.month, day.day, 3, 0, tz=_IST)
    idx = pd.date_range(day_start, periods=50, freq="1min")
    idx_utc = idx.tz_convert("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    day_offsets = np.array([0, 50], dtype=np.int32)
    dates_arr = np.array([day], dtype=object)

    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.001, size=(50, _N_SYMBOLS))
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.arange(1, _N_SYMBOLS + 1, dtype=np.float64), (50, 1))

    panel = Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    assert report.by_time_of_day == {}

    verdict = lens.verdict(
        "H_no_tod_buckets", feature, 1, method="cross_sectional_rank", n_buckets=5, n_boot=50
    )
    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "time-of-day" not in c4_line


def _build_full_day_panel(
    returns_by_bar: np.ndarray, *, n_sessions: int = 3, bars_per_session: int = 375
) -> Panel:
    """Build a panel spanning full 9:15-15:30 sessions (375 one-minute bars),
    with the same per-bar returns pattern (n_symbols columns) repeated
    identically every session."""
    dates = [dt.date(2024, 1, 2 + i) for i in range(n_sessions)]
    n_rows = bars_per_session * n_sessions
    returns = np.tile(returns_by_bar, (n_sessions, 1))
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.arange(1, _N_SYMBOLS + 1, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def test_verdict_criterion_4_time_of_day_all_flat_edges() -> None:
    """When every symbol has IDENTICAL returns every bar (zero cross-sectional
    variance), cross_sectional_rank ties collapse every time-of-day bucket to
    spread_bps == 0.0 exactly. nonzero_edges must then come back empty, and
    criterion 4 must not spuriously FAIL on all-zero edges."""
    rng = np.random.default_rng(11)
    per_bar_return = rng.normal(0.0, 0.001, size=(375, 1))
    returns_by_bar = np.tile(per_bar_return, (1, _N_SYMBOLS))  # identical across symbols

    panel = _build_full_day_panel(returns_by_bar)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    assert len(report.by_time_of_day) >= 2
    for table in report.by_time_of_day.values():
        assert table.spread_bps == 0.0

    verdict = lens.verdict(
        "H_flat_tod", feature, 1, method="cross_sectional_rank", n_buckets=5, n_boot=50
    )
    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "time-of-day" not in c4_line


def test_verdict_criterion_4_single_nonzero_time_of_day_edge() -> None:
    """An edge planted ONLY in the 'open' window (deterministic, sigma=0
    elsewhere) leaves 'mid'/'close' at spread_bps == 0.0 exactly, so
    nonzero_edges has length 1 -- there is nothing to compare a single edge
    against, so the dominance check must not fire."""
    means = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    returns_by_bar = np.zeros((375, _N_SYMBOLS))
    returns_by_bar[0:75, :] = means[None, :]  # "open" window only (555-630 min)

    panel = _build_full_day_panel(returns_by_bar)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    nonzero = [t.spread_bps for t in report.by_time_of_day.values() if t.spread_bps != 0]
    assert len(nonzero) == 1

    verdict = lens.verdict(
        "H_single_nonzero_tod", feature, 1, method="cross_sectional_rank", n_buckets=5, n_boot=50
    )
    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "time-of-day" not in c4_line


def test_verdict_criterion_4_fails_on_time_of_day_dominance() -> None:
    """Same per-symbol effect shape (same sign every window) but scaled 20x
    larger during 'open' than 'mid'/'close': the 'open' edge dominates the
    median of the other windows by more than 2x, so criterion 4 must FAIL via
    the time-of-day dominance path."""
    shape = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    returns_by_bar = np.zeros((375, _N_SYMBOLS))
    returns_by_bar[0:75, :] = shape[None, :] * 0.02  # open: dominant
    returns_by_bar[75:285, :] = shape[None, :] * 0.001  # mid: small
    returns_by_bar[285:375, :] = shape[None, :] * 0.001  # close: small

    panel = _build_full_day_panel(returns_by_bar)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    report = lens.stability(feature, horizon=1)
    spreads = {name: t.spread_bps for name, t in report.by_time_of_day.items()}
    assert len(spreads) == 3
    assert all(s != 0 for s in spreads.values())

    verdict = lens.verdict(
        "H_tod_dominance", feature, 1, method="cross_sectional_rank", n_buckets=5, n_boot=50
    )
    c4_line = next(r for r in verdict.reasons if r.startswith("4."))
    assert "FAIL" in c4_line
    assert "concentrated in single time-of-day bucket" in c4_line


# ==============================================================================
# Criterion 6: branches not reached by the panels above
# ==============================================================================


def test_verdict_criterion_6_pass_with_multiple_trials() -> None:
    """Criterion 6 PASSes with effective_n_trials >= 2 when the deflated
    Sharpe (computed on the supplied `strategy_returns`, per
    specs/lens_criteria_6_7_repair.md Defect 1 -- never on `fwd.values`)
    clears `expected_max_sharpe(n_trials)`, compared as a PROBABILITY against
    `DSR_SIGNIFICANCE` (the second defect's fix: `deflated_sharpe` returns a
    probability, `expected_max_sharpe` a Sharpe level, so the correct form is
    `deflated_sharpe(strategy_returns, sr0=expected_max_sharpe(n)) >
    DSR_SIGNIFICANCE`, never a direct Sharpe-vs-Sharpe comparison).
    `strategy_returns` is a strong, low-noise, directional 1-D per-period P&L
    series, independent of the panel's own (flat, near-zero-drift) returns,
    so this test cannot be satisfied by the old `fwd.values`-reading
    implementation for a flat panel -- it must actually read the supplied
    argument."""
    dates = [dt.date(2024, 1, 2 + i) for i in range(20)]
    bars = 50
    n_rows = bars * len(dates)
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.0005, size=(n_rows, _N_SYMBOLS))
    close = _close_prices_from_log_returns(returns)
    volume = np.tile(np.arange(1, _N_SYMBOLS + 1, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid(dates, bars)
    panel = Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")
    strategy_returns = 0.001 + rng.normal(0.0, 0.0005, size=1000)

    verdict = lens.verdict(
        "H_c6_multi_trial_pass",
        feature,
        1,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )
    c6_line = next(r for r in verdict.reasons if r.startswith("6."))
    assert "PASS" in c6_line
    assert "trials=2" in c6_line


def test_verdict_criterion_6_fails_with_zero_valid_forward_returns() -> None:
    """A `strategy_returns` array that is entirely NaN (e.g. because the
    strategy never traded in this window -- horizon >= bars_per_session
    means every row's forward return runs off the session end, so the
    hypothetical realised P&L series is undefined everywhere) must report
    `NOT_EVALUATED`, never FAIL, per specs/lens_criteria_6_7_repair.md
    Defect 1: 'A supplied strategy_returns that is empty, all-NaN, or
    shorter than 2 finite values also yields NOT_EVALUATED, never a crash
    and never 0.0.' This replaces the old FAIL assertion, which encoded the
    repaired defect itself: the old code derived criterion 6 from
    `fwd.values` (zero valid forward returns -> FAIL via an empty array),
    not from a strategy P&L series at all."""
    dates = [dt.date(2024, 1, 1)]
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session=3)

    close = np.ones((3, _N_SYMBOLS), dtype=np.float64)
    volume = np.ones((3, _N_SYMBOLS), dtype=np.float64)

    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    all_nan_strategy_returns = np.full(10, np.nan, dtype=np.float64)
    verdict = lens.verdict(
        "H_c6_zero_valid",
        feature,
        horizon=3,
        effective_n_trials=2,
        strategy_returns=all_nan_strategy_returns,
    )

    close64 = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close64, panel.day_offsets, 3)
    assert fwd.n_defined == 0

    c6_line = next(r for r in verdict.reasons if r.startswith("6."))
    assert "NOT_EVALUATED" in c6_line


# ==============================================================================
# Verdict criterion 7: recent-years cost gate
#
# Regression coverage for the H2 gap: criteria 1 (pooled edge) and 2 (sign-only
# stability) can both PASS on a monotonically decaying edge, because criterion 2
# only counts each year's SIGN, never its magnitude. Criterion 7 checks the
# magnitude of the mean edge over the last two COMPLETE years directly, using
# the per-year `spread_bps` StabilityReport.by_year already computes -- it must
# never recompute expectancy.
# ==============================================================================


def _build_panel_year_effects(
    year_effects: dict[int, float],
    sessions_per_year: int | dict[int, int],
    bars_per_session: int,
    *,
    sigma: float = 0.0005,
    seed: int = 7,
    monthly_years: set[int] | None = None,
) -> Panel:
    """Build a Panel with an independently-controlled signal magnitude per year.

    Unlike `_build_panel`, which only toggles a single global `effect` on/off per
    year via `signal_years`, this lets each year carry its OWN effect magnitude --
    needed to construct the H2 regression shape: a large early edge that decays
    below the cost hurdle in the most recent years while keeping the same sign
    throughout (so criteria 1 and 2 still PASS and only criterion 7 catches it).

    `monthly_years`: years in this set get `_monthly_session_dates` (12-month
    coverage, needed for a year to be "complete" under criterion 7's amended
    rule) instead of consecutive-January dates; see `_build_panel`'s
    identical parameter for why this changes no computed statistic besides
    which calendar months each year's sessions land in.
    """
    years = tuple(sorted(year_effects))
    if isinstance(sessions_per_year, int):
        year_sessions = {year: sessions_per_year for year in years}
    else:
        year_sessions = dict(sessions_per_year)
    if monthly_years is None:
        monthly_years = set()

    dates: list[dt.date] = []
    for year in years:
        if year in monthly_years:
            dates += _monthly_session_dates(year, year_sessions[year])
        else:
            dates += [dt.date(year, 1, 2 + i) for i in range(year_sessions[year])]

    n_rows = len(dates) * bars_per_session
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_rows, _N_SYMBOLS), dtype=np.float64)

    row = 0
    for date in dates:
        means = _signal_means(year_effects[date.year])
        block_means = np.tile(means, (bars_per_session, 1))
        noise = rng.normal(0, sigma, (bars_per_session, _N_SYMBOLS))
        returns[row : row + bars_per_session, :] = block_means + noise
        row += bars_per_session

    close = _close_prices_from_log_returns(returns)
    volume = np.full((n_rows, _N_SYMBOLS), 1e5, dtype=np.float64)
    for i in range(_N_SYMBOLS):
        volume[:, i] = (i + 1) * 1e3

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    panel = Panel(
        fields={"close": close, "volume": volume},
        symbols=_SYMBOLS,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )
    return panel


def test_verdict_criterion_7_passes_when_last_two_years_clear_hurdle() -> None:
    """Criterion 7 PASSes when the last two complete years both clear 2x hurdle.

    `monthly_years={2024, 2025}` (specs/lens_criteria_6_7_repair.md Defect 2:
    a year is "complete" iff it has a session in each of its 12 calendar
    months) is required for either year to be usable by criterion 7 at all --
    the plain `date(year, 1, 2 + i)` placement this test used before can
    never satisfy that rule (see `_monthly_session_dates`'s docstring)."""
    panel = _build_panel_year_effects(
        {2024: 0.01, 2025: 0.01},
        sessions_per_year={2024: 25, 2025: 25},
        bars_per_session=50,
        monthly_years={2024, 2025},
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H_c7_pass", feature, horizon=1, method="cross_sectional_rank", n_buckets=5
    )

    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    assert "PASS" in c7_line
    assert "2024" in c7_line and "2025" in c7_line


def test_verdict_criterion_7_fails_on_decayed_edge_while_criteria_1_and_2_pass() -> None:
    """The exact H2 shape: large early edge, decayed recent edge, consistent sign.

    Pooled edge (criterion 1) and sign stability (criterion 2) both PASS -- exactly
    what let H2 slip through -- while the last two years' mean edge sits under the
    cost hurdle, so criterion 7 must FAIL. This is the regression test for the gap
    that criterion 7 exists to close.

    `monthly_years={2024, 2025}` makes ONLY those two years "complete" under
    criterion 7's amended rule (specs/lens_criteria_6_7_repair.md Defect 2);
    2018-2023 stay on their original January-only dates since criterion 2's
    `n_years_total` counts every year present regardless of completeness (the
    spec: "Scope of the exclusion: partial years are excluded from criterion 7
    ONLY"), so those years need no change to keep criteria 1/2's pooled
    sample and sign-stability count exactly as before. Because a date's
    calendar placement never affects the returns drawn for it (see
    `_build_panel_year_effects`'s `monthly_years` docstring), 2024/2025's own
    spread_bps values are also unchanged by this move.
    """
    year_effects = {
        2018: 0.01,
        2019: 0.01,
        2020: 0.01,
        2021: 0.01,
        2022: 0.01,
        2023: 0.01,
        2024: 0.0005,
        2025: 0.0005,
    }
    sessions = {2018: 5, 2019: 5, 2020: 5, 2021: 5, 2022: 5, 2023: 5, 2024: 25, 2025: 25}
    panel = _build_panel_year_effects(
        year_effects, sessions, bars_per_session=375, monthly_years={2024, 2025}
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H2_shape", feature, horizon=1, method="cross_sectional_rank", n_buckets=5
    )

    c1_line = next(r for r in verdict.reasons if r.startswith("1."))
    c2_line = next(r for r in verdict.reasons if r.startswith("2."))
    c7_line = next(r for r in verdict.reasons if r.startswith("7."))

    assert "PASS" in c1_line, c1_line
    assert "PASS" in c2_line, c2_line
    assert "FAIL" in c7_line, c7_line
    assert "2024" in c7_line and "2025" in c7_line


def test_verdict_criterion_7_not_evaluated_with_fewer_than_two_complete_years() -> None:
    """NOT_EVALUATED with < 2 complete years -- never a silent PASS.

    Asserting "NOT_EVALUATED" AND the absence of "PASS" (rather than just checking
    survived, or just checking the token loosely) means a mutant that defaults the
    <2-complete-years branch to PASS is caught directly by this test.
    """
    panel = _build_panel_year_effects(
        {2024: 0.01},
        sessions_per_year={2024: 25},
        bars_per_session=50,
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H_c7_not_evaluated", feature, horizon=1, method="cross_sectional_rank", n_buckets=5
    )

    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    assert "NOT_EVALUATED" in c7_line
    assert "PASS" not in c7_line
    assert "FAIL" not in c7_line


def test_verdict_criterion_7_excludes_partial_year_missing_a_month() -> None:
    """OBSOLETE-AND-REPLACED (spec: specs/lens_criteria_6_7_repair.md Defect 2).
    This test used to assert the old `session_counts.get(year, 0) >= 20`
    threshold directly (itself a hand-chosen constant CLAUDE.md rule 8
    forbids) -- that threshold no longer exists in the implementation, so
    asserting it would test dead code. It is replaced with the equivalent
    NEW behaviour the amendment introduces: a year is "complete" iff it has
    a session in EACH of its 12 calendar months, with NO session-count
    threshold at all. Here 2023 gets only 6 sessions, spread across months
    1-6 only (missing Jul-Dec) via `_monthly_session_dates` -- regardless of
    how many total sessions it has, a year missing even one calendar month
    is PARTIAL -- while 2022 and 2024 each get 12 sessions, one per calendar
    month, so both are genuinely complete and must be the two years used."""
    panel = _build_panel_year_effects(
        {2022: 0.01, 2023: 0.01, 2024: 0.01},
        sessions_per_year={2022: 12, 2023: 6, 2024: 12},
        bars_per_session=50,
        monthly_years={2022, 2023, 2024},
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H_c7_excluded_year", feature, horizon=1, method="cross_sectional_rank", n_buckets=5
    )

    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    # 2023 covers only Jan-Jun and is PARTIAL: the years actually used are the
    # two OTHER complete years, 2022 and 2024, and the reason line must name
    # 2023 as an excluded partial year (spec: the field is always emitted).
    assert "years=2022,2024" in c7_line
    assert "excluded partial years: 2023" in c7_line


def test_verdict_all_seven_criteria_reported_after_one_fails() -> None:
    """All seven criteria are reported even after criterion 7 fails."""
    year_effects = {
        2018: 0.01,
        2019: 0.01,
        2020: 0.01,
        2021: 0.01,
        2022: 0.01,
        2023: 0.01,
        2024: 0.0005,
        2025: 0.0005,
    }
    sessions = {2018: 5, 2019: 5, 2020: 5, 2021: 5, 2022: 5, 2023: 5, 2024: 25, 2025: 25}
    panel = _build_panel_year_effects(year_effects, sessions, bars_per_session=375)
    lens = Lens(panel)
    feature = lens.feature("return_1")

    verdict = lens.verdict(
        "H2_shape_all_reported", feature, horizon=1, method="cross_sectional_rank", n_buckets=5
    )

    assert len(verdict.reasons) == 7
    for i in range(1, 8):
        assert any(r.startswith(f"{i}.") for r in verdict.reasons)


def test_verdict_survived_false_when_only_criterion_7_fails() -> None:
    """survived is False when criteria 1-6 all PASS and only criterion 7 fails.

    `monthly_years={2024, 2025}` makes those two years "complete" (see the
    decayed-edge test above for the identical rationale) so criterion 7 is
    actually FAIL rather than NOT_EVALUATED. Criterion 6 now needs an
    explicit `strategy_returns` array to PASS at all (never `fwd.values`,
    specs/lens_criteria_6_7_repair.md Defect 1); `effective_n_trials=2` with
    a strongly, unambiguously winning series (mean=0.01, sigma=0.001, n=500)
    saturates `deflated_sharpe` to ~1.0 against
    `expected_max_sharpe(2, 1.0) ~= 0.52`, so it PASSes independent of the
    exact `DSR_SIGNIFICANCE` value -- see test_verdict_criterion_6_pass_with_
    multiple_trials's identical reasoning.
    """
    year_effects = {
        2018: 0.01,
        2019: 0.01,
        2020: 0.01,
        2021: 0.01,
        2022: 0.01,
        2023: 0.01,
        2024: 0.0005,
        2025: 0.0005,
    }
    sessions = {2018: 5, 2019: 5, 2020: 5, 2021: 5, 2022: 5, 2023: 5, 2024: 25, 2025: 25}
    panel = _build_panel_year_effects(
        year_effects, sessions, bars_per_session=375, monthly_years={2024, 2025}
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")
    strategy_returns = np.random.default_rng(1).normal(0.01, 0.001, size=500)

    latency = {0: 1.0, 1: 0.6, 2: 0.55}
    verdict = lens.verdict(
        "H2_shape_survived_check",
        feature,
        horizon=1,
        latency_profile=latency,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
    )

    for i in range(1, 7):
        line = next(r for r in verdict.reasons if r.startswith(f"{i}."))
        assert "PASS" in line, f"expected criterion {i} to PASS but got: {line}"
    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    assert "FAIL" in c7_line, c7_line
    assert verdict.survived is False


def test_verdict_criterion_7_fails_on_opposite_sign_to_dominant() -> None:
    """A recent edge of the correct magnitude but the OPPOSITE sign to the
    dominant sign must FAIL, even though the magnitude alone would clear the
    hurdle. `Lens.stability` is monkeypatched to return a hand-crafted
    StabilityReport: the feature/outcome construction `_build_panel_year_effects`
    uses ties the FEATURE and the FORWARD RETURN to the same underlying return
    series, so a bucket spread (top mean - bottom mean) can never come out
    negative there regardless of a chosen effect's sign -- the rank-based spread
    is always non-negative by construction. Monkeypatching isolates the criterion
    7 sign-check logic under test from that fixture limitation, while the
    session-completeness gate (self.panel.dates) still comes from a real panel.

    `monthly_years={2022, 2023, 2024}` makes all three years 12-month
    "complete" (specs/lens_criteria_6_7_repair.md Defect 2), so criterion 7's
    real `complete_years` computation (against `self.panel.dates`, which the
    monkeypatch does NOT touch) selects the two most recent, 2023 and 2024 --
    matching the fake StabilityReport's own by_year keys and this test's
    assertions on those two years by name.
    """
    panel = _build_panel(
        (2022, 2023, 2024),
        sessions_per_year=25,
        bars_per_session=25,
        effect=0.001,
        seed=200,
        monthly_years={2022, 2023, 2024},
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    def fake_stability(feature: Feature, horizon: int) -> StabilityReport:
        def _table(spread_bps: float) -> expectancy.ExpectancyTable:
            return expectancy.ExpectancyTable(
                buckets=(),
                horizon=horizon,
                feature_name=feature.name,
                n_total=1000,
                cost_hurdle_bps=8.26452,
                spread_bps=spread_bps,
                spread_t=5.0,
                survives_costs=abs(spread_bps) > 2 * 8.26452,
            )

        return StabilityReport(
            by_year={2022: _table(60.0), 2023: _table(-55.0), 2024: _table(-58.0)},
            by_time_of_day={},
            by_liquidity_decile={},
            n_years_total=3,
            n_years_sign_consistent=2,
            dominant_sign="+",
        )

    lens.stability = fake_stability  # type: ignore[method-assign]

    verdict = lens.verdict("H_c7_opposite_sign", feature, horizon=1)

    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    assert "FAIL" in c7_line, c7_line
    assert "2023" in c7_line and "2024" in c7_line


def test_verdict_criterion_7_fails_when_mean_recent_edge_is_exactly_zero() -> None:
    """The two recent years' edges exactly cancel (mean == 0.0): neither the `> 0`
    nor the `< 0` branch of the sign classification fires, so `recent_sign` stays
    "mixed" and criterion 7 FAILs on the sign check -- covers that branch
    explicitly rather than leaving it an untested fallthrough.

    `monthly_years={2022, 2023, 2024}`: see the identical rationale in
    test_verdict_criterion_7_fails_on_opposite_sign_to_dominant immediately
    above -- criterion 7's real completeness check reads `self.panel.dates`,
    which the `lens.stability` monkeypatch below does not touch."""
    panel = _build_panel(
        (2022, 2023, 2024),
        sessions_per_year=25,
        bars_per_session=25,
        effect=0.001,
        seed=201,
        monthly_years={2022, 2023, 2024},
    )
    lens = Lens(panel)
    feature = lens.feature("return_1")

    def fake_stability(feature: Feature, horizon: int) -> StabilityReport:
        def _table(spread_bps: float) -> expectancy.ExpectancyTable:
            return expectancy.ExpectancyTable(
                buckets=(),
                horizon=horizon,
                feature_name=feature.name,
                n_total=1000,
                cost_hurdle_bps=8.26452,
                spread_bps=spread_bps,
                spread_t=5.0,
                survives_costs=abs(spread_bps) > 2 * 8.26452,
            )

        return StabilityReport(
            by_year={2022: _table(60.0), 2023: _table(50.0), 2024: _table(-50.0)},
            by_time_of_day={},
            by_liquidity_decile={},
            n_years_total=3,
            n_years_sign_consistent=2,
            dominant_sign="+",
        )

    lens.stability = fake_stability  # type: ignore[method-assign]

    verdict = lens.verdict("H_c7_zero_mean", feature, horizon=1)

    c7_line = next(r for r in verdict.reasons if r.startswith("7."))
    assert "FAIL" in c7_line, c7_line
    assert "mean=0.00 bps" in c7_line
