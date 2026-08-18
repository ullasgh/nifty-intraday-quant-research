"""Independent test suite (author B) for specs/lens_criteria_6_7_repair.md.

Written from that spec ALONE, before any implementation of the repair exists.
Does not read, import, or otherwise reference
tests/test_lens_criteria_repair_a.py -- two independent readings of the same
contract are the point; copying the other author's tests would destroy it.

Reuses panel/fixture helpers from tests/test_lens.py where they fit. Two
helpers had to be written fresh here because nothing in test_lens.py can
express what criterion 7's window-EDGE tests need: a panel whose first/last
calendar date is controlled precisely (test_lens._build_panel always anchors
every year at Jan 2 and never reaches December), and irregular per-day bar
counts (test_lens._session_grid takes one fixed bars_per_session for the
whole panel). CLAUDE.md rule 5 forbids assuming a fixed bars/session anyway.
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
    _all_pass_panel,
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

    Needed so criterion 7's window-edge fixtures can include an irregular
    session (e.g. a Muhurat-like 60-bar day) rather than assuming a fixed
    bar count for every session, per CLAUDE.md rule 5.
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
    `sessions_per_year` days, so its panel never reaches December and cannot
    exercise criterion 7's window-edge rule. This lets a caller control the
    panel's first/last calendar date directly, which is the entire subject
    under test.
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

    test_lens._signal_means (used by test_lens._build_panel and _dated_panel
    above) assigns 5 symbol means [-e, -e/2, 0, e/2, e], which sum to exactly
    zero -- fwd.values.flatten() over such a panel has mean 0 by
    construction, so it cannot demonstrate the bug. Here all symbols share
    one positive mean instead, so fwd.values (the UNCONDITIONAL forward
    return the old criterion 6 mistakenly deflated) drifts strongly positive.
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


def _call_verdict(
    panel: Panel,
    hypothesis_id: str = "H_REPAIR_B",
    *,
    latency_profile: dict[int, float] | None = _PASS_LATENCY,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = None,
    seed: int = 0,
    horizon: int = 1,
    n_boot: int = 100,
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
    verdict = _call_verdict(_all_pass_panel(), strategy_returns=None, effective_n_trials=1)

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason
    assert "strategy_returns" in reason


def test_criterion6_not_evaluated_does_not_count_as_pass_in_overall_verdict() -> None:
    """Spec item 2: NOT_EVALUATED must not silently count as PASS in the
    overall verdict -- mirrors criterion 5's existing precedent (see
    test_verdict_criterion5_not_evaluated_without_latency_profile in
    tests/test_lens.py: every other criterion PASSes, the NOT_EVALUATED
    criterion is skipped, and the hypothesis still survives)."""
    verdict = _call_verdict(_all_pass_panel(), strategy_returns=None, effective_n_trials=1)

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
        _all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=1
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "trial" in reason.lower()

    # Distinguishable from item 1's "no input at all" NOT_EVALUATED -- a
    # reader must be able to tell WHY it was not evaluated.
    none_reason = _call_verdict(
        _all_pass_panel(), strategy_returns=None, effective_n_trials=1
    ).reasons[_C6_LINE]
    assert reason != none_reason


@pytest.mark.parametrize(
    "strategy_returns, effective_n_trials, expected_token",
    [
        (np.random.default_rng(42).normal(0.003, 0.001, size=300), 2, "PASS"),
        (np.random.default_rng(99).normal(-0.002, 0.001, size=300), 2, "FAIL"),
        (np.random.default_rng(42).normal(0.003, 0.001, size=300), 100_000, "FAIL"),
    ],
    ids=["pass_signal_two_trials", "fail_negative_two_trials", "fail_same_signal_many_trials"],
)
def test_criterion6_deflates_supplied_strategy_returns_pass_and_fail(
    strategy_returns: np.ndarray, effective_n_trials: int, expected_token: str
) -> None:
    """Spec item 4: strategy_returns supplied with effective_n_trials>=2 ->
    real PASS/FAIL. Covers both outcomes driven by the returns constructed
    (cases 1-2), plus the same passing series flipped to FAIL purely by
    raising the trial count (case 3), isolating that knob."""
    verdict = _call_verdict(
        _all_pass_panel(),
        strategy_returns=strategy_returns,
        effective_n_trials=effective_n_trials,
    )

    reason = verdict.reasons[_C6_LINE]
    assert expected_token in reason
    other_token = "FAIL" if expected_token == "PASS" else "PASS"
    assert other_token not in reason
    assert "NOT_EVALUATED" not in reason


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
    """Spec item 5: empty / all-NaN / fewer than 2 finite values ->
    NOT_EVALUATED, never a crash and never a bare 0.0 masquerading as a real
    result. The call itself not raising is part of what this test checks."""
    verdict = _call_verdict(
        _all_pass_panel(), strategy_returns=strategy_returns, effective_n_trials=2
    )

    reason = verdict.reasons[_C6_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason


def test_criterion6_regression_guard_unconditional_drift_must_not_pass() -> None:
    """Spec item 6, the most important test in this suite: a panel whose
    UNCONDITIONAL forward returns (fwd.values) drift strongly positive, paired
    with a flat/negative strategy_returns series, must NOT report criterion 6
    as PASS.

    This is exactly the defect: the old implementation deflates
    fwd.values.flatten() -- every symbol's raw forward return, not the
    strategy's P&L -- so a market that merely drifted up gets reported as
    "multiple testing was accounted for". This test must fail against the old
    implementation (it cannot even run there: strategy_returns is not a
    parameter it recognizes)."""
    panel = _positive_drift_panel(n_days=40, bars_per_session=30, drift=0.02, seed=0)

    # Sanity: the panel really does drift strongly positive -- this is
    # exactly the quantity the OLD implementation deflated.
    close = panel.field("close").astype(np.float64)
    fwd = expectancy.forward_returns(close, panel.day_offsets, 1)
    assert np.nanmean(fwd.values) > 0.01

    flat_negative_returns = np.random.default_rng(7).normal(-0.001, 0.0005, size=250)

    verdict = _call_verdict(
        panel,
        latency_profile=None,
        strategy_returns=flat_negative_returns,
        effective_n_trials=2,
        n_boot=50,
    )

    reason = verdict.reasons[_C6_LINE]
    assert "PASS" not in reason
    assert "FAIL" in reason


# ---------------------------------------------------------------------------
# Criterion 7 -- recent-years cost gate, threshold-free completeness rule
# ---------------------------------------------------------------------------


def test_criterion7_excludes_partial_year_at_window_end_and_names_it() -> None:
    """Spec item 7: a window ending mid-year (2025-07-31, mirroring the real
    H2/H3 window described in the spec) excludes 2025 as partial, named in
    the reason."""
    dates = [
        dt.date(2021, 1, 1),
        dt.date(2021, 6, 15),
        dt.date(2022, 6, 15),
        dt.date(2023, 6, 15),
        dt.date(2024, 6, 15),
        dt.date(2025, 3, 10),
        dt.date(2025, 7, 31),
    ]
    panel = _dated_panel(dates, bars_per_session=20, signal_years=set(range(2021, 2026)))
    verdict = _call_verdict(panel, latency_profile=None, n_boot=50)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2023,2024" in reason
    assert "2025" in reason


def test_criterion7_year_complete_when_window_ends_exactly_on_dec31_boundary() -> None:
    """Spec item 8 + item 11: a window ending EXACTLY 2024-12-31 treats 2024
    as complete, and the two years actually used are named explicitly by year
    number. Includes a Muhurat-like 60-bar irregular session on the final
    day: criterion 7 is about window EDGES, so a fixed bars/session cannot be
    assumed here (CLAUDE.md rule 5)."""
    dates = [
        dt.date(2021, 1, 1),
        dt.date(2021, 6, 15),
        dt.date(2022, 6, 15),
        dt.date(2023, 6, 15),
        dt.date(2024, 6, 15),
        dt.date(2024, 12, 31),
    ]
    bars_per_session = [20, 20, 20, 20, 20, 60]  # last session is Muhurat-like
    panel = _dated_panel(dates, bars_per_session, signal_years=set(range(2021, 2025)))
    verdict = _call_verdict(panel, latency_profile=None, n_boot=50)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2023,2024" in reason
    assert "excluded partial years: none" in reason


def test_criterion7_excludes_partial_year_at_window_start_symmetric_rule() -> None:
    """Spec item 9: the completeness rule is symmetric -- a window STARTING
    mid-year (2018-06-01) excludes 2018 as partial too, not only a mid-year
    end (item 7)."""
    dates = [
        dt.date(2018, 6, 1),
        dt.date(2018, 10, 1),
        dt.date(2019, 6, 15),
        dt.date(2020, 6, 15),
        dt.date(2020, 12, 31),
    ]
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2018, 2019, 2020})
    verdict = _call_verdict(panel, latency_profile=None, n_boot=50)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2019,2020" in reason
    assert "2018" in reason


@pytest.mark.parametrize(
    "dates, signal_years",
    [
        (
            [
                dt.date(2023, 3, 1),
                dt.date(2023, 6, 15),
                dt.date(2024, 3, 1),
                dt.date(2024, 6, 30),
            ],
            {2023, 2024},
        ),
        (
            [dt.date(2023, 1, 1), dt.date(2023, 6, 15), dt.date(2023, 12, 31)],
            {2023},
        ),
    ],
    ids=["zero_complete_years", "one_complete_year"],
)
def test_criterion7_fewer_than_two_complete_years_is_not_evaluated(
    dates: list[dt.date], signal_years: set[int]
) -> None:
    """Spec item 10: with fewer than two complete years, criterion 7 must
    report NOT_EVALUATED -- never a silent PASS. Covers both the zero- and
    one-complete-year corners."""
    panel = _dated_panel(dates, bars_per_session=20, signal_years=signal_years)
    verdict = _call_verdict(panel, latency_profile=None, n_boot=50)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" in reason
    assert "PASS" not in reason
    assert "FAIL" not in reason


def test_criterion7_uses_the_two_most_recent_complete_years_explicitly() -> None:
    """Spec item 11: with three complete years available (2019, 2020, 2021),
    the two years actually USED are the two most recent (2020, 2021) --
    asserted by explicit year number, an earlier complete year must not
    silently sneak in just because it exists."""
    dates = [
        dt.date(2019, 1, 1),
        dt.date(2019, 6, 15),
        dt.date(2020, 6, 15),
        dt.date(2021, 6, 15),
        dt.date(2021, 12, 31),
    ]
    panel = _dated_panel(dates, bars_per_session=20, signal_years={2019, 2020, 2021})
    verdict = _call_verdict(panel, latency_profile=None, n_boot=50)

    reason = verdict.reasons[_C7_LINE]
    assert "NOT_EVALUATED" not in reason
    assert "years=2020,2021" in reason
    assert "2019" not in reason


# ---------------------------------------------------------------------------
# Both criteria: all seven always reported, and determinism
# ---------------------------------------------------------------------------


def test_verdict_reports_all_seven_criteria_in_order_after_a_failure() -> None:
    """Spec item 12: all seven criteria are always reported, in numbered
    order, even once one (or several) of them fail."""
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
    """Spec item 13: same inputs and seed -> identical verdict text, for both
    the raw reasons and the rendered explain()/to_markdown() views."""
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
