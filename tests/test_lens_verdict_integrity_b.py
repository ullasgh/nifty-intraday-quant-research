"""Independent test suite B for specs/lens_verdict_integrity.md.

Written from the spec alone (per CLAUDE.md rule 1), without reading any sibling
suite or any implementation change -- none exists yet. Covers all 10 numbered
"Test obligations" in the spec, fixing L1 (unevaluated criteria silently
dropped from the conjunction), L3 (criterion 5 hard-FAILs a negative-signed
edge), and L4 (`Lens.universe` stored and never read).

Originally written against unimplemented code (deliberately RED); updated
after the implementation landed to match AMENDMENT 1 and AMENDMENT 2 of the
spec, both now pinned rather than left as this suite's own guesses:
  - `HypothesisVerdict.outcome` is a plain string `Literal["SURVIVED",
    "KILLED", "INCONCLUSIVE"]`, not an enum (AMENDMENT 1.2) -- matches this
    suite's original reading.
  - The symbol count is `n_symbols_used` (AMENDMENT 1.3) -- matches this
    suite's original reading.
  - `cross_sectional_rank` enforces `min_names=5`; any fixture reaching it,
    including a RESTRICTED universe, must retain >= 5 symbols or every row
    goes all-NaN and every criterion degrades to FAIL for reasons unrelated
    to what is under test (AMENDMENT 2). The obligation-9/10 fixture below
    was rebuilt from 5/4 symbols to 6/5 for exactly this reason.
  - Obligation 9's second assertion originally checked `n_total`, which is a
    ROW count (`expectancy.n_defined`), not a symbol count, and is correctly
    unchanged by dropping one of several always-present symbols (AMENDMENT
    2). Replaced with a `n_symbols_used` check via `verdict()`; the
    `spread_bps` assertion (the real, stronger effect check) is unchanged.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nifty_quant.data.panel import Panel
from nifty_quant.research.lens import HypothesisVerdict, Lens

_IST = ZoneInfo("Asia/Kolkata")
_SYMBOLS = ("Q1", "Q2", "Q3", "Q4", "Q5")
_VOLUME_PROFILE = (2e3, 2e4, 2e5, 2e6, 2e7)


# ---------------------------------------------------------------------------
# Panel construction helpers (independent of tests/test_lens.py; built fresh
# from the Panel/Lens public API for this suite alone).
# ---------------------------------------------------------------------------


def _session_grid(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _signal_means(effect: float, n_symbols: int = 5) -> np.ndarray:
    """Symmetric per-symbol mean returns, evenly spaced in [-effect, +effect].

    For n_symbols=5 this is exactly [-effect, -effect/2, 0, effect/2, effect]
    (unchanged). Generalized so obligation-9/10 fixtures can use 6 symbols --
    AMENDMENT 2 pins `cross_sectional_rank`'s `min_names=5` floor, so a
    restricted universe used to demonstrate L4 must retain >= 5 symbols, which
    requires starting from >= 6.
    """
    return np.linspace(-effect, effect, n_symbols, dtype=np.float64)


def _close_from_log_returns(returns: np.ndarray) -> np.ndarray:
    cum = np.cumsum(returns, axis=0)
    log_close = np.vstack([np.zeros((1, returns.shape[1]), dtype=np.float64), cum])
    return np.exp(log_close)[:-1, :]


def _build_panel(
    year_months: dict[int, list[int]],
    bars_per_session: int,
    *,
    effect: float,
    sigma: float,
    symbols: tuple[str, ...] = _SYMBOLS,
    volume_profile: tuple[float, ...] = _VOLUME_PROFILE,
    seed: int = 7,
) -> Panel:
    """Build a Panel whose per-symbol mean return is fixed by rank (a clean

    cross-sectional edge), with explicit per-year month lists so callers can
    make some years calendar-complete (all 12 months present, for criterion
    7) and others not, independently of how many years exist in total (for
    criterion 2's sign-stability count).
    """
    n_symbols = len(symbols)
    dates: list[dt.date] = []
    for year in sorted(year_months):
        for month in year_months[year]:
            dates.append(dt.date(year, month, 2))

    rng = np.random.default_rng(seed)
    means = _signal_means(effect, n_symbols)
    n_rows = len(dates) * bars_per_session
    returns = np.zeros((n_rows, n_symbols), dtype=np.float64)
    row = 0
    for _date in dates:
        block_means = np.tile(means, (bars_per_session, 1))
        noise = rng.normal(0.0, sigma, size=(bars_per_session, n_symbols))
        returns[row : row + bars_per_session] = block_means + noise
        row += bars_per_session

    close = _close_from_log_returns(returns)
    volume = np.tile(np.array(volume_profile, dtype=np.float64), (n_rows, 1))
    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _seven_criterion_year_months() -> dict[int, list[int]]:
    """8 years total (2018-2025); only 2024/2025 are calendar-complete.

    Sign stability (criterion 2) is measured over all 8 years; the recent-years
    cost gate (criterion 7) only requires >= 2 *complete* years, so 2024/2025
    having every month present while the other 6 years have only Jan/Jun makes
    criterion 7 evaluate (not NOT_EVALUATED) without inflating row count 6x.
    """
    year_months = {year: [1, 6] for year in range(2018, 2026)}
    year_months[2024] = list(range(1, 13))
    year_months[2025] = list(range(1, 13))
    return year_months


def _base_seven_criterion_panel() -> Panel:
    return _build_panel(
        _seven_criterion_year_months(),
        bars_per_session=375,  # full 9:15-15:30 session: spans open/mid/close
        effect=0.0015,
        sigma=0.0001,
        seed=7,
    )


_STRATEGY_RETURNS_STRONG_POSITIVE = np.random.default_rng(1).normal(0.02, 0.001, size=500)


def _call_verdict(
    panel: Panel,
    *,
    hypothesis_id: str,
    latency_profile: dict[int, float] | None,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = None,
    universe: tuple[str, ...] | None = None,
    seed: int = 0,
) -> HypothesisVerdict:
    lens = Lens(panel, universe=universe, seed=seed)
    return lens.verdict(
        hypothesis_id,
        "return_1",
        1,
        latency_profile=latency_profile,
        effective_n_trials=effective_n_trials,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=5,
        n_boot=100,
    )


def _all_pass_verdict() -> HypothesisVerdict:
    """All seven criteria PASS and all evaluated.

    Verified against current (pre-fix) code: reasons = 7x PASS, survived=True,
    any_not_evaluated=False.
    """
    panel = _base_seven_criterion_panel()
    return _call_verdict(
        panel,
        hypothesis_id="H_ALL_PASS",
        latency_profile={0: 1.0, 1: 0.7, 2: 0.65},
        effective_n_trials=2,
        strategy_returns=_STRATEGY_RETURNS_STRONG_POSITIVE,
    )


def _one_not_evaluated_rest_pass_verdict() -> HypothesisVerdict:
    """Criterion 5 NOT_EVALUATED (no latency_profile), the other six PASS.

    Verified against current code: reasons = 6x PASS + 1x NOT_EVALUATED,
    any_not_evaluated=True, and (this is the L1 defect) survived=True.
    """
    panel = _base_seven_criterion_panel()
    return _call_verdict(
        panel,
        hypothesis_id="H_ONE_NOT_EVALUATED",
        latency_profile=None,
        effective_n_trials=2,
        strategy_returns=_STRATEGY_RETURNS_STRONG_POSITIVE,
    )


def _one_fail_one_not_evaluated_verdict() -> HypothesisVerdict:
    """Criterion 5 FAILs (retention < 0.5), criterion 6 NOT_EVALUATED

    (no strategy_returns), the other five PASS. Verified against current
    code: exactly one FAIL, one NOT_EVALUATED, five PASS; survived=False,
    any_not_evaluated=True.
    """
    panel = _base_seven_criterion_panel()
    return _call_verdict(
        panel,
        hypothesis_id="H_FAIL_PLUS_NOT_EVALUATED",
        latency_profile={0: 1.0, 1: 0.1, 2: 0.05},
        effective_n_trials=1,
        strategy_returns=None,
    )


def _all_evaluated_one_fail_verdict() -> HypothesisVerdict:
    """All seven criteria evaluated; criterion 5 FAILs, the other six PASS.

    Verified against current code: exactly one FAIL, no NOT_EVALUATED;
    survived=False, any_not_evaluated=False.
    """
    panel = _base_seven_criterion_panel()
    return _call_verdict(
        panel,
        hypothesis_id="H_ALL_EVALUATED_ONE_FAIL",
        latency_profile={0: 1.0, 1: 0.1, 2: 0.05},
        effective_n_trials=2,
        strategy_returns=_STRATEGY_RETURNS_STRONG_POSITIVE,
    )


# ---------------------------------------------------------------------------
# Obligation 1: all seven PASS and evaluated -> SURVIVED, survived True.
# ---------------------------------------------------------------------------


def test_ob1_all_seven_pass_and_evaluated_gives_survived_outcome() -> None:
    verdict = _all_pass_verdict()

    assert all("FAIL" not in r for r in verdict.reasons)
    assert all("NOT_EVALUATED" not in r for r in verdict.reasons)
    assert len(verdict.reasons) == 7
    assert verdict.any_not_evaluated is False

    assert verdict.outcome == "SURVIVED"
    assert verdict.survived is True


# ---------------------------------------------------------------------------
# Obligation 2 (THE L1 REGRESSION TEST, most important in this suite): one
# criterion NOT_EVALUATED, rest PASS -> INCONCLUSIVE, survived MUST be False.
# ---------------------------------------------------------------------------


def test_ob2_one_not_evaluated_rest_pass_gives_inconclusive_and_survived_false() -> None:
    verdict = _one_not_evaluated_rest_pass_verdict()

    assert "NOT_EVALUATED" in verdict.reasons[4]
    assert sum("NOT_EVALUATED" in r for r in verdict.reasons) == 1
    assert sum("FAIL" in r for r in verdict.reasons) == 0
    assert verdict.any_not_evaluated is True

    # This is the defect: current code drops the unevaluated criterion from
    # the conjunction and reports survived=True with 1 of 7 criteria unrun.
    assert verdict.outcome == "INCONCLUSIVE"
    assert verdict.survived is False


# ---------------------------------------------------------------------------
# Obligation 3: one FAIL plus one NOT_EVALUATED -> KILLED (KILLED beats
# INCONCLUSIVE -- a failure that WAS evaluated is decisive).
# ---------------------------------------------------------------------------


def test_ob3_one_fail_one_not_evaluated_gives_killed_not_inconclusive() -> None:
    verdict = _one_fail_one_not_evaluated_verdict()

    assert sum("FAIL" in r and "NOT_EVALUATED" not in r for r in verdict.reasons) == 1
    assert sum("NOT_EVALUATED" in r for r in verdict.reasons) == 1
    assert verdict.any_not_evaluated is True

    assert verdict.outcome == "KILLED"
    assert verdict.survived is False


# ---------------------------------------------------------------------------
# Obligation 4: `survived` is exactly `outcome == SURVIVED`, for every
# combination -- assert the property itself, not a reimplementation of it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_verdict",
    [
        _all_pass_verdict,
        _one_not_evaluated_rest_pass_verdict,
        _one_fail_one_not_evaluated_verdict,
        _all_evaluated_one_fail_verdict,
    ],
    ids=[
        "all_pass_survived",
        "one_not_evaluated_inconclusive",
        "fail_plus_not_evaluated_killed",
        "all_evaluated_one_fail_killed",
    ],
)
def test_ob4_survived_property_matches_outcome_survived_for_every_combination(
    make_verdict: object,
) -> None:
    verdict = make_verdict()  # type: ignore[operator]
    assert verdict.survived == (verdict.outcome == "SURVIVED")


# ---------------------------------------------------------------------------
# Obligation 5 (THE L3 REGRESSION TEST, second-most important): criterion 5
# with a NEGATIVE lag_0_edge that retains magnitude and sign at lags 1/2
# PASSES. Uses H2's actual sign: lag_0_edge = -24.30.
# ---------------------------------------------------------------------------


def _small_criterion5_panel() -> Panel:
    return _build_panel(
        {2024: [1]}, bars_per_session=50, effect=0.001, sigma=0.0005, seed=1
    )


def test_ob5_criterion5_negative_edge_retaining_magnitude_and_sign_passes() -> None:
    panel = _small_criterion5_panel()
    # retention at lag 1: |-14.58| / |-24.30| = 0.60 >= 0.5, same sign.
    # retention at lag 2: |-13.365| / |-24.30| = 0.55 >= 0.5, same sign.
    latency_profile = {0: -24.30, 1: -14.58, 2: -13.365}
    verdict = _call_verdict(
        panel, hypothesis_id="H2_reversal", latency_profile=latency_profile
    )

    assert "PASS" in verdict.reasons[4]
    assert "FAIL" not in verdict.reasons[4]


# ---------------------------------------------------------------------------
# Obligation 6: criterion 5 with an edge that retains magnitude but FLIPS
# sign at lag 1 FAILS.
# ---------------------------------------------------------------------------


def test_ob6_criterion5_magnitude_retained_but_sign_flips_at_lag1_fails() -> None:
    panel = _small_criterion5_panel()
    # |14.58| / |-24.30| = 0.60 >= 0.5 (magnitude retained) but the sign
    # flipped from - to +: this is not a surviving edge.
    latency_profile = {0: -24.30, 1: 14.58, 2: -13.365}
    verdict = _call_verdict(
        panel, hypothesis_id="H_sign_flip", latency_profile=latency_profile
    )

    assert "FAIL" in verdict.reasons[4]
    assert "PASS" not in verdict.reasons[4]


# ---------------------------------------------------------------------------
# Obligation 7: criterion 5 with lag_0_edge == 0 FAILS with a reason
# distinguishable from ordinary decay.
# ---------------------------------------------------------------------------


def test_ob7_criterion5_zero_lag0_edge_fails_with_distinguishable_reason() -> None:
    panel = _small_criterion5_panel()

    zero_edge_verdict = _call_verdict(
        panel,
        hypothesis_id="H_zero_edge",
        latency_profile={0: 0.0, 1: 0.0, 2: 0.0},
    )
    decayed_verdict = _call_verdict(
        panel,
        hypothesis_id="H_decayed",
        # A genuine (nonzero) edge that decays below the retention threshold:
        # ordinary decay, NOT "no edge to retain".
        latency_profile={0: 24.30, 1: 2.0, 2: 2.0},
    )

    zero_edge_reason = zero_edge_verdict.reasons[4]
    decayed_reason = decayed_verdict.reasons[4]

    assert "FAIL" in zero_edge_reason
    assert "FAIL" in decayed_reason

    # "There is no edge to retain" is a different finding from "the edge
    # decayed" (spec, L3): the zero-edge reason must name that distinctly,
    # e.g. via a "no edge"/"zero" marker the ordinary-decay reason lacks.
    zero_edge_lower = zero_edge_reason.lower()
    decayed_lower = decayed_reason.lower()
    zero_edge_markers = ("no edge", "zero edge", "undefined", "no retention")
    assert any(marker in zero_edge_lower for marker in zero_edge_markers)
    assert not any(marker in decayed_lower for marker in zero_edge_markers)


# ---------------------------------------------------------------------------
# Obligation 8: criterion 5's reported result carries an UNCALIBRATED marker
# for its (currently undocumented, rule-8-violating) threshold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latency_profile",
    [
        {0: -24.30, 1: -14.58, 2: -13.365},  # would PASS once L3 is fixed
        {0: -24.30, 1: 14.58, 2: -13.365},  # FAILs (sign flip)
        {0: 0.0, 1: 0.0, 2: 0.0},  # FAILs (no edge)
    ],
    ids=["pass_case", "sign_flip_fail_case", "zero_edge_fail_case"],
)
def test_ob8_criterion5_reason_carries_uncalibrated_threshold_marker(
    latency_profile: dict[int, float],
) -> None:
    panel = _small_criterion5_panel()
    verdict = _call_verdict(
        panel, hypothesis_id="H_uncalibrated", latency_profile=latency_profile
    )

    assert "UNCALIBRATED" in verdict.reasons[4]


# ---------------------------------------------------------------------------
# Obligation 9 (THE L4 REGRESSION TEST): a Lens constructed with a restricted
# universe computes its statistics on the restricted symbol set. Fixture is
# built so that including the excluded symbol WOULD visibly change the
# answer -- proven below by an independent direct check, not asserted
# tautologically.
# ---------------------------------------------------------------------------


_UNIVERSE_SYMBOLS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
_UNIVERSE_VOLUME = (2e3, 2e4, 2e5, 2e6, 2e7, 2e8)
# Excludes Q6, the single largest-magnitude (+edge) symbol. 5 symbols remain,
# meeting cross_sectional_rank's min_names=5 floor (AMENDMENT 2) -- restricting
# below that floor makes every row all-NaN and degrades the comparison to
# "a working answer versus no answer", which proves nothing about restriction.
_RESTRICTED_UNIVERSE = ("Q1", "Q2", "Q3", "Q4", "Q5")


def _universe_test_panel() -> Panel:
    # Symbol means by construction: Q1=-0.01, Q2=-0.006, Q3=-0.002, Q4=0.002,
    # Q5=0.006, Q6=+0.01 (the single largest-magnitude, positive edge). Six
    # symbols so that restricting the universe to exclude Q6 still leaves 5 --
    # >= cross_sectional_rank's min_names=5 (AMENDMENT 2) -- while still
    # removing the extreme positive-return name from the cross-sectional
    # ranking used by expectancy(), changing which symbols populate each rank
    # bucket and spread_bps.
    return _build_panel(
        {2024: [1, 6]},
        bars_per_session=250,
        effect=0.01,
        sigma=0.0005,
        symbols=_UNIVERSE_SYMBOLS,
        volume_profile=_UNIVERSE_VOLUME,
        seed=11,
    )


def test_ob9_universe_restriction_computes_statistics_on_restricted_set() -> None:
    panel = _universe_test_panel()

    full_lens = Lens(panel, seed=0)
    restricted_lens = Lens(panel, universe=_RESTRICTED_UNIVERSE, seed=0)

    full_exp = full_lens.expectancy("return_1", 1, n_buckets=5, method="cross_sectional_rank")
    restricted_exp = restricted_lens.expectancy(
        "return_1", 1, n_buckets=5, method="cross_sectional_rank"
    )

    # Non-vacuousness check: dropping Q6 from the ranking is NOT a no-op.
    # Confirmed directly (independently of Lens.universe wiring) that a panel
    # built with these six symbols yields a materially different spread_bps
    # once restricted to the 5 named above (~159.9 bps) vs the full six
    # (~179.9 bps) -- i.e. a correct restriction must change the observed
    # answer here. This is the effect check; it is the stronger of the two
    # assertions in this test because it demonstrates the restriction was
    # actually applied to the statistics, not merely recorded.
    assert restricted_exp.spread_bps != full_exp.spread_bps

    # n_total is a ROW count (expectancy.py: fwd.n_defined), not a symbol
    # count -- dropping one of six always-present symbols leaves every row
    # still defined, so n_total is correctly unchanged and must NOT be used
    # here (AMENDMENT 2). The symbol count itself is only recorded on
    # HypothesisVerdict as n_symbols_used (obligation 10's field), so confirm
    # the restriction's effect on symbol count via a verdict() call instead.
    full_verdict = _call_verdict(
        panel, hypothesis_id="H_ob9_full", latency_profile={0: 1.0, 1: 0.6, 2: 0.55}
    )
    restricted_verdict = _call_verdict(
        panel,
        hypothesis_id="H_ob9_restricted",
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        universe=_RESTRICTED_UNIVERSE,
    )
    assert restricted_verdict.n_symbols_used < full_verdict.n_symbols_used


# ---------------------------------------------------------------------------
# Obligation 10: the verdict records the symbol count actually used.
# ---------------------------------------------------------------------------


def test_ob10_verdict_records_symbol_count_actually_used() -> None:
    panel = _universe_test_panel()

    full_verdict = _call_verdict(
        panel,
        hypothesis_id="H_full_universe",
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
    )
    restricted_verdict = _call_verdict(
        panel,
        hypothesis_id="H_restricted_universe",
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        universe=_RESTRICTED_UNIVERSE,
    )

    assert full_verdict.n_symbols_used == 6
    assert restricted_verdict.n_symbols_used == 5
