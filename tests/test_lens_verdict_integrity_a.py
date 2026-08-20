"""Independent test suite for Lens verdict integrity (L1, L3, L4 fixes).

Tests are expected to RED against current code. Written from specs/lens_verdict_integrity.md.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nifty_quant.data.panel import Panel
from nifty_quant.research.lens import HypothesisVerdict, Lens

_IST = ZoneInfo("Asia/Kolkata")


def _session_grid(
    dates: list[dt.date], bars_per_session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build timestamp grid for a panel."""
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


def _close_prices_from_log_returns(returns: np.ndarray) -> np.ndarray:
    """Compute close prices from log returns."""
    cum = np.cumsum(returns, axis=0)
    log_close = np.vstack([np.zeros((1, returns.shape[1]), dtype=np.float64), cum])
    close_all = np.exp(log_close)
    return close_all[:-1, :]


def _build_panel(
    dates: list[dt.date],
    symbols: tuple[str, ...],
    bars_per_session: int = 30,
    effect: float = 0.01,
    sigma: float = 0.0005,
    seed: int = 0,
) -> Panel:
    """Build a minimal test panel with constant signal across symbols."""
    n_rows = len(dates) * bars_per_session
    n_symbols = len(symbols)
    rng = np.random.default_rng(seed)

    # Create returns with uniform positive effect across all symbols
    means = np.full((n_symbols,), effect, dtype=np.float64)
    returns = np.zeros((n_rows, n_symbols), dtype=np.float64)

    row = 0
    for _ in dates:
        noise = rng.normal(0.0, sigma, size=(bars_per_session, n_symbols))
        returns[row : row + bars_per_session] = means + noise
        row += bars_per_session

    close = _close_prices_from_log_returns(returns)
    volume = np.full((n_rows, n_symbols), 1e5, dtype=np.float32)

    ts, day_offsets, dates_arr = _session_grid(dates, bars_per_session)
    return Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )


def _build_test_panel(
    symbols: tuple[str, ...] = ("S00", "S01", "S02"),
    n_sessions: int = 5,
) -> Panel:
    """Build a basic test panel."""
    dates = [dt.date(2024, 1, 2 + i) for i in range(n_sessions)]
    return _build_panel(dates, symbols, bars_per_session=30, effect=0.01, sigma=0.0005)


def _call_verdict(
    panel: Panel,
    hypothesis_id: str = "TEST_HYP",
    latency_profile: dict[int, float] | None = None,
    effective_n_trials: int = 1,
    strategy_returns: np.ndarray | None = None,
) -> HypothesisVerdict:
    """Call Lens.verdict with sensible defaults for testing."""
    lens = Lens(panel, seed=0)
    return lens.verdict(
        hypothesis_id,
        "return_1",
        1,
        latency_profile=latency_profile,
        effective_n_trials=effective_n_trials,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=3,
        n_boot=100,
    )


# ============================================================================
# Obligation 1: All criteria PASS and all evaluated -> outcome SURVIVED, survived True
# ============================================================================


def test_obligation_1_all_criteria_pass_all_evaluated() -> None:
    """All seven criteria PASS and all evaluated -> outcome == SURVIVED, survived is True."""
    panel = _build_test_panel()

    # Provide all criteria inputs so none are NOT_EVALUATED
    latency_profile = {0: 1.0, 1: 0.6, 2: 0.55}  # Will pass criterion 5 (>= 0.5 retention)
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=latency_profile,
        effective_n_trials=2,  # >= 2 for criterion 6
        strategy_returns=strategy_returns,
    )

    # After fix, verdict should have outcome attribute; AttributeError is the signal
    assert verdict.outcome == "SURVIVED"
    assert verdict.survived is True


# ============================================================================
# Obligation 2: One criterion NOT_EVALUATED, rest PASS -> outcome INCONCLUSIVE, survived False
# This is the L1 regression test and MOST IMPORTANT test.
# ============================================================================


def test_obligation_2_one_not_evaluated_rest_pass() -> None:
    """One criterion NOT_EVALUATED, rest PASS -> outcome == INCONCLUSIVE, survived is False.

    This is the L1 regression test: survived must not be True when a criterion is unrun.
    """
    panel = _build_test_panel()

    # Provide ALL criteria inputs EXCEPT latency_profile (criterion 5)
    # so criterion 5 will be NOT_EVALUATED but others can pass
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=None,  # criterion 5 will be NOT_EVALUATED
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    # After fix, verdict should have outcome attribute; AttributeError is the signal
    assert verdict.outcome == "INCONCLUSIVE"
    assert verdict.survived is False
    assert verdict.any_not_evaluated is True


# ============================================================================
# Obligation 3: One criterion FAIL + one NOT_EVALUATED -> outcome KILLED (kills beat inconclusive)
# ============================================================================


def test_obligation_3_one_fail_one_not_evaluated_outcome_killed() -> None:
    """One criterion FAIL + one NOT_EVALUATED -> outcome == KILLED. KILLED beats INCONCLUSIVE."""
    panel = _build_test_panel(n_sessions=2)  # Smaller for easier failure

    # Provide few inputs and a strategy_returns that will likely fail criterion 6 (Sharpe)
    strategy_returns = np.random.RandomState(42).normal(-0.005, 0.001, size=100)  # Negative mean

    verdict = _call_verdict(
        panel,
        latency_profile=None,  # criterion 5 NOT_EVALUATED
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    # outcome should be KILLED because at least one criterion FAILed
    # KILLED takes precedence over INCONCLUSIVE
    assert verdict.outcome == "KILLED"


# ============================================================================
# Obligation 4: survived property is exactly outcome == SURVIVED
# ============================================================================


def test_obligation_4_survived_property_is_outcome_survived() -> None:
    """survived property must be exactly outcome == SURVIVED for all cases."""
    # Case 1: SURVIVED outcome
    panel1 = _build_test_panel()
    latency_profile = {0: 1.0, 1: 0.6, 2: 0.55}
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)
    verdict1 = _call_verdict(
        panel1,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )
    assert verdict1.survived == (verdict1.outcome == "SURVIVED")

    # Case 2: INCONCLUSIVE outcome
    panel2 = _build_test_panel()
    verdict2 = _call_verdict(
        panel2,
        latency_profile=None,  # NOT_EVALUATED criterion
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )
    assert verdict2.survived == (verdict2.outcome == "SURVIVED")


# ============================================================================
# Obligation 5: Criterion 5 with NEGATIVE lag_0_edge retains magnitude and sign -> PASS
# This is the L3 regression test. Use H2's actual edge: -24.30 bps
# ============================================================================


def test_obligation_5_negative_edge_retains_magnitude_and_sign_passes() -> None:
    """Criterion 5 with negative lag_0_edge that retains magnitude and sign -> PASS.

    This is the L3 regression test. H2's real edge is -24.30 bps, negative by construction
    because it is a reversal signal traded short-side. The criterion must not reject it for
    having the sign it is supposed to have.
    """
    panel = _build_test_panel()

    # H2's actual sign: negative edge that retains magnitude and sign at lags 1 and 2
    # lag_0_edge = -24.30 bps
    # Use multiplier relative to lag_0: retain >= 50%
    h2_lag_0_edge = -24.30  # negative, as per spec
    h2_lag_1_edge = -14.58  # 60% of magnitude (14.58 / 24.30 = 0.60), same sign
    h2_lag_2_edge = -12.15  # 50% of magnitude (12.15 / 24.30 = 0.50), same sign

    latency_profile = {
        0: h2_lag_0_edge,
        1: h2_lag_1_edge,
        2: h2_lag_2_edge,
    }

    # Provide other criteria inputs to avoid NOT_EVALUATED
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    # Criterion 5 should PASS for this profile
    c5_reason = verdict.reasons[4]  # Criterion 5 is the 5th reason (0-indexed)
    assert "PASS" in c5_reason, (
        f"Criterion 5 should PASS for negative edge with retained magnitude and sign. "
        f"Reason: {c5_reason}"
    )


# ============================================================================
# Obligation 6: Criterion 5 with edge that retains magnitude but FLIPS sign at lag 1 -> FAIL
# ============================================================================


def test_obligation_6_edge_flips_sign_fails() -> None:
    """Criterion 5 with edge that retains magnitude but FLIPS sign at lag 1 -> FAIL.

    NOTE: This test is GREEN on current code for the wrong reason. Current code FAILs
    *any* negative lag_0_edge (the L3 defect), so it happens to produce the required
    outcome via an over-broad rule. It is kept as a permanent regression test: once L3
    is fixed to compute retention on magnitude, a partial fix that accepts sign flips
    would newly break this test, making it a detector of that incomplete fix.
    """
    panel = _build_test_panel()

    # Edge flips sign at lag 1
    latency_profile = {
        0: 24.30,   # positive at lag 0
        1: -14.58,  # flipped sign at lag 1
        2: 12.15,   # same magnitude
    }

    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    c5_reason = verdict.reasons[4]
    assert "FAIL" in c5_reason


# ============================================================================
# Obligation 7: Criterion 5 with lag_0_edge == 0 -> FAIL with distinguishable reason
# ============================================================================


def test_obligation_7_zero_edge_fails_with_distinct_reason() -> None:
    """Criterion 5 with lag_0_edge == 0 FAILS with a reason distinct from ordinary decay."""
    panel = _build_test_panel()

    latency_profile = {
        0: 0.0,     # no edge
        1: 0.0,
        2: 0.0,
    }

    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    c5_reason = verdict.reasons[4]
    assert "FAIL" in c5_reason, f"Criterion 5 should FAIL for zero edge. Reason: {c5_reason}"
    # The reason should distinguish zero edge from ordinary decay
    has_zero_marker = (
        "zero" in c5_reason.lower()
        or "no edge" in c5_reason.lower()
        or "non-finite" in c5_reason.lower()
    )
    assert has_zero_marker, (
        f"Reason for zero edge FAIL should be distinct from ordinary decay. "
        f"Got: {c5_reason}"
    )


# ============================================================================
# Obligation 8: Criterion 5 result carries UNCALIBRATED marker for threshold
# ============================================================================


def test_obligation_8_criterion_5_uncalibrated_marker() -> None:
    """Criterion 5's reported result carries an UNCALIBRATED marker for its threshold."""
    panel = _build_test_panel()

    latency_profile = {0: 1.0, 1: 0.6, 2: 0.55}
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    verdict = _call_verdict(
        panel,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
    )

    c5_reason = verdict.reasons[4]
    assert "UNCALIBRATED" in c5_reason, (
        f"Criterion 5 reason should carry UNCALIBRATED marker since 0.5 threshold is unmeasured. "
        f"Got: {c5_reason}"
    )


# ============================================================================
# Obligation 9: Universe restriction computes on restricted symbol set (non-vacuous test)
# ============================================================================


def test_obligation_9_universe_restriction_non_vacuous() -> None:
    """Universe restriction must actually restrict symbols used in verdict computation.

    Non-vacuous test: a symbol excluded by universe WOULD visibly change the answer if included.
    """
    dates = [dt.date(2024, 1, 2 + i) for i in range(5)]
    symbols = ("S00", "S01", "S02", "S03", "S04")

    # Build panel where S00 is an outlier (strong positive effect) and S04 is opposite (negative)
    # This way, including vs excluding S00 will materially change the cross-sectional mean return
    n_rows = len(dates) * 30
    n_symbols = len(symbols)
    rng = np.random.default_rng(0)

    returns = np.zeros((n_rows, n_symbols), dtype=np.float64)
    # S00: strong positive effect (0.02)
    # S01-S03: neutral (0.0)
    # S04: strong negative effect (-0.02)
    means = np.array([0.02, 0.0, 0.0, 0.0, -0.02], dtype=np.float64)

    row = 0
    for _ in dates:
        noise = rng.normal(0.0, 0.0005, size=(30, n_symbols))
        returns[row : row + 30] = means + noise
        row += 30

    close = _close_prices_from_log_returns(returns)
    volume = np.full((n_rows, n_symbols), 1e5, dtype=np.float32)

    ts, day_offsets, dates_arr = _session_grid(dates, 30)
    panel_full = Panel(
        fields={"close": close.astype(np.float32), "volume": volume.astype(np.float32)},
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=dates_arr,
    )

    # Verdict on full universe (S00-S04): S00 is strong positive outlier
    latency_profile = {0: 1.0, 1: 0.6, 2: 0.55}
    strategy_returns = np.random.RandomState(0).normal(0.001, 0.005, size=100)

    lens_full = Lens(panel_full, universe=None, seed=0)
    verdict_full = lens_full.verdict(
        "TEST_FULL",
        "return_1",
        1,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=3,
        n_boot=100,
    )

    # Verdict on restricted universe (S01-S03, excluding outliers): closer to mean 0
    lens_restricted = Lens(panel_full, universe=("S01", "S02", "S03"), seed=0)
    verdict_restricted = lens_restricted.verdict(
        "TEST_RESTRICTED",
        "return_1",
        1,
        latency_profile=latency_profile,
        effective_n_trials=2,
        strategy_returns=strategy_returns,
        method="cross_sectional_rank",
        n_buckets=3,
        n_boot=100,
    )

    # The expectancy buckets should differ because universe restriction removes outliers
    # We check the expectancy table's mean_bps in at least one bucket
    assert len(verdict_full.expectancy.buckets) > 0, "Full verdict should have buckets"
    assert len(verdict_restricted.expectancy.buckets) > 0, "Restricted verdict should have buckets"

    full_mean = verdict_full.expectancy.buckets[0].mean_bps
    restricted_mean = verdict_restricted.expectancy.buckets[0].mean_bps

    # If universe restriction is working, the means should differ
    # Full includes outliers; restricted excludes them
    assert full_mean != restricted_mean, (
        f"Universe restriction should change expectancy stats. "
        f"Full mean: {full_mean}, Restricted mean: {restricted_mean}. "
        f"If they're equal, restriction is not working (L4 defect)."
    )


# ============================================================================
# Obligation 10: Verdict records the symbol count actually used
# ============================================================================


def test_obligation_10_verdict_records_symbol_count() -> None:
    """Verdict records the symbol count actually used."""
    dates = [dt.date(2024, 1, 2 + i) for i in range(5)]
    symbols = ("S00", "S01", "S02", "S03", "S04")
    panel = _build_panel(dates, symbols)

    # Full universe: 5 symbols
    lens_full = Lens(panel, universe=None, seed=0)
    verdict_full = lens_full.verdict(
        "TEST_FULL",
        "return_1",
        1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=2,
        strategy_returns=np.random.RandomState(0).normal(0.001, 0.005, size=100),
        method="cross_sectional_rank",
        n_buckets=3,
        n_boot=100,
    )

    # Restricted universe: 3 symbols
    lens_restricted = Lens(panel, universe=("S01", "S02", "S03"), seed=0)
    verdict_restricted = lens_restricted.verdict(
        "TEST_RESTRICTED",
        "return_1",
        1,
        latency_profile={0: 1.0, 1: 0.6, 2: 0.55},
        effective_n_trials=2,
        strategy_returns=np.random.RandomState(0).normal(0.001, 0.005, size=100),
        method="cross_sectional_rank",
        n_buckets=3,
        n_boot=100,
    )

    # After fix, verdict should have n_symbols_used attribute; AttributeError is signal
    assert verdict_full.n_symbols_used == 5
    assert verdict_restricted.n_symbols_used == 3
