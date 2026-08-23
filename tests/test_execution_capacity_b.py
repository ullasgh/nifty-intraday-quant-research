"""Suite B for spec H (execution_capacity.md), written from the spec ALONE.

Rule 1: this suite is independent of suite A -- neither agent saw the other's
tests, and neither saw an implementation (H1's ladder and H3's turnover-based
capacity are greenfield; H2's crossover constant is "nowhere named in code").

Import strategy: costs.py and fills.py already exist and are claimed correct
by the spec's own "Starting position" section, so tests against them are
imported at module level and may legitimately PASS (they exercise pre-existing
NSE cost-stack code, not this phase's deliverable). The genuinely new pieces
this phase must deliver -- a cost x participation ladder function, a named
crossover constant, and a turnover-based capacity function -- do not exist
anywhere in the tree yet, so those are imported LAZILY inside each test body.
That keeps a missing module a single RED test failure, not a whole-file
collection ERROR that would mask every other obligation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nifty_quant.execution.costs import (
    FillBatch,
    FixedBpsCost,
    NSEIntradayEquityCosts,
    ZeroCost,
)
from nifty_quant.execution.fills import FillModel, ZeroSlippage

# ---------------------------------------------------------------------------
# Obligation 1 + H2: the Rs 66,666.67 crossover regression.
# ---------------------------------------------------------------------------

# Spec H2 pins these five measured values verbatim (lines 47, 76-78). This is
# the one place a literal constant is legitimate under rule 8: it is not a
# hand-chosen threshold, it is a recorded MEASUREMENT the spec author already
# took against the real cost stack, and reproducing it here is exactly what a
# regression test is for.
_CROSSOVER_CASES = [
    (50_000.0, 10.6245),
    (66_666.0, 10.6245),
    (66_667.0, 10.6245),
    (100_000.0, 8.2645),
    (10_000_000.0, 3.5917),
]


@pytest.mark.parametrize("notional_per_leg, expected_bps", _CROSSOVER_CASES)
def test_round_trip_bps_matches_spec_measured_values(
    notional_per_leg: float, expected_bps: float
) -> None:
    """Catches: any change to the brokerage/STT/exchange/SEBI/IPFT/stamp/GST
    stack (or its composition) that shifts round_trip_bps away from the
    figures the spec explicitly measured and pinned. A regression that
    silently changed e.g. the GST base or the brokerage cap would move these
    numbers and this test would catch it; an implementation that returns a
    constant bps regardless of notional would fail the 100k/1cr rows.
    """
    model = NSEIntradayEquityCosts()
    got = model.round_trip_bps(notional_per_leg)
    assert got == pytest.approx(expected_bps, abs=5e-5)


def test_round_trip_bps_constant_below_crossover_decays_above() -> None:
    """Catches: an implementation that decays bps smoothly everywhere (no
    flat-brokerage regime) or one that is constant everywhere (no percentage
    regime) -- either would fail one half of this relative assertion. This is
    the RELATIVE-property form rule 8 asks for as an alternative to a bare
    literal: below the crossover, three genuinely different notionals must
    produce IDENTICAL bps; above it, bps must strictly DECREASE as notional
    grows.
    """
    model = NSEIntradayEquityCosts()
    below = [model.round_trip_bps(n) for n in (50_000.0, 60_000.0, 66_666.0)]
    above = [model.round_trip_bps(n) for n in (66_667.0, 100_000.0, 1_000_000.0, 10_000_000.0)]

    # Constant (to float tolerance) below the crossover.
    assert below[0] == pytest.approx(below[1], abs=5e-5)
    assert below[1] == pytest.approx(below[2], abs=5e-5)

    # Strictly decaying above the crossover -- a mutant that returns a flat
    # number post-crossover fails this monotonicity check even if it happens
    # to match one of the pinned literals above.
    for earlier, later in zip(above, above[1:]):
        assert later < earlier


def test_crossover_constant_is_derived_not_a_literal() -> None:
    """Obligation 2. Catches: hard-coding 66667 (or 66666.67) as a bare
    number anywhere in the ladder/crossover code instead of deriving it from
    the cost model's own brokerage_flat / brokerage_pct fields. We cannot
    grep the not-yet-written implementation from here, so this test instead
    demands that the crossover be exposed as a callable/derivation the
    project can point at that FOLLOWS the cost model, not the other way
    round: recomputing it from a DIFFERENT brokerage_pct must move the
    crossover proportionally rather than returning the old fixed number.
    A hard-coded 66667 would fail the second assertion below (it would not
    move when brokerage_pct changes).
    """
    from nifty_quant.execution.costs import brokerage_crossover_notional  # noqa: PLC0415

    model = NSEIntradayEquityCosts()
    derived = brokerage_crossover_notional(model)
    assert derived == pytest.approx(model.brokerage_flat / model.brokerage_pct, rel=1e-9)

    # A cost model with double the percentage rate must halve the crossover.
    # This is what "derived, not a literal" actually buys you: change the
    # input, the output moves. A literal 66667 baked into the ladder code
    # would not react to this at all.
    doubled_pct_model = NSEIntradayEquityCosts(brokerage_pct=model.brokerage_pct * 2.0)
    derived_doubled = brokerage_crossover_notional(doubled_pct_model)
    assert derived_doubled == pytest.approx(derived / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Obligation 3 + 4: GST base and STT/stamp sidedness.
# ---------------------------------------------------------------------------


def test_gst_base_excludes_stt_and_stamp_duty() -> None:
    """Obligation 3. Catches: a GST computation that (wrongly) folds STT or
    stamp duty into its base -- an easy mistake since GST is percentage-on-
    percentage. We assert the RELATION (gst == 18% of brokerage+exchange+sebi
    exactly, independent of stt/stamp) rather than a hand-picked number, so
    the check survives if the underlying rates in NSEIntradayEquityCosts are
    ever recalibrated.
    """
    model = NSEIntradayEquityCosts()
    fills = FillBatch(
        notional=np.array([100_000.0, 250_000.0], dtype=np.float64),
        is_buy=np.array([True, False], dtype=bool),
    )
    charges = model.charges(fills)

    expected_gst = model.gst_pct * (charges.brokerage + charges.exchange_txn + charges.sebi)
    assert np.allclose(charges.gst, expected_gst, rtol=1e-12)

    # A mutant that adds stt or stamp_duty into the GST base would move gst
    # away from this expectation while leaving stt/stamp_duty themselves
    # unaffected -- so also confirm those two are untouched by the GST calc.
    only_bs_sebi_gst = model.gst_pct * (charges.brokerage + charges.exchange_txn + charges.sebi)
    contaminated_gst = model.gst_pct * (
        charges.brokerage + charges.exchange_txn + charges.sebi + charges.stt + charges.stamp_duty
    )
    assert not np.allclose(only_bs_sebi_gst, contaminated_gst)
    assert np.allclose(charges.gst, only_bs_sebi_gst, rtol=1e-12)


def test_stt_sell_side_only_stamp_buy_side_only() -> None:
    """Obligation 4. Catches: STT charged on buys, or stamp duty charged on
    sells -- both would show up as a nonzero value on the wrong side of this
    two-row (buy, sell) fixture, which a "charge everything both ways" mutant
    would produce.
    """
    model = NSEIntradayEquityCosts()
    fills = FillBatch(
        notional=np.array([100_000.0, 100_000.0], dtype=np.float64),
        is_buy=np.array([True, False], dtype=bool),
    )
    charges = model.charges(fills)

    # Row 0 is a buy: STT must be exactly zero, stamp duty must be nonzero.
    assert charges.stt[0] == 0.0
    assert charges.stamp_duty[0] > 0.0

    # Row 1 is a sell: STT must be nonzero, stamp duty must be exactly zero.
    assert charges.stt[1] > 0.0
    assert charges.stamp_duty[1] == 0.0


# ---------------------------------------------------------------------------
# Obligation 8: FixedBpsCost(0.0) matches ZeroCost.
# ---------------------------------------------------------------------------


def test_fixed_bps_cost_zero_matches_zero_cost() -> None:
    """Obligation 8. Catches: FixedBpsCost carrying some residual charge at
    bps=0.0 (e.g. a floor or an off-by-one in the bps->fraction conversion)
    that would make the "0 bps column" of the H1 ladder silently non-zero,
    defeating the spec's stated purpose of that column (line 39: separating
    small-edge from large-cost diagnoses).
    """
    fills = FillBatch(
        notional=np.array([12_345.0, 987_654.0, 66_667.0], dtype=np.float64),
        is_buy=np.array([True, False, True], dtype=bool),
    )
    zero_charges = ZeroCost().charges(fills)
    fixed_zero_charges = FixedBpsCost(bps=0.0).charges(fills)

    assert np.array_equal(zero_charges.total, fixed_zero_charges.total)
    for field in ("brokerage", "stt", "exchange_txn", "sebi", "ipft", "stamp_duty", "gst"):
        assert np.array_equal(getattr(zero_charges, field), getattr(fixed_zero_charges, field))


# ---------------------------------------------------------------------------
# H1: the 7x5 cost x participation ladder -- greenfield, lazy-imported.
# ---------------------------------------------------------------------------

_COST_LADDER_BPS = (0, 2, 5, 8, 10, 15, 20)
_PARTICIPATION_LADDER = (0.005, 0.01, 0.02, 0.05, 0.10)


def _make_toy_strategy_inputs() -> dict:
    """A tiny, deliberately non-flat fixture: two symbols, two days, so a
    "return everything zeroed out" mutant is distinguishable from a working
    ladder. Rule 5: day boundaries are given as explicit offsets, never a
    fixed 375-bar assumption -- here the toy session is short on purpose
    (5 bars/day) precisely so no ladder implementation can get away with
    hard-coding 375.
    """
    rng = np.random.default_rng(0)
    n_days = 2
    bars_per_day = 5
    n_sym = 2
    n_rows = n_days * bars_per_day

    prices = 100.0 + rng.normal(0, 1, size=(n_rows, n_sym)).cumsum(axis=0)
    prices = np.abs(prices) + 50.0
    bar_traded_value = rng.uniform(5e5, 2e6, size=(n_rows, n_sym))
    gross_returns = rng.normal(0.0002, 0.001, size=n_rows).astype(np.float64)
    turnover = rng.uniform(0.05, 0.15, size=n_rows).astype(np.float64)
    day_offsets = np.arange(0, (n_days + 1) * bars_per_day, bars_per_day, dtype=np.int32)

    return {
        "prices": prices.astype(np.float64),
        "bar_traded_value": bar_traded_value.astype(np.float64),
        "gross_returns": gross_returns,
        "turnover": turnover,
        "day_offsets": day_offsets,
    }


def test_cost_participation_ladder_produces_35_cells_with_required_fields() -> None:
    """Obligation 5. Catches: a ladder builder that flattens the grid
    incorrectly (wrong cell count), or one that omits unfilled fraction from
    a cell -- both explicitly called out by the spec ("a cell missing the
    unfilled fraction fails", line 84). We do not assert exact numeric values
    here (no oracle exists for a greenfield function); we assert the SHAPE
    and PRESENCE-OF-FIELDS contract the spec obligates, which any correct
    implementation must satisfy regardless of its internals.
    """
    from nifty_quant.execution.capacity import cost_participation_ladder  # noqa: PLC0415

    inputs = _make_toy_strategy_inputs()
    ladder = cost_participation_ladder(
        cost_bps_grid=_COST_LADDER_BPS,
        participation_grid=_PARTICIPATION_LADDER,
        **inputs,
    )

    cells = list(ladder)
    assert len(cells) == len(_COST_LADDER_BPS) * len(_PARTICIPATION_LADDER)

    for cell in cells:
        # Each cell must carry all four spec-named quantities (line 83).
        assert hasattr(cell, "net_sharpe")
        assert hasattr(cell, "net_bps_per_day") or hasattr(cell, "net_bps_day")
        assert hasattr(cell, "turnover")
        assert hasattr(cell, "unfilled_fraction")
        # unfilled_fraction is a fraction, not a footnote value like -1/None
        # standing in for "not computed".
        assert 0.0 <= cell.unfilled_fraction <= 1.0
        assert math.isfinite(cell.net_sharpe)


def test_zero_bps_column_isolates_edge_from_cost() -> None:
    """Catches: a ladder that conflates the 0 bps column with any other
    column, or one that recomputes gross-vs-net incorrectly so the 0 bps
    column does not actually equal the gross (cost-free) result. Spec line
    39: "the 0 bps column is not padding" -- it must equal the gross-only
    Sharpe at every participation level, since cost is genuinely zero there.
    """
    from nifty_quant.execution.capacity import cost_participation_ladder  # noqa: PLC0415

    inputs = _make_toy_strategy_inputs()
    ladder = cost_participation_ladder(
        cost_bps_grid=(0, 20),
        participation_grid=(0.01,),
        **inputs,
    )
    cells = {(c.cost_bps, c.participation): c for c in ladder}

    zero_cost_cell = cells[(0, 0.01)]
    priced_cell = cells[(20, 0.01)]

    # At the same participation level, a strictly positive cost must not
    # IMPROVE net Sharpe relative to the zero-cost column -- a mutant that
    # inverted the cost sign, or that ignored the cost axis entirely and
    # returned identical Sharpe for every column, would fail one of these.
    assert priced_cell.net_sharpe <= zero_cost_cell.net_sharpe
    assert (
        priced_cell.net_bps_per_day != zero_cost_cell.net_bps_per_day
        or priced_cell.turnover == 0.0
    )


def test_starved_participation_raises_unfilled_and_reports_net_sharpe_alongside() -> None:
    """Obligation 6. Catches: an implementation that drops rows/cells where
    fills are starved (silently improving the reported average by excluding
    the worst cells), or one that reports unfilled fraction but not net
    Sharpe for the same cell (or vice versa) -- the spec explicitly forbids
    "silently improved by dropping the unfilled part" (line 86). We compare
    a starved (very low max_participation) run against a generous one on the
    IDENTICAL fixture and assert the RELATIVE direction only, per rule 8/9:
    no absolute unfilled-fraction threshold is asserted, only that starving
    participation must not decrease the reported unfilled fraction.
    """
    from nifty_quant.execution.capacity import cost_participation_ladder  # noqa: PLC0415

    inputs = _make_toy_strategy_inputs()
    starved = list(
        cost_participation_ladder(
            cost_bps_grid=(5,),
            participation_grid=(0.005,),
            **inputs,
        )
    )[0]
    generous = list(
        cost_participation_ladder(
            cost_bps_grid=(5,),
            participation_grid=(0.10,),
            **inputs,
        )
    )[0]

    assert starved.unfilled_fraction >= generous.unfilled_fraction
    # Both cells must carry a finite net Sharpe regardless of how starved the
    # fill is -- a NaN/omitted Sharpe on the starved cell is exactly the
    # "silently improved by dropping" failure mode the spec warns about.
    assert math.isfinite(starved.net_sharpe)
    assert math.isfinite(generous.net_sharpe)


# ---------------------------------------------------------------------------
# H3: capacity derived from turnover vs. the existing ADV-based figure.
# ---------------------------------------------------------------------------


def test_turnover_based_capacity_is_lower_than_adv_based_capacity() -> None:
    """Obligation 7. Catches: a capacity function that ignores turnover and
    just reproduces (or exceeds) the ADV-based figure -- the whole point of
    H3 is that turnover is the BINDING constraint at ~11% daily turnover, so
    ADV alone overstates. This is a relative assertion (rule 8): we do not
    assert an absolute rupee figure (no measured null exists to pin one),
    only the ORDERING the spec itself states must hold.
    """
    from nifty_quant.execution.capacity import (  # noqa: PLC0415
        capacity_from_adv,
        capacity_from_turnover,
    )

    median_adv = 50_00_00_000.0  # Rs 50 crore, a representative single-name ADV
    daily_turnover_fraction = 0.11  # spec line 60: "turns its book over ~11% daily"
    max_participation = 0.02

    adv_capacity = capacity_from_adv(median_adv=median_adv, max_participation=max_participation)
    turnover_capacity = capacity_from_turnover(
        median_adv=median_adv,
        max_participation=max_participation,
        daily_turnover_fraction=daily_turnover_fraction,
    )

    assert turnover_capacity < adv_capacity


def test_turnover_based_capacity_scales_inversely_with_turnover_fraction() -> None:
    """Catches: a turnover-based capacity function that is insensitive to the
    turnover input (e.g. it silently falls back to the ADV heuristic and
    ignores the argument) -- doubling daily turnover at fixed ADV must halve
    the derived capacity, since capacity ~ ADV*participation / turnover.
    """
    from nifty_quant.execution.capacity import capacity_from_turnover  # noqa: PLC0415

    median_adv = 50_00_00_000.0
    max_participation = 0.02

    low_turnover_capacity = capacity_from_turnover(
        median_adv=median_adv, max_participation=max_participation, daily_turnover_fraction=0.11
    )
    high_turnover_capacity = capacity_from_turnover(
        median_adv=median_adv, max_participation=max_participation, daily_turnover_fraction=0.22
    )

    assert high_turnover_capacity == pytest.approx(low_turnover_capacity / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# H4: unfilled quantity is a visible, first-class FillModel output (already
# in library code) -- verify it is never silently zeroed on a starved fill.
# ---------------------------------------------------------------------------


def test_fill_model_reports_nonzero_unfilled_notional_when_participation_binds() -> None:
    """Catches: a fill path that either (a) silently drops the shortfall so
    unfilled_notional reads back as 0.0 even though the order was only
    partially filled, or (b) requeues/carries the shortfall forward (the
    spec explicitly forbids requeuing in this phase, line 69) rather than
    reporting it in the SAME bar it occurred. We size an order well beyond
    max_participation * bar_traded_value so a full fill is impossible, and
    require the reported unfilled_notional to reflect that shortfall
    (present-but-zero would be the silent-drop failure mode).
    """
    model = FillModel(slippage=ZeroSlippage(), max_participation=0.01)
    orders = np.array([100_000.0], dtype=np.float64)  # shares
    prices = np.array([100.0], dtype=np.float64)  # desired notional = 1e7
    bar_traded_value = np.array([1_000_000.0], dtype=np.float64)  # cap = 1e4
    tradable = np.array([True], dtype=bool)

    result = model.fill(orders, prices, bar_traded_value, tradable)

    desired_notional = 100_000.0 * 100.0
    max_notional = 0.01 * 1_000_000.0
    expected_unfilled = desired_notional - max_notional

    assert result.unfilled_notional[0] == pytest.approx(expected_unfilled, rel=1e-9)
    assert result.unfilled_notional[0] > 0.0
    # The order must be PARTIALLY, not fully, filled -- fully filling here
    # would mean the participation cap was silently ignored.
    filled_notional = abs(result.filled_qty[0]) * result.fill_price[0]
    assert filled_notional < desired_notional
    assert filled_notional == pytest.approx(max_notional, rel=1e-6)
