"""Test-first coverage for Spec H: execution realism and capacity.

Obligation map
--------------
1. ``test_round_trip_bps_regression_across_brokerage_crossover`` pins the
   measured round-trip costs at Rs 50k, 66,666, 66,667, 100k, and 1 crore,
   including the flat-below/decaying-above crossover.
2. ``test_brokerage_crossover_notional_is_derived_from_instance_fields`` checks
   that the crossover is derived from the instance's brokerage fields.
3. ``test_gst_base_excludes_stt_and_stamp_duty`` checks the GST base.
4. ``test_stt_is_sell_side_and_stamp_is_buy_side`` checks charge directionality.
5. ``test_cost_ladder_has_all_cells_and_first_class_metrics`` checks the 35-cell
   surface, its ordering, and the required metrics on every cell.
6. ``test_ladder_exposes_fill_starvation_with_net_sharpe`` checks that
   starvation increases the reported unfilled fraction while net metrics remain
   reported rather than being recomputed by silently dropping the shortfall.
7. ``test_turnover_capacity_is_lower_than_adv_capacity`` checks the required
   turnover-versus-ADV capacity ordering; ``test_adv_capacity_uses_the_specified
   formula`` pins the existing ADV upper-bound calculation.
8. ``test_zero_fixed_bps_matches_zero_cost`` checks the zero-cost hook.

Additional H1 coverage is provided by
``test_zero_bps_column_has_higher_net_bps_than_costly_column``: the zero-cost
column must be meaningful rather than padding.

Spec concerns and defects recorded here
----------------------------------------
* The stale 4.55-bps quote is rejected.  The regression pins use the specified
  measured value of 10.6245 bps at the small-notional crossover points.
* ``capacity_from_turnover`` has no exact formula in the specification.  The
  required ordering is tested without inventing a value.
* There is a genuine mathematical tension in the capacity requirement: the
  naive constraint
  ``AUM <= participation_cap * ADV / (typical_weight * daily_turnover_frac)``
  is larger than the ADV-only expression when daily turnover is below one,
  which is the opposite of the required ordering.  This is documented and
  intentionally not silently resolved by testing a guessed formula.
* The prose participation percentages are represented by decimal fractions in
  the assumed API; the tests use the API units.
* The ladder's stated net-return formula is independent of fill completion.
  Unfilled notional is therefore tested as a separate first-class diagnostic,
  with no requeue or implicit fill-adjustment assumed.
* The compact ladder data below is one bar series, not a multi-session fixture,
  so it makes no fixed-bars-per-session assumption.
"""

import numpy as np
import pytest

from nifty_quant.execution.costs import (
    FillBatch,
    FixedBpsCost,
    NSEIntradayEquityCosts,
    ZeroCost,
)
from nifty_quant.execution.fills import FillModel, ZeroSlippage


def _ladder_inputs():
    """Return float64 ladder inputs with varying returns, turnover, and liquidity."""
    gross_returns = np.array(
        [0.0010, 0.0008, 0.0012, -0.0003, 0.0009, 0.0007],
        dtype=np.float64,
    )
    turnover = np.array(
        [0.11, 0.09, 0.13, 0.10, 0.12, 0.11],
        dtype=np.float64,
    )
    orders = np.array(
        [100.0, -200.0, 150.0, -50.0, 120.0, -80.0],
        dtype=np.float64,
    )
    prices = np.array(
        [100.0, 100.0, 200.0, 100.0, 150.0, 125.0],
        dtype=np.float64,
    )
    bar_traded_value = np.array(
        [1_000_000.0, 1_100_000.0, 1_400_000.0, 900_000.0, 1_300_000.0, 1_000_000.0],
        dtype=np.float64,
    )
    tradable = np.ones(gross_returns.shape, dtype=bool)
    return gross_returns, turnover, orders, prices, bar_traded_value, tradable


def test_round_trip_bps_regression_across_brokerage_crossover():
    # It catches brokerage costs that fail to stay flat below or decay above the crossover.
    costs = NSEIntradayEquityCosts()
    notionals = [50_000.0, 66_666.0, 66_667.0, 100_000.0, 10_000_000.0]
    expected_bps = [10.6245, 10.6245, 10.6245, 8.2645, 3.5917]

    measured = [costs.round_trip_bps(notional) for notional in notionals]

    for actual, expected in zip(measured, expected_bps):
        assert np.isclose(actual, expected, rtol=0.0, atol=0.001)

    assert np.isclose(measured[0], measured[1], rtol=0.0, atol=0.001)
    assert np.isclose(measured[1], measured[2], rtol=0.0, atol=0.001)
    assert measured[2] > measured[3] > measured[4]


def test_brokerage_crossover_notional_is_derived_from_instance_fields():
    # It catches a hard-coded default crossover that ignores an instance's brokerage fields.
    costs = NSEIntradayEquityCosts(
        brokerage_flat=30.0,
        brokerage_pct=0.0006,
    )

    expected = costs.brokerage_flat / costs.brokerage_pct

    assert np.isclose(
        costs.brokerage_crossover_notional,
        expected,
        rtol=0.0,
        atol=1e-9,
    )


def test_gst_base_excludes_stt_and_stamp_duty():
    # It catches GST being charged on sell-side STT or buy-side stamp duty.
    costs = NSEIntradayEquityCosts()
    fills = FillBatch(
        notional=np.array([100_000.0, 100_000.0], dtype=np.float64),
        is_buy=np.array([True, False], dtype=bool),
    )

    charges = costs.charges(fills)
    gst_base = charges.brokerage + charges.exchange_txn + charges.sebi
    inclusive_base = gst_base + charges.stt + charges.stamp_duty

    np.testing.assert_allclose(
        charges.gst,
        costs.gst_pct * gst_base,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(costs.gst_pct * inclusive_base > charges.gst)


def test_stt_is_sell_side_and_stamp_is_buy_side():
    # It catches STT or stamp duty being applied to both directions or the wrong direction.
    costs = NSEIntradayEquityCosts()
    notional = 10_000.0
    fills = FillBatch(
        notional=np.array([notional, notional], dtype=np.float64),
        is_buy=np.array([True, False], dtype=bool),
    )

    charges = costs.charges(fills)

    assert charges.stt[0] == pytest.approx(0.0, abs=1e-12)
    assert charges.stt[1] == pytest.approx(
        costs.stt_sell_pct * notional,
        rel=0.0,
        abs=1e-12,
    )
    assert charges.stamp_duty[0] == pytest.approx(
        costs.stamp_buy_pct * notional,
        rel=0.0,
        abs=1e-12,
    )
    assert charges.stamp_duty[1] == pytest.approx(0.0, abs=1e-12)


def test_cost_ladder_has_all_cells_and_first_class_metrics():
    # It catches missing ladder cells, transposed grid order, or cells that omit unfilled fraction.
    from nifty_quant.execution.capacity import (  # A2.1/A2.2: module+name pinned, aliased
        cost_participation_ladder as cost_ladder,
    )

    inputs = _ladder_inputs()
    gross_returns, turnover, orders, prices, bar_traded_value, tradable = inputs
    costs = (0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0)
    participations = (0.005, 0.01, 0.02, 0.05, 0.10)

    cells = cost_ladder(
        gross_returns,
        turnover,
        orders,
        prices,
        bar_traded_value,
        tradable,
    )

    expected_pairs = [
        (cost_bps, participation)
        for cost_bps in costs
        for participation in participations
    ]
    actual_pairs = [(cell.cost_bps, cell.participation) for cell in cells]

    assert len(cells) == 35
    assert actual_pairs == expected_pairs

    expected_turnover = float(np.mean(turnover))
    for cell in cells:
        assert np.isfinite(cell.net_sharpe)
        assert np.isfinite(cell.net_bps_per_day)
        assert cell.turnover == pytest.approx(
            expected_turnover,
            rel=0.0,
            abs=1e-12,
        )
        assert np.isfinite(cell.unfilled_fraction)
        assert 0.0 <= cell.unfilled_fraction <= 1.0


def test_zero_bps_column_has_higher_net_bps_than_costly_column():
    # It catches a ladder that ignores its cost axis or treats the zero-bps column as padding.
    from nifty_quant.execution.capacity import (  # A2.1/A2.2: module+name pinned, aliased
        cost_participation_ladder as cost_ladder,
    )

    gross_returns, turnover, orders, prices, bar_traded_value, tradable = (
        _ladder_inputs()
    )
    cells = cost_ladder(
        gross_returns,
        turnover,
        orders,
        prices,
        bar_traded_value,
        tradable,
        cost_bps_grid=(0.0, 20.0),
        participation_grid=(0.10,),
    )

    zero_cost_cell, costly_cell = cells
    expected_zero_bps_per_day = float(np.mean(gross_returns) * 1e4)
    expected_costly_bps_per_day = float(
        np.mean(gross_returns - (20.0 / 1e4) * turnover) * 1e4
    )

    assert zero_cost_cell.net_bps_per_day == pytest.approx(
        expected_zero_bps_per_day,
        rel=0.0,
        abs=1e-12,
    )
    assert costly_cell.net_bps_per_day == pytest.approx(
        expected_costly_bps_per_day,
        rel=0.0,
        abs=1e-12,
    )
    assert zero_cost_cell.net_bps_per_day > costly_cell.net_bps_per_day


def test_ladder_exposes_fill_starvation_with_net_sharpe():
    # It catches unfilled orders being dropped from the metric calculation or hidden from the cell.
    from nifty_quant.execution.capacity import (  # A2.1/A2.2: module+name pinned, aliased
        cost_participation_ladder as cost_ladder,
    )

    gross_returns, turnover, orders, prices, bar_traded_value, tradable = (
        _ladder_inputs()
    )
    cells = cost_ladder(
        gross_returns,
        turnover,
        orders,
        prices,
        bar_traded_value,
        tradable,
        cost_bps_grid=(5.0,),
        participation_grid=(0.005, 0.10),
    )

    low_participation_cell, high_participation_cell = cells
    desired_notional = np.abs(orders) * prices
    expected_low_unfilled = float(
        np.maximum(desired_notional - 0.005 * bar_traded_value, 0.0).sum()
        / desired_notional.sum()
    )
    expected_high_unfilled = float(
        np.maximum(desired_notional - 0.10 * bar_traded_value, 0.0).sum()
        / desired_notional.sum()
    )

    net_returns = gross_returns - (5.0 / 1e4) * turnover
    expected_sharpe = float(np.mean(net_returns) / np.std(net_returns, ddof=1))
    expected_net_bps_per_day = float(np.mean(net_returns) * 1e4)

    assert low_participation_cell.unfilled_fraction == pytest.approx(
        expected_low_unfilled,
        rel=0.0,
        abs=1e-12,
    )
    assert high_participation_cell.unfilled_fraction == pytest.approx(
        expected_high_unfilled,
        rel=0.0,
        abs=1e-12,
    )
    assert low_participation_cell.unfilled_fraction > high_participation_cell.unfilled_fraction

    for cell in cells:
        assert cell.net_sharpe == pytest.approx(
            expected_sharpe,
            rel=0.0,
            abs=1e-12,
        )
        assert cell.net_bps_per_day == pytest.approx(
            expected_net_bps_per_day,
            rel=0.0,
            abs=1e-12,
        )


def test_fill_model_reports_capped_notional_without_requeue():
    # It catches a fill engine that hides capped-away notional or requeues it
    # into a later zero-order bar.
    model = FillModel(slippage=ZeroSlippage(), max_participation=0.01)
    orders = np.array([100.0, 0.0], dtype=np.float64)
    prices = np.array([100.0, 100.0], dtype=np.float64)
    bar_traded_value = np.array([100_000.0, 100_000.0], dtype=np.float64)
    tradable = np.array([True, True], dtype=bool)

    result = model.fill(orders, prices, bar_traded_value, tradable)

    expected_desired = np.abs(orders) * prices
    expected_capped = np.minimum(expected_desired, 0.01 * bar_traded_value)
    expected_unfilled = expected_desired - expected_capped
    expected_filled_qty = expected_capped / prices

    np.testing.assert_allclose(
        result.unfilled_notional,
        expected_unfilled,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.filled_qty,
        expected_filled_qty,
        rtol=0.0,
        atol=1e-12,
    )
    assert result.unfilled_notional[0] > result.unfilled_notional[1]
    assert result.filled_qty[1] == pytest.approx(0.0, abs=1e-12)


def test_adv_capacity_uses_the_specified_formula():
    # It catches the ADV upper-bound helper drifting from the existing position-size estimate.
    from nifty_quant.execution.capacity import capacity_from_adv  # A2.1: pinned module

    weighted_median_adv = 1_234_567_890.0
    typical_weight = 0.0175
    participation_cap = 0.023

    expected = participation_cap * weighted_median_adv / typical_weight
    actual = capacity_from_adv(
        weighted_median_adv,
        typical_weight,
        participation_cap,
    )

    assert np.isclose(actual, expected, rtol=1e-12, atol=0.001)


def test_capacity_ratio_identity_and_binding_constraint():
    """REPLACED by the lead per AMENDMENT 1 + A2.5: the original test asserted the
    retracted ordering `turnover_capacity < adv_capacity`, which is backwards for a
    turnover fraction below one -- the trade constraint is the position constraint
    DIVIDED by turnover_frac, hence LOOSER. These are the amendment's own three
    mandated assertions."""
    from nifty_quant.execution.capacity import (  # A2.1: pinned module
        capacity_from_adv,
        capacity_from_turnover,
        combined_capacity,
    )

    weighted_median_adv = 4_000_000_000.0
    typical_weight = 0.02
    participation_cap = 0.02
    daily_turnover_frac = 0.11

    adv_capacity = capacity_from_adv(
        weighted_median_adv,
        typical_weight,
        participation_cap,
    )
    turnover_capacity = capacity_from_turnover(
        weighted_median_adv,
        typical_weight,
        participation_cap,
        daily_turnover_frac,
    )

    # (1) The directionless ratio identity -- holds for high- and low-turnover books alike.
    assert np.isclose(
        turnover_capacity / adv_capacity, 1.0 / daily_turnover_frac, rtol=1e-12, atol=0.0
    )

    # (2) Reported capacity is min(position, trade) and says WHICH constraint bound.
    combined = combined_capacity(
        weighted_median_adv,
        typical_weight,
        participation_cap,
        daily_turnover_frac,
    )
    assert combined.value == pytest.approx(min(adv_capacity, turnover_capacity))

    # (3) Direction regression: turnover_frac < 1 -> POSITION binds; > 1 -> TRADE binds.
    assert combined.binding == "position"
    high_turnover = combined_capacity(
        weighted_median_adv,
        typical_weight,
        participation_cap,
        1.6,
    )
    assert high_turnover.binding == "trade"
    assert high_turnover.value == pytest.approx(adv_capacity / 1.6)


def test_zero_fixed_bps_matches_zero_cost():
    # It catches FixedBpsCost(0.0) introducing nonzero charges or differing from ZeroCost.
    fills = FillBatch(
        notional=np.array([12_345.6, 98_765.4, 45_678.9], dtype=np.float64),
        is_buy=np.array([True, False, True], dtype=bool),
        n_orders=np.array([2.0, 3.0, 1.0], dtype=np.float64),
    )

    zero_charges = ZeroCost().charges(fills)
    fixed_zero_charges = FixedBpsCost(bps=0.0).charges(fills)

    zero_components = (
        zero_charges.brokerage,
        zero_charges.stt,
        zero_charges.exchange_txn,
        zero_charges.sebi,
        zero_charges.ipft,
        zero_charges.stamp_duty,
        zero_charges.gst,
    )
    fixed_zero_components = (
        fixed_zero_charges.brokerage,
        fixed_zero_charges.stt,
        fixed_zero_charges.exchange_txn,
        fixed_zero_charges.sebi,
        fixed_zero_charges.ipft,
        fixed_zero_charges.stamp_duty,
        fixed_zero_charges.gst,
    )

    for fixed_component, zero_component in zip(
        fixed_zero_components,
        zero_components,
    ):
        np.testing.assert_array_equal(fixed_component, zero_component)

    np.testing.assert_array_equal(fixed_zero_charges.total, zero_charges.total)
    assert fixed_zero_charges.sum().total == pytest.approx(
        zero_charges.sum().total,
        rel=0.0,
        abs=1e-12,
    )
