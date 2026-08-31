"""Coverage tests for capacity.py error paths and edge cases.

These tests cover branches and error paths not exercised by the main test suites,
ensuring 100% line and branch coverage as specified in the phase H deliverables.
"""

from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.execution.capacity import (
    CapacityResult,
    LadderCell,
    capacity_from_adv,
    capacity_from_turnover,
    combined_capacity,
    cost_participation_ladder,
)


class TestCostParticipationLadderValidation:
    """Error path coverage for cost_participation_ladder."""

    def test_gross_returns_not_1d(self) -> None:
        """Raises when gross_returns is not 1-D."""
        with pytest.raises(ValueError, match="gross_returns and turnover must be 1-D arrays"):
            cost_participation_ladder(
                gross_returns=np.array([[0.001, 0.002]], dtype=np.float64),
                turnover=np.array([0.10], dtype=np.float64),
            )

    def test_turnover_not_1d(self) -> None:
        """Raises when turnover is not 1-D."""
        with pytest.raises(ValueError, match="gross_returns and turnover must be 1-D arrays"):
            cost_participation_ladder(
                gross_returns=np.array([0.001], dtype=np.float64),
                turnover=np.array([[0.10, 0.11]], dtype=np.float64),
            )

    def test_shape_mismatch(self) -> None:
        """Raises when gross_returns and turnover have different lengths."""
        with pytest.raises(
            ValueError, match="gross_returns and turnover must have the same length"
        ):
            cost_participation_ladder(
                gross_returns=np.array([0.001, 0.002], dtype=np.float64),
                turnover=np.array([0.10], dtype=np.float64),
            )

    def test_gross_returns_not_finite(self) -> None:
        """Raises when gross_returns contains non-finite values."""
        with pytest.raises(ValueError, match="gross_returns must be finite"):
            cost_participation_ladder(
                gross_returns=np.array([0.001, np.nan, 0.002], dtype=np.float64),
                turnover=np.array([0.10, 0.11, 0.12], dtype=np.float64),
            )

    def test_turnover_not_finite(self) -> None:
        """Raises when turnover contains non-finite values."""
        with pytest.raises(ValueError, match="turnover must be finite"):
            cost_participation_ladder(
                gross_returns=np.array([0.001, 0.002, 0.003], dtype=np.float64),
                turnover=np.array([0.10, np.inf, 0.12], dtype=np.float64),
            )

    def test_orders_not_1d(self) -> None:
        """Raises when orders is not 1-D."""
        with pytest.raises(
            ValueError, match="orders, prices, bar_traded_value must be at least 1-D"
        ):
            cost_participation_ladder(
                gross_returns=np.array([0.001, 0.002], dtype=np.float64),
                turnover=np.array([0.10, 0.11], dtype=np.float64),
                orders=np.array(100.0),  # scalar, ndim=0
                prices=np.array([100.0, 100.0], dtype=np.float64),
                bar_traded_value=np.array([1e6, 1e6], dtype=np.float64),
            )

    def test_orders_prices_bar_traded_value_length_mismatch(self) -> None:
        """Raises when orders, prices, bar_traded_value don't match gross_returns."""
        with pytest.raises(
            ValueError,
            match="orders, prices, bar_traded_value must have the same length as gross_returns",
        ):
            cost_participation_ladder(
                gross_returns=np.array([0.001, 0.002], dtype=np.float64),
                turnover=np.array([0.10, 0.11], dtype=np.float64),
                orders=np.array([100.0], dtype=np.float64),  # length 1, mismatch
                prices=np.array([100.0], dtype=np.float64),
                bar_traded_value=np.array([1e6], dtype=np.float64),
            )

    def test_bar_traded_value_none_with_no_orders(self) -> None:
        """Cost ladder works when bar_traded_value is None and no orders given."""
        # This tests the branch where bar_traded_value_arr is None (line 182).
        # When bar_traded_value is None, fillable = participation * 1.0
        # So unfilled_fraction will be very high since desired_notional >> fillable.
        gross_returns = np.array([0.001, 0.002], dtype=np.float64)
        turnover = np.array([0.10, 0.11], dtype=np.float64)

        cells = cost_participation_ladder(
            gross_returns=gross_returns,
            turnover=turnover,
            bar_traded_value=None,
            cost_bps_grid=(0.0,),
            participation_grid=(0.01,),
        )

        assert len(cells) == 1
        cell = cells[0]
        # With capital_ref=1e6, desired = turnover * 1e6, but fillable = 0.01 * 1.0 = 0.01
        # So unfilled_fraction is very close to 1.0
        assert cell.unfilled_fraction > 0.99


class TestNetSharpeEdgeCases:
    """Test edge cases for net Sharpe computation when std is zero."""

    def test_flat_positive_returns_std_zero(self) -> None:
        """When all net returns are identical and positive, Sharpe is +inf."""
        gross_returns = np.array([0.001, 0.001, 0.001], dtype=np.float64)
        turnover = np.array([0.10, 0.10, 0.10], dtype=np.float64)

        cells = cost_participation_ladder(
            gross_returns=gross_returns,
            turnover=turnover,
            cost_bps_grid=(0.0,),  # cost=0, so net = gross (all 0.001)
            participation_grid=(0.01,),
        )

        assert len(cells) == 1
        assert cells[0].net_sharpe == float("inf")

    def test_flat_negative_returns_std_zero(self) -> None:
        """When all net returns are identical and negative, Sharpe is -inf."""
        gross_returns = np.array([0.001, 0.001, 0.001], dtype=np.float64)
        turnover = np.array([0.10, 0.10, 0.10], dtype=np.float64)

        # high cost: net = 0.001 - (200e-4)*0.10 = 0.001 - 0.002 = -0.001 (negative)
        cells = cost_participation_ladder(
            gross_returns=gross_returns,
            turnover=turnover,
            cost_bps_grid=(200.0,),
            participation_grid=(0.01,),
        )

        assert len(cells) == 1
        assert cells[0].net_sharpe == float("-inf")

    def test_flat_zero_returns_std_zero(self) -> None:
        """When all net returns are zero (std=0, mean=0), Sharpe is 0."""
        gross_returns = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        turnover = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        cells = cost_participation_ladder(
            gross_returns=gross_returns,
            turnover=turnover,
            cost_bps_grid=(0.0,),
            participation_grid=(0.01,),
        )

        assert len(cells) == 1
        assert cells[0].net_sharpe == 0.0


class TestCapacityFromAdvValidation:
    """Error path coverage for capacity_from_adv."""

    def test_negative_median_adv(self) -> None:
        """Raises when median_adv is negative."""
        with pytest.raises(ValueError, match="median_adv must be non-negative"):
            capacity_from_adv(median_adv=-1e6, typical_weight=0.02, max_participation=0.02)

    def test_negative_max_participation(self) -> None:
        """Raises when max_participation is negative."""
        with pytest.raises(ValueError, match="max_participation must be in"):
            capacity_from_adv(median_adv=1e9, typical_weight=0.02, max_participation=-0.01)

    def test_max_participation_over_one(self) -> None:
        """Raises when max_participation exceeds 1.0."""
        with pytest.raises(ValueError, match="max_participation must be in"):
            capacity_from_adv(median_adv=1e9, typical_weight=0.02, max_participation=1.5)

    def test_nonpositive_typical_weight(self) -> None:
        """Raises when typical_weight is non-positive."""
        with pytest.raises(ValueError, match="typical_weight must be positive"):
            capacity_from_adv(median_adv=1e9, typical_weight=0.0, max_participation=0.02)

    def test_typical_weight_none_returns_per_name_capacity(self) -> None:
        """When typical_weight is None, returns max_participation * median_adv."""
        capacity = capacity_from_adv(
            median_adv=1e8,
            typical_weight=None,
            max_participation=0.02,
        )
        expected = 0.02 * 1e8
        assert capacity == pytest.approx(expected)


class TestCapacityFromTurnoverValidation:
    """Error path coverage for capacity_from_turnover."""

    def test_none_daily_turnover_fraction(self) -> None:
        """Raises when daily_turnover_fraction is None."""
        with pytest.raises(ValueError, match="daily_turnover_fraction must be positive"):
            capacity_from_turnover(
                median_adv=1e9,
                typical_weight=0.02,
                max_participation=0.02,
                daily_turnover_fraction=None,
            )

    def test_zero_daily_turnover_fraction(self) -> None:
        """Raises when daily_turnover_fraction is zero."""
        with pytest.raises(ValueError, match="daily_turnover_fraction must be positive"):
            capacity_from_turnover(
                median_adv=1e9,
                typical_weight=0.02,
                max_participation=0.02,
                daily_turnover_fraction=0.0,
            )

    def test_negative_daily_turnover_fraction(self) -> None:
        """Raises when daily_turnover_fraction is negative."""
        with pytest.raises(ValueError, match="daily_turnover_fraction must be positive"):
            capacity_from_turnover(
                median_adv=1e9,
                typical_weight=0.02,
                max_participation=0.02,
                daily_turnover_fraction=-0.05,
            )

    def test_valid_computation(self) -> None:
        """Valid turnover-based capacity is adv_capacity / turnover_frac."""
        median_adv = 1e9
        typical_weight = 0.02
        max_participation = 0.02
        turnover_frac = 0.11

        adv_capacity = capacity_from_adv(median_adv, typical_weight, max_participation)
        turnover_capacity = capacity_from_turnover(
            median_adv, typical_weight, max_participation, turnover_frac
        )

        expected = adv_capacity / turnover_frac
        assert turnover_capacity == pytest.approx(expected)


class TestCombinedCapacityBindingConstraint:
    """Coverage for combined_capacity binding constraint logic."""

    def test_no_turnover_fraction_returns_position(self) -> None:
        """When no turnover_frac given, position constraint binds."""
        result = combined_capacity(
            median_adv=1e9,
            typical_weight=0.02,
            max_participation=0.02,
            daily_turnover_fraction=None,
        )
        assert result.binding == "position"
        expected_capacity = capacity_from_adv(1e9, 0.02, 0.02)
        assert result.value == pytest.approx(expected_capacity)

    def test_zero_turnover_fraction_returns_position(self) -> None:
        """When turnover_frac is zero, position constraint binds."""
        result = combined_capacity(
            median_adv=1e9,
            typical_weight=0.02,
            max_participation=0.02,
            daily_turnover_fraction=0.0,
        )
        assert result.binding == "position"

    def test_low_turnover_position_binds(self) -> None:
        """When turnover < 1, position constraint is tighter and binds."""
        median_adv = 1e9
        typical_weight = 0.02
        max_participation = 0.02
        turnover_frac = 0.1  # < 1

        result = combined_capacity(
            median_adv=median_adv,
            typical_weight=typical_weight,
            max_participation=max_participation,
            daily_turnover_fraction=turnover_frac,
        )

        position_capacity = capacity_from_adv(median_adv, typical_weight, max_participation)
        trade_capacity = capacity_from_turnover(
            median_adv, typical_weight, max_participation, turnover_frac
        )

        assert result.binding == "position"
        assert result.value == pytest.approx(position_capacity)
        # Trade capacity should be looser (higher) when turnover < 1
        assert trade_capacity > position_capacity

    def test_high_turnover_trade_binds(self) -> None:
        """When turnover > 1, trade constraint is tighter and binds."""
        median_adv = 1e9
        typical_weight = 0.02
        max_participation = 0.02
        turnover_frac = 2.0  # > 1

        result = combined_capacity(
            median_adv=median_adv,
            typical_weight=typical_weight,
            max_participation=max_participation,
            daily_turnover_fraction=turnover_frac,
        )

        position_capacity = capacity_from_adv(median_adv, typical_weight, max_participation)
        trade_capacity = capacity_from_turnover(
            median_adv, typical_weight, max_participation, turnover_frac
        )

        assert result.binding == "trade"
        assert result.value == pytest.approx(trade_capacity)
        # Trade capacity should be tighter (lower) when turnover > 1
        assert trade_capacity < position_capacity

    def test_combined_capacity_returns_named_tuple_type(self) -> None:
        """combined_capacity returns CapacityResult with value and binding fields."""
        result = combined_capacity(1e9, 0.02, 0.02, 0.11)
        assert isinstance(result, CapacityResult)
        assert hasattr(result, "value")
        assert hasattr(result, "binding")
        assert isinstance(result.value, float)
        assert result.binding in ("position", "trade")


class TestLadderCellConstruction:
    """Verify LadderCell is a frozen dataclass with required fields."""

    def test_ladder_cell_frozen(self) -> None:
        """LadderCell is frozen (immutable)."""
        cell = LadderCell(
            cost_bps=5.0,
            participation=0.01,
            net_sharpe=1.5,
            net_bps_per_day=50.0,
            turnover=0.10,
            unfilled_fraction=0.05,
        )

        with pytest.raises(AttributeError):
            cell.cost_bps = 10.0  # type: ignore

    def test_ladder_cell_has_all_fields(self) -> None:
        """LadderCell has all required fields."""
        cell = LadderCell(
            cost_bps=5.0,
            participation=0.01,
            net_sharpe=1.5,
            net_bps_per_day=50.0,
            turnover=0.10,
            unfilled_fraction=0.05,
        )

        assert cell.cost_bps == 5.0
        assert cell.participation == 0.01
        assert cell.net_sharpe == 1.5
        assert cell.net_bps_per_day == 50.0
        assert cell.turnover == 0.10
        assert cell.unfilled_fraction == 0.05
