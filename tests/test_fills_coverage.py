from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nifty_quant.execution.fills import (
    FillModel,
    SlippageModel,
    SqrtImpactSlippage,
    ZeroSlippage,
)


@dataclass(frozen=True)
class _AlwaysInfSlippage:
    def bps(self, notional: np.ndarray, bar_traded_value: np.ndarray) -> np.ndarray:
        shape = np.broadcast_shapes(
            np.asarray(notional).shape, np.asarray(bar_traded_value).shape
        )
        return np.full(shape, np.inf)


def test_protocol_stub_bps_body_is_otherwise_dead_code() -> None:
    # Protocol stub body is never executed via normal dispatch; call it directly.
    assert SlippageModel.bps(object(), np.array([1.0]), np.array([2.0])) is None


def test_sqrt_impact_bps_normal_scalar_exact() -> None:
    """Default coefficients: bps = 1.5 + 10*sqrt(100/400) = 6.5."""
    model = SqrtImpactSlippage()
    result = model.bps(np.array(100.0), np.array(400.0))
    assert result.item() == 6.5


def test_sqrt_impact_bps_array_multiple() -> None:
    """ratios [0, 9] -> sqrt [0, 3] -> bps [1.5, 31.5]."""
    model = SqrtImpactSlippage()
    result = model.bps(np.array([0.0, 900.0]), np.array([100.0, 100.0]))
    expected = np.array([1.5, 31.5])
    assert np.array_equal(result, expected)


def test_sqrt_impact_bps_zero_bar_traded_value_is_inf() -> None:
    result = SqrtImpactSlippage().bps(np.array([100.0]), np.array([0.0]))
    assert np.isinf(result).all()
    assert result[0] == np.inf


def test_sqrt_impact_bps_nan_bar_traded_value_is_inf() -> None:
    result = SqrtImpactSlippage().bps(np.array([100.0]), np.array([np.nan]))
    assert np.isinf(result).all()


def test_sqrt_impact_bps_negative_bar_traded_value_is_inf() -> None:
    result = SqrtImpactSlippage().bps(np.array([100.0]), np.array([-5.0]))
    assert np.isinf(result).all()


def test_sqrt_impact_bps_negative_notional_is_inf() -> None:
    result = SqrtImpactSlippage().bps(np.array([-1.0]), np.array([100.0]))
    assert np.isinf(result).all()


def test_sqrt_impact_bps_nan_notional_is_inf() -> None:
    result = SqrtImpactSlippage().bps(np.array([np.nan]), np.array([100.0]))
    assert np.isinf(result).all()


def test_sqrt_impact_bps_zero_notional_returns_half_spread() -> None:
    result = SqrtImpactSlippage().bps(np.array([0.0]), np.array([100.0]))
    assert result.item() == 1.5


def test_sqrt_impact_bps_broadcasts_scalar_notional() -> None:
    result = SqrtImpactSlippage().bps(
        np.array(100.0), np.array([400.0, 1600.0, 100.0])
    )
    expected = np.array([6.5, 4.0, 11.5])
    assert result.shape == (3,)
    assert np.allclose(result, expected, rtol=1e-15, atol=1e-15)


def test_sqrt_impact_bps_custom_coefficients() -> None:
    model = SqrtImpactSlippage(half_spread_bps=2.0, impact_coef=5.0)
    result = model.bps(np.array([400.0]), np.array([100.0]))
    assert result.item() == 12.0


def test_zero_slippage_returns_zero_scalar() -> None:
    result = ZeroSlippage().bps(np.array(100.0), np.array(400.0))
    assert result.shape == ()
    assert result.dtype == np.float64
    assert result.item() == 0.0


def test_zero_slippage_returns_zeros_for_broadcast_invalid_inputs() -> None:
    result = ZeroSlippage().bps(np.array([100.0, -1.0]), np.array([0.0, 500.0]))
    assert result.shape == (2,)
    assert result.dtype == np.float64
    assert np.array_equal(result, np.zeros(2))

    broadcast = ZeroSlippage().bps(np.array(10.0), np.array([1.0, 2.0]))
    assert broadcast.shape == (2,)
    assert np.array_equal(broadcast, np.zeros(2))


def test_fill_rejects_2d_orders() -> None:
    model = FillModel(ZeroSlippage())
    with pytest.raises(ValueError, match="orders, prices, bar_traded_value, tradable must be 1-D"):
        model.fill(
            orders=np.ones((2, 1)),
            prices=np.array([100.0]),
            bar_traded_value=np.array([1000.0]),
            tradable=np.array([True]),
        )


def test_fill_rejects_mismatched_lengths() -> None:
    model = FillModel(ZeroSlippage())
    with pytest.raises(ValueError, match="all fill inputs must have the same length"):
        model.fill(
            orders=np.array([1.0, 2.0, 3.0]),
            prices=np.array([100.0, 101.0]),
            bar_traded_value=np.array([1000.0, 1000.0]),
            tradable=np.array([True, True]),
        )


def test_fill_zero_orders_keeps_valid_prices_and_no_rejections() -> None:
    model = FillModel(ZeroSlippage())
    prices = np.array([100.0, 200.0, 300.0])
    result = model.fill(
        orders=np.array([0.0, 0.0, 0.0]),
        prices=prices,
        bar_traded_value=np.array([1000.0, 2000.0, 3000.0]),
        tradable=np.array([True, False, True]),
    )
    assert np.array_equal(result.filled_qty, np.zeros(3))
    assert np.array_equal(result.fill_price, prices)
    assert not result.rejected.any()
    assert np.array_equal(result.unfilled_notional, np.zeros(3))


def test_fill_empty_arrays() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([]),
        prices=np.array([]),
        bar_traded_value=np.array([]),
        tradable=np.array([]),
    )
    assert result.filled_qty.shape == (0,)
    assert result.fill_price.shape == (0,)
    assert result.rejected.shape == (0,)
    assert result.unfilled_notional.shape == (0,)


def test_fill_buy_fully_fillable_zero_slippage() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([1_000_000.0]),
        tradable=np.array([True]),
    )
    assert np.array_equal(result.filled_qty, np.array([10.0]))
    assert np.array_equal(result.fill_price, np.array([100.0]))
    assert np.array_equal(result.rejected, np.array([False]))
    assert np.array_equal(result.unfilled_notional, np.array([0.0]))


def test_fill_sell_fully_fillable_zero_slippage() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([-10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([1_000_000.0]),
        tradable=np.array([True]),
    )
    assert np.array_equal(result.filled_qty, np.array([-10.0]))
    assert np.array_equal(result.fill_price, np.array([100.0]))
    assert np.array_equal(result.rejected, np.array([False]))
    assert np.array_equal(result.unfilled_notional, np.array([0.0]))


def test_fill_buy_sell_price_asymmetry_with_sqrt_slippage() -> None:
    slippage = SqrtImpactSlippage()
    model = FillModel(slippage)
    result = model.fill(
        orders=np.array([10.0, -10.0]),
        prices=np.array([100.0, 100.0]),
        bar_traded_value=np.array([40000.0, 40000.0]),
        tradable=np.array([True, True]),
    )

    sqrt_ratio = np.sqrt(0.02)
    expected_bps = 1.5 + 10.0 * sqrt_ratio
    expected_buy_price = 100.0 * (1.0 + expected_bps / 10000.0)
    expected_sell_price = 100.0 * (1.0 - expected_bps / 10000.0)

    assert np.array_equal(result.filled_qty, np.array([8.0, -8.0]))
    assert result.fill_price[0] == pytest.approx(expected_buy_price, rel=1e-12)
    assert result.fill_price[1] == pytest.approx(expected_sell_price, rel=1e-12)
    assert result.fill_price[0] > 100.0 > result.fill_price[1]
    assert np.array_equal(result.rejected, np.array([False, False]))
    assert np.array_equal(result.unfilled_notional, np.array([200.0, 200.0]))


def test_fill_tradable_false_rejected_full_unfilled_price_still_computed() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([1_000_000.0]),
        tradable=np.array([False]),
    )
    assert np.array_equal(result.filled_qty, np.array([0.0]))
    assert np.array_equal(result.fill_price, np.array([100.0]))
    assert np.array_equal(result.rejected, np.array([True]))
    assert np.array_equal(result.unfilled_notional, np.array([1000.0]))


def test_fill_zero_bar_traded_value_rejected_full_unfilled() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([0.0]),
        tradable=np.array([True]),
    )
    assert result.filled_qty[0] == 0.0
    assert result.fill_price[0] == 100.0
    assert result.rejected[0]
    assert result.unfilled_notional[0] == 1000.0


def test_fill_nan_bar_traded_value_rejected_full_unfilled() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([np.nan]),
        tradable=np.array([True]),
    )
    assert result.filled_qty[0] == 0.0
    assert result.fill_price[0] == 100.0
    assert result.rejected[0]
    assert result.unfilled_notional[0] == 1000.0


def test_fill_negative_bar_traded_value_rejected_full_unfilled() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([-100.0]),
        tradable=np.array([True]),
    )
    assert result.filled_qty[0] == 0.0
    assert result.fill_price[0] == 100.0
    assert result.rejected[0]
    assert result.unfilled_notional[0] == 1000.0


def test_fill_invalid_prices_rejected_fill_price_zero_unfilled_zero() -> None:
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=np.array([10.0, 10.0]),
        prices=np.array([0.0, -5.0]),
        bar_traded_value=np.array([1_000_000.0, 1_000_000.0]),
        tradable=np.array([True, True]),
    )
    assert np.array_equal(result.filled_qty, np.array([0.0, 0.0]))
    assert np.array_equal(result.fill_price, np.array([0.0, 0.0]))
    assert np.array_equal(result.rejected, np.array([True, True]))
    assert np.array_equal(result.unfilled_notional, np.array([0.0, 0.0]))


def test_fill_nonfinite_orders_rejected() -> None:
    orders = np.array([np.nan, np.inf, -np.inf])
    prices = np.array([10.0, 20.0, 30.0])
    model = FillModel(ZeroSlippage())
    result = model.fill(
        orders=orders,
        prices=prices,
        bar_traded_value=np.array([1_000_000.0, 1_000_000.0, 1_000_000.0]),
        tradable=np.array([True, True, True]),
    )
    assert np.array_equal(result.filled_qty, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(result.fill_price, prices)
    assert np.array_equal(result.rejected, np.array([True, True, True]))
    assert np.array_equal(result.unfilled_notional, np.array([0.0, 0.0, 0.0]))


def test_fill_infinite_slippage_guard_rejects_otherwise_valid_order() -> None:
    model = FillModel(_AlwaysInfSlippage())
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([1_000_000.0]),
        tradable=np.array([True]),
    )
    assert np.array_equal(result.filled_qty, np.array([0.0]))
    assert np.array_equal(result.fill_price, np.array([100.0]))
    assert np.array_equal(result.rejected, np.array([True]))
    assert np.array_equal(result.unfilled_notional, np.array([1000.0]))


def test_fill_mixed_array_combines_paths() -> None:
    model = FillModel(ZeroSlippage())
    orders = np.array([10.0, -5.0, 10.0, 10.0, 10.0, np.nan])
    prices = np.array([100.0, 200.0, 100.0, 100.0, 0.0, 100.0])
    btv = np.array(
        [1_000_000.0, 1_000_000.0, 1_000_000.0, 0.0, 1_000_000.0, 1_000_000.0]
    )
    tradable = np.array([True, True, False, True, True, True])

    result = model.fill(orders=orders, prices=prices, bar_traded_value=btv, tradable=tradable)

    expected_filled_qty = np.array([10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    expected_fill_price = np.array([100.0, 200.0, 100.0, 100.0, 0.0, 100.0])
    expected_rejected = np.array([False, False, True, True, True, True])
    expected_unfilled = np.array([0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0])

    assert np.array_equal(result.filled_qty, expected_filled_qty)
    assert np.array_equal(result.fill_price, expected_fill_price)
    assert np.array_equal(result.rejected, expected_rejected)
    assert np.array_equal(result.unfilled_notional, expected_unfilled)


def test_fill_custom_max_participation_caps_notional() -> None:
    model = FillModel(ZeroSlippage(), max_participation=0.5)
    result = model.fill(
        orders=np.array([10.0]),
        prices=np.array([100.0]),
        bar_traded_value=np.array([1000.0]),
        tradable=np.array([True]),
    )
    assert np.array_equal(result.filled_qty, np.array([5.0]))
    assert np.array_equal(result.fill_price, np.array([100.0]))
    assert np.array_equal(result.rejected, np.array([False]))
    assert np.array_equal(result.unfilled_notional, np.array([500.0]))
