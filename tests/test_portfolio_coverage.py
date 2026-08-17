from __future__ import annotations

import numpy as np
import pytest

from nifty_quant.backtest.portfolio import GrossNotionalSizer, Portfolio, SizingResult


def test_equity_raises_value_error_when_prices_shape_does_not_match_shares() -> None:
    portfolio = Portfolio(shares=np.array([1.0, 2.0]), cash=0.0, initial_capital=0.0)
    with pytest.raises(ValueError, match="prices must be 1-D and match shares"):
        portfolio.equity(np.array([100.0]))


def test_apply_fills_raises_value_error_when_qty_and_price_shapes_mismatch() -> None:
    portfolio = Portfolio(shares=np.array([1.0, 2.0]), cash=0.0, initial_capital=0.0)
    with pytest.raises(ValueError, match="qty and price must be 1-D and match shares"):
        portfolio.apply_fills(
            qty=np.array([1.0]),
            price=np.array([100.0, 200.0]),
            charges=1.0,
        )


def test_finalize_shares_skips_whole_share_flooring_when_whole_shares_false() -> None:
    sizer = GrossNotionalSizer(whole_shares=False, min_trade_notional=0.0)
    result = sizer._finalize_shares(
        notional=np.array([250.0]),
        prices=np.array([50.0]),
        valid_price=np.array([True]),
    )
    np.testing.assert_allclose(result, np.array([5.0]))


def test_to_shares_raises_value_error_when_weights_and_prices_shapes_mismatch() -> None:
    sizer = GrossNotionalSizer()
    with pytest.raises(ValueError, match="weights and prices must have the same shape"):
        sizer.to_shares(
            weights=np.array([0.5, 0.5]),
            prices=np.array([100.0]),
            capital=1000.0,
        )


def test_to_shares_raises_value_error_when_bar_traded_value_shape_mismatches_weights() -> None:
    sizer = GrossNotionalSizer()
    with pytest.raises(
        ValueError,
        match="bar_traded_value must have the same shape as weights/prices",
    ):
        sizer.to_shares(
            weights=np.array([0.5]),
            prices=np.array([100.0]),
            capital=1000.0,
            bar_traded_value=np.array([1_000.0, 2_000.0]),
        )


def test_to_shares_empty_arrays_skip_waterfilling_loop_and_return_empty_result() -> None:
    sizer = GrossNotionalSizer()
    result = sizer.to_shares(
        weights=np.array([]),
        prices=np.array([]),
        capital=1_000_000.0,
        bar_traded_value=np.array([]),
    )

    assert isinstance(result, SizingResult)
    assert result.shares.shape == (0,)
    assert result.unmet_gross_pct == pytest.approx(0.0)


def test_portfolio_apply_fills_and_mark_matches_hand_computed_accounting_identity() -> None:
    # fill_value = 10*100 + (-5)*50 = 750
    # cash after fills = 100_000 - 750 - 12.5 = 99_237.5
    # mtm at mark = 10*110 + (-5)*40 = 900
    # equity = 99_237.5 + 900 = 100_137.5
    # cum_pnl = equity + cum_costs - initial_capital = 100_137.5 + 12.5 - 100_000 = 150.0
    portfolio = Portfolio(
        shares=np.array([0.0, 0.0]),
        cash=100_000.0,
        initial_capital=100_000.0,
    )

    portfolio.apply_fills(
        qty=np.array([10.0, -5.0]),
        price=np.array([100.0, 50.0]),
        charges=12.5,
    )

    assert portfolio.cash == pytest.approx(99_237.5)
    np.testing.assert_allclose(portfolio.shares, np.array([10.0, -5.0]))
    assert portfolio.cum_costs == pytest.approx(12.5)

    portfolio.mark(prices=np.array([110.0, 40.0]))
    assert portfolio.cum_pnl == pytest.approx(150.0)


def test_portfolio_equity_masks_nan_price_in_zero_share_column() -> None:
    # The zero-share column's NaN price is masked to 0.0:
    # mtm = 0*0.0 + 3*50 = 150; equity = 1_000 + 150 = 1_150.
    portfolio = Portfolio(shares=np.array([0.0, 3.0]), cash=1000.0, initial_capital=0.0)

    equity = portfolio.equity(prices=np.array([np.nan, 50.0]))

    assert equity == pytest.approx(1150.0)
    assert np.isfinite(equity)


def test_gross_notional_sizer_filters_dust_trades_below_min_notional() -> None:
    # target_notional = 0.05 * 1.0 * 100 = 5.0
    # raw_shares = 5.0 / 50.0 = 0.1 -> floor to 0; resulting notional < 25_000 -> zero shares.
    sizer = GrossNotionalSizer()

    dust = sizer.to_shares(weights=np.array([0.05]), prices=np.array([50.0]), capital=100.0)
    np.testing.assert_array_equal(dust, np.array([0.0]))

    # target_notional = 0.05 * 1.0 * 10_000_000 = 500_000
    # raw_shares = 500_000 / 50 = 10_000; notional = 10_000 * 50 = 500_000 >= 25_000 -> kept.
    cleared = sizer.to_shares(
        weights=np.array([0.05]),
        prices=np.array([50.0]),
        capital=10_000_000.0,
    )
    np.testing.assert_array_equal(cleared, np.array([10_000.0]))


def test_water_filling_all_names_at_capacity_reports_shortfall() -> None:
    # target_notional = [500_000, 300_000, 200_000]
    # capacity = 0.02 * [1_000_000, 100_000, 1_000_000] = [20_000, 2_000, 20_000]
    # final_abs = min(target, capacity) = [20_000, 2_000, 20_000]
    # orig_gross = 1_000_000, achieved = 42_000, shortfall = 958_000, unmet = 0.958.
    # No name has spare capacity, so the water-filling loop breaks immediately.
    sizer = GrossNotionalSizer(
        gross=1.0,
        max_weight=0.5,
        min_trade_notional=0.0,
    )

    result = sizer.to_shares(
        weights=np.array([0.5, 0.3, 0.2]),
        prices=np.array([100.0, 100.0, 100.0]),
        capital=1_000_000.0,
        bar_traded_value=np.array([1_000_000.0, 100_000.0, 1_000_000.0]),
        max_participation=0.02,
    )

    assert isinstance(result, SizingResult)
    assert result.unmet_gross_pct == pytest.approx(0.958)
    np.testing.assert_array_equal(result.shares, np.array([200.0, 20.0, 200.0]))


def test_water_filling_redistributes_shortfall_to_name_with_spare_capacity() -> None:
    # target_notional = [300_000, 100_000]
    # capacity = 0.02 * [100_000, 100_000_000] = [2_000, 2_000_000]
    # initial final_abs = [min(300_000, 2_000), min(100_000, 2_000_000)] = [2_000, 100_000]
    # orig_gross = 400_000, achieved = 102_000, shortfall = 298_000
    # free_mask: name0 final_abs == capacity, name1 has spare capacity.
    # total_free_share = 100_000, add to name1 = 298_000.
    # final_abs = [2_000, 398_000], achieved = 400_000, unmet = 0.0.
    # shares = [2_000 / 100, 398_000 / 100] = [20, 3980] with min_trade_notional=0.0.
    sizer = GrossNotionalSizer(
        gross=1.0,
        max_weight=1.0,
        min_trade_notional=0.0,
    )

    result = sizer.to_shares(
        weights=np.array([0.3, 0.1]),
        prices=np.array([100.0, 100.0]),
        capital=1_000_000.0,
        bar_traded_value=np.array([100_000.0, 100_000_000.0]),
        max_participation=0.02,
    )

    assert isinstance(result, SizingResult)
    assert result.unmet_gross_pct == pytest.approx(0.0)
    np.testing.assert_array_equal(result.shares, np.array([20.0, 3980.0]))
