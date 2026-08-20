"""TCA blotter record tests -- author A. Written from specs/tca_record.md, INCLUDING
AMENDMENT 1 and AMENDMENT 2 (2026-08-20).

No implementation exists yet: none of the record's new columns (`signal_ts`, `decision_ts`,
`order_ts`, `fill_ts`, `order_id`, `desired_qty`, `filled_qty`, `arrival_price`, `mid`,
`spread_bps`, `impact_bps`, `fees_bps`, `slippage_bps`) exist on `BacktestResult.trades`
today, and `SlippageModel.components(...)` / `SlippageComponents` do not exist on
`fills.py` today. Every test below is EXPECTED to fail or error against the current repo --
that RED state is the deliverable, not a defect in this suite. Nothing here is weakened,
skipped, or wrapped to make it pass, and nothing under src/ was touched to write it.

============================== SPEC DEFECTS -- ORIGINAL DRAFT, NOW RESOLVED ================

This suite was first written against the spec BEFORE Amendment 1 and independently found the
same defect Amendment 1 later confirmed and fixed: obligation 2's original claim that all
four timestamps collapse to one value at `decision_latency_bars = 0` is geometrically
impossible against the spec's own `order_ts` formula (`decision_ts` advanced by
`1 + decision_latency_bars` bars is a *minimum* one-bar advance even at latency 0).
Amendment 1 item 1 confirms this from `engine.py`'s own documented mandatory one-bar
execution lag (`fill_row = t + 1 + decision_latency_bars`, "a weight emitted for bar t is
never filled before t+1") and restates obligation 2. Tests below are written to the
RESTATED obligation 2, not the original.

Amendment 1 also resolves three other things this draft had flagged or worked around:
  - item 2 adds `order_id` (monotonically increasing int64, stable per order) -- tested
    directly below rather than proxied through `(symbol, order_ts, desired_qty)`.
  - item 3 takes multi-bar order splitting OUT of scope for this record: the current engine
    fills a `PendingOrder` exactly once. Obligation 7 is restated as a record-SHAPE test on a
    hand-constructed blotter (not by forcing the engine to split an order), plus a separate
    assertion that today's engine produces exactly one row per (partially-filled) order.
  - item 4 pins `arrival_price` to `open[order_ts]` -- tested directly below.
  - item 5 drops `fill_ratio`; `filled_frac` stays, and obligation 6 asserts on it, including
    verifying it already equals `filled_qty / desired_qty` (per Amendment 1's own
    instruction to check and report rather than assume).

This draft then independently reported two more things, BOTH verified directly by the lead
and BOTH resolved in AMENDMENT 2 rather than worked around here:

  - obligation 10's "preserves the object-dtype `intent`" requirement is unsatisfiable on
    this repo's pandas 3.0.5: a native `str` dtype replaces legacy `object` for string
    columns on parquet round-trip, even when the DataFrame was explicitly cast to `object`
    first (`engine.py:992-1009` does this today and it does not survive persistence).
    Amendment 2 item 1 restates obligation 10: preserve every VALUE exactly, int64 for
    `order_id` and the four timestamp columns, float64 for price/bps columns, and for
    `symbol`/`intent` a STRING dtype via `pandas.api.types.is_string_dtype(...)` (true for
    both `object` and `str`) rather than the specific `object` identity.
  - `desired_qty == 0` cannot arise through this engine (a fully rejected order emits ZERO
    blotter rows, not a row with `desired_qty == 0`). Amendment 2 item 2 resolves this by
    scoping the obligation down to a UNIT test on the ratio computation directly, and
    explicitly forbidding manufacturing the state through the engine (a different kind of
    false confidence from a vacuous assertion, but still false confidence). See that test's
    own docstring for the naming assumption this required.

================================================================================
"""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.portfolio import GrossNotionalSizer
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import FillBatch, NSEIntradayEquityCosts
from nifty_quant.execution.fills import FillModel, SqrtImpactSlippage, ZeroSlippage
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)

_IST = ZoneInfo("Asia/Kolkata")
CAPITAL = 1_000_000.0


# --------------------------------------------------------------------------------
# Shared fixture infrastructure -- follows the same pattern as
# tests/test_engine_adversarial_a.py (hand-verified against the real engine there);
# reproduced independently here, kept local to this file per the task rules.
# --------------------------------------------------------------------------------


def make_grid(
    day_lengths: list[int], start: dt.date = dt.date(2024, 1, 2)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-minute bars starting 09:15 IST per day, one entry in day_lengths per session."""
    dates: list[dt.date] = []
    d = start
    while len(dates) < len(day_lengths):
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)

    ts_chunks = []
    for day, n in zip(dates, day_lengths):
        day_start = pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=_IST)
        idx = pd.date_range(day_start, periods=n, freq="1min")
        idx_utc = idx.tz_convert("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        secs = ((idx_utc - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
        ts_chunks.append(secs)
    ts = np.concatenate(ts_chunks).astype(np.int64) if ts_chunks else np.empty(0, dtype=np.int64)
    day_offsets = np.concatenate([[0], np.cumsum(day_lengths)]).astype(np.int32)
    dates_arr = np.array(dates, dtype=object)
    return ts, day_offsets, dates_arr


def make_panel(
    symbols: tuple[str, ...],
    ts: np.ndarray,
    day_offsets: np.ndarray,
    dates: np.ndarray,
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> Panel:
    """Build a Panel from explicit float64 arrays, cast to float32 at rest like real data."""
    fields = {
        "open": np.asarray(open_, dtype=np.float64).astype(np.float32),
        "high": np.asarray(high, dtype=np.float64).astype(np.float32),
        "low": np.asarray(low, dtype=np.float64).astype(np.float32),
        "close": np.asarray(close, dtype=np.float64).astype(np.float32),
        "volume": np.asarray(volume, dtype=np.float64).astype(np.float32),
    }
    return Panel(fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates)


def row_at(day_offsets: np.ndarray, day: int, hhmm: str) -> int:
    """Row index of a given HH:MM within `day`, assuming the session starts 09:15."""
    hour, minute = (int(p) for p in hhmm.split(":"))
    minute_of_day = hour * 60 + minute
    return int(day_offsets[day]) + (minute_of_day - 9 * 60 - 15)


class _EmptyParams(BaseModel):
    pass


class TsScriptStrategy(Strategy):
    """Returns a scripted TargetPortfolio keyed by the decision's cursor timestamp.

    `state.ts` is the ts of the cursor bar (decision row t's cursor is t-1), NOT the
    decision row's own ts. `script` maps that cursor ts (plain int) to a TargetPortfolio;
    any decision whose cursor ts is not a key returns None (hold).
    """

    name = "ts_script"
    Params = _EmptyParams

    def __init__(
        self,
        params: BaseModel,
        script: dict[int, TargetPortfolio],
        *,
        decision_times: tuple[str, ...] | None,
        needs_intrabar_risk: bool = False,
    ) -> None:
        super().__init__(params)
        self._script = script
        self._decision_times = decision_times
        self._needs_intrabar_risk = needs_intrabar_risk

    def data_request(self) -> DataRequest:
        return DataRequest(
            decision_times=self._decision_times,
            needs_intrabar_risk=self._needs_intrabar_risk,
        )

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(
        self, view: MarketView, signals: dict, state: PortfolioState
    ) -> TargetPortfolio | None:
        return self._script.get(int(state.ts))


def default_config(**overrides) -> BacktestConfig:
    kwargs = dict(
        capital=CAPITAL,
        fill_model=FillModel(slippage=ZeroSlippage()),
        cost_model=NSEIntradayEquityCosts(),
        sizer=GrossNotionalSizer(),
    )
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


def _simple_single_fill_result(*, latency: int = 0, n_bars: int = 20, decision_hhmm: str = "09:20"):
    """One session, ample volume, a single entry decision -- a same-bar, full fill."""
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([n_bars])
    n_rows = len(ts)
    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e9)
    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )
    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {int(ts[row_at(day_offsets, 0, decision_hhmm) - 1]): target}
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=(decision_hhmm,))
    config = default_config(decision_latency_bars=latency)
    result = run_backtest(strategy, panel, config)
    return result, ts, day_offsets


# --------------------------------------------------------------------------------
# Obligation 1: four timestamps present, int64, ordered.
# --------------------------------------------------------------------------------


def test_obligation1_four_timestamps_present_int64_and_ordered():
    result, ts, day_offsets = _simple_single_fill_result(latency=0)
    trades = result.trades
    assert len(trades) > 0

    for col in ("signal_ts", "decision_ts", "order_ts", "fill_ts"):
        assert col in trades.columns, f"missing column {col!r}"
        assert trades[col].dtype == np.int64, f"{col} dtype is {trades[col].dtype}, expected int64"

    signal_ts = trades["signal_ts"].to_numpy()
    decision_ts = trades["decision_ts"].to_numpy()
    order_ts = trades["order_ts"].to_numpy()
    fill_ts = trades["fill_ts"].to_numpy()

    assert np.all(signal_ts <= decision_ts)
    assert np.all(decision_ts <= order_ts)
    assert np.all(order_ts <= fill_ts)


# --------------------------------------------------------------------------------
# Obligation 2 -- RESTATED by AMENDMENT 1 item 1: there is no same-bar fill in this
# engine (mandatory one-bar execution lag: `fill_row = t + 1 + decision_latency_bars`,
# "a weight emitted for bar t is never filled before t+1", engine.py module docstring).
# At decision_latency_bars = 0: signal_ts == decision_ts, and order_ts == fill_ts ==
# the label of the NEXT bar along the session index.
# --------------------------------------------------------------------------------


def test_obligation2_signal_equals_decision_order_equals_fill_next_bar_under_zero_latency():
    result, ts, day_offsets = _simple_single_fill_result(latency=0)
    trades = result.trades
    aaa = trades[trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    row = aaa.iloc[0]

    assert row["signal_ts"] == row["decision_ts"]
    assert row["order_ts"] == row["fill_ts"]
    assert row["order_ts"] != row["decision_ts"], (
        "the one-bar execution lag is structural and must be visible in the record"
    )

    decision_row = int(np.searchsorted(ts, row["decision_ts"]))
    expected_order_row = decision_row + 1
    assert row["order_ts"] == int(ts[expected_order_row])


# --------------------------------------------------------------------------------
# AMENDMENT 1 item 2: order_id is a monotonically increasing int64, stable across
# every fill row an order produces. The single-fill fixture's entry and forced EOD
# exit are two DIFFERENT orders, so they must get two DIFFERENT, increasing ids.
# --------------------------------------------------------------------------------


def test_order_id_present_int64_monotonic_and_distinct_per_order():
    result, ts, day_offsets = _simple_single_fill_result(latency=0)
    trades = result.trades
    assert "order_id" in trades.columns
    assert trades["order_id"].dtype == np.int64

    aaa = trades[trades["symbol"] == "AAA"].sort_values("ts")
    assert len(aaa) >= 2, "entry + forced EOD exit must both be present"
    order_ids = aaa["order_id"].to_numpy()
    assert len(np.unique(order_ids)) == len(order_ids), "distinct orders must get distinct id"
    assert np.all(np.diff(order_ids) > 0), "order_id must increase monotonically with creation"


# --------------------------------------------------------------------------------
# Obligation 3: order_ts advances decision_ts by 1+latency bars ALONG THE SESSION
# INDEX, verified on a 60-bar Muhurat-shaped session (never 375).
# --------------------------------------------------------------------------------


def test_obligation3_order_ts_advances_by_latency_plus_one_along_short_session():
    """60-bar Muhurat-shaped session (never 375 -- repo rule 5). An implementation that
    bakes in a fixed bars-per-day constant anywhere in the decision_ts -> order_ts row
    mapping would either index out of this 60-row panel's bounds or silently compute the
    wrong ts, since a real Muhurat session genuinely only has 60 rows to work with."""
    symbols = ("AAA", "BBB")
    N_BARS = 60
    ts, day_offsets, dates = make_grid([N_BARS])
    n_rows = len(ts)
    assert n_rows == 60

    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e9)
    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )

    decision_hhmm = "09:20"
    decision_row = row_at(day_offsets, 0, decision_hhmm)  # == 5
    latency = 3
    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {int(ts[decision_row - 1]): target}
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=(decision_hhmm,))
    config = default_config(decision_latency_bars=latency)
    result = run_backtest(strategy, panel, config)

    aaa = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa) > 0

    expected_order_row = decision_row + 1 + latency
    assert expected_order_row < n_rows, "fixture must keep the order row inside the 60-bar session"
    expected_order_ts = int(ts[expected_order_row])
    assert int(aaa["order_ts"].iloc[0]) == expected_order_ts


# --------------------------------------------------------------------------------
# Obligation 4: slippage_bps == spread_bps + impact_bps, on random notionals.
# Unit-level against SlippageModel directly -- no backtest needed.
# --------------------------------------------------------------------------------


def test_obligation4_slippage_components_sum_to_bps_random_notionals():
    rng = np.random.default_rng(20260820)
    notional = rng.uniform(1.0, 1_000_000.0, size=500)
    bar_traded_value = rng.uniform(1_000.0, 1e9, size=500)

    model = SqrtImpactSlippage()
    components = model.components(notional, bar_traded_value)
    bps = model.bps(notional, bar_traded_value)

    summed = components.spread_bps + components.impact_bps
    # Spec: "to within 1e-9 relative". bps() is required to be *redefined* as this exact
    # sum (computed once), so exact equality would also be justified, but we assert the
    # spec's own stated tolerance rather than a stricter bound this test wasn't asked for.
    np.testing.assert_allclose(bps, summed, rtol=1e-9, atol=0.0)


def test_slippage_components_dataclass_is_frozen():
    model = SqrtImpactSlippage()
    components = model.components(np.array([1000.0]), np.array([1e6]))
    with pytest.raises(FrozenInstanceError):
        components.spread_bps = np.array([0.0])


# --------------------------------------------------------------------------------
# Obligation 5: fees_bps == charges / notional * 1e4, via as_bps_of.
# --------------------------------------------------------------------------------


def test_obligation5a_fees_bps_equals_charges_over_notional_direct():
    rng = np.random.default_rng(7)
    notional = rng.uniform(1_000.0, 500_000.0, size=50)
    is_buy = rng.random(50) > 0.5
    n_orders = np.ones(50)
    fills = FillBatch(notional=notional, is_buy=is_buy, n_orders=n_orders)
    charges = NSEIntradayEquityCosts().charges(fills)

    expected = charges.total / notional * 1e4
    actual = charges.as_bps_of(notional)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_obligation5b_blotter_fees_bps_matches_charges_over_notional():
    result, ts, day_offsets = _simple_single_fill_result(latency=0)
    trades = result.trades
    aaa = trades[trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    assert "fees_bps" in trades.columns

    fees_bps = aaa["fees_bps"].to_numpy(dtype=np.float64)
    charges = aaa["charges"].to_numpy(dtype=np.float64)
    notional = aaa["notional"].to_numpy(dtype=np.float64)
    expected = np.where(notional != 0.0, charges / notional * 1e4, 0.0)
    np.testing.assert_allclose(fees_bps, expected, rtol=1e-9)


# --------------------------------------------------------------------------------
# Obligation 6 -- AMENDMENT 1 item 5: fill_ratio is NOT added; assert on the existing
# filled_frac instead, including verifying it already equals filled_qty / desired_qty.
# Also folds in Amendment 1 item 3's second half: today's engine fills a PendingOrder
# exactly once, so a partially-filled order must produce exactly ONE row.
# --------------------------------------------------------------------------------


def test_obligation6_partial_fill_desired_qty_greater_than_filled_and_remainder_recoverable():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)
    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)
    volume = np.full((n_rows, 2), 1e6)

    decision_row = row_at(day_offsets, 0, "09:20")
    fill_row = decision_row + 1  # latency 0 -> fill_row = decision_row + 1
    volume[fill_row, 0] = 50.0  # starve the fill row so participation caps the fill

    script = {int(ts[decision_row - 1]): TargetPortfolio(weights=np.array([0.10, 0.0]))}
    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=("09:20",))
    config = default_config(fill_model=FillModel(slippage=ZeroSlippage(), max_participation=0.02))
    result = run_backtest(strategy, panel, config)

    trades = result.trades
    aaa = trades[trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    row = aaa.iloc[0]

    assert row["desired_qty"] > row["filled_qty"]
    assert row["filled_frac"] < 1.0
    remainder = row["desired_qty"] - row["filled_qty"]
    assert remainder > 0.0
    # AMENDMENT 1 item 5: verify filled_frac already equals filled_qty / desired_qty
    # rather than assuming it -- a near-duplicate column with subtly different
    # semantics is how a blotter stops being trustworthy.
    assert row["filled_frac"] == pytest.approx(row["filled_qty"] / row["desired_qty"], rel=1e-9)

    # AMENDMENT 1 item 3: the current engine fills a PendingOrder exactly once --
    # a partial fill must show up as exactly one row, not several.
    assert len(aaa) == 1


# --------------------------------------------------------------------------------
# Obligation 7 -- RESTATED by AMENDMENT 1 item 3: multi-bar fills are OUT of scope
# for this record; the current engine never splits one PendingOrder across bars.
# Assert the record SHAPE supports multiple fill rows per order_id with distinct
# fill_ts summing to the total, on a HAND-CONSTRUCTED blotter -- not by forcing the
# engine to split an order. This intentionally does not exercise src/ code: it is a
# schema-capability check for a shape the engine does not yet produce (splitting is
# explicitly deferred, not silently dropped, per the amendment).
# --------------------------------------------------------------------------------


def test_obligation7_record_shape_supports_multiple_fill_rows_per_order_id():
    hand_built_blotter = pd.DataFrame(
        {
            "order_id": np.array([7, 7, 7], dtype=np.int64),
            "symbol": ["AAA", "AAA", "AAA"],
            "fill_ts": np.array([1_700_000_060, 1_700_000_120, 1_700_000_180], dtype=np.int64),
            "desired_qty": np.array([1000.0, 1000.0, 1000.0], dtype=np.float64),
            "filled_qty": np.array([400.0, 350.0, 250.0], dtype=np.float64),
        }
    )

    by_order = hand_built_blotter.groupby("order_id")
    n_distinct_fill_ts = by_order["fill_ts"].nunique().loc[7]
    assert n_distinct_fill_ts == 3, "one order_id must be able to span 3 distinct fill_ts"

    total_filled = by_order["filled_qty"].sum().loc[7]
    assert total_filled == pytest.approx(1000.0)
    assert total_filled == pytest.approx(
        hand_built_blotter["desired_qty"].iloc[0]
    ), "in this hand-built example the split fills exactly exhaust desired_qty"


# --------------------------------------------------------------------------------
# AMENDMENT 1 item 4: arrival_price is pinned to open[order_ts], the first price
# available once the order is live.
# --------------------------------------------------------------------------------


def test_arrival_price_pinned_to_open_at_order_ts():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)
    DISTINCTIVE_OPEN = 111.5
    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)

    decision_row = row_at(day_offsets, 0, "09:20")
    order_row = decision_row + 1  # latency 0 -> order_ts is the next bar
    open_[order_row, 0] = DISTINCTIVE_OPEN
    volume = np.full((n_rows, 2), 1e9)

    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )
    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {int(ts[decision_row - 1]): target}
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=("09:20",))
    result = run_backtest(strategy, panel, default_config())

    aaa = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    row = aaa.iloc[0]
    assert row["arrival_price"] == pytest.approx(DISTINCTIVE_OPEN)


# --------------------------------------------------------------------------------
# Obligation 8: mid is NaN when unavailable, never fabricated from close.
# --------------------------------------------------------------------------------


def test_obligation8_mid_is_nan_and_never_equals_close():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)
    DISTINCTIVE_CLOSE = 137.25  # non-round value -- an accidental mid=close would be unambiguous
    open_ = np.full((n_rows, 2), DISTINCTIVE_CLOSE)
    high = np.full((n_rows, 2), DISTINCTIVE_CLOSE)
    low = np.full((n_rows, 2), DISTINCTIVE_CLOSE)
    close = np.full((n_rows, 2), DISTINCTIVE_CLOSE)
    volume = np.full((n_rows, 2), 1e9)
    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )
    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {int(ts[row_at(day_offsets, 0, "09:20") - 1]): target}
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=("09:20",))
    result = run_backtest(strategy, panel, default_config())

    trades = result.trades
    assert "mid" in trades.columns
    mid = trades["mid"].to_numpy(dtype=np.float64)
    assert np.all(np.isnan(mid)), "mid must be NaN when no genuine mid is available"
    assert not np.any(mid == DISTINCTIVE_CLOSE), "mid must never be silently set to close"


# --------------------------------------------------------------------------------
# Obligation 9: ZeroSlippage gives spread_bps == impact_bps == slippage_bps == 0,
# while fees_bps may be non-zero.
# --------------------------------------------------------------------------------


def test_obligation9_zero_slippage_components_all_zero_fees_may_be_nonzero():
    model = ZeroSlippage()
    notional = np.array([10_000.0, 250_000.0, 0.0])
    bar_traded_value = np.array([1e6, 1e6, 1e6])

    components = model.components(notional, bar_traded_value)
    assert np.all(components.spread_bps == 0.0)
    assert np.all(components.impact_bps == 0.0)

    bps = model.bps(notional, bar_traded_value)
    assert np.all(bps == 0.0)

    # fees_bps is independent of the slippage model -- demonstrate non-zero fees under
    # ZeroSlippage via the same as_bps_of path obligation 5 exercises.
    is_buy = np.array([True, True, True])
    n_orders = np.array([1.0, 1.0, 1.0])
    fills = FillBatch(notional=notional, is_buy=is_buy, n_orders=n_orders)
    charges = NSEIntradayEquityCosts().charges(fills)
    fees_bps = charges.as_bps_of(notional)
    assert fees_bps[0] > 0.0
    assert fees_bps[1] > 0.0


def test_obligation9_blotter_zero_slippage_gives_zero_slippage_bps_end_to_end():
    result, ts, day_offsets = _simple_single_fill_result(latency=0)  # uses ZeroSlippage
    trades = result.trades
    aaa = trades[trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    assert np.all(aaa["spread_bps"].to_numpy(dtype=np.float64) == 0.0)
    assert np.all(aaa["impact_bps"].to_numpy(dtype=np.float64) == 0.0)
    assert np.all(aaa["slippage_bps"].to_numpy(dtype=np.float64) == 0.0)
    # charges are non-zero (brokerage/STT/etc apply regardless of slippage model)
    assert np.any(aaa["charges"].to_numpy(dtype=np.float64) > 0.0)


# --------------------------------------------------------------------------------
# Obligation 10 -- RESTATED by AMENDMENT 2 item 1: round-trip through parquet must
# preserve every VALUE exactly; int64 for order_id and the four timestamp columns;
# float64 for price/bps columns; and a STRING dtype (not necessarily legacy object)
# for symbol/intent, via pandas.api.types.is_string_dtype. Legacy object identity is
# NOT asserted -- verified unsatisfiable on this repo's pandas 3.0.5 even for the
# already-existing symbol/intent columns cast to object today in engine.py.
# --------------------------------------------------------------------------------


def test_obligation10_blotter_roundtrip_parquet_preserves_values_and_dtypes(tmp_path):
    result, ts, day_offsets = _simple_single_fill_result(latency=0)
    trades = result.trades
    assert len(trades) > 0

    path = tmp_path / "result.parquet"
    trades.to_parquet(path)
    reloaded = pd.read_parquet(path)

    # (a) every value preserved exactly -- dtype checked separately below since
    # symbol/intent are permitted to change string dtype (object <-> str) on this
    # pandas version without that counting as a value or a preservation failure.
    pd.testing.assert_frame_equal(trades, reloaded, check_dtype=False)

    # (b) int64 for order_id and all four timestamp columns
    for col in ("order_id", "signal_ts", "decision_ts", "order_ts", "fill_ts"):
        assert reloaded[col].dtype == np.int64, f"{col} dtype not preserved as int64"
    assert reloaded["ts"].dtype == np.int64  # existing column, unaffected by the amendment

    # (c) float64 for price and bps columns
    for col in (
        "price",
        "decision_price",
        "fill_price",
        "arrival_price",
        "mid",
        "shortfall_bps",
        "spread_bps",
        "impact_bps",
        "fees_bps",
        "slippage_bps",
    ):
        assert reloaded[col].dtype == np.float64, f"{col} dtype not preserved as float64"

    # (d) symbol/intent: a string dtype, not necessarily legacy object
    for col in ("symbol", "intent"):
        assert pd.api.types.is_string_dtype(reloaded[col]), f"{col} must be a string dtype"


# --------------------------------------------------------------------------------
# AMENDMENT 2 item 2: desired_qty == 0 cannot arise through the engine (a fully
# rejected order emits zero blotter rows -- confirmed against the existing
# `test_zero_bar_traded_value_rejects_order` fixture pattern in
# tests/test_engine_adversarial_a.py). Tested here as a UNIT on the ratio computation
# directly, per the amendment's explicit instruction NOT to manufacture the state
# through run_backtest -- doing so would test a path that cannot occur, which is its
# own kind of false confidence.
#
# NAMING ASSUMPTION, flagged rather than hidden: no standalone, importable ratio
# function exists in the repo today. The filled_qty / desired_qty divide Amendment 1
# item 5 requires filled_frac to satisfy is currently inline, scalar Python
# arithmetic inside `_record_trade`, a closure local to `run_backtest` (engine.py) --
# not a public symbol, and today its zero-desired-qty branch returns 0.0, not NaN
# (see engine.py's `filled_frac = 0.0` else-branch). This test targets the smallest
# reasonable public surface a correct fix would need to add to make this obligation
# unit-testable at all without going through the engine: a module-level, vectorized
# `compute_filled_frac(filled_qty, desired_qty)` in `nifty_quant.backtest.engine`,
# mirroring `Charges.as_bps_of` (costs.py:116), which is exactly this shape (a
# divide-by-zero-guarded elementwise ratio) and already exists as a public method. If
# the implementer places or names it differently, this import fails for a naming
# reason rather than a behavioural one -- that mismatch is itself worth reporting.
# --------------------------------------------------------------------------------


def test_filled_frac_ratio_is_nan_when_desired_qty_is_zero_unit():
    from nifty_quant.backtest.engine import compute_filled_frac

    filled_qty = np.array([0.0, 400.0, -200.0], dtype=np.float64)
    desired_qty = np.array([0.0, 1000.0, -1000.0], dtype=np.float64)
    ratio = compute_filled_frac(filled_qty, desired_qty)

    assert np.isnan(ratio[0]), "desired_qty == 0 must give NaN, not 0.0, not a ZeroDivisionError"
    assert ratio[1] == pytest.approx(0.4)
    assert ratio[2] == pytest.approx(0.2)


# --------------------------------------------------------------------------------
# Required invariant (spec "Cost decomposition" section): shortfall_bps must NOT be
# forced to equal slippage_bps + fees_bps -- the gap is the model's error and must
# stay visible. Constructed via a deliberate price jump between decision and fill so
# the two provably diverge.
# --------------------------------------------------------------------------------


def test_shortfall_bps_not_forced_equal_to_modeled_slippage_plus_fees():
    symbols = ("AAA", "BBB")
    ts, day_offsets, dates = make_grid([20])
    n_rows = len(ts)
    open_ = np.full((n_rows, 2), 100.0)
    high = np.full((n_rows, 2), 100.0)
    low = np.full((n_rows, 2), 100.0)
    close = np.full((n_rows, 2), 100.0)

    decision_row = row_at(day_offsets, 0, "09:20")
    fill_row = decision_row + 1
    # Pure drift, no cost model would predict this: a ~6% jump between the decision
    # reference and the fill bar's price.
    close[fill_row:, :] = 106.0
    open_[fill_row:, :] = 106.0
    high[fill_row:, :] = 106.0
    low[fill_row:, :] = 106.0
    volume = np.full((n_rows, 2), 1e9)

    panel = make_panel(
        symbols, ts, day_offsets, dates, open_=open_, high=high, low=low, close=close, volume=volume
    )
    target = TargetPortfolio(weights=np.array([0.10, 0.0]), meta={})
    script = {int(ts[decision_row - 1]): target}
    strategy = TsScriptStrategy(_EmptyParams(), script, decision_times=("09:20",))
    config = default_config(
        fill_model=FillModel(slippage=SqrtImpactSlippage(), max_participation=0.02)
    )
    result = run_backtest(strategy, panel, config)

    aaa = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa) > 0
    row = aaa.iloc[0]

    modeled_total = row["slippage_bps"] + row["fees_bps"]
    assert row["shortfall_bps"] != pytest.approx(modeled_total, rel=1e-6, abs=1e-6)
    # a ~6% price jump (~600 bps) dwarfs any bps-level modeled cost
    assert abs(row["shortfall_bps"] - modeled_total) > 50.0
