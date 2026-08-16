"""Tests for static universe definitions and filters against the real repo data."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from nifty_quant.universe.filters import combine, coverage_filter, liquidity_filter
from nifty_quant.universe.static import (
    INDEX_SYMBOLS,
    Universe,
    all_data_symbols,
    equity_symbols,
    load_universe,
    survivorship_report,
)


@pytest.fixture(scope="module")
def all_equity() -> Universe:
    return Universe(name="all_equity", symbols=equity_symbols())


def test_symbol_counts_and_index_exclusion() -> None:
    all_syms = all_data_symbols()
    assert len(all_syms) == 153

    equities = equity_symbols()
    assert len(equities) == 149
    assert not (INDEX_SYMBOLS & set(equities))
    for index_symbol in INDEX_SYMBOLS:
        assert index_symbol not in equities


def test_as_of_proxy_grows(all_equity: Universe) -> None:
    as_of_2018 = all_equity.as_of(date(2018, 1, 1))
    as_of_2025 = all_equity.as_of(date(2025, 1, 1))

    assert len(as_of_2018.symbols) < len(as_of_2025.symbols)
    assert as_of_2018.source == "availability_proxy"
    assert as_of_2018.as_of_date == date(2018, 1, 1)


def test_reliance_available_through_range(all_equity: Universe) -> None:
    for year in range(2018, 2026):
        as_of = all_equity.as_of(date(year, 6, 1))
        assert "RELIANCE" in as_of.symbols


def test_mask_aligns_to_input() -> None:
    universe = Universe(name="small", symbols=("A", "B", "C"))
    mask = universe.mask(["A", "X", "B", "Y", "C"])

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, False, True]


def test_survivorship_report_has_diagnostic_string(
    all_equity: Universe,
) -> None:
    report = survivorship_report(
        all_equity, date(2017, 1, 1), date(2026, 1, 1)
    )

    assert report.per_year[2017]["pct_active"] < report.per_year[2025]["pct_active"]

    warning = report.warning_line()
    assert warning
    assert any(ch.isdigit() for ch in warning)


def test_coverage_filter_strictness() -> None:
    symbols = equity_symbols()

    strict = coverage_filter(
        symbols, date(2017, 1, 1), date(2026, 1, 1), min_coverage=1.0
    )
    relaxed = coverage_filter(
        symbols, date(2017, 1, 1), date(2026, 1, 1), min_coverage=0.5
    )

    assert len(strict) < len(relaxed)


def test_liquidity_filter_keeps_above_threshold() -> None:
    symbols = ("S1", "S2", "S3", "S4")
    adv_by_symbol = {
        "S1": 100_000_000.0,
        "S2": 50_000_000.0,
        "S3": 49_999_999.0,
        "S4": 10_000_000.0,
    }

    kept = liquidity_filter(symbols, adv_by_symbol, min_adv_inr=50_000_000.0)

    assert kept == ("S1", "S2")


def test_combine_and_validation() -> None:
    a = np.array([True, False, True, False])
    b = np.array([True, True, False, False])

    assert combine(a, b).tolist() == [True, False, False, False]

    with pytest.raises(ValueError):
        combine(a, np.array([True, False, True]))

    with pytest.raises(ValueError):
        combine()


def test_all_equity_yaml_roundtrips() -> None:
    loaded = load_universe("all_equity")
    assert loaded.symbols == equity_symbols()
