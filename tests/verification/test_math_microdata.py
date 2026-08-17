from __future__ import annotations

import math

import numpy as np
import pytest

from nifty_quant.features.core import (
    breakout_down,
    breakout_up,
    cross_sectional_zscore,
    log_returns,
    parkinson_volatility,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_zscore,
    volume_zscore,
)
from nifty_quant.features.persistence import variance_ratio


def test_rolling_mean_hand_computed() -> None:
    # Two sessions: rows 0-2 and 3-4. window=2, default min_count=2.
    # x = [1, 3, 7, 100, 200]
    # row 0: only one value in window -> NaN
    # row 1: mean(1, 3) = 2.0
    # row 2: mean(3, 7) = 5.0
    # row 3: session start -> only one value -> NaN
    # row 4: mean(100, 200) = 150.0
    x = np.array([[1.0], [3.0], [7.0], [100.0], [200.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 5], dtype=np.int32)
    result = rolling_mean(x, window=2, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(2.0, rel=1e-9)
    assert result[2, 0] == pytest.approx(5.0, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert result[4, 0] == pytest.approx(150.0, rel=1e-9)


def test_rolling_std_hand_computed() -> None:
    # Two 3-row sessions (day_offsets [0,3,6]). window=3, default min_count=3, ddof=1.
    # x = [1, 2, 3, 10, 20, 30]
    # Session A rows 0-2: mean(1,2,3)=2; sample var =
    #   ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1) = (1+0+1)/2 = 1 -> std=1
    #   rows 0 and 1 have fewer than 3 values -> NaN
    # Session B rows 3-5: mean(10,20,30)=20; sample var =
    #   ((10-20)^2 + 0 + (30-20)^2) / 2 = (100+0+100)/2 = 100 -> std=10
    #   rows 3 and 4 have fewer than 3 values -> NaN
    x = np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = rolling_std(x, window=3, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 0])
    assert result[2, 0] == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert np.isnan(result[4, 0])
    assert result[5, 0] == pytest.approx(10.0, rel=1e-9)


def test_rolling_zscore_hand_computed() -> None:
    # Same sessions and x array as rolling_std. window=3, default min_count=3, ddof=1.
    # x = [1, 2, 3, 10, 20, 30]
    # Session A row 2: rolling_mean=2, rolling_std=1 -> z=(3-2)/1=1
    # Session B row 5: rolling_mean=20, rolling_std=10 -> z=(30-20)/10=1
    x = np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = rolling_zscore(x, window=3, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 0])
    assert result[2, 0] == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert np.isnan(result[4, 0])
    assert result[5, 0] == pytest.approx(1.0, rel=1e-9)


def test_rolling_max_hand_computed() -> None:
    # Two 3-row sessions, window=2, default min_count=2. x = [1, 5, 2, 10, 7, 9]
    # Session A:
    #   row 0: one value -> NaN
    #   row 1: max(1, 5) = 5
    #   row 2: max(5, 2) = 5
    # Session B (reset; row 3 must not use session A):
    #   row 3: one value -> NaN
    #   row 4: max(10, 7) = 10
    #   row 5: max(7, 9) = 9
    x = np.array([[1.0], [5.0], [2.0], [10.0], [7.0], [9.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = rolling_max(x, window=2, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(5.0, rel=1e-9)
    assert result[2, 0] == pytest.approx(5.0, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert result[4, 0] == pytest.approx(10.0, rel=1e-9)
    assert result[5, 0] == pytest.approx(9.0, rel=1e-9)


def test_rolling_min_hand_computed() -> None:
    # Two 3-row sessions, window=2, default min_count=2. x = [5, 1, 4, 10, 7, 9]
    # Session A:
    #   row 0: one value -> NaN
    #   row 1: min(5, 1) = 1
    #   row 2: min(1, 4) = 1
    # Session B:
    #   row 3: one value -> NaN
    #   row 4: min(10, 7) = 7
    #   row 5: min(7, 9) = 7
    x = np.array([[5.0], [1.0], [4.0], [10.0], [7.0], [9.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = rolling_min(x, window=2, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(1.0, rel=1e-9)
    assert result[2, 0] == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert result[4, 0] == pytest.approx(7.0, rel=1e-9)
    assert result[5, 0] == pytest.approx(7.0, rel=1e-9)


def test_log_returns_hand_computed() -> None:
    # Sessions rows 0-2 and 3-4. First row of each session is NaN.
    # close = [100, 110, 121, 200, 180]
    # Session A:
    #   row 0: NaN (session start)
    #   row 1: log(110/100) = log(1.1)
    #   row 2: log(121/110) = log(1.1)
    # Session B:
    #   row 3: NaN (session start)
    #   row 4: log(180/200) = log(0.9)
    close = np.array([[100.0], [110.0], [121.0], [200.0], [180.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 5], dtype=np.int32)
    result = log_returns(close, day_offsets=day_offsets)

    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(math.log(110.0 / 100.0), rel=1e-9)
    assert result[2, 0] == pytest.approx(math.log(121.0 / 110.0), rel=1e-9)
    assert np.isnan(result[3, 0])
    assert result[4, 0] == pytest.approx(math.log(180.0 / 200.0), rel=1e-9)


def test_parkinson_volatility_hand_computed() -> None:
    # Sessions rows 0-2 and 3-4. window=2, min_count=1 explicit so partial windows emit.
    # high/low ratios:
    #   row 0: 200/100=2 -> term ln(2)^2
    #   row 1: 300/100=3 -> term ln(3)^2
    #   row 2: 400/100=4 -> term (2*ln(2))^2 = 4*ln(2)^2
    #   row 3: 500/100=5 -> term ln(5)^2
    #   row 4: 900/100=9 -> term (2*ln(3))^2 = 4*ln(3)^2
    # rolling_mean of those terms over trailing window=2:
    #   row 0: ln(2)^2
    #   row 1: (ln(2)^2 + ln(3)^2)/2
    #   row 2: (ln(3)^2 + 4*ln(2)^2)/2
    #   row 3: ln(5)^2  (session reset)
    #   row 4: (ln(5)^2 + 4*ln(3)^2)/2
    # parkinson = sqrt(rolling_mean / (4*ln(2)))
    high = np.array([[200.0], [300.0], [400.0], [500.0], [900.0]], dtype=np.float64)
    low = np.array([[100.0], [100.0], [100.0], [100.0], [100.0]], dtype=np.float64)
    day_offsets = np.array([0, 3, 5], dtype=np.int32)
    result = parkinson_volatility(
        high, low, window=2, min_count=1, day_offsets=day_offsets
    )

    log2 = math.log(2.0)
    log3 = math.log(3.0)
    log5 = math.log(5.0)

    assert result[0, 0] == pytest.approx(
        math.sqrt((log2**2) / (4.0 * log2)), rel=1e-9
    )
    assert result[1, 0] == pytest.approx(
        math.sqrt(((log2**2 + log3**2) / 2.0) / (4.0 * log2)), rel=1e-9
    )
    assert result[2, 0] == pytest.approx(
        math.sqrt(((log3**2 + 4.0 * log2**2) / 2.0) / (4.0 * log2)), rel=1e-9
    )
    assert result[3, 0] == pytest.approx(
        math.sqrt((log5**2) / (4.0 * log2)), rel=1e-9
    )
    assert result[4, 0] == pytest.approx(
        math.sqrt(((log5**2 + 4.0 * log3**2) / 2.0) / (4.0 * log2)), rel=1e-9
    )


def test_volume_zscore_hand_computed() -> None:
    # deseasonalize=False because with True the first session at each minute is NaN by
    # construction, and a tiny 1-2 session fixture cannot hand-compute expanding per-minute
    # means. With False this is rolling_zscore(volume, window=2, default min_count=2,
    # ddof=1) with two sessions rows 0-2 and 3-4.
    # volume = [1, 5, 2, 20, 10]
    # row 0: one value -> NaN
    # row 1: window [1,5], mean=3, sample var=((1-3)^2+(5-3)^2)/(2-1)=8
    #       std=sqrt(8); z=(5-3)/sqrt(8)=1/sqrt(2)
    # row 2: window [5,2], mean=3.5, sample var=((5-3.5)^2+(2-3.5)^2)/(1)=4.5
    #       std=sqrt(4.5); z=(2-3.5)/sqrt(4.5)=-1/sqrt(2)
    # row 3: session start -> one value -> NaN
    # row 4: window [20,10], mean=15, sample var=((20-15)^2+(10-15)^2)/(1)=50
    #       std=sqrt(50); z=(10-15)/sqrt(50)=-1/sqrt(2)
    volume = np.array([[1.0], [5.0], [2.0], [20.0], [10.0]], dtype=np.float64)
    minute_of_day = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    day_offsets = np.array([0, 3, 5], dtype=np.int32)
    result = volume_zscore(
        volume,
        minute_of_day,
        window=2,
        deseasonalize=False,
        day_offsets=day_offsets,
    )

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert np.isnan(result[0, 0])
    assert result[1, 0] == pytest.approx(inv_sqrt2, rel=1e-9)
    assert result[2, 0] == pytest.approx(-inv_sqrt2, rel=1e-9)
    assert np.isnan(result[3, 0])
    assert result[4, 0] == pytest.approx(-inv_sqrt2, rel=1e-9)


def test_breakout_up_hand_computed() -> None:
    # Two 3-row sessions, window=2, default min_count=2.
    # breakout_up compares close[t] to rolling_max(high) shifted by one bar.
    # high = [10, 12, 11, 20, 18, 19]
    # close = [9, 11, 12.5, 100, 17, 18.5]
    # Session A:
    #   row 0: first bar of session -> False
    #   row 1: shifted rolling_max at row 0 is NaN (only one high) -> False
    #   row 2: shifted rolling_max at row 1 = max(10,12)=12;
    #          close 12.5 > 12 -> True
    # Session B (first bar must be False even though close=100 is above prior highs):
    #   row 3: first bar of session -> False
    #   row 4: shifted rolling_max at row 3 is NaN -> False
    #   row 5: shifted rolling_max at row 4 = max(20,18)=20;
    #          close 18.5 < 20 -> False
    high = np.array(
        [[10.0], [12.0], [11.0], [20.0], [18.0], [19.0]], dtype=np.float64
    )
    close = np.array(
        [[9.0], [11.0], [12.5], [100.0], [17.0], [18.5]], dtype=np.float64
    )
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = breakout_up(close, high, window=2, day_offsets=day_offsets)

    assert not result[0, 0]
    assert not result[1, 0]
    assert result[2, 0]
    assert not result[3, 0]
    assert not result[4, 0]
    assert not result[5, 0]


def test_breakout_down_hand_computed() -> None:
    # Two 3-row sessions, window=2, default min_count=2.
    # breakout_down compares close[t] to rolling_min(low) shifted by one bar.
    # low = [8, 10, 9, 20, 22, 19]
    # close = [11, 9, 7.5, 7, 21, 20.5]
    # Session A:
    #   row 0: first bar -> False
    #   row 1: shifted rolling_min at row 0 is NaN -> False
    #   row 2: shifted rolling_min at row 1 = min(8,10)=8;
    #          close 7.5 < 8 -> True
    # Session B (first bar must be False even though close=7 is below prior lows):
    #   row 3: first bar of session -> False
    #   row 4: shifted rolling_min at row 3 is NaN -> False
    #   row 5: shifted rolling_min at row 4 = min(20,22)=20;
    #          close 20.5 > 20 -> False
    low = np.array([[8.0], [10.0], [9.0], [20.0], [22.0], [19.0]], dtype=np.float64)
    close = np.array(
        [[11.0], [9.0], [7.5], [7.0], [21.0], [20.5]], dtype=np.float64
    )
    day_offsets = np.array([0, 3, 6], dtype=np.int32)
    result = breakout_down(close, low, window=2, day_offsets=day_offsets)

    assert not result[0, 0]
    assert not result[1, 0]
    assert result[2, 0]
    assert not result[3, 0]
    assert not result[4, 0]
    assert not result[5, 0]


def test_cross_sectional_zscore_hand_computed() -> None:
    # cross_sectional_zscore has no day_offsets parameter, so no session-boundary case
    # exists. Row-wise independence is exercised instead. Default min_names=5, so
    # explicit min_names=2 is required for this tiny fixture.
    # x shape (3,4):
    # row 0 [2, 4, nan, nan]:
    #   finite count 2 >= min_names; mean=3; population std (ddof=0)
    #   = sqrt(((2-3)^2 + (4-3)^2)/2) = 1
    #   col 0 z=(2-3)/1=-1; col 1 z=(4-3)/1=1; cols 2-3 NaN
    # row 1 [1, 3, 5, nan]:
    #   finite count 3 >= min_names; mean=3; population std
    #   = sqrt(((1-3)^2 + (3-3)^2 + (5-3)^2)/3) = sqrt(8/3)
    #   col 0 z=(1-3)/std=-2/sqrt(8/3)=-sqrt(3/2); col 1 z=0;
    #   col 2 z=(5-3)/std=+sqrt(3/2); col 3 NaN
    # row 2 [7, nan, nan, nan]:
    #   finite count 1 < min_names -> entire row NaN
    x = np.array(
        [
            [2.0, 4.0, np.nan, np.nan],
            [1.0, 3.0, 5.0, np.nan],
            [7.0, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    result = cross_sectional_zscore(x, min_names=2)

    assert result[0, 0] == pytest.approx(-1.0, rel=1e-9)
    assert result[0, 1] == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(result[0, 2])
    assert np.isnan(result[0, 3])

    std_row1 = math.sqrt(8.0 / 3.0)
    assert result[1, 0] == pytest.approx(-2.0 / std_row1, rel=1e-9)
    assert result[1, 1] == pytest.approx(0.0, abs=1e-9)
    assert result[1, 2] == pytest.approx(2.0 / std_row1, rel=1e-9)
    assert np.isnan(result[1, 3])

    assert np.isnan(result[2, 0])
    assert np.isnan(result[2, 1])
    assert np.isnan(result[2, 2])
    assert np.isnan(result[2, 3])


def test_variance_ratio_hand_computed() -> None:
    # Log-price x, q=2, two sessions rows 0-3 and 4-6. Sessions are independent.
    # Session A length=4, x=[0, 1, 3, 6]:
    #   1-period diffs [1,2,3]; mu=(6-0)/3=2
    #   var1 = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
    #   q-period diffs [3-0, 6-1] = [3, 5]
    #   m = q*(n-q+1)*(1-q/n) = 2*3*0.5 = 3
    #   varq = ((3-4)^2+(5-4)^2)/3 = 2/3
    #   VR = varq/var1 = 1.0; weight = n-1 = 3
    # Session B length=3, x=[0, 1, 0]:
    #   dx=[1,-1]; mu=(0-0)/2=0
    #   var1 = (1^2+(-1)^2)/2 = 1
    #   dq=[0-0]=[0]; q*mu=0
    #   m = 2*(3-2+1)*(1-2/3) = 4/3
    #   varq = 0/(4/3)=0 -> VR=0.0; weight = n-1 = 2
    # Weighted average = (3*1.0 + 2*0.0)/(3+2) = 3/5 = 0.6
    x = np.array([0.0, 1.0, 3.0, 6.0, 0.0, 1.0, 0.0], dtype=np.float64)
    day_offsets = np.array([0, 4, 7], dtype=np.int32)
    result = variance_ratio(x, q=2, day_offsets=day_offsets)

    assert result == pytest.approx(0.6, rel=1e-9)
