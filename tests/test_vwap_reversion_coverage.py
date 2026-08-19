"""Coverage for vwap_reversion.py validator edge cases.

Targets final gaps: session_start_time validator (lines 53-54, 56-57).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nifty_quant.strategy.plugins.vwap_reversion import VwapReversionParams


def test_session_start_time_validator_rejects_invalid_format() -> None:
    """Verify session_start_time validator raises ValueError for invalid HH:MM on line 53-54.

    Lines 51-54: try strptime(v, "%H:%M") -> ValueError from strptime is caught
    and re-raised as ValueError("session_start_time must be HH:MM").
    """
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09-15")  # Wrong separator
    assert "session_start_time must be HH:MM" in str(exc_info.value)


def test_session_start_time_validator_rejects_non_time_format() -> None:
    """Verify session_start_time validator rejects gibberish input."""
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="not_a_time")
    assert "session_start_time must be HH:MM" in str(exc_info.value)


def test_session_start_time_validator_rejects_malformed_hour() -> None:
    """Verify session_start_time validator rejects invalid hour."""
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="25:00")
    assert "session_start_time must be HH:MM" in str(exc_info.value)


def test_session_start_time_validator_rejects_malformed_minute() -> None:
    """Verify session_start_time validator rejects invalid minute."""
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09:75")
    assert "session_start_time must be HH:MM" in str(exc_info.value)


def test_session_start_time_validator_rejects_0915_exactly() -> None:
    """Verify session_start_time validator rejects 09:15 exactly (line 56-57).

    The 09:15 bar is structurally broken (close > high, pre-open call auction leak).
    Line 56: if minute <= 555, raise ValueError("session_start_time must be after 09:15").
    """
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09:15")
    assert "session_start_time must be after 09:15" in str(exc_info.value)


def test_session_start_time_validator_rejects_before_0915() -> None:
    """Verify session_start_time validator rejects times before 09:15."""
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09:14")
    assert "session_start_time must be after 09:15" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09:00")
    assert "session_start_time must be after 09:15" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="08:00")
    assert "session_start_time must be after 09:15" in str(exc_info.value)


def test_session_start_time_validator_accepts_0916() -> None:
    """Verify session_start_time validator accepts 09:16 (just after 09:15)."""
    params = VwapReversionParams(session_start_time="09:16")
    assert params.session_start_time == "09:16"


def test_session_start_time_validator_accepts_later_times() -> None:
    """Verify session_start_time validator accepts times after 09:15."""
    params = VwapReversionParams(session_start_time="10:00")
    assert params.session_start_time == "10:00"

    params = VwapReversionParams(session_start_time="15:20")
    assert params.session_start_time == "15:20"

    params = VwapReversionParams(session_start_time="23:59")
    assert params.session_start_time == "23:59"


def test_session_start_time_validator_accepts_default() -> None:
    """Verify default session_start_time passes validation."""
    params = VwapReversionParams()
    assert params.session_start_time == "09:16"


def test_session_start_time_boundary_at_556_minutes() -> None:
    """Verify session_start_time validator accepts exactly minute=556 (09:16).

    Line 55: minute = parsed.hour * 60 + parsed.minute
    For 09:16: minute = 9*60 + 16 = 540 + 16 = 556
    Line 56: if minute <= 555, reject. So 556 is the boundary.
    """
    # 09:16 = 556 minutes, should pass
    params = VwapReversionParams(session_start_time="09:16")
    assert params.session_start_time == "09:16"

    # 09:15 = 555 minutes, should reject
    with pytest.raises(ValidationError) as exc_info:
        VwapReversionParams(session_start_time="09:15")
    assert "session_start_time must be after 09:15" in str(exc_info.value)


def test_target_vol_ann_removed_it_was_dead_and_read_by_nothing() -> None:
    """specs/portfolio_vol_target.md section E: `target_vol_ann` was validated,
    hashed into the config hash, and written into the meta dict, but read by
    nothing (VwapReversionStrategy sizes with sign/sigma clipped to max_weight
    and gross, never referencing it). Dead-code removal, lead-adjudicated."""
    with pytest.raises(ValidationError):
        VwapReversionParams(target_vol_ann=0.15)

    params = VwapReversionParams()
    assert not hasattr(params, "target_vol_ann")
