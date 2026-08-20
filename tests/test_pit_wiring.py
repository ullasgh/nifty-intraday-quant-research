"""Coverage for `nifty_quant.universe.pit.eligibility_mask_to_bars`.

Not part of the frozen two-suite contract (test_pit_universe_a/b.py) -- this
is my own regression test for the CLI-wiring helper that broadcasts a
per-session eligibility mask to bar level for combination with
`data/validate.py`'s bar-level `tradable_mask`.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from nifty_quant.data.panel import Panel
from nifty_quant.universe.pit import compute_eligibility, eligibility_mask_to_bars
from nifty_quant.universe.static import Universe


def _session_ts(session_date: date, n_bars: int) -> np.ndarray:
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.combine(session_date, time(9, 15), tzinfo=ist)
    return np.asarray(
        [int(start.timestamp()) + 60 * i for i in range(n_bars)], dtype=np.int64
    )


def _make_panel(n_sessions: int, symbols: tuple[str, ...], bars_per_session: int = 3) -> Panel:
    session_dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n_sessions)]
    n_symbols = len(symbols)
    close_by_session = np.full((n_sessions, n_symbols), 100.0, dtype=np.float64)
    volume_by_session = np.full((n_sessions, n_symbols), 1.0e7, dtype=np.float64)

    close = np.repeat(close_by_session, bars_per_session, axis=0)
    volume = np.repeat(volume_by_session, bars_per_session, axis=0)

    ts = np.concatenate([_session_ts(d, bars_per_session) for d in session_dates])
    day_offsets = np.arange(
        0, (n_sessions + 1) * bars_per_session, bars_per_session, dtype=np.int32
    )

    return Panel(
        fields={
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        symbols=symbols,
        ts=ts,
        day_offsets=day_offsets,
        dates=np.asarray(session_dates, dtype=object),
    )


def test_broadcast_matches_session_mask_at_every_bar_of_the_session() -> None:
    n_sessions = 10
    bars_per_session = 3
    panel = _make_panel(n_sessions, ("AAA", "BBB"), bars_per_session)

    eligibility = compute_eligibility(
        panel, name="wiring-test", min_history_sessions=2, min_adv_inr=5.0e7
    )
    bar_mask = eligibility_mask_to_bars(panel, eligibility)

    assert bar_mask.shape == (panel.n_rows(), panel.n_symbols())
    assert bar_mask.dtype == np.bool_

    day_offsets = panel.day_offsets
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        for col, sym in enumerate(panel.symbols):
            expected = eligibility.mask[session_idx, col]
            assert np.all(bar_mask[start:end, col] == expected), (
                f"session {session_idx} symbol {sym}: bar-level mask does not match "
                f"session-level eligibility"
            )


def test_symbol_excluded_by_universe_restriction_is_ineligible_at_every_bar() -> None:
    n_sessions = 6
    panel = _make_panel(n_sessions, ("KEEP", "DROP"))
    universe = Universe(name="only_keep", symbols=("KEEP",))

    eligibility = compute_eligibility(
        panel, universe, min_history_sessions=1, min_adv_inr=0.0
    )
    bar_mask = eligibility_mask_to_bars(panel, eligibility)

    drop_col = panel.symbols.index("DROP")
    assert not np.any(bar_mask[:, drop_col])

    keep_col = panel.symbols.index("KEEP")
    assert np.any(bar_mask[:, keep_col])
