"""Point-in-time universe eligibility.

`compute_eligibility` answers "is this name part of the RESEARCH universe on
session d", causally, from bar presence and trailing liquidity alone -- it is
a **data availability proxy**, not real index membership: this repository has
no point-in-time membership history for NIFTY or any other index. It is also
**not survivorship correction**: this dataset contains ZERO delisted names
(every symbol present here runs to the end of the window), so no
`PitEligibility` result may ever be read as having corrected for survivorship
bias. What it does fix is the mirror-image problem -- newly listed names
carrying too little history to trust their cross-sectional signal.

Eligibility on session `d` requires, using only data strictly before `d`:
  1. LISTED -- at least `min_history_sessions` PRIOR sessions with a present
     bar (not necessarily contiguous, not necessarily including session 0).
  2. LIQUID -- trailing 20-CALENDAR-session ADV (rupee turnover, strictly
     prior, current session excluded) at least `min_adv_inr`.

Eligibility does NOT require a bar on session `d` itself. `present` (a bar
exists that day), `tradable` (a bar exists and is usable), and ELIGIBLE (in
the research universe as of `d`) are three distinct concepts (CLAUDE.md rule
7) and must never be conflated: a name can be eligible, present, and
tradable; eligible and absent (it listed weeks ago but simply has no bar
today); or present and tradable yet ineligible (it listed last week).

`min_history_sessions` and `min_adv_inr` are REQUIRED keyword arguments with
no default -- per CLAUDE.md rule 8, no cutoff in this codebase is a
hand-chosen constant; the caller must supply a value it can justify (or, for
the "collapse to the existing continuous-coverage control" case, pass
`n_sessions - 1`, which is the value that makes eligibility require presence
in effectively every prior session).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from nifty_quant.data.liquidity import compute_prior_adv_by_session
from nifty_quant.data.panel import Panel
from nifty_quant.universe.static import Universe

_ALLOWED_SOURCES = ("availability_proxy", "index_membership")


class _HashStr(str):
    """A string that is also callable, returning itself unchanged.

    Two independently-written, immutable test suites disagree on whether
    `PitEligibility.universe_hash` is an attribute compared directly
    (`result1.universe_hash == result2.universe_hash`) or a method that must
    be called (`result.universe_hash()`). Both usages are satisfied by a str
    subclass whose `__call__` returns `self`: as an attribute it behaves and
    compares exactly like the hash string it is; called, it returns that same
    string unchanged.
    """

    def __call__(self) -> "_HashStr":
        return self


@dataclass(frozen=True)
class PitEligibility:
    """Result of `compute_eligibility`. See module docstring for semantics.

    `mask`: bool, shape (n_sessions, n_symbols), aligned to `dates` (rows) and
    to the symbol order of the panel (and, if supplied, the universe) used to
    compute it -- NOT to the sorted `symbols` field below.

    `symbols`: the eligible-computation's full symbol set, sorted -- kept
    sorted (not just sorted for hashing) so two results compare equal only
    when they truly refer to the same set, never merely hash equal while the
    field itself differs in order.

    `column_symbols`: the symbol name for each `mask` COLUMN, in `mask`'s
    actual column order (unsorted) -- the only field that lets a caller map a
    column index back to a symbol name; `symbols` cannot be used for that,
    since it is deliberately sorted and therefore not aligned to `mask`.

    `universe_hash`: stable hash over (name, source, min_history_sessions,
    min_adv_inr, sorted symbol set, sorted eligible-symbol-set per session).
    Order of symbol *columns* never enters it -- only symbol *names* do -- so
    it is invariant to how the caller ordered the panel's columns.
    """

    mask: np.ndarray
    symbols: tuple[str, ...]
    column_symbols: tuple[str, ...]
    dates: np.ndarray
    source: str
    name: str
    min_history_sessions: int
    min_adv_inr: float
    universe_hash: _HashStr


def _universe_hash_string(
    *,
    name: str,
    source: str,
    min_history_sessions: int,
    min_adv_inr: float,
    symbols_sorted: tuple[str, ...],
    column_symbols: tuple[str, ...],
    mask: np.ndarray,
) -> str:
    per_session_eligible: list[list[str]] = []
    for row in mask:
        eligible_names = sorted(
            sym for sym, ok in zip(column_symbols, row.tolist()) if ok
        )
        per_session_eligible.append(eligible_names)

    payload = {
        "name": name,
        "source": source,
        "min_history_sessions": int(min_history_sessions),
        "min_adv_inr": float(min_adv_inr),
        "symbols": list(symbols_sorted),
        "eligible_by_session": per_session_eligible,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_eligibility(
    panel: Panel,
    universe: Universe | None = None,
    *,
    name: str | None = None,
    min_history_sessions: int,
    min_adv_inr: float,
    source: str = "availability_proxy",
) -> PitEligibility:
    """Compute the point-in-time eligibility mask for `panel`.

    `universe`, if given, restricts the computation to symbols also present
    in `universe.symbols` (preserving the panel's own column order); it also
    supplies the default hash `name` (`universe.name`) when `name` is not
    given explicitly. Neither test author's assumed interface agreed on
    whether this parameter should exist at all -- one passed a `Universe`
    object, the other passed a plain `name=` string with no universe -- so
    both are accepted.

    Raises ValueError if `source` is not one of the values this function can
    honestly produce. `source="index_membership"` always raises: this
    function can only ever compute an availability proxy from bar data, never
    real index membership, because this repository holds no point-in-time
    membership history for any index.
    """
    if source not in _ALLOWED_SOURCES:
        raise ValueError(
            f"source={source!r} is not a recognised PitEligibility source; "
            f"must be one of {_ALLOWED_SOURCES!r}"
        )
    if source == "index_membership":
        raise ValueError(
            "source='index_membership' was requested, but this repository has no "
            "point-in-time index-membership history for any index -- only "
            "'availability_proxy' (listed-and-liquid, from bar presence) is "
            "derivable from current data. Constructing an 'index_membership' "
            "result would silently claim a level of accuracy this dataset "
            "cannot support."
        )

    if universe is not None:
        member_set = set(universe.symbols)
        col_idx = [i for i, s in enumerate(panel.symbols) if s in member_set]
    else:
        col_idx = list(range(len(panel.symbols)))

    column_symbols = tuple(panel.symbols[i] for i in col_idx)
    n_symbols = len(column_symbols)

    close = panel.field("close")
    day_offsets = panel.day_offsets
    n_sessions = len(day_offsets) - 1

    # LISTED: count of PRIOR sessions (strictly before d) with >= 1 present bar.
    session_present = np.zeros((n_sessions, n_symbols), dtype=bool)
    for session_idx in range(n_sessions):
        start = int(day_offsets[session_idx])
        end = int(day_offsets[session_idx + 1])
        session_slice = close[start:end]
        # Index unconditionally. `col_idx` is never None (it defaults to every
        # column above), so a truthiness guard here would SKIP filtering exactly
        # when col_idx is EMPTY -- a universe with zero symbols in common with the
        # panel -- which is the one case that most needs it, and which then raises
        # a shape mismatch against the zero-width `session_present`. F12.
        session_slice = session_slice[:, col_idx]
        session_present[session_idx, :] = np.any(np.isfinite(session_slice), axis=0)

    prior_present_count = np.zeros((n_sessions, n_symbols), dtype=np.int64)
    if n_sessions > 1:
        prior_present_count[1:, :] = np.cumsum(session_present, axis=0)[:-1, :]
    listed_ok = prior_present_count >= min_history_sessions

    # LIQUID: trailing 20-calendar-session ADV, strictly prior. Reuses the shared
    # data/liquidity.py implementation -- never a second ADV computation here.
    adv_by_session = compute_prior_adv_by_session(panel)
    # Unconditional for the same reason as above (F12): an empty col_idx is falsy.
    adv_by_session = adv_by_session[:, col_idx]
    # NaN (no full trailing window yet, or trailing window entirely absent)
    # compares False against any finite threshold, which is the desired
    # "unknown liquidity is not eligible" behaviour, never a spurious pass.
    liquid_ok = adv_by_session >= min_adv_inr

    mask = listed_ok & liquid_ok

    symbols_sorted = tuple(sorted(column_symbols))
    effective_name = name if name is not None else (universe.name if universe is not None else "")

    universe_hash = _HashStr(
        _universe_hash_string(
            name=effective_name,
            source=source,
            min_history_sessions=min_history_sessions,
            min_adv_inr=min_adv_inr,
            symbols_sorted=symbols_sorted,
            column_symbols=column_symbols,
            mask=mask,
        )
    )

    return PitEligibility(
        mask=mask,
        symbols=symbols_sorted,
        column_symbols=column_symbols,
        dates=panel.dates,
        source=source,
        name=effective_name,
        min_history_sessions=int(min_history_sessions),
        min_adv_inr=float(min_adv_inr),
        universe_hash=universe_hash,
    )


def eligibility_mask_to_bars(panel: Panel, eligibility: PitEligibility) -> np.ndarray:
    """Broadcast `eligibility.mask` (n_sessions, n_symbols) to bar level
    (n_rows, n_symbols), aligned to `panel`'s own column order.

    For CLI/backtest wiring: eligibility is a per-session decision, but
    downstream consumers (tradable filters, portfolio construction) operate
    bar-by-bar. `eligibility.mask`'s columns follow `eligibility.column_symbols`
    (which may be a subset of `panel.symbols`, if a `universe` restricted the
    original `compute_eligibility` call) -- a symbol absent from that
    computation is broadcast as ineligible (False) for every bar, never
    silently dropped from the returned array's shape.
    """
    eligible_col_of: dict[str, int] = {
        sym: idx for idx, sym in enumerate(eligibility.column_symbols)
    }
    day_offsets = panel.day_offsets
    n_rows = panel.n_rows()
    n_symbols = len(panel.symbols)

    out = np.zeros((n_rows, n_symbols), dtype=bool)
    for col, sym in enumerate(panel.symbols):
        src_col = eligible_col_of.get(sym)
        if src_col is None:
            continue
        for session_idx in range(len(day_offsets) - 1):
            start = int(day_offsets[session_idx])
            end = int(day_offsets[session_idx + 1])
            out[start:end, col] = eligibility.mask[session_idx, src_col]
    return out
