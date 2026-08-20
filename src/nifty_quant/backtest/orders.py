"""Order lifecycle primitives shared by the backtest engine.

Pinned location (spec `order_lifecycle.md`, AMENDMENT 2 item 2):
``OrderIntent`` and ``PendingOrder`` live here, not in ``backtest/engine.py``.

Only ``ENTRY`` and ``EOD_EXIT`` ever populate ``PendingOrder.intent`` -- they are the
only two intents that pass through the netted pending-order queue and fill one bar
after being queued. ``RISK_EXIT`` and ``FORCED_EOD`` fill immediately, in the same
row they are decided, via a direct fill; they are blotter-only tags describing a
fill that already happened, never an order waiting in the queue. ``REBALANCE`` is
reserved for Phase G and is not emitted by the current engine; if it ever is, it
must behave exactly like ``ENTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OrderIntent(str, Enum):
    """Why an order was placed. Stored in the trade blotter as ``.value`` (a plain
    ``str``), never as the enum member itself -- see AMENDMENT 2 item 2. Subclassing
    ``str`` keeps `blotter_value == OrderIntent.X` comparisons and
    `blotter_value in set(OrderIntent)` membership checks working uniformly whether
    the caller holds the plain string or the enum member.
    """

    ENTRY = "entry"
    REBALANCE = "rebalance"
    RISK_EXIT = "risk_exit"
    EOD_EXIT = "eod_exit"
    FORCED_EOD = "forced_eod"


@dataclass(frozen=True)
class PendingOrder:
    """One order waiting in the netted queue for its fill row.

    ``intent`` is restricted in practice to ``OrderIntent.ENTRY`` or
    ``OrderIntent.EOD_EXIT`` -- see module docstring.

    ``order_id`` (spec ``tca_record.md`` AMENDMENT 1 item 2): a monotonically
    increasing int64 assigned at order creation (when this ``PendingOrder`` is
    built), stable across every fill row the order produces.
    """

    intent: OrderIntent
    shares: np.ndarray  # float64, length n_symbols, signed
    queued_row: int
    order_id: int
