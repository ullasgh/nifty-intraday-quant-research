"""Research experiment registry."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    config_hash: str
    ts: str
    strategy: str
    params_json: str
    split_id: str
    purpose: Literal["exploration", "confirmation"]
    sharpe_gross: float | None
    sharpe_net: float | None
    n_trades: int | None
    turnover: float | None
    breakeven_bps: float | None
    git_sha: str | None
    data_fingerprint: str | None
    code_version: str | None
    wall_s: float | None
    result_path: str | None
    error: str | None


_TRIAL_FIELDS = tuple(field.name for field in fields(TrialRecord))


class TrialRegistry:
    """SQLite-backed registry for research trial records."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                config_hash TEXT NOT NULL,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                params_json TEXT NOT NULL,
                split_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                sharpe_gross REAL,
                sharpe_net REAL,
                n_trades INTEGER,
                turnover REAL,
                breakeven_bps REAL,
                git_sha TEXT,
                data_fingerprint TEXT,
                code_version TEXT,
                wall_s REAL,
                result_path TEXT,
                error TEXT,
                UNIQUE(config_hash, split_id)
            )
            """
        )
        self._conn.commit()

    def record(self, rec: TrialRecord) -> None:
        """Append-only insert; duplicate (config_hash, split_id) rows are ignored."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO trials (
                config_hash, ts, strategy, params_json, split_id, purpose,
                sharpe_gross, sharpe_net, n_trades, turnover, breakeven_bps,
                git_sha, data_fingerprint, code_version, wall_s, result_path, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(getattr(rec, name) for name in _TRIAL_FIELDS),
        )
        self._conn.commit()

    def n_trials(
        self, *, strategy: str | None = None, purpose: str = "exploration"
    ) -> int:
        query = "SELECT COUNT(*) FROM trials WHERE purpose = ?"
        params: list[str] = [purpose]

        if strategy is not None:
            query += " AND strategy = ?"
            params.append(strategy)

        row = self._conn.execute(query, params).fetchone()
        return int(row[0]) if row is not None else 0

    def trial_sharpes(
        self, *, strategy: str | None = None, purpose: str = "exploration"
    ) -> np.ndarray:
        query = "SELECT sharpe_net FROM trials WHERE sharpe_net IS NOT NULL AND purpose = ?"
        params: list[str] = [purpose]

        if strategy is not None:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY ts"
        rows = self._conn.execute(query, params).fetchall()
        return np.array([row[0] for row in rows], dtype=float)

    def all(self, *, strategy: str | None = None) -> list[TrialRecord]:
        if strategy is None:
            rows = self._conn.execute("SELECT * FROM trials ORDER BY ts").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trials WHERE strategy = ? ORDER BY ts",
                (strategy,),
            ).fetchall()

        return [
            TrialRecord(**{name: row[name] for name in _TRIAL_FIELDS})
            for row in rows
        ]

    def var_trial_sharpes(self, *, strategy: str | None = None) -> float:
        """Variance of realized trial Sharpes - feeds expected_max_sharpe estimates.

        Must come from actual runs, never a textbook constant.
        """
        values = self.trial_sharpes(strategy=strategy, purpose="exploration")
        if values.size < 2:
            return 0.0
        return float(np.var(values, ddof=0))
