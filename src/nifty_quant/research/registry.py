"""Research experiment registry."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


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
    ruined: bool | None = None
    ruin_index: int | None = None


@dataclass(frozen=True)
class TrialMatrix:
    """Period-aligned daily return matrix assembled from trial artifacts."""

    matrix: np.ndarray
    trial_ids: tuple[str, ...]
    period_index: np.ndarray
    n_dropped: int
    drop_reasons: dict[str, str]
    alignment: Literal["inner"]

    def explain(self) -> str:
        """Return a human-readable summary of the assembled trial matrix."""
        n_kept = len(self.trial_ids)
        n_candidates = n_kept + self.n_dropped
        lines = [
            f"Alignment policy: {self.alignment} intersection of common periods.",
            (
                f"Found {n_candidates} candidate trials; kept {n_kept}; "
                f"dropped {self.n_dropped}."
            ),
        ]

        if self.drop_reasons:
            lines.append("Dropped trials:")
            for trial_id, reason in self.drop_reasons.items():
                lines.append(f"  - {trial_id}: {reason}")
        else:
            lines.append("No trials were dropped.")

        lines.append(f"Periods surviving alignment: {self.matrix.shape[0]}.")
        if self.matrix.shape[1] < 2:
            lines.append(
                "Too few usable trials for PBO/effective_n_trials: "
                "at least two trial columns are required."
            )
        if self.matrix.shape[0] == 0:
            lines.append("Zero overlapping periods remain across the kept trials.")

        return "\n".join(lines)


_TRIAL_FIELDS = tuple(field.name for field in fields(TrialRecord))


def _row_to_record(row: sqlite3.Row) -> TrialRecord:
    """Build a `TrialRecord` from a raw `trials` row, coercing `ruined` to bool.

    SQLite has no native bool type: `sqlite3` stores a Python `bool` as integer 0/1
    on insert but reads it back as a plain `int`, not `bool`. Consumers need
    `record.ruined is True`/`is False` to hold, so this explicitly casts the stored
    integer back to `bool` (preserving `None` for NULL/unrecorded).
    """
    values = {name: row[name] for name in _TRIAL_FIELDS}
    ruined = values["ruined"]
    values["ruined"] = None if ruined is None else bool(ruined)
    return TrialRecord(**values)


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
                ruined INTEGER,
                ruin_index INTEGER,
                UNIQUE(config_hash, split_id)
            )
            """
        )
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add any columns missing from a pre-existing `trials` table on disk.

        `CREATE TABLE IF NOT EXISTS` above only handles a brand-new file; an existing
        `trials.db` written before `ruined`/`ruin_index` existed keeps its old schema,
        which would otherwise make `record()`'s INSERT fail with `no column named
        ruined`. Idempotent: safe to call on every construction.
        """
        existing_columns: set[str] = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(trials)").fetchall()
        }

        for column_name in ("ruined", "ruin_index"):
            if column_name not in existing_columns:
                self._conn.execute(
                    f"ALTER TABLE trials ADD COLUMN {column_name} INTEGER"
                )

        self._conn.commit()

    def record(self, rec: TrialRecord) -> None:
        """Append-only insert; duplicate (config_hash, split_id) rows are ignored."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO trials (
                config_hash, ts, strategy, params_json, split_id, purpose,
                sharpe_gross, sharpe_net, n_trades, turnover, breakeven_bps,
                git_sha, data_fingerprint, code_version, wall_s, result_path, error,
                ruined, ruin_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

        return [_row_to_record(row) for row in rows]

    def var_trial_sharpes(self, *, strategy: str | None = None) -> float:
        """Variance of realized trial Sharpes - feeds expected_max_sharpe estimates.

        Must come from actual runs, never a textbook constant.
        """
        values = self.trial_sharpes(strategy=strategy, purpose="exploration")
        if values.size < 2:
            return 0.0
        return float(np.var(values, ddof=0))

    def build_trial_matrix(
        self,
        *,
        trial_ids: Sequence[str] | None = None,
        strategy: str | None = None,
        purpose: str = "exploration",
        alignment: Literal["inner"] = "inner",
    ) -> TrialMatrix:
        """Build a daily return matrix aligned on the intersection of timestamps.

        Reads each candidate trial's per-period return series from
        ``<result_path>/returns.parquet`` (two columns: ``ts`` -- int64 epoch-seconds
        UTC, one row per trading day, matching the daily-compounded basis produced by
        ``cli._daily_returns``; and ``return`` -- the daily compounded net return,
        upcast to float64 regardless of on-disk dtype).

        Only ``"inner"`` alignment is implemented: the matrix's period axis is the
        intersection of the ``ts`` values present in every *kept* trial. An outer join
        with NaN-fill was rejected because ``pbo_cscv`` and ``effective_n_trials``
        cannot accept NaNs, while silently imputing values such as zero would
        fabricate performance for periods in which a trial did not trade. Per-split
        pooling, which concatenates each trial's own periods without aligning them,
        was rejected because CSCV's correlation and rank structure requires every
        trial column to be observed over the same period axis; pooling would silently
        line up unrelated calendar periods across trials. Passing any other value
        raises ``NotImplementedError`` rather than silently falling back to inner.

        Missing or invalid per-trial artifacts (no result_path, missing
        ``returns.parquet``, bad schema, non-finite values, duplicate timestamps, an
        empty series, or a ``trial_ids`` entry not present in the registry at all) are
        recorded in ``TrialMatrix.drop_reasons`` rather than raising -- callers must
        consult ``.explain()`` or ``drop_reasons``/``n_dropped`` to know whether the
        assembled matrix is trustworthy. Degenerate cases (fewer than two kept
        trials, or zero overlapping periods) still return a valid ``TrialMatrix``
        rather than a bare ``nan``; ``pbo_cscv``/``effective_n_trials`` themselves
        raise ``ValueError`` on an unusable matrix, which is preferable to silently
        returning ``nan``.
        """
        rationale = (
            "Only inner alignment is implemented. An outer join with NaN-fill was "
            "rejected because pbo_cscv/effective_n_trials cannot accept NaNs, and "
            "zero-fill would fabricate performance for periods a trial did not trade. "
            "Per-split pooling was rejected because CSCV correlation/rank structure "
            "requires all trial columns on the same period axis; pooling would "
            "silently align unrelated calendar periods across trials."
        )
        if alignment != "inner":
            raise NotImplementedError(rationale)

        candidate_rows: list[tuple[str, sqlite3.Row | None]] = []

        if trial_ids is not None:
            seen_ids: set[str] = set()
            for config_hash in trial_ids:
                if config_hash in seen_ids:
                    continue
                seen_ids.add(config_hash)
                row = self._conn.execute(
                    """
                    SELECT * FROM trials
                    WHERE config_hash = ?
                    ORDER BY ts, rowid
                    LIMIT 1
                    """,
                    (config_hash,),
                ).fetchone()
                candidate_rows.append((config_hash, row))
        else:
            query = (
                "SELECT * FROM trials "
                "WHERE purpose = ? AND result_path IS NOT NULL"
            )
            params: list[str] = [purpose]
            if strategy is not None:
                query += " AND strategy = ?"
                params.append(strategy)
            query += " ORDER BY ts, rowid"

            seen_ids = set()
            rows = self._conn.execute(query, params).fetchall()
            for row in rows:
                config_hash = str(row["config_hash"])
                if config_hash in seen_ids:
                    continue
                seen_ids.add(config_hash)
                candidate_rows.append((config_hash, row))

        drop_reasons: dict[str, str] = {}
        kept_ids: list[str] = []
        loaded: list[tuple[np.ndarray, np.ndarray]] = []

        for config_hash, row in candidate_rows:
            if row is None:
                drop_reasons[config_hash] = "not found in registry"
                continue

            result_path = row["result_path"]
            if result_path is None:
                drop_reasons[config_hash] = "no result_path recorded for this trial"
                continue

            returns_path = Path(str(result_path)) / "returns.parquet"
            if not returns_path.is_file():
                drop_reasons[config_hash] = (
                    "missing returns.parquet (incomplete trial artifact)"
                )
                continue

            try:
                frame = pd.read_parquet(returns_path)
            except FileNotFoundError:
                drop_reasons[config_hash] = (
                    "missing returns.parquet (incomplete trial artifact)"
                )
                continue
            except Exception as exc:
                drop_reasons[config_hash] = (
                    f"unreadable returns.parquet ({type(exc).__name__})"
                )
                continue

            missing_columns = [
                name for name in ("ts", "return") if name not in frame.columns
            ]
            if missing_columns:
                drop_reasons[config_hash] = (
                    f"bad schema: missing column(s) {', '.join(missing_columns)}"
                )
                continue

            if len(frame.columns) != 2:
                column_names = ", ".join(str(name) for name in frame.columns)
                drop_reasons[config_hash] = (
                    "bad schema: expected exactly two columns ('ts', 'return'); "
                    f"found {column_names}"
                )
                continue

            if frame.empty:
                drop_reasons[config_hash] = "empty returns series"
                continue

            ts_dtype = frame["ts"].dtype
            if not bool(pd.api.types.is_integer_dtype(ts_dtype)):
                drop_reasons[config_hash] = (
                    f"ts column has non-integer dtype {ts_dtype}"
                )
                continue

            if bool(frame["ts"].isna().any()):
                drop_reasons[config_hash] = "ts column contains missing values"
                continue

            if bool(frame["ts"].duplicated().any()):
                drop_reasons[config_hash] = "ts column contains duplicate timestamps"
                continue

            try:
                ts_values = frame["ts"].to_numpy(dtype=np.int64, copy=True)
            except Exception as exc:
                drop_reasons[config_hash] = (
                    f"ts column could not be converted to int64 ({type(exc).__name__})"
                )
                continue

            try:
                return_values = frame["return"].to_numpy(dtype=np.float64, copy=True)
            except Exception as exc:
                drop_reasons[config_hash] = (
                    "return column could not be converted to float64 "
                    f"({type(exc).__name__})"
                )
                continue

            if not bool(np.isfinite(return_values).all()):
                drop_reasons[config_hash] = (
                    "return column contains non-finite values"
                )
                continue

            order = np.argsort(ts_values, kind="mergesort")
            sorted_ts = np.array(ts_values[order], dtype=np.int64, copy=True)
            sorted_returns = np.array(
                return_values[order],
                dtype=np.float64,
                copy=True,
            )
            kept_ids.append(config_hash)
            loaded.append((sorted_ts, sorted_returns))

        if loaded:
            common_ts = np.array(loaded[0][0], dtype=np.int64, copy=True)
            for ts_values, _ in loaded[1:]:
                common_ts = np.intersect1d(
                    common_ts,
                    ts_values,
                    assume_unique=True,
                )
            period_index = np.array(common_ts, dtype=np.int64, copy=True)
            matrix = np.empty(
                (period_index.size, len(loaded)),
                dtype=np.float64,
            )
            for column, (ts_values, return_values) in enumerate(loaded):
                positions = np.searchsorted(ts_values, period_index)
                matrix[:, column] = return_values[positions]
        else:
            period_index = np.empty(0, dtype=np.int64)
            matrix = np.empty((0, 0), dtype=np.float64)

        return TrialMatrix(
            matrix=matrix,
            trial_ids=tuple(kept_ids),
            period_index=period_index,
            n_dropped=len(drop_reasons),
            drop_reasons=drop_reasons,
            alignment="inner",
        )
