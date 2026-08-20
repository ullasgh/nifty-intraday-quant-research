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
    # ---- specs/run_provenance.md: fields added, never renaming/removing the above ----
    seed: int | None = None
    universe_name: str = ""
    universe_hash: str = ""
    panel_hash: str = ""
    start: str = ""
    end: str = ""
    cost_model_id: str = ""
    slippage_model_id: str = ""
    fill_model_id: str = ""
    embargo_components: str = "{}"
    parent_trial_id: str | None = None
    feature_version: str = ""


@dataclass(frozen=True)
class TrialMatrix:
    """Period-aligned daily return matrix assembled from trial artifacts."""

    matrix: np.ndarray
    trial_ids: tuple[str, ...]
    period_index: np.ndarray
    n_dropped: int
    drop_reasons: dict[str, str]
    alignment: Literal["inner"]
    # specs/pbo_dsr_wiring.md D1: which row (artifact path, ts, split_id) each KEPT
    # trial actually resolved to. A selection that can pick the wrong row on a
    # config_hash collision must say which row it picked -- keyed by trial_id
    # (config_hash), same order as `trial_ids`.
    resolved_artifacts: dict[str, tuple[str, str, str]]

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

        if self.trial_ids:
            lines.append("Resolved artifacts (kept trials):")
            for trial_id in self.trial_ids:
                resolved = self.resolved_artifacts.get(trial_id)
                if resolved is None:  # pragma: no cover - defensive, always populated
                    continue
                result_path, ts, split_id = resolved
                lines.append(
                    f"  - {trial_id}: path={result_path} ts={ts} split_id={split_id}"
                )

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
                seed INTEGER,
                universe_name TEXT,
                universe_hash TEXT,
                panel_hash TEXT,
                "start" TEXT,
                "end" TEXT,
                cost_model_id TEXT,
                slippage_model_id TEXT,
                fill_model_id TEXT,
                embargo_components TEXT,
                parent_trial_id TEXT,
                feature_version TEXT,
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

        # specs/run_provenance.md section A: add any of the twelve new columns
        # missing from a pre-existing trials.db, same idempotent pattern as above.
        _text_columns = (
            "universe_name",
            "universe_hash",
            "panel_hash",
            "start",
            "end",
            "cost_model_id",
            "slippage_model_id",
            "fill_model_id",
            "embargo_components",
            "parent_trial_id",
            "feature_version",
        )
        for column_name in _text_columns:
            if column_name not in existing_columns:
                self._conn.execute(
                    f'ALTER TABLE trials ADD COLUMN "{column_name}" TEXT'
                )
        if "seed" not in existing_columns:
            self._conn.execute("ALTER TABLE trials ADD COLUMN seed INTEGER")

        self._conn.commit()

    def record(self, rec: TrialRecord) -> None:
        """Append-only insert; duplicate (config_hash, split_id) rows are ignored."""
        columns = ", ".join(f'"{name}"' for name in _TRIAL_FIELDS)
        placeholders = ", ".join("?" for _ in _TRIAL_FIELDS)
        self._conn.execute(
            f"INSERT OR IGNORE INTO trials ({columns}) VALUES ({placeholders})",
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
                # specs/pbo_dsr_wiring.md D1: prefer the NEWEST row that actually has
                # an artifact -- the OLDEST row (previously `ORDER BY ts, rowid LIMIT
                # 1`) could be a stale walk-forward test slice sharing a config_hash
                # with a later, much longer sweep result. Fall back to the newest row
                # regardless of result_path only when NO row for this hash has an
                # artifact at all, so the drop-reason path below still reports "no
                # result_path recorded" instead of a misleading "not found".
                row = self._conn.execute(
                    """
                    SELECT * FROM trials
                    WHERE config_hash = ? AND result_path IS NOT NULL
                    ORDER BY ts DESC, rowid DESC
                    LIMIT 1
                    """,
                    (config_hash,),
                ).fetchone()
                if row is None:
                    row = self._conn.execute(
                        """
                        SELECT * FROM trials
                        WHERE config_hash = ?
                        ORDER BY ts DESC, rowid DESC
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
            # Ascending: the OUTPUT order of distinct trials (trial_ids/matrix
            # columns) is each config_hash's first-appearance order, oldest first
            # -- unrelated to, and preserved across, the D1 collision fix below.
            query += " ORDER BY ts ASC, rowid ASC"

            rows = self._conn.execute(query, params).fetchall()
            best_row_by_hash: dict[str, sqlite3.Row] = {}
            first_seen_order: list[str] = []
            for row in rows:
                config_hash = str(row["config_hash"])
                if config_hash not in best_row_by_hash:
                    first_seen_order.append(config_hash)
                # D1 collision fix: rows arrive in ascending (ts, rowid) order and
                # the WHERE clause already restricts to rows with an artifact, so
                # each later write for the same config_hash overwrites the
                # previous one here -- the row a hash ends up mapped to is always
                # the NEWEST one that has an artifact, even though the output
                # order above stays oldest-first.
                best_row_by_hash[config_hash] = row
            for config_hash in first_seen_order:
                candidate_rows.append((config_hash, best_row_by_hash[config_hash]))

        drop_reasons: dict[str, str] = {}
        kept_ids: list[str] = []
        loaded: list[tuple[np.ndarray, np.ndarray]] = []
        resolved_artifacts: dict[str, tuple[str, str, str]] = {}

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
            resolved_artifacts[config_hash] = (
                str(result_path),
                str(row["ts"]),
                str(row["split_id"]),
            )

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
            resolved_artifacts=resolved_artifacts,
        )
