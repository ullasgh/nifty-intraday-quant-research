"""Regression test for `TrialRegistry` migrating a pre-existing `trials.db`.

Found on a real end-to-end run against this repo's own `results/trials.db` (written before
the `ruined`/`ruin_index` columns existed): `CREATE TABLE IF NOT EXISTS` does not alter an
existing table's schema, so `TrialRegistry.record(...)` failed with
`sqlite3.OperationalError: table trials has no column named ruined` the moment a caller
tried to record a `TrialRecord` with those fields against an old-schema database file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nifty_quant.research.registry import TrialRecord, TrialRegistry


def _write_pre_migration_db(path: Path) -> None:
    """Create a `trials` table matching the schema before `ruined`/`ruin_index` existed."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE trials (
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
    conn.commit()
    conn.close()


def _make_record(config_hash: str) -> TrialRecord:
    return TrialRecord(
        config_hash=config_hash,
        ts="2024-01-01T00:00:00",
        strategy="volume_breakout",
        params_json="{}",
        split_id="full",
        purpose="exploration",
        sharpe_gross=1.0,
        sharpe_net=0.5,
        n_trades=5,
        turnover=0.3,
        breakeven_bps=10.0,
        git_sha=None,
        data_fingerprint=None,
        code_version="test",
        wall_s=1.0,
        result_path=None,
        ruined=True,
        ruin_index=7,
        error=None,
    )


def test_trial_registry_migrates_pre_existing_db_missing_ruined_columns(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trials.db"
    _write_pre_migration_db(db_path)

    # Pre-condition: the old schema genuinely lacks the new columns.
    conn = sqlite3.connect(str(db_path))
    old_columns = {row[1] for row in conn.execute("PRAGMA table_info(trials)").fetchall()}
    conn.close()
    assert "ruined" not in old_columns
    assert "ruin_index" not in old_columns

    registry = TrialRegistry(db_path)
    registry.record(_make_record("abc123"))

    records = registry.all(strategy="volume_breakout")
    assert len(records) == 1
    assert records[0].ruined is True
    assert records[0].ruin_index == 7


def test_trial_registry_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    _write_pre_migration_db(db_path)

    TrialRegistry(db_path)
    # Constructing it again against the now-migrated file must not raise (e.g. from a
    # duplicate `ALTER TABLE ADD COLUMN`).
    registry = TrialRegistry(db_path)
    registry.record(_make_record("def456"))
    assert len(registry.all()) == 1
