"""Tests for registry trial return artifacts and matrix assembly."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from nifty_quant.backtest.metrics import pbo_cscv
from nifty_quant.research.registry import TrialMatrix, TrialRecord, TrialRegistry


def _business_day_seconds(start: str, periods: int) -> NDArray[Any]:
    dates = pd.date_range(start=start, periods=periods, freq="B", tz="UTC")
    return np.asarray(dates.asi8 // 1_000_000_000, dtype=np.int64)


def _create_trial(
    registry: TrialRegistry,
    tmp_path: Path,
    trial_id: str,
    timestamps: NDArray[Any],
    returns: NDArray[Any],
    registry_ts: str,
    *,
    frame: pd.DataFrame | None = None,
    write_returns: bool = True,
) -> None:
    trial_dir = tmp_path / trial_id
    trial_dir.mkdir(parents=True)
    (trial_dir / "config.yaml").write_text("", encoding="utf-8")

    if write_returns:
        if frame is None:
            frame = pd.DataFrame(
                {
                    "ts": timestamps,
                    "return": pd.Series(
                        np.asarray(returns, dtype=np.float32),
                        dtype=np.float32,
                    ),
                }
            )
        frame.to_parquet(trial_dir / "returns.parquet", index=False)
    else:
        (trial_dir / "metrics.json").write_text("{}", encoding="utf-8")

    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts=registry_ts,
            strategy="test",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=str(trial_dir),
            error=None,
        )
    )


def test_build_trial_matrix_with_fully_overlapping_trials(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 40)
    returns_a = np.random.default_rng(101).normal(0.0, 0.01, 40).astype(np.float32)
    returns_b = np.random.default_rng(202).normal(0.0, 0.01, 40).astype(np.float32)

    _create_trial(
        registry,
        tmp_path,
        "trial-a",
        timestamps,
        returns_a,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-b",
        timestamps,
        returns_b,
        "2024-01-02T00:00:00Z",
    )

    result = registry.build_trial_matrix()

    assert result.matrix.shape == (40, 2)
    assert result.trial_ids == ("trial-a", "trial-b")
    assert result.n_dropped == 0
    assert np.array_equal(result.period_index, timestamps)
    assert bool(np.all(np.diff(result.period_index) > 0))
    assert not bool(np.isnan(result.matrix).any())


def test_build_trial_matrix_uses_true_intersection_and_source_values(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    base_timestamps = _business_day_seconds("2024-01-01", 15)
    timestamps_a = base_timestamps[:10]
    timestamps_b = base_timestamps[3:13]
    timestamps_c = base_timestamps[5:15]
    returns_a = np.linspace(0.001, 0.010, 10, dtype=np.float32)
    returns_b = np.linspace(-0.010, -0.001, 10, dtype=np.float32)
    returns_c = np.linspace(0.020, 0.029, 10, dtype=np.float32)

    _create_trial(
        registry,
        tmp_path,
        "trial-a",
        timestamps_a,
        returns_a,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-b",
        timestamps_b,
        returns_b,
        "2024-01-02T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-c",
        timestamps_c,
        returns_c,
        "2024-01-03T00:00:00Z",
    )

    expected_set = (
        set(int(value) for value in timestamps_a)
        & set(int(value) for value in timestamps_b)
        & set(int(value) for value in timestamps_c)
    )
    expected_periods = np.asarray(sorted(expected_set), dtype=np.int64)
    source_maps = [
        {int(timestamp): float(value) for timestamp, value in zip(timestamps, returns)}
        for timestamps, returns in (
            (timestamps_a, returns_a),
            (timestamps_b, returns_b),
            (timestamps_c, returns_c),
        )
    ]
    expected_matrix = np.column_stack(
        [
            [source_map[int(timestamp)] for timestamp in expected_periods]
            for source_map in source_maps
        ]
    ).astype(np.float64)

    result = registry.build_trial_matrix()

    assert result.matrix.shape == (expected_periods.size, 3)
    assert np.array_equal(result.period_index, expected_periods)
    assert np.array_equal(result.matrix, expected_matrix)
    assert result.trial_ids == ("trial-a", "trial-b", "trial-c")
    assert result.n_dropped == 0


def test_build_trial_matrix_handles_zero_and_one_available_trial(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")

    empty_result = registry.build_trial_matrix()

    assert isinstance(empty_result, TrialMatrix)
    assert empty_result.matrix.shape == (0, 0)
    assert empty_result.n_dropped == 0
    assert "Too few usable trials for PBO" in empty_result.explain()

    timestamps = _business_day_seconds("2024-01-01", 8)
    returns = np.linspace(-0.01, 0.01, 8, dtype=np.float32)
    _create_trial(
        registry,
        tmp_path,
        "only-trial",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
    )

    one_result = registry.build_trial_matrix()

    assert isinstance(one_result, TrialMatrix)
    assert one_result.matrix.shape == (8, 1)
    assert one_result.trial_ids == ("only-trial",)
    assert one_result.n_dropped == 0
    assert "Too few usable trials for PBO" in one_result.explain()


def test_build_trial_matrix_with_zero_overlapping_periods(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    base_timestamps = _business_day_seconds("2024-01-01", 20)
    timestamps_a = base_timestamps[:10]
    timestamps_b = base_timestamps[10:]
    returns_a = np.zeros(10, dtype=np.float32)
    returns_b = np.ones(10, dtype=np.float32)

    _create_trial(
        registry,
        tmp_path,
        "trial-a",
        timestamps_a,
        returns_a,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-b",
        timestamps_b,
        returns_b,
        "2024-01-02T00:00:00Z",
    )

    result = registry.build_trial_matrix()

    assert result.matrix.shape == (0, 2)
    assert result.trial_ids == ("trial-a", "trial-b")
    assert result.n_dropped == 0
    assert "Zero overlapping periods remain" in result.explain()


def test_build_trial_matrix_drops_missing_returns_artifact(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 6)
    returns = np.arange(6, dtype=np.float32) / 100.0

    _create_trial(
        registry,
        tmp_path,
        "valid-trial",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "broken-trial",
        timestamps,
        returns,
        "2024-01-02T00:00:00Z",
        write_returns=False,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ("valid-trial",)
    assert result.matrix.shape == (6, 1)
    assert result.n_dropped == 1
    assert "broken-trial" in result.drop_reasons
    assert "missing returns.parquet" in result.drop_reasons["broken-trial"]


def test_build_trial_matrix_drops_missing_return_column(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 5)
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "unrelated": np.ones(5, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "bad-schema",
        timestamps,
        np.zeros(5, dtype=np.float32),
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "bad-schema" in result.drop_reasons
    assert "return" in result.drop_reasons["bad-schema"]


def test_build_trial_matrix_drops_non_finite_returns(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 4)
    returns = np.asarray([0.01, np.nan, -0.01, 0.02], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(returns, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "non-finite",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "non-finite" in result.drop_reasons
    assert "non-finite" in result.drop_reasons["non-finite"]


def test_build_trial_matrix_drops_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 3)
    duplicate_timestamps = np.asarray(
        [timestamps[0], timestamps[0], timestamps[1]],
        dtype=np.int64,
    )
    returns = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": duplicate_timestamps,
            "return": pd.Series(returns, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "duplicate-ts",
        duplicate_timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "duplicate-ts" in result.drop_reasons
    assert "duplicate" in result.drop_reasons["duplicate-ts"]


def test_build_trial_matrix_reports_unknown_explicit_trial_id(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 7)
    returns = np.linspace(-0.02, 0.02, 7, dtype=np.float32)
    _create_trial(
        registry,
        tmp_path,
        "known-trial",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
    )

    result = registry.build_trial_matrix(
        trial_ids=["known-trial", "never-recorded"],
    )

    assert result.trial_ids == ("known-trial",)
    assert result.matrix.shape == (7, 1)
    assert result.n_dropped == 1
    assert result.drop_reasons["never-recorded"] == "not found in registry"


def test_build_trial_matrix_rejects_non_inner_alignment(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")

    with pytest.raises(NotImplementedError):
        registry.build_trial_matrix(alignment="outer")  # type: ignore[arg-type]


def test_build_trial_matrix_is_deterministic(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 12)
    returns_a = np.random.default_rng(11).normal(0.0, 0.01, 12).astype(np.float32)
    returns_b = np.random.default_rng(22).normal(0.0, 0.01, 12).astype(np.float32)

    _create_trial(
        registry,
        tmp_path,
        "trial-a",
        timestamps,
        returns_a,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-b",
        timestamps,
        returns_b,
        "2024-01-02T00:00:00Z",
    )

    first = registry.build_trial_matrix()
    second = registry.build_trial_matrix()

    assert np.array_equal(first.matrix, second.matrix)
    assert np.array_equal(first.period_index, second.period_index)
    assert first.trial_ids == second.trial_ids


def test_pbo_cscv_accepts_built_trial_matrix(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 60)

    for trial_number in range(8):
        returns = np.random.default_rng(1000 + trial_number).normal(
            0.0,
            0.01,
            size=60,
        ).astype(np.float32)
        _create_trial(
            registry,
            tmp_path,
            f"trial-{trial_number}",
            timestamps,
            returns,
            f"2024-02-{trial_number + 1:02d}T00:00:00Z",
        )

    matrix = registry.build_trial_matrix()
    pbo = pbo_cscv(matrix.matrix, n_splits=4)

    assert matrix.matrix.shape == (60, 8)
    assert isinstance(pbo, float)
    assert math.isfinite(pbo)
    assert 0.0 <= pbo <= 1.0


def test_build_trial_matrix_upcasts_float32_returns_to_float64(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 6)
    returns = np.asarray(
        [0.001, -0.002, 0.003, -0.004, 0.005, -0.006],
        dtype=np.float32,
    )
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(returns, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "float32-trial",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert frame["return"].dtype == np.dtype(np.float32)
    assert result.matrix.dtype == np.dtype(np.float64)
    assert np.array_equal(result.matrix[:, 0], returns.astype(np.float64))


def test_trial_matrix_explain_lists_dropped_trials_and_reasons(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 6)
    returns = np.linspace(-0.01, 0.01, 6, dtype=np.float32)

    _create_trial(
        registry,
        tmp_path,
        "kept-trial",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "dropped-trial",
        timestamps,
        returns,
        "2024-01-02T00:00:00Z",
        write_returns=False,
    )

    result = registry.build_trial_matrix()
    explanation = result.explain()

    assert "Dropped trials:" in explanation
    assert "  - dropped-trial: missing returns.parquet" in explanation
    assert "kept-trial" not in result.drop_reasons


def test_trial_registry_all_filters_by_strategy(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")

    registry.record(
        TrialRecord(
            config_hash="matching-trial",
            ts="2024-01-01T00:00:00Z",
            strategy="only_this_one",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=1.0,
            sharpe_net=0.8,
            n_trades=10,
            turnover=0.2,
            breakeven_bps=3.0,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=None,
            error=None,
        )
    )
    registry.record(
        TrialRecord(
            config_hash="other-trial",
            ts="2024-01-02T00:00:00Z",
            strategy="other_strategy",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=0.5,
            sharpe_net=0.4,
            n_trades=8,
            turnover=0.1,
            breakeven_bps=4.0,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=None,
            error=None,
        )
    )

    matching = registry.all(strategy="only_this_one")

    assert len(matching) == 1
    assert matching[0].config_hash == "matching-trial"
    assert matching[0].strategy == "only_this_one"
    assert all(record.config_hash != "other-trial" for record in matching)


def test_build_trial_matrix_deduplicates_explicit_trial_ids(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 5)
    returns_a = np.linspace(0.001, 0.005, 5, dtype=np.float32)
    returns_b = np.linspace(-0.005, -0.001, 5, dtype=np.float32)

    _create_trial(
        registry,
        tmp_path,
        "trial-a",
        timestamps,
        returns_a,
        "2024-01-01T00:00:00Z",
    )
    _create_trial(
        registry,
        tmp_path,
        "trial-b",
        timestamps,
        returns_b,
        "2024-01-02T00:00:00Z",
    )

    result = registry.build_trial_matrix(
        trial_ids=["trial-a", "trial-a", "trial-b"],
    )

    assert result.trial_ids == ("trial-a", "trial-b")
    assert len(set(result.trial_ids)) == len(result.trial_ids)
    assert result.n_dropped == 0
    assert result.matrix.shape == (5, 2)


def test_build_trial_matrix_filters_candidates_by_strategy(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 5)

    for trial_id, strategy, registry_ts in (
        ("foo-trial", "foo", "2024-01-01T00:00:00Z"),
        ("bar-trial", "bar", "2024-01-02T00:00:00Z"),
    ):
        trial_dir = tmp_path / trial_id
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.yaml").write_text("", encoding="utf-8")
        frame = pd.DataFrame(
            {
                "ts": timestamps,
                "return": pd.Series(
                    np.linspace(-0.01, 0.01, 5, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
        frame.to_parquet(trial_dir / "returns.parquet", index=False)
        registry.record(
            TrialRecord(
                config_hash=trial_id,
                ts=registry_ts,
                strategy=strategy,
                params_json="{}",
                split_id="full",
                purpose="exploration",
                sharpe_gross=None,
                sharpe_net=None,
                n_trades=None,
                turnover=None,
                breakeven_bps=None,
                git_sha=None,
                data_fingerprint=None,
                code_version=None,
                wall_s=None,
                result_path=str(trial_dir),
                error=None,
            )
        )

    result = registry.build_trial_matrix(strategy="foo")

    assert result.trial_ids == ("foo-trial",)
    assert "bar-trial" not in result.trial_ids
    assert "bar-trial" not in result.drop_reasons
    assert result.n_dropped == 0


def test_build_trial_matrix_registry_query_dedups_same_config_hash(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    trial_id = "dup-full-rows"
    timestamps = _business_day_seconds("2024-01-01", 5)

    first_dir = tmp_path / "dup-full-rows-first"
    first_dir.mkdir(parents=True)
    (first_dir / "config.yaml").write_text("", encoding="utf-8")
    frame_first = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(
                np.linspace(0.001, 0.005, 5, dtype=np.float32),
                dtype=np.float32,
            ),
        }
    )
    frame_first.to_parquet(first_dir / "returns.parquet", index=False)

    second_dir = tmp_path / "dup-full-rows-second"
    second_dir.mkdir(parents=True)
    (second_dir / "config.yaml").write_text("", encoding="utf-8")
    frame_second = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(
                np.linspace(-0.005, -0.001, 5, dtype=np.float32),
                dtype=np.float32,
            ),
        }
    )
    frame_second.to_parquet(second_dir / "returns.parquet", index=False)

    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts="2024-01-01T00:00:00Z",
            strategy="test",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=str(first_dir),
            error=None,
        )
    )
    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts="2024-01-02T00:00:00Z",
            strategy="test",
            params_json="{}",
            split_id="full-2",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=str(second_dir),
            error=None,
        )
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == (trial_id,)
    assert result.n_dropped == 0
    # D1 (specs/pbo_dsr_wiring.md): on a config_hash collision the NEWEST row with an
    # artifact wins, not the oldest. This test previously asserted `frame_first` -- i.e.
    # it DEFENDED the defect, which is why the defect survived: a sweep trial colliding
    # with an earlier walk-forward run silently loaded that run's short test slice,
    # collapsing T and making pbo_cscv raise while explain() reported "dropped 0".
    assert np.array_equal(result.matrix[:, 0], frame_second["return"].to_numpy(dtype=np.float64))


def test_build_trial_matrix_registry_query_uses_non_null_duplicate_row(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    trial_id = "walk-forward-trial"
    timestamps = _business_day_seconds("2024-01-01", 5)
    trial_dir = tmp_path / trial_id
    trial_dir.mkdir(parents=True)
    (trial_dir / "config.yaml").write_text("", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(
                np.linspace(-0.01, 0.01, 5, dtype=np.float32),
                dtype=np.float32,
            ),
        }
    )
    frame.to_parquet(trial_dir / "returns.parquet", index=False)

    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts="2024-01-01T00:00:00Z",
            strategy="test",
            params_json="{}",
            split_id="split-0",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=None,
            error=None,
        )
    )
    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts="2024-01-02T00:00:00Z",
            strategy="test",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=str(trial_dir),
            error=None,
        )
    )

    result = registry.build_trial_matrix()
    explicit_result = registry.build_trial_matrix(trial_ids=[trial_id])

    assert result.trial_ids == (trial_id,)
    assert result.n_dropped == 0
    # D1 (specs/pbo_dsr_wiring.md): the explicit-trial_ids path now prefers the newest row
    # that HAS an artifact, so both query paths agree. Previously the explicit path took
    # `ORDER BY ts LIMIT 1` -- the OLDEST row, here the one with no result_path -- and
    # dropped the trial, while the unfiltered path found it. **The same trial resolving
    # differently depending on which query you used WAS the defect**, and this test
    # asserted that asymmetry as correct.
    assert explicit_result.trial_ids == (trial_id,)
    assert trial_id not in explicit_result.drop_reasons


def test_build_trial_matrix_drops_explicit_trial_without_result_path(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    trial_id = "no-result-path"

    registry.record(
        TrialRecord(
            config_hash=trial_id,
            ts="2024-01-01T00:00:00Z",
            strategy="test",
            params_json="{}",
            split_id="full",
            purpose="exploration",
            sharpe_gross=None,
            sharpe_net=None,
            n_trades=None,
            turnover=None,
            breakeven_bps=None,
            git_sha=None,
            data_fingerprint=None,
            code_version=None,
            wall_s=None,
            result_path=None,
            error=None,
        )
    )

    result = registry.build_trial_matrix(trial_ids=[trial_id])

    assert result.trial_ids == ()
    assert result.matrix.shape == (0, 0)
    assert result.n_dropped == 1
    assert result.drop_reasons[trial_id] == "no result_path recorded for this trial"


def test_build_trial_matrix_drops_unreadable_returns_artifact(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 4)
    returns = np.linspace(-0.01, 0.01, 4, dtype=np.float32)

    _create_trial(
        registry,
        tmp_path,
        "corrupt-parquet",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
    )
    (tmp_path / "corrupt-parquet" / "returns.parquet").write_bytes(
        b"not a parquet file",
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "unreadable returns.parquet" in result.drop_reasons["corrupt-parquet"]


def test_build_trial_matrix_drops_returns_with_extra_column(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 4)
    returns = np.linspace(-0.01, 0.01, 4, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(returns, dtype=np.float32),
            "extra": pd.Series(np.ones(4, dtype=np.float32), dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "extra-column",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "expected exactly two columns" in result.drop_reasons["extra-column"]


def test_build_trial_matrix_drops_empty_returns_series(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = np.empty(0, dtype=np.int64)
    returns = np.empty(0, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": pd.Series([], dtype=np.int64),
            "return": pd.Series([], dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "empty-returns",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "empty returns series" in result.drop_reasons["empty-returns"]


def test_build_trial_matrix_drops_non_integer_timestamp_column(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 4)
    returns = np.linspace(-0.01, 0.01, 4, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": timestamps.astype(np.float64),
            "return": pd.Series(returns, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "float-timestamps",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "non-integer dtype" in result.drop_reasons["float-timestamps"]


def test_build_trial_matrix_drops_missing_nullable_timestamps(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 3)
    returns = np.asarray([0.01, -0.01, 0.02], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": pd.Series(
                [int(timestamps[0]), pd.NA, int(timestamps[2])],
                dtype="Int64",
            ),
            "return": pd.Series(returns, dtype=np.float32),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "missing-timestamps",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "missing values" in result.drop_reasons["missing-timestamps"]


def test_build_trial_matrix_drops_non_numeric_return_values(
    tmp_path: Path,
) -> None:
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    timestamps = _business_day_seconds("2024-01-01", 3)
    returns = np.asarray([0.01, 0.0, 0.02], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "return": pd.Series(
                ["0.01", "not-a-number", "0.02"],
                dtype=object,
            ),
        }
    )
    _create_trial(
        registry,
        tmp_path,
        "non-numeric-returns",
        timestamps,
        returns,
        "2024-01-01T00:00:00Z",
        frame=frame,
    )

    result = registry.build_trial_matrix()

    assert result.trial_ids == ()
    assert result.n_dropped == 1
    assert "could not be converted to float64" in result.drop_reasons[
        "non-numeric-returns"
    ]
