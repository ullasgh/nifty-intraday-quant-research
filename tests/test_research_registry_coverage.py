from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nifty_quant.research.registry as registry_module
from nifty_quant.research.registry import TrialRecord, TrialRegistry


def _make_record(config_hash: str, *, result_path: str | None = None) -> TrialRecord:
    return TrialRecord(
        config_hash=config_hash,
        ts="2024-01-01T00:00:00+00:00",
        strategy="test_strategy",
        params_json="{}",
        split_id="test_split",
        purpose="exploration",
        sharpe_gross=1.0,
        sharpe_net=1.0,
        n_trades=1,
        turnover=0.1,
        breakeven_bps=1.0,
        git_sha="abc123",
        data_fingerprint="fingerprint",
        code_version="1.0.0",
        wall_s=1.0,
        result_path=result_path,
        error=None,
        ruined=None,
        ruin_index=None,
    )


class _FakeBoolResult:
    """Mimics the `.any()` surface of the boolean Series returned by `.isna()`/`.duplicated()`."""

    def __init__(self, value: bool) -> None:
        self._value = value

    def any(self) -> bool:
        return self._value


class _FakeColumn:
    """Mimics only the `pandas.Series` API surface `build_trial_matrix` calls on `frame[col]`."""

    def __init__(
        self,
        dtype: np.dtype,
        values: list[object],
        *,
        raise_on_to_numpy: Exception | None = None,
    ) -> None:
        self.dtype = dtype
        self._values = values
        self._raise_on_to_numpy = raise_on_to_numpy

    def isna(self) -> _FakeBoolResult:
        return _FakeBoolResult(any(v is None for v in self._values))

    def duplicated(self) -> _FakeBoolResult:
        seen: set[object] = set()
        has_dup = False
        for v in self._values:
            if v in seen:
                has_dup = True
            seen.add(v)
        return _FakeBoolResult(has_dup)

    def to_numpy(self, dtype: object = None, copy: bool = True) -> np.ndarray:
        if self._raise_on_to_numpy is not None:
            raise self._raise_on_to_numpy
        return np.array(self._values, dtype=dtype)


class _FakeFrame:
    """Mimics only the `pandas.DataFrame` API surface `build_trial_matrix` calls."""

    def __init__(self, columns: dict[str, _FakeColumn]) -> None:
        self._columns = columns

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def empty(self) -> bool:
        first = next(iter(self._columns.values()))
        return len(first._values) == 0

    def __getitem__(self, key: str) -> _FakeColumn:
        return self._columns[key]


def test_build_trial_matrix_drops_on_read_parquet_race_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Targets registry.py:325-336 FileNotFoundError branch, reachable only by racing
    TOCTOU after is_file() passed."""
    db_path = tmp_path / "trials.db"
    registry = TrialRegistry(db_path)

    result_dir = tmp_path / "trial_race"
    result_dir.mkdir()
    (result_dir / "returns.parquet").touch()

    registry.record(_make_record("race1", result_path=str(result_dir)))

    def _raise_file_not_found(path: object, *args: object, **kwargs: object) -> None:
        raise FileNotFoundError("synthetic TOCTOU race")

    monkeypatch.setattr(registry_module.pd, "read_parquet", _raise_file_not_found)

    trial_matrix = registry.build_trial_matrix(trial_ids=["race1"])

    assert trial_matrix.n_dropped == 1
    assert trial_matrix.drop_reasons == {
        "race1": "missing returns.parquet (incomplete trial artifact)"
    }
    assert trial_matrix.trial_ids == ()


def test_build_trial_matrix_drops_ts_column_int64_conversion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Targets registry.py:374-380 int64 conversion failure branch, otherwise
    unreachable because real pandas int dtypes wrap rather than raise."""
    db_path = tmp_path / "trials.db"
    registry = TrialRegistry(db_path)

    result_dir = tmp_path / "trial_overflow"
    result_dir.mkdir()
    (result_dir / "returns.parquet").touch()

    registry.record(_make_record("overflow1", result_path=str(result_dir)))

    fake_frame = _FakeFrame(
        {
            "ts": _FakeColumn(
                np.dtype("int64"),
                [1, 2, 3],
                raise_on_to_numpy=OverflowError("synthetic int64 overflow"),
            ),
            "return": _FakeColumn(np.dtype("float64"), [0.1, 0.2, 0.3]),
        }
    )

    def _fake_read_parquet(path: object, *args: object, **kwargs: object) -> _FakeFrame:
        return fake_frame

    monkeypatch.setattr(registry_module.pd, "read_parquet", _fake_read_parquet)

    trial_matrix = registry.build_trial_matrix(trial_ids=["overflow1"])

    assert trial_matrix.n_dropped == 1
    assert trial_matrix.drop_reasons == {
        "overflow1": "ts column could not be converted to int64 (OverflowError)"
    }
    assert trial_matrix.trial_ids == ()
