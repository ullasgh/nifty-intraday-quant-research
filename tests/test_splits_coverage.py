import fcntl
import importlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

import nifty_quant.research.splits as splits


def _make_trading_dates(count: int) -> list[date]:
    dates: list[date] = []
    day = date(2024, 1, 1)
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def test_import_fallback_when_fcntl_import_raises() -> None:
    """The `except ImportError: fcntl = None` fallback at import time.

    Restoration is done by hand in a `finally`, NOT via monkeypatch, and that is the
    whole point of this comment.

    `monkeypatch.delitem(sys.modules, ...)` restores `sys.modules` correctly, but
    `importlib.import_module` ALSO rebinds the attribute on the parent package, and
    monkeypatch knows nothing about that. The result was two live module objects:

        sys.modules["nifty_quant.research.splits"]   -> the original
        nifty_quant.research.splits                  -> the reloaded copy

    Code reaching the module by `from ... import X` got one; code reaching it as a
    package attribute got the other. A test monkeypatching `WalkForwardSplitter.split`
    on one object therefore had no effect on the copy the CLI actually resolved, and the
    run fell through to real splitting on a short synthetic calendar.

    Cost of that leak: five tests in `test_pbo_dsr_wiring_a.py` and `test_cli_coverage.py`
    failing under some shuffle orders only -- they passed individually, in pairs, across
    25 seeds of their own file, and in the full suite in deterministic order. Found by
    bisecting a reproducing shuffle seed down to a two-test pair.

    Anything that deletes a module from `sys.modules` and re-imports it must restore the
    parent package attribute too, or it leaves module identity split behind it.
    """
    import nifty_quant.research as research_pkg

    original_mod = sys.modules["nifty_quant.research.splits"]
    real_fcntl = fcntl
    try:
        del sys.modules["nifty_quant.research.splits"]
        sys.modules["fcntl"] = None

        reloaded = importlib.import_module("nifty_quant.research.splits")
        assert reloaded.fcntl is None
    finally:
        sys.modules["fcntl"] = real_fcntl
        sys.modules["nifty_quant.research.splits"] = original_mod
        research_pkg.splits = original_mod


def test_record_read_works_when_fcntl_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = splits.HoldoutLock(path=tmp_path / "lock.json")
    monkeypatch.setattr(splits, "fcntl", None)

    assert lock.record_read("probe") == 1
    assert lock.read_count() == 1


@pytest.mark.parametrize("field", ["train_years", "test_years", "step_years"])
def test_non_positive_window_years_raise_value_error_not_embargo(
    field: str,
) -> None:
    kwargs = {"train_years": 3.0, "test_years": 1.0, "step_years": 1.0}
    kwargs[field] = 0.0

    with pytest.raises(ValueError) as exc_info:
        splits.WalkForwardSplitter(**kwargs).split(_make_trading_dates(10))

    assert type(exc_info.value) is ValueError
    assert field in str(exc_info.value)


def test_negative_embargo_days_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        splits.WalkForwardSplitter(embargo_days=-1).split(_make_trading_dates(10))

    assert type(exc_info.value) is ValueError
    assert "embargo_days" in str(exc_info.value)


def test_split_empty_trading_dates_returns_empty() -> None:
    assert splits.WalkForwardSplitter().split([]) == []


def test_window_length_rounds_to_zero_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        splits.WalkForwardSplitter(
            train_years=0.001,
            test_years=1.0,
            step_years=1.0,
        ).split(_make_trading_dates(10))

    assert type(exc_info.value) is ValueError
    assert "trading day" in str(exc_info.value)


@pytest.mark.parametrize("scheme", ["rolling", "anchored"])
def test_short_trading_dates_return_no_splits(scheme: str) -> None:
    result = splits.WalkForwardSplitter(scheme=scheme).split(_make_trading_dates(5))
    assert result == []


def test_holdout_range_empty_trading_dates_raises_value_error(
    tmp_path: Path,
) -> None:
    lock = splits.HoldoutLock(path=tmp_path / "unused.json")

    with pytest.raises(ValueError) as exc_info:
        lock.holdout_range([])

    assert "trading_dates" in str(exc_info.value)


def test_read_count_missing_file_returns_zero(tmp_path: Path) -> None:
    lock = splits.HoldoutLock(path=tmp_path / "nonexistent.json")

    assert lock.read_count() == 0


def test_read_count_non_dict_json_returns_zero(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text("[]", encoding="utf-8")
    lock = splits.HoldoutLock(path=path)

    assert lock.read_count() == 0


def test_atomic_write_cleans_up_tmp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _raise_oserror(_src: object, _dst: object) -> None:
        raise OSError("simulated replace failure")

    path = tmp_path / "lock.json"
    lock = splits.HoldoutLock(path=path)

    monkeypatch.setattr(splits.os, "replace", _raise_oserror)

    with pytest.raises(OSError):
        lock.record_read("probe")

    assert not path.exists()
    assert list(tmp_path.glob("lock.json.*.tmp")) == []


def test_record_read_existing_non_dict_json_resets_state(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    lock = splits.HoldoutLock(path=path)

    assert lock.record_read("probe") == 1
    assert lock.read_count() == 1

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert len(data["log"]) == 1
    assert data["log"][0]["reason"] == "probe"
