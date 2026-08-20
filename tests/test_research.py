from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from nifty_quant.research.contract import canonical_hash
from nifty_quant.research.registry import TrialRecord, TrialRegistry
from nifty_quant.research.splits import (
    EmbargoTooShortError,
    HoldoutLock,
    WalkForwardSplitter,
)
from nifty_quant.research.sweep import expand


def _trading_dates() -> list[date]:
    """All weekdays from 2018-01-01 through 2026-08-14, ascending."""
    start = date(2018, 1, 1)
    end = date(2026, 8, 14)
    current = start
    days: list[date] = []
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _dates_between(dates: list[date], start: date, end: date) -> set[date]:
    return {d for d in dates if start <= d <= end}


def test_no_overlap():
    dates = _trading_dates()
    assert len(dates) > 2000

    splitter = WalkForwardSplitter(
        scheme="rolling",
        train_years=3,
        test_years=1,
        step_years=1,
        embargo_days=5,
    )
    splits = splitter.split(dates)

    for split in splits:
        train_dates = _dates_between(dates, split.train[0], split.train[1])
        test_dates = _dates_between(dates, split.test[0], split.test[1])
        assert train_dates & test_dates == set()


def test_embargo_holds():
    dates = _trading_dates()
    splitter = WalkForwardSplitter(
        scheme="rolling",
        train_years=3,
        test_years=1,
        step_years=1,
        embargo_days=5,
    )
    splits = splitter.split(dates)

    for split in splits:
        gap_dates = [d for d in dates if split.train[1] < d < split.test[0]]
        assert len(gap_dates) >= split.embargo_days


def test_embargo_assertion_fires():
    dates = _trading_dates()

    with pytest.raises(EmbargoTooShortError):
        WalkForwardSplitter(embargo_days=5).split(
            dates, max_lookback_days=20
        )

    # Does not raise when embargo is long enough.
    splits = WalkForwardSplitter(embargo_days=25).split(
        dates, max_lookback_days=20
    )
    assert len(splits) > 0


def test_determinism_and_coverage():
    dates = _trading_dates()

    rolling = WalkForwardSplitter(
        scheme="rolling",
        train_years=3,
        test_years=1,
        step_years=1,
        embargo_days=5,
    )
    rolling_splits = rolling.split(dates)
    assert rolling_splits == rolling.split(dates)
    assert len(rolling_splits) > 1

    anchored = WalkForwardSplitter(
        scheme="anchored",
        train_years=3,
        test_years=1,
        step_years=1,
        embargo_days=5,
    )
    anchored_splits = anchored.split(dates)

    assert all(s.train[0] == anchored_splits[0].train[0] for s in anchored_splits)
    assert len({s.train[0] for s in rolling_splits}) > 1


def test_holdout_and_record_read(tmp_path):
    dates = _trading_dates()
    lock_path = tmp_path / "holdout.json"

    lock = HoldoutLock(path=lock_path, holdout_months=12)
    start, end = lock.holdout_range(dates)

    assert end == dates[-1]
    assert start > end - timedelta(days=400)
    assert start < end - timedelta(days=300)

    a = HoldoutLock(path=lock_path, holdout_months=12)
    assert a.record_read("peek1") == 1

    b = HoldoutLock(path=lock_path, holdout_months=12)
    assert b.record_read("peek2") == 2
    assert b.read_count() == 2

    c = HoldoutLock(path=lock_path)
    assert c.read_count() == 2


def _make_trial_record(
    *,
    config_hash: str,
    split_id: str,
    sharpe_net: float | None,
    error: str | None = None,
) -> TrialRecord:
    return TrialRecord(
        config_hash=config_hash,
        ts="2024-01-01T00:00:00+00:00",
        strategy="test_strategy",
        params_json="{}",
        split_id=split_id,
        purpose="exploration",
        sharpe_gross=None,
        sharpe_net=sharpe_net,
        n_trades=5,
        turnover=0.01,
        breakeven_bps=1.0,
        git_sha=None,
        data_fingerprint=None,
        code_version=None,
        wall_s=0.5,
        result_path=None,
        error=error,
    )


def test_registry(tmp_path):
    registry_path = tmp_path / "trials.db"
    reg = TrialRegistry(registry_path)

    records = [
        _make_trial_record(config_hash="h1", split_id="s1", sharpe_net=0.5),
        _make_trial_record(config_hash="h2", split_id="s2", sharpe_net=0.8),
        _make_trial_record(config_hash="h3", split_id="s3", sharpe_net=1.2),
        _make_trial_record(config_hash="h4", split_id="s4", sharpe_net=None, error="exploded"),
        _make_trial_record(config_hash="h5", split_id="s5", sharpe_net=2.0),
    ]

    for rec in records:
        reg.record(rec)

    assert reg.n_trials() == 5

    # Duplicate (config_hash, split_id) rows are deliberately ignored.
    duplicate = _make_trial_record(config_hash="h1", split_id="s1", sharpe_net=99.9)
    reg.record(duplicate)
    assert reg.n_trials() == 5
    assert len(reg.all()) == 5

    non_none_sharpes = sum(1 for r in records if r.sharpe_net is not None)
    sharpes = reg.trial_sharpes()
    assert sharpes.shape == (non_none_sharpes,)

    assert reg.var_trial_sharpes() == pytest.approx(np.var(sharpes, ddof=0))

    all_records = reg.all()
    assert any(r.error == "exploded" for r in all_records)
    assert reg.n_trials() == 5


def test_sweep_expansion():
    base_params = {"z": 0}
    sweep = {
        "entry_time": [9, 10, 11],
        "decision_time": [9, 10],
        "risk": [1, 2, 3],
    }

    results = expand(base_params, sweep)
    assert len(results) == 18
    assert all("z" in result for result in results)

    # Valid (entry_time, decision_time) pairs where decision_time > entry_time:
    #   entry_time=9  -> decision_time=10 (1 valid pair)
    #   entry_time=10 -> none
    #   entry_time=11 -> none
    # Total valid pairs: 1
    # Multiplied by len(risk)=3 -> expected count is 3.
    constrained = expand(base_params, sweep, ["decision_time > entry_time"])
    assert len(constrained) == 3


def test_sweep_constraint_security():
    pwn_path = Path("/tmp/pwned_test_research")
    pwn_path.unlink(missing_ok=True)

    with pytest.raises(ValueError):
        expand(
            {},
            {"x": [1]},
            ['x > 0 or __import__("os").system("touch /tmp/pwned_test_research")'],
        )
    assert not pwn_path.exists()
    pwn_path.unlink(missing_ok=True)

    pwn_path2 = Path("/tmp/pwned_test_research2")
    pwn_path2.unlink(missing_ok=True)

    with pytest.raises(ValueError):
        expand({}, {"x": [1]}, ['__import__("os") == 1'])
    assert not pwn_path2.exists()
    pwn_path2.unlink(missing_ok=True)


def test_config_hash():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    h = canonical_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)

    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
