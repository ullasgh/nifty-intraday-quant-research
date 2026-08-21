"""Tests for `scripts/run_phase_e_sweep.py`'s SHARD/MERGE plumbing.

Not a Rule 1 dual-suite spec unit (this is a one-off orchestration script, not a research
primitive under `specs/`) -- written directly by the same worker that wrote the script, as
directed. Scope: only the parts of the script that don't require loading the real panel
(sharding partition, spread-series alignment against the audited existing helper, and
merge_shards's arithmetic on synthetic shard files) -- exercising the full CLI against real
data is explicitly out of scope for this runner (see the script's own module docstring).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_phase_e_sweep as runner  # noqa: E402

from nifty_quant.research import feature_sweep as fs  # noqa: E402
from nifty_quant.research import sweep_features as sf  # noqa: E402
from nifty_quant.research.registry import TrialRecord  # noqa: E402


def test_parse_horizons_default_matches_registry():
    assert runner._parse_horizons(None) == list(sf.HORIZONS)


def test_parse_horizons_explicit_list_including_eod():
    assert runner._parse_horizons("1,5,EOD") == [1, 5, "EOD"]


@pytest.mark.parametrize("n_shards", [2, 3, 4])
def test_shards_are_disjoint_and_union_is_whole_registry(n_shards):
    """The exact invariant the task brief requires: shards partition FEATURE_REGISTRY.

    Calls the SCRIPT's own `shard_features_for`, not a re-derived slice expression, so a
    bug in that function (e.g. overlapping shards, or a dropped feature) fails this test.
    """
    registry = list(sf.FEATURE_REGISTRY)
    shards = [runner.shard_features_for(i, n_shards) for i in range(n_shards)]

    union_names = []
    for shard in shards:
        union_names.extend(f.name for f in shard)
    assert sorted(union_names) == sorted(f.name for f in registry)

    seen = set()
    for shard in shards:
        names = {f.name for f in shard}
        assert not (names & seen), "shards must be pairwise disjoint"
        seen |= names


def test_shard_features_for_rejects_out_of_range_shard_index():
    with pytest.raises(ValueError):
        runner.shard_features_for(4, 4)
    with pytest.raises(ValueError):
        runner.shard_features_for(-1, 4)


def test_raw_spread_series_matches_audited_helper_after_dropping_nans():
    """`_raw_spread_series` is a near-duplicate of `feature_sweep._bucket_spread_returns`
    that keeps NaN rows for cross-trial alignment; its finite entries must be IDENTICAL to
    the already-tested function's output, in the same order."""
    rng = np.random.default_rng(0)
    n_rows, n_symbols = 40, 6
    close = 100.0 + np.cumsum(rng.normal(size=(n_rows, n_symbols)), axis=0)
    day_offsets = np.array([0, 20, n_rows], dtype=np.int64)

    from nifty_quant.research import expectancy

    fwd = expectancy.forward_returns(close, day_offsets, horizon=1)
    feature_values = rng.normal(size=(n_rows, n_symbols))

    raw = runner._raw_spread_series(feature_values, fwd.values, day_offsets, n_buckets=3)
    audited = fs._bucket_spread_returns(feature_values, fwd.values, day_offsets, n_buckets=3)

    assert raw.shape == (n_rows,)
    np.testing.assert_allclose(raw[np.isfinite(raw)], audited)


def _fake_trial_record(strategy: str, error: str | None) -> TrialRecord:
    return TrialRecord(
        config_hash="c",
        ts="2026-01-01T00:00:00+00:00",
        strategy=strategy,
        params_json="{}",
        split_id="sweep",
        purpose="exploration",
        sharpe_gross=None,
        sharpe_net=None,
        n_trades=None,
        turnover=None,
        breakeven_bps=None,
        git_sha=None,
        data_fingerprint=None,
        code_version="test",
        wall_s=0.01,
        result_path=None,
        error=error,
    )


def _write_fake_shard(output_dir: Path, shard: int, n_shards: int, spread_series, records) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = runner._shard_output_path(output_dir, shard, n_shards)
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "shard": shard,
                "n_shards": n_shards,
                "records": records,
                "spread_series": spread_series,
                "n_rows": next(iter(spread_series.values())).shape[0] if spread_series else 0,
                "horizons": [1],
            },
            fh,
        )


def test_merge_shards_combines_disjoint_shard_files_and_reports_measured_quantities(tmp_path):
    # Two shards, two trials each -- one trial per shard deliberately made a near-duplicate
    # of another (high correlation) so effective_n_trials is measurably BELOW the planned
    # count of 4, and one trial in shard 1 recorded as FAILED (never dropped, per E5).
    rng = np.random.default_rng(1)
    base_series = rng.normal(scale=0.01, size=200)

    shard0_series = {
        ("feat_a", "1"): base_series,
        ("feat_b", "1"): base_series + rng.normal(scale=1e-6, size=200),  # near-duplicate
    }
    shard0_records = [
        _fake_trial_record("sweep::feat_a", None),
        _fake_trial_record("sweep::feat_b", None),
    ]
    shard1_series = {
        ("feat_c", "1"): rng.normal(scale=0.01, size=200),
    }
    shard1_records = [
        _fake_trial_record("sweep::feat_c", None),
        _fake_trial_record("sweep::feat_d", "ValueError: synthetic failure"),
    ]

    _write_fake_shard(tmp_path, 0, 2, shard0_series, shard0_records)
    _write_fake_shard(tmp_path, 1, 2, shard1_series, shard1_records)

    merged = runner.merge_shards(tmp_path, 2)

    assert merged["n_planned_run"] == 4  # 2 + 2 TrialRecords across both shard files
    assert merged["matrix_shape"] == (200, 3)  # 3 successful trials, all fully finite here

    # Near-duplicate feat_a/feat_b inflate correlation -> n_eff measurably below 3 trials.
    assert 1.0 <= merged["n_eff"] < 3.0

    assert 0.0 <= merged["pbo"] <= 1.0

    assert merged["var_trial_sharpes"] >= 0.0
    assert np.isfinite(merged["var_trial_sharpes"])

    assert set(merged["per_trial_sharpe"].keys()) == {
        ("feat_a", "1"),
        ("feat_b", "1"),
        ("feat_c", "1"),
    }
    for dsr in merged["per_trial_dsr"].values():
        assert np.isnan(dsr) or np.isfinite(dsr)

    assert len(merged["failed_records"]) == 1
    assert merged["failed_records"][0].strategy == "sweep::feat_d"


def test_merge_shards_never_reads_a_stale_third_file(tmp_path):
    """merge_shards(output_dir, n_shards=2) must only read shard_0_of_2 and shard_1_of_2,
    never a leftover shard_0_of_4 from a differently-shaped previous run."""
    series = {("only_feat", "1"): np.linspace(-0.01, 0.01, 50)}
    records = [_fake_trial_record("sweep::only_feat", None)]
    _write_fake_shard(tmp_path, 0, 4, series, records)  # stale, different n_shards
    _write_fake_shard(tmp_path, 0, 2, series, records)
    _write_fake_shard(tmp_path, 1, 2, series, records)

    merged = runner.merge_shards(tmp_path, 2)
    assert merged["n_planned_run"] == 2  # only the two n_shards=2 files, not the stale one


# ---------------------------------------------------------------------------
# Hardening: all-NaN / zero-variance trials must never collapse the WHOLE matrix.
# ---------------------------------------------------------------------------


def test_classify_trial_series_all_nan():
    """The `rolling_hurst`-at-window=390-vs-~375-bar-sessions case: 0 finite observations."""
    series = np.full(92520, np.nan)
    reason = runner._classify_trial_series(series)
    assert reason is not None
    assert "all-nan" in reason.lower()


def test_classify_trial_series_zero_variance():
    series = np.full(200, 0.5)
    reason = runner._classify_trial_series(series)
    assert reason is not None
    assert "zero-variance" in reason.lower()


def test_classify_trial_series_single_finite_observation():
    series = np.full(200, np.nan)
    series[10] = 0.01
    reason = runner._classify_trial_series(series)
    assert reason is not None
    assert "insufficient" in reason.lower()


def test_classify_trial_series_usable_returns_none():
    rng = np.random.default_rng(2)
    series = rng.normal(scale=0.01, size=200)
    assert runner._classify_trial_series(series) is None


def test_merge_shards_excludes_all_nan_column_without_collapsing_matrix(tmp_path):
    """The exact failure mode the coordinator flagged: ONE all-NaN column (e.g. an
    over-windowed rolling_hurst against day-bounded sessions) must be EXCLUDED, not allowed
    to make the "rows where ALL trials are finite" intersection empty for every other trial.
    """
    rng = np.random.default_rng(3)
    n_rows = 300
    good_a = rng.normal(scale=0.01, size=n_rows)
    good_b = rng.normal(scale=0.01, size=n_rows)
    all_nan = np.full(n_rows, np.nan)  # e.g. rolling_hurst(window=390) vs ~375-bar sessions

    series = {
        ("good_feat_a", "1"): good_a,
        ("good_feat_b", "1"): good_b,
        ("rolling_hurst", "1"): all_nan,
    }
    records = [
        _fake_trial_record("sweep::good_feat_a", None),
        _fake_trial_record("sweep::good_feat_b", None),
        _fake_trial_record("sweep::rolling_hurst", None),
    ]
    _write_fake_shard(tmp_path, 0, 1, series, records)

    merged = runner.merge_shards(tmp_path, 1)

    # The two good columns must still form a real, non-empty matrix.
    assert merged["matrix_shape"] == (n_rows, 2)
    assert merged["n_rows_before_intersection"] == n_rows
    assert merged["n_rows_after_intersection"] == n_rows
    assert merged["rows_dropped"] == 0
    assert np.isfinite(merged["n_eff"])
    assert np.isfinite(merged["pbo"])

    # The all-NaN column is reported as excluded, not silently dropped and not filled.
    assert set(merged["excluded_trials"].keys()) == {("rolling_hurst", "1")}
    assert "all-nan" in merged["excluded_trials"][("rolling_hurst", "1")].lower()
    # And it never contaminates the good columns' own per-trial Sharpe/DSR bookkeeping.
    assert np.isnan(merged["per_trial_sharpe"][("rolling_hurst", "1")])


def test_merge_shards_excludes_zero_variance_column(tmp_path):
    rng = np.random.default_rng(4)
    n_rows = 150
    good = rng.normal(scale=0.01, size=n_rows)
    constant = np.full(n_rows, 1.0)

    series = {("good_feat", "1"): good, ("constant_feat", "1"): constant}
    records = [
        _fake_trial_record("sweep::good_feat", None),
        _fake_trial_record("sweep::constant_feat", None),
    ]
    _write_fake_shard(tmp_path, 0, 1, series, records)

    merged = runner.merge_shards(tmp_path, 1)

    assert merged["matrix_shape"] == (n_rows, 1)  # only good_feat survives
    assert set(merged["excluded_trials"].keys()) == {("constant_feat", "1")}
    assert "zero-variance" in merged["excluded_trials"][("constant_feat", "1")].lower()
    # Only 1 usable column left -> effective_n_trials is trivially 1.0, with an explicit note.
    assert merged["n_eff"] == 1.0
    assert merged["n_eff_note"] is not None
    assert "1 usable" in merged["n_eff_note"]
    # PBO needs >= 2 columns -- explicitly NOT computed, never a bare unexplained nan.
    assert np.isnan(merged["pbo"])
    assert merged["pbo_note"] is not None
    assert "2 trial columns" in merged["pbo_note"]


def test_merge_shards_reports_warmup_row_drop_distinct_from_a_massive_drop(tmp_path):
    """Leading warmup NaNs (harmless, expected) must be distinguishable from a near-total
    intersection collapse (a red flag) purely from the reported row-drop accounting."""
    rng = np.random.default_rng(5)
    n_rows = 1000
    warmup = 30  # small, harmless leading warmup shared by both trials

    feat_a = rng.normal(scale=0.01, size=n_rows)
    feat_a[:warmup] = np.nan
    feat_b = rng.normal(scale=0.01, size=n_rows)
    feat_b[:warmup] = np.nan

    series = {("feat_a", "1"): feat_a, ("feat_b", "1"): feat_b}
    records = [
        _fake_trial_record("sweep::feat_a", None),
        _fake_trial_record("sweep::feat_b", None),
    ]
    _write_fake_shard(tmp_path, 0, 1, series, records)

    merged = runner.merge_shards(tmp_path, 1)

    assert merged["n_rows_before_intersection"] == n_rows
    assert merged["n_rows_after_intersection"] == n_rows - warmup
    assert merged["rows_dropped"] == warmup
    assert merged["rows_dropped_pct"] == pytest.approx(warmup / n_rows * 100.0)
    assert merged["rows_dropped_pct"] < 5.0  # a small, harmless warmup drop


def test_merge_shards_notes_insufficient_rows_after_exclusion_instead_of_bare_nan(tmp_path):
    """After exclusions, if T < 2 the report must say so explicitly rather than emit a bare,
    unexplained nan for effective_n_trials/PBO."""
    feat_a = np.array([0.01, np.nan, 0.02, np.nan, np.nan])  # finite at rows 0, 2
    feat_b = np.array([np.nan, np.nan, 0.03, 0.04, np.nan])  # finite at rows 2, 3 -- overlap=row 2

    series = {("feat_a", "1"): feat_a, ("feat_b", "1"): feat_b}
    records = [
        _fake_trial_record("sweep::feat_a", None),
        _fake_trial_record("sweep::feat_b", None),
    ]
    _write_fake_shard(tmp_path, 0, 1, series, records)

    merged = runner.merge_shards(tmp_path, 1)

    assert merged["matrix_shape"] == (1, 2)  # both usable, but only 1 row survives intersection
    assert not merged["excluded_trials"]
    assert np.isnan(merged["n_eff"])
    assert merged["n_eff_note"] is not None
    assert "T=1" in merged["n_eff_note"]
    assert np.isnan(merged["pbo"])
    assert merged["pbo_note"] is not None
    assert "T=1" in merged["pbo_note"]
