"""Tests for sizing an embargo over the complete dependence horizon.

Specification defects, ambiguities, and contradictions noted while writing these tests:

* The specification cites ``cli.py:851-852`` for the walk-forward call site, but at
  HEAD the relevant ``splitter.split(trading_dates)`` call is approximately at
  lines 657-658; the line references are stale.
* Section B says that ``split()`` accepts four components rather than one
  pre-summed value, but it does not define their exact parameter names, types, or
  units.  It also does not say whether PurgedKFold's replacement for
  ``embargo_frac`` is a dataclass field or a split-time keyword, or whether that
  value is in sessions or already-converted rows.
* Section A requires conversion through ``panel.day_offsets``, while
  WalkForwardSplitter receives ``trading_dates`` containing one entry per
  session.  A feature lookback such as 390 Hurst bars is naturally measured in
  bars, so the boundary at which bars become sessions is unspecified.
* The CLI has no specified API for obtaining the four strategy and feature
  properties, nor does it specify corresponding command-line options or config
  keys.  Consequently, the exact production wiring that arms the guard is
  underspecified.
* Item 5 calls the K-fold and walk-forward embargo regions “consistent in size”
  without defining whether that means exact equality, equality after conversion,
  or a tolerance when K-fold uses rows and walk-forward uses sessions.
* Item 2 does not specify a tie-break when multiple components have the same
  maximum value.  The dominant-term test therefore uses a uniquely dominant
  component.
* The wording that K-fold “purges on both sides” is potentially misleading:
  the amendment clarifies that the left operation is label-horizon purging and
  the right operation is a separate holding/dependence embargo.
* The behavior for missing, negative, non-finite, or otherwise undeclared
  component values is not fully specified, although the no-silent-zero
  constraint implies that an unavailable term must raise.
* The conversion behavior when ``day_offsets`` is omitted, or when a requested
  bar horizon exceeds the sessions available in the offsets array, is not
  explicitly defined.
* The provenance requirement is stated, but no exact provenance field or
  recording API is supplied; Amendment 1 explicitly puts provenance recording
  out of scope for these tests.
* Item 8 does not say whether “current behaviour” means WalkForwardSplitter,
  PurgedKFold, or both.  These tests use WalkForwardSplitter because it is the
  mechanism wired into ``nq walkforward``.
* The prescribed item-1 spy still observes the legacy ``max_lookback_days``
  keyword, whereas Section B and the assumed future API say that the four
  components replace that path.  The item-1 tests deliberately retain the
  prescribed legacy spy to expose the current CLI wiring defect.

AMENDMENT 1 was appended to the specification mid-draft on 2026-08-20.  These
tests incorporate its resolutions: the existing fraction-based embargo grows
with ``n_rows`` rather than having the originally claimed opposite direction;
the guard raises only for strict ``<`` and equality passes; all float
components are summed and rounded with one final ``ceil``; ``embargo_frac`` is
deleted; ``1/a`` applies only to exponential smoothing and invalid ``a`` raises;
and PurgedKFold's left purge remains label-horizon-only while its absolute
right embargo remains separate.

Assumed future API used by items 2-8:

# assumed, does not exist at HEAD — new module:
nifty_quant.research.embargo.ewma_effective_memory_sessions(a: float) -> float
    # returns 1.0 / a for 0 < a <= 1; RAISES (ValueError) for a <= 0 (undefined/infinite memory,
    # must not silently return 0 or inf)

nifty_quant.research.embargo.bars_to_sessions(n_bars: int, day_offsets: np.ndarray) -> int
    # smallest m such that the sum of the LAST m sessions' bar-counts (from day_offsets,
    # walking backward from the end) is >= n_bars

nifty_quant.research.embargo.required_embargo_sessions(
    feature_lookback: float, label_horizon: float, holding_period: float,
    execution_horizon: float,
) -> int
    # math.ceil(feature_lookback + label_horizon + holding_period + execution_horizon) -- the
    # four components are floats, summed first, and rounded to a whole session count ONCE
    # at the end (never per-component -- see AMENDMENT 1.2). This is the "one sizing rule"
    # of section A.

# assumed, does not exist at HEAD — WalkForwardSplitter.split() gains 4 new keyword-only params
# (units: SESSIONS, i.e. already comparable to trading_dates / embargo_days; may be non-integer
# before rounding, but pass whole-number test values so you don't also have to guess where
# split() itself rounds), REPLACING max_lookback_days for the "armed" path (max_lookback_days
# keeps working for item-1's HEAD-level test only, since that parameter already exists today):
WalkForwardSplitter.split(
    self, trading_dates, *,
    feature_lookback: float = 0, label_horizon: float = 0,
    holding_period: float = 0, execution_horizon: float = 0,
) -> list[Split]
    # required = math.ceil(feature_lookback + label_horizon + holding_period + execution_horizon)
    # raises EmbargoTooShortError iff self.embargo_days < required (strict; equality passes,
    # per AMENDMENT 1.2), naming whichever of the four components is largest (its NAME must
    # appear as a substring of str(exc), e.g. "holding_period")

# assumed, does not exist at HEAD — PurgedKFold's `embargo_frac: float = 0.01` field is DELETED
# (AMENDMENT 1.2 -- passing embargo_frac to the future PurgedKFold must raise TypeError) and
# replaced by a new absolute dataclass field:
PurgedKFold(..., embargo_sessions: int = 0)
    # combined with day_offsets passed to .split(): the embargo region right of each test block
    # is embargo_sessions worth of session rows (converted via day_offsets), instead of
    # ceil(embargo_frac * n_rows). This is an ABSOLUTE size -- it must not scale with n_rows.
    # label_horizon (purge, left of test) is UNCHANGED and stays separate from embargo_sessions
    # (AMENDMENT 1.2 -- do not unify the two).
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from nifty_quant.research.cv import PurgedKFold
from nifty_quant.research.splits import EmbargoTooShortError, WalkForwardSplitter
from tests.test_cli_tradable import (
    ALL_DATES,
    _make_panel,
    _patch_walkforward,
    _walkforward_args,
    app,
    runner,
)


def test_walkforward_cli_arms_the_embargo_guard(monkeypatch, tmp_path):
    """GREEN when implemented: verifies that nq walkforward arms the embargo guard with
    nonzero dependence-horizon components."""
    calls = []
    panel = _make_panel(ALL_DATES, [5] * 12, symbols=("A", "B"))
    _patch_walkforward(monkeypatch, tmp_path, panel, calls)

    captured = {}
    real_split = WalkForwardSplitter.split

    def _spy_split(
        self,
        trading_dates,
        *,
        feature_lookback=0,
        label_horizon=0,
        holding_period=0,
        execution_horizon=0,
        max_lookback_days=0,
    ):
        captured["feature_lookback"] = feature_lookback
        captured["label_horizon"] = label_horizon
        captured["holding_period"] = holding_period
        captured["execution_horizon"] = execution_horizon
        captured["max_lookback_days"] = max_lookback_days
        captured["embargo_days"] = self.embargo_days
        return real_split(
            self,
            trading_dates,
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
            max_lookback_days=max_lookback_days,
        )

    monkeypatch.setattr(WalkForwardSplitter, "split", _spy_split)

    runner.invoke(app, _walkforward_args())

    assert "feature_lookback" in captured, (
        "WalkForwardSplitter.split was never called by `nq walkforward`"
    )

    any_component_nonzero = any(
        captured.get(key, 0) > 0
        for key in ("feature_lookback", "label_horizon", "holding_period", "execution_horizon")
    )
    assert any_component_nonzero, (
        "nq walkforward must compute and forward at least one nonzero component "
        "(feature_lookback, label_horizon, holding_period, or execution_horizon, in sessions) "
        "so EmbargoTooShortError can fire. If all components are zero, the guard cannot be "
        "armed and is decoration. Got: "
        f"feature_lookback={captured.get('feature_lookback', 0)}, "
        f"label_horizon={captured.get('label_horizon', 0)}, "
        f"holding_period={captured.get('holding_period', 0)}, "
        f"execution_horizon={captured.get('execution_horizon', 0)}"
    )


def test_embargo_guard_fires_and_aborts_walkforward_once_armed(monkeypatch, tmp_path):
    """GREEN at HEAD: verifies that an armed guard aborts the CLI before backtesting."""
    calls = []
    panel = _make_panel(ALL_DATES, [5] * 12, symbols=("A", "B"))
    _patch_walkforward(monkeypatch, tmp_path, panel, calls)

    real_split = WalkForwardSplitter.split

    def _armed_split(self, trading_dates, *, max_lookback_days=0):
        return real_split(self, trading_dates, max_lookback_days=self.embargo_days + 1)

    monkeypatch.setattr(WalkForwardSplitter, "split", _armed_split)

    result = runner.invoke(app, _walkforward_args())

    assert result.exit_code != 0
    assert "embargo" in result.output.lower()
    assert calls == []


def test_item2_embargo_error_names_the_dominant_component():
    """RED at HEAD: verifies that a short embargo error identifies its dominant term."""
    trading_dates = pd.bdate_range("2024-01-01", periods=30).date.tolist()
    components = {
        "feature_lookback": 1.0,
        "label_horizon": 1.0,
        "holding_period": 1.0 / 0.10,
        "execution_horizon": 1.0,
    }
    dominant = max(components, key=components.__getitem__)

    with pytest.raises(EmbargoTooShortError) as exc_info:
        WalkForwardSplitter().split(trading_dates, **components)

    assert dominant in str(exc_info.value)


@pytest.mark.parametrize("a", (0.05, 0.10, 0.25))
def test_item3_ewma_memory_uses_effective_memory_formula(a):
    """RED at HEAD: verifies EMA holding memory is calculated as 1/a."""
    from nifty_quant.research.embargo import ewma_effective_memory_sessions

    assert ewma_effective_memory_sessions(a) == pytest.approx(1.0 / a)


def test_item3_ewma_zero_rate_raises():
    """RED at HEAD: verifies undefined infinite EMA memory raises ValueError."""
    from nifty_quant.research.embargo import ewma_effective_memory_sessions

    with pytest.raises(ValueError):
        ewma_effective_memory_sessions(a=0.0)


def test_item4_bars_to_sessions_uses_variable_day_offsets():
    """RED at HEAD: verifies bar horizons use session offsets rather than 375 bars per day."""
    from nifty_quant.research.embargo import bars_to_sessions

    day_offsets = np.array([0, 60, 165], dtype=np.int64)

    n_bars = 120
    converted = bars_to_sessions(n_bars, day_offsets)
    naive_375_sessions = math.ceil(n_bars / 375)

    assert converted == 2
    assert naive_375_sessions == 1
    assert converted != naive_375_sessions

    assert bars_to_sessions(90, day_offsets) == 1


def test_item5_walkforward_and_kfold_use_the_same_absolute_embargo_size():
    """RED at HEAD: verifies equal session sizing, with K-fold rows and walk-forward sessions."""
    from nifty_quant.research.embargo import required_embargo_sessions

    session_bars = 10
    day_offsets = np.arange(0, 751, session_bars, dtype=np.int64)
    trading_dates = pd.bdate_range(
        "2024-01-01", periods=len(day_offsets) - 1
    ).date.tolist()

    required = required_embargo_sessions(
        feature_lookback=2,
        label_horizon=2,
        holding_period=4,
        execution_horizon=2,
    )
    assert required == math.ceil(2 + 2 + 4 + 2)
    assert required == 10

    walkforward_splits = WalkForwardSplitter(
        train_years=0.05,
        test_years=0.05,
        step_years=0.05,
        embargo_days=required,
    ).split(
        trading_dates,
        feature_lookback=2,
        label_horizon=2,
        holding_period=4,
        execution_horizon=2,
    )
    assert walkforward_splits

    with pytest.raises(EmbargoTooShortError):
        WalkForwardSplitter(
            train_years=0.05,
            test_years=0.05,
            step_years=0.05,
            embargo_days=required - 1,
        ).split(
            trading_dates,
            feature_lookback=2,
            label_horizon=2,
            holding_period=4,
            execution_horizon=2,
        )

    date_positions = {day: position for position, day in enumerate(trading_dates)}
    for split_record in walkforward_splits:
        assert split_record.embargo_days == required
        train_end = date_positions[split_record.train[1]]
        test_start = date_positions[split_record.test[0]]
        assert test_start - train_end - 1 == required

    n_rows = int(day_offsets[-1])
    expected_embargo_rows = required * session_bars
    folds = PurgedKFold(
        n_splits=5,
        label_horizon=2,
        embargo_sessions=required,
    ).split(n_rows, day_offsets=day_offsets)

    non_final_folds = [
        fold
        for fold in folds
        if int(np.max(fold.test_idx)) + 1 + expected_embargo_rows <= n_rows
    ]
    assert non_final_folds

    for fold in non_final_folds:
        test_end = int(np.max(fold.test_idx)) + 1
        right_side = np.arange(
            test_end, test_end + expected_embargo_rows, dtype=np.int64
        )
        all_rows_after_test = np.arange(test_end, n_rows, dtype=np.int64)
        rows_after_embargo = np.arange(
            test_end + expected_embargo_rows, n_rows, dtype=np.int64
        )

        assert fold.n_embargoed == expected_embargo_rows
        assert np.intersect1d(
            fold.train_idx, right_side, assume_unique=True
        ).size == 0
        assert np.array_equal(
            np.intersect1d(
                fold.train_idx, all_rows_after_test, assume_unique=True
            ),
            rows_after_embargo,
        )


def test_item6_absolute_embargo_does_not_scale_with_sample_size():
    """RED at HEAD: verifies an absolute session embargo remains fixed as n_rows doubles."""
    session_bars = 10
    embargo_sessions = 5
    expected_fixed_width = embargo_sessions * session_bars
    sample_sizes = (500, 1000)

    observed_widths = []
    old_fraction_widths = []

    for n_rows in sample_sizes:
        day_offsets = np.arange(
            0, n_rows + session_bars, session_bars, dtype=np.int64
        )
        folds = PurgedKFold(
            n_splits=5,
            label_horizon=0,
            embargo_sessions=embargo_sessions,
        ).split(n_rows, day_offsets=day_offsets)

        eligible_folds = [
            fold
            for fold in folds
            if int(np.max(fold.test_idx)) + 1 + expected_fixed_width <= n_rows
        ]
        assert eligible_folds

        widths = [fold.n_embargoed for fold in eligible_folds]
        assert set(widths) == {expected_fixed_width}
        observed_widths.append(widths[0])

        old_fraction_widths.append(math.ceil(0.01 * n_rows))

    assert observed_widths == [expected_fixed_width, expected_fixed_width]
    assert all(
        expected_fixed_width != old_width for old_width in old_fraction_widths
    )


# RETIRED test_item6_documents_current_embargo_frac_scaling
#
# This test documented the fraction-based embargo width scaling behaviour of the old
# PurgedKFold API (embargo_frac parameter). That API and behaviour have been deleted per
# AMENDMENT 2.1 of the embargo_sizing spec: embargo_frac was not a property of the strategy
# and introduced coupling to sample size that the spec rejects. The test passed embargo_frac=
# to PurgedKFold, while test_item6_embargo_frac_is_rejected_as_deleted_api (which follows)
# requires exactly that call to raise TypeError. These two tests were mutually contradictory.
#
# The deletion test (test_item6_embargo_frac_is_rejected_as_deleted_api) was RED at HEAD and
# is the one meant to go green, demonstrating that the API change is correctly enforced. A
# test whose purpose was to document an old contract has no role once the contract changes.
#
# See AMENDMENT 2.2 of specs/embargo_sizing.md for the adjudication.


def test_item6_embargo_frac_is_rejected_as_deleted_api():
    """RED at HEAD: verifies the deleted embargo_frac keyword is rejected."""
    with pytest.raises(TypeError):
        PurgedKFold(n_splits=5, embargo_frac=0.01)


def test_item7_random_splits_are_disjoint_and_respect_both_regions():
    """RED at HEAD: verifies disjointness, label purging, and absolute right embargo, over
    random cases."""
    rng = random.Random(20260820)

    for _ in range(40):
        n_splits = rng.randint(2, 6)
        label_horizon = rng.randint(0, 15)
        embargo_sessions = rng.randint(0, 10)
        session_bars = rng.choice((5, 10, 20))

        min_sessions = math.ceil(200 / session_bars)
        max_sessions = 900 // session_bars
        n_sessions = rng.randint(min_sessions, max_sessions)
        n_rows = n_sessions * session_bars
        day_offsets = np.arange(
            0, n_rows + session_bars, session_bars, dtype=np.int64
        )

        folds = PurgedKFold(
            n_splits=n_splits,
            label_horizon=label_horizon,
            embargo_sessions=embargo_sessions,
        ).split(n_rows, day_offsets=day_offsets)

        for fold in folds:
            train_idx = np.asarray(fold.train_idx)
            test_idx = np.asarray(fold.test_idx)
            assert np.intersect1d(
                train_idx, test_idx, assume_unique=True
            ).size == 0

            test_start = int(np.min(test_idx))
            test_end = int(np.max(test_idx)) + 1

            purge_start = max(0, test_start - label_horizon)
            left_purge = np.arange(
                purge_start, test_start, dtype=np.int64
            )
            assert np.intersect1d(
                train_idx, left_purge, assume_unique=True
            ).size == 0

            embargo_width = embargo_sessions * session_bars
            right_embargo = np.arange(
                test_end,
                min(n_rows, test_end + embargo_width),
                dtype=np.int64,
            )
            assert np.intersect1d(
                train_idx, right_embargo, assume_unique=True
            ).size == 0


def test_item8_zero_components_match_current_walkforward_behavior():
    """RED at HEAD: verifies explicit zero components preserve zero-argument walk-forward
    behavior."""
    trading_dates = pd.bdate_range(
        "2020-01-01", periods=40
    ).date.tolist()

    with_components = WalkForwardSplitter(
        train_years=0.05,
        test_years=0.05,
    ).split(
        trading_dates,
        feature_lookback=0,
        label_horizon=0,
        holding_period=0,
        execution_horizon=0,
    )
    without_components = WalkForwardSplitter(
        train_years=0.05,
        test_years=0.05,
    ).split(trading_dates)

    assert with_components == without_components
