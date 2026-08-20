"""Red-state TDD spec file for `specs/embargo_sizing.md` (Phase B track B2).

The module `nifty_quant.research.embargo` does not exist yet; importing it here makes
most of this file fail at collection with an ImportError, which is the expected red
state (same convention as `test_purged_cv.py`).

API-surface assumptions (this author's own design decision, since the spec does not
pin exact signatures):

- New module `nifty_quant.research.embargo` with:
  - `EmbargoComponents` frozen dataclass: fields `feature_lookback`, `label_horizon`,
    `holding_period`, `execution_horizon` (all float, sessions); methods
    `required_sessions()` (sum) and `dominant_term()` (field name of largest value).
  - `effective_memory_sessions(a: float) -> float`: returns `1/a` for `0 < a <= 1`,
    raises `ValueError` otherwise.
  - `bars_to_sessions(bar_count: int, day_offsets) -> int`: converts a bar-count
    window to whole sessions by walking `day_offsets` (sorted array of row indices
    where each session starts, ending at `n_rows`). Never assumes a fixed
    bars-per-session stride (rule 5).

- `WalkForwardSplitter.split` signature changes from `max_lookback_days: int = 0` to
  four keyword-only float args `feature_lookback`, `label_horizon`, `holding_period`,
  `execution_horizon` (all default 0.0). It builds `EmbargoComponents` and raises
  `EmbargoTooShortError` if `self.embargo_days < components.required_sessions()`,
  with the dominant term's field name in the message. Calling `split(trading_dates)`
  with no kwargs is behaviourally identical to the current `max_lookback_days=0`
  behaviour (guard never fires).

- `cli.py` `walkforward` gains four new CLI options `--feature-lookback`,
  `--label-horizon`, `--holding-period`, `--execution-horizon` (all float, default
  0.0) forwarded to `splitter.split(...)`.

- `PurgedKFold.split` signature changes from `(self, n_rows, *, day_offsets=None)` to
  `(self, n_rows, *, day_offsets=None, feature_lookback=0.0, holding_period=0.0,
  execution_horizon=0.0)`. `label_horizon` stays a constructor field, in bars/rows,
  and drives ONLY the LEFT purge -- unchanged and untouched by the other three
  components. `embargo_frac` is DELETED (no longer accepted in the constructor).
  ADJUDICATED (lead correction, 2026-08-20; this file previously asserted the
  opposite): `PurgedKFold` is ASYMMETRIC, not two-sided-with-equal-width. The LEFT
  purge removes training rows whose label interval overlaps the test block -- a
  row at time `i` with horizon `h` has a label spanning `[i, i+h]`, which overlaps
  a test block starting at `t0` iff `i >= t0 - h`, so the purge width is exactly
  `label_horizon` and nothing else (feature lookback, holding period and execution
  lag do not extend a label backwards). The RIGHT embargo absorbs serial
  correlation the label-overlap rule does not catch, so its width is the full
  dependence horizon: `components.required_sessions()` (all four terms), converted
  to rows via `day_offsets`. `WalkForwardSplitter` is ONE-SIDED (only the
  train-end -> test-start gap) and uses the same all-four-component derivation as
  `PurgedKFold`'s right side. Merging the two `PurgedKFold` sides would purge far
  more training data on the left than any label-overlap rule justifies. Per
  `specs/embargo_sizing.md` AMENDMENT 1 item 2 and AMENDMENT 2 clause 6 ("must not
  be merged"), and confirmed by `test_embargo_sizing_a.py::test_item7`.

Spec defects, ambiguities, and internal contradictions found by close reading:

1. Boundary condition: the spec's `required_embargo >= ...` uses ">=" but the
   historical guard raises only on strict "<" (`self.embargo_days <
   max_lookback_days`). The spec never says whether `embargo_days ==
   required_sessions()` should pass or raise. This file assumes the historical
   strict "<" behaviour (equality passes), because the spec's item 8 says
   zero-lookback degenerates to current behaviour and the current guard is strict.

2. `holding_period` is defined only for an EMA-smoothed book (`1/a`). The spec never
   says what a non-smoothed, fixed-N-bar-hold strategy should pass — is it N bars
   directly, or does it need some transform? This file assumes the caller passes the
   holding period already expressed in sessions (or converts bars to sessions via
   `bars_to_sessions` themselves), and `effective_memory_sessions` is only for the
   EMA case.

3. Rounding: "sessions" is used as if always convertible from "bars" via
   `day_offsets`, but `label_horizon` defined in bars vs. `holding_period` derived
   as a continuous decay-to-a-factor number (`1/a`, generally non-integer) round
   differently. The spec never says whether/how to round the sum before comparing to
   integer `embargo_days`. This file assumes `required_sessions()` returns a float
   and the comparison `self.embargo_days < components.required_sessions()` is a
   float comparison (so `embargo_days=10` passes against `required_sessions()=10.0`
   but fails against `10.1`).

4. `PurgedKFold.embargo_frac`: the spec says it is "replaced." Per AMENDMENT 2,
   it is DELETED, not kept-but-ignored. Passing `embargo_frac` to the constructor
   must raise `TypeError`.

5. Section B says components are "recorded in the trial provenance (Phase B3)" —
   that is explicitly out of scope for this phase/spec and cannot be verified by
   this test file. Noted as untestable here.

6. The spec says `PurgedKFold`'s purge covers `label_horizon` on the left and
   `embargo_frac` on the right, then says both sides are derived from the same four
   components. CORRECTED (lead adjudication, 2026-08-20): purging and embargo bound
   two different dependencies and must not be unified. The left purge stays at
   `label_horizon` ALONE -- a training sample's label interval `[i, i+label_horizon]`
   overlaps the test block iff `i >= test_start - label_horizon`, and feature
   lookback, holding period and execution lag do not extend a label backwards. The
   right embargo grows to `required_sessions()` (the full four-component dependence
   horizon), because it exists to absorb serial correlation the label-overlap rule
   does not catch. Per AMENDMENT 1 item 2 and AMENDMENT 2 clause 6 ("must not be
   merged") of `specs/embargo_sizing.md`, and matching the source implementation in
   `nifty_quant.research.cv.PurgedKFold.split` (`purge_start = test_start -
   self.label_horizon`, independent of the other three components).

7. The spec says `embargo_frac` "shrinks as the sample grows — precisely backwards,"
   but the current code computes `embargo_width = math.ceil(self.embargo_frac *
   n_rows)`, which actually GROWS with sample size (not shrinks). The spec's wording
   is backwards relative to the code excerpt; this file tests the absolute-count
   behaviour (item 6) rather than the spec's literal "shrinks" claim.

LEAD-CORRECTED (disclosed, not authored by the delegated draft): three mechanical
bugs in the drafting pass were fixed by review before this file was ever run --
none change scope, weaken an assertion, or touch `src/`:
  - test_8's hand-computed expected `Split.id` used the wrong format
    (`f"split_{i}"` instead of `f"{scheme}_{i:03d}"`) and an off-by-one in
    `test_start_idx` (`train_days + 1 + embargo_days` instead of
    `train_end_idx + 1 + embargo_days`); corrected to replicate the documented
    algorithm verbatim.
  - test_6's regression-guard `holding_period` was `10`, which coincidentally
    equals `math.ceil(0.01 * 1000)` -- the OLD fraction-based width at
    `n_rows=1000` -- making the "must not equal the old fraction-based value"
    assertion pass even for an implementation that still (wrongly) used
    `embargo_frac`. Changed to `7`, which collides with neither
    `math.ceil(0.01 * 500) == 5` nor `math.ceil(0.01 * 1000) == 10`.
  - test_7's `PurgedKFold` property loop called `bars_to_sessions(required, ...)`
    with a SESSION count where the function's documented contract expects a BAR
    count, silently producing a near-zero separation threshold that would let a
    real embargo-sizing regression pass undetected. Replaced with a local
    `_sessions_to_rows` helper (the correct direction of conversion) so the
    property actually exercises the required separation.
"""

import math

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nifty_quant.cli import app
from nifty_quant.research.cv import PurgedKFold
from nifty_quant.research.embargo import (
    EmbargoComponents,
    bars_to_sessions,
    effective_memory_sessions,
)
from nifty_quant.research.splits import EmbargoTooShortError, Split, WalkForwardSplitter

# ---------------------------------------------------------------------------
# Local helpers (self-contained; do not import from other test files)
# ---------------------------------------------------------------------------

class _FakeCalendar:
    """Minimal stand-in for TradingCalendar used by CLI tests."""

    def __init__(self, dates):
        self._dates = list(dates)

    def session_dates(self, start=None, end=None, *, usable_only=True):
        return [
            d
            for d in self._dates
            if (start is None or d >= start) and (end is None or d <= end)
        ]


ALL_DATES = pd.bdate_range("2024-01-01", periods=12).date.tolist()


def _make_trading_dates(n_days=30):
    """Return n_days consecutive business days starting 2024-01-01."""
    return pd.bdate_range("2024-01-01", periods=n_days).date.tolist()


def _split_gap_sessions(split, trading_dates):
    """Return the number of trading-day positions between train end and test start.

    For a walk-forward split, train always precedes test. The gap is the number of
    trading dates strictly between train.end and test.start (i.e. the embargo width
    in trading days).
    """
    train_end_idx = trading_dates.index(split.train[1])
    test_start_idx = trading_dates.index(split.test[0])
    return test_start_idx - train_end_idx - 1


def _right_embargo_width_rows(fold, n_rows):
    """Return the number of rows embargoed on the right of the test block.

    This is the gap between the last test index and the first train index that
    appears after it (if any), or the gap to n_rows if no train index follows.
    """
    test_end = int(np.max(fold.test_idx)) if len(fold.test_idx) > 0 else -1
    train_after = fold.train_idx[fold.train_idx > test_end]
    if len(train_after) == 0:
        return n_rows - test_end - 1
    return int(np.min(train_after)) - test_end - 1


def _left_purge_width_rows(fold):
    """Return the number of rows purged on the left of the test block.

    This is the gap between the last train index before the test block and the
    first test index.
    """
    test_start = int(np.min(fold.test_idx)) if len(fold.test_idx) > 0 else 0
    train_before = fold.train_idx[fold.train_idx < test_start]
    if len(train_before) == 0:
        return test_start
    return test_start - int(np.max(train_before)) - 1


def _sessions_to_rows(n_sessions, day_offsets, start_row):
    """Return the row width of ceil(n_sessions) whole sessions walked FORWARD
    from `start_row`.

    Inverse direction of `bars_to_sessions` (which maps bars -> sessions); this
    maps a session count -> rows so a required-embargo-in-sessions figure can be
    compared against an actual row gap. POSITION-DEPENDENT: it locates the first
    session boundary at or after `start_row` and advances `n_sessions` further
    boundaries from there. This only coincides with "the first N sessions from
    row 0" when every session has the same length, which NSE sessions do not
    (rule 5: Muhurat is 60 bars, disaster-recovery sessions can be 105) --
    using a fixed-origin count here was the bug this helper used to have.

    Independently re-implemented: this walk is written directly against
    `day_offsets`, not imported from `_embargo_width_rows` in
    `nifty_quant.research.cv`, so this test keeps the ability to catch a bug in
    that production converter.

    Local test-only helper -- not part of the assumed production API.
    """
    day_offsets = np.asarray(day_offsets, dtype=np.int64)
    n_whole = int(math.ceil(n_sessions))
    if n_whole <= 0:
        return 0
    # Locate the first session boundary at or after start_row.
    session_idx = len(day_offsets) - 1
    for i, offset in enumerate(day_offsets):
        if offset >= start_row:
            session_idx = i
            break
    end_idx = min(session_idx + n_whole, len(day_offsets) - 1)
    return max(0, int(day_offsets[end_idx] - start_row))


# ---------------------------------------------------------------------------
# Test 1: EmbargoTooShortError fires from `nq walkforward`
# ---------------------------------------------------------------------------

def test_1_embargo_too_short_error_fires_from_cli(monkeypatch):
    """CLI-level test: passing components summing > embargo_days must fail.

    This is the load-bearing test: it must fail against HEAD (either because the
    CLI options don't exist yet, or because the embargo module import fails at
    collection).
    """
    import nifty_quant.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod.TradingCalendar,
        "from_index_bars",
        classmethod(lambda cls, symbol="NIFTY50": _FakeCalendar(ALL_DATES)),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "walkforward",
            "--strategy",
            "volume_breakout",
            "--config",
            "configs/strategies/volume_breakout.yaml",
            "--start",
            ALL_DATES[0].isoformat(),
            "--end",
            ALL_DATES[-1].isoformat(),
            "--train-years",
            "0.012",
            "--test-years",
            "0.012",
            "--feature-lookback",
            "3",
            "--label-horizon",
            "4",
        ],
    )
    assert result.exit_code != 0, "walkforward should fail when embargo is too short"
    assert "embargo" in result.output.lower(), (
        f"output should mention embargo, got: {result.output!r}"
    )
    assert "split setup failed" in result.output, (
        f"output should contain the _fail wrapper message, got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Error message names the dominant term
# ---------------------------------------------------------------------------

def test_2_error_message_names_dominant_term():
    """EmbargoTooShortError message must name whichever term dominates."""
    trading_dates = _make_trading_dates(30)

    # holding_period is dominant (20 > 1+1+1)
    splitter = WalkForwardSplitter(embargo_days=5)
    with pytest.raises(EmbargoTooShortError) as exc_info:
        splitter.split(
            trading_dates,
            feature_lookback=1,
            label_horizon=1,
            holding_period=20,
            execution_horizon=1,
        )
    assert "holding_period" in str(exc_info.value), (
        f"dominant term 'holding_period' not named in: {str(exc_info.value)!r}"
    )

    # feature_lookback is dominant (50 > 1+1+1)
    with pytest.raises(EmbargoTooShortError) as exc_info:
        splitter.split(
            trading_dates,
            feature_lookback=50,
            label_horizon=1,
            holding_period=1,
            execution_horizon=1,
        )
    assert "feature_lookback" in str(exc_info.value), (
        f"dominant term 'feature_lookback' not named in: {str(exc_info.value)!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: Effective memory formula for smoothed book
# ---------------------------------------------------------------------------

def test_3_effective_memory_sessions_formula():
    """Smoothed book with a=0.10 produces holding-period term of order 10."""
    # Assert against the formula, not a hardcoded number
    assert effective_memory_sessions(0.10) == pytest.approx(1.0 / 0.10), (
        "effective_memory_sessions(0.10) should equal 1/0.10"
    )
    # Independent sanity check: "of order 10"
    assert 9 < effective_memory_sessions(0.10) < 11, (
        "effective_memory_sessions(0.10) should be of order 10"
    )
    # Second data point
    assert effective_memory_sessions(0.5) == pytest.approx(2.0), (
        "effective_memory_sessions(0.5) should equal 2.0"
    )
    # Invalid values raise
    for bad_a in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            effective_memory_sessions(bad_a)


# ---------------------------------------------------------------------------
# Test 4: Component-to-session conversion uses day_offsets
# ---------------------------------------------------------------------------

def test_4_bars_to_sessions_uses_day_offsets_not_fixed_stride():
    """A 60-bar and 105-bar session converts correctly, differs from 375-bar assumption.

    bars_to_sessions walks BACKWARD from the end of day_offsets (lookback semantics).
    """
    day_offsets = np.array([0, 60, 165], dtype=np.int64)  # 60-bar then 105-bar session
    # Layout: Session 0 at rows [0, 59], Session 1 at rows [60, 164], end at row 165
    # Backward lookback from row 164:
    #   105 bars: rows [60, 164] (all of session 1) -> 1 session
    #   106 bars: rows [59, 164] (crosses into session 0) -> 2 sessions

    assert bars_to_sessions(105, day_offsets) == 1, (
        "105 bars (entire session 1) should span exactly 1 session"
    )
    assert bars_to_sessions(106, day_offsets) == 2, (
        "106 bars should span 2 sessions (crossing backward into session 0)"
    )
    # Explicitly catch regression to a fixed bars-per-day constant
    assert bars_to_sessions(106, day_offsets) != math.ceil(106 / 375), (
        "bars_to_sessions must not assume a fixed 375-bar session stride"
    )


# ---------------------------------------------------------------------------
# Test 5: Purged K-fold and walk-forward embargo widths consistent
# ---------------------------------------------------------------------------

def test_5_purged_kfold_and_walkforward_widths_consistent():
    """WalkForwardSplitter's one-sided gap and PurgedKFold's right embargo both use
    `required_sessions()`; PurgedKFold's left purge uses `label_horizon` alone and is
    a DIFFERENT (smaller, here) width by design -- purging and embargo bound
    different dependencies and must not be merged (spec AMENDMENT 1 item 2,
    AMENDMENT 2 clause 6). We assert all of these facts, including that the left and
    right PurgedKFold widths are NOT equal.
    """
    components = EmbargoComponents(
        feature_lookback=2,
        label_horizon=3,
        holding_period=4,
        execution_horizon=1,
    )
    required = components.required_sessions()
    assert required == pytest.approx(10.0), "required_sessions should be 10"

    # WalkForwardSplitter: embargo_days=10 should NOT raise, and the gap should be 10
    trading_dates = _make_trading_dates(60)
    splitter = WalkForwardSplitter(
        train_years=0.05, test_years=0.02, step_years=0.02, embargo_days=10
    )
    splits = splitter.split(
        trading_dates,
        feature_lookback=2,
        label_horizon=3,
        holding_period=4,
        execution_horizon=1,
    )
    assert len(splits) > 0, "expected at least one split"
    for s in splits:
        gap = _split_gap_sessions(s, trading_dates)
        assert gap == 10, (
            f"walk-forward gap should be exactly 10 trading days, got {gap}"
        )

    # PurgedKFold: same components, label_horizon=3 on constructor.
    #
    # Use 40 sessions of 10 rows each (n_rows=400) with n_splits=4, so each test
    # block is 10 sessions (100 rows) wide: block boundaries land at rows
    # [0,100), [100,200), [200,300), [300,400). We inspect fold index 1
    # (test_start=100, test_end=200) because it has >= label_horizon rows of
    # train before it (no left clipping at row 0) and >= required_sessions worth
    # of rows after it (no right clipping at n_rows) -- an interior fold, so both
    # measured widths equal their theoretical value with no boundary truncation.
    session_bars = 10
    n_rows = 400
    day_offsets = np.arange(0, n_rows + session_bars, session_bars, dtype=np.int64)
    kfold = PurgedKFold(n_splits=4, label_horizon=3)
    # label_horizon kwarg to split() is left at its default (0.0), so per
    # PurgedKFold.split's documented resolution the constructor's label_horizon
    # (3) is reused as the embargo-sum's label_horizon term -- the same value
    # feeds both the left purge and (as part of the sum) the right embargo.
    folds = kfold.split(
        n_rows,
        day_offsets=day_offsets,
        feature_lookback=2,
        holding_period=4,
        execution_horizon=1,
    )
    assert len(folds) == 4, f"expected 4 folds, got {len(folds)}"
    fold = folds[1]

    # Left purge width: derives from label_horizon ALONE (bars/rows, applied
    # directly by PurgedKFold.split as `purge_start = test_start -
    # self.label_horizon`) -- NOT from required_sessions(). Arithmetic:
    # expected_left_rows = label_horizon = 3.
    expected_left_rows = 3
    left_width = _left_purge_width_rows(fold)
    assert left_width == expected_left_rows, (
        f"left purge should be exactly label_horizon={expected_left_rows} rows, "
        f"got {left_width}"
    )

    # Right embargo width: derives from required_sessions() -- the SUM of all
    # four components (feature_lookback=2 + label_horizon=3 + holding_period=4 +
    # execution_horizon=1 = 10 sessions), converted to rows via day_offsets.
    # Arithmetic: expected_right_rows = required * session_bars = 10 * 10 = 100.
    expected_right_rows = int(required) * session_bars
    assert expected_right_rows == 100
    right_width = _right_embargo_width_rows(fold, n_rows)
    assert right_width == expected_right_rows, (
        f"right embargo should be required_sessions()-derived "
        f"({expected_right_rows} rows), got {right_width}"
    )

    # The contract under test: left (label_horizon-derived) and right
    # (required_sessions()-derived) widths must NOT be equal here, because
    # purging and embargo bound different dependencies and must not be merged
    # (spec AMENDMENT 1 item 2, AMENDMENT 2 clause 6). 3 != 10, so this fixture
    # does not coincidentally make them equal.
    assert left_width != right_width, (
        f"PurgedKFold's left purge (label_horizon-derived) and right embargo "
        f"(required_sessions()-derived) must differ, both came out {left_width}"
    )


# ---------------------------------------------------------------------------
# Test 6: embargo_frac no longer shrinks embargo as sample grows
# ---------------------------------------------------------------------------

def test_6_embargo_frac_no_longer_drives_width():
    """Doubling the sample leaves the absolute embargo unchanged."""
    kfold = PurgedKFold(n_splits=5, label_horizon=0)

    # Same components, different n_rows. holding_period=7 is deliberately NOT
    # equal to math.ceil(0.01 * 500) == 5 or math.ceil(0.01 * 1000) == 10 -- a
    # value of 10 here would coincidentally collide with the old fraction-based
    # width at n_rows=1000 and make the regression-guard assertion below pass
    # even for an implementation that (wrongly) still used embargo_frac.
    folds_500 = kfold.split(
        500,
        feature_lookback=0,
        holding_period=7,
        execution_horizon=0,
    )
    folds_1000 = kfold.split(
        1000,
        feature_lookback=0,
        holding_period=7,
        execution_horizon=0,
    )

    # Extract right-side embargo widths for the first fold of each
    width_500 = _right_embargo_width_rows(folds_500[0], 500)
    width_1000 = _right_embargo_width_rows(folds_1000[0], 1000)

    assert width_500 == width_1000, (
        f"absolute embargo width should be identical, got {width_500} vs {width_1000}"
    )
    # Explicitly catch regression to fraction-based sizing
    assert width_500 != math.ceil(0.01 * 500), (
        "embargo width must not be derived from embargo_frac * n_rows"
    )
    assert width_1000 != math.ceil(0.01 * 1000), (
        "embargo width must not be derived from embargo_frac * n_rows"
    )


# ---------------------------------------------------------------------------
# Test 7: Property test over many random configurations
# ---------------------------------------------------------------------------

def test_7_property_disjoint_and_separated_over_random_configs():
    """Train/test disjoint and separated by at least required embargo, >=30 configs."""
    rng = np.random.default_rng(42)  # fixed documented seed per rule 9

    # Walk-forward property
    for _ in range(30):
        n_days = int(rng.integers(20, 80))
        trading_dates = _make_trading_dates(n_days)
        feature_lookback = float(rng.integers(0, 5))
        label_horizon = float(rng.integers(0, 5))
        holding_period = float(rng.integers(0, 5))
        execution_horizon = float(rng.integers(0, 3))
        components = EmbargoComponents(
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        required = components.required_sessions()
        embargo_days = int(math.ceil(required)) + int(rng.integers(0, 3))
        splitter = WalkForwardSplitter(
            train_years=0.05,
            test_years=0.02,
            step_years=0.02,
            embargo_days=embargo_days,
        )
        splits = splitter.split(
            trading_dates,
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        for s in splits:
            gap = _split_gap_sessions(s, trading_dates)
            assert gap >= required, (
                f"walk-forward gap {gap} < required {required} "
                f"for components {components}"
            )

    # PurgedKFold property
    for _ in range(30):
        n_rows = int(rng.integers(50, 300))
        n_splits = int(rng.integers(2, 6))
        label_horizon = float(rng.integers(0, 5))
        feature_lookback = float(rng.integers(0, 5))
        holding_period = float(rng.integers(0, 5))
        execution_horizon = float(rng.integers(0, 3))
        components = EmbargoComponents(
            feature_lookback=feature_lookback,
            label_horizon=label_horizon,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        required = components.required_sessions()

        # Build day_offsets with variable session lengths
        session_lengths = []
        remaining = n_rows
        while remaining > 0:
            # When remaining < 10, allow smaller session size; rng.integers requires low < high
            length = int(rng.integers(min(10, remaining), min(50, remaining + 1)))
            session_lengths.append(length)
            remaining -= length
        day_offsets = np.cumsum([0] + session_lengths, dtype=np.int64)

        kfold = PurgedKFold(n_splits=n_splits, label_horizon=label_horizon)
        folds = kfold.split(
            n_rows,
            day_offsets=day_offsets,
            feature_lookback=feature_lookback,
            holding_period=holding_period,
            execution_horizon=execution_horizon,
        )
        for fold in folds:
            # Disjointness
            assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0, (
                f"train/test overlap in fold {fold.id}"
            )
            # Separation: purge (left of test) and embargo (right of test) bound
            # DIFFERENT dependencies and use DIFFERENT widths -- they must not be
            # merged (spec AMENDMENT 1 item 2, AMENDMENT 2 clause 6).
            #
            # LEFT purge derives from label_horizon ALONE. PurgedKFold.split
            # applies it directly in row/bar units (`purge_start = test_start -
            # self.label_horizon`, independent of day_offsets and of the other
            # three components), so the expected row separation is simply
            # label_horizon itself.
            required_rows_left = label_horizon

            # RIGHT embargo derives from required_sessions() -- the sum of all
            # four components -- converted to rows via day_offsets, walking
            # FORWARD from the fold's own test_end (position-dependent: see
            # _sessions_to_rows docstring).
            test_start = int(np.min(fold.test_idx))
            test_end = int(np.max(fold.test_idx))
            required_rows_right = _sessions_to_rows(required, day_offsets, test_end + 1)
            for idx in fold.train_idx:
                idx = int(idx)
                if idx < test_start:
                    assert test_start - idx - 1 >= required_rows_left, (
                        f"train index {idx} too close to test start {test_start} "
                        f"(gap {test_start - idx - 1} < {required_rows_left})"
                    )
                elif idx > test_end:
                    assert idx - test_end - 1 >= required_rows_right, (
                        f"train index {idx} too close to test end {test_end} "
                        f"(gap {idx - test_end - 1} < {required_rows_right})"
                    )


# ---------------------------------------------------------------------------
# Test 8: Zero-lookback degenerates to current behaviour
# ---------------------------------------------------------------------------

def test_8_zero_lookback_degenerates_to_current_behaviour():
    """Calling split(trading_dates) with no kwargs matches hand-computed expected splits."""
    trading_dates = _make_trading_dates(30)
    splitter = WalkForwardSplitter(
        train_years=0.05, test_years=0.02, step_years=0.02, embargo_days=5
    )

    # No kwargs: all four components default to 0.0, guard never fires
    splits = splitter.split(trading_dates)

    # Hand-compute expected splits by replicating the exact documented windowing
    # arithmetic from `research/splits.py` verbatim (scheme="rolling", the
    # default): train_days/test_days/step_days = round(*_years * 252);
    # test_start_idx = train_end_idx + 1 + embargo_days; id is
    # f"{scheme}_{split_idx:03d}".
    train_days = round(0.05 * 252)  # 13
    test_days = round(0.02 * 252)  # 5
    step_days = round(0.02 * 252)  # 5
    embargo_days = 5
    n = len(trading_dates)

    expected = []
    cursor = 0
    split_idx = 0
    while True:
        train_end_idx = cursor + train_days - 1
        if train_end_idx >= n:
            break
        test_start_idx = train_end_idx + 1 + embargo_days
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= n:
            break
        expected.append(
            Split(
                id=f"rolling_{split_idx:03d}",
                train=(trading_dates[cursor], trading_dates[train_end_idx]),
                test=(trading_dates[test_start_idx], trading_dates[test_end_idx]),
                embargo_days=embargo_days,
            )
        )
        cursor += step_days
        split_idx += 1

    assert len(splits) == len(expected), (
        f"expected {len(expected)} splits, got {len(splits)}"
    )
    for actual, exp in zip(splits, expected):
        assert actual.id == exp.id, f"id mismatch: {actual.id} != {exp.id}"
        assert actual.train == exp.train, f"train mismatch: {actual.train} != {exp.train}"
        assert actual.test == exp.test, f"test mismatch: {actual.test} != {exp.test}"
        assert actual.embargo_days == exp.embargo_days, (
            f"embargo_days mismatch: {actual.embargo_days} != {exp.embargo_days}"
        )

    # Also assert no raise even with embargo_days=0
    splitter_zero = WalkForwardSplitter(
        train_years=0.05, test_years=0.02, step_years=0.02, embargo_days=0
    )
    splits_zero = splitter_zero.split(trading_dates)
    assert len(splits_zero) > 0, "expected splits with embargo_days=0 and no kwargs"
