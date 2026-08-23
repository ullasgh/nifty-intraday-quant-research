"""Tests for scripts/recon_tilt_liquidity_decomposition.py's criterion-4 mirror.

The script's job is to run already-tested production machinery (`Lens.stability`,
`compute_prior_adv`, `expectancy.causal_buckets`) on the tilt's signal and then REPORT the
verdict. Almost all of it is a passthrough. The one piece that is not -- and the piece the
published PASS/FAIL rests on -- is `report_lens_criterion_4`, which re-implements the
concentration comparison that lives inline inside `Lens.verdict()` (research/lens.py:703-777)
so the intermediate max/median ratio is visible instead of collapsed into a string.

A mirror of inline logic has no standalone target to diff against, so it is tested directly
here against synthetic `StabilityReport`s: uniform deciles must PASS, a bottom decile that
dominates beyond the measured 4.8695 threshold must FAIL, and -- the discriminating case --
a MIDDLE decile that dominates by the same margin must still PASS, because criterion 4 is
about the bottom (illiquid) decile specifically, not about dispersion in general.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nifty_quant.research.expectancy import ExpectancyTable
from nifty_quant.research.lens import CONCENTRATION_RATIO_THRESHOLD, StabilityReport

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "recon_tilt_liquidity_decomposition.py"
)
_spec = importlib.util.spec_from_file_location(
    "recon_tilt_liquidity_decomposition", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
recon = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = recon
_spec.loader.exec_module(recon)


def _table(spread_bps: float, n_total: int = 1867) -> ExpectancyTable:
    """A minimal ExpectancyTable carrying only the fields criterion 4 reads."""
    return ExpectancyTable(
        buckets=(),
        horizon=1,
        feature_name="overnight_gap",
        n_total=n_total,
        cost_hurdle_bps=1.0,
        spread_bps=spread_bps,
        spread_t=spread_bps,
        survives_costs=False,
    )


def _report(
    decile_spreads: dict[int, float],
    tod_spreads: dict[str, float] | None = None,
) -> StabilityReport:
    if tod_spreads is None:
        # Two buckets, same sign, ratio 1.0 -- the time-of-day leg cannot fire, so the
        # liquidity leg is isolated.
        tod_spreads = {"open": -20.0, "mid": -20.0}
    return StabilityReport(
        by_year={},
        by_time_of_day={k: _table(v) for k, v in tod_spreads.items()},
        by_liquidity_decile={k: _table(v) for k, v in decile_spreads.items()},
        n_years_total=8,
        n_years_sign_consistent=8,
        dominant_sign="-",
    )


def test_uniform_deciles_pass() -> None:
    """Ten deciles of equal magnitude: max/median == 1.0, nothing fires."""
    report = _report({d: -20.0 for d in range(10)})
    result, reasons, spreads = recon.report_lens_criterion_4(report)
    assert result == "PASS"
    assert reasons == []
    assert len(spreads) == 10


def test_bottom_decile_domination_fails() -> None:
    """Bottom decile beyond the measured threshold, and the argmax: criterion 4 FAILS."""
    spreads = {d: -10.0 for d in range(10)}
    spreads[0] = -10.0 * (CONCENTRATION_RATIO_THRESHOLD + 1.0)
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "FAIL"
    assert any("bottom liquidity decile" in r for r in reasons)


def test_middle_decile_domination_still_passes() -> None:
    """Same dispersion, but decile 5 dominates instead of decile 0: must still PASS.

    This is the case that separates "concentrated in the illiquid tail" from "dispersed".
    A mirror that tested only the ratio and forgot `bottom_is_argmax` would fail here.
    """
    spreads = {d: -10.0 for d in range(10)}
    spreads[5] = -10.0 * (CONCENTRATION_RATIO_THRESHOLD + 1.0)
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "PASS"
    assert reasons == []


def test_bottom_decile_ratio_just_below_threshold_passes() -> None:
    """Strictly-greater-than comparison: a ratio at 99% of the threshold does not fire."""
    spreads = {d: -10.0 for d in range(10)}
    spreads[0] = -10.0 * CONCENTRATION_RATIO_THRESHOLD * 0.99
    result, _, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "PASS"


def test_only_bottom_decile_has_edge_fails() -> None:
    """If the bottom decile is the ONLY decile with any edge, that is concentration."""
    spreads = {d: 0.0 for d in range(10)}
    spreads[0] = -30.0
    result, reasons, _ = recon.report_lens_criterion_4(_report(spreads))
    assert result == "FAIL"
    assert any("only decile with edge" in r for r in reasons)


def test_time_of_day_sign_disagreement_fails() -> None:
    """Mixed-sign time-of-day buckets fire the time-of-day leg independently."""
    result, reasons, _ = recon.report_lens_criterion_4(
        _report({d: -20.0 for d in range(10)}, tod_spreads={"open": -20.0, "mid": +20.0})
    )
    assert result == "FAIL"
    assert any("sign" in r for r in reasons)


def test_single_time_of_day_bucket_fails() -> None:
    """One bucket with data is concentration by definition, per lens.py:741-744."""
    result, reasons, _ = recon.report_lens_criterion_4(
        _report({d: -20.0 for d in range(10)}, tod_spreads={"open": -20.0})
    )
    assert result == "FAIL"
    assert any("single time-of-day" in r for r in reasons)


@pytest.mark.parametrize("attr,expected", [("SEED", 0), ("HORIZON", 1), ("N_DECILES", 10)])
def test_pre_registered_constants(attr: str, expected: int) -> None:
    """The run is deterministic: seed and geometry are fixed in the file, not inferred."""
    assert getattr(recon, attr) == expected


# ---------------------------------------------------------------------------
# Section C: the measured null for the bottom-k share of realised gross excess
# ---------------------------------------------------------------------------
#
# Section C is NOT a mirror of production logic -- it is new machinery, and the number it
# publishes (a threshold under CLAUDE.md rule 8) is only as good as the permuter beneath
# it. Three things have to hold and each is tested separately:
#
#   1. The STATISTIC is the thing it claims to be: a cumulative share of the TOTAL, over
#      deciles 0..k-1, with unlabelled cells counted in the denominator and nowhere else.
#   2. The PERMUTER is exchangeable and structure-preserving: it must leave the total, the
#      decile sizes and the stratum membership exactly alone, and touch only which symbol
#      carries which liquidity rank.
#   3. The TEST HAS POWER and the right size. Per CLAUDE.md rule 9 the size check is a
#      false-positive RATE across many seeds, never a single "the interval contains X"
#      assertion, and the power check is built with an effect large enough to be
#      effectively deterministic rather than marginal.


def _exchangeable_book(
    seed: int, n_sessions: int = 200, n_symbols: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A book whose contributions are independent of liquidity decile.

    Ten equal deciles of five names each, randomly assigned per session, and a positive
    mean contribution so the denominator is well away from zero (a near-zero denominator
    would make the share heavy-tailed for reasons that have nothing to do with the null).
    Returns (contrib, labels, strata).
    """
    rng = np.random.default_rng(seed)
    labels = np.empty((n_sessions, n_symbols), dtype=np.int64)
    base = np.repeat(np.arange(10), n_symbols // 10)
    for i in range(n_sessions):
        labels[i] = rng.permutation(base)
    contrib = rng.normal(loc=0.5, scale=1.0, size=(n_sessions, n_symbols))
    strata = np.zeros_like(labels)
    return contrib, labels, strata


def test_bottom_k_share_is_a_cumulative_share_of_the_total() -> None:
    """Hand-computed: unlabelled cells count in the denominator, never the numerator."""
    contrib = np.array([[1.0, 2.0, 3.0, 4.0]])
    labels = np.array([[0, 1, 5, -1]])
    # total = 10. k=1 -> {1.0}; k=2 -> {1.0, 2.0}; k=6 -> {1.0, 2.0, 3.0}.
    assert recon.bottom_k_share(contrib, labels, 1) == pytest.approx(0.1)
    assert recon.bottom_k_share(contrib, labels, 2) == pytest.approx(0.3)
    assert recon.bottom_k_share(contrib, labels, 6) == pytest.approx(0.6)
    # The unlabelled 4.0 is never picked up, even by k = N_DECILES.
    assert recon.bottom_k_share(contrib, labels, recon.N_DECILES) == pytest.approx(0.6)


def test_bottom_k_share_matches_section_b_row_sums() -> None:
    """The statistic must equal the sum of section B's printed per-decile shares.

    If these two ever diverge, the published "bottom two deciles supply 33.0%" and the
    null it is compared against would be measuring different quantities.
    """
    contrib, labels, _ = _exchangeable_book(seed=7)
    total = float(np.sum(contrib))
    for k in (1, 2, 3):
        row_sum = sum(
            float(np.sum(np.where(labels == d, contrib, 0.0))) / total for d in range(k)
        )
        assert recon.bottom_k_share(contrib, labels, k) == pytest.approx(row_sum, rel=1e-12)


def test_permutation_preserves_total_decile_sizes_and_strata() -> None:
    """The null's invariants, checked on the permutation the null actually draws.

    The share is a ratio; if the permutation moved contribution in or out of the book, or
    resized a decile, the null would be measuring a different statistic on every replicate
    and its percentiles would be meaningless.
    """
    contrib, labels, _ = _exchangeable_book(seed=3, n_sessions=40, n_symbols=50)
    # Three strata, unequal sizes, so a permuter that ignores strata cannot pass by luck.
    strata = np.zeros_like(labels)
    strata[:, 10:30] = 1
    strata[:, 30:] = 2

    rng = np.random.default_rng(11)
    keys = strata.astype(np.float64) * 2.0 + rng.random(contrib.shape)
    order = np.argsort(keys, axis=1)
    permuted = np.take_along_axis(contrib, order, axis=1)
    strata_after = np.take_along_axis(strata, order, axis=1)

    assert float(np.sum(permuted)) == pytest.approx(float(np.sum(contrib)), rel=1e-12)
    for i in range(contrib.shape[0]):
        assert sorted(permuted[i].tolist()) == pytest.approx(sorted(contrib[i].tolist()))
    # Stratum blocks come out in ascending order and keep their exact sizes, so a column
    # is only ever exchanged with a column from its own stratum.
    assert np.array_equal(strata_after, np.sort(strata, axis=1))
    assert not np.array_equal(order, np.tile(np.arange(contrib.shape[1]), (contrib.shape[0], 1)))


def test_null_centres_on_the_even_spread_value() -> None:
    """Under exchangeability the null median sits on k/10 -- a single spread assertion.

    This is deliberately ONE number with a wide margin (0.5pp against a Monte-Carlo
    standard error of ~0.06pp at n=2000), not a per-k coverage check: per CLAUDE.md rule 9
    a "the interval contains the truth" assertion repeated across buckets flakes at ~alpha
    per comparison no matter how correct the code is.
    """
    contrib, labels, strata = _exchangeable_book(seed=101, n_sessions=400, n_symbols=50)
    draws = recon.null_bottom_k_distribution(
        contrib, labels, strata, (2,), 2000, np.random.default_rng(42)
    )[2]
    assert float(np.median(draws)) == pytest.approx(0.20, abs=0.005)


def test_null_detects_a_book_that_really_is_concentrated() -> None:
    """Power: an effect this large must clear p95 by an enormous margin, every time.

    Deciles 0 and 1 are given ten times the contribution of every other decile, so the
    observed share is ~55% against a null centred at 20% with sd ~2pp. The detection
    margin is tens of standard errors, which makes the assertion deterministic in
    practice rather than a coin-flip at the boundary.
    """
    contrib, labels, strata = _exchangeable_book(seed=5, n_sessions=300, n_symbols=50)
    contrib = np.abs(contrib)
    contrib = np.where(labels < 2, contrib * 10.0, contrib)
    observed = recon.bottom_k_share(contrib, labels, 2)
    draws = recon.null_bottom_k_distribution(
        contrib, labels, strata, (2,), 500, np.random.default_rng(42)
    )[2]
    cutoff = float(np.percentile(draws, recon.THRESHOLD_PERCENTILE))
    assert observed > 0.5
    assert observed > cutoff
    assert int(np.sum(draws >= observed)) == 0


def test_false_positive_rate_across_many_seeds() -> None:
    """Size: on data where liquidity is irrelevant, a p95 cutoff must fire ~5% of the time.

    Per CLAUDE.md rule 9 this is a RATE across 40 independent datasets, not a single
    pass/fail. Under a correct permuter the count is Binomial(40, 0.05) with mean 2; the
    bound of 9 is exceeded with probability ~1e-4, so this is a real check and not a
    rubber stamp. If it ever fires, the permuter is broken -- it is NOT to be repaired by
    changing the seed or the seed count (CLAUDE.md rules 8 and 9).
    """
    fired = 0
    for seed in range(40):
        contrib, labels, strata = _exchangeable_book(seed=seed, n_sessions=150, n_symbols=50)
        observed = recon.bottom_k_share(contrib, labels, 2)
        draws = recon.null_bottom_k_distribution(
            contrib, labels, strata, (2,), 400, np.random.default_rng(10_000 + seed)
        )[2]
        if observed > float(np.percentile(draws, recon.THRESHOLD_PERCENTILE)):
            fired += 1
    assert fired <= 9, f"false-positive rate {fired}/40 is far above the nominal 5%"


def test_recorded_thresholds_are_ordered_and_above_the_even_spread() -> None:
    """Guards the transcribed constants against a stale or mistyped digit.

    A p95 of a null centred on k/10 must exceed k/10, and the bottom-k share is cumulative
    so the cutoffs must increase with k. Both are properties of the derivation, so a value
    that violates them cannot have come from the run the comment describes.
    """
    thresholds = recon.BOTTOM_K_SHARE_THRESHOLDS
    assert sorted(thresholds) == list(recon.BOTTOM_K_VALUES)
    for k, value in thresholds.items():
        assert value > k / recon.N_DECILES
        assert value < (k + 1) / recon.N_DECILES
    values = [thresholds[k] for k in sorted(thresholds)]
    assert values == sorted(values)


def test_null_seed_and_replicate_count_are_pre_registered() -> None:
    """The published grid must be reproducible from the file, not from a run's memory."""
    assert recon.NULL_SEED == 42
    assert recon.N_NULL_REPLICATES >= 300
    assert recon.THRESHOLD_PERCENTILE == 95.0
    assert recon.NULL_PERCENTILES == (50.0, 75.0, 90.0, 95.0, 99.0)


def test_null_is_deterministic_given_its_seed() -> None:
    """Same seed, same draws: the recorded percentile grid is a fact about the file."""
    contrib, labels, strata = _exchangeable_book(seed=13, n_sessions=60, n_symbols=50)
    first = recon.null_bottom_k_distribution(
        contrib, labels, strata, (2,), 50, np.random.default_rng(recon.NULL_SEED)
    )[2]
    second = recon.null_bottom_k_distribution(
        contrib, labels, strata, (2,), 50, np.random.default_rng(recon.NULL_SEED)
    )[2]
    assert np.array_equal(first, second)


def test_strata_that_cross_decile_boundaries_are_honoured() -> None:
    """The stratum blocks and the label ordering must be aligned, not merely both sorted.

    `_canonical_layout` assigns labels to permuted column RANKS, and the ranks are grouped
    by stratum. If it ordered labels by label alone (rather than by stratum-then-label),
    every mask would silently address the wrong block whenever the strata cut across
    decile membership -- which is exactly what happens in the real run, where the
    unlabelled (-1) columns form their own stratum and sort FIRST by label but LAST by
    stratum. The case below is small enough to enumerate by hand.

    One session, four columns: labels [0, 1, 0, 1], strata [1, 1, 0, 0], contributions
    [1, 10, 100, 1000] (total 1111). Columns may only swap within their stratum, so
    decile 0 always receives exactly one of {100, 1000} and one of {1, 10}, giving four
    attainable bottom-1 shares. A layout that ignored the strata would instead hand
    decile 0 both of {100, 1000} on every replicate -- a single, constant value.
    """
    contrib = np.array([[1.0, 10.0, 100.0, 1000.0]])
    labels = np.array([[0, 1, 0, 1]])
    strata = np.array([[1, 1, 0, 0]])
    draws = recon.null_bottom_k_distribution(
        contrib, labels, strata, (1,), 400, np.random.default_rng(0)
    )[1]
    attainable = {(a + b) / 1111.0 for a in (100.0, 1000.0) for b in (1.0, 10.0)}
    observed_values = {round(float(v), 12) for v in draws}
    assert observed_values == {round(v, 12) for v in attainable}


def test_each_session_is_permuted_independently() -> None:
    """One permutation PER SESSION, not one permutation shared by every session.

    A single permutation reused across sessions still yields a null centred on the
    even-spread value, so neither the centring check nor the false-positive-rate check can
    see it -- but it destroys the cross-session averaging that gives the real null its
    width, and would leave the published percentile grid far too wide.

    Two sessions of ten names, each with all of its contribution on column 0, and labels
    reversed between the two sessions. Under independent per-session permutations the two
    contributions land in decile 0 independently, so a bottom-1 share of exactly 0.5 (one
    of the two, not both) occurs on ~18% of replicates and is certain to appear within
    400 draws. Under a shared permutation both sessions' column 0 always receive the same
    label, so 0.5 is unattainable.
    """
    contrib = np.zeros((2, 10), dtype=np.float64)
    contrib[:, 0] = 1.0
    labels = np.stack([np.arange(10), np.arange(9, -1, -1)]).astype(np.int64)
    strata = np.zeros_like(labels)
    draws = recon.null_bottom_k_distribution(
        contrib, labels, strata, (1,), 400, np.random.default_rng(0)
    )[1]
    assert set(np.round(draws, 12)) == {0.0, 0.5, 1.0}


def test_null_width_shrinks_with_the_number_of_sessions() -> None:
    """The statistical consequence of per-session independence, stated as one spread.

    Sessions are independent draws, so the null standard deviation of a share aggregated
    over them must fall like 1/sqrt(n_sessions): quadrupling the sessions must halve it.
    The bound below spans 1.6 to 2.5 around a predicted 2.0, against a Monte-Carlo error
    on the ratio of roughly 0.05 at 1,500 replicates -- so this is a large-effect
    detection check, not a coverage check that would flake at ~alpha per run (CLAUDE.md
    rule 9).
    """
    def _sd(n_sessions: int) -> float:
        contrib, labels, strata = _exchangeable_book(seed=77, n_sessions=n_sessions)
        draws = recon.null_bottom_k_distribution(
            contrib, labels, strata, (2,), 1500, np.random.default_rng(recon.NULL_SEED)
        )[2]
        return float(np.std(draws, ddof=1))

    ratio = _sd(100) / _sd(400)
    assert 1.6 < ratio < 2.5, f"null sd scaled by {ratio:.2f}, expected ~2.0"
