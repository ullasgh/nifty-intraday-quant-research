"""Suite B (independent reading) for Spec G -- portfolio construction, from
`specs/portfolio_construction.md` ALONE. No implementation exists yet: every test in this
file is EXPECTED to fail (RED) today. The file itself must still COLLECT cleanly, so any
symbol that does not exist yet is imported lazily, inside the test function that needs it,
never at module scope -- a missing module then produces a normal per-test failure/error,
not a collection error for the whole file.

SPEC DEFECTS / ASSUMPTIONS MADE (mirrors the style of tests/test_causality_guards_a.py):

* The spec never names the weighting-scheme registry's module or the exact call signature
  of a scheme function. This suite assumes:

      from nifty_quant.backtest.schemes import SCHEME_REGISTRY  # dict[str, Callable]
      # ADJUDICATED by specs/portfolio_construction.md AMENDMENT 3 (2026-08-31): module
      # pinned as backtest.weighting / WEIGHT_SCHEME_REGISTRY; imports below alias it.
      # Obligation 2's tuple-return guess adjudicated to apply_weight_scheme (A3.4).
      # keys: "equal_weight", "inverse_vol", "z_weight", "z_over_vol", "rank_weight",
      #       "risk_parity", "covariance_aware"
      fn(signal, sigma, corr, mask, *, gross=1.0) -> np.ndarray
      fn(signal, sigma, corr, mask, *, gross=1.0, max_weight=<float>) -> (np.ndarray, bool)
      #   signal, sigma: shape (n,) float64
      #   mask: shape (n,) bool, True = tradable; a False entry (present-but-not-tradable,
      #     or altogether absent) MUST receive weight exactly 0
      #   returns sum(|weights|) == gross within float tolerance
      #   when max_weight is supplied, the return becomes a 2-tuple (weights, clip_bound)
      #     where clip_bound is True iff the max_weight clip actually changed the book --
      #     the spec's own obligation 2 wording ("respects max_weight AND reports when the
      #     clip binds") requires a report distinct from the weights themselves, and no
      #     other channel for that report is named, so a tuple return is this suite's
      #     concrete guess.

* The spec calls the fourth positional argument `corr` for ALL seven schemes uniformly, but
  `VolTargetSizer.to_shares` (G2, already implemented) uses `corr` for a SCALAR median
  pairwise correlation -- a different type entirely. For `covariance_aware` to be anything
  other than a renamed `inverse_vol` (obligation 6 requires it to DIFFER once the covariance
  has real off-diagonal structure), the G1 `corr` argument must be able to carry a full (n,
  n) covariance matrix, not a scalar. This suite assumes `corr` is an (n, n) float64
  covariance matrix (diagonal = per-name variance) for every scheme; the five schemes that
  do not need cross-sectional structure (equal_weight, inverse_vol, z_weight, z_over_vol,
  rank_weight) are assumed to ignore it. This inconsistency between G1's and G2's `corr` is
  itself flagged as a spec defect worth resolving before implementation, not silently
  reconciled here.

* The spec's G3 "run as a pre-registered experiment" and "promotion on RECENT years" are
  described only in prose, with no module or function name. This suite assumes:

      from nifty_quant.research.portfolio_comparison import (
          run_scheme_comparison,      # (*, close, day_offsets, contract, registry) -> result
                                       #   with .effective_n_trials: float, writing one
                                       #   TrialRecord per scheme to `registry`
          evaluate_scheme_promotion,   # (*, returns, day_offsets, recent_n_sessions) -> result
                                       #   with .promoted: bool, decided on the recent window
      )

  `ResearchContract.register_trial()` / `.check_trial_count()` and `TrialRegistry` already
  exist and are used directly (not assumed) below -- Phase G's own gap is that nothing wires
  a 7-scheme comparison THROUGH them yet.

* `BacktestResult` does not carry `sigma_portfolio_ann` / `vol_target_achieved` /
  `clip_binding` today (those fields live only on the per-call `SizingResult` inside
  `portfolio.py` and are never surfaced by `run_backtest`). Obligations 7/8 assume these
  become arrays on `BacktestResult`, one entry per panel row, NaN/False on rows without a
  sizing call -- accessed directly (no `hasattr` guard), so a missing attribute is a genuine
  `AttributeError` failure, not a manufactured stub.

CONCRETE, ALREADY-VERIFIED DEFECT (not a spec ambiguity; reported by the lead, reproduced
below, not merely hypothesised): `run_tilt` (`src/nifty_quant/research/tilt.py:636`) writes
`chash = contract.contract_hash` and uses that SAME value for BOTH `TrialRecord.config_hash`
and `.contract_hash`. `ResearchContract.contract_hash` hashes only its own six sections plus
seed -- never a single `TiltConfig` field -- so `config_hash` is blind to every parameter a
scheme comparison would actually vary (smoothing, tilt, rebalance_every, ...). Combined with
`registry.py`'s `UNIQUE(config_hash, split_id)` + `INSERT OR IGNORE`, and `run_tilt` hardcoding
`split_id="full"` unconditionally, this SILENTLY DROPS every run after the first one recorded
against a given contract: a seven-scheme comparison recorded through this path would keep only
the first scheme per split and lose the rest with no error, understating the trial count that
feeds `effective_n_trials` -- exactly the multiple-testing failure mode Spec G exists to guard
against. `test_g3_...` above tests the NOT-YET-BUILT Phase G comparison runner; the tests below
pin this ALREADY-BUILT, ALREADY-BROKEN path (`run_tilt` + `TrialRegistry`) directly, and are
expected to fail against the CURRENT code, not merely against an absent module.
"""

from __future__ import annotations

import calendar
import dataclasses
import datetime as dt

import numpy as np
import pytest
from pydantic import BaseModel

from nifty_quant.backtest.engine import BacktestConfig, run_backtest
from nifty_quant.backtest.metrics import effective_n_trials
from nifty_quant.backtest.portfolio import VolTargetSizer
from nifty_quant.data.panel import Panel
from nifty_quant.features.core import SIGMA_FLOOR
from nifty_quant.research.registry import TrialRegistry
from nifty_quant.research.tilt import TiltConfig, run_tilt
from nifty_quant.strategy.base import (
    DataRequest,
    MarketView,
    PortfolioState,
    Strategy,
    TargetPortfolio,
)
from tests.contract_fixtures import minimal_contract

# ---------------------------------------------------------------------------------
# Minimal self-contained fixtures (no reliance on any other test file's fixtures).
# ---------------------------------------------------------------------------------


def _session_ts(start: dt.date, n_days: int, bars_per_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Epoch-second UTC int64 timestamps for `n_days` sessions of `bars_per_day` 1-min bars
    each, starting 09:15 Asia/Kolkata (03:45 UTC, no DST in India). Builds day_offsets
    explicitly from THIS fixture's own bars_per_day rather than assuming any fixed
    repo-wide session length elsewhere (rule 5)."""
    ts_chunks = []
    day_offsets = [0]
    day = start
    for _ in range(n_days):
        session_start_utc = calendar.timegm((day.year, day.month, day.day, 3, 45, 0))
        ts_chunks.append(session_start_utc + 60 * np.arange(bars_per_day, dtype=np.int64))
        day_offsets.append(day_offsets[-1] + bars_per_day)
        day += dt.timedelta(days=1)
    ts = np.concatenate(ts_chunks).astype(np.int64)
    return ts, np.array(day_offsets, dtype=np.int32)


def _make_panel(*, n_symbols: int, n_days: int, bars_per_day: int, seed: int) -> Panel:
    """A small dense panel with positive, finite prices on every row (no absent/halted
    names) -- deliberately simple, since obligations 7/8 are about sizer wiring, not about
    present/tradable edge cases (those are obligation 3's job)."""
    rng = np.random.default_rng(seed)
    ts, day_offsets = _session_ts(dt.date(2026, 1, 5), n_days, bars_per_day)
    n_rows = ts.shape[0]
    symbols = tuple(f"S{i}" for i in range(n_symbols))
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=(n_rows, n_symbols)), axis=0)
    close = np.abs(close) + 10.0
    open_ = close.copy()
    high = close * 1.001
    low = close * 0.999
    volume = np.full((n_rows, n_symbols), 100_000.0)
    fields = {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }
    dates = np.array(
        [
            dt.date(2026, 1, 5) + dt.timedelta(days=day_ix)
            for day_ix in range(n_days)
            for _ in range(bars_per_day)
        ],
        dtype=object,
    )
    return Panel(fields=fields, symbols=symbols, ts=ts, day_offsets=day_offsets, dates=dates)


class _TrivialStrategy(Strategy):
    """Emits the same fixed weight vector at every decision row (decision_times=None ->
    one decision per row after the first, per engine.py). `weights` must sum in absolute
    value to <= 1.0 (TargetPortfolio.validate's default gross cap)."""

    name = "trivial_fixed_weight"

    class Params(BaseModel):
        weights: tuple[float, ...]

    def data_request(self) -> DataRequest:
        return DataRequest(decision_times=None)

    def precompute(self, panel: Panel) -> dict:
        return {}

    def on_decision(self, view: MarketView, signals, state: PortfolioState):
        return TargetPortfolio(weights=np.array(self.params.weights, dtype=np.float64))


def _cov(sigma: np.ndarray) -> np.ndarray:
    """A diagonal covariance matrix from a per-name sigma vector -- "zero correlation" /
    "diagonal covariance" as used by obligations 5 and 6, and as the default input
    everywhere else in this suite where cross-sectional structure is not itself under
    test."""
    return np.diag(np.asarray(sigma, dtype=np.float64) ** 2)


# ---------------------------------------------------------------------------------
# G1 -- the weighting-scheme registry (obligations 1-6)
# ---------------------------------------------------------------------------------


def test_obligation_1_all_schemes_sum_to_declared_gross():
    # catches: a scheme whose weights do not sum (in absolute value) to the declared gross
    # -- e.g. a scheme that forgets to renormalize after masking or clipping.
    from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY

    n = 8
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    sigma = np.abs(rng.normal(0.2, 0.05, size=n)) + 0.1
    mask = np.ones(n, dtype=bool)
    gross = 1.0
    corr = _cov(sigma)

    assert set(SCHEME_REGISTRY) == {
        "equal_weight",
        "inverse_vol",
        "z_weight",
        "z_over_vol",
        "rank_weight",
        "risk_parity",
        "covariance_aware",
    }, "registry must declare exactly the seven schemes named in the spec, no more, no fewer"

    for name, fn in SCHEME_REGISTRY.items():
        w = np.asarray(fn(signal, sigma, corr, mask, gross=gross), dtype=np.float64)
        assert w.shape == (n,), f"{name}: weight vector has the wrong shape {w.shape}"
        assert np.isclose(np.sum(np.abs(w)), gross, atol=1e-8), (
            f"{name}: sum(|weights|)={np.sum(np.abs(w))} != declared gross {gross}"
        )


def test_obligation_2_max_weight_clip_and_report_for_every_scheme():
    # catches: a scheme that (a) exceeds max_weight after clipping, or (b) silently clips
    # without reporting it -- both are real defects the spec calls out ("respects max_weight
    # AND reports when the clip binds" are two independent requirements).
    from nifty_quant.backtest.weighting import (
        WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY,
    )  # noqa: I001 -- alias order mirrors the other obligation tests' single-alias import
    from nifty_quant.backtest.weighting import (
        apply_weight_scheme,
    )

    n = 6
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n)
    sigma = np.abs(rng.normal(0.2, 0.05, size=n)) + 0.1
    mask = np.ones(n, dtype=bool)
    corr = _cov(sigma)

    for name, fn in SCHEME_REGISTRY.items():
        raw = np.asarray(fn(signal, sigma, corr, mask, gross=1.0), dtype=np.float64)
        max_abs_raw = float(np.max(np.abs(raw)))
        assert max_abs_raw > 0.0, f"{name}: degenerate all-zero raw weights"

        # Derived from the scheme's OWN unclipped output -- not a hand-picked constant
        # (rule 8): half the largest raw weight is guaranteed to force a clip.
        binding_max_weight = max_abs_raw / 2.0

        # AMENDMENT 3 (A3.4/A3.7): the clip and its report live on apply_weight_scheme's
        # SchemeResult, not a tuple return from the bare scheme. Assertion substance
        # unchanged: clip respected, binding reported True iff it moved the book.
        result = apply_weight_scheme(
            name,
            signal=signal,
            sigma=sigma,
            corr=corr,
            tradable=mask,
            gross=1.0,
            max_weight=binding_max_weight,
        )
        w = np.asarray(result.weights, dtype=np.float64)
        max_abs_w = float(np.max(np.abs(w)))
        assert max_abs_w <= binding_max_weight + 1e-9, (
            f"{name}: max_weight={binding_max_weight} violated post-clip, max(|w|)={max_abs_w}"
        )
        assert result.clip_binding is True, (
            f"{name}: max raw weight {max_abs_raw} > max_weight {binding_max_weight}, so the "
            "clip demonstrably moved the book, but clip_binding was not reported True"
        )

        # A max_weight that cannot possibly bind (looser than gross itself) must report False.
        result_loose = apply_weight_scheme(
            name,
            signal=signal,
            sigma=sigma,
            corr=corr,
            tradable=mask,
            gross=1.0,
            max_weight=1.0 + 1e-6,
        )
        assert result_loose.clip_binding is False, (
            f"{name}: reported clip_binding=True even though max_weight=1.0+eps cannot bind "
            "against a gross-1.0 book -- a report that is always True carries no information"
        )


def test_obligation_3_no_weight_leaks_to_present_but_not_tradable():
    # catches: a scheme that assigns nonzero weight to a masked-out (present-but-not-tradable
    # or absent) name -- rule 7's present/tradable distinction leaking into a weight.
    from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY

    n = 6
    rng = np.random.default_rng(2)
    signal = rng.normal(size=n)
    sigma = np.abs(rng.normal(0.2, 0.05, size=n)) + 0.1
    corr = _cov(sigma)
    mask = np.array([True, True, True, False, False, False])

    for name, fn in SCHEME_REGISTRY.items():
        w = np.asarray(fn(signal, sigma, corr, mask, gross=1.0), dtype=np.float64)
        assert np.all(w[~mask] == 0.0), f"{name}: assigned nonzero weight to a masked-out name"
        # And the tradable set alone must still carry the full declared gross -- otherwise
        # masking silently shrank the book instead of reallocating within the tradable set.
        assert np.isclose(np.sum(np.abs(w)), 1.0, atol=1e-8), (
            f"{name}: masking a subset dropped total gross instead of reallocating it"
        )


def test_obligation_4_inverse_vol_floor_prevents_inf_not_some_other_clamp():
    # catches: inverse_vol dividing by a raw zero sigma directly (producing inf), OR
    # "fixing" that with an ad hoc clamp other than sigma_risk's derived SIGMA_FLOOR.
    from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY

    fn = SCHEME_REGISTRY["inverse_vol"]
    n = 4
    signal = np.array([1.0, 1.0, 1.0, 1.0])
    sigma_raw = np.array([0.0, np.nan, 0.05, 0.2])
    mask = np.ones(n, dtype=bool)
    corr = np.eye(n)  # ignored by inverse_vol per this suite's assumed interface

    w_raw = np.asarray(fn(signal, sigma_raw, corr, mask, gross=1.0), dtype=np.float64)
    assert np.all(np.isfinite(w_raw)), f"inverse_vol produced a non-finite weight: {w_raw}"
    assert w_raw[1] == 0.0, "inverse_vol assigned nonzero weight to a NaN-sigma name"

    # The floor -- not an unrelated special case -- must be what the scheme applies:
    # substituting SIGMA_FLOOR explicitly for the zero entry (leaving the NaN entry alone)
    # must reproduce the SAME weight vector the scheme derived internally from the raw
    # zero. If the scheme instead special-cased "sigma==0 -> weight 0" (rather than
    # flooring), this would fail because SIGMA_FLOOR > 0 yields a nonzero weight there.
    sigma_explicit_floor = sigma_raw.copy()
    sigma_explicit_floor[0] = SIGMA_FLOOR
    w_explicit_floor = np.asarray(
        fn(signal, sigma_explicit_floor, corr, mask, gross=1.0), dtype=np.float64
    )
    assert np.allclose(w_raw, w_explicit_floor, atol=1e-12), (
        f"inverse_vol(raw zero sigma)={w_raw} != inverse_vol(SIGMA_FLOOR-substituted)="
        f"{w_explicit_floor} -- the zero-sigma weight does not appear to come from "
        "sigma_risk's derived floor"
    )


def test_obligation_5_risk_parity_equal_vol_zero_corr_is_exactly_equal_weight():
    # catches: risk_parity not degenerating EXACTLY to equal weight on the one input where
    # "equal risk contribution" and "equal weight" are mathematically identical.
    from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY

    n = 5
    signal = np.ones(n)  # signal is irrelevant to risk_parity by construction
    sigma = np.full(n, 0.2)
    corr = _cov(sigma)  # zero off-diagonal correlation
    mask = np.ones(n, dtype=bool)

    raw = SCHEME_REGISTRY["risk_parity"](signal, sigma, corr, mask, gross=1.0)
    w = np.asarray(raw, dtype=np.float64)
    expected = np.full(n, 1.0 / n)
    assert np.allclose(w, expected, atol=1e-10), f"risk_parity={w} != equal weight {expected}"


def test_obligation_6_covariance_aware_diagonal_cov_is_exactly_inverse_vol():
    # catches: covariance_aware not degenerating EXACTLY to inverse_vol when the covariance
    # it was given carries no off-diagonal information (the one case where "using an
    # estimated covariance" and "using only the diagonal" must coincide).
    from nifty_quant.backtest.weighting import WEIGHT_SCHEME_REGISTRY as SCHEME_REGISTRY

    n = 5
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    sigma = np.abs(rng.normal(0.2, 0.05, size=n)) + 0.1
    corr = _cov(sigma)  # diagonal covariance
    mask = np.ones(n, dtype=bool)

    w_cov = np.asarray(
        SCHEME_REGISTRY["covariance_aware"](signal, sigma, corr, mask, gross=1.0), dtype=np.float64
    )
    w_inv = np.asarray(
        SCHEME_REGISTRY["inverse_vol"](signal, sigma, corr, mask, gross=1.0), dtype=np.float64
    )
    assert np.allclose(w_cov, w_inv, atol=1e-10), f"covariance_aware={w_cov} != inverse_vol={w_inv}"


# ---------------------------------------------------------------------------------
# G2 -- VolTargetSizer wired into the engine (obligations 7-8)
# ---------------------------------------------------------------------------------


def test_obligation_7_vol_target_sizer_reachable_and_sigma_populated():
    # catches: the engine still not supplying sigma/corr to a VolTargetSizer (today,
    # substituting it raises ValueError("sigma is required")), or a wiring that runs but
    # never surfaces sigma_portfolio_ann on the result.
    panel = _make_panel(n_symbols=4, n_days=2, bars_per_day=6, seed=10)
    strategy = _TrivialStrategy(params=_TrivialStrategy.Params(weights=(0.3, 0.3, 0.2, 0.2)))
    config = BacktestConfig(
        sizer=VolTargetSizer(target_vol_ann=0.15, gross=1.0, max_weight=0.5),
        capital=1e6,
    )
    contract = minimal_contract(seed=10)

    result = run_backtest(strategy, panel, config, contract=contract)

    sigma_arr = np.asarray(result.sigma_portfolio_ann, dtype=np.float64)
    assert sigma_arr.shape[0] == panel.ts.shape[0], (
        "sigma_portfolio_ann must carry one entry per panel row"
    )
    assert np.any(np.isfinite(sigma_arr)), (
        "sigma_portfolio_ann is never populated on any row -- the sizer never ran, or its "
        "diagnostics were dropped on the way out of run_backtest"
    )


def test_obligation_8_clip_that_breaks_target_is_reflected_through_the_engine():
    # catches: vol_target_achieved computed from PRE-clip weights (so it always reports
    # "target met" even though the clip moved the book) -- asserted end-to-end through
    # run_backtest, not on VolTargetSizer in isolation (that isolation coverage already
    # exists in tests/test_vol_target_{a,b}.py; G2's own obligation is that wiring
    # preserves it).
    panel = _make_panel(n_symbols=4, n_days=2, bars_per_day=6, seed=11)
    # A strategy target far more concentrated (0.85 in one name) than max_weight (0.10)
    # below, so the sizer's per-name clip step is forced to bind on every decision row.
    strategy = _TrivialStrategy(params=_TrivialStrategy.Params(weights=(0.85, 0.05, 0.05, 0.05)))
    target_vol_ann = 0.60
    config = BacktestConfig(
        sizer=VolTargetSizer(target_vol_ann=target_vol_ann, gross=1.0, max_weight=0.10),
        capital=1e6,
    )
    contract = minimal_contract(seed=11)

    result = run_backtest(strategy, panel, config, contract=contract)

    clip_arr = np.asarray(result.clip_binding, dtype=bool)
    achieved_arr = np.asarray(result.vol_target_achieved, dtype=np.float64)
    bound_rows = np.flatnonzero(clip_arr)
    assert bound_rows.size > 0, (
        "clip_binding never True despite max_weight=0.10 far tighter than the strategy's "
        "own 0.85-concentrated target -- the clip should have been forced to bind"
    )
    # A clip that actually moved the book generically breaks the declared target -- a
    # RELATIVE property (rule 8), not a hand-picked tolerance: reporting the target
    # itself, unchanged, on a row where the clip demonstrably bound is the exact silent-void
    # defect this phase exists to catch.
    for row in bound_rows:
        assert not np.isclose(achieved_arr[row], target_vol_ann, rtol=1e-6, atol=1e-9), (
            f"row {row}: clip_binding is True but vol_target_achieved ({achieved_arr[row]}) "
            f"still equals the declared target_vol_ann ({target_vol_ann}) -- looks like it "
            "was computed from PRE-clip weights"
        )


# ---------------------------------------------------------------------------------
# G3 -- the comparison as a pre-registered, multiple-testing-aware experiment
# (obligations 9-10, plus the trial-accounting property the spec's prose calls out)
# ---------------------------------------------------------------------------------


def test_obligation_9_effective_n_trials_is_below_seven_on_correlated_schemes():
    # catches: effective_n_trials overstating independence across seven heavily correlated
    # schemes (the spec names inverse_vol/z_over_vol and rank_weight/z_weight as sharing
    # structure) -- this test exercises the ALREADY-IMPLEMENTED generic
    # metrics.effective_n_trials on a fixture standing in for seven correlated trial-return
    # series, since no scheme implementation exists yet to generate real ones.
    rng = np.random.default_rng(4)
    t, n_trials = 500, 7
    common = rng.normal(0.0, 0.01, size=t)
    trial_returns = np.empty((t, n_trials))
    for i in range(n_trials):
        idiosyncratic = rng.normal(0.0, 0.005, size=t)
        trial_returns[:, i] = 0.7 * common + 0.3 * idiosyncratic

    n_eff = effective_n_trials(trial_returns)
    assert 1.0 < n_eff < 7.0, (
        f"effective_n_trials={n_eff} is not strictly between 1 and 7 on correlated inputs "
        "(the anti-overstatement property this obligation exists to check)"
    )


def test_g3_comparison_registers_one_trial_per_scheme_under_the_declared_contract():
    # catches: a seven-scheme comparison that evaluates each scheme WITHOUT registering it
    # as a trial -- the spec's own words: "a comparison that does not register its trials IS
    # the failure mode". Also catches effective_n_trials being reported as a flat 7 instead
    # of measured from the schemes' own correlated returns.
    from nifty_quant.research.portfolio_comparison import run_scheme_comparison

    n_planned = 7
    contract = minimal_contract(
        seed=20,
        validation={"scheme": "test", "holdout_intent": "never", "n_planned_trials": n_planned},
    )
    registry = TrialRegistry(_tmp_registry_path())

    rng = np.random.default_rng(21)
    t, n_symbols = 300, 4
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=(t, n_symbols)), axis=0)
    day_offsets = np.arange(0, t + 1, 1, dtype=np.int64)

    comparison = run_scheme_comparison(
        close=close, day_offsets=day_offsets, contract=contract, registry=registry
    )

    recorded = registry.all()
    assert len(recorded) == n_planned, (
        f"expected exactly one TrialRecord per scheme ({n_planned}), got {len(recorded)} -- "
        "an unregistered scheme comparison is the exact failure mode the spec names"
    )
    assert all(rec.contract_hash == contract.contract_hash for rec in recorded), (
        "every scheme's trial must be traceable to the ONE pre-registered contract for this "
        "run, not left blank or stamped with an ad hoc per-scheme hash"
    )
    # The declared n_planned_trials must actually bound this run: exactly at the limit must
    # not raise, one past it must (ResearchContract.check_trial_count already implements
    # this; what's missing is a Phase G runner that produces exactly n_planned trials).
    contract.check_trial_count(n_planned)
    with pytest.raises(ValueError):
        contract.check_trial_count(n_planned + 1)

    assert comparison.effective_n_trials < float(n_planned), (
        f"effective_n_trials={comparison.effective_n_trials} was not measured below the raw "
        "planned-trial count on schemes built from shared, correlated inputs"
    )


def test_obligation_10_promotion_is_decided_on_the_recent_window_not_pooled():
    # catches: a promotion decision computed from pooled history, so a scheme whose edge has
    # decayed away recently is still promoted because stale early years carry the pooled
    # Sharpe -- mirrors the repo's own recorded H2 pattern (sign-stable pooled edge, killed
    # once the recent-years-only gate is applied).
    from nifty_quant.research.portfolio_comparison import evaluate_scheme_promotion

    rng = np.random.default_rng(5)
    n_recent_sessions = 2 * 252
    old_years = rng.normal(0.0015, 0.01, size=6 * 252)
    recent_years = rng.normal(-0.0015, 0.01, size=n_recent_sessions)
    pooled_returns = np.concatenate([old_years, recent_years])
    day_offsets = np.arange(0, pooled_returns.shape[0] + 1, 1, dtype=np.int64)

    assert np.mean(pooled_returns) > 0.0, "fixture error: pooled mean must be positive"
    assert np.mean(recent_years) < 0.0, "fixture error: recent mean must be negative"

    result = evaluate_scheme_promotion(
        returns=pooled_returns, day_offsets=day_offsets, recent_n_sessions=n_recent_sessions
    )
    assert result.promoted is False, (
        "pooled returns carry a positive edge purely from stale years, but the most recent "
        "window is a clean sign flip -- promotion must refuse on the recent window "
        "regardless of the pooled statistics"
    )


def _tmp_registry_path():
    """A fresh on-disk sqlite path for TrialRegistry, isolated per test process."""
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="nq_portfolio_construction_b_")) / "trials.db"


# ---------------------------------------------------------------------------------
# run_tilt / TrialRegistry accounting -- a CONCRETE, already-reproducible defect
# (config_hash blind to every TiltConfig field; silent row-drop on collision).
# Fixture pattern follows the repo's own convention for a sparse-checkpoint tilt panel
# (as in tests/test_tilt_a.py's `_build_panel`/`_cycling_aggressive_panel`), written here
# independently and self-contained -- not imported from that file.
# ---------------------------------------------------------------------------------

_TILT_SYMBOLS = tuple(f"SYM{i}" for i in range(5))


def _tilt_ts_for(date: dt.date, hhmm: str) -> int:
    """Epoch-second UTC timestamp for `date` at exact IST label `hhmm`."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    session_start_utc = calendar.timegm((date.year, date.month, date.day, 3, 45, 0))
    minute_offset = (hour - 9) * 60 + (minute - 15)
    return session_start_utc + minute_offset * 60


def _tilt_flat_row(base: float = 100.0) -> np.ndarray:
    return np.array([base + i * 0.01 for i in range(len(_TILT_SYMBOLS))], dtype=np.float64)


def _tilt_panel(dates: list[dt.date], loser_of_day: list[int]) -> Panel:
    """One session per date, FOUR labelled bars per session (09:16/09:20 entry
    candidates, 15:15/15:20 exit candidates) so a TiltConfig perturbation of
    entry_hhmm/exit_hhmm resolves against a real bar rather than finding none. For
    date i, `loser_of_day[i]` is the single overnight loser (~3.0 below the ~100
    baseline), exactly as in the repo's own tilt fixtures."""
    sessions_sorted = sorted(zip(dates, loser_of_day), key=lambda item: item[0])
    ts_list: list[int] = []
    price_rows: list[np.ndarray] = []
    day_offsets = [0]
    row = 0
    for date, loser in sessions_sorted:
        entry_a = _tilt_flat_row(100.0)
        entry_a[loser] -= 3.0
        entry_b = entry_a - 0.01
        exit_a = entry_a + 0.05
        exit_b = exit_a + 0.01
        for hhmm, price_row in (
            ("09:16", entry_a),
            ("09:20", entry_b),
            ("15:15", exit_a),
            ("15:20", exit_b),
        ):
            ts_list.append(_tilt_ts_for(date, hhmm))
            price_rows.append(price_row)
            row += 1
        day_offsets.append(row)
    ts = np.array(ts_list, dtype=np.int64)
    price_arr = np.stack(price_rows).astype(np.float32)
    volume_arr = np.full(price_arr.shape, 1_000_000.0, dtype=np.float32)
    dates_arr = np.array([d for d, _ in sessions_sorted], dtype=object)
    return Panel(
        fields={"open": price_arr, "close": price_arr.copy(), "volume": volume_arr},
        symbols=_TILT_SYMBOLS,
        ts=ts,
        day_offsets=np.array(day_offsets, dtype=np.int32),
        dates=dates_arr,
    )


def _tilt_fixture() -> tuple[Panel, list[dt.date]]:
    dates = [dt.date(2022, 1, 3) + dt.timedelta(days=i) for i in range(20)]
    loser_of_day = [i % len(_TILT_SYMBOLS) for i in range(20)]
    return _tilt_panel(dates, loser_of_day), dates


def _base_tilt_config(dates: list[dt.date]) -> TiltConfig:
    return TiltConfig(start=dates[0], end=dates[-1], tilt="aggressive", smoothing=0.10)


def _recorded_config_hash(panel: Panel, config: TiltConfig, contract) -> str:
    """Run once against a FRESH, isolated registry and return the single config_hash
    written -- isolating one run's recorded hash from any other run's collisions."""
    registry = TrialRegistry(_tmp_registry_path())
    run_tilt(panel, config, contract=contract, registry=registry)
    recorded = registry.all()
    assert len(recorded) == 1, "expected exactly one row from a single isolated run_tilt call"
    return recorded[0].config_hash


# Every TiltConfig field except start/end (perturbed separately below, since a valid
# perturbation there depends on the fixture's own date range) mapped to a value that
# differs from `_base_tilt_config`'s. If a new TiltConfig field is ever added without a
# corresponding entry here, `test_tilt_param_perturbation_coverage_is_complete` fails --
# so a newly added parameter cannot silently escape this suite (per the coordinator's
# explicit requirement), instead of the coverage gap passing silently.
_TILT_PARAM_PERTURBATIONS: dict[str, object] = {
    "entry_hhmm": "09:20",
    "exit_hhmm": "15:15",
    "capital": 2_000_000.0,
    "tilt": "mild",
    "smoothing": 1.0,
    "rebalance_every": 2,
    "universe": "nifty50",
    "continuous_only": True,
    "seed": 1,
}


def test_tilt_param_perturbation_coverage_is_complete():
    # catches: a new TiltConfig field being added while this suite's coverage silently
    # stays stale -- the exact "escape hatch" the coordinator flagged as unacceptable.
    all_fields = {f.name for f in dataclasses.fields(TiltConfig)}
    covered = set(_TILT_PARAM_PERTURBATIONS) | {"start", "end"}
    assert covered == all_fields, (
        f"TiltConfig fields not covered by a hash-binding perturbation test: "
        f"{all_fields - covered}"
    )


@pytest.mark.parametrize("field_name", sorted(_TILT_PARAM_PERTURBATIONS))
def test_config_hash_changes_when_any_tilt_parameter_changes(field_name: str):
    # catches: run_tilt's recorded config_hash being blind to a TiltConfig field -- TODAY
    # config_hash is `contract.contract_hash` verbatim, which does not read `config` at
    # all, so this fails for EVERY field, not just `smoothing` (the lead's example: 0.10
    # vs 1.0 is the difference between +3.22 and -9.41 bps/day, hashed identically).
    panel, dates = _tilt_fixture()
    contract = minimal_contract(seed=100)
    config_a = _base_tilt_config(dates)
    perturbed_value = _TILT_PARAM_PERTURBATIONS[field_name]
    assert getattr(config_a, field_name) != perturbed_value, "perturbation must actually differ"
    config_b = dataclasses.replace(config_a, **{field_name: perturbed_value})

    hash_a = _recorded_config_hash(panel, config_a, contract)
    hash_b = _recorded_config_hash(panel, config_b, contract)
    assert hash_a != hash_b, (
        f"changing TiltConfig.{field_name} from {getattr(config_a, field_name)!r} to "
        f"{perturbed_value!r} left the recorded config_hash unchanged ({hash_a!r}) -- the "
        "hash binds none of this run's actual configuration"
    )


def test_config_hash_changes_when_end_date_changes():
    # catches: the same config_hash blindness for the one field (`end`) that cannot share
    # the generic perturbation dict, since a valid value depends on the fixture's own dates.
    panel, dates = _tilt_fixture()
    contract = minimal_contract(seed=101)
    config_a = _base_tilt_config(dates)
    config_b = dataclasses.replace(config_a, end=dates[-2])

    hash_a = _recorded_config_hash(panel, config_a, contract)
    hash_b = _recorded_config_hash(panel, config_b, contract)
    assert hash_a != hash_b, (
        f"changing TiltConfig.end from {config_a.end} to {config_b.end} left the recorded "
        f"config_hash unchanged ({hash_a!r})"
    )


def test_n_distinct_tilt_configs_recorded_against_one_split_yield_n_rows():
    # catches: the silent-drop defect itself -- registry.py's UNIQUE(config_hash, split_id)
    # + INSERT OR IGNORE means N distinct scheme configurations recorded through run_tilt
    # (which hardcodes split_id="full" and a config_hash blind to `config`) collapse into
    # ONE row with no error and no trace. THIS TEST FAILS TODAY against the existing code
    # path -- that is the point: a Phase G sweep recorded this way would silently keep only
    # the first scheme per split and lose the rest.
    panel, dates = _tilt_fixture()
    contract = minimal_contract(seed=102)
    registry = TrialRegistry(_tmp_registry_path())
    smoothing_values = (0.10, 0.50, 1.0)

    for smoothing in smoothing_values:
        config = dataclasses.replace(_base_tilt_config(dates), smoothing=smoothing)
        run_tilt(panel, config, contract=contract, registry=registry)

    recorded = registry.all()
    assert len(recorded) == len(smoothing_values), (
        f"recorded {len(recorded)} row(s) for {len(smoothing_values)} distinct scheme "
        "configurations against the same split -- some were silently dropped by the "
        "UNIQUE(config_hash, split_id) + INSERT OR IGNORE path"
    )


def test_registered_trial_count_feeding_effective_n_trials_matches_configs_run():
    # catches: the same silent-drop defect surfacing through the ACTUAL consumption path --
    # `TrialRegistry.n_trials()` is what would feed `effective_n_trials`/deflated Sharpe's
    # trial count, so an undercount here means the multiple-testing correction comes out
    # too permissive, exactly how a spurious winner gets manufactured from a seven-scheme
    # comparison. Distinct from the row-count test above: this pins the downstream QUANTITY
    # actually consumed, not merely the raw table contents.
    panel, dates = _tilt_fixture()
    contract = minimal_contract(seed=103)
    registry = TrialRegistry(_tmp_registry_path())
    rebalance_values = (1, 2, 3)

    for rebalance_every in rebalance_values:
        config = dataclasses.replace(_base_tilt_config(dates), rebalance_every=rebalance_every)
        run_tilt(panel, config, contract=contract, registry=registry)

    n_recorded = registry.n_trials(strategy="tilt")
    assert n_recorded == len(rebalance_values), (
        f"registry.n_trials(strategy='tilt')={n_recorded} but {len(rebalance_values)} distinct "
        "configurations were actually run -- the trial count feeding effective_n_trials "
        "understates how many configurations were actually compared"
    )
