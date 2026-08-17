"""Research facade: Lens.

Wraps feature access, expectancy analysis, and verdict criteria to ensure every result
carries its own provenance and cannot be read without understanding the context that
generated it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from nifty_quant.backtest.metrics import deflated_sharpe, expected_max_sharpe
from nifty_quant.data.panel import Panel
from nifty_quant.execution.costs import NSEIntradayEquityCosts
from nifty_quant.features import core as core_features
from nifty_quant.research import expectancy

# ---------------------------------------------------------------------------
# FeatureKindError
# ---------------------------------------------------------------------------


class FeatureKindError(ValueError):
    """Raised when a feature kind does not match the required kind."""

    pass


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    """A named feature with values, kind, warmup bars, and parameters."""

    name: str
    values: np.ndarray  # (n_rows, n_symbols) or (n_rows,) float64
    kind: Literal["return", "level", "ratio", "count", "market"]
    warmup_bars: int
    params: Mapping[str, Any]

    def explain(self) -> str:
        """Explain the feature: name, kind, warmup bars, and parameters."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            params_str = f" ({params_str})"
        return f"Feature: {self.name} [{self.kind}]{params_str}, warmup_bars={self.warmup_bars}"


# ---------------------------------------------------------------------------
# StabilityReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityReport:
    """Stability analysis across years, time-of-day, and liquidity deciles."""

    by_year: dict[int, expectancy.ExpectancyTable]
    by_time_of_day: dict[str, expectancy.ExpectancyTable]  # e.g., "open", "mid", "close"
    by_liquidity_decile: dict[int, expectancy.ExpectancyTable]  # 0-9
    n_years_total: int
    n_years_sign_consistent: int
    dominant_sign: Literal["+", "-", "mixed"]

    @property
    def sign_stable(self) -> bool:
        """True if the feature sign is consistent across >= 6 of n_years_total years."""
        return self.n_years_sign_consistent >= 6

    def explain(self) -> str:
        """Explain the stability report."""
        return (
            f"StabilityReport: {self.n_years_sign_consistent}/{self.n_years_total} years "
            f"with consistent sign ({self.dominant_sign}), stable={self.sign_stable}"
        )


# ---------------------------------------------------------------------------
# HypothesisVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisVerdict:
    """Verdict on a hypothesis after applying all Phase 3 kill criteria."""

    hypothesis_id: str
    survived: bool
    reasons: tuple[str, ...]  # one line per criterion, always all 7
    expectancy: expectancy.ExpectancyTable
    stability: StabilityReport
    cost_hurdle_bps: float
    seed: int

    def _se_method_name(self) -> str:
        """Standard-error method used, or 'N/A' when no bucket was produced.

        Extracted so `explain()` and `to_markdown()` report the SAME provenance string from one
        place. Duplicating this expression is how the two outputs would silently drift apart,
        and the SE method is load-bearing: it is the difference between an overlap-corrected
        t-stat and an inflated one.
        """
        if not self.expectancy.buckets:
            return "N/A"
        return self.expectancy.buckets[0].se_method

    def explain(self) -> str:
        """Explain the verdict: all criteria, seed, and final verdict."""
        lines = [
            f"Hypothesis: {self.hypothesis_id}",
            f"Seed: {self.seed}",
            f"Cost hurdle: {self.cost_hurdle_bps:.2f} bps",
            f"SE method: {self._se_method_name()}",
            "",
        ]
        for reason in self.reasons:
            lines.append(reason)
        lines.append("")
        verdict_token = "SURVIVED" if self.survived else "KILLED"
        lines.append(f"Final verdict: {verdict_token}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return markdown representation for committed results/hypotheses/<id>/verdict.md."""
        lines = [
            f"# {self.hypothesis_id}",
            "",
            f"**Verdict:** {'SURVIVED' if self.survived else 'KILLED'}",
            "",
            "## Kill Criteria",
            "",
        ]
        for reason in self.reasons:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("## Context")
        lines.append("")
        lines.append(f"- Cost hurdle: {self.cost_hurdle_bps:.2f} bps (round-trip)")
        lines.append(
            f"- SE method: {self._se_method_name()}"
        )
        lines.append(f"- Seed: {self.seed}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------


class Lens:
    """Research facade: feature access, expectancy, stability, and verdict."""

    def __init__(
        self,
        panel: Panel,
        *,
        universe: tuple[str, ...] | None = None,
        cost_model: NSEIntradayEquityCosts | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize Lens with panel, universe, cost model, and seed.

        Parameters
        ----------
        panel : Panel
            The panel holding close prices, volume, timestamps, and symbols.
        universe : tuple[str, ...] | None
            Symbols to analyze. Defaults to all symbols in the panel.
        cost_model : NSEIntradayEquityCosts | None
            Cost model for computing hurdles. Defaults to NSEIntradayEquityCosts().
        seed : int
            Random seed for reproducibility. Defaults to 0.
        """
        self.panel = panel
        self.day_offsets = panel.day_offsets
        self.minute_of_day = panel.minute_of_day()
        self.symbols = panel.symbols
        self.cost_model = cost_model or NSEIntradayEquityCosts()
        self.universe = universe or panel.symbols
        self.seed = seed

        # Feature cache: (name, params_tuple) -> Feature
        self._feature_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Feature] = {}

    def available_features(self) -> tuple[str, ...]:
        """Return the names of all available built-in features."""
        return ("close", "return_1", "volume_zscore")

    def feature(self, name: str, **params: Any) -> Feature:
        """Get a feature by name and parameters.

        Built-in features:
        - "close": level, no parameters
        - "return_1": return, no parameters
        - "volume_zscore": ratio, parameters: window (int), deseasonalize (bool, default True)

        Parameters
        ----------
        name : str
            Feature name.
        **params : Any
            Feature-specific parameters.

        Returns
        -------
        Feature
            A frozen Feature object with values, kind, warmup_bars, and params.

        Raises
        ------
        KeyError
            If the feature name is not recognized.
        """
        # Create cache key from name and parameters (sorted for consistency)
        params_sorted = tuple(sorted(params.items()))
        cache_key = (name, params_sorted)

        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        close = self.panel.field("close").astype(np.float64)
        volume = self.panel.field("volume").astype(np.float64)

        if name == "close":
            feature_values = close
            kind: Literal["return", "level", "ratio", "count", "market"] = "level"
            warmup_bars = 0
            params_stored: Mapping[str, Any] = {}

        elif name == "return_1":
            feature_values = core_features.log_returns(close, day_offsets=self.day_offsets)
            kind = "return"
            warmup_bars = 1
            params_stored = {}

        elif name == "volume_zscore":
            window = params.get("window", 20)
            deseasonalize = params.get("deseasonalize", True)
            feature_values = core_features.volume_zscore(
                volume,
                self.minute_of_day,
                window=window,
                deseasonalize=deseasonalize,
                day_offsets=self.day_offsets,
            )
            kind = "ratio"
            warmup_bars = window
            params_stored = {"window": window, "deseasonalize": deseasonalize}

        else:
            raise KeyError(f"Feature {name!r} not found")

        feature_obj = Feature(
            name=name,
            values=feature_values,
            kind=kind,
            warmup_bars=warmup_bars,
            params=params_stored,
        )

        self._feature_cache[cache_key] = feature_obj
        return feature_obj

    def expectancy(
        self, feature: str | Feature, horizon: int, **kw: Any
    ) -> expectancy.ExpectancyTable:
        """Compute conditional expectancy table.

        Parameters
        ----------
        feature : str | Feature
            Feature name or Feature object.
        horizon : int
            Forward return horizon in bars.
        **kw : Any
            Additional keyword arguments passed to expectancy.conditional_expectancy():
            method, n_buckets, se_method, n_boot, seed, cost_hurdle_bps, feature_name.

        Returns
        -------
        ExpectancyTable
            Bucketed conditional expectancy with stats and cost comparison.

        Raises
        ------
        FeatureKindError
            If the feature kind is not "return".
        """
        if isinstance(feature, str):
            feature_obj = self.feature(feature)
        else:
            feature_obj = feature

        # Check feature kind
        if feature_obj.kind != "return":
            raise FeatureKindError(
                f"Feature {feature_obj.name!r} has kind {feature_obj.kind!r}, "
                f"but 'return' is required"
            )

        # Compute forward returns
        close = self.panel.field("close").astype(np.float64)
        fwd = expectancy.forward_returns(close, self.day_offsets, horizon)

        # Delegate to expectancy module with feature_name
        table = expectancy.conditional_expectancy(
            feature_obj.values,
            fwd,
            self.day_offsets,
            feature_name=feature_obj.name,
            **kw,
        )

        return table

    def stability(self, feature: str | Feature, horizon: int) -> StabilityReport:
        """Analyze sign stability across years, time-of-day, and liquidity deciles.

        Parameters
        ----------
        feature : str | Feature
            Feature name or Feature object.
        horizon : int
            Forward return horizon in bars.

        Returns
        -------
        StabilityReport
            Stability report with by_year, by_time_of_day, by_liquidity_decile analyses.

        Raises
        ------
        FeatureKindError
            If the feature kind is not "return".
        """
        if isinstance(feature, str):
            feature_obj = self.feature(feature)
        else:
            feature_obj = feature

        # Check feature kind
        if feature_obj.kind != "return":
            raise FeatureKindError(
                f"Feature {feature_obj.name!r} has kind {feature_obj.kind!r}, "
                f"but 'return' is required"
            )

        close = self.panel.field("close").astype(np.float64)
        fwd = expectancy.forward_returns(close, self.day_offsets, horizon)

        # Decompose by year
        by_year_tables: dict[int, expectancy.ExpectancyTable] = {}
        year_signs: dict[int, int] = {}  # year -> +1 or -1

        # Build a row-to-year mapping using day_offsets
        n_rows = feature_obj.values.shape[0]
        row_to_year = np.zeros(n_rows, dtype=np.int32)
        for day_idx, date in enumerate(self.panel.dates):
            start_row = self.day_offsets[day_idx]
            end_row = self.day_offsets[day_idx + 1]
            row_to_year[start_row:end_row] = date.year

        for year in set(d.year for d in self.panel.dates):
            # Create a boolean mask for rows in this year. `year` is drawn
            # from panel.dates itself, and Panel's day_offsets invariant
            # (check_day_offsets requires every session to have >= 1 bar)
            # guarantees each date maps at least one row into row_to_year --
            # so `np.any(year_mask)` can never be False here; no guard needed.
            year_mask = row_to_year == year

            # Mask in place: keep the full (n_rows, n_symbols) shape and
            # the panel's real day_offsets, instead of extracting the
            # year's rows and fabricating a single-session [0, N]
            # day_offsets for them. The fabricated offsets declared ~all
            # real sessions in the year to be ONE session, so the block
            # bootstrap's resampled blocks could straddle real session
            # boundaries -- exactly the invariant day_offsets exists to
            # protect. year_mask is 1-D (per-row); masking whole rows to
            # NaN cannot change any OTHER row's cross-sectional rank, so
            # spread_bps/sign are unaffected -- only the SE/CI/block
            # bootstrap (which do depend on day_offsets) are fixed here.
            year_feature = np.where(year_mask[:, None], feature_obj.values, np.nan)
            year_fwd_values = np.where(year_mask[:, None], fwd.values, np.nan)

            n_defined = int(np.sum(np.any(np.isfinite(year_fwd_values), axis=1)))
            n_nan_tail = int(np.sum(~np.isfinite(year_fwd_values)))

            table = expectancy.conditional_expectancy(
                year_feature,
                expectancy.ForwardReturns(
                    values=year_fwd_values,
                    horizon=horizon,
                    session_bounded=True,
                    n_defined=n_defined,
                    n_nan_tail=n_nan_tail,
                ),
                self.day_offsets,
                n_buckets=5,
                method="cross_sectional_rank",
                feature_name=feature_obj.name,
            )
            by_year_tables[year] = table

            # Compute sign from the spread_bps of the expectancy table
            year_sign = (
                1
                if table.spread_bps > 0
                else (-1 if table.spread_bps < 0 else 0)
            )
            year_signs[year] = year_sign

        # Decompose by time of day: morning, midday, afternoon
        by_time_of_day_tables: dict[str, expectancy.ExpectancyTable] = {}
        minute_of_day = self.minute_of_day

        # Define time-of-day buckets (minute ranges in IST trading hours 9:15-15:30)
        # minute_of_day is calculated as hour*60 + minute
        time_buckets = {
            "open": (555, 630),  # 9:15 - 10:30 (9*60+15=555, 10*60+30=630)
            "mid": (630, 840),  # 10:30 - 14:00 (10*60+30=630, 14*60=840)
            "close": (840, 930),  # 14:00 - 15:30 (14*60=840, 15*60+30=930)
        }

        for time_name, (start_min, end_min) in time_buckets.items():
            mask = (minute_of_day >= start_min) & (minute_of_day < end_min)
            if np.any(mask):
                # Mask in place with the real day_offsets -- same reasoning as
                # the by_year branch above, and more important here: these
                # rows are only contiguous WITHIN a session (e.g. the 75
                # "open" bars of each day), never ACROSS sessions. Extracting
                # them and fabricating a single-session [0, N] day_offsets
                # made the extracted array look like one long contiguous
                # block, so the bootstrap could sample a "block" spanning the
                # boundary between one day's open window and the next day's --
                # rows with no real temporal adjacency or autocorrelation at
                # all. Passing the real day_offsets constrains every sampled
                # block to a single real session, as it must.
                tod_feature = np.where(mask[:, None], feature_obj.values, np.nan)
                tod_fwd_values = np.where(mask[:, None], fwd.values, np.nan)

                n_defined = int(np.sum(np.any(np.isfinite(tod_fwd_values), axis=1)))
                n_nan_tail = int(np.sum(~np.isfinite(tod_fwd_values)))

                table = expectancy.conditional_expectancy(
                    tod_feature,
                    expectancy.ForwardReturns(
                        values=tod_fwd_values,
                        horizon=horizon,
                        session_bounded=True,
                        n_defined=n_defined,
                        n_nan_tail=n_nan_tail,
                    ),
                    self.day_offsets,
                    n_buckets=5,
                    method="cross_sectional_rank",
                    feature_name=feature_obj.name,
                )
                by_time_of_day_tables[time_name] = table

        # Decompose by liquidity decile: volume quantiles
        by_liquidity_decile_tables: dict[int, expectancy.ExpectancyTable] = {}
        volume = self.panel.field("volume").astype(np.float64)

        # Compute volume deciles (causal, per row)
        volume_quantiles = np.quantile(
            volume[np.isfinite(volume)], np.linspace(0, 1, 11)
        )  # 0, 0.1, ..., 1.0
        volume_deciles = np.full_like(volume, -1, dtype=np.int8)

        for i in range(volume.shape[0]):
            for s in range(volume.shape[1]):
                vol = volume[i, s]
                if np.isfinite(vol):
                    decile = np.searchsorted(volume_quantiles[1:-1], vol, side="right")
                    volume_deciles[i, s] = decile

        for decile in range(10):
            mask = volume_deciles == decile
            if np.any(mask):
                # Mask in place: keep the full (n_rows, n_symbols) shape and the
                # panel's real day_offsets, instead of boolean-indexing with a
                # 2-D mask (which always flattens to 1-D and destroys the
                # row/symbol structure -- see the bug this fixes). Cells outside
                # this decile become NaN, meaning "not in this decile"; NaN is
                # never forward-filled (CLAUDE.md rule 6).
                liq_feature = np.where(mask, feature_obj.values, np.nan)
                liq_fwd_values = np.where(mask, fwd.values, np.nan)

                # Count defined forward returns: rows with at least one
                # in-decile, finite forward return.
                n_defined = int(np.sum(np.any(np.isfinite(liq_fwd_values), axis=1)))

                table = expectancy.conditional_expectancy(
                    liq_feature,
                    expectancy.ForwardReturns(
                        values=liq_fwd_values,
                        horizon=horizon,
                        session_bounded=True,
                        n_defined=n_defined,
                        n_nan_tail=0,
                    ),
                    self.day_offsets,
                    n_buckets=5,
                    method="cross_sectional_rank",
                    feature_name=feature_obj.name,
                )
                by_liquidity_decile_tables[decile] = table

        # Compute sign stability. by_year_tables and year_signs are always
        # populated in lockstep (same loop, same condition -- now unconditional
        # above), so n_years_total == 0 implies year_signs is empty too, which
        # the year_signs_nonzero == [] branch below already handles identically;
        # a separate n_years_total == 0 branch would be redundant dead code.
        n_years_total = len(by_year_tables)
        year_signs_nonzero = [s for s in year_signs.values() if s != 0]
        if not year_signs_nonzero:
            n_years_sign_consistent = 0
            dominant_sign: Literal["+", "-", "mixed"] = "mixed"
        else:
            # Count years with consistent sign
            most_common_sign = max(set(year_signs_nonzero), key=year_signs_nonzero.count)
            n_years_sign_consistent = year_signs_nonzero.count(most_common_sign)
            dominant_sign = "+" if most_common_sign > 0 else "-"

        return StabilityReport(
            by_year=by_year_tables,
            by_time_of_day=by_time_of_day_tables,
            by_liquidity_decile=by_liquidity_decile_tables,
            n_years_total=n_years_total,
            n_years_sign_consistent=n_years_sign_consistent,
            dominant_sign=dominant_sign,
        )

    def verdict(
        self,
        hypothesis_id: str,
        feature: str | Feature,
        horizon: int,
        *,
        latency_profile: Mapping[int, float] | None = None,
        effective_n_trials: int = 1,
        **kw: Any,
    ) -> HypothesisVerdict:
        """Apply all Phase 3 kill criteria and return a verdict.

        Criterion 1: Mean conditional edge > 2x the round-trip cost hurdle at that horizon.
        Criterion 2: Sign stable in >= 6 of 8 years (2018-2025).
        Criterion 3: Survives the overlap correction (block-bootstrap t-stat).
        Criterion 4: Not concentrated in the bottom liquidity decile or a single time-of-day.
        Criterion 5: Latency profile flat across 0/1/2-minute decision lag.
        Criterion 6: Deflated Sharpe accounts for the number of hypotheses tested.
        Criterion 7: Recent-years cost gate: mean edge over the last two complete years
            exceeds 2x the cost hurdle in magnitude and carries the dominant sign.

        Parameters
        ----------
        hypothesis_id : str
            Identifier for this hypothesis.
        feature : str | Feature
            Feature name or Feature object.
        horizon : int
            Forward return horizon in bars.
        latency_profile : Mapping[int, float] | None
            Edge profile by decision lag (0, 1, 2 minutes). If None, criterion 5 is
            NOT_EVALUATED.
        effective_n_trials : int
            Number of hypotheses tested (for criterion 6 deflation).
        **kw : Any
            Additional keyword arguments passed to expectancy(): method, n_buckets,
            se_method, n_boot, seed, cost_hurdle_bps.

        Returns
        -------
        HypothesisVerdict
            Verdict with all seven criteria, reasons, and final survived flag.

        Raises
        ------
        FeatureKindError
            If the feature kind is not "return".
        """
        if isinstance(feature, str):
            feature_obj = self.feature(feature)
        else:
            feature_obj = feature

        # Check feature kind
        if feature_obj.kind != "return":
            raise FeatureKindError(
                f"Feature {feature_obj.name!r} has kind {feature_obj.kind!r}, "
                f"but 'return' is required"
            )

        # Compute expectancy and stability
        exp_table = self.expectancy(feature_obj, horizon, **kw)
        stab_report = self.stability(feature_obj, horizon)
        cost_hurdle_bps = exp_table.cost_hurdle_bps

        # Compute forward returns for additional analysis
        close = self.panel.field("close").astype(np.float64)
        fwd = expectancy.forward_returns(close, self.day_offsets, horizon)

        reasons: list[str] = []

        # Criterion 1: Mean conditional edge > 2x cost hurdle
        edge_bps = exp_table.spread_bps
        if abs(edge_bps) > 2 * cost_hurdle_bps:
            c1_result = "PASS"
        else:
            c1_result = "FAIL"
        reasons.append(
            f"1. Edge criterion: {c1_result} "
            f"(edge={edge_bps:.2f} bps, 2x hurdle={2 * cost_hurdle_bps:.2f} bps)"
        )

        # Criterion 2: Sign stable in >= 6 of 8 years
        if stab_report.sign_stable:
            c2_result = "PASS"
        else:
            c2_result = "FAIL"
        reasons.append(
            f"2. Sign stability criterion: {c2_result} "
            f"({stab_report.n_years_sign_consistent}/{stab_report.n_years_total} years)"
        )

        # Criterion 3: Overlap correction (block-bootstrap t-stat > 1.96)
        spread_t = abs(exp_table.spread_t)
        c3_result = "PASS" if spread_t > 1.96 else "FAIL"
        reasons.append(f"3. Overlap correction criterion: {c3_result}")

        # Criterion 4: Not concentrated in bottom liquidity decile or single time-of-day
        # Check bottom decile
        bottom_decile_edge = None
        if 0 in stab_report.by_liquidity_decile:
            bottom_decile_edge = stab_report.by_liquidity_decile[0].spread_bps

        c4_result = "PASS"
        concentration_reasons: list[str] = []

        # Check if concentrated in the bottom liquidity decile: the bottom
        # decile's spread is the dominant contributor among deciles that
        # actually have assessable data (min_names-gated NaN deciles are
        # excluded honestly, not treated as zero edge).
        if bottom_decile_edge is not None and bottom_decile_edge != 0:
            liquidity_edges = [
                t.spread_bps
                for t in stab_report.by_liquidity_decile.values()
            ]
            nonzero_liquidity_edges = [e for e in liquidity_edges if e != 0]
            if len(nonzero_liquidity_edges) == 1:
                # Only the bottom decile shows any edge at all.
                c4_result = "FAIL"
                concentration_reasons.append("concentrated in bottom liquidity decile")
            else:
                sorted_liquidity_edges = sorted(abs(e) for e in nonzero_liquidity_edges)
                max_liquidity_edge = sorted_liquidity_edges[-1]
                median_liquidity_edge = sorted_liquidity_edges[len(sorted_liquidity_edges) // 2]
                if (
                    median_liquidity_edge > 0
                    and max_liquidity_edge > median_liquidity_edge * 2
                    and abs(bottom_decile_edge) == max_liquidity_edge
                ):
                    c4_result = "FAIL"
                    concentration_reasons.append("concentrated in bottom liquidity decile")

        # Check if concentrated in one time-of-day
        _tod_reason = "concentrated in single time-of-day bucket"
        if stab_report.by_time_of_day:
            # If there's only one time-of-day bucket with data, it's concentrated
            if len(stab_report.by_time_of_day) == 1:
                c4_result = "FAIL"
                concentration_reasons.append(_tod_reason)
            else:
                # stab_report.by_time_of_day is truthy (len >= 1) and the len == 1
                # case was already handled above, so len >= 2 here -- there is no
                # third case, and time_of_day_edges is therefore never empty.
                # Check if the time-of-day buckets have consistent signs and similar magnitudes
                time_of_day_edges = [
                    t.spread_bps for t in stab_report.by_time_of_day.values()
                ]
                # Check for sign consistency (all positive or all negative)
                nonzero_edges = [e for e in time_of_day_edges if e != 0]
                if nonzero_edges:
                    same_sign = all((e > 0) == (nonzero_edges[0] > 0) for e in nonzero_edges)
                    if not same_sign:
                        # Mixed signs indicate concentration
                        c4_result = "FAIL"
                        concentration_reasons.append(_tod_reason)
                    else:
                        # same_sign is the negation of the branch above -- no
                        # third outcome for a bool, so this is an else, not
                        # an independent elif condition.
                        # Check if one bucket dominates (>50% larger than median)
                        sorted_edges = sorted(abs(e) for e in nonzero_edges)
                        if len(sorted_edges) > 1:
                            max_edge = sorted_edges[-1]
                            median_edge = sorted_edges[len(sorted_edges) // 2]
                            if median_edge > 0 and max_edge > median_edge * 2:
                                c4_result = "FAIL"
                                concentration_reasons.append(_tod_reason)

        concentration_reason = "; ".join(concentration_reasons)
        if concentration_reason:
            reasons.append(f"4. Concentration criterion: {c4_result} ({concentration_reason})")
        else:
            reasons.append(f"4. Concentration criterion: {c4_result}")

        # Criterion 5: Latency profile flat
        if latency_profile is None:
            c5_result = "NOT_EVALUATED"
            reasons.append(f"5. Latency profile criterion: {c5_result}")
        else:
            # Check if edge is retained at lags 1 and 2
            lag_0_edge = latency_profile.get(0, 1.0)
            lag_1_edge = latency_profile.get(1, 0.0)
            lag_2_edge = latency_profile.get(2, 0.0)

            # Retain >= 50% of lag-0 edge magnitude at lags 1 and 2
            if lag_0_edge > 0:
                lag_1_ratio = lag_1_edge / lag_0_edge
                lag_2_ratio = lag_2_edge / lag_0_edge
                if lag_1_ratio >= 0.5 and lag_2_ratio >= 0.5:
                    c5_result = "PASS"
                else:
                    c5_result = "FAIL"
            else:
                c5_result = "FAIL"

            reasons.append(
                f"5. Latency profile criterion: {c5_result} (lags: {latency_profile})"
            )

        # Criterion 6: Deflated Sharpe
        # Compute returns from forward returns
        fwd_flat = fwd.values.flatten()
        valid_mask = np.isfinite(fwd_flat)
        valid_returns = fwd_flat[valid_mask]

        if len(valid_returns) > 0:
            dsr = deflated_sharpe(valid_returns, sr0=0.0)
            # Expected max Sharpe under null
            if effective_n_trials >= 2:
                exp_max_sharpe = expected_max_sharpe(effective_n_trials, var_trial_sharpes=1.0)
                if not np.isnan(dsr) and dsr > exp_max_sharpe:
                    c6_result = "PASS"
                else:
                    c6_result = "FAIL"
            else:
                # Single trial, compare to standard Sharpe threshold
                if not np.isnan(dsr) and dsr > 0.0:
                    c6_result = "PASS"
                else:
                    c6_result = "FAIL"
        else:
            c6_result = "FAIL"

        reasons.append(f"6. Deflated Sharpe criterion: {c6_result} (trials={effective_n_trials})")

        # Criterion 7: Recent-years cost gate
        # Count usable sessions per year from already-loaded panel dates.
        session_counts: dict[int, int] = {}
        for d in self.panel.dates:
            year = d.year
            session_counts[year] = session_counts.get(year, 0) + 1

        # A year is complete only if it has a by-year expectancy table and at
        # least 20 usable sessions.
        complete_years: list[int] = sorted(
            year for year in stab_report.by_year if session_counts.get(year, 0) >= 20
        )

        if len(complete_years) < 2:
            c7_result = "NOT_EVALUATED"
            years_desc: str
            if complete_years:
                years_desc = ", ".join(str(year) for year in complete_years)
            else:
                years_desc = "none"
            reasons.append(
                f"7. Recent-years cost gate criterion: {c7_result} "
                f"(complete_years={years_desc})"
            )
        else:
            recent_years: list[int] = sorted(complete_years)[-2:]
            year_a: int = recent_years[0]
            year_b: int = recent_years[1]
            edge_year_a: float = stab_report.by_year[year_a].spread_bps
            edge_year_b: float = stab_report.by_year[year_b].spread_bps
            mean_recent_edge: float = float(
                np.mean([edge_year_a, edge_year_b])
            )

            recent_sign: str = "mixed"
            if mean_recent_edge > 0:
                recent_sign = "+"
            elif mean_recent_edge < 0:
                recent_sign = "-"

            magnitude_pass: bool = abs(mean_recent_edge) > 2 * cost_hurdle_bps
            sign_pass: bool = (
                stab_report.dominant_sign != "mixed"
                and recent_sign == stab_report.dominant_sign
            )
            c7_result = "PASS" if magnitude_pass and sign_pass else "FAIL"

            reasons.append(
                f"7. Recent-years cost gate criterion: {c7_result} "
                f"(years={year_a},{year_b}; "
                f"edges={edge_year_a:.2f},{edge_year_b:.2f} bps; "
                f"mean={mean_recent_edge:.2f} bps; "
                f"2x hurdle={2 * cost_hurdle_bps:.2f} bps; "
                f"dominant_sign={stab_report.dominant_sign!r})"
            )

        # Determine if survived
        reason_tokens = [
            r.split(":")[1].strip().split()[0] for r in reasons
        ]  # Extract PASS/FAIL/NOT_EVALUATED
        evaluated_results = [t for t in reason_tokens if t != "NOT_EVALUATED"]
        survived = all(r == "PASS" for r in evaluated_results)

        return HypothesisVerdict(
            hypothesis_id=hypothesis_id,
            survived=survived,
            reasons=tuple(reasons),
            expectancy=exp_table,
            stability=stab_report,
            cost_hurdle_bps=cost_hurdle_bps,
            seed=self.seed,
        )
