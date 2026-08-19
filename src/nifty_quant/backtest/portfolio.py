"""Portfolio state accounting and P&L tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Portfolio:
    """Single-book portfolio state.

    cum_costs accumulates only modelled charges, never slippage. cum_pnl satisfies
    cum_pnl == cash + sum(shares * prices) + cum_costs - initial_capital.
    """

    shares: np.ndarray
    cash: float
    initial_capital: float
    cum_costs: float = 0.0
    cum_pnl: float = 0.0
    # F4: last finite close observed per symbol, used ONLY to mark a symbol that
    # is currently HELD (shares != 0) when its current price is missing. This is
    # a deliberate, counted exception to rule 6 (never silently forward-fill) --
    # every use is tallied in `n_stale_marks` and surfaced on `BacktestResult`,
    # never silent. It is not used to invent a price for absent/flat symbols,
    # which are already masked to 0.0 contribution regardless of `last_close`.
    last_close: np.ndarray | None = None
    n_stale_marks: int = 0
    # Internal: the price array most recently passed to `mark()`. The engine
    # legitimately calls `mark()` more than once per row, and (via
    # prev_close == this row's close at the next row's top-of-loop mark, or a
    # decision-row mark repeated by a later same-row flush) sometimes calls it
    # again later with the IDENTICAL array. Without this guard each such repeat
    # would recount the same stale substitution in `n_stale_marks`; `cum_pnl` is
    # still recomputed every call (state such as shares/cash may have changed),
    # only the stale-mark bookkeeping is skipped on a genuine repeat.
    _last_marked_prices: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.last_close is None:
            self.last_close = np.full(self.shares.shape, np.nan, dtype=np.float64)

    def equity(self, prices: np.ndarray) -> float:
        """Return cash plus mark-to-market value, float64.

        A non-finite (NaN/inf) price on a symbol with shares == 0 contributes
        exactly 0.0 (absent-or-flat masking, unchanged). A zero or negative
        FINITE price is a legitimate (if extreme) market price and is used
        as-is -- this is what ruin detection depends on. A non-finite price on
        a symbol with shares != 0 (F4: a held position with a missing bar) is
        marked at that symbol's last observed finite close instead of the raw
        (NaN) price -- read-only here; `last_close` itself is advanced, and
        `n_stale_marks` incremented, only by `mark()`, so repeated read-only
        `equity()` calls in the same row never double-count a stale mark.
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape != self.shares.shape or prices.ndim != 1:
            raise ValueError("prices must be 1-D and match shares")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            held = self.shares != 0
            price_valid = np.isfinite(prices)
            use_stale = held & ~price_valid & np.isfinite(self.last_close)

            mark_prices = np.where(use_stale, self.last_close, prices)
            safe_prices = np.where(held, mark_prices, 0.0)
            mtm = float(np.dot(self.shares, safe_prices))
        return float(self.cash + mtm)

    def _advance_last_close(self, prices: np.ndarray) -> int:
        """Update `last_close` from every finite price in *prices* and return the
        number of currently-held symbols that needed a stale substitution against
        the PRE-update `last_close` (i.e. against what `equity()` would have used
        for this same call). Called exactly once per genuine mark event, from
        `mark()` -- never from `equity()` -- so the count is never doubled by the
        redundant `equity()` calls the engine makes against an already-marked
        price array within the same row.
        """
        assert self.last_close is not None
        held = self.shares != 0
        price_valid = np.isfinite(prices) & (prices != 0.0)
        stale_now = int(np.sum(held & ~price_valid & np.isfinite(self.last_close)))
        self.last_close = np.where(price_valid, prices, self.last_close)
        return stale_now

    def apply_fills(self, qty: np.ndarray, price: np.ndarray, charges: float) -> None:
        """Apply fills, updating cash, shares, cum_costs; cum_pnl is untouched."""
        qty = np.asarray(qty, dtype=np.float64)
        price = np.asarray(price, dtype=np.float64)
        if (
            qty.ndim != 1
            or price.ndim != 1
            or qty.shape != price.shape
            or qty.shape != self.shares.shape
        ):
            raise ValueError("qty and price must be 1-D and match shares")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Mask zero-fill columns the same way: a NaN fill price at a zero-qty
            # index must not make the cash delta non-finite.
            safe_price = np.where(qty != 0, price, 0.0)
            fill_value = float(np.dot(qty, safe_price))

        self.cash -= fill_value + charges
        self.shares = self.shares + qty
        self.cum_costs += charges

    def mark(self, prices: np.ndarray) -> None:
        """Recompute cum_pnl from the current snapshot and prices, then advance
        `last_close`/`n_stale_marks` (F4) for this mark event. `equity()` runs
        first so it reads the PRE-advance `last_close` -- the state as of the
        prior mark -- and only after that does this call become the new
        "last observed" state for the next one.

        `cum_pnl` is recomputed on EVERY call, including a repeat with the same
        `prices` array (shares/cash may have moved between calls, e.g. a fill
        between two marks against the same close). `last_close`/`n_stale_marks`
        are advanced only when `prices` differs from the last call's array, so a
        genuine repeat -- which the engine's per-row/per-session bookkeeping
        sometimes produces deliberately -- cannot double-count a stale mark.
        """
        prices_arr = np.asarray(prices, dtype=np.float64)
        self.cum_pnl = float(self.equity(prices_arr) + self.cum_costs - self.initial_capital)
        if self._last_marked_prices is None or not np.array_equal(
            prices_arr, self._last_marked_prices, equal_nan=True
        ):
            self.n_stale_marks += self._advance_last_close(prices_arr)
            self._last_marked_prices = prices_arr.copy()


@dataclass(frozen=True)
class SizingResult:
    """Result of capacity-aware sizing (returned only when bar_traded_value is given).

    shares: clean signed share counts, same contract as the legacy plain-array return.
    unmet_gross_pct: fraction (0.0-1.0) of the intended gross notional (sum of
        abs(target notional) before capacity capping) that could not be sized because
        every name with spare room was already at its own capacity cap. 0.0 means
        capacity was not binding (or was fully absorbed by renormalisation).

    The four fields below are populated only by `VolTargetSizer` (specs/portfolio_vol_target.md,
    amendment 4 item 3). They default to NaN/False so `GrossNotionalSizer`'s existing
    construction sites, which only ever pass `shares` and `unmet_gross_pct`, are untouched.
    NaN means "this result did not come from a vol-targeted sizer" -- it is not a
    missing value to be filled in later, and callers must treat it as "not
    applicable" rather than coercing it to zero.
    """

    shares: np.ndarray
    unmet_gross_pct: float
    sigma_portfolio_ann: float = float("nan")
    """Ex-ante portfolio vol from the RAW (unscaled, unclipped, sigma-zeroed) weights."""
    vol_scale_applied: float = float("nan")
    """target_vol_ann / sigma_portfolio_ann, computed BEFORE any clip."""
    vol_target_achieved: float = float("nan")
    """Realised ex-ante portfolio vol from the FINAL weights, AFTER every step."""
    clip_binding: bool = False
    """True iff max_weight, max_gross, or capacity water-filling moved the book off
    the vol target (amendment 5). Quantisation -- whole_shares rounding and
    min_trade_notional dust filtering -- never sets this (amendment 4 item 2)."""


def _finalize_shares_shared(
    notional: np.ndarray,
    prices: np.ndarray,
    valid_price: np.ndarray,
    *,
    whole_shares: bool,
    min_trade_notional: float,
) -> np.ndarray:
    """Convert clean notional to share counts with rounding and dust filtering.

    Shared step-7 logic (spec: whole_shares rounding, then min_trade_notional dust
    filter) -- used by both GrossNotionalSizer and VolTargetSizer so the two sizers
    quantise identically for the same field values (specs/portfolio_vol_target.md
    amendment 2: "every field VolTargetSizer shares with GrossNotionalSizer carries
    the same default", and by extension the same behaviour).
    """
    safe_prices = np.where(valid_price, prices, 1.0)
    raw_shares = np.zeros_like(notional)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(notional, safe_prices, out=raw_shares, where=valid_price)

    if whole_shares:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            abs_shares = np.abs(raw_shares)
            signs = np.where(raw_shares > 0, 1.0, np.where(raw_shares < 0, -1.0, 0.0))
            floor_abs = np.floor(abs_shares)
        floor_abs = np.where(np.isfinite(floor_abs), floor_abs, 0.0)
        raw_shares = signs * floor_abs

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        notional = np.abs(raw_shares * prices)
    notional = np.where(np.isfinite(notional), notional, 0.0)
    raw_shares = np.where(notional < min_trade_notional, 0.0, raw_shares)

    result: np.ndarray = np.nan_to_num(raw_shares, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _capacity_water_fill(
    target_notional: np.ndarray,
    bar_traded_value: np.ndarray,
    max_participation: float,
) -> tuple[np.ndarray, bool, float]:
    """Cap each name's notional at max_participation * bar_traded_value and
    redistribute any shortfall into names with spare room (iterative water-filling).

    Shared step-6 logic, lifted unchanged from GrossNotionalSizer.to_shares's
    original capacity path so both sizers "reuse the pattern" (spec section A
    required-tests note 7) bit-for-bit. Returns
    (capacity_adjusted_notional, capacity_bound, unmet_gross_pct).

    capacity_bound is True iff at least one name's notional was reduced below its
    desired (pre-capacity) value by the capacity cap -- whether or not the
    shortfall was later fully redistributed elsewhere. Amendments 4/5: capacity
    moving the book AT ALL is what sets clip_binding, not merely an unfilled
    residual, because a reshuffled cross-section can miss the vol target even at
    unchanged total gross.
    """
    bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
    if bar_traded_value.shape != target_notional.shape:
        raise ValueError("bar_traded_value must have the same shape as weights/prices")

    desired_abs = np.abs(target_notional)
    sign = np.sign(target_notional)
    valid_btv = np.isfinite(bar_traded_value) & (bar_traded_value > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        capacity = np.where(valid_btv, max_participation * bar_traded_value, 0.0)
        capacity = np.where(np.isfinite(capacity) & (capacity > 0), capacity, 0.0)

    final_abs = np.minimum(desired_abs, capacity)
    orig_gross = float(np.sum(desired_abs))
    capacity_bound = bool(np.any(final_abs < desired_abs))

    # Iterative water-filling: redistribute the shortfall created by capped names
    # into names that still have spare room under their OWN capacity (not their own
    # original desired_abs -- survivors may exceed their original per-name target,
    # exactly like the engine's absent-symbol renormalisation). Only names with
    # desired_abs > 0 participate. At most len(target_notional) rounds: each round
    # either saturates at least one more name or fully closes the shortfall.
    for _ in range(target_notional.shape[0]):
        achieved = float(np.sum(final_abs))
        shortfall = orig_gross - achieved
        if shortfall <= 1e-9 * max(orig_gross, 1.0):
            break
        free_mask = (desired_abs > 0) & (final_abs < capacity)
        if not np.any(free_mask):
            break
        free_share = np.where(free_mask, final_abs, 0.0)
        total_free_share = float(np.sum(free_share))
        if total_free_share <= 0.0:
            break  # pragma: no cover - free_share sum invariant
            # Whenever free_mask has any True entry, that entry's final_abs is provably
            # > 0. On the first water-filling round, free_mask True implies final_abs ==
            # desired_abs (since final_abs < capacity strictly) and desired_abs > 0 is
            # part of the mask definition. On later rounds a still-free name only ever
            # has add >= 0 applied, so it never returns to 0. Hence total_free_share > 0
            # whenever free_mask is non-empty. Kept as a guard against a future refactor
            # of the water-filling loop; excluded from coverage rather than deleted so
            # the invariant stays asserted in the code.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            add = np.where(
                free_mask,
                shortfall * free_share / total_free_share,
                0.0,
            )
        final_abs = final_abs + add
        final_abs = np.minimum(final_abs, capacity)

    achieved_gross = float(np.sum(final_abs))
    if orig_gross <= 0.0:
        unmet_gross_pct = 0.0
    else:
        unmet_gross_pct = max(0.0, (orig_gross - achieved_gross) / orig_gross)
    if not np.isfinite(unmet_gross_pct):
        unmet_gross_pct = 0.0  # pragma: no cover - finite ratio invariant
        # capacity is forced finite and non-negative earlier in this function by
        # np.where(np.isfinite(capacity) & (capacity > 0), capacity, 0.0); desired_abs
        # is finite/non-negative upstream; so final_abs = min(desired_abs, capacity) is
        # finite. final_abs <= desired_abs pointwise gives achieved_gross <= orig_gross,
        # and orig_gross <= 0.0 is ruled out by the preceding branch, so the ratio cannot
        # be NaN or +/-inf. Kept because a non-finite unmet_gross_pct would silently
        # poison downstream metrics with NaN rather than raising.

    capacity_adjusted_notional = sign * final_abs
    capacity_adjusted_notional = np.where(
        np.isfinite(capacity_adjusted_notional), capacity_adjusted_notional, 0.0
    )
    return capacity_adjusted_notional, capacity_bound, float(unmet_gross_pct)


@dataclass(frozen=True)
class GrossNotionalSizer:
    """Convert weights to share orders with dust and participation guards."""

    gross: float = 1.0
    max_weight: float = 0.10
    min_trade_notional: float = 25_000.0
    whole_shares: bool = True

    def _finalize_shares(
        self, notional: np.ndarray, prices: np.ndarray, valid_price: np.ndarray
    ) -> np.ndarray:
        """Convert clean notional to share counts with rounding and dust filtering."""
        return _finalize_shares_shared(
            notional,
            prices,
            valid_price,
            whole_shares=self.whole_shares,
            min_trade_notional=self.min_trade_notional,
        )

    def to_shares(
        self,
        weights: np.ndarray,
        prices: np.ndarray,
        capital: float,
        *,
        bar_traded_value: np.ndarray | None = None,
        max_participation: float = 0.02,
    ) -> np.ndarray | SizingResult:
        """Return clean signed share counts; optionally capacity-aware.

        When bar_traded_value is None, behaviour is bit-identical to the original,
        capacity-unaware implementation and the return type is a plain np.ndarray.
        When bar_traded_value is given, each name's target notional is capped at
        max_participation * bar_traded_value (default matches FillModel.max_participation
        in nifty_quant.execution.fills -- this is a documented coupling, not a shared
        constant, to avoid a circular import; callers that must stay in sync should pass
        max_participation=config.fill_model.max_participation explicitly), the shortfall
        is renormalised into names with spare capacity (mirroring the engine's existing
        absent-symbol renormalisation, which lets survivors exceed their own original
        target), and the result is returned as a SizingResult exposing unmet_gross_pct.
        """
        weights = np.asarray(weights, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        if weights.shape != prices.shape:
            raise ValueError("weights and prices must have the same shape")

        valid_price = np.isfinite(prices) & (prices > 0)
        valid_weight = np.isfinite(weights)
        safe_weight = np.where(valid_weight, weights, 0.0)
        clipped_weight = np.clip(safe_weight, -self.max_weight, self.max_weight)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            target_notional = clipped_weight * self.gross * capital
        target_notional = np.where(np.isfinite(target_notional), target_notional, 0.0)
        target_notional = np.where(valid_price, target_notional, 0.0)

        if bar_traded_value is None:
            return self._finalize_shares(target_notional, prices, valid_price)

        bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
        if bar_traded_value.shape != weights.shape:
            raise ValueError("bar_traded_value must have the same shape as weights/prices")

        capacity_adjusted_notional, _capacity_bound, unmet_gross_pct = _capacity_water_fill(
            target_notional, bar_traded_value, max_participation
        )
        shares = self._finalize_shares(capacity_adjusted_notional, prices, valid_price)
        return SizingResult(shares=shares, unmet_gross_pct=unmet_gross_pct)


@dataclass(frozen=True)
class VolTargetSizer:
    """Size a book to a declared portfolio-level annualized vol target.

    specs/portfolio_vol_target.md, amendment 5: the pipeline is exactly seven
    steps, in exactly this order:

        1. w_raw       <- caller's weights (already signal_i / sigma_i)
        2. w_vol        <- w_raw * (sigma_target_ann / sigma_portfolio_ann)
        3. w_clip       <- clip(w_vol, -max_weight, +max_weight)          [per-name]
        4. notional     <- w_clip * gross * capital                       [gross is a
                           MULTIPLIER, identical to GrossNotionalSizer:148]
        5. notional     <- proportional rescale IF max_gross is not None and
                           sum(|notional|) > max_gross * capital, so that
                           sum(|notional|) == max_gross * capital          [whole-book]
        6. notional     <- capacity cap + water-filling                   [per-name]
        7. shares       <- whole_shares rounding, then min_trade_notional dust filter

    Step 5 comes AFTER step 4 so max_gross bounds the FINAL exposure for any value
    of gross. clip_binding is True iff step 3, 5, or 6 moved the book -- never step
    4 (a declared multiplier) or step 7 (quantisation).

    Portfolio vol under a constant-correlation covariance (Sigma = D C D, D =
    diag(sigma_i), C = (1-rho) I + rho * 1 1') is computed in closed form, never as
    a materialised n x n matrix:

        w' Sigma w = (1 - rho) * sum_i (w_i sigma_i)^2 + rho * (sum_i w_i sigma_i)^2

    When `corr` is unavailable or non-finite, amendment 1's endpoint-max fallback is
    used instead. var(rho) is LINEAR in rho, so its maximum over rho in [0, 1] is at
    an endpoint -- conservative for any sign structure, no case analysis needed:

        var_conservative = max(sum_i a_i^2, (sum_i a_i)^2),  a_i = w_i * sigma_i

    Every field this sizer shares with GrossNotionalSizer carries the SAME default
    (amendment 2): a divergent default is an undeclared behaviour change.
    """

    target_vol_ann: float
    gross: float = 1.0
    max_weight: float = 0.10
    min_trade_notional: float = 25_000.0
    whole_shares: bool = True
    max_gross: float | None = None

    def to_shares(
        self,
        weights: np.ndarray,
        prices: np.ndarray,
        capital: float,
        *,
        bar_traded_value: np.ndarray | None = None,
        max_participation: float = 0.02,
        sigma: np.ndarray | None = None,
        corr: float | None = None,
    ) -> SizingResult:
        """Return a SizingResult unconditionally (amendment 1 item 2): if diagnostics
        vanished whenever bar_traded_value is None, clip_binding would be invisible
        on the default path, reproducing the exact silent-miss this spec exists to
        remove.

        `sigma` (per-name annualized vol, already annualized by the caller -- this
        sizer never sees a Panel or day_offsets) is required; a name with a
        non-finite sigma_i gets weight zero, exactly as a non-finite price does
        today, and does not poison the portfolio vol estimate. `corr` is the
        measured median pairwise correlation rho (a scalar), or None/non-finite if
        unavailable, in which case the conservative endpoint-max fallback is used.
        """
        if sigma is None:
            raise ValueError("sigma is required")

        weights = np.asarray(weights, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if weights.shape != prices.shape:
            raise ValueError("weights and prices must have the same shape")
        if weights.shape != sigma.shape:
            raise ValueError("weights and sigma must have the same shape")

        valid_price = np.isfinite(prices) & (prices > 0)
        valid = np.isfinite(weights) & np.isfinite(sigma)

        # Step 1: w_raw -- non-finite weight or non-finite sigma zeroes that name,
        # exactly as a non-finite price does today (spec section C).
        w_raw = np.where(valid, weights, 0.0)
        sigma_masked = np.where(valid, sigma, 0.0)

        sigma_portfolio_ann, vol_scale_applied = self._vol_scale(w_raw, sigma_masked, corr)

        # Step 2 cont.
        w_vol = w_raw * vol_scale_applied

        # Step 3: clip to max_weight (per-name).
        w_clip = np.clip(w_vol, -self.max_weight, self.max_weight)
        step3_binds = bool(np.any(w_clip != w_vol))

        # Step 4: notional = w_clip * gross * capital -- gross is a bare multiplier,
        # identical semantics to GrossNotionalSizer:148. Never sets clip_binding.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            notional = w_clip * self.gross * capital
        notional = np.where(np.isfinite(notional), notional, 0.0)
        notional = np.where(valid_price, notional, 0.0)

        # Step 5: proportional whole-book rescale IF max_gross binds, AFTER the
        # gross multiply (amendment 5) so max_gross bounds sum(|notional|)/capital
        # for any value of gross.
        step5_binds = False
        if self.max_gross is not None:
            gross_cap_notional = self.max_gross * capital
            current_abs_sum = float(np.sum(np.abs(notional)))
            if current_abs_sum > gross_cap_notional and current_abs_sum > 0.0:
                notional = notional * (gross_cap_notional / current_abs_sum)
                step5_binds = True

        # Step 6: capacity cap + water-filling (per-name, existing pattern, shared
        # with GrossNotionalSizer via _capacity_water_fill).
        step6_binds = False
        unmet_gross_pct = 0.0
        if bar_traded_value is not None:
            bar_traded_value = np.asarray(bar_traded_value, dtype=np.float64)
            if bar_traded_value.shape != weights.shape:
                raise ValueError(
                    "bar_traded_value must have the same shape as weights/prices"
                )
            notional, step6_binds, unmet_gross_pct = _capacity_water_fill(
                notional, bar_traded_value, max_participation
            )

        # Step 7: whole_shares rounding, then min_trade_notional dust filter. Never
        # sets clip_binding (amendment 4 item 2: a flag that is always True from
        # rounding noise carries no information).
        shares = _finalize_shares_shared(
            notional,
            prices,
            valid_price,
            whole_shares=self.whole_shares,
            min_trade_notional=self.min_trade_notional,
        )

        clip_binding = step3_binds or step5_binds or step6_binds

        vol_target_achieved = self._realised_vol(
            shares, prices, capital, valid_price, sigma_masked, corr
        )

        return SizingResult(
            shares=shares,
            unmet_gross_pct=unmet_gross_pct,
            sigma_portfolio_ann=sigma_portfolio_ann,
            vol_scale_applied=vol_scale_applied,
            vol_target_achieved=vol_target_achieved,
            clip_binding=clip_binding,
        )

    def _vol_scale(
        self, w: np.ndarray, sigma: np.ndarray, corr: float | None
    ) -> tuple[float, float]:
        """Step 2: sigma_portfolio_ann (closed form) and vol_scale_applied.

        `w`/`sigma` are the RAW (sigma-zeroed, unscaled, unclipped) inputs -- the
        "BEFORE the clips" diagnostic the spec requires.
        """
        variance = self._closed_form_variance(w, sigma, corr)
        sigma_portfolio_ann = float(np.sqrt(max(variance, 0.0)))

        if sigma_portfolio_ann > 0.0 and np.isfinite(sigma_portfolio_ann):
            vol_scale_applied = self.target_vol_ann / sigma_portfolio_ann
        else:
            # No measured risk to scale against (an all-zero or fully-masked book).
            # Levering an undefined ratio up from zero risk is not scale-invariant
            # and not conservative, so we do not lever at all rather than guess.
            vol_scale_applied = 0.0

        return sigma_portfolio_ann, float(vol_scale_applied)

    @staticmethod
    def _closed_form_variance(w: np.ndarray, sigma: np.ndarray, corr: float | None) -> float:
        """w' Sigma w in closed form, Sigma = D C D, C = (1-rho) I + rho * 1 1':

            w' Sigma w = (1 - rho) * sum_i (w_i sigma_i)^2 + rho * (sum_i w_i sigma_i)^2

        Never materialises an n x n matrix (constraint, spec section "Constraints").
        When `corr` is None or non-finite, falls back to amendment 1's conservative
        endpoint-max (var(rho) is linear in rho, so its max over rho in [0, 1] sits
        at an endpoint -- conservative for any sign structure).
        """
        a = w * sigma
        sum_a_sq = float(np.sum(a * a))
        sum_a = float(np.sum(a))

        if corr is not None and np.isfinite(corr):
            rho = float(corr)
            return (1.0 - rho) * sum_a_sq + rho * (sum_a**2)
        return max(sum_a_sq, sum_a**2)

    def _realised_vol(
        self,
        shares: np.ndarray,
        prices: np.ndarray,
        capital: float,
        valid_price: np.ndarray,
        sigma_masked: np.ndarray,
        corr: float | None,
    ) -> float:
        """vol_target_achieved: realised ex-ante portfolio vol from FINAL weights,
        AFTER every step -- the same closed-form variance (with the same
        rho/fallback choice) applied to what was actually sized, not what was
        requested.

        AMENDMENT 6 (specs/portfolio_vol_target.md): "final weights" are
        `shares * prices / capital`, normalised by `capital` ALONE -- NOT by
        `capital * self.gross`. An earlier version of this method divided the
        `gross` multiply back out on the theory that `gross` is a declared,
        non-binding leverage dial (true) and therefore should not appear in "did
        we hit the target" (false). That made `vol_target_achieved` read "0.15,
        on target" for a book actually running at `target_vol_ann * gross` --
        the original target_vol_ann-in-a-meta-dict defect wearing a better name,
        this time inside the diagnostic built to catch it. `gross` scales the
        REAL book, so it must appear in the REALISED-vol figure. Two separate
        questions get two separate fields: `vol_scale_applied` reports whether
        steps 1-3's targeting arithmetic did its job (gross-free, as before);
        `vol_target_achieved` reports the actual risk the book carries, gross
        included. `clip_binding` is unaffected by this fix -- `gross` still never
        sets it, because it is a declared dial, not a constraint that fought the
        target; a large `gross` legitimately produces a large, correctly-reported
        `vol_target_achieved` with `clip_binding == False`.
        """
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            final_weights = np.where(valid_price, shares * prices / capital, 0.0)
        final_weights = np.where(np.isfinite(final_weights), final_weights, 0.0)
        variance = self._closed_form_variance(final_weights, sigma_masked, corr)
        return float(np.sqrt(max(variance, 0.0)))
