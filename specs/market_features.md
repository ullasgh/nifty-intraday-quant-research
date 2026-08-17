# Spec: `nifty_quant.features.market` — market-state and cross-sectional features

**Status:** contract for TDD. Tests written from this document alone, before implementation.

## Why this exists

`features/core.py` has per-symbol primitives (rolling stats, breakout, Parkinson vol,
deseasonalization, cross-sectional z/rank). It has **nothing about the market as a system** —
no beta, no breadth, no dispersion, no correlation, no volatility regime, no order-flow proxy.
Every hypothesis in Phase 3 needs those. This module supplies them.

**Available inputs that make this possible**: `NIFTY50`, `NIFTY100`, `NIFTYBANK` and — crucially
— **`INDIAVIX` at 1-minute resolution**, all in `data/bars/1/`. `INDIAVIX` is the only
forward-looking volatility series in the dataset.

## Hard rules (all enforced by tests)

- **Causal**: row `t` may depend only on rows `<= t`. Every public function carries `@causal`
  from `nifty_quant.guards`.
- **Session-aware**: sessions come from `day_offsets`. NEVER assume 375 bars (Muhurat is 60,
  DR-Saturday 105). Nothing may span a session boundary unless the docstring says so explicitly.
- **NaN means "no bar"** and is never forward-filled. NaN propagates.
- **float32 at rest, float64 in motion.** Cast on entry.
- **No hand-chosen thresholds** (CLAUDE.md rule 8). These functions return continuous values;
  they must not bucket, gate or threshold anything internally.
- Vectorized numpy. No `iterrows`, no per-bar Python loops. Reuse `features/core.py` primitives.

## Public API

```python
# --- market-relative -------------------------------------------------------
def rolling_beta(returns, market_returns, window, *, day_offsets, min_count=None) -> np.ndarray
    """Rolling OLS beta of each symbol on the market, cov/var over `window` bars, session-aware.
    Shape (n_rows, n_symbols). NaN until `min_count` finite pairs exist."""

def beta_residual_return(returns, market_returns, window, *, day_offsets) -> np.ndarray
    """r_i - beta_i * r_mkt, using a STRICTLY PRIOR beta (beta at t-1 applied to return at t).
    Using the contemporaneous beta is a lookahead leak and the tests check for it."""

def sector_relative_return(returns, sector_ids, *, min_names=3) -> np.ndarray
    """r_i minus the equal-weighted mean return of its sector at the same row.
    `sector_ids` is an int array of shape (n_symbols,); -1 means unknown -> output NaN.
    Sectors with fewer than `min_names` present symbols yield NaN. Cross-sectional only, so
    causal by construction."""

# --- market regime ---------------------------------------------------------
def breadth(returns, *, min_names=5) -> np.ndarray
    """(n_advancing - n_declining) / n_present, per row. Shape (n_rows,). Zero returns count
    as neither. NaN when fewer than `min_names` present."""

def cross_sectional_dispersion(returns, *, min_names=5) -> np.ndarray
    """Cross-sectional standard deviation of returns per row, ddof=1. Shape (n_rows,)."""

def median_pairwise_correlation(returns, window, *, day_offsets, min_names=5) -> np.ndarray
    """Median off-diagonal entry of the rolling correlation matrix. Shape (n_rows,).
    EXPENSIVE -- document the cost and compute it on a stride if needed, but the returned
    array must still be one value per row (forward-filled from the last computed stride point
    is FORBIDDEN; use NaN between stride points and say so)."""

def vol_ratio(returns, short_window, long_window, *, day_offsets) -> np.ndarray
    """sigma_short / sigma_long per symbol. > 1 means volatility expansion.
    Raises ValueError if short_window >= long_window."""

def rv_to_vix_ratio(realized_vol_ann, vix_level) -> np.ndarray
    """Annualised realized vol divided by INDIAVIX (which is quoted in annualised PERCENT, so
    divide by 100 before comparing). > 1 means realized exceeds implied. The unit mismatch is
    the single most likely bug here and has its own test."""

# --- order-flow proxies from OHLCV ----------------------------------------
def close_location_value(high, low, close) -> np.ndarray
    """(2*close - high - low) / (high - low), in [-1, +1]. +1 = closed on the high.
    Where high == low (no range), return 0.0, not NaN and not inf."""

def signed_volume_proxy(high, low, close, volume) -> np.ndarray
    """close_location_value * volume. A crude buy/sell imbalance proxy; the only order-flow
    signal available without a book."""

def amihud_illiquidity(returns, traded_value, window, *, day_offsets) -> np.ndarray
    """Rolling mean of |return| / traded_value, scaled by 1e6 for readability. Higher = more
    price impact per rupee. traded_value <= 0 contributes NaN, never inf."""

# --- session structure -----------------------------------------------------
def bars_since_open(day_offsets, n_rows) -> np.ndarray
    """0-based bar index within its session, shape (n_rows,) int32. Handles irregular sessions."""

def overnight_return(close, day_offsets) -> np.ndarray
    """log(first close of session d / last close of session d-1), broadcast to every row of
    session d. First session is NaN. This is the ONE function permitted to span a session
    boundary -- that is its purpose."""

def opening_range(high, low, day_offsets, n_bars) -> tuple[np.ndarray, np.ndarray]
    """(or_high, or_low) over the first `n_bars` of each session, broadcast to every row of
    that session. Rows BEFORE bar `n_bars` must be NaN -- broadcasting the completed range
    backwards onto the bars that formed it is a lookahead leak, and the tests check it."""
```

## Sector map

Add `nifty_quant/universe/sectors.py`: a hand-written `SECTOR_MAP: dict[str, str]` for the 149
equities, plus `sector_ids(symbols) -> np.ndarray[int32]`. Unknown symbols map to `-1`.
**Document loudly that this is a CURRENT-DAY map, not point-in-time** — the same survivorship
caveat the universe already carries. A symbol that changed sector historically is mislabelled.

## Required tests (`tests/test_market_features.py`)

**Causality (highest value — one per public function):**
1-13. `test_<fn>_is_causal` — perturb rows after `t`, assert row `t` unchanged, at
`Strictness.FULL`. Follow `tests/verification/test_causality.py`'s probe pattern.
14. `test_beta_residual_uses_prior_beta_not_contemporaneous` — the specific leak.
15. `test_opening_range_is_nan_before_range_completes` — the specific leak.

**Session handling:**
16. `test_all_functions_respect_irregular_sessions` — 375-bar day then 60-bar Muhurat day,
parametrized over every function.
17. `test_nothing_spans_a_session_boundary_except_overnight_return`
18. `test_overnight_return_first_session_is_nan`
19. `test_bars_since_open_restarts_each_session`

**Correctness (hand-computed microdata, in `tests/verification/test_math_microdata.py` style):**
20. `test_rolling_beta_matches_hand_computed_cov_over_var`
21. `test_beta_of_market_on_itself_is_one`
22. `test_breadth_counts_advancing_minus_declining_over_present`
23. `test_breadth_zero_returns_count_as_neither`
24. `test_dispersion_matches_numpy_std_ddof_one`
25. `test_clv_equals_plus_one_when_close_equals_high`
26. `test_clv_equals_minus_one_when_close_equals_low`
27. `test_clv_is_zero_when_high_equals_low` — not NaN, not inf
28. `test_clv_bounded_in_minus_one_to_one` — property test over random OHLC
29. `test_vol_ratio_greater_than_one_on_expanding_vol`
30. `test_vol_ratio_rejects_short_ge_long`
31. `test_rv_to_vix_divides_vix_by_100` — the unit trap; VIX 20.0 vs realized 0.20 gives 1.0
32. `test_amihud_higher_for_illiquid_symbol`
33. `test_amihud_non_positive_traded_value_yields_nan_not_inf`
34. `test_median_pairwise_correlation_of_identical_series_is_one`
35. `test_median_pairwise_correlation_of_independent_series_is_near_zero`
36. `test_sector_relative_return_sums_to_zero_within_sector`
37. `test_sector_relative_unknown_sector_is_nan`
38. `test_sector_below_min_names_is_nan`
39. `test_opening_range_matches_first_n_bars_high_low`

**NaN / degenerate:**
40. `test_nan_bars_propagate_and_are_not_filled` — parametrized over every function
41. `test_all_nan_symbol_never_poisons_a_cross_sectional_statistic`
42. `test_min_names_gates_breadth_dispersion_and_correlation`
43. `test_constant_series_zero_variance_degenerate_cases`

**Dtype/contract:**
44. `test_all_outputs_are_float64`
45. `test_output_shapes_match_contract` — per-symbol (n_rows, n_sym) vs market-level (n_rows,)
46. `test_sector_ids_maps_unknown_to_minus_one`
