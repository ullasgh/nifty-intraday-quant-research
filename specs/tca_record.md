# Spec C1 — the transaction-cost-analysis record

Status: spec, written before implementation. Author: lead. Date 2026-08-20.

## Why

Execution is where this program's strategies die. `volume_breakout` measured gross Sharpe -0.048
and net -0.233 with **73.7% of desired notional unfilled**, and the edge does not survive one
minute of latency. Those are execution facts, and the blotter as it stands cannot express them:
it records what happened to the orders that filled, not what was wanted and missed, and it reports
one blended slippage number rather than the components a cost decision needs.

The goal of this record is that a reader can answer, from the blotter alone and without re-running
anything: *how much of my shortfall was spread, how much was impact, how much was fees, and how
much was simply not getting filled.*

## Current state — verified, not assumed

The blotter has **13 columns** (`engine.py:346-360`, built at `:399-417`, cast at `:992-1009`):
`ts`, `symbol`, `qty`, `price`, `notional`, `is_buy`, `charges`, `decision_price`, `fill_price`,
`shortfall_bps`, `participation`, `filled_frac`, `intent`. It is written to
`trial_dir / "result.parquet"` (`cli.py:728`); no dedicated reader exists.

Two corrections to the external review's framing, both established by inspection:

1. **Fees were never conflated into `shortfall_bps`.** `shortfall_bps` (`engine.py:378`) is
   `1e4 * (price_f - decision_price) / decision_price * side` — purely directional price
   slippage against the decision reference. Statutory charges live in their own `charges`
   column. The review's "collapsing all three into shortfall_bps" is wrong about fees.

2. **Spread and impact ARE conflated, one level down.** `SqrtImpactSlippage.bps`
   (`fills.py:20-39`) returns a single scalar per order,
   `half_spread_bps + impact_coef * impact`. The two components are computed and then
   immediately summed, so by the time the engine sees a number they are unrecoverable. This is
   the real defect and it is in `fills.py`, not the blotter.

`costs.as_bps_of(notional)` exists at `costs.py:116`, computes `(self.total / notional) * 1e4`
with a divide-by-zero guard, and has **zero call sites**. It is exactly the function needed to
express fees in bps and it has never been called.

On timestamps: only ONE time concept exists — bar index `t`, mapped to `ts` via `panel.ts[t]`.
`decision_time` is an "HH:MM" string used to *select* decision bars, not a recorded field.
`decision_latency_bars` (default 0) makes an order decided at bar `t` execute at `t + 1 + latency`.
`signal_time`, `order_time` and `fill_time` do not exist.

## The record

Every column below is per fill row. Existing columns keep their current names and semantics; this
is additive, so anything reading the current blotter keeps working.

### Timestamps — four, all int64 epoch-seconds UTC

Per repo rule 4 bars are left-labelled: the bar labelled T covers `[T, T+60s)`. Each timestamp is
the label of the bar the event belongs to, never an interpolated intra-bar time. We do not have
intra-bar timing and must not imply we do.

| column | meaning |
|---|---|
| `signal_ts` | label of the last bar whose data entered the signal. This is the causality anchor: nothing at or after `signal_ts + 60s` may have informed the decision. |
| `decision_ts` | label of the bar at which the target was formed. Equals `signal_ts` when the strategy decides on the bar it last observed. |
| `order_ts` | label of the bar at which the order became live. `decision_ts` advanced by `1 + decision_latency_bars` bars **along the session index**, never by a fixed 60s multiple (rule 5: sessions vary; Muhurat is 60 bars). |
| `fill_ts` | label of the bar the fill occurred in. Equals `order_ts` for a single-bar fill; for an order that fills across bars there is one row per bar, each with its own `fill_ts`. |

The four collapse to the same value under `decision_latency_bars = 0` with same-bar fill. They
must still all be present, because a record that only stores them when they differ cannot be used
to prove they did not.

### Quantities — what was wanted, not only what happened

| column | meaning |
|---|---|
| `desired_qty` | signed shares the strategy asked for, before any participation cap |
| `filled_qty` | signed shares actually filled on this row (existing `qty`) |
| `fill_ratio` | `filled_qty / desired_qty`, NaN when `desired_qty == 0` |

`filled_frac` already exists and must be reconciled, not duplicated: if it already means
`filled_qty / desired_qty` then `fill_ratio` is an alias and only one survives — the implementer
must check and report which, rather than adding a second column with the same content. This is the
kind of near-duplicate that makes a blotter untrustworthy.

The 73.7%-unfilled figure must be recoverable from these columns alone.

### Prices

| column | meaning |
|---|---|
| `arrival_price` | reference price at `order_ts` — the price a costless immediate execution would have received |
| `decision_price` | existing; reference at `decision_ts` |
| `fill_price` | existing; achieved price |
| `mid` | mid at `fill_ts` where available; NaN otherwise, never silently substituted with close |

`mid` is NaN for this dataset unless a genuine mid is available. Rule 6's spirit applies: a
fabricated mid is worse than an absent one. Do NOT set `mid = close` and do not fill it forward.

### Cost decomposition — the identity that must hold

| column | meaning |
|---|---|
| `spread_bps` | half-spread component, signed against the trade direction |
| `impact_bps` | market-impact component |
| `fees_bps` | statutory + brokerage charges as bps of notional, via `costs.as_bps_of` |
| `slippage_bps` | `spread_bps + impact_bps` |
| `shortfall_bps` | existing; total realised price shortfall vs `decision_price` |

**Required invariant, asserted in tests:**

    slippage_bps == spread_bps + impact_bps      (to within 1e-9 relative)

and the total cost of the row in bps is `slippage_bps + fees_bps`. `shortfall_bps` is NOT required
to equal that sum — it is measured against realised prices and includes adverse selection and
price drift between decision and fill, which are not costs the model charged. **Any implementation
that forces those two to agree has hidden the very quantity this record exists to expose:** the
gap between modelled cost and realised shortfall is the model's error, and it must stay visible.

### Interface change in `fills.py`

`SlippageModel` (Protocol, `fills.py:11`) currently exposes only `bps(...) -> np.ndarray`. It gains:

    def components(self, notional, bar_traded_value) -> SlippageComponents

returning a frozen dataclass with `spread_bps` and `impact_bps` arrays. `bps()` remains and MUST
be redefined as `components(...).spread_bps + components(...).impact_bps` — computed once, not
twice — so the blended number can never drift from its own parts. `ZeroSlippage` returns zeros for
both.

Keep `half_spread_bps=1.5` and `impact_coef=10.0` as the named baseline. They are **assumed, not
measured**, and a prior calibration attempt established they cannot be recovered from OHLCV alone:
regressing on this data returns the assumed constants back (intercept 1.5044 vs 1.5, slope 9.9828
vs 10.0). Do not present them as calibrated, and do not let this spec's structure imply they are.
Rule 8 is not satisfied here and the record must say so rather than quietly look rigorous.

## What this does not do

It does not add intra-bar timing, a real mid, or a measured impact coefficient — none of the three
is obtainable from 1-minute OHLCV. It makes the absence explicit and machine-readable instead of
implicit, which is the honest available improvement.

## Test obligations

Dual independent suites per rule 1, written from this spec alone.

1. The four timestamps exist on every row, are int64 epoch-seconds, and satisfy
   `signal_ts <= decision_ts <= order_ts <= fill_ts`.
2. Under `decision_latency_bars = 0` and same-bar fill, all four are equal.
3. Under `decision_latency_bars = k > 0`, `order_ts` is `k+1` bars after `decision_ts`
   **counted along the session**, verified on a SHORT session (use a 60-bar Muhurat-shaped
   fixture, not a 375-bar one) so a fixed-stride implementation fails.
4. `slippage_bps == spread_bps + impact_bps` exactly, on random notionals.
5. `fees_bps` equals `charges / notional * 1e4`, and `as_bps_of` is the function that computed it.
6. A partial fill records `desired_qty` > `filled_qty`, `fill_ratio` < 1, and the unfilled
   remainder is recoverable.
7. An order filling across three bars produces three rows with distinct `fill_ts` and
   `sum(filled_qty)` equal to the total filled.
8. `mid` is NaN when no mid is available, and is never equal to `close` by construction.
9. `ZeroSlippage` gives `spread_bps == impact_bps == slippage_bps == 0` while `fees_bps` may be
   non-zero.
10. Round-trip: blotter written to parquet and read back preserves every dtype, including the
    int64 timestamps and the object-dtype `intent`.

---

# AMENDMENT 1 — 2026-08-20. Obligation 2 was WRONG, and four ambiguities.

A test author checked obligation 2 against the engine instead of accepting it, and found it
contradicts the engine's own documented invariant. They wrote the test literally as specified and
reported the contradiction rather than quietly adjusting it. That is the correct behaviour and the
spec is what changes.

## 1. Obligation 2 is wrong — there is no same-bar fill in this engine

`engine.py`'s module docstring states: *"Mandatory one-bar execution lag: a weight emitted for bar
t is never filled before t+1"*, implemented as `fill_row = t + 1 + decision_latency_bars`. So even
at `decision_latency_bars = 0`, an order is filled at `t + 1`. My obligation 2 claimed all four
timestamps collapse to one value at latency 0. They cannot, and no correct implementation could
satisfy it — the test as written would be permanently unsatisfiable.

**Restated obligation 2:** at `decision_latency_bars = 0`, `signal_ts == decision_ts`, and
`order_ts == fill_ts == ` the label of the NEXT bar along the session index. The one-bar lag is
structural and must be visible in the record rather than assumed away.

The general ordering in the spec body (`signal_ts <= decision_ts <= order_ts <= fill_ts`) is
unaffected and remains correct.

## 2. Rows need an `order_id`

Obligation 7 groups fill rows belonging to one order, and the record named no key for it. The
author proxied with `(symbol, order_ts, desired_qty)`, which is not guaranteed unique.

**Add `order_id`** — a monotonically increasing int64 assigned at order creation, stable across
every fill row the order produces.

## 3. Multi-bar fills are NOT in scope for C1

The current engine fills each `PendingOrder` exactly once; it never splits an order across bars.
Obligation 7 as written implied new execution behaviour, which is a much larger change than a
record format and does not belong in a TCA spec.

**Restated obligation 7:** the record SHAPE must support multiple fill rows per `order_id` with
distinct `fill_ts` and `sum(filled_qty)` equal to the total — assert this on a hand-constructed
blotter, not by forcing the engine to split. Separately assert that a partially-filled order
today produces exactly one row whose unfilled remainder is recoverable from
`desired_qty - filled_qty`.

Splitting fills across bars is deferred and recorded here as deferred, not silently dropped.

## 4. `arrival_price` is pinned to `open[order_ts]`

Ambiguous between `close[order_ts - 1]` and something at `order_ts`. Pinned: **the open of the bar
labelled `order_ts`**. That is the first price available once the order is live, so it is the price
a costless immediate execution would have received — which is what arrival price means in TCA.
`decision_price` keeps its existing definition and the two differ exactly when the price moved
across the lag, which is the quantity worth seeing.

## 5. `fill_ratio` is NOT added; `filled_frac` stays

The spec body flagged this and asked the implementer to check. Resolved: keep the existing
`filled_frac` and do not add a second column. The implementer must VERIFY that `filled_frac`
already equals `filled_qty / desired_qty` and report if it does not — a near-duplicate column with
subtly different semantics is how a blotter stops being trustworthy.

Obligation 6 asserts on `filled_frac`.

---

# AMENDMENT 2 — 2026-08-20. Obligation 10 is unsatisfiable on pandas 3.x; and an untestable edge.

## 1. Obligation 10's "object-dtype" requirement is wrong

A test author reported that this repo runs **pandas 3.0.5**, where string columns round-trip
through parquet as a native `str` dtype rather than legacy `object` — even when the DataFrame was
explicitly cast with `.astype({"symbol": object, "intent": object})`, which is exactly what
`engine.py:992-1009` does today.

Verified directly by the lead rather than taken on report:

    pandas 3.0.5
    before: {'symbol': 'object', 'intent': 'object', 'ts': 'int64'}
    after : {'symbol': 'str',    'intent': 'str',    'ts': 'int64'}
    values equal: True | ts dtype int64: True

So obligation 10 as written ("preserves every dtype, including ... the object-dtype `intent`") can
never pass, for columns that already exist, independent of anything this spec adds. The author left
it as a genuine failing test rather than weakening it, which was correct — the spec is what changes.

**Restated obligation 10:** a blotter written to parquet and read back must preserve
(a) every VALUE exactly, (b) the int64 dtype of `order_id` and all four timestamp columns, and
(c) float64 for the price and bps columns. String-valued columns (`symbol`, `intent`) must come
back as a string dtype — assert `pandas.api.types.is_string_dtype(...)`, which is true for both
legacy `object` and pandas 3.x `str`. Do NOT assert the legacy `object` identity.

Note for the implementer: the existing `.astype({... : object})` in `engine.py` is not preserved
across persistence on this pandas version. It is harmless — values survive — but it should not be
described as a dtype guarantee, because it is not one.

## 2. `desired_qty == 0` is unobservable and the obligation is scoped accordingly

The author reported that a fully rejected order emits ZERO blotter rows, so a row with
`desired_qty == 0` cannot arise through the engine. The NaN branch of `filled_frac` is therefore
untestable end-to-end.

**Resolution:** test it as a UNIT on the ratio computation directly (given `desired_qty == 0`, the
ratio is NaN, not 0.0 and not a ZeroDivisionError), and do NOT attempt to manufacture it through
the engine. Manufacturing an impossible engine state to satisfy an obligation tests a path that
cannot occur, which is a different kind of false confidence from a vacuous assertion but is still
false confidence.

Recorded as scoped-down rather than dropped.

---

# AMENDMENT 3 — 2026-08-20. My AMENDMENT 2 item 2 was factually WRONG.

## 1. `desired_qty == 0` IS reachable. Correcting myself.

AMENDMENT 2 stated that a row with `desired_qty == 0` "cannot arise through the engine" because a
fully rejected order emits zero blotter rows, and scoped the NaN branch to a unit test on that
basis. That is wrong.

A pluggable fill model can report a PHANTOM fill for a symbol with no desired order, and the engine
records it as a real trade — it goes through `apply_fills` like any other. Two pre-existing tests
cover exactly this and have for some time:

    test_engine_coverage.py::test_filled_frac_zero_when_denom_zero
    test_engine_coverage3.py::test_filled_frac_forced_zero_for_phantom_fill_with_zero_desired_order

Both require `filled_frac == 0.0` — finite, not NaN.

## 2. The semantics are pinned to 0.0, and there must be exactly ONE implementation

`filled_frac` for `desired_qty == 0` is **0.0**, not NaN. Reasons, in order of weight:

1. It is the existing, deliberately-tested behaviour ("forced_zero" is in the test's own name), and
   changing tested engine behaviour to suit a newly-written spec is backwards.
2. The case is pathological — it requires a test-double fill model inventing a fill — so the
   aggregation argument for NaN carries little weight against breaking a real covered path.

**Critical: do not leave two implementations with different semantics.** The implementer added a
standalone `compute_filled_frac` returning NaN while `_record_trade` kept returning 0.0, precisely
to avoid breaking those tests. That is the near-duplicate-with-subtly-different-semantics hazard
this spec's body warns about, one level down. `compute_filled_frac` must return **0.0** for a zero
denominator and `_record_trade` must CALL it, so there is a single definition.

Obligation 6's unit assertion changes accordingly: `desired_qty == 0` yields `0.0`, not NaN, and
never raises.

## 3. Obligation 6's `len(rows_for_symbol) == 1` is wrong in BOTH suites

Both suites starve an entry to a 1-share partial fill in a 20-bar session, then assert the symbol
has exactly one blotter row. It has two: `square_off_time="15:20"` is never reached inside 20
one-minute bars from 09:15, so the engine's mandatory session-end flattening emits a second,
structurally-required row (`qty=-1.0`, `intent="forced_eod"`).

Every preceding assertion in both tests is correct — the partial-fill row does have
`desired_qty > filled_qty`, `filled_frac < 1.0`, and `filled_frac == filled_qty / desired_qty`.
Only the trailing count assertion is wrong.

**Restated:** select the row by `order_id` (or by `intent`), not by symbol, and assert exactly one
row for THAT order. Then additionally assert the forced-EOD row EXISTS — a partially-filled
position must still end the session flat, and asserting that makes the test stronger than the
count it replaces.

## 4. Note for the record: dual suites do not protect against a SHARED misreading

Both suites were written independently and both made the same wrong assumption — one blotter row
per symbol. Independence protects against differing misreadings of a spec; it does not protect
against a shared misunderstanding of the ENGINE, because both authors read the same engine.

This is the first time in this program the dual-suite mechanism has failed to catch something, and
it is worth knowing its shape: agreement between the suites is evidence, not proof.

---

# AMENDMENT 4 — 2026-08-20. Two corrections to AMENDMENT 3, both found by measurement.

## 1. The intent is `eod_exit`, not `forced_eod`. I asserted this without checking.

AMENDMENT 3 stated the session-end row carries `intent="forced_eod"`. It carries **`eod_exit`**.

`square_off_row_for_day` clamps to the session's LAST row (`engine.py:329-338`) when the configured
`square_off_time` is never reached inside the fixture's bars, so the flattening routes through the
ordinary EOD_EXIT queue path. `FORCED_EOD` is a terminal safety net that only fires if EOD_EXIT
itself fails to flatten — a strictly rarer event.

The implementer printed `result.trades` and read the value rather than taking my word. That
distinction matters beyond the label: a test asserting `forced_eod` would have been asserting that
the ordinary square-off path had FAILED, which is a materially different claim about the engine
than "the position was closed at session end".

## 2. Only ONE pre-existing test actually covers `desired_qty == 0`, not two

AMENDMENT 3 cited two tests as requiring a finite `0.0`. Verified by mutation:

- `test_engine_coverage3.py::test_filled_frac_forced_zero_for_phantom_fill_with_zero_desired_order`
  — REAL. Reverting the branch to NaN fails it (`nan == 0.0`).
- `test_engine_coverage.py::test_filled_frac_zero_when_denom_zero` — **VACUOUS**. It only asserts
  `result.trades.empty` and never reaches the `filled_frac` branch at all. It stayed GREEN under the
  same mutation, and its name promises something it does not test.

So my evidence for the pinned semantics was half as strong as I claimed. The conclusion does not
change — one real covering test is still a real covering test, and 0.0 remains pinned — but the
overstatement is recorded because "two tests cover this" was doing argumentative work it had not
earned.

**The vacuous test is a pre-existing gap and is adjudicated for repair:** it must be made to
actually reach the `filled_frac` computation with a zero denominator, or renamed to say what it
tests. A test whose name claims a branch it never executes is worse than no test — it is why the
branch looked covered.

## 3. One test still asserts the retracted NaN contract

`tests/test_tca_record_a.py::test_filled_frac_ratio_is_nan_when_desired_qty_is_zero_unit` was
written under AMENDMENT 2's premise and asserts NaN. AMENDMENT 3 retracted that premise and pinned
0.0. **Adjudicated: update it to assert `0.0`**, and rename it so the name matches the contract.

Running tally of tests found asserting a defect or a retracted contract in this program: **15**.
