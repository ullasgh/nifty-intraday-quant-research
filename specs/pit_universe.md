# Spec: point-in-time universe selection

Phase B track B1.

## Why this exists, and what it CANNOT fix

`configs/universe/*.yaml` are static symbol lists — today's constituents applied to all history.
`load_universe` (`universe/static.py:114`) reads them verbatim and all three CLI research paths
use the raw list (`cli.py:539`, `:883`, `:1255`).

`Universe.as_of(d)` already exists (`static.py:41-85`) and is **never called in production** —
only from tests. It is honest about itself: its result carries `source="availability_proxy"`
(`static.py:84`) and its docstring (`:49-54`) says "This is a data availability proxy, not real
index membership ... this repository has no point-in-time membership history."

**State the boundary before writing any code.** This dataset contains:
- no index-membership history for NIFTY or any other index
- **zero delisted names** — `scripts/recon_survivorship.py` established that all 149 symbols run
  to the end of the window; the wealth-destroyers stayed listed and the delisted names were
  never downloaded

Therefore true survivorship correction is **not achievable here**, and no API introduced by this
spec may imply otherwise. What we can fix is the *other* half of the problem, which the H2 work
already showed is the live one: **new listings**. 18 names IPO'd inside 2018-2025 (IRCTC,
SBICARD, NYKAA, ETERNAL, JIOFIN, LICI, HYUNDAI, ...) and they contribute signal in exactly the
recent years where the tilt candidate's significance is being judged. The measured deltas run
the wrong way for a survivorship story — tiny and wrong-signed early (2018 +1.54, 2019 +0.57 bps)
and largest recently (2023 +7.52, 2024 +7.94 bps) — so what the continuous-coverage control
actually measures is new listings, not survivor bias.

The deliverable is therefore: **make eligibility as-of-date, wire it in, and name it accurately.**

## Required behaviour

### A. Eligibility is evaluated per session, causally

A symbol is eligible on session `d` when both hold, using only data strictly before `d`:

1. **Listed:** it has at least `min_history_sessions` prior sessions with a present bar. This is
   what removes the newly-listed-names artifact.
2. **Liquid:** its trailing ADV clears the threshold. Reuse `data/validate.py:736-739`
   (`compute_prior_adv`-style trailing-20, current session excluded) — do NOT write a second ADV.
   `universe/filters.py:57-73 liquidity_filter` takes a caller-supplied `adv_by_symbol` mapping
   with no time axis; it must not be used for this, and its docstring should say so.

Output is a boolean `(n_sessions, n_symbols)` eligibility mask, not a symbol list. A per-session
mask is the only representation that can express "eligible from March onward".

### B. Naming that cannot mislead

The result keeps a `source` field. Permitted values:

    "availability_proxy"     -- listed-and-liquid, computed from bar presence (what we can do)
    "index_membership"       -- reserved; requires membership data we do not have

`as_of()` must never return `"index_membership"` from the current data. If a future data refresh
adds membership history, that is a new source value, not a redefinition of this one.

The docstring states the delisting boundary explicitly, at the API surface, so a caller reading
only the signature learns it.

### C. Wiring

All three CLI research paths take the eligibility mask and apply it wherever the universe is used
— cross-sectional ranks, portfolio construction, and the tradable mask — instead of the raw list.
A name ineligible on session `d` is excluded from that session's cross-section entirely; it is
not merely given zero weight, because a name present in a cross-sectional rank changes every
other name's rank.

`min_history_sessions` is NOT a hand-chosen constant (rule 8). Derive it: measure how many prior
sessions a newly-listed name needs before its cross-sectional signal statistics become
indistinguishable from an established name's, and record the derivation next to the value. If the
measurement is inconclusive, report that to the lead rather than picking a round number.

### D. `universe_hash`

A stable hash of (universe name, sorted eligible symbol set per session, parameters). It feeds
Phase B3's `TrialRecord.universe_hash` so a published number can be tied to the exact membership
that produced it.

### E. Sector map

`universe/sectors.py:3-6` carries the same current-day caveat. Not fixed here — there is no
point-in-time sector history either — but any sector-relative feature must surface the caveat in
its provenance rather than only in a source comment.

## Required tests

1. A name whose first present bar is mid-window is INELIGIBLE before that date and eligible
   `min_history_sessions` after it.
2. Eligibility on session `d` is unchanged by mutating any data at or after `d` (causality).
3. A name that fails the trailing-liquidity test on session `d` is excluded from that session's
   cross-section, and the remaining names' ranks are computed as if it were absent — not as if it
   had zero weight.
4. The eligibility mask has shape `(n_sessions, n_symbols)` and no fixed session-length
   assumption anywhere (rule 5).
5. `source` is `"availability_proxy"` on all current data; constructing an
   `"index_membership"` result from current data raises.
6. `universe_hash` is stable across runs and changes when the eligible set changes on any session.
7. Survivorship reporting still fires and still says what it can and cannot measure.
8. Applying the mask reproduces the continuous-coverage universe used in the H2 and tilt work
   when `min_history_sessions` is set to span the whole window — i.e. the new machinery is a
   generalisation of the existing control, not a different answer.

## Constraints

- Never modify anything under `data/` (rule 2).
- `present` and `tradable` stay distinct (rule 7); eligibility is a THIRD concept and must not be
  merged into either. A name can be present, tradable, and still ineligible for research because
  it listed last week.
