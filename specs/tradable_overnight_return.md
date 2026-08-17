# Spec: `features.market.tradable_overnight_return` — label-resolved, tradable-endpoint variant

**Status:** contract for TDD. Two independent test suites from this document alone, before
implementation. Raised by an H2 test author who noticed the existing function contradicts H2's own
formula; confirmed on real data before writing this.

## The defect in the existing `overnight_return`

```python
def overnight_return(close, day_offsets):  # existing, stays as-is
    """log(first_close_of_day_d / last_close_of_day_d-1)"""
```

It resolves endpoints **positionally** — `close[day_offsets[i]]` and `close[day_offsets[i]-1]` —
not by time label. Measured on RELIANCE 2024 (249 sessions):

    session FIRST bar label: 09:15 in 248 of 249 sessions
    session LAST  bar label: 15:29 in 246 of 249 sessions

So in practice it computes `log(close[09:15] / close[15:29, prev])`. **Neither endpoint is
tradable under this repo's own rules**: the earliest honest entry is 09:16, and `square_off_time`
is 15:20. Concretely, 2024-03-15:

    close[09:15] = 1427.55     open[09:16]  = 1427.50    (~0.4 bps apart — minor)
    close[15:20] = 1417.65     close[15:29] = 1419.00    (~9.5 bps apart — MATERIAL)

The exit leg is nine minutes past square-off. That injects ~9.5 bps of un-capturable drift into a
signal whose H2 edge is ~24 bps against a 16.53 bps hurdle — over half the apparent edge could be
post-square-off movement nobody can trade.

Secondary issue: the 09:15 bar is *sometimes* corrupt (`close > high` from pre-open call-auction
leakage — 61 instances across 60 symbols in 2024, not every session, so this is a tail risk rather
than a systematic one). Resolving by label avoids it either way.

**Do NOT change `overnight_return`'s behaviour.** It has passing tests and a documented contract,
and "first/last close of session" is a legitimate quantity for other purposes. Add the variant, and
add a loud warning to the existing docstring pointing at it.

## Public API

```python
def tradable_overnight_return(
    open_: np.ndarray,          # (n_rows, n_symbols) or (n_rows,)
    close: np.ndarray,          # same shape
    day_offsets: np.ndarray,
    minute_of_day: np.ndarray,  # (n_rows,), from Panel.minute_of_day()
    *,
    entry_hhmm: str = "09:16",
    exit_hhmm: str = "15:20",
) -> np.ndarray:
    """log(open[entry_hhmm, d] / close[exit_hhmm, d-1]), broadcast to every row of session d.

    Endpoints resolved BY TIME LABEL, never by row position. Uses the entry OPEN (what you can
    actually transact at) and the prior session's exit CLOSE (at square-off, not the session's
    final print).

    Returns float64, shape matching the input's symbol dimension. NaN where either endpoint is
    absent for that symbol-session — per symbol, never per panel.
    """
```

## Binding rules

- **Label resolution, per session.** Use `minute_of_day` to locate `entry_hhmm` within session `d`
  and `exit_hhmm` within session `d-1`. NEVER `day_offsets[i]` / `day_offsets[i]-1`.
- **`d-1` is the previous session present in `day_offsets`**, not the previous calendar day.
- **Per-SYMBOL NaN.** If symbol A has both endpoints and symbol B is missing one, A is finite and B
  is NaN for that session. A panel-level drop would discard good data.
- **A session missing the entry label yields NaN for that whole session** (e.g. a ~60-bar Muhurat
  session has no 15:20, so the NEXT session's value is NaN). It must NOT reach further back for an
  older close — that would silently bridge across a missing day.
- First session is NaN (no prior).
- **No forward-fill anywhere.** NaN means absent (CLAUDE.md rule 6).
- float64 in motion; accepts float32 input and casts (rule 3).
- Carries `@causal` — but note it legitimately spans ONE session boundary backwards, exactly like
  `overnight_return`. Row `t` still depends only on rows `<= t`.
- Vectorized; a loop over sessions is acceptable, over bars is not.
- Never assume 375 bars per session (rule 5).

## Required tests

Two INDEPENDENT suites: `tests/test_tradable_overnight_deepseek.py`,
`tests/test_tradable_overnight_luna.py`.

1. Hand-computed: `log(open[09:16, d] / close[15:20, d-1])` on a synthetic 2-session panel.
2. **Differs from `overnight_return` on realistic bars.** Build a session with distinct
   09:15/09:16/15:20/15:29 values and assert the two functions return DIFFERENT numbers, with the
   new one matching the label-resolved formula. This is the regression test for the whole defect.
3. Ignores the 09:15 bar entirely: corrupt it (`close > high`, the real observed defect) and assert
   the result is bit-identical.
4. Ignores bars after `exit_hhmm`: plant a 15:29 bar with a wildly different close; result unchanged.
5. Custom labels honoured (`entry_hhmm="09:20"`, `exit_hhmm="15:00"`).
6. Session missing the entry label -> NaN for that session, counted, no reach-back.
7. Muhurat (~60 bars, no 15:20) -> NEXT session NaN, and specifically NOT bridged to an older close.
8. Per-symbol independence: A finite, B NaN in the same session.
9. First session NaN.
10. Broadcast to every row of its session.
11. NaN input propagates; no forward-fill.
12. float32 input -> float64 output; values match the float64 computation within float32 tolerance.
13. Irregular sessions (375-bar then 60-bar) resolve correctly.
14. 1-D and 2-D inputs both supported, shapes correct.
15. `@causal` probe passes at `Strictness.FULL`.
16. **Label handling — RESOLVED 2026-08-17. Two DISTINCT cases, both mandatory.** The original
    wording ("raises or yields NaN, pick one") was under-specified, and the two independent test
    authors each picked a different case — correctly, because they are different situations:
    - **Malformed format** (e.g. `"25:99"`, `"9am"`, `"0916"`) -> **raise `ValueError`**. Mirrors
      `SessionGrid.rows_at_time`'s `datetime.strptime` contract in `calendar.py`. A caller typo
      must fail loudly, not silently return NaN that reads as "no data".
    - **Well-formed but absent from the data** (e.g. `entry_hhmm="03:00"`, a valid time no NSE
      session contains) -> **all-NaN, no exception**. Consistent with "NaN means absent"
      (CLAUDE.md rule 6) and identical in spirit to a session that simply lacks the label.
    The distinction matters: one is a programming error, the other is a data condition. Collapsing
    them either way loses information. Both suites' existing tests pass under this resolution.
17. Determinism.

Plus: assert the existing `overnight_return` docstring now contains a pointer to this function, so
the trap is discoverable from where someone would hit it.

## Constraints

100% line and branch coverage measured with pragma exclusion DISABLED. Fully annotated. `ruff`
clean. Do not modify `overnight_return`'s behaviour or its existing tests.
