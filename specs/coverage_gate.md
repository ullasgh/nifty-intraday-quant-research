# Spec: `scripts/coverage_gate.py`

**Status:** contract for TDD. Tests are written from this document alone, before implementation.

## Why this exists

The project rule is 100% line **and** branch coverage across the repo. Measured baseline on
2026-08-17 is **77%** (407 tests passing), so 100% cannot be enforced globally on day one
without blocking all work. A single global `fail_under` also has a second problem: it hides
*which* module regressed, and it lets a module silently rot from 99% to 90% as long as the
total stays above the line.

This script enforces a **per-module ratchet** instead: each module has a floor in
`coverage_floor.json`, the floor can only rise, and any module falling below its floor fails
the build with a message naming the module and the delta.

`fail_under` is deliberately absent from `pyproject.toml` so ad-hoc `--cov` runs during
development stay exit-0 and the gate remains a deliberate, visible act (`make gate`).

## Inputs

- `--report PATH` — a coverage JSON report (`--cov-report=json:PATH`). Default `.coverage.json`.
- `--floor PATH` — the ratchet file. Default `coverage_floor.json`.
- `--update` — instead of checking, RAISE floors to currently-measured values and rewrite the
  file. Never lowers a floor. Exits 0 and reports what it raised.
- `--target N` — the eventual goal, default read from the floor file's `_target` key (100).

`coverage_floor.json` schema: top-level `modules` maps a repo-relative module path to an integer
percentage. Keys beginning with `_` (`_comment`, `_baseline`, `_target`) are metadata and must
be preserved verbatim across `--update` rewrites.

## Coverage definition

The percentage compared against the floor is **combined line and branch coverage**, matching what
`pytest --cov-branch --cov-report=term` prints in its `Cover` column, computed from the JSON
report as:

```
covered   = summary.covered_lines + summary.covered_branches
total     = summary.num_statements + summary.num_branches
pct       = 100.0 * covered / total       # total == 0  ->  pct = 100.0
```

Truncate toward zero when comparing to an integer floor, so a module at 89.97% does not pass a
floor of 90. Report the untruncated value to one decimal place.

## Behaviour

**Check mode (default):**
1. Load both files. A missing or malformed report/floor file is a usage error, exit code **2**,
   with a message naming the file and the problem.
2. For each module in the coverage report under `src/nifty_quant/`:
   - Not present in `modules` -> **UNGATED**, a failure. New code must be added to the floor
     file explicitly (at 100), so it cannot slip in ungated.
   - `pct < floor` -> **REGRESSED**, a failure.
   - `floor <= pct < target` -> **DEBT**, not a failure. Report it.
   - `pct >= target` -> **OK**.
3. For each module in `modules` absent from the report -> **MISSING**, a failure. This catches a
   module deleted or renamed without updating the floor, and a test run that silently skipped a
   package.
4. Print a table sorted worst-first: module, floor, actual, delta, status. Then a summary line
   with counts per status, the overall percentage, and the number of modules still at 100.
5. Exit **1** if any REGRESSED, UNGATED or MISSING. Exit **0** otherwise (DEBT alone passes).

**Update mode (`--update`):**
1. Same load and comparison.
2. For each module where `pct > floor`, raise the floor to `floor(pct)`. Never lower.
3. UNGATED modules are ADDED at their measured truncated value, and a warning is printed that a
   new module was auto-added below target and should be brought to 100.
4. MISSING modules are left untouched and reported — `--update` must not delete entries, because
   a skipped test run would otherwise silently erase the ratchet.
5. Rewrite the floor file preserving all `_`-prefixed metadata, with `modules` keys sorted, 2-space
   indent, and a trailing newline, so the diff is reviewable.
6. Print what changed. Exit **0**.

## Output contract

Every failure line must be actionable on its own — module path, floor, actual, and what to do.
This is the debuggability requirement: someone reading only CI output must know which module to
open, without re-running anything.

## Required tests (`tests/test_coverage_gate.py`)

Synthetic report/floor JSON in `tmp_path`. No test may run pytest recursively or read the real
`.coverage.json`.

### Percentage computation
1. `test_pct_combines_lines_and_branches`
2. `test_pct_is_100_when_no_statements_and_no_branches`
3. `test_pct_truncates_toward_zero_against_integer_floor` — 89.97 fails a floor of 90
4. `test_pct_reported_to_one_decimal`

### Check mode
5. `test_all_modules_at_target_exits_zero`
6. `test_module_below_floor_exits_one_and_names_it`
7. `test_module_between_floor_and_target_is_debt_and_exits_zero`
8. `test_module_in_report_but_not_in_floor_is_ungated_and_exits_one`
9. `test_module_in_floor_but_not_in_report_is_missing_and_exits_one`
10. `test_modules_outside_src_nifty_quant_are_ignored`
11. `test_table_sorted_worst_first`
12. `test_summary_line_counts_each_status`
13. `test_failure_message_contains_module_floor_actual_and_delta`
14. `test_exact_equality_with_floor_passes` — boundary: pct == floor is OK, not a regression

### Update mode
15. `test_update_raises_floor_to_measured_value`
16. `test_update_never_lowers_a_floor` — the core ratchet invariant
17. `test_update_adds_ungated_module_and_warns`
18. `test_update_does_not_delete_missing_modules`
19. `test_update_preserves_underscore_metadata_verbatim`
20. `test_update_writes_sorted_keys_two_space_indent_trailing_newline`
21. `test_update_exits_zero_even_when_check_would_fail`
22. `test_update_is_idempotent` — running twice on the same report changes nothing the second time

### Errors
23. `test_missing_report_file_exits_two`
24. `test_missing_floor_file_exits_two`
25. `test_malformed_json_exits_two_and_names_the_file`
26. `test_floor_file_without_modules_key_exits_two`

### Determinism
27. `test_output_is_deterministic_for_identical_input`

## Constraints

- Standard library only (`json`, `argparse`, `pathlib`, `sys`). No new dependencies.
- Every function annotated (`disallow_untyped_defs = true`); `ruff check` clean.
- The script itself is excluded from the coverage measurement it performs (it is tooling, not
  library code) but IS covered by its own tests to 100%.
- Must not import `nifty_quant` — it has to run even when the package is broken.
