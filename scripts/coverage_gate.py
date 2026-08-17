#!/usr/bin/env python3
"""Coverage ratchet gate: enforces per-module floor that can only rise."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict


class Summary(TypedDict):
    """Coverage summary from a coverage.py JSON report."""

    covered_lines: int
    num_statements: int
    covered_branches: int
    num_branches: int


class ModuleResult(TypedDict):
    """Result of evaluating one module against its floor."""

    module: str
    floor: int
    actual: float
    truncated: int
    status: str


def load_report(path: Path) -> dict[str, Summary]:
    """Load coverage.py JSON report and extract module summaries."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: report file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(2)

    if "files" not in data:
        print(f"ERROR: report file {path} missing 'files' key", file=sys.stderr)
        sys.exit(2)

    summaries: dict[str, Summary] = {}
    for module, info in data["files"].items():
        if "summary" in info:
            summaries[module] = info["summary"]
    return summaries


def load_floor(path: Path) -> tuple[dict[str, int], int, dict[str, Any]]:
    """Load floor file, return (modules dict, target, full data)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: floor file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(2)

    if "modules" not in data:
        print(f"ERROR: floor file {path} missing 'modules' key", file=sys.stderr)
        sys.exit(2)

    target = data.get("_target", 100)
    return data["modules"], target, data


def compute_percentage(summary: Summary) -> float:
    """Compute combined line+branch coverage percentage."""
    covered = summary["covered_lines"] + summary["covered_branches"]
    total = summary["num_statements"] + summary["num_branches"]
    if total == 0:
        return 100.0
    return 100.0 * covered / total


def truncate(pct: float) -> int:
    """Truncate percentage toward zero for floor comparison."""
    return int(pct)


def evaluate_modules(
    report: dict[str, Summary],
    floor_modules: dict[str, int],
    target: int,
) -> list[ModuleResult]:
    """Evaluate each module and return results sorted worst-first."""
    results: list[ModuleResult] = []

    # Only consider modules under src/nifty_quant/
    nifty_modules_in_report = {
        m: s for m, s in report.items() if m.startswith("src/nifty_quant/")
    }
    nifty_modules_in_floor = {
        m: f for m, f in floor_modules.items() if m.startswith("src/nifty_quant/")
    }

    # Check each module in the report
    for module, summary in nifty_modules_in_report.items():
        actual_pct = compute_percentage(summary)
        truncated_pct = truncate(actual_pct)

        if module not in nifty_modules_in_floor:
            status = "UNGATED"
            floor_val = -1
        else:
            floor_val = nifty_modules_in_floor[module]
            if truncated_pct < floor_val:
                status = "REGRESSED"
            elif actual_pct >= target:
                status = "OK"
            else:
                status = "DEBT"

        results.append(
            {
                "module": module,
                "floor": floor_val,
                "actual": actual_pct,
                "truncated": truncated_pct,
                "status": status,
            }
        )

    # Check for missing modules
    for module, floor_val in nifty_modules_in_floor.items():
        if module not in nifty_modules_in_report:
            results.append(
                {
                    "module": module,
                    "floor": floor_val,
                    "actual": -1.0,
                    "truncated": -1,
                    "status": "MISSING",
                }
            )

    # Sort worst-first: failures first, by delta (most negative = worst)
    def sort_key(r: ModuleResult) -> tuple[int, float, str]:
        status_order = {"REGRESSED": 0, "UNGATED": 1, "MISSING": 2, "DEBT": 3, "OK": 4}
        if r["status"] == "MISSING":
            delta = -999.0
        elif r["actual"] < 0:
            delta = -999.0
        else:
            delta = r["actual"] - r["floor"]
        return (status_order[r["status"]], delta, r["module"])

    results.sort(key=sort_key)

    return results


def format_table(results: list[ModuleResult]) -> str:
    """Format results as a table."""
    lines: list[str] = []

    for r in results:
        if r["status"] == "MISSING":
            line = (
                f"{r['module']:<50} {r['floor']:>6} {'N/A':>8} {'N/A':>5} {'N/A':>6} "
                f"{r['status']:<10}"
            )
        else:
            delta = r["truncated"] - r["floor"]
            line = (
                f"{r['module']:<50} {r['floor']:>6} {r['actual']:>7.1f} {r['truncated']:>5} "
                f"{delta:>6} {r['status']:<10}"
            )
        lines.append(line)

    return "\n".join(lines)


def format_summary(results: list[ModuleResult]) -> str:
    """Format summary line with status counts."""
    status_counts = {
        "OK": sum(1 for r in results if r["status"] == "OK"),
        "DEBT": sum(1 for r in results if r["status"] == "DEBT"),
        "REGRESSED": sum(1 for r in results if r["status"] == "REGRESSED"),
        "UNGATED": sum(1 for r in results if r["status"] == "UNGATED"),
        "MISSING": sum(1 for r in results if r["status"] == "MISSING"),
    }
    ok_count = status_counts["OK"]
    return (
        f"OK={ok_count} DEBT={status_counts['DEBT']} "
        f"REGRESSED={status_counts['REGRESSED']} UNGATED={status_counts['UNGATED']} "
        f"MISSING={status_counts['MISSING']} | {ok_count} at target"
    )


def check_mode(
    report: dict[str, Summary],
    floor_modules: dict[str, int],
    target: int,
) -> tuple[int, str]:
    """Check mode: verify coverage meets floors. Return (exit_code, output)."""
    results = evaluate_modules(report, floor_modules, target)
    output_lines = [format_table(results), format_summary(results)]
    output = "\n".join(output_lines)

    failures = [
        r for r in results if r["status"] in ("REGRESSED", "UNGATED", "MISSING")
    ]
    exit_code = 1 if failures else 0
    return (exit_code, output)


def update_mode(
    report: dict[str, Summary],
    floor_modules: dict[str, int],
    target: int,
    floor_path: Path,
    all_floor_data: dict[str, Any],
) -> tuple[int, str]:
    """Update mode: raise floors and add ungated modules."""
    # Only consider modules under src/nifty_quant/
    nifty_modules_in_report = {
        m: s for m, s in report.items() if m.startswith("src/nifty_quant/")
    }

    warned: list[str] = []
    updated_modules = dict(floor_modules)

    # Update existing modules (raise only) and add new modules
    for module, summary in nifty_modules_in_report.items():
        if module.startswith("src/nifty_quant/"):
            actual_pct = compute_percentage(summary)
            truncated_pct = truncate(actual_pct)

            if module in updated_modules:
                # Raise only
                if truncated_pct > updated_modules[module]:
                    updated_modules[module] = truncated_pct
            else:
                # Add ungated module
                updated_modules[module] = truncated_pct
                warned.append(module)

    # Preserve metadata and write
    updated_data = {
        k: v for k, v in all_floor_data.items() if k.startswith("_")
    }
    updated_data["modules"] = {
        k: updated_modules[k] for k in sorted(updated_modules.keys())
    }

    floor_path.write_text(json.dumps(updated_data, indent=2) + "\n", encoding="utf-8")

    # Format output
    output_lines: list[str] = []
    if warned:
        output_lines.append("WARNING: new modules auto-added below target:")
        for m in warned:
            output_lines.append(f"  {m} at {updated_modules[m]}%")
    output_lines.append("Floor file updated.")

    return (0, "\n".join(output_lines))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Coverage ratchet gate")
    parser.add_argument(
        "--report", default=".coverage.json", help="Coverage report path"
    )
    parser.add_argument(
        "--floor", default="coverage_floor.json", help="Floor file path"
    )
    parser.add_argument("--update", action="store_true", help="Update floor file")
    parser.add_argument("--target", type=int, help="Target coverage percentage")

    args = parser.parse_args()

    report_path = Path(args.report)
    floor_path = Path(args.floor)

    # Load files
    report = load_report(report_path)
    floor_modules, default_target, all_floor_data = load_floor(floor_path)
    target = args.target if args.target is not None else default_target

    if args.update:
        exit_code, output = update_mode(report, floor_modules, target, floor_path, all_floor_data)
    else:
        exit_code, output = check_mode(report, floor_modules, target)

    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
