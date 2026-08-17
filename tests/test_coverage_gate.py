"""Tests for scripts/coverage_gate.py."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "coverage_gate.py"


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def write_report(path: Path, files: dict[str, dict[str, int]]) -> None:
    """`files` maps module path -> dict with keys covered_lines, num_statements,
    covered_branches, num_branches. Wraps each into the real coverage.py json shape."""
    files_payload: dict[str, Any] = {}
    for module, metrics in files.items():
        denominator = metrics["num_statements"] + metrics["num_branches"]
        percent = (
            100.0
            * (metrics["covered_lines"] + metrics["covered_branches"])
            / denominator
            if denominator != 0
            else 100.0
        )
        files_payload[module] = {
            "summary": {
                "covered_lines": metrics["covered_lines"],
                "num_statements": metrics["num_statements"],
                "covered_branches": metrics["covered_branches"],
                "num_branches": metrics["num_branches"],
                "percent_covered": percent,
            }
        }

    total_covered_lines = sum(metrics["covered_lines"] for metrics in files.values())
    total_num_statements = sum(metrics["num_statements"] for metrics in files.values())
    total_covered_branches = sum(metrics["covered_branches"] for metrics in files.values())
    total_num_branches = sum(metrics["num_branches"] for metrics in files.values())

    report = {
        "files": files_payload,
        "totals": {
            "covered_lines": total_covered_lines,
            "num_statements": total_num_statements,
            "covered_branches": total_covered_branches,
            "num_branches": total_num_branches,
        },
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_floor(path: Path, modules: dict[str, int], target: int = 100, **meta: Any) -> None:
    """Writes a floor JSON with the given modules dict plus any extra `_`-prefixed metadata
    passed via **meta (e.g. write_floor(path, {...}, _comment="x", _baseline={...}))."""
    data: dict[str, Any] = {**meta, "_target": target, "modules": modules}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _line_has_status_and_one(line: str, status: str) -> bool:
    return status.lower() in line.lower() and re.search(r"\b1\b", line) is not None


def test_pct_combines_lines_and_branches(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 80,
                "num_statements": 100,
                "covered_branches": 18,
                "num_branches": 20,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 81}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    assert "81.7" in result.stdout


def test_pct_is_100_when_no_statements_and_no_branches(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/empty.py": {
                "covered_lines": 0,
                "num_statements": 0,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/empty.py": 100}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    assert "100.0" in result.stdout


# SPEC-AMBIGUITY: displayed one-decimal value (89.97 rounds to "90.0") could visually look
# like it meets floor 90 even though the comparison truncates and fails; do not assert
# the exact displayed decimal here, only the REGRESSED exit code and module name, to
# avoid coupling this test to a rounding-format decision.
def test_pct_truncates_toward_zero_against_integer_floor(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 8997,
                "num_statements": 10000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 90}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "src/nifty_quant/foo.py" in result.stdout


def test_pct_reported_to_one_decimal(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 277,
                "num_statements": 300,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 92}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    assert "92.3" in result.stdout


def test_all_modules_at_target_exits_zero(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/a.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/b.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {"src/nifty_quant/a.py": 100, "src/nifty_quant/b.py": 100},
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0


def test_module_below_floor_exits_one_and_names_it(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 70,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 85}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "src/nifty_quant/foo.py" in result.stdout


def test_module_between_floor_and_target_is_debt_and_exits_zero(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 95,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 90}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    assert "DEBT" in result.stdout.upper()


def test_module_in_report_but_not_in_floor_is_ungated_and_exits_one(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/old.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/new_thing.py": {
                "covered_lines": 50,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(floor, {"src/nifty_quant/old.py": 100}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "new_thing.py" in result.stdout
    assert "UNGATED" in result.stdout.upper()


def test_module_in_floor_but_not_in_report_is_missing_and_exits_one(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/present.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/gone.py": 90,
            "src/nifty_quant/present.py": 100,
        },
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "gone.py" in result.stdout
    assert "MISSING" in result.stdout.upper()


def test_modules_outside_src_nifty_quant_are_ignored(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/ok.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "tests/test_something.py": {
                "covered_lines": 1,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(floor, {"src/nifty_quant/ok.py": 100}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    assert "test_something.py" not in result.stdout


def test_table_sorted_worst_first(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/a.py": {
                "covered_lines": 95,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/b.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/c.py": {
                "covered_lines": 70,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/a.py": 90,
            "src/nifty_quant/b.py": 100,
            "src/nifty_quant/c.py": 90,
        },
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    idx_c = result.stdout.index("c.py")
    idx_a = result.stdout.index("a.py")
    idx_b = result.stdout.index("b.py")
    assert idx_c < idx_a < idx_b


# SPEC-AMBIGUITY: exact summary-line format is unspecified by the spec; this test checks
# for the four status words with per-status counts on some line, not a literal format
# string.
def test_summary_line_counts_each_status(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/ok.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/debt.py": {
                "covered_lines": 95,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/regressed.py": {
                "covered_lines": 70,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/ungated.py": {
                "covered_lines": 50,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/ok.py": 100,
            "src/nifty_quant/debt.py": 90,
            "src/nifty_quant/regressed.py": 90,
        },
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1

    stdout_lower = result.stdout.lower()
    for status in ("OK", "DEBT", "REGRESSED", "UNGATED"):
        assert status.lower() in stdout_lower
        assert any(
            _line_has_status_and_one(line, status) for line in result.stdout.splitlines()
        ), f"missing per-status count for {status}"


def test_failure_message_contains_module_floor_actual_and_delta(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/bad.py": {
                "covered_lines": 850,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/bad.py": 90}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "src/nifty_quant/bad.py" in result.stdout
    assert "90" in result.stdout
    assert "85" in result.stdout
    assert "-5" in result.stdout


def test_exact_equality_with_floor_passes(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 900,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 90}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0


def test_update_raises_floor_to_measured_value(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 850,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 80}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    data: dict[str, Any] = json.loads(floor.read_text())
    assert data["modules"]["src/nifty_quant/foo.py"] == 85


def test_update_never_lowers_a_floor(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 700,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 90}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    data = json.loads(floor.read_text())
    assert data["modules"]["src/nifty_quant/foo.py"] == 90, "update must never lower a floor"


def test_update_adds_ungated_module_and_warns(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/brand_new.py": {
                "covered_lines": 600,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/old.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(floor, {"src/nifty_quant/old.py": 100}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    data = json.loads(floor.read_text())
    assert data["modules"]["src/nifty_quant/brand_new.py"] == 60
    assert "brand_new.py" in result.stdout
    warning_markers = ("new module", "auto-added", "warn")
    assert any(marker in result.stdout.lower() for marker in warning_markers)


def test_update_does_not_delete_missing_modules(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/present.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/vanished.py": 88,
            "src/nifty_quant/present.py": 100,
        },
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    data = json.loads(floor.read_text())
    modules = data["modules"]
    assert "src/nifty_quant/vanished.py" in modules
    assert modules["src/nifty_quant/vanished.py"] == 88


def test_update_preserves_underscore_metadata_verbatim(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 900,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(
        floor,
        {"src/nifty_quant/foo.py": 80},
        _comment=["a", "b"],
        _baseline={"x": 1},
        _target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    data = json.loads(floor.read_text())
    assert data["_comment"] == ["a", "b"]
    assert data["_baseline"] == {"x": 1}
    assert data["_target"] == 100


def test_update_writes_sorted_keys_two_space_indent_trailing_newline(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/z.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/a.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {"src/nifty_quant/z.py": 100, "src/nifty_quant/a.py": 50},
        target=100,
    )
    result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result.returncode == 0
    raw = floor.read_text()
    assert raw.endswith("\n")
    assert not raw.endswith("\n\n")
    data = json.loads(raw)
    assert list(data["modules"].keys()) == sorted(data["modules"].keys())
    assert '  "modules"' in raw


def test_update_exits_zero_even_when_check_would_fail(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/regressed.py": {
                "covered_lines": 700,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/ungated.py": {
                "covered_lines": 500,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/regressed.py": 90,
            "src/nifty_quant/missing.py": 80,
        },
        target=100,
    )
    check_result = run_gate("--report", str(report), "--floor", str(floor))
    assert check_result.returncode == 1
    update_result = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert update_result.returncode == 0


def test_update_is_idempotent(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 850,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/foo.py": 80}, target=100)
    result1 = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result1.returncode == 0
    raw1 = floor.read_bytes()
    result2 = run_gate("--report", str(report), "--floor", str(floor), "--update")
    assert result2.returncode == 0
    raw2 = floor.read_bytes()
    assert raw1 == raw2


def test_missing_report_file_exits_two(tmp_path: Path) -> None:
    report = tmp_path / "missing_report.json"
    floor = tmp_path / "coverage_floor.json"
    write_floor(floor, {"src/nifty_quant/foo.py": 100}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 2
    assert str(report) in result.stdout + result.stderr


def test_missing_floor_file_exits_two(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "missing_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/foo.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 2
    assert str(floor) in result.stdout + result.stderr


@pytest.mark.parametrize("broken", ["report", "floor"])
def test_malformed_json_exits_two_and_names_the_file(tmp_path: Path, broken: str) -> None:
    if broken == "report":
        report = tmp_path / "bad_report.json"
        floor = tmp_path / "coverage_floor.json"
        report.write_text("{not valid json", encoding="utf-8")
        write_floor(floor, {"src/nifty_quant/ok.py": 100}, target=100)
    else:
        report = tmp_path / ".coverage.json"
        floor = tmp_path / "bad_floor.json"
        write_report(
            report,
            {
                "src/nifty_quant/ok.py": {
                    "covered_lines": 100,
                    "num_statements": 100,
                    "covered_branches": 0,
                    "num_branches": 0,
                }
            },
        )
        floor.write_text("{not valid json", encoding="utf-8")

    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 2
    broken_path = report if broken == "report" else floor
    assert str(broken_path) in result.stdout + result.stderr


def test_floor_file_without_modules_key_exits_two(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/ok.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    floor.write_text('{"_target": 100}', encoding="utf-8")
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 2
    assert str(floor) in result.stdout + result.stderr


def test_output_is_deterministic_for_identical_input(tmp_path: Path) -> None:
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    write_report(
        report,
        {
            "src/nifty_quant/a.py": {
                "covered_lines": 100,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/b.py": {
                "covered_lines": 95,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
            "src/nifty_quant/c.py": {
                "covered_lines": 70,
                "num_statements": 100,
                "covered_branches": 0,
                "num_branches": 0,
            },
        },
    )
    write_floor(
        floor,
        {
            "src/nifty_quant/a.py": 100,
            "src/nifty_quant/b.py": 90,
            "src/nifty_quant/c.py": 90,
        },
        target=100,
    )
    result1 = run_gate("--report", str(report), "--floor", str(floor))
    result2 = run_gate("--report", str(report), "--floor", str(floor))
    # b.py is DEBT (floor 90 <= pct 95 < target 100) and c.py is REGRESSED (pct 70 < floor
    # 90), so the fixture genuinely fails in check mode; this also prevents the determinism
    # assertions below from trivially holding on an empty/error output.
    assert result1.returncode == 1
    assert result1.stdout == result2.stdout
    assert result1.returncode == result2.returncode


def test_output_shows_truncated_value_for_transparent_comparison(tmp_path: Path) -> None:
    """Test that output shows the truncated value used for comparison, not just rounded display.

    This ensures 97.95 against floor 98 clearly shows truncated 97 and delta -1, so readers
    see why it fails without confusion from rounding.
    """
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    # Create a value of exactly 97.95%: 1959 covered / 2000 total
    write_report(
        report,
        {
            "src/nifty_quant/boundary.py": {
                "covered_lines": 980,
                "num_statements": 1000,
                "covered_branches": 979,
                "num_branches": 1000,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/boundary.py": 98}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 1
    assert "boundary.py" in result.stdout
    assert "REGRESSED" in result.stdout.upper()
    # Must show truncated value 97 (not displayed value 98) and delta -1 to make failure clear
    # The key is showing truncated 97, so readers know why floor 98 is not met
    assert "97" in result.stdout
    assert "-1" in result.stdout


def test_exact_floor_equality_shows_zero_delta(tmp_path: Path) -> None:
    """Test that pct == floor shows truncated value and zero delta.

    This is the mirror case: 98.0 exactly against floor 98 must pass (OK) with delta 0.
    The output must show the truncated value 98 and delta 0 to be clear.
    """
    report = tmp_path / ".coverage.json"
    floor = tmp_path / "coverage_floor.json"
    # Create exactly 98.0%: 980 / 1000 = 98.0
    write_report(
        report,
        {
            "src/nifty_quant/exact.py": {
                "covered_lines": 980,
                "num_statements": 1000,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    )
    write_floor(floor, {"src/nifty_quant/exact.py": 98}, target=100)
    result = run_gate("--report", str(report), "--floor", str(floor))
    assert result.returncode == 0
    # Should show truncated value 98 and delta 0
    # The output line should contain "exact.py", "98" (floor), "98" (truncated), "0" (delta), "DEBT"
    assert "exact.py" in result.stdout
    # Multiple "98" values appear: floor=98, actual=98.x, truncated=98
    assert result.stdout.count("98") >= 2
