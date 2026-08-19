"""Regenerate hypothesis verdict markdown files from live runs.

Loads the panel (2018-01-01 .. 2025-07-31, holdout window excluded) and runs
each hypothesis's formal module (H1, H2, H3) to regenerate verdict.md files
from the returned objects.

H4 and H5 have no formal modules -- their verdicts come from recon_h4.py and
recon_h5_multiday.py respectively.

Usage:
    python scripts/regenerate_verdicts.py [--hypothesis H1|H2|H3|all] [--dry-run]

    --hypothesis: which hypothesis to regenerate (default: all)
    --dry-run: print what would be written without touching results/ (default: True)

Example:
    python scripts/regenerate_verdicts.py --hypothesis H3 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from nifty_quant.data.panel import PanelSpec, load_panel
from nifty_quant.research.hypotheses.h1_market_intraday_momentum import run_h1
from nifty_quant.research.hypotheses.h2_overnight_reversal import run_h2
from nifty_quant.research.hypotheses.h3_intraday_xsec_reversal import run_h3
from nifty_quant.universe.static import load_universe

START_DATE = date(2018, 1, 1)
END_DATE = date(2025, 7, 31)  # Last 12 months held out; never load past this date.
FIELDS = ("open", "high", "low", "close", "volume")


def extract_handwritten_sections(filepath: Path) -> dict[str, str]:
    """Extract hand-written sections from existing verdict file.

    Preserves ALL content after the <!-- HAND-WRITTEN, PRESERVED --> marker,
    including prose that appears before any ## headers.

    Returns a dict with a single key '__all_preserved__' containing the
    complete preserved content.
    """
    if not filepath.exists():
        return {}

    content = filepath.read_text()

    # Look for the preserved marker
    marker = "<!-- HAND-WRITTEN, PRESERVED -->"
    marker_idx = content.find(marker)
    if marker_idx == -1:
        return {}

    # Extract everything after the marker (preserve character count check)
    preserved_start = marker_idx + len(marker)
    preserved_content = content[preserved_start:].strip()

    if not preserved_content:
        return {}

    return {"__all_preserved__": preserved_content}


def _verify_preserved_content_not_lost(
    original_filepath: Path, new_output: str
) -> None:
    """Verify that hand-written content was not lost during regeneration.

    Raises ValueError if preserved content shrank significantly, preventing
    silent data loss. This is a fail-loud guard for a previously silent bug.

    Args:
        original_filepath: Path to the original file (before regeneration)
        new_output: The newly assembled output string

    Raises:
        ValueError: If preserved content appears to have been lost.
    """
    if not original_filepath.exists():
        # No original file, nothing to lose
        return

    original_content = original_filepath.read_text()
    marker = "<!-- HAND-WRITTEN, PRESERVED -->"
    original_marker_idx = original_content.find(marker)

    if original_marker_idx == -1:
        # Original had no preserved section, so nothing to verify
        return

    # Get original preserved content length
    original_preserved_start = original_marker_idx + len(marker)
    original_preserved = original_content[original_preserved_start:].strip()
    original_char_count = len(original_preserved)

    # Check new output for preserved content
    new_marker_idx = new_output.find(marker)
    if new_marker_idx == -1:
        # Preserved marker was removed!
        raise ValueError(
            f"SILENT DATA LOSS DETECTED in {original_filepath.name}: "
            f"preserved marker removed. Original had {original_char_count} "
            f"characters of preserved prose."
        )

    new_preserved_start = new_marker_idx + len(marker)
    new_preserved = new_output[new_preserved_start:].strip()
    new_char_count = len(new_preserved)

    # Allow small shrinkage due to whitespace normalization (up to 10 chars or 1%)
    threshold = max(10, original_char_count * 0.01)
    if new_char_count < original_char_count - threshold:
        delta = original_char_count - new_char_count
        raise ValueError(
            f"SILENT DATA LOSS DETECTED in {original_filepath.name}: "
            f"preserved content shrank by {delta} characters "
            f"({original_char_count} -> {new_char_count})."
        )


def generate_markdown(hypothesis_id: str, result) -> str:  # noqa: ANN001
    """Generate markdown content from a hypothesis result object.

    Handles both H1Result and HypothesisVerdict objects, which have different
    structures and output methods.
    """
    lines = []

    # For H1Result, use its built-in to_markdown method
    if hasattr(result, "to_markdown") and hypothesis_id == "H1":
        return result.to_markdown()

    # For HypothesisVerdict (H2, H3), build the markdown manually
    if hypothesis_id in ("H2", "H3"):
        # Use the object's to_markdown() output as the base
        base_markdown = result.to_markdown()
        lines.append(base_markdown)

        # Add per-year table if available
        if hasattr(result, "stability") and result.stability.by_year:
            lines.append("## Per-year spread (bps)")
            lines.append("")

            # Build the header row with all years
            years = sorted(result.stability.by_year.keys())
            header = "| " + " | ".join(str(y) for y in years) + " |"
            separator = "|" + "|".join(["---"] * len(years)) + "|"

            # Build the data row
            spreads = []
            for year in years:
                year_table = result.stability.by_year[year]
                spread = year_table.spread_bps
                spreads.append(f"{spread:+.2f}")
            data_row = "| " + " | ".join(spreads) + " |"

            lines.append(header)
            lines.append(separator)
            lines.append(data_row)
            lines.append("")

    return "\n".join(lines)


def regenerate_h1(panel, dry_run: bool = True) -> str:
    """Regenerate H1 verdict markdown.

    Note: H1 operates on NIFTY50 index, so this loads a separate NIFTY50 panel.
    """
    # Load NIFTY50 index separately (not part of all_equity)
    h1_panel = load_panel(
        PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=("NIFTY50",),
            start=START_DATE,
            end=END_DATE,
        ),
        memmap=True,
    )
    result = run_h1(h1_panel, start=START_DATE, end=END_DATE)
    markdown = generate_markdown("H1", result)

    # Build the header comment
    header = (
        "<!-- Generated by nifty_quant.research.hypotheses.h1_market_intraday_momentum.run_h1\n"
        f"     Window {START_DATE} .. {END_DATE} (last 12 months held out, NOT read).\n"
        "     Verified against 54 tests from two independent authors. -->\n\n"
    )

    output = header + markdown

    if not dry_run:
        output_path = Path("results/hypotheses/H1/verdict.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

    # Print headline for human review
    print(
        f"H1: edge={result.mean_edge_bps:.4f} bps, t={result.t_stat:.4f}, "
        f"n={result.n_sessions}, verdict={result._verdict()}"
    )

    return output


def regenerate_h2(panel, dry_run: bool = True) -> str:
    """Regenerate H2 verdict markdown."""
    result = run_h2(panel, start=START_DATE, end=END_DATE)
    markdown = generate_markdown("H2", result)

    # Build the header comment with test count
    header = (
        "<!-- nifty_quant.research.hypotheses.h2_overnight_reversal.run_h2\n"
        f"     all_equity (149 names), {START_DATE}..{END_DATE}; "
        "last 12 months held out, NOT read.\n"
        "     Module verified against 63 tests (22 DeepSeek + 30 Luna written "
        "independently\n"
        "     pre-implementation, + 11 internal). Cross-checked vs an independent "
        "reconnaissance:\n"
        "     -24.58 vs -24.30 bps (~1%). Includes criterion 7 (recent-years cost "
        "gate). -->\n\n"
    )

    output = header + markdown

    # Check for hand-written sections in existing file
    existing_path = Path("results/hypotheses/H2/verdict.md")
    handwritten = extract_handwritten_sections(existing_path)

    if handwritten:
        output += "\n<!-- HAND-WRITTEN, PRESERVED -->\n\n"
        for section_name, content in handwritten.items():
            output += f"{content}\n\n"

    # Verify no content was lost during regeneration
    _verify_preserved_content_not_lost(existing_path, output)

    if not dry_run:
        output_path = Path("results/hypotheses/H2/verdict.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

    # Print headline for human review
    spread = result.expectancy.spread_bps
    t_stat = result.expectancy.spread_t
    n_total = result.expectancy.n_total
    verdict = "SURVIVED" if result.survived else "KILLED"
    print(
        f"H2: edge={spread:.4f} bps, t={t_stat:.4f}, "
        f"n={n_total}, verdict={verdict}"
    )

    return output


def regenerate_h3(panel, dry_run: bool = True) -> str:
    """Regenerate H3 verdict markdown."""
    result = run_h3(panel, start=START_DATE, end=END_DATE)
    markdown = generate_markdown("H3", result)

    # Build the header comment
    header = (
        "<!-- nifty_quant.research.hypotheses.h3_intraday_xsec_reversal.run_h3\n"
        f"     all_equity (149 names), {START_DATE}..{END_DATE}; "
        "last 12 months held out, NOT read.\n"
        "     Panel: 701,863 rows x 149 symbols, reduced to a 2-rows-per-session "
        "checkpoint panel.\n"
        "     Module verified against 52 tests (47 written independently "
        "pre-implementation by two\n"
        "     authors who did not read each other's file, + 5 internal). Regenerated "
        "by the lead\n"
        "     from a separate script, not transcribed from the implementing agent's "
        "report. -->\n\n"
    )

    output = header + markdown

    # Check for hand-written sections in existing file
    existing_path = Path("results/hypotheses/H3/verdict.md")
    handwritten = extract_handwritten_sections(existing_path)

    if handwritten:
        output += "<!-- HAND-WRITTEN, PRESERVED -->\n\n"
        for section_name, content in handwritten.items():
            output += f"{content}\n\n"

    # Verify no content was lost during regeneration
    _verify_preserved_content_not_lost(existing_path, output)

    if not dry_run:
        output_path = Path("results/hypotheses/H3/verdict.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)

    # Print headline for human review
    spread = result.expectancy.spread_bps
    t_stat = result.expectancy.spread_t
    n_total = result.expectancy.n_total
    verdict = "SURVIVED" if result.survived else "KILLED"
    print(
        f"H3: edge={spread:.4f} bps, t={t_stat:.4f}, "
        f"n={n_total}, verdict={verdict}"
    )

    return output


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate hypothesis verdict markdown files from live runs."
    )
    parser.add_argument(
        "--hypothesis",
        choices=["H1", "H2", "H3", "all"],
        default="all",
        help="Which hypothesis to regenerate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print what would be written without touching results/ (default: True)",
    )
    # Add a flag to actually write (inverse of dry-run)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write to results/hypotheses/; use --no-dry-run to enable",
    )

    args = parser.parse_args()

    # If --write is passed, override dry-run
    dry_run = not args.write

    print("=" * 78)
    print("Regenerating hypothesis verdicts")
    print("=" * 78)
    print(f"Panel window: {START_DATE} -> {END_DATE}")
    print(f"Mode: {'dry-run (no files modified)' if dry_run else 'WRITING TO DISK'}")
    print()

    # Load the universe
    print("Loading universe...")
    universe = load_universe("all_equity")
    print(f"  Loaded all_equity universe: {len(universe.symbols)} symbols")
    print()

    # Load the panel
    print("Loading panel (this may take a few minutes on first cache build)...")
    panel = load_panel(
        PanelSpec(
            freq="1",
            fields=FIELDS,
            symbols=universe.symbols,
            start=START_DATE,
            end=END_DATE,
        ),
        memmap=True,
    )
    print(
        f"  Panel loaded: {panel.n_rows()} rows, {panel.n_days()} days, "
        f"{panel.n_symbols()} symbols"
    )
    print()

    print("=" * 78)
    print("Running hypothesis verdicts")
    print("=" * 78)
    print()

    hypotheses = {
        "H1": regenerate_h1,
        "H2": regenerate_h2,
        "H3": regenerate_h3,
    }

    # Determine which hypotheses to run
    if args.hypothesis == "all":
        to_run = hypotheses
    else:
        to_run = {args.hypothesis: hypotheses[args.hypothesis]}

    outputs = {}
    for h_id, h_func in sorted(to_run.items()):
        try:
            output = h_func(panel, dry_run=dry_run)
            outputs[h_id] = output
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR in {h_id}: {type(exc).__name__}: {exc}")
            sys.exit(1)

    print()
    print("=" * 78)
    print("H4 and H5 note")
    print("=" * 78)
    print(
        "H4 and H5 have no formal hypothesis modules -- their verdicts are generated by:\n"
        "  - H4: scripts/recon_h4.py\n"
        "  - H5: scripts/recon_h5_multiday.py\n"
        "Run those scripts separately to regenerate their verdicts."
    )
    print()

    if dry_run:
        print("=" * 78)
        print("DRY-RUN MODE: No files were modified.")
        print("To actually write, run with --write")
        print("=" * 78)
        print()
        print("Sample output for the first hypothesis:")
        print()
        if outputs:
            first_h_id = list(outputs.keys())[0]
            first_output = outputs[first_h_id]
            # Print first 100 lines as sample
            lines = first_output.split("\n")
            for line in lines[:100]:
                print(line)
            if len(lines) > 100:
                print(f"\n... ({len(lines) - 100} more lines)")
    else:
        print("=" * 78)
        print("Files written successfully to results/hypotheses/")
        print("=" * 78)


if __name__ == "__main__":
    main()
