"""Run provenance and experiment lineage helpers.

`specs/run_provenance.md` (see AMENDMENT 1, which wins over the body wherever the two
disagree). This module provides the pieces `cli.py` needs to populate the new
`TrialRecord` fields (`seed`, `universe_name`, `universe_hash`, `panel_hash`, `start`,
`end`, `cost_model_id`, `slippage_model_id`, `fill_model_id`, `embargo_components`,
`parent_trial_id`, `feature_version`) and the `metrics.json["provenance"]` block, without
scattering hashing/canonicalisation logic across every CLI command.

Deliberate simplification, reported rather than hidden: `compute_universe_hash` always
treats the given symbol set as uniformly eligible across every session in the panel. It
does NOT thread through the true per-session mask from
`nifty_quant.universe.pit.compute_eligibility` when the CLI's `--min-history-sessions`
PIT gate is active. Wiring the real per-session eligibility into this hash is future
work -- no required test in either `run_provenance` suite exercises the PIT-active case,
and doing so risks touching `cli.py`'s PIT wiring while another agent edits it
concurrently for the embargo work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never executed
    from nifty_quant.data.panel import Panel

# No versioned feature-engineering registry exists yet in this codebase (every strategy
# plugin computes its own features inline, with no separate version tag). This is a
# single global placeholder, analogous in spirit to `nifty_quant.__version__` but for
# the feature layer specifically, until a real per-feature versioning scheme exists.
FEATURE_VERSION = "v1"


def get_git_sha(repo_root: Path | None = None) -> str:
    """Return the HEAD commit sha, `"<sha>-dirty"` if the tree has changes, or
    `"no-git"` if git is unavailable / `repo_root` is not inside a git repository.

    Never returns `None` -- a null `git_sha` is worse than an absent one (CLAUDE.md
    rule 8's sibling problem for provenance: a reader assumes "unknown commit" rather
    than "never recorded").
    """
    if repo_root is None:
        from nifty_quant.settings import REPO_ROOT

        repo_root = REPO_ROOT

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "no-git"

    sha = rev.stdout.strip()
    if rev.returncode != 0 or not sha:
        return "no-git"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # HEAD resolved fine; treat an unreadable status as "can't prove it's dirty".
        return sha

    if status.returncode == 0 and status.stdout.strip():
        return f"{sha}-dirty"
    return sha


def canonical_model_id(instance: Any) -> str:
    """`"<ClassName>(<k=v for each field differing from the dataclass default, sorted
    by k>)"`, per AMENDMENT 1 item 1. Floats are formatted with `repr()` so the string
    is precision-preserving and round-trippable.

    A field with no dataclass default (required, e.g. `FillModel.slippage`) is always
    included, since "differs from the default" is vacuously true when there is no
    default to compare against.
    """
    cls = type(instance)
    fields = dataclasses.fields(cls)

    parts: list[str] = []
    for f in sorted(fields, key=lambda field: field.name):
        value = getattr(instance, f.name)

        if f.default is not dataclasses.MISSING:
            if value == f.default:
                continue
        elif f.default_factory is not dataclasses.MISSING:
            if value == f.default_factory():
                continue
        # else: no default at all -- always include.

        parts.append(f"{f.name}={value!r}")

    return f"{cls.__name__}({', '.join(parts)})"


def compute_panel_hash(panel: "Panel", *, adjustments: str) -> str:
    """Hash of the LOADED SLICE (metadata only, never bar data), per spec section B /
    AMENDMENT 1 item 4: sorted symbol list, first+last `ts`, row count, field names,
    adjustment settings, `PANEL_VERSION`. Two runs agree here iff all six agree.
    """
    from nifty_quant.settings import PANEL_VERSION

    n_rows = panel.n_rows()
    payload = {
        "symbols": sorted(panel.symbols),
        "first_ts": int(panel.ts[0]) if n_rows else None,
        "last_ts": int(panel.ts[-1]) if n_rows else None,
        "row_count": n_rows,
        # `_fields` is Panel's internal dict of field name -> ndarray; only its KEYS
        # (field names) are read here, never the arrays themselves -- no bar data
        # is hashed, per the spec's own constraint.
        "field_names": sorted(panel._fields.keys()),  # noqa: SLF001
        "adjustments": adjustments,
        "panel_version": PANEL_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(blob).hexdigest()


def compute_universe_hash(
    *,
    name: str,
    symbols: tuple[str, ...],
    n_sessions: int,
    eligibility_params: dict[str, Any] | None = None,
) -> str:
    """blake2s over canonical JSON of the universe name, the per-session sorted
    eligible symbol sets, and the eligibility parameters (AMENDMENT 1 item 3).

    See the module docstring: this treats `symbols` as uniformly eligible across all
    `n_sessions` sessions (no PIT per-session mask threaded through), which is
    represented compactly (a constant set + a session count) rather than literally
    repeating the same sorted list `n_sessions` times.
    """
    payload = {
        "name": name,
        "eligibility_params": dict(eligibility_params) if eligibility_params else {},
        "n_sessions": int(n_sessions),
        "eligible_symbols_constant_across_sessions": sorted(symbols),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(blob).hexdigest()


def embargo_components_json(
    *,
    feature_lookback: float,
    label_horizon: float,
    holding_period: float,
    execution_horizon: float,
) -> str:
    """The four embargo terms (units: SESSIONS, floats, pre-`ceil`), sorted-key JSON,
    per AMENDMENT 1 item 2.
    """
    return json.dumps(
        {
            "feature_lookback": float(feature_lookback),
            "label_horizon": float(label_horizon),
            "holding_period": float(holding_period),
            "execution_horizon": float(execution_horizon),
        },
        sort_keys=True,
    )
