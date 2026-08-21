"""Concurrency regression test: >=4 processes calling load_panel() on a COLD cache.

Distinct from the existing ``test_concurrent_load_panel_race`` in ``test_panel.py``,
which pre-warms the per-year raw cache in-process before racing two processes on the
*re*-materialization path only (its docstring calls the first-ever-build race
"out of scope"). This file targets exactly that first-build race: ``load_panel``
unconditionally calls ``panel_builder.build_panel`` (writes the per-year raw cache
directly into place with no tmp-dir/rename of its own), so N processes racing on a
completely cold cache used to observe a partially-written ``.npy`` file or a
``meta.json`` mid-rewrite by another process.

Workers are separate `subprocess` children (never `multiprocessing`/
`ProcessPoolExecutor`, whose workers are children of the pytest runner and can take
it down with them): a truncated/racing mmap can SIGBUS-kill the reading process
outright, an uncatchable hardware fault. A crash therefore shows up as a
non-zero/negative exit code on the child, asserted on below, rather than crashing
the test process.

Uses real committed `data/bars/1/...` parquet (read-only; nothing under `data/` is
ever written) restricted to a single year and 2 symbols to keep the built cache
small. `CACHE_ROOT` is overridden via the `NQ_CACHE_ROOT` env var (the only
settings override that propagates across a subprocess boundary) to a fresh
`tmp_path`, so every run starts genuinely cold.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nifty_quant.data.manifest import Manifest

N_WORKERS = 5
_YEAR = 2018

_WORKER_SCRIPT = """
import datetime
import hashlib
import json
import sys

import numpy as np

from nifty_quant.data.panel import PanelSpec, load_panel


def main() -> None:
    symbols = tuple(sys.argv[1].split(","))
    field = sys.argv[2]
    start = datetime.date.fromisoformat(sys.argv[3])
    end = datetime.date.fromisoformat(sys.argv[4])
    out_path = sys.argv[5]

    spec = PanelSpec(freq="1", fields=(field,), symbols=symbols, start=start, end=end)
    panel = load_panel(spec, memmap=True)
    arr = np.array(panel.field(field), dtype=np.float32)
    checksum = hashlib.sha256(arr.tobytes()).hexdigest()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"checksum": checksum, "shape": list(arr.shape)}, fh)


if __name__ == "__main__":
    main()
"""


def test_cold_cache_concurrent_load_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """>=4 processes racing load_panel() on a cold cache must all succeed identically."""
    try:
        manifest = Manifest.load()
    except FileNotFoundError:
        pytest.skip("Real data/MANIFEST.json not found; skipping.")

    candidates = sorted(sym for sym, cov in manifest.coverage.items() if _YEAR in cov.years)
    if len(candidates) < 2:
        pytest.skip(f"Need >=2 symbols with {_YEAR} data, found {len(candidates)}.")
    symbols = tuple(candidates[:2])
    field = "close"
    start = datetime.date(_YEAR, 1, 1)
    end = datetime.date(_YEAR, 1, 10)

    cache_root = tmp_path / "cache"
    monkeypatch.setattr("nifty_quant.settings.CACHE_ROOT", cache_root)
    assert not cache_root.exists(), "cache_root must be genuinely cold before the race"

    worker_script = tmp_path / "_worker.py"
    worker_script.write_text(_WORKER_SCRIPT, encoding="utf-8")

    env = dict(os.environ)
    env["NQ_CACHE_ROOT"] = str(cache_root)

    common_args = [",".join(symbols), field, start.isoformat(), end.isoformat()]
    out_paths = [tmp_path / f"out_{i}.json" for i in range(N_WORKERS)]

    procs = [
        subprocess.Popen(
            [sys.executable, str(worker_script), *common_args, str(out_paths[i])],
            env=env,
        )
        for i in range(N_WORKERS)
    ]

    returncodes = []
    for proc in procs:
        try:
            returncodes.append(proc.wait(timeout=90))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("a worker subprocess timed out (deadlock?) and was killed")

    for i, rc in enumerate(returncodes):
        assert rc == 0, (
            f"worker {i} crashed/failed with exit code {rc} "
            "(a SIGBUS from a racing/truncated mmap shows up as a negative code on POSIX)"
        )

    results = [json.loads(p.read_text(encoding="utf-8")) for p in out_paths]

    checksums = {r["checksum"] for r in results}
    shapes = {tuple(r["shape"]) for r in results}
    assert len(checksums) == 1, f"workers disagreed on data: {checksums}"
    assert len(shapes) == 1, f"workers disagreed on shape: {shapes}"

    # Sanity: the shared checksum must match a fresh, non-memmapped, in-process read
    # against the now-warm cache (not just "workers agree with each other" -- they
    # could all agree on a corrupt result).
    from nifty_quant.data.panel import PanelSpec, load_panel

    spec = PanelSpec(freq="1", fields=(field,), symbols=symbols, start=start, end=end)
    reference = load_panel(spec, memmap=False)
    ref_arr = np.array(reference.field(field), dtype=np.float32)
    ref_checksum = hashlib.sha256(ref_arr.tobytes()).hexdigest()

    assert next(iter(checksums)) == ref_checksum
    assert list(next(iter(shapes))) == list(ref_arr.shape)
