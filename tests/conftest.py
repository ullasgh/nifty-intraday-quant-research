"""Suite-wide fixtures: guard strictness, and holdout-lock isolation."""

import json
import os
from pathlib import Path

import pytest

from nifty_quant import guards


@pytest.fixture(autouse=True)
def _restore_guard_strictness():
    saved = guards.get_strictness()
    yield
    guards.set_strictness(saved)


# Deliberately absurd. No synthetic calendar in this suite reaches 2099, so nothing
# intersects it and the refusal never fires by accident. A real-looking date would make
# the isolation invisible the next time someone extends a fixture calendar.
_NO_HOLDOUT = {
    "holdout_start": "2099-01-01",
    "holdout_end": "2099-12-31",
    "count": 0,
    "log": [],
}


@pytest.fixture(scope="session")
def _real_holdout_boundary():
    """The TRUE production holdout boundary, derived once per session from the real calendar.

    Used only for tests marked `holdout_aware`. Deriving it (rather than hard-coding
    "2025-08-14") keeps it correct as the calendar grows, and it is the same derivation the
    marked tests perform for themselves.
    """
    import tempfile

    from nifty_quant.calendar import TradingCalendar
    from nifty_quant.research.splits import HoldoutLock

    scratch = Path(tempfile.mkdtemp(prefix="nq_real_boundary_")) / "lock.json"
    dates = TradingCalendar.from_index_bars("NIFTY50").session_dates()
    start, end = HoldoutLock(path=scratch).holdout_range(dates)
    return {
        "holdout_start": start.isoformat(),
        "holdout_end": end.isoformat(),
        "count": 0,
        "log": [],
    }


@pytest.fixture(autouse=True)
def _isolated_holdout_lock(request, tmp_path, monkeypatch):
    """Point every test at its own holdout lock, pre-seeded to a window nothing reaches.

    `nq walkforward` refuses splits that intersect the stored holdout. Roughly fifteen
    pre-existing tests drive walkforward to exercise CLI plumbing, PBO and returns
    persistence on synthetic calendars SHORTER than `holdout_months = 12`. They already
    redirect `settings.RESULTS_ROOT` to their own `tmp_path`, so they find NO lock there,
    first-ever initialisation manufactures a boundary spanning their whole window, and
    their own splits necessarily intersect it. They fail on a holdout they were never about.

    Seeding a far-future boundary states the truth -- these tests have no holdout. That is
    honest in a way neither relaxing the guard nor sprinkling `--allow-holdout` through
    twenty files would be: `--allow-holdout` asserts "I intend to read the holdout", and
    that is false for every one of them.

    Two details that a previous attempt at this fixture got wrong, recorded so they are not
    reintroduced:

    * **Seed into `tmp_path`, do not replace `default_holdout_lock_path`.** Those tests set
      `RESULTS_ROOT = tmp_path` themselves, and `tmp_path` here is the SAME directory the
      test receives, so the seed lands exactly where the code will look. Replacing the
      helper function instead overrode the redirection those tests perform deliberately,
      which broke the holdout suites in full-suite order while passing in isolation.
    * **Also default `RESULTS_ROOT` to `tmp_path`.** A test that reaches
      `default_holdout_lock_path()` without redirecting would otherwise hit the real
      `results/holdout_lock.json`, whose one-shot read counter guards a genuinely unspent
      out-of-sample window. Spending it from a stray test run is unrecoverable -- there is
      no second first look at out-of-sample data.

    Tests that ARE about the holdout seed their own lock in the test body, which runs after
    this fixture and overwrites the far-future seed. That is intended.
    """
    from nifty_quant import settings

    monkeypatch.setattr(settings, "RESULTS_ROOT", tmp_path)

    if request.node.get_closest_marker("holdout_aware") is not None:
        # This test is ABOUT holdout protection, so it must see the real boundary rather
        # than the far-future stand-in that disarms the guard. It still gets its own file:
        # the production lock's one-shot counter must never be reachable from a test run.
        state = request.getfixturevalue("_real_holdout_boundary")
    else:
        state = _NO_HOLDOUT

    (tmp_path / "holdout_lock.json").write_text(json.dumps(state), encoding="utf-8")


# Tests-first suites whose implementation does not exist yet. They are RED BY DESIGN and are
# the deliverable of the spec-then-tests workflow, not a broken build.
#
# Setting NQ_SKIP_PENDING=1 excludes them, so the gate can be run against the IMPLEMENTED
# surface while these are in flight. Without the flag they collect and fail, which is the
# correct default -- a pending suite must never become invisible by accident.
#
# Delete each entry as its implementation lands. An entry left here after implementation
# would silently stop gating real code, so the list is deliberately explicit rather than a
# wildcard.
_PENDING_SUITES = (
    # Phase D. specs/feature_layer.md -- gap-stitched price path, breakout strength,
    # Garman-Klass / Rogers-Satchell, efficiency ratio, and research/ic.py. No implementation
    # written yet, so these are RED by design.
    "test_feature_layer_a.py",
    "test_feature_layer_b.py",
)

# Removed as their implementations landed and went green, which is the point of the list being
# explicit: test_research_contract_{a,b}, test_tca_record_{a,b}, test_overlap_se_{a,b},
# test_lens_verdict_integrity_{a,b}. An entry left here after implementation would silently stop
# gating real code -- exactly the failure this list is shaped to prevent.

collect_ignore = list(_PENDING_SUITES) if os.environ.get("NQ_SKIP_PENDING") == "1" else []
