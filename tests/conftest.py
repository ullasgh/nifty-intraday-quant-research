"""Snapshot and restore global guard strictness around every test."""

import pytest

from nifty_quant import guards


@pytest.fixture(autouse=True)
def _restore_guard_strictness():
    saved = guards.get_strictness()
    yield
    guards.set_strictness(saved)
