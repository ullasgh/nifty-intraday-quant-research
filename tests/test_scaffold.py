import importlib

import pytest

import nifty_quant
from nifty_quant import settings

STUB_MODULES = [
    "nifty_quant",
    "nifty_quant.settings",
    "nifty_quant.cli",
    "nifty_quant.config",
    "nifty_quant.calendar",
    "nifty_quant.data",
    "nifty_quant.data.manifest",
    "nifty_quant.data.panel",
    "nifty_quant.data.panel_builder",
    "nifty_quant.data.checkpoints",
    "nifty_quant.data.validate",
    "nifty_quant.universe",
    "nifty_quant.universe.static",
    "nifty_quant.features",
    "nifty_quant.features.core",
    "nifty_quant.features.persistence",
    "nifty_quant.features.calibrate",
    "nifty_quant.strategy",
    "nifty_quant.strategy.base",
    "nifty_quant.strategy.registry",
    "nifty_quant.strategy.plugins",
    "nifty_quant.execution",
    "nifty_quant.execution.costs",
    "nifty_quant.execution.fills",
    "nifty_quant.backtest",
    "nifty_quant.backtest.engine",
    "nifty_quant.backtest.portfolio",
    "nifty_quant.backtest.metrics",
    "nifty_quant.research",
    "nifty_quant.research.splits",
    "nifty_quant.research.registry",
    "nifty_quant.research.sweep",
]


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_version_is_string() -> None:
    assert isinstance(nifty_quant.__version__, str)


def test_settings_paths_exist() -> None:
    assert settings.DATA_ROOT.exists()
    assert settings.MANIFEST_PATH.exists()


def test_bars_1m_has_at_least_150_symbols() -> None:
    assert settings.BARS_1M.exists()
    symbols = [p for p in settings.BARS_1M.iterdir() if p.is_dir()]
    assert len(symbols) >= 150
