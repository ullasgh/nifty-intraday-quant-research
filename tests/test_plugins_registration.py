import nifty_quant.strategy.plugins  # noqa: F401  (imported for its registration side effect)
from nifty_quant.strategy import registry


def test_builtin_plugins_register_on_import():
    assert "volume_breakout" in registry.available()
    assert "xsec_zscore" in registry.available()

    volume_breakout = registry.get("volume_breakout")
    xsec_zscore = registry.get("xsec_zscore")

    assert volume_breakout.name == "volume_breakout"
    assert xsec_zscore.name == "xsec_zscore"
