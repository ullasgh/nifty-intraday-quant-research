"""Additional coverage tests for src/nifty_quant/config.py.

Targets specific branches not exercised by tests/test_config.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from nifty_quant.config import RunConfig, canonical_json, load_yaml


def test_load_yaml_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml(path)


def test_load_yaml_non_mapping_top_level_raises_valueerror(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")

    with pytest.raises(ValueError, match="mapping/dict"):
        load_yaml(path)


def test_canonical_json_accepts_basemodel_instance_directly() -> None:
    cfg = RunConfig(
        strategy="s",
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
    )

    expected = (
        '{"capital":10000000.0,"costs":"nse_intraday_default",'
        '"decision_latency_bars":0,"end":"2020-12-31","params":{},"seed":0,'
        '"square_off_time":"15:20","start":"2020-01-01","strategy":"s",'
        '"universe":"all_equity"}'
    )

    assert canonical_json(cfg) == expected
    assert canonical_json(cfg) == canonical_json(cfg.model_dump(mode="python"))


def test_canonical_json_non_serializable_value_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="type set is not JSON serializable"):
        canonical_json({"a": {1, 2, 3}})
