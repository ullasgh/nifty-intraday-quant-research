from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nifty_quant.config import RunConfig, canonical_json, config_hash, load_run_config, load_yaml


def test_run_config_unknown_extra_key_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        RunConfig(
            strategy="s",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            bogus_field="x",
        )


def test_config_hash_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_hex_length_and_charset() -> None:
    h = config_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_changes_when_value_changes() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_canonical_json_date_iso() -> None:
    assert "2024-01-01" in canonical_json({"d": date(2024, 1, 1)})


def test_canonical_json_order_independent() -> None:
    assert canonical_json({"x": 1, "y": 2}) == canonical_json({"y": 2, "x": 1})


def test_load_run_config_valid_minimal(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    data = {
        "strategy": "momentum",
        "start": "2020-01-01",
        "end": "2020-12-31",
    }
    path.write_text(yaml.safe_dump(data))

    cfg = load_run_config(path)

    assert isinstance(cfg, RunConfig)
    assert cfg.strategy == "momentum"
    assert cfg.start == date(2020, 1, 1)
    assert cfg.end == date(2020, 12, 31)
    assert cfg.capital == 1e7
    assert cfg.seed == 0


def test_load_run_config_missing_required_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"start": "2020-01-01", "end": "2020-12-31"}))

    with pytest.raises(ValidationError):
        load_run_config(path)
