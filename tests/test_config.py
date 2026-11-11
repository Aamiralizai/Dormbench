import pytest
import yaml

from dormbench.config import ConfigError, load_config


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_relative_paths_resolve_against_data_root(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "paths": {"data_root": str(tmp_path / "D"), "atlas_dir": "GSE176078"}
    }))
    cfg = load_config(cfg_file)
    assert cfg.paths.atlas_dir == tmp_path / "D" / "GSE176078"


def test_absolute_paths_are_left_alone(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "paths": {"data_root": str(tmp_path), "atlas_dir": str(tmp_path / "elsewhere")}
    }))
    cfg = load_config(cfg_file)
    assert cfg.paths.atlas_dir == tmp_path / "elsewhere"
