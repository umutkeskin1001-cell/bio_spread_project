from pathlib import Path

import yaml


def test_prime_config_feature_windows_match_model_budget():
    cfg = yaml.safe_load(Path("config/cassiopeia_prime.yaml").read_text())
    assert cfg["model"]["max_windows"] == 56
    assert sum(cfg["features"]["max_windows"]) == cfg["model"]["max_windows"]


def test_pyproject_defines_ci_extras():
    text = Path("pyproject.toml").read_text()
    assert "[project.optional-dependencies]" in text and "dev =" in text and "api =" in text


def test_prime_config_odd_kernel():
    cfg = yaml.safe_load(Path("config/cassiopeia_prime.yaml").read_text())
    assert cfg["model"]["window_conv_kernel"] % 2 == 1


def test_config_feature_strides_match_windows():
    cfg = yaml.safe_load(Path("config/cassiopeia_prime.yaml").read_text())
    assert len(cfg["features"]["window_sizes"]) == len(cfg["features"]["strides"])
