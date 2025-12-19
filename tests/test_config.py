# tests/test_configs.py
import pytest
from pathlib import Path
from omegaconf import OmegaConf
from ralfs.core import load_config

def test_all_configs_valid():
    """Test that all config files are valid YAML."""
    config_dir = Path("configs")
    
    for yaml_file in config_dir.rglob("*.yaml"):
        try:
            cfg = OmegaConf.load(yaml_file)
            assert cfg is not None
        except Exception as e:
            pytest.fail(f"Invalid config {yaml_file}: {e}")

def test_default_config_loads():
    """Test default config loads successfully."""
    config = load_config("configs/ralfs.yaml")
    assert config.task == "train"
    assert config.data.dataset == "arxiv"

def test_config_overrides():
    """Test config overrides work."""
    config = load_config(
        "configs/ralfs.yaml",
        overrides=["data.dataset=pubmed"]
    )
    assert config.data.dataset == "pubmed"