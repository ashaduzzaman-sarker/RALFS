"""
Basic setup tests to ensure package is installed correctly.
"""
import pytest
import sys
from pathlib import Path


def test_python_version():
    """Test Python version is 3.10+"""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_package_importable():
    """Test RALFS package can be imported"""
    import ralfs
    assert ralfs.__name__ == "ralfs"


def test_cli_importable():
    """Test CLI module is importable"""
    from ralfs import cli
    assert hasattr(cli, "app")


def test_project_structure():
    """Test essential directories exist"""
    root = Path(__file__).parent.parent
    assert (root / "ralfs").exists()
    assert (root / "configs").exists()
    assert (root / "tests").exists()
    assert (root / "pyproject.toml").exists()


def test_config_files_exist():
    """Test config files are present"""
    root = Path(__file__).parent.parent
    assert (root / "configs" / "ralfs.yaml").exists()
    assert (root / "configs" / "data" / "default.yaml").exists()


@pytest.mark.slow
def test_dependencies_installed():
    """Test critical dependencies are installed"""
    import torch
    import transformers
    import sentence_transformers
    import datasets
    import hydra
    import typer
    
    assert torch.__version__ >= "2.4.0"
    assert transformers.__version__ >= "4.44.0"