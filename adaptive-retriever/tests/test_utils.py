"""
Tests for Utility Modules: config, device, logger, seed.
"""
import os
import tempfile
import pytest
import torch
from src.utils.config import ConfigDict, load_config, save_config
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def test_config_dict_dot_and_bracket_access():
    cfg = ConfigDict({"model": {"name": "bge", "params": {"dim": 384}}})

    # Dot access
    assert cfg.model.name == "bge"
    assert cfg.model.params.dim == 384

    # Bracket access
    assert cfg["model"]["name"] == "bge"
    assert cfg["model"]["params"]["dim"] == 384

    # Mutation
    cfg.model.name = "minilm"
    assert cfg.model.name == "minilm"

    # Conversion back to dict
    plain = cfg.to_dict()
    assert isinstance(plain, dict)
    assert not isinstance(plain, ConfigDict)
    assert plain["model"]["name"] == "minilm"


def test_config_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = os.path.join(tmp_dir, "test_config.yaml")
        original = ConfigDict({"experiment": {"name": "test_exp", "seed": 42}})
        save_config(original, cfg_file)

        loaded = load_config(cfg_file)
        assert loaded.experiment.name == "test_exp"
        assert loaded.experiment.seed == 42


def test_device_resolution():
    cpu_device = resolve_device("cpu")
    assert cpu_device.type == "cpu"

    info = get_device_info(cpu_device)
    assert info["type"] == "cpu"

    auto_device = resolve_device("auto")
    assert auto_device.type in ("cuda", "cpu", "mps")


def test_logger_creation():
    logger = setup_logger("TestLogger", log_level="DEBUG")
    assert logger.name == "TestLogger"
