"""
Configuration parsing and schema validation utilities for AdaptiveRetriever.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigDict(dict):
    """
    Dictionary subclass enabling attribute-style dot access as well as dict-style key access.
    Recursively converts nested dictionaries to ConfigDict instances.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [ConfigDict(item) if isinstance(item, dict) else item for item in value]

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Recursively converts ConfigDict back into a standard Python dict."""
        plain_dict: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                plain_dict[key] = value.to_dict()
            elif isinstance(value, list):
                plain_dict[key] = [item.to_dict() if isinstance(item, ConfigDict) else item for item in value]
            else:
                plain_dict[key] = value
        return plain_dict


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Loads a YAML configuration file into a ConfigDict.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ConfigDict containing parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration at {path}: {e}")

    return ConfigDict(raw_dict)


def save_config(config: Union[ConfigDict, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Saves configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary or ConfigDict.
        output_path: Destination path for YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict() if isinstance(config, ConfigDict) else config
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
