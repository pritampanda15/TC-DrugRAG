"""Configuration loading utilities."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and process configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(obj):
    """Recursively substitute ${VAR} with environment variables."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            # Handle default values: ${VAR:default}
            if ":" in var_name:
                var_name, default = var_name.split(":", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_name, obj)
        return obj
    else:
        return obj
