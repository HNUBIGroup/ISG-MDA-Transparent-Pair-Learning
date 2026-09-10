"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"model", "training", "data", "output", "metrics"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")
    return _expand(config)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def resolve_path(value: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path

