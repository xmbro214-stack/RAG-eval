"""Configuration helpers for the RAG evaluation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = OmegaConf.load(path)
    return OmegaConf.to_container(loaded, resolve=True)  # type: ignore[return-value]


def config_value(
    section: dict[str, Any],
    key: str,
    *,
    env_key: str | None = None,
    default: Any = None,
) -> Any:
    value = section.get(key)
    if value not in (None, ""):
        return value
    env_name = section.get(env_key or f"{key}_env")
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value not in (None, ""):
            return env_value
    return default
