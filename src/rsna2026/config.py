"""Configuration loading with no third-party dependency."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML experiment configuration."""
    config_path = Path(path)
    with config_path.open("rb") as file:
        return tomllib.load(file)
