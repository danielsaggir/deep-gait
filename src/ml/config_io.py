"""Load YAML config for ML scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {p}")
    return data
