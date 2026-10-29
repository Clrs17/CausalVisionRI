"""Minimal config loader for phase 2 (dot-access nested dict, same convention
as phase 1 src.utils.Config)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data
        for k in key.split("."):
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def load_cfg(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
