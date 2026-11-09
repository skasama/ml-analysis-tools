from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated analysis settings loaded from YAML."""

    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
        return value

    def get(self, name: str, default=None):
        return self.values.get(name, default)


def load_config(path) -> AnalysisConfig:
    """Load a YAML mapping into an AnalysisConfig."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return AnalysisConfig(values)
