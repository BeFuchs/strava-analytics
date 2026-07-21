"""Load and validate the athlete configuration from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when the athlete config is missing or invalid."""


@dataclass(frozen=True)
class AthleteConfig:
    ftp_watts: int
    threshold_hr: int
    weight_kg: float
    max_hr: int


_REQUIRED_FIELDS = ("ftp_watts", "threshold_hr", "weight_kg", "max_hr")


def load_config(path: str | Path) -> AthleteConfig:
    """Read a YAML config file and return a validated ``AthleteConfig``."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or not isinstance(raw.get("athlete"), dict):
        raise ConfigError(f"{path}: expected a top-level 'athlete' mapping")
    athlete = raw["athlete"]

    missing = [field for field in _REQUIRED_FIELDS if field not in athlete]
    if missing:
        raise ConfigError(f"{path}: missing athlete fields: {', '.join(missing)}")

    for field in _REQUIRED_FIELDS:
        value = athlete[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{path}: athlete.{field} must be a positive number")

    return AthleteConfig(
        ftp_watts=int(athlete["ftp_watts"]),
        threshold_hr=int(athlete["threshold_hr"]),
        weight_kg=float(athlete["weight_kg"]),
        max_hr=int(athlete["max_hr"]),
    )
