"""Typed loaders for the small, reviewable YAML configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    """Raised when a project configuration file is invalid."""


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SettingsError(f"Expected a mapping in {path}")
    return value


@dataclass(frozen=True)
class Market:
    """One futures root in the fixed research universe."""

    root: str
    name: str
    category: str


@dataclass(frozen=True)
class UniverseSettings:
    """The fixed universal and MBP research universes."""

    markets: tuple[Market, ...]
    core_mbp_markets: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> UniverseSettings:
        """Load and validate the futures universe."""
        raw = _read_yaml(path)
        markets = tuple(Market(**item) for item in raw.get("markets", []))
        roots = [market.root for market in markets]
        core = tuple(str(root) for root in raw.get("core_mbp_markets", []))
        if len(markets) != 26 or len(set(roots)) != 26:
            raise SettingsError("The universal futures universe must contain 26 unique roots.")
        missing = sorted(set(core) - set(roots))
        if missing:
            raise SettingsError(f"Core MBP roots are absent from the universe: {missing}")
        return cls(markets=markets, core_mbp_markets=core)


@dataclass(frozen=True)
class DataSettings:
    """Data request, budget, symbology, and storage assumptions."""

    dataset: str
    budget_limit_usd: float
    horizons_years: dict[str, int]
    universal_schemas: tuple[str, ...]
    core_mbp_schema: str
    continuous_template: str
    definition_template: str
    expected_drive_windows: str | None
    working_space_multiplier: float
    safety_headroom_multiplier: float

    @classmethod
    def load(cls, path: Path) -> DataSettings:
        """Load the data plan and reject unsafe or incomplete values."""
        raw = _read_yaml(path)
        schemas = raw["schemas"]
        symbology = raw["symbology"]
        storage = raw["storage"]
        settings = cls(
            dataset=str(raw["dataset"]),
            budget_limit_usd=float(raw["budget_limit_usd"]),
            horizons_years={str(key): int(value) for key, value in raw["horizons_years"].items()},
            universal_schemas=tuple(str(item) for item in schemas["universal"]),
            core_mbp_schema=str(schemas["core_mbp"]),
            continuous_template=str(symbology["continuous_template"]),
            definition_template=str(symbology["definition_template"]),
            expected_drive_windows=storage.get("expected_drive_windows"),
            working_space_multiplier=float(storage["working_space_multiplier"]),
            safety_headroom_multiplier=float(storage["safety_headroom_multiplier"]),
        )
        required = {*settings.universal_schemas, settings.core_mbp_schema}
        missing = sorted(required - settings.horizons_years.keys())
        if missing:
            raise SettingsError(f"Missing data horizon(s): {missing}")
        if settings.budget_limit_usd <= 0:
            raise SettingsError("budget_limit_usd must be positive")
        if settings.working_space_multiplier < 1 or settings.safety_headroom_multiplier < 1:
            raise SettingsError("Storage multipliers must be at least 1.0")
        return settings
