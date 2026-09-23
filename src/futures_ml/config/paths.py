"""Portable, centralized paths for all large data and model artifacts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_ENV = "FUTURES_ML_DATA_ROOT"
EXPECTED_DRIVE_ENV = "FUTURES_ML_EXPECTED_DRIVE"


class PathConfigurationError(RuntimeError):
    """Raised when data-root configuration violates a storage safety rule."""


@dataclass(frozen=True)
class DataPaths:
    """Every large-data path derived from one external root."""

    root: Path
    raw_data_root: Path
    processed_data_root: Path
    features_root: Path
    batch_root: Path
    cache_root: Path
    temp_root: Path
    model_artifact_root: Path

    @classmethod
    def from_root(cls, root: Path) -> DataPaths:
        """Build the full path layout from a single configured root."""
        root = root.expanduser().resolve(strict=False)
        raw = root / "raw"
        return cls(
            root=root,
            raw_data_root=raw,
            processed_data_root=root / "processed",
            features_root=root / "features",
            batch_root=root / "batches",
            cache_root=root / "cache",
            temp_root=root / "temp",
            model_artifact_root=root / "artifacts" / "models",
        )

    def directories(self) -> tuple[Path, ...]:
        """Return every directory that the project may create."""
        return (
            self.root,
            self.raw_data_root / "ohlcv_1m",
            self.raw_data_root / "ohlcv_1s",
            self.raw_data_root / "mbp1",
            self.raw_data_root / "definitions",
            self.raw_data_root / "statistics",
            self.raw_data_root / "status",
            self.processed_data_root / "bars_1m",
            self.processed_data_root / "bars_5m",
            self.processed_data_root / "bars_15m",
            self.processed_data_root / "bars_30m",
            self.processed_data_root / "bars_1h",
            self.processed_data_root / "bars_4h",
            self.processed_data_root / "bars_1d",
            self.features_root,
            self.batch_root,
            self.cache_root,
            self.temp_root,
            self.model_artifact_root,
        )

    def create(self) -> None:
        """Create the external data layout after all safety checks pass."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def configure_process_temp(self) -> None:
        """Keep this process's temporary files on the configured data volume."""
        os.environ["TEMP"] = str(self.temp_root)
        os.environ["TMP"] = str(self.temp_root)


@dataclass(frozen=True)
class StorageInfo:
    """Capacity information for the volume containing the data root."""

    drive: str
    total_bytes: int
    free_bytes: int

    @classmethod
    def for_path(cls, path: Path) -> StorageInfo:
        """Inspect the closest existing ancestor of a configured path."""
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if not probe.exists():
            raise PathConfigurationError(f"No existing ancestor found for data root: {path}")
        usage = shutil.disk_usage(probe)
        drive = path.drive or path.anchor or str(probe)
        return cls(drive=drive, total_bytes=usage.total, free_bytes=usage.free)


def _normalized_drive(value: str) -> str:
    return value.rstrip("\\/").upper()


def validate_expected_drive(root: Path, expected_drive: str | None) -> None:
    """Enforce a configured Windows drive without reducing POSIX portability."""
    if os.name != "nt" or not expected_drive:
        return
    actual = _normalized_drive(root.drive)
    expected = _normalized_drive(expected_drive)
    if not actual:
        raise PathConfigurationError(f"Data root must be an absolute path, got: {root}")
    if actual != expected:
        raise PathConfigurationError(
            f"Storage safety gate failed: data root is on {actual}, expected {expected}."
        )


def resolve_data_paths(
    *,
    expected_drive: str | None = None,
    create: bool = False,
) -> DataPaths:
    """Resolve the environment-configured root and optionally create its layout."""
    configured = os.getenv(DATA_ROOT_ENV, "").strip()
    if not configured:
        raise PathConfigurationError(
            f"{DATA_ROOT_ENV} is not configured. Refusing to use an implicit location."
        )
    paths = DataPaths.from_root(Path(configured))
    drive_gate = os.getenv(EXPECTED_DRIVE_ENV, expected_drive or "").strip() or None
    validate_expected_drive(paths.root, drive_gate)
    if create:
        paths.create()
    return paths
