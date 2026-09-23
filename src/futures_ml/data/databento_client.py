"""Safe construction and capability checks for the Databento client."""

from __future__ import annotations

import os
from typing import Any

API_KEY_ENV = "DATABENTO_API_KEY"


class DatabentoConfigurationError(RuntimeError):
    """Raised when Databento cannot be used safely with current configuration."""


def api_key_is_configured() -> bool:
    """Return whether an API key exists without exposing its value."""
    return bool(os.getenv(API_KEY_ENV, "").strip())


def create_historical_client() -> Any:
    """Create the official client using its environment-variable key lookup."""
    if not api_key_is_configured():
        raise DatabentoConfigurationError(
            f"{API_KEY_ENV} is not configured; no authenticated metadata request was made."
        )
    import databento as db

    return db.Historical()


def verify_dataset_and_schemas(
    client: Any,
    *,
    dataset: str,
    required_schemas: set[str],
) -> tuple[str, ...]:
    """Verify the dataset and required schemas against live metadata."""
    datasets = set(client.metadata.list_datasets())
    if dataset not in datasets:
        raise DatabentoConfigurationError(f"Dataset is not currently available: {dataset}")
    schemas = tuple(client.metadata.list_schemas(dataset=dataset))
    missing = sorted(required_schemas - set(schemas))
    if missing:
        raise DatabentoConfigurationError(
            f"Required schema(s) are not currently available for {dataset}: {missing}"
        )
    return schemas
