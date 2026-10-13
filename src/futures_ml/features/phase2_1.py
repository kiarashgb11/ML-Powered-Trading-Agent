"""Controlled Phase 2.1 own, raw-anchor, and relative-anchor feature views."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from futures_ml.features.cross_market import ANCHOR_HORIZONS, ANCHOR_ROOTS
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A

RAW_ANCHOR_FEATURES = tuple(
    f"{root}_return_{horizon}m"
    for root in ANCHOR_ROOTS
    for horizon in ANCHOR_HORIZONS
)
RELATIVE_ANCHOR_FEATURES = tuple(
    f"relative_to_{root}_return_{horizon}m"
    for root in ANCHOR_ROOTS
    for horizon in ANCHOR_HORIZONS
)
EXPERIMENT_FEATURES: dict[str, tuple[str, ...]] = {
    "A_1m": MODEL_FEATURES_A,
    "B_1m": (*MODEL_FEATURES_A, *RAW_ANCHOR_FEATURES),
    "C_1m": (*MODEL_FEATURES_A, *RELATIVE_ANCHOR_FEATURES),
}


def required_source_columns(experiment: str) -> tuple[str, ...]:
    """Return stored columns needed to construct an experiment view."""
    if experiment not in EXPERIMENT_FEATURES:
        raise ValueError(f"Unknown Phase 2.1 experiment: {experiment}")
    if experiment == "A_1m":
        return MODEL_FEATURES_A
    return (*MODEL_FEATURES_A, *RAW_ANCHOR_FEATURES)


def add_relative_market_features(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Add target-minus-anchor returns without filling unavailable anchors.

    The target return is the row's own backward-looking ``log_return_hm``.
    Polars null propagation preserves missing exact-time anchor observations.
    For a valid self-anchor row, subtracting the same exact-time return yields
    zero while retaining a common feature schema across all target roots.
    """
    return frame.with_columns(
        *[
            (pl.col(f"log_return_{horizon}m") - pl.col(f"{root}_return_{horizon}m")).alias(
                f"relative_to_{root}_return_{horizon}m"
            )
            for root in ANCHOR_ROOTS
            for horizon in ANCHOR_HORIZONS
        ]
    )


def select_experiment_view(
    frame: pl.DataFrame | pl.LazyFrame,
    experiment: str,
    *,
    metadata: Sequence[str] = ("timestamp", "root"),
    target: str = "future_return_1m",
) -> pl.DataFrame | pl.LazyFrame:
    """Project a common source frame into one controlled Phase 2.1 matrix."""
    if experiment not in EXPERIMENT_FEATURES:
        raise ValueError(f"Unknown Phase 2.1 experiment: {experiment}")
    viewed = add_relative_market_features(frame) if experiment == "C_1m" else frame
    return viewed.select(*metadata, *EXPERIMENT_FEATURES[experiment], target)
