"""Exact-UTC limited cross-market context for Experiment B."""

from __future__ import annotations

import polars as pl

ANCHOR_ROOTS = ("NQ", "ES", "ZN", "CL", "GC", "6E")
ANCHOR_HORIZONS = (1, 5, 15)
CROSS_MARKET_FEATURES = tuple(
    f"{root}_return_{horizon}m" for root in ANCHOR_ROOTS for horizon in ANCHOR_HORIZONS
) + ("NQ_ES_return_spread_5m",)


def anchor_return_table(
    root_frames: dict[str, pl.DataFrame], anchor_roots: tuple[str, ...] = ANCHOR_ROOTS
) -> pl.DataFrame:
    """Create one exact-timestamp wide table; no as-of matching or forward fill."""
    wide: pl.DataFrame | None = None
    for root in anchor_roots:
        frame = root_frames[root].select(
            "timestamp",
            *[
                pl.col(f"log_return_{horizon}m").alias(f"{root}_return_{horizon}m")
                for horizon in ANCHOR_HORIZONS
            ],
        )
        wide = frame if wide is None else wide.join(frame, on="timestamp", how="full", coalesce=True)
    if wide is None:
        raise ValueError("At least one anchor root is required")
    return wide.sort("timestamp").with_columns(
        (pl.col("NQ_return_5m") - pl.col("ES_return_5m")).alias(
            "NQ_ES_return_spread_5m"
        )
    )


def add_cross_market_features(frame: pl.DataFrame, anchors: pl.DataFrame) -> pl.DataFrame:
    """Left join anchors at exactly the current UTC timestamp."""
    return frame.join(anchors, on="timestamp", how="left")
