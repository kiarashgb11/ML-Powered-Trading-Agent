"""Exact-clock future-return labels that cannot cross contract segments."""

from __future__ import annotations

import polars as pl

LABEL_PREFIX = "future_return_"


def add_future_return(frame: pl.DataFrame, horizon_minutes: int) -> pl.DataFrame:
    """Add y_t^h = log(close[t+h] / close[t]) using an exact timestamp self-join."""
    name = f"{LABEL_PREFIX}{horizon_minutes}m"
    timestamp_dtype = frame.schema["timestamp"]
    lookup = frame.select(
        "contract_segment_id",
        (pl.col("timestamp") - pl.duration(minutes=horizon_minutes))
        .cast(timestamp_dtype)
        .alias("timestamp"),
        pl.col("close").alias(f"_future_close_{horizon_minutes}m"),
    )
    joined = frame.join(lookup, on=["contract_segment_id", "timestamp"], how="left")
    future = pl.col(f"_future_close_{horizon_minutes}m")
    return joined.with_columns(
        pl.when((pl.col("close") > 0) & (future > 0))
        .then((future / pl.col("close")).log())
        .otherwise(None)
        .alias(name)
    ).drop(f"_future_close_{horizon_minutes}m")


def add_future_labels(
    frame: pl.DataFrame, horizons: tuple[int, ...] = (1, 5, 15)
) -> pl.DataFrame:
    """Add exact-time future returns and a diagnostic five-minute direction."""
    result = frame
    for horizon in horizons:
        result = add_future_return(result, horizon)
    return result.with_columns(
        pl.when(pl.col("future_return_5m") > 0)
        .then(pl.lit("UP"))
        .when(pl.col("future_return_5m") < 0)
        .then(pl.lit("DOWN"))
        .otherwise(None)
        .alias("direction_5m")
    )
