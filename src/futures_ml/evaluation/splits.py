"""Global chronological splits with explicit boundary purging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class SplitBoundaries:
    """Shared UTC split boundaries for every futures root."""

    data_start: datetime
    train_validation_boundary: datetime
    validation_test_boundary: datetime
    data_end: datetime
    purge_minutes: int

    def to_dict(self) -> dict[str, str | int]:
        values = asdict(self)
        return {key: value.isoformat() if isinstance(value, datetime) else value for key, value in values.items()}


def calculate_global_boundaries(
    data_start: datetime,
    data_end: datetime,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_minutes: int = 75,
) -> SplitBoundaries:
    """Divide elapsed clock time 70/15/15 and floor boundaries to a minute."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be below one")
    span = data_end - data_start

    def floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)

    return SplitBoundaries(
        data_start=data_start,
        train_validation_boundary=floor_minute(data_start + span * train_fraction),
        validation_test_boundary=floor_minute(
            data_start + span * (train_fraction + validation_fraction)
        ),
        data_end=data_end,
        purge_minutes=purge_minutes,
    )


def split_expression(name: str, boundaries: SplitBoundaries) -> pl.Expr:
    """Return a non-overlapping chronological split expression.

    A full 75-minute gap is excluded on each side of each boundary. This is
    deliberately more conservative than the 60-minute exact-return lookback
    plus 15-minute longest label horizon.
    """
    purge = timedelta(minutes=boundaries.purge_minutes)
    timestamp = pl.col("timestamp")
    if name == "train":
        return timestamp < boundaries.train_validation_boundary - purge
    if name == "validation":
        return (timestamp >= boundaries.train_validation_boundary + purge) & (
            timestamp < boundaries.validation_test_boundary - purge
        )
    if name == "test":
        return timestamp >= boundaries.validation_test_boundary + purge
    raise ValueError(f"Unknown split: {name}")


def assign_split(frame: pl.DataFrame, boundaries: SplitBoundaries) -> pl.DataFrame:
    """Label retained rows and mark purged rows explicitly as null."""
    return frame.with_columns(
        pl.when(split_expression("train", boundaries))
        .then(pl.lit("train"))
        .when(split_expression("validation", boundaries))
        .then(pl.lit("validation"))
        .when(split_expression("test", boundaries))
        .then(pl.lit("test"))
        .otherwise(None)
        .alias("split")
    )


def assert_split_integrity(frame: pl.DataFrame, boundaries: SplitBoundaries) -> None:
    """Fail if a row is assigned inconsistently or test overlaps earlier data."""
    assigned = assign_split(frame, boundaries)
    counts = assigned.group_by("timestamp").agg(pl.col("split").drop_nulls().n_unique().alias("n"))
    if counts.filter(pl.col("n") > 1).height:
        raise AssertionError("A timestamp belongs to more than one split")
    test = assigned.filter(pl.col("split") == "test")
    if test.height and test["timestamp"].min() < boundaries.validation_test_boundary:
        raise AssertionError("Test observations overlap the validation period")
