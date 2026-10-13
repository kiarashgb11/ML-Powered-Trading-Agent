"""Predeclared expanding-window folds for Phase 2.2 temporal replication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding training window followed by a purged evaluation window."""

    fold_id: str
    training_start: datetime
    training_end_exclusive: datetime
    evaluation_start: datetime
    evaluation_end_exclusive: datetime
    boundary: datetime
    purge_minutes: int

    def to_dict(self) -> dict[str, str | int]:
        """Serialize timestamps without losing their UTC offsets."""
        return {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in asdict(self).items()
        }


def build_walk_forward_folds(
    *,
    data_start: datetime,
    evaluation_boundaries: tuple[datetime, ...],
    purge_minutes: int,
) -> tuple[WalkForwardFold, ...]:
    """Build fixed folds from ordered evaluation boundaries.

    Boundaries include the first evaluation boundary and the exclusive end of
    the last evaluation period. Training expands from ``data_start``. The
    interval ``[boundary-purge, boundary+purge)`` is unused by that fold.
    """
    if len(evaluation_boundaries) < 2:
        raise ValueError("At least two evaluation boundaries are required")
    if tuple(sorted(evaluation_boundaries)) != evaluation_boundaries:
        raise ValueError("Evaluation boundaries must be strictly chronological")
    if len(set(evaluation_boundaries)) != len(evaluation_boundaries):
        raise ValueError("Evaluation boundaries must be unique")
    if evaluation_boundaries[0] <= data_start:
        raise ValueError("The first evaluation boundary must follow data_start")
    purge = timedelta(minutes=purge_minutes)
    folds = tuple(
        WalkForwardFold(
            fold_id=f"fold_{index:02d}",
            training_start=data_start,
            training_end_exclusive=boundary - purge,
            evaluation_start=boundary + purge,
            evaluation_end_exclusive=evaluation_boundaries[index],
            boundary=boundary,
            purge_minutes=purge_minutes,
        )
        for index, boundary in enumerate(evaluation_boundaries[:-1], start=1)
    )
    validate_walk_forward_folds(folds)
    return folds


def validate_walk_forward_folds(folds: tuple[WalkForwardFold, ...]) -> None:
    """Reject temporal overlap, insufficient purging, or non-expanding folds."""
    previous_evaluation_end: datetime | None = None
    previous_training_end: datetime | None = None
    for fold in folds:
        purge = timedelta(minutes=fold.purge_minutes)
        if fold.training_end_exclusive + 2 * purge != fold.evaluation_start:
            raise ValueError(f"{fold.fold_id} does not preserve the two-sided purge")
        if fold.training_start >= fold.training_end_exclusive:
            raise ValueError(f"{fold.fold_id} has no training history")
        if fold.training_end_exclusive >= fold.evaluation_start:
            raise ValueError(f"{fold.fold_id} training overlaps evaluation")
        if fold.evaluation_start >= fold.evaluation_end_exclusive:
            raise ValueError(f"{fold.fold_id} has no evaluation period")
        if previous_evaluation_end is not None and fold.evaluation_start < previous_evaluation_end:
            raise ValueError("Evaluation folds overlap")
        if previous_training_end is not None and fold.training_end_exclusive <= previous_training_end:
            raise ValueError("Training windows do not expand")
        previous_evaluation_end = fold.evaluation_end_exclusive
        previous_training_end = fold.training_end_exclusive


def fold_training_expression(fold: WalkForwardFold) -> pl.Expr:
    """Select only observations known before a fold's purged boundary."""
    return (pl.col("timestamp") >= fold.training_start) & (
        pl.col("timestamp") < fold.training_end_exclusive
    )


def fold_evaluation_expression(fold: WalkForwardFold) -> pl.Expr:
    """Select the immediately following purged evaluation period."""
    return (pl.col("timestamp") >= fold.evaluation_start) & (
        pl.col("timestamp") < fold.evaluation_end_exclusive
    )
