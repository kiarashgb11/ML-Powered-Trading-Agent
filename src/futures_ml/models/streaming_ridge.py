"""Bounded-memory standardized Ridge fitting from sufficient statistics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


def _numeric_matrix(frame: Any, features: tuple[str, ...]) -> np.ndarray:
    if isinstance(frame, pl.DataFrame):
        return frame.select(*features).to_numpy().astype(np.float64, copy=False)
    return np.asarray(frame.loc[:, list(features)], dtype=np.float64)


def _root_values(frame: Any) -> np.ndarray:
    if isinstance(frame, pl.DataFrame):
        return frame["root"].to_numpy()
    return np.asarray(frame.loc[:, "root"])


@dataclass
class StreamingRidgeRegressor:
    """Prediction-compatible result of a two-pass, full-train Ridge fit."""

    features: tuple[str, ...]
    roots: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    intercept_: float
    coef_: np.ndarray
    alpha: float
    training_rows: int

    def _design(self, frame: Any) -> np.ndarray:
        values = _numeric_matrix(frame, self.features)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.take(self.medians, np.nonzero(invalid)[1])
        values = (values - self.means) / self.scales
        root_values = _root_values(frame)
        one_hot = np.column_stack([root_values == root for root in self.roots]).astype(float)
        return np.column_stack([values, one_hot])

    def predict(self, frame: Any) -> np.ndarray:
        """Predict from a Polars or pandas-like frame."""
        return self.intercept_ + self._design(frame) @ self.coef_

    def get_feature_names_out(self) -> list[str]:
        """Return numeric and one-hot coefficient labels in fitted order."""
        return [*self.features, *[f"root_{root}" for root in self.roots]]


def fit_streaming_ridge(
    batch_factory: Callable[[], Iterable[tuple[pl.DataFrame, np.ndarray]]],
    *,
    features: tuple[str, ...],
    roots: tuple[str, ...],
    medians: np.ndarray,
    alpha: float,
) -> StreamingRidgeRegressor:
    """Fit exact Ridge normal equations while keeping source batches bounded.

    ``medians`` must have been learned from training observations only. Means
    and population standard deviations are computed over every supplied row.
    The second pass accumulates X'X and X'y; no validation/test row is accepted.
    """
    count = 0
    sums = np.zeros(len(features), dtype=np.float64)
    square_sums = np.zeros(len(features), dtype=np.float64)
    for frame, _target in batch_factory():
        values = _numeric_matrix(frame, features)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.take(medians, np.nonzero(invalid)[1])
        count += len(values)
        sums += values.sum(axis=0)
        square_sums += np.square(values).sum(axis=0)
    if not count:
        raise ValueError("Cannot fit Ridge without training observations")
    means = sums / count
    variances = np.maximum(square_sums / count - np.square(means), 0.0)
    scales = np.sqrt(variances)
    scales[scales == 0] = 1.0

    width = 1 + len(features) + len(roots)
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros(width, dtype=np.float64)
    second_pass_rows = 0
    for frame, target in batch_factory():
        values = _numeric_matrix(frame, features)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.take(medians, np.nonzero(invalid)[1])
        root_values = _root_values(frame)
        target = np.asarray(target, dtype=np.float64)
        numeric = (values - means) / scales
        one_hot = np.column_stack([root_values == root for root in roots]).astype(float)
        design = np.column_stack([np.ones(len(values)), numeric, one_hot])
        gram += design.T @ design
        cross += design.T @ target
        second_pass_rows += len(values)
    if second_pass_rows != count:
        raise RuntimeError("Full-Ridge source changed between streaming passes")
    penalty = np.eye(width, dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(gram + penalty, cross)
    return StreamingRidgeRegressor(
        features=features,
        roots=roots,
        medians=np.asarray(medians, dtype=np.float64),
        means=means,
        scales=scales,
        intercept_=float(solution[0]),
        coef_=solution[1:],
        alpha=alpha,
        training_rows=count,
    )
