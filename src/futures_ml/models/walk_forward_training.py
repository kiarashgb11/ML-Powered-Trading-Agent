"""Bounded-memory Phase 2.2 fold loading, fitting inputs, and evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from futures_ml.evaluation.metrics import prediction_deciles, regression_metrics
from futures_ml.evaluation.walk_forward import (
    WalkForwardFold,
    fold_evaluation_expression,
    fold_training_expression,
)
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root
from futures_ml.models.phase2_1_training import (
    METRIC_NAMES,
    SOURCE_FEATURES,
    SOURCE_METADATA,
    TARGET,
    experiment_xy,
    grouped_metrics,
)
from futures_ml.models.training import deterministic_even_sample


def load_fold_training_sample(
    feature_root: Path,
    roots: tuple[str, ...],
    fold: WalkForwardFold,
    maximum_rows_per_root: int,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int], dict[str, list[float]]]:
    """Load one deterministic expanding-window sample and train-only regimes."""
    frames: list[pl.DataFrame] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    thresholds: dict[str, list[float]] = {}
    for root in roots:
        frame = (
            scan_partitioned_root(feature_root, root)
            .filter(eligibility_expression(TARGET) & fold_training_expression(fold))
            .select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
            .collect()
        )
        eligible_counts[root] = frame.height
        finite_volatility = frame["realized_vol_15m"].filter(
            frame["realized_vol_15m"].is_finite()
        )
        thresholds[root] = [
            float(finite_volatility.quantile(1 / 3)),
            float(finite_volatility.quantile(2 / 3)),
        ]
        sampled = deterministic_even_sample(frame, maximum_rows_per_root)
        selected_counts[root] = sampled.height
        frames.append(sampled)
    return pl.concat(frames, rechunk=True), eligible_counts, selected_counts, thresholds


def load_fold_evaluation_root(
    feature_root: Path,
    root: str,
    fold: WalkForwardFold,
) -> pl.DataFrame:
    """Load every eligible evaluation row for one root and fold."""
    return (
        scan_partitioned_root(feature_root, root)
        .filter(eligibility_expression(TARGET) & fold_evaluation_expression(fold))
        .select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
        .collect()
    )


def _macro_average(per_root: dict[str, dict[str, float | int]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in per_root.values()], dtype=float)
        output[name] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    return output


def _regime_labels(
    roots: np.ndarray,
    volatility: np.ndarray,
    thresholds: dict[str, list[float]],
) -> np.ndarray:
    regimes = np.full(len(roots), "unavailable", dtype="U11")
    for root, (low, high) in thresholds.items():
        root_mask = roots == root
        finite = root_mask & np.isfinite(volatility)
        regimes[finite & (volatility <= low)] = "low"
        regimes[finite & (volatility > low) & (volatility <= high)] = "medium"
        regimes[finite & (volatility > high)] = "high"
    return regimes


def evaluate_fold_models(
    models: dict[str, Any],
    feature_root: Path,
    roots: tuple[str, ...],
    fold: WalkForwardFold,
    volatility_thresholds: dict[str, list[float]],
) -> dict[str, Any]:
    """Evaluate frozen A/B models together in one full fold scan."""
    actual_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    hour_parts: list[np.ndarray] = []
    volatility_parts: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in models}
    per_root: dict[str, dict[str, dict[str, float | int]]] = {name: {} for name in models}
    for root in roots:
        frame = load_fold_evaluation_root(feature_root, root, fold)
        actual = frame[TARGET].to_numpy()
        actual_parts.append(actual)
        root_parts.append(np.full(frame.height, root, dtype="U3"))
        hour_parts.append(frame["chicago_hour"].to_numpy())
        volatility_parts.append(frame["realized_vol_15m"].to_numpy())
        for experiment, model in models.items():
            x, _ = experiment_xy(frame, experiment)
            predicted = np.asarray(model.predict(x), dtype=float)
            predictions[experiment].append(predicted)
            per_root[experiment][root] = regression_metrics(actual, predicted)

    actual = np.concatenate(actual_parts)
    roots_all = np.concatenate(root_parts)
    hours = np.concatenate(hour_parts).astype(int)
    volatility = np.concatenate(volatility_parts).astype(float)
    regimes = _regime_labels(roots_all, volatility, volatility_thresholds)
    output: dict[str, Any] = {}
    for experiment in models:
        predicted = np.concatenate(predictions[experiment])
        output[experiment] = {
            "overall": regression_metrics(actual, predicted),
            "macro_average": _macro_average(per_root[experiment]),
            "per_root": per_root[experiment],
            "volatility_regimes": grouped_metrics(actual, predicted, regimes),
            "chicago_hours": grouped_metrics(actual, predicted, hours),
            "prediction_deciles": prediction_deciles(actual, predicted),
        }
    return output


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """Summarize a metric across folds, including positive-sign consistency."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "mean": math.nan,
            "median": math.nan,
            "standard_deviation": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
            "positive_folds": 0,
            "finite_folds": 0,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "standard_deviation": float(np.std(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "positive_folds": int(np.sum(finite > 0)),
        "finite_folds": int(len(finite)),
    }


def root_ranks(per_root: dict[str, dict[str, float | int]]) -> dict[str, int]:
    """Rank roots from highest to lowest finite Pearson with deterministic ties."""
    ordered = sorted(
        per_root,
        key=lambda root: (
            -float(per_root[root]["pearson"])
            if math.isfinite(float(per_root[root]["pearson"]))
            else math.inf,
            root,
        ),
    )
    return {root: rank for rank, root in enumerate(ordered, start=1)}
