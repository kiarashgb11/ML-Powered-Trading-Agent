"""Loading, evaluation, and diagnostics for the Phase 2.1 ablation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from futures_ml.evaluation.metrics import prediction_deciles, regression_metrics
from futures_ml.evaluation.splits import SplitBoundaries, split_expression
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A
from futures_ml.features.phase2_1 import (
    EXPERIMENT_FEATURES,
    RAW_ANCHOR_FEATURES,
    add_relative_market_features,
    select_experiment_view,
)
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root
from futures_ml.models.training import deterministic_even_sample

SOURCE_FEATURES = (*MODEL_FEATURES_A, *RAW_ANCHOR_FEATURES)
SOURCE_METADATA = (
    "timestamp",
    "root",
    "instrument_id",
    "contract_segment_id",
)
TARGET = "future_return_1m"
METRIC_NAMES = ("mae", "rmse", "r2", "pearson", "spearman", "direction_accuracy")


def load_common_training_samples(
    feature_root: Path,
    roots: tuple[str, ...],
    boundaries: SplitBoundaries,
    rows_per_root: tuple[int, ...],
) -> tuple[dict[int, pl.DataFrame], dict[str, int], dict[int, dict[str, int]]]:
    """Load direct even samples from one common eligible source for every cap."""
    samples: dict[int, list[pl.DataFrame]] = {cap: [] for cap in rows_per_root}
    eligible_counts: dict[str, int] = {}
    fit_counts: dict[int, dict[str, int]] = {cap: {} for cap in rows_per_root}
    for root in roots:
        frame = (
            scan_partitioned_root(feature_root, root)
            .filter(eligibility_expression(TARGET) & split_expression("train", boundaries))
            .select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
            .collect()
        )
        eligible_counts[root] = frame.height
        for cap in rows_per_root:
            sampled = deterministic_even_sample(frame, cap)
            samples[cap].append(sampled)
            fit_counts[cap][root] = sampled.height
    return (
        {cap: pl.concat(parts, rechunk=True) for cap, parts in samples.items()},
        eligible_counts,
        fit_counts,
    )


def experiment_xy(frame: pl.DataFrame, experiment: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Create X/y from a common sampled source frame."""
    viewed = select_experiment_view(
        frame,
        experiment,
        metadata=("root",),
        target=TARGET,
    )
    assert isinstance(viewed, pl.DataFrame)
    return viewed.drop(TARGET), viewed[TARGET].to_numpy()


def load_evaluation_root(
    feature_root: Path,
    root: str,
    boundaries: SplitBoundaries,
    split: str,
) -> pl.DataFrame:
    """Collect one root/split with all columns needed by A/B/C and diagnostics."""
    return (
        scan_partitioned_root(feature_root, root)
        .filter(eligibility_expression(TARGET) & split_expression(split, boundaries))
        .select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
        .collect()
    )


def predict_experiment(model: Any, frame: pl.DataFrame, experiment: str) -> np.ndarray:
    """Predict an experiment after applying its controlled feature view."""
    x, _ = experiment_xy(frame, experiment)
    return np.asarray(model.predict(x), dtype=float)


def _macro_average(per_root: Mapping[str, Mapping[str, float | int]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in per_root.values()], dtype=float)
        output[name] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    return output


def evaluate_models(
    models: Mapping[str, tuple[str, Any]],
    feature_root: Path,
    roots: tuple[str, ...],
    boundaries: SplitBoundaries,
    split: str,
    *,
    retain_metadata_for: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Evaluate many models in one root-at-a-time pass over all eligible rows."""
    retain_metadata_for = retain_metadata_for or set()
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"actual": [], "predicted": [], "root": []} for name in models
    }
    metadata_parts: dict[str, list[np.ndarray]] = {
        "timestamp": [],
        "chicago_hour": [],
        "chicago_day_of_week": [],
        "realized_vol_15m": [],
        "root": [],
    }
    per_root: dict[str, dict[str, dict[str, float | int]]] = {name: {} for name in models}
    for root in roots:
        frame = load_evaluation_root(feature_root, root, boundaries, split)
        actual = frame[TARGET].to_numpy()
        for name, (experiment, model) in models.items():
            predicted = predict_experiment(model, frame, experiment)
            per_root[name][root] = regression_metrics(actual, predicted)
            parts[name]["actual"].append(actual)
            parts[name]["predicted"].append(predicted)
            parts[name]["root"].append(np.full(frame.height, root, dtype="U3"))
        if retain_metadata_for:
            metadata_parts["timestamp"].append(frame["timestamp"].to_numpy())
            metadata_parts["chicago_hour"].append(frame["chicago_hour"].to_numpy())
            metadata_parts["chicago_day_of_week"].append(
                frame["chicago_day_of_week"].to_numpy()
            )
            metadata_parts["realized_vol_15m"].append(
                frame["realized_vol_15m"].to_numpy()
            )
            metadata_parts["root"].append(np.full(frame.height, root, dtype="U3"))

    results: dict[str, Any] = {}
    cache: dict[str, dict[str, np.ndarray]] = {}
    joined_metadata = (
        {name: np.concatenate(values) for name, values in metadata_parts.items()}
        if retain_metadata_for
        else {}
    )
    for name in models:
        actual = np.concatenate(parts[name]["actual"])
        predicted = np.concatenate(parts[name]["predicted"])
        roots_all = np.concatenate(parts[name]["root"])
        results[name] = {
            "overall": regression_metrics(actual, predicted),
            "macro_average": _macro_average(per_root[name]),
            "per_root": per_root[name],
            "prediction_deciles": prediction_deciles(actual, predicted),
        }
        if name in retain_metadata_for:
            cache[name] = {
                "actual": actual,
                "predicted": predicted,
                "root": roots_all,
                **joined_metadata,
            }
    return results, cache


def grouped_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    """Return core signal metrics for each finite group value."""
    rows: list[dict[str, Any]] = []
    for group in np.unique(groups):
        mask = groups == group
        metrics = regression_metrics(actual[mask], predicted[mask])
        rows.append({"group": group.item() if hasattr(group, "item") else group, **metrics})
    return rows


def relative_view(frame: pl.DataFrame) -> pl.DataFrame:
    """Public test helper returning source data with all relative columns."""
    result = add_relative_market_features(frame)
    assert isinstance(result, pl.DataFrame)
    return result


def validate_identical_sample_rows(samples: Mapping[str, pl.DataFrame]) -> None:
    """Fail unless A/B/C samples have identical root/timestamp row identities."""
    reference: pl.DataFrame | None = None
    for name, frame in samples.items():
        identity = frame.select("root", "timestamp")
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise AssertionError(f"Training row identities differ for {name}")


def feature_names(experiment: str) -> tuple[str, ...]:
    """Return the stable numeric input order for an experiment."""
    return EXPERIMENT_FEATURES[experiment]
