"""Bounded-memory loading, fitting, evaluation, and artifact persistence."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn

from futures_ml.evaluation.metrics import prediction_deciles, regression_metrics
from futures_ml.evaluation.splits import SplitBoundaries, split_expression
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root


def deterministic_even_sample(frame: pl.DataFrame, maximum_rows: int) -> pl.DataFrame:
    """Select evenly spaced chronological rows without random shuffling."""
    if frame.height <= maximum_rows:
        return frame
    indices = np.linspace(0, frame.height - 1, maximum_rows, dtype=np.int64)
    return frame[indices.tolist()]


def load_training_sample(
    feature_root: Path,
    roots: tuple[str, ...],
    features: list[str],
    target: str,
    boundaries: SplitBoundaries,
    maximum_rows_per_root: int,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
    """Load an auditable even sample from each root's eligible train period."""
    frames: list[pl.DataFrame] = []
    eligible_counts: dict[str, int] = {}
    fit_counts: dict[str, int] = {}
    for root in roots:
        frame = (
            scan_partitioned_root(feature_root, root)
            .filter(eligibility_expression(target) & split_expression("train", boundaries))
            .select("timestamp", "root", *features, target)
            .collect()
        )
        eligible_counts[root] = frame.height
        sampled = deterministic_even_sample(frame, maximum_rows_per_root)
        fit_counts[root] = sampled.height
        frames.append(sampled)
    return pl.concat(frames, rechunk=True), eligible_counts, fit_counts


def evaluate_model(
    model: Any,
    feature_root: Path,
    roots: tuple[str, ...],
    features: list[str],
    target: str,
    boundaries: SplitBoundaries,
    split: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Predict one root at a time, retaining only compact result arrays."""
    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    per_root: dict[str, dict[str, float | int]] = {}
    for root in roots:
        frame = (
            scan_partitioned_root(feature_root, root)
            .filter(eligibility_expression(target) & split_expression(split, boundaries))
            .select("root", *features, target)
            .collect()
        )
        actual = frame[target].to_numpy()
        predicted = np.asarray(model.predict(frame.select("root", *features)), dtype=float)
        actual_parts.append(actual)
        prediction_parts.append(predicted)
        root_parts.append(np.full(frame.height, root, dtype=f"U{max(2, len(root))}"))
        per_root[root] = regression_metrics(actual, predicted)
    actual_all = np.concatenate(actual_parts)
    predicted_all = np.concatenate(prediction_parts)
    roots_all = np.concatenate(root_parts)
    overall = regression_metrics(actual_all, predicted_all)
    metric_names = ("mae", "rmse", "r2", "pearson", "spearman", "direction_accuracy")
    macro = {}
    for name in metric_names:
        values = np.asarray([float(per_root[root][name]) for root in roots], dtype=float)
        macro[name] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    return (
        {
            "overall": overall,
            "macro_average": macro,
            "per_root": per_root,
            "prediction_deciles": prediction_deciles(actual_all, predicted_all),
        },
        actual_all,
        predicted_all,
        roots_all,
    )


def save_model_artifact(
    model: Any,
    directory: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    """Persist the fitted pipeline and its complete reproducibility contract."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model_pipeline.joblib", compress=3)
    preprocessor = model.named_steps.get("preprocessor")
    enriched = dict(metadata)
    enriched["encoded_feature_order"] = (
        preprocessor.get_feature_names_out().tolist() if preprocessor is not None else []
    )
    enriched["library_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    (directory / "metadata.json").write_text(
        json.dumps(enriched, indent=2, allow_nan=True), encoding="utf-8"
    )


def finite_json(value: Any) -> Any:
    """Recursively replace non-finite floats with null for strict JSON output."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
