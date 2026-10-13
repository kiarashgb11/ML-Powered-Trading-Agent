"""Frozen-model and append-only prospective evaluation primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from futures_ml.evaluation.metrics import prediction_deciles, regression_metrics
from futures_ml.features.phase2_1 import EXPERIMENT_FEATURES
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root
from futures_ml.models.phase2_1_training import SOURCE_FEATURES, SOURCE_METADATA, TARGET
from futures_ml.models.training import deterministic_even_sample, finite_json

PHASE2_3_ROOTS = ("ZN", "ZC", "ZB")
PHASE2_3_EXPERIMENTS = ("A_1m", "B_1m")
PREDICTION_EVENT = "prediction"
OUTCOME_EVENT = "outcome"
LOG_KEY = ("timestamp", "root", "model_id", "event_type")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an immutable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_row_hash(frame: pl.DataFrame) -> str:
    """Hash ordered training-row identity without depending on Parquet bytes."""
    identity = frame.select(
        pl.col("timestamp").dt.convert_time_zone("UTC").dt.to_string("%+"),
        pl.col("root"),
        pl.col("instrument_id").cast(pl.String),
        pl.col("contract_segment_id").cast(pl.String),
    )
    digest = hashlib.sha256()
    for row in identity.iter_rows():
        digest.update(("\x1f".join(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def historical_training_expression(cutoff: datetime) -> pl.Expr:
    """Select eligible historical rows strictly before the frozen cutoff."""
    return eligibility_expression(TARGET) & (pl.col("timestamp") < cutoff)


def prospective_expression(cutoff: datetime) -> pl.Expr:
    """Select observations strictly after the historical research cutoff."""
    return pl.col("timestamp") > cutoff


def causal_prediction_eligibility_expression() -> pl.Expr:
    """Require completed feature warm-up without consulting a future label."""
    return pl.col("rolling_vwap_240m").is_not_null() & pl.col(
        "rolling_vwap_240m"
    ).is_finite()


def load_historical_training_sample(
    feature_root: Path,
    root: str,
    cutoff: datetime,
    maximum_rows: int,
) -> tuple[pl.DataFrame, int, list[float]]:
    """Load one root's deterministic final sample and frozen volatility tertiles."""
    if root not in PHASE2_3_ROOTS:
        raise ValueError(f"Phase 2.3 root is not preselected: {root}")
    frame = (
        scan_partitioned_root(feature_root, root)
        .filter(historical_training_expression(cutoff))
        .select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
        .sort("timestamp")
        .collect()
    )
    volatility = frame["realized_vol_15m"].filter(
        frame["realized_vol_15m"].is_finite()
    )
    thresholds = [
        float(volatility.quantile(1 / 3)),
        float(volatility.quantile(2 / 3)),
    ]
    return deterministic_even_sample(frame, maximum_rows), frame.height, thresholds


def distribution_profile(
    frame: pl.DataFrame, features: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    """Capture robust feature-distribution statistics without using outcomes."""
    output: dict[str, dict[str, float | int | None]] = {}
    for feature in features:
        values = frame[feature]
        finite = values.filter(values.is_finite())
        q1 = float(finite.quantile(0.25)) if len(finite) else None
        median = float(finite.median()) if len(finite) else None
        q3 = float(finite.quantile(0.75)) if len(finite) else None
        output[feature] = {
            "observations": frame.height,
            "finite_observations": len(finite),
            "missing_rate": float(1 - len(finite) / frame.height) if frame.height else None,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": None if q1 is None or q3 is None else q3 - q1,
        }
    return output


def training_medians(model: Any, features: tuple[str, ...]) -> dict[str, float | None]:
    """Extract the fitted numeric imputer medians in frozen feature order."""
    imputer = model.named_steps["preprocessor"].named_transformers_["numeric"].named_steps[
        "imputer"
    ]
    return {
        feature: float(value) if math.isfinite(float(value)) else None
        for feature, value in zip(features, imputer.statistics_, strict=True)
    }


def load_prospective_inputs(
    feature_root: Path,
    root: str,
    cutoff: datetime,
) -> pl.DataFrame:
    """Read only metadata and causal features for post-cutoff prediction."""
    return (
        scan_partitioned_root(feature_root, root)
        .filter(
            prospective_expression(cutoff)
            & causal_prediction_eligibility_expression()
        )
        .select(*SOURCE_METADATA, *SOURCE_FEATURES)
        .sort("timestamp")
        .collect()
    )


def load_prospective_outcomes(
    feature_root: Path,
    root: str,
    cutoff: datetime,
) -> pl.DataFrame:
    """Read labels separately after predictions can be generated at time t."""
    return (
        scan_partitioned_root(feature_root, root)
        .filter(prospective_expression(cutoff) & eligibility_expression(TARGET))
        .select(*SOURCE_METADATA, TARGET)
        .sort("timestamp")
        .collect()
    )


def predict_frozen_model(
    model: Any,
    frame: pl.DataFrame,
    *,
    model_id: str,
    experiment: str,
    data_version: str,
    recorded_at_utc: datetime,
) -> pl.DataFrame:
    """Generate prediction events without exposing the estimator to labels."""
    if experiment not in PHASE2_3_EXPERIMENTS:
        raise ValueError(f"Unsupported frozen experiment: {experiment}")
    features = EXPERIMENT_FEATURES[experiment]
    missing = sorted(set((*SOURCE_METADATA, *features)) - set(frame.columns))
    if missing:
        raise ValueError(f"Prospective prediction input is missing columns: {missing}")
    predicted = np.asarray(model.predict(frame.select(*features)), dtype=float)
    return frame.select(*SOURCE_METADATA).with_columns(
        pl.lit(model_id).alias("model_id"),
        pl.lit(PREDICTION_EVENT).alias("event_type"),
        pl.Series("prediction", predicted),
        pl.lit(None, dtype=pl.Float64).alias(TARGET),
        pl.lit(data_version).alias("data_version"),
        pl.lit(recorded_at_utc).alias("recorded_at_utc"),
    )


def attach_outcomes(
    prediction_events: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    recorded_at_utc: datetime,
) -> pl.DataFrame:
    """Create immutable outcome events by exact row identity."""
    predictions = prediction_events.filter(pl.col("event_type") == PREDICTION_EVENT)
    outcome_view = outcomes.select(
        "timestamp", "root", "instrument_id", "contract_segment_id", TARGET
    )
    joined = predictions.drop(TARGET).join(
        outcome_view,
        on=("timestamp", "root", "instrument_id", "contract_segment_id"),
        how="inner",
        validate="m:1",
    )
    return joined.with_columns(
        pl.lit(OUTCOME_EVENT).alias("event_type"),
        pl.lit(recorded_at_utc).alias("recorded_at_utc"),
    ).select(*prediction_events.columns)


def read_event_log(directory: Path) -> pl.DataFrame:
    """Read all immutable log parts, or return an empty frame."""
    files = sorted(directory.glob("part-*.parquet")) if directory.exists() else []
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files).sort(*LOG_KEY)


def append_unique_events(events: pl.DataFrame, directory: Path) -> int:
    """Append new keyed events as a new part without modifying old parts."""
    if events.is_empty():
        return 0
    if events.select(pl.struct(LOG_KEY).is_duplicated().any()).item():
        raise ValueError("The prospective event batch contains duplicate keys")
    existing = read_event_log(directory)
    new_events = events
    if not existing.is_empty():
        new_events = events.join(existing.select(*LOG_KEY), on=LOG_KEY, how="anti")
    if new_events.is_empty():
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    logical = json.dumps(
        new_events.select(*LOG_KEY).sort(*LOG_KEY).to_dicts(),
        default=str,
        sort_keys=True,
    ).encode("utf-8")
    suffix = hashlib.sha256(logical).hexdigest()[:16]
    target = directory / f"part-{suffix}.parquet"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite prospective log part: {target}")
    new_events.write_parquet(target, compression="zstd")
    return new_events.height


def verify_frozen_artifacts(manifest: Mapping[str, Any]) -> None:
    """Fail if any frozen pipeline or selected-row artifact hash changed."""
    for entry in manifest["models"]:
        artifact = Path(entry["artifact_path"])
        selected = Path(entry["selected_rows_path"])
        if sha256_file(artifact) != entry["artifact_sha256"]:
            raise RuntimeError(f"Frozen model hash mismatch: {entry['model_id']}")
        if sha256_file(selected) != entry["selected_rows_artifact_sha256"]:
            raise RuntimeError(f"Frozen selected-row artifact changed: {entry['model_id']}")


def volatility_labels(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Assign regimes from historical thresholds only."""
    low, high = thresholds
    labels = np.full(len(values), "unavailable", dtype="U11")
    finite = np.isfinite(values)
    labels[finite & (values <= low)] = "low"
    labels[finite & (values > low) & (values <= high)] = "medium"
    labels[finite & (values > high)] = "high"
    return labels


def _group_metrics(
    actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {"group": str(group), **regression_metrics(actual[groups == group], predicted[groups == group])}
        for group in np.unique(groups)
    ]


def drift_profile(
    frame: pl.DataFrame,
    historical: Mapping[str, Mapping[str, float | int | None]],
    features: tuple[str, ...],
) -> dict[str, dict[str, float | int | None]]:
    """Compare prospective median/IQR/missingness with frozen history."""
    prospective = distribution_profile(frame, features)
    output: dict[str, dict[str, float | int | None]] = {}
    for feature in features:
        past = historical[feature]
        new = prospective[feature]
        old_median = past["median"]
        new_median = new["median"]
        old_iqr = past["iqr"]
        standardized_shift = None
        if (
            old_median is not None
            and new_median is not None
            and old_iqr is not None
            and float(old_iqr) > 0
        ):
            standardized_shift = (float(new_median) - float(old_median)) / float(old_iqr)
        output[feature] = {
            "historical_median": old_median,
            "prospective_median": new_median,
            "historical_iqr": old_iqr,
            "prospective_iqr": new["iqr"],
            "historical_missing_rate": past["missing_rate"],
            "prospective_missing_rate": new["missing_rate"],
            "median_shift_in_historical_iqr": standardized_shift,
        }
    return output


def evaluate_model_events(
    completed_events: pl.DataFrame,
    prospective_features: pl.DataFrame,
    *,
    historical_profile: Mapping[str, Mapping[str, float | int | None]],
    features: tuple[str, ...],
    volatility_thresholds: list[float],
) -> dict[str, Any]:
    """Compute frozen-model confirmation diagnostics for one model/window."""
    actual = completed_events[TARGET].to_numpy()
    predicted = completed_events["prediction"].to_numpy()
    deciles = prediction_deciles(actual, predicted)
    decile_actual = np.asarray([row["mean_actual"] for row in deciles], dtype=float)
    decile_summary = {
        "top_decile_mean_actual": float(decile_actual[-1]),
        "bottom_decile_mean_actual": float(decile_actual[0]),
        "top_minus_bottom": float(decile_actual[-1] - decile_actual[0]),
        "decile_order_spearman": float(
            spearmanr(np.arange(1, 11), decile_actual).statistic
        ),
    }
    volatility = completed_events.join(
        prospective_features.select(
            "timestamp", "root", "instrument_id", "contract_segment_id", "realized_vol_15m"
        ),
        on=("timestamp", "root", "instrument_id", "contract_segment_id"),
        how="left",
        validate="m:1",
    )["realized_vol_15m"].to_numpy()
    return finite_json(
        {
            "overall": regression_metrics(actual, predicted),
            "prediction_deciles": deciles,
            "decile_summary": decile_summary,
            "volatility_regimes": _group_metrics(
                actual,
                predicted,
                volatility_labels(volatility, volatility_thresholds),
            ),
            "drift": drift_profile(prospective_features, historical_profile, features),
        }
    )
