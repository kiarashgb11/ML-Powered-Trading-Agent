"""Train and evaluate the Phase 2 zero, Ridge, and HGB baselines."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.evaluation.splits import calculate_global_boundaries  # noqa: E402
from futures_ml.features.cross_market import CROSS_MARKET_FEATURES  # noqa: E402
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A  # noqa: E402
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root  # noqa: E402
from futures_ml.models.baselines import (  # noqa: E402
    ZeroRegressor,
    make_hist_gradient_boosting_pipeline,
    make_ridge_pipeline,
)
from futures_ml.models.training import (  # noqa: E402
    evaluate_model,
    finite_json,
    load_training_sample,
    save_model_artifact,
)


def _config() -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "config" / "phase2.yaml").read_text(encoding="utf-8"))


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _write_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")


def _fit_and_evaluate(
    *,
    experiment: str,
    scope: str,
    feature_root: Path,
    roots: tuple[str, ...],
    features: list[str],
    target: str,
    boundaries: Any,
    config: dict[str, Any],
    artifacts_root: Path,
    include_zero: bool,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    modeling = config["modeling"]
    training, eligible_counts, fit_counts = load_training_sample(
        feature_root,
        roots,
        features,
        target,
        boundaries,
        int(modeling["max_train_rows_per_root"]),
    )
    x_train = training.select("root", *features)
    y_train = training[target].to_numpy()
    include_root = scope == "pooled"
    models: dict[str, Any] = {
        "ridge": make_ridge_pipeline(
            features,
            alpha=float(modeling["ridge_alpha"]),
            include_root=include_root,
        ),
        "hist_gradient_boosting": make_hist_gradient_boosting_pipeline(
            features,
            include_root=include_root,
            random_seed=int(modeling["random_seed"]),
            **modeling["hist_gradient_boosting"],
        ),
    }
    if include_zero:
        models = {"zero": ZeroRegressor(), **models}
    results: dict[str, Any] = {
        "eligible_train_counts_by_root": eligible_counts,
        "model_fit_counts_by_root": fit_counts,
        "eligible_train_rows": sum(eligible_counts.values()),
        "model_fit_rows": training.height,
        "original_feature_count": len(features) + int(include_root),
        "models": {},
    }
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        print(f"[{experiment} {scope}] fitting {name} on {training.height:,} rows", flush=True)
        model.fit(x_train, y_train)
        validation, _, _, _ = evaluate_model(
            model,
            feature_root,
            roots,
            features,
            target,
            boundaries,
            "validation",
        )
        print(f"[{experiment} {scope}] final test evaluation: {name}", flush=True)
        test, actual, predicted, prediction_roots = evaluate_model(
            model,
            feature_root,
            roots,
            features,
            target,
            boundaries,
            "test",
        )
        model_result: dict[str, Any] = {"validation": validation, "test": test}
        if name != "zero":
            metadata = {
                "experiment": experiment,
                "scope": scope,
                "model": name,
                "feature_order": features,
                "root_encoding": list(roots) if include_root else [],
                "target": target,
                "split_boundaries": boundaries.to_dict(),
                "random_seed": int(modeling["random_seed"]),
                "configuration": config,
                "eligible_train_counts_by_root": eligible_counts,
                "model_fit_counts_by_root": fit_counts,
            }
            output = artifacts_root / experiment / scope / name
            save_model_artifact(model, output, metadata=metadata)
            model_result["artifact"] = str(output)
            model_result["encoded_feature_count"] = len(
                model.named_steps["preprocessor"].get_feature_names_out()
            )
        prediction_cache[name] = (actual, predicted, prediction_roots)
        results["models"][name] = model_result
    return results, prediction_cache


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config = _config()
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows, create=True)
    paths.configure_process_temp()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.temp_root / "matplotlib"))
    processed = json.loads(
        (paths.processed_data_root / "phase2_processed_manifest.json").read_text(encoding="utf-8")
    )
    starts = [_iso(row["min_timestamp"]) for row in processed["roots"]]
    ends = [_iso(row["max_timestamp"]) for row in processed["roots"]]
    boundaries = calculate_global_boundaries(
        min(starts),
        max(ends),
        train_fraction=float(config["split"]["train_fraction"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        purge_minutes=int(config["purge_minutes"]),
    )
    roots = tuple(config["roots"])
    target = str(config["primary_target"])
    experiments = {
        "experiment_a": (paths.features_root / "experiment_a", list(MODEL_FEATURES_A)),
        "experiment_b": (
            paths.features_root / "experiment_b",
            [*MODEL_FEATURES_A, *CROSS_MARKET_FEATURES],
        ),
    }
    payload: dict[str, Any] = {
        "version": config["version"],
        "target": target,
        "split_boundaries": boundaries.to_dict(),
        "experiments": {},
    }
    plot_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    checkpoint = paths.model_artifact_root / "phase2_model_results.partial.json"
    for experiment, (feature_root, features) in experiments.items():
        payload["experiments"][experiment] = {}
        pooled, cache = _fit_and_evaluate(
            experiment=experiment,
            scope="pooled",
            feature_root=feature_root,
            roots=roots,
            features=features,
            target=target,
            boundaries=boundaries,
            config=config,
            artifacts_root=paths.model_artifact_root,
            include_zero=experiment == "experiment_a",
        )
        payload["experiments"][experiment]["pooled"] = pooled
        plot_cache[f"{experiment}_pooled_hgb"] = cache["hist_gradient_boosting"]
        _write_checkpoint(payload, checkpoint)
        nq_only, _ = _fit_and_evaluate(
            experiment=experiment,
            scope="nq_only",
            feature_root=feature_root,
            roots=("NQ",),
            features=features,
            target=target,
            boundaries=boundaries,
            config=config,
            artifacts_root=paths.model_artifact_root,
            include_zero=experiment == "experiment_a",
        )
        payload["experiments"][experiment]["nq_only"] = nq_only
        _write_checkpoint(payload, checkpoint)

    sample = (
        scan_partitioned_root(paths.features_root / "experiment_b", "NQ")
        .filter(eligibility_expression(target))
        .filter(pl.col("timestamp") < boundaries.train_validation_boundary)
        .select(
            "timestamp",
            "root",
            "instrument_id",
            "contract_segment_id",
            *MODEL_FEATURES_A,
            *CROSS_MARKET_FEATURES,
            "future_return_1m",
            "future_return_5m",
            "future_return_15m",
            "direction_5m",
        )
        .head(1)
        .collect()
        .row(0, named=True)
    )
    payload["real_training_row"] = _row_to_json(sample)
    payload["plot_cache_path"] = str(paths.cache_root / "phase2_plot_predictions.joblib")
    joblib.dump(plot_cache, paths.cache_root / "phase2_plot_predictions.joblib", compress=3)
    strict = finite_json(payload)
    output = paths.model_artifact_root / "phase2_model_results.json"
    output.write_text(json.dumps(strict, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "reports" / "baseline_metrics.json").write_text(
        json.dumps(strict, indent=2), encoding="utf-8"
    )
    print(f"Complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
