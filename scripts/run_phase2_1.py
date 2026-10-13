"""Run the Phase 2.1 one-minute cross-market ablation without rebuilding data."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
import yaml
from dotenv import load_dotenv
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.evaluation.splits import calculate_global_boundaries  # noqa: E402
from futures_ml.features.phase2_1 import (  # noqa: E402
    EXPERIMENT_FEATURES,
    RAW_ANCHOR_FEATURES,
    RELATIVE_ANCHOR_FEATURES,
    add_relative_market_features,
)
from futures_ml.models.baselines import (  # noqa: E402
    PersistenceRegressor,
    ReversalRegressor,
    ZeroRegressor,
    make_hist_gradient_boosting_pipeline,
    make_ridge_pipeline,
)
from futures_ml.models.phase2_1_training import (  # noqa: E402
    TARGET,
    evaluate_models,
    experiment_xy,
    grouped_metrics,
    load_common_training_samples,
    load_evaluation_root,
)
from futures_ml.models.streaming_ridge import fit_streaming_ridge  # noqa: E402
from futures_ml.models.training import (  # noqa: E402
    deterministic_even_sample,
    finite_json,
    save_model_artifact,
)


def _config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "phase2_1.yaml").read_text(encoding="utf-8")
    )


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False), encoding="utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata(
    *,
    experiment: str,
    scope: str,
    model_name: str,
    features: tuple[str, ...],
    roots: tuple[str, ...],
    boundaries: Any,
    config: dict[str, Any],
    fit_counts: dict[str, int],
    parameters: dict[str, Any],
    preprocessor: str,
) -> dict[str, Any]:
    split = boundaries.to_dict()
    return {
        "version": config["version"],
        "target": TARGET,
        "feature_list": list(features),
        "feature_order": list(features),
        "experiment": experiment,
        "scope": scope,
        "model": model_name,
        "root_encoding": list(roots) if scope == "pooled" else [],
        "preprocessor": preprocessor,
        "train_period": {
            "start": split["data_start"],
            "end_exclusive_before_purge": split["train_validation_boundary"],
        },
        "validation_period": {
            "boundary_start": split["train_validation_boundary"],
            "boundary_end": split["validation_test_boundary"],
        },
        "test_period": {
            "boundary_start": split["validation_test_boundary"],
            "end": split["data_end"],
        },
        "purge_minutes_each_side": split["purge_minutes"],
        "training_sample_count": int(sum(fit_counts.values())),
        "training_sample_counts_by_root": fit_counts,
        "random_seed": int(config["modeling"]["random_seed"]),
        "configuration": config,
        "model_parameters": parameters,
    }


def _fit_or_load(
    *,
    experiment: str,
    scope: str,
    model_name: str,
    training: pl.DataFrame,
    roots: tuple[str, ...],
    boundaries: Any,
    config: dict[str, Any],
    artifact_root: Path,
    fit_counts: dict[str, int],
) -> Any:
    features = EXPERIMENT_FEATURES[experiment]
    output = artifact_root / experiment / scope / model_name
    model_path = output / "model_pipeline.joblib"
    include_root = scope == "pooled"
    root_step = "pooled root one-hot" if include_root else "no root encoding"
    if model_name == "ridge":
        parameters = {"alpha": float(config["modeling"]["ridge_alpha"])}
        preprocessor = f"train median imputation; train StandardScaler; {root_step}"
    elif model_name == "hist_gradient_boosting":
        parameters = dict(config["modeling"]["hist_gradient_boosting"])
        preprocessor = f"train median imputation; {root_step}; no numeric scaling"
    else:
        raise ValueError(model_name)
    if model_path.exists():
        print(f"[{experiment} {scope}] resume {model_name}", flush=True)
        metadata_path = output / "metadata.json"
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing["preprocessor"] = preprocessor
        _write_json(existing, metadata_path)
        return joblib.load(model_path)
    if model_name == "ridge":
        model = make_ridge_pipeline(
            list(features),
            alpha=parameters["alpha"],
            include_root=include_root,
        )
    else:
        model = make_hist_gradient_boosting_pipeline(
            list(features),
            include_root=include_root,
            random_seed=int(config["modeling"]["random_seed"]),
            **parameters,
        )
    x_train, y_train = experiment_xy(training, experiment)
    print(f"[{experiment} {scope}] fitting {model_name} on {len(y_train):,} rows", flush=True)
    model.fit(x_train, y_train)
    save_model_artifact(
        model,
        output,
        metadata=_metadata(
            experiment=experiment,
            scope=scope,
            model_name=model_name,
            features=features,
            roots=roots,
            boundaries=boundaries,
            config=config,
            fit_counts=fit_counts,
            parameters=parameters,
            preprocessor=preprocessor,
        ),
    )
    return model


def _assign_results(
    payload: dict[str, Any],
    evaluation: dict[str, Any],
    split: str,
    *,
    scope: str = "pooled",
) -> None:
    for compound, metrics in evaluation.items():
        experiment, model_name = compound.split("/", 1)
        models = payload["experiments"][experiment][scope].setdefault("models", {})
        models.setdefault(model_name, {})[split] = metrics


def _coefficient_importance(model: Any, limit: int = 15) -> list[dict[str, Any]]:
    names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    coefficients = np.asarray(model.named_steps["model"].coef_, dtype=float)
    order = np.argsort(np.abs(coefficients))[::-1][:limit]
    return [
        {
            "feature": names[index].replace("numeric__", "").replace("root__", ""),
            "coefficient": float(coefficients[index]),
            "sign": "positive" if coefficients[index] >= 0 else "negative",
            "magnitude": float(abs(coefficients[index])),
        }
        for index in order
    ]


def _validation_importance_sample(
    source_root: Path,
    roots: tuple[str, ...],
    boundaries: Any,
    rows_per_root: int,
) -> pl.DataFrame:
    parts = [
        deterministic_even_sample(
            load_evaluation_root(source_root, root, boundaries, "validation"),
            rows_per_root,
        )
        for root in roots
    ]
    return pl.concat(parts, rechunk=True)


def _permutation_importance(
    model: Any,
    source: pl.DataFrame,
    experiment: str,
    *,
    repeats: int,
    seed: int,
    limit: int = 15,
) -> list[dict[str, Any]]:
    x, y = experiment_xy(source, experiment)
    baseline = r2_score(y, model.predict(x))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for name in x.columns:
        original = x[name].to_numpy()
        decreases: list[float] = []
        for _ in range(repeats):
            permuted = x.with_columns(pl.Series(name, original[rng.permutation(len(original))]))
            decreases.append(float(baseline - r2_score(y, model.predict(permuted))))
        rows.append(
            {
                "feature": name,
                "mean_validation_r2_decrease": float(np.mean(decreases)),
                "standard_deviation": float(np.std(decreases)),
            }
        )
    rows.sort(key=lambda row: row["mean_validation_r2_decrease"], reverse=True)
    return [
        row
        for row in rows[:limit]
    ]


def _fit_size_model(
    experiment: str,
    training: pl.DataFrame,
    config: dict[str, Any],
) -> Any:
    model = make_hist_gradient_boosting_pipeline(
        list(EXPERIMENT_FEATURES[experiment]),
        include_root=True,
        random_seed=int(config["modeling"]["random_seed"]),
        **config["modeling"]["hist_gradient_boosting"],
    )
    x, y = experiment_xy(training, experiment)
    model.fit(x, y)
    return model


def _full_training_batches(
    source_root: Path,
    roots: tuple[str, ...],
    boundaries: Any,
    experiment: str,
) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    for root in roots:
        frame = load_evaluation_root(source_root, root, boundaries, "train")
        yield experiment_xy(frame, experiment)


def _save_streaming_ridge(
    model: Any,
    output: Path,
    metadata: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model_pipeline.joblib", compress=3)
    enriched = {
        **metadata,
        "encoded_feature_order": model.get_feature_names_out(),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _write_json(enriched, output / "metadata.json")


def _volatility_thresholds(training: pl.DataFrame, roots: tuple[str, ...]) -> dict[str, list[float]]:
    thresholds: dict[str, list[float]] = {}
    for root in roots:
        values = (
            training.filter(pl.col("root") == root)["realized_vol_15m"]
            .filter(training.filter(pl.col("root") == root)["realized_vol_15m"].is_finite())
        )
        thresholds[root] = [float(values.quantile(1 / 3)), float(values.quantile(2 / 3))]
    return thresholds


def _diagnostics(cache: dict[str, np.ndarray], thresholds: dict[str, list[float]]) -> dict[str, Any]:
    actual = cache["actual"]
    predicted = cache["predicted"]
    roots = cache["root"]
    volatility = cache["realized_vol_15m"].astype(float)
    regimes = np.full(len(actual), "unavailable", dtype="U11")
    for root, (low, high) in thresholds.items():
        root_mask = roots == root
        finite = root_mask & np.isfinite(volatility)
        regimes[finite & (volatility <= low)] = "low"
        regimes[finite & (volatility > low) & (volatility <= high)] = "medium"
        regimes[finite & (volatility > high)] = "high"
    weekday_names = np.asarray(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    weekdays = weekday_names[cache["chicago_day_of_week"].astype(int)]
    return {
        "by_chicago_hour": grouped_metrics(
            actual,
            predicted,
            cache["chicago_hour"].astype(int),
        ),
        "volatility_feature": "realized_vol_15m",
        "volatility_thresholds_by_root_from_primary_train": thresholds,
        "by_volatility_regime": grouped_metrics(actual, predicted, regimes),
        "by_chicago_weekday": grouped_metrics(actual, predicted, weekdays),
    }


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config = _config()
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows, create=True)
    paths.configure_process_temp()
    source_root = paths.features_root / str(config["source_experiment"])
    artifact_root = paths.model_artifact_root / "phase2_1"
    checkpoint = artifact_root / "phase2_1_results.partial.json"
    processed = json.loads(
        (paths.processed_data_root / "phase2_processed_manifest.json").read_text(
            encoding="utf-8"
        )
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
    caps = tuple(sorted(set(config["modeling"]["hgb_sensitivity_rows_per_root"])))
    primary_cap = int(config["modeling"]["primary_train_rows_per_root"])
    print("Loading common direct samples for A/B/C", flush=True)
    samples, eligible_counts, fit_counts = load_common_training_samples(
        source_root,
        roots,
        boundaries,
        caps,
    )
    primary = samples[primary_cap]
    payload: dict[str, Any] = {
        "version": config["version"],
        "status": "running",
        "source_feature_version": config["source_feature_version"],
        "source_feature_root": str(source_root),
        "target": TARGET,
        "split_boundaries": boundaries.to_dict(),
        "configuration": config,
        "eligible_train_counts_by_root": eligible_counts,
        "training_sample_counts_by_cap": fit_counts,
        "primary_training_rows": primary.height,
        "experiments": {
            experiment: {
                "numeric_feature_count": len(features),
                "feature_order": list(features),
                "pooled": {},
                "nq_only": {},
            }
            for experiment, features in EXPERIMENT_FEATURES.items()
        },
    }

    pooled_models: dict[str, Any] = {}
    nq_models: dict[str, Any] = {}
    primary_counts = fit_counts[primary_cap]
    nq_training = primary.filter(pl.col("root") == "NQ")
    for experiment in EXPERIMENT_FEATURES:
        for model_name in ("ridge", "hist_gradient_boosting"):
            pooled_models[f"{experiment}/{model_name}"] = _fit_or_load(
                experiment=experiment,
                scope="pooled",
                model_name=model_name,
                training=primary,
                roots=roots,
                boundaries=boundaries,
                config=config,
                artifact_root=artifact_root,
                fit_counts=primary_counts,
            )
            nq_models[f"{experiment}/{model_name}"] = _fit_or_load(
                experiment=experiment,
                scope="nq_only",
                model_name=model_name,
                training=nq_training,
                roots=("NQ",),
                boundaries=boundaries,
                config=config,
                artifact_root=artifact_root,
                fit_counts={"NQ": primary_counts["NQ"]},
            )
        pooled_models[f"{experiment}/zero"] = ZeroRegressor()
        pooled_models[f"{experiment}/persistence"] = PersistenceRegressor()
        pooled_models[f"{experiment}/reversal"] = ReversalRegressor()

    print("Validation evaluation on every eligible observation", flush=True)
    pooled_evaluation_models = {
        name: (name.split("/", 1)[0], model) for name, model in pooled_models.items()
    }
    nq_evaluation_models = {
        name: (name.split("/", 1)[0], model) for name, model in nq_models.items()
    }
    validation, _ = evaluate_models(
        pooled_evaluation_models,
        source_root,
        roots,
        boundaries,
        "validation",
    )
    _assign_results(payload, validation, "validation")
    nq_validation, _ = evaluate_models(
        nq_evaluation_models,
        source_root,
        ("NQ",),
        boundaries,
        "validation",
    )
    _assign_results(payload, nq_validation, "validation", scope="nq_only")
    _write_json(payload, checkpoint)

    best_hgb_experiment = max(
        EXPERIMENT_FEATURES,
        key=lambda name: validation[f"{name}/hist_gradient_boosting"]["overall"]["pearson"],
    )
    best_ridge_experiment = max(
        EXPERIMENT_FEATURES,
        key=lambda name: validation[f"{name}/ridge"]["overall"]["pearson"],
    )
    payload["validation_selections"] = {
        "best_primary_hgb_experiment": best_hgb_experiment,
        "best_primary_ridge_experiment": best_ridge_experiment,
        "selection_metric": "pooled validation Pearson",
    }

    print(f"HGB training-size sensitivity: {best_hgb_experiment}", flush=True)
    sensitivity: dict[str, Any] = {}
    sensitivity_models: dict[int, Any] = {}
    for cap in caps:
        if cap == primary_cap:
            model = pooled_models[f"{best_hgb_experiment}/hist_gradient_boosting"]
        else:
            model = _fit_size_model(best_hgb_experiment, samples[cap], config)
            output = artifact_root / "training_size" / best_hgb_experiment / f"{cap}_per_root"
            save_model_artifact(
                model,
                output,
                metadata=_metadata(
                    experiment=best_hgb_experiment,
                    scope="pooled",
                    model_name="hist_gradient_boosting_training_size",
                    features=EXPERIMENT_FEATURES[best_hgb_experiment],
                    roots=roots,
                    boundaries=boundaries,
                    config=config,
                    fit_counts=fit_counts[cap],
                    parameters=config["modeling"]["hist_gradient_boosting"],
                    preprocessor="train median imputation; pooled root one-hot; no scaling",
                ),
            )
        sensitivity_models[cap] = model
        evaluated, _ = evaluate_models(
            {f"{best_hgb_experiment}/hgb_{cap}": (best_hgb_experiment, model)},
            source_root,
            roots,
            boundaries,
            "validation",
        )
        sensitivity[str(cap)] = evaluated[f"{best_hgb_experiment}/hgb_{cap}"]
        _write_json(payload | {"training_size_sensitivity": sensitivity}, checkpoint)
    chosen_hgb_cap = max(
        caps,
        key=lambda cap: sensitivity[str(cap)]["overall"]["pearson"],
    )
    payload["training_size_sensitivity"] = {
        "experiment": best_hgb_experiment,
        "selection_split": "validation",
        "rows_per_root": sensitivity,
        "chosen_rows_per_root": chosen_hgb_cap,
    }

    if bool(config["modeling"].get("run_full_ridge", False)):
        print(f"Full-training Ridge: {best_ridge_experiment}", flush=True)
        capped_ridge = pooled_models[f"{best_ridge_experiment}/ridge"]
        medians = capped_ridge.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ].named_steps["imputer"].statistics_
        full_output = artifact_root / best_ridge_experiment / "pooled" / "ridge_full_train"
        full_path = full_output / "model_pipeline.joblib"
        if full_path.exists():
            full_ridge = joblib.load(full_path)
        else:
            full_ridge = fit_streaming_ridge(
                lambda: _full_training_batches(
                    source_root,
                    roots,
                    boundaries,
                    best_ridge_experiment,
                ),
                features=EXPERIMENT_FEATURES[best_ridge_experiment],
                roots=roots,
                medians=np.asarray(medians, dtype=float),
                alpha=float(config["modeling"]["ridge_alpha"]),
            )
            full_counts = eligible_counts
            _save_streaming_ridge(
                full_ridge,
                full_output,
                _metadata(
                    experiment=best_ridge_experiment,
                    scope="pooled",
                    model_name="ridge_full_train",
                    features=EXPERIMENT_FEATURES[best_ridge_experiment],
                    roots=roots,
                    boundaries=boundaries,
                    config=config,
                    fit_counts=full_counts,
                    parameters={"alpha": float(config["modeling"]["ridge_alpha"])},
                    preprocessor=(
                        "primary-train-sample median imputation; all-train StandardScaler; "
                        "pooled root one-hot; two-pass sufficient-statistics Ridge"
                    ),
                ),
            )
        full_validation, _ = evaluate_models(
            {f"{best_ridge_experiment}/ridge_full_train": (best_ridge_experiment, full_ridge)},
            source_root,
            roots,
            boundaries,
            "validation",
        )
        full_metrics = full_validation[f"{best_ridge_experiment}/ridge_full_train"]
        capped_metrics = validation[f"{best_ridge_experiment}/ridge"]
        full_supported = (
            full_metrics["overall"]["pearson"] > capped_metrics["overall"]["pearson"]
        )
        payload["full_ridge"] = {
            "experiment": best_ridge_experiment,
            "capped_validation": capped_metrics,
            "full_validation": full_metrics,
            "validation_supports_full": full_supported,
            "median_note": (
                "Imputation medians are learned from the deterministic 1.6M training sample; "
                "all eligible train rows determine scaling statistics and Ridge coefficients."
            ),
        }
    else:
        full_ridge = None

    print("Validation-only feature importance", flush=True)
    importance_source = _validation_importance_sample(
        source_root,
        roots,
        boundaries,
        int(config["modeling"]["permutation_rows_per_root"]),
    )
    payload["feature_importance"] = {}
    for experiment in EXPERIMENT_FEATURES:
        ridge = pooled_models[f"{experiment}/ridge"]
        hgb = pooled_models[f"{experiment}/hist_gradient_boosting"]
        payload["feature_importance"][experiment] = {
            "ridge_standardized_coefficients": _coefficient_importance(ridge),
            "hgb_validation_permutation": _permutation_importance(
                hgb,
                importance_source,
                experiment,
                repeats=int(config["modeling"]["permutation_repeats"]),
                seed=int(config["modeling"]["random_seed"]),
            ),
            "permutation_observations": importance_source.height,
            "permutation_repeats": int(config["modeling"]["permutation_repeats"]),
        }

    ranking = validation[f"{best_hgb_experiment}/hist_gradient_boosting"]["per_root"]
    market_count = int(config["modeling"]["market_specific_roots"])
    selected_roots = tuple(
        sorted(
            roots,
            key=lambda root: float(ranking[root]["pearson"]),
            reverse=True,
        )[:market_count]
    )
    print(f"Market-specific HGB validation diagnostic: {selected_roots}", flush=True)
    market_models: dict[str, tuple[str, Any]] = {}
    for root in selected_roots:
        training = primary.filter(pl.col("root") == root)
        model = make_hist_gradient_boosting_pipeline(
            list(EXPERIMENT_FEATURES[best_hgb_experiment]),
            include_root=False,
            random_seed=int(config["modeling"]["random_seed"]),
            **config["modeling"]["hist_gradient_boosting"],
        )
        x, y = experiment_xy(training, best_hgb_experiment)
        model.fit(x, y)
        market_models[root] = (best_hgb_experiment, model)
        save_model_artifact(
            model,
            artifact_root / "market_specific" / best_hgb_experiment / root,
            metadata=_metadata(
                experiment=best_hgb_experiment,
                scope=f"market_specific_{root}",
                model_name="hist_gradient_boosting",
                features=EXPERIMENT_FEATURES[best_hgb_experiment],
                roots=(root,),
                boundaries=boundaries,
                config=config,
                fit_counts={root: training.height},
                parameters=config["modeling"]["hist_gradient_boosting"],
                preprocessor="train median imputation; no root encoding; no scaling",
            ),
        )
    market_specific: dict[str, Any] = {}
    for root, (experiment, model) in market_models.items():
        val_result, _ = evaluate_models(
            {f"{experiment}/market_specific": (experiment, model)},
            source_root,
            (root,),
            boundaries,
            "validation",
        )
        market_specific[root] = {
            "selection_basis": ranking[root],
            "pooled_validation": ranking[root],
            "market_specific_validation": val_result[f"{experiment}/market_specific"][
                "overall"
            ],
        }
    payload["market_specific_diagnostic"] = {
        "experiment": best_hgb_experiment,
        "roots_selected_on_validation": list(selected_roots),
        "results": market_specific,
    }
    _write_json(payload, checkpoint)

    print("Final test evaluation on every eligible observation", flush=True)
    retain = {
        f"{experiment}/hist_gradient_boosting" for experiment in EXPERIMENT_FEATURES
    }
    test, prediction_cache = evaluate_models(
        pooled_evaluation_models,
        source_root,
        roots,
        boundaries,
        "test",
        retain_metadata_for=retain,
    )
    _assign_results(payload, test, "test")
    nq_test, _ = evaluate_models(
        nq_evaluation_models,
        source_root,
        ("NQ",),
        boundaries,
        "test",
    )
    _assign_results(payload, nq_test, "test", scope="nq_only")

    chosen_size_model = sensitivity_models[chosen_hgb_cap]
    chosen_size_test, _ = evaluate_models(
        {f"{best_hgb_experiment}/chosen_size": (best_hgb_experiment, chosen_size_model)},
        source_root,
        roots,
        boundaries,
        "test",
    )
    payload["training_size_sensitivity"]["chosen_test"] = chosen_size_test[
        f"{best_hgb_experiment}/chosen_size"
    ]
    if full_ridge is not None and payload["full_ridge"]["validation_supports_full"]:
        full_test, _ = evaluate_models(
            {f"{best_ridge_experiment}/ridge_full_train": (best_ridge_experiment, full_ridge)},
            source_root,
            roots,
            boundaries,
            "test",
        )
        payload["full_ridge"]["chosen_test"] = full_test[
            f"{best_ridge_experiment}/ridge_full_train"
        ]

    for root, (experiment, model) in market_models.items():
        test_result, _ = evaluate_models(
            {f"{experiment}/market_specific": (experiment, model)},
            source_root,
            (root,),
            boundaries,
            "test",
        )
        payload["market_specific_diagnostic"]["results"][root][
            "pooled_test"
        ] = test[f"{experiment}/hist_gradient_boosting"]["per_root"][root]
        payload["market_specific_diagnostic"]["results"][root][
            "market_specific_test"
        ] = test_result[f"{experiment}/market_specific"]["overall"]

    thresholds = _volatility_thresholds(primary, roots)
    payload["test_diagnostics"] = {
        experiment: _diagnostics(
            prediction_cache[f"{experiment}/hist_gradient_boosting"],
            thresholds,
        )
        for experiment in EXPERIMENT_FEATURES
    }
    sample_source = (
        primary.filter(pl.col("root") == "NQ")
        .drop_nulls(list(RAW_ANCHOR_FEATURES))
        .head(1)
    )
    relative_sample = add_relative_market_features(sample_source)
    assert isinstance(relative_sample, pl.DataFrame)
    sample_columns = [
        "timestamp",
        "root",
        "instrument_id",
        "contract_segment_id",
        *EXPERIMENT_FEATURES["A_1m"],
        *RAW_ANCHOR_FEATURES,
        *RELATIVE_ANCHOR_FEATURES,
        TARGET,
    ]
    payload["real_training_row"] = {
        key: _json_value(value)
        for key, value in relative_sample.select(*sample_columns).row(0, named=True).items()
    }
    payload["prediction_cache_path"] = str(paths.cache_root / "phase2_1_predictions.joblib")
    joblib.dump(
        prediction_cache,
        paths.cache_root / "phase2_1_predictions.joblib",
        compress=3,
    )
    payload["status"] = "complete"
    final_external = artifact_root / "phase2_1_results.json"
    _write_json(payload, final_external)
    _write_json(payload, PROJECT_ROOT / "reports" / "phase2_1_metrics.json")
    print(f"Complete: {final_external}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
