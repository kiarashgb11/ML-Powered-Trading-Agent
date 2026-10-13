"""Run the frozen Phase 2.2 expanding-window temporal replication study."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
import yaml
from dotenv import load_dotenv
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.evaluation.walk_forward import build_walk_forward_folds  # noqa: E402
from futures_ml.features.phase2_1 import EXPERIMENT_FEATURES  # noqa: E402
from futures_ml.models.baselines import make_hist_gradient_boosting_pipeline  # noqa: E402
from futures_ml.models.phase2_1_training import experiment_xy  # noqa: E402
from futures_ml.models.training import finite_json, save_model_artifact  # noqa: E402
from futures_ml.models.walk_forward_training import (  # noqa: E402
    evaluate_fold_models,
    load_fold_training_sample,
    root_ranks,
    summarize_values,
)


def _config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "phase2_2.yaml").read_text(encoding="utf-8")
    )


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _model_metadata(
    *,
    config: dict[str, Any],
    fold: Any,
    experiment: str,
    scope: str,
    roots: tuple[str, ...],
    fit_counts: dict[str, int],
) -> dict[str, Any]:
    features = EXPERIMENT_FEATURES[experiment]
    include_root = scope == "pooled"
    return {
        "version": config["version"],
        "phase2_1_commit": config["phase2_1_commit"],
        "fold": fold.to_dict(),
        "target": config["primary_target"],
        "feature_list": list(features),
        "feature_order": list(features),
        "experiment": experiment,
        "scope": scope,
        "model": "hist_gradient_boosting",
        "root_encoding": list(roots) if include_root else [],
        "preprocessor": (
            "fold-train median imputation; pooled root one-hot; no numeric scaling"
            if include_root
            else "fold-train median imputation; no root encoding; no numeric scaling"
        ),
        "training_sample_count": sum(fit_counts.values()),
        "training_sample_counts_by_root": fit_counts,
        "random_seed": int(config["modeling"]["random_seed"]),
        "configuration": config,
        "model_parameters": config["modeling"]["hist_gradient_boosting"],
        "interpretation": "historical temporal replication; not a pristine new holdout",
    }


def _fit_or_load(
    *,
    training: pl.DataFrame,
    config: dict[str, Any],
    fold: Any,
    experiment: str,
    scope: str,
    roots: tuple[str, ...],
    fit_counts: dict[str, int],
    output: Path,
) -> Any:
    model_path = output / "model_pipeline.joblib"
    if model_path.exists():
        print(f"[{fold.fold_id} {experiment} {scope}] resume", flush=True)
        return joblib.load(model_path)
    model = make_hist_gradient_boosting_pipeline(
        list(EXPERIMENT_FEATURES[experiment]),
        include_root=scope == "pooled",
        random_seed=int(config["modeling"]["random_seed"]),
        **config["modeling"]["hist_gradient_boosting"],
    )
    x_train, y_train = experiment_xy(training, experiment)
    print(
        f"[{fold.fold_id} {experiment} {scope}] fitting {len(y_train):,} rows",
        flush=True,
    )
    model.fit(x_train, y_train)
    save_model_artifact(
        model,
        output,
        metadata=_model_metadata(
            config=config,
            fold=fold,
            experiment=experiment,
            scope=scope,
            roots=roots,
            fit_counts=fit_counts,
        ),
    )
    return model


def _assert_counts_match_declaration(
    declaration: dict[str, Any],
    eligible: dict[str, int],
    selected: dict[str, int],
) -> None:
    if eligible != declaration["eligible_training_rows_by_root"]:
        raise RuntimeError("Eligible training counts differ from the predeclared fold")
    if selected != declaration["selected_training_rows_by_root"]:
        raise RuntimeError("Selected training counts differ from the predeclared fold")


def _aggregate_metric_stability(
    folds: list[dict[str, Any]], experiment: str
) -> dict[str, Any]:
    names = ("mae", "rmse", "r2", "pearson", "spearman", "direction_accuracy")
    return {
        name: summarize_values(
            [float(fold["experiments"][experiment]["overall"][name]) for fold in folds]
        )
        for name in names
    }


def _aggregate_comparison(folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        a = fold["experiments"]["A_1m"]["overall"]
        b = fold["experiments"]["B_1m"]["overall"]
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "pearson_delta_b_minus_a": float(b["pearson"] - a["pearson"]),
                "spearman_delta_b_minus_a": float(b["spearman"] - a["spearman"]),
                "r2_delta_b_minus_a": float(b["r2"] - a["r2"]),
            }
        )
    return {
        "folds": rows,
        "pearson": summarize_values([row["pearson_delta_b_minus_a"] for row in rows]),
        "spearman": summarize_values([row["spearman_delta_b_minus_a"] for row in rows]),
        "r2": summarize_values([row["r2_delta_b_minus_a"] for row in rows]),
        "b_beats_a_pearson_folds": sum(row["pearson_delta_b_minus_a"] > 0 for row in rows),
        "a_beats_b_pearson_folds": sum(row["pearson_delta_b_minus_a"] < 0 for row in rows),
        "b_beats_a_spearman_folds": sum(row["spearman_delta_b_minus_a"] > 0 for row in rows),
        "b_beats_a_r2_folds": sum(row["r2_delta_b_minus_a"] > 0 for row in rows),
    }


def _aggregate_root_stability(
    folds: list[dict[str, Any]], roots: tuple[str, ...], experiment: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for root in roots:
        pearson = [
            float(fold["experiments"][experiment]["per_root"][root]["pearson"])
            for fold in folds
        ]
        spearman = [
            float(fold["experiments"][experiment]["per_root"][root]["spearman"])
            for fold in folds
        ]
        ranks = [int(fold["experiments"][experiment]["root_ranks"][root]) for fold in folds]
        output[root] = {
            "pearson": summarize_values(pearson),
            "spearman": summarize_values(spearman),
            "ranks": ranks,
            "average_rank": float(np.mean(ranks)),
            "rank_standard_deviation": float(np.std(ranks)),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
        }
    return output


def _aggregate_group_stability(
    folds: list[dict[str, Any]], experiment: str, key: str
) -> dict[str, Any]:
    groups = sorted(
        {
            str(row["group"])
            for fold in folds
            for row in fold["experiments"][experiment][key]
        }
    )
    output: dict[str, Any] = {}
    for group in groups:
        rows = [
            next(
                row
                for row in fold["experiments"][experiment][key]
                if str(row["group"]) == group
            )
            for fold in folds
        ]
        output[group] = {
            name: summarize_values([float(row[name]) for row in rows])
            for name in ("r2", "pearson", "spearman")
        }
    return output


def _aggregate_deciles(folds: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        deciles = fold["experiments"][experiment]["prediction_deciles"]
        means = np.asarray([float(row["mean_actual"]) for row in deciles])
        ordering = float(spearmanr(np.arange(1, 11), means).statistic)
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "realized_decile_order_spearman": ordering,
                "top_minus_bottom_mean_actual": float(means[-1] - means[0]),
                "fully_non_decreasing": bool(np.all(np.diff(means) >= 0)),
            }
        )
    return {
        "folds": rows,
        "ordering_spearman": summarize_values(
            [row["realized_decile_order_spearman"] for row in rows]
        ),
        "top_minus_bottom": summarize_values(
            [row["top_minus_bottom_mean_actual"] for row in rows]
        ),
        "fully_non_decreasing_folds": sum(row["fully_non_decreasing"] for row in rows),
        "positive_top_minus_bottom_folds": sum(
            row["top_minus_bottom_mean_actual"] > 0 for row in rows
        ),
    }


def _aggregate_dedicated(folds: list[dict[str, Any]], roots: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for root in roots:
        rows = [fold["dedicated_models"][root] for fold in folds]
        output[root] = {
            "pooled_pearson": summarize_values(
                [float(row["pooled_b_metrics"]["pearson"]) for row in rows]
            ),
            "dedicated_pearson": summarize_values(
                [float(row["dedicated_b_metrics"]["pearson"]) for row in rows]
            ),
            "pearson_delta_dedicated_minus_pooled": summarize_values(
                [
                    float(
                        row["dedicated_b_metrics"]["pearson"]
                        - row["pooled_b_metrics"]["pearson"]
                    )
                    for row in rows
                ]
            ),
            "dedicated_beats_pooled_folds": sum(
                row["dedicated_b_metrics"]["pearson"]
                > row["pooled_b_metrics"]["pearson"]
                for row in rows
            ),
        }
    return output


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config = _config()
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows, create=True)
    paths.configure_process_temp()
    source_root = paths.features_root / config["source_experiment"]
    artifact_root = paths.model_artifact_root / "phase2_2"
    declaration_path = PROJECT_ROOT / "reports" / "phase2_2_fold_definitions.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    if declaration["status"] != "predeclared_before_training":
        raise RuntimeError("Fold declaration is not frozen before training")
    folds = build_walk_forward_folds(
        data_start=_iso(config["data_start"]),
        evaluation_boundaries=tuple(_iso(value) for value in config["evaluation_boundaries"]),
        purge_minutes=int(config["purge_minutes"]),
    )
    declaration_by_id = {row["fold_id"]: row for row in declaration["folds"]}
    partial_path = artifact_root / "phase2_2_results.partial.json"
    completed: dict[str, Any] = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("version") == config["version"]:
            completed = {row["fold_id"]: row for row in partial.get("fold_results", [])}

    roots = tuple(config["roots"])
    dedicated_roots = tuple(config["dedicated_roots"])
    cap = int(config["training_rows_per_root"])
    for fold in folds:
        if fold.fold_id in completed:
            print(f"[{fold.fold_id}] completed checkpoint found", flush=True)
            continue
        print(f"[{fold.fold_id}] loading expanding training sample", flush=True)
        training, eligible_counts, fit_counts, thresholds = load_fold_training_sample(
            source_root,
            roots,
            fold,
            cap,
        )
        _assert_counts_match_declaration(
            declaration_by_id[fold.fold_id], eligible_counts, fit_counts
        )
        models: dict[str, Any] = {}
        for experiment in tuple(config["experiments"]):
            models[experiment] = _fit_or_load(
                training=training,
                config=config,
                fold=fold,
                experiment=experiment,
                scope="pooled",
                roots=roots,
                fit_counts=fit_counts,
                output=artifact_root
                / fold.fold_id
                / experiment
                / "pooled"
                / "hist_gradient_boosting",
            )
        print(f"[{fold.fold_id}] evaluating pooled A/B", flush=True)
        experiment_results = evaluate_fold_models(
            models,
            source_root,
            roots,
            fold,
            thresholds,
        )
        declared_evaluation = declaration_by_id[fold.fold_id][
            "eligible_evaluation_rows_by_root"
        ]
        observed_evaluation = {
            root: int(experiment_results["A_1m"]["per_root"][root]["observations"])
            for root in roots
        }
        if observed_evaluation != declared_evaluation:
            raise RuntimeError("Evaluation counts differ from the predeclared fold")
        for experiment in experiment_results:
            experiment_results[experiment]["root_ranks"] = root_ranks(
                experiment_results[experiment]["per_root"]
            )

        dedicated_results: dict[str, Any] = {}
        for root in dedicated_roots:
            root_training = training.filter(pl.col("root") == root)
            model = _fit_or_load(
                training=root_training,
                config=config,
                fold=fold,
                experiment="B_1m",
                scope=f"dedicated_{root}",
                roots=(root,),
                fit_counts={root: fit_counts[root]},
                output=artifact_root
                / fold.fold_id
                / "B_1m"
                / "dedicated"
                / root
                / "hist_gradient_boosting",
            )
            evaluation = evaluate_fold_models(
                {"B_1m": model},
                source_root,
                (root,),
                fold,
                {root: thresholds[root]},
            )["B_1m"]["overall"]
            dedicated_results[root] = {
                "pooled_b_metrics": experiment_results["B_1m"]["per_root"][root],
                "dedicated_b_metrics": evaluation,
            }
        completed[fold.fold_id] = {
            "fold_id": fold.fold_id,
            "definition": fold.to_dict(),
            "eligible_training_rows_by_root": eligible_counts,
            "selected_training_rows_by_root": fit_counts,
            "volatility_thresholds_by_root": thresholds,
            "experiments": experiment_results,
            "dedicated_models": dedicated_results,
        }
        checkpoint = {
            "version": config["version"],
            "status": "running",
            "fold_declaration": declaration,
            "fold_results": [completed[key] for key in sorted(completed)],
        }
        _write_json(checkpoint, partial_path)

    fold_results = [completed[fold.fold_id] for fold in folds]
    aggregate: dict[str, Any] = {
        "model_stability": {
            experiment: _aggregate_metric_stability(fold_results, experiment)
            for experiment in tuple(config["experiments"])
        },
        "a_vs_b": _aggregate_comparison(fold_results),
        "root_stability": {
            experiment: _aggregate_root_stability(fold_results, roots, experiment)
            for experiment in tuple(config["experiments"])
        },
        "volatility_regime_stability": {
            experiment: _aggregate_group_stability(
                fold_results, experiment, "volatility_regimes"
            )
            for experiment in tuple(config["experiments"])
        },
        "time_of_day_stability": {
            experiment: _aggregate_group_stability(
                fold_results, experiment, "chicago_hours"
            )
            for experiment in tuple(config["experiments"])
        },
        "prediction_decile_stability": {
            experiment: _aggregate_deciles(fold_results, experiment)
            for experiment in tuple(config["experiments"])
        },
        "dedicated_model_stability": _aggregate_dedicated(
            fold_results, dedicated_roots
        ),
    }
    payload = {
        "version": config["version"],
        "status": "complete",
        "interpretation": (
            "historical temporal stability replication; Phase 2.1 already inspected later periods, "
            "so no fold is described as a pristine new final test"
        ),
        "configuration": config,
        "fold_declaration": declaration,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _write_json(payload, artifact_root / "phase2_2_results.json")
    _write_json(payload, PROJECT_ROOT / "reports" / "phase2_2_walk_forward_metrics.json")
    print(f"Complete: {artifact_root / 'phase2_2_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
