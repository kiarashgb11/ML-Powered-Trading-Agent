"""Generate and score append-only post-cutoff Phase 2.3 predictions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.features.phase2_1 import EXPERIMENT_FEATURES  # noqa: E402
from futures_ml.models.prospective import (  # noqa: E402
    OUTCOME_EVENT,
    PHASE2_3_ROOTS,
    PREDICTION_EVENT,
    append_unique_events,
    attach_outcomes,
    evaluate_model_events,
    load_prospective_inputs,
    load_prospective_outcomes,
    predict_frozen_model,
    read_event_log,
    verify_frozen_artifacts,
)
from futures_ml.models.training import finite_json  # noqa: E402


def _config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "phase2_3.yaml").read_text(encoding="utf-8")
    )


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _checkpoint_results(
    *,
    manifest: dict[str, Any],
    events: pl.DataFrame,
    inputs: dict[str, pl.DataFrame],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    cutoff = _iso(checkpoint["start_exclusive"])
    end = _iso(checkpoint["end_exclusive"])
    model_results: dict[str, Any] = {}
    for entry in manifest["models"]:
        root = entry["root"]
        model_id = entry["model_id"]
        completed = events.filter(
            (pl.col("event_type") == OUTCOME_EVENT)
            & (pl.col("model_id") == model_id)
            & (pl.col("timestamp") > cutoff)
            & (pl.col("timestamp") < end)
        )
        feature_window = inputs[root].filter(
            (pl.col("timestamp") > cutoff) & (pl.col("timestamp") < end)
        )
        model_results[model_id] = evaluate_model_events(
            completed,
            feature_window,
            historical_profile=entry["historical_feature_profile"],
            features=EXPERIMENT_FEATURES[entry["experiment"]],
            volatility_thresholds=[
                entry["volatility_thresholds"]["low_max"],
                entry["volatility_thresholds"]["medium_max"],
            ],
        )
    comparisons: dict[str, Any] = {}
    for root in PHASE2_3_ROOTS:
        a = model_results[f"{root}-A-final"]["overall"]
        b = model_results[f"{root}-B-final"]["overall"]
        comparisons[root] = {
            "b_minus_a_pearson": b["pearson"] - a["pearson"],
            "b_minus_a_spearman": b["spearman"] - a["spearman"],
            "b_minus_a_r2": b["r2"] - a["r2"],
            "selection_status": "pending_research_review",
        }
    return {
        "days": checkpoint["days"],
        "role": checkpoint["role"],
        "start_exclusive": checkpoint["start_exclusive"],
        "end_exclusive": checkpoint["end_exclusive"],
        "per_model_metrics": model_results,
        "a_vs_b_deltas": comparisons,
    }


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    paths = resolve_data_paths(create=True)
    paths.configure_process_temp()
    config = _config()
    cutoff = _iso(config["historical_cutoff"])
    artifact_root = paths.model_artifact_root / "phase2_3"
    manifest_path = artifact_root / "model_freeze_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Freeze the Phase 2.3 candidates before prospective evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_frozen_artifacts(manifest)
    feature_root = paths.features_root / config["source_experiment"]
    log_root = paths.root / "artifacts" / "prospective" / "phase2_3" / "prediction_log"
    now = datetime.now(timezone.utc)

    inputs = {
        root: load_prospective_inputs(feature_root, root, cutoff)
        for root in PHASE2_3_ROOTS
    }
    available = {root: frame.height for root, frame in inputs.items()}
    observed_max = {
        root: frame["timestamp"].max().isoformat() if frame.height else None
        for root, frame in inputs.items()
    }
    if any(
        frame.height and frame["timestamp"].min() <= cutoff for frame in inputs.values()
    ):
        raise RuntimeError("Historical rows reached the prospective input set")

    appended_predictions = 0
    for entry in manifest["models"]:
        root = entry["root"]
        if inputs[root].is_empty():
            continue
        model = joblib.load(entry["artifact_path"])
        events = predict_frozen_model(
            model,
            inputs[root],
            model_id=entry["model_id"],
            experiment=entry["experiment"],
            data_version=config["data_version"],
            recorded_at_utc=now,
        )
        appended_predictions += append_unique_events(events, log_root)

    current_events = read_event_log(log_root)
    appended_outcomes = 0
    if not current_events.is_empty():
        for root in PHASE2_3_ROOTS:
            predictions = current_events.filter(
                (pl.col("root") == root) & (pl.col("event_type") == PREDICTION_EVENT)
            )
            if predictions.is_empty():
                continue
            outcomes = load_prospective_outcomes(feature_root, root, cutoff)
            attached = attach_outcomes(predictions, outcomes, recorded_at_utc=now)
            appended_outcomes += append_unique_events(attached, log_root)
        current_events = read_event_log(log_root)

    completed_checkpoints: list[dict[str, Any]] = []
    for checkpoint in manifest["predeclared_checkpoints"]:
        end = _iso(checkpoint["end_exclusive"])
        # Evidence after the endpoint proves the fixed calendar window is closed,
        # even when a market has no qualifying bar exactly at the endpoint.
        if all(
            frame.height and frame["timestamp"].max() >= end
            for frame in inputs.values()
        ):
            completed_checkpoints.append(
                _checkpoint_results(
                    manifest=manifest,
                    events=current_events,
                    inputs=inputs,
                    checkpoint=checkpoint,
                )
            )

    total_rows = sum(available.values())
    status = "awaiting_prospective_data" if total_rows == 0 else "collecting_prospective_data"
    if any(row["days"] == config["primary_checkpoint_days"] for row in completed_checkpoints):
        status = "primary_checkpoint_ready_for_review"
    latest = completed_checkpoints[-1] if completed_checkpoints else None
    latest_models = latest["per_model_metrics"] if latest else {}
    payload = {
        "version": config["version"],
        "status": status,
        "freeze_manifest_reference": str(manifest_path),
        "freeze_manifest_frozen_at_utc": manifest["frozen_at_utc"],
        "historical_cutoff": manifest["historical_cutoff"],
        "prospective_timestamp_rule": "strictly greater than historical_cutoff",
        "model_ids": [entry["model_id"] for entry in manifest["models"]],
        "predeclared_checkpoints": manifest["predeclared_checkpoints"],
        "current_data": {
            "rows_by_root": available,
            "total_source_rows": total_rows,
            "maximum_timestamp_by_root": observed_max,
            "prediction_events_appended_this_run": appended_predictions,
            "outcome_events_appended_this_run": appended_outcomes,
            "log_path": str(log_root),
        },
        "evaluation_window": (
            {
                "days": latest["days"],
                "role": latest["role"],
                "start_exclusive": latest["start_exclusive"],
                "end_exclusive": latest["end_exclusive"],
            }
            if latest
            else None
        ),
        "data_start": latest["start_exclusive"] if latest else None,
        "data_end": latest["end_exclusive"] if latest else None,
        "observations": {
            model_id: row["overall"]["observations"]
            for model_id, row in latest_models.items()
        },
        "overall_metrics": {
            model_id: row["overall"] for model_id, row in latest_models.items()
        },
        "per_model_metrics": latest_models,
        "a_vs_b_deltas": latest["a_vs_b_deltas"] if latest else {},
        "decile_metrics": {
            model_id: {
                "deciles": row["prediction_deciles"],
                "summary": row["decile_summary"],
            }
            for model_id, row in latest_models.items()
        },
        "volatility_regime_metrics": {
            model_id: row["volatility_regimes"]
            for model_id, row in latest_models.items()
        },
        "drift_metrics": {
            model_id: row["drift"] for model_id, row in latest_models.items()
        },
        "completed_windows": completed_checkpoints,
        "selection_status": "no_model_selected",
        "results_note": (
            "No historical observation is substituted for prospective evidence."
            if total_rows == 0
            else "Models remain frozen; completed calendar checkpoints only are reported."
        ),
    }
    output = PROJECT_ROOT / "reports" / "phase2_3_prospective_metrics.json"
    _write_json(payload, output)
    _write_json(payload, artifact_root / "phase2_3_prospective_metrics.json")
    print(f"Phase 2.3 prospective status: {status}; source rows={total_rows}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
