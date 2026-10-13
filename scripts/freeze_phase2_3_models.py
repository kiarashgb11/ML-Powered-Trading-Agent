"""Train and hash the six immutable Phase 2.3 prospective candidates."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.features.phase2_1 import EXPERIMENT_FEATURES  # noqa: E402
from futures_ml.models.baselines import make_hist_gradient_boosting_pipeline  # noqa: E402
from futures_ml.models.phase2_1_training import TARGET  # noqa: E402
from futures_ml.models.prospective import (  # noqa: E402
    PHASE2_3_EXPERIMENTS,
    PHASE2_3_ROOTS,
    distribution_profile,
    load_historical_training_sample,
    selected_row_hash,
    sha256_file,
    training_medians,
    verify_frozen_artifacts,
)
from futures_ml.models.training import finite_json, save_model_artifact  # noqa: E402


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


def _fit_one(
    *,
    root: str,
    experiment: str,
    training: pl.DataFrame,
    eligible_count: int,
    volatility_thresholds: list[float],
    config: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    short_experiment = experiment[0]
    model_id = f"{root}-{short_experiment}-final"
    output = artifact_root / root / short_experiment
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {output}")
    output.mkdir(parents=True)
    features = EXPERIMENT_FEATURES[experiment]
    selected_columns = (
        "timestamp",
        "root",
        "instrument_id",
        "contract_segment_id",
    )
    selected_path = output / "selected_training_rows.parquet"
    training.select(*selected_columns).write_parquet(selected_path, compression="zstd")
    logical_row_hash = selected_row_hash(training)

    model = make_hist_gradient_boosting_pipeline(
        list(features),
        include_root=False,
        random_seed=int(config["modeling"]["random_seed"]),
        **config["modeling"]["hist_gradient_boosting"],
    )
    model.fit(training.select(*features), training[TARGET].to_numpy())
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version": config["version"],
        "phase2_2_commit": config["phase2_2_commit"],
        "model_id": model_id,
        "root": root,
        "experiment": experiment,
        "target": TARGET,
        "target_definition": "ln(close[t+1 exact real minute] / close[t]); same root and contract segment",
        "feature_names": list(features),
        "feature_order": list(features),
        "training_start": training["timestamp"].min().isoformat(),
        "training_end": training["timestamp"].max().isoformat(),
        "training_end_inclusive": True,
        "historical_cutoff_exclusive": config["historical_cutoff"],
        "eligible_historical_rows": eligible_count,
        "training_sample_count": training.height,
        "selected_row_hash": logical_row_hash,
        "selected_rows_path": str(selected_path),
        "training_medians": training_medians(model, features),
        "historical_feature_profile": distribution_profile(training, features),
        "volatility_thresholds": {
            "source": "all eligible historical rows strictly before cutoff",
            "feature": "realized_vol_15m",
            "low_max": volatility_thresholds[0],
            "medium_max": volatility_thresholds[1],
        },
        "preprocessing": "historical-sample median imputation only; no root encoding; no scaling",
        "model_parameters": config["modeling"]["hist_gradient_boosting"],
        "random_seed": config["modeling"]["random_seed"],
        "created_at_utc": created_at,
        "immutable_for_prospective_confirmation": True,
    }
    save_model_artifact(model, output, metadata=metadata)
    artifact_path = output / "model_pipeline.joblib"
    metadata_path = output / "metadata.json"
    return {
        **metadata,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "metadata_path": str(metadata_path),
        "selected_rows_artifact_sha256": sha256_file(selected_path),
        "selected_rows_size_bytes": selected_path.stat().st_size,
        "library_versions": json.loads(metadata_path.read_text(encoding="utf-8"))[
            "library_versions"
        ],
    }


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    paths = resolve_data_paths(create=True)
    paths.configure_process_temp()
    config = _config()
    if tuple(config["roots"]) != PHASE2_3_ROOTS:
        raise RuntimeError("Phase 2.3 roots differ from the preselected ZN/ZC/ZB set")
    if tuple(config["experiments"]) != PHASE2_3_EXPERIMENTS:
        raise RuntimeError("Phase 2.3 experiments must be frozen A-1m and B-1m")
    cutoff = _iso(config["historical_cutoff"])
    feature_root = paths.features_root / config["source_experiment"]
    artifact_root = paths.model_artifact_root / "phase2_3"
    external_manifest = artifact_root / "model_freeze_manifest.json"
    repo_manifest = PROJECT_ROOT / "reports" / "phase2_3_model_freeze_manifest.json"
    if external_manifest.exists():
        manifest = json.loads(external_manifest.read_text(encoding="utf-8"))
        verify_frozen_artifacts(manifest)
        _write_json(manifest, repo_manifest)
        print("Phase 2.3 candidates already frozen; hashes verified", flush=True)
        return 0
    if artifact_root.exists() and any(artifact_root.rglob("*")):
        raise RuntimeError(
            "A partial Phase 2.3 artifact tree exists without a freeze manifest; "
            "manual audit is required before proceeding"
        )

    entries: list[dict[str, Any]] = []
    for root in PHASE2_3_ROOTS:
        print(f"[{root}] selecting deterministic historical sample", flush=True)
        training, eligible_count, thresholds = load_historical_training_sample(
            feature_root,
            root,
            cutoff,
            int(config["training_rows_per_root"]),
        )
        if training.height != int(config["training_rows_per_root"]):
            raise RuntimeError(f"{root} did not supply the frozen 200,000-row sample")
        if training["timestamp"].max() >= cutoff:
            raise RuntimeError(f"{root} sample reaches or crosses the historical cutoff")
        for experiment in PHASE2_3_EXPERIMENTS:
            print(f"[{root} {experiment}] fitting final frozen candidate", flush=True)
            entries.append(
                _fit_one(
                    root=root,
                    experiment=experiment,
                    training=training,
                    eligible_count=eligible_count,
                    volatility_thresholds=thresholds,
                    config=config,
                    artifact_root=artifact_root,
                )
            )

    frozen_at = datetime.now(timezone.utc)
    checkpoints = [
        {
            "days": days,
            "role": "primary" if days == config["primary_checkpoint_days"] else "monitoring",
            "start_exclusive": cutoff.isoformat(),
            "end_exclusive": (cutoff + timedelta(days=days)).isoformat(),
        }
        for days in config["checkpoints_days"]
    ]
    manifest = {
        "version": config["version"],
        "status": "frozen_awaiting_prospective_data",
        "frozen_at_utc": frozen_at.isoformat(),
        "historical_cutoff": cutoff.isoformat(),
        "prospective_rule": "timestamps must be strictly greater than historical_cutoff",
        "preselected_roots": list(PHASE2_3_ROOTS),
        "experiments": list(PHASE2_3_EXPERIMENTS),
        "model_count": len(entries),
        "models": entries,
        "predeclared_checkpoints": checkpoints,
        "primary_checkpoint_days": config["primary_checkpoint_days"],
        "selection_rule": (
            "At the completed 90-day checkpoint, retain B only for a root when it "
            "consistently and meaningfully improves A; prefer A for ties or A wins."
        ),
        "no_retraining_rule": "All six pipelines, imputers, samples, features, and parameters remain immutable during confirmation.",
        "configuration": config,
    }
    _write_json(manifest, external_manifest)
    _write_json(manifest, repo_manifest)
    verify_frozen_artifacts(manifest)
    print(f"Frozen six candidates: {external_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
