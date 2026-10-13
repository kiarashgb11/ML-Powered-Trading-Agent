"""Freeze Phase 2.2 folds and row counts before any model is fitted."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.evaluation.walk_forward import (  # noqa: E402
    build_walk_forward_folds,
    fold_evaluation_expression,
    fold_training_expression,
)
from futures_ml.features.pipeline import eligibility_expression, scan_partitioned_root  # noqa: E402


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "phase2_2.yaml").read_text(encoding="utf-8")
    )


def build_declaration(config: dict[str, Any], feature_root: Path) -> dict[str, Any]:
    """Count eligible train/evaluation rows for every predeclared fold."""
    boundaries = tuple(_iso(value) for value in config["evaluation_boundaries"])
    folds = build_walk_forward_folds(
        data_start=_iso(config["data_start"]),
        evaluation_boundaries=boundaries,
        purge_minutes=int(config["purge_minutes"]),
    )
    cap = int(config["training_rows_per_root"])
    rows: list[dict[str, Any]] = []
    for fold in folds:
        eligible_train: dict[str, int] = {}
        selected_train: dict[str, int] = {}
        eligible_evaluation: dict[str, int] = {}
        for root in config["roots"]:
            scan = scan_partitioned_root(feature_root, root).filter(
                eligibility_expression(config["primary_target"])
            )
            counts = scan.select(
                fold_training_expression(fold).sum().alias("train"),
                fold_evaluation_expression(fold).sum().alias("evaluation"),
            ).collect().row(0, named=True)
            eligible_train[root] = int(counts["train"])
            selected_train[root] = min(cap, int(counts["train"]))
            eligible_evaluation[root] = int(counts["evaluation"])
        rows.append(
            {
                **fold.to_dict(),
                "purge_interval_start": fold.training_end_exclusive.isoformat(),
                "purge_interval_end": fold.evaluation_start.isoformat(),
                "eligible_training_rows_by_root": eligible_train,
                "selected_training_rows_by_root": selected_train,
                "eligible_evaluation_rows_by_root": eligible_evaluation,
                "eligible_training_rows": sum(eligible_train.values()),
                "selected_training_rows": sum(selected_train.values()),
                "eligible_evaluation_rows": sum(eligible_evaluation.values()),
            }
        )
    return {
        "version": config["version"],
        "status": "predeclared_before_training",
        "source_feature_version": config["source_feature_version"],
        "target": config["primary_target"],
        "training_cap_per_root": cap,
        "folds": rows,
    }


def render_markdown(declaration: dict[str, Any]) -> str:
    """Render the immutable fold plan without any performance interpretation."""
    lines = [
        "# Phase 2.2 Predeclared Walk-Forward Folds",
        "",
        "**Status: frozen before model fitting. No performance result was available when these boundaries and counts were saved.**",
        "",
        "Training expands from 2021-09-23. Each evaluation period is approximately six months and begins only after a 75-minute two-sided train/evaluation purge. Evaluation periods do not overlap. The 200,000-row/root cap uses deterministic evenly spaced observations; an early root below the cap contributes all available eligible training rows.",
        "",
        "| Fold | Train start | Train end (exclusive) | Purge interval | Evaluation start | Evaluation end (exclusive) | Eligible train | Selected train | Evaluation rows |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for fold in declaration["folds"]:
        lines.append(
            f"| {fold['fold_id']} | {fold['training_start']} | "
            f"{fold['training_end_exclusive']} | "
            f"[{fold['purge_interval_start']}, {fold['purge_interval_end']}) | "
            f"{fold['evaluation_start']} | {fold['evaluation_end_exclusive']} | "
            f"{fold['eligible_training_rows']:,} | {fold['selected_training_rows']:,} | "
            f"{fold['eligible_evaluation_rows']:,} |"
        )
    lines += ["", "## Per-root counts", ""]
    for fold in declaration["folds"]:
        lines += [
            f"### {fold['fold_id']}",
            "",
            "| Root | Eligible training | Selected training | Eligible evaluation |",
            "|---|---:|---:|---:|",
        ]
        for root, eligible in fold["eligible_training_rows_by_root"].items():
            lines.append(
                f"| {root} | {eligible:,} | "
                f"{fold['selected_training_rows_by_root'][root]:,} | "
                f"{fold['eligible_evaluation_rows_by_root'][root]:,} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config = _config()
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows, create=True)
    feature_root = paths.features_root / config["source_experiment"]
    declaration = build_declaration(config, feature_root)
    repo_json = PROJECT_ROOT / "reports" / "phase2_2_fold_definitions.json"
    repo_md = PROJECT_ROOT / "reports" / "phase2_2_fold_definitions.md"
    external = paths.model_artifact_root / "phase2_2" / "fold_definitions.json"
    serialized = json.dumps(declaration, indent=2, allow_nan=False)
    for path in (repo_json, external):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != declaration:
            raise RuntimeError(f"Refusing to change an existing fold declaration: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    repo_md.write_text(render_markdown(declaration), encoding="utf-8")
    print(f"Predeclared {len(declaration['folds'])} folds before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
