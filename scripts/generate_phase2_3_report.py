"""Render the Phase 2.3 freeze manifest and prospective status report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_JSON = PROJECT_ROOT / "reports" / "phase2_3_model_freeze_manifest.json"
METRICS_JSON = PROJECT_ROOT / "reports" / "phase2_3_prospective_metrics.json"
MANIFEST_MD = PROJECT_ROOT / "reports" / "phase2_3_model_freeze_manifest.md"
REPORT_MD = PROJECT_ROOT / "reports" / "phase2_3_dedicated_prospective_report.md"


def _short_hash(value: str) -> str:
    return f"`{value[:16]}...`"


def _number(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Phase 2.3 Model Freeze Manifest",
        "",
        f"**Status: {manifest['status'].upper()}**",
        "",
        f"Frozen at UTC: `{manifest['frozen_at_utc']}`",
        "",
        f"Historical cutoff: `{manifest['historical_cutoff']}`",
        "",
        f"Prospective rule: {manifest['prospective_rule']}",
        "",
        "The six candidates below were trained and hashed before any post-cutoff performance was examined. Phase 2, 2.1, and 2.2 artifacts were not overwritten.",
        "",
        "| Model | Root | Experiment | Features | Training span | Rows | Selected-row hash | Pipeline hash |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for model in manifest["models"]:
        lines.append(
            f"| {model['model_id']} | {model['root']} | {model['experiment']} | "
            f"{len(model['feature_order'])} | `{model['training_start']}` through "
            f"`{model['training_end']}` | {model['training_sample_count']:,} | "
            f"{_short_hash(model['selected_row_hash'])} | "
            f"{_short_hash(model['artifact_sha256'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen model contract",
            "",
            "- Target: exact next-real-minute, same-root, same-contract-segment `future_return_1m`.",
            "- A: the unchanged 35 own-market features.",
            "- B: A plus the unchanged 18 raw exact-UTC anchor returns.",
            "- Dedicated HGB: learning rate 0.08, 150 iterations, 31 leaves, minimum leaf 100, L2 0.1, seed 42.",
            "- Training: 200,000 deterministic evenly spaced eligible observations per target root, all strictly before the cutoff.",
            "- Preprocessing: per-model historical median imputation only. Imputer medians and full historical robust feature profiles are stored in model metadata and this JSON manifest.",
            "- Frozen volatility thresholds: per-root tertiles from all eligible historical rows strictly before the cutoff.",
            "- Immutability: pipelines, preprocessing, features, samples, and parameters cannot change during prospective confirmation.",
            "",
            "## Predeclared checkpoints",
            "",
            "| Days | Role | Start exclusive | End exclusive |",
            "|---:|---|---|---|",
        ]
    )
    for checkpoint in manifest["predeclared_checkpoints"]:
        lines.append(
            f"| {checkpoint['days']} | {checkpoint['role']} | "
            f"`{checkpoint['start_exclusive']}` | `{checkpoint['end_exclusive']}` |"
        )
    lines.extend(
        [
            "",
            "The 7/30/60-day checkpoints are monitoring diagnostics. Only the predeclared 90-day checkpoint is primary. No winner is selected automatically.",
            "",
            "## External artifacts",
            "",
        ]
    )
    for model in manifest["models"]:
        lines.append(f"- `{model['model_id']}`: `{Path(model['artifact_path']).parent}`")
    return "\n".join(lines) + "\n"


def _results_sections(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for window in metrics["completed_windows"]:
        lines.extend(
            [
                f"## Completed {window['days']}-day checkpoint",
                "",
                f"Role: **{window['role']}**. Window: `{window['start_exclusive']}` to `{window['end_exclusive']}`.",
                "",
            ]
        )
        for root in ("ZN", "ZC", "ZB"):
            lines.extend(
                [
                    f"### {root} A vs B",
                    "",
                    "| Model | N | MAE | RMSE | R² | Pearson | Spearman | Direction diagnostic |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for experiment in ("A", "B"):
                model_id = f"{root}-{experiment}-final"
                row = window["per_model_metrics"][model_id]["overall"]
                lines.append(
                    f"| {model_id} | {_number(row['observations'])} | {_number(row['mae'])} | "
                    f"{_number(row['rmse'])} | {_number(row['r2'])} | {_number(row['pearson'])} | "
                    f"{_number(row['spearman'])} | {_number(row['direction_accuracy'])} |"
                )
            delta = window["a_vs_b_deltas"][root]
            lines.extend(
                [
                    "",
                    f"B-A: Pearson `{_number(delta['b_minus_a_pearson'])}`, Spearman "
                    f"`{_number(delta['b_minus_a_spearman'])}`, R² `{_number(delta['b_minus_a_r2'])}`.",
                    "",
                ]
            )
        lines.extend(
            [
                "### Prediction deciles",
                "",
                "Per-model deciles and top-minus-bottom/order statistics are stored in the machine-readable metrics bundle.",
                "",
                "### Volatility regimes",
                "",
                "Low/medium/high diagnostics use only the thresholds frozen from historical training data.",
                "",
                "### Drift diagnostics",
                "",
                "Median, IQR, missing rate, and median shift in historical-IQR units are stored for every model feature. Drift does not trigger retraining.",
                "",
            ]
        )
    return lines


def _report_markdown(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    current = metrics["current_data"]
    lines = [
        "# Phase 2.3 Dedicated Prospective Confirmation",
        "",
        f"**Status: {metrics['status'].upper().replace('_', ' ')}**",
        "",
        "## Objective",
        "",
        "Determine separately for ZN, ZC, and ZB whether the dedicated one-minute signal survives genuinely new data and whether B's 18 raw anchor returns add enough repeatable value to justify their complexity. This is frozen-model confirmation, not model improvement or strategy research.",
        "",
        "## Why ZN/ZC/ZB were preselected",
        "",
        "Phase 2.2 preselected these roots before Phase 2.3: dedicated B beat pooled B on each root in all 8/8 historical folds. No root was added after inspecting prospective performance.",
        "",
        "## Frozen A definition",
        "",
        "Each dedicated A model uses exactly the original 35 own-market numeric features.",
        "",
        "## Frozen B definition",
        "",
        "Each dedicated B model uses the same 35 features plus 18 raw exact-time NQ/ES/ZN/CL/GC/6E returns at 1/5/15 minutes. Relative C features, new anchors, spreads, horizons, and indicators are absent.",
        "",
        "## Historical cutoff",
        "",
        f"The frozen cutoff is `{manifest['historical_cutoff']}`. Training rows are strictly earlier; prospective rows must be strictly later. Old Phase 2.2 folds are never relabeled as prospective evidence.",
        "",
        "## Six frozen model IDs",
        "",
        ", ".join(f"`{entry['model_id']}`" for entry in manifest["models"]),
        "",
        "## Training counts",
        "",
        "| Model | Selected rows | Eligible historical rows | Last selected timestamp |",
        "|---|---:|---:|---|",
    ]
    for model in manifest["models"]:
        lines.append(
            f"| {model['model_id']} | {model['training_sample_count']:,} | "
            f"{model['eligible_historical_rows']:,} | `{model['training_end']}` |"
        )
    lines.extend(
        [
            "",
            "## Freeze hashes",
            "",
            "| Model | Logical selected-row hash | Pipeline SHA-256 |",
            "|---|---|---|",
        ]
    )
    for model in manifest["models"]:
        lines.append(
            f"| {model['model_id']} | `{model['selected_row_hash']}` | `{model['artifact_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Prospective evaluation protocol",
            "",
            "Predictions are generated from metadata and causal feature columns only. Outcomes are read and attached in a separate exact-identity step after prediction events exist. The external log is append-only; existing parts are never rewritten. Pipelines and preprocessing hashes are verified before every evaluation run, and no fit operation occurs in the evaluator.",
            "",
            "A and B are compared independently within each root using N, MAE, RMSE, R², Pearson, Spearman, and diagnostic direction agreement. B-A Pearson, Spearman, and R² are reported. Ties favor simpler A only at the primary checkpoint and only after research review.",
            "",
            "## Predeclared checkpoints",
            "",
            "| Days | Role | End exclusive |",
            "|---:|---|---|",
        ]
    )
    for checkpoint in manifest["predeclared_checkpoints"]:
        lines.append(
            f"| {checkpoint['days']} | {checkpoint['role']} | `{checkpoint['end_exclusive']}` |"
        )
    lines.extend(
        [
            "",
            "The 7/30/60-day windows are monitoring only. The 90-day window is the primary checkpoint; short-window results cannot choose a winner.",
            "",
            "## Leakage safeguards",
            "",
            "- Exact t+1-minute, same-contract-segment target; no next-row substitution or rollover crossing.",
            "- Exact-UTC anchor joins without as-of matching or forward filling.",
            "- Historical timestamps are strictly before the cutoff; prospective timestamps are strictly after it.",
            "- Prediction inputs exclude every future-return label.",
            "- Outcome attachment is a separate exact timestamp/root/instrument/segment join.",
            "- Frozen model and selected-row hashes are checked before evaluation.",
            "- No retraining, imputer update, feature update, tuning, or automatic drift response is permitted.",
            "",
            "## Current amount of new data available",
            "",
            f"Post-cutoff source rows: **{current['total_source_rows']:,}**. By root: "
            + ", ".join(f"{root}={count:,}" for root, count in current["rows_by_root"].items())
            + ".",
            "",
        ]
    )
    if metrics["status"] == "awaiting_prospective_data":
        lines.extend(
            [
                "## AWAITING PROSPECTIVE DATA",
                "",
                "Models frozen; prospective evaluator ready; awaiting genuinely new data.",
                "",
                "No historical observation was reused, no metric was invented, and no paid data was purchased or downloaded.",
                "",
            ]
        )
    else:
        lines.extend(_results_sections(metrics))
    lines.extend(
        [
            "## Interpretation",
            "",
            "No A/B selection is currently authorized. The models remain immutable until a completed predeclared checkpoint is reviewed; the primary decision belongs to the 90-day checkpoint.",
            "",
            "## Artifact locations",
            "",
            f"- Freeze manifest: `{metrics['freeze_manifest_reference']}`",
            f"- Model tree: `{Path(metrics['freeze_manifest_reference']).parent}`",
            f"- Append-only prediction log: `{current['log_path']}`",
            "- Repository metrics: `reports/phase2_3_prospective_metrics.json`",
            "",
            "## Next checkpoint",
            "",
            "Acquire genuinely post-cutoff data only after explicit approval, build it through the unchanged causal feature pipeline, and rerun the prospective evaluator without retraining. Stop before strategy, P&L, cost, risk, or execution work.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    MANIFEST_MD.write_text(_manifest_markdown(manifest), encoding="utf-8")
    REPORT_MD.write_text(_report_markdown(manifest, metrics), encoding="utf-8")
    print(f"Wrote {MANIFEST_MD} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
