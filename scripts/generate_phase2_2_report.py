"""Generate the Phase 2.2 walk-forward report and four focused plots."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from futures_ml.config.paths import resolve_data_paths  # noqa: E402

MPL_CACHE = resolve_data_paths().temp_root / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "reports" / "phase2_2_walk_forward_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase2_2_walk_forward_stability_report.md"
FIGURES = PROJECT_ROOT / "reports" / "figures"
EXPERIMENTS = ("A_1m", "B_1m")
IMPORTANT_ROOTS = ("ZN", "ZC", "ZB", "ZF")


def _number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def _percent(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.3f}%"


def _metric_row(label: str, metric: dict[str, Any]) -> str:
    return (
        f"| {label} | {metric['observations']:,} | {_number(metric['mae'])} | "
        f"{_number(metric['rmse'])} | {_number(metric['r2'])} | "
        f"{_number(metric['pearson'])} | {_number(metric['spearman'])} | "
        f"{_percent(metric['direction_accuracy'])} |"
    )


def _plots(results: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    folds = results["fold_results"]
    labels = [fold["fold_id"].replace("fold_", "F") for fold in folds]
    for metric, title, filename in (
        ("pearson", "Pooled Pearson by walk-forward fold", "phase2_2_pearson_by_fold.png"),
        ("spearman", "Pooled Spearman by walk-forward fold", "phase2_2_spearman_by_fold.png"),
    ):
        plt.figure(figsize=(8, 4.5))
        for experiment, color in (("A_1m", "#2563eb"), ("B_1m", "#dc2626")):
            values = [
                fold["experiments"][experiment]["overall"][metric] for fold in folds
            ]
            plt.plot(labels, values, marker="o", label=experiment.replace("_", "-"), color=color)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Predeclared chronological fold")
        plt.ylabel(metric.title())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / filename, dpi=150)
        plt.close()

    root_stability = results["aggregate"]["root_stability"]["B_1m"]
    roots = list(results["configuration"]["roots"])
    means = [root_stability[root]["pearson"]["mean"] for root in roots]
    stds = [root_stability[root]["pearson"]["standard_deviation"] for root in roots]
    order = np.argsort(means)[::-1]
    ordered_roots = [roots[index] for index in order]
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        ordered_roots,
        [means[index] for index in order],
        yerr=[stds[index] for index in order],
        fmt="o",
        capsize=3,
        color="#2563eb",
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Root (ordered by B-1m mean Pearson)")
    plt.ylabel("Mean Pearson ± fold SD")
    plt.title("Per-root B-1m temporal stability")
    plt.tight_layout()
    plt.savefig(FIGURES / "phase2_2_per_root_pearson.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for root in IMPORTANT_ROOTS:
        values = [
            fold["experiments"]["B_1m"]["per_root"][root]["pearson"] for fold in folds
        ]
        plt.plot(labels, values, marker="o", label=root)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Predeclared chronological fold")
    plt.ylabel("Pearson")
    plt.title("Predeclared strong roots across folds (B-1m)")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(FIGURES / "phase2_2_strong_roots_by_fold.png", dpi=150)
    plt.close()


def _fold_definition_section(results: dict[str, Any]) -> list[str]:
    declaration = results["fold_declaration"]
    lines = [
        "| Fold | Training start | Training end exclusive | Purge interval | Evaluation start | Evaluation end exclusive | Selected train | Evaluation N |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for fold in declaration["folds"]:
        lines.append(
            f"| {fold['fold_id']} | {fold['training_start']} | "
            f"{fold['training_end_exclusive']} | "
            f"[{fold['purge_interval_start']}, {fold['purge_interval_end']}) | "
            f"{fold['evaluation_start']} | {fold['evaluation_end_exclusive']} | "
            f"{fold['selected_training_rows']:,} | {fold['eligible_evaluation_rows']:,} |"
        )
    lines += [
        "",
        "The declaration below was saved before training; performance did not alter a boundary or count. Fold 1 uses all 187,416 available ZW rows; every other root/fold reaches 200,000.",
        "",
    ]
    for fold in declaration["folds"]:
        lines += [
            f"### {fold['fold_id']} per-root declaration",
            "",
            "| Root | Eligible train | Selected train | Evaluation N |",
            "|---|---:|---:|---:|",
        ]
        for root in results["configuration"]["roots"]:
            lines.append(
                f"| {root} | {fold['eligible_training_rows_by_root'][root]:,} | "
                f"{fold['selected_training_rows_by_root'][root]:,} | "
                f"{fold['eligible_evaluation_rows_by_root'][root]:,} |"
            )
        lines.append("")
    return lines


def _fold_results(results: dict[str, Any], experiment: str) -> list[str]:
    lines = [
        "| Fold | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in results["fold_results"]:
        metric = fold["experiments"][experiment]["overall"]
        lines.append(_metric_row(fold["fold_id"], metric))
    return lines


def _stability_summary(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Folds | Positive Pearson folds | Mean Pearson | Median Pearson | Pearson SD | Worst Pearson | Mean Spearman | Mean R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        stability = results["aggregate"]["model_stability"][experiment]
        lines.append(
            f"| {experiment.replace('_', '-')} HGB | 8 | "
            f"{stability['pearson']['positive_folds']} | "
            f"{_number(stability['pearson']['mean'])} | "
            f"{_number(stability['pearson']['median'])} | "
            f"{_number(stability['pearson']['standard_deviation'])} | "
            f"{_number(stability['pearson']['minimum'])} | "
            f"{_number(stability['spearman']['mean'])} | "
            f"{_number(stability['r2']['mean'])} |"
        )
    return lines


def _root_stability_table(results: dict[str, Any], experiment: str) -> list[str]:
    rows = results["aggregate"]["root_stability"][experiment]
    lines = [
        "| Root | Mean Pearson | Median Pearson | Pearson SD | Worst | Best | Positive folds | Mean Spearman | Positive Spearman folds | Avg rank | Rank range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(rows, key=lambda root: rows[root]["average_rank"])
    for root in ordered:
        row = rows[root]
        lines.append(
            f"| {root} | {_number(row['pearson']['mean'])} | "
            f"{_number(row['pearson']['median'])} | "
            f"{_number(row['pearson']['standard_deviation'])} | "
            f"{_number(row['pearson']['minimum'])} | {_number(row['pearson']['maximum'])} | "
            f"{row['pearson']['positive_folds']}/8 | {_number(row['spearman']['mean'])} | "
            f"{row['spearman']['positive_folds']}/8 | {row['average_rank']:.2f} | "
            f"{row['best_rank']}-{row['worst_rank']} |"
        )
    return lines


def _regime_tables(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for experiment in EXPERIMENTS:
        lines += [
            f"### {experiment.replace('_', '-')} fold/regime metrics",
            "",
            "| Fold | Regime | N | R² | Pearson | Spearman |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for fold in results["fold_results"]:
            for row in fold["experiments"][experiment]["volatility_regimes"]:
                lines.append(
                    f"| {fold['fold_id']} | {row['group']} | {row['observations']:,} | "
                    f"{_number(row['r2'])} | {_number(row['pearson'])} | "
                    f"{_number(row['spearman'])} |"
                )
        lines += [
            "",
            "| Regime | Mean Pearson | Pearson SD | Positive Pearson folds | Mean Spearman | Mean R² |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        aggregate = results["aggregate"]["volatility_regime_stability"][experiment]
        for regime in ("low", "medium", "high", "unavailable"):
            row = aggregate[regime]
            lines.append(
                f"| {regime} | {_number(row['pearson']['mean'])} | "
                f"{_number(row['pearson']['standard_deviation'])} | "
                f"{row['pearson']['positive_folds']}/8 | "
                f"{_number(row['spearman']['mean'])} | {_number(row['r2']['mean'])} |"
            )
        lines.append("")
    return lines


def _time_tables(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    targeted = {"11", "12", "21", "22"}
    for experiment in EXPERIMENTS:
        lines += [
            f"### {experiment.replace('_', '-')} prementioned-hour replication",
            "",
            "| Fold | Chicago hour | N | R² | Pearson | Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for fold in results["fold_results"]:
            for row in fold["experiments"][experiment]["chicago_hours"]:
                if str(row["group"]) in targeted:
                    lines.append(
                        f"| {fold['fold_id']} | {row['group']} | {row['observations']:,} | "
                        f"{_number(row['r2'])} | {_number(row['pearson'])} | "
                        f"{_number(row['spearman'])} |"
                    )
        lines += [
            "",
            "| Hour | Mean Pearson | Pearson SD | Positive folds | Mean Spearman | Mean R² |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        aggregate = results["aggregate"]["time_of_day_stability"][experiment]
        for hour in sorted(aggregate, key=int):
            row = aggregate[hour]
            lines.append(
                f"| {hour} | {_number(row['pearson']['mean'])} | "
                f"{_number(row['pearson']['standard_deviation'])} | "
                f"{row['pearson']['positive_folds']}/8 | "
                f"{_number(row['spearman']['mean'])} | {_number(row['r2']['mean'])} |"
            )
        lines.append("")
    return lines


def _decile_tables(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for experiment in EXPERIMENTS:
        lines += [
            f"### {experiment.replace('_', '-')} evaluation deciles",
            "",
            "| Fold | Decile | N | Mean prediction | Mean actual | Median actual |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for fold in results["fold_results"]:
            for row in fold["experiments"][experiment]["prediction_deciles"]:
                lines.append(
                    f"| {fold['fold_id']} | {row['decile']} | {row['observations']:,} | "
                    f"{_number(row['mean_prediction'])} | {_number(row['mean_actual'])} | "
                    f"{_number(row['median_actual'])} |"
                )
        aggregate = results["aggregate"]["prediction_decile_stability"][experiment]
        lines += [
            "",
            f"Realized-decile ordering Spearman averaged `{_number(aggregate['ordering_spearman']['mean'])}` and was positive in {aggregate['ordering_spearman']['positive_folds']}/8 folds. The top-minus-bottom realized return was positive in {aggregate['positive_top_minus_bottom_folds']}/8 folds. All ten realized means were non-decreasing in {aggregate['fully_non_decreasing_folds']}/8 folds.",
            "",
        ]
    return lines


def _dedicated_tables(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Fold | Root | Pooled B Pearson | Dedicated B Pearson | Dedicated minus pooled |",
        "|---|---|---:|---:|---:|",
    ]
    for fold in results["fold_results"]:
        for root, row in fold["dedicated_models"].items():
            pooled = row["pooled_b_metrics"]["pearson"]
            dedicated = row["dedicated_b_metrics"]["pearson"]
            lines.append(
                f"| {fold['fold_id']} | {root} | {_number(pooled)} | "
                f"{_number(dedicated)} | {_number(dedicated - pooled)} |"
            )
    lines += [
        "",
        "| Root | Pooled mean Pearson | Dedicated mean Pearson | Mean delta | Delta SD | Dedicated wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for root, row in results["aggregate"]["dedicated_model_stability"].items():
        delta = row["pearson_delta_dedicated_minus_pooled"]
        lines.append(
            f"| {root} | {_number(row['pooled_pearson']['mean'])} | "
            f"{_number(row['dedicated_pearson']['mean'])} | {_number(delta['mean'])} | "
            f"{_number(delta['standard_deviation'])} | "
            f"{row['dedicated_beats_pooled_folds']}/8 |"
        )
    return lines


def build_report(results: dict[str, Any]) -> str:
    comparison = results["aggregate"]["a_vs_b"]
    a_stability = results["aggregate"]["model_stability"]["A_1m"]
    b_stability = results["aggregate"]["model_stability"]["B_1m"]
    b_roots = results["aggregate"]["root_stability"]["B_1m"]
    lines = [
        "# Phase 2.2 Rolling Walk-Forward Stability Study",
        "",
        "**Status: COMPLETE. Eight folds were predeclared before fitting. Phase 2 and Phase 2.1 data, models, metrics, and reports remain unchanged.**",
        "",
        "## Objective",
        "",
        "Determine whether the small one-minute predictive relationship found in Phase 2.1 remains positive and reasonably stable through different chronological market periods. This is historical temporal replication, not model invention, feature selection, strategy research, or a new pristine final holdout.",
        "",
        "## Frozen experimental design",
        "",
        "- A-1m: the original 35 own-market features.",
        "- B-1m: the same 35 features plus the frozen 18 exact-UTC NQ/ES/ZN/CL/GC/6E returns at 1/5/15 minutes.",
        "- Target: exact-time, same-segment `future_return_1m`.",
        "- Model: HGB with `learning_rate=0.08`, `max_iter=150`, `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=0.1`, seed 42.",
        "- Training cap: 200,000 deterministic evenly spaced observations/root; all available rows when below the cap.",
        "- Every fold refits median imputation and root one-hot encoding on that fold's training sample only.",
        "- No hyperparameter search, feature selection, relative C features, new indicators, new anchors, or target changes.",
        "",
        "## Fold definitions",
        "",
        *_fold_definition_section(results),
        "## Leakage safeguards",
        "",
        "Each fold trains strictly before its evaluation window. A 150-minute unused interval implements 75 minutes on each side of the boundary. Exact next-minute labels and anchor joins remain same-timestamp/same-segment; no next-available row, as-of join, forward fill, or rollover crossing is allowed. Label columns are rejected from `X`. The runner asserts actual train and evaluation counts against the predeclared files and saves a separate preprocessor/model per fold.",
        "",
        "Phase 2.1 already examined later portions of this history. These are replication windows over known historical data, not eight new untouched tests.",
        "",
        "## A-1m walk-forward results",
        "",
        *_fold_results(results, "A_1m"),
        "",
        "![A/B Pearson by fold](figures/phase2_2_pearson_by_fold.png)",
        "",
        "## B-1m walk-forward results",
        "",
        *_fold_results(results, "B_1m"),
        "",
        "![A/B Spearman by fold](figures/phase2_2_spearman_by_fold.png)",
        "",
        "## A vs B fold-by-fold comparison",
        "",
        "| Fold | B-A Pearson | B-A Spearman | B-A R² | Pearson winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison["folds"]:
        delta = row["pearson_delta_b_minus_a"]
        lines.append(
            f"| {row['fold_id']} | {_number(delta)} | "
            f"{_number(row['spearman_delta_b_minus_a'])} | "
            f"{_number(row['r2_delta_b_minus_a'])} | {'B' if delta > 0 else 'A'} |"
        )
    lines += [
        "",
        f"B beat A on Pearson in {comparison['b_beats_a_pearson_folds']}/8 folds, Spearman in {comparison['b_beats_a_spearman_folds']}/8, and R² in {comparison['b_beats_a_r2_folds']}/8. Mean B-A Pearson was `{_number(comparison['pearson']['mean'])}`. The small Phase 2.1 B advantage does not repeat consistently; the raw cross-market increment is best treated as noise at this resolution.",
        "",
        "## Per-market stability",
        "",
        "### A-1m root stability",
        "",
        *_root_stability_table(results, "A_1m"),
        "",
        "### B-1m root stability",
        "",
        *_root_stability_table(results, "B_1m"),
        "",
        "![Per-root B stability](figures/phase2_2_per_root_pearson.png)",
        "",
        "ZN, ZC, ZB, and ZF were positive in every fold and occupied the top four average B ranks. NQ remained weakest: mean Pearson was "
        f"`{_number(b_roots['NQ']['pearson']['mean'])}`, with {b_roots['NQ']['pearson']['positive_folds']}/8 positive folds and average rank `{b_roots['NQ']['average_rank']:.2f}`.",
        "",
        "## Root rank stability",
        "",
        "The B-1m average ranks were ZN 1.12, ZC 2.12, ZB 2.75, and ZF exactly 4.00. ZN ranked first or second in every fold; ZF ranked fourth in all eight. Equities remained weak: ES averaged rank 12.12, RTY 14.38, and NQ 15.38.",
        "",
        "![Predeclared roots by fold](figures/phase2_2_strong_roots_by_fold.png)",
        "",
        "## Volatility-regime stability",
        "",
        "Regime cutoffs are root-specific tertiles learned from all eligible training rows within each fold, never from evaluation. The diagnostic was repeated without changing model behavior.",
        "",
        *_regime_tables(results),
        "Low-volatility strength repeated: both models had positive low-regime Pearson in 8/8 folds, with mean Pearson about 0.126 versus about 0.047 in high volatility. This is replicated descriptive structure, not a trading filter.",
        "",
        "## Time-of-day stability",
        "",
        "The prementioned 11-12 and 21-22 Chicago hours are shown fold by fold, followed by all-hour aggregates. No hour was selected or removed after seeing results.",
        "",
        *_time_tables(results),
        "The strongest aggregate region shifted toward 21-23 Chicago time. Hour 22 was strongest for both models and positive in 8/8 folds. The earlier 11-12 observation remained positive but was not the dominant stable region.",
        "",
        "## Prediction-decile stability",
        "",
        *_decile_tables(results),
        "Both models had a positive top-minus-bottom realized-return spread and strongly positive decile-order correlation in every fold. Perfect pointwise monotonicity was stricter: 3/8 folds for A and 4/8 for B.",
        "",
        "## Optional ZN/ZC/ZB dedicated-model replication",
        "",
        "This secondary analysis used only the three roots predeclared from Phase 2.1 and the same frozen B features/hyperparameters.",
        "",
        *_dedicated_tables(results),
        "",
        "Dedicated B models beat the pooled B prediction on their root in all 8/8 folds: mean Pearson improvements were about 0.0183 for ZN, 0.0091 for ZC, and 0.0435 for ZB. This supports market-specific modeling as a research direction, not a strategy or market-selection rule.",
        "",
        "## Aggregate stability statistics",
        "",
        *_stability_summary(results),
        "",
        "A and B were positive on Pearson, Spearman, and R² in all eight folds. Fold dispersion was modest relative to the mean: A Pearson mean/SD was "
        f"`{_number(a_stability['pearson']['mean'])}` / `{_number(a_stability['pearson']['standard_deviation'])}`; B was `{_number(b_stability['pearson']['mean'])}` / `{_number(b_stability['pearson']['standard_deviation'])}`.",
        "",
        "## Scientific interpretation",
        "",
        "### Explicit answers",
        "",
        "**Q1. Does A-1m maintain positive predictive signal across most folds?** Yes. Pearson, Spearman, and R² were positive in 8/8 folds. Mean Pearson was 0.0751 and the worst fold remained positive at 0.0561.",
        "",
        "**Q2. Does B-1m maintain positive predictive signal across most folds?** Yes. All three primary stability metrics were positive in 8/8 folds. Mean Pearson was 0.0750 and the worst fold was 0.0549.",
        "",
        "**Q3. Does B consistently outperform A?** No. B won only 3/8 Pearson folds, 2/8 Spearman folds, and 3/8 R² folds; its mean deltas were slightly negative. Phase 2.1's tiny B advantage was mostly noise.",
        "",
        "**Q4. Are ZN, ZC, ZB, and ZF consistently stronger?** Yes. They were the top four roots by average B rank and all had positive Pearson in 8/8 folds.",
        "",
        "**Q5. Does ZN retain its unusually strong correlation?** Yes. ZN Pearson was positive in 8/8 folds, averaged 0.274, ranged from 0.155 to 0.367, and ranked first or second every time. Its dedicated model also beat pooled B in 8/8 folds.",
        "",
        "**Q6. Is low-volatility predictive strength repeatable?** Yes descriptively: low-volatility Pearson was positive in every fold and materially exceeded the medium/high regimes for both A and B. This still needs prospective confirmation before use as a filter.",
        "",
        "**Q7. Are prediction deciles monotonically ordered across most folds?** Broadly ordered, but not perfectly monotonic. Decile-order Spearman was strongly positive and the top-minus-bottom spread was positive in 8/8 folds; every adjacent realized mean was non-decreasing in only 3 A folds and 4 B folds.",
        "",
        "**Q8. How much does performance vary through time?** Pooled Pearson SD was about 0.010 for both models, with A spanning 0.056-0.091 and B 0.055-0.090. Root variation was larger, but the four leading roots retained their ordering unusually well.",
        "",
        "**Q9. Is the signal stable enough for the next stage?** It is stable enough to justify a narrowly scoped confirmatory market-specific experiment. It is not evidence of profitability and does not justify strategy/backtest/live work yet.",
        "",
        "**Q10. What should the next experiment be?** Predeclare a dedicated A-vs-B replication for ZN/ZC/ZB on genuinely new forward data (or a newly reserved future period). This isolates whether specialization alone explains the gain and whether raw anchors add anything within dedicated models, without reusing these folds for selection.",
        "",
        "## Limitations",
        "",
        "- All folds reuse the historical dataset already partly inspected in Phase 2.1; they are temporal replications, not pristine new holdouts.",
        "- Overlapping expanding training histories make fold estimates dependent even though evaluation windows do not overlap.",
        "- No confidence intervals, multiple-testing correction, costs, spreads, slippage, latency, capacity, or execution assumptions are included.",
        "- Pooled results are heavily influenced by stronger rates/agriculture roots; NQ/ES remain much weaker.",
        "- Regime, hour, root-rank, and dedicated-model findings are diagnostics, not trading rules.",
        "- Predictive correlation and ordered deciles do not establish tradable profitability.",
        "",
        "## Recommended next experiment",
        "",
        "Freeze dedicated A-1m and B-1m HGB models for ZN, ZC, and ZB and evaluate them prospectively on newly arriving data or a genuinely uninspected future holdout. Keep the same exact-time target, features, hyperparameters, training cap, roll safety, and purge policy. The primary question should be whether dedicated A retains the gain and whether B adds a repeatable increment over A within each market. Do not begin strategy or P&L work before that confirmation.",
        "",
        "## Artifacts",
        "",
        "- Frozen declaration: `reports/phase2_2_fold_definitions.{md,json}` and `D:\\futures-ml-data\\artifacts\\models\\phase2_2\\fold_definitions.json`.",
        "- Machine-readable results: `reports/phase2_2_walk_forward_metrics.json`.",
        "- External results/models: `D:\\futures-ml-data\\artifacts\\models\\phase2_2`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if results.get("status") != "complete":
        raise RuntimeError("Phase 2.2 result bundle is incomplete")
    _plots(results)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    print(f"Wrote {REPORT_PATH} and four Phase 2.2 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
