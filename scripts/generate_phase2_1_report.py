"""Render the completed Phase 2.1 result bundle as a research report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "reports" / "phase2_1_metrics.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "phase2_1_one_minute_report.md"
EXPERIMENTS = ("A_1m", "B_1m", "C_1m")
MODELS = ("ridge", "hist_gradient_boosting")


def _number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def _percent(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.3f}%"


def _metric_row(label: str, values: dict[str, Any]) -> str:
    return (
        f"| {label} | {values['observations']:,} | {_number(values['mae'])} | "
        f"{_number(values['rmse'])} | {_number(values['r2'])} | "
        f"{_number(values['pearson'])} | {_number(values['spearman'])} | "
        f"{_percent(values['direction_accuracy'])} |"
    )


def _metrics_table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| Configuration | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_metric_row(label, values) for label, values in rows)
    return lines


def _primary_table(results: dict[str, Any], split: str) -> list[str]:
    lines = [
        "| Experiment | Target | Input set | Ridge R² | Ridge Pearson | HGB R² | HGB Pearson | HGB Spearman |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    descriptions = {
        "A_1m": "35 own-market features",
        "B_1m": "35 own + 18 raw anchor returns",
        "C_1m": "35 own + 18 target-relative returns",
    }
    for experiment in EXPERIMENTS:
        models = results["experiments"][experiment]["pooled"]["models"]
        ridge = models["ridge"][split]["overall"]
        hgb = models["hist_gradient_boosting"][split]["overall"]
        lines.append(
            f"| {experiment.replace('_', '-')} | `future_return_1m` | {descriptions[experiment]} | "
            f"{_number(ridge['r2'])} | {_number(ridge['pearson'])} | "
            f"{_number(hgb['r2'])} | {_number(hgb['pearson'])} | "
            f"{_number(hgb['spearman'])} |"
        )
    return lines


def _macro_table(results: dict[str, Any], split: str) -> list[str]:
    lines = [
        "| Experiment | Model | Macro MAE | Macro RMSE | Macro R² | Macro Pearson | Macro Spearman | Macro direction accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        for model_name in MODELS:
            values = results["experiments"][experiment]["pooled"]["models"][model_name][
                split
            ]["macro_average"]
            lines.append(
                f"| {experiment.replace('_', '-')} | "
                f"{'Ridge' if model_name == 'ridge' else 'HGB'} | "
                f"{_number(values['mae'])} | {_number(values['rmse'])} | "
                f"{_number(values['r2'])} | {_number(values['pearson'])} | "
                f"{_number(values['spearman'])} | {_percent(values['direction_accuracy'])} |"
            )
    return lines


def _per_market_tables(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for experiment in EXPERIMENTS:
        lines += [
            f"### {experiment.replace('_', '-')} final primary models on test",
            "",
            "| Root | Model | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        models = results["experiments"][experiment]["pooled"]["models"]
        for root in results["configuration"]["roots"]:
            for model_name in MODELS:
                metric = models[model_name]["test"]["per_root"][root]
                label = "Ridge" if model_name == "ridge" else "HGB"
                lines.append(
                    f"| {root} | {label} | {metric['observations']:,} | "
                    f"{_number(metric['mae'])} | {_number(metric['rmse'])} | "
                    f"{_number(metric['r2'])} | {_number(metric['pearson'])} | "
                    f"{_number(metric['spearman'])} | {_percent(metric['direction_accuracy'])} |"
                )
        lines.append("")
    return lines


def _importance_table(rows: list[dict[str, Any]], kind: str) -> list[str]:
    if kind == "ridge":
        output = [
            "| Rank | Feature | Standardized coefficient | Sign | Magnitude |",
            "|---:|---|---:|---|---:|",
        ]
        for rank, row in enumerate(rows, start=1):
            output.append(
                f"| {rank} | `{row['feature']}` | {_number(row['coefficient'])} | "
                f"{row['sign']} | {_number(row['magnitude'])} |"
            )
        return output
    output = [
        "| Rank | Feature | Mean validation R² decrease | Repeat SD |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        output.append(
            f"| {rank} | `{row['feature']}` | "
            f"{_number(row['mean_validation_r2_decrease'])} | "
            f"{_number(row['standard_deviation'])} |"
        )
    return output


def _diagnostic_tables(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for experiment in EXPERIMENTS:
        diagnostic = results["test_diagnostics"][experiment]
        lines += [
            f"### {experiment.replace('_', '-')} HGB by Chicago hour",
            "",
            "| Hour | N | Pearson | Spearman | Direction accuracy |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in diagnostic["by_chicago_hour"]:
            lines.append(
                f"| {row['group']} | {row['observations']:,} | {_number(row['pearson'])} | "
                f"{_number(row['spearman'])} | {_percent(row['direction_accuracy'])} |"
            )
        lines += [
            "",
            "Volatility regimes use root-specific tertiles of backward-looking "
            "`realized_vol_15m`, learned only from the deterministic primary training sample.",
            "",
            "| Volatility regime | N | Pearson | Spearman | Direction accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in diagnostic["by_volatility_regime"]:
            lines.append(
                f"| {row['group']} | {row['observations']:,} | {_number(row['pearson'])} | "
                f"{_number(row['spearman'])} | {_percent(row['direction_accuracy'])} |"
            )
        lines += [
            "",
            "| Chicago weekday | N | Pearson | Spearman | Direction accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in diagnostic["by_chicago_weekday"]:
            lines.append(
                f"| {row['group']} | {row['observations']:,} | {_number(row['pearson'])} | "
                f"{_number(row['spearman'])} | {_percent(row['direction_accuracy'])} |"
            )
        lines.append("")
    return lines


def build_report(results: dict[str, Any], old: dict[str, Any]) -> str:
    train_rows = results["primary_training_rows"]
    validation_n = results["experiments"]["A_1m"]["pooled"]["models"]["ridge"][
        "validation"
    ]["overall"]["observations"]
    test_n = results["experiments"]["A_1m"]["pooled"]["models"]["ridge"]["test"][
        "overall"
    ]["observations"]
    b_hgb = results["experiments"]["B_1m"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]
    a_hgb = results["experiments"]["A_1m"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]
    old_a = old["experiments"]["experiment_a"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]["test"]["overall"]
    old_b = old["experiments"]["experiment_b"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]["test"]["overall"]
    lines = [
        "# Phase 2.1 One-Minute Return Prediction and Cross-Market Ablation",
        "",
        "**Status: COMPLETE. Phase 2 artifacts were preserved; no data was downloaded or purchased.**",
        "",
        "## Objective",
        "",
        "Test whether the original 35 causal own-market features predict the next exact one-minute return, and whether 18 exact-UTC cross-market returns help when represented raw (B-1m) or relative to the target market (C-1m). This is predictive-signal research only; no strategy, P&L, costs, execution, or live integration was implemented.",
        "",
        "## Exact target",
        "",
        "`future_return_1m[t] = ln(close[t+1 real minute] / close[t])`. The lookup requires the same root, the same contiguous `contract_segment_id`, and exactly `t+1 minute`. A missing minute produces a null target; the next available row is never substituted. Labels never enter `X`.",
        "",
        "## Data and splits",
        "",
        "The run reused the immutable Phase 2 `experiment_b` Parquet partitions as a common source. A and B are column projections; C is derived in memory as target-own return minus the exact-time anchor return. No Phase 2 feature file was changed.",
        "",
        f"- Roots: {', '.join(results['configuration']['roots'])}.",
        f"- Train/validation boundary: `{results['split_boundaries']['train_validation_boundary']}`.",
        f"- Validation/test boundary: `{results['split_boundaries']['validation_test_boundary']}`.",
        f"- Purge: {results['split_boundaries']['purge_minutes']} minutes on both sides of both boundaries.",
        f"- Eligible training observations before the cap: {sum(results['eligible_train_counts_by_root'].values()):,}.",
        f"- Primary fit: {train_rows:,} observations, exactly 100,000 deterministic evenly spaced rows per root.",
        f"- Full validation: {validation_n:,} observations; untouched test: {test_n:,} observations.",
        "- Preprocessing: train-median imputation for both model families; train-only scaling for Ridge; no numeric scaling for HGB; pooled root one-hot encoding.",
        "",
        "## Experiment A-1m",
        "",
        "A-1m uses the unchanged 35 own-market inputs and root identity. It asks whether a market's own current and historical OHLCV state predicts its next exact minute.",
        "",
        "## Experiment B-1m",
        "",
        "B-1m uses the same 35 inputs plus 18 raw exact-UTC returns: NQ, ES, ZN, CL, GC, and 6E at 1/5/15-minute lookbacks. The old `NQ_ES_return_spread_5m` is excluded, leaving exactly 53 numeric features.",
        "",
        "## Experiment C-1m",
        "",
        "C-1m uses the same 35 inputs plus 18 `own_return_h - anchor_return_h` features. Raw anchors are not simultaneously included. A valid self-anchor is exactly zero; if either operand is unavailable, the relative feature remains null. Because own return plus own-minus-anchor can reconstruct the anchor, C changes representation rather than adding information.",
        "",
        "## Training sample counts",
        "",
        "| Root | Eligible train | Primary fit | 50k study | 100k study | 200k study |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    counts = results["training_sample_counts_by_cap"]
    for root in results["configuration"]["roots"]:
        lines.append(
            f"| {root} | {results['eligible_train_counts_by_root'][root]:,} | "
            f"{counts['100000'][root]:,} | {counts['50000'][root]:,} | "
            f"{counts['100000'][root]:,} | {counts['200000'][root]:,} |"
        )
    lines += [
        "",
        "A/B/C row identities are inherited from one common sampled source, so the primary comparison uses identical root/timestamp observations.",
        "",
        "## Validation results",
        "",
        *_primary_table(results, "validation"),
        "",
        "Macro-average validation metrics (each root receives equal weight):",
        "",
        *_macro_table(results, "validation"),
        "",
        "B-1m HGB had the highest primary pooled validation Pearson; C-1m Ridge had the highest capped Ridge validation Pearson. These choices were made before test evaluation.",
        "",
        "## Test results",
        "",
        *_primary_table(results, "test"),
        "",
        "Macro-average test metrics (each root receives equal weight):",
        "",
        *_macro_table(results, "test"),
        "",
        "Direction accuracy for the six primary learned models:",
        "",
        *_metrics_table(
            [
                (
                    f"{experiment.replace('_', '-')} {'Ridge' if model == 'ridge' else 'HGB'}",
                    results["experiments"][experiment]["pooled"]["models"][model]["test"][
                        "overall"
                    ],
                )
                for experiment in EXPERIMENTS
                for model in MODELS
            ]
        ),
        "",
        "The learned models beat the zero/persistence/reversal sanity baselines on R² and correlation, but the absolute effect remains small. B's test Pearson is microscopically above A's while its R² and Spearman are slightly below A's; that is not robust evidence of incremental cross-market value.",
        "",
        "## Per-market results",
        "",
        "These tables expose heterogeneity rather than allowing large rate/agricultural effects to disappear inside pooled averages.",
        "",
        *_per_market_tables(results),
        "## NQ-only results",
        "",
        "| Experiment | Model | Validation Pearson | Test R² | Test Pearson | Test Spearman | Pooled model on NQ Pearson |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        for model_name in MODELS:
            nq = results["experiments"][experiment]["nq_only"]["models"][model_name]
            pooled_nq = results["experiments"][experiment]["pooled"]["models"][model_name][
                "test"
            ]["per_root"]["NQ"]
            lines.append(
                f"| {experiment.replace('_', '-')} | {'Ridge' if model_name == 'ridge' else 'HGB'} | "
                f"{_number(nq['validation']['overall']['pearson'])} | "
                f"{_number(nq['test']['overall']['r2'])} | "
                f"{_number(nq['test']['overall']['pearson'])} | "
                f"{_number(nq['test']['overall']['spearman'])} | "
                f"{_number(pooled_nq['pearson'])} |"
            )
    lines += [
        "",
        "NQ-only models were approximately zero or negative on test. Every pooled counterpart produced a positive NQ Pearson, so pooled learning helped this NQ sanity check, though NQ's signal remains weak.",
        "",
        "## Naive baseline results",
        "",
        "The naive predictions are feature-representation independent, so the same test values apply to A/B/C.",
        "",
        *_metrics_table(
            [
                (
                    name.replace("_", " ").title(),
                    results["experiments"]["A_1m"]["pooled"]["models"][name]["test"][
                        "overall"
                    ],
                )
                for name in ("zero", "persistence", "reversal")
            ]
        ),
        "",
        "Zero has undefined Pearson/Spearman because its predictions are constant. Its direction score is not conventional model accuracy; it mainly counts exactly zero realized returns. Persistence and reversal both have very negative R².",
        "",
        "## Training-size sensitivity",
        "",
        f"Validation selected `{results['training_size_sensitivity']['experiment']}` for this diagnostic.",
        "",
        "| Rows/root | Pooled rows | Validation R² | Validation Pearson | Validation Spearman |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cap, row in results["training_size_sensitivity"]["rows_per_root"].items():
        metric = row["overall"]
        lines.append(
            f"| {int(cap):,} | {int(cap) * 16:,} | {_number(metric['r2'])} | "
            f"{_number(metric['pearson'])} | {_number(metric['spearman'])} |"
        )
    chosen = results["training_size_sensitivity"]["chosen_test"]["overall"]
    lines += [
        "",
        f"Validation chose {results['training_size_sensitivity']['chosen_rows_per_root']:,} rows/root. Its locked test result was R² `{_number(chosen['r2'])}`, Pearson `{_number(chosen['pearson'])}`, and Spearman `{_number(chosen['spearman'])}`. Performance continued improving through 200k/root, although the gain from 100k to 200k was modest.",
        "",
        "### Full-training Ridge",
        "",
    ]
    full = results["full_ridge"]
    lines += _metrics_table(
        [
            ("Capped 1.6M Ridge validation", full["capped_validation"]["overall"]),
            (
                f"Full {sum(results['eligible_train_counts_by_root'].values()):,}-row Ridge validation",
                full["full_validation"]["overall"],
            ),
        ]
    )
    lines += [
        "",
        f"Validation support for replacing the capped Ridge: **{str(full['validation_supports_full']).upper()}**. The full fit slightly improved R² but reduced Pearson and Spearman, so it was not promoted to final test evaluation. {full['median_note']}",
        "",
        "### Market-specific HGB diagnostic",
        "",
        "Roots were selected only by the pooled B-1m HGB validation Pearson.",
        "",
        "| Root | Pooled val Pearson | Market-specific val Pearson | Pooled test Pearson | Market-specific test Pearson |",
        "|---|---:|---:|---:|---:|",
    ]
    for root, row in results["market_specific_diagnostic"]["results"].items():
        lines.append(
            f"| {root} | {_number(row['pooled_validation']['pearson'])} | "
            f"{_number(row['market_specific_validation']['pearson'])} | "
            f"{_number(row['pooled_test']['pearson'])} | "
            f"{_number(row['market_specific_test']['pearson'])} |"
        )
    lines += [
        "",
        "All three market-specific models improved validation and test Pearson over the pooled model on their root. This is a useful diagnostic, not permission to select markets on test or to build a strategy.",
        "",
        "## Feature importance",
        "",
        "Ridge values are coefficients after train-only standardization. HGB values are deterministic three-repeat permutation importance on 32,000 validation observations (2,000/root), scored as decrease in validation R². Test was not used. Importance reflects model reliance, not causality, mechanism, profitability, or robustness.",
        "",
    ]
    for experiment in EXPERIMENTS:
        importance = results["feature_importance"][experiment]
        lines += [
            f"### {experiment.replace('_', '-')} Ridge top 15",
            "",
            *_importance_table(importance["ridge_standardized_coefficients"], "ridge"),
            "",
            f"### {experiment.replace('_', '-')} HGB top 15 validation permutation importance",
            "",
            *_importance_table(importance["hgb_validation_permutation"], "hgb"),
            "",
        ]
    c_ridge_relative = [
        row
        for row in results["feature_importance"]["C_1m"]["ridge_standardized_coefficients"]
        if row["feature"].startswith("relative_to_")
    ]
    c_hgb_relative = [
        row
        for row in results["feature_importance"]["C_1m"]["hgb_validation_permutation"]
        if row["feature"].startswith("relative_to_")
    ]
    lines += [
        "C-1m did use relative features: "
        f"{len(c_ridge_relative)} appear in Ridge's top 15 and {len(c_hgb_relative)} in HGB's top 15. "
        "However, C did not outperform B, so model reliance did not translate into a superior representation.",
        "",
        "## Time-of-day diagnostics",
        "",
        "All following subgroup results are exploratory test diagnostics. They must not be used to retroactively tune a trading hour, regime, or weekday.",
        "",
        *_diagnostic_tables(results),
        "## Volatility diagnostics",
        "",
        "Across A/B/C, low-volatility rows had the strongest Pearson (about 0.136-0.138), medium volatility was next (about 0.108-0.109), and high volatility was weakest (about 0.037-0.038). This concentration is descriptive and needs new-period replication.",
        "",
        "## Prediction-decile analysis",
        "",
    ]
    for experiment in EXPERIMENTS:
        deciles = results["experiments"][experiment]["pooled"]["models"][
            "hist_gradient_boosting"
        ]["test"]["prediction_deciles"]
        lines += [
            f"### {experiment.replace('_', '-')} HGB",
            "",
            "| Decile | N | Mean prediction | Mean actual | Median actual |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in deciles:
            lines.append(
                f"| {row['decile']} | {row['observations']:,} | "
                f"{_number(row['mean_prediction'])} | {_number(row['mean_actual'])} | "
                f"{_number(row['median_actual'])} |"
            )
        lines.append("")
    sample = results["real_training_row"]
    lines += [
        "For all three HGB models, higher prediction deciles generally correspond to higher mean realized returns, but the relationship is small and is not a profitability result.",
        "",
        "## Real X/y sample",
        "",
        f"One real eligible NQ training row: timestamp `{sample['timestamp']}`, root `{sample['root']}`, instrument `{sample['instrument_id']}`, segment `{sample['contract_segment_id']}`.",
        "",
        "The metadata identifies the observation. `X` contains the inputs below; `y` is the observed next-minute return. Future labels are forbidden from `X`.",
        "",
        "### A-1m: all 35 own-market inputs",
        "",
        "| Feature | Value |",
        "|---|---:|",
    ]
    for feature in results["experiments"]["A_1m"]["feature_order"]:
        lines.append(f"| `{feature}` | {_number(sample[feature], 9)} |")
    lines += [
        "",
        "### B-1m: examples from the 18 raw anchors",
        "",
        "| Feature | Value |",
        "|---|---:|",
    ]
    for feature in ("ES_return_1m", "ES_return_5m", "ES_return_15m", "ZN_return_1m", "ZN_return_5m", "ZN_return_15m"):
        lines.append(f"| `{feature}` | {_number(sample[feature], 9)} |")
    lines += [
        "",
        "### C-1m: corresponding relative examples",
        "",
        "| Feature | Value |",
        "|---|---:|",
    ]
    for feature in (
        "relative_to_ES_return_1m",
        "relative_to_ES_return_5m",
        "relative_to_ES_return_15m",
        "relative_to_ZN_return_1m",
        "relative_to_ZN_return_5m",
        "relative_to_ZN_return_15m",
    ):
        lines.append(f"| `{feature}` | {_number(sample[feature], 9)} |")
    lines += [
        "",
        f"`y = future_return_1m = {_number(sample['future_return_1m'], 9)}`.",
        "",
        "Mathematically, `x_t = [35 own-market features, optional cross-market representation]` and `y_t = ln(P_(t+1)/P_t)`. One sample means: at one specific minute for one futures root, these were all known input conditions, and this was the return that actually happened one minute later.",
        "",
        "## Leakage safeguards",
        "",
        "Automated tests cover all required safeguards:",
        "",
        "1. exact `t+1` target construction;",
        "2. no next-available-row substitution;",
        "3. all future-return labels rejected from `X`;",
        "4. backward-looking own-market lookbacks;",
        "5. no feature crossing a contract segment;",
        "6. no target crossing a contract segment;",
        "7. exact-timestamp anchor joins;",
        "8. no stale/forward-filled anchors;",
        "9. target-root return used as the relative first term;",
        "10. relative formula is target minus anchor;",
        "11. valid self-anchor equals zero;",
        "12. preprocessing statistics remain train-only;",
        "13. validation/test observations are transform-only, never fit inputs; and",
        "14. the original 75-minute boundary purge remains intact.",
        "",
        "## Comparison with previous 5-minute study",
        "",
        "| Phase/model | Target | Test N | R² | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|",
        f"| Phase 2 A HGB | `future_return_5m` | {old_a['observations']:,} | {_number(old_a['r2'])} | {_number(old_a['pearson'])} | {_number(old_a['spearman'])} |",
        f"| Phase 2 B HGB | `future_return_5m` | {old_b['observations']:,} | {_number(old_b['r2'])} | {_number(old_b['pearson'])} | {_number(old_b['spearman'])} |",
        f"| Phase 2.1 A HGB | `future_return_1m` | {a_hgb['test']['overall']['observations']:,} | {_number(a_hgb['test']['overall']['r2'])} | {_number(a_hgb['test']['overall']['pearson'])} | {_number(a_hgb['test']['overall']['spearman'])} |",
        f"| Phase 2.1 B HGB | `future_return_1m` | {b_hgb['test']['overall']['observations']:,} | {_number(b_hgb['test']['overall']['r2'])} | {_number(b_hgb['test']['overall']['pearson'])} | {_number(b_hgb['test']['overall']['spearman'])} |",
        "",
        "These are different targets with different distributions and noise; a larger metric does not establish that one horizon is intrinsically better. The clean Phase 2.1 comparison is A-1m vs B-1m vs C-1m on the same rows.",
        "",
        "## Scientific interpretation",
        "",
        "### Explicit answers",
        "",
        "**Q1. Can the original 35 own-market features predict next-minute returns better than the naive baselines?** Yes in this held-out period: A-1m Ridge/HGB have positive R² and correlations while persistence/reversal are strongly negative-R². The effect is still small.",
        "",
        "**Q2. Does adding raw cross-market information improve the 1-minute model?** Not robustly. B HGB won validation Pearson by only about 0.00030; on test it had a tiny Pearson edge over A but slightly worse R² and Spearman. Ridge B was worse than A on test.",
        "",
        "**Q3. Do target-relative cross-market features outperform raw anchor returns?** No. C did not beat B in the primary HGB validation or test comparison, and its small capped-Ridge validation lead did not persist to test.",
        "",
        "**Q4. Which markets appear strongest on validation?** For the validation-selected B HGB, ZN, ZC, and ZB had the highest Pearson, followed by ZF.",
        "",
        "**Q5. Do those stronger validation markets remain stronger on test?** Yes: ZN, ZB, and ZC were also the top three test Pearson roots (with their order changing), and ZF remained fourth. This is a locked diagnostic, not a new selection rule.",
        "",
        "**Q6. Which features appear most important?** `close_location_in_bar` dominated every HGB permutation ranking. Root identity, volume, short EMA distance, range/wick shape, and short returns were also recurrent. Some raw/relative one-minute anchors appeared, but cross-market representations did not improve the aggregate result robustly.",
        "",
        "**Q7. Does more training data improve the model?** Yes for the B HGB validation study: Pearson increased from about 0.0721 at 50k/root to 0.0791 at 100k and 0.0807 at 200k. Full-data Ridge did not improve all selection metrics and was not promoted.",
        "",
        "**Q8. Does signal appear concentrated by time of day or volatility?** Exploratorily, yes. Low-volatility rows were much stronger than high-volatility rows, and Chicago hours around 11-12 and 21-22 were frequently among the strongest. These test subgroups require independent replication and must not be optimized in place.",
        "",
        "**Q9. Is the evidence strong enough to justify more sophisticated models?** Not yet. The 200k/root gain and market heterogeneity justify a replication/stability experiment, but the weak aggregate R² and failed cross-market ablation do not yet justify deep or highly tuned models.",
        "",
        "**Q10. What is the single best next scientific experiment?** Run a predeclared rolling walk-forward stability study of the B-1m 200k/root HGB and the simpler A-1m control across multiple non-overlapping chronological folds, reporting root-level and volatility-regime stability. This tests whether the observed signal repeats through time before model complexity increases.",
        "",
        "## Limitations",
        "",
        "- Results cover one fixed historical split; subgroup stability is not independently replicated.",
        "- Predictive metrics are small and do not account for fees, spread, slippage, latency, execution, capacity, or risk.",
        "- Exact missing minutes and flat bars cause legitimate nulls handled by train-only median imputation.",
        "- HGB permutation importance uses a documented 32,000-row validation sample for computational practicality.",
        "- The full Ridge uses primary-training-sample medians, then all eligible training rows for scaling statistics and sufficient-statistics coefficients.",
        "- C is a reparameterization because own returns remain in the feature set; it does not add independent information.",
        "",
        "## Recommended next experiment",
        "",
        "Pre-register a rolling walk-forward replication of A-1m and the validation-chosen 200k/root B-1m HGB. Freeze features/hyperparameters, create several chronological train/validation folds with the same rollover and 75-minute purge safeguards, and evaluate whether per-root and low/medium/high-volatility correlations retain their sign and approximate rank. Stop before strategies or P&L.",
        "",
        "## Artifact locations",
        "",
        "- Results: `D:\\futures-ml-data\\artifacts\\models\\phase2_1\\phase2_1_results.json`.",
        "- Models: `D:\\futures-ml-data\\artifacts\\models\\phase2_1\\{A_1m,B_1m,C_1m}\\{pooled,nq_only}`.",
        "- Full Ridge: `D:\\futures-ml-data\\artifacts\\models\\phase2_1\\C_1m\\pooled\\ridge_full_train`.",
        "- Training-size artifacts: `D:\\futures-ml-data\\artifacts\\models\\phase2_1\\training_size`.",
        "- Market-specific artifacts: `D:\\futures-ml-data\\artifacts\\models\\phase2_1\\market_specific`.",
        "- Prediction cache: `D:\\futures-ml-data\\cache\\phase2_1_predictions.joblib`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if results.get("status") != "complete":
        raise RuntimeError("Phase 2.1 results are not complete")
    old = json.loads((PROJECT_ROOT / "reports" / "baseline_metrics.json").read_text("utf-8"))
    OUTPUT_PATH.write_text(build_report(results, old), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
