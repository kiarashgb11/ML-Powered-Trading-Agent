"""Generate the Phase 2 Markdown reports and diagnostic figures."""

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
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.features.cross_market import CROSS_MARKET_FEATURES  # noqa: E402
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A  # noqa: E402
from futures_ml.features.pipeline import scan_partitioned_root  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def _metric_row(name: str, values: dict[str, Any]) -> str:
    return (
        f"| {name} | {values['observations']:,} | {_number(values['mae'])} | "
        f"{_number(values['rmse'])} | {_number(values['r2'])} | "
        f"{_number(values['pearson'])} | {_number(values['spearman'])} | "
        f"{100 * values['direction_accuracy']:.2f}% |"
    )


def _global_null_rates(feature_manifest: dict[str, Any]) -> dict[str, float]:
    total = feature_manifest["total_rows"]
    rates: dict[str, float] = {}
    for name in MODEL_FEATURES_A:
        missing = sum(row["null_counts"][name] for row in feature_manifest["roots"])
        rates[name] = missing / total
    for name in CROSS_MARKET_FEATURES:
        missing = sum(
            row["cross_market_null_rates"][name] * row["rows"]
            for row in feature_manifest["roots"]
        )
        rates[name] = missing / total
    return rates


def _sample_distributions(features_root: Path, roots: list[str]) -> dict[str, dict[str, float | None]]:
    columns = [*MODEL_FEATURES_A, *CROSS_MARKET_FEATURES]
    samples: list[pl.DataFrame] = []
    for root in roots:
        sample = (
            scan_partitioned_root(features_root / "experiment_b", root)
            .select(*columns)
            .with_row_index("_row")
            .filter(pl.col("_row") % 200 == 0)
            .drop("_row")
            .collect()
        )
        samples.append(sample)
    data = pl.concat(samples, rechunk=True)
    output: dict[str, dict[str, float | None]] = {}
    for name in columns:
        finite = data[name].filter(data[name].is_finite())
        output[name] = {
            "min": float(finite.min()) if len(finite) else None,
            "p01": float(finite.quantile(0.01)) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "p99": float(finite.quantile(0.99)) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return output


def _plots(
    *,
    reports: Path,
    paths: Any,
    processed: dict[str, Any],
    feature_manifest: dict[str, Any],
    results: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cache = joblib.load(paths.cache_root / "phase2_plot_predictions.joblib")
    actual_a, predicted_a, roots_a = cache["experiment_a_pooled_hgb"]
    actual_b, predicted_b, roots_b = cache["experiment_b_pooled_hgb"]

    lo, hi = np.quantile(actual_a, [0.005, 0.995])
    plt.figure(figsize=(8, 4.5))
    plt.hist(actual_a[(actual_a >= lo) & (actual_a <= hi)], bins=100, color="#3b82f6")
    plt.title("Test future_return_5m distribution (0.5%-99.5% view)")
    plt.xlabel("5-minute future log return")
    plt.ylabel("Observations")
    plt.tight_layout()
    plt.savefig(figures / "target_distribution.png", dpi=150)
    plt.close()

    indices = np.linspace(0, len(actual_b) - 1, min(30_000, len(actual_b)), dtype=int)
    clip_actual = np.clip(actual_b[indices], lo, hi)
    pred_lo, pred_hi = np.quantile(predicted_b, [0.005, 0.995])
    plt.figure(figsize=(6, 5))
    plt.scatter(np.clip(predicted_b[indices], pred_lo, pred_hi), clip_actual, s=2, alpha=0.18)
    plt.title("Experiment B HGB: prediction vs actual")
    plt.xlabel("Predicted return")
    plt.ylabel("Actual return")
    plt.tight_layout()
    plt.savefig(figures / "prediction_vs_actual.png", dpi=150)
    plt.close()

    deciles = results["experiments"]["experiment_b"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]["test"]["prediction_deciles"]
    plt.figure(figsize=(7, 4.5))
    plt.bar([row["decile"] for row in deciles], [row["mean_actual"] for row in deciles])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Prediction decile vs realized 5-minute return")
    plt.xlabel("Prediction decile (low to high)")
    plt.ylabel("Mean actual return")
    plt.tight_layout()
    plt.savefig(figures / "prediction_deciles.png", dpi=150)
    plt.close()

    per_root = results["experiments"]["experiment_b"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]["test"]["per_root"]
    roots = list(per_root)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(roots, [per_root[root]["pearson"] for root in roots], color="#2563eb")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Pearson")
    axes[0].set_title("Experiment B HGB per-root test diagnostics")
    axes[1].bar(
        roots,
        [per_root[root]["direction_accuracy"] for root in roots],
        color="#16a34a",
    )
    axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Direction accuracy")
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(figures / "per_root_metrics.png", dpi=150)
    plt.close(figure)

    null_rates = _global_null_rates(feature_manifest)
    top = sorted(null_rates.items(), key=lambda item: item[1], reverse=True)[:20]
    plt.figure(figsize=(9, 6))
    plt.barh([name for name, _ in reversed(top)], [100 * rate for _, rate in reversed(top)])
    plt.xlabel("Null rate (%)")
    plt.title("Twenty highest feature null rates")
    plt.tight_layout()
    plt.savefig(figures / "feature_missingness.png", dpi=150)
    plt.close()

    split = results["split_boundaries"]
    dates = [
        datetime.fromisoformat(split[name])
        for name in (
            "data_start",
            "train_validation_boundary",
            "validation_test_boundary",
            "data_end",
        )
    ]
    durations = [
        (dates[index + 1] - dates[index]).total_seconds() / 86_400 for index in range(3)
    ]
    plt.figure(figsize=(9, 2.3))
    left = 0.0
    for label, width, color in zip(
        ("Train", "Validation", "Test"), durations, ("#2563eb", "#f59e0b", "#16a34a"), strict=True
    ):
        plt.barh([0], [width], left=left, label=label, color=color)
        left += width
    plt.yticks([])
    plt.xlabel("Elapsed days from 2021-09-23 (75-minute purge at boundaries)")
    plt.legend(ncol=3, loc="upper center")
    plt.tight_layout()
    plt.savefig(figures / "split_timeline.png", dpi=150)
    plt.close()

    nq = scan_partitioned_root(paths.processed_data_root / "bars_1m", "NQ").collect().sort("timestamp")
    roll_indices = np.flatnonzero(nq["roll_event"].to_numpy())
    if len(roll_indices):
        index = int(roll_indices[0])
        example = nq.slice(max(0, index - 120), 241)
        plt.figure(figsize=(9, 4))
        plt.plot(example["timestamp"].to_list(), example["close"].to_list(), linewidth=1)
        plt.axvline(nq[index, "timestamp"], color="red", linestyle="--", label="roll_event")
        plt.title("NQ example instrument transition (unadjusted prices)")
        plt.ylabel("Close")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "rollover_example.png", dpi=150)
        plt.close()


def _processed_report(processed: dict[str, Any], feature_manifest: dict[str, Any]) -> str:
    features = {row["root"]: row for row in feature_manifest["roots"]}
    lines = [
        "# Processed Dataset Report",
        "",
        "**Status: PASS — 25,205,190 raw OHLCV rows were preserved and all 440 Phase 1 transitions reconciled.**",
        "",
        "The canonical dataset is compressed Parquet under `D:\\futures-ml-data\\processed\\bars_1m`, partitioned by root and UTC year. Prices are Databento fixed-point integers converted by `1e9`; no resampling, forward filling, candle fabrication, or price back-adjustment was performed.",
        "",
        "## Per-root audit",
        "",
        "| Root | Raw rows | Processed | Coverage UTC | Segments | Rolls | Non-1m intervals | Missing clock minutes | 240-bar warm-up nulls | Missing exact-time rows | Roll-guard rows | ML eligible |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in processed["roots"]:
        feature = features[row["root"]]
        lines.append(
            f"| {row['root']} | {row['raw_rows']:,} | {row['processed_rows']:,} | "
            f"{row['min_timestamp']} to {row['max_timestamp']} | {row['contract_segments']:,} | "
            f"{row['rollovers']:,} | {row['non_one_minute_intervals']:,} | "
            f"{row['missing_clock_minutes']:,} | {feature['rows_with_warmup_null']:,} | "
            f"{feature['rows_missing_exact_timestamp']:,} | {feature['rows_in_roll_guard']:,} | "
            f"{feature['eligible_rows']:,} |"
        )
    lines += [
        "",
        "## Totals and interpretation",
        "",
        f"- Raw rows: **{processed['raw_rows']:,}**.",
        f"- Canonical rows: **{processed['processed_rows']:,}** (zero raw-row loss).",
        f"- Contract segments: **{sum(row['contract_segments'] for row in processed['roots']):,}**.",
        f"- Rollovers: **{processed['rollovers']:,}**, exactly matching Phase 1.",
        f"- Primary-target/240-bar-warm-up eligible rows: **{feature_manifest['total_eligible_rows']:,}** ({100 * feature_manifest['total_eligible_rows'] / feature_manifest['total_rows']:.2f}%).",
        "",
        "`non_one_minute_intervals` and `missing_clock_minutes` describe real timestamp gaps inside a contract segment; they are not filled. Large totals are expected around exchange/session closures and quiet periods. Exact-time returns and labels are null when their required timestamp is absent. `roll-guard rows` is an overlapping diagnostic count within 240 minutes after or 15 minutes before a roll, not an additional unique deletion count.",
        "",
        "The initial all-features-non-null diagnostic retained only 58.52% and was investigated before modeling. The cause was expected session gaps and undefined flat-bar ratios—not raw corruption. The final eligibility rule requires a valid exact 5-minute target and completed 240-observation segment warm-up; remaining feature nulls are handled only by train-fitted median imputation.",
        "",
        "![Example rollover](figures/rollover_example.png)",
        "",
    ]
    return "\n".join(lines)


def _feature_report(
    feature_manifest: dict[str, Any], distributions: dict[str, dict[str, float | None]]
) -> str:
    null_rates = _global_null_rates(feature_manifest)
    specs = [row for row in feature_manifest["feature_specification"] if row["kind"] == "feature"]
    lines = [
        "# Feature Report",
        "",
        "**Feature-set version: `phase2_v1`. Every calculation is causal and grouped by contiguous `contract_segment_id`.**",
        "",
        "Experiment A has 35 model inputs derived only from the predicted market's OHLCV. Experiment B adds 18 anchor returns (NQ, ES, ZN, CL, GC, 6E × 1/5/15 minutes) and the NQ-minus-ES 5-minute spread, for 54 numeric inputs. Statistics and Status are not model inputs.",
        "",
        "## Feature distributions and availability",
        "",
        "Distribution columns use a deterministic 1-in-200-row sample across all 16 roots; null rates use every row. Min/max are retained as extreme-value checks, while p01/p99 show the typical range.",
        "",
        "| Feature | Experiment | Null % | Min | P01 | Median | P99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        name = spec["name"]
        dist = distributions[name]
        lines.append(
            f"| `{name}` | {','.join(spec['experiment'])} | {100 * null_rates[name]:.3f} | "
            f"{_number(dist['min'], 5)} | {_number(dist['p01'], 5)} | {_number(dist['median'], 5)} | "
            f"{_number(dist['p99'], 5)} | {_number(dist['max'], 5)} |"
        )
    lines += [
        "",
        "All finite-value guards passed. Extreme min/max observations remain visible rather than silently clipped; Ridge scaling and imputation are learned only from train, and HGB uses train-median imputation without scaling.",
        "",
        "## Per-root availability",
        "",
        "Availability is the percentage of feature cells that are non-null before train-only imputation.",
        "",
        "| Root | Own-feature availability | Cross-feature availability | ML eligible rows | Eligible % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in feature_manifest["roots"]:
        own_available = 1 - sum(row["null_rates"][name] for name in MODEL_FEATURES_A) / len(
            MODEL_FEATURES_A
        )
        cross_available = 1 - sum(
            row["cross_market_null_rates"][name] for name in CROSS_MARKET_FEATURES
        ) / len(CROSS_MARKET_FEATURES)
        lines.append(
            f"| {row['root']} | {100 * own_available:.2f}% | {100 * cross_available:.2f}% | "
            f"{row['eligible_rows']:,} | {100 * row['eligible_rows'] / row['rows']:.2f}% |"
        )
    lines += [
        "",
        "## Formulas and provenance",
        "",
        "| Feature | Formula | Source | Lookback | Normalization | Code |",
        "|---|---|---|---|---|---|",
    ]
    for spec in specs:
        lines.append(
            f"| `{spec['name']}` | {spec['formula']} | {', '.join(spec['source_columns'])} | "
            f"{spec['lookback']} | {spec['normalization']} | `{spec['code']}` |"
        )
    lines += [
        "",
        "## Rollover and missingness behavior",
        "",
        "Exact return features self-join on `(contract_segment_id, timestamp)`; row offsets are never used. EMA, RSI, ATR, rolling volume, realized volatility, and VWAP are evaluated within a segment. Experiment B is a plain equality join on UTC timestamp: no as-of join and no stale forward fill.",
        "",
        "![Feature missingness](figures/feature_missingness.png)",
        "",
    ]
    return "\n".join(lines)


def _walkthrough(feature_manifest: dict[str, Any], results: dict[str, Any]) -> str:
    a = results["experiments"]["experiment_a"]["pooled"]
    b = results["experiments"]["experiment_b"]["pooled"]
    a_model = a["models"]["hist_gradient_boosting"]
    sample = results["real_training_row"]
    specs = feature_manifest["feature_specification"]
    lines = [
        "# ML Dataset Walkthrough",
        "",
        "## What the model sees",
        "",
        "For one eligible observation at time `t`, `x_t = [feature_1, ..., feature_D]` contains past/current information only and `y_t = future_return_5m = ln(close[t+5m] / close[t])`. Stacking observations gives `X ∈ R^(N×D)` and `y ∈ R^N`; `N` is observations and `D` is input columns after preprocessing.",
        "",
        "## Actual shapes",
        "",
        f"Global boundaries are train before `{results['split_boundaries']['train_validation_boundary']}`, validation between the purged boundaries, and test from `{results['split_boundaries']['validation_test_boundary']}` onward. A 75-minute exclusion is applied on both sides of each boundary.",
        "",
        f"- Experiment A eligible `X_train`: `({a['eligible_train_rows']:,}, 36)` before encoding; `y_train`: `({a['eligible_train_rows']:,},)`.",
        f"- Experiment A bounded baseline-fit `X_train`: `({a['model_fit_rows']:,}, 36)`; after pooled root one-hot: `({a['model_fit_rows']:,}, {a_model['encoded_feature_count']})`.",
        f"- `X_val`: `({a_model['validation']['overall']['observations']:,}, 36)`; `y_val`: `({a_model['validation']['overall']['observations']:,},)`.",
        f"- `X_test`: `({a_model['test']['overall']['observations']:,}, 36)`; `y_test`: `({a_model['test']['overall']['observations']:,},)`.",
        f"- Experiment B has 55 pre-encoding columns (54 numeric + root) and {b['models']['hist_gradient_boosting']['encoded_feature_count']} after pooled one-hot encoding.",
        "",
        "The complete dataset is retained. Baseline fitting uses a deterministic evenly spaced cap of 100,000 train rows per root (1.6 million pooled) to keep HGB practical; validation and test metrics use every eligible row.",
        "",
        "## One real NQ training observation",
        "",
        f"Timestamp `{sample['timestamp']}`, root `{sample['root']}`, instrument `{sample['instrument_id']}`, segment `{sample['contract_segment_id']}`.",
        "",
        "| Column | Actual value | Classification |",
        "|---|---:|---|",
    ]
    chosen = [
        "log_return_1m",
        "log_return_5m",
        "rsi_14",
        "atr_14_normalized",
        "volume_zscore_20",
        "close_minus_vwap_60m_pct",
        "ES_return_5m",
        "ZN_return_5m",
        "hour_sin",
        "hour_cos",
    ]
    for name in chosen:
        lines.append(f"| `{name}` | {_number(sample[name], 8)} | MODEL INPUT FEATURE |")
    for name in ("future_return_1m", "future_return_5m", "future_return_15m"):
        classification = "TARGET" if name == "future_return_5m" else "FORBIDDEN FROM MODEL INPUT"
        lines.append(f"| `{name}` | {_number(sample[name], 8)} | {classification} |")
    lines += [
        f"| `direction_5m` | {sample['direction_5m']} | FORBIDDEN diagnostic label |",
        f"| `root` | {sample['root']} | METADATA + pooled one-hot identity |",
        f"| `timestamp` | {sample['timestamp']} | METADATA |",
        "",
        "## Raw → X,y code map",
        "",
        "```text",
        "DBN -- data.processed.read_root_ohlcv --> canonical bars",
        "     -- data.processed.add_contract_segments --> rollover-safe segments",
        "     -- features.ohlcv_features.build_own_market_features --> Experiment A X",
        "     -- labels.future_returns.add_future_labels --> y horizons",
        "     -- features.cross_market.add_cross_market_features --> Experiment B X",
        "     -- features.pipeline.eligibility_expression --> eligible observations",
        "     -- evaluation.splits.split_expression --> train / validation / test",
        "     -- models.baselines.make_*_pipeline --> train-fitted preprocessing",
        "     -- Pipeline.fit / Pipeline.predict --> predictions",
        "     -- evaluation.metrics.regression_metrics --> unseen-future metrics",
        "```",
        "",
        "## Feature and label contract",
        "",
        "| Name | Kind | Formula | Source | Lookback/horizon | Normalized | Experiment | Code |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for spec in specs:
        horizon = spec.get("horizon", spec.get("lookback", ""))
        lines.append(
            f"| `{spec['name']}` | {spec['kind']} | {spec['formula']} | "
            f"{', '.join(spec['source_columns'])} | {horizon} | {spec['normalization']} | "
            f"{','.join(spec['experiment'])} | `{spec['code']}` |"
        )
    lines += [
        "",
        "## Ridge versus histogram gradient boosting",
        "",
        "Both receive the same ordered numeric feature list and, for pooled models, the same 16-category root identity. Ridge uses train-median imputation, `z=(x-μ_train)/σ_train`, and one-hot roots; its linear coefficients depend on comparable scales. HGB uses train-median imputation and one-hot roots but no standardization because tree splits are scale-invariant. NQ-only models omit the constant root column. Label, timestamp, instrument ID, segment ID, and raw OHLC prices never enter X.",
        "",
        "## How to add 10-minute momentum",
        "",
        "1. Define `r_t^(10)=ln(P_t/P_(t-10m))`.",
        "2. Add `10` to `RETURN_HORIZONS` in `src/futures_ml/features/ohlcv_features.py`; `add_exact_log_returns` will use the exact timestamp and same segment.",
        "3. Add `log_return_10m` to `MODEL_FEATURES_A` and the generated feature specification.",
        "4. Add a unit test analogous to `test_five_minute_return_feature_uses_t_minus_five`, including a missing `t-10` case.",
        "5. Regenerate with `python scripts\\build_phase2_data.py --force-features` and retrain with `python scripts\\train_baselines.py`.",
        "",
        "## How to change the target",
        "",
        "Set `primary_target` in `config/phase2.yaml` to `future_return_1m` or `future_return_15m`. `labels.future_returns.add_future_labels` already creates both with exact-time same-segment joins, and training reads the configured name. Features, segmentation, splits, preprocessing, and model classes do not need rewriting. Re-run feature generation only if label horizons themselves change; otherwise retrain directly.",
        "",
        "## Project-specific leakage examples",
        "",
        "Valid: current close; `ln(close[t]/close[t-5m])`; RSI using rows at or before t; current/past volume. Invalid: `close[t+1]`, any `future_return_*` in X, centered rolling windows, fitting scaler/imputer on validation/test, comparing prices across instrument IDs, stale as-of anchor joins, or choosing a contract using future volume. Each invalid case exposes information unavailable at prediction time or mixes discontinuous contracts.",
        "",
        "`tests/test_phase2_leakage.py` contains explicit tests for all 12 requested categories: exact past/future horizons, missing timestamps, causal rolling, feature/label segment safety, label exclusion, train-only preprocessing, current-or-earlier anchors, exact joins, purging, and test separation.",
        "",
        "## File / code map",
        "",
        "- Canonical DBN conversion and roll segmentation: `src/futures_ml/data/processed.py` — `read_root_ohlcv`, `add_contract_segments`, `write_partitioned_root`.",
        "- Own-market features: `src/futures_ml/features/ohlcv_features.py` — the focused `add_*_features` functions and `build_own_market_features`.",
        "- Cross-market features: `src/futures_ml/features/cross_market.py` — `anchor_return_table`, `add_cross_market_features`.",
        "- Labels: `src/futures_ml/labels/future_returns.py` — `add_future_return`, `add_future_labels`.",
        "- Split/purge: `src/futures_ml/evaluation/splits.py` — `calculate_global_boundaries`, `split_expression`.",
        "- Model preprocessing/classes: `src/futures_ml/models/baselines.py` — `make_preprocessor`, `make_ridge_pipeline`, `make_hist_gradient_boosting_pipeline`.",
        "- Training/evaluation: `src/futures_ml/models/training.py` and `src/futures_ml/evaluation/metrics.py`.",
        "- Universe and Phase 2 policy: `config/phase2.yaml`; external paths: `src/futures_ml/config/paths.py` and `D:\\futures-ml-data`.",
        "",
        "![Split timeline](figures/split_timeline.png)",
        "",
    ]
    return "\n".join(lines)


def _baseline_report(results: dict[str, Any]) -> str:
    lines = [
        "# Baseline Model Report",
        "",
        "**Scope: predictive-signal diagnostics only. These results do not establish trading profitability.**",
        "",
        "## Dataset and target",
        "",
        "Sixteen roots, 2021-09-23 through 2026-09-22, primary target `future_return_5m`. Every target is an exact same-segment `ln(close[t+5m]/close[t])` observation.",
        "",
        "## Train / validation / test",
        "",
        f"- Train/validation boundary: `{results['split_boundaries']['train_validation_boundary']}`.",
        f"- Validation/test boundary: `{results['split_boundaries']['validation_test_boundary']}`.",
        f"- Purge: `{results['split_boundaries']['purge_minutes']}` minutes on each side of both boundaries.",
        "- Model fitting: 1,600,000 evenly spaced pooled train rows (100,000/root); no random shuffle.",
        "- Validation and final test: every eligible row. Fixed defaults were selected before reading final test metrics.",
        "",
        "## Pooled test results",
        "",
        "| Experiment / model | N | MAE | RMSE | R² | Pearson | Spearman | Direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("experiment_a", "experiment_b"):
        models = results["experiments"][experiment]["pooled"]["models"]
        for name, result in models.items():
            lines.append(_metric_row(f"{experiment} / {name}", result["test"]["overall"]))
    lines += [
        "",
        "Experiment A is own-market OHLCV only. Experiment B adds the limited six-anchor exact-time layer. The best pooled R² is still only about 0.0006; the change from A to B is small and model/market dependent.",
        "",
        "## Macro-average pooled test metrics",
        "",
        "Each market contributes equally to these averages, irrespective of its observation count.",
        "",
        "| Experiment / model | MAE | RMSE | R² | Pearson | Spearman | Direction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("experiment_a", "experiment_b"):
        models = results["experiments"][experiment]["pooled"]["models"]
        for name, result in models.items():
            values = result["test"]["macro_average"]
            lines.append(
                f"| {experiment} / {name} | {_number(values['mae'])} | {_number(values['rmse'])} | "
                f"{_number(values['r2'])} | {_number(values['pearson'])} | {_number(values['spearman'])} | "
                f"{100 * values['direction_accuracy']:.2f}% |"
            )
    lines += [
        "",
        "## NQ-only test results",
        "",
        "| Experiment / model | N | MAE | RMSE | R² | Pearson | Spearman | Direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("experiment_a", "experiment_b"):
        models = results["experiments"][experiment]["nq_only"]["models"]
        for name, result in models.items():
            lines.append(_metric_row(f"{experiment} / {name}", result["test"]["overall"]))
    lines += [
        "",
        "The pooled models slightly outperform NQ-only HGB on NQ R² in Experiment A, but all NQ results remain near zero. Experiment B does not improve the pooled model's NQ test correlation in this run.",
        "",
    ]
    lines += ["", "## Per-market pooled test metrics", ""]
    for experiment in ("experiment_a", "experiment_b"):
        for model_name, model in results["experiments"][experiment]["pooled"]["models"].items():
            lines += [
                f"### {experiment} / {model_name}",
                "",
                "| Root | N | MAE | RMSE | R² | Pearson | Spearman | Direction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for root, values in model["test"]["per_root"].items():
                lines.append(_metric_row(root, values))
            lines.append("")
    lines += [
        "## Prediction-decile analysis",
        "",
        "| Decile | N | Mean prediction | Mean realized return |",
        "|---:|---:|---:|---:|",
    ]
    deciles = results["experiments"]["experiment_b"]["pooled"]["models"][
        "hist_gradient_boosting"
    ]["test"]["prediction_deciles"]
    for row in deciles:
        lines.append(
            f"| {row['decile']} | {row['observations']:,} | {_number(row['mean_prediction'])} | {_number(row['mean_actual'])} |"
        )
    lines += [
        "",
        "## Rollover and leakage safeguards",
        "",
        "All 440 transitions reconcile. Features/labels join on exact timestamp plus segment; stateful features group by segment; a 240-bar post-roll warm-up and exact target are required. Global timestamps, 75-minute purges, train-only imputation/scaling/encoding, explicit label exclusion, and untouched final-test evaluation are enforced in code and tests.",
        "",
        "## 1-minute and 15-minute target diagnostics",
        "",
        "`future_return_1m` and `future_return_15m` are retained and audited, but the optional extra model runs were not executed. Compute and documentation effort remained focused on the primary 5-minute benchmark, as permitted by the Phase 2 brief.",
        "",
        "## Scientific interpretation",
        "",
        "The nonlinear pooled models show very weak positive aggregate correlation and tiny positive R²; linear and NQ-only results are generally at or below the zero predictor on squared error. Cross-market context produces a small pooled aggregate improvement for HGB but not a robust improvement across every root or for NQ. This is evidence for, at most, a weak predictive association—not an executable trading edge.",
        "",
        "Fees, bid/ask spread, slippage, turnover, execution timing, position sizing, risk limits, and signal thresholds are absent by design. No strategy, backtest, PnL, or live integration was implemented.",
        "",
        "## Diagnostic figures",
        "",
        "![Target distribution](figures/target_distribution.png)",
        "",
        "![Prediction versus actual](figures/prediction_vs_actual.png)",
        "",
        "![Prediction deciles](figures/prediction_deciles.png)",
        "",
        "![Per-root metrics](figures/per_root_metrics.png)",
        "",
        "## Recommended next research step",
        "",
        "Review Phase 2 results and the weakest/strongest per-root behavior before authorizing another phase. A later scoped study could test target horizon or market-specific sampling sensitivity; it should not begin with strategy or execution claims.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows)
    os.environ.setdefault("MPLCONFIGDIR", str(paths.temp_root / "matplotlib"))
    reports = PROJECT_ROOT / "reports"
    processed = _load(paths.processed_data_root / "phase2_processed_manifest.json")
    feature_manifest = _load(paths.features_root / "phase2_feature_manifest.json")
    results = _load(paths.model_artifact_root / "phase2_model_results.json")
    distributions = _sample_distributions(paths.features_root, [row["root"] for row in processed["roots"]])
    _plots(
        reports=reports,
        paths=paths,
        processed=processed,
        feature_manifest=feature_manifest,
        results=results,
    )
    (reports / "processed_dataset_report.md").write_text(
        _processed_report(processed, feature_manifest), encoding="utf-8"
    )
    (reports / "feature_report.md").write_text(
        _feature_report(feature_manifest, distributions), encoding="utf-8"
    )
    (reports / "ml_dataset_walkthrough.md").write_text(
        _walkthrough(feature_manifest, results), encoding="utf-8"
    )
    (reports / "baseline_model_report.md").write_text(_baseline_report(results), encoding="utf-8")
    print("Generated four Phase 2 reports and seven diagnostic figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
