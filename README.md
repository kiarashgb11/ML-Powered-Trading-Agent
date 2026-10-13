# ML-Powered Futures Research Platform

## Project Goal

This repository is a reproducible research platform for testing whether historical futures data contains repeatable out-of-sample predictive signal. The intended progression is:

Databento historical data → validated research datasets → feature engineering → ML return prediction → later strategy research → later risk management → eventually ProjectX/TopstepX paper execution.

This is **not a live trading bot**. It does not connect to a broker, place orders, or claim that predictive metrics imply profitability.

## Current Phase

Phase 1 established:

- portable and safe external storage;
- Databento dataset/schema verification;
- cost, record-count, and billable-size estimation;
- budget and disk-capacity gates;
- resumable historical acquisition and streaming validation infrastructure.

The final V1 acquisition completed and passed validation. It contains 16 markets over `2021-09-23` inclusive through `2026-09-23` exclusive: five years of `ohlcv-1m`, Definitions, Statistics, and Status from `GLBX.MDP3`. All 64 root/schema partitions passed checksum, DBN readability, schema, record-count, and data-quality checks.

Phase 2 is now complete. The project has a canonical rollover-safe one-minute Parquet dataset, Experiment A own-market features, Experiment B exact-time anchor features, exact future-return labels, global chronological splits, automated leakage tests, and zero/Ridge/histogram-gradient-boosting baselines. All 25,205,190 OHLCV rows were preserved, all 440 rollovers reconciled, and 23,832,724 rows are eligible under the exact 5-minute-target plus 240-bar-warm-up rule.

The strongest pooled five-minute test result remains very small: Experiment B histogram gradient boosting achieved approximately `R²=0.000576` and `Pearson=0.0240`.

Phase 2.1 is also complete. It preserved the 35 own-market features, changed the target to the exact next-minute log return, and compared A-1m (own market), B-1m (18 raw exact-time anchor returns), and C-1m (18 target-relative returns) on identical 1.6-million-row training samples. A-1m HGB achieved test `R²=0.003713`, Pearson `0.06924`, and Spearman `0.11511`; B-1m won validation Pearson by a tiny margin but did not improve test R² or Spearman, and C-1m did not outperform B. A validation-only size study selected B-1m HGB at 200,000 rows/root, whose locked test Pearson was `0.07322`. These remain predictive-signal diagnostics, not evidence of profitability. The project still stops before strategies, backtests, fees, execution, risk, or live integration.

Phase 2.2 is complete. Eight predeclared expanding-window folds evaluated frozen A-1m and B-1m HGB models over non-overlapping approximately six-month periods, with a 75-minute purge on both sides of every train/evaluation boundary. Both models had positive pooled Pearson, Spearman, and R² in all eight folds. Mean Pearson was `0.075061` for A and `0.074987` for B; B won only three of eight folds on Pearson and R², so the cross-market increment did not show a repeatable aggregate benefit. ZN, ZC, ZB, and ZF were the strongest and most stable roots, while equity-index roots remained weak. This is evidence of historical temporal stability in a small predictive relationship, not evidence of tradability or profitability.

## Acquired V1 Universe

The acquired V1 universe contains 16 CME futures roots:

| Category | Markets |
|---|---|
| US equity indexes | NQ (Nasdaq-100), ES (S&P 500), RTY (Russell 2000) |
| Interest rates | ZF (5-Year Treasury), ZN (10-Year Treasury), ZB (30-Year Treasury) |
| Energy | CL (Crude Oil), NG (Natural Gas) |
| Metals | GC (Gold), HG (Copper) |
| FX | 6E (Euro), 6J (Japanese Yen), 6B (British Pound) |
| Agriculture | ZC (Corn), ZS (Soybeans), ZW (Wheat) |

MBP-1 was explicitly postponed and was not acquired. Do not expand the acquired universe without documenting and approving the change.

## Data Plan

Completed acquisition:

- NQ, ES, RTY, ZF, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, and ZW.
- `ohlcv-1m`, `definition`, `statistics`, and `status` for every root.
- `2021-09-23` inclusive through `2026-09-23` exclusive UTC.
- Dataset: `GLBX.MDP3`.
- 57,259,145 records in 64 validated root/schema partitions.
- 807,206,615 bytes (0.807 GB decimal) downloaded on disk; the preflight billable raw estimate was 13.803 GB.

Price-series requests use volume-ranked continuous symbols such as `ES.v.0` with `stype_in="continuous"`. Databento continuous prices are original and not back-adjusted. Every record's `instrument_id` must therefore be retained, contract transitions must become explicit roll events, and future work must drop samples whose feature or label windows cross a rollover. Definition requests use parent symbols such as `ES.FUT` to cover underlying contracts.

Missing one-minute bars are not automatically errors: Databento emits no OHLCV record when no qualifying trade occurred. OHLCV-1s, MBP-1/10, MBO, and separate Trades data were not acquired.

## Storage

Large data never belongs in Git. Every large path is derived by [paths.py](src/futures_ml/config/paths.py) from:

```text
FUTURES_ML_DATA_ROOT
```

The current desktop must use:

```text
D:\futures-ml-data
```

Another machine may use a different absolute path and can override the Windows drive gate with `FUTURES_ML_EXPECTED_DRIVE`. The derived external layout is:

```text
FUTURES_ML_DATA_ROOT/
├── raw/{ohlcv_1m,definitions,statistics,status}/
├── processed/{bars_1m,bars_5m,bars_15m,bars_30m,bars_1h,bars_4h,bars_1d}/
├── features/
├── batches/
├── cache/
├── temp/
└── artifacts/models/
```

The estimator refuses to infer a fallback location. On Windows it verifies the expected drive before creating directories. During data work, process-level `TEMP` and `TMP` will be redirected to the external `temp` directory.

## Environment Setup

Python 3.10 or newer is required. Initial development was inspected with Python 3.13.1.

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,research]"
Copy-Item .env.example .env
```

Populate `.env` locally:

```dotenv
DATABENTO_API_KEY=your_key_here
FUTURES_ML_DATA_ROOT=D:\futures-ml-data
```

Never commit `.env`, print the key, or place a real key in documentation. The official client reads `DATABENTO_API_KEY` from the environment.

## Main Commands

Refresh the exact final-plan cost/storage gate from the repository root:

```powershell
python scripts\estimate_final_acquisition.py
```

Resume the manifest-driven acquisition without resubmitting completed jobs:

```powershell
python scripts\download_data.py
```

Regenerate the final data-quality report from the external manifest:

```powershell
python scripts\validate_data.py
```

The earlier alternative/revised/add-on estimators remain available for audit history. They do not alter the completed acquisition.

Run tests and lint checks:

```powershell
python -m pytest
python -m ruff check .
```

The final estimator writes:

- `reports/data_cost_report.csv`
- `reports/data_cost_report.json`
- `reports/data_cost_report.md`

The acquisition manifest is external at `D:\futures-ml-data\batches\acquisition_manifest.{json,csv}`. The validation workflow writes `reports/data_quality_report.{md,json}`.

Build or resume the derived Phase 2 datasets (raw DBN is never modified):

```powershell
python scripts\build_phase2_data.py
```

Regenerate feature partitions after an intentional feature-definition change:

```powershell
python scripts\build_phase2_data.py --force-features
```

Train the fixed baseline matrix and evaluate the untouched test period:

```powershell
python scripts\train_baselines.py
python scripts\generate_phase2_reports.py
```

Run or resume the separate Phase 2.1 one-minute ablation, then regenerate its report:

```powershell
python scripts\run_phase2_1.py
python scripts\generate_phase2_1_report.py
```

Predeclare the Phase 2.2 folds, run/resume the frozen walk-forward study, and regenerate its report:

```powershell
python scripts\predeclare_phase2_2_folds.py
python scripts\run_phase2_2_walk_forward.py
python scripts\generate_phase2_2_report.py
```

Large outputs remain external:

- canonical bars: `D:\futures-ml-data\processed\bars_1m\`;
- rollover tables: `D:\futures-ml-data\processed\rollovers\`;
- Experiment A/B features: `D:\futures-ml-data\features\experiment_{a,b}\`;
- fitted pipelines: `D:\futures-ml-data\artifacts\models\`.
- Phase 2.1 fitted pipelines/results: `D:\futures-ml-data\artifacts\models\phase2_1\`.
- Phase 2.2 fold declarations, fitted pipelines, metadata, and results: `D:\futures-ml-data\artifacts\models\phase2_2\`.

The Phase 2 reports are [processed_dataset_report.md](reports/processed_dataset_report.md), [feature_report.md](reports/feature_report.md), [ml_dataset_walkthrough.md](reports/ml_dataset_walkthrough.md), and [baseline_model_report.md](reports/baseline_model_report.md). The Phase 2.1 findings are in [phase2_1_one_minute_report.md](reports/phase2_1_one_minute_report.md). The Phase 2.2 design and results are in [phase2_2_fold_definitions.md](reports/phase2_2_fold_definitions.md) and [phase2_2_walk_forward_stability_report.md](reports/phase2_2_walk_forward_stability_report.md).

## Cost and Storage Method

For every symbol/schema row, the estimator calls the official Databento Python methods:

- `Historical.metadata.get_cost(...)`
- `Historical.metadata.get_record_count(...)`
- `Historical.metadata.get_billable_size(...)`

It also checks `list_datasets`, `list_schemas`, and `get_dataset_range`. The last fully completed UTC boundary common to every requested schema becomes the exclusive end date; horizons are derived from that boundary rather than hard-coded.

Databento's billable size is uncompressed raw binary used for billing, not final compressed file size. The default storage model is deliberately explicit:

```text
expected working space = total billable bytes × 3.0
required free space     = expected working space × 1.25
```

These multipliers are centralized in `config/data.yaml`. Download authorization requires estimated cost ≤ USD 120 **and** sufficient free space. The estimator itself never downloads data.

## Repository Structure

```text
config/                 fixed universe, data plan, model defaults
src/futures_ml/config/  typed settings and portable external paths
src/futures_ml/data/    Databento client checks and cost estimator
src/futures_ml/features/ causal own/cross-market feature transformations
src/futures_ml/labels/   exact-time rollover-safe return labels
src/futures_ml/models/   baseline pipelines, training, and artifacts
src/futures_ml/evaluation/ chronological splits and predictive metrics
src/futures_ml/interfaces/ future provider-neutral boundaries
scripts/                runnable workflows
models/                 small local pointers/placeholders only; artifacts live externally
reports/                generated human-readable research reports
tests/                  deterministic unit tests
```

## ML Methodology

V1 uses the acquired one-minute OHLCV bars to construct backward-looking features and predict `future_return_5m = log(close[t+5m] / close[t])`. The first models are a zero-return predictor, Ridge regression, and `HistGradientBoostingRegressor`, plus NQ-only sanity models. Experiment A uses 35 own-market inputs; Experiment B adds 19 limited exact-time cross-market inputs.

Phase 2.1 separately predicts `future_return_1m = log(close[t+1m] / close[t])`. A-1m retains the same 35 inputs, B-1m adds 18 raw anchors (the old standalone NQ/ES spread is excluded), and C-1m substitutes 18 target-relative anchor returns. Zero, persistence, reversal, Ridge, and HGB comparisons use the same eligible rows; model selection and importance use validation only.

Phase 2.2 freezes A-1m and B-1m, the Phase 2.1 HGB hyperparameters, and a 200,000-row/root deterministic training cap. It uses expanding training windows and eight non-overlapping evaluation folds. Each fold independently fits preprocessing and the model using only that fold's training data. Fold boundaries and per-root counts were written before any Phase 2.2 model was fit.

All evaluation must use chronological train/validation/test splits. Preprocessing is fit on training data only, overlapping windows are purged near split boundaries, label columns never enter features, and samples crossing a contract transition are removed. The final test period remains untouched during model and hyperparameter decisions.

## Roadmap

1. V1 — complete: historical acquisition and raw-data validation.
2. V2 — complete: clean one-minute research dataset, rollover-safe features/labels, and pooled/NQ-only five-minute-target baseline evaluation.
3. V2.1 — complete: exact one-minute target, raw-vs-relative cross-market ablation, training-size/full-Ridge/market-specific diagnostics.
4. V2.2 — complete: predeclared rolling walk-forward stability study of frozen A-1m and B-1m HGB models.
5. V3 — later scoped MBP-1 experiment after a new estimate and approval.
6. Next recommended research — a predeclared dedicated A-vs-B replication for ZN/ZC/ZB on genuinely new forward data or a newly reserved future holdout.
7. V4 — heuristic and model-assisted strategy research, only after a new explicit phase request.
8. V5 — risk management, costs, slippage, and later strategy walk-forward research.
9. V6 — possible provider-neutral ProjectX/TopstepX paper integration.

Official references used for the current integration are the [Databento Historical API reference](https://databento.com/docs/api-reference-historical), [schema guide](https://databento.com/docs/knowledge-base), and [continuous symbology guide](https://databento.com/docs/standards-and-conventions/symbology).
