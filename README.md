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

The strongest pooled test result remains very small: Experiment B histogram gradient boosting achieved approximately `R²=0.000576` and `Pearson=0.0240`. This is a predictive-signal diagnostic, not evidence of profitability. Phase 2 intentionally stops before strategies, backtests, fees, execution, risk, or live integration.

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

Large outputs remain external:

- canonical bars: `D:\futures-ml-data\processed\bars_1m\`;
- rollover tables: `D:\futures-ml-data\processed\rollovers\`;
- Experiment A/B features: `D:\futures-ml-data\features\experiment_{a,b}\`;
- fitted pipelines: `D:\futures-ml-data\artifacts\models\`.

The Phase 2 reports are [processed_dataset_report.md](reports/processed_dataset_report.md), [feature_report.md](reports/feature_report.md), [ml_dataset_walkthrough.md](reports/ml_dataset_walkthrough.md), and [baseline_model_report.md](reports/baseline_model_report.md).

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

All evaluation must use chronological train/validation/test splits. Preprocessing is fit on training data only, overlapping windows are purged near split boundaries, label columns never enter features, and samples crossing a contract transition are removed. The final test period remains untouched during model and hyperparameter decisions.

## Roadmap

1. V1 — complete: historical acquisition and raw-data validation.
2. V2 — complete: clean one-minute research dataset, rollover-safe features/labels, and pooled/NQ-only baseline ML evaluation.
3. V3 — later scoped MBP-1 experiment after a new estimate and approval.
4. V4 — heuristic and model-assisted strategy research.
5. V5 — risk management, costs, slippage, and walk-forward research.
6. V6 — possible provider-neutral ProjectX/TopstepX paper integration.

Official references used for the current integration are the [Databento Historical API reference](https://databento.com/docs/api-reference-historical), [schema guide](https://databento.com/docs/knowledge-base), and [continuous symbology guide](https://databento.com/docs/standards-and-conventions/symbology).
