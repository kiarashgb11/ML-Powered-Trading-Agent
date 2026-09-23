# ML-Powered Futures Research Platform

## Project Goal

This repository is a reproducible research platform for testing whether historical futures data contains repeatable out-of-sample predictive signal. The intended progression is:

Databento historical data → validated research datasets → feature engineering → ML return prediction → later strategy research → later risk management → eventually ProjectX/TopstepX paper execution.

This is **not a live trading bot**. It does not connect to a broker, place orders, or claim that predictive metrics imply profitability.

## Current Phase

Phase 1 establishes:

- portable and safe external storage;
- Databento dataset/schema verification;
- cost, record-count, and billable-size estimation;
- budget and disk-capacity gates;
- infrastructure for later historical acquisition and baseline ML research.

The repository currently implements setup and the non-downloading estimator. Historical downloading, feature generation, and model training remain gated on a successful cost/storage report.

## Futures Universe

The fixed V1 universe contains 26 CME futures roots:

| Category | Markets |
|---|---|
| US equity indexes | NQ (Nasdaq-100), ES (S&P 500), RTY (Russell 2000), YM (Dow Jones) |
| Interest rates | ZT (2-Year Treasury), ZF (5-Year Treasury), ZN (10-Year Treasury), ZB (30-Year Treasury) |
| Energy | CL (Crude Oil), NG (Natural Gas), RB (RBOB Gasoline), HO (Heating Oil / ULSD) |
| Metals | GC (Gold), SI (Silver), HG (Copper) |
| FX | 6E (Euro), 6J (Japanese Yen), 6B (British Pound), 6A (Australian Dollar), 6C (Canadian Dollar), 6S (Swiss Franc) |
| Agriculture | ZC (Corn), ZW (Wheat), ZS (Soybeans), ZM (Soybean Meal), ZL (Soybean Oil) |

The six core MBP markets are NQ, ES, CL, GC, ZN, and 6E. Do not expand either universe without documenting and approving the change.

## Data Plan

Subject to the USD 120 budget gate and storage gate:

- All 26 markets: approximately five years of `ohlcv-1s`.
- NQ, ES, CL, GC, ZN, and 6E: approximately three years of `mbp-1`.
- All 26 markets: `definition`, `statistics`, and `status` where available.
- Dataset: `GLBX.MDP3`, verified at estimator runtime.

Price-series requests use volume-ranked continuous symbols such as `ES.v.0` with `stype_in="continuous"`. Databento continuous prices are original and not back-adjusted. Every record's `instrument_id` must therefore be retained, contract transitions must become explicit roll events, and future work must drop samples whose feature or label windows cross a rollover. Definition estimates use parent symbols such as `ES.FUT` to cover underlying contracts.

One-second bars are the universal source. Slower bars will be derived deterministically rather than separately purchased. Missing seconds are not automatically errors: Databento emits no OHLCV record when no qualifying trade occurred.

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
├── raw/{ohlcv_1s,mbp1,definitions,statistics,status}/
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
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Populate `.env` locally:

```dotenv
DATABENTO_API_KEY=your_key_here
FUTURES_ML_DATA_ROOT=D:\futures-ml-data
```

Never commit `.env`, print the key, or place a real key in documentation. The official client reads `DATABENTO_API_KEY` from the environment.

## Main Commands

Run the current cost/storage gate from the repository root:

```powershell
python scripts\estimate_data_cost.py
```

Equivalent installed command:

```powershell
futures-ml-estimate
```

Run tests and lint checks:

```powershell
python -m pytest
python -m ruff check .
```

The estimator writes the following only after all configuration and authenticated metadata checks succeed:

- `reports/data_cost_report.csv`
- `reports/data_cost_report.json`
- `reports/data_cost_report.md`

The future phase commands are reserved as follows and are **not implemented yet**: `scripts/download_data.py`, `scripts/validate_data.py`, `scripts/build_features.py`, and `scripts/train_baseline.py`. They must not be added or run until the current budget/storage gate is resolved.

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
src/futures_ml/features/ future resampling and feature transformations
src/futures_ml/labels/   future leakage-safe return labels
src/futures_ml/models/   future baseline models
src/futures_ml/evaluation/ future chronological evaluation
src/futures_ml/interfaces/ future provider-neutral boundaries
scripts/                runnable workflows
models/                 small local pointers/placeholders only; artifacts live externally
reports/                generated human-readable research reports
tests/                  deterministic unit tests
```

## ML Methodology

V1 will aggregate 1-second OHLCV into 1-minute bars, construct backward-looking features, and predict `future_return_5m = log(close[t+5m] / close[t])`. The first models will be a zero-return predictor, Ridge regression, and `HistGradientBoostingRegressor`, plus an NQ-only sanity model.

All evaluation must use chronological train/validation/test splits. Preprocessing is fit on training data only, overlapping windows are purged near split boundaries, label columns never enter features, and samples crossing a contract transition are removed. The final test period remains untouched during model and hyperparameter decisions.

## Roadmap

1. V1 — historical acquisition, validation, 1-minute dataset, and OHLCV baseline.
2. V2 — pooled and NQ-only baseline ML evaluation.
3. V3 — MBP-1 feature experiment on six core markets.
4. V4 — heuristic and model-assisted strategy research.
5. V5 — risk management, costs, slippage, and walk-forward research.
6. V6 — possible provider-neutral ProjectX/TopstepX paper integration.

Official references used for the current integration are the [Databento Historical API reference](https://databento.com/docs/api-reference-historical), [schema guide](https://databento.com/docs/knowledge-base), and [continuous symbology guide](https://databento.com/docs/standards-and-conventions/symbology).
