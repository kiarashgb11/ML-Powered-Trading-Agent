# Project Mission

Build a careful, reproducible futures ML research platform. The immediate scientific question is whether historical market information contains repeatable out-of-sample signal for future returns. This is not currently a production or live trading bot.

# Current Phase

Phase 1 and Phase 2 are complete: repository/storage setup, Databento acquisition and validation, canonical processing, rollover-safe feature/label construction, chronological splitting, and baseline predictive evaluation.

Current handoff (2026-09-23):

- Final universe: NQ, ES, RTY, ZF, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, and ZW.
- Historical range: `2021-09-23` inclusive through `2026-09-23` exclusive UTC.
- Completed schemas for every root: `ohlcv-1m`, `definition`, `statistics`, and `status`; OHLCV-1s, MBP-1/10, MBO, and separate Trades were not acquired.
- Exact data root: `D:\futures-ml-data`. The external manifest is `D:\futures-ml-data\batches\acquisition_manifest.{json,csv}`.
- Final preflight estimate: USD 111.16002381, 57,259,145 records, and 13.803003 GB billable raw. Both the USD 120 budget gate and 51.761260 GB storage gate passed.
- Acquisition result: 64/64 root/schema partitions validated, 57,259,145 records, and 807,206,615 bytes (0.807207 decimal GB) downloaded on disk.
- Storage after acquisition: 707,556,085,760 bytes (707.556086 decimal GB) free on `D:`.
- Validation result: PASS. Checksums and DBN readability passed; invalid OHLC rows, negative-volume rows, duplicate one-minute bars, suspicious gaps over four days, and failed/incomplete partitions are all zero. The continuous OHLCV series contains 440 recorded instrument transitions.
- Known data issues: none found by the current raw-data checks. A missing one-minute bar can legitimately mean no qualifying trade, and definition snapshots can include instrument event timestamps before the requested start.
- MBP-1 is postponed to a later incremental experiment.

Phase 2 handoff (2026-09-23):

- Canonical processed status: PASS. `25,205,190/25,205,190` OHLCV rows are preserved as compressed Parquet under `D:\futures-ml-data\processed\bars_1m`; no resampling, fill, or back-adjustment was used.
- Rollover status: `456` contiguous contract segments and exactly `440` rollovers, matching Phase 1. Segment metadata is under `D:\futures-ml-data\processed\rollovers`.
- Feature-set version: `phase2_v1`. Experiment A has 35 own-market numeric inputs. Experiment B adds 19 limited exact-UTC anchor inputs. Definitions aid interpretation; Statistics and Status are not ML inputs.
- Eligibility: `23,832,724/25,205,190` rows (94.55%) have a valid exact 5-minute target and completed 240-observation segment warm-up. Other legitimate feature nulls remain missing until train-only median imputation.
- Global split: data starts `2021-09-23T00:00:00Z`; train/validation boundary `2025-03-24T04:47:00Z`; validation/test boundary `2025-12-23T02:23:00Z`; data ends `2026-09-22T23:59:00Z`; a 75-minute purge is applied on both sides of each boundary.
- Target: `future_return_5m = ln(close[t+5m]/close[t])`, exact timestamp and same contract segment. Exact 1-minute and 15-minute labels are retained but were not used for the full model matrix.
- Models trained: zero, Ridge, and `HistGradientBoostingRegressor` for pooled Experiment A; Ridge/HGB for pooled Experiment B; matching NQ-only models (plus the Experiment A zero reference).
- Compute contract: complete features are retained; baseline fitting uses a deterministic evenly spaced cap of 100,000 train observations per root (1.6 million pooled). Validation and test metrics use every eligible observation.
- Experiment A pooled HGB test: `R²=0.000489`, Pearson `0.02233`, Spearman `0.06743` on `3,525,964` observations.
- Experiment B pooled HGB test: `R²=0.000576`, Pearson `0.02403`, Spearman `0.06720` on `3,525,964` observations. The aggregate improvement is tiny and not consistent across every root.
- NQ-only HGB remains approximately zero/negative R²; pooled-vs-NQ and all per-root results are in `reports/baseline_model_report.md`.
- Model artifacts: `D:\futures-ml-data\artifacts\models\{experiment_a,experiment_b}\{pooled,nq_only}\...`; each saved pipeline has adjacent metadata containing feature order, encoding, target, splits, seed, versions, and configuration.
- Known issues: none in raw/canonical integrity. Legitimate missing minutes and flat bars create feature nulls. Predictive metrics are extremely weak and cannot be interpreted as profitability. No 1m/15m model diagnostic was run because the primary matrix and audit work were prioritized.
- Exact next recommended task: review the Phase 2 reports and decide whether a narrowly scoped target-horizon or market-specific sampling study is warranted. Do not begin strategy/backtest/live work without a new explicit phase request.

# Core Architecture Decisions

- Databento is the historical/research provider.
- ProjectX/TopstepX is only a possible later live/execution provider.
- Internal features and models must remain provider-neutral.
- Raw data is immutable; transformations write to separate processed/feature areas.
- Processed data uses partitioned compressed Parquet; prefer Polars/PyArrow and lazy or streaming execution.
- Universal OHLCV baseline comes first; MBP-1 is a separate incremental experiment.
- Primary V1 target is future 5-minute log return.
- Use simple baselines before deep learning, reinforcement learning, or LLMs.
- Strategy and execution layers come only after credible out-of-sample model evaluation.

# Futures Universe

Acquired V1 roots: NQ, ES, RTY, ZF, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, ZW.

MBP-1 is postponed; no MBP data is currently present.

# Data Horizons

- `ohlcv-1m`: five years for all 16 acquired roots.
- `definition`, `statistics`, and `status`: same five-year request window for all 16 roots.
- `mbp-1`: postponed; none acquired.
- Do not silently change horizons or markets.

# Storage Rules

- Current desktop: `FUTURES_ML_DATA_ROOT=D:\futures-ml-data`.
- Large files, caches, batch output, temporary extraction, and artifacts must not go to `C:` or Git.
- Other modules consume `futures_ml.config.paths`; never scatter drive-letter paths through code.
- On another machine, configure another root/drive explicitly rather than changing path construction throughout the repository.

# Budget Rules

- Starting Databento credit is approximately USD 125.
- Automatic cost ceiling is USD 120.
- If total estimate exceeds USD 120, stop and present alternatives in this order: shorten MBP from 3→2→1 years; reduce MBP roots from six to NQ/ES/CL/GC; then consider OHLCV from 5→4→3 years.
- Never spend more, shrink the plan, submit a batch job, or download automatically without the documented gate.

# Data Leakage Rules

- Features at time `t` use only observations at or before `t`.
- Never randomly shuffle chronological observations.
- Fit preprocessing only on training data.
- Never include labels in feature columns.
- Use chronological train/validation/test splits and purge overlapping windows near boundaries.
- Remove samples whose feature lookback or target horizon crosses a rollover.
- Never use future contract information improperly in selection or features.

# Futures Rollover Rules

Databento continuous futures are not back-adjusted. Preserve `instrument_id` and actual contract mappings, create an explicit `roll_event` whenever the underlying changes, and do not calculate features, returns, or labels across a transition. V1 favors dropping unsafe samples over synthetic back-adjustment.

# Feature Philosophy

Raw inputs may include OHLCV, trade price/size, top-of-book quotes/sizes/counts, contract metadata, statistics, and status. Derived features may include backward-looking returns, RSI, ATR, VWAP distance, volume z-score, book/trade imbalance, and a small reviewable set of cross-market returns. Avoid arbitrary feature proliferation.

# ML Philosophy

Primary V1 question: “Given everything known at time t, what is the expected futures return over the next five minutes?” Predict returns rather than exact prices. Begin with zero prediction, Ridge, and histogram gradient boosting. Compare pooled multi-market results with an NQ-only sanity model. Predictive performance is not evidence of trading profitability.

# Strategy and Deployment Roadmap

Later research may examine momentum, trend following, mean reversion, VWAP, breakouts, order flow, and regime-specific strategies. Do not implement them before baseline evaluation.

Future architecture: Databento historical → standard internal schema → features → saved model. Later, ProjectX live data may feed the same schema/features, followed by a separate strategy, risk manager, and execution provider. Signal markets (for example NQ) may differ from execution instruments (for example MNQ).

# Coding Conventions

- Python type hints and public API docstrings.
- Small focused modules; no giant scripts or hidden global state.
- Deterministic seeds, structured logging, centralized configuration, and no secrets.
- Tests for transformations, leakage boundaries, rollover rules, and resume behavior.
- Prefer streaming/lazy partitioned processing over giant pandas concatenations.
- Avoid unnecessary services, databases, orchestration, or cloud infrastructure in V1.

# Before Major Changes

Stop and explain the finding, why it matters, and the recommended solution before materially changing dataset universe, horizon, target, roll methodology, provider architecture, feature schema, train/test method, budget, or storage design.

# Documentation Discipline

After a major decision or milestone, update this handoff, update `README.md` when developer workflow changes, update configuration, and record the reason. Never store credentials or secrets here.
