# Project Mission

Build a careful, reproducible futures ML research platform. The immediate scientific question is whether historical market information contains repeatable out-of-sample signal for future returns. This is not currently a production or live trading bot.

# Current Phase

Phase 1, Phase 2, Phase 2.1, and Phase 2.2 are complete. Phase 2.3 model freezing and evaluator preparation are complete; prospective confirmation is awaiting genuinely new data. The repository now covers acquisition/validation, canonical rollover-safe features and labels, baseline and ablation studies, temporal replication, and immutable dedicated candidates for prospective evaluation.

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

Phase 2.1 handoff (2026-10-12):

- Phase 2 outputs remain intact. Phase 2.1 reuses `D:\futures-ml-data\features\experiment_b` as an immutable common source; no data was downloaded and no Phase 2 Parquet/model/report was overwritten.
- Target: `future_return_1m = ln(close[t+1m]/close[t])`, requiring the same root, same contract segment, and exact next real minute. The existing global boundaries and 75-minute purge are unchanged.
- A-1m has the original 35 own-market features. B-1m adds 18 raw exact-UTC NQ/ES/ZN/CL/GC/6E returns at 1/5/15 minutes and excludes the old standalone NQ/ES spread. C-1m adds 18 target-minus-anchor returns instead of raw anchors.
- Primary fits use identical deterministic 100,000-row/root timestamps across A/B/C: 1,600,000 pooled observations. Validation uses all 3,519,199 eligible rows and test uses all 3,551,762 eligible rows.
- Primary pooled HGB test: A `R²=0.003713`, Pearson `0.069236`, Spearman `0.115111`; B `R²=0.003704`, Pearson `0.069417`, Spearman `0.114225`; C `R²=0.003642`, Pearson `0.069009`, Spearman `0.114085`.
- Cross-market conclusion: B won primary validation Pearson by only `0.000297` over A, but did not improve test R² or Spearman. C did not outperform B. Raw/relative cross-market additions therefore showed no robust aggregate incremental benefit.
- Training-size result: B HGB validation Pearson increased `0.07209 -> 0.07911 -> 0.08074` at 50k/100k/200k rows per root. Validation selected 200k/root; locked test Pearson was `0.07322` and R² `0.004082`.
- Full-training Ridge used all 16,914,241 eligible train rows for scaling statistics and coefficients, with train-only medians learned from the common primary sample. It slightly improved validation R² but reduced Pearson/Spearman, so validation did not promote it to test.
- Validation selected ZN, ZC, and ZB for the small market-specific HGB diagnostic. Each market-specific model improved validation and locked-test Pearson over the pooled model on its root; this is diagnostic evidence only.
- Feature importance: `close_location_in_bar` dominated HGB validation permutation importance; root identity, volume, short EMA distance, range/wick shape, and short returns recurred. Relative features were used by C but did not improve its aggregate result.
- Exploratory test diagnostics were strongest in low-volatility rows and varied by Chicago hour. They were not used for model selection and must be independently replicated.
- Artifacts: `D:\futures-ml-data\artifacts\models\phase2_1`; complete report: `reports/phase2_1_one_minute_report.md`; compact result bundle: `reports/phase2_1_metrics.json`.
- Known limitations: one fixed split, small aggregate effect, approximate validation permutation importance, and no transaction-cost/execution analysis. Predictive metrics are not profitability evidence.
- Exact next recommended task: predeclare a rolling walk-forward stability study of the A-1m control and validation-chosen B-1m 200k/root HGB across multiple chronological folds, preserving rollover and purge safeguards. Do not begin strategy/backtest/live work without a new explicit phase request.

Phase 2.2 handoff (2026-10-13):

- Phase 2 and Phase 2.1 outputs remain intact. Phase 2.2 reuses the immutable Phase 2 feature source and changes no feature, label, universe, rollover rule, or historical row.
- Eight expanding-window folds were declared and counted before training. Evaluation windows are non-overlapping and approximately six months, running from `2022-09-23` through the existing `2026-09-23` data end. Every boundary has a 75-minute purge on both sides.
- Frozen models: A-1m HGB with 35 own-market inputs and B-1m HGB with those inputs plus 18 raw exact-UTC anchors. HGB settings remain learning rate `0.08`, 150 iterations, 31 leaves, minimum leaf size 100, L2 `0.1`, seed 42.
- Training uses deterministic evenly spaced samples capped at 200,000 observations/root. Fold 1 uses all 187,416 eligible ZW observations; every other root/fold reaches the cap. Preprocessing is independently fit within each fold.
- Pooled A results across eight folds: mean Pearson `0.075061`, SD `0.010030`; mean Spearman `0.107394`; mean R² `0.004342`. Pearson, Spearman, and R² are positive in `8/8` folds.
- Pooled B results: mean Pearson `0.074987`, SD `0.010177`; mean Spearman `0.107207`; mean R² `0.004323`. All three metrics are positive in `8/8` folds.
- B does not have a stable aggregate advantage: it wins only `3/8` folds on Pearson and R² and `2/8` on Spearman. Mean B-minus-A Pearson is `-0.000074`.
- B per-root Pearson is strongest and positive in all folds for ZN (`0.27360` mean), ZC (`0.23134`), ZB (`0.21063`), and ZF (`0.16703`). NQ remains weak (`0.00939`, positive `6/8`); ES and RTY are also much weaker than the leading rates/agriculture roots.
- The Phase 2.1 low-volatility diagnostic replicated in all eight folds. Broad prediction-decile ordering also replicated, but strict fold-level monotonicity occurred in only `3/8` A folds and `4/8` B folds.
- Dedicated B models for ZN/ZC/ZB beat the pooled B model on their own root in `8/8` folds, with mean Pearson lifts of `0.01832`, `0.00909`, and `0.04346`, respectively. This is specialization evidence only, not a trading result.
- Artifacts: `D:\futures-ml-data\artifacts\models\phase2_2`; declarations: `reports/phase2_2_fold_definitions.{md,json}`; compact metrics: `reports/phase2_2_walk_forward_metrics.json`; report: `reports/phase2_2_walk_forward_stability_report.md`.
- Known limitations: all folds reuse the same historical five-year dataset, the relationship remains small, pooled results are dominated by stronger roots, and there is no fee, slippage, execution, backtest, or profitability analysis.
- Exact next recommended task: if new authority and genuinely new forward data or a newly reserved future holdout become available, predeclare a dedicated A-vs-B replication for ZN/ZC/ZB to separate market specialization from anchor-feature value. Do not begin strategy/backtest/live work without a new explicit phase request.

Phase 2.3 handoff (2026-10-13):

- Purpose: compare dedicated A vs B independently for the Phase 2.2-preselected ZN, ZC, and ZB roots using only genuinely new timestamps. This is model confirmation, not strategy research.
- Historical cutoff: `2026-09-23T00:00:00Z`. Training timestamps are strictly earlier; prospective timestamps must be strictly later. No Phase 2.2 fold may be relabeled as prospective evidence.
- Frozen candidates: `ZN-A-final`, `ZN-B-final`, `ZC-A-final`, `ZC-B-final`, `ZB-A-final`, and `ZB-B-final`. A has the unchanged 35 own-market features; B adds the unchanged 18 raw exact-time anchors. Relative C features are absent.
- Every candidate uses 200,000 deterministic eligible rows from its root and the frozen HGB settings: learning rate `0.08`, 150 iterations, 31 leaves, minimum leaf 100, L2 `0.1`, seed 42. Dedicated models omit root encoding.
- Eligible historical counts before sampling are ZN `1,584,770`, ZC `909,854`, and ZB `1,396,266`. A and B use identical selected-row identities within each root.
- The JSON/Markdown freeze manifests are `reports/phase2_3_model_freeze_manifest.{json,md}`. They record full feature order, training spans/counts, selected-row hashes, imputer medians, historical robust feature profiles, frozen volatility thresholds, pipeline hashes, artifact paths, creation time, versions, and cutoff.
- External model artifacts are under `D:\futures-ml-data\artifacts\models\phase2_3`. Earlier Phase 2/2.1/2.2 trees were not overwritten.
- The prospective evaluator verifies frozen hashes before use, never calls `fit`, reads prediction inputs without labels, attaches exact outcomes separately, and writes new immutable parts under `D:\futures-ml-data\artifacts\prospective\phase2_3\prediction_log`.
- Checkpoints were frozen before the local prospective scan: 7/30/60 days are monitoring only; 90 calendar days is primary. Decisions are per root. At the primary review, clear B improvement may retain B; ties or A wins prefer simpler A.
- Current local post-cutoff status: ZN `0`, ZC `0`, ZB `0` source rows. `reports/phase2_3_prospective_metrics.json` contains the empty result schema and `reports/phase2_3_dedicated_prospective_report.md` states `AWAITING PROSPECTIVE DATA`. No metric was invented and no paid data was purchased/downloaded.
- No retraining, imputer/threshold update, model selection, or drift-triggered adaptation is allowed during confirmation. Direction agreement, deciles, volatility regimes, and feature drift are diagnostics only.
- Exact next checkpoint: after explicit data-acquisition approval, build genuinely post-cutoff data through the unchanged pipeline and run the frozen evaluator. Do not begin strategy, P&L, fee, risk, or execution work automatically.

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
