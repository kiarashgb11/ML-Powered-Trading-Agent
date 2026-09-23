# Project Mission

Build a careful, reproducible futures ML research platform. The immediate scientific question is whether historical market information contains repeatable out-of-sample signal for future returns. This is not currently a production or live trading bot.

# Current Phase

Phase 1: repository setup, external-storage safety, live Databento cost/size/count estimation, and only then historical acquisition.

Current handoff (2026-09-23):

- Complete: initial package/config/test skeleton; portable path resolver; authenticated original-plan and alternative-plan Databento estimators; current dataset, schemas, API signatures, and availability verified. Generated CSV/JSON/Markdown reports are in `reports/`.
- Files added/changed in this milestone: cost reports, `scripts/estimate_alternative_plans.py`, `src/futures_ml/data/alternative_estimator.py`, README workflow, and lint configuration.
- Storage snapshot: Windows `D:` is a ready fixed NTFS volume with about 931.51 GiB total and 277.57 GiB (298.04 decimal GB) free. `D:\futures-ml-data` exists and is empty.
- Dataset/download status: no market data downloaded, no batch job submitted, and no purchase made.
- Latest experiment: none.
- Original-plan result: USD 4,162.99; 1,403.78 GB billable raw; 17.51 billion records; about 4.21 TB expected working space. Budget and storage both fail.
- Prescribed 1-second reductions: all fail; the smallest remains USD 1,915.73 and requires about 1.95 TB free with headroom.
- First passing candidate: **B3**, three years of `ohlcv-1m` for all 26 markets, five years of definitions/statistics/status, and no MBP-1. Estimate: USD 115.92; 20.52 GB billable raw; 61.57 GB expected working space; 76.97 GB required with headroom. Budget and storage pass.
- Latest review-only comparison: revised 15-root universe NQ, ES, RTY, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, ZW. Five-year OHLCV-1m plus same-horizon Definitions/Statistics/Status is USD 105.24 and requires 51.28 GB free with headroom. Four-year is USD 84.29 and requires 42.03 GB. Both pass independently.
- Standalone MBP comparison for NQ/ES/CL/GC: one month is USD 58.56 and requires 130.99 GB with headroom; three months is USD 196.11 and requires 438.69 GB; six months is USD 405.51 and requires 907.11 GB. Only one month passes the budget independently; six months also fails current storage.
- Combined result: no requested base-plus-MBP combination passes USD 120. The cheapest is four-year base plus one-month MBP at USD 142.84. Current D: free space is 710.83 decimal GB (662.01 GiB).
- Intended base decision: BASE-5Y is the intended plan—five years of OHLCV-1m plus Definitions/Statistics/Status for NQ, ES, RTY, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, and ZW. Estimate: USD 105.24. This intent is recorded in `config/intended_plan.yaml`, but download authorization remains false.
- Optional five-year add-ons: ZF costs USD 5.91, 6C costs USD 6.04, and 6A costs USD 6.50. Every single and pair fits the USD 120 threshold. All three together cost USD 123.70 and fail.
- Largest fitting choices: ZF+6C at USD 117.20, ZF+6A at USD 117.66, or 6C+6A at USD 117.78. Their remaining budget margins are USD 2.80, USD 2.34, and USD 2.22 respectively; all pass storage.
- Unresolved decision: optional add-on selection is pending. No plan is authorized for download.
- Exact next task: user selects one pair, one root, or no add-ons. Then update `config/intended_plan.yaml`, refresh the complete selected estimate immediately, and request explicit download approval before implementing or running acquisition.

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

Fixed 26 roots: NQ, ES, RTY, YM, ZT, ZF, ZN, ZB, CL, NG, RB, HO, GC, SI, HG, 6E, 6J, 6B, 6A, 6C, 6S, ZC, ZW, ZS, ZM, ZL.

Core MBP markets: NQ, ES, CL, GC, ZN, 6E.

# Data Horizons

- `ohlcv-1s`: approximately five years for all 26 roots.
- `mbp-1`: approximately three years for the six core roots.
- Also estimate/acquire `definition`, `statistics`, and `status` as planned.
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
