# Phase 2.3 Dedicated Prospective Confirmation

**Status: AWAITING PROSPECTIVE DATA**

## Objective

Determine separately for ZN, ZC, and ZB whether the dedicated one-minute signal survives genuinely new data and whether B's 18 raw anchor returns add enough repeatable value to justify their complexity. This is frozen-model confirmation, not model improvement or strategy research.

## Why ZN/ZC/ZB were preselected

Phase 2.2 preselected these roots before Phase 2.3: dedicated B beat pooled B on each root in all 8/8 historical folds. No root was added after inspecting prospective performance.

## Frozen A definition

Each dedicated A model uses exactly the original 35 own-market numeric features.

## Frozen B definition

Each dedicated B model uses the same 35 features plus 18 raw exact-time NQ/ES/ZN/CL/GC/6E returns at 1/5/15 minutes. Relative C features, new anchors, spreads, horizons, and indicators are absent.

## Historical cutoff

The frozen cutoff is `2026-09-23T00:00:00+00:00`. Training rows are strictly earlier; prospective rows must be strictly later. Old Phase 2.2 folds are never relabeled as prospective evidence.

## Six frozen model IDs

`ZN-A-final`, `ZN-B-final`, `ZC-A-final`, `ZC-B-final`, `ZB-A-final`, `ZB-B-final`

## Training counts

| Model | Selected rows | Eligible historical rows | Last selected timestamp |
|---|---:|---:|---|
| ZN-A-final | 200,000 | 1,584,770 | `2026-09-22T23:54:00+00:00` |
| ZN-B-final | 200,000 | 1,584,770 | `2026-09-22T23:54:00+00:00` |
| ZC-A-final | 200,000 | 909,854 | `2026-09-22T18:18:00+00:00` |
| ZC-B-final | 200,000 | 909,854 | `2026-09-22T18:18:00+00:00` |
| ZB-A-final | 200,000 | 1,396,266 | `2026-09-22T23:52:00+00:00` |
| ZB-B-final | 200,000 | 1,396,266 | `2026-09-22T23:52:00+00:00` |

## Freeze hashes

| Model | Logical selected-row hash | Pipeline SHA-256 |
|---|---|---|
| ZN-A-final | `a617ccebc89d525263279b0fc801c9234a2704fdfd3d865b8e917f11abebf783` | `441a3b4b7df53d9c6b97e04aba7a2114656143a9708d375f1adce0a053a6b4ae` |
| ZN-B-final | `a617ccebc89d525263279b0fc801c9234a2704fdfd3d865b8e917f11abebf783` | `9f7ff77ef3af399f2317af463ecdf0a995da7ad16c80908e84d3b5e9e18ba86b` |
| ZC-A-final | `691b895c2c22311e0559533baaa62574f06c4fd019d384377822d094e3483713` | `6b4c8f64d1d3e81dba55f786837bc55f13e4fd072e2085923b1059c9c2c316ef` |
| ZC-B-final | `691b895c2c22311e0559533baaa62574f06c4fd019d384377822d094e3483713` | `b1348535beccbe8ab7ab00fadf0d723128bb916be3f45948f300f8f50563c4b9` |
| ZB-A-final | `d2f8e6495ce30bced3b4498149c1f7256a2e30457b794bf5a8afbd572e742c79` | `d529db8c0fcb6e2460da0b849601e82c96528d1c771da07b9637b880e9f722e1` |
| ZB-B-final | `d2f8e6495ce30bced3b4498149c1f7256a2e30457b794bf5a8afbd572e742c79` | `c4cb378202fca4f5935177f1f4b6f2421e19951f2397f3aa1d4a903ad8fd1447` |

## Prospective evaluation protocol

Predictions are generated from metadata and causal feature columns only. Outcomes are read and attached in a separate exact-identity step after prediction events exist. The external log is append-only; existing parts are never rewritten. Pipelines and preprocessing hashes are verified before every evaluation run, and no fit operation occurs in the evaluator.

A and B are compared independently within each root using N, MAE, RMSE, R², Pearson, Spearman, and diagnostic direction agreement. B-A Pearson, Spearman, and R² are reported. Ties favor simpler A only at the primary checkpoint and only after research review.

## Predeclared checkpoints

| Days | Role | End exclusive |
|---:|---|---|
| 7 | monitoring | `2026-09-30T00:00:00+00:00` |
| 30 | monitoring | `2026-10-23T00:00:00+00:00` |
| 60 | monitoring | `2026-11-22T00:00:00+00:00` |
| 90 | primary | `2026-12-22T00:00:00+00:00` |

The 7/30/60-day windows are monitoring only. The 90-day window is the primary checkpoint; short-window results cannot choose a winner.

## Leakage safeguards

- Exact t+1-minute, same-contract-segment target; no next-row substitution or rollover crossing.
- Exact-UTC anchor joins without as-of matching or forward filling.
- Historical timestamps are strictly before the cutoff; prospective timestamps are strictly after it.
- Prediction inputs exclude every future-return label.
- Outcome attachment is a separate exact timestamp/root/instrument/segment join.
- Frozen model and selected-row hashes are checked before evaluation.
- No retraining, imputer update, feature update, tuning, or automatic drift response is permitted.

## Current amount of new data available

Post-cutoff source rows: **0**. By root: ZN=0, ZC=0, ZB=0.

## AWAITING PROSPECTIVE DATA

Models frozen; prospective evaluator ready; awaiting genuinely new data.

No historical observation was reused, no metric was invented, and no paid data was purchased or downloaded.

## Interpretation

No A/B selection is currently authorized. The models remain immutable until a completed predeclared checkpoint is reviewed; the primary decision belongs to the 90-day checkpoint.

## Artifact locations

- Freeze manifest: `D:\futures-ml-data\artifacts\models\phase2_3\model_freeze_manifest.json`
- Model tree: `D:\futures-ml-data\artifacts\models\phase2_3`
- Append-only prediction log: `D:\futures-ml-data\artifacts\prospective\phase2_3\prediction_log`
- Repository metrics: `reports/phase2_3_prospective_metrics.json`

## Next checkpoint

Acquire genuinely post-cutoff data only after explicit approval, build it through the unchanged causal feature pipeline, and rerun the prospective evaluator without retraining. Stop before strategy, P&L, cost, risk, or execution work.
