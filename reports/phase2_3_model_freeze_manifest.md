# Phase 2.3 Model Freeze Manifest

**Status: FROZEN_AWAITING_PROSPECTIVE_DATA**

Frozen at UTC: `2026-10-13T05:00:31.707464+00:00`

Historical cutoff: `2026-09-23T00:00:00+00:00`

Prospective rule: timestamps must be strictly greater than historical_cutoff

The six candidates below were trained and hashed before any post-cutoff performance was examined. Phase 2, 2.1, and 2.2 artifacts were not overwritten.

| Model | Root | Experiment | Features | Training span | Rows | Selected-row hash | Pipeline hash |
|---|---|---|---:|---|---:|---|---|
| ZN-A-final | ZN | A_1m | 35 | `2021-09-23T04:32:00+00:00` through `2026-09-22T23:54:00+00:00` | 200,000 | `a617ccebc89d5252...` | `441a3b4b7df53d9c...` |
| ZN-B-final | ZN | B_1m | 53 | `2021-09-23T04:32:00+00:00` through `2026-09-22T23:54:00+00:00` | 200,000 | `a617ccebc89d5252...` | `9f7ff77ef3af399f...` |
| ZC-A-final | ZC | A_1m | 35 | `2021-09-23T05:23:00+00:00` through `2026-09-22T18:18:00+00:00` | 200,000 | `691b895c2c22311e...` | `6b4c8f64d1d3e81d...` |
| ZC-B-final | ZC | B_1m | 53 | `2021-09-23T05:23:00+00:00` through `2026-09-22T18:18:00+00:00` | 200,000 | `691b895c2c22311e...` | `b1348535beccbe8a...` |
| ZB-A-final | ZB | A_1m | 35 | `2021-09-23T05:30:00+00:00` through `2026-09-22T23:52:00+00:00` | 200,000 | `d2f8e6495ce30bce...` | `d529db8c0fcb6e24...` |
| ZB-B-final | ZB | B_1m | 53 | `2021-09-23T05:30:00+00:00` through `2026-09-22T23:52:00+00:00` | 200,000 | `d2f8e6495ce30bce...` | `c4cb378202fca4f5...` |

## Frozen model contract

- Target: exact next-real-minute, same-root, same-contract-segment `future_return_1m`.
- A: the unchanged 35 own-market features.
- B: A plus the unchanged 18 raw exact-UTC anchor returns.
- Dedicated HGB: learning rate 0.08, 150 iterations, 31 leaves, minimum leaf 100, L2 0.1, seed 42.
- Training: 200,000 deterministic evenly spaced eligible observations per target root, all strictly before the cutoff.
- Preprocessing: per-model historical median imputation only. Imputer medians and full historical robust feature profiles are stored in model metadata and this JSON manifest.
- Frozen volatility thresholds: per-root tertiles from all eligible historical rows strictly before the cutoff.
- Immutability: pipelines, preprocessing, features, samples, and parameters cannot change during prospective confirmation.

## Predeclared checkpoints

| Days | Role | Start exclusive | End exclusive |
|---:|---|---|---|
| 7 | monitoring | `2026-09-23T00:00:00+00:00` | `2026-09-30T00:00:00+00:00` |
| 30 | monitoring | `2026-09-23T00:00:00+00:00` | `2026-10-23T00:00:00+00:00` |
| 60 | monitoring | `2026-09-23T00:00:00+00:00` | `2026-11-22T00:00:00+00:00` |
| 90 | primary | `2026-09-23T00:00:00+00:00` | `2026-12-22T00:00:00+00:00` |

The 7/30/60-day checkpoints are monitoring diagnostics. Only the predeclared 90-day checkpoint is primary. No winner is selected automatically.

## External artifacts

- `ZN-A-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZN\A`
- `ZN-B-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZN\B`
- `ZC-A-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZC\A`
- `ZC-B-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZC\B`
- `ZB-A-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZB\A`
- `ZB-B-final`: `D:\futures-ml-data\artifacts\models\phase2_3\ZB\B`
