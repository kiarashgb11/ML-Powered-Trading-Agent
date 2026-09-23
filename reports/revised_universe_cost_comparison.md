# Revised Universe Databento Cost Comparison

Metadata estimates only. No market data was downloaded, no batch job was submitted, and no scenario is authorized.

- Dataset: `GLBX.MDP3`
- End boundary: `2026-09-22` exclusive UTC
- Revised universe (15): NQ, ES, RTY, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, ZW
- MBP universe (4): NQ, ES, CL, GC
- Budget threshold: `$120.00`
- D: free: `710.83 GB` decimal (`662.01 GiB`)

## OHLCV-1m plus Definitions, Statistics, and Status

Each metadata schema uses the same horizon as its OHLCV plan.

| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| BASE-5Y | $105.24 | 55,416,886 | 13.68 | 41.03 | 51.28 | PASS | PASS |
| BASE-4Y | $84.29 | 44,668,232 | 11.21 | 33.62 | 42.03 | PASS | PASS |

## Standalone MBP-1 estimates for NQ/ES/CL/GC

| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| MBP-1M | $58.56 | 436,628,587 | 34.93 | 104.79 | 130.99 | PASS | PASS |
| MBP-3M | $196.11 | 1,462,309,322 | 116.98 | 350.95 | 438.69 | FAIL | PASS |
| MBP-6M | $405.51 | 3,023,704,948 | 241.90 | 725.69 | 907.11 | FAIL | FAIL |

## Combined comparison

These rows are arithmetic combinations of the independently estimated base and MBP plans.

| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| BASE-5Y+MBP-1M | $163.80 | 492,045,473 | 48.61 | 145.82 | 182.27 | FAIL | PASS |
| BASE-5Y+MBP-3M | $301.36 | 1,517,726,208 | 130.66 | 391.98 | 489.98 | FAIL | PASS |
| BASE-5Y+MBP-6M | $510.75 | 3,079,121,834 | 255.57 | 766.72 | 958.40 | FAIL | FAIL |
| BASE-4Y+MBP-1M | $142.84 | 481,296,819 | 46.14 | 138.41 | 173.02 | FAIL | PASS |
| BASE-4Y+MBP-3M | $280.40 | 1,506,977,554 | 128.19 | 384.58 | 480.72 | FAIL | PASS |
| BASE-4Y+MBP-6M | $489.80 | 3,068,373,180 | 253.10 | 759.31 | 949.14 | FAIL | FAIL |

## Interpretation

- `PASS` only means the current estimate is within the configured budget or capacity gate.
- All scenarios remain unauthorized until the user selects one explicitly.
- Cost should be refreshed immediately before any acquisition because the budget margin may be small.
- Billable raw size is not the same as final compressed download size.
