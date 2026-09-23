# Databento Data Cost and Storage Report

Generated: `2026-09-23T04:26:51.122965+00:00`
Dataset: `GLBX.MDP3`
Latest complete end boundary (exclusive, UTC): `2026-09-22`

## Decision

**Download status: BLOCKED**

- Budget: `$4,162.99` estimated / `$120.00` limit — FAIL
- Storage: `298.04 GB` free / `5,264.16 GB` required with headroom — FAIL

No download is performed by the estimator. A downloader may proceed only when this report says AUTHORIZED.

## Plan totals

| Plan component | Records | Billable raw GB | Cost USD |
|---|---:|---:|---:|
| OHLCV-1s | 509,515,043 | 28.533 | $1,860.1296 |
| Definitions | 34,504,157 | 17.942 | $28.4069 |
| Statistics | 15,559,008 | 1.245 | $1.1592 |
| Status | 337,999 | 0.014 | $0.0504 |
| MBP-1 | 16,950,540,959 | 1,356.043 | $2,273.2447 |
| **GRAND TOTAL** | **17,510,457,166** | **1,403.777** | **$4,162.9908** |

## Storage

- Configured root: `D:\futures-ml-data`
- Volume: `D:`
- Total capacity: `1,000.20 GB`
- Available capacity: `298.04 GB`
- Billable raw binary size: `1,403.777 GB`
- Expected working space: `4,211.330 GB` (3.00x raw)
- Required free space with safety margin: `5,264.162 GB` (1.25x working)

The API's billable size is uncompressed raw binary used for billing, not a promise of final compressed download size. The working estimate reserves additional room for immutable raw files, processed Parquet, features, and temporary intermediates.

## Detailed requests

See `data_cost_report.csv` or `data_cost_report.json` for every symbol/schema request.

## Methodology notes

- Estimates come from Databento `get_cost`, `get_record_count`, and `get_billable_size` metadata calls.
- Date ranges are midnight-to-midnight UTC and the end is exclusive.
- OHLCV-1s, statistics, status, and MBP-1 use volume-ranked continuous symbols such as `ES.v.0`.
- Definitions use parent symbols such as `ES.FUT` to cover underlying contracts.
- Databento notes that estimates may over-report ranges not divisible by ten minutes; definition estimates are accurate only for whole-day ranges. This plan uses whole days.
