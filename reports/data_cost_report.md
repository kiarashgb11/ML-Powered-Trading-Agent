# Databento Data Cost and Storage Report

Generated: `2026-09-23T14:41:16.368436+00:00`
Dataset: `GLBX.MDP3`
Latest complete end boundary (exclusive, UTC): `2026-09-23`

## Decision

**Download status: AUTHORIZED**

- Budget: `$111.16` estimated / `$120.00` limit — PASS
- Storage: `708.37 GB` free / `51.76 GB` required with headroom — PASS

No download is performed by the estimator. A downloader may proceed only when this report says AUTHORIZED.

## Plan totals

| Plan component | Records | Billable raw GB | Cost USD |
|---|---:|---:|---:|
| OHLCV-1m | 25,205,190 | 1.411 | $92.0187 |
| Definitions | 22,353,264 | 11.624 | $18.4032 |
| Statistics | 9,494,675 | 0.760 | $0.7074 |
| Status | 206,016 | 0.008 | $0.0307 |
| **GRAND TOTAL** | **57,259,145** | **13.803** | **$111.1600** |

## Storage

- Configured root: `D:\futures-ml-data`
- Volume: `D:`
- Total capacity: `1,000.20 GB`
- Available capacity: `708.37 GB`
- Billable raw binary size: `13.803 GB`
- Expected working space: `41.409 GB` (3.00x raw)
- Required free space with safety margin: `51.761 GB` (1.25x working)

The API's billable size is uncompressed raw binary used for billing, not a promise of final compressed download size. The working estimate reserves additional room for immutable raw files, processed Parquet, features, and temporary intermediates.

## Detailed requests

See `data_cost_report.csv` or `data_cost_report.json` for every symbol/schema request.

## Methodology notes

- Estimates come from Databento `get_cost`, `get_record_count`, and `get_billable_size` metadata calls.
- Date ranges are midnight-to-midnight UTC and the end is exclusive.
- Non-definition requests use volume-ranked continuous symbols such as `ES.v.0`.
- Definitions use parent symbols such as `ES.FUT` to cover underlying contracts.
- Databento notes that estimates may over-report ranges not divisible by ten minutes; definition estimates are accurate only for whole-day ranges. This plan uses whole days.
