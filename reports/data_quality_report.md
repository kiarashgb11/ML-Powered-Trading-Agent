# Historical Data Quality Report

**Overall validation status: PASS**

- Data root: `D:\futures-ml-data`
- Requested range: `2021-09-23` inclusive to `2026-09-23` exclusive UTC
- Validated partitions: `64/64`
- Actual downloaded disk usage: `0.807 GB`
- Remaining D: free space: `707.556 GB`
- Failed or incomplete partitions: `0`

## Root summary

| Root | Schemas | Records | Disk GB | Actual coverage | Invalid OHLC | Negative volume | Duplicate bars | Suspicious >4-day gaps | Roll transitions | Metadata coverage | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|:---:|
| NQ | 4/4 | 4,243,972 | 0.084 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| ES | 4/4 | 2,651,504 | 0.045 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| RTY | 4/4 | 2,797,078 | 0.048 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| ZF | 4/4 | 1,838,063 | 0.023 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| ZN | 4/4 | 1,873,553 | 0.023 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:58:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| ZB | 4/4 | 1,722,889 | 0.021 | 2021-09-23T00:01:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| CL | 4/4 | 8,578,641 | 0.118 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 60 | complete | PASS |
| NG | 4/4 | 11,678,422 | 0.184 | 2021-09-23T00:02:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 62 | complete | PASS |
| GC | 4/4 | 3,815,663 | 0.050 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 25 | complete | PASS |
| HG | 4/4 | 4,921,006 | 0.045 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 25 | complete | PASS |
| 6E | 4/4 | 2,499,682 | 0.036 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| 6J | 4/4 | 2,424,753 | 0.033 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:59:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| 6B | 4/4 | 2,260,737 | 0.030 | 2021-09-23T00:00:00+00:00 to 2026-09-22T23:58:00+00:00 | 0 | 0 | 0 | 0 | 20 | complete | PASS |
| ZC | 4/4 | 1,807,991 | 0.020 | 2021-09-23T00:00:00+00:00 to 2026-09-22T18:19:00+00:00 | 0 | 0 | 0 | 0 | 32 | complete | PASS |
| ZS | 4/4 | 2,391,766 | 0.027 | 2021-09-23T00:00:00+00:00 to 2026-09-22T18:19:00+00:00 | 0 | 0 | 0 | 0 | 25 | complete | PASS |
| ZW | 4/4 | 1,753,425 | 0.021 | 2021-09-23T00:00:00+00:00 to 2026-09-22T18:19:00+00:00 | 0 | 0 | 0 | 0 | 31 | complete | PASS |

## Validation interpretation

- Every downloaded file was matched to Databento's reported size and checksum, then opened as DBN and streamed to count records.
- OHLC checks require `high >= open/close/low`, `low <= open/close`, nonnegative volume, and unique `(timestamp, instrument_id)` bars.
- Roll transitions count changes in `instrument_id` within the volume-ranked continuous OHLCV series. The raw series is not back-adjusted.
- Suspicious gaps are OHLCV timestamp jumps longer than four days. Missing one-minute bars are otherwise not treated as errors because bars require qualifying trades.
- Duplicate-bar checks apply to OHLCV. Definition, statistics, and status schemas may legitimately contain multiple distinct events for one timestamp and instrument.
- Metadata coverage requires nonzero Definition, Statistics, and Status records for every root.
- Coverage in the root table is OHLCV event-time coverage. Definition snapshots can include instruments whose original event timestamps predate the requested start boundary.

## Failed or incomplete partitions

None.
