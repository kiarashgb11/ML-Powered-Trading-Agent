# BASE-5Y Optional Market Add-on Comparison

Metadata estimates only. BASE-5Y is the intended base, but neither it nor any add-on combination is authorized for download by this report.

- Dataset: `GLBX.MDP3`
- End boundary: `2026-09-22` exclusive UTC
- Intended base roots (15): NQ, ES, RTY, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, ZW
- Optional roots: ZF, 6C, 6A
- Budget threshold: `$120.00`
- Current D: free: `708.37 GB` decimal (`659.72 GiB`)

## Intended base

BASE-5Y is estimated at `$105.24`, `13.68 GB` billable raw, and `51.28 GB` required with headroom.

## Individual add-on estimates

Each root includes five years of OHLCV-1m, Definitions, Statistics, and Status.

| Root | Add-on cost USD | Records | Raw GB |
|---|---:|---:|---:|
| ZF | $5.91 | 1,838,022 | 0.125 |
| 6C | $6.04 | 2,206,039 | 0.220 |
| 6A | $6.50 | 2,429,509 | 0.235 |

## BASE-5Y plus add-on combinations

| Add-on roots | Add-on cost USD | Combined cost USD | Budget remaining USD | Combined raw GB | Required with headroom GB | Budget | Storage |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| ZF | $5.91 | $111.16 | $8.84 | 13.801 | 51.75 | PASS | PASS |
| 6C | $6.04 | $111.29 | $8.71 | 13.896 | 52.11 | PASS | PASS |
| 6A | $6.50 | $111.74 | $8.26 | 13.911 | 52.17 | PASS | PASS |
| ZF + 6C | $11.96 | $117.20 | $2.80 | 14.021 | 52.58 | PASS | PASS |
| ZF + 6A | $12.41 | $117.66 | $2.34 | 14.036 | 52.63 | PASS | PASS |
| 6C + 6A | $12.54 | $117.78 | $2.22 | 14.131 | 52.99 | PASS | PASS |
| ZF + 6C + 6A | $18.45 | $123.70 | $-3.70 | 14.256 | 53.46 | FAIL | PASS |

## Result

Largest fitting combination(s): ZF + 6C, ZF + 6A, 6C + 6A.

All combinations remain review-only. Refresh the estimate immediately before any download.
