# Databento Alternative Plan Estimates

The original plan failed both budget and storage gates. These alternatives are estimates for review only. No plan is authorized for download without explicit approval.

Budget limit: `$120.00`
Available disk: `298.04 GB`

| Priority | Plan | Cost USD | Raw GB | Working GB | Required with headroom GB | Budget | Storage |
|---:|---|---:|---:|---:|---:|:---:|:---:|
| 1 | A1: 5y OHLCV-1s + 2y MBP-1 (6 markets) | $3,505.09 | 1,011.32 | 3,033.97 | 3,792.46 | FAIL | FAIL |
| 2 | A2: 5y OHLCV-1s + 1y MBP-1 (6 markets) | $2,829.52 | 608.33 | 1,824.99 | 2,281.23 | FAIL | FAIL |
| 3 | A3: 5y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC) | $2,703.18 | 532.96 | 1,598.89 | 1,998.61 | FAIL | FAIL |
| 4 | A4: 4y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC) | $2,305.31 | 526.86 | 1,580.58 | 1,975.73 | FAIL | FAIL |
| 5 | A5: 3y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC) | $1,915.73 | 520.88 | 1,562.65 | 1,953.32 | FAIL | FAIL |
| 6 | B1: 5y OHLCV-1m (26 markets), metadata, no MBP-1 | $174.17 | 21.42 | 64.25 | 80.32 | FAIL | PASS |
| 7 | B2: 4y OHLCV-1m (26 markets), metadata, no MBP-1 | $144.91 | 20.97 | 62.91 | 78.63 | FAIL | PASS |
| 8 | B3: 3y OHLCV-1m (26 markets), metadata, no MBP-1 | $115.92 | 20.52 | 61.57 | 76.97 | PASS | PASS |

## Details

### A1: 5y OHLCV-1s + 2y MBP-1 (6 markets)

First prescribed reduction: retain universal five-year 1-second bars and shorten MBP-1 to two years.

### A2: 5y OHLCV-1s + 1y MBP-1 (6 markets)

Second prescribed reduction: retain universal five-year 1-second bars and shorten MBP-1 to one year.

### A3: 5y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)

Third prescribed reduction: retain five-year 1-second bars and reduce the one-year MBP universe to four markets.

### A4: 4y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)

Fourth prescribed reduction: shorten universal 1-second history to four years after reducing MBP.

### A5: 3y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)

Fifth prescribed reduction: shorten universal 1-second history to three years after reducing MBP.

### B1: 5y OHLCV-1m (26 markets), metadata, no MBP-1

V1-oriented candidate: buy the one-minute resolution used by the first model and postpone all MBP-1. This changes the approved source granularity and requires explicit user approval.

### B2: 4y OHLCV-1m (26 markets), metadata, no MBP-1

V1-oriented candidate using the one-minute resolution consumed by the first model, retaining five years of metadata, and postponing MBP-1. This changes the approved source granularity and requires explicit user approval.

### B3: 3y OHLCV-1m (26 markets), metadata, no MBP-1

V1-oriented candidate using the one-minute resolution consumed by the first model, retaining five years of metadata, and postponing MBP-1. This changes the approved source granularity and requires explicit user approval.
