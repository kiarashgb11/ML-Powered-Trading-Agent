# Phase 2.1 One-Minute Return Prediction and Cross-Market Ablation

**Status: COMPLETE. Phase 2 artifacts were preserved; no data was downloaded or purchased.**

## Objective

Test whether the original 35 causal own-market features predict the next exact one-minute return, and whether 18 exact-UTC cross-market returns help when represented raw (B-1m) or relative to the target market (C-1m). This is predictive-signal research only; no strategy, P&L, costs, execution, or live integration was implemented.

## Exact target

`future_return_1m[t] = ln(close[t+1 real minute] / close[t])`. The lookup requires the same root, the same contiguous `contract_segment_id`, and exactly `t+1 minute`. A missing minute produces a null target; the next available row is never substituted. Labels never enter `X`.

## Data and splits

The run reused the immutable Phase 2 `experiment_b` Parquet partitions as a common source. A and B are column projections; C is derived in memory as target-own return minus the exact-time anchor return. No Phase 2 feature file was changed.

- Roots: NQ, ES, RTY, ZF, ZN, ZB, CL, NG, GC, HG, 6E, 6J, 6B, ZC, ZS, ZW.
- Train/validation boundary: `2025-03-24T04:47:00+00:00`.
- Validation/test boundary: `2025-12-23T02:23:00+00:00`.
- Purge: 75 minutes on both sides of both boundaries.
- Eligible training observations before the cap: 16,914,241.
- Primary fit: 1,600,000 observations, exactly 100,000 deterministic evenly spaced rows per root.
- Full validation: 3,519,199 observations; untouched test: 3,551,762 observations.
- Preprocessing: train-median imputation for both model families; train-only scaling for Ridge; no numeric scaling for HGB; pooled root one-hot encoding.

## Experiment A-1m

A-1m uses the unchanged 35 own-market inputs and root identity. It asks whether a market's own current and historical OHLCV state predicts its next exact minute.

## Experiment B-1m

B-1m uses the same 35 inputs plus 18 raw exact-UTC returns: NQ, ES, ZN, CL, GC, and 6E at 1/5/15-minute lookbacks. The old `NQ_ES_return_spread_5m` is excluded, leaving exactly 53 numeric features.

## Experiment C-1m

C-1m uses the same 35 inputs plus 18 `own_return_h - anchor_return_h` features. Raw anchors are not simultaneously included. A valid self-anchor is exactly zero; if either operand is unavailable, the relative feature remains null. Because own return plus own-minus-anchor can reconstruct the anchor, C changes representation rather than adding information.

## Training sample counts

| Root | Eligible train | Primary fit | 50k study | 100k study | 200k study |
|---|---:|---:|---:|---:|---:|
| NQ | 1,233,143 | 100,000 | 50,000 | 100,000 | 200,000 |
| ES | 1,232,421 | 100,000 | 50,000 | 100,000 | 200,000 |
| RTY | 1,164,147 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZF | 1,078,197 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZN | 1,129,968 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZB | 1,007,580 | 100,000 | 50,000 | 100,000 | 200,000 |
| CL | 1,200,149 | 100,000 | 50,000 | 100,000 | 200,000 |
| NG | 1,005,326 | 100,000 | 50,000 | 100,000 | 200,000 |
| GC | 1,208,796 | 100,000 | 50,000 | 100,000 | 200,000 |
| HG | 1,150,931 | 100,000 | 50,000 | 100,000 | 200,000 |
| 6E | 1,192,353 | 100,000 | 50,000 | 100,000 | 200,000 |
| 6J | 1,190,782 | 100,000 | 50,000 | 100,000 | 200,000 |
| 6B | 1,083,080 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZC | 650,508 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZS | 767,354 | 100,000 | 50,000 | 100,000 | 200,000 |
| ZW | 619,506 | 100,000 | 50,000 | 100,000 | 200,000 |

A/B/C row identities are inherited from one common sampled source, so the primary comparison uses identical root/timestamp observations.

## Validation results

| Experiment | Target | Input set | Ridge R² | Ridge Pearson | HGB R² | HGB Pearson | HGB Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| A-1m | `future_return_1m` | 35 own-market features | 0.00216317 | 0.0479948 | 0.0046063 | 0.0788151 | 0.111112 |
| B-1m | `future_return_1m` | 35 own + 18 raw anchor returns | 0.00271929 | 0.0535335 | 0.00462148 | 0.0791122 | 0.110289 |
| C-1m | `future_return_1m` | 35 own + 18 target-relative returns | 0.00277205 | 0.0538631 | 0.00456401 | 0.0788208 | 0.109802 |

Macro-average validation metrics (each root receives equal weight):

| Experiment | Model | Macro MAE | Macro RMSE | Macro R² | Macro Pearson | Macro Spearman | Macro direction accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| A-1m | Ridge | 0.000233029 | 0.000382885 | 0.00878608 | 0.0828193 | 0.115677 | 40.145% |
| A-1m | HGB | 0.000231989 | 0.000382431 | 0.00967786 | 0.092223 | 0.120583 | 40.093% |
| B-1m | Ridge | 0.0002335 | 0.000382881 | 0.00664245 | 0.0805036 | 0.114375 | 40.113% |
| B-1m | HGB | 0.00023197 | 0.000382442 | 0.00949654 | 0.0923118 | 0.120706 | 40.082% |
| C-1m | Ridge | 0.00023347 | 0.000382883 | 0.00653371 | 0.0802476 | 0.113672 | 40.097% |
| C-1m | HGB | 0.000231941 | 0.000382446 | 0.00953242 | 0.0916531 | 0.119425 | 40.085% |

B-1m HGB had the highest primary pooled validation Pearson; C-1m Ridge had the highest capped Ridge validation Pearson. These choices were made before test evaluation.

## Test results

| Experiment | Target | Input set | Ridge R² | Ridge Pearson | HGB R² | HGB Pearson | HGB Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| A-1m | `future_return_1m` | 35 own-market features | 0.00252603 | 0.0505739 | 0.00371347 | 0.069236 | 0.115111 |
| B-1m | `future_return_1m` | 35 own + 18 raw anchor returns | 0.00227143 | 0.0493717 | 0.00370446 | 0.0694166 | 0.114225 |
| C-1m | `future_return_1m` | 35 own + 18 target-relative returns | 0.00221086 | 0.0487958 | 0.00364189 | 0.0690091 | 0.114085 |

Macro-average test metrics (each root receives equal weight):

| Experiment | Model | Macro MAE | Macro RMSE | Macro R² | Macro Pearson | Macro Spearman | Macro direction accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| A-1m | Ridge | 0.000254062 | 0.000404037 | 0.0113516 | 0.0961591 | 0.124333 | 40.499% |
| A-1m | HGB | 0.000253155 | 0.000403783 | 0.0112737 | 0.100199 | 0.129949 | 40.494% |
| B-1m | Ridge | 0.000254715 | 0.000404266 | 0.00634975 | 0.0860031 | 0.119061 | 40.342% |
| B-1m | HGB | 0.000253134 | 0.000403798 | 0.0109828 | 0.0998878 | 0.129988 | 40.508% |
| C-1m | Ridge | 0.00025467 | 0.000404273 | 0.00643625 | 0.0857455 | 0.11889 | 40.347% |
| C-1m | HGB | 0.000253105 | 0.0004038 | 0.0111093 | 0.100273 | 0.129235 | 40.501% |

Direction accuracy for the six primary learned models:

| Configuration | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-1m Ridge | 3,551,762 | 0.000249879 | 0.000505859 | 0.00252603 | 0.0505739 | 0.10255 | 41.264% |
| A-1m HGB | 3,551,762 | 0.00024894 | 0.000505558 | 0.00371347 | 0.069236 | 0.115111 | 41.254% |
| B-1m Ridge | 3,551,762 | 0.000250482 | 0.000505924 | 0.00227143 | 0.0493717 | 0.101084 | 41.113% |
| B-1m HGB | 3,551,762 | 0.000248919 | 0.000505561 | 0.00370446 | 0.0694166 | 0.114225 | 41.269% |
| C-1m Ridge | 3,551,762 | 0.000250443 | 0.000505939 | 0.00221086 | 0.0487958 | 0.100994 | 41.116% |
| C-1m HGB | 3,551,762 | 0.000248903 | 0.000505576 | 0.00364189 | 0.0690091 | 0.114085 | 41.259% |

The learned models beat the zero/persistence/reversal sanity baselines on R² and correlation, but the absolute effect remains small. B's test Pearson is microscopically above A's while its R² and Spearman are slightly below A's; that is not robust evidence of incremental cross-market value.

## Per-market results

These tables expose heterogeneity rather than allowing large rate/agricultural effects to disappear inside pooled averages.

### A-1m final primary models on test

| Root | Model | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | Ridge | 264,349 | 0.000215832 | 0.000340757 | -0.00259151 | 0.0135161 | 0.0281441 | 50.573% |
| NQ | HGB | 264,349 | 0.000215618 | 0.000340314 | 1.8372e-05 | 0.00930331 | 0.026453 | 50.436% |
| ES | Ridge | 264,477 | 0.000144208 | 0.000235891 | -0.00473274 | 0.0140537 | 0.0302849 | 45.823% |
| ES | HGB | 264,477 | 0.000143393 | 0.000235341 | -5.57562e-05 | 0.0121036 | 0.0287959 | 45.590% |
| RTY | Ridge | 254,658 | 0.000222607 | 0.000370038 | -0.00230047 | 0.0158785 | 0.0275208 | 47.475% |
| RTY | HGB | 254,658 | 0.000222074 | 0.000369625 | -6.63322e-05 | 0.0102371 | 0.0275158 | 47.395% |
| ZF | Ridge | 210,761 | 4.60473e-05 | 7.05815e-05 | 0.0439892 | 0.211736 | 0.248868 | 31.508% |
| ZF | HGB | 210,761 | 4.41644e-05 | 7.0888e-05 | 0.0356683 | 0.218899 | 0.256098 | 31.660% |
| ZN | Ridge | 225,177 | 7.70698e-05 | 0.000114496 | 0.0788965 | 0.317648 | 0.341652 | 33.295% |
| ZN | HGB | 225,177 | 7.50025e-05 | 0.000116602 | 0.0447048 | 0.327224 | 0.35619 | 33.572% |
| ZB | Ridge | 191,918 | 0.00014024 | 0.000216962 | 0.0438658 | 0.270497 | 0.295348 | 30.431% |
| ZB | HGB | 191,918 | 0.000138545 | 0.000218004 | 0.0346625 | 0.293263 | 0.314544 | 30.540% |
| CL | Ridge | 257,971 | 0.000622758 | 0.00105137 | -0.000833936 | 0.0128521 | 0.0458203 | 47.105% |
| CL | HGB | 257,971 | 0.000622125 | 0.00105103 | -0.000188431 | 0.00605635 | 0.0459836 | 46.990% |
| NG | Ridge | 226,335 | 0.000656063 | 0.00106103 | 0.00164096 | 0.0408491 | 0.0890671 | 42.291% |
| NG | HGB | 226,335 | 0.000655415 | 0.0010606 | 0.00244907 | 0.052491 | 0.0983894 | 42.277% |
| GC | Ridge | 259,686 | 0.0003218 | 0.000523195 | -0.000511334 | 0.0221424 | 0.0425784 | 50.450% |
| GC | HGB | 259,686 | 0.000321944 | 0.00052295 | 0.000424886 | 0.0206436 | 0.0421826 | 50.510% |
| HG | Ridge | 235,739 | 0.000338708 | 0.000524758 | 0.00113191 | 0.0380777 | 0.0468775 | 46.789% |
| HG | HGB | 235,739 | 0.000338252 | 0.000524993 | 0.00023593 | 0.0178949 | 0.0448594 | 46.840% |
| 6E | Ridge | 252,424 | 6.95635e-05 | 0.000106551 | -0.0135949 | 0.0471072 | 0.0687034 | 39.917% |
| 6E | HGB | 252,424 | 6.80941e-05 | 0.00010576 | 0.0014009 | 0.0471864 | 0.0684068 | 39.905% |
| 6J | Ridge | 245,277 | 8.45207e-05 | 0.000141162 | 0.00424525 | 0.0831148 | 0.131082 | 36.364% |
| 6J | HGB | 245,277 | 8.26693e-05 | 0.000140979 | 0.00683002 | 0.0878682 | 0.13682 | 36.250% |
| 6B | Ridge | 219,831 | 8.40128e-05 | 0.000125852 | -0.00588144 | 0.0529673 | 0.0730399 | 35.689% |
| 6B | HGB | 219,831 | 8.2193e-05 | 0.000125275 | 0.00331232 | 0.0580052 | 0.074843 | 35.803% |
| ZC | Ridge | 136,473 | 0.000360594 | 0.000550741 | 0.0254663 | 0.230925 | 0.280938 | 35.578% |
| ZC | HGB | 136,473 | 0.000362934 | 0.000546485 | 0.0404693 | 0.2583 | 0.301875 | 35.627% |
| ZS | Ridge | 159,581 | 0.000236595 | 0.00035991 | 0.0066357 | 0.0823481 | 0.116112 | 37.172% |
| ZS | HGB | 159,581 | 0.000234509 | 0.000360281 | 0.00459053 | 0.0899706 | 0.121584 | 37.138% |
| ZW | Ridge | 147,105 | 0.000444367 | 0.000671298 | 0.00620067 | 0.0848322 | 0.123291 | 37.532% |
| ZW | HGB | 147,105 | 0.000443555 | 0.000671392 | 0.00592225 | 0.0937321 | 0.134637 | 37.377% |

### B-1m final primary models on test

| Root | Model | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | Ridge | 264,349 | 0.000215948 | 0.000340904 | -0.0034563 | 0.010854 | 0.0273225 | 50.615% |
| NQ | HGB | 264,349 | 0.000215606 | 0.000340277 | 0.000232931 | 0.0155787 | 0.0273344 | 50.405% |
| ES | Ridge | 264,477 | 0.000144408 | 0.000236145 | -0.00689928 | 0.00918671 | 0.0287804 | 45.758% |
| ES | HGB | 264,477 | 0.00014338 | 0.000235309 | 0.000220209 | 0.0175377 | 0.0292883 | 45.585% |
| RTY | Ridge | 254,658 | 0.000222679 | 0.000370277 | -0.00359656 | 0.011413 | 0.0299246 | 47.603% |
| RTY | HGB | 254,658 | 0.000222077 | 0.000369606 | 3.71217e-05 | 0.0127263 | 0.028124 | 47.465% |
| ZF | Ridge | 210,761 | 4.80255e-05 | 7.20555e-05 | 0.00364388 | 0.165274 | 0.218571 | 30.661% |
| ZF | HGB | 210,761 | 4.41201e-05 | 7.09286e-05 | 0.0345634 | 0.215482 | 0.25623 | 31.713% |
| ZN | Ridge | 225,177 | 7.88266e-05 | 0.00011536 | 0.0649385 | 0.259943 | 0.303518 | 32.045% |
| ZN | HGB | 225,177 | 7.48201e-05 | 0.000116833 | 0.0409106 | 0.326979 | 0.357592 | 33.593% |
| ZB | Ridge | 191,918 | 0.000142065 | 0.000216977 | 0.0437312 | 0.239127 | 0.272584 | 29.591% |
| ZB | HGB | 191,918 | 0.000138584 | 0.000217933 | 0.0352881 | 0.29238 | 0.315587 | 30.570% |
| CL | Ridge | 257,971 | 0.000622881 | 0.00105174 | -0.00152629 | 0.0070502 | 0.0441691 | 47.033% |
| CL | HGB | 257,971 | 0.000622076 | 0.00105093 | 1.55884e-05 | 0.0101044 | 0.0455009 | 46.995% |
| NG | Ridge | 226,335 | 0.000656457 | 0.00106092 | 0.00184871 | 0.0436471 | 0.0901622 | 42.295% |
| NG | HGB | 226,335 | 0.000655479 | 0.00106069 | 0.00228354 | 0.0500838 | 0.0967872 | 42.226% |
| GC | Ridge | 259,686 | 0.000321951 | 0.000523195 | -0.000511613 | 0.019367 | 0.0382332 | 50.339% |
| GC | HGB | 259,686 | 0.000321949 | 0.000523022 | 0.000150196 | 0.013646 | 0.041979 | 50.447% |
| HG | Ridge | 235,739 | 0.000338428 | 0.000524468 | 0.00223472 | 0.04833 | 0.0596951 | 47.279% |
| HG | HGB | 235,739 | 0.000338257 | 0.00052501 | 0.000171305 | 0.0162941 | 0.0446435 | 46.863% |
| 6E | Ridge | 252,424 | 7.00388e-05 | 0.000107167 | -0.0253489 | 0.0390668 | 0.0645817 | 39.844% |
| 6E | HGB | 252,424 | 6.80685e-05 | 0.00010577 | 0.00121266 | 0.0455498 | 0.0682929 | 39.952% |
| 6J | Ridge | 245,277 | 8.55135e-05 | 0.000141783 | -0.00453031 | 0.0703571 | 0.122155 | 35.924% |
| 6J | HGB | 245,277 | 8.2638e-05 | 0.00014099 | 0.00666899 | 0.0868554 | 0.136959 | 36.418% |
| 6B | Ridge | 219,831 | 8.4489e-05 | 0.000126124 | -0.0102432 | 0.0580513 | 0.0833668 | 36.177% |
| 6B | HGB | 219,831 | 8.21638e-05 | 0.000125287 | 0.00312648 | 0.0565232 | 0.0751447 | 35.762% |
| ZC | Ridge | 136,473 | 0.000361537 | 0.000550217 | 0.027321 | 0.223442 | 0.273889 | 35.289% |
| ZC | HGB | 136,473 | 0.00036299 | 0.000546453 | 0.0405817 | 0.256859 | 0.301479 | 35.648% |
| ZS | Ridge | 159,581 | 0.000237161 | 0.000359791 | 0.00729246 | 0.0855349 | 0.121943 | 37.362% |
| ZS | HGB | 159,581 | 0.000234493 | 0.000360308 | 0.00444206 | 0.0883704 | 0.121202 | 37.130% |
| ZW | Ridge | 147,105 | 0.000445035 | 0.00067113 | 0.00669803 | 0.0854046 | 0.12608 | 37.649% |
| ZW | HGB | 147,105 | 0.000443446 | 0.000671427 | 0.00581976 | 0.0932352 | 0.133661 | 37.355% |

### C-1m final primary models on test

| Root | Model | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | Ridge | 264,349 | 0.000215938 | 0.000340911 | -0.00349623 | 0.00992557 | 0.0270563 | 50.592% |
| NQ | HGB | 264,349 | 0.000215614 | 0.000340283 | 0.000198379 | 0.0143457 | 0.0262737 | 50.386% |
| ES | Ridge | 264,477 | 0.000144397 | 0.000236151 | -0.00694361 | 0.00814936 | 0.0284214 | 45.780% |
| ES | HGB | 264,477 | 0.000143382 | 0.000235311 | 0.000201812 | 0.0169978 | 0.0289044 | 45.612% |
| RTY | Ridge | 254,658 | 0.000222674 | 0.000370255 | -0.00347944 | 0.0116683 | 0.029454 | 47.601% |
| RTY | HGB | 254,658 | 0.000222059 | 0.000369586 | 0.000143937 | 0.0143689 | 0.0285253 | 47.463% |
| ZF | Ridge | 210,761 | 4.79425e-05 | 7.20604e-05 | 0.00350769 | 0.164711 | 0.217347 | 30.614% |
| ZF | HGB | 210,761 | 4.4186e-05 | 7.08898e-05 | 0.0356192 | 0.217507 | 0.254643 | 31.656% |
| ZN | Ridge | 225,177 | 7.87369e-05 | 0.000115173 | 0.067967 | 0.26626 | 0.309471 | 32.243% |
| ZN | HGB | 225,177 | 7.4858e-05 | 0.000116787 | 0.0416693 | 0.327656 | 0.35432 | 33.581% |
| ZB | Ridge | 191,918 | 0.000141949 | 0.000217036 | 0.043217 | 0.237521 | 0.271954 | 29.633% |
| ZB | HGB | 191,918 | 0.000138565 | 0.000217875 | 0.0357998 | 0.291323 | 0.313939 | 30.545% |
| CL | Ridge | 257,971 | 0.000622882 | 0.00105178 | -0.00160551 | 0.00609365 | 0.0440181 | 47.006% |
| CL | HGB | 257,971 | 0.000622147 | 0.00105107 | -0.000265678 | 0.00254612 | 0.0418717 | 46.908% |
| NG | Ridge | 226,335 | 0.000656419 | 0.00106096 | 0.00178807 | 0.0429925 | 0.089359 | 42.273% |
| NG | HGB | 226,335 | 0.000655384 | 0.00106069 | 0.00228947 | 0.0504992 | 0.0969002 | 42.313% |
| GC | Ridge | 259,686 | 0.000321936 | 0.000523186 | -0.000474327 | 0.0190896 | 0.0384045 | 50.348% |
| GC | HGB | 259,686 | 0.000321945 | 0.000522982 | 0.000305473 | 0.0176026 | 0.0422012 | 50.473% |
| HG | Ridge | 235,739 | 0.000338421 | 0.000524476 | 0.0022041 | 0.0479175 | 0.0596652 | 47.274% |
| HG | HGB | 235,739 | 0.00033823 | 0.000524942 | 0.000430975 | 0.0211964 | 0.0460303 | 46.916% |
| 6E | Ridge | 252,424 | 7.00203e-05 | 0.000107165 | -0.025313 | 0.0391654 | 0.0647002 | 39.807% |
| 6E | HGB | 252,424 | 6.80939e-05 | 0.000105769 | 0.00122282 | 0.0461864 | 0.0682119 | 39.897% |
| 6J | Ridge | 245,277 | 8.54737e-05 | 0.000141782 | -0.00451364 | 0.0699501 | 0.121067 | 35.921% |
| 6J | HGB | 245,277 | 8.26685e-05 | 0.000140988 | 0.00670311 | 0.0866167 | 0.136543 | 36.248% |
| 6B | Ridge | 219,831 | 8.44635e-05 | 0.000126146 | -0.0105973 | 0.0565898 | 0.0812922 | 36.105% |
| 6B | HGB | 219,831 | 8.21622e-05 | 0.000125287 | 0.00312982 | 0.0565697 | 0.0749421 | 35.833% |
| ZC | Ridge | 136,473 | 0.000361398 | 0.000550284 | 0.027082 | 0.223133 | 0.274065 | 35.354% |
| ZC | HGB | 136,473 | 0.000362483 | 0.000546701 | 0.0397112 | 0.256417 | 0.299169 | 35.611% |
| ZS | Ridge | 159,581 | 0.000237099 | 0.000359831 | 0.00707321 | 0.0842422 | 0.120954 | 37.356% |
| ZS | HGB | 159,581 | 0.000234422 | 0.000360299 | 0.00448956 | 0.0894511 | 0.121401 | 37.167% |
| ZW | Ridge | 147,105 | 0.000444971 | 0.000671175 | 0.00656397 | 0.084519 | 0.125004 | 37.638% |
| ZW | HGB | 147,105 | 0.000443472 | 0.000671332 | 0.00609955 | 0.095083 | 0.133879 | 37.407% |

## NQ-only results

| Experiment | Model | Validation Pearson | Test R² | Test Pearson | Test Spearman | Pooled model on NQ Pearson |
|---|---|---:|---:|---:|---:|---:|
| A-1m | Ridge | 0.0083224 | -0.00331092 | -0.00129713 | 0.0069602 | 0.0135161 |
| A-1m | HGB | 0.00393411 | -0.00200198 | -0.00121179 | -0.00394915 | 0.00930331 |
| B-1m | Ridge | 0.00564402 | -0.00430424 | -0.00385887 | 0.00174629 | 0.010854 |
| B-1m | HGB | 0.00206441 | -0.00184918 | 0.000460334 | -0.00422114 | 0.0155787 |
| C-1m | Ridge | 0.00602285 | -0.00392578 | -0.00181528 | 0.00160663 | 0.00992557 |
| C-1m | HGB | -0.00883962 | -0.00173657 | -0.000935188 | -0.000991377 | 0.0143457 |

NQ-only models were approximately zero or negative on test. Every pooled counterpart produced a positive NQ Pearson, so pooled learning helped this NQ sanity check, though NQ's signal remains weak.

## Naive baseline results

The naive predictions are feature-representation independent, so the same test values apply to A/B/C.

| Configuration | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero | 3,551,762 | 0.000247502 | 0.0005065 | -3.42143e-07 | undefined | undefined | 25.626% |
| Persistence | 3,422,043 | 0.000379114 | 0.00073586 | -1.07931 | -0.0328396 | -0.0740646 | 38.256% |
| Reversal | 3,422,043 | 0.00035232 | 0.00071208 | -0.947087 | 0.0328396 | 0.0740646 | 43.864% |

Zero has undefined Pearson/Spearman because its predictions are constant. Its direction score is not conventional model accuracy; it mainly counts exactly zero realized returns. Persistence and reversal both have very negative R².

## Training-size sensitivity

Validation selected `B_1m` for this diagnostic.

| Rows/root | Pooled rows | Validation R² | Validation Pearson | Validation Spearman |
|---:|---:|---:|---:|---:|
| 50,000 | 800,000 | 0.00420668 | 0.0720908 | 0.108884 |
| 100,000 | 1,600,000 | 0.00462148 | 0.0791122 | 0.110289 |
| 200,000 | 3,200,000 | 0.00494468 | 0.0807364 | 0.111996 |

Validation chose 200,000 rows/root. Its locked test result was R² `0.00408195`, Pearson `0.0732194`, and Spearman `0.116009`. Performance continued improving through 200k/root, although the gain from 100k to 200k was modest.

### Full-training Ridge

| Configuration | N | MAE | RMSE | R² | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capped 1.6M Ridge validation | 3,519,199 | 0.000223191 | 0.000438982 | 0.00277205 | 0.0538631 | 0.100212 | 40.939% |
| Full 16,914,241-row Ridge validation | 3,519,199 | 0.000222765 | 0.000438981 | 0.00277761 | 0.0531598 | 0.0992499 | 40.935% |

Validation support for replacing the capped Ridge: **FALSE**. The full fit slightly improved R² but reduced Pearson and Spearman, so it was not promoted to final test evaluation. Imputation medians are learned from the deterministic 1.6M training sample; all eligible train rows determine scaling statistics and Ridge coefficients.

### Market-specific HGB diagnostic

Roots were selected only by the pooled B-1m HGB validation Pearson.

| Root | Pooled val Pearson | Market-specific val Pearson | Pooled test Pearson | Market-specific test Pearson |
|---|---:|---:|---:|---:|
| ZN | 0.300889 | 0.310108 | 0.326979 | 0.342132 |
| ZC | 0.264151 | 0.273667 | 0.256859 | 0.261761 |
| ZB | 0.233562 | 0.265773 | 0.29238 | 0.325367 |

All three market-specific models improved validation and test Pearson over the pooled model on their root. This is a useful diagnostic, not permission to select markets on test or to build a strategy.

## Feature importance

Ridge values are coefficients after train-only standardization. HGB values are deterministic three-repeat permutation importance on 32,000 validation observations (2,000/root), scored as decrease in validation R². Test was not used. Importance reflects model reliance, not causality, mechanism, profitability, or robustness.

### A-1m Ridge top 15

| Rank | Feature | Standardized coefficient | Sign | Magnitude |
|---:|---|---:|---|---:|
| 1 | `close_location_in_bar` | -1.4498e-05 | negative | 1.4498e-05 |
| 2 | `close_to_ema_5` | -1.37778e-05 | negative | 1.37778e-05 |
| 3 | `upper_wick_pct` | 9.90124e-06 | positive | 9.90124e-06 |
| 4 | `log_return_1m` | -7.78558e-06 | negative | 7.78558e-06 |
| 5 | `lower_wick_pct` | -7.59439e-06 | negative | 7.59439e-06 |
| 6 | `log_return_5m` | 7.12124e-06 | positive | 7.12124e-06 |
| 7 | `close_minus_vwap_60m_pct` | -4.93494e-06 | negative | 4.93494e-06 |
| 8 | `ema20_vs_ema50` | 4.29595e-06 | positive | 4.29595e-06 |
| 9 | `close_to_ema_20` | -4.21129e-06 | negative | 4.21129e-06 |
| 10 | `return_open_to_close` | 3.99517e-06 | positive | 3.99517e-06 |
| 11 | `root_NG` | 3.86701e-06 | positive | 3.86701e-06 |
| 12 | `root_ZC` | -3.82795e-06 | negative | 3.82795e-06 |
| 13 | `realized_vol_60m` | -2.94734e-06 | negative | 2.94734e-06 |
| 14 | `ema5_vs_ema20` | 2.53708e-06 | positive | 2.53708e-06 |
| 15 | `root_ZS` | 2.44617e-06 | positive | 2.44617e-06 |

### A-1m HGB top 15 validation permutation importance

| Rank | Feature | Mean validation R² decrease | Repeat SD |
|---:|---|---:|---:|
| 1 | `close_location_in_bar` | 0.00709761 | 0.00011487 |
| 2 | `root` | 0.00292132 | 0.000286901 |
| 3 | `volume` | 0.00125383 | 0.000182963 |
| 4 | `close_to_ema_5` | 0.000711962 | 0.000295706 |
| 5 | `range_pct` | 0.00065659 | 0.000134995 |
| 6 | `upper_wick_pct` | 0.000460775 | 8.98522e-05 |
| 7 | `realized_vol_5m` | 0.00036988 | 3.67591e-05 |
| 8 | `chicago_minute` | 0.000308572 | 6.73926e-05 |
| 9 | `rolling_volume_mean_20` | 0.000245192 | 2.7267e-05 |
| 10 | `log_return_5m` | 0.000239252 | 0.000177713 |
| 11 | `log_return_15m` | 0.000183321 | 2.2785e-05 |
| 12 | `rsi_14` | 0.000141528 | 9.94522e-05 |
| 13 | `return_open_to_close` | 5.5446e-05 | 1.52913e-05 |
| 14 | `rolling_volume_mean_60` | 5.44937e-05 | 0.000119949 |
| 15 | `close_to_ema_20` | 4.76989e-05 | 7.37048e-05 |

### B-1m Ridge top 15

| Rank | Feature | Standardized coefficient | Sign | Magnitude |
|---:|---|---:|---|---:|
| 1 | `close_location_in_bar` | -1.60952e-05 | negative | 1.60952e-05 |
| 2 | `close_to_ema_5` | -1.39535e-05 | negative | 1.39535e-05 |
| 3 | `log_return_1m` | -9.77747e-06 | negative | 9.77747e-06 |
| 4 | `upper_wick_pct` | 9.35157e-06 | positive | 9.35157e-06 |
| 5 | `lower_wick_pct` | -7.12693e-06 | negative | 7.12693e-06 |
| 6 | `log_return_5m` | 6.63551e-06 | positive | 6.63551e-06 |
| 7 | `close_minus_vwap_60m_pct` | -4.66992e-06 | negative | 4.66992e-06 |
| 8 | `CL_return_1m` | 4.64758e-06 | positive | 4.64758e-06 |
| 9 | `NQ_return_1m` | 4.29853e-06 | positive | 4.29853e-06 |
| 10 | `ema20_vs_ema50` | 4.1147e-06 | positive | 4.1147e-06 |
| 11 | `close_to_ema_20` | -4.04033e-06 | negative | 4.04033e-06 |
| 12 | `root_NG` | 3.86538e-06 | positive | 3.86538e-06 |
| 13 | `NQ_return_5m` | -3.86122e-06 | negative | 3.86122e-06 |
| 14 | `return_open_to_close` | 3.85169e-06 | positive | 3.85169e-06 |
| 15 | `root_ZC` | -3.8429e-06 | negative | 3.8429e-06 |

### B-1m HGB top 15 validation permutation importance

| Rank | Feature | Mean validation R² decrease | Repeat SD |
|---:|---|---:|---:|
| 1 | `close_location_in_bar` | 0.00687263 | 0.000756763 |
| 2 | `root` | 0.00222157 | 0.000135285 |
| 3 | `volume` | 0.000980516 | 0.000205919 |
| 4 | `close_to_ema_5` | 0.000600391 | 0.000214542 |
| 5 | `upper_wick_pct` | 0.000258457 | 3.41483e-05 |
| 6 | `log_return_5m` | 0.000150869 | 8.53877e-05 |
| 7 | `NQ_return_1m` | 0.000146311 | 0.000101697 |
| 8 | `range_pct` | 0.000131362 | 0.000222954 |
| 9 | `CL_return_1m` | 0.000128538 | 1.99088e-05 |
| 10 | `hour_sin` | 0.000116108 | 5.47608e-05 |
| 11 | `ES_return_5m` | 9.9326e-05 | 4.70241e-05 |
| 12 | `ES_return_15m` | 8.63478e-05 | 6.05421e-05 |
| 13 | `log_return_30m` | 6.86329e-05 | 1.63701e-06 |
| 14 | `rolling_volume_mean_20` | 4.44004e-05 | 0.00012062 |
| 15 | `volume_zscore_20` | 2.89493e-05 | 0.000144078 |

### C-1m Ridge top 15

| Rank | Feature | Standardized coefficient | Sign | Magnitude |
|---:|---|---:|---|---:|
| 1 | `log_return_1m` | 1.90187e-05 | positive | 1.90187e-05 |
| 2 | `close_location_in_bar` | -1.58704e-05 | negative | 1.58704e-05 |
| 3 | `close_to_ema_5` | -1.39285e-05 | negative | 1.39285e-05 |
| 4 | `log_return_5m` | 1.13651e-05 | positive | 1.13651e-05 |
| 5 | `upper_wick_pct` | 9.42247e-06 | positive | 9.42247e-06 |
| 6 | `relative_to_6E_return_1m` | -8.79674e-06 | negative | 8.79674e-06 |
| 7 | `relative_to_GC_return_1m` | -8.71164e-06 | negative | 8.71164e-06 |
| 8 | `lower_wick_pct` | -7.17182e-06 | negative | 7.17182e-06 |
| 9 | `relative_to_ES_return_5m` | -6.62748e-06 | negative | 6.62748e-06 |
| 10 | `relative_to_NQ_return_1m` | -6.61169e-06 | negative | 6.61169e-06 |
| 11 | `relative_to_NQ_return_5m` | 6.50285e-06 | positive | 6.50285e-06 |
| 12 | `relative_to_NQ_return_15m` | -5.51793e-06 | negative | 5.51793e-06 |
| 13 | `relative_to_CL_return_1m` | -4.90637e-06 | negative | 4.90637e-06 |
| 14 | `relative_to_ES_return_15m` | 4.68452e-06 | positive | 4.68452e-06 |
| 15 | `close_minus_vwap_60m_pct` | -4.637e-06 | negative | 4.637e-06 |

### C-1m HGB top 15 validation permutation importance

| Rank | Feature | Mean validation R² decrease | Repeat SD |
|---:|---|---:|---:|
| 1 | `close_location_in_bar` | 0.00773606 | 0.000557847 |
| 2 | `root` | 0.00305754 | 0.00028227 |
| 3 | `volume` | 0.00154138 | 0.000214275 |
| 4 | `close_minus_vwap_240m_pct` | 0.000670308 | 0.000135882 |
| 5 | `range_pct` | 0.000602031 | 0.000181966 |
| 6 | `close_to_ema_5` | 0.000463009 | 0.000195255 |
| 7 | `relative_to_NQ_return_1m` | 0.00043162 | 4.74679e-06 |
| 8 | `chicago_minute` | 0.000401239 | 0.000113531 |
| 9 | `atr_14_normalized` | 0.000370932 | 3.37494e-06 |
| 10 | `close_to_ema_20` | 0.000249169 | 5.25443e-05 |
| 11 | `lower_wick_pct` | 0.000215812 | 0.000121739 |
| 12 | `relative_to_6E_return_1m` | 0.00019606 | 1.07529e-05 |
| 13 | `relative_to_ES_return_1m` | 0.000175182 | 4.36507e-05 |
| 14 | `relative_volume_20` | 0.000174119 | 6.4774e-05 |
| 15 | `hour_cos` | 0.000133832 | 7.72018e-05 |

C-1m did use relative features: 8 appear in Ridge's top 15 and 3 in HGB's top 15. However, C did not outperform B, so model reliance did not translate into a superior representation.

## Time-of-day diagnostics

All following subgroup results are exploratory test diagnostics. They must not be used to retroactively tune a trading hour, regime, or weekday.

### A-1m HGB by Chicago hour

| Hour | N | Pearson | Spearman | Direction accuracy |
|---:|---:|---:|---:|---:|
| 0 | 141,601 | 0.0650552 | 0.0989363 | 39.255% |
| 1 | 162,873 | 0.0823241 | 0.116184 | 40.153% |
| 2 | 169,999 | 0.0923587 | 0.114133 | 40.651% |
| 3 | 169,719 | 0.0804657 | 0.113108 | 40.652% |
| 4 | 165,566 | 0.0853488 | 0.117981 | 40.133% |
| 5 | 166,668 | 0.0758002 | 0.123064 | 40.311% |
| 6 | 174,594 | 0.0560272 | 0.10539 | 40.616% |
| 7 | 172,854 | 0.061226 | 0.10037 | 42.834% |
| 8 | 166,838 | 0.0486468 | 0.078005 | 44.024% |
| 9 | 183,600 | 0.0535598 | 0.0892506 | 43.605% |
| 10 | 182,814 | 0.0805805 | 0.10862 | 42.753% |
| 11 | 181,325 | 0.0949579 | 0.131769 | 42.347% |
| 12 | 176,256 | 0.101007 | 0.146673 | 41.765% |
| 13 | 151,700 | 0.0595986 | 0.122917 | 42.021% |
| 14 | 140,340 | 0.0253257 | 0.13703 | 41.906% |
| 15 | 128,383 | 0.0791039 | 0.167577 | 41.764% |
| 17 | 111,163 | 0.0181855 | 0.116167 | 42.036% |
| 18 | 109,780 | 0.0455722 | 0.11088 | 41.121% |
| 19 | 153,434 | 0.072303 | 0.107396 | 40.529% |
| 20 | 148,961 | 0.0732991 | 0.107324 | 40.131% |
| 21 | 136,819 | 0.0984665 | 0.121686 | 39.776% |
| 22 | 129,862 | 0.100237 | 0.127783 | 39.402% |
| 23 | 126,613 | 0.075072 | 0.124579 | 39.415% |

Volatility regimes use root-specific tertiles of backward-looking `realized_vol_15m`, learned only from the deterministic primary training sample.

| Volatility regime | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| high | 1,093,060 | 0.0368174 | 0.0697322 | 46.102% |
| low | 1,255,603 | 0.137736 | 0.157669 | 36.519% |
| medium | 1,188,038 | 0.109055 | 0.130109 | 41.868% |
| unavailable | 15,061 | 0.0717832 | 0.154304 | 35.655% |

| Chicago weekday | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| Friday | 511,839 | 0.0757639 | 0.114801 | 41.693% |
| Monday | 710,246 | 0.0656305 | 0.119469 | 41.187% |
| Sunday | 192,368 | 0.0570687 | 0.112789 | 41.250% |
| Thursday | 685,504 | 0.0664177 | 0.109194 | 41.165% |
| Tuesday | 738,422 | 0.070207 | 0.117647 | 41.158% |
| Wednesday | 713,383 | 0.0735707 | 0.114767 | 41.190% |

### B-1m HGB by Chicago hour

| Hour | N | Pearson | Spearman | Direction accuracy |
|---:|---:|---:|---:|---:|
| 0 | 141,601 | 0.0678602 | 0.0995608 | 39.294% |
| 1 | 162,873 | 0.0875081 | 0.115405 | 40.088% |
| 2 | 169,999 | 0.0869695 | 0.113291 | 40.726% |
| 3 | 169,719 | 0.0842331 | 0.112714 | 40.683% |
| 4 | 165,566 | 0.0890473 | 0.116643 | 40.136% |
| 5 | 166,668 | 0.078936 | 0.122887 | 40.294% |
| 6 | 174,594 | 0.0552212 | 0.104682 | 40.590% |
| 7 | 172,854 | 0.0522105 | 0.09874 | 42.902% |
| 8 | 166,838 | 0.0424192 | 0.0761107 | 44.043% |
| 9 | 183,600 | 0.0572614 | 0.0894769 | 43.620% |
| 10 | 182,814 | 0.0796584 | 0.108902 | 42.763% |
| 11 | 181,325 | 0.092912 | 0.130072 | 42.318% |
| 12 | 176,256 | 0.100146 | 0.146021 | 41.839% |
| 13 | 151,700 | 0.0647565 | 0.121671 | 42.048% |
| 14 | 140,340 | 0.0170513 | 0.134868 | 41.944% |
| 15 | 128,383 | 0.0755322 | 0.164876 | 41.861% |
| 17 | 111,163 | 0.0443854 | 0.115879 | 42.117% |
| 18 | 109,780 | 0.0464066 | 0.110581 | 41.134% |
| 19 | 153,434 | 0.0759013 | 0.106887 | 40.528% |
| 20 | 148,961 | 0.0696595 | 0.105608 | 40.071% |
| 21 | 136,819 | 0.0904967 | 0.120759 | 39.834% |
| 22 | 129,862 | 0.0950271 | 0.126859 | 39.316% |
| 23 | 126,613 | 0.0717241 | 0.123663 | 39.406% |

Volatility regimes use root-specific tertiles of backward-looking `realized_vol_15m`, learned only from the deterministic primary training sample.

| Volatility regime | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| high | 1,093,060 | 0.0381493 | 0.0691259 | 46.101% |
| low | 1,255,603 | 0.136335 | 0.156268 | 36.540% |
| medium | 1,188,038 | 0.108302 | 0.129036 | 41.891% |
| unavailable | 15,061 | 0.0457022 | 0.148579 | 35.635% |

| Chicago weekday | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| Friday | 511,839 | 0.0782045 | 0.114356 | 41.720% |
| Monday | 710,246 | 0.0632045 | 0.118488 | 41.202% |
| Sunday | 192,368 | 0.0616042 | 0.112237 | 41.228% |
| Thursday | 685,504 | 0.0650033 | 0.10838 | 41.198% |
| Tuesday | 738,422 | 0.0700425 | 0.116515 | 41.151% |
| Wednesday | 713,383 | 0.0751862 | 0.11375 | 41.210% |

### C-1m HGB by Chicago hour

| Hour | N | Pearson | Spearman | Direction accuracy |
|---:|---:|---:|---:|---:|
| 0 | 141,601 | 0.064188 | 0.100046 | 39.289% |
| 1 | 162,873 | 0.0862551 | 0.114888 | 40.134% |
| 2 | 169,999 | 0.0887927 | 0.113058 | 40.661% |
| 3 | 169,719 | 0.0806786 | 0.113046 | 40.675% |
| 4 | 165,566 | 0.0874189 | 0.116943 | 40.150% |
| 5 | 166,668 | 0.0758779 | 0.121524 | 40.277% |
| 6 | 174,594 | 0.0624431 | 0.104647 | 40.619% |
| 7 | 172,854 | 0.0494414 | 0.0996071 | 42.907% |
| 8 | 166,838 | 0.047622 | 0.0769667 | 44.056% |
| 9 | 183,600 | 0.0611633 | 0.0897066 | 43.630% |
| 10 | 182,814 | 0.0769681 | 0.10774 | 42.804% |
| 11 | 181,325 | 0.0926639 | 0.129459 | 42.306% |
| 12 | 176,256 | 0.0941032 | 0.145841 | 41.789% |
| 13 | 151,700 | 0.0609259 | 0.121821 | 42.060% |
| 14 | 140,340 | 0.0282323 | 0.136915 | 41.959% |
| 15 | 128,383 | 0.0823047 | 0.164597 | 41.718% |
| 17 | 111,163 | 0.0347704 | 0.11317 | 41.979% |
| 18 | 109,780 | 0.0400534 | 0.108494 | 41.096% |
| 19 | 153,434 | 0.0679231 | 0.106331 | 40.519% |
| 20 | 148,961 | 0.07098 | 0.106023 | 40.108% |
| 21 | 136,819 | 0.0970736 | 0.119494 | 39.747% |
| 22 | 129,862 | 0.0935906 | 0.126553 | 39.387% |
| 23 | 126,613 | 0.0682074 | 0.123033 | 39.402% |

Volatility regimes use root-specific tertiles of backward-looking `realized_vol_15m`, learned only from the deterministic primary training sample.

| Volatility regime | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| high | 1,093,060 | 0.0367062 | 0.0687265 | 46.106% |
| low | 1,255,603 | 0.136777 | 0.156324 | 36.526% |
| medium | 1,188,038 | 0.108894 | 0.129201 | 41.871% |
| unavailable | 15,061 | 0.0610154 | 0.150795 | 35.854% |

| Chicago weekday | N | Pearson | Spearman | Direction accuracy |
|---|---:|---:|---:|---:|
| Friday | 511,839 | 0.0744802 | 0.114184 | 41.703% |
| Monday | 710,246 | 0.0678866 | 0.118389 | 41.185% |
| Sunday | 192,368 | 0.0562369 | 0.110413 | 41.204% |
| Thursday | 685,504 | 0.06744 | 0.108969 | 41.212% |
| Tuesday | 738,422 | 0.0688136 | 0.116454 | 41.155% |
| Wednesday | 713,383 | 0.0713941 | 0.113234 | 41.184% |

## Volatility diagnostics

Across A/B/C, low-volatility rows had the strongest Pearson (about 0.136-0.138), medium volatility was next (about 0.108-0.109), and high volatility was weakest (about 0.037-0.038). This concentration is descriptive and needs new-period replication.

## Prediction-decile analysis

### A-1m HGB

| Decile | N | Mean prediction | Mean actual | Median actual |
|---:|---:|---:|---:|---:|
| 1 | 355,177 | -3.0888e-05 | -7.29022e-05 | 0 |
| 2 | 355,177 | -1.02892e-05 | -2.0361e-05 | 0 |
| 3 | 355,176 | -6.69593e-06 | -1.11938e-05 | 0 |
| 4 | 355,176 | -1.84667e-06 | -1.06006e-07 | 0 |
| 5 | 355,176 | -5.5339e-07 | -1.49281e-06 | 0 |
| 6 | 355,176 | 5.86375e-07 | 2.30758e-06 | 0 |
| 7 | 355,176 | 7.78363e-07 | 4.80813e-06 | 0 |
| 8 | 355,176 | 7.08485e-06 | 1.25607e-05 | 0 |
| 9 | 355,176 | 1.08296e-05 | 1.81635e-05 | 0 |
| 10 | 355,176 | 3.12737e-05 | 7.11788e-05 | 0 |

### B-1m HGB

| Decile | N | Mean prediction | Mean actual | Median actual |
|---:|---:|---:|---:|---:|
| 1 | 355,177 | -3.11373e-05 | -7.12652e-05 | 0 |
| 2 | 355,177 | -9.94078e-06 | -2.04095e-05 | 0 |
| 3 | 355,176 | -6.5348e-06 | -1.19284e-05 | 0 |
| 4 | 355,176 | -1.82954e-06 | -3.68276e-07 | 0 |
| 5 | 355,176 | -7.38323e-07 | -1.12044e-06 | 0 |
| 6 | 355,176 | 3.71849e-07 | 2.6266e-06 | 0 |
| 7 | 355,176 | 6.20073e-07 | 2.55871e-06 | 0 |
| 8 | 355,176 | 7.32756e-06 | 1.3821e-05 | 0 |
| 9 | 355,176 | 1.07131e-05 | 1.94733e-05 | 0 |
| 10 | 355,176 | 3.08117e-05 | 6.95751e-05 | 0 |

### C-1m HGB

| Decile | N | Mean prediction | Mean actual | Median actual |
|---:|---:|---:|---:|---:|
| 1 | 355,177 | -3.01392e-05 | -7.2436e-05 | 0 |
| 2 | 355,177 | -1.0376e-05 | -1.78932e-05 | 0 |
| 3 | 355,176 | -6.67395e-06 | -1.1166e-05 | 0 |
| 4 | 355,176 | -1.50507e-06 | -1.09841e-06 | 0 |
| 5 | 355,176 | -4.87018e-07 | -1.26694e-06 | 0 |
| 6 | 355,176 | 5.45903e-07 | 2.11449e-06 | 0 |
| 7 | 355,176 | 6.58812e-07 | 5.89528e-06 | 0 |
| 8 | 355,176 | 6.48742e-06 | 1.25816e-05 | 0 |
| 9 | 355,176 | 1.05815e-05 | 1.87419e-05 | 0 |
| 10 | 355,176 | 3.14099e-05 | 6.74901e-05 | 0 |

For all three HGB models, higher prediction deciles generally correspond to higher mean realized returns, but the relationship is small and is not a profitability result.

## Real X/y sample

One real eligible NQ training row: timestamp `2021-09-23T06:02:00+00:00`, root `NQ`, instrument `2770`, segment `1`.

The metadata identifies the observation. `X` contains the inputs below; `y` is the observed next-minute return. Future labels are forbidden from `X`.

### A-1m: all 35 own-market inputs

| Feature | Value |
|---|---:|
| `log_return_1m` | -0.000180958257 |
| `log_return_5m` | 1.64523745e-05 |
| `log_return_15m` | -0.000394775805 |
| `log_return_30m` | 4.93579355e-05 |
| `log_return_60m` | 9.87183074e-05 |
| `return_open_to_close` | -0.000180941885 |
| `range_pct` | 0.00019742687 |
| `upper_wick_pct` | 1.64522391e-05 |
| `lower_wick_pct` | 0 |
| `close_location_in_bar` | 0 |
| `close_to_ema_5` | -7.28649065e-05 |
| `close_to_ema_20` | -0.000114697327 |
| `close_to_ema_50` | -0.000105354947 |
| `ema5_vs_ema20` | -4.18324204e-05 |
| `ema20_vs_ema50` | 9.34237989e-06 |
| `rsi_14` | 39.5061728 |
| `atr_14_normalized` | 0.000176273991 |
| `realized_vol_5m` | 0.000184443562 |
| `realized_vol_15m` | 0.000122128446 |
| `realized_vol_60m` | 0.000137582221 |
| `volume` | 43 |
| `rolling_volume_mean_20` | 46.65 |
| `rolling_volume_mean_60` | 54.3 |
| `volume_zscore_20` | -0.124622133 |
| `volume_zscore_60` | -0.281439741 |
| `relative_volume_20` | 0.921757771 |
| `close_minus_vwap_60m_pct` | -0.000192220529 |
| `close_minus_vwap_240m_pct` | 0.000107708357 |
| `chicago_hour` | 1 |
| `chicago_minute` | 2 |
| `chicago_day_of_week` | 3 |
| `hour_sin` | 0.267238376 |
| `hour_cos` | 0.963630453 |
| `day_of_week_sin` | 0.433883739 |
| `day_of_week_cos` | -0.900968868 |

### B-1m: examples from the 18 raw anchors

| Feature | Value |
|---|---:|
| `ES_return_1m` | -0.00011368804 |
| `ES_return_5m` | 5.68488673e-05 |
| `ES_return_15m` | 0 |
| `ZN_return_1m` | 0 |
| `ZN_return_5m` | -0.000117515718 |
| `ZN_return_15m` | -0.000235017627 |

### C-1m: corresponding relative examples

| Feature | Value |
|---|---:|
| `relative_to_ES_return_1m` | -6.72702166e-05 |
| `relative_to_ES_return_5m` | -4.03964928e-05 |
| `relative_to_ES_return_15m` | -0.000394775805 |
| `relative_to_ZN_return_1m` | -0.000180958257 |
| `relative_to_ZN_return_5m` | 0.000133968092 |
| `relative_to_ZN_return_15m` | -0.000159758178 |

`y = future_return_1m = -0.000131626576`.

Mathematically, `x_t = [35 own-market features, optional cross-market representation]` and `y_t = ln(P_(t+1)/P_t)`. One sample means: at one specific minute for one futures root, these were all known input conditions, and this was the return that actually happened one minute later.

## Leakage safeguards

Automated tests cover all required safeguards:

1. exact `t+1` target construction;
2. no next-available-row substitution;
3. all future-return labels rejected from `X`;
4. backward-looking own-market lookbacks;
5. no feature crossing a contract segment;
6. no target crossing a contract segment;
7. exact-timestamp anchor joins;
8. no stale/forward-filled anchors;
9. target-root return used as the relative first term;
10. relative formula is target minus anchor;
11. valid self-anchor equals zero;
12. preprocessing statistics remain train-only;
13. validation/test observations are transform-only, never fit inputs; and
14. the original 75-minute boundary purge remains intact.

## Comparison with previous 5-minute study

| Phase/model | Target | Test N | R² | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Phase 2 A HGB | `future_return_5m` | 3,525,964 | 0.0004894 | 0.0223287 | 0.0674291 |
| Phase 2 B HGB | `future_return_5m` | 3,525,964 | 0.000575652 | 0.0240327 | 0.0672046 |
| Phase 2.1 A HGB | `future_return_1m` | 3,551,762 | 0.00371347 | 0.069236 | 0.115111 |
| Phase 2.1 B HGB | `future_return_1m` | 3,551,762 | 0.00370446 | 0.0694166 | 0.114225 |

These are different targets with different distributions and noise; a larger metric does not establish that one horizon is intrinsically better. The clean Phase 2.1 comparison is A-1m vs B-1m vs C-1m on the same rows.

## Scientific interpretation

### Explicit answers

**Q1. Can the original 35 own-market features predict next-minute returns better than the naive baselines?** Yes in this held-out period: A-1m Ridge/HGB have positive R² and correlations while persistence/reversal are strongly negative-R². The effect is still small.

**Q2. Does adding raw cross-market information improve the 1-minute model?** Not robustly. B HGB won validation Pearson by only about 0.00030; on test it had a tiny Pearson edge over A but slightly worse R² and Spearman. Ridge B was worse than A on test.

**Q3. Do target-relative cross-market features outperform raw anchor returns?** No. C did not beat B in the primary HGB validation or test comparison, and its small capped-Ridge validation lead did not persist to test.

**Q4. Which markets appear strongest on validation?** For the validation-selected B HGB, ZN, ZC, and ZB had the highest Pearson, followed by ZF.

**Q5. Do those stronger validation markets remain stronger on test?** Yes: ZN, ZB, and ZC were also the top three test Pearson roots (with their order changing), and ZF remained fourth. This is a locked diagnostic, not a new selection rule.

**Q6. Which features appear most important?** `close_location_in_bar` dominated every HGB permutation ranking. Root identity, volume, short EMA distance, range/wick shape, and short returns were also recurrent. Some raw/relative one-minute anchors appeared, but cross-market representations did not improve the aggregate result robustly.

**Q7. Does more training data improve the model?** Yes for the B HGB validation study: Pearson increased from about 0.0721 at 50k/root to 0.0791 at 100k and 0.0807 at 200k. Full-data Ridge did not improve all selection metrics and was not promoted.

**Q8. Does signal appear concentrated by time of day or volatility?** Exploratorily, yes. Low-volatility rows were much stronger than high-volatility rows, and Chicago hours around 11-12 and 21-22 were frequently among the strongest. These test subgroups require independent replication and must not be optimized in place.

**Q9. Is the evidence strong enough to justify more sophisticated models?** Not yet. The 200k/root gain and market heterogeneity justify a replication/stability experiment, but the weak aggregate R² and failed cross-market ablation do not yet justify deep or highly tuned models.

**Q10. What is the single best next scientific experiment?** Run a predeclared rolling walk-forward stability study of the B-1m 200k/root HGB and the simpler A-1m control across multiple non-overlapping chronological folds, reporting root-level and volatility-regime stability. This tests whether the observed signal repeats through time before model complexity increases.

## Limitations

- Results cover one fixed historical split; subgroup stability is not independently replicated.
- Predictive metrics are small and do not account for fees, spread, slippage, latency, execution, capacity, or risk.
- Exact missing minutes and flat bars cause legitimate nulls handled by train-only median imputation.
- HGB permutation importance uses a documented 32,000-row validation sample for computational practicality.
- The full Ridge uses primary-training-sample medians, then all eligible training rows for scaling statistics and sufficient-statistics coefficients.
- C is a reparameterization because own returns remain in the feature set; it does not add independent information.

## Recommended next experiment

Pre-register a rolling walk-forward replication of A-1m and the validation-chosen 200k/root B-1m HGB. Freeze features/hyperparameters, create several chronological train/validation folds with the same rollover and 75-minute purge safeguards, and evaluate whether per-root and low/medium/high-volatility correlations retain their sign and approximate rank. Stop before strategies or P&L.

## Artifact locations

- Results: `D:\futures-ml-data\artifacts\models\phase2_1\phase2_1_results.json`.
- Models: `D:\futures-ml-data\artifacts\models\phase2_1\{A_1m,B_1m,C_1m}\{pooled,nq_only}`.
- Full Ridge: `D:\futures-ml-data\artifacts\models\phase2_1\C_1m\pooled\ridge_full_train`.
- Training-size artifacts: `D:\futures-ml-data\artifacts\models\phase2_1\training_size`.
- Market-specific artifacts: `D:\futures-ml-data\artifacts\models\phase2_1\market_specific`.
- Prediction cache: `D:\futures-ml-data\cache\phase2_1_predictions.joblib`.
