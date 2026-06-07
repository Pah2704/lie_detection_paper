# Paper Ablation Summary

Source configs:

- Single-stream: `configs/paper_ablation_static.yaml`, `configs/paper_ablation_flow.yaml`, `configs/paper_ablation_audio.yaml`.
- Prior comparison: `configs/paper_gated_neutral_no_prior.yaml`, `configs/paper_gated_no_prior.yaml`, `configs/paper_gated_prior_kl.yaml`.
- Exploratory vote-score-sum: `configs/paper_gated_vote_score_sum_g05_floor005.yaml`.
- Focal pilot: `configs/paper_gated_vote_score_sum_g05_floor005_focal.yaml`.

## How to Read for Paper

- `Gated neutral no-prior` uses uniform `model.gate_init_weights=[1/3, 1/3, 1/3]` and `training.gate_prior_weight=0.0`.
- `Gated prior-init w/o Prior-KL` keeps `model.gate_init_weights=[0.10, 0.45, 0.45]` but sets `training.gate_prior_weight=0.0`.
- `Gated Prior-KL` uses `training.gate_prior_weight=0.2` and prior `[0.10, 0.45, 0.45]`, adding a KL penalty only when the weight is positive.
- Read AUC as ranking quality, Cal. BA as the validation-calibrated operating point, and Cal. F1 Lie as lie-class detection quality.

Short interpretation:

- Prior-KL is essentially tied on AUC (-0.04 pp) and slightly lower on Cal. BA (-0.31 pp).
- Prior-KL improves lie-class F1 by +3.99 pp and macro F1 by +0.27 pp.
- The safest claim is that Prior-KL shifts the gated model toward better lie-class behavior, not that it uniformly improves every metric.
- Against prior-init w/o Prior-KL, Prior-KL improves Cal. F1 Lie in 6/9 paired runs.
- Against neutral no-prior, Prior-KL improves Cal. F1 Lie in 5/9 paired runs.

Main-method interpretation after the vote-score-sum expansion:

- `Vote-score-sum gamma=0.5 floor=0.05` is stronger than fixed Prior-KL on AUC (+1.90 pp), Cal. BA (+2.05 pp), Cal. F1 Lie (+1.78 pp), and Cal. Macro F1 (+2.43 pp).
- It is also stronger than prior-init w/o Prior-KL on AUC (+1.86 pp), Cal. BA (+1.74 pp), Cal. F1 Lie (+5.77 pp), and Cal. Macro F1 (+2.70 pp).
- Its weak point is raw threshold-0.5 Lie F1, mainly because some runs are poorly calibrated at 0.5 despite good AUC. Use validation-calibrated threshold metrics for the operating-point claim.

## Single-Stream Fold1-Fold3

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static only | 3 | fold1,fold2,fold3 | 42 | 65.33 +- 1.71 | 59.16 +- 3.20 | 60.91 +- 1.86 | 54.39 +- 7.38 | 59.23 +- 3.18 |
| Flow only | 3 | fold1,fold2,fold3 | 42 | 54.46 +- 7.03 | 50.07 +- 0.09 | 53.21 +- 5.85 | 50.39 +- 9.79 | 52.49 +- 6.28 |
| Audio only | 3 | fold1,fold2,fold3 | 42 | 58.03 +- 2.50 | 51.39 +- 2.41 | 56.04 +- 2.38 | 53.09 +- 6.40 | 55.22 +- 2.84 |

## Prior-KL Contribution

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gated prior-init w/o Prior-KL | 9 | fold1,fold2,fold3 | 42,123,2025 | 64.20 +- 4.20 | 57.23 +- 6.68 | 59.27 +- 3.62 | 52.75 +- 11.45 | 57.18 +- 4.93 |
| Gated neutral no-prior | 9 | fold1,fold2,fold3 | 42,123,2025 | 61.90 +- 3.31 | 56.27 +- 5.26 | 57.34 +- 2.85 | 51.78 +- 12.67 | 55.16 +- 5.10 |
| Gated Prior-KL | 9 | fold1,fold2,fold3 | 42,123,2025 | 64.16 +- 4.56 | 57.94 +- 5.61 | 58.96 +- 4.16 | 56.74 +- 10.67 | 57.45 +- 5.15 |

## Paired Prior-KL Delta vs Prior-Init w/o Prior-KL

Delta is `Gated Prior-KL - Gated prior-init w/o Prior-KL` in percentage points, paired by fold and seed.

| Metric | Mean delta pp | Std pp | Prior-KL wins | No-prior wins | Ties |
| --- | --- | --- | --- | --- | --- |
| AUC | -0.04 | 4.59 | 5/9 | 4/9 | 0/9 |
| BA@0.5 | +0.71 | 5.62 | 4/9 | 3/9 | 2/9 |
| Cal. BA | -0.31 | 3.96 | 6/9 | 3/9 | 0/9 |
| Cal. F1 Lie | +3.99 | 11.22 | 6/9 | 3/9 | 0/9 |
| Cal. Macro F1 | +0.27 | 4.13 | 5/9 | 4/9 | 0/9 |

## Paired Delta Details vs Prior-Init w/o Prior-KL

| Fold | Seed | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | 42 | -8.29 | -7.99 | -5.17 | +1.33 | -4.23 |
| fold1 | 123 | +3.22 | +5.00 | +0.55 | -3.24 | -0.54 |
| fold1 | 2025 | +1.69 | -1.16 | +2.84 | +5.73 | +3.55 |
| fold2 | 42 | +4.69 | +2.88 | +1.13 | +3.82 | +1.47 |
| fold2 | 123 | -1.75 | +0.00 | -2.96 | -5.34 | -3.54 |
| fold2 | 2025 | +4.56 | +10.23 | +5.95 | +9.15 | +6.53 |
| fold3 | 42 | -6.07 | -6.33 | -6.61 | -8.00 | -5.68 |
| fold3 | 123 | +1.90 | +3.72 | +1.35 | +30.16 | +4.00 |
| fold3 | 2025 | -0.31 | +0.00 | +0.08 | +2.28 | +0.87 |

## Paired Prior-KL Delta vs Neutral No-Prior

Delta is `Gated Prior-KL - Gated neutral no-prior` in percentage points, paired by fold and seed.

| Metric | Mean delta pp | Std pp | Prior-KL wins | No-prior wins | Ties |
| --- | --- | --- | --- | --- | --- |
| AUC | +2.26 | 5.46 | 6/9 | 3/9 | 0/9 |
| BA@0.5 | +1.67 | 8.78 | 6/9 | 3/9 | 0/9 |
| Cal. BA | +1.62 | 4.37 | 6/9 | 3/9 | 0/9 |
| Cal. F1 Lie | +4.96 | 17.20 | 5/9 | 4/9 | 0/9 |
| Cal. Macro F1 | +2.29 | 5.98 | 5/9 | 4/9 | 0/9 |

## Paired Delta Details vs Neutral No-Prior

| Fold | Seed | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| fold1 | 42 | -8.96 | -8.54 | -4.54 | +3.89 | -3.14 |
| fold1 | 123 | +3.66 | +5.58 | +0.76 | -8.89 | -1.53 |
| fold1 | 2025 | +1.81 | +0.96 | +2.26 | +4.67 | +2.85 |
| fold2 | 42 | +8.73 | +7.01 | +5.81 | +14.60 | +7.47 |
| fold2 | 123 | -2.13 | -11.45 | -2.92 | -10.02 | -4.52 |
| fold2 | 2025 | +6.77 | +7.98 | +8.22 | +3.60 | +7.86 |
| fold3 | 42 | +4.70 | +9.82 | +0.53 | -2.52 | +2.13 |
| fold3 | 123 | +6.19 | +11.57 | +6.08 | +45.81 | +12.67 |
| fold3 | 2025 | -0.45 | -7.92 | -1.64 | -6.47 | -3.18 |

## Exploratory Adaptive Prior-KL Pilot

This is a seed-42 pilot only, not part of the complete 9-run ablation above. The adaptive prior interpolates from low-face `[0.05, 0.45, 0.50]` to high-face `[0.50, 0.25, 0.25]` with KL weight `0.1`.

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gated Adaptive Prior-KL | 3 | fold1,fold2,fold3 | 42 | 62.48 | 56.44 | 59.09 | 55.87 | 58.28 |
| Gated Prior-KL | 3 | fold1,fold2,fold3 | 42 | 64.72 | 60.12 | 59.10 | 62.76 | 58.85 |
| Gated neutral no-prior | 3 | fold1,fold2,fold3 | 42 | 63.23 | 57.35 | 58.51 | 57.44 | 56.70 |
| Gated prior-init w/o Prior-KL | 3 | fold1,fold2,fold3 | 42 | 67.94 | 63.93 | 62.65 | 63.71 | 61.67 |

Interpretation: current Adaptive Prior-KL should not be expanded to 3 seeds yet. It is essentially tied with fixed Prior-KL on Cal. BA, but lower on AUC and Cal. F1 Lie. Window-level `face_valid_ratio` is usually near 1.0, so the current adaptive signal has too little low-face variation and mostly behaves like a high-static fixed prior.

## Exploratory Reliability-Confidence Voting Pilot

This is a seed-42 pilot only. The rule computes clip-level fusion weights as `w_m = r_m * |s_m - 0.5|^gamma`, using `face_valid_ratio` as reliability for static/flow, audio reliability `1.0`, and `gamma=1.0`.

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gated Reliability-Confidence | 3 | fold1,fold2,fold3 | 42 | 60.60 | 56.12 | 68.21 | 56.76 | 44.56 | 53.39 |
| Gated Prior-KL | 3 | fold1,fold2,fold3 | 42 | 64.72 | 60.12 | 55.03 | 59.10 | 62.76 | 58.85 |
| Gated Adaptive Prior-KL | 3 | fold1,fold2,fold3 | 42 | 62.48 | 56.44 | 41.94 | 59.09 | 55.87 | 58.28 |
| Gated neutral no-prior | 3 | fold1,fold2,fold3 | 42 | 63.23 | 57.35 | 42.49 | 58.51 | 57.44 | 56.70 |
| Gated prior-init w/o Prior-KL | 3 | fold1,fold2,fold3 | 42 | 67.94 | 63.93 | 68.95 | 62.65 | 63.71 | 61.67 |

Fold-level details:

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 6 | 58.49 | 66.13 | 62.04 | 69.08 | 60.18 | 50.37 | 57.64 | 0.7089 |
| fold2 | 1 | 63.87 | 58.33 | 52.11 | 69.10 | 56.27 | 52.77 | 55.63 | 0.5270 |
| fold3 | 2 | 56.98 | 57.35 | 54.21 | 66.45 | 53.84 | 30.53 | 46.90 | 0.5308 |

Interpretation: this direction is technically cleaner than a fixed hand-written prior because it uses per-clip reliability and stream confidence. However, the current `gamma=1.0` pilot is not strong enough for the main paper claim. It improves default-threshold Lie F1, but calibrated Lie F1 collapses, especially on fold3. Before expanding to 3 seeds, the confidence vote should be softened, for example with `gamma=0.5` or `0.25`, a small confidence floor, and possibly a cap on maximum stream weight.

## Exploratory Softened Reliability-Confidence Voting Pilot

This is a seed-42 pilot only. The softened rule computes `w_m = r_m * (0.05 + |s_m - 0.5|^gamma)` and keeps `confidence_detach=true`.

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reliability-confidence gamma=1.0 floor=0.00 | 3 | fold1,fold2,fold3 | 42 | 60.60 | 56.12 | 68.21 | 56.76 | 44.56 | 53.39 |
| Reliability-confidence gamma=0.5 floor=0.05 | 3 | fold1,fold2,fold3 | 42 | 67.20 | 60.26 | 63.54 | 61.27 | 56.15 | 59.94 |
| Reliability-confidence gamma=0.25 floor=0.05 | 3 | fold1,fold2,fold3 | 42 | 61.16 | 56.37 | 66.65 | 58.05 | 49.10 | 55.56 |
| Gated Prior-KL | 3 | fold1,fold2,fold3 | 42 | 64.72 | 60.12 | 55.03 | 59.10 | 62.76 | 58.85 |
| Gated prior-init w/o Prior-KL | 3 | fold1,fold2,fold3 | 42 | 67.94 | 63.93 | 68.95 | 62.65 | 63.71 | 61.67 |

Fold-level details for `gamma=0.5, floor=0.05`:

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 22 | 60.10 | 70.10 | 63.29 | 64.26 | 61.16 | 49.87 | 58.10 | 0.6497 |
| fold2 | 1 | 63.84 | 60.49 | 54.29 | 65.59 | 58.88 | 56.52 | 58.42 | 0.5176 |
| fold3 | 31 | 60.38 | 71.02 | 63.19 | 60.77 | 63.76 | 62.05 | 63.32 | 0.4889 |
| Mean | - | 61.44 | 67.20 | 60.26 | 63.54 | 61.27 | 56.15 | 59.94 | - |

Fold-level details for `gamma=0.25, floor=0.05`:

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 6 | 59.40 | 66.36 | 60.89 | 68.71 | 61.84 | 59.78 | 61.30 | 0.6400 |
| fold2 | 1 | 63.84 | 60.64 | 55.44 | 63.05 | 58.33 | 55.19 | 57.73 | 0.5132 |
| fold3 | 2 | 56.71 | 56.48 | 52.79 | 68.19 | 53.97 | 32.32 | 47.66 | 0.5219 |
| Mean | - | 59.98 | 61.16 | 56.37 | 66.65 | 58.05 | 49.10 | 55.56 | - |

Interpretation: `gamma=0.5, floor=0.05` is the best reliability-confidence variant so far. It recovers the hard-voting failure and outperforms fixed Prior-KL on AUC, BA@0.5, F1 Lie@0.5, Cal. BA, and Cal. Macro F1 for seed 42. It still trails Prior-KL on Cal. F1 Lie and trails the prior-init no-prior baseline overall. `gamma=0.25` should not be expanded to 3 seeds because it underperforms mainly on fold3.

## Vote-Score-Sum 3-Seed Expansion

The rule changes the fusion decision from weight-averaging stream logits to class-support voting:

`support_m = r_m * (0.05 + max(p_m)^0.5)`

`S_class = sum support_m for streams voting for class`

The final lie score is determined by the difference between lie support and truth support. This directly encodes the paper argument that one modality should receive high influence only when it is both reliable and agrees with the class-level evidence. It also lets two aligned streams beat one strong opposing stream unless the opposing stream has substantially higher reliability/support.

Full 3-seed mean comparison:

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vote-score-sum gamma=0.5 floor=0.05 | 9 | fold1,fold2,fold3 | 42,123,2025 | 66.06 +- 3.82 | 58.62 +- 4.18 | 53.90 +- 21.48 | 55.50 +- 9.26 | 61.01 +- 3.08 | 58.52 +- 10.05 | 59.87 +- 4.43 |
| Gated Prior-KL | 9 | fold1,fold2,fold3 | 42,123,2025 | 64.16 +- 4.56 | 57.94 +- 5.61 | 63.30 +- 12.23 | 53.18 +- 11.87 | 58.96 +- 4.16 | 56.74 +- 10.67 | 57.45 +- 5.15 |
| Gated prior-init w/o Prior-KL | 9 | fold1,fold2,fold3 | 42,123,2025 | 64.20 +- 4.20 | 57.23 +- 6.68 | 68.61 +- 1.85 | 50.97 +- 13.48 | 59.27 +- 3.62 | 52.75 +- 11.45 | 57.18 +- 4.93 |
| Gated neutral no-prior | 9 | fold1,fold2,fold3 | 42,123,2025 | 61.90 +- 3.31 | 56.27 +- 5.26 | 50.45 +- 27.90 | 49.73 +- 12.86 | 57.34 +- 2.85 | 51.78 +- 12.67 | 55.16 +- 5.10 |

Fold-level details:

| Fold | Seed | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 42 | 22 | 63.89 | 68.54 | 63.23 | 61.70 | 62.78 | 61.80 | 58.41 | 60.99 | 0.5294 |
| fold1 | 123 | 11 | 60.26 | 63.01 | 58.11 | 63.37 | 57.99 | 59.46 | 52.71 | 57.84 | 0.6483 |
| fold1 | 2025 | 14 | 62.82 | 62.55 | 57.24 | 49.52 | 55.42 | 57.01 | 39.22 | 51.85 | 0.6476 |
| fold2 | 42 | 9 | 68.16 | 69.37 | 64.46 | 70.21 | 64.11 | 64.29 | 68.52 | 64.21 | 0.5578 |
| fold2 | 123 | 9 | 73.68 | 66.68 | 58.69 | 66.32 | 57.84 | 62.53 | 66.16 | 62.51 | 0.5876 |
| fold2 | 2025 | 10 | 66.32 | 66.90 | 61.20 | 54.20 | 59.64 | 59.82 | 63.77 | 59.77 | 0.3201 |
| fold3 | 42 | 7 | 66.47 | 70.15 | 57.96 | 69.98 | 54.34 | 64.67 | 62.61 | 64.16 | 0.7170 |
| fold3 | 123 | 4 | 58.97 | 58.72 | 56.51 | 49.01 | 54.85 | 56.15 | 47.74 | 54.25 | 0.5043 |
| fold3 | 2025 | 7 | 57.42 | 68.60 | 50.20 | 0.81 | 32.52 | 63.34 | 67.57 | 63.30 | 0.1421 |

Paired deltas against key 3-seed baselines:

| Comparison | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum - Gated Prior-KL | +1.90 | +0.68 | -9.39 | +2.32 | +2.05 | +1.78 | +2.43 |
| Vote-score-sum - Gated prior-init w/o Prior-KL | +1.86 | +1.39 | -14.71 | +4.53 | +1.74 | +5.77 | +2.70 |
| Vote-score-sum - Gated neutral no-prior | +4.16 | +2.35 | +3.46 | +5.77 | +3.67 | +6.75 | +4.72 |

Paired win counts:

| Comparison | AUC wins | Cal. BA wins | Cal. F1 Lie wins | Cal. Macro F1 wins |
| --- | ---: | ---: | ---: | ---: |
| Vote-score-sum vs Gated Prior-KL | 5/9 | 5/9 | 5/9 | 5/9 |
| Vote-score-sum vs prior-init w/o Prior-KL | 6/9 | 6/9 | 7/9 | 8/9 |
| Vote-score-sum vs neutral no-prior | 8/9 | 8/9 | 6/9 | 8/9 |

Interpretation: the 3-seed expansion supports vote-score-sum as a stronger paper direction than fixed Prior-KL. It improves the main ranking metric AUC (+1.90 pp vs Prior-KL, +1.86 pp vs prior-init no-KL) and improves calibrated operating-point metrics, especially Cal. BA and Cal. Macro F1. The strongest contribution claim should therefore be ranking/calibrated balanced performance from clip-adaptive agreement-aware fusion, not default-threshold Lie F1.

Important caveat: `F1 Lie@0.5` is unstable and lower than prior-init no-KL because vote-score-sum can produce poorly calibrated raw scores in some runs. The clearest case is `fold3 seed2025`, where AUC is high (68.60) but threshold 0.5 predicts almost all samples as Truth (F1 Lie 0.81); validation calibration fixes this run (Cal. F1 Lie 67.57, Cal. BA 63.34). For the paper, report calibrated threshold metrics as the operating point and discuss threshold sensitivity.

Decision: use `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0` as the main proposed fusion method. Keep `Gated Prior-KL`, `Gated prior-init w/o Prior-KL`, and `Gated neutral no-prior` as baselines. Do not add `consensus_bonus=0.25` unless a later section explicitly studies calibration/threshold stability.

## Focal Loss Pilot for Vote-Score-Sum

This is a seed-42 pilot only. It keeps the main proposed fusion rule `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0` and changes the training loss from CE to focal loss using `configs/paper_gated_vote_score_sum_g05_floor005_focal.yaml`.

Mean comparison against the CE version on the same seed and folds:

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum CE | 3 | fold1,fold2,fold3 | 42 | 69.35 +- 0.81 | 61.88 +- 3.45 | 67.30 +- 4.85 | 60.41 +- 5.30 | 63.59 +- 1.56 | 63.18 +- 5.08 | 63.12 +- 1.85 |
| Vote-score-sum focal | 3 | fold1,fold2,fold3 | 42 | 64.14 +- 3.88 | 54.55 +- 7.87 | 22.71 +- 39.34 | 42.50 +- 18.20 | 61.59 +- 2.22 | 59.01 +- 8.24 | 60.63 +- 2.98 |

Fold-level focal details:

| Fold | Best epoch | Val AUC | Test AUC | BA@0.5 | F1 Lie@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold1 | 5 | 60.20 | 61.09 | 50.00 | 0.00 | 59.90 | 56.70 | 59.15 | 0.2834 |
| fold2 | 29 | 70.38 | 68.51 | 63.64 | 68.14 | 64.11 | 68.16 | 64.05 | 0.5129 |
| fold3 | 17 | 62.82 | 62.82 | 50.00 | 0.00 | 60.77 | 52.17 | 58.68 | 0.3256 |

Interpretation: focal loss should not be expanded to 3 seeds in its current form. It improves Fold2 Lie F1 but hurts Fold1 and Fold3 sharply at threshold 0.5, where both folds predict only Truth. Validation calibration partly recovers the operating point, but the calibrated mean is still below the CE vote-score-sum baseline. The split imbalance is also small in these folds, so focal class-alpha does not act as a strong Lie-class correction; it mainly emphasizes hard examples.

Post-hoc calibration note: calibrating directly for `f1_lie` raises mean F1 Lie but lowers balanced and macro performance because the selected thresholds become very low and over-predict Lie. For the main paper table, keep balanced/macro-oriented calibrated metrics as the operating-point claim and discuss F1 Lie as a threshold-sensitive metric.

## Early-Stopping and Consensus-Bonus Pilots

These are seed-42 pilots only. Both keep the main vote-score-sum formulation (`gamma=0.5`, `floor=0.05`, CE loss). The two isolated changes are:

- `paper_gated_vote_sum_earlystop_macro_f1`: changes only `training.early_stopping_metric` from `val_auc_roc` to `val_macro_f1`.
- `paper_gated_vote_sum_bonus01`: changes only `model.vote_consensus_bonus` from `0.0` to `0.1`, while keeping early stopping on `val_auc_roc`.

Mean comparison against the CE vote-score-sum seed-42 baseline:

| Method | Runs | Folds | Seeds | AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vote-score-sum CE | 3 | fold1,fold2,fold3 | 42 | 69.35 +- 0.81 | 61.88 +- 3.45 | 67.30 +- 4.85 | 60.41 +- 5.30 | 63.59 +- 1.56 | 63.18 +- 5.08 | 63.12 +- 1.85 |
| Early-stop macro F1 | 3 | fold1,fold2,fold3 | 42 | 66.58 +- 6.71 | 61.74 +- 5.21 | 62.97 +- 6.35 | 61.46 +- 5.11 | 59.57 +- 5.36 | 68.67 +- 1.91 | 56.93 +- 8.04 |
| Consensus bonus 0.1 | 3 | fold1,fold2,fold3 | 42 | 65.81 +- 6.07 | 58.66 +- 5.67 | 48.78 +- 28.99 | 53.33 +- 11.84 | 59.43 +- 3.86 | 65.88 +- 6.32 | 58.43 +- 4.19 |

Fold-level details:

| Method | Fold | Best epoch | Val metric | Test AUC | BA@0.5 | F1 Lie@0.5 | Macro F1@0.5 | Cal. BA | Cal. F1 Lie | Cal. Macro F1 | Cal. threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early-stop macro F1 | fold1 | 4 | 55.42 macro | 58.85 | 55.97 | 55.97 | 55.78 | 53.83 | 68.16 | 48.46 | 0.2341 |
| Early-stop macro F1 | fold2 | 9 | 61.20 macro | 70.04 | 63.12 | 68.35 | 62.89 | 60.45 | 70.79 | 57.85 | 0.3246 |
| Early-stop macro F1 | fold3 | 9 | 57.67 macro | 70.86 | 66.12 | 64.59 | 65.70 | 64.45 | 67.07 | 64.46 | 0.4039 |
| Consensus bonus 0.1 | fold1 | 11 | 66.19 AUC | 58.82 | 52.49 | 15.92 | 40.06 | 56.06 | 58.59 | 56.05 | 0.1640 |
| Consensus bonus 0.1 | fold2 | 10 | 66.32 AUC | 68.90 | 63.64 | 59.73 | 62.81 | 58.61 | 69.22 | 55.97 | 0.1850 |
| Consensus bonus 0.1 | fold3 | 23 | 65.66 AUC | 69.71 | 59.84 | 70.70 | 57.12 | 63.64 | 69.84 | 63.26 | 0.6495 |

Interpretation: both pilots pass the narrow Cal. F1 Lie gate (`>=63`) and reduce Cal. F1 Lie standard deviation relative to the 9-run CE vote-score-sum baseline (`10.05`). However, this is not a clean improvement for the paper. Early stopping on macro F1 raises Cal. F1 Lie by selecting much lower thresholds, but it hurts Cal. BA (-4.02 pp vs CE seed-42), Cal. Macro F1 (-6.19 pp), and AUC (-2.77 pp). The low Fold1 threshold (0.2341) indicates a Lie-biased operating point rather than a better ranking model.

Consensus bonus 0.1 is also not a good expansion candidate. It improves Fold3 calibrated F1 Lie, but Fold1 collapses at threshold 0.5 (F1 Lie 15.92), producing very high default-threshold variance. Its seed-42 mean is below CE on AUC (-3.54 pp), BA@0.5 (-3.22 pp), Macro F1@0.5 (-7.08 pp), Cal. BA (-4.16 pp), and Cal. Macro F1 (-4.69 pp).

Decision: do not expand either pilot to seeds 123 and 2025. If the original narrow selection rule must be applied mechanically, `consensus_bonus=0.1` beats early-stop macro F1 on Cal. Macro F1 among the two pilots, but it is still worse than the CE vote-score-sum main method and should not replace it. Keep `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0, early_stopping_metric=val_auc_roc` as the main method.

## Missing Runs

The original Prior-KL, neutral/prior-init no-prior, single-stream ablations, `vote_score_sum, gamma=0.5, floor=0.05, consensus_bonus=0.0` 3-seed expansion, focal-loss pilot, early-stop macro-F1 pilot, and consensus-bonus pilot are complete. No additional run is required before drafting the core ablation table.
