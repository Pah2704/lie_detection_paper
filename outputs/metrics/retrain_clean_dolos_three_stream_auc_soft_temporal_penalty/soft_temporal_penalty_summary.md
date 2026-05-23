# Soft Temporal Penalty Summary

Experiment: `retrain_clean_dolos_three_stream_auc_soft_temporal_penalty`

## Change

Cross-attention keeps global audio-to-visual access, but receives a Gaussian distance penalty before softmax:

`bias(i, j) = -((i - j)^2 / (2 * sigma^2))`, with `sigma = 6.0`.

This is softer than hard local attention: nearby visual tokens are preferred, but distant tokens are still available if their learned attention score is strong enough.

## Standalone Cross-Attention Results

| Fold | Clean cross AUC | Soft penalty AUC | Clean calibrated BA | Soft calibrated BA | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| fold1 | 59.12 | 56.86 | 56.42 | 51.46 | worse |
| fold2 | 65.73 | 64.73 | 60.94 | 61.27 | mixed |
| fold3 | 56.92 | 64.34 | 52.75 | 58.92 | better |
| mean | 60.59 | 61.98 | 56.70 | 57.22 | better overall, not uniformly |

## Ensemble With Gated Prior-KL Temporal Mask

Output directory:

`outputs/metrics/prediction_level_ensemble_retrain_clean_temporal_mask_soft_cross`

| Method | Mean weights | Test AUC | Test BA | Test F1 |
| --- | --- | ---: | ---: | ---: |
| gated prior-KL | cross=0.00, gated=1.00 | 64.38 | 59.39 | 60.27 |
| soft cross only | cross=1.00, gated=0.00 | 61.98 | 57.22 | 48.36 |
| ensemble raw AUC | cross=0.62, gated=0.38 | 65.20 | 60.00 | 59.06 |
| ensemble raw BA | cross=0.65, gated=0.35 | 65.43 | 60.54 | 61.39 |

Final report directory for this ablation:

`outputs/metrics/final_report_clean_temporal_mask_soft_cross`

Compared with the previous temporal-mask final report (`outputs/metrics/final_report_clean_temporal_mask`):

| Final candidate | Test AUC | Calibrated BA |
| --- | ---: | ---: |
| previous `ensemble_raw_auc_roc` | 65.32 | 59.61 |
| new `ensemble_raw_balanced_accuracy` | 65.43 | 60.54 |

This is a small AUC gain and a clearer BA gain.

## Interpretation

Soft temporal penalty is useful as an ensemble branch, but not as a uniformly stronger standalone model.

- It fixes much of the fold3 weakness of the clean cross-attention baseline.
- It slightly improves fold2 calibrated BA, despite a small AUC drop.
- It hurts fold1, so it should not replace gated prior-KL as the primary single model.
- Best use: keep `ensemble_raw_balanced_accuracy` as the strongest DOLOS-only final candidate if balanced accuracy is the main reporting metric.

Hard local attention remains rejected because it reduced fold3 AUC and calibrated BA.
