# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.01`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/dolos_three_stream_gated_logits_prior_kl` | `fold1,fold2,fold3` | yes |
| `gated_temp_aux` | `outputs/metrics/dolos_three_stream_gated_logits_temp_aux` | `fold1,fold3` | no |
| `gated_margin_norm` | `outputs/metrics/dolos_three_stream_gated_margin_norm` | `fold1` | no |
| `audio_only` | `outputs/metrics/dolos_fold3_audio_only` | `fold3` | no |
| `spatial_only` | `outputs/metrics/dolos_fold3_spatial_only` | `fold3` | no |
| `spatial_skip1s` | `outputs/metrics/dolos_fold3_spatial_skip1s` | `fold3` | no |
| `spatial_clean_faces` | `outputs/metrics/dolos_fold3_spatial_clean_faces` | `fold3` | no |
| `flow_only` | `outputs/metrics/dolos_fold3_flow_only` | `fold3` | no |
| `flow_clean_faces` | `outputs/metrics/dolos_fold3_flow_clean_faces` | `fold3` | no |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 61.04 | 62.39 | 61.39 | 56.35 | 48.30 | 54.26 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.75 | 61.36 | 63.94 | 59.18 | 55.53 | 58.14 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.76, gated_prior_kl=0.24 | 62.58 | 64.81 | 64.87 | 60.08 | 58.62 | 58.91 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.61, gated_prior_kl=0.39 | 63.46 | 63.19 | 65.05 | 60.17 | 54.95 | 58.91 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.76, gated_prior_kl=0.24 | 62.61 | 64.81 | 64.87 | 60.01 | 58.58 | 58.85 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.61, gated_prior_kl=0.39 | 63.47 | 63.19 | 65.05 | 60.17 | 54.95 | 58.91 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 61.54 | 62.82 | 61.77 | 57.45 | 55.56 | 56.98 | `[[146, 81], [127, 130]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.76, gated_prior_kl=0.24 | 64.05 | 69.98 | 65.34 | 59.76 | 56.00 | 58.89 | `[[160, 67], [131, 126]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.77, gated_prior_kl=0.23 | 64.10 | 69.98 | 65.33 | 59.54 | 55.88 | 58.69 | `[[159, 68], [131, 126]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 64.79 | 64.84 | 60.95 | 58.26 | 57.92 | 58.09 | `[[141, 88], [114, 139]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 64.70 | 62.94 | 67.73 | 60.44 | 49.74 | 57.80 | `[[189, 40], [156, 97]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.91, gated_prior_kl=0.09 | 65.97 | 65.79 | 66.58 | 61.30 | 53.07 | 59.39 | `[[183, 46], [145, 108]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.79, gated_prior_kl=0.21 | 66.44 | 64.37 | 67.54 | 61.93 | 53.69 | 60.00 | `[[185, 44], [144, 109]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.91, gated_prior_kl=0.09 | 66.00 | 65.79 | 66.59 | 61.30 | 53.07 | 59.39 | `[[183, 46], [145, 108]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.79, gated_prior_kl=0.21 | 66.48 | 64.37 | 67.54 | 61.93 | 53.69 | 60.00 | `[[185, 44], [144, 109]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00 | 53.48 | 55.45 | 57.93 | 52.58 | 35.69 | 48.11 | `[[174, 45], [182, 63]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00 | 59.02 | 58.33 | 62.33 | 59.66 | 61.28 | 59.63 | `[[129, 90], [97, 148]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.60, gated_prior_kl=0.40 | 57.72 | 58.65 | 62.68 | 59.19 | 66.79 | 58.46 | `[[93, 126], [59, 186]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 59.10 | 58.33 | 62.35 | 60.36 | 59.87 | 60.13 | `[[141, 78], [107, 138]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.60, gated_prior_kl=0.40 | 57.72 | 58.65 | 62.69 | 59.19 | 66.79 | 58.46 | `[[93, 126], [59, 186]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.04, gated_prior_kl=0.96 | 59.10 | 58.33 | 62.35 | 60.36 | 59.87 | 60.13 | `[[141, 78], [107, 138]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
- Best mean result here is `ensemble_raw_auc_roc` / `ensemble_logit_auc_roc`: mean test AUC 65.05 and mean calibrated BA 60.17.
- Compared with `gated_prior_kl`, this is +1.11 AUC points and +0.99 calibrated BA points.
- The improvement is not from one fold only: fold1 improves AUC, fold2 improves calibrated BA, and fold3 keeps the gated model dominant with only a small cross-attention contribution.

## Decision

Use prediction-level ensemble as the current final candidate:

- Primary: `ensemble_raw_auc_roc`, because it is simple and matches the best mean result.
- Equivalent fallback: `ensemble_logit_auc_roc`, same mean AUC/BA in this run.
- Keep `gated_prior_kl` as the strongest single trained model baseline.
