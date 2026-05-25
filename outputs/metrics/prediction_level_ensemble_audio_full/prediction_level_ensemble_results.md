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
| `audio_only` | `outputs/metrics/dolos_fold3_audio_only` | `fold1,fold2,fold3` | yes |
| `spatial_only` | `outputs/metrics/dolos_fold3_spatial_only` | `fold3` | no |
| `spatial_skip1s` | `outputs/metrics/dolos_fold3_spatial_skip1s` | `fold3` | no |
| `spatial_clean_faces` | `outputs/metrics/dolos_fold3_spatial_clean_faces` | `fold3` | no |
| `flow_only` | `outputs/metrics/dolos_fold3_flow_only` | `fold3` | no |
| `flow_clean_faces` | `outputs/metrics/dolos_fold3_flow_clean_faces` | `fold3` | no |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00 | 61.04 | 62.39 | 61.39 | 56.35 | 48.30 | 54.26 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00 | 61.75 | 61.36 | 63.94 | 59.18 | 55.53 | 58.14 |
| `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00 | 67.43 | 67.28 | 58.03 | 56.04 | 53.09 | 55.22 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.28, gated_prior_kl=0.04, audio_only=0.68 | 69.13 | 71.41 | 62.73 | 57.86 | 52.61 | 56.69 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.17, gated_prior_kl=0.00, audio_only=0.83 | 69.88 | 69.18 | 61.77 | 58.21 | 53.02 | 56.96 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.23, gated_prior_kl=0.04, audio_only=0.73 | 69.52 | 71.77 | 62.62 | 57.68 | 52.73 | 56.58 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.16, gated_prior_kl=0.00, audio_only=0.84 | 69.87 | 69.18 | 61.76 | 57.98 | 52.95 | 56.75 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00 | 61.54 | 62.82 | 61.77 | 57.45 | 55.56 | 56.98 | `[[146, 81], [127, 130]]` |
| fold1 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00 | 63.84 | 65.71 | 56.51 | 55.59 | 58.53 | 55.59 | `[[119, 108], [106, 151]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.67, gated_prior_kl=0.05, audio_only=0.28 | 65.12 | 71.26 | 65.21 | 59.20 | 54.83 | 58.20 | `[[161, 66], [135, 122]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.25, gated_prior_kl=0.00, audio_only=0.75 | 66.29 | 66.88 | 63.46 | 59.83 | 60.45 | 59.70 | `[[140, 87], [108, 149]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.50, gated_prior_kl=0.08, audio_only=0.42 | 66.13 | 72.33 | 64.98 | 58.90 | 55.31 | 58.08 | `[[157, 70], [132, 125]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.23, gated_prior_kl=0.00, audio_only=0.77 | 66.24 | 66.88 | 63.43 | 59.14 | 60.24 | 59.06 | `[[136, 91], [107, 150]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00 | 64.79 | 64.84 | 60.95 | 58.26 | 57.92 | 58.09 | `[[141, 88], [114, 139]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00 | 64.70 | 62.94 | 67.73 | 60.44 | 49.74 | 57.80 | `[[189, 40], [156, 97]]` |
| fold2 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00 | 68.68 | 67.86 | 60.92 | 58.61 | 54.71 | 57.86 | `[[158, 71], [131, 122]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.00, audio_only=0.84 | 72.70 | 73.10 | 64.34 | 60.47 | 52.09 | 58.54 | `[[181, 48], [147, 106]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.24, gated_prior_kl=0.00, audio_only=0.76 | 73.33 | 72.38 | 65.11 | 60.88 | 52.58 | 58.96 | `[[182, 47], [146, 107]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.00, audio_only=0.84 | 72.67 | 73.10 | 64.34 | 60.47 | 52.09 | 58.54 | `[[181, 48], [147, 106]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.24, gated_prior_kl=0.00, audio_only=0.76 | 73.37 | 72.38 | 65.11 | 60.88 | 52.58 | 58.96 | `[[182, 47], [146, 107]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00 | 53.48 | 55.45 | 57.93 | 52.58 | 35.69 | 48.11 | `[[174, 45], [182, 63]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00 | 59.02 | 58.33 | 62.33 | 59.66 | 61.28 | 59.63 | `[[129, 90], [97, 148]]` |
| fold3 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00 | 69.77 | 68.27 | 56.66 | 53.91 | 46.04 | 52.22 | `[[153, 66], [152, 93]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.01, gated_prior_kl=0.06, audio_only=0.93 | 69.58 | 69.87 | 58.65 | 53.91 | 50.91 | 53.32 | `[[136, 83], [133, 112]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.01, gated_prior_kl=0.00, audio_only=0.99 | 70.02 | 68.27 | 56.75 | 53.91 | 46.04 | 52.22 | `[[153, 66], [152, 93]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.02, gated_prior_kl=0.05, audio_only=0.93 | 69.75 | 69.87 | 58.54 | 53.68 | 50.79 | 53.12 | `[[135, 84], [133, 112]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.01, gated_prior_kl=0.00, audio_only=0.99 | 70.02 | 68.27 | 56.75 | 53.91 | 46.04 | 52.22 | `[[153, 66], [152, 93]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
