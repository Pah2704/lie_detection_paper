# Prediction-Level Ensemble

Folds: `fold1, fold2, fold3`.
Grid step: `0.02`.
Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.

## Selected Models

| Model | Metrics Dir | Folds | Selected |
| --- | --- | --- | --- |
| `cross_attention_auc` | `outputs/metrics/dolos_three_stream_auc` | `fold1,fold2,fold3` | yes |
| `gated_prior_kl` | `outputs/metrics/dolos_three_stream_gated_logits_prior_kl` | `fold1,fold2,fold3` | yes |
| `gated_temp_aux` | `outputs/metrics/dolos_three_stream_gated_logits_temp_aux` | `fold1,fold3` | no |
| `gated_margin_norm` | `outputs/metrics/dolos_three_stream_gated_margin_norm` | `fold1` | no |
| `audio_only` | `outputs/metrics/dolos_fold3_audio_only` | `fold1,fold2,fold3` | yes |
| `spatial_only` | `outputs/metrics/dolos_fold3_spatial_only` | `fold1,fold2,fold3` | yes |
| `spatial_skip1s` | `outputs/metrics/dolos_fold3_spatial_skip1s` | `fold3` | no |
| `spatial_clean_faces` | `outputs/metrics/dolos_fold3_spatial_clean_faces` | `fold3` | no |
| `flow_only` | `outputs/metrics/dolos_fold3_flow_only` | `fold3` | no |
| `flow_clean_faces` | `outputs/metrics/dolos_fold3_flow_clean_faces` | `fold3` | no |

## Mean Across Folds

| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=0.00 | 61.04 | 62.39 | 61.39 | 56.35 | 48.30 | 54.26 |
| `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00, spatial_only=0.00 | 61.75 | 61.36 | 63.94 | 59.18 | 55.53 | 58.14 |
| `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00, spatial_only=0.00 | 67.43 | 67.28 | 58.03 | 56.04 | 53.09 | 55.22 |
| `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=1.00 | 60.47 | 63.20 | 63.12 | 58.76 | 48.19 | 55.80 |
| `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.18, gated_prior_kl=0.04, audio_only=0.75, spatial_only=0.03 | 69.55 | 71.87 | 62.85 | 57.99 | 52.89 | 56.85 |
| `ensemble_raw_auc_roc` | cross_attention_auc=0.20, gated_prior_kl=0.01, audio_only=0.74, spatial_only=0.05 | 70.11 | 70.32 | 62.64 | 57.63 | 49.57 | 55.72 |
| `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.22, gated_prior_kl=0.05, audio_only=0.73, spatial_only=0.01 | 69.52 | 71.87 | 62.68 | 57.99 | 52.89 | 56.85 |
| `ensemble_logit_auc_roc` | cross_attention_auc=0.17, gated_prior_kl=0.00, audio_only=0.78, spatial_only=0.05 | 70.15 | 69.64 | 62.60 | 57.20 | 48.62 | 55.18 |

## Per-Fold Results

| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fold1 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=0.00 | 64.85 | 66.88 | 65.28 | 58.21 | 51.29 | 56.59 | `[[168, 59], [148, 109]]` |
| fold1 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00, spatial_only=0.00 | 61.54 | 62.82 | 61.77 | 57.45 | 55.56 | 56.98 | `[[146, 81], [127, 130]]` |
| fold1 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00, spatial_only=0.00 | 63.84 | 65.71 | 56.51 | 55.59 | 58.53 | 55.59 | `[[119, 108], [106, 151]]` |
| fold1 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=1.00 | 58.65 | 62.61 | 66.57 | 59.79 | 53.83 | 58.39 | `[[169, 58], [141, 116]]` |
| fold1 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.38, gated_prior_kl=0.06, audio_only=0.50, spatial_only=0.06 | 66.19 | 72.33 | 65.47 | 58.90 | 55.31 | 58.08 | `[[157, 70], [132, 125]]` |
| fold1 | `ensemble_raw_auc_roc` | cross_attention_auc=0.36, gated_prior_kl=0.02, audio_only=0.56, spatial_only=0.06 | 66.51 | 69.98 | 65.34 | 59.26 | 56.14 | 58.54 | `[[156, 71], [129, 128]]` |
| fold1 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.50, gated_prior_kl=0.08, audio_only=0.42, spatial_only=0.00 | 66.13 | 72.33 | 64.98 | 58.90 | 55.31 | 58.08 | `[[157, 70], [132, 125]]` |
| fold1 | `ensemble_logit_auc_roc` | cross_attention_auc=0.26, gated_prior_kl=0.00, audio_only=0.68, spatial_only=0.06 | 66.56 | 67.95 | 65.22 | 57.98 | 53.27 | 56.92 | `[[159, 68], [139, 118]]` |
| fold2 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=0.00 | 64.79 | 64.84 | 60.95 | 58.26 | 57.92 | 58.09 | `[[141, 88], [114, 139]]` |
| fold2 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00, spatial_only=0.00 | 64.70 | 62.94 | 67.73 | 60.44 | 49.74 | 57.80 | `[[189, 40], [156, 97]]` |
| fold2 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00, spatial_only=0.00 | 68.68 | 67.86 | 60.92 | 58.61 | 54.71 | 57.86 | `[[158, 71], [131, 122]]` |
| fold2 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=1.00 | 64.35 | 66.75 | 67.00 | 62.43 | 60.13 | 61.95 | `[[161, 68], [115, 138]]` |
| fold2 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.00, audio_only=0.84, spatial_only=0.00 | 72.70 | 73.10 | 64.34 | 60.47 | 52.09 | 58.54 | `[[181, 48], [147, 106]]` |
| fold2 | `ensemble_raw_auc_roc` | cross_attention_auc=0.24, gated_prior_kl=0.00, audio_only=0.76, spatial_only=0.00 | 73.33 | 72.38 | 65.11 | 60.88 | 52.58 | 58.96 | `[[182, 47], [146, 107]]` |
| fold2 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.16, gated_prior_kl=0.00, audio_only=0.84, spatial_only=0.00 | 72.67 | 73.10 | 64.34 | 60.47 | 52.09 | 58.54 | `[[181, 48], [147, 106]]` |
| fold2 | `ensemble_logit_auc_roc` | cross_attention_auc=0.24, gated_prior_kl=0.00, audio_only=0.76, spatial_only=0.00 | 73.37 | 72.38 | 65.11 | 60.88 | 52.58 | 58.96 | `[[182, 47], [146, 107]]` |
| fold3 | `cross_attention_auc` | cross_attention_auc=1.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=0.00 | 53.48 | 55.45 | 57.93 | 52.58 | 35.69 | 48.11 | `[[174, 45], [182, 63]]` |
| fold3 | `gated_prior_kl` | cross_attention_auc=0.00, gated_prior_kl=1.00, audio_only=0.00, spatial_only=0.00 | 59.02 | 58.33 | 62.33 | 59.66 | 61.28 | 59.63 | `[[129, 90], [97, 148]]` |
| fold3 | `audio_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=1.00, spatial_only=0.00 | 69.77 | 68.27 | 56.66 | 53.91 | 46.04 | 52.22 | `[[153, 66], [152, 93]]` |
| fold3 | `spatial_only` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.00, spatial_only=1.00 | 58.41 | 60.26 | 55.80 | 54.06 | 30.63 | 47.06 | `[[193, 26], [196, 49]]` |
| fold3 | `ensemble_raw_balanced_accuracy` | cross_attention_auc=0.00, gated_prior_kl=0.06, audio_only=0.92, spatial_only=0.02 | 69.77 | 70.19 | 58.73 | 54.59 | 51.26 | 53.94 | `[[139, 80], [133, 112]]` |
| fold3 | `ensemble_raw_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.90, spatial_only=0.10 | 70.49 | 68.59 | 57.48 | 52.75 | 40.00 | 49.66 | `[[164, 55], [170, 75]]` |
| fold3 | `ensemble_logit_balanced_accuracy` | cross_attention_auc=0.00, gated_prior_kl=0.06, audio_only=0.92, spatial_only=0.02 | 69.77 | 70.19 | 58.73 | 54.59 | 51.26 | 53.94 | `[[139, 80], [133, 112]]` |
| fold3 | `ensemble_logit_auc_roc` | cross_attention_auc=0.00, gated_prior_kl=0.00, audio_only=0.90, spatial_only=0.10 | 70.54 | 68.59 | 57.49 | 52.75 | 40.00 | 49.66 | `[[164, 55], [170, 75]]` |

## Interpretation

- This report only compares models that have both validation and test predictions for the selected folds.
- The main comparison target is `gated_prior_kl`, because it is the current primary model.
- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.
